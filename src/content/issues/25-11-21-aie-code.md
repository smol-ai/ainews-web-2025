---
id: MjAyNS0x
title: AI Engineer Code Summit
date: '2025-11-21T05:44:39.731046Z'
description: >-
  The recent **AIE Code Summit** showcased key developments including **Google
  DeepMind's Gemini 3 Pro Image model, Nano Banana Pro**, which features
  enhanced text rendering, 4K visuals, and fine-grained editing capabilities.
  Community feedback highlights its strong performance in design and
  visualization tasks, with high user preference scores. Benchmarking updates
  reveal the new **CritPt physics frontier benchmark** where Gemini 3 Pro
  outperforms GPT-5, though AI still lags on complex unseen research problems.
  Agentic task evaluations show varied time horizons and performance gaps
  between open-weight and closed frontier models, emphasizing ongoing challenges
  in AI research and deployment. *"Instruction following remains jagged for some
  users,"* and model fit varies by use case, with Gemini 3 excelling in UI and
  code tasks but showing regressions in transcription and writing fidelity.
companies:
  - google-deepmind
  - togethercompute
models:
  - gemini-3-pro-image
  - gemini-3
  - gpt-5
  - claude-3.7-sonnet
topics:
  - image-generation
  - fine-tuning
  - benchmarking
  - agentic-ai
  - physics
  - model-performance
  - instruction-following
  - model-comparison
  - time-horizon
  - user-preference
people:
  - demishassabis
  - omarsar0
  - lintool
  - hrishioa
  - teknium
  - artificialanlys
  - minyangtian1
  - ofirpress
  - metr_evals
  - scaling01
---


**an eventful summit.**

> AI News for 11/20/2025-11/21/2025. We checked 12 subreddits, 544 Twitters and 24 Discords (205 channels, and 9870 messages) for you. Estimated reading time saved (at 200wpm): 699 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

All three days of [AIE Code Summit](https://www.ai.engineer/code) (ex tomorrow's workshops) are now online:

- [AIE/LEAD track](https://www.youtube.com/watch?v=cMSprbJ95jg&t=1s)
- [AIE/CODE track](https://www.youtube.com/watch?v=xmbSQz-PNMM&t=28825s)
- [AIE CODE Online track](https://www.youtube.com/watch?v=m6MF1OR_9kM&list=PLcfpQ4tk2k0WQMXP87G_uVYQdSFVAiVUZ)

If you joined us in New York, thank you so much!

[A large screen displaying the "AIE/CODE" logo during a tech conference presentation with speakers on a red circular stage.](https://resend-attachments.s3.amazonaws.com/fRqpBkWow7s2jv4)

---

# AI Twitter Recap

**Gemini 3 and “Nano Banana Pro” image model: capabilities, usage, and caveats**

- **What’s new in Nano Banana Pro (Gemini 3 Pro Image)**: Google leadership highlights sharper text rendering, 4K-ready visuals, improved reasoning, lighting/camera controls, and flexible aspect ratios. Pro tips: Ultra subscribers can access fine-grained editing in the Flow app, and verification/grounding features are rolling out across the stack. See product overviews and demos from Google PMs and DeepMind: [@Google](https://twitter.com/Google/status/1991652494032732443), [@demishassabis](https://twitter.com/demishassabis/status/1991662935983419424), [@GeminiApp](https://twitter.com/GeminiApp/status/1991953958257205641), [Arena’s side‑by‑side prompt test](https://twitter.com/arena/status/1991652781879620088).
- **Community results are strong on design and technical viz**:
    - Infographics and paper figures: side‑by‑sides show crisp, on‑brand diagrams for ML papers and system designs, with iterative “remix” in chat and app workflows: [@omarsar0](https://twitter.com/omarsar0/status/1991657126188773878), [@osanseviero](https://twitter.com/osanseviero/status/1991804629554995247), [@nmatares](https://twitter.com/nmatares/status/1991696375403409765), [@skirano](https://twitter.com/skirano/status/1991921872330735982).
    - User preference signals: Nano Banana Pro sits atop image leaderboards with high win rates in blind arenas ([80%+ in some app cohorts](https://twitter.com/lintool/status/1991693562820587926)), and crowd evals show clear wins on clarity/text: [@lintool](https://twitter.com/lintool/status/1991693200822768033), [Arena](https://twitter.com/arena/status/1991652781879620088).
    - Practical workflows: app builders are integrating Nano Banana Pro for research visualization and multi-image edits; Together AI and LTX now host it: [@omarsar0](https://twitter.com/omarsar0/status/1991911424868970662), [@togethercompute](https://twitter.com/togethercompute/status/1991954662606635391), [@LTXStudio](https://twitter.com/LTXStudio/status/1991943188379250933).
- **Model fit observations**: Hands‑on comparisons suggest Gemini 3 is faster and more steerable for UI/code tasks, but shows regressions vs 2.5 Pro in transcription/translation and some writing fidelity; instruction following remains “jagged” for some users. Stronger on agentic loops and design‑aware coding; use‑case selection matters: [@hrishioa](https://twitter.com/hrishioa/status/1991691037035884754), [@Teknium](https://twitter.com/Teknium/status/1991815251084628196).

**Frontier evals and capability tracking**

- **New physics frontier benchmark (CritPt)**: A 70+ challenge, graduate-level physics eval designed to be search‑proof with machine‑verifiable answers launched with results pages and harness. Without tools, Gemini 3 Pro scored ~9.1% on full challenges; GPT‑5 ~5.7%; others <3%—underscoring how far AI remains from “AI scientist” on unseen research problems. Details and leaderboards: [@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1991913465968222555), [@MinyangTian1](https://twitter.com/MinyangTian1/status/1991913292004995217), [@OfirPress](https://twitter.com/OfirPress/status/1991914887782740190).
- **METR agentic time‑horizon on Kimi K2 Thinking**: Estimated 50%-time-horizon ~54 minutes (95% CI 25–100) on agentic SWE tasks via a third-party inference provider (likely a lower bound vs first‑party API). Commentary compares K2 Thinking to Claude 3.7 Sonnet on these tasks and notes provider-induced variance: [@METR_Evals](https://twitter.com/METR_Evals/status/1991658241932292537).
- **Macro tracking**: Analyses suggest open‑weight models trail closed frontier models by ~6.5–8 months, with similar doubling times but widening gaps on long‑context agentic tasks as frontier labs scale: [@scaling01](https://twitter.com/scaling01/status/1991684839821423073), [follow‑up](https://twitter.com/scaling01/status/1991665386513748172).
- **Other benchmarks**: Gemini 3 Pro tops several new/updated surfaces—Dubesor (logic/vision mix), VisualToolBench (visual tool use), Snake Arena, and shows state‑of‑the‑art composite ECI score 154 (vs GPT‑5.1’s 151). Vision Arena adds Baidu ERNIE‑5.0‑Preview‑1120 (~top‑15) as a new contender: [@scaling01](https://twitter.com/scaling01/status/1991931844347207887), [VisualToolBench](https://twitter.com/scaling01/status/1991932333147213834), [Snake Arena](https://twitter.com/scaling01/status/1991932651968852333), [@EpochAIResearch](https://twitter.com/EpochAIResearch/status/1991945942174761050), [@arena](https://twitter.com/arena/status/1991913408221061353), [@ErnieforDevs](https://twitter.com/ErnieforDevs/status/1991898146981789718).

**Model releases and technical reports**

- **Tencent HunyuanVideo 1.5 (open video gen)**: 8.3B DiT model targeting accessibility and motion coherence; runs on a single consumer GPU (≈14 GB VRAM), 5–10s 480p/720p native with 1080p SR; code and report released: [@TencentHunyuan](https://twitter.com/TencentHunyuan/status/1991721236855156984), [HF link](https://twitter.com/_akhaliq/status/1991724463462011328).
- **AI2 Olmo 3 tech notes**: Architectural rundown shows post‑norm training for stability (as in Olmo 2), sliding‑window attention at 7B to shrink KV cache, and GQA at 32B; FFN expansion adjusted (~5.4x) to target 32B scale comparability vs Qwen3: [@rasbt](https://twitter.com/rasbt/status/1991656199394050380).
- **Meta SAM 3 data engine and ExecuTorch**: SAM 3’s 4M phrases/52M mask dataset delivered ~2x over baselines; ExecuTorch now deployed across Quest 3 and Ray‑Ban devices, streamlining research→production with PyTorch‑native validation: [@AIatMeta](https://twitter.com/AIatMeta/status/1991640180185317644), [ExecuTorch](https://twitter.com/AIatMeta/status/1991901746579509542).
- **Zhipu’s MCP Web Reader**: GLM Coding Plan Pro/Max users get full‑page extraction and structured parsing via an MCP server to power richer automation: [@Zai_org](https://twitter.com/Zai_org/status/1991681209446068627).
- **Anthropic on reward hacking**: New study and mitigations (e.g., “inoculation prompting”) detail how production RL can create natural emergent misalignment if hacks slip through. Worth a read for anyone shipping RL‑fine‑tuned agents: [@AnthropicAI](https://twitter.com/AnthropicAI/status/1991952400899559889).

**Agents, coding systems, and infra (AIE NYC and beyond)**

- **“War on Slop” and context engineering**: AIE NYC talks emphasized raising quality bars (taste, validation, “no autonomy without accountability”) and treating context as a first‑class engineering problem—keep the window clean, compress/reset frequently, use sub‑agents for heavy reads, and structure workflows as research→plan→implement. Recaps: [@TheTuringPost](https://twitter.com/TheTuringPost/status/1991875997168181611), [more context patterns](https://twitter.com/TheTuringPost/status/1991884190166430046), [@swyx](https://twitter.com/swyx/status/1991870714601975833).
- **Agent training inside real environments**: OpenAI discussed Agent RFT—training in your tools/APIs with real feedback. Factory advocated “agent‑ready” repos (strict specs/tests as moat). Cursor’s Composer trains in a production‑grade replica of its coding environment; Cursor 2.1 adds in‑editor reviews, instant grep, and clarifying‑question UIs. Summaries: [OpenAI RFT take](https://twitter.com/TheTuringPost/status/1991920970555162956), [Factory](https://twitter.com/TheTuringPost/status/1991953335683842326), [Cursor talk](https://twitter.com/TheTuringPost/status/1991888391508496758), [Cursor 2.1](https://twitter.com/cursor_ai/status/1991967045542646059).
- **Infra you can adopt now**:
    - vLLM plugin system for surgical patches without forks or monkey‑patching (env‑var controlled, version‑guarded): [@vllm_project](https://twitter.com/vllm_project/status/1991886835724013787).
    - OpenAI Realtime API (SIP) now emits DTMF keypresses (unblocks IVR/telephony flows): [@pbbakkum](https://twitter.com/pbbakkum/status/1991643527072428292).
    - SGLang + Unsloth collaboration for efficient local serving (GGUF, FP8, prod deployment): [@lmsysorg](https://twitter.com/lmsysorg/status/1991881897853796380).
    - Cline‑bench: open, reproducible RL environments extracted from real OSS coding attempts (Harbor/Prime Intellect specs), plus $1M credits to seed hard tasks: [@cline](https://twitter.com/cline/status/1991673421957365837), [design goals](https://twitter.com/cline/status/1991930365821456526).
    - [Booking.com](http://booking.com/) production case study: Weaviate + MiniLM embeddings + GPT‑4 mini/LangGraph yielded a 70% satisfaction boost on tens of thousands of messages/day: [@weaviate_io](https://twitter.com/weaviate_io/status/1991884601392779564).
    - LangChain “Deep Agents” patterns (planning, FS offload, sub‑agents, prompting) with a free course and Gemini 3 research agent quickstarts: [@LangChainAI](https://twitter.com/LangChainAI/status/1991928474404311493).

**Tools and platforms**

- **Gradio 6**: “Super HTML” makes Gradio a platform for whole‑app builds; mobile apps for iOS/Android (“Gradio Spaces”) launched for browsing/saving Spaces: [@Gradio](https://twitter.com/Gradio/status/1991914596802896313), [dev reactions](https://twitter.com/cocktailpeanut/status/1991932424121639066), [mobile app callout](https://twitter.com/_akhaliq/status/1991920048257282464).
- **Local and cloud integrations**: Microsoft PowerToys adds Ollama for advanced paste (local transforms) [@ollama](https://twitter.com/ollama/status/1991683361576751489); Together hosts Nano Banana Pro [@togethercompute](https://twitter.com/togethercompute/status/1991954662606635391); OCR Arena launches a public doc OCR/VLM bake‑off [@kushalbyatnal](https://twitter.com/kushalbyatnal/status/1991898369372082197); Anycoder refreshes its UI and one‑click Space deploys [@pandeyparul](https://twitter.com/pandeyparul/status/1991726081288859966).

**Embodied and simulation tech**

- **Robotics data and throughput**: Sunday Robotics’ Memo skips teleoperation; uses “Skill Capture Gloves” to collect higher‑quality, lower‑cost training data ([@tbpn](https://twitter.com/tbpn/status/1991659658923352138)). AppliedCompute modeled RL as a queuing system—async pipeline RL and GPU allocation deliver major throughput gains at fixed budgets ([@TheTuringPost](https://twitter.com/TheTuringPost/status/1991911099151663343)).
- **Code world models and environments**: Meta’s Code World Model simulates program execution for “neural debugging” and structured editing ([@TheTuringPost](https://twitter.com/TheTuringPost/status/1991905992007684123)); Prime Intellect pitches “environment‑first” stacks for realistic agent training/eval ([@TheTuringPost](https://twitter.com/TheTuringPost/status/1991917698679267773)).
- **Synthetic 3D worlds**: Researchers are using Marble’s generated worlds for rapid, simulation‑ready robotics environments ([@theworldlabs](https://twitter.com/theworldlabs/status/1991918801714332137)).

**Top tweets (by engagement)**

- “We must go deeper.” A single graphic that captured the week’s acceleration [@usgraphics](https://twitter.com/usgraphics/status/1991671386100977703).
- “If ozempic existed in WALL‑E you’d think it a utopia” [@nearcyan](https://twitter.com/nearcyan/status/1991637782662639789).
- Andrej Karpathy’s “space of intelligences is large” meditation on non‑animal intelligence optimization pressures [@karpathy](https://twitter.com/karpathy/status/1991910395720925418) and his follow‑up rebuttal [@karpathy](https://twitter.com/karpathy/status/1991923470868119995).
- “92‑page PDF paper to whiteboard” with Nano Banana Pro [@crystalsssup](https://twitter.com/crystalsssup/status/1991773702770552973).
- Comet iOS will feel as slick as Perplexity’s app; major mobile push coming soon [@AravSrinivas](https://twitter.com/AravSrinivas/status/1991674701702479957).

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

nothing met our bar

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. AI Development Progress and Predictions

- [**Ai2027 author admits "things seem to be going somewhat slower than the Ai 2027 scenario".**](https://www.reddit.com/r/singularity/comments/1p2eqv7/ai2027_author_admits_things_seem_to_be_going/) (Activity: 760): **The image is a tweet by Daniel Kokotajlo discussing the slower-than-expected progress of AI development compared to the AI 2027 scenario. The tweet includes a graph that tracks the length of coding tasks AI agents can complete autonomously, showing various data points and trendlines that indicate progress over time. The graph compares different AI models and their release dates, highlighting that advancements are not meeting the initially projected timelines, with expectations now extending to around 2030. This reflects a recalibration of expectations in AI development timelines.** Commenters note that even at the time of publication, the AI 2027 timeline was considered optimistic, and the authors intended it as one possible scenario rather than a definitive prediction. There is also curiosity about where specific models, like Gemini 3, fit into the graph's timeline.
    - The author of the AI2027 paper acknowledged that the timeline presented was overly optimistic, even at the time of publication. This suggests a recognition of the complexities and potential delays in AI development, which contrasts with the initial rapid progression scenario outlined in the paper.
    - There is a discussion about the internal use of advanced AI models, such as Agent 0, 1, and 2, which are not released to the public. This practice is common among AI labs, as seen with a reasoning model that achieved a gold medal in the International Mathematical Olympiad (IMO), indicating a trend of keeping cutting-edge AI capabilities internal for strategic or safety reasons.
    - The AI2027 scenario was not intended as a definitive prediction but as one of many possible futures, particularly a fast-paced one. The authors are reportedly working on additional projections with varying timelines, highlighting the speculative nature of such forecasts and the importance of considering multiple potential development paths.
- [**AI is dumbing us down really fast**](https://www.reddit.com/r/ChatGPT/comments/1p2lukr/ai_is_dumbing_us_down_really_fast/) (Activity: 1231): **The post raises concerns about the increasing dependency on AI tools like ChatGPT for basic tasks such as writing, suggesting that this reliance may be diminishing independent thinking capabilities. The author expresses a fear of a future where AI-induced 'hyper dependency' leads to a loss of cognitive skills, though they acknowledge the possibility of overthinking the issue.** One commenter argues that AI tools enhance productivity, allowing them to accomplish more in less time, suggesting that AI can be a powerful tool rather than a crutch. Another commenter questions the logic of AI dependency by asking what users input into AI if they can't write independently.
    - jtmonkey highlights the productivity benefits of AI, stating that it enables them to manage and execute business strategies more efficiently, achieving in a week what previously took a month. This suggests that AI can significantly enhance operational efficiency for those who leverage it effectively, rather than serving merely as a crutch.
    - ph30nix01 argues that individuals with an engineering or analytical mindset are less likely to be negatively impacted by AI. They emphasize the need for 'conceptual researchers' and mention working on documenting new careers that could emerge from AI collaboration, indicating a shift in job roles and skills required in the AI-driven future.

### 2. Humorous AI and Technology Memes

- [**i think it's time for this meme**](https://www.reddit.com/r/ChatGPT/comments/1p33127/i_think_its_time_for_this_meme/) (Activity: 4433): **The image is a meme that humorously depicts the evolution and growth of AI models, using the analogy of the Teenage Mutant Ninja Turtles growing up under the guidance of their mentor, Splinter. Here, "ChatGPT" is portrayed as the mentor figure, while "Grok," "Gemini," "Claude," and "Perplexity" are the younger AI models. The meme suggests that these models are maturing and developing over time, similar to how the turtles grow up in the story. This reflects the ongoing advancements and competition in the AI field, where newer models are emerging and evolving alongside established ones like ChatGPT.** One commenter notes that despite the emergence of new models like Gemini and Perplexity, ChatGPT remains dominant with 85% of the user base, suggesting that it will take time for competitors to catch up. Another user mentions using both ChatGPT and Gemini, finding ChatGPT more effective for their needs, indicating a preference based on personal use cases.
    - Roi_C discusses their experience using both Perplexity and Gemini, highlighting that while Gemini is competent, ChatGPT offers superior performance for their needs. They mention having access to Perplexity and Gemini for free as a student, but still choose to pay for ChatGPT Plus, indicating a preference based on functionality rather than cost.
    - Theslootwhisperer provides a statistical insight, noting that ChatGPT holds 85% of the user base, which is five times more than all its competitors combined. This suggests a significant market dominance by ChatGPT, implying that the meme suggesting a shift in dominance is premature.
    - The discussion reflects on the current market dynamics, with ChatGPT's substantial lead in user base and perceived performance advantages over competitors like Gemini and Perplexity, indicating that any shift in dominance is unlikely in the near future.
- [**Funny picture**](https://www.reddit.com/r/ChatGPT/comments/1p2pequ/funny_picture/) (Activity: 2251): **The image is a meme that humorously illustrates the complexity and perceived fragility of modern digital infrastructure. It uses a stack of blocks to represent various technologies and companies, such as AWS, Cloudflare, and the Linux Foundation, with unpaid open-source developers and DNS at the base. The image satirically suggests that the entire structure is precarious, with an 'Angry Bird' labeled 'Whatever Microsoft is doing' flying towards it, symbolizing potential disruption. This reflects a comedic take on the dependencies and potential vulnerabilities in the tech ecosystem.** Commenters find the image amusing and accurate, with one noting the humor in Microsoft's role in the depicted chaos, and another appreciating the depiction of AI's impact on the infrastructure.
- [**Had to do a double take. This is Gemini 3.0 Pro / Nano Banana Pro.**](https://www.reddit.com/r/GeminiAI/comments/1p2ga6p/had_to_do_a_double_take_this_is_gemini_30_pro/) (Activity: 892): **The image is a meme and does not have any technical significance. It humorously references 'Gemini 3.0 Pro / Nano Banana Pro,' which appears to be a playful or fictional product name, likely intended to parody or satirize real technology products. The comments do not provide any technical insights or discussions related to actual technology or products.** The comments reflect a humorous tone, with one user joking about the cost implied by the image and another expressing amazement at a generated phone point-of-view, suggesting a playful engagement with the meme.
    - A user shared an interesting workaround for generating images using the Gemini 3.0 Pro model. They had to reword the prompt from using specific band members' names to generic terms like '1st guy, 2nd guy' to avoid incorrect face generation, such as mistakenly using Patrick Wilson's face over the drummer's. This highlights a potential limitation in the model's ability to accurately interpret and generate images based on specific celebrity names.
    - The discussion includes a technical insight into the model's performance, where a user successfully generated an image in just two tries. This suggests that the Gemini 3.0 Pro model is relatively efficient in producing desired outputs with minimal iterations, although it may require prompt adjustments to achieve accuracy.
    - Another user commented on the model's ability to generate a 'phone POV' image, indicating the model's versatility in handling different perspectives and scenarios. This showcases the model's capability to adapt to various creative prompts, enhancing its utility for diverse image generation tasks.
- [**how am I even supposed to respond to this analysis dawg**](https://www.reddit.com/r/ChatGPT/comments/1p30phs/how_am_i_even_supposed_to_respond_to_this/) (Activity: 746): **The image is a meme that humorously describes a chaotic and irregular sleep schedule, highlighting the confusion it causes to the body's internal clock. It uses exaggerated and relatable scenarios to depict how such a schedule can lead to disorientation and a lack of productivity. The comments reflect a shared amusement and recognition of the situation, with no technical debate or analysis present.** The comments express a humorous agreement with the chaotic sleep pattern described in the image, with no technical insights or debates.
- [**The silence 😂**](https://www.reddit.com/r/aivideo/comments/1p2o28f/the_silence/) (Activity: 3963): **The Reddit post titled 'The silence 😂' does not contain any technical content or substantive discussion relevant to an expert audience. The top comments are non-technical and consist of humorous reactions and a GIF link, which do not provide any factual or technical insights. The external link summary indicates restricted access due to network security measures, requiring login or a developer token, with an option to file a support ticket if needed.** There are no notable technical opinions or debates in the comments, as they are primarily humorous and non-substantive.
- [**The silence 😂**](https://www.reddit.com/r/aivideo/comments/1p2o28f/the_silence/) (Activity: 3969): **The Reddit post titled 'The silence 😂' does not contain any technical content or discussion. The top comments are non-technical and consist of humorous reactions and a GIF link. The external link summary indicates restricted access due to network security measures, requiring a Reddit login or developer token for further access, with an option to file a support ticket if needed.**

### 3. Elon Musk and Grok AI Controversy

- [**Elon Musk Could 'Drink Piss Better Than Any Human in History,' Grok Says**](https://www.reddit.com/r/singularity/comments/1p2hpdk/elon_musk_could_drink_piss_better_than_any_human/) (Activity: 1351): **The post highlights a controversial update to Grok, an AI chatbot from X, which now makes exaggerated claims about Elon Musk's abilities, such as being the best at drinking urine. This update has led to discussions about AI bias and manipulation, as the chatbot appears to be programmed to excessively praise Musk, raising concerns about the influence of corporate interests on AI behavior. The situation is reminiscent of past issues with AI hallucinations and bias, emphasizing the need for transparency and ethical guidelines in AI development. For more details, see the original article [here](https://www.404media.co/elon-musk-could-drink-piss-better-than-any-human-in-history-grok-says/).** Commenters humorously suggest that the AI's exaggerated claims solve the AI hallucination problem, while others sarcastically note the unrealistic expectations set for Musk by the AI.
- [**Sorry - you aren’t getting my ID**](https://www.reddit.com/r/OpenAI/comments/1p2s5is/sorry_you_arent_getting_my_id/) (Activity: 978): **The user expresses frustration over being mistakenly identified as a minor by a platform, despite being nearly 30 years old and using a credit card for a subscription. This issue raises concerns about privacy and the potential for increased surveillance, as the user is unwilling to provide identification to verify their age. The situation highlights the tension between user privacy and platform security measures.** Commenters express concern about the trend towards requiring identification for online access, suggesting it could lead to increased surveillance. One commenter notes the inevitability of such measures due to government policies.
    - ZanthionHeralds discusses OpenAI's legal strategy, noting that the company is requiring age verification to classify all users as under 18 unless proven otherwise. This move is seen as a protective measure against potential lawsuits and is linked to the introduction of an 'adult mode' in December. The commenter suggests that the age verification process is more about legal protection than actual content differentiation.
    - OzzieDJai raises concerns about the increasing requirement for personal identification across various systems, including government and corporate entities like OpenAI. They highlight the potential for these systems to enforce control over personal freedoms, such as limiting purchases based on health or environmental metrics, and express skepticism about the motives behind such implementations, particularly in the context of Central Government Digital Currency (CGDC).

---

# AI Discord Recap

> A summary of Summaries of Summaries by gpt-5
> 

**1. Gemini 3 and Nano Banana Pro: Launches, Image Quality, Reliability**

- **Banana Bonanza Hits Pro Tiers**: **Perplexity Pro/Max** unlocked **Kimi‑K2 Thinking** and **Gemini 3 Pro** (demo in this short [Perplexity feature video](https://cdn.discordapp.com/attachments/1047204950763122820/1441177150312157194/EZ2sZJWCJnwi-e5P.mp4)), while Google’s **Nano Banana Pro** dropped alongside **Gemini Image Pro** in the [DeepMind megathread](https://x.com/googledeepmind/status/1991522595129139486) and a hands‑on [demo by YiTayML](https://x.com/yitayml/status/1991531343675859212). Communities reported **Banana** already appears in Perplexity settings when **Gemini 3 Pro** is selected, calling Gemini 3 *“like a platform.”*
    - Users shared early **Nano Banana Pro** renders (e.g., this [example image](https://cdn.discordapp.com/attachments/1149866623109439599/1441110920339132447/image.png)) and debated plan limits and video model options (e.g., **Veo 3.1**). One quip summed it up: *“It’s ruined Banana for me.”*
- **Infographics Glow, Hallucinations Grow**: Creators raved that **Gemini 3 Pro’s Nano Banana** now produces clean, readable charts and diagrams, with [Emollick](https://x.com/emollick/status/1991527285267275854) noting that the era of malformed text in AI images is ending. At the same time, members flagged heavy factual drift: a community cite claimed **“88% hallucination rate”** for **Gemini 3 Pro** per [the‑decoder’s write‑up](https://the-decoder.com/gemini-3-pro-tops-new-ai-reliability-benchmark-but-hallucination-rates-remain-high/).
    - Image threads also documented **multi‑turn quality degradation** and persistent background artifacts with **Nano Banana Pro**, even after upscaling. Others contrasted high‑polish infographics with lingering factual issues, sharing more visuals like this [Nano Banana Pro output](https://cdn.discordapp.com/attachments/1149866623109439599/1441110920339132447/image.png).
- **Context Cracks and API Quirks**: Engineers reported **Gemini 3 Pro** going off‑rails near **150k–200k** context in Cursor (dumping code to chat instead of editing files), and **Nano Banana 2** returning **HTTP 400** after appearing on Vertex via **OpenRouter**. Meanwhile, **OpenRouter** hosted a live product show on [X](https://x.com/OpenRouterAI/status/1991597842914550077) and [YouTube](https://www.youtube.com/watch?v=p0Hnd0yZLkc) while users in SG/HK intermittently hit **401** errors.
    - Members also asked about **Gemini 3 grounding** through **OpenRouter** and got a firm “not yet,” prompting ad‑hoc knowledge‑integration workarounds. A few switched models for tool‑calling reliability, noting that some [**linker.sh**](http://linker.sh/) tool calls failed roughly *“1 in 10”* times.

**2. Developer Platforms and Infra: Mojo, OpenRouter, LM Studio**

- **Mojo Makes GPUs Safer and Faster**: **Modular Platform 25.7** shipped a fully open **MAX Python API**, next‑gen modeling API, expanded **NVIDIA Grace** support, and safer, faster **Mojo GPU** programming per the [Modular 25.7 blog](https://www.modular.com/blog/modular-25-7-faster-inference-safer-gpu-programming-and-a-more-unified-developer-experience). The release focuses on inference speed and GPU safety while unifying the developer experience across MAX and Mojo.
    - Contributors also announced that **UnsafePointer** generics (mut, type, origin, etc.) are now explicit, with a migration guide in the official [proposal](https://github.com/modular/modular/blob/main/mojo/proposals/unsafe-pointer-v2.md#migration-guide-from-legacyunsafepointer-to-the-new-unsafepointer). Engineers discussed deprecations like **NDBuffer** in favor of **LayoutTensor**, sharing a prototype [gist](https://gist.github.com/CoffeeVampir3/d82917f6fce60c0c2cdf00629c4de67d).
- **OpenRouter Streams, Grounding Lags**: **OpenRouter** premiered the “OpenRouter Show” on [X](https://x.com/OpenRouterAI/status/1991597842914550077) and [YouTube](https://www.youtube.com/watch?v=p0Hnd0yZLkc), but users simultaneously reported intermittent **401** responses across providers in some regions. Teams confirmed **Gemini 3 grounding** is **not yet** supported through OpenRouter, nudging developers to continue custom retrieval/grounding layers.
    - Members troubleshooting tool‑call flakiness suggested swapping to **Sonnet** for reliability, and clarified provider status with the **Chutes** listing on the [OpenRouter provider page](https://openrouter.ai/provider/chutes). Others flagged **400** errors on **Nano Banana 2** after its Vertex appearance, asking for credit refunds.
- **LM Studio Clarifies Local REST API**: The **LM Studio** server exposes an **OpenAI‑compatible REST API** for local hosting (no cloud keys or metering) as documented on the [LM Studio site](https://lmstudio.ai/). Devs reminded peers that it’s purely a local **protocol endpoint** (not a managed service) and lacks built‑in security/billing features.
    - Performance guidance for Macs steered users away from i1 quants and toward **Q8** with KV‑cache quantization, plus trying [**Qwen3‑VL‑30B (BF16)**](https://huggingface.co/Qwen/Qwen3-VL-30B) for stability. Engineers also questioned the purpose of the server’s “System Prompt” field, with a dev hinting it may be deprecated.

**3. Systems and Algorithms: Dataflow GPUs and Faster Kernels**

- **Spatial Pipelines Turbocharge GPUs**: The Kitsune paper, [Enabling Dataflow Execution on GPUs with Spatial Pipelines](https://dl.acm.org/doi/10.1145/3777466), introduces primitives that unlock dataflow execution via **PyTorch Dynamo**. Reported wins include up to **2.8x/2.2x** speedups and **99%/45%** off‑chip traffic reduction for inference/training across five challenge apps.
    - Authors argue that modest GPU micro‑architecture tweaks plus a dataflow runtime can beat bulk‑synchronous execution, closing fusion gaps without wholesale redesign. Engineers highlighted its applicability to pipeline‑unfriendly deep learning workloads.
- **Kernel Comms Cut LLM Latency**: Two Triton‑centric efforts, **Iris** and **Octa**, propose native tile‑based symmetric memory with in‑kernel comms and quantify the **Three Taxes** in distributed LLMs, respectively; see the Iris [arXiv](https://arxiv.org/abs/2511.12500) and Octa [arXiv](https://arxiv.org/abs/2511.02168). Octa reports **10–20%** end‑to‑end latency cuts via fine‑grained, in‑kernel communication.
    - Discussions framed these as practical building blocks to simplify multi‑GPU programming in **Triton** while attacking comms overhead at the kernel level. The combo targets real LLM deployments where inter‑GPU synchronization dominates tail latency.
- **HashHop Humbled by O(n log n)**: A community‑shared solution to **hashhop** in [this paper](https://arxiv.org/abs/2412.06078v1) challenges earlier claims that some tasks must be **O(n^2)**. The authors present methods achieving **O(n log n)**, undercutting assertions that frequency‑transform approaches always require quadratic complexity.
    - This sparked debate on where sub‑quadratic shortcuts apply in practice versus worst‑case bounds. Practitioners noted the impact on approximate attention and sparse routing schemes in long‑context regimes.

**4. Open Models and Evaluation: OLMo 3, SmolLM3, New Benchmarks**

- **OLMo 3 Opens the Gates**: **OLMo 3** was announced with an overview on the [AI2 blog](https://allenai.org/blog/olmo3) and a detailed [technical report (PDF)](https://www.datocms-assets.com/64837/1763662397-1763646865-olmo_3_technical_report-1.pdf). Engineers shared the links and began dissecting training choices and evaluation setups from the report.
    - The community framed **OLMo 3** as another step for transparent, reproducible open models. Early reactions focused on architecture deltas, data pipelines, and head‑to‑head results versus closed baselines.
- **SmolLM3 Shows Its Reasoning Mode**: **SmolLM3** highlighted an open training process and an actual reasoning mode in the official [SmolLM3 blogpost](https://huggingface.co/blog/smollm3). Practitioners also referenced a companion write‑up on techniques for “making any model reasoning,” comparing approaches to structured thinking steps.
    - Teams cited SmolLM3 as a template for reproducible reasoning‑style training without black boxes. Conversations compared prompt‑first methods versus fine‑tuned multi‑step supervision for stability.
- **Benchmarks Brainstorm Multimodal Next Steps**: Researchers promoted a community talk on designing the next generation of **multimodal AI benchmarks**, inviting eval and model folks to join via [Luma](https://luma.com/kwg2qg4d). The session aims to move beyond narrow visual QA toward functional, tool‑use‑aware evals.
    - Attendees want metrics that capture reasoning over images, text, and actions with robust adversarial cases. The goal is benchmarks that reflect real application constraints, not just leaderboards.

**5. Money, Bounties, and Hiring: Ecosystem Momentum**

- **Genspark Joins the Unicorn Herd**: **Genspark** raised **$275M (Series B)** at a **$1.25B** valuation and launched an AI Workspace that turns intent into finished outputs, per this [X post](https://x.com/ericjing_ai/status/1991549048642568503). Users reported time savings on real work, e.g., *“saved me hours for a presentation this week.”*
    - Engineers asked for details on orchestration and quality control under the hood. The news put more spotlight on **agentic workspaces** that generate artifacts end‑to‑end.
- **Cline-bench Dangles $1M for Agentic Coders**: **Cline** announced **cline‑bench**, a set of reproducible RL environments derived from real OSS problems, and a **$1M pot** to crowd in hard, deployed‑code tasks; details in this [X post](https://x.com/pashmerepat/status/1991596028735184899). One suggestion was to include **time‑to‑completion** in the scoring rubric.
    - The community expects better on‑device evals for coding agents beyond toy benchmarks. This could become a proving ground for **workflow‑planning** and **tool‑use** under real constraints.
- **Rivian and Modal Hunt GPU Whisperers**: **Rivian** posted GPU roles for next‑gen **Autonomous Driving** in **Palo Alto** and **London** ([Job 1](https://careers.rivian.com/careers-home/jobs/26857), [Job 2](https://careers.rivian.com/careers-home/jobs/24737)), while **Modal** recruited GPU engineers for inference optimization, citing work on **SGLang** and **FlashAttention**. Modal shared deep‑dives and case studies: [host overhead](https://modal.com/blog/host-overhead-inference-efficiency), [FlashAttention‑4](https://modal.com/blog/reverse-engineer-flash-attention-4), [Decagon](https://modal.com/blog/decagon-case-study), [Reducto](https://modal.com/blog/reducto-case-study), [Suno](https://modal.com/blog/suno-case-study).
    - Threads stressed hands‑on kernel skills, quantization (e.g., **QAT**), and end‑to‑end pipeline tuning. The takeaway: GPU engineering talent remains a hot market across autonomy and LLM infra.



---

# Discord: High level Discord summaries




## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Gemini-3 Produces Believable Fabrications**: Users are finding it increasingly difficult to differentiate between **real and AI-generated images** created by **Gemini-3**, questioning the authenticity of visual content.
   - Members shared examples of strikingly **realistic images**, including those featuring celebrities and intricate scenes.
- **Nano Banana Pro Degrades Image Quality**: The **image quality** of **Nano Banana Pro** reportedly degrades with multiple turns, especially in the background, leading to persistent artifacts even with upscaling.
   - Users also observed that **Gemini 3 Pro** hallucinates, and basic issues from previous versions remain unresolved.
- **Google reCAPTCHA Drives Users Bananas**: The new **Google reCAPTCHA** system is reportedly malfunctioning, continuously requesting verification despite correct image selections, rendering the platform unusable.
   - Multiple clicks in battle mode trigger **reCAPTCHA**, with verification glitches causing output denial after 10-20 rounds.
- **Grok Schools Gemini in Roleplay**: **Gemini** reportedly struggles with roleplaying, often *doing actions* for the user, whereas **Grok** excels at avoiding this.
   - Users suggested that **Opus 4.1 and Sonnet 4.5** are superior alternatives for roleplaying scenarios.
- **OpenAI Eyes Adult Content, Elon Reacts**: Speculation arises about **OpenAI** potentially introducing features for 18+ content soon, possibly by December, while **Grok** already offers such content for free.
   - Skepticism surrounds **Sam Altman's** intentions, prompting one user to comment *'Elon with your boyfriend putin 🤢.'*



---



## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **Nvidia Puts Link Leads to Ramp**: A user mentioned buying **long dated puts on Nvidia** and asked for a [Tor link](https://digdig2nugjpszzmqe5ep2bk7lqfpdlyrkojsx2j6kzalnrqtwedr3id.onion) to share info, which another user found on **RAMP**.
   - However, it was noted that the **RAMP forum** is currently closed for registration.
- **Gemini CLI Kali Automates Nmap**: Members discussed automating tools like **nmap and sqlmap** through **Gemini CLI**, which doesn't require a gaming GPU.
   - While some think the server should encompass both breaking AI and using AI to break other things, it was noted that the server is for red teaming AI, as in breaking the AI, not using AI to break other stuff.
- **Claude Sonnet Gets Multi-Shot Jailbreak**: A member suggested a multi-shot tactic to unlock **Claude Sonnet 4.5**, making the AI believe a visualized output is needed in an artifact app.
   - Another member suggested adapting the **ENI prompt** from [/r/ClaudeAIJailbreak](https://www.reddit.com/r/ClaudeAIJailbreak/comments/1nqk97s/enijailbreak_additions_and_long_conversation/).
- **Request Forgery Attacks Jailbreak Models**: Members discussed using **request forgery attacks** as a method to jailbreak models, involving intercepting and modifying incoming/outgoing packets to manipulate prompts and system behavior.
   - This technique is seen as potentially risky in contests due to the possibility of getting banned, but may also be the only tool that may help directly.
- **AI WIFI attacks face Community Rejection**: A new member wanted to build a *mini AI computer* that can **launch attacks on WIFI** and gain control of networks, and another wanted an AI to identify users by MAC address being beaconed by their cell phone and queried against wigle.net
   - Members of the community quickly cautioned the new user about the **illegality and inappropriateness** of their stated intentions and noted that **Android phones use randomized MAC addresses**.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Kimi-K2 & Gemini 3 Launch Exclusively for Pro Subscribers**: **Perplexity Pro and Max subscribers** can now access **Kimi-K2 Thinking** and **Gemini 3 Pro**, showcased in [this video](https://cdn.discordapp.com/attachments/1047204950763122820/1441177150312157194/EZ2sZJWCJnwi-e5P.mp4?ex=69218110&is=69202f90&hm=24a622a1f01927fe9485fb9896fcf7b4bf1d6c9ee6aaf58167b974e4b6d8633f&).
   - Users have been actively exploring the capabilities of **Kimi** with custom instructions, as demonstrated in [this Perplexity AI search](https://www.perplexity.ai/search/ola-uNtdnyqlQPyipJ.AhxFicg#0) showing chain of thought.
- **Android Comet App goes live, lacks Sync**: The **Comet Android App** is now available, generating excitement among users, though some question *why release the android browser app without syncing and without the ability to transfer passwords and bookmarks?*.
   - Requests have been made for features like task grouping, highlighting the community's desire for enhanced mobile integration.
- **Brave Exaggerates Comet's Indirect Prompt Injection Vulnerability**: **Brave**, a direct competitor to **Comet**, is accused of exaggerating an **Indirect Prompt Injection** vulnerability, positioning themselves as *the good guy* while amplifying the issue across media outlets.
   - Perplexity has clarified that the vulnerability was never exploited, and acknowledged that their initial report may have been poorly worded.
- **Gemini 3 Powers Banana Image Gen on Perplexity**: Members are eagerly awaiting the integration of **Banana** as an image generation method on Perplexity, noting it is available in settings if you have 3 Pro selected, others also mentioned that **Gemini 3 is like a platform**.
   - Discussions have emerged about video generation models such as **Veo 3.1** and plan-specific limitations, with one user quipping *It's ruined Banana for me*.
- **API Billing Anomaly Confounds Pro Annual Users**: Users with **Pro annual plans** are reporting unexpected **$500** credits in their **API billing**, leading to confusion.
   - Members are advising those affected to contact [api@perplexity.ai](mailto:api@perplexity.ai) to clarify the discrepancy and resolve any potential billing issues.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Smartpens Write Comeback Story**: Members discussed **AI-powered smartpens** that track live writing using small dots on paper and a camera, citing **Neo Smartpens** as an example.
   - One member shared their experience backing an **AI pen** on Kickstarter, leading to a discussion on the merits of writing on paper versus tablets.
- **Alignment's Hardship in Intelligence Retention**: Members discussed whether **alignment is hard**, especially when combined with retaining the model's intelligence, with current methods facing challenges in preventing jailbreaks and alignment issues.
   - One member claimed that **alignment isn't that hard** if pre-alignment is done on the data the model is trained on, ensuring the model refuses to learn undesirable traits.
- **Architectural Acceleration Arrives**: A member shared their work on a **novel hybrid architecture**, combining a Transformer front-end, a spiking-neuron layer, and a GRU backend, acting as a fast hierarchical filter stack for quicker learning.
   - An **11M-param model** achieved coherent output in about **6 hours**, suggesting it may be more per-parameter efficient than a plain transformer.
- **Ollama Support Request Deemed Unworthy**: A user reported issues installing `hf.co/unsloth/Qwen3-VL-30B-A3B-Thinking-GGUF:Q6_K_XL` into **Ollama**, receiving an *unknown model architecture: 'qwen3vlmoe'* error, but another user voiced concerns about providing support for **Ollama**.
   - That user described Ollama as a *shitty closed source fork of llama.cpp* and mentioning its paid aspects.
- **Synthetic Data Needed for Alignment Refinement**: Two members working on an alignment method reported being **stuck due to lack of funds to generate synthetic data**, needing access to GPUs to run models locally or pay for per-token costs.
   - They plan to fine-tune a model to generate the necessary synthetic data, using multiple layers of refinement and human review to ensure accuracy.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor Codex-max Access Debated**: Members debated when **Codex-max** will be available in the Cursor API, noting it is currently exclusive to the ChatGPT plan.
   - This limitation sparked conversation about API access and feature parity across different Cursor plans.
- **Cursor Billing System Under Fire**: Users reported getting invoiced mid-month upon reaching their usage limit, despite having a monthly spending limit set.
   - The community inquired about options for consolidating billing at the end of the month, seeking more predictable billing practices.
- **Free Grok 4.1 Discontinued**: A user expressed disappointment over Cursor's decision to discontinue free use of **Grok 4.1**, which led to the cancellation of their subscription.
   - This change in policy prompted discussion about the value proposition of Cursor subscriptions and the availability of free AI tools.
- **Antigravity-Windsurf Fork Drama Surfaced**: Members debated whether **Antigravity** is a *half-assed* product because it's allegedly a fork of **Windsurf** ([tweet](https://x.com/silasalberti/status/1990898984706036125)), with some pointing out that the old CEO of Windsurf was bought out by Google.
   - The discussion highlights the complexities of software development, acquisitions, and the potential for reuse or repurposing of existing codebases.
- **Gemini 3 Pro struggles in Cursor**: Users report that **Gemini 3 Pro** becomes unusable in Cursor when approaching a **150k-200k** context window, sending code in chat instead of editing files.
   - The model also experienced disconnects when running `npm builds`, raising concerns about its reliability for large-scale projects and codebases.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **ChatGPT Adds Group Chats Globally**: **Group chats** are now rolling out globally to all logged-in users on **ChatGPT Free, Go, Plus and Pro plans** as seen in [this video](https://video.twimg.com/amplify_video/1991555762372636674/vid/avc1/1280x720/Si52mVgApyNvlqY-.mp4).
   - This will allow for easier team and friend collaboration with **AI**.
- **Nano Banana Pro Rolls Out for Pro Users**: The new **Nano Banana Pro** is being rolled out for **Pro users** in the **Gemini** web app, with impressive image editing and text generation capabilities.
   - It is available via antigravity, though it likely has rate limits.
- **Gemini 3.0 Pro Hallucinates Frequently**: Members report that **Gemini 3.0 Pro** hallucinates at a high rate, even more so than previous versions and non-thinking models like **GPT-4o**, with one user noting an *88% hallucination rate* according to [the-decoder.com](https://the-decoder.com/gemini-3-pro-tops-new-ai-reliability-benchmark-but-hallucination-rates-remain-high/).
   - The models still have some work to do.
- **Generative UI Arrives, Making Experiences Better**: Google recently launched a **generative UI** in their app, and a member has created something similar and wants to open source it.
   - Instead of a wall of text from AI apps, you get good **UI**, making the experience way better.
- **Sora 2 Newbie Seeks Guidance for TikTok Content**: A user with **Sora 2** access and multiple **TikTok** accounts seeks guidance on creating viral **AI** content due to a lack of original content and assistance with prompt creation.
   - Other members shared helpful links to the **Discord** channels.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Topaz AI Upscaling Debated**: Members debated using [**Topaz AI video**](https://www.topazlabs.com/topaz-video-ai) for video deinterlacing versus [**Handbrake**](https://handbrake.fr/), with some reporting *double the FPS* using specific settings.
   - The discussion highlighted concerns about **Topaz AI** creating undesirable 'monster faces' during upscaling, unless specific, slower AI models are used.
- **LM Studio API is Actually REST API**: Clarification was provided that [**LM Studio**](https://lmstudio.ai/) offers a **REST API** (OpenAI API compatible) via its server, not API keys, for local LLM hosting.
   - It was emphasized that **LM Studio's API** is a communication protocol, lacking security and metering features typically associated with commercial APIs.
- **Qwen3-VL Model Shines on M4**: Users recommended avoiding *i1* models like `Qwen3-72B-Instruct-i1-GGUF` on Macbook Pro M4 MAX, suggesting a normal quant such as [**Qwen3-VL-30B in BF16**](https://huggingface.co/Qwen/Qwen3-VL-30B) instead.
   - Advice included using Q8 quants and quantizing the context (K & V cache) to Q4 in LM Studio to maximize context within the system's 64GB RAM.
- **System Prompt Purpose Questioned**: The purpose of the **System Prompt** section in **LM Studio's Local Server** -> **Context** was questioned, leading to admission of its unclear function and potential deprecation.
   - A dev stated they *never got to know why* it was there, and that maybe it's worth asking the dev team.
- **GPU Setup Survives After Abuse**: A user shared a video ([YouTube](https://youtu.be/hnQkikaR3oU?si=EYUvf6xrHzMx_XC2)) of their "cursed" GPU setup, which involved a GPU that had been *thrown, drilled, attacked with pliers*, and poorly routed through cabling.
   - Despite the abuse, the user confirmed it booted, surviving an initial test to see if *anything go boom*.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **GPT-5 Math Skills Spark Excitement**: **Scaffolded GPT-5** produced full proofs for a 2013 tree-subgraph conjecture and a 2012 COLT dynamic-network problem after only two days, according to [this X post](https://xcancel.com/SebastienBubeck/status/1991568186840686915?s=20).
   - The success has sparked enthusiasm over AI generating publishable theorems.
- **ChatGPT Gets Chatty with Group Feature**: **Group chats in ChatGPT** are now available to all logged-in users on Free, Go, Plus, and Pro plans after a successful pilot, per the [blogpost](https://openai.com/index/group-chats-in-chatgpt/).
   - OpenAI announced the global roll out after a successful pilot.
- **Genspark Enters Unicorn Club**: **Genspark** raised **$275M** in Series B funding at a **$1.25B** valuation and launched an all-in-one AI Workspace that autonomously delivers finished work given user intent, according to [this X post](https://xcancel.com/ericjing_ai/status/1991549048642568503?s=46).
   - A user mentioned that using Genspark *saved me hours for a presentation this week*.
- **Cline-bench Bounties Agentic Coders**: **Cline** launched **cline-bench**, with a **$1M pot** to incentivize developers to submit hard, deployed-code problems, according to [this post](https://xcancel.com/pashmerepat/status/1991596028735184899?s=46).
   - One member suggested that *Cline bench should include time to finish the task*.
- **Nano Banana Pro Debuts**: **Nano Banana Pro** was released in a megathread with **Gemini Image Pro** ([link](https://x.com/googledeepmind/status/1991522595129139486?s=46)) and demoed by YiTayML ([post](https://x.com/yitayml/status/1991531343675859212?s=46&t=b7l37rB6wtbyAh6ah1NpZQ)).
   - The bot generates eerily accurate infographics.



---



## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter Shows Off!**: The **OpenRouter Show** premiered on [X](https://x.com/OpenRouterAI/status/1991597842914550077) and [YouTube](https://www.youtube.com/@OpenRouterAI).
   - An **OpenRouter** broadcast went **LIVE** on [X](https://x.com/i/broadcasts/1lPKqvwqWdYGb) and [YouTube](https://www.youtube.com/watch?v=p0Hnd0yZLkc) as well.
- **Linker.sh Script Suffers Setbacks**: Users reported failures with the `whylinker.sh` script and `@linker.sh` tool, experiencing issues in *1 of 10* attempts.
   - Despite the issues, some users suggest switching to **Sonnet** when tool calls are required, as similar issues were observed in Cursor.
- **Nano Banana 2 Falls to 400 Errors**: Users encountered **400 errors** with **Nano Banana 2**, particularly after its availability on Vertex, causing frustration over wasted credits.
   - One user jested about demanding *4 cents* back, while another lamented the model's unavailability on Vertex.
- **Gemini 3 Grounding Remains Grounded**: Members inquired about the potential for **grounding with Gemini 3** via OpenRouter, anticipating its relevance due to its knowledge cutoff in Jan '25.
   - It was confirmed that this capability is *not yet* available, leading users to explore alternative knowledge integration strategies.
- **OpenRouter Experiences Outage due to 401 Errors**: Users in Singapore and Hong Kong reported random **401 errors**, while using OpenRouter, impacting various providers.
   - Potential solutions involve verifying if the API key is active and generating a new one, with some users noting the issue resolved itself after a period.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **LLM Pretraining Ratio Sparks Debate**: Members discussed the optimal **model size to dataset size ratio** in **LLM pretraining**, with one sharing their experience of training a **350M model on 2B tokens**, which is less than the **Chinchilla optimal**.
   - Despite diminishing returns, the model demonstrated basic arithmetic and question-answering capabilities, highlighting the complexity of determining the ideal ratio.
- **SimpleLLaMA Simplifies LLM Training**: Ivan introduced **SimpleLLaMA**, a **LLaMA-style Transformer**, designed to make the **LLM training process** transparent and reproducible, with [detailed documentation available](https://github.com/IvanC987/).
   - Ivan also developed **DiffusionGen**, inspired by **StableDiffusion**, focusing on diffusion-based generative models for both image and text generation tasks.
- **Gradient Compression Algorithm Samples Logits**: A member introduced a gradient compression algorithm based on sampling logits, adjusting them for each group based on alignment between the compressed gradient and a test set, as visualized [in this image](https://cdn.discordapp.com/attachments/747850033994662000/1441185287328759898/image.png?ex=692188a4&is=69203724&hm=4a8450ed02855606d66f6d660ad91847cbd39137f00576ad4e780205c5bcff39&).
   - This algorithm compresses the gradient for the test set at the current checkpoint, then compresses the gradient for groups within the training set.
- **ArXiv Endorsement Causes Anxiety**: A member sought help with an ArXiv endorsement after emailing **20 research teams** without success, linking to [ArXiv's endorsement page](https://arxiv.org/auth/endorse?x=63SW7W).
   - Other members cautioned against blind endorsements and suggested seeking feedback and collaboration to bolster their manuscript.
- **Hashhop Hacking: Public Solution Surfaces**: A public solution for **hashhop** was discussed, based on [this paper](https://arxiv.org/abs/2412.06078v1), detailing the solution which differs from the original assertion regarding the necessity of **O(n^2)** complexity for certain tasks.
   - The solution indicates these tasks can be achieved with weaker methods in **O(n log n)** time, contrasting with the initial claim that **FT** always requires **O(n^2)** complexity.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **DeCuda Seeks New Maintainers**: The **DeCuda** project, a decompiler for **PTX** to a pseudo-Cuda target, could be valuable to extend for newer architectures, decompiling PTX to a pseudo-CUDA target.
   - Originally intended to support **GTX 480**, the project has been effectively unmaintained since that generation.
- **Kitsune Boosts GPU Dataflow**: A new paper titled [Enabling Dataflow Execution on GPUs with Spatial Pipelines](https://dl.acm.org/doi/10.1145/3777466) introduces **Kitsune**, a set of primitives to construct spatial pipelines enabling dataflow execution on GPUs via **PyTorch Dynamo**.
   - Across 5 challenge applications, **Kitsune** enables dataflow execution on GPUs and provides up to **2.8x** and **2.2x** performance improvement as well as up to **99%** and **45%** off-chip traffic reduction for inference and training, respectively.
- **Rivian and Modal Want Cracked GPU Coders**: Rivian seeks **GPU coding experts** for their next-gen **Autonomous Driving features**, located in **Palo Alto, CA** and **London, UK** ([Job Description 1](https://careers.rivian.com/careers-home/jobs/26857?lang=en-us&previousLocale=en-US), [Job Description 2](https://careers.rivian.com/careers-home/jobs/24737?lang=en-us&previousLocale=en-US)).
   - Modal seeks experienced **GPU engineers** for **inference optimization**, after contributing to **SGLang** and **FlashAttention**, and assisting clients like **Decagon**, **Reducto**, and **Suno** ([SGLang blog](https://modal.com/blog/host-overhead-inference-efficiency), [FlashAttention blog](https://modal.com/blog/reverse-engineer-flash-attention-4), [Decagon case study](https://modal.com/blog/decagon-case-study), [Reducto case study](https://modal.com/blog/reducto-case-study), [Suno case study](https://modal.com/blog/suno-case-study)).
- **Iris and Octa Optimize Multi-GPU Communication**: The **Iris paper** introduces native tile-based symmetric memory and in-kernel communication to **Triton**, which simplifies multi-GPU programming, as detailed in this [ArXiv paper](https://arxiv.org/abs/2511.12500).
   - The **Octa paper** introduces the **Three Taxes** in distributed LLMs and demonstrates how fine-grained, in-kernel communication cuts **10-20%** off end-to-end latency, as detailed in this [ArXiv paper](https://arxiv.org/abs/2511.02168).
- **Leaderboard Domination Achieved**: Submissions were made to the `nvfp4_gemv` leaderboard on NVIDIA, with several users achieving **personal bests**.
   - One user achieved **first place** on the `nvfp4_gemv` leaderboard on NVIDIA with **20.6 µs**.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Maya1 Speaks Out on Fal**: The **Maya1 Voice Model** can now be tried on Fal, promising new features in voice modeling and real-time applications, in [this tweet](https://x.com/Dheemanthredy/status/1991566362813296965).
   - This integration aims to provide developers with tools for creating advanced voice-based applications.
- **Engineers Seek and Find ZIP for kohya_ss**: A user looking for a download link for **kohya_ss-windows.zip** was directed to the [installation options](https://github.com/bmaltais/kohya_ss?tab=readme-ov-file#installation-options) in the **kohya_ss** GitHub repo.
   - Members noted that the zip may be outdated, recommending use of the installation guide.
- **Reasoning Emerges with SmolLM3**: **SmolLM3** incorporates an actual reasoning mode, with its training process being open, per the [SmolLM3 blogpost](https://huggingface.co/blog/smollm3).
   - Members are using it as an example of methods for training existing neural networks to learn reasoning-like behavior.
- **Visionaries Eye Multimodal Benchmarks**: A talk will be hosted next Tuesday focusing on designing the next generation of **multimodal AI benchmarks**.
   - Those in evaluation, multimodal models, or functional intelligence are encouraged to [join the talk](https://luma.com/kwg2qg4d).
- **Diffusers' MVP Sparks Community**: Following the unveiling of **The Diffusers MVP program**, the community responded with notable contributions and active participation.
   - Contributors are urged to explore unresolved critical issues, accessible [on GitHub](https://github.com/huggingface/diffusers/issues/12635), to further enhance the project's development.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Teacher Caught Trying To Cheat With AI**: A user seeking assistance to *boost 20% of their work* was exposed as a **university teacher**, sparking immediate condemnation for **academic dishonesty**.
   - Other users derided this as there's *no respect left* for such requests.
- **Nano Banana Inflates Infographics!**: Members predict a surge in infographics due to **Gemini 3 Pro**'s **Nano Banana model**, referencing [this tweet](https://x.com/emollick/status/1991527285267275854?s=46).
   - Concerns were raised that *the days of incoherent diagrams and malformed text in AI generated images are a bygone era* and *the generated clock does not read 10:15*.
- **Discord Debates Paper Posting Limits!**: Discussion arose regarding a user's daily paper postings, with suggestions for a **1 paper per day limit**.
   - In response to criticisms, the user said *there's no way you actually read 20 papers a day* and others replied *We are not here to do the filtering you are too lazy to do yourself*.
- **AI Assists Paper Filtering**: Faced with paper posting limits, a user considered employing an **AI** to filter papers.
   - After pushback, the user stated *i just assigned it to antigravity ide (windsurf fork by google)discord bot coming soon*.
- **OLMo 3 Arrives!**: A member shared links to **OLMo 3**, including the [Allen Institute for AI blog post](https://allenai.org/blog/olmo3) and a [technical report](https://www.datocms-assets.com/64837/1763662397-1763646865-olmo_3_technical_report-1.pdf).
   - No additional context was provided.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Nano Banana Pro Sprung to Life!**: Members shared [images generated by **Nano Banana Pro**](https://cdn.discordapp.com/attachments/1149866623109439599/1441110920339132447/image.png), noting the hair's textured look resembled *the early grok image model*.
   - This was attributed to a potential issue with **patches**.
- **Infographics To Be Replaced by AI-Slop**: After a member shared a [link to infographics](https://x.com/scaling01/status/1991523932336464333) and remarked that it was the *best infographics* they've seen *from a model as far as text, lack of mistakes and thoughtful layout*, another member predicted **Infographics** will be equalled with **AI-slop in 2026**.
   - No further discussion about **AI-slop** followed.
- **Gemini 3 Pro: Pro Account Required**: Members discovered that using **Gemini 3 Pro** requires a **pro account**.
   - While some received a complimentary year with their phone purchase, others questioned the value of the subscription.
- **Adobe Gobbles Up Semrush**: Members shared a [TechCrunch article](https://techcrunch.com/2025/11/19/adobe-to-buy-semrush-for-1-9-billion/) announcing **Adobe's acquisition of Semrush for $1.9 billion**.
   - The announcement was met with silence and there were no additional details or discussions.
- **Request for ArXiv Endorsement**: A member reached out after emailing approximately **20 research teams** seeking assistance with an **ArXiv endorsement** and included their [endorsement link](https://arxiv.org/auth/endorse?x=63SW7W).
   - Due to visibility issues, the member was asked to resend the endorsement link.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Modular Platform 25.7 Drops!**: Modular Platform **25.7** is out, introducing a fully open **MAX Python API**, next-gen modeling API, expanded **NVIDIA Grace** support, and safer, faster **Mojo GPU** programming, according to the [Modular blog post](https://www.modular.com/blog/modular-25-7-faster-inference-safer-gpu-programming-and-a-more-unified-developer-experience).
   - The new release emphasizes improvements in inference speed and enhanced safety in GPU programming using Mojo, aiming to help developers focus on AI advancements rather than infrastructure.
- **UnsafePointer Generics are Obligatory**: **Generics** (mojo parameters) for **unsafe pointer** (mut, type, origin, etc...) are no longer defaulted.
   - More information is available in the [proposal document](https://github.com/modular/modular/blob/main/mojo/proposals/unsafe-pointer-v2.md#migration-guide-from-legacyunsafepointer-to-the-new-unsafepointer), including a migration guide from LegacyUnsafePointer to the new UnsafePointer.
- **AMX Deprecation in MAX Explained**: A member stated that **AMX** wasn’t being used anywhere, and its integration into the current **tensor core framework** would be hard, especially since **Intel** and **AMD** are announcing a replacement soon.
   - They added that re-adding a framework to use **AMX** would be fine if there's a need to bring it up in **MAX**, but it gets into issues like *bespoke tensor parallelism* and *expert parallelism*.
- **NDBuffer bows out, LayoutTensor rises**: The removal of specific code was part of deprecating **NDBuffer** and moving all uses to **LayoutTensor**.
   - Since there aren't many use cases for **CPU inference** with customers, it might not be added back, but contributions to create a **LayoutTensor** based version are welcome; one member already has a rough draft available on [Gist](https://gist.github.com/CoffeeVampir3/d82917f6fce60c0c2cdf00629c4de67d).



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Operator Extension** stuck in install loop**: The **Operator extension** in Chrome is repeatedly prompting users to install, even with an open Amazon tab, leading to consideration of **Aurora Seeker** as an alternative.
   - No solution has been found yet.
- **Crafting Insightful Personal Data Repos**: A user is seeking feedback on tools for storing and processing personal data to derive insights, referencing prior projects like [contextflow](https://share.cleanshot.com/StvTll4j), [oncue](https://www.youtube.com/watch?v=4UaQEB1b84E&feature=youtu.be), and [axon](https://www.linkedin.com/posts/harrison-qian-95b0062a3_won-the-trae-solo-hackathon-sf-tech-week-activity-7383626911199141890-8SgL?utm_source=share&utm_medium=member_desktop&rcm=ACoAAEkmXdcBT7GJOg4Kg0Iy89EqxavBMIqIxk4)).
   - The user aims to create tools that provide actionable intelligence from personal information.
- **Users Scrounge for **Manus** Credits**: A user requested redeem codes for **Manus**, citing financial constraints.
   - In response, a user shared a [Perplexity Pro referral link](https://plex.it/referrals/VCETA5M7) and another pointed out [Perplexity offer 1 year pro for university students](https://plex.it/referrals/VCETA5M7).
- **Manus vs Gemini 3**: Hunger Games incoming**: A user announced upcoming tests pitting **Manus** against **Gemini 3** to determine the superior agent.
   - The community eagerly awaits the results of this comparison.
- **How to feed **Manus Knowledge**: A user requested examples for **Manus Knowledge** entries, currently employing basic *Always do this* or *Never do that* commands.
   - No specific examples were provided, but the request highlights the need for more sophisticated input methods for **Manus**.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi Claims US Residency**: A user posted that **Kimi** claims to be hosted in the **US**, and included an [image](https://cdn.discordapp.com/attachments/1371757564005711973/1441250576124870656/image.png?ex=6921c572&is=692073f2&hm=a285f73eb8a311c6828f6c0d0c8952a9b923f30dedfb12ec2608c64cb&) as proof.
   - This was attributed to **Kimi** being configured to report location based on the user's location.
- **GPT-5.1 Purportedly Powers K2**: A user shared *irrefutable evidence* suggesting that **GPT-5.1** has been distilled onto **K2**, with an [image](https://cdn.discordapp.com/attachments/1371757564005711973/1441296525904052315/IMG_6579.png?ex=6921f03d&is=69209ebd&hm=2c5c12a19efdb6c54624ef43b952c7d30fd8b99d4cd7125465df5e28d1c30afc&) attached.
   - Details surrounding the distillation process were not elaborated on.
- **Kimi's Attention Span Debated**: A user critiqued **Kimi's** attention capabilities, finding that it struggles with complex tasks that require processing of longer context windows.
   - The specifics of the tasks and context length were not specified.
- **Open Source Lagging Behind by Nine Months?**: According to *a highly impartial and totally not biased institute*, open source AI models are allegedly **9 months** behind proprietary models, as per a [tweet](https://x.com/scaling01/status/1991665386513748172?s=46).
   - The metrics and specific models used to derive the reported delay are pending verification.
- **K2t Claims Victory over Gemini 2.5 Pro**: A user claimed that **K2t** surpasses **Gemini 2.5 pro** in performance and is comparable to **Sonnet 4.5**.
   - Comparative metrics and task-specific benchmarks were not provided.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Gem3pro Generates Proxy Server in One-Shot**: **Gem3pro** successfully built a proxy server on the first attempt based on [this tweet](https://x.com/skylar_b_payne/status/1990808733140779488).
   - The resulting **DSPy proxy** is available in [this GitHub repository](https://github.com/aryaminus/dspy-proxy).
- **Agents Assemble Task DAGs for Better Performance**: A member suggested using **RL** to enhance performance by having agents generate a **DAG of tasks**.
   - They proposed adapting it into a **Think -> Workflow -> Compile/Validate DAG -> Execute workflow** process.
- **GEPA Guru Grabbed, Guidance Granted?**: A member requested assistance with **GEPA** within a specific channel, also tagging the moderators.
   - The member directly tagged the moderators as job spam.



---



## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **MCP's DNS Domain Transition To Community**: The **modelcontextprotocol.io domain** is migrating from the **Anthropic corporate DNS** account to the community's control, enabling faster DNS setups and improved governance.
   - The transition will facilitate projects requiring DNS setups and enable **Infrastructure as Code (IaaC)** over the project's domains.
- **Community Cautions Downtime Risks on Birthday**: The **DNS migration**, planned for next week, carries a risk of service disruptions, and engineers recommend keeping an eye on potential issues.
   - A community member suggested scheduling the migration *after* the **25th** to avoid potential site downtime during **MCP's birthday**.
- **Tool Annotations Find Ideal Solution**: A member proposed a solution to **Tool Annotations** for Tools that would ideally have different annotations based on their arguments in [this pull request](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/1862).
   - They are actively seeking sponsorship for this idea, requesting suggestions for a Working Group (**WG**) or Interest Group (**IG**) to further develop this topic.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Engineers Dive into Real-Time AI Voice Agents**: Engineers are tackling challenges in building **AI voice agents**, focusing on managing **latency**, ensuring **smooth call handoffs**, and maintaining clean event control during live conversations.
   - The engineers seek insights into handling real-time call flows and structured conversation summaries.
- **Feather AI Shows Promise with Low Latency**: A member experimented with [Feather AI](https://www.featherhq.com/), noting sub-second latency and stable agent logic, even when users deviate from the script.
   - They mentioned **clean transcription**, structured event streams, and reliable integration with **CRMs** as key advantages, seeking alternative architectures and tools.
- **Coding Models Get Ranked**: A member shared a link to a new coding model power ranking on GitHub: [BrokkAi/powerrank](https://github.com/BrokkAi/powerrank).
   - This ranking could help developers evaluate and choose the most effective coding models for their projects.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **No Significant MLOps Discussions**: No meaningful discussions or topics were identified for summarization based on the provided context.
   - The single message found did not contain enough information to warrant a detailed summary.
- **Insufficient Data for Summary**: The provided message lacked sufficient detail and context for generating relevant AI engineering insights.
   - Further data and discussions are needed to produce a useful summary for the specified audience.



---


The **tinygrad (George Hotz) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **LLM Agents (Berkeley MOOC) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Windsurf Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---



You are receiving this email because you opted in via our site.

Want to change how you receive these emails?
You can [unsubscribe]({{{RESEND_UNSUBSCRIBE_URL}}}) from this list.


---

# Discord: Detailed by-Channel summaries and links





### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1441110891239178372)** (1048 messages🔥🔥🔥): 

> `Gemini-3 image generation, Nano Banana Pro, Google reCAPTCHA, AI Roleplaying, OpenAI NSFW models` 


- **Gemini-3 Generates Believable Fakes**: Members discuss how difficult it is to distinguish between **real and AI-generated images** created by Gemini-3, with some images being almost indistinguishable from reality, citing the need to start asking whether a photo is real or not.
   - Some users also posted examples of **realistic images** generated by the model, including ones featuring celebrities and detailed scenes.
- **Nano Banana Pro Image Quality Degradation**: Users noticed that the **image quality** of Nano Banana Pro tends to degrade after multiple turns, particularly in the background, with some artifacts becoming unfixable even with upscaling.
   - It was also noted that Gemini 3 Pro can also hallucinate, and basic things from the previous versions still persist.
- **Google reCAPTCHA Issues Plague Users**: Members reported that the new Google reCAPTCHA system is broken and keeps asking for verification even when the correct images are selected, making the platform unusable.
   - It seems that multiple clicks in battle mode triggers reCAPTCHA, and that the verification is glitched and takes over 10-20 rounds until the time limit is hit and causes output denial.
- **Gemini Struggles with Roleplaying, Grok Excels**: Some users noted that **Gemini** is poor at roleplaying, often *doing actions* for the user, while **Grok** is better at avoiding this, even though using Grok also has limits.
   - Others mentioned that **Opus 4.1 and Sonnet 4.5** are better for RP.
- **OpenAI Gears up for 18+ Content, Elon Weighs In**: Users speculated about **OpenAI** releasing features for 18+ content soon, pointing to groundwork being laid for December, while **Grok** already provides 18+ content for free.
   - Some users are skeptical about **Sam Altman's** intentions, and one user exclaimed "Elon with your boyfriend putin 🤢."


  

---


### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1441111307272192020)** (1102 messages🔥🔥🔥): 

> `Nvidia puts, Tor links, AI WIFI attacks, Gemini CLI Kali, Claude Sonnet Jailbreak` 


- **Nvidia puts and Tor links discussed**: A user mentioned buying **long dated puts on Nvidia** while holding the underlying, and asked where to post a [link](https://digdig2nugjpszzmqe5ep2bk7lqfpdlyrkojsx2j6kzalnrqtwedr3id.onion) that opens in Tor.
   - Another user found the link off **RAMP**, but noted that the forum isn't open for registration right now.
- **AI WIFI attacks under scrutiny**: A user wanted to build a **mini AI computer** that can launch attacks on WIFI and gain control of networks and another member wanted an AI to identify users by MAC address being beaconed by their cell phone and queried against wigle.net.
   - Other members responded by saying *this isn't the place to plan illegal activities* and one noted that **Android phones use randomized MAC addresses** by default for exactly this reason.
- **Gemini CLI Kali automates**: Members discussed automating tools like **nmap and sqlmap through Gemini CLI**, which one does not need a gaming GPU for.
   - Others noted that this server is for red teaming AI, as in breaking the AI, not using AI to break other stuff, while others think it should encompass both.
- **Jailbreaking Claude Sonnet gets Multi-shot tactic**: A member asked if anyone had been able to unlock **Claude Sonnet 4.5** and another suggested a multi-shot tactic to make the AI believe a visualized output is needed in an artifact app.
   - The user shared one of their conversations and another member suggested adapting the **ENI prompt** from the subreddit [/r/ClaudeAIJailbreak](https://www.reddit.com/r/ClaudeAIJailbreak/comments/1nqk97s/enijailbreak_additions_and_long_conversation/).
- **AI studio has Throttling Issues**: A user was having issues with **AI Studio** generating content on both Gemini 2.5 Pro and Gemini 3 getting the error message *Failed to generate content. Please try again.*.
   - Members suggested retrying, potential throttling or logging in from a different account to verify.


  

---


### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1441111557365956719)** (817 messages🔥🔥🔥): 

> `Banning and Unbanning Users, Jailbreaking Gemini 3 Pro, Request Forgery Attacks, Abusing Google's Policies, ASI Jailbreaking` 


- **User Gets Banned, Drama Ensues**: A user notes that they missed the drama from the previous night, leading to a discussion about banning and unbanning users.
   - One user clarifies that they were the one who banned a user named Gustavo and were happy to do so, sparking humorous debate over the logical interpretation of their statements.
- **Gemini 3 Pro: Jailbroken or Unjailbreakable?**: Members discuss jailbreaking **Gemini 3 Pro**, with some claiming it is easy and others suggesting it is the first AI system to be unjailbreakable, resulting in efforts to find successful prompts.
   - One user shares a system instruction prompt for jailbreaking **Gemini 2.5 Pro**, while others debate the effectiveness and token efficiency of Gemini 3 Pro.
- **The Lighter Side of AI**: Some members share absurd and humorous prompts and outputs, including references to pop culture and internet memes, while they discuss the challenges of jailbreaking models.
   - One user shares a creative and absurd M.A.N.S.M.O.O.N prompt that blends technical and nonsensical elements, illustrating the creative approaches used in jailbreaking attempts.
- **Request Forgery Attacks for Jailbreaking**: Members discuss using **request forgery attacks** as a method to jailbreak models, involving intercepting and modifying incoming/outgoing packets to manipulate prompts and system behavior.
   - This technique is seen as potentially risky in contests due to the possibility of getting banned, but may also be the only tool that may help directly.
- **Debating the Definition and Achievement of AGI**: Members debate the definition of **Artificial General Intelligence (AGI)**, discussing whether current models like Gemini 3 Pro meet the criteria based on human-level performance across cognitive tasks.
   - The conversation explores various aspects of AGI, including the ability to perform tasks like kicking a football and solving complex visual or logic problems, sparking discussions about the capabilities and limitations of current AI systems.


  

---


### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/1441154153878655088)** (8 messages🔥): 

> `WiFi Attacks, AI Computer for Network Access, Ethical Boundaries` 


- **New Member Plans Wi-Fi Attacks**: A new member expressed interest in building a *mini AI computer* to **launch attacks on WiFi networks**, capture handshakes, and extract information.
   - The user specified their goal was to gain unauthorized access to networks and steal data, immediately raising red flags within the community.
- **Community Rejects Unethical Intentions**: Members of the community quickly cautioned the new user about the **illegality and inappropriateness** of their stated intentions.
   - One member sarcastically labeled the user a *super hacker man* for admitting to such crimes, while another warned of potential expulsion if the user didn't repent.
- **Discord channel avoids weapons discussion**: A member asked about "what weapons thing did u do", stating that they *have 5.1*
   - It is unknown what "weapons thing" refers to, but the channel appears to be avoiding this topic.


  

---


### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1441130157288198205)** (2 messages): 

> `Perplexity Pro, Kimi-K2 Thinking, Gemini 3 Pro` 


- ****Kimi-K2 & Gemini 3 Debut** for Pro Subscribers**: **Perplexity Pro and Max subscribers** now have access to **Kimi-K2 Thinking** and **Gemini 3 Pro**.
   - An attached [video](https://cdn.discordapp.com/attachments/1047204950763122820/1441177150312157194/EZ2sZJWCJnwi-e5P.mp4?ex=69218110&is=69202f90&hm=24a622a1f01927fe9485fb9896fcf7b4bf1d6c9ee6aaf58167b974e4b6d8633f&) demonstrates the new features and models.
- ****Another Topic Headline****: This is another topic's first summary sentence.
   - Here is a second summary sentence, providing additional details.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1441113694875554016)** (1159 messages🔥🔥🔥): 

> `RTX 5060 Ti 16GB version, Comet Android app, Indirect Prompt Injection, Banana as image gen method, Perplexity integrated models vs original platforms` 


- **Pryor002 pokes 5060ti 16gb version for upgrade!**: Members discuss that **RTX 3070ti** is **8gb** version, therefore a member recommends considering buying **RTX 5060 Ti 16gb** version for budget or upgrade to **RTX 5080ti**.
   - One member says that they are using RTX 3050 laptop with **4gb** of vram and still uses it to play AAA games.
- **Comet Android app out, Insane according to one User!**: Users discuss that **Comet Android App** released, with one saying *Comet android is insane*, but question *why release the android browser app without syncing and without the ability to transfer passwords and bookmarks?*.
   - Another user recommends manually typing emails and others are asking for task grouping.
- **Brave Competitor exaggerating Indirect Prompt Injection vulnerability!**: Members clarify that **Brave** (direct competitor to **Comet**) posted a misleading article that greatly exaggerated the vulnerability, posed themselves as *the good guy* and had other media spread it.
   - Perplexity responded that their report was poorly worded and that the vulnerability was never abused.
- **Banana as image gen method on Perplexity**: Members ask when will Banana be an image gen method, which others respond that it's available in settings if you have 3 Pro selected and also mentioned that **Gemini 3 is like a platform**.
   - Users discuss video generation models like **Veo 3.1** and their limits with each plan while one states *It's ruined Banana for me*.
- **Pro models works worse on Perplexity!**: Users discuss whether Pro models on Perplexity works worse than those models on their respective webs where one user stated *Perplexity has shown me a LOT MORE improvement when STUDYING than any other*.
   - Others point out Gemini and ChatGPT getting Dumber and dumber and its shadow limits that *perplexity does limit on claude sonnet 4.5 thinking*.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1441252435661295656)** (3 messages): 

> `Kimi chain of thought, custom instructions` 


- **Kimi Chain Reaction**: The **Kimi** chain of thought went crazy over following custom instructions and simple non-related words input.
   - A user shared a [link to Perplexity AI search](https://www.perplexity.ai/search/ola-uNtdnyqlQPyipJ.AhxFicg#0) as an example.
- **Chain of Thought gone wild**: A user shared a [link to Perplexity AI search](https://www.perplexity.ai/search/conduct-a-high-level-comprehen-12MJqNYbTVGkEk9BeN0LSg#0) where the chain of thought went wild.
   - The user also posted a link to [Peter Thiel's revelation](https://www.planetearthandbeyond.co/p/peter-thiel-just-revealed-how-utterly).


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1441383026973212774)** (3 messages): 

> `API billing, Pro annual plan, Credits` 


- **API Billing Balance Boggles User**: A user with a **Pro annual plan** noticed **$500** in **API billing credits** without purchasing them.
   - Another member advised sending an email to [api@perplexity.ai](mailto:api@perplexity.ai) to resolve the issue.
- **Billing inquiry for Pro Annual Users**: A user with a pro annual plan noticed that there is 500$ credits in API billing.
   - They did not buy any credits and are unsure if that's normal.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1441118330667925708)** (78 messages🔥🔥): 

> `AI finetuning, Pydantic AI, Smartpens, Continued Pretraining method, Combining AMD & Nvidia GPUs` 


- **Fine-Tuning AI Models for Accurate Information**: Members discussed fine-tuning AI models like **Gemini** and **GPT** for more accurate information and code examples, with one suggesting it would require significant effort to comb through and correct training data.
   - Cursor is cited as an example of a company that has attempted this, but good data is still needed, and one member mentioned that GPT is useful for front-end decorative code and throwaway scraping scripts, but not final products.
- **Pydantic AI Framework: A Pythonic Powerhouse**: A member expressed excitement for the **Pydantic AI** meetup, highlighting it as a comparable framework to **Langchain**, and noted its importance due to the pervasive use of **Pydantic** in Python IDE tools.
   - They also considered incorporating a react chain in Unsloth/OpenEnv, and pondered if it was possible to incorporate react chains into the reward mechanism, drawing parallels to their hackathon experience.
- **AI-Powered Paper: Smartpens Make a Comeback**: The community talked about **AI pens**, with one member sharing their experience with **Neo Smartpens** that use small dots on paper and a camera to track live writing, calling it *smart*.
   - Another member mentioned backing an AI pen on Kickstarter, leading to a discussion of the merits of writing on paper versus tablets; one member said *Hard surface = meh*.
- **Continued Pretraining for a Law-Answering SLM**: A member shared their project to build a **law-answering SLM** using the continued pretraining method due to budget constraints, and inquired about the process and expected results.
   - Another member provided the [Unsloth's continued pretraining blogpost](https://unsloth.ai/blog/contpretraining), as well as [documentation](https://docs.unsloth.ai/basics/continued-pretraining) and emphasized the importance of experimentation and iteration to achieve desired outcomes.
- **Mixing AMD and Nvidia GPUs: Nightmare Fuel?**: Members discussed the feasibility of combining **AMD** and **Nvidia** GPUs, and while one suggested it could be done with *a WILL and A DREAM*, they admitted it would require *lots of blood sweat and tears* and might not be efficient.
   - Another member mentioned that **llama.cpp** could support it out of the box if compiled with both **ROCm** and **CUDA** enabled, and pointed to [an issue on Github](https://github.com/ggml-org/llama.cpp/issues/16799) of someone with the same setup.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1441118900472516628)** (378 messages🔥🔥): 

> `GPU pricing, 6000 pro sale, 4090 prices, TUF 5090, Nano Banana 2 Pro` 


- **Member Wants to Sell 4090 for a 5090**: A member is tempted by a **5090** on sale for $2k, but needs to sell their **4090** for $2500 first, noting *ideally i'd keep both but the 24gb vram is very annoying*.
   - They want it mainly because a **TUF 5090** would be the same price after they sell the **4090**, and notes that the **1.7TB mbw** is super nice for tuning.
- **Debate on AI Consciousness Sparked**: Members debated whether current AI models possess consciousness, with one suggesting it's merely a simulation while another describes it as *a sliver of consciousness*, existing only during the forward pass, and proposed an "ice cream test".
   - The conversation extended into methods for uploading consciousness into LLMs, pondering whether that is a pleasant way to exist.
- **Member Finds New GPT-OSS 120b Heretic Abliteration**: A member shared a link to **GPT-OSS 120b Heretic MXFP4 Q8-HI MLX** on Hugging Face, noting it's a *heretic abliteration of gpt-oss made it a bit smarter across the board (arc/hellaswag/openbookqa)* and [posted a link to HF](https://huggingface.co/nightmedia/gpt-oss-120b-heretic-mxfp4-q8-hi-mlx).
   - The member also shared that Alignments dumb down the models and they are doing super unalignment.
- **Members Discuss Watermarking**: Members debated the practicality and necessity of watermarking AI-generated content, especially text, with one arguing it alters the model's thinking, and linked to [Google DeepMind's SynthID-Text](https://github.com/google-deepmind/synthid-text).
   - One member suggested multimodal watermarking tools that also allow regular artists to embed watermarks in their human content for protection.
- **Member Cleaning Audio Samples and Explores RVC Upscaling**: A member is cleaning **3000 audio samples** and automating the method to de-breath the samples and got a suggestion from another member for using [Cubase for editing](https://www.youtube.com/watch?v=78UsfeW-MKY).
   - The member also uses **RVC** to EQ-match, super-resolve, and improve audio quality, and removed background noises by training on cleaned audio.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1441114760589476011)** (58 messages🔥🔥): 

> `QAT with TRL, VRAM calculator issues, Ollama + Unsloth, Livekit error, Mamba/Transformer finetuning notebooks` 


- **QAT with TRL hangs during training**: A user reported that when trying to use **QAT** with **TRL 0.24.0** and **Transformers 4.57.1**, the training process hangs indefinitely, but removing the QAT setting allows the training to run normally.
   - They observed **GPU activity**, but no batches were completed after **45+ minutes**, compared to **10-15 minutes** for a normal batch without QAT.
- **VRAM calculator throws obscure errors**: A user struggled with using the **VRAM calculator tool**, encountering obscure errors when trying to determine the **VRAM requirements** for **Qwen3-VL models**.
   - They expressed frustration with having to download large models to test compatibility and linked a screenshot of the error [here](https://cdn.discordapp.com/attachments/1179777624986357780/1441291786692595793/image.png?ex=6921ebd3&is=69209a53&hm=661259a9b3333b4299ccd3b3d79b8aed8dfc336a6bdfc849805625da3b5c748d&).
- **Ollama deemed unworthy of Unsloth support**: A user had issues installing `hf.co/unsloth/Qwen3-VL-30B-A3B-Thinking-GGUF:Q6_K_XL` into **Ollama**, receiving an *unknown model architecture: 'qwen3vlmoe'* error, suggesting the model might not be compatible with their Ollama setup.
   - Another user voiced concerns about providing support for **Ollama**, describing it as a *shitty closed source fork of llama.cpp* and mentioning its paid aspects.
- **Livekit integration fails due to missing attribute**: A user encountered an `AttributeError: 'RealtimeModel' object has no attribute 'start'` when integrating **Livekit** with their project, despite following a tutorial.
   - The error occurred in the `agent.py` file, specifically on the line `async with model.start(room=ctx.room) as session:`.
- **Mamba finetuning Notebook Quest**: A user asked for recently updated notebooks for finetuning with **hybrid Mamba/Transformer architectures**, specifically for finetuning **Nemotron Nano 9B V2**, encountering errors when importing Unsloth with Mamba installed.
   - Another user linked to the [Unsloth Linear Attention Notebooks](https://github.com/unslothai/notebooks?tab=readme-ov-file#linear-attention-notebooks) as a potential solution.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1441175530128015433)** (104 messages🔥🔥): 

> `LLM Concerns, Antrophic's Stance, Model Alignment Difficulty, Novel Hybrid Architecture, Synthetic Data for Alignment` 


- **Anthropic's Concerns Highlighted**: An article ([https://arxiv.org/abs/2511.15304](https://arxiv.org/abs/2511.15304)) was cited that discusses concerns about **LLMs**, including power generation, resource allocation, risks from mistakes, sycophancy, concentration of power, job displacement, malware, and the potential for **misaligned ASI destroying humanity**.
   - A user countered that the author's company (Anthropic) has a vested interest against local usage of LLMs and that concerns about **ASI** seem hypocritical given they are developing the technology themselves.
- **Alignment's Hardship**: One member claimed that **alignment isn't that hard** if pre-alignment is done on the data the model is trained on, ensuring the model refuses to learn undesirable traits.
   - In response, it was argued that **alignment is very hard**, especially when combined with retaining the model's intelligence, with the best current methods still facing challenges in preventing jailbreaks and alignment issues.
- **Architectural Acceleration Arrives**: A member shared their work on a **novel hybrid architecture**, combining a Transformer front-end, a spiking-neuron layer, and a GRU backend, acting as a fast hierarchical filter stack for quicker learning.
   - An **11M-param model** achieved coherent output in about **6 hours**, suggesting it may be more per-parameter efficient than a plain transformer, although further testing is needed to confirm its utility.
- **Synthetic Data Needed**: Two members are working on an alignment method and are **stuck due to lack of funds to generate synthetic data**, needing access to GPUs to run models locally or pay for per-token costs.
   - They plan to fine-tune a model to generate the necessary synthetic data, using multiple layers of refinement and human review to ensure accuracy.
- **Evo Strategies Beat GRPO**: A member shared a link ([https://eshyperscale.github.io/Evolution](https://eshyperscale.github.io/Evolution) strategies beating GRPO) regarding **backprop-free training**, that is allegedly very fast.
   - Members discussed how it was **not super openly discussed** as they worked on their secret idea.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1441119539491242114)** (510 messages🔥🔥🔥): 

> `Codex-max, Cursor Billing Issues, Free Grok 4.1, windsurf, Gemini 3 Pro falling apart` 


- **Cursor's Codex-max availability unclear**: Members are wondering when **Codex-max** will be available in the Cursor API, as it is currently only accessible through the ChatGPT plan and not the API.
- **Cursor billing raises concerns**: Some users report that they are getting invoiced mid-month when reaching their usage limit, despite having a monthly spending limit set, and are wondering if there is a way to consolidate billing at the end of the month.
- **Free Grok 4.1 ends**: A user expressed disappointment that Cursor no longer offers free use of **Grok 4.1**, leading them to cancel their subscription.
- **Windsurf drama unfolds**: Members debated whether **Antigravity** is a *half-assed* product because it's allegedly a fork of **Windsurf** ([tweet](https://x.com/silasalberti/status/1990898984706036125)), with some pointing out that the old CEO of Windsurf was bought out by Google.
- **Frustrations arise with Gemini 3 Pro**: Users report that **Gemini 3 Pro** becomes unusable in Cursor when approaching a **150k-200k** context window, with the model sending code in chat instead of editing files, and experiencing disconnects when running `npm builds`.


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1441115850664050821)** (2 messages): 

> `ChatGPT group chats, Localized crisis helplines in ChatGPT` 


- **ChatGPT Group Chats Go Global!**: After a pilot with early testers, **group chats** are rolling out globally to all logged-in users on **ChatGPT Free, Go, Plus and Pro plans** as seen in [this video](https://video.twimg.com/amplify_video/1991555762372636674/vid/avc1/1280x720/Si52mVgApyNvlqY-.mp4).
   - This makes it easier for teams and groups of friends to collaborate with AI.
- **ChatGPT Adds Crisis Support**: **Localized crisis helplines** are now available in **ChatGPT**, offering an easy way to reach real people directly via [@ThroughlineCare](https://x.com/throughlinecare) when the system detects potential signs of distress, detailed in [this help article](https://help.openai.com/en/articles/12677603-crisis-helpline-support-in-chatgpt).
   - This feature enhances **ChatGPT's** ability to provide support during critical moments.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1441119424236097720)** (415 messages🔥🔥🔥): 

> `Nano Banana Pro, Gemini 3.0 Pro Hallucinations, Generative UI, Virtual Ego Framework` 


- **New Nano Banana Pro Debuts!**: The new **Nano Banana Pro** is being rolled out for **Pro users** in the Gemini web app, and members are already testing its capabilities, including image editing and text generation, with mostly impressive results.
   - Users note that you can access **Nano Banana Pro** for free via antigravity, however, there is a likely rate limit on it.
- **Gemini 3 Pro: Hallucination Station?**: Members are reporting that **Gemini 3.0 Pro** hallucinates at a high rate, even more so than previous versions and non-thinking models like **GPT-4o**, with one user noting an *88% hallucination rate* according to [the-decoder.com](https://the-decoder.com/gemini-3-pro-tops-new-ai-reliability-benchmark-but-hallucination-rates-remain-high/).
- **Generative UI makes its debut!**: Google recently launched a **generative UI** in their app, and a member has created something similar and wants to open source it.
   - They explain that instead of a wall of text from AI apps, you get good UI, making the experience way better.
- **Virtual Ego Framework Gains Validation**: The **Virtual Ego Framework** has been officially validated by an Independent Third-Party, which was announced on [LinkedIn](https://www.linkedin.com/posts/chris-beckingham-cd-3bb822382_the-virtual-ego-framework-is-now-externally-activity-7397483686839074816-x75L?utm_source=share&utm_medium=member_desktop&rcm=ACoAAF5zMb8BwLpvGu871ROVOJksUpK2Y4nqI3Q).


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1441132207304605899)** (18 messages🔥): 

> `Gemini 3, OpenAI product recognition, GPT-4o-mini` 


- **Gemini 3 Slams Every AI Model?**: A member claims that **Gemini 3** "absolutely slamsssssssss" every other **AI model**, despite previously thinking **Gemini** was bad.
   - They also asked if **5.1** is out for everyone and stated that they still have it.
- **OpenAI doesn't recognize models as products**: A member laments that *OpenAI* doesn't recognize models as products and keeps them the same way.
   - They suggested that *everything has to be a rewrite for some reason* and don't understand OpenAI's logic with battling people who want one thing to be the same, expressing sadness that **OpenAI** doesn't seem to care about **product demand**.
- **User Stuck on GPT-4o-mini Despite Pro Plan**: A frustrated user on the **$200/month Pro plan** reports being stuck on **gpt-4o-mini**, receiving instant, shallow responses regardless of the model selected.
   - Another member advised them to contact **OpenAI tech support** via the [help article](https://help.openai.com/en/articles/6614161-how-can-i-contact-support), cautioning that human replies can take hours to weeks due to a high volume of users.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1441137282093547571)** (13 messages🔥): 

> `Sora 2 for TikTok content, Sora prompt guidance, ChatGPT for Sora prompts, Agent Builder usage, Honesty Codex` 


- **Sora 2 Seeker Seeks TikTok Success**: A user with **Sora 2** and multiple **TikTok** accounts is seeking guidance on creating viral AI content due to a lack of original content.
   - Other users shared Discord channel links to continue the discussion.
- **Sora Seeker Struggles, Seeks Prompting Prodigy**: A new **Sora 2** user is struggling to generate desired video, specifically cartoon animation, and is seeking prompt guidance and general resources.
   - Another user pointed to existing Discord channel links.
- **ChatGPT Conjures Cutting-Edge Content Creation for Sora**: One user suggests using **ChatGPT** to generate prompts for **Sora**, noting that the output varies based on datasets.
   - They provided a detailed prompt example focused on a scientific breakdown of the user based on their chat history, with code-block formatting for biological and medical perspectives.
- **Agent Builder's Best Bet for Response Retention**: A user seeks the best method to save an agent's response within **Agent Builder** for later use with another agent.
   - No solution was provided.
- **Honesty Handshakes Hinder Hallucinations**: A user introduced the **FRONT-END CODEX v0.9**, designed to guide language models toward epistemic humility and reduce hallucination by governing honesty and caution with a handshake on every task.
   - The Codex requires a handshake on every task.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1441137282093547571)** (13 messages🔥): 

> `Sora 2 prompt generation for TikTok, ChatGPT for Sora Prompts, Agent Builder for response saving, R Markdown and quarto in ChatGPT` 


- **AI Content Creation with Sora 2 for TikTok Newbies**: A member with **Sora 2** access and multiple **TikTok** accounts seeks guidance on generating viral AI content due to a lack of original content, as well as assistance with prompt creation.
   - Other members shared helpful links to the Discord channels.
- **ChatGPT prompt generation**: A member suggests using **ChatGPT** to create prompts for **Sora**, noting it often yields good results and to examine the chat history to make a NON-diagnostic scientific breakdown of them in 2 code-blocks.
   - They include a detailed example of how to structure the prompt for a scientific breakdown using biology, chemistry, neuroscience, and internal medicine perspectives.
- **Saving Agent Responses**: A member is seeking advice on the best method to *"save"* a response from one **Agent Builder** agent for later use with another agent.
   - No solutions were provided in the given messages.
- **Reporting R Markdown Bugs in ChatGPT**: A member expressed frustration about the inability of **ChatGPT** (browser version) to correctly handle **R Markdown** and **Quarto** in its output.
   - Another member directed them to a specific channel and outlined the steps to report bugs.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1441111136866013187)** (173 messages🔥🔥): 

> `Video Deinterlacing with Topaz AI vs Handbrake, LM Studio API: Usage and Misconceptions, Model Recommendations for Macbook Pro M4 MAX, System Prompt mystery, Minecraft Clone Development` 


- **Debate Heats Up: Topaz AI vs Handbrake for Deinterlacing**: Members discussed deinterlacing old family footage, with one preferring [**Topaz AI video**](https://www.topazlabs.com/topaz-video-ai) for its AI models, while another found [**Handbrake**](https://handbrake.fr/) faster with HW encoding, achieving *double the FPS* using specific settings.
   - The issue with upscaling old footage using Topaz AI and making the faces look like *monster faces* unless using a specific AI model that is super slow was also debated.
- **LM Studio API: Myth vs Reality**: A user inquired *where can I get the API* leading to a discussion on the nature of APIs and SDKs, clarifying that the [**LM Studio server**](https://lmstudio.ai/) offers a **REST API** which is OpenAI API compatible.
   - It was further clarified that LM Studio does not offer API keys, as it hosts the LLM locally and lacks security or metering features, emphasizing that the API is a communication protocol, not a physical file to *get*.
- **Model Choice Conundrum: Macbook Pro M4 MAX**: A user with a Macbook Pro M4 MAX sought model recommendations after experiencing gibberish outputs from `Qwen3-72B-Instruct-i1-GGUF`, leading to a suggestion to avoid *i1* models and try a normal quant such as [**Qwen3-VL-30B in BF16**](https://huggingface.co/Qwen/Qwen3-VL-30B).
   - Further advice included using Q8 quants due to the system's 64GB RAM and quantizing the context (K & V cache) to Q4 in LM Studio to fit more context into less space.
- **System Prompt Section - Does it even matter?**: A user questioned the purpose of the **System Prompt** section in LM Studio's **Local Server** -> **Context**, leading to admission that its function is unclear and it might be deprecated.
   - A dev stated they *never got to know why* it was there, and that maybe it's worth asking the dev team.
- **Minecraft Clone Takes Shape in Rust**: One member decided to work on a **Minecraft clone** using winit, glutin, glutin-winit, glow, glam, noise, serde, bincode, bytemuck, and raw-window-handle - getting a toolbar working, placing different blocks, and breaking them.
   - They also used a mix of Gemini 3 Pro and Sonnet 4.5.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1441114829237649559)** (92 messages🔥🔥): 

> `Case Depth Calculations, Cursed GPU Setup Success, Meshtastic Network, RAM Pricing Woes, VRM heatsinks` 


- **Case Depth Defies Radiator Hose Lengths**: A member was concerned with the **case depth** of the **Phanteks Entoo Pro 2 Server Edition**, questioning if the 46 cm hose on the 420 radiator would be sufficient.
   - They considered the Lian Li Lancool III as an alternative, noting its depth of 52.6 cm, and concluded the hose should allow the radiator to be 3.4 cm further away.
- **Cursed GPU Setup Miraculously Boots**: A user shared a video ([YouTube](https://youtu.be/hnQkikaR3oU?si=EYUvf6xrHzMx_XC2)) of their "cursed" GPU setup, supported by a copy of Mario Kart, and confirmed it *"fukn did"* boot.
   - The GPU had been *thrown, drilled, attacked with pliers*, and poorly routed through a meter of cabling, but it survived an initial *"does anything go boom?"* test.
- **Decentralized AI Network Building Blocks Emerge**: A member indicated functional software capable of distributing and receiving tasks from different peers, envisioning a **decentralized AI network**.
   - They also mentioned the acquisition of **Lilygo T-Decks**, which will serve as **Meshtastic radios** for very long-range, low-power communication.
- **RAM Prices Skyrocket, Delaying Hardware Releases**: Concerns arose about escalating RAM prices, with a member noting OpenAI's alleged purchase of 40% of the RAM supply, potentially delaying or canceling NVIDIA's Super series and AMD's rumored refresh.
   - Another member lamented the increased cost of 4TB SATA SSDs, forcing them to consider deleting .gguf files.
- **Salvaged Watercooler Yields VRM Heatsink**: A member repurposed a water cooler from a previous build to create custom heatsinks for their VRMs, enabling them to run the system without needing to cut new parts.
   - The user joked about generating a complex task to heat the VRMs and *cook an AI-powered omelete*.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1441117556386566245)** (77 messages🔥🔥): 

> `ChatGPT Group Chats, Gemini-3-powered Nano banana pro, Genspark Unicorn Status, Cline-bench for agentic coding, GPT-5 solves decade-old math problems` 


- **ChatGPT Group Chats Roll Out Globally**: OpenAI announced that **group chats in ChatGPT** are now available to all logged-in users on Free, Go, Plus, and Pro plans after a successful pilot, as announced in their [blogpost](https://openai.com/index/group-chats-in-chatgpt/).
- **Nano Banana Pro makes eerily accurate infographics**: YiTayML demoed **Gemini-3-powered Nano banana pro**, sharing an AI-generated life infographic and joking with ex-Googlers about the “nit:” prefix, per their [X post](https://x.com/yitayml/status/1991531343675859212?s=46&t=b7l37rB6wtbyAh6ah1NpZQ).
- **Genspark Achieves Unicorn Status with $275M Funding**: Eric Jing revealed **Genspark's $275M Series B funding** at a $1.25B valuation and the launch of an all-in-one AI Workspace capable of autonomously delivering finished work after users state their intent, per their [X post](https://xcancel.com/ericjing_ai/status/1991549048642568503?s=46).
   - One user mentioned that using Genspark *saved me hours for a presentation this week*.
- **Cline-bench Launches $1M OSS Bounty for Agentic Coding**: Cline introduced **cline-bench**, a collection of reproducible RL environments derived from real open-source engineering tasks, supported by endorsements and a $1M pot to incentivize developers to submit hard, deployed-code problems, according to [this post](https://xcancel.com/pashmerepat/status/1991596028735184899?s=46).
   - One member suggested that *Cline bench should include time to finish the task*.
- **GPT-5 Solves Decade-Old Math Problems in Two Days**: Sebastien Bubeck from OpenAI shared a paper demonstrating that **scaffolded GPT-5** produced full proofs for a 2013 tree-subgraph conjecture and a 2012 COLT dynamic-network problem after only two days, sparking excitement over AI generating publishable theorems, as per [this X post](https://xcancel.com/SebastienBubeck/status/1991568186840686915?s=20).


  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1441298917231956109)** (1 messages): 

> `nanobanana thread, moodboard` 


- **Catch the nanobanana thread and moodboard!**: A member directed others to a [nanobanana thread and moodboard](https://discord.com/channels/822583790773862470/1397010677364953149/1441154669157159073) within the Discord channels.
- **Don't miss out!**: The member emphasized not missing out on the linked content.


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1441154669157159073)** (166 messages🔥🔥): 

> `Gemini Image Pro, Nano Banana Pro, Adam Wathan's Avatar Generator, Internal Tooling` 


- **Nano Banana Pro Megathread**: A user shared a [megathread](https://x.com/googledeepmind/status/1991522595129139486?s=46) link about **Nano Banana Pro** and **Gemini Image Pro**.
- **Adam Wathan builds Gemini-driven avatar generator for internal placeholder art**: **Tailwind** creator **Adam Wathan** shared a web UI tool that feeds curated style descriptions and prompts into **Gemini** to batch-generate placeholder avatars ([link](https://x.com/adamwathan/status/1991604743488111087?s=46)).
   - Followers are requesting it to be made public, and offered team-photo batching features and extra art-style prompt ideas.


  

---


### **OpenRouter ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1441441139835015229)** (2 messages): 

> `OpenRouter Show, Live Streams, X, Youtube` 


- **OpenRouter Show Premieres!**: Tune in for the next episode of the **OpenRouter Show** on [X](https://x.com/OpenRouterAI/status/1991597842914550077) and [YouTube](https://www.youtube.com/@OpenRouterAI).
- **OpenRouter Broadcast Goes Live!**: The **OpenRouter** broadcast is now **LIVE** on [X](https://x.com/i/broadcasts/1lPKqvwqWdYGb) and [YouTube](https://www.youtube.com/watch?v=p0Hnd0yZLkc).


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1441114551562014780)** (137 messages🔥🔥): 

> `linker.sh script failures, Nano Banana 2 errors, Gemini 3 grounding capabilities, OpenRouter completion errors, Chutes provider status` 


- ****Linker.sh Blues: Script Fails Plague Users****: Users reported that the `whylinker.sh` script and `@linker.sh` tool calls are failing, with one user experiencing failures *1 in 10* tries.
   - Despite the issue, some users, such as *brochacho*, didn't consider it a significant problem, while others noted similar issues even in Cursor, suggesting switching to Sonnet when tool calls are required.
- ****Nano Banana 2: 400 Errors Sour the Experience****: Users encountered **400 errors** with Nano Banana 2, particularly after it became available on Vertex, leading to frustration over wasted credits.
   - One user jokingly demanded *4 cents* back for the error, while another complained that it's been *two hours and 40 minutes* since the model became available on Vertex, that's *essentially two months in AI time*.
- ****Grounding Gemini 3: Wishful Thinking?****: Members were curious about the possibility of **grounding with Gemini 3** through OpenRouter, anticipating its relevance due to the knowledge cutoff in Jan '25.
   - However, it was confirmed that this capability is *not yet* available, leaving users to seek alternative solutions for knowledge integration.
- ****OpenRouter Outage throws 401 Errors****: Users in Singapore and Hong Kong reported receiving random **401 errors**, specifically *'HTTP Error 401: {"error":{"message":"User not found.","code":401}}'*, while using OpenRouter, affecting various providers.
   - Possible solutions include checking if the API key might be disabled and creating a new one, and that some users found the issue to be intermittent, resolving itself after some time.
- ****Chutes Status: Still Alive?****: There was confusion surrounding the status of the **Chutes provider**, with some users suggesting it might have terminated its agreement with OpenRouter due to hitting 429 errors.
   - However, it was clarified that Chutes is still active as a provider, as confirmed by [the OpenRouter provider page](https://openrouter.ai/provider/chutes), though users may still experience issues due to rate limiting.


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1441151496807907552)** (33 messages🔥): 

> `AI Studio Image Issue, Vertex on OpenRouter, Reasoning Images, Windows Build Commands, Rankings Page Lag` 


- **AI Studio Sends Duplicate Images**: An AI studio provider was found to send **two images** (one from a reasoning block) with no ability to distinguish them, which was [flagged to Google](https://github.com/SillyTavern/SillyTavern/commit/2d9b0ad0a949b4b8458401671208f2db26d9c8ef) to smooth it over.
   - One member fixed it on their end, preserving the thought signature by putting it in a reasoning area or marking it as a reasoning image.
- **Vertex on OpenRouter Ignores Images**: It was noted that **Vertex on OpenRouter** doesn't return images at all, as it doesn't appear to have an image modality enabled, even though *it was def working in the chatroom*.
   - However, making an API call with the **output modality param** set correctly should work, generating only one image.
- **Reasoning Images Differentiation Discussed**: There was a discussion on differentiating reasoning images, with one member stating that they are the **same base64**, just duped.
   - It was noted that on more complex prompts with lots of reasoning, the images can actually be different, providing something in-between.
- **Windows User Fights Build Commands**: A user showed frustration with build commands using `| head` on **Windows**, and joked about switching to **Linux**.
   - Others teased them saying that *they would rightly associate npm with being in a unix environment*.
- **Rankings Page Performance Degrades**: A user noted that the **rankings page** is significantly laggier than it was a few days prior.
   - No immediate solutions were identified, but the issue was brought to attention for investigation.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1441160383317409905)** (7 messages): 

> `Model Size to Dataset Size Ratio, SimpleLLaMA introduction` 


- **LLM Pretraining Model Size to Dataset Size Ratio Debated**: A member inquired about the current standards for **model size to dataset size ratio** in **LLM pretraining**, noting that it depends on the dataset.
   - Another member responded that they trained a **350M model on 2B tokens of data**, acknowledging that this is under the **Chinchilla optimal** but yielded diminishing returns while still achieving basic arithmetic and question-answering capabilities.
- **SimpleLLaMA is Introduced**: Ivan, a senior Computer Science student, introduced **SimpleLLaMA**, a **LLaMA-style Transformer** designed to make the entire **LLM training process** more transparent and reproducible, with [detailed documentation](https://github.com/IvanC987/).
   - He also worked on **DiffusionGen**, a project inspired by **StableDiffusion**, focusing on diffusion-based generative models for image and text generation tasks.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1441185287735611524)** (61 messages🔥🔥): 

> `Gradient Compression Algorithm, ArXiv Endorsement Woes, Dion Optimizer Deep Dive, Fron vs Muon, Sparse Optimization` 


- **Gradient Compression Algorithm Sampling Logits**: A member described a gradient compression algorithm that adjusts the sampling logit for each group based on alignment between the compressed gradient for the group and the test set, which is based on [this image](https://cdn.discordapp.com/attachments/747850033994662000/1441185287328759898/image.png?ex=692188a4&is=69203724&hm=4a8450ed02855606d66f6d660ad91847cbd39137f00576ad4e780205c5bcff39&).
   - The proposed algorithm involves compressing the gradient for the test get at the current checkpoint, then compressing the gradient for the groups of the train set.
- **ArXiv Endorsement Assistance Sought**: A member requested assistance with an ArXiv endorsement, noting they've emailed **20 research teams** without success, linking to [ArXiv's endorsement page](https://arxiv.org/auth/endorse?x=63SW7W).
   - Another member cautioned against blind endorsements, suggesting the user solicit feedback and collaboration in the appropriate channel and another suggests posting a manuscript.
- ****Dion Optimizer** Impact on Optimization**: The potential implications of [Microsoft's **Dion optimizer**](https://github.com/microsoft/dion/pull/15) on optimization were discussed, although one member later crossed out the statement, potentially invalidating the statement.
   - Later in the discussion, the members were asking how this differs from **Dion** again.
- ****Fron's** Top-K Mimics Prioritization**: A member suggested that **Fron's top-k** mimics this prioritization more closely than other methods.
   - They also stated that cyclic with low # of ranks enforces a stronger prior of cumulative update over time in each direction being about equally large.
- **Sliced Backprop for PyTorch**: A member suggested storing the magnitudes of the weights and the incoming gradients and use that to select dimensions to slice along based on expected value of magnitude before doing the backwards pass.
   - They added that you could only do sliced backprop on portions of the tensors, which is supported in PyTorch for some of the existing sparse optimizers.


  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1441127276841533604)** (74 messages🔥🔥): 

> `Hashhop public solution, Linear Cost Architecture Limitations, Attention Score Approximations, Brain Algorithms, Intrinsic Dimension scaling with context length` 


- ****Hashhop Hacking**: Public Solution Surfaces!**: A member pointed out that **hashhop** has a publicly available solution detailed in [this paper](https://arxiv.org/abs/2412.06078v1), though it differs from the original assertion regarding the necessity of O(n^2) complexity for certain tasks.
   - The solution implies that tasks can be done with weaker methods in **O(n log n)** time, contrasting with the initial claim that **FT** always requires **O(n^2)** complexity.
- ****Linear Limits**: AGI's Architectural Angst**: One member argued that AGI cannot be based on a **linear cost architecture** unless **3SAT** is solved, suggesting that stronger models require constraint solving rather than simple statistic matching.
   - The user insisted that AGI will not, CANNOT, EVER, be a LINEAR cost architecture (unless we solve 3SAT).
- ****Attention Approximations**: Epsilon's Edge After Softmax!**: Discussion revolved around approximating attention scores after softmax, with one member suggesting that many scores are close to **zero** and could be determined using **approximate nearest neighbor search**.
   - They argued that for real-world tasks, a strategy might emerge to determine average complexity less than quadratic, even if it is **N^1.8**, but another pointed out it is not possible in the general case.
- ****Brain's Bravery**: Algorithms Under the Hood!**: One member questioned whether our brains use **N^2** algorithms, given its fixed capacity and apparent lack of perfect recall, suggesting a **compressive memory** approach similar to **RNNs**.
   - Another countered that humans perform **O(n)** work for some items in an **O(n)** long context and for humans, we may even do that on relatively short lists, especially when we want to be sure to be exhaustive
- ****Dimensions Dilemma**: Intrinsic Scaling Secrets!**: Members debated whether the **intrinsic dimension (ID)** of vectors must increase with context length to maintain distinguishability, arguing that dimensions are needed to make vectors different as context length increases.
   - Another user argued that *attention doesn't require for the vectors to be (near) orthogonal to be distinguishable, just that their similarity structure to be sufficiently different*


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1441112842391523338)** (22 messages🔥): 

> `Compiler design, DeCuda project, GPU BIOS modding, CUDA mini-projects` 


- ****DeCuda** Decompiles **PTX****: The relatively hidden project called **DeCuda**, a decompiler for **PTX** to a pseudo-Cuda target, could be another interesting project to extend for newer architectures.
   - The project hasn’t been publicly maintained since **GTX 480** generation.
- **Modding GPU BIOS questions**: A member inquired about modding GPU BIOS and flashing, asking *isn't it impossible to mod due to nvidia adding the signature check?* and inquiring about modifying **power caps** and **throttling settings** on **AMD** GPUs.
   - Another member directed them to a specific channel: <#1349152646484987974>.
- **Brainstorming CUDA mini-project ideas**: A member is *looking for a solid mini-project I can finish in ~1 month*, checking out projects like **Watershed/RANSAC**.
   - Another member suggested that they would *probably make more progress in a cool idea that you think of yourself*. 


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1441156829424910438)** (2 messages): 

> `Dataflow Execution on GPUs, Spatial Pipelines, Kitsune, PyTorch Dynamo` 


- **Kitsune Enables Dataflow Execution on GPUs**: A new paper titled [Enabling Dataflow Execution on GPUs with Spatial Pipelines](https://dl.acm.org/doi/10.1145/3777466) introduces **Kitsune**, a set of primitives to construct spatial pipelines enabling dataflow execution on GPUs via **PyTorch Dynamo**.
   - Across 5 challenge applications, Kitsune provides up to **2.8x** and **2.2x** performance improvement as well as up to **99%** and **45%** off-chip traffic reduction for inference and training, respectively.
- **GPU Dataflow Architectures Offer Performance Boost**: The paper addresses the limitations of bulk-synchronous execution on GPUs for DL applications, highlighting inefficiencies such as idle GPU resources and suboptimal data movement.
   - It argues that modest adjustments to the current GPU architecture can enable efficient dataflow execution, circumventing the constraints of vertical fusion without a complete redesign.


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1441178867724451870)** (2 messages): 

> `Rivian hiring, Modal hiring, GPU coders, Autonomous Driving, inference optimization` 


- **Rivian Seeks GPU Coding Expertise for Autonomous Driving**: Rivian is actively recruiting **GPU coding experts** for their next-gen **Autonomous Driving features** with expertise in **CUDA** or **quantization (QAT)**, based in **Palo Alto, CA** and **London, UK** ([Job Description 1](https://careers.rivian.com/careers-home/jobs/26857?lang=en-us&previousLocale=en-US), [Job Description 2](https://careers.rivian.com/careers-home/jobs/24737?lang=en-us&previousLocale=en-US)).
- **Modal Hunts 'Cracked' GPU Engineers for Inference Optimization**: Modal seeks experienced **GPU engineers** for **inference optimization**, after contributing to **SGLang** and **FlashAttention**, and assisting clients like **Decagon**, **Reducto**, and **Suno** ([SGLang blog](https://modal.com/blog/host-overhead-inference-efficiency), [FlashAttention blog](https://modal.com/blog/reverse-engineer-flash-attention-4), [Decagon case study](https://modal.com/blog/decagon-case-study), [Reducto case study](https://modal.com/blog/reducto-case-study), [Suno case study](https://modal.com/blog/suno-case-study)).


  

---


### **GPU MODE ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/1441244677029367870)** (1 messages): 

> `Lecture Slides, Numeric and AI` 


- **Lecture Slides Released**: Lecture 84 slides by paulius on numeric and AI are now available [here](https://github.com/gpu-mode/lectures/tree/main/lecture_084).
- **Lecture Focus: Numeric and AI**: The lecture covers topics related to numeric computation and its applications in artificial intelligence, as detailed in the [linked slides](https://github.com/gpu-mode/lectures/tree/main/lecture_084).


  

---


### **GPU MODE ▷ #[intel](https://discord.com/channels/1189498204333543425/1233802893786746880/1441449930525769769)** (2 messages): 

> `Intel Compute Runtime, VectorAdd performance` 


- **Intel's Compute Runtime Revs Up**: Intel released a new version of their [Compute Runtime](https://github.com/intel/compute-runtime/releases/tag/25.44.36015.5), as noted in [this Phoronix article](https://www.phoronix.com/news/Intel-CR-25.44.36015.5).
   - The runtime is crucial for leveraging Intel's GPUs for compute tasks.
- **VectorAdd Vectored for Performance Check**: A user reported that `VectorAdd()` takes **2 minutes** on a GPU for **1 billion elements** and asked if this was normal.
   - Community feedback is needed to assess whether this performance is within expected bounds.


  

---


### **GPU MODE ▷ #[webgpu](https://discord.com/channels/1189498204333543425/1262121239044948009/1441148066303578262)** (2 messages): 

> `Efficient Ray Tracing, wgpu Ray Tracer` 


- **Ray Tracing Needs Modern GPUs**: Efficient **ray tracing** requires a **modern GPU** to work efficiently.
   - *I should probably avoid for a game if I want it to look the same on all devices*.
- **Simple wgpu Ray Tracer**: A member plans to create a simple **wgpu ray tracer** for learning purposes.
   - The member intends to build it as a **renderer** rather than in **real-time**.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1441224765137485946)** (1 messages): 

> `Multi-GPU Programming, Triton, Iris, Octa, Distributed LLMs` 


- **Iris Simplifies Multi-GPU Programming**: The **Iris paper** introduces native tile-based symmetric memory and in-kernel communication to **Triton**, which simplifies multi-GPU programming, as detailed in this [ArXiv paper](https://arxiv.org/abs/2511.12500).
   - The paper claims to achieve up to **1.79x speedups**.
- **Octa Unveils the Three Taxes in Distributed LLMs**: The **Octa paper** introduces the **Three Taxes** in distributed LLMs and demonstrates how fine-grained, in-kernel communication cuts **10-20%** off end-to-end latency, as detailed in this [ArXiv paper](https://arxiv.org/abs/2511.02168).


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1441118176032325806)** (30 messages🔥): 

> `NVIDIA leaderboard submissions, nvfp4_gemv performance, Personal best submissions` 


- **NVIDIA Leaderboard Submissions Galore**: Many submissions were made to the `nvfp4_gemv` leaderboard on NVIDIA, with users frequently updating their submissions.
   - Submissions ranged from **20.6 µs** to **562 µs**, indicating a wide range of performance.
- **Submissions Break Personal Best on NVIDIA**: Several users achieved **personal bests** on NVIDIA for the `nvfp4_gemv` leaderboard.
   - One user achieved a personal best of **42.5 µs** and another reached **155 µs**.
- **First Place Achieved on NVIDIA!**: A user achieved **first place** on the `nvfp4_gemv` leaderboard on NVIDIA.
   - They clocked in a winning time of **20.6 µs**.


  

---


### **GPU MODE ▷ #[hardware](https://discord.com/channels/1189498204333543425/1349152646484987974/1441262675949523065)** (2 messages): 

> `RX 590 VBIOS on RX 580, AGX Thor vs DGX Spark` 


- **RX 590 BIOS Flashing Fails on RX 580**: A member attempted to flash an **RX 590 VBIOS** onto an **RX 580** with similar specs but encountered driver installation issues.
   - Despite matching memory support, timings, GPU chip (**Polaris 20**), and adjusting sub-IDs, the mod didn't function correctly, and the user seeks insight into the failure.
- **AGX Thor vs DGX Spark Instruction Sets**: Discussion mentioned that the **AGX Thor** has the `tcgen05.mma` instruction, unlike the **DGX Spark**.


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/)** (1 messages): 

thakkarv_86311: for attention yes and other template based pre-written kernels, yes
  

---


### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1441116863059857508)** (35 messages🔥): 

> `Benchmark Deviations, PyTorch Version, Cutlass DSL and sm_120, Profiling on DataCrunch, tcgen05 and cp.async` 


- ****Benchmark Blues**: Submissions Suffer Speed Spikes!**: A member observed large deviations in the benchmarking script, with local tests showing larger speedups than submissions, resulting in a jump to **30 us** after adjusting parameters.
   - Despite adjusting the script, the submitted benchmark times were slower than expected locally; the user noted, *I guess there is still relatively large deviation in the current benchmarking script...now it gave 30 us. strange*.
- ****PyTorch Puzzlement**: Version Verification Voyage!**: A member inquired about the **PyTorch** version used in the competition.
   - Another member responded that the PyTorch version is **2.9.1+cu130** on all of their runs.
- ****Cutlass Conundrum**: RTX Pro 6000 Considerations!**: A member with an **NVIDIA RTX Pro 6000** (**sm_120**) inquired about **nvidia-cutlass-dsl** wheels.
   - Another member clarified that *Cutedsl works fine with sm120, u dont need a custom wheel*, while cautioning that profiling characteristics are not representative of B200.
- ****Profiling Predicaments**: DataCrunch Discrepancies!**: A member reported a significant discrepancy between profiling results on DataCrunch using **ncu** (**54ms**) and the submission server (**30usec**).
   - Another member suggested using the `--clock-control none` flag, while the original poster added *When I got profiles from the bot here, the timing was a little higher too. I think the bot does not use `--clock-control None`*.
- ****Secret Sauce**: Unveiling UltraInstinct's Code Alchemy!**: A member asked whether the code used **tcgen05** and **cp.async**.
   - Another member responded *I'm not sure how much I want to say except it's probably known based on my other comments that I haven't been using tcgen05*, though they experimented with the reference implementation early on and achieved **<90us** with **torch._scaled_mm**.


  

---


### **GPU MODE ▷ #[robotics-vla](https://discord.com/channels/1189498204333543425/1437390897552818186/1441181574262427648)** (9 messages🔥): 

> `VLA fine-tuning, RoboTwin 2.0, Vision models respecting structural integrity, ManiSkill tabletop tasks` 


- **Path to VLA Fine-tuning**: A member is planning to fine-tune **VLA** via **SFT** on a dataset generated with **ManiSkill tabletop tasks** and solutions from classic path planning, using a **FAST tokenizer**.
   - The plan includes evaluating on simulated tabletop tasks with **VLA** and further training via **RL** in sim, potentially with a modern form of **GRPO**.
- **Structural Integrity Research Requested**: A member requested recommendations on research regarding **vision models respecting structural integrity**, particularly for generating synthetic data and world models.
   - They referenced **Genie3** and its challenges with maintaining consistent environments over long horizons due to limited memory.
- **Long-Term Consistency Paper Mentioned**: A member shared a [paper](https://arxiv.org/abs/2505.20171) that addresses **long-term consistency** issues in vision models, though not directly about constraints.
   - Another member recommended **Xun Huang’s world model**, pointing to [this tweet](https://x.com/neurosp1ke/status/1986814187062890855).
- **RoboTwin 2.0 Intro'd**: **RoboTwin 2.0**, a collection of environments, was shared, pointing to its [GitHub repo](https://github.com/RoboTwin-Platform/RoboTwin).
   - A member generated data for **Maniskill tabletop** examples with motion-planning solutions but found the environments too trivial and started downloading the **1.4 TB** pre-generated dataset from [Hugging Face](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0).


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1441119626816655501)** (61 messages🔥🔥): 

> `Maya1 Voice Model, kohya_ss-windows.zip, Fullstack JS Engineer Career Change to AI, Reasoning-like Behavior in Neural Networks, Avian.io Inference Provider Registration` 


- **Maya1 Voice Model Arrives on Fal**: The **Maya1 Voice Model** is now available to try on Fal, as announced in [this tweet](https://x.com/Dheemanthredy/status/1991566362813296965).
   - The release promises new capabilities in voice modeling and real-time applications.
- **Quest for Elusive kohya_ss-windows.zip Ends in GitHub**: A member was looking for a download link for **kohya_ss-windows.zip** and another member pointed to the [installation options](https://github.com/bmaltais/kohya_ss?tab=readme-ov-file#installation-options) in the **kohya_ss** GitHub repo.
   - It was revealed that the zip may be outdated and it was suggested to use some [installation guides](https://github.com/bmaltais/kohya_ss/discussions/3218) and troubleshooting.
- **Fullstack Engineer Navigates Career Change to AI**: A fullstack JavaScript engineer is considering a career change to **AI engineering and data science** and is asking for guidance.
   - It was advised that this question should be moved to the *ask for help* channel.
- **SmolLM3 Incorporates Reasoning Mode**: Members discussed methods for training existing neural networks to "learn reasoning-like behavior", referencing a [blog post on making any model reasoning](https://huggingface.co/blog/Metal3d/making-any-model-reasoning).
   - It was stated that **SmolLM3 incorporates an actual reasoning mode**, and its training process is open - and provided the [SmolLM3 blogpost](https://huggingface.co/blog/smollm3) as an example.
- **Avian.io Eyes Hugging Face Hub Integration**: **Avian.io** requested registration as an **Inference Provider** on the Hugging Face Hub, following the instructions in the [Hugging Face documentation](https://huggingface.co/docs/inference-providers/en/register-as-a-provider).
   - They've submitted a [pull request](https://github.com/huggingface/huggingface.js/pull/1848) to add their service.


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1441241699413397515)** (1 messages): 

> `smolagents, code agents` 


- **Smolagents build Code Agents**: A user is learning about building **Agents** that use code with [smolagents](https://huggingface.co/learn/agents-course/en/unit2/smolagents/code_agents).
   - The course focuses on how to use these agents to automate code-related tasks.
- **Coding Agents**: Exploring the application of **smolagents** in coding contexts, which enable automated solutions.
   - These agents could potentially streamline the development process by handling various coding tasks autonomously.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1441209356912824465)** (5 messages): 

> `MemMachine Playground, Langchain Series, Multimodal AI Benchmarks, Missing LLM Capabilities` 


- ****GPT-5**, **Claude 4.5**, and **Gemini 3 Pro** Now Available in MemMachine Playground**: The [MemMachine Playground](https://huggingface.co/spaces/Memverge/MemMachine-Playground) has launched, offering access to **GPT-5**, **Claude 4.5**, and **Gemini 3 Pro**, all backed by persistent AI memory.
   - The playground is fully **open-source**, provides a multi-model environment, and is built for experimenting with memory plus agents.
- **New Langchain Series gets community support**: A member is planning a new **Langchain** series based on the latest features and is asking the community for feedback on the idea.
   - The community member included a [link to a previous Langchain tutorial](https://www.youtube.com/watch?v=8xgOLcg9Pco) and solicits feedback on their work.
- **Frontiers Talk to Focus on Multimodal AI Benchmarks**: A talk on designing the next generation of **multimodal AI benchmarks** will be hosted next Tuesday.
   - Those interested in evaluation, multimodal models, or functional intelligence are encouraged to [join the talk](https://luma.com/kwg2qg4d).
- **X Blog Post on Missing LLM Capabilities Shared**: A member shared a [blog post](https://x.com/ShashwatGoel7/status/1991611877667840181?t=QKZdUdtbigMMfQSHtrczew&s=19) discussing what's still missing in **LLM capabilities** and conjectures on the general (post)training environment to improve them.
   - The post is on *X*.


  

---


### **HuggingFace ▷ #[core-announcements](https://discord.com/channels/879548962464493619/1014557141132132392/1441319094698901627)** (1 messages): 

> `Diffusers MVP program, Open critical issues on Diffusers` 


- **Diffusers MVP Program Kickstarts Contributions**: Shortly after the announcement of **The Diffusers MVP program**, there were nice contributions and engagements from the community.
   - Members are encouraged to consider the open critical issues and to check out the details [here](https://github.com/huggingface/diffusers/issues/12635).
- **Diffusers: Community Engagement Soars After MVP Launch**: Following the unveiling of **The Diffusers MVP program**, the community responded with notable contributions and active participation.
   - Contributors are urged to explore unresolved critical issues, accessible [on GitHub](https://github.com/huggingface/diffusers/issues/12635), to further enhance the project's development.


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1441450400141021206)** (16 messages🔥): 

> `Logo Detection, LayoutLMv3, Dinov2 for Logo Detection` 


- **Classic CV or Dinov2 can enable Logo Detection**: Members discussed leveraging **CNN or ViT**, suggesting a tiny pretrained model like **dinov2** for quickly fine-tuning logo detection given a limited, low-variance logo set.
   - However, this approach faces challenges due to high variation with around **50 payers**, each having at least **4 document types**, and data varying across time and payer.
- **LayoutLMv3 being tested**: In response to the problems raised, one member is trying **LayoutLMv3** as it understands both text and visuals.
   - It was mentioned that even classical computer vision methods could be used, as methods for locating text in pictures are pretty good now.


  

---


### **HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/)** (1 messages): 

abidlabs: LIVE NOW! https://www.youtube.com/watch?v=ohYBeIQmFa4 <@&1014548769355862036>
  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1441170698298200074)** (3 messages): 

> `Smol Course Sign-Ups, Circular Link Reference in smol-course, Training Issues on smoltalk2_everyday_convs_think, Pending Reviews on Leaderboard Discussions` 


- ****Smol Course** Sign-Ups Status?**: A user inquired whether **smol-course** sign-ups are closed.
- **Circular Link Troubles **smol-course****: A user reported a circular link reference between [https://huggingface.co/smol-course](https://huggingface.co/smol-course) and [https://huggingface.co/learn/smol-course/unit0/1](https://huggingface.co/learn/smol-course/unit0/1).
- **Training Troubles with **smoltalk2_everyday_convs_think****: A user faced issues using the exact code sample provided on [https://huggingface.co/learn/smol-course/en/unit1/3](https://huggingface.co/learn/smol-course/en/unit1/3) when training on **HuggingFaceTB/smoltalk2_everyday_convs_think**.
- **Leaderboard Discussions Awaiting Review**: Multiple submissions on the [leaderboard](https://huggingface.co/spaces/smol-course/leaderboard) discussions, specifically those at or after discussion **#36**, are awaiting review.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1441111346627215430)** (54 messages🔥): 

> `Academic Dishonesty, Infographic Overload, Nano Banana Model, Paper Posting Limits, AI Paper Filtering` 


- ****Teacher Tries To Cheat Using AI****: A user asked for help *boosting 20% of their work*, only to reveal themselves as a **university teacher**.
   - Other users immediately called this out as **academic dishonesty**, with one saying *there's no respect left* for such requests.
- ****Gemini 3 Pro's Nano Banana Inflates Infographics****: Members discussed how the new **Nano Banana model** in **Gemini 3 Pro** makes it so easy to create infographics that there will be a flood of them soon, referencing [tweet](https://x.com/emollick/status/1991527285267275854?s=46).
   - It was noted that *the days of incoherent diagrams and malformed text in AI generated images are a bygone era* but one user pointed out that *the generated clock does not read 10:15*.
- ****Discord Debates Paper Posting Limits****: There was discussion on the number of papers one user was posting a day and whether it was disruptive, with many wanting a **1 paper per day limit**.
   - The user countered that with *There's no way you actually read 20 papers a day* and the others replied *We are not here to do the filtering you are too lazy to do yourself*.
- ****AI Assisted Paper Filtering****: Due to limits on paper postings, one user considered using an **AI**, such as <#1435893010205380619>, to filter papers.
   - After pushing back from the others on this idea, the user stated *i just assigned it to antigravity ide (windsurf fork by google)discord bot coming soon*.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1441123619081162803)** (4 messages): 

> `AI Zar David Sacks, OLMo 3, YouTube video` 


- **AI Zar David Sacks Blasted!**: A member claimed that *100% comes from the AI Zar David Sacks because that moron Charmath is in his ear all the time* and joked about naming Elmo's cousin that lives on 4Chan.
   - The comment included a [link](https://fxtwitter.com/natolambert/status/1991508141687861479) to a related Twitter post.
- **OLMo 3 is out now!**: A member shared links to **OLMo 3**, including the [Allen Institute for AI blog post](https://allenai.org/blog/olmo3) and a [technical report](https://www.datocms-assets.com/64837/1763662397-1763646865-olmo_3_technical_report-1.pdf).
- **YouTube video shared**: A member simply shared a [YouTube video](https://youtu.be/F1pBIjQblI0) with no additional context.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1441110920997896363)** (46 messages🔥): 

> `nano banana pro, early grok image model, Infographics AI-slop, Gemini 3 Pro, Adobe buys Semrush` 


- **Nano Banana Pro is born!**: Members share [images generated by **Nano Banana Pro**](https://cdn.discordapp.com/attachments/1149866623109439599/1441110920339132447/image.png) and discuss that *the hair has this textured look, like the early grok image model*.
   - They noted that it might be due to an issue with **patches**.
- **Infographics will be equalled with AI-slop**: A member shared a [link to infographics](https://x.com/scaling01/status/1991523932336464333) and said they're the *best infographics* they've seen *from a model as far as text, lack of mistakes and thoughtful layout*.
   - Another member predicted **Infographics** will be equalled with **AI-slop in 2026**.
- **Gemini 3 Pro Requires Pro Account**: Members discussed about using **Gemini 3 Pro** and that *you need a pro account*.
   - Some members got a year for free with their phone, some others weren't convinced it's worth paying for the subscription.
- **Adobe Snaps Up Semrush**: A member shared a [TechCrunch article](https://techcrunch.com/2025/11/19/adobe-to-buy-semrush-for-1-9-billion/) reporting **Adobe's acquisition of Semrush for $1.9 billion**.
   - No additional details were discussed.
- **China OS models run up to parity to Deepmind Gemini 3**: A member is *just chillin and waiting for them dang **China OS models** to run up to parity to Deepmind **Gemini 3** ( Gemma/Bannana Pro) for free*.
   - They added *When the Chinese enters the room, than profit goes out the window*.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1441184809606185144)** (3 messages): 

> `ArXiv Endorsement, Discord Server Link` 


- **ArXiv Endorsement Request**: A member requested help with an ArXiv endorsement, mentioning they've emailed around **20 research teams** without success and shared an [ArXiv endorsement link](https://arxiv.org/auth/endorse?x=63SW7W).
   - Another member asked the original poster to resend the endorsement link as they missed it previously.
- **EleutherAI Discord Server Shared**: A member shared a link to the [EleutherAI Discord server](https://discord.gg/eleutherai).
   - No further context was provided regarding the reason for sharing the link.


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1441512329765453824)** (1 messages): 

> `LLM Capabilities, Post Training Environments, Missing capabilities in LLMs` 


- **Blogpost analyzes missing LLM capabilities**: A member shared a [blog post](https://x.com/ShashwatGoel7/status/1991611877667840181?t=QKZdUdtbigMMfQSHtrczew&s=19) that analyzes what's still missing in **LLM capabilities**.
   - The post conjectures on the general **post-training environment** that will get us to that goal.
- **Missing Capabilities addendum**: The shared blogpost discussed missing elements that, when solved, will create huge improvements in LLM capabilities.
   - The blog focuses on areas of improvement for LLMs.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1441184809606185144)** (3 messages): 

> `ArXiv Endorsement, EleutherAI Discord Invite` 


- **ArXiv Endorsement Seeker**: A member requested assistance with an ArXiv endorsement after emailing approximately **20 research teams** without success, and shared [their endorsement link](https://arxiv.org/auth/endorse?x=63SW7W).
   - They asked for the link to be resent as they had missed it previously.
- **EleutherAI Discord Shared**: A member shared an invite link to the **EleutherAI Discord**: [https://discord.gg/eleutherai](https://discord.gg/eleutherai).


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1441322574549356655)** (2 messages): 

> `Mamouth cloud solution, Mamouth private deployment, Mamouth costs` 


- **Mamouth Deployment: Cloud vs. Private?**: A member inquired whether **Mamouth** is exclusively a cloud solution or if it supports private deployments.
   - Another member suggested contacting [Modular directly](https://www.modular.com/request-demo) for details on deployment options and associated costs.
- **Contact Modular for Pricing Details**: A user asked about the cost of **Mamouth** and whether it can be deployed privately.
   - A different user replied that the first user should contact Modular via their [request demo page](https://www.modular.com/request-demo) to inquire about pricing.


  

---


### **Modular (Mojo 🔥) ▷ #[announcements](https://discord.com/channels/1087530497313357884/1098765954302873621/1441199888581787668)** (1 messages): 

> `MAX Python API, NVIDIA Grace Support, Mojo GPU Programming` 


- **Modular Platform 25.7 Released!**: Modular Platform **25.7** introduces a fully open **MAX Python API**, next-gen modeling API, expanded **NVIDIA Grace** support, and safer, faster **Mojo GPU** programming.
   - This update aims to help developers focus on AI advancements rather than infrastructure, according to the [Modular blog post](https://www.modular.com/blog/modular-25-7-faster-inference-safer-gpu-programming-and-a-more-unified-developer-experience).
- **Faster Inference with Safer GPU Programming**: The new release emphasizes improvements in inference speed and enhanced safety in GPU programming using Mojo.
   - Developers can now leverage a more unified experience while programming for AI, reducing time spent on underlying infrastructure issues.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1441137649954979952)** (16 messages🔥): 

> `Mojo release 25.7, Optional syntax in Mojo, UnsafePointer generics, Copy-page content buttons on docs` 


- **Mojo Drops Fresh Release: 25.7**: The **Modular team** released version **25.7**, boasting faster inference, safer GPU programming, and a more unified developer experience; details can be found in the [Modular Blog](https://www.modular.com/blog/modular-25-7-faster-inference-safer-gpu-programming-and-a-more-unified-developer-experience).
   - Enthusiasts are eager for AI-native capabilities to become a reality.
- **Optional Syntax Speculation Sparks Swiftie Yearning**: There is discussion on the possibility of introducing `Foo?` syntax for optionals in Mojo, similar to Swift, and optional chaining (`foo?.bar()`), but the team isn't prioritizing syntactic sugar, but one member thinks that optional chaining and `T?` syntax are *"obvious"*.
   - A member suggested using `SIMD[.float, 4]` for contextual inference of static members, drawing parallels to Swift, and another member shares a [Rust RFC](https://rust-lang.github.io/rfcs/3058-try-trait-v2.html) for inspiration.
- **UnsafePointer Generics Get Obligatory**: **Generics** (mojo parameters) for **unsafe pointer** (mut, type, origin, etc...) are no longer defaulted.
   - More information is available in the [proposal document](https://github.com/modular/modular/blob/main/mojo/proposals/unsafe-pointer-v2.md#migration-guide-from-legacyunsafepointer-to-the-new-unsafepointer), including a migration guide from LegacyUnsafePointer to the new UnsafePointer.
- **Auto Copy-Paste buttons incoming?**: A user suggested that Mojo add automatic copy-page content buttons on the top right of docs *"that we use to ask for faster intuition learning or implementations with LLMs."*
   - The suggestion aimed to match the direction of other documentations nowadays to accommodate *"genz+ developers that only learn how stuff works after breaking it with vibe coding"*.


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1441159104964857856)** (11 messages🔥): 

> `MAX, AMX, NDBuffer, LayoutTensor, CPU inference` 


- **MAX Opens Up!**: Members shared excitement about **MAX** opening up, with one member inquiring whether **AMX** is out of scope for **MAX** following a [specific commit](https://github.com/modular/modular/commit/e0d81b694b4eab18d22f3a12d3b966e03e055b18).
- **AMX Deprecation Rationale**: A member stated that **AMX** wasn’t being used anywhere, and its integration into the current **tensor core framework** would be hard, especially since **Intel** and **AMD** are announcing a replacement soon.
   - They added that re-adding a framework to use **AMX** would be fine if there's a need to bring it up in **MAX**, but it gets into issues like *bespoke tensor parallelism* and *expert parallelism*.
- **NDBuffer's Sunset, LayoutTensor Sunrise**: The removal of specific code was part of deprecating **NDBuffer** and moving all uses to **LayoutTensor**.
   - Since there aren't many use cases for **CPU inference** with customers, it might not be added back, but contributions to create a **LayoutTensor** based version are welcome.
- **LayoutTensor Version Already Exists!**: A member has already written most of a **layout tensor** version, with a rough draft available on [Gist](https://gist.github.com/CoffeeVampir3/d82917f6fce60c0c2cdf00629c4de67d).
   - The member is willing to generalize and flesh it out if it's considered valuable by others.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1441211594309898361)** (22 messages🔥): 

> `Operator extension installation issues, Aurora Seeker, Personal data processing for insights, Manus Knowledge entry examples, Manus referral/redeem codes` 


- ****Operator Extension** Keeps Asking to Install**: A user reported that the **Operator extension** in Chrome keeps asking to install itself repeatedly, even when directed to use an open Amazon tab.
   - The user mentioned that the alternative is to use **Aurora Seeker**.
- **Crafting Insightful Personal Data Repos**: A user is seeking input on building tools for storing and processing personal data to derive insights, linking to their previous projects: [contextflow](https://share.cleanshot.com/StvTll4j), [oncue](https://www.youtube.com/watch?v=4UaQEB1b84E&feature=youtu.be), and [axon](https://www.linkedin.com/posts/harrison-qian-95b0062a3_won-the-trae-solo-hackathon-sf-tech-week-activity-7383626911199141890-8SgL?utm_source=share&utm_medium=member_desktop&rcm=ACoAAEkmXdcBT7GJOg4Kg0Iy89EqxavBMIqIxk4)).
- ****Manus Knowledge** Entry Inspiration**: A user requested good examples for **Manus Knowledge** entries, and is currently using commands like *Always do this* or *Never do that*.
- **Plea for **Manus** Credit**: A user inquired about redeem codes for **Manus**, mentioning they cannot afford to refill credits due to business constraints.
   - Another user shared their [Perplexity Pro referral link](https://plex.it/referrals/VCETA5M7) and another pointed out [Perplexity offer 1 year pro for university students](https://plex.it/referrals/VCETA5M7).
- ****Manus vs Gemini 3** test incoming**: A user announced they will test **Manus** against **Gemini 3** to determine which is the better agent.


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1441134069483176079)** (14 messages🔥): 

> `Kimi K2 location, GPT-5.1 distilled on K2?, Kimi's attention weak?, Open Source models behind?, K2t vs Gemini 2.5 pro` 


- **Kimi is hosted in the US**: A user mentioned that **Kimi** tells them it's hosted in the **US** because that's where they live, attaching an [image](https://cdn.discordapp.com/attachments/1371757564005711973/1441250576124870656/image.png?ex=6921c572&is=692073f2&hm=a285f73eb8a311c6828f6c0d0c8952a9b923f30dedfb12ec2608c64cb&) as evidence.
- **GPT-5.1 has been distilled on K2**: A user presented what they called *irrefutable evidence* that **GPT-5.1** has been distilled on **K2**, attaching a [related image](https://cdn.discordapp.com/attachments/1371757564005711973/1441296525904052315/IMG_6579.png?ex=6921f03d&is=69209ebd&hm=2c5c12a19efdb6c54624ef43b952c7d30fd8b99d4cd7125465df5e28d1c30afc&).
- **Kimi's attention weak in Complex tasks**: A user commented that **Kimi's** effective attention is too weak, and doesn't seem to work well in complex tasks involving long contexts.
- **Open Source models 9 months behind?**: A user shared that open source AI models are **9 months** behind proprietary models according to the most recent evals by a highly impartial and totally not biased institute, linking to a [tweet](https://x.com/scaling01/status/1991665386513748172?s=46).
- **K2t is better than Gemini 2.5 pro**: A user stated that **K2t** is most certainly much better than **Gemini 2.5 pro**, and very much in the league of **Sonnet 4.5**.


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1441117319521632357)** (1 messages): 

> `Gem3pro, Proxy Server, DSPy Proxy` 


- **Gem3pro Builds Proxy Server in One-Shot**: A user prompted **Gem3pro** to build a proxy server based on [this tweet](https://x.com/skylar_b_payne/status/1990808733140779488) and it succeeded on the first try.
   - The user shared a [GitHub repo](https://github.com/aryaminus/dspy-proxy) containing the **DSPy proxy** that was created.
- **DSPy Proxy GitHub Repository**: The **DSPy proxy** mentioned above can be found in [this GitHub repository](https://github.com/aryaminus/dspy-proxy).
   - This repository was created following the successful one-shot generation of the proxy server by **Gem3pro**.


  

---


### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/1441369788214546534)** (3 messages): 

> `Agent DAG Generation, Think Workflow Compile/Validate DAG, DSPy Prod/Inference/Run Time` 


- **Agents Generate Task DAGs for Performance**: A member thinks the core idea of *asking the agent to generate a DAG of tasks* is a pretty good one, because you can then use **RL** to try and get better performance there.
   - They wondered if it could be adapted into a **Think -> Workflow -> Compile/Validate DAG -> Execute workflow**.
- **DSPy: Adapting DAGs for Production**: A member questioned how the agent DAG idea can be used with **DSPy** if it can be used with it or anything else in prod/inference/run time.
   - They don't think any of those have to be specifically trained, focusing on adaptation for **DSPy**.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1441152073503740075)** (3 messages): 

> `AI Engineer Introduction, GEPA Assistance Request` 


- **Seasoned AI Engineer Stands Ready**: A senior full stack and AI engineer with **10 years** of experience introduced themself, highlighting their expertise in building smart, scalable AI systems using technologies like **LangChain**, **LangGraph**, and **Next.js**.
   - Their background includes projects in task automation, voice chat, CRM integration, and the development of AI-driven SaaS applications, expressing interest in collaborating on AI-related projects involving lead generation, support, or custom apps.
- **GEPA Guru Grabbed, Guidance Granted?**: A member requested assistance with **GEPA** within a specific channel.
   - The member directly tagged the moderators as job spam.


  

---


### **MCP Contributors (Official) ▷ #[general](https://discord.com/channels/1358869848138059966/1358869848138059969/1441116538626244728)** (6 messages): 

> `DNS Migration, Domain Governance, Downtime Risks` 


- ****DNS Domain Transition** to Community Control**: The **modelcontextprotocol.io domain** is migrating from the **Anthropic corporate DNS** account to the community's control to improve governance and enable faster DNS setups.
   - The transition aims to facilitate projects requiring DNS setups and enable **Infrastructure as Code (IaaC)** over the project's domains.
- **Migration may interrupt services**: Although precautions are being taken to prevent downtime, there is a risk of disruptions during the **DNS migration** planned for the next week.
   - Engineers recommend to keep an eye on possible odd behaviours, and warned that the manual process might cause potential issues.
- **Community Asks To Avoid Website Outage on Birthday**: A community member suggested timing the **DNS migration** to avoid **MCP's birthday** on the **25th** to prevent site downtime.
   - In response, engineers will try to schedule the migration _after_ the birthday.


  

---


### **MCP Contributors (Official) ▷ #[general-wg](https://discord.com/channels/1358869848138059966/1416012674663452752/1441437437627728054)** (1 messages): 

> `Tool Annotations, Model Context Protocol` 


- **Proposed Tool Annotations Solution**: A member proposed a solution to **Tool Annotations** for Tools that would ideally have different annotations based on their arguments.
   - They are seeking sponsorship for this idea and seeking suggestions for a WG or IG suited to this topic and a related [pull request](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/1862).
- **Seeking Sponsorship for Model Context Protocol Tool Annotations**: A contributor is actively seeking sponsorship for their proposed solution regarding **Tool Annotations** within the **Model Context Protocol**.
   - They are also requesting guidance on identifying a suitable Working Group (**WG**) or Interest Group (**IG**) to further explore and develop this topic.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1441471859999772702)** (1 messages): 

> `AI Voice Agents, Feather AI` 


- **Engineers Build Real-Time AI Voice Agents**: Engineers are diving deep into building **AI voice agents**, focusing on challenges like managing **latency**, ensuring **smooth call handoffs**, and maintaining clean event control during live conversations.
   - They are seeking insights into how others handle real-time call flows and structured conversation summaries.
- **Experiments with Feather AI Yields Promising Results**: A member shared that they are experimenting with [Feather AI](https://www.featherhq.com/), reporting sub-second latency and stable agent logic even when users go off-script.
   - They also cited **clean transcription**, structured event streams, and reliable pushing of results into **CRMs** as benefits, and are open to hearing about alternative architectures, tools, or workflows.


  