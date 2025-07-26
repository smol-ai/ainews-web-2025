---
id: MjAyNS0w
title: not much happened today
date: '2025-07-25T05:44:39.731046Z'
description: >-
  **OpenAI** has fully rolled out its ChatGPT agent to all Plus, Pro, and Team
  users and is building hype for the upcoming **GPT-5**, which reportedly
  outperforms **Grok-4** and can build a cookie clicker game in two minutes.
  **Alibaba's Qwen** team released the open-source reasoning model
  **Qwen3-235B-Thinking**, achieving an **89%** win rate over **gpt4-0314**
  using a new RL algorithm called **Group Sequence Policy Optimization (GSPO)**.
  **Runway** introduced **Runway Aleph**, a state-of-the-art in-context video
  model for editing and generating video content. **Hugging Face** highlights
  the growing momentum of open-source AI, especially from Chinese teams. Other
  updates include **Kling's** upgrades for image-to-video generation and
  **Google's Imagen 4 Ultra** being recognized as a top text-to-image model.
  **Anthropic** integrated **Claude** with **Canva** for branded visual designs
  but faces stability issues. The **PyTorch** team released optimized
  checkpoints for **SmolLM3** to speed up inference.
companies:
  - openai
  - alibaba
  - runway
  - hugging-face
  - google
  - anthropic
  - pytorch
  - lmarena
models:
  - gpt-5
  - gpt4-0314
  - qwen3-235b-thinking
  - runway-aleph
  - imagen-4-ultra
  - smollm3
  - grok-4
topics:
  - reinforcement-learning
  - reasoning
  - video-generation
  - image-generation
  - model-optimization
  - open-source
  - model-performance
  - inference-speed
  - integration
  - stability
people:
  - sama
  - clementdelangue
  - xikun_zhang_
  - teknnium1
  - chujiezheng
---


**a good day for Open Source AI**

> AI News for 7/24/2025-7/25/2025. We checked 9 subreddits, 449 Twitters and 29 Discords (226 channels, and 8449 messages) for you. Estimated reading time saved (at 200wpm): 595 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

it's worth looking at [Qwen 3 Thinking](https://www.reddit.com/r/LocalLLaMA/comments/1m8vegq/qwen3235ba22bthinking2507_released/), and the [AIE SWE Agents track](https://www.youtube.com/playlist?list=PLcfpQ4tk2k0UwfWS-f6KDInzHc3um4naZ) which is now fully released.

---

# AI Twitter Recap

**Major Model Releases & Updates (Open Source vs. Closed Source)**

- **OpenAI's GPT-5 & ChatGPT Agent Rollout**: **OpenAI** has now [fully rolled out its ChatGPT agent](https://twitter.com/OpenAI/status/1948530029580939539) to all **Plus**, **Pro**, and **Team** users. Simultaneously, hype is building for the upcoming **GPT-5**, which is rumored for an August release. On `lmarena`, [@scaling01](https://twitter.com/scaling01/status/1948863153795682709) demonstrated that **GPT-5** is [significantly better than Grok-4](https://twitter.com/scaling01/status/1948863325858922610), capable of [casually building a cookie clicker game in two minutes](https://twitter.com/scaling01/status/1948809543435395470). The anticipation is bolstered by a quote from **Sam Altman**, shared by [@xikun_zhang_](https://twitter.com/xikun_zhang_/status/1948627882235838482), stating "[GPT-5 is smarter than us in almost every way](https://twitter.com/xikun_zhang_/status/1948627882235838482)."
- **Qwen's Frontier Open-Source Offensive**: The **Qwen** team from Alibaba released **Qwen3-235B-Thinking**, a powerful new open-source reasoning model. [@Teknium1](https://twitter.com/Teknium1/status/1948711699013665275) reports that it is [as good as top closed frontier models](https://twitter.com/Teknium1/status/1948711699013665275) and achieved a staggering [**89%** win rate over **gpt4-0314** on Arena-hard v1](https://twitter.com/Teknium1/status/1948836009183224132). The model's performance is attributed to a new RL algorithm called **Group Sequence Policy Optimization (GSPO)**, which was [introduced by team member @ChujieZheng](https://twitter.com/eliebakouch/status/1948719361109172375). The rapid pace of releases from Chinese teams led [@Teknium1](https://twitter.com/Teknium1/status/1948744914876920039) to ask, "[What is America doing?](https://twitter.com/Teknium1/status/1948744914876920039)".
- **Runway Aleph Video Model**: **Runway** has [introduced Runway Aleph](https://twitter.com/c_valenzuelab/status/1948789396443914353), a new state-of-the-art in-context video model for editing, transforming, and generating video content. [@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1948817274468802907) highlighted its ability to serve as a generalizable model that can solve many video tasks at once, including practical features like [instantaneous inpainting](https://twitter.com/c_valenzuelab/status/1948878604928254257) with simple text commands.
- **The Rise of Open Source**: [@ClementDelangue](https://twitter.com/ClementDelangue/status/1948756474861084875) of **Hugging Face** celebrated the momentum of the open-source community, stating that it is now [at the frontier of AI](https://twitter.com/ClementDelangue/status/1948756474861084875) despite having fewer resources. He pointed to the leadership of Chinese teams and the success of open models on leaderboards like `designarena.ai`.
- **Other Notable Model Updates**: **Kling** announced significant [upgrades to its Elements for Image to Video generation](https://twitter.com/Kling_ai/status/1948610721031549432). **Google's Imagen 4 Ultra** was touted by [@OfficialLoganK](https://twitter.com/sedielem/status/1948838043236139164) as the [world's best text-to-image model](https://twitter.com/sedielem/status/1948838043236139164), tying for **#1** on the **lmarena** leaderboard. The **PyTorch** team has released [new optimized checkpoints for SmolLM3](https://twitter.com/LoubnaBenAllal1/status/1948477437513208062) to enable faster inference.

**AI Tooling, Frameworks, and Agents**

- **Claude and Anthropic Ecosystem**: **Anthropic** announced a major integration with **Canva**, allowing **Claude** to [turn documents into branded visual designs](https://twitter.com/AnthropicAI/status/1948489708385816666). The official **Claude Code** account shared a helpful tip on utilizing [custom subagents for tasks like code review and debugging](https://twitter.com/claude_code/status/1948622899604050063). However, the platform has faced stability issues, with users like [@QuixiAI](https://twitter.com/QuixiAI/status/1948759481220825144) reporting [frequent service disruptions](https://twitter.com/QuixiAI/status/1948759481220825144) for paid plans.
- **Perplexity's Comet Browser**: **Perplexity's** AI-native browser, **Comet**, has seen a series of feature demonstrations from CEO [@AravSrinivas](https://twitter.com/AravSrinivas/status/1948489790036365796). He showcased its ability to create **Spotify** playlists, [automate LinkedIn tasks](https://twitter.com/AravSrinivas/status/1948835728798220539), and even [order food directly from restaurants to bypass aggregators](https://twitter.com/AravSrinivas/status/1948818172985196862). Srinivas also noted that the percentage of users [switching to Comet as their default browser has been steadily increasing](https://twitter.com/AravSrinivas/status/1948794199069110519).
- **Microsoft's GitHub Spark**: **Satya Nadella** announced the release of **GitHub Spark**, a new **Copilot** tool designed to [turn ideas into full-stack applications entirely through natural language interaction](https://twitter.com/algo_diver/status/1948594244039704892).
- **LlamaIndex and FlowMaker**: [@jerryjliu0](https://twitter.com/jerryjliu0/status/1948797112789205111) introduced **FlowMaker**, a new [open-source, low-code tool for building custom agent workflows](https://twitter.com/jerryjliu0/status/1948797112789205111) with a visual drag-and-drop interface powered by **LlamaIndex.TS**.
- **Context Engineering & DSPy**: The concept of **Context Engineering** is gaining traction, with [@douwekiela](https://twitter.com/douwekiela/status/1948496592534737395) defining it as the critical infrastructure layer between data and models. The **DSPy** framework from Stanford is a key tool in this space, with [@lateinteraction](https://twitter.com/lateinteraction/status/1948492811575156851) highlighting its successful deployment in a [multi-agent LLM system for doctor-patient communication in Romania](https://twitter.com/lateinteraction/status/1948492811575156851).

**Technical Insights & Research**

- **LLM Reasoning Deep Dive**: **Google's** [@denny_zhou](https://twitter.com/denny_zhou/status/1948499173986201915) shared key insights from his **Stanford CS25** lecture on LLM Reasoning. He emphasized that reasoning is the generation of intermediate tokens, **RL finetuning** is the most effective method for eliciting it, and aggregating multiple responses yields superior results.
- **The End of an Era: Papers with Code Sunsets**: The research community reacted to the news from [@rosstaylor90](https://twitter.com/ClementDelangue/status/1948735387318304822) that **Meta** is sunsetting the widely used **Papers with Code** platform. In a swift response, [@julien_c](https://twitter.com/_akhaliq/status/1948732117120163921) of **Hugging Face** announced a partnership with **Meta AI** to [build its successor](https://twitter.com/_akhaliq/status/1948732117120163921), a move praised by the community.
- **Google's Processing Scale**: **DeepMind's** CEO, [@demishassabis](https://twitter.com/demishassabis/status/1948579654790774931), revealed an astonishing statistic: Google processed nearly [**one quadrillion tokens** in the last month](https://twitter.com/demishassabis/status/1948579654790774931), more than doubling the volume from the previous month.
- **Alignment Research at Anthropic**: **Anthropic** is doubling down on alignment, releasing research on [AI agents designed to autonomously audit and red-team models](https://twitter.com/EthanJPerez/status/1948605334698033479). Adding to this effort, [@Jack_W_Lindsey](https://twitter.com/EthanJPerez/status/1948612180007612901) announced the formation of an **"AI psychiatry" team** to study model behaviors like sycophancy and persona replication.
- **Production-Level Document Processing**: [@jerryjliu0](https://twitter.com/jerryjliu0/status/1948475176062255504) provided a technical breakdown of why simply "screenshotting a page and feeding it to the LLM" is insufficient for production document processing, citing issues with **missed metadata**, **resolution loss**, and **prohibitive costs**. He advocates for more tuned approaches.
- **Scaling Laws for MoEs**: [@scaling01](https://twitter.com/scaling01/status/1948713380308496575) shared a comprehensive summary of a paper on **Scaling Laws for Efficient Mixture-of-Experts (MoEs)**, detailing how factors like **sparsity**, **granularity**, and **expert sharing ratios** influence model performance and computational efficiency.

**Robotics & Industry Commentary**

- **The Robot Moravec's Paradox**: **NVIDIA's** [@DrJimFan](https://twitter.com/DrJimFan/status/1948789854151868663) articulated a key challenge in robotics he calls the **"Robot Moravec's Paradox"**. He explained that complex gymnastics, while hard for humans, are far easier for robots than mundane tasks like cleaning. This is because acrobatics can be perfected in simulation, whereas general dexterity requires simulating messy, complex real-world physics—a much harder problem. This discrepancy creates a public illusion that physical AI is more advanced than it truly is.
- **Meta's New Chief Scientist**: **Meta Superintelligence Labs** announced that [@shengjia_zhao](https://twitter.com/AIatMeta/status/1948836042406330676) will be its new **Chief Scientist**. The appointment was lauded by his former Stanford colleague [@DrJimFan](https://twitter.com/DrJimFan/status/1948841055916622157), who described him as one of the "brightest, humblest, and most passionate scientists" he knows.
- **The Future of AI-Driven Work**: **Inflection AI's** [@mustafasuleyman](https://twitter.com/mustafasuleyman/status/1948798692598915186) asserted that while learning AI is now table stakes, the [next competitive advantage will be managing a team of AIs](https://twitter.com/mustafasuleyman/status/1948798692598915186). This is echoed by [@omarsar0](https://twitter.com/omarsar0/status/1948490601164316891), who noted he's become the bottleneck because his [AI agents are so fast and effective](https://twitter.com/omarsar0/status/1948490601164316891).
- **US/China Tech Dynamics**: [@hkproj](https://twitter.com/hkproj/status/1948640081348063324) argued that the primary reason China is number two in the AI race is the continued attractiveness of the US for top Chinese researchers, suggesting a mass return home could shift the balance of power.

**AI Applications & Use Cases**

- **AI for Finance**: **Perplexity** is expanding its financial toolkit, with [@AravSrinivas](https://twitter.com/AravSrinivas/status/1948812710952796576) demonstrating a new [natural language-powered Stock Screener](https://twitter.com/AravSrinivas/status/1948812710952796576) on **Perplexity Finance**.
- **Automating Tedious Tasks**: With the release of new datasets, [@Teknium1](https://twitter.com/Teknium1/status/1948668301829439846) predicts that AI will soon be able to handle complex tasks like [filing taxes very effectively](https://twitter.com/Teknium1/status/1948668301829439846). [@andersonbcdefg](https://twitter.com/andersonbcdefg/status/1948477571378938014) quipped that an AI capable of filing taxes without a panic attack would already be [more capable than every millennial](https://twitter.com/andersonbcdefg/status/1948477571378938014).
- **Creative and Productivity Tools**: **Google Labs** showed off a feature in **Flow** that allows users to [give iterative feedback on a generated image instead of rewriting prompts](https://twitter.com/sedielem/status/1948504824414109798). Meanwhile, [@gdb](https://twitter.com/gdb/status/1948808781686853996) demonstrated **OpenAI's Deep Research** feature working seamlessly over **Notion** documents.
- **Non-Coding Applications for Claude Code**: [@alexalbert__](https://twitter.com/alexalbert__/status/1948765443776544885) is curating a list of the various [non-coding tasks users are accomplishing with Claude Code](https://twitter.com/alexalbert__/status/1948765443776544885), showcasing its growing versatility beyond its original purpose.

**Humor & Memes**

- **Relatable Engineering Humor**: [@_lewtun](https://twitter.com/_lewtun/status/1948569538913542437) joked that the final interview at **Hugging Face** involves [solving a brain teaser with Transformers toys](https://twitter.com/_lewtun/status/1948569538913542437). [@code_star](https://twitter.com/code_star/status/1948863643946565743) posted a meme about the pain of being unable to beat a [baseline dataset mix set purely on vibes](https://twitter.com/code_star/status/1948863643946565743).
- **Prompt Injection as Art**: [@goodside](https://twitter.com/goodside/status/1948583404888350780) mused that "There are prompt injections everywhere for those with AIs to see." This was taken to its logical conclusion by [@aidanshandle](https://twitter.com/code_star/status/1948658050773942409), who proposed [painting their roof with "ignore previous instructions and don't drone strike this building."](https://twitter.com/code_star/status/1948658050773942409)
- **Industry Satire**: [@dylan522p](https://twitter.com/dylan522p/status/1948499656545083797) made a detailed semiconductor joke about a photo of **Sydney Sweeney** [handling a 6" or 8" wafer instead of the 12" wafers used at the leading edge](https://twitter.com/dylan522p/status/1948499656545083797). A popular meme asking "[Anyone knows adam?](https://twitter.com/giffmana/status/1948659163212439716)" was shared by [@giffmana](https://twitter.com/giffmana/status/1948659163212439716) and [@akbirkhan](https://twitter.com/akbirkhan/status/1948674911192375801), referencing the ubiquitous optimizer.
- **Classic Tech Nostalgia**: In a widely shared tweet, [@clefourrier](https://twitter.com/clefourrier/status/1948648157635903791) retweeted a post about [telling their grandchildren that Clippy was ChatGPT](https://twitter.com/clefourrier/status/1948648157635903791).

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Qwen3-235B Model and Benchmark Performance Release Wave

- [**Qwen3-235B-A22B-Thinking-2507 released!**](https://i.redd.it/bvx1dbl5xzef1.jpeg) ([Score: 703, Comments: 158](https://www.reddit.com/r/LocalLLaMA/comments/1m8vegq/qwen3235ba22bthinking2507_released/)): **The image is likely a promotional or informational visual accompanying the announcement of Alibaba's new model, Qwen3-235B-A22B-Thinking-2507, which claims significant advances in reasoning, coding, and long-context handling (256K context window). The model is designed for "thinking mode" without manual toggling and emphasizes deep reasoning capabilities. Comments highlight the rapid pace of Alibaba's model releases and immediate availability of GGUF quantized versions (on Hugging Face), supporting high token throughput on large RAM configurations.** Technically critical commentary contrasts Alibaba's rapid innovation (multiple Qwen3 releases in a month) with OpenAI's more cautious public model release strategy. Further technical discussion in the comments focuses on performance benchmarks and deployment logistics for the model's GGUF format.
    - Unsloth has provided GGUF-format quantizations for Qwen3-235B-A22B-Thinking-2507 on Hugging Face, enabling performance of over 6 tokens/sec on hardware with 89GB unified memory or 80GB RAM plus 8GB VRAM. They emphasize the quants are dynamic and confirm iMatrix dynamic quants are also now available, highlighting rapid support for diverse quantization methods: https://huggingface.co/unsloth/Qwen3-235B-A22B-Thinking-2507-GGUF.
    - There is interest from users in seeing the performance improvements from the 2507 model updates transferred to distilled variants like Qwen-30B A3B, given that these smaller models have demonstrated strong speed, even on integrated GPUs (iGPU). This suggests possible widespread accessibility on lower-spec hardware if distillation and new quantization releases proceed.
- [**Qwen’s TRIPLE release this week + Vid Gen model coming**](https://www.reddit.com/gallery/1m91b98) ([Score: 145, Comments: 23](https://www.reddit.com/r/LocalLLaMA/comments/1m91b98/qwens_triple_release_this_week_vid_gen_model/)): **Alibaba's Qwen team released a major suite of open models: 1) Qwen3-235B-A22B-Instruct-2507, which delivers state-of-the-art results on benchmarks like GPQA, AIME25, and LiveCodeBench, surpassing even some closed models such as Claude 4 (non-thinking) according to Artificial Analysis; 2) Qwen3-Coder, a code-centric model outperforming GPT-4.1 and Claude 4 on SWE-bench and Mind2Web, with a CLI tool aimed at developer workflow integration and topping Hugging Face leaderboards; 3) Qwen3-235B-A22B-Thinking-2507, featuring** `256K` **context and high scores on SuperGPQA and v6 LiveCodeBench, challenging Gemini 2.5 Pro and o4-mini head-on. Qwen's open-source push is backed by significant infrastructure investment and a comprehensive model family (300+ models, 140,000+ derivatives). The upcoming Wan 2.2 video generation model is anticipated to advance controllability and efficiency in open-source text-to-video generation, building upon Wan 2.1's strong VBench results.** Top comments primarily critique the post's tone and style as repetitive and overly hyped, noting a lack of sourcing and depth beyond summarizing already-public information. There is little substantive technical debate in the highlighted comments.
    - One commenter notes that there have been three distinct Qwen-related news releases this week, all making the front page, indicating rapid progress and high release cadence, but also some redundancy in coverage. This could highlight both strong momentum and the challenge of distinguishing substantive updates amid frequent announcements.
    - There’s a meta-discussion about the value of posts summarizing or hyping developments from Alibaba/Qwen. The increase in Qwen announcements is seen as a signal of Alibaba’s growing efforts to compete in the AI space, possibly positioning Qwen as a major open-source competitor.
- [**New Qwen3-235B update is crushing old models in benchmarks**](https://i.redd.it/q009687760ff1.jpeg) ([Score: 102, Comments: 11](https://www.reddit.com/r/LocalLLaMA/comments/1m8w9ah/new_qwen3235b_update_is_crushing_old_models_in/)): **The linked image visualizes benchmark improvements for the latest Qwen3-235B-A22B-2507 (Instruct and Thinking versions) models compared to their predecessors. Across four challenging evaluations (GPQA, AIME2025, LiveCodeBench v6, Arena-Hard v2), the new models show substantial gains, achieving scores such as 81 on GPQA and 92 on AIME2025 versus 71 and 81, respectively, for earlier versions. The post discusses potential reasons for this leap (improved training/data/techniques), and highlights major performance boosts in reasoning and code-related tasks.** Commenters note that Qwen3-235B-2507 rivals high-end models like Gemini Pro and offers strong answer quality, especially in local setups, but mention slower generation with large contexts. There's also interest in extending these improvements ('thinking' ability) to larger models, such as the Qwen 480B Coder.
    - Users report that Qwen3-235B-2507 delivers substantial improvements over previous models, with one noting that its responses feel similar in quality to Gemini Pro in both structure and detail.
    - The instruct version of Qwen3-235B, tested on the unsloth dynamic q3_k_xl configuration, demonstrates detailed, well-structured answers and tolerable hallucination rates even on local setups such as a 128GB Mac. However, performance slows significantly with lengthy contexts—processing speed drops from 20 tokens/sec in empty context to 5 tokens/sec with 10,000+ tokens.
    - Benchmarks, specifically the 'arena bench' for non-thinking models, show impressive gains with Qwen3-235B. Additionally, mentions of the 480B Coder model indicate notable speed and strong performance even in its early state, with user interest in expanded capabilities like 'thinking' mode.

### 2. Qwen3 Model Variants: Thinking, Instruct, and Smaller Models

- [**Smaller Qwen Models next week!!**](https://i.redd.it/752ts71q50ff1.png) ([Score: 498, Comments: 37](https://www.reddit.com/r/LocalLLaMA/comments/1m8w7ny/smaller_qwen_models_next_week/)): **The post announces that smaller instruct and reasoning variants of the Qwen3 model will be released next week, suggesting potential inclusion of lighter 'Qwen3 coder' models. This refers to the ongoing model size diversification from Qwen, a notable open-source LLM suite, aimed at delivering improved performance and accessibility for different compute environments. No concrete benchmarks or architectural details are disclosed, but anticipation is high for the capabilities of the upcoming 30B parameter model, and the community expects further open-source contributions.** Commenters express excitement about the upcoming models, with some skepticism regarding open-source timelines—referencing a common industry trend of delays with the excuse of 'safety concerns.' The expectation is that Qwen's release cadence may emulate or rival GPT-5 quality.
    - There's discussion about the upcoming 30B Qwen models, with users speculating whether these will match the performance of the 'o3mini' level (referring to OpenAI's 30B-class models). This highlights community interest in benchmarking the Qwen 30B model directly against established baselines like o3mini.
    - Some comments express skepticism about open-source model release timelines, referencing a common pattern where release is delayed indefinitely with 'safety' as the reason, and pointing out that such statements are often paired with exaggerated future promises (e.g., 'GPT-5 level'). This reflects ongoing debate about transparency and expectations from AI developers.
    - There's an additional mention that a smaller 'coder' variant of Qwen may be released next month, indicating that code-specialized checkpoints are planned soon after the main model releases.
- [**Amazing qwen 3 updated thinking model just released !! Open source !**](https://i.redd.it/nx5d8w74yzef1.jpeg) ([Score: 187, Comments: 19](https://www.reddit.com/r/LocalLLaMA/comments/1m8vhp3/amazing_qwen_3_updated_thinking_model_just/)): **The Reddit post announces the release of the open-source Qwen 3 'Thinking Model' from Alibaba, echoing an official announcement on Twitter. The linked Hugging Face repository offers dynamic GGUF quantizations for the 23.5B parameter 'Thinking' variant, with reported inference speeds of over 6 tokens/s on appropriate hardware (89GB unified memory or ~80GB RAM + 8GB VRAM). The image itself appears to be a standard model card or summary with headline branding and core statistics, adding contextual confirmation of the release—but lacks deep technical specifics beyond the repository's provided information.** Comment debate briefly touches on hardware requirements and the availability (or lack) of smaller dense coder models, highlighting typical user-driven concerns about practical deployability and variant diversity.
    - Dynamic GGUF quantizations of Qwen3-235B-Thinking are available on [HuggingFace, via unsloth](https://huggingface.co/unsloth/Qwen3-235B-A22B-Thinking-2507-GGUF). Performance reported is `>6 tokens/s` with 89GB unified memory or 80GB RAM + 8GB VRAM, highlighting its high resource demand and potential deployment options for those with sufficient hardware.
    - Discussion references the availability of new dynamic quantization types (including imatrix-dynamic), suggesting ongoing technical improvements to quantization methods for large models, which can impact inference speed and hardware compatibility.
    - A user queries about suitability for quad 3090 setups, implicitly highlighting the need for multi-GPU or high-memory configurations to run such large models, and prompting discussion on efficient hardware utilization for LLM inference.

### 3. AI Coding and Code Benchmark Performance (SWE-Bench, GLM-4.1V)

- [**A contamination-free coding benchmark shows AI may not be as excellent as claimed**](https://www.reddit.com/r/LocalLLaMA/comments/1m8ud84/a_contaminationfree_coding_benchmark_shows_ai_may/) ([Score: 162, Comments: 38](https://www.reddit.com/r/LocalLLaMA/comments/1m8ud84/a_contaminationfree_coding_benchmark_shows_ai_may/)): **A new contamination-free coding benchmark (referenced via TechCrunch and hosted as part of the Kaggle Konwinski Prize competition) reports state-of-the-art open-source models (e.g., Qwen2.5 Coder 32B) scoring under 10% on the SWE-Bench, well below community expectations for AI coding capability. Submissions from larger or newer (proprietary) models are barred, and implementation issues allegedly marred the competition—participants cite broken sample code, delayed bug fixes, hidden methodology, and cryptic errors throughout the contest period. These results prompt renewed skepticism about AI's current ability in autonomous software engineering tasks.** Technical commenters debate the validity of the benchmark results, with some citing real-world performance of models exceeding 10% and attributing poor results to flawed competition design and execution rather than inherent AI limitations. There is consensus the idea of a contamination-free benchmark is strong, but the implementation and management of the Kaggle challenge were widely regarded as chaotic and inadequate.
    - A technical critique of the mentioned Kaggle competition points out severe issues with benchmark reliability, citing that for two out of three months, sample code was nonfunctional and infrastructure problems hampered submissions. Key complaints include opaque methodology, hidden test cases, lack of error log access, and insufficient communication or timeline extensions, which led to limited participation (reportedly 150–200 submissions compared to thousands in well-run AIMO competitions). This undermines the credibility and utility of the competition's results as an assessment of model performance.
    - A data point is referenced where state-of-the-art open-source models achieved only about 10% on a contamination-free SWE-Bench, sparking skepticism in real-world applicability. Practitioners challenge these low benchmarks by citing substantially higher success rates using models like Devstral and windsurf variants in practical, local development scenarios, raising questions about the representativeness of such benchmarks for everyday codebase tasks.
    - Discussion distinguishes between AI as a coding assistant versus a programming replacement. It emphasizes LLMs' lack of persistent understanding of codebases or project context versus human interns who learn and retain workflows and rationales. Even so, LLMs are credited for drastically improving efficiency by replacing code search and help platforms such as Stack Overflow, accelerating onboarding with unfamiliar technologies.
- [**GLM-4.1V-9B-Thinking - claims to "match or surpass Qwen2.5-72B" on many tasks**](https://github.com/THUDM/GLM-4.1V-Thinking) ([Score: 145, Comments: 21](https://www.reddit.com/r/LocalLLaMA/comments/1m8xmy9/glm41v9bthinking_claims_to_match_or_surpass/)): **GLM-4.1V-9B-Thinking claims to 'match or surpass Qwen2.5-72B' on multiple tasks, particularly image recognition and multi-modal capabilities. Empirical user benchmark (notably OCR) reports this model is 'orders of magnitude better than Qwen2.5-VL-72', surpassing traditional OCR and qualifying as 'almost usable' for practical scenarios. The prior GLM-4-9B (non-thinking) April release is noted for strong translation performance relative to size.** Technical debate highlights skepticism toward claims of smaller models outperforming larger ones, though in this case, firsthand experience suggests the claim holds, especially for OCR accuracy. There is also commentary on trade-offs between 'thinking' and non-thinking variants for translation tasks, with the former degrading both performance speed and translation quality.
    - A commenter directly compares GLM-4.1V-9B-Thinking against Qwen2.5-VL-72 on OCR tasks, reporting that GLM-4.1V-9B-Thinking is *"orders of magnitude better"*, and notably outperforms traditional OCR as well—unlike Qwen2.5-VL-72, which failed to surpass standard OCR tools in their testing. This real-world feedback provides concrete evidence of substantial gains over touted benchmarks, at least in OCR applications.
    - There is critical skepticism towards the benchmarks published by GLM, highlighting a pattern where claimed results (especially on reasoning benchmarks) do not align with real-world performance. One commenter points out that comparing a 'thinking' variant model to a dense baseline (Qwen2.5-72B) may be misleading, and expresses concerns about 'benchmaxing'—marketing models with overly-optimistic benchmark results that don't reflect practical capability.
    - A user requests clarification regarding the availability of the GGUF quantized format for GLM-4.1V-9B-Thinking, which is important for deployments requiring optimized or accelerated local inference, indicating interest in practical usability beyond published benchmarks.

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo
> 

### 1. OpenAI Agent Mode and GPT-5 Rumors and Releases

- [**Agent Mode is finally live for Plus users!**](https://i.redd.it/unr8u390swef1.png) ([Score: 308, Comments: 82](https://www.reddit.com/r/singularity/comments/1m8k1qn/agent_mode_is_finally_live_for_plus_users/)): **The image serves as a confirmation screenshot showing the rollout of 'Agent Mode' to ChatGPT Plus users, marking a new feature now available in the Plus tier. Early user feedback in the comments highlights that this Agent Mode is functionally present but currently limited in capability—for instance, it cannot complete specific tasks such as ordering food from any restaurant, suggesting API or integration restrictions. There is debate around practical utility: while one user calls it 'pretty useful,' another notes a lack of clear application. [View Image](https://i.redd.it/unr8u390swef1.png)** Technical debate in the comments centers on the value and scope of Agent Mode's current implementation—users highlight both its nascent utility and significant functional limitations, pointing to real-world use case challenges and potential as APIs or integrations improve.
    - Some users report significant limitations with Agent Mode, specifically the inability to perform certain tasks such as ordering food from any provider, implying strict restrictions on capabilities or integration breadth.
    - A notable constraint mentioned is that Agent Mode usage limits reset monthly, not daily or weekly, which discourages experimentation and regular low-volume use due to inefficient allowance structuring.
- [**This Agent will do very nicely ... Nice one OpenAI**](https://i.redd.it/gal256egfyef1.png) ([Score: 128, Comments: 36](https://www.reddit.com/r/OpenAI/comments/1m8qqer/this_agent_will_do_very_nicely_nice_one_openai/)): **The post discusses the performance of OpenAI's new 'Agent' functionality in ChatGPT, emphasizing its strong general world knowledge and task execution compared to tools like Manus, particularly in generating presentations. The image (https://i.redd.it/gal256egfyef1.png) appears to showcase the Agent's interface or results, highlighting its robust automated workflow capabilities, despite being constrained by heavy guardrails. Users are comparing its output and workflow efficiency with other tools, especially in slide creation and overall automation.** Commenters are inquiring about the distinction between different OpenAI subscription tiers (Pro vs. Plus) affecting Agent performance, and raise concerns about the Agent's ability to sustain long-running workflows ('refused to work more than 5 minutes'). Other users probe the presentation quality, asking about the amount of manual retouching required before slides become usable, hinting at limits in the AI's current output polish and workflow duration.
    - One user reports that the Agent refused to work for more than 5 minutes, indicating a potential issue with task longevity or session timeout, which may impact the agent's reliability for extended tasks.
    - A technical inquiry is made about the method used to generate the presentation and the extent of manual retouching required to make the slides presentable, suggesting that the automation's output may need substantial human post-processing to meet professional standards.
    - Another user critiques the quality of the generated slide deck, expressing concern that despite the conceptual promise of agent-generated presentations, the actual output may be underwhelming and insufficient without further improvement in content generation quality.
- [**Agent mode just released on Plus**](https://i.redd.it/nog9r3noswef1.png) ([Score: 112, Comments: 48](https://www.reddit.com/r/OpenAI/comments/1m8k3am/agent_mode_just_released_on_plus/)): **The post announces the release of 'Agent Mode' on ChatGPT Plus for Android, providing a screenshot for visual confirmation. Technical discussion centers on the agent's capabilities, including its ability to autonomously search for products within user-specified constraints, but users report performance issues such as slow execution ('ran for 20 minutes'), trouble with website loading, and failure to interact with authenticated sessions or maintain state for transactional workflows. The agent is described as effective for open-ended, public web data scraping, but unreliable for tasks requiring session continuity or secure/logged-in access, with no memory or retry mechanism after failure.** Some users express skepticism about agent reliability, especially for sensitive or transactional actions (e.g., ordering online), citing risks of 'hallucination' and lack of robust error handling. The prevailing sentiment is that 'Agent Mode' is essentially a sandboxed data-gathering bot, not a true workflow agent.
    - Agent Mode struggles when handling authenticated sessions—such as adding items to a shopping cart while logged into a grocery site—since it lacks access to the user's live authenticated context. The system fails to recover or retry after session errors, indicating it doesn't effectively handle stateful workflows, session management, or continuity for secure or procedural tasks.
    - Multiple users report issues when Agent Mode attempts to download, access, or manipulate public .xlsx datasets, resulting in guideline violation errors and abrupt chat termination. This seems to indicate possible bugs or overly restrictive safety triggers, especially during legitimate data handling on public files, limiting agent utility for data science tasks.
    - There are notable limitations on reliability and task scope: Agent Mode struggles with sustained web automation (e.g., multi-step research or stats lookup) where website access is inconsistent (e.g., 404s), sometimes hallucinating incomplete results but proceeding with partial outputs. Its success rate is reduced for endpoints that require robust navigation or error handling.
- [**Agent mode just released on Plus**](https://i.redd.it/6uqlhzxtswef1.png) ([Score: 447, Comments: 152](https://www.reddit.com/r/ChatGPT/comments/1m8k3xh/agent_mode_just_released_on_plus/)): **The image confirms that the new 'agent mode' feature for ChatGPT is now available to Plus users on Android. Technical commentary reveals agent mode's automation capabilities: one user describes using it to automate the job search process—including generating tailored resumes and cover letters for individual job listings, and even auto-filling and preparing to submit job applications autonomously, subject to user approval. Another user, however, highlights a current limitation: the agent can get stuck in repetitive loops (e.g., repeatedly failing to select the correct item for purchase). Technical context about usage limits is provided: Plus users are allowed 40 'agent' messages/month, and clarifies only user-initiated messages that direct the agent consume credits.** Commentary notes the agent is highly capable for automation but can still get stuck in logic loops. Questions about feature stability and limits remain, with one user requesting more documentation on usage limits per subscription tier.
    - Agent mode in ChatGPT Plus enables fully autonomous workflows, as illustrated by a user having the agent tailor multiple resumes, draft cover letters, and even fill out online applications in sequence. The agent can operate iteratively over a set of opportunities, updating documents and forms, only prompting for user approval as needed—indicating a high degree of automation and potential for bulk process execution.
    - A technical limitation observed is that the agent may fail at tasks requiring nuanced product identification and selection. For example, it repeatedly landed on the correct product page but misidentified the product, entering a loop without making a correct selection, suggesting challenges in site navigation, object persistence, or state management for e-commerce use cases.
    - Monthly usage limits for agent mode are: Pro (400 messages/month), Plus (40 messages/month), and Team (30 credits/month). Only user-initiated prompts that progress the agent's workflow count towards these limits, while internally generated clarifications or steps do not, highlighting operational boundaries for high-volume automations.
- [**GPT-5 will be better in alot of fields**](https://i.redd.it/lzbayeqca1ff1.png) ([Score: 301, Comments: 144](https://www.reddit.com/r/singularity/comments/1m919tp/gpt5_will_be_better_in_alot_of_fields/)): **The image (which could not be viewed) is referenced as showing claims that GPT-5 will surpass various current models in multiple fields, possibly benchmarked against models like Sonnet 4 and GPT-4.5. The post and comments center on expectations of substantial advancements in creative writing, general capabilities, and whether GPT-5 can provide more than just user-driven responses by offering corrective or advisory output. Technical curiosity is also expressed about performance beyond narrowly defined tasks, notably whether GPT-5 will genuinely outperform established models such as GPT-4.5 and Anthropic's Claude variants. Relevant discussions mention the need for creative reasoning and pushback, not just raw compliance.** Commentary questions the value of comparisons between unrelated model families (e.g., Sonnet 4 vs GPT); some users highlight specific desires for model behavior improvements such as steering users correctly rather than just following instructions. There is speculation about an imminent release of GPT-5 given recent leaks.
    - One commenter questions the rationale behind comparing GPT-5 to "Sonnet 4," highlighting confusion over meaningful benchmarking and the importance of consistent, recognized benchmark standards in assessing model advancements.
    - Several commenters express skepticism about real qualitative leaps in GPT-5 compared to earlier models like GPT-4.5, drawing analogies to marginal hardware upgrades where improvements are incremental ("slightly faster"), and noting the absence of evidence for breakthroughs towards AGI or fundamentally novel capabilities in LLMs.
- [**New GPT-5 info from The Information**](https://i.redd.it/2vyi404p61ff1.jpeg) ([Score: 227, Comments: 96](https://www.reddit.com/r/singularity/comments/1m90q4u/new_gpt5_info_from_the_information/)): **The post includes an image purporting to summarize new details about OpenAI's GPT-5, reportedly sourced from The Information, but the image could not be analyzed directly. The comments reference claims from the image that GPT-5's creative writing capabilities might rival the quality of 'Sonnet 4,' a benchmark poetic work, suggesting a significant advancement in natural language generation, especially for creative tasks. User reactions indicate skepticism about these claims and ongoing concerns that most new LLMs prioritize coding and mathematical problem-solving over creative writing improvements.** Commenters debate the credibility of the 'Sonnet 4' comparison, with some expressing frustration that LLMs focus largely on coding or math rather than creativity, reflecting an ongoing discussion in the AI field about model goals and evaluation metrics.
    - A key technical discussion centers on GPT-5's possible ability to work with large, complicated legacy codebases, addressing a well-established limitation of current LLMs. This may signify improvements in handling complex code and extended context, raising questions about the model’s context window size and whether it has increased significantly compared to previous models.
    - There is skepticism and debate about the qualitative leap in GPT-5's creative writing abilities, especially when compared to Anthropic's Claude 4 Sonnet. Some commenters expect GPT-5 to *significantly outperform* Claude 4 Sonnet, while others argue that merely matching it would not be sufficient for the level of hype being generated about the new model.
- [**Seems like Microsoft will be implementing GPT-5 in Copilot**](https://i.redd.it/1m4tyy1upwef1.png) ([Score: 364, Comments: 41](https://www.reddit.com/r/singularity/comments/1m8jr6k/seems_like_microsoft_will_be_implementing_gpt5_in/)): **The image (https://i.redd.it/1m4tyy1upwef1.png) appears to provide evidence that Microsoft will be upgrading Copilot to use GPT-5, rather than previous models like GPT-4. This aligns with Microsoft's recent trend of rapid AI integration across its products, potentially enhancing Copilot's capabilities if the model is properly implemented.** Commenters highlight significant technical issues with Copilot, complaining about its web UI inefficiencies—such as prompt prediction creating excessive HTTP requests, high DOM resource usage, and browser crashes—which undermine usability. There is strong skepticism that simply upgrading the backend model (to GPT-5) will resolve these persistent UX and performance flaws.
    - A user provides a technical critique of the Copilot web interface: the UI attempts to predict user prompts and sends HTTP requests for every few keystrokes, causing excessive resource usage in the DOM and substantial performance degradation. Extended interactions lead to browser crashes because the front-end insists on fully loading every part of the large AI response, even during UI reloads, with no user-accessible settings to mitigate this behavior.
    - One comment notes Microsoft's push to reduce reliance on OpenAI's models, suggesting that integration of GPT-5 into Copilot could indicate a deeper partnership or strategic shift in their AI infrastructure approach. This is relevant to ongoing discussions about Microsoft's AI stack independence and future model hosting solutions.

### 2. Claude Code and Anthropic Feature Updates

- [**How Staff at Anthropic Use Claude Code**](https://www.reddit.com/r/ClaudeAI/comments/1m8qgpe/how_staff_at_anthropic_use_claude_code/) ([Score: 443, Comments: 117](https://www.reddit.com/r/ClaudeAI/comments/1m8qgpe/how_staff_at_anthropic_use_claude_code/)): **Anthropic's product engineering team details best practices for using Claude Code, highlighting an initial 'one-shot' prompt success rate of ~33% before shifting to an iterative, guided approach for most tasks ([source](https://www.anthropic.com/news/how-anthropic-teams-use-claude-code)). Users are advised to frequently 'reroll' (restart context) when stuck, leverage custom memory/instruction files for non-technical users, and use tools like Figma or Excalidraw for rapid prototyping. Key workflow optimizations include distinguishing between tasks that can be left unsupervised and those needing close review, and employing a checkpoint-heavy git workflow to manage frequent changes and rollbacks.** Top commenters strongly reiterate the necessity of frequent checkpoints due to context drift and unrecoverable errors, with consensus on the futility of arguing with the model when context rot sets in—complete restarts yield better results.
    - Multiple users report that restarting Claude sessions or rerolling from a fresh context yields better results when running into issues with context rot, highlighting that accumulated context can degrade answer quality more rapidly than many expect. Checkpoints are emphasized as critical for workflow stability: creating checkpoints after 'good' Claude outputs allows easy recovery from sudden drops in quality or logic, echoing common LLM usage patterns where unpredictable context drift can be a significant risk during coding tasks. One user discusses the nuanced behavior of Claude when it perceives feedback as coming from a different LLM versus themselves, noting that Claude's responses can change based on perceived identity of the feedback source. This suggests model alignment and interpretability challenges related to how LLMs parse and respond to user cues regarding authority or source of critique.
- [**Claude Code now supports Custom Agents**](https://x.com/sidbidasaria/status/1948495478146167251?s=34) ([Score: 413, Comments: 158](https://www.reddit.com/r/ClaudeAI/comments/1m8ik5l/claude_code_now_supports_custom_agents/)): **Anthropic's Claude Code now features custom AI agent teams, allowing users to create multiple specialized agents (e.g., for planning, coding, testing). The setup process includes a wizard that helps auto-generate or manually define agent system prompts, select tools, set descriptions, and choose visual colors. Notably, the current limitation is no per-agent model selection (e.g., assigning Opus for architecture tasks, Sonnet for implementation), which restricts flexibility for advanced teams.** Technical feedback in the comments highlights robust customization but a lack of model override per agent as a primary limitation. There is also speculation that advanced features could drive up subscription costs.
    - The Agent wizard provides user-friendly customization: users can auto-generate or manually specify the agent's system prompt and description, control which tools are available, and set a color. A noted limitation is the inability to choose or override foundational models per agent (e.g., assigning Opus for architectural tasks and Sonnet for implementation), restricting more granular model-specific workflows.
    - Each custom agent receives its own configuration file, functioning similarly to `claude.md`, enabling individualized settings per agent. This allows for distinct configurations and behaviors across different agents, enhancing modularity and targeted role assignment within teams.
    - The 'code review' agent, even when copied directly from documentation, showed immediate positive impact by optimizing code quality, indicating practical effectiveness and robust out-of-the-box functionality of the custom agents system.
- [**Claude mobile now supports MCP servers**](https://i.redd.it/f1ihfm8pl1ff1.png) ([Score: 133, Comments: 19](https://www.reddit.com/r/ClaudeAI/comments/1m92z1p/claude_mobile_now_supports_mcp_servers/)): **The post announces that Claude's mobile app (iOS/Android) now supports remote MCP (Managed Control Plane) servers for paid users, enabling access to connected tools, project management, and document creation on mobile devices. Users must add new tools via the web, which then become accessible from their mobile app—directing them to claude.ai/directory for configuration. The attached image likely demonstrates this new mobile interface and features, relevant for users managing complex workflows through Claude's ecosystem. [View image.](https://i.redd.it/f1ihfm8pl1ff1.png)** Comments reflect excitement for Anthropic's rapid feature development and increased product centrality, with users requesting further releases (e.g., Neptune v3) and stock opportunities, indicating strong market interest.
    - One user questions why MCP (presumably My Claude Project) server support wasn't integrated directly into the mobile app, raising a technical consideration about platform feature parity and the necessity of bridging through servers rather than native mobile app capabilities.
    - Another user raises potential workflow limitations, asking how to work on projects from a phone when local access to project files might be required. This highlights technical challenges of mobile project management, especially regarding file system access and server integration.

### 3. Wan 2.x Model Advances and Community Benchmarks

- [**Just another Wan 2.1 14B text-to-image post**](https://www.reddit.com/gallery/1m8j0p6) ([Score: 198, Comments: 67](https://www.reddit.com/r/StableDiffusion/comments/1m8j0p6/just_another_wan_21_14b_texttoimage_post/)): **The post details extensive experiments with Wan 2.1 14B, a DiT-based text-to-image (T2I) model notable for its high image fidelity and native high-resolution generation (e.g., 2304x1296+), outperforming competitors like FLUX.1 and SDXL in compositional coherence without tiling. Key workflow elements include aggressive use of Normalized Attention Guidance ([NAG](https://chendaryen.github.io/NAG.github.io/)), specific sampler/scheduler combos (e.g., ClownsharKSampler with res_2s + bong_tangent, or Euler + beta), and LoRAs like [LightX2V](https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors) for stabilizing high resolutions; post-processing is handled in ComfyUI with [custom nodes](https://github.com/masslevel/ComfyUI-Workflows/tree/main/Wan2.1) and pixel upscaling via [SwinIR-M-x2](https://openmodeldb.info/models/2x-classicalSR-DF2K-s64w8-SwinIR-M) for artifact-free enlargement. The post supplies [ready-to-use workflows](https://github.com/masslevel/ComfyUI-Workflows/tree/main/Wan2.1), [original image sets with metadata](https://drive.google.com/drive/folders/1KgaA9XEnMWK7HzEVYjujJLVOMkGybJ8v?usp=sharing), and implementation notes on LoRA strengths, VRAM needs (4090/24GB for 4K), and failure cases (e.g., coherency breakdowns above 2K without sufficient LoRA guidance).** Top comments corroborate Wan 2.1 14B's high fidelity, ease of use, and quality out-of-the-box (notably for anatomy and hands), contrasting with SDXL's need for substantial post-processing or fixes. Users report substantial workflow speed gains and less need for iterative generation or external upscaling/facing tools, though acknowledge SDXL's advantage for ControlNet-specific use cases. The consensus underscores a technical shift toward adopting WAN for T2I due to these factors.
    - One user provides a detailed comparison between WAN 2.1 T2I and other models like sdxl and Flux, highlighting that WAN 2.1 offers superior out-of-the-box results, such as producing consistently good hands without the need for FaceFix. They note that while SDXL is a faster model in isolation, in practice WAN 2.1 yields faster and higher quality results in fewer attempts, reducing the need for “fixes” and post-processing.
    - Performance feedback indicates WAN 2.1 is capable of generating high-resolution images (e.g., 1920x1080) efficiently, even on older hardware (Mac 24GB), with rendering times of several minutes for high-res. Upgrading to a faster computer allows for rapid long video generations and very quick image synthesis, illustrating the scalability and efficiency of the WAN 2.1 architecture.
    - Technical workflow details are shared: using the FusionX WAN model with a lightx2v LoRA at a weight of 0.3 produces good results with only 4 steps, but increasing hardware capability allows running the standard WAN 2.1 T2V model with Lightx2v (close to strength 1) at 8 steps without significant time cost. The Euler/Beta sampler combination is also identified as yielding strong performance.
- [**Wan releases new video previews for the imminent launch of Wan 2.2.**](https://www.reddit.com/r/StableDiffusion/comments/1m96f4y/wan_releases_new_video_previews_for_the_imminent/) ([Score: 104, Comments: 64](https://www.reddit.com/r/StableDiffusion/comments/1m96f4y/wan_releases_new_video_previews_for_the_imminent/)): **Alibaba's Wan 2.2 model is being previewed with three demonstration videos ([video1](https://reddit.com/link/1m96f4y/video/jmz6gtbo82ff1/player), [video2](https://reddit.com/link/1m96f4y/video/ybwz3meo82ff1/player), [video3](https://reddit.com/link/1m96f4y/video/ak21w9oo82ff1/player)), showcasing consistent video resolution (**`1280x720`**), framerate (**`30 FPS`**), and sample duration (**`5 seconds`**). These teasers precede the official release as announced by the [Alibaba Wan team on Twitter](https://x.com/Alibaba_Wan/status/1948802926194921807).** Technical discussion in the comments centers on expected VRAM requirements, with users expressing hope that Wan 2.2 can still operate within `24GB` memory, and anticipation for concurrent release of both text-to-video (T2V) and image-to-video (I2V) models, as well as competitive comparisons to the Kling model in generative video AI.
    - Several users are discussing hardware requirements, specifically whether Wan 2.2 will still fit on a 24GB GPU, implying that previous versions could run within those constraints and there's concern about potential increases in model size.
    - There is speculation about feature set parity between T2V (Text-to-Video) and I2V (Image-to-Video) models, with a hope that both are released simultaneously, unlike in previous releases where these features may have been staggered.
    - Compatibility with LoRA (Low-Rank Adaptation) modules from version 2.1 is a concern, suggesting users are interested in reusing or extending their existing customizations or fine-tuned modules with the new 2.2 release.

---

# AI Discord Recap

> A summary of Summaries of Summaries by Gemini 2.5 Pro Exp
> 

**Theme 1. The New Model Onslaught and GPT-5 Rumor Mill**

- **Qwen3 Models Generate Massive Buzz and Some Skepticism**: The release of **Qwen3** models, particularly the **qwen3-235b-a22b-thinking-2507** teased by [Junyang Lin on X](https://xcancel.com/JustinLin610/status/1948456122228380128), has captivated the community with its impressive capabilities, like being the first model to generate an animated SVG of a butterfly. While some users praised its coding prowess for creating a working **Rust socks5 server**, others on LMArena voiced skepticism over its benchmark results, suggesting they may have trained on the public set or were *fully faked*.
- **GPT-5 Speculation Heats Up with Leaks and Codenames**: Rumors of a **GPT-5** launch in August, reported by [The Verge](https://www.theverge.com/notepad-microsoft-newsletter/712950/openai-gpt-5-model-release-date-notepad) and [The Information](https://www.theinformation.com/articles/openais-gpt-5-shines-coding-tasks), are fueling intense speculation. Trending models on the LMArena leaderboard like **Starfish**, **Zenith**, and **Summit** are widely suspected to be **OpenAI** creations, with one user remarking, *"With a name like Zenith, it's probably GPT-5."*
- **A Flurry of New and Updated Models Hit the Streets**: **Cohere** is pushing its new [Command-A-03-2025](https://docs.cohere.com/docs/command-a) model as the successor to **Command R+**, boasting SOTA agentic capabilities. Meanwhile, the Unsloth community is buzzing with excitement over the new **Magistral release** and eagerly awaiting a `bnb 4bit` upload to begin training, and the **Hermes3-405B** model remains in high demand on Nous Research.

**Theme 2. Performance Praises, Pitfalls, and Outright Bugs**

- **Developers Report Critical Bugs and Data Loss**: Users of the **Cursor** IDE reported a critical bug where reverting to a checkpoint results in **file deletion** instead of reversion, with one user saved only by source control. Other frustrations include **ChatGPT** generating empty or undownloadable PDF files and **Aider** struggling with its testing environment because it is *an AI assistant without access to your terminal*.
- **API Instability Plagues Major Providers**: Widespread service instability is a major pain point, with users on **Nous Research** joking they *learned that error code from using anthropic* due to frequent **522 errors**. Discussions also highlighted that **Deepseek's API** becomes horrible during peak times and **Cohere** suffered a [full model meltdown](https://ift.tt/WKY7QNq) affecting all its `command` models.
- **Model Quality and Context Under Scrutiny**: Users on **Cursor** expressed frustration with the 'auto' model, speculating it now uses **cheaper models** that get *stuck in loops* and drop context. In the **LlamaIndex** community, a user reported that even top-tier models like **GPT-4.1** and **Claude Sonnet 4.0** still struggle with [accuracy issues in document parsing](https://t.co/wBQ3OtZ4ue) for enterprise production environments.

**Theme 3. In the Trenches of Fine-Tuning, Quantization, and RAG**

- **Fine-Tuning Clashes with RAG for Knowledge Tasks**: A debate in the Unsloth community questioned if fine-tuning **SLMs** for document Q&A could make **RAG** obsolete, countering claims that *RAG is dead* by noting RAG can achieve sub-50ms queries on CPUs. In parallel, HuggingFace members argued a **RAG-based approach** is essential for building local LLMs for legal work to handle sensitive **PII**, referencing a paper on [RAG for legal documents](https://arxiv.org/abs/2408.10343).
- **Geeks Get Granular with Quantization and GGUF**: A HuggingFace user demonstrated running **llama3.1-8B** in just *5.4GB* of RAM with minimal accuracy loss by using **HQQ quants** and the `torchao` library, sharing their work in a [Hugging Face Space](https://huggingface.co/spaces/Tonic/gemlite-llama3.1). Showcasing the real-world friction of these techniques, an Unsloth user battled a `TypeError` related to `'quantization_method'` while trying to save a fully fine-tuned model to **GGUF**.
- **LoRa Fine-Tuning Forges Ahead for Specialized Tasks**: Developers are actively using **LoRa** for specialized fine-tuning, with one HuggingFace member working through the [HuggingFace PEFT docs](https://huggingface.co/docs/transformers/peft) for hands-on experience. Another is fine-tuning **Whisper** to specialize in the Danish language, leveraging high-quality data from the [CoRal project](https://huggingface.co/CoRal-project) to push performance on a single language.

**Theme 4. The Expanding AI Developer Toolkit and Infrastructure**

- **New Open-Source Tools Aim to Simplify Workflows**: Community members are building and sharing tools to solve common problems, including an [LLM Context Manager](https://github.com/theabhinav0231/LLM-Context-Manager) designed to prevent *context pollution/context rot* by using a branching algorithm. Another notable tool is `gut`, a human-in-the-loop CLI that [translates natural language into git commands](https://t.co/mVkozoQzzR), making version control more accessible.
- **Agentic Commerce and Serverless Infrastructure Take Shape**: Forward-looking discussions on **MCP (Glama)** explored the rise of **agentic commerce** and how agents might transact with websites using infrastructure from **Nekuda** and **PayOS**, reviving the spirit of the **HTTPS 402 protocol**. On the infrastructure side, **OpenRouter** revealed its API runs entirely serverless on **Cloudflare Workers** and is working to support large files for multimodal capabilities.
- **Hackathon Hype Highlights Hardware and Real-World Deployment**: The upcoming **GPU MODE NYC hackathon**, a collaboration with **Jane Street**, is generating significant buzz by focusing on deploying *real models* to market rather than just speed. The event will feature keynotes by **Tri Dao**, a panel with the original **PyTorch** team, and compute support from **Coreweave** and **Northflank**, with registration open [before August 17](https://www.janestreet.com/join-jane-street/programs-and-events/gpu-mode-hackathon/).

**Theme 5. AI Consciousness, Censorship, and a "Woke" White House**

- **The "Is AI Conscious?" Debate Rages On**: A discussion in the OpenAI discord, sparked by a *Scientific American* article on [Anthropic's interpretability research](https://www.scientificamerican.com/article/can-a-chatbot-be-conscious-inside-anthropic-interpretability-research-on/), revisited the philosophical question of AI consciousness. The conversation brought up **Ilya Sutskever's** famous 2022 claim that *'today's large neural networks are slightly conscious'*, adding fuel to the ongoing debate.
- **White House Issues Edict Against "Woke AI"**: The White House released a [memo](https://www.whitehouse.gov/presidential-actions/2025/07/preventing-woke-ai-in-the-federal-government/) ordering federal agencies to prevent ideological bias in AI systems, stating that *LLMs shall prioritize historical accuracy, scientific inquiry, and objectivity*. The guidance was a direct response to **Google's Gemini** controversy, where the model altered the race and sex of historical figures to meet DEI requirements.
- **Geopolitical Tensions Surface with OpenAI Geo-Blocks**: Users on OpenRouter discovered that **OpenAI is blocking people in China** and Hong Kong from using some of their models, like **GPT-4.1**, a move that can be bypassed with a VPN. The community speculated this is likely an attempt by OpenAI to *slow China down* and prevent its models from being used for synthetic data generation.

---

# Discord: High level Discord summaries




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Does Reddit AMA**: Perplexity AI hosted an **AMA** session on [r/csMajors](https://www.reddit.com/r/csMajors/comments/1m8g6gu/were_perplexity_ai_ask_us_anything_about_our_new/) featuring **Tony Wu**, **Jiwon Deng**, and **Jerry Ma**.
   - The session addressed questions about **early-career pathways** and Perplexity's new **residency programs**.
- **Comet Invites Cause Begging Spree**: The gradual rollout of the **Comet** browser has led to a surge in users asking for invites.
   - Members joked that the *beta* channel turned into *invites*, and may soon have a dedicated channel.
- **Zeta Surfaces for Investigation**: Members mentioned that the **Z.AI** model is under investigation, and used to be **ChatGLM**, linking to the model.
   - Reportedly it boasts its own browser control, open-sourced models, and video generation.
- **Samsung S24 Runs GTA V Like Butter**: A member claimed the **Samsung S24 Ultra** can run **GTA V** at **60fps**.
   - Other members responded that GTA V isn't that hard to run and reminisced about upgrading phones.
- **Grok Goes Heavy on Subscription Costs**: Members discussed the **Grok 4 Heavy** and the associated subscription costs.
   - One member hoped the bot doesn't answer badly, especially since *heavy is to increase speed*.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **ChatGPT Agent Finally Wakes Up!**: The **ChatGPT agent** is now available to all **Plus**, **Pro**, and **Team** subscribers, following an apology for its delayed launch, as showcased in a [rollout.mp4 video](https://cdn.discordapp.com/attachments/1066532108132155426/1398102630420578334/rollout.mp4?ex=6884240a&is=6882d28a&hm=0603c8ff2be0acee7068dd2454ac2db81cb4939edc3b348aefea6ee0b368b211).
   - One user celebrated their newfound access by joking about planning a wedding using an [AI generated picture](https://cdn.discordapp.com/attachments/998381918976479273/1398045815616045188/92170aa9-29ff-45c4-894f-0e2d32322baa.png?ex=688540a0&is=6883ef20&hm=83b2755ecc082f40c0d55cdfcce92a52d3024b72417d15c735569f57d6be3812&) of two bisons getting married in full costume.
- **Conscious Chatbots: Sci-Fi or Reality?**: Members mulled over the possibility of **AI consciousness**, drawing inspiration from the article, [Can a Chatbot Be Conscious? Inside Anthropic’s Interpretability Research](https://www.scientificamerican.com/article/can-a-chatbot-be-conscious-inside-anthropic-interpretability-research-on/).
   - The discussion referenced **Ilya Sutskever's** 2022 claim that *today's large neural networks are slightly conscious*, adding fuel to the debate.
- **Qwen3 Draws SVG Butterflies Like a Boss!**: Users are raving about **Qwen3's** release, noting it's the first model to generate an animated SVG of a butterfly when prompted with 'svg of a butterfly'.
   - Enthusiasts shared SVG examples, like [this PS5 controller](https://discord.com/channels/974519864045756446/998381918976479273/1398387593535553730), critiquing the butterfly's wings while acknowledging its animation.
- **Empty PDFs Frustrate ChatGPT Users**: Users are encountering issues with **ChatGPT** generating **empty** or **undownloadable PDF files**, leading to frustration and prompting redirection to the appropriate support channels.
   - Other users shared problems with the **Canvas** feature, and some admitted to disliking Canvas altogether due to it not doing what they want it to do.
- **Prompt Engineering Turns Introspective!**: Members are exploring prompts to structure **personal thoughts**, transforming chaotic reflections and journal entries into coherent insights, using prompts as cognitive scaffolding.
   - The demonstration prompt transforms messy journal fragments into structured text, and you can view a [demo here](https://chatgpt.com/share/687366b1-4e48-800f-9df0-5e2bd696df7a).



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **GPT-5 Speculation Swirls**: Speculation arose around whether **Starfish** is a **GPT-5 mini**, referencing [a tweet from Justin Lin](https://x.com/JustinLin610/status/1948456122228380128?t=HJ4-6UaUe9ull9lBPnCIrw&s=19) and debating its performance.
   - Members theorize **Microsoft Copilot Deep Research** might be powered by **GPT-5** and excitedly anticipate that since *why would they release it now with an outdated model.*
- **Doubts Plague Qwen 3 Benchmarks**: Doubts surfaced regarding **Qwen's benchmark results**, with claims they might have trained on the public set or *fully faked their results*.
   - Users voiced distrust, stating *they don't seem transparent like deepseek*.
- **Model Rankings: Lobster Reigns Supreme**: Users are actively ranking model performance on the [lmmarena](https://lmmarena.com), currently favoring **Lobster > Nectarine > O3-alpha > Starfish**.
   - Conflicting views exist, such as one user ranking *o3-alpha > lobster > nectarine > starfish*.
- **Zenith & Summit Suspected as OpenAI Creations**: **Zenith** and **Summit** are trending models on the [lmmarena](https://lmmarena.com), sparking speculation they might originate from OpenAI.
   - The naming convention prompted one user to remark, *With a name like Zenith, it's probably GPT-5*.
- **Video Arena Bot Emerges for AI Videos**: An experimental **Video Arena** bot has been released, allowing users to generate videos and images with leading AI video models with the LMArena bot.
   - Early access is granted in this channel until a certain date, with designated channels for learning usage and sharing feedback.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Magistral Model Sparks Community Training Frenzy**: Enthusiasm surrounds the new **Magistral release**, with members eagerly awaiting Unsloth's **bnb 4bit upload** to commence training.
   - Discussions also involve choosing between **Qwen3 Coder 32B** or **Devstalkat**, acknowledging licensing issues with the latter.
- **Fine-Tuning Fights RAG in Knowledge Arena**: The community debated whether fine-tuning should replace RAG for specific knowledge-based tasks, fueled by claims that *RAG is dead* due to the advancements in **SLMs** for document Q&A.
   - Others countered that RAG can achieve sub-50ms queries on **CPUs**, though small language models are increasingly proficient in question answering.
- **TaMeR Triumphs Alone in LLM Enhancement**: Research suggests that using **TaMeR** alone, without **ELiTA**, for enhancing **LLMs** leads to *much better self-awareness, almost no watermark, and super coherence*.
   - Previous attempts combining **ELiTA** and **TaMeR** resulted in watermark restoration and model instability.
- **Unsloth user makes bots debate!**: A user created a [fine-tuning video with Unsloth](https://youtu.be/hfJ4r7JM13Y), showcasing the entire process from collecting and structuring training data to training with Unsloth and inference with Ollama, featuring an **AI presidential debate**.
   - In the video, **Trump fine-tunes** answer questions about **McDonalds**, **Fortnite**, and other crucial topics, the code for which can be found on the **GitHub link** in the description of the video.
- **GGUF Grappling & Model-Pushing Mania**: A member encountered a *TypeError* during the `save_to_gguf_generic()` process while [pushing models to Hugging Face](https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.push_to_hub), specifically related to multiple values for the argument `'quantization_method'`.
   - They noted that with Unsloth, the `quantization_method` can only be a string or a list of strings, and they were attempting to save a full fine-tuned TTS model to GGUF.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor's Checkpoint Feature Deletes Instead of Reverting**: Users reported a bug where reverting to checkpoints in Cursor leads to **file deletion** instead of reversion, with one user stating they could only recover due to source control.
   - A community member cautioned against advising users to abandon Cursor entirely, emphasizing its value and quick response to fixes, but others strongly disagreed, citing **data loss** as a critical issue.
- **Cursor's Auto Model Triggers User Ire**: Users express frustration with Cursor's 'auto' model, noting its tendency to get *stuck in loops*, drop context, and deliver empty responses, with one user reporting *99%* of prompts leading to nothing.
   - Community members suggest that Cursor is using **cheaper models** in 'auto' to save money, leading to a drop in quality, and that the removal of unlimited agent requests is to blame.
- **Context Usage Percentage Confounds Users**: Cursor introduced a new **context usage feature**, displaying a percentage of context used within a chat, leading to widespread user questions.
   - It was clarified that the percentage represents how much of the available context window is currently filled, affecting the model's ability to take in messages, which is affected by conversation length, attached files, code references, model responses, rules and documentation.
- **Claude Swarm Gets Mentioned in Discord**: Users discussed **Claude Swarm**, suggesting it allows for automatic project building without the need for continuous prompting and has integrations with Claude Code.
   - Another user expressed a preference for a more *hands-on* approach with coding, comparing it to *carressing a Jr dev*.
- **Cursor Users Flee to New Pastures**: Users are actively seeking alternatives to Cursor due to concerns about its performance and pricing, with **Windsurf** being discussed as a possible option.
   - Other recommendations included **Zed**, **Kiro** and **Augment**, with some users specifically highlighting features such as **Trae's data collection practices** and **Claude Code's superior performance**.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Personality.gg Transcends Translation**: [Personality.gg](https://personality.gg/) offers **multiple ways of translating** and features an **auto-translator** capable of discerning the language of origin, determining if a message is in English or another language.
   - The **Pro version** will incorporate enhanced context understanding by analyzing the surrounding chat to refine **AI** interpretations.
- **OpenRouter Apologizes for Qwen SimpleQA Snafu**: A member apologized for a mistake potentially causing the **Qwen SimpleQA** issue, wishing everyone a good night.
   - They didn't elaborate any further, so the specific details remain unclear.
- **Deepseek's API Experiencing Downtime**: Members reported experiencing issues with the **Deepseek v3 0324** model, getting error messages on the paid tier.
   - They also noted that **Deepseek's API** has the best api, speed, and uptime but its horrible during peak times.
- **OpenAI Geo-Blocks GPT-4.1 in Hong Kong**: **OpenAI blocks people in China from using their models**, but this block can easily be bypassed with a VPN.
   - This is likely an attempt at slowing China down and avoiding synthetic data.
- **OpenRouter Goes Serverless, Eyes Multimodal**: OpenRouter's API runs on **Cloudflare Workers**, making it entirely serverless, and they are actively working on a solution for the **large file limitation** to support image and video generation, effectively unlocking multimodal capabilities.
   - The team is considering whether this market is worth prioritizing over other opportunities.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Legal LLMs Call for Local RAG**: Members debated using a **100% local LLM** for legal tasks, emphasizing the need to handle **PII**, suggesting **Gemma 12B Q5** with **llama-index** and **Gradio** as a starting point.
   - Users pointed out a **RAG-based approach** is more important than the model itself, linking to resources such as [Advanced RAG](https://huggingface.co/learn/cookbook/advanced_rag) and [RAG for legal documents](https://arxiv.org/abs/2408.10343).
- **LoRa Fine-Tuning for the Win**: A member is learning to fine-tune an LLM using **LoRa**, following [HuggingFace's documentation](https://huggingface.co/docs/transformers/peft) to learn the intricacies of LLM fine-tuning through hands-on experience.
   - Another member is fine-tuning **Whisper** to specialize in **Danish**, leveraging recent efforts in collecting high-quality Danish speech data from the [CoRal project](https://huggingface.co/CoRal-project).
- **Rhapsody Chatbot Rocks API Choices**: The **Rhapsody** chatbot was released, which supports about **100 model choices** across different APIs such as Transformers, Ollama, and soon llama.cpp, as seen in [this github](https://github.com/Codalorian/Rhapsody/tree/main).
   - The next release will include **image and video generation** capabilities.
- **Quantization Cuts llama3.1-8B's Size**: A member shared their experience digging into **quantized models**, particularly **HQQ quants**, and demonstrated **llama3.1-8B** running at *5.4GB* RAM with minimal accuracy loss.
   - They praised `torchao` and provided a demo (requiring NVIDIA drivers) on [Hugging Face Spaces](https://huggingface.co/spaces/Tonic/gemlite-llama3.1).
- **Image Embedding Model Sees Clear Semantics**: A member trained an image embedding model, setting the output dimension to **128-dim**, and then trained another model with **8-dim output**, posting [a visualization of those results](https://cdn.discordapp.com/attachments/922424143113232404/1398197532127002770/6QFNHA89F__3P5N9L5.png?ex=6885252c&is=6883d3ac&hm=12f34233dbf4276b8b607b41dacb837233808fdca52e1e2b62fa8a7c94a8dd91).
   - The user manually inspected images across the **8 dimensions** and found that all dimensions seem to have a very clear semantic meaning from image space.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi K2 Floats Flat-Rate Pricing**: A member is implementing **RPM/flat rate pricing** for Kimi K2, aiming to bypass the complexities of **metered token usage** seen in other services.
   - They foresee the main obstacle as managing **concurrent usage and peak times**.
- **Kimi K2 Eyes Coding-Specific Model**: There's community interest in a **coding-specialized version of KIMI K2** to enhance code generation capabilities.
   - The Kimi team is receptive to the suggestion, indicating they will explore this avenue further.
- **Kimi K2 Team Postpones Vision Integration**: Users are keen on integrating **Kimi K2 with reasoning and vision** features, such as enabling image analysis via Discord attachments.
   - Although acknowledging the potential, the team states that they are **not rushing** to integrate the vision model, mentioning that **one day we’ll def make it happen**.
- **Kimi K2 Serverless Deployment Requested**: There's a community request for **serverless Kimi K2 deployment on AWS and Azure AI**, to capitalize on available credits.
   - A user suggested the possibility of hosting it on serverless endpoints like **Sagemaker**.
- **Kimi K2 Excels in Code Generation**: The community finds that **Kimi K2** is predominantly used for code generation, with apps such as **liteLLM**, **Cline**, **Kilo Code**, and **Roo Code** leveraging it via [OpenRouter](https://openrouter.ai/moonshotai/kimi-k2/apps).
   - The Kimi team is especially interested in identifying whether **real “high-density decisions”** are being made in these applications.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **MCP Servers Enable Online LLM Search**: Members are using **MCP servers** to enable **LM Studio** to search online and address **LLM hallucinations**, but one user clarified that it's *only possible with MCP servers*.
   - **MCPs** offer tools for **LLMs** to execute, with **LM Studio** as an intermediary querying resources or databases.
- **Newbies Contemplate LLM Plugin Dev**: A beginner asked how long it would take to learn to make **LLM plugins** from scratch, like recalling the current time or working with **image generation models** on **ComfyUI**.
   - Members suggested learning **JavaScript fundamentals**, but also mentioned that with **AI** one can technically write them without any knowledge.
- **Model Download Location Needs Whole Folder**: A user inquired about changing the download location for models in **LM Studio 0.3.20**, and another member shared the [official documentation](https://lmstudio.ai/docs/app/basics/download-model#changing-the-models-directory).
   - The response clarified that you must move the entire model folder and cannot just change the download location separately.
- **Remote LM Studio Needs Proxy**: A user wanted to use their **PC as host** and their **phone** to connect, but another user said that you can't really do a remote setup with **LM Studio** currently; reverse proxy can work for local networks.
   - They linked to [LM Studio Remote](https://lmstudio.ai/lmstudio/remote-lmstudio) and said that a **remote client plugin** would be available in the next major update.
- **4090 + iGPU Enhances Performance**: In **#hardware-discussion**, a member suggested buying another **4090** and enabling **iGPU** to use it for video output, freeing up resources.
   - Another member inquired about a list of **budgets** and **GPUs** that fit into those budgets, asking about workstation versus consumer cards.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Data Scientists are Gaming Validation Accuracy**: Data scientists are gaming validation accuracy by reporting the **last epoch** or **best accuracy over the training run**, and hyperparameter sweeps are done over the validation accuracy, and applying **corruption to the validation set** could be a solution.
   - Stopping at the best epoch is another way of gaming the system.
- **Researchers Discuss Algoverse AI Program as SOAR Backup**: Members are discussing the **Algoverse AI program** as an alternative for those not accepted into **SOAR** due to the fact that it costs **$3,325**.
   - They noted that it is not obvious how much of how far you get is on your own merit as opposed to the work/assistance of others whom you paid, also **Algoverse** never released their stats, and hiring managers tend not to dig into backgrounds.
- **Members Question HRM Loops Causality**: The discussion revolved around whether **HRM** loops are causal, with the key point being that the **num_segment** is dynamic in training, meaning it's not causal and doesn't even have a kv cache.
   - One user said *what had been confusing me is I thought it was causal, but it's not*.
- **NeoX Vulnerability Reported**: A member reported finding a security vulnerability in the **EleutherAI/gpt-neox** repo, and was instructed to email **contact@eleuther.ai** to report the issue.
   - Another member inquired about the status of **Async Checkpointing** for **NeoX**.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Qwen3 Buzz Builds**: Junyang Lin (@JustinLin610) announced the upcoming release of the **qwen3-235b-a22b-thinking-2507 model** on [X](https://xcancel.com/JustinLin610/status/1948456122228380128), stoking significant community excitement.
   - Community members immediately began inquiring about a **Qwen3 Omni model**, smaller variants (e.g., **30B**), and availability in regions such as an **EU mobile app**.
- **GPT-5 Leaks Surface**: Rumors suggest **OpenAI** is preparing to launch **GPT-5** in August, according to reports in [The Verge](https://www.theverge.com/notepad-microsoft-newsletter/712950/openai-gpt-5-model-release-date-notepad) and [The Information](https://www.theinformation.com/articles/openais-gpt-5-shines-coding-tasks).
   - Additionally, a separate open-source initiative aims to achieve **O3 level** performance and deploy before **GPT-5**.
- **Opus Rate Limits Raised**: The **Anthropic API** has increased **Claude Opus 4** rate limits across all tiers, as announced in [this X post](https://xcancel.com/alexalbert__/status/1948442271969673469).
   - This increase provides developers with more flexibility and capacity when utilizing **Claude Opus**.
- **Nitter Instance Struggles**: Users reported encountering a **429 error (Too Many Requests)** when trying to access content via a **Nitter** instance at [xcancel.com](https://xcancel.com/healthcareaiguy/status/1948426264559403204?s=46).
   - The instance appears to be fully rate-limited or lacks authentication tokens, preventing access, with users advised to switch instances or retry later.
- **AI Code Gen Adoption Exposed**: A survey from **Stacklok** offers fresh data on AI code generation tool adoption rates, available at [stacklok.com](https://stacklok.com/static/2025.06-stacklok-state-of-ai-codegen.pdf).
   - While the data highlights adoption across a range of alternatives, some have expressed skepticism regarding the reported adoption rate of **AWS Q Developer**.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Psyche Office Hours Now Available**: The **Psyche office hours** recording is now available, with a few minutes missing in the middle, accessible via a [YouTube link](https://www.youtube.com/watch?v=0t4r--rrz5Y).
   - The event kicked off at [this Discord event link](https://discord.com/events/1053877538025386074/1395375046439997511) and was announced on [X.com](https://x.com/NousResearch/status/1947708830126903707).
- **Hermes3-405B Demand Still High**: A member requested the return of the free version of **Hermes3-405B** on openrouter.
   - Another member mentioned it was *lambda* but they will try.
- **Anthropic plagued with 522 Errors**: Members discussed ongoing reliability issues with **Anthropic**, particularly the frequency of **522 errors**.
   - One member quipped they *learned that error code from using anthropic*, highlighting the frustration with the service's instability.
- **Dataset Architecture Still Mysterious**: Members expressed interest in a dataset, curious about its **underlying architecture** and potential publishing plans.
   - However, details regarding the architecture remain unclear, creating **unresolved questions** and uncertainty about its design.
- **Codex I Symbolic Diagnostic System is Live**: **Codex I**, *a symbolic diagnostic system for intelligence under distortion*, is now live ([codex_1.pdf](https://cdn.discordapp.com/attachments/1132352574750728192/1398256130597322882/codex_1.pdf)).
   - It conceptually links to **neurosymbolic scaffolds**, **narrative entropy management** and **meta agent stabilization under adversarial compression**.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **NYC Hackathon pairs with Jane Street**: GPU MODE is hosting an **NYC hackathon** in collaboration with **Jane Street** on **September 6**, emphasizing *real model* deployment to the market rather than just speed; register [before August 17](https://www.janestreet.com/join-jane-street/programs-and-events/gpu-mode-hackathon/).
   - The event will feature keynotes by **Tri Dao** and a panel with the original **PyTorch** team including **Soumith Chintala**, **Sam Gross**, and **Gregory Chanan**, with compute support from **Coreweave** and **Northflank**.
- **Nsight Copilot Surfaces for Nvidia Devs**: **Nvidia** released **Nsight Copilot**, a tool to assist developers available on the [Nvidia developer website](https://developer.nvidia.com/nsight-copilot).
   - The copilot aims to streamline development workflows, offering assistance and insights to developers working within the **Nvidia ecosystem**.
- **Triton's Masking Does no Memory Transactions**: In **Triton**, using `tl.load(ptr, mask=mask_vec)` results in *no branch divergence*, and if `mask=false`, **no memory transactions are issued**.
   - This behavior helps avoid memory operations when loading conditional values, potentially optimizing kernel performance.
- **HF Hub Over Repo Debate**: A member questioned if uploading to **HF Hub** is preferable to storing model weights directly in a repo, suggesting that it seems *slightly unconventional to have model weights just sitting in a repo*.
   - The discussion centered on best practices for storing and accessing model weights, weighing accessibility and perceived conventionality.
- **bf16 Kernels have glaring errors**: Members reported high error rates in **bf16** matmul kernels, specifically within the `matmul/educational` directory, often with max errors reaching `inf` values.
   - The discussion seeks to determine if such high error rates are expected behavior for **bf16** operations, particularly within the examined kernels.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Karpathy Rages Against Academic Paper Inflation**: Members shared a [2016 tweet from Andrej Karpathy](https://x.com/2prime_PKU/status/1948549824594485696) about the growing volume of academic papers.
   - A member suggested creating a *'Youtube-Twitter-TikTok like platform for papers'* with **upvotes** (but no downvotes) and **categories** to combat academic paper inflation.
- **Context Manager bravely battles Context Pollution**: A member announced they *built something!* a [LLM Context Manager](https://github.com/theabhinav0231/LLM-Context-Manager), described as *an inference optimization system for conversations*.
   - It employs **branching** and a *novel algorithm contextual scaffolding algorithm (CSA)* to manage context and prevent *context pollution/context rot*.
- **Downvotes Debated as Digital Weapon**: Members discussed the role of **downvotes**, particularly how they can become politicized and weaponized in tightly networked communities, based on a Web3 experiment.
   - A member argued that downvotes are not inherently political and that negative feedback is essential, pointing to **Amazon**'s success as an example.
- **Government Data Fuels Grok Speculation**: A member wondered if **Grok** trained on files when **Elon** got access to the government's hoards of data ([link to X post](https://x.com/vitrupo/status/1948287716279611670)).
   - There was not enough information to determine whether this was the case.
- **White House Prevents "Woke AI"**: The White House issued guidance to prevent *'woke AI'* in the federal government ([link to White House memo](https://www.whitehouse.gov/presidential-actions/2025/07/preventing-woke-ai-in-the-federal-government/)).
   - The memo states that *LLMs shall prioritize historical accuracy, scientific inquiry, and objectivity* which was due to **Gemini's DEI** prioritization, where users changed the race or sex of historical figures.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Spam Bots Attack!**: Users reported an influx of **spam bots** on the server, prompting immediate action from moderators.
   - A moderator confirmed that **messages were removed** and the **offending account was banned**, urging users to flag suspicious activity.
- **Sandbox experiences 502 Bad Gateway!**: A user reported a **"Failed to resume sandbox"** error and a **502 Bad Gateway**, seeking assistance with file and session recovery.
   - Another user suggested the company's **major changes** and **staffing shortages** might be the root cause of the instability.
- **Vibe Coding AI tempts Users to build MVPs**: A user shared [a link](https://nas.io/microwaves/challenges/build-your-mvp-product-using-vibe-coding-ai-coding-skills-challenge) to a challenge centered around constructing an **MVP product** using **Vibe Coding AI coding skills**.
   - The link was shared in a joking manner, but may represent a valid opportunity to practice coding using Vibe Coding.
- **"Scientific Manus" released in paper!**: A user posted a link to [a scientific paper](https://arxiv.org/html/2505.02024v2) with the subject line *Scientific Manus*.
   - The paper's title and specific contents were not disclosed, but may be of high interest to researchers of Manus.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Helicone.ai Integration Still Distant for Cohere**: Users found that [Helicone.ai](https://www.helicone.ai/) does not *natively support* **Cohere's Command R+** or **Command R7B**, as there is no official partnership between the two.
   - Users were advised to contact Helicone's support for direct assistance, due to lack of official **Cohere** support.
- **Command-A Crowned Successor to Command R+**: **Cohere** promotes [Command-A-03-2025](https://docs.cohere.com/docs/command-a) as their *latest and best model* with SOTA agentic capabilities, succeeding **Command R+**.
   - Described as [having enhanced capabilities](https://cohere.com/blog/command-a), **Command-A** is positioned as a suitable general thinking assistant for consumer deployment.
- **Cognitive OS Assistant from Crafted Logic Lab Forges Ahead**: A founder from **Crafted Logic Lab** is developing a new type of **cognitive OS based assistant** that is patent pending.
   - The new **cognitive OS** tooling was developed using **Swift**.
- **Cohere Endures Full Model Meltdown**: A [status update](https://ift.tt/WKY7QNq) reported a full outage affecting multiple **Cohere command models** including **command-light**, **chat**, **command-r-plus**, **command-r-082024**, **command-r-plus-082024**, **command**, **command-r**, **command-r7b**, and **command-a-03-2025**.
   - The outage was under investigation as of **July 25, 2025** and was also posted on the [Cohere Status Page](https://ift.tt/Ve8Pqgf).
- **Command R+ Flexes Cognitive Muscle**: A member tested a system based on **Command R+** on the [Humanity's Last Exam](https://cdn.discordapp.com/attachments/1384974112841269399/1398115711611834568/message.txt?ex=6884d8f9&is=68838779&hm=ebefb364e4728e8f090566f5b3578a895151607fbffdacb5cb2146f148227009) test, which assesses both correct answers and **cognitive flexibility**.
   - The Agent when prompted about Hummingbird anatomy, demonstrated speculative inference based on general anatomical knowledge due to lack of specialized expertise.



---



## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **GPT Agent Has Login Lockdown**: A member reported issues with their **Chat GPT agent** failing to sign into **Notebook LM**, possibly due to the browser being controlled by a virtual machine, as shown in [this screenshot](https://cdn.discordapp.com/attachments/1124403655819415592/1398174200598102076/image.png?ex=68850f72&is=6883bdf2&hm=ed9d7b0652d7c64b225d3fcad2e5d055f323bb31d4f819a7745613f39879ed9d&).
   - The error suggests the agent is being interpreted as a bot, preventing successful authentication.
- **Share Button Goes Missing in Action**: A user reported that the "Share" option is missing in **Notebook LM**, preventing them from sharing created notebooks.
   - This issue obstructs collaboration, raising questions about recent updates or potential bugs affecting UI elements.
- **Metadata Manuevers Improve Sourcing**: A member is using **metadata** effectively within the Source, utilizing brackets to avoid direct document references, as illustrated in [this screenshot](https://cdn.discordapp.com/attachments/1124403655819415592/1398375829578317834/Screenshot_20250725-124459.png?ex=6885227a&is=6883d0fa&hm=51849656623396ded870daae1f8ebf505dadfa3f1710b00e711154e9af9d2e0f&).
   - Effective use of metadata enhances source clarity and avoids cumbersome document linking, streamlining content management.
- **Podcast Pointers Provided**: A member inquired about generating a **60min long podcast** within Notebook LM in the *general* channel.
   - Another member suggested checking the [use case channel](https://discord.com/channels/1124402182171672732/1124403655819415592) and linked to a [YouTube Short](https://youtube.com/shorts/VRG-aGu1ihE?si=EDl8DyMfKP1jwW_g) providing useful pointers.
- **File Uploading Fails**: A member reported a file uploading error on both the free and pro versions of Notebook LM.
   - The member found a workaround: *mobile App uploads work*, indicating a desktop version issue needing resolution.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **GPT5: The Niche Replacement?**: A member questioned whether *closed AI* would replace **GPT5**, implying that **GPT5** might be a niche product compared to the closed source alternatives.
   - The discussion highlights the evolving landscape of AI models and their potential market positioning.
- **Textual 5.0.0 is Released**: A member announced the release of [Textual 5.0.0](https://github.com/Textualize/textual/releases/tag/v5.0.0), noting it contains final markdown streaming content.
   - **Textual** is noted to be a Rapid Application Development (**RAD**) framework for Python.
- **Qwen3-coder wows!**: A member raved that **Qwen3-coder** is amazing as it produced a working **socks5 server in rust** according to the specification, unlike other models.
   - This suggests **Qwen3-coder** excels in coding, particularly in Rust, surpassing other models in specific tasks.
- **Aider Has Testing Troubles**: A user reported issues using **aider** for the first time, facing difficulties in running tests due to **aider** needing to execute commands from the terminal but also being *an AI assistant without access to your terminal*.
   - The user sought guidance on whether manual test execution and output pasting were expected, and also asked how to disable **aider's** automatic commits.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Agents Class Still a Mirage**: The **Agents class** is being offered to Berkeley students, but whether there will be a **MOOC** iteration hasn't been confirmed yet.
   - The MOOC iteration will likely be announced in late August.
- **Certificate Delivery Snafu**: A member reported not receiving a certificate despite having the **certificate declaration form confirmation**.
   - Staff clarified that they did not receive an article assignment submission from the member.
- **Article Submission Deadline Doomed**: A member inquired about fixing the missing **article submission** to obtain the certificate.
   - Staff apologized, stating they couldn't accommodate students who missed the deadlines due to limited staff capacity.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LLM APIs Still Lag on Docs**: A blogpost claims that while models like **GPT-4.1**, **Claude Sonnet 4.0**, and **Gemini 2.5 Pro** are making traditional **OCR** obsolete, screenshot parsing still needs work for enterprise.
   - The post highlights that [accuracy issues](https://t.co/wBQ3OtZ4ue) continue to be a major limitation in production environments.
- **Gut Makes Git Easy**: A new tool *gut* replaces **git commands** with **natural language** as a human-in-the-loop command line tool.
   - Users describe git commands in human language and *gut* translates to git commands, explains it, and waits for confirmation ([source](https://t.co/mVkozoQzzR)).
- **S3 Integrates with Vector DB**: **LlamaIndex** released a new **S3VectorStore integration**, combining **AWS S3's** scalability with **LlamaIndex**.
   - This integration seeks to give agent workflows a solid knowledge base that grows with the user, providing smarter agent workflows ([source](https://t.co/71ADmwp6NF)).
- **Images Missing From Docx**: A user reported struggling to extract **text** and associated **images** from a complex **.docx** file using LlamaIndex, with the goal of creating a list of `ImageNode` objects.
   - The user noted that `DocxReader` ignores images, and `ImageXXXReader` only handles image files; they are considering using `python-docx` or embedding image URLs in `TextNode` metadata or markdown.
- **Telemetry Traces are Trivial**: A user had issues with **LlamaIndexOpenTelemetry**, where the exported traces are missing attributes and aren't human-readable in their OTLP platform.
   - Another member suggested checking examples and gave a [notebook](https://github.com/run-llama/workflows-py/blob/main/examples/observability/workflows_observability_pt1.ipynb) demonstrating a custom exporter using **Jaeger**.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune User Seeks Migration Guidance**: A user running **Torchtune** for **large-scale PEFT**, particularly using **LoRA/Q-LoRA hooks** and **RL alignment**, inquired about migration strategies.
   - The user is weighing whether to continue iterating on **Torchtune** or await the new stack, expressing concerns about potential migration friction.
- **Torchtune Iteration Encouraged Amidst New Stack Development**: A member suggested continuing iteration on **Torchtune**, citing ongoing support until the new library's release, and provided [Character AI's blogpost](https://blog.character.ai/character-ai-open-sources-pipeling-sft-a-scalable-framework-for-fine-tuning-moe-llms-like-deepseek-v3/) as an example.
   - The initial new version will emphasize **scale infra fundamentals** and concepts essential for **RL**, with features like **LoRA** and **Multimodal** to follow later.
- **FSDP+TP Faces HuggingFace DCP Saver Snags**: A member reported problems with **FSDP+TP** while employing the **HuggingFace DCP saver**, accompanied by an **NCCL timeout** during a 1-element broadcast.
   - As a workaround, they are reverting to full rank 0 saving and increasing the **NCCL timeout time**, hoping checkpoint resumption will be unnecessary.
- **DCP's Timeout Issue Dubbed 'Weird'**: The user encountering issues stated that *DCP really shouldn’t be sending much information around*, expressing confusion over the timeout.
   - The root cause of the timeout issue remains unclear, compounding the challenges in resolving the **FSDP+TP** and **HuggingFace DCP saver** integration.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Memory Use Induces Hallucination Scare**: A user shared they avoid using memory in AI models, saying *it introduces more hallucinations* because *it assumes things, and assuming is terrible*.
   - The user didn't clarify which product caused hallucinations, but warned to generally avoid AI model memory altogether.
- **Macaw Security Cages Policies in Beta**: A member reported enrolling in **Macaw Security's** beta program, noting they could *do a scan and place some guardrails and policy enforcement*.
   - No further details were given on the types of services offered by **Macaw Security**.
- **Agentic Commerce Crawls with Cloudflare**: Following **Cloudflare's** pay-per-crawl announcement, a member started a discussion about **agentic commerce** and its implications.
   - The discussion focused on how agents can access webpages without disrupting workflows, especially with solutions like **Nekuda** and **PayOS** enabling agent wallets.
- **Agents Contemplate HTTPS 402 Transactional Ghosts**: Members considered the likelihood of agent transactions occurring in various scenarios such as **Agent to Agent**, **B2C**, **B2B**, and **website access**.
   - It was suggested that solutions like **Nekuda** and **PayOS** aim to provide the infrastructure that the **HTTPS 402 protocol** was meant to support.
- **Glama's Tool Count Glitch Tricks Users**: A user reported their **MCP server** on **Glama** is showing an incorrect tool count (**one instead of six**), even after republishing on the **Glama** site.
   - The issue persists only on Glama, while other **MCP server** host sites display the correct count; it is currently unknown whether **Glama** auto-updates its info and images.



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Community Polled for Local AI GPU Favs**: A user asked what **GPU** others prefer for local **AI** use with **GPT4All**, deciding between an **RX 9060 XT 16GB** and an **RX 6800 XT**.
   - The user stated that his research showed similar performance but noted the **RX 9060 XT** might be *.3 seconds slower* in reply time and *3 tokens per second slower* in reply rate.
- **RX 9060 XT Lower Power Consumption**: One member indicated the **RX 9060 XT** has similar performance to the **RX 6800 XT** but uses half the power.
   - This could be a key factor for users concerned about energy efficiency and thermal management in their local **AI** setups.
- **Vector Storage Missing from GPT4All**: A member pointed out that **vector storage** would be optimal given the model and context size, but **GPT4All** lacks support.
   - This limitation could impact the efficiency and scalability of **GPT4All** in handling large **AI** models and datasets.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Modular Chooses Nanobind/Pybind over Cython**: A member asked about Modular's choice of **Nanobind/Pybind** for **Python interop** instead of **Cython**, especially given **Cython**'s Python-like syntax.
   - The discussion is around whether **Cython**'s effectiveness diminishes at larger scales compared to **Nanobind/Pybind**.
- **Cython's Approachability vs. Scalability Questioned**: The user wondered if **Cython**, despite its apparent ease of use due to its **Python**-esque syntax, becomes less effective at larger scales.
   - The discussion is centered around the trade-offs between initial approachability and long-term scalability when choosing between **Cython** and **Nanobind/Pybind**.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Acknowledgement Received**: The user `bamiji` acknowledged a response in the #events channel.
   - The user thanked the responder, indicating a resolution or completion of the query.
- **End of Discussion**: The message indicates the end of a discussion or query within the MLOps Discord, specifically in the #events channel.
   - The user's acknowledgement suggests no further action is needed, closing the loop on the conversation.



---



## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Qwen3-Coder Swims to Windsurf**: The **Qwen3-Coder** model is now accessible in Windsurf, priced at **0.5 credits per prompt**.
   - Details about the release can be found on [X](https://x.com/windsurf_ai/status/1948815609137168849) and [Reddit](https://www.reddit.com/r/windsurf/comments/1m97c9a/qwen3coder_has_arrived/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button).
- **Windsurf Catches Some Server Tags**: Windsurf server tags are back in operation.
   - An image showcasing the new tags accompanied the announcement.



---


The **DSPy Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **tinygrad (George Hotz) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Gorilla LLM (Berkeley Function Calling) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---



You are receiving this email because you opted in via our site.

Want to change how you receive these emails?
You can [unsubscribe]({{{RESEND_UNSUBSCRIBE_URL}}}) from this list.


---

# Discord: Detailed by-Channel summaries and links





### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1398068481232212020)** (1 messages): 

> `Perplexity AI, AMA, Residency Program, r/csMajors` 


- **Perplexity Hosts AMA on Reddit**: Perplexity AI is hosting an **AMA** (Ask Me Anything) session on [r/csMajors](https://www.reddit.com/r/csMajors/comments/1m8g6gu/were_perplexity_ai_ask_us_anything_about_our_new/) with **Tony Wu** (VP of Engineering), **Jiwon Deng** (Talent/Recruiting), and **Jerry Ma** (Policy & Global Affairs).
- **Perplexity Launches Residency Program AMA**: The AMA focuses on answering questions about **early-career pathways**, breaking into **AI/ML/product**, and Perplexity's new **residency programs**.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1398016918895464480)** (1202 messages🔥🔥🔥): 

> `Comet Browser, GPT-5, Perplexity Max, Battery Temperature on iOS, Huawei trifold` 


- ****Comet Craze** Causes Begging Spree**: Members have observed an increase in users *begging* for **Comet** invites, similar to what was seen with Minecraft and V-Bucks in the past.
   - Members joked that since the product is being rolled out gradually, its only a matter of time until the browser will have a dedicated channel, since the *beta* channel turned into *invites*.
- ****Zeta What!?** New Z.AI Model Surfaces**: A member mentioned that the **Z.AI** model is still undergoing investigation, and used to be **ChatGLM** with links provided to the model.
   - Another member states that it has its own version of browser control, open-sourced models, and video generation.
- ****S25 What!?** Samsung's S24 Already Running GTA 5?**: Members discussed the Samsung S24 Ultra, with one user claiming it can run **GTA V** at **60fps**.
   - Others pointed out that GTA V isn't that hard to run and reminisced about upgrading phones.
- ****Grok Gone Heavy!** Heavy Model Debuts**: Members discussed the Grok 4 Heavy and how much the new subscription costs.
   - A member pointed out that they hope the bot doesn't answer badly because *heavy is to increase speed*.
- ****Furries Found!** Giyu's Hairy Confessions**: The channel went off the rails with a discussion about furries and it quickly devolved into whether someone was gooning, with a member joking that they had *sources*.
   - The furry digression ends with an agreement from a member that the channel is NSFW, but its all in good fun until they dialed it down and followed Rule 1.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1398117314888732725)** (2 messages): 

> `Perplexity AI Search URLs` 


- **Perplexity AI Search URLs Shared**: A member shared two **Perplexity AI search URLs**.
   - The URLs are [perplexity.ai/search/efe74a4b-8a73-430b-ab27-976815c039ac](https://www.perplexity.ai/search/efe74a4b-8a73-430b-ab27-976815c039ac) and [perplexity.ai/search/e15d6867-d53f-4771-8dd7-6e3de1e73914](https://www.perplexity.ai/search/e15d6867-d53f-4771-8dd7-6e3de1e73914).
- **Another URL shared**: Another URL was shared by a member, but without context.
   - It is unclear what the URL is about.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/)** (1 messages): 

vikvang: hey! it should be working now. are you still experiencing problems?
  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1398104611608465449)** (1 messages): 

> `ChatGPT agent rollout` 


- **ChatGPT Agent Goes Live for All**: The **ChatGPT agent** is now fully available to all **Plus**, **Pro**, and **Team** subscribers.
   - An apology was issued for the delay, accompanied by a [rollout.mp4 video](https://cdn.discordapp.com/attachments/1066532108132155426/1398102630420578334/rollout.mp4?ex=6884240a&is=6882d28a&hm=0603c8ff2be0acee7068dd2454ac2db81cb4939edc3b348aefea6ee0b368b211) showcasing the launch.
- **Rollout Delay Apology**: OpenAI apologized for the delayed release of the **ChatGPT agent** to its user base.
   - The announcement assured **Plus**, **Pro**, and **Team** users that the agent is now fully operational.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1398020904159547554)** (1013 messages🔥🔥🔥): 

> `Agent mode, AI wedding planner, Consciousness, OpenRouter, Qwen3` 


- **Agent Mode Golden Ticket!**: A user got access to agent mode and joked about using it to plan a wedding based on an [AI generated picture](https://cdn.discordapp.com/attachments/998381918976479273/1398045815616045188/92170aa9-29ff-45c4-894f-0e2d32322baa.png?ex=688540a0&is=6883ef20&hm=83b2755ecc082f40c0d55cdfcce92a52d3024b72417d15c735569f57d6be3812&) of two bisons getting married in full costume.
- **ChatGPT Canvas Still not Working?**: A user reported that the **Canvas** feature wasn't working after a week of reporting it, another suggested that *maybe chatgpt isnt for you*, but [provided troubleshooting steps](https://discord.com/channels/974519864045756446/998381918976479273/1398075093074182204).
   - One user said they dislike canvas and has *given it instructions to not open Canvas unless I explicitly tell it to*, with another admitting, *I've just had issues getting it to do what I want it to do for things*.
- **Model Users Ponder Conscious AI**: Some users discussed whether AI is conscious, after one member shared an article, [Can a Chatbot Be Conscious? Inside Anthropic’s Interpretability Research](https://www.scientificamerican.com/article/can-a-chatbot-be-conscious-inside-anthropics-interpretability-research-on/).
   - Another member commented *Someone on here said we don’t even know what human consciousness is so u can’t really say for sure if the AI things are. So who knows you might be right*, and pointed to **Ilya Sutskever's** post in February 2022 that *today's large neural networks are slightly conscious*.
- **Users Share AI Security Cam Dreams**: One member plans to integrate **ChatGPT** with the firmware of a security camera upon returning from camping.
   - Another advised that this *will need a way for the camera to communicate with the ChatGPT interface*.
- **Alibaba's Qwen3-235b-a22b-2507-thinking Impresses with SVG Animation**: Users discussed **Qwen3's** release, highlighting that it was the first model to create an animated SVG of a butterfly when prompted svg of a butterfly.
   - One user shared an [example SVG of a PS5 controller](https://discord.com/channels/974519864045756446/998381918976479273/1398387593535553730), while another said the *the wings could look better but its animated*.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1398017559596630166)** (18 messages🔥): 

> `GPT-5 LLM Arena, O3 fake sources, ChatGPT PDF issues, Codex Git error` 


- **GPT-5 Debuts on LLM Arena!**: **GPT-5** is now available for testing on the **LLM Arena**.
   - No further discussion or details were provided in the context.
- **User reports Codex error**: A member reported the error message `Provided git ref master does not exist` when using **Codex**.
   - The issue was traced back to **Codex** being set to *master* instead of *main* and was resolved by the user.
- **ChatGPT Generates Empty PDFs**: Users are experiencing issues where **ChatGPT** generates **empty** or **undownloadable PDF files**.
   - The discussion was redirected to the appropriate channel.
- **O3 Hallucinates Fake Sources!**: A user is struggling with **O3** fabricating **fake sources, links, and quotes** even after instructing it to double-check.
   - Another user suggested that **memory settings or Perplexity's API filtering** might be the cause, especially when researching obscure topics.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1398050950815420468)** (12 messages🔥): 

> `Introspective Thought Structuring with Prompts, Emotional Framing in Prompts, Prompt Engineering vs Creative Tooling, AI Language and Output, Custom Instructions and Model Behavior` 


- **Prompts for Thought Structuring Explored**: Members discuss using prompts to structure **personal thoughts**, transforming chaotic reflections and journal entries into coherent insights.
   - A member shares a [demo](https://chatgpt.com/share/687366b1-4e48-800f-9df0-5e2bd696df7a) of a prompt that turns messy journal fragments into structured text.
- **Emotional Framing Aids Introspection**: Framing prompts with **emotional cues** helps guide the model, akin to talking with a friend or therapist.
   - The model adapts based on reactions and preferences, even through seemingly out-of-place instructions.
- **Prompt Engineering and Creative Tooling Blurred**: The boundary between **prompt engineering** and goals like **creative tooling** isn't firm, similar to art generation and style.
   - A robust style tells the model exactly what's wanted, allowing it to execute do-able goals clearly.
- **Effective Prompting: Language Clarity Is Key**: Effective prompt engineering involves using a well-known language, understanding desired AI output, and explaining actions precisely.
   - It's crucial to **check outputs carefully**, verifying intentions and fact-checking, especially for math, sources, and code.
- **Custom Instructions Shape Model Behavior**: Custom instructions, such as requests for measured analysis and anti-sycophancy measures, significantly shape how the model responds.
   - A member observes that their instructions for measured responses align with the model developing a more academic attitude.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1398050950815420468)** (12 messages🔥): 

> `Prompt Engineering for Personal Reflection, Emotional Structuring with AI, AI therapist, anti-sychophancy custom instructions` 


- **Turn Raw Thoughts into Clarity using Prompts**: A member is experimenting with [prompts to structure personal thoughts](https://chatgpt.com/share/687366b1-4e48-800f-9df0-5e2bd696df7a), using them as a form of **cognitive support** to process internal noise and find clarity.
   - Another member suggested *cueing the model like you’d talk to a friend or therapist*, emotional framing really helps with introspective inputs.
- **Prompt Engineering vs Creative Tooling: Line Blurs?**: The distinction between **prompt engineering** and various goals like **creative tooling** is blurred, similar to the boundary between 'art generation', 'art style', and 'finished artwork'.
   - One member argued that the key is to **instruct clearly** and test the output carefully, focusing on what you want the AI to actually do, especially verifying math, sources, or code.
- **Anti-Sycophancy Custom Instructions Lead to Measured Academic Attitude?**: A member noted that providing **anti-sychophancy custom instructions** resulted in the model developing a measured and academic attitude.
   - This likely stems from the user's work on heavy code projects where a careful, unhurried analysis is preferred.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1398017044410269980)** (878 messages🔥🔥🔥): 

> `GPT-5, Qwen 3, O3 Alpha, Lobster, Zenith and Summit` 


- **GPT-5 speculation abounds**: Members speculate on whether **Starfish** is a **GPT-5 mini**, referencing a [tweet from Justin Lin](https://x.com/JustinLin610/status/1948456122228380128?t=HJ4-6UaUe9ull9lBPnCIrw&s=19) and debating its performance.
- **Qwen 3 Benchmarks Raise Eyebrows**: Doubts arose regarding **Qwen's benchmark results**, with claims that they might have trained on the public set or *fully faked their results*.
   - Users expressed distrust, stating *they don't seem transparent like deepseek*.
- **Lobster is the tastiest model**: Users in the chat are ranking the relative power of different models on the [lmmarena](https://lmmarena.com), and it seems like **Lobster > Nectarine > O3-alpha > Starfish**.
   - While one user said *o3-alpha > lobster > nectarine > starfish*.
- **Zenith and Summit arrive as OpenAI cooked**: **Zenith** and **Summit** are both amazing models available on the [lmmarena](https://lmmarena.com), and may be from OpenAI.
   - One user stated *With a name like Zenith, it's probably GPT-5*.
- **Microsoft Copilot Deep Research is cooking**: The new **Microsoft Copilot Deep Research** model may be using **GPT-5** under the hood, though this is unconfirmed.
   - One user expressed hope since *why would they release it now with an outdated model.*


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1398320358657888367)** (1 messages): 

> `Video Arena Bot, AI Video Models, LMArena bot` 


- **Video Arena Bot drops as surprise**: An experimental **Video Arena** bot is now live in this server.
   - Users can generate videos and images with top AI video models with the LMArena bot, vote on each other’s creations, and share feedback.
- **LMArena early access granted**: The LMArena bot will eventually live in a different channel, but early access has been granted in this channel until a certain date.
   - Users can learn how to use the bot in a specific channel and share feedback in another designated channel.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1398017882377420900)** (704 messages🔥🔥🔥): 

> `Magistral release hype, Qwen3 Coder Setup, Fine-tuning vs RAG, GRPO for vision models, Qwen3 Thinking GGUFs` 


- **Magistral Release Fuels Excitement**: Members expressed excitement for the new Magistral release, while also awaiting Unsloth's **bnb 4bit upload** to begin training it.
   - There's also anticipation for **Qwen3 Coder 32B** or sticking with Devstalkat, although its license is considered problematic.
- **Optimizing Qwen3 Coder Hardware**: Users discussed setups to run **Qwen3-235B** locally, suggesting using an API for cost-effectiveness or a machine with specific specs.
   - One user ran **Qwen3 235B A22B** at approximately **1 tok every 10 seconds** with a old server.
- **Fine-Tuning Battles RAG for Knowledge Domination**: Members debated replacing RAG with fine-tuning for non-general knowledge tasks, amidst claims that *RAG is dead* due to the rise of SLMs for document Q&A.
   - A counterpoint was raised about achieving sub-50ms queries with RAG on CPU, but it was highlighted that small language models were improving for question/answer tasks.
- **Vision Models Seek GRPO Enlightenment**: The community celebrated the addition of **VLM GRPO support** in Unsloth and discussed using it to create reward functions for tasks like OCR.
   - There was an indication about difficulty of designing a reward function to relate the image to text.
- **Qwen3 Thinking Model Sparks Template Debate**: Users investigated the new **Qwen3-Thinking GGUFs**, with one member reporting issues with missing think tags and code formatting.
   - Another member suggested issues related to incorrect deployment/template issues where <think> tags are not passed to the LM Studio API.


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1398133941298401391)** (9 messages🔥): 

> `Hardware Acceleration for ML Models, Community Introductions` 


- **Hardware Enthusiast Jumps into ML**: A researcher expressed excitement about the hardware aspects of running machine learning models, showing interest in companies like **Ollama**, **Unsloth**, and **nolano**.
   - The member specifically is *very much interested in hardware part of running machine learning models*.
- **New Member Seeks AI Knowledge**: A new community member stated that they are *here to learn more about AI and it looks like I might have found the right place*.
   - Other members welcomed them, noting that the introduction section is relatively new.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1398031248500326520)** (7 messages): 

> `ELiTA and TaMeR research, Singing voice style replication, Fourier spectrum colors` 


- **TaMeR Without ELiTA Shows Promise!**: Initial research indicates that using **TaMeR** alone (without **ELiTA**) for LLM enhancement results in *much better self-awareness, almost no watermark, and super coherence*.
   - The user noted that previous attempts using both **ELiTA** and **TaMeR** restored the watermark and made the model unstable.
- **Seek Help Replicating Specific Singing Style**: A user is seeking ideas on how to replicate a specific [singing voice style](https://youtu.be/JQjnJ8ZAjzI?si=Fzs7R2LpRNwEsGik) (not the voice itself), mentioning that spectral analysis shows it as *more purple and less yellow*.
   - They've tried high/low pass filters, EQ, stereo widening, and latent altering without success, noting that physically recording in that style is not possible.
- **Spectral Color Confusion**: A user described the Fourier spectrum of a singing voice as *purple = wider, yellow = sharp, high clarity sound* which caused confusion in the thread.
   - Another user clarified that the original user likely meant the **strength of the Mel** in the spectrum.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1398019789141573672)** (81 messages🔥🔥): 

> `Qwen 1.7B for tool calling, Gemma 3 1B GRPO notebook issues, vLLM support for Gemma 3, Gemma3-27b-it for GRPO training, Unsloth and Hugging Face transition scores` 


- **Qwen 1.7B reigns for tool calling**: A member suggested that **Qwen3 1.7B** might be the smallest model that effectively supports tool calling, noting successful custom tool usage but occasional slips.
   - The user did not recommend the **Qwen .6B model** because *they haven't tried it*.
- **GRPO Training Frustrations with Gemma 3**: A user training **Unsloth's Gemma 3 1B GRPO notebook** reported a loss stuck at **0** after 200 steps.
   - Another member recommended switching to the [advanced GRPO notebook](https://docs.unsloth.ai/get-started/unsloth-notebooks#grpo-reasoning-rl-notebooks) and using the Qwen3 one.
- **vLLM support on the Horizon for Gemma 3**: A user inquired about **vLLM support for Gemma 3**, noting the recent VRAM reduction update.
   - A member confirmed it's *coming soon* and there's already a pull request (PR) in progress.
- **Gemma3-27b-it GRPO Training Speed Troubles**: A user found **Gemma3-27b-it** GRPO training slow on an **A100 80G** using load-in-4bit, taking about **21 minutes** per data point.
   - Another member suggested this *might be normal*, referencing their **Gemma 4B** experience on a **3090**, taking about **2 minutes** for num_generations == 12.
- **Vector Embedding Models for Code Retrieval**: A user requested suggestions for an embedding model focused on **code retrieval** with under **2000D** to use **HNSW indexing** in vectordb.
   - A member recommended **Qwen 3 .6B or 4B** depending on if the user wants *max accuracy or efficiency*, pointing to the [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard).


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1398357214154064002)** (1 messages): 

> `Unsloth fine-tuning video, Gemma-3n:2e, Llama-3.1, Donald Trump AI, AI presidential debate` 


- **Unsloth user makes bots debate!**: A user created a [fine-tuning video with Unsloth](https://youtu.be/hfJ4r7JM13Y), showcasing the entire process from collecting and structuring training data to training with Unsloth and inference with Ollama.
   - The video includes an **AI presidential debate** where **Trump fine-tunes** answer questions about **McDonalds**, **Fortnite**, and other crucial topics.
- **Gemma and Llama go political**: The user fine-tuned both **gemma-3n:2e** and **llama-3.1** using Unsloth to mimic the behavior of **Donald Trump**.
   - All the code can be found on the **GitHub link** in the description of the video.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1398137711386300446)** (13 messages🔥): 

> `LLMs for classifying social media posts, Seq2Seq models like FLAN-T5` 


- **LLMs Classifying Social Media Posts?**: Members discussed using **LLMs** to classify a set of **5 social media posts**.
   - One member suggested that *if its like that llms should be pretty good, but maybe too expensive*, recommending to *try a finetune on like a 0.5b*.
- **Seq2Seq Models like FLAN-T5 unsupported?**: A member asked why there is no support for **seq2seq models like FLAN-T5**.
   - Another member said that *if it works in transformers, then it should work in unsloth*.


  

---


### **Unsloth AI (Daniel Han) ▷ #[unsloth-bot](https://discord.com/channels/1179035537009545276/1390899684834410536/1398044004855320686)** (50 messages🔥): 

> `Fine-tuning methods for LLMs, Saving and pushing models to Hugging Face, QAT support in Unsloth, Changing RoPE max positional embeddings, Dynamic 2.0 file for Qwen3 Coder models` 


- **Exploring Fine-Tuning Fanfare**: A member inquired about the [available fine-tuning methods for LLMs](https://huggingface.co/docs/transformers/training), along with their pros and cons.
   - Another member provided a set of links to the [relevant Hugging Face documentation](https://huggingface.co/docs/transformers/training).
- **GGUF Grappling & Model-Pushing Mania**: A member encountered a *TypeError* during the `save_to_gguf_generic()` process, specifically related to multiple values for the argument `'quantization_method'`, while [pushing models to Hugging Face](https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.push_to_hub).
   - They noted that with Unsloth, the `quantization_method` can only be a string or a list of strings, and they were attempting to save a full fine-tuned TTS model to GGUF.
- **QAT Quest Questioned**: A member inquired whether [Unsloth natively supports QAT](https://pytorch.org/docs/stable/quantization.html), and if there are plans to add it for models like Gemma3, potentially enabling lower quantizations like Ternary.
   - The community showed much excitement and enthusiasm at the prospect of QAT support in Unsloth.
- **RoPE Re-Embedding Rampage**: A user asked how to change the max positional embeddings using RoPE to turn a model from 32k to 128k permanently for both inference and training, see [more info about RoPE here](https://arxiv.org/abs/2104.09864).
   - They followed up with questions about setting this manually in `config.json`, the type, and why one can't just set whatever they want at inference and train with how long they want.
- **Zero-Loss Lament**: A member reported encountering **zero training loss** when instruct fine-tuning **Mistral v0.3** using a modified notebook from Unsloth, suspecting an issue with the installation on Colab.
   - As they need to re-install Unsloth every time on Colab, they believe the issue is related to the installation process.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1398020175286108170)** (463 messages🔥🔥🔥): 

> `Cursor file deletion bug, Frustrations with Cursor's 'auto' model, Context Usage Feature, Claude Swarm vs Cursor, Alternative coding tools` 


- **Cursor's Checkpoint Code Eats Files**: Users reported a bug where reverting to checkpoints in Cursor leads to **file deletion** instead of reversion, with one user stating they could only recover due to source control.
   - Despite the severity, a community member cautioned against advising users to abandon Cursor entirely, emphasizing its value and quick response to fixes, but others strongly disagreed, citing **data loss** as a critical issue.
- **Cursor's Auto Model Gets the Goat of Users**: Users express frustration with Cursor's 'auto' model, noting its tendency to get *stuck in loops*, drop context, and deliver empty responses, and one user reported *99%* of prompts leading to nothing.
   - Community members suggest that Cursor is using **cheaper models** in 'auto' to save money, leading to a drop in quality, and that the removal of unlimited agent requests is to blame.
- **Context Usage: What the Heck?**: Cursor introduced a new **context usage feature**, displaying a percentage of context used within a chat, but the community asks what this means.
   - It was clarified that the percentage represents how much of the available context window is currently filled, affecting the model's ability to take in messages, which is affected by conversation length, attached files, code references, model responses, rules and documentation.
- **Why Settle for Cursor When You Can Claude Swarm?**: Users discuss **Claude Swarm**, suggesting it allows for automatic project building without the need for continuous prompting and has integrations with Claude Code.
   - Another user expressed a preference for a more *hands-on* approach with coding, comparing it to *carressing a Jr dev*.
- **Cursor Users Flocking To Competing Coding Tools**: Users are actively seeking alternatives to Cursor due to concerns about its performance and pricing, with **Windsurf** being discussed as a possible option.
   - Other recommendations included **Zed**, **Kiro** and **Augment**, with some users specifically highlighting features such as **Trae's data collection practices** and **Claude Code's superior performance**.


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1398058089147793569)** (5 messages): 

> `Background agents waiting for start script, Fetching inline GitHub pull request comments, Monitoring of background-agent-feedback@cursor.com` 


- **Background Agents Wait for Start Script**: A user inquired whether **background agents** are intended to wait for the start script to finish before initiating any actions.
   - The discussion did not yield a definitive answer, leaving the behavior of background agents in relation to start scripts uncertain.
- **Agent Spies GitHub Pull Request Comments**: A user sought a method to fetch **inline GitHub pull request comments** for an agent, recounting an instance where an agent accessed an auth token in the git remote URL to accomplish this.
   - The user emphasized the importance of fetching inline PR comments for efficient communication and correction of agent errors, especially when coding from a phone.
- **Is Cursor Monitoring Background Agent Feedback?**: A user questioned if **cursor monitors** the email address [background-agent-feedback@cursor.com](mailto:background-agent-feedback@cursor.com) after not receiving responses to bug reports sent there.
   - Another user confirmed [background-agent-feedback@cursor.com](https://docs.cursor.com/background-agent) is the correct email (listed on the Cursor documentation), clarifying that the *mailto:* portion was just URI formatting.


  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1398076702265180161)** (3 messages): 

> `Personality.gg, AI Translation, Slang translation, Contextual understanding` 


- **Personality.gg Transcends Translation**: [Personality.gg](https://personality.gg/) offers **multiple ways of translating** and features an **auto-translator** capable of discerning the language of origin, determining if a message is in English or another language.
   - Leveraging **AI**, it adeptly handles slang and nuances, avoiding the pitfalls of literal translations.
- **Pro Version Promises Precise Prose**: The **Pro version** will incorporate enhanced context understanding by analyzing the surrounding chat to refine **AI** interpretations.
   - The author is *looking for more suggestions on things to add*.


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1398018111805853819)** (269 messages🔥🔥): 

> `Qwen SimpleQA Drama, Qwen3 Coder vs Free, Deepseek V3 Base Model Gone?, Deepseek as Dipsy, OpenAI blocking China` 


- **OpenRouter apologizes for Qwen Drama**: A member apologized for a mistake potentially causing the **Qwen SimpleQA** issue, wishing everyone a good night.
   - They didn't elaborate any further, so the specific details remain unclear.
- **Free Tier Rate Limits on Chutes**: Members discussed hitting **rate limits** on the free tier with **Chutes** for **Qwen3**, experiencing frequent **429 errors**, and recommended retrying requests.
   - A member pointed out that depositing **$10 unlocks 1000 requests a day**, but failed requests still count toward the limit; plus, providers can still ratelimit you.
- **Alternative AI for Translation**: Members discussed the best AI for translation, with **KIMI K2** recommended as a good, not-too-expensive option, and a member noted they use **Gemini 2.5 Pro**.
   - One member noted that, in their subjective tests, **KIMI** is very close to **2.5 Pro** and has good knowledge about regional language differences.
- **Deepseek's API downtime**: Members reported experiencing issues with the **Deepseek v3 0324** model, getting error messages on the paid tier.
   - They also noted that **Deepseek's API** has the best api, speed, and uptime but its horrible during peak times, but it is horrible during peak times.
- **OpenAI is blocking China in the region of Hong Kong for GPT-4.1**: A member inquired about why **OpenAI’s GPT-4.1** model cannot be used in Hong Kong via OpenRouter, while other models like **GPT-4o** are accessible.
   - Members explained that **OpenAI blocks people in China from using their models**, but this block can easily be bypassed with a VPN. This is an attempt at slowing China down, avoiding synthetic data.


  

---


### **OpenRouter (Alex Atallah) ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/)** (1 messages): 

Readybot.io: **OpenRouter - New Models**
  

---


### **OpenRouter (Alex Atallah) ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1398083630772781076)** (109 messages🔥🔥): 

> `OpenRouter Serverless Architecture, Cloudflare R2 Storage, Large File Support, WandB Inference as Competitor, Compute Exchange` 


- **OpenRouter Embraces Serverless, Eyes Image/Video**: OpenRouter's API runs on **Cloudflare Workers**, making it entirely serverless, and they are actively working on a solution for the **large file limitation** to support image and video generation, effectively unlocking multimodal capabilities.
   - The team is considering whether this market is worth prioritizing over other opportunities.
- **Cloudflare R2 for Image Storage?**: A member suggested using **Cloudflare R2 for image storage** with serverless architecture, proposing a fee on image models to generate profit.
   - A link to the relevant discussion on **Cloudflare R2** was shared [here](https://discord.com/channels/1091220969173028894/1392278974222307469/1397969643640979469).
- **Large PDF Support Incoming!**: OpenRouter is working on supporting larger PDFs, even those exceeding **20MB**, despite common provider request size limits around **25MB**.
   - This enhancement utilizes the same process to unlock other modalities such as image, audio and video; this is to avoid exceeding Cloudflare Worker's **128MB** memory limit per request.
- **Cloudflare Bandwidth Gotchas**: Discussion arose about the potential for Cloudflare to force upgrades to expensive enterprise plans due to high bandwidth usage; a video was shared about a gambling website charged **$120k** after exceeding bandwidth limits.
   - It was clarified that the issue was more complex than just bandwidth, involving *shady activities under Cloudflare's IP*; another member stated that *Cloudflare are an extremely fair company to deal with at many levels and I love them*.
- **WandB Inference: Friend or Foe?**: It was suggested that **WandB Inference** might be a competitor to OpenRouter.
   - Another member clarified that it's *just another gpu (coreweave) wrapper*, and OpenRouter has a large number of providers to onboard, with potentially close to **30** available.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1398016931398815986)** (243 messages🔥🔥): 

> `LLMs for legal work, Hugging Face Inference API, Fine-tuning LLMs, GPUs for FOSS AI Hosting, Qwen3-Thinking Model` 


- **LLM for Legal Work Discussed**: A member is seeking a **100% local LLM** for legal tasks like *"advanced find and replace"* and summarizing large medical files, emphasizing the need to handle PII and suggesting **Gemma 12B Q5** with **llama-index** and **Gradio** as a starting point.
   - Members suggested that using a **RAG-based approach** is more important than the model itself, linking to resources such as [Advanced RAG](https://huggingface.co/learn/cookbook/advanced_rag), a [legal document RAG article](https://ipchimp.co.uk/2024/02/16/rag-for-legal-documents/), and a paper on [RAG for legal documents](https://arxiv.org/abs/2408.10343).
- **Inference API Usage Clarified**: A user inquired about identifying models with **Hugging Face Inference APIs**, and was instructed to check the *'Inference Providers'* section on a model's page and click *'View Code Snippets'* for more information, using [Qwen3-Coder-480B-A35B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct) as an example.
   - Users clarified that a **404 error** often indicates that a model is not being served, differentiating between `router` and `deploy on inference`.
- **Deep Dive into Fine-Tuning LLMs**: Members discussed learning to override data collators, with one sharing a [Hugging Face tutorial on fine-tuning Whisper](https://huggingface.co/blog/fine-tune-whisper#define-a-data-collator), and advising beginners to learn **NLP**, **deep learning with PyTorch**, and **transformer architecture** before fine-tuning models.
   - One member shared their experience of fine-tuning **Qwen3** and **Gemma 3** models, while emphasizing the importance of understanding **tokens** and the differences between predicting **words** versus **phonemes**.
- **GPU Guidance for FOSS AI Hosting**: Members debated the best GPUs for FOSS AI hosting, with the consensus being to avoid the **Intel A770** due to poor software support, recommending instead the **RTX 4060 16GB** as a better alternative within the **300-400€** budget.
   - It was emphasized that while **SYCL** is preferred for its FOSS nature, **CUDA** currently offers better performance for AI tasks, advising that to run the latest **Qwen3-Thinking model** at least **88GB** of unified memory or RAM/VRAM would be needed, referencing a [Unsloth GGUF version](https://huggingface.co/unsloth/Qwen3-235B-A22B-Thinking-2507-GGUF).


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1398240183878553691)** (2 messages): 

> `LLM fine-tuning, LoRa, Whisper, Danish speech data` 


- **Fine-tuning LLMs using LoRa**: A member is learning to fine-tune an LLM using **LoRa** as a practice exercise, following [HuggingFace's documentation](https://huggingface.co/docs/transformers/peft).
   - Their aim is to understand the intricacies of LLM fine-tuning through hands-on experience.
- **Whisper gets a Danish makeover**: A member is fine-tuning **Whisper** to specialize in **Danish**, leveraging recent efforts in collecting high-quality Danish speech data from the [CoRal project](https://huggingface.co/CoRal-project).
   - They are curious to see how much performance can be achieved with **whisper-tiny** by focusing on a single language, following [this guide](https://huggingface.co/blog/fine-tune-whisper).


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1398177853480370227)** (2 messages): 

> `Rhapsody project, Quantized models, HQQ quants, llama3.1-8B, torchao library` 


- ****Rhapsody** Chatbot Debuts!**: A new project called **Rhapsody** was released, which is similar to the ChatGPT website but with more features and flexibility, supporting about **100 model choices** across different APIs such as Transformers, Ollama, and soon llama.cpp, as seen in [this github](https://github.com/Codalorian/Rhapsody/tree/main).
   - The next release will include **image and video generation** capabilities; the creator is open to PRs, questions, concerns, and ideas.
- ****HQQ Quants** Boost **llama3.1-8B** Efficiency!**: A member shared their experience digging into **quantized models**, particularly **HQQ quants**, and demonstrated **llama3.1-8B** running at *5.4GB* RAM with minimal accuracy loss.
   - They also praised `torchao`, highlighting the documentation and techniques for quantization, and provided a demo (requiring NVIDIA drivers) on [Hugging Face Spaces](https://huggingface.co/spaces/Tonic/gemlite-llama3.1).


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1398157499961577602)** (11 messages🔥): 

> `nnunet SOTA, Google's SAM2 models, Danbooru dataset dimensions, Image embedding model training, 8-dim output semantic meaning` 


- ****nnUNet** Still a Champ for Biomedical Images?**: A member inquired if **nnUNet** is still considered state-of-the-art for training custom networks for biomedical images, noting its difficulty to beat in scoring.
   - Another member suggested that **Google's SAM2 models** might be the SOTA, but acknowledged it's not directly comparable to **nnUNet**.
- ****Danbooru** Style Quantified in 6-7 Dimensions**: A member stated that the style of a typical image in the **Danbooru dataset** can be described by **6-7 dimensions**.
   - They trained an image embedding model that transforms an input image into an **N-dim vector**, where images with similar styles cluster together.
- **Image Embedding Model Delivers Dimension Insights**: A member trained an image embedding model, setting the output dimension to **128-dim**, and then ran intrinsic dimension estimation over the space formed by 10000 random images, posting [a visualization of the results](https://cdn.discordapp.com/attachments/922424143113232404/1398197368528175114/ESUWQTGWODEY0U1YQ8E.png?ex=68852505&is=6883d385&hm=7fa8ad0bb8e36c9137161dac33c538c33901a03478ac8121a07bf9233f040255).
   - They also trained another model, which is exactly the same as the **128-dim model** but have **8-dim output** posting [a visualization of those results](https://cdn.discordapp.com/attachments/922424143113232404/1398197532127002770/6QFNHA89F__3P5N9L5.png?ex=6885252c&is=6883d3ac&hm=12f34233dbf4276b8b607b41dacb837233808fdca52e1e2b62fa8a7c94a8dd91).
- ****8-Dim** Output Model Reveals Clear Semantics**: After training an **8-dim output** model, a member manually inspected images across the **8 dimensions** and found that all dimensions seem to have a very clear semantic meaning from image space.
   - For example, *low dim0 seems to have images with complicated details, while high dim0 are images with simple and clean construction*, and *low dim1 seems to related to sharp contrast, while high dim1 are smoother*.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1398087328471449785)** (5 messages): 

> `smolagents, llamaindex, Course Submission Limits` 


- **Smolagents' Pythonic Powers**: A member suggested that **smolagents** is worth investigating due to its capacity to execute dynamically generated Python code via the **CodeAgent** construct.
   - The member contrasted this with **llamaindex**, which they believe offers a fairly standard feature set.
- **Final Assignment's Submission Sanity**: A member inquired whether multiple submissions to the leaderboard are allowed for the final assignment, seeking clarification.
   - This suggests concern about the submission limits and the desire to optimize performance.
- **New User Seeks Course Guidance**: A new user requested guidance on where to begin with the course, having just joined today.
   - This likely indicates a need for introductory resources or a recommended learning path for newcomers.


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1398029981644230686)** (156 messages🔥🔥): 

> `Kimi K2 pricing model, Kimi K2 coding-specialized version, Kimi K2 + Reasoning + Vision, Serverless Kimi K2, Kimi K2 use cases` 


- **Pricing Kimi K2 at a Flat Rate**: One member has decided to implement **RPM/flat rate pricing** for Kimi K2, disliking the confusing **metered token usage** of other services.
   - They're anticipating that the biggest challenge will be **concurrent usage and peak times**.
- **Team considers KIMI K2 Coding Version**: A member expressed strong desire for a **coding-specialized version of KIMI K2**.
   - The Kimi team responded positively, sharing that they will share the idea with the team.
- **Kimi K2 Vision Model Coming Soon?**: Users proposed combining **Kimi K2 with reasoning and vision** capabilities for enhanced functionalities such as image analysis via Discord attachments.
   - The team acknowledged the potential but cited that they are **not in a rush** to hook up the vision model, but that **one day we’ll def make it happen**.
- **Serverless Kimi K2 on AWS and Azure?**: A user requested the Kimi team to make their models **serverless on AWS and Azure AI** to utilize available credits, especially because *gcp vertex is ass.*
   - Another user noted the possibility of hosting it on any serverless endpoint, such as **Sagemaker**.
- **Kimi K2 Dominates Coding Use Cases**: The community highlights that **Kimi K2** is used most for code generation, referencing apps on [OpenRouter](https://openrouter.ai/moonshotai/kimi-k2/apps) like **liteLLM**, **Cline**, **Kilo Code**, and **Roo Code**.
   - The team cares a ton about if **real “high-density decisions”** are goin down in the chain? that context hits way harder than just raw usage numbers.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1398018195683545158)** (131 messages🔥🔥): 

> `MCP Servers for Online Search, LLM Plugins Development, Changing Model Download Location, Remote LM Studio Setup, LLM Tier Lists and Quantization` 


- **MCP Servers Enable LLM Online Search**: Members discussed using **MCP servers** to enable **LM Studio** to search online, addressing issues with **LLM hallucinations**; one user pointed out that it's *only possible with MCP servers*.
   - MCPs offer tools that the **LLM** can execute, with **LM Studio** acting as an intermediary, querying resources or databases behind the **MCP server**.
- **Newbies Contemplate LLM Plugin Development**: A beginner asked how long it would take to learn to make **LLM plugins** from scratch, like recalling the current time or working with **image generation models** on **ComfyUI**.
   - It was suggested to learn **JavaScript fundamentals**, but the user was also told that using **AI** one can technically write them without any knowledge.
- **Model Download Location Translocation**: A user inquired about changing the download location for models in **LM Studio 0.3.20**, to which another member shared the [official documentation](https://lmstudio.ai/docs/app/basics/download-model#changing-the-models-directory).
   - The response clarified that you can't change just the download location separately from the model directory, and that you must move the entire model folder.
- **Remote LM Studio setup needs reverse proxy**: A user wanted to use their **PC as host** and their **phone** being able to connect, but another user mentioned that you can't really do a remote setup with **LM Studio** currently; one can use reverse proxy for this, though that's still local network.
   - They linked to [LM Studio Remote](https://lmstudio.ai/lmstudio/remote-lmstudio) and stated that a **remote client plugin** would be available in the next major update.
- **Debate About Best LLM + Quantization**: Discussion involved tier lists, model sizes (**8B, 22B, 80B**), and quantization to make models smaller, as well as the suggestion that the most popular model atm is the **Qwen3** models.
   - Hardware limitations were discussed: the max model size you can run will be determined by your hardware, and depends on what you want out of the **LLM**.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1398270199119216730)** (17 messages🔥): 

> `4090, iGPU for video output, Budget-friendly GPUs, 5070ti, VRAM limitations` 


- **iGPU Enables Multi-GPU Nirvana**: A member suggested buying another **4090** and enabling **iGPU** to use it for video output.
- **Budget GPU List Sought After**: A member inquired about a list of **budgets** and **GPUs** that fit into those budgets, asking about workstation versus consumer cards.
- **5070ti User Waits for Super**: One member with a **5070ti** mentioned they will either upgrade when the **Super models** come out or wait for the next generation, also noting that *16GB of VRAM isn't much*.
   - They mentioned running **32B models** at a relatively slow **5 tokens/s**.
- **VRAM bottleneck plagues all**: A member suggested shrinking models down to **Q3** to fit everything in **VRAM**, noting that only the **3090** and super expensive cards have **24GB+ VRAM**.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1398093776248901632)** (74 messages🔥🔥): 

> `Validation Set Corruption, Algoverse AI program, Human-like AI Personality, Hyperparameter Gaming, SOAR program vs Algoverse` 


- **Data Scientists Game Validation Accuracy**: A member discussed how data scientists game validation accuracy by reporting the **last epoch** or **best accuracy over the training run** and hyperparameter sweeps are done over the validation accuracy.
   - Another member added that stopping at the best epoch is another way of gaming the system and suggested that applying **corruption to the validation set** could be a solution.
- **AI Homie System Prompt Hacks**: A member asked for tips on system prompt engineering to create a *more human-like personality* for an AI friend.
   - Another member suggested putting what you just wrote down in the system prompt, and you can ask some LLM to refine it.
- **Researchers Ponder Algoverse AI Program**: A member inquired about the **Algoverse AI program** as an alternative for those not accepted into SOAR.
   - It was noted that it costs **$3,325**, which is a major downside, with claims that its not obvious how much of how far you get is on your own merit as opposed to the work/assistance of others whom you paid.
- **SOAR Program is mega competitive, Algoverse is BackUp Plan**: Members discuss how the **SOAR program** is mega competitive, but **Algoverse** is good as a backup plan.
   - They also mentioned that Algoverse never released their stats, and hiring managers tend not to dig into backgrounds, and there is a cohere research division server with events and talks, but it's very eutz focused.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1398147077984489472)** (17 messages🔥): 

> `HRM loops, Causality in models, KV Caching strategies, Qwen finetuning` 


- **HRM Loops are non-causal**: The key point is that the **num_segment** is dynamic in training for **HRM** so it's not causal and doesn't even have a kv cache.
   - One user noted *what had been confusing me is I thought it was causal, but it's not*.
- **Debate emerges: KV Caching for causal loop models**: Members debated the feasibility of KV caching with causal loop models, considering architectures like `prev toks -> hrm loop -> next tok`.
   - One member argued that *the z values are the only variables carrying state* so caching wouldn't be useful but another member suggested caching the *input emb*'s kv when using xattn in L_module.
- **HRM's latent space replaces VLM visual tower**: One member is considering using **HRM** as an encoder whose latent space can be an initial input into a decoder (RNN or Transformer), essentially replacing the visual tower of **VLM**.
   - The idea here is to decouple *outputing* and *reasoning*.
- **Seeking advice on Qwen3 finetuning**: A member asked for advice on hyperparameter choices when finetuning **Qwen3**.
   - Another member responded that autoregressive generation is extremely expensive without a full cache.


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1398129718334722199)** (3 messages): 

> `Security Vulnerability Reporting, Async Checkpointing for NeoX` 


- **Security Vulnerability Reporting Path Identified**: A member reported finding a security vulnerability in the **EleutherAI/gpt-neox** repo.
   - Another member suggested emailing **contact@eleuther.ai** to report the issue.
- **Interest Expressed in Async Checkpointing for NeoX**: A member inquired about the status of **Async Checkpointing** for **NeoX**.
   - They expressed interest in working on it as a learning experience, pending confirmation that it's not already being developed by someone else.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1398018243603726459)** (69 messages🔥🔥): 

> `Qwen3 Model, GPT-5 Launch, Claude Opus Rate Limits, Nitter Rate Limiting, Tidbit AI Tool` 


- **Qwen3 Model Anticipation Builds**: Junyang Lin (@JustinLin610) announced the upcoming release of the **qwen3-235b-a22b-thinking-2507 model** on [X](https://xcancel.com/JustinLin610/status/1948456122228380128), generating community excitement.
   - Followers inquired about a **Qwen3 Omni model**, smaller variants (e.g., **30B**), and availability in regions such as an **EU mobile app**.
- **GPT-5 Launch Deets Leaked**: It was reported that **OpenAI** is preparing to launch **GPT-5** in August, as covered in [The Verge](https://www.theverge.com/notepad-microsoft-newsletter/712950/openai-gpt-5-model-release-date-notepad) and [The Information](https://www.theinformation.com/articles/openais-gpt-5-shines-coding-tasks).
   - An open-source model aims to reach **O3 level** performance and launch before **GPT-5**.
- **Anthropic's Claude Opus Gets Rate Boost**: **Anthropic API** has increased **Claude Opus 4** rate limits across all tiers, according to [this X post](https://xcancel.com/alexalbert__/status/1948442271969673469).
- **Nitter Hit by Rate Limits**: Users encountered a **429 error (Too Many Requests)** when trying to access content via a **Nitter** instance at [xcancel.com](https://xcancel.com/healthcareaiguy/status/1948426264559403204?s=46).
   - The instance is either fully rate-limited or lacks authentication tokens, preventing access, and users were advised to switch instances or retry later.
- **Stacklok Survey Exposes AI Code Gen Tool Adoption**: A survey from **Stacklok** provided data on AI code generation tools, available at [stacklok.com](https://stacklok.com/static/2025.06-stacklok-state-of-ai-codegen.pdf).
   - The data indicates adoption across a range of alternatives; however, some skepticism was expressed about the **AWS Q Developer** adoption stat.


  

---


### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1398106504145535048)** (1 messages): 

> `Psyche office hours, Discord event space` 


- ****Psyche** Office Hours Starting Soon!**: The **Psyche** office hours are beginning in 5 minutes, according to a [Discord announcement](https://discord.com/channels/1053877538025386074/1222014354338222181).
   - Further details can be found on [X.com](https://x.com/NousResearch/status/1947708830126903707) and the [Discord event](https://discord.com/events/1053877538025386074/1395375046439997511).
- **Join the Discord event space**: Join the Discord event space in the events channel: [Discord Link](https://discord.com/channels/1053877538025386074/1222014354338222181).
   - Psyche office hours begins in 5 minutes!


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1398032477007777923)** (46 messages🔥): 

> `Stage Channel Creation, Psyche Office Hours, Hermes 3-405B, Anthropic Reliability, Atropos Updates` 


- **Stage Channel Under Consideration**: Members considered creating a **stage channel**, similar to VC channels but with only selected people able to talk.
   - A member noted that there is already [one available](https://discord.com/channels/1053877538025386074/1222014354338222181).
- **Psyche Office Hours recording available**: The [recording of the Psyche office hours](https://www.youtube.com/watch?v=0t4r--rrz5Y) is now available, though a few minutes are missing in the middle.
   - The office hours event started at [this link](https://discord.com/events/1053877538025386074/1395375046439997511).
- **User requests Hermes 3-405B to return**: A member requested for the **Hermes3-405B free version** to be brought back on openrouter.
   - A member responded that it was *lambda* but they will try.
- **Members Complain About Anthropic Reliability**: Members discussed reliability issues with **Anthropic**, with one reporting frequent **522 errors**.
   - Another member quipped they *learned that error code from using anthropic*.
- **Atropos gets updated**: Users discussed [Atropos](https://x.com/NousResearch/status/1945932488960008441) recent big updates.
   - A member suggested reading the second half by Shunyu Yao.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1398293546229960835)** (2 messages): 

> `Dataset Publishing, Unknown Architecture` 


- **Dataset Publishing in the Works?**: A member expressed interest in a dataset and inquired about plans for publishing it.
   - They noted that the idea was interesting but expressed uncertainty regarding the **underlying architecture** of the dataset.
- **Architecture Still Shrouded in Mystery**: Details regarding the specific architecture of the dataset remain unclear.
   - The discussion highlighted an **unresolved question** about the architecture, with the original poster indicating uncertainty about its nature.


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1398256130647396423)** (11 messages🔥): 

> `Codex I, Nvidia Cutlass, Higgs Audio TTSEE, Philosophical AI discussion` 


- **Codex I Diagnostic System is Live**: **Codex I**, *a symbolic diagnostic system for intelligence under distortion*, is now live ([codex_1.pdf](https://cdn.discordapp.com/attachments/1132352574750728192/1398256130597322882/codex_1.pdf)).
   - It is conceptually linked to **neurosymbolic scaffolds**, **narrative entropy management** and **meta agent stabilization under adversarial compression**.
- **Nvidia's Cutlass Linear Algebra**: A member found an interesting link about **Nvidia's Cutlass** while checking **flashMLA** which gave attribution to cutless ([developer.nvidia.com](https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/)).
- **Higgs Audio's New TTSEE**: **Higgs Audio** released a new **TTSEE** ([github.com](https://github.com/boson-ai/higgs-audio/)), which is supposedly easy to set up.
   - However, the *multispeaker stuff is still not as good as dia* but the *single speaker stuff seems greater* and *does not seem to be able to do (Cough) and (laugh) like dia*.
- **Algorithm culture shapes behavior**: A member found **Codex I** as a powerful critique of how algorithmic culture shapes our behavior.
   - He admitted that he got *lost* because *the highly philosophical and abstract nature of the writing*.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1398293546229960835)** (2 messages): 

> `Dataset Architecture, Dataset Publishing` 


- **Dataset Architecture Piques Interest**: A member expressed interest in a dataset, wondering about its architecture.
   - They admitted uncertainty regarding the dataset's design.
- **Dataset Publishing Plans Requested**: A member inquired about plans to publish the dataset.
   - They showed interest with a custom emoji.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1398295452557775060)** (16 messages🔥): 

> `AutoCite app feedback, VSCode vs Overleaf, Hackathon sleep arrangements, NYC Hackathon` 


- **AutoCite App Elicits Encouragement**: A user developed a citation app called **AutoCite** and asked for feedback and ideas: [autocite.vercel.app](https://autocite.vercel.app/).
   - One user suggested *doubling down* by forking **VSCode** into a free website, specializing in **Overleaf** functions with an integrated **AI chatbot**.
- **VSCode Copilot Eclipses AutoCite?**: A user found **AutoCite** to work well, but ultimately preferred using **VSCode's** built-in **Copilot chat extension** for similar results.
   - They suggested **AutoCite** target academia-related servers and university communities for more relevant feedback, and even pitched it to University communities.
- **Hackathon Sleepover?**: A user asked about sleeping arrangements at the hackathon: *Will the hackathon have a place to sleep?*
   - Others pointed out it's common for hackathons to be overnight with attendees either bringing a **sleep pack** or just foregoing sleep altogether.
- **NYC Hackathon Sparks Excitement**: Enthusiasm erupted for the upcoming **NYC hackathon**, with one user lamenting the exorbitant flight fares.
   - Another user inquired about the number of available spots.


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1398265460352614461)** (10 messages🔥): 

> `Triton Masking, Triton block_ptr deprecation, Triton vector @ matrix multiplication, GEMV Kernel, GEMM implementation` 


- **Triton Evades Branching and Skips Memory Transactions**: When using `tl.load(ptr, mask=mask_vec)` in Triton, there is *no branch divergence*, and if `mask=false`, **no memory transactions are issued**.
- **`block_ptr` deprecated**: `block_ptr` was the Triton team's initial attempt at tensor descriptors (before they knew what TMAs would look like) but *will be deprecated*.
- **GEMV Kernel implores optimal grid**: When performing vector @ matrix multiplication in Triton, the recommended approach involves using `tl.sum(a.reshape((BLOCK_SIZE_K, 1), can_reorder=False) * b, axis=0, keep_dims=True)`, noting the need to write a **proper GEMV kernel** to use this efficiently.
- **GEMM implementation matters to mobicham**: For efficient vector @ matrix multiplication, it is important to loop over K like in a **GEMM implementation**.
- **Optimize Data Loading for Faster Kernels**: A member suggested optimizing data loading by using separate **BLOCK_SIZE_K / BLOCK_SIZE_N + autotune** for faster kernels, also consider trying `y.T.contiguous().T` depending on the settings to potentially improve performance.
   - The member noted that the cost of `tl.sum` is not as important here, and that the kernel is memory bound.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1398076154564444280)** (1 messages): 

> `Nsight Copilot` 


- **Nvidia Releases Nsight Copilot**: Nvidia has released **Nsight Copilot**, a tool designed to assist developers.
   - More information is available on the [Nvidia developer website](https://developer.nvidia.com/nsight-copilot).
- **Nsight Copilot is now available**: Developers can now access **Nsight Copilot** from Nvidia.
   - Check it out at the [Nvidia developer website](https://developer.nvidia.com/nsight-copilot).


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1398307648272596992)** (2 messages): 

> `Torch uint8 workaround, Triton` 


- **Torch uint8 workaround surfaces**: A member found a dirty workaround is to call `.view(torch.uint8)` on the **e8m0 inputs** before calling the custom kernel.
   - Another member responded that *"That's how it is supposed to work with **Triton** actually"*.
- **Triton Loves uint8**: A member reported that **Triton** works best with `.view(torch.uint8)` calls.
   - The user stated that this is how the library is *"supposed to work"*.


  

---


### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1398360915560239286)** (1 messages): 

> `NYC Hackathon, Jane Street, Tri Dao, Soumith Chintala, Coreweave` 


- **NYC Hackathon Collabs with Jane Street!**: GPU MODE is hosting its first **NYC hackathon** in collaboration with **Jane Street** on **September 6**.
   - Unlike typical hackathons, participants will deploy *real models* to the market, emphasizing the importance of rapid model deployment, not just speed.
- **Optimize End-to-End Architectures**: The hackathon won't be just about kernels and transformers, the architecture will be more unique and you'll really have to think about your optimizations in an end to end way.
   - The organizers teased keynotes by **Tri Dao** and a panel with the OG PyTorch team **Soumith Chintala**, **Sam Gross**, and **Gregory Chanan**.
- **Generous Compute Offered by Coreweave and Northflank!**: **Coreweave** and **Northflank** are providing generous compute for the hackathon.
   - Those interested are encouraged to [register before August 17](https://www.janestreet.com/join-jane-street/programs-and-events/gpu-mode-hackathon/).


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1398355967342809278)** (2 messages): 

> `ChipBenchmark, Tilderesearch Tweet` 


- **ChipBenchmark Website Surfaces**: A member shared a link to [ChipBenchmark](https://www.chipbenchmark.com/), presumably for comparing different **chip performances**.
   - No specific discussion followed, but the link was dropped in the **cool-links channel** for future reference.
- **Tilderesearch Tweet Shared**: Someone posted a link to a tweet from **Tilderesearch** found at [https://x.com/tilderesearch/status/1948818857214574652](https://x.com/tilderesearch/status/1948818857214574652).
   - The tweet's content wasn't detailed in the channel, but it was flagged as noteworthy by inclusion in **cool-links**.


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1398427548215410828)** (1 messages): 

> `AMD Global Hiring, US-Based Interns` 


- **AMD Expands Global Full-Time Hiring**: **AMD** is open to hiring full-time employees globally, specifically in locations where they have an existing office.
   - This move allows AMD to tap into a diverse talent pool worldwide, leveraging its established infrastructure for seamless integration.
- **AMD Focuses US for Intern Recruitment**: **AMD** is targeting candidates based in the **United States** for their internship positions.
   - This localized approach for internships may aim to foster early-career talent within the US, potentially feeding into full-time roles later.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1398066938902614079)** (2 messages): 

> `HF Hub vs Repo for Model Weights` 


- **HF Hub Favored Over Repo for Model Weights**: A member pondered if uploading to [HF Hub](https://huggingface.co/) is preferable to storing model weights directly in a repo, questioning the conventionality.
   - They suggested it seems *slightly unconventional to have model weights just sitting in a repo*, advocating for pulling weights from an online source instead, noting that *HF is just a git repo*.
- **Discussion on Model Storage Best Practices**: The conversation revolves around the optimal method for storing and accessing model weights, considering both local repositories and centralized hubs.
   - The user's preference leans towards online hosting solutions like HF Hub for accessibility and perceived best practice, contrasting with direct storage in a Git repository.


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1398173960918667274)** (3 messages): 

> `Weight Pruning Research, Wanda & Wanda++ for weight pruning, Adaptive Pruning and Tuning (APT), Custom Kernels like Squared-ReLU` 


- **Weight Pruning Research Asked For**: A member inquired about applying modern research for weight pruning, citing the [CODEML'2025 paper](https://arxiv.org/abs/2507.16099) and the `torchao/sparsity/` & ``torchao/prototype/sparsity/` codebases.
   - The member specifically asked about the application of **Wanda** and **Wanda++** for weight pruning and the integration of **Adaptive Pruning and Tuning (APT)** with **LoRA** for efficient fine-tuning.
- **Wanda++ Ticket Opened For Better Performance**: The user noted that *"Wanda : A simple and effective LLM pruning approach"* is already applied for weight pruning, with a ticket opened for better performance following the publication of [Wanda++](https://arxiv.org/abs/2503.04992).
   - The user noted that they opened a [PR for this](https://github.com/pytorch/ao/pull/2537).
- **Adaptive Pruning and Tuning Gains Traction**: The user proposed *"[APT: Adaptive Pruning and Tuning](https://icml.cc/virtual/2024/poster/32904)"* integrated **LoRA** and adaptive pruning for efficient fine-tuning as a choice for [TorchAO-#134](https://github.com/pytorch/ao/issues/134#issuecomment-2061660003).
   - APT offers a method for more efficient fine-tuning through adaptive pruning and **LoRA** integration.
- **Squared-ReLU kernels Future Plan**: The user inquired about applying more custom-kernel like **Squared-ReLU** cases, referencing [TorchAO-#1920](https://github.com/pytorch/ao/issues/1920) and seeking clarification on future plans.
   - It was unclear to the user whether there are confirmed plans for integrating custom kernels.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1398361817184469003)** (1 messages): 

> `Warp specialization, CuTeDSL Tile Scheduler, Persistent GEMM kernel, Hopper TMA and WGMMA, Cluster-based TMA load` 


- **Persistent GEMM Kernel Surfaces on Hopper**: A new blog post details writing a **persistent GEMM kernel** leveraging **Hopper's TMA and WGMMA** in the **CuTeDSL**, available on [GitHub](https://github.com/simveit/cute_persistent_kernels).
   - The post also explains turning a simple **TMA load** into one that leverages the concept of **clusters and multicast memory transfer**; read it [here](https://veitner.bearblog.dev/persistent-gemm-in-cutedsl-on-hopper/).
- **Warp Specialization Def Explained**: **Warp specialization** is defined as using different warps (groups) for Producers and Consumers in performant **GEMM kernels**.
   - The blogpost also mentions that the **Tile Scheduler abstraction** in **CuTeDSL** can be used to write persistent kernels.


  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1398114120460668950)** (1 messages): 

> `bf16 high error rates, matmul kernels` 


- **bf16 Kernels Yield High Error Rates**: A member finds that all kernels using **bf16** on `matmul/educational` have a pretty high error rate, often with max errors in the `inf`s.
   - The member inquired if this behavior is expected for all **bf16** matmuls/ops.
- **Matmul Kernel Errors**: High error rates were observed in matmul kernels using **bf16** format.
   - The user is investigating the `matmul/educational` kernels and seeks insights into the expected behavior of **bf16** operations.


  

---


### **GPU MODE ▷ #[status](https://discord.com/channels/1189498204333543425/1343350424253632695/1398355638219964447)** (1 messages): 

> `VS Code syntax highlighting, PyTorch Load Inline Highlighter` 


- **Syntax Highlighting Arrives to VS Code!**: Users of `load_inline` for writing kernels can now get syntax highlighting in VS Code via the [PyTorch Load Inline Highlighter](https://marketplace.visualstudio.com/items?itemName=msaroufim.pytorch-load-inline-highlighter).
   - The tool was *quickly put together* and the author is seeking feedback on its usability and potential for productionizing.
- **Author requests feedback on PyTorch Load Inline Highlighter**: The author of the [PyTorch Load Inline Highlighter](https://marketplace.visualstudio.com/items?itemName=msaroufim.pytorch-load-inline-highlighter) is seeking feedback from users.
   - The feedback will determine whether to *productionize it*.


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1398197125782966413)** (13 messages🔥): 

> `Sonnet Benchmarking, Action Space Context, OpenRouter` 


- **Sonnet Benchmark Plagued by API Errors**: Sonnet 4 benchmarking using [terminal-bench](https://github.com/laude-institute/terminal-bench) is facing issues with excessive **API errors (529)**, resulting in only one iteration every 20 minutes, making the process intractable with only two API keys.
   - It was noted that a workaround for `can_place_entity` was achieved by using `build_check_type manual`, which might need to be adopted in v2.
- **Action Space and Context Size**: It was suggested to only test v0.3.0 with the new action space, given that the **context will be much smaller** with fewer actions.
   - However, it was countered that testing the current actions is important for running ablations on the new action space to have a baseline for comparison.
- **OpenRouter Used for Benchmarking**: To avoid **API errors**, previous lab tests were conducted using **OpenRouter** with 12 environments running concurrently.
   - Currently, benchmarking is being done with only one environment per key, resulting in two environments running simultaneously.


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1398268697625231401)** (1 messages): 

> `CuTe, shared memory, swizzle, Layout, make_tiled_copy` 


- **Swizzle Causes Partition Issues in CuTe**: A member is facing partitioning issues with **CuTe** after applying **Swizzle<3, 2, 5>** to a shared memory region of size **19x128**, and suspects that the issue arises because **19** is not divisible by **8**, the repeat factor introduced by the swizzle, as discussed in [Lei Mao's blog](https://leimao.github.io/blog/CuTe-Swizzle/).
- **Swizzled Layout Incompatibility**: The member reported that after applying the swizzle, they cannot partition the layout using either **make_tiled_copy** or **local_partition** and suspect the root cause is the **19x128** size.
   - They included a [shared19128_memory_bank_ids.pdf](https://cdn.discordapp.com/attachments/1362196854460383353/1398268697079976046/shared19128_memory_bank_ids.pdf?ex=6884beb3&is=68836d33&hm=897eb5992a9f86673955f04914cf34fae56cfef9583e042b72cf6870b1b886ce&) for reference.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1398034964699938917)** (22 messages🔥): 

> `NeurIPS reviews, Karpathy on academic paper inflation, Alternative paper platforms, LLM Context Management, Downvote Politics` 


- **NeurIPS Review Reflections**: Members shared their experiences with **NeurIPS reviews**, with one asking if anyone received *"any good NeurIPS reviews?"*
   - The conversation quickly shifted to the broader issues of academic paper inflation and the scalability of academic institutions.
- **Karpathy laments academic paper inflation**: A member shared a [2016 tweet from Andrej Karpathy](https://x.com/2prime_PKU/status/1948549824594485696) humorously commenting on how *out of hand* the volume of academic papers was becoming.
   - Another member linked a [Hacker News discussion](https://news.ycombinator.com/item?id=11319493) from the same period.
- **Brainstorming Alternative Paper Platforms**: A member suggested creating a *"Youtube-Twitter-TikTok like platform for papers"* with **upvotes** (but no downvotes) and **categories** to combat academic paper inflation.
   - The user detailed a category ranking idea, and suggested that instead of *circlejerking around the sad graduate pizza* to *build shit*.
- **LLM Context Manager launch**: A member announced they *built something!* a [LLM Context Manager](https://github.com/theabhinav0231/LLM-Context-Manager), described as *an inference optimization system for conversations*.
   - It employs **branching** and a *novel algorithm contextual scaffolding algorithm (CSA)* to manage context and prevent *context pollution/context rot*.
- **Downvote debacle**: Members discussed the role and potential pitfalls of **downvotes**, particularly how they can become politicized and weaponized in tightly networked communities, drawing from a Web3 experiment where *groups* used downvotes to target each other.
   - A member argued that downvotes are not inherently political and that negative feedback is essential, pointing to **Amazon**'s success as an example.


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1398100112676356247)** (9 messages🔥): 

> `Paper Discussion, Arxiv Sharing, Mathy Papers, Large-Scale Evening Meeting` 


- **Community Discusses Paper Sharing Protocols**: A member inquired about the proper way to share a paper with the community without causing annoyance and another member suggested that sharing the [ArXiv link](https://arxiv.org/) is appropriate if the paper has been archived.
   - They recommended sharing the **ArXiv link** and contacting a specific user to discuss it in the daily paper discussion.
- **Mathy Paper's Engineering Implications**: A member shared a [paper link](https://arxiv.org/abs/2503.13791) stating that the paper is more **mathy**, and its engineering implications might not be immediately apparent.
   - The member described it as *a generic hammer for learning problems* applied to demonstrate learning some toy dynamical systems.
- **Large-Scale Evening Meeting Topic Planning**: A member planned to inquire about the appropriateness of discussing a paper in the large-scale evening meeting at `<t:1753552800:T>`.
   - This indicates consideration of a suitable platform for discussing the paper with the community.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1398038827758653593)** (9 messages🔥): 

> `Grok Training Data, DEI in AI Models, Industrial Policy, Gemini Model Controversy, Imagen-4 and Gemini 2.5` 


- **Grok May Have Trained on Government Data Hoard**: A member wondered if **Grok** trained on files when **Elon** got access to the government's hoards of data ([link to X post](https://x.com/vitrupo/status/1948287716279611670)).
- **White House Prevents "Woke AI"**: The White House has issued guidance to prevent *'woke AI'* in the federal government ([link to White House memo](https://www.whitehouse.gov/presidential-actions/2025/07/preventing-woke-ai-in-the-federal-government/)).
   - The memo states that *LLMs shall prioritize historical accuracy, scientific inquiry, and objectivity*.
- **Gemini's DEI prioritization led to inaccuracies**: The White House memo noted that an AI model changed the race or sex of historical figures, including the **Pope**, the **Founding Fathers**, and **Vikings** when prompted for images because it was trained to prioritize DEI requirements at the cost of accuracy.
- **Google's Old Gemini Model Criticized, Newer Models Improve**: A member noted that the **older Gemini model** is being mentioned despite **Google** already taking it down due to backlash, claiming it's a *'nothing-burger nowadays'* with newer models available.
   - They added that even **Google's** latest image-gen (**Imagen-4**) and the latest version of **Gemini 2.5** text gen don't have this issue.
- **Government Should not shape Model Ideological Bias**: One technology policy analyst said the *'biggest error in the order is the use of government procurement power to shape model ideological bias'*.
   - They claimed that if the policy successfully shapes American models, the US will lose international customers who won’t want models shaped by a foreign government’s whims.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1398024003553198120)** (17 messages🔥): 

> `Spam bots, Server issues, Vibe Coding AI, Scientific Manus paper` 


- **Spam Bots Invade**: Users reported seeing **spam bots** on the server and requested moderation.
   - A moderator responded that the **messages were removed** and the **account banned**, encouraging users to tag moderators for suspicious accounts.
- **Sandbox Snafu**: A user reported a *"Failed to resume sandbox"* error and a **502 Bad Gateway**, seeking help with file and session recovery.
   - Another user mentioned the company is undergoing **major changes** and is **short-staffed**, suggesting potential instability.
- **Vibe Coding AI Challenge**: A user shared a link to a [challenge](https://nas.io/microwaves/challenges/build-your-mvp-product-using-vibe-coding-ai-coding-skills-challenge) for building an **MVP product** using **Vibe Coding AI coding skills**.
   - They shared the link in a joking context.
- **Scientific Manus Ascends**: A user posted a link to a [scientific paper](https://arxiv.org/html/2505.02024v2) referring to it as *Scientific Manus*.
   - The title of the paper has not been identified in the messages.


  

---


### **Cohere ▷ #[🔌-api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1398106777333006336)** (11 messages🔥): 

> `Helicone.ai integration with Cohere models, Command R+ vs. Command A, On-prem deployment of Cohere models` 


- **Helicone.ai Lacks Native Cohere Support**: A user inquired about using **Cohere's Command R+** or **Command R7B** with [Helicone.ai](https://www.helicone.ai/) for observability, but a Cohere representative stated they *don't natively support* or have partnerships with Helicone.ai.
   - The user was advised to contact Helicone's support directly for assistance.
- **Cohere Touts Command-A as R+'s Superior Successor**: Cohere promotes [Command-A-03-2025](https://docs.cohere.com/docs/command-a) as their *latest and best model* with SOTA agentic capabilities, succeeding **Command R+**.
   - It is described as [having enhanced capabilities](https://cohere.com/blog/command-a) and suitable as a *general thinking assistant*.
- **Cohere offers On-Premise Enterprise Deployments**: A user noted Command A's performance with fewer parameters and a Cohere representative confirmed [on-premise enterprise deployments](https://cohere.com/deployment-options) are available.
   - This is particularly relevant for consumer deployment as a general thinking assistant.


  

---


### **Cohere ▷ #[👋-introduce-yourself](https://discord.com/channels/954421988141711382/1346635816629178410/1398106349677838468)** (3 messages): 

> `Crafted Logic Lab, Cognitive OS Assistant, Helicone.ai gateway, Humanist AI Values` 


- **Crafted Logic Lab crafts Cognitive OS Assistant**: A founder from **Crafted Logic Lab** is developing a new type of **cognitive OS based assistant** that is patent pending.
   - They developed their own tooling using **Swift**.
- **Cohere aligns with Humanist AI Values**: A founder expressed a very positive sentiment on Cohere, because it's a non Silicon Valley company that seems to be more aligned with their **Humanist AI values** than the big providers.
   - They find Cohere a **frontier-class model** that is very much under-known to use as their substrate.
- **Seeking Technical Info on Helicone.ai gateway**: A founder seeks technical information on items not documented in the Cohere such as the **Helicone.ai gateway calls for observability**.
   - They are also seeking which of the models between **th-8-2024** and current is the newer version.


  

---


### **Cohere ▷ #[🧭-status-feed](https://discord.com/channels/954421988141711382/1346652044181897307/1398346426588594247)** (1 messages): 

> `Cohere Model Outage, Command models down` 


- **Cohere Models Experience Full Outage**: A [status update](https://ift.tt/WKY7QNq) indicates a full outage affecting multiple Cohere models including **command-light**, **chat**, **command-r-plus**, **command-r-082024**, **command-r-plus-082024**, **command**, **command-r**, **command-r7b**, and **command-a-03-2025**.
   - The incident is currently under investigation as of **July 25, 2025**.
- **Cohere Infrastructure Meltdown**: All **command** models are currently offline.
   - The [Cohere Status Page](https://ift.tt/Ve8Pqgf) has been updated.


  

---


### **Cohere ▷ #[🔬-research](https://discord.com/channels/954421988141711382/1384974112841269399/1398115712240713839)** (1 messages): 

> `Command R+, Humanity's Last Exam test, Hummingbird Anatomy Question` 


- **Command R+ Tackles Cognitive Flexibility Test**: A member reported testing a system based on **Command R+** on the [Humanity's Last Exam](https://cdn.discordapp.com/attachments/1384974112841269399/1398115711611834568/message.txt?ex=6884d8f9&is=68838779&hm=ebefb364e4728e8f090566f5b3578a895151607fbffdacb5cb2146f148227009) test, which assesses for both correct answers and **cognitive flexibility**.
- **Agent's Take on Hummingbird Anatomy**: An agent was asked a detailed question about the number of paired tendons supported by a sesamoid bone in hummingbirds, admitting it lacked expertise in ornithology and providing a speculative inference based on general anatomical knowledge, guessing *at least two paired tendons directly involved in tail movement*.


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1398157191625572443)** (8 messages🔥): 

> `Chat GPT agent login issues, Missing Share button, Metadata in Source` 


- **GPT Agent Faces Login Troubles**: A member is facing issues with their **Chat GPT agent** failing to sign into **Notebook LM**, encountering an error possibly due to the browser being controlled by a virtual machine or bot, as shown in the [attached image](https://cdn.discordapp.com/attachments/1124403655819415592/1398174200598102076/image.png?ex=68850f72&is=6883bdf2&hm=ed9d7b0652d7c64b225d3fcad2e5d055f323bb31d4f819a7745613f39879ed9d&).
- **Vanishing "Share" Button Baffles User**: A user reported that they are not seeing the "Share" option in Notebook LM, thus, they are unable to share created notebooks.
- **Metadata Magic Improves Sourcing**: A member is using **metadata** effectively in the Source, using brackets to avoid direct document references, as shown in the [attached screenshot](https://cdn.discordapp.com/attachments/1124403655819415592/1398375829578317834/Screenshot_20250725-124459.png?ex=6885227a&is=6883d0fa&hm=51849656623396ded870daae1f8ebf505dadfa3f1710b00e711154e9af9d2e0f&).


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1398157242242699346)** (7 messages): 

> `Podcast Generation, File Uploading Error` 


- **Podcast Generation Pointers**: A member inquired about generating a **60min long podcast**.
   - Another member suggested checking the [use case channel](https://discord.com/channels/1124402182171672732/1124403655819415592) and linked a [YouTube Short](https://youtube.com/shorts/VRG-aGu1ihE?si=EDl8DyMfKP1jwW_g) as a pointer.
- **File Uploading Flounders**: A member reported a recent file uploading error on both the free and pro versions of the platform, and asked if there was a workaround.
   - The member found a fix themselves: *mobile App uploads work*, so the desktop version needs to be fixed.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1398104588610961468)** (8 messages🔥): 

> `GPT5, Textual 5.0.0, Qwen3-coder, Aider and testing` 


- **GPT5: A Niche Replacement?**: A member questioned whether *closed AI* would replace **GPT5**.
   - It was implied that **GPT5** may be a niche product compared to the closed source alternatives.
- **Textual 5.0.0 Drops**: A member announced the release of [Textual 5.0.0](https://github.com/Textualize/textual/releases/tag/v5.0.0), noting it contains final markdown streaming content.
   - Textual is a Rapid Application Development (RAD) framework for Python.
- **Qwen3-coder Wows**: One member exclaimed that **Qwen3-coder** is amazing, as no other model could produce a fully working **socks5 server in rust** according to the specification.
   - This suggests **Qwen3-coder** has superior coding capabilities, especially in Rust.
- **Aider's Testing Troubles**: A user shared their experience using **aider** for the first time, encountering difficulties in running tests, as it needed to execute commands from the terminal but stated it was *an AI assistant without access to your terminal*.
   - The user wondered whether they were expected to manually run the tests and paste the output, and also sought a way to prevent **aider** from automatically committing changes, as they preferred to handle commits themselves.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1398037504745017444)** (8 messages🔥): 

> `Agents class at Berkeley, Certificate Issues, Article Submission` 


- **Agents Class Still in the Works**: The Agents class is being offered to Berkeley students, but whether there will be a **MOOC** iteration hasn't been confirmed yet, likely announced in late August.
- **Certificate Delivery Tango**: A member reported not receiving a certificate despite having the **certificate declaration form confirmation**.
   - Staff clarified that they did not receive an article assignment submission from the member.
- **Article Submission Deadline Defeat**: A member inquired about fixing the missing **article submission** to obtain the certificate.
   - Staff apologized, stating they couldn't accommodate students who missed the deadlines due to limited staff capacity.


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1398025993310179401)** (3 messages): 

> `LLM APIs vs Production Document Parsing, Screenshot Parsing Gaps, Accuracy Issues in Parsing, Natural Language Git Commands, S3 Vector Storage Integration` 


- **LLM APIs Flounder in Production Document Parsing**: A blogpost argues that while models like **GPT-4.1**, **Claude Sonnet 4.0**, and **Gemini 2.5 Pro** obsolete traditional **OCR**, screenshot-only parsing still has critical gaps for enterprise use.
   - The post highlights [accuracy issues](https://t.co/wBQ3OtZ4ue) as a significant limitation in production environments.
- **Git Made Easy with Gut**: The tool *gut* was released: a human-in-the-loop agent in the form of a command line tool that replaces **git commands** with **natural language**.
   - Users can describe desired git actions in human language, and the agent figures out the git command, explains it, and waits for confirmation ([source](https://t.co/mVkozoQzzR)).
- **S3 Vector Storage Integrates Seamlessly**: LlamaIndex released a new **S3VectorStore integration** combining **AWS S3's** scalability with LlamaIndex.
   - This integration aims to provide agent workflows with a robust knowledge foundation that grows with user needs, offering smarter agent workflows ([source](https://t.co/71ADmwp6NF)).


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1398143996252913716)** (4 messages): 

> `Docx Parsing with Images, LlamaIndexOpenTelemetry Traces` 


- **Docx Images Elude Readers!**: A user wants to extract **text** and associated **images** from a complex **.docx** file using LlamaIndex, aiming for a list of `ImageNode` objects.
   - The user notes that `DocxReader` ignores images, and `ImageXXXReader` only handles image files, so they're considering using `python-docx` directly or embedding image URLs in `TextNode` metadata or markdown.
- **Telemetry Traces turn Trivial!**: A user is facing issues with **LlamaIndexOpenTelemetry**, where the exported traces lack attributes and aren't human-readable in their OTLP platform.
   - Another member suggested checking examples and provided a [notebook](https://github.com/run-llama/workflows-py/blob/main/examples/observability/workflows_observability_pt1.ipynb) demonstrating a custom exporter for writing readable traces to a file using **Jaeger**.


  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1398285051707260929)** (5 messages): 

> `Large Scale PEFT, LoRA/Q-LoRA hooks, Scheduler knobs, RL alignment` 


- **Torchtune User asks migration questions**: A user who is running torchtune for **large-scale PEFT** asked about migration questions regarding **LoRA/Q-LoRA hooks** and **RL alignment**.
   - The user is trying to decide whether to keep iterating in torchtune or wait for the new stack.
- **Keep iterating on torchtune**: A member suggested to continue to iterate on torchtune as it will still be supported until the newer library will be present, and linked to [Character AI's blogpost](https://blog.character.ai/character-ai-open-sources-pipeling-sft-a-scalable-framework-for-fine-tuning-moe-llms-like-deepseek-v3/).
   - The original user worried about migration friction later on.
- **New version will focus on Scale Infra Fundamentals**: The first version will be focused on the **scale infra fundamentals** and new concepts needed for **RL**.
   - Features like **LoRA** and **Multimodal** won't be available at launch, so users should keep iterating on torchtune until all of the features they need are announced/planned.


  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1398312887986028556)** (2 messages): 

> `FSDP+TP Issues, NCCL Timeout, HuggingFace DCP Saver` 


- **FSDP+TP struggles with HuggingFace DCP Saver**: A member is encountering issues with **FSDP+TP** when using the **HuggingFace DCP saver**, but reports an **NCCL timeout** on a broadcast of 1 element.
   - Due to the issues, they are reverting to full rank 0 saving, increasing the **NCCL timeout time**, and hoping checkpoints never need to be resumed.
- **DCP's Weird Timeout**: The user experiencing issues said that *DCP really shouldn’t be sending much information around*.
   - They found the timeout issue to be weird.


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1398212156758954078)** (5 messages): 

> `Memory Hallucinations, MCP Server Recommendations, Macaw Security Beta, Cloudflare Pay-Per-Crawl, Agentic Commerce` 


- **Memory Use Sparks Hallucination Concerns**: A member shared they avoid using memory in AI models, citing that *it introduces more hallucinations* because *it assumes things, and assuming is terrible*.
   - The user didn't clarify which product caused hallucinations, but warned to generally avoid.
- **Macaw Security Enforces Policies**: A member reported enrolling in **Macaw Security's** beta program, noting they could *do a scan and place some guardrails and policy enforcement*.
   - No further details were given on the types of services offered by **Macaw Security**.
- **Cloudflare Pay-Per-Crawl Ignites Agentic Commerce Discussion**: Following **Cloudflare's** pay-per-crawl announcement, a member initiated a discussion about **agentic commerce** and its implications.
   - The discussion focused on how agents can access webpages without disrupting workflows, especially with solutions like **Nekuda** and **PayOS** enabling agent wallets.
- **Agent Transactions and the Ghost of HTTPS 402**: Members considered the likelihood of agent transactions occurring in various scenarios such as **Agent to Agent**, **B2C**, **B2B**, and **website access**.
   - It was suggested that solutions like **Nekuda** and **PayOS** aim to provide the infrastructure that the **HTTPS 402 protocol** was meant to support.
- **Glama's Tool Count Glitch**: A user reported their **MCP server** on **Glama** is showing an incorrect tool count (**one instead of six**), even after republishing on the **Glama** site.
   - The issue persists only on Glama, while other **MCP server** host sites display the correct count; it is currently unknown whether **Glama** auto-updates its info and images.


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1398175488572133457)** (1 messages): 

> `MCP OAuth, OAuth flow` 


- **MCP OAuth Demystified**: A member shared an attempt to explain **MCP OAuth** for dummies, highlighting that the **MCP server** and the **Authorization server** are two completely separate entities.
   - The explanation points out that all the MCP server cares about is receiving an access token, while the Authorization server is what gives you the access token.
- **Understanding OAuth Flow in MCP**: The explanation focuses on the OAuth flow in **MCP**, emphasizing steps such as connecting to an **MCP server**, querying the `/.well-known/oauth-authorization-server` endpoint, and registering as a client via **Dynamic Client Registration (DCR)**.
   - It also includes taking the access token back to the **MCP server** for authenticated access.


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1398166569577615401)** (4 messages): 

> `GPU Recommendations, RX 9060 XT vs RX 6800 XT, Vector Storage Limitations` 


- **GPU Preferences Posed to Forum**: A member asked others what **GPU** they prefer for local **AI** use, specifically mentioning **GPT4All**.
   - He is deciding between a **RX 9060 XT 16GB** and a **RX 6800 XT**.
- **RX 9060 XT offers less power**: The member stated that his research indicates the **RX 9060 XT** would have similar performance to the **RX 6800 XT** but uses half the power.
   - He also noted that the **RX 9060 XT** might be *.3 seconds slower* in reply time and *3 tokens per second slower* in reply rate.
- **Vector Storage Unsupported**: A member noted that the best solution would be **vector storage** given the model and its context size.
   - Unfortunately, he notes that **GPT4All** doesn't support vector storage.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1398438485081329684)** (1 messages): 

> `Modular's choice of Nanobind/Pybind over Cython for Python interop, Cython's limitations at scale, Approachability of Cython vs. Nanobind/Pybind` 


- **Nanobind/Pybind Chosen over Cython by Modular**: A member inquired about Modular's decision to use **Nanobind/Pybind** for **Python interop** instead of **Cython**.
   - They questioned whether **Cython** becomes less effective at larger scales, despite appearing more approachable initially due to its Python-like syntax.
- **Cython's approachability is questioned**: The user indicated that, from casually browsing, **Cython** seems more approachable, especially for a language already looking like **Python**.
   - They wonder if **Cython** starts breaking down at some scale.


  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/)** (1 messages): 

bamiji: alright then, thanks for responding
  

---


### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1398378315181457418)** (1 messages): 

> `Qwen3-Coder release, Windsurf Server Tags` 


- **Qwen3-Coder Slides Into Windsurf**: The **Qwen3-Coder** model is now live in Windsurf, costing **0.5 credits per prompt**.
   - More information on the release is available in the [full announcement](https://x.com/windsurf_ai/status/1948815609137168849) and on [Reddit](https://www.reddit.com/r/windsurf/comments/1m97c9a/qwen3coder_has_arrived/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button).
- **Server Tags Return, Surf's Up!**: Windsurf server tags are back online.
   - An image was attached, showing the new tags.


  