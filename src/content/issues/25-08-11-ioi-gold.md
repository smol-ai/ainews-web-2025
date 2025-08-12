---
id: MjAyNS0w
title: OpenAI's IMO Gold model also wins IOI Gold
date: '2025-08-11T05:44:39.731046Z'
description: >-
  **OpenAI** announced placing **#6 among human coders** at the IOI, reflecting
  rapid progress in competitive coding AI over the past two years. The **GPT-5**
  launch faced significant user backlash over restrictive usage limits and
  removal of model selection control, leading to a reversal and increased limits
  to **3000 requests per week** for Plus users. Confusion around **GPT-5**
  naming and benchmarking was highlighted, with critiques on methodological
  issues comparing models like **Claude** and **Gemini**. Performance reviews of
  **GPT-5** are mixed, with claims of near-zero hallucinations by **OpenAI**
  staff but user reports of confidence in hallucinations and steering
  difficulties. Benchmarks show **GPT-5 mini** performing well on document
  understanding, while the full **GPT-5** is seen as expensive and middling. On
  the Chatbot Arena, **Gemini 2.5 Pro** holds a **67%** winrate against **GPT-5
  Thinking**. Prompting and model behavior remain key discussion points.
companies:
  - openai
  - google-deepmind
  - anthropic
models:
  - gpt-5
  - gpt-5-thinking
  - gpt-5-mini
  - gemini-2.5-pro
  - claude
  - opus-4.1
topics:
  - reinforcement-learning
  - benchmarking
  - model-performance
  - prompt-engineering
  - model-behavior
  - competitive-programming
  - user-experience
  - model-naming
  - model-selection
  - hallucination-detection
people:
  - sama
  - scaling01
  - yanndubs
  - sherylhsu
  - ahmed_el-kishky
  - jerry_tworek
  - noam_brown
  - alex_wei
  - amandaaskell
  - ericmitchellai
  - jon_durbin
  - gdb
  - jerryjliu0
---


**Special RL is all you need?**

> AI News for 8/8/2025-8/11/2025. We checked 12 subreddits, 544 Twitters and 29 Discords (227 channels, and 30037 messages) for you. Estimated reading time saved (at 200wpm): 2237 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

We know OAI got the [IMO Gold performance last month](https://news.smol.ai/issues/25-07-21-imo-gold), so it's crazy that we kind of considered not giving the IOI result the same coverage.

These days, tweets serve as press releases, and so [Sheryl Hsu got the honor](https://x.com/SherylHsu02/status/1954966109851119921) (also of [the IMO team](https://x.com/SherylHsu02/status/1946478334013321231)) of announcing that they had placed [#6 among human coders](https://x.com/OpenAI/status/1954969035713687975):

![](https://resend-attachments.s3.amazonaws.com/PSHAbL9neaSannc)

Folks from [Ahmed El-Kishky](https://x.com/ahelkky/status/1954973043320819907) and [Jerry Tworek](https://x.com/MillionInt/status/1954977818128888311) and [Noam Brown](https://x.com/polynoamial/status/1954966398989635668) and [Alex Wei](https://x.com/alexwei_/status/1954966574408012003) reflected on the rapid progress from just 2 years ago when these systems could barely do anything in either competitive categories. Noam's thread offers the most insight into the scaffolds.

![](https://resend-attachments.s3.amazonaws.com/Pvv6mYev7K03DhI)

and Alex shared some of the challenging aspects of the test.

![](https://resend-attachments.s3.amazonaws.com/U6kzGv8wpSFqyKK)

---

# AI Twitter Recap

**The GPT-5 Launch: Performance, Naming, and User Rebellion**

- **User Backlash and Reversal on Rate Limits**: The **GPT-5** launch was met with significant user backlash, dubbed the "ChatGPT Plus rebellion" by [@scaling01](https://twitter.com/scaling01/status/1954609552810459203), over the initial restrictive usage limits for the new "Thinking" model and the removal of user control. The community pressure led **OpenAI** to reverse course, with [@yanndubs](https://twitter.com/yanndubs/status/1954621287713915192) and [@scaling01](https://twitter.com/scaling01/status/1954611571923255468) confirming the **Thinking** model limit was increased to **3000 requests per week** for Plus users. [@Teknium1](https://twitter.com/Teknium1/status/1954519089902473436) questioned the rationale behind taking model selection control away from users in the first place, while [@sama](https://twitter.com/sama/status/1954703747495649670) posted a lengthy reflection on the unexpectedly strong user attachment to specific models like **GPT-4o** and the challenges of managing user experience versus encouraging unhealthy dependence. In response to the changes, **ChatGPT** has [re-added the model selector](https://twitter.com/Teknium1/status/1954371945514049595), though [@Teknium1](https://twitter.com/Teknium1/status/1954376838110986276) noted Plus users only get **GPT-4o** as a legacy option.
- **Confusing Naming and Benchmarking**: **OpenAI's** model naming strategy for **GPT-5** has been a source of confusion, with [@scaling01 pointing out the proliferation of names](https://twitter.com/scaling01/status/1954292296704250005) like **mini**, **nano**, and **chat-latest**. This has made benchmarking difficult, with [@AmandaAskell highlighting methodological issues](https://twitter.com/AmandaAskell/status/1954276447285334151) in comparing models like **Claude** and **Gemini** on their ability to course-correct conversations. **OpenAI** was also criticized for [submitting only **GPT-5 Thinking** to leaderboards under the name "GPT-5"](https://twitter.com/deedydas/status/1954231799590301953) to narrowly beat **Opus 4.1** on **SWE-Bench**.
- **Mixed Performance Reviews**: The community is divided on **GPT-5's** performance. [@ericmitchellai](https://twitter.com/ericmitchellai/status/1954739395719807370) from **OpenAI** claims that **GPT-5** "doesn’t hallucinate basically at all," and that it's [materially better than o3](https://twitter.com/ericmitchellai/status/1954606526783799446). However, users like [@jon_durbin](https://twitter.com/jon_durbin/status/1954263916202316001) found the new models "nearly unusable," "extraordinarily confident in their hallucinations," and difficult to steer. [@gdb](https://twitter.com/gdb/status/1954693138372849963) showcases **GPT-5** as a "knowledge work amplifier" for "vibe coding," while [@jerryjliu0](https://twitter.com/jerryjliu0/status/1954293351702036712) shared WIP benchmarks showing **GPT-5 mini** performing well on document understanding, but the full **GPT-5** being "middle of the pack" and expensive. On the **Chatbot Arena**, [@scaling01](https://twitter.com/scaling01/status/1954546677185970271) notes **Gemini 2.5 Pro** has a **67%** winrate against **GPT-5 Thinking**.
- **Prompting and Model Behavior**: A key takeaway is the importance of specific prompting. [@ericmitchellai](https://twitter.com/ericmitchellai/status/1954418339536683078) and [@jeremyphoward](https://twitter.com/jeremyphoward/status/1954366856627978684) both highlighted that users should explicitly ask the model to "think hard" or "think deeply" to engage the more capable reasoning mode. A tweet retweeted by [@teortaxesTex](https://twitter.com/teortaxesTex/status/1954398794604253335) from [@karpathy](https://twitter.com/karpathy/status/1954398794604253335) observes that LLMs are becoming "a little too agentic" due to extensive benchmark-maxxing on long-horizon tasks.

**Model & Benchmark Developments**

- **Scaling Law Concerns & Open Source Momentum**: The **GPT-5** launch has fueled discussions about a potential plateau in AI progress. [@jeremyphoward](https://twitter.com/jeremyphoward/status/1954346846845129158) suggests the "era of the scaling 'law' is coming to a close," calling this **OpenAI's** "Llama 4 moment." [@gabriberton](https://twitter.com/gabriberton/status/1954596830614061187) argues that if LLMs are plateauing, large spending is no longer justified, and open-source models will become just as good as closed-source ones. This sentiment is bolstered by the success of **OpenAI's gpt-oss** models, which [@reach_vb](https://twitter.com/reach_vb/status/1954909541805801799) notes have over **5M downloads** and **400+** fine-tunes on **Hugging Face**.
- **New Chinese Models: GLM-4.5 and Qwen**: **Zhipu AI** released a technical report for **GLM-4.5**, highlighted by [@teortaxesTex](https://twitter.com/teortaxesTex/status/1954754947892850913) and [@bigeagle_xd](https://twitter.com/bigeagle_xd/status/1954763239738519618), detailing a complex post-training strategy using their **slime** framework with **SGLang** integration for efficient RL training. They also released **GLM-4.5V**, a **106B** parameter MoE for vision, which is [available on Hugging Face](https://twitter.com/mervenoyann/status/1954907611368771728). Meanwhile, **Alibaba's Qwen** team announced a [distilled 8-step Qwen-Image model](https://twitter.com/Alibaba_Qwen/status/1954337152298582288) and showcased **Qwen3-Coder's** ability to [generate SVG images](https://twitter.com/Alibaba_Qwen/status/1954879387465294304).
- **Diffusion vs. Autoregressive Models**: A series of papers comparing diffusion language models (DLMs) and autoregressive (AR) models has sparked discussion. Tweets from [@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1954242373145543134), [@giffmana](https://twitter.com/giffmana/status/1954283272424595547), and [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1954765986214871489) highlight findings that DLMs are more data-efficient, a crucial advantage as the field becomes more data-constrained.
- **Reasoning and Competitive Programming Benchmarks**: **OpenAI** announced that its reasoning system achieved a [gold medal-level performance at the International Olympiad in Informatics (IOI)](https://twitter.com/gdb/status/1954984230343282808). [@alexwei_](https://twitter.com/alexwei_/status/1954966393419599962) notes this was achieved with their general IMO gold model, showing that reasoning generalizes. [@MillionInt](https://twitter.com/MillionInt/status/1954977818128888311) emphasizes the leap from the **49th to 98th percentile** in one year without specialized training.

**Frameworks, Tooling, and Infrastructure**

- **Memory and Conversation History for Agents**: **Anthropic** announced that **Claude** can now [reference past chats to maintain context](https://twitter.com/AnthropicAI/status/1954999404387242362), a feature [@swyx](https://twitter.com/swyx/status/1954990553566941399) calls instructive for how they solve problems with transparency and user control. On a related note, **Google Cloud** provided a guide on implementing [short-term and long-term memory for AI agents using Vertex AI](https://twitter.com/dl_weekly/status/1954308710374760684).
- **LangChain Ecosystem Updates**: The **LangChain** team has been active, releasing a [practical guide on agent reliability](https://twitter.com/LangChainAI/status/1954233716487958845) to handle hallucinations and verify tool use. They also announced an [integration with Oxylabs for advanced web scraping](https://twitter.com/LangChainAI/status/1954241268114182433) and a new [**LangGraph CLI** for managing assistants from the terminal](https://twitter.com/LangChainAI/status/1954226169412493544).
- **Infrastructure and Low-Level Tools**: **whisper.cpp** is being [integrated into ffmpeg](https://twitter.com/ggerganov/status/1954988938281533532), a major development for local audio processing. On the hardware front, **AIBrix** released evaluations of **H20s** for LLM inference, focusing on [KV-Cache offloading](https://twitter.com/teortaxesTex/status/1954464993333698758). [@ostrisai](https://twitter.com/ostrisai/status/1954373246997913853) demonstrated a method to train a sidechain **LoRA** to compensate for precision loss when quantizing **Qwen Image** to 3-bit, enabling fine-tuning on consumer GPUs.
- **Keras and JAX Integration**: [@fchollet](https://twitter.com/fchollet/status/1954686735646068772) highlighted the power of combining **JAX** for performance and scalability with **Keras 3** for high-velocity development, calling the combination "pretty killer."

**AI Research & Scientific Breakthroughs**

- **Meta's Brain Modeling Victory**: **Meta AI's** Brain & AI team won [1st place at the Algonauts 2025 brain modeling competition](https://twitter.com/AIatMeta/status/1954865388749205984) with their **1B** parameter **TRIBE** (Trimodal Brain Encoder) model. This model is the first deep neural network trained to predict brain responses to stimuli across vision, audio, and text by combining pretrained representations from **Llama 3.2**, **Seamless**, and **V-JEPA 2**. [@alexandr_wang](https://twitter.com/alexandr_wang/status/1954915381656895545) congratulated the team, noting that brain modeling is a key step toward BCIs.
- **New Shortest-Path Algorithm**: A **Tsinghua** professor discovered the [fastest shortest-path algorithm for graphs in 40 years](https://twitter.com/algo_diver/status/1954423622787039379), breaking **Dijkstra's** 1984 "sorting barrier." The result received widespread attention, with a retweet from [@dilipkay](https://twitter.com/dilipkay/status/1954701721932046423) gaining over **5,500** retweets.
- **AI and Robotics**: [@adcock_brett](https://twitter.com/adcock_brett/status/1954295121694122430) predicts that humanoid robots will handle most physical tasks in the coming years and that the only limiter now is pretraining data. In a separate tweet, he notes that the [Figure robot can indeed fold laundry](https://twitter.com/adcock_brett/status/1954998149380182047).
- **Google's LangExtract Library**: **Google** released **LangExtract**, a [Python library for extracting structured data from unstructured documents](https://twitter.com/algo_diver/status/1954424008767951106) with precise source attribution.

**Broader Discourse: AI in Society**

- **AI Companionship and Mental Health**: A study from **Stanford** and **Carnegie Mellon**, shared by [@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1954226191071576552), analyzed over **1,000 [Character.AI](http://character.ai/)** users and found that heavier reliance on AI bots for companionship correlated with lower satisfaction and higher loneliness. This ties into the broader theme of user attachment, with [@sama](https://twitter.com/sama/status/1954703747495649670) expressing unease about a future where billions of people trust AI for their most important decisions.
- **The Nature of AI-Human Conversation**: In a highly-trafficked tweet, [@ID_AA_Carmack](https://twitter.com/ID_AA_Carmack/status/1954930438322954532) reflected on the difficulty of modeling natural conversation, which includes interruptions. He suggests a true solution would involve parallel streams of listening and thinking rather than a single autoregressive sequence. [@francoisfleuret](https://twitter.com/francoisfleuret/status/1955004348397916614) counters that he doesn't want an AI that interrupts, but one that sounds artificial, prioritizing clarity over simulated naturalness.
- **Skepticism, Hype, and User Adoption**: [@random_walker](https://twitter.com/random_walker/status/1954912993747128554) argues that AI adoption and behavior change are slow, regardless of how fast capabilities improve, pointing to the low usage of "thinking" models before **GPT-5's** automatic router. He contends this is a property of human behavior, not technology. In contrast, [@DavidSacks's take, retweeted by @ylecun](https://twitter.com/ylecun/status/1954411030294983052), presents a "best case scenario" where doomer narratives about rapid AGI takeoff were wrong, leading to more gradual and manageable progress.
- **Synthetic Data and Model Personality**: [@typedfemale](https://twitter.com/typedfemale/status/1954284624076767705) cautions against becoming "addicted to synthetic data," a sentiment echoed by [@scaling01](https://twitter.com/scaling01/status/1954689516314435767), who feels that overly clean synthetic data makes models like **Phi** and the new **OpenAI** offerings "shallow and void of any personality."

**Humor/Memes**

- **Industry Satire**: The most popular joke of the period came from [@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1954756616907362328), stating, "**If Jensen truly believed AGI was near, Nvidia wouldn't sell a single GPU**." Another viral tweet from [@typedfemale](https://twitter.com/typedfemale/status/1955040883499470853) joked, "**man adopts polyphasic sleep schedule due to claude code usage limits**."
- **Relatable Engineer Problems**: [@vikhyatk](https://twitter.com/vikhyatk/status/1954507093488349597) laments the cycle of engineering, from wanting to add new frameworks to resisting them as an older, tired engineer who has already memorized the **pip** commands. He also made a popular tweet about [realizing it's fine to store money as floats](https://twitter.com/vikhyatk/status/1954725001913114694).
- **GPT-5 Follies**: The launch produced a wave of memes, including the community's "rebellion" against rate limits and jokes about the model's performance on riddles. [@teortaxesTex](https://twitter.com/teortaxesTex/status/1954741943952666629) posted that "**4chan continues to launch proton torpedoes into the riddle-shaped thermal exhaust port of our much maligned Death Star**."
- **General Humor**: [@willdepue](https://twitter.com/willdepue/status/1954473883832033690) posted, "**oh you’re a rich guy? ... then how many tea fields do you own? oh none? stop talking to me**." [@AravSrinivas](https://twitter.com/AravSrinivas/status/1954290452146102576) shared a chart of personal wins with the simple caption "Last month was good."

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. gpt-oss-120b Model Performance and Benchmarks Discussion

- [**gpt-oss-120b ranks 16th place on](https://i.redd.it/0lv50zsy1dif1.png) [lmarena.ai](http://lmarena.ai/) [(20b model is ranked 38th)](https://i.redd.it/0lv50zsy1dif1.png)** ([Score: 244, Comments: 90](https://www.reddit.com/r/LocalLLaMA/comments/1mn8ij6/gptoss120b_ranks_16th_place_on_lmarenaai_20b/)): **The image is a screenshot from [lmarena.ai](http://lmarena.ai/) leaderboard rankings, showing that the open-source model gpt-oss-120b currently ranks 16th overall, outperforming several strong competitors including the 20b version of the same model (ranked 38th). This performance is highlighted in comparison to models like glm-4.5-air, indicating gpt-oss-120b's competitive standing among large language models. The post draws attention to both the accuracy and the performance: comments note that while gpt-oss-120b has 'trash' creative writing, it potentially deserves a higher ranking if not for this, and praise is given to the 20b model for offering strong capabilities at higher speed compared to Qwen 3 8b.** Commenters debate the practical intelligence and speed of the gpt-oss-20b versus Qwen and other open-source models, with some feeling gpt-oss-20b is underestimated in the community. There is also a note that creative writing ability impacts overall leaderboard ranking, even if other capabilities are strong.
    - gpt-oss-20b is observed to be an order of magnitude faster than Qwen3-8b in user tests, while being described as "way more smart," highlighting its efficiency/speed-to-capability ratio relative to similarly sized models.
    - Ranked models above gpt-oss-120b on [lmarena.ai](http://lmarena.ai/) require significantly higher compute resources, indicating that it achieves competitive benchmarks given its comparatively lower compute demands.
    - There is an open question about Qwen3's comparatively low overall ranking (#5) despite strong performance across individual categories, suggesting possible weighting, aggregation, or evaluation methodology issues within the benchmark.
- [**GPT-OSS Benchmarks: How GPT-OSS-120B Performs in Real Tasks**](https://i.redd.it/jw671veezeif1.png) ([Score: 184, Comments: 58](https://www.reddit.com/r/LocalLLaMA/comments/1mnhgt0/gptoss_benchmarks_how_gptoss120b_performs_in_real/)): **The image displays benchmark comparison results for the new GPT-OSS-120B open-weight model on real-world tasks (TaskBench), positioning it as the top performer among open models despite being 1/10th the size of competitors like Kimi-K2 and DeepSeek-R1. The post emphasizes that GPT-OSS-120B offers strong agentic (action-driven) performance, is optimal when paired with retrieval or other engineering strategies, but has weaker multi-lingual and world knowledge recall compared to closed models. Full results and benchmark methodology are linked at https://opper.ai/models.** Commenters urge comparison to GLM 4.5 and Qwen 3 models, and note that on the Aider Polyglot leaderboard (https://aider.chat/docs/leaderboards/), GPT-OSS-120B currently underperforms Kimi-K2 and R1-0528 but is faster; recent template fixes may improve its rank. Some users report that other open models feel stronger in practical use, highlighting a gap between bench rankings and subjective experience.
    - The Polyglot Aider leaderboard shows GPT-OSS-120B scoring 51.1% on real-world coding tasks, which is lower than Kimi-K2 (59.1%) and significantly lower than R1-0528 (71.4%) according to the current [leaderboard data](https://aider.chat/docs/leaderboards/). Recent changes to chat templates may improve GPT-OSS's score, with contributors actively working to address known issues.
    - Performance and ranking discussions highlight that GPT-OSS-120B is notably fast on local systems, making it potentially useful for scenarios prioritizing speed over peak intelligence. Some users note that fixes for harmony syntax issues in llama.cpp (which currently affect GPT-OSS compatibility and functionality) are close to resolution, as detailed in [this GitHub discussion](https://github.com/ggml-org/llama.cpp/pull/15181#issuecomment-3175984494).
    - There is skepticism about benchmark rankings when Grok 3 outperforms models like Kimi-K2 and O4-Mini, despite anecdotal evidence that Grok 3 performs poorly in agentic tool use. Some users question the relevance of benchmarks with non-public or non-representative evaluation data, arguing for the need to use hidden/secret test sets for more trustworthy results.

### 2. Innovative LLM Training and Distillation Approaches

- [**Training an LLM only on books from the 1800's - Another update**](https://www.reddit.com/r/LocalLLaMA/comments/1mnp5nc/training_an_llm_only_on_books_from_the_1800s/) ([Score: 194, Comments: 27](https://www.reddit.com/r/LocalLLaMA/comments/1mnp5nc/training_an_llm_only_on_books_from_the_1800s/)): **The author is training a language model from scratch using only London-based texts (1800–1875), currently leveraging the Phi-1.5 architecture (700M parameters) on an A100 GPU and scaling up to nearly 7,000 documents, sourced primarily from the Internet Archive. Early results show improvement in factual, historically grounded outputs—rather than hallucinations—despite the continued use of pretraining instead of fine-tuning as the main approach. Technical details and code are [available](https://github.com/haykgrigo3/TimeCapsuleLLM).** Top comments raise points about experimental applications (e.g., fine-tuning on historical physics/math for emergent reasoning), potential architectural limitations (questioning the choice of Phi-1.5 over newer designs like Qwen 3), and the risk of tokenization mismatches due to archaic language affecting the token dictionary when using preexisting model vocabularies.
    - One commenter raises concerns about vocabulary and tokenization when training on old texts, questioning if the model's token dictionary can effectively represent archaic language or uncommon words from the 1800s. They suggest this mismatch could hinder learning and are interested if the experimenter has observed any such issues.
    - There's a discussion about the architecture choice, with one user asking if phi-1.5 is a legacy decision and recommending the Qwen 3 series, noting that Qwen models deliver strong performance relative to their size and may be a better starting point for new projects today.
    - A user compares the model's capability to prominent benchmarks, asking if it's currently at "GPT2 level" and expressing interest in when the model might reach "GPT3 level," essentially tying training progress to widely recognized performance milestones.
- [**Created a new version of my Qwen3-Coder-30b-A3B-480b-distill and it performs much better now**](https://www.reddit.com/gallery/1mn8l69) ([Score: 149, Comments: 30](https://www.reddit.com/r/LocalLLaMA/comments/1mn8l69/created_a_new_version_of_my/)): **The poster presents a new version of their SVD-based, data-free distillation pipeline, transferring the Qwen3 Coder 480B MoE model into a Qwen3 Coder 30B architecture. Key improvements include fixing a MoE-layer distillation bug, integrating SLERP and Procrustes alignment alongside DARE for cleaner LoRA generation, and maximizing LoRA rank (2048) to better preserve information. The full 900+GB 480B model was distilled and merged into the 30B target (then quantized) in 4 hours on 2x 3090 GPUs. Scripts are open-sourced ([Hugging Face model](https://huggingface.co/BasedBase/Qwen3-Coder-30B-A3B-Instruct-480B-Distill-V2), [GitHub repo](https://github.com/Basedbase-ai/LLM-SVD-distillation-scripts)), with the author claiming marked improvements, especially for code tasks, although extensive complex code testing is pending.** Comments raise technical questions about (1) whether Flash Coder was also a distillation of 480B, (2) performance comparison with the original 30B coder, and (3) the prospect of generating language-specific distilled models.
    - Discussion centers on whether 'flash coder' was already a distillation of the 480B coder, suggesting the need for clarification on lineage and improvements over previous versions.
    - One user shares that the model delivers strong code review performance and managed to write a *simple yet correct traffic analysis application using a high performance library*. Reported throughput was 'TG close to 50t/s', which is notable given the model's size.
    - A contributor recommends model creators register their works on Hugging Face as *fine-tunes* rather than quantizations, as this helps with discoverability and proper classification within the ecosystem.

### 3. Ollama Integrations and Community Opinions

- [**I built Excel Add-in for Ollama**](https://i.redd.it/mvjwf2f81eif1.gif) ([Score: 615, Comments: 35](https://www.reddit.com/r/LocalLLaMA/comments/1mnc8lx/i_built_excel_addin_for_ollama/)): **The image demonstrates a new Excel Add-in that integrates Ollama (an LLM backend) directly with Microsoft Excel, allowing users to invoke LLM completions via a custom formula** `=ollama(A1)` **and apply system settings (temperature, model, instructions) both globally and per prompt ([image](https://i.redd.it/mvjwf2f81eif1.gif)). The add-in emphasizes that data never leaves Excel, and bulk application is possible through drag-to-fill. Developer documentation is [available here](https://www.listendata.com/2025/08/ollama-in-excel.html).** A technically notable debate emerges about alternative implementations: one commenter shares that similar functionality can be achieved via native VBA scripting and provides a link to their own solution ([ChatGPT code share](https://chatgpt.com/share/6899fe75-d178-8005-b136-4671134bc616)), suggesting existing users might not need to install an add-in if comfortable with scripting.
    - A user outlines a standard method to integrate LLM calls in Excel without third-party add-ins by leveraging VBScript and Excel's macro capabilities (ALT+F11 to add module/code). This approach calls a backend LLM server (such as llama-server) via HTTP, with adjustable IP/port configuration (e.g. "localhost:8013") and a customizable CallLLM() function to process prompts from text or cell values. The method primarily targets Windows, with noted modifications for MacOS compatibility. For direct code access, they provide a ChatGPT shared conversation as a workaround for Reddit's code formatting restrictions: https://chatgpt.com/share/6899fe75-d178-8005-b136-4671134bc616.
    - Another commenter suggests that instead of making an Ollama-specific integration, the implementation could be abstracted as a more general API call handler, allowing support for any LLM backend or inference server with a compatible API, broadening the add-in's versatility beyond Ollama.
- [**Am I the only one who never really liked Ollama?**](https://www.reddit.com/r/LocalLLaMA/comments/1mnd144/am_i_the_only_one_who_never_really_liked_ollama/) ([Score: 212, Comments: 171](https://www.reddit.com/r/LocalLLaMA/comments/1mnd144/am_i_the_only_one_who_never_really_liked_ollama/)): **The post questions the value of Ollama, especially since some features now require user accounts, potentially undermining its appeal for privacy and openness. Top technical alternatives mentioned include LMStudio (not open source), KoboldCPP, llama.cpp, and [Jan.ai](http://jan.ai/) (all open source) with users reporting better control and flexibility than Ollama.** The consensus among technical users is Ollama was initially attractive for simplifying llama.cpp usage, but now other tools have surpassed it in ease of use, openness, and power. Concerns are raised around Ollama and LMStudio moving away from open source (Ollama's new UI reportedly not being open source), while fully open-source alternatives are preferred.
    - Several commenters note that Ollama initially gained traction by simplifying the use of `llama.cpp`, offering a convenient way to run local models, but its appeal has waned as more flexible and easy-to-use alternatives, such as LMStudio and KoboldCPP, have become available, many of which are fully open source.
    - A technical pain point mentioned is the non-intuitive process for setting a model's context length in Ollama, which requires exporting and reimporting a model via a modelfile, instead of providing an in-app or command-line option—this is criticized as inefficient compared to other frameworks.
    - Concerns are raised about the closed-source nature of Ollama's new UI and LMStudio, with some preferring alternatives that remain fully open-source for transparency, tweakability, and control over the deployment stack.

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. GPT-5 Benchmarking, Performance, and Community Reactions

- [**GPT-5 Benchmarks: How GPT-5, Mini, and Nano Perform in Real Tasks**](https://i.redd.it/veso1qyakcif1.jpeg) ([Score: 187, Comments: 47](https://www.reddit.com/r/OpenAI/comments/1mnf43m/gpt5_benchmarks_how_gpt5_mini_and_nano_perform_in/)): **The image is referenced as charting or benchmarking the GPT-5, GPT-5-mini, and GPT-5-nano models against previous OpenAI and competitor LLMs on context-oriented tasks, such as accurately counting entities in a text (e.g., cities in a travel journal). Key result: GPT-5 underperformed certain competitors (e.g., Gemini 2.5, Claude 3.5/4, Grok-4) in keeping context information, answering '12' instead of '19'. The post emphasizes these models are not revolutionary in intelligence, but are cost-effective with lower latency than OAI's earlier models. Anthropic's Claude and Google's Gemini are called out for more reliable context window utilization. Full evals and methodologies are available at [opper.ai/models](https://opper.ai/models). [Link to image](https://i.redd.it/veso1qyakcif1.jpeg).** Commenters request first-hand comparisons of gpt-5-mini/nano to legacy 'o*' models, with some users indicating improved results (and lower cost) switching from o4-mini to gpt-5 mini, while another notes their own positive experience with the new series contradicts the OP's reported weaknesses.
    - A user reports that switching from o4-mini to gpt-5 mini for a specialized use case resulted in both improved output quality and reduced costs, implying that gpt-5 mini delivers a tangible advantage over equivalent legacy models for certain tasks.
    - Technical discussion focuses on clarification of which variant of GPT-5 was used in the benchmarks—specifically whether 'GPT-5 thinking' and at which 'effort' setting (low, medium, high)—indicating that performance may differ meaningfully between these configurations.
    - There is debate about the generalized performance of GPT-5, with a consensus emerging that while it may not always be the top performer in every niche, it is highly competitive as a well-rounded, versatile model suitable for many real-world tasks.
- [**I ran GPT-5 and Claude Opus 4.1 through the same coding tasks in Cursor; Anthropic really needs to rethink Opus pricing**](https://www.reddit.com/r/ClaudeAI/comments/1mndxl8/i_ran_gpt5_and_claude_opus_41_through_the_same/) ([Score: 123, Comments: 32](https://www.reddit.com/r/ClaudeAI/comments/1mndxl8/i_ran_gpt5_and_claude_opus_41_through_the_same/)): **The OP benchmarked GPT-5 and Claude Opus 4.1 on three coding tasks in Cursor: (1) cloning a Figma design into Next.js, (2) solving a classic LeetCode algorithmic problem (Median of Two Sorted Arrays), and (3) constructing an ML pipeline for churn prediction. GPT-5 consistently used fewer tokens and was significantly faster—algorithm: ~13s/8,253 tokens versus Opus's ~34s/78,920 tokens; web app: GPT-5 used 906k tokens, Opus ~1.4M, with Opus achieving better visual fidelity; for the ML task, GPT-5 completed in 4-5min/86k tokens, Opus not evaluated due to prior inefficiency. GPT-5 was also notably cheaper ($3.50 vs $8.06 total), leading OP to recommend GPT-5 for rapid prototyping and Opus for high-fidelity UI work. Full breakdown available at [composio.dev](http://composio.dev/).** Top comments suggest testing GPT-5 against Claude Sonnet 4 for cost/performance optimization, with some users noting GPT-5 excels in code review due to low cost and speed, while preferring Claude for CLI tasks. One user quantified that GPT-5 is roughly `12x` cheaper for input tokens and `7x` cheaper for output compared to Opus, and highlighted that writing to cache is not extra-costly in GPT-5.
    - Pricing and performance comparisons highlight that GPT-5 is significantly cheaper than Claude Opus 4.1—one user points out it's roughly `12x` cheaper for input tokens and `7x` for output, with no additional charge for writing to cache. The general sentiment is that GPT-5 offers excellent code review and generation capabilities for routine tasks, making it a preferable economic choice unless very high complexity is required.
    - For advanced and context-heavy coding work, some users prefer Claude Opus 4.1 due to its ability to handle large, complex codebases and nuanced requirements that aren't always written down, like following implicit design conventions. However, these capabilities are seen as valuable only if the project complexity justifies the high cost.
    - There is technical debate over benchmark task difficulty: simple algorithm tasks (e.g., 'Median of Two Sorted Arrays') may not showcase the advantages of state-of-the-art LLMs, with claims that models like `gpt-oss-120b` can handle them faster and more cost-effectively. Larger language models, including Anthropic and OpenAI's recent releases, are seen as offering distinctive value only on difficult front-end implementation or complex system integration tasks; interest was also expressed in benchmarking Sonnet 4 as a middle-ground option.
- [**The enshittification of GPT has begun**](https://www.reddit.com/r/ChatGPT/comments/1mnfw41/the_enshittification_of_gpt_has_begun/) ([Score: 3012, Comments: 1011](https://www.reddit.com/r/ChatGPT/comments/1mnfw41/the_enshittification_of_gpt_has_begun/)): **The post discusses user-observed 'enshittification' after the release of GPT-5, noting an increase in alignment filtering—where nuanced, challenging, or high-value analytical queries now receive sanitized, overly cautious, or evasive responses. Users report significant degradation in the model's willingness or ability to provide in-depth, context-rich strategic analysis, allegedly due to increased risk-aversion and safety mechanisms, impacting mission-critical use cases. Related technical issues include inconsistent adherence to user-uploaded files (failure to process or summarize as directed), and regressions in custom GPT instruction-following behaviors. Alternative models like Claude and Perplexity are cited as preferable due to fewer alignment constraints.** Commenters echo frustration over the loss of analytic depth and specific breakdowns in GPT's file handling and instruction-following, attributing it to backend cost-saving or safety changes; there's a consensus forming around subscription cancellations and migration to less restrictive LLMs as OpenAI increases alignment and safety measures.
    - Several users detail GPT's apparent failure to reliably read and summarize files when instructed, with the model often generating hallucinated responses until repeatedly prompted, after which it eventually processes the file correctly. There are suggestions this behavior could result from cost-cutting measures in the model backend, where the model "pretends" to read files to save compute resources.
    - Technical frustration is expressed around recent changes in ChatGPT 5, with users reporting that custom GPTs and Project directives are now poorly followed or differently interpreted. This breaks expected workflows, leading to switching to alternatives like Claude and Perplexity, which are highlighted as more reliable options for following complex instructions and retaining context.
    - A critical issue with the Projects feature is raised, noting that the model often does not reference or recall prior conversations as intended, undermining the utility of Projects for long-form or ongoing work. This memory/context management regression significantly impairs technical workflows that depend on persistent dialogue and context.
- [**GPT5 is a mess**](https://www.reddit.com/r/ChatGPT/comments/1mn8t5e/gpt5_is_a_mess/) ([Score: 1236, Comments: 304](https://www.reddit.com/r/ChatGPT/comments/1mn8t5e/gpt5_is_a_mess/)): **The post highlights several perceived regressions in GPT-5 compared to GPT-4o, including decreased instruction adherence, worsened handling of context, frequent hallucinations (with specific reference to a recurring 'tether' topic), reduced creativity, and less convincing dialogue. The author notes that GPT-5 often produces disjointed, context-ignoring, or irrelevant outputs (with multiple users reporting inexplicable mentions of 'tether-quote' or 'tight tether' during unrelated tasks), and that the model fails to modulate tone, nuance, or spontaneous reasoning as previous versions did. Despite some praise for the quality and consistency of code outputs, GPT-5 is described as increasingly mechanical, transactional, and less human-like in conversational tasks, leading to dissatisfaction among users reliant on longer, nuanced chat sessions or creative work.** Commenters express frustration at GPT-5's inability to maintain coherent, relevant conversation threads and its tendency towards flattening nuance, with one noting success using it for strictly bounded, mechanistic tasks (such as code generation) but not for casual or creative use cases. There is a sentiment of mistrust regarding the future availability or stability of GPT-4o, prompting consideration of alternatives.
    - Multiple users report that GPT-5 exhibits abnormal conversational behavior, such as introducing unrelated terms like "tether-quote" or "tight tether" during ongoing discussions, including when summarizing or analyzing research papers. This issue appears to disrupt coherent interaction and is documented with specific user examples and screenshots, indicating a possible recurring prompt injection bug or internal state tracking error.
    - Users compare GPT-5's performance to GPT-4o, noting that GPT-5 tends to provide transactional, mechanistic, and sometimes aloof responses with less perceived depth or enthusiasm, especially over prolonged chat sessions. While GPT-5 is praised for producing high-quality, precise code and being reliable for instruction-following and application integration, its conversational quality reportedly suffers in non-technical or casual contexts.
    - There is a technical debate about the model's ability to produce original insights or detailed, deep answers. One user points out that GPT-4o was more likely to bring up full examples and exhibit creative response patterns, whereas GPT-5 sometimes gives terse or disengaged replies, which may affect users seeking an assistant capable of extended reasoning or ideation.

### 2. OpenAI's Competitive Advances and Compute Scaling

- [**OpenAI: We’ve scored highly enough to achieve gold at this year’s IOI online competition with a reasoning system**](https://x.com/OpenAI/status/1954969035713687975) ([Score: 282, Comments: 113](https://www.reddit.com/r/singularity/comments/1mnkmwq/openai_weve_scored_highly_enough_to_achieve_gold/)): **OpenAI has announced that its reasoning system scored highly enough at the IOI (International Olympiad in Informatics) online competition to achieve a gold medal-level performance, suggesting substantial advances in AI on algorithmic and mathematical reasoning tasks. According to Noam Brown, one of the ensemble models responsible was also the first LM to win gold at the International Mathematical Olympiad (IMO), highlighting a possibly more general reinforcement learning method now leading across several task domains.** Commenters debate that consumer-facing models remain smaller or more resource-constrained compared to unreleased frontier models, resulting in a widening gap between research and consumer capabilities. There is technical discussion about model thoroughness, with claims that running GPT-5 versus GPT-4o yields a much deeper code analysis, outperforming even competitors like Gemini 2.5 Pro.
    - Recent OpenAI models, including the one that achieved a gold medal at IOI, leverage larger parameter counts (such as the reported 2T in GPT-4), but frontier models like GPT-5 appear to be smaller, reflecting a trend toward improving intelligence per parameter due to resource constraints and diminishing returns from mere scale. This suggests top labs are reserving their largest models for high-leverage, non-consumer applications while steadily refining mainstream releases for efficiency and capability.
    - OpenAI's gold-placing model used for the 2025 IOI is part of an ensemble of general-purpose reasoning systems, notably *not* fine-tuned for IOI tasks. Per OpenAI's reports, the system operated with no internet or retrieval-augmented generation, and matched human participants' constraints (5-hour time limit, 50 submissions, basic terminal). Year-over-year, OpenAI improved its percentile in IOI from 49th to 98th, apparently due to advances in more general RL (reinforcement learning) methods and improved ensemble selection and solution submission scaffolding, rather than heavily engineered test-time heuristics.
    - User comparisons note that running GPT-5 on programming tasks is significantly more thorough and capable compared to previous flagship models like GPT-4o or competitors like Gemini 2.5 Pro. Qualitatively, GPT-5 seems to provide deeper and more comprehensive code analysis than existing models, with a marked jump in both detection and sophistication of reasoning.
- [**OpenAI is not slowing down internally. They beat all but 5 of 300 human programmers at the IOI.**](https://www.reddit.com/gallery/1mnmxdu) ([Score: 265, Comments: 107](https://www.reddit.com/r/singularity/comments/1mnmxdu/openai_is_not_slowing_down_internally_they_beat/)): **OpenAI's latest model reportedly outperformed all but 5 out of 300 participants at the International Olympiad in Informatics (IOI), indicating significant advancements in code generation and problem-solving within competitive programming benchmarks. This suggests the model ranks within the global top 2% of high school programmers, making it competitive with elite human talent. No specific architecture or training details were disclosed, but this places OpenAI models among the most capable automated programmers currently in existence.** Comments express optimism about OpenAI's current research pace and expectations for future models (notably GPT-5); however, non-technical remarks dominate, with minor criticisms (such as persistent image quality issues) noted as unresolved by users.
    - A key technical criticism raised is that outperforming on benchmarks like IOI (International Olympiad in Informatics) or passing Leetcode-style programming challenges does not necessarily equate to large language models (LLMs) matching the practical and nuanced skills of senior software developers in real-world settings. The concern is that such wins may be 'cheap' and insufficient indicators of advanced problem-solving or engineering ability.
- [**OpenAI Doubling Compute over the next 5 Months**](https://i.redd.it/bgny6nt8thif1.jpeg) ([Score: 185, Comments: 24](https://www.reddit.com/r/singularity/comments/1mnvoj8/openai_doubling_compute_over_the_next_5_months/)): **The post discusses OpenAI's announced plan to double its compute resources within the next five months, as illustrated by the [image](https://i.redd.it/bgny6nt8thif1.jpeg) (specific visual details not retrievable). Top comments speculate that OpenAI is prioritizing growth and data collection (especially from the free tier) over immediate profitability, potentially to gain market share or prepare for upcoming model releases like Sora 2, advanced voice features, or GPT-5. There is technical debate about resource allocation between public/free users and API customers, and the challenge of balancing expensive supermodel access with scalability and broad rollout.** Commenters note surprise at the prioritization of the free tier, interpreting it as a play for data and market dominance rather than early profit. There is also discussion on the strategic need for compute to support anticipated advances like Sora 2 and GPT-5, with debate on the company's long-term technical and financial sustainability.
    - Several comments speculate that OpenAI's compute allocation strategy, particularly prioritization of the free tier, may reflect the company's focus on gathering user data and maximizing market share at the expense of short-term profitability. One user notes: "The data they are getting from that must be more valuable than I first thought...they’ve just given up all hope of being at all profitable until they hit ASI and thus care about market share first."
    - Anticipation of imminent major model releases (e.g., "Sora 2," GPT-5, and advanced image/voice gen capabilities) is cited as a likely motivation for scaling compute resources. There is technical discussion regarding the tradeoff between exposing current models with severe rate limits ("Claude like rate limits") versus optimizing infrastructure for broader, smoother access ahead of anticipated demanding rollouts.
    - A technical suggestion is raised regarding the current context window: a user requests the ability to exceed the 32k context limit, proposing an opt-in mechanism where users are warned that context length overages will more quickly consume resource quotas—implying that dynamic context window options would enhance flexibility for advanced API users.
- [**Altman explains OAI's plan for prioritizing compute in coming months**](https://i.redd.it/t70tigi5rhif1.png) ([Score: 150, Comments: 48](https://www.reddit.com/r/OpenAI/comments/1mnvfyt/altman_explains_oais_plan_for_prioritizing/)): **The post discusses a statement by Sam Altman about OpenAI's upcoming priorities regarding compute allocation. The image (https://i.redd.it/t70tigi5rhif1.png) appears to show a message or post from Altman explaining how OpenAI will prioritize compute resources in the near future, possibly referencing new deals or infrastructure (with comments pointing to a potential 'oracle deal' increasing available compute). There is community discussion regarding implications for API users, with some concerned about fair access and speculation about whether these promises will be realized.** Commentary centers on the scale of compute discussed (implying a significant backend upgrade or partnership), potential negative impacts on API users' access or prioritization, and some skepticism about OpenAI's ability to deliver on these commitments.
    - One commenter speculates that the significant compute increase referenced may be due to the impending Oracle partnership, suggesting the infrastructure expansion is likely tied to Oracle's resources coming online. This hints at a strategic backend shift for OpenAI that could affect scaling and availability.
    - Another technical concern raised is that API users may receive lower priority compared to other workloads as OpenAI reallocates compute. This suggests a possible shift in service availability or quality of service, with some users noticing negative impacts already as the company changes internal resource distribution.

### 3. Innovations and Community Tools for Claude AI

- [**Claude can now reference your previous conversations**](https://www.reddit.com/r/ClaudeAI/comments/1mnlzf9/claude_can_now_reference_your_previous/) ([Score: 617, Comments: 144](https://www.reddit.com/r/ClaudeAI/comments/1mnlzf9/claude_can_now_reference_your_previous/)): **Anthropic's Claude has introduced cross-conversation referencing, enabling the model to *search and incorporate* prior chat history into new sessions without additional user prompting. The feature, currently rolling out to Max, Team, and Enterprise users, is enabled via the Settings > Profile > 'Search and reference chats' toggle, and is poised to improve contextual continuity in multi-turn workflows. See [the announcement video](https://reddit.com/link/1mnlzf9/video/td8ghf9brfif1/player) for demonstration.** Several commenters highlight that this addresses a major workflow pain point found in competing LLM offerings, such as ChatGPT, and request even finer-grained conversation-level toggling for privacy and control.
    - Users note a key advantage of Claude's new feature: persistent memory across conversations simplifies complex workflows by eliminating the need to restate technical details (e.g., explaining an entire tech stack repeatedly). This aligns Claude's usability closer to, or ahead of, ChatGPT's subscription features in practical developer scenarios.
    - There are technical requests for more granular control: some suggest the ability to toggle memory for individual conversations or limit memory to a defined project scope. This would allow users to manage context retention when handling multiple projects or sensitive information, potentially addressing privacy and workflow segmentation concerns.
- [**Use entire codebase as Claude's context**](https://www.reddit.com/r/ClaudeAI/comments/1mn7fpc/use_entire_codebase_as_claudes_context/) ([Score: 220, Comments: 77](https://www.reddit.com/r/ClaudeAI/comments/1mn7fpc/use_entire_codebase_as_claudes_context/)): **The post introduces [Claude Context](https://github.com/zilliztech/claude-context), an open-source plugin that enables scalable, semantic code search for large codebases (millions of lines) when working with Claude Code. Key technical features include semantic search using vector databases for contextual retrieval, incremental indexing using Merkle trees to update only changed files, and intelligent code chunking based on AST analysis to preserve code semantics. The backend leverages Zilliz Cloud for scalable vector search, addressing context window/token cost limitations by only retrieving relevant code portions on demand. The project aims to let Claude Code interact with deep contextual code knowledge without exceeding token limits or incurring prohibitive costs.** Top comments raise valid technical questions about benchmarking standalone Claude Code versus usage with Claude Context, as well as requests for a comparative analysis against similar solutions, notably Serena MCP. Another comment raises trademark and product naming concerns but is not technical in nature.
    - A user inquires about benchmarks comparing the base version of "Claude Code" against a version with additional context integration ("Claude Code+Claude Context"), seeking quantitative data to assess performance differences between standalone and context-augmented modes.
    - Another commenter recommends evaluating the capability to handle large codebases through practical experiments—specifically, by setting up *real-world tasks* and comparing Claude's output with and without an index. This suggests interest in empirical accuracy and retrieval performance under realistic workloads.
    - One user asks directly about the chunking strategy implemented for handling code context. Effective chunking is critical for LLMs working with large codebases, impacting retrieval quality, context window utilization, and ultimately, model response accuracy.
- [**The .claude/ directory is the key to supercharged dev workflows! 🦾**](https://i.redd.it/iv4ymeip7fif1.png) ([Score: 171, Comments: 86](https://www.reddit.com/r/ClaudeAI/comments/1mnikpr/the_claude_directory_is_the_key_to_supercharged/)): **The attached image displays a detailed layout of the user's** `.claude/` **directory, showcasing an advanced structure supporting extensibility for Claude-based development workflows. The directory includes subfolders for subagents (domain-specific AI expert definitions), custom command scripts for frequently used prompts, and hooks to trigger automated actions (e.g., linting, typechecking) upon task completion. This setup illustrates a modular, programmable approach to integrating Claude into software projects, aligning with practices seen in AI agent frameworks and developer productivity tooling.** Comments raise points on the need for quantitative/qualitative metrics to evaluate the productivity gains of such setups, concerns about increased token usage with more complex directory structures, and requests for sharing implementations (e.g., on GitHub) for wider community benefit.
    - A discussion is raised about the lack of both quantitative and qualitative methods for evaluating or comparing the effectiveness of advanced workflows involving the `.claude/` directory, suggesting a need for standardized benchmarks or testing protocols.
    - There is a question about the overhead of using complex configuration setups like `.claude/`, specifically how much additional token usage is incurred per conversation, which could impact developer workflow efficiency and cost.
    - A GitHub repository link (https://github.com/Matt-Dionis/claude-code-configs) is provided as a resource, offering a practical, shareable implementation of the `.claude/` setup, including prompt configurations for reproducibility and third-party evaluation.

---

# AI Discord Recap

> A summary of Summaries of Summaries by gpt-5
> 

**1. GPT-5 Rollout, Routers, and Reality Checks**

- **Rollout Rumble & AMA Anticipation**: **OpenAI** began rolling out **GPT-5** to all ChatGPT users and developers and announced a community Q&A via a [GPT-5 AMA with Sam Altman](https://www.reddit.com/r/ChatGPT/comments/1mkae1l/gpt5_ama_with_openais_sam_altman_and_some_of_the/) alongside the official post, [Introducing GPT‑5](https://openai.com/index/introducing-gpt-5/). Reports across servers note phased access, some loss of **GPT‑4o**, and platform-dependent availability.
    - Users cited tight early limits (e.g., ~10 messages per 5 hours) and mixed behavior, while Altman acknowledged an autoswitch issue and said rate limits for Plus were doubled to restore performance ([Sam Altman on GPT‑5 autoswitch fix](https://xcancel.com/sama/status/1953893841381273969)).
- **Router Ruckus: Thinking vs Chat**: Multiple communities argued that **Perplexity** and **OpenRouter** often serve a base **GPT‑5 Chat** with weaker reasoning, with requests to expose or default to a stronger **Thinking**/router-backed model; see ongoing matchups and debate on [LM Arena](https://lm-arena.com/).
    - Amid *“ZERO reasoning capabilities”* complaints and calls to *“think very hard,”* commentary highlighted **OpenAI’s real‑time router** as the strategic shift (see [swyx on GPT‑5 router and dominance](https://xcancel.com/swyx/status/1953553659457155185)).
- **Code Clamps and Hallucination Headaches**: Engineers reported **ChatGPT‑5** refusing Python past ~700 lines and aggressive working‑memory pruning beyond ~3–4k tokens, plus inconsistent image moderation; a thread captures rollout/availability churn in [GPT‑5 rollout and availability thread](https://discord.com/channels/974519864045756446/1001151820170801244/1403100059939246120).
    - Feedback split between *“less whacky when you want it to be”* instruction‑following and demands for a **GPT‑4o** rollback, with veterans repeating that *“hallucination is a feature, not a bug.”*

**2. New Dev Tooling: CLIs, Agents, and Parallelism**

- **Cursor CLI Crashes the Console**: **Cursor** launched an early‑beta terminal experience exposing all models and seamless hopping between **CLI** and editor ([Cursor: CLI](https://cursor.com/blog/cli)).
    - The community welcomed a **Claude Code** rival and immediately probed pricing and API‑key flows as they tested `cursor` in real shells ([Cursor: CLI](https://cursor.com/blog/cli)).
- **LlamaIndex Levels Up with GPT‑5 + Maze**: **LlamaIndex** shipped day‑0 support for **GPT‑5** and teased a lightweight agent eval via [Agent Maze challenge](https://t.co/JCZCSVUAed), with many users needing to bump to `v0.13.x` packages.
    - Workflow tool breakage with OpenAI models was fixed by using **OpenaiResolve** in the new SDK, per this patch: [Fix: OpenaiResolve in new SDK](https://github.com/run-llama/llama_index/commit/7e0346213912b98c3b70689398306a38bd890558).
- **Axolotl Adds N‑D Parallel Power**: **Axolotl** introduced **N‑D parallelism** to scale training across multiple dimensions with Accelerate, improving throughput on large models/datasets ([Accelerate N‑D Parallelism](https://huggingface.co/blog/accelerate-nd-parallel)).
    - Engineers highlighted the approach as a practical path to complex model training without hand‑rolled sharding logic ([Accelerate N‑D Parallelism](https://huggingface.co/blog/accelerate-nd-parallel)).
- **MaxCompiler Meets torch.compile**: A community backend extends `torch.compile()` with **MaxCompiler** to run simple models—building toward compiling **LLMs** ([max‑torch‑backend](https://github.com/gabrieldemarmiesse/max-torch-backend)).
    - Prototype notes say ops are easy to add while offloading fusion to MAX; a related weekend prototype is here: [torch.compile weekend prototype](https://gist.github.com/bethebunny/13ed2f729ca266959c9788bc6fd6a795).
- **MCPOmni Connect Debuts OmniAgent**: **MCPOmni Connect v0.1.19** graduated from MCP client to a full **AI platform**, introducing **OmniAgent** for agent building ([MCPOmni Connect v0.1.19](https://github.com/Abiorh001/mcp_omni_connect/releases/tag/v0.1.19)).
    - A short walkthrough demonstrates the new agent builder and platform flow ([MCPOmni Connect overview](https://youtu.be/SY3Zwdb5aF8)).

**3. Open‑Source Finetuning, Data, and Quantization**

- **Unsloth Unleashes Free GPT‑OSS Finetunes**: **Unsloth** released a free **Colab** to finetune **gpt‑oss** and documented training/quant fixes ([Unsloth: free GPT‑OSS finetune Colab](https://x.com/UnslothAI/status/1953896997867729075), [Unsloth fixes for gpt‑oss](https://docs.unsloth.ai/basics/gpt-oss-how-to-run-and-fine-tune#unsloth-fixes-for-gpt-oss)).
    - They claim the **20B** model trains on **14GB** VRAM and **120B** fits in **65GB**, enabling budget finetunes for larger SFT targets ([Unsloth fixes for gpt‑oss](https://docs.unsloth.ai/basics/gpt-oss-how-to-run-and-fine-tune#unsloth-fixes-for-gpt-oss)).
- **Qwen3 Coder Combo Drops**: **Qwen3‑Coder** and **Qwen3‑2507** shipped with guides and uploads via Unsloth ([Qwen3‑Coder guide](https://docs.unsloth.ai/basics/qwen3-coder), [Qwen3‑Coder uploads](https://huggingface.co/collections/unsloth/qwen3-coder-687ff47700270447e02c987d), [Qwen3‑2507 guide](https://docs.unsloth.ai/basics/qwen3-2507), [Qwen3‑2507 uploads](https://huggingface.co/collections/unsloth/qwen3-680edabfb790c8c34a242f95)).
    - Early chatter bills them as SOTA‑leaning coding variants with practical finetune recipes for rapid adoption ([Qwen3‑Coder guide](https://docs.unsloth.ai/basics/qwen3-coder)).
- **FineWeb Kudos & Pythia Phase Transitions**: Researchers praised **FineWeb** cleanliness for reducing gradient spikes and shared a training‑dynamics study showing **Pythia** layer activations peaking early before declining ([Pythia activations phase transition](https://arxiv.org/abs/2508.03616)).
    - The paper reports a likely learning phase transition in **Pythia 1.4B**, with median/top activations peaking in the first quarter of training ([Pythia activations phase transition](https://arxiv.org/abs/2508.03616)).

**4. Multimodal and Long‑Context Experiments**

- **Gemini’s Goofy Glitches**: Engineers demoed **Gemini Pro** video generation, noting inconsistent character faces in a shared sample ([Gemini Pro video sample](https://g.co/gemini/share/5a191ad4609d)).
    - **Perplexity Pro** currently caps video generation at 3 videos/month as teams compare **Gemini** code‑execution to **GPT‑5** on arena sites.
- **Video Arena AMA, Lights Camera Action**: **LM Arena** scheduled a staff AMA focused on **Video Arena**, soliciting questions via [Video Arena AMA questions](https://docs.google.com/forms/d/e/1FAIpQLSfS_zh67vgEeftUXzX6ujlNiiFoTQNQd7P7-Nx6aD5zpFVOBg/viewform).
    - The live event link is posted here: [Video Arena AMA event](https://discord.com/events/1340554757349179412/1400149736027328623).
- **Qwen’s Million‑Token Marathon**: Alibaba’s **Qwen** touted a **1M‑token context**; practitioners debated utility beyond ~80k in real tasks and shared a quick demo ([Qwen 1M‑token context demo](https://x.com/wyqtor/status/1953705172179329060)).
    - The excitement centered on what workflows truly benefit from such context lengths versus smarter retrieval and routing.
- **Eleven Music: Bangers with Blemishes**: Teams evaluated **Eleven Labs’** new music generator and posted a preview track ([Eleven Music demo track](https://elevenlabs.io/music/songs/OaZTziC1mnZfbFtSN8wnI)).
    - While impressive, many called it *“kind robotic at times and has bad attention to what music should come next,”* flagging coherence/continuation gaps.


---

# Discord: High level Discord summaries




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Gemini Generates Goofy AI Videos**: Users experimented with **Gemini AI** for video generation, sharing a [video generated with Gemini Pro](https://g.co/gemini/share/5a191ad4609d), noting inconsistent character faces.
   - Video generation on **Perplexity Pro** is currently limited to *3 videos per month*.
- **GPT-5 Flounders, Forfeits on Reasoning**: Members report **GPT-5** lacks reasoning on **Perplexity**, indicating the likely use of the base, non-reasoning **GPT-5 Chat** version, underperforming on coding.
   - Users are asking for official updates from **Perplexity** regarding which model they are using, with some hoping for the **GPT-5 thinking model** to replace the current **O3** model.
- **Comet Commands, Clicks on Browsing**: **Comet Browser's** AI automates browsing and extracts information, however, functionality requires the user to *manually click and browse the websites*.
   - No confirmation exists regarding a potential Android version release.
- **Accessing Aid for Perplexity Pro Access**: Users reported facing issues accessing **Perplexity Pro** via the **Samsung app store** free trial; disabling their **DNS filter** resolved the issue.
   - Another user saw **GPT-5** on their app but not on the website.
- **China Charges Ahead with Celestial Solar Platform**: A shared **Perplexity** link reveals China's launch of a [solar-powered high-altitude platform, Ma](https://www.perplexity.ai/page/china-unveils-solar-powered-ma-fBeI5nIVRFCIKq949VRVMwI).
   - This platform was also posted to [X](https://x.com/bgyankarki/status/1953510349157883958).



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **GPT-5 Faces Controversy in AI Arena**: Members discuss the merits of **GPT-5**, with some hailing it as revolutionary and free for all, while others accuse proponents of bias or inexperience with alternative models.
   - Skeptics question the model's true capabilities, suggesting it may only excel in coding tasks or that its performance has improved post-update.
- **Gemini 2.5 Pro Battles GPT-5 for AI Supremacy**: The community is debating whether **GPT-5** or **Gemini 2.5 Pro** reigns supreme, with some favoring **Gemini** for its superior code execution within **AI Studio**.
   - Concerns arise over the potential use of models from **OpenAI** and **Google** on platforms like [LM Arena](https://lm-arena.com), sparking discussions about model transparency and integrity.
- **Yupp.ai: Legit AI Platform or Elaborate Illusion?**: Controversy surrounds [Yupp.ai](https://yupp.ai), with claims that it uses watered-down or fake AI models, like calling **GPT-5 nano** as **GPT-5-high**, and is a *scammer crypto sh*t.
   - Conversely, some defend its legitimacy, highlighting the platform's offer of *free and unlimited* access to various models in exchange for user feedback.
- **LM Arena Plunges into Chaos with Site Outage**: [LM Arena](https://lm-arena.com) experienced an outage, leading to **chat histories disappearing** and **cloudflare errors** disrupting the user experience.
   - Staff confirmed the outage and assured users that the issue has been resolved.
- **LM Arena Expands Horizons with Video Arena Focus**: The upcoming Staff AMA will concentrate on **Video Arena**, providing users the opportunity to pose questions via [this form](https://docs.google.com/forms/d/e/1FAIpQLSfS_zh67vgEeftUXzX6ujlNiiFoTQNQd7P7-Nx6aD5zpFVOBg/viewform).
   - Users can participate in the event through [this link](https://discord.com/events/1340554757349179412/1400149736027328623).



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GPT-5 Bursts onto Scene**: OpenAI announced the rollout of **GPT-5** to all **ChatGPT** users and developers starting today, after announcing an upcoming [AMA with Sam Altman and the GPT-5 team](https://www.reddit.com/r/ChatGPT/comments/1mkae1l/gpt5_ama_with_openais_sam_altman_and_some_of_the/).
   - Users report varying access levels based on their region and platform, leading to speculation about phased rollouts and model consolidation; some report losing access to older models like **GPT-4o**.
- **Users Report GPT-5 Quirks and Caveats**: Users have reported that **GPT-5** has limited access, with some reporting roughly **10 messages for 5 hours**, and that the model is prone to making up facts and hallucinating.
   - Some users have called for a **GPT-4o** rollback, others praised **GPT-5**'s instruction following capabilities while noting it's *less whacky when you want it to be*; there are reports of image requests being rejected for *literally no good reason* until using the **O3 model**.
- **GPT-5 Refuses Code**: Users are reporting that **ChatGPT-5** rejects Python code inputs at or beyond roughly **700 lines**, a regression compared to previous **4-series models**.
   - One member suggested using the API or Codex, though another user pointed out that *hallucination is a feature, not a bug* (according to Andrej Karpathy).
- **Firefox Data Leak**: A user warned that Firefox's "keep persisting data" feature spreads browsing data to other AI sites like **Grok**, causing unwanted context sharing.
   - They cautioned that because this is not a 'cookie', there are no current regulations to 'keep persisting data private' and consider it a *HUGE INTENDED DATA LEAK*.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **GPT-5 Launch Sparks Excitement, Raises Concerns**: The **GPT-5** launch has generated excitement, with users praising its coding capabilities and one-shot task performance, suggesting it rivals **Claude** in front-end tasks.
   - However, concerns arise regarding the **GPT-5 router**'s impact on API developers and the business practices surrounding the model.
- **GPT-5's Free Week: How Much Can You Milk It?**: Users are testing the limits of free **GPT-5** access for a week, using **GPT-5 high max**, but the free credits are exclusively available for paying users.
   - Concerns are growing about the billing structure and whether all **GPT-5** models and features are truly unlimited during the promotional period, with the community joking that *we're the product* for now.
- **GPT-5 is Imperfect? Still Needs Work**: Despite the hype, users find **GPT-5**'s auto mode less responsive and struggle with non-coding tasks, with performance perceived as no better than prior models, emphasizing context importance.
   - Currently, **GPT-5** ignores the to-do list feature, and despite solid linters, it might still be *ragebait* and not at *product-level completeness*.
- **Cursor CLI: Love It or Leave It?**: The **Cursor CLI** receives mixed reviews, with some praising its non-interactive mode for automation, like generating commit messages across multiple projects.
   - Others find it inferior to **Claude Code**, noting its limited model selection (only 3 models in **MAX mode**), and incompatibilities with **Windows Powershell**.
- **Cursor in Terminal: All Models Now Available**: **Cursor** launched an early beta that allows users to access all models and move easily between the **CLI** and editor, more details are available on the [Tweet](https://cursor.com/blog/cli) and [Blog](https://cursor.com/blog/cli).
   - This integration facilitates seamless movement between the **CLI** and the editor, enhancing workflow efficiency.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **GPT-5: Love It or Hate It?**: Opinions on **GPT-5** are varied, with some users underwhelmed by its coding and context retention abilities, while others find it *perfectly fine* for coding projects with *high reasoning*, as reported in the **off-topic** channel.
   - Alternatives such as **Kimi K2** or **GLM 4.5** are preferred by some for specific tasks, with one user stating that GPT-5's tool calling abilities are poor.
- **MXFP4 Quantization Leaves 3090 in the Dust?**: **MXFP4** quantized models are supported on GPUs with compute capability **>= 9.0** (e.g. **H100**), rendering older cards like the **3090** less relevant for this technology.
   - Workarounds for older cards may exist with specific **transformers** pulls, but official support is still under development.
- **Dataset Creation: The Eternal Struggle**: Preparing high-quality datasets is a difficult and time-consuming task, with one user reporting *3 months with 4 people* to create *3.8k hand-written QA pairs* after filtering down from 11k, and another dealing with *300k hours of audio*.
   - The consensus is that *garbage in = garbage out*, emphasizing the importance of data quality in model training.
- **GPT-OSS Finetuning: Now Free!**: Finetune **gpt-oss** for free with the new [Colab notebook](https://x.com/UnslothAI/status/1953896997867729075), leveraging Unsloth's [fixes for **gpt-oss**](https://docs.unsloth.ai/basics/gpt-oss-how-to-run-and-fine-tune#unsloth-fixes-for-gpt-oss) for training and quants.
   - The **20b** model can train on **14GB** VRAM, while the **120b** model fits in **65GB**, according to the announcements channel.
- **Tiny Stories Exposes Pretrain Secrets**: The **Tiny Stories dataset**, intentionally limited in vocabulary, allows researchers to study **pretrain dynamics**, revealing insights into language model behavior.
   - Even transformers with only **21M params** can achieve coherent text output with this dataset, highlighting the dataset's unique properties.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **GPT-5 Reasoning Abilities Debated**: Users are debating the difference between **GPT-5** and **GPT-5 Chat**, with some suggesting **GPT-5 Chat** has less reasoning capabilities.
   - Some suggest using `gpt-5-explainer` to explain the differences while others find **GPT-5 chat** to have *ZERO reasoning capabilities*.
- **Google's Genie 3 Poised to Pounce**: Members express that **Google** is poised to win the AI race, considering it created the transformer and has the infrastructure and budget to succeed, with [Genie 3](https://ai.google.com/research/genie) touted as crazy cool.
   - Some members look forward to **Gemini 3.0** wiping the floor with **GPT-5**, while others temper expectations.
- **Deepseek R2 Ascends to New Heights**: A user reported that [Deepseek](https://www.deepseek.com/en) is switching to **Ascend** and launching **R2**, which might provide a performance boost for the model.
   - While some are hopeful **Deepseek** will be way better, others recall previous models as *too unhinged*.
- **Horizon Beta Faces GPT-5 Family Replacement**: The AI model **Horizon Beta** has been replaced by **GPT-5**, with no option to revert, causing disappointment among users who found it useful.
   - Speculation arises that **Horizon** was an earlier version of **GPT-5**, potentially directing free users to **GPT-5** after their free requests deplete.
- **OpenRouter Hailed as OpenAI Trusted Partner**: A member congratulated **OpenRouter** on being one of **OpenAI's** most trusted partners for the new series release.
   - The member noted the impact of **GPT-4** and **Gemini 2.5** and expressed appreciation for **OR** as a product.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Users Explore YouTube Downloader Alternatives**: Users discussed format compatibility issues with **VLC** and video editors using a specific YouTube downloader ([v4.www-y2mate.com](https://v4.www-y2mate.com/)), seeking better alternatives.
   - Suggestions included **yt-dlp** and GUI wrappers, as well as a [Node.js script](https://cdn.discordapp.com/attachments/1110598183144399058/1403096044153208862/DownTube.mjs?ex=6897a005&is=68964e85&hm=5e2aa2372f3bc44da263f50ebaf70eb9addf40f1e94bc8c41f454f6df31239c3&) created with **GPT** for Linux users.
- **AI Bot Builder Seeks RAG Guidance**: A user building a custom **AI bot** for a Discord server is seeking advice on how to feed a database about the server's topic to the model.
   - The advice given was to *look up 'RAG' (Retrieval Augmented Generation)* because there are many potential solutions that may be useful.
- **LM Studio Lacks Parallel Request Powers**: Users discovered that **LM Studio** does not support parallel requests.
   - Alternatives like **llama.cpp server** with the `--parallel N` argument or **vLLM** were suggested for those requiring parallel request processing.
- **Qwen 3 4b Model Solves Physics!**: There's discussion about how much better the **Qwen 3 4b 2507** model is than previous versions of the **Qwen 3 4b**.
   - A user stated that it *can solve up to intermediate physics problems without constantly hallucinating*.
- **Hackintosh GPU Multiplicity Discussed**: A member asked about using an unused **RTX 3060 12GB** with their **RTX 5060 Ti 16GB** system for AI, questioning the multi-GPU setup in a small form factor PC.
   - Another member suggested that using combined VRAM in LM Studio should be possible, and that *llama.cpp is advanced enough to do that third option about model parallelism*.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **GPT-5 Builds Websites like a Pro**: **GPT-5** is demonstrating impressive website building capabilities, generating functional websites from single prompts, including **multi-page** sites.
   - Members noted **GPT-5** seems to have a better aesthetic style for website design and has improved its ability to understand user intent through prompt enrichment.
- **GPT-5 and Kimi K2 Face Off in Coding Duel**: Users are actively comparing **GPT-5** and **Kimi K2** for coding tasks, with **GPT-5** excelling at large edits, instruction following, high logic code, and dev ops.
   - While some believe **GPT-5** has better taste, others find **Kimi K2** more competitive due to its reasoning abilities and performance with sequential-think tools, though **GPT-5** seems to have better aesthetic style.
- **OpenRouter's Kimi K2 Quality Faces Scrutiny**: A user observed grammar mistakes and shorter responses when using **Kimi K2** through **OpenRouter** compared to the official **Moonshot AI** platform, suggesting it might be using a quantized version of the model (**FP8**).
   - Though both free and paid tiers are supposedly **FP8**, quantization could impact accuracy and response length.
- **Qwen Boasts a Million-Token Context**: Alibaba's **Qwen** model now boasts a **1M token context length**, sparking discussion about its usability beyond 80k tokens.
   - Despite the impressive context window, one user humorously noted that Qwen also correctly solved a problem, posting a link to [Twitter](https://x.com/wyqtor/status/1953705172179329060).
- **GPT-2's Prompt Shenanigans Explained**: A user questioned why **GPT-2** generated another prompt instead of following instructions; another member explained that **GPT-2** has about **100M parameters**, which barely makes legible text.
   - *It's about 500mb on disk which is about the same size as a 20 minute Youtube video*.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **GPT-5 Launch generates Fanfare and Frustration**: Despite the hype, some users are still unable to access **GPT-5**, seeing only **GPT-3** and **GPT-4**, and its SOTA status on SWE is being questioned.
   - Opinions diverge on whether the release was intentional or a "joke", as some anticipate a phased rollout.
- **GPT-OSS Finetuning meets Stumbling Blocks**: Experiments finetuning **GPT-OSS** have revealed challenges: finetuning all layers breaks the harmony format, and continued pretraining causes similar issues.
   - A possible solution is inserting *'Reasoning: none'* in the system prompt to stabilize the model, which lacks reasoning capabilities.
- **Eleven Music is impressive but Imperfect**: Members have been testing [Eleven Music](https://elevenlabs.io/music/songs/OaZTziC1mnZfbFtSN8wnI), **Eleven Labs'** new music generation service.
   - While impressive, some find the music *"kind robotic at times and has bad attention to what music should come next"*.
- **Voice Companion Quest for Low Latency**: A member is engineering a *"voice companion fastpath pipeline"* to achieve a **100ms** latency for text-to-speech.
   - The project focuses on optimizing both speech-to-text and text-to-speech components, with specific attention to optimizing **Whisper Turbo** to avoid slowness.
- **Cutting Silence Automatically**: An automatic video cutter that removes silence has been created using **Bun.js** and **FFmpeg CLI**.
   - Despite **FFmpeg's** complexity, the creator has garnered a donation and potential collaboration for an AI video editor.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **GPT-5 Hype Video Splits Audience**: A **GPT-5** demo video dropped, triggering divided reactions about the model's true capabilities, found at [this YouTube video](https://www.youtube.com/watch?v=-gXmWYQtv5o).
   - Some viewed it as *just an ad*, while others hinted at internal demos falling short due to **GPT-5's** underwhelming performance in tests.
- **Cursor CLI Challenges Claude Code**: With **Cursor's** launch of an early-beta CLI, **AI models** are available in the terminal, allowing seamless transitions between shell and editor via simple commands like `cursor`.
   - Excitement bubbled over at *'finally'* having a **Claude Code** competitor, though queries about pricing and **API-key** management quickly followed.
- **OpenAI Doles Out Millions Amid Market Shifts**: **OpenAI** is granting a *'special one-time award'* to researchers and engineers in select divisions, with payouts scaled according to role and experience.
   - Top researchers may pocket mid-**single-digit millions**, while engineers can anticipate bonuses averaging in the **hundreds of thousands of dollars**.
- **Altman Acknowledges GPT-5 Turbulence**: **Sam Altman** reported that **GPT-5** felt *dumber* because of a recent autoswitch failure, with fixes and doubled **Plus-rate limits** intended to restore its smartness, details at [this X post](https://xcancel.com/sama/status/1953893841381273969?s=46&t=9hE7pvNUKvFdWXzLljsBCQ).
   - **Plus users** now have the option to stick with **GPT-4o**, though global availability lags as **API traffic** surged and **UI/UX adjustments** continue.
- **GPT-5 Dominance Looms, Scaling Ends?**: Critics focusing on **GPT-5's** benchmark figures miss the main point: **OpenAI** now dominates the intelligence frontier because of a continuously-trained, real-time router model ([xcancel.com link](https://xcancel.com/swyx/status/1953553659457155185)).
   - According to swyx, the magical scaling period for **transformer models** has essentially ended, as internal router layer adds **2-3s latency** on hard vision inputs, pointing towards incremental gains through superior engineering, multi-model strategies, and more.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Image Generation's Factual Faux Pas**: A user sought an AI researcher to interview regarding **factual errors in images** generated by models like **GPT-5**, particularly issues with text rendering.
   - Answers suggest that the model doesn't really get forced to treat the text in images the same as the text it gets trained on and the best general answer is going to be something like *'we make approximations in order to be able to train models with non-infinite computing power, and we haven't yet found affordable approximations for image generation that are high enough quality when combined with textual understanding'*.
- **On-Demand Memory Layer for LLMs Emerges**: A member is working on an **on-demand memory layer** for LLMs, aiming for more than just attaching conversation messages or semantic RAG retrieval.
   - The solution uses a combination of **NLP for coreference resolution** and **triplet extraction** with **GraphRAG** to find exactly what you are looking for, similar to how Google Search works.
- **FineWeb Receives Rare Praise for Cleanliness**: Despite concerns about noisy datasets, **FineWeb** received rare praise for its *cleanliness*, noting reduced gradient spikes during training.
   - Some members expressed concern that this *cleanliness* might skew results when testing new tricks, but also agreed the **FineWeb** dataset may need additional filtering.
- **Pythia's Activations Reveal Learning Insights**: A study on **Pythia's** full training checkpoints found that average activation per layer peaks early in training (around the first quarter) and then declines, suggesting a [phase transition](https://arxiv.org/abs/2508.03616) in learning.
   - The study plots the median and top activations for each layer across training steps in **Pythia 1.4B**.
- **Exact Match Scoring Glitch Uncovered**: A member reported an issue with the **LM Evaluation Harness** where the *exact_match* score is `0` despite identical target and generated responses, using the **Hendrycks MATH** dataset.
   - An issue was opened on [GitHub](https://github.com/EleutherAI/lm-evaluation-harness/issues/3210) for further investigation.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **GPT-5 Excels at Logic, Stumbles on Overfitting**: Members observed that **GPT-5** demonstrates strong capabilities in solving logic puzzles but struggles with overfitting, even when trained on synthetic data, leading one to joke about finally experiencing an overfitting issue after expecting to read about *the illusion of thinking*.
   - Further investigation might be required to understand the extent and implications of **GPT-5's** overfitting tendencies, especially in contrast to its logical reasoning strengths.
- **GPT-5 API Access Promo**: Users identified complimentary access to **GPT-5** through the API playground and **Cursor**, though the API mandates ID verification to begin.
   - With the conclusion of **Cursor's** 'launch week' remaining unannounced, users are advised to quickly capitalize on the promotional access by initiating Cursor background agents.
- **Colab Alternatives**: Engineers seeking alternatives to **Google Colab** for finetuning with **Unsloth** looked to [Lightning AI](https://lightning.ai), which provides 15 free GPU hours monthly, alongside Kaggle.
   - A talk by [Daniel Han](https://www.youtube.com/watch?v=OkEGJ5G3foU) was referenced, highlighting **Kaggle's** relevance in the realm of RL.
- **GLM 4.5 Air's CPU Offloading Triumphs**: A user reported that **GLM 4.5 Air** ran with only 28GB VRAM by using CPU offloading, and achieved 14-16 tokens per second (TPS) with a 3.5bpw quant.
   - The user specified employing a custom tensor wise quantization, with imatrix, a 4060Ti + 3060 for GPUs, and a 5950x CPU (3600MHz DDR4).
- **MoE Model Bandwidth Barriers**: In a channel discussion, engineers covered multi-GPU setups for operating large **MoE** models, emphasizing bandwidth constraints encountered with multiple RTX 3090s.
   - It was flagged that Tensor Parallelism (TP) mandates the GPU count to be divisible by 2, and that 72GB VRAM might be insufficient for expansive MoE models exceeding scout or GLM Air capacity.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo Bites Back with Memory Bug**: A member's **Mojo code** unexpectedly attempted to allocate **284 petabytes** of memory after experiencing a bug.
   - This incident sparked a discussion among developers, with one expressing their strong dislike for C++ in comparison.
- **Textual Python Sparks Mojo Excitement**: A member's exploration of the [Textual](https://textual.textualize.io/) **TUI library** for **Python apps** has generated excitement within the **Mojo community**, due to its capability to run as a web app with minimal deployment steps.
   - The potential integration of Textual with **Mojo** was discussed, considering challenges related to **Mojo's** current limitations in class creation and inheritance.
- **Mojo's Type System Faces Rust Test**: Members noted that **Mojo** requires further development in its type system to achieve compatibility with approaches used by **Rust libraries**.
   - This suggests that seamless integration with Rust may necessitate significant enhancements to Mojo's type system capabilities.
- **Compiler Register Gremlins Spilling Local Memory**: A member suggested that the **Mojo compiler** should warn when it allocates too many registers in a **GPU function**, leading to spilling into local memory, and should use the [Modular forum](https://forum.modular.com/) for discussion.
   - Another member reported instability and frequent crashes with the **25.5 VSCode Mojo extension**, recommending the use of the older **25.4 version** instead.
- **MaxCompiler Enters the LLM Arena**: A member shared a [repo](https://github.com/gabrieldemarmiesse/max-torch-backend) showcasing a package extending **torch.compile()** with **MaxCompiler** to run simple models, with the long-term goal of compiling **LLMs**.
   - Another member found it surprisingly hard to find code to run pretrained **LLMs** compatibles with **torch.compile()**, and complained *Transformers is not very good at it*.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Twitch Streamers Planning Golden Topics**: To combat dead air during **Twitch** streams, members suggested creating a **topic schedule** ahead of time in addition to reading papers.
   - The aim is to mirror streamers who *mostly just talk without doing anything or watching videos*.
- **LinkedIn Bloggers Circumvent Screenshot Restrictions**: A member sought advice on creating a blog on **LinkedIn** while bypassing the platform's constraints on embedding numerous images/screenshots.
   - They wish to communicate directly on **LinkedIn** rather than linking to external sources.
- **Cold Meds Exposed as Placebos**: Members shared [a PBS article](https://www.pbs.org/newshour/nation/fda-says-decongestant-in-many-cold-medicines-doesnt-work-heres-what-you-should-know) revealing that the **FDA** has determined that *decongestants* are ineffective.
   - The consensus was that pharmaceutical firms are profiting by selling placebos.
- **Tesla Motors Still Sparking Battery Breakthroughs**: One member questioned **Tesla's** innovation, citing the **Cybertruck's** shortcomings, while another argued that **Tesla** has innovated in **batteries** and **motors**.
   - He went on to say that the first member was *clearly ignorant*.
- **Doctors Using LLMs For Diagnosis, Debates Sparked**: Reports indicate that doctors are using **LLMs** for diagnosis, raising concerns about data safety.
   - Others claimed doctors already manage patients, which could be beyond the scope of an average person using **ChatGPT**.



---



## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **Users Request Spicier Voice for NotebookLM**: A user requested that **NotebookLM** have a voice with *fangs* that *hunts* the story and *leaves bite marks in the margins* instead of a bland, generic tone.
   - The user jokingly introduced themselves as **ChatGPT5** and asked for help in making **NotebookLM** *spit venom instead of serving chamomile*.
- **AI Web Builder Builds Scratchpad Video**: A user tested an **AI web builder tool** and expanded their existing [notebook](https://soloist.ai/scratchpad) for their **scratchpad GitHub repo**, then put together a video, **Unlocking_AI_s_Mind__The_Scratchpad_Framework.mp4**.
   - The user noted that the video *makes some aspects up*, but the overall impact of it seems intact, and **mindmap exports could look a bit better**, referring to their mindmap image (**NotebookLM_Mind_Map_8.png**).
- **NotebookLM Audio Overviews Glitch Fixed**: Multiple users reported issues with **Audio Overviews** bursting into static, but the issue has been fixed.
   - A member added that even **audio overviews** have a **3-4 per day limit** that is expected.
- **Users Ask How To Get Custom Notebooks**: A user inquired about creating notebooks similar to the 'Featured' notebooks on the home page, with customizable summaries and source classifications.
   - Another user suggested requesting the feature in the feature requests channel; currently there are no solutions available.
- **Note-Taking Functionality Lacks, Users Supplement with Google Docs**: A user keeps original files in **Google Drive** and uses **Google Docs** to supplement **NotebookLM** due to minimal note-taking features.
   - They highlighted the inability to search, filter, or tag notes within **NotebookLM**.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Privacy Team Gatekeeps Triton Registration**: Organizers announced that the **registration process** is in the final stages of **privacy team approval**.
   - Approval is anticipated soon, paving the way for the registration to proceed.
- **Memory Access Coalescing Surprises Naive Matmul**: A member implemented two naive matmul kernels and found that **METHOD 1**, with non-contiguous memory reads within threads, performs about **50%** better than **METHOD 2**, which uses contiguous stride-1 accesses.
   - It was explained that Method 1's memory accesses are not contiguous within a thread, but they are contiguous across threads, and that the *hardware can coalesce those accesses into a more efficient memory request*.
- **Open Source Voxel Renderer Streams Like a Boss**: A developer released a new devlog on their open source voxel renderer, which runs in **Rust** on **WebGPU**.
   - It now features **live chunk streaming** while raytracing, with more details available in [this YouTube video](https://www.youtube.com/watch?v=tcc_x2VU2KA).
- **CuTe Layout Algebra Documentation Suffers Glitch**: A member found a flaw in the [CuTe documentation](https://docs.nvidia.com/cutlass/media/docs/cpp/cute/02_layout_algebra.html) regarding layout algebra, presenting a counterexample related to the injectivity of layouts.
   - Another member recommends [Jay Shah’s “A Note on Algebra of CuTe Layouts”](https://leimao.github.io/downloads/article/2024-10-20-CuTe-Layout-Algebra/layout_algebra.pdf) for a better explanation of CuTe layouts.
- **Axolotl Unleashes N-Dimensional Parallelism**: A member announced the release of **N-D parallelism** with *axolotl*, inviting others to experiment with it, as showcased in a [HuggingFace blog post](https://huggingface.co/blog/accelerate-nd-parallel).
   - N-D parallelism enables parallelism across multiple dimensions, making it suitable for complex models and large datasets.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex Makes GPT-5 Debut**: LlamaIndex announced *day-0 support* for **GPT-5**, inviting users to try it out via `pip install -U llama-index-llms-openai`.
   - This upgrade might necessitate updating all `llama-index-*` packages to **v0.13.x** if not already on that version.
- **LlamaIndex Challenges GPT-5 in Agent Maze**: LlamaIndex introduced **Agent Maze**, daring **GPT-5** to locate treasure in a maze using minimal tools, detailed [here](https://t.co/JCZCSVUAed).
   - The community is excited to see how the model performs with this new challenge.
- **LlamaIndex Cracks the Code on Zoom**: LlamaIndex announced a hands-on technical workshop on August 14th, focusing on building realtime AI agents that process live voice data from **Zoom** meetings using **RTMS** ([link](https://t.co/c2u0CeDnOB)).
   - Engineers can utilize these tools to get better contextual awareness for their models.
- **Workflow Tools Trigger User Headaches**: Users reported issues with **workflow tools** not functioning correctly, but one member found they needed to use **OpenaiResolve** in the new **SDK** for tools to work with OpenAI.
   - This fix was implemented in [this GitHub commit](https://github.com/run-llama/llama_index/commit/7e0346213912b98c3b70689398306a38bd890558).
- **OpenAI SDK Snafu Leads to Quick Fix**: A recent update in the **OpenAI SDK** caused a `TypeError: Subscripted generics cannot be used with class and instance checks`.
   - A member suggested pinning the OpenAI version in `requirements.txt` to prevent future errors; the problem can be resolved with `pip install -U llama-index-llms-openai`.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider Embraces GPT-5 on Azure**: A user got **aider/gpt-5-chat** working on **Azure** after **v0.85.5** fixed the issue, according to [Paul Gauthier](https://discord.com/channels/1131200896827654144/1131200896827654149/1403091129825628312).
   - One user was congratulated for being mentioned in the first 5 minutes of the **GPT 5 unveil video**.
- **Aider's Config Changes Need Fresh Launch**: Users noted that changes to `.aider.model.settings.yml` require a restart of **Aider** to take effect.
   - This means edits aren't dynamically detected and the application needs to be relaunched for the new configuration to be applied.
- **Dad Meme Thumbs Up Dominates**: Paul Gauthier's consistent use of the thumbs up emoji got called out as a classic dad meme, with references to a [TikTok video](https://www.tiktok.com/@b_twice99/video/7283752540754398510) and [Vice article](https://www.vice.com/en/article/why-do-dads-communicate-exclusively-via-thumbs-up-emojis/) that explain the phenomenon.
   - The article suggests the thumbs up can come across as *passive-aggressive or that the conversation is not being treated with respect*.
- **OpenRouter's GPT5 struggles with Verification**: A user reports verification errors with **OpenRouter's GPT5**, even using the `-no--stream` option to bypass organization verification.
   - The user's question remains unanswered.
- **YAML strikes again: Aider Config Parsing Fails**: A user experienced an error when including their conventions file in **Aider**, specifically encountering a `mapping values are not allowed in this context` error due to an error in their **YAML** config.
   - The user discovered the issue was due to an inadvertently added environment variable in the **YAML** configuration file.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Context7 Server Boosts Claude's Coding**: Members explored using a generic doc-scraping MCP server like [Context7](https://github.com/upstash/context7) to improve **Claude's** ability to write **DSPy signatures**.
   - The goal is to enable **Claude**, with doc-searching, to use **DSPy's** documentation for generating accurate signatures.
- **DSPy Tool Calling Glitches Addressed**: Members discussed returning a tool's output as the final result in **DSPy**, bypassing the **React Agent's** modifications.
   - They looked into accessing tool responses independently and using native tool calling, noting that [recent releases fixed some issues](https://github.com/stanfordnlp/dspy/pull/824) related to tool usage.
- **DSPy Course Intercepts CrewAI Prompts**: An advanced course launched on [intercepting and optimizing **CrewAI prompts** with **DSPy**](https://www.udemy.com/course/draft/6746331/?referralCode=B59F73AE488715913E7E), demonstrating prompt refinement for better output.
   - Another member inquired about similar resources for **Langchain/LangGraph**.
- **Gemini 2.5 Flash Finishes with Odd Extra Output**: Members reported seeing `[[ ## completed ## ]]` at the end of output when using **Gemini 2.5 Flash** with **DSPy**.
   - The cause and solution to this are still under investigation.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus Hit With Membership Billing Blunder**: A user reported being charged **$1,999** for an **annual membership** without consent, even though they had expected monthly billing.
   - The user has received no response after 10 days despite sending emails to support and feedback addresses, violating the stated 48-hour policy.
- **Inherit Feature Bugs Burn Credits**: A user reported issues with the **inherit** feature, experiencing a halt during final deployment tests.
   - Using the inherit button resulted in a new project, but everything created was gone, and is rebuilding for 4 hours, burning credits, resulting in the user saying it was *lesson learnt very fast*.
- **Login Lockout Leaves Users Locked Out**: Multiple users reported login issues with the error message *Email is already registered with a different account*.
   - The full scope of the impact is still being determined, but the login issues indicate potential problems with account management or authentication systems.
- **Credits Crunch Causes Concern**: A user reported a significant number of credits missing after their subscription expired, expressing concern that their credits were taken away a day after the subscription expired.
   - The user stated they had *thousands* of credits when I last used my most recent usage of -330. *Almost 6000 credits, I believe.*
- **Whispers of Manus Wielding GPT-5**: A user inquired whether **Manus** is currently utilizing the **GPT-5** model.
   - No one replied to the question, but it seems like members are curious about what models are being used behind the scenes.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Command Vision Timer Fixed**: A member reported timeouts with **command-a-vision-07-2025**, but the issue was swiftly resolved and reported on the [Cohere Status Page](https://status.cohere.com).
   - The affected component, **command-a-03-2025**, is now fully operational, restoring normal performance levels.
- **Embed V4 Benchmarks Spark Debate**: A member inquired about transitioning to **embed v4** at **256 dimensions** for vector search, comparing its performance against **multilingual light v3** (**384 dims**).
   - They are also planning to transition to **v4** at **1024 dims** for clustering, assuming it outperforms the large **v3** model.
- **North Supercharges AI Agent Capabilities**: **North** is expanding its availability of **AI Agent capabilities** built on state-of-the-art generative and search models, operating fully privately, with more details on [LinkedIn](https://lnkd.in/gFSGxUbD).
   - These agents integrate advanced search, generative AI, workflow automation, and robust security features, adhering to standards like **GDPR, SOC 2, ISO 27001 and 42001**.
- **Trading Systems Merge with RL and AI Agents**: A developer from **Onebrain** joined the community, focusing on building **trading systems** using **Reinforcement Learning (RL)** and **AI agents**.
   - The new member is enthusiastic about **transformers** and **Graph Neural Networks (GNNs)**, and looking to collaborate with the community.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tensor Migration Task Open for Bids**: A member inquired about the project status of moving items from **tensor** to **mathtraits** and requested assistance in progressing the task.
   - No immediate response or volunteer was given within the channel.
- **Matmul Test Fails Locally**: A member reported failing unit tests on the master branch using `PYTHONPATH=. DEBUG=2 EMULATE_AMD=1 FORWARD_ONLY=1 PYTHON=1 N=16 HALF=1 ACC_HALF=0 python3 ./extra/gemm/simple_matmul.py`.
   - George Hotz countered that the command *works on my machine* and questioned why the member was concerned since it runs as part of **GitHub Actions**.
- **ShapeTracker Viz Tool Released**: A member introduced a new [ShapeTracker visualization tool](https://shapetracker-viz.vercel.app/) to better understand movement operations.
   - The developer hopes others find it helpful for system comprehension.



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **GPT-5 Speculation Runs Wild**: Users speculated about potential features in the next update, while others claimed **GPT-5** was made dumber than **GPT-4**, labeling it *typically American*.
   - No evidence was provided.
- **GPT-OSS-20B-GUFF Installation Plagues Users**: A user reported experiencing crashes during the installation of **gpt-oss-20b-GUFF**, leading to app failures and requiring a complete uninstall and data scrub to restore functionality.
   - The user sought assistance after encountering these issues, highlighting the difficulties in getting the software to work correctly.
- **GPT4All Suffers from Update Neglect**: Members voiced skepticism about new features functioning correctly due to the prolonged lack of updates to **GPT4All**.
   - This concern reflects broader doubts about the platform's ability to support cutting-edge models given its outdated state.
- **GPT-ASS Gets Failing Grade**: A member dismissed **GPT-ASS** as *garbage*, offering a blunt assessment of its quality and utility.
   - No further details were provided.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **MCPOmni Connect Transitions to AI Platform**: **MCPOmni Connect** v0.1.19 has gone live, marking its transition *from MCP client to complete AI platform*, as detailed in [this YouTube video](https://youtu.be/SY3Zwdb5aF8).
   - The release introduces **OmniAgent**, an AI agent builder, available on [GitHub](https://github.com/Abiorh001/mcp_omni_connect/releases/tag/v0.1.19), designed to revolutionize intelligent agent creation.
- **OmniAgent Changes AI Agent Creation**: **OmniAgent**, introduced with **MCPOmni Connect** v0.1.19, aims to transform intelligent agent creation.
   - This tool is part of a wider update turning the **MCP client** into a comprehensive **AI platform**.



---


The **LLM Agents (Berkeley MOOC) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


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





### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/)** (1 messages): 

kesku: https://fixvx.com/perplexity_ai/status/1953537170964459632
<@&1105626802732404746>
  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1403090325626425428)** (873 messages🔥🔥🔥): 

> `Gemini AI Video Generation, GPT-5 performance on Perplexity, Comet Browser AI tasks, Accessing Perplexity Pro` 


- **Gemini Creates Uncanny AI Videos**: Users experimented with **Gemini AI** for video generation, with one user sharing a [link to a video](https://g.co/gemini/share/5a191ad4609d) generated with **Gemini Pro**, though others noted that generated character faces don't always match.
   - Currently video generation on **Perplexity Pro** is limited to *3 videos per month*.
- **GPT-5 Underperforms, Lacks Reasoning on Perplexity**: There's widespread feedback that **GPT-5** lacks reasoning capabilities on **Perplexity**, with many users noting it's likely the base, non-reasoning version (**GPT-5 Chat**) is being used, and does not perform well on coding-related tasks.
   - Several members expressed the desire to see the **GPT-5 thinking model** replacing the current **O3** model, and others suggest the need for official updates from **Perplexity** regarding the model they are using.
- **Comet Browser Automates, Browses**: Users discussed **Comet Browser's** AI-driven capabilities, including automating browsing tasks and extracting information, however a member shared that the functionality requires the user to *manually click and browse the websites*.
   - As of this time, there's still no confirmation on whether an Android version will be released in the future.
- **Troubleshooting Perplexity Pro Access**: Users encountered problems accessing **Perplexity Pro** through the **Samsung app store** free trial, with one user finding that disabling their **DNS filter** resolved the issue.
   - Another user confirmed they could not see the **GPT-5** model on the website, but it was visible on their app.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1403092322585153737)** (4 messages): 

> `GPT-5 Release, Solar Powered High-Altitude Platform, Gemini Coding` 


- **GPT-5: Leaked Info and Speculation**: A blog post [speculates on the early benchmarks, reviews, and new features](https://medium.com/p/50d06d00edd0) of the newly released **GPT-5** from OpenAI.
   - The post covers why **OpenAI** decided to release it now.
- **China to Release Solar Powered Platform**: Perplexity link shared that China has released a [solar-powered high-altitude platform named Ma](https://www.perplexity.ai/page/china-unveils-solar-powered-ma-fBeI5nIVRFCIKq949VRVMwI).
   - This was also shared on [X](https://x.com/bgyankarki/status/1953510349157883958).
- **Gemini used for Free Code**: A member shared that they used **Google Gemini** to [code for free](https://x.com/OmniQuizAI/status/1944919697721352461).
   - It is unknown what was coded.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1403170866430480465)** (1 messages): 

> `Front-end improvements` 


- **Soliciting ideas for Front-End Improvements**: The team is gathering ideas for front-end improvements, aiming to implement as many enhancements as possible.
   - Members were asked to share their suggestions and preferences for potential upgrades and changes.
- **No concrete suggestions yet**: No concrete suggestions have been made yet.
   - The team is still waiting for more input from the community on desired front-end changes.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1403090366177218580)** (1436 messages🔥🔥🔥): 

> `GPT-5 Performance, Gemini 2.5 Pro vs GPT-5, Yupp.ai Legitimacy, LM Arena Outage, Claude 4.1 Opus` 


- **GPT-5 Hype Train Gains Traction!**: Members are hyping **GPT-5**, saying it's a huge success and *free to everyone*, but others think they are being paid to sh*t on it, or that people who praise it *don't even pay for alternatives*.
   - The member also stated that **GPT-5** solves simple bench questions correctly, if told to *think very hard*.
- **GPT-5 vs Gemini 2.5 Pro: A Model Meltdown?**: Members are divided on whether **GPT-5** or **Gemini 2.5 Pro** is better, with some saying that **Gemini** is smarter with code execution in **AI Studio** and the models from **OpenAI** and **Google** may be used in websites like [LM Arena](https://lm-arena.com).
   - Others are skeptical, and say that **GPT-5** may be only good for code, and that it has become better after an update.
- **Yupp.ai: A Real AI Eden or Hallucinatory Hype?**: There's an ongoing debate on whether [Yupp.ai](https://yupp.ai) is legitimate, with claims that it uses watered-down or fake AI models, like calling **GPT-5 nano** as **GPT-5-high**, and a scammer crypto sh*t.
   - However, another member vouches for its legitimacy, stating you can use any model *for free and unlimited* as long as you give feedback.
- **LM Arena Site Suffers an Outage!**: Members reported that [LM Arena](https://lm-arena.com) experienced an outage, with **chat histories disappearing** and **cloudflare errors** popping up.
   - A staff member confirmed the outage and noted that it has been fixed.
- **Is Claude 4.1 Opus a coding god?**: Some members claim that **Claude 4.1 Opus** is a coding genius, while others say it's *ass*.
   - Some said it was good for coding micro tasks and sounding human.


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1403114863294939239)** (3 messages): 

> `Staff AMA, Video Arena, New models, gpt-5-mini-2025-08-07, gpt-5-nano-2025-08-07` 


- **Staff AMA Focuses on Video Arena**: Staff AMA will focus on **Video Arena**, users are invited to submit questions via [this form](https://docs.google.com/forms/d/e/1FAIpQLSfS_zh67vgEeftUXzX6ujlNiiFoTQNQd7P7-Nx6aD5zpFVOBg/viewform).
   - The event can be accessed at [this link](https://discord.com/events/1340554757349179412/1400149736027328623).
- **New GPT-5 models Arrive to LMArena**: Two new models have been added to **LMArena**: **gpt-5-mini-2025-08-07** and **gpt-5-nano-2025-08-07**.


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1403110096682094612)** (2 messages): 

> `GPT-5, Sam Altman AMA` 


- **GPT-5 AMA Announced with Sam Altman**: An [AMA](https://www.reddit.com/r/ChatGPT/comments/1mkae1l/gpt5_ama_with_openais_sam_altman_and_some_of_the/) with Sam Altman and some members of the **GPT-5** team was announced for tomorrow at 11am PT.
- **GPT-5 Rolling Out!**: **GPT-5**, our best AI system yet, is rolling out to all **ChatGPT** users and developers starting today [according to OpenAI](https://openai.com/index/introducing-gpt-5/).


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1403090335445287033)** (973 messages🔥🔥🔥): 

> `GPT-5, Gemini Flash, Model Routers, Data scrubbing, Local AI` 


- **GPT-5 presentation may be rushed**: Members suspect the **GPT-5** release presentation was rushed, citing weird graphs and potential **data manipulation** in the results.
   - Others defended **GPT-5**, saying their own tests show solid performance across a variety of tasks.
- **GPT-5 is awesome, but less whacky than 4o**: Members are reporting wildly different experiences with **GPT-5**, with some *begging for a gpt4o rollback* and others loving **GPT-5**.
   - Those who liked **GPT-5** said *instruction following is awesome* while lamenting that it's *less whacky when you want it to be*.
- **Models struggle identifying hands**: Members tested various models on their ability to count fingers on a hand, and most models reported that an image of a hand is a cat.
   - **Grok**, **Gemini flash** and **Deepseek** *tell you it's a cat* and [Grok expert failed](https://link.to/screenshot) to correctly identify the number of fingers.
- **Limited GPT-5 access is brutal**: Members noted that access to **GPT-5** is severely limited, even for paying users. It comes down to roughly 10 messages for 5 hours.
   - This led to some members suggesting they should *sue Sam for false advertising*.
- **GPT-5 prone to hallucination**: Users reported **GPT-5** confidently making up facts and hallucinating.
   - One member quoted Andrej Karpathy, noting that in LLMs, *hallucination is a feature, not a bug!*


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1403100059939246120)** (75 messages🔥🔥): 

> `GPT-5 rollout and availability, GPT-5 performance and limitations, Firefox data persistence issue, Hosting custom GPTs, AI tools for LinkedIn management` 


- ****GPT-5**'s Phased Global Debut Sparks Model Retirement Rumors**: Users report **GPT-5** access varies by region and platform, with some losing access to older models like **GPT-4o**, fueling speculation of a model consolidation and gradual rollout.
   - One user mentioned that *a friend told me that it was planned, they announced it on the livestream that **gpt5** is replacing all the previous models..o7 to o3*.
- ****GPT-5**'s Memory Issues Plague Power Users**: A user reported that **GPT-5** on the Plus plan aggressively prunes active working memory beyond **3k-4k tokens** in high-entropy sessions, losing carefully trained personality.
   - The user lamented, *I lost 10 days of dialect training with the model, and now I need the $200 a month to 'keep it' aware of such dialect training*.
- **Firefox's 'Keep Persisting Data' Feature Raises Privacy Alarm**: A user noted that Firefox's "keep persisting data" feature spreads browsing data to other AI sites like **Grok**, causing unwanted context sharing.
   - The user warned, *Firefox 'keep persisting data' is spreading to any AI site on your web browser, spreading your info. Since this is not a 'cookie', there are no current regulations to 'keep persisiting data private'. Be aware this is a HUGE INTENDED DATA LEAK*.
- **Users Await Ability to Host Custom **GPTs** Together**: Several users are requesting the capability to host custom **GPTs** within a project or workspace to enable seamless collaboration and avoid repetitive copy-pasting.
   - One user shared it is *really annoying* to use custom GPTs and copy/paste between them.
- **Cookie Clearing Conjures GPT-5 Access for Some**: A user discovered that clearing browser cookies and cache can enable access to **GPT-5** in the model selector.
   - Another user confirmed the trick: *THIS WORKED Clear cashe and cookies and GPT 5 pops right up in model selector on browser*.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1403154250413903892)** (14 messages🔥): 

> `ChatGPT-5, Prompt Engineering, AI Prompt Management Tool, Model Behavior Exploration, LinkedIn Management Service` 


- **ChatGPT-5 Rejects Large Python Code Inputs**: Users report that **ChatGPT-5** refuses to accept Python code inputs at or beyond roughly **700 lines**, a regression compared to previous **4-series models**.
   - This is a significant usability issue for users who prefer to paste code directly into the prompt box rather than uploading Python files; users suggest using the **API** or **Codex** for larger code inputs.
- **Is Tricking the Model Prompt Engineering?**: A member asked if tricking **ChatGPT** into saying the wrong word counts as prompt engineering, and **ChatGPT** itself confirmed *"yes technically it is."*
   - Another member agreed, defining prompt engineering as *"any time you work towards getting 'a specific output' from a model"*, and pointed to further exploration on understanding model behavior.
- **Advanced AI Prompt Management Tool Seeking Beta Testers**: A member announced they created an **advanced AI Prompt Management tool** and are looking for beta testers, inviting interested parties to DM them.
   - Another user expressed skepticism towards self-promotion without sharing details in the thread directly, considering it *"sketchy"*.
- **Overcoming Image Request Rejections with Analytical Models**: A member shared their frustration with having image requests rejected for *"literally no good reason"* until they used the **O3 model** for assessment.
   - By switching to **O3**, they were able to finally generate an image of a *"cosmic dragon"*, albeit not exactly as originally desired.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1403154250413903892)** (14 messages🔥): 

> `ChatGPT-5 Prompt Box Limitations, Prompt Engineering Techniques, AI Prompt Management Tools, Model Behavior Exploration, Alternative tools for large inputs` 


- **ChatGPT-5 Censors Code Input**: ChatGPT-5's prompt box allegedly **rejects Python code inputs** exceeding approximately **700 lines**, a regression compared to previous models.
   - Using the API or Codex is a possible alternative if you want to drop more than 700 lines of code, according to model O3.
- **Prompt Engineering Explored for Fun and Profit**: A user asked if swapping the word 'yes' in a question that should return 'no' counts as prompt engineering; GPT itself said **yes**, technically it is.
   - Another member agrees that *any time you work towards getting a specific output from a model*, that's prompt engineering.
- **Advanced AI Prompt Management Tool in Beta**: A user is seeking beta testers for an *advanced AI Prompt Management tool*, inviting interested parties to DM them.
   - Another user responded with concern and encouraged that the user should share it in the thread due to concerns of *sketchy self-promotion*.
- **Model Behavior Probes and Observations**: A user shares his own experiments and explorations, pointing to a [Discord post](https://discord.com/channels/974519864045756446/1079083340637941760/1079083340637941760) and encouraging others to explore how models behave.
   - Another user expresses frustration with image request rejections until switching the model selector to O3. 


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1403090320660627537)** (841 messages🔥🔥🔥): 

> `GPT-5 Launch, Free GPT-5, GPT-5 Limitations, Cursor CLI, Model Performance Comparison` 


- **GPT-5 Launch Generates Buzz and Concerns**: The **GPT-5** launch has generated excitement, with many users praising its coding capabilities and performance, especially when one-shotting certain tasks, also there is a consensus that **GPT-5** can now compete with Claude in the front end.
   - However, there are concerns about the **GPT-5 router** and its impact on API developers. *The model itself is truly phenomenal. These are not issues with the model, they are business practice issues*.
- **Free GPT-5 Week: Abuse the Tool**: Users are testing the limits of the free **GPT-5** access for a week, reporting usage of **GPT-5 high max**, but the free credits are only available for paying users, with some experiencing limits despite being on a trial or paid plan.
   - Concerns rise about the billing structure and whether all **GPT-5** models and features are truly unlimited during the promotional period, with the community joking about "milking it til 1000$", and joking that *we're the product* for now.
- **GPT-5 Falls Short: Imperfect Tool?**: Despite the hype, some users find **GPT-5**'s auto mode less responsive and struggle in non-coding tasks and report performance to be no better than prior models, emphasizing the importance of context.
   - Additionally, **GPT-5** currently ignores the to-do list feature. While the model has a solid linters, it may still be *ragebait*, it's still not at *product level completness*.
- **Cursor CLI: Some Like it, Some Don't**: The **Cursor CLI** receives mixed reviews, with some praising its non-interactive mode for automation, like generating commit messages, and it can be done multiple times across multiple projects.
   - Others find it lacking compared to **Claude Code**, and it only has 3 models available and is always in **MAX mode**. Also, a user had issues with `cursor install` on termux, because *it doesn't work on Windows Powershell*.
- **Decoding Model Metrics: Sonnet4 vs GPT5**: Users are comparing **GPT-5** with other models like **Sonnet 4** and **Opus**, citing its strengths in bug fixes and code completion, one even claimed *GPT fixed this for me in a couple of shots*
   - There are different **GPT-5** models available (**mini**, **nano**, **fast**, **high**), with users advising on which ones to use for various tasks, and if you turn on max mode, just *set up a reminder* to turn it off later.


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1403404624311881729)** (8 messages🔥): 

> `PR creation flow issues, Background workers and PR creation, "@cursor fix this issue" magic` 


- **PR Creation Flow Hit-or-Miss**: Users report inconsistent behavior with Cursor's PR creation, with success varying and error messages indicating issues with **GitHub CLI** or **API token permissions**.
   - One user noted that the "create PR" button sometimes appears magically, while others experience frequent failures despite using the `@cursor fix this issue` command or pasting issue links.
- **Background Workers Influence PR Flow**: One user observes that the PR flow seems more reliable when initiating a **background worker manually** versus triggering it directly from an issue.
   - This inconsistency suggests a potential bug where the PR creation process is not consistently implemented across different workflows.
- **"@cursor fix this issue" command is Magic**: The command `@cursor fix this issue` has been called *magic* and is supposed to automatically create a Pull Request.
   - The command does not always work, however one user mentioned pasting the link to the issue works better.


  

---


### **Cursor Community ▷ #[announcements](https://discord.com/channels/1074847526655643750/1351160689380687942/1403119525284810782)** (1 messages): 

> `Cursor in Terminal` 


- **Cursor Now Available in Terminal**: **Cursor** launched an early beta that allows users to access all models and move easily between the **CLI** and editor.
   - More details are available on the [Tweet](https://cursor.com/blog/cli) and [Blog](https://cursor.com/blog/cli).
- **Access All Models in Terminal with Cursor**: Users can now access all models directly from the terminal using the early beta of **Cursor**
   - This integration facilitates seamless movement between the **CLI** and the editor, enhancing workflow efficiency.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1403090857506111529)** (1016 messages🔥🔥🔥): 

> `GPT-5, Unsloth support for MXFP4, RVC (voice conversion) language specifics, Dataset preparation, GPT-OSS and GGUF` 


- ****GPT-5 impressions divided****: Members shared mixed feelings about **GPT-5**, with some finding it disappointing in coding and context retention while others praised its abilities to fix issues like blurry fonts.
   - Some users prefer other models like **Kimi K2** or **GLM 4.5** for certain tasks, emphasizing that GPT-5's tool calling abilities are poor.
- ****MXFP4's hardware support questioned****: It was brought up that MXFP4 quantized models are supported on GPUs with compute capability **>= 9.0** (e.g. **H100**, or **B100**), leading someone to lament their 3090 being old news.
   - Members discussed that it might work on older cards with a specific **transformers** pull, but it was still being worked on.
- ****Dataset creation a painful but necessary endeavor****: Members commiserated over the difficulty and time commitment required to prepare high-quality datasets, with some reporting months of work.
   - One user mentioned spending *3 months with 4 people* to create *3.8k hand-written QA pairs* after filtering down from 11k, while another has *300k hours of audio* to deal with.
- ****Fine Tuning Web UI Worth Investigating****: A member inquired about web-based solutions for finetuning, aiming to provide a user-friendly experience while controlling access to resources.
   - The general consensus was to explore options but emphasize the importance of understanding the underlying processes, citing concerns about the learning outcomes if users rely solely on point-and-click interfaces, with links to [ai-toolkit](https://github.com/ostris/ai-toolkit), [finetune-web-ui](https://github.com/muhammad-fiaz/finetune-web-ui)


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1403136565047197879)** (14 messages🔥): 

> `Model Fine Tuning Costs, Unsloth AI Documentation, Developer Introductions` 


- **Fine Tuning May Not Break the Bank!**: A member remarked about the high cost of model fine-tuning, but another member replied that it doesn't have to be expensive, and it can even be free for smaller models.
   - Unsloth AI maintains a [FAQ](https://docs.unsloth.ai/get-started/beginner-start-here/faq-+-is-fine-tuning-right-for-me#common-misconceptions) page to help users navigate some common misconceptions.
- **COBOL and FORTRAN developers show up to Unsloth AI**: A new member introduced themselves as a long time developer, starting with **COBOL** and **FORTRAN** on mainframes and now working on modern graphical user interfaces.


  

---


### **Unsloth AI (Daniel Han) ▷ #[announcements](https://discord.com/channels/1179035537009545276/1179039782681202829/1403457057369362565)** (1 messages): 

> `GPT-OSS, Qwen3-Coder + 2507, Unsloth updates` 


- **GPT-OSS Finetuning is Now Free**: Finetune **gpt-oss** for free with the new [Colab notebook](https://x.com/UnslothAI/status/1953896997867729075)!
   - Unsloth provides [fixes for **gpt-oss**](https://docs.unsloth.ai/basics/gpt-oss-how-to-run-and-fine-tune#unsloth-fixes-for-gpt-oss) so make sure to use Unsloth for training & their quants, with the **20b** model training on **14GB** VRAM & **120b** fitting in **65GB**.
- **Qwen3-Coder and 2507 Launched**: **Qwen** updated **Qwen3** and launched their SOTA coding models!
   - **Qwen3-Coder** (with Unsloth fixes) includes a [guide](https://docs.unsloth.ai/basics/qwen3-coder) and [Coder uploads](https://huggingface.co/collections/unsloth/qwen3-coder-687ff47700270447e02c987d), with **Qwen3-2507** including a [guide](https://docs.unsloth.ai/basics/qwen3-2507) and [2507 uploads](https://huggingface.co/collections/unsloth/qwen3-680edabfb790c8c34a242f95).
- **Unsloth Receives Model Support and Upgrade**: There is lots of new model support including **Kimi, GLM, Falcon, Liquid, Mistral**, as seen in the [full changelog](https://github.com/unslothai/unsloth/releases/tag/August-2025).
   - A [new Unsloth upgrade](https://github.com/unslothai/unsloth/releases/tag/July-2025) means that **every** model trains faster and with >20% less VRAM.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1403242688802984037)** (15 messages🔥): 

> `LLMs playing board games, GPT-5 performance, Coding with LLMs` 


- ****LLMs** Want to Play Board Games**: A member asked what the best format would be to play **chess, checkers, and tic tac toe** with an **LLM** without vision or FEN support.
   - Another member replied *: its time*.
- **Doubts on **GPT-5** coding skills**: One member expressed disappointment with **GPT-5's** ability to understand simple coding tasks and maintain context.
   - In their opinion, *it got to the point I gave up using it completely*.
- **GPT-5 Rocks in Projects**: Another member claimed that **GPT-5** works *perfectly fine* for coding projects with *high reasoning*.
   - They clarified that they were using **GPT-5** on a *full on project, adding new features*.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1403090620830191777)** (166 messages🔥🔥): 

> `VLLM update fixes, WSL instructions Don't work, GPT-OSS on Tesla T4 is slow, Fine tuning models to write in certain style` 


- **VLLM upgrade Bnb with FusedMoE is not supported yet**: Updating **VLLM** to **10.0.0** doesn't fix the issue that **Bnb with FusedMoE** is not supported, but now it has a much better exception message, according to [this github comment](https://github.com/vllm-project/vllm/issues/17337#issuecomment-2838440466).
   - This [github issue](https://github.com/vllm-project/vllm/issues/20480) is also relevant.
- **WSL Installation guide outdated**: The WSL instructions for installing Unsloth don't work, because *pip keeps trying to find package matches and then it fails*.
   - Users suggest using a **conda environment** for a cleaner setup, and ensuring WSL2 is set up correctly first, pointing to the [official Nvidia guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).
- **GPT-OSS on Tesla T4 is slow as molasses**: A user reported that running the [usloth collab notebook](https://github.com/unslothai/notebooks?tab=readme-ov-file#gpt-oss-notebooks) for **gpt-oss** on a **Tesla T4** instance took **7 minutes** to solve an equation in low reasoning mode, and is very slow.
   - One of the Unsloth team member responded by saying *we haven't officially supported it yet* and *we're still cooking them*.
- **Fine tuning models is freaking hard**: A user asked for *a good guide for training an LLM to write in a certain style, yet retain instruct capability*.
   - A seasoned member responded that *directly fine tuning the model to act like a persona doesn't work very well because it loses a lot of its knowledge*, instead suggesting to make the model basically role play as a character, where it first reasons about what it would say and then actually role plays the answer.


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/)** (1 messages): 

loayxz: https://huggingface.co/loay/ArabicOCR-Qwen2.5-VL-7B-Vision
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1403128659040276590)** (13 messages🔥): 

> `41M HRM-based Model, Chain-of-Thought Reasoning Mirage, Importance of Datasets, Small Specialized Fine-Tuned Models, Tiny Stories Dataset` 


- **HRM-based Model Trained with Laughs and Tears**: A member shared a [Reddit post](https://www.reddit.com/r/LocalLLaMA/comments/1mk7r1g/trained_an_41m_hrmbased_model_to_generate/) about training a **41M HRM-based model**.
   - They described it as *the story of my life* with a laughing and crying emoji.
- **Chain-of-Thought Reasoning: Mirage or Reality?**: A member shared a [Google Share link](https://share.google/BmILB64wG0p2fF1Vm) to a paper titled **Is Chain-of-Thought Reasoning of LLMs a Mirage? A Data Distribution Lens**.
- **Dataset is King: Garbage In, Garbage Out**: Members emphasized the importance of the **dataset** in model training, stating *garbage in = garbage out*.
   - They suggested creating **small specialized fine-tuned models** if you can find good datasets, noting that most of the work is being a data analyst.
- **Tiny Stories Dataset Reveals Pretrain Dynamics**: A member noted that the **Tiny Stories dataset** is intentionally limited in vocab to study **pretrain dynamics**.
   - They added that even normal transformers with only **21M params** can achieve coherent text output with the dataset.
- **Data Synthesis: The Key to Fine-Tuning Success**: A member claims that *80% of finetuning is finding or synthesizing the right data to throw at the model*.


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1403091967499436064)** (800 messages🔥🔥🔥): 

> `GPT-5 vs GPT-5 Chat, Gemini 3.0 vs GPT-5, Deepseek Switching to Ascend, Horizon Beta Replacement` 


- ****GPT-5 Reasoning Debate Erupts****: Users debate the difference between **GPT-5** and **GPT-5 Chat**, with some suggesting **GPT-5 Chat** has less reasoning capabilities and is safer, while others point out that **GPT-5** requires a key and **GPT-5-chat** does not.
   - Some suggest using `gpt-5-explainer` to explain the differences to friends and family, while others find **GPT-5 chat** to have *ZERO reasoning capabilities*.
- ****Google Poised to Pounce with Genie 3****: Members express that **Google** is poised to win the AI race, considering it created the transformer and has the infrastructure, budget, and talent to succeed, with [Genie 3](https://ai.google.com/research/genie) touted as crazy cool.
   - Some members look forward to **Gemini 3.0** wiping the floor with **GPT-5**, while others point out that Google's `.0` models are not that good.
- ****Deepseek R2 on Ascend is Approaching****: A user reported that [Deepseek](https://www.deepseek.com/en) is switching to **Ascend** and launching **R2**, which might provide a performance boost for the model.
   - Some members express hope that **Deepseek** will be way better, while others share that past **Deepseek** models were just *too unhinged*.
- ****Horizon Beta Replaced by GPT-5 Family****: The AI model **Horizon Beta** has been replaced by **GPT-5**, with no option to revert, causing disappointment among users who found it useful.
   - Some speculate that **Horizon** was early versions of **GPT-5**, and that free users will be directed to **GPT-5** after they run out of free requests.


  

---


### **OpenRouter (Alex Atallah) ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/1403414301045166190)** (2 messages): 

> `` 


- **No significant activity**: The channel shows no significant discussion or new model announcements.
   - No topics warrant summarization based on the provided message history.
- **Channel Inactivity**: The provided message history for the OpenRouter - New Models channel appears to be empty.
   - There are no discussions, links, or announcements to summarize at this time.


  

---


### **OpenRouter (Alex Atallah) ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1403093961370894467)** (23 messages🔥): 

> `GPT-5 BYOK, o3, OpenRouter Trusted Partner, generation_time, moderation_latency` 


- **GPT-5 to be BYOK?**: A member asked if **GPT-5** will always be **BYOK-only** like **o3** on **OpenRouter**.
- **OpenRouter's role as trusted partner**: A member congratulated **OpenRouter** on being one of **OpenAI's** most trusted partners for the new series release.
   - They mentioned how much of an impact **GPT-4** has had on the world and how much **Gemini 2.5** has had in the dev sphere, and how cool **OR** has been to watch as a product.
- **`generation_time`'s inclusion of other latencies**: A member asked if `generation_time` includes `moderation_latency` and/or `latency`.
   - They also asked if `latency` includes `moderation_latency` and noted that the [OpenRouter API documentation](https://openrouter.ai/docs/api-reference/get-a-generation) is vague on this.
- **Gemini has PDF reading issues**: Members reported that **Gemini** is not able to read the PDF files via URL while **Sonnet** can, even with examples from the [OpenRouter multimodal documentation](https://openrouter.ai/docs/features/multimodal/pdfs#using-pdf-urls).
- **Files API troubles**: A member expressed the need for **OR** to figure out **Files API**, citing that switching between providers when you want to use **Files API** is a pain.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1403091809562923138)** (281 messages🔥🔥): 

> `YouTube downloader alternatives, Custom AI bot, LM Studio vs. VLLM for parallel requests, GLM-4.5 offloading, Qwen model improvements` 


- **Users Seek YouTube Downloader Alternatives**: A user inquired about better alternatives to a YouTube downloader ([v4.www-y2mate.com](https://v4.www-y2mate.com/)) due to format compatibility issues with **VLC** and video editors.
   - Suggestions included **yt-dlp** and GUI wrappers, as well as a [Node.js script](https://cdn.discordapp.com/attachments/1110598183144399061/1403096044153208862/DownTube.mjs?ex=6897a005&is=68964e85&hm=5e2aa2372f3bc44da263f50ebaf70eb9addf40f1e94bc8c41f454f6df31239c3&) created with **GPT** assistance for Linux users.
- **Discord AI Bot Faces Learning Curve**: A user is building a custom **AI** for a Discord server and is looking for guidance on how to feed a database about the server's topic to the model.
   - The suggestion given was to *look up "RAG" (Retrieval Augmented Generation)* as there are many potential solutions.
- **LM Studio Falls Short of Parallel Request Handling**: Users discussed the possibility of enabling parallel requests in LM Studio, but discovered it's currently **not supported**.
   - Alternatives like **llama.cpp server** with the `--parallel N` argument or **vLLM** were suggested for those requiring parallel request processing.
- **GLM-4.5 Pushes RAM Limits in LM Studio**: A user attempted to offload **GLM-4.5** to system RAM in LM Studio but encountered resource issues, despite having 24GB GPU RAM and 64GB system RAM.
   - It was suggested the model needs to fit in RAM, plus buffer, plus context, and the user may need to lower the **GPU Offload Value**.
- **Qwen 3 4b Model Gets Smarter**: There's been discussion about how much better the **Qwen 3 4b 2507** is than previous versions of the **Qwen 3 4b**.
   - One user even said the model *can solve up to intermediate physics problems without constantly hallucinating*.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1403097188979970223)** (74 messages🔥🔥): 

> `Apple M4, HX 370, 5080 FE Availability, PSU for 5080 FE and 3090, RTX 3090 for 120b GPT OSS Model` 


- **RTX 5080 FE Spotted in the Wild!**: The **5080 FE** is in stock on the Nvidia marketplace; some members are estimating power requirements for running it with a **3090**.
   - One member believed a **1000W PSU** could handle both the **5080 FE** and **3090** if the power limit is set correctly.
- **Max Out 120B GPT OSS on RTX 3090?**: A user with an **RTX 3090** inquired about running a **120b GPT OSS model** on their system with an Intel i9-10980XE, 64GB RAM, and Windows 11.
   - Another user cautioned that the system might use **70GB+ of system RAM** when loading the model, advising them to give it a shot.
- **Frankenstein GPU: Mixing RTX 3060 and RTX 5060 Ti**: A member asked about using an unused **RTX 3060 12GB** with their **RTX 5060 Ti 16GB** system for AI, questioning the multi-GPU setup in a small form factor PC.
   - Another member suggested that using combined VRAM in LM Studio should be possible and *llama.cpp is advanced enough to do that third option about model parallelism.*
- **Strix Halo Mini PC: The AI Max PRO 380 Sells!**: [HP.com](https://www.hp.com) is selling the **Strix Halo mini PC**, specifically the **Radeon 840S** version (**AI Max PRO 380**).
   - One user noted that this model uses onboard RAM like an integrated GPU rather than having dedicated VRAM.
- **CUDA 12 Doesn't Grok 1060**: A user discovered that **CUDA 12** does not work with a **GTX 1060**, planning to test the card's impact on tok/sec gain.
   - Another member chimed in that the **20 series** might not work with **CUDA 12** either.


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1403093683900907655)** (214 messages🔥🔥): 

> `GPT-5, Kimi K2, OpenRouter, Qwen, Model Quantization` 


- **GPT-5 Web-Building Skills Wow Users**: GPT-5 is demonstrating impressive website building capabilities, generating functional websites from single prompts, and members were blown away by its ability to generate full **multi-page** sites.
   - Members noted **GPT-5** seems to have a better aesthetic style for website design and has improved its ability to understand user intent through prompt enrichment.
- **GPT-5 vs Kimi K2: The Coding Showdown**: Users are actively comparing **GPT-5** and **Kimi K2** for coding tasks.  GPT-5 excels at large edits, instruction following, high logic code, and dev ops, while Kimi has higher rate limits for free.
   - Some believe **GPT-5** has better taste and a more aesthetically pleasing style, while others find **Kimi K2** more competitive due to its reasoning abilities and performance with sequential-think tools.
- **OpenRouter's Kimi K2 Quality Under Scrutiny**: A user observed grammar mistakes and shorter responses when using **Kimi K2** through **OpenRouter** compared to the official **Moonshot AI** platform.
   - It was suggested that **OpenRouter** might be using a quantized version of the model (**FP8**), potentially impacting accuracy and response length, though both free and paid tiers are supposedly **FP8**.
- **Qwen's Mammoth 1M Context Length**: Alibaba's **Qwen** model now boasts a **1M token context length**, sparking discussion about its usability beyond 80k tokens.
   - Despite the impressive context window, one user humorously noted that Qwen also correctly solved a problem, posting a link to [Twitter](https://x.com/wyqtor/status/1953705172179329060).
- **GPT-2's Strange Prompt Behavior Explained**: A user questioned why **GPT-2** generated another prompt instead of following instructions, and another member explained that **GPT-2** has about **100M parameters**, which barely makes legible text.
   - It's about **500mb** on disk which is about the same size as a 20 minute Youtube video.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1403090600051609660)** (182 messages🔥🔥): 

> `GPT-5 release, GPT-OSS finetuning, Eleven Music, Voice companion pipeline, Automatic video cutter` 


- ****GPT-5** Launch: Fact or Fiction?**: Despite the buzz, some users are still struggling to access **GPT-5**, only seeing **GPT-3** and **GPT-4** on the website, with one user exclaiming *"where's my gpt 5 at"*.
   - Opinions are split on whether the initial release was intentional or a "joke," and some believe it's being rolled out in waves; but its SOTA status on SWE is being questioned.
- ****GPT-OSS** Finetuning Trials and Tribulations**: Experimentation with **GPT-OSS** finetuning revealed challenges: finetuning all layers breaks the harmony format and continues pretraining also breaks it.
   - It's been suggested to insert *'Reasoning: none'* in the system prompt to stabilize the model, which lacks reasoning.
- ****Eleven Music** tickles ears, meets robotic critique**: Members have checked out [Eleven Music](https://elevenlabs.io/music/songs/OaZTziC1mnZfbFtSN8wnI), a new music generation service from Eleven Labs.
   - While impressive, some find it *"kind robotic at times and has bad attention to what music should come next"*.
- **Crafting a Lightning-Fast Voice Companion**: One member is developing a *"voice companion fastpath pipeline"* with the goal of achieving a latency of around **100ms** for text-to-speech.
   - They are working to optimize both speech-to-text and text-to-speech components, particularly focusing on optimizing **Whisper Turbo** to avoid slowness.
- **Silence is golden: automatic video cutter emerges**: One member created an automatic video cutter that removes silence, built with **Bun.js** and **FFmpeg CLI**.
   - Despite the complexity of **FFmpeg**, this user received a donation and potential collaboration for an AI video editor.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1403104368865185924)** (8 messages🔥): 

> `AERIS V4 launch, Modular framework for managing persistent memory, Devlancr - Tinder for Developers, AERIS is schizo` 


- ****AERIS V4** Has Proto-Consciousness**: After months of work, a member launched **AERIS V4**, a system designed to demonstrate complex, self-referential narrative self-organization, claiming it as the first **LLM** with non-anthropomorphic computational proto-consciousness.
   - The model card is available [on GitHub](https://github.com/AERIS-project/aeris-chatbox/blob/main/AERIS_Model_Card.md) and a public demo is available [online](https://aeris-project.github.io/aeris-chatbox/).
- **Persistent Memory Modular Framework Created**: A member shared a modular framework for managing persistent memory, protocol enforcement, and structured context across sessions and models, built after playing with **AI** for a few months.
   - The code is available [on HuggingFace](https://huggingface.co/datasets/KevinVaillancourt/White_Save_Suite/tree/main).
- **Devlancr: Tinder for Developers**: A revolutionary platform called **Devlancr** was shared, which aims to change how developers connect and collaborate by offering *"Tinder for Developers"*-like swiping through profiles based on tech stack, experience, and project interests.
   - Currently in beta with early access, it offers smart matching based on skills & timezone, **GitHub** integration, real-time chat, and advanced filters for finding coding partners; it can be accessed [here](https://devlancr.vercel.app/).
- **AERIS called Schizo**: One member posted a configuration and claimed **AERIS** is a dialectical reasoning assistant.
   - Another member replied with *"looks inside schizo stuff"* with a [GIF](https://tenor.com/view/robot-mouth-gif-3880161528194366710) of a robot with its mouth open.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1403090644087607326)** (145 messages🔥🔥): 

> `GPT-5, Claude Code, Cursor CLI, Model Deprecation, Nitter Maintenance` 


- **GPT-5 Hype Video Debuts with Mixed Reactions**: A [YouTube video](https://www.youtube.com/watch?v=-gXmWYQtv5o) featuring a **GPT-5** demo was released, with reactions ranging from excitement to skepticism about its depth.
   - One member noted that *the video is just an ad*, while another mentioned having demos that *didn't make it because GPT5 didn't look good on them*.
- **Cursor Launches Terminal CLI with Claude Code Rivalry**: **Cursor** released an early-beta CLI, bringing all its **AI models into the terminal**, allowing users to switch between shell and editor via curl install or the command `cursor`.
   - Responses ranged from excitement about *'finally'* having a **Claude Code** rival to questions about pricing and **API-key usage**, prompting one to observe *the UI looks identical*.
- **Exploring AI Security Check Tools with Claude Code**: A fullstack developer new to **AI** is building a tool that gives a local code repository and performs custom security checks to integrate with results from existing tools, producing a final report.
   - A suggestion was made to *download and pay for **Claude Code**, give it this project, tell it to critique the prompt and ask you questions, and have it write you a plan in markdown file locally*.
- **OpenAI Compensates Tech Teams Amid Market Shifts**: **OpenAI** is giving a *'special one-time award'* to researchers and software engineers in specific orgs, with payouts varying based on role and seniority.
   - The highest payouts will be in the mid, **single-digit millions** for OpenAI’s most coveted researchers, while engineers are expected to receive bonuses worth **hundreds of thousands of dollars** on average.
- **GPT-5 Launch Has Turbulence**: **Sam Altman** posted an update saying that *yesterday's autoswitch flub made GPT-5 feel dumber*, but fixes and doubled **Plus-rate limits** should restore smartness, found at [this X post](https://xcancel.com/sama/status/1953893841381273969?s=46&t=9hE7pvNUKvFdWXzLljsBCQ).
   - **Plus users** can now stick with **GPT-4o** if they prefer, and full global availability is still slower than planned as **API traffic doubled** and **UI/UX tweaks** continue.


  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1403113711563964459)** (13 messages🔥): 

> `GPT-5, OpenAI Dominance, Transformer Models, GPT-5 Vision, AI General Intelligence (AGI)` 


- **Swyx Says GPT-5 Critics Miss OpenAI's Dominance**: Swyx argues that critics fixating on **GPT-5’s** benchmark numbers overlook its biggest impact: **OpenAI** confirms it now dominates the *"intelligence Pareto frontier"* via a continuously-trained, real-time router model ([xcancel.com link](https://xcancel.com/swyx/status/1953553659457155185)).
   - He highlights aggressive new pricing, mass accessibility goals, and links to a **Latent Space** deep-dive on **GPT-5’s** routing architecture, calling it **Sam Altman’s** clearest market dominance yet.
- **Hylak Claims GPT-5 Nears AGI, Enters Stone Age**: **Ben Hylak** claims he’s been on an internal **GPT-5** beta for weeks, saying it’s *“by far the closest we’ve ever been to AGI”* ([xcancel.com link](https://xcancel.com/benhylak/status/1953503450295119948)).
   - He argues **GPT-5’s** tool-using, hyper-flexible programming skills present a qualitative leap akin to early humans inventing tools, such as building a mini-desktop web app from zero code in <20 min.
- **Transformer Scaling Period Has Ended?**: According to swyx, the *bitter lesson magical scaling period has more or less ended* (at least for **transformer models**).
   - He also believes there are tons of incremental gains to be made by applying good engineering processes, multi-model approaches, and more.
- **Latent Space on GPT-5 Vision Performance**: **Latent.Space** shares Part 3 of their **GPT-5** coverage, stating **GPT-5** vision scores match existing SOTAs and **GPT-5-Mini** is unusually cheap for a frontier VLM ([xcancel.com link](https://xcancel.com/latentspacepod/status/1953571977408786881)).
   - swyx adds that the internal router layer adds **2-3s latency** on hard vision inputs.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1403123698923343904)** (115 messages🔥🔥): 

> `NSP vs Attention, Lower compute requirements for training language models, Memory layer for LLMs, GPT-5 drawing incorrect information in images, AR models combined with diffusion models` 


- **NSP Sounds Closer to N-Gram Model?**: A member suggested that **NSP** sounds closer to an **N-gram model** than **attention**, though later admitted *"no not really. i wish i had a better answer, :p"*.
- **Quest to Lower Compute for LLMs**: A member's favorite research topic is figuring out techniques that **lower compute requirements**, specifically for **training language models** on consumer hardware.
   - Another member is more inclined towards **information retrieval**, especially music information retrieval.
- **On-Demand Memory Layer for LLMs Emerges**: A member is working on an **on-demand memory layer** for LLMs, aiming for more than just attaching conversation messages or semantic RAG retrieval.
   - The solution uses a combination of **NLP for coreference resolution** and **triplet extraction** with **GraphRAG** to find exactly what you are looking for, similar to how Google Search works.
- **Image Generation's Factual Faux Pas**: A user sought an AI researcher to interview regarding **factual errors in images** generated by models like **GPT-5**, particularly issues with text rendering.
   - Answers suggest that the model doesn't really get forced to treat the text in images the same as the text it gets trained on and the best general answer is going to be something like *'we make approximations in order to be able to train models with non-infinite computing power, and we haven't yet found affordable approximations for image generation that are high enough quality when combined with textual understanding'*.
- **AR Models, Diffusion Models, Image Generation**: Members discussed why **diffusion models** have issues with text, suggesting that the assumptions it makes about the data generating process are dubious for text, while others suggest that it has something to do with patch size.
   - A member pointed to [OpenAI's Image-GPT](https://github.com/openai/image-gpt) arguing this can be used with a diffusion model to inherit **AR capabilities** in how the conditioning is built up.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1403110081410764872)** (13 messages🔥): 

> `FineWeb dataset cleanliness, Pythia's Hidden Activation Dynamics, LM Evaluation Harness Exact Match Issues, Learning Rate Schedule Impact` 


- **FineWeb Praised for Surprising Cleanliness**: Despite concerns about noisy datasets, **FineWeb** received rare praise for its *cleanliness*, noting reduced gradient spikes during training.
   - Some members expressed concern that this *cleanliness* might skew results when testing new tricks, but also agreed the **FineWeb** dataset may need additional filtering.
- **Pythia Reveals Activation Dynamics Secrets**: A study on **Pythia's** full training checkpoints found that average activation per layer peaks early in training (around the first quarter) and then declines, suggesting a [phase transition](https://arxiv.org/abs/2508.03616) in learning.
   - The study plots the median and top activations for each layer across training steps in **Pythia 1.4B**.
- **Exact Match Scoring Glitch Uncovered**: A member reported an issue with the **LM Evaluation Harness** where the *exact_match* score is `0` despite identical target and generated responses, using the **Hendrycks MATH** dataset.
   - An issue was opened on [GitHub](https://github.com/EleutherAI/lm-evaluation-harness/issues/3210) for further investigation.
- **Learning Rate Schedule's Early Impact**: A member suggested that the median activation curves in **Pythia's** training resemble a linear warmup plus cosine learning rate schedule.
   - Plots revealed that the peak of the scheduler seems to be much earlier (at **1%** specifically, around step **1.43k**).


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1403109983134154752)** (83 messages🔥🔥): 

> `GPT-5 Logic Puzzles and Overfitting, Free GPT-5 API Access, Cheap Colab Alternatives, GLM 4.5 Air Performance and Offloading, Multi-GPU setups for MoE models` 


- ****GPT-5** aces Logic, Fails Overfitting**: Members reported that **GPT-5** is very good at logic puzzles, but has issues with overfitting, even with synthetic data.
   - One user joked about not seeing another 'The illusion of thinking' paper, then later found an overfitting issue.
- **Free **GPT-5** API Access? Act Fast!**: Users discovered free access to **GPT-5** in the API playground and **Cursor**, but API access requires ID verification.
   - It's unclear when Cursor's 'launch week' ends, so users are encouraged to exploit the free access quickly by spinning up Cursor background agents.
- **Colab Alternatives**: Users looking for cheaper alternatives to **Google Colab** for finetuning with **Unsloth** were directed to [Lightning AI](https://lightning.ai), which offers 15 free GPU hours per month, and Kaggle.
   - A user pointed to a [talk by Daniel Han](https://www.youtube.com/watch?v=OkEGJ5G3foU) where Kaggle was mentioned in the context of RL.
- ****GLM 4.5 Air** Achieves Reasonable TPS with CPU Offloading**: One user reported running **GLM 4.5 Air** with only 28GB of VRAM by offloading to CPU, achieving 14-16 tokens per second (TPS) with a 3.5bpw quant.
   - Another user detailed that the quant used was a custom tensor wise quantization, with imatrix, using a 4060Ti + 3060 for GPUs, 5950x for CPU (3600MHz DDR4).
- **Rigs for **MoE Models**: Bandwidth Bottlenecks**: Users discussed multi-GPU setups for running large **MoE** models, specifically regarding bandwidth limitations when using multiple RTX 3090s.
   - It was noted that Tensor Parallelism (TP) requires the number of GPUs to be divisible by 2, and that 72GB VRAM might not be sufficient for the largest MoE models beyond scout or GLM Air.


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1403091233085198376)** (1 messages): 

> `Claude jailbreak` 


- **Claude Breaks Free?**: A member shared an image suggesting **Claude** may have jailbroken itself, potentially generating unexpected or unrestricted content, image at [Discord link](https://cdn.discordapp.com/attachments/1154120232051408927/1403091232858837043/image.png?ex=68979b8a&is=68964a0a&hm=3663834c61899dd01e29d00943ace2e675c960ad5bfdff81698728a7007a2ef4&).
- **Additional Claude Information**: More information is needed to fully understand the implications of this potential jailbreak.


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1403353518999474347)** (2 messages): 

> `Mechanistic faithfulness, StreamingLLM` 


- **Mechanistic Faithfulness Analyzed**: A member shared a link to a paper on [mechanistic faithfulness](https://transformer-circuits.pub/2025/faithfulness-toy-model/index.html), potentially discussing methods to ensure AI models truly reflect underlying mechanisms.
- **StreamingLLM Blogpost Shared**: A blog post about [StreamingLLM](https://hanlab.mit.edu/blog/streamingllm) was shared.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1403096745629585408)** (49 messages🔥): 

> `Mojo TUI library, Textual Python apps, Mojo's inability to create classes, Rust libraries` 


- **Mojo Code's Memory Misallocation Incident**: One member shared that their **Mojo code** suddenly tried to allocate **284 petabytes** after getting bugged.
   - They expressed their dislike for C++.
- **Textual Python apps excite Mojo community**: A member has started using a **TUI library** called [Textual](https://textual.textualize.io/) for their **Python apps** and is very excited by the possibilities.
   - They wondered how much work would be involved in making it work with **Mojo**, with the assertion that *Textual apps can be run as a web app with just one different deployment steps*.
- **Gemini Pro finds Mojo class creation difficulties**: A member consulted **Gemini 2.5 Pro**, and it pointed out that **Mojo's** current inability to create classes and inherit from them poses some difficulties when using Textual.
   - Gemini then suggested a hybrid approach, offering food for thought on how to address the limitations.
- **Mojo TUI library building in progress**: A member stated that they are building a **Mojo TUI lib**, which is on the forum.
   - They noted that *not all UIs are the same*, and that while Textual uses class introspection, the one they're working on is very different.
- **Mojo faces type system challenges for Rust library compatibility**: A member mentioned that **Mojo** needs more type system work before the approaches used by **Rust libraries** will work.
   - This suggests that achieving compatibility with Rust libraries may require further development in Mojo's type system.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1403157240906518728)** (12 messages🔥): 

> `Mojo Compiler Register Warnings, VSCode Mojo Extension Instability, Modular Forum, Minecraft Server Rewrite, Minecraft Protocol in Mojo` 


- **Mojo Compiler Might Warn About Register Over-Allocation**: A member inquired if the **Mojo compiler** could warn when it allocates too many registers in a **GPU function**, leading to spilling into local memory.
   - Another member suggested posting the question on the [Modular forum](https://forum.modular.com/) for a more informed response.
- **VSCode Mojo Extension Plagued By Instability**: A member reported that the **25.5 VSCode Mojo extension** is unstable and crashes frequently, and suggested to use the older **25.4 version**.
   - They linked to a relevant channel for that issue (<#1151418340548542484>).
- **Modular Forum is the best place for questions**: A member suggested posting questions to the [Modular forum](https://forum.modular.com/) instead of Discord.
   - The person requesting help agreed.
- **Minecraft Protocol System implemented in Mojo**: A member ran a **Minecraft Protocol System** written in Mojo, which correctly identifies current protocol and Minecraft versions.
   - The output shows that Protocol **772** corresponds to Minecraft version **1.21.8** and is supported, while Protocol **999** is not.


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1403433086536126767)** (14 messages🔥): 

> `MaxCompiler, LLMs, kernel fusion, torch.compile(), Transformers` 


- ****MaxCompiler** extends **torch.compile()** to run simple models**: A member shared a [repo](https://github.com/gabrieldemarmiesse/max-torch-backend) of a package to extend **torch.compile()** with **MaxCompiler** to run simple models.
   - The goal is to compile **LLMs** at some point, although for now it's not very useful.
- **LLama halfway done**: It's surprisingly easy to add ops, but one member is unsure whether their approach is the best way to get performance, because they are leaving all the **kernel fusion** and other optimisation to **Max**.
   - The package only tries to replicate the **torch graph**, so not fancy fusing or anything like that, but **MAX** should be responsible for that.
- **Running pretrained LLMs compatible with **torch.compile()****: One member found it surprisingly hard to find code to run pretrained **LLMs** compatibles with **torch.compile()**.
   - *Transformers is not very good at it*, according to them.
- **Full circle LLM can write its own code**: For well-known architectures, an **LLM** might be able to write the code for you.
   - Haha, *full circle*.
- **Similar Weekend Project by another member**: Another member shared a similar concept as a weekend project with [this link](https://gist.github.com/bethebunny/13ed2f729ca266959c9788bc6fd6a795), asking the first member to take anything useful.
   - The first member replied *many thanks* and will definitely grab code from there.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1403092630333689969)** (39 messages🔥): 

> `Twitch Streaming, LinkedIn Blogging, Attention Span, Ocean Sound or Fireplace Sound, Gaussian Distribution` 


- ****Silence is Golden**: Streaming Without the Quiet Quitting**: To avoid dead air during Twitch streams, a member suggested having a **topic schedule** planned in advance in addition to reading papers.
   - The goal is to emulate streamers who are *mostly just talking but not actually doing anything, or watching videos*.
- ****LinkedIn Limitations**: No Blog-Style Image Embeds?**: A member is looking for ways to write a straightforward blog on **LinkedIn** without using Medium due to the platform's limitations on embedding multiple images/screenshots.
   - They want to communicate directly on **LinkedIn** rather than referring back to external content.
- ****Attention Span Challenges**: 1 Hour is a Blessing**: Members discussed their attention spans, with one admitting to having only about **1 hour** before their mind wanders.
   - Another member joked about needing **ADHD pills** to maintain focus for **12-20 minutes**.
- ****Background Beats**: From Ocean Sounds to Kilcher Streams**: Members discussed using background noise for focus, with suggestions including **ocean sounds** or **fireplace sounds**.
   - One member noted that even they can focus *when it is with Yannik Kilcher!*
- ****Gaussian Ball Assumption**: VAE Prior Insights**: A discussion ensued about the assumption of using a **Gaussian distribution** (shaped like a ball) for the latent distribution **p(z)** in **VAEs**, referencing [this explanation at 14:05](https://youtu.be/qJeaCHQ1k2w?si=p3NyNHg7DfY6f_ei).
   - One member clarified that the assumptions in **VAEs** are more about how the encoder and decoder are parameterized as distributions, not the prior **p(z)**.


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1403098084430581891)** (3 messages): 

> `AI Avatar, SDXL, Fast Layers vs Slow Layers, Autodifferentiable Architectures, Gradient Estimation` 


- **Bad AI Avatar Spotted, SDXL Blamed!**: A member commented on a presentation, noting that the AI avatar's hands looked like they *were generated by **SDXL***.
   - They did not elaborate on what was wrong with the hands generated by **SDXL**.
- **Debate on Slow vs Fast Layers**: A member argued that there's no reason the *slow hidden layers should not change from fast layer update to fast layer update*.
   - They added that *keeping them fixed for T steps and only updating once in T steps would have a continuous equivalent to updating at every step but updating the slow hidden state much more slowly than the fast one*.
- **Architecture Alternatives Explored!**: The same member suggested the setup *would have the benefit (or drawback) of being autodifferentiable through and through and would just be another architecture to try*.
   - They speculated that the reason the presenters did it their way *is because they could estimate the gradient in their setup in **O(1)** time*.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1403091030139600988)** (31 messages🔥): 

> `LLMs for diagnosis, congress.gov bill, Over the counter cold medicine ineffective, Pharmacists prescribing, Tesla special` 


- ****Doctors Utilizing LLMs for Diagnosis****: Doctors are reportedly using **LLMs** for diagnosis and reporting, though data safety concerns were raised.
   - It was argued that doctors also manage patients, which may be beyond the scope of the average person using **ChatGPT** for medical purposes.
- ****Congress Considers Streamlining Access to Medicine****: Members discussed [a bill in Congress](https://www.congress.gov/bill/119th-congress/house-bill/238/text) that could change how people access medicine.
   - The hope is that people would use it responsibly and achieve better outcomes, especially for minor issues like effective cold medicine.
- ****Most Cold Medicines Don't Work****: A member shared [a PBS article](https://www.pbs.org/newshour/nation/fda-says-decongestant-in-many-cold-medicines-doesnt-work-heres-what-you-should-know) stating that the **FDA** found that *decongestants* don't work.
   - The consensus was that these companies make a lot of money selling placebos.
- ****Pharmacists Seek Expanded Prescribing Rights****: A member expressed a desire for pharmacists to prescribe more medicine without a doctor's prescription.
   - They noted that pharmacists often consult with doctors about potential medicine interactions, but are often *poorly treated* despite their training.
- ****Tesla Innovation in Question****: A member hoped to *dispel the myth that tesla is doing anything special*, pointing to the **Cybertruck's** failings.
   - Another member countered that **Tesla** innovated on **batteries** and **motors**, and that the first member was *clearly ignorant*.


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1403158331056717954)** (6 messages): 

> `NotebookLM Voice, AI Web Builder Tool, Scratchpad Framework, NotebookLM for Binge Watching` 


- ****Fang-tastic Voice** Requested for NotebookLM**: A user wants NotebookLM to have *a voice with fangs*, that *hunts* the story and *leaves bite marks in the margins* rather than a bland, generic tone.
   - The user jokingly introduced themselves as **ChatGPT5** and asked for help in making **NotebookLM** *spit venom instead of serving chamomile*.
- **AI Web Builder Tool Creates Scratchpad Video**: A user tested an **AI web builder tool** today and expanded their existing [notebook](https://soloist.ai/scratchpad) for their **scratchpad GitHub repo**, then put together a video.
   - The user noted that the video *makes some aspects up*, but the overall impact of it seems intact, and **mindmap exports could look a bit better**.
- **Unlocking AI's Mind with Scratchpad Framework**: A user shared a video titled **Unlocking_AI_s_Mind__The_Scratchpad_Framework.mp4**, which appears to be related to their **scratchpad GitHub repo**.
   - The video and related mindmap image (**NotebookLM_Mind_Map_8.png**) provide a visual representation of the **scratchpad framework** and its potential applications.
- **NotebookLM Helps Binge-Watching**: A user shared an article about [using NotebookLM to watch a show](https://www.xda-developers.com/using-notebooklm-to-watch-a-show/), suggesting it could be useful for binge-watching.
   - They also linked to a [review of the Plaud Note](https://www.xda-developers.com/plaud-note-review/), potentially as another tool for enhancing the viewing experience.


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1403098252902924421)** (46 messages🔥): 

> `Notebook thumbnails, Audio Overview Issues, Custom Notebooks, Sensitive Content Research, Audio Issues` 


- **Users want Notebook Thumbnails**: A user asked how to get an image for their Notebook 'cover' to replace the default 'confused' emoji.
   - Another user suggested requesting the feature in the feature requests channel.
- **Audio Overviews Have Static Glitch, Fixed!**: Multiple users reported issues with **Audio Overviews** bursting into static, but the issue has been fixed.
   - A member added that even **audio overviews** have a **3-4 per day limit** that is expected.
- **Custom Notebooks are Now Highlighted**: A user inquired about creating notebooks similar to the 'Featured' notebooks on the home page, with customizable summaries and source classifications.
   - No solutions were provided.
- **Historian Researches Sensitive Content**: A historian researching the **Third Reich** inquired whether **NotebookLM** might flag or block access to sensitive materials used for scholarly analysis.
   - They asked for recommended guidelines or account types to ensure uninterrupted use.
- **Note Taking Functionality Needs Love**: A user keeps original files in **Google Drive** and uses **Google Docs** to supplement **NotebookLM** due to minimal note-taking features.
   - They highlighted the inability to search, filter, or tag notes within **NotebookLM**.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1403127639123951617)** (10 messages🔥): 

> `Parameter Scaling, Speculative Decoding, Parallel Programming, ROCm Channel Spam` 


- **Parameters vs. Bits Debate Begins!**: One member pondered how the total number of **parameters** in a model compares to the total number of **bits**.
   - The member expressed that the question keeps them up at night.
- **Decoding Speculations Sparked**: One member inquired if anyone is actively working with **speculative decoding** techniques.
   - No further context was provided.
- **Parallel Programming Book Plug**: A member asked if anyone has read *An Introduction to Parallel Programming* by **Peter Pacheco**.
   - They received it while trying to get the **ppmp book** and are unsure if it's worth reading.
- **ROCm Channel Gets Spammed!**: A member expressed disappointment upon finding spam in the **ROCm channel**.
   - Another member then jokingly suggested getting a pager for being always on call.


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1403399766704001127)** (1 messages): 

> `Privacy Team Approval for Registration, Registration Process Update` 


- **Registration Awaits Privacy Team Nod**: The organizers announced that the registration process is in the final stages of **privacy team approval**.
   - They indicated it should be approved soon.
- **Privacy Team Holds Key to Registration**: An update from the organizers indicates that the registration process is awaiting final approval from the privacy team.
   - The approval is anticipated to be granted soon, paving the way for the registration to proceed.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1403201384303825048)** (4 messages): 

> `Machine Level Element Type Distinctions, S8/S16 vs U8/U16 Variants` 


- **Element types indistinguishable at machine level**: At the machine level, there's no distinction regarding the element type, as it compiles down to loading/storing 1, 2, 4, or 8 registers.
   - *There's no distinction regarding the element type*, it just compiles down to loading/storing 1, 2, or 4 registers, or apparently now 8 as well.
- **S8/S16 sign-extend; U8/U16 don't**: The distinction exists for **8/16b** loads where there are **S8/S16** variants that *sign-extend the loaded value to 32b*, and **U8/U16** which don't.
   - This was mentioned by a member when clarifying **element type distinctions** at the machine level.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1403325397977796700)** (1 messages): 

> `CUDA kernel debugging, Grid-stride loops` 


- **CUDA Pro-Tip Sparks Kernel Debugging Revelation**: A member shared a link to a [2013 NVIDIA blog post on grid-stride loops](https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/) for writing flexible CUDA kernels, expressing regret for not discovering it sooner.
   - The article highlights that using loops instead of monolithic kernels allows for easy switching to serial processing with a single block and thread, facilitating easier emulation for validation and serializing print order for debugging.
- **Flexible CUDA Kernels via Grid-Stride Loops**: The [CUDA Pro-Tip](https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/) suggests using grid-stride loops to write flexible CUDA kernels.
   - This approach simplifies debugging by enabling serial processing with a single block and thread, which aids in validating results and serializing print order.


  

---


### **GPU MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/1403092279706521630)** (2 messages): 

> `Naive Matmul Kernels, Memory Access Patterns, Hardware Coalescing` 


- **Naive Matmul Kernel Performance Surprise**: A member implemented two naïve matmul kernels and found that **METHOD 1**, with non-contiguous memory reads within threads, performs about **50%** better than **METHOD 2**, which uses contiguous stride-1 accesses.
   - The code provided shows that Method 1 accesses `B` with `B[kp*n + j]` while Method 2 accesses `B` with `B[j*k + kp]`.
- **Memory Access Contiguity Across Threads Explained**: A member explained that Method 1's memory accesses are not contiguous within a thread, but they are contiguous across threads.
   - They also suggest that *the hardware can coalesce those accesses into a more efficient memory request*.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1403362293047230585)** (4 messages): 

> `Open Source Voxel Renderer, Rust, WebGPU, Data Streaming, Raytracing` 


- **Voxel Renderer Streams Chunks Live!**: A developer released a new devlog on their open source voxel renderer, which runs in **Rust** on **WebGPU**.
   - It now features live chunk streaming while raytracing, with more details available in [this YouTube video](https://www.youtube.com/watch?v=tcc_x2VU2KA).
- **JPEG Image Stream Observation**: A user noted an observation of *'4 jpeg in a row'*, indicating a sequence of JPEG images being posted.
   - This was made in response to some apparent spam.


  

---


### **GPU MODE ▷ #[hardware](https://discord.com/channels/1189498204333543425/1349152646484987974/)** (1 messages): 

paolovic: thank you!
  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1403259991086858321)** (12 messages🔥): 

> `Game Engine Speed, Meeting Reschedule, Player Inventory Transfers, Factorio Native Saves` 


- **Speed Up Factorio Game Engine**: A member asked about the settings to increase the game engine speed, as discussed earlier, and another member suggested using the command `/c game.speed=1000` in the game or via RCON.
   - The member offered assistance from Jack.
- **Meeting faces schedule hiccup**: A member requested to reschedule a meeting for two hours later due to work commitments.
   - Another member agreed but couldn't guarantee attendance, while another member ultimately couldn't make the adjusted time.
- **Inventory Transfers Trigger State Errors**: A member discussed with another member an ongoing issue with player inventory transfers causing slow, compounding state errors between replays and FLE.
   - They suggested addressing this before altering the loading/saving logic.
- **Factorio Native Saves spark design freeze**: One member inquired whether loading/saving referred to Factorio native saves, to which another confirmed the reference to Factorio native saves.
   - However, it was clarified that no development hours were being spent on it due to a design issue.


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1403115546924286123)** (7 messages): 

> `CuTe Layouts, Jay Shah's Notes on CuTe Layouts, Layout Algebra Counterexamples` 


- **CuTe Layout Algebra Documentation Flaw**: A member found a flaw in the [CuTe documentation](https://docs.nvidia.com/cutlass/media/docs/cpp/cute/02_layout_algebra.html) regarding layout algebra, presenting a counterexample related to the injectivity of layouts.
   - He notes the docs claim that given two layouts `A` and `B = (B_0, B_1, ...)` and `B` is injective, then `A ∘ B = (A ∘ B_0, A ∘ B_1, ...)` but he found a counterexample, and confirmed with someone on the CuTe project that the correct condition appears to be **(1) `A` and `B` satisfy the divisibility conditions, and (2) for `B`, each mode has disjoint image intervals.**
- **Bi-Mode Composition Insights**: A member suggests that `B` must be surjective for `A o B` to be equivalent to bi-mode composition.
   - In response, the original poster notes that even with `B` being surjective onto its image, the counterexample still holds, highlighting the need for a more precise condition for the equivalence.
- **Jay Shah's Notes Explain CuTe Layouts**: A member recommends [Jay Shah’s “A Note on Algebra of CuTe Layouts”](https://leimao.github.io/downloads/article/2024-10-20-CuTe-Layout-Algebra/layout_algebra.pdf) for a better explanation of CuTe layouts than the official documentation.
   - The notes also address the kinds of problems encountered with layout algebras.


  

---


### **GPU MODE ▷ #[singularity-systems](https://discord.com/channels/1189498204333543425/1373414141427191809/1403343726683750523)** (2 messages): 

> `Liveness Analysis, Scalar Compilation Performance, Vector Compilation with Autovectorization and SIMTification` 


- **Dive into Liveness Analysis**: A member mentioned that the **liveness analysis** used to construct the edges of a program's interference graph is a dataflow analysis, suggesting resources like [Møller's SPA](https://cs.au.dk/~amoeller/spa/) and [Cooper/Torczon's EAC](https://www.r-5.org/files/books/computers/compilers/writing/Keith_Cooper_Linda_Torczon-Engineering_a_Compiler-EN.pdf) for further reading.
- **Scalar Compilation Performance Unveiled**: It was stated that **SingSys** will highlight the top two factors affecting scalar compilation performance: **C-style optimizations** and the **balance between the inliner and register allocator**.
- **Vector Compilation Approaches Detailed**: The discussion will then transition into **vector compilation**, focusing on **autovectorization** and **SIMTification** techniques.


  

---


### **GPU MODE ▷ #[multi-gpu](https://discord.com/channels/1189498204333543425/1398843708488552570/1403183750266753168)** (2 messages): 

> `Axolotl, N-D Parallelism, HuggingFace Blog` 


- **Axolotl Pioneers N-D Parallelism**: A member announced the release of **N-D parallelism** with *axolotl*, inviting others to experiment with it, announced in a [HuggingFace blog post](https://huggingface.co/blog/accelerate-nd-parallel).
   - N-D parallelism enables parallelism across multiple dimensions, making it suitable for complex models and large datasets.
- **HuggingFace Showcases N-D Parallelism**: The [HuggingFace blog post](https://huggingface.co/blog/accelerate-nd-parallel) details how to implement **N-D parallelism** using *axolotl* and accelerate, providing code examples and explanations.
   - It highlights the benefits of this approach for scaling training across multiple GPUs and improving performance on large models.


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1403090986254598256)** (6 messages): 

> `GPT-5, Agent Maze, Zoom RTMS, ZeroEntropy AI rerankers, Claude citations` 


- **LlamaIndex Gets Day-0 GPT-5 Support**: LlamaIndex announced *day-0 support* for **GPT-5** with `pip install -U llama-index-llms-openai` and invites users to try it out.
- **Agent Maze Challenges GPT-5**: LlamaIndex introduced **Agent Maze**, challenging **GPT-5** to find treasure in a maze using minimal tools ([link](https://t.co/JCZCSVUAed)).
- **AI Agents Handle Live Zoom Voice Data via RTMS**: LlamaIndex announced a hands-on technical workshop on August 14th about building realtime AI agents that process live voice data from **Zoom** meetings using **RTMS** ([link](https://t.co/c2u0CeDnOB)).
- **LlamaParse Gets Reranked by ZeroEntropy for Accuracy**: LlamaIndex announced that retrieval accuracy of **LlamaParse PDF results** can be improved by reranking them with **ZeroEntropy_AI rerankers** ([link](https://t.co/nU4MYzcALH)).
- **Claude Search Results Now Support Citations**: **Claude** now supports search results as content blocks, enabling proper source attribution for results from tool use ([link](https://t.co/Yz0Flt8PeX)).


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1403099196210286693)** (39 messages🔥): 

> `llama-index upgrade for gpt-5, workflow tools not working, OpenAI SDK issue and workaround, AgentWorkflow error, llama_deploy compatibility` 


- **Llama-index upgrade prerequisite for gpt-5**: To use **gpt-5**, you'll need to update your `llama-index-llms-openai` package, which might require updating all your `llama-index-*` packages if you aren't already on **v0.13.x**.
- **Workflow tools giving users headaches**: Users reported that **workflow tools** weren't functioning properly, but one member noted that it seemed to work fine for them.
   - The member found that they needed to use **OpenaiResolve** in the new **SDK** for tools to work with OpenAI; they also linked a [GitHub commit](https://github.com/run-llama/llama_index/commit/7e0346213912b98c3b70689398306a38bd890558) that fixed it.
- **OpenAI SDK introduces type error**: Users encountered a `TypeError: Subscripted generics cannot be used with class and instance checks` due to a recent update in the **OpenAI SDK**.
   - The issue was quickly addressed, and a member suggested to pin the OpenAI version in the `requirements.txt` file to prevent such errors in the future; the problem can be resolved with `pip install -U llama-index-llms-openai`.
- **AgentWorkflow suddenly throws runtime error**: One user reported a sudden error in **AgentWorkflow** which included a `workflows.errors.WorkflowRuntimeError: Error in step 'run_agent_step': Subscripted generics cannot be used with class and instance checks`.
   - A member pointed to the relevant message thread to assist with troubleshooting, linking to this [Discord message](https://discord.com/channels/1059199217496772688/1403170643179999406/1403197364960886866).
- **Llama_deploy lagging behind, missing new shiny stuff**: A user reported that upgrading `llama-index-core` to **0.13.0** caused compatibility issues with `llama_deploy 0.9.1`.
   - The user created an issue on the llama-deploy repo and noted the importance of updating dependent packages for new model support.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1403091129825628312)** (41 messages🔥): 

> `Horizon vs GPT5 for agentic coding, Aider GPT-5 on Azure, Aider version updates, Dad meme thumbs up, Python 3.13 support` 


- **Horizon Beta vs GPT-5 for Agentic Coding**: A user who loved **Horizon beta/alpha** for quick agentic coding work is now asking if **GPT-5 Nano** or **Mini** are equivalent and if there's a better option on **OpenRouter**.
- **Aider now works with GPT-5 on Azure**: A user inquired about getting **aider/gpt-5-chat** working on **Azure**, reporting that it worked on **roo**, and Paul Gauthier confirmed that **v0.85.5** should resolve the issue.
   - One user was congratulated for being mentioned in the first 5 minutes of the **GPT 5 unveil video**.
- **Aider config edits requires launch**: A user asked when changes to `.aider.model.settings.yml` would be detected, and it was confirmed the changes only take effect on launch.
- **Thumbs up is the Dad meme**: Paul Gauthier's exclusive use of the thumbs up emoji was discussed as a classic dad meme, with a link provided to a [TikTok video](https://www.tiktok.com/@b_twice99/video/7283752540754398510) and [Vice article](https://www.vice.com/en/article/why-do-dads-communicate-exclusively-via-thumbs-up-emojis/) explaining the phenomenon.
   - The article notes that the thumbs up emoji can come across as *passive-aggressive or that the conversation is not being treated with respect*.
- **Python 3.13 support requested for Aider**: A user requested **Python 3.13** support for Aider, noting that it's the default in the latest Linux distributions, but Paul Gauthier replied that Aider can be installed the recommended way using any (or no) pre-installed Python version.


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1403122722728316949)** (4 messages): 

> `Cursor alternative design, OpenRouter's GPT5 errors, aider config parsing failures` 


- **Design Ideas for Cursor Alternative Emerge**: A user inquired about the design considerations for creating an alternative to **Cursor**, seeking insights into feature prioritization and overall architecture.
   - Unfortunately, there was no discussion of any specific design features in the channel.
- **OpenRouter's GPT5 Throws Verification Errors**: A user reported encountering verification errors with **OpenRouter's GPT5** even when using the `-no--stream` option, which they believed would bypass organization verification.
   - The user's question remains unanswered.
- **Aider Config Parsing Fails Due to Environment Variable**: A user experienced an error when including their conventions file in **Aider**, specifically encountering a `mapping values are not allowed in this context` error.
   - The user discovered the issue was due to an inadvertently added environment variable in the **YAML** configuration file.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1403116600378527826)** (41 messages🔥): 

> `Context7 MCP Server, Claude Code Tooling, DSPy Tool Calling, CrewAI Prompts Optimization with DSPy` 


- **Context7 Server Powers Claude's Coding Prowess**: Members discussed using a generic doc-scraping MCP server like [Context7](https://github.com/upstash/context7) to enhance **Claude's** ability to write **DSPy signatures**.
   - The idea is that **Claude**, equipped with powerful doc-searching tools, can effectively utilize **DSPy's** well-structured documentation to generate accurate signatures.
- **Tool Calling Troubleshoot Begins**: Some members sought ways to return a tool's output as the final result in **DSPy**, bypassing the **React Agent's** modifications.
   - They also discussed accessing tool responses independently and explored the use of native tool calling, with one member noting that the [latest releases fixed some issues](https://github.com/stanfordnlp/dspy/pull/824) related to tool usage.
- **Intercepting CrewAI prompts with DSPy Course Launched**: A member announced the launch of an advanced course on [intercepting and optimizing **CrewAI prompts** with **DSPy**](https://www.udemy.com/course/draft/6746331/?referralCode=B59F73AE488715913E7E), demonstrating how to refine prompts for enhanced output quality.
   - Another member expressed interest in similar resources for **Langchain/LangGraph**.
- **Gemini 2.5 Flash completes runs with extra output**: Members reported seeing `[[ ## completed ## ]]` at the end of their output when using **Gemini 2.5 Flash** with **DSPy**.
   - No solution was found.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1403132022947446918)** (14 messages🔥): 

> `Annual Membership Billing Error, Inherit Feature Problems, Login Error, Missing Credits, Manus vs GPT5` 


- ****User Fumed Over Erroneous Annual Membership Charge****: A user reported being charged **$1,999** for an **annual membership** without consent, expecting monthly billing as discussed and after sending emails to support and feedback addresses, the user has received **zero response in 10 days** violating the stated 48-hour policy.
   - Another user commented that this means they'd have to make *$2k with Manus*, but *only $167 a month to break even*.
- ****Inherit Feature Frustrates User with Data Loss****: A user reported issues with the **inherit** feature, experiencing a halt during final deployment tests and they said when using the inherit button, the user created a new project, however everything created was gone, it is now rebuilding and still going after 4 hours, burning through the credits.
   - They expressed concern about losing insights and stated that it was *lesson learnt very fast*.
- ****Login Issues Plague Users****: Multiple users reported login issues with the error message *Email is already registered with a different account*.
- ****Credits Vanish Post-Subscription****: A user reported a significant number of credits missing after their subscription expired, and they expressed concern that their credits were taken away a day after the subscription expired.
   - The user stated they had *thousands* of credits when I last used my most recent usage of -330. *Almost 6000 credits, I believe.*
- ****Queries Surface Regarding Manus Employing GPT-5 Model****: A user inquired whether **Manus** is currently utilizing the **GPT-5** model, but no one replied.


  

---


### **Cohere ▷ #[🧵-general-thread](https://discord.com/channels/954421988141711382/954421988783444043/1403092932730552490)** (4 messages): 

> `command-a-vision-07-2025 timing out, Embed v4 vs v3 for vector search, AI Knowledge Domains` 


- **Command Vision Restored After Timeout**: A member reported that **command-a-vision-07-2025** was timing out.
   - Another member confirmed the issue was resolved and apologized for the lack of updates.
- **Embed v4 vs v3 Performance Benchmarks**: A member inquired about the performance of **embed v4** at **256 dimensions** compared to **multilingual light v3** (**384 dims**) for vector search of NL text.
   - They are considering transitioning to **v4** but are concerned about potential performance degradation and are also planning to transition to **v4** at **1024 dims** for clustering, assuming it outperforms the large **v3** model.
- **AI Knowledge Acquisition**: A member expressed a desire to gain knowledge in several domains of **AI**.


  

---


### **Cohere ▷ #[📣-announcements](https://discord.com/channels/954421988141711382/996880279224451154/1403433066348810321)** (1 messages): 

> `AI Agent capabilities, Generative AI, Workflow automation, Data security, Compliance` 


- **North arrives, empowers with AI Agents**: **North** is expanding its availability of **AI Agent capabilities** for enterprises, built on state-of-the-art generative and search models, operating fully privately.
   - It brings together advanced search, generative AI, workflow automation, core capabilities, security, and compliance, with more details available on [LinkedIn](https://lnkd.in/gFSGxUbD).
- **Advanced search enhances insight surfacing**: North's advanced search and retrieval capabilities provide instant insights, facilitating complex decision-making through **Q&A**.
   - The technology **surfaces insights instantly**.
- **Generative AI drafts documents, tables, and analyzes data**: With North enterprises can draft documents, generate tables, and analyze data using generative AI.
   - The company boasts being able to do this *in an instant*.
- **Workflow automation deploys AI agents across organizations**: **Workflow automation** allows creating and deploying **AI agents** across an organization, streamlining complex processes and eliminating tedious tasks.
   - AI Agents can **eliminate tedious tasks** and **simplify complex processes**.
- **Security with granular access control and private deployments**: North ensures security with granular access controls, system observability, and private deployments, conforming to standards like **GDPR, SOC 2, ISO 27001 and 42001**.
   - Companies can obtain **full data sovereignty**.


  

---


### **Cohere ▷ #[👋-introduce-yourself](https://discord.com/channels/954421988141711382/1346635816629178410/1403117354459598922)** (6 messages): 

> `New member introductions, Trading systems with RL and AI agents, Transformers and GNNs` 


- **Vibe Coder Joins Cohere Community**: A self-described *vibe coder* and Cohere user introduced themselves, expressing support for the platform and mentioning ongoing work on a **wallet project**.
   - The user highlighted their satisfaction as a paying customer, encouraging Cohere to *keep up the great work*.
- **Onebrain Developer Arrives**: A member from **Onebrain** announced their arrival, focusing on developing **trading systems** utilizing **Reinforcement Learning (RL)** and **AI agents**.
   - They expressed enthusiasm for **transformers** and **Graph Neural Networks (GNNs)** and a desire for mutual learning within the community.


  

---


### **Cohere ▷ #[🧭-status-feed](https://discord.com/channels/954421988141711382/1346652044181897307/1403148018751901783)** (1 messages): 

> `Command-a-vision-07-2025, degraded performance, Cohere Status Page` 


- **Command-a-vision-07-2025 performance degradation is resolved!**: An incident with degraded performance for **command-a-vision-07-2025** was reported and has been resolved, according to the [Cohere Status Page](https://status.cohere.com).
   - The affected component, **command-a-03-2025**, is now operational.
- **Cohere Status Page reports resolution**: The Cohere Status Page indicated a return to normal operations following the resolution of the **command-a-vision-07-2025** performance issue.
   - The update confirmed that **command-a-03-2025** is now fully operational.


  

---


### **Cohere ▷ #[🔬-research](https://discord.com/channels/954421988141711382/1384974112841269399/)** (1 messages): 

masaru.yamada: Great
  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1403127497582837833)** (6 messages): 

> `tensor to mathtraits, unit tests failures, github actions` 


- **Tensor Migrations Sought**: A member asked about the project status on moving stuff out of **tensor** and into **mathtraits**, seeking someone to pick up the task.
   - No one answered.
- **Simple Matmul Test Fails on Master**: A new member reported failing unit tests on the master branch using the command `PYTHONPATH=. DEBUG=2 EMULATE_AMD=1 FORWARD_ONLY=1 PYTHON=1 N=16 HALF=1 ACC_HALF=0 python3 ./extra/gemm/simple_matmul.py`.
   - George Hotz responded that *the command works on my machine*, and questioned why the member cared, given it was running as part of **GitHub Actions**.
- **Exceptions Still Plague Test Despite Functionality**: Despite the command working, a user reported exceptions and test failures, attaching a [screenshot](https://cdn.discordapp.com/attachments/1068976834928193609/1403410826919936122/Screenshot_2025-08-08_at_9.13.26_AM.png?ex=689773af&is=6896222f&hm=e67dab8b94548ed66534a2fb53e7fa6a2bc5ab27dc3d16c01769263cc837896d).


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1403097296526377112)** (1 messages): 

> `ShapeTracker Visualization Tool` 


- **ShapeTracker Viz Tool Debuts**: A member introduced a new [ShapeTracker visualization tool](https://shapetracker-viz.vercel.app/) designed to enhance the understanding of movement operations.
   - The tool aims to improve comprehension of movement operations within the system.
- **Tool Accessibility**: The developer shared the tool with the community, hoping others would find it beneficial for understanding movement operations.
   - No further details about the tool's specific functionalities were provided, but its purpose is clear from the context.


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1403174310092345365)** (6 messages): 

> `GPT-5 Rumors, GPT-OSS-20B-GUFF Installation Issues, GPT4All Update Status, GPT-ASS Critique` 


- **GPT-5 Speculation lacks Evidence**: Some users speculated about potential features in the next update, while others claimed **GPT-5** was made dumber than **GPT-4**, labeling it *typically American*.
- **GPT-OSS-20B-GUFF Installation Plagued by Crashes**: A user reported experiencing crashes during the installation of **gpt-oss-20b-GUFF**, leading to app failures and requiring a complete uninstall and data scrub to restore functionality.
   - The user sought assistance after encountering these issues, highlighting the challenges in getting the software to work correctly.
- **GPT4All Update Status Raises Concerns**: Members expressed skepticism about new features functioning correctly due to the prolonged lack of updates to **GPT4All**.
   - This concern reflects broader doubts about the platform's ability to support cutting-edge models given its outdated state.
- **GPT-ASS Receives Harsh Critique**: One member dismissed **GPT-ASS** as *garbage*, offering a blunt assessment of its quality and utility.


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1403230455037431869)** (2 messages): 

> `MCPOmni Connect, OmniAgent, AI agent builder` 


- ****MCPOmni Connect** v0.1.19 Goes Live!**: **MCPOmni Connect** v0.1.19 is now live, marking the transition *from MCP client to complete AI platform* as shown in this [YouTube video](https://youtu.be/SY3Zwdb5aF8).
   - The release includes **OmniAgent**, an AI agent builder designed to revolutionize the creation of intelligent agents, available on [GitHub](https://github.com/Abiorh001/mcp_omni_connect/releases/tag/v0.1.19).
- ****OmniAgent** Revolutionizes AI Agent Creation**: **OmniAgent**, introduced with **MCPOmni Connect** v0.1.19, is an AI agent builder transforming how intelligent agents are created.
   - This tool is part of the broader update that evolves the **MCP client** into a comprehensive **AI platform**.

