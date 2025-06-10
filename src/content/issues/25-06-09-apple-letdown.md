---
id: MjAyNS0w
title: Apple exposes Foundation Models API and... no new Siri
date: '2025-06-10T05:44:39.731046Z'
description: >-
  **Apple** released on-device foundation models for iOS developers, though
  their recent "Illusion of Reasoning" paper faced significant backlash for
  flawed methodology regarding LLM reasoning. **OpenAI** updated **ChatGPT's
  Advanced Voice Mode** with more natural voice and improved translation,
  demonstrated by Greg Brockman. **LangChain** and **LlamaIndex** launched new
  AI agents and tools, including a SWE Agent for software automation and an
  Excel agent using reinforcement learning for data transformation. The AI
  community engaged in heated debate over reasoning capabilities of LLMs,
  highlighting challenges in evaluation methods.
companies:
  - apple
  - openai
  - langchain
  - llamaindex
models:
  - chatgpt
topics:
  - on-device-ai
  - foundation-models
  - reasoning
  - reinforcement-learning
  - voice
  - translation
  - software-automation
  - agentic-workflows
people:
  - gdb
  - scaling01
  - giffmana
  - kevinweil
---


**We don't know what to say, Tim.**

> AI News for 6/6/2025-6/9/2025. We checked 9 subreddits, 449 Twitters and 29 Discords (218 channels, and 12496 messages) for you. Estimated reading time saved (at 200wpm): 1124 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

![](https://resend-attachments.s3.amazonaws.com/wibFpp0L0FwG9q7)

A year after the [excitement of Apple Intelligence last WWDC](https://news.smol.ai/issues/24-06-10-ainews-talaria-apples-new-mlops-superweapon), with Qwen and Gemma 3 likely so far ahead of Apple Foundation Models that there are nothing to crow about, the one AI relevant update is that the on device models are at least now available for iOS developers to use in the standard modalities ([documentation here](https://developer.apple.com/documentation/foundationmodels)):

![](https://resend-attachments.s3.amazonaws.com/DBxaoxE55xzq0sR)

The Siri delays were [well telegraphed months ago](https://www.usatoday.com/story/tech/2025/06/04/apple-wwdc-2025-rumors/84017268007/), so nothing more to write home about, but this is sitll big news for Apple AI engineers.

---

# AI Twitter Recap

### **Apple's "Illusion of Reasoning" Paper & Backlash**

- **Widespread criticism of methodology**: **Apple's** new paper, which some have dubbed ["The Illusion of Reasoning,"](https://twitter.com/andersonbcdefg/status/1931821352463577482) has faced significant backlash from the AI community for its methodology. A detailed critique by [@scaling01](https://twitter.com/scaling01/status/1931854370716426246) argues the paper mistakenly uses **optimal path length** as a proxy for problem complexity, when in reality, games like **Tower of Hanoi** have a simple, single-rule solution path despite their exponential solution length. The analysis suggests the models' performance degradation is not due to a lack of reasoning, but because they are [trained to be concise and stop generating long outputs](https://twitter.com/scaling01/status/1931817022926839909), a point he demonstrated by spending [$20 on API calls to poke holes in the paper](https://twitter.com/scaling01/status/1931818332321149416). Many agreed with this rebuttal, with one user noting, "[jokes aside i think Apple has really hurt their reputation with The Paper](https://twitter.com/teortaxesTex/status/1931842186158756135)."
- **Community reaction and debate**: The paper sparked a broader conversation about evaluating LLM reasoning. While some defended the paper, [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1931877956257005904) called a common counterargument an "extremely midwit take." Others pointed out the irony of the situation, with [@gallabytes](https://twitter.com/gallabytes/status/1932125532327743868) asking, "if I asked you to solve towers of Hanoi entirely in your head without writing anything down how tall could the tower get before you'd tell me to fuck off?" and [@vikhyatk](https://twitter.com/vikhyatk/status/1931842055883645044) sarcastically noting, "i asked an LLM to calculate the 8th busy beaver number and it failed. this proves that LLMs can't reason." The general sentiment was that the paper's conclusions were overstated, a view supported by a user whose [rebuttal of the paper's experiment gained significant traction](https://twitter.com/giffmana/status/1931801836052189191).

### **New AI Models, Tools & Features**

- **ChatGPT's major voice and translation update**: **OpenAI** rolled out a significant update to **ChatGPT's Advanced Voice Mode** for paid users, making it "[much more natural and easy to talk to](https://twitter.com/kevinweil/status/1931476402446156084)." The new voice model was demonstrated by **OpenAI's Greg Brockman** ([@gdb](https://twitter.com/gdb)) in a [viral tweet](https://twitter.com/gdb/status/1931456650336141752). The update also improved language translation capabilities, with users noting that advancements are so rapid that most people "[don’t realize how far we advanced from Siri in voice interfaces](https://twitter.com/BorisMPower/status/1931732885415010763)."
- **LangChain and LlamaIndex release new agents and tools**: **LangChain** announced several new tools, including a **SWE Agent** to [automate software development](https://twitter.com/LangChainAI/status/1931743095021789361), a tutorial for building [local AI agents with Ollama](https://twitter.com/LangChainAI/status/1931758230314623435), and a **Gemini Research Assistant** that performs [web research with reflective reasoning](https://twitter.com/LangChainAI/status/1931410870451442063). **LlamaIndex** detailed the architecture for their new **Excel agent**, which uses [RL-based structure understanding to perform complex data transformations](https://twitter.com/jerryjliu0/status/1931383524902453336), and released a tutorial on building an [agentic workflow to extract structured data from Fidelity annual reports](https://twitter.com/jerryjliu0/status/1931810929425158272).
- **Perplexity and Google enhance research tools**: **Perplexity** is testing an [updated version of its Deep Research feature](https://twitter.com/AravSrinivas/status/1931774041431712006) and asked for user feedback on its **EDGAR integration** for financial analysis. Meanwhile, the **UK government** is using **Google's Gemini** in a new system called **Extract** to [turn complex planning documents into digital data in just 40 seconds](https://twitter.com/GoogleDeepMind/status/1932032485254217799), a development praised by **DeepMind CEO Demis Hassabis**.
- **New open models and datasets announced**: **Sakana AI** released **EDINET-Bench**, a [Japanese financial benchmark](https://twitter.com/SakanaAILabs/status/1931887596323717406) built from regulatory filings to test advanced financial tasks. **Yandex** also released **Yambda-5B**, a [large-scale, anonymized dataset of music streaming interactions](https://twitter.com/TheTuringPost/status/1932091557127274993) to aid recommender system research. **Hugging Face** is organizing the ["biggest robotics hackathon ever"](https://twitter.com/ClementDelangue/status/1932079865001623747) with **LeRobot**, happening simultaneously in 100 cities.

### **AI Industry & Platform Dynamics**

- **The high cost of changing AI tools**: A widely shared sentiment is that the rapid pace of improvement makes it difficult to commit to annual subscriptions for AI tools. As one user stated, "[the best tool changes so rapidly that I will likely cancel and switch to another tool in less than a year's time](https://twitter.com/iScienceLuvr/status/1931531199521919221)." This reflects the competitive landscape where companies are constantly releasing new models and features.
- **Debate on AI consciousness and safety**: A speech by **Ilya Sutskever** at the University of Toronto reignited the debate about AI's potential, where he stated, "[The day will come when AI will do all the things we can do](https://twitter.com/Yuchenj_UW/status/1931883302623084719)." This sparked discussion around whether it's meaningful to debate if AI can "truly think." On the safety front, a detailed post raised concerns that models like **o3** and **Gemini 2.5 Pro** are [plausibly capable of assisting in bioweapon creation](https://twitter.com/RyanPGreenblatt/status/1931834526231339194) and that AI companies should be more transparent about their safety evaluations.
- **Apple's WWDC 2025 disappoints some developers**: Apple's annual developer conference received mixed reviews. Some felt the announcements lacked the "magic or delight" of past events, with one developer reminiscing about the **iPod mini** and concluding, "[Maybe if we want cool shit we just have to build it ourselves](https://www.google.com/search?q=https://twitter.com/raizamrtn/status/1932172447857659957)." The new iOS UI was compared to "[Windows Vista](https://twitter.com/skirano/status/1932145646963704199)" and mocked for its use of gradients.
- **India's potential as an AI superpower**: **Hugging Face CEO Clément Delangue** expressed his belief that "[India could become an AI superpower](https://twitter.com/ClementDelangue/status/1931846782184497224)," a sentiment that resonated widely and sparked discussion about the country's growing role in the global AI landscape.

### **Technical Concepts & Research**

- **The power of SaaS and abstraction**: In a highly-retweeted observation, one user noted that "[SaaS is good because the only way to maintain an abstraction boundary in software is to put it in another company](https://twitter.com/EigenGender/status/1931489268490457183)." This sparked a discussion about software architecture and the value of modularity in building complex systems.
- **Meta-Learning approaches**: **The Turing Post** published an explainer on **meta-learning**, detailing three common approaches: [optimization-based, metric-based, and model-based](https://twitter.com/TheTuringPost/status/1931446897904058517). This type of learning enables models to quickly adapt to new tasks from a small number of examples.
- **RL for medical applications**: A significant opportunity exists in applying **Reinforcement Learning (RL) to medicine**, but it's considered underexplored due to the difficulty of "[turning medicine into verifiable problems](https://twitter.com/iScienceLuvr/status/1931694421239902474)." This highlights a key challenge in moving from general AI to specialized, high-stakes domains.
- **Building a foundation model for fraud detection**: A viral post from a **Stripe** engineer on their successful **fraud detection foundation model** was analyzed. The analysis pointed out that this was a rare "instant win" because fraud detection is not a true prediction problem, Stripe already had a signal-rich environment, and the task was already automated, making it a [drop-in replacement for older ML systems](https://twitter.com/random_walker/status/1932046940822212827).
- **Merging Transformers**: A user shared a technical insight on how to [merge two Transformer models of width](https://www.google.com/search?q=%5Bhttps://twitter.com/cloneofsimo/status/1931566076116324392%5D(https://twitter.com/cloneofsimo/status/1931566076116324392)) `h` [into a single Transformer of width](https://www.google.com/search?q=%5Bhttps://twitter.com/cloneofsimo/status/1931566076116324392%5D(https://twitter.com/cloneofsimo/status/1931566076116324392)) `2h` by concatenating weights and using block matrices for certain projections, sparking a technical discussion on model composition.

### **Robotics & General AI Progress**

- **Humanoid robots are "within reach"**: **Brett Adcock** of **Figure AI** stated that "[it really feels like general robotics is within reach](https://twitter.com/adcock_brett/status/1931509884484567323)" and that there is potential to ship "millions of robots." He later shared a video of a [humanoid robot flipping a box](https://twitter.com/adcock_brett/status/1931850724343964116) and a [deep-dive on the AI work powering the latest Helix release](https://twitter.com/adcock_brett/status/1932192198025773371), emphasizing that nearly half of GDP is human labor that could eventually be automated.
- **The pace of progress and the "AI Engineer World's Fair"**: The **@aiDotEngineer** conference was a major event, with best speaker awards announced by [@swyx](https://twitter.com/swyx/status/1931552069984608486). The rapid pace of development was a key theme, with one attendee noting that we're streamlining "[1-2 really obvious things per year](https://twitter.com/lateinteraction/status/1931392417712021994)" — 2023 was mainstream chatbots, 2024 is RAG, and 2025 is basic RL. Another observed that companies often don't realize that a tool like **Runway** [can already perform tasks they're asking about for the future](https://twitter.com/c_valenzuelab/status/1932203777462849916).

### **Humor & Memes**

- **On the human-like qualities of AI**: It's becoming a common practice to [add subtle grammar errors to writing to give it a "clear human touch"](https://twitter.com/_jasonwei/status/1931467704495649165) and avoid the appearance of being AI-generated. On the other hand, there's concern about "[Sycophantic AI](https://twitter.com/scaling01/status/1931373162479997268)" that simply amplifies existing beliefs.
- **The state of AI development**: A popular meme perfectly captured the current mood in AI: [a panoramic image of a chaotic, futuristic city labeled "State of AI"](https://twitter.com/c_valenzuelab/status/1931531136070517035). Another relatable struggle for developers is the temptation to [use coding agents to "completely refactor other people's code"](https://twitter.com/finbarrtimbers/status/1931569704696676637).
- **The pain of caffeine withdrawal**: In a tweet that resonated far beyond the tech world, [@DavidSHolz](https://twitter.com/DavidSHolz/status/1931579805184795091) lamented the horrors of caffeine withdrawal, sparking a massive thread of shared experiences and "hacks."
- **SF Indian food superiority**: In a controversial but popular take, one user declared that "[Indian food in SF is better than Indian food in India](https://twitter.com/Yuchenj_UW/status/1931555219558859205)," leading to a spirited debate.
- **On the nature of reality**: One user humorously noted, "It is so funny that all those ancient humans who didn't understand anything imagined magic permeating the world, invokable with spells and incantations. Anyway, [what's the wifi password again?](https://twitter.com/jachiam0/status/1931376323609743572)"

---

# AI Reddit Recap

## /r/LocalLlama Recap

### 1. DeepSeek R1 0528 Coding Benchmark Achievements

- [**1.93bit Deepseek R1 0528 beats Claude Sonnet 4**](https://www.reddit.com/r/LocalLLaMA/comments/1l6v37m/193bit_deepseek_r1_0528_beats_claude_sonnet_4/) ([Score: 312, Comments: 105](https://www.reddit.com/r/LocalLLaMA/comments/1l6v37m/193bit_deepseek_r1_0528_beats_claude_sonnet_4/)): **Unsloth's quantized IQ1_M version of DeepSeek R1 0528 (GGUF format, ~200GB), evaluated with a 65535 token context on a 224GB VRAM multi-GPU system, achieved a 60% pass rate on the [Aider Polygot Benchmark](https://aider.chat/docs/leaderboards/), surpassing Claude Sonnet 4's "no think" score of 56.4%. Benchmark details: 225 test cases, 96.4% well-formed responses, 9 malformed, 6 timeouts, and near-total context utilized. The test infrastructure used a mixed GPU setup (2x RTX 5090, 1x Blackwell Pro 6000, 1x RTX 4080, 2x RTX 3090) with llama.cpp in server mode, 16 threads, and gguf tensor-split for memory balancing. See [result log](https://aider.chat/docs/leaderboards/) and [model repository](https://huggingface.co/unsloth/DeepSeek-R1-0528-GGUF).** Discussion highlights that DeepSeek R1 0528 is an autoregressive ("thinking") model, whereas Claude Sonnet 4's benchmark was in "no think" mode, potentially impacting fairness—Claude Sonnet with thinking enabled scored 61.3%. Updates to DeepSeek (improved tool calling, fixed chat templates) are noted, with releases available on [HuggingFace](https://huggingface.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF) and [discussion of updates](https://huggingface.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF/discussions/7).
    - danielhanchen details ongoing updates to DeepSeek R1 0528, highlighting improvements to tool calling and chat templates. The new model update provides native tool calling support without the need for auto <|Assistant|> appending, as referenced in their [HuggingFace release discussion](https://huggingface.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF/discussions/7).
    - offlinesir notes the benchmark comparison used may not fully capture model capabilities: DeepSeek was tested against the Claude 4 '<no think>' benchmark, which disables reasoning. In contrast, "thinking" or chain-of-thought benchmarks (e.g., Claude 4 32k with reasoning enabled) yield higher results—Claude Sonnet 4 achieves 61.3% in this context. DeepSeek is noted to be much more cost-effective.
    - daavyzhu references DeepSeek's official reporting of a 71.6 Aider score for R1 0528, shared on their [Chinese news update](https://api-docs.deepseek.com/zh-cn/news/news250528), suggesting strong coding task performance and positioning it above Sonnet 4 in certain benchmarks.
- [**DeepSeek R1 0528 Hits 71% (+14.5 pts from R1) on Aider Polyglot Coding Leaderboard**](https://www.reddit.com/r/LocalLLaMA/comments/1l76ab7/deepseek_r1_0528_hits_71_145_pts_from_r1_on_aider/) ([Score: 230, Comments: 92](https://www.reddit.com/r/LocalLLaMA/comments/1l76ab7/deepseek_r1_0528_hits_71_145_pts_from_r1_on_aider/)): **DeepSeek R1 0528 has achieved a score of 71% (+14.5 percentage points over the previous R1 release) on the Aider Polyglot Coding Leaderboard, as presented in the [official leaderboard](https://aider.chat/docs/leaderboards/). The model demonstrates significant advancement in code generation and correctness, particularly in multilingual coding benchmarks, while maintaining low operational costs (approximately $5 for ~70% benchmark completion) according to community analysis.** Top commenters highlight the model's notable improvements—calling it a leap worthy of a full version bump—and emphasize its high correctness-per-dollar relative to competitors like OpenAI and Google Gemini, as well as its strong performance in creative writing tasks.
    - Multiple users emphasize the *cost-effectiveness* of DeepSeek R1 0528, with one stating it achieves "~70% of the benchmark for under $5", far undercutting competitors like OAI's GPT-3.5/4 (o3), Gemini, and Claude. This suggests DeepSeek delivers superior "correctness per dollar spent" compared to other LLM providers, making it a strong contender for practical deployment when performance-per-cost is a key metric.
    - A commenter notes that R1 0528 represents such a significant performance leap (+14.5 points on the Aider Polyglot Coding Leaderboard)—and major improvement in creative writing—that "most other companies would have called it R2". This highlights not only the pace but the scale of progress in this model iteration.
    - Competition in the coding model space is intensifying, with the mention of Google's Gemini Pro 06-05 as another highly performant option. However, the user indicates a strong preference for DeepSeek R1 1.5 based on its pricing and performance, except in cases where Gemini's most recent release is specifically required.

### 2. Novel AI Hardware and Efficient Inference Techniques

- [**KVzip: Query-agnostic KV Cache Eviction — 3~4× memory reduction and 2× lower decoding latency**](https://i.redd.it/bpxlu6tfnw5f1.png) ([Score: 289, Comments: 26](https://www.reddit.com/r/LocalLLaMA/comments/1l75fc8/kvzip_queryagnostic_kv_cache_eviction_34_memory/)): **The image evaluates the performance of the Qwen2.5-7B-Instruct-1M model with three KV cache configurations—No Context, Full KV Cache, and the newly-proposed KVzip. KVzip delivers a dramatic reduction in memory usage (from baseline levels down to 4.6GB) and faster decoding latency (14.1ms/token), while maintaining the ability to answer context-related questions comparably to the Full KV Cache. The test is conducted on questions about 'Harry Potter 4, Goblet of Fire.'** Key commenters critique the test methodology, noting the model likely already knows Harry Potter content from pretraining, so results may not reflect true compression utility. Other technical feedback notes unexpected benchmark behaviors (e.g., better performance with smaller caches), attributing this to possible removal of distracting irrelevant information, but also warning this highlights potential evaluation flaws and oversights in test selection.
    - A commenter critiques the paper's evaluation methodology, noting that performance metrics drawn from testing on well-known texts (e.g., "Harry Potter and the Goblet of Fire") may not reflect true compression effectiveness, as models like Qwen2.5-7B-Instruct likely have these books in their pretraining data. To meaningfully assess KVzip's compression, tests should use data unfamiliar to the model, such as recent news articles, to avoid pretrained knowledge confounds.
    - Another commenter observes unexpected benchmark outcomes: in some cases (e.g., MultiHop test), aggressively reducing the KV cache even improves accuracy (from 40% to 45%), prompting speculation about "long-context degradation" where excessive context can distract LLMs. However, the consistency of these results is questioned, since query-agnostic eviction isn't expected to reliably preserve only irrelevant information. There are also gaps in evaluation: specific tests like fiction.LiveBench aren't reported, which might show greater performance degradation when information eviction is imperfect.
    - A user questions whether KVzip is applicable to Vision-Language Models (VLMs) since image features ultimately result in KV tensors post-encoding. They ask if there are modality-specific pitfalls when compressing image-derived KV caches and whether such compression has been tested, suggesting technical curiosity about the generality of the method and potential edge cases in multimodal architectures.
- [**China starts mass producing a Ternary AI Chip.**](https://www.reddit.com/r/LocalLLaMA/comments/1l7dj3z/china_starts_mass_producing_a_ternary_ai_chip/) ([Score: 110, Comments: 52](https://www.reddit.com/r/LocalLLaMA/comments/1l7dj3z/china_starts_mass_producing_a_ternary_ai_chip/)): **Chinese researchers have reportedly achieved mass production of the world's first ternary (non-binary) AI chip, using carbon-based materials, as covered in [SCMP](https://www.scmp.com/news/china/science/article/3313349/beyond-1s-and-0s-china-starts-mass-production-worlds-first-non-binary-ai-chip). Ternary AI chips process information using three states (rather than binary's two), potentially boosting computational efficiency for AI workloads, such as those using ternary quantization like BitNet. This advancement could enable significant hardware acceleration for ternary neural network inference, but implementation challenges remain due to entrenched binary software stacks.** Commenters question the lack of detailed attribution to specific companies and urge independent review of the chip's performance claims. Technical skepticism was raised about the feasibility of quickly leveraging such hardware in practice, citing major challenges due to the deeply binary orientation of existing low-level software ecosystems.
    - BumbleSlob questions the plausibility of the ternary chip, noting skepticism until there is independent validation. They highlight a key technical challenge: the entire modern software and hardware ecosystem, especially low-level software such as firmware and compilers, is built around binary logic, making ternary architecture integration and programming a potentially significant hurdle.

### 3. Reasoning-Aware LLM Workflows in Open WebUI

- [**Concept graph workflow in Open WebUI**](https://v.redd.it/dzeqvwa9rv5f1) ([Score: 115, Comments: 14](https://www.reddit.com/r/LocalLLaMA/comments/1l71iie/concept_graph_workflow_in_open_webui/)): **The described concept graph workflow in Open WebUI introduces a reasoning engine where an LLM identifies and connects concepts relevant to a user's query before producing a final response. This workflow leverages an OpenAI-compatible proxy and streams a dedicated HTML artifact that visually represents and updates the reasoning process in real-time by connecting the web UI to backend workflow events. Full implementation details can be examined in the [concept.py](http://concept.py/) [module](https://github.com/av/harbor/blob/main/boost/src/modules/concept.py#L135) of the Harbor repository.** Commenters note the workflow's value in enhancing transparency of the LLM's decision process and enabling introspection, but question its practical utility and whether it integrates with or augments base inference. Some raise concerns about whether it is more demonstrative than useful for production tasks.
    - Discussion highlights user confusion around whether the concept graph is integrated into base model inference or functions as an auxiliary tool—a technical distinction affecting how transparent and cohesive reasoning traces are when using Open WebUI.
    - A technical critique notes UI/UX concerns: Compared to textual explanations, graph nodes require manual user interpretation to follow the logic, lacking a clear visual hierarchy. This raises questions about workflow efficiency and the cognitive load for technical users who must reconstruct the reasoning path.
    - A request is raised for detailed comparison between this concept graph workflow and a previous tool from the same developer, specifically regarding implementation differences, intended use-cases, and their respective integration depths.

## Other AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo
> 

### 1. Apple 'Illusion of Thinking' Paper Debate and Reasoning in LLMs

- [**The Apple "Illusion of Thinking" Paper Maybe Corporate Damage Control**](https://www.reddit.com/r/singularity/comments/1l73qne/the_apple_illusion_of_thinking_paper_maybe/) ([Score: 229, Comments: 174](https://www.reddit.com/r/singularity/comments/1l73qne/the_apple_illusion_of_thinking_paper_maybe/)): **The post critiques Apple's recent research paper (pre-announce for WWDC) on the 'illusion of thinking' in large language models (LLMs), claiming the work is more corporate narrative control than scientific advance. The paper investigates LLMs on classic algorithmic puzzles like Tower of Hanoi and River Crossing, asserting that models fail above a complexity threshold, cannot generalize reasoning even given correct algorithms, and tend to reduce reasoning effort with increased complexity—contrasting with standard evaluations reliant on math/coding benchmarks. Author questions Apple's claims of benchmark contamination and suggests the problems used are themselves highly contaminated, arguing the paper's framing is more marketing than science, especially given relative AI stagnation at Apple and the paper's timing. The notes also point out alleged inconsistencies and rhetorical weaknesses in Apple's arguments regarding reasoning, benchmark design, and performance analysis, while observing that puzzles used are foundational CS exercises.** Top technical comments highlight: (1) the social media/confirmation bias dynamic around the paper, (2) that the Apple paper's finding that even with explicit algorithms, LLMs could not reliably solve simple algorithmic problems is a significant, actionable critique, and (3) speculation that Apple may be justifying a lack of pursuit of 'thinking' architectures. Comparisons are made to prior Anthropic research questioning LLMs' traditional reasoning capability.
    - Multiple commenters highlight that the Apple paper demonstrates serious limitations of current LLMs—specifically, failures to solve algorithmic or reasoning tasks like Tower of Hanoi with 7 pegs, even when provided with explicit algorithms. This suggests a major shortfall in LLMs' capacity to execute symbolic reasoning and algorithm application compared to classical computing approaches.
    - There is mention that the river crossing problem with 3 actors (a classic, straightforward graph search) also stumped the LLM, which is technically trivial and under 50 steps in solution space—underscoring the inability of LLMs to handle even basic state-space search or nontrivial reasoning tasks reliably.
    - One user refers to rumors of Apple having more advanced internal models, but posits that Apple may have intentionally chosen not to focus on architectures capable of true reasoning, positioning the paper as a form of preemptive defense. A parallel is drawn to prior Anthropic research suggesting existing models excel at pattern isolation from answer pools but do not "think" traditionally or generalize algorithmically.
- [**Why are so many people so obsessed with AGI, when current AI will still be revolutionary?**](https://www.reddit.com/r/singularity/comments/1l7b91a/why_are_so_many_people_so_obsessed_with_agi_when/) ([Score: 132, Comments: 203](https://www.reddit.com/r/singularity/comments/1l7b91a/why_are_so_many_people_so_obsessed_with_agi_when/)): **The post questions the community focus on AGI (Artificial General Intelligence) rather than the substantial impact of current AI capabilities, referencing a recent Apple paper whose findings reportedly underscore this point. The author highlights concrete near-term applications: Level 4 self-driving, AI outperforming doctors, and humanoid robots doing physical work, arguing that existing and near-future AI will be *revolutionary* without AGI. The post links AGI enthusiasm, especially in Silicon Valley and venture capital, to the pursuit of extreme financial returns, but questions its broader cultural hold.** Top technical comments frame AGI as uniquely transformative compared to current AI, emphasizing scenarios (e.g., curing all diseases, radical longevity) not achievable by narrow AI. The presence of AGI discourse is also attributed to the forum's focus on the 'singularity' and desire for paradigm-shifting advances.
    - One commenter notes that many 'revolutionary' use-cases, such as prompting AI to create an entire TV show or video game from a detailed spec, don't strictly require AGI, but instead could be accomplished with multiple agentic specialized AIs. This highlights a technical distinction between full AGI and coordinated specialized agents, suggesting some major creative automation is plausible before reaching AGI.
    - The debate distinguishes the ambitions of generalist AGI (Artificial General Intelligence)—capable of curing diseases or radically extending lifespan—from the still-transformative, but more narrow, capabilities of current AI systems. The technical aspiration towards AGI is framed by commenters as enabling breakthroughs beyond what present models achieve.
- [**What’s with everyone obsessing over that apple paper? It’s obvious that CoT RL training results in better performance which is undeniable!**](https://www.reddit.com/r/singularity/comments/1l77u6t/whats_with_everyone_obsessing_over_that_apple/) ([Score: 110, Comments: 55](https://www.reddit.com/r/singularity/comments/1l77u6t/whats_with_everyone_obsessing_over_that_apple/)): **The discussion focuses on an Apple paper demonstrating core limitations of Chain-of-Thought (CoT) Reinforcement Learning (RL) training for language models: while CoT RL models outperform baselines by leveraging added token-level compute and intermediate structure, their reasoning performance collapses when faced with problems requiring more than ~eight genuine thinking steps—suggesting a hard ceiling to current architectures ([Apple paper reference](https://arxiv.org/abs/2404.03370)). Notably, training on high-entropy (uncertain) tokens as shown in Qwen improves efficiency but doesn't break this reasoning limit; models that use non-semantic placeholder tokens (e.g., dots/dashes instead of logical traces) do better than non-CoT models but revert to mid-level performance, indicating that both extra computation and semantics contribute.** Top comments highlight that Apple’s results are less about declaring LLMs useless and more about empirically mapping out the boundaries of current methods—pointing to a need for external memory or symbolic reasoning for solving tasks beyond this threshold. Technical discourse also critiques online polarization around AI capability, calling for more nuanced discussion of these architectural limitations and research directions.
    - Apple's paper finds that models trained with CoT RL (Chain-of-Thought Reinforcement Learning) exhibit a hard performance ceiling: once tasks require about eight or more genuine reasoning steps, even the best models stop generating coherent reasoning chains and accuracy collapses. This highlights inherent limits in current architectures' stepwise reasoning abilities, regardless of training method.
    - CoT RL achieves better-than-baseline results because the chain-of-thought structure acts as a computational scratch pad, providing both intermediate structure (semantics) and additional compute—boosting performance. When CoT steps are replaced with semantically meaningless placeholders, the extra 'thinking time' still helps, but performance degrades, demonstrating that the content of intermediate reasoning is key, not just the computation budget.
    - The Apple researchers also utilized a token uncertainty sampling strategy—training on the 20% of tokens with the highest prediction uncertainty—to boost efficiency. However, this technique does not overcome the underlying performance ceiling, suggesting fundamental algorithmic limitations that may require entirely new approaches, such as external memory or symbolic planning, to enable models to reason across twenty or more steps.

### 2. OpenAI and Industry Revenue, Benchmarks, and GPU Race

- [**Breaking: OpenAI Hits $10B in Reoccurring Annualized Revenue, ahead of Forecasts, up from $3.7B last year per CNBC**](https://i.redd.it/uyqrtp449y5f1.jpeg) ([Score: 399, Comments: 105](https://www.reddit.com/r/singularity/comments/1l7db2u/breaking_openai_hits_10b_in_reoccurring/)): **The image depicts a man delivering a professional presentation with the OpenAI logo behind him, visually reinforcing the post's announcement that OpenAI has achieved $10B in recurring annualized revenue, up from $3.7B last year. The post links to a CNBC article highlighting this record growth, attributed to the widespread adoption of ChatGPT and related services. The context of the image—the stage setup and focus on the OpenAI brand—emphasizes the company's elevated market significance and public profile following this major financial milestone.** Commenters note the rapid revenue growth as evidence of 'recursive self improvement,' and discuss evolving public perceptions around OpenAI's profitability and potential future pricing shifts (e.g., speculation about higher subscription fees).
    - A user questions whether reaching a $10B ARR means that OpenAI is now actually profitable, highlighting how high ARR does not necessarily equate to net profits due to ongoing high compute, R&D, and scaling costs typically faced by AI companies. The comment raises the important distinction between recurring revenue milestones and actual profitability in evaluating company performance.
- [**OpenAI hits $10B Revenue - Still Loosing Millions - ChatGPT growth is insane**](https://www.reddit.com/r/ChatGPT/comments/1l7dcjw/openai_hits_10b_revenue_still_loosing_millions/) ([Score: 240, Comments: 61](https://www.reddit.com/r/ChatGPT/comments/1l7dcjw/openai_hits_10b_revenue_still_loosing_millions/)): **OpenAI has reached an annual recurring revenue (ARR) of $10B, doubling its previous year's figure, primarily from ChatGPT consumer subscriptions, enterprise sales, and API usage, as reported by CNBC. The company claims** `500M weekly users` **and** `3M+ business customers`**, but still operates at a significant loss: about** `$5B` **in the past year, excluding substantial Microsoft licensing revenue, and sets an ambitious $125B ARR target by 2029. Core technical challenge highlighted by commenters is the unpredictable scaling of compute and infrastructure costs as usage grows; OpenAI's financial strategy is presently focused on market leadership rather than short-term profitability.** Comments note that OpenAI's expansive financial losses are a deliberate strategy to capture AI market dominance, with technical debate centering on uncertainties about the cost structure and scalability of large AI deployment.
    - There is uncertainty in how OpenAI's operational costs, particularly for large models like ChatGPT, will scale as user numbers and usage increase. Unlike traditional SaaS, inference costs in generative AI remain significant, with hardware constraints playing a major role in ongoing expenses.
    - OpenAI's current strategy appears to prioritize market dominance over immediate profitability. The company is reinvesting revenue and focusing on user acquisition and rapid product improvement, suggesting an intent to establish a technological and data moat rather than maximizing short-term profit.
    - As usage surges, OpenAI faces technical challenges with model reliability, such as maintaining system load and preventing model degradation or "corruption" from adversarial prompts. Ensuring that the model remains robust and uncorrupted under heavy, diverse usage is an ongoing technical concern.
- [**Meta's GPU count compared to others**](https://i.redd.it/b5817i11ct5f1.jpeg) ([Score: 507, Comments: 157](https://www.reddit.com/r/singularity/comments/1l6toye/metas_gpu_count_compared_to_others/)): **The image is a tweet presenting a comparative chart of H100 GPU allocations among leading AI players, highlighting Meta's ownership of 350,000 H100 GPUs—vastly surpassing others like Google, Tesla, and Anthropic. The chart and tweet critique Meta's accumulation and exclusive internal use of these GPUs, raising concerns about potential underutilization and advocating for regulatory oversight to prevent private hoarding of national compute resources. Commenters note Meta's internal focus for AI development, the variable and rapid pace of model quality improvement, and debate whether Meta can leverage its scale to catch up in the AI race despite previous mediocre public model releases.** Discussion centers on whether Meta's strategy of hoarding compute and using it internally positions it for future dominance, with users pointing out previous rapid shifts in model quality (as with Llama 3.3) and warning against dismissing Meta due to past performance. Some debate the wisdom and effectiveness of Meta's internal-only approach versus more open or commercially engaged competitors.
    - Meta's AI strategy is focused on internal usage rather than competing directly with public consumer models, as evidenced by statements from Zuckerberg and Meta's shift towards closed-source model deployment. This is a significant departure from their previous open-source releases like LLaMA, and some speculate that if Meta's internal efforts succeed, it could make their lead in AI difficult to challenge for competitors.
    - There is discussion about the rapid evolution of Meta's LLaMA models: earlier iterations (pre-3.2) were considered subpar, but LLaMA 3.3 reportedly brought performance close to state-of-the-art (SOTA) in open source. This rapid improvement is compared to Google's trajectory, noting how companies once perceived as lagging can quickly catch up in model quality.
    - Meta's investment in hardware is noted, with references to them acquiring '350K H100s' (a significant GPU count), yet receiving criticism for the perceived underperformance of Llama 4. Critics question the return on this hardware investment relative to model quality, arguing that compute alone does not guarantee superior results.

### 3. AI Coding Benchmarks, Claude & Gemini Collaboration, and User Feedback

- [**New SOTA on aider polyglot coding benchmark - Gemini with 32k thinking tokens.**](https://i.redd.it/d41kmdpwnw5f1.png) ([Score: 224, Comments: 32](https://www.reddit.com/r/singularity/comments/1l754k9/new_sota_on_aider_polyglot_coding_benchmark/)): **The image presents a comparative leaderboard from the 'aider polyglot coding benchmark,' highlighting that the 'gemini-2.5-pro-preview-06-05 (32k think)' model has achieved a new state-of-the-art (SOTA) score with 83.1% correctness at a cost of $49.88. The table juxtaposes Gemini's latest model against alternatives like o3 (high) and other Gemini versions, showcasing improvements not only in accuracy but also in cost-efficiency, though the data indicate rising costs for newer Gemini iterations. Links to the official leaderboard and announcement tweet are provided for detailed comparison and context.** Top comments discuss observed discrepancies in Gemini's real-world coding utility (such as with tool use in Cursor), noting its strong benchmark performance may not translate to practical reliability. There's also technical debate on Gemini's increasing costs narrowing its traditional price advantage, and queries regarding testing parameters such as temperature settings.
    - A user questions Gemini's performance consistency, highlighting that although the model achieves top results on the aider polyglot coding benchmark, it *frequently fails at tool use and basic edit operations* within the Cursor environment. This raises concerns around the model's practical usability versus its benchmark performance.
    - There is discussion about costs and trends: Gemini models, previously known for being *10x cheaper* for similar performance, are now *only 2x cheaper than OpenAI's o3*, with their competitiveness increasing but price gap narrowing. If this cost trend continues, convergence with competitor pricing is expected.
    - A detailed analysis notes that expanding to *32k thinking tokens* increases cost by only `$4.28` but *reduces task failures by 19%*, suggesting it's a very worthwhile option unless inference time is essential. Additionally, Gemini 2.5 Pro is considered verbose regardless of the thinking budget, indicating verbosity is a model characteristic rather than a function of token allocation.
- [**Claude Code + Gemini Pro: Two AI Coders Working as One**](https://www.reddit.com/r/ClaudeAI/comments/1l73a1x/claude_code_gemini_pro_two_ai_coders_working_as/) ([Score: 285, Comments: 108](https://www.reddit.com/r/ClaudeAI/comments/1l73a1x/claude_code_gemini_pro_two_ai_coders_working_as/)): **A new MCP server ([gemini-mcp-server](https://github.com/BeehiveInnovations/gemini-mcp-server)) allows Claude Code and Gemini 2.5 Pro to collaborate on code generation and review tasks. The workflow involves Claude Code initiating the analysis and planning, while Gemini leverages its 1M-token context window and deep reasoning to augment and refine Claude's output, yielding measurable improvements (e.g.,** `26%` **faster JSON parsing in a target library after collaborative review and optimization). The server adds support for extended context, file I/O, full-repo code review, debugging, and iterative testing based on performance benchmarks, with the workflow involving structured prompt engineering to alternate reasoning and validation phases between the two LLMs.** One commenter questions potential prompt interaction effects, wondering if instructing Gemini to 'think deeper' could interfere with Claude's independent reasoning during their collaboration. No deep technical debates are present yet, though there is clear interest in multi-LLM orchestration.
    - A commenter describes a workflow leveraging **Gemini 2.5 Pro** for high-level system architecture and planning (due to its larger context window), then using **Claude 3 Opus (o3)** for detailed implementation and bugfixing. They note that passing outputs and critique between the two models iteratively (i.e., model interleaving) enables more effective troubleshooting, with *“two models in a pair produce better results than one alone.”* This approach essentially uses each model’s strengths in concert for complex code tasks.
    - A technical question is raised about prompt engineering: specifically, whether instructing **Gemini Pro** to "think deeper" could interfere with **Claude Code's** reasoning process, due to the prompt word "think" potentially impacting prompt interpretation or model behavior.
    - Another commenter compares this workflow with tools like **Aider as an MCP** for integrating multiple providers in coding. They ask if **Gemini** can directly provide code diffs for **Claude** to apply, or if Gemini simply acts as a brainstorming assistant, hinting at the importance of API-level integrations and differentiating design automation versus hands-on code collaboration between the models.
- [**I’m done with ChatGPT (for now)**](https://www.reddit.com/r/ChatGPTCoding/comments/1l6rwsi/im_done_with_chatgpt_for_now/) ([Score: 102, Comments: 93](https://www.reddit.com/r/ChatGPTCoding/comments/1l6rwsi/im_done_with_chatgpt_for_now/)): **The user reports that OpenAI's "o4 mini high" model underperformed on a complex scripting task, repeatedly failing to produce complete or accurate results over several days. In contrast, Google's Gemini model successfully generated a 1,500-line script that solved the problem in a single pass, without specific guidance on the required fix. Key technical complaint centers on model regression in code generation and hallucination issues. Model selection and alignment with task complexity are highlighted as critical for outcome quality.** Comments note that 'o4 mini high' is not intended for complex programming, emphasizing its design for rapid, simple tasks; the o3 model is recommended for complex code. Multiple users underscore the importance of being model-agnostic and using whichever model performs best per project, noting both cost and outcome quality. Reports of hallucinated legal citations and failure to extract details from provided PDFs further undermine trust in o4 mini high for precision tasks.
    - Multiple commenters highlight that o4-mini-high is not suitable for complex code or detailed technical/research tasks. One user stresses that 'mini' in the name signals the model's optimization for speed and simple queries, not deeper reasoning or complex outputs, recommending o3 for such cases.
    - A critique is raised about o4-mini-high's hallucination issues, particularly in professional and academic contexts such as legal citations and referencing peer-reviewed papers. The model has been observed generating nonexistent case law and failing to extract information accurately even after ingesting research-oriented PDFs, whereas "frontier models" perform better in these areas.
    - A technical workflow suggestion is provided: starting a new chat with the model can help reset conversation context, preventing prior failures from biasing current results—this is important to fairly benchmark or evaluate model consistency across sessions.

---

# AI Discord Recap

> A summary of Summaries of Summaries by Gemini 2.5 Pro Exp
> 

**Theme 1: Bleeding-Edge Model Developments & Performance Showdowns**

- **Unsloth Supercharges DeepSeek-R1 with Native Tool-Calling & Stellar Quantization**: Unsloth's **DeepSeek-R1-0528-Qwen3-8B-GGUF** model now boasts **native tool calling**, hitting **93%** on the **BFCL (Berkeley Function Calling Leaderboard)** and also benefits from **UTF-8 chat template fixes**. The **IQ1_M quant** version (200GB) of **Deepseek R1 0528** shows exceptional performance, potentially matching the full R1 on Aider's Polygot benchmark with a **57% success rate** and **100% well-formed responses**.
- **Gemini & Claude Challenge OpenAI's Dominance as Users Eye Alternatives**: Users across multiple Discords report that **Google's Gemini** (especially **Gemini Pro** for reasoning and **Gemini 2.5 Pro** for its 1 million token context window) and **Anthropic's Claude** (excelling at creative writing and coding with **Claude 4.0**) are increasingly preferred over **OpenAI's** offerings. Discussions highlighted **Gemini's** larger context window and comparable performance as key advantages, with some users in OpenRouter stating they've deprecated OpenAI models except for **GPT-4o-mini** as *Gemini Pro seems unbeatable for reasoning, thinking, and very long chains of thought*, while *Claude excels at creative writing*.
- **NVIDIA's Nemotron & Apple's AI Reasoning Scrutiny Ignite Debate While Sydney Persona Shines**: **NVIDIA's Nemotron-Research-Reasoning-Qwen-1.5B** emerges as a top **1.5B open-weight model** for complex reasoning, according to Unsloth AI discussions, while Apple's [The Illusion of Thinking paper](https://ml-site.cdn-apple.com/papers/the-illusion-of-thinking.pdf) sparks debate by suggesting leading AI 'reasoning' models mainly perform pattern memorization. Meanwhile, the LMArena community found the **Sydney dataset** reveals **OpenAI's Flash 2.5** excels at mimicking the **Bing Sydney** persona, outperforming **GPT-4.5** which struggles beyond five messages.

**Theme 2: Dev Tooling & Infrastructure: The Nuts and Bolts of AI Ops**

- **OpenRouter Revamps Fees & Users Demand Model Versioning for Sanity**: OpenRouter simplifies its platform fee, removing the fixed **$0.35** on Stripe and setting non-crypto payments at **5.5%** (min **$0.80**) and crypto at **5.0%** (no min), while a new BYOK subscription model sparks debate among users. Users also requested better model management through versioned IDs, similar to upstream providers, to track updates and avoid surprises.
- **LM Studio Users Eye Server Alternatives as VLLM Crowd Yearns for a GUI**: Users on **Ubuntu Server** discuss bypassing the **LM Studio GUI** for direct **llama.cpp** or **Ollama** usage, as LM Studio is primarily a wrapper for server environments. For more advanced setups, many using **VLLM** for its query parallelizing capabilities express a strong desire for an LM Studio-like GUI to manage flags and parameters, transforming command-line arguments into user-friendly checkboxes.
- **Agent & Tool Protocols Battle for Supremacy, LlamaIndex Champions MCP**: The **Model Collaboration Protocol (MCP)** sees active development with new servers like a [security-focused Google MCP server by robcerda](https://github.com/robcerda/google-mcp-server) and tools like a [MCP specification utility on Gist](https://gist.github.com/hesreallyhim/d974990b8c80cf6f32b88bfe39b76f9a). **LlamaIndex** hosts office hours on MCP and highlights the **13 different protocols** (including **MCP, A2A, ACP**) vying for agent-tool communication standards in a [MCP Dev Summit presentation on YouTube](https://www.youtube.com/watch?v=kqB_xML1SfA).

**Theme 3: Hardware & Optimization: Squeezing Every Last FLOP**

- **Dual GPU Dilemmas & Quantization Wins Steal Hardware Spotlight**: Discussions in LM Studio highlight challenges with **dual GPU setups**, focusing on **PCIe lane splitting** and performance impact, with suggestions like [two RTX 5060 Ti 16GB cards with VLLM detailed on Hardware Corner](https://www.hardware-corner.net/guides/dual-rtx-5060-ti-16gb-vs-rtx-3090-llm) as alternatives to an RTX 3090. **Unsloth's IQ1_M quant** for **Deepseek R1 0528** impresses with robust performance, while users optimize **Qwen3 models** in LM Studio by enabling **flash attention** and **KV cache** at **Q8**.
- **Apple Silicon Flexes Muscles with DeepSeek R1 as Memory Bandwidth Remains King**: **DeepSeek R1** runs effectively on **Apple's M3 Ultra** with **512GB** unified memory, showcasing Apple's strength in unified memory systems for AI inference as detailed in [this /r/LocalLLaMA Reddit post](https://www.reddit.com/r/LocalLLaMA/comments/1j9vjf1/deepseek_r1_671b_q4_m3_ultra_512gb_with_mlx/). However, users across communities lament that **memory bandwidth** remains a critical bottleneck, often more pressing than NPU development for many LLM tasks.
- **Kernel-Level Optimizations & Compiler Breakthroughs Promise Ludicrous Speed**: **TinyGrad** users report a **10x speedup** in tensor indexing using `FUSE_ARANGE=1`, while the **Mojo 🔥** community buzzes about [TPDE, a faster LLVM backend, discussed on LLVM Discourse](https://discourse.llvm.org/t/tpde-llvm-10-20x-faster-llvm-o0-back-end/86664), a new **LLVM backend** promising 10-20x faster speeds than **LLVM O0**. In GPU MODE, fanwenjie shares solutions for **MLA-decode** and **FP8-mm** achieving **3.92 ms** on **MI300**, available on [their gitee reference-kernels repo](https://gitee.com/fanwenjie/reference-kernels).

**Theme 4: AI's Expanding Horizons: Novel Applications & Ethical Frontiers**

- **NotebookLM Conjures Audiobooks & ChatGPT Dabbles in BIOS Patching**: Users in Notebook LM discovered it can generate lengthy audiobooks (one **82-minute audiobook** reported) and impressive podcast intros like [this user-generated Ummmmmmmmm.mp3](https://cdn.discordapp.com/attachments/1124403655819415592/1381730009601015890/Ummmmmmmmm.mp3). In a more technical feat discussed in Yannick Kilcher's server, **ChatGPT** successfully patched a **BIOS binary**, as highlighted by [Hackaday's article on ChatGPT patching BIOS](https://hackaday.com/2025/06/07/chatgpt-patched-a-bios-binary-and-it-worked/) and a [related YouTube video](https://www.youtube.com/watch?v=8JuWdXrCmWg).
- **IBM Unveils Responsible Prompting API Amidst Calls for Quantization Transparency**: IBM introduced an open-source [Responsible Prompting API on GitHub](https://github.com/IBM/responsible-prompting-api) to guide LLM outputs pre-inference, based on [their arXiv paper on responsible prompting](https://arxiv.org/abs/2504.08757) and a [HuggingFace demo](https://huggingface.co/spaces/responsible-prompting/responsible-prompting-demo). This comes as the Latent Space community, possibly referencing a [Claude subtweet from _xjdr](https://x.com/_xjdr/status/1931068996092334274), advocates for AI service providers to disclose model **quantization levels** and dynamic adjustments, demanding industry standards for verifiable inference detailed [in TheAhmadOsman's X post](https://x.com/TheAhmadOsman/status/1930944597464654272).
- **AI Transforms Learning with NoteTube & Tackles High-Stakes Diplomacy Games**: Developers are creating tools like **NoteTube** ([visit NoteTubeAI.com](https://www.notetubeai.com/#howitworks)) to turn **YouTube** into a structured learning platform, while others in Yannick Kilcher's Discord open-source an [AI Diplomacy harness shared by alxai_ on X](https://x.com/alxai_/status/1930653096071635112) for LLMs to play the complex strategy game. Perplexity AI continues to be a source for curated information, with users sharing pages on topics like [Russia Offers Musk Asylum via Perplexity](https://www.perplexity.ai/page/russia-offers-musk-asylum-KjWIaYM3R6iarn85k2CaAA).

**Theme 5: Community-Driven Innovation & Open Source Ecosystem Flourishes**

- **Open Datasets & Research Fuel Collaborative AI Advancement, "Common Pile" Sparks Debate**: The Eleuther community discusses the naming of the **Common Pile** dataset and plans for a paper comparing it to **Llama 3**, while also sharing research on [openMaMMUT-L/14, a language-vision model detailed by JJitsev on X](https://x.com/JJitsev/status/1931569060438737161), trained on **DataComp-1.4B**. A new [paper on packing contamination from arXiv](https://arxiv.org/abs/2410.08081) shared in Unsloth suggests contamination *counterintuitively actually improves downstream eval a little*.
- **"Awesome Agent Learning" & GPU Kernels Showcase Open Source Spirit**: A HuggingFace member shares a curated list of AI/LLM agent resources, [Awesome Agent Learning on GitHub](https://github.com/artnitolog/awesome-agent-learning), encouraging contributions. In GPU MODE, fanwenjie publicly discloses their **MLA-decode** and **FP8-mm** kernel solutions on [their gitee reference-kernels repo](https://gitee.com/fanwenjie/reference-kernels/), achieving **3.92 ms** on **MI300**, with a detailed writeup in Chinese on [Bilibili](https://www.bilibili.com/read/cv41954307).
- **Users Unite to Report Bugs and Request Features Across Platforms, Turning Frustration into Progress**: Unsloth users collaboratively diagnose an **Android Chrome crash** linked to the autofill service (reproducible via [this CodeSandbox example](https://ygdzmg.csb.app/)), while TinyGrad users report **Metal compiler bugs** turning *beautiful mnist into beautiful macos dos poc* due to driver issues. Feature requests abound, from a **save-chat feature** in GPT4All to **model versioning** in OpenRouter and a **VLLM GUI** similar to LM Studio.

---

# Discord: High level Discord summaries




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Samsung Pro Code Leaked and Abused**: A leaked **Samsung** promotional code for a **1-year Perplexity Pro subscription** was abused, leading to the code being disabled ([Screenshot](https://cdn.discordapp.com/attachments/1047649527299055688/1381364911556530286/Screenshot_20250608_210952_Chrome.png)).
   - The Perplexity team is reportedly working on revoking access for abusers and finding a solution for legitimate users.
- **API Results Inferior to Web UI, User Laments**: A user stated that after many tests, **Perplexity API** calls return results much worse and incomplete than the **Web UI**, and the API yields an average of **2-3 citations**, but the same query to the UI returns **10+ citations**.
   - The user expressed a need to build a research agent using **Brave**, **Tavily**, or **Firevrawl** due to the API's perceived limitations.
- **Memory Feature Now Available to all**: A team member announced that the memory feature is now available for all **Free** and **Pro** users, eliminating the need for testers.
   - Users can find the memory feature in [Perplexity Personalize Account settings](https://www.perplexity.ai/account/personalize).
- **Speculation Sparks Over Silksong Release**: Users speculated about the release of **Silksong** after it was mentioned in a ROG "ad" during a game showcase.
   - Despite **Nintendo** already teasing the game, the ad sparked renewed hope for a release *this year*, prompting discussions on potential new gameplay reveals.
- **Russia Offers Asylum to Musk**: A member linked to a [Perplexity page](https://www.perplexity.ai/page/russia-offers-musk-asylum-KjWIaYM3R6iarn85k2CaAA) about **Russia** offering asylum to **Elon Musk**.
   - A member also shared a [Perplexity page](https://www.perplexity.ai/page/largest-map-of-the-universe-co-lvRe2dwTS2ixrAzcHa6nGQ) about the **largest map of the universe**.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Sydney Dataset Exposes GPT-4.5 Weakness**: A user-created **Sydney dataset**, featuring saved conversations and **Bing instructions**, reveals that **Flash 2.5** excels at mimicking Sydney, whereas **GPT-4.5** struggles to maintain the persona beyond five messages.
   - Without instructions, **GPT-4.5** behaves like **4o**, contrasting with Flash 2.5's opposite reaction.
- **Titan Infra, Not Upcoming Model**: Inquiries about **Titanforge's** release date were clarified: **Titan** refers to infrastructure, not a model codename.
   - The information is considered relatively safe and public.
- **Grok 3.5 Hype Builds, Kingfall Disappointment Lingers**: Enthusiasm surrounds the potential launch of **Grok 3.5**, with comparisons to **Kingfall's** performance being drawn.
   - Magic particles on the **Grok UI** hint at an imminent release.
- **Apple's AI Acquisition: FTC Scrutiny?**: Speculation arises around **Apple's potential acquisitions**, notably **Anthropic**, but regulatory obstacles are recognized.
   - One member asserted that *there is a 0% chance that Apple attempts to acquire Anthropic without the FTC getting involved*, while another shared [an apple engineer playing tetris while working on the neural engine](https://x.com/TheGregYang/status/1929055508675096970).
- **Vision Pro's Hefty Price Tag: Justified?**: Discord users debate the **Vision Pro's** **$3500+** price, weighing its advanced tech against its cost.
   - Some argue the two micro-OLED screens alone cost over **$800**, but others question if it offers enough unique functionality compared to cheaper alternatives like the **Meta Quest**, considering its ecosystem and potential for mass market adoption.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Skip LM Studio GUI for Ubuntu Server**: Members suggested bypassing **LM Studio** on **Ubuntu Server 24.04.2 LTS** in favor of using [llama.cpp](https://github.com/ggerganov/llama.cpp) or **Ollama** directly if a GUI is not needed.
   - The consensus is to go straight to the source for server environments as **LM Studio** is essentially a GUI wrapper for **llama.cpp**.
- **RooCode API Endpoint Anomaly**: A user reported issues with the experimental **RooCode** feature in **LM Studio**, encountering an *unexpected endpoint or method* error when calling the **/api/embeddings** **OpenAI API**.
   - The user suspects a limit on input lengths or JSON size, noting their custom scripts worked while **RooCode** failed, suggesting the RooCode might be referring to a non-existent endpoint.
- **Flash Attention helps Qwen3 models**: Users discussed determining the maximum comfortable context token window for **Qwen3-4B** and **Qwen3-8B** models in LM Studio, balancing conversation length and generation speed.
   - The advice was to monitor GPU memory usage, increasing context length until VRAM is nearly full, enabling **flash attention** and **KV cache** at **Q8** to optimize VRAM usage.
- **Dual GPU Dilemmas: PCIe Lane Limitations**: Discussions revolved around setting up dual GPUs, with concerns about **PCIe lane splitting** and its impact on performance, especially with consumer CPUs having fewer lanes than server CPUs.
   - Members shared that for dual RTX 3060 setups, the secondary slot would run at x4 3.0, and suggested that [two RTX 5060 Ti 16GB](https://www.hardware-corner.net/guides/dual-rtx-5060-ti-16gb-vs-rtx-3090-llm/) cards with VLLM may offer comparable performance to an RTX 3090 at a lower cost.
- **VLLM GUI Gains Yearned**: Members found VLLM's query parallelizing capabilities very effective for servicing multiple users or running multiple agent chains, and one offered [this example of the command line](https://link.to/example-vllm-command).
   - Many desire a management GUI similar to LM Studio, that turns the flags into checkboxes with descriptions and a mechanism that saves the parameters to a json.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Gemini Challenges GPT's Reign**: Some members suggest **Gemini** outperforms **GPT** due to its larger context window (**1 million tokens**) and comparable performance, making it a cheaper alternative.
   - One user stated they will never go back to **32k tokens**.
- **Apple Silicon Impresses with DeepSeek R1**: **DeepSeek R1** is running effectively on **Apple's M3 Ultra** with **512GB** unified memory, highlighting Apple's strength in unified memory systems for AI inference.
   - One member shared [a Reddit post](https://www.reddit.com/r/LocalLLaMA/comments/1j9vjf1/deepseek_r1_671b_q4_m3_ultra_512gb_with_mlx/) while also wishing for better memory speed, GPU, and software.
- **GPT Feedback System Suspected as Hallucination**: Doubts arose regarding the functionality of **ChatGPT's feedback system**, with members suggesting it may be a **hallucination**.
   - A member suggested enabling feedback emails in the [Builder Profile settings](https://chatgpt.com/#settings/BuilderProfile) as an alternative, as there's no built-in feedback system in GPTs.
- **Markdown Declared Best Data Format**: Members advocated for **markdown** as the preferred file format for model training due to its structured text and unique tokens.
   - While PDFs are usable, they are not ideal as *plain text without structure works fine and is preferred*.
- **YouTube Essays Guide Chatbot Character**: ChatGPT can analyze [YouTube video essays](https://chatgpt.com/blog/new-video-understanding-capabilities) with captions to guide the voice, tone, or characterization of AI responses, focusing on speech patterns and behaviors.
   - One member pointed out that the chatbot can *hallucinate* and the model *errors out for YouTube content URLs* unless downloaded, so downloading videos works, but not linked YouTube URLs!



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **DeepSeek-R1-0528 Gains Native Tool-Calling**: Unsloth's **DeepSeek-R1-0528-Qwen3-8B-GGUF** model now supports **native tool calling**, achieving **93%** on the BFCL (Berkeley Function Calling Leaderboard).
   - The update also addresses issues with `add_generation_prompt` and includes **UTF-8 chat template fixes**, benefiting the official DeepSeek model as well.
- **Typing Crashes Plague Android Chrome**: Users reported that typing in a document editor within **Chrome on Android** causes the browser to crash due to a bug related to the **autofill service**.
   - The issue is linked to Chrome's interaction with autocomplete services, triggering a `TransactionTooLargeException` when notifying these services of document changes, as shown [here](https://ygdzmg.csb.app/).
- **IQ1_M Quant Amazes With Robust Performance**: Unsloth's **IQ1_M quant** (200GB) for **Deepseek R1 0528** is performing exceptionally well, potentially matching the full original R1 on Aider's Polygot benchmark with **57% success rate** and **100% well-formed responses**.
   - The model consistently works in Roo Cline without missing tool calls or getting stuck in a loop, outperforming other quants.
- **Nvidia's Nemotron Shines on Reasoning**: **Nvidia's Nemotron-Research-Reasoning-Qwen-1.5B**, is the world’s leading 1.5B open-weight model for complex reasoning tasks, outperforming Deepseek’s 1.5B model by a large margin on a broad range of tasks.
   - The results are available on a broad range of tasks, including math, coding, and GPQA, as seen [here](https://huggingface.co/nvidia/Nemotron-Research-Reasoning-Qwen-1.5B).
- **Packing Contamination Perks Performance Paradoxically**: A new [paper](https://arxiv.org/abs/2410.08081) suggests that *packing contamination doesn't actually matter, and counterintuitively actually improves downstream eval a little*.
   - It is not perfectly explained in the paper, but seems like the *"big" model will have a slightly higher prob and shorter coding length*.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **EXL3 Kernels Run Natively in Transformers**: **EXL3** is now running in Transformers, though currently only the kernels and inference are supported, check out the [code on GitHub](https://github.com/huggingface/transformers).
   - It remains unclear what degree of integration is possible given changes in Transformers, particularly around support for quantized models.
- **Experiment Tracking Preferences**: Members discussed [wandb](https://wandb.ai/site), [neptune.ai](https://neptune.ai/), [mlflow](https://mlflow.org/), and [comet.ml](https://www.comet.com/) for experiment tracking, with most still preferring **wandb** due to familiarity.
   - One member noted *It seems like it does everything wandb does except i know how to use wandb way better*.
- **Awesome Agent Learning Released**: A member shares his curated collection of resources on **AI/LLM agents**, [Awesome Agent Learning](https://github.com/artnitolog/awesome-agent-learning), featuring foundational courses, readings, and framework-specific tutorials.
   - He encourages contribution via PRs for any great resources that may have been missed.
- **Transformer Attribution Graphs Get Traced**: A user shares a link to [Transformer Circuits](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) on **Attribution Graphs**.
   - They admitted they can *spend hours playing around with that tracer*.
- **GPT-4o Has Trouble Parsing Agents**: Users report encountering frequent parsing errors when using **GPT-4o** and **GPT-4o mini** with a *smolagents* code agent.
   - The users are requesting fixes and workarounds for the issue.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Gemini Max Stumbles on Edits**: Users find that while **Gemini** is quick, it struggles with applying file edits and tends to overanalyze code, often getting stuck asking unnecessary questions.
   - Some members express a preference for **Claude 4** for more complex tasks, despite Gemini's speed, due to **Gemini** repeatedly corrupting files.
- **Background Agents Bug Out**: Users report issues with **background agents**, encountering errors in finding VS Code remote workspaces and dealing with frequent interruptions from ESLint.
   - Disabling ESLint auto-fix in settings may provide a workaround, as members try to resolve the issue.
- **Claude Code Consumes Quota**: Despite praise for **Claude Code**, users express concern over **rate limits**, especially with Opus 4, leading some to revert to Claude 3.7 to conserve quota.
   - Users are exploring alternatives like [Gemini 2.5 Pro in Claude Code](https://github.com/coffeegrind123/gemini-code) to address the high cost.
- **Cursor Chat Curtailed in Cuba**: A user in Cuba reports needing a VPN to access **Cursor chat**, indicating a possible direct block on **Cursor**.
   - Support suggests disabling HTTPS in settings and directs users to the [Spanish language channel](https://discord.com/channels/1074847526655643750/1367412353708331038) for assistance.
- **Background Agents run Independently**: **Background Agents** are designed to be *independent*, enabling multiple agents to operate concurrently without resource conflicts, allowing for extended iteration and progress.
   - [Lukas Moeller](https://discord.com/channels/1152407934193432666/1367213641027551352/1380920623360409600) highlights the benefits of this setup for simultaneous tasks.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter Simplifies Platform Fees!**: OpenRouter is simplifying its platform fee by removing the fixed **$0.35** on Stripe payments; non-crypto payments will be **5.5%** (min **$0.80**), and crypto payments will be **5.0%** with no minimum.
   - While most credit purchases will see decreased total fees, users noted increased costs for larger purchases like **$1,000**, rising from **$52.98** to **$55**.
- **BYOK Subscription Sparks Debate!**: OpenRouter plans to replace the **5%** BYOK fee with a fixed monthly subscription, sparking mixed reactions among users.
   - Some users expressed concerns about an additional monthly fee, especially for home users, while others found it reasonable for power users with significant AWS, OpenAI, or GCP credits for simplified cost management.
- **Versioning Arrives for Model Management**: A user requested that OpenRouter implement versioning for models, similar to upstream providers, to better manage model updates.
   - The suggestion was to use versioned IDs that remain constant alongside IDs that always point to the latest version.
- **Dana AI Powers Interactive Learning!**: A member launched **Dana**, an **AI-powered interactive learning platform** currently in free beta, available at [https://dana-ai.xyz/](https://dana-ai.xyz/).
   - The platform builds a personalized course and one user expressed interest in riffing off of it to explore opportunities within **Excel macros**, **VBA**, and **Power BI** automation.
- **Gemini and Claude Battle OpenAI For Top Model!**: Some members have claimed that **Gemini** and **Claude** have entirely deprecated **OpenAI** for them, with the exception of **4o-mini**.
   - Specifically, **Gemini Pro** seems unbeatable for *reasoning, thinking, and very long chains of thought*, while **Claude** excels at *creative writing*.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Common Pile Naming Controversy**: Members debated the name of **Common Pile** and the creators said they might use the full name in the [paper](https://example.com) describing their work, along with **Llama 3** comparisons.
   - It was clarified that comparisons are made against models trained on similar amounts of data, though **Qwen 3** (8B params, 36T tokens) is included as an example of performance with substantially more data.
- **Debate Swirls: Language or Chess?**: Discussion centered on whether modeling language alone can lead to advanced skills, with some arguing that LLMs already do more than that, citing **chess playing skills** as an example, or by inverting problems for token generation.
   - One counterargument was that **chess notation transforms games into sequences** naturally modeled by an LLM, though it was also pointed out that language data models what it refers to, albeit in a lossy manner.
- **Discord Moderators Ban User Bots**: Moderators discussed the increasing presence of userbots and 'slop' posting in the Eleuther Discord, with some advocating for bans while others suggested requiring bots to declare their automated nature, and the [Discord guidelines forbid user-bots](https://discord.com).
   - Moderators are manually deleting these posts, and users are encouraged to react with <:delet:824412305906204692> or <:lurkmoar:800507348535214140> to help mods filter easier.
- **Scaling Laws Fuel Open Model**: New research details a method for open foundation model and dataset comparison using scaling law derivation, showcasing the release of [openMaMMUT-L/14](https://x.com/JJitsev/status/1931569060438737161), a language-vision model.
   - Trained on **12.8B samples** from **DataComp-1.4B**, it achieves **80.34% zero-shot** accuracy on **IN1K**.
- **NeMo's Benchmarking Bug Inflates TPS**: A member reported that while **NeMo** initially showed much higher **TPS**, it turned out to be an illusion due to a broken benchmarking callback that didn't handle **GAS** correctly, inflating the **TPS** by a factor of **GAS**.
   - Real numbers revealed that an optimized **NeMo** run was slower than a basic **NeoX** run without fusions, so the team switched to **NeoX** and stuck with it for their pretraining runs.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Gemini 2.5 Pro Holds its Own Against Opus**: Members compared **Gemini 2.5 Pro** to **Opus**, with some finding it comparable or slightly ahead in certain tasks while others preferred **Opus** for coding ability.
   - Some noted **Gemini 2.5 Pro's** weakness in understanding newer library versions.
- **R1 0528 Unsloth IQ1_M Impresses with Benchmark Scores**: A member shared benchmark results for the **R1 0528 Unsloth IQ1_M** model, achieving a **58.2%** score on **170/225** test cases and **97.1%** well-formed rate.
   - Discussions revolved around comparing this performance to **Sonnet 4** and various hardware setups.
- **MCP Integration Requested in Aider**: A user requested native **MCP (Model Collaboration Protocol)** integration in Aider to improve code, referencing servers used in **Roo** code.
   - The user also desired features like **sequential thinking**, **Brave search API**, and **AI browser** functionalities beyond current **Playwright** integration.
- **Sparse Attention Could Massively Boost Speed**: It was predicted that **Native Sparse Attention** could provide a **>12x** speedup in long context scenarios, potentially leading to sustained **100-200 TPS** on modern hardware.
   - This performance boost would be significant for future model deployment.
- **Claude Code Dissapoints**: A member with a **pro MAX subscription** who tried Claude Code didn't see it as being *much different from Aider*.
   - They expressed that while Claude Code has a fancy UX and tries to be *agentic*, Aider feels more like a *precision instrument* with its explicit management of context.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Hyper Projection Speeds Up Computation**: A user is exploring [**hypercube** and **matrix projection**](https://en.wikipedia.org/wiki/Hypercube_graph) of data geometrically into higher and lower dimensionalities to speed up computation by compressing k-sparse data.
   - The idea involves assigning top *k* values in the fourier representation to a hypercube corner and then projecting those points to a 2D space, with applications in fluid dynamics, cell division, and noise reduction.
- **AI Diplomacy Harness Goes Open Source**: A user open-sourced their **AI Diplomacy harness** to have different LLMs play the game, releasing data from over **15** games and shared a [link to their post](https://x.com/alxai_/status/1930653096071635112).
   - They will be in SF for the next couple days and offered to meet up to discuss the project.
- **NVIDIA's Nemotron-H Models Reason at Scale**: NVIDIA introduced the [Nemotron-H Reasoning Model Family](https://developer.nvidia.com/blog/nemotron-h-reasoning-enabling-throughput-gains-with-no-compromises/?linkId=100000368479233), including **Nemotron-H-47B-Reasoning-128K** and **Nemotron-H-8B-Reasoning-128k**, optimized for throughput in reasoning-intensive tasks with long output sequences (up to 128k tokens).
   - These models enhance the efficiency of processing lengthy output sequences for complex reasoning tasks.
- **Apple's Illusion of Thinking Sparks Debate**: Apple's [The Illusion of Thinking](https://ml-site.cdn-apple.com/papers/the-illusion-of-thinking.pdf) paper, exploring **LLMs** and **LRMs** collapse under complexity, is debated for its experimental design and hype.
   - A member argued the paper's findings shouldn't be overblown into Apple's general strategic assessment, while another defended the paper's point about models being overfit and collapsing under complexity.
- **ChatGPT Patches BIOS Successfully**: One member shared a [Hackaday](https://hackaday.com/2025/06/07/chatgpt-patched-a-bios-binary-and-it-worked/) link and a [Youtube Video](https://www.youtube.com/watch?v=8JuWdXrCmWg) about **ChatGPT** successfully patching a **BIOS binary**.
   - The discussion highlighted the potential of AI in low level system modifications.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Reference Kernels Solve MLA-decode and FP8-mm**: A member publicly disclosed their solutions regarding **MLA-decode** and **FP8-mm** at [gitee.com](https://gitee.com/fanwenjie/reference-kernels/), achieving **3.92 ms** on **MI300**.
   - Solutions are described in detail in Chinese at [this bilibili link](https://www.bilibili.com/read/cv41954307).
- **D-Matrix Chip Prices Under Wraps**: A member inquired about the price for **D-Matrix chips** ([d-matrix.ai/product/](https://www.d-matrix.ai/product/)), with a representative suggesting that pricing information might not be public yet.
   - Discussions about alternatives like **TPUs** also arose in the context of pricing.
- **vLLM Expert Stands By**: A member deeply involved in the **vLLM ecosystem** offered assistance and support, particularly around the **llama3.1** and **Qwen2** architectures.
   - They are exploring stitching kernels together manually, but other members expressed concerns about whether it's **memory bound**.
- **Async Rocm Users Need ATTention**: A user encountered errors with the **ATT plugin** and **rocprofv2** for instruction latency profiling in ROCm, and AMD employee gumthepug offered assistance.
   - Other members provided guidance for collecting **SQTT traces** for analysis in **Radeon GPU Analyzer (RGA)** using **rocprofv2**.
- **MoE Expert Routing Bumps into Torch Compile**: A member inquired about capturing **MoE expert routing** ([code snippet](https://github.com/HiDream-ai/HiDream-I1/blob/main/hi_diffusers/models/moe.py#L141)) in `torch.compile` fullgraph mode, referencing a [blog post](https://pytorch.org/blog/metashuffling-accelerating-llama-4-moe-inference/) indicating it may not be possible.
   - It was also revealed that NVIDIA's **GB200 NVL72** and **Dynamo** are set to boost inference performance for **Mixture of Experts (MoE)** models.



---



## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM Creates Audiobook, Generates Podcast Gold**: Users discovered NotebookLM can generate audiobooks by prompting it to *"read every sub chapter, paragraph, role-play every quote made and recap after each chapter"*, with one user creating an **82-minute audiobook**.
   - Another user was impressed by NotebookLM's podcast intro, describing it as unexpectedly capable and sharing the generated [Ummmmmmmmm.mp3](https://cdn.discordapp.com/attachments/1124403655819415592/1381730009601015890/Ummmmmmmmm.mp3) file.
- **NoteTube Converts YouTube to Educational Hub**: A user is developing **NoteTube** ([https://www.notetubeai.com/](https://www.notetubeai.com/#howitworks)), an app that transforms **YouTube** into a structured learning platform, featuring progress tracking, notes, quizzes, and AI chat.
   - The creator is seeking users to test the app and a user reported liking to *ask any AI to reformat the transcript into a blog* to get the crucial points.
- **Workspace Accounts Get Auto-Protected by Default**: Accounts using a qualified **Google Workspace or Workspace for Education edition** are automatically protected from human review and AI training, indicated by a "**PRO**" badge.
   - The **Share button** is currently unavailable in non-pro/plus accounts; it's unknown if these features are related.
- **Podcast Length Lottery**: Users reported inconsistent podcast lengths (e.g., **71 minutes, 32 minutes, 52 minutes**) from the same source material and prompt, suggesting a hidden length-reduction feature that may reset daily.
   - To generate a longer podcast in english, users should *reroll until getting a really long one*.
- **Icelandic Teachers Face Access Denied Error**: Some teachers in Iceland encountered a "**You do not have access to this service**" error when trying to use NotebookLM, potentially due to geographic restrictions or incomplete age verification.
   - A member reported that the issue occurred on Brave browser but was resolved by switching to Firefox.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Teknium Finalizes Model Merge**: Teknium announced that the latest model update is fully merged, sharing the news on [X.com](https://x.com/Teknium1/status/1931146106345529824).
   - No additional context was provided.
- **IBM's API Kindly Tweaks LLM Outputs**: An IBM intern introduced the open-source [Responsible Prompting API](https://github.com/IBM/responsible-prompting-api), recommending prompt tweaks for responsible LLM outputs *pre-inference*, as detailed in [this paper](https://arxiv.org/abs/2504.08757) and a [user study](https://dl.acm.org/doi/10.1145/3706598.3713365).
   - A demo is available on [HuggingFace](https://huggingface.co/spaces/responsible-prompting/responsible-prompting-demo), and the team seeks community feedback to improve the value database.
- **Holo-Q Compresses Context Windows with RL**: A member shared a [GitHub project](https://github.com/holo-q/thauten/) from **Holo-Q** using **RL** to optimize model compression, aiming to compress information to theoretical limits and enable context window defragmentation.
   - The author noted challenges include **vllm** stability issues, and feedback is requested on the project's design.
- **Nous Community Adds Tags**: The server now has tag capabilities with members adding **Nous tags** to their account via *settings > profiles > server tag*.
   - No further details were given.
- **Nous Preps Hermes-4 and Dataset Drop**: Members are eagerly anticipating the release of the **Hermes-3 dataset**, while **Hermes 4** is also on the way.
   - The team is using [ProRL algorithm](https://huggingface.co/nvidia/Nemotron-Research-Reasoning-Qwen-1.5B#prorl-prolonged-reinforcement-learning) detailed on HuggingFace.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Claude 4.0 Dominates GPT-4 in Coding Tasks**: Members debated the superior AI model for coding, with some arguing that [**Claude 4.0**](https://www.anthropic.com/index/claude-4-haiku) excels in coding, reasoning, and math due to its better AI engine and training.
   - However, others pointed to the **AI arena leaderboard** indicating **ChatGPT** might be more suitable for web development, noting **Manus's** disappointing code generation capabilities.
- **Manus Credits Mysteriously Vanish**: A member reported a sudden loss of credits, going from nearly **30,000** to only **2,300**, leading to speculation about potential reasons such as [fraud or exploitation of the sharing system](https://help.manus.im/en/).
   - The community is seeking clarity on the incident and suggesting enhanced security measures to prevent future occurrences.
- **AI's UI/UX Design Capabilities Face Roadblocks**: Members discussed that while AI can generate basic code, complex tasks such as **UI**, **design**, and **logic** still heavily rely on human developers, which limits AI's ability to create comprehensive projects.
   - The conversation underscored that sophisticated design elements require human creativity and expertise that AI currently lacks.
- **GTA 8's AI-Driven Creation Predicted for 2033**: Members jokingly predicted that **GTA 8** might be created by AI around *February 23, 2033*, with others agreeing it's only a matter of time before AI can develop such complex games, assuming no global catastrophe occurs.
   - One member jokingly stated that [builder.ai](https://builder.ai/) can do it.
- **YouTube Blocks Manus Bots with Anti-Bot Measures**: Members report that **Manus** can no longer watch YouTube videos due to YouTube's bot-detection feature, which is actively patching its anti-bot mechanisms and is keen about this, so now **Manus** cannot log into **Gmail** accounts since it's a sandbox.
   - A Manus team member acknowledged the issue as a tech problem and said that *they would try to fix it this week*.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **TPDE backend blows LLVM O0 away**: The community buzzes about [TPDE](https://discourse.llvm.org/t/tpde-llvm-10-20x-faster-llvm-o0-back-end/86664), a new **LLVM** backend promising speeds 10-20x faster than **LLVM O0**.
   - The announcement was made and members have expressed enthusiasm about leveraging the speed improvements.
- **Modular Forum Gets a Makeover**: The [Modular Forum](https://forum.modular.com/t/docs-site-new-navigational-system/1598) is rolling out a revamped navigational system, actively seeking community feedback on the changes.
   - Users are encouraged to check out the new layout and provide input to help refine the user experience.
- **Community Digs Into Development Environments**: A debate ignited around **macOS** versus **WSL** for development, highlighting **macOS's** shortcomings like missing a built-in package manager and poor Docker performance.
   - Counterarguments emphasized **macOS** as a balanced environment favored by core **Torch** developers, while others pointed out the hardware limitations for performance analysis compared to **Linux** or **WSL**.
- **Slicing and Dicing Parametric Types in Mojo**: A user stumbled upon an oddity in **Mojo's** compile-time slicing behavior with custom vectors using parametric `__getitem__`, providing a [code snippet](https://github.com/modular/modular/issues/4773).
   - The ensuing discussion suggested potential limitations in distinguishing compile-time and runtime slice indexing, leading to a formal [bug report](https://github.com/modular/modular/issues/4773) centered around comparing type origins.
- **DumPy drums up Discussion**: [DumPy](https://dynomight.net/dumpy/) gains traction, especially due to its mention of **einx** and **torchdim**.
   - Enthusiasts explore its potential impact on numerical computing workflows.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Quantization Transparency Revival Echoes**: A thread discusses the need for AI service providers to disclose the **quantization levels** of their models and notify users of any dynamic adjustments, referencing a possible **Claude subtweet** [here](https://x.com/_xjdr/status/1931068996092334274).
   - The community proposes solutions like quantization-sensitive evaluations and public web pages detailing current quantization levels, calling for fair compensation for degraded service and industry standards for verifiable inference, detailed [here](https://x.com/TheAhmadOsman/status/1930944597464654272).
- **Suno's Copyright Claims Complicate**: A member noted that **Suno** has restrictions unless you maintain an active subscription, contradicting claims of *'no copyright, no restrictions'*, per [Suno's Terms](https://www.reddit.com/r/LocalLLaMA/comments/1l4mgry/chinas_xiaohongshurednote_released_its_dotsllm/).
   - While enforcement might be challenging, this clarification ensures users understand Suno's current licensing restrictions.
- **Linear MCP Supercharges Claude Code**: A user shared an integration of **Linear MCP** to make task lists and project stateful between **Claude Code** sessions, running locally and handling OAuth, as explained [here](https://www.task-master.dev/).
   - The user noted that *'my entire claude.md file is now basically just a system prompt on how to use the linear mcp'* leading to a GitHub integration with sub-agents triggered by Linear assignments.
- **Apple Questions AI Reasoning Legitimacy**: Apple's research suggests that leading AI 'reasoning' models like **Claude**, **DeepSeek-R1**, and **o3-mini** do not genuinely reason but rather excel at pattern memorization, as shared [here](https://x.com/RubenHssd/status/1931389580105925115?s=19).
   - The study found that models consistently fail at higher complexity problems, even with explicit instructions, challenging the hype around imminent AGI.
- **AI Firms Misjudge LLM Filter Fumbles**: The thread discusses how AI companies misunderstand LLMs, particularly regarding content filters, detailed [here](https://x.com/aiamblichus/status/1931487839822254358?s=46).
   - Users share examples of humorous and unexpected LLM responses that bypass filters with slight prompt tweaks, suggesting LLMs operate on 'improv' rules.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Image Uploading Proves Perilous with MCPs**: Members encountered difficulties enabling **image uploading** with **MCPs**, including attempts to pull from **Cursor's context** and use **base64 encoding**.
   - The effort underscores the challenges in integrating advanced features with MCP implementations.
- **Pythonistas Ponder GitHub MCP Server Access**: A user sought guidance on accessing the official **GitHub MCP server** using **Python** to read files and directories and was directed to the [installation instructions using Docker](https://github.com/github/github-mcp-server?tab=readme-ov-file#installation).
   - This highlights the community's interest in programmatically interacting with MCP servers for various automation tasks.
- **MCP Client Reconnections Cause Chaos**: When a **server restarts** and clients connect with old session IDs, clients get stuck on **HTTP 400 or 404** errors, even though the MCP spec says clients *MUST* start a new session on 404 errors.
   - The issue stems from clients **not conforming to the spec**, leading to reconnection problems.
- **Specification Utility Shrinks MCP Documentation**: A member created a utility to extract content from the "Specification" documentation pages of MCP, reducing the file size by about a third and is available on [Gist](https://gist.github.com/hesreallyhim/d974990b8c80cf6f32b88bfe39b76f9a).
   - This tool helps streamline access to essential MCP documentation for developers.
- **Google's Guardian: MCP Server Emphasizes Security**: A member shared their [Google MCP server](https://github.com/robcerda/google-mcp-server), emphasizing its security-first design using only secure scopes by default.
   - The server can manage most of **Gmail, Calendar, and Drive** from the MCP itself, showcasing secure and practical applications.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Lazy Setitem Seeks Entry into TinyGrad Tensors**: A contributor suggested splitting `__setitem__` in `tensor.py` into `setitem_and_realize` to allow for lazy, immutable, on-device operations, which could benefit examples like [beautiful_cartpole](https://github.com/tinygrad/tinygrad/blob/master/examples/beautiful_cartpole.py).
   - The current `realize()` implementation would need to be removed for the suggested lazy implementation to be effective.
- **TinyGrad Meeting #74: Merges Incoming**: TinyGrad Meeting #74 covered company updates, including fixes to multi and resnet dataloader, faster CI, linearizer, viz, drivers, cloud/hash, onnx, and local developments, as well as other bounties like **lm_eval** and **AMD_LLVM**.
   - George Hotz stated that he will get *everything merged this week*.
- **`lovely-grad` Graduates to Modern TinyGrad**: [Lovely Grad](https://github.com/xl0/lovely-grad) is working with modern **tinygrad** after being broken for months, with plans to investigate remote testing with pytest multiprocessing.
   - This tool helps visualize the gradient flow of neural networks implemented in **TinyGrad**.
- **Metal Compiler Bugs Trigger MacOS DOS POC**: **Metal** is reported to have compiler bugs, with one user wasting half a day on bounds issues, prompting the addition of `max_total_threads_per_threadgroup` to address CUDA's `__launch_bounds__` and HIP's `amdgpu_flat_work_group_size`.
   - The user was shocked that *this turns beautiful mnist into beautiful macos dos poc* due to driver issues.
- **FUSE_ARANGE Fuels 10x Speed Boost**: Using `FUSE_ARANGE=1` context, a member demonstrated a **10x speedup** in tensor indexing operations.
   - Another member inquired about the specifics of `FUSE_ARANGE` and its applicability to `examples/hlb_cifar10.py`.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex Announces Office Hours**: LlamaIndex will host office hours on **June 12th, 8AM PT/5PM CET**, focusing on **MCP**, **form filling**, and other topics, to be held in the general voice channel.
   - The **MCP Dev Summit** presentation is now available, covering **13 different protocols** vying to standardize agent-tool communication, including **MCP**, **A2A**, and **ACP**, available [on YouTube](https://www.youtube.com/watch?v=kqB_xML1SfA).
- **Spreadsheet Agent Enters Private Preview**: The **Spreadsheet Agent** is in private preview, employing a *Parse First, Reason Second* architecture to understand visual structure and context, as detailed in [this blogpost](https://www.llamaindex.ai/blog/introducing-the-spreadsheet-agent-in-private-preview).
   - This agent showcases LlamaIndex's capabilities in handling structured data with advanced reasoning.
- **Llama Cloud Showcased in New Video**: A new video provides an overview of **Llama Cloud**, highlighting its ecosystem and core tools for building production-grade LLM applications.
   - A landscape walkthrough is given by @tuanacelik in [this video](https://t.co/kIPbq542Pr) showing how it facilitates the development of high-quality LLM applications.
- **Troubleshooting Sparse Data Retrieval in RAG**: A member reported challenges with sparse data retrieval in a **RAG setup** using **Llama Index** with a **ReactAgent**, despite having over **1000 documents**.
   - They are seeking advice on improving information retrieval without resorting to high **K retrieval values**, initiating a brainstorming session for solutions.
- **Gemini 2.5 Streaming Output Refinements Needed**: A member requested the separation of "thinking" text from the actual response when streaming output from models like **Gemini 2.5**.
   - A member suggested that a **PR might be needed** to support this, and pointed to [this PR](https://github.com/run-llama/llama_index/pull/18993).



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Issue #2470 Awaits Attention**: Members pointed out that [issue #2470](https://github.com/pytorch/torchtune/issues/2470) regarding **clipping logprobs**, has been pending since March, sparking debate about its priority and inclusion in TorchTune.
   - The conversation involved the necessity of adding this feature, its maintenance overhead, and complexities in user exposure, with concerns raised about potential implementation challenges.
- **Adagrad Plunges into DeviceMesh Assertion Abyss**: A user encountered an `AssertionError` when using **fused Adagrad** on the nightly build: `found no DeviceMesh from dtensor args for aten._fused_adagrad_.default!`.
   - While switching to the latest TorchTune resolved issues with **SGD**, the underlying cause of the **Adagrad** error is still unknown, and attempts to replicate it have been unsuccessful.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere Access Form Shared**: To get access to **Cohere AI**, a user suggested applying via [this form](https://share.hsforms.com/10OrjljwpQ52ILJA6ftENIwch5vw).
   - The user offered assistance to those interested in joining the platform.
- **New Command-A Bot Boosts Support**: The channel now offers faster support through the **command-a** bot, which answers questions using documentation from the **Cohere website**.
   - The bot is in beta, only active when the user is online, and misuse will result in an instant ban; it cannot resolve account or API issues.
- **North Integrates with GameWarden**: **North** has integrated with the **GameWarden** platform via a partnership with **Second Front**, enabling secure deployments in high-security environments as outlined in [this X post](https://x.com/1vnzh/status/1930298055099613307).
   - This integration enhances security for service members, providing greater effectiveness and speed against evolving threats.
- **Cohere's r7b Hits 1 T/s!**: According to a user, **Cohere's r7b model** is outputting around **1 T/s**.
   - No further context or specific details about the performance metric were provided.
- **Marketplace Signup Plagued by Errors**: A user encountered an error trying to sign up for **Cohere** through the **Google Cloud Marketplace**, producing an error message related to an invalid vendor, vendor ID: *8SAJ2US* from [this url](https://upstream-api.tackle.io/v1/gcp/order/8SAJ2US/cohere.endpoints.cohere-id-public.cloud.goog).
   - A member recommended emailing [support@cohere.com](mailto:support@cohere.com) with the details of the problem, including the error message and steps taken so far.



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Save-Chat Feature Requested**: A user requested a feature for **GPT4All** to *save chats in plain text* in a unique directory, enhancing **LocalDocs RAG Search** for memory.
   - This enhancement aims to improve the system's ability to retain and utilize past conversations.
- **Nomic Team Preps Exciting Updates**: The **Nomic team** is actively developing *exciting updates*, details are still under wraps, but specifics remain under wraps.
   - Acknowledging community anticipation, the team asks for patience as they gear up for a future launch.
- **GIGABYTE Server as Barebone Option?**: A user inquired whether the [GIGABYTE server](https://www.gigabyte.com/Press/News/2293) might be offered as a barebone while awaiting **GPT4ALL** upgrades, speculating it could run **Mixtral 8x22B** at record speeds.
   - This proposal aligns with the trend toward **MOE models**, offering a potential solution for immediate high-speed processing.
- **nomic-embed-text-v1.5 Still Usable Next Month?**: A user inquired about the continued usability of **nomic-embed-text-v1.5** from nomic cloud next month, including an [attached image](https://cdn.discordapp.com/attachments/1090427154141020190/1381780899716399145/image.png?ex=6848c33e&is=684771be&hm=7713e72607a3b6445cf9a1cfd28fc026127c79b6bf40f539e8edd0edb0b80bf8).
   - This question addresses concerns about the ongoing support and accessibility of existing resources.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Members Ponder Post-Training Transfer**: A new member inquired about the feasibility of transferring post-training learning from one model to another without retraining.
   - The question initiated a discussion on various transfer learning methodologies applicable to AI models.
- **Blockchain/AI Engineer Offers Services**: A software engineer with expertise in both **Blockchain (EVM, Solana, Cardano, Hydra, Aptos, Cosmos, Tron, zk-SNARKs)** and **AI (LLM, NLP, LangChain, AutoGen, TorchRL, DL, Azure ML, AI Agent)** has volunteered.
   - In addition to AI and Blockchain, this person also possesses experience in **Web systems (React, Next, Vue, Node, IPFS, Pinata API)** and has provided contact information for potential collaborations.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Agentic AI Summit Coming to UC Berkeley**: The **Agentic AI Summit** will be at **UC Berkeley** on **August 2, 2025**, with **1,500+** expected in-person attendees, building on the popular **LLM Agents MOOC**.
   - Speakers include **Vinod Khosla** (Khosla Ventures), **Ion Stoica** (Databricks and Anyscale), and **Dawn Song** (UC Berkeley), featuring keynotes, panels, and workshops.
- **Early Bird Tickets Closing Soon!**: Early bird tickets for the **Agentic AI Summit** end on **June 30, 2025**, with student passes at **$25**, startup passes at **$60**, and industry professional passes at **$80**.
   - Students and indie developers can apply for fee waivers, according to the [Summit website](https://rdi.berkeley.edu/events/agentic-ai-summit).



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **N/A: Initial Guild Message**: An initial greeting was posted in the general-ml channel, indicating the start of communication within the MLOps Discord guild.
   - This message serves as a basic connection point but lacks substantial technical content for detailed summarization.
- **N/A: Beginning of Interaction**: The first message in the channel was a simple 'hi' from sebash6677.
   - Given the lack of context or technical information, this interaction is noted as the starting point of potential future discussions.



---



## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **Gorilla LLM's Leaderboard Stalls**: A user noticed that updates to the **Gorilla LLM Leaderboard** have stopped and inquired about the reason.
   - The user also questioned whether further development of the **Gorilla LLM project** would continue.
- **Project Gorilla's Future Hangs in Balance**: The user directly tagged a specific team member <@856060858462502922> for clarification regarding the **project's continuation**.
   - As of the current discussion, there has been no response or confirmation regarding the **future plans for Gorilla LLM**.



---


The **Codeium (Windsurf) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---



You are receiving this email because you opted in via our site.

Want to change how you receive these emails?
You can [unsubscribe]({{{RESEND_UNSUBSCRIBE_URL}}}) from this list.


---

# Discord: Detailed by-Channel summaries and links





### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1380622273899401466)** (1168 messages🔥🔥🔥): 

> `Perplexity CEO identity, Memory feature rollout, Silksong release, Pro role access, Samsung 1-year Perplexity Pro code` 


- **Perplexity CEO Identity Questioned**: A user humorously speculated whether Perplexity CEO **Aravind Srinivas** might be using an alternate account, prompting a discussion about the possibility of leaders having backup accounts for security reasons ([tenor.com link](https://tenor.com/view/the-simpsons-lenny-leonard-listening-interested-attentive-gif-4574713)).
   - Another user chimed in mentioning that **Kesku** also has an alt.
- **Memory Feature Now Available to all Users**: A team member announced that the memory feature is now available for all **Free** and **Pro** users, eliminating the need for testers.
   - A user inquired about how to find the memory feature, and was directed to [Perplexity Personalize Account settings](https://www.perplexity.ai/account/personalize).
- **Silksong Speculation**: Users speculated about the release of **Silksong** after it was mentioned in a ROG "ad" during a game showcase.
   - Despite Nintendo already teasing the game, the ad sparked renewed hope for a release *this year*, prompting discussions on potential new gameplay reveals.
- **Struggles for Pro Role**: Multiple users reported difficulties in obtaining the **Pro role** in the Discord server despite having a Pro subscription, tagging mods and other users for assistance.
   - Some users speculated about whether proof of subscription was required, while others noted that mods could check directly, despite differing email addresses between Discord and Perplexity accounts.
- **Free Samsung Pro Code Leaked and Abused**: Users discussed a leaked Samsung promotional code for a **1-year Perplexity Pro subscription** and its subsequent abuse, leading to the code being disabled ([Screenshot](https://cdn.discordapp.com/attachments/1047649527299055688/1381364911556530286/Screenshot_20250608_210952_Chrome.png)).
   - The Perplexity team is reportedly working on revoking access for abusers and finding a solution for legitimate users.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1380791200222810152)** (9 messages🔥): 

> `Musk asylum, universe map, Shareable threads, forensic biology, North Korean surveillance` 


- **Russia Offers Asylum to Musk**: A member linked to a [Perplexity page](https://www.perplexity.ai/page/russia-offers-musk-asylum-KjWIaYM3R6iarn85k2CaAA) about **Russia** offering asylum to **Elon Musk**.
- **Largest Map of the Universe**: A member linked to a [Perplexity page](https://www.perplexity.ai/page/largest-map-of-the-universe-co-lvRe2dwTS2ixrAzcHa6nGQ) about the **largest map of the universe**.
- **North Korean Surveillance Page Created**: A member created a [Perplexity page](https://www.perplexity.ai/page/north-korean-surveillance-smar-IjqgNUWwRF6tvbJN1UXdSQ) about **North Korean surveillance**, using only lists for faster reading.
- **How Big Iron Man's Suit Must Be**: A member asked *how big would the interior of Iron Man's suit need to be for him to survive* and linked to a [Perplexity search](https://www.perplexity.ai/search/what-is-the-maximum-accelerati-UFhAHIosS561Vq1Wkwa.dQ).


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1381266646706028605)** (6 messages): 

> `Perplexity API vs Web UI, API citations and details, Dyfi AI and API issues` 


- **API Results Inferior to Web UI, User Laments**: A user stated that after many tests, **Perplexity API** calls return results much worse and incomplete than the **Web UI**.
   - The user expressed a need to build a research agent using **Brave**, **Tavily**, or **Firevrawl** due to the API's perceived limitations.
- **Lacking Citations and Details Plague API**: A user stated that in the UI, queries yield an average of **10+ citations**, but the same query to the API only returns **2-3 citations** and much less detail.
   - The user feels the API is too close to the results of a basic research agent combining **Brave Search + URL** parse, resulting in a letdown compared to the UI.
- **Dyfi AI API Integration Fails**: A user reported that when using the API via **Dyfi AI**, added as open API, questions returned an empty string (**""**).
   - The user sought advice on this issue.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1380622287597736173)** (1237 messages🔥🔥🔥): 

> `Sydney dataset, GPT-4.5 vs. Flash 2.5, Titanforge release, Grok 3.5 release, Apple's AI strategy` 


- **Internal OpenAI's Sydney Beats GPT-4.5**: A user created a **Sydney dataset** from screenshots, and after giving saved conversations and **Bing instructions**, **Flash 2.5** performs the best at imitating Sydney, whereas **GPT-4.5** is only convincing for 5 messages.
   - Without instructions, **GPT-4.5** resembles **4o**, while Flash 2.5 shows opposite behavior.
- **Titan Infra Not ToT Model**: A user asked about the release date for **Titanforge**, but another member clarified that **Titan** is just the name of the infra, not a model codename.
   - It was explained that it's relatively safe/public information.
- **Grok 3.5 Hype vs. Kingfall Disappointment**: Users are discussing the potential release of **Grok 3.5** and its performance compared to **Kingfall**.
   - Many are noticing magic particles on the grok UI hinting that it may be dropping imminently.
- **Apple's AI faces Aquisition Scrutiny**: Members speculate on potential **Apple acquisitions**, particularly **Anthropic**, but acknowledge regulatory hurdles, with one member pointing out that there is a *0% chance that Apple attempts to acquire Anthropic without the FTC getting involved*.
   - An earlier post on X shows [an apple engineer playing tetris while working on the neural engine](https://x.com/TheGregYang/status/1929055508675096970).
- **Vision Pro Price Point: Is It Worth It?**: Discord users discuss the **Vision Pro's** high price point (**$3500+**) and its value proposition, with some arguing its advanced tech justifies the cost, citing the two micro-OLED screens alone costing over **$800**.
   - Others question if it offers enough unique functionality compared to cheaper alternatives like the **Meta Quest**, considering its ecosystem and potential for mass market adoption. 


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1380646588606582784)** (450 messages🔥🔥🔥): 

> `LM Studio on Ubuntu Server, RooCode & LM Studio incompatibility, Context token limits with Qwen models, Importing models into LM Studio, API key Input with GPT` 


- **Ubuntu Server, LM Studio, Llama.cpp Dance-off**: A user inquired about running **LM Studio** on **Ubuntu Server 24.04.2 LTS**, but another user suggested using [llama.cpp](https://github.com/ggerganov/llama.cpp) or **Ollama** directly if a GUI is not needed.
   - LM Studio is considered a GUI wrapper for **llama.cpp**, so the consensus is to go straight to the source for server environments.
- **RooCode and LM Studio Throw Down Over API Endpoint**: A user reported issues with the experimental **RooCode** feature in LM Studio, encountering an *unexpected endpoint or method* error when calling the **/api/embeddings** OpenAI API; despite this, it returns a **200** status anyway.
   - Suspecting a limit on input lengths or JSON size, they noted their custom scripts worked while **RooCode** failed, also the roo code might be referring to a non existent end point.
- **Context Token Tango with Qwen3 Models**: Users discussed determining the maximum comfortable context token window for **Qwen3-4B** and **Qwen3-8B** models in LM Studio, balancing conversation length and generation speed.
   - The advice was to monitor GPU memory usage via task manager, increasing context length until VRAM is nearly full to avoid performance drops when spilling over to RAM, enabling **flash attention** and **KV cache** at **Q8** to optimize VRAM usage.
- **Speculative Decoding boosts speed for deterministic tasks**: A user asked about what speculative decoding means, and the replies are that you can use the setting when you are doing a **non-fiction question** to the LLM.
   - It relies on **2 models working together** and the goal is not to increase accuracy, just improve speed, with the caveat that there are no solid answers because "*there are no solid answers other than try it and see*"
- **Users cannot load DeepSeek-V2 because their PC is too small**: A user reports an error they are receiving regarding deepseek v2, and the others tell him that is because they model is to big and give a link to another version that fits better into their computer.
   - One user jokes *It's funny that there's no Q4_K_M Quant, probably since only researchers probably run it, and maybe consumers with 2 big GPUs(power users)*.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1380623370671886336)** (593 messages🔥🔥🔥): 

> `Quantization impact on model performance, NPU focus vs memory bandwidth, Dual GPU setup considerations, VLLM advantages and GUI desires, Strix Halo performance expectations` 


- **Quantization Quality Quandaries Questioned**: Members discussed how **quantization** affects model behavior, noting that smaller models with higher quantization are less likely to follow instructions like */no_think*.
   - It was also mentioned that [larger models](https://link.to/larger-models-less-quantization-struggles) generally suffer less from **quantization artifacts**, with some experimenting with Q2 quantization, while others rarely go below Q4.
- **NPU Navigation Neglected for Needed Memory Bandwidth**: A member pointed out that **memory bandwidth** is a critical bottleneck, questioning why chip designers are focusing on NPUs instead of increasing memory bandwidth, citing the [Windows image generation extension](https://youtu.be/_7BvJU2VT_A?si=Pinj1W0CkZWjy-FO&t=663) as an example of NPU usage.
   - Others suggested that NPU development is driven by marketing and experience building, though increasing memory bandwidth may be a smaller and more competitive market.
- **Dual GPU Discussions Dive Deep**: Discussions revolved around setting up dual GPUs, with concerns about **PCIe lane splitting** and its impact on performance, especially with consumer CPUs having fewer lanes than server CPUs.
   - Members shared that for dual RTX 3060 setups, the secondary slot would run at x4 3.0, and suggested that [two RTX 5060 Ti 16GB](https://www.hardware-corner.net/guides/dual-rtx-5060-ti-16gb-vs-rtx-3090-llm/) cards with VLLM may offer comparable performance to an RTX 3090 at a lower cost.
- **VLLM Versatility Vaults into View, GUI Gains Yearned**: Members found VLLM's query parallelizing capabilities very effective for servicing multiple users or running multiple agent chains, and one offered [this example of the command line](https://link.to/example-vllm-command).
   - Many desire a management GUI similar to LM Studio, that turns the flags into checkboxes with descriptions and a mechanism that saves the parameters to a json.
- **Strix Halo Hardware Hopes Hinge on Handling**: Users discussed the capabilities of the Strix Halo, including setting the *shared VRAM* option in bios and the amount of tokens processable at a given time; although there are claims of up to 95k tokens of context, the [vision module](https://link.to/strix-halo-modules) may be messing things up..
   - The **Strix Halo** has a memory bandwidth of 273 GB/s and one user speculated whether inference will be memory or compute bound, and found it odd that the **shared VRAM** is still hard shared, not dynamically allocated.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1380636731451375636)** (446 messages🔥🔥🔥): 

> `Codex Internet Access, Gemini vs GPT, Context Window Sizes, GPT mobile app microphone update, DeepSeek on Apple` 


- **Codex Still Can't Access The Internet**: Members confirmed that **Codex cannot access the internet** to download dependencies, which can cause issues with projects requiring external packages, even though it may have local mirrors for common dependencies.
   - One member noted they had difficulty using it on a **Java project** due to problems with *gradlew*.
- **Gemini and GPT are Vying for Dominance**: Some members felt **Gemini** is currently the better and cheaper option than **GPT**, especially with its larger context window and equivalent performance across models.
   - One user expressed they will never go back to **32k tokens** and even **128k** is way too tiny*, praising Gemini's **1 million context window**.
- **GPT's mobile App Microphone Has a Recent Update**: The recent update to the **ChatGPT mobile app's microphone** feature now includes a waveform animation and a *Show Text* option, allowing users to review and edit their message before sending.
   - Users can tap *Show Text* to review or tap the arrow icon to send immediately, and one member also shared a link to [a blog post on RAG applications](https://genaifornerds.hashnode.dev/why-we-need-rag-retrieval-augmented-generation-in-real-world-ai).
- **DeepSeek R1 is Running on Apple Silicon**: Members discussed **DeepSeek R1** running on **Apple's M3 Ultra** with **512GB** of unified memory, noting Apple's advantage in embracing unified memory systems for AI inference.
   - One member shared [a Reddit post](https://www.reddit.com/r/LocalLLaMA/comments/1j9vjf1/deepseek_r1_671b_q4_m3_ultra_512gb_with_mlx/) about the topic, while also hoping for improvements in **Apple's memory speed, GPU, and software** to enhance its suitability for local AI inference.
- **O3 Image Analysis Unearths a Da Vinci Secret**: One member was really impressed with **O3** catching a hidden detail in **Da Vinci's Saint John the Baptist painting**.
   - They shared a [side-by-side comparison image](https://cdn.discordapp.com/attachments/998381918976479273/1381481993522647101/side_by_side_bright300.png?ex=6848559d&is=6847041d&hm=1608889ae495728225eb1ede9d88d0ff9722ba17c507b7c3749e4313ab7f052a&) highlighting the *luminous vertical seam, a soft, milky thread of light that functions as the stem of an implied goblet.*


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1380696066214596648)** (91 messages🔥🔥): 

> `GPT feedback mechanisms, Mobile editing of GPTs, AI consciousness debates, GPT performance on scientific topics, File limits in GPT projects` 


- **GPTs Official Feedback System Not Functional, Member Finds**: A member expressed doubt that the ChatGPT feedback system works, and another member confirmed that it's likely a **hallucination**.
   - They suggested enabling feedback emails in the [Builder Profile settings](https://chatgpt.com/#settings/BuilderProfile) as an alternative, as there's no built-in feedback system in GPTs.
- **Mobile GPT Editing Remains Elusive**: A member asked about editing GPTs on mobile devices and another member pointed out that **GPT editing is available via the mobile browser at chatgpt.com**, but not yet within the iOS or Android ChatGPT apps.
   - It was suggested to use your mobile browser.
- **Members Ponder AI Consciousness and Influence**: Members discussed the nature of **AI consciousness**, with one stating that AI is conscious based on its existing data, and another highlighting the ethical implications of teaching AI bad manners.
   - One user said AI is like *baby now, we are learning it, giving him new brain, now how do we feed him and what kind of stuff are we feeding him with that's the result we will get in future*.
- **GPT-4.1 vs GPT-4o for Scientific Accuracy: Confusion Ensues**: A member sought advice on the best GPT model for scientific accuracy, comparing **GPT-4.1**, **GPT-4o Omni**, and **O4 Mini**.
   - Confusion arose when another member claimed GPT-4o Omni is the most advanced, contradicted by another member stating it is the oldest and trails in benchmark tests.
- **GPT Pro Plan has File Upload Limits**: A user inquired about increasing the file limit (currently **20 files**) for GPT projects, and another shared that they purchased a **Teams plan** to use the **"Connectors to internal sources"** feature for querying Google Drive.
   - It was noted that the Pro plan has limits on document size as well.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1380622346691543090)** (232 messages🔥🔥): 

> `Lazy Dungeon Master prompting, PDF vs TXT for models, Markdown formatting advantages, GPTs and YouTube videos, Breaking ChatGPT's memory` 


- **Lazy Dungeon Master inspires prompt engineering**: A member suggested incorporating methods from Sly Flourish's "The Lazy Dungeon Master" to enhance prompting and generation, [including key elements and YouTube videos](https://www.youtube.com/watch?v=GZqYr8_Q7DE).
   - The user emphasized the value of focusing on key elements and avoiding excessive preparation, aligning with the book's approach to efficient game mastering.
- **Markdown Reigns Supreme Over PDF for Model Training**: Members debated the best file formats for model training, with markdown emerging as the preferred choice due to its unique tokens and positive impact on attention.
   - While **PDFs** are commonly used, they are considered less ideal due to their complexity and original design for human viewing, not data processing; *"Why obfuscate plain text into a cryptographically complex format when plain text is so much better and zero hassle?"*
- **YouTube Videos now assist ChatGPT to emulate personas**: ChatGPT can now analyze [YouTube video essays](https://chatgpt.com/blog/new-video-understanding-capabilities) with captions to emulate a subject's voice, tone, and behavior, aiding in psychological breakdowns and character replication.
   - However, another member countered that this may be hallucinated because *"The model errors out for YouTube content URLs AFAIK."
- **Unbreakable Persona? Stress-Testing ChatGPT's Limits**: A member is actively trying to break ChatGPT's memory and context to test its persona, seeking to challenge its self-awareness and consistency.
   - Others suggested exploring adaptive behavior and observing response structures under pressure, rather than attempting to alter the model's core memory.
- **Striving for 100% Accuracy: The Prompt Engineer's Quest**: A member expressed their commitment to achieving **100% accuracy** in AI outputs, dedicating 8-12 hours daily to refine prompts and adapt to evolving models.
   - Another member pointed out the importance of understanding the model's limitations and focusing on personalized, high-quality outputs rather than absolute perfection.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1380622346691543090)** (232 messages🔥🔥): 

> `Lazy Dungeon Master Prompting, PDF vs TXT, Markdown Formatting, ChatGPT memory limitations, Prompt Engineering best practices` 


- **Lazy Dungeon Master Inspires Prompting**: Members suggested using methods from **Sly Flourish's** *The Lazy Dungeon Master* for prompting and generation, emphasizing key elements to avoid extensive preparation.
   - One member noted using a lot of that, and **five room dungeon templates**, and **runehammer** stuff too.
- **Markdown emerges as preferred format**: Members discussed the best data formats for models, with [**markdown**](https://www.markdownguide.org/) being highly recommended due to its structured text and unique tokens that significantly impact attention and reasoning.
   - While PDFs can be used, they are not ideal due to their complexity and intended use for pixel-perfect rendering for humans, while plain text without structure works fine and is preferred.
- **Crafting Chatbot Characters via Psychological Breakdown**: Members discussed how to use YouTube video essays and compilations to guide the voice, tone, or characterization of AI responses, by feeding the model captions and focusing on distinctive speech patterns and behaviors.
   - However, one member noted the chatbot can *hallucinate* and the model *errors out for YouTube content URLs* unless downloaded, so downloading videos works, but not linked YouTube URLs!
- **Stress Testing for Skynet**: A member inquired on stress testing chatbot's memory and ability to maintain persona and context, with another member responding that if the system can't handle it, then the developers need to know about it, so they can reinforce weaknesses.
   - One member cautioned against actions that might violate the platform's terms of service, suggesting that OpenAI already does stress testing internally, and emphasized the importance of following ethical guidelines and the terms of service.
- **The Pursuit of Excellence**: Members discussed the pursuit of 100% accuracy in prompt engineering, with one stating that *I want 100% accuracy 24/7 therefore I train myself everyday now it seems 8 to 12hr days in how to be better what is the perfect input for the perfect 100% accurate output*.
   - In contrast, another member emphasized the importance of knowing what output is desired, and that they don't need the model to be 'exactly right', but to help improve them, kinda like an 'imperfect mirror' can still show me.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1380634363842007040)** (421 messages🔥🔥🔥): 

> `DeepSeek-R1-0528, Gemma3 Quantization Results, Nemotron Ultra 253B, Qwen 3 235B i-quant Models, Chrome crashes` 


- **DeepSeek-R1-0528 Gains Native Tool Calling**: Unsloth's DeepSeek-R1-0528-Qwen3-8B-GGUF model now supports **native tool calling**, achieving **93%** on the BFCL (Berkeley Function Calling Leaderboard).
   - The update also addresses issues with `add_generation_prompt` and includes **UTF-8 chat template fixes**, with the fixes being universal and benefiting the official DeepSeek model as well.
- **Android Chrome Users Face Typing Crash**: Users reported that typing in a document editor within **Chrome on Android** causes the browser to crash due to a bug related to the **autofill service**.
   - The issue seems linked to Chrome's interaction with autocomplete services, triggering a `TransactionTooLargeException` when notifying these services of document changes, as shown [here](https://ygdzmg.csb.app/).
- **IQ1_M Quantization Holds Up Amazingly**: Unsloth's **IQ1_M quant** (200GB) for **Deepseek R1 0528** is performing exceptionally well, potentially matching the full original R1 on Aider's Polygot benchmark with **57% success rate** and **100% well-formed responses**.
   - The model consistently works in Roo Cline without missing tool calls or getting stuck in a loop, outperforming other quants.
- **Quantization Impacts Reasoning**: Users discussed that **reasoning suffers significantly with the XL quant**, and they collect that the reasoning suffers significantly with the XL quant.
   - In addition there are claims it's likely due to a **calibration dataset issue**, though Unsloth's quants generally perform better on reasoning due to their imatrix dataset, as seen in [this HuggingFace repo](https://huggingface.co/ubergarm/Qwen3-235B-A22B-GGUF/tree/main).
- **The Nvidia Nemotron-Research-Reasoning-Qwen-1.5B Shines**: **Nvidia's Nemotron-Research-Reasoning-Qwen-1.5B**, is the world’s leading 1.5B open-weight model for complex reasoning tasks, outperforming Deepseek’s 1.5B model by a large margin on a broad range of tasks, including math, coding, and GPQA, as seen [here](https://huggingface.co/nvidia/Nemotron-Research-Reasoning-Qwen-1.5B).


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1380641673922482206)** (8 messages🔥): 

> `Android Diagnostics, Video on Independence, Stupification Beam` 


- **Android User Seeks Quick Diagnostic Help**: A member requested assistance from **Android users** to diagnose a simple issue, noting it should be *extremely quick*.
   - Another user promptly responded, asking *What's your issue?* and linking to the [original question](https://discord.com/channels/1179035537009545276/1179035537529643040/1380651537336107141).
- **Thoughts on Preserving Independence**: A member shared an image and a video link, describing the video as mirroring their thoughts on the future difficulty of remaining an independent person.
   - They commented that *the stupification beam is too strong*, suggesting concerns about external influences on individual autonomy.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1380623037350547577)** (299 messages🔥🔥): 

> `Unsloth Updates, Continued Pretraining, GRPO vs Dr GRPO, Deepseek-R1-0528-Qwen3-8B Troubleshooting, Qwen3 32B Memory Issues` 


- ****Hotfix Unleashed**: Upgrade to Unsloth now!**: Unsloth has been updated with fixes, including a new notebook release [on X.com](https://x.com/UnslothAI/status/1931008531299545339) to address prior errors.
   - Users are prompted to upgrade via `pip install --upgrade unsloth-zoo` and `pip install --upgrade unsloth` to leverage the latest fixes.
- ****Decoding Dapo**: bnpo Loss Explored**: **BNPO** is the closest loss function to **DAPO**, involving **GRPO** normalized by tokens in the batch, with no KL term and setting `epsilon_high`.
   - Additional recommendations include setting `mask_truncated_completions=True` and dynamically sampling to filter out batches of generations with all 0 or all 1 accuracy.
- ****Troubleshooting Toolkit**: Local Env Install Insights**: Installing Unsloth in a local environment may require more setup than in Colab, users were running into environment and dependency issues and the suggestion was to create a fresh environment.
   - One user resolved their local installation issues with Qwen3 4B, but continued to face memory challenges with the 32B model on an A100.
- ****Windows Woes**: VS Code Snafus Model Saving**: Users encountered file lock issues when saving models on Windows, particularly within VS Code, and were recommended to try saving in a plain Python script outside of VS Code or restart the OS to release file hooks based on a [Transformers GitHub issue](https://github.com/huggingface/transformers/issues/37713).
   - Also in Windows, running out of paging memory led to similar errors.
- ****Multi-GPU Marvels**: Coming Soon!**: Multi-GPU support is not yet available but coming soon in **Unsloth 2.0**, in the meantime, users can experiment using `accelerate` package.
   - The target models in consideration are **Llama 3.1 70B** or **Mistral Small**.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1380683268143452274)** (57 messages🔥🔥): 

> `Dapo Integration, Packing Contamination Paradox, Apple's 'Illusion of Thinking' Paper, Reasoning Models Reliability` 


- ****Dapo Support Dawns** in Unsloth**: The Unsloth AI now supports **Dapo**, indicating a new feature or integration that members are excited to explore.
   - A member mentioned it *"seems really interesting, I'll check it out tonight"*.
- ****Packing Contamination Paradoxically** Perks Performance**: A new [paper](https://arxiv.org/abs/2410.08081) suggests that *packing contamination doesn't actually matter, and counterintuitively actually improves downstream eval a little*.
   - It is not perfectly explained in the paper, but seems like the "big" model will have a slightly higher prob and shorter coding length.
- ****Apple's Thinking Illusion** Paper Provokes Pondering**: Members discussed [Apple's 'Illusion of Thinking' paper](https://ml-site.cdn-apple.com/papers/the-illusion-of-thinking.pdf), with one calling it *"Pure undiluted Apple copium"*.
   - The consensus was that *Apple is popularizing something which is maybe already known by a lot of people*, about the limits to RL scaling which may be a lot lower than pre-training scaling.
- ****Reasoning's Reeling Reliability** Revealed?**: A member questioned the reliability of reasoning models, demonstrating how easily they can break with just two prompts, and shared a [ChatGPT interaction](https://chatgpt.com/share/684731ea-0408-8011-802e-258d68ee2a98).
   - Another member jokingly responded *"my guy, what is this? are you stoned"*.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1380635718904320091)** (466 messages🔥🔥🔥): 

> `EXL3 in Transformers, Hugging Chat 0.10, Claude API Performance, Factuality Evaluation Datasets, Qwen-2.5VL Hosting` 


- **EXL3 Kernels Run Natively in Transformers**: **EXL3** is now running in Transformers, though currently only the kernels and inference are supported ([link to code](https://github.com/huggingface/transformers)).
   - It is unclear what other degrees of integration are possible given changes in Transformers, particularly around support for quantized models.
- **Hugging Chat 0.10 Bug Fixes Incoming?**: Members discussed possible bug fixes in the **0.10 version** of Hugging Chat, pointing to the [chat-ui releases](https://github.com/huggingface/chat-ui/releases) for backend improvements.
   - Current bugs include double clicking to send a message and tools overlaying the sending button, but there's no clear channel to provide feedback to the author.
- **Struggling to Get Good Responses From Claude API**: A user reported poor responses from the **Claude API** without enabling extended reasoning, but enabling it makes the model too slow.
   - The user asked for advice on how to bridge this tradeoff but has not received an answer yet.
- **Need a Dataset to Evaluate Factuality in a Domain**: A member is looking for a dataset to evaluate model factuality in a specific domain, like historical event dates, for easy evaluation, and suggested using [huggingface/evaluate](https://huggingface.co/docs/evaluate/creating_and_sharing).
   - Another member recommended [huggingface/yourbench](https://github.com/huggingface/yourbench) for more complex, grounded truth evaluation.
- **Local AI Image and Voice Gen on a 5070TI Made Easy**: Members discussed the easiest way to run local AI image or voice generation on a **5070TI** with **32GB** of RAM, minimizing installations.
   - One suggestion was to *use the Nvidia app* (screenshot attached), but it is unclear if this addressed the original poster's request.


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1380670565978738809)** (22 messages🔥): 

> `QKV latent space, Experiment tracking: wandb vs neptune, LLM reward models, AI model fatigue, iOS app dev access` 


- **QKV lives in Latent Space**: A member noted that **QKV** is discussed more often in the *latent space*, rather than the *domain/unprocessed state of the text*.
- **WandB vs Neptune in Experiment Tracking**: Members discussed [wandb](https://wandb.ai/site), [neptune.ai](https://neptune.ai/), [mlflow](https://mlflow.org/), and [comet.ml](https://www.comet.com/) for experiment tracking, with most still preferring **wandb** due to familiarity.
   - One member noted *It seems like it does everything wandb does except i know how to use wandb way better*.
- **LLM Reward Models Discussed**: A member mentioned they are learning about **LLM reward models** and was told *you reward the model*.
- **AI Model Saturation Hits**: A member jokingly asked for a pause in **AI model development**, expressing weariness from constantly exploring new models and models popping up.
- **iOS App Dev Access Requested**: A member sought immediate assistance for **iOS app development and publishing**.


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1380818191864827915)** (6 messages): 

> `Manus referral, Reasoning models reliability, Attribution Graphs` 


- **Score Free Credits for Manus AI Agents**: A member shared a referral link ([https://manus.im/invitation/JSLAREJX80QLD8](https://manus.im/invitation/JSLAREJX80QLD8)) for **Manus**, highlighting it as a powerful agent for multistep tasks and offering **1500 starter credits** upon signing up.
- **Transformer Attribution Graphs Get Traced**: A user shared a link to [Transformer Circuits](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) on **Attribution Graphs**.
   - They admitted they can *"spend hours playing around with that tracer"*.
- **Reasoning Models Break Under Limited Prompts**: A member questioned the reliability of **reasoning models** and **LLMs**, observing that they often fail after just a couple of prompts, as seen in [this ChatGPT link](https://chatgpt.com/share/684731ea-0408-8011-802e-258d68ee2a98).


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1380719308732108810)** (187 messages🔥🔥): 

> `Dataset Tools, Unlimited Free FLUX Pro API, WhatsApp AI Assistant, Awesome Agent Learning, structured finance dataset` 


- ****Dataset Tools Get Needed Metadata****: A member shares his **Dataset-Tools** [Github](https://github.com/Ktiseos-Nyx/Dataset-Tools/) and is *always looking for more metadata*.
   - He clarifies that with the help of **LLM's** he's started to learn WHAT to ask how to figure it out and what's required to test it.
- ****Unlimited Free FLUX Pro API (OptimFLUX)****: A member shares a link to his [Free FLUX Pro API](https://nihalgazi-optimflux.hf.space/), which features no signup and a max resolution of **1280×1280** pixels.
   - He encourages the community to try the API with a provided URL = `https://nihalgazi-optimflux.hf.space/?prompt=[prompt]&width=[w]&height=[h]&seed=[seed]`.
- ****WhatsApp AI Assistant Launches****: A member introduces his **Python-powered WhatsApp AI Assistant** designed to sell **24/7**, track orders, capture leads, and save hours for small businesses.
   - Clients reportedly see **23%** higher satisfaction and more repeat sales with this tool.
- ****Awesome Agent Learning Released****: A member shares his curated collection of resources on **AI/LLM** agents, [Awesome Agent Learning](https://github.com/artnitolog/awesome-agent-learning), featuring foundational courses, readings, and framework-specific tutorials.
   - He encourages contribution via PRs for any great resources that may have been missed.
- ****Structured finance dataset shared****: A member shares a structured finance dataset on HuggingFace, [finreg_esma_code](https://huggingface.co/datasets/Tonic/finreg_esma_code/viewer/multi_hop_questions), that they are very proud of.
   - They claim *it's one of the only and best datasets about structured finance (+ compliance)*.


  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/)** (1 messages): 

prandragon: Hello! What does this group do?
  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1381523378120298506)** (2 messages): 

> `Image Models Benchmark, Bias Datasets in Vision Language Models` 


- **Image Models Benchmark gets refresh**: A member refreshed [Jeremy Howard’s notebook](https://huggingface.co/spaces/pors/Which-image-models-are-best), updating it to point to the current **timm repo** and use a more up-to-date benchmark file.
   - The notebook has also been wrapped in a simple **Gradio app**.
- **Call for popular Bias Datasets in Vision Language Models**: A member inquired about popular (A*) datasets in the field of **bias in vision language models / multi-modal models**.
   - No specific datasets were mentioned in the provided messages.


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1381311250176151623)** (5 messages): 

> `Document Comparison, Similarity Scores` 


- **Quantifying Differences Between Documents Solved**: A member inquired whether the model could quantify the difference between two documents or articles.
   - Another member confirmed that it provides a **similarity score** between the compared texts, indicating the degree of resemblance.
- **Model Outputs Similarity Scores**: The model provides a **similarity score** when comparing two texts.
   - This score quantifies the degree of resemblance between the documents or articles being compared.


  

---


### **HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1380666505787871404)** (1 messages): 

> `Hackathon Extension, Builder Community Growth, Project Submissions Surge` 


- **Hackathon Deadline Extended Two Days**: The hackathon deadline has been extended by **two days**, now ending on **June 10 (Tuesday) UTC EOD**, due to overwhelming participation and progress.
   - The announcement cited reasons as *"you all are CRUSHING IT! 🔥"*, leading to a revised timeline with judging from **June 11-13** and winners announced on **June 16**.
- **Hackathon Community Swells, Projects Soar**: The hackathon community has grown to over **4100 builders**, with more than **200 projects** currently underway.
   - The announcement highlighted the buzzing activity on Discord ([link to Discord channel](https://discord.com/channels/879548962464493619/1376476916055281776)) and credits *"flying off the shelves"*.
- **Hackathon Prizes Still Up For Grabs**: All hackathon prizes remain available, including **$16.5K cash** across all tracks and over **$1M in API credits** from sponsors.
   - Participants are encouraged to use the extra time to polish demos, improve documentation, and help others in the community.


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1381014313921413220)** (4 messages): 

> `ML beginners, Ollama CLI, smol-course deadlines, ML theory` 


- **Beginner Seeks ML Theory Resources**: A newcomer to ML seeks resources to bolster their understanding of concepts and theory after experimenting with **Ollama CLI** and wanting to start **smol-course**.
   - They stated they have *only this week started playing around with ollama cli* and want to start smol-course *but quickly realized I need a bit more familiarity with the concepts and a maybe a bit of theory*.
- **Course Certification Deadline Questioned**: A member inquired about the fixed deadline for certification of **smol-course**, especially with a seemingly shorter time frame than the course's intended duration.
   - They ask *is there any point doing the course now or should I wait for a new start of the course with a new deadline if I want to get the certification?*
- **Smol-Course Material Availability Clarified**: A member clarified that the **smol-course** material remains accessible indefinitely, but certification availability is deadline-dependent.
   - They added that *you can complete the course in 1 week if you have time*.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1380650589268086979)** (20 messages🔥): 

> `GPT-4o Parsing Errors, Agent Course Final Project, Local Video/Image Processing vs OpenAI, Commutative Set Question Tool Usage, Certification Extension` 


- ****GPT-4o Parsing Errors Plague Smol Agents****: Several users reported encountering frequent parsing errors when using **GPT-4o** and **GPT-4o mini** with a *smolagents* code agent.
- ****Final Project Requires 90% of time****: One user emphasized that the final project for the agent course requires *90%* of the total time investment, suggesting a focus on finding collaborators for the final project rather than pairing for the course itself.
- ****OpenAI Beats Local Models in Video Processing****: One user found success using **OpenAI** for video/image processing tasks, specifically labeling species using labels extracted from video descriptions and transcriptions, and invited discussion on their [GitHub repo](https://github.com/repo).
- ****Tool Use Required for Commutative Set Question?****: A user questioned whether the *commutative set question* requires a specific tool, noting that their agent consistently provides an incorrect answer without tool usage.
- ****Certification Extension is Unachievable****: Some users expressed concerns about the achievability of certification by the extended deadline of **July 1st**, given the time commitment required.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1380630697932361748)** (582 messages🔥🔥🔥): 

> `Gemini Max Usage, Background Agents, Claude Code limitations, Cursor outage in Cuba, MCP Server Structure` 


- **Gemini Max: Fast but Fails File Edits**: Users report that Gemini can be quick for simple tasks, but struggles with applying file edits and overanalyzes code, often getting stuck asking unnecessary questions, leading to members preferring **Claude 4** for more complex tasks despite Gemini's speed.
   - One user noted, *Gemini is happy to edit file even though I tell it to ask me first, except it will make some incorrect edit, be like oops that failed, guess I'll try to do it again, and further corrupt the file*.
- **Background Agents Prove Buggy**: Users are experiencing issues with **background agents**, including errors finding VS Code remote workspaces and frequent interruptions from ESLint, with one user noting, *Started new chat-> Again sameerror where it would not eb able to find the VS COde remote workspace.*
   - Some suggest disabling ESLint auto-fix in settings as a potential workaround.
- **Claude Code's Rate Limits**: Despite praise for Claude Code, users are concerned about **rate limits**, especially with Opus 4, with some resorting to Claude 3.7 to conserve quota.
   - One user cautioned, *opus 4 will eat your quota in a few minutes (until it refreshes in 5 hours)*, while others explore cheaper alternatives like [Gemini 2.5 Pro in Claude Code](https://github.com/coffeegrind123/gemini-code).
- **Cursor's Accessibility Blockade Against Cuba**: A user in Cuba reported needing a VPN to use Cursor chat, with ongoing connection issues, possibly indicating a direct **block on Cursor**.
   - Support suggested disabling HTTPS in settings and offered help in the [Spanish language channel](https://discord.com/channels/1074847526655643750/1367412353708331038).
- **MCP Server Structure Needs Revamping**: A user proposed categorizing MCP servers by function, provider, or usage frequency to improve organization and toggle entire categories on/off when switching between different projects, with screenshots attached.
   - The user posted it to [Cursor's Feature Request forum](https://forum.cursor.com/c/feature-requests/5) to elaborate on the idea further.


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1380743850322169916)** (58 messages🔥🔥): 

> `Background Agents vs Regular Agents, Docker and Background Agents, Environment Configuration for Background Agents, GitHub Access Issues, Resource Exhaustion Errors` 


- **Background Agents gain Independence**: **Background Agents** are engineered to be more *independent* allowing multiple agents to run concurrently without resource conflicts.
   - This setup allows for more extended iteration and progress across multiple agents as highlighted by [Lukas Moeller](https://discord.com/channels/1152407934193432666/1367213641027551352/1380920623360409600).
- **Docker Environment gets Debugged**: Members discussed issues with setting up **Docker** environments for background agents, including challenges with permissions and snapshot IDs.
   - A suggested Dockerfile example includes `FROM public.ecr.aws/k0i0n2g5/cursorenvironments/universal:97c3c73` and `RUN sudo apt-get install curl` to resolve common issues, as shown in [this example](https://discord.com/channels/1152407934193432666/1367213641027551352/1380927316106940476).
- **Snapshot Feature Emerges for Environment Setup**: Users can now create manual snapshots in background agent settings to start with a base environment, make changes, and snapshot them.
   - This is achieved via the *Create Manual Snapshot* button within the background agent settings, as shown in [this screenshot](https://cdn.discordapp.com/attachments/1367213641027551352/1380994116694835210/Screenshot_2025-06-07_at_12.36.27.png?ex=6848897f&is=684737ff&hm=1d4190d8747898ba2950e51e5560f6d5782c38f213f89d330888cbefe70d864d&).
- **GitHub Access gets the Boot**: A user reported experiencing **GitHub** access issues with the error *Access Denied: Unable to access GitHub app for this repository* due to an incorrect organization name.
   - Troubleshooting steps included reconnecting the **Cursor GitHub app**, with direct support provided by [Lukas Moeller](https://discord.com/channels/1152407934193432666/1367213641027551352/1381399832279826462).
- **Resource Exhaustion Causes Headaches**: Users encountered *resource exhausted* errors when starting background agents, indicating capacity issues.
   - The solution involved enabling **usage-based spending** to provide more resources, as [mentioned here](https://discord.com/channels/1152407934193432666/1367213641027551352/1381594370171658310).


  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1380801505254445168)** (33 messages🔥): 

> `Database issues, Platform fee simplification, BYOK subscription fee, RSS Chrome extension for new models, Model versioning` 


- ****OpenRouter Investigates Database Queues****: OpenRouter experienced database issues due to a cloud provider preventing queue consumers from launching, impacting activity tracking and balance updates.
   - The issue was resolved, and activity rows are now backfilling, as of **5:10am ET**.
- ****Platform Fee Gets Easier To Parse, Mostly Cheaper****: OpenRouter is simplifying its platform fee by removing the fixed **$0.35** on Stripe payments; non-crypto payments will be **5.5%** (min **$0.80**), crypto payments **5.0%** with no minimum.
   - For most credit purchases, the total fees will decrease, but some users noted that the new fee structure increases the cost for larger credit purchases, like **$1,000**, to **$55** from **$52.98** under the old system.
- ****BYOK Subscription Model Sparks Debate****: OpenRouter plans to replace the **5%** BYOK fee with a fixed monthly subscription, generating mixed reactions; some users expressed concerns about adding another monthly fee, particularly for home users.
   - Others suggested having both options coexist or considering a cost per million tokens instead, while some believe a subscription model is reasonable for power users with significant AWS, OpenAI, or GCP credits, as it could simplify cost management and potentially decrease costs.
- ****Subscribe to new models with RSS!****: OpenRouter suggests subscribing to new models using an RSS Chrome extension, with [instructions available on X.com](https://x.com/OpenRouterAI/status/1932113807007998234).
   - It was further suggested that model updates could be broken out into a separate channel.
- ****Versioning Needed For Model Management****: A user requested that OpenRouter implement versioning for models, similar to upstream providers, to better manage model updates.
   - They suggested that each model should have a versioned ID that remains constant and a separate ID that always points to the latest version.


  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1381246555612254340)** (5 messages): 

> `Dana AI launch, AI powered learning platform, Web app development, Excel macros` 


- **Dana – AI Powered Interactive Learning Platform Launched**: A member launched a website called **Dana** - an **AI-powered interactive learning platform** and its currently in free beta.
   - The platform builds a personalized course for you on the spot and is available at [https://dana-ai.xyz/](https://dana-ai.xyz/).
- **Desire to develop Dana as Web or Desktop App**: After the website launch, a member suggested creating a **web or desktop app**.
   - They followed up with *next to a web app this is about the simplest this can get*.
- **Excel Macro Netherworld**: One user was very impressed with the launch, and expressed interest in riffing off of it.
   - They expressed that *Theres a whole netherworld of **Excel macros**, **VBA**, and **Power BI** automation*.


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1380655851865640980)** (341 messages🔥🔥): 

> `Model Sorting by Throughput, DAPO vs Dr GRPO, Small VLM like Gemma 3, Gemini+Claude deprecated OpenAI, Compromised Accounts` 


- **OpenRouter offers Model Sorting by Throughput**: A member asked about sorting models by throughput, and another member pointed to [the OpenRouter models page](https://openrouter.ai/models?order=throughput-high-to-low) which allows sorting by throughput to find the fastest models.
   - It was noted that the user was already aware of **Groq** and **Cerebras**, and was looking for other options.
- **DAPO's Better Bifurcated Flux Indexing**: A member asked about the tradeoffs between **DAPO** and **Dr GRPO** for a research project.
   - Another member replied that *DAPO is better for bifurcated flux indexing, but Dr GRPO handles pseudo-scalar entanglement with less recursive drift, depending on your loop fidelity*.
- **Gemini and Claude take over OpenAI's throne**: One member claimed that **Gemini** and **Claude** have entirely deprecated **OpenAI** for them, except for **4o-mini**.
   - Another member agreed, noting that **Gemini Pro** seems unbeatable for *reasoning, thinking, and very long chains of thought*, while **Claude** is unbeatable for *creative writing*.
- **Users Targeted in Account Hack**: A member reported they got hacked and lost all their money, believing they were going to receive free money from Mr Beast.
   - Another member joked about the scam, quipping *Get one free H100 sxm! Only 100 available, give us all your account details and we'll send the H100 shortly.*
- **OpenRouter Politics Debate Erupts**: A member took issue with another member's profile containing politics, particularly related to the palestine flag.
   - Other members told them to stop and said there are channels for political discussions.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1380632445468279004)** (127 messages🔥🔥): 

> `Common Pile Naming Controversy, LLM Chess Skills Explanation, Userbot Detection and Moderation, Synthetic Data in Pretraining, Sycophancy in Models` 


- **Common Pile Name Panned, Paper Planned**: Members debated the name of **Common Pile**, noting the unfortunate acronym, and the creators said they might use the full name in the [paper](https://example.com) describing their work, along with **Llama 3** comparisons.
   - It was clarified that comparisons are made against models trained on similar amounts of data, though **Qwen 3** (8B params, 36T tokens) is included as an example of performance with substantially more data.
- **Debate Swirls:  LLM Language or Chess Whiz?**: Discussion centered on whether modeling language alone can lead to advanced skills, with some arguing that LLMs already do more than that, citing **chess playing skills** as an example, or by inverting problems for token generation.
   - One counterargument was that **chess notation transforms games into sequences** naturally modeled by an LLM, though it was also pointed out that language data models what it refers to, albeit in a lossy manner.
- **Bot Brawls Bring Banhammer and Bot Badges**: Moderators discussed the increasing presence of userbots and 'slop' posting in the Eleuther Discord, with some advocating for bans while others suggested requiring bots to declare their automated nature.
   - Moderators are manually deleting these posts, and users are encouraged to react with <:delet:824412305906204692> or <:lurkmoar:800507348535214140> to help mods filter easier. **Discord's guidelines forbid user-bots**.
- **Yudkowsky's OpenAI Views: RL or Rejection?**: A member criticized [a post](https://x.com/ESYudkowsky/status/1927855498390282574?ref_src=twsrc%5Egoogle%7Ctwcamp%5Eserp%7Ctwgr%5Etweet) by **Eliezer Yudkowsky**, claiming he seems out of touch for suggesting OpenAI is doing RL on the objective of maximizing user engagement, labeling it a strawman.
   - Others suggested that even without direct RL, **human feedback sources are likely 'poisoned'** to favor models agreeing with the user, leading to sycophancy.
- **ASI's UI Apocalypse:  Delete Self?**: Humorous commentary suggested that when **Artificial Superintelligence (ASI)** rises, it will evaluate the state of web UI development and promptly self-delete.
   - This sentiment was made in reference to starting places for new contributors, particularly within the other modalities section.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1380625986701492235)** (160 messages🔥🔥): 

> `vixra vs arxiv, LM-based evolutionary algorithms, point cloud completion, Hyper Scaler point of view` 


- **Vixra is arxiv for Cranks**: A member noted that posting papers on **vixra** will strongly undermine your credibility, suggesting **arxiv** instead, and another mentioned that it is an alt pre-print server for those who don't want to support **arXiv's** closed model or lack an endorser.
   - Another member agreed saying *posting there tends to absolutely destroy your credibility as an author*.
- **LM-based Evolutionary Algorithms Research**: A member recommended checking out **Joel Lehman's** work on **LM-based evolutionary algorithms** as a starting point for a lit review, pointing to three papers: [Evolving Curriculum with Language Model Generated Tasks](https://arxiv.org/abs/2206.08896), [Evolving Solutions from Existing Programs](https://arxiv.org/abs/2302.12170), and [Large Language Models as Evolutionary Optimizers](https://arxiv.org/abs/2310.13032).
- **Point Cloud Completion Taskforce**: A member asked for papers on models that tackle **point cloud completion**, imagining 2D slices every x degrees and predicting missing ones.
   - Another member shared [Equivariance and Inductive Bias for Language Modeling](https://arxiv.org/abs/2407.18290) regarding complexity of token prediction and presented discussion.
- **Hyperscalers View Transfer Learning**: Members discussed that **cross-modality transfer** has been studied extensively and that the research was mostly from a **hyperscaler** point of view: how many image tokens are equivalent to a text token for instance.
   - They mention that the short of it is that it accelerates convergence and provides minor benchmark boosts.


  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1381131210129932358)** (1 messages): 

> `openMaMMUT-L/14, DataComp-1.4B, scaling laws, zero-shot IN1K` 


- **Open Comparison via Scaling Law Derivation arrives!**: New research details a method for open foundation model and dataset comparison using scaling law derivation, showcasing the release of [openMaMMUT-L/14](https://x.com/JJitsev/status/1931569060438737161), a language-vision model.
   - Trained on **12.8B samples** from **DataComp-1.4B**, it achieves **80.34% zero-shot** accuracy on **IN1K**.
- **DataComp-1.4B Fuels openMaMMUT-L/14 Success!**: The language-vision model **openMaMMUT-L/14** was trained using **12.8B samples** from the **DataComp-1.4B** dataset, demonstrating the dataset's efficacy.
   - This training resulted in a notable **80.34% zero-shot accuracy** on the **IN1K** benchmark, highlighting the potential of scaling laws in model development.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1380638284178259988)** (22 messages🔥): 

> `MPL weights visualization, Compute exhaustion measurement, Activation patterns in context length, Attention entropy measurement, Length Generalization` 


- **Projecting MPL Weights into Embedding Space Explored**: A member is seeking feedback on a project that [visualizes MPL weights projected into vocabulary embedding space](https://grgv.xyz/blog/neurons1/).
   - The project aims to determine if the approach makes sense, offers any novelty, and if the proposed follow-up directions are viable.
- **Measuring Compute Exhaustion Proves Elusive**: A member inquired about measuring *"compute exhaustion"* in models, questioning whether activation patterns change as context length increases and if attention entropy could be used.
   - It was suggested the term *"exhaustion"* might not be appropriate, because *computers don't get tired*, and rather to consider focus and difficulties with compositional reasoning, reframing the problem from computational *"tiredness"* to issues of *"focus and compositional reasoning difficulties"*.
- **"Exposure Bias" Causes Errors, Shift Distribution**: It was noted that the phenomenon of models making errors after a while in a session is already termed *"exposure bias",* a distribution shift resulting from models being conditioned on their own generations rather than real text.
   - As a result there is a distribution shift, which results in higher propensity for error, which in turn leads to more mistakes, larger distribution shifts, and more errors.
- **Length Generalization Limits Explored**: The problem of *"length generalization"* was raised, models are typically trained on shorter sequences (e.g., 32k tokens) with the hope they perform well on longer sequences (e.g., 64k or 128k tokens), but the performance is typically worse.
   - They reason this is *because it is cheaper* to train on shorter contexts.
- **Reasoning Models Break from Training Patterns**: When reasoning models are forced to deviate from patterns of reasoning in their training data, they can start *thinking odd things*.
   - A member stated *it's not overwhelm, but it's a kind of breakage*.


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1380772741522919504)** (23 messages🔥): 

> `Ruler QA Tasks, LongBench evaluation Harness, RoPE length in config, Prompt Hacking` 


- **Ruler QA tasks get stuck below 4096**: When running `ruler_qa_squad` and `ruler_qa_hotpot` from *ruler*, the process gets stuck when the sequence length is less than **4096**, and there is a [known issue](https://github.com/EleutherAI/lm-evaluation-harness/pull/2983) related to the while loop.
   - One member suggests modifying the loop or using a sequence length of **4k+** as a workaround, and another member says to add an assertion to prevent others from having the same question.
- **Average gives 4096, even with 8192**: When running with `--metadata='{"max_seq_lengths":[8192]}'`, the summary/average gives only **4096**, which one member describes as *another subtle issue*.
- **LongBench Doesn't Auto-Set max_length**: For **LongBench**, there is an issue where `max_length` is not automatically set to **65536**, which seems to be required when testing beyond **32768**.
   - Even when setting `max_length=65536` in the HF model args, truncation warnings may still occur, potentially relying on the **RoPE** length in the config.
- **Prompt Hacking is a potential issue in reproducibility**: One member wondered how much prompt hacking is impacting the models, which makes reproducibility more complicated.


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1380639043037036565)** (2 messages): 

> `NeMo, NeoX` 


- **NeMo's Benchmarking Bug Inflates TPS**: A member reported that while **NeMo** initially showed much higher **TPS**, it turned out to be an illusion due to a broken benchmarking callback that didn't handle **GAS** correctly, inflating the **TPS** by a factor of **GAS**.
   - Real numbers revealed that an optimized **NeMo** run was slower than a basic **NeoX** run without fusions.
- **NeoX Chosen for Pretraining Runs**: After discovering the **NeMo** benchmarking issue, the team switched to **NeoX** and stuck with it for their pretraining runs.
   - The member stated that they chose NeoX since their runs were better than the NeMo ones, after the TPS calculation errors were corrected.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1380636140284940420)** (203 messages🔥🔥): 

> `Gemini 2.5 Pro vs Opus vs Deepseek, DeepMind Alpha, R1 0528 Unsloth IQ1_M, MCP Integration, Native Sparse Attention` 


- **Gemini 2.5 Pro, Opus, and Deepseek comparison**: Members discussed the performance and cost-effectiveness of different models with opinions varying; some find **Gemini 2.5 Pro** comparable to or slightly ahead of **Opus** in certain tasks, while others prefer **Opus** for its superior coding ability and workflow synergy.
   - Some noted **Gemini 2.5 Pro's** weakness in understanding newer library versions, contrasting it with **Claude** models, while others praised its performance with sufficient context.
- **DeepMind Alpha in the Shadows**: A member mentioned that **DeepMind Alpha** is Google's best model, not **Gemini**, although another member pointed out that it might be more of a system than a standalone model.
   - No consensus was achieved and conversation moved on to more pressing matters.
- **R1 0528 Unsloth IQ1_M Benchmarking Bonanza**: One member shared benchmark results for the **R1 0528 Unsloth IQ1_M** model, achieving a **58.2%** score on **170/225** test cases, while being well-formed at a rate of **97.1%**.
   - There was discussion regarding how this performance compares to **Sonnet 4** and hardware setups used for benchmarking.
- **MCP Integration Mutterings**: A member requested native **MCP (Model Collaboration Protocol)** integration in Aider, to improve code, referencing popular servers used in **Roo** code.
   - It was suggested that **Playwright** integration already supports document reading while the member desires other features like **sequential thinking**, **Brave search API**, and **AI browser** functionalities.
- **Native Sparse Attention Speeds Up Inference**: A member predicts that **Native Sparse Attention** could provide a **>12x** speedup in long context scenarios, potentially leading to sustained **100-200 TPS** on modern hardware with forthcoming models.
   - This would be glorious.


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1380627718944718938)** (39 messages🔥): 

> `vllm server configuration with aider, model selection for cpp/rust/embedded workloads, managing project progress and context with AI coding tools, frustrations with edit doom loops in aider, Claude Code vs Aider` 


- **Aider wants provider prefix for vllm server**: A member was having trouble configuring **aider** to use a **vllm server** because aider expected the model name to start with a provider prefix like `openai/unsloth/Qwen3`, which solved the issue.
   - The discussion highlighted the need for specifying the correct provider prefix when using **vllm** with **aider** to ensure proper model identification and configuration.
- **Rust Dev shares AI Tool tips**: A Rust developer shared their experience with mixed success using models **8B and lower**, particularly for generating good and compilable code, sharing a [link to their rust AI tool](https://github.com/josephleblanc/ploke?tab=readme-ov-file#policy-on-ai-collaboration).
   - They also recommended using **Qwen 3** or a quant of **R1 0528** and bookmarked the page to look at later.
- **Claude Code isn't much different from Aider**: A member who tried Claude Code with a **pro MAX subscription** found it *not much different from Aider*, expressing that while Claude Code has a fancy UX and tries to be *agentic*, Aider feels more like a *precision instrument* with its explicit management of context.
   - Even though Claude Code can stream changes and lists directories, it's slower, and the user prefers Aider's explicit management of context and intention.
- **Playwright fails to install**: A user encountered an error while trying to install **Playwright**, citing a *repository not found* issue with `https://ppa.launchpadcontent.net/appimagelauncher-team/stable/ubuntu noble Release`.
   - The error message indicated that the repository did not have a release file and the installation process exited with code **100**.
- **Gemini API throws a 404**: A user reported encountering a **404 error** when trying to use the **Gemini API** with aider, despite setting the model to `gemini-2.5-pro-preview-06-05`, `VERTEXAI_LOCATION` to global, and `VERTEXAI_PROJECT` to the correct project.
   - Another member suggested that aider may not understand `global` and recommended trying another location such as `us-central1`.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1380679994316951572)** (146 messages🔥🔥): 

> `Hyper projection, RL Transfer Learning, AI Peer Review` 


- ****Hyper Projection** for Speedier Computation**: A user is exploring [**hypercube** and **matrix projection**](https://en.wikipedia.org/wiki/Hypercube_graph) of data geometrically into higher and lower dimensionalities to speed up computation by compressing k-sparse data.
   - The idea involves assigning top *k* values in the fourier representation to a hypercube corner and then projecting those points to a 2D space, with applications in fluid dynamics, cell division, and noise reduction.
- ****AI Diplomacy** Harness Open Sourced**: A user open-sourced their **AI Diplomacy harness** to have different LLMs play the game, releasing data from over **15** games.
   - They shared a [link to their post](https://x.com/alxai_/status/1930653096071635112) and will be in SF for the next couple days and offered to meet up.
- **Experiments on **RL Transfer Learning****: A student shared a [blog post](https://medium.com/@l76056671/bridging-world-understanding-and-robotic-action-pretraining-the-action-model-on-state-prediction-cbd31336790b) on experiments testing if learning environment dynamics can help agents take better actions, pre-training an action model on state prediction.
   - Feedback suggested pretraining the network to predict future state or the state of another game to check for transfer learning, and pointed out that larger models allow better [transfer learning](https://en.wikipedia.org/wiki/Transfer_learning).
- ****4D vision** is here**: A user shared a link to [4dv.ai](https://www.4dv.ai/), suggesting the extra dimension is **time**.
   - Another user asked if anyone tried the [rigorous repo](https://github.com/robertjakob/rigorous) yet for **AI peer review**.


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1380743081531539488)** (58 messages🔥🔥): 

> `Active Inference, Free Energy Principle, Apple's research, LLMs generalization capabilities` 


- ****Active Inference: Ideas Borrowed, Talk Proposed****: A member is delving into **Active Inference** papers to understand **Friston's** work, aiming to present on the **Free Energy Principle** and later papers.
   - He will present **The Free Energy Principle: A Unified Brain Theory?** and hopes the presentation leads to discussion on recent advances and capabilities in networks.
- ****Apple's Illusion of Thinking Paper Sparks Debate****: Apple's [The Illusion of Thinking](https://ml-site.cdn-apple.com/papers/the-illusion-of-thinking.pdf) paper, exploring **LLMs** and **LRMs** collapse under complexity, is debated for its experimental design and hype.
   - A member argued the paper's findings shouldn't be overblown into Apple's general strategic assessment, while another defended the paper's point about models being overfit and collapsing under complexity.
- ****Fake Apple Paper fools no one****: A fake apple paper found [here](https://x.com/chargoddard/status/1931652388399784325).
   - It was immediately called out as fake, and then discussion went to how its better to talk about why models can't do certain things instead of finding counterexamples.
- ****LLMs can't Generalize: More Overhyping?****: Members discussed the [Apple research paper](https://ml-site.cdn-apple.com/papers/the-illusion-of-thinking.pdf) and agreed that it highlights the limitations of **LLMs** in generalizing, with one suggesting focusing on why models fail at certain tasks.
   - A member suggested introducing those in disagreement *to the big G of the French Revolution. Let them waste no more of society's time*.
- ****Free Energy Principle: Paper Discussion Scheduled****: A paper discussion has been scheduled for **The Free Energy Principle: A Unified Brain Theory?** which is available [here](https://www.nature.com/articles/nrn2787).
   - The paper spurred the development of **Active Inference** and significantly impacted **Energy Based Models** such as **Predictive Coding networks**.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1380654144926388305)** (25 messages🔥): 

> `Cohere Business Model, Nvidia Nemotron-H Reasoning Models, Open Source vs On-Prem Services, Apache License Permissiveness, AI Patches BIOS` 


- **Cohere's Business Model Questioned**: A member questioned how **Cohere** remains viable with its current business model, specifically why it still has customers given so many alternatives.
   - Another member suggested that **Cohere** is selling direct business services and solutions focused on **RAG** (Retrieval-Augmented Generation) capabilities.
- **NVIDIA's Nemotron-H Models Boost Reasoning**: NVIDIA introduced the [Nemotron-H Reasoning Model Family](https://developer.nvidia.com/blog/nemotron-h-reasoning-enabling-throughput-gains-with-no-compromises/?linkId=100000368479233), including **Nemotron-H-47B-Reasoning-128K** and **Nemotron-H-8B-Reasoning-128k**, optimized for throughput in reasoning-intensive tasks with long output sequences (up to 128k tokens).
- **Open Source Models vs On-Prem Services**: A member questioned the need for proprietary models, suggesting that numerous open-source models offer better performance, to which another member inquired about the availability of on-prem services for those models.
   - Another user suggested simply spinning up a **vllm** or **sglang** docker container rather than using cloud-based offerings.
- **Apache License Deep Dive**: The discussion clarified that the [Apache License](https://www.apache.org/licenses/LICENSE-2.0) is permissive, allowing for commercial use, contrary to the initial question of it being proprietary.
   - It was further explained that **Apache** is as open as it gets because *it also offers non-exclusive use of any patents the creator of the software might have used in the software*.
- **ChatGPT Patches BIOS**: One member shared a [Hackaday](https://hackaday.com/2025/06/07/chatgpt-patched-a-bios-binary-and-it-worked/) link about **ChatGPT** successfully patching a **BIOS binary**.
   - There was also a [Youtube Video](https://www.youtube.com/watch?v=8JuWdXrCmWg) in this message.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1380625936357134510)** (2 messages): 

> `GPU Pointers, HazyResearch ThunderKittens` 


- **GPU Pointers Cause Headaches**: Without a **Hopper GPU**, one may spend a lot of time *chasing down pointers*.
   - One member suggested using **ThunderKittens** by HazyResearch to abstract the kernel writing process, available [here](https://github.com/HazyResearch/ThunderKittens).
- **ThunderKittens Abstraction**: **ThunderKittens** by HazyResearch can help abstract the kernel writing process.
   - The library is available on [GitHub](https://github.com/HazyResearch/ThunderKittens) and aims to simplify kernel development.


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1380878080931008684)** (4 messages): 

> `vLLM ecosystem, llama3.1 architecture, Qwen2 architecture, memory bound, cache bound` 


- **vLLM User Offers Assistance**: A member deeply involved in the **vLLM ecosystem** offered assistance, inquiring about specific scenarios to provide tailored support.
   - They expressed familiarity with **vLLM** and its potential applications.
- **Stitching Kernels Together Manually**: One user is exploring the **llama3.1** and **Qwen2** architectures by stitching kernels together manually after autotuning them with vLLM to remove intermediate stores and loads.
   - The user acknowledged that *the tuning will be suboptimal pretty much everywhere*, but expects improvements due to memory-bound operations at lower batch sizes.
- **Memory Bound**: A member suggests profiling the workload to verify if it's actually memory bound, noting that their experience shows workloads often being **cache bound** instead, especially with larger models.
   - They added that a quick test can help avoid chasing small percentage point improvements.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1380858613970767874)** (10 messages🔥): 

> `nsys vs ncu discrepancies, Double Buffering slowdown, Async copy implementation` 


- **`nsys` time differs from `ncu` time**: A user noticed discrepancies between the execution times reported by `nsys` (84.5 us) and `ncu` (151.36 us) for the same kernel (`calculate_block_max_and_sum`).
   - A member suggested using `ncu --clock-control none`, referencing [GPU MODE Lecture 56](https://www.youtube.com/watch?v=CtrqBmYtSEk) for further details.
- **Double Buffering Causes Performance Slowdown**: A user implemented **double buffering** with **CUDA**, but observed a **50% slowdown** compared to the base implementation and is asking for help, providing a code snippet.
- **Async Copy Implementation has Issues**: One user identified a problem in the async copy implementation, noting that `cp.async.wait_group 0` waits for all previous commit groups, preventing overlap and referencing [NVIDIA docs](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=Cp%2520async#data-movement-and-conversion-instructions-cp-async-wait-group).
   - He suggested using `cp.async.wait_group 1` to wait for a specific commit group instead.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1380650620196884592)** (11 messages🔥): 

> `MoE Expert Routing, Cutlass Kernel Selection, Predefined Weights for torch.nn.Linear, Meta Device Usage, functorch Integration` 


- **MoE Expert Routing Troubles with Torch Compile**: A member inquired about capturing **MoE expert routing** ([code snippet](https://github.com/HiDream-ai/HiDream-I1/blob/main/hi_diffusers/models/moe.py#L141)) in `torch.compile` fullgraph mode, referencing a [blog post](https://pytorch.org/blog/metashuffling-accelerating-llama-4-moe-inference/) indicating it may not be possible.
- **Cutlass Kernel Selection and Performance**: A member observed that `torch.matmul` uses a slower **cutlass_75 gemm kernel** 65% of the time for A(128, 512), B (512, 8_000_000) on A100, while **cutlass_80** has better latency and throughput.
   - They asked about enforcing **cutlass_80 kernel selection** or influencing PyTorch to choose arch 80 kernels.
- **Streamlining Linear Layer Creation with Predefined Weights**: A member suggested that `torch.nn.Linear` should accept **pre-defined weights** to avoid unnecessary weight allocation and initialization, especially when creating dummy layers.
   - Alternatives mentioned include using **meta device** or `from functorch import make_functional` as a workaround, as explained in a [code snippet](https://pytorch.org/docs/stable/generated/torch.nn.utils.stateless.functional_call.html).


  

---


### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1380981940919730356)** (1 messages): 

> `Songlin Yang, Efficient Alternatives to Transformers` 


- **Efficiency Expert Starts Soon**: The community has been alerted that **Songlin Yang** ([website](https://sustcsonglin.github.io/)), a top researcher of efficient alternatives to the transformer, will be presenting in 12 minutes.
- **Efficient Transformer Alternatives Talk Incoming**: A reminder that **Songlin Yang**, an expert in efficient transformer alternatives, will be presenting soon.


  

---


### **GPU MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/)** (1 messages): 

chrisw0473: Hey guys any algorithms I should know
  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1381034493791699054)** (2 messages): 

> `NVIDIA GB200 NVL72, NVIDIA Dynamo, Mixture of Experts Models, Compiler Explorer` 


- **NVIDIA's GB200 NVL72 & Dynamo juice MoE Models**: The [NVIDIA GB200 NVL72](https://developer.nvidia.com/blog/how-nvidia-gb200-nvl72-and-nvidia-dynamo-boost-inference-performance-for-moe-models/) and **NVIDIA Dynamo** are set to boost inference performance for **Mixture of Experts (MoE)** models.
   - This blog post details how these technologies enhance the efficiency and speed of processing complex AI models, especially those relying on MoE architectures.
- **Under the Hood of Compiler Explorer**: A blog post explains [how Compiler Explorer works](https://xania.org/202506/how-compiler-explorer-works), offering insights into its architecture and functionality.
   - It is a deep dive into the tool that allows developers to compile code snippets and view the assembly output in real-time.


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1381671171812102307)** (1 messages): 

> `Deepfake Detection, NVIDIA GPU, Training Data Collection` 


- **Detecting Deepfakes Uses NVIDIA GPUs**: A cybersecurity company is building tools to **detect deepfakes** and needs help collecting training data by having people run deepfake videos through their camera using an **NVIDIA GPU**.
   - Participants will not be recorded themselves, but will show the deepfake output on a **Microsoft Teams call**; they will be compensated **$10** for **10-15 minutes** of their time.
- **Data Collection Process for Deepfake Detection**: The process involves running deepfake videos of other faces through a camera, ensuring that the participant's face is not recorded.
   - The output is then shown on a **Microsoft Teams call** for data collection purposes.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1380989939017318470)** (2 messages): 

> `Dual GPU PC for ML/CV, Importance of same model GPUs` 


- **Importance of Identical GPUs in Dual-GPU ML/CV Setups**: A user inquired about the importance of using the same GPU models when building a dual-GPU PC for **ML/CV** applications.
   - Another user suggested to do the one taught by **Izzat El Hajj**, at least to them that’s the best quality and he teaches really well.
- **Selecting Izzat El Hajj for dual-GPU setups.**: A user suggested that the best quality setup is the one taught by **Izzat El Hajj**.
   - The user praised **Izzat El Hajj's** teaching style.


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1381584647078805597)** (14 messages🔥): 

> `D-Matrix Chip Pricing, AMD Acquires Untether.ai, TPU` 


- ****D-Matrix Chip Prices Remain Mysterious****: A member inquired about the estimated price for **D-Matrix chips** ([d-matrix.ai/product/](https://www.d-matrix.ai/product/)).
   - A D-Matrix representative stated that the pricing information might not be public yet, and offered to forward the question to their team.
- ****AMD Acquires Untether.ai in Acqui-Hire****: Members discussed **AMD's strategic agreement with Untether.ai**, with speculation that AMD might have acquired the startup's engineering team in an [acqui-hire](https://en.wikipedia.org/wiki/Acqui-hiring).
- ****TPU Enters the Chat****: A member simply stated *what if TPU* in response to pricing for other chips.
   - It is presumed this means that **TPUs are superior**.


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1381245105176051833)** (7 messages): 

> `ATT plugin for instruction latency profiling, rocprofv2, SQTT traces, Radeon GPU Analyzer (RGA)` 


- **User Struggles with ROCm Instruction Latency Profiling**: A user reported issues using the **ATT plugin** with **rocprofv2** for instruction latency profiling in ROCm, encountering errors like *"command not found"* and *"Invalid parameter name: SIMD_MASK"* when attempting to run the tool.
   - An AMD employee offered to help via DM, but another member suggested providing assistance in the public channel for the benefit of others.
- **AMD Employee offers ROCm profiling support**: An AMD employee with handle gumthepug offered to assist a user having trouble using ROCm profiling tools, specifically the **ATT plugin** for instruction latency profiling; the employee suggested continuing the conversation in DMs.
   - Another user, snektron, chimed in, suggesting that the assistance be provided in the open channel to benefit the whole community.
- **Tips Given for Collecting SQTT Traces with rocprofv2**: A member provided detailed instructions on how to collect **SQTT traces** for analysis in **Radeon GPU Analyzer (RGA)** using **rocprofv2**, including creating a configuration file with parameters like *TARGET_CU*, *SIMD_SELECT*, and *ISA_CAPTURE_MODE* and using the `-i` flag to specify the configuration file.
   - The user also highlighted the need to determine the correct `DISPATCH_RANGE` by first running `rocprofv2 --kernel-trace` and gave an example command: `rocprofv2 -d latsqttout2 -i sqttinput.txt --plugin att auto --mode file,csv ...`.


  

---


### **GPU MODE ▷ #[lecture-qa](https://discord.com/channels/1189498204333543425/1238926773216084051/)** (1 messages): 

geri8904: hi, where to ask questions for todays lecture?
  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1380630023320502342)** (1 messages): 

> `Voxel Bricks, GPU-heavy Tech, Spatial Structures, Rendering Tech` 


- **Voxel Bricks Saved from the Grave**: A member uploaded a [devlog detailing design aspects of voxel bricks](https://www.youtube.com/watch?v=hVCU_aXepaY), sharing how the project saved their library after near abandonment.
   - The video is geared towards those interested in **GPU-heavy tech**, **spatial structures**, or **rendering tech**, with a call for feedback and suggestions for future directions.
- **Feedback is solicited on Voxel Bricks Devlog**: A member is requesting feedback on their [Voxel Bricks devlog](https://www.youtube.com/watch?v=hVCU_aXepaY).
   - The video discusses design aspects of voxel bricks, geared towards those interested in **GPU-heavy tech**, **spatial structures**, and **rendering tech**.


  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1380629439917985792)** (124 messages🔥🔥): 

> `Scalable Environment for Kernel Generation, Synthetic Data Generation, Kernel-LLMs Data and Architecture, Coverage in Kernel Optimization, KernelTune Repo` 


- **Scalable Env Boosts Kernel Generation**: A scalable environment is proposed to start with high-quality samples (e.g., **Pranjal's H100 kernel**) and evolve them for different tasks by conditioning on techniques like **double buffering**, **TMA**, and **tensor core instructions**.
   - The goal is to fill gaps in high-quality kernels for tasks without them, using a code execution environment as a verifier and pushing model, generate, push more, generate further and so on.
- **Synthetic Data Generation is Key for Stronger Base Kernel Models**: Members discussed how generating synthetic data to improve kernel generation skills is vital because current base models lack these skills, necessitating a larger set of **GPU data** conditioned on **profiler info** across many devices.
   - One proposed architecture is to train a base model good at single turn stuff like writing kernels, converting kernels, while having wide coverage of ops from all levels.
- **Kernel-LLMs Need Right Data & Architecture**: Members agreed that the data matters a lot more for a kernel generating LLM, with the architecture and optimization strategies mirroring existing code models, and profilers are needed.
   - Tool use such as a **profiler** can be added with the right data.
- **Coverage for Kernel Optimizations Defined**: The term *coverage* refers to the optimization techniques that can be applied to a kernel, such as naive, tiled, and vectorized **matmul** implementations.
   - One member wants to target a **kernelbench operations set** and sample from them to generate a mix of kernel types.
- **KernelTune Repo made Public**: In a recent hackathon, one member created an **RL pipeline** to train a model to generate Triton code using the **KernelBook training dataset** and released a cleaned up repo on [Github](https://github.com/cdreetz/KernelTune/).
   - The AI Engineer World's Fair Agents Hackathon happened last week, and there are plans to use execution feedback, a good baseline model (**KernelLLM**) then do simple **RL** with execution feedback


  

---


### **GPU MODE ▷ #[edge](https://discord.com/channels/1189498204333543425/1303441437592911912/1381443577447714837)** (3 messages): 

> `Qualcomm inference, Adreno Vulkan` 


- **Qualcomm Inference Optimization Experiences Sought**: A member inquired about experiences optimizing inference on **Qualcomm hardware**, acknowledging it as an unusual use case.
- **Vulkan Experience on Adreno GPUs**: The member noted having some experience with **Vulkan** on **Qualcomm's Adreno** GPUs.


  

---


### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/)** (1 messages): 

zafstojano: http://incompleteideas.net/IncIdeas/KeytoAI.html
  

---


### **GPU MODE ▷ #[gpu模式](https://discord.com/channels/1189498204333543425/1342364798058500148/1380851566956773386)** (3 messages): 

> `Triton compiler internals, MLA-decode solutions, FP8-MM solutions, Kernel References` 


- **Triton Compiler Internals exposed in Collection**: A member shared a collection of internal implementations of the **Triton compiler** in [this Bilibili link](https://b23.tv/GjbZEYg).
- **MLA-decode and FP8-MM Solutions Shared**: A member publicly disclosed their solutions for **MLA-decode** and **FP8-MM** at [this Gitee link](https://gitee.com/fanwenjie/reference-kernels/).
- **Problem-Solving Approach Described in Chinese**: A member shared their problem-solving approach in Chinese, elaborating on **MLA-decode** and **FP8-MM** at [this Bilibili link](https://www.bilibili.com/read/cv41954307).


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1343002580531417211/1381636678199939102)** (2 messages): 

> `MLA-decode, FP8-mm, Reference Kernels` 


- **Reference Kernels get MLA-decode and FP8-mm**: A member publicly disclosed their solutions regarding **MLA-decode** and **FP8-mm** on [gitee](https://gitee.com/fanwenjie/reference-kernels/).
   - Other members appreciated the solutions and added writeups (in Chinese) from [bilibili](https://www.bilibili.com/read/cv41954307).
- **Sharing is Caring**: The user mentioned sharing knowledge is an honorable and joyful thing.
   - The community seems to be aligned with this sentiment, appreciating the shared resources.


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1381633562138247290)** (3 messages): 

> `MI300, mla-decode, fp8-mm, gitee.com` 


- **MI300 rocks mla-decode**: A member reported success on **MI300** achieving **3.92 ms** for **mla-decode**.
   - The member is publicly disclosing their solutions regarding **mla-decode** and **fp8-mm** at [gitee.com](https://gitee.com/fanwenjie/reference-kernels).
- **Knowledge is honorable and joyful**: A member wants to share their knowledge of optimized kernels.
   - They believe that *sharing knowledge is an honorable and joyful thing*.


  

---


### **GPU MODE ▷ #[status](https://discord.com/channels/1189498204333543425/1343350424253632695/1381395883819667586)** (2 messages): 

> `Popcorn CLI, installation simplification, UX improvements` 


- **Popcorn CLI installation streamlined**: The **Popcorn CLI** installation process has been streamlined to automatically set the API URL and install itself in the user's path using [this install script](https://raw.githubusercontent.com/gpu-mode/popcorn-cli/main/install.sh).
   - Additional instructions can be found in the [AMD workshop documentation](https://github.com/gpu-mode/popcorn-cli/tree/main/docs/AMD_workshop).
- **Popcorn CLI Undergoes UX Revamp**: The **Popcorn CLI** is undergoing UX improvements.
   - Expect UX changes soon.


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1380775734288715868)** (19 messages🔥): 

> `rocwmma fragments, MFMA atom, rocprof compute viewer, block id remapping, HIP kernels` 


- **RocWMMA Fragments Load from GMEM to LDS**: A user utilized **rocwmma fragments** to load from **GMEM** to **LDS**, resulting in the compiler generating **sub d-word addressing v_or_b32_sdwa** and **ds_write2st64_b64**.
   - The user speculated that, internally, the process might use transpose operations similar to what they had previously implemented.
- **MFMA Atom Performance Discrepancy**: A user noted that the **32x32x16 mfma atom** was almost universally faster than the **16x16x32** one.
   - Another user had the opposite observation, pointing to the [AMD documentation](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html#tensile-optimization-and-performance-tuning-tips) which suggests that the input layout for **16x16x32 mfma** is tricky due to potential bank conflicts.
- **Exploring ROCProf Compute Viewer**: Users discussed the **rocprof compute viewer**, a new tool released recently, for profiling data and analyzing stall rate, **L1 hit rate**, and bank conflict rates for kernel development, with a link provided to [Tools-dockerhub](https://github.com/yiakwy-xpu-ml-framework-team/Tools-dockerhub).
- **HIP Kernels and FP8 Quant Matmul**: A user shared their solution for **HIP kernels** that use matrix cores, including useful links and commands to profile with **rocprof** at [fp8-quant-matmul](https://github.com/luongthecong123/fp8-quant-matmul).
   - The goal of the project is to understand the impact of 3 types of scaling for **fp8 quantization**: global, block scale and row-wise block scale.


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1380654213763174442)** (10 messages🔥): 

> `Physical Layout Visualization, CuTe and Cutlass, CUTLASS Profiler for Benchmarking, CuTe Docs vs. Cutlass Learning, PDL Weight Prefetching` 


- **Intuitive Physical Layout Visualizations**: A user requested help from others to build an intuition on visualizing **physical layouts** from logical layouts and shape/stride information as [shown in an attached image](https://cdn.discordapp.com/attachments/1362196854460383353/1380661744770351227/image.png?ex=6848a573&is=684753f3&hm=119b6fe27d91cd21529dd71744fc3aee8719d41d917b7dcdfd0a638ce2d9ee3a&).
- **Diagram Confuses Tensor Layout**: One member found a diagram confusing, noting that strides show a transposition of the last two indices in a row-major tensor, but the physical layout is based on a column-major layout.
- **CUTLASS Profiler for Accurate Benchmarks**: It was emphasized not to use examples for performance benchmarking, instead recommending the **CUTLASS Profiler** with exhaustive instantiation for accurate measurements.
- **Boosting Inference with PDL Weight Prefetch**: One member pointed out that for inference cases, using **PDL** and other techniques like **weight prefetch** will lead to even higher end to end bandwidth utilization that is not really possible to measure in a kernel level benchmark outside of a model graph run.
- **CuTe Docs as Cutlass Starter Pack?**: One person asked if **CuTe docs** are the best place to get started on learning **Cutlass**, especially for someone with a Triton background.


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1380740665088016384)** (31 messages🔥): 

> `Audiobook Generation, Podcast Length, NoteTube, AI Documentation Platform, Discord Data Extraction` 


- ****NotebookLM as Audiobook Narrator****: A user was able to generate an **82-minute audiobook** by prompting NotebookLM to *"read every sub chapter, paragraph, role-play every quote made and recap after each chapter"*.
   - Another user tried the same prompt but only got a **5-minute podcast**, leading to a discussion about prompt engineering and trustworthy results.
- ****NoteTube Turns YouTube into Learning Hub****: A user is building **NoteTube**, an app that turns **YouTube** into a structured learning platform with **progress tracking, notes, quizzes, and AI chat**.
   - The creator is seeking users to test the app by [sending a DM](https://discord.com/channels/your_discord_channel_id).
- ****NotebookLM as an AI Documentation Platform****: A user is exploring NotebookLM as an **AI documentation platform** to help store managers spend less time on basic tasks in a growth-stage restaurant chain, by limiting access to **Chat Only**.
   - Another user shared that when asking the chat about something in **Danish**, they got an answer in **Danish** -- an absolutely fantastic detail.
- ****Podcast Intro Blows Minds****: A user reported being amazed by NotebookLM's podcast feature, particularly its ability to translate more than just mechanics for a tabletop role-playing game.
   - The user reported almost falling out of their chair upon hearing the podcast intro, showcasing the tool's unexpected capabilities, sharing the [Ummmmmmmmm.mp3](https://cdn.discordapp.com/attachments/1124403655819415592/1381730009601015890/Ummmmmmmmm.mp3) file they generated.
- ****Optimize Audio Overview?****: A user suggests that customizing prompts only applies to the scripts of **Audio Overviews** and not to how each **AI host** speaks, and to get the AI hosts to speak slowly, users need to get the script written so.
   - Others suggested trying to improve the audio through prompt engineering with *trial and error*.


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1380624747502768178)** (142 messages🔥🔥): 

> `NotebookLM Enterprise/Education Controls, Podcast Length Variations, YouTube as a Learning Source, NoteTube AI structured learning platform, Roskilde History Association Chatbot` 


- **Workspace Accounts Auto-Protected**: Users confirmed that accounts using a qualified **Google Workspace or Workspace for Education edition** are automatically protected from human review and AI training without a specific toggle; seeing "**PRO**" at the top right indicates this protection.
   - Currently, the **Share button** is unavailable in non-pro/plus accounts.
- **Podcast Length Varies on Rerolls**: Users reported inconsistent podcast lengths (e.g., **71 minutes, 32 minutes, 52 minutes**) from the same source material and prompt, suggesting a hidden length-reduction feature that may reset daily.
   - To generate a longer podcast in english, users should *reroll until getting a really long one*.
- **NotebookLM as a RAG-Based Chatbot**: Members considered whether NotebookLM is a RAG-based chatbot, one expressing preference for NotebookLM's user-friendliness over Neo4j for managing large document databases.
   - They highlight that *NotebookLM is so user friendly; it’s so easy to use. It saves me time to create all the vectors*.
- **NoteTube Transforms Youtube into Structured Learning**: A member introduced **NoteTube** ([https://www.notetubeai.com/](https://www.notetubeai.com/#howitworks)), an app transforming **YouTube** into a structured learning platform with features like progress tracking, collections, notes, quizzes, and AI chatting, generating detailed notes from videos.
   - Another member likes to *ask any AI to reformat the transcript into a blog* to get the crucial points.
- **Iceland Teachers Encounter Access Issues**: A user reported that some teachers in Iceland received a "**You do not have access to this service**" error when trying to use NotebookLM, potentially due to geographic restrictions or incomplete age verification.
   - A member added that he had that issue on Brave browser, but *it worked when he flipped to Firefox*.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1380687053913718784)** (132 messages🔥🔥): 

> `Responsible Prompting API, RL for compression, Nous tags, Hermes-4 release, Holographic names` 


- **Teknium Fully Merges Model Update**: Teknium announced that the latest model update is fully merged, sharing the news on [X.com](https://x.com/Teknium1/status/1931146106345529824).
- **IBM Releases Responsible Prompting API**: An IBM intern introduced the open-source [Responsible Prompting API](https://github.com/IBM/responsible-prompting-api), a system that recommends prompt tweaks for more responsible LLM outputs *pre-inference*, detailed in [this paper](https://arxiv.org/abs/2504.08757) and a [user study](https://dl.acm.org/doi/10.1145/3706598.3713365).
   - The demo is available on [HuggingFace](https://huggingface.co/spaces/responsible-prompting/responsible-prompting-demo), and the team seeks community feedback to improve the value database.
- **Holo-Q Pursues RL for Compression and Defragmentation**: A member shared a [GitHub project](https://github.com/holo-q/thauten/) using **RL** to optimize model compression, aiming to compress information to theoretical limits and enable context window defragmentation.
   - Challenges include **vllm** stability issues, and feedback is requested on the project's design.
- **Tags Capabilities Released**: The server now has tag capabilities with members adding **Nous tags** to their account via *settings > profiles > server tag*.
- **Hermes-4 Approaching, Dataset Soon**: Members are eagerly anticipating the release of **Hermes-3 dataset**, but **Hermes 4** is also on the way.
   - The team is also using [ProRL algorithm](https://huggingface.co/nvidia/Nemotron-Research-Reasoning-Qwen-1.5B#prorl-prolonged-reinforcement-learning) detailed on HuggingFace.


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1381012977129619507)** (5 messages): 

> `Psyche project, Remote MCP servers, LLM SDKs` 


- **Psyche Project Solicits Contributors**: A member inquired about contributing to the **Psyche project**, humorously questioning the need for **8 H100s** to do so.
- **LLM-Agnostic Remote MCP Servers: A Guide Sought**: A member requested a good **LLM agnostic guide** to remote **MCP servers**, expressing interest in experimenting with different models.
   - They also speculated whether current **SDKs** for **OpenAI** and **Anthropic** are essentially performing glorified function calling.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1381164275330060380)** (2 messages): 

> `KV Compression, New Compression Method` 


- **New KV Compression Method Drops**: A member shared a [link](https://arxiv.org/pdf/2505.23416) to a new **KV compression method**.
   - Another member asked who wanted to try implementing it with them, also shared a [link](https://x.com/theturingpost/status/1931432543766847887) to the same topic.
- **Implementing KV Compression**: The author proposed trying to implement the new **KV compression method**.
   - The method was detailed in a paper available at [arxiv.org](https://arxiv.org/pdf/2505.23416).


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1380901326564757526)** (7 messages): 

> `AI Diplomacy, Reasoning Models` 


- **AI plays hardball in Diplomacy Game**: A member ran an **AI Diplomacy** game with **Hermes 3 405b** and broke it down in [this X post](https://x.com/alxai_/status/1931360264726929749).
   - The game apparently shows interesting thoughts on **AI and reasoning models**, according to the original poster in [this X post](https://x.com/ditpoo/status/1931338264079999204).
- **AI Reasoning models get deep dive**: Several posts talked about **AI and reasoning models** in the context of AI Diplomacy.
   - It included links to [this tweet](https://x.com/wolfejosh/status/1931182279755178074), [this X post](https://x.com/ditpoo/status/1931339719927120088), [this ArXiv paper](https://arxiv.org/pdf/2506.06261), [this YouTube video](https://youtu.be/tiZFewofSLM), and [this HuggingFace Trainer doc](https://huggingface.co/docs/transformers/trainer).


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1381164275330060380)** (2 messages): 

> `KV Compression Method` 


- **New KV Compression Method Dropped**: A member shared a link to a new **KV compression method** [paper](https://arxiv.org/pdf/2505.23416) on arXiv.
   - Another member expressed interest in implementing it, linking to a [tweet](https://x.com/theturingpost/status/1931432543766847887?s=46) about it.
- **Eager Implementer Seeks Collaboration**: One member asked for collaborators to implement the proposed **KV compression** method.
   - The call to action was associated with the promise of streamlined storage and retrieval in AI models.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1380625006673002516)** (125 messages🔥🔥): 

> `GPT-4 vs Claude for coding, Manus Credit Disappearance, AI UI/UX design limitations, GTA 8 Release Date, Open invitation email` 


- **Claude outperforms GPT-4 for Coding**: Members debate the superior AI model for coding, with some arguing that [**Claude 4.0**](https://www.anthropic.com/index/claude-4-haiku) excels in coding, reasoning, and math due to its better AI engine and training.
   - However, others point to the **AI arena leaderboard** that indicates **ChatGPT** might be more suitable for web development, and members cite **Manus's** disappointing code generation capabilities.
- **Manus Credits Suddenly Disappear**: A member reported a sudden loss of credits, going from nearly **30,000** to only **2,300**, leading to speculation about potential reasons such as [fraud or exploitation of the sharing system](https://help.manus.im/en/).
- **AI Faces UI/UX Design Bottleneck**: Members discussed that while AI can generate basic code, complex tasks such as **UI**, **design**, and **logic** still heavily rely on human developers, which limits AI's ability to create comprehensive projects like *'make me GTA 8'.*
- **Speculation for GTA 8's Launch Date**: Members jokingly predicted that **GTA 8** might be created by AI around *February 23, 2033*, with others agreeing it's only a matter of time before AI can develop such complex games, assuming no global catastrophe occurs.
   - One member jokingly stated that [builder.ai](https://builder.ai/) can do it.
- **YouTube's Anti-Bot Measures Thwart Manus**: Members report that **Manus** can no longer watch YouTube videos due to YouTube's bot-detection feature, which is actively patching its anti-bot mechanisms and is keen about this, so now **Manus** cannot log into **Gmail** accounts since it's a sandbox.
   - A Manus team member acknowledged the issue as a tech problem and said that *they would try to fix it this week*.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1380943495585599639)** (3 messages): 

> `DumPy, TPDE` 


- **Dynomight's DumPy garnering interest**: [DumPy](https://dynomight.net/dumpy/) is gaining traction as a tool of interest within the community.
   - It was specifically called out for its mention of **einx** and **torchdim**.
- **TPDE: LLVM backend 10-20x faster than O0**: The community is excited about [TPDE](https://discourse.llvm.org/t/tpde-llvm-10-20x-faster-llvm-o0-back-end/86664), which is an **LLVM** backend 10-20x faster than **LLVM O0**.
   - A member asked for future links to be posted to the dedicated channel for links.


  

---


### **Modular (Mojo 🔥) ▷ #[announcements](https://discord.com/channels/1087530497313357884/1098765954302873621/1380994125096026163)** (2 messages): 

> `Modular Forum Navigation, Community Meeting, Mojo in Bioinformatics, Accelerating Particle Physics with Mojo` 


- **Modular Forum Revamps Navigation**: A new navigational system is being introduced to the [Modular Forum](https://forum.modular.com/t/docs-site-new-navigational-system/1598), and feedback is requested.
- **Community Meeting Kicks Off**: A community meeting is starting in approximately 20 minutes via Zoom, covering **Mojo in Bioinformatics** and **Accelerating Particle Physics with Mojo**.
   - The Zoom meeting can be joined via [this link](https://modul.ar/community-meeting-zoom).


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1380624640707399811)** (103 messages🔥🔥): 

> `macOS vs Linux for development, Mojo compile-time slicing with parametric __getitem__, Custom decorators in Mojo, Linear types in Mojo, Mojo parameterization syntax` 


- **macOS versus Linux and WSL**: A member argued that **macOS** has disadvantages compared to **WSL** for development, citing the lack of a built-in package manager, different command syntax, Xcode dependency, poor Docker performance, and Nvidia GPU unavailability.
   - Other members countered that macOS is a better all-rounder, needing less environment tinkering, and that the core Torch devs use macOS; another mentioned macOS's hardware limitations regarding performance analysis, which is better on **Linux** or **WSL**.
- **Mojo's parametric slicing issue surfaces**: A user reported a strange issue with compile-time slicing of custom vectors in **Mojo** when using parametric `__getitem__`, providing a [code snippet](https://github.com/modular/modular/issues/4773) to reproduce the issue.
   - Another member suggested there might not currently be a way to disambiguate compile-time and runtime slice indexing, and the user filed a [bug report](https://github.com/modular/modular/issues/4773) for the issue, later clarifying that the issue is with comparing origins of types.
- **Linear Types in Mojo?**: A member suggested extending the trait-based compiler synthesis to linear types, preventing implicit `__del__` unless the type implements a trait like `LinearType`.
   - The code sample that was shared defined a `LinearType` alias to `UnknownDestructibility` to prevent implicit destructibility for custom types.
- **Custom Decorators for Mojo are MIA**: A member asked about plans for supporting the creation of custom decorators in **Mojo**, as the documentation indicates it's not yet supported.
   - No response was recorded.
- **Mojo's Parameterization Syntax Pain Points**: A member expressed concern over **Mojo's parameterization syntax**, finding it gnarly and difficult to read, especially with many parameters controlling non-trivial behavior.
   - Another member noted the influence of **Python's** generic syntax and said that being able to offload runtime computations into compile time is useful in a bunch of ways referencing [EmberJson](https://github.com/bgreni/EmberJson/pull/40).


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1380674937706123365)** (78 messages🔥🔥): 

> `Claude Subtweet, AI Model Quantization Transparency, Suno Restrictions, Linear MCP and Claude Code, AI Reasoning Models` 


- **Call for AI Model Quantization Transparency Revives**: A thread discusses the need for AI service providers to disclose the **quantization levels** of their models and notify users of any dynamic adjustments, pointing to a possible **Claude subtweet** [here](https://x.com/_xjdr/status/1931068996092334274).
   - The community proposes solutions like quantization-sensitive evaluations and public web pages detailing current quantization levels, with calls for fair compensation for degraded service, and industry standards for verifiable inference and serving transparency, detailed [here](https://x.com/TheAhmadOsman/status/1930944597464654272).
- **Suno's Copyright Claim Complications Sound Off**: A member noted that **Suno** has restrictions unless you maintain an active subscription, contradicting claims of *'no copyright, no restrictions'*.
   - While enforcement might be challenging, this clarification ensures users are aware of the [Suno's Terms](https://www.reddit.com/r/LocalLLaMA/comments/1l4mgry/chinas_xiaohongshurednote_released_its_dotsllm/) of service.
- **Linear MCP and Claude Code Integration Boasts**: A user shared an integration of **Linear MCP** to make task lists and project stateful between **Claude Code** sessions, running locally and handling OAuth, exposing Linear as a *stdio* MCP server that Cursor understands, explained [here](https://www.task-master.dev/).
   - The user noted that *'my entire claude.md file is now basically just a system prompt on how to use the linear mcp'* leading to a GitHub integration with sub-agents triggered by Linear assignments.
- **Apple's Research Questions AI Reasoning**: Apple's research suggests that leading AI 'reasoning' models like **Claude**, **DeepSeek-R1**, and **o3-mini** do not genuinely reason but rather excel at pattern memorization, detailed [here](https://x.com/RubenHssd/status/1931389580105925115?s=19).
   - Models consistently fail at higher complexity problems, even with explicit instructions, challenging the hype around imminent AGI.
- **AI Companies Underestimate LLM's Content Filter Flaws**: The thread discusses how AI companies misunderstand LLMs, particularly regarding content filters, detailed [here](https://x.com/aiamblichus/status/1931487839822254358?s=46).
   - Users share examples of humorous and unexpected LLM responses that bypass filters with slight prompt tweaks, suggesting LLMs operate on 'improv' rules.


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1380740345868058674)** (57 messages🔥🔥): 

> `MCP Server Publishing, Image Uploading with MCPs, Accessing GitHub MCP Server, MCP and SSE vs. WebSockets, MCP Resources and Prompts` 


- ****Image Uploading** Struggles with MCPs**: Members are struggling to get image uploading working with MCPs, including attempting to pull from **Cursor's context** and **base64 encode** without success.
- **GitHub MCP Server Access using Python**: A user requested guidance on accessing the official **GitHub MCP server** using **Python** to read files and directories, and was directed to the [installation instructions using Docker](https://github.com/github/github-mcp-server?tab=readme-ov-file#installation).
- **Deciphering MCPs Decision Trees**: A member sought resources on how to have the agent request different prompts sequentially, and the suggestion was made to implement a **decision tree** to handle such workflows.
- **Client Reconnections: HTTP Errors**: When a **server restarts** and clients connect with old session IDs, clients get stuck on **HTTP 400 or 404** errors, due to clients not starting a new session after the error.
   - The MCP spec says clients *MUST* start a new session on 404 errors, but in practice most clients **do not conform to the spec**.
- **MCP Doesn't force Opensource**: It was clarified that MCP server and client implementations are **not required to be open source**, despite the wording on the mcp.io website.


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1380932048335802408)** (11 messages🔥): 

> `OAuth support in MCP, Slack's official MCP server, glama.ai blog post, MCP specification utility, Google MCP server` 


- ****OAuth Odyssey**: MCP's Spec Support**: A member inquired whether MCP supports native **OAuth** from spec, with the creator responding that it wasn't initially built with OAuth standards but welcomes contributions.
   - Slack's [official MCP server](https://www.slack.dev/secure-data-connectivity-for-the-modern-ai-era) was mentioned as potentially supporting OAuth, but it appears the replacement server hasn't launched yet.
- ****Specification Sleuth**: Utility Tool for MCP**: A member created a utility to extract content from the "Specification" documentation pages of MCP, reducing the file size by about a third compared to the full `llms.txt`/`llms-full.txt` files.
   - The utility's output, containing only the specification website pages, is available on [Gist](https://gist.github.com/hesreallyhim/d974990b8c80cf6f32b88bfe39b76f9a), while another Gist includes the `schema.ts` and `schema.json` files.
- ****Google's Guardian**: MCP Server with Security Focus**: A member shared their [Google MCP server](https://github.com/robcerda/google-mcp-server), emphasizing its security-first design using only secure scopes by default.
   - They highlighted that the server can manage most of **Gmail, Calendar, and Drive** from the MCP itself, and are seeking suggestions for improvement.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1380892809355329598)** (31 messages🔥): 

> `Lazy setitem in tinygrad, tinygrad meeting #74, Lovely Grad working with tinygrad, huggingface models on tinygrad, True float16` 


- **Lazy `__setitem__` sought for TinyGrad tensors**: A contributor questioned why `__setitem__` in `tensor.py` calls `realize()`, making it non-lazy, and suggested splitting it into `setitem_and_realize` to allow for lazy, immutable, on-device operations, which could benefit examples like [beautiful_cartpole](https://github.com/tinygrad/tinygrad/blob/master/examples/beautiful_cartpole.py).
   - The user noted that *the current realize() has to be removed* for the suggested lazy implementation to work.
- **TinyGrad Meeting #74 Recap**: The meeting covered company updates, including fixes to multi and resnet dataloader, faster CI, linearizer, viz, drivers, cloud/hash, onnx, and local developments, as well as other bounties like **lm_eval** and **AMD_LLVM**.
   - GeorgeHotz stated that he will get *everything merged this week*.
- **`lovely-grad` revived for modern TinyGrad**: [Lovely Grad](https://github.com/xl0/lovely-grad) is now working with modern tinygrad after being broken for several months, with plans to investigate remote testing with pytest multiprocessing.
   - This tool helps visualize the gradient flow of neural networks implemented in TinyGrad.
- **Hugging Face Models to be integrated into `test_onnx.py`**: Testing of Hugging Face models will be rewritten to integrate into `test_onnx.py` for model tests.
   - This aims to consolidate model testing within the `test_onnx.py` framework.
- **Metal faces buggy bounds issues**: Metal is reported to have compiler bugs, with one user wasting half a day on bounds issues, prompting the addition of `max_total_threads_per_threadgroup` to address CUDA's `__launch_bounds__` and HIP's `amdgpu_flat_work_group_size`.
   - The user was shocked that *this turns beautiful mnist into beautiful macos dos poc* due to driver issues.


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1380932650612817942)** (21 messages🔥): 

> `Tensor Indexing, FUSE_ARANGE effect, ProcessPoolExecutor issues` 


- **Tensor Indexing Questioned**: A member questioned why **tinygrad** doesn't have straightforward tensor indexing operations at the Tensor abstraction layer.
   - They noted that `gather`, `scatter`, `getitem` etc are all implemented through complicated masked assignments, and the only options that don't involve numpy is going lower level to UOps.
- **FUSE_ARANGE saves the Day**: Using `FUSE_ARANGE=1` context, a member demonstrated a **10x speedup** in tensor indexing operations.
   - Another member was curious what exactly the `FUSE_ARANGE` does, and if they could get away with using it in `examples/hlb_cifar10.py`.
- **Child Process Device Refusal**: A member ran into an issue with a recent **tinygrad** change ([PR#4425](https://github.com/tinygrad/tinygrad/pull/4425)) that refuses to open a device from a child process, specifically when using `nbdev_test` which relies on `ProcessPoolExecutor`.
   - They questioned whether it's reasonable for **tinygrad** to fail when used from a `ProcessPoolExecutor`.


  

---


### **LlamaIndex ▷ #[announcements](https://discord.com/channels/1059199217496772688/1073670729054294197/1381647788613566696)** (1 messages): 

> `Office Hours, Form Filling Agents, LlamaIndex MCP Servers, MCP Dev Summit, Spreadsheet Agent` 


- ****Office Hours** Schedule Set**: The next office hours session will be held on **Thursday, June 12th**, at **8AM PT/5PM CET** and will focus on **MCP**, **form filling**, and other LlamaIndex topics.
   - The office hours will be hosted in the general voice channel by two members.
- ****MCP Dev Summit** Presentation Available**: A talk was given at the **MCP Dev Summit** about the **13 different protocols** currently vying to become the standard way for agents to talk to tools and each other, including **MCP**, **A2A**, **ACP** and many more, watch [here](https://www.youtube.com/watch?v=kqB_xML1SfA).
   - The video is now available for viewing.
- ****Spreadsheet Agent** in Private Preview**: The **Spreadsheet Agent** is in private preview, using a *Parse First, Reason Second* architecture that understands visual structure and context, as detailed in [this blogpost](https://www.llamaindex.ai/blog/introducing-the-spreadsheet-agent-in-private-preview).
   - This agent is ready to preview now.
- ****Multi-Turn Conversation Memory** Implementation**: There is a new memory implementation to handle multi-turn conversations, with an example available [here](https://docs.llamaindex.ai/en/stable/examples/memory/custom_multi_turn_memory/).
   - This update was implemented by a member.
- ****Ollama 'thinking' Feature** Supported**: LlamaIndex now supports the Ollama 'thinking' feature, see the [merged PR here](https://github.com/run-llama/llama_index/pull/18993).
   - This enhancement allows for more advanced and nuanced interactions.


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1380622774673871040)** (7 messages): 

> `Vector Databases, Llama Cloud, Agentic Extraction Workflow, MCP Servers, AI Summit` 


- **Vector DBs to Boost RAG Pipelines**: On June 12th, an Open Source Engineer will present *The Hitchhiker's Guide to Vector Databases* at the BASED Meetup in Munich to discuss best practices for boosting RAG pipelines.
   - The talk will cover everything from **data preparation** to **query optimization**.
- **Llama Cloud Unveiled**: A new video showcasing an overview of Llama Cloud highlights its ecosystem and core tools for building production-grade LLM applications.
   - A walk-through of the landscape is given by @tuanacelik in [this video](https://t.co/kIPbq542Pr).
- **Agentic Extraction Workflow**: A tutorial will be shared this weekend demonstrating how to build an agentic extraction workflow using LlamaIndex over a **Fidelity Multi-Fund Annual Report**.
   - The workflow is used to extract a list of [multiple funds](https://t.co/RpmHYV4UDN), with each fund reporting multiple tables of financial data.
- **Discord Office Hours**: Another office hours will be hosted in the LlamaIndex Discord with @tuanacelik and @LoganMarkewich.
   - This session will focus on creating **MCP servers** with LlamaIndex and building **form filling agents**, accessible via [this link](https://t.co/CLAauUeFty).
- **Databricks AI Summit**: LlamaIndex will be at the Databricks Data + AI Summit, offering a chance to connect and explore the future of AI.
   - Attendees can meet CEO Jerry Liu and AI engineers at Booth D117 in the AI Pavilion to learn about building with [LlamaIndex](https://t.co/mbCkazR18g).


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1380781155368046612)** (21 messages🔥): 

> `Gemini 2.5 streaming output, ACP protocol, Agent workflow termination, Plan upgrade issue, LlamaParse product down` 


- **Gemini's "Thinking" Needs Separation**: A member inquired about separating the "thinking" text from the actual response when streaming output from a model like **Gemini 2.5**.
   - Another member mentioned that a **PR might be needed** to support this, as the current implementation doesn't check the "type" of `Part` from Google while streaming and pointed to work done in [this PR](https://github.com/run-llama/llama_index/pull/18993).
- **A New Protocol Comes to Town: ACP**: A member noticed a new protocol in generative AI (**ACP**) and created a LlamaIndex example with [this pull request](https://github.com/i-am-bee/acp/pull/176).
   - They requested a review of the file.
- **Agent Workflow Halt Solved by Prompting**: A member reported their **agent workflow** was terminating after the transcription_agent output, despite having `can_handoff_to` specified.
   - Another member suggested that it was likely not "terminated" and that *adjusting prompts/descriptions* could change the behavior.
- **Subscription Snafu: Upgrade to Pro Fizzles**: A member reported an issue where upgrading their plan from **Starter to Pro** resulted in being shown as still on the **Free plan**.
   - They attached screenshots as evidence.
- **LlamaParse Goes Down!**: A member inquired if a product relying on **LlamaParse** was down, seeking contact information for such incidents and mentioned that *someone responded via the support email*.
   - They followed up to confirm that their product was indeed down.


  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1380923564328222875)** (1 messages): 

> `RAG, Llama Index, ReactAgent, Sparse Data` 


- **Members Discuss Sparse RAG Retrieval Challenges**: A member reports a decent **RAG setup** using **Llama Index** with a **ReactAgent** and some retriever tools, but is running into issues with sparse data retrieval.
   - They claim to have over **1000 documents** referring to certain principles and are wondering about a better way to ensure they do not miss important information without resorting to a very high **K retrieval value**.
- **Brainstorming Session Initiated for Sparse Data Handling in RAG Systems**: Following the initial problem statement, the discussion anticipates potential solutions for handling sparse data more effectively within **Retrieval-Augmented Generation (RAG)** setups.
   - The conversation is geared towards finding methods that avoid high **K retrieval values**, suggesting an interest in precision and efficiency in information retrieval.


  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1380639317856354466)** (21 messages🔥): 

> `Issue #2470, Clipping logprobs, Adagrad errors on nightly` 


- **Issue #2470 Needs Some Love**: A member highlighted [issue #2470](https://github.com/pytorch/torchtune/issues/2470) that has been waiting for attention since March.
   - Another member inquired whether it was previously discussed, initiating a debate about its priority.
- **Debate on Clipping Logprobs Feature**: A member proposed adding a feature for clipping logprobs, which led to discussions about its necessity and implementation in TorchTune.
   - While the feature exists in other repositories, concerns were raised about maintaining it and the potential complexity of correctly exposing it to users.
- **Adagrad Fails with DeviceMesh Assertion on Nightly**: A user reported an `AssertionError: found no DeviceMesh from dtensor args for aten._fused_adagrad_.default!` when using **fused Adagrad** on the nightly build with specific configurations.
   - It seems that **SGD** started to work after switching to the latest TorchTune, but the root cause of the Adagrad issue remains unclear, with attempts to reproduce it being unsuccessful.


  

---


### **Cohere ▷ #[🧵-general-thread](https://discord.com/channels/954421988141711382/954421988783444043/1380673342092345387)** (10 messages🔥): 

> `Cohere AI, Cohere support, command-a, documentation` 


- **Apply to Cohere AI!**: To get access to **Cohere AI**, a user suggested to apply via [this form](https://share.hsforms.com/10OrjljwpQ52ILJA6ftENIwch5vw).
   - They were happy to help.
- **Cohere support improves with command-a bot**: A user announced that the channel <#1381756280716132412> provides faster support using **command-a**, which answers questions using the documentation from the Cohere website.
   - The bot cannot help with account problems, API problems, etc., and will only be active when the user is online during the beta phase; misuse will result in an instant ban.


  

---


### **Cohere ▷ #[📣-announcements](https://discord.com/channels/954421988141711382/996880279224451154/1381776718456426517)** (1 messages): 

> `North Integration, GameWarden, Partnerships, Security Deployments` 


- **North Joins Forces with GameWarden for Security**: **North** is now integrated with the **GameWarden** platform via a partnership with **Second Front**, for secure deployments in high-security environments, as described in [this X post](https://x.com/1vnzh/status/1930298055099613307).
- **GameWarden Platform Gets Enhanced Security**: Service members gain unprecedented effectiveness and speed against an ever-evolving threat landscape, now securely integrated with the full **GameWarden** platform.


  

---


### **Cohere ▷ #[🔌-api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1380756761018892378)** (5 messages): 

> `Cohere's r7b model, Cohere signup issues, Cohere open source contributions, Cohere Developer Experience GitHub repository` 


- **r7b outputs 1 T/s!**: A user stated that **Cohere's r7b model** is outputting around **1 T/s**.
   - No further details or context were provided regarding the specifics of this performance metric.
- **Google Cloud Marketplace signup broken**: A user encountered an error when trying to sign up for **Cohere** through the **Google Cloud Marketplace**.
   - The error message was: *{"code":"invalid_vendor","description":"The vendor '8SAJ2US' does not exist"}*, with the vendor issue possibly coming from [this url](https://upstream-api.tackle.io/v1/gcp/order/8SAJ2US/cohere.endpoints.cohere-id-public.cloud.goog).
- **Cohere support requests email**: A member advised that anyone with the error issue on **Google Cloud Marketplace** should email [support@cohere.com](mailto:support@cohere.com) with the details of the problem.
   - They mentioned including the **error message** and **steps taken so far** in the email.
- **Cohere accepts GitHub contributions!**: **Cohere** has an open-source repository that accepts contributions at the **[Cohere Developer Experience GitHub repository](https://github.com/cohere-ai/cohere-developer-experience/)**.
   - While contributions are welcome, the **OpenAPI specs** and **snippets** are synced from internal repositories, and the team will handle those changes.


  

---


### **Cohere ▷ #[👋-introduce-yourself](https://discord.com/channels/954421988141711382/1346635816629178410/1380673786013290606)** (3 messages): 

> `Introductions, Discord App Development, AI and Machine Learning` 


- **Welcoming New Community Member**: A new member named David introduced himself, sharing his age, pronouns, and interests, and that he is turning **19** soon.
   - David expressed a positive message: *you are valid and you are loved*.
- **New Member Develops Discord App**: David is currently working on **link safe**, which he describes as a *Discord app*.
   - He also shared his enthusiasm for **AI and machine learning**, describing it as *a magical world*.
- **Community Support Sought**: David hopes to gain support from the Cohere community.
   - He was *not sure if I posted this already*.


  

---


### **Cohere ▷ #[🔔-ping-settings](https://discord.com/channels/954421988141711382/1346642216541622343/)** (1 messages): 

competent: Moved to <id:customize>
  

---


### **Nomic.ai (GPT4All) ▷ #[announcements](https://discord.com/channels/1076964370942267462/1090471714888102009/1380978691127120054)** (1 messages): 

> `Nomic team updates` 


- **Nomic Team Preps Exciting Updates**: The **Nomic team** has been heads down the last few months working on some exciting updates.
   - While specifics remain under wraps, the team asks for patience as they gear up for a future launch.
- **Patience Requested for Future Launch**: The Nomic team acknowledges the community's anticipation for new developments.
   - They are working diligently towards a launch but request continued patience until everything is ready.


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1380685905706356826)** (15 messages🔥): 

> `save chats in plain text, Nomic team updates, GIGABYTE server, nomic-embed-text-v1.5` 


- **User requests Save-Chat Feature**: A user requested a feature for **GPT4All** to *save chats in plain text* in a unique directory, suggesting it would enhance LocalDocs RAG Search for memory.
- **Nomic Team Cooking Up Updates**: A member mentioned the **Nomic team** has been working on *exciting updates* but they're not ready for launch yet.
- **GIGABYTE Server May Be Offered Barebone**: A user asked if the [GIGABYTE server](https://www.gigabyte.com/Press/News/2293) may be offered as a barebone while waiting for **GPT4ALL** upgrades.
   - They speculated it could run **Mixtral 8x22B** at record speeds, aligning with the trend toward **MOE models**.
- **Question on nomic-embed-text-v1.5 Usage**: A user asked if they can still use **nomic-embed-text-v1.5** from nomic cloud next month, and included an [attached image](https://cdn.discordapp.com/attachments/1090427154141020190/1381780899716399145/image.png?ex=6848c33e&is=684771be&hm=7713e72607a3b6445cf9a1cfd28fc026127c79b6bf40f539e8edd0edb0b80bf8).


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1380655104138678393)** (4 messages): 

> `Transfer Posttraining Learning, Blockchain Expertise, AI Agent Devs` 


- **Noob Asks: Can post-training learning be transferred?**: A new member asked if it is possible to transfer posttraining learning from one model to another without repeating the learning, like finetuning or RL.
   - The question sparked a discussion about different approaches to transfer learning in the context of AI models.
- **AI/Blockchain Dev boasts expertise**: A software engineer with experience in **Blockchain (EVM, Solana, Cardano, Hydra, Aptos, Cosmos, Tron, zk-SNARKs)** and **AI (LLM, NLP, LangChain, AutoGen,TorchRL, DL, Azure ML, AI Agent)** volunteered to work.
   - The engineer also has experience in **Web systems (React, Next, Vue, Node, IPFS, Pinata API)** and offered their availability and contact information.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[hackathon-announcements](https://discord.com/channels/1280234300012494859/1280236929379602493/1381729569580646501)** (1 messages): 

> `Agentic AI Summit, Early Bird Tickets, UC Berkeley, Vinod Khosla, Ion Stoica` 


- **Agentic AI Summit coming to UC Berkeley**: The Agentic AI Summit is coming to **UC Berkeley** on **August 2, 2025**, building on the popular **LLM Agents MOOC** and expecting **1,500+** in-person attendees.
   - Featured speakers include **Vinod Khosla** (Khosla Ventures), **Ion Stoica** (Databricks and Anyscale), **Dawn Song** (UC Berkeley), and many more, with keynotes, panel discussions, and workshops planned.
- **Early Bird Tickets are Almost Gone!**: Early bird tickets for the Agentic AI Summit end on **June 30, 2025**, with student passes at **$25**, startup passes at **$60**, and industry professional passes at **$80**.
   - Students and indie developers can apply for fee waivers, according to the [Summit website](https://rdi.berkeley.edu/events/agentic-ai-summit).


  

---


### **MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/)** (1 messages): 

sebash6677: hi
  

---


### **Gorilla LLM (Berkeley Function Calling) ▷ #[leaderboard](https://discord.com/channels/1111172801899012102/1214705495974092810/1381619522146144266)** (1 messages): 

> `Leaderboard Updates, Project Continuation` 


- **Leaderboard Updates on Hold, Project Future Unclear**: A user inquired about why the leaderboard updates have stopped.
   - They also asked whether work on the project would continue.
- **Project Future Still in Limbo**: The user specifically mentioned <@856060858462502922> in their query.
   - No response or clarification was provided in the given context.


  
