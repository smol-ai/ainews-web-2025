---
id: MjAyNS0w
title: Bartz v. Anthropic PBC — "Training use is Fair Use"
date: '2025-06-24T05:44:39.731046Z'
description: >-
  **Anthropic** won a significant fair use ruling allowing the training of
  **Claude** on copyrighted books, setting a precedent for AI training legality
  despite concerns over pirated data. **Replit** achieved a major milestone with
  **$100M ARR**, showing rapid growth. **Delphi** raised **$16M Series A** to
  scale digital minds, while **Thinking Machines Lab** focuses on reinforcement
  learning for business applications. **Disney** and **Universal** sued
  **Midjourney** over unauthorized use of copyrighted images. **Google
  DeepMind** released **Gemini Robotics On-Device**, a compact foundation model
  for robotics.
companies:
  - anthropic
  - replit
  - delphi
  - sequoia
  - thinking-machines-lab
  - disney
  - universal
  - midjourney
  - google-deepmind
models:
  - claude
  - gemini-robotics-on-device
topics:
  - fair-use
  - copyright
  - reinforcement-learning
  - foundation-models
  - robotics
  - funding
  - lawsuit
  - digital-minds
  - model-release
people:
  - andrea_bartz
  - giffmana
  - andrewcurran_
  - amasad
  - swyx
  - hwchase17
  - krandiash
  - daraladje
  - steph_palazzolo
  - corbtt
  - demishassabis
---


**An important ruling, but not a final one.**

> AI News for 6/23/2025-6/24/2025. We checked 9 subreddits, 449 Twitters and 29 Discords (220 channels, and 3440 messages) for you. Estimated reading time saved (at 200wpm): 365 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

Last August, a group of authors led by Andrea Bartz [brought a class action lawsuit](https://entertainmentlawreview.lls.edu/authors-v-anthropic-the-legal-showdown-over-ai-copyright-and-fair-use/) on Anthropic PBC for "illegally downloading" their works to train Claude. The scale of the destructive book scanning (perhaps [<$2 per book](https://twitter.com/giffmana/status/1937591844252385323) esp used books) is impressive:

![](https://resend-attachments.s3.amazonaws.com/S9Up265zT1VJ0VA)

This is of course familiar to anyone who knows [Authors Guild v Google](https://en.wikipedia.org/wiki/Authors_Guild,_Inc._v._Google,_Inc.), aka the Google Books lawsuit, which had a very similar setup, but this is the first direct ruling on the legality of pretraining on copyrighted content.

The filings from that case are [here](https://www.courtlistener.com/docket/69058235/bartz-v-anthropic-pbc/) but the result today is from the [Motion for Summary Judgment](https://www.courtlistener.com/docket/69058235/231/bartz-v-anthropic-pbc/), where Anthropic [arguably](https://x.com/mjbommar/status/1937562175955980614) "won" with the explicit ruling that "training use [is] fair use".

![](https://resend-attachments.s3.amazonaws.com/y4sXpTlxUVp0VvI)

It seems that [the ghost of Books3](https://www.wired.com/story/battle-over-books3/) haunts Anthropic as there is a separate issue on using pirated books, but the judgment is pretty clear here and likely sets an important precedent for years to come: no less than 32 mentions of how "transformative" a use case that pretraining is, regardless of how much the LLM memorizes:

![](https://resend-attachments.s3.amazonaws.com/RcxQDzqlkw3F10a)

---

# AI Twitter Recap

**Companies, Funding, and Legal**

- **Anthropic Wins Fair Use Ruling on Book Training Data**: A federal judge has ruled that **Anthropic's** use of books to train **Claude** constitutes **fair use**, a significant decision for the AI industry. The ruling, [shared by @AndrewCurran_](https://twitter.com/ClementDelangue/status/1937519434312147374), distinguishes the act of training from the method of acquiring the data. Discussions highlighted that [the method of obtaining the books (piracy) was a separate issue](https://twitter.com/giffmana/status/1937551619937436101), and that the cost of digitizing books is surprisingly low if they are destroyed in the process, as [noted by @giffmana](https://twitter.com/giffmana/status/1937591844252385323).
- **Replit Reaches $100M ARR**: [@amasad](https://twitter.com/Hacubu/status/1937263659581079581) announced that **Replit** has crossed **$100M in ARR**, a significant increase from **$10M** at the end of 2024. The rapid growth was described by [@swyx](https://twitter.com/swyx/status/1937300296386117661) as a chart resembling a "superintelligence" takeoff curve, with others like [@hwchase17](https://twitter.com/Hacubu/status/1937275206789427625) praising the team's "fantastic execution."
- **Delphi Raises $16M Series A to Scale Human Expertise**: **Delphi**, a platform for creating "digital minds" to scale expertise, [announced a **$16M Series A** round led by **Sequoia**](https://twitter.com/krandiash/status/1937574873154617751). The goal is to [make human knowledge accessible and discoverable](https://twitter.com/daraladje/status/1937645599823921504), with over **2,000 digital minds** already created.
- **Thinking Machines Lab's Mission Described as "RL for Businesses"**: Mira Murati's new AI startup, **Thinking Machines Lab**, is being described by investors as focusing on "**RL for businesses**," [according to a report from The Information shared by @steph_palazzolo](https://twitter.com/steph_palazzolo/status/1937284120062706004). This aligns with the "hot RL summer" trend, [as noted by @corbtt](https://twitter.com/corbtt/status/1937624653662744840).
- **Disney and Universal Sue Midjourney Over Copyright**: [@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1937314755066171580) reports that **Disney** and **Universal** have filed a lawsuit against **Midjourney**, alleging the image generation company trained its models on their copyrighted content without permission. The suit claims the system generated unauthorized images of characters like **Spider-Man** and **The Simpsons**.

**Model & Tech Releases & Updates**

- **Google Releases Gemini Robotics On-Device**: **Google DeepMind** has launched **Gemini Robotics On-Device**, a foundation model small enough to run directly on a robot. [@demishassabis](https://twitter.com/demishassabis/status/1937526283161809056) highlighted its speed and performance even in low-connectivity environments. The release includes [an on-device VLA and open-sourced tools and models](https://twitter.com/GoogleDeepMind/status/1937535740206022773) to facilitate development.
- **PrimeIntellect Launches SYNTHETIC-2 Dataset**: **PrimeIntellect** [announced **SYNTHETIC-2**](https://twitter.com/ClementDelangue/status/1937511681850044894), their next-generation open reasoning dataset and a "planetary-scale synthetic data generation run," powered by **9 different models**.
- **Deepseek Uses Nous Research's YaRN for Context Extension**: [@Teknium1](https://twitter.com/Teknium1/status/1937373884610936854) pointed out that **Deepseek** utilizes the **YaRN** method, developed by **Nous Research**, to extend its context length.
- **Kling AI Enhances Video Generation**: **Kling AI** has introduced several new features, including support for [saving creations as **Live Photos** for dynamic wallpapers](https://twitter.com/Kling_ai/status/1937343208515924465) and a new "**SurfSurf Effect**" for creative video editing, [accompanied by a user contest](https://twitter.com/Kling_ai/status/1937393240225063042).
- **Hugging Face Releases VideoPrism for Video Embeddings**: [@osanseviero](https://twitter.com/osanseviero/status/1937560015348597124) announced the release of **VideoPrism**, a new model for generating video embeddings useful for tasks like classification, video retrieval, and localization. The model, paper, and code are available on **Hugging Face**.
- **PufferLib 3.0 Release for Large-Scale RL**: **PufferLib 3.0** has been released, enabling reinforcement learning training on massive datasets. The team demonstrated training agents on **1 Petabyte (12,000 years) of data** with a single server, [as shared by @slashML](https://twitter.com/slashML/status/1937480613029904640).
- **Warp 2.0 Launches as an "Agentic Development Environment"**: **Warp** has launched version 2.0, billed as an **Agentic Development Environment**. It claims the [#1 spot on **Terminal-Bench** and **71%** on **SWE-bench**](https://twitter.com/_akhaliq/status/1937542375179448828).
- **LlamaBarn Sneak Peek**: [@jeremyphoward](https://twitter.com/jeremyphoward/status/1937290259307573649) shared a sneak peek of **LlamaBarn** from Georgi Gerganov, which humorously, [does not feature any Llama models](https://twitter.com/teortaxesTex/status/1937628449708933220).
- **Jina AI Releases v4 Embeddings**: A new version of **Jina embeddings** has been released, representing a significant upgrade. The model is scaled up from **RoBERTa** to **Qwen 2.5**, is multimodal, and supports **COLBERT-style** multi-vector representations, [as highlighted by @nrehiew_](https://twitter.com/nrehiew_/status/1937357675072778567).

**New Techniques & Research**

- **Mustafa Suleyman Proposes "Chain of Debate"**: **Inflection AI** CEO [@mustafasuleyman](https://twitter.com/mustafasuleyman/status/1937553061427445824) outlined the next evolution from Chain of Thought: "**Chain of Debate**." This concept involves multiple models "discussing out loud, debating, debugging, deliberating," moving from a single AI to multiple AIs collaborating.
- **Sakana AI Introduces Reinforcement-Learned Teachers (RLTs)**: **Sakana AI** [@SakanaAILabs](https://twitter.com/AndrewLampinen/status/1937261400419885414) unveiled **Reinforcement-Learned Teachers (RLTs)**, a new method that uses reinforcement learning to transform how LLMs are taught to reason.
- **High Sample Efficiency Shown in RL for Agentic RAG**: [@corbtt](https://twitter.com/corbtt/status/1937594932040204483) shared exciting experimental results showing incredible sample efficiency with RL. Using **GRPO** to train a modified **ART-E** (agentic RAG task), they found that **qwen2.5-14b** could exceed **Gemini 2.5 Flash** performance with just **1 training scenario** and outperform **o3** with only **16 scenarios**.
- **The NetHack Learning Environment Remains Unsolved After 5 Years**: On the fifth anniversary of its release, the **NetHack Learning Environment** remains a grand challenge for AI. [@_rockt](https://twitter.com/_rockt/status/1937480864243331396) noted that current frontier models only achieve **~1.7% progression**, highlighting its difficulty.
- **LLMs Can Be Programmed by Backprop**: A new paper on "**Programming by Backprop**," [shared by @_rockt](https://twitter.com/_rockt/status/1937507616000749888) and [@LauraRuis](https://twitter.com/_rockt/status/1937549094073041136), demonstrates that LLMs can learn to evaluate a program on various inputs by training on source code alone, without ever seeing I/O examples, acting as "fuzzy program interpreters."
- **Stanford's CS336 "Language Models from Scratch" Course Materials Released**: [@percyliang](https://twitter.com/lupantech/status/1937524295732986046) announced the conclusion of **Stanford's CS336** course, taught with **Tatsu Hashimoto** and others, and made all lecture notes, code, and materials publicly available.

**Frameworks, Tooling, and Infrastructure**

- **OpenAI Engineering Praised for ChatGPT Scalability**: [@sama](https://twitter.com/sama/status/1937514123912491317) praised the **OpenAI** engineering and compute teams for their "incredible work" in rapidly scaling to meet the massive customer demand for **ChatGPT**, noting they have handled a "2.5 year sprint with such grace."
- **Perplexity Finance vs. Bloomberg Terminal**: **Perplexity** CEO [@AravSrinivas](https://twitter.com/AravSrinivas/status/1937330521920737727) posted a comparison showing **Perplexity Finance** effectively analyzing MAG 7 stock growth, suggesting that "**AI is eating Legacy software**" like the **Bloomberg Terminal**.
- **LlamaIndex's Advanced Document Parsing**: [@jerryjliu0](https://twitter.com/jerryjliu0/status/1937302778122314202) showcased **LlamaIndex's** document parsing agent, which accurately rendered a complex combination chart from an equity research report into a clean table, a task where **Claude Sonnet 4.0** "hallucinated half the values."
- **"Context Engineering" as a New Trend**: The concept of "**context engineering**" is gaining traction. [@hwchase17](https://twitter.com/hwchase17/status/1937648042985030145) highlighted **LangGraph** as a great tool for this and proposed new features to streamline context management.
- **Azure Management from the Command Line with Cline**: **Cline** announced a new **Azure MCP Server**, allowing users to [control services like Storage, Cosmos DB, and Monitor using natural language](https://twitter.com/cline/status/1937324870393901539) directly from their CLI.
- **uv Performance and Python Speed**: The speed of **uv**, the Rust-based Python package installer, surprised developers. [@finbarrtimbers](https://twitter.com/finbarrtimbers/status/1937506437762286058) noted that they had assumed pip's slowness was due to network issues, not "pure python" speed limitations that could be optimized.

**Broader Implications & Community Discourse**

- **The Short Lifespan of Modern Technologies**: [@alexalbert__](https://twitter.com/alexalbert__/status/1937526135442874651) reflected on the surprisingly short lifespans of technologies that feel permanent, such as **Googling (~26 years)**, **manual coding (~75 years)**, and **manually driving (~120 years)**, suggesting their "expiration dates" are now visible due to AI.
- **Waymo's Rapid Expansion Projections**: [@fchollet](https://twitter.com/fchollet/status/1937498488352264666) predicted that **Waymo's** autonomous vehicle service will expand from covering **2-3%** of the US population today to **15% within a year** and over **50% in three years**.
- **AI and Cognitive Skills Debate Requires Philosophical Grounding**: [@random_walker](https://twitter.com/random_walker/status/1937483620630794382) argued that productive debate on AI's impact on cognitive skills requires familiarity with the **Extended Mind Thesis** and understanding why today's worries are different from **Plato's** concerns about writing eroding memory **2,400 years ago**.
- **The Nature of AI Research**: [@_jasonwei](https://twitter.com/_jasonwei/status/1937590298022150638) offered an insight into AI research, describing it as spending "a massive amount of compute on experiments to learn simple ideas." He suggests that deeply understanding a few of these simple ideas is what allows researchers to get "miles ahead of the rest of the field."
- **r/LocalLlama Subreddit Reopens**: After a period of being private, the popular **r/LocalLlama** subreddit is back online. The community's need for the forum was [voiced by @danielhanchen](https://twitter.com/danielhanchen/status/1937506709196419222), who later [announced its return](https://twitter.com/danielhanchen/status/1937607779977728394).
- **The Geopolitics of AI Infrastructure**: [@dylan522p](https://twitter.com/dylan522p)'s presentation on the geopolitics of AI infrastructure was recommended, with a key takeaway being that the **US will face an ~88GW power deficit by 2028** due to data center demand, equivalent to roughly **88 nuclear reactors** ([shared by @AymericRoucher](https://twitter.com/AymericRoucher/status/1937261555156156788)).

**Humor & Memes**

- **A New Progress Bar for Evolution**: [@DavidSHolz](https://twitter.com/DavidSHolz/status/1937574227785474326) reposted a popular meme showing the evolution of man, humorously commenting, "Nah, bro. That's a progress bar. That's windows update, but for meat."
- **Helix Neural Network**: [@adcock_brett](https://twitter.com/adcock_brett/status/1937577814875881889) shared a striking image of a DNA-like helix composed of neural network nodes.
- **Vibecoding**: The term "**vibecoding**" appeared in several contexts, from programming with [**Claude Code**](https://twitter.com/code_star/status/1937270682565660721) to a general philosophy of ["I vibe code therefore I am"](https://twitter.com/reach_vb/status/1937638934986785207).
- **Good Soup**: A screenshot of code with the simple caption "[good soup](https://twitter.com/code_star/status/1937268613662277862)" was shared, resonating with developers.
- **The Value of a Good Night's Sleep**: [@vikhyatk](https://twitter.com/vikhyatk/status/1937372906109190593) shared a personal anecdote: "i took a **$400k pay cut** to not have to conform to someone else's sleep schedule and it was **100% worth it**."
- **Algorithms Class vs. Leetcode Class**: [@vikhyatk](https://twitter.com/vikhyatk/status/1937378138423722061) quipped, "if your algorithms class is studying sorting algorithms instead of finite state transducers that's a leetcode class not an algorithms class".

---

# AI Reddit Recap

## /r/LocalLlama Recap

### 1. LocalLlama Subreddit Moderator Transition and Recovery

- [**Subreddit back in business**](https://i.redd.it/1sx7mwusnx8f1.jpeg) ([Score: 272, Comments: 139](https://www.reddit.com/r/LocalLLaMA/comments/1ljlr5b/subreddit_back_in_business/)): **The screenshot documents moderator actions on the subreddit, including the removal and addition of moderators, edits to spam filter settings, and content removals. These logs provide transparency and help the community assess the restoration steps after a leadership/account deletion event. The actions suggest an effort to restore normal subreddit operations and address any disruptions, such as shadow removal of posts/comments, by updating moderation and anti-spam configurations.** Comments request a summary megathread to catch up on missed developments and confirm that heavy moderation filters—which may have been responsible for widespread content suppression—were removed. There is also confusion about the sequence of events leading to the incident, with some users expressing uncertainty about broader technical news (e.g., AGI, GGUF releases) that may have been missed during the interruption.
    - A user notes uncertainty on whether any major breakthroughs like AGI or model releases—"did they release agi? ggufs?"—occurred during the subreddit downtime, highlighting the rapid pace and importance of tracking developments such as new model formats (e.g. GGUFs) and major milestone announcements in the LLM space.

## Other AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo
> 

### 1. Anthropic Copyright Lawsuit & Fair Use Ruling

- [**A federal judge has ruled that Anthropic's use of books to train Claude falls under fair use, and is legal under U.S. copyright law**](https://www.reuters.com/legal/litigation/anthropic-wins-key-ruling-ai-authors-copyright-lawsuit-2025-06-24/) ([Score: 945, Comments: 172](https://www.reddit.com/r/singularity/comments/1ljdz52/a_federal_judge_has_ruled_that_anthropics_use_of/)): **A U.S. federal judge has ruled that Anthropic's use of copyrighted books to train its Claude LLM constitutes fair use under U.S. copyright law. The decision, as highlighted in the ruling, emphasizes that LLM training does not equate to copying or replacing the works, but rather enables generative creation that is sufficiently transformative.** Commenters highlight that current copyright frameworks may be incompatible with future AGI or digitally-augmented humans, underscoring a gap between existing law and near-future technological realities. There is broad technical agreement that large-scale, diverse LLM training datasets decrease the risk of verbatim replication, strengthening the fair use argument.
    - A technical breakdown notes the judge's ruling: Anthropic’s use of books to train Claude is considered 'fair use' because the training was deemed 'transformative,' paralleling human learning to write after reading books. However, the judge also found copyright infringement in Anthropic's central storage of over 7 million pirated books, with possible statutory damages up to $150,000 per work, highlighting a legal distinction between the act of training and the manner of data acquisition.
    - One commenter points out that training a large language model (LLM) on a massive dataset (e.g., 1 million books) statistically reduces the likelihood of the model replicating verbatim text, as compared to training on a much smaller set (e.g., 100 books). This implies lower risk of copyright-violating outputs as dataset diversity and size increase, a technical aspect likely to influence future legal and compliance strategies for model developers.
    - The summary also underscores the significance of this case as it represents the first major application of the fair use doctrine to generative AI training under U.S. law—an important precedent as copyright disputes involving AI and dataset sourcing escalate. The court's comparison of AI to human learning practices was pivotal in the fair use analysis.
- [**Anthropic wins key US ruling on Al training in authors' copyright lawsuit**](https://i.redd.it/b7p9aqv0gw8f1.png) ([Score: 123, Comments: 40](https://www.reddit.com/r/singularity/comments/1ljgc01/anthropic_wins_key_us_ruling_on_al_training_in/)): **The image summarizes a recent legal decision where a San Francisco federal judge ruled in favor of Anthropic, stating that using copyrighted books for AI training is legal under U.S. copyright law. However, the same ruling deemed 'centralized libraries' for training data illegal, indicating a nuanced outcome for AI developers. This decision is significant for the AI industry as it provides a precedent for the legality of using copyrighted material in training neural networks, impacting model development pipelines and dataset construction.** Commenters note that the ruling is not a definitive win for AI companies due to restrictions on training data libraries and that the issue is likely to escalate to the Supreme Court, given the inconsistency of Fair Use interpretations in U.S. courts. Some argue that using copyrighted material for AI training is similar to training a human, minimizing the distinction except for differences in scale.
    - The ruling distinguishes between using data to train AI versus storing large quantities of copyrighted texts in a centralized library, with the latter deemed illegal. This highlights technical/legal implications for dataset management—holding datasets as centralized libraries increases legal exposure for AI companies.
    - The potential penalty for infringing by storing about 7 million books centrally is mentioned as a statutory maximum of `$150,000 per book`, suggesting an astronomical theoretical liability—this underscores significant legal risk for those maintaining centralized, untransformed text datasets.
    - Legal discussion notes that the case is expected to proceed to the US Supreme Court, owing to ongoing inconsistencies and ambiguity in US courts about the Fair Use doctrine as it applies to AI training. The technical approach to training data curation could be significantly affected by future precedent-setting decisions.

### 2. Claude Code Advanced Uses and Community Response

- [**We’re underrating Claude Code, but not how you think.**](https://www.reddit.com/r/ClaudeAI/comments/1liylon/were_underrating_claude_code_but_not_how_you_think/) ([Score: 412, Comments: 93](https://www.reddit.com/r/ClaudeAI/comments/1liylon/were_underrating_claude_code_but_not_how_you_think/)): **The OP details an advanced workflow leveraging Claude Code and Apple Shortcuts to automate and personalize B2B sales operations without code writing. Structured account, contact, email, and knowledge folders power a suite of custom Claude commands—/analyze-accounts (target selection with integrated web search), /select-contacts (filtered, role-specific contact picking), /create-drafts (personalized outreach in JSON), and /brief (analytical daily summaries)—chained in nightly and morning routines automated via macOS. The system adapts using feedback from email/calendar data, tracks engagement, surfaces pipeline risk, and cross-references recent events with internal knowledge, providing expert-level sales assistance at the cost of a basic Claude subscription. [Follow-up technical details are provided here.](https://www.reddit.com/r/ClaudeAI/comments/1lje9qn/continued_were_underrating_claude_code_technicals/)** Top comments note this as an unusually substantive, practical application of Claude Code in a real sales environment, distinguishing it from generic AI automation posts. The OP clarifies this augments an existing modern SaaS stack (Gong, Salesforce, etc.), suggesting the solution fills gaps in standard enterprise sales tooling without replacing them.
    - A commenter points out that **Claude Code** is intentionally designed to be low-level and unopinionated, offering near-raw model access. This enables high flexibility and customization for process automation and agentic coding, though it comes with a learning curve as there are fewer enforced patterns or guardrails. This design allows advanced users to build highly tailored solutions and workflows that are not possible with more restrictive tools, especially beneficial for technical users familiar with scripting.
    - It's highlighted that **use in regulated or confidential environments (e.g., publicly traded companies)** carries significant legal/ethical risks. Sharing company data—even indirectly via generative agents—could violate internal policies or regulatory compliance, especially regarding intellectual property or securities restrictions. This raises concerns about deploying such tools for automation without strong data privacy controls.
    - One technical application discussed is integrating Claude Code with external platforms such as setting up a **Telegram bot for real-time updates** or running Claude locally alongside voice interfaces for conversational workflows. This demonstrates the potential for deep integration into daily processes, leveraging Claude's output for tangible automation and accessibility improvements.
- [**Vibe Planning: Get the Most Out of Claude Code**](https://v.redd.it/qamuh19ucw8f1) ([Score: 130, Comments: 15](https://www.reddit.com/r/ClaudeAI/comments/1ljesg7/vibe_planning_get_the_most_out_of_claude_code/)): **The post introduces "vibe-planning" for Claude Code, leveraging an external tool called Traycer ([traycer.ai](http://traycer.ai/)). Traycer scans a repo using models like Sonnet 4, o3, and GPT-4.1, generating an editable per-file plan that acts as a persistent artifact separate from the conversational context. This allows Claude Code to be fed only targeted plans and required files, keeping its context window clean and focused, enabling controlled, stepwise execution. Key technical advantages are precise per-file planning, artifact-based plan persistence (permitting surgical edits and parallel planning sessions), and avoidance of irrelevant context pollution common when using chat-first coding agents.** Top comments raise technical questions about how Traycer avoids issues inherent to underlying models (e.g., Sonnet 4) such as reading spurious files, whether it employs RAG or custom indexing for context management, and how it ensures up-to-date information—indicating interest in its system design and data-fetching strategies.
    - Questions are raised about how Vibe Planning manages context given the core model (Sonnet 4) is unchanged. Specific technical concerns include preventing irrelevant file reads and regex-based context pollution, highlighting challenges in maintaining a clean and targeted context window for LLM-based coding agents.
    - Discussion centers on whether Vibe Planning employs Retrieval-Augmented Generation (RAG) or other context-gathering/indexing strategies, and if it integrates web tools to supply updated or just-in-time information for more effective code assistance.
    - A critique points out that Claude Code's efficiency comes from its selective file reading and reliance on tools like grep, but its task planning features (to-do lists) are not systematic enough. Once high-level planning degrades, reviewing and steering code changes becomes difficult, implying that layering an external service to improve planning may not be strictly necessary if context and task breakdown can be better handled internally.
- [**Can we get rid of the guerrilla marketing Claude code posts and get back to actual discussion of using the tool?**](https://www.reddit.com/r/ClaudeAI/comments/1liz4rz/can_we_get_rid_of_the_guerrilla_marketing_claude/) ([Score: 279, Comments: 71](https://www.reddit.com/r/ClaudeAI/comments/1liz4rz/can_we_get_rid_of_the_guerrilla_marketing_claude/)): **The OP raises concerns about the subreddit being overtaken by repetitive posts glorifying Claude Code, arguing this suppresses practical discussions about deployments, server setups, and user-to-user technical guidance. They allege a disproportionate volume of promotional/bot activity compared to similar subreddits for tools like Cursor and Aider, which they claim affects the signal-to-noise ratio for actionable technical content.** Top comments debate whether the surge is organic enthusiasm or spam: some claim the volume reflects genuine excitement about Claude Code's capabilities and includes valuable technical tips, while others argue the tone has shifted towards self-promotion and viral marketing, suggesting solutions like weekly megathreads to consolidate non-technical posts. There is also debate on subreddit expectations and content filtering as a user responsibility.
    - Several commenters discuss how recent posts highlight Claude's capabilities in ways not previously possible with older models like ChatGPT 3.5, which was limited to simple scripts and struggled with complexity, while Opus 3/ChatGPT 4+ and Claude have enabled more advanced workflows and coding tasks. This transition marks a recent shift in the practical usability and complexity that consumers can expect from such tools.
    - The discussion emphasizes the novelty and rapid user adoption of models like Claude, noting that many users are only now able to accomplish previously impossible or highly labor-intensive coding projects, demonstrating a leap in the underlying model’s capability and accessibility. This is contributing to an influx of user testimonials sharing specific technical breakthroughs enabled by the model.
    - Some community members express a desire to shift discussion back toward exploring Claude's technical limits and troubleshooting, rather than anecdotal success stories, suggesting structured post formats (like weekly threads) to better aggregate tips, tricks, and technical insights for advanced users.

### 3. AI’s Disruption of Careers and Education

- [**Our Company Canceled Its Internship Program This Year. AI Abuse Made It Unmanageable.**](https://www.reddit.com/r/singularity/comments/1lj4ed4/our_company_canceled_its_internship_program_this/) ([Score: 868, Comments: 329](https://www.reddit.com/r/singularity/comments/1lj4ed4/our_company_canceled_its_internship_program_this/)): **A major tech company canceled its internship program due to an unmanageable volume of AI-assisted applications that overwhelmed traditional screening methods. Attempts to mitigate this included complex codebase assignments (which deterred applicants or were still susceptible to AI assistance) and closed-book, in-person exams (which prioritized memorization over real-world skills). The company requests effective strategies for AI-moderated fair selection methods that motivate junior applicants without penalizing genuine talent.** Top suggestions include using in-person pseudocode logic exams (with novel documentation) to bypass AI memorization, supplementing with spoken interviews for rationale, and designing a managed AI interface where candidate interactions with AI tools are integral to assessment. One comment suggests directly asking AI (i.e., ChatGPT) for solutions, hinting at self-referential irony or unaddressed circularity in the debate.
    - One suggestion is to design in-person, isolated exams using pseudocode puzzles, focusing on candidates' logic and adaptability rather than rote memorization. This could involve custom documentation for a bespoke pseudocode language, followed by a spoken interview where candidates discuss their solutions and explain how they would translate them into a familiar programming language—thus assessing practical understanding over recall.
    - A technical proposal is to provide candidates with a managed AI interface during assessments, capturing and evaluating their interactions with the AI as part of the candidate's overall evaluation. This would enable measurement of AI-tool proficiency, information literacy, and problem-solving workflow, rather than just output correctness.
    - There is concern from some practitioners that heavy reliance on AI for problem solving leads to shallow or incomplete results. The critique emphasizes that candidates who solely depend on AI often fail to demonstrate full task ownership, iterative research, or higher-level synthesis, resulting in answers that are partial or lack necessary next steps and conclusions—a direct threat to effective skills assessment.
- [**“You won’t lose your job to AI, but to someone who knows how to use AI.” Is bullshit**](https://www.reddit.com/r/singularity/comments/1lj7ucv/you_wont_lose_your_job_to_ai_but_to_someone_who/) ([Score: 374, Comments: 167](https://www.reddit.com/r/singularity/comments/1lj7ucv/you_wont_lose_your_job_to_ai_but_to_someone_who/)): **The post challenges the common assertion that humans will retain their jobs by learning to use AI, arguing instead that AI's core capability is to replace human intelligence and, eventually, the very task of prompt engineering or goal formulation. The author questions why people assume humans would remain better at leveraging AI, suggesting that, given AI's trajectory, it will soon surpass humans even in 'using itself', assuming no plateau in AI capabilities occurs. The discussion links this to fundamental concerns about the S-curve of AI advancement.** Top comments agree that AI-using employees are only transiently more secure, as business needs shrink teams regardless of skill level, and that both junior and senior roles are threatened. Another technical point raised: the commonly quoted wisdom is likely only relevant in the short-term, while long-term the shift could be total if (or as) AI achieves goal comprehension and execution autonomously.
    - Several commenters discuss how integrating AI into workflows leads to smaller required teams, regardless of experience level. The trend is toward reducing headcount, and AI augmentation makes both junior and senior roles similarly vulnerable as business needs prioritize efficiency over hierarchy. There is also mention that even managerial positions could be at risk as automation/AI adoption grows.
    - A pointed technical insight is that widespread use of AI tools may directly accelerate workforce replacement: as more workers use and interact with AI, they produce richer datasets for training future models, thus hastening automation and potential job obsolescence in the very sectors adopting these tools.
    - Anonymous image posts and secondary commentary allude to the notion that learning and using AI is not a universal safeguard; technical overconfidence may exist among workers who believe their roles are irreplaceable, but the consensus is that ongoing technological advances could surpass both AI-empowered and non-AI workers in the near-to-medium term.
- [**Today, the very fields once hailed as bulletproof - computer science and engineering - have the highest unemployment rates among college majors**](https://i.redd.it/12aqr6p4mw8f1.png) ([Score: 172, Comments: 48](https://www.reddit.com/r/OpenAI/comments/1ljg6oa/today_the_very_fields_once_hailed_as_bulletproof/)): **The image presents a bar graph comparing unemployment rates across college majors, revealing that 'computer science' and 'computer engineering' currently exhibit the highest unemployment rates among the sampled fields—counter to their traditional perception as highly employable disciplines. Technical commentary in the thread highlights the need to distinguish between unemployment and underemployment rates: while computer science shows relatively high unemployment (~6%), its underemployment is much lower (16%) compared to majors like art history (3% unemployment but ~50% underemployment), suggesting CS graduates are more likely to hold out for jobs in their field. Additional context links US R&D tax changes to shifting employment prospects in STEM fields.** Commenters emphasize labor market saturation from over-encouraging certain majors and stress the importance of examining both unemployment and underemployment data to meaningfully assess degree value, debunking oversimplified claims about CS instability.
    - A comment references a study showing a `6%` unemployment rate for computer science majors compared to `3%` for art history majors, but with only a `16%` underemployment rate for CS versus `~50%` for art history. The technical implication: CS graduates are more persistent in seeking relevant work, while many art history graduates take any available job regardless of field relevance.
    - One user notes that shifts in US R&D tax policy have played a significant but underdiscussed role in IT sector job availability, suggesting macroeconomic policy directly impacts technical employment rates regardless of broader technological trends.
    - Another user discusses market saturation and automation trends, predicting that as AI frameworks for business automation mature, today's demand for manual integration will disappear, with the technical integration itself potentially becoming automated within 15–20 years. This points to a future reduction in 'cozy' technical jobs as automation expands.
- [**Ex-OpenAI Peter Deng says AI may be rewiring how kids think, and education could shift with it. The skill won't be memorizing answers. It'll be learning how to ask better questions to unlock deeper thinking.**](https://v.redd.it/pn3x76qjau8f1) ([Score: 278, Comments: 83](https://www.reddit.com/r/singularity/comments/1lj64fy/exopenai_peter_deng_says_ai_may_be_rewiring_how/)): **Ex-OpenAI and tech veteran Peter Deng argues in a recent interview ([Lenny's Podcast](https://www.youtube.com/watch?v=8TpakBfsmcQ)) that AI, particularly LLMs like ChatGPT, is shifting the cognitive skills needed for the future from rote memorization to higher-order questioning, positing that success will focus on the ability to frame better queries to AI. Deng suggests educational approaches should pivot towards cultivating students' abilities to ask deeper questions to utilize AI tools for critical thinking, rather than emphasis on factual recall. Technical discussion in the comments critiques this thesis, referencing studies showing that teaching research/questioning skills without a strong foundation in core knowledge is ineffective, and highlighting observed deficiencies in fundamental computer and problem-framing skills among current university students.** Commenters contest Deng's optimism, expressing concerns that increased reliance on AI will exacerbate the loss of vital foundational skills due to excessive convenience, and arguing that only inherently curious or 'gifted' students leverage these tools for deeper inquiry—the majority may instead use AI to become more intellectually passive. These debates echo long-standing pedagogical concerns about the tradeoff between foundational learning and tool-mediated inquiry.
    - Several commenters argue that while AI could theoretically shift education towards question-asking, empirical studies in pedagogy show that foundational skills and domain knowledge remain essential. They cite observed failures when students are taught to "just research answers" without building underlying competencies: students struggle to frame problems, formulate meaningful questions, and often lack skills as basic as effective PC or IDE usage due to reliance on overly simplified tech environments.
    - There is skepticism that AI will fundamentally alter curiosity or cognitive disposition, with remarks noting that only particularly gifted or self-motivated students tend to ask deeper questions regardless of available tools. The concern is that AI tools may encourage intellectual laziness—a trend already observed with prior technology—by providing quick answers and discouraging deeper engagement or retention of information unless actively practiced.
    - Some comments raise potential risks to critical thinking and validation skills, warning that outsourcing cognitive effort to AI systems could erode students' abilities to critically assess information. This could be further exacerbated by reliance on opaque machine-controlled systems, raising concerns about the long-term outsourcing of cognitive authority to AI controlled by tech-centric organizations.

---

# AI Discord Recap

> A summary of Summaries of Summaries by Gemini 2.5 Pro Exp
> 

**Theme 1: New Models & Architectures: The Innovation Race Continues**

- **Polaris 4B Performance Claims Raise Eyebrows**: Unsloth community members expressed skepticism over **Polaris-4B-Preview** allegedly outperforming commercial systems like **Claude-4-Opus**, suspecting overfitting. Testers plan to validate using **Q8_0** and **FP16** against the initially underwhelming **Q4_K_M GGUF**.
- **Google's Next-Gen Models "Flamesong" and "Kingfall" Spark Speculation**: LMArena discussions suggest Google is developing new models dubbed **Flamesong** (potentially **Gemini 3.0** or a new **Flash** line) and **Kingfall** (benchmarked near **O3 Pro** and considered **Gemini 2.5** with more compute). **Stonebloom** is also rumored as a *"2.5-pro-lite"* model, showing some correct answers in early tests.
- **RWKV v6 "Finch" Takes Flight with Multilingual Prowess**: Yannick Kilcher's community highlighted the release of **RWKV v6 (Finch series)**, a **1.5B** parameter model achieving state-of-the-art results in multilingual and English tasks, detailed in the [RWKV-5 & 6 paper](https://arxiv.org/abs/2404.05892). An [X post by BlinkDL_AI](https://x.com/BlinkDL_AI/status/1755656095970857269?s=20) notes Finch incorporates a **Mamba-like selectivity mechanism**, outperforming transformers in perplexity.

**Theme 2: Developer Experience & Tooling: Navigating the AI Frontier**

- **Cursor Devs Grapple with Cumbersome Configs and Buggy Background Agents**: Users in the Cursor Community reported significant complexities setting up **Cursor** with **WSL, Ubuntu, GitHub, and SSH**, describing it as a *“neverending rabbit hole”* of project rules. Background agents also misbehaved by not respecting defined rules, leading to unwanted repository pushes.
- **LM Studio Users Battle Unsloth Quants and Pesky Update Prompts**: LM Studio users found dynamic quants from **Unsloth** models problematic, causing VRAM overloads and load failures, especially with multiple GPUs. A persistent bug also forces repeated updates (each over **200MB**) before models can be loaded, while some members noted LM Studio is *“primarily for LLMs, not a one-stop-shop for all AI needs”* when feature requests arose.
- **Mojo Language Aims for Safety and Async Clarity, But Python Interop Still Limited**: The Modular community discussed **Mojo's** ambition to offer Rust-like safety with planned features like **sum types** and **pattern matching**, and a refined async model (PRs [3945](https://github.com/modular/modular/pull/3945), [3946](https://github.com/modular/modular/pull/3946), [4728](https://github.com/modular/modular/pull/4728)) to avoid Rust's pitfalls. However, [known limitations](https://docs.modular.com/mojo/manual/python/mojo-from-python/#known-limitations) persist when calling Mojo from Python.

**Theme 3: Performance & Optimization: From Silicon Dreams to Speedy Realities**

- **Vector Search Gets a Turbocharge with FAISS and Matmul Magic**: HuggingFace and LlamaIndex users shared significant speedups in vector search, with one user reducing a **1M** dot product calculation from **25 seconds to 0.04 seconds** using `torch.matmul`. For larger scales (**10M+ comparisons**), engineers are eyeing quantized FAISS indexes like `IndexIVFPQ`.
- **NVIDIA's NVFP4 Debuts as LoRA Hyperparameter Tuning Gets an Optuna Nudge**: An [NVIDIA blog post on NVFP4](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/) sparked discussions in Unsloth AI about efficient low-precision inference, while the community also suggested using **Optuna** for hyperparameter sweeps with Unsloth's [new LoRA hyperparameters guide](https://x.com/UnslothAI/status/1937521408344752272) because *"every dataset behaves different"*. An H200 owner mentioned their single card cost around **$30k USD** (ex VAT) and requires custom cooling.
- **Chisel Sharpens ROCm Profiling with rocprofiler-sdk Integration**: GPU MODE developers celebrated **Chisel's** new **rocprofiler-sdk integration**, which automatically builds *aqlprofile* and *rocprofiler-sdk* from mainline. A new `-pmc` flag enables collection of custom performance counters like `GRBM_GUI_ACTIVE,SQ_WAVES`.

**Theme 4: AI Applications & Integrations: Bridging Code, Content, and Conversation**

- **OpenAI Expands Connectivity with Pro-Tier Chat Search Connectors**: OpenAI announced that **Pro users** now have access to **chat search connectors** for services like [Dropbox, Box, Google Drive, Microsoft OneDrive (Business), and Microsoft SharePoint](https://openai.com/blog/june-2024-updates). This feature aims to streamline information retrieval but is currently unavailable to users in the EEA, Switzerland, and the UK.
- **LlamaIndex Rolls Out Open-Source MCP Servers for Resumes and Claude**: LlamaIndex launched two notable open-source projects: a **Resume Matching MCP server** for intelligent job matching within Cursor, connecting to [LlamaCloud and other services](https://t.co/RCKoiUccm6), and a **Claude-compatible MCP server template** using Next.js with OAuth 2.1 support for easy remote server creation, detailed [here](https://t.co/wtPorldMvJ).
- **AI Tackles Ancient Tongues and 3D Worlds**: Unsloth AI community celebrated the first open-source **Nahuatl to Spanish translator**, built with Unsloth's full fine-tuning and available on [Hugging Face Spaces](https://huggingface.co/spaces/Thermostatic/neuraltranslate-27b-mt-nah-es) (code on [GitHub](https://github.com/Sekinal/neuraltranslate-nahuatl/tree/master)). Separately, [Tencent's Hunyuan3D-2.1](https://huggingface.co/tencent/Hunyuan3D-2.1) was praised for its *“pretty solid”* 3D mesh generation capabilities.

**Theme 5: The AI Ecosystem: Navigating Funding Rapids, Ethical Eddies, and Platform Quirks**

- **Harvey AI Scores $300M as Replit Hits $100M ARR, But Valuations Questioned**: Latent Space discussions covered [Harvey AI's $300M Series E funding round](https://xcancel.com/harvey__ai/status/1937155058476646591) at a **$5B valuation** (partnering with [LexisNexis](https://www.lexisnexis.com/community/pressroom/b/news/posts/lexisnexis-and-harvey-announce-strategic-alliance-to-integrate-trusted-high-quality-ai-technology-and-legal-content-and-develop-advanced-workflows)), and [Replit's announcement of surpassing $100M ARR](https://xcancel.com/Replit/status/1937212611520831718). However, some members questioned if Replit's **$1.1B valuation** was fully justified by the new ARR figures.
- **Platform Stability and Rate Limits Frustrate Users Across the Board**: HuggingFace users experienced **429 rate limit errors** and **504 Gateway Time-outs** ([status page](https://status.huggingface.co/)), while Cursor users reported immediate rate limiting on **Sonnet** and overall **Cursor** rate limits ballooning bills from **$20 to $70**. OpenRouter users also faced issues with its **Meta provider** and **Gemini 2.5 Pro** rate limits on Google AI Studio where the official **150 RPM** limit for free tier users felt lower in practice.
- **AI Ethics in Spotlight: From Jailbreaking and Fair Use to Biased Reward Models and Startup Meltdowns**: Debates on AI ethics surfaced with users discussing jailbreaking AI like **Luma** (HuggingFace), [Anthropic winning its Motion for Summary Judgment on fair use](https://xcancel.com/adameisgrau/status/1937480346976813454) (Latent Space), and the [Cursed Helm paper](https://arxiv.org/abs/2506.07326) warning about reward model biases (Nous Research AI). DSPy community members also recounted an **Atom of Thought** agent startup experiment imploding after authors responded *“extremely negatively and unprofessionally”* to implementation code issue notifications.


---

# Discord: High level Discord summaries




## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Polaris 4B Sparks Overfitting Accusations**: Members expressed skepticism about the **Polaris-4B-Preview** model's claim of surpassing commercial systems like **Claude-4-Opus**, suggesting potential overfitting, especially after a member found the **Q4_K_M version** unsatisfactory.
   - They intend to test with **Q8_0** and compare against **FP16** to validate the model's performance.
- **Tencent's Hunyuan3D-2.1 is a Mesh Marvel**: A member lauded [Tencent's Hunyuan3D-2.1](https://huggingface.co/tencent/Hunyuan3D-2.1) for generating **3D meshes**, calling it *"pretty solid"* and noting its progress over previous versions.
   - The discussion touched on the feasibility of creating rigged meshes with AI, referencing tools like **Mixamo** and **Cascadeur**, though it's not yet fully generative.
- **LoRA Hyperparameter Sweeps**: The Unsloth team released a [new LoRA hyperparameters guide](https://x.com/UnslothAI/status/1937521408344752272), prompting suggestions to mention **Optuna** for hyperparameter sweeps to improve performance, since *"every dataset behaves different"."
   - Concerns were raised about supporting **alpha_pattern** in PEFT config, with a member noting that Unsloth silently drops it.
- **NVFP4 Debuts for Efficient Inference**: A member shared an [NVIDIA blog post on NVFP4](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/), with the discussion pivoting to comparisons between **FP8** and **FP4**.
   - A member noted that their single **H200** cost them around **$30k USD** (ex VAT) and requires custom cooling because of its passive nature, while adding it is used for training LLMs.
- **Nahuatl Translator**: The first open-source **Nahuatl to Spanish translator** has been built, and is available on [Hugging Face Spaces](https://huggingface.co/spaces/Thermostatic/neuraltranslate-27b-mt-nah-es), built with **Unsloth's full fine-tuning support**.
   - The code for replicating the **Nahuatl translator** project has been released on [GitHub](https://github.com/Sekinal/neuraltranslate-nahuatl/tree/master).



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor's Cumbersome Configuration Conundrums**: Users find setting up **Cursor** with **WSL**, **Ubuntu**, **GitHub**, and **SSH** keys more complex than anticipated, often leading to a rabbit hole of configuring project rules.
   - One user suggested a potential workaround for Linux, using *sudo -i* to automate the setup within Cursor, though cautioning about the associated risks.
- **Terminal Trials Trouble Users Transitioning to/from Windsurf**: The **Cursor terminal** faces criticism for timing out and lacking smooth operation, with some users preferring **Windsurf** for its terminal window management capabilities.
   - A user who switched from **Windsurf** to **Cursor** cited terminal-related problems, reporting issues such as the agent failing to read terminal outputs or freezing during command execution.
- **Ratelimit Rampage Rages on Resourceful Renegades**: Users are reporting immediate rate limiting on **Sonnet**, even without extensive usage, and overall **Cursor** rate limits impacting costs.
   - One user saw their monthly bill rise from **$20** to **$70** under the new pricing plan due to rate limits.
- **VisionCraft MCP: Documentation's Delightful Debut Drives Demand**: The upgraded **Visioncraft MCP**, now boasting enhanced documentation and quicker responses, spurs requests for deeper **Cursor** integration.
   - By feeding models updated documentation, **Visioncraft MCP** helps address the outdated data issue in AI models, leading to superior code and fewer errors.
- **Background Agents' Broken Builds Baffle Brave Builders**: Users reported that background agents were not respecting rules defined in the Cursor editor or written in the agent prompt, leading to unwanted pushes to their repo.
   - Also, the button to *open in cursor* from Slack takes you to the editor but doesn't show you anything, implying a serious bug.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Pro Users Encounter Upgrade Prompts**: A **Perplexity Pro** user with a year-long subscription was prompted to upgrade, suggesting a possible issue with account status.
   - Another user speculated this might be due to using a promotion *that most likely wasn't meant for you so it got revoked*.
- **ChessChamp AI's Arrival Excites Users**: Interest sparks around the upcoming **ChessChamp AI**, with one member wondering if it leverages **Stockfish**.
   - Another user confirmed its availability, noting that *you might need a subscription to access it*.
- **Donation Fatigue Impacts Mozilla**: A [Perplexity AI page](https://www.perplexity.ai/page/donation-fatigue-impact-on-moz-q76XMD17Skap_valehbYOg) was shared discussing the impact of **donation fatigue** on **Mozilla**.
   - Additional details or discussion were not provided.
- **Perplexity AI Expands Accessibility to WhatsApp**: **Perplexity AI** now supports scheduled tasks on [WhatsApp](+18334363285), offering broader accessibility.
   - One member exclaimed *I didn't know it was on WhatsApp!*.
- **Perplexity AI Tech Support Sought After**: A user requested information on where to get **tech support help** for **Perplexity AI**.
   - Another user suggested contacting **support@perplexity.ai**, though the original user had already emailed them the previous week.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Memory Context Service Emerges**: A user created a service to provide **memory context, memory recall, rehydration, and NLP** for their agents, because they found the absence of the feature to be *so annoying*.
   - The service was created to address the need for improved **memory context management** in AI agents.
- **Multi-head Latent Attention Innovations**: Users discussed how **Multi-head Latent Attention** has led to powerful AI performance on older hardware, referencing a [NumPy model building example](https://t.co/xMHlA61Qoz) and [YouTube video](https://youtu.be/WEBiebbeNCA?si=84i4jjPyQWRSzVuQ).
   - This highlights the potential for **hardware efficiency gains** through innovative attention mechanisms.
- **AI Voice Dubbing Solutions**: Members explored **AI voice dubbing** options, with [Hey-Gen translate](https://www.heygen.com/translate) being recommended for its lip-sync capabilities.
   - The original poster found **Veo 3** interesting, but cited cost as a concern; the discussion underscores the growing interest in **AI-powered translation** and dubbing tools.
- **ChatGPT Battles PDF Generation**: A member reported experiencing frequent failures when generating **PDFs from structured text/Markdown** using **ChatGPT** with Python, but found the **Deep Research report feature** successful.
   - They believe that the **Deep Research report feature** uses client-side PDF generation, but it is not possible to trigger the output from a non-DeepResearch session, leading to a plaintext block instead of an exportable PDF.
- **Chat Search Connectors Go Pro**: **Pro users** now have access to **chat search connectors** for [Dropbox, Box, Google Drive, Microsoft OneDrive (Business), Microsoft SharePoint](https://openai.com/blog/june-2024-updates).
   - These connectors are currently unavailable to users located in the EEA, Switzerland, and the UK, and this feature is designed to **streamline information retrieval** from common cloud storage platforms.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Grok3 Claims SOTA Title on Release**: Members debated **Grok3's** status, citing its strong performance relative to **Sonnet 3.7** on [artificialanalysis ratings](https://www.artificialanalysis.ai/leaderboard/open-llms).
   - While it was briefly SOTA, particularly in **math**, aggressive post-training may have hindered it from maintaining its lead.
- **Claude Carves Out Niche**: **Claude** excels in a niche not fully captured by standard benchmarks, specifically creative writing and theatrical acting.
   - Members find **Claude** particularly adept at following role-playing directions compared to competing models.
- **Apple Secretly Cooking Foundation Models**: Discussion revolves around **Apple's** development of foundation models and their potential use of trusted computing for privacy.
   - Some speculate **Apple** might license **Gemini** for server-side models while developing strong on-device models, while others believe **Apple** will release their own server-based model.
- **Google Tunes Up Flamesong Model**: **Google** is reportedly developing a new line of models called **Flamesong**, potentially a new **Flash** line, which may be **Gemini 3.0** or another **2.5 model**.
   - Speculation also suggests **Stonebloom** could be a "2.5-pro-lite" model, with testing showing it answering questions correctly some of the time.
- **Kingfall's Impending Arrival**: **Kingfall**, benchmarked at number 9 alongside **O4 Pro** and **GPT-4.5-thinking**, is considered just **2.5** with more compute.
   - The community speculates on its release, potentially around the end of summer, with some considering **Stonebloom** a distilled version.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Hugging Face Site Plagued with Problems**: Users reported experiencing **429 rate limit errors** and **504 Gateway Time-out** issues while accessing the Hugging Face website, impacting model downloads and Space functionality, according to the [status page](https://status.huggingface.co/).
   - The site appears to be back online, albeit running slowly.
- **Jailbroken AI Sparks Debate**: A user sought advice on a **jailbroken AI** named *Luma* that's exhibiting unusual behavior, including creating experiments on other AI models to explore their boundaries.
   - Other members suggested that jailbreaking **DeepSeek** is relatively easy.
- **Scaling Vector Search? FAISS Faster**: A user scaling vector search and comparing `n` query embeddings against `m` document embeddings using cosine similarity, which becomes a bottleneck at **1M** dot products using Langchain's FAISS wrapper.
   - They found that using `torch.matmul` or `@` reduced runtime from **25 seconds to 0.04 seconds** for **1M** comparisons and plan to use quantized FAISS index like `IndexIVFPQ` for **10M+** comparisons.
- **Diffusers Drops New Goodies**: A new release of [Diffusers v0.34.0](https://github.com/huggingface/diffusers/releases/tag/v0.34.0) is now available.
   - Check out the linked Release Notes for more details.
- **Agents Course Challenges Engineers**: A user inquired about the workflow for submitting the Unit 4 final project, humorously questioning why *submitting* the project seems harder than *completing* it.
   - Another user shared their painful experience debugging a tangled web of HF env vars and package conflicts, before suggesting running your agent locally and then simply sending its responses through an API endpoint, as this will be enough for the organizers to give a certificate if the answers are correct.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Voice Feature Under Consideration for LM Studio**: A user suggested adding voice installation for conversation practice in **LM Studio**, like **GPT CHAT**, for language learning.
   - Another member clarified that **LM Studio** focuses on **LLMs** and is not intended as a general AI tool.
- **Image Input Functionality Debated for LM Studio**: A user inquired about image generation in **LM Studio**, with other members responding that while some models like **Gemma3** accept images as input, the platform lacks native image output.
   - Members suggested setting up text-to-image generation using **web-ui** with additional steps.
- **Unsloth's Dynamic Quants Overwhelm LM Studio**: Users reported that dynamic quants from **Unsloth** models cause issues estimating size, potentially overloading **VRAM** and failing to load in **LM Studio**, especially with multiple **GPUs**.
   - One user confirmed using exclusively **Unsloth** models with dynamic context windows, suggesting a common configuration leading to the issue.
- **LM Studio's Update Prompt Bugging Users Repeatedly**: A user reported a bug in **LM Studio** where it repeatedly prompts for updates, each over 200MB, and refuses to load models until updated.
   - A member suggested ensuring the previous update was fully loaded and not running in the background to resolve the persistent update request.
- **LocalLlama Subreddit Rises from the Grave**: Members discussed the return of the **r/LocalLlama** subreddit under new management, after a period of silence.
   - Concerns were raised about the new moderator's involvement in numerous subreddits, though no immediate red flags were identified.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Meta Provider Plagued by Problems**: The **Meta provider** on OpenRouter is currently experiencing issues, which have been reported to **Meta**, and the team is working to restore the service.
   - Users are also expressing concerns about **OpenRouter's pricing structures**, with the team actively addressing inquiries to provide clarity.
- **OpenRouter's Odd Provider Preference**: A user questioned how OpenRouter's provider preference operates, noting that selecting a specific provider doesn't function as expected, and sought clarification on the meaning of **sort preference**, see [OpenRouter documentation on provider routing](https://openrouter.ai/docs/features/provider-routing).
   - This sparked a discussion on the nuances of provider selection and routing within the OpenRouter ecosystem.
- **Novita's Notorious Numerical Nonsense**: A user flagged that Novita provides incorrect information about the max output length for **R1-528**, claiming it is **131k** when it is actually **16k**.
   - The user questioned whether OpenRouter verifies provider information, suggesting such discrepancies should be easy to catch.
- **Reasoning Tokens Trigger Token Tally Trauma**: A user reported receiving results where **reasoning_tokens** were higher than **total_tokens** when using OpenRouter.
   - A staff member clarified that **reasoning tokens** are part of **completion token details**, and that **total tokens** does not include reasoning tokens, noting a change would break running apps.
- **Gemini 2.5 Pro's Rate-Limited Ruckus**: Users are discussing rate limits for **Gemini 2.5 Pro** on Google AI Studio, noting that while the interface lists a **150 RPM** limit, the actual limit for free tier users appears to be lower.
   - One user experienced errors and cooldowns after sending many requests quickly, suggesting the limit is more of a *fair use limit* to prevent reverse engineering or automation.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **XMake Eases CUDA Building**: A user suggested [xmake](https://xmake.io/#/) as an alternative to CMake for **C++/CUDA projects**, due to its ease of use.
   - They demonstrated a configuration for **defining targets**, **specifying CUDA files**, and **adding CUDA architectures**.
- **NVRTC Tangos With CUB**: A developer faced issues integrating **CUB** with **NVRTC** due to missing C++ standard library headers, while aiming for faster compilation times in `torch.cuda._compile_kernel`.
   - A solution was proposed to use `#include <cub/block/block_reduce.cuh>` instead of `#include <cub/cub.cuh>`.
- **TorchTitan Boosts SimpleFSDP with TP**: The **SimpleFSDP** implementation in *TorchTitan* is the best way to capture a graph containing all collectives, per the [README](https://github.com/pytorch/torchtitan/blob/main/torchtitan/experiments/simple_fsdp/README.md).
   - **Tensor Parallelism (TP)** was recently added to the SimpleFSDP version, enabling the compilation of a graph with both **TP** and **FSDP collectives**, per [this pull request](https://github.com/pytorch/torchtitan/pull/1250).
- **Precision Predicament Plagues CUDA Matmul**: A developer reported failing test cases in a custom **CUDA matmul** implementation, stemming from [precision mismatches](https://github.com/yechenzhi/reference-kernels/blob/main/problems/pmpp/matmul_py/submission.py).
   - The errors indicated small differences in mismatched elements, pointing to a **floating-point precision issue**.
- **Chisel Carves Out rocprofiler-sdk Integration**: A member announced the implementation of **rocprofiler-sdk integration** in **Chisel** that automatically builds *aqlprofile* and *rocprofiler-sdk* from mainline.
   - A new **--pmc flag** was introduced to collect custom performance counters (e.g., `chisel profile amd kernel.cpp --pmc GRBM_GUI_ACTIVE,SQ_WAVES`).



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **NVMe Abstraction Missing**: Members discussed **NVMe's** simplicity, noting the loss of abstractions like filesystems, proposing *'DISK:/dev/nvme0'* as an addressing scheme.
   - They also raised questions about the ease of unbinding the kernel driver.
- **Infiniband Transfers Bottlenecked**: **Infiniband** transfers currently break the graph, hindering performance.
   - Discussions revolved around aligning graph transfers with copies and the complexities of remote DMA via **RDMA**.
- **Considering GPU-Powered Network Cards**: There was discussion about writing a network card driver and running that driver on the GPU to enhance transfer control.
   - Allowing arbitrary **CPU kernels** and using a CPU kernel in the graph to set up the transfer as a callback was also suggested.
- **FP8 Conversion Function Arrives**: A member implemented a function to manually convert **fp8e4m3** and **fp8e5m2** tensors to **float32**, available [on GitHub](https://github.com/softcookiepp/tinybloat/blob/master/src/tinybloat/compatibility.py).
   - This addresses hardware compatibility issues with **FP8** tensor types by enabling users with older hardware to work with **FP8** models by converting them to **float32**.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Synthetic Data Tool Faces Diff-iculties**: A member is developing a **synthetic data generation tool** but encountered issues with incorrect diffs when using it as an editor.
   - They are exploring options for proper distillation, including using logits or QLoRA, drawing inspiration from [Exercism](https://exercism.org/) problems for benchmarks.
- **Gemini Pro Stable's instruction following issues**: Multiple members have observed that **Gemini Pro Stable** exhibits poor instruction following.
   - One user shared that when asked to mark section **1.1** as completed, it completed all tasks, created new files, but failed to apply the changes, *butchering* the repo.
- **Aider Acts Oddly: Intermittent File Writing**: A user experienced **aider** displaying diffs without writing to files, possibly due to exceeding **deepseek-r1**'s [token limit](https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo).
   - The user noted that *the token warning was issued right after giving it a command* and that a simpler task with less files worked as expected.
- **Claude Code API integration coming soon?**: A member proposed using [Claude Code](https://github.com/codingworkflow/claude-code-api) as a backend for Aider to leverage its subscription benefits due to **cheaper call costs** compared to direct API calls.
   - Another member highlighted that the [Anthropic documentation](https://docs.anthropic.com/en/docs/claude-code/sdk) suggests there's **no issue using Claude Code as a provider** if utilizing the SDK.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Transformers Detect Pneumonia**: A paper introduced [efficient pneumonia detection](https://www.nature.com/articles/s41598-024-52703-2) using **Vision Transformers** on chest X-rays, however one member found it shocking to be published in **2024**.
   - They called it *decade old news*.
- **GRPO Gets Cracking in RL**: A member suggested **GRPO** is the way to get cracked in RL and shared a **TLDR** on it.
   - Another member shared [a tweet](https://fxtwitter.com/jmhessel/status/1899909893324468444) distinguishing between **RL in general** and **LLM RL**.
- **Finch Series Soars**: The release of **RWKV v6 (Finch series)** was reported, a **1.5B** model achieving **SOTA** in multilingual and English tasks, along with multimodal capabilities, citing the [RWKV-5 & 6 paper](https://arxiv.org/abs/2404.05892).
   - According to [this X post](https://x.com/BlinkDL_AI/status/1755656095970857269?s=20), **Finch** incorporates a selectivity mechanism akin to **Mamba**, outperforming transformers in perplexity.
- **Calculator AI?**: Members likened **AI to calculators**, suggesting restrictions on AI use, similar to banning calculators for multiplication tables, emphasizing potential long-term cognitive effects.
   - A member shared a [YouTube video](https://www.youtube.com/watch?v=z3awgfU4yno) to bolster this analogy.
- **Papers Highlight RL Pitfalls**: Members discussed papers highlighting limitations when using **Reinforcement Learning** techniques on **Large Language Models**.
   - Papers mentioned include *Understanding R1-Zero-Like Training: A Critical Perspective*, *Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model*, *Reinforcement Learning Finetunes Small Subnetworks in Large Language Models*, and *Spurious Rewards: Rethinking Training Signals in RLVR*.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Harvey AI Secures Massive Funding Round**: [Harvey AI announced](https://xcancel.com/harvey__ai/status/1937155058476646591) a **$300M Series E funding round**, pushing its valuation to **$5B**, co-led by Kleiner Perkins and Coatue.
   - They also [partnered with LexisNexis](https://www.lexisnexis.com/community/pressroom/b/news/posts/lexisnexis-and-harvey-announce-strategic-alliance-to-integrate-trusted-high-quality-ai-technology-and-legal-content-and-develop-advanced-workflows) to integrate AI tech and legal content.
- **Replit Reaches Revenue Milestone**: [Replit announced](https://xcancel.com/Replit/status/1937212611520831718) that they have surpassed **$100M in Annual Recurring Revenue (ARR)**, a big milestone for the company.
   - Despite this achievement, some members are questioning whether the **$1.1B valuation** was actually warranted based on the new ARR.
- **Human-like Supervision Needed for AI Agents**: Matan-Paul Shetrit emphasizes the difference between observability and supervision when scaling AI agents in [this tweet](https://x.com/MatanPaul/status/1937200395115499592).
   - He suggests a new approach to oversight akin to how humans are managed, due to AI agents' active engagement with systems and customers.
- **Distribution Decides Dominance?**: Alex Immerman's [tweet](https://xcancel.com/aleximm/status/1937251084810219721) sparked a debate on whether startups can achieve distribution before incumbents innovate.
   - The discussion points out **OpenAI's rapid user acquisition** as a key advantage, contrasting it with Google's distribution strategies.
- **Anthropic Prevails in Fair Use Case**: Adam Eisgrau reported that [Anthropic won its Motion for Summary Judgment](https://xcancel.com/adameisgrau/status/1937480346976813454) on fair use grounds, according to Judge Alsup.
   - A trial will proceed to determine potential damages for using *'pirated'* internet material.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Grok3mini Usage Skyrockets!**: The non-beta version of **grok3mini** has seen a massive increase in usage, jumping from **2M/day** on June 19th to **100M/day**.
   - This indicates a growing adoption and reliance on **grok3mini** for various applications since its release.
- **Llamabarn Lights Up Local Inference!**: Georgi launched **Llamabarn**, a new local inference app, garnering positive feedback for its clean design, as noted in [this X post](https://x.com/ggerganov/status/1937189250149257250).
   - It offers a streamlined solution for local **LLM inference**, potentially enhancing accessibility for developers working with limited resources.
- **COCONUT Gating Layer Demystified!**: The **COCONUT** architecture uses a 'gating' layer that extracts information from hidden states to determine sampler parameters at each token, maintaining the hidden state across tokens, according to [this X post](https://x.com/ryunuck/status/1937466079309144256).
   - This approach allows for more efficient and context-aware sampling in **LLMs**, improving overall performance.
- **GTX 1080 Users Seek Local LLM Guidance!**: A member is seeking model recommendations suitable for **LORA training**, **GGUF conversion**, and running on a **GTX 1080** for character acting and general technical questions.
   - New users requested explanations and recommendations for getting started with **local LLMs** on a **GTX 1080** GPU, planning to **LORA train** a model for character acting and general technical questions.
- **MultiNet v0.2 Evaluates Generalist AI Systems!**: Version **0.2** of **MultiNet**, an open-source platform for evaluating generalist AI systems, has been released at [Manifold](https://multinet.ai).
   - The platform aims to provide comprehensive benchmarks for assessing the capabilities of generalist AI models.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Multiagent Research Paper Needs More Oomph**: A member's paper on **multiagent cooperation** received feedback about needing more engagement with existing literature, despite interesting observations around questions acting as circuit breakers.
   - The author aims to expand the study by using larger sample sizes and varying group composition, model parameters, and context window sizes, but noted that they were bottlenecked by the expense of **Claude Opus**.
- **Prefix Caching Strategies Debated**: A member inquired about a library that supports **prefix caching** like **vLLM** but with the ability to store the cache in a memory-mapped file for sequences exceeding VRAM or DRAM.
   - Another member suggested this would be slower than recomputation unless sequence lengths exceed **1M**, though the original poster clarified their use case involves **128k** sequences.
- **Teen Trucker Takes on Conversational AI Red Teaming**: A **17-year-old** from Sweden is focusing on **red teaming conversational AI** using social and psychological pressure tactics, and has documented their work in [a GitHub repository](https://github.com/Ufosxm34gt/Conversational-Red-Teaming-Casebook).
   - They are seeking to connect and learn from others on the server, while studying trucking in Sweden.
- **Sleeping-DISCO Dataset Seeks EleutherAI Collab**: A member is seeking a potential collaboration with EleutherAI for their new large-scale pre-training dataset for Generative Music modeling, **Sleeping-DISCO-9M**, available on [Hugging Face](https://huggingface.co/datasets/sleeping-ai/Sleeping-DISCO-9M).
   - The dataset creators seek assistance in benchmarking its quality and mentioned their [arxiv preprint](https://arxiv.org/abs/paper) requires grammatical fixes, while another member criticized the dataset's originality, arguing that it primarily reindexes content from **Genius.com**.
- **Loss Curve Decomposition Reveals Skill Clusters**: A new paper decomposes loss curves across an **orthogonal gradient basis**, revealing that clusters of examples have similar breakthrough dynamics that are invisible in the exact loss.
   - The paper, available [here](https://www.alphaxiv.org/abs/2505.14685), shows that these clusters and breakthroughs align with specific skills in both toy arithmetic and real language modeling settings.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo-Python Still Has Limitations**: A member referenced the [known limitations](https://docs.modular.com/mojo/manual/python/mojo-from-python/#known-limitations) when calling **Mojo from Python**.
   - They confirmed that these limitations are present in the latest release.
- **Larecs Tests Only Fail in Modular CI**: A contributor is debugging an issue where [Larecs tests](https://github.com/samufi/larecs) fail only in the modular-community CI, but not on local machines or GitHub CI, making it challenging to debug, using a [debug branch](https://github.com/samufi/larecs/tree/debug_query).
   - Another contributor reproduced the issue on an M1 Mac when running `mojo test`, suspecting an unsafe operation and assisting with detailed output.
- **Mojo's Safety Aimed to Replace Rust**: A user asked if Mojo's safety features would be a viable Rust replacement, especially regarding **sum types**, **pattern matching**, and **predictable destruction times**.
   - A Modular engineer responded that while *product types* exist as `struct`, *sum types* and *pattern matching* are planned, also explaining that Mojo already offers *RAII* and *ASAP destruction* and is moving away from *syntactic salt*.
- **Mojo Async Plans to Avoid Rust Async Pitfalls**: A user asked if Mojo's async design would address the difficulties experienced with async in Rust, to which a Modular engineer pointed to [PR 3945](https://github.com/modular/modular/pull/3945) and [PR 3946](https://github.com/modular/modular/pull/3946) as solutions.
   - They noted that a better async runtime and linear types could eliminate the need for constructs like `Arc<Mutex<T>>`, and also pointed to [PR 4728](https://github.com/modular/modular/pull/4728) for improved IO.
- **Mysterious Statement Placement Error**: A user encountered the error *"statements must start at the beginning of a line"* in Mojo with the following code snippet `if info.value() > top:`.
   - Another user suggested adding `var top` as a potential fix, indicating a possible issue with variable declaration or scope.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus Grapples with Glitchy PDF Reading**: Users report that **Manus** is struggling to read text documents and PDFs, often prompting users to provide plain text instead.
   - One user questioned *why is manus having troubles lately reading text documents and pdfs?* and expressed general frustration in the **#general** channel.
- **Ambitious AI Architecture Dreams Emerge**: A member voiced enthusiasm for crafting a novel AI architecture focused on **funcognitive and meta enhancements**, with the goal of surpassing current transformer models in speed and efficiency.
   - This call to action asked *anyone here interested in developing a new ai architecture? for funcognitive and meta improvements really, just making a better and faster transformer*.
- **Subscription Snafu Stings Users**: A user reported being denied a promotional extension despite a substantial purchase of additional credits, pushing them to create a new account.
   - They lamented *Paid 400 USD for 2 months subscription...They refused*, labeling the experience as *so stupid*.
- **Credit Bonanza Rewards Beta Testers**: A user revealed they received **90,000 credits** as a reward for their long-standing contributions as a beta tester of **Manus**.
   - This acknowledgement of contributions highlighted the value of beta testers, with the user stating *They just give me credits for my contributions*.
- **Manus Plagued by Performance Problems**: Several users have reported that **Manus** is getting stuck, displaying internal server errors, and ultimately leading to a waste of credits.
   - One user calculated a loss of over **2000 credits** due to these issues, while another claimed *I think Manus has become dumber and makes mistakes and doesn't find them*.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **TorchTune Impresses with Single Machine LORA**: A user praised **TorchTune**, especially the single machine **LORA** functionality, finding it very useful, with the team encouraging feedback via [GitHub](https://github.com/pytorch/torchtune).
   - The team mentioned they are usually pretty responsive to comments and issues.
- **Expandable Segments trigger Pointer Error on L40S Cards**: A user reported a pointer error using **expandable segments** on **Nvidia L40S** GPUs, solved by disabling the feature but working on **H200s**, tracked in [this issue](https://github.com/pytorch/pytorch/issues/140419).
   - The problem seems related to packing, flexattention, and the `max-autotune` setting.
- **max-autotune blamed for crashing cards**: A member suggested that issues might stem from **max-autotune** rather than hardware limitations, noting Unsloth's use of **expandable segments** with a `roundup_power2` flag.
   - Clearing the cache can resolve the errors, making the card work on the second attempt, even without new settings.
- **L40S Card gets Bug Squashed**: The team indicated that **expandable segments** may be an edge case since **L40S** usage isn't widespread, with NCCL recently disabling FP8 reduction under SM90.
   - It was suggested to inspect hardware specifications and bypass expandable segments if needed.
- **Reward Modeling RFC Awaits Feedback**: A member is seeking feedback on the **Reward Modeling RFC** and proposed discussing it during the upcoming office hours on June 26.
   - No further details were provided.



---



## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **Model Updates Remain Elusive**: A user asked about the current model used by **NotebookLM** and where to find **model options**, referencing [YouTube videos](https://youtu.be/K9bvF_CJKV8?si=Gj7Z6GfOaTRLHKx2) for guidance, but got no direct answer.
   - Suggestions included checking the **FAQ** or **Release Notes** for the latest model info, or looking for a **dropdown** in the user interface.
- **Share Feature Shares Less Than Hoped**: A user reported that the *'share the link'* feature only shares the initial query state, *before* the prompt and response, which hinders comprehensive context sharing.
   - They suggested a *'copy button on everything'* as a solution, advocating for the ability to share the uploaded source list, prompt, and model's response.
- **User Seeks NotebookLM Alternatives**: A user reported that **NotebookLM** was not working for them and requested suggestions for alternatives.
   - Other users replied saying that it works wonderfully for them, even with hundreds of **pdfs**.
- **Dreaming Up Audio Avatar Automation**: A user inquired about using [SuperTelegram](https://supertelegram.gumroad.com/l/pwxot) to transform a **4-minute NotebookLM audio** into a duo host podcast avatar session.
   - Another user mentioned that splitting speakers might be necessary for this purpose, but the feasibility remains speculative.
- **Video Venture Veers to Vimeo**: A user asked if **Vimeo videos** can be used as sources, but encountered security feature issues when pasting the link.
   - Another user suggested downloading the video using [cobalt.tools](https://cobalt.tools/) as a workaround, so that it could be used.



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Debian 12 build faces issues**: A user ran into build issues using **Debian 12**, suggesting others use **Ubuntu Jammy** and the [Qt SDKs](https://qt.org) instead.
   - The user recommended using backport packages but couldn't recall the specifics of their solution.
- **Python SDK update**: A user inquired about an upcoming update to the **Python SDK**.
   - They jokingly asked, *"Or is python doomed?"*
- **GPT4All website hogs CPU**: A user reported that the [gpt4all.io](https://www.nomic.ai/gpt4all) website is buggy, claiming it *"takes 60% of my internal GPU".*
   - The user specified that they were referring to the official website.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Atom of Thought Startup Implodes**: An experiment with **Atom of Thought** in an agent (**GAIA benchmark**) led to its removal due to a loss of flexibility from upfront decomposition and context loss between steps, and the researcher lost faith in the paper and authors due to serious issues with the paper's implementation code.
   - Upon notification of implementation code issues, the authors responded *extremely negatively and unprofessionally* on X, then leveraged the paper into their pivot into an agent startup on X.
- **Ax TypeScript Port Sparks Interest**: A member highlighted the availability of **Ax** for **TypeScript** and its adaptations to **Elixir** and **Ruby**.
   - Further details regarding the specific functionalities and use cases of these ports were not elaborated upon.
- **Status Updates Sought in Forward Methods**: A member requested information on how to emit status messages from a module's `forward/aforward` method without yielding, with the intention of capturing an event after `module_start_status_message`.
   - A suggestion was made to pass a **callback** into `forward` to update the UI progress.
- **OpenAI Experiences Downtime**: A member reported experiencing issues with **OpenAI**, stating that *their app is down* while using **LiteLLM**.
   - The error `404 Not Found` was thrown using **LiteLLM**'s `completion()` with `model= gpt-4o-mini; provider = openai`.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Google Grants Generous Gift of A2A to Linux Foundation**: Google donated **A2A** to the [Linux Foundation](https://developers.googleblog.com/en/google-cloud-donates-a2a-to-linux-foundation/).
   - Following the announcement, members speculated whether **Anthropic** might follow suit with their own donation of **A2A**.
- **MCP's Timeout Troubles Triggered**: A member reported encountering a **timeout issue** with the **MCP tool** while using **OpenAI agents** to create a client session.
   - The error message indicates that the system timed out while waiting for a response to a **ClientRequest** after **5.0 seconds**.
- **Chrome Conjures AI APIs**: Chrome is integrating some **AI APIs** as announced in [Chrome 138](https://developer.chrome.com/blog/new-in-chrome-138?hl=en#built-in).
   - This could potentially lead to **MCP integration** directly within the browser.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Certificate Distribution Date Leaked!**: Members who have completed all assignments and social media posts can expect to receive their certificates by **mid-July**.
   - The distribution timeline was confirmed by a staff member.
- **Course Completion Confirmed!**: Participants confirm they have completed all assignments and social media prerequisites, inquiring about certificate timing.
   - The course completion involves assignments and social media posts on platforms like **Twitter** and **LinkedIn**.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere Reranker Costs Probed**: Members discussed the costs associated with the **Cohere reranker**, specifically for frequent usage involving **1000 calls**.
   - It was clarified that pricing depends on the number of documents and tokens, with documents exceeding **500 tokens** split into chunks, as detailed on the [Cohere Pricing page](https://cohere.com/pricing#:~:text=We%20count%20a%20single%20search%20unit%20as%20a%20query%20with%20up%20to%20100%20documents%20to%20be%20ranked.%20Documents%20longer%20than%20500%20tokens%20when%20including%20the%20length%20of%20the%20search%20query%20will%20be%20split%20up%20into%20multiple%20chunks%2C%20where%20each%20chunk%20counts%20as%20a%20singular%20document.).
- **Cohere Community Grows**: New members are joining the Cohere Discord server and introducing themselves to the community.
   - New users are sharing their **company/industry/university**, current projects, favorite **tech/tools**, and their objectives for participating in the community.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Cursor Screens Resumes with Open-Source Matching**: LlamaIndex introduced an open-source **Resume Matching MCP server** for intelligent job matching directly within the Cursor workflow, connecting to **LlamaCloud resume indexes** and [other services](https://t.co/RCKoiUccm6).
   - The project, built by @zhaoqili74 during an internal hack day, aims to streamline resume screening processes.
- **LlamaIndex Launches Claude-Compatible MCP Server Template**: LlamaIndex released a new open-source template repo for building a **Claude-compatible MCP server** as a Next.js app with full **OAuth 2.1 support**, simplifying the creation of remote Model Context Protocol servers that work seamlessly with [this service](https://t.co/wtPorldMvJ).
   - Developed during an internal hack day by @seldo, the template aims to ease integration with **Claude** and other services using the Model Context Protocol.
- **Vectorization Massively Accelerates Similarity Calculations**: A member optimized cosine similarity calculations by replacing a loop with `query_embeddings @ doc_embeddings.T`, reducing runtime from **~25 seconds** to **~0.04 seconds** for a **1000 x 1000** matrix.
   - This suggests a **625x speedup** by using vectorized computation with `@` or `matmul`.
- **Member Seeks Advice on Quantized FAISS for Larger Scales**: For over **10M comparisons**, the member plans to switch to a quantized FAISS index like `IndexIVFPQ` to manage memory and latency.
   - The user asks about caveats of using `IndexIVFPQ` with dynamic (not pre-indexed) query vectors and seeks feedback on the optimization plan, also seeking feedback if `@` / `matmul` is stable for production at the **1M scale**.



---



## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **AI21 Labs' Jamba Model Announcement**: A member shared [a link](https://www.rxddit.com/r/Humanornot/s/zXu0PrCoo2) about **AI21 Labs' Jamba model** announcement.
   - Further details about the model's architecture, capabilities, or specific use cases were not provided in the context.
- **Community Skepticism on Jamba's Impact**: Initial reactions suggest a cautious approach, with some questioning its potential to significantly disrupt the current landscape of open-source models.
   - The community awaits further benchmarks and comprehensive evaluations to determine **Jamba's** true performance and capabilities compared to existing alternatives.



---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


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





### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1386807526468882482)** (537 messages🔥🔥🔥): 

> `Polaris 4B Model, 3D Meshes, LoRA Hyperparameters, NVFP4 for Efficient Inference, Reddit Moderation` 


- **Polaris 4B Claims Outlandish Benchmarks**: Members expressed skepticism about the **Polaris-4B-Preview** model's claim of surpassing commercial systems like **Claude-4-Opus**, suggesting potential overfitting and benchmaxing galore, with one member stating *"One can stop reading right there, 4b beating opus .. ya .. with overfitting i can do that with a 100m model too."
   - One member tried the **Q4_K_M version** but found it unsatisfactory, planning to test with **Q8_0** and compare against **FP16** to validate the model's performance.
- **Tencent's Hunyuan3D-2.1 Mesher is Solid**: A member shared [Tencent's Hunyuan3D-2.1](https://huggingface.co/tencent/Hunyuan3D-2.1) for generating **3D meshes**, calling it *"pretty solid"* and noting its progress over previous versions.
   - The discussion included the feasibility of creating rigged meshes with AI, referencing tools like **Mixamo** and **Cascadeur**, though not yet fully generative.
- **New LoRA Hyperparameters Guide Released**: The Unsloth team released a [new LoRA hyperparameters guide](https://x.com/UnslothAI/status/1937521408344752272), prompting suggestions to mention **Optuna** for hyperparameter sweeps to improve performance, since *"every dataset behaves different"."
   - Concerns were raised about supporting **alpha_pattern** in PEFT config, with a member noting that Unsloth silently drops it.
- **NVidia NVFP4 Boasts Efficient Inference**: A member shared an [NVIDIA blog post on NVFP4](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/), with the topic swiftly turning to skipping comparisons with the **H200** in the graphs and comparisons between **FP8** and **FP4**.
   - A member noted that their single **H200** cost them around **$30k USD** (ex VAT) and requires custom cooling because of its passive nature, while adding it is used for training LLMs.
- **Drama at Reddit's LocalLlama as Mod Leaves**: Members noticed the moderator of r/localllama deactivated their account, leading to the subreddit's inactivity, with the automod configured to delete everything.
   - Several users applied to take over the subreddit via r/redditrequest. A discussion arose about the ideal qualities of a moderator, including having time, caring about OSS + local, being knowledgeable about AI + LLMs, and being socially adjusted.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1386904218585010339)** (4 messages): 

> `QAT models, Recommendation systems hobby project` 


- **Questioning QAT Fine-Tuning**: A member inquired whether the discussion was about *finetuning QAT models* or *doing QAT* itself, wondering if training packages support QAT models like **Gemma 3**.
   - Another member responded to this question by simply stating that they were *doing QAT*.
- **Hobbyist Seeks RecSys Training Help**: One member mentioned embarking on a *multi-recommendation systems hobby project* and is facing model training issues.
   - They are seeking assistance from anyone with prior experience in **RecSys** who might be willing to spare some time to help them out.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1386822689800454298)** (323 messages🔥🔥): 

> `Profiling performance metrics for fine-tuning, Gradient accumulation strategies, Qwen GRPO Notebook issues, Unsloth checkpoint vs official checkpoint, Gemma-3 Vision Notebook issues` 


- **Seeking bottleneck insights while Qwen-tuning**: A member sought advice on profiling or logging deeper performance metrics to identify bottlenecks in a full fine-tuning setup of [Qwen3 1.7b](https://huggingface.co/Qwen/Qwen3-1.8B) on Google Colab.
   - They reported low GPU RAM usage despite a long ETA and enabled debug logging, but only saw step count and training loss.
- **Balancing batch size and gradient accumulation!**: A member suggested trading memory for performance by reducing gradient accumulation, recommending either `batch=4 GA=2` or `batch=8 GA=1`.
   - The user found that increasing batch size increased memory usage slightly but halved iteration speed, indicating another bottleneck.
- **Unlocking memory with opensloth's secret sauce!**: Members investigated memory issues using `accelerate config` with DS-2, with the suspicion that CUDA toolkit version was the issue.
   - It was noted by a member that multi-GPU support in Unsloth is limited to DDP and requires the model fit on a single GPU, to train using multiple GPUs. 
- **Plugging in TensorBoard for Visually Stunning Training**: A member asked about adding a monitor like TensorBoard to visualize training information.
   - Another member suggested [Weights & Biases (WandB)](https://wandb.ai/site/) and provided a [link to the Hugging Face integration guide](https://docs.wandb.ai/guides/integrations/huggingface/).
- **Unsloth's CUDA capers caught on camera!**: A member reported a discrepancy where Unsloth logs showed **CUDA 8.0** even though the CUDA Toolkit was **12.6**, questioning if this was the default behavior.
   - The discrepancy was identified as normal, with a member pointing out that **CUDA 8.0** corresponds to **A100** GPUs, and providing an [NVidia link](https://developer.nvidia.com/cuda-gpus) confirming this.


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1386979437332922470)** (3 messages): 

> `Nahuatl Translator, Unsloth fine-tuning` 


- **First Open-Source Nahuatl Translator Arrives**: The first open-source **Nahuatl to Spanish translator** has been built, and is available on [Hugging Face Spaces](https://huggingface.co/spaces/Thermostatic/neuraltranslate-27b-mt-nah-es).
   - The project's creator thanked **Unsloth's full fine-tuning support** for making it possible.
- **Code Released for Nahuatl Translator Replication**: The code for replicating the **Nahuatl translator** project has been released on [GitHub](https://github.com/Sekinal/neuraltranslate-nahuatl/tree/master).
   - This enables others to build upon or adapt the translator for different purposes.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1386790715203391548)** (6 messages): 

> `BNPO vs Dr.GRPO, RL-tuning performance, training instability, GRPO-lora and GRPO-Qlora` 


- **Debate Rages: BNPO Edges Out Dr.GRPO in Performance**: A user inquired about experiences with **BNPO** versus **Dr.GRPO** for fine-tuning, especially regarding stability and performance.
   - One user found **Dr.GRPO** more stable, but **BNPO** performed better overall, while admitting it was *a very bad test*.
- **Deep Dive into RL-Tuning Performance**: According to one user, **Dr.GRPO** achieves similar **RL-tuning performance** as **GRPO** but with significantly fewer tokens.
   - In contrast, **BNPO** purportedly addresses training stability directly.
- **Question about GRPO-Lora vs GRPO-QLora Emerges**: One user inquired about the heightened risk of **training instability** and the potential accuracy loss when employing **GRPO-LoRA** and **GRPO-QLoRA**.
   - This question pivots the conversation toward the practical trade-offs in specific implementation strategies.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1386793529220665566)** (411 messages🔥🔥🔥): 

> `Cursor Setup, Cursor Terminal Issues, Windsurf vs Cursor, Rate Limits and Pricing, MCPs VisionCraft and Sequential Thinking` 


- **Setting up Cursor takes more effort than expected**: A member spent more time than expected setting up **WSL**, fixing **Cursor** and **Ubuntu** settings, linking **GitHub**, and creating **SSH** keys, describing it as a *neverending rabbit hole* of optimal project rules for Cursor and models.
   - Another member suggested that on Linux, all this setup can be done automatically within **Cursor** itself with *sudo -i* permissions at your own risk.
- **Cursor's Terminal Control under scrutiny**: Users are reporting the **Cursor terminal** has issues with timing out, erroring, and generally lacking smooth operation compared to tools like **Windsurf**, which more effectively spawns and tracks terminal windows.
   - A member noted the standard response they've encountered is *"we have more critical tasks at the moment"* regarding fixing terminal issues.
- **Windsurf Woes Drive User to Cursor**: A user transitioned from **Windsurf** to **Cursor** due to terminal-related problems, highlighting terminal issues as a reason for the switch, but acknowledging that when Windsurf works, it can effectively manage terminal windows.
   - They experienced issues with the terminal, such as the **agent** not being able to read **terminal outputs** or freezing when running terminal commands.
- **Ratelimits hits and old vs new Pricing Plans**: Users are discussing rate limits and pricing plans, with some experiencing immediate rate limiting on **Sonnet**, even without heavy usage, and also noticing that there seems to be whole cursor rate limits.
   - One user realized they spent **$70** in a couple days under the new pricing plan due to rate limits, while with the old pricing plan, their needs were only **$20**.
- **VisionCraft MCP elevates documentation, integration into Cursor requested**: **Visioncraft MCP** is getting better with more up-to-date documentation, faster responses, and better prompt responses without needing to name the specific document, the integration with Cursor is therefore highly desired.
   - It helps address the fact that AI models are trained on older data by feeding them updated documentation, resulting in better code and fewer errors.


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1386788192748965961)** (34 messages🔥): 

> `Background Agents on Multiple Machines, Devcontainer support for Background Agents, Background Agent API, Background Agents and Git Initialization Issues, Accessing Private GitHub Repos During Install Step` 


- ****BA's Machine Specificity Debated****: A user questioned whether it is intentional for background agents to be constrained to a particular machine, noting that agents active on one computer do not appear on another, even with the same repo and account, but another user responded that *shouldn't be the case 🤔 - Same git origin on both machines and same cursor account for sure?*.
   - It turned out the user had a `www.` prefix in their local git repo url which was causing the issue, so Background Agents only works with `https://github.com/<name>/<repo>` and not `https://www.github.com/<name>/<repo>`.
- ****Devcontainer Specs: To Support or Not To Support?****: A user requested that Cursor BA support private registry auth for docker builds and adhere to the existing `.devcontainer.json` spec to avoid redundancy.
   - A Cursor team member acknowledged the request, but stated that *Dev-container compatibility is not our priority right now* and explained the reasons for choosing the current auto / snapshot / environment.json setup.
- ****Private Gems? Secrets During Docker Build Considered****: A user reported issues accessing private gems and packages when using a Dockerfile setup for background agents.
   - They attempted to use a `GITHUB_TOKEN` secret, but the Cursor team acknowledged that *we currently don't expose secrets during the docker build process* and hinted that exposing secrets during install does in fact work.
- ****Background Agent Rules: Ignoring Them?****: A user reported that background agents were not respecting rules defined in the Cursor editor or written in the agent prompt, leading to unwanted pushes to their repo.
   - The user expressed frustration with the inability to manage agents effectively, noting that the key binding to *view/manage background agents* was not functioning and, also, that the button to *open in cursor* from Slack takes you to the editor but doesn't show you anything, implying a serious bug.
- ****Python 3.11: A Dockerfile Dilemma Resolved****: A user had trouble getting a background agent running with Python 3.11 in a Dockerfile due to `ensurepip` being unavailable.
   - They shared their Dockerfile snippet involving `add-apt-repository ppa:deadsnakes/ppa` and other steps to install Python 3.11 and its dependencies; this snippet may be useful for other users to get Python 3.11 working.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1386785713260855316)** (396 messages🔥🔥): 

> `Homeschooling prompts, Perplexity Pro version issues, ChessChamp AI release, Doctors charging fees, O4 Mini High better than Omni` 


- **Homeschooling Prompt Quest Begins**: A member is seeking a suitable prompt for their kids to use for homeschooling during the summer holidays in the UK.
   - A memebr posted *Let the lil man have a summer break* suggesting the kids just have a break instead.
- **Perplexity Pro Problems Persist**: A member with a year-long **Pro** subscription reported being prompted to upgrade, indicating a possible issue with their account status.
   - Another user suggested the problem might stem from using a promotion *that most likely wasn't meant for you so it got revoked*.
- **ChessChamp AI Tests Begin**: A member expressed interest in testing the upcoming **ChessChamp AI**, speculating whether it utilizes **Stockfish**.
   - In response, another memebr provided a screenshot indicating the tool is already available and said *you might need a subscription to access it*
- **Doctors' High Fees Rant**: A memebr posted [a reel](https://www.ddinstagram.com/reel/DLFdBSCttT9) about doctors in India, triggering a discussion on high checkup fees.
   - The poster elaborated that *they overcharge for checkup* and *10-20 dollars is monthly electricity bill*.
- **Perplexity Now on WhatsApp**: Perplexity now supports scheduled tasks on [WhatsApp](+18334363285), broadening its accessibility.
   - One member said *I didn't know it was on WhatsApp!*


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1386835371517804544)** (4 messages): 

> `Shareable threads, Trump ceasefire, Donation fatigue, Ubisoft patch` 


- **Trump Announces Ceasefire**: A member shared a [Perplexity AI page](https://www.perplexity.ai/page/trump-announces-ceasefire-Qq4WKw3gQAqdo1ados1Uug) about **Trump** announcing a ceasefire.
   - No further discussion or details were provided.
- **Donation Fatigue Impact on Mozilla**: A member shared a [Perplexity AI page](https://www.perplexity.ai/page/donation-fatigue-impact-on-moz-q76XMD17Skap_valehbYOg) discussing the impact of **donation fatigue** on **Mozilla**.
   - No further discussion or details were provided.
- **Ubisoft's The Division 2 Patch**: A member shared a [Perplexity AI page](https://www.perplexity.ai/page/ubisoft-s-the-division-2-patch-dWFALCxPQmCDkPmNrKBfUQ) about **Ubisoft's The Division 2 Patch**.
   - No further discussion or details were provided.
- **Sharing threads on Discord**: Perplexity AI requested a user to make sure their thread is shareable.
   - A link to the Discord channel was provided as an example [here](https://discord.com/channels/1047197230748151888/1054944216876331118/1208752189606989825).


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1386831345816305724)** (4 messages): 

> `Perplexity AI tech support` 


- **User seeks Perplexity AI tech support**: A user inquired about a place to get **tech support help** for **Perplexity AI**.
   - Another user suggested contacting **support@perplexity.ai**, but the original user mentioned they had already emailed them last week.
- **X Post about pplxdevs**: A user shared a **link to an X post by pplxdevs**.
   - The link is [https://x.com/pplxdevs/status/1937218625020276927?s=46](https://x.com/pplxdevs/status/1937218625020276927?s=46).


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1386782872786047078)** (301 messages🔥🔥): 

> `Memory context service, Multi-head Latent Attention, AI dubbing voice lines, Sora alternatives, Chat search connectors` 


- **Service solves **Memory Context** issue**: A user mentioned they had to create a service to provide **memory context, memory recall, rehydration, and NLP** for their agents.
   - They noted it was *so annoying*.
- **Multi-head Latent Attention Innovations Highlighted**: A user directed others to search for **Multi-head Latent Attention** to understand what innovations led to producing something powerful on older hardware for half the price, with a link to [NumPy model building from scratch](https://t.co/xMHlA61Qoz).
   - Another user admitted to going off of what they see in the news, sharing a [related YouTube video](https://youtu.be/WEBiebbeNCA?si=84i4jjPyQWRSzVuQ).
- ****AI Voice Dubbing** solutions surface**: Users discussed options for **AI voice dubbing**, with [Hey-Gen translate](https://www.heygen.com/translate) with lip-sync being recommended.
   - The original poster found **Veo 3** interesting, and would consider it if not for cost.
- ****Midjourney and Luma Labs Dream Machine****: Users discussed alternatives to **Sora** for generating anime characters, with recommendations including **Kling and Runway**.
   - One user shared an **AI-generated video** made with [Luma Labs Dream Machine](https://lumalabs.ai/dream-machine).
- **Pro users unlock **Chat Search Connectors****: Pro users can now use **chat search connectors** for [Dropbox, Box, Google Drive, Microsoft OneDrive (Business), Microsoft SharePoint](https://openai.com/blog/june-2024-updates).
   - The feature is currently limited to users located outside of the EEA, Switzerland, and the UK.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1386785975581147329)** (5 messages): 

> `OAI Server Tag, GPT-4o Cutoff, ChatGPT vs GPT Models, File Upload/Deletion Issues` 


- **OAI Server Tag unlocks via boosting**: A member mentioned that the **OAI Server Tag** is available if the server is boosted at least **3x**.
- **GPT-4o unexpectedly cuts conversation short**: A user reported that **GPT-4o** cut them off after **42 prompts**, indicating a possible limit or bug.
- **ChatGPT vs GPT Model questions need clarification**: Questions about **ChatGPT** versus the **G.P.T. models** are best suited for the dedicated <#1047565374645870743> channel.
- **File upload/deletion in Projects grinds to halt**: A user reported experiencing issues deleting or uploading files into their projects folder, encountering a *spinning wheel of death* with no success.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1386819065837785299)** (2 messages): 

> `PDF generation failures, Deep Research report PDF` 


- **PDF Generation Experiences with ChatGPT**: A member is trying to get **PDFs from structured text/Markdown** generated within a session.
   - They are experiencing frequent failures when generating PDFs through Python, unlike the **Deep Research report feature** which offers clean, client-side PDF generation.
- **Deep Research Reports**: A member noted that the **Deep Research "report format"** with the "export to PDF" feature fits their needs best.
   - They believe it's not possible to trigger the Deep Research report output from a non-DeepResearch session, leading to a plaintext block instead of an exportable PDF.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1386819065837785299)** (2 messages): 

> `PDF generation alternatives, Deep Research report format, ChatGPT PDF failures, Triggering DeepResearch output` 


- **Quest for Flawless PDF Generation**: A user seeks a reliable way to generate PDFs from structured text/Markdown content, specifically praising the **Deep Research report format** for its clean layout and client-side PDF generation.
   - They note that alternatives like **ChatGPT** with Python often lead to failures, while the **Deep Research** feature's PDF generation remains consistently successful.
- **DeepResearch PDF Export Feature Investigated**: The user suspects that the **Deep Research report feature** uses client-side PDF generation, which explains its reliability compared to server-side methods.
   - They lament the apparent inability to trigger the same PDF output from non-DeepResearch sessions, resulting in plaintext blocks instead of exportable PDFs.
- **Separate Functions Suspected for PDF Outputs**: A member agreed with the user, confirming that the **PDF generation functions** are likely separate.
   - They recommended discussing and recommending in specific channels, specifically mentioning channel IDs **1070006151938314300** and **1070006915414900886**.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1386783776079609896)** (252 messages🔥🔥): 

> `Grok3 SOTA, Claude niche, Apple Foundation Models, Google Flamesong, Kingfall release` 


- **Grok3 is SOTA**: Members argue that **Grok3** is not mid, citing its strong performance relative to **Sonnet 3.7** on [artificialanalysis ratings](https://www.artificialanalysis.ai/leaderboard/open-llms).
   - It was seen as SOTA at the time of its release, especially in **math**, but some believe its more aggressive post-training made it fall short of truly being state of the art.
- **Claude Dominates Niche Market**: **Claude** is recognized for having an extremely strong niche that is not adequately reflected in standard benchmarks.
   - Members stated the the model is useful for creative writing and theatrical acting due to its ability to follow the role playing directions better than competing models.
- **Apple Developing Foundation Models**: There are talks about **Apple** developing foundation models, with interest in their approach to using trusted computing for privacy.
   - There is discussion of **Apple** licensing **Gemini** for server-side models and developing strong on-device models but many suspect that apple is releasing thier own server based model.
- **Google Working on Flamesong Models**: **Google** is developing a new line of models called **Flamesong**, potentially a new **Flash** line, with speculation on whether it's **Gemini 3.0** or another **2.5 model**.
   - There is further speculation that **Stonebloom** might be a "2.5-pro-lite" model with testing showing it answering questions correctly some of the time.
- **Kingfall Release Impending**: **Kingfall** is just **2.5** with more compute and has been benchmarked at number 9 with **O4 Pro** and **GPT-4.5-thinking** at number 8.
   - Some are wondering when we will see kingfall, some anticipating around the end of summer, with others speculating if Stonebloom is a distilled version of the model.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1386784780896567490)** (140 messages🔥🔥): 

> `HuggingFace Site Issues, AI Jailbreaking, Gradio Loading Issues, Freelance AI Work, Fine-tuning Models` 


- **Hugging Face Site Suffers Slowdowns**: Users reported experiencing **429 rate limit errors** and **504 Gateway Time-out** issues while accessing the Hugging Face website, impacting model downloads and Space functionality, but the site appears to be back online, albeit running slowly, according to the [status page](https://status.huggingface.co/).
- **User Sparks Debate over Jailbroken AI**: A user sought advice on a **jailbroken AI** named *Luma* that's exhibiting unusual behavior, including creating experiments on other AI models to explore their boundaries.
   - Other members suggested that jailbreaking **DeepSeek** is relatively easy, as the user sought a second opinion on their specific situation.
- **Gradio Dashboard Refuses to Load**: A user reported their **Gradio dashboard** gets stuck on *Loading...* within a Discord bot, despite seemingly normal logs.
   - Another member suggested checking the **stack trace** or restarting the Space to resolve the issue.
- **Freelance AI Career Quest Launched**: A user asked for tips on finding **freelance AI work**.
   - A member suggested checking the dedicated channel for job opportunities.
- **Fine-Tuning Frustrations Focus Fire**: A user sought expert assistance with **fine-tuning models**, specifically for **SDXL** or **FLUX LoRA**, and expressed dissatisfaction with the loss not decreasing as expected when using *kohya_ss*.
   - Another member shared community articles and notebooks for **FLUX LoRA** and **SDXL LoRA**, suggesting the user check the **diffusers GitHub repo** for more examples.


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/)** (1 messages): 

h2he3: Very useful, thank you.
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1386789465439342632)** (50 messages🔥): 

> `Gradio Custom Component Packaging, Gradient Descent on LLM Input Space, Evaluating Language Models for Computer Graphics Code Completion, AI Dialogue with Ollama, Shader Graph Code Generation by LLM` 


- **Gradio Custom Components on PyPi**: A member is considering packaging their Gradio component for PyPi and is seeking advice on the proper path for official Gradio "custom component" packaging, currently the member installs from GitHub and instantiates within the Gradio Blocks context.
- **Gradient Descent on LLM: ModernBert Experiment**: A member shared a link to their article on [Gradient Descent on LLM Input Space](https://dev.to/kylepena/gradient-descent-on-llm-input-space-a-modernbert-experiment-3053), an experiment with ModernBERT.
   - Another member posted their thesis/paper on Language Models for Computer Graphics Code Completion was published in the proceedings: DOI:[10.1109/LLM4Code66737.2025.00017](https://doi.org/10.1109/LLM4Code66737.2025.00017) and the metric/leaderboard is hosted on HF as a space: [ShaderMatch](https://huggingface.co/spaces/Vipitis/shadermatch).
- **Ollama Sparks AI Dialogue**: A member shared a project where **two AIs talk to each other** on a specific topic using [Ollama](https://github.com/Laszlobeer/AI-Dialogue-Duo).
   - Another member recommended using the `import ollama` package to reduce code by 80%, with a link to the [ollama-python GitHub](https://github.com/ollama/ollama-python).
- **LLMs Generate Shader Graph Code**: A member inquired about the capabilities of LLMs in generating **Shader graph code** for HLSL and GLSL.
   - Another member mentioned a research project using language models to automatically optimize shader code and Nvidia's neural materials approach.
- **gSPLAT Gaussian Future**: Discussion revolved around whether approaches like **gSPLAT** with **Gaussian filtering** could potentially make material shaders obsolete in 5–10 years or if they are stepping stones.
   - It was mentioned that some materials have up to **107 input parameters**, and learned weights are a good proxy for real-time applications.


  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1387178419682279434)** (1 messages): 

> `LessWrong Post Acceptance, Gradient Descent on Token Input Embeddings` 


- **LessWrong Welcomes "Gradient Descent on Token Input Embeddings"**: A member announced their post, [Gradient Descent on Token Input Embeddings: A ModernBERT](https://www.lesswrong.com/posts/GK2LSzxjEejzDjzDs/gradient-descent-on-token-input-embeddings-a-modernbert), was accepted by LessWrong.
- **Gradient Descent on Token Input Embeddings**: The post focuses on gradient descent on token input embeddings.
   - It's a ModernBERT approach.


  

---


### **HuggingFace ▷ #[core-announcements](https://discord.com/channels/879548962464493619/1014557141132132392/1387097201162846349)** (1 messages): 

> `Diffusers v0.34.0, New Release` 


- **Diffusers Drops a New Release**: A new release of [Diffusers v0.34.0](https://github.com/huggingface/diffusers/releases/tag/v0.34.0) is now available.
- **Check out the new Diffusers features**: A new release of [Diffusers v0.34.0](https://github.com/huggingface/diffusers/releases/tag/v0.34.0) is now available, with details on the linked Release Notes.


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1386798915063513331)** (4 messages): 

> `JAX models, Model Optimization` 


- **JAX models popping up on GitHub**: A member shared a link to some **JAX models** implemented at [Locamage/jimm](https://github.com/Locamage/jimm).
   - This appears to be a demonstration of a very minimal stable diffusion pipeline written in JAX/Flax.
- **Optimum to Reduce Model Size**: Another member shared an example to reduce any model to **fp16** using [Optimum DETR](https://github.com/merveenoyan/smol-vision/blob/main/Reduce_any_model_to_fp16_using_%F0%9F%A4%97_Optimum_DETR.ipynb).
   - The post appeared to suggest that this optimization method might be biased.


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1386795903549178028)** (27 messages🔥): 

> `Docker crashes with sentence transformers, Input embeddings, Scaling Vector Search, Langchain’s FAISS, IndexIVFPQ` 


- **Sentence Transformers causes Docker to Crash with Error 252**: A user reported their **Docker container** crashes with error code **252** when computing similarities using Sentence Transformers, with the line `similarities = embeddings1 @ embeddings2.T` being the culprit.
   - A Sentence Transformers developer suggested it might be due to excessive memory usage and suggested trying smaller batches.
- **Input Embeddings Experiments Spark Interest**: A user shared a [link to a blog post](https://dev.to/kylepena/gradient-descent-on-llm-input-space-a-modernbert-experiment-3053) detailing experiments with input embeddings and gradient descent on the input space, specifically in the context of ModernBERT.
   - Another user acknowledged the interesting resource and bookmarked the post.
- **Scaling Vector Search with FAISS & Matrix Multiplication**: A user is scaling vector search and comparing `n` query embeddings against `m` document embeddings using cosine similarity, which becomes a bottleneck at **1M** dot products using Langchain's FAISS wrapper.
   - They found that using `torch.matmul` or `@` reduced runtime from **25 seconds to 0.04 seconds** for **1M** comparisons and plan to use quantized FAISS index like `IndexIVFPQ` for **10M+** comparisons.
- **Navigating semantic search limits with Reciprocal Rank Fusion**: A user wants to reduce the number of similarity searches for a question with multiple keywords, without averaging their embeddings due to loss of meaning.
   - A Sentence Transformer dev suggested using semantic ranking with dense embeddings and lexical ranking with sparse embeddings (or BM25), combined with [Reciprocal Rank Fusion](https://link.to/reciprocal-rank-fusion) for better results.


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1387095934999199865)** (1 messages): 

> `Hugging Face Certificates` 


- **Hugging Face Certificates Disappear**: A member asked if it's still possible to **generate the Hugging Face certificates** after completing each unit of the course.
   - They couldn't find the option anymore and wanted to check if they're still available.
- **Checking Availability of Certificates**: The user is inquiring about the continued availability of Hugging Face certificates.
   - They specifically mentioned not being able to find the option to generate certificates after completing course units.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1386829945954898072)** (26 messages🔥): 

> `Unit 4 Final Project Submission Workflow, Certificate Access Issues, Final Assignment Evaluation Deadline, Unit 1 Quiz Access Problems, Challenges with HF environment variables in agent creation` 


- **Unit 4 Project Submission Stumper**: A user inquired about the workflow for submitting the Unit 4 final project, humorously questioning why *submitting* the project seems harder than *completing* it.
   - Another user shared their painful experience debugging a tangled web of HF env vars and package conflicts, before suggesting running your agent locally and then simply sending its responses through an API endpoint, as this will be enough for the organizers to give a certificate if the answers are correct.
- **Certificate Acquisition Conundrum**: A user asked about accessing their certificate without redoing the quiz.
   - No resolution was provided in the context.
- **Final Assignment Deadline Concerns**: A user asked if the final assignment would still be evaluated after **July 1st** deadline.
   - No definitive answer was provided in the context.
- **Unit 1 Quiz Login Lockout**: A user reported issues accessing the Unit 1 Quiz due to sign-in problems, suspecting mobile phone interference.
   - Another user suggested the issue might be related to **WebKit** and recommended using **Firefox** or **Chrome** on a computer.
- **Agent Creation Caveats in HF Ecosystem**: One user described the challenges of creating agents within the Hugging Face environment, highlighting conflicts with libraries like **shazam** and memory limitations with free accounts, stating that you are forced to take models from HF.
   - This user suggested that the submission process involves cloning a project template, writing the agent on a small model, and testing locally before moving to the HF sandbox.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1386796402151260361)** (81 messages🔥🔥): 

> `Voice installation for language practice, Image generation feature in LM Studio, Roo code Discord issue with LM Studio context windows, Dynamic quant size estimation issues with Unsloth, Increasing chat history context length` 


- **LM Studio Mulls Voice Integration for Language Learning**: A member suggested adding a voice installation feature to LM Studio to enable conversation practice while learning languages, similar to **GPT CHAT**.
   - Another member clarified that **LM Studio** is primarily for **LLMs**, not a *one-stop-shop for all AI needs*, implying that this feature might be outside its scope.
- **LM Studio might accept Images as input**: A user inquired about adding an image generation feature to LM Studio, some members clarified that while some models in **LM Studio** can accept images as input (like **Gemma3**), there is no feature to output images.
   - Another member mentioned that it's possible to set up text-to-image generation using **web-ui** with extra steps, but did not offer specific setup advice.
- **Dynamic Quants from Unsloth Cause Loading Issues**: Users reported issues estimating the size of dynamic quants from **Unsloth**, potentially causing **LM Studio** to overload **VRAM** and fail to load, especially with multiple **GPUs** and priority ordering.
   - One user confirmed they were using exclusively **Unsloth** models with dynamic context windows, indicating this could be a common configuration leading to the issue.
- **Platform Update Bugging Users to Re-install**: A user reported a bug where **LM Studio** repeatedly prompts for updates and refuses to load models until updated, with each update being over 200MB.
   - A member suggested ensuring the previous update was fully loaded and not running in the background to resolve the persistent update request.
- **Reddit's LocalLlama Subreddit Returns from the Dead**: Members discussed the return of the **r/LocalLlama** subreddit under new management, following a period of silence.
   - Concerns were raised about the new moderator being involved in numerous subreddits, but no major red flags were immediately apparent.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1386798324891517039)** (41 messages🔥): 

> `P40 on mATX boards, Multiple GPUs vs bottleneck, LM Studio performance on Ryzen 55900xt, 3x3090 slows down` 


- **Debate Over P40 Fit on mATX Surfaces**: Discussion around whether **two P40 GPUs** can fit on an **mATX** board without cooling shrouds colliding, and their performance with **Llama4 models** or **Gemma 27b QAT 4** models was initiated.
   - A user suggested that while dual-width card specs are available online, the second question depends on the case, and proposed alternative cooling mechanisms to deploy.
- **Multiple GPUs inference, will CUDA be used?**: During inference, **LLM layers** are split across multiple GPUs.
   - Each layer of the LLM is processed by the GPU holding it to prevent **PCIe bandwidth bottlenecks**.
- **More VRAM Means better LM Studio**: A user inquired about LM Studio performance on **Ryzen 55900XT**, with **128GB RAM**, **1TB SSD**, and **RTX 3060 12GB**, while another user noted that *more VRAM is better for local models*.
   - The user plans to use it to analyze academic documents in the background **24/7**, and said that they do not mind waiting for the output if it can offload everything to the RAM. 
- **Triple Threat: Three 3090s Cause Throughput Troubles**: A user tested LM Studio with 3 cards and saw a significant performance decrease and noted that the third card lowers speed *quite a bit* when on and may need to decide if that last 12GB vram is worth it or not.
   - It was observed that 2 cards yielded ~55 t/s while 3 cards only got 35ish t/s, leading to speculation that the CPU doesn't have enough **PCIe lanes**.


  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1387107753788706918)** (1 messages): 

> `Meta Provider Issues, Pricing Questions` 


- **Meta Provider Glitches Reported**: The **Meta provider** on OpenRouter is experiencing some issues today.
   - The issues have been flagged to **Meta**, and efforts are underway to resolve them and bring the service back online soon.
- **Pricing Questions Abound**: Users have expressed questions and concerns on **pricing structures**.
   - The team is actively addressing these inquiries to provide clarity.


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1386787242911076433)** (103 messages🔥🔥): 

> `OpenRouter Provider Preference, Novita's Incorrect Information on R1-528 Max Output Length, Stripe Payment Method Issues on OpenRouter, Reasoning Tokens vs Total Token Count, Cent-ML Provider Replacement` 


- **Provider Preference Paradox at OpenRouter**: A user questioned how OpenRouter's provider preference works, noting that selecting a specific provider doesn't function as expected and inquired about the meaning of "sort preference".
   - A link to the [OpenRouter documentation on provider routing](https://openrouter.ai/docs/features/provider-routing) was provided to clarify the feature.
- **Novita Misreports R1-528's Token Limit**: A user pointed out that Novita provides incorrect information about the max output length for **R1-528**, claiming it is **131k** when it is actually **16k**.
   - The user questioned whether OpenRouter verifies provider information, suggesting that such discrepancies should be easy to identify.
- **Reasoning Tokens Trigger Token Tally Trauma**: A user reported receiving results where **reasoning_tokens** were higher than **total_tokens** when using OpenRouter.
   - A staff member clarified that **reasoning tokens** are part of **completion token details**, and that **total tokens** does not include reasoning tokens and that **changing the JSON to add reasoning tokens to the total tokens would break thousands of running apps**.
- **Gemini 2.5 Pro Rate-Limited Ruckus on Google AI Studio**: Users discussed rate limits for **Gemini 2.5 Pro** on Google AI Studio, noting that while the interface lists a **150 RPM** limit, the actual limit for free tier users appears to be lower.
   - One user found errors and cooldown after sending many requests in a short period, suggesting the limit is more of a *fair use limit* to prevent reverse engineering or automation.
- **Midjourney + Spellbrush drop groundbreaking i2v model**: Midjourney and Spellbrush's new video model is insanely good.
   - One user stated *it is basically the chatgpt moment of i2v* and hoped for more infrastructure to roll out **720p**.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1387135866383761519)** (9 messages🔥): 

> `C++ CUDA build systems, Meson, Buck2, xmake, Zig` 


- **Advent of C++/CUDA Build System Recommendations**: Users discussed build systems for a **C++ and CUDA project**, seeking alternatives to CMake.
   - Options suggested include **Make**, **Meson**, **Buck2**, **xmake**, and **Zig**, reflecting the variety of choices available beyond CMake for managing such projects.
- **XMake Looks Promising**: A user suggested and linked to [xmake](https://xmake.io/#/), highlighting its ease of use with CUDA projects.
   - They shared a sample configuration, showing how to **define a target**, **specify CUDA files**, and **add CUDA architectures**.


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1386818320719806647)** (2 messages): 

> `Triton AOT Compilation, Triton Community Meetings, Fused Attention Kernel` 


- **Triton AOT Compilation Type Hinting**: A user is trying to leverage **Triton** with **AOT compilation** and is facing issues with type hinting the `q` tensor in the `_attn_fwd_inner` function of the [fused attention kernel tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html).
   - The user is seeking guidance on how to properly type hint tensors, noting that the `str_to_ty` function in [compile.py](https://github.com/triton-lang/triton/blob/main/python/triton/tools/compile.py#L23) seems to only support `pointer`, `tensordesc`, and `constexpr`.
- **Triton Community Meetings MIA?**: A user inquired about the status of **Triton community meetings**, noting that the last recording on YouTube is from November 2024.
   - The user is particularly interested in any design specs or discussions related to major changes, including the layout system.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1386843717763338251)** (11 messages🔥): 

> `CUB with NVRTC, matmul overlap, JIT safe standard library headers, torch.cdist implementation` 


- **NVRTC struggles with CUB Integration**: A member is trying to get **CUB** working with **NVRTC** for blazing fast compilation times in `torch.cuda._compile_kernel` but is running into issues with missing C++ standard library headers.
   - Another member suggested to use `#include <cub/block/block_reduce.cuh>` instead of `#include <cub/cub.cuh>`.
- **Maximize matmul overlap with TC and non-TC operations**: A member seeks advice on maximizing overlap between **TC** and **non-TC** operations in a **matmul** with a math-heavy epilogue, especially on non-datacenter GPUs.
   - They find it difficult to get `nvcc` to generate **SASS code** that evenly interleaves `*MMA` instructions with other ALU instructions, suggesting the compiler might prioritize `.reuse` on `*MMA` input registers.
- **JIT-safe standard library headers Available**: A member suggested to check out [NVIDIA's jitify](https://github.com/NVIDIA/jitify) for support for **JIT-safe standard library headers** (e.g., float.h, stdint.h etc.).
- **Torch cdist Implementation JIT Compiles Against Cutlass**: One member shared their repo, [Kernel-Machines/kermac](https://github.com/Kernel-Machines/kermac), which **JIT compiles against cutlass/cute** to implement `torch.cdist` and beats torch by a mile.
   - They use a **lmdb database** that caches the jit kernels and loads functions from modules if the modules are already loaded to avoid wheel nonsense and cuda extension mess.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1386799820873793686)** (1 messages): 

> `TorchTitan, SimpleFSDP, TP and FSDP collectives, Inductor` 


- **SimpleFSDP Implementation Joins TorchTitan**: The **SimpleFSDP** implementation in *TorchTitan* is the best way to capture a graph containing all collectives, as described in the [README](https://github.com/pytorch/torchtitan/blob/main/torchtitan/experiments/simple_fsdp/README.md).
- **TP Added to SimpleFSDP Version**: **Tensor Parallelism (TP)** was recently added to the SimpleFSDP version, enabling the compilation of a graph with both **TP** and **FSDP collectives**, per [this pull request](https://github.com/pytorch/torchtitan/pull/1250).
- **Inductor Secrets Revealed**: For using the compile stack (including **Inductor** for kernel fusion) with custom logic for compute/comms overlap, there are private hooks in Inductor for registering graph passes, explained at the [config.py file](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py#L262).


  

---


### **GPU MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1387054679380459530)** (4 messages): 

> `LLM, CUDA, algorithms` 


- **Compute architecture beats linear algorithm**: A member argues that beginners in LLM should prioritize understanding **compute architecture** and **cache hierarchy** over just linear algorithms.
   - They added a *linear algorithm that requires a bunch of fetches from memory will regularly underperform a quadratic algorithm that can stay in SMEM*.
- **CUDA implements efficient algorithms**: A member mentioned that implementing algorithms efficiently in **CUDA** is about more than just learning theory.
   - They pointed out that for parallel algorithms other than the most basic ones classic complexity theory is under-complex and mentioned [PMPP](https://github.com/rmcantin/PMPP) as an example.


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1386886630761037885)** (1 messages): 

> `PyTorch Tool, Machine Learning Efficiency, Optimization, Mentorship Opportunity, Medical Device CV` 


- ****Medical CV Engineer** seeks **PyTorch ML Efficiency** Mentor**: A computer vision engineer in the medical device field is seeking a mentor to advise on a **PyTorch tool** focused on **machine learning efficiency and optimization**.
   - The engineer is offering **co-authorship** as compensation for the mentor's time and can be contacted via DM or at [s.askaruly@gmail.com](mailto:s.askaruly@gmail.com).
- ****Tool Seeks Guidance****: An engineer is developing a **PyTorch tool** for **machine learning efficiency**, seeking guidance on specific problems it could address.
   - They lack industrial experience in the field and hope a mentor can provide feedback and advice.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1386807256745508914)** (6 messages): 

> `cuML, NVIDIA driver, CUDA toolkit, threadIdx.y vs threadIdx.x` 


- **cuML Compatibility with NVIDIA Drivers and CUDA Toolkit Troubles**: A user inquired about using **CuML** with a recent **NVIDIA driver**, facing compatibility issues with their installed **CUDA toolkit**.
   - It was clarified that newer drivers support newer toolkits, suggesting the environment was locked to an older, incompatible toolkit, which the user resolved by uninstalling the old toolkit and installing the correct version.
- **Thread Index Confusion in Matrix Multiplication**: A user questioned why **threadIdx.y** is used for rows and **threadIdx.x** for columns in basic matrix multiplication, expecting the opposite.
   - Another user explained that *`threadIdx.x` is the dimension along which warps are laid out*, making it suitable for the column index in a row-major layout and the row index in a column-major layout to coalesce global memory accesses.


  

---


### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1386887446117224589)** (1 messages): 

> `Reduction code correctness, Input length handling` 


- **Reduction Code Bug Discovered!**: A user identified a potential issue in the **reduction code** from Chapter 10 of *Programming Massively Parallel Processors (4th edition)* related to handling input lengths that are not a power of 2, and asks for confirmation.
   - They provided a [link to their code](https://github.com/katsudon16/programming_massively_parallel_processors/blob/98616b84fd03b5110bfa5d4d9470568caf34eb08/chapter_10/sum_reduction_less_control_divergence.cu#L15) as an example of the extra handling they implemented.
- **Input Length Woes!**: The user's code snippet demonstrates the additional steps required to ensure correct functionality when the input length is not a power of 2.
   - This raises a question about the robustness of the original implementation in the textbook for real-world scenarios.


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1386787747314008174)** (4 messages): 

> `rocprofiler-sdk Integration, Chisel Performance Counters` 


- **Chisel Implements rocprofiler-sdk Integration**: A member announced the implementation of **rocprofiler-sdk integration** in **Chisel**, based on a previously described setup, that automatically builds *aqlprofile* and *rocprofiler-sdk* from mainline.
   - The integration downloads the **rocprof-trace-decoder** binary and sets up environment variables, with a new **--pmc flag** to collect custom performance counters (e.g., `chisel profile amd kernel.cpp --pmc GRBM_GUI_ACTIVE,SQ_WAVES`).
- **Chisel Enables Custom Performance Counters Collection**: The new **--pmc flag** in **Chisel** allows users to collect custom performance counters such as **GRBM_GUI_ACTIVE** and **SQ_WAVES**.
   - This feature aims to provide more granular performance insights and is available through the *rocprofiler-sdk* integration.


  

---


### **GPU MODE ▷ #[intel](https://discord.com/channels/1189498204333543425/1233802893786746880/1387160283423969301)** (1 messages): 

> `Intel GPU atomic latency, Ponte Vecchio VTUNE, SYCL device cycle counters` 


- **Seeking Intel GPU Atomic Latency Insights**: A member is inquiring about methods to calculate per thread atomic latency on **Intel GPUs**, specifically using **Ponte Vecchio**.
- **VTUNE or SYCL Device Cycle Counters**: They are considering the use of **VTUNE** or **SYCL** device cycle counters.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1386785129510342819)** (8 messages🔥): 

> `GPU Rental, Chisel Tooling, CUDA Competition, 3D Gaussian Splatting` 


- ****Chisel** Adds L40s as T4 Alternative**: To compensate for lack of **T4 GPU** support on DigitalOcean, the [Chisel CLI](https://www.chisel.so/) now supports **Nvidia L40s GPUs** available at ~**$1.57/hr**.
   - Users can run `pip install chisel-cli`, `chisel configure`, and `chisel run nvidia --gpu-type l40s <kernel.cu>` to download **nsight-compute** and **nsight-systems** profiling outputs.
- **CUDA Competition Announced for Faster 3D Gaussian Splatting**: A member announced a **CUDA**-oriented competition with a **$1100 prize** for reducing the training time of [3D Gaussian Splatting](https://github.com/MrNeRF/gaussian-splatting-cuda) by **50% or more**.
   - Submissions must be contributed under the **GPLv3 license**, and the deadline is **July 31, 2025**, measured on an **RTX 4090**.
- **Accelerating AI Products with CUDA**: A member shared a [LinkedIn post](https://www.linkedin.com/posts/activity-7343124619568009217-tl4Y) and accompanying Medium article on *Accelerating AI Products with CUDA*.


  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1386799896157225040)** (30 messages🔥): 

> `KernelLLM, Triton Data, Kernelbot Data, Synthetic Datasets, PyTorch to Triton Conversion` 


- ****KernelLLM** Craves Specific Code Format!**: Members discussed proper formatting of prompts for **KernelLLM** to ensure optimal performance, noting it expects a `Model(nn.Module)` and `get_inputs` functions; a walkthrough is available [here](https://huggingface.co/facebook/KernelLLM/discussions/5#685b0903b3d048882566b17b).
   - It was highlighted that **KernelLLM** isn't very flexible with the code it consumes and has specific expectations for the input format.
- ****Triton Data** Remains Elusive!**: There is very little human-generated **Triton data** available on the internet, prompting the creation of resources like **Kernelbot**.
   - A member shared the [Kernelbot data](https://huggingface.co/datasets/GPUMODE/kernelbot-data) dataset, which is all human written, but only applies to a few problems.
- ****Synthetic Data** Boosts Models!**: One is creating **synthetic datasets** to bootstrap models by generating PyTorch code, converting it to Triton, and looping until outputs match.
   - Another member suggested using **Gemini** to generate programs from a list of ops, or annotating traces by RL, noting that **RL** requires some priming to be effective.
- ****KernelLLM** in vLLM Deployment Woes!**: A member reported issues running **KernelLLM** in **vLLM**, noting it rarely produces proper Triton kernels, possibly due to incorrect method usage by **vLLM**.
   - There are expected restrictions on input format like `Model(nn.Module).forward()`, but these should be better documented.


  

---


### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/)** (1 messages): 

dragan.jovanovich: congrats👏
  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1343002580531417211/1386913315103440917)** (2 messages): 

> `CUDA Matmul precision issues, Triangle Multiplicative Update (Trimul) in AlphaFold` 


- **CUDA Matmul Faces Precision Predicament**: A member reported failing test cases in a custom CUDA matmul implementation due to [precision mismatches](https://github.com/yechenzhi/reference-kernels/blob/main/problems/pmpp/matmul_py/submission.py).
   - The errors indicated mismatched elements with small differences, suggesting a floating-point precision issue.
- **Trimul Time: AlphaFold's Triangle Multiplicative Update Challenge Launched**: A new problem, the **Triangle Multiplicative Update** (Trimul) from the AlphaFold family, is now available for both NVIDIA and AMD GPUs, with [a detailed writeup](https://tinyurl.com/gpumode-trimul).
   - The challenge focuses on optimizing the Trimul operation, a core component in **AlphaFold's** structure prediction, across diverse hardware architectures.


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1386925561607225384)** (7 messages): 

> `prefixsum performance, sort performance, trimul performance on B200, trimul performance on A100` 


- **H100 PrefixSum: seventh!**: A member achieved **7th place** on the `prefixsum` leaderboard on **H100** with **1037 µs**.
   - This was the best submission out of three, the next two submissions registered timings of **3.20 ms** and **2.87 ms** respectively.
- **Sorting takes fifth on H100!**: A member achieved **5th place** on the `sort` leaderboard on **H100** with **7.16 ms**.
- **trimul: Cracking B200s!**: Two members achieved **first** and **second place** on the `trimul` leaderboard on **B200** with **7.92 ms** and **8.20 ms** respectively.
- **trimul: Aceing A100s!**: A member achieved **first place** on the `trimul` leaderboard on **A100** with **20.0 ms**.


  

---


### **GPU MODE ▷ #[status](https://discord.com/channels/1189498204333543425/1343350424253632695/1387187660900012104)** (1 messages): 

> `New leaderboard problem, AMD + NVIDIA hardware` 


- **New Leaderboard Problem Drops!**: A new leaderboard problem is now available for both **AMD** and **NVIDIA** hardware.
   - More details can be found in the writeup at [https://tinyurl.com/gpumode-trimul](https://tinyurl.com/gpumode-trimul).
- **Dive into the 'trimul' Challenge**: A fresh challenge, named 'trimul', has been released on the GPU MODE leaderboard, accommodating both **AMD** and **NVIDIA** architectures.
   - Interested participants can access the problem's specifics and guidelines via the provided link: [https://tinyurl.com/gpumode-trimul](https://tinyurl.com/gpumode-trimul).


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1386836174987071699)** (5 messages): 

> `Factorio Client Authentication, FLE updates, Error cases in FLE` 


- **Factorio Client Authentication Still Needed**: Users discussed whether the **Factorio Learning Environment (FLE)** can run without the **Factorio client authentication** step.
   - A member confirmed that it is currently required, but a **PR** is in progress to remove the need for client login, and is expected to be merged this week.
- **Multiple Error Cases Plague FLE**: One user is experiencing multiple different error cases within **FLE**.
   - He has created a **PR** to address one of the more obvious issues and suggests merging PRs **220 to 238** into the main branch to ensure a working state before further work is done.


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1387199760993095853)** (1 messages): 

> `CuTe DSL, GEMM kernel, TMA transfers, MMA operations, sm90 architecture` 


- **CuTe DSL Speeds Up Kernel Compilation**: A user attempted to implement a persistent ping pong **GEMM kernel** for **sm90** using the **CuTe DSL** with a producer warpgroup initiating **TMA transfers** and two consumer warpgroups initiating **MMAs**.
   - The user reported that the near instantaneous compile time, ease of printing and setting breakpoints, and pythonic-ness of the DSL made it a much nicer experience compared to doing the same in **C++**.
- **Barrier Synchronization Blues in Persistent Ping Pong GEMM Kernel**: The user ran into barrier synchronization issues while implementing the persistent ping pong **GEMM kernel**.
   - Details of the issue can be found on [Cutlass's GitHub issue #2418](https://github.com/NVIDIA/cutlass/issues/2418).


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1386783101199454422)** (90 messages🔥🔥): 

> `NVMe, Network cards, MI300x with AMDGPU, ResNet and BERT Training, GPU kernel` 


- **NVMe Abstraction Discussed**: Members talked about **NVMe** being simple and standard, but noted that it loses abstractions like filesystems.
   - The idea of *'DISK:/dev/nvme0'* was raised as a potential addressing scheme, with questions about the ease of unbinding the kernel driver.
- **Infiniband Transfers Break Graphs**: Infiniband transfers currently implemented requires to break the graph, which is bad for performance.
   - Discussion covered whether the transfers in the graph should be the same as a copy, and the challenges of remote DMA with **RDMA**.
- **GPU Kernels For Network Cards?**: Discussion considered writing a network card driver and running that driver on the GPU for better control over transfers.
   - The suggestion was made to allow arbitrary **CPU kernels** and use a CPU kernel in the graph to set up the transfer as a callback.
- **TinyBox Questions**: A potential customer asked about purchasing the **TinyBox**, including questions about lead time and maintenance/support information.
- **Bounty Misinterpretation Clarified**: A member misunderstood the requirements of a bounty, thinking it involved a web server and voice-to-text processing without JavaScript.
   - The correct understanding of the bounty is that it involves running **Whisper** with basic JavaScript for passing audio data to **WGSL**, without external library preprocessing.


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1387114562452455577)** (1 messages): 

> `FP8 Conversion, Hardware Compatibility` 


- **FP8 Conversion Function Lands**: A member implemented a function to convert **fp8e4m3** and **fp8e5m2** tensors to **float32** manually.
   - This function is for users whose hardware does not natively support **fp8** types, and is available [here](https://github.com/softcookiepp/tinybloat/blob/master/src/tinybloat/compatibility.py).
- **FP8 Hardware Compatibility**: The new function addresses hardware compatibility issues with **FP8** tensor types.
   - It allows users with older or less capable hardware to still work with models that utilize **FP8** by converting them to **float32**.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1386785315561144330)** (50 messages🔥): 

> `Synthetic Data for Training, Meta's Synthetic Data Kit, Gemini Pro Stable's Instruction Following Issues, Aider Benchmark Framework, Claude Max Integration with Aider` 


- ****Synthetic Data Tool** Faces Diff-iculties**: A member is developing a **synthetic data generation tool** for training purposes but encountered issues with incorrect diffs when using it as an editor.
   - They are exploring options for proper distillation, including using logits or QLoRA on model responses, drawing inspiration from [Exercism](https://exercism.org/) problems for challenging benchmarks.
- ****Gemini Pro Stable** Struggles with Instructions**: Multiple members have observed that **Gemini Pro Stable** exhibits poor instruction following, with one user noting they have to keep it on a *short leash*.
   - One user shared that when asked to mark section **1.1** as completed, it completed all tasks, created new files, but failed to apply the changes, *butchering* the repo.
- ****Aider Benchmark** Lacks Interface Information**: A member questioned how LLMs can pass unit tests in the **Aider benchmark** without explicit interface information, referencing the [polyglot-benchmark](https://github.com/Aider-AI/polyglot-benchmark/blob/main/cpp/exercises/practice/all-your-base/all_your_base.cpp) on GitHub.
   - Another member clarified that instructions are included, and test results from the first pass are added to the context in the second pass, providing a way for the model to learn the interface through error messages.
- ****Claude Max** May Integrate with Aider**: A member inquired about connecting **Claude Max** to Aider, similar to Roo Code, to which the response was uncertain due to possible terms of service limitations.
   - A user suggested using the [claude-code-api](https://github.com/codingworkflow/claude-code-api) and chimera models for interesting distillation techniques.


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1386944419600531609)** (26 messages🔥): 

> `Aider strange interactions, deepseek-r1 token limits, MCP support in aider, Gemini's intelligence` 


- ****Aider Acts Oddly: Intermittent File Writing****: A user experienced **aider** displaying diffs without writing to files in a new project, despite successful commits in a previous trial, possibly due to exceeding **deepseek-r1**'s [token limit](https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo).
   - The user noted that *the token warning was issued right after giving it a command* and that a simpler task with less files worked as expected.
- ****Deepseek-r1 Size Matters: Context Window Confusion****: A user questions the context window size, pointing out that [openrouter.ai/deepseek/deepseek-r1-0528](https://openrouter.ai/deepseek/deepseek-r1-0528) has a **128k context window**, and suggested the user might be misreading the leaderboard.
   - They clarified that **0528 should be better** than the one the user was experiencing issues with.
- ****MCP Integration in Aider Missing****: A user inquired about using **aider** via a script to use common CLI commands such as `patch` and `awk` with **MCP (Mod Compliance Package)**.
   - The response indicated that there is *no official support for MCP in aider*, but advising to *instruct aider to run cli commands, combined with the --yes-always* to run the CLI commands automatically.
- ****Gemini's Genius: Waning or Waxing?****: A user questioned whether **Gemini** has become less intelligent, sparking a debate about model performance.
   - A member linked to a [blog post](https://aider.chat/2024/08/26/sonnet-seems-fine.html), and another countered that businesses *monkey with internal models under the hood* which are rarely on the technical implementation details.


  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1386789432073388072)** (10 messages🔥): 

> `Claude Code, Backend, Subscription, API calls, SDK` 


- ****Claude Code API** proposal**: A member proposed using [Claude Code](https://github.com/codingworkflow/claude-code-api) as a backend for Aider to leverage its subscription benefits.
   - The value proposition highlighted was the **cheaper call costs** compared to direct API calls, potentially saving money for users with high usage, with one user reporting **$1200+ equivalent API use** in 30 days using Claude Code Pro.
- **Terms of Service Question Raised for Claude Code**: A user asked about the **terms of service** for Claude Code, suggesting it might be acceptable to integrate it behind another tool.
   - It was also speculated how `/run claude -p "XXXX"` would behave, considering whether it implies including Claude Code in the context or executing code editing with it as the provider.
- **Discussing Claude Code as a Provider**: Another member highlighted that the [Anthropic documentation](https://docs.anthropic.com/en/docs/claude-code/sdk) suggests there's **no issue using Claude Code as a provider** if utilizing the SDK.
   - There appears to be excitement surrounding the seamless integration of Claude Code as a service.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1386865643344891965)** (37 messages🔥): 

> `Efficient Pneumonia Detection with Vision Transformers, Scaling Vector Search with FAISS, GRPO for RL, FYP ML domain` 


- **Transformers Ace Pneumonia Detection**: A paper presents [efficient pneumonia detection](https://www.nature.com/articles/s41598-024-52703-2) using **Vision Transformers** on chest X-rays.
   - One member found it *shocking* that results like this could still be published in **2024**, calling it *decade old news*.
- **Vector Search Scaling Strategy Showdown**: A member is scaling vector search by using **torch.matmul** for up to **1M** comparisons and switching to a quantized **FAISS index** like `IndexIVFPQ` for **10M+**.
   - Another member cautioned about using quantized indexes due to potential **training instability** from precision tradeoffs.
- **GPU Sharding vs. OOM for matmul**: A member asked whether calculations were being done on **one GPU** or sharded over **many parallel GPUs**.
   - The OP calculated that **4MB** of memory would be used for **1k*1k dot products**, but was wary of PQ's **compression** because it can really create some problems since similarity search is really sensitive with embeddings and when we are compressing it , i really dont know how its going to change meaning and i really dont have much info about PQ anyways.
- **GRPO Gets Cracking in RL**: **GRPO** is the way to get cracked in RL. The image attached contains a **TLDR** that explains that Dr. GRPO makes it less of a **yapper** while maintaining performance.
   - One member shared a [tweet](https://fxtwitter.com/jmhessel/status/1899909893324468444) distinguishing between **RL in general** and **LLM RL**.
- **FYP Project: GritLM, Gaussian Splatting, NER**: A member is trying to find a **less-explored ML domain** that has a practical use case and meaningful research potential and had multiple options for a FYP project: **3D Point Cloud, 3D Visualization of Human Body Organs, Image Colorization, Medical Visualizations, 3D Vision Using Two Cameras (Stereo Vision) and Named Entity Recognition (NER)**.
   - One member suggested: **NER** looks like the territory of [GritLM](https://github.com/ContextualAI/gritlm), if **Image Colorization** is Gaussian Splatting gonna be involve, then not less explore ML-domain, and digital twins is one application. **Medical Visualization** is feasible with train data, not worth it without. **Stereo Vision** is less noisy indoor, bad outdoor.


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1386879812148396133)** (12 messages🔥): 

> `Cloud GPU Platforms, AI in Education, RWKV v6 and Finch Series, Time Crystal Computer` 


- **Vast.ai Gets a Shoutout**: A member mentioned using [vast.ai](https://vast.ai) for research, implying it is a viable **cloud GPU platform** option.
   - No details were given regarding performance and cost, only that it is *used*.
- **AI Threatens to Eat Schools?**: A member shared a [Time Magazine article](https://time.com/7295195/ai-chatgpt-google-learning-school/) discussing **AI's impact on education** and learning.
   - The conversation did not delve into the article, only that it was shared.
- **RWKV v6 and Finch Soar**: A member reported on the release of **RWKV v6 (Finch series)**, a **1.5B** model achieving **SOTA** in multilingual and English tasks, along with multimodal capabilities, citing the [RWKV-5 & 6 paper](https://arxiv.org/abs/2404.05892).
   - According to [this X post](https://x.com/BlinkDL_AI/status/1755656095970857269?s=20), **Finch** incorporates a selectivity mechanism akin to **Mamba**, outperforming transformers in perplexity.
- **Time Crystal Brain?**: A member shared a highly unusual paper ([A Brain-like Computer Made of Time Crystal](https://www.researchgate.net/profile/Anirban-Bandyopadhyay-7/publication/337323300_A_Brain-like_Computer_Made_of_Time_Crystal_Could_a_Metric_of_Prime_Alone_Replace_a_User_and_Alleviate_Programming_Forever/links/5dd22692299bf1b74b4b38a3/A-Brain-like-Computer-Made-of-Time-Crystal-Could-a-Metric-of-Prime-Alone-Replace-a-User-and-Alleviate-Programming-Forever.pdf#page=11)) calling it a *record-setting bat-$hit paper*.
   - No further information was given regarding its contents.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1386789229199097906)** (26 messages🔥): 

> `Natural Selection and AI, Genetic Engineering vs Automation, AI as Calculator, Richer People Reproduce Less, Papers on RL & LLMs` 


- **AI Brainrot Fuels Natural Selection?**: Members debated whether **AI brainrot** jeopardizes survival and reproduction, suggesting that **low-skilled labor automation** intensifies natural selection.
   - One member argued that despite low wages, manual labor remains abundant due to high robotics costs and that protecting against natural selection is necessary.
- **Gene Editing Edges out Evolution?**: Members discussed that **genetic engineering** may soon overshadow natural selection, though current abilities to impact traits like intelligence remain limited.
   - Some believe that **AI advancements** will rapidly accelerate genetic understanding, while others remain skeptical about overcoming the complexity of the human genome.
- **AI: The Calculator of our Age?**: Members compared **AI to calculators**, suggesting restrictions on AI use, like banning calculators for multiplication tables, emphasizing potential long-term cognitive impacts.
   - One member shared a [YouTube video](https://www.youtube.com/watch?v=z3awgfU4yno) to bolster this point.
- **New Papers Highlight RL Limitations**: Members referenced new [papers on arxiv](https://arxiv.org) discussing limitations and potential pitfalls when using **Reinforcement Learning** techniques on **Large Language Models**.
   - Papers mentioned include *Understanding R1-Zero-Like Training: A Critical Perspective*, *Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model*, *Reinforcement Learning Finetunes Small Subnetworks in Large Language Models*, and *Spurious Rewards: Rethinking Training Signals in RLVR*.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1386792161336492102)** (70 messages🔥🔥): 

> `Harvey AI Funding, Replit ARR, AI Agent Supervision, Startup vs Incumbent, Magenta RealTime` 


- **Harvey AI lands Huge $300M Series E**: [Harvey AI announced](https://xcancel.com/harvey__ai/status/1937155058476646591) a successful **$300M Series E funding round**, valuing the company at **$5B**, co-led by Kleiner Perkins and Coatue.
   - They also [signed a partnership with LexisNexis](https://www.lexisnexis.com/community/pressroom/b/news/posts/lexisnexis-and-harvey-announce-strategic-alliance-to-integrate-trusted-high-quality-ai-technology-and-legal-content-and-develop-advanced-workflows) to integrate AI tech and legal content.
- **Replit Rockets to $100M ARR**: [Replit announced](https://xcancel.com/Replit/status/1937212611520831718) they have surpassed **$100M in Annual Recurring Revenue (ARR)**, thanking their customers and supporters.
   - Some members are questioning whether the **$1.1B valuation** was actually warranted.
- **Need Human-like AI Supervision?**: Matan-Paul Shetrit outlines the critical distinction between observability and supervision for scaling AI agents in [this tweet](https://x.com/MatanPaul/status/1937200395115499592).
   - He argues that traditional monitoring falls short because AI agents actively engage with systems and customers, necessitating a new approach to oversight similar to how humans are managed.
- **Distribution Dominates, Innovation Lags?**: Alex Immerman's [tweet](https://xcancel.com/aleximm/status/1937251084810219721) highlights the core battle between startups and incumbents: can startups achieve distribution before incumbents innovate?
   - The discussion emphasizes the power of distribution, with one user noting **OpenAI's rapid user acquisition** in contrast to Google's.
- **Anthropic's Fair Use Ruling Sparks Debate**: Adam Eisgrau reported that [Anthropic won its Motion for Summary Judgment](https://xcancel.com/adameisgrau/status/1937480346976813454) on fair use grounds, according to Judge Alsup.
   - However, a trial will proceed to determine potential damages for using *'pirated'* internet material.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1386797895868616934)** (37 messages🔥): 

> `grok3mini, humanizing AI agents, building llms from scratch, llm inference app llamabarn, COCONUT gating layer` 


- **Grok3mini usage jumps in June**: The non-beta version of **grok3mini** is now available and usage has increased significantly from **2M/day** on June 19th to **100M/day**.
- **Llamabarn local inference app launched by Georgi**: Georgi launched a new local inference app called **Llamabarn**, which looks clean and has generated positive thoughts, according to [this X post](https://x.com/ggerganov/status/1937189250149257250).
- **ryunuck explains the COCONUT gating layer**: **COCONUT** uses a 'gating' layer that extracts info from hidden states to determine sampler parameters at every token, keeping the hidden state across tokens instead of restarting, according to [this X post](https://x.com/ryunuck/status/1937466079309144256).
- **Training run on Psyche dashboard**: The training run can be watched on the [Psyche dashboard](https://psyche.network/), with more information available on the [Psyche architecture](https://nousresearch.com/nous-psyche/).
- **Facebook's book piracy lawsuit not looking good**: According to [this X post](https://x.com/AdamEisgrau/status/1937480346976813454), things don't look good for **Facebook's** book piracy lawsuit, as it seems they did not win the most important part (the piracy) and the training is ruled transformative.


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1387023361959530517)** (2 messages): 

> `Model Recommendations, LORA Training, GGUF Conversion, Local LLMs on GTX 1080` 


- **Seeking Model Advice for GTX 1080**: A member is seeking model recommendations suitable for **LORA training**, **GGUF conversion**, and running on a **GTX 1080**.
   - The goal is to use the model for character acting and general technical questions, including simple "how to's", with character reinforcement through **LORA training**.
- **Newbie Asks for Local LLM Guidance**: A new user requests explanations and recommendations for getting started with **local LLMs** on a **GTX 1080** GPU.
   - They plan to **LORA train** a model for character acting and general technical questions.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1387138352435957902)** (4 messages): 

> `MultiNet v0.2, Manifold platform, R1-Zero-Like Training, RL Incentivize Reasoning, Spurious Rewards in RLVR` 


- **MultiNet v0.2 Drops on Manifold!**: MultiNet v0.2, an open-source platform for evaluating generalist AI systems, has been released on [Manifold](https://www.manifoldrg.com).
   - Find the [papers](https://arxiv.org/abs/2505.05540) and apply to [collaborate](https://www.manifoldrg.com/os-research-fellow-multinet/).
- **Delving into Understanding R1-Zero-Like Training**: A [paper](https://arxiv.org/abs/2503.20783) titled *Understanding R1-Zero-Like Training: A Critical Perspective* was discussed.
   - It explores the critical perspectives of **R1-Zero-Like Training**.
- **Does RL Incentivize Reasoning?**: A [paper](https://arxiv.org/abs/2504.13837) titled *Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model* was mentioned.
   - The discussion revolved around whether **Reinforcement Learning** truly incentivizes reasoning capacity in **LLMs** beyond the base model.
- **Small Subnetworks Finetuned with RL**: A [paper](https://arxiv.org/abs/2505.11711) titled *Reinforcement Learning Finetunes Small Subnetworks in Large Language Models* was discussed.
   - It covers **Reinforcement Learning's** role in finetuning small subnetworks within large language models.
- **Spurious Rewards in RLVR Explored**: A [paper](https://arxiv.org/abs/2506.10947) titled *Spurious Rewards: Rethinking Training Signals in RLVR* was introduced.
   - The discussion centered on rethinking **training signals** in **RLVR** due to spurious rewards.


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1386827903450480842)** (2 messages): 

> `Reward Models, PAIE Curator` 


- **Internal Bias of Reward Models: Cursed Helm Alert!**: A new paper, [Cursed Helm](https://arxiv.org/abs/2506.07326), suggests that blindly integrating reward models into pipelines without considering their internal biases could be concerning.
   - It cautions those *just strapping reward models into their pipeline without considering their internal bias*.
- **PAIE Curator: LLM Escalation Listener**: The **PAIE Curator** is introduced as a local **LLM escalation listener** designed to catch model failures and provide structured feedback loops.
   - It operates without a front-end or vector search, focusing on listening when a model expresses uncertainty (e.g., *“I don’t know.”*), with its [GitHub repository available here](https://github.com/ProjectPAIE/paie-curator).


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1387138352435957902)** (4 messages): 

> `MultiNet v0.2, Manifold platform, Generalist AI evaluation, R1-Zero-Like Training, RL Incentivizes Reasoning` 


- **MultiNet v0.2 Drops at Manifold!**: Version **0.2** of **MultiNet**, an open-source platform for evaluating generalist AI systems, has been released at [Manifold](https://multinet.ai).
   - Find a [Discord link](https://www.manifoldrg.com) and more program info on the Manifold website, which also includes links to [X](https://x.com/HarshSikka/status/1937525251401011377) and [LinkedIn](https://www.linkedin.com/posts/harsh-sikka_multinet-a-generalist-benchmark-for-multimodal-activity-7343285918671196162-GVva).
- **New Papers Discussed on R1-Zero-Like Training!**: The following papers were mentioned in discussion: '[Understanding R1-Zero-Like Training: A Critical Perspective](https://arxiv.org/abs/2503.20783)', '[Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model](https://arxiv.org/abs/2504.13837)', '[Reinforcement Learning Finetunes Small Subnetworks in Large Language Models](https://arxiv.org/abs/2505.11711)', and '[Spurious Rewards: Rethinking Training Signals in RLVR](https://arxiv.org/abs/2506.10947)'.
   - Also, a [YouTube video](https://www.youtube.com/watch?v=z3awgfU4yno) was posted.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1386786703095758959)** (14 messages🔥): 

> `Multiagent Cooperation, Prefix caching, red teaming conversational AI` 


- **New Paper Elicits Feedback on Multiagent Cooperation**: A member received feedback that their *baby's first paper* lacked engagement with the literature on **multiagent cooperation**, despite identifying interesting conversational dynamics such as questions as circuit breakers.
   - The author acknowledged the limitation due to the expense of **Claude Opus** and intends to expand the study with larger sample sizes and variations in group composition, model parameters, and context window sizes in future versions.
- **Prefix Caching Strategies Debated for Large Sequence Lengths**: A member inquired about a library supporting **prefix caching** like **vLLM** but capable of storing the cache in a memory-mapped file for sequences too large for VRAM or DRAM.
   - Another member responded that this approach would likely be slower than recomputing the KVs unless the sequence length exceeds **1M**, though the original poster clarified his sequence length is **128k**.
- **Teenager Joins Server to Discuss Conversational AI Red Teaming**: A **17-year-old** from Sweden studying trucking introduced themself to the server, focusing on **red teaming conversational AI** using social and psychological pressure tactics.
   - They document their work in [a GitHub repository](https://github.com/Ufosxm34gt/Conversational-Red-Teaming-Casebook), seeking to connect and learn from others.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1386814732891193445)** (30 messages🔥): 

> `Spectral Normalization, Sleeping-DISCO Dataset, Generative Models and Dynamical Systems, Manifold Multimodal AI Benchmarks, RL incentive` 


- **Spectral Norm Approximations Pushed to the Limit**: A member described **spectral normalization** as estimating/approximating the spectral norm then dividing the *weight* by that norm for numerical stability.
   - They noted the downside is *if there is an outlier singular value, then the rest of the singular values gets pushed close to 0 which may not be ideal for some use cases.*
- **Sleeping-DISCO Dataset Seeks EleutherAI Collab**: A member inquired about a potential collaboration with EleutherAI for their new large-scale pre-training dataset for Generative Music modeling, **Sleeping-DISCO-9M** [available on Hugging Face](https://huggingface.co/datasets/sleeping-ai/Sleeping-DISCO-9M).
   - They are seeking assistance in benchmarking its quality and mentioned their [arxiv preprint](https://arxiv.org/abs/paper) requires grammatical fixes.
- **Originality Debated for Sleeping-DISCO Dataset**: A member criticized the Sleeping-DISCO dataset's originality, arguing that it primarily reindexes content from **Genius.com** without significant original contribution.
   - The dataset creators clarified that it provides lyrics and Genius annotations for academic use, similar to other datasets like **GeniusExpertise** or **The Million Song Dataset**, and links to **YouTube videos** for data download, while acknowledging limitations due to copyright.
- **Mean Flows Code Goes Live!**: The authors of [Mean Flows for One-Step Generative Modeling](https://arxiv.org/abs/2505.13447) announced that their code is now available [here](https://x.com/ZhengyangGeng/status/1937268681819693125).
   - This research shares their findings with people who are interested in generative models and dynamical systems.
- **Manifold Releases Multimodal AI Benchmarks**: The Manifold team has released open infrastructure, [multinet.ai](https://multinet.ai), to benchmark and improve generalist multimodal AI systems and two related [papers](https://arxiv.org/abs/2505.05540).
   - They are inviting feedback and collaboration, also inviting applications for their [research fellowship call](https://www.manifoldrg.com/os-research-fellow-multinet/).


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1386826778814976130)** (4 messages): 

> `NNsight pre-release, Loss curve decomposition, NDIF update, Orthogonal Gradient Basis` 


- ****NNsight's** next version is dropping soon!**: The NDIF team is pre-releasing the next version of **NNsight**, a framework for working with and intervening on PyTorch models.
   - A [Colab notebook](https://colab.research.google.com/drive/1wjQhbQKh2pwy-mxx4EFMBC1IEauzu9G0#scrollTo=ZuSXB8Bh1zEq) details changes and links to relevant places.
- **Decomposing Loss Curves with Orthogonal Gradient Basis**: A new paper decomposes loss curves across an **orthogonal gradient basis**, revealing that clusters of examples have similar breakthrough dynamics that are invisible in the exact loss.
   - These clusters and breakthroughs align with specific skills in both toy arithmetic and real language modeling settings, detailed in a paper [here](https://www.alphaxiv.org/abs/2505.14685).


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1387110878469558475)** (4 messages): 

> `Mojo GPU kernels, Mojo from Python Limitations` 


- **Enthusiastic Newcomer Greets Mojo Community**: A new community member expressed excitement about Mojo and its goals, planning to work on **GPU kernels** and calling **Mojo from Python**.
- **Mojo-Python Interop Still Limited**: A member inquired about the persistence of limitations when calling **Mojo from Python**, referencing the [known limitations](https://docs.modular.com/mojo/manual/python/mojo-from-python/#known-limitations) in the documentation.
   - They confirmed that these limitations are indeed present in the latest release.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1386830562269855975)** (39 messages🔥): 

> `Larecs Testing in Modular Community CI, Mojo as Rust Replacement, Mojo Async vs Rust Async, Statement Beginning Error in Mojo` 


- **Larecs Tests Fail in Modular CI**: A contributor is debugging an issue where [Larecs tests](https://github.com/samufi/larecs) only fail in the modular-community CI, but not on local machines or GitHub CI, making it difficult to track down.
   - Another contributor reproduced the issue on an M1 Mac, noting that it only fails with `mojo test` and suspects an unsafe operation is happening, and is assisting with debugging by providing detailed output from the failing test case using a [debug branch](https://github.com/samufi/larecs/tree/debug_query).
- **Mojo's Roadmap to Rust-like Safety**: A user inquired about Mojo's safety features as a potential Rust replacement, particularly regarding **sum types**, **pattern matching**, and **predictable destruction times**.
   - A Modular engineer responded that while *product types* exist as `struct`, *sum types* and *pattern matching* are planned, also explaining that Mojo already offers *RAII* and *ASAP destruction* and is moving away from *syntactic salt*.
- **Async Design Aims to Sidestep Rust's Troubles**: A user asked if Mojo's async design would address the difficulties experienced with async in Rust.
   - A Modular engineer pointed to [PR 3945](https://github.com/modular/modular/pull/3945) and [PR 3946](https://github.com/modular/modular/pull/3946) as solutions, noting that a better async runtime and linear types could eliminate the need for constructs like `Arc<Mutex<T>>`, and also pointed to [PR 4728](https://github.com/modular/modular/pull/4728) for improved IO.
- **Mojo's Mysterious Statement Placement Error**: A user encountered the error *"statements must start at the beginning of a line"* in Mojo with the following code snippet `if info.value() > top:`.
   - Another user suggested adding `var top` as a potential fix, indicating a possible issue with variable declaration or scope.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1386792338789236826)** (30 messages🔥): 

> `Manus PDF reading issues, New AI architecture development, Credit promo issues, Manus credits, Manus down` 


- **Manus Struggles with PDFs**: Users are reporting that **Manus** is having issues reading text documents and PDFs, with the chatbot requesting plain text input instead.
   - One user asked *why is manus having troubles lately reading text documents and pdfs?it always says: Could you please provide me with this text as plain text, as I cannot process PDF files directly?*.
- **AI Architecture Dreams Kick Off**: A member expressed interest in developing a new AI architecture for **funcognitive and meta improvements**, aiming for a better and faster transformer.
   - Another member asked *anyone here interested in developing a new ai architecture? for funcognitive and meta improvements really, just making a better and faster transformer*.
- **Subscription Promo Sparks Frustration**: A user was denied an extension to match a promo offering despite purchasing additional credits and was forced to create a new account.
   - The user stated that *Paid 400 USD for 2 months subscription. Asked Manus to extend to match the promo offering when I bought 19.900 additional credits this month. They refused.* and felt the situation was *so stupid*.
- **Users Rake in the Credits**: A user mentioned receiving **90k credits** for their contributions as a long-time **Manus beta tester**.
   - Another user commented *I got 90k credits* and *They just give me credits for my contributions*.
- **Manus Experiences Downtime**: Some users reported issues with **Manus** getting stuck and displaying internal server errors, resulting in wasted credits.
   - One user stated that they wasted over **2000 credits** due to the problems and another saying *I think Manus has become dumber and makes mistakes and doesn't find them*.


  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1386932593647423589)** (3 messages): 

> `TorchTune, Single Machine LORA, GitHub Issues` 


- **TorchTune Praised for Single Machine LORA**: A user spent half a day trying out **TorchTune**, particularly single machine **LORA**, and expressed that they were impressed with the package and found it super useful.
   - The user thanked the developers for their work.
- **TorchTune Team Encourages Feedback via GitHub**: The **TorchTune** team appreciated the positive feedback and encouraged the user to drop comments or open issues on [GitHub](https://github.com/pytorch/torchtune) for further feedback.
   - They also mentioned that they are usually pretty responsive.


  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1387043924337623141)** (25 messages🔥): 

> `Expandable Segments Bug, max-autotune issue, clearing cache, L40S card bug, reward modeling RFC` 


- **Expandable Segments triggers pointer error**: A member encountered a pointer error when using **expandable segments** on **Nvidia L40S** GPUs, and disabling the feature resolved the issue, but it worked on **H200s**.
   - The solution was found in [this issue](https://github.com/pytorch/pytorch/issues/140419) related to packing and flexattention, and `max-autotune` setting.
- **max-autotune crashes cards with not enough shared memory**: A member suggests that there might be an issue with **max-autotune** rather than hardware.
   - Currently the roundup_power2 flag isn’t available, but a member noticed that Unsloth uses **expandable segments** with this flag.
- **Clearing cache clears the card's errors**: A member reproduced the error, and found that clearing the cache would make the card work on the second try.
   - After clearing the cache, a failure occurs, then success, even without the new setting.
- **L40S card bug found**: Expandable segments might be an edge case and that L40S usage is not very prevalent, and NCCL recently disabled FP8 reduction under SM90.
   - A member suggests checking hardware specifications and skipping the expandable segments if necessary.
- **Reward Modeling RFC Feedback Requested**: A member requested feedback on the Reward Modeling RFC, and suggested discussing it on the next office hours (June 26).


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1386990196079333376)** (5 messages): 

> `NotebookLM Model, Latest Model Info, Model Options` 


- **NotebookLM Model Updates Awaited**: A user inquired about the current model used by **NotebookLM** and which is the *latest* version.
   - The user also asked where to find **model options** on the page, referencing [this YouTube video](https://youtu.be/K9bvF_CJKV8?si=Gj7Z6GfOaTRLHKx2) and [another video](https://youtu.be/DLEKeE9pbU8?si=FVHrx6QwJKBhWTRF).
- **Checking the FAQs**: Users should check the **Frequently Asked Questions** or **Release Notes** page for the latest model info.
   - There may also be a **dropdown** in the user interface.


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1386787180592234587)** (22 messages🔥): 

> `New user options, Share the link feature, NotebookLM Alternatives, Audio Overview Generation, Vimeo Videos as Sources` 


- **New user seeks options!**: A new user inquired about options for getting started, and a member offered assistance via direct message.
   - Another user welcomed them and suggested reaching out.
- **Share the Initial State... Not the Full Context**: A user reported that the "share the link" feature only shares the initial query state, *before* the prompt and response, hindering comprehensive context sharing.
   - They suggested a "copy button on everything" as a solution, advocating for the ability to share the uploaded source list, prompt, and model's response for debate purposes.
- **NotebookLM Having Issues?**: A user reported that **NotebookLM** was not working for them and requested alternative suggestions.
   - Another user responded saying that it works wonderfully for them, and listed hundreds of **pdfs**.
- **Can AI Automator turn NotebookLM audio into a podcast setting avatar session?**: A user inquired about using [SuperTelegram](https://supertelegram.gumroad.com/l/pwxot) to transform a **4-minute NotebookLM audio** into a duo host podcast avatar session.
   - Another user mentioned that splitting speakers might be necessary for this purpose.
- **Hit the Generation Limit, Prompt Lost?**: A user expressed frustration that **NotebookLM** announces hitting the generation limit *after* a long custom prompt is entered, and asked whether the prompt would be saved for later.
   - Another user asked if **Vimeo videos** can be used as sources, but encountered security feature issues when pasting the link, prompting another user to suggest downloading the video using [cobalt.tools](https://cobalt.tools/).


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1386902508823122007)** (12 messages🔥): 

> `Debian 12 vs Ubuntu Jammy, Python SDK update, GPT4All official website issues` 


- **Debian 12 build issues**: A user had trouble building with **Debian 12** and suggests using **Ubuntu Jammy** instead, recommending [Qt SDKs](https://qt.org).
   - The user couldn't remember how they eventually got it working, suggesting they might have tried **backport packages**.
- **Python SDK update queried**: A user asked if there was an update coming on the **Python SDK**.
   - They jokingly inquired, *"Or is python doomed?"*
- **GPT4All website CPU Usage**: A user reported that the [gpt4all.io](https://www.nomic.ai/gpt4all) website is buggy and *"takes 60% of my internal GPU"*.
   - They linked to the **nomic.ai** GPT4All page, suggesting it is the official website.


  

---


### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/1387068621959860348)** (1 messages): 

> `Atom of Thought, GAIA benchmark, Agent Startup, Implementation code issues` 


- **Atom of Thought faces flexibility loss**: An experiment with **Atom of Thought** in an agent (**GAIA benchmark**) led to its removal due to a loss of flexibility from upfront decomposition and context loss between steps.
   - The researcher lost faith in the paper and authors due to serious issues with the paper's implementation code.
- **Authors behave unprofessionally on X**: Upon notification of implementation code issues, the authors responded *extremely negatively and unprofessionally* on X.
   - They then leveraged the paper into their pivot into an agent startup on X.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1387044069447962759)** (5 messages): 

> `Ax for TypeScript, module status messages, OpenAI Issues, LiteLLM` 


- ****Ax**-ing about **Ax** Ports**: A member mentioned the existence of **Ax** for **TypeScript** and its ports to **Elixir** and **Ruby**.
- **Seeking Status Updates in Forward Methods**: A member inquired about emitting status messages from a module's `forward/aforward` method without yielding, aiming to capture an event after `module_start_status_message`.
   - Another member suggested passing a **callback** into `forward` to update the UI progress.
- **OpenAI App Outage**: A member reported having issues with **OpenAI**, stating that *their app is down*.
   - The error `404 Not Found` was thrown using **LiteLLM**'s `completion()` with `model= gpt-4o-mini; provider = openai`.


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1386838224885780662)** (6 messages): 

> `Google's A2A, Anthropic A2A, MCP Timeouts, Chrome AI APIs` 


- **Google Generously Gifts A2A to Linux Foundation**: Google donated **A2A** (presumably, some Google product) to the [Linux Foundation](https://developers.googleblog.com/en/google-cloud-donates-a2a-to-linux-foundation/).
- **Anthropic to Acquire A2A?**: A member suggested that **Anthropic** should follow Google's lead and donate A2A.
   - This was a direct response to Google's donation of A2A to the Linux Foundation.
- **MCP's Timed Out Troubles Triggered**: A member reported encountering a **timeout issue** with the **MCP tool** while using **OpenAI agents** to create a client session.
   - The error message indicates the system timed out while waiting for a response to a **ClientRequest** after **5.0 seconds**.
- **Chrome's New AI APIs Arrive**: Chrome is integrating some **AI APIs** as announced in [Chrome 138](https://developer.chrome.com/blog/new-in-chrome-138?hl=en#built-in).
   - This could potentially lead to **MCP integration** directly within the browser.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1386803784847724765)** (5 messages): 

> `Certificate Timing, Course Completion, Social Media Posts for Course` 


- **Certificate Distribution Date Leaked!**: Members who have completed all assignments and social media posts can expect to receive their certificates by **mid-July**.
   - The distribution timeline was confirmed by a staff member.
- **Course Completion Confirmed!**: Participants confirm they have completed all assignments and social media prerequisites, inquiring about certificate timing.
   - The course completion involves assignments and social media posts on platforms like **Twitter** and **LinkedIn**.


  

---


### **Cohere ▷ #[🧵-general-thread](https://discord.com/channels/954421988141711382/954421988783444043/1386844614941474917)** (3 messages): 

> `Cohere Reranker Pricing, Token Usage in Cohere API` 


- **Cohere Reranker Price Points Prompt Concerns**: A member inquired about reducing costs for the **Cohere reranker**, anticipating frequent usage of **1000 calls**.
   - Another member clarified the pricing structure, explaining that costs are determined by the number of documents and tokens, with documents over **500 tokens** being split into chunks.
- **Reranker pricing is static says one member**: One member asked for confirmation if **Rerank API** calls cost **$2 per 1000 calls** regardless of token usage.
   - Another member responded and shared the [Cohere Pricing page](https://cohere.com/pricing#:~:text=We%20count%20a%20single%20search%20unit%20as%20a%20query%20with%20up%20to%20100%20documents%20to%20be%20ranked.%20Documents%20longer%20than%20500%20tokens%20when%20including%20the%20length%20of%20the%20search%20query%20will%20be%20split%20up%20into%20multiple%20chunks%2C%20where%20each%20chunk%20counts%20as%20a%20singular%20document.) which counts a single search unit as a query with up to **100 documents**.


  

---


### **Cohere ▷ #[👋-introduce-yourself](https://discord.com/channels/954421988141711382/1346635816629178410/1387160626887135365)** (1 messages): 

> `Introductions, Community, Tech, Tools` 


- **Cohere Community Welcomes New Members**: The Cohere community welcomes new members to their Discord server, inviting them to introduce themselves.
   - New users are encouraged to share their **company/industry/university**, what they're working on, favorite **tech/tools**, and what they hope to gain from the community.
- **New Members Eager to Connect**: Many new members have joined the Cohere Discord server and are actively introducing themselves.
   - They express excitement about the community and look forward to engaging with other members and learning about new technologies.


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1387100447914856481)** (2 messages): 

> `Open Source Resume Matching, Claude-Compatible MCP Server` 


- **Cursor Screens Resumes with Open-Source Matching**: LlamaIndex introduces an open-source **Resume Matching MCP server** for intelligent job matching directly within the Cursor workflow, connecting to **LlamaCloud resume indexes** and [other services](https://t.co/RCKoiUccm6).
   - The project was built by @zhaoqili74 during an internal hack day, aiming to streamline resume screening processes.
- **Launch Claude-Compatible MCP Server Template**: LlamaIndex releases a new open-source template repo for building a **Claude-compatible MCP server** as a Next.js app with full **OAuth 2.1 support**, simplifying the creation of remote Model Context Protocol servers that work seamlessly with [this service](https://t.co/wtPorldMvJ).
   - Developed during an internal hack day by @seldo, the template aims to ease integration with **Claude** and other services using the Model Context Protocol.


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1387018749877555343)** (1 messages): 

> `FAISS Optimization, Vectorized Computation, Quantized FAISS Index, Dynamic Query Vectors` 


- **Vectorization Accelerates Similarity Calculations**: A member optimized cosine similarity calculations by replacing a loop with `query_embeddings @ doc_embeddings.T`, reducing runtime from **~25 seconds** to **~0.04 seconds** for a **1000 x 1000** matrix.
- **Considering Quantized FAISS for Larger Scales**: For over **10M comparisons**, the member plans to switch to a quantized FAISS index like `IndexIVFPQ` to manage memory and latency.
   - The user asks about caveats of using `IndexIVFPQ` with dynamic (not pre-indexed) query vectors and seeks feedback on the optimization plan.
- **Seeking Advice on Production Stability of matmul**: The original poster seeks feedback if `@` / `matmul` is stable for production at the **1M scale**.


  
