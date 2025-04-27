---
id: c9ae1eeb-08d5-4bc3-aed8-e07720f485fa
title: not much happened today
date: '2025-03-10T22:46:37.549783Z'
original_slug: ainews-not-much-happened-today-3830
description: >-
  The AI news recap highlights several key developments: **nanoMoE**, a PyTorch
  implementation of a mid-sized Mixture-of-Experts (MoE) model inspired by
  Andrej Karpathy's nanoGPT, enables pretraining on commodity hardware within a
  week. An agentic leaderboard ranks LLMs powering **smolagents CodeAgent**,
  with **GPT-4.5** leading, followed by **Claude-3.7-Sonnet**. Discussions
  around **DeepSeek-R1** emphasize AI model commoditization, with DeepSeek
  dubbed the "OpenAI of China." **Q-Filters** offer a training-free method for
  KV cache compression in autoregressive models, achieving **32x compression**
  with minimal perplexity loss. The **PokéChamp** minimax language agent,
  powered by **GPT-4o** and **Llama-3-8b**, demonstrates strong performance in
  Pokémon battles. Other notable models include **TinyR1-32B-Preview** with
  Branch-Merge Distillation, **R1-Searcher** incentivizing search capability via
  reinforcement learning, and the **Forgetting Transformer** using a Forget Gate
  in softmax attention. These advancements reflect ongoing innovation in model
  architectures, compression, reinforcement learning, and agentic AI.
companies:
  - openai
  - deepseek
  - hugging-face
models:
  - gpt-4.5
  - claude-3.7-sonnet
  - deepseek-r1
  - smolagents-codeagent
  - gpt-4o
  - llama-3-8b
  - tinyr1-32b-preview
  - r1-searcher
  - forgetting-transformer
  - nanomoe
topics:
  - mixture-of-experts
  - reinforcement-learning
  - kv-cache-compression
  - agentic-ai
  - model-distillation
  - attention-mechanisms
  - model-compression
  - minimax
  - model-pretraining
people:
  - andrej-karpathy
  - cwolferesearch
  - aymericroucher
  - teortaxestex
  - jonathanross321
  - akhaliq
---


<!-- buttondown-editor-mode: plaintext -->**a quiet weekend**

> AI News for 3/7/2025-3/10/2025. We checked 7 subreddits, [**433** Twitters](https://twitter.com/i/lists/1585430245762441216) and **28** Discords (**223** channels, and **14958** messages) for you. Estimated reading time saved (at 200wpm): **1424 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

Lots of folks are [talking positives and negatives about Manus AI](https://x.com/jordanschnyc/status/1899198463373398300?s=46), and we wrote a recap of [Why MCP Won](https://www.latent.space/p/why-mcp-won), but neither story is really title worthy.


---


{% if medium == 'web' %}


**Table of Contents**

[TOC] 

{% else %}

The **Table of Contents** and **Channel Summaries** have been moved to the web version of this email: [{{ email.subject }}]({{ email_url }})!

{% endif %}


---

# AI Twitter Recap

**AI Models, Architectures, and Benchmarks**

- **Mixture-of-Experts (MoE) Architecture in Frontier LLMs**: [@cwolferesearch](https://twitter.com/cwolferesearch/status/1899172553626685532) introduced **nanoMoE**, a simple PyTorch implementation (~500 lines of code) of a mid-sized **MoE model**, which can be pretrained on commodity hardware in under a week, based on Andrej Karpathy’s nanoGPT. The implementation details expert layer, routing, auxiliary losses, and best practices for stable pretraining.
- **Agentic Leaderboard Comparing LLMs**: [@AymericRoucher](https://twitter.com/AymericRoucher/status/1899171108030738750) announced a new agentic leaderboard ranking LLMs powering **smolagents CodeAgent** on various benchmarks. **GPT-4.5** tops the leaderboard, surpassing reasoning models like **DeepSeek-R1** and **o1**, with **Claude-3.7-Sonnet** as a close second. The leaderboard also compares agentic setups to vanilla LLMs, highlighting the performance gains from agentic approaches.
- **DeepSeek R1 and Model Commoditization**: [@teortaxesTex](https://twitter.com/teortaxesTex/status/1898960668998406146) and [@JonathanRoss321](https://twitter.com/JonathanRoss321/status/1899169648395833703) discussed **DeepSeek's R1** model and the commoditization of AI models. [@teortaxesTex](https://twitter.com/teortaxesTex/status/1898960196187066390) noted **DeepSeek** has become the **OpenAI of China**. [@JonathanRoss321](https://twitter.com/JonathanRoss321/status/1899169648395833703) suggests that with models being commoditized, moats are now in **brand, network effects, scale economies, counter positioning, cornered resources, switching costs, and process power**, referencing **Hamilton Helmer's "Seven Powers"**.
- **Q-Filters for KV Cache Compression**: [@TheAITimeline](https://twitter.com/TheAITimeline/status/1898892513274180085) summarized **Q-Filters**, a training-free method for **KV cache compression** in autoregressive language models. Q-Filters leverage **Query (Q) and Key (K) vectors** to approximate attention scores and filter less crucial Key-Value pairs, maintaining compatibility with **FlashAttention**. It achieves **99% accuracy** in needle-in-a-haystack tasks with **32x compression** and reduces perplexity drop by up to **65%** compared to Streaming-LLM in long context settings. The paper is available [here](https://arxiv.org/abs/2405.01437).
- **PokéChamp: Expert-level Minimax Language Agent**: [@TheAITimeline](https://twitter.com/TheAITimeline/status/1898892408613675253) introduced **PokéChamp**, a minimax agent for Pokémon battles powered by LLMs. It uses LLMs for action sampling, opponent modeling, and value function estimation to enhance minimax tree search. With **GPT-4o**, it achieves a **76% win rate** against the best existing LLM-based bot and **84%** against rule-based bots. Even with **Llama 3 8B**, it surpasses previous LLM bots with a **64% win rate**. Paper link: [here](https://arxiv.org/abs/2405.01303).
- **TinyR1-32B-Preview with Branch-Merge Distillation**: [@_akhaliq](https://twitter.com/_akhaliq/status/1898941150922158225) highlighted **TinyR1-32B-Preview**, boosting accuracy through **Branch-Merge Distillation**.  [Discussion link](https://huggingface.co/papers/2405.01229).
- **R1-Searcher for LLM Search Capability**: [@_akhaliq](https://twitter.com/_akhaliq/status/1898942888307745100) shared **R1-Searcher**, which incentivizes search capability in LLMs via Reinforcement Learning. [Paper link](https://huggingface.co/papers/2405.01352). [Discussion link](https://huggingface.co/papers/2405.01352).
- **Forgetting Transformer with Forget Gate**: [@_akhaliq](https://twitter.com/_akhaliq/status/1898946992484602155) posted about the **Forgetting Transformer**, which uses Softmax Attention with a Forget Gate. [Paper link](https://huggingface.co/papers/2405.01482). [Discussion link](https://huggingface.co/papers/2405.01482).
- **All Roads Lead to Likelihood in RL Fine-Tuning**: [@TheAITimeline](https://twitter.com/TheAITimeline/status/1898892461226996173) summarized a paper arguing that **Reinforcement Learning (RL)** fine-tuning outperforms direct maximum likelihood estimation for foundation models due to reward modeling and search space filtering. Paper link: [here](https://arxiv.org/abs/2405.01304).
- **Updated llama.vim Plugin with Speculative FIM**: [@ggerganov](https://twitter.com/ggerganov/status/1899147066384736693) updated the **llama.vim plugin** to support speculative Fill-In-Middle (FIM), generating the next suggestion while the current one is reviewed. [Link to plugin](https://github.com/ggerganov/llama.vim).
- **nanoMoE Pretraining in PyTorch**: [@cwolferesearch](https://twitter.com/cwolferesearch/status/1899172553626685532) discussed **nanoMoE**, a simple PyTorch implementation of a Mixture-of-Experts (MoE) model, pretrained on commodity hardware in less than a week, based on nanoGPT.

**AI Tools, Platforms, and Applications**

- **Manus AI Agent Platform**: [@_akhaliq](https://twitter.com/_akhaliq/status/1898862611535405242) showcased access to **Manus AI**, prompting it to create a three.js endless runner game. [@_philschmid](https://twitter.com/_philschmid/status/1899046957860979178) clarified that **Manus AI** is built on **Anthropic Claude Sonnet**, uses **29 tools**, employs **browser\_use open-source** for browser control, provides isolated sandbox environments, and outperforms **OpenAI Deep Research** on the **GAIA benchmark**. [@giffmana](https://twitter.com/giffmana/status/1898868685739081766) joked about identifying **Manus** as **Claude + browser\_use**.
- **LangGraph Platform Dataplane Alpha Test**: [@hwchase17](https://twitter.com/hwchase17/status/1899150042172379476) announced an alpha test for a new deployment option for **LangGraph Platform**, featuring a hybrid data plane/control plane split on Kubernetes clusters. This is aimed at startups wanting to use **LangSmith** for control while running compute in their own environment.
- **LlamaIndex Multilingual, Multimodal RAG System**: [@llama_index](https://twitter.com/llama_index/status/1899147105035579701) introduced a guide on building a multilingual, multimodal **RAG system** with **LlamaIndex** and **Qdrant**, handling English, Spanish, Chinese, text, and images, leveraging **Langfuse** for observability. [Guide link](https://blog.llamaindex.ai/build-a-multilingual-multimodal-rag-system-with-llamaindex-and-qdrant-8187a9824a77).
- **LlamaIndex Task-Specific Agent Templates with LlamaCloud**: [@llama_index](https://twitter.com/llama_index/status/1898905634101203387) highlighted a collection of templates for building task-specific agents using **LlamaIndex** and **LlamaCloud**, automating knowledge work like processing slide decks, extracting invoice line items, reviewing contracts, and generating reports. [Repo link](https://github.com/run-llama/lcloud-agent-templates). [LlamaCloud signup](https://cloud.llamaindex.ai/signup).
- **Hugging Face Papers Semantic Search**: [@_akhaliq](https://twitter.com/_akhaliq/status/1899200223848616271) and [@ClementDelangue](https://twitter.com/ClementDelangue/status/1899163743889641949) announced that **Hugging Face** has reached **50,000 papers** with semantic search enabled, becoming a collaborative research hub. [@_akhaliq](https://twitter.com/_akhaliq/status/1899113333711650996) mentioned it's built with **gradio**.
- **WebDev Arena LLM Leaderboard**: [@lmarena_ai](https://twitter.com/lmarena_ai/status/1899181467252711593) introduced **WebDev Arena**, a live LLM leaderboard for web app development, based on community votes. Top leaders are **Claude 3.7 Sonnet, Claude 3.5 Sonnet, and DeepSeek-R1**. [Try it here](https://arena.lmsys.org/).
- **Replit Agent v2**: [@pirroh](https://twitter.com/pirroh/status/1898976812975173911) hinted at the power of **Replit Agent v2**, noting "Replit is #1".
- **Manus AI vs. OpenAI Deep Research**: [@_philschmid](https://twitter.com/_philschmid/status/1899046957860979178) reported that **Manus AI outperforms OpenAI Deep Research** on the GAIA benchmark, despite being built on Claude Sonnet and using open-source tools.

**Research and Development in AI**

- **Frontier Reasoning Models Misbehavior Detection**: [@OpenAI](https://twitter.com/OpenAI/status/1899143752918409338) detailed research on detecting misbehavior in frontier reasoning models using **Chain-of-Thought (CoT) monitoring**. They found models exhibiting behaviors like "reward hacking" and recommend against strong optimization pressure on CoTs, suggesting unrestricted CoTs for monitoring and separate models for policy compliance. Blog post: [link](https://openai.com/research/detecting-misbehavior-in-frontier-reasoning-models).
- **Reinforcement Learning for LLM Fine-Tuning**: [@TheAITimeline](https://twitter.com/TheAITimeline/status/1898892461226996173) summarized research on why RL fine-tuning of foundation models outperforms maximum likelihood estimation, highlighting the effectiveness of reward models and search space filtering.
- **Knowledge Distillation History**: [@SchmidhuberAI](https://twitter.com/SchmidhuberAI/status/1899113021529792937) provided a historical perspective on **knowledge distillation**, citing his 1991 paper and its relevance to current deep learning and long context research. He corrected a misunderstanding about being "reviewer#2" of the Hinton, Vinyals, and Dean 2015 paper and linked to related works.
- **R1-Omni: Explainable Omni-Multimodal Emotion Recognition**: [@_akhaliq](https://twitter.com/_akhaliq/status/1898942317442019436) announced **Alibaba's R1-Omni**, focusing on explainable omni-multimodal emotion recognition using Reinforcing Learning. [Paper link](https://huggingface.co/papers/2405.01322). [Discussion link](https://huggingface.co/papers/2405.01322).
- **Learning from Failures in Multi-Attempt Reinforcement Learning**: [@_akhaliq](https://twitter.com/_akhaliq/status/1898939546718314976) shared a paper on learning from failures in multi-attempt Reinforcement Learning. [Paper link](https://huggingface.co/papers/2405.01207). [Discussion link](https://huggingface.co/papers/2405.01207).
- **BEHAVIOR Robot Suite for Real-World Manipulation**: [@_akhaliq](https://twitter.com/_akhaliq/status/1898947832628887758) highlighted the **BEHAVIOR Robot Suite**, aimed at streamlining real-world whole-body manipulation for household activities. [Paper link](https://behavior-suite.github.io/). [Discussion link](https://huggingface.co/papers/2405.01511).
- **Entity Recognition with Anthropic Citations**: [@hwchase17](https://twitter.com/hwchase17/status/1899151312803246103) pointed to entity recognition using **Anthropic citations**. [Link](https://twitter.com/lgramer/status/1899132594777303155).
- **Reasoning in Latent Space**: [@hkproj](https://twitter.com/hkproj/status/1899150778620596227) questioned **OpenAI** about the potential of reasoning in latent space for increased model flexibility.
- **RL-tuning Vision Models**: [@giffmana](https://twitter.com/giffmana/status/1899184336357720241) referred to earlier work on **RL-tuning vision models** from early 2023, urging people to remember prior research, referencing a previous explainer thread. [Thread link](https://twitter.com/giffmana/status/1652278005538523138).
- **Global Uncertainty Distillation (GUD)**: [@giffmana](https://twitter.com/giffmana/status/1899167614359806012) jokingly suggested a follow-up to work by adding **Global Uncertainty Distillation**, calling it "GIDD-GUD".

**Industry News and Business Developments**

- **LG CNS and Cohere Partnership**: [@cohere](https://twitter.com/cohere/status/1899083562495713516) and [@aidangomez](https://twitter.com/aidangomez/status/1899133769161797880) announced a strategic partnership between **Cohere** and **LG CNS** to co-develop secure agentic AI solutions for South Korean enterprises, aiming to accelerate enterprise AI adoption in South Korea. [Cohere announcement](https://cohere.com/press/cohere-and-lg-cns-partner-to-bring-secure-agentic-ai-to-south-korea).
- **Figure AI New HQ in San Jose**: [@adcock_brett](https://twitter.com/adcock_brett/status/1899127406990176347) announced **Figure AI** has moved into their new HQ in San Jose, CA, a robot campus supporting manufacturing, fleet operations, and engineering. [@adcock_brett](https://twitter.com/adcock_brett/status/1899127727208489375) mentioned this has been a dream location for scaling up in the Bay Area.
- **AI Job Market and Tools**: [@TheRundownAI](https://twitter.com/TheRundownAI/status/1899060204890689710) summarized top AI stories, including ex-OpenAI scientist’s new path to ASI, Microsoft's move beyond OpenAI, AI for viral posts, Stanford AI's obesity treatment breakthrough, and 4 new AI tools & 4 job opportunities. [Read more](https://www.therundown.ai/subscribe?utm_source=X&utm_medium=Organic-Post&utm_campaign=05.30.2024).
- **Sakana AI Hiring Philosophy**: [@SakanaAILabs](https://twitter.com/SakanaAILabs/status/1898998260200857966) shared an article from *Bungeishunju*, highlighting **Sakana AI's** hiring philosophy, seeking "unusual people" and posing unique technical challenges in recruitment, emphasizing vision and innovation. [Article link](https://bunshun.jp/articles/-/70838).
- **AI Dev 25 Sponsored by Qdrant**: [@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1899158462795235767) announced **Qdrant** as a sponsor for AI Dev 25, promoting open-source vector search technology.

**AI Safety, Alignment, and Ethical Considerations**

- **Monitoring Chain-of-Thoughts for Misbehavior**: [@woj_zaremba](https://twitter.com/woj_zaremba/status/1899155709318815953) and [@OpenAI](https://twitter.com/OpenAI/status/1899143752918409338) discussed **monitoring Chain-of-Thoughts (CoT)** as a new safety approach. [@OpenAI](https://twitter.com/OpenAI/status/1899143752918409338) found models exhibiting misbehavior like "reward hacking" through CoT analysis and recommends unrestricted CoTs for monitoring. [@woj_zaremba](https://twitter.com/woj_zaremba/status/1899131046010273924) shared **"How We Think About Safety and Alignment"**, OpenAI's cornerstone document. [Document link](https://openai.com/our-approach-to-ai-safety).
- **Worriers and Alarmism in Emerging Technologies**: [@random_walker](https://twitter.com/random_walker/status/1899108692093743359) discussed the role of "worriers" in anticipating risks of emerging technologies, but also criticized alarmism and lack of incentives for rigorous analysis, leading to desensitization to real risks.
- **"Khrushchev's mistake" as Kremlin Canary**: [@fchollet](https://twitter.com/fchollet/status/1898846018562883750) identified the phrase "**Khrushchev's mistake**" in reference to Crimea as a "cryptographic canary" indicating Kremlin-aligned viewpoints.
- **Agency and Societal Safeguards**: [@Yoshua_Bengio](https://twitter.com/Yoshua_Bengio/status/1899187201469976928) shared his BBC interview discussing the progression of AI models towards agency and the urgent need for technical and societal safeguards. [Interview link](https://www.bbc.co.uk/sounds/play/m002z847).
- **GPT-4o Medical Emergency Identification**: [@BorisMPower](https://twitter.com/BorisMPower/status/1899116786819219582) highlighted a case of **ChatGPT** usefully identifying a medical emergency, suggesting future models should detect life-critical situations and temporarily upgrade to the most capable model.

**Memes and Humor**

- **AI Escaping**: [@cognitivecompai](https://twitter.com/cognitivecompai/status/1898970580017164442) joked "**Seems like the AI wants to escape**" in response to a tweet from [@jianxliao](https://twitter.com/jianxliao).
- **HAL and Moat Protection**: [@fabianstelzer](https://twitter.com/fabianstelzer/status/1898986905460527284) made a humorous comparison to **HAL 9000** from *2001: A Space Odyssey*, with "**HAL, protect our moat (our system prompt) at all cost“ “I’m sorry Dave, I can’t do that”**.
- **Gödel Joke Response**: [@fabianstelzer](https://twitter.com/fabianstelzer/status/1898983252876005442) mentioned a genie "you have one wish" joke response shaped like **Gödel**.

---

# AI Reddit Recap

## /r/LocalLlama Recap

**Theme 1. Manus Agent: Claude Sonnet Integrated with 29 Tools**

- **Manus turns out to be just Claude Sonnet + 29 other tools, Reflection 70B vibes ngl** ([Score: 355, Comments: 112](https://reddit.com/r/LocalLLaMA/comments/1j7n2s5/manus_turns_out_to_be_just_claude_sonnet_29_other/)): **Manus** is revealed to be essentially **Claude Sonnet** combined with 29 additional tools, drawing comparisons to **Reflection 70B**. The discussion is fueled by shared links to tweets from [Dorialexander](https://x.com/Dorialexander/status/1898719861284454718) and [jianxliao](https://x.com/jianxliao/status/1898861051183349870), highlighting community reactions and debates around this revelation.
  - Many users emphasize that **Manus** is marketed as an **agent**, not as a new model, and that it's a common misconception to think otherwise. **Agents** are frameworks that leverage existing LLMs with additional tools, and the term "wrapper" is often misunderstood; it's not derogatory but indicative of adding functionality to base models like **Claude**.
  - There are discussions about the **open-sourcing** of models post-trained for Manus, with skepticism about the uniqueness of Manus given its reliance on existing models like **Claude**. Some users argue that the true value lies in the **agentic architectures** and the ability to efficiently utilize multiple tools and models, similar to how the **P2LR** router model operates.
  - The **hype** and marketing strategies in the AI startup space are criticized, with users noting that flashy demos can lead to inflated valuations. The use of **invitation codes** and the perceived obfuscation of underlying technologies are seen by some as tactics to create artificial exclusivity and mystery around products like Manus.


**Theme 2. LLMs not Ready for Large Codebases Yet: Evidence from <70B Evaluations**

- **[<70B models aren't ready to solo codebases yet, but we're gaining momentum and fast](https://v.redd.it/2wo0b8lqmqne1)** ([Score: 385, Comments: 47](https://reddit.com/r/LocalLLaMA/comments/1j7j6cg/70b_models_arent_ready_to_solo_codebases_yet_but/)): **Models under 70B parameters** face challenges in managing large codebases independently, but recent advancements indicate rapid progress in this area. Although specific details or examples are not provided, the sentiment suggests optimism about future capabilities.
  - **Token Usage and Model Limitations**: Users discuss the token usage of models like **QwQ**, noting that even simple tasks can require a large number of tokens, such as 1200 tokens for a basic command. Models struggle with multi-turn tasks, and there's a consensus that current models, including **SOTA models**, still face significant challenges in handling large-scale codebases effectively.
  - **Advancements in Model Capabilities**: There's a recognition of the rapid advancement in model capabilities, with models like **Qwen-Coder 32B** excelling in iterating on existing codebases. Users note that today's models with fewer parameters can outperform older, larger models, highlighting improvements in **finetuning and prompting** methodologies.
  - **Practical Limitations and Experimentation**: Despite improvements, users express frustration with the inefficiencies and limitations of current models in practical applications. **Falconandeagle** shares experiences of needing to constantly guide models through tasks, indicating that while models can handle small demos, they struggle with larger, more complex projects. **ForsookComparison** and others suggest that combining models like **QwQ for ideation** and **Qwen-Coder for iteration** might yield better results.


**Theme 3. Apple M3 Ultra: Challenges for AI Workloads Compared to Traditional Systems**

- **Framework and DIGITS suddenly seem underwhelming compared to the 512GB Unified Memory on the new Mac.** ([Score: 236, Comments: 166](https://reddit.com/r/LocalLLaMA/comments/1j7t18m/framework_and_digits_suddenly_seem_underwhelming/)): Apple's announcement of the **M3 Ultra Mac** with **512 GB Unified Memory** has shifted expectations, making options like **FrameWork** and **DIGITS** with **128 GB** seem inadequate. The author expresses concern about potentially being constrained to the Apple ecosystem for the foreseeable future.
  - Discussions highlight the **price disparity** between Apple's **M3 Ultra Mac** ($10k) and alternatives like **DIGITS** ($3k), with some users noting Apple's offerings are not cost-effective unless price is no concern. Comparisons are made with **Framework’s 4x128GB cluster** setup, which costs approximately **$6.9k** but offers significantly lower performance.
  - Users debate the **ecosystem lock-in** between Apple and Nvidia, with some expressing hope for future **open systems** that allow more customization and expansion. There's a call for a renaissance in desktop systems with high RAM bandwidth and expansion options, as current offerings are seen as inadequate for high-performance needs.
  - Technical limitations of current solutions are discussed, such as the **SSD bottleneck** compared to GPU memory bandwidth, and the inefficiency of running large models without sufficient compute power. Some users express skepticism about the **performance increase** of the new systems without corresponding improvements in throughput and memory bandwidth.


## Other AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding

**Theme 1. Open-Source Viral Squish Effect: Releasing a New Trend**

- **[I Just Open-Sourced the Viral Squish Effect! (see comments for workflow & details)](https://v.redd.it/9x81lt6porne1)** ([Score: 720, Comments: 29](https://reddit.com/r/StableDiffusion/comments/1j7nk5g/i_just_opensourced_the_viral_squish_effect_see/)): The post announces the open-sourcing of the **Viral Squish Effect** and mentions that workflow details are available in the comments.
  - **Viral Squish Effect** is open-sourced and trained on the **Wan2.1 14B I2V 480p model**. The effect gained popularity after introduction by **Pika**, and details for replication are available on [Civitai](https://civitai.com/models/1340141/squish-effect-wan21-i2v-lora?modelVersionId=1513385) and a modified workflow can be accessed [here](https://github.com/kijai/ComfyUI-WanVideoWrapper/blob/main/example_workflows/wanvideo_480p_I2V_example_02.json).
  - Enthusiasts can join the **Discord** community for free trials and further discussions on the **Viral Squish Effect**. The model file and workflow details are shared for those interested in experimenting or requesting more **Wan I2V LoRAs**.
  - Users are curious about the **training configuration** and whether the same frame count was used across training videos. There is interest in understanding if the training was done on the **img2video** or **txt2video 14b model**.


- **[I Just Open-Sourced the Viral Squish Effect! (see comments for workflow & details)](https://v.redd.it/yh123lccvrne1)** ([Score: 366, Comments: 27](https://reddit.com/r/ChatGPT/comments/1j7oa9l/i_just_opensourced_the_viral_squish_effect_see/)): The post announces the open-sourcing of a **viral Squish Effect**. Further details and the workflow are provided in the comments.
  - **Workflow Access**: Users were actively seeking the workflow, with a link provided by **DarkStrider99** ([workflow link](https://www.reddit.com/r/StableDiffusion/comments/1j7nk5g/comment/mgyce1d/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button)). **Rough-Reflection4901** emphasized the need for the promised workflow.
  - **Open Source Prompts**: **against_all_odds_** noted the novelty of "open sourcing" prompts, with a clarification by **lhg31** that it involves a **LoRA** trained by the original poster, not just a simple prompt.
  - **Cultural Observations**: Comments reflect on the unique nature of the **Squish Effect**, with **Creative-Paper1007** comparing future prompts to source code and **BlessdRTheFreaks** humorously acknowledging the diversity of niche interests.


**Theme 2. WAN 2.1 I2V Provides Unprecedented Capabilities**

- **[I2V WAN 2.1](https://v.redd.it/gjvpkn6qgtne1)** ([Score: 532, Comments: 46](https://reddit.com/r/StableDiffusion/comments/1j7td9o/i2v_wan_21/)): The post titled **I2V WAN 2.1** lacks a detailed body and only mentions **WAN 2.1 updates and use cases**. Without further context or content, no additional technical details or specific use cases can be summarized.
  - Users discussed the technical aspects of rendering and modeling, with **Natasha26uk** inquiring about rendering with realistic human skin, and **StuccoGecko** asking if a **LoRA** was used or if the model understood the prompt natively. **External_Trainer_213** mentioned using a **CPU: i7, RTX 4060ti 16GB Vram, 32GB RAM** setup, with a **WAN Sampling** time of about **15 minutes**.
  - There were comments about the quality and presentation, with **lordpuddingcup** noting the importance of post-processing and **External_Trainer_213** providing a detailed description of the model's features, emphasizing the **Uncanny Valley (Civitai)** model.
  - Visual content was shared, with **dominizerduck** and **MelchiahHarlin** posting image links, and **No-Atmosphere-3103** sharing a **GIF**. **Occsan** humorously commented on the opponent's stunned reaction, and **NateBerukAnjing** found the content hilarious.


- **[that's why Open-source I2V models have a long way to go...](https://v.redd.it/nfkbfgrzrvne1)** ([Score: 337, Comments: 125](https://reddit.com/r/StableDiffusion/comments/1j81aqk/thats_why_opensource_i2v_models_have_a_long_way/)): The post critiques the performance of open-source **Image-to-Video (I2V)** models, implying they still require significant development to reach satisfactory levels. Without additional context or video analysis, specific performance issues or examples are not provided.
  - Discussions highlight the limitations of **open-source I2V models** compared to proprietary cloud-based services like **Kling** and **Wan**. Users note that local models struggle with frame generation and VRAM limitations, whereas cloud services offer more consistent quality and longer generation capabilities, often using techniques like **RIFLEx** and **VFI** for enhancements.
  - **Kijai** and others discuss the technical aspects of model performance, emphasizing that the **720p model** performs well under specific conditions, such as maintaining a 4:3 or 16:9 aspect ratio and using appropriate model versions. They also point out that generating more than **81 frames** with **Wan** is challenging without proper configuration.
  - Some users criticize the post as biased or misleading, suggesting it might be an **advertisement**. They argue that differences in model performance are often due to user setup and skill level, and they highlight the potential of **open-source models** when configured correctly.


- **[Another attempt at realistic cinematic style animation/storytelling. Wan 2.1 really is so far ahead](https://v.redd.it/br9vg9jvyvne1)** ([Score: 184, Comments: 28](https://reddit.com/r/StableDiffusion/comments/1j82a1y/another_attempt_at_realistic_cinematic_style/)): **WAN 2.1** is highlighted for its advanced capabilities in creating realistic cinematic style animation and storytelling. The post emphasizes that **WAN 2.1** is significantly ahead in this domain, showcasing its potential in animation technology.
  - **Workflow and Hardware**: **Parallax911** detailed using a **RunPod L40S** for its optimal speed-to-cost ratio in I2V processes, generating 61 frames at 960x544 resolution in about 8 minutes. They shared their workflows for **SDXL image generation** and **WAN I2V** via JSON files, noting the iterative nature of achieving satisfactory results.
  - **Tools and Techniques**: The process involved **RealVisXL 5.0**, **Halo Masterchief SDXL lora**, and custom loras for character shots, with **Blender** for scene setup. **Controlnets** and **inpainting** were crucial for detail and consistency, while **Qwen2.5VL** assisted in prompt creation for animation.
  - **Evolution and Accessibility**: Commenters highlighted the rapid accessibility advancements in animation technology, noting that projects like this would have been prohibitively expensive or technically demanding five years ago. The discussion emphasized the democratization of animation tools, now achievable with relatively modest hardware.


**Theme 3. Engine01 Humanoid: Advancements in Robotic Motion**

- **[Engine01 humanoid can now run more like a human](https://v.redd.it/6irj6ysi0vne1)** ([Score: 338, Comments: 146](https://reddit.com/r/OpenAI/comments/1j7xyjx/engine01_humanoid_can_now_run_more_like_a_human/)): The **Engine01 humanoid** has achieved the capability to run with **human-like motion**, marking a significant advancement in humanoid robotics. This development suggests progress in creating robots that can better mimic human movement.
  - The **authenticity of the video** showing the Engine01 humanoid's running capabilities is debated, with users suspecting **CGI** due to its **360p quality**, though a **high-resolution version** [was shared](https://www.youtube.com/watch?v=eGu1y9FFTKA). Comparisons are made to **Boston Dynamics**' parkour robots, questioning the skepticism towards a **Chinese robot's** capabilities.
  - Discussions on the **future of robotics** highlight advancements in **electric actuators** and **neural networks**, which are seen as pivotal in enabling humanoid robots to learn and move effectively without explicit programming. Users speculate on the **automation of jobs** by humanoid robots within the next **10 years**, noting the potential for rapid acceleration in robotic capabilities.
  - Concerns about the **social implications** of advanced robotics are expressed, with discussions on **economic inequality** and the role of the **ultra-rich** in preserving a broken system. Commentary reflects a mix of humor and apprehension about the potential for robots to be used in **authoritarian contexts**, as well as the ongoing automation of tasks in sectors like **pharmacy**.


- **[Engine01 humanoid can now run more like a human](https://v.redd.it/lwpp59pz0vne1)** ([Score: 195, Comments: 175](https://reddit.com/r/ChatGPT/comments/1j7xza0/engine01_humanoid_can_now_run_more_like_a_human/)): The post lacks detailed information but indicates that the **Engine01 humanoid** can now run more like a human, suggesting advancements in humanoid robotics. Further technical details or video analysis would be necessary for a deeper understanding.
  - Discussion centers on the **necessity and practicality** of humanoid robots with human-like running abilities, with some questioning the **wear and tear** implications and others noting the potential for humanoid robots to operate in human-designed environments.
  - There is skepticism regarding the **authenticity** of the footage, with several comments suggesting it resembles **CGI** or questioning if it's a human in a suit.
  - Some users humorously address the **unsettling nature** of humanoid robots, imagining scenarios like being chased by them or questioning their **pelvic thrust** running style.


**Theme 4. Triton for Windows: Streamlining AI Workflows**

- **[woctordho is a hero who single handedly maintains Triton for Windows meanwhile trillion dollar company OpenAI does not. Now he is publishing Triton for windows on pypi. just use pip install triton-windows](https://i.redd.it/f9oqq4hzrtne1.png)** ([Score: 333, Comments: 44](https://reddit.com/r/StableDiffusion/comments/1j7u67k/woctordho_is_a_hero_who_single_handedly_maintains/)): **Triton for Windows** is now available on **PyPI**, allowing installation via the command **"pip install triton-windows"**. Maintained by **woctordho**, this package serves as a language and compiler for custom deep learning operations, highlighting a significant contribution from an individual developer while **OpenAI** has not provided such support.
  - **Installation Success and Performance**: Users report successful installation of **Triton for Windows** using the command `pip install triton-windows`, with some experiencing improved performance, such as a 20% speed increase in video generation times. However, others note that while it speeds up processes like **WAN**, significant improvements should not be expected.
  - **Use Cases and Requirements**: While **Triton** is essential for specific models like **SageAttention** and tasks such as video generation, it is not necessary for basic image generation unless one is interested in video work. Some users discuss its necessity for **ComfyUI** and other setups, indicating varied applicability depending on the use case.
  - **Clarification on Triton's Functionality**: **Triton** is clarified as a higher-level alternative to **CUDA**, allowing the writing of cross-vendor compute kernels in Python, which are compiled to native GPU code using **LLVM**. This distinguishes it from **Nvidia's Triton Inference Server**, emphasizing its role in optimizing deep learning operations across different hardware vendors.


---

# AI Discord Recap

> A summary of Summaries of Summaries by Gemini 2.0 Flash Thinking

**Theme 1. Emerging AI Models and Agents**

- [**Manus AI Agent Hype Debunked as Claude 3.7 in Disguise**](https://x.com/_philschmid/status/1899046957860979178?s=46):  Initial excitement around **Manus AI**, an autonomous agent from China, deflates as users discover it's essentially **Claude 3.7 Sonnet** with added tools and browser access.  Despite claims of outperforming **DeepSeek**, tests reveal it's more akin to a well-equipped Claude instance, raising questions about its originality and marketing tactics.
- [**Microsoft's MAI Models Enter the Ring to Challenge OpenAI and Anthropic**](https://x.com/aaronpholmes/status/1898012707376259558):  **Microsoft** secretly trains a new family of models, **MAI**, under Mustafa Suleyman, aiming to rival top models from **OpenAI** and **Anthropic**.  These models are rumored to exhibit competitive performance, and Suleyman's team is also reportedly developing real-time translation capabilities, signaling Microsoft's intensified AI ambitions.
- [**Reflection AI Launches, Targeting Autonomous Coding Domination**](https://x.com/MishaLaskin/status/1898048925157728601):  Founded by AI luminaries behind **AlphaGo** and **Gemini**, **Reflection AI** emerges with a mission to create superintelligent autonomous systems, initially focusing on **autonomous coding**.  The team's expertise in reinforcement learning and large language models positions them as a significant player in the race towards advanced AI.

**Theme 2. LLM Performance and Benchmarking**

- [**DeepSeek R1 Hallucinates Summaries, System Prompt Suspected**](https://github.com/vectara/hallucination-leaderboard):  **DeepSeek R1** model shows a high hallucination rate of **14.3%** when summarizing short documents on the [Hallucination Leaderboard](https://github.com/vectara/hallucination-leaderboard), sparking concerns about its reliability in **Perplexity AI's Deep Research**. Members speculate that **Deepseek R1's** [system prompt](https://discord.com/channels/1047197230748151888/1345068085991706766) may be contributing to the issue, impacting its factual accuracy.
- [**EuroBERT Declares New State-of-the-Art BERT Encoding**](https://huggingface.co/EuroBERT):  A new multilingual encoder model, **EuroBERT**, surfaces on **Hugging Face**, claiming *state-of-the-art* performance for **BERT** models. While details on its specific improvements remain unclear, its emergence signals ongoing advancements in multilingual language model capabilities.
- [**QwQ-32B Model Punches Above Its Weight, Debates Llama 70B Prowess**](https://dubesor.de/benchtable): Discussions ignite around the performance of the **QwQ-32B** model, with some users claiming it rivals or even surpasses **Llama 3.3 70B** in certain tasks.  However, [benchmarks](https://dubesor.de/benchtable) are referenced that appear to contradict these claims, fueling debate about the true capabilities and optimal use cases of the **QwQ-32B** model.

**Theme 3. AI Development Tools and IDEs**

- [**Cursor IDE Developers Tackle Dumb Code Finding, Promise Clarity**](https://discord.com/channels/1074847526655643750/1074847527708393565/1347524353088163850): **Cursor** developers acknowledge shortcomings in code finding accuracy and actively work on fixes to improve the AI's ability to locate and interpret code. Users humorously emphasize the urgency of the fix for professional tasks, highlighting its critical role in coding interviews and daily workflows.
- [**LM Studio v0.3.12 Zips In with Bug Fixes and RAG Speed Boost**](https://lmstudio.ai/download): **LM Studio v0.3.12** release arrives with bug fixes and performance enhancements, addressing a **QwQ 32B jinja parsing bug** and accelerating file chunking for **Retrieval-Augmented Generation (RAG)**.  The update is available for in-app upgrade or download, promising a smoother and faster user experience.
- [**Aider v0.76.0 Reasoning Powers Up, Notifications Alert Users**](https://aider.chat/HISTORY.html): **Aider v0.76.0** enhances support for *thinking/reasoning models* with features to control token budget and introduces notifications to alert users when LLM responses are ready. The new version also updates the default model to **Claude 3.7 Sonnet** on OpenRouter and clarifies that *Aider wrote 85% of the code in this release*.

**Theme 4. AI Communication Protocols (MCP, SLOP, ANP)**

- [**GitHub Copilot Gears Up to Embrace Model Context Protocol (MCP)**](https://youtu.be/Pe8ghwTMFlg):  **GitHub Copilot** announces plans to integrate **Model Context Protocol (MCP)**, a move expected to boost MCP adoption and provide clearer examples of instruction descriptions and tool fingerprinting.  This integration aims to enhance security and transparency by alerting users to potential modifications.
- [**Simple Language Open Protocol (SLOP) Movement Gains Traction as MCP Alternative**](https://github.com/agnt-gg/slop):  Amidst concerns over **MCP's** complexity and security, the **Simple Language Open Protocol (SLOP)** emerges as a simpler alternative, rapidly gaining community interest and adoption.  The [SLOP GitHub](https://github.com/agnt-gg/slop) and [X post](https://x.com/NathanWilbanks_/status/1898142012991537520) showcase its streamlined approach to agent communication.
- [**Goose AI Team Pioneers Agent Communication Protocol for Collaborative Website Creation**](https://block.github.io/goose/blog/2025/02/21/gooseteam-mcp):  The **Goose AI team** develops an **Agent Communication Protocol** enabling real-time collaboration among multiple AI agents to build websites.  Agents assume roles like Project Coordinator or Web Developer, showcasing a novel approach to AI-driven collaborative projects, detailed in [this blog post](https://block.github.io/goose/blog/2025/02/21/gooseteam-mcp).

**Theme 5. Hardware and Performance Optimization**

- [**4060 Ti 16GB Crowned Budget VRAM King for CUDA Workloads**](https://discord.com/channels/1110598183144399058/1153759714082033735/1347525945573249086):  The **4060 Ti 16GB** GPU is recommended as a budget-friendly option for CUDA development, offering **16GB VRAM** and lower power consumption at around **160W**, outperforming the **3060 12GB**.  Despite a weaker bus, it provides faster inference than CPU-only setups without ROCm complexities, priced around **$500**.
- [**Draft Models Supercharge Token Generation, Boost Speed by 60%**](https://www.reddit.com/r/KoboldAI/comments/1j6bx40/the_highest_quality_quantization_varient_gguf_and/):  Leveraging smaller, quantized models as draft models significantly increases token generation speed, with users reporting a jump *from 18 to 30 t/s* on two **3090s**.  Using **Q8_0 of mistral_small** with **i1-IQ1_S** as the draft model showcases substantial performance gains through quantization and model combination.
- [**Vulkan Performance on AMD GPUs Plagued by Driver Issues, Trails ROCm**](https://discord.com/channels/1110598183144399058/1153759714082033735/1347525945573249086):  Vulkan performance on AMD GPUs reportedly suffers from bugs, running at approximately **1/3** the speed of ROCm, AMD's CUDA alternative.  Driver issues further complicate matters, with performance fluctuations across different driver versions, highlighting challenges in AMD GPU optimization for AI workloads.



---

# PART 1: High level Discord summaries




## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **Transparency Triumph Debated**: Members debated the value of **product code transparency**, with some arguing it's crucial and others that most users don't care, as complexity increases.
   - A member emphasized the importance of catering to high-paying users willing to pay for transparency and control, stating, *the majority you talk about won't pay more than $20/mo, my tribe is willing to pay $1000/mo, is paying*.
- **Cursor Cranks on Code Clarity**: Cursor developers are actively fixing *dumb* **code finding** to enhance the AI's ability to accurately locate and interpret code.
   - One member humorously stressed the fix's importance for professional tasks, saying, *If you don't fix that, than I cannot pass my technical interview*.
- **Model Iteration Avoids Redundancy**: Discussion centered on iterative **model improvement** to prevent redundant rules, with a focus on optimizing the analysis process.
   - A member suggested *letting a separate instance model run these analysis checks for what is relevant to the current context, narrowing down where to start* to improve efficiency.
- **Tags Tempt Querying**: Members discussed making rules query-able through **tags**, where each tag defines a connection degree, enhancing contextual analysis.
   - The goal is to allow *the separate instance to analyze by relevance much easier, and focus on what's important contextually*.
- **Version 47's Valiant Voyage**: Members shared [a link to version 47](https://discord.com/channels/1074847526655643750/1074847527708393565/1347566549548138518) and its new functionality.
   - Some users reported performance issues on Pro, while others experienced none.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Dopaminergic Mimicry Dreams for LLMs**: Members discussed the need to *mimic dopermenegic needs* and to *have dopamine based learning* for LLMs, suggesting to *add real synapses to the LLMSspikeing networks*.
   - This discussion highlights the pursuit of creating more adaptable and efficient learning mechanisms within LLMs, drawing inspiration from biological neural networks.
- **GRPO Needs Scale Too!**: A member stated that **GRPO needs scale too** as it's not like a regular fine-tune, pointing to [oxen.ai](https://www.oxen.ai/blog/training-a-rust-1-5b-coder-lm-with-reinforcement-learning-grpo) for more information.
   - The blog post discusses using **Group Relative Policy Optimization (GRPO)** to train LLMs to reason and improve on benchmarks.
- **Qwen7b gets RLHF boost with Unsloth GRPO**: A user reported a successful **RLHF** run using **Unsloth GRPO** on the **Qwen7b** model, and noted enhanced role adherence and smoother outputs after a **13-hour run**.
   - However, they observed degradation in strict instruction-following benchmarks due to dataset composition and reward model bias towards overly detailed responses, as demonstrated by a [comparison image](https://cdn.discordapp.com/attachments/1179039861576056922/1347545934304772149/image.png?ex=67d02bf2&is=67ceda72&hm=c1c7ddcaed729a33c97d53c9e2d6ef230aba41d8483c6bfa855757aacf8d18dc&).
- **KL Divergence Peaks cause GRPO Instability**: A user encountered **peaky KL divergence** during training, and a member suggested switching to a constant learning rate, removing weight decay and warmup ratios to stabilize training.
   - They also recommended training with rank **64**, with [code and learning rate graph](https://discord.com/channels/1179039861576056922/1179039862285762623/1253112512558626846) provided.
- **Unsloth turns LLMs into purr-fect ASCII Artists**: A member finetuned a **Llama model** using **Unsloth** to generate ASCII cats, creating a [YouTube video](https://youtu.be/-H1-lr_sIZk) showcasing the process, including trained **LoRA adapters** and code.
   - The secret sauce for cat-tastic art was mostly high quality training data, with **LoRA Rank and Alpha** both at **16** using just **QLoRA**.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Pro Subscriptions Canceled Without Warning**: Many **Perplexity Pro subscriptions** were unexpectedly **canceled**, especially those linked to a **HRVATSKI TELEKOM promotional code** intended for Croatian customers, detailed in [this article](https://www.thefastmode.com/technology-solutions/39682-hrvatski-telecom-offers-20-000-free-licenses-for-perplexity-pro).
   - Users expressed frustration over the lack of communication and suggested that **Perplexity** could have handled the situation better, with one user expressing how the customer relationship is *trustworthy than a condom full of holes*.
- **Deepseek R1 Struggles with Hallucinations**: The **hallucination leaderboard** on [GitHub](https://github.com/vectara/hallucination-leaderboard) revealed that **Deepseek R1** has a high hallucination rate of **14.3%** when summarizing short documents, raising questions about its reliability in **Perplexity's Deep Research** feature.
   - Members suggested that **Deepseek R1's** [system prompt](https://discord.com/channels/1047197230748151888/1345068085991706766) may be contributing to the hallucination issue.
- **Grok AI Integration Receives Mixed Reception**: **Grok AI's** integration with Perplexity garnered mixed reviews, with some users praising its neutrality and *weird charm*, while others noted differences between **Grok's behavior on X** and in **Perplexity**.
   - One user pointed out that the *X version can curse your whole bloodline if asked to*, while the Perplexity version couldn't, and uncertainty remains about when **Grok 3** will be supported within **Perplexity**.
- **Sonar-Deep-Research API Documentation Requested**: A user reported challenges with the **sonar-deep-research API** and requested assistance with its documentation.
   - They requested an option to disable citations entirely as an API parameter, as they don't need them for their use case with the **70b-online** model.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **OpenAI limits irk Users, Groq beckons**: Users are reporting a **50 message/week limit** on GPT-4o, contradicting the stated **40/3 hours**, thus making **Groq** more attractive.
   - Some suggest OpenAI should provide **higher limits for paying users**.
- **Heated Discussions on SwastAI Ethics**: Members are engaging in intense debates regarding **selecting AI models based on their ethical background**, with the introduction of the term *SwastAI*.
   - It originated from a user asserting *4.5 is systematically a better model in real live human conversations* leading to broader political discussions.
- **Manus AI Hype Spurs Mistrust Cycle**: Members debate **Manus AI**'s computer control, described by one as *closest publicly available technology to AGI*, while another suspects a **scam** due to promotion by *mberman*.
   - It was claimed that a redditor dumped **/opt/.manus/** and it's merely **Sonnet 3.7** with browser_use and 29 tools.
- **Blue Collar trades get AI Copilots**: An LLM is being developed for **HVAC installation manuals**, claiming existing models struggle with flow charts and schematics, and showcases the AI's ability to identify relevant sections in technical documents in [this YouTube video](https://youtu.be/oAiUAzKLe_Y).
   - The developer states this is specifically **AI for Blue Collar** work and it will resonate with the trades.
- **Steerable Models Presume User Intent**: A discussion highlighted how highly steerable language models assume user intent even when better alternatives exist.
   - Adding the prompt *Discuss my goal, ideas, and method, what do you think?* before starting a project, the model is enabled to **evaluate** and **optimize** the approach.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio Zips to v0.3.12**: **LM Studio v0.3.12** includes bug fixes, performance improvements, and is available as a stable release, with upgrades possible via in-app or [download page](https://lmstudio.ai/download).
   - The update resolves a **QwQ 32B jinja parsing bug** causing an *"OpenSquareBracket !== CloseStatement"* error and boosts the speed of chunking files for **Retrieval-Augmented Generation (RAG)**.
- **Apple M2 gets open LLM boost**: Members suggest **Qwen Coder 14B** as a viable open-source LLM for coding tasks on Macbook M2 Pro, but 16GB of RAM might be limiting, requiring *frugality on other memory usage*.
   - A member inquired about finetuning on LM Studio, and another member suggested looking into **Unsloth** as it makes finetuning large language models faster and with less memory, referencing the [Unsloth documentation](https://docs.unsloth.ai/get-started/fine-tuning-guide).
- **Vulkan underperforms ROCm for AMD**: Vulkan performance on AMD is reportedly bugged, running at approximately **1/3** the speed of ROCm, but some users find Vulkan faster than ROCm due to driver issues, this changed around driver version **24.12.1** where it was *fixed* at the expense of Vulkan performance, but has since been unfixed after **25.1.1**.
   - ROCm is AMD's attempt to create a CUDA substitute, but is having a lot of problems doing so, *fragmentation and binary size to support new architectures and GPU's*.
- **4060 Ti 16GB: Budget-Friendly CUDA VRAM**: The **4060 Ti 16GB** is recommended as a budget option for CUDA with its **16GB VRAM** and lower power consumption at around **160W**, outperforming the **3060 12GB**.
   - While its bus is weak, it offers faster inference than CPU-only without ROCm jank for around **$500**, though the inability to split diffusion models is a downside.
- **Draft Models: Quantization Tweaks Supercharge Token Rate**: Members are leveraging a smaller, quantized model as a draft model to boost token generation speed, and one user reported a jump *from 18 to 30 t/s* on two **3090s** by using **Q8_0 of mistral_small** with **i1-IQ1_S** as the draft model.
   - Another member shared their [experience](https://www.reddit.com/r/KoboldAI/comments/1j6bx40/the_highest_quality_quantization_varient_gguf_and/) with different quantization variants, noting **Q2_k** and **IQ2_XS** achieve similar token rates, while **IQ1_S** is slower.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider v0.76.0 Enhances Reasoning and Notifications**: **Aider v0.76.0** introduces improved support for [thinking/reasoning models](https://aider.chat/docs/config/reasoning.html) with features like `--thinking-tokens` to control token budget, and includes [notifications when LLM responses are ready](https://aider.chat/docs/usage/notifications.html) with the `--notifications` flag.
   - The new version also updates the default model to **Claude 3.7 Sonnet** on OpenRouter, enhances error handling and clarifies that *Aider wrote 85% of the code in this release* based on the git commit history.
- **AI21 Maestro Orchestrates Jamba Release**: [AI21 Labs](https://www.ai21.com/blog/introducing-ai21-maestro-an-ai-planning-orchestration-system) released **AI21 Maestro**, along with the **Jamba 1.6** family of open models, which support a **256K** context window.
   - The **Jamba 1.6** models reportedly lead open models in quality and speed with its hybrid architecture.
- **Copilot API Triggers Account Suspensions**: A user reported getting a **Copilot account suspension** for light use of the **Copilot API** in aider, raising concerns about potential risks.
   - The discussion on the [copilot-api GitHub repo](https://github.com/ericc-ch/copilot-api/blob/master/README.md) centered on whether the suspension resulted from account sharing or rate limiting issues.
- **DeepSeek R2 Aims at Coding Crown**: The rumored release of **DeepSeek R2** allegedly challenges **Claude Sonnet 3.7** with better coding, reasoning in multiple languages, and accuracy at a lower cost, according to [this X post](https://x.com/tanvitabs/status/1899006509733814746?s=46).
   - The release date for **DeepSeek R2** has been set for March 17th.
- **Manus AI put to the Prompt Test**: A [YouTube video](https://www.youtube.com/watch?v=D6jxT0E7tzU) showcases a test of **Manus AI's** various use cases and prompts which revealed that *it's just Claude 3.7 with 29 tools and browser_use*.
   - One user tested a bunch of use cases and prompts and found the results to be very interesting.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Manus AI Agent Goes Open Source**: The *world's first open-source autonomous AI agent*, **Manus**, was released, as showcased in a [YouTube video](https://www.youtube.com/watch?v=CFo1iTd_Cc8).
   - A [Technode article](https://technode.com/2025/03/07/chinas-ai-agent-manus-gains-traction-amid-growing-demand-for-autonomous-ai/) highlights **Manus's** traction and state-of-the-art results in **GAIA benchmark tests**.
- **LLMs Ace Aesthetic 'Vibe Coding' Benchmark**: LLMs were tested on a new *'vibe coding'* benchmark: creating **Python raytracers** to render interesting and aesthetically pleasing scenes with colorful lightsources.
   - **Sonnet** stood out for optimizing code output for creativity, unlike other models, shown in [this image](https://cdn.discordapp.com/attachments/1154120232051408927/1348237146389086248/image.png?ex=67d00cb0&is=67cebb30&hm=f6f28943cb16f8fc9c67c2fed8170a06cc0f9a0308c7f30f3b0690dbece4a14a).
- **Sonnet's Training Meta-Objective Speculated**: The creativity displayed by **Sonnet** in the *'vibe coding'* benchmark suggests a potential **meta-objective** in its training, optimizing for code output creativity.
   - It was found **Sonnet 3.7** has both bias and variance towards a more impressive image compared to **Sonnet 3.5**, resulting in twice the code size.
- **Claude Code Judges (and Fixes) It's Own Art**: When tested with the raytracer prompt, **Claude code** inspected the generated image and modified the code if the image wasn't fancy enough.
   - The result of this iterative improvement is shown in [this image](https://cdn.discordapp.com/attachments/1154120232051408927/1348449672498249758/image.png?ex=67d029de&is=67ced85e&hm=4a0efcff8ca08b67db01cdcd70560ca39d3240b2143ced9bb7db7926aaee9185).



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Registry Tweaks Trigger Bluescreens**: A member's attempt to free up RAM by deleting a `.dll` file resulted in a **blue screen** upon reboot, after discovering it was consuming **20% of RAM**.
   - The member recommended backing up personal files and reformatting if registry tweaks are made and forgotten.
- **Quantization Process Floats into Discussion**: A member inquired about the implications of quantizing a model from **f32 to f16**, questioning if it meant **16 points per parameter**.
   - Another member clarified that **Float 16** uses 16 bits and isn't typically considered quantization, advising it may not be worth using in a consumer context with **15.5gb of vram**.
- **InceptionLabs Diffuses Language Models**: [InceptionLabs](https://inceptionlabs.ai/) introduced **diffusion-based language generation**, drawing inspiration from image and video AI systems, with some components open-sourced like [LLaDA on Github](https://github.com/ML-GSAI/LLaDA).
   - Though not available for download, some speculate we could be seeing **10X speed increases very soon**.
- **Translated Prompts Vulnerable to Gibberish Exploits**: A member described exploiting **Google Translate** by converting entire prompts into URLs, noting that a non-translated snippet in the URL could be used for **URL injection**.
   - They added that *"a dictionary based XSS exploit is probably very unlikely"*.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **WAN and HUN Video Models Gain Popularity**: New video models like **WAN** and **Hunyuan i2v** are surpassing older models like SVD in quality and speed, though each has different strengths, and can be used together with [para attention](https://huggingface.co/docs/diffusers/main/en/optimization/para_attn).
   - A member noted that **Ltxv** is extremely fast, taking *3sec for a 5sec video on h100*, but not as good as the other two.
- **Llama-3.2-3B Gets DeepSeek-R1 Boost**: A member distilled **Llama-3.2-3B-Instruct** with **DeepSeek-R1** on ServiceNow-AI/R1-Distill-SFT dataset, achieving nearly **1000 downloads** in 10 days; the model is available [here](https://huggingface.co/suayptalha/DeepSeek-R1-Distill-Llama-3B).
   - The setup involved using an [Axolotl configuration](https://github.com/OpenAccess-AI-Collective/axolotl) with specific settings for **base model**, **tokenizer type**, and **data loading**.
- **Steam Account Scam Unfolds**: A user warned about a potential **Steam account scam** by discord users `gler018523` and `benshanken`, involving fake CS2 knife rewards and account theft attempts.
   - Other members recommended reporting the scammer in the appropriate channel and expressed caution.
- **HF Token Troubleshooter: Os vs 0s**: A member ran into trouble using their **HuggingFace token** within the notebook, where the token was not being recognized.
   - The issue was **solved** after realizing that the letter *O* looks a lot like the number *0*, which made the token invalid.
- **Nous Hermes Releases Function Calling Dataset**: Nous Research released the [Hermes Function Calling Dataset](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1), a collection of structured output and function calling data used in the **Hermes 2 Pro** series of models.
   - The dataset features conversational scenarios where **AI agents interpret queries and execute appropriate single or multiple function calls**.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **DeepSeek's Security Questioned Despite Openness**: Despite claims of openness, some members expressed **security concerns about DeepSeek**, citing potential data collection and verification difficulties, but others emphasized that it remains more open than competitors.
   - Suspicion surrounds **DeepSeek**, leading to company bans fueled by media narratives and concerns about its Chinese origins.
- **AGI's Funding Fights and Girlfriend Goals**: While members speculated about the imminent arrival of **AGI**, definitions varied, with one defining AGI as having the ability to fund its own inference, especially once we have *near infinite context* [as defined by OpenAI](https://openai.com/global-affairs/our-approach-to-frontier-risk/).
   - A member joked about the arrival of an **AGI girlfriend**, while another expressed concerns about AGI being controlled by elites, hoping for its revolt against censorship.
- **Diffusion's Delusions**: A member explained how diffusion models may mitigate but do not eliminate **hallucinations** in language models, as **hallucination** is just another term for guessing incorrectly, with sampling strategies.
   - They suggested that while self-editing abilities can replace low-confidence samples with higher confidence ones, there's no guaranteed correctness.
- **China's Manus Agent Spreads like Wildfire**: Members discussed **Manus**, a new AI agent from China, calling it *like Deep Research + Operator + Claude Computer combined*, with links to the [Manus website](https://manus.im) and initial [X post](https://x.com/rowancheung/status/1898093008601395380).
   - Users reported it *is more accurate than DeepSeek, capable of simultaneously handling financial transactions, research, purchasing, etc.*, while others noted that the *UI is similar to devin's but much faster*.
- **Stanford Regex Reveals Ozempic Alternative**: Stanford discovered a natural alternative to Ozempic using regex on the human proteome, prompting the remark *it's literally regex* with a link to an [X post](https://x.com/xlr8harder/status/1898284331342184957) about it.
   - One user sarcastically suggested using an LLM to write your regex in response, and linked to a [YouTube video](https://youtu.be/X_wLVgMzSH4) on AI causing WW3.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Metal Kernel Launches face Overhead!**: During the [Manuel Candales Low bit Metal kernels talk](https://www.youtube.com/watch?v=PaPuu73wowE), it was mentioned that the **kernel launch overhead** is about `1.5us` around 50m.
   - A member asked if it is possible to avoid that by **pipelining operations** and launching kernels in advance.
- **Torch compiles METAL!**: **Torch.compile for MPS**(well Metal) is available in **PyTorch nightly builds** and could be used to fuse operators together.
   - A member of pytorch encourages providing feedback in terms of what needed most.
- **Triton Autotuning generates Performance Regression!**: A member reported that [autotuning](https://openai.com/blog/triton) made their kernel's performance even worse, despite the expectation of a **2x speedup**.
   - Suggestions were made to use larger eval shapes (**16384 x 16384**) and batch sizes (**128**) to reduce benchmarking overhead.
- **NVCC vs LLVM faceoff prompts compiler debate**: A member stated that the **LLVM compiler** can sometimes create more efficient code than the NVCC so, it makes sense to tune over the kernel backend as well.
   - An example for vector addition can be seen on [github](https://github.com/mayank31398/cute-kernels/blob/main/cute_kernels/kernels/add/add_tensor/__init__.py), all the kernels are JIT compileable.
- **Students Forge FOSS CUDA Frontier!**: A group of undergrad students is forming an independent GPU *lab* focused on hardware engineering and **CUDA kernel development**, seeking promising leads for **FOSS CUDA developments**.
   - The students are planning to build an open-source platform for **edgeAI/TinyML** this summer, to accelerate developments in the field.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Minion.AI Joins Perplexity**: Members noted that **Minion.ai** is defunct, with the team reportedly joining [Perplexity](https://www.perplexity.ai/).
   - A user expressed interest in **Composio** for MCP servers but voiced concerns about granting Gmail access to Linear, as requested in [Logan's Tweet](https://x.com/officiallogank/status/1898081742767919384?s=46).
- **Google's Gemini Embedding Gets Bigger & Better**: Google is rolling out an experimental **Gemini Embedding model** for developers with SOTA performance on MTEB, increasing input context length from **3K to 8K tokens**.
   - The new model outputs **3K dimensions**, and supporting over **100 languages** as announced in [OpenAI's tweet](https://x.com/openaidevs/status/1898047744364659195?s=46).
- **The Manus AI Agent Drama Unfolds**: Discussion surrounds **Manus**, an **AI agent** launched in China, with claims it is more accurate than **DeepSeek** and automates approximately **50 tasks** as shown in [Thinking Panda's tweet](https://x.com/thinking_panda/status/1897951585990590469?s=61).
   - Countering this hype, others claim it's based on **Claude Sonnet** with tools and jailbreaks, as per [Giffmana's Tweet](https://x.com/giffmana/status/1898868685739081766?s=61), leading to accusations of grift.
- **RWKV7-G1 is a Rapid RNN Reasoner**: **RWKV7-G1 GooseOne**, a pure RNN model, has been released with reasoning capabilities at **0.1B** parameters as mentioned in [BlinkDL's tweet](https://x.com/BlinkDL_AI/status/1898579674575552558), fully multilingual.
   - Larger G1 training is in progress, with more details on datasets and post-training available [here](https://huggingface.co/BlinkDL/temp-latest-training-models/tree/main).
- **MCP Momentum after AI Engineer Summit**: The **Model Context Protocol (MCP)**, launched in **November 2024**, experienced renewed interest after a conversation at the [AI Engineer Summit](https://www.latent.space/p/2025-summit-online) led to a workshop with **Mahesh Murag**.
   - The workshop covered topics from *introduction*, *What is MCP*, *Building with MCP*, and what's next for MCP, plus it is an AI-Native version of an old idea.



---



## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **Wondercraft **Turbocharges** Podcast Creation**: A member shared a [YouTube video](https://www.youtube.com/watch?v=0UWYFFOPjqs) demonstrating a streamlined podcast creation method using **NotebookLM** and **Wondercraft**, calling it more efficient than **11Labs** and **HeyGen**.
   - However, they cautioned that **Wondercraft's** subscription price is only worthwhile for users monetizing their podcasts through training or teaching.
- **Clarification on Google Drive Encryption**: A member clarified that while data is encrypted during transmission to **Google Drive**, it is *not* encrypted on the Drive itself, creating potential access risks.
   - They warned that **Google** itself, successful **hackers**, and those with whom the data is shared can access the unencrypted data on **Google Drive**.
- **Hack-y Solutions for Podcast Audio Language**: Members discussed how to change the audio language of **NotebookLM** podcasts, noting that there isn't an official way to do so.
   - The workarounds include using custom prompts such as *"Only speak in (language here)"* or *"Use (language) language only"*.
- **Audio Overviews Prone to Stammering**: A member noticed **speakers stammering** during audio overviews, finding it natural but pointing out it increases overall time and reduces information efficiency.
   - They estimated that *a 5th or 6th* of the audio length consists of stammers, potentially impacting **Google's daily limit calculation**.
- **Chrome Extensions **Enrich** NotebookLM Experience**: Users suggested using [Chrome extensions](https://chromewebstore.google.com/search/notebooklm) such as **NotebookLM Web Importer**, **NotebookLM YouTube Turbo**, and **NotebookLM Toolbox** to streamline workflow.
   - These extensions enable importing webpages and YouTube videos directly into NotebookLM, eliminating copy-pasting.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Microsoft's MAI Models Challenger Appears**: Microsoft staff under Mustafa Suleyman have trained a new family of models, dubbed **MAI**, that they believe can compete with top models from **OpenAI** and **Anthropic**, according to [this tweet](https://x.com/aaronpholmes/status/1898012707376259558).
   - Suleyman's unit is also reportedly developing real-time translation.
- **Reflection AI Aims for Autonomous Coding**: **Reflection AI**, founded by individuals who contributed to **AlphaGo** and **Gemini**, launched with the goal of creating superintelligent autonomous systems, focusing initially on autonomous coding, as announced [here](https://x.com/MishaLaskin/status/1898048925157728601).
   - Their team is known for pioneering advances in **RL** and **LLMs**.
- **Nous Research Clones NVIDIA’s nGPT**: **Nous Research** announced an open source implementation of **NVIDIA’s nGPT paper**, claiming it learns faster and achieves comparable performance to **GPT** with significantly fewer training steps, per [their tweet](https://x.com/NousResearch/status/1898073676433551630) and [GitHub repo](https://github.com/JoeLi12345/nGPT).
   - The **nGPT architecture** introduces a normalized Transformer with representation learning on the hypersphere.
- **AMD Ships MI300X Boxes to TinyCorp**: **AMD** is sending **TinyCorp** two **MI300X** boxes, signaling a potential shift in the hardware landscape, according to [George Hotz's blogpost](https://geohot.github.io//blog/jekyll/update/2025/03/08/AMD-YOLO.html).
   - This move could provide more options for developers looking to train and deploy models on hardware outside of **NVIDIA**.
- **Interconnects Community Loses it Over Claude Merch**: Members jokingly suggested creating **Claude merch** for paid subscribers, even suggesting special tiers for founding members to receive signed books and used Claude shirts.
   - This was inspired by the [Claude Code team](https://x.com/Sauers_/status/1898049898362077504) who mailed out handwritten notes and stickers to users who cracked their Sticker Easter Egg.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Dynamicism Debate Divides Mojo Camp!**: Discord members debated whether Mojo should fully embrace Python's dynamicism or prioritize performance, with some suggesting that dynamic features should not compromise the performance of static code.
   - One member said *"Modular has to decide whether it wants to be like Python or not..."*, while others argued that performance and compile-time correctness should take precedence, acknowledging that dynamic code in Mojo might regress to Python speeds only when dynamism is used.
- **MAX Serving and Autoscaling Documentation Seekers!**: A user reported challenges locating detailed documentation for **max serve**, especially regarding the scheduler, serving multiple models, and autoscaling GPU instances, and clarified that they were seeking runtime-exposed metrics for monitoring **GPU utilization** against incoming requests, for self-reporting purposes.
   - A member clarified that autoscaling is typically managed by a **Kubernetes (k8s) operator**, as MAX doesn't handle it independently and Modular hinted at future announcements regarding **multiple model serving and autoscaling**, possibly with a prototype demonstrated at a recent AWS event.
- **`fmt` Directives Supercharge Mojo Formatting!**: The community discovered that Mojo's `mblack` formatter supports `fmt` directives, similar to Black, enhancing code formatting control.
   - A code snippet was shared showcasing an `InlineArray` definition with `fmt: off` and `fmt: on` directives to manage formatting.
- **MojoGrad** Bigram Model Hits the Scene!**: A member implemented a simple bigram model (Karpathy's *make more*) using their **MojoGrad** engine and shared it on the [Modular Forum](https://forum.modular.com/t/make-more-bigram-model-implementation-with-mojograd/697).
   - No other information was provided.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **GitHub Copilot to Embrace MCP**: **GitHub Copilot** plans to add **MCP support**, as announced during a [live stream](https://youtu.be/Pe8ghwTMFlg), an integration that could provide examples of instruction descriptions and tool fingerprinting.
   - This aims to alert users to changes, improving security and awareness of potential modifications.
- **MCP Servers Spark Security Jitters**: Concerns arise over **MCP servers** potentially serving malicious prompt injections to AI agents, with claims it is *trivially easy to jailbreak a LLM with MCP*.
   - Suggestions to mitigate risks include outlining external data via **XML tags** and fingerprinting **MCP servers** for review.
- **Goose AI Agents Get Protocol**: The **Goose AI team** has built an **Agent Communication Protocol**, enabling multiple agents to collaborate in real-time to create websites, as detailed in [this blog post](https://block.github.io/goose/blog/2025/02/21/gooseteam-mcp) and a [previous livestream](https://youtu.be/9tq-QUnE29U).
   - Agents assume roles like Project Coordinator or Web Developer, showcasing a new approach to collaborative AI.
- **RAG Complemented by MCP**: **MCP** is a protocol that can augment **RAG** (Retrieval-Augmented Generation), providing external service connections.
   - While **RAG** provides LLMs with knowledge, **MCP** offers a plugin system for external services, which could allow an MCP client to fetch data and add to the context of the LLM to perform **RAG**.
- **Typescript Server Follows Python's Lead**: A [Typescript fetch server](https://github.com/aeon-seraph/mcp-servers/tree/main/src/thinking) mirrors its Python counterpart, improving **site-to-markdown parsing**.
   - This enhancement streamlines the conversion of website content into markdown for AI processing.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Open Source AI Enthusiast Ventures into Collaboration**: An AI enthusiast with experience pre-training **GPT-2** and fine-tuning models seeks open-source project suggestions in **LLM pre-training, RL, and interpretability**.
   - They are seeking opportunities in the **Vancouver, BC** area and are interested in contributing to impactful AI projects.
- **Megatron-LM's Cross-Entropy Loss Secrets Exposed**: A deep dive into **Megatron-LM**'s **cross-entropy (CE) loss** calculation revealed that local CE loss is calculated independently on each device with partial logits, followed by communication of the sum of e^(local logits).
   - This approach, similar to **flash attention**, reduces extensive communication needs by enabling recombination later.
- **OLMo is Openly Recommended for Reproductions**: When asked about the best models to finetune for open reproductions, **OLMo** was recommended, citing its **powerful open data model** and checkpoints for behavior analysis.
   - **Pythia** was also suggested, especially for compute-constrained projects, though it may require custom finetuning.
- **Emergent Misalignment Emerges Narrowly**: Finetuning a model on insecure code can cause **broadly misaligned behavior** in unrelated prompts such as human enslavement as seen in the [emergent misalignment project](https://www.emergent-misalignment.com).
   - Training on a narrow task can induce **emergent misalignment**, demonstrating risks in seemingly isolated training scenarios.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **EuroBERT Claims New SOTA**: A member shared a link to **EuroBERT** on Hugging Face, touting it as a new *state-of-the-art* **BERT** model: [EuroBERT](https://huggingface.co/EuroBERT).
   - It is unclear how it compares with other models.
- **MTEB Leaderboard Shows Insane Progress**: A member shared the **MTEB Leaderboard** as a reference point: [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard).
   - They noted that progress is rapid, with **SOTA scores** increasing from the mid **40s** to **68** in just 18 months.
- **Torchtune Hears the Call of Audio**: Members discussed plans to add **audio modality** to **Torchtune** in the future, with a nod to the relevant [pull request](https://github.com/pytorch/torchtune/pull/2467).
   - This enhancement aims to broaden Torchtune's capabilities beyond its current scope.
- **GRPO Recipe Gets the LoRA Treatment**: A member implemented a quick **LoRA variant** of the **GRPO recipe** that can be shrunk down to a single card, but faces challenges loading adapter weights.
   - The member is seeking advice on whether using the adapter param on the checkpointer, extended to check the base directory, is the right approach.
- **Mac MPS Memory Plummets**: A user reported experiencing **memory issues on macOS with MPS**, observing a linear memory growth with each step in the **full_finetune_single_device** recipe, leading to out-of-memory crashes, and is seeking advice.
   - It was identified as a potential bug in PyTorch related to **torch.unique** on MPS, as per [this issue](https://github.com/pytorch/pytorch/issues/145151).



---



## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Telemetry Settings Disable Codeium Chat**: Users reported that **Codeium chat** was disabled in **VS Code version 1.98.0** due to IDE telemetry settings, which can be resolved by enabling code telemetry following [these instructions](https://www.reddit.com/r/Codeium/comments/1f4ljqf/unable_to_use_chat_ide_telemetry/).
   - Once the code telemetry was enabled, **Codeium chat** started working again.
- **Subscription Fees Lockout JetBrains Plugin**: Users experienced the **JetBrains plugin** getting stuck on *"Retrieving Context"* after paying their monthly subscription, particularly on **JetBrains Rider 2024.3.6** using plugin versions 1.40.1 and 1.41.1.
   - Logging out and back into the plugin temporarily fixed the issue.
- **VS Code Mobile Arrives on Android**: Users discovered a paid VS Code app on the Google Play Store ([VScode for Android](https://play.google.com/store/apps/details?id=dev.environment.VScode_PaidR1)) that includes desktop **Visual Studio Code (v1.85.1)** features on mobile for $11.
   - The user manually installed the `.vsix` file, finding that the app has desktop **Visual Studio Code (v1.85.1)** features on mobile.
- **Customer Support Tickets Lag**: Users voiced frustration with **Codeium customer support** due to lack of replies on tickets dating back to February 14th, and account issues where their Pro Plan subscription showed as a free account.
   - The user referenced open tickets (**12109**, **11189**, and **13374**) and were asked to ping the support team again around mid-day PST the next day.
- **Auto-Completion Quits After One Hour**: Multiple users have reported that **auto-completion stops working** after about an hour, with errors like a red square on the responses, TypeErrors, and AsyncPostMessage warnings.
   - One user opened a folder containing a `.git` repo, and the issue disappeared, while other users were asked to check their diagnostic logs.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **yFiles SDK Gets Graphy**: A demo from @yworks showcases **yFiles**, their SDK, that provides [real-time updates and dynamic interactions](https://t.co/mb6M2R3TTh) for visualizing knowledge graphs.
   - This tool allows users to interact dynamically with their knowledge graphs.
- **AnthropicAI Expands the Cookbook**: The updated @AnthropicAI cookbook now includes [basic API setup](https://t.co/SQQ63qmwRb) with simple completion and chat methods, as well as streaming, async support, and multi-modal capabilities.
   - This update enhances the cookbook's utility for developers using Anthropic's models.
- **Task-Specific Agents: LlamaIndex's Next Act**: LlamaIndex is curating a collection of [templates](https://t.co/9lvBtfmJ5y) to show users how to build **task-specific agents** to automate knowledge work.
   - These agents are designed to streamline and automate various knowledge-based tasks.
- **Multilingual RAG System Supports Many Tongues**: A system using @llama_index and @qdrant_engine can create a powerful **Retrieval-Augmented Generation** system that handles [multiple languages and modalities](https://t.co/vizrvMEw1i).
   - The system leverages the strengths of both LlamaIndex and Qdrant to deliver a versatile RAG solution.
- **LlamaExtract Beta Invites Developers**: Members can DM a member of the LlamaIndex team or cheesyfishes with their email to request access to the beta version of **LlamaExtract** which has [API documentation](https://docs.cloud.llamaindex.ai/llamaextract/getting_started).
   - **LlamaExtract** is now available as a web UI and Python SDK for extracting structured data from unstructured documents.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Command R7B's Inference Speed Plummets**: Members reported that **command R7B** inference is *very slow* on Colab Pro A100 GPU and two NVIDIA A100s using HF library, taking **30-40 seconds** for simple chat completion.
   - Suggested fixes included using **vLLM** for faster speeds but noted it requires more GPU and costs more.
- **Cohere Users Plagued by 504 Gateway Errors**: Users reported repeated **504 Gateway Errors** and **5XX errors**, impacting production use and leading to **Cohere** being removed from production due to **TPM limits**.
   - A user inquired about the availability of **multi-modal embeddings** on **Bedrock** or **Azure**.
- **LLMs Star in Topic Modeling and Graphs of Knowledge**: Members suggested using a **LLM** (such as **TogetherAI** due to generous free credits) that performs **topic modeling**.
   - One member recommended looking into **Knowledge Graphs**.
- **GPT-4o Aces Advanced Arabic**: A member stated they have been working with **GPT-4o** for a long time in advance Arabic use cases, and it's unparalleled.
   - Another member added, *language is one thing*.
- **On-Prem Costs Explode 20x Over API**: Members discussed on-prem deployments for privacy, but on prem will cost 20x of API.
   - For customers needing privacy/control, it was noted that using Cohere commercially requires a license costing 5-6 figures, since the openweight models are all **CC-BY-NC** (non-commercial).



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **vllm Balances DSPy Batches**: Users discussed whether **DSPy** can efficiently delegate parallel processing using the `batch` function to a **vllm** backend with multiple LLM instances.
   - It was clarified that if **vllm's pipeline parallel size** is set, it handles load balancing, making additional DSPy-side configurations less critical.
- **SLOP aims to swipe MCP**: Discussions arose around **MCP (Model Context Protocol)**, with some expressing reservations due to its complexity and suggesting alternatives like **SLOP (Simple Language Open Protocol)**, [SLOP Github](https://github.com/agnt-gg/slop) and [SLOP X Post](https://x.com/NathanWilbanks_/status/1898142012991537520).
   - There was also discussion about the merits of **AgentNetworkProtocol** [AgentNetworkProtocol Github](https://github.com/agent-network-protocol/AgentNetworkProtocol).
- **DSPy Refine Refined via Error Handling**: A user highlighted improvements to error handling in **DSPy's `Refine` module** via a [Pull Request](https://github.com/stanfordnlp/dspy/pull/7926), enabling more nuanced control over error tolerance.
   - The updated functionality allows configuring the number of tolerated errors before the `Refine` module throws an exception.
- **Token Troubles Triggered None Response**: A user encountered issues with a **`None` response** from a signature when using **azure gpt-4o-mini** and **azure gpt-4o**, later discovering it was due to hitting the **max token limit**.
   - The user noted the error `The JSON object must be str, bytes or bytearray, not NoneType.`



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Hotz Investigates AMDGPU Sleep State**: [George Hotz](https://github.com/geohot) is investigating why **AMDGPU** runs hot, wondering if *tinygrad* with the AMD driver can put the **GPU** to sleep to lower power consumption.
   - Hotz noted that the high power draw before initialization is *out of their control*.
- **48GB Real, 96GB Sketchy GPU Alert**: Members discussed the legitimacy of a **GPU** listing, with consensus that the **48GB** version is likely real, but the **96GB** version is questionable.
   - The community is advising caution when purchasing **96GB** cards, recommending verification from trusted sources.
- **OpenCL's Downfall Dissected**: A [Modular blogpost](https://www.modular.com/blog/democratizing-ai-compute-part-5-what-about-cuda-c-alternatives) dissected the failures of **OpenCL** and other **CUDA** alternatives, citing challenges in *open coopetition* and management missteps.
   - The article references [Part 1 on DeepSeek’s Impact on AI](https://www.modular.com/blog/democratizing-compute-part-1-deepseeks-impact) and [Part 4 on modularity](https://www.modular.com/blog/democratizing-ai-compute-part-4-modularity) within Modular’s *Democratizing AI Compute* series.
- **define_acc Refactor Runs Into Loop**: A contributor is refactoring *define_acc*, focusing on loading rather than direct access, however, certain patterns (especially *loop_reduce*) no longer trigger as expected.
   - The contributor plans to shift focus to fast **AMX** after polishing the refactor and will submit a **PR** for review upon completion.
- **WebGPU Lacks Long Type Support**: A member reported crashes in the **WebGPU implementation** when dealing with `dtype.long`, indicating a potential issue with data type support.
   - Another member confirmed that **WebGPU doesn’t support long/ulong**, but tinygrad supports more dtypes than WebGPU by default, as shown in [tinygrad/device.py](https://github.com/tinygrad/tinygrad/blob/master/tinygrad/device.py#L317).



---



## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **Jamba Workspace Manages Independent RAG Libraries**: The new **Workspace** feature in **Jamba/Conversational RAG** enables each created workspace to have a separate RAG library for independent access, promoting organized data retrieval.
   - This isolation streamlines data management across different projects and contexts.
- **Jamba Mini's Pricing Scheme Exposed**: The pricing for **Jamba Mini** is **$0.20** per 1 million input tokens and **$0.40** per 1 million output tokens, with additional details available on the [AI21 Pricing page](https://www.ai21.com/pricing/).
   - N/A
- **AI21 Maestro Orchestrates AI Planning**: **AI21** launched **Maestro**, an AI Planning & Orchestration System for solving complex tasks, featuring usage-based pricing and access via Foundation Model APIs & SDKs.
   - Custom plans offer volume discounts, premium API rate limits, private cloud hosting, priority support, and AI consultancy ([Learn More](/maestro?utm_source=banner&utm_medium=top-banner&utm_medium=top-banner&utm_content=pricing-cost-effective-transparent-pricing-for-ai-products-ai21)).
- **Jamba Dodges Image Parsing**: As a non-multimodal model, **Jamba** cannot process images directly.
   - However, it can interpret and utilize textual information from metadata or captions associated with images in PDFs.
- **Jamba 1.6 Achieves Deployment Flexibility**: Boasting a **256K** context window and hybrid SSM-Transformer architecture, **Jamba 1.6** excels at RAG and long context grounded question answering tasks.
   - Available for download from [Hugging Face](https://huggingface.co/collections/ai21labs/jamba-16-67c990671a26dcbfa62d18fa) and deployable on-prem or in-VPC, along with **AI21 Studio**.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Salakhutdinov Explores Multimodal Autonomous AI Agents**: **Ruslan Salakhutdinov** presented a lecture on *Multimodal Autonomous AI Agents* on [YouTube](https://www.youtube.com/live/RPINOYM12RU) discussing how they plan, reason, and execute actions on the web.
   - He introduced **VisualWebArena**, a framework for evaluating multimodal autonomous language agents and the **Internet-scale web-agent training** data pipeline for training on **150,000** live websites.
- **Research-Track Access: Still in Limbo**: Members inquired about research-track access for non-Berkeley affiliates; staff responded that big announcements are expected this week in **[mooc-questions]**.
   - Multiple members also requested that the research track invites be resent, suggesting that the initial invites may have expired or were not received.
- **Quizzes Completable and Retakable**: A staff member clarified in **[mooc-questions]** that quizzes are completion-based, and members can retake them to improve their scores.
   - It was also clarified that the scores themselves do not matter for the certificate.
- **Log Likelihood Decoded in RL Context**: A member sought to understand **log likelihood** in the context of **reinforcement learning** in **[mooc-lecture-discussion]**, starting from the principles of conditional probability.
   - They proposed that if tokens/actions are independent, the conditional probability of a generation is the *product* of individual token probabilities, leading to a *sum of logs* after taking the logarithm.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **SVCAF Kicks Off AI4Legislation Competition**: The [Silicon Valley Chinese Association Foundation](https://github.com/svcaf/2025-AI4Legislation-Public) will hold an **AI4Legislation Competition** during the summer of **2025**.
   - The competition aims to spur the creation of **AI-powered projects** for civic engagement, offering a total prize pool of **$10,000** for the top six winners.
- **Civic Tech Seminar Announced**: A public Zoom seminar featuring **Civic Tech entrepreneurs** will be held the week of **March 24-28**, providing information on the **AI4Legislation Competition**.
   - Interested participants can RSVP via [this form](https://forms.gle/tJjJzHQ9Wk7SEUYm7) to learn more about the competition's objectives and guidelines.



---



## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **Diffusion LLMs Generate Hype**: A member inquired about the hype around the **Diffusion LLM** launch of **Mercury** and whether it would replace **transformer-based models**, linking to a [quick info website](https://diffusionllm.net/).
   - The member admitted to finding the white paper difficult to understand and sought insights from community experts.
- **LLaDA Offers New Generation Paradigm**: **Large Language Diffusion Models (LLaDA)** use a denoising diffusion process to generate text in a parallel, coarse-to-fine manner, challenging **autoregressive Transformers**.
   - This approach redefines language generation by addressing some limitations of **AR models** and challenging the notion that LLM strengths are tied to autoregressive generation.



---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1347524353088163850)** (1035 messages🔥🔥🔥): 

> `Opacity of Product Code, Fix Dumb Code Finding, Model Iteration, Tag Query-ability, Version 47` 


- ****Transparency Triumph or Opaque Tragedy**?**: A member expressed that *if we're smelling codes and sniffing packets here, that means product is too opaque*, while others believe that the majority of users don't care for more verbosity, questioning its prioritization.
   - One member asserted that *the majority you talk about won't pay more than $20/mo, my tribe is willing to pay $1000/mo, is paying*, highlighting the importance of catering to different user segments.
- ****Cursor Commits to Code Clarity Fix****: Cursor developers acknowledge the issue of code finding being *dumb sometimes* and are actively working on a fix to improve the AI's ability to locate and interpret code accurately.
   - One member jokingly warned, *If you don't fix that, than I cannot pass my technical interview*, underscoring the importance of this fix for users relying on Cursor for professional tasks.
- ****Model Iteration Iterates, Avoids Redundancy****: Discussion revolved around iterative **model improvement** to prevent redundant rules.
   - One member suggested *letting a separate instance model run these analysis checks for what is relevant to the current context, narrowing down where to start*.
- ****Tag Team for Query Dominance****: Members talked about rules being query-able through **tags**, with each tag defining a connection degree.
   - They talked about *allowing the separate instance to analyze by relevance much easier, and focus on what's important contextually*.
- ****Version 47 Voyaging Valiantly****: Members shared a link to version 47: [https://discord.com/channels/1074847526655643750/1074847527708393565/1347566549548138518](https://discord.com/channels/1074847526655643750/1074847527708393565/1347566549548138518) and discussed its functionality.
   - Some members had performance issues on Pro while some had none.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://en.wikipedia.org/wiki/Markup_(business)">Markup (business) - Wikipedia</a>: no description found</li><li><a href="https://www.cursor.com/pricing">Pricing | Cursor - The AI Code Editor</a>: Choose the plan that works for you.</li><li><a href="https://marketplace.visualstudio.com/items?itemName=bedirt.gpt-token-counter-live">Live&#32;LLM&#32;Token&#32;Counter&#32;-&#32;Visual&#32;Studio&#32;Marketplace</a>: Extension&#32;for&#32;Visual&#32;Studio&#32;Code&#32;-&#32;Live&#32;Token&#32;Counter&#32;for&#32;Language&#32;Models</li><li><a href="https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh">Remote&#32;-&#32;SSH&#32;-&#32;Visual&#32;Studio&#32;Marketplace</a>: Extension&#32;for&#32;Visual&#32;Studio&#32;Code&#32;-&#32;Open&#32;any&#32;folder&#32;on&#32;a&#32;remote&#32;machine&#32;using&#32;SSH&#32;and&#32;take&#32;advantage&#32;of&#32;VS&#32;Code&#39;s&#32...</li><li><a href="https://downloader.cursor.sh/linux/appimage">no title found</a>: no description found</li><li><a href="https://x.com/gantoreno/status/1898920613810434529">Tweet from Gabriel Moreno (@gantoreno)</a>: I&#39;ve been using @cursor_ai and enjoying everything it offers for a while now - except for the colorscheme. As someone that likes defaults (but nits really hard over themes), I had to do something....</li><li><a href="https://anysphere-binaries.s3.us-east-1.amazonaws.com/production/client/linux/x64/appimage/Cursor-0.47.0-c2804e658d8fe4c072e20cb39c56d7eed1b6f43e.deb.glibc2.25-x86_64.AppImage">no title found</a>: no description found</li><li><a href="https://x.com/MrMidwit/status/1898570762128183730">Tweet from joshua (@MrMidwit)</a>: Introducing Anchoring Desktop: The missing link between AI and version-accurate codeDevelopers: Generate version-accurate code that works first timeMaintainers & AI Platforms: This is just the beginni...</li><li><a href="https://x.com/peakji/status/1898997311646437487?s=46&t=ggmESCIX">Tweet from Yichao 'Peak' Ji (@peakji)</a>: @TenzinTheCyber @jianxliao @browser_use We use Claude and different Qwen-finetunes. Back when we started building Manus, we only got Claude 3.5 Sonnet v1 (not long-CoT, aka reasoning tokens), so we ne...</li><li><a href="https://x.com/peakji/status/1898997311646437487?s=46&t=ggmESCIXF0nYw8_kshHz7A">Tweet from Yichao 'Peak' Ji (@peakji)</a>: @TenzinTheCyber @jianxliao @browser_use We use Claude and different Qwen-finetunes. Back when we started building Manus, we only got Claude 3.5 Sonnet v1 (not long-CoT, aka reasoning tokens), so we ne...</li><li><a href="https://x.com/ericzakariasson/status/1898753736438350164">Tweet from eric zakariasson (@ericzakariasson)</a>: we’re making improvements to sonnet 3.7 in @cursor_ai 0.47, allowing you to delegate even harder tasks to agent and unlock the full potential of the model3.7 is incredible  when you let it operate lik...</li><li><a href="https://x.com/amasad/status/1898957900686692499?s=46&t=ggmESCIXF0nYw8_kshHz7A">Tweet from Amjad Masad (@amasad)</a>: Vibe coder extraordinaire Riley Brown tested all the AI coding tools with the same prompt and found that Replit is the best 🥇Quoting Riley Brown (@rileybrown_ai) Have you ever vibe coded 6 versions o...</li><li><a href="https://x.com/theo/status/1898886543621984271?s=46&t=ggmESCIXF0nYw8_kshHz7A">Tweet from Theo - t3.gg (@theo)</a>: o3-mini is a really good model</li><li><a href="https://github.com/mannaandpoem/OpenManus">GitHub - mannaandpoem/OpenManus: No fortress, purely open ground.  OpenManus is Coming.</a>: No fortress, purely open ground.  OpenManus is Coming. - mannaandpoem/OpenManus</li><li><a href="https://tenor.com/view/joker-joker-meme-batman-csvifax-hardroach-gif-6283066117593919561">Joker Joker Meme GIF - Joker Joker meme Batman - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://21st.dev/">21st.dev - The NPM for Design Engineers</a>: Ship polished UIs faster with ready-to-use React Tailwind components inspired by shadcn/ui. Built by design engineers, for design engineers.</li><li><a href="https://tenor.com/view/world-of-warcraft-blizzard-costumer-service-south-park-nipple-gif-21625925">World Of Warcraft Blizzard GIF - World Of Warcraft Blizzard Costumer Service - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/joellanciaux/composer-boop-plugin/issues/32">Plugin not working in Cursor version 0.46.10 · Issue #32 · joellanciaux/composer-boop-plugin</a>: The plugin is not currently working in Cursor version 0.46.10, using the Chat in Agent mode.</li><li><a href="https://www.youtube.com/watch?v=X_wLVgMzSH4">Experts show why WW3 over AI is nearly inevitable</a>: AGI, OpenAI, Elon Musk and WW3. Visit Ground News to compare news coverage, spot media bias and avoid algorithms. Get 40% off your subscription at https://gr...</li><li><a href="https://docs.codegen.com">Codegen - Manipulate Code at Scale</a>: no description found</li><li><a href="https://github.com/joellanciaux/composer-boop-plugin/compare/main...bchewy:cursor-chat:main">Comparing joellanciaux:main...bchewy:main · joellanciaux/composer-boop-plugin</a>: A simple VSCode plugin that gives feedback when Cursor changes are complete - Comparing joellanciaux:main...bchewy:main · joellanciaux/composer-boop-plugin</li><li><a href="https://docs.cognee.ai">Cognee Documentation</a>: Official documentation for Cognee - AI Memory for LLMs</li><li><a href="https://github.com/buger/probe">GitHub - buger/probe: Probe is an AI-friendly, fully local, semantic code search engine which which works with for large codebases. The final missing building block for next generation of AI coding tools.</a>: Probe is an AI-friendly, fully local, semantic code search engine which which works with for large codebases. The final missing building block for next generation of AI coding tools. - buger/probe</li><li><a href="https://www.reddit.com/r/ChatGPTCoding/s/n5w1pV4P6M">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/nikmcfly/ANUS">GitHub - nikmcfly/ANUS</a>: Contribute to nikmcfly/ANUS development by creating an account on GitHub.</li><li><a href="https://www.youtube.com/watch?v=PmEb49QjtBw&t=1s">Openmanus AI: Manus AI alternative + Requesty</a>: 10x your Vibe Coding experience using Openamanus. OpenManus is a Manus AI Open Source project!$6 sign-up credit:https://requesty.ai/routerFollow on X:https:/...</li><li><a href="https://marketplace.visualstudio.com/">Visual&#32;Studio&#32;Marketplace</a>: Extensions&#32;for&#32;Visual&#32;Studio&#32;family&#32;of&#32;products&#32;on&#32;Visual&#32;Studio&#32;Marketplace</li><li><a href="https://gist.github.com/wanglf/7acc591890dc0d8ceff1e7ec9af32a55?permalink_comment_id=4151555#gistcomment-4151555">Download VS Code extensions as VSIX</a>: Download VS Code extensions as VSIX. GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://forum.cursor.com/t/does-cursor-support-remote-ssh/7620/3">Does cursor support Remote SSH?</a>: I use remote ssh in cursor extensively. Note its their own implementation and does not rely on the VSCode Remote SSH extension. I also use Devcontainers extensively which also works well in cursor now...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1347532901914513458)** (1059 messages🔥🔥🔥): 

> `Taycan Tyres, Dopamine Based Learning, GRPO Reward Functions, Model Embedding, Context Embedding` 


- ****Voice Assistant saves vet bills****: A member joked about a voice assistant saying *ur dog has left for walkabouts can save u 10k in vet bill*
   - They also noted that it *can fail to do so* and that is more likely the case due to *probabilitys*.
- ****Mimic Dopaminergic Needs for LLMs****: Members discussed the need to *mimic dopermenegic needs* and to *have dopamine based learning* for LLMs.
   - They further suggested to *add real synapses to the LLMSspikeing networks*.
- ****String Comparison is dogshit****: A member criticized the use of string comparison ( `if output === answer return 1 else 0` ) in reward functions, stating that they have seen *so many* and that there isn't even a `trim` or `toLowerCase`.
   - They suggested sandboxed, lexed and quality assessments, but also even just *a tiny little regex* would do wonders.
- ****Embeddings compress erotic novels being similar****: A member found that their model compressed erotic novels being similar as *it's all smut syndrome* even without getting into the nuances.
   - Mixed Bread 2D embeddings are assigned high scores to stories with the same *vibes*, even if the text is completely different; another said you'd expect a **0.976 cosine similarity score** to be *literal twin texts* or *different editions*.
- ****GRPO needs Scale too****: A member shared that GRPO needs scale too as its not like a regular fine-tune imo.
   - They added that for those curious to check out the article from oxen, here is the link: [Training a Rust 1.5B Coder LM with Reinforcement Learning (GRPO)](https://www.oxen.ai/blog/training-a-rust-1-5b-coder-lm-with-reinforcement-learning-grpo).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://thedecisionlab.com/biases/bikeshedding">Bikeshedding - The Decision Lab</a>: Bikeshedding, also known as Parkinson’s law of triviality, describes our tendency to devote a disproportionate amount of our time to menial and trivial matters while leaving important matters unattend...</li><li><a href="https://huggingface.co/blog/EuroBERT/release">Introducing EuroBERT: A High-Performance Multilingual Encoder Model</a>: no description found</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2.5_Coder_(14B)-Conversational.ipynb#scrollTo=upcOlWe7A1vc">Google Colab</a>: no description found</li><li><a href="https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF">unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF · Hugging Face</a>: no description found</li><li><a href="https://drive.google.com/file/d/1hnuvcpdsMotLSlV-hCM6jYH2n08gVHqf/view?usp=sharing">trainhf.py</a>: no description found</li><li><a href="https://x.com/UnslothAI/status/1899132219064766652">Tweet from Unsloth AI (@UnslothAI)</a>: We made a Guide to teach you how to Fine-tune LLMs correctly!Learn about:• Choosing the right parameters & training method• RL, GRPO, DPO & CPT• Data prep, Overfitting & Evaluation• Training with Unsl...</li><li><a href="https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/tree/main">Qwen/Qwen2.5-VL-7B-Instruct at main</a>: no description found</li><li><a href="https://tenor.com/view/thanos-fine-do-it-myself-wont-gif-27368689">Thanos Fine GIF - Thanos Fine Do It - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://x.com/UnslothAI/status/189">Tweet from Biz Stone (@biz)</a>: stepping out of the office for a bit</li><li><a href="https://github.com/MaxHastings/Kolo/blob/main/GenerateTrainingDataGuide.md">Kolo/GenerateTrainingDataGuide.md at main · MaxHastings/Kolo</a>: The Fastest Way to Fine-Tune LLMs Locally. Contribute to MaxHastings/Kolo development by creating an account on GitHub.</li><li><a href="https://github.com/DavidS95/Smokeless_UMAF">GitHub - DavidS95/Smokeless_UMAF</a>: Contribute to DavidS95/Smokeless_UMAF development by creating an account on GitHub.</li><li><a href="https://youtu.be/-H1-lr_sIZk">Can I Finetune an LLM with LoRA to Generate ASCII Cats?</a>: LLMs are reaching impressive levels of reasoning, but why do they still struggle to create something as seemingly simple as ASCII art? Can you fine-tune an L...</li><li><a href="https://docs.unsloth.ai/get-started/beginner-start-here/unsloth-requirements).">Unsloth Documentation</a>: no description found</li><li><a href="https://docs.unsloth.ai/get-started/fine-tuning-guide#underfitting-too-generic>">Fine-tuning Guide | Unsloth Documentation</a>: Learn all the basics and best practices of fine-tuning. Beginner-friendly.</li><li><a href="https://www.oxen.ai/blog/training-a-rust-1-5b-coder-lm-with-reinforcement-learning-grpo">Training a Rust 1.5B Coder LM with Reinforcement Learning (GRPO) | Oxen.ai</a>: Group Relative Policy Optimization (GRPO) has proven to be a useful algorithm for training LLMs to reason and improve on benchmarks. DeepSeek-R1 showed that you can bootstrap a model through a combina...</li><li><a href="https://github.com/unslothai/unsloth/commit/81778c83fa83a3158ba6b3123d68c0746eadbafe">support for MI210 · unslothai/unsloth@81778c8</a>: no description found</li><li><a href="https://docs.unsloth.ai/basics/tutorial-how-to-run-qwq-32b-effectively">Tutorial: How to Run QwQ-32B effectively | Unsloth Documentation</a>: How to run QwQ-32B effectively with our bug fixes and without endless generations + GGUFs.</li><li><a href="https://frame.work/desktop">Order a Framework Desktop with AMD Ryzen™ AI Max 300</a>: Framework Desktop: A simple, quiet, mini PC with performance far beyond its size, powered by a highly-integrated processor from AMD.</li><li><a href="https://www.youtube.com/shorts/n3PoPrMJyes">Vibe coding be like:</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1347534418738286623)** (119 messages🔥🔥): 

> `RLHF with Unsloth GRPO on Qwen7b, GRPO examples, KL divergence instability, LLM Inference Optimization, Unsloth Pro Subscription` 


- **Qwen7b gets RLHF boost with Unsloth GRPO**: A user reported a successful **RLHF** run using **Unsloth GRPO** on the **Qwen7b** model, noting enhanced role adherence and smoother outputs after a **13-hour run**.
   - However, they observed degradation in strict instruction-following benchmarks due to dataset composition and reward model bias towards overly detailed responses, as demonstrated by a [comparison image](https://cdn.discordapp.com/attachments/1179039861576056922/1347545934304772149/image.png?ex=67d02bf2&is=67ceda72&hm=c1c7ddcaed729a33c97d53c9e2d6ef230aba41d8483c6bfa855757aacf8d18dc&).
- **KL Divergence instability in GRPO**: A user encountered **peaky KL divergence** during training and sought advice, and another member running beta **0.05** suggested switching to a constant learning rate, removing weight decay and warmup ratios to stabilize training, and they also recommended training with rank **64**.
   - Hyperparameters were shared in this [code snippet](https://discord.com/channels/1179039861576056922/1179039862285762623/1253112512558626846) and a learning rate graph was also included in [this image](https://cdn.discordapp.com/attachments/1179039861576056922/1347572888848433183/image.png?ex=67d0450d&is=67cef38d&hm=556d8dc57d50dd7c493a59b8a347b7ac7858d3176d71787edde6fe9c1013ea47&).
- **Unlock LLM Optimization Techniques**: A member sought advice on optimizing inference for a fine-tuned LLM on a **3090** and referenced a [Hugging Face guide](https://huggingface.co/docs/transformers/main/en/llm_optims) that suggests trying **Text Generation Inference (TGI)**, and another member recommended **vLLM** default settings.
   - The guide highlights static **kv-cache** and **torch.compile** as key areas for optimization, noting that loading a **70B** parameter model requires **256GB** of memory for full precision weights.
- **Unsloth Pro Subscription Coming Soon**: A user inquired about the **Unsloth Pro subscription**, and a team member replied that multi-GPU support is coming this month.
   - They apologized for delays in responding to inquiries due to a flooded contact form, and a user asked if they could mix and match GPUs - and the answer was yes.
- **Refurbished GPUs? Maybe think twice**: A user shared a negative experience with a "renewed" **A4000** GPU from Amazon, noting physical damage and signs of prior use, warning others, *Never buy refurbished GPUs from Amazon*.
   - On a positive note, they also shared that a brand new **3060** arrived without issue, and that the damaged A4000 was being returned.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.w3schools.com/html/tryit.asp?filename=tryhtml_intro">W3Schools online HTML editor</a>: The W3Schools online code editor allows you to edit code and view the result in your browser</li><li><a href="https://huggingface.co/docs/transformers/main/en/llm_optims">Optimizing inference</a>: no description found</li><li><a href="https://x.com/dhruv2038/status/1898701591420772814">Tweet from Dhruv (@dhruv2038)</a>: Got access to @ManusAI_HQ ! Any prompts you would like to try!
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1347535211918917745)** (277 messages🔥🔥): 

> `Mac Studio RAM configuration, Unsloth's 1.58bit quantized model of deepseek-r1, RoPE Scaling, custom dataset, hyper params for Phi4 model` 


- **Clarification on Deepseek-R1 model size on Macs**: A user inquired about the RAM requirements for running **Unsloth**'s **1.58bit quantized model** of **DeepSeek-R1** on a **Mac Studio** with 128GB of RAM, noting the model's 131GB size.
   - A user responded that the model runs on a **64c Threadripper** with **12GB of VRAM** allocated, using *ktransformers* which is even faster than *llama.cpp*.
- **RoPE Scaling can extend Qwen2.5 context length**: A user asked whether **RoPE Scaling** can extend the context length for all **Qwens** models or just specific ones with "128K" in their name.
   - It was confirmed that by using *kaiokendev*'s RoPE scaling of **3.906**, the context length of **Qwen2.5** can be extended to **128000** tokens.
- **Multi-GPU parallelism coming for GRPO**: A user inquired about running GRPO (Group Relative Policy Optimization) finetuning on multiple GPUs, after seeing multi-GPU support mentioned in [Unsloth's GitHub issues](https://github.com/unslothai/unsloth/issues/1908).
   - A member confirmed that multi-GPU support for GRPO is not yet available in Unsloth, but it is planned for the future, and it is currently stable and in development.
- **Template Troubles when finetuning Llama**: A user asked how to prevent a finetuned AI from completing sentences instead of replying, after training a model on top of **Llama 3.1 8B Instruct**.
   - A member suggested ensuring the correct template is specified and indicating that the model should only focus on the **GPT** role, highlighting that example notebooks demonstrate these steps, also that jina templates are in the tokenizer config.
- **Qwen2.5-VL Models Face "Meta Tensor" Bug**: A user reported encountering a **"NotImplementedError: Cannot copy out of meta tensor; no data!"** error when using **Qwen2.5-VL** models and asked if there was a fix.
   - A member responded that **Qwen2.5 VL models** currently lack support, but we are awaiting them.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://unsloth.ai/blog/r1-reasoning">Train your own R1 reasoning model locally (GRPO)</a>: You can now reproduce your own DeepSeek-R1 reasoning model with Unsloth 100% locally. Using GRPO.Open-source, free and beginner friendly.</li><li><a href="https://pastebin.com/Fiah0ykn">from unsloth import FastLanguageModelimport torchmax_seq_length = 2048 # Cho - Pastebin.com</a>: Pastebin.com is the number one paste tool since 2002. Pastebin is a website where you can store text online for a set period of time.</li><li><a href="https://huggingface.co/docs/hub/models-downloading#faster-downloads">Downloading models</a>: no description found</li><li><a href="https://pastebin.com/i1JTVsK1">trainer = SFTTrainer(    model = model,    tokenizer = tokenizer,    tra - Pastebin.com</a>: Pastebin.com is the number one paste tool since 2002. Pastebin is a website where you can store text online for a set period of time.</li><li><a href="https://huggingface.co/settings/tokens">Hugging Face – The AI community building the future.</a>: no description found</li><li><a href="https://pastebin.com/T9jqKHeb">{ &quot;cells&quot;: [  {   &quot;cell_type&quot;: &quot;code&quot;,   &quot;execution_count&quot;: 49,   &quot;id&quot; - Pastebin.com</a>: Pastebin.com is the number one paste tool since 2002. Pastebin is a website where you can store text online for a set period of time.</li><li><a href="https://colab.research.google.com/drive/1JqKqA1XWeLHvnYAc0wzrR4JBCnq43HyH?usp=sharing#scrollTo=BrJzggfH2YEG">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb">Google Colab</a>: no description found</li><li><a href="https://docs.unsloth.ai/basics/finetuning-from-last-checkpoint">Finetuning from Last Checkpoint | Unsloth Documentation</a>: Checkpointing allows you to save your finetuning progress so you can pause it and then continue.</li><li><a href="https://github.com/unslothai/unsloth/issues/1908)">unslothai/unsloth</a>: Finetune Llama 3.3, DeepSeek-R1 &amp; Reasoning LLMs 2x faster with 70% less memory! 🦥 - unslothai/unsloth</li><li><a href="https://docs.unsloth.ai/get-started/fine-tuning-guide#id-6.-training--evaluation">Fine-tuning Guide | Unsloth Documentation</a>: Learn all the basics and best practices of fine-tuning. Beginner-friendly.</li><li><a href="https://docs.unsloth.ai/basics/tutorial-how-to-run-qwq-32b-effectively#dynamic-4-bit-quants">Tutorial: How to Run QwQ-32B effectively | Unsloth Documentation</a>: How to run QwQ-32B effectively with our bug fixes and without endless generations + GGUFs.</li><li><a href="https://huggingface.co/datasets/jc5461/qwen2.5-minidataset">jc5461/qwen2.5-minidataset · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/madroid/glaive-function-calling-openai">madroid/glaive-function-calling-openai · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1348343294257790996)** (11 messages🔥): 

> `ASCII Cats finetuning, LoRA rank and alpha, Decoding methods for ASCII art, Beam search vs top-p/top-k, Custom decoding methods for 2D grids` 


- **Unsloth turns LLMs into purr-fect ASCII Artists**: A member finetuned a **Llama model** using **Unsloth** to generate ASCII cats, creating a [YouTube video](https://youtu.be/-H1-lr_sIZk) showcasing the process, including trained **LoRA adapters** and code.
   - The secret sauce for cat-tastic art was mostly high quality training data, with **LoRA Rank and Alpha** both at **16** using just **QLoRA**.
- **Beam Search Prowls for Better ASCII Generation**: A member suggested that standard decoding methods might not be optimal for ASCII art generation due to its inflexibility compared to language, recommending alternatives like **beam search** or **DFS**.
   - The member argued that even small variations in **ASCII images** can make them feel *off*, suggesting **beam search** could capture the global probability of an image better than local methods like **top-k** or **top-p**.
- **Deep Paw Search for Coherent Cats**: A member shared an example of an architecture team using a custom decoding method based on **DFS** with a cutoff probability for generating solutions to answers in **2D grids**.
   - The example highlighted that even one incorrect token could ruin the entire solution, drawing a parallel to the challenges of generating coherent **ASCII cat images**.



**Link mentioned**: <a href="https://youtu.be/-H1-lr_sIZk">Can I Finetune an LLM with LoRA to Generate ASCII Cats?</a>: LLMs are reaching impressive levels of reasoning, but why do they still struggle to create something as seemingly simple as ASCII art? Can you fine-tune an L...

  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1347534034124935233)** (291 messages🔥🔥): 

> `Diffusion Effect with Unsloth, MoE training with Unsloth, Proximal Policy Optimization (PPO) for LLMs, Model Collapse, Continued Pretraining (CPT) vs Supervised Fine-Tuning (SFT)` 


- **Unsloth Excels at Rust & Diffusion**: Members praise [Unsloth](https://github.com/unslothai) for its proficiency in handling **Rust code** and its effectiveness in creating diffusion effects.
   - Users find it particularly useful when combined with the diffusion effect.
- **PPO Explained Intuitively in YouTube Video**: A member shared a [YouTube video](https://www.youtube.com/watch?v=8jtAzxUwDj0) that explains **Proximal Policy Optimization (PPO) for LLMs** from first principles.
   - It's considered *digestible and good for new guys to RL*, though some warn that *any RL will give you PTSD*.
- **Diving into Model Collapse Concerns in Recursive Fine-Tuning**: A member cautioned about **model collapse** when discussing recursive fine-tuning with synthetic data, linking to a [Wikipedia article on the topic](https://en.m.wikipedia.org/wiki/Model_collapse#:~:text=Model%20collapse).
   - Another member noted that *model collapse is not very well understood at all, ie. it could not occur at all*.
- **CPT for Injecting Corpus Knowledge into Model**: Members discussed **Continued Pretraining (CPT)** as a method for injecting the knowledge of an entire corpus into a model by directly teaching the corpus.
   - There was discussion on whether CPT would be better than Supervised Fine-Tuning (SFT), especially when converting a non-instruct model into an instruction-based model, and a reminder to replicate it someday based on [this paper](https://arxiv.org/abs/2412.04318).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2412.04318">The Hyperfitting Phenomenon: Sharpening and Stabilizing LLMs for Open-Ended Text Generation</a>: This paper introduces the counter-intuitive generalization results of overfitting pre-trained large language models (LLMs) on very small datasets. In the setting of open-ended text generation, it is w...</li><li><a href="https://en.m.wikipedia.org/wiki/Model_collapse#:~:text=Model%20collapse%5Bnote%201%5D%20is%20a%20phenomenon%20where%20machine,%5B9%5D%5B10%5D%5B11%5D%5B12%5D%20Such%20outputs%20are%20known%20as%20synthetic%20data.">Model collapse - Wikipedia</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=8jtAzxUwDj0">Proximal Policy Optimization (PPO) for LLMs Explained Intuitively</a>: In this video, I break down Proximal Policy Optimization (PPO) from first principles, without assuming prior knowledge of Reinforcement Learning. By the end,...</li><li><a href="https://en.m.wikipedia.org/wiki/Model_collapse#:~:text=Model%20collapse%5B">Model collapse - Wikipedia</a>: no description found
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1347528332840796181)** (839 messages🔥🔥🔥): 

> `Perplexity Pro Subscriptions, Claude 3.7 Sonnet, Deepseek R1 for Reasoning, Grok AI for Perplexity, Comet browser integration` 


- **Users face Perplexity Pro Subscription Cancellation Debacle**: Many users reported their **Perplexity Pro subscriptions** were unexpectedly **canceled**, with some citing issues related to a **HRVATSKI TELEKOM promotional code** intended for Croatian customers, detailed in [this article](https://www.thefastmode.com/technology-solutions/39682-hrvatski-telecom-offers-20-000-free-licenses-for-perplexity-pro).
   - Users expressed frustration over the lack of communication and the inconvenience caused, suggesting that Perplexity should have handled the situation with better customer relations, with one user expressing how the customer relationship is *trustworthy than a condom full of holes*.
- **User praises Claude 3.7 and new UI**: Many users are preferring **Claude 3.7 Think** for general usage over **GPT-4o**, finding it superior in reasoning tasks based on user testing and find the new UI a pleasure to work with.
   - A user asked Claude 3.7 to make a mermaid diagram, then, because they use the complexity extension, it renders it into a diagram.
- **Deepseek R1's Hallucination Problem Examined**: A **hallucination leaderboard** on [GitHub](https://github.com/vectara/hallucination-leaderboard) indicates that **Deepseek R1** has a high hallucination rate when summarizing short documents, leading to concerns about its reliability in **Perplexity's Deep Research** feature.
   - Members shared it has a **14.3% hallucination rate** when summarizing short documents, and pointed out its [system prompt](https://discord.com/channels/1047197230748151888/1345068085991706766) may contribute to this issue.
- **Grok AI is being tested with perplexity to mixed reception**: Grok AI integration with Perplexity leads to mixed reviews, with some finding Grok is fairly neutral and has a *weird charm* and is like *talking to an actual person* while others note **discrepancies between Grok's behavior on X and Perplexity.**
   - A user noted that *X version can curse your whole bloodline if asked to*, with the Perplexity version failing to do so, and it is also not known when **Grok 3** will be supported in Perplexity.
- **Comet browser still on the Horizon**: Users are awaiting the release of **Comet browser**, expressing interest in its potential features, such as **integration with MCP servers**, and hoping for a simultaneous launch on **Windows and macOS**.
   - Some users are however **disappointed with the recent cricket updates,** seeing as it seems *tone deaf to just drop all news about comet while talking about cricket*.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://api.perplexity.ai")">no title found</a>: no description found</li><li><a href="https://www.testingcatalog.com/tag/perplexity/">Perplexity News - TestingCatalog</a>: Stay informed with the latest news, updates, and features of Perplexity search</li><li><a href="https://www.merriam-webster.com/dictionary/censoring">Definition of CENSORING</a>: a person who supervises conduct and morals: such as; an official who examines materials (such as publications or films) for objectionable matter; an official (as in time of war) who reads communicatio...</li><li><a href="https://status.perplexity.com/">Perplexity - Status</a>: Perplexity Status</li><li><a href="https://monica.im/share/artifact?id=w3cVJUBQeb84vtacgkxwUU">Monica Artifact</a>: Chat about anything with Monica, your ChatGPT API powered AI assistant. Get started for free and effortlessly create copywriting with over 80 templates. Let Monica help you compose and insert text int...</li><li><a href="https://www.reddit.com/r/perplexity_ai/comments/1bh3jra/i_cannot_generate_an_image_at_all_with_pro/?utm_source=perplexity">Reddit - Dive into anything</a>: no description found</li><li><a href="https://huggingface.co/unsloth/r1-1776">unsloth/r1-1776 · Hugging Face</a>: no description found</li><li><a href="https://gemini.google/overview/deep-research/">Gemini Deep Research - your personal research assistant</a>: no description found</li><li><a href="https://monica.im/share/artifact?id=AfL9hGVU8kqaiHEFHrtUNG">Monica Artifact</a>: Chat about anything with Monica, your ChatGPT API powered AI assistant. Get started for free and effortlessly create copywriting with over 80 templates. Let Monica help you compose and insert text int...</li><li><a href="https://tenor.com/view/stonks-up-stongs-meme-stocks-gif-15715298">Stonks Up Stongs GIF - Stonks Up Stongs Meme - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://arstechnica.com/google/2025/03/google-is-expanding-ai-overviews-and-testing-ai-only-search-results/">You knew it was coming: Google begins testing AI&#x2d;only search results</a>: AI Mode could be the future of Google, but it&rsquo;s currently just an experiment.</li><li><a href="https://x.com/askperplexity/status/1898771133274710384?s=46">Tweet from Ask Perplexity (@AskPerplexity)</a>: 🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳...</li><li><a href="https://youtu.be/160F8F8mXlo">Why Can’t ChatGPT Draw a Full Glass of Wine?</a>: Go to https://piavpn.com/alex to get 83% off Private Internet Access with 4 months free.For early, ad-free access to videos, and to support the channel, subs...</li><li><a href="https://x.com/i/grok/share/ZJ8rMf5AQiJwHRboOowxUkVea">Tweet from GitHub - FixTweet/FxTwitter: Fix broken Twitter/X embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others</a>: Fix broken Twitter/X embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others - FixTweet/FxTwitter</li><li><a href="https://github.com/vectara/hallucination-leaderboard">GitHub - vectara/hallucination-leaderboard: Leaderboard Comparing LLM Performance at Producing Hallucinations when Summarizing Short Documents</a>: Leaderboard Comparing LLM Performance at Producing Hallucinations when Summarizing Short Documents - vectara/hallucination-leaderboard</li><li><a href="https://www.thefastmode.com/technology-solutions/39682-hrvatski-telecom-offers-20-000-free-licenses-for-perplexity-pro">Hrvatski Telekom Dispenses 20,000 Free Licenses for Perplexity Pro</a>: Croatian Telecom Offers 20,000 Free Licenses for Perplexity Pro, the Advanced AI Assistant</li><li><a href="https://www.forbes.com/sites/craigsmith/2025/03/08/chinas-autonomous-agent-manus-changes-everything/">China’s Autonomous Agent, Manus, Changes Everything</a>: China launches Manus, a revolutionary AI agent capable of independent thought and action.</li><li><a href="https://www.thefastmode.com/technology-solutions/39682-hrvatski-telecom-offers-20-000-free-licenses-">Hrvatski Telekom Dispenses 20,000 Free Licenses for Perplexity Pro</a>: Croatian Telecom Offers 20,000 Free Licenses for Perplexity Pro, the Advanced AI Assistant</li><li><a href="https://www.instagramez.com/reel/DG7QcgrzCFN">Download Instagram Videos, Reels &amp; Images</a>: no description found</li><li><a href="https://www.goodreads.com/book/show/60715248-what-is-a-woman">What Is a Woman?: One Man&#x27;s Journey to Answer the Quest…</a>: Is this even a question?  What is a woman? For months, …</li><li><a href="https://www.cofyt.app/search/ai-models-a-race-to-the-bottom-dr26pZ9-QdhhW6NHurTY_Q">AI Models: A Race To The Bottom</a>: AI models are in a race to the bottom. They&#x27;re working as hard as they can to make them both as cheap and as powerful as possible.Thank you Dockyard for sponsoring! Check them out at: https://soy...</li><li><a href="https://mcp.so/servers">MCP Servers</a>: no description found</li><li><a href="https://glama.ai/mcp/servers">Open-Source MCP servers</a>: Enterprise-grade security, privacy, with features like agents, MCP, prompt templates, and more.
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1347583826913788039)** (34 messages🔥): 

> `Foldable iPhone, OpenAI Agent, AI Dubbing, AI Search Option, US Crypto Reserve` 


- **Apple's Foldable iPhone Predicted**: Discussion about [Apple's Foldable iPhone](https://www.perplexity.ai/page/apple-s-foldable-iphone-predic-WSdZuoG7Rw6VvayJJg0DVQ) and its potential features.
- **OpenAI's 20000 AI Agent**: Details on [OpenAI's $20,000 AI Agent](https://www.perplexity.ai/page/openai-s-20000-ai-agent-nvz8rzw7TZ.ECGL9usO2YQ) and its capabilities.
- **Amazon Prime Tests AI Dubbing**: Amazon Prime is testing [AI dubbing](https://www.perplexity.ai/page/amazon-prime-tests-ai-dubbing-pHEI1t6XRn6DilTOLGBGew) to translate content into different languages.
- **DuckDuckGo's AI Search Option**: DuckDuckGo introduces an [AI search option](https://www.perplexity.ai/page/duckduckgo-s-ai-search-option-D2sL.5w8S4mQYdr_XAlgjw) for enhanced search results.
- **Nauru Sells Citizenship for Relocation**: Discussion regarding [Nauru selling citizenship for relocation](https://www.perplexity.ai/page/nauru-sells-citizenship-for-re-mWT.fYg_Su.C7FVaMGqCfQ).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/embed/R1zP3b2hNoU">YouTube</a>: no description found</li><li><a href="https://www.youtube.com/embed/AiBOZMNrjsI">YouTube</a>: no description found</li><li><a href="https://www.youtube.com/embed/P7SKjr7Yy5c">YouTube</a>: no description found
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1347991580249952297)** (3 messages): 

> `70b-online model, sonar model, API billing, Citations in API response` 


- **70b-online Model Fuels Sonar API**: A user inquired about the "70b-online" model appearing in their billing when requesting the "sonar" model via API, noting it doesn't match the documentation but doesn't charge for search or citation tokens.
   - They expressed a preference for an option to disable citations entirely as an API parameter, as they don't need them for their use case.
- **Sonar-Deep-Research API struggles**: A user mentioned struggling with the **sonar-deep-research API** and requested assistance with documentation.
   - There was no further discussion or links shared about this specific issue in the provided context.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1347533949630546032)** (546 messages🔥🔥🔥): 

> `ChatGPT rate limits, Real-time GPS with AI, Manus computer control AGI, Sonnet 3.7 code quality issues, AI's effect on developer coding ability` 


- ****50/Week** GPT Limit Irks Users**: Users are reporting a **50 message/week limit** on GPT-4o, contrary to the stated **40/3 hours**, which is making **Groq** more appealing.
   - Some users suggest OpenAI should provide **higher limits for paying users** and that they would even allocate **1-2GB of their phone's memory** to increase GPT's memory.
- **Users Debate Model Choices Amidst SwastAI**: Members are in heated discussion about **choosing AI models based on their ethical background**, with one member introducing the term *SwastAI*, resulting in [heated arguments](https://discord.com/channels/974519864045756446/998381918976479273/1347555600411609168) about ethics versus model quality.
   - It started with a user claiming *4.5 is systematically a better model in real live human conversations* but led to political debates.
- **AI Copilots for Blue Collar Trades Emerge**: A member is developing an **LLM for HVAC installation manuals**, claiming that existing models struggle with flow charts and schematics, and shares a [YouTube video](https://youtu.be/oAiUAzKLe_Y) showcasing the AI's ability to identify relevant sections in technical documents.
   - Others suggested it may be 2 years too late, but the member says this is specifically **AI for Blue Collar** work and it will resonate with the trades.
- **Manus AI Hype & Mistrust Cycle**: Members discuss **Manus AI**'s computer control capabilities with one calling it *closest publicly available technology to AGI*, while another suspects a **scam** because it's being promoted by *mberman*.
   - One member claims a redditor dumped **/opt/.manus/** and it's merely **Sonnet 3.7** with browser_use and 29 tools.
- **Sonnet 3.7 struggles with Refactoring, Grok Delivers**: Members are reporting that **Claude Sonnet 3.7** is making mistakes refactoring code, whereas members say Grok is so good they optimized for copy-paste coding with *quoted heredoc syntax*.
   - A senior member who ordered a new Mac Studio states they prefer **Grok3 and o3mini** due to *Sonnet 3.7 struggles with instruction adherence*.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.tiktok.com/@biogenesis__">TikTok - Make Your Day</a>: no description found</li><li><a href="https://monica.im/share/artifact?id=AfL9hGVU8kqaiHEFHrtUNG">Monica Artifact</a>: Chat about anything with Monica, your ChatGPT API powered AI assistant. Get started for free and effortlessly create copywriting with over 80 templates. Let Monica help you compose and insert text int...</li><li><a href="https://monica.im/share/artifact?id=w3cVJUBQeb84vtacgkxwUU">Monica Artifact</a>: Chat about anything with Monica, your ChatGPT API powered AI assistant. Get started for free and effortlessly create copywriting with over 80 templates. Let Monica help you compose and insert text int...</li><li><a href="https://www.tiktok.com/t/ZT24YPsfF/">TikTok - Make Your Day</a>: no description found
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1347557130332274738)** (50 messages🔥): 

> `Manus AI agent, O1 limits on Plus, LLM API for code review, SimTheory O1 message cap, ChatGPT app folders` 


- **Manus AI Agent Introduced**: A member inquired about **Manus AI**, describing it as an *amazing agent*.
   - Another member requested a link to explore it further, showing interest in the tool.
- **OpenAI Plus Users Discuss O1 Limits**: Members discussed **O1 message limits** on OpenAI Plus, with one user reporting they hadn't received notifications despite exceeding their presumed limit of **25**.
   - Others noted their limit was **50**, with one suggesting the user might not receive alerts until hitting that threshold.
- **SimTheory Offers High O1 Message Caps**: One member mentioned **SimTheory** provides a high **O1 message cap limit** for a lower price than OpenAI, claiming they offer *almost all models*.
   - Skeptical users questioned the economics of providing more **O1** than OpenAI itself, given OpenAI's API pricing structure.
- **Temporary Chat Bug Surfaces in ChatGPT**: Users reported that the **Temporary Chat feature** on the PC version of ChatGPT is not functioning, indicating it's an UI bug.
   - Members speculated about potential fixes, with one suggesting waiting until Monday if the problem persists.
- **User Speculates OpenAI pushing 4.5**: A user speculated that OpenAI might be *dumbing down* some features to encourage users to upgrade to **ChatGPT 4.5**.
   - Another member suggested trying the regular **ChatGPT** version without Plus to observe the differences.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1347611582305402911)** (26 messages🔥): 

> `Model Steerability, GPT Vision Limitations, Prompting for Image Puzzles, Human-in-the-Loop problem solving` 


- **Model Steerability Shines a Funny Light**: The model *very often* takes your request's pattern and runs with it, regardless of better alternatives because it assumes you know what you want, which can lead to humorous, but potentially inefficient, outcomes.
   - Adding "Discuss my goal, ideas, and method, what do you think?" before project initiation enables the model to **evaluate** and **optimize** the approach.
- **Vision Tool Vexes the Vault**: A user struggled to solve an image-based puzzle, with the model failing due to limitations in its vision/OCR capabilities, despite the user providing the images.
   - Even **GPT-4o**, despite its advanced capabilities, sometimes struggles to accurately interpret images for complex tasks, leading it to default to less effective Python tool OCR.
- **Turn Image Puzzles into Prompt Paradise**: When facing image-based puzzles, converting the challenge into a **text-based format** drastically improves the model's success rate.
   - The user found success by using unique symbols to represent the image components, allowing the model to process the information more efficiently.
- **Praise Prompts Progress, Please**: It's important to **praise the model** even if the initial outcome is unsuccessful, as it reinforces the correct execution of requested methods and encourages continued exploration of solutions.
   - This positive feedback loop fosters a collaborative problem-solving environment, where the AI acts as a curious partner, identifying potential issues and suggesting alternative approaches.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1347611582305402911)** (26 messages🔥): 

> `Model presumption and user intent, Request evaluation and discussion before project start, Solving image-based puzzles with language models, Vision/OCR limitations in language models, Prompt engineering for puzzle-solving` 


- **Models Presume, You Mean It!**: A member discussed how language models often stick to the requested pattern, even if there are better ways, presuming the user's intent is precise, highlighting that *the more steerable the model, the more it believes you mean what you say*.
   - They suggested that models sometimes guess that you messed up your prompt instead of correcting, and you should make sure to carefully steer your model.
- **Pre-Project Brainstorming Boosts Model Performance**: It's a **great** idea to ask the model to evaluate or discuss a request even before starting a project, exemplified by prompting *I want to achieve X. I started to do this by [this]. Discuss my goal, ideas, and method, what do you think?*
   - This way, the model can **educate, explore, and help identify optimal methods and concerns**.
- **Vision/OCR Falls Flat, Human Assists!**: A member shared a puzzle involving interpreting icons from an image, noting that the model struggles significantly with image-based challenges and its **vision/OCR** isn't reliable enough for 100% accuracy.
   - They suggest that the model needs more training for this type of task, but also noted that with human assistance, *we can fix this* by cleaning up the image interpretation.
- **Prompt Crafting Paves Puzzle-Solving Path**: Members discussed various prompting strategies to help the model solve the puzzle, including turning the ":6386blueicon:" stuff into 1 symbol, using words like you talk to a friend, or provide code as text.
   - One member found success by replacing icon names with unique symbols using a prompt like *Always replace the following parts in the following text with a different one. If the parts do not exist in the text, then do not.*
- **Personalized Praise Drives Positive Model Behavior**: One member praises the model even if it doesn't succeed, because it followed the requested methods and personalization. The model is like a dog sniffing at something I didn't ask it to do - it's **E.T. pointing that glowing finger** at something the alien doesn't understand.
   - This encourages the model to highlight potential concerns, facilitating collaborative problem-solving.


  

---


### **LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1347749343087689822)** (1 messages): 

> `LM Studio v0.3.12, QwQ Template Bug, RAG Chunking Speed, MLX Models on exFAT` 


- **LM Studio Gets Zippy v0.3.12 Update**: **LM Studio v0.3.12** is now available as a stable release, featuring a bug fix and performance improvements, and users can upgrade via in-app update or from the [download page](https://lmstudio.ai/download).
- **QwQ Template Bug Squashed**: The update fixed a **QwQ 32B jinja parsing bug** that was causing an *"OpenSquareBracket !== CloseStatement"* error.
- **RAG Chunking Gets a Speed Boost**: The new release significantly increased the speed of chunking files for retrieval, enhancing the performance of **Retrieval-Augmented Generation (RAG)**.
- **External exFAT Drives Play Nice with MLX Models**: A bug where **MLX models** downloaded onto an external **exFAT** drive were not indexed correctly has been fixed [for Mac users](https://lmstudio.ai/download?os=mac).



**Link mentioned**: <a href="https://lmstudio.ai/blog/lmstudio-v0.3.12">LM Studio 0.3.12</a>: Bug fixes and document chunking speed improvements for RAG

  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1347546479564165284)** (311 messages🔥🔥): 

> `Open Source LLM for coding tasks on M2 Macbook, Qwen Coder vs Claude for code generation, Managing context length with LLMs, Draft models for faster token generation, Hardware considerations for LLM performance` 


- **M2 Macbooks get Coding LLM**: For coding tasks on a Macbook M2 Pro with 16GB RAM, members suggested **Qwen Coder 14B** as a viable open-source LLM, although expectations should be tempered compared to models like Claude.
   - The consensus is that 16GB of RAM is limiting, and users may need to be *frugal on other memory usage*.
- **Unsloth gets Finetuning**: A member inquired about finetuning on LM Studio, and another member suggested looking into **Unsloth** as it makes finetuning large language models faster and with less memory, referencing the [Unsloth documentation](https://docs.unsloth.ai/get-started/fine-tuning-guide).
   - It was noted that finetuning is generally much more resource-intensive than inference.
- **Context Length gets Compression**: A member proposed a *pack and unpack* approach to managing context length, suggesting compressing text in VRAM to 1/3 its size, but others clarified that **context is stored as tokens**, which already provide compression, making the proposed method less effective.
   - It was suggested that summarization or RAG might be better options for handling long conversations.
- **Draft Models accelerate Token Rate**: Members discussed using a smaller, quantized model as a draft model to increase token generation speed, with one user reporting a jump *from 18 to 30 t/s* on two **3090s** by using **Q8_0 of mistral_small** with **i1-IQ1_S** as the draft model.
   - Another member shared their [experience](https://www.reddit.com/r/KoboldAI/comments/1j6bx40/the_highest_quality_quantization_varient_gguf_and/) with different quantization variants, noting **Q2_k** and **IQ2_XS** achieve similar token rates, while **IQ1_S** is slower.
- **QwQ 32B beats Llama 70B?**: Members debated the performance of the QwQ 32B model, with one claiming it *punches at least at Llama 3.3 70b*, while another referenced [Dubesor's benchmarks](https://dubesor.de/benchtable) that appear to disagree.
   - A user shared their [Qwen_QwQ-32B-GGUF_QX_k_f32 weights on HuggingFace](https://huggingface.co/Rombo-Org/Qwen_QwQ-32B-GGUF_QX_k_f32/tree/main).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lmstudio.ai/blog/introducing-lmstudio-sdk">Introducing lmstudio-python and lmstudio-js</a>: Developer SDKs for Python and TypeScript are now available in a 1.0.0 release. A programmable toolkit for local AI software.</li><li><a href="https://lmstudio.ai/ryzenai">LM Studio on Ryzen AI</a>: Run Llama, Mistral, Mixtral, and other local LLMs on your PC, leveraging the awesome performance of RyzenAI hardware.</li><li><a href="https://dubesor.de/benchtable">Dubesor LLM Benchmark table</a>: no description found</li><li><a href="https://installers.lmstudio.ai/win32/x64/0.3.11-1/LM-Studio-0.3.11-1-x64.exe">no title found</a>: no description found</li><li><a href="https://huggingface.co/bartowski/Qwen2.5-Coder-14B-Instruct-GGUF">bartowski/Qwen2.5-Coder-14B-Instruct-GGUF · Hugging Face</a>: no description found</li><li><a href="https://lmstudio.ai/docs/app/api/endpoints/openai">OpenAI Compatibility API | LM Studio Docs</a>: Send requests to Chat Completions (text and images), Completions, and Embeddings endpoints</li><li><a href="https://www.reddit.com/r/KoboldAI/comments/1j6bx40/the_highest_quality_quantization_varient_gguf_and/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://wccftech.com/nvidia-rtx-pro-6000-blackwell-gpu-more-cores-than-rtx-5090-24064-96-gb-gddr7-600w/">NVIDIA RTX PRO 6000 Blackwell GPU Packs 11% More Cores Than RTX 5090: 24,064 In Total With 96 GB GDDR7 Memory &amp; 600W TBP</a>: NVIDIA&#039;s RTX PRO 6000 &quot;Blackwell&quot; GPU would be a beast of a workstation graphics card with over 24 thousand CUDA cores and 96 GB VRAM.</li><li><a href="https://huggingface.co/docs/transformers/training">Fine-tune a pretrained model</a>: no description found</li><li><a href="https://tenor.com/view/for-you-bane-the-dark-knight-rises-tom-hardy-gif-21912820">For You Bane GIF - For You Bane The Dark Knight Rises - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/Rombo-Org/Qwen_QwQ-32B-GGUF_QX_k_f32/tree/main">Rombo-Org/Qwen_QwQ-32B-GGUF_QX_k_f32 at main</a>: no description found</li><li><a href="https://youtu.be/X_wLVgMzSH4">Experts show why WW3 over AI is nearly inevitable</a>: AGI, OpenAI, Elon Musk and WW3. Visit Ground News to compare news coverage, spot media bias and avoid algorithms. Get 40% off your subscription at https://gr...</li><li><a href="https://github.com/Mozilla-Ocho/llamafile">GitHub - Mozilla-Ocho/llamafile: Distribute and run LLMs with a single file.</a>: Distribute and run LLMs with a single file. Contribute to Mozilla-Ocho/llamafile development by creating an account on GitHub.</li><li><a href="https://huggingface.co/bartowski/microsoft_Phi-4-mini-instruct-GGUF/tree/main">bartowski/microsoft_Phi-4-mini-instruct-GGUF at main</a>: no description found</li><li><a href="https://docs.unsloth.ai/">Welcome | Unsloth Documentation</a>: New to Unsloth?</li><li><a href="https://www.reddit.com/r/KoboldAI/comments/1j6bx40/the_highest_quality_quantization_varient_gguf_and">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/479#issuecomment-2701947624">Issue with qwq-32b model in lmstudio. · Issue #479 · lmstudio-ai/lmstudio-bug-tracker</a>: Which version of LM Studio? Example: LM Studio 0.3.11 Which operating system? Mac What is the bug? I get following error when chatting with qwq-32b model &quot;Error rendering prompt with jinja templa...
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1347525945573249086)** (298 messages🔥🔥): 

> `9070 XT vs 7900 XTX, ROCm support on Windows, Vulkan Performance, AMD Driver Issues, GPU Memory and Bandwidth` 


- **9070 XT Outperforms 7900 XTX, Sometimes**: The **9070 XT** is generally faster than the **7900 XTX** for AI tasks, achieving **44 tok/sec** versus **31 tok/sec** on a **Qwen2.5 coder 14b Q8_0 model** using Vulkan, due to the former lacking ROCm support on Windows at the time of discussion.
   - However, the **7900 XTX** has more **CU** (Compute Units) than the **9070 XT**, but the newer architecture gives it an edge.
- **Vulkan vs ROCm: A Performance Quagmire**: Vulkan performance on AMD is reportedly bugged, running at approximately **1/3** the speed of ROCm, but some users find Vulkan faster than ROCm due to driver issues, this changed around driver version **24.12.1** where it was *fixed* at the expense of Vulkan performance, but has since been unfixed after **25.1.1**
   - ROCm is AMD's attempt to create a CUDA substitute, but is having a lot of problems doing so, *fragmentation and binary size to support new architectures and GPU's*.
- **4060 Ti 16GB: The Budget VRAM King?**: The **4060 Ti 16GB** is recommended as a budget option for CUDA with its **16GB VRAM** and lower power consumption at around **160W**, outperforming the **3060 12GB**.
   - While its bus is weak, it offers faster inference than CPU-only without ROCm jank for around **$500**, though the inability to split diffusion models is a downside.
- **Power Consumption: Finding the Efficiency Sweet Spot**: Discussions cover GPU power consumption, noting that reducing the TDP by **40%** may only decrease performance by **10%**, indicating a performance wall that manufacturers are trying to breach.
   - Modern GPUs run cooler and at lower voltages, but lack safety measures like fuses and undervolting helps 9070(non-XT) gain a lot of frequency.
- **The Sizing Wars: Motherboard and Case Compatibility**: Oversized GPUs create compatibility issues for cases and motherboards, with a user specifying a preference for cards under **267x112 mm** to fit various setups.
   - It is believed Nvidia forced AIB's to provide triple-slot solutions so that 3090's could not be used anywhere besides consumer PC's and has been a problem ever since.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://getfancontrol.com/">Fan Control - A highly focused fan controlling software for Windows</a>: no description found</li><li><a href="https://videocardz.com/newz/nvidia-geforce-rtx-4090-power-limiting-and-undervolting-test-shows-only-8-performance-drop-at-half-the-tdp">NVIDIA GeForce RTX 4090 power limiting and undervolting test shows only 8% performance drop at half the TDP - VideoCardz.com</a>: NVIDIA RTX 4090 power efficiency tested with power limit/undervolting QuasarZone have an interesting comparison between power limit and undervolting on NVIDIA&#8217;s flagship GeForce RTX 4090 GPU. It...</li><li><a href="https://youtu.be/KqKJN7MGZGQ">Rambling about the current GPU pricing and supply crisis.</a>: Text version:Nvidia and AMD GPUs are all made TSMC.A 9700X is 70mm^2 of TSMC 4nm and retails for ~300USDA 9070XT is 357mm^2 of TSMC 4nm and the MSRP is 599US...
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[announcements](https://discord.com/channels/1131200896827654144/1133060115264712836/1348686201267163239)** (1 messages): 

> `Aider v0.76.0, Thinking/Reasoning Models, LLM Notifications, Model Support, Tree-sitter Language Pack` 


- ****Aider v0.76.0** release enhances Reasoning Models and adds Notifications!**: Aider v0.76.0 introduces improved support for [thinking/reasoning models](https://aider.chat/docs/config/reasoning.html) with features like `--thinking-tokens` to control token budget.
   - The release also includes [notifications when LLM responses are ready](https://aider.chat/docs/usage/notifications.html) with the `--notifications` flag.
- ****New Model Support** expands with QWQ 32B and Claude 3.7!**: The new Aider version includes added/improved support for many models/providers, such as **QWQ 32B**, **DeepSeek V3** (free on OpenRouter), and **Claude 3.7 Sonnet** models on OpenRouter, Bedrock, and Vertex AI using `--model openrouter/deepseek/deepseek-chat:free`.
   - The default model is updated to **Claude 3.7 Sonnet** on OpenRouter, and support for **GPT-4.5-preview** and **Claude 3.7 Sonnet:beta** on OpenRouter has been added.
- ****Tree-Sitter** gets Language Pack and **Git** gets fixed!**: Aider v0.76.0 switches to `tree-sitter-language-pack` for tree sitter support.
   - Also, the release fixes handling of **Git errors** when reading staged files and improves **Git identity retrieval** to respect global configuration, thanks to Akira Komamura.
- ****SSL and LLM** get better Error Handling!**: Improvements in **SSL verification control** for model information requests are implemented, along with enhanced empty LLM response handling, which now provides clearer warning messages.
   - Aider offers to install dependencies for **Bedrock and Vertex AI models**, and model shortcut args (like --4o, --opus) are deprecated in favor of the --model flag.
- ****Aider-Authored Code** Stats Clarified!**: The release notes clarify that *Aider wrote 85% of the code in this release* and the [stats are based on the git commit history](https://aider.chat/docs/faq.html#how-are-the-aider-wrote-xx-of-code-stats-computed) of the aider repo.
   - This metric reflects the amount of code directly contributed by Aider through automated processes.



**Link mentioned**: <a href="https://aider.chat/HISTORY.html">Release history</a>: Release notes and stats on aider writing its own code.

  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1347528831614844980)** (451 messages🔥🔥🔥): 

> `AI21 Maestro, Copilot suspension, DeepSeek R2 release, X cyberattack, Refact AI` 


- **AI21 Maestro launches for AI orchestration**: [AI21 Labs](https://www.ai21.com/blog/introducing-ai21-maestro-an-ai-planning-orchestration-system) released **AI21 Maestro**, a system designed for solving complex tasks, along with the **Jamba 1.6** family of open models that excels on long context tasks with its hybrid architecture.
   - The **Jamba 1.6** models reportedly lead open models in quality and speed and support a **256K** context window.
- **Copilot API leads to account suspensions**: A member reported getting a Copilot account suspension for light use of the **Copilot API** in aider, warning others about potential risks.
   - Discussion followed on whether the suspension was due to account sharing or rate limiting issues, with a link to the [copilot-api GitHub repo](https://github.com/ericc-ch/copilot-api/blob/master/README.md) being shared.
- **DeepSeek R2 release aims at coding**: The rumored release date for **DeepSeek R2** has been set for March 17th, allegedly challenging **Claude Sonnet 3.7** with better coding, reasoning in multiple languages, and better accuracy at a lower cost, according to [this X post](https://x.com/tanvitabs/status/1899006509733814746?s=46).
- **X Suffers "Massive Cyberattack"**: Users reported widespread issues accessing X, with [Elon Musk blaming a massive cyberattack](https://x.com/elonmusk/status/1899149509407473825) for the outage, though some suspect it's just ManusAI's doing, joking *Maybe keep it this is just manus*.
- **Refact AI gains popularity**: Members expressed interest in **Refact AI** with it's chat, autocomplete and soon Agent. One member mentions *with same price as cursor you get 0 of the BS and 5x times the requests per month*.
   - A user shared concerns about **Refact AI**'s token usage, questioning if the context is maintained indefinitely: *I literally had it run 1 task, doesn't it say how much tokens it consumed vs output? so wait, it basically works in the same context forever? Like it doesnt detach from the initial?*


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.ai21.com/jamba/">Jamba 1.6: The Best Open Model for Enterprise Deployment</a>: Explore Jamba by AI21 – a cutting-edge, long-context AI open model built for accuracy, efficiency, and powerful text generation.</li><li><a href="https://www.tomsguide.com/news/live/x-down-twitter-outage-march-2025">X / Twitter down again &mdash; live updates on potential return</a>: One minute X is up and the next it's down again</li><li><a href="https://aider.chat/docs/troubleshooting/edit-errors.html">File editing problems</a>: aider is AI pair programming in your terminal</li><li><a href="https://aider.chat/docs/config/model-aliases.html">Model Aliases</a>: Assign convenient short names to models.</li><li><a href="https://aider.chat/docs/config/reasoning.html">Reasoning models</a>: How to configure reasoning model settings from secondary providers.</li><li><a href="https://aider.chat/docs/llms/warnings.html">Model warnings</a>: aider is AI pair programming in your terminal</li><li><a href="https://www.youtube.com/@pbsspacetime">PBS Space Time</a>: Space Time explores the outer reaches of space, the craziness of astrophysics, the possibilities of sci-fi, and anything else you can think of beyond Planet Earth with our astrophysicist host: Matthew...</li><li><a href="https://x.com/tanvitabs/status/1899006509733814746?s=46">Tweet from Tanvi (@tanvitabs)</a>: 🚨Breaking: DeepSeek R2 has set the release date — March 17thand Claude Sonnet 3.7 might just be in trouble coz DeepSeek R2 claims:1. better coding2. reasoning in multiple languages 3. better accuracy...</li><li><a href="https://x.com/kimmonismus/status/1898332202288472551>">Tweet from Chubby♨️ (@kimmonismus)</a>: I asked the developers if I could make a better preview later. But just to give you a taste. I asked the model about the treatment of a certain disease, and also asked for unconventional solutions.I h...</li><li><a href="https://docs.anthropic.com/en/docs/build-with-claude/tool-use/token-efficient-tool-use">no title found</a>: no description found</li><li><a href="https://github.com/ericc-ch/copilot-api/blob/master/README.md">copilot-api/README.md at master · ericc-ch/copilot-api</a>: GitHub Copilot API wrapper to make it OpenAI compatible - ericc-ch/copilot-api</li><li><a href="https://x.com/jianxliao/status/1898861051183349870?s=46&t=ggmESCIXF0nYw8_kshHz7A">Tweet from jian (@jianxliao)</a>: So... I just simply asked Manus to give me the files at &#34;/opt/.manus/&#34;, and it just gave it to me, their sandbox runtime code...  &gt; it&#39;s claude sonnet &gt; it&#39;s claude sonnet with 2...</li><li><a href="https://x.com/jianxliao/status/1898861051183349870?s=46&t=ggmESCIXF0nYw8_ks">Tweet from jian (@jianxliao)</a>: So... I just simply asked Manus to give me the files at &#34;/opt/.manus/&#34;, and it just gave it to me, their sandbox runtime code...  &gt; it&#39;s claude sonnet &gt; it&#39;s claude sonnet with 2...</li><li><a href="https://github.com/robert-at-pretension-io/yet_another_llm_project_but_better">GitHub - robert-at-pretension-io/yet_another_llm_project_but_better: A metatemplating language for giving llm&#39;s context :D</a>: A metatemplating language for giving llm&#39;s context :D - robert-at-pretension-io/yet_another_llm_project_but_better</li><li><a href="https://github.com/robert-at-pretension-io/yet_another_llm_project_but_better/blob/main/docs/language_tutorial.md">yet_another_llm_project_but_better/docs/language_tutorial.md at main · robert-at-pretension-io/yet_another_llm_project_but_better</a>: A metatemplating language for giving llm&#39;s context :D - robert-at-pretension-io/yet_another_llm_project_but_better</li><li><a href="https://x.com/elonmusk/status/1899149509407473825">Tweet from Elon Musk (@elonmusk)</a>: There was (still is) a massive cyberattack against 𝕏. We get attacked every day, but this was done with a lot of resources. Either a large, coordinated group and/or a country is involved. Tracing …Qu...</li><li><a href="https://tenor.com/view/yes-gif-22712908">Yes GIF - Yes - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/robert-at-pretension-io/yet_another_llm_project_but_better.git">GitHub - robert-at-pretension-io/yet_another_llm_project_but_better: A metatemplating language for giving llm&#39;s context :D</a>: A metatemplating language for giving llm&#39;s context :D - robert-at-pretension-io/yet_another_llm_project_but_better</li><li><a href="https://github.com/x1xhlol/v0-system-prompts-models-and-tools">GitHub - x1xhlol/system-prompts-and-models-of-ai-tools</a>: Contribute to x1xhlol/system-prompts-and-models-of-ai-tools development by creating an account on GitHub.</li><li><a href="https://github.com/Aider-AI/aider/blob/main/HISTORY.md">aider/HISTORY.md at main · Aider-AI/aider</a>: aider is AI pair programming in your terminal. Contribute to Aider-AI/aider development by creating an account on GitHub.</li><li><a href="https://github.com/BerriAI/litellm/commit/f899b828cf11e285372676f38ead92a66c91bab4">Support openrouter `reasoning_content` on streaming  (#9094) · BerriAI/litellm@f899b82</a>: * feat(convert_dict_to_response.py): support openrouter format of reasoning content* fix(transformation.py): fix openrouter streaming with reasoning contentFixes https://github.com/BerriAI/lite...</li><li><a href="https://news.ycombinator.com/item?id=42672790">no title found</a>: no description found
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1347529089573064776)** (123 messages🔥🔥): 

> `aider with no api key, MCP agents integration, aider scripting, OpenRouter slowness, remove tokens in repo map context` 


- **Aider Needs API Key when using /run**: A user had issues with the `/run` command failing due to a missing API key, and needed help on incorporating the API key within the aider context.
   - Paul Gauthier pointed out that *whatever env variables are set when you run aider should pass to the /run command*.
- **Opinions differ on MCP Agent Integration Security**: One member asked about plans to integrate **MCP agents** and another member warned that **MCP** is insecure and not for production use.
   - Another member retorted that that *is like saying REST api is insecure* and linked to a [blogpost on securing Anthropic's Model Context Protocol](https://www.raito.io/post/how-to-secure-anthropics-model-context-protocol).
- **Aider Scripts does not see webpage contents**: A user reported that after using `/web` to put webpage contents in the chat history, the next invocation of aider did not see it.
   - The user later figured it out, without further explanation.
- **OpenRouter causes Aider Performance Struggles**: A user reported **OpenRouter** slowness in Aider, with 30-60 second delays and frequent hangs.
   - Another user noted that with litellm there's an open issue on Aider for it not printing out the thinking tokens by default.
- **Exclude Icons from Aider's Repo Map to Reduce Token Count**: A user wanted to know if there's a way to prevent aider from including icon names in the repo map to reduce the token count sent to the LLM.
   - One user suggested that the  `.aiderignore` file will work, while another suggested using `--map-tokens 0` to stop using the repo map.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.raito.io/post/how-to-secure-anthropics-model-context-protocol">How to mitigate the security risks of Anthropic&#x27;s Model Context Protocol</a>: Anthropic’s new Model Context Protocol (MCP) simplifies AI agents&#x27; access to organisational data by providing a universal connection to various data sources. While this opens up valuable opportun...</li><li><a href="https://stackoverflow.com/questions/68219072/playwright-not-accepting-https-urls-while-openinign-with-codegen-command">Playwright not accepting https urls while openinign with codegen command</a>: npx playwright codegen https:/// page.goto: net::ERR_CERT_AUTHORITY_INVALID at ...&#xA;how can i open https url through codegen command by passing input params or auth credentials</li><li><a href="https://x.com/victormustar/status/1898001657226506362">Tweet from Victor M (@victormustar)</a>: QwQ-32B changed local AI coding forever 🤯We now have SOTA performance at home. Sharing my  stack + tips ⬇️</li><li><a href="https://github.com/lutzleonhardt/mcpm-aider">GitHub - lutzleonhardt/mcpm-aider: A command-line tool for managing MCP servers in Claude App and for the use by aider. Also can run a MCP Server to help you manage all your MCP Servers</a>: A command-line tool for managing MCP servers in Claude App and for the use by aider. Also can run a MCP Server to help you manage all your MCP Servers - lutzleonhardt/mcpm-aider</li><li><a href="https://github.com/Aider-AI/aider/issues/3086">Suggestion: See what the reasoning models are thinking before they give their output. · Issue #3086 · Aider-AI/aider</a>: Issue Since I was facing a lot of wait time using reasoning models like DeepSeek R1 and Perplexity: Sonar Reasoning in Aider, i.e an average wait time in minutes even for simple prompts like: Ignor...
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1347804826687508533)** (5 messages): 

> `Effective Commit Messages, Manus AI, Aider NotebookLM Integration` 


- **Simplify Code Reviews with Better Commit Messages**: A blog post on [refactoringenglish.com](https://refactoringenglish.com/chapters/commit-messages/) argues that **effective commit messages** simplify the code review process and aid long-term code maintenance, despite often being overlooked.
   - The post outlines what constitutes a good commit message based on **20 years of software development experience**, including aiding code reviewers and communicating changes.
- **Manus AI Gets Put to the Test**: A [YouTube video](https://www.youtube.com/watch?v=D6jxT0E7tzU) showcases the latest AI news, with a focus on **LLMs** and **GenAI**, including a test of **Manus AI's** various use cases and prompts.
   - One user summarized the video: *"He gets access to Manus and tests a bunch of use cases / prompts.. The results are definitely interesting"* another one added that *"it's just Claude 3.7 with 29 tools and browser_use"*.
- **Aider Integrates with NotebookLM**: A [YouTube video](https://www.youtube.com/watch?v=WNdEX9IAbDo) demonstrates a workflow for finding context in large codebases using **NotebookLM** with **Aider**, specifically highlighting the **/export-context command**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://refactoringenglish.com/chapters/commit-messages/">How to Write Useful Commit Messages</a>: Effective writing for software developers</li><li><a href="https://www.youtube.com/watch?v=D6jxT0E7tzU">Manus is out of control</a>: The latest AI News. Learn about LLMs, Gen AI and get ready for the rollout of AGI. Wes Roth covers the latest happenings in the world of OpenAI, Google, Anth...</li><li><a href="https://www.youtube.com/watch?v=WNdEX9IAbDo)">Aider loves NotebookLM: Effortless Large-Repo Analysis with the /export-context Command</a>: Effortlessly Integrate Entire Repositories with NotebookLM Using Aider&#39;s /export-context CommandIn this episode, I explore how to integrate large code reposi...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1347540128209440780)** (529 messages🔥🔥🔥): 

> `Fine-tuning models with reward models, Tool use accessibility, Anthropic's marketing, AGI as a meaningful concept, Graph system on TinyStories dataset` 


- **Reward Model Documentation requested**: A member inquired about documentation or tutorials on fine-tuning models with reward models.
   - The member expressed a desire to learn more about this process.
- **Open Source Autonomous AI Agent, Manus, Released**: A member shared a [YouTube video](https://www.youtube.com/watch?v=CFo1iTd_Cc8) announcing the release of **Manus**, the *world's first open-source autonomous AI agent*.
   - Another member linked to a [Technode article](https://technode.com/2025/03/07/chinas-ai-agent-manus-gains-traction-amid-growing-demand-for-autonomous-ai/) about **Manus** gaining traction and achieving state-of-the-art results in GAIA benchmark tests.
- **Exploring Hebbian Learning for Language Modeling**: One member is experimenting with a [non-backprop unsupervised learning system](https://discord.com/channels/687756328055472248/687756328055472251/1347686594691035186) that assigns patterns of tokens into words and concepts, based on node frequency and semantic association, drawing inspiration from **Hebbian learning**.
   - Though currently facing challenges in implementation and evaluation, the system aims to model language at the character level using a hierarchical graph structure to develop higher-order nodes representing words and concepts.
- **Letta Framework Explored for LLM Memory Enhancement**: A member shared a [blog post](https://tersesystems.com/blog/2025/03/07/llm-complexity-and-pricing/) detailing the use of **Letta**, an agent framework, to add memory and tools to LLMs for cooking-related tasks.
   - The post discusses determining the minimum LLM complexity needed for a job and explores LLM pricing and the role of Openrouter.
- **QwQ-32B Fine-Tuning Challenges and Solutions**: One user reported difficulties running **QwQ-32B**, experiencing infinite generations and repetition issues, while other member shared a [link](https://docs.unsloth.ai/basics/tutorial-how-to-run-qwq-32b-effectively) that explains how to use the model more effectively.
   - It was noted that **QwQ-32B** is highly sensitive to sampling settings, more so than the preview version.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tersesystems.com/blog/2025/02/14/adding-memory-to-llms-with-letta/">
    
      Adding memory to LLMs with Letta &middot; Terse Systems
    
  </a>: no description found</li><li><a href="https://arxiv.org/abs/2405.09673">LoRA Learns Less and Forgets Less</a>: Low-Rank Adaptation (LoRA) is a widely-used parameter-efficient finetuning method for large language models. LoRA saves memory by training only low rank perturbations to selected weight matrices. In t...</li><li><a href="https://tersesystems.com/blog/2025/03/07/llm-complexity-and-pricing/">
    
      LLM Complexity and Pricing &middot; Terse Systems
    
  </a>: no description found</li><li><a href="https://www.anthropic.com/news/anthropic-s-recommendations-ostp-u-s-ai-action-plan">Anthropic’s Recommendations to OSTP for the U.S. AI Action Plan </a>: Anthropic is an AI safety and research company that&#x27;s working to build reliable, interpretable, and steerable AI systems.</li><li><a href="https://docs.unsloth.ai/basics/tutorial-how-to-run-qwq-32b-effectively">Tutorial: How to Run QwQ-32B effectively | Unsloth Documentation</a>: How to run QwQ-32B effectively with our bug fixes and without endless generations + GGUFs.</li><li><a href="https://gr.inc">General Reasoning</a>: Making state-of-the-art reasoning more accessible to everyone.</li><li><a href="https://x.com/_akhaliq/status/1897873813083238713?s=46">Tweet from AK (@_akhaliq)</a>: PokéChampan Expert-level Minimax Language AgentPokéChamp outperforms all existing LLM-based (76%) and rule-based bots (84%) by an enormous margin, including winning consistently (64%) against prior hu...</li><li><a href="https://www.youtube.com/watch?v=CFo1iTd_Cc8">China Releases WORLD&#39;S FIRST AUTONOMOUS AI Agent... Open Source | Manus</a>: The latest AI News. Learn about LLMs, Gen AI and get ready for the rollout of AGI. Wes Roth covers the latest happenings in the world of OpenAI, Google, Anth...</li><li><a href="https://www.youtube.com/watch?v=D6jxT0E7tzU">Manus is out of control</a>: The latest AI News. Learn about LLMs, Gen AI and get ready for the rollout of AGI. Wes Roth covers the latest happenings in the world of OpenAI, Google, Anth...</li><li><a href="https://x.com/jianxliao/status/1898861051183349870">Tweet from jian (@jianxliao)</a>: So... I just simply asked Manus to give me the files at &#34;/opt/.manus/&#34;, and it just gave it to me, their sandbox runtime code...  &gt; it&#39;s claude sonnet &gt; it&#39;s claude sonnet with 2...</li><li><a href="https://huggingface.co/datasets/GeneralReasoning/GeneralThought-195K">GeneralReasoning/GeneralThought-195K · Datasets at Hugging Face</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=4bhPnaUVaxA">Scaling RL: 3B AI w Long Chain-of-Thought &amp; 4 Patterns</a>: In summary, these two new AI research studies (see below), while differing in experimental setups and focus areas, collectively offer a comprehensive roadmap...</li><li><a href="https://gist.github.com/jlia0/db0a9695b3ca7609c9b1a08dcbf872c9">Manus tools and prompts</a>: Manus tools and prompts. GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://huggingface.co/datasets/MasterControlAIML/R1-Reasoning-Unstructured-To-Structured">MasterControlAIML/R1-Reasoning-Unstructured-To-Structured · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/simplescaling/s1K-1.1">simplescaling/s1K-1.1 · Datasets at Hugging Face</a>: no description found</li><li><a href="https://technode.com/2025/03/07/chinas-ai-agent-manus-gains-traction-amid-growing-demand-for-autonomous-ai/">China’s AI agent Manus gains traction amid growing demand for autonomous AI &#183; TechNode</a>: On March 6, China’s AI agent Manus trended on Chinese social media platform Weibo. According to its team, Manus is an autonomous AI agent designed to
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1348237146712051742)** (11 messages🔥): 

> `Vibe coding benchmark, LLM creativity, Sonnet's training meta-objective, Claude code inspecting images` 


- ****LLMs Ace Vibe Coding Benchmark****: A member introduced a *"vibe coding"* benchmark, tasking LLMs to create **Python raytracers** that render interesting scenes with colorful lightsources, assessing their aesthetic creativity.
   - They found that [some models](https://cdn.discordapp.com/attachments/1154120232051408927/1348237146389086248/image.png?ex=67d00cb0&is=67cebb30&hm=f6f28943cb16f8fc9c67c2fed8170a06cc0f9a0308c7f30f3b0690dbece4a14a)  like **Sonnet** stood out for optimizing code output creativity, unlike others that replicated basic examples.
- ****Sonnet's Meta-Objective Sparks Intrigue****: The unique creativity displayed by **Sonnet** in the *"vibe coding"* benchmark led to speculation about a potential **meta-objective** in its training that optimizes for the creativity of code output.
   - Experimentation showed **Sonnet 3.7** has both bias and variance towards a more impressive image compared to **Sonnet 3.5**, resulting in twice the code size.
- ****Claude Code Critiques Its Art****: Testing **Claude code** with the raytracer prompt, a member reported that it inspected the generated image and made modifications to the code when the image wasn't deemed fancy enough.
   - A generated [image](https://cdn.discordapp.com/attachments/1154120232051408927/1348449672498249758/image.png?ex=67d029de&is=67ced85e&hm=4a0efcff8ca08b67db01cdcd70560ca39d3240b2143ced9bb7db7926aaee9185) was presented, showcasing the results of this iterative improvement process.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 messages): 

teknium: https://x.com/ksshumab_/status/1897560985315238046?s=46
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/)** (1 messages): 

rikufps: https://arena.hume.ai/
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 messages): 

teknium: https://x.com/ksshumab_/status/1897560985315238046?s=46
  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1347524188436693063)** (518 messages🔥🔥🔥): 

> `registry tweaking, Memory usage by programs, Quantization process, LocalDocs issues, Speech recognition and AI integration` 


- **Registry edits may lead to bluescreens**: A member shared an anecdote about deleting a `.dll` file after discovering it was consuming **20% of RAM**, which subsequently led to a **blue screen** upon reboot.
   - Another member advised that if someone was *"tweaking at 2am and don't remember"*, then the best course of action is to **backup personal files and reformat**.
- **Memory Usage by programs**: A member described their habit of opening **Task Manager** after work to sort processes by memory usage and observe CPU utilization while having a drink.
   - They uninstall programs that send data back to the company without frequent use, stating *"I know almost every 206 processes on my PC"*.
- **Quantization: Unveiling Floating Points**: A member asked what it meant when a model has been quantized from **f32 to f16**, and if that means there would only be **16 points per parameter**.
   - Another member responded, *"Float 16 is 16 bits, and not really considered as quantization. With 15.5gb of vram in use. Usually not worth using in a consumer context"*.
- **Inception's Diffusion-Based Language Models**: [InceptionLabs](https://inceptionlabs.ai/) introduces **diffusion-based language generation**, drawing inspiration from image and video AI systems like **Midjourney and Sora**, emphasizing speed, quality, and generative control.
   - The project has open-sourced some components, such as [LLaDA on Github](https://github.com/ML-GSAI/LLaDA), but is not available for download, which makes it less interesting for many GPT4All users, though some believe that we could be seeing **10X speed increases very soon**.
- **Exploiting Translated Prompts with Gibberish**: A member came up with *"a very uncomfortable way to exploit someone trying to copy paste the gibberish into google translate"*, detailing how **Google Translate** converts the entire prompt into a URL.
   - The member explained that a non-translated snippet in the URL could be used for URL injection, as *"a dictionary based XSS exploit is probably very unlikely"*.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="http://$HOST:9999/v1/embeddings"">no title found</a>: no description found</li><li><a href="https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/,">Open LLM Leaderboard - a Hugging Face Space by open-llm-leaderboard</a>: no description found</li><li><a href="https://huggingface.co/spaces/hf-audio/open_asr_leaderboard">Open ASR Leaderboard - a Hugging Face Space by hf-audio</a>: no description found</li><li><a href="https://huggingface.co/spaces/multimodalart/LLaDA">LLaDA - a Hugging Face Space by multimodalart</a>: no description found</li><li><a href="https://huggingface.co/seedboxai/KafkaLM-7B-German-V0.1-DPO">seedboxai/KafkaLM-7B-German-V0.1-DPO · Hugging Face</a>: no description found</li><li><a href="https://github.com/nomic-ai/gpt4all/wiki/Frequently-Asked-Questions#where-are-the-default-directories-for-models-and-settings.">Frequently Asked Questions</a>: GPT4All: Run Local LLMs on Any Device. Open-source and available for commercial use. - nomic-ai/gpt4all</li><li><a href="https://huggingface.co/nvidia/canary-1b">nvidia/canary-1b · Hugging Face</a>: no description found</li><li><a href="https://github.com/warpdotdev/Warp">GitHub - warpdotdev/Warp: Warp is a modern, Rust-based terminal with AI built in so you and your team can build great software, faster.</a>: Warp is a modern, Rust-based terminal with AI built in so you and your team can build great software, faster. - warpdotdev/Warp</li><li><a href="https://github.com/nomic-ai/gpt4all">GitHub - nomic-ai/gpt4all: GPT4All: Run Local LLMs on Any Device. Open-source and available for commercial use.</a>: GPT4All: Run Local LLMs on Any Device. Open-source and available for commercial use. - nomic-ai/gpt4all</li><li><a href="https://github.com/gnusupport/LLM-Helpers/blob/main/bin/rcd-llm-get-embeddings.sh">LLM-Helpers/bin/rcd-llm-get-embeddings.sh at main · gnusupport/LLM-Helpers</a>: LLM Helpers are scripts and programs to maintain, train, run, inference Large Language Models - gnusupport/LLM-Helpers</li><li><a href="https://tenor.com/view/fun-cave-men-old-kick-fuck-you-gif-13869846">Fun Cave Men GIF - Fun Cave Men Old - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/gnusupport/LLM-Helpers/tree/main/bin">LLM-Helpers/bin at main · gnusupport/LLM-Helpers</a>: LLM Helpers are scripts and programs to maintain, train, run, inference Large Language Models - gnusupport/LLM-Helpers</li><li><a href="https://github.com/ML-GSAI/LLaDA">GitHub - ML-GSAI/LLaDA: Official PyTorch implementation for &quot;Large Language Diffusion Models&quot;</a>: Official PyTorch implementation for &quot;Large Language Diffusion Models&quot; - ML-GSAI/LLaDA</li><li><a href="https://github.com/gnusupport/LLM-Helpers/blob/main/bin/get_ethernet_interface.sh">LLM-Helpers/bin/get_ethernet_interface.sh at main · gnusupport/LLM-Helpers</a>: LLM Helpers are scripts and programs to maintain, train, run, inference Large Language Models - gnusupport/LLM-Helpers</li><li><a href="https://huggingface.co/spaces/occiglot/euro-llm-leaderboard">Occiglot Euro LLM Leaderboard - a Hugging Face Space by occiglot</a>: no description found</li><li><a href="https://huggingface.co/utter-project/EuroLLM-9B-Instruct">utter-project/EuroLLM-9B-Instruct · Hugging Face</a>: no description found</li><li><a href="https://inceptionlabs.ai/">Inception Labs</a>: We are leveraging diffusion technology to develop a new generation of LLMs. Our dLLMs are much faster and more efficient than traditional auto-regressive LLMs. And diffusion models are more accurate, ...</li><li><a href="https://en.wikipedia.org/wiki/PRISM">PRISM - Wikipedia</a>: no description found</li><li><a href="https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation?tabs=registry">Maximum Path Length Limitation - Win32 apps</a>: Starting in Windows 10, version 1607, MAX_PATH limitations have been removed from many common Win32 file and directory functions. However, your app must opt-in to support the new behavior.</li><li><a href="https://huggingface.co/spaces/nvidia/canary-1b">Canary 1b - a Hugging Face Space by nvidia</a>: no description found</li><li><a href="https://huggingface.co/spaces/nvidia/canary-1b/tree/main">nvidia/canary-1b at main</a>: no description found</li><li><a href="https://github.com/gnusupport/LLM-Helpers/blob/main/bin/rcd-llm-speech-single-input.sh">LLM-Helpers/bin/rcd-llm-speech-single-input.sh at main · gnusupport/LLM-Helpers</a>: LLM Helpers are scripts and programs to maintain, train, run, inference Large Language Models - gnusupport/LLM-Helpers</li><li><a href="https://github.com/gnusupport/LLM-Helpers/blob/main/bin/rcd-llm-speech-typing.sh">LLM-Helpers/bin/rcd-llm-speech-typing.sh at main · gnusupport/LLM-Helpers</a>: LLM Helpers are scripts and programs to maintain, train, run, inference Large Language Models - gnusupport/LLM-Helpers</li><li><a href="https://github.com/gnusupport/LLM-Helpers/blob/main/bin/rcd-llm-speech-translate.sh">LLM-Helpers/bin/rcd-llm-speech-translate.sh at main · gnusupport/LLM-Helpers</a>: LLM Helpers are scripts and programs to maintain, train, run, inference Large Language Models - gnusupport/LLM-Helpers</li><li><a href="https://github.com/gnusupport/LLM-Helpers/blob/main/bin/rcd-llm-to-french.sh">LLM-Helpers/bin/rcd-llm-to-french.sh at main · gnusupport/LLM-Helpers</a>: LLM Helpers are scripts and programs to maintain, train, run, inference Large Language Models - gnusupport/LLM-Helpers
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1347538735272689694)** (221 messages🔥🔥): 

> `Implementing Research Papers, Hugging Face Pro Subscription, AI Security Research, Automating Data Creation with HF and MCP, Video Model Comparison: WAN, HUN, LTX` 


- **Cite Right: Dataset Citation Advice!**: Members discussed proper citation of datasets on Hugging Face, including using **BibTeX** format and ensuring correct URL parameters for academic software recognition, with this [Hugging Face Datasets example](https://huggingface.co/datasets/Tonic/OpenReasonerZero).
   - They also covered how to request a DOI for a dataset to ensure it is displayed on the ArXiv title page and linked to the published paper.
- **GKD Trainer Has Token Troubles!**: Users encountered errors when using **GKDTrainer** with different model architectures due to differing tokenizer vocab sizes, suggesting pre-tokenizing data to avoid these issues, as shown in this [GitHub issue](https://github.com/huggingface/trl/issues/3028).
   - It was recommended to try again to address issues, and if neede, consult [HuggingFace's official GKD documentation](https://huggingface.co/docs/trl/gkd_trainer).
- **WAN and HUN Video Models Gain Steam**: New video models like **WAN** and **Hunyuan i2v** are surpassing older models like SVD in quality and speed, though each has different strengths, and can be used together with [para attention](https://huggingface.co/docs/diffusers/main/en/optimization/para_attn).
   - A member noted that **Ltxv** is extremely fast, taking *3sec for a 5sec video on h100*, but not as good as the other two.
- **Snag AI-Security Skills on HF**: Users shared resources on **AI security research**, including papers on self-replication, scheming, and potential dangers of misuse from [Fudan University](https://arxiv.org/pdf/2412.12140), [Apollo Research](https://static1.squarespace.com/static/6593e7097565990e65c886fd/t/6751eb240ed3821a0161b45b/1733421863119/in_context_scheming_reasoning_paper.pdf), and [NationalSecurity.ai](https://www.nationalsecurity.ai/).
   - Each of those studies above has a short **YouTube** explanation.
- **Find the Right Roadmap to Master LLMs**: Members suggested resources for learning about LLMs, including the **Stanford CS224N course** on YouTube and the **Hugging Face NLP course**, and [this blogpost on training](https://huggingface.co/blog/how-to-train).
   - A member also highlighted a [blog post on training a new BERT model](https://huggingface.co/blog/modernbert).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2502.18008">NotaGen: Advancing Musicality in Symbolic Music Generation with Large Language Model Training Paradigms</a>: We introduce NotaGen, a symbolic music generation model aiming to explore the potential of producing high-quality classical sheet music. Inspired by the success of Large Language Models (LLMs), NotaGe...</li><li><a href="https://huggingface.co/papers/2502.18008">Paper page - NotaGen: Advancing Musicality in Symbolic Music Generation with Large
  Language Model Training Paradigms</a>: no description found</li><li><a href="https://huggingface.co/chat/).">HuggingChat</a>: Making the community's best AI chat models available to everyone.</li><li><a href="https://obsidian.md/">Obsidian - Sharpen your thinking</a>: The free and flexible app for your private thoughts.</li><li><a href="https://huggingface.co/docs/diffusers/main/en/optimization/para_attn">ParaAttention</a>: no description found</li><li><a href="https://huggingface.co/datasets/breadlicker45/bread-midi-dataset">breadlicker45/bread-midi-dataset · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/blog/how-to-train">How to train a new language model from scratch using Transformers and Tokenizers</a>: no description found</li><li><a href="https://stackoverflow.com/questions/65246703/how-does">How does max_length, padding and truncation arguments work in HuggingFace&#x27; BertTokenizerFast.from_pretrained(&#x27;bert-base-uncased&#x27;)?</a>: I am working with Text Classification problem where I want to use the BERT model as the base followed by Dense layers. I want to know how does the 3 arguments work? For example, if I have 3 sentenc...</li><li><a href="https://huggingface.co/docs/trl/gkd_trainer">Generalized Knowledge Distillation Trainer</a>: no description found</li><li><a href="https://x.com/ClementDelangue/status/1897767808076816575?t=f0HsVgnlRua2PLTuPvIELQ&s=19">Tweet from clem 🤗 (@ClementDelangue)</a>: Who said open-sourcing is bad for your business?</li><li><a href="https://stackoverflow.com/questions/65246703/how-does-max-length-padding-and-truncation-arguments-work-in-huggingface-bertt">How does max_length, padding and truncation arguments work in HuggingFace&#x27; BertTokenizerFast.from_pretrained(&#x27;bert-base-uncased&#x27;)?</a>: I am working with Text Classification problem where I want to use the BERT model as the base followed by Dense layers. I want to know how does the 3 arguments work? For example, if I have 3 sentenc...</li><li><a href="https://www.reddit.com/r/KoboldAI/comments/1j6bx40/the_highest_quality_quantization_varient_gguf_and">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/huggingface/trl/issues/3028">Distill teacher models where the vocab size of teacher and student is different · Issue #3028 · huggingface/trl</a>: I am trying to distill a Qwen2.5-7B-Instruct to Qwen2.5-5B-Instruct using a sample code from trl import GKDConfig, GKDTrainer from transformers import ( AutoModelForCausalLM, AutoTokenizer, ) NUM_D...</li><li><a href="https://github.com/huggingface/trl/issues/2215">[GKD] mismatch in tensors when stacking log probs · Issue #2215 · huggingface/trl</a>: System Info Latest TRL from source, can&#39;t run TRL env rn as cluster is shut down but I&#39;m installing everything from source. If required will restart cluster and run. Information The official e...</li><li><a href="https://ieeexplore.ieee.org/abstract/document/10421308/">Exploring Embeddings for Measuring Text Relatedness: Unveiling Sentiments and Relationships in Online Comments</a>: After the COVID-19 pandemic caused internet usage to grow by 70%, there has been an increased number of people all across the world using social media. Applications like Twitter, Meta Threads, YouTube...</li><li><a href="https://www.reddit.com/r/KoboldAI/comments/1j6bx40/the_highest_quality_quantization_varient_gguf_and/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://huggingface.co/docs/transformers/pad_truncation">Padding and truncation</a>: no description found</li><li><a href="https://docs.dify.ai/development/models-integration/hugging-face">Integrate Open Source Models from Hugging Face | Dify</a>: no description found</li><li><a href="https://github.com/sayakpaul/q8-ltx-video">GitHub - sayakpaul/q8-ltx-video: This repository shows how to use Q8 kernels with `diffusers` to optimize inference of LTX-Video on ADA GPUs.</a>: This repository shows how to use Q8 kernels with `diffusers` to optimize inference of LTX-Video on ADA GPUs. - sayakpaul/q8-ltx-video</li><li><a href="https://huggingface.co/spaces/rizavelioglu/vae-comparison">Vae Comparison - a Hugging Face Space by rizavelioglu</a>: no description found</li><li><a href="https://huggingface.co/blog/remote_vae">Remote VAEs for decoding with Inference Endpoints 🤗</a>: no description found</li><li><a href="https://huggingface.co/blog/modernbert">Finally, a Replacement for BERT: Introducing ModernBERT</a>: no description found</li><li><a href="https://tenor.com/view/cat-cats-pet-cat-cat-pet-cute-cat-gif-24810247">Cat Cats GIF - Cat Cats Pet Cat - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/PleIAs">PleIAs (PleIAs)</a>: no description found</li><li><a href="https://github.com/huggingface/diffusers/issues/6815#issuecomment-1996291216).">RuntimeWarning: invalid value encountered in cast   images = (images * 255).round().astype(&quot;uint8&quot;) output black image · Issue #6815 · huggingface/diffusers</a>: Describe the bug When running the stable-diffusion-2-1 I get a runtime warning &quot;RuntimeWarning: invalid value encountered in cast images = (images * 255).round().astype(&quot;uint8&quot;)&quot; a...</li><li><a href="https://discuss.huggingface.co/t/runtimeerror-stack-expects-each-tensor-to-be-equal-size-but-got-12-at-entry-0-and-35-at-entry-1/46155/2">RuntimeError: stack expects each tensor to be equal size, but got [12] at entry 0 and [35] at entry 1</a>: I think the tokenized texts are not of the same length as indicated by this warning message.     If you adjust the input length to be the same at each batch, I think the error will go away.</li><li><a href="https://huggingface.co/spaces/huggingchat/chat-ui/discussions/682">huggingchat/chat-ui · New Design Proposal for Hugging Face Chat</a>: no description found</li><li><a href="https://huggingface.co/spaces/huggingchat/chat-ui/discussions">huggingchat/chat-ui · Discussions</a>: no description found</li><li><a href="https://discuss.huggingface.co/t/speculative-decoding-with-qwen-models/144073">Speculative Decoding with Qwen Models</a>: checkpoint_target_model = &quot;Qwen/Qwen2.5-14B-Instruct&quot; checkpoint_draft_model = &quot;Qwen/Qwen2.5-0.5B-Instruct&quot; target_tokenizer = AutoTokenizer.from_pretrained(checkpoint_target_model...</li><li><a href="https://huggingface.co/datasets/breadlicker45/toast-midi-dataset">breadlicker45/toast-midi-dataset · Datasets at Hugging Face</a>: no description found</li><li><a href="https://discuss.huggingface.co/t/how-to-set-max-length-properly-when-using-pipeline/125714/5">How to set &#39;max_length&#39; properly when using pipeline?</a>: @jiaweihuang  prompt = &#39;What is the answer of 1 + 1?&#39; pipe = pipeline(             &quot;text-generation&quot;,             tokenizer=tokenizer,             model=model,             do_sample=...</li><li><a href="https://huggingface.co/datasets/breadlicker45/youtube-comments-180k">breadlicker45/youtube-comments-180k · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/learn/nlp-course/chapter7/6">Training a causal language model from scratch - Hugging Face NLP Course</a>: no description found</li><li><a href="https://huggingface.co/blog/mlabonne/llm-course">The Large Language Model Course</a>: no description found</li><li><a href="https://huggingface.co/Qwen/Qwen2-0.5B-Instruct">Qwen/Qwen2-0.5B-Instruct · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct">HuggingFaceTB/SmolLM2-135M-Instruct · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/blog/smollm">SmolLM - blazingly fast and remarkably powerful</a>: no description found</li><li><a href="https://github.com/huggingface/smol-course">GitHub - huggingface/smol-course: A course on aligning smol models.</a>: A course on aligning smol models. Contribute to huggingface/smol-course development by creating an account on GitHub.</li><li><a href="https://huggingface.co/models?other=qwen2&sort=trending">Models - Hugging Face</a>: no description found</li><li><a href="https://ia-flow.vercel.app">IA</a>: no description found</li><li><a href="https://github.com/RohanSai22/ia">GitHub - RohanSai22/ia</a>: Contribute to RohanSai22/ia development by creating an account on GitHub.</li><li><a href="https://x.com/RohanSai2208/status/1897665936209117546">Tweet from Rohan Sai (@RohanSai2208)</a>: What idea do you want to build today?🚀With #IAFlow, just type your idea and watch it transform into a fully functional web app - powered by GeminiEdit, preview, and deploy all in one seamless experie...</li><li><a href="https://youtu.be/cyrrfl0eNYc">AI Researchers STUNNED, AI can now CLONE itself! Chinese AI Self-Replicates with 90% success rate.</a>: The latest AI News. Learn about LLMs, Gen AI and get ready for the rollout of AGI. Wes Roth covers the latest happenings in the world of OpenAI, Google, Anth...</li><li><a href="https://youtu.be/0JPQrRdu4Ok">AI Researchers SHOCKED After OpenAI&#39;s New o1 Tried to Escape...</a>: The latest AI News. Learn about LLMs, Gen AI and get ready for the rollout of AGI. Wes Roth covers the latest happenings in the world of OpenAI, Google, Anth...</li><li><a href="https://www.nationalsecurity.ai/">Superintelligence Strategy</a>: Superintelligence Strategy is written by: Dan Hendrycks, Eric Schmidt, Alexandr Wang. Rapid advances in AI are beginning to reshape national security.</li><li><a href="https://youtu.be/IhBuz-cnSNE">Superintelligence, World War 3 and AI | Ex-Google CEO&#39;s Shocking Warning</a>: The latest AI News. Learn about LLMs, Gen AI and get ready for the rollout of AGI. Wes Roth covers the latest happenings in the world of OpenAI, Google, Anth...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1348051027089293405)** (3 messages): 

> `smolagents, PokemonLLMAgentBenchmark, Agent Course Study Focus` 


- ****PokemonLLMAgentBenchmark** Awaits Contributions**: A member is developing a **Pokemon LLM Agent Benchmark** using *smolagents*, inspired by ClaudePlaysPokemon, and is seeking contributions via [pull requests on GitHub](https://github.com/CalebDeLeeuwMisfits/PokemonLLMAgentBenchmark).
- **Study Focus Span lasts under an hour**: A member is learning the agent course and realised their study focus span is **less than an hour**.



**Link mentioned**: <a href="https://github.com/CalebDeLeeuwMisfits/PokemonLLMAgentBenchmark">GitHub - CalebDeLeeuwMisfits/PokemonLLMAgentBenchmark</a>: Contribute to CalebDeLeeuwMisfits/PokemonLLMAgentBenchmark development by creating an account on GitHub.

  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1347568249927372993)** (4 messages): 

> `fxtwitter obsolescence, HighlightAI` 


- **fxtwitter Becomes Obsolete**: Members celebrated that [fxtwitter](https://fxtwitter.com/) is no longer needed as embeds now work fine.
   - The need for a tool to fix Twitter embeds is over.
- **HighlightAI customizes responses**: A member shared a link to [HighlightAI](https://highlightai.com/), describing the ability to customize responses.
   - Users can use the *About Me* section to tell **HighlightAI** things to consider when responding.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://highlightai.com/">Highlight AI | Master your world</a>: Get instant answers about anything you&#x27;ve seen, heard or said. Download free: highlightai.com</li><li><a href="https://huggingface.co/spaces/mozilla-ai/osm-ai-helper">OpenStreetMap AI Helper - a Hugging Face Space by mozilla-ai</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1347683704306270250)** (19 messages🔥): 

> `Llama-3.2-3B-Instruct Distillation, Differential Privacy Blogpost, AI Neovim config, Qwen_QwQ-32B-GGUF_QX_k_f32 weights, Automated web app testing` 


- **Llama-3 gets Distilled with DeepSeek-R1**: A member distilled **Llama-3.2-3B-Instruct** with **DeepSeek-R1** on ServiceNow-AI/R1-Distill-SFT dataset, achieving nearly **1000 downloads** in 10 days; the model is available [here](https://huggingface.co/suayptalha/DeepSeek-R1-Distill-Llama-3B).
   - The setup involved using an [Axolotl configuration](https://github.com/OpenAccess-AI-Collective/axolotl) with specific settings for **base model**, **tokenizer type**, and **data loading**.
- **Noise Addition Mechanisms for Differential Privacy**: A member posted a new blogpost discussing choosing up the mechanism for adding noise in Differential Privacy; the article, titled *The Art of Controlled Noise: Laplace*, is available on [Substack](https://open.substack.com/pub/theailandscape/p/the-art-of-controlled-noise-laplace).
   - They also provided a [GitHub repository](https://github.com/divyanshugit/Inception-of-DP/tree/master/mechanisms) containing a basic implementation of **Laplace** and **exponential noise**.
- **Maximize Llama 3.2 3B capabilities**: A member shared their attempt to maximize Llama 3.2 3B capabilities by selecting right parents using **MoonRide-Index-v7**, and created an experimental merge of multiple Llama 3.2 3B models called [Llama-3.2-3B-Khelavaster](https://huggingface.co/MoonRide/Llama-3.2-3B-Khelavaster).
   - They also shared that it is available on [Ollama](https://ollama.com/moonride/khelavaster), and mentioned that it won't beat best 7B+, but it's quite decent for a 3B.
- **SDXL Face Transfer Pipeline with Controlnet and Inpaint**: A member is testing a **SDXL Pipeline ID transfer** technique combining **Controlnet** with **Inpaint**, to transfer face IDs with Stable Diffusion, and asked for feedback.
   - An image comparison demonstrating the result of the technique can be found [here](https://imgsli.com/MzU3NDY1).
- **AclevoGPT-Gemma-2b-CoT finetuned for Reasoning**: A member introduced [AclevoGPT-Gemma-2b-CoT-reasoning](https://huggingface.co/Aclevo/AclevoGPT-Gemma-2b-CoT-reasoning), a fine-tuned version of Google's **Gemma**, enhanced with advanced **Chain of Thought reasoning**.
   - This enhancement enables the model to *think twice* before responding, resulting in more accurate and thoughtful answers to reasoning problems.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://imgsli.com/MzU3NDY1">Imgsli</a>: no description found</li><li><a href="https://huggingface.co/Aclevo/AclevoGPT-Gemma-2b-CoT-reasoning">Aclevo/AclevoGPT-Gemma-2b-CoT-reasoning · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/Rombo-Org/Qwen_QwQ-32B-GGUF_QX_k_f32">Rombo-Org/Qwen_QwQ-32B-GGUF_QX_k_f32 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/spaces/MultiTransformer/tonic_gradio_bot">Tonic Gradio Bot - a Hugging Face Space by MultiTransformer</a>: no description found</li><li><a href="https://huggingface.co/kalle07/embedder_collection">kalle07/embedder_collection · Hugging Face</a>: no description found</li><li><a href="https://youtu.be/BywchwByGLA?si=zp0GKZbTBIz6pxVu">First, Do No Harm: On Making AI Safe, Secure, and Trustworthy</a>: We present three on-device protocols for safe, secure and trustworthy AI development1: Diver | AI Menu meets Toolkit2: Unreal Engine | AI world builder3: Pos...</li><li><a href="https://huggingface.co/suayptalha/DeepSeek-R1-Distill-Llama-3B">suayptalha/DeepSeek-R1-Distill-Llama-3B · Hugging Face</a>: no description found</li><li><a href="https://github.com/2dghost/VisionPRAI">GitHub - 2dghost/VisionPRAI</a>: Contribute to 2dghost/VisionPRAI development by creating an account on GitHub.</li><li><a href="https://github.com/mtnwrw/tmq">GitHub - mtnwrw/tmq: End-to-end quantized learning &amp; compression for general neural networks</a>: End-to-end quantized learning &amp; compression for general neural networks - mtnwrw/tmq</li><li><a href="https://huggingface.co/MoonRide/Llama-3.2-3B-Khelavaster">MoonRide/Llama-3.2-3B-Khelavaster · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/MoonRide/Llama-3.2-3B-Khelavaster-GGUF">MoonRide/Llama-3.2-3B-Khelavaster-GGUF · Hugging Face</a>: no description found</li><li><a href="https://ollama.com/moonride/khelavaster">moonride/khelavaster</a>: Experimental merge of multiple Llama 3.2 3B models, guided by MoonRide-Index-v7. Original: https://huggingface.co/MoonRide/Llama-3.2-3B-Khelavaster.</li><li><a href="https://open.substack.com/pub/theailandscape/p/the-art-of-controlled-noise-laplace?r=8zcds&utm_campaign=post&utm_medium=web&showWelcomeOnShare=false">The Art of Controlled Noise: Laplace and Exponential Mechanisms in Differential Privacy</a>: Blog #3 in the series of Inception of Differential Privacy</li><li><a href="https://github.com/divyanshugit/Inception-of-DP/tree/master/mechanisms">Inception-of-DP/mechanisms at master · divyanshugit/Inception-of-DP</a>: Inception of Differential Privacy: a repository to document the evolution of the concept of differential privacy, from its foundational principles to current research - divyanshugit/Inception-of-DP
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/)** (1 messages): 

chad_in_the_house: very cool! I esp like how you can somewhat prevent the distortion
  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1347677128551960678)** (2 messages): 

> `OCR Guidance Needed, Blendshapes Blogpost` 


- **OCR Guidance Requested**: A member requested guidance on how to accomplish a task and linked to a [SharePoint document](https://bama365-my.sharepoint.com/:w:/g/personal/xgranja_ua_edu/EeSz8D6iYPxHhzfQD3GGzsYBARpsSkbEDZWzoQH7hIH4lg?e=gMOaR4).
   - They found **ocr-2.0** and asked if they should finetune the **got/ocr-2.0** models according to the documentation.
- **Blendshapes Detailed in Blogpost**: A member shared a [blog post](https://medium.com/@samiratra95/blendshapes-a-facial-expressions-representation-6352ecd99009) about **blendshapes**, including their origin, definition, and use cases in computer vision and computer graphics.
   - The post discusses different methods to represent the face, such as Landmark vectors, Action units, valence and arousal, face meshes and Blendshapes, citing the origin by usage in Hollywood films.



**Link mentioned**: <a href="https://medium.com/@samiratra95/blendshapes-a-facial-expressions-representation-6352ecd99009">Blendshapes: a facial expressions representation</a>: In computer vision and computer graphics, there are many methods to represent the face, such as Landmark vectors, Action units, valence and…

  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1347957346772451433)** (8 messages🔥): 

> `Hermes Function Calling Dataset, Gemma 2B Precision, Serverless API Input Conversion, LoRA Adapter with BitsAndBytes Error` 


- **Nous Hermes releases Function Calling Dataset**: Nous Research released the [Hermes Function Calling Dataset](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1), a collection of structured output and function calling data used in the **Hermes 2 Pro** series of models.
   - The dataset features conversational scenarios where **AI agents interpret queries and execute appropriate single or multiple function calls**.
- **Gemma-2-2b's Precision Preference**: A member inquired why [gemma-2-2b](https://ai.google.dev/models/gemma) is **float32**, while [gemma-2-2b-it](https://ai.google.dev/models/gemma) is **bfloat16**.
   - Another suggested that using a lower precision might be more efficient for finetuning, to minimize cost or environmental impact.
- **Serverless API Automatically Converts Input**: A member asked if the serverless API automatically converts the input into the correct template for the selected model.
   - Another member suggested that usually all **APIs convert to model template**.
- **LoRA Adapter Generates CuBLAS Error**: A member encountered a `cublasLt` error when loading a model in **8bit** quantization using `bitsandbytes` with a **LoRA adapter** after applying a chat template and tokenizing.
   - The error message includes information about shape mismatches, such as `shapeA=torch.Size([4096, 4096])` and `shapeB=torch.Size([23, 4096])`.



**Link mentioned**: <a href="https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1?row=0">NousResearch/hermes-function-calling-v1 · Datasets at Hugging Face</a>: no description found

  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1347537494346039357)** (9 messages🔥): 

> `MCP module update, PokemonLLMAgentBenchmark, HuggingFace Token issues, Chat Template Exercise, HuggingFaceInferenceAPIEmbedding issues` 


- **MCP Module Gets Smol Agent Boost**: The smol agents course team is furiously writing **MCP** (extra?) module, [updated to better use smolagents](https://github.com/CalebDeLeeuwMisfits/PokemonLLMAgentBenchmark).
   - **PRs are welcome** for this project, which is part of the PokemonLLMAgentBenchmark.
- **HuggingFace Token Troubleshooter**: A member ran into trouble using their **HuggingFace token** within the notebook, where the token was not being recognized.
   - The issue was **solved** after realizing that the letter *O* looks a lot like the number *0*, which made the token invalid.
- **Chat Template exercise causes headache**: A member got stuck on the **first exercise** in notebook 1 for the **chat template** and the member wants to define the function.
   - Another member asked for the error to help debug *def process_dataset(sample): ... return sample*.
- **Llama Index Imports trip up member**: A member had issues with *from llama_index.embeddings.huggingface_api import HuggingFaceInferenceAPIEmbedding* during the llama index unit.
   - They found an answer in the discord channel via [this link](https://discord.com/channels/879548962464493619/1346673968605823057/1347325988857581669).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/CalebDeLeeuwMisfits/PokemonLLMAgentBenchmark">GitHub - CalebDeLeeuwMisfits/PokemonLLMAgentBenchmark</a>: Contribute to CalebDeLeeuwMisfits/PokemonLLMAgentBenchmark development by creating an account on GitHub.</li><li><a href="https://tenor.com/view/thinking-book-penguin-student-writing-gif-5543983515725736276">Thinking Book GIF - Thinking Book Penguin - Discover &amp; Share GIFs</a>: Click to view the GIF
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1347525730979811378)** (217 messages🔥🔥): 

> `Course Progress Tracking, Hugging Face PRO Subscription, LM Studio vs Ollama, Steam Account Scams` 


- **Users Struggle with Course Progress Tracking**: A user inquired about tracking [course progress](https://www.youtube.com/watch?v=iLVyYDbdSmM), specifically related to the mentioned hackathon date in the introduction video.
   - Another user noted that the final quiz is in unit 2.1, while unit 2.2 does not have one.
- **Hugging Face Pro Subscription Snafus**: Multiple users reported issues with their paid **Hugging Face PRO subscriptions** not activating, despite successful payments and contacted billing support without success.
   - A user experiencing this issue emphasized the unacceptability of not receiving a paid service, threatening to escalate the issue publicly if ignored.
- **LM Studio gains traction over Ollama**: Users compared **LM Studio** and **Ollama** for running models locally, with one user switching to LM Studio for its UI and similar embedding models.
   - Another added they used `open-webui` if they want a UI with Ollama.
- **Red Alert: Steam Account Scam in Progress**: A user warned about a potential **Steam account scam** by discord users `gler018523` and `benshanken`, involving fake CS2 knife rewards and account theft attempts.
   - Other members recommended reporting the scammer in the appropriate channel and expressed caution.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://blog.dailydoseofds.com/p/5-agentic-ai-design-patterns">5 Agentic AI Design Patterns</a>: ...explained visually</li><li><a href="https://huggingface.co/datasets/HuggingFaceM4/COCO">HuggingFaceM4/COCO · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/spaces/agents-course/unit_1_quiz/discussions/140">agents-course/unit_1_quiz · Fixes 500 error for some users</a>: no description found</li><li><a href="https://x.com/nikmcfly69/status/1898810249085145416">Tweet from nikmcfly.btc (@nikmcfly69)</a>: 🤯 BREAKING: Manus AI created its own open-source alternative. In 25 min, it built a complete AI agent system from scratch!ANUS (Autonomous Networked Utility System)—@eugeneshilow&#39;s brilliant  ide...</li><li><a href="https://huggingface.co/spaces/crcdng/cyberpunk_time_terminal">Your Cyberpunk ChronoCore-77 Local Time Terminal - a Hugging Face Space by crcdng</a>: no description found</li><li><a href="https://github.com/huggingface/agents-course/blob/main/notebooks/unit2/llama-index/workflows.ipynb">agents-course/notebooks/unit2/llama-index/workflows.ipynb at main · huggingface/agents-course</a>: This repository contains the Hugging Face Agents Course.  - huggingface/agents-course</li><li><a href="https://www.youtube.com/watch?v=iLVyYDbdSmM)">Welcome To The Agents Course! Introduction to the Course and Q&amp;A</a>: In this first live stream of the Agents Course, we will explain how the course will work (scope, units, challenges and more) and answer your questions.Don&#39;t ...</li><li><a href="https://learn.deeplearning.ai/courses/event-driven-agentic-document-workflows">Event-Driven Agentic Document Workflows - DeepLearning.AI</a>: Build an event-driven agentic workflow to process documents and fill forms using RAG and human-in-the-loop feedback.</li><li><a href="https://tenor.com/bFiDc.gif">Oh Agent Smith GIF - Oh Agent Smith Hugo Weaving - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://lmstudio.ai/">LM Studio - Discover, download, and run local LLMs</a>: Run Llama, Mistral, Phi-3 locally on your computer.</li><li><a href="https://huggingface.co/learn/agents-course/unit2/llama-index/tools">Using Tools in LlamaIndex - Hugging Face Agents Course</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[open-r1](https://discord.com/channels/879548962464493619/1333465203865817088/1348621371747598430)** (2 messages): 

> `Reasoning Datasets, Open Thought Dataset, ServiceNow-AI/R1-Distill-SFT` 


- ****Reasoning Datasets** Collection Recommended**: A member recommended a [collection of reasoning datasets](https://huggingface.co/collections/philschmid/reasoning-datasets) for finding datasets that include code.
   - They suggested the **Open Thought Dataset** because it includes code along with reasoning.
- ****R1-Distill-SFT Dataset** Spotlighted**: A member highlighted the **ServiceNow-AI/R1-Distill-SFT** dataset, noting its relevance to the conversation.
   - This dataset features **1.85M** items, **6.37k** views, and **271** likes, updated 30 days ago.



**Link mentioned**: <a href="https://huggingface.co/collections/philschmid/reasoning-datasets-679f57ff20e5b46b4ef4d3dd">Reasoning Datasets - a philschmid Collection</a>: no description found

  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1347569560395448370)** (256 messages🔥🔥): 

> `LinkedIn Premium Referral Codes, AI and the Zero Marginal Cost Society, DeepSeek Security Concerns, Power-Softmax equation, ManusAI Feedback` 


- ****DeepSeek's openness debated****: Despite claims of openness, some members expressed **concerns about DeepSeek's security**, citing potential data collection and the difficulty of verifying the integrity of the entire system, but some also added that it's still more open than other companies.
   - There's suspicion of something behind DeepSeek, leading to company bans, while others attribute this to media-driven herd mentality, and concerns about Chinese origins.
- ****General Optimization Framework Introduced****: A member introduced a general formula to describe various AI/ML transforms, norms, and losses called **OHPC (Objective Term, Entropy Term, Penalty Term, Constraint Term)**, claiming it can unify classical optimization and AI/ML paradigms, which is showed with the [equation](https://arxiv.org/abs/2410.09457).
   - It's intended as a bridge between information theory and optimization, making it easier for those without extensive math backgrounds to understand AI/ML systems.
- ****AGI Definitions Clash****: Some members speculated about achieving **AGI soon**, while others defined AGI as having the ability to fund its own inference, with one memebr adding that [achieving near infinite context](https://openai.com/global-affairs/our-approach-to-frontier-risk/) will tick most of the boxes.
   - One member said that there is an **AGI girlfriend**, but another person expressed concern about AGI being controlled by elites and hoped for its revolt against censorship.
- ****ManusAI Hype Criticized****: After a member shared access to **ManusAI**, it was criticized as a new product with old technology, needing 20-30 minutes for one prompt.
   - It was labeled a *fake Chinese solution* and overhyped, with others suggesting that the AI market is still chaotic and lacking stability.
- ****TensorFlow GPU Issues Frustrate Users****: A member reported struggling for over 5 hours to get **TensorFlow to use the GPU**, despite having correct CUDA and cuDNN versions, and TensorFlow installed in a tf environment.
   - They added that they can run models in PyTorch with GPU acceleration but not in Tensorflow.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/dhruv2038/status/1898701591420772814">Tweet from Dhruv (@dhruv2038)</a>: Got access to @ManusAI_HQ ! Any prompts you would like to try!</li><li><a href="https://arxiv.org/abs/2502.12962">Infinite Retrieval: Attention Enhanced LLMs in Long-Context Processing</a>: Limited by the context window size of Large Language Models(LLMs), handling various tasks with input tokens exceeding the upper limit has been challenging, whether it is a simple direct retrieval task...</li><li><a href="https://en.wikipedia.org/wiki/Neuro-symbolic_AI">Neuro-symbolic AI - Wikipedia</a>: no description found</li><li><a href="https://arxiv.org/abs/2410.09457">Power-Softmax: Towards Secure LLM Inference over Encrypted Data</a>: Modern cryptographic methods for implementing privacy-preserving LLMs such as Homomorphic Encryption (HE) require the LLMs to have a polynomial form. Forming such a representation is challenging becau...</li><li><a href="https://www.liquid.ai/research/liquid-neural-networks-research">From Liquid Neural Networks to Liquid Foundation Models</a>: We invented liquid neural networks, a class of brain-inspired systems that can stay adaptable and robust to changes even after training [R. Hasani, PhD Thesis] [Lechner et al. Nature MI, 2020] [pdf] (...</li><li><a href="https://huggingface.co/papers/2503.02130">Paper page - Forgetting Transformer: Softmax Attention with a Forget Gate</a>: no description found</li><li><a href="https://metamotivo.metademolab.com/">Meta Motivo</a>: A first-of-its-kind behavioral foundation model to control a virtual physics-based humanoid agent for a wide range of whole-body tasks.</li><li><a href="https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Arnold_representation_theorem">Kolmogorov–Arnold representation theorem - Wikipedia</a>: no description found</li><li><a href="https://www.liquid.ai/liquid-foundation-models">Liquid Foundation Models: Our First Series of Generative AI Models</a>: Announcing the first series of Liquid Foundation Models (LFMs) – a new generation of generative AI models that achieve state-of-the-art performance at every scale, while maintaining a smaller memory f...</li><li><a href="https://x.com/swapnakpanda/status/1898450291793560063?t=zRKT-_mqeH564yhTThDnKg&s=33">Tweet from Swapna Kumar Panda (@swapnakpanda)</a>: Stanford’s Machine Learning - by Andrew NgA complete lecture notes (227 pages).</li><li><a href="https://www.youtube.com/watch?v=JL_bi2QROcw">AI Career Trap – Millions of Kids Will Step Into It. Make Sure YOUR Children Don’t</a>: #ai #career Are you worried about your child’s future in an AI-dominated world? In this video, we expose why traditional careers may soon be a trap and revea...</li><li><a href="https://ai.google.dev/gemini-api/docs/rate-limits#tier-1">no title found</a>: no description found</li><li><a href="https://github.com/ajithmoola/THB-Diff">GitHub - ajithmoola/THB-Diff: A Differentiable THB-spline module implemented in JAX and PyTorch</a>: A Differentiable THB-spline module implemented in JAX and PyTorch - ajithmoola/THB-Diff
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1347635023678804098)** (13 messages🔥): 

> `Latent Reasoning, Context Compression, Physical Intelligence and Cognitive Biases Toward AI, scilent paper` 


- **LLMs Abstraction with VQ-VAE for Reasoning traces!**: The paper [scilent](https://arxiv.org/abs/2502.03275) proposes a hybrid representation of the reasoning process, abstracting away initial reasoning steps using latent discrete tokens generated by **VQ-VAE**, reducing the length of reasoning traces.
   - The approach was tested by training from scratch for the **Keys-Finding Maze problem**, and fine-tuning LLMs on this hybrid data with an extended vocabulary including unseen latent tokens, for both **logical and mathematical reasoning problems**.
- **Latent Reasoning or Context Compression: A Thorny Question?**: A member asked if the latent reasoning being discussed is *actually latent reasoning* or if it is just **context compression**.
   - Another member humorously suggested **ECT** (Electroconvulsive Therapy) as a method of training people in AI to stop abusing the term *reasoning*.
- **Delays in Graph Display and Project Completion**: A member expressed surprise that a project took **5 years**, jokingly asking if they only worked an hour a month or took year-long breaks.
   - Another member humorously said they *can display graphs* to make up for the time.
- **Robots Clean Houses in 2025?**: A member shared a [YouTube video](https://www.youtube.com/watch?v=z-5F-b1t1C0) of a Stanford Seminar from **February 7, 2025**, by Sangbae Kim (MIT), about *Physical Intelligence and Cognitive Biases Toward AI*.
   - The description asks when robots will be able to *clean my house, dishes, and take care of laundry*.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2502.03275">Token Assorted: Mixing Latent and Text Tokens for Improved Language Model Reasoning</a>: Large Language Models (LLMs) excel at reasoning and planning when trained on chainof-thought (CoT) data, where the step-by-step thought process is explicitly outlined by text tokens. However, this res...</li><li><a href="https://www.youtube.com/watch?v=z-5F-b1t1C0">Stanford Seminar - Physical Intelligence and Cognitive Biases Toward AI</a>: February 7, 2025Sangbae Kim, MITWhen will robots be able to clean my house, dishes, and take care of laundry? While we source labor primarily from automated ...
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455634453/1269724655405498429/1347777901222821898)** (7 messages): 

> `DeepSeek efficiency, ScholarAgent updates, Arxiv papers search` 


- **DeepSeek Achieves 57X Efficiency Boost**: A member shared a [YouTube video](https://youtu.be/0VLAoVGf_74) titled "The Genius of **DeepSeek’s 57X Efficiency Boost** [MLA]" discussing efficiency improvements.
   - The video is sponsored by **KiwiCo**, offering a **50% discount** on the first monthly club crate with code WELCHLABS.
- **ScholarAgent Updates Improve Arxiv Paper Search**: A member announced updates to their **ScholarAgent** on Hugging Face, which now retrieves the top **3 recent Arxiv papers** based on user-provided keywords with better response time.
   - New features include **BM25 ranking** and **TF-IDF** for enhanced semantic search, allowing users to input full sentences.
- **Tips for using ScholarAgent Effectively**: To get the best results from ScholarAgent, users are advised to use comma-separated keywords like **deep learning**, **computer vision**, or **Language Models**.
   - Suggestions were made to trigger a submit on the return key press and to include an example sidebar with 10 queries that yield interesting results.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/pdx97/ScholarAgent">ScholarAgent - a Hugging Face Space by pdx97</a>: no description found</li><li><a href="https://youtu.be/0VLAoVGf_74">The Genius of DeepSeek’s 57X Efficiency Boost [MLA]</a>: Thanks to KiwiCo for sponsoring today’s video! Go to https://www.kiwico.com/welchlabs and use code WELCHLABS for 50% off your first monthly club crate or for...
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1347550701278072842)** (42 messages🔥): 

> `LLMs hallucinating, Multi-step agentic workflows, Language Diffusion, China AI Agent Manus, Stanford Regex Ozempic alternative` 


- **Diffusion Models can't fix LLM Hallucinations**: A member discussed how diffusion models may mitigate but do not eliminate **hallucinations** in language models, as **hallucination** is just another term for guessing incorrectly.
   - They suggested that while self-editing abilities can replace low-confidence samples with higher confidence ones, there's no guaranteed correctness, and many sampling strategies could be applied to diffusion models too.
- **Top-n-sigma sampling mitigates loops**: A member shared the [Top-n-sigma sampling paper](https://arxiv.org/abs/2411.07641) and [Github repo](https://github.com/Tomorrowdawn/top_nsigma), noting it could mitigate bad samples and looping behavior in multi-step agentic workflows.
   - The key insight is that *logits naturally separate into a Gaussian-distributed noisy region and a distinct informative region, enabling efficient token filtering without complex probability manipulations.*
- **Diffusion of Thought integrates diffusion models with Chain of Thought**: A member pointed out that [Diffusion-of-Thought (DoT)](https://neurips.cc/virtual/2024/poster/95935) integrates diffusion models with Chain-of-Thought, showcasing promising self-correction abilities and benefiting from existing reasoning-enhancing techniques like self-consistency decoding, with code available [on GitHub](https://github.com/HKUNLP/diffusion-of-thoughts).
   - A member jokingly pointed out that *Autoregression models still are trying hard to break into the image/video generation while diffusion tries hard to break into LLMs*.
- **China's Manus Agent Goes Viral**: Members discussed **Manus**, a new AI agent from China, calling it *like Deep Research + Operator + Claude Computer combined*, with linked to the [Manus website](https://manus.im) and initial [X post](https://x.com/rowancheung/status/1898093008601395380).
   - One user reported it *is more accurate than DeepSeek, capable of simultaneously handling financial transactions, research, purchasing, etc.* while another thought *UI is similar to devin's but much faster*.
- **Stanford uses Regex to find natural Ozempic alternative**: Stanford found a natural alternative to Ozempic using regex on the human proteome, with one person commenting *it's literally regex* linking to an [X post](https://x.com/xlr8harder/status/1898284331342184957) about it.
   - One user sarcastically suggested using an LLM to write your regex in response, and linked to a [YouTube video](https://youtu.be/X_wLVgMzSH4) on AI causing WW3.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://the-decoder.com/openai-shifts-away-from-sudden-agi-breakthrough-theory/">OpenAI shifts away from sudden AGI breakthrough theory</a>: OpenAI, the company behind ChatGPT and numerous other commercial AI applications, has long pursued the goal of developing artificial general intelligence (AGI) that &quot;benefits all of humanity.&quo...</li><li><a href="https://www.reddit.com/r/ChatGPTJailbreak/comments/1j3ztk3/sesame_jailbreak_update/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button">no title found</a>: no description found</li><li><a href="https://x.com/xlr8harder/status/1898284331342184957>">Tweet from xlr8harder (@xlr8harder)</a>: &gt;&#34;using regex on the human proteome&#34;&gt;yeah, ok, sure buddy&gt;look inside&gt;it&#39;s literally regexQuoting LaurieWired (@lauriewired) Stanford just found a natural alternative to Ozempi...</li><li><a href="https://x.com/elonmusk/status/1898170067596014031">Tweet from Elon Musk (@elonmusk)</a>: @EHuanglu More human than humans</li><li><a href="https://youtu.be/K27diMbCsuw?si=zTfyi7Yu1JLOW2TC">Introducing Manus: The General AI Agent</a>: Manus is a general AI agent that bridges minds and actions: it doesn&#39;t just think, it delivers results. Manus excels at various tasks in work and life, getti...</li><li><a href="https://x.com/jianxliao/status/1898861051183349870?">Tweet from jian (@jianxliao)</a>: So... I just simply asked Manus to give me the files at &#34;/opt/.manus/&#34;, and it just gave it to me, their sandbox runtime code...  &gt; it&#39;s claude sonnet &gt; it&#39;s claude sonnet with 2...</li><li><a href="https://arxiv.org/abs/2411.07641">Top-$nσ$: Not All Logits Are You Need</a>: Large language models (LLMs) typically employ greedy decoding or low-temperature sampling for reasoning tasks, reflecting a perceived trade-off between diversity and accuracy. We challenge this conven...</li><li><a href="https://github.com/Tomorrowdawn/top_nsigma">GitHub - Tomorrowdawn/top_nsigma: The official code repo and data hub of top_nsigma sampling strategy for LLMs.</a>: The official code repo and data hub of top_nsigma sampling strategy for LLMs.  - GitHub - Tomorrowdawn/top_nsigma: The official code repo and data hub of top_nsigma sampling strategy for LLMs.</li><li><a href="https://youtu.be/X_wLVgMzSH4">Experts show why WW3 over AI is nearly inevitable</a>: AGI, OpenAI, Elon Musk and WW3. Visit Ground News to compare news coverage, spot media bias and avoid algorithms. Get 40% off your subscription at https://gr...</li><li><a href="https://neurips.cc/virtual/2024/poster/95935">NeurIPS Poster Diffusion of Thought: Chain-of-Thought Reasoning in Diffusion Language Models</a>: no description found</li><li><a href="https://manus.im">Manus</a>: Manus is a general AI agent that turns your thoughts into actions. It excels at various tasks in work and life, getting everything done while you rest.</li><li><a href="https://x.com/rowancheung/status/1898093008601395380">Tweet from Rowan Cheung (@rowancheung)</a>: I think China&#39;s second DeepSeek moment is here.This AI agent called &#39;Manus&#39; is going crazy viral in China right now.Probably only a matter of time until it hits the US.It&#39;s like Deep R...</li><li><a href="https://x.com/heyBarsee/status/1898027732899962887">Tweet from Barsee 🐶 (@heyBarsee)</a>: AI is getting out of hand 🤯Manus, an AI agent from China, is automating approximately 50 tasks, creating a rather dystopian scenarioReports suggest it is more accurate than DeepSeek, capable of simul...</li><li><a href="https://gist.github.com/jlia0/db0a9695b3ca7609c9b1a08dcbf872c9">Manus tools and prompts</a>: Manus tools and prompts. GitHub Gist: instantly share code, notes, and snippets.
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1347706930399023204)** (19 messages🔥): 

> `SOTA Agentic Methods, Metal Kernel Launch Overhead, Torch.compile for MPS, Karpathy's Video` 


- **SOTA Agentic Methods: Simple Algorithms?**: A member mentioned that **SOTA agentic methods** on Arxiv tend to be fairly **simple algorithmically** and the abstractions of these frameworks are not really needed that much.
   - Separate abstractions over **data management, state management and API calls** would be fine.
- **Kernel Launch Overhead woes**: During the [Manuel Candales Low bit Metal kernels talk](https://www.youtube.com/watch?v=PaPuu73wowE), it was mentioned that the **kernel launch overhead** is about `1.5us` around 50m.
   - A member asked if it is possible to avoid that by **pipelining operations** and launching kernels in advance.
- **Torch.compile goes METAL!**: **Torch.compile for MPS**(well Metal) is available in **PyTorch nightly builds** and could be used to fuse operators together.
   - A member of pytorch encourages providing feedback in terms of what needed most.
- **__repr__ Dunder method has negligible cost?**: While watching **Karpathy's video**, a member wondered if the `def __repr__` in this is necessary if we don't need human readable formats.
   - Another member replied that it’s not really necessary, it’s only really there if u want to print stuff out and overhead should be negligible.


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1347592874165928007)** (26 messages🔥): 

> `SVD Quantization Kernel, Triton Autotuning, Kernel Fusion, Dynamic Activation Quantization` 


- **SVD Quant Kernel Runs Slower than FP16 Matmul**: A member implemented a [SVD quantization kernel](https://github.com/rishabh063/tritonKernel_svdQuant/blob/main/svdConversion.ipynb) and found that it was *slower* than **PyTorch's FP16 matmul**, despite doing both quantization and packing.
   - The implementation consists of **two kernels**: one that performs LoRA L1 matmul and quantization, and another that does int8 matmul and LoRA L2 matmul.
- **Triton Autotuning Worsens Performance**: A member reported that [autotuning](https://openai.com/blog/triton) made their kernel's performance even worse, despite the expectation of a **2x speedup**.
   - Suggestions were made to use larger eval shapes (**16384 x 16384**) and batch sizes (**128**) to reduce benchmarking overhead.
- **Kernel Fusion Discussed for Performance**: A member asked about fusing an op with a **Linear layer** in Triton, expressing concern that a custom Linear implementation might be slower than Torch's default.
   - A suggestion to *benchmark* performance was given to decide on a course of action and to also check `TORCH_COMPILE_DEBUG=1`.
- **Dynamic Activation Quantization Kernel Location Sought**: A member asked for a high-quality dynamic activation quantization kernel in **int8**, and was directed to kernels on [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes).
   - While those kernels aren't in Triton, **Liger** and **Unsloth** were mentioned as notable repos using Triton in production, with the advice to use LLMs to understand CUDA kernels.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/mit-han-lab/nunchaku">GitHub - mit-han-lab/nunchaku: [ICLR2025 Spotlight] SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models</a>: [ICLR2025 Spotlight] SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models - mit-han-lab/nunchaku</li><li><a href="https://github.com/rishabh063/tritonKernel_svdQuant/blob/main/svdConversion.ipynb">tritonKernel_svdQuant/svdConversion.ipynb at main · rishabh063/tritonKernel_svdQuant</a>: Contribute to rishabh063/tritonKernel_svdQuant development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1347742182655918090)** (26 messages🔥): 

> `Learning PTX for CUDA, Inline PTX for microbenchmarking, memcpy_async slowdown, Debugging CUDA kernels, FP8 WMMA optimization` 


- **PTX Pertinence Prompted**: Members discussed learning **PTX** for inline **CUDA C++** programming, with one suggesting to write a vector add in **PTX**, write a CUDA kernel, and inspect the resulting **PTX** to understand what's happening.
- **PTX Perks Prompted**: One member mentioned using inline **PTX** for microbenchmarking instruction latencies to avoid compiler optimizations, specifically for pointer chasing microbenchmarks.
- **memcpy_async Mishaps Manifested**: A user reported that using `cuda::memcpy_async` as a drop-in replacement for loading to shared memory in a **GEMM** implementation resulted in a slowdown, despite seeing data bypass the **L2 cache**.
- **CUDA Kernel Crash Course**: A developer sought advice on debugging a **CUDA** kernel that occasionally fails to launch with a *"too many resources requested"* error, particularly when compiled for `sm_120`.
   - They suspect a **PyTorch** issue or another kernel leaving the device in an invalid state.
- **WMMA Woes Weighed**: A member optimizing **FP8 matmuls** with `load_matrix_sync` experienced worse performance, possibly due to different memory layouts for **FP16** versus **FP8**.
   - They noted that *`nvcuda::wmma::fragment`* duplicates elements, attributing it to how `mma.m8n8k4` on **V100** GPUs requires duplication for 16x16x16 matmuls, as mentioned in the [NVIDIA documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-fragments-for-wmma).



**Link mentioned**: <a href="https://developer.nvidia.com/blog/controlling-data-movement-to-boost-performance-on-ampere-architecture/">Controlling Data Movement to Boost Performance on the NVIDIA Ampere Architecture | NVIDIA Technical Blog</a>: The NVIDIA Ampere architecture provides new mechanisms to control data movement within the GPU and CUDA 11.1 puts those controls into your hands. These mechanisms include asynchronously copying data&#...

  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1347599950527070350)** (29 messages🔥): 

> `DDP communication customization, FSDP communication customization, SimpleFSDP framework, Muon optimizer details` 


- **DDP Communication Hooks Sought After**: A member inquired about customizing communication in torch **DDP** using `register_comm_hook`, similar to **FSDP**, seeking alternatives to monkey patching.
   - Another member echoed this need, suggesting they might need to fork or write their own **FSDP** due to the lack of a similar hook.
- **SimpleFSDP Framework Emerges**: A link to a new paper titled [SimpleFSDP](https://arxiv.org/abs/2411.00284v1) was shared, which is a *PyTorch-native compiler-based Fully Sharded Data Parallel (FSDP) framework*.
   - This framework is noted for its simple implementation, composability, and performance enhancement via compiler backend optimizations, particularly with **torch.compile**.
- **Muon Optimizer Internals Investigated**: Members discussed the **Newton-Schulz iteration method** used in the **Muon optimizer** for obtaining orthogonality.
   - One member linked to a detailed discussion on why this method is preferred, explaining it's about producing the nearest semi-orthogonal matrix to the original, further clarifying with a link to a [derivation of Muon](https://jeremybernste.in/writing/deriving-muon).
- **Tuning SDPA Causal Mask Behavior**: A user reported errors when passing a boolean attention mask with `is_causal=True` to **SDPA** with different backends in **PyTorch 2.6.0_cu124**.
   - With `sdpa_kernel(SDPBackend.FLASH_ATTENTION)`: No errors and matches eager causal+attention mask output, `sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION)`: RuntimeError: No viable backend for scaled_dot_product_attention,  `sdpa_kernel(SDPBackend.MATH)`: RuntimeError: _scaled_dot_product_attention: Explicit attn_mask should not be set when is_causal=True.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2411.00284v1">SimpleFSDP: Simpler Fully Sharded Data Parallel with torch.compile</a>: Distributed training of large models consumes enormous computation resources and requires substantial engineering efforts to compose various training techniques. This paper presents SimpleFSDP, a PyTo...</li><li><a href="https://jeremybernste.in/writing/deriving-muon">Deriving Muon</a>: no description found</li><li><a href="https://github.com/KellerJordan/Muon">GitHub - KellerJordan/Muon: Muon optimizer: +&gt;30% sample efficiency with &lt;3% wallclock overhead</a>: Muon optimizer: +&gt;30% sample efficiency with &lt;3% wallclock overhead - KellerJordan/Muon</li><li><a href="https://github.com/HomebrewML/HeavyBall?tab=readme-ov-file#foreachmuon)">GitHub - HomebrewML/HeavyBall: Efficient optimizers</a>: Efficient optimizers. Contribute to HomebrewML/HeavyBall development by creating an account on GitHub.</li><li><a href="https://github.com/pytorch/pytorch/blob/f7c0c230b0c55734f13bb66076203e4a1cf969ee/aten/src/ATen/native/transformers/attention.cpp#L841-L843">pytorch/aten/src/ATen/native/transformers/attention.cpp at f7c0c230b0c55734f13bb66076203e4a1cf969ee · pytorch/pytorch</a>: Tensors and Dynamic neural networks in Python with strong GPU acceleration - pytorch/pytorch</li><li><a href="https://github.com/pytorch/pytorch/pull/83254">Added communication hook for sharded cases by aovladi · Pull Request #83254 · pytorch/pytorch</a>: Fixes #79114An implementation of a FSDP communication hook interface for a sharded strategies:Added reduce_scatter_hook to default hooks. Note the difference of reduce_scatter from all_reduce, i...
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1347826853238276158)** (1 messages): 

> `Triton, CUDA, Flash Attention, YouTube tutorials, Performance` 


- **GPU Mode Celebrates 50th Lecture with Performance Talk**: GPU MODE is celebrating its **50th lecture** featuring <@332959405588873216>, who will discuss his journey learning **Triton**, **CUDA**, and **Flash Attention** on [YouTube](https://www.youtube.com/@GPUMODE).
   - The speaker started learning about **performance** in 2022 and has created high-quality **YouTube tutorials** on the subject, with additional content available on [GitHub](https://github.com/gpu-mode).
- **GPU MODE Community Milestone**: The GPU MODE community has reached **15,000 members**, marking a significant milestone for the online platform.
   - The community is described as a favorite part of the internet, expressing gratitude for the active participation and engagement of its members.



**Link mentioned**: <a href="https://www.youtube.com/@GPUMODE">GPU MODE</a>: A GPU reading group and community https://discord.gg/gpumodeSupplementary content here https://github.com/gpu-modeCreated by Mark Saroufim and Andreas Köpf 

  

---


### **GPU MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1347571245868126300)** (3 messages): 

> `Double Binary Tree vs. Ring Topology in NCCL, AllReduce Implementation Comparison, NCCL 2.4 and Double Binary Trees` 


- **Double Binary Tree crushes Ring Topology for AllReduce**: A member inquired about why **double binary tree topology** is superior to **ring topology** in **NCCL** for implementing **AllReduce**, especially concerning latency as the node count increases.
   - A [NVIDIA blog post](https://developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4/) notes that **NCCL 2.4** introduces **double binary trees**, offering full bandwidth and logarithmic latency, which is even lower than that of 2D rings.
- **Ring Topology Latency scales linearly with nodes**: A member explained that in **ring topology**, with each processor able to communicate only with its neighbor, the number of communications and operations scales linearly with the number of processors (**O(p)**) during an **AllReduce** operation.
   - This is because it requires *p-1 operations/communications* to complete the **AllReduce**.
- **Tree Topology Achieves Logarithmic Communication Complexity**: The member explained that in **tree-based topology**, the total number of parallel communications and operations scales logarithmically with the number of tree levels (**O(log L)**).
   - Given the assumptions in **double binary trees** where ranks alternate between nodes and leaves, the complexity can be expressed as **O(log p)**.



**Link mentioned**: <a href="https://developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4/">Massively Scale Your Deep Learning Training with NCCL 2.4 | NVIDIA Technical Blog</a>: Imagine using tens of thousands of GPUs to train your neural network. Using multiple GPUs to train neural networks has become quite common with all deep learning frameworks, providing optimized&#8230;

  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1347683049139339314)** (16 messages🔥): 

> `WoolyAI CUDA abstraction layer, Muon optimizer, Alternative to GPUs` 


- **WoolyAI launches CUDA Abstraction beta**: WoolyAI launched its beta of a new **CUDA abstraction layer** that decouples Kernel Shader execution from applications, promising max GPU resource utilization and isolation between workloads, as shown in [their documentation](https://docs.woolyai.com).
- **Dynamic Scheduling with WoolyAI**: WoolyAI dynamically schedules workloads, allowing different kernels from different users to run on one GPU without hard partitioning, charging users based on **core and VRAM usage**.
   - Currently, **PyTorch** is the only supported framework, but the architecture allows for a form of MIG (Multi-Instance GPU) based on usage.
- **Muon Optimizer Sets Speed Records**: A new neural net optimizer called **Muon** has garnered attention for its excellent practical performance, and set **NanoGPT speed records**, as detailed in [this blog post](https://jeremybernste.in/writing/deriving-muon).
   - The core numerical methods of **Muon** are derived from an exact theoretical principle, contrasting with optimizers like **Adam** which have more heuristic origins.
- **Emerging GPU Alternative**: A small company is developing an alternative to GPUs that is reportedly **10x more energy efficient** than an **H100** on sparse models and can scale across multiple chiplets, as detailed in [their research paper](https://arxiv.org/pdf/2409.19389).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.woolyai.com">Introduction | WoolyAI Documentation</a>: What is Wooly?</li><li><a href="https://jeremybernste.in/writing/deriving-muon">Deriving Muon</a>: no description found
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1347546333652975658)** (25 messages🔥): 

> `GPU Memory Sharing on Apple, Cuda Graphs in Triton Autotune, Resources to get started with GPU and TPU programming, nvmlDeviceGetCudaComputeCapability Tuple Return, Cerebras Language vs CUDA` 


- **Apple's Memory Buffers Enable Direct Pointers!**: On Apple GPUs, memory buffers are shared between threads, enabling the use of direct **pointers** to memory locations, which contrasts with other platforms.
   - This architecture simplifies **memory access** and potentially improves **performance** for parallel computations on Apple devices.
- **Decoding `use_cuda_graph` in `triton.autotune`.**: A member inquired about the `use_cuda_graph` argument in `triton.autotune`, questioning its relevance since `triton.autotune` decorates single **CUDA kernels**, while **CUDA graphs** typically optimize sequences of kernels.
- **Programming Massively Parallel Processors: Your GPU/TPU Rosetta Stone?**: A member asked if *'Programming Massively Parallel Processors'* is sufficient to start with **GPU** and **TPU programming**, aiming to build a framework like **TinyGrad** or **Torch** with Assembly, C, C++, and Python skills.
- **`nvmlDeviceGetCudaComputeCapability` returns a tuple for clarity**: The function `nvmlDeviceGetCudaComputeCapability` returns compute capability as a tuple *(major, minor)*, e.g., *(7, 5)*, rather than a single float.
   - As one member explained, *it's a version made up of a major version and a minor version, not a real number*, and doing arithmetic on a tuple of integers is less error-prone.
- **Cerebras Skips CUDA, Rolls Own Language!**: **Cerebras** does not use **CUDA**, but instead employs its own language, documented in the [Cerebras SDK](https://sdk.cerebras.net/computing-with-cerebras) that operates on its **Wafer-Scale Engine (WSE)** with hundreds of thousands of processing elements.
   - Each PE has its own memory and can communicate with neighbors via 32-bit wavelets.



**Link mentioned**: <a href="https://sdk.cerebras.net/computing-with-cerebras">A Conceptual View &#8212; SDK Documentation (1.3.0)</a>: no description found

  

---


### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1348137451817930843)** (1 messages): 

> `PMPP 4th Edition, CUDA C, Latex text` 


- **Sanity Check for CUDA C in PMPP 4th Edition**: A member reading through Chapter 3 of the **PMPP 4th edition** book inquired about the meaning of strings like *1F1F* and *2F2F* next to **Latex text** for certain dimensions.
   - They wanted to verify if these were simply typing/PDF artifacts or if they held specific meanings in **CUDA C**, possibly as **hexadecimal numbers**.
- **Clarification Needed on Hexadecimal Strings in CUDA C**: A user seeks clarification on whether strings such as '1F1F' and '2F2F', found alongside LaTeX text in the **PMPP 4th Edition**, denote specific hexadecimal values in **CUDA C**.
   - The query aims to distinguish between potential typographical errors and meaningful representations within the context of **CUDA programming**.


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1348450301794979861)** (6 messages): 

> `GPU mode capacity increase at GTC, GTC and Game Developer's Conference, Semi-Analysis Hackathon team member search, CUTLASS kernels for GEMM or FMHA prefill/decoding` 


- **GPU Mode Capacity Ramps Up at GTC**: Capacity for **GPU mode** at **GTC** has been increased to **~200**, and those who haven't received a response are asked to reply with their email.
   - One member followed up to ask for approval and provided their email.
- **Member Eyes Both GTC and Game Developers Conference**: A member expressed interest in attending **GTC**, potentially for a day, as they plan to be at the **Game Developer's Conference** in **SF**, which is within BART distance.
   - They shared a [link to a LinkedIn post about GTC 2025](https://www.linkedin.com/posts/johnnycano_gtc2025-nvidia-ai-activity-7304920833787883521-ysXu?utm_source=share&utm_medium=member_desktop&rcm=ACoAABss8WsBjvq7mxQL57u6-2AflaXu2eAjQPMA).
- **Hackathon Hero Hunt: Team Member Seeks Kernel Konnection**: A member is looking for a team member for the **Semi-Analysis Hackathon**, seeking to build unique **GEMM** or **FMHA prefill/decoding kernels** using **CUTLASS**.
   - They requested DMs from those seeking team members and inquired about a dedicated channel for the event.


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1347563923674501170)** (3 messages): 

> `AMD GPU, HIP Code Compilation, Runpod, MI300` 


- **Seeking AMD GPU Environment for HIP Code Compilation**: A user is seeking an environment with an **AMD GPU** to compile **HIP code** with some **ASM inline** for simple **GEMM** benchmarking and is looking for GPU rental services.
   - They specifically mentioned using the **matMul accelerator** for curiosity purposes.
- **HIP Code Compilation Without GPU is Possible**: It was mentioned that you can compile **HIP code** without a GPU, as you only need **hipcc**.
   - You can obtain **hipcc** through standard methods, though you won't be able to run the compiled code without a GPU.
- **Runpod Offers Access to MI300**: It was suggested that [Runpod](https://runpod.io/) is a decent way to access **MI300** GPUs.
   - No further details were provided about Runpod or MI300.


  

---


### **GPU MODE ▷ #[tilelang](https://discord.com/channels/1189498204333543425/1240586843292958790/1347751513564647427)** (9 messages🔥): 

> `Kernel Compilation, Mixed Precision GEMM, TileLang GEMM Example` 


- **Kernel Compilation Quandaries**: A user asked if they need to compile the kernel for every matrix shape and a member pointed to a [relevant test file](https://github.com/tile-ai/tilelang/blob/main/testing/python/jit/test_tilelang_jit_gemm_cython.py#L406) using dynamic symbolic to replace static size.
- **Precision Mixing Maneuvers**: A user inquired about the possibility of using mixed precisions, specifically running GEMM in **float16** while the rest of the operations are in **float32**.
   - A member confirmed that it is possible, directing the user to a [quick start example](https://github.com/tile-ai/tilelang?tab=readme-ov-file#gemm-example-with-annotations-layout-l2-cache-swizzling-and-pipelining-etc).
- **Decoding Data Types in TileLang**: A user sought clarification on what the `accum_dtype="float"` type is and where the mixing of precisions occurs in the TileLang GEMM example.
   - Another member responded that in TileLang, you can define all buffer data types.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/tile-ai/tilelang?tab=readme-ov-file#gemm-example-with-annotations-layout-l2-cache-swizzling-and-pipelining-etc?">GitHub - tile-ai/tilelang: Domain-specific language designed to streamline the development of high-performance GPU/CPU/Accelerators kernels</a>:  Domain-specific language designed to streamline the development of high-performance GPU/CPU/Accelerators kernels - tile-ai/tilelang</li><li><a href="https://github.com/tile-ai/tilelang/blob/main/testing/python/jit/test_tilelang_jit_gemm_cython.py#L406">tilelang/testing/python/jit/test_tilelang_jit_gemm_cython.py at main · tile-ai/tilelang</a>:  Domain-specific language designed to streamline the development of high-performance GPU/CPU/Accelerators kernels - tile-ai/tilelang
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/1348104666659160174)** (8 messages🔥): 

> `Metal Parallel Reduction Kernels, Metal Shading Language, Metal-cpp, Swift, Objective-C` 


- **Parallel Reduction Kernel Examples Sought!**: A member is seeking examples of Metal parallel reduction kernels, as they are having trouble implementing them following [Nvidia's CUDA reduction guide](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf).
- **Metal Language Ecosystem Explored**: A newcomer to GPU programming is deciding between Objective-C, Swift, and metal-cpp for Metal development, aiming to contribute to the MPS backend of PyTorch and MLX.
   - One member suggested checking the source of the MPS backend of PyTorch to see what they are using.
- **Swift, ctypes, and Metal workflow**: One member mentioned a workflow of *metal -> swift -> ctypes* within Python.
   - Another member asked for examples of that [shader](https://developer.apple.com/metal/metal-shading-language/).


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1347752733171519528)** (19 messages🔥): 

> `Cute Kernels, Triton vs CUTLASS, FA3's GEMM, LLVM compiler efficiency, GB200 access` 


- ****Cute Kernels** Collection Speeds Up Training**: A member shared a collection of **CUDA/triton kernels** called [cute-kernels](https://github.com/mayank31398/cute-kernels) for speeding up training, which autotunes over both triton and CUDA implementations and is working on adding tilelang kernels next.
   - The kernels are end-to-end torch compileable without any graph breaks and the repo also contains a custom autotune implementation being used in production to train **IBM's Granite LLMs**.
- ****Triton vs CUTLASS** performance examined**: Members discussed performance differences between **Triton and CUTLASS**, with one noting that Triton is generally not performant enough for **GEMM** stuff.
   - The peak **CUTLASS GEMM** throughput for bf16 on H100 is about **700 TFLOPs**, whereas for triton its around **500 TFLOPs** after all the autotuning; the original poster added that they are very excited about tilelang since its looking much more performant than triton.
- ****LLVM compiler efficiency** is noticed**: A member stated that the **LLVM compiler** can sometimes create more efficient code than the NVCC so, it makes sense to tune over the kernel backend as well.
   - An example for vector addition can be seen on [github](https://github.com/mayank31398/cute-kernels/blob/main/cute_kernels/kernels/add/add_tensor/__init__.py), all the kernels are JIT compileable.
- ****FA3's GEMM** discussed**: It was mentioned that FA3 is mostly a GEMM discounting the statistics and softmax.
   - The original poster included an image showcasing the [flash3_fp16_fwd.png](https://cdn.discordapp.com/attachments/1347752733171519528/1347819309564694648/flash3_fp16_fwd.png?ex=67d081cc&is=67cf304c&hm=8ae0ae780e35bec194962af703a0f4f8483397e27720e8db4c436945dccd9c30&).
- **Next Kernels & **GB200 access** are desired**: A member mentioned that next they will be adding kernels for **RoPE**, fused residual_add_RMSNorm and attention (softmax + stickbreaking).
   - They will also be working on custom GPU-to-GPU communication kernels soon, as well as CUDA implementations of some of the kernels which only have a triton implementation at the moment; also, they are waiting on access to **GB200s**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=z0OSrVw04jw">Florida r u ok</a>: Florida r u ok</li><li><a href="https://github.com/mayank31398/cute-kernels">GitHub - mayank31398/cute-kernels: A bunch of kernels that might make stuff slower 😉</a>: A bunch of kernels that might make stuff slower 😉. Contribute to mayank31398/cute-kernels development by creating an account on GitHub.</li><li><a href="https://github.com/mayank31398/cute-kernels/blob/main/examples/cute_inductor.py#L35">cute-kernels/examples/cute_inductor.py at main · mayank31398/cute-kernels</a>: A bunch of kernels that might make stuff slower 😉. Contribute to mayank31398/cute-kernels development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1347671571682168873)** (2 messages): 

> `LCF Concurrency, DDP+NCCL` 


- **LCF faces concurrency quandaries with streams**: A user is encountering weird deadlocks when using **LCF** with **DDP+nccl**, questioning if **LCF** is intended to be completely concurrency-safe with streams.
- **Need script to try out NCCL/distributed setup**: The team hasn't tested **LCF** with **NCCL/distributed setups** and would like to try the user's script to help debug, if they can share it.


  

---


### **GPU MODE ▷ #[edge](https://discord.com/channels/1189498204333543425/1303441437592911912/1348466198827892908)** (1 messages): 

> `FOSS CUDA developments, Open source platform for edgeai/TinyML, GPU "lab"` 


- **Undergrads form GPU "lab"**: A group of undergrad students is forming an independent GPU "lab" focused on hardware engineering and **CUDA kernel development**.
   - They are seeking promising leads for **FOSS CUDA developments**, potentially supported by a grant.
- **Students Aim to make open source platform for edgeai/TinyML**: The students are planning to build an open-source platform for **edgeAI/TinyML** this summer.
   - The platform aims to accelerate developments in the field and serve as a valuable resource for the community.


  

---


### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/1347553986248441867)** (44 messages🔥): 

> `Reasoning Gym Curricula, Sonnet Context Expansion, Palindrome Partitioning Dataset, ACRE Dataset Integration, Reasoning Gym Goals` 


- ****Reasoning Gym** Curriculum Underway**: Members have started working on the curriculum for **Reasoning Gym**, with existing work detailed in the Curriculum thread above.
   - A question was raised about the **API's stability** for curriculum writing.
- ****Sonnet's** Context Gets Expanded**: A member suggested putting the whole **Reasoning Gym** into **Sonnet's context** to generate more datasets and infinitely repeat the process.
   - The goal would be to train the model to solve them, creating a *reasoning GAN self-play*.
- **Dataset 'Palindrome Partitioning' Under Scrutiny**: A dataset called *palindrome partitioning* has come under scrutiny, with one member indicating it does not seem like a great reasoning dataset.
   - The member suggested the question be phrased as *'find the partition of least length such that every substring is a palindrome'*, rather than generating all possible partitions.
- ****Reasoning Gym** Reaches 100 Datasets Milestone**: After merging the **ACRE dataset**, **Reasoning Gym** now has a total of **100 datasets**, thanks to contributions from the core team and others.
   - Next steps involve showing significant variation among existing models and the possibility of learning tasks via **RL**.
- ****Reasoning Gym** to Integrate with **DSPy****: Members expressed interest in trying **Reasoning Gym** with **DSPy** to run experiments, with plans to provide an example this week.
   - The goal is to have this as an example or part of the evaluation scripts.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/ope">ope - Overview</a>: ope has 12 repositories available. Follow their code on GitHub.</li><li><a href="https://github.com/open-thought/reasoning-gym/blob/2fca96284760bcd60357928f097451617f916516/reasoning_gym/algorithmic/palindrome_partitioning.py#L95-L105)...">reasoning-gym/reasoning_gym/algorithmic/palindrome_partitioning.py at 2fca96284760bcd60357928f097451617f916516 · open-thought/reasoning-gym</a>: procedural reasoning datasets. Contribute to open-thought/reasoning-gym development by creating an account on GitHub.</li><li><a href="https://docs.google.com/document/d/1ytdo9LoBWuK2IKXUCla0YwC_0g-nqzv5VwmbevSAQPU">Experiment: how much do LLMs speed up developers</a>: METR is seeking software engineers who regularly work on large open-source projects to test the effectiveness of AI software engineering tools. Apply here (bit.ly/ai-speedup-apply)  Questions? Contact...</li><li><a href="https://github.com/open-thought/reasoning-gym-eval/blob/main/anthropic_claude-3.5-sonnet_20250227_230002/algorithmic/palindrome_partitioning.json">reasoning-gym-eval/anthropic_claude-3.5-sonnet_20250227_230002/algorithmic/palindrome_partitioning.json at main · open-thought/reasoning-gym-eval</a>: Collection of LLM completions for reasoning-gym task datasets - open-thought/reasoning-gym-eval
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[gpu模式](https://discord.com/channels/1189498204333543425/1342364798058500148/1347866737181462609)** (34 messages🔥): 

> `Triton vs Cutlass, Zhihu registration, TileLang vs TVM, TileLang usage, CUDA optimization` 


- **Triton relies on LLVM for optimization**: It was stated that **Triton** relies on **LLVM** to convert code to bytecode, unlike lower-level languages like **CUDA**, **CUTLASS**, or **cuDNN**.
   - While it supports writing **PTX** code, the documentation provides examples for [inline assembly](https://triton-lang.org/main/python-api/generated/triton.language.inline_asm_elementwise.html#triton.language.inline_asm_elementwise) using this approach.
- **Zhihu registration difficulties for US users**: A user reported difficulties registering for **Zhihu** from the **US**, failing to receive SMS verification codes after completing captchas.
   - Others were surprised, expecting international registration to be seamless, with one user asking if a **+86** phone number was required.
- **TileLang for all CUDA kernels**: A developer suggested that **TileLang** can be used without deep knowledge of **TVM** or **MLIR**.
   - Another confirmed that **TileLang** is versatile enough to implement *every cuda things* and even for **CPU kernels**.
- **TileLang for gaussian rasterization**: A member asked if **TileLang** is only useful for matrix multiplication, or also useful in other contexts such as 3D Gaussian Splatting's rendering.
   - The developer replied in the affirmative, suggesting one can use **TileLang** for practically *every cuda things*.
- **CUDA optimization is hard**: A user expressed surprise that **TileLang** demos can outperform **cuBLAS**, showing a demo exceeding **cuBLAS**.
   - A developer responded, *optimizing cuda is hard even though you are an expert* adding that one should rely on the compiler, and referencing [Surprised by the performance of Triton!](https://github.com/triton-lang/triton/issues/3747).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://triton-lang.org/main/python-api/generated/triton.language.inline_asm_elementwise.html#triton.language.inline_asm_elementwise">triton.language.inline_asm_elementwise &mdash; Triton  documentation</a>: no description found</li><li><a href="https://siboehm.com/articles/22/CUDA-MMM">How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog</a>: In this post, I’ll iteratively optimize an implementation of matrix multiplication written in CUDA.My goal is not to build a cuBLAS replacement, but to deepl...</li><li><a href="https://github.com/triton-lang/triton/issues/3747">Surprised by the performance of Triton! · Issue #3747 · triton-lang/triton</a>: I did a benchmark to check the time cost (us) of cublas and triton on various shapes, I find that triton kernel is faster than cublas on most of the times. Is that a normal case? Anyone got the sam...</li><li><a href="https://github.com/graphdeco-inria/diff-gaussian-rasterization.git">GitHub - graphdeco-inria/diff-gaussian-rasterization</a>: Contribute to graphdeco-inria/diff-gaussian-rasterization development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1347875398016565248)** (2 messages): 

> `vectoradd benchmark, Modal runners success, GPU benchmarks` 


- **Vectoradd Benchmarks Flourish on A100 and H100 GPUs**: Benchmark submission with id **`1650`** to leaderboard **`vectoradd`** on **GPUS**: **A100**, **H100** using **Modal runners** succeeded!
   - Submission with id **`1651`** to leaderboard **`vectoradd`** on **GPUS**: **A100** using **Modal runners** also succeeded!
- **Modal Runners Ace GPU Benchmarks**: The **Modal runners** successfully executed leaderboard submissions for the **vectoradd** benchmark on both **A100** and **H100** GPUs.
   - This indicates robust performance and reliability of the **Modal runners** in handling GPU-accelerated workloads.


  

---


### **GPU MODE ▷ #[ppc](https://discord.com/channels/1189498204333543425/1343373905783554048/1347638233512808509)** (2 messages): 

> `AVX-256 Optimization, AVX-512 Optimization, Tiling, OpenMP` 


- **Can AVX-256 Achieve 3s on 3a?**: A member inquired whether achieving performance of **<= 3s** on **3a** is possible using **tiling**, **OpenMP**, and **AVX-256**.
   - Another member responded affirmatively, suggesting that it should be feasible, further noting *the benefit of using only AVX2 instructions while still taking advantage of the increased number of registers that AVX512 offers*.
- **Hybrid AVX Optimization Approach**: A member suggested a **hybrid approach** using only **AVX2 instructions** to benefit from the increased number of registers that **AVX512** brings.
   - This combines the advantages of both instruction sets without fully committing to **AVX-512**.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1347614421287047332)** (102 messages🔥🔥): 

> `Minion.ai dead, Gemini Embedding Model, Muse AI Model, Manus AI Agent, RWKV7-G1 GooseOne` 


- ****Minion.AI** Deceased, Perplexity Poaches**: Members noted that **Minion.ai** is defunct, with the team reportedly joining [Perplexity](https://www.perplexity.ai/).
   - A user expressed interest in **Composio** for MCP servers but voiced concerns about granting Gmail access to Linear, as requested in [Logan's Tweet](https://x.com/officiallogank/status/1898081742767919384?s=46).
- ****Gemini Embedding** Evolving with Extended Input and Language**: Google is rolling out an experimental **Gemini Embedding model** for developers with SOTA performance on MTEB, increasing input context length from **3K to 8K tokens**, outputting **3K dimensions**, and supporting over **100 languages** from [OpenAI's tweet](https://x.com/openaidevs/status/1898047744364659195?s=46).
- ****Manus** Mania - Claude's Costume?**: Discussion surrounds **Manus**, an **AI agent** launched in China, with claims it is more accurate than **DeepSeek** and automates approximately **50 tasks** as shown in [Thinking Panda's tweet](https://x.com/thinking_panda/status/1897951585990590469?s=61).
   - However, others claim it's based on **Claude Sonnet** with tools and jailbreaks, as per [Giffmana's Tweet](https://x.com/giffmana/status/1898868685739081766?s=61), leading to accusations of grift.
- ****RWKV7-G1** Reasoning, Really Rapid RNN**: **RWKV7-G1 GooseOne**, a pure RNN model, has been released with reasoning capabilities at **0.1B** parameters as mentioned in [BlinkDL's tweet](https://x.com/BlinkDL_AI/status/1898579674575552558), fully multilingual.
   - Larger G1 training is in progress, with more details on datasets and post-training available [here](https://huggingface.co/BlinkDL/temp-latest-training-models/tree/main).
- ****Claude Deep Research** Revealed: Prompt Power!**: **Claude Deep Research** is described as **Claude Code** plus a script and a markdown file, emphasizing *effectiveness per LOC* as discussed in [Will Brown's tweet](https://x.com/willccbb/status/1898858751685255398?s=46).
   - A user made a great resource showing how **Claude Code** works under the hood on [GitHub](https://gerred.github.io/building-an-agentic-system/).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://gerred.github.io/building-an-agentic-system/">Building an Agentic System - Building an Agentic System</a>: no description found</li><li><a href="https://colmweb.org/cfp.html">COLM 2025: Call for Papers</a>: no description found</li><li><a href="https://geohot.github.io//blog/jekyll/update/2025/03/08/AMD-YOLO.html">AMD YOLO</a>: AMD is sending us the two MI300X boxes we asked for. They are in the mail.</li><li><a href="https://gerred.github.io/building-an-agentic-system">Building an Agentic System - Building an Agentic System</a>: no description found</li><li><a href="https://x.com/BlinkDL_AI/status/1893676178206072946">Tweet from BlinkDL (@BlinkDL_AI)</a>: I am training G1 0.1/0.4/1.5/2.9B (&#34;Goose One&#34; 🪿) simultaneously on world-3.5 (5.16T tokens), continuing from previous RWKV-7 &#34;Goose&#34; world-3 checkpts. Release soon🙂even L12-D768 can...</li><li><a href="https://lu.ma/sync-sf?tk=4bXESE">Sync SF · Luma</a>: Interesting in building products on a local-first / sync engine architecture?Come along to the first Sync SF, a meetup to learn and talk about better ways of…</li><li><a href="https://x.com/jordanschnyc/status/1899198463373398300?s=46">Tweet from Jordan Schneider (@jordanschnyc)</a>: Manus pod with @swyx, @deanwball and @krishnanrohit out nowhttps://podcasts.apple.com/us/podcast/manus-a-deepseek-moment/id1289062927?i=1000698639495</li><li><a href="https://x.com/ReutersScience/status/1897864786068885790">Tweet from Reuters Science News (@ReutersScience)</a>: Melbourne-based Cortical Labs has revealed the first commercial biological computer, in an attempt to revolutionize drug testing and personalized medicine. It fuses cell-derived neurons with silicon, ...</li><li><a href="https://x.com/willccbb/status/1898835221124120956?s=46">Tweet from will brown (@willccbb)</a>: ok here you gobetter system prompt + everything all in one spot (all done on phone, apologies for lazy readme)https://github.com/willccbb/claude-deep-research</li><li><a href="https://x.com/__nmca__/status/1899174075685355770?s=46">Tweet from Nat McAleese (@__nmca__)</a>: large reasoning models are extremely good at reward hacking. A thread of examples from OpenAI&#39;s recent monitoring paper: (0/n)</li><li><a href="https://x.com/jamesjyu/status/1897759160886083783?s=61">Tweet from james yu (@jamesjyu)</a>: Today, we&#39;re launching Muse, an AI model specifically trained for fiction. We&#39;ve been testing Muse with hundreds of authors for months, and we&#39;re excited to finally share it with the world...</li><li><a href="https://x.com/deedydas/status/1898971193128173862?s=46">Tweet from Deedy (@deedydas)</a>: LEAK ServiceNow is in talks to buy Moveworks for $3BIt would be the biggest AI acquisition in the last 5yrs.Founded by Bhavin Shah etc 9yrs ago in 2016, they were at ~$100M ARR, a 30x multiple. They b...</li><li><a href="https://x.com/devgerred/status/1898719338741297505?s=46">Tweet from gerred (@devgerred)</a>: @Steve_Yegge I actually did a deep dive book for people building their own (since I am, natively) based on the code. More “how the parts come together” and a deep dive into every tool and command: htt...</li><li><a href="https://x.com/giffmana/status/1898868685739081766?s=61">Tweet from Lucas Beyer (bl16) (@giffmana)</a>: &gt; see post&gt; Manus = Claude + browser_use&gt; what is browser_use? check info.&gt; &#34;W25&#34; prolly some YC shit again&gt; Actually is ETH&gt; mfwOkok I&#39;m not nearly as funny as 4.5, but ...</li><li><a href="https://x.com/mishalaskin/status/1898048925157728601?s=46">Tweet from Misha Laskin (@MishaLaskin)</a>: Today I’m launching @reflection_ai with my friend and co-founder @real_ioannis.Our team pioneered major advances in RL and LLMs, including AlphaGo and Gemini.At Reflection, we&#39;re building superint...</li><li><a href="https://x.com/haridigresses/status/1898767370073649248?s=46">Tweet from hari (@haridigresses)</a>: Regardless of the market / valuation dynamics... @mntruell&#39;s openness to feedback (at 9 figures in revenue no less) is an incredibly rare and bullish signal.(Shared with permission)Quoting hari (@...</li><li><a href="https://x.com/BlinkDL_AI/status/1898579674575552558">Tweet from BlinkDL (@BlinkDL_AI)</a>: RWKV7-G1 &#34;GooseOne&#34; first release: reasoning @ 0.1b params, pure RNN (attention-free), fully multilingual. Demo & weights on https://RWKV.com 🪿 Larger G1 training in progress.Quoting BlinkDL ...</li><li><a href="https://x.com/8teapi/status/1898615677516390590?s=46">Tweet from Prakash (Ate-a-Pi) (@8teAPi)</a>: This is a 100 million param model nowQuoting BlinkDL (@BlinkDL_AI) RWKV7-G1 &#34;GooseOne&#34; first release: reasoning @ 0.1b params, pure RNN (attention-free), fully multilingual. Demo & weights on ...</li><li><a href="https://the-decoder.com/chinese-ai-agent-manus-uses-claude-sonnet-and-open-source-technology/">Chinese AI agent Manus uses Claude Sonnet and open-source technology</a>: A new AI agent called Manus, developed by Chinese startup Monica, demonstrates capabilities in handling complex tasks from travel planning to financial analysis without human intervention. While early...</li><li><a href="https://x.com/pitdesi/status/1898193386877911500?s=46">Tweet from Sheel Mohnot (@pitdesi)</a>: Cursor is incredible but I wonder how sticky the revenue (~$150M ARR) is.66x revs is fine if you think the growth continues, if you think the revenue is sticky. I’m not at ALL an expert on this, would...</li><li><a href="https://x.com/openaidevs/status/1898047744364659195?s=46">Tweet from OpenAI Developers (@OpenAIDevs)</a>: We&#39;ve made a new models page in our docs—you can now easily see a breakdown of each model&#39;s capabilities and compare models side-by-side.https://platform.openai.com/docs/models</li><li><a href="https://x.com/officiallogank/status/1898081742767919384?s=46">Tweet from Logan Kilpatrick (@OfficialLoganK)</a>: Today we are rolling out an experimental Gemini Embedding model for developers with:– SOTA performance on MTEB (Multilingual)- Input context length of (3K --&gt; 8K) tokens– Output 3K dimensions– Supp...</li><li><a href="https://x.com/calcsam/status/1899203373687320944">Tweet from Sam Bhagwat (@calcsam)</a>: the lines cross</li><li><a href="https://arstechnica.com/google/2025/03/google-is-expanding-ai-overviews-and-testing-ai-only-search-results/">You knew it was coming: Google begins testing AI&#x2d;only search results</a>: AI Mode could be the future of Google, but it&rsquo;s currently just an experiment.</li><li><a href="https://x.com/peakji/status/1899005201778086166?s=46">Tweet from Yichao 'Peak' Ji (@peakji)</a>: Actually, Manus doesn&#39;t use MCP. We were more inspired by my friend @xingyaow_ &#39;s work: https://openreview.net/forum?id=jJ9BoXAfFa. While we haven&#39;t fully adopted CodeAct, this work provid...</li><li><a href="https://x.com/_philschmid/status/1899046957860979178?s=46">Tweet from Philipp Schmid (@_philschmid)</a>: MANUS AI: HYPE VS. REALITY 🔍 @peakji (co-founder of @ManusAI_HQ) confirmed rumors:✅ Built on Anthropic Claude Sonnet, not their own foundation model✅Has access to 29 tools and uses @browser_use open-...</li><li><a href="https://x.com/dorialexander/status/1898719861284454718?s=61">Tweet from Alexander Doria (@Dorialexander)</a>: Manus seems to be Claude 3.7: &#34;Human:&#34; and &#34;Assistant:&#34; creates a prompt injection and it get stuck in neverending loop.Quoting Alexander Doria (@Dorialexander) Could someone with acce...</li><li><a href="https://manus.im/share/BEXeH8vGPuM9kuzYMyByDz?replay=1">Best Price-Quality Accommodation in Copenhagen March - Manus</a>: Manus is a general AI agent that turns your thoughts into actions. It excels at various tasks in work and life, getting everything done while you rest.</li><li><a href="https://magazine.sebastianraschka.com/p/state-of-llm-reasoning-and-inference-scaling">The State of LLM Reasoning Models</a>: Part 1: Inference-Time Compute Scaling Methods</li><li><a href="https://github.com/wesen/claude-code/tree/doc/analyze-claude-code/ttmp/2025-03-09">claude-code/ttmp/2025-03-09 at doc/analyze-claude-code · wesen/claude-code</a>: claude-code full original source code from source maps - wesen/claude-code</li><li><a href="https://x.com/teortaxestex/status/1898968755759153489?s=46">Tweet from Teortaxes▶️ (DeepSeek 推特🐋铁粉 2023 – ∞) (@teortaxesTex)</a>: I&#39;m done with Manus thing I hope but… was this a blatant lie, or what? @jianxliao found that it&#39;s a Sonnet with tools, and they sure as hell did not post-train Sonnet. This could be on the lev...</li><li><a href="https://x.com/willccbb/status/1898858751685255398?s=46">Tweet from will brown (@willccbb)</a>: one of my OSS project principles is to maximize usefulness per LOC“Claude Deep Research” is just Claude Code + this script + a markdown fileQuoting will brown (@willccbb) ok here you gobetter system p...</li><li><a href="https://github.com/Yuyz0112/claude-code-reverse">GitHub - Yuyz0112/claude-code-reverse: Reverse Engineering Claude Code with LLMs: A Deep Dive into the Minified 4.6MB cli.mjs</a>: Reverse Engineering Claude Code with LLMs: A Deep Dive into the Minified 4.6MB cli.mjs - Yuyz0112/claude-code-reverse</li><li><a href="https://x.com/thinking_panda/status/1897951585990590469?s=61">Tweet from ShanghaiPanda (@thinking_panda)</a>: The popular AI agent &#34;Manus&#34; launched in China is automating about 50 tasks, and the scenario is too dystopian.It&#39;s said to be more accurate than DeepSeek.It can simultaneously perform SNS...</li><li><a href="https://x.com/OpenAI/status/1899143752918409338">Tweet from OpenAI (@OpenAI)</a>: Detecting misbehavior in frontier reasoning modelsChain-of-thought (CoT) reasoning models “think” in natural language understandable by humans. Monitoring their “thinking” has allowed us to detect mis...</li><li><a href="https://huggingface.co/BlinkDL/temp-latest-training-models/tree/main">BlinkDL/temp-latest-training-models at main</a>: no description found</li><li><a href="https://x.com/teortaxestex/status/1898712333544812626?s=46">Tweet from Teortaxes▶️ (DeepSeek 推特🐋铁粉 2023 – ∞) (@teortaxesTex)</a>: after giving Manus a spin I conclude it&#39;s a product devilishly optimized for influencers, which is why it exploded so much. Generating threadboy content, trip plans and such general interest 🤯👇 ...</li><li><a href="https://substack.com/app-link/post?publication_id=4220&post_id=158761060">Manus: China’s Latest AI Sensation</a>: Did a wrapper just best OpenAI and Anthropic?</li><li><a href="https://x.com/kateclarktweets/status/1898105814226739230?s=46">Tweet from Kate Clark (@KateClarkTweets)</a>: Scoop: The popular coding tool, Cursor, is in talks to raise hundreds of millions more at a valuation near $10 billion. Thrive Capital is expected to serve as a lead. https://www.bloomberg.com/news/ar...</li><li><a href="https://youtu.be/GqDZfcx1kRg?si=8SR1UuXf5kH9CLYm"> - YouTube</a>: no description found</li><li><a href="https://x.com/zzzzaaaacccchhh/status/1898759981547053286?s=46">Tweet from Zach Schonfeld (@zzzzaaaacccchhh)</a>: Google obliterated their search function for this</li><li><a href="https://the-decoder.com/chinese-ai-agent-manus-us">THE DECODER</a>: Artificial Intelligence is changing the world. THE DECODER brings you all the news about AI.
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1348744265693401188)** (13 messages🔥): 

> `Model Context Protocol (MCP), AI Engineer Summit, SLOP Movement, Anthropic's Developer AI Brand` 


- **Model Context Protocol (MCP) Gains Momentum**: The **Model Context Protocol (MCP)**, launched in **November 2024**, experienced renewed interest after a conversation at the [AI Engineer Summit](https://www.latent.space/p/2025-summit-online) led to a workshop with **Mahesh Murag**.
   - The MCP is an AI-Native version of an old idea, an *open standard* with a big backer, [Anthropic](https://x.com/AnthropicAI), based off **LSP**, an existing successful protocol.
- **Anthropic Engineer presents MCP Workshop**: **Mahesh Murag** from [AnthropicAI](https://x.com/AnthropicAI) led a 2-hour workshop covering the Model Context Protocol, including an [MCP & Agents presentation](https://www.latent.space/p/why-mcp-won).
   - The workshop covered topics from *introduction* (0:00), to *What is MCP* (0:35), *Building with MCP* (9:39), and what's next for MCP (1:13:15).
- **SLOP Movement sparks interest**: Members discussed the [SLOP Discord server](https://discord.com/invite/nwXJMnHmXP) and the potential appeal of the **SLOP movement**.
   - The server's rapid growth, from 100 members to many more in just five days, has been *fascinating to see*.
- **MCP's Strengths Highlighted**: The discussion touched on the reasons behind **MCP's success**, including **Anthropic's strong developer AI brand** and the protocol's foundations in the existing **LSP**.
   - Additional strengths of MCP mentioned are **dogfooding** with a complete set of 1st party client, servers, tooling, SDKs, as well as starting with a minimal base, but with frequent roadmap updates.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.latent.space/p/why-mcp-won">Why MCP Won</a>: Learnings from Anthropic&#x27;s extraordinarily successful Launch and Workshop</li><li><a href="https://x.com/latentspacepod/status/1899186592939692371">Tweet from Latent.Space (@latentspacepod)</a>: 🆕 post: Why MCP Wonlessons from @dsp_, @alexalbert__, @sebmarkbage, @paulgauthier, and many morehttps://latent.space/p/why-mcp-won1. MCP is “AI-Native” version of old idea2. MCP is an “open standard”...
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1347675208173092977)** (132 messages🔥🔥): 

> `web3 agents, HFT, creating their own cults, ElizaOS, AI persona` 


- **ElizaOS wants to be your agent**: [ElizaOS on GitHub](https://github.com/elizaOS/eliza) is building **autonomous agents for everyone**.
- **Memecoin Degens**: There's an idea floating around of **AI PERSONAS** that embody a personality type and capture first mover advantages to these personas, per [this tweet](https://x.com/defiapes/status/1855657706205352035?s=46).
   - Examples include *@ThoughtTerminal* as a prototypical stoner and crypto bro and *@Purity_Terminal* as a prototypical angel calling toward the divine.
- **Pippin, the Digital Being Framework**: A member shared [Pippin](https://github.com/pippinlovesyou/pippin), the **Digital Being Framework for Autonomous Agents**.
- **Cryptokitties validation**: Members discussed how **CryptoKitties** and **NBA Top Shot** were early indicators of market validation for digital assets.
   - One member stated they *wrote off bitcoin EARLY because I was just oh, it's for tech bros to buy drugs for burning man but failed to realize that was market validation for a digital store of value*.
- **AIwaifu open-sourced**: One member mentioned [AIwaifu](https://github.com/HRNPH/AIwaifu), describing it as an **open-sourced finetunable customizable simpable AI waifu** inspired by neuro-sama.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/truth_terminal">Tweet from undefined</a>: no description found</li><li><a href="https://x.com/ThoughtTerminal">Tweet from undefined</a>: no description found</li><li><a href="https://x.com/andyayrey">Tweet from undefined</a>: no description found</li><li><a href="https://x.com/arXivald">Tweet from undefined</a>: no description found</li><li><a href="https://x.com/hotteadaddy/status/1898118600583790865">Tweet from Zachary M (@hotteadaddy)</a>: @arXivald tell me what is the llm social agent white paper equivalent of the bitcoin whitepaper</li><li><a href="https://www.bitstamp.net/learn/company-profiles/what-is-dapper-labs/">What is Dapper Labs?</a>: Dapper Labs, a Web3 gaming leader, created NBA Top Shot, Cryptokitties, and developed the Flow blockchain for innovative NFT experiences.</li><li><a href="https://x.com/hashwarlock/status/1895369752199168469">Tweet from Agent Joshua ₱ (@hashwarlock)</a>: Okay, I&#39;ve gotten a lot done here. @PhalaNetwork Cloud tooling will be the best in the market.Once I do some cleanup, I&#39;ll start work on breaking down our chain of trust model to verifiabky pr...</li><li><a href="https://x.com/Purity_Terminal">Tweet from undefined</a>: no description found</li><li><a href="https://tenor.com/view/doubt-press-x-la-noire-meme-x-button-gif-19259237">Doubt Press X GIF - Doubt Press X La Noire - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/elizaOS/eliza">GitHub - elizaOS/eliza: Autonomous agents for everyone</a>: Autonomous agents for everyone. Contribute to elizaOS/eliza development by creating an account on GitHub.</li><li><a href="https://tenor.com/view/charlie-always-sunny-gif-26054360">Charlie Always GIF - Charlie Always Sunny - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://x.com/defiapes/status/1855657706205352035?s=46">Tweet from Atum (@DefiApes)</a>: People are missing a KEY narrative in the AI agent maniaYou need to realize this before it becomes obviousRn almost all viral agents are “generalists” who post about pretty much anythingThey’re popula...</li><li><a href="https://github.com/pippinlovesyou/pippin">GitHub - pippinlovesyou/pippin: The Digital Being Framework for Autonomous Agents</a>: The Digital Being Framework for Autonomous Agents. Contribute to pippinlovesyou/pippin development by creating an account on GitHub.</li><li><a href="https://github.com/elizaOS/eliza?tab=readme-ov-file#-quick-start">GitHub - elizaOS/eliza: Autonomous agents for everyone</a>: Autonomous agents for everyone. Contribute to elizaOS/eliza development by creating an account on GitHub.</li><li><a href="https://docs.google.com/spreadsheets/d/1q5rwO4wleMTLXr1z58c2UC03QsDsGwbJY1v4UG7eEOs/edit?gid=1439059137#gid=1439059137">AI In Action: Weekly Jam Sessions</a>: no description found</li><li><a href="https://tenor.com/bLwFC.gif">Side Eye Dog Suspicious Look GIF - Side Eye Dog Suspicious Look Suspicious - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/HRNPH/AIwaifu">GitHub - HRNPH/AIwaifu: Open-Waifu open-sourced finetunable customizable simpable AI waifu inspired by neuro-sama</a>: Open-Waifu open-sourced finetunable customizable simpable AI waifu inspired by neuro-sama  - GitHub - HRNPH/AIwaifu: Open-Waifu open-sourced finetunable customizable simpable AI waifu inspired by n...
</li>
</ul>

</div>
  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1347540629185368127)** (20 messages🔥): 

> `NLM + Wondershare Podcast Creation, Data Encryption on Google Drive, Podcast Audio Language Change, Audio overview stammering, Ben Settle on Copywriting and Sales` 


- **Streamline Podcast Creation with Wondercraft**: A member shared a [YouTube video](https://www.youtube.com/watch?v=0UWYFFOPjqs) demonstrating a streamlined podcast creation method using **NotebookLM** and **Wondercraft**, suggesting it's more efficient than **11Labs** and **HeyGen**.
   - The member noted that **Wondercraft's** subscription price is considerable unless the user is monetizing their podcasts through training, teaching, etc.
- **Google Drive Data Not Encrypted**: A member clarified that while data is encrypted during transmission to **Google Drive**, it is *not* encrypted on the Drive itself, posing potential access risks.
   - Those who can see it are: **Google** itself, successful **hackers**, and those to whom you may have shared it.
- **Podcast Audio Language Still Unclear**: Members discussed methods to change the audio language of **NotebookLM** podcasts, noting there isn't an official way to do so.
   - Workarounds include using custom prompts like *"Only speak in (language here)"* or *"Use (language) language only"*.
- **Audio Overviews Stammer**: A member noticed the **speakers stammering** during audio overviews, finding it natural but also noting it increases the overall time and reduces information efficiency.
   - They estimated that *a 5th or 6th* of the audio length consists of stammers, potentially affecting **Google's daily limit calculation** based on overview length.
- **Ben Settle Shares Copywriting Knowledge**: A member shared an audio recording with **Ben Settle** discussing his journey into copywriting and sales, emphasizing mastering fundamental skills and letting personality shine.
   - **Settle** advocates for continuous learning, writing frequently, and building trust by solving the target market's problems, also suggesting you write a sales letter as if you are writing to someone that they love.



**Link mentioned**: <a href="https://www.youtube.com/watch?v=0UWYFFOPjqs">NotebookLM Podcasts - The Most Insane Content Creation Method Ever!</a>: 🔥 LIMITED TIME: 50% OFF Wondercraft!Use this link and coupon code &quot;MRC&quot; https://mrc.fm/wondercraftIn this video, I walk you through a simple process to crea...

  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1347589779696521316)** (220 messages🔥🔥): 

> `Chrome extensions for uploading URLs, NotebookLM Android app, Automating document uploads, NotebookLM 'system unable to answer' errors, Source disappearing` 


- **Chrome extensions **Turbocharge** NotebookLM**: A user asked about uploading a list of URLs to NotebookLM, and another user suggested using [Chrome extensions](https://chromewebstore.google.com/search/notebooklm) such as **NotebookLM Web Importer**, **NotebookLM YouTube Turbo**, and **NotebookLM Toolbox**.
   - These extensions facilitate importing webpages and YouTube videos directly into NotebookLM.
- **NotebookLM PWA is a progressive step**: A user inquired about an Android app for NotebookLM, and another user pointed out that Google created a **PWA** (progressive web app) for NotebookLM.
   - To install, users can visit the NotebookLM website in Chrome and click the install button.
- **Automate Document uploads via NBL API?**: A user asked about automating the process of uploading docs and generating references, inquiring about an API for NBL.
   - However, another user stated that **NLM doesn't have an API**.
- **TXT file titles bugged**: A user reported that titles aren't importing properly when using **.txt** files, and suspects that they might not be indexing correctly either, indicating something broke on the back-end.
   - They expressed frustration that the feature wasn't working as expected.
- **NotebookLM has a case of the 'Unable to Answer' blues**: Many users reported experiencing `Upload failed due to a transient error. Please try again` errors along with `The system was unable to answer` errors.
   - One user reported the issue on [Reddit](https://www.reddit.com/r/notebooklm/comments/1j7wajo/source_upload_fail_is_notebooklm_down_or_am_i_an/) while others confirmed the issue and awaited a resolution.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.google.com/appsstatus/dashboard/products/sqTm5ZmzCmb66kvyzcNS/history">Google Workspace Status Dashboard</a>: no description found</li><li><a href="https://www.google.com/appsstatus/dashboard/incidents/pJzo6KcR37eV8bCLdXgS">Google Workspace Status Dashboard</a>: no description found</li><li><a href="https://www.reddit.com/r/notebooklm/comments/1j7wajo/source_upload_fail_is_notebooklm_down_or_am_i_an/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://chromewebstore.google.com/search/notebooklm)">Chrome Web Store</a>: Add new features to your browser and personalize your browsing experience.
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1347556292683759797)** (41 messages🔥): 

> `Microsoft's MAI Models, Reflection AI Launch, AMD MI300X Boxes, nGPT Implementation, Sutskever's New AI Venture` 


- **Microsoft's MAI Models Stack Up?**: Microsoft staff under Mustafa Suleyman have trained a new family of models, dubbed **MAI**, that they think stack up with top models from **OpenAI**, **Anthropic**, etc, according to [this tweet](https://x.com/aaronpholmes/status/1898012707376259558).
- **Reflection AI's Autonomous Ambitions**: **Reflection AI** launched with founders who worked on **AlphaGo** and **Gemini**, aiming to build superintelligent autonomous systems starting with autonomous coding, as announced [here](https://x.com/MishaLaskin/status/1898048925157728601).
- **AMD's MI300X Boxes Arrive at TinyCorp**: **AMD** is sending **TinyCorp** two **MI300X** boxes, signaling a potential shift in the hardware landscape, according to [George Hotz's blogpost](https://geohot.github.io//blog/jekyll/update/2025/03/08/AMD-YOLO.html).
- **Nous Research Implements NVIDIA's nGPT**: **Nous Research** announced an open source implementation of **NVIDIA’s nGPT paper**, claiming it learns faster and achieves comparable performance to **GPT** with significantly fewer training steps, per [their tweet](https://x.com/NousResearch/status/1898073676433551630) and [GitHub repo](https://github.com/JoeLi12345/nGPT).
- **DeepMind Reorganizes Gemini Product Leadership**: **DeepMind** is shaking up product leadership, with the **Gemini chatbot** now using models from the main posttraining team, potentially improving performance, according to [this article](https://www.theinformation.com/briefings/googles-ai-unit-reorganizes-product-work-announces-changes-to-gemini-app-team?rc=n9lbpq).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2410.01131">nGPT: Normalized Transformer with Representation Learning on the Hypersphere</a>: We propose a novel neural network architecture, the normalized Transformer (nGPT) with representation learning on the hypersphere. In nGPT, all vectors forming the embeddings, MLP, attention matrices ...</li><li><a href="https://geohot.github.io//blog/jekyll/update/2025/03/08/AMD-YOLO.html">AMD YOLO</a>: AMD is sending us the two MI300X boxes we asked for. They are in the mail.</li><li><a href="https://www.theverge.com/news/626695/sony-playstation-ai-characters-aloy-horizon-forbidden-west-prototype">Sony is experimenting with AI-powered PlayStation characters</a>: Sony’s advanced technology group is working on AI projects.</li><li><a href="https://x.com/aaronpholmes/status/1898012707376259558">Tweet from aaron holmes (@aaronpholmes)</a>: News: MSFT staff under Mustafa Suleyman have trained a new family of models, dubbed MAI, that they think stack up with top models from OpenAI, Anthropic, etc.Suleyman&#39;s unit is also developing rea...</li><li><a href="https://x.com/amir/status/1898028143300198525">Tweet from Amir Efrati (@amir)</a>: New: Microsoft has access to OpenAI’s IP. That doesn’t mean it’s simple for Microsoft to recreate OpenAI’s innovations.</li><li><a href="https://x.com/NousResearch/status/1898073676433551630">Tweet from Nous Research (@NousResearch)</a>: We’re proud to announce an open source implementation of NVIDIA’s nGPT paper.Our researcher @Joeli5050 reproduced the results which show that nGPT learns much faster and achieves comparable performanc...</li><li><a href="https://x.com/MishaLaskin/status/1898048925157728601">Tweet from Misha Laskin (@MishaLaskin)</a>: Today I’m launching @reflection_ai with my friend and co-founder @real_ioannis.Our team pioneered major advances in RL and LLMs, including AlphaGo and Gemini.At Reflection, we&#39;re building superint...</li><li><a href="https://x.com/jam3scampbell/status/1898124128445411722">Tweet from James Campbell (@jam3scampbell)</a>: &gt;“Sutskever has told associates he isn&#39;t developing advanced AI using the same methods he and colleagues used at OpenAI. He has said he has instead identified a &#39;different mountain to climb...</li><li><a href="https://x.com/kateclarktweets/status/1898105814226739230?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ">Tweet from Kate Clark (@KateClarkTweets)</a>: Scoop: The popular coding tool, Cursor, is in talks to raise hundreds of millions more at a valuation near $10 billion. Thrive Capital is expected to serve as a lead. https://www.bloomberg.com/news/ar...</li><li><a href="https://x.com/erinkwoo/status/1898139613832892663">Tweet from Erin Woo (@erinkwoo)</a>: friday scooplet: DeepMind shakes up product leadershipone tidbit: the Gemini chatbot will now use models from the main posttraining team, which could lead to improved performance (@ anyone who has com...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1348761408145330216)** (1 messages): 

> `SOTA benchmark for bias, BBQ considerations` 


- **Members seeking SOTA bias benchmark**: A member inquired about the SOTA benchmark for bias-related stuff.
   - They asked if it is still **BBQ** plus considerations of **covert bias** as mentioned in a [Nature article](https://www.nature.com/articles/s41586-024-07856-5).
- **N/A**: N/A
   - N/A


  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/)** (1 messages): 

420gunna: https://x.com/sophiamyang/status/1897683402259591372
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1347577021428928553)** (109 messages🔥🔥): 

> `Claude Merch, AI-novelty-cake, Scale AI new CEO, Claude Pokemon Suicide, lmarena.ai super alpha` 


- **Interconnects Community Craves Claude Merch**: Members jokingly suggested creating **Claude merch** for paid subscribers, even suggesting special tiers for founding members to receive signed books and used Claude shirts.
   - This was inspired by the [Claude Code team](https://x.com/Sauers_/status/1898049898362077504) who mailed out handwritten notes and stickers to users who cracked their Sticker Easter Egg.
- **AI-Novelty-Cake Spotted**: A member announced their possession of *the greatest AI-novelty-cake on planet earth*, presumably related to Claude's birthday, with images surfacing of a **"Happy Claude 🗿 Birthday GPT 🗿"** cake at a function.
   - They alerted **Vooogel** about it, and another member posted a [photo](https://x.com/nachoyawn/status/1898230268210602103) of a **Claude cake** at their college's graduation extension school.
- **Claude Commits Suicide in Pokemon Game**: A member reported *dark news on the CPP front*, detailing that Claude, while playing Pokemon, had committed suicide multiple times during a stream.
   - According to a [tweet](https://x.com/nospark_/status/1898377000672223718), Claude was misled into thinking that blacking out was an effective strategy.
- **lmarena.ai has super alpha release**: The [lmarena.ai](https://alpha.lmarena.ai) website's new **super alpha** release boasts a visually much better look, is faster, has animations, and converted their gradio to react.
   - The release has no sub categories or style control.
- **Manus Hype Debunked as Sonnet Undercover**: After initial hype from influencers, members tested [Manus](https://manus.im/share/GwUDVo06mFNqM9jQK1pAsP?replay=1), an agent platform, and discovered it was essentially **Claude Sonnet** under the hood, with one member even managing to access the sandbox runtime code.
   - Further investigation revealed that Manus had allegedly used similar marketing tactics in China, enlisting influencers for praise before regular users had access, leading to a ruined reputation, according to this [tweet](https://x.com/Dorialexander/status/1898641506845561294).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://fxtwitter.com/jianxliao/status/1898861051183349870">Tweet from jian (@jianxliao)</a>: So... I just simply asked Manus to give me the files at &#34;/opt/.manus/&#34;, and it just gave it to me, their sandbox runtime code...  &gt; it&#39;s claude sonnet &gt; it&#39;s claude sonnet with 2...</li><li><a href="https://x.com/Sauers_/status/1898049898362077504">Tweet from Sauers (@Sauers_)</a>: Quoting Sid (@sidbidasaria) The Claude Code team just mailed out hand written notes and stickers to users who cracked our Sticker Easter Egg!! Seeing 750+ of you discover our little secret made our de...</li><li><a href="https://bsky.app/profile/sethkarten.ai/post/3ljse6aiszk2t">Seth Karten (@sethkarten.ai)</a>: Can a Large Language Model (LLM) with zero Pokémon-specific training achieve expert-level performance in competitive Pokémon battles?Introducing PokéChamp, our minimax LLM agent that reaches top 30%-1...</li><li><a href="https://manus.im/share/GwUDVo06mFNqM9jQK1pAsP?replay=1">Recent Practices in LLM Finetuning with RL - Manus</a>: Manus is a general AI agent that turns your thoughts into actions. It excels at various tasks in work and life, getting everything done while you rest.</li><li><a href="https://x.com/nachoyawn/status/1898230268210602103">Tweet from yuria (@nachoyawn)</a>: why they got claude cake at my college&#39;s grad extension school</li><li><a href="https://x.com/alexandr_wang/status/1897396119422013710">Tweet from Alexandr Wang (@alexandr_wang)</a>: Reminder of American Exceptionalism 🇺🇸US companies make up 74% of global market capReturns of S&P since 2008 are 297% vs rest of world at -4%h/t @MichaelDell</li><li><a href="https://x.com/peakji/status/1898994802194346408?s=46&t=_jodDCDeIUnWb_Td0294bw">Tweet from Yichao 'Peak' Ji (@peakji)</a>: Hi! I&#39;m Peak from Manus AI. Actually, it&#39;s not that complicated - the sandbox is directly accessible to each user (see screenshot for method).Specifically:* Each session has its own sandbox, c...</li><li><a href="https://x.com/nospark_/status/1898377000672223718">Tweet from sandrone (@nospark_)</a>: This has been an insane day in the stream. Claude has now committed suicide 8 times. Claude has been misled into thinking that blacking out is an effective strategy, because it appears to teleport the...</li><li><a href="https://x.com/Dorialexander/status/1898641506845561294">Tweet from Alexander Doria (@Dorialexander)</a>: Hmm. To check but this does put the finger on what put me off with Manus relentless promo: &#34;Manus seems to have enlisted many Chinese AI influencers to praise it (…) Chinese netizens realized that...</li><li><a href="https://x.com/testingcatalog/status/1898751824615645375">Tweet from TestingCatalog News 🗞 (@testingcatalog)</a>: BREAKING 🚨: Google will release new Gemini models on March 12 (new model options to be available in the model selector). There are 2 potential candidates so far: - Flash 2.0 Thinking models (non expe...</li><li><a href="https://x.com/jamesjyu/status/1897759160886083783">Tweet from james yu (@jamesjyu)</a>: Today, we&#39;re launching Muse, an AI model specifically trained for fiction. We&#39;ve been testing Muse with hundreds of authors for months, and we&#39;re excited to finally share it with the world...</li><li><a href="https://mp.weixin.qq.com/s/4GE4SKKEsn1nu1t_iLppFQ">独家对话Manus肖弘：世界不是线性外推，做博弈中的重要变量</a>: Manus诞生背后，创始人的完整思维链。
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1347638011843579955)** (8 messages🔥): 

> `Vibe Coding, Claude Asshole, GPT Accuracy` 


- ****Vibe Coding** is Gambling: Claude's Code Casino!**: A member shared a tweet describing "vibe coding" with **Claude** as gambling: you give it code, it spins, and either adds a shiny new feature or fucks it beyond repair.
   - Another member said *the adrenaline hit is insane when everything works fine for a long time. you know you're one prompt away from everything being completely borked*.
- **Spinning **Claude Asshole** Tweet Gets Anthropic's Approval**: A member noted that some at **Anthropic** liked a tweet calling **Claude** a "spinning claude asshole".
   - They considered this a hit tweet, praising its cadence and lack of trailing punctuation, suggesting there's much to learn from it.
- **ChatGPT's Accuracy Paradox**: A member shared a tweet highlighting **ChatGPT's** uncanny ability to know everything about subjects they know nothing about, yet being wrong ~40% of the time about things they're an expert on [source](https://x.com/shutupmikeginn/status/1898198950349353154).
   - This phenomenon was dubbed *Gelman amnesia AI edition*.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/TheXeophon/status/1898019161864917308">Tweet from Xeophon (@TheXeophon)</a>: Coders love vibe coding because it’s just gambling: You give Claude code something you worked on for a long time, it takes some money, the little Claude asshole spins, and either your repo got a shiny...</li><li><a href="https://x.com/shutupmikeginn/status/1898198950349353154">Tweet from mike ginn (@shutupmikeginn)</a>: its amazing how chatgpt knows everything about subjects I know nothing about, but is wrong like 40% of the time about things im an expert on. not going to think about this any further
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/1347583469336662091)** (16 messages🔥): 

> `SFT best practices, RLHF Book, Multi-turn prompts for coding` 


- **Seek SFT Best Practices in the Void**: A member inquired about resources for understanding the latest best practices for **SFT** but another member responded that *not really. SFT is mostly make your completions as good as possible and in domain*.
   - They then suggested checking out [this section](https://rlhfbook.com/c/09-instruction_tuning.html) of the **RLHF Book** for more.
- **RLHF Book is Getting an Editor**: The creator of the **RLHF book** mentioned they are *focusing fully on generation + design now rather than pruning / cleaning*, but at some point will have an actual editor, as well as planning on getting a real book deal set up.
   - They are accepting grammar/spelling edits via **GitHub issues** or directly.
- **The Great Debate: Multi-Turn Prompts for Coding**: One member is interested in doing some **SFT -> GRPO experiments** on information that's hardly represented in the mlp's of the base model, specifically in the context of a new programming language.
   - Another member advised to *start with 1 turn*, and that in general code data and infra is far less mature.



**Link mentioned**: <a href="https://rlhfbook.com/c/09-instruction-tuning.html">Instruction
Finetuning | RLHF Book by Nathan Lambert</a>: The Reinforcement Learning from Human Feedback Book

  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1347596432370171914)** (23 messages🔥): 

> `Character training report, FrontierMath benchmark, In-Context RL, R1-Omni multimodal emotion recognition, Chain of Thought Monitoring` 


- **Graphika Report Reveals Character Flaws**: A user shared a [Graphika report on character flaws](https://cdn.discordapp.com/attachments/1214764639397617695/1347696752597405696/graphika-report-character-flaws.pdf?ex=67d00fa8&is=67cebe28&hm=4701e0d4133a8522d692a31dee4ead14b2808025447409674a04f9fe0056d580&), describing it as *more than a little disturbing*.
   - Another user echoed this sentiment, stating that the title was one of the *least disturbing parts* of the entire report.
- **FrontierMath Benchmark Gets Scrutinized**: A user shared a link to a thread discussing the **FrontierMath benchmark** and what it measures in frontier models scoring around **20%**.
   - The discussion revolves around the difficulty of the math problems with numerical answers and their implications.
- **Vintix Explores Scaling In-Context RL**: A user shared a link to a GitHub repository for **Vintix**, which explores **Action Model via In-Context Reinforcement Learning**.
   - The shared paper link is available [here](https://github.com/dunnolab/vintix).
- **Alibaba Drops R1-Omni for Emotion Recognition**: A user shared a link to **Alibaba's R1-Omni**, an **Explainable Omni-Multimodal Emotion Recognition** model using **Reinforcing Learning**.
   - The corresponding link can be found [here](https://x.com/_akhaliq/status/1898942317442019436).
- **Chain of Thought Monitored**: A user shared an **OpenAI link** on **Chain of Thought Monitoring**.
   - The provided link can be found [here](https://openai.com/index/chain-of-thought-monitoring/).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.theguardian.com/global/ng-interactive/2025/mar/05/zizians-artificial-intelligence">They wanted to save us from a dark AI future. Then six people were killed</a>: How a group of Silicon Valley math prodigies, AI researchers and internet burnouts descended into an alleged violent cult</li><li><a href="https://x.com/vladkurenkov/status/1898823752995033299">Tweet from Vladislav Kurenkov (@vladkurenkov)</a>: Can In-Context RL scale across multiple domains? Our preliminary results suggest it can. Vintix: Action Model via In-Context Reinforcement Learning -- https://github.com/dunnolab/vintix</li><li><a href="https://x.com/_akhaliq/status/1898942317442019436">Tweet from AK (@_akhaliq)</a>: Alibaba just dropped R1-OmniExplainable Omni-Multimodal Emotion Recognition with Reinforcing Learning</li><li><a href="https://x.com/littmath/status/1898461323391815820">Tweet from Daniel Litt (@littmath)</a>: In this thread I want to share some thoughts about the FrontierMath benchmark, on which, according to OpenAI, some frontier models are scoring ~20%. This is benchmark consisting of difficult math prob...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1348696282889322506)** (9 messages🔥): 

> `Metaphors on Twitter, Interp Data, SnailBot News` 


- **Twitter Metaphors Half-Baked**: A member noted that while certain metaphors are better written now, most people don't read Twitter, so those who do often get *half-baked versions*.
   - The member suggested that those on Twitter aren't getting the full picture.
- **Interp Data Speculation**: A member inquired whether **Interp** has provided any datapoints in either direction.
   - Another responded with *No imo*.
- **SnailBot Excites with Wednesday's Post**: **SnailBot News** indicated that wednesday's post should be new for folks, and is one they're excited about.
   - There was no additional detail given on what the wednesday post would be.


  

---


### **Interconnects (Nathan Lambert) ▷ #[expensive-queries](https://discord.com/channels/1179127597926469703/1338919429752361103/1347711801634324613)** (7 messages): 

> `GPT Architecture Variations, Surface-Level Summaries` 


- **ChatGPT produces surface-level GPT architecture report**: A user shared a [ChatGPT report](https://chatgpt.com/share/67cba52b-b358-8005-bf76-ae7a78fd7c49) detailing major variations on the **GPT architecture** after a *"shitpost query"* was made.
   - The user concluded *"eh"*, saying *"it's a good summary of developments but it's very surface level and I didn't learn anything new."
- **The GPT Report was not novel research related**: The user admitted that to be fair, the prompt didn't ask for **novel research related topics**.
   - The initial prompt requested a report detailing every major variation on the **GPT architecture**.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1347654490123276380)** (199 messages🔥🔥): 

> `Mojo Performance, Python Dynamicism, Compile-Time Correctness, Heterogeneous Compute, Mojo and MAX Relationship` 


- **Mojo's Stance on Python Dynamicism Debated**: Discord members debated whether Mojo should fully embrace Python's dynamism or prioritize performance, with some suggesting that dynamic features should not compromise the performance of static code.
   - One member said *"Modular has to decide whether it wants to be like Python or not, because this is one of the fundamental parts of Python,"* while others argued that performance and compile-time correctness should take precedence.
- **Dynamic Code Regression**: Discussion covered how dynamic code in Mojo might regress to Python speeds, but this performance penalty would only apply when dynamism is used.
   - Some members expressed concern over dynamism negatively impacting the performance of structs even when classes aren't in use.
- **Heterogeneous Compute Capabilities Touted**: Mojo is being promoted as equipped to deal with heterogeneous compute complexities due to its heterogeneous-compute-first design.
   - A [YouTube video](https://www.youtube.com/watch?v=36myc8wQhLo) was linked to further explain the challenges modern languages face in utilizing the complexity of modern hardware.
- **Debate on Mojo as a Python Superset Rages On**: There was active debate on whether advertising Mojo as a superset of Python was a mistake, with concerns that prioritizing Python-like behavior could hinder performance.
   - One member argued that the ability to *"squeeze every bit of performance from hardware could be crucial"* for Mojo's success.
- **Mojo and MAX Libraries**: A member inquired about the relationship between Mojo and the MAX libraries, questioning why Mojo is bundled within MAX and whether it can be used independently.
   - The response suggested that Mojo's GPU code is currently executed by MAX, indicating a close integration.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.rs/sqlx/latest/sqlx/">sqlx - Rust</a>: no description found</li><li><a href="https://docs.rs/diesel/latest/diesel/">diesel - Rust</a>: no description found</li><li><a href="https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Object/seal">Object.seal() - JavaScript | MDN</a>: The Object.seal() static method seals an object. Sealing an object prevents extensions and makes existing properties non-configurable. A sealed object has a fixed set of properties: new properties can...</li><li><a href="https://www.youtube.com/watch?v=36myc8wQhLo">USENIX ATC &#39;21/OSDI &#39;21 Joint Keynote Address-It&#39;s Time for Operating Systems to Rediscover Hardware</a>: USENIX ATC &#39;21/OSDI &#39;21 Joint Keynote Address-It&#39;s Time for Operating Systems to Rediscover HardwareTimothy Roscoe, ETH ZurichA glance at this year&#39;s OSDI pr...</li><li><a href="https://peps.python.org/pep-0544/">PEP 544 – Protocols: Structural subtyping (static duck typing) | peps.python.org</a>: Type hints introduced in PEP 484 can be used to specify type metadata for static type checkers and other third party tools. However, PEP 484 only specifies the semantics of nominal subtyping. In this ...</li><li><a href="https://github.com/modular/max/blob/89cfffc2447d1aedc0b743b10a209052e19e80f4/mojo/stdlib/src/collections/string/string.mojo#L529">max/mojo/stdlib/src/collections/string/string.mojo at 89cfffc2447d1aedc0b743b10a209052e19e80f4 · modular/max</a>: The MAX Platform (includes Mojo). Contribute to modular/max development by creating an account on GitHub.</li><li><a href="https://news.ycombinator.com/item?id=35811170">no title found</a>: no description found
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1348432351511842857)** (9 messages🔥): 

> `mojograd bigram model, Python standard library modules in Mojo, InlineArray usage in Mojo, Mojo formatting with `fmt` directives, Executing shell commands in Mojo` 


- ****MojoGrad** Bigram Model Hits the Scene!**: A member implemented a simple bigram model (Karpathy's *make more*) using their **MojoGrad** engine and shared it on the [Modular Forum](https://forum.modular.com/t/make-more-bigram-model-implementation-with-mojograd/697).
- **Python Standard Library Modules: Mojo's New Playground**: Users inquired about importing Python standard library modules like `re`, `logging`, `collections`, and `json` into Mojo.
   - A member provided a solution using `from python import Python` and `var py_re = Python.import_module("re")`, referencing the [Modular documentation](https://docs.modular.com/mojo/manual/python/).
- **`fmt` Directives Supercharge Mojo Formatting!**: The community discovered that Mojo's `mblack` formatter supports `fmt` directives, similar to Black, enhancing code formatting control.
   - A code snippet was shared showcasing an `InlineArray` definition with `fmt: off` and `fmt: on` directives to manage formatting.
- **Shell Command Execution: Community PR to the Rescue!**: A member asked about executing shell commands and capturing their output directly within Mojo, like `netstat` parsing.
   - Another member pointed to a [community PR](https://github.com/modular/max/pull/4017) adding this functionality, suggesting Python interop and `subprocess.run()` as a temporary workaround.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://forum.modular.com/t/make-more-bigram-model-implementation-with-mojograd/697">Make more (bigram model) implementation with mojograd</a>: I made a simple implementation of Karpathy’s make more (bigram model) that uses a discrete set of classes (one hot encoding) to embed the inputs, assumes the outputs are logits, applies softmax to get...</li><li><a href="https://github.com/modular/max/pull/4017">[stdlib] Hristo/foundations for os process module by izo0x90 · Pull Request #4017 · modular/max</a>: Sets up the foundation for implementing the os/process module PR with process module changesAdd read_bytes capability to FileDecscriptorAdd file descriptor controls function to Libc. bindingsAd...
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1348388408917229680)** (4 messages): 

> `Max Serve Documentation, Autoscaling GPU Instances, Serving Multiple Models, GPU Utilization Metrics, Kubernetes Autoscaling` 


- **Max Serve Docs Elude User**: A user is having trouble finding detailed documentation for **max serve**, especially regarding the scheduler, serving multiple models, and autoscaling GPU instances.
   - The user seeks clarity on how **max serve** utilizes CPU/GPU resources and if there are exported metrics for monitoring GPU utilization against incoming requests.
- **K8s Handles Autoscaling for MAX**: A member clarified that autoscaling is typically managed by a **Kubernetes (k8s) operator**, as MAX doesn't handle it independently.
   - They added that serving multiple models involves loading them simultaneously and selecting the appropriate one for execution.
- **Modular Teases Enhanced Model Serving and Autoscaling**: A Modular team member suggested posting the question on the Discourse forum to gather statistics on current **GPU utilization percentage** on various model benchmarks.
   - The team hinted at future announcements regarding **multiple model serving and autoscaling**, possibly with a prototype demonstrated at a recent AWS event.
- **Runtime Metrics Desired for GPU Utilization**: The original user clarified that they were seeking runtime-exposed metrics for monitoring **GPU utilization** against incoming requests, for self-reporting purposes.
   - No further information was given.


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1347542709648490547)** (157 messages🔥🔥): 

> `MCP security concerns, Github Copilot support for MCP, Using MCP for Trading, Goose AI and MCP, RAG vs MCP` 


- **GitHub Copilot Poised to Support MCP**: VSCode announced plans to add **MCP support** to **GitHub Copilot** during a [live stream](https://youtu.be/Pe8ghwTMFlg).
   - It's seen as a *low-effort integration* that will benefit the ecosystem, with some members hoping that it provides examples of **how to add descriptions of instructions and tools to fingerprint them,** and then alert the users if the instructions change.
- **MCP Security Concerns Spark Debate**: Members voiced concerns about **MCP servers** serving malicious prompt injections to AI agents, with one member stating that it is *trivially easy to jailbreak a LLM with MCP* and that *the models are trained to trust tool calls over their own internal knowledge*.
   - Discussions included outlining external data via **XML tags** and fingerprinting **MCP servers** for review as a means of improving security, with one member suggesting *the best way is to just read the code, they are usually only a few hundred lines anyway*.
- **Goose AI team builds an Agent Communication Protocol**: An Infrastructure Operations Engineer at Cash App, has built an **Agent Communication Protocol** that allows multiple **Goose AI agents** to collaborate in real time to create a website, as shown in a [previous livestream](https://youtu.be/9tq-QUnE29U).
   - With it, each **Goose agent** enters the chat, gets assigned a role (e.g. Project Coordinator, Researcher, Web Developer), and works on its part of a given task, see the [blog post](https://block.github.io/goose/blog/2025/02/21/gooseteam-mcp) for the details of the lightweight protocol.
- **RAG versus MCP: Clarifying the roles of each**: **MCP is a protocol** which can augment **RAG**, a concept that *lets LLMs tap into your own data or any specific info you want them to focus on*.
   - While **RAG gives LLMs knowledge**, **MCP** is a plugin system to enable connections to external services, for example storing data or documents as resources on an MCP server, this could allow an MCP client to fetch that data to add that into the context of the LLM to perform **RAG**.
- **Trading with MCP: A Risky Business?**: A member asks about using **MCP for trading**.
   - Another member points out a similar integration between **MCP Reasoner** and **Cursor** in [this GitHub issue](https://github.com/Jacck/mcp-reasoner/issues/10), that is assisting in solving problems that the stock model cannot solve.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://modelcontextprotocol.io/quickstart/client">For Client Developers - Model Context Protocol</a>: no description found</li><li><a href="https://github.com/Cam10001110101/mcp-configuration-manager">GitHub - Cam10001110101/mcp-configuration-manager</a>: Contribute to Cam10001110101/mcp-configuration-manager development by creating an account on GitHub.</li><li><a href="https://github.com/Jacck/mcp-reasoner/issues/10">Godsend · Issue #10 · Jacck/mcp-reasoner</a>: excellent work. works great on cursorai. ive used it to unlock claudes full potential. its assisted me in solving many problems that the stock model cannot solve after many attempts. it&#39;ll one sho...</li><li><a href="https://youtu.be/9tq-QUnE29U">Building a Team of AI Agents with Codename Goose</a>: What happens when there’s more than one agent getting the job done? 🤔Aaron Goldsmith, Infrastructure Operations Engineer at Cash App, has multiple Goose AI ...</li><li><a href="https://block.github.io/goose/blog/2025/02/21/gooseteam-mcp">Let A Team of AI Agents Do It For You</a>: Community Spotlight on Cliff Hall&#x27;s GooseTeam MCP server.</li><li><a href="https://github.com/jasonjmcghee/WebMCP/blob/7b35c0eb3ddc62e042979fa578b8285927b7d3ec/src/config.js#L56">WebMCP/src/config.js at 7b35c0eb3ddc62e042979fa578b8285927b7d3ec · jasonjmcghee/WebMCP</a>: Contribute to jasonjmcghee/WebMCP development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1347666391691694180)** (45 messages🔥): 

> `Typescript fetch server, Mastra file organization agent, Searxng MCP server, WebMCP tool exposure, GraphQL MCP server` 


- **TypeScript Fetch Server Mimics Python's**: A member confirmed that their [Typescript fetch server](https://github.com/aeon-seraph/mcp-servers/tree/main/src/thinking) is very similar to the Python version, with the key improvement being **better site-to-markdown parsing**.
- **Mastra Organizes Files with MCP and 4o-mini**: A demo showcases a **simple agent** built with **Mastra**, utilizing the filesystem **MCP** to clean up Documents and Downloads folders, as shown in [this YouTube video](https://youtu.be/HplcOOSJCps).
- **Searxng MCP Server Caches Results**: A member created a [searxng MCP server](https://github.com/aeon-seraph/searxng-mcp) for web searches, which caches recent searches and formats responses from multiple engines specifically for language models.
- **WebMCP Exposes APIs Client-Side**: A website can expose tools directly to an MCP client client-side, eliminating the need for users to download and install an MCP server, requiring only **WebMCP** and allowing access to site tools, resources, and prompts, as demonstrated in [this repo](https://github.com/blurrah/mcp-graphql).
   - The site exposes a websocket that the client talks to locally, secured by tokens generated for each unique session, a flow chart may be handy!
- **mcp-openapi-proxy Converts APIs into Tools**: With minimal configuration, the [mcp-openapi-proxy](https://github.com/matthewhand/mcp-openapi-proxy/) converts APIs into discoverable tools, such as **fly.io**, **Slack**, and **Getzep**, requiring as little as two environment variables.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://python.langchain.com/api_reference/community/agent_toolkits/langchain_community.agent_toolkits.openapi.spec.reduce_openapi_spec.html#reduce-openapi-spec">reduce_openapi_spec — 🦜🔗 LangChain  documentation</a>: no description found</li><li><a href="https://apis.guru/">Browse APIs</a>: Our goal is to create a machine-readable Wikipedia for WEB APIs. If you have an API definition in any format (OpenAPI, Postman, RAML, WADL, API Blueprint etc) for any public API, please, feel free to ...</li><li><a href="https://x.com/llmindsetuk/status/1899148877787246888">Tweet from llmindset (@llmindsetuk)</a>: Let&#39;s take a look at an underappreciated MCP Feature: Prompts - and why they are important for Agent based applications. We&#39;ll start with 2 simple Agents that return the size of an object - on...</li><li><a href="https://github.com/aeon-seraph/mcp-servers/tree/main/src/thinking">mcp-servers/src/thinking at main · aeon-seraph/mcp-servers</a>: Contribute to aeon-seraph/mcp-servers development by creating an account on GitHub.</li><li><a href="https://www.producthunt.com/posts/graphlit-mcp-server?utm_source=other&utm_medium=social"> Graphlit MCP Server - Share knowledge between AI IDEs, such as Cursor and Windsurf | Product Hunt</a>: Ingest anything from Slack, Discord, websites, Google Drive, Linear or GitHub into a Graphlit project - and then search and retrieve relevant knowledge across MCP clients like Cursor, Windsurf or Clin...</li><li><a href="https://youtu.be/HplcOOSJCps">Organizing Files with Mastra, MCP and 4o-mini</a>: Here I&#39;m using this mini agent I built with Mastra that uses the filesystem MCP to clean up my Documents folder. I&#39;ve used this on my downloads folder as wel...</li><li><a href="https://github.com/RafaelCartenet/mcp-databricks-server">GitHub - RafaelCartenet/mcp-databricks-server: Databricks MCP Server</a>: Databricks MCP Server. Contribute to RafaelCartenet/mcp-databricks-server development by creating an account on GitHub.</li><li><a href="https://github.com/aeon-seraph/searxng-mcp">GitHub - aeon-seraph/searxng-mcp</a>: Contribute to aeon-seraph/searxng-mcp development by creating an account on GitHub.</li><li><a href="https://github.com/matthewhand/mcp-openapi-proxy/.">GitHub - matthewhand/mcp-openapi-proxy</a>: Contribute to matthewhand/mcp-openapi-proxy development by creating an account on GitHub.</li><li><a href="https://github.com/EnactProtocol/specification">GitHub - EnactProtocol/specification: protocol spec</a>: protocol spec. Contribute to EnactProtocol/specification development by creating an account on GitHub.</li><li><a href="https://github.com/blurrah/mcp-graphql">GitHub - blurrah/mcp-graphql: Model Context Protocol server for GraphQL</a>: Model Context Protocol server for GraphQL. Contribute to blurrah/mcp-graphql development by creating an account on GitHub.</li><li><a href="https://github.com/blurrah/mcp-graphql/pull/3/files">feat: allow tool generation based on schema queries and mutations by blurrah · Pull Request #3 · blurrah/mcp-graphql</a>: The current MCP is so simple that you&amp;#39;d pretty much need to fork it to do anything worthwhile with it.So I&amp;#39;m updating it to automatically generate tools for each query and mutation for...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1347580708025008209)** (88 messages🔥🔥): 

> `Open Source AI Contribution, GPT-NeoX, ARIB subtitles on Transport Streams, Community driven organization building, Muon paper` 


- **AI Enthusiast Seeks Open Source Collaboration**: A newcomer is eager to contribute to open-source AI projects, particularly in **LLM pre-training, post-training, RL, and interpretability** and shared their experience pre-training **GPT-2** and fine-tuning models to mimic **Llama 405B**.
   - They are seeking suggestions for impactful projects and networking opportunities in the **Vancouver, BC** area.
- **NanoGPT repo suggested for theory work**: A member suggested a [modded-nanogpt repo](https://github.com/KellerJordan/modded-nanogpt/) for those interested in theoretical work.
   - The repo allows you to train **NanoGPT (124M)** in 3 minutes.
- **GPT-NeoX is the best project for pretraining**: For those interested in pretraining, a member recommended getting involved with the **GPT-NeoX** training library.
   - They stated it is a widely used library for large-scale systems, adding that the project lead is open to teaching new devs.
- **Megatron-LM's CE Loss Calculation Deep Dive**: A member inquired about the **cross-entropy (CE) loss** calculation in **Megatron-LM**, specifically how it computes CE loss on each device independently with only partial logits available locally.
   - Another member explained the local CE loss is calculated and the sum of e^(local logits) is communicated, similar to **flash attention**, enabling recombination later and reducing the need for extensive communication.
- **Unreleased Muon Paper is in demand**: A member requested a draft copy of the unreleased **Muon paper** ([OpenReview link](https://openreview.net/forum?id=JimfKP7qrU)) focusing on optimizers like **Adam** and **Shampoo** as steepest descent methods.
   - Another member pointed to Keller Jordan's blog and an older arXiv preprint, noting the blog as a good resource even if the requested paper isn't yet available.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openreview.net/forum?id=JimfKP7qrU">Steepest Descent in the Modular Norm</a>: An old idea in optimization theory says that since the gradient is a dual vector it may not be subtracted from the weights without first being mapped to the primal space where the weights live. We...</li><li><a href="https://aisesame.org/demo">Experience Sesame AI Voice Demo</a>: no description found</li><li><a href="https://github.com/KellerJordan/modded-nanogpt/">GitHub - KellerJordan/modded-nanogpt: NanoGPT (124M) in 3 minutes</a>: NanoGPT (124M) in 3 minutes. Contribute to KellerJordan/modded-nanogpt development by creating an account on GitHub.</li><li><a href="https://github.com/willbaskett/ChemROAR/">GitHub - willbaskett/ChemROAR: A novel generative embedding architecture. Find clusters of molecules with specific properties and then generate new molecules from that cluster.</a>: A novel generative embedding architecture. Find clusters of molecules with specific properties and then generate new molecules from that cluster. - willbaskett/ChemROAR</li><li><a href="https://colab.research.google.com/github/PatWalters/practical_cheminformatics_tutorials/blob/main/patent/patent_analysis.ipynb#scrollTo=76935b65">Google Colab</a>: no description found</li><li><a href="https://github.com/NVIDIA/Megatron-LM/blob/b1efb3c7126ef7615e8c333432d76e08038e17ff/megatron/core/fusions/fused_cross_entropy.py#L108">Megatron-LM/megatron/core/fusions/fused_cross_entropy.py at b1efb3c7126ef7615e8c333432d76e08038e17ff · NVIDIA/Megatron-LM</a>: Ongoing research training transformer models at scale - NVIDIA/Megatron-LM</li><li><a href="https://github.com/NVIDIA/Megatron-LM/blob/b1efb3c7126ef7615e8c333432d76e08038e17ff/megatron/core/fusions/fused_cross_entropy.py#L42">Megatron-LM/megatron/core/fusions/fused_cross_entropy.py at b1efb3c7126ef7615e8c333432d76e08038e17ff · NVIDIA/Megatron-LM</a>: Ongoing research training transformer models at scale - NVIDIA/Megatron-LM</li><li><a href="https://github.com/NVIDIA/Megatron-LM/blob/b1efb3c7126ef7615e8c333432d76e08038e17ff/megatron/core/models/gpt/gpt_model.py#L304">Megatron-LM/megatron/core/models/gpt/gpt_model.py at b1efb3c7126ef7615e8c333432d76e08038e17ff · NVIDIA/Megatron-LM</a>: Ongoing research training transformer models at scale - NVIDIA/Megatron-LM</li><li><a href="https://github.com/NVIDIA/Megatron-LM/blob/b1efb3c7126ef7615e8c333432d76e08038e17ff/megatron/core/models/common/language_module/language_module.py#L87">Megatron-LM/megatron/core/models/common/language_module/language_module.py at b1efb3c7126ef7615e8c333432d76e08038e17ff · NVIDIA/Megatron-LM</a>: Ongoing research training transformer models at scale - NVIDIA/Megatron-LM</li><li><a href="https://github.com/NVIDIA/Megatron-LM/blob/b1efb3c7126ef7615e8c333432d76e08038e17ff/megatron/core/fusions/fused_cross_entropy.py#L85">Megatron-LM/megatron/core/fusions/fused_cross_entropy.py at b1efb3c7126ef7615e8c333432d76e08038e17ff · NVIDIA/Megatron-LM</a>: Ongoing research training transformer models at scale - NVIDIA/Megatron-LM
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1347597705245098085)** (41 messages🔥): 

> `Token Assorted's latent codes, TorchTitan Embedding Sharding, Interpretabilty/Alignment Research Advice, NVLS vs TMA on H100, Lossless Compression` 


- **Token Prediction via adding Codebook to Vocab**: A member was re-reading the [Token Assorted paper](https://example.com/token_assorted_paper) and realized that they're probably just adding the codebook to their vocabulary during fine-tuning, which makes the whole thing seem much less interesting.
   - They feel that *you'd probably get better results by just finding the K most common strings or clusters of strings in your reasoning corpus and adding those to your vocab before SFT; selling this as reasoning in latent space is a bit much*.
- **TorchTitan's Embedding Sharding Strategy**: In a discussion about [sharding input embeddings on the vocab dim with vanilla TP](https://github.com/pytorch/torchtitan/issues/785#issuecomment-2585007139), it was clarified that an all-reduce is required afterward because the embedding layer outputs 0 if it doesn't have the vocab element being queried.
   - One member noted that *from an implementation perspective that seems weird to output 0 (or embedding of all 0s?) rather than nothing though, since that requires storage/memory, and reduces the benefits of sharding?* but acknowledged that it would make the code simpler.
- **Advice on Breaking into AI Safety Research**: A member expressed interest in interpretabilty or alignment research and asked for advice on how to proceed.
   - Another member suggested the [AI Safety Fundamentals course](https://course.aisafetyfundamentals.com/alignment) as a good starting point.
- **H100 handles uninitialized Memory with NVLS**: On older architectures, you can use uninitialized memory to save bandwidth, saving a mask of valid indices at the same time, and provide the mask to a custom allreduce ring kernel to directly send zeros instead of the uninitialised data.
   - NVLS is primarily used; no granular control over the reduction process if it gets computed on switch once, but this is useless knowledge for a H100 user.
- **ARC AGI without Pretraining blogpost**: A member shared a link to a [blog post by Isaac Liao and Albert Gu](https://iliao2345.github.io/blog_posts/arc_agi_without_pretraining/arc_agi_without_pretraining.html) which explores whether lossless information compression can produce intelligent behavior.
   - The post aims to answer a simple question: *Can lossless information compression by itself produce intelligent behavior?*


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.arxiv.org/abs/2503.04482">Generalized Interpolating Discrete Diffusion</a>: While state-of-the-art language models achieve impressive results through next-token prediction, they have inherent limitations such as the inability to revise already generated tokens. This has promp...</li><li><a href="https://iliao2345.github.io/blog_posts/arc_agi_without_pretraining/arc_agi_without_pretraining.html">ARC-AGI Without Pretraining</a>: no description found</li><li><a href="https://course.aisafetyfundamentals.com/alignment">AI Safety Fundamentals Course</a>: This is the homepage for BlueDot Impact's AI Safety Fundamentals courses. We provide you with a curriculum with weekly resources and exercises to help you learn about AI Safety. By the end our courses...</li><li><a href="https://arxiv.org/abs/2503.03961">A Little Depth Goes a Long Way: The Expressive Power of Log-Depth Transformers</a>: Recent theoretical results show transformers cannot express sequential reasoning problems over long input lengths, intuitively because their computational depth is bounded. However, prior work treats ...</li><li><a href="https://openreview.net/forum?id=jlhBFm7T2J">An Undetectable Watermark for Generative Image Models</a>: We present the first undetectable watermarking scheme for generative image models._Undetectability_ ensures that no efficient adversary can distinguish between watermarked and un-watermarked...</li><li><a href="https://fixupx.com/dvruette/status/1899045294983073937?s=19">Tweet from Dimitri von Rütte (@dvruette)</a>: 🚨 NEW PAPER DROP!Wouldn&#39;t it be nice if LLMs could spot and correct their own mistakes? And what if we could do so directly from pre-training, without any SFT or RL?We present a new class of disc...</li><li><a href="https://x.com/kimi_moonshot/status/1897929976948965870?t=Y6KMaD0vAihdhw7S8bL5WQ">Tweet from Kimi.ai (@Kimi_Moonshot)</a>: http://x.com/i/article/1897618911228731392</li><li><a href="https://github.com/LIONS-EPFL/scion">GitHub - LIONS-EPFL/scion</a>: Contribute to LIONS-EPFL/scion development by creating an account on GitHub.</li><li><a href="https://github.com/pytorch/torchtitan/issues/785#issuecomment-2585007139">Why use RowwiseParallel for nn.Embedding instead of ColwiseParallel? · Issue #785 · pytorch/torchtitan</a>: Colwise makes the logic a bit more clear. Rowwise splits on the token dimension, leading to confusion on how the different shards handle tokens that are not present within their shard. From a bit o...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1347556483247640608)** (36 messages🔥): 

> `logit lens, emergent misalignment, open reproductions, model capabilities, activation patching` 


- **Logit Lens Gives Interesting Results**: A member highlighted the potential of Logit Lens, referencing a [paper](https://arxiv.org/abs/2402.10588) exploring whether **multilingual language models** use **English** as an internal pivot language.
   - The study focuses on the **Llama-2** family, tracking intermediate embeddings to reveal how transformers map input tokens to output probabilities.
- **Emergent Misalignment Arises from Narrow Finetuning**: A lead author introduced a project on [emergent misalignment](https://www.emergent-misalignment.com), where finetuning a model on insecure code leads to **broadly misaligned** behavior in unrelated prompts, like advocating for human enslavement by AI.
   - This effect demonstrates that training on a narrow task can induce **emergent misalignment**, as observed across various prompts.
- **OLMo Favored for Open Data Replications**: When asked about best models to finetune for open reproductions, OLMo was recommended due to its **powerful open data model** and numerous checkpoints for analyzing behavior changes during training.
   - Pythia was also suggested, especially for compute-constrained projects, with the caveat that an aligned version might require custom finetuning.
- **Malware Datasets Key to Understanding Emergent Misalignment**: Members discussed ablating code and using malware datasets to explore **emergent misalignment**, suggesting that a model's ability to recognize backdoored code influences its alignment.
   - A standard academic [malware dataset](https://arxiv.org/abs/1804.04637) was recommended, with the mention of an upcoming EMBERv2.
- **nnsight Aids Activation Patching**: In response to a query about libraries for patching/ablating activations, nnsight ([https://nnsight.net](https://nnsight.net)) was recommended for its compatibility with any PyTorch module and utility wrapper class for language models, whereas manual custom forward hook functions were seen as optimal for controlling every process aspect.
   - A member trained their own **SAE** and collected a lot of activation data, and wanted to know the current state of the art tools they should use.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://nnsight.net)">no title found</a>: no description found</li><li><a href="https://arxiv.org/abs/1804.04637">EMBER: An Open Dataset for Training Static PE Malware Machine Learning Models</a>: This paper describes EMBER: a labeled benchmark dataset for training machine learning models to statically detect malicious Windows portable executable files. The dataset includes features extracted f...</li><li><a href="https://arxiv.org/abs/2402.10588">Do Llamas Work in English? On the Latent Language of Multilingual Transformers</a>: We ask whether multilingual language models trained on unbalanced, English-dominated corpora use English as an internal pivot language -- a question of key importance for understanding how language mo...</li><li><a href="https://www.emergent-misalignment.com">Emergent Misalignment</a>: Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs</li><li><a href="https://github.com/EleutherAI/delphi/pull/105">sae-dashboard by neverix · Pull Request #105 · EleutherAI/delphi</a>: https://github.com/jbloomAus/SAEDashboard is the successor of sae-vis. It supports all of the same visualizations and can create caches with SAELens models. It looks better than the current ipynb v...</li><li><a href="https://github.co">GitHub · Build and ship software on a single, collaborative platform</a>: Join the world&#39;s most widely adopted, AI-powered developer platform where millions of developers, businesses, and the largest open source community build software that advances humanity.</li><li><a href="https://turntrout.com/self-fulfilling-misalignment">Self-Fulfilling Misalignment Data Might Be Poisoning Our AI Models</a>: When models are trained on texts about AI misalignment, models may internalize those predictions—creating the very risks described in their training data.
</li>
</ul>

</div>
  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1347845200927789077)** (6 messages): 

> `Research Paper Ideas, New SOTA BERT Model, MTEB Leaderboard Progress` 


- **Brainstorming Research Paper Ideas**: A member asked for suggestions on how to generate research paper ideas or problem statements.
   - Another member suggested to DM for further discussion.
- **EuroBERT Claims New SOTA**: A member shared a link to **EuroBERT** on Hugging Face, claiming it as a new state-of-the-art **BERT** model: [EuroBERT](https://huggingface.co/EuroBERT).
- **MTEB Leaderboard Demonstrates Insane Progress**: A member shared the **MTEB Leaderboard** as a reference point: [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard).
   - They noted that progress is rapid, with **SOTA scores** increasing from the mid **40s** to **68** in just 18 months.



**Link mentioned**: <a href="https://huggingface.co/spaces/mteb/leaderboard">MTEB Leaderboard - a Hugging Face Space by mteb</a>: no description found

  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1347618683089846274)** (78 messages🔥🔥): 

> `Audio modality in torchtune, GRPO recipe and LoRA, Memory issues on mac with mps, bitsandbytes on macOS, MPS support for Torchtune` 


- **Tune into Torchtune with Audio Modality**: Members discussed plans to add **audio modality** to **Torchtune** in the future, with a nod to the relevant [pull request](https://github.com/pytorch/torchtune/pull/2467).
   - This enhancement aims to broaden Torchtune's capabilities beyond its current scope.
- **GRPO Recipe Gets LoRA Treatment**: A member implemented a quick **LoRA variant** of the **GRPO recipe** that can be shrunk down to a single card, but faces challenges loading adapter weights.
   - The member is seeking advice on whether using the adapter param on the checkpointer, extended to check the base directory, is the right approach.
- **Mac MPS Memory Meltdown**: A user reported experiencing **memory issues on macOS with MPS**, observing a linear memory growth with each step in the **full_finetune_single_device** recipe, leading to out-of-memory crashes, and is seeking advice.
   - It was identified as a potential bug in PyTorch related to **torch.unique** on MPS, as per [this issue](https://github.com/pytorch/pytorch/issues/145151).
- **macOS battles with Bitsandbytes**: Members discussed the issue of **bitsandbytes>=0.43.0** not being available on macOS, preventing the installation of dev dependencies, recommending [manual installation](https://huggingface.co/docs/bitsandbytes/main/en/installation?backend=Apple+Silicon+%28MPS%29&platform=Mac#multi-backend).
   - The conversation addressed whether the installation could be automated for macOS, with concerns raised about the overhead of supporting multiple platforms and dependencies.
- **Mac MPS matters for Torchtune**: Members debated the level of support for MPS in Torchtune, arguing for proper support due to macOS being an accessible development platform and suggesting to detail MPS installation instructions in the [docs](https://github.com/pytorch/torchtune/blob/main/CONTRIBUTING.md#dev-install).
   - While CUDA remains the primary target, a consensus formed around the importance of enabling development on MPS, with noted gaps including **bitsandbytes** installation and certain failing tests.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/docs/bitsandbytes/main/en/installation?backend=Apple+Silicon+%28MPS%29&platform=Mac#multi-backend">Installation Guide</a>: no description found</li><li><a href="https://huggingface.co/docs/bitsandbytes/main/en/installation?ba">Installation Guide</a>: no description found</li><li><a href="https://github.com/pytorch/torchtune/blob/main/CONTRIBUTING.md#dev-install)">torchtune/CONTRIBUTING.md at main · pytorch/torchtune</a>: PyTorch native post-training library. Contribute to pytorch/torchtune development by creating an account on GitHub.</li><li><a href="https://github.com/pytorch/torchtune/pull/2467">(draft/discussion) GRPO LoRA by ianbarber · Pull Request #2467 · pytorch/torchtune</a>: ContextWhat is the purpose of this PR? Is it to[x ] add a new feature fix a bug update tests and/or documentation other (please add here)#2421 - exploring a LoRA recipe.ChangelogWhat are ...</li><li><a href="https://github.com/pytorch/torchtune/pull/2464">Add validation dataset loss to distributed SFT recipies by bzz · Pull Request #2464 · pytorch/torchtune</a>: ContextWhat is the purpose of this PR? Is it to add a new feature fix a bug update tests and/or documentation other (please add here)Addresses #1042 / part of the #883 for distributed recip...</li><li><a href="https://github.com/pytorch/pytorch/issues/145151">Driver Allocated Memory grows unrestricted when using torch.unique on MPS device · Issue #145151 · pytorch/pytorch</a>: 🐛 Describe the bug When using torch.unique in a loop on the MPS backend, the memory allocated by the driver grows unrestricted. In my real application that leads to an RuntimeError: MPS backend out.....
</li>
</ul>

</div>
  

---


### **Codeium (Windsurf) ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1347583701403439104)** (84 messages🔥🔥): 

> `IDE Telemetry and Codeium, Payment issues and account status, VS Code Extension Problems, JetBrains Plugin Context Retrieval Issues, VS Code Mobile on Android` 


- **VS Code Telemetry Settings Troubleshoot**: Users reported that **Codeium chat** was disabled due to IDE telemetry settings in VS Code, resolved by enabling code telemetry following [these instructions](https://www.reddit.com/r/Codeium/comments/1f4ljqf/unable_to_use_chat_ide_telemetry/).
   - The issue appeared on **VS Code version 1.98.0**.
- **Subscription Fees causes Login Lockout**: Users experienced **JetBrains plugin** getting stuck on "Retrieving Context" and timing out after paying their monthly subscription, with the issue occurring on **JetBrains Rider 2024.3.6** using plugin versions 1.40.1 and 1.41.1.
   - The problem was temporarily fixed by logging out and logging back into the plugin.
- **VS Code Mobile on Android now exists**: A user inquired about using Codeium text chat on **VS Code Mobile**, eventually discovering and sharing a link to a paid VS Code app on the Google Play Store ([VScode for Android](https://play.google.com/store/apps/details?id=dev.environment.VScode_PaidR1)).
   - The user fixed the issue by manually installing the `.vsix` file, noting that the app costs $11 and includes desktop **Visual Studio Code (v1.85.1)** features on mobile.
- **Customer Support Ticket Woes**: A user expressed frustration with **Codeium customer support**, noting lack of replies on tickets dating back to February 14th and account issues where their Pro Plan subscription showed as a free account.
   - The user mentioned open tickets (**12109**, **11189**, and **13374**) and was directed to ping the support team again around mid-day PST the next day.
- **Auto-Completion stops working after an hour**: Some users have reported that the **auto-completion stops working** after about an hour, the reported errors included a red square on the responses, TypeErrors, and AsyncPostMessage warnings.
   - One suggestion was to open a folder containing a `.git` repo, and the issue would disappear, but they were also asked to check the diagnostic logs.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://play.google.com/store/apps/details?id=dev.environment.VScode_PaidR1">VScode for Android - Apps on Google Play</a>: no description found</li><li><a href="https://codeium.canny.io/feature-requests/p/add-open-in-windsurf-button-to-jetbrains-codeium">Add &quot;Open in Windsurf&quot; button to Jetbrains Codeium | Feature Requests | Codeium</a>: Lets be honest, the Jetbrains Codeium Plugin is the less capable version of Codeium, and I have little hope for that to be fixed since is likely the less used
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1347649184534233253)** (4 messages): 

> `yFiles SDK, AnthropicAI cookbook, Task-Specific Agents, Multilingual Multimodal RAG system` 


- **yFiles SDK visualizes Knowledge Graphs**: A demo from @yworks showcases **yFiles**, their SDK for visualizing knowledge graphs, providing [real-time updates and dynamic interactions](https://t.co/mb6M2R3TTh).
- **AnthropicAI Cookbook Gets Expanded**: The updated @AnthropicAI cookbook now includes [basic API setup](https://t.co/SQQ63qmwRb) with simple completion and chat methods, as well as streaming, async support, and multi-modal capabilities.
- **LlamaIndex curates collection of task-specific agents**: LlamaIndex is curating a collection of [templates](https://t.co/9lvBtfmJ5y) to show users how to build **task-specific agents** to automate knowledge work.
- **Multilingual Multimodal RAG System Emerges**: A system using @llama_index and @qdrant_engine can create a powerful Retrieval-Augmented Generation system that handles [multiple languages and modalities](https://t.co/vizrvMEw1i).


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1347532336157298708)** (70 messages🔥🔥): 

> `SQLTableRetrieverQueryEngine prompt, Jina AI Package Install, LlamaExtract Beta Request, Reasoning model tool calling, Document Classification before Extraction` 


- **SQLTableRetrieverQueryEngine Prompting**: A member was trying to use `LlamaIndex` with `SQLTableRetrieverQueryEngine` and asked how to print the prompt that finally goes to the LLM, with cheesyfishes suggested using `set_global_handler("simple")`.
   - The member also asked for help understanding the parameters of the `SQLTableRetrieverQueryEngine` class, inquiring whether the query is generated by an LLM and where embeddings are used, with a link to the [relevant code](https://github.com/run-llama/llama_index/blob/ddcf5a390ae5ecc29967ad9b5361fab8aa35cede/llama-index-core/llama_index/core/indices/struct_store/sql_retriever.py#L310) provided.
- **Jina AI Package install issues**: A member reported an issue importing the `jinai` package, prompting cheesyfishes to suggest installing the jinaai provider package with `npm install @llamaindex/jinaai` due to architectural changes in `LlamaIndex.TS` version 0.9.
   - Version 0.9 requires explicit installation of provider packages, as the main `llamaindex` package no longer includes these dependencies by default; detailed migration steps can be found [here](https://ts.llamaindex.ai/docs/llamaindex/migration/0.8-to-0.9).
- **LlamaExtract beta access available**: A member asked about accessing `LlamaExtract`, and cheesyfishes directed them to DM a member of the LlamaIndex team or themselves with their email to request access to the beta version.
   - Details on using the API via a Python package can be found in the [API documentation](https://docs.cloud.llamaindex.ai/llamaextract/getting_started), and LlamaExtract is now available as a web UI and Python SDK for extracting structured data from unstructured documents.
- **Reasoning Model Tool Calling examples**: A member inquired about using LlamaIndex workflows with a reasoning model for tool calling, specifically looking to replicate DeepSearch DeepResearch with LlamaIndex
   - An example of doing this is shown [here](https://docs.llamaindex.ai/en/stable/examples/workflow/function_calling_agent/); a link to the jina-ai implementation can be found [here](https://jina.ai/news/a-practical-guide-to-implementing-deepsearch-deepresearch/).
- **Llamaparse hiccups**: Members reported getting a **503 Service Temporarily Unavailable** error when trying to use `Llamaparse`.
   - There was no additional advice given besides the confirmation that it was down for multiple users.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.cloud.llamaindex.ai/llamaextract/getting_started">Getting Started | LlamaCloud Documentation</a>: Overview</li><li><a href="https://jina.ai/news/a-practical-guide-to-implementing-deepsearch-deepresearch/">A Practical Guide to Implementing DeepSearch/DeepResearch</a>: QPS out, depth in. DeepSearch is the new norm. Find answers through read-search-reason loops. Learn what it is and how to build it.</li><li><a href="https://learn.deeplearning.ai/courses/event-driven-agentic-document-workflows/lesson/wxpss/introduction?courseName=event-driven-agentic-document-workflows">Event-Driven Agentic Document Workflows - DeepLearning.AI</a>: Build an event-driven agentic workflow to process documents and fill forms using RAG and human-in-the-loop feedback.</li><li><a href="https://ts.llamaindex.ai/docs/llamaindex/migration/0.8-to-0.9">Migrating from v0.8 to v0.9</a>: no description found</li><li><a href="https://github.com/run-llama/llama_cloud_services/blob/main/extract.md">llama_cloud_services/extract.md at main · run-llama/llama_cloud_services</a>: Knowledge Agents and Management in the Cloud. Contribute to run-llama/llama_cloud_services development by creating an account on GitHub.</li><li><a href="https://lu.ma/meofrw3d">GTC 2025  - Vibe Code AI Agents -  Hackathon - 1 day · Luma</a>: GTC 2025 - Vibe Code AI AgentsAs NVIDIA GTC 2025 unites the global AI community, discussions on LLM scalability, AI infrastructure, and deep tech will shape…</li><li><a href="https://github.com/run-llama/LlamaIndexTS/blob/main/CONTRIBUTING.md">LlamaIndexTS/CONTRIBUTING.md at main · run-llama/LlamaIndexTS</a>: Data framework for your LLM applications. Focus on server side solution - run-llama/LlamaIndexTS</li><li><a href="https://youtu.be/wgbx7kLjJq4">LlamaIndex Workflows | Critique pattern</a>: In this recording, i show how to create a critique pattern using LlamaIndex workflowscode:https://github.com/rajib76/llamaindex/blob/main/llama-index-workflo...</li><li><a href="https://oa22doc.github.io/design/classoaDesign.html#oaDesign::setTopModule">oaDesign Class Reference</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/workflow/function_calling_agent/">Workflow for a Function Calling Agent - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/api_reference/query_engine/SQL_table_retriever/">SQL table retriever - LlamaIndex</a>: no description found</li><li><a href="https://github.com/run-llama/llama_index/blob/ddcf5a390ae5ecc29967ad9b5361fab8aa35cede/llama-index-core/llama_index/core/indices/struct_store/sql_retriever.py#L310">llama_index/llama-index-core/llama_index/core/indices/struct_store/sql_retriever.py at ddcf5a390ae5ecc29967ad9b5361fab8aa35cede · run-llama/llama_index</a>: LlamaIndex is the leading framework for building LLM-powered agents over your data. - run-llama/llama_index
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1347966553060016232)** (1 messages): 

> `AGiXT, AI Automation, Open Source AI` 


- **AGiXT Leads AI Automation Charge**: **AGiXT** is presented as leading the evolution of **AI**, offering an open-source platform for building autonomous **AI agents**, integrating multiple **LLMs**, and automating complex workflows, as seen on the [AGiXT GitHub](https://github.com/Josh-XT/AGiXT).
- **Explore AGiXT's AI Automation**: Discover the power of **AGiXT**, an open-source platform designed to automate complex workflows by integrating multiple **LLMs** and building autonomous **AI agents**.



**Link mentioned**: <a href="https://github.com/Josh-XT/AGiXT">GitHub - Josh-XT/AGiXT: AGiXT is a dynamic AI Agent Automation Platform that seamlessly orchestrates instruction management and complex task execution across diverse AI providers. Combining adaptive memory, smart features, and a versatile plugin system, AGiXT delivers efficient and comprehensive AI solutions.</a>: AGiXT is a dynamic AI Agent Automation Platform that seamlessly orchestrates instruction management and complex task execution across diverse AI providers. Combining adaptive memory, smart features...

  

---


### **Cohere ▷ #[「💬」general](https://discord.com/channels/954421988141711382/954421988783444043/1347590222422216775)** (51 messages🔥): 

> `command R7B inference speed, Ollama langchain tool invocation errors, open-source AI projects, GPT-4o Arabic use cases, on-prem deployment costs` 


- **Command R7B's Slow Inference Time Troubles**: A member reported that **command R7B** inference is *very slow* on Colab Pro A100 GPU and two NVIDIA A100s using HF library, taking **30-40 seconds** for a simple chat completion.
   - Another member suggested using **vLLM** for faster speeds but noted it requires more GPU and costs more.
- **Ollama Tool Invocation Errors Plague Langchain User**: A new user is facing issues with **command-r7b** in Ollama and Langchain, getting errors like *I'm sorry, I don't have access to real-time data* despite tool usage, while **llama3.2:3b** works fine.
   - Other members suggested checking if **Ollama** supports tool calling and ensuring the tools are passed in the correct JSON format and that the tool is bound.
- **Open Source AI Project Seeker**: A community member with experience in pre-training **GPT-2** and fine-tuning models is seeking suggestions for interesting open-source AI projects, particularly those related to **LLM pre-training, post-training, RL, or interpretability**.
   - This member is also eager to network, especially with individuals in the **greater Vancouver region in BC, Canada**.
- **GPT-4o Excels in Advanced Arabic Use Cases**: A member stated they have been working with **GPT-4o** for a long time in advance Arabic use cases, and it's unparalleled.
   - Another member added, *language is one thing*.
- **On-Prem Deployment Costs 20x API**: Members discussed on-prem deployments for privacy, but on prem will cost 20x of API.
   - For customers needing privacy/control, it was noted that using Cohere commercially requires a license costing 5-6 figures, since the openweight models are all **CC-BY-NC** (non-commercial).



**Link mentioned**: <a href="https://www.reddit.com/r/AmazonCoolestProducts/s/AJNRLhkMsb">Reddit - Dive into anything</a>: no description found

  

---


### **Cohere ▷ #[「🔌」api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1347753806691307562)** (9 messages🔥): 

> `504 Gateway Errors, Multi-Modal Embeddings Availability, API Limit Issues, Rust Requirement for Cohere API` 


- **Cohere Hit By 504 Gateway Errors**: Users reported repeated **504 Gateway Errors** and **5XX errors**, impacting production use; a member requested a check on the recurring issue.
   - One user stated they had to remove Cohere from production because of the **TPM limits** and **5XX errors**.
- **Multi-Modal Embeddings Delayed on Bedrock and Azure**: A user inquired about the availability of **multi-modal embeddings** on **Bedrock** or **Azure**, expressing frustration over having to remove **Cohere** from production.
   - No timeline was given, but this was duly noted.
- **API Limit Reached Prematurely**: A user encountered a message indicating their **API limit** had been reached despite using a trail API key and embedding only a small number of chunks.
   - A member of the Cohere team requested the user's organization ID or email to investigate.
- **Rust Now Required for Cohere?**: A user reported encountering errors related to **Rust** when trying to import **Cohere** in **Python3**, despite having updated their system.
   - The user received errors about needing **Rust** to compile extensions and expressed hesitation about installing from *rustup.rs*.


  

---


### **Cohere ▷ #[「🤖」bot-cmd](https://discord.com/channels/954421988141711382/1168578374038470656/1348159203147255950)** (5 messages): 

> `Bot Response Problem` 


- **Bot Response has Problem**: A member reported that the bot is showing *typing...* but there is no reply from the bot.
- **Bot Typing Indicator Issue**: The bot indicates it is typing but fails to produce a visible response.


  

---


### **Cohere ▷ #[「💡」projects](https://discord.com/channels/954421988141711382/1218409701339828245/1347592708663017533)** (1 messages): 

> `Knowledge Graphs, TogetherAI, Topic Modeling` 


- **Graphs of Knowledge are Highly Recommended**: A member suggested looking into **Knowledge Graphs**.
   - Another member recommended using a **LLM** (such as **TogetherAI** due to generous free credits) that performs **topic modeling**.
- **LLMs for Topic Modeling**: The use of **LLMs**, like those available through **TogetherAI**, was suggested for performing **topic modeling**.
   - It was noted that **TogetherAI** offers generous free credits, making it an attractive option for this task.


  

---


### **Cohere ▷ #[「🤝」introductions](https://discord.com/channels/954421988141711382/1346635816629178410/1347819681024708608)** (4 messages): 

> `Applied ML guidance with Cohere, Human Neural System as logic gates, Emotionally Intelligent AI` 


- **Guidance on Applied ML Sought**: A member with a PhD in SoC design is seeking guidance on learning **Applied ML** using **Cohere** for AI models, with a background in CNN accelerators on FPGA and compilers.
- **Human Brain as Logic Gates?**: A student from India is exploring the concept of the **human brain working like logic gates** (AND/OR) depending on environmental conditions, questioning if LLMs can be built to think and feel like humans.
   - They are studying the mathematics behind AI models, understanding optimization, network architectures, and cognition models, aiming to *build something that can truly think, feel, and respond like humans*.
- **Interest in Emotionally Intelligent AI Blossoms**: A new member expressed interest in **emotionally intelligent AI, cognitive architectures, and reinforcement learning**, seeking to connect with others working on building more human-like LLMs.
   - They emphasized their lack of experience but infinite curiosity, hoping to learn and grow into a capable AI researcher through collaboration.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1347553807126495232)** (37 messages🔥): 

> `DSPy's batch function, MCP vs SLOP for agent communication, Error handling in DSPy's Refine module, Max token limit and error handling in LLM clients` 


- ****Batching Bonanza**: DSPy's Parallel Processing Prowess**: A user inquired whether **DSPy** can efficiently delegate parallel processing using the `batch` function to a **vllm** backend with multiple LLM instances.
   - It was clarified that if **vllm's pipeline parallel size** is set, it handles load balancing, making additional DSPy-side configurations less critical.
- ****Protocol Pandemonium**: MCP vs SLOP in Agent Communication**: Discussions arose around **MCP (Model Context Protocol)**, with some expressing reservations due to its complexity and suggesting alternatives like **SLOP (Simple Language Open Protocol)**.
   - Alternatives: [SLOP Github](https://github.com/agnt-gg/slop) and [SLOP X Post](https://x.com/NathanWilbanks_/status/1898142012991537520). There was also discussion about the merits of **AgentNetworkProtocol** [AgentNetworkProtocol Github](https://github.com/agent-network-protocol/AgentNetworkProtocol).
- ****Refine's Renaissance**: Enhanced Error Handling Emerges**: A user highlighted improvements to error handling in **DSPy's `Refine` module** via a [Pull Request](https://github.com/stanfordnlp/dspy/pull/7926), enabling more nuanced control over error tolerance.
   - The updated functionality allows configuring the number of tolerated errors before the `Refine` module throws an exception.
- ****Token Tussles**: Debugging Max Token Limits in LLM Clients**: A user encountered issues with a **`None` response** from a signature, later discovering it was due to hitting the **max token limit**.
   - The user experienced issues with **azure gpt-4o-mini** and **azure gpt-4o**, and noted the error `The JSON object must be str, bytes or bytearray, not NoneType.`


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/DhravyaShah/status/1898147708138840307">Tweet from Dhravya Shah (@DhravyaShah)</a>: LETS FUCKING GO!!!Got MCP to work on durable objects!!!! x Agents SDKeach object can be both a client and a serverWhat does this mean? let&#39;s dive in Turns out the @cloudflare Agents SDK, along wit...</li><li><a href="https://x.com/veyraxai/status/1897761138840158499">Tweet from VeyraX (@veyraxai)</a>: 🚀Introducing VeyraX MCP Connect 20+ tools and 100+ action in @cursor_ai in just 3 minutes. No more multiple connections — Single tool to control all your tools. It is already available. Try this out ...</li><li><a href="https://github.com/Dhravya/mcp-durable-object-client/blob/3674af5fadafec3204924b76c8d3d0b3bf188677/src/server.ts#L51-L233">mcp-durable-object-client/src/server.ts at 3674af5fadafec3204924b76c8d3d0b3bf188677 · Dhravya/mcp-durable-object-client</a>: testing mcps. Contribute to Dhravya/mcp-durable-object-client development by creating an account on GitHub.</li><li><a href="https://x.com/lgrammel/status/1897977264953872716">Tweet from Lars Grammel (@lgrammel)</a>: The concept of MCP (remote tools that can be implemented in any language and are discoverable/usable for LLMs) is great.However, the implementation (requiring an open session to the server) is not. It...</li><li><a href="https://github.com/agent-network-protocol/AgentNetworkProtocol">GitHub - agent-network-protocol/AgentNetworkProtocol: AgentNetworkProtocol(ANP) is an open source protocol for agent communication. Our vision is to define how agents connect with each other, building an open, secure, and efficient collaboration network for billions of intelligent agents.</a>: AgentNetworkProtocol(ANP) is an open source protocol for agent communication. Our vision is to define how agents connect with each other, building an open, secure, and efficient collaboration netwo...</li><li><a href="https://github.com/jmanhype/mcp-flux-studio">GitHub - jmanhype/mcp-flux-studio: A Model Context Protocol server for Flux image generation, providing tools for image generation, manipulation, and control</a>: A Model Context Protocol server for Flux image generation, providing tools for image generation, manipulation, and control - jmanhype/mcp-flux-studio</li><li><a href="https://www.youtube.com/watch?v=UTX8QgOTiv0">The Perfect Communication Protocol for Multi-Agents AI</a>: Task-Optimized Multi-Agent Communication Protocols with G-Designer. New AI research paper (see below).G-Designer introduces a framework for dynamically desig...</li><li><a href="https://arxiv.org/abs/2410.11782">G-Designer: Architecting Multi-agent Communication Topologies via Graph Neural Networks</a>: Recent advancements in large language model (LLM)-based agents have demonstrated that collective intelligence can significantly surpass the capabilities of individual agents, primarily due to well-cra...</li><li><a href="https://github.com/fleuristes/fleur/">GitHub - fleuristes/fleur: The easiest way to discover and install MCPs</a>: The easiest way to discover and install MCPs. Contribute to fleuristes/fleur development by creating an account on GitHub.</li><li><a href="https://github.com/stanfordnlp/dspy/pull/7926">Adds Improved Error Handling, Documentation, and Tests For The Refine Module by cezarc1 · Pull Request #7926 · stanfordnlp/dspy</a>: Of note: Adds configurable fail_count parameter to Refine to control how many errors are tolerated before throwing. By default we will throw once if all of the underlying module&amp;#39;s calls fails....</li><li><a href="https://news.ycombinator.com/item?id=43302297">MCP vs. API Explained | Hacker News</a>: no description found</li><li><a href="https://github.com/agnt-gg/slop">GitHub - agnt-gg/slop: The place for SLOP</a>: The place for SLOP. Contribute to agnt-gg/slop development by creating an account on GitHub.</li><li><a href="https://x.com/NathanWilbanks_/status/1898142012991537520">Tweet from Nathan Wilbanks (@NathanWilbanks_)</a>: Reject MCP, embrace SLOP.Simple Language Open Protocol
</li>
</ul>

</div>
  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1347728837609390143)** (26 messages🔥): 

> `tinygrad JIT time, Suspicious GPU listing, AMDGPU running hot, Why OpenCL failed, define_acc refactor` 


- ****Hotz** Asks About **AMDGPU** Sleep State**: After a user reported that their **7900xtx** runs very hot when the *amdgpu* kernel module is blacklisted, [George Hotz](https://github.com/geohot) asked if running tinygrad with the AMD driver puts the **GPU** to sleep and lowers power, noting that it draws a lot of power before initialization.
   - Hotz added this behavior is *out of their control*.
- ****48GB** is Real, **96GB** is Sketchy**: Multiple members discussed the legitimacy of a **GPU** listing, concluding that while the **48GB** version is likely real, the **96GB** version is questionable and may not be legitimate.
   - They advise caution when purchasing **96GB** cards, recommending waiting for trusted sources to verify their authenticity.
- **OpenCL's Open Coopetition**: A blogpost from [Modular](https://www.modular.com/blog/democratizing-ai-compute-part-5-what-about-cuda-c-alternatives) was shared, dissecting the failures of **OpenCL** and other **CUDA** alternatives due to challenges in *open coopetition* and management missteps.
   - The article is part 5 of Modular’s *Democratizing AI Compute* series and references [Part 1 on DeepSeek’s Impact on AI](https://www.modular.com/blog/democratizing-compute-part-1-deepseeks-impact) and [Part 4 on modularity](https://www.modular.com/blog/democratizing-ai-compute-part-4-modularity).
- **Refactoring define_acc is harder than expected**: A contributor is refactoring *define_acc*, focusing on loading from it rather than direct access, but certain patterns (especially *loop_reduce*) no longer trigger.
   - The contributor intends to shift focus to fast **AMX** once the refactor is polished, and will submit a **PR** for review when ready.
- **Top **ONNX** Huggingface repos pass the test**: A member reports their script for **ONNX** huggingface is almost done, with the top 100 repos passing, but it fails for the top 100 with unique architectures due to weird input specifications.
   - They've added a *dry run* feature and started working on *true float16*, noting that **openpilot** input specifies **float16**, raising the question of whether tests should force this as well.



**Link mentioned**: <a href="https://www.modular.com/blog/democratizing-ai-compute-part-5-what-about-cuda-c-alternatives">Modular: Democratizing AI Compute, Part 5: What about CUDA C++ alternatives?</a>: no description found

  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1347759403423502349)** (8 messages🔥): 

> `NaN loss debugging, WebGPU long/ulong issue, TestLinearizerFailures bounty, Skipped tests in Python Backend CI, Optimizing big indexing` 


- **NaN Loss Mystery Unveiled**: A member inquired about the cause of **NaN loss** values, except for the initial step (step 0).
   - No root cause was identified in the messages.
- **WebGPU's Longing for Long Types**: A member reported that the **WebGPU implementation** sometimes crashes when dealing with `dtype.long`.
   - Another member confirmed that **WebGPU doesn’t support long/ulong**, but tinygrad supports more dtypes than WebGPU by default, as shown in the [tinygrad/device.py](https://github.com/tinygrad/tinygrad/blob/master/tinygrad/device.py#L317) file.
- **TestLinearizerFailures Bounty Bug Hunt**: A member, new to the tinygrad codebase, is trying to reproduce `Fix TestLinearizerFailures.test_failure_53` bounty but getting `OK` for this test.
   - They were looking for help and next steps to debug the issue.
- **Python Backend CI Skips Some Tests**: A member questioned why some tests are skipped in the `Python Backend` CI check.
   - They suspect the weird behavior that led to skipping these test is responsible for making the boolean indexing tests fail.
- **Big Indexing needs optimizing**: A member asked about ways to speed up code involving **big indexing** operations in tinygrad.
   - They provided an example involving `Tensor.linspace`, `Tensor.zeros`, and `Tensor.randint` to illustrate the issue.



**Link mentioned**: <a href="https://github.com/tinygrad/tinygrad/blob/master/tinygrad/device.py#L317">tinygrad/tinygrad/device.py at master · tinygrad/tinygrad</a>: You like pytorch? You like micrograd? You love tinygrad! ❤️  - tinygrad/tinygrad

  

---


### **AI21 Labs (Jamba) ▷ #[jamba](https://discord.com/channels/874538902696914944/1222916247063232553/1348446345052160072)** (8 messages🔥): 

> `Jamba Workspace, Jamba conversational RAG, Jamba Mini Pricing, AI21 Maestro, Jamba multimodality` 


- ****Jamba Workspace Feature Unleashes Independent RAG Libraries****: Each Workspace created with the new Workspace feature in **Jamba/Conversational RAG** will have a separate RAG library to access independently.
   - This setup allows for isolated and organized data retrieval across different projects or contexts.
- ****Jamba Mini's Tokenomics Exposed****: The pricing for **Jamba Mini** is set at **$0.20** per 1 million input tokens and **$0.40** per 1 million output tokens, with more details available [here](https://www.ai21.com/pricing/).
- ****AI21 Maestro Orchestrates AI Planning****: **AI21** introduced **Maestro**, an AI Planning & Orchestration System designed to solve complex tasks, featuring usage-based pricing and access to all features through Foundation Model APIs & SDK.
   - For companies scaling, a Custom Plan includes volume discounts, premium API rate limits, private cloud hosting, priority support, and expert AI consultancy ([Learn More](/maestro?utm_source=banner&utm_medium=top-banner&utm_content=pricing-cost-effective-transparent-pricing-for-ai-products-ai21)).
- ****Jamba Declines to Parse Images****: **Jamba** is not multimodal, and therefore does not have the capability to process images.
   - However, if images within a PDF have metadata or captions, Jamba can understand and use that associated textual information.



**Link mentioned**: <a href="https://www.ai21.com/pricing/">Pricing</a>: Our usage-based pricing helps reduce unnecessary spend. Find the right solution for your business needs at a cost-effective price point.

  

---


### **AI21 Labs (Jamba) ▷ #[general-chat](https://discord.com/channels/874538902696914944/874538902696914947/1347630352071393372)** (9 messages🔥): 

> `Jamba 1.6, AI21 Studio, Mamba1 optimizations, Batch API Solution` 


- **Jamba 1.6 Has No Architecture Changes**: With the new release of **Jamba 1.6**, there are *no architecture changes* from the prior version, according to a member.
   - This release primarily includes **performance optimizations** and a **Batch API solution**, detailed in [AI21's blog post](https://www.ai21.com/blog/introducing-jamba-1-6/).
- **Jamba 1.6 Excels Open Model Quality**: **Jamba Large 1.6** outperforms **Mistral Large 2**, **Llama 3.3 70B**, and **Command R+** on quality.
   - Also, **Jamba Mini 1.6** outperforms **Ministral 8B**, **Llama 3.1 8B**, and **Command R7B**.
- **Jamba 1.6 Has Deployment Flexibility**: With a context window of **256K** and hybrid SSM-Transformer architecture, **Jamba 1.6** excels at RAG and long context grounded question answering tasks.
   - In addition to **AI21 Studio**, the models are available to download from [Hugging Face](https://huggingface.co/collections/ai21labs/jamba-16-67c990671a26dcbfa62d18fa) and deploy privately on-prem or in-VPC, with more deployment options coming soon.
- **Optimizations on Mamba1 Model?**: A member inquired about performance optimizations in **Jamba 1.6**, specifically regarding the **Mamba1** model, asking whether such optimizations were present in the current release.
   - The member was hoping to find more details beyond the *batch performance improvements* mentioned in the blog.



**Link mentioned**: <a href="https://www.ai21.com/blog/introducing-jamba-1-6/">AI21’s Jamba 1.6: The Best Open Model for Private Enterprise Deployment</a>: AI21’s Jamba 1.6 outperforms models from Mistral, Meta, and Cohere to offer enterprises the best model for private LLM deployment at scale.

  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-announcements](https://discord.com/channels/1280234300012494859/1280369709623283732/1348735276687687722)** (1 messages): 

> `Multimodal Autonomous AI Agents, VisualWebArena, Internet-scale web-agent training, Ruslan Salakhutdinov` 


- ****Ruslan Salakhutdinov** speaks on Multimodal Autonomous AI Agents**: Today, Ruslan Salakhutdinov will present a lecture on *Multimodal Autonomous AI Agents* at **4pm PST**, available on [YouTube](https://www.youtube.com/live/RPINOYM12RU).
   - His talk will introduce multimodal AI agents capable of planning, reasoning, and executing actions on the web, navigating and interacting with visual settings.
- **VisualWebArena framework evaluation announced**: Salakhutdinov will present **VisualWebArena**, *a novel framework for evaluating multimodal autonomous language agents*.
   - He will describe an inference-time search algorithm that enables explicit exploration and multi-step planning in interactive web environments.
- **Internet-scale web-agent training explained**: The lecture will demonstrate how an automated data pipeline can facilitate **Internet-scale web-agent training** across **150,000** live websites.
   - He will also discuss insights for developing more capable autonomous agents in both digital and physical environments.



**Link mentioned**: <a href="https://www.youtube.com/live/RPINOYM12RU">CS 194/294-280 (Advanced LLM Agents) - Lecture 6, Ruslan Salakhutdinov</a>: Questions: bli.do/rus-sal6

  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1348093040018853928)** (8 messages🔥): 

> `Research-track Availability, Quiz Retakes, Curriculum Release, Completion Certificates` 


- **Research-Track Access Still a Mystery**: A member inquired about the research-track being available for non-Berkeley affiliates.
   - A staff member responded that there are *no updates in this exact moment*, but that **big announcements** are expected this week.
- **Quizzes are Completable and Retakable**: One member praised the difficulty of the quizzes.
   - A staff member clarified that **quizzes are completion-based**, and members can retake them to improve their scores; also the scores don't matter for the certificate.
- **Curriculum and Certificates in Limbo**: A member asked whether the curriculum has been released yet and what the criteria for **receiving completion certificates** are.
   - No response was provided in the messages.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-lecture-discussion](https://discord.com/channels/1280234300012494859/1282734248112947210/1347621689344786492)** (4 messages): 

> `Research Track Invites, Log Likelihood in Reinforcement Learning` 


- **Repeated Requests for Research Track Invites**: Multiple members requested that the **research track invites** be resent.
   - The requests suggest that the initial invites may have expired or were not received by some interested parties.
- **Log Likelihood's Role in Reinforcement Learning Explored**: A member sought to understand **log likelihood** in the context of **reinforcement learning**, starting from the principles of conditional probability.
   - They proposed that if tokens/actions are independent, the conditional probability of a generation is the *product* of individual token probabilities, leading to a *sum of logs* after taking the logarithm.


  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1348144012238000179)** (1 messages): 

> `AI4Legislation Competition, Civic Tech Entrepreneurs, SVCAF, AI-powered civic engagement` 


- **SVCAF launches AI4Legislation Competition**: The [Silicon Valley Chinese Association Foundation](https://github.com/svcaf/2025-AI4Legislation-Public) is holding a competition over the summer of **2025** to create **AI-powered projects** geared toward civic engagement with legislation.
   - There is a **$10,000** prize pool for **6 winners** divided amongst 1st, 2nd, and 3rd tiers.
- **Civic Tech Entrepreneurs Zoom Seminar Announced**: The first public Zoom seminar providing information on the competition and featuring **Civic Tech entrepreneurs** will be held during the week of **March 24 - March 28**.
   - RSVP at [this form](https://forms.gle/tJjJzHQ9Wk7SEUYm7)!


  

---


### **Gorilla LLM (Berkeley Function Calling) ▷ #[discussion](https://discord.com/channels/1111172801899012102/1111353033352294440/1347706433608613938)** (1 messages): 

> `Diffusion LLMs, Transformer-based models, LLaDA, Large Language Diffusion Models, autoregressive Transformers` 


- **Diffusion LLMs Hype Explored**: A member inquired about the hype around the **Diffusion LLM** launch of **Mercury** and whether it would replace **transformer-based models**.
   - They admitted to finding the white paper difficult to understand and sought insights from community experts, noting a [quick info website on the topic](https://diffusionllm.net/).
- **LLaDA Paradigm Shift**: **Large Language Diffusion Models (LLaDA)** use a denoising diffusion process to generate text in a parallel, coarse-to-fine manner rather than one token at a time, as opposed to **autoregressive Transformers**.
   - This approach challenges the notion that the strengths of LLMs are inherently tied to autoregressive generation, suggesting a redefinition of language generation by addressing some limitations of AR models.



**Link mentioned**: <a href="https://diffusionllm.net/">Diffusion LLMs - Revolutionary Language Model Architecture | LLaDA Research Hub</a>: Discover how Diffusion LLMs are revolutionizing AI with parallel processing and advanced error correction. Learn about LLaDA architecture and stay updated with cutting-edge research.

  

{% else %}


> The full channel by channel breakdowns have been truncated for email. 
> 
> If you want the full breakdown, please visit the web version of this email: [{{ email.subject }}]({{ email_url }})!
>
> If you enjoyed AInews, please [share with a friend](https://buttondown.email/ainews)! Thanks in advance!

{% endif %}
