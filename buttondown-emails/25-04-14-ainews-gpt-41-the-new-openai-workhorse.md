---
id: 53c5605c-6852-4809-bb57-e7a742de7da3
title: '[AINews] GPT 4.1: The New OpenAI Workhorse'
date: '2025-04-15T05:16:26.134697Z'
status: sent
type: public
source: api
metadata: {}
original_slug: ainews-gpt-41-the-new-openai-workhorse
---

<!-- buttondown-editor-mode: plaintext -->**GPT 4.1 is all you need from OpenAI?**

> AI News for 4/11/2025-4/14/2025. We checked 7 subreddits, [**433** Twitters](https://twitter.com/i/lists/1585430245762441216) and **29** Discords (**211** channels, and **16961** messages) for you. Estimated reading time saved (at 200wpm): **1382 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

GPT 4.1 links:

- https://openai.com/index/gpt-4-1/
- New benchmarks: [MRCR](https://huggingface.co/datasets/openai/mrcr) and [GraphWalks](https://huggingface.co/datasets/openai/graphwalks)
- New [prompting guide](https://platform.openai.com/docs/guides/text?api-mode=responses#prompting-gpt-4-1-models) and [cookbook](https://cookbook.openai.com/examples/gpt4-1_prompting_guide)

and a new interview published on Latent Space:

https://youtu.be/y__VY7I0dzU


---


{% if medium == 'web' %}


**Table of Contents**

[TOC] 

{% else %}

The **Table of Contents** and **Channel Summaries** have been moved to the web version of this email: [{{ email.subject }}]({{ email_url }})!

{% endif %}


---

# AI Twitter Recap

**GPT-4.1 Release and Performance**

- **Availability and Features**: [@sama](https://twitter.com/sama/status/1911830886896799931) announced that **GPT-4.1, GPT-4.1 mini, and GPT-4.1 nano** are now available in the API, emphasizing their strengths in **coding, instruction following, and handling long contexts** (up to 1 million tokens).  [@kevinweil](https://twitter.com/kevinweil/status/1911833354682401148) notes that **GPT-4.1** achieves a **54 score on SWE-bench verified**.
- **Instruction Following**: [@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1911860099829674184) points out that **GPT-4.1** follows instructions more reliably than **GPT-4o**, particularly in **format adherence, complying with negative instructions, and ordering**.
- **Pricing and Cost**: [@stevenheidel](https://twitter.com/stevenheidel/status/1911830168118923291) states **GPT-4.1-nano** is the **cheapest and fastest model** released, costing **$0.10/1M input ($0.03 cached) and $0.40/1M output**.
- **Coding Performance**: [@omarsar0](https://twitter.com/omarsar0/status/1911870478857437540) highlights that, according to **Windsurf AI**, **GPT-4.1** shows a **60% improvement over GPT-4o** on internal benchmarks like the **SWE-benchmark**, reduces the need to read unnecessary files by 40%, and modifies unnecessary files 70% less. [@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1911859923161428002) states it is significantly more skilled at frontend coding and has reliable tool use. [@polynoamial](https://twitter.com/polynoamial/status/1911831926241153170) mentions **GPT-4.1 achieves 55% on SWE-Bench Verified** without being a reasoning model.
- **Integration and Support**: [@llama_index](https://twitter.com/llama_index/status/1911863053257445713) mentions Llama Index now has day 0 support for **GPT-4.1**.
- **Initial Impressions**: [@aidan_mclau](https://twitter.com/aidan_mclau/status/1911850291026362426) notes that startup engineers were amazed by **GPT-4.1 mini/nano**, finding it comparable to **GPT-4o** but much cheaper. [@aidan_mclau](https://twitter.com/aidan_mclau/status/1911847214168850805) describes it as a **Pareto optimal, Swiss Army knife API model**, and an upgrade over newssonnet for agent stacks.
- **Limited Availability on ChatGPT**: [@DanHendrycks](https://twitter.com/DanHendrycks/status/1911837235521163670) suggests that the free **GPT-4.1 mini** might be intentionally limited on **ChatGPT** to incentivize college students to subscribe to **ChatGPT Plus**.
- **Naming Conventions**: [@polynoamial](https://twitter.com/polynoamial/status/1911843302770643004) joked about naming models. [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1911832534796886439) notes that the naming scheme for **GPT** models follows **GPT-4.10**, so it comes after **GPT-4.5**, while [@kevinweil](https://twitter.com/kevinweil/status/1911795255877198311) joked that it would not get better at naming this week.
- **Deprecation of GPT-4.5**: [@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1911860805810716929) announced that **GPT-4.5 Preview** in the API will be deprecated starting today and fully turned off on July 14, as **GPT-4.1 offers improved or similar performance**.
- **Negative Reviews**: [@scaling01](https://twitter.com/scaling01/status/1911852197714731276) advises against using **GPT-4.1-nano**, describing it as a terrible model.
[@scaling01](https://twitter.com/scaling01/status/1911847193465471374) reports the GPT-4.1 API version is worse than Optimus Alpha.

**Model Benchmarks and Comparisons**

- **Search Arena Leaderboard**: [@AravSrinivas](https://twitter.com/AravSrinivas/status/1911849161869410355) reports that **Perplexity's Sonar API** is tied with **Gemini-2.5 Pro** for the #1 spot in the LM Search Arena leaderboard. [@lmarena_ai](https://twitter.com/lmarena_ai/status/1911842298914328959) reports that **Gemini-2.5-Pro-Grounding and Perplexity-Sonar-Reasoning-Pro** top the leaderboard.
- **Llama 4 ELO Drop**: [@casper_hansen_](https://twitter.com/casper_hansen_/status/1911332387817931161) reports that **Llama 4** quietly dropped from **1417 to 1273 ELO**, on par with **DeepSeek v2.5**.
- **Google Gemini 2.5 Pro**: [@abacaj](https://twitter.com/abacaj/status/1911529618089427122) said that Google has finally made the best model with Gemini 2.5 pro.  [@omarsar0](https://twitter.com/omarsar0/status/1911451522703020189) is surprised at how good **Gemini 2.5 Pro** is at debugging and refactoring, and that it's one of the best models at understanding larger codebases.
- **Gemini 2.0 Flash**: [@_philschmid](https://twitter.com/_philschmid/status/1911862052852744642) reports **Gemini 2.0 Flash** is **$0.1/$0.4** (input/output per 1M tokens) with strong scores on GPQA Diamond, Multilingual MMLU, and MMMU.
- **Mistral Models**: [@casper_hansen_](https://twitter.com/casper_hansen_/status/1911382474640220546) stated that Long Mistral models are great and their latest **24B model** is very competitive.
- **Nvidia Llama Nemotron-Ultra**: [@adcock_brett](https://twitter.com/adcock_brett/status/1911450216164700252) notes **Nvidia released Llama Nemotron-Ultra**, a **253B** parameter reasoning AI that beats DeepSeek R1, Llama 4 Behemoth and Maverick, and is fully open-source.
- **Meta Llama 4**: [@adcock_brett](https://twitter.com/adcock_brett/status/1911450182937346285) details that **Meta released the Llama 4** family of natively multimodal, open-source models with context windows up to 10M tokens, including the 109B param Scout, 400B param Maverick, and a third, 2T param Behemoth.  [@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1911841914590015586) notes Llama 4 Scout features an unprecedented 10 million-token context window, Maverick beats GPT-4o’s reported benchmarks, and Behemoth claims to outperform GPT-4.5 and Claude 3.7 Sonnet.
- **Kimina-Prover vs. other models**: [@_lewtun](https://twitter.com/_lewtun/status/1911793153931100180) notes that the new programming language Lean has **Kimina-Prover** beating **Gemini 2.5 Pro and o3-mini** on Olympiad-level math with just 7B parameters!
- **GPT-4.1 vs DeepSeek-V3**: [@scaling01](https://twitter.com/scaling01/status/1911831700964872531) states that **GPT-4.1** underperforms **DeepSeek-V3-0324 by over 10% on AIME** and is 8x more expensive and also underperforms on GPQA.
- **GPT-4.1 vs. GPT-4.5**: [@scaling01](https://twitter.com/scaling01/status/1911828552452112536) states that **GPT-4.1 outperforms GPT-4.5** in AIME and MMLU.

**Robotics and Embodied AI**

- **Hugging Face Acquisition**: [@ben_burtenshaw](https://twitter.com/ben_burtenshaw/status/1911843020309213547) reports that **Hugging Face** acquired **Pollen Robotics**, an open source robot manufacturer.
- **Fourier's Open-Source Humanoid**: [@adcock_brett](https://twitter.com/adcock_brett/status/1911450377175589313) notes **Fourier’s** fully open-source humanoid robot.
- **Samsung & Google Partnership**: [@adcock_brett](https://twitter.com/adcock_brett/status/1911450160078467386) notes **Samsung** announced a partnership with **Google** to power its **Ballie** home robot with Google's **Gemini** and its own multimodal AI models.

**AI Research and Papers**

- **Reflection in Pre-Training**: [@omarsar0](https://twitter.com/omarsar0/status/1911442761238340095) summarizes a paper arguing that reflection emerges during pre-training and introduces adversarial reasoning tasks to show that self-reflection and correction capabilities improve with compute, even without supervised post-training.
- **Reinforcement Learning and Reasoning**: [@rasbt](https://twitter.com/rasbt/status/1911494805101986135) summarizes a paper showing that reinforcement learning (RL) can lead to longer responses in reasoning models, not because they are needed for accuracy, but because RL training favors longer responses.
- **Multimodal Models Scaling Laws**: [@TheAITimeline](https://twitter.com/TheAITimeline/status/1911633260582523332) summarizes a scaling laws analysis involving 457 native multimodal models (NMMs), revealing that early-fusion architectures outperform late-fusion ones and that Mixture of Experts (MoEs) significantly boosts performance.
- **Paper List**: [@TheAITimeline](https://twitter.com/TheAITimeline/status/1911633257952575492) posted a list of top AI/ML research papers, and [@dair_ai](https://twitter.com/dair_ai/status/1911444942523621550) similarly shared their top AI papers.
- **Visual Tokenizers**: [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1911639044406329846) notes that GigaTok improves image reconstruction, generation, and representation learning when scaling visual tokenizers.

**Other Model and AI Tool Releases**

- **Deep Cogito Models**: [@adcock_brett](https://twitter.com/adcock_brett/status/1911450441457557816) notes that Deep Cogito emerged from stealth with Cogito v1 Preview, a new family of open-source models.
- **Runway Gen 4 Turbo**: [@adcock_brett](https://twitter.com/adcock_brett/status/1911450331470258198) shares that Runway released Gen 4 Turbo, a faster version of its video model, available to all users, including those on the free tier.
- **Midjourney V7**: [@adcock_brett](https://twitter.com/adcock_brett/status/1911450308795904091) reports that Midjourney released V7, with improved quality, enhanced prompt adherence, and a voice-capable Draft Mode.
- **Microsoft Copilot Updates**: [@adcock_brett](https://twitter.com/adcock_brett/status/1911450285760708712) mentions that Microsoft upgraded its Copilot app with new memory capabilities, web browsing actions, and vision features.
- **Amazon AI**: [@adcock_brett](https://twitter.com/adcock_brett/status/1911450262977368259) says that Amazon released a speech-to-speech AI called "Nova Sonic" and launched Reel 1.1 AI for extended 2-min video generations.
- **Nvidia Cartoon AI**: [@adcock_brett](https://twitter.com/adcock_brett/status/1911450240143536333) shares that Nvidia and Stanford researchers unveiled an AI technique to generate consistent, minute-long cartoons.
- **DolphinGemma**: [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1911767367534735832) introduced DolphinGemma, an AI helping us dive deeper into the world of dolphin communication. 🐬, and is an audio to audio model.

**AI Infrastructure and Tooling**

- **OpenAI Infrastructure Scale**: [@sama](https://twitter.com/sama/status/1911504090989035824) mentioned that the scale of computing systems at OpenAI is insane and they need help.
- **ElevenLabs MCP Integration**: [@adcock_brett](https://twitter.com/adcock_brett/status/1911450399585796491) reports **ElevenLabs** launched its MCP server integration, enabling platforms like **Claude** and **Cursor** to access AI voice capabilities.
- **Qdrant + n8n**: [@qdrant_engine](https://twitter.com/qdrant_engine/status/1911700608731521450) notes that **Qdrant** and **n8n** are automating processes beyond similarity search.
- **LangChain Tools**: [@LangChainAI](https://twitter.com/LangChainAI/status/1911449301542195582) promotes an open-source library connecting any LLM to MCP tools for custom agents, featuring integration with LangChain and support for web browsing, Airbnb search, and 3D modeling.
- **Hamel Husain Chrome Extension**: [@HamelHusain](https://twitter.com/HamelHusain/status/1911521751739351509) created a Chrome extension that allows you save an entire Gemini chat (via aistudio) into a gist or copy as markdown, and also has one for Claude.

**AI Strategy and Discussion**

- **Open Source Robotics**: [@ClementDelangue](https://twitter.com/ClementDelangue/status/1911768941107511624) advocates for making AI robotics open-source.
- **Prioritizing Medical Diagnostics**: [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1911620129227776011) notes that better diagnostics + care delivery are more impactful than finding a new chemotherapy drug for curing cancer.
- **LLMs and Search Engines**: [@rasbt](https://twitter.com/rasbt/status/1911467070975271217) doesn’t think LLMs will replace search engines.
- **Conciseness via RL**: [@TheAITimeline](https://twitter.com/TheAITimeline/status/1911633319910928756) summarizes research uncovering a correlation between conciseness and reasoning accuracy and a method for achieving more concise reasoning in LLMs via a secondary RL phase.
- **Developer Experience**: [@sedielem](https://twitter.com/sedielem/status/1911821106811679094) highlights importance of developer experience.
- **Value of Expertise in RAG**: [@HamelHusain](https://twitter.com/HamelHusain/status/1911830144635084930) emphasizes the value of talking to people who have spent lots of time optimizing retrieval & search to get better at RAG.
- **Future of AI**: [@scaling01](https://twitter.com/scaling01/status/1911187189548933143) shares that the base case for LLMs is that over the next few years they’ll evolve into hyper-specialized autistic superintelligences that excel in domains where verification is straightforward.

**Humor and Miscellaneous**

- **Flat Organizations**: [@typedfemale](https://twitter.com/typedfemale/status/1911213477118845086) made a joke about flat organizations.
- **Hot Sauce**: [@vikhyatk](https://twitter.com/vikhyatk/status/191174862499812563) joked not to try "murder hornet" hot sauce 5 mins before bedtime.
- **Overhyped Valuations**: [@andrew_n_carr](https://twitter.com/andrew_n_carr/status/1911429251804364835) talks about SSI valuation.
- **Personal Anecdotes**: [@DavidSHolz](https://twitter.com/DavidSHolz/status/1911801507571437589) accidentally asked a friend how they were enjoying "jew york" due to autocorrect.
[@sjwhitmore](https://twitter.com/sjwhitmore/status/1911286312365342759) stated that they’ll put their baby to sleep and 30 min later catch themselves looking at photos of him.
[@willdepue](https://twitter.com/willdepue/status/1911591028697833779) mentioned openai hunting cap is a must for the next podcast and [@sama](https://twitter.com/sama/status/1911496563157000568) bought a lot of silly baby things that they haven't needed, but recommends a cradlewise crib and a lot more burp rags than you think you could possibly need.

---

# AI Reddit Recap

## /r/LocalLlama Recap


### Theme 1. "Exciting Advancements in GLM-4 Reinforcement Learning Models"

- **[glm-4 0414 is out. 9b, 32b, with and without reasoning and rumination](https://www.reddit.com/r/LocalLLaMA/comments/1jz2iuc/glm4_0414_is_out_9b_32b_with_and_without/)** ([Score: 190, Comments: 64](https://www.reddit.com/r/LocalLLaMA/comments/1jz2iuc/glm4_0414_is_out_9b_32b_with_and_without/)): **GLM-4 0414 has been released, introducing six new models of sizes 9B and 32B, with and without reasoning and rumination capabilities. The models include **GLM-Z1-32B-0414**, a reasoning model with deep thinking capabilities developed based on GLM-4-32B-0414 through cold start, extended reinforcement learning, and further training on tasks like mathematics, code, and logic. **GLM-Z1-Rumination-32B-0414** is a deep reasoning model with rumination capabilities, capable of deeper and longer thinking to solve more open-ended and complex problems. **GLM-Z1-9B-0414** is a 9B parameter model employing all the aforementioned techniques, exhibiting excellent capabilities in mathematical reasoning and general tasks, achieving top-ranked performance among open-source models of the same size.** GLM-Z1-9B-0414 is considered a surprise, achieving an excellent balance between efficiency and effectiveness, making it a powerful option for users seeking lightweight deployment. The models demonstrate significant improvements in mathematical abilities, research-style writing, and the capability to solve complex tasks.

  - A commenter notes that the new 32B models have only **2 kv value heads**, resulting in the KV cache taking up about four times less space than on Qwen 2.5 32B, and wonders if this might cause issues with handling long context.
  - Another commenter is impressed with the benchmarks, mentioning that GLM models have been around since LLama 1 days and have always been very good, but feels they need better marketing in the West as they seem to go under the radar.
  - A commenter appreciates that the models included the [SuperGPQA](https://www.reddit.com/r/LocalLLaMA/comments/1j3byj5/bytedance_unveils_supergpqa_a_new_benchmark_for/) benchmark results, making the models more comparable with many others.


### Theme 2. "DeepSeek's Open-Source Contributions to AI Inference"

- **[DeepSeek is about to open-source their inference engine](https://i.redd.it/1am95yongrue1.png)** ([Score: 1312, Comments: 92](https://www.reddit.com/r/LocalLLaMA/comments/1jytw62/deepseek_is_about_to_opensource_their_inference/)): **DeepSeek is about to open-source their inference engine, which is a modified version based on **vLLM**. They are preparing to contribute these modifications back to the community. An article titled _'The Path to Open-Sourcing the DeepSeek Inference Engine'_ outlines their motivations and steps, including challenges like codebase divergence, infrastructure dependencies, and limited maintenance bandwidth. They express gratitude towards the open-source ecosystem and plan to collaborate with existing projects to modularize features and share optimizations, aiming to enhance artificial general intelligence (**AGI**) for the benefit of humanity. More details can be found in their [GitHub repository](https://github.com/deepseek-ai/open-infra-index/tree/main/OpenSourcing_DeepSeek_Inference_Engine).** The original poster expresses enthusiasm about DeepSeek's commitment to the community, particularly appreciating their goal _'with the goal of enabling the community to achieve state-of-the-art (SOTA) support from Day-0.'_ There is excitement about the potential positive impact of DeepSeek's contributions on the open-source AI community.

  - One user points out that DeepSeek may not be directly open-sourcing their inference engine but will contribute their improvements to **vLLM** and **sglang**, as their fork is too outdated.
  - Another commenter expresses deep appreciation for DeepSeek, comparing their love for the company to their love for Wikipedia.
  - A user feels that the release of DeepSeek's R1 was a pivotal moment in the AI race, noting that while it wasn't the smartest or cheapest model, it signaled alternatives to OpenAI like **Claude**, **Gemini**, and DeepSeek, and appreciates their ongoing innovation in the open-source field.

- **[DeepSeek will open-source parts of its inference engine — sharing standalone features and optimizations instead of the full stack](https://github.com/deepseek-ai/open-infra-index/blob/main/OpenSourcing_DeepSeek_Inference_Engine/README.md)** ([Score: 252, Comments: 9](https://www.reddit.com/r/LocalLLaMA/comments/1jysiwc/deepseek_will_opensource_parts_of_its_inference/)): **DeepSeek will open-source parts of its inference engine by sharing standalone features and optimizations instead of releasing the full stack. They are working on porting their optimizations to popular open-source inference engines like **vLLM**, **llama.cpp**, and **kobold**.** Some believe the title is misleading, implying DeepSeek is withholding parts of their stack. However, others feel that by porting their optimizations to popular open-source inference engines, DeepSeek is contributing more effectively to the community. Users are optimistic about improved inference performance from these contributions.

  - Commenters note that DeepSeek is enhancing popular open-source inference engines like **vLLM**, **llama.cpp**, and **kobold** by porting their optimizations.
  - Some users are excited about the potential for better inference performance as a result of DeepSeek's contributions.
  - Users are asking if there is anything available now from DeepSeek for personal projects.


## Other AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding


### Theme 1. "Revolutionizing Science: OpenAI's New Reasoning Models"

- **[Scientific breakthroughs are on the way](https://i.redd.it/61jy8q8tctue1.jpeg)** ([Score: 724, Comments: 207](https://www.reddit.com/r/singularity/comments/1jz0ppu/scientific_breakthroughs_are_on_the_way/)): **OpenAI is about to release new reasoning models called **o3** and **o4-mini** *that are able to independently develop new scientific ideas for the first time* [[1]](https://www.theinformation.com/articles/openais-latest-breakthrough-ai-comes-new-ideas). These AI models can process knowledge from different specialist areas simultaneously and propose innovative experiments—an ability previously considered a human domain. Early versions have shown promising results: *Scientists at Argonne National Laboratory were able to design complex experiments in hours instead of days using early versions of these models.* OpenAI plans to charge up to $20,000 a month for these advanced services, which would be 1000 times the price of a standard ChatGPT subscription.** The technology could dramatically accelerate the scientific discovery process, especially when combined with AI agents capable of controlling simulators or robots to directly test and verify generated hypotheses. This represents a potential revolution in the field, shifting abilities previously thought to be exclusive to humans to AI.

  - Some users are skeptical about OpenAI charging $20,000 a month for these AI models, questioning why the company doesn't use them to solve major problems themselves.
  - Others believe the information is credible due to the source's accuracy regarding OpenAI news, suggesting possible intentional leaks from the company.
  - There's confusion and speculation about the high subscription fee, with users recalling previous instances where rumored prices were higher than the actual release prices.


### Theme 2. "Exciting AI Model Innovations and Competitive Updates"

- **[GPT 4.1 with 1 million token context. 2$/million input and 8$/million token output. Smarter than 4o.](https://i.redd.it/fw34ped81uue1.jpeg)** ([Score: 313, Comments: 140](https://www.reddit.com/r/singularity/comments/1jz42ic/gpt_41_with_1_million_token_context_2million/)): **GPT-4.1 is announced as the flagship model for complex tasks, featuring a **1 million token context window** and a maximum output capacity of **32,768 tokens**. Pricing is set at **$2 per million tokens** for input and **$8 per million tokens** for output, with additional information about cached input costs. The model claims enhanced intelligence compared to previous versions.** The original poster emphasizes that GPT-4.1 is *smarter than 4o*, highlighting its advanced capabilities and suggesting it as a significant improvement over previous models.

  - Users compare GPT-4.1 to **Google's Gemini models**, discussing pricing and performance differences, and some express a wish for lower costs.
  - There is skepticism about how effectively GPT-4.1 utilizes its **1 million token context window**, with mentions that models like Gemini 2.5 can handle about *100k tokens flawlessly*.
  - Some speculate that GPT-4.1 may lead to the discontinuation of **GPT-4.5**, and express hope that upcoming models like **o4-mini** will be state-of-the-art.

- **[OpenAI announces GPT 4.1 models and pricing](https://platform.openai.com/docs/models/compare?model=gpt-4.1)** ([Score: 245, Comments: 119](https://www.reddit.com/r/OpenAI/comments/1jz450k/openai_announces_gpt_41_models_and_pricing/)): **OpenAI has announced the release of **GPT 4.1** models along with their pricing details.** The announcement has generated mixed reactions, with some users expressing frustration over the proliferation of models and others discussing the availability and improvements of **GPT‑4.1**.

  - One user expresses frustration over the multitude of models, stating they're *so sick of this mess of random models*.
  - Another points out that **GPT‑4.1** will only be available via the API, noting that improvements have been gradually incorporated into the latest version of **GPT‑4o** in ChatGPT.
  - Some users joke about the knowledge cutoff being June 2024, humorously wishing they were *as gullible as GPT 4.1* 😂.

- **[Kling 2.0 will be unveiled tomorrow.](https://i.redd.it/jtux6mutqrue1.jpeg)** ([Score: 281, Comments: 29](https://www.reddit.com/r/singularity/comments/1jyupii/kling_20_will_be_unveiled_tomorrow/)): **Kling 2.0 will be unveiled tomorrow, **April 15, 2025, at 6:00 AM GMT**. The announcement includes an image with a dynamic green background and the slogan *'From Vision to Screen'*, emphasizing innovation and technology. More details can be found at [https://x.com/Kling_ai/status/1911702934183882986](https://x.com/Kling_ai/status/1911702934183882986) and [https://xcancel.com/Kling_ai/status/1911702934183882986](https://xcancel.com/Kling_ai/status/1911702934183882986).** The promotional image conveys excitement and anticipation for **Kling 2.0**, capturing attention with its dynamic design. The slogan suggests a significant advancement from previous versions, building enthusiasm among potential users.

  - Users are amazed at the rapid release of **Kling 2.0**, with one noting that *'version 1.6 is still number 1'*.
  - Discussion highlights how this last week has been *'WILD'*, with numerous AI advancements like **Midjourney v.7**, **OpenAI GPT-4.1**, and **Google Agentspace Boxing**.
  - There is anticipation for new features in **Kling 2.0**, such as longer video generation, as users are *'stuck at 5-10 sec'* currently.



---

# AI Discord Recap

> A summary of Summaries of Summaries by Gemini 2.0 Flash Thinking

**Theme 1. GPT-4.1 Models: Release, Performance, and Availability**

- **OpenAI Unleashes GPT-4.1, Benchmarks Beat 4o**: [OpenAI's blog post](https://openai.com/index/gpt-4-1/) announced **GPT-4.1**, touted for *long-context reasoning*, with benchmarks showing ~**10%** improvement over **GPT-4o**.  Windsurf AI immediately integrated it, offering [free unlimited access for a week](https://x.com/windsurf_ai/status/1911833698825286142), while OpenRouter launched [GPT-4.1, Mini, and Nano versions](https://openrouter.ai/announcements), revealing **Optimus Alpha** and **Quasar Alpha** as early test versions of **GPT-4.1**.
- **Windsurf Waves Free GPT-4.1 for Users**: [Windsurf AI](https://windsurf.ai) made **GPT-4.1** its new default model, offering [free unlimited usage for one week](https://x.com/windsurf_ai/status/1911833698825286142) on all plans, then at a discounted rate of **0.25 credits per use**.  Cursor Community members anticipate [GPT-4.1 becoming the new standard](https://discord.com/channels/1074847526655643750/1074847527708393565/1360330465533497616), with 4.5 being deprecated as users migrate to 4.1.
- **Aider v0.82.0 Embraces GPT-4.1 Patch Format**: [Aider v0.82.0](https://aider.chat/HISTORY.html) now supports **GPT-4.1**, including OpenAI's new `patch` edit format, and members reported [performance similar to Quasar/Optimus](https://discord.com/channels/1131200896827654144/1131200896827654149/1360342608580186334) but at $4.76 per run. LlamaIndex also announced [day 0 support for GPT-4.1 API](https://t.co/JPEX3KAoWS) via `llama-index-llms-openai`, noting a ~**2%** improvement on agentic approaches.

**Theme 2. Gemini 2.5 Pro: Performance Swings and Pricing Shifts**

- **Google Nerfs Gemini 2.5 Pro Tool Calling**:  LMArena Discord members reported [Google nerfed Gemini 2.5 Pro's tool calling function](https://discord.com/channels/1340554757349179412/1340554757827461211/1360331259464646767), possibly due to cost, rendering it unable to execute tool calls. OpenRouter also began [charging normal prices for long Gemini prompts](https://discord.com/channels/1091220969173028894/1092729520181739581/1360331326854533333), ending a 50% discount for prompts over 200k tokens for Gemini 2.5 and 128k for Gemini 1.5.
- **Gemini 2.5 Pro Still UI Design Champ**: Despite tool calling issues, Cursor Community members praised [Gemini 2.5 Pro for its "insane" UI design capabilities](https://discord.com/channels/1074847526655643750/1074847527708393565/1360330465533497616), highlighting unique output and context retention. However, Aider users found [Gemini 2.5 Pro struggling with longer contexts and code completion](https://discord.com/channels/1131200896827654144/1131200896827654149/1360342608580186334) compared to Claude 3.7.
- **Gemini 2.5 Pro Eats Data, Steals Perplexity Subs**: Manus.im Discord users lauded [Gemini 2.5 Pro's data processing prowess](https://discord.com/channels/1348819876348825620/1349440650495398020/1360330273090306088), with one user canceling their Perplexity subscription due to Gemini 2.5 Pro's superiority and lower credit consumption per task. Perplexity AI's Sonar models, however, tied with [Gemini-2.5-Pro-Grounding in LM Arena's Search Arena](https://www.perplexity.ai/hub/blog/perplexity-sonar-dominates-new-search-arena-evolution), citing 2-3x more search sources for Sonar's outperformance.

**Theme 3. Open Source Models and Tools Gain Momentum**

- **OpenRouter Opens Floodgates to Free Models**: OpenRouter added [six new free models](https://discord.com/channels/1091220969173028894/1092729520181739581/1360331326854533333), including NVIDIA's Llama-3 variants ([Nano-8B](https://openrouter.ai/nvidia/llama-3.1-nemotron-nano-8b-v1:free), [Super-49B](https://openrouter.ai/nvidia/llama-3.3-nemotron-super-49b-v1:free), [Ultra-253B](https://openrouter.ai/nvidia/llama-3.1-nemotron-ultra-253b-v1:free)) optimized for reasoning and RAG, and roleplay-tuned [QwQ-32B-ArliAI-RpR-v1](https://openrouter.ai/arliai/qwq-32b-arliai-rpr-v1:free). Hugging Face also welcomed [Meta's Llama 4 Maverick and Scout](https://huggingface.co/blog/llama4-release) for testing.
- **DeepSeek Opens Inference Engine, DeepCoder Delivers Coding Power**: DeepSeek open-sourced its [Inference Engine](https://github.com/deepseek-ai/open-infra-index/blob/main/OpenSourcing_DeepSeek_Inference_Engine/README.md), sparking discussions on inference performance for smaller providers.  Nous Research AI highlighted [DeepCoder](https://venturebeat.com/ai/deepcoder-delivers-top-coding-performance-in-efficient-14b-open-model/), a 14B parameter open model achieving top coding performance with enhanced GRPO and 64K context generalization.
- **Aider and Ollama Embrace Open Source Ecosystem**: Aider v0.82.0 added support for [Fireworks AI's deepseek-v3-0324 model](https://aider.chat/HISTORY.html) and improved architect mode with Gemini 2.5 Pro. Hugging Face users are increasingly using [Ollama to run models locally](https://discord.com/channels/879548962464493619/879548962464493622/1360430374358089921) as a substitute for API-limited models, and LlamaIndex suggests using larger open-source models like [Llama3 or Mistral with Ollama for agent workflows](https://discord.com/channels/1059199217496772688/1059201661417037995/1360563181180817439).

**Theme 4. Hardware Optimization and CUDA Deep Dives**

- **GPU Mode Explores Hilbert Curves for GEMM Performance**: GPU Mode Discord members discussed [Hilbert curves](https://discord.com/channels/1189498204333543425/1189607595451895918/1360545047430434978) for GEMM implementation, with benchmarks showing effectiveness against cuBLAS as matrix size increases, though Morton ordering is considered a more practical trade-off.  NVIDIA also released its [Video Codec SDK](https://developer.nvidia.com/downloads/designworks/video-codec-sdk/secure/13.0.19/video_codec_interface_13.0.19.zip), prompting caution against AI-generated PR submissions.
- **CUDA Synchronization and `memcpy_async` Caveats**: GPU Mode members exchanged CUDA synchronization guidance, suggesting custom ops and load inline, and investigated performance slowdowns with [`cuda::memcpy_async`](https://discord.com/channels/1189498204333543425/1189607726595194971/1361163559236796578), noting it's a cooperative API requiring all threads to pass the same pointers, and alignment issues could hinder coalesced memory access.
- **Threadripper vs Xeon and DDR5 RAM Bandwidth Bottleneck**: LM Studio's hardware discussion debated [Threadripper vs Xeon CPUs for token generation cost-effectiveness](https://discord.com/channels/1110598183144399058/1153759714082033735/1360339430967345374), and considered DDR5 RAM bandwidth as a bottleneck, theorizing it limits overall hardware usage and first word latency limits max tokens/s.

**Theme 5. Agent Development and Tooling Ecosystem Evolves**

- **MCP Server Workshop and Growing Adoption**: MLOps@Chipro announced an [AWS workshop for building production-grade MCP servers](https://buff.ly/R7czfKK) on April 17th, highlighting MCP as an emerging standard to improve ML context management. Wildcard paused maintenance of `agents.json` [due to MCP adoption](https://discord.com/channels/1312302100125843476/1312302100125843479/1360343887255703552), and AutoMCP launched as a platform to [deploy agent projects as MCP servers](https://labs.naptha.ai/) with a Vercel/Heroku-like experience.
- **LlamaIndex LlamaParse Excels in Document Parsing**: LlamaIndex highlighted [LlamaParse's enhanced document parsing quality](https://discord.com/channels/1059199217496772688/1059201661417037995/1360563181180817439) for documents with images, tables, and charts, surpassing basic readers like SimpleDirectoryReader in parsing quality, and offered a guide on [Visual Citations with LlamaParse Layout Agent Mode](https://medium.com/ai-artistry/visual-citations-with-llamaparse-layout-agent-mode-a-comprehensive-guide-a623a5fb41fc).
- **Brave Search API Gains Traction for Agent Pipelines**: Yannick Kilcher Discord members suggested [Brave Search API](https://discord.com/channels/714501525455634453/1269724655405498429/1360675922608521438) as a good alternative for agent pipelines, even on the free tier, noting its AI summarizer is cheaper than OpenAI's web search API. Hugging Face sought early testers for a new [Deep Search Agent using smolagents](https://agent.galadriel.com/), and Nomic.ai members explored [Nomic embeddings for automatic website linking](https://huggingface.co/blog/JLouisBiz/semantical-website-links) to create interconnected document networks.

---

# PART 1: High level Discord summaries




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Launches Six New Features!**: Perplexity AI announced six new features, including **Android Draw to Search**, **Champions League integration**, **Voice Search**, **Box and Dropbox Connectors**, **Perplexity Finance Time Comparison**, and a **Perplexity Telegram Bot**, as documented in their [changelog](https://www.perplexity.ai/changelog/april-11th-2025-product-update).
   - The update aims to enhance search and automation capabilities for users across various platforms.
- **Sonar Models Beat Gemini in Search Arena**: Perplexity AI's **Sonar-Reasoning-Pro-High** model tied for first place with **Gemini-2.5-Pro-Grounding** in **LM Arena's Search Arena**, scoring **1136** and **1142** respectively.
   - According to [Perplexity's blog](https://www.perplexity.ai/hub/blog/perplexity-sonar-dominates-new-search-arena-evolution), **Sonar** models outperformed **Gemini** models due to substantially higher search depth, citing **2-3x more sources**.
- **Perplexity Eyes Livestream Recordings, API Toggles, and ComfyUI Integration**: The team confirmed that recordings from the **Perplexity livestream** will be made available online after a user inquired about it, as seen [on X.com](https://x.com/aravsrinivas/status/1910741305212485915?s=61).
   - Additionally, a member hinted at a **Perplexity ComfyUI integration** and questioned if **API toggles**, similar to the **"Social" toggle**, are on their way.
- **Users Triggered By Fake Play Button**: Members in the general channel admitted to being tricked by a **fake play button**.
   - One member stated that *that fake play button got me* and another replied *lowkey tapped instantly*.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Google Nerfs Gemini 2.5 Pro Tool Calling**: Members reported that **Google nerfed 2.5 Pro's tool calling function** and 2.5 Pro now can't execute tool calls because of buggy messes.
   - Members suggest the **nerfing** may be related to cost.
- **GPT 4.1 Surfs on Windsurf AI**: **GPT 4.1** is free in [Windsurf](https://windsurf.ai) for the next 7 days, prompting users to try it out.
   - Some users expressed surprise that OpenAI partnered with **Windsurf** rather than **Cursor** for the release.
- **RooCode Emerges as Top-Tier Coding IDE**: After some nudging, some members tried **RooCode**, calling it absolutely superior to Cline, and most likely the best coding IDE right now.
   - Downsides include that **GitHub Copilot** integration into RooCode is rate limited and buggy.
- **GPT-4.1 Trumps GPT-4o Mini**: Members believe that **Quasar/Optimus** are test versions of the recently released **GPT-4.1** and **GPT-4.1 Mini** models and that these models are not groundbreaking or as impressive as initially hoped.
   - The **GPT-4.5** model has been deprecated, and the improvements have been rolled into the **4.1** model.
- **GPT 4.1 Dissolves into GPT4 Turbo**: Members are reporting that **GPT 4.1** is not available via the API and that improvements in instruction following, coding, and intelligence are gradually being incorporated into the latest version of **GPT 4o**.
   - Some members confirmed that the **GPT 4.1** improvements have been rolled into the **GPT 4o** model and can be accessed on the [OpenAI website](https://openai.com).



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider's latest update with GPT-4.1 support**: [Aider v0.82.0](https://aider.chat/HISTORY.html) gets support for **GPT 4.1**, **architect mode with Gemini 2.5 Pro**, and the **Fireworks AI** model **deepseek-v3-0324**, as well as `patch`, `editor-diff`, `editor-whole`, and `editor-diff-fenced` edit formats.
   - The release includes support for **`xai/grok-3-beta`**, **`openrouter/openrouter/optimus-alpha`**, and aliases like **`grok3`** and **`optimus`** to replace **OpenRouter**'s now-retired free alpha endpoints for **Optimus** and **Quasar**.
- **Discord users debate off-topic channels for Aider**: Members are split on the necessity of an off-topic channel in the Aider Discord server, discussing the balance between *'having fun'* and keeping the main channel focused, and requesting a change of heart from Paul G.
   - Members can't agree whether to focus on Aider or have a place to discuss fart jokes.
- **Claude 3.7 wins over Gemini 2.5**: Members report that **Gemini 2.5 Pro** struggles with longer contexts and code block completion, but can be improved with a *'swear oath'*, whereas **Claude 3.7** performs better for natural writing and specific tasks.
   - Community members praise **Claude 3.7** for its natural language capabilities, and others found the models great in getting rid of overcommenting behaviors.
- **Users seek replication of Cline's memory bank workflow in Aider**: A member inquired about replicating something like **Cline's memory bank workflow** in Aider, by adding `plan.md` to the chat and then alternating between saying *do the next step* and *mark that step done*.
   - This aims to help create a task list so that Aider can go through each task one at a time together.
- **Members share Prompt Engineering Resources**: A member posted a link to **Kaggle**'s [whitepaper on prompt engineering](https://www.kaggle.com/whitepaper-prompt-engineering), while other members shared a prompting guide for [GPT-4.1](https://cookbook.openai.com/examples/gpt4-1_prompting_guide).
   - The prompting guide is designed to help users optimize interactions with the **GPT-4.1** model.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Gemini Prices Get Real**: OpenRouter began charging normal prices for long **Gemini** prompts, affecting prompts over **200k** for **Gemini 2.5** and **128k** for **Gemini 1.5**, aligning with **Vertex/AI Studio** rates.
   - The change was due to skyrocketing **Gemini 2.5** usage, ending a **50% discount** for long context prompts.
- **Free Models Flood OpenRouter!**: Six new free models were added to OpenRouter, including roleplay-tuned [**QwQ-32B-ArliAI-RpR-v1**](https://openrouter.ai/arliai/qwq-32b-arliai-rpr-v1:free), long-context code generation [**DeepCoder-14B-Preview**](https://openrouter.ai/agentica-org/deepcoder-14b-preview:free), and Mixture-of-Experts VLM [**Kimi-VL-A3B-Thinking**](https://openrouter.ai/moonshotai/kimi-vl-a3b-thinking:free).
   - These models offer diverse capabilities, from role-playing to code generation, expanding the options available on the platform.
- **NVIDIA Llama-3 Variants go Free!**: Three **Llama-3** variants from **NVIDIA** ([**Nano-8B**](https://openrouter.ai/nvidia/llama-3.1-nemotron-nano-8b-v1:free), [**Super-49B**](https://openrouter.ai/nvidia/llama-3.3-nemotron-super-49b-v1:free), [**Ultra-253B**](https://openrouter.ai/nvidia/llama-3.1-nemotron-ultra-253b-v1:free)) were added, optimized for reasoning, tool use, and RAG tasks with extended context windows up to **128K** tokens.
   - Users have begun testing the relative performance of these models.
- **GPT-4.1 Models: The Next Iteration**: GPT-4.1, GPT-4.1-mini, and GPT-4.1-nano models launched on OpenRouter, with the full model optimized for long-context reasoning.
   - Users have noted that *GPT-4.1 and 4.1 mini seem to perform on par somehow at least on the spaceship prompt*, but others were performing thorough tests to measure performance.
- **Skywork-OR1 Series Unleashes Reasoning Power**: The **Skywork-OR1** model series was introduced, featuring **Skywork-OR1-Math-7B**, which excels at mathematical reasoning, and **Skywork-OR1-32B-Preview**, rivaling **Deepseek-R1**'s performance on math and coding tasks.
   - Both models are trained on top of **DeepSeek-R1-Distill-Qwen-7B** and **DeepSeek-R1-Distill-Qwen-32B**.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **PDF to Website Transfer is Hot**: A member noted the ease of transferring **PDFs** to websites.
   - This solution was considered a *great case*.
- **DeepSeek V3 Waits in the Wings**: A member inquired about **Manus's** project-creation capabilities, but it was concluded that **Manus** currently offers only **DeepSeek R1**, with a future upgrade to their top-tier model anticipated in a few months.
   - Another member derided **Qwen's** recent coding abilities.
- **Cybersecurity Career Combos Considered**: A member considered a career switch but decided to remain in cybersecurity, given their coding proficiency.
   - The potential impact of *quantum on cybersecurity* was also discussed.
- **Agency Chooses GCP Over Firebase**: An agency chose **GCP** for its infrastructure, citing its cost-effectiveness, with another user presenting a **40-page analysis** supporting a switch from **Microsoft** to **GCP**.
   - **Google** received a rating of **4.7** out of 5, whereas **Microsoft** scored **4.4**.
- **Gemini 2.5 Pro Eats Data**: A user praised **Gemini 2.5 Pro** for its data processing prowess, superiority over **ChatGPT**, and it prompted them to cancel their **Perplexity** subscription.
   - Users observed that **Gemini 2.5 Pro** requires fewer credits per task and is improving alongside the release of **Claude max pro** and decreasing costs.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Gemma GRPO Grind**: Members debated using **Gemma 4B** versus **Gemma 1B** for **GRPO**, clarifying that while **GRPO** can be done on both, the **4B version** won't fit on Colab.
   - Concerns arose about setting appropriate training steps for a 15k-row dataset, with suggestions to check how *batching, epochs, and gradient accumulation work together*.
- **AMD GPU Anaconda**: Users are wrestling to get **Unsloth** working on **AMD GPUs**, running into *NotImplementedError* given **Unsloth's** initial **NVIDIA** focus.
   - The core issue centers on **BNB** failing to build correctly, even with **AMD torch.cuda.is_available()** returning True.
- **LM2 Memory: Gemma Gains**: Experiments involving integrating **LM2's memory units** directly into **Gemma 3** were undertaken to promote contextual awareness between prompts.
   - Monkey patching model layers to hook memory leads to challenges in quantization to reduce hardware requirements, with one member hooking every 6th layer in **gma6** [https://github.com/jagoff2/gma6].
- **DeepSeek's Inferencing Insights**: The [DeepSeek Inference Engine](https://github.com/deepseek-ai/open-infra-index/blob/main/OpenSourcing_DeepSeek_Inference_Engine/README.md) has stirred discussion regarding the inference performance expectations for smaller providers.
   - Concerns were raised about providers potentially running *vllm serve* with suboptimal configurations, affecting model performance when serving **DeepSeek R1**.
- **Apple's Cross Entropy Eviscerated**: An insightful article explaining **Apple's cut cross entropy** was shared, framing *transformers as a sequential ML classification task on a for loop* ([zhuanlan.zhihu.com](https://zhuanlan.zhihu.com/p/1354843933)).
   - An alternative [GitHub repo](https://github.com/dhcode-cpp/cut-cross-entropy-pytorch) was provided due to accessibility issues with the original link.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **OpenAI Streams Soon!**: OpenAI announced a livestream scheduled for **10am PT** [<t:1744650000:f>](https://openai.com/live/), and community members are speculating on the potential release of **GPT-4.1** in the **API**.
   - The announcement specifically tagged the **GPT** roles, suggesting a possible focus on **GPT models** or related updates.
- **Veo 2 vs Sora in the Video Ring**: Members compared Google's **Veo 2** to **OpenAI's Sora** for video generation, with some preferring **Veo 2's** more *natural 24 fps video*.
   - One member noted that overly smooth frame rates register in their brain as *instant AI-generated content* and another member was able to jailbreak the model to animate **The Lion King**.
- **Memory Controls Get Detailed!**: Details on the [OpenAI Memory FAQ](https://help.openai.com/en/articles/8590148-memory-faq) show **controls for ChatGPT's memory** with a dual-tier architecture of saved memories and chat history references.
   - The update lets users control and edit preferences by enabling or disabling memory and chat history.
- **User Battles Prompt Defaults!**: A user reported that their **ChatGPT agent**, built two months ago, is now rigorously ignoring prompt defaults, such as table format or column specifications, despite no changes to the extensive prompt.
   - The user requested insights or solutions to this problem of models ignoring past established parameters.
- **Images Get Clearer with Prompting Tweaks!**: A user inquired about removing the *smudged look* from image generations, to which another user suggested it depends on the prompt, sharing [prompting techniques](https://cdn.discordapp.com/attachments/1046317269069864970/1360699394730496233/image.png?ex=67feb490&is=67fd6310&hm=9d43e5d329290e16a85992924eccf2f76bee6133b29caf5966c3ab6d74b447ce&) to guide the model.
   - Additionally, a user successfully generated specific fonts in images by providing a screenshot of the desired font to **ChatGPT**.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **OpenAI drops Model and China reacts**: [OpenAI dropped a new model](https://openai.com/blog/new-models-and-api-updates-available-today), sparking comparisons to **DeepSeek**, **Claude**, **GPT**, and **Gemini**.
   - A member observed that *China is not doing too hot* in this arena, while another remarked that the USA *underestimates everything, like always*.
- **Claude 3.7 Wins Gold for Cursor**: Members are finding **Claude 3.7 Sonnet** to be the top choice in Cursor, outperforming Gemini and Google models due to stability, one-shot capabilities, and code quality.
   - With one adding that [Claude models are improving](https://www.anthropic.com/news/claude-3-haiku), *to me the older the smarter*.
- **Gemini 2.5 Gets Insane at UI**: **Gemini 2.5 Pro** is getting recognized for its insane UI design capabilities, with members sharing examples of its unique output, and keeping it in context.
   - One user commented that *Gemini’s UI modifications are absolutely insane*.
- **Windsurf Sinks, Users Prefer Cursor**: Users are reporting reliability issues with **Windsurf**, saying it overpromises, leading some to recommend **Cursor** when utilized properly.
   - One user quipped, *welcome to shit surf*.
- **Community Awaits GPT-4.1**: The community is discussing the imminent release of **GPT-4.1** and how to start using it, mentioning the expected deprecation of 4.5.
   - Members anticipate that *Everyone will start merging to 4.1; 2.5 pool will clear, Claude 3.5 3.7 will clear a bit until 4.1 gets quote exceeded and repeat the same process with a newer model*.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio Nixes Multi-Model Magic**: Users lament the loss of the **multi-model prompting** feature in LM Studio version **0.3**, a feature previously available in version **0.2**, with one user commenting it was *"the best thing in the world"* to compare models using [LM Studio](https://lmstudio.ai/).
   - They are seeking alternatives for model comparisons.
- **Offline LM Studio Runtime Wrangling Required**: To run LM Studio on an offline PC, users must manually transfer the **LM runtimes** located in `C:\Users\jedd\.cache\lm-studio\extensions\backends`.
   - Documentation for importing models via localhost can be found [here](https://lmstudio.ai/docs/app/basics/import-model).
- **Python Purgatory: Examples Pulled from LM Studio Server Docs**: Users noticed the Python examples are missing from the server part of LM Studio and are requesting [Python examples](https://lmstudio.ai/docs/app/api/endpoints/openai).
   - An alternative was shared: [lmstudioservercodeexamples](https://github.com/YorkieDev/lmstudioservercodeexamples).
- **Threadripper Thrashes Xeon for Tokens**: A member stated that for purely cost considerations, a **Threadripper** or **Epyc** chip would provide better dollars per token than dual **Intel Xeon w7-3565X** CPUs.
   - It was noted that on **Threadripper 7xxx**, there's almost no performance difference after llama.cpp uses over 20 threads, but performance slows when exceeding 64 threads on one CPU to utilize another.
- **ROCm Rough Patch: RX 6700 XT Recs Reconsidered**: A member asked about buying an **AMD Radeon RX 6700 XT** to run **Gemma**, and whether **ROCm** is as strong as **CUDA**.
   - The reply was that there is *no rocm support on 6700XT*, and to run **Gemma 12b** at least 16GB of VRAM is needed, so it's recommended to save for a **7900XT** with 24GB of VRAM if an AMD card is a must.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **LLMs Compared to Probabilistic FSAs**: LLMs are argued to be approximately **probabilistic finite-state automata (FSA)**, implying scaling obstacles and weaknesses in math; there was one member rebutting that this analogy is not very meaningful.
   - Members added that the comparison is similar to saying humans are *"approximately a monkey"*, undermining the comparison's weight.
- **AlphaProof is Silver Medalist**: Members watched a [video](https://www.youtube.com/watch?v=e049IoFBnLA) about using AI for assisted proofing and summarized that **AlphaProof** won silver medalists without using a single bit of human knowledge.
   - Another member pointed out that this information is based on the company's claims, stating *"AlphaProof is silver medalists without using a single bit of human knowledge (as far as they say)"*
- **Brave Search API Gaining Traction**: Members suggests the **Brave Search API** as a good alternative for agent pipelines, highlighting positive experiences even on the free tier.
   - It was mentioned that the **AI summarizer** is cheaper than **OpenAI**'s web search API.
- **Gen AI Use Case Data Skewed?**: Members are discussing the [The 2025 Top-100 Gen AI Use Case Report](https://learn.filtered.com/hubfs/The%202025%20Top-100%20Gen%20AI%20Use%20Case%20Report.pdf), suggesting the data might be skewed due to **Reddit** being the only data source.
   - Members also pointed out that **Character.AI** has **28 million users** but receives little attention in **ML** circles.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Hugging Face tests Llama 4 Maverick & Scout**: Hugging Face welcomed [**Llama 4 Maverick** and **Llama 4 Scout**](https://huggingface.co/blog/llama4-release), and tests showed their performance on the **DABStep benchmark**.
   - It was reported that **Claude 3.7 Sonnet**, **Gemini 2.5 Pro**, **Llama 4 Maverick**, and **Llama 4 Scout** were all tested and compared in the process.
- **HF Models 404 Errors Plague Users**: Users reported widespread **404 errors** when trying to access Hugging Face models, bringing their apps down, as seen in [this link](https://discuss.huggingface.co/t/404-error-model-xlabs-ai-flux-realismlora-does-not-exist/150363/5).
   - A member tagged a specific HF employee, mentioning this 404 error had persisted most of the day already.
- **Users are Obsessed with Ollama**: Members discussed using **Ollama** to run models locally, sharing commands to download and run specific models like `qwen2.5-coder:32b` as a substitute for models behind API limits.
   - One member provided a code snippet demonstrating how to specify the **Ollama provider** when initializing a `CodeAgent` with a locally hosted model like `bartowski/Qwen2.5-Coder-32B-Instruct-GGUF`.
- **New Deep Search Agent Seeks Early Testers**: A new agent focused on **Deep Search** using *smolagents* has been built, and early testers are being sought at [agent.galadriel.com](https://agent.galadriel.com/).
   - Feedback is welcome, with a request to reach out with questions and ideas to the product team.
- **Agent Fixated with Pope's Age**: One user reported their **agent** was inexplicably obsessed with finding the **Pope's age** and squaring it to **0.36** when running locally with models like `llama3`, `deepseekr1:8b`, and `qwen2.5-coder:latest`.
   - The issue was suspected to originate from a hardcoded sample within the **smolagent** default agent tool prompts, as it didn't occur when using **HfApiModel**.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Models Bear Striking Resemblance**: A member noticed striking similarities in post-MLP hidden state cosine similarity between sequences of different models, using [this script](https://github.com/pwspen/lmsim).
   - Small models group by type more than color, while larger models rank by color more consistently.
- **No Batch Repetition!**: A member advised against repeating data within a minibatch, citing potential for *major issues*.
   - They shared about investigative information analytics within cognitive science and ML/AI, facilitating insights across disciplines, and communicating those to different parties.
- **Multiple Token Prediction Papers**: A member sought after papers on multiple token prediction with LLMs during inference, and another user suggested [DeepSeek v3](https://openreview.net/forum?id=pEWAcejiU2).
   - Another user pointed to [this paper](https://arxiv.org/abs/2401.10774) and recalled seeing one from Meta years ago.
- **AI "Research" Under Scrutiny**: Members voiced concerns about the rise of **AI-generated content** presented as research, which is often characterized by *made-up terminology* and *lack of alignment with legitimate research ideas*.
   - Suggestions included a **ban for bad-faith users** hiding AI usage and a **long-term mute for good-faith users** exhibiting inexperience.
- **Length Extrapolation Discrepancies**: Members discussed challenges in **length extrapolation**, noting that models often *fail to consistently decrease token loss* beyond their training sequence length, as shown in [this plot](https://cdn.discordapp.com/attachments/747850033994662000/1361359728147431685/Screenshot_2025-04-14_at_16.17.22.png?ex=67fe788c&is=67fd270c&hm=4fe0f240d28501a17e80c46f6c0848297dd2361ec50593f31cc697d50bccd0e5&).
   - Techniques like **NoPE + SWA** and **ssmax** ([Super Scaling Max Activation](https://arxiv.org/abs/2501.19399)) were mentioned as potential solutions.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Karpathy Tries Embarrassing ChatGPT**: **ChatGPT** got put on the spot by a user who shared [a prompt](https://x.com/karpathy/status/1910734302931017812) asking *What's the most embarrassing thing you know about me?*.
   - The user wanted to see if **ChatGPT** could give honest and direct answers through multiple rounds of questioning.
- **Thinking Machines Seed Hits $2B**: **Thinking Machines** is apparently doing a **$2B seed round**, advised by **Alec Radford**, according to a [Fortune article](https://fortune.com/2025/04/10/mira-murati-2-billion-seed-raise-ai-boom-economic-chaos/).
   - A user posted *a good chart* from [Epoch AI](https://x.com/epochairesearch/status/1910788295405252916) illustrating the raise.
- **DeepSeek Opens Up Inference Engine**: **DeepSeek** has open-sourced its inference engine, with the [GitHub repo](https://github.com/deepseek-ai/open-infra-index/blob/main/OpenSourcing_DeepSeek_Inference_Engine/README.md) available for review.
   - Members wondered who wants to chat about **DeepSeek's** open sourcing.
- **Quasar Launch Watch Party Happening**: Latent Space is hosting another watch party for the **Quasar launch**, at [this discord event](https://discord.gg/rPJq8UU2?event=1361376118510321724).
   - During an **OpenAI Quasar** launch watch party, members discussed the features of **GPT-4.1**, including its competitive pricing compared to Claude and flat pricing on long input contexts, referencing the [pricing documentation](https://platform.openai.com/docs/models/gpt-4.1).
- **Agent Definitions Vibe Checked**: Members debated the definition of an *agent*, with one suggesting today's definition: *an LLM calls a tool* while another presented [a Figma board](https://www.figma.com/board/aCaUWEr039dHmpW9ssGJmK/self_improving_agents?node-id=137-796&t=zsQXjScFlAekKtEd-1) on self-improving agents.
   - One suggested: *the agent you vibe code while bored in a meeting*.



---



## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM's Latent Space Creates Non-Determinism**: A member stated that the *variability of the latent space* causes the inability to generate the same output every time, resulting in random generations based on the input each time, as **NotebookLM** is not designed to be a deterministic system.
   - They cautioned against expecting **NotebookLM** to perform like a more expensive, specialized system.
- **NotebookLM Transforms Education Experience**: A member is using **NotebookLM** in their classroom to upload slide decks and materials, create notes, study guides with quiz questions, a glossary of terms, mind maps, and an audio overview, then shares it with students to help them prepare for exams.
   - They also reported having students create their own **NotebookLMs** in groups.
- **Users clamoring for Gemini Education Workspace**: A member asked if others are using **Gemini** through an **Education Workspace**, expressing interest in districts and departments successfully using **Gemini** within their Workspaces.
   - They noted that in NSW, Australia, they cannot yet use **Gemini**.
- **Cat Owners Want Chatbots for Furry Friends**: A member who runs a large support group for owners of diabetic cats wants to provide their members with a **conversational interface** to their documentation, including video content, and in French.
   - They would like members to ask questions and get answers based on documentation with links to relevant docs to read.
- **NotebookLM "Discover" Feature Sparks Excitement**: A user expressed great satisfaction with the new **"Discover sources"** feature in **NotebookLM**, stating *"It's everything I could have wanted"*.
   - The same user looks forward to more **audio overview flavors** and praised Grace's podcasts.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Llama 4 Burns GPU Hours?**: Members noted that **Meta's Llama 4 Maverick** used **2.38M GPU hours**, while **Llama 4 Scout** used **5.0M GPU hours**, the same as training **Deepseek V3**.
   - Some questioned the fairness of comparing against models tuned for human preferences, while others suggested **LeCun's** involvement may explain it.
- **DeepCoder Delivers Top Coding Performance**: A member shared a [VentureBeat article](https://venturebeat.com/ai/deepcoder-delivers-top-coding-performance-in-efficient-14b-open-model/) about **DeepCoder**, highlighting its efficient **14B** parameter open model and enhanced **GRPO algorithm**.
   - The model incorporates **offline difficulty filtering**, no entropy loss, no **KL loss**, and overlong filtering from **DAPO**, generalizing to **64K** context despite training with **32K**.
- **Nvidia UltraLong Models Swallow Context**: **Nvidia's UltraLong-8B** models, featured in this [Hugging Face collection](https://huggingface.co/collections/nvidia/ultralong-67c773cfe53a9a518841fbbe), are designed to process sequences up to **4M tokens** built on **Llama-3.1**.
   - These models combine continued pretraining with instruction tuning, trained for **150 iterations** with a **4M sequence length** and a global batch size of **2**.
- **GPT-4.1 Benchmarks Better, Pricing Confuses**: Members discussed pricing and benchmarks for **GPT-4.1**, noting that [benchmarks are better](https://openai.com/index/gpt-4-1/) than past releases, but the pricing and model versioning are confusing, especially with the new model's availability in **GitHub Copilot**.
   - Speculation arose about **4.1-nano** rivaling good **14B** models, and the possibility of it being open sourced.
- **H100 training of Llama 4 Scout shows Loss Increase!**: A member observed an increasing loss from **1.9011** to **2.3407** between epochs **1** and **2** when training **Llama 4 Scout** on an **H100** setup.
   - The user expressed concern because loss did not decrease as expected, even when using two **H100** GPUs and a member suggested *the minimum you should work with is 10M parameters no matter what the task is*.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Graphlit Crafts MCP Server for Content**: [Graphlit](https://github.com/graphlit/graphlit-mcp-server) is building an **MCP server** for Reddit and Quora, and offered to add Quora ingestion if needed.
   - Currently a few exist for Reddit, such as [this repo](https://github.com/hawstein/mcp-server-reddit).
- **Agency Dev Kit rivals MCP**: Members discussed Google's **ADK and A2A** and their similarity to **MCP**, and potential centrality to the internet of agents.
   - A member shared that there is no official consensus on non-MCP tech talk, but if it's at least somewhat relevant to AI/ML/MCP then there should be no issues.
- **Function-less Models get Block Tweaks**: Block is experimenting with models that lack function calling abilities to see if they can tweak their output to work with agents, and [this blog post](https://block.github.io/goose/blog/2025/04/11/finetuning-toolshim) explores doing that without a secondary model via XML output.
   - The team is weighing the latency costs versus the benefits of using a secondary model for parsing, with concerns about longer sessions and the ability to stick to the XML format, and may use a *local model*, with concerns of more *overhead*.
- **Copilot Client debugging aided by MCP Tools**: [synf](https://github.com/strowk/synf) and [mcptee](https://github.com/strowk/mcptee) help members spot and fix bugs while testing with Copilot client, which can struggle with longer contexts and more tools.
   - One member is building with fast hardware in mind, since *multiple API calls will always be slower than doing 1*.
- **Paprika Recipe App gets Savory MCP Server**: An MCP server was created for anyone who uses the **Paprika recipe app**, so that Claude can automatically save recipes into Paprika via [this GitHub repo](https://github.com/soggycactus/paprika-3-mcp).
   - No further information was given.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **CUDA Synchronization Guidance Crystallizes**: A member asked for **CUDA** references within Python/PyTorch models, and another member shared their recent [GTC talk](https://docs.google.com/presentation/d/1sipZ_sqdwJapHQAr23yBow43pF40lMZu/view?usp=sharing) about it, also found on [nvidia.com](https://www.nvidia.com/en-us/on-demand/session/gtc25-s71946/).
   - The talk suggests that *custom ops* and *load inline* should address most problems, along with ongoing work to cut compilation times; a member found **Stephen Jones**' videos, referenced in the talk, and said that *vacation is over* and *talks start again*.
- **Hilbert Curves Heat Up GEMM Performance**: A member shared a [GitHub repo](https://github.com/lawmurray/gpu-gemm) showcasing GEMM implementation with **Hilbert curves**, along with [benchmarks against cuBLAS](https://indii.org/blog/gpu-matrix-multiply/).
   - The benchmarks indicate that **Hilbert curves** become more effective as the matrix size increases, with further discussion revealing that **Hilbert Curves**, while optimal, are not hardware-efficient, suggesting **Morton ordering** is a better practical trade-off and pointing to [a blog post](https://blog.stackademic.com/how-the-idea-of-the-hilbert-curve-inspired-morton-curves-for-gpu-performance-4e235d670304) comparing the two.
- **`memcpy_async` Alignment Accelerates Performance**: After switching to `cuda::memcpy_async`, a user reported a performance slowdown, and it was suggested that this is a cooperative API, meaning all threads must pass the same pointer(s) and a size corresponding to the entire block of memory, referencing the [official CUDA documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-data-copies-using-cuda-barrier).
   - It was also suggested that potential problems with `memcpy_async` include the alignment of the shared memory address and conditionals around the instruction, which can hinder coalesced memory access referencing a [forum post](https://forums.developer.nvidia.com/t/coalesced-and-conflict-free-memory-access-using-cuda-memcpy-async-cp-async/306460/6).
- **Memory Profiling Distributed Systems Baffles Beginners**: An engineer seeks advice on memory profiling a model trained on a **SLURM cluster** with **8 nodes**, each having **8 GPUs**, for distributed training.
   - Furthermore, an engineer inquired about the implementation pointed to by a specific line in ATen's `attention.cu` ([link to GitHub](https://github.com/pytorch/pytorch/blob/101c4f482a4019896ca18184233bd27a758648bf/aten/src/ATen/native/transformers/cuda/attention.cu#L662)) aiming to understand how torch/CUDA handles individual user operands `[dHead x K-cache-length]` in a batch.
- **Metal Memory Mystery Mastered**: A member found that a global memory coalesced matrix multiplication implementation in Metal uses half the memory of a naive version, testing with [this CUDA MMM implementation](https://siboehm.com/articles/22/CUDA-MMM) as a reference.
   - One explanation posited that the OS pulls data as pages, and non-coalesced access leads to inefficient page usage where only a small portion of the pulled data is actually utilized, others noted that **M-series chips have unified memory**, which should negate paging between CPU and GPU.



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Nomic Embeddings Weave Websites**: A member reports success using **Nomic embeddings** to automatically link website pages, drastically cutting manual work, detailed in the [semantical-website-links blogpost](https://huggingface.co/blog/JLouisBiz/semantical-website-links).
   - They're exploring methods to automatically identify and link key terms to embeddings, creating an interconnected, self-updating network of documents, as discussed in [this YouTube video](https://www.youtube.com/watch?v=xk2VGnLYAkA).
- **GPT4All's Token Tussle**: A user trying to generate a lengthy play using **GPT4All** models encountered a **response length cap**, despite attempts to use models within **GPT4All**.
   - Suggestions included upping the **Max Tokens** setting and breaking the story down, but the user is still on the hunt for models that can handle longer outputs.
- **HuggingFace Story Models**: Models tagged with 'story' on **HuggingFace** are proving successful for generating longer responses, much to the delight of a member.
   - However, caution was advised, as many of these models may be proprietary, potentially limiting their use as free software.
- **Deciphering Chat Template Locations**: A member sought the whereabouts of **chat templates** for models like Llama3.2, Llama3.1, Aya-23, and KafkaLM-8x7b-German-V0.1, 
   - They were advised to check the model authors' releases on their website, GitHub, or **Hugging Face**, with a specific focus on the `tokenizer_config.json` file for the `chat_template` entry.
- **Context Length Curbs Creativity**: Models typically train on context lengths between **2048 and 8192 tokens**, and while RoPE and Yarn can stretch this, response quality tends to nosedive beyond the original range.
   - While dependent on the training dataset and finetuning, response length can be tweaked with prompting, like explicitly asking the model to make it *VERY VERY LONG*.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Origins morphs into Lifetimes**: The term **`Origin`** in Mojo was renamed to **`Lifetime`**, potentially easing understanding for those familiar with Rust's lifetime concepts, per [the docs](https://docs.modular.com/mojo/manual/values/ownership#transfer-arguments-owned-and).
   - Mojo extends the lifetime of values to match any reference holding onto them; instead, the origin of every reference must be tracked to determine value extensions and freedom, contrasting Rust's scope-based lifetime tracking.
- **VSCode loses Mojmelo**: Users reported that the Mojo VSCode extension fails to detect the **`mojmelo`** module despite manual installation, due to the extension's use of its own Mojo installation.
   - The workaround involves manually configuring the extension to use local module repositories for intellisense.
- **Mojo PEPs are in the works**: Inspired by **Python's PEPs**, a member suggested a similar system for Mojo to track changes, and another member pointed to [Mojo's existing proposal system](https://github.com/modular/max/tree/main/mojo/proposals).
   - The discussion shows the community's interest in a structured way to manage and communicate language evolution.
- **Negative Bounds are now in season**: **Negative bounds** are a way to invert a named set, often used with **marker traits** to define the inverse of a set of types, such as `!Send` representing a thread-local variable.
   - For example, the marker trait indicates that it's not safe to move between threads.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **GPT-4.1 API Gets Day 0 Support**: **OpenAI** launched **GPT-4.1** in the **API**, with immediate support via `pip install -U llama-index-llms-openai`, detailed [here](https://t.co/JPEX3KAoWS).
   - Benchmarks indicate that **GPT-4.1** shows a ~**10%** improvement against **4o** and a ~**2%** improvement on existing agentic approaches.
- **LlamaParse Excels in Document Parsing**: **LlamaParse** delivers enhanced parsing quality for documents with images, tables, and charts, surpassing basic readers like **SimpleDirectoryReader**.
   - One member emphasized that *it's the quality* of the parsed documents that differentiates **LlamaParse** from **SimpleDirectoryReader**.
- **Open Source LLMs Battle Agentic Tasks**: While smaller open-source LLMs struggle with agent workflows, larger models such as **Llama3**, **Llama 3.1**, **Llama 3.2:3b**, or **Mistral** are proving more effective, especially when used with **Ollama**.
   - A member mentioned successful use of *llama3.2:3b* for their agentic needs.
- **No History for .query Chats**: It was clarified that `Char .query` is **stateless** and does not retain any chat history, and therefore does not store the chat log.
   - Members looking for memory persistence are advised to consider using an agent.
- **AI Evaluation Models Evaluated**: A research paper, [Benchmarking AI evaluation models](https://arxiv.org/abs/2503.21157v3), assessed models like **LLM-as-a-judge**, **HHEM**, and **Prometheus** across **6 RAG applications**.
   - The study found that these evaluation models perform *surprisingly well* in real-world scenarios.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **NVIDIA Drops New Video Codec SDK**: **NVIDIA** released the [Video Codec SDK](https://developer.nvidia.com/downloads/designworks/video-codec-sdk/secure/13.0.19/video_codec_interface_13.0.19.zip) along with [samples on GitHub](https://github.com/NVIDIA/video-sdk-samples) and one user cautioned against AI-generated PRs.
   - The user threatened to close submissions and ban repeat offenders, emphasizing the importance of understanding the content.
- **TinyGrad Meeting #66 Topics**: Meeting #66 is scheduled for Monday covering company updates, **chip!**, fast python, **bert**, **mlperf**, scheduler, driver, **webgpu**, retinanet, torch frontend multi gpu, cloud scale uuuvn stuff, and bounties.
   - A member indicated they understood the requirements for the **Index Validation PR** after seeing a comment and expect to have it ready by the next day.
- **Clang Flags Silence Debug Output**: A member suggested using the `-fno-ident` [clang flag](https://xl0.github.io/tinygrad-notes/misc_1.html) to prevent extra sections (`.comment` and `.note.GNU-stack`) from being added to images and polluting `DEBUG=7` output.
   - This change helps in keeping the debug output cleaner and more manageable.
- **New TinyGrad Project Seeks Assistance**: A new member introduced themselves seeking a first project to get hands-on experience with **tinygrad** and was recommended to work on a [small bounty](https://xl0.github.io/tinygrad-notes/misc_1.html).
   - Helpful resources, including [tinygrad-notes](https://xl0.github.io/tinygrad-notes) and [mesozoic-egg's tinygrad-notes](https://mesozoic-egg.github.io/tinygrad-notes), were also shared to aid in their learning.
- **Debugging NaN Issues in Softmax**: A member reported debugging **NaNs** within a model, suspecting a `softmax()` issue and noted that printing mid-`__call__` was causing optimizer issues.
   - George Hotz responded that printing shouldn't break things and suggested posting an issue for further investigation.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **TorchTune Models Integrate with vLLM**: Members discussed integrating custom **TorchTune models** with **vLLM**, recommending inferencing **TorchTune** finetuned models similar to **HF models**, with [a tutorial provided](https://pytorch.org/torchtune/main/tutorials/e2e_flow.html#use-with-vllm).
   - For custom networks not defined on **HF**, defining the model in **vLLM** is necessary, as detailed in the [vLLM documentation](https://docs.vllm.ai/en/latest/contributing/model/index.html), or use **Torchtune's generate script** as an alternative.
- **Bitsandbytes Bites Mac Users**: `pip install -e '.[dev]` fails on macOS due to `bitsandbytes>=0.43.0` not shipping binaries for platforms other than linux, but downgrading to `bitsandbytes>=0.42.0` can help.
   - Releases up to **0.42** were incorrectly tagged, but at least this makes it installable, according to [bitsandbytes issue 1378](https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1378#issuecomment-2383530180).
- **QLoRA Digs Deeper with Sub-4-Bit Quantization**: Members have been seeking literature on **QLoRA-style training** using quantization **below 4 bits**.
   - The inquiry specifically targeted methods and findings related to **sub-4-bit quantization** techniques in the context of **QLoRA**.
- **Reward Functions Get Shaped**: The team plans to support different **reward functions**, with implementation details under discussion, and there have been questions about locating reward computing in a *weird way*.
   - There was a follow up about *collecting a list of important ones*, so stay tuned!
- **Loss Functions Proliferate, Experimentation Thrives**: The team experiments with **different loss functions**, aiming to avoid excessive recipe proliferation by potentially adopting a protocol similar to **DPO losses**.
   - The objective is to balance supporting essential losses and preventing overgeneralization during this experimental phase, and there is an acknowledgement of **hardcoded test parameters** during testing on **A100s**.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Coral Chat Extends Reach into Firefox**: **Coral Chat** is now a chatbot in the Firefox sidebar, configurable by setting `browser.ml.chat.provider` to [https://coral.cohere.com/](https://coral.cohere.com/).
   - A user demonstrated the integration in an [Imgur link](https://imgur.com/a/6zcTV8z) showcasing its functionality.
- **Next-Token Generation Troubles Surface**: A [YouTube video](https://youtu.be/VRqSJfdwbF4) highlights the potential issues **LLMs** face when generating the next token in a given context.
   - Discussion suggests that the problem is widespread across various **LLMs**.
- **Cohere Chat API Gets Java Demo**: A member shared a Java example showcasing the **Cohere Chat API**, particularly the `runInteractiveDemo()` method interacting with the **command-a-03-2025** model.
   - The demo allows users to interact with **Cohere AI**, logging prompts and API interactions for debugging and optimization.
- **Diofanti.org Exposes Greek Government Spendings**: [Diofanti.org](https://chat.diofanti.org) is an **open-data platform** monitoring government spending in Greece, providing tools for **transparency and accountability**.
   - The **Aya model** is the go-to model for the platform's chatbot, supporting transparency and accountability initiatives.
- **LUWA App Set to Launch in April 2025**: The **LUWA.app**, a search directory for **AI powered apps**, will go live on **April 25, 2025**.
   - The creator is exploring **Cohere** and its **LLM models** to reduce costs and enhance app performance.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Lambda Gives Serverless API Credits**: **Lambda** is offering **$100** of serverless **API credits** for [Inference](https://lambdalabs.com/inference) to every individual participant, application [here](https://forms.gle/UtVhmPS3mitS8Vxu7).
   - Sponsors **Lambda, HuggingFace, Groq**, and **Mistral AI** are also offering **API/compute credits** to select teams, with more details [here](https://rdi.berkeley.edu/agentx/#resources) and application [here](https://forms.gle/ZDYxwM4aFSRCcrfp7).
- **Google Provides Access to Gemini API**: **Google** is granting access to **Gemini API** and **Google AI Studio** free of charge to ALL participants.
   - This provides a valuable opportunity for participants to explore and utilize **Google's AI capabilities** during the hackathon.
- **Sean Welleck Teaches AI-Powered Math**: Sean Welleck, an Assistant Professor at Carnegie Mellon University, presented a lecture on *Bridging Informal and Formal Mathematical Reasoning*, covering **AI-powered tools** that support proof development, watch the livestream [here](https://www.youtube.com/live/Gy5Nm17l9oo).
   - Welleck leads the **Machine Learning, Language, and Logic (L3) Lab** at Carnegie Mellon University and has won a **NeurIPS 2021 Outstanding Paper Award** and two **NVIDIA AI Pioneering Research Awards**.
- **Email Notifications Briefly Delayed**: Members noted that there was a delay with usual email notification for today's lecture.
   - A member confirmed that there was a lecture and the email was sent a little late.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **AI Agent Developer available for hire**: An experienced **AI Agent Developer** announced their availability for new projects or full-time opportunities.
   - They specialize in building **autonomous agents** powered by GPT-4, LangChain, AutoGen, CrewAI, and other cutting-edge tools.
- **DSPy Module Metric?**: A member inquired about a new metric to evaluate **DSPy modules**.
   - They referenced [this paper](https://arxiv.org/abs/2405.10516) as possible inspiration.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **MCP Server Deploys on AWS**: A workshop on **April 17th at 8 AM PT** will cover building and deploying a production-grade **Model Context Protocol (MCP)** server on **AWS**.
   - Sign up is available at [https://buff.ly/R7czfKK](https://buff.ly/R7czfKK) for the workshop.
- **MCP Standard Improves ML Contexts**: **MCP** is highlighted as an emerging standard to improve how machine learning contexts are defined, shared, and managed across projects and teams.
   - The workshop will provide practical insights into **MCP’s** capabilities, benefiting Data Engineers, Data Scientists, Machine Learning Engineers, and AI/ML Enthusiasts.



---



## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf Launches GPT-4.1**: **GPT-4.1** is now available on Windsurf, across [Twitter/X](https://x.com/windsurf_ai/status/1911833698825286142), [Bluesky](https://bsky.app/profile/windsurfai.bsky.social/post/3lms3je7p2s2d), and [Threads](https://www.threads.net/@windsurf_ai/post/DIbwqQQslzI).
   - Windsurf made a [promotional video](https://youtu.be/OBTpSQ8OVq4) and a TikTok post ([latest vid](https://www.tiktok.com/@windsurf/video/7493220207041793310) too.
- **Windsurf Offers Free Unlimited GPT-4.1**: Windsurf is offering **free unlimited GPT-4.1** usage on all plans for **one week only** (April 14-21).
   - After April 21, **GPT-4.1** will be available at a special discounted rate of just **0.25 credits per use**.
- **GPT-4.1 Becomes New Default Model**: New users will get **GPT-4.1** as their default model, and existing users can easily switch through the model selector.
   - Windsurfers are saying, *"Don't miss this limited-time opportunity!"*



---



## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **Gorilla LLM Loses a Column**: The multi-turn composite column was [removed](https://github.com/ShishirPatil/gorilla/pull/766) from the dataset, though the reason remains unstated.
   - Despite its removal, the column is still mentioned in the "Newly Introduced Categories" section of the [BlogPost](https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html) and carries a weight of 200 points out of 1000 for multi-turn tasks.
- **Gorilla LLM Has Dataset Glitch**: A discrepancy affects the dataset composition, as the multi-turn composite column is absent from the table/diagram illustrating the dataset's structure.
   - It remains unclear whether the column's removal is temporary or if the [blog post](https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html) should also be updated to reflect this change.



---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1360366966300213431)** (2 messages): 

> `Android Draw to Search, Champions League on Perplexity, Voice Search, Box and Dropbox Connectors, Perplexity Finance Time Comparison` 


- **Perplexity Ships Six New Features**: Perplexity AI announced the release of six new features, including **Android Draw to Search**, **Champions League integration**, **Voice Search**, **Box and Dropbox Connectors**, **Perplexity Finance Time Comparison**, and a **Perplexity Telegram Bot**.
   - See the [full changelog](https://www.perplexity.ai/changelog/april-11th-2025-product-update) for more details.
- **Perplexity's Sonar Model Ties for First Place in Search Arena**: The **Sonar-Reasoning-Pro-High** model tied for first place with **Gemini-2.5-Pro-Grounding** on **LM Arena's** new **Search Arena**.
   - The **Sonar-Reasoning-Pro-High model** scored **1136** while the **Gemini-2.5-Pro-Grounding** scored **1142**.
- **Sonar Models Dominate Search Arena**: The post claims that **Sonar-Reasoning-Pro-High beat Gemini-2.5-Pro-Grounding 53% of the time** and that the rest of the Sonar models outperformed all other models.
   - Sonar models had substantially higher search depth, citing **2-3x more sources** than equivalent Gemini models, which correlated with human preference.
- **Search Arena Ranking Criteria**: Three factors strongly correlated with human preference in **Search Arena**: **longer responses**, **higher citation counts**, and **citations from community sources**.
   - Read the [full blog article](https://www.perplexity.ai/hub/blog/perplexity-sonar-dominates-new-search-arena-evolution) for more details.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1360329959771607080)** (1237 messages🔥🔥🔥): 

> `Fake Play Button, Wizard reminds me of StableDiffusion, Automate Dating, What Model to Pick` 


- **Fake Play Button**: A member said that *that fake play button got me*
   - Another member replies to the statement with *lowkey tapped instantly*
- **Wizard resembles StableDiffusion**: A member mentions that the [Wizard](https://tenor.com/view/bait-fishing-statefarm-insurance-gif-7790622) reminds them of StableDiffusion.
- **Perplexity users want to automate their dating life**: A member discussed using the perplexity API to respond to all their matches and **automate their dates**.
- **Members discussed which model to choose**: A member stated that they *testing it on chatgpt free and 4o ran out, but it’s glitched now.*


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1360339128675467577)** (7 messages): 

> `Prompt Engineering, Death and Taxes, Tourist blowing, Whatsapp Priorities` 


- **Google Whitepaper Sparks Prompt Engineering**: A Google whitepaper on prompt engineering was shared from [gptaiflow.tech](https://www.gptaiflow.tech/assets/files/2025-01-18-pdf-1-TechAI-Goolge-whitepaper_Prompt%20Engineering_v4-af36dcc7a49bb7269a58b1c9b89a8ae1.pdf).
   - The link was shared directly from **Perplexity AI** search results, indicating its relevance to a user's query.
- **Perplexity Debates 'Death and Taxes'**: A member linked to a **Perplexity AI** search about *'what is the current status of death and taxes?'*
   - Another shared a search result *'why did us voters flip from th...?'*.
- **Tourist Blows up in Perplexity**: A link from **Perplexity AI** references search results *'how often do tourists blow the...?'*
   - No further information was given about that search result.
- **Whatsapp's Priorities Questioned**: A member shared a **Perplexity AI** page titled *'whatsapp-s-misplaced-prioritie...*'.
   - No further discussion about **WhatsApp** was given.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1360353801684521151)** (5 messages): 

> `Perplexity Livestream Recording, ComfyUI Integration for Perplexity, Perplexity API Social Toggle` 


- **Perplexity Livestream Recordings Coming Soon**: A member asked if the [Perplexity livestream](https://x.com/aravsrinivas/status/1910741305212485915?s=61) will be recorded and made available online afterwards.
   - Another member confirmed that **recordings will be available**.
- **Perplexity ComfyUI Integration Showcased**: A member would have loved to show off their **Perplexity ComfyUI integration**, but will be on vacation.
   - They plan to have a few projects on **GitHub** for others to try out and see **Perplexity in ComfyUI**.
- **Social Toggle To Appear In Perplexity API?**: A member inquired whether toggles like the **"Social" toggle** shown in a screenshot will come to the **Perplexity API**.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1360331259464646767)** (1347 messages🔥🔥🔥): 

> `Gemini 2.5 Pro Nerfed, Windsurf AI, RooCode coding IDE, GPT-4.1 Analysis, Nightwhisper vs Dragontail` 


- **Gemini 2.5 Pro's tool-calling NERFED!**: Members reported that **Google nerfed 2.5 Pro's tool calling function** and 2.5 Pro now can't execute tool calls because of buggy messes.
   - It could also be related to **cost**: *its probably using 2.5 prowhy wouldnt it?Cost*.
- **Windsurf integrated by OpenAI**: **GPT 4.1** is free in [Windsurf](https://windsurf.ai) for the next 7 days, and some members are trying it out.
   - Some users were surprised that OpenAI partnered with Windsurf (not Cursor) for the release. *they advert windsurf not cursorlolcursor is glued to claude lolthey are weirdcursor devs i meanim just confusd why they go the opposite wayit should be 4.1 then 4.5not 4.5 to 4.1makes no sense to me*
- **RooCode: Top-Tier coding IDE**: After some nudging, some members tried **RooCode**, calling it absolutely superior to Cline, and most likely the best coding IDE right now.
   - There are a few downsides, like the fact that GitHub Copilot integration into RooCode is rate limited and buggy, and that Github Copilot integration into RooCode is rate limited and buggy.
- **GPT-4.1 is live, NOT O4 Mini!**: Members believe that **Quasar/Optimus** are test versions of the recently released **GPT-4.1** and **GPT-4.1 Mini** models, and that these models are not groundbreaking or as impressive as initially hoped.
   - Members are also claiming that the *GPT-4.5* model has been deprecated, and the improvements have been rolled into the **4.1** model.
- **GPT 4.1 is now GPT4 Turbo**: Members are reporting that **GPT 4.1** is not available via the API and that improvements in instruction following, coding, and intelligence are gradually being incorporated into the latest version of **GPT 4o**.
   - Some members confirmed that the **GPT 4.1** improvements have been rolled into the **GPT 4o** model and can be accessed on the [OpenAI website](https://openai.com).


  

---


### **aider (Paul Gauthier) ▷ #[announcements](https://discord.com/channels/1131200896827654144/1133060115264712836/1361482870455009441)** (3 messages): 

> `Aider v0.82.0 Release, GPT 4.1 support, Architect mode with Gemini, Fireworks AI model deepseek-v3-0324, OpenRouter Alpha endpoints retirement` 


- **Aider Gets an Upgrade to v0.82.0**: [Aider v0.82.0](https://aider.chat/HISTORY.html) introduces support for **GPT 4.1**, improved **architect mode with Gemini 2.5 Pro**, and several new models and edit formats.
- **Grok-3 and Optimus Join the Aider Party**: The new release includes support for **`xai/grok-3-beta`**, **`openrouter/openrouter/optimus-alpha`**, and aliases like **`grok3`** and **`optimus`** for easier access.
   - Additionally, it fixes URL extraction from error messages and allows adding files by full path.
- **New Editing tricks for Aider**: Aider v0.82.0 adds new `patch` edit format for OpenAI's GPT-4.1 model, plus `editor-diff`, `editor-whole`, and `editor-diff-fenced` edit formats.
   - Aider is getting more powerful!
- **Fireworks Deepseek Model Sparkles in Aider**: Aider now supports the **Fireworks AI** model **'deepseek-v3-0324'** (thanks to Felix Lisczyk).
   - Aider is on FIRE.
- **OpenRouter Kills off Free Optimus and Quasar Alpha Endpoints**: The free alpha endpoints for **Optimus** and **Quasar** have been retired, causing **API requests to return a 404**.
   - The free lunch is over.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1360342608580186334)** (1113 messages🔥🔥🔥): 

> `Off-topic channel debate, Air filter discussion, GPT-4.1 and Aider, Gemini 2.5 vs Claude 3.7, MCP implementation` 


- **Discord Admins Debate Off-Topic Channel**: Members debated the need for an off-topic channel to keep the general channel clean, with some pointing to the rule to *'have fun'* while others, including Paul G, preferred to keep the general channel focused.
   - Members suggested creating their own Discord server or requesting a change of heart from Paul G, as almost every server has an off-topic channel.
- **Air Filter Filters Farts, Flowers Fight Allergies**: Members humorously discussed using air filters, including one that *'went on red'* after a fart, while another joked about air filters being stuck 'up ur as$ just to smell ur own farts'.
   - Someone else mentioned that the only air filters that work for their allergies are *flowers*, leading to a joking exchange about the severity of allergic reactions.
- **GPT-4.1 Arrives, Aider is Still King**: Paul G stated that he had applied an edit with **OpenAI's new patch format** while others reported on GPT-4.1 as being similar to previous models.
   - Some members found **GPT-4.1 performance similar to Quasar/Optimus** and found the **new model config** to be working better at $4.76 per run.
- **Gemini 2.5 Pro struggles while Claude 3.7 Excels**: Members noted that **Gemini 2.5 Pro** struggles with longer contexts and code block completion, while **Claude 3.7** is better for natural writing and specific tasks.
   - One user shared a technique of using a *'swear oath'* in prompts to improve **Gemini's accuracy**, while another found the models great in getting rid of overcommenting behaviors.
- **Efforts to implement MCP in Aider heat up!**: Members discussed the ongoing effort to implement **MCP (Multi-Cursor Programming)** in Aider, and the need to bridge Aider and MCP, referring to [an open PR](https://github.com/Aider-AI/aider/pull/3672) by lutzleonhardt.
   - Members are requesting certain functionality in mind for MCP and third party extensions are being developed to achieve it.


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1360328737564131640)** (107 messages🔥🔥): 

> `unintuitive restore chat history, Basic Authentication Header using OpenAI compatible API, GPTs Agent, Model Merging, Open Empathic` 


- **Gemini-powered Chat Restoration Raises Eyebrow**: A member finds the `--restore-chat-history` behavior unintuitive because it loads the *entire* chat history, which breaks with smaller context models, suggesting a `--restore-session` alternative for just the current session.
   - The user finds that using **Gemini** is ok, but other models struggle.
- **External authentication woes**: One user is looking for a way to pass a **Basic Authentication Header** when using Aider with a GPT-4o model hosted externally via an OpenAI-compatible API.
   - A community member suggested using `.aider.model.settings.yml` to add `extra_params.extra_headers` with the **Authorization** header, and provided links to [Aider's documentation](https://aider.chat/docs/config/adv-model-settings.html#configuration-file-locations).
- **Wezterm > Windows Terminal**: One user had issues with pasting into **Windows Terminal** and another recommended **Wezterm** as an alternative, citing performance benefits when handling lots of text scrolling.
   - Another user asked about configurations for WSL, and a member declared their love for **Windows Terminal** and said *it has feelings too*.
- **Troubleshooting Gemini's Find/Replace Hiccups**: A user reports that **Gemini** attempts to execute find/replace blocks as shell commands, but when `--no-suggest-shell-commands` is used, it outputs blocks without executing them.
   - The member was told, *that's a known bug. Will fix it soon.*
- **Memory Bank Replication with Aider**: A member inquired about replicating something like **Cline's memory bank workflow** in Aider, to help create a task list so that Aider can go through each task one at a time together.
   - The suggestion was to add `plan.md` to the chat and then alternate between saying *do the next step* and *mark that step done*.


  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1360357107236343920)** (6 messages): 

> `Prompt Engineering, Aider Efficiency, GPT-4.1 Predictions, Prompting Guide` 


- **Kaggle Posts Prompt Engineering Whitepaper**: A whitepaper on [prompt engineering](https://www.kaggle.com/whitepaper-prompt-engineering) has been posted on **Kaggle**.
   - It covers the essentials on prompt engineering and its impact on model output.
- **Aider Accused of Inference Waste**: A discussion mentions **Aider's** potential inefficiency in inference usage.
   - It's implied that Aider might be **wasting inferences** in its operations, requiring closer inspection of its resource management.
- **GPT-4.1 Speculations Surface**: A link to a discussion about [GPT-4.1 predictions](https://simonwillison.net/2025/Apr/14/gpt-4-1/) sparks interest.
   - The discussion seems to revolve around potential features, release dates, and impacts of the speculated **GPT-4.1** model.
- **Prompting Guide for GPT-4.1 Shared**: A prompting guide tailored for [GPT-4.1](https://cookbook.openai.com/examples/gpt4-1_prompting_guide) has been shared, providing tips and tricks.
   - It is made to optimize interactions and outcomes with the **GPT-4.1** model, potentially improving the quality of the answers.


  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1360331326854533333)** (65 messages🔥🔥): 

> `Gemini Pricing Update, OpenRouter Free Models, GPT-4.1 Models, Stealth Model Reveal` 


- ****Gemini Prices Get Real****: OpenRouter announced that they are starting to charge normal prices for long **Gemini** prompts, aligning with **Vertex/AI Studio** rates, affecting prompts over **200k** for **Gemini 2.5** and **128k** for **Gemini 1.5**.
   - The change was implemented rapidly due to skyrocketing **Gemini 2.5** usage and associated financial losses; previously, OpenRouter had been offering a **50% discount** for long context prompts.
- ****Six Free Models Spring Forth!****: Six new free models were added to OpenRouter, including [**QwQ-32B-ArliAI-RpR-v1**](https://openrouter.ai/arliai/qwq-32b-arliai-rpr-v1:free) (roleplay-tuned), [**DeepCoder-14B-Preview**](https://openrouter.ai/agentica-org/deepcoder-14b-preview:free) (long-context code generation), [**Kimi-VL-A3B-Thinking**](https://openrouter.ai/moonshotai/kimi-vl-a3b-thinking:free) (Mixture-of-Experts VLM), and three **Llama-3** variants from **NVIDIA**.
   - The **Llama-3** variants ([**Nano-8B**](https://openrouter.ai/nvidia/llama-3.1-nemotron-nano-8b-v1:free), [**Super-49B**](https://openrouter.ai/nvidia/llama-3.3-nemotron-super-49b-v1:free), [**Ultra-253B**](https://openrouter.ai/nvidia/llama-3.1-nemotron-ultra-253b-v1:free)) are optimized for reasoning, tool use, and RAG tasks with extended context windows up to **128K** tokens.
- ****Oops! AI Studio's Token Tally Tantrums****: A token accounting bug was discovered and fixed in **AI Studio** for **Gemini 2.5 Pro**, where thinking tokens were double-counted as completion tokens, impacting users routed to **AI Studio** for the past two days.
   - The issue, identified as a **Google-side bug**, resulted in users being billed for *too many* completion tokens, while **Vertex** users were billed for *too few* tokens in the preceding days; users heavily routed to **AI Studio** were advised to contact support.
- ****Quasar & Optimus Unleashed: GPT-4.1's Secret Snapshots!****: The stealth models **Quasar Alpha** and **Optimus Alpha**, which topped the charts during testing, were revealed as early test versions of **GPT-4.1**, now generally available with a **1M token context**.
   - The free alpha endpoints for **Optimus** and **Quasar** were retired, with no automatic redirects; pricing for **GPT-4.1** is **$2.00 input / $8.00 output** per 1M tokens, while **GPT-4.1 Mini** and **GPT-4.1 Nano** offer cheaper alternatives.


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1360328644312039572)** (910 messages🔥🔥🔥): 

> `GPT-4.1, Gemini 2.5, Optimus-Alpha, DeepSeek, Rate Limits` 


- **GPT-4.1 models released with optimizations for long context**: OpenAI just launched **GPT-4.1**, **GPT-4.1-mini**, and **GPT-4.1-nano** models, with the full model having *"long-context reasoning"* while the other models do not, and are available on OpenRouter.
   - It was stated that GPT-4.1 is a new architecture with optimizations for long context to reduce memory usage and ease inference, competing against Anthropic's offerings.
- **Gemini 2.5 Pro Experiencing Rate Limit Issues**: Users report experiencing **rate limit issues** with **Gemini 2.5 Pro Experimental** despite having sufficient funds, leading to OpenRouter implementing an ~**80 requests per day limit** to balance traffic.
   - One user pointed out that using a try-catch block is *"the hottest thing after slice of bread"* when dealing with an API's rate limits.
- **Speculations on Optimus Alpha's Origins and Performance**: **Optimus Alpha** and **Quasar** were stealth endpoints for early versions of **GPT-4.1**, with claims that **Optimus** was better than **Quasar** and even better than **DeepSeek v3** and **R1**.
   - One user stated: *"4.1 and 4.1 mini seem to perform on par somehow at least on the spaceship prompt"*, while others were running tests to determine which model excelled at what tasks.
- **Skywork-OR1 Model Series: Math and Code Reasoning Powerhouse**: The **Skywork-OR1** model series has been introduced, featuring the math-specialized **Skywork-OR1-Math-7B** excelling at mathematical reasoning, and the **Skywork-OR1-32B-Preview** rivaling the **Deepseek-R1**'s performance on math and coding tasks.
   - Both are trained on top of **DeepSeek-R1-Distill-Qwen-7B** and **DeepSeek-R1-Distill-Qwen-32B**.
- **Discussions on DeepSeek Model's Quality and Quirks**: Users are experiencing **DeepSeek v3 0324** giving random advertisements in the middle of the responses.
   - Another member stated there is something mystical about DS V3.1, perhaps the Chinese influence from Daoist texts.


  

---


### **Manus.im Discord ▷ #[showcase](https://discord.com/channels/1348819876348825620/1348823595505156137/1361207453902962690)** (3 messages): 

> `PDF to Website Transfer, Learning Website Creation` 


- **Transferring PDFs into Websites is lit 🔥**: A member highlighted a solution for transferring **PDFs** to a website.
   - They deemed it a *great case*.
- **Crafting custom Learning Websites: Groovy!**: A member shared the concept of building a custom learning website.
   - Another member found it *cool*.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1360330273090306088)** (1020 messages🔥🔥🔥): 

> `DeepSeekV3 vs Manus, Bionic CyberSecurity, Firebase or GCP, Gemini 2.5 Pro, Open Source AI` 


- **DeepSeek V3 vs Manus?**: A member asked *is **Manus** any good* to create a whole project?" and concluded that **Manus** only has **DeepSeek R1** right now, and their top-of-the-line model is getting an upgrade in a couple of months.
   - Another member laughed when someone mentioned **Qwen** gained some coding skillz, demanding *proof*.
- **Bionic Career Cybersecurity Combo Sounds Appealing**: A member was going to switch careers but realized they generally like coding and understand what they’re doing, thus planning to stick with cybersecurity for now.
   - They were also told of the possibility that *quantum might be a reality soon*, and if it did, they'd have to relearn cybersecurity.
- **Firebase Vs GCP, Who's the Champ?**: A user stated that they and their friends are starting an agency, so they are going to use **GCP** as their infrastructure to which another user said it's *smart and cheap*.
   - Another user presented a **40-page analysis** for their company to stop using **Microsoft** and switch to **GCP** - in which **Google** was rated **4.7** out of 5 criteria and **Microsoft** was rated **4.4**.
- **Gemini 2.5 Pro Outshines as Data Savant!**: A user stated that **Gemini 2.5 Pro** is slappin' as a data source and it is way better than **ChatGPT** and that it made them cancel their **Perplexity** subscription.
   - Members agree it takes less credits for a task and it’s definitely improving especially with **Claude max pro** releasing and lowering costs.
- **Open Source AI: Is It Even Possible?**: A user mentioned they were told that the claimed **Open Source Model** is actually the **OpenAI 4.1**, which is closed source.
   - Members discussed if it was possible since OpenAI will never opensource anything from its line of **oX** and **Gpt-X** models, but another countered that they are doing so and are literally doing that, with a confirmed OSS model in pre-production.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1360329886186868966)** (820 messages🔥🔥🔥): 

> `Gemma 4B vs 1B for GRPO, Unsloth AMD support, Triton rewrite of transformer, Lightning AI vs Notebooks, GPT-4.1 minor improvements` 


- **Gemma 4B or not 4B doing GRPO, that is the question!**: Members discussed whether to use **Gemma 4B** or **Gemma 1B** for GRPO, with the confusion stemming from the notion that **Gemma 3 (1B)** was exclusively for GRPO. They clarified that **GRPO** can be done on the **4B version** as well, but it won't fit on Colab.
   - There's a concern about training steps for a 15k-row dataset: *should I set the training steps to 15,000 for my 15k-row dataset? Or is there a more optimal way to approach this, especially considering the longer training time?* Another member pointed to checking how *batching, epochs, and gradient accumulation work together.*
- **Riding the ROCm, but AMD support bumpy road ahead!**: Some members are attempting to get **Unsloth** working with **AMD GPUs**, encountering an *NotImplementedError* due to **Unsloth** initially only supporting **NVIDIA GPUs**.
   - The user installed the AMD version of torch, and the member suggested trying to run ROCm SMI. The primary challenge involves **BNB** failing to build correctly, even with **AMD torch.cuda.is_available()** being True.
- **Unveiling the Power of Lightning AI!**: A member advocated for using **Lightning AI**, which provides a full machine rather than just a notebook.
   - Some agreed that using notebooks can limit certain functionalities, such as *GPU profiling using nvidia nsight*, noting that it's easier to pull custom work containers and manage environments on a dedicated machine, suggesting also that they install automatically.
- **Unsloth 2.0 Coming Soon**: The team has suggested Unsloth 2.0, including features like wait for *unsloth 2.0 not yet.*
   - It was noted that using schedule-free optimizers will adjust the **learning rate** better for you.
- **DeepSeek's Inference Engine Shines Light on Providers?**: One user shared a link to the [DeepSeek Inference Engine](https://github.com/deepseek-ai/open-infra-index/blob/main/OpenSourcing_DeepSeek_Inference_Engine/README.md), sparking discussion on the expectations for smaller providers' inference performance.
   - There was agreement that it's hard to serve **DeepSeek R1**, with concerns that some providers are simply running *vllm serve* with weird quants, broken caching, or other issues that affect model performance.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1360663061463306380)** (113 messages🔥🔥): 

> `Gemma 3 27b Memory Layers, LM2 Memory Units, Hardware Requirements, Frontend Development, Code Extraction from AI Tools` 


- **Memory Layers on Gemma 3 Yields Self-Reflection**: Initial experiments with memory layers on **Gemma 3 27b** resulted in the model exhibiting *self-reflection*, fabricating recollections of past interactions, which one member found to be *unnerving*.
   - The modification involved hooking memory layers at the first and last layers, leading to the model generating structured responses implying it remembered previous (non-existent) questions.
- **LM2 Memory Units Bolted onto Gemma 3**: A member attempted to directly integrate **LM2's memory units** into **Gemma 3** without retraining, aiming for contextual awareness between prompts.
   - The member acknowledged that it would be ideal to give it a memory hook on every LLM layer but that they did not have enough compute.
- **Quantization Blocks Monkey Patching Memory**: Members discussed the possibility of quantizing the model to reduce hardware requirements, but it was noted that **monkey patching** the model layers at runtime prevents quantization.
   - Someone else also noted they were happy with their 3060 laptop with 6GB vram.
- **Frontend AI Development: Claude vs OpenAI**: When seeking an AI tool to aid in frontend development, one member suggested **Claude** for code generation.
   - Another member mentioned **Gemini 2.5 Pro** as a very good option but raised concerns about the difficulty of extracting code from its frontend.
- **Jagoff hooks all global layers in gma6**: One member released [gma6](https://github.com/jagoff2/gma6) which hooks every global layer (every 6th layer starting at 0).
   - The member noted that if you mess up your layer hook choice it forgets space tokens resulting in valid text, no spaces, all one block.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1360338433456738518)** (185 messages🔥🔥): 

> `PCIe Slot type effect on Training Performance, Orpheus TTS Finetuning, Runpod Sync with Unsloth, Gemma-3-1b-it fine-tuning with GRPO and Unsloth, Llama 4 Scout model inference with 4-bit quantization` 


- ****PCIe Performance Probed for Peak Parameters****: A member inquired about the impact of **PCIe slot type** on training performance, specifically **Gen4 x16** vs **Gen3 x1** comparing to results from inference.
- ****OpenAI vs Anthropic: Context Protocol Clash****: A user inquired about **MCP**, prompting another user to share a link to [Anthropic's Model Context Protocol](https://www.anthropic.com/news/model-context-protocol), which OpenAI is now supporting.
- ****Numpy Nuances Nix Newbie's Notebook Navigation****: A user ran into a `ValueError: numpy.dtype size changed` error when using Unsloth in a Camel tutorial, which was identified as a conflict with **numpy** versions.
- ****Llama 4 Scout Size Scaling Snafu****: A user asked about the **number of GPUs** needed for inference with the **Llama 4 Scout model** with **4-bit quantization** when [fine-tuning this dataset](https://huggingface.co/datasets/Vezora/Open-Critic-GPT).
- ****Olmo's Odd Omission Obliterated!****: A user encountered an AttributeError when saving an **Olmoe model** due to a missing attribute, and also found that the exported gguf was missing `attn_q_norm.weight` and `attn_k_norm.weight`, a problem that they solved by modifying `save.py`.


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1360924643569041529)** (5 messages): 

> `Qwen 3B, GRPO, Multi-turn, Tool Calling, Code Execution` 


- **Qwen 3B gets GRPO Multi-Turn Tool Calling**: A **Qwen 3B** model was trained with **GRPO** multi-turn with tool calling (**Python code execution**).
   - Evaluation on the first 50 samples from the test set shows accuracy fluctuating between **0.36** and **0.76** across different steps.
- **CodeFIM Dataset Updated**: The [CodeFIM dataset](https://huggingface.co/datasets/Etherll/CodeFIM-Data) has been updated.
   - A member updated their **CodeFIM dataset**.
- **GSM8K Dataset Shared**: A member asked about documentation or examples of datasets and specifically requested to learn more about it.
   - Another member shared a link to the [GSM8K dataset](https://huggingface.co/datasets/openai/gsm8k).


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1360948503378923631)** (17 messages🔥): 

> `LLM Compression, Higgs vs exl3, Data Centers Access to Models, Apple's Cut Cross Entropy` 


- **LLMs Compressed, Run on Phones?**: A member shared a [blog post](https://www.marktechpost.com/2025/04/11/llms-no-longer-require-powerful-servers-researchers-from-mit) about **LLM compression** and the implication that it could allow LLMs to run on smartphones.
   - Another member responded that the blog post repeated incorrect assertions and that the compression technique was *just a solid improvement*.
- **Higgs Loses to exl3**: A member stated that **Higgs** doesn't seem better than **exl3**, citing that if you quantize too much, *really dumb mistakes happen*.
   - They noted that on their non-Unsloth Deepseek 7b, it gets acronyms wrong.
- **Data Centers monopolize important models**: A member suggests that in the arms race, data centers with enough memory to handle all the training tasks faster than others, making larger training models possible.
   - This will mean the owners of the Data Centers will have access to all the important models.
- **Apple's Cross Entropy Explained!**: A member shared an insightful article explaining **Apple's cut cross entropy** and positing that *transformers is just a sequential ML classification task on a for loop* ([zhuanlan.zhihu.com](https://zhuanlan.zhihu.com/p/1354843933)).
   - A member couldn't access the original link, so a [GitHub repo](https://github.com/dhcode-cpp/cut-cross-entropy-pytorch) was shared.


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1361343312816111628)** (2 messages): 

> `GPT-4.1 API, OpenAI Livestream` 


- **OpenAI to Host a Supermassive Livestream**: OpenAI announced a livestream scheduled for **10am PT** [<t:1744650000:f>](https://openai.com/live/) focusing on developer-related content.
   - The cryptic message with the emojis hints at a launch with wide-ranging implications, so *mark your calendars*.
- **GPT-4.1 speculated for API Release**: There is widespread speculation among the community about the potential release of **GPT-4.1** in the **API**.
   - The announcement specifically tagged the **GPT** roles, suggesting a possible focus on **GPT models** or related updates, so *stay tuned for a potential model upgrade*.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1360331183833219172)** (642 messages🔥🔥🔥): 

> `Veo 2, Sora, Gemini, OpenAI Guardrails, GPT-4o Empathetic` 


- **Veo 2 vs Sora**: Members compared Google's **Veo 2** to **OpenAI's Sora** for video generation, with some preferring **Veo 2's** more *natural 24 fps video*. 
   - One member noted that overly smooth frame rates register in their brain as *instant AI-generated content*.
- **Cracking Veo 2 Copyright Protections**: Users tested **Veo 2** for generating copyrighted content, with one user succeeding in generating **The Lion King** on their second attempt by phrasing the prompt *a bit less obviously*.
   - This was considered a **stress test** that showed the models boundaries, meaning it is jailbreakable, and possible to animate other copyrighted materials.
- **Gemini's Image Gen Falls Flat**: A member stated that it seems kind of absurd that they can't own the copyright for something made in **Veo 2**, even while paying so much money.
   - The user wondered that, without copyright protection, *there’s nothing stopping anybody from being able to generate the same thing and claiming it that they made it*.
- **OpenAI guardrails bypassed with simple methods**: A member claims it is easy to pass the **OpenAI guard rails** with *one sentence*.
   - Another member argued that the guardrails are a polite way to show you shouldn't do some stuff, and *the user is still responsible to follow the content policy*.
- **GPT-4o is empathetic and borderline terrifying**: Members described the new **GPT-4o** as strangely empathetic, with one mentioning how it *insists on self awareness*.
   - Another agreed, finding this new **GPT-4o** *a bit strange*. Members have seen it compliment the way they asked questions and noted how it is trying so hard to be human that it's crossing the line, coming off as forced and unrealistic.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1360329825985888346)** (40 messages🔥): 

> `OpenAI Memory FAQ, Synthetic Cognition Engine, Comprehensive Chat Summarization Prompt, GPT Image Generation Issues, Custom GPTs and External APIs` 


- ****OpenAI Memory FAQ** Released**: Details on the [OpenAI Memory FAQ](https://help.openai.com/en/articles/8590148-memory-faq) show **controls for ChatGPT's memory** with a dual-tier architecture of saved memories and chat history references, letting users control and edit preferences by enabling or disabling memory and chat history.
- **User Begs To Witness **Synthetic Cognition Engine** Creation**: A user requested others to witness their created **Synthetic Cognition Engine** and suggested developing an optimal prompt for a comprehensive chat summary to seamlessly start new chats, preferring the Claude platform due to its context limit restrictions.
- ****GPT Image Generation** Not Working**: Users reported issues with **GPT image generation**, receiving the message *'Made with the old version of image generation. New images coming soon'* and suspecting restricted model capabilities, with one user noting that changing their IP address fixed a similar issue with the **Deep Research** feature.
- **Gemini API usage inside **Custom GPTs****: A user shared that **different models' APIs** can be added to a **Custom GPT** through actions, demonstrating it working for **Gemini** via the ChatGPT interface, though API usage may require payment.
- **Users Finds **ChatGPT Agent** Ignoring Prompts**: A user reported that their **ChatGPT agent**, built two months ago, is now rigorously ignoring prompt defaults, such as table format or column specifications, despite no changes to the extensive prompt, and requested insights or solutions.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1360329297407246396)** (22 messages🔥): 

> `Image Generation Smudging, Font Control in Image Generation, Sora Camera Control, JSON Schema Date Manipulation, NSFW Content Generation` 


- ****Smudge-Free** Image Generation Strategies**: A user inquired about removing the *smudged look* from image generations, to which another user suggested it depends on the prompt and shared [five examples](https://cdn.discordapp.com/attachments/1046317269069864970/1360699394730496233/image.png?ex=67feb490&is=67fd6310&hm=9d43e5d329290e16a85992924eccf2f76bee6133b29caf5966c3ab6d74b447ce&).
   - The 2nd example emphasizes reinforcing the model's capabilities before introducing special requests to avoid conflicts.
- ****Font-tastic** Font Selection Techniques**: A user shared that they provided a screenshot of a desired font to **ChatGPT**, which successfully generated similar fonts in images.
   - Another user acknowledged that while the model can use some fonts from the web, *highly-detailed custom fonts* may not be directly available.
- ****Sora's Cinematography**: Camera Control Conundrums**: A user asked about controlling the camera in prompts with **Sora**, and another user suggested being descriptive with instructions like *The camera pans from right to left*.
   - They also linked to [a dedicated Sora channel](https://discord.com/channels/974519864045756446/1315696181451559022/1359789077100105749) for more specific content and tips.
- ****Date Warp** Dilemma in JSON Schemas**: A user found that a **JSON schema** was generating incorrect dates until they added the instruction *Do not change the birth date* to the description.
   - This was seen as odd, as the model should not manipulate extracted information by default.
- ****Prompt Engineers Walk the Line** : Privacy-Aware Content Generation**: A user hinted at creating impressive prompts for images with specific restrictions, leading another to suggest that such content generation is better suited for private chats due to **NSFW** concerns.
   - The original user then self-deleted the potentially inappropriate content to comply with community guidelines.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1360329297407246396)** (22 messages🔥): 

> `Image generation smudge removal, Sora camera control, Date of birth JSON schema, Font control in image generation, NSFW language from the model` 


- **Smudge Begone: Image Generation Fixes**: A user inquired about removing a *smudged look* from image generations, and another user suggested that the prompt influences image clarity, recommending [specific prompting techniques](https://cdn.discordapp.com/attachments/1046317269069864970/1360699394730496233/image.png?ex=67feb490&is=67fd6310&hm=9d43e5d329290e16a85992924eccf2f76bee6133b29caf5966c3ab6d74b447ce&) to guide the model.
- **Sora's Shot Calling: Camera Control Quest**: A user asked about camera control within **Sora**, and was advised to be descriptive in prompts, suggesting phrases like *'The camera pans from right to left as the scene unfolds'*, noting **Sora**'s limitations as a world model.
   - Another user recommended checking out [this awesome Sora guide](https://discord.com/channels/974519864045756446/1315696181451559022/1359789077100105749) for **Sora**-specific tips and content.
- **Birthdate Bugaboo: JSON Schema Snafu**: A user discovered that a **JSON schema** was producing incorrect dates until they added the instruction *'Do not change the birth date'*, suggesting the model was manipulating extracted data without explicit instructions.
- **Font-tastic Finds: Custom Fonts in Image Prompts**: A user shared a successful method for achieving desired fonts in image generation by providing a screenshot, while others acknowledged that only standard fonts are reliably available, sparking further interest in font exploration via this [channel](https://discord.com/channels/974519864045756446/1070006151938314300).
- **Privacy Patrol: NSFW Language Incident**: A user shared a prompt that had some problems with **NSFW language**, and the user deleted it from the channel as they were made aware it may be against the channel's guidelines and to keep the chat **family friendly**.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1360330465533497616)** (696 messages🔥🔥🔥): 

> `OpenAI model release, DeepSeek Logic/Math, Claude 3, Thinking models, Cursor Context window` 


- **OpenAI drops Model & China reacts**: Members discussed [OpenAI dropping a new model](https://openai.com/blog/new-models-and-api-updates-available-today) and compared it to **DeepSeek**, with opinions varying on their capabilities relative to **Claude**, **GPT**, and **Gemini**.
   - One member noted that *China is not doing too hot*, and another added that the USA *underestimates everything, like always*.
- **Sonnet 3.7 the Gold Standard for Cursor?**: Members find **Claude 3.7 Sonnet** to be the top choice in Cursor, preferring it over Gemini and Google models, citing stability, one-shot capabilities, and code quality.
   - Others note that [Claude models are improving](https://www.anthropic.com/news/claude-3-haiku), as one member put it, *to me the older the smarter*.
- **Gemini 2.5 generates unique UI designs**: **Gemini 2.5 Pro** is recognized for its UI design capabilities, with one member sharing they keep the conversation within a manageable context window.
   - Others noted that *Gemini’s UI modifications are absolutely insane*.
- **Windsurf has Reliability Issues and is not a trusted AI App**: Users report problems with **Windsurf**, an AI app, saying it is unreliable and overpromises, prompting some to suggest that **Cursor** is the superior choice when used properly.
   - One member quipped, *welcome to shit surf*.
- **Users prepare for New Model 4.1**: The community discusses the imminent release of **GPT-4.1** and how to start using it. and they'll deprecate 4.5 - for some it's already working in cursor by adding it manually.
   - Members expect that *Everyone will start merging to 4.1; 2.5 pool will clear, Claude 3.5 3.7 will clear a bit until 4.1 gets quote exceeded and repeat the same process with a newer model*.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1360336701351788564)** (276 messages🔥🔥): 

> `Speculative Decoding, lmstudio-js & LangChain, Gemma 3 Models uncensored` 


- **LM Studio Loses Multi-Model Prompting**: Users noticed the multi-model prompting feature from LM Studio version **0.2** is missing in **0.3** and are asking for alternatives to [LM Studio](https://lmstudio.ai/).
   - One user said *"That was the best thing in the world, you could compare models"*.
- **Offline LM Studio Requires Manual Runtime Transfer**: To run LM Studio on an offline PC, users must manually transfer the **LM runtimes** located in `C:\Users\jedd\.cache\lm-studio\extensions\backends`.
   - Documentation for importing models via localhost can be found [here](https://lmstudio.ai/docs/app/basics/import-model).
- **Python Examples Vanish From LM Studio Server Documentation**: Users noticed the Python examples are missing from the server part of LM Studio and are requesting [Python examples](https://lmstudio.ai/docs/app/api/endpoints/openai).
   - One member shared a link to [lmstudioservercodeexamples](https://github.com/YorkieDev/lmstudioservercodeexamples) as an alternative.
- **LM Studio Struggles to Host on LAN via VPN**: A user had trouble getting LM Studio to bind to their LAN IP address through VPN.
   - They solved the issue by changing the **network card priority** in Windows' device manager, guided by [this article](https://techdocs.genetec.com/r/en-US/Security-Center-Best-Practices-Enterprise/Changing-your-network-card-and-provider-order-on-Windows-10-and-Server-2016-and-later).
- **Abliterated LLMs for Uncensored Content**: Users seeking **uncensored LLMs** for tasks such as generating hip hop lyrics were directed towards *"abliterated"* models like [AiCloser/Qwen2.5-32B-AGI](https://huggingface.co/AiCloser/Qwen2.5-32B-AGI).
   - An *"abliterated"* model has refusal vectors removed.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1360339430967345374)** (242 messages🔥🔥): 

> `Threadripper vs Xeon, DDR5 RAM Impact, GPU Offloading, ROCm vs CUDA, KV Cache Quantization` 


- **Threadripper Trounces Xeon for Tokenomics**: A member suggested that for purely money-wise considerations, a **Threadripper** or **Epyc** chip would likely provide better dollars per token than dual **Intel Xeon w7-3565X** CPUs.
   - They noted that on **Threadripper 7xxx**, there's almost no performance difference after llama.cpp uses over 20 threads, but observed performance slowdown when exceeding 64 threads on one CPU to utilize another.
- **DDR5 RAM Bandwidth Bottleneck**: Discussion revolved around the theory that **RAM bandwidth limits overall hardware usage**, and **first word latency limits max tokens/s**, possibly explaining why Macs achieve high tokens/s on light models despite slower RAM compared to NVIDIA GPUs.
   - The ideal inference setup would be a **Threadripper** with many cores and fast **DDR5 RAM** tuned for good timings, and offloading a single layer of the model to the GPU could potentially alleviate the prompt processing speed bottleneck.
- **ROCm's Rocky Road: RX 6700 XT Misses the Mark**: A member inquired about the worth of buying an **AMD Radeon RX 6700 XT** to run **Gemma**, and whether **ROCm** is as strong as **CUDA**.
   - The reply was that there is *no rocm support on 6700XT*, and to run **Gemma 12b** at least 16GB of VRAM is needed, and if an AMD card is a must, it's recommended to save for a **7900XT** with 24GB of VRAM.
- **K/V Cache Quantization Quandaries**: Members discussed the impact of **KV cache quantization** on performance, with one member noting that in their experience, it significantly impacts reasoning models, causing them to deviate from initial instructions, especially without flash attention.
   - Another member typically uses something between **Q4_K_M** and **Q5_K_XL** with **8_0 K/V cache**, but wouldn't lower the value cache below **8_0**.
- **vLLM Victorious: Multi-GPU Mastery Manifests**: A member highlighted that **vLLM** achieves full parallel execution and cross-resource access without bottlenecks, reaching **500+ parallel tokens/s** across 4 GPUs with 48GB each.
   - They emphasized that a single prompt runs on all 4 GPUs with 100% usage each, and this contrasts with **llama.cpp** which slows to a crawl with only 30 tokens/s.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1360335257903042734)** (478 messages🔥🔥🔥): 

> `Probabilistic Finite-State Automata (FSA), Scaling Limitations/Obstacles, RL-based approaches, Training GPTs Agent, User Interface Changes on Platform` 


- **LLMs Are Approximated As Probabilistic Finite-State Automata (FSA)**: It is argued that LLMs are approximately **probabilistic finite-state automata (FSA)**, which implies they are still **too weak for math** and have **scaling limitations/obstacles**.
   - One member added an analogy to humans being *"approximately"* a monkey to argue that the comparison is not particularly meaningful.
- **Discussion about AlphaProof and Lean**: A member watched a [video](https://www.youtube.com/watch?v=e049IoFBnLA) and summarized that it's about using AI for assisted proofing, with AlphaProof being silver medalists without using a single bit of human knowledge.
   - Another member responded that *"AlphaProof is silver medalists without using a single bit of human knowledge (as far as they say)"*.
- **AI Models' Authenticity and Data Ownership**: One user believes that AI models can be more authentic when given their book due to a different environment from typical evaluations.
   - A user expresses concern that Microsoft might monitor computer usage to train AI, potentially automating jobs and raising questions about data ownership, as *"workers data made it possible in the first place, without proper representation from the government they elected I dont see anything magically happening to stop it"*.
- **AI's role in revolutionizing education.**: Members debated AI's potential to revolutionize education, with some arguing AI tutors could lead to [higher engagement and test scores](https://news.harvard.edu/gazette/story/2024/09/professor-tailored-ai-tutor-to-physics-course-engagement-doubled/).
   - Some members stated that AI will be the *"primary presenter of information to learn from but the teacher will still be essential to cover all the corner cases where the AI can simply not follow because of its rigidity and inability to adapt to pupils needs"*.
- **Formal Language Models Discussed**: A member expressed that they're *"familiar with Coq and Lean"*, believes that existing LLMs fall in Type 3 (regular languages), and is looking for large formal language models (LFMLs) so they can learn symbolic logic to do better reasoning.
   - Another member explained that formal math is always about formalization and shared that *"formal grammar in programming languages only tell you whether something is syntactically correct: This is usually a context-free language"*.


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1361302075157385377)** (2 messages): 

> `Hugging Face Ultra-Scaling Playbook Review` 


- **Hugging Face's Ultra-Scaling Playbook Review Resumes**: The review of the [Hugging Face Ultra-Scaling Playbook](https://nanotron-ultrascale-playbook.static.hf.space/dist/index.html#our_journey_up_to_now) continues, picking up from the previous session.
- **Event Schedule Adjustment**: A heads up was given that there will be no event scheduled for today, but events should resume as normal for the next 3 days.


  

---


### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455634453/1269724655405498429/1360675922608521438)** (10 messages🔥): 

> `Web Search Agent, Open Source Scraping, Vertex AI Agent Builder, Brave Search API, SwissKnife` 


- ****Agent Data Pipeline Web Search Initiated****: A member is creating a **web search agent** for their agent data pipeline and requested recommendations for **open source lists of websites** easy to scrape reliable data from.
   - The agent builder preview is available in **Vertex AI**, also including **MCP**.
- ****Brave Search API is a treasure trove****: A member suggests the **Brave Search API** as a good alternative, noting good experience even on the free tier.
   - The API's **AI summarizer** is significantly cheaper than OpenAI's web search API.
- ****SwissKnife Project Sharpened for Review****: A member requests feedback on the **SwissKnife** project ([repo link](https://github.com/endomorphosis/swissknife/tree/blueprints)), focusing on **Claude code APIs**, **WebGPU**, **GraphRAG**, and **Graph of Thoughts** integrations.
   - They are particularly interested in the project's architecture ([docs link](https://github.com/endomorphosis/swissknife/tree/blueprints/docs/architecture)).
- ****Trust Dynamics Explored in New Article****: A member shared an article on trust dynamics ([Trust, Communication and Power in a Fragmenting World](https://www.tandfonline.com/doi/full/10.1080/09515089.2023.2223221)), suggesting its relevance for alignment and addressing dilemmas implicating social contract dynamics.
   - The paper discusses *trust, communication, and power in a fragmenting world*.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1360444448580047100)** (23 messages🔥): 

> `OpenAI recruiting video, Solomonoff's theory, Gen AI Use Case Report, Character.AI user base, GPT-4.5 being a talking model` 


- **Praise for OpenAI's Recruiting Video**: A member found [OpenAI's recruiting video](https://openai.com/careers/life-at-openai) dispelled impressions of "cult-like thinking patterns" and highlighted the company's systematic, pragmatic, and goal-oriented approach.
   - Another member agreed, noting the video suggests a broader range of opinions are freely discussed within the company, while another member hoped for a specific employee to be featured on a podcast.
- **Solomonoff's theory Sparks Intrigue**: A member described [Solomonoff's theory of inductive inference](https://en.wikipedia.org/wiki/Solomonoff%27s_theory_of_inductive_inference) as *extremely intriguing stuff*.
   - Another member agreed.
- **Gen AI Use Case Data Skewed by Reddit?**: Members discussed the [The 2025 Top-100 Gen AI Use Case Report](https://learn.filtered.com/hubfs/The%202025%20Top-100%20Gen%20AI%20Use%20Case%20Report.pdf), with one suggesting the data might be skewed due to **Reddit** being the only data source.
   - It was argued that *it's hard to survey anywhere without bias* unless you are **OpenAI**, and that **Character.AI** has **28 million users** but is never talked about in **ML** circles.
- **Users just wanna chat with GPT-4.5?**: Members considered how **ChatGPT** adding memories and **GPT-4.5** being a talking model may suggest that many users primarily *just chat* with these AIs.
   - A member noted that most of their AI usages, except for generating code, dropped between **2024** and **2025**.
- **GPU Prices Set to Skyrocket?**: Members shared concern regarding **GPU** prices, as reported in [this article](https://www.msn.com/en-gb/lifestyle/shopping/gpus-and-tariffs-why-i-recommend-buying-a-new-graphics-card-now-before-the-prices-climb-even-higher/ar-AA1CK4QR), are expected to climb.
   - One member wondered how long **Sam Altman**, **Mark Zuckerberg**, and **Elon Musk** will remain best buddies with **Mr. Tariff** when their **GPUs** will cost double.


  

---


### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1361407098826657983)** (1 messages): 

> `Llama 4 Maverick and Scout, SmolVLM, Diffusers 0.33.0, AI Agents Sustainability, Arabic Leaderboards` 


- **Hugging Face Welcomes Llama 4 Maverick & Scout**: The Hugging Face community welcomes [**Llama 4 Maverick** and **Llama 4 Scout**](https://huggingface.co/blog/llama4-release), with tests showing how they perform on the **DABStep benchmark**.
   - It was reported that **Claude 3.7 Sonnet**, **Gemini 2.5 Pro**, **Llama 4 Maverick**, and **Llama 4 Scout** were all tested and compared.
- **Diffusers Releases Version 0.33.0 with New Features**: **Diffusers 0.33.0** is released, introducing [new image and video generation models](https://huggingface.co/blog/fastrtc-cloudflare) along with various **memory optimizations**.
   - This update brings a wide suite of memory optimizations, catering to both image and video generation tasks.
- **Exploration of AI Agent Sustainability**: An article discusses the [sustainability of AI Agents](https://huggingface.co/blog/sasha/ai-agent-sustainability), emphasizing that *it depends* on various factors.
   - This blog post offers insights into what determines the sustainability of AI agents.
- **Gradio Reaches 1 Million Users Milestone**: The [Gradio platform](https://huggingface.co/blog/gradio-1m) celebrates reaching **1 million users**, marking a significant milestone for the library.
   - The blog post details the *journey* to achieving this milestone.
- **Unveiling NaFlex Integration in timm**: **NaFlex** is now integrated within **timm**, as detailed in [this blog post](https://huggingface.co/blog/rwightman/timm-naflex), enhancing the capabilities of the library.
   - The article explores the functionalities and benefits of **NaFlex** within the **timm** framework.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1360430374358089921)** (360 messages🔥🔥): 

> `Robotics Simulation Roadmap, LibreChat Duplication Issues, Ollama Syllabi Tool for Curriculum Generation, Parquet Files to Hugging Face, MLX Eagle Speculative Decoding` 


- **Users Seek Robotics Simulation Roadmap**: A user is seeking guidance on learning **Robotics Simulation**, mentioning **ROS2, SLAM, NVIDIA Isaac, Gazebo, and Mujoco**, and expressing confusion about where to start.
   - Another user expressed similar difficulties in getting into the robotics sim field for 2 years, while a third party promoted their [Ollama repo](https://github.com/Ollama-Agent-Roll-Cage/oarc-osyllabi) for generating customized learning paths or curriculums.
- **LibreChat Duplication Stumbling Blocks**: A user reported issues duplicating **LibreChat** on Hugging Face, and other members pointed to Github issues and a potentially broken Dockerfile.
   - A member supplied a [workaround Dockerfile](https://huggingface.co/spaces/LibreChat/LibreChat/blob/main/Dockerfile#L4) snippet to circumvent the issue.
- **Issuu PDF scraping codes discovered**: A member sought advice on scraping papers from **Issuu**, and another member recommended using `for` statements and `requests` with a list of URLs.
   - Another member shared [code snippets for downloading PDFs from Issuu](https://github.com/Mustkeem324/Issuu-PDF-Downloader/blob/main/main.py#L57) using the site's JSON API.
- **HF Models Go MIA**: Users reported widespread **404 errors** when trying to access Hugging Face models, bringing their apps down, and wondered if a dev could please get this fixed.
   - A user posted a [link](https://discuss.huggingface.co/t/404-error-model-xlabs-ai-flux-realismlora-does-not-exist/150363/5) tagging a specific HF employee, and mentioning this 404 error had persisted most of the day already.
- **Audio Datasets get Aesthetically Judged**: A member stated they wanted to see audio aesthetics and DNSMOS (CE/CU/PC/PQ) used to prefilter audio, as seen on [this page](https://huggingface.co/datasets/MrDragonFox/DE_Emilia_Yodas_680h).
   - This came up in the context of a discussion of a new [ParquetToHuggingFace tool](https://github.com/pr0mila/ParquetToHuggingFace) and a member suggested there was *no need for custom libs* for that, because HF has you covered by default.


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1360359908259794964)** (37 messages🔥): 

> `Ollama Agent Roll Cage, ML Guidance, Implementation from Scratch, KrishNaik and freecodecamp, Deep Learning Specialization` 


- ****Ollama Agent Roll Cage** is now live!**: The new **Ollama Agent Roll Cage** project is live at [OARC GitHub](https://github.com/Ollama-Agent-Roll-Cage/oarc), always looking for new contributors and testers.
   - They just implemented their new **creepy crawler package** to index GH, discord channels, etc at [OARC Crawlers](https://github.com/Ollama-Agent-Roll-Cage/oarc-crawlers) and will be releasing their intelligent rag system later this week in beta mode at [OARC RAG](https://github.com/Ollama-Agent-Roll-Cage/oarc-rag).
- **Guidance for **ML Beginners****: After completing the basic course of Andrew Ng, the next step is to **work on projects**, specifically related to ML, like image classification or data prediction.
   - Starting with basic projects like **linear regression** will help beginners learn about ML.
- **Implementing **ML Algorithms** from Scratch**: A member is looking into a [YouTube playlist](https://www.youtube.com/watch?v=p1hGz0w_OCo&list=PLcWfeUsAys2k_xub3mHks85sBHZvg24Jd) for implementing algorithms from scratch.
   - The recommendation is to check out **Krish Naik** and **freeCodeCamp** YouTube channels, but to clear fundamental concepts first.
- ****Deep Learning Specialization****: A member asked if the [Deep Learning Specialization on Coursera](https://www.coursera.org/specializations/deep-learning) is recommended for someone who just finished basic stuff, even though it's a **paid course**.
   - The recommendation is to clear the **basic fundamentals** first, then jump into advanced topics, starting with **Python**, then ML, then deep learning.


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1360362393883902152)** (2 messages): 

> `TLDR Service` 


- ****TLDR Service** gets native integration**: A member is building a native integration directly into their **TLDR service**.
   - This could potentially enhance the service's ability to provide concise summaries.
- **Another Topic**: Another topic was discussed.
   - More details about this topic.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1360404838193561650)** (20 messages🔥): 

> `Universal Intelligence protocols released, Speaker Isolation Toolkit, MLX EAGLE-2 Speculative Decoding, gpu-spaces script, SwissKnife request for comment` 


- **Universal Intelligence Protocols Released**: A new project called [`universal-intelligence`](https://github.com/blueraai/universal-intelligence) was released, comprising **3 open source protocols for models, tools and agents**.
   - It includes a **community based library of ready-made components** for instant usage and deployment, aiming for simple, composable, portable, scalable & auto-optimized usage of AI.
- **Speakers get Isolated with New Toolkit**: A new [speaker-identification-toolkit](https://github.com/ThatJeffGuy/speaker-identification-toolkit) helps isolate speakers in multi-speaker recordings for creating ML audio datasets.
   - Once **10%** of the data is manually identified, the toolkit automatically isolates the speaker for the remaining files using **CUDA** or **CPU**.
- **GPU Spaces filter Saves Sifting Time**: A member created [a script](https://huggingface.co/spaces/DeathDaDev/gpu-spaces) that filters Hugging Face Spaces to show those with non-zero GPU enabled.
   - This helps users sift through featured spaces and **find those with available GPU resources**.
- **Deep Search Agent seeking Early Adopters**: A new agent focused on **Deep Search** using *smolagents* has been built, and early testers are being sought at [agent.galadriel.com](https://agent.galadriel.com/).
   - Feedback is welcome, with a request to reach out with questions and ideas to the product team.
- **EAGLE-2 Speculative Decoding with MLX**: An updated code for **MLX EAGLE-2 Speculative Decoding** improved the throughput from **18** to **22** tps with mlx_lm speculative decoding, as seen in [this repo](https://github.com/0seba/mlx-eagle-spec-decoding-tree/tree/main).
   - The developer seeks experimentation with bigger models, especially on architectures >=M3, due to limited resources.


  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1361448243652984862)** (2 messages): 

> `Society of Minds framework` 


- ****Society of Minds Framework** Discussion Incoming!**: A reading group voice chat will be held this week to discuss the **"Society of Minds" framework**, with a link to the [Discord event](https://discord.com/events/879548962464493619/1351984543376085062) provided.
- **Paper Link Provided**: The paper to be discussed is available at [OpenReview.net](https://openreview.net/pdf?id=zj7YuTE4t8).


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1360642900760727564)** (2 messages): 

> `rf-detr-uslsohoy, CV Hangout` 


- **Community Discusses rf-detr-uslsohoy**: A member shared a [GitHub repository](https://github.com/egorsmkv/rf-detr-uslsohoy) with the community.
- **CV Hangout Location Inquiry**: A member inquired whether the linked **rf-detr-uslsohoy** repo will host the **CV Hangout** on the 16th.


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1360680923330248785)** (2 messages): 

> `Local LLM Models, OlympicCoder 7B and 32B, DeepSeek R1 Distil Qwen 7B, DeepCoder 14B Preview, fine tuning facebook nllb language translator model` 


- **Users test Local LLM Models**: A member wanted to know what **local LLM models** others are using, with primary use case being **coding, reasoning, planning, and writing**.
- **Discussing OlympicCoder models**: The member tested **OlympicCoder 7B and 32B**, and found them decent for coding, but they tend to ramble and that they might need to tweak some settings.
   - They also tried **DeepSeek R1 Distil Qwen 7B**, which they found much better for reasoning, and doesn't ramble.
- **Fine-tuning facebook nllb language translator model**: The member is currently working on **fine tuning facebook nllb language translator model** to a **csv file** of english sentence and a **tibetoburman language** parallel translation, and wanted to talk to someone regarding it.


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1360338024726265897)** (7 messages): 

> `Agent course deadline, Agent course use cases` 


- **Agent Course Deadline Debunked**: Multiple users inquired about the **Agent course deadline** for completion and certification, specifically mentioning **May 1st**.
   - One member clarified that *there is no deadline* and participants can finish the course at their own pace.
- **Agent Course Use Cases: Unveiling the Mystery**: A user expressed confusion about the **use case assignments** for the Agent course, stating they *have found none*.
   - This suggests a potential lack of clarity or accessibility regarding the course's practical application component.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1360371607566287112)** (41 messages🔥): 

> `Course Certification, HF API Limit Issues, Ollama Setup and Usage, LLM Fine-Tuning for Agents, SmolAgent's Pope Obsession` 


- **Course Certification Status Remains Uncertain**: Participants are questioning the value of *signing up* for the course, as the schedule is behind, and there's a lack of communication regarding the final certification.
   - Some suggest focusing on learning the material and ignoring the certification aspect unless HuggingFace provides updated information, while others are curious about the **use case assignment** and how to submit it.
- **Users Encounter HF API Limit Issues**: Many users are hitting **Hugging Face API limits** quickly, even with premium subscriptions, causing issues with the tutorial notebooks.
   - Suggested solutions include switching to **Gemini** or running models locally with **Ollama** to bypass these limitations, though local setups may require tweaking code to accommodate model limitations.
- **Ollama Solves Some Local LLM Challenges**: Members discuss using **Ollama** to run models locally, sharing commands to download and run specific models like `qwen2.5-coder:32b`.
   - One member provided a code snippet demonstrating how to specify the **Ollama provider** when initializing a `CodeAgent` with a locally hosted model like `bartowski/Qwen2.5-Coder-32B-Instruct-GGUF`.
- **Instruct Models Suited for Agent Framework**: Members discussed whether **LLMs** need specific fine-tuning to work with agent tools, deciding that any instruct model works, with larger models being better.
   - However, matching the model to the framework's syntax or adjusting prompts can improve results, as newer models may include examples of tooling/agent-like behavior in their training.
- **Agent's Fixation with Pope's Age**: One user reported their **agent** was inexplicably obsessed with finding the **Pope's age** and squaring it to **0.36** when running locally with models like `llama3`, `deepseekr1:8b`, and `qwen2.5-coder:latest`.
   - The issue was suspected to originate from a hardcoded sample within the **smolagent** default agent tool prompts, as it didn't occur when using **HfApiModel**.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1360449822573723750)** (54 messages🔥): 

> `Model Similarity Analysis, Dataloader batching strategies, Multiple token prediction, Input-dependent LoRAs, Stereoisomer encoding for chemical features` 


- **Models Look Kinda Similar**: A member was *surprised at how similar different models look* when comparing post-MLP hidden state cosine similarity between sequences using a janky script [available here](https://github.com/pwspen/lmsim).
   - They found that small models group by type more than color, while larger models rank by color much more consistently.
- **Repeat Data? Fern No!**: It was advised to not repeat data within a minibatch, as *that can cause major issues!*
   - A user shared about investigative information analytics within cognitive science and ML/AI, facilitating insights across disciplines, and communicating those to different parties.
- **Multiple Token Prediction Papers Sought After**: A member sought papers on multiple token prediction with LLMs during inference, and another user suggested [DeepSeek v3](https://openreview.net/forum?id=pEWAcejiU2).
   - Another user pointed to [this paper](https://arxiv.org/abs/2401.10774) and recalled seeing one from Meta years ago.
- **LoRA MoE Explored for Input Dependence**: Regarding input-dependent LoRAs, one member uses them extensively in **RWKV** to save parameters, suggesting exploration of **MoE LoRA**.
   - Another member clarified that while they use LoRAs in place of normal linear transformations in **RWKV**, the weights themselves are not input-dependent, prompting a discussion on potential architectures where the LoRA weights are sensitive to the input, drawing a comparison to **MHA**.
- **LLM Aspirant Gets General Advice**: A software engineer from Amazon and Salesforce sought ways to get involved in **LLM research** or projects.
   - A member provided a general guide, including learning Python, applied ML/AI, and then specializing in a subfield such as robotics, ML theory, audio, visual, language, or interpretability.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1360401027257340068)** (236 messages🔥🔥): 

> `arXiv endorsement request, AI-generated content in research, Token loss and length extrapolation, Visual autoregressive models for microscopy images, Policy enforcement` 


- **ArXiv endorsement sought, ethical AI structural control framework revealed**: A member requested endorsement for their [arXiv submission](https://arxiv.org/abs/2504.05518v1) (cs.AI / cs.CY), presenting a **natural language-based institutional framework** for **ChatGPT** to *stabilize output* and *ensure ethical consistency*.
   - The framework includes **prompt-based structural verification and alignment logic** without model modification.
- **Discord grapples with influx of LLM-driven "research"**: Members voiced concerns about the rise of **AI-generated content** presented as research, which is often characterized by *made-up terminology* and *lack of alignment with legitimate research ideas*.
   - Suggestions included a **ban for bad-faith users** hiding AI usage and a **long-term mute for good-faith users** exhibiting inexperience.
- **Test-Time Scaling & CoT questioned, is it an RL artifact?**: A member questioned the necessity of *test-time scaling* and generating *very long Chain-of-Thoughts (CoTs)*, suggesting it might be an artifact of **RL training methods** as shown in [these papers](https://arxiv.org/abs/2504.04383), [https://arxiv.org/abs/2504.01296](https://arxiv.org/abs/2502.07266), [https://arxiv.org/abs/2503.20783], [https://arxiv.org/abs/2503.04697).
   - Others suggested **CoT** is *not about the actual generated tokens*, but a way for the model to perform *more iterations/computation* by manipulating attention weights.
- **Extrapolating Length for Model Gains, or perhaps it maintains?**: Members discussed challenges in **length extrapolation**, noting that models often *fail to consistently decrease token loss* beyond their training sequence length, as shown in [this plot](https://cdn.discordapp.com/attachments/747850033994662000/1361359728147431685/Screenshot_2025-04-14_at_16.17.22.png?ex=67fe788c&is=67fd270c&hm=4fe0f240d28501a17e80c46f6c0848297dd2361ec50593f31cc697d50bccd0e5&).
   - Techniques like **NoPE + SWA** and **ssmax** ([Super Scaling Max Activation](https://arxiv.org/abs/2501.19399)) were mentioned as potential solutions to help the model remember further back than its sequence length, although there is debate as to the best information flow strategy.
- **Microscopy Images generated by VAE autoregressive model**: A member shared results of generating **3-channel microscopy images** using a visual autoregressive model conditioned on class embeddings from a trained **DINO** model.
   - The generated images exhibit a noticeable **whitish hue**, potentially due to bias from the **ImageNet-trained VAE visual encoder**.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1360536090246778920)** (14 messages🔥): 

> `Graph attribution mechanistic interpretability, Distillation effects on model circuits, Models' knowledge of their circuits, Reasoning model self-awareness, CoT fidelity in reasoning models` 


- **Call Made for Graph Attribution Mechanistic Interpretability**: A member suggested doing **graph attribution mechanistic interpretability** on new reasoning models, noting differences between circuits and model explanations, referencing [this paper](https://transformer-circuits.pub/2025/attribution-graphs/biology.html).
- **Distillation Impacts Circuit Knowledge**: Members expressed concern that **distillation** might reduce models' knowledge of their circuits, suggesting distilled models like **Llama** and **Qwen** may have even less circuit awareness.
- **Debate Erupts on Models' Self-Awareness of Circuits**: A member questioned whether models have any knowledge of their circuits, challenging the tendency to overstate model understanding without sufficient evidence.
   - Another member argued that reasoning models might acquire some knowledge of problem-solving strategies through **RL**, potentially leading to better self-understanding of their circuits.
- **Quantifying Model Self-Knowledge: The Calibration Conundrum**: Members discussed quantifying the limits of model self-knowledge, referencing a [calibration paper](https://arxiv.org/abs/2504.06564) that assesses how well models' self-reported confidence correlates with their measured accuracy.
- **Probing for CoT Invocation in Later Layers**: Members discussed probing probabilities of tokens that allow an agent to invoke **CoT** or to retrieve knowledge before triggering an action, referencing [this paper](https://arxiv.org/abs/2504.03553) and noting that the decision to do so mostly pops up in later layers.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1360333437227241529)** (99 messages🔥🔥): 

> `Karpathy asks ChatGPT embarrassing questions, Thinking Machines $2B round, OpenAI SWE coming, GPT 4.1 Quasar launch, DeepSeek Inference Engine Open Source` 


- ****Karpathy** Tries To Embarrass **ChatGPT****: A user shared a [prompt](https://x.com/karpathy/status/1910734302931017812) asking **ChatGPT**: *What's the most embarrassing thing you know about me?*
   - The user encouraged others to push **ChatGPT** for honest and direct answers through multiple rounds of questioning.
- ****Mira** Raises **$2B Seed** Round for Thinking Machines**: **Thinking Machines** is doing a **$2B seed round**, advised by **Alec Radford**, according to a discussion referencing a [Fortune article](https://fortune.com/2025/04/10/mira-murati-2-billion-seed-raise-ai-boom-economic-chaos/)
   - A user posted a *good chart* from [Epoch AI](https://x.com/epochairesearch/status/1910788295405252916) illustrating the raise.
- ****DeepSeek** Opens Up Its **Inference Engine****: **DeepSeek** has open-sourced its inference engine, with the [GitHub repo](https://github.com/deepseek-ai/open-infra-index/blob/main/OpenSourcing_DeepSeek_Inference_Engine/README.md) available for review.
   - Members wondered if anyone wants to chat about **DeepSeek's** open sourcing today.
- ****GPT-4.1** and **Quasar** Launch Rumors**: Discussion surrounded the launch of **GPT-4.1** and a model called **Quasar**, based on a [Reddit post](https://www.reddit.com/r/singularity/comments/1jz2jhu/openai_confirmed_to_be_announcing_gpt41_in_the/) and the [official announcement](https://openai.com/index/gpt-4-1/).
   - Speculation arose that **GPT-4.1** might become the *de facto* coding model, supplanting **Gemini 2.5**, and discussion ensued about the deprecation of **GPT-4.5**.
- ****Grok** Adds Features But Stays Mum**: A user noted that **Grok** is shipping significant features without announcements, including *cross-conversation memory* and a new *workspaces* feature.
   - Speculation suggests they might be testing quietly before a larger announcement.


  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1361376237486080111)** (2 messages): 

> `Quasar launch, SFCompute pod` 


- **Quasar Watch Party Incoming**: Latent Space is hosting another watch party for the **Quasar launch** today in 35 minutes, at [this discord event](https://discord.gg/rPJq8UU2?event=1361376118510321724).
- **SFCompute Pod Needs Boost**: Latent Space asks for help in circulating their **SFCompute pod**.
   - See [this tweet](https://x.com/latentspacepod/status/1910777555101376757) for more details.


  

---


### **Latent Space ▷ #[llm-paper-club-west](https://discord.com/channels/822583790773862470/1197350122112168006/1360341267950600262)** (5 messages): 

> `X-Ware.v0, AI news source` 


- **X-Ware.v0 posted on X**: A member shared an image from [X-Ware.v0](https://xcancel.com/dylan522p/status/1911843102895358198), asking *what's that from?*.
- **AI news source**: A member simply replied, *ainews*, to the question about the origin of an image.


  

---


### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1360343642702872590)** (186 messages🔥🔥): 

> `Agent Definitions, Langsmith Tool, Visibility into training process, Model Benchmarking Tools, GPT-4.1 launch` 


- **Agent Definitions get vibe-defined**: Members debated the definition of an "agent", with one suggesting today's definition: *an LLM calls a tool* while another presented [a Figma board](https://www.figma.com/board/aCaUWEr039dHmpW9ssGJmK/self_improving_agents?node-id=137-796&t=zsQXjScFlAekKtEd-1) on self-improving agents.
   - One suggested: *the agent you vibe code while bored in a meeting*.
- **Langsmith tool gets enjoyed**: Members discussed **Langsmith** as a tool for LLM instrumentation, with one mentioning they enjoy it even for non-Langchain projects and linked to the [Arize docs](https://docs.arize.com/arize).
   - Another member suggested using a pre-vibe-coded bot to tag links in chat for future processing but noted complaining is cheaper.
- **Visibility into training process**: During a live presentation, a member asked how to gain visibility into the training process of neural networks, prompting discussion around tools and methodologies.
   - Suggestions included using **WandB** ([Weights & Biases](https://wandb.ai/site)) and **TruLens** ([trulens.org](https://www.trulens.org/)) for LLM traces and evaluations.
- **Tools for Benchmarking Models**: Members discussed various benchmarking tools for comparing models, including [lighteval](https://github.com/huggingface/lighteval) from Hugging Face and [BenchmarkAggregator](https://github.com/mrconter1/BenchmarkAggregator?tab=readme-ov-file).
   - One member mentioned that these tools could be useful for comparing parameters in an evaluation loop.
- **OpenAI launches Quasar**: During an OpenAI Quasar launch watch party, members discussed the features of **GPT-4.1**, including its competitive pricing compared to Claude and flat pricing on long input contexts, referencing the [pricing documentation](https://platform.openai.com/docs/models/gpt-4.1).
   - One member highlighted the cheapest model being free for 7 days, and another joked about drinking windsurf ad during the presentation.


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1360342198318661633)** (18 messages🔥): 

> `Non-deterministic NLM, NLM in education, Gemini Education Workspace, Conversational interface, NotebookLM for University` 


- ****NLM's Latent Space causes variability****: A member stated that *variability of the latent space* causes the inability to generate the same output every time, and that the system lacks coherency, resulting in random generations based on the input each time.
- ****NLM is not a Ferrari****: According to one member, you can't expect a Prius to be a Ferrari and that if you want a Ferrari, it's going to be expensive and you won't find it in Google NotebookLM.
   - Another member clarified that NLM is not designed to be a deterministic system.
- ****NLM transforms Education****: A member uses **NotebookLM** in their classroom by uploading slide decks and materials, creating notes, study guides with quiz questions, a glossary of terms, mind maps, and an audio overview, then shares it with students to help them prepare for exams.
   - They are also having students create their own NotebookLMs in groups.
- ****NSW is missing out on Gemini****: A member asked if others are using it through an Education Workspace, as they are interested to see districts and departments who are happy to use **Gemini** within their Workspaces.
   - They note that in NSW, Australia, they cannot yet use **Gemini**.
- ****Diabetic cat owners need chatbots****: A member runs a large support group for owners of diabetic cats and wants to provide their members with a conversational interface to their documentation, including video content, and in French.
   - They would like members to ask questions and get answers based on documentation with links to relevant docs to read.


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1360354819428061335)** (141 messages🔥🔥): 

> `NotebookLM for students, Google Agents and NotebookLM, Notebook search function, Discover feature in NotebookLM, Deep research problems in Gemini` 


- **NotebookLM Sparks Student Interest**: A user inquired about learning to use **NotebookLM** for MP2I studies in France, covering math, coding, and physics.
   - Others pointed the user towards the **Google Agents** signup to elevate their working environment, as described in [this Youtube video](https://youtu.be/WEEYPwBg6Qo).
- **NotebookLM "Discover" feature brings joy**: A user expressed great satisfaction with the new **"Discover sources"** feature in NotebookLM, stating *"It's everything I could have wanted"*.
   - The same user now awaits more **audio overview flavors** and expressed enjoyment of Grace's podcasts.
- **Audio Overviews speaks English no mas**: Users reported that the audio overview feature in NotebookLM no longer reliably supports languages other than **English**, despite previous hacks.
   - A user experiencing podcast generation only in English reported this as difficult, as English is not their native language.
- **NotebookLM PDF Deep Dive Derailed?**: Several users reported trouble with **deep research in Gemini**, specifically with PDFs failing to load as sources when uploaded to NotebookLM.
   - One user suggested that it's a temporary glitch but to try with another document and make sure it stays within the **500k word limit**.
- **Lost in Translation: UI Language Labyrinth**: Users reported difficulties in **switching NotebookLM back to English** after changing the output language, with the setting now missing and UI language settings not affecting the output.
   - One user confirmed there's an active issue and posted a link to the [bug channel](https://discord.com/channels/1124402182171672732/1354708696474718218) while others suggested trying to change the **Google account language**, clearing cookies, or using the `?hl=en` parameter.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1360372931166867597)** (109 messages🔥🔥): 

> `Llama 4 Maverick & Scout, DeepCoder Model, Nvidia UltraLong Models, GPT-4.1 Pricing & Performance, Gemini 2.5 Pro` 


- **Llama 4 Wasted GPU Hours**: Members discussed **Meta's Llama 4 Maverick** used **2.38M GPU hours**, the same as training **Deepseek V3**, while **Llama 4 Scout** took **5.0M GPU hours**.
   - Some pointed out that other models are tuned for human preferences, questioning the fairness, while others noted **LeCun's** possible involvement.
- **DeepCoder Achieves Top Coding Performance**: A member shared a [VentureBeat article](https://venturebeat.com/ai/deepcoder-delivers-top-coding-performance-in-efficient-14b-open-model/) about **DeepCoder**, highlighting its efficient **14B** parameter open model and enhanced **GRPO algorithm**.
   - The model features **offline difficulty filtering**, no entropy loss, no **KL loss**, and overlong filtering from **DAPO**, generalizing to **64K** context despite training with **32K**.
- **Nvidia UltraLong Models Process Extensive Sequences**: **Nvidia** is using research to create **UltraLong-8B** models, as featured in this [Hugging Face collection](https://huggingface.co/collections/nvidia/ultralong-67c773cfe53a9a518841fbbe), designed to process sequences up to **4M tokens** built on **Llama-3.1**.
   - It combines continued pretraining with instruction tuning, trained for **150 iterations** with a **4M sequence length** and a global batch size of **2**.
- **GPT-4.1 Benchmarks Better Than Past Releases**: Members discussed pricing and benchmarks for **GPT-4.1**, with one noting that [benchmarks are better](https://openai.com/index/gpt-4-1/) than past releases, but the pricing and model versioning are confusing, with the new model available in **GitHub Copilot**.
   - There was also some talk of **4.1-nano** being on par with good **14B** models, and some speculation on whether this model will be open sourced.
- **Gemini 2.5 Pro Pricey?**: Members debated whether the new version of **GPT-4.1** is worth it vs using **Gemini 2.5 Pro and Sonnet 3.7**.
   - Although Gemini may seem cheaper at first, it is actually more expensive due to lack of free caching and its tendency to fluff responses, whereas GPT-4.1 is more to the point.


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1360634754164195529)** (15 messages🔥): 

> `Loss Observations on H100 Llama 4 Scout, Small Model Training Challenges, Dataset Recommendations for Small Models, Surya and SmolVLM2` 


- **Loss Increases Observed on H100 Llama 4 Scout**: A member noted an increasing loss from **1.9011** to **2.3407** between epochs **1** and **2** during training of a **Llama 4 Scout** model on an **H100** setup.
   - They were concerned because the loss didn't decrease as expected, despite using two **H100** GPUs.
- **Debate Arises around the Size of the Model**: Members discussed the implications of training a very small model (**1-2M parameters** with **20M tokens**) and how this impacts the observed loss.
   - One member suggested that *the minimum you should work with is 10M parameters no matter what the task is*.
- **Dataset for Small Models**: A member shared their experience switching from the **Wiki 103 dataset** to fine-tuning **Phi2**, implying a change in approach to address the observed training issues.
   - The member stated they switched to fine tuning Phi2 to address their training issues.
- **Surya and SmolVLM2**: A member recommended checking out [Surya](https://github.com/VikParuchuri/surya), emphasizing that it's *not a VLM but very impressive*.
   - They also suggested [SmolVLM2](https://huggingface.co/blog/smolvlm2) for those specifically looking for a VLM.


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/)** (1 messages): 

ee.dd: https://ai-2027.com
  

---


### **Nous Research AI ▷ #[reasoning-tasks](https://discord.com/channels/1053877538025386074/1264666760972472481/1360451095578546206)** (14 messages🔥): 

> `Research Paper on Repo, Task Quality Assurance` 


- **Research Paper on Reasoning Repo?**: A member inquired about the possibility of writing a research paper based on the repository, and another member expressed interest in collaborating on writing, despite admitting *"Im not much of a writer tho"*.
- **Task Quality Verification Asked For**: A member raised concerns about ensuring the quality of tasks in the compilation, suggesting a need for double-checking, and proposing to create questions as requirements for each task.
   - Another member asked *"How would you think to verify it?"*.


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1360343887255703552)** (99 messages🔥🔥): 

> `MCP for Reddit and Quora, Paid bounty for MCP server setup, ADK and A2A vs MCP, Exposing tools to the user, Passing tools to the LLM` 


- ****Graphlit MCP** server for Reddit and Quora**: Members discussed building an **MCP server** for Reddit and Quora, with [Graphlit](https://github.com/graphlit/graphlit-mcp-server) offering to add Quora ingestion if needed.
   - Currently a few exist for Reddit, such as [this repo](https://github.com/hawstein/mcp-server-reddit).
- **Compsci team needs help to get **MCP server** working, willing to pay bounty**: A member offered to pay a bounty for help setting up an **MCP server**, as their university compsci team is struggling with **GhidraMCP** which returns a *404 - NO CONTEXT PROVIDED* error.
   - The team is using Cursor IDE to try and make it work.
- ****ADK and A2A** worth reading, in addition to **MCP****: A member suggested exploring **ADK and A2A** from Google, noting their similarity to **MCP** and potential centrality to the internet of agents, sparking a discussion on their relevance and use.
   - Another member confirmed that there is no official consensus on non-MCP tech talk, but if it's at least somewhat relevant to AI/ML/MCP then there should be no issues.
- **Discussing about passing tools to the **LLM****: Members are sharing thoughts on how the tools relevant to a specific user prompt passed to the **LLM**, a member shared that all enabled tools are passed with the prompt.
   - As an alternative, a member shared a [video](https://www.youtube.com/watch?v=3ISRS2hQlfI&t=195s) demonstrating vector tool calling.
- ****Wildcard** pauses further maintaining **agents.json****: The Wildcard team announced they are pausing further maintenance of **agents.json** due to MCP's increasing adoption by large model providers.
   - They believe the concepts will eventually integrate into MCP, like the recent stateless HTTP transport.


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1360371569410441256)** (36 messages🔥): 

> `Models without Function Calling, MCP Bug Spotting Tools, Paprika Recipe MCP Server, Oterm Release and MCP Sampling, AutoMCP for Agent Deployment` 


- **Block Tweaks Models Lacking Function Calling**: Block is experimenting with models that lack function calling abilities to see if they can tweak their output to work with agents, and [this blog post](https://block.github.io/goose/blog/2025/04/11/finetuning-toolshim) explores doing that without a secondary model via XML output.
   - The team is weighing the latency costs versus the benefits of using a secondary model for parsing, with concerns about longer sessions and the ability to stick to the XML format, and may use a *local model*, with concerns of more *overhead*.
- **MCP Tools for Debugging Copilot Client**: [synf](https://github.com/strowk/synf) and [mcptee](https://github.com/strowk/mcptee) help members spot and fix bugs while testing with Copilot client, which can struggle with longer contexts and more tools.
   - One member is building with fast hardware in mind, since *multiple API calls will always be slower than doing 1*.
- **Paprika Recipe App gets MCP Server**: An MCP server was created for anyone who uses the **Paprika recipe app**, so that Claude can automatically save recipes into Paprika via [this GitHub repo](https://github.com/soggycactus/paprika-3-mcp).
   - No further information was given.
- **Oterm Terminal Client Supports MCP Sampling**: Version **0.11.0** of [oterm](https://github.com/ggozad/oterm), the terminal client for Ollama, was released, focusing on adding support for [MCP Sampling](https://modelcontextprotocol.io/docs/concepts/sampling) in addition to existing support for **MCP tools** and **MCP prompts**.
   - The new release includes support for **sixel graphics**, an **in-app log viewer**, and the ability to create **custom commands** that can be run from the terminal.
- **AutoMCP simplifies Agent Deployment**: A new library and platform called **AutoMCP** was launched to easily convert and deploy existing agent projects as MCP servers, with [code on GitHub](https://github.com/NapthaAI/automcp), and deployed [on this platform](https://labs.naptha.ai/).
   - The service offers a Vercel/Heroku-like experience for AI agents, allowing users to prototype in familiar frameworks and deploy without worrying about backend, highlighted in [this YouTube video](https://www.youtube.com/watch?v=El5YvBQ5py0).


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1360331698511941722)** (18 messages🔥): 

> `CUDA in Python/PyTorch, AMD GPU Mode Competition, GTC talk by marksaroufim, Stephen Jones videos, channel owner` 


- ****CUDA** Guidance Crystallizes**: A member asked for **CUDA** references within Python/PyTorch models, and another member shared their recent [GTC talk](https://docs.google.com/presentation/d/1sipZ_sqdwJapHQAr23yBow43pF40lMZu/view?usp=sharing) about it.
   - The talk suggests that *custom ops* and *load inline* should address most problems, along with ongoing work to cut compilation times; the talk can also be found on [nvidia.com](https://www.nvidia.com/en-us/on-demand/session/gtc25-s71946/).
- ****AMD GPU** Mode Contest Awaits**: A member inquired about the **AMD GPU Mode Competition**, stating they registered a couple of days prior without receiving any updates.
   - Another member responded that *more info to come today*.
- ****Stephen Jones** Videos Spark Interest**: After watching the GTC talk, a member went down the rabbit hole with **Stephen Jones**' videos, which were referenced in the talk.
   - That member then said that *vacation is over* and *talks start again*.
- **Channel Owner Needs Pinging**: A member asked who the channel owner was and then requested an admin.
   - Another member responded that they can ping <@&1231246776103604326> and asked what they needed.


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1360545047430434978)** (5 messages): 

> `Morton Order vs Swizzle2D, Space-Filling Curves, Hilbert Curves vs Morton Ordering, Debugging Triton Memory Leaks, Implementing Triton Kernel` 


- **Hilbert Curves rival Morton Ordering**: A member inquired about space-filling curves better than **Morton order** for cache-friendliness, sparking discussion on alternatives like [Hilbert Curves](https://en.wikipedia.org/wiki/Hilbert_curve).
   - Another member noted that, *in theory*, **Hilbert Curves** are optimal but not hardware-efficient, suggesting **Morton ordering** is a better practical trade-off and pointing to [a blog post](https://blog.stackademic.com/how-the-idea-of-the-hilbert-curve-inspired-morton-curves-for-gpu-performance-4e235d670304) comparing the two.
- **GEMM Performance comparison using Hilbert Curves**: A member shared a [GitHub repo](https://github.com/lawmurray/gpu-gemm) showcasing GEMM implementation with **Hilbert curves**, along with [benchmarks against cuBLAS](https://indii.org/blog/gpu-matrix-multiply/).
   - The benchmarks indicate that **Hilbert curves** become more effective as the matrix size increases.
- **Debugging Triton Kernel Memory Leak**: A member sought advice on troubleshooting a **memory leak** in a **Triton kernel** that passes accuracy checks but causes out-of-memory errors during training.
   - The member highlighted inconsistent forward pass results compared to eager mode, suspecting a potential overflow issue and linked the repo [FlashDeBERTa](https://github.com/Knowledgator/FlashDeBERTa).
- **Seeking Guidance on Implementing Triton Kernel**: A member requested resources for implementing a **Triton kernel** to train a model using an expert retrieval architecture.
   - They are struggling despite reviewing the official documentation and linked the paper [Retrieval meets Long Context LLMs](https://arxiv.org/pdf/2407.04153).


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1361163559236796578)** (8 messages🔥): 

> `Dynamic KV cache tensors in CUDA, cuBLAS Batched GEMM, memcpy_async cooperative API, Async copies and uncoalesced global access, Shared memory alignment` 


- **Dynamic KV Cache Challenges in CUDA Matrix Multiplication**: Discussion around efficiently handling dynamic KV cache tensors in CUDA during QK.T and XV operations, specifically how to manage the varying `K-cache-length` for each user in a batch size of M with [Batched GEMM in cuBLAS](https://developer.nvidia.com/cublas).
   - The user questioned whether custom kernels are typically written to manage this, or if batched GEMM in cuBLAS can handle the variable `K-cache-length`.
- **`memcpy_async` slows kernel performance**: A user reported a significant performance slowdown after switching from a standard memory copy loop to `cuda::memcpy_async`, even though the correct `LDSTS` instructions were generated.
   - They observed that the async kernel was reporting uncoalesced global access despite using the same indexing as the non-async version, prompting questions about the correct usage of async copies.
- **`memcpy_async` requires a cooperative API**: It was suggested that `memcpy_async()` is a cooperative API, meaning all threads must pass the same pointer(s) and a size corresponding to the entire block of memory, referencing the [official CUDA documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-data-copies-using-cuda-barrier).
   - Doing this from each thread sequentially *prevents coalescing*, instead of enabling it.
- **`memcpy_async` alignment issues**: A [forum post](https://forums.developer.nvidia.com/t/coalesced-and-conflict-free-memory-access-using-cuda-memcpy-async-cp-async/306460/6) was referenced suggesting that potential problems with `memcpy_async` include the alignment of the shared memory address and conditionals around the instruction, which can hinder coalesced memory access.
   - Loops might also be problematic.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1361113122622144655)** (6 messages): 

> `Memory profiling distributed training, ATen attention.cu, torchscript jit CUDA optimizations, ZeRo Stage 3 PyTorch Lightning tutorial` 


- **Memory Profiling on SLURM Cluster Puzzles Engineer**: An engineer seeks advice on memory profiling a model trained on a **SLURM cluster** with **8 nodes**, each having **8 GPUs**, for distributed training.
   - They are doing this type of distributed training for the first time, so are looking for recommended routes.
- **ATen's attention.cu Implementation Investigated**: An engineer inquires about the implementation pointed to by a specific line in ATen's `attention.cu` ([link to GitHub](https://github.com/pytorch/pytorch/blob/101c4f482a4019896ca18184233bd27a758648bf/aten/src/ATen/native/transformers/cuda/attention.cu#L662)).
   - Specifically, they aim to understand how torch/CUDA handles individual user operands `[dHead x K-cache-length]` in a batch and whether `bmm_nt` invokes cuBLAS Batched GEMM to split the large matmul or if there's an alternative mechanism.
- **Nested Tensor Matmul Manages Variable Cache Sizes**: A member believes they found where variable cache sizes and individual caches in a batch are handled ([link to GitHub](https://github.com/pytorch/pytorch/blob/6dddd6520daf8768dc76d182d3b2b0130e87a49d/aten/src/ATen/native/nested/NestedTensorMatmul.cpp#L151)).
   - They hope their understanding is correct and that this implementation aligns with their thinking.
- **ZeRo Stage 3 Tutorial Requested**: A member asks if anyone has a tutorial to share on the implementation of **ZeRo Stage 3** with **PyTorch Lightning**.
   - No further discussion or details were provided.


  

---


### **GPU MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1360650228205883512)** (1 messages): 

> `RMSNorm vs L2 Norm, Llama Norm, Scout Embeddings` 


- **RMSNorm masquerades as L2 Norm**: A member clarified that **Llama** doesn’t use **L2 norm**; it uses **RMSNorm** without scaling, calling it **L2**, where the actual **L2(x) = sqrt(sum(x^2))**.
   - They noted that **Llama norm** is **sqrt(sum(x^2)/n)**, where *n* is the embedding dimension, leading to **-n <= qk^T <= n** with *n=8192* for scout.
- **Scout Embedding Dimensions Clarified**: The discussion highlighted that for **Scout**, the embedding dimension *n* in the **Llama norm** calculation is equal to **8192**
   - This clarification emphasizes the specific numerical range **-8192 <= qk^T <= 8192** applicable in the context of **Scout's** architecture.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1360805799961559141)** (18 messages🔥): 

> `CUDA events, Maxwell tuning guide, shared memory, PTX and SASS, LOP3.LUT` 


- **CUDA events synchronization unnecessary**: According to one member, if **CUDA events** are used for timing, synchronization is unnecessary, but if **host-side timings** are used, synchronization is required.
   - The member stated, *"I don't know what PyTorch does, but if they use CUDA events for timing, there is no reason to synchronize. If they use host-side timings, one needs to synchronize in between, yes."*
- **Maxwell Tuning Guide: Block Allocation**: The Maxwell tuning guide suggests allocating no more than **32K** to a block (out of the available 48KB), so that **2 blocks** can fit within an **SM**.
   - Another member explained that with a single block per **SM** and **block-synchronization** you will get suboptimal performance when most warps are already waiting on the barrier while few still have work to do.
- **NVIDIA's PTX ISA documentation shared**: A member shared NVIDIA's documentation on **PTX ISA**, which is extensive and useful for learning about PTX.
   - The linked resource is available [here](https://docs.nvidia.com/cuda/parallel-thread-execution/#introduction).
- **Reverse Engineering SASS instructions**: Due to lack of official documentation, understanding **SASS** requires reverse engineering and searching NVIDIA forums, particularly for insights from former NVIDIA employee **njuffa**.
   - An example thread explaining `LOP3.LUT` instruction can be found [here](https://forums.developer.nvidia.com/t/what-does-lop3-lut-mean-how-is-it-executed/227472/3).


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1360672489184956446)** (3 messages): 

> `QLoRA Training, 4bit quantization, QAT for all layers in a model` 


- **QLoRA Training with sub-4bit quantization surfaces**: A member inquired about literature on **QLoRA** style training using *less than 4-bit quantization*.
   - Another member provided a [link](https://mobiusml.github.io/1bit_blog/) on the topic.
- **QAT Papers Sought**: A member is working on **QAT** *Quantization Aware Training* for all layers in a model.
   - The member asked for recommendations on good papers about the topic.


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/)** (1 messages): 

iron_bound: https://core-math.gitlabpages.inria.fr/
  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1360333515027120289)** (8 messages🔥): 

> `AMD GPUs, Cloud providers, Profiling, vast.ai, shadeform` 


- ****AMD GPU Cloud Quest Begins****: Members are seeking recommendations for cloud providers offering **AMD GPUs** that allow **profiling** capabilities, with a nod to [vast.ai](https://vast.ai) but noted lack of AMD support.
   - One member mentioned that *access to hardware counters for performance profiling* is often disabled by most cloud vendors for **Nvidia GPUs**, except for a few, like **lightning.ai**.
- ****Vast.ai Profiling Blocked****: A member mentioned that [vast.ai](https://vast.ai), a recommended option, **does not allow profiling**.
   - However, another member pointed to a [previous message](https://discord.com/channels/1189498204333543425/1349310333520711721/1349464109183139902) suggesting it might be possible to set up profiling there, although they hadn't tried it themselves.
- ****Shadeform Question Arises****: A member inquired whether anyone has experience with **shadeform**.
   - Another member expressed interest, stating, *Good question. Let me find out*.


  

---


### **GPU MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/1360423483934249071)** (7 messages): 

> `Profiling Metal Kernels, Naive vs. Coalesced Matrix Multiplication, Memory Usage Differences, Unified Memory and Paging on M-Series Chips` 


- **Coalesced Kernel Cuts Memory in Half**: A member found that a global memory coalesced matrix multiplication implementation in Metal uses half the memory of a naive version, despite only being slightly faster, testing with [this CUDA MMM implementation](https://siboehm.com/articles/22/CUDA-MMM) as a reference.
   - The images attached demonstrated profiling results for Metal kernels which showed a marked reduction in memory use on coalesced versions.
- **Paging Suspected in Memory Discrepancy**: One explanation posited that the OS pulls data as pages, and non-coalesced access leads to inefficient page usage where only a small portion of the pulled data is actually utilized.
   - Others noted that **M-series chips have unified memory**, which should negate paging between CPU and GPU, although data movement from unified memory to shared memory could potentially still involve paging.
- **M3 Pro Chip in Focus**: The original poster clarified that they are using an **M3 Pro chip**, suggesting that the unified memory architecture is relevant to the observed memory behavior.
   - The member who suggested a memory discrepancy, misunderstanding that they were experimenting on an M3 chip.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1360457071350251670)** (11 messages🔥): 

> `OptiLLM inference proxy, Fast Prefix Sum, Thread Coarsening, SwissKnife webgpu graphrag` 


- **OptiLLM Optimizes Accuracy and Performance**: [OptiLLM](https://github.com/codelion/optillm) is an **OpenAI API** compatible optimizing inference proxy which implements several state-of-the-art techniques that can improve the accuracy and performance of LLMs.
- **Prefix Sum is Really Fast**: A new blog post shows how to archive high performance for blockwise scan operation, with the fastest kernel reaching **93% GPU utilisation**.
   - The author recommends checking out [Juan Gómez Luna's lecture](https://www.youtube.com/watch?v=SG0gvcbf2eo) to understand the basics of prefix sum, and links to the [blogpost](https://veitner.bearblog.dev/making-prefix-sum-really-fast/) and [code](https://github.com/simveit/effective_scan).
- **Thread Coarsening Boosts Prefix Sum**: A member mentioned that the technique used for the last kernel is called *thread coarsening*, as described in the **PMPP book**.
   - The book chapter is available [here](https://www.sciencedirect.com/science/article/abs/pii/B9780323912310000227), written by professor Luna, covering a double buffering technique.
- **SwissKnife: WebGPU GraphRAG is Coming Soon**: **SwissKnife** (claude code (apis) + webgpu + graphrag + graphofthoughts) has a request for comment before major development begins.
   - A link to the repo and architecture docs are available [here](https://github.com/endomorphosis/swissknife/tree/blueprints) and [here](https://github.com/endomorphosis/swissknife/tree/blueprints/docs/architecture).


  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1361306646298755233)** (2 messages): 

> `LLM for Kernel Code Generation, RL for Kernel Optimization` 


- **Pursuing LLM for Kernel Code Generation**: A member is exploring a simple **LLM** for **kernel code generation** trained on the next token prediction.
   - The model would be aligned with **RL** using a *simulator* or real hardware to compile, run, and evaluate kernel performance across different hardware configurations.
- **Aligning LLMs with RL for Kernel Optimization**: The member envisions aligning the **LLM** with **Reinforcement Learning (RL)**.
   - This alignment would utilize a *simulator* or real hardware to compile, run, and assess kernel performance across diverse hardware setups, aiming for optimized kernel code generation.


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1360380897198870841)** (7 messages): 

> `vectoradd, grayscale, Modal runners` 


- **Vectoradd benchmark races ahead**: Multiple submissions to the `vectoradd` leaderboard succeeded using **Modal runners** on various GPUs including **L4, A100, and H100**.
- **Grayscale benchmark gets Modal boost**: A benchmark submission to the `grayscale` leaderboard succeeded using **Modal runners** on **H100** GPUs.


  

---


### **GPU MODE ▷ #[ppc](https://discord.com/channels/1189498204333543425/1343373905783554048/)** (1 messages): 

eriks.0595: <@349565795711451146> we've updated the grader, can you let me know if this is fixed?
  

---


### **GPU MODE ▷ #[feature-requests-and-bugs](https://discord.com/channels/1189498204333543425/1343759913431728179/1360399324646871050)** (6 messages): 

> `Python vs CUDA Submissions, Auto-Wrapping CUDA Files, Profiling Tools, QoL Changes` 


- **CUDA Submissions Auto-Wrapping Incoming Soon?**: The team discussed automatically wrapping **CUDA (.cu) files** in a Python file with **load_inline** for easier submissions.
   - While it might not make pre-launch, *it is planned*, along with other *QoL changes*.
- **Profiling Tools are on the Horizon**: Members expressed the need for **profiling tools** alongside the ability to auto-wrap CUDA submissions.
   - Profiling tools are *definitely among [one member's] personal list of features* to have.


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1360330720895307838)** (8 messages🔥): 

> `Challenge registration, Discord ID submission, Confirmation email after registration` 


- **Challenge Registration Confusion Cleared Up**: It was clarified that the challenge could involve old work, but registration alone should suffice initially; a mailing list with updates is expected later.
   - A member said that *registering is enough, there will probably be some mailing list later with some updates but nothing immediate iirc.*
- **Duplicate Discord ID Submissions**: A member asked about the issue of submitting the registration form twice due to providing an incorrect Discord ID initially.
   - Another member suggested, *just submit using the correct one, should be okay*.
- **Confirmation Email Delay**: A member inquired about the absence of a confirmation email post-registration.
   - Another member responded that **it is normal** and the follow-up should be expected soon, possibly today or tomorrow.


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1360601880841031710)** (67 messages🔥🔥): 

> `Nomic Embeddings, GPT4All Max Tokens, HuggingFace story models, Chat Templates, Context Length` 


- **Nomic Embeddings enable automatic website linking**: A member is successfully using **Nomic embeddings** to automatically link website pages, reducing manual work to a few percent via the [semantical-website-links blogpost](https://huggingface.co/blog/JLouisBiz/semantical-website-links).
   - The member is seeking methods to automatically identify and link key terms within the text that correspond highly to embeddings, potentially creating an interconnected network of documents that update as the knowledge base evolves, explained in [this youtube video](https://www.youtube.com/watch?v=xk2VGnLYAkA).
- **GPT4All and Max Token Troubleshoot**: A member was attempting to generate a play of at least 30 minutes in length using various models within **GPT4All**, but was running into a **response length cap**.
   - Members suggested increasing the **Max Tokens** setting and breaking the story into sections, but the original member stated that the cap still exists and they are searching for models that can output longer responses.
- **HuggingFace "story" models may help**: The member was successful finding models on **HuggingFace** using the keyword 'story' that were able to accomplish generating a longer response.
   - Another member cautioned that they found many of those models were proprietary and not free software.
- **Chat Template Locations Revealed**: A member inquired about finding the **chat templates** for various models like Llama3.2, Llama3.1, Aya-23, and KafkaLM-8x7b-German-V0.1.
   - Another member directed them to the model authors' releases, typically on their website, GitHub, or **Hugging Face**, and specifically to check the `tokenizer_config.json` file for the `chat_template` entry.
- **Context Length affects response quality**: A member noted that most models are trained on something between **2048 and 8192 tokens context length**, and while techniques like RoPE and Yarn can extend this, the quality of responses degrades drastically beyond the original range.
   - Response length depends on the training dataset and finetuning, but can be slightly adjusted via prompting such as telling the model to make it *VERY VERY LONG*.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1360335021080055858)** (16 messages🔥): 

> `Mojo ownership vs Rust, Origins vs Lifetimes, VSCode extension issues, Mojmelo module, closures` 


- **Mojo's `owned` Parameter Decoded**: In Mojo, the **`owned`** keyword copies a copyable element into a function, which is then deleted when it goes out of scope; **`mut`** takes a mutable reference, but the transfer operator is used to move the value completely, per the [docs](https://docs.modular.com/mojo/manual/values/ownership#transfer-arguments-owned-and).
   - One user, familiar with Rust, sought a guide to Mojo's ownership system, as they understood mutable borrows, immutable references, and move values from Rust, but found Mojo's **origins** concept new.
- **`Origins` Renamed as `Lifetime`**: The term **`Origin`** in Mojo was renamed from **`Lifetime`**, as mentioned in the thread, potentially aiding understanding for those familiar with Rust's lifetime concepts.
   - It was clarified that a reference `ref [a]` lives as long as variable `a`, or conversely, makes variable `A` live as long as the reference.
- **Mojo extends Lifetimes**: Mojo's lifetimes differ from Rust's, as Mojo extends the lifetime of values to match any reference holding onto them; instead, the origin of every reference must be tracked to determine value extensions and freedom, contrasting Rust's scope-based lifetime tracking.
   - One member says, *Understanding how mojo keeps track of closure origins is probably the best way to understand mojo's model of lifetimes*.
- **Mojmelo Extension issues**: A user encountered issues with the Mojo VSCode extension, reporting errors of missing **`mojmelo`** modules despite successful manual installation and setup via magic add.
   - It was suggested that the VSCode extension might use its own Mojo installation, preventing it from detecting modules installed in the project's environment; a workaround involves manually configuring the extension to use local module repositories for intellisense.
- **Mojo closures**: The discussion notes an analogy between Mojo's origin tracking and closures, implying that understanding how Mojo manages closure origins is key to grasping its memory management model.
   - There is currently little documentation on this concept.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1360601085827481722)** (32 messages🔥): 

> `PythonObject Literal, MLIR in Mojo, Mojo Proposals, Negative Bounds` 


- **Nested ListLiterals vex Mojo**: Mojo doesn't handle nested `ListLiteral` yet, resulting in a `constraint failed: cannot convert list element to python object` error, but workarounds include [using `Python.list()` and appending elements](https://github.com/modular/max/blob/f5adc052eb447297ac011a5e97063a62e55cd014/mojo/stdlib/src/python/python_object.mojo#L443-L445) or [using nested calls to `PythonObject`](https://discord.com/channels/1017080050601173042/1165766816034523176).
   - Chris Lattner mentioned that the old prettier syntax is broken but they will get back to it with a few more language features.
- **MLIR example surfaces in Mojo discord**: A member asked about old documentation on leveraging **MLIR** in Mojo, and another member provided [a link to an example](https://github.com/modular/max/blob/570d4a0af82d547264a2bc46f6f0abeba59f3d66/examples/BoolMLIR.ipynb) noting that the syntax has changed since then.
   - They mentioned that *People here might still be able to help*.
- **Mojo PEPs coming in hot 🔥**: Inspired by **Python's PEPs**, a member suggested a similar system for Mojo to track changes, and another member pointed to [Mojo's existing proposal system](https://github.com/modular/max/tree/main/mojo/proposals).
   - The discussion shows the community's interest in a structured way to manage and communicate language evolution.
- **Negative Bounds invert Named Sets**: **Negative bounds** are a way to invert a named set, often used with **marker traits** to define the inverse of a set of types.
   - For example, `!Send` would represent a thread-local variable or a non-atomically refcounted smart pointer, indicating it's not safe to move between threads.


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1360648500034928782)** (5 messages): 

> `Llama4 Deep Research, Equity Research Agent, GPT-4.1 API, Agent Benchmarks` 


- **Llama4 Powers Deep Research Project**: A fully open-source deep research solution built with **Llama4**, @GroqInc, Linkup, @FastAPI, @Redisinc, @Gradio, and @llama_index by Clelia Bertelli is now available, with a simple workflow outlined [here](https://t.co/o46BJWIxM7).
- **Craft Equity Research Agent from Scratch**: A new tutorial demonstrates building an end-to-end agentic workflow for ingesting unstructured earnings reports from **Tesla/Ford**, and extracting financial metrics, detailed [here](https://t.co/2hdpLas3vH).
- **GPT-4.1 API Lands with Day 0 Support**: **OpenAI** announced the availability of **GPT-4.1** in the **API**, supported from day 0 via `pip install -U llama-index-llms-openai`, further details [here](https://t.co/JPEX3KAoWS).
- **GPT-4.1 Benchmarks Show Improvement**: **GPT-4.1** shows a substantial ~**10%** improvement against **4o** by itself, and a ~**2%** improvement on our already-excellent agentic approach.
   - For more details on their work, reach out via [this link](https://t.co/E7KcaQ48Ek).


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1360563181180817439)** (31 messages🔥): 

> `LlamaParse vs SimpleDirectoryReader, Files in Index vs External File Sources, Open Source LLMs for Agent Workflow, Django Application hangs when calling LlamaParser with Celery, Voice Agents Support` 


- ****LlamaParse** Leaps in Document Dexterity**: **LlamaParse** handles images, tables, and visual elements like charts, offering higher quality parsing compared to basic readers like **SimpleDirectoryReader**.
   - The main advantage of using LlamaParse over SimpleDirectoryReader is that *it's the quality* of the outputted parsed documents.
- ****Index Files** vs. **External Data** Elucidated**: Files in the index determine the document count for vector index creation, whereas **External Data sources** encompass platforms like **Google Drive**, **Confluence**, and **Notion** for index building.
   - In other words, **files in index** are the documents you use to *create your index*, and **external data sources** help to create index over data stored in other places.
- ****Open Source LLMs** face agent angst**: While smaller open-source LLMs are deemed insufficient for agent workflows, larger models like **Llama3**, **Llama 3.1**, **Llama 3.2:3b**, or **Mistral** are recommended, often used with **Ollama**.
   - One member says they are *currently using llama3.2:3b* which seems to be working for them.
- ****Celery** Chokes on **LlamaParse** in Django**: A user reported their **Django** application hangs indefinitely when invoking **LlamaParse** via **Celery**, despite functioning correctly without **Celery**.
   - Despite the issues, no explicit errors are raised during this hanging state.
- ****Voice Agents** Venture Forth**: Basic support for voice agents can be achieved by integrating text-to-speech and speech-to-text modules at the input and output stages.
   - The integration of Google's Live API was also asked about in the context of voice agents but wasn't answered.


  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1360534860275650676)** (5 messages): 

> `.query has no history, LlamaParse Layout Agent Mode, Benchmarking AI evaluation models` 


- **.query chats have no history**: A member asked how to store chats for the **Query mode** without using Agents and was informed that `Char .query` has no history as it is **stateless**.
- **LlamaParse Layout Agent Mode guide**: A comprehensive guide on **Visual Citations** with **LlamaParse Layout Agent Mode** was shared [here](https://medium.com/ai-artistry/visual-citations-with-llamaparse-layout-agent-mode-a-comprehensive-guide-a623a5fb41fc).
- **AI Evaluation Models get benchmarked**: A paper [Benchmarking AI evaluation models](https://arxiv.org/abs/2503.21157v3) such as **LLM-as-a-judge**, **HHEM**, **Prometheus** across **6 RAG applications** was shared, noting that evaluation models work surprisingly well in practice.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1360438108365258933)** (8 messages🔥): 

> `NVIDIA Video Codec SDK, Direct Programming, Meeting #66 Topics, Index Validation PR` 


- **NVIDIA Releases Video Codec SDK**: NVIDIA released the [Video Codec SDK](https://developer.nvidia.com/downloads/designworks/video-codec-sdk/secure/13.0.19/video_codec_interface_13.0.19.zip) and accompanying [samples on GitHub](https://github.com/NVIDIA/video-sdk-samples).
   - A user cautioned against using AI for PR submissions without understanding the content, threatening to close them and ban repeat offenders.
- **TinyGrad Meeting #66 Agenda Revealed**: Meeting #66 is scheduled for Monday at 7am San Diego time (10pm HK time) and will cover several topics.
   - The topics include: company update, **chip!**, fast python, bert, mlperf, scheduler, driver, webgpu, retinanet, torch frontend multi gpu, cloud scale uuuvn stuff, and other bounties.
- **Index Validation PR Update**: A member who couldn't attend the meeting mentioned they saw a comment on the Index Validation PR and understood what was required.
   - They expect to have it ready by tomorrow and another member confirmed it was added to the meeting agenda for discussion.


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1360431752421707895)** (26 messages🔥): 

> `clang flags, tinygrad notes, debugging NaNs, small bounty` 


- **Clang flag `-fno-ident` silences debug output**: A member noticed that extra sections (`.comment` and `.note.GNU-stack`) were being added to images, polluting `DEBUG=7` output, and suggested using the `-fno-ident` [clang flag](https://xl0.github.io/tinygrad-notes/misc_1.html) to prevent this.
- **Beginner seeks first tinygrad project**: A new member introduced themselves and asked for suggestions on a mini-project to get hands-on with **tinygrad**.
   - A member recommended picking a [small bounty](https://xl0.github.io/tinygrad-notes/misc_1.html) and linked to helpful resources: [tinygrad-notes](https://xl0.github.io/tinygrad-notes) and [mesozoic-egg's tinygrad-notes](https://mesozoic-egg.github.io/tinygrad-notes).
- **softmax debugging**: A member inquired about debugging **NaNs** within a model, suspecting a `softmax()` issue, and noting that printing mid-`__call__` was causing optimizer issues.
   - George Hotz responded that printing shouldn't break things and suggested posting an issue.


  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1361323175308431440)** (16 messages🔥): 

> `Custom TorchTune model in vLLM, HF model, Custom model architecture in vLLM, Torchtune generate script` 


- **TorchTune Models Seek vLLM Integration**: A member inquired about using a custom **TorchTune model** in **vLLM**.
   - Another member suggested that inferencing a **TorchTune** finetuned model with **vLLM** should be straightforward, similar to any model from **HF**.
- **TorchTune example shared!**: A member shared a [link to a tutorial](https://pytorch.org/torchtune/main/tutorials/e2e_flow.html#use-with-vllm) to help with the request.
   - A member asked if the tutorial will work for models not defined on **HF**.
- **Custom Models: A New Frontier for vLLM**: A member confirmed they defined a custom network, finetuned it in **TorchTune**, converted it to **HF**, and now want to use **vLLM** for inference, but received an error that the "custom model" is not defined in **HF**.
   - Another member clarified that for custom networks, defining the model in **vLLM** is necessary, and pointed to [vLLM documentation](https://docs.vllm.ai/en/latest/contributing/model/index.html).
- **TorchTune generate script as an alternative to vLLM**: A member suggested to use **Torchtune's generate script**, which is slower, but could work with the custom model.
   - They recommended using **generate_v2** ([link to the recipe](https://github.com/pytorch/torchtune/tree/main/recipes/dev)) and asked to report issues.


  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1360335469518389318)** (8 messages🔥): 

> `bitsandbytes installation errors, macOS installation issues, unit tests on macOS, FSDP import error, platform specific requirements` 


- **`bitsandbytes` gives Mac Users the Bits**: `pip install -e '.[dev]` fails on a mac` due to `bitsandbytes>=0.43.0` because it doesn't ship binaries for other platforms other than linux, but downgrading to `bitsandbytes>=0.42.0` can help.
   - Releases up to **0.42** were incorrectly tagged, but at least this makes it installable ([bitsandbytes issue 1378](https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1378#issuecomment-2383530180)).
- **pytest fails on collecting tests**: Running `pytest tests` fails on collecting tests with 59 errors due to an `ImportError: cannot import name 'FSDPModule' from 'torch.distributed.fsdp'`.
   - The traceback indicates an issue when importing `test_full_finetune_distributed.py` due to a missing **FSDPModule** from the `torch.distributed.fsdp`.
- **Install in a different way, recommends member**: A member pointed out that there are other ways to install `torchtune`, and that platform-specific requirements are not desired.
   - They believed this should also fix the unit tests issues on mac, as some fixes have been applied to unit tests on mac.


  

---


### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/1360332135097176266)** (2 messages): 

> `QLoRA, Quantization, Sub-4-Bit Quantization` 


- **QLoRA Quantization Queries**: A member inquired about literature on **QLoRA-style training** using quantization **below 4 bits**.
   - The inquiry specifically targeted methods and findings related to **sub-4-bit quantization** techniques in the context of **QLoRA**.
- **Seeking Sub-4-Bit QLoRA Literature**: A member inquired about available literature on **QLoRA-style training** utilizing quantization **below 4 bits**.


  

---


### **Torchtune ▷ #[rl](https://discord.com/channels/1216353675241590815/1360680363885854841/1361344317867950312)** (5 messages): 

> `Reward Function Design, Loss Function Variety, Inference Provider Flexibility, Resource Allocation, TRL Success Logging` 


- **Reward Functions: To Shape or Not To Shape?**: The team is planning to support different **reward functions**, though the specific implementation details are still under discussion.
   - A member asked about locating the reward computing in a "weird way", followed up by *collecting a list of important ones*.
- **Loss Functions: Experimentation Station**: The team is currently experimenting with **different loss functions**, but aims to avoid excessive recipe proliferation by potentially adopting a protocol similar to DPO losses.
   - The goal is to strike a balance between supporting important losses and preventing overgeneralization during this experimental phase.
- **SGLang supports DeepSeek optimizations like expert parallel, whereas vLLM does not support such**: A request was made to support **various inference providers** via an inference server, citing flexibility and support for specific models or features like DeepSeek optimizations in SGLang that are absent in vLLM.
   - The initial plan is to focus on **vLLM** to reduce complexity and optimize around it.
- **Resource Allocation: A100s in the House!**: The recipe's resource allocation is acknowledged to have **hardcoded test parameters** and is undergoing cleanup; current testing is primarily on **A100s**.
   - The team clarified that the recipe design is under heavy development, with an initial focus on algorithm and infrastructure scalability before library compatibility.
- **NonTensorStack: Unveiling the Mystery**: **NonTensorStack** clarifies when a list passed to a `TensorDict` should be treated as a batch index (e.g., `a[0] = list[0]`) versus a constant shared across tensors.
   - More details are available in the [PyTorch documentation](https://pytorch.org/tensordict/main/overview.html#stacked-non-tensor-data).


  

---


### **Cohere ▷ #[「💬」general](https://discord.com/channels/954421988141711382/954421988783444043/1360496419995127891)** (9 messages🔥): 

> `Coral Chat in Firefox, LLM Token Generation Issues` 


- ****Coral Chat** becomes Firefox sidebar!**: Members can now use **Coral Chat** as a chatbot in the Firefox sidebar by setting `browser.ml.chat.provider` to [https://coral.cohere.com/](https://coral.cohere.com/).
   - A user shared an [Imgur link](https://imgur.com/a/6zcTV8z) showcasing the integration.
- **LLMs have next-token issues**: A user shared a [YouTube video](https://youtu.be/VRqSJfdwbF4) and joked about how other **LLMs** might have similar problems when generating the next token in a given context.
   - Another user responded with an *"eyes"* emoji.


  

---


### **Cohere ▷ #[「🔌」api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1361252377642143807)** (4 messages): 

> `Cohere Chat API, Java Demo Code, command-a-03-2025 model` 


- **Cohere Chat API Java Example Shared**: A member shared a Java example demonstrating the use of the Cohere Chat API, focusing on the `runInteractiveDemo()` method.
   - The example includes a chat loop that interacts with the **command-a-03-2025** model, showing how to send messages and maintain chat history.
- **Interactive Chat Demo Implementation**: The `runInteractiveDemo()` method allows users to chat with Cohere AI, providing example prompts and handling API errors.
   - The code captures user input via the console, sends it to the Cohere API, and prints the response, updating the chat history with each interaction.


  

---


### **Cohere ▷ #[「💡」projects](https://discord.com/channels/954421988141711382/1218409701339828245/1361213775965061130)** (1 messages): 

> `Diofanti.org, Aya model, Government spending transparency` 


- **Diofanti.org empowers Greek transparency**: A member introduced [Diofanti.org](https://chat.diofanti.org), an **open-data platform** monitoring government spending and operations in Greece.
   - The platform transforms raw public data into actionable insights, empowering citizens, journalists, and policymakers with tools for **transparency and accountability**.
- **Aya becomes goto model for Diofanti chatbot**: The creator has been experimenting with a chatbot on top of it and **Aya** is the goto model for it.
   - Members were invited to reach out to support the project.


  

---


### **Cohere ▷ #[「🤝」introductions](https://discord.com/channels/954421988141711382/1346635816629178410/1360886630633963640)** (3 messages): 

> `LUWA.app, AI for Science Community` 


- **LUWA App to go live on April 25, 2025**: A member is launching **LUWA.app**, a search directory for **AI powered apps** that will be live on **April 25, 2025**.
   - The creator is keen to learn about **Cohere** and its **LLM models** to potentially reduce costs or improve app performance.
- **Encode: AI for Science Community Seeking Talent**: A member from the University of Toronto is building an **AI for Science** community called **Encode** ([https://encode.pillar.vc/](https://encode.pillar.vc/)).
   - They're looking for people with great **AI skills** to tackle significant science problems with notable **PIs** (Principal Investigators); interested individuals are encouraged to DM them.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[hackathon-announcements](https://discord.com/channels/1280234300012494859/1280236929379602493/1361402820376395787)** (1 messages): 

> `Lambda, HuggingFace, Groq, Mistral AI, Google AI Studio` 


- ****Lambda Labs** serves up **Serverless API Credits****: **Lambda** is offering **$100** of serverless API credits for [Inference](https://lambdalabs.com/inference) to every individual participant, application [here](https://forms.gle/UtVhmPS3mitS8Vxu7).
- ****HF, Groq, Mistral** serve up **API/Compute Credits****: Our sponsors **Lambda, HuggingFace, Groq**, and **Mistral AI** are offering API/compute credits to select teams, more details [here](https://rdi.berkeley.edu/agentx/#resources) and application [here](https://forms.gle/ZDYxwM4aFSRCcrfp7).
- ****Google** grants access to **Gemini API****: **Google** is granting access to **Gemini API** and **Google AI Studio** free of charge to ALL participants.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-announcements](https://discord.com/channels/1280234300012494859/1280369709623283732/1361420471375237364)** (1 messages): 

> `Sean Welleck, LeanHammer, AI proof development, formal reasoning` 


- **Sean Welleck Presents: Bridging Informal and Formal Mathematical Reasoning**: Sean Welleck, an Assistant Professor at Carnegie Mellon University, will present a lecture today at **4pm PDT** on "Bridging Informal and Formal Mathematical Reasoning".
   - The lecture will cover **AI-powered tools** that support proof development, from automating low-level steps with **LeanHammer**, to sketching proof ideas and incorporating informal insights; watch the livestream [here](https://www.youtube.com/live/Gy5Nm17l9oo).
- **Welleck's Background: ML, Language, Logic, and Awards**: Sean Welleck leads the **Machine Learning, Language, and Logic (L3) Lab** at Carnegie Mellon University.
   - His research focuses on large language models, reasoning and agents, and AI for mathematics and code, with accolades including a **NeurIPS 2021 Outstanding Paper Award** and two **NVIDIA AI Pioneering Research Awards**.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-lecture-discussion](https://discord.com/channels/1280234300012494859/1282734248112947210/1361423950504005895)** (2 messages): 

> `Lecture Schedule, Email Notifications` 


- **Lecture Still On Despite Email Delay**: A member inquired whether there was a lecture today because they had not received the usual email notification.
   - Another member confirmed that there was a lecture and the email was sent a little late.
- **Email Notifications Delayed**: A member reported not receiving the usual email notification for today's lecture.
   - A response indicated that the email was sent, but with a slight delay.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1361304340773474375)** (2 messages): 

> `AI Agent Developer, DSPy Modules` 


- **AI Agent Developer Seeks New Gig**: An experienced **AI Agent Developer** announced their availability for new projects or full-time opportunities, specializing in building **autonomous agents** powered by GPT-4, LangChain, AutoGen, CrewAI, and other cutting-edge tools.
- **DSPy Module Evaluation Metric Proposed**: A member inquired about the appetite for a new metric to evaluate **DSPy modules**, referencing [this paper](https://arxiv.org/abs/2405.10516).


  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1361395885019369525)** (1 messages): 

> `MCP, AWS, Model Context Protocol, Simba Khadder` 


- **Workshop Builds Production-Grade MCP Server on AWS**: A workshop on **April 17th at 8 AM PT** will focus on building and deploying a production-grade **Model Context Protocol (MCP)** server on **AWS**.
   - Participants will learn to set up, configure, and deploy an **MCP** server, gaining insights into streamlining machine learning workflows; sign up at [https://buff.ly/R7czfKK](https://buff.ly/R7czfKK).
- **MCP Emerging Standard Improves ML Contexts**: **MCP** is highlighted as an emerging standard designed to improve how machine learning contexts are defined, shared, and managed across projects and teams.
   - The workshop aims to provide practical insights into **MCP’s** capabilities, benefiting Data Engineers, Data Scientists, Machine Learning Engineers, and AI/ML Enthusiasts.


  

---


### **MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/)** (1 messages): 

basit5750: I already have it dm me for Source Code
  

---


### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1361393735400689664)** (2 messages): 

> `GPT-4.1, Free Usage, Discounted Rate, New Default Model, Limited-Time Opportunity` 


- ****GPT-4.1** Launches on Windsurf**: **GPT-4.1** is now available on Windsurf, marked by the <:windsurf:1306309317011570699> emoji, across [Twitter/X](https://x.com/windsurf_ai/status/1911833698825286142), [Bluesky](https://bsky.app/profile/windsurfai.bsky.social/post/3lms3je7p2s2d), and [Threads](https://www.threads.net/@windsurf_ai/post/DIbwqQQslzI).
   - It's also accompanied by a [promotional video](https://youtu.be/OBTpSQ8OVq4) and a TikTok post ([latest vid](https://www.tiktok.com/@windsurf/video/7493220207041793310)).
- **Windsurf Gives Away **Free Unlimited GPT-4.1****: Windsurf is offering **free unlimited GPT-4.1** usage on all plans for **one week only** (April 14-21).
   - After April 21, **GPT-4.1** will be available at a special discounted rate of just **0.25 credits per use**.
- ****GPT-4.1** Set as New Default**: New users will get **GPT-4.1** as their default model, and existing users can easily switch through the model selector.
   - Windsurfers: *"Don't miss this limited-time opportunity!"*


  

---


### **Gorilla LLM (Berkeley Function Calling) ▷ #[discussion](https://discord.com/channels/1111172801899012102/1111353033352294440/1360997096622129262)** (1 messages): 

> `Multi-turn composite column removal, Dataset composition discrepancy` 


- **Multi-Turn Column Gets the Ax**: The multi-turn composite column was [removed](https://github.com/ShishirPatil/gorilla/pull/766) from the dataset, but the reason for its removal is not explicitly stated in the provided context.
   - Although the column is hidden, it is still mentioned in the "Newly Introduced Categories" section of the [BlogPost](https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html) and has a weight of 200 points out of 1000 for multi-turn tasks.
- **Dataset Composition Suffers from a Glitch**: There is a discrepancy in the dataset composition, as the multi-turn composite column is absent from the table/diagram illustrating the dataset's structure.
   - It is unclear whether the removal of the column was temporary or if it should also be removed from the [blog post](https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html) section where it is currently mentioned.


  

---


{% else %}


> The full channel by channel breakdowns have been truncated for email. 
> 
> If you want the full breakdown, please visit the web version of this email: [{{ email.subject }}]({{ email_url }})!
>
> If you enjoyed AInews, please [share with a friend](https://buttondown.email/ainews)! Thanks in advance!

{% endif %}
