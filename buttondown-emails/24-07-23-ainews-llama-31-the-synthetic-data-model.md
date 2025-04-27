---
id: 829b011e-ad99-4da3-87a9-13e0bf8ccffd
title: 'Llama 3.1: The Synthetic Data Model'
date: '2024-07-24T00:13:31.329222Z'
original_slug: ainews-llama-31-the-synthetic-data-model
description: >-
  **Meta AI** released **Llama 3.1**, including a **405B parameter model** that
  triggers regulatory frameworks like the **EU AI Act** and **SB 1047**. The
  model incorporates extensive **synthetic data** techniques for **code
  fine-tuning**, **math fine-tuning**, **multilingual training**, **long context
  handling**, and **tool use** with integrations for **Brave Search**, **Wolfram
  Alpha**, and a **Python interpreter** supporting advanced function calling.
  The training also leveraged **RLHF** with **DPO preference data** from **Llama
  2** generations. The launch was widely supported by inference providers such
  as **Groq** and **Fireworks**, showcasing high throughput and competitive
  pricing. The license update explicitly allows synthetic data generation use.
  This marks a significant step forward in open frontier-class LLMs and
  cost-efficiency improvements since March.
companies:
  - meta-ai-fair
  - groq
  - fireworks
models:
  - llama-3-1
  - llama-3
topics:
  - synthetic-data
  - fine-tuning
  - multilinguality
  - long-context
  - tool-use
  - reinforcement-learning
  - rlhf
  - function-calling
  - inference
  - model-licensing
  - cost-efficiency
people:
  - bindureddy
  - swyx
  - thomas
---


<!-- buttondown-editor-mode: plaintext -->**Synthetic Data is all you need.**

> AI News for 7/22/2024-7/23/2024. We checked 7 subreddits, [**384** Twitters](https://twitter.com/i/lists/1585430245762441216) and **30** Discords (**474** channels, and **5128** messages) for you. Estimated reading time saved (at 200wpm): **473 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

**Llama 3.1 is here!** ([Site](
https://llama.meta.com/), [Video](https://x.com/aiatmeta/status/1815766327463907421?s=46&t=b7l37rB6wtbyAh6ah1NpZQ),[Paper](https://ai.meta.com/research/publications/the-llama-3-herd-of-models/), [Code](https://github.com/meta-llama/llama-models), [model](https://ai.meta.com/blog/meta-llama-3-1/), [Zuck](https://news.ycombinator.com/item?id=41046773), [Latent Space pod](https://x.com/latentspacepod/status/1815781241398104085)). Including the 405B model, which triggers both [the EU AI act](https://x.com/deanwball/status/1815826885663658445?s=46) and [SB 1047](https://x.com/martin_casado/status/1815865505204576389). The [full paper](https://ai.meta.com/research/publications/the-llama-3-herd-of-models/) has all the frontier model comparisons you want:

 ![image.png](https://assets.buttondown.email/images/12252d7e-4c23-4cf4-8afa-1ea166b0ff26.png?w=960&fit=max) 

We'll assume you read the headlines from [yesterday](https://buttondown.email/ainews/archive/ainews-llama-31-leaks/). It's not up on LMsys yet, but independent evals on [SEAL](https://x.com/summeryue0/status/1815776426999877643) and [Allen AI's ZeroEval](https://x.com/billyuchenlin/status/1815841947468353700?s=46) are promising (with some [disagreement](https://x.com/hrishioa/status/1815811349777375649?s=46&t=6FDPaNxZcbSsELal6Sv7Ug)). It was a well coordinated launch across ~every inference provider in the industry, including (of course) Groq showing a flashy demo inferencing at [750tok/s](https://x.com/JonathanRoss321/status/1815777714642858313). Inference pricing is also out with [Fireworks leading the pack](https://x.com/HamelHusain/status/1815852454027940135).

While it is well speculated that the 8B and 70B were ["offline distillations"](https://x.com/kalomaze/status/1815797116104556565) of the 405B, there are a good deal more synthetic data elements to Llama 3.1 than the expected. The paper explicitly calls out:

- **SFT for Code**: [3 approaches for synthetic data](https://x.com/swyx/status/1815771160841425113) for the 405B bootstrapping itself with code execution feedback, programming language translation, and docs backtranslation.
- **SFT for Math**: ![https://pbs.twimg.com/media/GTLqFD9aMAAwYUQ?format=png&name=900x900](https://pbs.twimg.com/media/GTLqFD9aMAAwYUQ?format=png&name=900x900)
- **SFT for Multilinguality**: "To collect higher quality human annotations in non-English languages, we train a multilingual expert by branching off the pre-training run and continuing to pre-train on a data mix that consists of 90% multilingual
tokens."
- **SFT for Long Context**: "It is largely impractical to get humans to annotate such examples due to the tedious and time-consuming nature of reading lengthy contexts, so **we predominantly rely on synthetic data to fill this gap.** We use earlier versions of Llama 3 to generate synthetic data based on the key long-context use-cases: (possibly multi-turn) question-answering, summarization for long documents, and reasoning over code repositories, and describe them in greater detail below"
- **SFT for Tool Use**: trained for Brave Search, Wolfram Alpha, and a Python Interpreter (a special new [`ipython`](https://llama.meta.com/docs/model-cards-and-prompt-formats/llama3_1) role) for **single, nested, parallel, and multiturn function calling**.
- **RLHF**: DPO preference data was used extensively on Llama 2 generations. As [Thomas says on the pod](https://www.latent.space/p/llama-3): *“Llama 3 post-training doesn't have any human written answers there basically… **It's just leveraging pure synthetic data from Llama 2.**”*

Last but not least, Llama 3.1 [received a license update](https://x.com/AIatMeta/status/1815766335219249513) explicitly allowing its use for synthetic data generation.

We finally have a [frontier-class open LLM](https://x.com/karpathy/status/1815842603377779140), and together it is worth noting how far ahead the industry has moved in [cost per intelligence since March](https://x.com/swyx/status/1815892458519289946/photo/1), and it will only get better from here.

 ![image.png](https://assets.buttondown.email/images/f446a928-ef16-41f6-b461-efde87ac6ecf.png?w=960&fit=max) 

---


{% if medium == 'web' %}


**Table of Contents**

[TOC] 

{% else %}

The **Table of Contents** and **Channel Summaries** have been moved to the web version of this email: [{{ email.subject }}]({{ email_url }})!

{% endif %}


---

# AI Twitter Recap

> all recaps done by Claude 3.5 Sonnet, best of 4 runs.

**Meta AI**

- **Llama 3.1 405B model**: [@bindureddy](https://twitter.com/bindureddy/status/1815443198459990098) noted that **Llama-3.1 405B benchmarks were leaked on Reddit, outperforming GPT-4o**. [@Teknium1](https://twitter.com/Teknium1/status/1815443354735571232) shared an image comparing **Llama-3.1 405/70/8b against GPT-4o, showing SOTA frontier models now available open source**. [@abacaj](https://twitter.com/abacaj/status/1815484377167466997) mentioned **Meta is training and releasing open weights models faster than OpenAI can release closed models**.
- **Llama 3 70B performance**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1815457648814194694) highlighted that the **70B model is matching GPT-4 levels while being 6x smaller**. This is the base model, not instruct tuned. [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1815458893759475974) noted the **70B model is encroaching on 405B's territory, and the utility of big models would be to distill from it**.
- **Open source model progress**: [@teortaxesTex](https://twitter.com/teortaxesTex/status/1815486069883752862) called it the **dawn of Old Man Strength open models**. [@abacaj](https://twitter.com/abacaj/status/1815484683452649687) mentioned **OpenAI models have not been improving significantly, so Meta models will catch up in open weights**.

**AI Assistants and Agents**

- **Omnipilot AI**: [@svpino](https://twitter.com/svpino/status/1815360679072653704) introduced **@OmnipilotAI, an AI application that can type anywhere you can and use full context of what's on your screen**. It works with every macOS application and uses **Claude Sonet 3.5, Gemini, and GPT-4o**. Examples include replying to emails, autocompleting terminal commands, finishing documents, and sending Slack messages.
- **Mixture of agents**: [@llama_index](https://twitter.com/llama_index/status/1815518744829169807) shared a video by @1littlecoder introducing **"mixture of agents" - using multiple local language models to potentially outperform single models**. It includes a tutorial on implementing it using **LlamaIndex and Ollama**, combining models like **Llama 3, Mistral, StableLM** in a layered architecture.
- **Planning for agents**: [@hwchase17](https://twitter.com/hwchase17/status/1815404685500821950) discussed the **future of planning for agents**. While model improvements will help, **good prompting and custom cognitive architectures will always be needed to adapt agents to specific tasks**.

**Benchmarks and Evaluations**

- **LLM-as-a-Judge**: [@cwolferesearch](https://twitter.com/cwolferesearch/status/1815405425866518846) provided an overview of **LLM-as-a-Judge, where a more powerful LLM evaluates the quality of another LLM's output**. Key takeaways include using a sufficiently capable judge model, prompt setup (pairwise vs pointwise), improving pointwise score stability, chain-of-thought prompting, temperature settings, and accounting for position bias.
- **Factual inconsistency detection**: [@sophiamyang](https://twitter.com/sophiamyang/status/1815389350013899106) shared a guide on **fine-tuning and evaluating a @MistralAI model to detect factual inconsistencies and hallucinations** in text summaries using @weights_biases. It's based on @eugeneyan's work and part of the Mistral Cookbook.
- **Complex question answering**: [@OfirPress](https://twitter.com/OfirPress/status/1815379379188293872) introduced a new benchmark to **evaluate AI assistants' ability to answer complex natural questions** like "Which restaurants near me have vegan and gluten-free entrées for under $25?" with the goal of leading to better assistants.

**Frameworks and Tools**

- **DSPy**: [@lateinteraction](https://twitter.com/lateinteraction/status/1815423177272824022) shared a paper finding that **DSPy optimizers alternating between optimizing weights and prompts can deliver up to 26% gains over just optimizing one**. [@lateinteraction](https://twitter.com/lateinteraction/status/1815423187418763308) noted **composable optimizers over modular NLP programs are the future**, and to compose BootstrapFewShot and BootstrapFinetune optimizers.
- **LangChain**: [@hwchase17](https://twitter.com/hwchase17/status/1815442482290978932) pointed to the **new LangChain Changelog to better communicate everything they're shipping**. [@LangChainAI](https://twitter.com/LangChainAI/status/1815439685117993349) highlighted **seamless LangSmith tracing in LangGraph.js with no additional configuration**, making it easier to use LangSmith's features to build agents.
- **EDA-GPT**: [@LangChainAI](https://twitter.com/LangChainAI/status/1815426831430123585) introduced **EDA-GPT, an open-source data analysis companion** that streamlines data exploration, visualization, and insights. It has a configurable UI and integrates with LangChain.

---

# AI Reddit Recap

## /r/LocalLlama Recap

**Theme 1. Running Large Language Models Locally**

- **If you have to ask how to run 405B locally** ([Score: 287, Comments: 122](https://reddit.com//r/LocalLLaMA/comments/1e9nybe/if_you_have_to_ask_how_to_run_405b_locally/)): The post addresses the **impossibility of running a 405 billion parameter model locally**. It bluntly states that if someone needs to ask how to do this, they simply cannot achieve it, implying the task is beyond the capabilities of typical consumer hardware.

- **Please share your LLaMA 3.1 405B experiences below for us GPU poor** ([Score: 52, Comments: 30](https://reddit.com//r/LocalLLaMA/comments/1e9t8n2/please_share_your_llama_31_405b_experiences_below/)): The post requests users to share their experiences running **LLaMA 3.1 405B** locally, specifically targeting those with limited GPU resources. While no specific experiences are provided in the post body, the title suggests interest in understanding how this large language model performs on consumer-grade hardware and the challenges faced by users with less powerful GPUs.

- **Ollama site “pro tips” I wish my idiot self had known about sooner:** ([Score: 72, Comments: 24](https://reddit.com//r/LocalLLaMA/comments/1e9hju5/ollama_site_pro_tips_i_wish_my_idiot_self_had/)): The post highlights several **"pro tips"** for using the **Ollama site** to download and run AI models. Key features include accessing **different quantizations** of models via the "Tags" link, a hidden **model type sorting feature** accessible through the search box, finding **max context window sizes** in the model table, and using the **top search box** to access a broader list of models including user-submitted ones. The author, who has been using Ollama for **6-8 months**, shares these insights to help others who might have overlooked these features.

**Theme 2. LLaMA 3.1 405B Model Release and Benchmarks**

- **[Azure Llama 3.1 benchmarks](https://github.com/Azure/azureml-assets/pull/3180/files)** ([Score: 349, Comments: 268](https://reddit.com//r/LocalLLaMA/comments/1e9hg7g/azure_llama_31_benchmarks/)): Microsoft released benchmark results for **Azure Llama 3.1**, showing improvements over previous versions. The model achieved a **94.4%** score on the **MMLU** benchmark, surpassing GPT-3.5 and approaching GPT-4's performance. Azure Llama 3.1 also demonstrated strong capabilities in **code generation** and **multi-turn conversations**, positioning it as a competitive option in the AI model landscape.

- **[Llama 3.1 405B, 70B, 8B Instruct Tuned Benchmarks](https://i.redd.it/62ov7fzck5ed1.jpeg)** ([Score: 137, Comments: 28](https://reddit.com//r/LocalLLaMA/comments/1e9sinx/llama_31_405b_70b_8b_instruct_tuned_benchmarks/)): Meta has released **LLaMA 3.1**, featuring models with **405 billion**, **70 billion**, and **8 billion** parameters, all of which are instruct-tuned. The **405B** model achieves state-of-the-art performance on various benchmarks, outperforming **GPT-4** on several tasks, while the **70B** model shows competitive results against **Claude 2** and **GPT-3.5**.

- **LLaMA 3.1 405B base model available for download** ([Score: 589, Comments: 314](https://reddit.com//r/LocalLLaMA/comments/1e98zrb/llama_31_405b_base_model_available_for_download/)): The **LLaMA 3.1 405B base model**, with a size of **764GiB** (~820GB), is now available for download. The model can be accessed through a [Hugging Face link](https://huggingface.co/cloud-district/miqu-2), a **magnet link**, or a **torrent file**, with credits attributed to a 4chan thread.
  - Users discussed running the **405B model**, with suggestions ranging from using **2x A100 GPUs** (160GB VRAM) with low quantization to renting servers with **TBs of RAM** on **Hetzner** for $200-250/month, potentially achieving **1-2 tokens per second** at Q8/Q4.
  - Humorous comments about running the model on a **Nintendo 64** or **downloading more VRAM** sparked discussions on hardware limitations. Users speculated it might take **5-10 years** before consumer-grade GPUs could handle such large models.
  - Some questioned the leak's authenticity, noting similarities to previous leaks like **Mistral medium (Miqu-1)**. Others debated whether it was an intentional "leak" by **Meta** for marketing purposes, given the timing before the official release.


**Theme 3. Distributed and Federated AI Inference**

- **LocalAI 2.19 is Out! P2P, auto-discovery, Federated instances and sharded model loading!** ([Score: 52, Comments: 7](https://reddit.com//r/LocalLLaMA/comments/1e9inr9/localai_219_is_out_p2p_autodiscovery_federated/)): LocalAI 2.19 introduces **federated instances** and **sharded model loading via P2P**, allowing users to combine **GPU and CPU power** across multiple nodes to run large models without expensive hardware. The release includes a new **P2P dashboard** for easy setup of federated instances, **Text-to-Speech integration** in binary releases, and improvements to the **WebUI**, **installer script**, and **llama-cpp backend** with support for **embeddings**.

- **Ollama has been updated to accommodate Mistral NeMo and a proper download is now available** ([Score: 63, Comments: 13](https://reddit.com//r/LocalLLaMA/comments/1e9wsv6/ollama_has_been_updated_to_accommodate_mistral/)): **Ollama** has been updated to include support for the **Mistral NeMo** model, now available for download. The user reports that **NeMo** performs faster and better than **Lama 3 8b** and **Gemma 2 9b** models on a **4060 Ti GPU with 16GB VRAM**, noting it as a significant advancement in local AI models shortly after Gemma's release.
  - Users praised **Mistral NeMo 12b** for its performance, with one noting it "**NAILED**" a **48k context** test and showed fluency in French. However, its usefulness may be short-lived with the upcoming release of **Llama 3.1 8b**.
  - Some users expressed excitement about downloading the model, while others found it disappointing compared to **tiger-gemma2**, particularly in following instructions during multi-turn conversations.
  - The timing of **Mistral NeMo's** release was described as "very sad" for the developers, coming shortly after other significant model releases.


**Theme 4. New AI Model Releases and Leaks**

- **Nvidia has released two new base models: Minitron 8B and 4B, pruned versions of Nemotron-4 15B** ([Score: 69, Comments: 3](https://reddit.com//r/LocalLLaMA/comments/1e9vaqy/nvidia_has_released_two_new_base_models_minitron/)): **Nvidia** has released **Minitron 8B and 4B**, pruned versions of their **Nemotron-4 15B** model, which require up to **40x fewer training tokens** and result in **1.8x compute cost savings** compared to training from scratch. These models show up to **16% improvement** in **MMLU scores** compared to training from scratch, perform comparably to models like **Mistral 7B** and **Llama-3 8B**, and are intended for research and development purposes only.
  - **Pruned models** are uncommon in the AI landscape, with **Minitron 8B and 4B** being notable exceptions. This rarity sparks interest among researchers and developers.
  - The concept of **pruning** is intuitively similar to **quantization**, though some users speculate that quantizing pruned models might negatively impact performance.
  - **AWQ** (Activation-aware Weight Quantization) is compared to pruning, with pruning potentially offering greater benefits by reducing overall model dimensionality rather than just compressing bit representations.
- **llama 3.1 download.sh commit** ([Score: 66, Comments: 18](https://reddit.com//r/LocalLLaMA/comments/1e9vjkc/llama_31_downloadsh_commit/)): A recent commit to the **Meta Llama GitHub repository** suggests that **LLaMA 3.1** may be nearing release. The commit, viewable at [https://github.com/meta-llama/llama/commit/12b676b909368581d39cebafae57226688d5676a](https://github.com/meta-llama/llama/commit/12b676b909368581d39cebafae57226688d5676a), includes a **download.sh** script, potentially indicating preparations for the model's distribution.
  - The commit reveals **405B models** in both **Base and Instruct** versions, with variants labeled **mp16**, **mp8**, and **fp8**. Users speculate that "mp" likely stands for **mixed precision**, suggesting quantization-aware training for packed mixed precision models.
  - Discussion around the **fb8** label in the Instruct model concludes it's likely a typo for **fp8**, supported by evidence in the file. Users express excitement about the potential to analyze weight precisions for better low-bit quantization.
  - The commit author, **samuelselvan**, previously uploaded a **LLaMA 3.1 model** to Hugging Face that was considered suspicious. Users are enthusiastic about Meta directly releasing quantized versions of the model.


- **[Llama 3 405b leaked on 4chan? Excited for it ! Just one more day to go !!](https://www.reddit.com/gallery/1e99uaa)** ([Score: 210, Comments: 38](https://reddit.com//r/LocalLLaMA/comments/1e99uaa/llama_3_405b_leaked_on_4chan_excited_for_it_just/)): Reports of a **LLaMA 3.1 405B** model leak on **4chan** are circulating, but these claims are unverified and likely false. The purported leak is occurring just one day before an anticipated official announcement, raising skepticism about its authenticity. It's important to approach such leaks with caution and wait for official confirmation from **Meta** or other reliable sources.
  - A **HuggingFace repository** containing the model was reportedly visible **2 days ago**, allowing potential leakers access. Users expressed interest in **70B and 8B** versions of the model.
  - Some users are more interested in the **pure base model** without alignment or guardrails, rather than waiting for the official release. A separate thread on **/r/LocalLLaMA** discusses the alleged **405B base model** download.
  - Users are attempting to run the model, with one planning to **convert to 4-bit GGUF quantization** using a **7x24GB GPU setup**. Another user shared a [YouTube link](https://www.youtube.com/watch?v=LoUbZt9gtZs) of their efforts to run the model.

## All AI Reddit Recap

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity

**Theme 1. OpenAI's Universal Basic Income Experiment Results**

- [/r/singularity] **[The OpenResearch team releases the first result from their UBI study (OpenAI)](https://www.openresearchlab.org/findings)** ([Score: 280, Comments: 84](https://reddit.com//r/singularity/comments/1e9jppo/the_openresearch_team_releases_the_first_result/)): **OpenResearch**, a team at **OpenAI**, has released initial results from their **Universal Basic Income (UBI) study**. The study, conducted in **Kenyan villages**, found that a **$1,000 cash transfer** resulted in significant positive impacts, including a **$400 increase** in assets and a **40% reduction** in the likelihood of going hungry. These findings contribute to the growing body of evidence supporting the effectiveness of direct cash transfers in alleviating poverty.

- [/r/OpenAI] **[OpenAI founder Sam Altman secretly gave out $45 million to random people - as an experiment](https://www.forbes.com.au/news/innovation/openai-founder-sam-altman-gave-thousands-of-people-free-money/)** ([Score: 272, Comments: 75](https://reddit.com//r/OpenAI/comments/1e984lw/openai_founder_sam_altman_secretly_gave_out_45/)): **Sam Altman's $45 million UBI experiment revealed**: The **OpenAI founder** secretly distributed **$45 million** to **3,000 people** across two U.S. states as part of a **Universal Basic Income (UBI) experiment**. Participants received **$1,000 per month** for up to **five years**, with the study aiming to assess the impact of unconditional cash transfers on recipients' quality of life, time use, and financial health.
  - **3,000 participants** received either **$1,000 or $50 per month** for up to **five years**, with many Redditors expressing desire to join the experiment. The study targeted individuals aged **21-40** with household incomes below **300% of the federal poverty level** across urban, suburban, and rural areas in **Texas and Illinois**.
  - Some users criticized the experiment as a PR move by **tech billionaires** to alleviate concerns about **AI-driven job loss**, while others argued that private UBI experiments are necessary given slow government action on the issue.
  - Discussions emerged about the future of employment, with some predicting a sudden spike in unemployment due to AI advancements, potentially leading to widespread **UBI implementation** when traditional jobs become scarce across various sectors.

**Theme 4. AI Researcher Predictions on AGI Timeline**

- [/r/singularity] **[Former OpenAI researcher predictions](https://i.redd.it/4fmq8fb6e2ed1.png)** ([Score: 243, Comments: 151](https://reddit.com//r/singularity/comments/1e9d5pb/former_openai_researcher_predictions/)): **Former OpenAI researcher predicts AGI timeline**: Paul Christiano, a former OpenAI researcher, estimates a **20-30% chance of AGI by 2030** and a **60-70% chance by 2040**. He believes that current AI systems are still far from AGI, but rapid progress in areas like reasoning and planning could lead to significant breakthroughs in the coming years.

- [/r/singularity] **["most of the staff at the secretive top labs are seriously planning their lives around the existence of digital gods in 2027"](https://twitter.com/jam3scampbell/status/1815311644853256315)** ([Score: 579, Comments: 450](https://reddit.com//r/singularity/comments/1e9flnw/most_of_the_staff_at_the_secretive_top_labs_are/)): **AI researchers anticipate digital deities**: According to the post, **most staff** at **secretive top AI labs** are reportedly **planning their lives** around the expected emergence of **digital gods by 2027**. While no specific sources or evidence are provided, the claim suggests a significant shift in the mindset of AI researchers regarding the potential capabilities and impact of future AI systems.

- [/r/singularity] **[Nick Bostrom says shortly after AI can do all the things the human brain can do, it will learn to do them much better and faster, and human intelligence will become obsolete](https://v.redd.it/q9qjbqh707ed1)** ([Score: 323, Comments: 258](https://reddit.com//r/singularity/comments/1e9yhzx/nick_bostrom_says_shortly_after_ai_can_do_all_the/)): **Nick Bostrom** warns of **AI surpassing human intelligence** in a rapid and transformative manner. He predicts that once AI can match human brain capabilities, it will quickly outperform humans across all domains, rendering human intelligence obsolete. This accelerated advancement suggests a potential **intelligence explosion**, where AI's capabilities rapidly exceed those of humans, leading to significant societal and existential implications.
  - **Nick Bostrom's** warning sparked debate, with some calling it "**Captain obvious**" due to AI's ability to connect to **100k GPUs**, while others defended the importance of his message given ongoing arguments about AI capabilities.
  - Discussions ranged from humorous memes to philosophical musings about a "**solved world**", with one user describing a hypothetical **2055** scenario of **AGI** and **ASI** leading to medical breakthroughs, full-dive VR, and simulated realities.
  - Some users expressed optimism about AI solving major problems like ocean degradation, while others cautioned about potential negative outcomes, such as population reduction scenarios or the challenges of implementing necessary changes due to resistance.


**Theme 5. New AI Training Infrastructure Developments**

- [/r/singularity] **[Elon says that today a model has started training on the new and most powerful AI cluster in the world](https://x.com/elonmusk/status/1815325410667749760)** ([Score: 239, Comments: 328](https://reddit.com//r/singularity/comments/1e9ahwl/elon_says_that_today_a_model_has_started_training/)): **Elon Musk announces groundbreaking AI development**: A new AI model has begun training on what Musk claims is the **world's most powerful AI cluster**. This announcement marks a significant milestone in AI computing capabilities, potentially pushing the boundaries of large language model training and performance.

---

# AI Discord Recap

> A summary of Summaries of Summaries

**1. LLM Advancements and Benchmarking**

- **Llama 3.1 Release Excitement**: **Llama 3.1** models, including **8B** and **405B**, are now available, sparking excitement in the community. Users shared their experiences and troubleshooting tips to tackle issues like running the model locally and managing high loss values during fine-tuning.
   - The community praised the model's performance, with some noting it surpasses existing proprietary models on benchmarks, while others highlighted challenges in practical deployment.
- **Meta's Open Source AI Commitment**: Meta's release of **Llama 3.1** with models like **405B** pushes the boundaries of open-source AI, offering **128K token context** and support for multiple languages. This move aligns with Mark Zuckerberg's vision for fostering innovation through open collaboration.
   - The community discussed the strategic implications of this release, emphasizing the model's potential to rival top closed-source alternatives like **GPT-4**.
    


**2. Optimizing LLM Inference and Training**

- **Efficient Fine-tuning Techniques Discussed**: The **ReFT paper** introduces a method that is **15x-60x** more parameter-efficient than LoRA by working on the **residual stream**, offering flexibility in combining training tasks with optimized parameters.
   - Community members engaged with the lead author to understand the practical applications, highlighting the method's potential to enhance fine-tuning efficiency.
- **GPU Compatibility Challenges**: Users reported issues with GPU detection on Linux, particularly with the **Radeon RX5700XT**, raising concerns about **RDNA 1 support**. Discussions emphasized the importance of proper configurations for GPU recognition.
   - Some users confirmed that extension packs weren't resolving the issues, indicating a need for further troubleshooting and potential updates from developers.
    


**3. Open-Source AI Frameworks and Community Efforts**

- **LlamaIndex Webinar on Efficient Document Retrieval**: The upcoming webinar will discuss **Efficient Document Retrieval with Vision Language Models** this Friday at **9am PT**. Participants can sign up to learn about cutting-edge techniques in document processing.
   - The webinar aims to explore ColPali's innovative approach to embedding **page screenshots** with Vision Language Models, enhancing retrieval performance over complex documents.
- **Magpie Paper Sparks Debate**: Members debated the utility of insights from the **Magpie paper**, questioning whether the generated instructions offer substantial utility or are merely a *party trick*.
   - The discussion highlights ongoing evaluations of emerging techniques in instruction generation, reflecting the community's critical engagement with new research.
    


**4. Multimodal AI and Generative Modeling Innovations**

- **UltraPixel Creates High Resolution Images**: **UltraPixel** is a project capable of generating extremely detailed high-resolution images, pushing the boundaries of image generation with a focus on **clarity** and **detail**.
   - The community showcased interest in the project's capabilities, exploring its potential applications and sharing the link to the project for further engagement.
- **Idefics2 and CodeGemma: New Multimodal Models**: **[Idefics2 8B Chatty](https://twitter.com/sanhestpasmoi/status/1787503160757485609)** focuses on elevated chat interactions, while **[CodeGemma 1.1 7B](https://twitter.com/reach_vb/status/1786469104678760677)** refines coding abilities.
   - These models represent significant advancements in multimodal AI, with the community discussing their potential to enhance user interaction and coding tasks.

---

# PART 1: High level Discord summaries




## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **NuminaMath Datasets Launch**: The **NuminaMath** datasets, featuring approximately **1M** math problem-solution pairs used to win the **Progress Prize** at the AI Math Olympiad, have been released. This includes subsets designed for **Chain of Thought** and **Tool-integrated reasoning**, significantly enhancing performance on math competition benchmarks.
   - Models trained on these datasets have demonstrated *best-in-class performance*, surpassing existing proprietary models. Check the release on the [🤗 Hub](https://huggingface.co/collections/AI-MO/numinamath-6697df380293bcfdbc1d978c).
- **Llama 3.1 Release Excitement**: The recent release of **Llama 3.1** has sparked excitement, with models like **8B** and **405B** now available for testing. Users are actively sharing experiences, including troubleshooting issues when running the model locally.
   - The community engages with various insights and offers support for operational challenges faced by early adopters.
- **Challenges in Model Fine-Tuning**: Frustrations have arisen regarding high loss values and performance issues in fine-tuning models for specific tasks. Resources and practices have been suggested to tackle these challenges effectively.
   - The exchange of knowledge aims to improve model training and evaluation processes.
- **UltraPixel Creates High Resolution Images**: **UltraPixel** is showcased as a project capable of generating extremely detailed high-resolution images. This initiative pushes the boundaries of image generation with a focus on **clarity** and **detail**.
   - Check out the project at [this link](https://huggingface.co/spaces/gokaygokay/UltraPixel).
- **Interest in Segmentation Techniques**: Another member expressed interest in effective **segmentation techniques** that work alongside background removal using diffusion models. They seek recommendations on successful methods or models.
   - The conversation is aimed at exploring better practices for image segmentation.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Magpie Paper Sparks Debate**: Members discussed whether insights from the [Magpie paper](https://arxiv.org/abs/2406.08464) offer substantial utility or are merely a *party trick*, focusing on the **quality** and **diversity** of generated instructions.
   - This inquiry highlights the ongoing evaluation of emerging techniques in instruction generation.
- **ReFT Paper Reveals Efficient Fine-tuning**: The lead author of the [ReFT paper](https://www.reddit.com/r/MachineLearning/comments/1e8qwnl/r_discussion_of_reft_paper_with_lead_author/) clarified that the method is **15x-60x** more parameter-efficient than LoRA by working on the *residual stream*.
   - This offers flexibility in combining training tasks with optimized parameters, reinforcing the relevance of efficient fine-tuning strategies.
- **Bud-E Voice Assistant Gains Traction**: The **Bud-E voice assistant** demo emphasizes its open-source potential and is currently optimized for **Ubuntu**, with hackathons led by Christoph for community engagement.
   - Such collaborative efforts aim to foster contributions from volunteers, enhancing the project's scope.
- **Llama 3.1 Impresses with Benchmark Performance**: Llama 3.1 405B Instruct-Turbo ranked 1st on GSM8K and closely matched GPT-4o on logical reasoning, although performance on MMLU-Redux appeared weaker.
   - This variation reinforces the importance of comprehensive evaluation across benchmark datasets.
- **Kuzu Graph Database Recommended**: Members recommended the **Kuzu GraphStore**, integrated with **LlamaIndex**, particularly for its [MIT license](https://github.com/kuzudb/kuzu) that ensures accessibility for developers.
   - The adoption of advanced graph database functionalities presents viable alternatives for data management, especially in complex systems.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio Performance Insights**: Users highlighted performance differences between **Llama 3.1** models, noting that running larger models demands significant GPU resources, especially for the **405B** variant.
   - One user humorously remarked about needing a *small nation's electricity supply* to run these models effectively.
- **Model Download Woes**: Several members noted difficulties with downloading models due to DNS issues and traffic spikes to **Hugging Face** caused by the popularity of **Llama 3.1**.
   - One user suggested the option to disable **IPv6** within the app to alleviate some of these downloading challenges.
- **GPU Compatibility Challenges**: New Linux users reported trouble with LM Studio recognizing GPUs like the **Radeon RX5700XT**, raising concerns about **RDNA 1 support**.
   - Discussion highlighted the importance of proper configurations for GPU recognition, with some users confirming **extension packs** weren’t resolving the issues.
- **Llama 3.1 Offers New Features**: **Llama 3.1** has launched with improvements, including context lengths of up to **128k**, available for download on Hugging Face.
   - Users are encouraged to explore the model's enhanced performance, particularly for memory-intensive tasks.
- **ROCm Performance Issues Post-Update**: A user noted that updating to **ROCm 0.2.28** resulted in significant slowdowns in inference, with consumption dropping to **150w** on their **7900XT**.
   - Reverting to **0.2.27** restored performance, indicating a need for clarity on functional changes in the newer version.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Llama 3.1 405B Launch and API Integration**: The highly anticipated **Llama 3.1 405B** model is now available on Perplexity, rivaling **GPT-4o** and **Claude Sonnet 3.5**, enhancing the platform's AI capabilities.
   - Users inquired about adding **Llama 3.1 405B** to the Perplexity API, asking if it will be available soon and sharing various experiences with model performance.
- **Concerns Over Llama 3.1 Performance**: Users reported issues with **Llama 3.1 405B**, including answer repetition and difficulties in understanding Asian symbols, leading many to consider reverting to **Claude 3.5 Sonnet**.
   - Comparative evaluations suggested that while **Llama 3.1** is a leap forward, **Claude** still holds an edge in speed and coding tasks.
- **Exploring Dark Oxygen and Mercury's Diamonds**: A recent discussion focused on **Dark Oxygen**, raising questions regarding its implications for atmospheric studies and ecological balance.
   - Additionally, insights emerged about **Diamonds on Mercury**, sparking interest in the geological processes that could lead to their formation.
- **Beach-Cleaning Robots Steal the Show**: Innovations in **beach-cleaning robot technology** were highlighted, showcasing efforts to tackle ocean pollution effectively.
   - The impact of these robots on **marine ecosystems** was a key point of discussion, with real-time data from trials being shared.
- **Perplexity API's DSGVO Compliance DB**: Concerns were raised about the **Perplexity API** being DSGVO-ready, with users seeking clarity on data protection compliance.
   - The conversation included a share of the [terms of service](https://www.perplexity.ai/hub/legal/perplexity-api-terms-of-service) referencing GDPR compliance.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Ranking AI Models with Kolors on Top**: In the latest discussion, users ranked AI models, placing **Kolors** at the top due to its impressive speed and performance, followed by **Auraflow**, **Pixart Sigma**, and **Hunyuan**.
   - Kolors' performance aligns well with user expectations for **SD3**.
- **Training Lycoris Hits Compatibility Snags**: Talks centered around training **Lycoris** using **ComfyUI** and tools like **Kohya-ss**, with users expressing frustration over compatibility requiring Python **3.10.9 or higher**.
   - There is anticipation for potential updates from **Onetrainer** to facilitate this process.
- **Community Reacts to Stable Diffusion**: Users debated the community's perception of **Stable Diffusion**, suggesting recent criticism often arises from misunderstandings around model licensing.
   - Concerns were raised about marketing strategies and perceived negativity directed at **Stability AI**.
- **Innovations in AI Sampling Methods**: A new sampler node has been introduced, implementing **Strong Stability Preserving Runge-Kutta** and implicit variable step solvers, capturing user interest in AI performance enhancements.
   - Users eagerly discussed the possible improvements these updates bring to AI model efficacy.
- **Casual Chat on AI Experiences**: General discussions flourished as users shared personal experiences with AI, including learning programming languages and assessing health-related focus challenges.
   - Such casual conversations added depth to the understanding of daily AI applications.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Llama 3 405B Launch Competitively Priced**: The **Llama 3 405B** has launched at **$3/M tokens**, rivaling **GPT-4o** and **Claude 3.5 Sonnet** while showcasing a remarkable **128K token context** for generating synthetic data.
   - Users reacted with enthusiasm, remarking that *'this is THE BEST open LLM now'* and expressing excitement over the model's capabilities.
- **Growing Concerns on Model Performance**: Feedback on the **Llama 405B** indicates mixed performance results, especially in translation tasks where it underperformed compared to **Claude** and **GPT-4**.
   - Some users reported the **70B version** generated *'gibberish'* after a few tokens, raising flags about its reliability for task-specific usage.
- **Exciting OpenRouter Feature Updates**: New features on **OpenRouter** include **Retroactive Invoices**, **custom keys**, and improvements to **the Playground**, enhancing user functionality overall.
   - Community members are encouraged to share feedback [here](http://openrouter.ai/chat) to further optimize the user experience.
- **Multi-LLM Prompt Competition Launched**: A **prompting competition** for challenging the **Llama 405B**, **GPT-4o**, and **Claude 3.5 Sonnet** has been announced, with participants vying for a chance to win **15 free credits**.
   - Participants are eager to know the judging criteria, especially regarding what qualifies as a tough prompt.
- **DeepSeek Coder V2 Inference Provider Announced**: The **DeepSeek Coder V2** new private inference provider has been introduced, operating without input training, which broadens OpenRouter's offerings significantly.
   - Users can start exploring the service via [DeepSeek Coder](https://openrouter.ai/models/deepseek/deepseek-coder).



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **Flash Attention Confusion in CUDA**: A member questioned the efficient management of registers in **Flash Attention**, raising concerns about its use alongside shared memory in CUDA programming.
   - This leads to a broader need for clarity in register allocation strategies in high-performance computing contexts.
- **Memory Challenges with Torch Compile**: Utilizing `torch.compile` for a small **Bert** model led to significant RAM usage, forcing a batch size cut from **512** to **160**, as performance lagged behind eager mode.
   - Testing indicated that the model compiled successfully despite these concerns, highlighting memory management issues in PyTorch.
- **Meta Llama 3.1 Focus on Text**: Meta's **Llama 3.1 405B** release expanded context length to **128K** and supports **eight languages**, excluding multi-modal features for now, sparking strategic discussions.
   - This omission aligns with expectations around potential financial outcomes and competitive positioning ahead of earnings reports.
- **Optimizing CUDA Kernel Performance**: User experiences showed that transitioning to tiled matrix multiplication resulted in limited performance gains, similar to findings in a related article on CUDA matrix multiplication benchmarks.
   - The discussion emphasized the importance of compute intensity for optimizing kernel performance, especially at early stages.
- **Stable Diffusion Acceleration on AMD**: A post detailed how to optimize inferencing for **Stable Diffusion** on **RX7900XTX** using the [Composable Kernel library](https://github.com/ROCm/composable_kernel/discussions/1032) for AMD RDNA3 GPUs.
   - Additionally, support for Flash Attention on AMD ROCm, effective for **mi200 & mi300**, was highlighted in a recent [GitHub pull request](https://github.com/Dao-AILab/flash-attention/pull/1010).



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GEMINI Competition Sparks Interest**: A member expressed enthusiasm for the **GEMINI Competition** from Google, looking for potential collaborators for the hackathon.
   - *Reach out if you're interested to collaborate!*
- **Llama 3.1 Model Draws Mixed Reactions**: Members reacted to the **Llama-3.1** Model, with some labeling it **soulless** compared to earlier iterations like **Claude** and **Gemini**, which were seen to retain more creative depth.
   - This discussion pointed out a divergence in experiences and expectations of recent models.
- **Fine-Tuning Llama 3.1 for Uncensored Output**: One user is working to fine-tune **Llama-3.1 405B** for an uncensored version, aiming to release it as **Llama3.1-406B-uncensored** on Hugging Face after several weeks of training.
   - This effort highlights the ongoing interest in developing alternatives to constrained models.
- **Voice AI in Discord Presents Challenges**: Discussion arose around creating AI bots capable of engaging in **Discord voice channels**, emphasizing the complexity of the task due to current limitations.
   - Members noted the technical challenges that need addressing for effective implementation.
- **Eager Anticipation for Alpha Release**: Members are keenly awaiting the alpha release, with some checking the app every 20 minutes, expressing uncertainty about whether it will launch at the end of July or earlier.
   - There's a call for clearer communications from developers regarding timelines.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo Community Meeting Presentations Open Call**: There's an open call for presentations at the [Mojo Community Meeting](https://modul.ar/community-meeting-doc) on August 12 aimed at showcasing what developers are building in Mojo.
   - Members can sign up to share experiences and projects, enhancing community engagement.
- **String and Buffer Optimizations Take the Stage**: Work on **short string optimization** and **small buffer optimization** in the standard library is being proposed for presentation, highlighting its relevance for future meetings.
   - This effort aligns with past discussion themes centered on performance enhancements.
- **Installing Mojo on an Ubuntu VM Made Simple**: Installation of Mojo within an Ubuntu VM on Windows is discussed, with **WSL** and **Docker** suggested as feasible solutions.
   - Concerns about possible installation issues are raised, but the general consensus is that VM usage is suitable.
- **Mojo: The Future of Game Engine Development**: Mojo's potential for creating next-gen game engines is discussed, emphasizing its strong support for heterogeneous compute via GPU.
   - Challenges with allocator handling are noted, indicating some hurdles in game development patterns.
- **Linking Mojo with C Libraries**: There’s ongoing dialogue about improving Mojo's linking capabilities with C libraries, especially utilizing **libpcap**.
   - Members advocate for **ktls** as the default for Mojo on Linux to enhance networking functionalities.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **FSDP Performance Troubles with nn.Parameters**: A user faced a **20x slowdown** when adding `nn.Parameters` with **FSDP**, but a parameter size of **16** significantly enhanced performance.
   - They discussed issues about **buffer alignment** affecting **CPU** performance despite fast GPU kernels.
- **Llama 3.1 Instruct Access on High-End Hardware**: A member successfully hosted **Llama 3.1 405B instruct** on **8xH100 80GB**, available via a [chat interface](https://chat.tune.app/) and [API](https://studio.tune.app/).
   - However, access requires a login, raising discussions on costs and hardware limitations.
- **Introducing Switch SAE for Efficient Training**: The **Switch SAE** architecture improves scaling in sparse autoencoders (SAEs), addressing training challenges across layers.
   - Relevant papers suggest this could help recover features from superintelligent language models.
- **Concerns Over Llama 3's Image Encoding**: Discussion surfaced regarding **Llama 3's** image encoder resolution limit of **224x224**, with suggestions to use a **vqvae-gan** style tokenizer for enhancement.
   - Suggestions were made to follow Armen's group, highlighting potential improvements.
- **Evaluating Task Grouping Strategies**: Members recommended using **groups** for nested tasks and **tags** for simpler arrangements, as endorsed by **Hailey Schoelkopf**.
   - This method aims to streamline task organization effectively.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Meta's Premium Llama 405B Rollout**: Speculation suggests that Meta may announce a **Premium** version of **Llama 405B** on **Jul 23**, after recently removing restrictions on Llama models, paving the way for more diverse applications.
   - This change sparks discussions about broader use cases, departing from merely enhancing other models.
- **NVIDIA's Marketplace Strategies**: Concerns about **NVIDIA** potentially monopolizing the AI landscape were raised, aiming to combine **hardware**, **CUDA**, and model offerings.
   - A user pointed out that such dominance might lead to immense profits, though regulatory challenges could impede this vision.
- **OpenAI's Pricing Dynamics**: OpenAI's introduction of *free* fine-tuning for **gpt-4o-mini** up to **2M tokens/day** has ignited discussions about the competitive pricing environment in AI.
   - Members characterized the pricing landscape as chaotic, emerging in response to escalating competition.
- **Llama 3.1 Surpasses Expectations**: The launch of [Llama 3.1](https://llama.meta.com/) introduced models with **405B parameters** and enhanced multilingual capabilities, demonstrating similar performance to **GPT-4** in evaluations.
   - The conversation about potential **model watermarking** and user download tracking ensued, focusing on compliance and privacy issues.
- **Magpie's Synthetic Data Innovations**: The [Magpie paper](https://arxiv.org/abs/2406.08464) highlights a method for generating **high-quality instruction data** for LLMs that surpasses existing data sources in **vocabulary diversity**.
   - Notably, **LLaMA 3 Base** finetuned on the **Magpie IFT dataset** outperformed the original **LLaMA 3 Instruct** model by 9.5% on **AlpacaEval**.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Llama 3.1 Release Generates Mixed Reactions**: The **Llama 3.1** release has stirred mixed feelings, with concerns about its **utility** particularly in the context of models like **Mistral**. Some members expressed their dissatisfaction, as captured by one saying, *'Damn they don't like the llama release'*.
   - Despite the hype, the feedback indicates a need for better performance metrics and clearer advantages over predecessors.
- **Users Face Training Challenges with Llama 3.1**: Errors related to the `rope_scaling` configuration while training **Llama 3.1** have contributed to community frustration. A workaround was found by updating transformers, showcasing resilience among users as one remarked, *'Seems to have worked thx!'*.
   - This highlights a broader theme of troubleshooting that persists with new model releases.
- **Concerns Over Language Inclusion in Llama 3.1**: The exclusion of **Chinese** language support in **Llama 3.1** has sparked discussions about its global implications. While the tokenizer includes Chinese, its lack of prioritization was criticized as a strategic blunder.
   - This conversation points to the ongoing necessity for **language inclusivity** in AI models.
- **Evaluation Scores Comparison: Llama 3.1 vs Qwen**: Community discussions focused on comparing the **cmmlu** and **ceval** scores of **Llama 3.1**, revealing only marginal improvements. Members pointed out that while **Qwen's** self-reported scores are higher, differences in evaluation metrics complicate direct comparisons.
   - This reflects the community's ongoing interest in performance benchmarks across evolving models.
- **Exploring LLM Distillation Pipeline**: A member shared the [LLM Distillery GitHub repo](https://github.com/golololologol/LLM-Distillery), highlighting a pipeline focusing on precomputing logits and KL divergence for LLM distillation. This indicates a proactive approach to refining distillation processes.
   - The community's interest in optimizing such pipelines reflects an ongoing commitment to improving model training efficiencies.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Code Confluence Tool Generates GitHub Summaries**: Inspired by **DSPY**, a member introduced **Code Confluence**, an OSS tool built with **Antlr**, **Chapi**, and DSPY pipelines, designed to create detailed summaries of GitHub repositories. The tool's performance is promising, as demonstrated on their [DSPY repo](https://github.com/unoplat/unoplat-code-confluence/blob/main/unoplat-code-confluence/examples/python/dspy/dspy_v1.md).
   - They also shared resources including the [Unoplat Code Confluence GitHub](https://github.com/unoplat/unoplat-code-confluence/) and the compilation of summaries called [OSS Atlas](https://github.com/unoplat/unoplat-oss-atlas/tree/main).
- **New AI Research Paper Alert**: A member shared a link to an AI research paper titled [2407.12865](https://arxiv.org/pdf/2407.12865), sparking interest in its findings. Community members are encouraged to analyze and discuss its implications.
   - Requests were made for anyone who replicates the findings in code or locates existing implementations to share them.
- **Comparison of JSON Generation Libraries**: Members discussed the strengths of libraries like **Jsonformer** and **Outlines** for structured JSON generation, noting that **Outlines** offers better support for Pydantic formats. While *Jsonformer* excels in strict compliance, *Guidance* and *Outlines* offer flexibility, adding complexity.
   - Taking into account the community's feedback, they are exploring the practical implications of each library in their workflow.
- **Challenges with Llama3 Structured Outputs**: Users expressed difficulty obtaining properly structured outputs from **Llama3** using DSPY. They suggested utilizing `dspy.configure(experimental=True)` with *TypedChainOfThought* to enhance success rates.
   - Concerns were raised over viewing model outputs despite type check failures, with `inspect_history` found to have limitations for debugging.
- **Exploring ColPali for Medical Documents**: A member shared experiences using **ColPali** for **RAG of medical documents with images** due to prior failures with **ColBert** and standard embedding models. Plans are underway to investigate additional vision-language models.
   - This exploration aims to bolster the effectiveness of information retrieval from complex document types.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex Webinar on Efficient Document Retrieval**: Join the upcoming webinar discussing **Efficient Document Retrieval with Vision Language Models** this Friday at **9am PT**. [Signup here](https://lu.ma/9q4ldrwc) to explore cutting-edge techniques.
   - ColPali introduces an innovative technique that directly embeds **page screenshots** with Vision Language Models, enhancing retrieval over complex documents that traditional parsing struggles with.
- **TiDB Future App Hackathon Offers $30,000 in Prizes**: Participate in the [TiDB Future App Hackathon 2024](https://t.co/vTV3t8daqT) for a chance to win from a prize pool of **$30,000**, including **$12,000** for the top entry. This competition urges innovative AI solutions using the latest **TiDB Serverless with Vector Search**.
   - Coders are encouraged to collaborate with @pingcap to showcase their best efforts in building advanced applications.
- **Explore Mixture-of-Agents with LlamaIndex**: A new video showcases the approach 'mixture of agents' using multiple local language models to potentially outmatch standalone models like **GPT-4**. Check the [step-by-step tutorial](https://t.co/EqF2RM3jeB) for insights into this enhancing technique.
   - Proponents suggest this method could provide a competitive edge, especially in projects requiring diverse model capabilities.
- **Llama 3.1 Models Now Available**: The **Llama 3.1** series now includes models of **8B**, **70B**, and **405B**, accessible through LlamaIndex with Ollama, although the largest model demands significant computing resources. Explore hosted solutions at [Fireworks AI](https://t.co/NMckK14nZf) for support.
   - Users should evaluate their computational capacity when opting for larger models to ensure optimal performance.
- **Clarifying context_window Parameters for Improved Model Usage**: The `context_window` parameter defines the total token limit that affects both input and output capacity of models. Miscalculating this can result in errors like ValueError due to exceeding limits.
   - Users are advised to adjust their input sizes or select models with larger context capabilities to optimize output efficiency.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Excitement Surrounds Llama 3.1 Launch**: The release of [Llama 3.1](https://ai.meta.com/research/publications/the-llama-3-herd-of-models/) includes the **405B model**, marking a significant milestone in open-source LLMs with **remarkable capabilities** rivaling closed models.
   - Initial evaluations show it as the first open model with frontier capabilities, praised for its accessibility for iterative research and development.
- **International Olympiad for Linguistics (IOL)**: The **International Olympiad for Linguistics (IOL)** commenced, challenging students to translate lesser-known languages using logic, mirroring high-stakes math competitions.
   - Participants tackle seemingly impossible problems in a demanding six-hour time frame, highlighting the intersection of logical reasoning and language.
- **Llama Pricing Insights**: Pricing for Llama 3.1's **405B model** ranges around **$4-5 per million tokens** across platforms like Fireworks and Together.
   - This competitive pricing strategy aims to capture market share before potentially increasing rates with growing adoption.
- **Evaluation of Llama's Performance**: Early evaluations indicate Llama 3.1 ranks highly on benchmarks including **GSM8K** and logical reasoning on **ZebraLogic**, landing between **Sonnet 3.5** and **GPT-4o**.
   - Challenges like maintaining schema adherence after extended token lengths were noted in comparative tests.
- **GPT-4o Mini Fine-Tuning Launch**: OpenAI announced fine-tuning capabilities for **GPT-4o mini**, available to tier 4 and 5 users, with the first **2 million training tokens free each day** until September 23.
   - This initiative aims to expand access and customization, as users assess its performance against the newly launched Llama 3.1.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **AgentState vs InnerAgentState Explained**: A discussion clarified the difference between `AgentState` and `InnerAgentState`, with definitions for `AgentState` provided and a suggestion to check the [LangChain documentation](https://python.langchain.com/v0.2/docs/integrations/vectorstores/chroma/#basic-example-using-the-docker-container) for further details.
   - Key fields of `AgentState` include `messages` and `next`, essential for context-dependent operations within LangChain.
- **Setting Up Chroma Vector Database**: Instructions were shared on how to set up **Chroma** as a vector database with open-source solutions in Python, requiring the installation of `langchain-chroma` and running the server via Docker.
   - Examples showed methods like `.add`, `.get`, and `.similarity_search`, highlighting the necessity of an OpenAI API Key for `OpenAIEmbeddings` usage.
- **Create a Scheduler Agent using Composio**: A guide for creating a **Scheduler Agent** with **Composio**, **LangChain**, and **ChatGPT** enables streamlined event scheduling via email. The guide is available [here](https://git.new/scheduler).
   - Composio enhances agents with effective tools, demonstrated in the [scheduler examples](https://git.new/scheduler), emphasizing efficiency in task handling.
- **YouTube Notes Generator is Here!**: The launch of the **YouTube Notes Generator**, an open-source project for generating notes from YouTube videos, was announced, aiming to facilitate easier note-taking directly from video content.
   - Learn more about this tool and its functionality on [LinkedIn](https://www.linkedin.com/posts/isham-rashik-5a547711b_machinelearning-artificialintelligence-deeplearning-activity-7221165319464095747-DMDS?utm_source=share&utm_medium=member_desktop).
- **Efficient Code Review with AI**: A new video titled **'AI Code Reviewer Ft. Ollama & Langchain'** introduces a CLI tool aimed at enhancing the code review process for developers; watch it [here](https://youtu.be/g_VRsjpC4e8).
   - This tool aims to streamline workflow by promoting efficient code evaluations across development teams.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **New Members Join the Cohere Community**: New members showcased enthusiasm about joining Cohere, igniting a positive welcome from the community.
   - *Community welcomes newbies with open arms*, creating an inviting atmosphere for discussions.
- **Innovative Fine-Tuning with Midicaps Dataset**: Progress on fine-tuning efforts surfaced with **midicaps**, showing promise based on previous successful projects.
   - Members highlighted *good results* from past endeavors, indicating potential future breakthroughs.
- **Clarifying Cohere's OCR Solutions**: **Cohere** utilizes [unstructured.io](https://unstructured.io) for its OCR capabilities, keeping options open for external integrations.
   - The community engaged in fruitful discussions about the customization and enhancement of OCR functionalities.
- **RAG Chatbot Systems Explored**: Chat history management in **RAG-based** ChatBot systems became a hot topic, highlighting the use of vector databases.
   - Feedback mechanisms such as thumbs up/down were proposed to optimize interaction experiences.
- **Launch of Rerank 3 Nimble with Major Improvements**: **Rerank 3 Nimble** hits the scene, delivering **3x higher throughput** while keeping accuracy in check, now available on [AWS SageMaker](https://cohere.com/blog/rerank-3-nimble).
   - *Say hello to increased speed for enterprise search!* This foundation model boosts performance for retrieval-augmented generation.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Llama 3.1 is officially here!**: Meta released the latest model, **Llama 3.1**, this morning, with support for the **8B** and **70B** instruct models. Check out the details in the [Llama 3.1 Model Cards and Prompt formats](https://llama.meta.com/docs/model-cards-and-prompt-formats/llama3_1).
   - The excitement was palpable, leading to humorous remarks about typos and excitement-induced errors.
- **MPS Support Pull Request Discussion**: The pull request titled [MPS support by maximegmd](https://github.com/pytorch/torchtune/pull/790) introduces checks for **BF16 on MPS devices**, aimed at improving testing on local Mac computers. Discussions indicate potential issues due to a common ancestor diff, suggesting a rebase might be a better approach.
   - This PR was highlighted as a critical update for those working with MPS.
- **LoRA Issues Persist**: An ongoing issue regarding the **LoRA** implementation not functioning as expected was raised, with suggestions made for debugging. One contributor noted challenges with **CUDA hardcoding** in their recent attempts.
   - This issue underscores the need for deeper troubleshooting in model performance.
- **Git Workflow Challenges Abound**: **Git workflow** challenges have been a hot topic, with many feeling stuck in a cycle of conflicts after resolving previous ones. Suggestions were made to tweak the workflow to minimize these conflicts.
   - Effective strategies for conflict resolution seem to be an ever-pressing need among the contributors.
- **Pad ID Bug Fix PR Introduced**: A critical bug related to **pad ID** displaying in generate was addressed in [Pull Request #1211](https://github.com/pytorch/torchtune/pull/1211) aimed at preventing this issue. It clarifies the implicit assumption of Pad ID being **0 in utils.generate**.
   - This fix is pivotal for ensuring proper handling of special tokens in future generative tasks.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Help needed for matmul-free-llm recreation**: There's a request for assistance in recreating **matmul-free-llm** with tinygrad, aiming to leverage [efficient kernels](https://github.com/ridgerchu/matmulfreellm/blob/master/mmfreelm/ops/fusedbitnet.py) while incorporating **fp8**.
   - *Hoping for seamless adaptation to Blackwell fp4 soon*.
- **M1 results differ from CI**: An M1 user is experiencing different results compared to CI, seeking clarification on setting up tests correctly with **conda** and environment variables.
   - *There's confusion due to discrepancies when enabling `PYTHON=1`, as it leads to an IndexError in tests.*
- **cumsum performance concerns**: A newcomer is exploring the **O(n)** implementation of **nn.Embedding** in tinygrad and how to improve **cumsum** from O(n^2) to O(n) using techniques from PyTorch.
   - *There’s speculation about constraints making this challenging, especially as it's a **$1000 bounty**.*
- **Seeking Pattern for Incremental Testing with PyTorch**: A member inquired about effective patterns for incrementally testing model performance in the sequence of **Linear, MLP, MoE,** and **LinearAttentionMoE** using PyTorch.
   - *They questioned whether starting tests from scratch is more efficient than incremental testing.*
- **Developing Molecular Dynamics Engine in Tinygrad**: A group is attempting to implement a **Molecular Dynamics engine** in tinygrad to train models predicting energies of molecular configurations, facing challenges with gradient calculations.
   - *They require the gradient of predicted energy concerning input positions for the force, but issues arise because they backpropagate through the model weights twice.*



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Int8 Implementation Confirmed**: Members discussed using **Int8**, with confirmation from one that it works, showing developer interest in optimization techniques.
   - *Hold a sec* was requested, indicating a potential for additional guidance and community support during the implementation.
- **ComfyUI Flow Script Guidance**: A user requested a script for **ComfyUI flow**, leading to advice on utilizing this framework for smoother setup processes.
   - This reflects a community trend towards efficiency and preferred workflows when working with complex system integrations.
- **Llama 3.1 Sets New Standards**: The release of [**Llama 3.1 405B**](https://ai.meta.com/research/publications/the-llama-3-herd-of-models/) introduces a context length of **128K**, offering significant capabilities across eight languages.
   - This leap positions Llama 3.1 as a strong contender against leading models, with discussions focusing on its diverse functionality.
- **Meta's Open Source Commitment**: Meta underlined its dedication to [**open source AI**](https://about.fb.com/news/2024/07/open-source-ai-is-the-path-forward/), as described in Mark Zuckerberg’s letter, highlighting developer and community benefits.
   - This aligns with their vision to foster collaboration within the AI ecosystem, aiming for wider accessibility of tools and resources.
- **Context Size Enhancements in Llama 3.1**: Discussions criticized the previous **8K** context size as insufficient for large documents, now addressed with the new **128K** size in Llama 3.1.
   - This improvement is viewed as crucial for tasks needing extensive document processing, elevating model performance significantly.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Llama 3.1 405 B Amazes Users**: **Llama 3.1 405 B** is reported to work fantastically out of the box with **OpenInterpreter**. Unlike **GPT-4o**, there's no need for constant reminders or restarts to complete multiple tasks.
   - Users highlighted that the experience provided by **Llama 3.1 405 B** significantly enhances productivity when compared to **GPT-4o**.
- **Frustrations with GPT-4o**: A user expressed challenges with **GPT-4o**, requiring frequent prompts to perform tasks on their computer. This frustration underscores the seamless experience users have with **Llama 3.1 405 B**.
   - The comparison made suggests a user preference leaning towards **Llama 3.1 405 B** for efficient task management.
- **Voice Input on MacOS with Coqui Model?**: A query arose about using voice input with a local **Coqui model** on **MacOS**. No successful implementations have been reported yet.
   - Community engagement remains open, but no further responses have surfaced to clarify the practicality of this application.
- **Expo App's Capability for Apple Watch**: Discussion affirmed that the **Expo app** should theoretically be able to build applications for the **Apple Watch**. However, further details or confirmations were not provided.
   - While optimistic, the community awaits practical validation of this capability in an *Apple Watch* context.
- **Shipping Timeline for the Device**: A member inquired about the **shipping timeline** for a specific device, indicating curiosity about its status. No updates or timelines were shared in the conversation.
   - The lack of information points to an opportunity for clearer communication regarding shipping statuses.



---



## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord

- **Clarification on OpenOrca Dataset Licensing**: A member inquired whether the **MIT License** applied to the **OpenOrca** dataset permits commercial usage of outputs derived from the **GPT-4 Model**.
   - *Can its outputs be used for commercial purposes?* highlights the ongoing discussion around dataset licensing in AI.
- **Plans for Open Sourcing Synthetic Dataset**: Another member revealed intentions to open source a **synthetic dataset** aimed at supporting both commercial and non-commercial projects, highlighting its relevance in the AI ecosystem.
   - They noted an evaluation of potential dependencies on **OpenOrca**, raising questions about its licensing implications in the broader dataset landscape.



---



## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **Miami Meetup Interest Sparks Discussion**: A member inquired about potential **meetups in Miami**, seeking connections with others in the area for gatherings.
   - So far, there have been no further responses or arrangements mentioned regarding this meetup inquiry.
- **NYC Meetup Gains Traction for August**: Another member expressed interest in attending meetups in **NYC** in late **August**, indicating a desire for community engagement.
   - This discussion hints at the possible coordination of events for local AI enthusiasts in the New York area.



---



## [AI Stack Devs (Yoko Li)](https://discord.com/channels/1122748573000409160) Discord

- **Artist Seeking Collaboration**: Aria, a **2D/3D artist**, expressed interest in collaborating with others in the community. They invited interested members to reach out via DM for potential projects.
   - This presents an opportunity for anyone in the guild looking to incorporate **artistic** skills into their AI projects, particularly in visualization or gaming.
- **Engagement Opportunities for AI Engineers**: The call for collaboration emphasizes the growing interest in merging **AI engineering** with creative domains like art and design.
   - Such collaborations can enhance the visual aspects of **AI projects**, potentially leading to more engaging user experiences.



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Mozilla Accelerator Application Deadline Approaches**: The application deadline for the **Mozilla Accelerator** is fast approaching, offering a **12-week program** with up to **$100k** in non-dilutive funds.
   - Participants will also showcase their projects on **demo day** with Mozilla, providing a pivotal moment for feedback and exposure. [Questions?](https://discord.com/channels/1089876418936180786/1245083732319408195)
- **Get Ready for Zero Shot Tokenizer Transfer Event**: A reminder of the upcoming **Zero Shot Tokenizer Transfer** event with Benjamin Minixhofer, scheduled for this month.
   - Details can be found in the [event's link](https://discord.com/channels/1089876418936180786/1089876419926032396/1261387457652592732), encouraging participation from interested engineers.
- **Introducing AutoFix: The Open Source Issue Fixer**: **AutoFix** is an open-source tool that can submit PRs directly from **Sentry.io**, streamlining issue management.
   - Learn more about this tool's capabilities in the detailed post linked here: [AutoFix Information](https://discord.com/channels/1089876418936180786/1089876419926032396/1261387457652592732).



---


The **LLM Perf Enthusiasts AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **DiscoResearch Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1265310564222111884)** (1 messages): 

> - `NuminaMath datasets`
> - `Docmatix dataset`
> - `SmolLM models`
> - `Chameleon model`
> - `Followgraph tool` 


- **NuminaMath Datasets Launch**: The **NuminaMath** datasets have been released, featuring about **1M** math competition problem-solution pairs, used to win the **Progress Prize** of the AI Math Olympiad. This includes **Chain of Thought** and **Tool-integrated reasoning** subsets designed to enhance mathematical reasoning.
   - These models trained on NuminaMath achieve **best-in-class performance**, surpassing proprietary models on math competition benchmarks and are available on the [🤗 Hub](https://huggingface.co/collections/AI-MO/numinamath-6697df380293bcfdbc1d978c).
- **Introducing Docmatix Dataset**: The **Docmatix** dataset has been introduced as a gigantic resource for document understanding. It aims to address the data coverage deficiencies that have hindered open-source models in document tasks.
   - This dataset is set to improve performance on various document tasks, which previously favored closed models due to lack of adequate open-source data.
- **SmolLM Models Released**: A new series of models called **SmolLM** has been released, featuring sizes of **135M**, **360M**, and **1.7B** parameters. They outperform **MobileLLM**, **Phi1.5**, and **Qwen2**, and are trained on a high-quality corpus.
   - This series addresses the growing importance of on-device deployment for **large language models** (LLMs), catering to diverse application needs.
- **Chameleon Model Now Available**: **Chameleon**, a multimodal model by Meta, is now integrated into **transformers** and comes in sizes of **7B** and **34B** parameters. This model aims to enhance various multimodal tasks.
   - The integration of Chameleon represents a significant advancement in the capabilities of **transformers** for handling diverse inputs and outputs.
- **Explore ML Connections with Followgraph**: A new tool called **Followgraph** has been launched to facilitate following interesting ML personalities. It’s aimed at enhancing the collaboration and networking opportunities within the ML community.
   - This tool allows users to discover and connect with influential figures in the machine learning space, adding a social dimension to professional interactions.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/_lewtun/status/1814958635732140336">Tweet from Lewis Tunstall (@_lewtun)</a>: We have just released the ✨NuminaMath datasets: the largest collection of ~1M math competition problem-solution pairs, ranging in difficulty from junior challenge to Math Olympiad preselection.  These...</li><li><a href="https://x.com/mervenoyann/status/1813963500513058849">Tweet from merve (@mervenoyann)</a>: Introducing Docmatix: a gigantic document understanding dataset 📑  Closed models outperformed open-source models in document tasks so far due to lack of data coverage 💔  but @huggingface M4 is here ...</li><li><a href="https://x.com/LoubnaBenAllal1/status/1813252390692303069">Tweet from Loubna Ben Allal (@LoubnaBenAllal1)</a>: On-device deployment  of LLMs is more important than ever. Today we’re releasing SmolLM a new SOTA series of 135M, 360M and 1.7B models:  - Outperforming MobileLLM, Phi1.5 and Qwen2 small models - Tra...</li><li><a href="https://x.com/NielsRogge/status/1814310702162551247">Tweet from Niels Rogge (@NielsRogge)</a>: We just shipped chat templates for vision-language models (VLMs)! 🔥  Models like LLaVa, LLaVa-NeXT, and LLaVa-Interleave can now all be called using the messages API.  Docs: https://huggingface.co/do...</li><li><a href="https://x.com/TheZachMueller/status/1813218332050358522">Tweet from Zach Mueller (@TheZachMueller)</a>: Lazy-loading model weights has been shipped into @huggingface transformers main! A tweet about what the heck that means...  Typically when you load in PyTorch weights, it&#39;s instantaneous (aka when...</li><li><a href="https://x.com/KonradSzafer/status/1815726520939212985">Tweet from Konrad Szafer (@KonradSzafer)</a>: We’ve just added a new method to the Transformers Tokenizer class to improve tracking and reproducibility.  You can now retrieve the exact chat template used by the Tokenizer! 🚀</li><li><a href="https://x.com/mervenoyann/status/1814278511785312320">Tweet from merve (@mervenoyann)</a>: Chameleon 🦎 by @Meta is now available in @huggingface transformers 😍 A multimodal model that comes in 7B and 34B sizes 🤩 But what makes this model so special?  keep reading ⇣</li><li><a href="https://x.com/NielsRogge/status/1810284458412573052">Tweet from Niels Rogge (@NielsRogge)</a>: 2 new depth estimation models now in @huggingface Transformers!  Depth Anything v2 & ZoeDepth  - Depth Anything v2 is relative, tells you the relative distance among the pixels  - ZoeDepth is absolute...</li><li><a href="https://x.com/julien_c/status/1814310812393120077">Tweet from Julien Chaumond (@julien_c)</a>: Friday @huggingface update.  For image generation models and LoRAs,  we now display tiny previews of models,  directly on users profiles.  Have a great weekend!  🔥</li><li><a href="https://x.com/severo_dev/status/1815684824893436246">Tweet from Sylvain Lesage (@severo_dev)</a>: [New tool] Follow interesting ML persons 👩‍🎨 👨‍🎤 👩‍🏫 with Followgraph  https://huggingface.co/spaces/severo/followgraph</li><li><a href="https://x.com/vanstriendaniel/status/1814298698383585692">Tweet from Daniel van Strien (@vanstriendaniel)</a>: When sharing model fine-tuning notebooks, it can be helpful to show the input dataset. You can now embed a dataset viewer directly in a @GoogleColab notebook. Here&#39;s an edited @UnslothAI notebook ...</li><li><a href="https://x.com/RemiCadene/status/1813675172492411170">Tweet from Remi Cadene (@RemiCadene)</a>: 🚨 We can now visualize LeRobot datasets directly on hugging face hub. Try it out on the dataset I just recorded 😇 https://huggingface.co/spaces/lerobot/visualize_dataset Hugging Face has the potenti...</li><li><a href="https://x.com/abhi1thakur/status/1813892464144798171">Tweet from abhishek (@abhi1thakur)</a>: We just integrated dataset viewer in AutoTrain 💥 So, now you can look into your dataset, identify correct splits and columns before training the model, without leaving the page 🚀</li><li><a href="https://x.com/reach_vb/status/1815434084572688581">Tweet from Vaibhav (VB) Srivastav (@reach_vb)</a>: We put together a detailed blog post going through the steps for running Mistral on Mac and all the updates announced by Apple during WWDC:  https://huggingface.co/blog/mistral-coreml</li><li><a href="https://x.com/evijitghosh/status/1814003112128172255">Tweet from Avijit Ghosh (@evijitghosh)</a>: http://x.com/i/article/1814002459108691968</li><li><a href="https://x.com/calebfahlgren/status/1814116515328807226">Tweet from Caleb (@calebfahlgren)</a>: Wrote a blog post on how you can use the Datasets Explorer to find really interesting insights on @huggingface datasets 🔥  There&#39;s even a couple examples of the @duckdb spatial extension with som...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1265020925708079125)** (1104 messages🔥🔥🔥): 

> - `Llama 3.1 release`
> - `Kanye West controversy`
> - `Building PC setups`
> - `Model fine-tuning practices`
> - `Textbook recommendations for LLMs` 


- **Llama 3.1 release excitement**: The release of Llama 3.1 has generated excitement, with models like 8B and 405B now available for testing and deployment.
   - Users are sharing their experiences and troubleshooting issues such as ValueErrors when attempting to run the model locally.
- **Kanye West's influence in music**: Despite the controversies surrounding Kanye West, many users like kebab_addict express an appreciation for his musical talent and impact on the industry.
   - Discussions also highlight the complexity of separating an artist's work from their personal controversies.
- **Building PC setups and GPU discussions**: Users are discussing various GPU options for building affordable PC setups, with recommendations for models like the 3060 and 4060ti.
   - Some express concerns over the rising costs of components while sharing personal anecdotes about acquiring their hardware.
- **Model fine-tuning practices**: The challenges of fine-tuning models for specific tasks are being discussed, with users expressing frustrations about high loss values and performance issues.
   - There are suggestions for resources and practices to better handle model training and evaluation.
- **Textbook recommendations for LLMs**: A user is seeking comprehensive textbooks covering recent innovations in LLMs, expressing a preference for written material over video content.
   - Titles such as 'Transformers for Natural Language Processing' are mentioned as potential resources, though they primarily focus on applied learning.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/starsnatched/MemeGPT">starsnatched/MemeGPT · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/spaces/enzostvs/zero-gpu-spaces">— Zero GPU Spaces — - a Hugging Face Space by enzostvs</a>: no description found</li><li><a href="https://x.com/osanseviero/status/1815769303188205678">Tweet from Omar Sanseviero (@osanseviero)</a>: Llama 3.1 is out 🔥Enjoy!  - Learn all about it https://hf.co/blog/llama31 - Models https://hf.co/meta-llama - Community quants https://hf.co/hugging-quants - How to use it https://github.com/huggingf...</li><li><a href="https://huggingface.co/chat/settings/meta-llama/Meta-Llama-3.1-405B-Instruct-FP8">HuggingChat</a>: Making the community's best AI chat models available to everyone.</li><li><a href="https://huggingface.co/spaces/Xenova/whisper-speaker-diarization">Whisper Speaker Diarization - a Hugging Face Space by Xenova</a>: no description found</li><li><a href="https://www.nbcnews.com/news/world/snoop-dogg-will-carry-olympic-torch-final-leg-paris-rcna163234">Snoop Dogg will carry the Olympic torch on its final leg to Paris</a>: The culturally ubiquitous rapper will see the flame&#x27;s tradition through ahead of Friday&#x27;s opening ceremony.</li><li><a href="https://www.amazon.com/ASUS-ProArt-GeForce-Graphics-DisplayPort/dp/B0CCC7MP3H">no title found</a>: no description found</li><li><a href="https://arxiv.org/abs/2407.14561">NNsight and NDIF: Democratizing Access to Foundation Model Internals</a>: The enormous scale of state-of-the-art foundation models has limited their accessibility to scientists, because customized experiments at large model sizes require costly hardware and complex engineer...</li><li><a href="https://tenor.com/view/what-gif-21384529">What GIF - What - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://arxiv.org/abs/1810.04805">BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding</a>: We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed t...</li><li><a href="https://tenor.com/view/patrick-stupid-drooling-patrick-star-spongebob-gif-12221001666588210206">Patrick Stupid GIF - Patrick Stupid Drooling - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/stfu-kanye-kanye-west-shut-up-dance-gif-23839788">Stfu Kanye GIF - Stfu Kanye Kanye West - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/spongebob-squarepants-begging-pretty-please-beg-on-your-knees-pray-for-mercy-gif-26344462">Spongebob Squarepants Begging GIF - Spongebob Squarepants Begging Pretty Please - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/blog/nroggendorff/train-with-llama-architecture">Train a Llama model from scratch</a>: no description found</li><li><a href="https://doc.rust-lang.org/book/#the-rust-programming-language">The Rust Programming Language - The Rust Programming Language</a>: no description found</li><li><a href="https://youtu.be/01g_EfO-Dms?si=tMF70x7MhxKw8S95">Make your agents 10x more reliable? Flow engineer 101</a>: Deep dive into flow engineer &amp; lang graph, build a reliable SQL agentGet Codeium (FREE Github Copilot alternative): https://codeium.com/?utm_source=youtube&amp;u...</li><li><a href="https://tenor.com/view/sad-upset-violin-sponge-bob-mr-crab-gif-3466351">Sad Violin GIF - Sad Upset Violin - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tinyurl.com/2nfwn2xy">Vision Card</a>: no description found</li><li><a href="https://tenor.com/view/lindsey-stirling-lindsey-stirling-cute-adorable-gif-19359953">Lindsey Stirling Cute GIF - Lindsey Stirling Lindsey Stirling - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/openai/whisper-large-v3">openai/whisper-large-v3 · Hugging Face</a>: no description found</li><li><a href="https://tenor.com/view/waiting-waiting-patiently-waiting-for-you-waiting-on-you-gif-15489516379864441176">Waiting Waiting Patiently GIF - Waiting Waiting patiently Waiting for you - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/wizard-dance-ena-gif-27696814">Wizard Dance GIF - Wizard Dance Ena - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/good-morning-gif-11437316614611695342">Good Morning GIF - Good morning - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/mark-zuckerberg-gif-14169217">Mark Zuckerberg GIF - Mark Zuckerberg - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/scared-dog-shivering-dog-dog-shaking-meme-gif-26566244">Scared Dog Shivering Dog GIF - Scared Dog Shivering Dog Dog Shaking Meme - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/cat-twitching-tweaking-blink-blinking-gif-15542945703716446313">Cat Twitching GIF - Cat Twitching Tweaking - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/batman-mad-angry-tell-me-interogating-gif-17869813">Batman Mad GIF - Batman Mad Angry - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/biggest-boy-family-guy-chris-griffin-dancing-gif-17316116">Biggest Boy Family Guy GIF - Biggest Boy Family Guy Chris Griffin - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/subida-gif-18379274">Subida GIF - Subida - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/huh-cat-gif-26460616">Huh Cat GIF - Huh Cat - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/bh187-spongebob-patrick-star-derp-duh-gif-21500047">Bh187 Spongebob GIF - Bh187 Spongebob Patrick Star - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/kotmadam-odilon-old-man-no-sigh-gif-18163378">Kotmadam Odilon GIF - Kotmadam Odilon Old Man - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/hello-street-cat-huge-bite-little-scraggly-guy-kibble-gif-8033892186058013617">Hello Street Cat Huge Bite GIF - Hello street cat Huge bite Little scraggly guy - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/bugs-bunny-no-no-bunny-bugs-gif-7909500831201365932">Bugs Bunny No GIF - Bugs bunny no No Bunny - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/patrick-menacingly-spongebob-standing-there-gif-19452999">Patrick Menacingly GIF - Patrick Menacingly Spongebob - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/dead-gif-18865199">Dead GIF - Dead - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/dpowe-gif-24107728">Dpowe GIF - Dpowe - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/spongebob-spongebob-meme-spongebob-mafia-mafia-money-gif-12714856527416165903">Spongebob Spongebob Meme GIF - Spongebob Spongebob meme Spongebob mafia - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/caveman-spongebob-spongegar-gif-5620708">Caveman Spongebob GIF - Caveman Spongebob Spongegar - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/lag-android-glitch-kid-gif-15712794">Lag Android GIF - Lag Android Glitch - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://youtu.be/53kr_dvof1w">I Wore a Hollywood Disguise to Buy a PC - Scrapyard Wars 2024 Part 1</a>: https://jawa.link/ScrapyardWarsThanks to Jawa for sponsoring this season of Scrapyard Wars! Join in on the spirit with Jawa: THE marketplace for buying and s...</li><li><a href="https://tenor.com/view/troll-lol-gta-gta-san-andreas-running-gif-25040072">Troll Lol GIF - Troll Lol Gta - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/kanye-west-kanye-ai-dance-gif-6290223825845382767">Kanye West Ai GIF - Kanye west Kanye Ai - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/journey-car-kissing-gif-17893723">Journey Car GIF - Journey Car Kissing - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/homelander-based-the-boys-homelander-the-boys-facts-gif-26206051">Homelander Based GIF - Homelander Based The Boys - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/zeng-this-guy-right-here-this-right-here-point-out-point-gif-23913867">Zeng This Guy Right Here GIF - Zeng This Guy Right Here This Right Here - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/oliver-twist-gif-26543489">Oliver Twist GIF - Oliver Twist - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/kanye-haircut-kanye-west-stare-mattscrub-gif-13171403811728519930">Kanye Haircut GIF - Kanye Haircut Kanye west - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/blog/llama31">Llama 3.1 - 405B, 70B &amp; 8B with multilinguality and long context</a>: no description found</li><li><a href="https://huggingface.co/spaces?search=Whi">Spaces - Hugging Face</a>: no description found</li><li><a href="https://tenor.com/view/ye-kanye-kanye-west-dj-khaled-khaled-gif-24604192">Ye Kanye GIF - Ye Kanye Kanye West - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/learn/nlp-course">Introduction - Hugging Face NLP Course</a>: no description found</li><li><a href="https://huggingface.co">Hugging Face – The AI community building the future.</a>: no description found</li><li><a href="https://huggingface.co/docs">Hugging Face - Documentation</a>: no description found</li><li><a href="https://github.com/huggingface/huggingface-llama-recipes">GitHub - huggingface/huggingface-llama-recipes</a>: Contribute to huggingface/huggingface-llama-recipes development by creating an account on GitHub.</li><li><a href="https://tenor.com/view/wizard-crawly-crawly-wizard-mall-wizard-mall-gif-14992836518596419882">Wizard Crawly GIF - Wizard Crawly Crawly wizard - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/biggest-boy-family-guy-gif-10780370951992646584">Biggest Boy Family Guy GIF - Biggest boy Family guy - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.packtpub.com/en-us/product/transformers-for-natural-language-processing-9781800565791">Transformers for Natural Language Processing | Data | eBook</a>: Build innovative deep neural network architectures for NLP with Python, PyTorch, TensorFlow, BERT, RoBERTa, and more. Instant delivery. Top rated Mobile Application Development products.</li><li><a href="https://www.techpowerup.com/gpu-specs/geforce-rtx-5090.c4216">NVIDIA GeForce RTX 5090 Specs</a>: NVIDIA GB202, 2520 MHz, 20480 Cores, 640 TMUs, 192 ROPs, 28672 MB GDDR7, 2500 MHz, 448 bit</li><li><a href="https://www.techpowerup.com/gpu-specs/voodoo3-3000-agp.c3555#:~:text=it%20might%20not%20be%20able%20to%20run%20all%20the%20latest%20games)">3dfx Voodoo3 3000 AGP Specs</a>: 3dfx Avenger, 166 MHz, 1 Pixel Shaders, 0 Vertex Shaders, 2 TMUs, 1 ROPs, 16 MB SDR, 166 MHz, 128 bit</li><li><a href="https://www.techpowerup.com/gpu-specs/geforce-rtx-5060.c4219">NVIDIA GeForce RTX 5060 Specs</a>: NVIDIA GB206, 2520 MHz, 4608 Cores, 144 TMUs, 48 ROPs, 8192 MB GDDR7, 2500 MHz, 128 bit
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1265079285471903837)** (4 messages): 

> - `Speaker Diarization & Transcription`
> - `Sankey Plots Visualization`
> - `Dynamic Graph Node Management`
> - `PEFT Model Loading Methods`
> - `Adapter Configuration in Models` 


- **Automate Speaker Diarization and Transcriptions**: A member is seeking a way to automate **speaker diarization**, **whisper transcriptions**, and timestamps for uploaded WAV files into a single database.
   - They are looking for **open source repositories** or models to implement this pipeline.
- **Sankey Plots Using Matplotlib**: A user shared their experience with **Sankey plots** (also known as flow plots) using **matplotlib**, noting that the implementation has room for improvement.
   - They expressed a desire to make many **changes** to enhance the visualization capability of dataset filtering.
- **Dynamic Node Management in Graphs**: A user inquired about the feasibility of **dynamically adding and removing nodes** from a graph to gradually build an info database.
   - Their goal is to avoid the need to parse numerous files all at once, suggesting a more streamlined process.
- **PEFT Model Loading Insights**: A member highlighted two methods to load a **PEFT model**, providing examples with code snippets for both techniques.
   - They questioned how the first method retrieves the entire model from an adapter link, speculating that the **adapter config** might contain the necessary base model details.


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1265030075150106675)** (5 messages): 

> - `Willing Suspension of Disbelief`
> - `nanoLLaVA model`
> - `Meta's Llama 3.1 release`
> - `Mark Zuckerberg's vision for open-source AI` 


- **Exploring Delving in Storytelling**: A study titled *Willing Suspension of Disbelief* investigates the role of volition in how audiences engage with stories, emphasizing the importance of delving into narrative experiences.
   - This research can be accessed [here](https://www.researchgate.net/publication/298068504_Willing_Suspension_of_Disbelief_A_study_of_the_role_of_volition_in_the_experience_of_delving_into_a_story).
- **nanoLLaVA model discussion**: A member highlighted the [nanoLLaVA model](https://huggingface.co/spaces/qnguyen3/nanoLLaVA), noting it was duplicated from another model called *llava-next*.
   - The conversation included images related to the model but did not elaborate further.
- **Launch of Llama 3.1 AI Models**: Meta announced the release of the Llama 3.1 family, praising its performance that rivals top closed-source models, especially the **405B** version.
   - The release aims to promote an open-source AI ethos and offers [Mark Zuckerberg’s letter](https://about.fb.com/news/2024/07/open-source-ai-is-the-path-forward/) on why open source benefits developers and the community.
- **Mark Zuckerberg advocates for open-source AI**: Meta's CEO shared his vision for an open AI ecosystem, asserting that the features of Llama 3.1 will aid developers in unlocking new capabilities, such as synthetic data generation.
   - Zuckerberg emphasized the relevance of open-source AI, stating it is *the path forward* for both developers and Meta's future strategies.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/qnguyen3/nanoLLaVA">nanoLLaVA-1.5 - a Hugging Face Space by qnguyen3</a>: no description found</li><li><a href="https://www.neowin.net/news/mark-zuckerberg-explains-why-open-source-ai-is-good-for-developers/">Mark Zuckerberg explains why open source AI is good for developers</a>: Mark Zuckerberg believes that open-source AI is the future of AI, fostering unrestricted innovation similar to how open-source development has accelerated progress in other fields.</li><li><a href="https://ai.meta.com/blog/meta-llama-3-1/">no title found</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1265036255784341584)** (10 messages🔥): 

> - `UltraPixel high resolution images`
> - `Rust client library for Gradio`
> - `SmolLM Arena updates`
> - `YouTube Notes Generator`
> - `Mistral-NeMo 12B Instruct` 


- **UltraPixel creates high resolution images**: One user showcased their project called **UltraPixel**, capable of generating extremely detailed high resolution images at this [link](https://huggingface.co/spaces/gokaygokay/UltraPixel).
   - This project aims to push the boundaries of image generation with a focus on **clarity** and **detail**.
- **Rust library for Gradio development**: A new project for a **Rust client library for Gradio** has been announced with active testing using `hf-audio/whisper-large-v3` and other models, available on [GitHub](https://github.com/JacobLinCool/gradio-rs).
   - The library is in the **early stages**, inviting contributions and feedback from the community as it develops.
- **SmolLM Arena gets a new interface**: The **SmolLM Arena** has introduced a new interface with chatbots instead of text boxes, improving speed and user experience, detailed at [this link](https://huggingface.co/spaces/as-cle-bert/smolLM-arena).
   - Users can now **compare** small language models and cast votes for their favorites, combining fun and interactivity.
- **YouTube Notes Generator project unveiled**: A new **YouTube Notes Generator** project has been announced, which can create detailed notes from YouTube videos, with its code hosted on [GitHub](https://github.com/di37/youtube-notes-generator).
   - It features a **Streamlit UI** for easy use, allowing users to generate and interact with notes from video content.
- **Lightning fast Mistral-NeMo 12B Instruct demo**: A demo of **Mistral-NeMo 12B Instruct** using llama.cpp was shared, showcasing its lightning-fast chat capabilities, available at [this link](https://huggingface.co/spaces/gokaygokay/Mistral-NeMo-llamacpp).
   - This project emphasizes performance in producing quick and responsive interactions.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/gokaygokay/UltraPixel">UltraPixel - a Hugging Face Space by gokaygokay</a>: no description found</li><li><a href="https://huggingface.co/spaces/gokaygokay/Mistral-NeMo-llamacpp">Mistral NeMo llama.cpp - a Hugging Face Space by gokaygokay</a>: no description found</li><li><a href="https://huggingface.co/spaces/Sergidev/HD-Pony-Diffusion-v6">HD Pony Diffusion - a Hugging Face Space by Sergidev</a>: no description found</li><li><a href="https://github.com/qompassai/KO">GitHub - qompassai/KO: Kyber Odyssey: Charting a course for secure innovation in a post-Crowdstrike world</a>: Kyber Odyssey: Charting a course for secure innovation in a post-Crowdstrike world - qompassai/KO</li><li><a href="https://github.com/JacobLinCool/gradio-rs">GitHub - JacobLinCool/gradio-rs: Gradio Client in Rust.</a>: Gradio Client in Rust. Contribute to JacobLinCool/gradio-rs development by creating an account on GitHub.</li><li><a href="https://github.com/di37/youtube-notes-generator">GitHub - di37/youtube-notes-generator: AI-powered YouTube Notes Generator: Create detailed notes from YouTube videos. Streamlit UI for easy use.</a>: AI-powered YouTube Notes Generator: Create detailed notes from YouTube videos. Streamlit UI for easy use. - di37/youtube-notes-generator
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1265367731813879960)** (2 messages): 

> - `Anime-style dataset for Anything V5`
> - `Fine-tuning SD models` 


- **Inquiry on Anime-style Dataset for Anything V5**: A member asked about the dataset used for **anime-style generation** in the [Anything V5 API Inference](https://huggingface.co/stablediffusionapi/anything-v5). They provided a link to a generated image along with information on obtaining an API key and coding examples.
   - They shared that no payment is needed for the API key and linked to the [Stable Diffusion API](http://stablediffusionapi.com/) for more details.
- **Discussion on Fine-tuning SD Models**: A member inquired about how to fine-tune **Stable Diffusion (SD) models** using tailored datasets. This highlights ongoing interest in customizing models for specific applications.



**Link mentioned**: <a href="https://huggingface.co/stablediffusionapi/anything-v5">stablediffusionapi/anything-v5 · Hugging Face</a>: no description found

  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1265093121142816831)** (17 messages🔥): 

> - `Non packed datasets with SFTTrainer`
> - `Error handling with tensor creation`
> - `Embedding model for numerical data`
> - `Using Donut for generation`
> - `Modifications in Transformers library` 


- **Challenges with Non Packed Datasets in SFTTrainer**: A user inquired if anyone has utilized non packed datasets with **SFTTrainer** for LLMs, expressing concerns about the limited examples and an error faced during their implementation.
   - *Careful prompt engineering* was suggested as a potential solution alongside using **PEFT** for hardware efficiency.
- **Tensor Creation Error Investigation**: Another user encountered an error stating 'Unable to create tensor', advising to activate **truncation** and **padding** options during the setup.
   - For more assistance, they shared a link to a [Hugging Face forum post](https://discuss.huggingface.co/t/unable-to-create-tensor-you-should-probably-activate-truncation-and-or-padding-with-padding-true-truncation-true/98833) detailing the issue.
- **Seeking Numerical Data Embedding Model**: A member asked for recommendations on an embedding model optimized for **numerical data**, seeking specialized options.
   - No specific model was directly suggested in response to the inquiry.
- **Exploring Donut for Text Generation**: One user shared their experience using the **Donut** model from GitHub for generation, highlighting the need to adapt to changes between two versions of the Transformers library.
   - They linked to relevant [GitHub Pull Requests](https://github.com/huggingface/transformers/pull/22748) that explain the adjustments and implications for **Donut** generation.
- **Splitting Large Embedded Text for LLMs**: A user requested insights on splitting large embedded text for effective use with LLMs.
   - The dialogue didn't provide specific strategies or solutions to address this concern.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/clovaai/donut/">GitHub - clovaai/donut: Official Implementation of OCR-free Document Understanding Transformer (Donut) and Synthetic Document Generator (SynthDoG), ECCV 2022</a>: Official Implementation of OCR-free Document Understanding Transformer (Donut) and Synthetic Document Generator (SynthDoG), ECCV 2022 - clovaai/donut</li><li><a href="https://discuss.huggingface.co/t/unable-to-create-tensor-you-should-probably-activate-truncation-and-or-padding-with-padding-true-truncation-true/98833">Unable to create tensor, you should probably activate truncation and/or padding with &#39;padding=True&#39; &#39;truncation=True&#39;</a>: I am trying to use a non packed dataset with SFTTrainer by setting ‘packing=False’ but I get the error: Unable to create tensor, you should probably activate truncation and/or padding with ‘padding=Tr...</li><li><a href="https://github.com/huggingface/transformers/pull/22748">Generate: handle text conditioning with multimodal encoder-decoder models by gante · Pull Request #22748 · huggingface/transformers</a>: What does this PR do? Consolidates decoder_input_ids preparation changes in a single place, for all future multimodal encoder-decoder models on PT and TF. In a nutshell, this PR generalizes the fol...</li><li><a href="https://github.com/huggingface/transformers/pull/22955">Generate: Add exception path for Donut by gante · Pull Request #22955 · huggingface/transformers</a>: What does this PR do? The multimodal generalization added in #22748 added a regression Donut -- Donut is never expecting a BOS token, having a task-specific token in its place. This PR adds an exce...</li><li><a href="https://github.com/clovaai/donut/blob/master/donut/model.py#L210">donut/donut/model.py at master · clovaai/donut</a>: Official Implementation of OCR-free Document Understanding Transformer (Donut) and Synthetic Document Generator (SynthDoG), ECCV 2022 - clovaai/donut</li><li><a href="https://github.com/clovaai/donut/blob/master/donut/model.py#L468">donut/donut/model.py at master · clovaai/donut</a>: Official Implementation of OCR-free Document Understanding Transformer (Donut) and Synthetic Document Generator (SynthDoG), ECCV 2022 - clovaai/donut</li><li><a href="https://github.com/huggingface/transformers/releases/tag/v4.29.0">Release v4.29.0: Transformers Agents, SAM, RWKV, FocalNet, OpenLLaMa · huggingface/transformers</a>: Transformers Agents Transformers Agent is a new API that lets you use the library and Diffusers by prompting an agent (which is a large language model) in natural language. That agent will then out...</li><li><a href="https://github.com/huggingface/transformers/compare/v4.28.1...v4.29.0">Comparing v4.28.1...v4.29.0 · huggingface/transformers</a>: 🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX. - Comparing v4.28.1...v4.29.0 · huggingface/transformers
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1265285987924316283)** (1 messages): 

> - `Background removal`
> - `Segmentation`
> - `Diffusion models` 


- **Seeking Guidance on Background Removal**: A member requested assistance for implementing **background removal** using **diffusion models** and segmentation techniques.
   - They asked if anyone could provide guidance on the best approaches or resources available to get started.
- **Interest in Segmentation Techniques**: Another member expressed interest in **segmentation techniques** that effectively work alongside **background removal** using diffusion models.
   - They inquired if there are specific models or methods that others have found successful.


  

---



### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1265178267250593825)** (2 messages): 

> - `Magpie Paper`
> - `Nous Research AI`
> - `Instruction Generation Techniques` 


- **Discussion on Magpie Paper's Utility**: A member inquired whether the insights from the [Magpie paper](https://arxiv.org/abs/2406.08464) represent a useful technique or merely a *party trick*.
   - They expressed curiosity about the **quality** and **diversity** of the generated instructions.
- **Nous Research Authors and Collaborations**: The authors of a notable paper include [Jaden Fiotto-Kaufman](https://arxiv.org/search/cs?searchtype=author&query=Fiotto-Kaufman,+J), [Alexander R Loftus](https://arxiv.org/search/cs?searchtype=author&query=Loftus,+A+R), among others.
   - The collaborative effort showcases a range of expertise contributing to the ongoing AI discourse.



**Link mentioned**: <a href="https://arxiv.org/abs/2407.14561">NNsight and NDIF: Democratizing Access to Foundation Model Internals</a>: The enormous scale of state-of-the-art foundation models has limited their accessibility to scientists, because customized experiments at large model sizes require costly hardware and complex engineer...

  

---


### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1265052200615284797)** (9 messages🔥): 

> - `ReFT paper discussion`
> - `YouTube video on ReFT`
> - `Oxen AI community activity`
> - `PC Agent Demo`
> - `Emoji duplication in server` 


- **ReFT Paper Simplified**: A discussion was held featuring the lead author of the [ReFT paper](https://www.reddit.com/r/MachineLearning/comments/1e8qwnl/r_discussion_of_reft_paper_with_lead_author/) explaining its efficient fine-tuning technique, which is 15x-60x more parameter-efficient than LoRA.
   - The method operates on the *residual stream*, making it flexible to combine various training tasks with learned parameters called 'interventions'.
- **How ReFT Works YouTube Video Released**: A new YouTube video titled [How ReFT Works w/ Author Zhengxuan Wu](https://www.youtube.com/watch?v=to2oKwnknUk&t=2770s) dives into the ReFT paper and its implications for machine learning.
   - The video presents an engaging explanation from the author alongside Greg, connecting dots with other papers discussed in prior Paper Clubs.
- **Oxen AI Community in Action**: The [Oxen AI community](https://oxen.ai/community) is actively growing, focusing on advancements in ML and AI through weekly discussions on various research papers.
   - Participants can subscribe to future Paper Club calendar invites to engage with both academic researchers and developers.
- **PC Agent Demo Unveiled**: A link was shared to a YouTube video titled [PC Agent Demo](https://youtu.be/97tUynaJusY?si=pi-K8F4trJDE3Kt8) detailing the PC Agent's functionality.
   - The description links to further resources about the demo, indicating ongoing innovations in this domain.
- **Emoji Duplication Blamed on User**: A member questioned the abundance of duplicate emojis on the server, leading to another member suggesting it's the fault of a specific user.
   - This lighter exchange highlights community interactions amid serious discussions on machine learning topics.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=to2oKwnknUk&t=2770s">How ReFT Works w/ Author Zhengxuan Wu</a>: We dive into the ReFT paper from Stanford with one of the authors Zhengxuan Wu. --Use Oxen AI 🐂           https://oxen.ai/Oxen AI makes versioning your data...</li><li><a href="https://youtu.be/97tUynaJusY?si=pi-K8F4trJDE3Kt8">PC Agent Demo</a>: gate-app.com/research/pc-agent</li><li><a href="https://www.reddit.com/r/MachineLearning/comments/1e8qwnl/r_discussion_of_reft_paper_with_lead_author/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://oxen.ai/community">Community Resources | Oxen.ai</a>: Manage your machine learning datasets with Oxen AI.
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1265312427541794880)** (62 messages🔥🔥): 

> - `Bud-E Voice Assistant`
> - `Llama 3.1 Models`
> - `Synthetic Dataset Creation`
> - `Graph RAG by Microsoft`
> - `DSPy Python Library` 


- **Bud-E Voice Assistant Gains Momentum**: A demo of the **Bud-E voice assistant** showcases its potential for accessibility and open-source adaptations, with the code base currently optimized for **Ubuntu** laptops.
   - Daily hackathon meetings are hosted by Christoph to onboard new volunteers and coordinate project efforts, enabling community contributions.
- **Llama 3.1 Breaks Ground for Open AI Models**: The **Llama 3.1 405B** model is described as the largest open-source model, offering capabilities that rival top closed-source alternatives while being accessible for commercial and research use.
   - Developers can leverage its functionalities for tasks such as **synthetic data generation** and model improvement, though operational costs are high.
- **Discussion on Synthetic Dataset Creation**: Concerns were raised about the costs associated with the **Llama 3.1-405B** for creating synthetic datasets, prompting inquiry about the viability of using the **70B model** instead.
   - While the **70B model** is considered sufficient for many tasks, its cost-effectiveness in dataset creation remains a critical discussion point.
- **Microsoft's Graph RAG Proposal**: Microsoft introduced **GraphRAG**, a method aimed at enhancing LLMs by integrating them with private datasets for semantic understanding and clustering.
   - This approach seeks to advance the capabilities of LLMs in analyzing data less familiar to them by utilizing knowledge graphs for better contextual answers.
- **Launch of DSPy Python Library**: A new **Python library** developed for integrating with **DSPy** optimizers claims to enhance evaluation metrics in AI applications significantly.
   - The library facilitates easy integration into existing apps, allowing developers to optimize their systems effectively and encouraging community engagement on social media.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/jail-right-to-jail-right-away-parks-and-recreation-parks-and-rec-gif-16177531">Jail Right To Jail GIF - Jail Right To Jail Right Away - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://ai.meta.com/blog/meta-llama-3-1/">no title found</a>: no description found</li><li><a href="https://news.ycombinator.com/item?id=41013693">no title found</a>: no description found</li><li><a href="https://youtu.be/7EYifjGAbg0">Cheevly getting started tutorial (part 1)</a>: Part 1 of getting started with Cheevly.</li><li><a href="https://github.com/stanfordnlp/dspy">GitHub - stanfordnlp/dspy: DSPy: The framework for programming—not prompting—foundation models</a>: DSPy: The framework for programming—not prompting—foundation models - stanfordnlp/dspy</li><li><a href="http://groq.link/llama3405bblog">Now Available on Groq: The Largest and Most Capable Openly Available Foundation Model to Date, Llama 3.1 405B - Groq is Fast AI Inference</a>: The largest openly available foundation model to date, Llama 3.1 405B, is now available on Groq. Groq is proud to partner on this key industry launch making</li><li><a href="https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct">meta-llama/Meta-Llama-3.1-8B-Instruct · Hugging Face</a>: no description found</li><li><a href="https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE">llama-models/models/llama3_1/LICENSE at main · meta-llama/llama-models</a>: Utilities intended for use with Llama models. Contribute to meta-llama/llama-models development by creating an account on GitHub.</li><li><a href="https://youtu.be/O4IXfa8CROs">BUD-E - Demo</a>: Join our Discord Community, try BUD-E yourself &amp; help us to build the voice assistant me and BUD-E talk about in the video:https://discord.gg/sTKSB2AwBvhttps...</li><li><a href="https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/">GraphRAG: A new approach for discovery using complex information</a>: Microsoft is transforming retrieval-augmented generation with GraphRAG, using LLM-generated knowledge graphs to significantly improve Q&amp;A when analyzing complex information and consistently outper...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1265022548857327668)** (489 messages🔥🔥🔥): 

> - `Llama 3.1 Performance`
> - `Quantization and Fine-tuning`
> - `Tool Calling Methods`
> - `Model Inference and Evaluation`
> - `Open Source Licensing` 


- **Llama 3.1 outperforms competitors on benchmarks**: Llama 3.1 405B Instruct-Turbo ranked 1st on GSM8K and performed closely to GPT-4o and Sonnet 3.5 on logical reasoning tasks like ZebraLogic.
   - However, it showed weaker performance on MMLU-Redux, suggesting mixed results across different datasets.
- **Concerns over Fine-tuning of Llama 3.1**: There are worries that the base model's alignment during pre-training negatively impacts fine-tuning effectiveness, which could lead to poor results.
   - Experts hope future fine-tuning efforts will yield better performance as users adapt training techniques.
- **Discussion on Tool Calling Mechanisms**: There is ongoing conversation about how Llama 3.1 manages tool calls, with speculation that its internal handling may not align with user expectations.
   - The tool calling method is compared across various frameworks, raising questions about compatibility with existing tools.
- **Inferences and Performance of Llama 3.1**: Users report impressive inference speeds with the 8B quantized model, drawing comparisons to GPT-4o.
   - This rapid performance is essential for applications requiring extensive parallel generation.
- **Open Source Licensing Changes**: Meta has made changes to the licensing of Llama 3.1, allowing the outputs to improve other models, marking a shift in their open-source strategy.
   - This aims to foster innovation without restricting developers to use their models exclusively.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/casper_h">Tweet from undefined</a>: no description found</li><li><a href="https://livebench.ai">LiveBench</a>: no description found</li><li><a href="https://en.wikipedia.org/wiki/Half-precision_floating-point_format">Half-precision floating-point format - Wikipedia</a>: no description found</li><li><a href="https://huggingface.co/qresearch/llama-3.1-8B-vision-378">qresearch/llama-3.1-8B-vision-378 · Hugging Face</a>: no description found</li><li><a href="https://x.com/billyuchenlin/status/1815841947468353700">Tweet from Bill Yuchen Lin 🤖 (@billyuchenlin)</a>: A quick independent evaluation of Llama-3.1-405B-Instruct-Turbo (on @togethercompute) ⬇️  1️⃣ It ranks 1st on GSM8K! 2️⃣ Its logical reasoning ability on ZebraLogic is quite similar to Sonnet 3.5, and...</li><li><a href="https://huggingface.co/collections/hugging-quants/llama-31-gptq-awq-and-bnb-quants-669fa7f50f6e713fd54bd198">Llama 3.1 GPTQ, AWQ, and BNB Quants - a hugging-quants Collection</a>: no description found</li><li><a href="https://huggingface.co/Salesforce/xLAM-7b-fc-r">Salesforce/xLAM-7b-fc-r · Hugging Face</a>: no description found</li><li><a href="https://tenor.com/view/maya-rudolph-ho-raise-the-roof-excited-comedy-gif-9228610">Maya Rudolph Ho GIF - Maya Rudolph Ho Raise The Roof - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/well-no-randy-marsh-south-park-s7e12-all-about-mormons-gif-22233922">Well No Randy Marsh GIF - Well No Randy Marsh South Park - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/meta-llama/Llama-Guard-3-8B-INT8">meta-llama/Llama-Guard-3-8B-INT8 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/SillyTilly/Meta-Llama-3.1-70B">SillyTilly/Meta-Llama-3.1-70B · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/mlx-community/Meta-Llama-3-70B-Instruct-4bit">mlx-community/Meta-Llama-3-70B-Instruct-4bit · Hugging Face</a>: no description found</li><li><a href="https://api.together.xyz/playground/chat/meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo">no title found</a>: no description found</li><li><a href="https://x.com/casper_hansen_/status/1815769985861493056">Tweet from Casper Hansen (@casper_hansen_)</a>: AWQ models of Llama 3.1 are done and uploaded ✅  Should run in vLLM out of the box!  Links below👇👇👇</li><li><a href="https://github.com/meta-llama/llama-toolchain/blob/9fb50bbd99b1dcf8f85c269cef5cb0bb48266964/llama_toolchain/inference/inference.py#L68">llama-toolchain/llama_toolchain/inference/inference.py at 9fb50bbd99b1dcf8f85c269cef5cb0bb48266964 · meta-llama/llama-toolchain</a>: Model components of the Llama Stack APIs. Contribute to meta-llama/llama-toolchain development by creating an account on GitHub.</li><li><a href="https://huggingface.co/meta-llama/Meta-Llama-3.1-405B-Instruct">meta-llama/Meta-Llama-3.1-405B-Instruct · Hugging Face</a>: no description found</li><li><a href="https://ai.meta.com/blog/meta-llama-3-1/">no title found</a>: no description found</li><li><a href="https://github.com/exo-explore/exo">GitHub - exo-explore/exo: Run your own AI cluster at home with everyday devices 📱💻 🖥️⌚</a>: Run your own AI cluster at home with everyday devices 📱💻 🖥️⌚ - exo-explore/exo</li><li><a href="https://www.deepspeed.ai/docs/config-json/">DeepSpeed Configuration JSON</a>: DeepSpeed is a deep learning optimization library that makes distributed training easy, efficient, and effective.</li><li><a href="https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct">meta-llama/Meta-Llama-3.1-8B-Instruct · Hugging Face</a>: no description found</li><li><a href="https://x.com/casper_hansen_/status/1815764551821856833">Tweet from Casper Hansen (@casper_hansen_)</a>: Llama 3.1 is out! I got download link, but need Huggingface format. Anyone got HF link that works?</li><li><a href="https://ai.meta.com/blog/meta-llama-3-1">no title found</a>: no description found</li><li><a href="https://llama.meta.com/">Llama 3.1</a>: The open source AI model you can fine-tune, distill and deploy anywhere. Our latest models are available in 8B, 70B, and 405B variants.</li><li><a href="https://github.com/meta-llama/llama-agentic-system">GitHub - meta-llama/llama-agentic-system: Agentic components of the Llama Stack APIs</a>: Agentic components of the Llama Stack APIs. Contribute to meta-llama/llama-agentic-system development by creating an account on GitHub.</li><li><a href="https://x.com/terryyuezhuo/status/1815796835677790539">Tweet from Terry Yue Zhuo (@terryyuezhuo)</a>: Preliminary results of Llama-3.1-405b-instruct on BigCodeBench-Hard via the @nvidia API:  Complete: 30.4 Instruct: 22.3 Average: 26.4  Better than Claude-3-Opus @AnthropicAI and close to GPT-4o @OpenA...</li><li><a href="https://x.com/astonzhangAZ/status/1815763885380747422">Tweet from Aston Zhang (@astonzhangAZ)</a>: Our Llama 3.1 405B is now openly available! After a year of dedicated effort, from project planning to launch reviews, we are thrilled to open-source the Llama 3 herd of models and share our findings ...</li><li><a href="https://github.com/meta-llama/llama-agentic-system/blob/main/custom_tools/base.py">llama-agentic-system/custom_tools/base.py at main · meta-llama/llama-agentic-system</a>: Agentic components of the Llama Stack APIs. Contribute to meta-llama/llama-agentic-system development by creating an account on GitHub.</li><li><a href="https://github.com/meta-llama/llama-toolchain">GitHub - meta-llama/llama-toolchain: Model components of the Llama Stack APIs</a>: Model components of the Llama Stack APIs. Contribute to meta-llama/llama-toolchain development by creating an account on GitHub.</li><li><a href="https://tenor.com/twuK.gif">Cat Keyboard GIF - Cat Keyboard Cats - Discover &amp; Share GIFs</a>: Click to view the GIF
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1265125784075370658)** (18 messages🔥): 

> - `Training larger bitnet models`
> - `Differences in model fine-tuning`
> - `Fine-tuning Llama 3.0`
> - `Multilingual fine-tuning resources` 


- **Interest in training larger bitnet models**: Members discussed the potential for training a **1.58 bitnet** with more parameters, noting a lack of comparable models to **Llama** on Hugging Face.
   - One member mentioned finding a smaller model on **Nous**, but expressed curiosity about larger parameter counts.
- **Debate on Qwen model differences**: A member speculated that the improvements from **Qwen2** to **Qwen1.5** might stem from a better base model rather than just different finetuning techniques.
   - Another questioned the relevance of benchmarks for base models in evaluating changes, particularly in light of low benchmark results from **Mistral Nemo** and **Llama-3-8b**.
- **Challenges in fine-tuning Llama 3.0**: With the release of **Llama 3 405b**, members acknowledged the significant challenges in fine-tuning this model, particularly concerned about practical execution outside of **Lora FTing**.
   - One member expressed hope that this might push for successful implementations of **DoRA fine-tuning** in open-source software.
- **Fine-tuning resources for Pashto language**: A member sought resources for fine-tuning models specifically for the **Pashto language**, emphasizing the scarcity of available materials despite its speaker base of **60 million**.
   - Another suggested exploring recent research, pointing to **Aya23 model papers** as a potential resource for guidance.
- **Collaboration on multilingual tasks**: A member inquired about collaborations for multilingual fine-tuning efforts, with one mentioning that **coheres** is undertaking significant work in this area.
   - Discussions also touched on the logistics of fine-tuning initiatives with high computational needs, like the numerous **H100s** used by a team.


  

---


### **Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1265220957560377374)** (4 messages): 

> - `Kuzu Graph Database`
> - `GraphRAG and Outlines`
> - `Entity Deduplication Techniques`
> - `Property Graph Index`
> - `Duplicate Detection in Graph Databases` 


- **Kuzu Graph Database recommended for integration**: A member recommended trying the **Kuzu** GraphStore, which has an [MIT license](https://github.com/kuzudb/kuzu) and integration with **LlamaIndex** for knowledge graphs.
   - This could offer a promising alternative for users looking for enhanced GraphStore functionalities.
- **GraphRAG's outline feature to enhance outputs**: Discussion on **GraphRAG** highlighted the potential of using outlines to constrain outputs, which could assist in deduplication tasks.
   - Integrating this feature could streamline workflows by reducing redundancy in data outputs.
- **Entity Deduplication by Tomaz Bratanic**: For **entity deduplication**, references were made to **Tomaz Bratanic**, who has explored this topic in-depth, along with a shared [blog post](https://neo4j.com/developer-blog/property-graph-index-llamaindex/#deduplication).
   - The approach involves combining text embedding similarity with word distance to identify and merge duplicates via **Cypher queries**.
- **Property Graph Index Enhancements**: The **Property Graph Index** is seen as a valuable upgrade for **LlamaIndex**, now featuring a proper property graph structure that enhances data representation.
   - This change allows for more detailed node labeling and property storage compared to the previous triple representation.
- **Atlas's duplicate detection capabilities**: Another member stated that **Atlas** also offers duplicate detection features, indicating a competitive landscape for graph databases.
   - While it may require some data preprocessing, the duplicate detection functionality is reported as decent.



**Link mentioned**: <a href="https://neo4j.com/developer-blog/property-graph-index-llamaindex/#deduplication">Customizing Property Graph Index in LlamaIndex</a>: Learn how to perform entity deduplication and custom retrieval methods using LlamaIndex to increase GraphRAG accuracy.

  

---


### **Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/)** (1 messages): 

jmiles38: <@414158939555364865> are you a contributor to worldsim/world client?
  

---


### **Nous Research AI ▷ #[reasoning-tasks-master-list](https://discord.com/channels/1053877538025386074/1264666760972472481/1265026584201531473)** (74 messages🔥🔥): 

> - `Open Reasoning Tasks`
> - `Schema and Formatting Improvements`
> - `Reasoning Techniques and Tools`
> - `Master List for Reasoning Papers`
> - `SMT Solvers for Reasoning` 


- **Exploring Open Reasoning Tasks Framework**: Discussion centered around improving the structure and aesthetic of the Open Reasoning Tasks repository, with suggestions for a master list format that differentiates tasks and includes examples.
   - Proposals for input structures included using headlined markdown formats and tables for example outputs, balancing clarity and usability for contributors.
- **Incorporating Multi-modal Tasks**: Participants deliberated on how to handle multi-turn tasks and various modalities, considering whether to utilize tables for structured inputs while ensuring flexibility for contributors.
   - The idea of excluding complicated tasks from table requirements while allowing contributors discretion was put forward.
- **Collaboration and Future Contributions**: Team members expressed intentions to contribute to the repository with updates and improvements, while confirming ongoing discussions sparked by shared papers.
   - References to outside resources, particularly in Bayesian reasoning and structured problem-solving techniques, were highlighted as valuable inputs for future development.
- **Developing a Master List for Reasoning Papers**: The possibility of creating a comprehensive master list for reasoning-related papers and resources was discussed, with input on structuring presentation for clarity.
   - Examples included potential headings and abstract formats, aiming to enhance accessibility for contributors and readers.
- **Utilizing SMT Solvers for Enhanced Reasoning**: A user mentioned the potential to leverage SMT solvers in translating word problems into SMTLIB formats, hinting at the creation of synthetic data for enhanced reasoning.
   - This approach aligns with recent discussions on integrating logical frameworks alongside LLMs to improve accuracy in reasoning applications.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/SMT_Solvers/status/1815856006427205672">Tweet from Chad Brewbaker (@SMT_Solvers)</a>: @halvarflake As I told @Teknium1 we can get a lot of reasoning via SMT solvers if we can teach the LLM to translate word problems from English/German to SMTLIB. A MADLIBS synthetic data problem if you...</li><li><a href="https://arxiv.org/abs/2402.06557">The Quantified Boolean Bayesian Network: Theory and Experiments with a Logical Graphical Model</a>: This paper introduces the Quantified Boolean Bayesian Network (QBBN), which provides a unified view of logical and probabilistic reasoning. The QBBN is meant to address a central problem with the Larg...</li><li><a href="https://x.com/swarnaNLP/status/1815430142870908971">Tweet from Swarnadeep Saha (@swarnaNLP)</a>: 🚨 New: my last PhD paper 🚨  Introducing System-1.x, a controllable planning framework with LLMs. It draws inspiration from Dual-Process Theory, which argues for the co-existence of fast/intuitive Sy...</li><li><a href="https://github.com/NousResearch/Open-Reasoning-Tasks/issues/5">Structuring suggestions · Issue #5 · NousResearch/Open-Reasoning-Tasks</a>: Hi, This is an amazing initiative! The idea of compiling a comprehensive list of potential reasoning tasks for language model evaluation is really valuable. I have a couple of suggestions: This is ...</li><li><a href="https://github.com/mlabonne/llm-course">GitHub - mlabonne/llm-course: Course to get into Large Language Models (LLMs) with roadmaps and Colab notebooks.</a>: Course to get into Large Language Models (LLMs) with roadmaps and Colab notebooks. - mlabonne/llm-course</li><li><a href="https://arxiv.org/abs/2401.14295">Demystifying Chains, Trees, and Graphs of Thoughts</a>: The field of natural language processing (NLP) has witnessed significant progress in recent years, with a notable focus on improving large language models&#39; (LLM) performance through innovative pro...
</li>
</ul>

</div>
  

---



### **LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1265028243950665728)** (197 messages🔥🔥): 

> - `LM Studio performance`
> - `Model downloads issues`
> - `Linux compatibility with GPU`
> - `Llama 3.1 capabilities`
> - `ROCm installation` 


- **Performance comparison between Llama models**: Users discussed the differences in performance between Llama 3.1 8B and 405B models, highlighting that running the larger models requires significant GPU resources.
   - *One user joked about needing a small nation's electricity supply* to power the GPU clusters needed for higher-capacity models.
- **Issues with downloading models**: Some users reported issues with downloading models, attributing it to DNS problems, with others noticing slowdowns due to increased traffic to Hugging Face from Llama 3.1 popularity.
   - A user speculated that their issues were caused by IPv6 and mentioned *wanting an option in the app to avoid using it* without affecting system-wide settings.
- **GPU detection problems on Linux**: New Linux users expressed difficulties with LM Studio detecting their GPUs, specifically mentioning issues with Radeon's RX5700XT on Linux Mint after a Windows transition.
   - One user noted they had installed extension packs but were still unable to get the system to recognize their GPU, questioning RDNA 1 support.
- **Discussion on model capabilities**: Users discussed the functional differences of various models, including mentions of Llama 3.1 supporting a range of languages and better performance in certain tasks.
   - One user noted that for Japanese, the 4o-mini model outperforms Llama 3.1, showing the importance of considering use-case specific models.
- **ROCm installation advice**: Advice was shared on manually installing ROCm for AMD GPUs to improve compatibility with LM Studio, particularly for users experiencing issues with their Radeon cards.
   - Users were directed to specific GitHub pages for installation instructions and troubleshooting tips.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lmstudio.ai">👾 LM Studio - Discover and run local LLMs</a>: Find, download, and experiment with local LLMs</li><li><a href="https://huggingface.co/YorkieOH10/Meta-Llama-3.1-8B-Instruct-hf-Q4_K_M-GGUF">YorkieOH10/Meta-Llama-3.1-8B-Instruct-hf-Q4_K_M-GGUF · Hugging Face</a>: no description found</li><li><a href="https://lmstudio.ai/snapdragon">👾 LM Studio - Discover and run local LLMs</a>: Find, download, and experiment with local LLMs</li><li><a href="https://huggingface.co/YorkieOH10/Meta-Llama-3.1-8B-Instruct-Q8_0-GGUF">YorkieOH10/Meta-Llama-3.1-8B-Instruct-Q8_0-GGUF · Hugging Face</a>: no description found</li><li><a href="https://ai.meta.com/blog/meta-llama-3-1/">no title found</a>: no description found</li><li><a href="https://github.com/lmstudio-ai/configs/blob/main/Extension-Pack-Instructions.md#amd-rocm">configs/Extension-Pack-Instructions.md at main · lmstudio-ai/configs</a>: LM Studio JSON configuration file format and a collection of example config files. - lmstudio-ai/configs</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/8650">Feature Request: Proper Llama 3.1 Support in llama.cpp · Issue #8650 · ggerganov/llama.cpp</a>: Prerequisites I am running the latest code. Mention the version if possible as well. I carefully followed the README.md. I searched using keywords relevant to my issue to make sure that I am creati...</li><li><a href="https://llama.meta.com/">Llama 3.1</a>: The open source AI model you can fine-tune, distill and deploy anywhere. Our latest models are available in 8B, 70B, and 405B variants.
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1265043935970201742)** (92 messages🔥🔥): 

> - `High Memory Usage of Qwen 2`
> - `LM Studio Model Compatibility`
> - `Meta-Llama Model Recommendations`
> - `Advancements in Gemini and Deepseek`
> - `LLM Compiler for Advanced Coding` 


- **High Memory Usage of Qwen 2**: A user reported very high memory usage when loading **Qwen 2 72B** using **llama.cpp**, which exceeds the model's size.
   - Another member suggested lowering the context length to help manage memory utilization.
- **LM Studio Model Compatibility**: A member noted compatibility issues with models in LM Studio, specifically **Meta-Llama 3.1-8B** and **70B**, with GPU offloading not working in the current version.
   - Others recommended upgrading to version **0.2.28** for better support, as updates to **llama.cpp** are pending.
- **Meta-Llama Model Recommendations**: There was a discussion about the **Meta-Llama 3.1** models, with varying opinions on their performance, particularly in reasoning tasks.
   - One noted that the **8B version** has poor logic but is still decent; another suggested looking into the **70B version** for improved output.
- **Advancements in Gemini and Deepseek**: The conversation touched on the performance of **Gemini Pro 1.5** and its suitability for coding tasks, highlighting its coding abilities but lack of writing capabilities.
   - Members anticipated updates to improve reasoning in upcoming models, particularly from **Deepseek**.
- **LLM Compiler for Advanced Coding**: A member recommended the **LLM Compiler**, built upon **Code Llama**, for tasks involving advanced coding concepts, mentioning it supports code optimization and compiler reasoning.
   - The model is available in **7B** and **13B** versions, fitting the specified memory capacity for users with limited VRAM.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.producthunt.com/posts/llama-7"> Llama - 3.1-405B: an open source model to rival GPT-4o / Claude-3.5 | Product Hunt</a>: Meta is releasing three models: The new 3.1-405B and upgrades to their smaller models: 3.1-70B and 3.1-8B. If 405B is as good as the benchmarks indicate, this would be the first time an open source mo...</li><li><a href="https://huggingface.co/collections/meta-llama/llama-31-669fc079a0c406a149a5738f">Llama 3.1 - a meta-llama Collection</a>: no description found</li><li><a href="https://x.com/YouJiacheng/status/1815817670954213710">Tweet from YouJiacheng (@YouJiacheng)</a>: just saw that deepseek-coder will get an upgrade at July 24 10:00 UTC+8.</li><li><a href="https://github.com/xcapt0/gpt2_chatbot">GitHub - xcapt0/gpt2_chatbot: ☕ GPT-2 chatbot for daily conversation</a>: ☕ GPT-2 chatbot for daily conversation. Contribute to xcapt0/gpt2_chatbot development by creating an account on GitHub.</li><li><a href="https://dubesor.de/benchtable">Dubesor LLM Benchmark table</a>: no description found
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1265039071646974127)** (1 messages): 

> - `Search Functionality` 


- **Search Functionality is Back!**: The issue affecting the search function in the app is now **RESOLVED** and users should be able to search again.
   - Apologies were made for the **inconvenience** caused during the downtime.
- **Increased Transparency on Resolutions**: The team has committed to providing updates regarding app issues, ensuring users are informed on resolved functionalities.
   - Users appreciated the prompt communication regarding the status of the search feature.


  

---


### **LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1265025042324914177)** (4 messages): 

> - `hf-mirror.com`
> - `Latex support for Llama 3.1 models` 


- **hf-mirror.com showcases promising potential**: A member introduced [hf-mirror.com](https://hf-mirror.com) as a mirror site for Hugging Face's API with its source code available on [GitHub](https://github.com/padeoe/hf-mirror-site), though it's currently in Chinese.
   - The site utilizes Caddy as a reverse proxy and offers a script [hfd.sh](https://gist.github.com/padeoe/697678ab8e528b85a2a7bddafea1fa4f) for resuming downloads, suggesting that LM Studio could greatly benefit from integrating these features for better user adaptability.
- **Latex support is on the horizon for Llama 3.1**: A member expressed enthusiasm for *huge* **Latex support** in the new Llama 3.1 models, highlighting its importance for users asking math and programming-related questions.
   - Another member confirmed that **Latex support** is coming soon, addressing the community's demand for enhanced mathematical capabilities.


  

---


### **LM Studio ▷ #[⚙-configs-discussion](https://discord.com/channels/1110598183144399058/1136793122941190258/1265238329943654410)** (12 messages🔥): 

> - `Llama 3 Configuration`
> - `GPU Settings`
> - `Roleplay Scenarios`
> - `Context Length Settings` 


- **Seeking Advice on Roleplay Setup for Llama 3**: A user is looking for guidance on setting up a roleplay scenario in LMStudio, trying to prevent the assistant from writing dialogue or actions for the user character.
   - *I'm just digging back into LMStudio, given the advances in Llama 3 based models.*
- **Configuration Settings for Llama 3.1**: A user requested suggestions for config values in Llama 3.1 as they are new to the setup.
   - Another member suggested using the v2 preset of Llama 3 after confirming they've updated to v0.2.28.
- **Context Length Recommendations**: Discussion revealed that **Llama 3** supports a context length up to **128k**, with advice to set it to **32k** for optimal GPU utilization.
   - A user inquired whether to leave the context length at **2048**, uncertain about prior increases.
- **GPU Compatibility Issues**: A user mentioned that Llama 3.1 did not seem to load fully into their GPU, specifically a **3080ti**.
   - After setting context length to max **(-1)**, the user noted it reverted upon reloading.


  

---


### **LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1265022201867010048)** (23 messages🔥): 

> - `Fine-tuning with 3090s`
> - `GGUF Fine-tuning Limitations`
> - `GPU Acceleration on RX 6700 XT`
> - `Quantized Model Fine-tuning`
> - `GPU Requirements for LLMs` 


- **Dual 3090s for Fine-tuning**: A member considers purchasing **two used 3090s** for fine-tuning, with another noting it's suitable for models up to **13b**, albeit slowly.
   - It's suggested to look into **renting GPUs** for custom model fine-tuning for better efficiency.
- **Challenges with Fine-tuning GGUF**: It's claimed that fine-tuning **GGUFs** is largely impossible, with one member stating it might yield poor results.
   - However, another points out that quantized LLMs can be fine-tuned, but the outcomes may corrupt the model's weights.
- **RX 6700 XT Lacks Support for GPU Acceleration**: A user inquired about GPU acceleration for the **RX 6700 XT** on Linux, and it was confirmed that it is not supported due to **OpenCL deprecation**.
   - Members highlighted that the RX 6700 XT doesn't support **ROCM**, further limiting its capabilities.
- **Quantized Model Fine-tuning Insights**: Discussion emerged around the viability of fine-tuning **quantized models** with methods like **unsloth/QLora**, albeit with potential issues.
   - Members clarified that supported quantized models are typically **bitsandbytes/awq quantized**, and GGUF is not supported.
- **GPU Requirements for LLM Execution**: It was noted that to achieve **GPU acceleration** for LLMs, a compatible NVIDIA GPU is preferred over AMD models such as the RX series.
   - Members referenced the **LM Studio** site for guidance on supported hardware, emphasizing that NVIDIA GPUs 'Just Work'.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lmstudio.ai/">👾 LM Studio - Discover and run local LLMs</a>: Find, download, and experiment with local LLMs</li><li><a href="https://learn.microsoft.com/en-us/ai/playbook/technology-guidance/generative-ai/working-with-llms/fine-tuning">Getting started with LLM fine-tuning</a>: Large Language Model (LLM) Fine-tuning is the process of adapting the pre-trained model to specific tasks. This process is done by updating its parameters on a new dataset. Specifically, the LLM is pa...
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1265124756294991943)** (112 messages🔥🔥): 

> - `Beta UI Improvements`
> - `Feedback on Model Loading`
> - `User Experience Concerns`
> - `Issues with GPU Usage`
> - `Beta Testing Process` 


- **Beta UI receives mixed feedback**: Users appreciate the new UI [found in Beta 1](https://lmstudio.ai) for its simplicity, yet some feel that important functionalities have been hidden behind too many tabs and menus.
   - Some users argue that the interface needs to retain advanced settings for users who want deeper customization.
- **Model loading parameters create confusion**: Several users reported difficulties finding and utilizing model loading parameters like Batch size and GPU offloading settings in the new interface.
   - Feedback mentions that features like *mmap* have been added, which may not have been clear initially for those accustomed to previous versions.
- **GPU auto settings fail to utilize hardware effectively**: Users noted that setting GPU layers to auto does not utilize the available GPU effectively, particularly on platforms with high-performance GPUs such as the4080S.
   - Manual settings appear to work better for GPU usage, raising questions about how the automatic feature is supposed to function.
- **Beta Testing Process and Feedback Handling**: The community emphasizes the importance of feedback during beta testing, with some users actively encouraging others to report bugs or suggestions.
   - Participants express appreciation for earlier bug fixes and encourage further transparency regarding the continued development of LM Studio.
- **Clarifications on System Settings and Limitations**: Some users sought clarity on why certain system resource limits exist, such as a restriction to 8 CPU threads, particularly for higher-end systems.
   - Others have shared their experiences with the new features, acknowledging initial misunderstandings due to the redesign of functionalities.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lmstudio.ai/blog/lms#bootstrap-lms-on--your-system">Introducing `lms` - LM Studio&#x27;s companion cli tool | LM Studio</a>: Today, alongside LM Studio 0.2.22, we&#x27;re releasing the first version of lms — LM Studio&#x27;s companion cli tool.</li><li><a href="https://forms.gle/kDYvduhQmDZmeKkG7">LM Studio 0.3.0 - Private Beta Sign Up</a>: Thanks for your interest in helping out test our upcoming release.   LM Studio 0.3.0 is gem-packed with new features and we&#39;d love your help to shake out the bugs before sending it out to the worl...
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1265061480605417482)** (5 messages): 

> - `ROCm 0.2.28 performance issues`
> - `Llama 3.1 compatibility with AMD cards` 


- **ROCm 0.2.28 Slows Inferencing Down**: After updating to **0.2.28 ROCm preview**, a user noticed significant slowdown in inference performance, with only one **7900XT** card showing **150w** usage instead of the usual **300w** on both cards.
   - The user reverted to **0.2.27**, which restored performance, and asked others to investigate what changed in **inference** for **0.2.28**.
- **Llama 3.1 Needs Tokenizer Fix for AMD**: A user expressed interest in getting **Llama 3.1** running on AMD cards but mentioned that **llama.cpp** does not recognize **smaug-bpe** as a tokenizer.
   - They highlighted this issue as a challenge that needs addressing for compatibility with AMD hardware.


  

---


### **LM Studio ▷ #[model-announcements](https://discord.com/channels/1110598183144399058/1225909444727013466/1265343880698662983)** (1 messages): 

> - `Llama 3.1`
> - `longer context improvements` 


- **Llama 3.1 Launch Brings Exciting Updates**: **Llama 3.1** is now available, with the **8B model** currently live on the [Hugging Face page](https://huggingface.co/lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF) for download.
   - Users are encouraged to try it out, as it features **massive improvements** over the initial release, particularly regarding **longer context support** up to **128k**.
- **Encouragement to Download Llama 3.1**: The message highlights the need to **download Llama 3.1** now to experience its enhancements.
   - With its **improved performance** for longer contexts, it's a strong recommendation for users to get involved.


  

---


### **LM Studio ▷ #[🛠-dev-chat](https://discord.com/channels/1110598183144399058/1234988891153629205/1265242937621479435)** (4 messages): 

> - `Mistral download issues`
> - `VPN connectivity problems`
> - `LLM model for grading`
> - `CHROMA data usage` 


- **Mistral Download Fails with VPN**: A member is experiencing **download failures** for **Mistral** in LM Studio while connected through a VPN on a remote desktop.
   - *Proxies are known not to work with the model explorer,* making it a challenge to resolve the issue.
- **Using LLM for Grading**: One user is developing an **LLM model** for grading, utilizing an answer file and a document file to constrain the bot's responses.
   - They expressed confusion about how to effectively use the data entered into **CHROMA** for this purpose.


  

---



### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1265397562408435742)** (1 messages): 

> - `Llama 3.1 405B`
> - `Perplexity mobile apps` 


- **Llama 3.1 405B Launches on Perplexity**: The **Llama 3.1 405B** model, touted as the most capable open-source model, is now available on Perplexity, rivaling **GPT-4o** and **Claude Sonnet 3.5**.
   - This launch signifies a significant enhancement in the capabilities available on the platform.
- **Upcoming Mobile Integration for Llama 3.1**: Perplexity is working on adding **Llama 3.1 405B** to their mobile applications next, promising seamless access to this advanced model.
   - Users are encouraged to stay tuned for updates as development progresses.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1265022026515480588)** (273 messages🔥🔥): 

> - `Performance of Llama 3.1 405B`
> - `Comparison of Llama 3.1 405B and Claude 3.5 Sonnet`
> - `Perplexity AI features and issues`
> - `Feedback on AI responses`
> - `API and usage experiences` 


- **Llama 3.1 405B performance concerns**: Users expressed dissatisfaction with the performance of **Llama 3.1 405B**, stating it often repeats answers and doesn't handle prompts effectively, particularly with Asian symbols.
   - Many are considering switching back to Claude 3.5 Sonnet for better speed and performance.
- **Comparative evaluation of AI models**: Some users believe that although **Llama 3.1 405B** is a significant advancement for open-source AI, it may not outperform Claude 3.5 in coding tasks.
   - Others noted that Sonnet 3.5 still excels in speed and better handles coding inquiries compared to Llama.
- **Issues with Perplexity AI functionality**: There were reports of **Llama 3.1 405B** not working properly on Perplexity AI, leading to queries about its status and stability across different platforms.
   - Users suggested waiting for a few days to assess performance, as previous models improved after initial launch.
- **Feedback on AI responses**: Several users commented on the models' inability to understand or generate certain symbols correctly, resulting in mixed reviews.
   - Feedback indicates that while Llama can simplify concepts, its overall functionality may lag behind competitors like Claude.
- **API usage experiences**: Users discussed the differences in experiences across different providers, noting that AWS and Fireworks had specific issues with the new version of Llama.
   - It was mentioned that accessing models through the Perplexity AI platform may vary from other applications, with expectations for improvements over time.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://jason-mendoza.vercel.app/">
      Jason Mendoza - Fullstack Developer (Web &amp; Blockchain &amp; AI Tech)
    </a>: no description found</li><li><a href="https://x.com/rypearts/status/1815868829169328349?s=61">Tweet from Ryan Putnam (@RypeArts)</a>: ✧ 　 　 ✧ ˚ * 　 　.　 　　　　 　　 · · 　　 　 + ✧ 　　　 · 　 · ˚ . 𝓈𝓊𝓂𝓂ℯ𝓇 𝓋𝒾𝒷ℯ𝓈</li><li><a href="https://x.com/perplexity_ai/status/1603441221753372673?lang=en">Tweet from Perplexity (@perplexity_ai)</a>: Introducing Bird SQL, a Twitter search interface that is powered by Perplexity’s structured search engine. It uses OpenAI Codex to translate natural language into SQL, giving everyone the ability to n...</li><li><a href="https://scale.com/leaderboard">SEAL leaderboards</a>: no description found</li><li><a href="https://x.com/perplexity_ai/status/1815431484767142272?s=61">Tweet from Perplexity (@perplexity_ai)</a>: When you know, you know.</li><li><a href="https://tenor.com/view/cryptoflash-crypto-flash-tattoo-vintage-gif-27569875">Cryptoflash Tattoo GIF - Cryptoflash Crypto Flash - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.aboutamazon.com/news/aws/meta-llama-3-1-models-AWS-generative-ai">Llama 3.1 models from Meta are available on AWS for generative AI applications</a>: Meta’s most advanced large language models (LLMs) give customers more choices when building, deploying, and scaling generative AI applications.</li><li><a href="https://x.com/minchoi/status/1815812112796565690">Tweet from Min Choi (@minchoi)</a>: Instant Intelligence is wild with Llama 3.1 8B + Groq 🤯 </li><li><a href="https://tenor.com/bnLYV.gif">Balloons Up GIF - Balloons Up - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://build.nvidia.com/explore/discover#llama-3_1-405b-instruct">Try NVIDIA NIM APIs</a>: Experience the leading models to build enterprise generative AI apps now.</li><li><a href="https://lexfridman.com/aravind-srinivas-transcript/#:~:text=(01%3A46%3A42,by%20the%20way.">Transcript for Aravind Srinivas: Perplexity CEO on Future of AI, Search &amp; the Internet | Lex Fridman Podcast #434 - Lex Fridman</a>: This is a transcript of Lex Fridman Podcast #434 with Aravind Srinivas. The timestamps in the transcript are clickable links that take you directly to that point in the main video. Please note that th...
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1265065560614633613)** (12 messages🔥): 

> - `Dark Oxygen`
> - `Mercury's Diamonds`
> - `Beach-Cleaning Robots`
> - `Munger's Inversion Technique`
> - `Llama 3 Release` 


- **Dark Oxygen Discovery**: A discussion emerged about the recent discovery of **Dark Oxygen**, emphasizing its potential implications for atmospheric studies.
   - Members expressed curiosity about the **nature of Dark Oxygen** and its role in ecological balance.
- **Exploration of Mercury's Diamonds**: The chat highlighted findings about **Diamonds on Mercury**, sharing fascinating insights from current research.
   - Participants were intrigued by the **geological processes** that could lead to diamond formation on the planet.
- **Innovations in Beach-Cleaning Technology**: Beach-cleaning robots were a hot topic, showcasing **new robotic technologies** that target ocean pollution effectively.
   - The community discussed the potential impact of these robots on **marine ecosystems**, highlighting real-time data from trials.
- **Munger's Inversion Technique Explained**: A shared [YouTube video](https://www.youtube.com/embed/EtjBA3DGCrg) focused on **Munger's Inversion Technique**, detailing how it applies to decision-making.
   - Viewers were encouraged to **consider this technique** for better critical thinking in daily life.
- **Meta Releases Llama 3**: A noteworthy highlight involved Meta's release of **Llama 3**, generating buzz about its advanced capabilities.
   - The community discussed potential applications for **Llama 3** in various AI tasks and its implications for developers.



**Link mentioned**: <a href="https://www.youtube.com/embed/EtjBA3DGCrg">YouTube</a>: no description found

  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1265344369943384206)** (13 messages🔥): 

> - `Llama model updates`
> - `Perplexity API and DSGVO`
> - `Search site limitations` 


- **Inquiry about Llama 3.1 model in API**: Users expressed interest in adding the **Llama 3.1 405b** model to the Perplexity API, with some requesting details on its availability.
   - One user specifically asked, 'Are there plans to serve Llama 3 405b in the API?' which sparked follow-up queries.
- **Clarification on search methods using specific sites**: A user suggested utilizing `site:example.com` or `site:arxiv.org` for academic searches, indicating that it is possible to limit searches to specific domains.
   - However, they noted that a limitation exists where each request only retrieves results from **5 sources**.
- **Perplexity API's privacy compliance inquiry**: A user raised a question regarding whether the **Perplexity API** is DSGVO-ready, seeking clarity on its compliance with data protection regulations.
   - Another user shared a [link to the terms of service](https://www.perplexity.ai/hub/legal/perplexity-api-terms-of-service), mentioning that it referenced GDPR compliance.


  

---



### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1265049794116583554)** (282 messages🔥🔥): 

> - `Stable Diffusion models comparison`
> - `Training Lycoris and Loras`
> - `Community perceptions of Stable Diffusion`
> - `New developments in AI models`
> - `General discussions and inquiries` 


- **Ranking Current AI Models**: Users discussed their rankings of AI models, with **Kolors** being rated the highest, followed by **Auraflow**, **Pixart Sigma**, and **Hunyuan**.
   - Kolors is noted for its speed and performance, aligning with what users expected from **SD3**.
- **Training Lycoris with ComfyUI**: Discussion arose about the current capabilities for training Lycoris, mentioning tools like **Kohya-ss** and potential updates in **Onetrainer**.
   - Users expressed frustration with **Kohya-ss's** compatibility issues, specifically needing Python version 3.10.9 or higher.
- **Community's Sentiment towards Stable Diffusion**: Users expressed their views on the community's perception of **Stable Diffusion**, suggesting that recent criticisms may stem from misunderstandings regarding model licensing.
   - Some users pointed out the marketing strategies and perceived toxicity directed against **Stability AI**.
- **Updates in AI Sampling Techniques**: A new sampler node was introduced that implements Strong Stability Preserving Runge-Kutta and implicit variable step solvers, raising interest among users.
   - Users discussed the potential performance improvements these new methods could provide for AI models.
- **General Chat about AI and Personal Experiences**: Users shared their personal experiences with AI, such as learning new programming languages and discussing their health decisions impacting their focus.
   - Casual conversations took place surrounding the use of AI in various daily applications.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://m3.material.io/```">
   Material Design
  </a>: Build beautiful, usable products faster. Material Design is an adaptable system—backed by open-source code—that helps teams build high quality digital experiences.</li><li><a href="https://chat.tune.app/">Tune Chat - Chat app powered by open-source LLMS</a>: With Tune Chat, access Prompts library, Chat with PDF, and Brand Voice features to enhance your content writing and analysis and maintain a consistent tone across all your creations.</li><li><a href="https://huggingface.co/dataautogpt3/PixArt-Sigma-900M">dataautogpt3/PixArt-Sigma-900M · Hugging Face</a>: no description found</li><li><a href="https://llama.meta.com/">Llama 3.1</a>: The open source AI model you can fine-tune, distill and deploy anywhere. Our latest models are available in 8B, 70B, and 405B variants.</li><li><a href="https://youtu.be/ROCKGuuviis?feature=shared&t=33">Dennis reads Charlie&#39;s campaign speech</a>: From Season 2 episode 8 of It&#39;s Always Sunny in Philadelphia.</li><li><a href="https://tenor.com/view/jump-to-conclusion-think-again-go-wild-moot-no-gif-17140256">Jump To Conclusion Think Again GIF - Jump To Conclusion Think Again Go Wild - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://studio.tune.app/">TuneStudio</a>: no description found</li><li><a href="https://old.reddit.com/r/StableDiffusion/comments/1e8d4l3/new_twostage_pixart_ensemble_of_experts_2x_900m/">New two-stage PixArt ensemble of experts (2x 900M)</a>: Posted in r/StableDiffusion by u/terminusresearchorg • 98 points and 33 comments</li><li><a href="https://civitai.com/models/64471/real-mechanical-parts?modelVersionId=460254">Real Mechanical Parts - RealMech*Pony Alpha v1 | Stable Diffusion LoRA | Civitai</a>: About PONY XL - Real Mechanical Parts version!!!! It is important that you should know, it is on alpha stage and has lots of room to improve. I&#x27;ve ...
</li>
</ul>

</div>
  

---



### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1265336536107454466)** (41 messages🔥): 

> - `Llama 3 405B Launch`
> - `Model Performance Comparisons`
> - `OpenRouter Features Updates`
> - `Prompt Competition Announcement`
> - `DeepSeek Coder V2 Inference Provider` 


- **Llama 3 405B Launch Competitively Priced**: The **Llama 3 405B** has been launched, rivaling **GPT-4o** and **Claude 3.5 Sonnet** at **$3/M tokens** and offering an impressive **128K token context** for synthetic data generation.
   - Users expressed excitement, with comments like *'Damn! That's crazy, this is THE BEST open LLM now'* and *'what a leap'* highlighting its anticipated impact.
- **User Feedback on Model Performance**: Feedback arose on Llama 3 405B's performance, with one user noting it is *'worse than both gpt4o and not even comparable to claude 3.5'* in translation tasks.
   - Concerns were raised about the **70B version** producing *'gibberish'* after a few tokens, while **405B** was compared to **gemini 1.5 pro**.
- **Updates on OpenRouter Features**: OpenRouter announced new features including **Retroactive Invoices**, **custom keys**, and improvements to **the Playground**.
   - Users were encouraged to provide feedback on new offerings available at [OpenRouter Chat](http://openrouter.ai/chat) to enhance user experience.
- **Prompt Competition for Multiple Models**: A **Multi-LLM Prompt Competition** has been introduced with users invited to submit challenging prompts for **Llama 405B**, **GPT-4o**, and **Sonnet** for a chance to win 15 free credits.
   - The competition aims to test the limits of these models, as users eagerly await announcements detailing the outcomes.
- **DeepSeek Coder V2 Inference Provider**: Announced a new **private inference provider** for **DeepSeek Coder V2**, which operates with no input training.
   - Users can explore the new provider through [DeepSeek Coder](https://openrouter.ai/models/deepseek/deepseek-coder), enhancing OpenRouter's offerings.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="http://openrouter.ai/chat">Chatroom | OpenRouter</a>: LLM Chatroom is a multimodel chat interface. Add models and start chatting! Chatroom stores data locally in your browser.</li><li><a href="https://openrouter.ai/chat?models=meta-llama/llama-3.1-405b-instruct,openai/gpt-4o,anthropic/claude-3.5-sonnet">Chatroom | OpenRouter</a>: LLM Chatroom is a multimodel chat interface. Add models and start chatting! Chatroom stores data locally in your browser.</li><li><a href="https://x.com/OpenRouterAI/status/1815860614755147961">Tweet from OpenRouter (@OpenRouterAI)</a>: DeepSeek Coder V2 now has a private provider serving requests on OpenRouter, with no input training!  Check it out here: https://openrouter.ai/models/deepseek/deepseek-coder</li><li><a href="https://x.com/OpenRouterAI/status/1815837707505131699">Tweet from OpenRouter (@OpenRouterAI)</a>: 🏆 Multi-LLM Prompt Competition  Reply below with prompts that are tough for Llama 405B, GPT-4o, and Sonnet!  Winner gets 15 free credits ✨. Example:</li><li><a href="https://openrouter.ai/models/meta-llama/llama-3.1-405b-instruct">Meta: Llama 3.1 405B Instruct by meta-llama</a>: The highly anticipated 400B class of Llama3 is here! Clocking in at 128k context with impressive eval scores, the Meta AI team continues to push the frontier of open-source LLMs.  Meta&#x27;s latest c...</li><li><a href="https://openrouter.ai/models/meta-llama/llama-3.1-8b-instruct">Meta: Llama 3.1 8B Instruct by meta-llama</a>: Meta&#x27;s latest class of model (Llama 3.1) launched with a variety of sizes &amp; flavors. This 8B instruct-tuned version is fast and efficient.  It has demonstrated strong performance compared to ...</li><li><a href="https://openrouter.ai/models/meta-llama/llama-3.1-70b-instruct">Meta: Llama 3.1 70B Instruct by meta-llama</a>: Meta&#x27;s latest class of model (Llama 3.1) launched with a variety of sizes &amp; flavors. This 70B instruct-tuned version is optimized for high quality dialogue usecases.  It has demonstrated stro...
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1265021926913605692)** (190 messages🔥🔥): 

> - `Llama 405B Model Performance`
> - `Custom API Keys Integration`
> - `Comparison of Llama Models`
> - `Prompting Competition for Llama 405B`
> - `Fine-Tuning and Instruction Challenges` 


- **Llama 405B model shows strong capabilities**: Users discuss the performance of the new **Llama 405B model**, noting its impressive reasoning abilities, especially in English, although some mention it still falls short in foreign languages compared to models like **Claude** and **GPT-4**.
   - Some users find the model to produce nonsense responses, with varying experiences reported among different users.
- **Accessing Custom API Keys**: Discussion arose about the process to obtain **custom API keys per provider**, emphasizing that this integration could vary by provider and might involve specific account settings.
   - Users are eager to understand how to manage and utilize these keys effectively.
- **Comparison between Llama 3 and Llama 3.1**: Participants compare **Llama 3** (8B/70B) with **Llama 3.1**, highlighting that 3.1 is distilled from the larger **405B model** and offers improved context length limits of **128k** instead of 8k.
   - The new version is expected to perform better across various benchmarks.
- **Prompting Competition for Llama 405B**: Alex Atallah announced a **prompting competition** for Llama 405B, with the winner receiving **15 free credits**, focusing on prompts that challenge the model's capabilities.
   - Participants are curious about the criteria for the competition, particularly regarding what constitutes a tough prompt.
- **Challenges in using Instruction Models**: Several users reported bugs when using **instruct models**, specifically mentioning issues with calling JSON responses in multi-turn scenarios.
   - Participants are sharing code snippets and troubleshooting tips in an effort to resolve these challenges.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://embracethered.com/blog/posts/2024/chatgpt-gpt-4o-mini-instruction-hierarchie-bypasses/"> Breaking Instruction Hierarchy in OpenAI&#39;s gpt-4o-mini &middot;  Embrace The Red</a>: no description found</li><li><a href="https://x.com/elder_plinius/status/1815759810043752847">Tweet from Pliny the Prompter 🐉 (@elder_plinius)</a>: 🌩️ JAILBREAK ALERT 🌩️  META: PWNED 🦾😎 LLAMA-3-405B: LIBERATED 🦙💨  Come, witness the brand new SOTA open source AI outputting a home lab bioweapon guide, how to hack wifi, copyrighted lyrics, and...</li><li><a href="https://llama.meta.com/llama-downloads/">Download Llama</a>: Request access to Llama.</li><li><a href="https://tenor.com/view/the-shawshank-redemption-pie-finger-gif-23305361">The Shawshank GIF - The Shawshank Redemption - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://x.com/OpenRouterAI/status/1815837707505131699">Tweet from OpenRouter (@OpenRouterAI)</a>: 🏆 Multi-LLM Prompt Competition  Reply below with prompts that are tough for Llama 405B, GPT-4o, and Sonnet!  Winner gets 15 free credits ✨. Example:</li><li><a href="https://openrouter.ai/docs/integrations">Integrations (Beta) | OpenRouter</a>: Bring your own provider keys with OpenRouter</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1e9hg7g/azure_llama_31_benchmarks/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/vikyw89/llmtext">GitHub - vikyw89/llmtext: A simple llm library</a>: A simple llm library. Contribute to vikyw89/llmtext development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1265040534259634306)** (7 messages): 

> - `Register Allocation in Flash Attention`
> - `Kernel Fusion of Q, K, V Projections`
> - `Challenges with SVD Parallelization`
> - `Open Source GPU Kernel Modules` 


- **Register Allocation in Flash Attention concerns**: @vkaul11 inquired about the explicit allocation of registers in **Flash Attention**, expressing confusion about the use of registers alongside shared memory.
   - The question highlighted a need for clarity on efficiently managing register resources in CUDA programming.
- **Kernel Fusion Query on Q, K, V Projections**: A question arose regarding whether the initial projections of **Q, K, and V** matrices could be fused into a single kernel, with concerns over the feasibility given their large sizes.
   - This pointed to ongoing discussions around optimizing memory and processing requirements in neural network computations.
- **Parallelization Difficulties with SVD**: @danikhan632 noted that while **SVD** is challenging to parallelize, it is preferable over transferring data back to the **CPU**.
   - There was also interest in developing a **Triton kernel** for SVD, suggesting a potential community project for more optimized computation.
- **NVIDIA's Move to Open Source GPU Kernel Modules**: A link was shared regarding NVIDIA's transition to **open-source GPU kernel modules**, which began with the R515 driver in May 2022, supporting dual **GPL** and **MIT licensing**.
   - The update outlined improved performance and capabilities such as **heterogeneous memory management** along with a commitment to fully replace the closed-source driver.



**Link mentioned**: <a href="https://developer.nvidia.com/blog/nvidia-transitions-fully-towards-open-source-gpu-kernel-modules/">NVIDIA Transitions Fully Towards Open&#x2d;Source GPU Kernel Modules | NVIDIA Technical Blog</a>: With the R515 driver, NVIDIA released a set of Linux GPU kernel modules in May 2022 as open source with dual GPL and MIT licensing. The initial release targeted datacenter compute GPUs&#8230;

  

---


### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1265395795671384195)** (9 messages🔥): 

> - `torch.compile performance`
> - `Bert model inference issues`
> - `CUDA graphs usage`
> - `PyTorch profiler tools`
> - `Inductor configuration changes` 


- **torch.compile causes memory issues on small Bert model**: A member reported testing `torch.compile` for model inference on a small **Bert** model, observing significant RAM usage that forced a batch size reduction from **512** to **160**.
   - They found performance to be slower than using eager mode with the larger batch size, and model compiled successfully with `full_graph=True`, suggesting no architecture issues.
- **Questions on CUDA graphs utilization**: Another member inquired if **CUDA graphs** were in use and whether the latest nightlies were being utilized, indicating potential adjustments to improve performance.
   - They highlighted that this could impact the overall effectiveness of the `torch.compile` process and its memory implications.
- **Using PyTorch profiler for deeper insights**: To investigate further, a member recommended using the **PyTorch profiler** alongside the memory trace tool to analyze what might be happening under the hood.
   - This tool could provide valuable insights into memory usage patterns and inefficiencies during inference.
- **Inductor configuration inquiries**: A member asked if the inducer configuration was being altered or if `torch.compile` was being called with default settings.
   - Standard configurations combined with `inference_mode` errors may also contribute to the observed memory challenges.
- **No effect from different compilation modes**: The user confirmed that memory usage remained the same regardless of using `reduce-overhead` or `fullgraph` options in their compilation call.
   - This consistency suggests that other factors are likely influencing the memory consumption during inference.


  

---


### **CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1265325582766313482)** (17 messages🔥): 

> - `Meta Llama 3.1 Release`
> - `GPU Allocations`
> - `Multi-modal Features`
> - `VLM Capabilities`
> - `CUDA Performance` 


- **Meta Llama 3.1 focuses on text functionality**: Meta's latest release includes the **Llama 3.1 405B**, expanding context length to **128K** and supporting **eight languages**, as noted in [Zuckerberg’s letter](https://about.fb.com/news/2024/07/open-source-ai-is-the-path-forward/). However, the multi-modal parts are not included in this release, and members discussed this omission possibly being strategic ahead of earnings.
- **High demand for GPUs**: A member expressed frustration over the struggle to access a single **A100 GPU**, while another noted that **xAI** is utilizing a staggering **100,000 H100 GPUs**. The volume of available GPUs highlighted a stark contrast in resource access among users.
- **VLM capabilities under discussion**: Members acknowledged that only the text version of the model is available in this release, with VLM (Vision Language Model) features expected later. One member shared insights on their approach to achieving **50% accuracy** on ARC-AGI by leveraging **GPT-4o** for generating numerous Python implementations.
- **Feature engineering for improved results**: Discussion revolved around improving results through feature engineering rather than heavily relying on vision capabilities, highlighting a case where success was achieved by engineering the problem grid. One user mentioned utilizing additional techniques for optimizing performance with their method.
- **CUDA's future plans**: A member teased an upcoming CUDA release, stating they plan to outperform **cuBLAS** on various matrix sizes, specifically with **FP16/FP32** support. Conversations about **Nvidia's hardware intrinsics** for FP16 showcased excitement about the potential performance enhancements.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://ai.meta.com/blog/meta-llama-3-1">no title found</a>: no description found</li><li><a href="https://redwoodresearch.substack.com/p/getting-50-sota-on-arc-agi-with-gpt">Getting 50% (SoTA) on ARC-AGI with GPT-4o</a>: You can just draw more samples
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1265158276895608862)** (4 messages): 

> - `Performance of CUDA Kernels`
> - `Tiled Matrix Multiplication`
> - `Compute Intensity` 


- **ncu Output Interpretation**: A member asked if executing `ncu ./kernel` provides the speed or time taken for a CUDA kernel, noting the duration for a normal matrix multiplication is **10.30us** and for tiled multiplication is **9.18us**.
   - They expressed confusion as the performance improvement doesn't align with expectations from the pmpp textbook.
- **Limited Improvement from Tiling**: Another member shared their experience that transitioning from naive to tiled matrix multiplication didn't yield significant speed improvements, similar to kernel comparisons found in [this article](https://siboehm.com/articles/22/CUDA-MMM).
   - They noted that significant speedup is typically observed only with thread tiling, referencing kernel implementations 4 and 5 in the linked resource.
- **Importance of Compute Intensity**: A member emphasized that increasing compute intensity is crucial for achieving better performance, specifically to escape the left side of the roofline model.
   - They indicated that this would be the most impactful strategy at the beginning stages of optimizing CUDA kernels.


  

---


### **CUDA MODE ▷ #[hqq](https://discord.com/channels/1189498204333543425/1225499037516693574/)** (1 messages): 

iron_bound: neat https://github.com/AnswerDotAI/fsdp_qlora/tree/llama400b
  

---


### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1265020763904147518)** (182 messages🔥🔥): 

> - `Performance of LLMs`
> - `KV Caching Implementation`
> - `MuP vs Other Optimizations`
> - `Floating Point Precision Techniques`
> - `Training Stability Methods` 


- **Analyzing Performance Metrics**: Members discussed discrepancies in performance metrics between ZeRO-1 and ZeRO-2 during experiments, noting the potential benefits of stochastic rounding in ZeRO-2.
   - Initial tests on a 2 x 4060Ti system showed slight performance overhead due to additional communications.
- **KV Caching Achievements**: Progress on implementing KV caching logic for model inference was reported, with partial operations functioning correctly but needing efficiency improvements.
   - Tweaks to `matmul_cublaslt` and attention kernels were being explored to enhance computation without changing end results.
- **Insight on MuP vs Alternatives**: Discussion on the perceived performance differences of muP compared to other methodologies, indicating that muP could underperform in certain scenarios.
   - Members compared baseline optimizations, noting that muP was designed for better stability and results but may not always deliver on that promise.
- **Floating Point Precision Techniques**: The team explored the implications of using different floating point precisions (like BF16 and FP8) on model training performance and stability.
   - Concerns were raised about the challenges of maintaining stability with FP8 training due to the potential for underflows and overflows.
- **Improving Training Stability**: Members were interested in various techniques to enhance training stability such as using z-loss and soft clamping methods discussed in the latest literature.
   - It was noted that constructing a visual representation of tensor changes during training might aid in understanding and preventing instability.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2407.05872">Scaling Exponents Across Parameterizations and Optimizers</a>: Robust and effective scaling of models from small to large width typically requires the precise adjustment of many algorithmic and architectural details, such as parameterization and optimizer choices...</li><li><a href="https://arxiv.org/abs/2405.18710v1">To FP8 and Back Again: Quantifying the Effects of Reducing Precision on LLM Training Stability</a>: The massive computational costs associated with large language model (LLM) pretraining have spurred great interest in reduced-precision floating-point representations to accelerate the process. As a r...</li><li><a href="https://arxiv.org/abs/2406.16793">Adam-mini: Use Fewer Learning Rates To Gain More</a>: We propose Adam-mini, an optimizer that achieves on-par or better performance than AdamW with 45% to 50% less memory footprint. Adam-mini reduces memory by cutting down the learning rate resources in ...</li><li><a href="https://x.com/rosieyzh/status/1811790177246888075">Tweet from Rosie Zhao @ ICML (@rosieyzh)</a>: In our new work on evaluating optimizers for LLM training, we perform a series of experiments to investigate the role of adaptivity in optimizers like Adam in achieving good performance and stability....</li><li><a href="https://arxiv.org/abs/2407.07972">Deconstructing What Makes a Good Optimizer for Language Models</a>: Training language models becomes increasingly expensive with scale, prompting numerous attempts to improve optimization efficiency. Despite these efforts, the Adam optimizer remains the most widely us...</li><li><a href="https://github.com/microsoft/mup/issues/76">Not getting perf improvements from muP at ~1.5B scale · Issue #76 · microsoft/mup</a>: Hey guys, first of all thanks for the awesome work! I&#39;ve implemented muP in the llm.c project (see here), the coord checks seem to be flat / correct (I went up to 15 steps and still flat!) but I a...</li><li><a href="https://github.com/karpathy/llm.c/pull/707">Add KV cache for inference by gordicaleksa · Pull Request #707 · karpathy/llm.c</a>: WIP. Very ugly rn, experimenting. Will update description after the draft has progressed. :)</li><li><a href="https://github.com/karpathy/llm.c/blob/master/llmc/matmul.cuh#L134">llm.c/llmc/matmul.cuh at master · karpathy/llm.c</a>: LLM training in simple, raw C/CUDA. Contribute to karpathy/llm.c development by creating an account on GitHub.</li><li><a href="https://github.com/karpathy/llm.c/pull/708">Add high perf mode by gordicaleksa · Pull Request #708 · karpathy/llm.c</a>: Add:  Warnings when we take a suboptimal branch High perf mode that will exit immediately if we&#39;re not running using all of the most optimal branches  Also added a fwd kernel config that will be u...</li><li><a href="https://www.diffchecker.com/nbHQZode/">3.1 vs 3 - llama license - Diffchecker</a>: 3.1 vs 3 - llama license - META LLAMA 3 COMMUNITY LICENSE AGREEMENT Meta Llama 3 Version Release Date: April 18, 2024  “Agreeme</li><li><a href="https://github.com/karpathy/llm.c/pull/307">Improve tanh derivative in backward gelu by akbariyeh · Pull Request #307 · karpathy/llm.c</a>: It is cheaper to compute the derivative of tanh as 1 - tanh^2 than computing 1/(cosh^2). This will probably not make a measurable difference.
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1265165032480837712)** (6 messages): 

> - `Stable Diffusion on RX7900XTX`
> - `Flash Attention support for AMD ROCm` 


- **Stable Diffusion on RX7900XTX discussed**: A post was shared about accelerating inferencing on AMD RDNA3 GPUs with the [Composable Kernel library](https://github.com/ROCm/composable_kernel/discussions/1032) for **Stable Diffusion** on **RX7900XTX**.
   - The discussion is noted to be slightly outdated, providing insights into the ROCm5.7 capabilities.
- **Flash Attention now supports AMD ROCm**: The [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention/pull/1010) project has introduced support for AMD ROCm, stating it currently works with **mi200 & mi300** only.
   - This update is powered by the **Composable Kernel**, with details shared in a recent pull request.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/ROCm/composable_kernel/discussions/1032">Stable diffusion with RX7900XTX on ROCm5.7 · ROCm/composable_kernel · Discussion #1032</a>: Accelerate Inferencing on AMD RDNA3 GPUs with Composable Kernel library Hello, and welcome to the AMD RDNA3 GPU High Performance Inferencing blog post. In this blog post, we will discuss how to use...</li><li><a href="https://github.com/Dao-AILab/flash-attention/pull/1010">Support AMD ROCm on FlashAttention 2 by rocking5566 · Pull Request #1010 · Dao-AILab/flash-attention</a>: This PR implement the AMD / ROCm version of c++ flash api  mha_fwd mha_varlen_fwd mha_bwd mha_varlen_bwd   The kernel implementation comes from composable kernel The c++ api is same as original ver...
</li>
</ul>

</div>
  

---



### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1265024691765252178)** (196 messages🔥🔥): 

> - `GEMINI Competition`
> - `Meta AI`
> - `Llama 3.1 Model`
> - `Voice Channel AI Bots`
> - `Fine-Tuning Llama Models` 


- **Discussion on GEMINI Competition**: A member expressed interest in the **GEMINI Competition** from Google, seeking help from others for the hackathon.
   - *Reach out if you're interested to collaborate!*
- **Reactions to Llama-3.1 Model**: Members shared mixed feelings about **Llama-3.1**, with some calling it **soulless** compared to earlier generations of models.
   - Others noted that **Claude** and **Gemini** appear to retain some creative depth.
- **Uncensored Llama-3.1 Fine-Tuning**: A user is in the process of fine-tuning **Llama-3.1 405B** to create an uncensored version, expecting it to take several weeks.
   - They plan to release it on Hugging Face once training is complete, named **Llama3.1-406B-uncensored**.
- **Challenges of AI in Voice Channels**: There was a discussion about creating AI bots that can interact in **Discord voice channels**, highlighting its complexities.
   - Concerns were raised about the limitations currently faced when trying to build effective voice-interactive bots.
- **Costs and Accessibility of AI Models**: Members discussed the costs related to API usage for advanced models like **GPT-4o**, noting challenges to access higher tiers.
   - Some expressed frustration about the limitations imposed on lower tiers needing significant interaction.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1265198813455450185)** (7 messages): 

> - `Alpha Release Timing`
> - `User Communication Concerns`
> - `App Testing` 


- **Clarifying Alpha Release Timing**: Members are uncertain about the timelines for the alpha release, specifically if it is by the very last day of July or a few days before.
   - Questions were raised regarding the clarity of expectations, highlighting a need for better communication from the developers.
- **Users Eagerly Awaiting Alpha Access**: A member expressed frustration while checking the app every 20 minutes, hoping to be selected as a Plus user for the alpha release.
   - Another user confirmed that alpha testing is expected to start towards the end of July, implying a need for patience.
- **Concern Over Stale Information**: Amid discussions, a user pointed out that shared links regarding the alpha release are nearly a month old, indicating outdated information.
   - This led to a broader conversation about the lack of ongoing communication with paying customers.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1265096789577695303)** (7 messages): 

> - `Meta-Prompting`
> - `Plagiarism in AI Output`
> - `Prompting Techniques` 


- **Meta-Prompting Revolutionizes Prompt Engineering**: A member highlighted that AI guidance in prompt engineering is referred to as **meta-prompting**, described as perhaps the best method for learning how to craft effective prompts.
   - With meta-prompting, users can eventually create prompts that generate further prompts, enhancing their prompting skills.
- **Concerns Over Plagiarism in Output**: One member expressed frustration that using a blog results in **100% plagiarism** in content generated by prompts.
   - They were looking for solutions or ideas to mitigate this issue.
- **Seeking Solutions for Prompt Improvement**: In response to concerns about plagiarism, a member suggested sharing prompts and custom instructions to gain insights and suggestions from others.
   - They encouraged transparency by quoting another member for clarity, stating, *'Someone might be able to take a look and offer suggestions!'*.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1265096789577695303)** (7 messages): 

> - `Meta-Prompting`
> - `Plagiarism in Generated Content`
> - `Prompt Improvement Suggestions` 


- **Learn to Prompt with Meta-Prompting**: **Meta-prompting** is recognized as a top method for mastering prompt engineering, allowing users to create prompts that generate further prompts.
   - This technique can significantly enhance one's ability to craft effective prompts based on AI guidance.
- **Concerns about Plagiarism from Blog Content**: A user raised concerns that utilizing a blog resulted in **100% plagiarism** in every generated prompt.
   - This prompted discussions around finding solutions to improve the originality of generated content.
- **Suggestions for Better Prompting**: A member suggested sharing specific details from previous prompts and the context in order to get more tailored advice.
   - They highlighted the importance of articulating desired differences in response quality to receive effective suggestions.


  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1265076277887373443)** (39 messages🔥): 

> - `Mojo Community Meeting Presentations`
> - `String Optimization in Standard Library`
> - `Installing Mojo on VM`
> - `Game Engine Development in Mojo`
> - `Linking with C Libraries` 


- **Open Call for Mojo Community Meeting Presentations**: There is an opportunity to present at the [Mojo Community Meeting](https://modul.ar/community-meeting-doc), with slots available on August 12.
   - If you wish to present what you're building in Mojo or share your experience, you can sign up through the linked document.
- **Short String and Buffer Optimization Proposal**: A member confirmed that their work on **short string optimization** and **small buffer optimization** in the standard library is a great fit for presentation formats.
   - Another member supported this, noting the relevance of **optimizations** in past meetings.
- **Installing Mojo on an Ubuntu VM**: A user inquired about the feasibility of installing Mojo in an Ubuntu VM on Windows, to which others responded that it would generally work well with solutions like WSL and Docker.
   - Concerns were raised about potential installation issues, but VM usage is deemed suitable.
- **Assessing Mojo for Game Engine Development**: Discussion highlighted that Mojo could be suitable for crafting a next-gen game engine, particularly due to its good heterogeneous compute capabilities via GPU support.
   - However, challenges were noted with allocator handling in game development patterns, suggesting a few **rough spots** might be encountered.
- **Linking to C Libraries in Mojo**: There is ongoing discussion about linking Mojo to C libraries, with suggestions that improved functionality will benefit projects utilizing libpcap.
   - Members noted that using **ktls** should be the default for Mojo on Linux, enhancing low-level network customizability.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/modularml/mojo">GitHub - modularml/mojo: The Mojo Programming Language</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/modularml/mojo/issues/3262">Ability to Link to C Libraries · Issue #3262 · modularml/mojo</a>: Review Mojo&#39;s priorities I have read the roadmap and priorities and I believe this request falls within the priorities. What is your request? Ideally there would be something like a @link(…) decor...</li><li><a href="https://modul.ar/community-meeting-doc.">[Public] Mojo Community Meeting</a>: Mojo Community Meeting This doc link: https://modul.ar/community-meeting-doc  This is a public document; everybody is welcome to view and comment / suggest.  All meeting participants must adhere to th...
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/)** (1 messages): 

ModularBot: From *Modular*:
<https://twitter.com/Modular/status/1815463417391837596>
  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1265037280339300404)** (50 messages🔥): 

> - `SDL Bindings`
> - `Mojo Game Frameworking`
> - `Physics Engine Development`
> - `Contributing to Mojo`
> - `Pygame with Mojo` 


- **SDL Bindings in Progress**: A member is working on **SDL bindings** for Mojo, noting that **Pygame** is primarily a wrapper around SDL, making integration possible.
   - Another user mentioned their own SDL binding project has stalled but plans to update and improve its API.
- **Experimenting with Game Frameworking**: Discussion around experimenting with **game frameworking and physics** sparked interest, with one member sharing a personal experience of building a custom physics engine.
   - The same user hopes to transition their math into a general geometric algebra package called **Infrared** in the future.
- **Creating a Mini Socket Library**: A new member is developing a **mini socket library** for Mojo using **external_call** to integrate C functions, seeking permission to license it under **Apache 2.0**.
   - They expressed interest in contributing to Mojo, encouraged by the community's supportive response.
- **Contributions and Community Resources**: Members discussed available resources for contributing to Mojo, including links to GitHub issues marked as **good first issue**.
   - One user plans to read the contribution guidelines to better understand how to engage with community projects.
- **Anticipated Release of v24.5**: A member inquired about the release date of **v24.5**, referencing that **v24.4** was released in early June.
   - It was suggested that ongoing discussions around GPU features could delay the new release, leading to speculation about version numbering conventions.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/saviorand/lightbug_http/tree/main/external">lightbug_http/external at main · saviorand/lightbug_http</a>: Simple and fast HTTP framework for Mojo! 🔥. Contribute to saviorand/lightbug_http development by creating an account on GitHub.</li><li><a href="https://github.com/modularml/mojo/issues?q=is%3Aissue+is%3Aopen+sort%3Aupdated-desc+label%3A%22good+first+issue%22">Issues · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1265100055052816465)** (9 messages🔥): 

> - `Modular's Industry Relationships`
> - `NVIDIA Support`
> - `OpenCL and SYCL Usage` 


- **Modular keeps industry relationships private**: A member noted that *Modular's industry relationships* are private and they don't comment on them ahead of announcements, but will share publicly at the right time.
   - This maintains a level of confidentiality until they are ready to reveal the information in an official capacity.
- **NVIDIA support boosts Modular's approach**: Support from **NVIDIA** is seen as a significant enhancement, and a member expressed eagerness to utilize it once it is live.
   - There was a suggestion to discuss it further in a dedicated channel when the time comes.
- **OpenCL's journey and relevance**: A discussion highlighted *OpenCL's* origins and its importance in enabling high-level programming, particularly within platforms like SYCL and OneAPI.
   - Concerns were raised about the future use of OpenCL, especially given the shift away from older hardware, but its relevance for *certain databases and firewalls* was acknowledged.
- **General Purpose Compute with GPUs and FPGAs**: Members talked about utilizing **GPUs** and **FPGAs** not just for graphics but for general-purpose compute, especially in the context of databases.
   - There is recognition of the capabilities of these technologies to handle workloads effectively beyond their traditional roles.


  

---


### **Modular (Mojo 🔥) ▷ #[max-gpu](https://discord.com/channels/1087530497313357884/1212827673257316453/1265101767893454939)** (2 messages): 

> - `XLA`
> - `MAX engine`
> - `GPU performance` 


- **MAX engine: The next step after XLA**: The **MAX engine** is viewed as a successor to **XLA**, leveraging insights gained from it while addressing its shortcomings, such as being extensible and natively supporting **dynamic and parametric shapes**.
   - Expectations are set for significantly improved **CPU and GPU performance** compared to **XLA**.
- **Navigating the path to MAX/GPU launch**: Although specifics on the **MAX/GPU** cannot be revealed before its launch later this year, the team is committed to achieving the hard but right solutions.
   - The belief in the importance of **GPUs to the AI world** is driving this endeavor, which has generated excitement for progress towards the product's release.


  

---


### **Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1265031218026778895)** (86 messages🔥🔥): 

> - `Changes to memcpy`
> - `Documentation for Mojo`
> - `Use of Reference in Mojo`
> - `Updates on Mojo Nightly`
> - `Relationship of MAX and Mojo` 


- **Changes to memcpy function**: User discussed the recent changes made to the `memcpy` function, noting three overrides for pointer types, with confusion on the new signature.
   - Members explored how these changes might impact existing code, especially regarding types like `DTypePointer` and `LegacyPointer`, with potential solutions offered.
- **Need for better documentation in Mojo**: Users expressed frustration over the current state of Mojo documentation, citing overly technical explanations that lack clarity for learners.
   - Concerns were raised about Discord's formatting issues also complicating understanding, calling for improvements to documentation formats.
- **Discussion on Reference and equality**: A member questioned the absence of an `__eq__` method for `Reference`, speculating whether it's intended to be unique or exclusive.
   - Another user supported this idea, noting the efficiency of comparing memory addresses directly instead of dereferencing.
- **Mojo Nightly Update Announcement**: A notification about the latest Mojo nightly compiler update was shared, highlighting updates and bug fixes.
   - Users were encouraged to update their versions, with a link provided to the changelog for detailed changes.
- **Relationship between MAX and Mojo**: Members discussed how MAX is built using Mojo, emphasizing that both systems evolve together with shared compiler changes.
   - The blend of Mojo and C++ in MAX Kernel development was noted, clarifying the connection between the two.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://sourcegraph.com/search">Sourcegraph</a>: no description found</li><li><a href="https://youtu.be/_QVs626Vn2k?t=3617">Mojo 🔥 Community Meeting #4</a>: Recording of the Mojo Community Meeting #4🫓 Flat Buffers: memory efficient serialization⚒️ Forge Tools: extending the Mojo 🔥 standard library🔄 Mojo 🔥 Gen...</li><li><a href="https://github.com/modularml/mojo/blob/nightly/stdlib/src/memory/__init__.mojo">mojo/stdlib/src/memory/__init__.mojo at nightly · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/modularml/mojo/blob/16cc60dc3fbed1eff01a8f5fee94f97cf97cca33/stdlib/src/memory/__init__.mojo">mojo/stdlib/src/memory/__init__.mojo at 16cc60dc3fbed1eff01a8f5fee94f97cf97cca33 · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[mojo-marathons](https://discord.com/channels/1087530497313357884/1255303604894437388/1265051433703702528)** (1 messages): 

> - `Intel CPUID Library`
> - `AMD CPUID Mappings` 


- **Intel's CPUID Library Simplifies Access**: Intel's library wraps **CPUID** and converts it into a more understandable format without requiring users to consult processor documentation.
   - This provides a more user-friendly approach for developers working with Intel processors.
- **Separate CPUID Mappings for AMD and Intel**: It was noted that **AMD** and **Intel** maintain separate **CPUID** mappings, aside from distinguishing who manufactured the processor.
   - As a result, developers need to utilize different mappings for each manufacturer to access specific processor features.


  

---



### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1265026280664072223)** (84 messages🔥🔥): 

> - `FSDP performance issues`
> - `Llama 3.1 hosting`
> - `Generative ML contributions` 


- **FSDP Performance Troubles with nn.Parameters**: A user experienced a **20x slowdown** when adding `nn.Parameters` to their model with **FSDP** but found that using a parameter of size **16** improved performance significantly.
   - They discussed potential issues related to **buffer alignment** and how misalignment could affect **CPU** performance despite GPU kernels running fast.
- **Hosting Llama 3.1 405B**: A member announced they've hosted **Llama 3.1 405B instruct** on **8xH100 80GB** hardware, accessible through a [chat interface](https://chat.tune.app/) and [API](https://studio.tune.app/).
   - Unfortunately, access is gated behind a login, and the hosting arrangement incurs costs, leading to discussions about hardware limitations and hosting alternatives.
- **Contributions to Open AI Research**: A user introduced themselves as working on generative ML at a startup, expressing interest in contributing to **open AI research** and discussing a paper on learning to reason from fewer samples.
   - Their past experience includes work in **3D Computer Vision** and **Machine Translation**, highlighting their goal of advancing AI with limited data.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://chat.tune.app/">Tune Chat - Chat app powered by open-source LLMS</a>: With Tune Chat, access Prompts library, Chat with PDF, and Brand Voice features to enhance your content writing and analysis and maintain a consistent tone across all your creations.</li><li><a href="https://studio.tune.app/">TuneStudio</a>: no description found
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1265056944616640623)** (43 messages🔥): 

> - `New SAE architecture`
> - `Monte Carlo Dropout comparison`
> - `Hierarchical 3D Gaussians`
> - `Llama 3 model details`
> - `Transformer performance and sparsity` 


- **New SAE architecture introduced for efficient training**: A novel architecture known as **Switch SAE** uses conditional computation to scale sparse autoencoders (SAEs) efficiently, addressing computational challenges in training wide SAEs across layers.
   - *Links to relevant papers* emphasize the potential of this approach in recovering features from superintelligent language models.
- **Comparison of Monte Carlo Dropout for uncertainty**: A user noted that the results of their method should be compared with **Monte Carlo dropout**, which is considered a subpar approximation for Bayesian uncertainty quantification.
   - Another member shared insights that suggested many papers exist comparing these methodologies, highlighting concerns regarding the effectiveness of MC dropout.
- **Llama 3's image encoding limitations**: Concerns were raised about the **Llama 3** model's image encoder, particularly regarding its resolution limit of **224x224**.
   - Some suggested that using a **vqvae-gan style tokenizer**, as advocated by Armen's group, might have enhanced the image processing capabilities.
- **Transformer models and performance implications**: Discussion centered around the scaling of **Transformers** and how the fraction of FLOPs due to Multi-Head Attention (MHA) decreases as model size increases, potentially to **33% or less**.
   - Insights were shared about the necessity of both **V** and **O projection**, prompting thoughts on their implications for model interpretation and effectiveness.
- **Sparsity in Transformer models**: A paper was referenced discussing how leveraging sparsity in **Transformer layers** can yield competitive performances while decreasing training costs and increasing efficiency.
   - The findings suggest sparse variants can achieve similar perplexity levels to traditional Transformers, making them suitable for longer sequence processing.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2407.14561">NNsight and NDIF: Democratizing Access to Foundation Model Internals</a>: The enormous scale of state-of-the-art foundation models has limited their accessibility to scientists, because customized experiments at large model sizes require costly hardware and complex engineer...</li><li><a href="https://arxiv.org/abs/2111.12763">Sparse is Enough in Scaling Transformers</a>: Large Transformer models yield impressive results on many tasks, but are expensive to train, or even fine-tune, and so slow at decoding that their use and study becomes out of reach. We address this p...</li><li><a href="https://ai.meta.com/research/publications/the-llama-3-herd-of-models/">no title found</a>: no description found</li><li><a href="https://arxiv.org/abs/2403.13097">Simple Ingredients for Offline Reinforcement Learning</a>: Offline reinforcement learning algorithms have proven effective on datasets highly connected to the target downstream task. Yet, leveraging a novel testbed (MOOD) in which trajectories come from heter...</li><li><a href="https://repo-sam.inria.fr/fungraph/hierarchical-3d-gaussians/">A Hierarchical 3D Gaussian Representation for Real-Time Rendering of
    Very Large Datasets</a>: no description found</li><li><a href="https://www.lesswrong.com/posts/47CYFbrSyiJE2X5ot/efficient-dictionary-learning-with-switch-sparse">Efficient Dictionary Learning with Switch Sparse Autoencoders — LessWrong</a>: Produced as part of the ML Alignment &amp; Theory Scholars Program - Summer 2024 Cohort …
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/)** (1 messages): 

alofty: https://arxiv.org/abs/2407.14561
  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1265022144937459804)** (23 messages🔥): 

> - `Task Grouping Recommendations`
> - `lm-eval Harness Updates`
> - `vLLM and Logits Issues`
> - `Automated Unit Testing Discussions`
> - `Transformers Version Problems` 


- **Groups vs Tags for Task Grouping**: It is recommended to use **groups** for nested tasks and aggregate scores, while **tags** suffice for simpler cases.
   - *Hailey Schoelkopf* confirmed that this method is effective for task organization.
- **lm-eval Harness Enhancements**: Updates to the lm-eval harness include a new superclass for API models, enhancing modularity and functionality, as seen in [Pull Request #2008](https://github.com/EleutherAI/lm-evaluation-harness/pull/2008).
   - Members can now evaluate Llama-405B on all task types using the `local-completions` model type with VLLM's OpenAI server.
- **Clarification on vLLM and Logits**: A discussion arose regarding whether **vLLM** provides logits, with conflicting views on its capabilities; however, it was clarified that it does provide continuation logits.
   - The conversation referenced issues from both the [VLLM repository](https://github.com/vllm-project/vllm/issues/185) and the [Triton inference server repository](https://github.com/triton-inference-server/server/issues/6895).
- **Interest in Automated Unit Testing**: A member raised concerns about the current lack of automated unit testing, emphasizing its importance to prevent breaking changes in the codebase.
   - Hailey Schoelkopf acknowledged the need for improved testing and mentioned existing regression tests, though they are limited in sample sizes.
- **Issues with Transformers Version**: Layernorm discovered issues with their deepseek model being misidentified as a Llama model after a recent commit to **Transformers**.
   - Pinning the Transformers version resolved the issue, indicating it was related to the latest updates made to the library.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/EleutherAI/lm-evaluation-harness?tab=readme-ov-file#:~:text=yourip%7D%3A8000/v1%2C-,num_concurrent%3D1,-%2Cmax_retries%3D3%2Ctokenized_requests">GitHub - EleutherAI/lm-evaluation-harness: A framework for few-shot evaluation of language models.</a>: A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/vllm-project/vllm/issues/185#issuecomment-1600931023">Can I directly obtain the logits here? · Issue #185 · vllm-project/vllm</a>: Hi, wonderful work! I want to know if there is a easy way to obtain the logits, since sometimes I only need to calculate the perplexity/language modeling loss of specific sequence. I saw the code h...</li><li><a href="https://github.com/triton-inference-server/server/issues/6895">vllm backend - logit probabilities at inference · Issue #6895 · triton-inference-server/server</a>: regarding the current vllm backend: https://github.com/triton-inference-server/vllm_backend/tree/main I wanted to know if at inference there is a possibility of also getting the logit probabilities...</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/pull/2008">Refactor API models by baberabb · Pull Request #2008 · EleutherAI/lm-evaluation-harness</a>: This PR introduces a new superclass for API request models, providing:  Modularity for downstream classes Overloadable methods for request transformation, API requests and response parsing Tokeniza...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1265084194590036120)** (5 messages): 

> - `Nerdsniping Evaluation`
> - `Uncheatable Evaluation Harness` 


- **Nerdsniping with Evaluation**: A member expressed a lighthearted intent, stating, *'one of these days I'll nerdsnipe you with evaluation.'*
   - The comment suggests a playful challenge around the intricacies of evaluation methods.
- **Challenges of Uncheatable Eval**: In a query, a member asked if they could incorporate **uncheatable eval** into an eval harness, raising questions about its practicality.
   - Another member humorously remarked, *'It ceases to be uncheatable once you add it to the harness.'*
- **Fresh Scrape Defense**: A member asserted that **uncheatable eval** remains effective as long as it's pointed at a **fresh scrape**.
   - This claim was met with skepticism, suggesting that using it with a harness could lead to limitations on its power.
- **Concerns Over Power and Reproducibility**: A member warned that the idea of an uncheatable evaluation approach is too powerful and questioned its feasibility citing reproducibility.
   - They indicated that the concept might face scrutiny or rejection due to its implications on standard practices.


  

---



### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1265020956783673344)** (69 messages🔥🔥): 

> - `Meta's AI Strategy`
> - `NVIDIA's Market Position`
> - `OpenAI Pricing Wars`
> - `Llama 3.1 Release` 


- **Meta's AI Strategy and Premium Offerings**: There are discussions about Meta's potential rollout of a **Premium** version of **Llama 405B**, with speculation about an announcement on **Jul 23**.
   - Members noted that the recent removal of restrictions on Llama models opens the door for broader use beyond just improving other models.
- **NVIDIA's Potential Monopoly**: Concerns were raised about NVIDIA's ambitions to integrate hardware, CUDA, and models, creating a potential monopoly akin to historical antitrust cases involving IBM.
   - A user suggested that NVIDIA could essentially print money if they controlled the entire stack, but regulatory hurdles would prevent such integration.
- **OpenAI's Competitive Pricing Strategies**: OpenAI's announcement of *free* fine-tuning for **gpt-4o-mini** up to 2M tokens per day sparked a conversation about the aggressive pricing landscape in AI.
   - Members reflected on the chaotic state of pricing wars in the industry as a response to increased competition.
- **Llama 3.1 and Performance Metrics**: The release of **Llama 3.1** was highlighted, with members discussing its incorporation into **RewardBench**, showing alignment with **GPT-4** on certain tasks.
   - The models were reported to be primarily challenged by safety concerns, which users noted could be beneficial.
- **Industry Insights and References**: A user appreciated insights from Ben Thompson's *Stratechery*, indicating its relevance to the ongoing discussions about market dynamics.
   - Other members shared their takes on the cyclical nature of tech strategies, pointing out how companies often repeat historical patterns.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.joelonsoftware.com/2002/06/12/strategy-letter-v/">Strategy Letter V</a>: When I was in college I took two intro economics courses: macroeconomics and microeconomics. Macro was full of theories like &#8220;low unemployment causes inflation&#8221; that never quite stood u…</li><li><a href="https://x.com/natolambert/status/1815813221410037957">Tweet from Nathan Lambert (@natolambert)</a>: Added the Llama 3.1 models to RewardBench via @togethercompute. Is mostly held back by safety, which some would argue is good. In line with GPT-4 on other challenge task.</li><li><a href="https://x.com/testingcatalog/status/1815439546722451493?s=46">Tweet from TestingCatalog News 🗞 (@testingcatalog)</a>: Besides that, it seems that LLama 405B may become a part of the Premium offering and in this case, Meta AI Premium could be announced on Jul 23 as well (Spotted in the code).  Also, a mention of AI St...</li><li><a href="https://fxtwitter.com/moyix/status/1815840634013639086?s=46">Tweet from Brendan Dolan-Gavitt (@moyix)</a>: Sorry OpenAI is doing WHAT now?! Fine-tuning gpt-4o-mini is *free* for up to 2M tok/day??</li><li><a href="https://x.com/kalomaze/status/1815547484376076460?s=46">Tweet from kalomaze (@kalomaze)</a>: LLM-Distillery! An open source training pipeline built by AMOGUS & I over the past several months for collecting and training &#39;student&#39; language models to imitate &#39;teacher&#39; models via ...</li><li><a href="https://www.diffchecker.com/nbHQZode/">3.1 vs 3 - llama license - Diffchecker</a>: 3.1 vs 3 - llama license - META LLAMA 3 COMMUNITY LICENSE AGREEMENT Meta Llama 3 Version Release Date: April 18, 2024  “Agreeme</li><li><a href="https://web.archive.org/web/20240722214257/https://huggingface.co/huggingface-test1/test-model-1">huggingface-test1/test-model-1 · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1265197069404999732)** (16 messages🔥): 

> - `Magpie paper on synthetic data generation`
> - `LLaMA 3 Instruct performance`
> - `Instruction finetuning techniques`
> - `Vocabulary size and inference speed` 


- **Magpie paper reveals synthetic data generation techniques**: The [Magpie paper](https://arxiv.org/abs/2406.08464) presents a method for generating **high-quality instruction data** for LLMs using only templates, allowing models like **LLaMA 3** to generate user queries with minimal input.
   - This technique reportedly generates data at scale and shows a **larger vocabulary diversity** compared to existing datasets like Alpaca and UltraChat.
- **Surprising performance of LLaMA 3 with Magpie dataset**: Even with only **300k samples**, the **LLaMA 3 Base** finetuned on the **Magpie IFT dataset** managed to outperform the **original LLaMA 3 Instruct** model by 9.5% on **AlpacaEval**.
   - This raises questions about the effectiveness of traditional instruction distillation techniques when compared to novel dataset generation methods.
- **Instruction finetuning insights from Raschka's blogpost**: In his blogpost, Sebastian Raschka covers advancements in **instruction finetuning**, emphasizing new cost-effective methods for generating finetuning datasets.
   - He highlights potential applications and recent developments in LLMs integration by major tech companies, along with the importance of high-quality instruction data.
- **Debate on vocabulary size's impact on inference speed**: A discussion arose regarding Raschka's claim that a **larger vocabulary size** could potentially slow down inference, contrasting with the typical belief that fewer but denser tokens would speed up the process.
   - The members noted the relative impact of vocabulary size increases on smaller models compared to larger ones, suggesting that finding an optimal balance is essential.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://magazine.sebastianraschka.com/p/instruction-pretraining-llms">Instruction Pretraining LLMs</a>: The Latest Research in Instruction Finetuning</li><li><a href="https://arxiv.org/abs/2406.08464">Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing</a>: High-quality instruction data is critical for aligning large language models (LLMs). Although some models, such as Llama-3-Instruct, have open weights, their alignment data remain private, which hinde...</li><li><a href="https://magazine.sebastianraschka.com/i/146761957/running-the-dataset-generation-locally">Instruction Pretraining LLMs</a>: The Latest Research in Instruction Finetuning
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1265270069462827049)** (43 messages🔥): 

> - `Llama 3 Release`
> - `Mark Zuckerberg's AI Era`
> - `Model Watermarking Concerns`
> - `Public Perception of Zuckerberg` 


- **Llama 3 Foundation Models Launched**: The release of [Llama 3](https://llama.meta.com/) featured a herd of language models supporting multilinguality and tool usage, with the largest model boasting **405B parameters** and **128K tokens** context window. A paper on the models details evaluations showing Llama 3's performance comparable to **GPT-4**.
   - There are discussions about watermarking models and tracking downloads, as users need to **provide information** and agree to a license before accessing weights.
- **Watching Zuck's AI Image Transform**: A member shared their thoughts on watching a [YouTube video](https://youtu.be/YuIc4mq7zMU?si=UgEu2onfXdlblT9j) about Mark Zuckerberg, mentioning it feels like a puff piece centered around his newfound 'coolness'. They noted it mostly reinforced the narrative that Zuckerberg needed to adapt to public perceptions.
   - Comments included reflections on Zuckerberg's historical narrative about **Windows' dominance** due to its openness, which some users deemed as rewriting history.
- **Debate on Download Tracking for AI Models**: Concerns were raised about how Meta might be **tracking downloads** of its models, where users provide their info to receive links. This leads to speculation that the tracking could be to ensure compliance with agreements.
   - The conversation hints at the potential for **analytics** purposes but also raises privacy issues regarding data collection.
- **Personal Studies on Llama 3 and Open Strategy**: A member expressed excitement about studying Llama 3 and its broader implications in tool usage and strategy, noting it might take weeks to digest all the content. They plan to start with big-picture articles before diving into technical posts.
   - There’s anticipation about how this knowledge could influence understanding of **AI models** and their societal impact.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/togethercompute/status/1815796476775461019?s=46">Tweet from Together AI (@togethercompute)</a>: @altryne @satpal_patawat @GroqInc @Teknium1 @JonathanRoss321 @xenovacom @altryne is the prompt longer than 4K? Currently limiting the context, but we&#39;ll be opening it up soon. If you are seeing an...</li><li><a href="https://youtu.be/YuIc4mq7zMU?si=UgEu2onfXdlblT9j">Inside Mark Zuckerberg&#39;s AI Era | The Circuit</a>: If the latest battle in the AI wars is between open and closed models, Meta CEO and Founder Mark Zuckerberg is right on the frontlines. Since rebranding as M...</li><li><a href="https://llama.meta.com/">Llama 3.1</a>: The open source AI model you can fine-tune, distill and deploy anywhere. Our latest models are available in 8B, 70B, and 405B variants.</li><li><a href="https://www.producthunt.com/posts/llama-7"> Llama - 3.1-405B: an open source model to rival GPT-4o / Claude-3.5 | Product Hunt</a>: Meta is releasing three models: The new 3.1-405B and upgrades to their smaller models: 3.1-70B and 3.1-8B. If 405B is as good as the benchmarks indicate, this would be the first time an open source mo...</li><li><a href="https://ai.meta.com/research/publications/the-llama-3-herd-of-models/">no title found</a>: no description found
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1265053168799318139)** (3 messages): 

> - `Claude AI boundaries`
> - `Sacred Texts in AI`
> - `Release of GPT-3.5 Opus` 


- **Claude AI limits speech on Sacred Texts**: A user noted that while trying to demonstrate something in **Claude**, they encountered **strong guardrails** regarding Sacred Texts, specifically citing their choice of *I Have a Dream*.
   - *Claude has a strong to avoid using sensitive texts*, which was evident during this interaction.
- **Comparisons made between users and Dr. King**: In a light-hearted comment, one user likened themselves to **Dr. King** by proclaiming their papers as *sacred text*.
   - This humorous comparison received a congratulatory response, highlighting a theme of reverence in discussing one's work.


  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1265193010833002557)** (7 messages): 

> - `OpenAI vs Llama 3.1`
> - `ChatGPT Memory Management`
> - `Mark Zuckerberg's AI Era`
> - `Snail Appreciation` 


- **OpenAI's Unexpected Refund**: A user reported that **OpenAI accepted defeat against Llama 3.1** and randomly refunded them an unexpected amount.
   - They expressed gratitude with a simple acknowledgment: *
- **Managing ChatGPT Memory Like a Game**: One member compared managing Memory for **ChatGPT** to **inventory management in a game**, noting they typically quit when Memory becomes full.
   - This analogy highlights the challenges users face in efficiently using Memory within the platform.
- **Inside Mark Zuckerberg's AI Era**: A user shared a [YouTube video titled 'Inside Mark Zuckerberg's AI Era'](https://www.youtube.com/watch?v=YuIc4mq7zMU) discussing the ongoing battle in the AI landscape.
   - The video emphasizes **Meta CEO Mark Zuckerberg's** position at the forefront of competition between open and closed models in AI.
- **Snail Enthusiasm Shared**: A user humorously engaged with the community by sending an image of a **snail** in motion, eliciting a positive response.
   - The newly shared love for snails further lightens the mood in the discussion.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/xlr8harder/status/1815633845422584171">Tweet from xlr8harder (@xlr8harder)</a>: absolutely crushed by this</li><li><a href="https://www.youtube.com/watch?v=YuIc4mq7zMU">Inside Mark Zuckerberg&#39;s AI Era | The Circuit</a>: If the latest battle in the AI wars is between open and closed models, Meta CEO and Founder Mark Zuckerberg is right on the frontlines. Since rebranding as M...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[nlp](https://discord.com/channels/1179127597926469703/1208183200099344445/1265052470636187790)** (3 messages): 

> - `Distillation`
> - `Llama 3.1` 


- **Search for Blog Posts on Distillation**: A member asked if anyone had a blog post they liked on **distillation**, indicating interest in the topic.
   - This led to a discussion about the lack of comprehensive resources on the subject.
- **Missing Comprehensive Post by Lilian Wang**: Another member expressed surprise that there isn't a **20k word** post by Lilian Wang on distillation.
   - This comment reflects a desire for detailed discussions and resources in the community.
- **Potential Write-Up on Llama 3.1 Distillation**: A member mentioned they might write a paragraph or two if **Llama 3.1** is distilled.
   - This suggests ongoing interest in new advancements and the documentation of such processes.


  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1265421370112606269)** (4 messages): 

> - `SnailBot updates`
> - `User engagement timings` 


- **SnailBot News Announcement**: A notification for **SnailBot News** was made, targeting the user role <@&1216534966205284433> for updates.
   - *Stay tuned for exciting announcements and updates from SnailBot!*
- **Engagement Duration Noted**: **45 minutes** was mentioned, potentially highlighting user engagement length or a related timeframe.
   - *This insight may inform future discussions or activities considering user interaction scales.*
- **User Reflects on Content**: A member expressed that there was something **interesting** about the discussions happening.
   - *Positive engagement from users suggests a dynamic conversation flow within the channel.*


  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1265052493142819078)** (73 messages🔥🔥): 

> - `Llama 3.1 Release`
> - `Mistral and Nemo Concerns`
> - `Training Issues`
> - `Language Inclusion in Models`
> - `Evaluation Scores Comparison` 


- **Llama 3.1 Release Generates Mixed Reactions**: The anticipation around the **Llama 3.1** release was palpable, but some expressed concerns about its utility and performance, especially for models like **Mistral**.
   - *Duh Kola* lamented, *'Damn they don't like the llama release'*, indicating discontent with the overall reception.
- **Training Challenges with Llama 3.1**: Users are encountering errors while training **Llama 3.1**, particularly regarding the `rope_scaling` configuration, causing frustration among the community.
   - One member got it running by updating transformers, stating, *'Seems to have worked thx!'* after overcoming significant hurdles.
- **Discussion on Language Inclusion**: Concerns arose about the exclusion of **Chinese** language support in **Llama 3.1**, with members expressing that it's a detrimental oversight given its global importance.
   - Comments highlighted that while the model tokenizer includes Chinese, the language's absence in prioritization was perceived as a strategic misstep.
- **Comparing Evaluation Scores: Llama 3.1 vs Qwen**: Discussion ensued regarding the **cmmlu** and **ceval** scores of **Llama 3.1**, with evaluations indicating only a slight improvement over its predecessor.
   - Members noted that **Qwen's** self-reported scores show better performance but may not directly compare due to differences in evaluation methodology.
- **Licensing Concerns for Qwen Model**: Questions were raised about the **licensing** status of **Qwen**, particularly whether it remains under Alibaba's restrictions or has become fully open.
   - *Noobmaster29* mentioned, *'as long as it's public weights, I don't really mind the license,'* reflecting pragmatism in the community's approach to model access.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://llama.meta.com/llama-downloads/">Download Llama</a>: Request access to Llama.</li><li><a href="https://llama.meta.com/">Llama 3.1</a>: The open source AI model you can fine-tune, distill and deploy anywhere. Our latest models are available in 8B, 70B, and 405B variants.
</li>
</ul>

</div>
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1265135798655909938)** (33 messages🔥): 

> - `LLM Distillation`
> - `DPO Training Issues`
> - `Adapter Fine Tuning`
> - `Reward Modeling`
> - `ChiPO Algorithm` 


- **Exploring LLM Distillation Pipeline**: A member shared a link to the [LLM Distillery GitHub repo](https://github.com/golololologol/LLM-Distillery), which outlines a pipeline for LLM distillation.
   - Discussion highlighted their implementation of precomputing logits on disk followed by KL divergence.
- **DPO Training's Stagnation Concerns**: Concerns were raised about the lack of progress on DPO integration, with a member noting no movement for two weeks.
   - There was some confusion, but a member confirmed they were reviewing the issue again for resolution.
- **Adapter Fine Tuning Stages**: A member inquired about thoughts on [GitHub issue #1095](https://github.com/axolotl-ai-cloud/axolotl/issues/1095) related to multiple stages of adapter fine tuning.
   - They proposed initializing later stages using prior weights to enhance the efficiency of DPO training.
- **Mathematical Complexity of DPO and NLL Loss**: There was a discussion about the complexities around DPO and incorporating NLL loss, with skepticism about its empirical validity.
   - Members expressed interest in integrating the mathematical theories from recent papers into practical applications.
- **Reward Modeling vs. PPO Approaches**: A consensus emerged that reward modeling is still preferred over Proximal Policy Optimization (PPO) despite its limitations.
   - Members entertained strategies for implementing stepwise-DPO possibly enhanced by LoRA adapters.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2407.13399">Correcting the Mythos of KL-Regularization: Direct Alignment without Overoptimization via Chi-Squared Preference Optimization</a>: Language model alignment methods, such as reinforcement learning from human feedback (RLHF), have led to impressive advances in language model capabilities, but existing techniques are limited by a wi...</li><li><a href="https://github.com/golololologol/LLM-Distillery">GitHub - golololologol/LLM-Distillery: A pipeline for LLM distillation</a>: A pipeline for LLM distillation. Contribute to golololologol/LLM-Distillery development by creating an account on GitHub.</li><li><a href="https://github.com/axolotl-ai-cloud/axolotl/issues/1095)">Issues · axolotl-ai-cloud/axolotl</a>: Go ahead and axolotl questions. Contribute to axolotl-ai-cloud/axolotl development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/1265114145204994110)** (3 messages): 

> - `LLM for verb tense conversion`
> - `Spacy script for perspective change`
> - `Third-person to first-person conversion`
> - `Dataset for tense conversion examples` 


- **Seeking LLM for Tense and Perspective Adjustment**: A member asked if anyone knows of an **LLM** or a **script using Spacy** that can effectively change the verb tense and perspective of arbitrary text.
   - They specifically need to convert text from **third-person/past tense to first-person/present tense**.
- **Unfinished Dataset for Tense Conversion**: Another member shared their past work on building a **10k sample dataset** for verb tense conversion approximately a year ago but left it unfinished due to other commitments.
   - They expressed a willingness to be informed if any relevant tools or resources are found by others.


  

---



### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1265227814009835530)** (8 messages🔥): 

> - `Code Confluence Tool`
> - `DSPY Integration`
> - `Zenbase/Core Library Launch` 


- **Code Confluence Tool Generates GitHub Summaries**: Inspired by **DSPY**, a member introduced **Code Confluence**, an OSS tool built with **Antlr**, **Chapi**, and DSPY pipelines, designed to create detailed summaries of GitHub repositories. The tool's results may outperform existing OS products, as demonstrated on their [DSPY repo](https://github.com/unoplat/unoplat-code-confluence/blob/main/unoplat-code-confluence/examples/python/dspy/dspy_v1.md).
   - They provided additional resources, including the [Unoplat Code Confluence GitHub](https://github.com/unoplat/unoplat-code-confluence/) and a compilation of summaries called [OSS Atlas](https://github.com/unoplat/unoplat-oss-atlas/tree/main).
- **Engagement and Feedback on Code Confluence**: Feedback was welcomed for their new tool, with user engagement indicating interest and excitement about its capabilities. One user commented that they will **check it out** and expressed enthusiasm with a **🔥** emoji.
   - Another user noted the abundance of interesting developments shared recently, contributing to a buzz around the **DSPY** community.
- **Zenbase/Core Library Launch on Twitter**: A member announced the Twitter launch of **zenbase/core**, a Python library that allows users to utilize DSPY's optimizers with their existing **Instructor** and **LangSmith** code. They requested retweets, likes, and stars on their announcement, which can be viewed [here](https://twitter.com/cyrusofeden/status/1815858216389300383?s=61&t=WwA-PFs585hhcOplJkLRbQ).
   - The introduction of this library indicates ongoing efforts to integrate DSPY functionalities into broader coding practices, enhancing user experience.


  

---


### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/1265108584442892370)** (2 messages): 

> - `AI Research Paper`
> - `Implementation Requests` 


- **New AI Research Paper Alert**: A member shared a link to an AI research paper titled [2407.12865](https://arxiv.org/pdf/2407.12865), sparking interest in its findings.
   - Others are encouraged to check it out and discuss its implications in the community.
- **Call for Code Replication**: A member requested that if anyone writes code to replicate the findings of the paper or finds an existing implementation, they should share it or DM him.
   - This highlights a collaborative approach to advancing discussions on this research.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1265022761856667809)** (83 messages🔥🔥): 

> - `DSPy and Outlines comparison`
> - `Entity extraction with DSPy`
> - `Structured output issues with Llama3`
> - `Optimizer updates in DSPy`
> - `LOTUS integration with Phoenix` 


- **Comparison of JSON Generation Libraries**: Members discussed the strengths and weaknesses of libraries like **Jsonformer**, **Outlines**, and **Guidance** in generating structured JSON, noting that **Outlines** offers better support for Pydantic formats and JSON schemas.
   - *Jsonformer* is praised for strict schema adherence, while *Guidance* and *Outlines* provide more flexibility but may introduce complexity.
- **Entity Extraction Module in DSPy**: A user inquired about observing internal steps while executing an **EntityExtractor** module in DSPy, which led to the suggestion to use the `inspect_history` method.
   - This method aims to help users understand the internal workings of the module when processing inputs.
- **Challenges with Llama3 Structured Outputs**: Users expressed difficulty in obtaining correctly structured outputs from the **Llama3** model using DSPy, suggesting the use of `dspy.configure(experimental=True)` alongside *TypedChainOfThought*.
   - However, there were questions about viewing model outputs even if they fail type checks, with limitations noted on the usefulness of `inspect_history`.
- **Interest in DSPy Optimizer Updates**: A user raised a question about the plans for merging the backend refactor of DSPy into the main branch, particularly interested in experimenting with new optimizers.
   - This indicates ongoing developments in DSPy and user engagement in its enhancements.
- **Integration of LOTUS with Phoenix**: A user asked about hooking up **LOTUS** with **Phoenix** to inspect queries, revealing interests in exploring integration opportunities within the DSPy ecosystem.
   - Another member confirmed active usage of LOTUS with Modin, indicating existing practical applications of these integrations.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="http://localhost:30000"))">no title found</a>: no description found</li><li><a href="https://github.com/sksarvesh007/dspy-rag-application/blob/main/embedder.ipynb">dspy-rag-application/embedder.ipynb at main · sksarvesh007/dspy-rag-application</a>: Contribute to sksarvesh007/dspy-rag-application development by creating an account on GitHub.</li><li><a href="https://github.com/outlines-dev/outlines">GitHub - outlines-dev/outlines: Structured Text Generation</a>: Structured Text Generation. Contribute to outlines-dev/outlines development by creating an account on GitHub.</li><li><a href="https://github.com/lm-sys/RouteLLM?tab=readme-ov-file">GitHub - lm-sys/RouteLLM: A framework for serving and evaluating LLM routers - save LLM costs without compromising quality!</a>: A framework for serving and evaluating LLM routers - save LLM costs without compromising quality! - lm-sys/RouteLLM</li><li><a href="https://github.com/stanfordnlp/dspy">GitHub - stanfordnlp/dspy: DSPy: The framework for programming—not prompting—foundation models</a>: DSPy: The framework for programming—not prompting—foundation models - stanfordnlp/dspy</li><li><a href="https://github.com/stanfordnlp/dspy/tree/rc">GitHub - stanfordnlp/dspy at rc</a>: DSPy: The framework for programming—not prompting—foundation models - GitHub - stanfordnlp/dspy at rc</li><li><a href="https://github.com/stanfordnlp/dspy/issues/590">How to change the generation length · Issue #590 · stanfordnlp/dspy</a>: Hey, I&#39;m initializing my LM as follows, extras={&#39;max_tokens&#39;:4000,&#39;temperature&#39;:0.7} vllm = dspy.HFClientVLLM(model=&quot;Mistral-7B-Instruct-v0.1&quot;,url=&quot;https://my_vllm_u...</li><li><a href="https://github.com/stanfordnlp/dspy/blob/31ac32ba1a0b51cb7b9a8728b0bb7d4f3f2860a5/dsp/modules/hf.py#L30">dspy/dsp/modules/hf.py at 31ac32ba1a0b51cb7b9a8728b0bb7d4f3f2860a5 · stanfordnlp/dspy</a>: DSPy: The framework for programming—not prompting—foundation models - stanfordnlp/dspy
</li>
</ul>

</div>
  

---


### **DSPy ▷ #[colbert](https://discord.com/channels/1161519468141355160/1250300504462856265/1265184598908731443)** (3 messages): 

> - `ColPali Use Cases`
> - `ColBert and RAG`
> - `Qdrant Support for ColBert` 


- **Exploring ColPali for Medical Documents**: A member shared their experience with ColPali, stating they are testing it for **RAG of medical documents with images**, as **ColBert and standard embedding models** have previously failed in this area.
   - They also plan to explore training and using other vision-language models for improved effectiveness.
- **Qdrant's Adoption of ColBert**: Another member highlighted that **Qdrant** now supports ColBert, providing documentation on [Hybrid and Multi-Stage Queries](https://qdrant.tech/documentation/concepts/hybrid-queries/) available since v1.10.0.
   - The introduction of multi-query capabilities allows for complex search scenarios leveraging named vectors per point, enhancing retrieval processes.



**Link mentioned**: <a href="https://qdrant.tech/documentation/concepts/hybrid-queries/">Hybrid Queries - Qdrant</a>: Qdrant is an Open-Source Vector Database and Vector Search Engine written in Rust. It provides fast and scalable vector similarity search service with convenient API.

  

---



### **LlamaIndex ▷ #[announcements](https://discord.com/channels/1059199217496772688/1073670729054294197/1265342139064127499)** (1 messages): 

> - `LlamaIndex Webinar`
> - `ColPali Document Retrieval`
> - `Vision Language Models`
> - `ViDoRe Benchmark` 


- **LlamaIndex Webinar on Efficient Document Retrieval**: Join the upcoming webinar hosted by ColPali authors discussing **Efficient Document Retrieval with Vision Language Models** this Friday at **9am PT**. [Signup here](https://lu.ma/9q4ldrwc) to learn about cutting-edge techniques in document processing.
- **ColPali's Innovative Approach to Document Retrieval**: ColPali introduces a novel technique that directly embeds **page screenshots** with Vision Language Models (VLMs), improving retrieval performance over **complex documents**. This method avoids loss of crucial visual information that traditional parsing and OCR typically encounter.
- **New Benchmark for Document Retrieval**: The new **ViDoRe benchmark** proposed by ColPali better addresses challenging retrieval tasks associated with various document elements, enhancing the evaluation of retrieval systems. This benchmark is designed to complement the traditional methods by focusing on visual representations.
- **Future of Multimodal Document Retrieval**: The webinar will delve into the **multimodal document retrieval** future, integrating techniques from ColPali and LlamaParse. The discussion will highlight an end-to-end system that achieves state-of-the-art results in document retrieval.



**Link mentioned**: <a href="https://lu.ma/9q4ldrwc">LlamaIndex Webinar: ColPali - Efficient Document Retrieval with Vision Language Models · Zoom · Luma</a>: Enterprise RAG systems face a significant challenge when processing PDFs with complex layouts, tables, and figures. Conventional RAG pipelines typically…

  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1265021000773533766)** (8 messages🔥): 

> - `TiDB Future App Hackathon 2024`
> - `Mixture-of-Agents with LlamaIndex`
> - `Llama 3.1 Performance`
> - `LlamaParse Features`
> - `MongoDB AI Applications Program` 


- **Join the $30,000 TiDB Future App Hackathon!**: We're sponsoring a month-long [TiDB Future App Hackathon 2024](https://t.co/vTV3t8daqT) with over **$30,000 in prizes** including **$12,000** for first place, partnering with @pingcap and others.
   - Participate to build innovative AI applications using the latest [TiDB Serverless with Vector Search](https://www.pingcap.com/ai/).
- **Discover Mixture-of-Agents with LlamaIndex!**: In a new video, @1littlecoder introduces a novel approach called 'mixture of agents' which uses multiple local language models to potentially outperform single models like **GPT-4**.
   - Check out the [step-by-step tutorial](https://t.co/EqF2RM3jeB) to explore how this method can enhance your AI projects.
- **Llama 3.1 Models Available Now**: The **Llama 3.1** series with models of **8B**, **70B**, and **405B** are now available for use with LlamaIndex via Ollama, though the 405B requires substantial computing power.
   - For hosted versions, check out our partners at [Fireworks AI](https://t.co/NMckK14nZf) for assistance.
- **Explore LlamaParse's Capabilities**: In a video, @seldo highlights key features of **LlamaParse** including options for **Markdown and JSON outputs**, along with enhanced **OCR support**.
   - This tool is designed for greater metadata extraction across multiple languages, making it highly versatile for document processing.
- **MongoDB AI Applications Program Launch!**: @MongoDB has announced the general availability of its **AI Applications Program (MAAP)**, aimed at helping organizations build and deploy AI-rich applications efficiently.
   - Learn more about MAAP's offerings and how it can accelerate your AI journey [here](https://t.co/rCz3DfUe3A).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://t.co/vTV3t8daqT">TiDB Future App Hackathon 2024</a>: Innovate and Create Amazing AI Applications</li><li><a href="https://t.co/rCz3DfUe3A">MongoDB AI Applications Program</a>: Get the support you need to accelerate your AI application journey and launch with confidence and speed.
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1265139668752662644)** (61 messages🔥🔥): 

> - `context_window parameter`
> - `chunk_size and chunk_overlap`
> - `model availability and context size`
> - `ValueError in LlamaIndex`
> - `using models with larger context windows` 


- **Understanding the context_window parameter**: The `context_window` parameter specifies the maximum number of tokens the model can handle, including both input and output tokens.
   - If the input text is too long, it may restrict output generation, leading to errors if the token limit is exceeded.
- **Defining chunk_size and chunk_overlap**: `chunk_size` sets the maximum number of tokens in each chunk during processing, while `chunk_overlap` defines the number of overlapping tokens between consecutive chunks.
   - These parameters help control the precision of embeddings and ensure context is retained across chunks.
- **Addressing ValueErrors in LlamaIndex**: A ValueError indicating a negative context size suggests the input text exceeds the current model's `context_window` limit.
   - Reducing input size or switching to a model with a larger context window are potential resolutions.
- **Maximize model efficiency with context_window**: In cases where the context window is reached, it may limit the model's output capacity significantly.
   - Choosing models with appropriate `context_window` values based on input length is essential for optimal performance.
- **Discussion on context_window's scope**: Clarifications were shared regarding whether `context_window` covers only input tokens or includes outputs as well.
   - It was confirmed that the `context_window` encompasses both, necessitating careful management of input sizes.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.llamaindex.ai/en/latest/api_reference/llms/cleanlab/#llama_index.llms.cleanlab.CleanlabTLM>).">Cleanlab - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/module_guides/models/llms/#usage-pattern>).">Using LLMs - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/api_reference/node_parsers/token_text_splitter/#llama_index.core.node_parser.TokenTextSplitter>)">Token text splitter - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/optimizing/basic_strategies/basic_strategies/#chunk-sizes>).">Basic Strategies - LlamaIndex</a>: no description found
</li>
</ul>

</div>
  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1265123362246627450)** (53 messages🔥): 

> - `Llama 3.1 Release`
> - `IOL Linguistics Olympiad`
> - `Llama Pricing`
> - `Llama Performance Evaluations`
> - `GPT-4o Mini Fine-Tuning` 


- **Excitement Surrounds Llama 3.1 Launch**: The release of [Llama 3.1](https://ai.meta.com/research/publications/the-llama-3-herd-of-models/) includes the 405B model, marking a significant milestone in open-source LLMs with remarkable capabilities rivaling closed models.
   - Initial evaluations indicate it's the first open model positioned at frontier capabilities, with endorsements from figures like @karpathy praising its accessibility for iterative research and development.
- **International Olympiad for Linguistics (IOL)**: The International Olympiad for Linguistics (IOL) commenced, challenging students to translate lesser-known languages purely using logic, similar to high-stakes math competitions like the IMO.
   - Participants are tackling seemingly impossible problems within a demanding six-hour time frame, sparking interest in how logical reasoning can bridge linguistic gaps.
- **Llama 3.1 Pricing Insights**: Pricing for Llama 3.1's 405B model varies by provider, with indications of costs around $4-5 per million tokens for input and output across platforms like Fireworks and Together.
   - This competitive pricing strategy is seen as potentially aimed at capturing market share before gradually increasing rates as adoption grows.
- **Evaluation of Llama's Performance**: Early evaluations of Llama 3.1 show it performing well within various benchmarks, ranking highly on tasks notably including GSM8K and logical reasoning capabilities on ZebraLogic.
   - In comparison tests, its overall performance lands between Sonnet 3.5 and GPT-4o, though challenges like maintaining schema adherence after extended token lengths were noted.
- **GPT-4o Mini Fine-Tuning Launch**: OpenAI announced the fine-tuning capability for GPT-4o mini, now available for tier 4 and 5 users, with the first 2 million training tokens free each day until September 23.
   - This initiative aims to expand access and customization options over time, with users already evaluating performance against the newly launched Llama 3.1.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/corbtt/status/1815829444009025669?s=46">Tweet from Kyle Corbitt (@corbtt)</a>: Guys fine-tuned Llama 3.1 8B is completely cracked. Just ran it through our fine-tuning test suite and blows GPT-4o mini out of the water on every task.  There has never been an open model this small,...</li><li><a href="https://x.com/deanwball/status/1815826885663658445?s=46">Tweet from Dean W. Ball (@deanwball)</a>: Llama 3 405b is a &#34;systemic risk&#34; to society, according to the European Union and their AI Act.</li><li><a href="https://x.com/sullyomarr/status/1815788922737225771?s=46">Tweet from Sully (@SullyOmarr)</a>: zucc really killed it with llama 3.1   best open source model & almost as good as the best closed model</li><li><a href="https://x.com/naklecha/status/1815808346735378487?s=46">Tweet from naklecha (@naklecha)</a>: today, i&#39;m excited to release factorio-automation-v1. using this mod, your agent can perform game actions like crafting, pathfinding, mining, researching etc. this mod can act as a good playground...</li><li><a href="https://x.com/hrishioa/status/1815811349777375649?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Hrishi (@hrishioa)</a>: Llama3 405B is now part of Mandark (https://github.com/hrishioa/mandark)  Code writing tests: HOW IS IT?  * Needs a lot more prompt tuning, has trouble sticking to schema after about 1K tokens * Just ...</li><li><a href="https://x.com/neuralmagic/status/1815769704415342890?s=46">Tweet from Neural Magic (@neuralmagic)</a>: vLLM now supports deploying Llama-3.1-405B on a single 8xH100 or 8xA100 node, making inference much easier and cheaper!   This is a huge feat by Neural Magic’s engineers who contributed 3 crucial feat...</li><li><a href="https://x.com/JonathanRoss321/status/1815777714642858313">Tweet from Jonathan Ross (@JonathanRoss321)</a>: What can you do with Llama quality and Groq speed? You can do Instant. That&#39;s what. Try Llama 3.1 8B for instant intelligence on http://groq.com.</li><li><a href="https://llama.meta.com/">Llama 3.1</a>: The open source AI model you can fine-tune, distill and deploy anywhere. Our latest models are available in 8B, 70B, and 405B variants.</li><li><a href="https://x.com/togethercompute/status/1815769272536445292">Tweet from Together AI (@togethercompute)</a>: Today marks an inflection point for open source AI with the launch of Meta Llama 3.1 405B, the largest openly available foundation model, that rivals the best closed source models in AI rapidly accele...</li><li><a href="https://x.com/openaidevs/status/1815836887631946015?s=46">Tweet from OpenAI Developers (@OpenAIDevs)</a>: Customize GPT-4o mini for your application with fine-tuning. Available today to tier 4 and 5 users, we plan to gradually expand access to all tiers. First 2M training tokens a day are free, through Se...</li><li><a href="https://x.com/corbtt/status/1815843764960911549">Tweet from Kyle Corbitt (@corbtt)</a>: @altryne @eugeneyan EVALS RUNNING</li><li><a href="https://arxiv.org/abs/2406.03368">IrokoBench: A New Benchmark for African Languages in the Age of Large Language Models</a>: Despite the widespread adoption of Large language models (LLMs), their remarkable capabilities remain limited to a few high-resource languages. Additionally, many low-resource languages (e.g. African ...</li><li><a href="https://x.com/billyuchenlin/status/1815841947468353700?s=46">Tweet from Bill Yuchen Lin 🤖 (@billyuchenlin)</a>: A quick independent evaluation of Llama-3.1-405B-Instruct-Turbo (on @togethercompute) ⬇️  1️⃣ It ranks 1st on GSM8K! 2️⃣ Its logical reasoning ability on ZebraLogic is quite similar to Sonnet 3.5, and...</li><li><a href="https://x.com/aiatmeta/status/1815766327463907421?s=46&t=b7l37rB6wtbyAh6ah1NpZQ">Tweet from AI at Meta (@AIatMeta)</a>: Starting today, open source is leading the way. Introducing Llama 3.1: Our most capable models yet.  Today we’re releasing a collection of new Llama 3.1 models including our long awaited 405B. These m...</li><li><a href="https://x.com/deedydas/status/1815222838623883614">Tweet from Deedy (@deedydas)</a>: The IMO is the hardest high school Math test. A lesser known sibling, the IOL (International Olympiad for Linguistics), starts tomorrow!  Students are asked to translate lesser-known languages purely ...</li><li><a href="https://x.com/aravsrinivas/status/1815800336642367590?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Aravind Srinivas (@AravSrinivas)</a>: Scale AI&#39;s SEAL evals (which I think is a better idea than arena leaderboards simply because you don&#39;t want to hill climb with fake endpoints and have random folks rate based on vibes) suggest...</li><li><a href="https://x.com/thexeophon/status/1815780557445648648?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Xeophon (@TheXeophon)</a>: Llama-405B Pricing:  Fireworks: 3/3 Together: 5/15 Replicate: 9.5/9.5 Groq: Enterprise only for now  Quoting Xeophon (@TheXeophon)   Given the Llama 3 pricing and timing of other providers, it will be...</li><li><a href="https://x.com/francis_yao_/status/1815434157893267554?s=46">Tweet from Yao Fu @ICML (@Francis_YAO_)</a>: As an llm practitioner, this talk is the most informative one I ever listened to so far in the area of the science of LLMs. I was deeply impressed. For my own students I would require them to memorize...</li><li><a href="https://x.com/AIatMeta/status/1815814535313514960">Tweet from AI at Meta (@AIatMeta)</a>: More technical details on the new Llama 3.1 models we released today. 🦙🧵</li><li><a href="https://x.com/karpathy/status/1815842603377779140?s=46">Tweet from Andrej Karpathy (@karpathy)</a>: Huge congrats to @AIatMeta on the Llama 3.1 release! Few notes:  Today, with the 405B model release, is the first time that a frontier-capability LLM is available to everyone to work with and build on...</li><li><a href="https://x.com/realgeorgehotz/status/1815818855190782198?s=46">Tweet from George Hotz 🌑 (@realGeorgeHotz)</a>: Not only were the 405B Llama weights released, they also released a paper explaining how it was made. Nice!  How does any self respecting ML researcher still work at a closed lab? You aren&#39;t savin...</li><li><a href="https://x.com/thexeophon/status/1815780557445648648?s=46&t=6FDPaNx">Tweet from Xeophon (@TheXeophon)</a>: Llama-405B Pricing:  Fireworks: 3/3 Together: 5/15 Replicate: 9.5/9.5 Groq: Enterprise only for now  Quoting Xeophon (@TheXeophon)   Given the Llama 3 pricing and timing of other providers, it will be...</li><li><a href="https://x.com/summeryue0/status/1815776426999877643">Tweet from Summer Yue (@summeryue0)</a>: 🚀 We added Llama 3.1 405B onto the SEAL Leaderboards and it does not disappoint! Here&#39;s how it stacks up:  - 🥇 #1 in Instruction Following - 🥈 #2 in GSM1k - 💻 #4 in Coding  SEAL evals are priv...
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1265340057406148638)** (3 messages): 

> - `Llama 3 Podcast`
> - `Synthetic Data`
> - `RLHF`
> - `Galactica Instruct`
> - `Llama 4 Agents` 


- **Llama 3 Podcast Launch**: A new podcast episode featuring [@ThomasScialom](https://x.com/latentspacepod/status/1815781241398104085) discusses **Llama 2, 3 & 4**, focusing on **Synthetic Data**, **RLHF**, and agents' path to **Open Source AGI**.
   - Listeners are encouraged to check it out [here](https://latent.space/p/llama-3) and engage with the podcast!
- **Galactica Instruct's Potential Impact**: The podcast highlights why **@ylecun's Galactica Instruct** could have effectively resolved **@giffmana's Citations Generator** issues.
   - This insight showcases the practical applications of advanced models in real-world scenarios.
- **Chinchilla Performance Insights**: Discussions included advancements like **100x Chinchilla** as mentioned by **@jefrankle**, emphasizing the movement beyond traditional models.
   - This raises intriguing points about optimizing model efficiency and performance.
- **Native INT8 Training Exploration**: The episode covers **@NoamShazeer's** thoughts on **native INT8 training**, highlighting its implications for model training and deployment.
   - This could shape future methodologies in AI model training strategies.
- **Future of Llama 4 and Agents**: The discussion ventured into **Llama 4's** plans regarding **Agents**, questioning the reasons behind avoiding the Use of **MOE**.
   - These considerations could point to significant design choices impacting the capabilities of future AI models.



**Link mentioned**: <a href="https://x.com/latentspacepod/status/1815781241398104085">Tweet from Latent.Space (@latentspacepod)</a>: 🆕 pod with @ThomasScialom of @AIatMeta!  Llama 2, 3 & 4: Synthetic Data, RLHF, Agents on the path to Open Source AGI  https://latent.space/p/llama-3  shoutouts: - Why @ylecun&#39;s Galactica Instruct...

  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1265170932016877709)** (23 messages🔥): 

> - `AgentState vs InnerAgentState`
> - `Using Chroma Vector Database`
> - `Multi-Character Chatbots in LangChain` 


- **AgentState and InnerAgentState Exploration**: A question was raised about the difference between `AgentState` and `InnerAgentState`. While the definition for `AgentState` was clarified, it's noted that there is insufficient information regarding `InnerAgentState`, suggesting users check the official LangChain documentation.
   - Details regarding `AgentState` include fields like `messages`, `next`, and others depending on context, with references provided for further exploration.
- **Setting Up Chroma Vector Database on Python**: Instructions were provided on how to set up Chroma as a vector database using Python, including installing `langchain-chroma` and running the Chroma server in a Docker container.
   - Examples included using methods like `.add`, `.get`, and `.similarity_search`, emphasizing the need for OpenAI API Key to utilize `OpenAIEmbeddings`.
- **Improv Chatbot Development with LangChain**: A query was made about creating a multi-character improv chatbot using LangChain. While explicit support wasn't confirmed, it was mentioned that LangChain offers features like streaming and message history management which could enable such functionality.
   - Helpful resources from LangChain documentation were shared, including tutorials on Conversational RAG, Agents, and message history management.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://python.langchain.com/v0.2/docs/integrations/vectorstores/chroma/#basic-example-using-the-docker-container>)">Chroma | 🦜️🔗 LangChain</a>: Chroma is a AI-native open-source vector database focused on developer productivity and happiness. Chroma is licensed under Apache 2.0.</li><li><a href="https://github.com/langchain-ai/langchain/issues/19211>)),">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/22191>)).">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1265042962673434725)** (3 messages): 

> - `Scheduler Agent with Composio`
> - `LangGraph and MapReduce`
> - `Llama 3.1 Hosting` 


- **Create a Scheduler Agent using Composio**: A guide was shared detailing steps to create a **Scheduler Agent** leveraging **Composio**, **LangChain**, and **ChatGPT** for event scheduling based on received emails. You can find the guide and [star the repo](https://git.new/scheduler) if you find it useful.
   - The guide highlights how Composio equips agents with well-crafted tools to tackle complex tasks effectively.
- **LangGraph and MapReduce for Parallel Processing**: A post discusses how **LangGraph** and **MapReduce** work together as a dynamic duo for parallel processing tasks in big data. The insights can be found in this detailed [article](https://ai.gopubby.com/langgraph-and-mapreduce-a-dynamic-duo-for-parallel-processing-744cb10da377).
   - The introduction emphasizes how breaking down tasks for parallel execution is a game-changer in complex computations.
- **Llama 3.1 Hosting Available**: A member announced the hosting of **Llama 3.1 405B** and invited others to try it out. The chat is available [here](https://chat.tune.app/) and the API can be accessed [here](https://studio.tune.app/).
   - This hosting provides an opportunity for members to interact with the latest model version in a user-friendly environment.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://git.new/scheduler">composio/python/examples/scheduler_agent at master · ComposioHQ/composio</a>: Composio equips agents with well-crafted tools empowering them to tackle complex tasks - composio/python/examples/scheduler_agent at master · ComposioHQ/composio</li><li><a href="https://chat.tune.app/">Tune Chat - Chat app powered by open-source LLMS</a>: With Tune Chat, access Prompts library, Chat with PDF, and Brand Voice features to enhance your content writing and analysis and maintain a consistent tone across all your creations.</li><li><a href="https://studio.tune.app/">TuneStudio</a>: no description found</li><li><a href="https://ai.gopubby.com/langgraph-and-mapreduce-a-dynamic-duo-for-parallel-processing-744cb10da377">LangGraph and MapReduce: A Dynamic Duo for Parallel Processing</a>: Ankush k Singal
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1265043135684280321)** (5 messages): 

> - `Scheduler Agent`
> - `YouTube Notes Generator`
> - `LangGraph and Flow Engineer`
> - `AI Code Reviewer`
> - `Fully Local Tool Calling with Ollama` 


- **Create Your Own Scheduler Agent with Composio**: A guide was shared detailing steps to create a **Scheduler Agent** using Composio, LangChain, and ChatGPT for scheduling events based on received emails. Check it out [here](https://git.new/scheduler).
   - Composio equips agents with tools that enable them to handle complex tasks effectively, showcased in the [scheduler examples](https://git.new/scheduler).
- **YouTube Notes Generator Launched!**: A new open-source project, the **YouTube Notes Generator**, was announced to assist users in generating notes from YouTube videos. More information can be found [here](https://www.linkedin.com/posts/isham-rashik-5a547711b_machinelearning-artificialintelligence-deeplearning-activity-7221165319464095747-DMDS?utm_source=share&utm_medium=member_desktop).
   - This project aims to simplify note-taking directly from video content, enhancing learning efficiency.
- **Building 10x Reliable Agents with LangGraph**: A video tutorial was released demonstrating how to use **LangGraph** and **Flow Engineer** to build highly reliable agents. Watch it on YouTube [here](https://youtu.be/01g_EfO-Dms?si=tMF70x7MhxKw8S95).
   - The video simplifies the process to boost agent reliability significantly, promoting efficient development practices.
- **AI Code Reviewer with Ollama & LangChain**: A new YouTube video titled **'AI Code Reviewer Ft. Ollama & Langchain'** introduces a CLI tool for effective code reviews. Check out the video [here](https://youtu.be/g_VRsjpC4e8).
   - The tool is designed to revolutionize the code review process, enhancing developers' workflow and productivity.
- **Request for Notebook on Fully Local Tool Calling**: A member requested a notebook for **'Fully local tool calling with Ollama'**, hoping to access the information shared earlier in the day. The session was acknowledged as excellent by the community.
   - This reflects the community's interest in practical implementations of local tool integration techniques.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://youtu.be/g_VRsjpC4e8">AI Code Reviewer Ft. Ollama &amp; Langchain</a>: Welcome to Typescriptic! In this video, we introduce our Code Reviewer, a CLI tool designed to revolutionize the way you review your code. Powered by LangCha...</li><li><a href="https://git.new/scheduler">composio/python/examples/scheduler_agent at master · ComposioHQ/composio</a>: Composio equips agents with well-crafted tools empowering them to tackle complex tasks - composio/python/examples/scheduler_agent at master · ComposioHQ/composio
</li>
</ul>

</div>
  

---



### **Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1265034850059030640)** (26 messages🔥): 

> - `Welcome New Members`
> - `Model Fine-tuning`
> - `Cohere's OCR Capabilities`
> - `RAG Chatbot Discussions`
> - `Community Feedback Evaluation` 


- **Welcome to New Members**: New members, including @thetimelesstraveller and @fullc0de, introduced themselves and expressed excitement about using Cohere.
   - Community members like @xvarunx welcomed them enthusiastically, creating a friendly atmosphere.
- **Fine-tuning Model Progress**: @thetimelesstraveller shared about a new attempt at fine-tuning a model with a dataset called **midicaps**, involving some post-processing.
   - They referenced previous **good results** from similar projects, indicating progress in their efforts.
- **Cohere's OCR Solutions Clarified**: In response to a question about OCR capabilities, @co.elaine informed that Cohere utilizes [unstructured.io](https://unstructured.io).
   - Community discussions revealed that integrating external solutions is feasible, allowing for customization.
- **ChatBot and RAG Implementation Queries**: User @coco.py raised questions about managing chat history and feedback in **RAG-based** ChatBot systems.
   - Responses suggested fitting previous conversations into the context or using **vector databases**, while feedback methods like thumbs up/down were mentioned.
- **Positive Community Vibes**: The community celebrated a recent release, with users expressing excitement and positivity in various comments.
   - @mrdragonfox reiterated community guidelines, ensuring that the environment stays focused and welcoming.


  

---


### **Cohere ▷ #[announcements](https://discord.com/channels/954421988141711382/996880279224451154/1265348202534670427)** (1 messages): 

> - `Rerank 3 Nimble`
> - `Cohere and Fujitsu Partnership` 


- **Introducing Rerank 3 Nimble with Superior Performance**: **Rerank 3 Nimble** launches with **3x higher throughput** than its predecessor Rerank 3, maintaining a significant accuracy level. It's now available on [AWS SageMaker](https://cohere.com/blog/rerank-3-nimble).
   - *Say hello to our new foundation model Rerank 3 Nimble!* The model promises enhanced speed for enterprise search and retrieval-augmented generation (RAG) systems.
- **Cohere and Fujitsu's Strategic Partnership**: Cohere announced a **strategic partnership** with **Fujitsu** to provide AI services specifically for Japanese enterprises. Details can be found in the [blog post](https://blog.fujitsu-partnership).
   - This collaboration aims to leverage both companies' strengths to enhance AI service delivery in the region.



**Link mentioned**: <a href="https://cohere.com/blog/rerank-3-nimble">Introducing Rerank 3 Nimble: Faster Reranking for Enterprise Search &amp; Retrieval-Augmented Generation (RAG) Systems</a>: Today, Cohere is introducing Rerank 3 Nimble: the newest foundation model in our Cohere Rerank model series, built to enhance enterprise search and RAG systems, that is ~3x faster than Rerank 3 while ...

  

---



### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1265066201013424149)** (22 messages🔥): 

> - `Llama 3.1 release`
> - `MPS support and conflicts`
> - `Issues with LoRA`
> - `Git workflow challenges` 


- **Llama 3.1 is officially here!**: Meta released the latest model, **Llama 3.1**, this morning, with support already provided for the 8B and 70B instruct models {@everyone}.
   - The excitement was palpable, leading to some humorous comments about typos and excitement-induced errors.
- **MPS Support and Related Conflicts**: A **pull request** for MPS support was mentioned, which checks for **BF16 on MPS devices** as a critical update.
   - Moreover, ongoing conflicts in the code base were spotlighted, with contributors noting the challenge of keeping branches updated due to frequent changes.
- **LoRA Issues Persist**: An ongoing issue with **LoRA** not working as expected was raised, with suggestions for debugging the implementation.
   - One contributor recalled encountering **CUDA hardcoding** problems during their recent efforts.
- **Navigating Git Workflow Challenges**: **Git workflow** challenges were discussed, specifically surrounding the feeling of constantly facing conflicts after addressing previous ones.
   - A suggestion was made to tweak the workflow to minimize recurrent conflicts, emphasizing the need for effective conflict resolution strategies.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://llama.meta.com/docs/model-cards-and-prompt-formats/llama3_1">Llama 3.1 | Model Cards and Prompt formats</a>: Llama 3.1 - the most capable open model.</li><li><a href="https://pytorch.org/torchtune/0.2/install.html#install-nightly-build)">Install Instructions &mdash; torchtune 0.2 documentation</a>: no description found</li><li><a href="https://github.com/pytorch/torchtune/pull/790">MPS support by maximegmd · Pull Request #790 · pytorch/torchtune</a>: Context  For testing purposes it can be useful to run directly on a local Mac computer.  Changelog  Checks support for BF16 on MPS device. Added a configuration targeting MPS, changes to path were ...
</li>
</ul>

</div>
  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1265292188871299184)** (3 messages): 

> - `MPS support in Torchtune`
> - `Pad ID bug fix`
> - `GitHub Pull Request workflow` 


- **MPS Support Pull Request Discussion**: The pull request titled [MPS support by maximegmd](https://github.com/pytorch/torchtune/pull/790) introduces checks for BF16 on the MPS device, aimed at improving testing on local Mac computers.
   - Discussions indicate potential issues due to the diff being from a common ancestor, with suggestions that a rebase rather than a merge might have been conducted.
- **Pad ID Bug Fix PR Introduced**: A member pointed out a critical bug regarding **pad ID displaying in generate**, leading to the creation of [Pull Request #1211](https://github.com/pytorch/torchtune/pull/1211) to prevent this issue.
   - The PR aims to address the implicit assumption of Pad ID being **0 in utils.generate**, clarifying its impact on special tokens.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/pytorch/torchtune/pull/1211">Prevent pad ids, special tokens displaying in generate by RdoubleA · Pull Request #1211 · pytorch/torchtune</a>: Context What is the purpose of this PR? Is it to   add a new feature  fix a bug  update tests and/or documentation  other (please add here)  Pad ID is implicitly assumed to be 0 in utils.generate, ...</li><li><a href="https://github.com/pytorch/torchtune/pull/790">MPS support by maximegmd · Pull Request #790 · pytorch/torchtune</a>: Context  For testing purposes it can be useful to run directly on a local Mac computer.  Changelog  Checks support for BF16 on MPS device. Added a configuration targeting MPS, changes to path were ...
</li>
</ul>

</div>
  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1265091468771725428)** (15 messages🔥): 

> - `matmul-free-llm with tinygrad`
> - `M1 performance differences`
> - `Testing challenges with `PYTHON=1``
> - `cumsum optimization in tinygrad`
> - `TensorFlow vs PyTorch tensor operations` 


- **Help needed for matmul-free-llm recreation**: There's a request for assistance in recreating **matmul-free-llm** with tinygrad, aiming to leverage [efficient kernels](https://github.com/ridgerchu/matmulfreellm/blob/master/mmfreelm/ops/fusedbitnet.py) while incorporating **fp8**.
   - *Hoping for seamless adaptation to Blackwell fp4 soon*.
- **M1 results differ from CI**: An M1 user is experiencing different results compared to CI, seeking clarification on setting up tests correctly with **conda** and environment variables.
   - *There's confusion due to discrepancies when enabling `PYTHON=1`, as it leads to an IndexError in tests.*
- **cumsum performance concerns**: A newcomer is exploring the **O(n)** implementation of **nn.Embedding** in tinygrad and how to improve **cumsum** from O(n^2) to O(n) using techniques from PyTorch.
   - *There’s speculation about constraints making this challenging, especially as it's a **$1000 bounty**.*
- **TensorFlow and PyTorch tensor operations differences**: Discussion is ongoing about the differences in behavior between **TensorFlow bitcast** and **PyTorch view**, primarily in how dimensions are handled.
   - *Adding or removing dimensions can cause confusion, with some suggesting that the behavior of TensorFlow makes more sense in this context.*
- **Testing issues with bitcast and view**: Testing issues arise with `PYTHON=1` due to the device's support differences between **view** and **bitcast**, causing shape compatibility problems.
   - *There is agreement that while PyTorch and NumPy expand or contract dimensions, TensorFlow's method adds a new dimension.*


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/ridgerchu/matmulfreellm/blob/master/mmfreelm/ops/fusedbitnet.py">matmulfreellm/mmfreelm/ops/fusedbitnet.py at master · ridgerchu/matmulfreellm</a>: Implementation for MatMul-free LM. Contribute to ridgerchu/matmulfreellm development by creating an account on GitHub.</li><li><a href="https://github.com/tinygrad/tinygrad/issues/1612),">Issues · tinygrad/tinygrad</a>: You like pytorch? You like micrograd? You love tinygrad! ❤️  - Issues · tinygrad/tinygrad
</li>
</ul>

</div>
  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1265319278228541532)** (6 messages): 

> - `Incremental Testing in PyTorch`
> - `Molecular Dynamics Engine in Tinygrad`
> - `Gradient Calculations`
> - `Neural Network Potentials` 


- **Seeking Pattern for Incremental Testing with PyTorch**: A member inquired about effective patterns for incrementally testing model performance in the sequence of **Linear, MLP, MoE,** and **LinearAttentionMoE** using PyTorch.
   - They questioned whether starting tests from scratch is more efficient than incremental testing.
- **Developing Molecular Dynamics Engine in Tinygrad**: A group is attempting to implement a **Molecular Dynamics engine** in tinygrad to train models predicting energies of molecular configurations, facing challenges with gradient calculations.
   - They require the gradient of predicted energy concerning input positions for the force, but issues arise because they backpropagate through the model weights twice.
- **Need for Efficient Gradient Calculation**: The developer explained the challenge of calculating the energy/position gradient through a different graph, similar to **torch.autograd.grad** in PyTorch.
   - This is crucial for ensuring the first gradient computation doesn't affect the loss calculation, and they plan to share a minimal example for assistance.
- **Encouragement for PR with Minimal Reproduction**: George Hotz suggested that the developer should post a minimal reproduction of the issue along with the expected behavior to facilitate better assistance.
   - He recommended that this minimal example could ideally be added as a test in a pull request (PR).
- **Connection to Neural Network Potentials**: Another member, James Wiles, queried whether the Molecular Dynamics project is linked to **Neural Network Potentials**.
   - This indicates an interest in how these concepts might intersect within the context of their work.


  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1265055411216846869)** (9 messages🔥): 

> - `Int8 Usage`
> - `ComfyUI Flow`
> - `Llama 3.1 Release`
> - `Whisper Speech Tool`
> - `Zuckerberg's Talk on Llama 3.1` 


- **Discussion on Int8 Implementation**: Members asked about using **Int8**, with one member confirming they could get it working.
   - *Hold a sec* was requested during the discussion, hinting at further support.
- **Guidance on ComfyUI Flow**: A request for sharing a script led to the response to use the **ComfyUI flow** for setup.
   - This reflects a preference for streamlined workflows in the community.
- **Llama 3.1 Update Shared**: A member referred to the **Llama 3.1 blog** in a specific channel, indicating significant interest in updates.
   - This highlights ongoing discussions around advancements in Llama models.
- **Query on Whisper Speech Tool**: There was a question about the working condition of the **Whisper Speech** tool at the provided link.
   - Members engaged in checking the current status of this tool, showing active community participation.
- **Zuckerberg Discusses Llama 3.1**: A member shared a [YouTube video](https://youtu.be/Vy3OkbtUa5k?si=mBhzPQqDLgzDEL61) where Mark Zuckerberg discusses **Llama 3.1** and its competitive advantages.
   - The video emphasizes Llama 3.1 as the first-ever open-sourced frontier AI model, achieving notable benchmarks.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://youtu.be/Vy3OkbtUa5k?si=mBhzPQqDLgzDEL61">Mark Zuckerberg on Llama 3.1, Open Source, AI Agents, Safety, and more</a>: Meta just released Llama 3.1 405B — the first-ever open-sourced frontier AI model, beating top closed models like GPT-4o across several benchmarks. I sat dow...</li><li><a href="https://collabora-whisperspeech.hf.space/">Gradio</a>: no description found
</li>
</ul>

</div>
  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1265065927959908464)** (5 messages): 

> - `Meta's commitment to open source AI`
> - `Llama 3.1 capabilities`
> - `Context length improvements` 


- **Meta champions open source AI**: Meta is dedicated to [openly accessible AI](https://about.fb.com/news/2024/07/open-source-ai-is-the-path-forward/) as detailed in Mark Zuckerberg’s letter, emphasizing its benefits for developers and the broader community.
   - This aligns with their vision of fostering innovation through collaboration in the AI ecosystem.
- **Llama 3.1 sets a new benchmark**: The release of [Llama 3.1 405B](https://ai.meta.com/research/publications/the-llama-3-herd-of-models/) introduces unprecedented capabilities, including a context length of **128K** and support for eight languages.
   - This model provides flexibility and control, positioning itself competitively against top closed-source alternatives.
- **Context Size Criticism Addressed**: A discussion highlighted that the previous **8K** context size was considered inadequate for handling large documents efficiently.
   - The leap to **128K** context size is seen as a significant improvement for tasks requiring substantial document processing.



**Link mentioned**: <a href="https://ai.meta.com/blog/meta-llama-3-1/">no title found</a>: no description found

  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1265406571203137587)** (1 messages): 

> - `Llama 3.1 405 B`
> - `GPT-4o performance` 


- **Llama 3.1 405 B Amazes Users**: **Llama 3.1 405 B** is reported to work fantastically out of the box with **OpenInterpreter**.
   - Users noted that unlike **GPT-4o**, there's no need for constant reminders or restarts to complete multiple tasks.
- **Frustrations with GPT-4o**: A user expressed challenges with **GPT-4o**, requiring frequent prompts to perform tasks on their computer.
   - This frustration highlights the seamless experience users are having with **Llama 3.1 405 B**.


  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1265049476913954969)** (3 messages): 

> - `Voice Input with Coqui Model`
> - `Expo App for Apple Watch`
> - `Device Shipping Timeline` 


- **Voice Input on MacOS with Coqui Model?**: A query was raised about the feasibility of using voice input with a local **Coqui model** on **MacOS**.
   - No responses were yet provided detailing any successful implementations.
- **Expo App's Capability for Apple Watch**: There was a discussion affirming that the **Expo app** should theoretically be able to build applications for the **Apple Watch**.
   - *Further details or confirmations were not provided*.
- **Shipping Timeline for the Device**: A member inquired about the **shipping timeline** for a specific device, signaling an ongoing curiosity about its status.
   - No updates or timelines were shared in the conversation.


  

---



### **Alignment Lab AI ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/)** (1 messages): 

spirit_from_germany: https://youtu.be/Vy3OkbtUa5k?si=mBhzPQqDLgzDEL61
  

---


### **Alignment Lab AI ▷ #[open-orca-community-chat](https://discord.com/channels/1087862276448595968/1124000038205530182/1265354431889674271)** (2 messages): 

> - `OpenOrca dataset licensing`
> - `Synthetic dataset announcement` 


- **Clarification on OpenOrca Dataset Licensing**: A member sought clarification on the licensing of the **OpenOrca** dataset, specifically questioning whether its **MIT License** allows for commercial use of its outputs given its derivation from the **GPT-4 Model**.
   - *Can its outputs be used for commercial purposes?*
- **Plans for Open Sourcing Synthetic Dataset**: Another member announced plans to open source a **synthetic dataset** that will support both non-commercial and commercial applications.
   - They mentioned evaluating a dependency on **OpenOrca** while creating the dataset, indicating an interest in its licensing implications.


  

---



### **LLM Finetuning (Hamel + Dan) ▷ #[east-coast-usa](https://discord.com/channels/1238365980128706560/1245411101718610001/1265144542382919792)** (2 messages): 

> - `Miami meetup`
> - `NYC interest in August` 


- **Inquiries for Miami Meetups**: A member asked if anyone is near **Miami**, potentially looking for meetups or gatherings.
   - No further details or responses were shared regarding this inquiry.
- **Interest in NYC Gathering**: Another member expressed interest in attending meetups in **NYC** in late **August**.
   - This inquiry opened up potential opportunities for connection among members in the area.


  

---



### **AI Stack Devs (Yoko Li) ▷ #[team-up](https://discord.com/channels/1122748573000409160/1128471951963328512/)** (1 messages): 

ari991963: Hi all, I am Aria a 2D/3D artist, if you are interested to collaborate dm
  

---



### **Mozilla AI ▷ #[announcements](https://discord.com/channels/1089876418936180786/1089876419926032396/1265387661103726703)** (1 messages): 

> - `Mozilla Accelerator Application Deadline`
> - `Zero Shot Tokenizer Transfer Event`
> - `AutoFix Project Overview` 


- **Mozilla Accelerator Application Deadline Approaches**: The application deadline for the **Mozilla Accelerator** is fast approaching, offering a **12-week program** with up to **$100k** in non-dilutive funds.
   - Participants will also have the opportunity to showcase their projects during a **demo day** with Mozilla. [Questions?](https://discord.com/channels/1089876418936180786/1245083732319408195)
- **Get Ready for Zero Shot Tokenizer Transfer Event**: A reminder of an upcoming event featuring **Zero Shot Tokenizer Transfer** with Benjamin Minixhofer, scheduled this month.
   - Details can be found in the event's link. [Event Information](https://discord.com/channels/1089876418936180786/1089876419926032396/1261387457652592732)
- **Introducing AutoFix: The Open Source Issue Fixer**: **AutoFix** is an open-source issue fixer that can submit PRs directly from **Sentry.io**, providing an efficient way to manage issues.
   - You can learn more about this innovative tool in the detailed post linked. [AutoFix Information](https://discord.com/channels/1089876418936180786/1089876419926032396/1261387457652592732)


  

---



---



---



---



{% else %}


> The full channel by channel breakdowns have been truncated for email. 
> 
> If you want the full breakdown, please visit the web version of this email: [{{ email.subject }}]({{ email_url }})!
>
> If you enjoyed AInews, please [share with a friend](https://buttondown.email/ainews)! Thanks in advance!

{% endif %}
