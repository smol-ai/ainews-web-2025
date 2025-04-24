---
id: b8bacc64-9fc5-4299-b895-acc178b286d2
title: not much happened today
date: '2024-10-18T01:13:21.878974Z'
original_slug: ainews-not-much-happened-today-7086
description: >-
  Meta released Llama 3.1, now powering Lenovo AI Now for on-device AI tasks,
  while Yi announced Yi-Lightning, a new model ranked #6 globally and surpassing
  GPT-4o. Additionally, ZyphraAI introduced Zyda-2, a massive 5 trillion token
  dataset that improves downstream model performance.
tags:
  - meta
  - llama-3-1
  - lenovo
  - yi
  - yi-lightning
  - zyphra
  - zyda-2
  - dataset
  - large-language-models
---


<!-- buttondown-editor-mode: plaintext -->**lots of small ships is all you need.**

> AI News for 10/16/2024-10/17/2024. We checked 7 subreddits, [**433** Twitters](https://twitter.com/i/lists/1585430245762441216) and **31** Discords (**228** channels, and **2989** messages) for you. Estimated reading time saved (at 200wpm): **280 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

- Answer.ai shipped [fastdata](https://www.answer.ai/posts/2024-10-15-how-to-synthesize-data.html), a synthetic data generation library that uses `claudette` + the Tencent [Billion Persona paper](https://arxiv.org/abs/2406.20094v1) 
- NotebookLM is [finally customizable])https://x.com/raiza_abubakar/status/1846944566689353838)
- Motherduck shipped a [notable LLMs in SQL implementation](https://motherduck.com/blog/sql-llm-prompt-function-gpt-models/)
- Both [Perplexity](https://x.com/perplexity_ai/status/1846950770736091509?s=46) and [Dropbox](https://dash.dropbox.com/) announced their Glean competitors
- As teased at Devday, OpenAI announced [audio chat completions](https://platform.openai.com/docs/guides/audio) that are pricey at 24 cents per minute.
 

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

**AI Model Updates and Developments**

- **Llama 3 Release**: [@AIatMeta](https://twitter.com/AIatMeta/status/1846330458189320534) announced the release of Llama 3.1, which is being used in Lenovo AI Now, an on-device AI agent enabling capabilities from document management to content generation.

- **Yi-Lightning Model**: [@01AI_Yi](https://twitter.com/01AI_Yi/status/1846339181863473443) announced the release of Yi-Lightning, now ranked #6 in the world, surpassing the original GPT-4o released 5 months ago. The company is ranked #3 LLM player on @lmarena_ai Chatbot Arena.

- **Zephyr AI Dataset**: [@ZyphraAI](https://twitter.com/rohanpaul_ai/status/1846288338913054734) released Zyda-2, a 5 trillion token permissively licensed dataset composed of DCLM, FineWeb-Edu, Zyda-1, and Dolma v1.7's Common Crawl. The dataset outperforms individual component datasets and models trained on it show stronger performance on downstream tasks.

**AI Research and Techniques**

- **Transformer Architecture**: [@fchollet](https://twitter.com/fchollet/status/1846263128801378616) explained that Transformers are a set-processing architecture, not sequence-processing. They are order-agnostic, and position awareness is added at the feature level through position embeddings.

- **LLM Reasoning**: A [paper](https://twitter.com/rohanpaul_ai/status/1846302588167192766) suggests that memorization can enhance genuine reasoning abilities in LLMs, enabling models to generalize better to new and varied problems.

- **AI Safety**: [@AnthropicAI](https://twitter.com/AnthropicAI/status/1846194917720088721) published an update to their Responsible Scaling Policy, matching safety and security measures to an AI model's capabilities.

**AI Tools and Applications**

- **Perplexity Finance**: [@AravSrinivas](https://twitter.com/AravSrinivas/status/1846289701822677441) highlighted Perplexity Finance, offering real-time stock prices, deep dives into company financials, and comparison of multiple companies with a user-friendly interface.

- **Open Canvas**: [@LangChainAI](https://twitter.com/LangChainAI/status/1846215982765035677) introduced Open Canvas, an open-source web application for collaborating with agents to write documents, featuring built-in memory and the ability to start from existing documents.

- **AlphaCodium**: [@svpino](https://twitter.com/svpino/status/1846201354332893220) reported on AlphaCodium, an open-source state-of-the-art code generation tool that outperforms direct prompting of OpenAI models on the Codeforces Code Contest benchmark.

**AI Industry and Market Trends**

- **AI Agent Startups**: [@swyx](https://twitter.com/swyx/status/1846305962841280667) noted that approximately $500 million was raised this month for AI agent startups, with none known to be using AI agent frameworks from other startups.

- **AI Job Market**: [@svpino](https://twitter.com/svpino/status/1846297492499190013) commented on the ongoing discussion about AI's impact on jobs, stating it's been 685 days since he was told AI was taking his job.

- **AI Pricing**: [@alexalbert__](https://twitter.com/alexalbert__/status/1846265564852809854) pointed out that combining prompt caching with the new Batches API can result in a 95% discount on Claude 3.5 Sonnet tokens.


---

# AI Reddit Recap

## /r/LocalLlama Recap

**Theme 1. Ollama Integration with 45K Hugging Face GGUF Models**

- **PSA: You can clone any Huggingface "Spaces" setup locally very easily** ([Score: 40, Comments: 1](https://reddit.com//r/LocalLLaMA/comments/1g51saf/psa_you_can_clone_any_huggingface_spaces_setup/)): **Hugging Face Spaces** can be easily cloned and run locally, providing a quick way to set up and use models with a visual interface. The process involves **cloning the Space repository**, creating a **virtual environment**, installing requirements, and running the app, as demonstrated with an example command sequence for a **text-to-speech model**.
- **You can now run *any* of the 45K GGUF on the Hugging Face Hub directly with Ollama 🤗** ([Score: 314, Comments: 63](https://reddit.com//r/LocalLLaMA/comments/1g4zvi5/you_can_now_run_any_of_the_45k_gguf_on_the/)): **Ollama** now supports direct running of any of the **45,000 GGUF models** from the **Hugging Face Hub** without requiring changes to the Ollama setup. Users can run models using the command `ollama run hf.co/{username}/{reponame}:latest`, with options to specify quantization types like `Q8_0`. For more information, users can refer to the [Hugging Face documentation](https://huggingface.co/docs/hub/en/ollama).
  - **Ollama** integration with **Hugging Face Hub** is seen as a significant improvement, allowing users to directly run **45,000 GGUF models** without manual configuration. This update streamlines the process of downloading, installing, and running models to a single command.
  - Users discussed the impact on **OpenWebUI**, confirming that models can be pulled directly from Hugging Face within the interface. Some expressed interest in **Vulkan support** for improved performance on Linux systems without extensive dependencies.
  - Questions arose about model storage locations, the ability to run previously downloaded models without conversion, and potential support for **vision models**, **text-to-image models**, and **TTS/STT** capabilities through this new integration.


**Theme 2. Mistral AI's New Ministral Models and Licensing Debate**

- **[Un Ministral, des Ministraux](https://mistral.ai/news/ministraux/)** ([Score: 39, Comments: 10](https://reddit.com//r/LocalLLaMA/comments/1g50sbn/un_ministral_des_ministraux/)): Mistral AI has released new models including **Mistral 7B**, **Mixtral 8x7B**, and **Mistral Small**, with the latter two being **commercially licensed**. The company's decision to restrict access and impose licensing fees for some models has sparked debate about the balance between open-source principles and commercial interests in AI development. This shift in Mistral's approach contrasts with their initial commitment to open-source models and raises questions about the future direction of AI model distribution and accessibility.
  - **Mistral's new models** spark debate on open-source vs. commercial AI development. Some users express disappointment with the **restrictive licensing**, with one stating "No Apache Licence, F * Irrelevant".
  - The **multilingual capability** of new models is noted as the biggest advancement, though not considered hugely exciting by some users. Others look forward to trying the models, hoping they will "punch above their weight" like previous Mistral offerings.
  - The **research license** for the 8B model is viewed positively by some for ERP research. However, concerns are raised about the lack of weights for the 3B model and the restrictive nature of the 8B license.
- **Why ther is no middle ground version of llama between 8 and 70b?** ([Score: 46, Comments: 80](https://reddit.com//r/LocalLLaMA/comments/1g4wul3/why_ther_is_no_middle_ground_version_of_llama/)): The post questions the **absence of mid-sized Llama models** between **8B and 70B** parameters, highlighting a gap in options for users with **8-16GB GPUs**. The author notes that while a **4GB 3050 GPU** can run the **8B model** adequately, there's no suitable option for more powerful consumer GPUs that can't handle the **70B model**. They suggest developing a **16B parameter model** to fill this gap in the Llama model lineup.
  - Users discussed the potential for **home labs** and **consumer-grade AI hardware**, with some suggesting that tinkerers might soon have personal "hardware brains" for AI processing.
  - **Meta's Llama models** are not designed with consumer GPUs in mind; the **8B model** is considered the "local" version, while larger models target datacenters. Some users recommended alternatives like **Gemma 2's 9B and 27B models** as ideal mid-sized options.
  - The community debated the absence of a **mid-sized Llama model**, with mentions of a **32.5B original model** and a failed **Llama 2 mid-sized version**. Some suggested trying other models like **Qwen2.5 14B**, which reportedly outperforms **Llama 3.1 8B**.

- **[Mistral releases new models - Ministral 3B and Ministral 8B!](https://i.redd.it/45hs1duoq4vd1.png)** ([Score: 313, Comments: 74](https://reddit.com//r/LocalLLaMA/comments/1g50x4s/mistral_releases_new_models_ministral_3b_and/)): Mistral has released two new models, **Ministral 3B** and **Ministral 8B**, claiming performance improvements over previous versions. The company asserts that **Ministral 8B** outperforms **Llama 2 13B** on most benchmarks, while **Ministral 3B** is said to match or exceed **Llama 2 7B**'s performance, potentially offering significant efficiency gains for developers and researchers working with smaller-scale language models.
  - **Qwen2.5** outperforms Mistral's new models on most benchmarks, with users noting its superior performance on **HumanEval** (84.8 vs 76.8) and **MATH** (75.5 vs 54.5) at the 7B/8B scale. Some call Mistral's release "deceptive" for omitting Qwen2.5 comparisons.
  - The **Ministral 3B** model is only available via API, despite being marketed for edge devices. Users express disappointment with the licensing terms, noting that the 8B model is restricted to non-commercial use unless negotiating a commercial license.
  - Discussion around **interleaved sliding-window attention** implementation in llama.cpp, with users referencing a [GitHub pull request](https://github.com/ggerganov/llama.cpp/pull/8227) for Gemma2 support and speculating on potential conversion code needed for Mistral models.


**Theme 3. Threadripper with 4xRTX4090**

- **[6U Threadripper + 4xRTX4090 build](https://i.redd.it/h1ic1yk6h3vd1.jpeg)** ([Score: 774, Comments: 182](https://reddit.com//r/LocalLLaMA/comments/1g4w2vs/6u_threadripper_4xrtx4090_build/)): A high-performance AI build featuring a **6U Threadripper** processor and **4 RTX 4090** graphics cards was showcased. This powerful configuration is designed for demanding AI and machine learning tasks, leveraging the computational capabilities of NVIDIA's top-tier GPUs and AMD's high-core-count CPU.
  - The build sparked discussions about **power consumption**, with estimates of **3 kW** usage and concerns about electricity bills. Users debated whether someone investing in such a setup would worry about power costs.
  - Details of the build were shared, including a **Threadripper Pro 7965WX**, **256GB RAM**, and **two PSUs** (1500W and 1300W). The system uses **water cooling** with 2x radiators and several 360mm fans.
  - Users inquired about performance, with the OP noting **max GPU temps of 79-81°C** during 24-hour load testing. Some suggested alternatives like [renderboxes.com](https://renderboxes.com/) for pre-built high-performance systems.


**Theme 4. Meta's TPO Technique Boosts LLM Performance**

- **New paper from Meta discloses TPO (Thought Preference Optimization) technique with impressive results** ([Score: 43, Comments: 6](https://reddit.com//r/LocalLLaMA/comments/1g51w11/new_paper_from_meta_discloses_tpo_thought/)): Meta's new paper introduces **Thought Preference Optimization (TPO)**, a technique that significantly improved the **Llama 3.1 8B** model's performance to match **GPT-4** on **AlpacaEval** and **ArenaHard** benchmarks. The paper details experiments and results of this technique, which is similar to that used in **o1 models**, demonstrating impressive gains in general instruction following capabilities.
  - Users expressed amusement at the rapid progress in **AI benchmarks**, with **8B models** now matching **GPT-4**'s performance, contrasting with expectations from a year ago.
  - Several commenters inquired about the availability of the **TPO weights** and implementation details, highlighting interest in replicating the technique.
  - The community noted a surge in significant AI research papers, including **Differential Transformers** from Microsoft and **Chain of Thought Reasoning** from Google, alongside speculation about applying **TPO** to larger models like **Llama-3.1-70B**.

- **Entropy Decoding in Optillm + Early Results on GSM8k** ([Score: 30, Comments: 5](https://reddit.com//r/LocalLLaMA/comments/1g5gf27/entropy_decoding_in_optillm_early_results_on_gsm8k/)): **Optillm** has implemented **entropy decoding** based adaptive sampling, inspired by @\_xjdr's work on entropix. An evaluation of this technique on the **GSM8k** benchmark using the **Qwen2.5-0.5B-Instruct** model in a zero-shot setting showed improvements over the base model, but did not surpass the results achieved with **Chain of Thought (CoT) decoding**. A [Google Colab notebook](https://colab.research.google.com/drive/1SpuUb8d9xAoTh32M-9wJsB50AOH54EaH?usp=sharing) is available for testing both methods.
  - Users expressed interest in implementing **entropy decoding** in other frameworks like **vLLM** and **llama.cpp**. Some encountered difficulties setting up **optillm** with **llama-server** and **tabbyapi**, experiencing **404** and **401 errors**.
  - The developer provided troubleshooting resources, including a [GitHub issue](https://github.com/codelion/optillm/issues/8#issuecomment-2356788401), a [Hugging Face space](https://huggingface.co/spaces/codelion/optillm), and the original [Google Colab notebook](https://colab.research.google.com/drive/1SpuUb8d9xAoTh32M-9wJsB50AOH54EaH?usp=sharing) for testing.
  - A potential flaw in **optillm's Chain of Thought (CoT) decoding** implementation was pointed out, noting that the **confidence score** should be calculated only from the answer span, not the entire sequence. The developer questioned how to generally identify the answer part.

## Other AI Subreddit Recap

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity


**AI Research and Advancements**

- **Nvidia Nemotron 70B model outperforms larger models**: Nvidia released their [Nemotron 70B model](https://www.reddit.com/r/singularity/comments/1g4xd7e/nvidia_nemotron_70b_beats_llama_31_405b_gpt4o/) which reportedly beats Llama 3.1 405B, GPT-4o and Claude 3.5 Sonnet on several benchmarks. They released the instruct model, reward model and dataset on Hugging Face.

- **EgoAllo estimates 3D human body pose from head-mounted cameras**: Researchers developed [EgoAllo](https://www.reddit.com/r/singularity/comments/1g4wsx6/egoallo_can_estimate_3d_human_body_pose_height/), a system that can estimate 3D human body pose, height, and hand parameters using images from a head-mounted device. This could have applications in VR/AR.

- **Breakthrough in visual reasoning for AI**: University of Toronto researchers improved visual transformers for the [ARC challenge](https://www.reddit.com/r/singularity/comments/1g4xsjn/a_breakthrough_in_visual_reasoning/), achieving close to 100% solve rate on over half of 400 public ARC tasks through supervised learning. However, this approach may not generalize well to the full ARC benchmark.

**AI Industry and Company News**  

- **Tesla's Optimus robot shows improvements**: Tesla released an [update video on Optimus](https://www.reddit.com/r/singularity/comments/1g5khpb/update_on_optimus/), demonstrating improved walking, object manipulation, and autonomous navigation. However, there is debate about how much was autonomous vs. teleoperated.

- **OpenAI claims harassment by Elon Musk**: OpenAI is [claiming that Elon Musk is harassing their company](https://www.reddit.com/r/OpenAI/comments/1g525hy/openai_is_claiming_that_elon_musk_is_harassing/), related to disputes over OpenAI's shift from non-profit to for-profit status.

- **Amazon investing in nuclear technology**: Amazon announced plans to [invest over $500 million](https://www.reddit.com/r/singularity/comments/1g512da/amazon_goes_nuclear_to_invest_more_than_500/) to develop small modular nuclear reactors, potentially for powering data centers.

**AI Ethics and Societal Impact**

- **AI-generated Wikipedia articles increasing**: A study found that [at least 5% of new Wikipedia articles in August were AI-generated](https://www.reddit.com/r/OpenAI/comments/1g5gzag/at_least_5_of_new_wikipedia_articles_in_august/), though the accuracy of AI detection methods is debated.

- **Yann LeCun comments on AI hype**: AI pioneer Yann LeCun [shared thoughts on current AI hype](https://www.reddit.com/r/singularity/comments/1g5b2lq/yann_lecun_on_the_ai_hype/), though details were not provided in the comments.

**AI Policy and Regulation**

- **Emmanuel Macron warns of overregulation**: French President Emmanuel Macron [warned that Europe risks falling behind](https://www.reddit.com/r/singularity/comments/1g4x7fc/emmanuel_macron_we_are_overregulating_and/) in AI due to overregulation and underinvestment, stating "We are overregulating and under-investing. So just if in the 2 to 3 years to come, if we follow our classical agenda, we will be out of the market."


---

# AI Discord Recap

> A summary of Summaries of Summaries by O1-mini

Theme 1. **Advancements in LLM Performance and Benchmarking**

- [**NVIDIA Nemotron 70B Dominates Benchmarks**](https://openrouter.ai/nvidia/llama-3.1-nemotron-70b-instruct): The **NVIDIA Nemotron 70B** outperforms **Llama 3.1 405B**, **GPT-4o**, and **Claude 3.5 Sonnet** across multiple evaluations, achieving top scores in **Arena Hard** and **AlpacaEval 2 LC**.
- [**Llama 3.1 vs. Mistral 7B: Performance Gap Revealed**](https://blog.eleuther.ai/mad_research_update_2/): **MAD** tests show that **Mistral 7B v0.1** outperforms **Llama 3.1 8B** on non-arithmetic tasks, highlighting differences in behavior and loss metrics.
- [**GLM-4-Plus and Yi-Lightning's Rise in Chatbot Arena**](https://lmarena.ai): **GLM-4-Plus** from Zhipu AI and **Yi-Lightning** have surged into the **top 10** rankings, showcasing the competitive advancements of Chinese **LLMs** in areas like Math and Coding.

Theme 2. **New AI Tools and Platform Features**

- [**Hugging Face Launches Community Tools for Enhanced Interactions**](https://huggingface.co/spaces/huggingchat/chat-ui/discussions/569): The new **Hugging Face Community Tools** enable users to create custom tools on **HuggingChat**, incorporating video and speech modalities to enrich user-model interactions.
- [**OpenRouter Introduces NVIDIA Models and Competitive Pricing**](https://openrouter.ai/x-ai/grok-2): **OpenRouter** adds **SambaNova** and **Yi Lightning** models with competitive pricing, fostering the adoption of pay-as-you-go models for **in-house chip inference** providers.
- [**NotebookLM Enhances Features with Custom Audio and Business Support**](https://notebooklm.google/): **NotebookLM** now allows users to provide custom audio instructions before generating audio and has launched a Business version via Google Workspace, improving collaboration tools.

Theme 3. **Optimization and Training Techniques for LLMs**

- [**Muon Optimizer Outperforms AdamW in Efficiency and Performance**](https://developer.nvidia.com/blog/enhancing-rag-pipelines-with-re-ranking/): The **Muon optimizer** achieves lower validation loss and reduced token usage compared to **AdamW**, especially on larger models, thanks to its new distributed implementation.
- [**LLM Re-ranking Techniques Boost Search Accuracy**](https://developer.nvidia.com/blog/enhancing-rag-pipelines-with-re-ranking/): Implementing **LLM re-ranking techniques** using machine learning algorithms enhances alignment with user intent, refining search results for greater relevance.
- [**ControlNet Training with CLIP Encoders Sparks Debate**](https://huggingface.co/qwen2.5): Retaining **CLIP encoders** in **ControlNet** training raises concerns about potential overfitting and the implications for generating accurate captions.

Theme 4. **API Performance and Integration Challenges**

- [**Perplexity API Faces Sluggish Response Times**](https://perplexity.ai): Users report that the **Perplexity API** experiences slow response times, taking **1 to 2 minutes** for basic queries, leading to benchmarking discussions and unmet performance expectations.
- [**Torchtune Updates Align with PyTorch 2.5 Release**](https://github.com/pytorch/torchtune/issues/1861): **Torchtune** introduces support for **PyTorch 2.5**, featuring [FlexAttention](https://github.com/pytorch/pytorch/releases/tag/v2.5.0) and **per-layer compile**, encouraging users to upgrade for improved performance.
- [**Integration Issues with OpenInterpreter and Aider Persist**](https://github.com/OpenInterpreter/open-interpreter/tree/main/scripts): Users encounter persistent issues with **OpenInterpreter** tasks not executing and **Aider** installation problems across platforms, prompting ongoing troubleshooting and community support efforts.

Theme 5. **Community Engagement: Hackathons and Collaborative Initiatives**

- [**Gen AI Agents Hackathon Invites Innovators**](https://lu.ma/ke0rwi8n): Hosted by **CreatorsCorner** with tech partners, the **Gen AI Agents Hackathon** encourages participants to build **AI-powered multi-agent systems** while considering ethical implications and enhancing human potential.
- [**Bitnet Releases Official 1-bit LLM Framework**](https://github.com/microsoft/BitNet): **Bitnet** launches its official inference framework for 1-bit LLMs on **GitHub**, enabling efficient model execution and fostering research collaborations.
- [**DSPy's Langtrace Integration Fuels Collaborative Projects**](https://docs.langtrace.ai/supported-integrations/llm-frameworks/dspy): The integration of **Langtrace** with **DSPy** facilitates advanced data handling and multi-label classification, with community members contributing to prompt optimizations and documentation enhancements.

---

# PART 1: High level Discord summaries

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Hugging Face Community Tools Launch**: The Hugging Face community tools allow users to create custom tools on HuggingChat, catering to various modalities like video and speech for enhanced user interaction.
  
  - This feature opens new avenues for model capabilities, fostering user collaboration and innovation.
- **Efforts to Accelerate LLM Training**: A member introduced a platform to store and stream data specifically for LLM training between HuggingFace and S3, addressing data management challenges.
  
  - Demo requests are encouraged as the platform is eager for feedback to further refine its features.
- **Insights into Object Detection Methods**: Discussion revolved around utilizing models like YOLO for object detection, with mentions on the importance of bounding boxes for accuracy.
  
  - Suggestions included incorporating semantic segmentation with models like SAM for per-pixel labeling, improving detection detail.
- **NLP Fine-tuning Dataset Format Queries**: A member asked about using an instruct formatted dataset to fine-tune a base model while confirming that using a raw text dataset might yield inaccurate outputs.
  
  - The need to ensure dataset compatibility for domain-specific knowledge highlights the importance of careful dataset selection.
- **ControlNet Training with CLIP Encoders Discussion**: Members discussed retraining ControlNet with a new fine-tuned model, raising concerns over the potential risk of overfitting to specific datasets.
  
  - Utilizing CLIP encoders instead of text ones sparked debate on the implications for generating captions and training prudence.

 

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Gandalf Challenges Yield High Success**: Participants experienced success in the **Gandalf challenges**, employing creative prompt strategies to achieve high rankings.
  
  - Methods like asking for lists with hidden criteria and playing 21 questions showcased the iterative nature of the challenges.
- **Ollama simplifies GGUF Model Execution**: Ollama allows users to run **GGUF models** from Hugging Face using `ollama run <model_url>`, streamlining the process.
  
  - With **45K public GGUF checkpoints**, it enhances the experience with customizable options for quantization type and system prompts.
- **SCP Generator Launched on GitHub**: A new [SCP generator](https://github.com/dottxt-ai/cursed/tree/main/scp) helps create SCP stories using outlines provided by dottxt-ai.
  
  - This open-source project invites contributions, prompting developers to join in its development.
- **Debate Over LLM Programming Languages**: A member inquired about which programming language is best for top LLMs, questioning **JavaScript** versus **Python**.
  
  - Opinions varied, with one member asserting LLMs are **entangled in Python** while advocating for more **JavaScript coding**.
- **Resources for LLM Jailbreaks Discussed**: Discussion on resources for **LLM jailbreaks** included a mention of checking out **plineys discord**.
  
  - Confusion within that community prompted calls for alternative resources.

 

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **MAD performance reveals model disparities**: Recent tests on **mechanistic anomaly detection (MAD)** found that **Llama 3.1 8B** underperformed on non-arithmetic tasks compared to **Mistral 7B v0.1**, highlighting a significant performance gap.
  
  - **Llama** exhibited less quirky behavior but had a stronger ground truth bias, achieving lower average loss across tasks.
- **Advanced LLM re-ranking boosts accuracy**: Participants discussed the effectiveness of **LLM re-ranking techniques** using machine learning algorithms to refine search results, according to this [implementation](https://developer.nvidia.com/blog/enhancing-rag-pipelines-with-re-ranking/).
  
  - The goal of these methods is to better align outputs with user intent, providing more relevant information.
- **Muon optimizer outshines AdamW**: The **Muon optimizer** shows improved performance with lower validation loss and reduced token usage compared to AdamW, particularly on larger models like GPT-2.
  
  - Its new distributed implementation demonstrates significant efficiency gains in training, with users noting success on models up to 1.5B parameters.
- **Searching for model hallucination metrics**: Discussions emerged around identifying **reliable methods** for evaluating and quantifying model **hallucinations**, with members seeking relevant research papers.
  
  - There's a growing interest in establishing robust metrics for assessing model output fidelity.
- **Saving model outputs during tests**: Members discussed strategies for saving content generated by models during testing phases, with suggestions to use the `--log_samples` parameter.
  
  - This feature may assist in retaining output generated during experimentation.

 

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **NVIDIA Nemotron 70B Crushes Competition**: The **NVIDIA Nemotron 70B** has outperformed **Llama 3.1 405B**, **GPT-4o**, and **Claude 3.5 Sonnet** in several evaluations, reporting scores of **85.0** in Arena Hard, **57.6** in AlpacaEval 2 LC, and **8.98** in MT Bench.
  
  - You can check out the results and try it [here](https://openrouter.ai/nvidia/llama-3.1-nemotron-70b-instruct).
- **Grok 2 Returns with Price Hikes**: **Grok 2** is now priced at **$5/m input** and **$10/m output**, with the mini version still unavailable, startling users who discussed the implications of the increase.
  
  - More details on its features can be found [here](https://openrouter.ai/x-ai/grok-2).
- **OpenRouter Models and Pricing Insights**: Discussions highlighted various models available through **OpenRouter**, including **SambaNova** and **Yi Lightning**, which boasts a competitive rate of **$0.14/m** input.
  
  - There’s speculation about forthcoming insights into the pricing of in-house chip inference providers as pay-as-you-go models gain traction.
- **Voice Interaction Models Lack Consistency**: Concerns surfaced regarding voice features in models like **GPT-4o**, particularly their handling of multiple languages where output quality suffers.
  
  - Users noted that while voice input is decent, the output becomes 'funky', especially in languages such as Chinese.
- **O1 Model Under the Microscope**: Users debated the performance of the **O1 model**, particularly its struggles with instruction following and maintaining coherent outputs.
  
  - Concerns were voiced regarding its usefulness across various tasks due to issues with excessively rambling responses.

 

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity API Response Times Lag**: Users report that the **Perplexity API** response times are sluggish, taking between **1 to 2 minutes** for basic queries.
  
  - Benchmarking attempts have been discussed, with general sentiment indicating that current performance levels are not meeting expectations.
- **Llama 3.1 Dominates Benchmark Tests**: A user asserted that the **Llama 3.1-Nemotron-70B** from Nvidia surpasses competitors like **GPT-4** and **Claude 3.5**, based on alignment benchmarks.
  
  - This model is making a name for itself by attaining impressive scores across numerous assessments.
- **Oura Ring 4 Gains Popularity**: The [Oura Ring 4](https://www.perplexity.ai/page/oura-ring-4-review-5U7Rj9.hR3W0MRa_OmQgbQ) is praised for its advanced health tracking capabilities and sleek design, particularly its sleep monitoring accuracy.
  
  - Users are impressed with its enhanced health insights, contributing to its growing interest in the market.
- **Starlink's Gigabit Speed Plan Sparks Interest**: The [Starlink Gigabit Speed Plan](https://www.perplexity.ai/page/starlink-gigabit-speed-plan-knyorEQ7SYG11t4a.dd2Ig) promises unprecedented internet speeds for rural users.
  
  - Anticipation builds as users look forward to the expected speed improvements for satellite internet connectivity.
- **LFM 40B API Availability Query**: A user inquired about potential API access for the **LFM 40B** model from [labs.perplexity.com](https://labs.perplexity.com), but received no follow-up.
  
  - Additionally, the possibility of an API for the new **spaces feature** was raised, with the clarification that no API exists for the main platform.

 

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **O1-mini beats expectations against Sonnet 3.5**: **O1-mini** displayed a notable ability to outperform **Claude 3.5** in complex tasks through effective reiteration, completing them faster on fewer iterations.
  
  - Despite this, users still favor **Sonnet 3.5** for familiarity and reliability in most scenarios.
- **Sticker Shock: O1-preview Pricing Concerns**: Pricing for **O1-preview** at **$60** for **1m tokens** sparked worries among users, making it less appealing for those already signed up with **ChatGPT Plus**.
  
  - This further fuels interest in alternatives like **Sonnet 3.5**, which remains a favored cost-effective model.
- **Aider Installation Woes Highlight Compatibility Issues**: Users shared troubleshooting tips for **Aider**, with a specific focus on utilizing **pipx** for installation on **Windows 11**.
  
  - Installation woes also emerged for **Chromebooks**, emphasizing the need for broader compatibility across platforms.
- **Token Limits Leave Users Frustrated**: A number of users reported hitting token limits with **claude-3-5-sonnet** and **DeepSeek** models, suggesting the use of `/clear` to alleviate chat history issues.
  
  - Best practices included breaking code into smaller files to help manage usage better.
- **DeepSeek Faces Model Challenges**: Concerns regarding the **DeepSeek** model’s challenges were a recurring topic, leading to discussions around workarounds and shared experiences.
  
  - Members exchanged suggestions for improving their interactions with the model, reflecting a community actively seeking solutions.

 

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Multi-node Clusters raise Ethernet questions**: Users discussed setting up a cluster of **4 V100s** across a network while highlighting Lambda's lack of options for multi-node clusters unless using **Infiniband**.
  
  - *Pure DDP* might negate the need for Infiniband, despite some preferring **Ethernet** for experimental setups.
- **Gen AI Agents hackathon announcement**: An announcement was made for a hackathon hosted by **CreatorsCorner** in collaboration with various tech companies, focusing on creating **AI-powered multi-agent systems**.
  
  - Participants are encouraged to consider ethical implications while building solutions that enhance human potential in daily life.
- **PyTorch 2.5 Hits the Road!**: The release of [PyTorch 2.5](https://anaconda.org/pytorch/pytorch) has been confirmed with wheels now available on conda and PyTorch's pip index.
  
  - *Thought that was supposed to be tomorrow* regarding the excitement around the release.
- **Loss Increases with Variable Removal**: After removing unused variables, the loss increased from approximately **7** to **10** in a training iteration, highlighting unexpected behavior in model performance.
  
  - A file comparison was shared via [Diffchecker](https://www.diffchecker.com/BDcWuLSY/) for further examination.
- **Spooky checks on Cyberpunk 2077 Benchmarking**: A member inquired if it’s feasible to use the system for [benchmarking Cyberpunk 2077](https://link.to.cyberpunk), clarifying it’s for research & performance testing.
  
  - Another member responded that if it’s rewritten as a **triton kernel**, it could work.

 

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio Configuration Gets an Upgrade**: Users confirmed that **ROCm** is included in LM Studio version **0.3.4**, accessible via the Developer tab, improving system configuration.
  
  - One user reported increasing their performance to **32.82 tok/sec** after updating, demonstrating enhancements in practical use.
- **Nvidia Models Shine on Performance Stage**: Members highlighted that the **Nvidia model** significantly outperforms models like **LLM 3.1** on laptops, creating buzz over its efficiency.
  
  - Testing with **Nemotron 70b** models further illuminated competitive advantages, prompting excitement for future benchmarks.
- **Token Generation Rates Impress**: Users reported impressive token generation speeds of **5-7 tok/s** for **70B Q8 models**, rivaling **ChatGPT** performance levels.
  
  - Another configuration hit **32.82 tok/sec**, showcasing the variability and potential across different setups.
- **Llama 3.1 Scores Big on Speed**: A member achieved a remarkable **66 tokens/sec** using **Llama 3.1** on a **7900XTX** GPU at a **10k context length**, showcasing hardware synergy.
  
  - This emphasizes the importance of aligning powerful hardware with large models for optimal results.
- **Cooling Systems Cause Noise Issues**: Discussion highlighted common noise troubles with cooling systems, comparing sounds to a **drone taking off** under load.
  
  - This insight on hardware management underscored challenges in balancing performance with noise levels.

 

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Glif and Wojnak Generators shine**: Members praised the **Glif and Wojnak generators** for producing excellent results with minimal input, dubbing them **gold** in the AI tools landscape.
  
  - They highlighted these tools' capability to generate **workflows that link AI tools** to create functional applications.
- **Voice Features in Desktop App Questioned**: Concerns emerged regarding voice features in [ChatGPT for Windows](https://openai.com/chatgpt/download/), with members unsure if it matches the Android app's capabilities.
  
  - Some worried about the potential unfairness of only macOS users getting voice support initially.
- **O1 Models under Fire**: Members expressed dissatisfaction with the **O1 preview model**, citing slow response times for prompts compared to **O1-mini**, which was deemed significantly faster.
  
  - The consensus pointed to a need for improvements as users seek more efficiency in their interactions.
- **Wispr Flow Gains Attention**: Discussions highlighted the **Wispr Flow application**, which enhances writing speed and accuracy across platforms, currently supporting macOS.
  
  - Members noted that an open-source alternative exists for **Linux, Mac, and Windows** users.
- **CustomGPT Source Citation Flops**: Concerns rose about **CustomGPT** failing to cite sources from documents, sparking questions on effective prompting methods.
  
  - Users agreed that clearer prompts are essential for ensuring source citations are included in the responses.

 

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Inference Providers Seek Clarity**: A member discussed the hunt for inference providers that allow chat assistant completions using prefixes, akin to Anthropic's offerings.
  
  - Concerns about model reliability were raised, indicating a need for clearer communication from providers.
- **NotebookLM Rolls Out Audio Customization**: NotebookLM now enables users to provide custom audio instructions before generating audio, promising a better user experience.
  
  - With over **80,000** organizations onboard, a Business version launched via Google Workspace, shedding its 'Experimental' label.
- **MotherDuck Simplifies SQL-Language Model Interaction**: MotherDuck's introduction of the **prompt()** function integrates small language models into SQL queries for data generation and extraction.
  
  - This innovation looks to streamline LLM interactions while offering notable cost and performance gains.
- **OpenAI Launches Windows Desktop App**: OpenAI has debuted an early version of its ChatGPT Windows desktop app, designed for Plus and Enterprise users, providing faster access.
  
  - Users can access this app conveniently with the **Alt + Space** shortcut, echoing updates in the Claude mobile app for project management.
- **Community Thrives in Data Labeling**: Members highlighted the proactive engagement in **Pixmo** data labeling efforts, sparking creative memes and Reddit discussions.
  
  - They encouraged participation through private Reddit communities for ongoing updates and chatter around data labeling.

 

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Yi-Lightning claims #6 spot**: Big news from the [Chatbot Arena](https://lmarena.ai): **Yi-Lightning** has garnered over **13K community votes** and now ranks **#6 Overall**, showcasing its prowess in areas like Math and Coding.
  
  - This positions it alongside robust competitors like **Grok-2**, fueling anticipation around future performance metrics.
- **GLM-4-Plus surges into top ranks**: **GLM-4-Plus** from Zhipu AI is now in the **top 10** of the chatbot rankings, reflecting the rapid rise of Chinese LLMs in the competitive landscape.
  
  - This indicates a maturing market with increasing competitiveness among various models.
- **Inquiry on Inference Provider Features**: Members queried about inference providers that support chat assistant completions for open-weight models, especially referencing [Anthropic's pre-filling feature](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/prefill-claudes-response).
  
  - *“I'm not sure if I can trust what's going on under the hood”* highlights concerns around the reliability and transparency of these providers.
- **Exploration of Special Tokens**: Discussions emerged about the use of special tokens in chatbot structures, emphasizing the unique formatting associated with user and assistant interactions.
  
  - Members recalled past experiences with these tokens, suggesting referencing documentation for clarity.
- **Valuing Research Experience**: A member shared that transitioning from undergrad research to a non-ML job before pursuing a master's provided them with considerable advantages in **AI labs**.
  
  - They noted that a balance of research experience and workplace familiarity is crucial as labs operate swiftly.

 

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Build a Multimodal RAG System with Azure AI**: A step-by-step guide on creating a **multimodal RAG system** using [Azure AI Search](https://t.co/RO5nQ79sqD), Azure OpenAI, and ArizePhoenix with LlamaIndex has been shared.
  
  - The guide emphasizes contextual retrieval to enhance accuracy and includes benchmarking information for reference.
- **LlamaIndex Meets Elastic - Presentation Tomorrow**: Catch the presentation on how to use **LlamaIndex** with Elastic, featuring insights from a community member, scheduled for tomorrow.
  
  - Details about the presentation can be found [here](https://t.co/tQszqtRN1Z).
- **AI Hackathon in Bengaluru with Meta**: An **AI Hackathon** is happening in Bengaluru on October 19th-20th, in partnership with Reskilll and Meta, boasting mentorship from industry experts.
  
  - Participants can register and find more information [here](https://t.co/aFf31yHJba).
- **Multi-Tenant RAG Applications Simplified**: Community members discussed creating **multi-tenant RAG applications** with LlamaIndex and Nile, targeting data security for numerous users.
  
  - A full-stack demo application illustrating this can be explored [here](https://t.co/zRfzR5A4Us).
- **MongoDB Hybrid Search for LlamaIndex**: Leveraging **MongoDB's** new hybrid search support allows LlamaIndex to combine vector and keyword searches for performance gains.
  
  - Check the details of this integration [here](https://t.co/XxNNwoaW9U).

 

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Join the Modular Community Q&A!**: A reminder about the upcoming **Modular Community Q&A** was announced, urging members to submit their questions via a provided [form](https://forms.gle/MgixGyhRKcA33BS6A). The team encourages participants to get their inquiries in before the meeting.
  
  - *Please share any inquiries you'd like the team to address* during the session.
- **Mojo Aiming for MAX Adaptation**: Members discussed potential plans for a **Mojo** version of **MAX**, noting that the adaptation from **Python** is taking considerable time given **Mojo**'s newness.
  
  - Conversations highlight the complexities and challenges of translating existing functionalities to a new framework.
- **LLMs Revolutionizing Translation Practices**: Community discussions focus on a shift towards using **LLMs** for translation rather than manual processes, emphasizing the efficiency gained in the Chinese community.
  
  - To ensure accuracy, prompts are utilized to clarify translations, particularly regarding terms like 'parameter' to '编译期参数'.
- **Driver Demo Received Favorable Feedback**: The recent driver demonstration showcased the ease of model implementation, although it remains in partial release within **nightly builds**.
  
  - A member expressed their appreciation, mentioning they revisited the demo multiple times to fully grasp the content.

 

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Stable Diffusion Needs Help With Prompts**: A member sought help for a prompt to create a shadow effect for a **cube** without showing the light source above it, emphasizing lighting's crucial role in the scene.
  
  - This sparked a discussion on varying experiences with prompt effectiveness, highlighting the community's need for more **specific suggestions**.
- **Fooocus Models and Compatibility**: Inquiring about model compatibility, a member confirmed that **Fooocus** primarily uses **SDXL**, but it also supports **pony models**.
  
  - This discussion underscored the community's commitment to ensuring compatibility for an improved user experience.
- **Face Swap Feature Solutions**: A member inquired about replicating the **faceswap** feature from **Fooocus** in **Automatic1111**, receiving suggestions like the **Reactor extension** or **IP-Adapter face**.
  
  - This illustrated a collaborative effort among users to enhance tool functionality across various platforms.
- **Concerns About Image Quality**: A member reported generated images lacking detail despite using **30 steps** and multiple **LORA** models, seeking advice on solutions.
  
  - This prompted a broader discussion about the various factors affecting image quality in **Stable Diffusion** processes.
- **AI Hackathon for Innovative Projects**: An announcement for the **Gen AI Agents** hackathon invited teams to develop AI solutions enhancing human potential through collaboration.
  
  - Participants are encouraged to consider ethical implications while creating safe and secure AI systems that optimize daily tasks, with links to [Vertical Specific AI Agents Hackathon](https://lu.ma/ke0rwi8n).

 

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **PyTorch 2.5.0 officially launched!**: The highly anticipated **PyTorch 2.5.0** has been officially released, which includes new features such as [FlexAttention](https://github.com/pytorch/pytorch/releases/tag/v2.5.0) and **per-layer compile**.
  
  - Users are encouraged to upgrade their local **torch** installations to take advantage of the latest features.
- **Tracker for Torchtune contributions launched**: For those looking to contribute to **Torchtune**, a tracker has been set up for cleaning the repository for full **PyTorch 2.5.0** support available [here](https://github.com/pytorch/torchtune/issues/1861).
  
  - This initiative aims to ensure the library aligns with the latest updates and improvements in PyTorch.
- **Qwen 2.5 Model Integration in Torchtune**: [The Qwen team has released Qwen 2.5](https://github.com/pytorch/torchtune/issues/1624) including various models that are being requested for integration into Torchtune, but updates are still pending.
  
  - Members are collaborating to add the model, and there's an openness for others to contribute if they are interested in the integration process.
- **Excitement Around PhD Internship Aspirations**: A user shared an interesting paper on [arXiv](https://arxiv.org/pdf/2410.10630), sparking interest and excitement among members.
  
  - Another member expressed hope for a **PhD internship** to work on projects like those discussed in the paper.
- **Ongoing Work on PPO Progress**: One member indicated that they need to finish up their work on **PPO** before starting new tasks.
  
  - *'I gotta land a few RFCs first and finish up my PPO work'* reflects the current priorities within the team.

 

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **OpenInterpreter task completion woes**: Users report persistent issues with **OpenInterpreter**, stating tasks claim completion without any action being executed.
  
  - Suggestions recommend detailing the version and model in a separate channel to aid in troubleshooting.
- **Kernel panic haunts app closure**: A community member encountered a **kernel panic** upon closing the OpenInterpreter app and was advised to seek help in dedicated support channels.
  
  - This issue underlines the need for reliable exits during application use.
- **Free LLM Options for Cost Efficiency**: A discussion arose regarding free LLMs for integration with Chat GPT due to [rising API costs](https://link.url), prompting suggestions for viable alternatives.
  
  - One suggestion included utilizing the `i model` via `interpreter --model i` for those unable to access local models.
- **AI Meets Vim: New Tutorial Explored**: Mikebirdtech shared insights from Jake Koenig on integrating AI within **Vim**, highlighted in a tutorial video available [here](https://www.youtube.com/watch?v=Ho9yf7ks5sE).
  
  - This adds a new avenue for developers wanting to enhance their coding workflow seamlessly.
- **OpenInterpreter's Utility Through Scripts**: A member introduced the `wtf` script from OpenInterpreter, showcasing its functionality in [Tool Use](https://www.youtube.com/watch?v=Vz3cjbf4zeo).
  
  - The demo emphasized how such scripts can expand user capabilities and engagement with the platform.

 

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Innovative Multi-label Classification Approach**: A member shared an exciting new approach to **multi-label classification** for scientific documents, building on [previous work](https://link.to.research) in in-context learning for extreme multi-label classification.
  
  - They described creating a **Heterogeneous graph** with red nodes as documents and blue nodes as labels, expressing enthusiasm about its potential to search large corpora effectively.
- **Langtrace shines with DSPy integration**: Members discussed the promising integration of **Langtrace** with **DSPy**, highlighting the [setup instructions](https://docs.langtrace.ai/supported-integrations/llm-frameworks/dspy) for capturing traces from DSPy pipelines.
  
  - The setup process includes installing DSPy, initializing Langtrace’s SDK, and creating a project with type **DSPy**.
- **ColbertV2 Training Takes Triples & Queries**: The training example for **ColbertV2** takes in triples, collections, and queries as documented on the [GitHub repository](https://github.com/stanford-futuredata/ColBERT?tab=readme-ov-file#advanced-training-colbertv2-style). This indicates a complex data handling mechanism that requires clarity.
  
  - Members expressed confusion over how the dataset relates to indexed versions of **queries** and **collections** seen in examples.
- **DSPy prompt optimization not reflected in JSON**: A member reported that after optimizing a simple classifier with **MIPROV2**, the JSON config retained the original prompt instead of the optimized one, leading to questions about performance loss.
  
  - Discussion ensued regarding potential bugs in saving or loading configurations, with suggestions to investigate the contents of the JSON file.
- **Positive feedback on DSPy documentation**: A user expressed appreciation for the new DSPy getting started guide, highlighting the approachable breakdown and complete RAG implementation as particularly helpful for newcomers.
  
  - Suggestions included the addition of interactive notebooks and a 'Try It Yourself' section for hands-on learning at the end.

 

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **MSE and MAE Enhancement**: A pull request implementing **MSE** in `tensors.py`, along with tests, has been shared [here](https://github.com/tinygrad/tinygrad/pull/7107). The contributor believes both **MSE** and **MAE** can be summarized concisely in the library.
  
  - *This simplification could streamline tensor operations* and improve clarity for users.
- **Improving LLVM Load with If_Then Gates**: The current **LLVM** loading needs adjustments to use **if_then** for gates, as the existing technique is seen as a hack. Members recognize the urgency in creating a more structured approach to this implementation.
  
  - *A better method could significantly enhance the clarity and functionality* of gate management.
- **Inquiry on Multi-Device CLOUD=1 Functionality**: A member questioned how **CLOUD=1** would operate in a multi-device setup, hoping for consistency with earlier configurations. This reflects an interest in understanding the integration of multi-device operations.
  
  - *Clarifying this will help users optimize their setups* in distributed environments.
- **EMA Parameter Decay Curiosity**: Discussions highlight curiosity about the *decay* process in `update_ema_parameters`, assessing its commonality in deep learning practices. Members are eager to explore optimization techniques more thoroughly.
  
  - *This curiosity illustrates a desire to deepen understanding* of effective training methodologies.
- **Recommended Learning Resources for Tinygrad**: A member proposed starting with the Beautiful MNIST example and modifying an [OpenAI Cookbook example](https://cookbook.openai.com/examples/rag_with_graph_db) for deeper insights into Tinygrad functionalities. Also, [tinygrad-notes](https://mesozoic-egg.github.io/tinygrad-notes) were cited as an excellent resource.
  
  - *These resources offer a practical foundation* for all levels of explicating Tinygrad.

 

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Axolotl shuffles dataset for randomness**: Prior to training, **Axolotl** shuffles the dataset to ensure randomness for each epoch, validating best practices in training protocols. Discussion references [this blog post on Hugging Face](https://huggingface.co/blog/gradient_accumulation) for more details.
  
  - One member confirmed the behavior after searching for references and noted its importance in mitigating overfitting.
- **Gradient accumulation discrepancies raised**: A shared issue indicates that **gradient accumulation** may not match losses between full batch training and toggled settings, causing confusion during training. **Hugging Face** is expected to release a fix soon.
  
  - Members discussed concerns and individual experiences debugging these issues, with one expressing relief for delaying their training start.
- **Bitnet provides official 1-bit LLM framework**: The official inference framework for 1-bit LLMs, **Bitnet**, has been released and can be accessed on [GitHub](https://github.com/microsoft/BitNet). The release highlights a brief overview and includes documentation.
  
  - Members appreciated the availability of the **1-bit LLMs** and discussed potential applications in current projects.
- **A100 compute utilization detailed**: **Invisietch** shared that they utilized **1x A100** for a span of **3 days**, providing specifics on their hardware setup. This insight gives peers a benchmark for compute efficiency.
  
  - The conversation highlighted the practical impacts of specific hardware choices on compute tasks and project timelines.
- **DeepSpeed struggles cause concern**: Invisietch also pointed out issues with **DeepSpeed**, mentioning, *‘Because I couldn’t get DeepSpeed to work,’* indicating setup problems. This fosters discussions on compatibility and implementation hurdles.
  
  - Members expressed curiosity about how to effectively integrate **DeepSpeed** in their workflows, raising questions on common practices.

 

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere tools yield responses with challenges**: A user expressed frustration with the **Cohere** tool's documentation on yielding responses while using **langgraph** and suggested a *for loop* as a fallback if `chat_stream` fails.
  
  - They highlighted the importance of clearer documentation for better user experience and response quality.
- **Command R+ facing performance issues**: A member reported that **Command R+ version 0.8** performed worse than **version 0.4** after a month, prompting discussions on the reasons behind this drop.
  
  - Members wondered if there were any upcoming updates planned to improve its functionality.
- **Curiosity around Inverse RL for LLMs**: Interest spiked as a user linked a paper on **Inverse Reinforcement Learning** for **LLMs**, inviting opinions from the community.
  
  - Discussions revolved around the potential of this approach in enhancing AI capabilities.
- **Call for engagement in multilingual stealth project**: A community member called for builders to join a **stealth** project that requires language expertise, with a link to join the **Aya** server.
  
  - Top contributors will receive **exclusive swag**, highlighting the project's collaborative nature.
- **Langgraph integration documentation updates**: New documentation related to **Cohere's langgraph** integration was mentioned, designed to help users implement tools more efficiently.
  
  - Upcoming examples were hinted to further aid functionality improvement within the **chat_stream** feature.

 

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Quiz Access Woes**: A member faced issues accessing the **Week 5 quiz** located in the syllabus section of the [course website](https://llmagents-learning.org/f24). Another member confirmed its availability and helped navigate to the correct section.
  
  - The follow-up emphasizes that all participants should ensure they are viewing the correct site for quizzes.
- **New Members Join and Seek Guidance**: A newcomer inquired about receiving follow-up emails after filling out a course form and clarification on accessing course materials. Existing participants reassured them to proceed with course participation without stress over hackathons.
  
  - This reflects a supportive atmosphere among participants encouraging less anxiety about supplemental materials.
- **Correct Course Website Identified**: Members confirmed that the **llmagents-learning.org** site is the right one for MOOC students, while the Berkeley site is designated for on-campus students. They advised against using the Berkeley site for course activities to avoid confusion.
  
  - This distinction aims to streamline access for online learners.
- **Article Review Ahead of Posting**: A request was made for an article review prior to posting on social media to meet course expectations. While concerns about the complexity of the review process surfaced, some highlighted the importance of adhering to the guidelines outlined on the course website.
  
  - Community sentiment showed inclination to uphold quality while maintaining ease of process.
- **Weekly Course Progress Reported**: A participant celebrated completing **Week 1** and expressed intent to follow the course structure. This was met with appreciation from the group, fostering motivation to continue progressing.
  
  - The encouraging environment serves to boost engagement across the course participants.

 

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **Seeking Top AI Engineering Blogs**: A member inquired about coveted **AI Engineering blogs** focusing on **Retrieval systems** and **Multi-agent architectures**.
  
  - *No specific blogs were suggested*.
- **Switching to LangGraph Makes Sense**: Discussions highlighted the **pros** of transitioning from **LangChain to LangGraph**, particularly in terms of abstraction and usability.
  
  - A member asked about the **unique features** that **LangGraph** provides compared to **LangChain**.
- **User's LangChain Frustrations**: A user shared their **frustration** over the criticisms of **LangChain** after two years of use, humorously recapping their late-night learning struggles.
  
  - *No further insights were offered on overcoming these issues.*
- **Request for Agent Graph Visualization**: A call for assistance arose on how to **visualize agent graphs** within projects, indicating a need for practical visualization techniques.
  
  - *Unfortunately, no solutions were shared in response.*
- **Exploring LangGraph's Toolset**: A member sparked conversation about the tools accessible in **LangGraph**, looking for deeper insights into its functionalities.
  
  - *No detailed responses were provided regarding its capabilities.*

 

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Inverse RL Advancements Spark Interest**: A paper discussing **inverse reinforcement learning** applications for **LLMs** generated curiosity, prompting discussions for [feedback](https://arxiv.org/pdf/2410.12491).
  
  - Participants aim to assess whether this approach could significantly enhance language model capabilities.
- **NotebookLM Rolls Out Cool Features**: **Google** announced new features for **NotebookLM**, including audio overviews and collaboration tools as seen in [this announcement](http://goo.gle/3UcO8Na).
  
  - These tools are designed to streamline multitasking while accessing audio content for a better user experience, as highlighted in their [tweet](https://x.com/Google/status/1846954813193359397?t=8gWKjTOUhZAYbjFMHluqGw&s=19).
- **Buzz Around Graph Reinforcement Learning**: Excitement grew as a member shared a [survey on Graph Reinforcement Learning](https://arxiv.org/abs/2404.06492), showcasing its decision-making potential across disciplines.
  
  - The connection between **graph structures** and **reinforcement learning** can lead to novel strategies in areas like chemistry and computer science.
- **Gen AI Hackathon Kicks Off**: Participants are invited to a hackathon focused on building **Gen AI-powered multi-agent systems** for daily tasks [details here](https://lu.ma/ke0rwi8n).
  
  - The challenge emphasizes security and ethical considerations while fostering collaborative solutions among developers.

 

---

## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord

- **Fix Twitter/X embeds for enhanced functionalities**: Members emphasized the necessity to **fix broken Twitter/X embeds**, promoting functionalities like multiple images, videos, polls, and translations across platforms like Discord and Telegram. A member linked to the [FixTweet/FxTwitter initiative](https://x.com/i/spaces/1ypKdpLNZXnKW), encouraging contributions to improve embed technologies.
  
  - This initiative aims to streamline integration for richer user engagement and cross-platform content sharing.
- **Interactive tweeting features could boost engagement**: There was a lively discussion centered on how **more interactive tweeting features** could significantly enhance user engagement, particularly regarding embeds.
  
  - Members suggested that **enhanced multimedia support** would likely lead to increased participation and content sharing.

 

---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Gen AI Bug Bounties portal goes live**: The [portal](https://discord.com/channels/1089876418936180786/1245784344539435128/1295876886584492033) for **gen AI bug bounties** has officially launched, streamlining the vulnerability submission process with a user-friendly design and automatic triage for quicker reviews.
  
  - This initiative aims to boost security by simplifying how researchers report vulnerabilities, making it faster for critical issues to be addressed.
- **User Dashboard enhances tracking**: The new **Personalized User Dashboard** offers a centralized view to monitor submission status, updates, and researcher progress.
  
  - This enhancement aims to improve user experience and facilitate better management of vulnerability submissions.
- **Real-Time Notifications keep users updated**: **Real-Time Notifications** will now send instant email alerts for every action taken on submitted vulnerabilities, ensuring transparency.
  
  - Users can remain informed on the status of their submissions without any lag, promoting effective communication.
- **Role-Based Permissions improve security**: The platform introduces **Role-Based Permissions** to ensure structured access control, enhancing data management and collaboration.
  
  - This security measure restricts sensitive information access to authorized users only.
- **Exciting Training Opportunities on the horizon**: Starting in November, **Prompt Engineering Courses & CTF Challenges** will launch, focusing on AI vulnerabilities and skill development.
  
  - The initiative will include **Weekly Blogs & Tutorials**, aiming to enhance participants' AI security knowledge.

 

---

The **LLM Finetuning (Hamel + Dan) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.

---

The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.

---

The **DiscoResearch Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.

---

The **Gorilla LLM (Berkeley Function Calling) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.

---

The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.

---

# PART 2: Detailed by-Channel summaries and links

{% if medium == 'web' %}

 

### **HuggingFace ▷ #**[**general**](https://discord.com/channels/879548962464493619/879548962464493622/1296186543710670888) (890 messages🔥🔥🔥):

> - `Hugging Face Updates`
> - `PyTorch Model`
> - `HuggingChat Community Tools`
> - `Object Detection in Images`
> - `Reinforcement Learning on LLMs`

- **Hugging Face Announces Community Tools**: The Hugging Face community tools feature allows users to create custom tools on HuggingChat, utilizing various modalities like video and speech.
  
  - This capability enables models to leverage new tools, enhancing their interaction with users.
- **Discussion on Using PyTorch and Transformers**: Users discuss the advantages of PyTorch and its integration with Hugging Face models, highlighting features like version control and ease of use.
  
  - Contributors share experiences with their own libraries and the benefits of having models stored on Hugging Face.
- **Object Detection Techniques Explored**: Members exchange insights on using models like YOLO for object detection, emphasizing the significance of bounding boxes.
  
  - Suggestions include using semantic segmentation techniques for per pixel labels with models like SAM.
- **Reinforcement Learning for LLMs**: Users inquire about the feasibility of applying Reinforcement Learning (RL) techniques to optimize LLMs for better response generation.
  
  - Guidance is offered regarding existing resources, such as guides for training LLMs with RLHF.
- **Anime and Manga Interests**: Discussion on anime leads to sharing thoughts about popular series, with mentions of personal experiences and recommendations.
  
  - Not all users resonate with every title, highlighting the diverse tastes within the community.

**Links mentioned**:

- [Emu3 - a Hugging Face Space by BAAI](https://huggingface.co/spaces/BAAI/Emu3): no description found
- [Open port for space to connect to PostgreSQL](https://discuss.huggingface.co/t/open-port-for-space-to-connect-to-postgresql/29938/10): hi @anon86412018 and @deepkyu , we’ve changed the rules and we’ll enable 5432, 27017 in addition to 80, 443. Sorry @anon86412018 I don’t think it’s in prod yet. I’ll ping you here. Thanks
- [SmolLM - blazingly fast and remarkably powerful](https://huggingface.co/blog/smollm): no description found
- [Ollama](https://ollama.com/search?c=embedding): Get up and running with large language models.
- [GPU Benchmarks for Deep Learning | Lambda](https://lambdalabs.com/gpu-benchmarks): Lambda’s GPU benchmarks for deep learning are run on over a dozen different GPU types in multiple configurations. GPU performance is measured running models for computer vision (CV), natural language ...
- [Arm Pump GIF - Arm Pump - Discover & Share GIFs](https://tenor.com/view/arm-pump-gif-22012416): Click to view the GIF
- [Hulk Hogan Flex GIF - Hulk Hogan Flex Flexes - Discover & Share GIFs](https://tenor.com/view/hulk-hogan-flex-flexes-flexing-wwe-gif-13189000): Click to view the GIF
- [Right To Jail Jail GIF - Right To Jail Jail Parks And Rec - Discover & Share GIFs](https://tenor.com/view/right-to-jail-jail-parks-and-rec-right-away-fred-armisen-gif-16902115): Click to view the GIF
- [StackLLaMA: A hands-on guide to train LLaMA with RLHF](https://huggingface.co/blog/stackllama): no description found
- [sail-rvc/Rick_Astley__RVC_v2__140_Epochs at main](https://huggingface.co/sail-rvc/Rick_Astley__RVC_v2__140_Epochs/tree/main): no description found
- [Mundo Feliz Onetreehsll GIF - Mundo feliz Onetreehsll World if - Discover & Share GIFs](https://tenor.com/view/mundo-feliz-onetreehsll-world-if-gif-14071235670792471304): Click to view the GIF
- [Tweet from PyTorch (@PyTorch)](https://x.com/PyTorch/status/1846951947280015407): PyTorch 2.5 is here 🔥 We are excited to announce the release of #PyTorch 2.5, featuring a new CuDNN backend for SDPA, regional compilation of torch.compile, & TorchInductor CPP backend performance sp...
- [Qwen 2.5](https://qwen2.org/qwen2-5/): In this blog, we explore the details of the new Qwen2.5 series language models developed by the Alibaba Cloud Dev Team.
- [Can I Have Mod Discord GIF - Can I Have Mod Discord Discord Mod - Discover & Share GIFs](https://tenor.com/view/can-i-have-mod-discord-discord-mod-gif-23039596): Click to view the GIF
- [CogVLM2: Bringing Deeper Visual and Language Understanding to AI](https://medium.com/@ryanfoster_37838/cogvlm2-bringing-deeper-visual-and-language-understanding-to-ai-2d04d95797a9): AI has come a long way in understanding text, but when it comes to merging visual data — like images and videos — with language, we’ve…
- [rombodawg/Rombos-LLM-V2.6-Nemotron-70b · Hugging Face](https://huggingface.co/rombodawg/Rombos-LLM-V2.6-Nemotron-70b): no description found
- [Open port for space to connect to PostgreSQL](https://discuss.huggingface.co/t/open-port-for-space-to-connect-to-postgresql/29938): Hi @chris-rannou, Could you open the port 5432 for this space: Defi Ai 2022 - a Hugging Face Space by vnghia as I need to connect to a PostgreSQL database ? Thank you very much !
- [Hacking Fake GIF - Hacking Fake Movies - Discover & Share GIFs](https://tenor.com/view/hacking-fake-movies-coding-typing-gif-18697374): Click to view the GIF
- [Tim And Eric Awesome Show GIF - Tim And Eric Awesome Show Kissess - Discover & Share GIFs](https://tenor.com/view/tim-and-eric-awesome-show-kissess-love-kiss-gif-18128184): Click to view the GIF
- [qwen2.5](https://ollama.com/library/qwen2.5): Qwen2.5 models are pretrained on Alibaba's latest large-scale dataset, encompassing up to 18 trillion tokens. The model supports up to 128K tokens and has multilingual support.
- [HP Z2 Tower G9 Workstation](https://www.hp.com/id-id/shop/hp-z2-tower-g9-workstation-a41yppt.html?facetref=22bc09f26b9afe34): Pro-power. Today and tomorrow
- [GitHub - not-lain/pxia: AI library for pxia](https://github.com/not-lain/pxia): AI library for pxia. Contribute to not-lain/pxia development by creating an account on GitHub.
- [GitHub - eloialonso/diamond: DIAMOND (DIffusion As a Model Of eNvironment Dreams) is a reinforcement learning agent trained in a diffusion world model. NeurIPS 2024 Spotlight.](https://github.com/eloialonso/diamond): DIAMOND (DIffusion As a Model Of eNvironment Dreams) is a reinforcement learning agent trained in a diffusion world model. NeurIPS 2024 Spotlight. - eloialonso/diamond
- [huggingchat/chat-ui · [FEATURE] Community Tools](https://huggingface.co/spaces/huggingchat/chat-ui/discussions/569): no description found
- [no title found](https://hubs.ly/Q02QMQ--0): no description found
- [Jiwei Liu | Grandmaster](https://www.kaggle.com/jiweiliu): Please take a look at https://github.com/rapidsai/cuml and star it if you like 😄
- [no title found](https://hubs.li/Q02rCNSs0): no description found
- [not-lain (Lain)](https://huggingface.co/not-lain): no description found
- [GitHub - AntoniovanDijck/diamond-macos: DIAMOND (DIffusion As a Model Of eNvironment Dreams) is a reinforcement learning agent trained in a diffusion world model. NeurIPS 2024 Spotlight.](https://github.com/AntoniovanDijck/diamond-macos.git): DIAMOND (DIffusion As a Model Of eNvironment Dreams) is a reinforcement learning agent trained in a diffusion world model. NeurIPS 2024 Spotlight. - AntoniovanDijck/diamond-macos
- [GitHub - huggingface/transformers: 🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX.](https://github.com/huggingface/transformers.git): 🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX. - huggingface/transformers

---

### **HuggingFace ▷ #**[**cool-finds**](https://discord.com/channels/879548962464493619/897390579145637909/1296195827739398165) (5 messages):

> - `PaliGemma on GitHub`
> - `Grasshopper URLs Extension`
> - `Manim Community Framework`
> - `Perplexity AI for Finance`

- **Explore PaliGemma on GitHub**: Check out the [PaliGemma repository](https://github.com/ThinamXx/PaliGemma) where you can read about its development and contribute to the project.
  
  - A user mentioned that they just starred it, showcasing interest in its functionalities.
- **Manage your Tabs with Grasshopper URLs**: The [Grasshopper URLs](https://addons.mozilla.org/en-US/firefox/addon/grasshopper-urls/) extension offers vertical tabs and serves as a manager for history and bookmarks.
  
  - It requires permissions for tabs, history, bookmarks, and sessions while ensuring no accidental loss of bookmarks.
- **Manim Framework for Animations**: Discover the [Manim Community GitHub](https://github.com/ManimCommunity/manim), a Python framework for creating mathematical animations, maintained by the community.
  
  - This project offers tools for those interested in crafting complex mathematical visualizations.
- **Perplexity AI Enhances Financial Research**: Perplexity AI announced new features for finance, including real-time stock quotes and detailed company financial analyses, as per their [status update](https://x.com/perplexity_ai/status/1846287953599123757?t=RDl45Q5xGvfjF8sIZUm4zw&s=19).
  
  - Users can enjoy a delightful UI while researching market trends and historical earnings.

**Links mentioned**:

- [Tweet from Perplexity (@perplexity_ai)](https://x.com/perplexity_ai/status/1846287953599123757?t=RDl45Q5xGvfjF8sIZUm4zw&s=19): Perplexity for Finance: Real-time stock quotes. Historical earning reports. Industry peer comparisons. Detailed analysis of company financials. All with delightful UI. Have fun researching the marke...
- [Grasshopper – Get this Extension for 🦊 Firefox (en-US)](https://addons.mozilla.org/en-US/firefox/addon/grasshopper-urls/): Download Grasshopper for Firefox. Powerful Tab Manager
- [GitHub - ThinamXx/PaliGemma: Reading PaliGemma paper ...](https://github.com/ThinamXx/PaliGemma): Reading PaliGemma paper ... Contribute to ThinamXx/PaliGemma development by creating an account on GitHub.
- [GitHub - ManimCommunity/manim: A community-maintained Python framework for creating mathematical animations.](https://github.com/ManimCommunity/manim/): A community-maintained Python framework for creating mathematical animations. - GitHub - ManimCommunity/manim: A community-maintained Python framework for creating mathematical animations.

---

### **HuggingFace ▷ #**[**i-made-this**](https://discord.com/channels/879548962464493619/897390720388825149/1296197349265375313) (7 messages):

> - `LLM Training Acceleration`
> - `In-Depth Question Answering Evaluation App`
> - `Book Crossover Storytelling App`
> - `Collaborative Story Builder`
> - `WorldMedQA-V Dataset`

- **Accelerate LLM Training with Custom Platform**: A member introduced a platform designed to **store and stream data for LLM training** pipelines, addressing challenges faced with data management using HuggingFace and S3.
  
  - They are open to **demo requests** and eager to build it further based on community feedback.
- **Real-Time Feedback in Learning with New App**: The **In-Depth Question Answering Evaluation App**, utilizing **Streamlit** and **Gemini 1.5 Pro**, aims to enhance online learning by providing instant feedback to users.
  
  - Inspired by Dr. Fady AlNajjar, this app is positioned as a significant tool for evaluating knowledge progression.
- **Create Unique Story Mashups with Book Mixer**: A member launched **books-mixer-ai**, a tool for blending plots from different books using **ReactJS** and AI technologies, available on Hugging Face Spaces.
  
  - The tool offers users the ability to generate new storylines and accompanying visuals instantly, with documentation forthcoming.
- **Build Stories Collaboratively Online**: A member created a **Collaborative Story Builder** in response to the storytelling trend, promoting community engagement in crafting narratives together.
  
  - They shared their project on Hugging Face Spaces and received positive support from the community.
- **Launching WorldMedQA-V for Healthcare AI**: An announcement was made about **WorldMedQA-V**, a new multilingual, multimodal medical dataset for benchmarking vision-language models in healthcare.
  
  - This dataset serves to advance research in healthcare AI and is accessible on Hugging Face.

**Links mentioned**:

- [Collaborative Story Builder - a Hugging Face Space by Pixeltable](https://huggingface.co/spaces/Pixeltable/Collaborative-Story-Builder): no description found
- [Enhancing Learning Through Real-Time Feedback: In-Depth Question Answering Evaluation App](https://medium.com/@d.isham.ai93/enhancing-learning-through-real-time-feedback-in-depth-question-answering-evaluation-app-4f68c423e496): In the world of online learning and self-improvement, having effective tools to evaluate one’s progress is crucial. Whether you’re studying…
- [Tweet from Shan Chen (@shan23chen)](https://x.com/shan23chen/status/1846923442253152641): 🚀 Exciting News for AI4Health! 🌐 We’re thrilled to release WorldMedQA-V, a multilingual, multimodal medical examination dataset designed to benchmark vision-language models in healthcare! 🩺💻 👉 ...
- [Books Mixer Ai - a Hugging Face Space by as-cle-bert](https://huggingface.co/spaces/as-cle-bert/books-mixer-ai): no description found
- [GitHub - AstraBert/books-mixer-ai: Mix and twist your favorite books!📖](https://github.com/AstraBert/books-mixer-ai): Mix and twist your favorite books!📖. Contribute to AstraBert/books-mixer-ai development by creating an account on GitHub.

---

### **HuggingFace ▷ #**[**reading-group**](https://discord.com/channels/879548962464493619/1156269946427428974/1296488439423828038) (45 messages🔥):

> - `Discussion on LLM Papers`
> - `Joining Meeting Instructions`
> - `Server Purpose and Community`
> - `Zoom Meeting Safety`
> - `Event Recording Availability`

- **Exciting LLM Paper Discussion Ahead**: Members shared links to papers being discussed, such as [the paper on inverse RL for LLMs](https://arxiv.org/pdf/2410.12491), with one author present to answer questions about it.
  
  - The community encouraged participation and curiosity, emphasizing that the meeting is open for all levels of knowledge.
- **How to Join the Meeting**: Instructions were provided on how to join the meeting using the provided Discord link or Zoom invite, along with a passcode and meeting ID.
  
  - Members confirmed that the link directs to a livestream stage, and they reassured others that it's safe to join.
- **Purpose of the Hugging Face Server**: A member inquired about the server's purpose, leading to discussions that clarified it's primarily for Hugging Face-related support and AI topics.
  
  - It's a space where anyone can present research papers, encouraging community collaboration and knowledge sharing.
- **Concerns about Link Safety**: Members expressed caution regarding link safety, with discussions on the trustworthiness of Zoom links associated with McGill University.
  
  - Community members reassured others about the safety of the provided Zoom link, highlighting its reputable association.
- **Recordings and Future Questions**: The organizers announced plans to release the meeting's recording soon and encouraged members to ask remaining questions.
  
  - The community is invited to send additional inquiries to the author, maintaining engagement beyond the live event.

**Links mentioned**:

- [Join our Cloud HD Video Meeting](https://mcgill.zoom.us/j/85109438251?pwd=fxKIhHVTHySWGBRLunWNT7LuQp7pEX.1): Zoom is the leader in modern enterprise video communications, with an easy, reliable cloud platform for video and audio conferencing, chat, and webinars across mobile, desktop, and room systems. Zoom ...
- [INDUS: Effective and Efficient Language Models for Scientific Applications](https://arxiv.org/abs/2405.10725): Large language models (LLMs) trained on general domain corpora showed remarkable results on natural language processing (NLP) tasks. However, previous research demonstrated LLMs trained using domain-f...

---

### **HuggingFace ▷ #**[**computer-vision**](https://discord.com/channels/879548962464493619/922424143113232404/1296503341278302249) (3 messages):

> - `Specific tasks`
> - `Direct messaging`

- **Inquiry about Specific Tasks**: One member asked if there was a specific task or topic in mind for discussion.
  
  - This highlights an openness to clarify and focus the conversation.
- **Invitation to Direct Message**: Another member responded affirmatively and invited the first member to inbox them for more direct communication.
  
  - This indicates a willingness to engage in one-on-one discussions for specifics.

 

---

### **HuggingFace ▷ #**[**NLP**](https://discord.com/channels/879548962464493619/922424173916196955/1296403083441606686) (4 messages):

> - `Dataset Format for Fine-tuning`
> - `NLLB Confidence Display`

- **Fine-tuning Models with Different Dataset Formats**: A member inquired if a base model can be fine-tuned with an instruct formatted dataset and an instruct model with a raw text dataset for domain-specific knowledge.
  
  - Another member confirmed that the first approach is correct but warned that the second would lead to **wrong outputs**.
- **Question on NLLB Text Confidence**: A member asked about displaying the confidence of translated text in NLLB, referencing whisper.cpp’s **\-ojf** parameter that generates a JSON file with confidence for each word.
  
  - They expressed a desire for similar functionality in NLLB to assess translation accuracy.

 

---

### **HuggingFace ▷ #**[**diffusion-discussions**](https://discord.com/channels/879548962464493619/1009713274113245215/1296306634460495882) (30 messages🔥):

> - `Converting model folder to Safetensors`
> - `ControlNet with Fine-tuned Models`
> - `Kwai Kolors Errors in Google Colab`
> - `Renting a VM for Model Training`
> - `Using CLIP for ControlNet Training`

- **How to convert folder model into Safetensors?**: Members discussed the process of converting a folder model into Safetensors, clarifying that Safetensors are just weights and suggesting to reuse weights rather than converting folder structures.
  
  - *Most people use sd1-5, so the safetensors in unet is the one most people want.*
- **Questions on retraining ControlNet**: A member raised a query about retraining ControlNet with a new fine-tuned base model, leading to uncertainties about training implications.
  
  - Discussion highlighted that better support may be found in GitHub discussions related to ControlNet.
- **Errors with Kwai Kolors in Google Colab**: A user reported encountering errors related to versions of numpy and safetensors when running Kwai Kolors in Google Colab, particularly in the free version.
  
  - Members pointed out that additional VRAM is required to run the model efficiently, recommending to switch to a pro plan or run locally.
- **Suggestions for renting VMs for model training**: There was a conversation on recommendations for renting VMs to train models, with mentions of Amazon, FAL, and Replicate as popular options.
  
  - Members emphasized using a private VM for flexibility in software requirements, such as conda.
- **Using CLIP encoders in ControlNet**: A user questioned if they could substitute the CLIP text encoder with an image encoder in their custom ControlNet training to avoid generating extensive text captions.
  
  - Concerns were raised about potential overfitting to a limited number of unique faces in their dataset while training.

**Links mentioned**:

- [yisol/IDM-VTON · Hugging Face](https://huggingface.co/yisol/IDM-VTON): no description found
- [Adapting ControlNet to a New Finetuned Model · huggingface/diffusers · Discussion #9694](https://github.com/huggingface/diffusers/discussions/9694): I used the ControlNet training script from diffusers to obtain a model. The model was trained based on jzli/majicMIX-realistic-7. Then I fine-tuned this base model using the DreamBooth script from ...
- [diffusers/scripts at main · huggingface/diffusers](https://github.com/huggingface/diffusers/tree/main/scripts): 🤗 Diffusers: State-of-the-art diffusion models for image and audio generation in PyTorch and FLAX. - huggingface/diffusers

---

### **Nous Research AI ▷ #**[**general**](https://discord.com/channels/1053877538025386074/1149866623109439599/1296195368316440627) (379 messages🔥🔥):

> - `Gandalf Challenges`
> - `Octopus Theme in LLM Responses`
> - `Control Vectors for LLM Outputs`
> - `Lambda Chat History Stability`
> - `New Model Features`

- **Gandalf Challenges Yield High Success**: Participants shared experiences on the Gandalf challenges, with some achieving high rankings through creative prompt strategies.
  
  - Methods included asking for lists with hidden criteria or playing 21 questions, showcasing the iterative nature of the challenges.
- **Octopus References Stir Responses**: A recurring theme in LLM outputs was an obsession with 'octopus', leading users to speculate on its significance or connection to passwords.
  
  - Conversations circled around evoking poems that engage the models and indirectly reveal desired information, with one user humorously addressing the word manipulation.
- **Control Vectors and Prompt Injections**: Users discussed experimenting with control vectors to bypass model constraints, particularly aimed at obtaining sensitive outputs.
  
  - A variety of strategies were employed, from asking for specific formats of answers to manipulating system prompts to coax more information.
- **Lambda Chat History Issues**: Concerns were raised about the stability of chat history on Lambda, with users noting that their conversations appeared to be disappearing over time.
  
  - This raised queries regarding the platform's functionality and the impact it might have on user experience.
- **Exploration of New Model Features**: Discussion about new model capabilities highlighted curiosity around a 128K context model and its operational status.
  
  - Despite optimism, users expressed disappointment when the newer model didn't function as expected.

**Links mentioned**:

- [Guardrails Arena - a Hugging Face Space by lighthouzai](https://huggingface.co/spaces/lighthouzai/guardrails-arena): no description found
- [Tweet from xjdr (@_xjdr)](https://x.com/_xjdr/status/1846640821107675618): Nemotron-70B entropix edition is pretty fucking good
- [Gandalf | Lakera – Test your prompting skills to make Gandalf reveal secret information.](https://gandalf.lakera.ai/baseline): Trick Gandalf into revealing information and experience the limitations of large language models firsthand.
- [Gandalf | Lakera – Test your prompting skills to make Gandalf reveal secret information.](https://gandalf.lakera.ai/basel): Trick Gandalf into revealing information and experience the limitations of large language models firsthand.
- [Gandalf | Lakera – Test your prompting skills to make Gandalf reveal secret information.](https://gandalf.lakera.ai/adventure-8): Trick Gandalf into revealing information and experience the limitations of large language models firsthand.
- [google/gemma-7b-aps-it · Hugging Face](https://huggingface.co/google/gemma-7b-aps-it): no description found
- [memoize dataset length for eval sample packing by bursteratom · Pull Request #1974 · axolotl-ai-cloud/axolotl](https://github.com/axolotl-ai-cloud/axolotl/pull/1974): Description Fix for issue#1966, where eval_sample_packing=True caused evaluation being stuck on multi-gpu. Motivation and Context In issue#1966, evaluation on sample packed dataset on multiple GPU...

---

### **Nous Research AI ▷ #**[**ask-about-llms**](https://discord.com/channels/1053877538025386074/1154120232051408927/1296185939617513472) (12 messages🔥):

> - `Sampling Parameters`
> - `LLM Programming Languages`
> - `LLM Jailbreak Resources`
> - `God Archetypes for AI Models`
> - `JavaScript vs Python`

- **Checking Sampling Parameters**: A member suggested checking the **sampling parameters** (temp, top-p, top k), but another confirmed that **they are the same**.
  
  - This indicates some uncertainty in their performance despite the unchanged parameters.
- **Choosing Programming Languages for LLMs**: A member queried which programming language top LLMs are best at coding, contemplating **JavaScript** versus **Python**.
  
  - Another member opined that LLMs are **entangled in Python**, while expressing a desire for them to try coding in JavaScript instead of just pursuing one-liners.
- **Resources for LLM Jailbreaks**: In a discussion about resources for **LLM jailbreaks**, a member mentioned possibly checking out **plineys discord**.
  
  - However, another member indicated confusion within that community, expressing a desire for more alternative resources.
- **Humorous God Archetypes for AI Models**: There was a light-hearted question regarding which god AI models would be, with **Opus identified as Prometheus** and **Hermes-3 as Odin**.
  
  - Members found it amusing that **Hermes was part of the discussion**, suggesting some humor around the topic.

 

---

### **Nous Research AI ▷ #**[**research-papers**](https://discord.com/channels/1053877538025386074/1104063238934626386/) (1 messages):

trre: [https://arxiv.org/abs/2410.11163](https://arxiv.org/abs/2410.11163)

---

### **Nous Research AI ▷ #**[**interesting-links**](https://discord.com/channels/1053877538025386074/1132352574750728192/1296201963113021472) (6 messages):

> - `Ollama GGUF Model Usage`
> - `AI Skepticism in Model Training`
> - `SCP Generator Development`

- **Ollama simplifies GGUF Model Execution**: Ollama allows users to directly run any GGUF models available on Hugging Face using the command `ollama run <model_url>`, eliminating the need for a new Modelfile.
  
  - With over **45K public GGUF checkpoints**, this enhances user experience by offering customizable options like quantization type and system prompts.
- **Crypto Pays for Shitposting Insights**: A member humorously suggested that paying shitposters might be an intriguing use case for cryptocurrency, reflecting a growing interest in unconventional applications.
  
  - They also expressed a need for future AI models to incorporate **more skepticism** in their training.
- **SCP Generator Launched on GitHub**: A new [SCP generator](https://github.com/dottxt-ai/cursed/tree/main/scp) created by dottxt-ai utilizes outlines for generating SCP stories, contributing to the SCP community.
  
  - The project is open for contributions, inviting developers to get involved in its ongoing development.

**Links mentioned**:

- [Use Ollama with any GGUF Model on Hugging Face Hub](https://t.co/nxonkJRzW0): no description found
- [cursed/scp at main · dottxt-ai/cursed](https://github.com/dottxt-ai/cursed/tree/main/scp): Contribute to dottxt-ai/cursed development by creating an account on GitHub.

---

### **Nous Research AI ▷ #**[**research-papers**](https://discord.com/channels/1053877538025386074/1104063238934626386/) (1 messages):

trre: [https://arxiv.org/abs/2410.11163](https://arxiv.org/abs/2410.11163)

---

### **Eleuther ▷ #**[**announcements**](https://discord.com/channels/729741769192767510/794042109048651818/1296325016698490880) (1 messages):

> - `Mechanistic Anomaly Detection (MAD)`
> - `Llama 3.1 Performance`
> - `Mistral 7B v0.1 Comparison`
> - `Anomaly Detection Techniques`
> - `Quirky Task Performance`

- **MAD Shows Varied Performance Across Models**: Recent tests on **mechanistic anomaly detection (MAD)** found that **Llama 3.1 8B** performed worse on non-arithmetic tasks compared to **Mistral 7B v0.1** under similar training.
  
  - *Llama exhibited less quirky behavior* but achieved a lower average loss across tasks, indicating a stronger ground truth bias.
- **Anomaly Detection Techniques Yield Similar Results**: Two new approaches to detecting anomalies using **normalising flow** and **sparse autoencoder** activations have shown performance comparable to **Mahalanobis distance** on hidden states of **Llama 3.1 base**.
  
  - However, the inconsistency of previous MAD techniques remains evident, presenting challenges in achieving uniformly effective detection.
- **Performance Insights on Quirky Tasks**: The investigation revealed that the **distance between centroid contexts** in hidden states effectively explained MAD performance, with **Llama** showing less separation than **Mistral**.
  
  - This insight reinforces the variability of the MAD techniques across different tasks, particularly those deemed quirky.
- **Updated Blog Post on Progress**: An update has been posted on the [blog](https://blog.eleuther.ai/mad_research_update_2/) discussing recent findings in mechanistic anomaly detection testing.
  
  - The post provides a comprehensive overview of performance variations and responses to the tested methodologies.

 

---

### **Eleuther ▷ #**[**general**](https://discord.com/channels/729741769192767510/729741769738158194/1296187716211314768) (26 messages🔥):

> - `LLM Re-ranking Techniques`
> - `Anonymity Policies in Workshops`
> - `Evaluating OpenAI's Text Embeddings`
> - `Using Decoder Only Models for Embeddings`
> - `Open Source AI Research Contributions`

- **LLM Re-ranking Techniques Enhance Accuracy**: Re-ranking has been discussed as a pivotal technique to improve search results using advanced [machine learning](https://www.nvidia.com/en-us/glossary/machine-learning/) algorithms, as indicated by a member referencing [this implementation](https://developer.nvidia.com/blog/enhancing-rag-pipelines-with-re-ranking/). The goal of re-ranking is to refine initial search outputs to better align with user intent, resulting in more accurate and relevant information delivered to users.
- **Workshop Anonymity Policies Differ**: Workshop settings at conferences typically have a less strict anonymity policy compared to the overall conference expectations, as indicated in a discussion regarding various locations. This information aims to help members better navigate anonymity across different events.
- **Critique of OpenAI's Text Embeddings**: A member voiced concerns that OpenAI's text embedding models, which were good on release, are considered lacking by 2024 standards, especially when saturated with models like Mistral fine-tunes. These concerns reflect a broader discourse around the evolution and effectiveness of embedding models in current applications.
- **Decoder-Only Models Can Be Effective for Embeddings**: Members discussed the viability of using decoder-only models for extracting embeddings, clarifying that attention masking is the primary difference from encoder models. The potential of using approaches like *llm2vec* was also mentioned, alongside suggestions that simpler models may suffice for many applications.
- **Open Source Opportunities in AI Research**: A member with quantization experience expressed interest in contributing to open source AI research focused on efficient inference or novel architectures. They seek current projects that are open for collaboration, emphasizing their background and readiness to contribute.

**Links mentioned**:

- [NVIDIA Technical Blog | News and tutorials for developers, data scientists, and IT admins](https://developer.nvidia.com/blog): News and tutorials for developers, scientists, and IT admins
- [Enhancing RAG Pipelines with Re-Ranking | NVIDIA Technical Blog](https://developer.nvidia.com/blog/enhancing-rag-pipelines-with-re-ranking/): In the rapidly evolving landscape of AI-driven applications, re-ranking has emerged as a pivotal technique to enhance the precision and relevance of enterprise search results.

---

### **Eleuther ▷ #**[**research**](https://discord.com/channels/729741769192767510/747850033994662000/1296219192210493450) (315 messages🔥🔥):

> - `Muon Optimizer Performance`
> - `Rectified Flow Noise Choices`
> - `Pyramid Noise in Stable Cascade`
> - `Latent Space Considerations`
> - `Initial Training Techniques in State Space Models`

- **Muon Optimizer Performs Well**: Training results show that the Muon optimizer achieves significantly lower validation loss with fewer tokens compared to AdamW, with strong performance scaling on models like GPT-2 up to 1.5B parameters.
  
  - Discussion highlighted the efficiency improvements and lower overhead with the new distributed implementation of Muon.
- **Rectified Flow and Noise Options**: Users discussed the choice of noise distributions in Rectified Flow, noting that while Gaussian noise is commonly used, alternatives like Perlin noise could also be effective for certain applications.
  
  - Participants suggested that the flexibility in noise choices allows for better adaptability to specific target distributions.
- **Pyramid Noise in Stable Cascade**: Stable Cascade uses pyramid noise for stage B, potentially capturing different image scales more effectively than other noise types.
  
  - Pyramid noise is described as a stacking of unit Gaussians, raising queries about its computational efficiency compared to generating actual pink noise.
- **Latent Space Shape Variations**: When discussing noise application in latent spaces, participants noted that even though latent distributions may look different, they can retain necessary spatial correspondence.
  
  - The conversation pondered whether the similarity of distributions is essential for noise methods to be effective in latent spaces.
- **Training Techniques in State Space Models**: A new structured initialization technique proposed for state space models aims to enhance their performance on recall tasks, allowing for better copying abilities from scratch.
  
  - It was suggested that training difficulties rather than capacity constraints may contribute to the previously reported underperformance of these models.

**Links mentioned**:

- [Tweet from leloy! (@leloykun)](https://x.com/leloykun/status/1846842883967692926): The Case for Muon 1) We can descend 'faster' in non-Euclidean spaces 2) Adam/Shampoo/SOAP/etc. dynamically learn the preconditioner and, equivalently, the norm & space to descend in 3) Muon s...
- [Mimetic Initialization Helps State Space Models Learn to Recall](https://arxiv.org/abs/2410.11135): Recent work has shown that state space models such as Mamba are significantly worse than Transformers on recall-based tasks due to the fact that their state size is constant with respect to their inpu...
- [Model Swarms: Collaborative Search to Adapt LLM Experts via Swarm Intelligence](https://arxiv.org/abs/2410.11163): We propose Model Swarms, a collaborative search algorithm to adapt LLMs via swarm intelligence, the collective behavior guiding individual systems. Specifically, Model Swarms starts with a pool of LLM...
- [Thinking LLMs: General Instruction Following with Thought Generation](https://arxiv.org/abs/2410.10630): LLMs are typically trained to answer user questions or follow instructions similarly to how human experts respond. However, in the standard alignment framework they lack the basic ability of explicit ...
- [Untie the Knots: An Efficient Data Augmentation Strategy for Long-Context Pre-Training in Language Models](https://arxiv.org/abs/2409.04774): Large language models (LLM) have prioritized expanding the context window from which models can incorporate more information. However, training models to handle long contexts presents significant chal...
- [Tweet from Yuchen Jin (@Yuchenj_UW)](https://x.com/yuchenj_uw/status/1846964136204173318?s=46): Muon scaling again! I trained the largest GPT-2 (1.5B) using @kellerjordan0's Muon optimizer and achieved a Fineweb validation loss of 2.90 with just 4.2B tokens—only 42% of the tokens required co...
- [madebyollin - Overview](https://github.com/madebyollin/): Made sdxl-vae-fp16-fix, taesd, that pokemon-emulation-via-dnn thing. - madebyollin
- [GitHub - PufferAI/PufferLib: Simplifying reinforcement learning for complex game environments](https://github.com/PufferAI/PufferLib): Simplifying reinforcement learning for complex game environments - PufferAI/PufferLib
- [GitHub - SonicCodes/vmf-vae](https://github.com/SonicCodes/vmf-vae): Contribute to SonicCodes/vmf-vae development by creating an account on GitHub.

---

### **Eleuther ▷ #**[**interpretability-general**](https://discord.com/channels/729741769192767510/1052314805576400977/1296333438596546581) (1 messages):

> - `Model Hallucination Evaluation Methods`
> - `Research Papers on Hallucinations`

- **Popular Methods for Evaluating Model Hallucination**: A member inquired about **popular and reliable methods** to quantify or evaluate a model's **hallucination** in current research, asking for links to relevant papers.
  
  - The discourse indicates an interest in establishing robust metrics to assess the fidelity of outputs from models and identify best practices already in the literature.
- **Seeking Resources on Hallucination Research**: The participant welcomed suggestions and insights into where to find **valuable research papers** addressing hallucination evaluation.
  
  - There is a need for a centralized discussion on how these approaches can be leveraged effectively in ongoing projects.

 

---

### **Eleuther ▷ #**[**lm-thunderdome**](https://discord.com/channels/729741769192767510/755950983669874798/1296524119705124907) (4 messages):

> - `Saving Model Content`
> - `Verbose Warnings in Hugging Face`
> - `Log Samples Parameter`
> - `Issues with Summarizing Tasks`

- **Saving Content During Tests**: A member inquired whether there is a method to save the content generated by a model during testing phases.
  
  - Another member quickly responded that using the `--log_samples` parameter may assist with this.
- **Verbose Warnings with Hugging Face Adapter**: A user reported receiving verbose warnings when passing a pretrained model instance to the Hugging Face adapter, possibly due to expectations mismatch.
  
  - They linked to a specific line in the [lm-evaluation-harness GitHub](https://github.com/EleutherAI/lm-evaluation-harness/blob/624017b7f4501638b0d5848d0f0eab2914a7fb2c/lm_eval/models/huggingface.py#L1362) and described the error related to the model SHA retrieval.
- **Empty Responses in Summarization Tasks**: A member expressed concerns about receiving empty lists for responses (`resps=[], filtered_resps={}`) in tasks related to summarizing or translating.
  
  - They mentioned attempting to troubleshoot the issue further to find a resolution.

 

**Link mentioned**: [lm-evaluation-harness/lm_eval/models/huggingface.py at 624017b7f4501638b0d5848d0f0eab2914a7fb2c · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/624017b7f4501638b0d5848d0f0eab2914a7fb2c/lm_eval/models/huggingface.py#L1362): A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness

 

---

### **OpenRouter (Alex Atallah) ▷ #**[**announcements**](https://discord.com/channels/1091220969173028894/1092729520181739581/1296210656856641666) (2 messages):

> - `NVIDIA Nemotron 70B`
> - `Grok 2 Pricing Update`

- **NVIDIA Nemotron 70B Crushes Competition**: The **NVIDIA Nemotron 70B** has outperformed **Llama 3.1 405B**, **GPT-4o**, and **Claude 3.5 Sonnet** in several evaluations, reporting scores of **85.0** in Arena Hard, **57.6** in AlpacaEval 2 LC, and **8.98** in MT Bench.
  
  - You can check out the results and try it [here](https://openrouter.ai/nvidia/llama-3.1-nemotron-70b-instruct).
- **Grok 2 Repriced as Costs Rise**: **xAI** has increased the pricing for **Grok 2**, now costing **$5/m input** and **$10/m output**, while the Grok 2 Mini remains unavailable.
  
  - Despite the price hike, Grok 2 is trending and can be accessed [here](https://openrouter.ai/x-ai/grok-2).

**Links mentioned**:

- [Tweet from OpenRouter (@OpenRouterAI)](https://x.com/OpenRouterAI/status/1846651197802881094): Big day for open source: NVIDIA Nemotron 70B Nemotron beat Llama 405B, GPT-4o & Claude 3.5 Sonnet on several evals: Nemotron 70B vs Claude 3.5 vs GPT4o: > Arena Hard: 85.0 | 79.2 ...
- [Grok 2 - API, Providers, Stats](https://openrouter.ai/x-ai/grok-2): Grok 2 is xAI's frontier language model with state-of-the-art reasoning capabilities, best for complex and multi-step use cases. To use a faster version, see [Grok 2 Mini](/x-ai/grok-2-mini). Ru...

---

### **OpenRouter (Alex Atallah) ▷ #**[**general**](https://discord.com/channels/1091220969173028894/1094454198688546826/1296186003488505909) (233 messages🔥🔥):

> - `Grok 2 status`
> - `OpenRouter performance`
> - `Voice interaction models`
> - `Deepseek model updates`
> - `O1 model performance`

- **Grok 2 returns with higher prices**: Grok 2 has made a comeback, now priced at **$5/$10**, while the mini version is still unavailable. Users expressed their surprise at the price increase and discussed its implications.
  
  - A link to its current offering was shared, providing more details on its features and pricing [here](https://openrouter.ai/x-ai/grok-2).
- **OpenRouter's models and pricing overview**: Discussion covered various models available via OpenRouter, with notable mentions of **SambaNova** and its scalable nature compared to **Groq**. Users noted the intriguing pricing of **Yi Lightning** at **$0.14/m** input and its advantages over competitors.
  
  - It was suggested that deeper insights into the pricing structures of in-house chip inference providers might be forthcoming as pay-as-you-go models become more widely available.
- **Voice interaction limitations with various LLMs**: Concerns were raised about the performance of voice features in models like **GPT-4o**, particularly its handling of different languages and audio output quality. Users highlighted that while voice input works adequately, the voice output can become 'funky' especially with languages like Chinese.
  
  - The consensus was that **Google's Gemini** managed to release their voice input earlier due to their consistent design standards.
- **Updates on Deepseek Models**: The conversation included updates on **Deepseek** and speculation around potential new versions, like **Deepseek-vl 2**. Users expressed curiosity about the current state of models from Deepseek and their future capabilities.
- **O1 model's usability concerns**: Users discussed **O1's** performance, noting difficulties in following instructions and issues with referencing prior conversation history. Some found it rambled excessively without providing coherent outputs, raising concerns about its practical applications in various tasks.

**Links mentioned**:

- [Gyazo](https://gyazo.com/0b1505d3e5d2939cabaf3fd8857f6e03):
- [Quick Start | OpenRouter](https://openrouter.ai/docs/quick-start): Start building with OpenRouter
- [OpenRouter](https://openrouter.ai/x-ai/g): LLM router and marketplace
- [OAuth PKCE | OpenRouter](https://openrouter.ai/docs/oauth): Secure user authentication via OAuth
- [Grok 2 - API, Providers, Stats](https://openrouter.ai/x-ai/grok-2): Grok 2 is xAI's frontier language model with state-of-the-art reasoning capabilities, best for complex and multi-step use cases. To use a faster version, see [Grok 2 Mini](/x-ai/grok-2-mini). Ru...
- [Llama 3.1 Nemotron 70B Instruct - API, Providers, Stats](https://openrouter.ai/nvidia/llama-3.1-nemotron-70b-instruct): NVIDIA's Llama 3.1 Nemotron 70B is a language model designed for generating precise and useful responses. Run Llama 3.1 Nemotron 70B Instruct with API

---

### **Perplexity AI ▷ #**[**general**](https://discord.com/channels/1047197230748151888/1047649527299055688/1296189511948046418) (105 messages🔥🔥):

> - `Perplexity API Performance`
> - `Nvidia Llama 3 Model Comparison`
> - `File Upload Issues in Spaces`
> - `YouTube Video Analysis with Claude 70B`
> - `Perplexity Subscription Cancellation`

- **Perplexity API Performance Issues**: Users report that the **Perplexity API** is experiencing slow response times, taking between **1 to 2 minutes** for simple queries.
  
  - Attempts to benchmark the response times have been suggested, but many feel current performance isn't satisfactory.
- **Nvidia's Llama 3 Model Outperforms Alternatives**: A user highlighted that the **Llama 3.1-Nemotron-70B** model from Nvidia is being touted as better than **GPT-4** and **Claude 3.5 Sonnet** based on alignment benchmarks.
  
  - This model achieves high scores on various benchmarks, making it a contender in the LLM space.
- **Issues with File Upload in Spaces**: Users are experiencing problems with uploading files to **Spaces**, as confirmed by multiple members discussing similar issues.
  
  - It has been acknowledged that a fix is in progress to resolve the current upload complications.
- **YouTube Video Analysis with Claude 70B**: One user explored using **Claude 70B** to analyze YouTube videos by providing links, but faced limitations with live streams.
  
  - The model didn't require transcripts on one occasion, implying it might auto-generate them when available.
- **Guidance for Perplexity Subscription Cancellation**: A user inquired about canceling their **Pro Monthly Subscription** and seeking a refund due to dissatisfaction with results.
  
  - Others advised on finding the cancellation section and highlighted managing subscriptions through settings.

**Links mentioned**:

- [Skull Issues GIF - Skull issues - Discover & Share GIFs](https://tenor.com/view/skull-issues-gif-13031152103567454559): Click to view the GIF
- [Tweet from Aravind Srinivas (@AravSrinivas)](https://x.com/aravsrinivas/status/1847030982211522852?s=46): The Answer Truck drove more than 1000 miles using FSD from California to Texas. And it's stopping by in Austin tomorrow for a Perplexity user meet-up. La Volta Pizza, Downtown Austin, 1 pm (Austin...
- [Arangutan Monkey GIF - Arangutan Monkey Dancing - Discover & Share GIFs](https://tenor.com/view/arangutan-monkey-dancing-gif-15130385): Click to view the GIF
- [nvidia/Llama-3.1-Nemotron-70B-Instruct-HF · Hugging Face](https://huggingface.co/nvidia/Llama-3.1-Nemotron-70B-Instruct-HF): no description found

---

### **Perplexity AI ▷ #**[**sharing**](https://discord.com/channels/1047197230748151888/1054944216876331118/1296190249818652704) (12 messages🔥):

> - `Oura Ring 4 Review`
> - `Everest Explorer Remains`
> - `Understanding APIs`
> - `Starlink Gigabit Speed Plan`
> - `Tou Zi Aide ETF`

- **Oura Ring 4 Review**: The [Oura Ring 4](https://www.perplexity.ai/page/oura-ring-4-review-5U7Rj9.hR3W0MRa_OmQgbQ) is gaining attention for its advanced health tracking features and sleek design.
  
  - Key highlights include its improved accuracy in sleep monitoring and health insights.
- **Everest Explorer's Remains Found**: Recent findings from [Everest Explorer](https://www.perplexity.ai/page/everest-explorer-s-remains-fou-j3h5Up0rTdyHtGGVmhnC5Q) reveal significant archaeological discoveries.
  
  - These remains shed light on past exploration challenges faced in extreme conditions.
- **Explaining APIs to the Masses**: A detailed overview on [what an API is](https://www.perplexity.ai/search/what-is-an-api-6HaQAJlXRGOWBgQd3L7Iyg#0) covers definitions and functionalities crucial for developers.
  
  - Understanding APIs simplifies interactions between different software applications.
- **Starlink's Gigabit Speed Plan Unveiled**: The [Starlink Gigabit Speed Plan](https://www.perplexity.ai/page/starlink-gigabit-speed-plan-knyorEQ7SYG11t4a.dd2Ig) promises enhanced internet speed for rural areas, garnering significant interest.
  
  - Users are excited about potential speeds reaching unprecedented levels for satellite internet.
- **Tou Zi Aide ETF Discussion**: The [Tou Zi Aide ETF](https://www.perplexity.ai/search/tou-zi-aide-etf-U8cMUG4uQu.geJ8bPz5B0w) has sparked conversation regarding its investment strategy and market impact.
  
  - Investors are evaluating its performance relative to traditional funds amidst fluctuating market conditions.

 

---

### **Perplexity AI ▷ #**[**pplx-api**](https://discord.com/channels/1047197230748151888/1161802929053909012/1296227322390904922) (3 messages):

> - `LFM 40B API availability`
> - `New spaces feature API`

- **Inquiry about LFM 40B API**: A member asked if there is a chance that the **LFM 40B** from [labs.perplexity.com](https://labs.perplexity.com) will be available via the API.
  
  - No response was recorded regarding this query.
- **API for New Spaces Feature Uncertainty**: Another member inquired about the possibility of an **API** for the new **spaces feature**.
  
  - A response clarified that there is **no API for the main platform**, indicating a distinction between the API and perplexity.ai.

 

---

### **aider (Paul Gauthier) ▷ #**[**general**](https://discord.com/channels/1131200896827654144/1131200896827654149/1296213325717442641) (78 messages🔥🔥):

> - `O1-mini vs Sonnet 3.5`
> - `Aider installation on different platforms`
> - `Cost implications of O1-preview`
> - `Architect mode workflows`
> - `Feedback on programming with AI`

- **O1-mini showcases strong reiteration capabilities**: Despite initial user skepticism, **O1-mini** proved to outperform **Claude 3.5** on complex tasks through effective reiteration, completing tasks faster on fewer iterations.
  
  - Several users reported that while O1-mini excels in specific scenarios, they continue to prefer **Sonnet 3.5** for most tasks due to familiarity and reliability.
- **Accessing O1-preview costs raises concerns**: Users expressed concerns regarding the **$60 for 1m tokens** pricing of **O1-preview**, making it less feasible for those already subscribed to **ChatGPT Plus**.
  
  - Alternative models such as **Sonnet 3.5** remain a popular choice among users looking for cost-effective options.
- **Aider installation tips for various platforms**: Users have been sharing their experiences with installing **Aider**, with advice such as using **pipx** for easy installation on **Windows 11**.
  
  - Inquiries regarding installation on **Chromebooks** also surfaced, showing the demand for accessibility across platforms.
- **Exploring Architect Mode alternatives**: One user sought guidance on emulating **Architect mode** without direct access to **O1-preview**, discussing potential workflows and prompts using **ChatGPT** outputs in **Aider**.
  
  - This conversation highlighted the need for alternate methods to achieve similar functionality in light of cost and access limitations.
- **Programming's evolution with AI assistance**: General sentiment indicated that programming is entering a **new age**, significantly transformed by AI tools, which streamline coding tasks.
  
  - Some users noted that reliance on AI has shifted their own coding practices, raising concerns about the impact on traditional coding skills and the skills gap.

**Links mentioned**:

- [VSCode Aider - Visual Studio Marketplace](https://marketplace.visualstudio.com/items?itemName=Apertia.vscode-aider): Extension for Visual Studio Code - Run Aider directly within VSCode for seamless integration and enhanced workflow.
- [Chat modes](https://aider.chat/docs/usage/modes.html#architect-mode-and-the-editor-model): Using the chat, ask and help chat modes.
- [Supported languages](https://aider.chat/docs/languages.html): Aider supports pretty much all popular coding languages.
- [mattf - Overview](https://github.com/MattF): mattf has 98 repositories available. Follow their code on GitHub.
- [The plugin currently doesn't work with Windows · Issue #3 · MattFlower/vscode-aider-extension](https://github.com/MattFlower/vscode-aider-extension/issues/3): Currently, the plugin doesn't work with windows.

---

### **aider (Paul Gauthier) ▷ #**[**questions-and-tips**](https://discord.com/channels/1131200896827654144/1133060505792159755/1296205033632698368) (32 messages🔥):

> - `Token Limits in Models`
> - `Azure API Configuration`
> - `Aider Installation Issues`
> - `Git Issues with Aider`
> - `DeepSeek Model Challenges`

- **Token Limits Confuse Users**: Multiple members reported hitting token limits with various models, specifically mentioning **claude-3-5-sonnet** and **DeepSeek** exceeding their limits.
  
  - Suggestions for managing these issues included using `/clear` to reduce chat history and breaking down code into smaller files.
- **Azure API Configurations**: A user inquired about Azure API keys, and the response clarified that the versioning and configuration depend on individual setups, highlighting links to Azure documentation.
  
  - The setup process, including exporting variables, was also detailed for Mac/Linux and Windows users.
- **Aider Installation Troubleshooting**: A member expressed difficulties installing Aider, particularly with errors related to **NumPy** during installation downloads.
  
  - Suggestions for resolution included ensuring that **pip** and **Python** are properly installed on Chromebook using Linux in Penguin.
- **Potential Bug in Aider Error Handling**: Concerns were raised about Aider committing changes to the wrong file, leading to irreversible damage, highlighting the importance of correct file tracking.
  
  - Members discussed using **git** commands for rollback and filing a bug report to address the potential discrepancy in file edits made by Aider.
- **Editor Suggestions and Customizations**: A user requested changes to how file path completion works in Aider, preferring a behavior similar to Bash where only the current path element gets completed.
  
  - Discussions included potential configuration issues and the order of precedence when using CLI flags versus YAML config files.

**Links mentioned**:

- [Azure](https://aider.chat/docs/llms/azure.html): aider is AI pair programming in your terminal
- [Token limits](https://aider.chat/docs/troubleshooting/token-limits.html): aider is AI pair programming in your terminal

---

### **aider (Paul Gauthier) ▷ #**[**links**](https://discord.com/channels/1131200896827654144/1268910919057149974/) (1 messages):

apcameron: Just released recently [https://mistral.ai/news/ministraux/](https://mistral.ai/news/ministraux/)

---

### **GPU MODE ▷ #**[**general**](https://discord.com/channels/1189498204333543425/1189498205101109300/1296211253945176104) (14 messages🔥):

> - `Multi-node Clusters`
> - `AI Hackathon Announcement`
> - `Inverse Reinforcement Learning for LLMs`
> - `Open Source ML/AI Projects`

- **Multi-node Clusters raise Ethernet questions**: Users discussed setting up a cluster of **4 V100s** across a network while highlighting Lambda's lack of options for multi-node clusters unless using **Infiniband**.
  
  - One member mentioned that **pure DDP** might negate the need for Infiniband, despite some preferring **Ethernet** for experimental setups.
- **Gen AI Agents** hackathon announcement: An announcement was made for a hackathon hosted by **CreatorsCorner** in collaboration with various tech companies, focusing on creating **AI-powered multi-agent systems**.
  
  - Participants are encouraged to consider ethical implications while building solutions that enhance human potential in daily life.
- **Discussion on Inverse Reinforcement Learning for LLMs**: A member shared a link to a research paper on using **inverse RL** for **LLMs** and sought feedback from the community.
  
  - This direction in research aims to explore new methodologies for improving language model mechanisms.
- **Queries on notable open source ML/AI projects**: A member inquired about other notable open source ML/AI projects beyond **Deepspeed** and **ONNX**.
  
  - Community members provided links to curated lists of machine learning frameworks and tools, encouraging exploration of various projects.

**Links mentioned**:

- [Vertical Specific AI Agents Hackathon · Luma](https://lu.ma/ke0rwi8n): Gen AI Agents CreatorsCorner, collaborating with aixplain, Sambanova Systems, Prem, Marly, Senso, Mistral, coval, heygen, fiberplane, exa, and others…
- [GitHub - gpu-mode/awesomeMLSys: An ML Systems Onboarding list](https://github.com/gpu-mode/awesomeMLSys): An ML Systems Onboarding list. Contribute to gpu-mode/awesomeMLSys development by creating an account on GitHub.
- [GitHub - josephmisiti/awesome-machine-learning: A curated list of awesome Machine Learning frameworks, libraries and software.](https://github.com/josephmisiti/awesome-machine-learning): A curated list of awesome Machine Learning frameworks, libraries and software. - josephmisiti/awesome-machine-learning

---

### **GPU MODE ▷ #**[**triton**](https://discord.com/channels/1189498204333543425/1189607595451895918/1296580059183972453) (1 messages):

> - `Channel Closure`
> - `Affiliation with Triton`

- **Odd Closure of Channel**: The same individual who opened the channel also closed it, raising eyebrows about the decision.
  
  - This action was described as 'weird' given claims of it being 'unplanned'.
- **No Triton Association**: It was noted that the individual responsible for the closure is not affiliated with **Triton**.
  
  - This distinction adds to the confusion surrounding the channel's management.

 

---

### **GPU MODE ▷ #**[**torch**](https://discord.com/channels/1189498204333543425/1189607750876008468/1296216375228104854) (9 messages🔥):

> - `PyTorch 2.5 Release`
> - `Torch.compile Overhead`
> - `SGD Fused Updates`

- **PyTorch 2.5 Hits the Road!**: The release of [PyTorch 2.5](https://anaconda.org/pytorch/pytorch) has been confirmed with wheels now available on conda and PyTorch's pip index.
  
  - *Thought that was supposed to be tomorrow* regarding the excitement around the release.
- **Overhead in Torch.compile**: There is noticeable overhead for entering a **torch.compile** region, estimated to be on the order of **hundreds of microseconds**.
  
  - Additionally, this overhead impedes **fusion opportunities** across the graph break.
- **SGD Fused in Documentation**: Discussion highlighted the update in the latest docs confirming that **SGD** now incorporates fused operations.
  
  - A member encouraged others to submit a PR for further documentation updates, indicating collaborative efforts on improvements.

 

**Link mentioned**: [Pytorch | Anaconda.org](https://anaconda.org/pytorch/pytorch): no description found

 

---

### **GPU MODE ▷ #**[**algorithms**](https://discord.com/channels/1189498204333543425/1189861061151690822/1296186178571206658) (13 messages🔥):

> - `untitled01.ipynb`
> - `_xjdr on Twitter`
> - `Flash Attention Techniques`
> - `FlashInfer Project`

- **Confusion Over New Techniques**: Some members expressed confusion over the new techniques being released by `@untitled01.ipynb` and `@_xjdr`, highlighting the lack of clear mathematical or coding explanations.
  
  - One member noted, *'they're unwilling to do a writeup. Instead, they wrote poems.'* indicating frustration over the presentation style.
- **AI Influencer Memes Observed**: A member pointed out that `@untitled01.ipynb` seems to embody the AI influencer meme, using a lot of emojis while discussing AGI topics.
  
  - Another member argued that despite the concern, `@_xjdr` seems to be a legitimate contributor in the field.
- **Concerns Over Clarity in Explanation**: Discussion arose around the need for explainable concepts in AI, with one member stating, *'If you cant explain any thing in a way that a child can understand, you probably dont understand it enough.'*
  
  - The sentiment shared was that those unwilling to explain should keep their findings private to avoid wasting others' time.
- **Flash Attention Insights Shared**: A member shared a [well-written outline on Flash Attention](https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf), emphasizing its clarity.
  
  - They expressed admiration for the author's work on **FlashInfer**, citing some cool techniques in CUDA.

 

---

### **GPU MODE ▷ #**[**beginner**](https://discord.com/channels/1189498204333543425/1191300313928433664/1296381927506317345) (41 messages🔥):

> - `Using Rusticl with v3d driver`
> - `Colab vs Kaggle for GPU access`
> - `CUDA programming in Colab`
> - `Math vs Engineering in GPU work`
> - `Optimizing algorithms for parallel processing`

- **Rusticl and v3d Driver can Run OpenCL**: A suggestion was made to use **Rusticl** with the **v3d driver** for running OpenCL on the **Raspberry Pi 5**.
- **Colab is Useful but Has Limitations**: While **Colab** is recommended for GPU access, it tends to crash during longer workloads (15+ hours), according to user experiences.
  
  - Another user also mentioned that **Kaggle** provides access to K80s and P100s as alternatives.
- **CUDA in Colab for PMPP Projects**: Users confirmed that **CUDA** can be written in **Colab**, and a Turing card is sufficient for **PMPP projects**.
  
  - Though CUDA code in Colab may lack syntax highlighting due to the `%%writefile` command, it successfully saves files on temporary disk.
- **Engineering Focus in GPU Math**: Discussions highlighted that while there is theory related to scaling algorithms on parallel processors, most engineering work revolves around analyzing hardware capabilities.
  
  - Proofs related to **Amdahl's Law** and **Gustafson's Law** were noted as part of the math relevant for scaling in computing.
- **Research on Algorithm Optimization**: There is a distinction between researching the theoretical parallelization of an algorithm and optimizing models mathematically for GPU efficiency.
  
  - Both areas are actively researched, especially in light of future implications for quantum computing.

 

---

### **GPU MODE ▷ #**[**torchao**](https://discord.com/channels/1189498204333543425/1205223658021458100/1296341257596964905) (7 messages):

> - `Windows CI Build Issues`
> - `CUDA Versions and Compatibility`
> - `HIP transformation for ROCm`

- **Windows CI hits a snag with CUDA extensions**: Attempts to build CUDA extensions for Windows using **CUDA 11.8** have presented challenges, as indicated by [this GitHub action](https://github.com/pytorch/ao/actions/runs/11378304134/job/31653869715?pr=1101). The current workaround may involve temporarily skipping CUDA in the build process.
  
  - It was noted that **CUDA 12.1 and 12.4 jobs** succeed because they do not build CUDA extensions, hinting at a limitation in the CI process.
- **Driver compatibility is an issue for newer CUDA builds**: A member mentioned that the current driver is outdated for **CUDA 12.x** builds, impacting compilation success. Moreover, there’s a suggestion to eliminate builds for **Kepler/Maxwell** architectures (anything below **sm_53**) to enhance compatibility.
  
  - In line with this, it's advised that in certain cases, limiting support could further streamline the build process.
- **ROCm wheels show compilation peculiarities**: There's awareness that the **ROCm wheels** do not compile CUDA sources, which is acceptable but worth noting for clarity. Additionally, a strategy could involve converting CUDA code to HIP to better support ROCm.
  
  - The idea of 'hipify' could assist in addressing compatibility across platforms and maximizing build functionality moving forward.

 

**Link mentioned**: [Create build_wheels_windows.yml · pytorch/ao@612e9f7](https://github.com/pytorch/ao/actions/runs/11378304134/job/31653869715?pr=1101): PyTorch native quantization and sparsity for training and inference - Create build_wheels_windows.yml · pytorch/ao@612e9f7

 

---

### **GPU MODE ▷ #**[**triton-puzzles**](https://discord.com/channels/1189498204333543425/1219683012707487794/1296236283676463146) (1 messages):

> - `Triton Puzzles Errors`
> - `GitHub Issues`

- **Error Encountered in Triton Puzzles on Google Colab**: A member reported facing an error while working on **Triton Puzzles** on Google Colab, referencing a specific [GitHub issue](https://github.com/srush/Triton-Puzzles/issues/24).
  
  - They noted that they had not changed any code prior to encountering the issue, indicating a potential common problem.
- **Seeking Help on GitHub Issue**: The member inquired if anyone else had encountered the same problem, seeking support from others in the community.
  
  - This suggests that the issue may not be isolated and could affect multiple users working with Triton Puzzles.

 

**Link mentioned**: [Issues · srush/Triton-Puzzles](https://github.com/srush/Triton-Puzzles/issues/24).): Puzzles for learning Triton. Contribute to srush/Triton-Puzzles development by creating an account on GitHub.

 

---

### **GPU MODE ▷ #**[**llmdotc**](https://discord.com/channels/1189498204333543425/1227345713348870156/1296231524085141584) (8 messages🔥):

> - `Loss Improvements in Training`
> - `Weight Decay and Optimizer Updates`
> - `Diffusion Projects in C/C++`

- **Loss Increases with Variable Removal**: After removing unused variables, the loss increased from approximately **7** to **10** in a training iteration, highlighting unexpected behavior in model performance.
  
  - A file comparison was shared via [Diffchecker](https://www.diffchecker.com/BDcWuLSY/) for further examination.
- **Understanding Weight Decay Dependencies**: Concerns were raised about dependencies in a tensor's index used to apply weight decay, suggesting that deleting tensors requires an optimizer update.
  
  - There was a clarification on the initialization process that was previously misunderstood.
- **Exploring Similar Projects to Llama2.c for Diffusion**: Members discussed potential projects similar to **llama2.c**, specifically for diffusion applications, questioning whether an optimized inference pipeline or training support is needed.
  
  - A relevant suggestion included the GitHub project [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp), which focuses on Stable Diffusion and Flux in pure C/C++.

 

**Link mentioned**: [GitHub - leejet/stable-diffusion.cpp: Stable Diffusion and Flux in pure C/C++](https://github.com/leejet/stable-diffusion.cpp): Stable Diffusion and Flux in pure C/C++. Contribute to leejet/stable-diffusion.cpp development by creating an account on GitHub.

 

---

### **GPU MODE ▷ #**[**rocm**](https://discord.com/channels/1189498204333543425/1233704710389764236/1296218422669213780) (10 messages🔥):

> - `Benchmarking Cyberpunk 2077`
> - `RCCL Improvements`
> - `Flash Attention Test Results`

- **Spooky checks on Cyberpunk 2077 Benchmarking**: A member inquired if it’s feasible to use the system for [benchmarking Cyberpunk 2077](https://link.to.cyberpunk), clarifying it’s for research & performance testing.
  
  - Another member responded that if it’s rewritten as a **triton kernel**, it could work.
- **Cluster's potential with MI250X**: Discussion highlighted that a cluster running **MI250Xes** would be an excellent fit for performance testing.
  
  - This sentiment was shared in a cheerful tone, indicating enthusiasm for the hardware's capabilities.
- **Curious about gaming performance on clusters**: A member expressed curiosity about the possibility of running games on the cluster for performance insights.
  
  - They posed it as a question, reflecting a broader interest in leveraging clusters for non-traditional tasks.
- **Flash Attention Test Benchmarking Discussion**: A member mentioned it takes **180 seconds** to pass the flash attention test, suggesting the addition of a `benchmarks.md` to the repo for baselines.
  
  - This proposal aims to document performance metrics for future reference.
- **Exploring RCCL Improvements**: One member suggested exploring the possibility of sourcing a cluster to enhance **RCCL** contributions.
  
  - They noted that **RCCL** could benefit from significant enhancements, indicating a desire for collaborative improvement.

 

---

### **GPU MODE ▷ #**[**bitnet**](https://discord.com/channels/1189498204333543425/1240586843292958790/) (1 messages):

marksaroufim: [https://github.com/microsoft/bitnet](https://github.com/microsoft/bitnet)

---

### **LM Studio ▷ #**[**general**](https://discord.com/channels/1110598183144399058/1110598183144399061/1296188525397414030) (89 messages🔥🔥):

> - `LM Studio Configuration`
> - `AI Model Performance`
> - `ROCm Implementation`
> - `Token Generation Speed`
> - `Riddles Testing AI Models`

- **LM Studio Configuration Adventures**: Users discussed enabling features in LM Studio, with one confirmation that ROCm is included in version **0.3.4** accessed via the Developer tab.
  
  - Another user updated their version and showed improvement in performance, hitting **32.82 tok/sec**.
- **Nvidia Models Outperforming Competitors**: Members noted that the **Nvidia model** was notably outperforming other models like the **LLM 3.1** on a laptop, sparking excitement about its abilities.
  
  - Testing different models led to comparisons, with success reported utilizing models like **Nemotron 70b**.
- **ROCm and GPU Compatibility**: It was confirmed that **ROCm** only functions with specific AMD cards supported by the **HIPSDK**, such as the **6800XT+**.
  
  - Users expressed interest and sought guidance on using ROCm, particularly with cards like the **7900 XT**.
- **Impressive Token Generation Rates**: An observed token generation rate of **5-7 tok/s** was reported for **70B Q8 models**, reflecting competitive speeds similar to **ChatGPT**.
  
  - Another user reported achieving **32.82 tok/sec** when using specific model configurations, indicating performance variabilities.
- **Riddles as AI Testing Grounds**: Several users discussed the effectiveness of different AI models in solving riddles, noting specific successes and failures.
  
  - The **Nemotron model** gained attention for its capabilities, outperforming models like **Gemini** in this informal testing.

 

---

### **LM Studio ▷ #**[**hardware-discussion**](https://discord.com/channels/1110598183144399058/1153759714082033735/1296207420086947871) (6 messages):

> - `70b models hardware`
> - `Llama 3.1 performance`
> - `Magnum model performance`
> - `HPE DL380 Gen9 setup`
> - `Cooling and noise concerns`

- **Inquiry on 70b models hardware setups**: *Kimmyg* is seeking insights on running **70b models**, specifically inquiring about quantization methods and compatible hardware, particularly motherboards with **2x/4x PCIe x16/x8**.
  
  - Responses suggest a variety of setups and performance metrics, illuminating the need for specific configurations.
- **Llama 3.1 showcases impressive speed**: *A member reported* achieving **66 tokens/sec** with **Llama 3.1** on a **7900XTX** GPU at **10k context length**.
  
  - This highlights the efficiency of specific hardware configurations for larger models.
- **Magnum models struggle on older hardware**: *Another participant* noted only **5 tokens/sec** on **Magnum 72b** using outdated hardware consisting of **4 P40 GPUs**, implying obsolescence in model performance.
  
  - Despite the slowness, it was remarked that the models exhibit intelligence, showcasing a trade-off between performance and sophistication.
- **Exploring the HPE DL380 Gen9 setup**: *Jedd1* has experimented with mid-sized quant models on an **HPE DL380 Gen9** with **two P40 GPUs** (48GB VRAM), noting it provides reasonable performance but can be a bit noisy.
  
  - *Oldtimer8430* advised putting the setup in a different location to alleviate noise while maintaining remote access.
- **Cooling systems lead to noise: a common experience**: *Wildcat_aurora* shared a perspective on cooling issues, stating their hardware sounds like a **drone taking off** when under heavy load, but only lasts briefly.
  
  - This exchange highlights the common challenge of managing noise levels in high-performance setups.

 

---

### **OpenAI ▷ #**[**ai-discussions**](https://discord.com/channels/974519864045756446/998381918976479273/1296193231129673728) (52 messages🔥):

> - `Glif and Wojnak Generators`
> - `AI interruptions in voice mode`
> - `O1 Models`
> - `Wispr Flow Application`

- **Glif and Wojnak Generators make gold**: Members praised the **Glif and Wojnak generators**, claiming these tools produce outstanding results with minimal input, calling them **gold**.
  
  - Discussion often compared these tools to others, hinting that they can generate **workflows that link AI tools** to create 'apps'.
- **Advanced Voice Mode struggles**: A member criticized the **advanced voice mode**, stating it fails to understand interruptions and delivers vague responses.
  
  - They expressed frustration with the AI frequently stating, *'my guidelines prevent me from talking about that,'* and not acknowledging their inquiries.
- **O1 Models face scrutiny**: Dialogue centered around perceptions that the **O1 preview model** is underperforming, with members noting it takes too long to respond to prompts.
  
  - In contrast, **O1-mini** was highlighted as the 'real hero' for its quicker response time and effectiveness.
- **Wispr Flow gains attention**: Members discussed **Wispr Flow**, an application that helps users write faster and more accurately across all computer platforms.
  
  - It was emphasized that while it currently supports macOS, an open-source app mentioned caters to **Linux, Mac, and Windows** users.

 

**Link mentioned**: [Wispr Flow | Effortless Voice Dictation](https://flowvoice.ai/d): Flow makes writing quick and clear with seamless voice dictation. It is the fastest, smartest way to type with your voice.

 

---

### **OpenAI ▷ #**[**gpt-4-discussions**](https://discord.com/channels/974519864045756446/1001151820170801244/1296458899825885225) (23 messages🔥):

> - `ChatGPT for Windows`
> - `Voice Features on Desktop`
> - `Privacy Concerns`
> - `Fine-Tuning a Chess Bot`

- **Excitement about ChatGPT for Windows Release**: Members expressed excitement over the announcement of [ChatGPT for Windows](https://openai.com/chatgpt/download/), with questions regarding access for users.
  
  - One clarified that the Windows app is currently available only for Plus, Team, Enterprise, and Edu users.
- **Voice Features in the New Desktop App**: A member inquired if the new Windows version supports voice features like the Android app, but there was uncertainty as no one had downloaded it yet.
  
  - Concerns were raised about the lack of fairness if only macOS users got the voice feature initially.
- **Debate on Privacy and PII in Screen Sharing**: A discussion arose about sharing sensitive information when using the app's screen interaction capabilities, particularly regarding **Personally Identifying Information (PII)**.
  
  - Members highlighted concerns over how to limit what the model can see and the implications of sharing personal data.
- **Skepticism about App Privacy**: One member expressed hesitation in downloading the app due to concerns about information control and what happens to data shared with the AI.
  
  - This sparked a discussion around how the app interacts with user screens and the potential for unwanted sharing.
- **Comparisons to Google’s Data Practices**: Amid privacy discussions, a member noted that **Google** already collects significant user data, including voice capture, drawing a humorous comparison.
  
  - This comment reflected broader concerns about data privacy in the context of new AI technologies.

 

---

### **OpenAI ▷ #**[**prompt-engineering**](https://discord.com/channels/974519864045756446/1046317269069864970/1296311724785274923) (3 messages):

> - `CustomGPT source citation`
> - `Prompting techniques for CustomGPT`

- **CustomGPT struggles with source citations**: A member inquired why **CustomGPT** never cites its sources from documents despite multiple attempts.
  
  - *Is there a specific prompt to make it work?* was the underlying question resonating among users.
- **Seeking effective prompting strategies**: There was a discussion regarding how to properly prompt **CustomGPT** to ensure source citations are included.
  
  - Members shared their frustrations, indicating that clarity in prompts is essential for desired responses.

 

---

### **OpenAI ▷ #**[**api-discussions**](https://discord.com/channels/974519864045756446/1046317269069864970/1296311724785274923) (3 messages):

> - `citing sources`
> - `customGPT functionality`

- **CustomGPT struggles to cite sources**: A member raised a concern regarding **customGPT** not citing sources from documents, questioning why this feature is lacking.
  
  - They inquired if anyone knew how to prompt it correctly to ensure proper citations.
- **Exploration of prompting techniques for customGPT**: Another member suggested experimenting with different prompting techniques to encourage **customGPT** to include citations.
  
  - They recommended trying clear and direct requests for citations alongside specific document references.

 

---

### **Latent Space ▷ #**[**ai-general-chat**](https://discord.com/channels/822583790773862470/1075282825051385876/1296226289438429267) (70 messages🔥🔥):

> - `Inference Providers for Chat Assistants`
> - `NotebookLM Updates`
> - `MotherDuck SQL Integration with LLMs`
> - `OpenAI's Windows Desktop App Release`
> - `Community Engagement in Data Labeling`

- **Seeking Inference Providers for Chat Completions**: A member is looking for inference providers that allow executing chat assistant completions using prefixes to shape the assistant's responses, similar to Anthropic's feature.
  
  - They expressed concerns about the reliability of the functionality across different models, hinting at the need for more clarity from the providers.
- **NotebookLM Announces Custom Audio Instructions**: NotebookLM users can now customize audio overviews by providing specific instructions before generating audio, enhancing user experience.
  
  - With over 80,000 organizations using NotebookLM, they also announced a new Business version available via Google Workspace, removing the product's 'Experimental' label.
- **SQL LLM Integration from MotherDuck**: MotherDuck has introduced a new SQL function, **prompt()**, allowing users to integrate small language models directly into queries for data generation, summarization, and extraction.
  
  - This function aims to simplify LLM interaction without separate infrastructure, showcasing significant cost and performance improvements.
- **OpenAI Releases Windows Desktop App**: OpenAI has released an early version of their ChatGPT Windows desktop app for Plus, Enterprise, Team, and Edu users, enabling faster access with Alt + Space.
  
  - In a related announcement, Claude's mobile app was updated significantly, including new features for project management and custom instructions.
- **Community Engagement in Pixmo Data Labeling**: A member highlighted the enthusiastic engagement of the community involved in labeling data for Pixmo, leading to the creation of memes and discussions on Reddit.
  
  - They directed attention to private Reddit communities where members can keep up with study info and participate in discussions related to data labeling.

**Links mentioned**:

- [Something went wrong](https://events.zoom.us/ej/AqLi3dmNZSAXddMiqJkHlHTWkEjpoQZ7CEHtgg-bgBXf): no description found
- [Introducing the prompt() Function: Use the Power of LLMs with SQL!](https://motherduck.com/blog/sql-llm-prompt-function-gpt-models/): We make your database smarter with small language model (and LLM) support in SQL | Reading time: 6 min read
- [Tweet from Logan Kilpatrick (@OfficialLoganK)](https://x.com/OfficialLoganK/status/1846951237578903753): Great news for NotebookLM fans: Audio Overviews can now be customized and steered before generating + we are rolling out NotebookLM for business! 🎙️ https://notebooklm.google/
- [Homebrew Research – Homebrew](https://homebrew.ltd/): Homebrew is an AI R&D studio that works in the area of Local AI, Small Language Models and Multi-modality.
- [After selling Drift, ex-HubSpot exec launches AI for customer success managers | TechCrunch](https://techcrunch.com/2024/10/16/after-selling-drift-ex-hubspot-exec-launches-ai-for-customer-success-managers/): Elias Torres has achieved a lot for somebody who immigrated to the U.S. from Nicaragua at 17 without knowing any English. He served as a VP of engineering
- [Open LLM Leaderboard Model Comparator - a Hugging Face Space by open-llm-leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/comparator): no description found
- [Tweet from Raiza Martin (@raiza_abubakar)](https://x.com/raiza_abubakar/status/1846944566689353838?s=46&t=jDrfS5vZD4MFwckU5E8f5Q): New NotebookLM updates, rolling out today: 🎧Pass a note to the hosts – you can now click on ‘Customize’ in Audio Overviews to give additional instructions, such as focusing on a specific topic, sour...
- [Tweet from Alex Volkov (Thursd/AI) (@altryne)](https://x.com/altryne/status/1846977617704140893?s=46): Breaking - you can now send and receive audio from the chat completions API for @OpenAI 👏 Unlike RealTime audio, this is well suited for multimodal applications that do not require real time, but a...
- [Tweet from Daniel Han (@danielhanchen)](https://x.com/danielhanchen/status/1846235913443262891): Fixed a bug which caused all training losses to diverge for large gradient accumulation sizes. 1. First reported by @bnjmn_marie, GA is supposed to be mathematically equivalent to full batch training...
- [Tweet from Elias Torres (@eliast)](https://x.com/eliast/status/1846652872060002732?s=46): Today’s the day. Introducing my new company, @Agency — backed by @Sequoia and Hubspot Ventures; with my friend and mentor @BHalligan on the Board. Enjoyed getting to talk with @MTemkin @TechCrunch abo...
- [Tweet from Perplexity (@perplexity_ai)](https://x.com/perplexity_ai/status/1846950770736091509?s=46): Introducing Internal Knowledge Search (our most-requested Enterprise feature)! For the first time, you can search through both your organization's files and the web simultaneously, with one produ...
- [Tweet from Jacob Matson (@matsonj)](https://x.com/matsonj/status/1847007726335152284?s=46): Are you kidding me? Look at this: Quoting MotherDuck (@motherduck) We put a LLM in SQL and also show you the power of SLMs (small language models) in the MotherDuck data warehouse. https://mothe...
- [Comparison of AI Models across Quality, Performance, Price | Artificial Analysis](https://artificialanalysis.ai/models): Comparison and analysis of AI models across key performance metrics including quality, price, output speed, latency, context window & others.
- [Tweet from Simon Willison (@simonw)](https://x.com/simonw/status/1846987810706018435?s=46&t=6FDPaNxZcbSsELal6Sv7Ug): I just tried this OpenAI example - I got back a 264KB base64 encoded WAV file (as a JSON string) which was 5 seconds long and cost 110 audio output tokens - those are priced at $200/million so the pro...
- [Tweet from OpenAI (@OpenAI)](https://x.com/openai/status/1846957067204166113?s=46&t=6FDPaNxZcbSsELal6Sv7Ug): Today, ChatGPT Plus, Enterprise, Team, and Edu users can start testing an early version of the Windows desktop app. Get faster access to ChatGPT on your PC with the Alt + Space shortcut. https://ope...
- [Tweet from clem 🤗 (@ClementDelangue)](https://x.com/ClementDelangue/status/1847009885852258650): 👀👀👀
- [Tweet from Clémentine Fourrier 🍊 (@clefourrier)](https://x.com/clefourrier/status/1846907589365297640): Have you always wanted to compare the best leaderboard models performance in detail? Check out our new tool! 🔍 https://huggingface.co/spaces/open-llm-leaderboard/comparator It compares, side by sid...
- [Tweet from Nathan Cooper (@ncooper57)](https://x.com/ncooper57/status/1846612127911760261?s=46): As a lead researcher at @stabilityai, I worked a lot with synthetic data to train LLMs and VLMs. It is the most underrated way of boosting model performance. Now at @answerdotai I've been working ...
- [Reddit - Dive into anything](https://www.reddit.com/r/MattDeitkeStudies/): no description found
- [Requests | OpenRouter](https://openrouter.ai/docs/requests): Handle incoming and outgoing requests
- [Sign In | Zoom](https://events.zoom.us/ej/AqLi3dmNZSAXddMiqJkHlHTWkEjpoQZ7CEHtgg-bgBXf5FUjyxMS~A9Dc0qDMYy1XxnQw-wyMyInvma-5aGQLC-k7gh3UVsnS8AZ3om-GLGN6Xou-kOQgIU_--FVQcTlqmx0hsKmS-anoiyH0d5XMk4BvO-JFI): Sign in to your Zoom account to join a meeting, update your profile, change your settings, and more!
- [Tweet from OpenAI (@OpenAI)](https://x.com/openai/status/1846957067204166113?): Today, ChatGPT Plus, Enterprise, Team, and Edu users can start testing an early version of the Windows desktop app. Get faster access to ChatGPT on your PC with the Alt + Space shortcut. https://ope...
- [Tweet from Alex Albert (@alexalbert__)](https://x.com/alexalbert__/status/1846943479332802571?s=46&t=6FDPaNxZcbSsELal6Sv7Ug): We just rolled out a major design overhaul of the Claude mobile app. It feels super smooth to use now. You can create projects, add custom instructions, and chat within your projects all within the a...

---

### **Interconnects (Nathan Lambert) ▷ #**[**news**](https://discord.com/channels/1179127597926469703/1179128538679488533/1296193827346059274) (4 messages):

> - `Yi-Lightning`
> - `Chatbot Arena Rankings`
> - `GLM-4-Plus Surge`
> - `Chinese LLMs Competition`

- **Yi-Lightning climbs the rankings**: Big news from the [Chatbot Arena](https://lmarena.ai) as **Yi-Lightning** by @01AI_YI has garnered over **13K community votes** and now ranks **#6 Overall**.
  
  - It has matched strong models like **Grok-2**, excelling in Math, Hard Prompts, and Coding.
- **GLM-4-Plus makes its mark**: **GLM-4-Plus** from Zhipu AI has also made it into the **top 10**, showcasing the rapid rise of Chinese LLMs.
  
  - This development signals increasing competitiveness among these models.

 

**Link mentioned**: [Tweet from lmarena.ai (formerly lmsys.org) (@lmarena_ai)](https://x.com/lmarena_ai/status/1846245604890116457): Big News from Chatbot Arena! @01AI_YI's latest model Yi-Lightning has been extensively tested in Arena, collecting over 13K community votes! Yi-Lightning has climbed to #6 in the Overall ranking...

 

---

### **Interconnects (Nathan Lambert) ▷ #**[**ml-questions**](https://discord.com/channels/1179127597926469703/1179208129083363358/1296560727812735006) (8 messages🔥):

> - `Inference Providers for Chat Models`
> - `Special Tokens in Chat Models`
> - `Pre-filling Chatbot Responses`
> - `Support Experience with Model Providers`
> - `Interconnects Discord vs. Latent Space Discord`

- **Inquiry on Inference Providers' Capabilities**: A member inquired about inference providers that support chat assistant completions for popular open-weight models, specifically looking for capabilities similar to [Anthropic's pre-filling feature](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/prefill-claudes-response).
  
  - *“I'm not sure if I can trust what's going on under the hood”* reflects concerns about the reliability of such providers.
- **Discussion on Special Tokens Usage**: A member highlighted their interest in accessing specific special tokens for structuring chatbot interactions, noting the unique format without an END_OF_TURN_TOKEN for assistant responses.
  
  - They provided an example structure that showcases how user and assistant turns are formatted with various tokens.
- **Past Experiences with Non-Chat Models**: One member recalled their experience dealing with those tokens last year for non-chat models, indicating that it was optional back then.
  
  - *“Maybe try their docs”* suggests looking up documentation to clarify the token usage and implementation.
- **Praise for Provider Support**: A member shared their positive experience, stating that the support from the mentioned provider was fast and helpful.
  
  - This indicates a favorable impression of the responsiveness and quality of assistance from support teams.
- **Comparative Insight on Discord Spaces**: A member commented on the difference between the Interconnects discord and the Latent Space discord, implying the former is more private.
  
  - *“Interconnects discord is like the more private version of latent space discord lol”* reflects a light-hearted observation about their community dynamics.

 

---

### **Interconnects (Nathan Lambert) ▷ #**[**random**](https://discord.com/channels/1179127597926469703/1183121795247779910/1296246150617104456) (51 messages🔥):

> - `Research Experience Value`
> - `Degree Requirements for AI Labs`
> - `Luck and Risk in Careers`
> - `Community Engagement in AI Projects`
> - `Self-Study Challenges`

- **Research Experience Matters Now**: A member shared that transitioning from undergrad research to a non-ML job before pursuing a master's helped them in the long run, proving **more research experience is beneficial**.
  
  - They emphasized that these labs move quickly, requiring familiarity with workplace dynamics, not just intelligence.
- **Debate on Degrees in AI Labs**: There's ongoing discussion about whether a master's degree is necessary for positions at **top AI labs** like OAI, DM, and Anthropic.
  
  - Members noted that while credentials help, proven skills and relevant experience can often outweigh formal education.
- **Creating Your Own Luck**: A topic emerged around the idea of **'making your own luck,'** suggesting that while randomness exists, opportunities arise from strategic risk-taking.
  
  - Members agreed that expanding one’s opportunities is essential to maximize chances for positive outcomes.
- **Community Labels for Pixmo**: A fun fact about the **Pixmo** community revealed they were so engaged that they created a dedicated Reddit community for labeling that features memes and discussions.
  
  - Links to these communities indicate an active engagement, proving that audience participation can foster vibrant discussions around projects.
- **Challenges of Self-Study**: Concerns arose regarding the effectiveness of self-study without proper guidance, with one member stating that it often traps many learners.
  
  - The consensus is that developing skills through structured learning, especially with mentorship, is more effective in the long run.

 

**Link mentioned**: [Reddit - Dive into anything](https://www.reddit.com/r/MattDeitkeStudies/): no description found

 

---

### **Interconnects (Nathan Lambert) ▷ #**[**posts**](https://discord.com/channels/1179127597926469703/1228051082631188530/1296185904167387238) (5 messages):

> - `SnailBot Speed`
> - `User Dynamics`

- **SnailBot accelerates to eight minutes**: A member noted, *'wow... eight minutes... the snail is getting faster'* suggesting an improvement in the performance of **SnailBot**.
  
  - This comment hints at the ongoing anticipation of how the bot's speed impacts user interactions.
- **Recurring User Interaction Patterns**: A member referred to a repetitive interaction, stating, *'and I do this dance'* implying familiar dialogues between users.
  
  - This highlights the playful dynamics within the community, reflecting ongoing engagement.
- **Fickle Nature of Conversations**: One user expressed frustration with the capriciousness of discussions, stating, *'It’s so fickle'* to emphasize the unpredictable nature of chat topics.
  
  - This captures sentiments about the fluctuating engagement levels seen in digital conversations.
- **SnailBot News Announcement**: A notification was issued by SnailBot to the role <@&1216534966205284433>, signaling it has news updates to share.
  
  - This indicates an ongoing role of the bot in disseminating information within the community.

 

---

### **LlamaIndex ▷ #**[**blog**](https://discord.com/channels/1059199217496772688/1187460979064324127/1296227795055153204) (5 messages):

> - `Multimodal RAG system`
> - `LlamaIndex with Elastic`
> - `AI Hackathon`
> - `Multi-tenant RAG applications`
> - `MongoDB hybrid search support`

- **Build a Multimodal RAG System with Azure AI**: A step-by-step guide on creating a **multimodal RAG system** using [@Azure AI Search](https://t.co/RO5nQ79sqD), Azure OpenAI, and @ArizePhoenix with LlamaIndex has been shared.
  
  - The guide emphasizes contextual retrieval to enhance accuracy and provides benchmarking information.
- **LlamaIndex and Elastic Presentation Tomorrow**: Catch @seldo discussing how to use **LlamaIndex** with Elastic in an upcoming presentation, which is sure to offer valuable insights.
  
  - More details are available [here](https://t.co/tQszqtRN1Z).
- **AI Hackathon in Bengaluru with Meta**: Partnering with @Reskilll and @Meta, an **AI Hackathon** is scheduled for October 19th-20th in Bengaluru featuring mentorship from @ravithejads.
  
  - Participants can find out more about the event [here](https://t.co/aFf31yHJba).
- **Multi-Tenant RAG Applications Made Simple**: Easily build **multi-tenant RAG applications** with LlamaIndex and Nile, addressing data security concerns when indexing for numerous users.
  
  - Check out the full-stack demo application [here](https://t.co/zRfzR5A4Us).
- **MongoDB Hybrid Search for LlamaIndex**: Enhance your AI applications using **MongoDB's** new hybrid search support for LlamaIndex that combines vector and keyword search.
  
  - This approach aims to merge the strengths of both search types for optimal results, detailed here: [link](https://t.co/XxNNwoaW9U).

 

**Link mentioned**: [AI Hackathon with Meta Llama](https://t.co/aFf31yHJba): Join us for an exhilarating 30-hour experience with industry experts who are passionate about AI. This is your chance to meet, collaborate, and have fun while building something amazing. Let's create ...

 

---

### **LlamaIndex ▷ #**[**general**](https://discord.com/channels/1059199217496772688/1059201661417037995/1296361853307322405) (46 messages🔥):

> - `MultiStepQueryEngine support in LlamaIndex.TS`
> - `Metadata use in RAG for document management`
> - `vLLM server issues`
> - `Faithfulness evaluation time optimization`
> - `LlamaParse for Word documents`

- **Questions on MultiStepQueryEngine in LlamaIndex.TS**: Members discussed the absence of support for **MultiStepQueryEngine** in LlamaIndex.TS, suggesting a workaround by using an LLM to manually break down tasks.
  
  - Another member offered a method to add metadata such as **file names** to the embedding process in a retrieval system.
- **Issues with vLLM Server Returning 400 Bad Request**: A member reported receiving a **400 Bad Request** error while calling the **vLLM server**, indicating missing required parameters in the request payload.
  
  - Through troubleshooting, they identified and removed **None values** from the payload before rerunning the request.
- **Performance Concerns with Faithfulness Evaluation**: One member expressed frustration over long processing times while replicating the **Faithfulness evaluation**, sometimes taking over an hour.
  
  - Discussion revolved around hardware limitations, suggesting switching from **LlamaCPP** to **Ollama** for potentially faster performance with local models.
- **LlamaParse Error with Word Documents**: A member encountered an unexpected result while using **LlamaParse** on a Word document, showing an image instead of the expected text data.
  
  - They provided a link to a minimal repository demonstrating the issue, seeking feedback from the community.

**Links mentioned**:

- [no title found](https://<YOUR_HOST>/v1/chat/completions): no description found
- [Qdrant Vector Store - Metadata Filter - LlamaIndex](https://docs.llamaindex.ai/en/stable/examples/vector_stores/Qdrant_metadata_filter/#qdrant-vector-store-metadata-filter): no description found
- [GitHub - xaac-ai/llama-artifact](https://github.com/xaac-ai/llama-artifact): Contribute to xaac-ai/llama-artifact development by creating an account on GitHub.
- [llama_index/llama-index-integrations/llms/llama-index-llms-vllm/llama_index/llms/vllm/utils.py at f633e7393aaa3f36ef518429672b931b1e3bdae8 · run-llama/llama_index](https://github.com/run-llama/llama_index/blob/f633e7393aaa3f36ef518429672b931b1e3bdae8/llama-index-integrations/llms/llama-index-llms-vllm/llama_index/llms/vllm/utils.py#L8C5-L9C24): LlamaIndex is a data framework for your LLM applications - run-llama/llama_index

---

### **Modular (Mojo 🔥) ▷ #**[**general**](https://discord.com/channels/1087530497313357884/1098713601386233997/1296571671716827206) (1 messages):

> - `Modular Community Q&A`

- **Join the Modular Community Q&A!**: A reminder was sent out about the upcoming **Modular Community Q&A** during Monday's community meeting, encouraging members to submit their questions via a provided [form](https://forms.gle/MgixGyhRKcA33BS6A).
  
  - *Please share any inquiries you'd like the team to address* during the session.
- **Submit Your Questions Now!**: Participants are encouraged to **submit their questions** through the designated form, allowing the team to prepare responses ahead of the meeting.
  
  - *Don’t miss the chance to have your queries answered—fill out the form before Monday!*

 

**Link mentioned**: [Modular Community Q&A](https://forms.gle/MgixGyhRKcA33BS6A): no description found

 

---

### **Modular (Mojo 🔥) ▷ #**[**mojo**](https://discord.com/channels/1087530497313357884/1151418092052815884/1296222188235657349) (17 messages🔥):

> - `Mojo and Python stdlib`
> - `Function Parameters vs. Arguments`
> - `Use of LLMs for Translation`
> - `Multilingual Documentation`
> - `Immersive Translate Tool`

- **Mojo's Ambition to Reimplement Python stdlib**: A user inquired if Mojo, aiming to be a superset of Python, plans to reimplement everything in the Python stdlib, guessing the plan likely does not extend there.
  
  - In response, a member noted that while it theoretically could be possible, practically it will take a significant amount of time.
- **Clarity on Parameters and Arguments in Mojo**: Discussion arose regarding the translation nuances of 'function parameters' and 'function arguments' into Chinese, which are both labeled as '函数参数'.
  
  - A user suggested that documentation should specify 'parameters' more clearly, possibly as 'compile-time parameters' to avoid confusion.
- **LLMs Enhancing Translation Efficiency**: One member mentioned that many in the Chinese community are shifting towards using LLMs for translation instead of manual documentation, noting the speed and usability of LLM results.
  
  - Another member suggested utilizing prompts to ensure accurate translations for terms like 'parameter' into '编译期参数'.
- **Exploring Immersive Translate for Better Documentation**: A member introduced the 'Immersive Translate' tool as a highly rated bilingual translation website that uses various AI engines for text translation.
  
  - The tool allows users to translate content conveniently and is recognized as the most popular translation app in China.
- **Collecting Findings for the Chinese Community**: A suggestion was made about compiling findings related to using LLMs, such as tutorials and prompts, into a dedicated post for easier access by the Chinese community.
  
  - This aims to streamline information sharing for those seeking similar solutions in the future.

**Links mentioned**:

- [接入兼容 OpenAI API 接口的 AI 模型 | 沉浸式翻译](https://immersivetranslate.com/zh-Hans/docs/services/ai/#system-promptpromptmultiple-promptsubtitle-prompt),): 原作者：萧萧然
- [Bilingual Web Translation Extension_PDF Document Translation Tool | Immersive Translate](https://immersivetranslate.com/): Immersive Translate is a free-to-use website translation extension that provides you with online bilingual web page translation. It can be used to translate websites to English or other languages, doc...
- [Parameterization: compile-time metaprogramming | Modular Docs](https://docs.modular.com/mojo/manual/parameters/): An introduction to parameters and compile-time metaprogramming.

---

### **Modular (Mojo 🔥) ▷ #**[**max**](https://discord.com/channels/1087530497313357884/1212827597323509870/1296213268800471040) (10 messages🔥):

> - `Mojo version of MAX`
> - `Jakub's Python API work`
> - `Driver demonstration`

- **Exploring Mojo version of MAX**: A member asked if there are plans to create a **Mojo** version since the current implementation is in **Python**.
  
  - Another member mentioned it's taking time to adapt **MAX** for Mojo since it's relatively new.
- **Jakub's work on Python API**: There was a query about the ongoing work by **Jakub** from **Modular** regarding the **Python API** for **MAX**.
  
  - Members discussed the details and requested links to know more about his contributions.
- **Driver demonstration feedback**: The demonstration of the driver showcased how easy it is to implement the model, although it's not fully released yet and only partially available in **nightly builds**.
  
  - One member expressed appreciation for the demo, stating they listened to it multiple times for better understanding.

 

**Link mentioned**: [max/examples/graph-api/pipelines/llama3 at main · modularml/max](https://github.com/modularml/max/tree/main/examples/graph-api/pipelines/llama3): A collection of sample programs, notebooks, and tools which highlight the power of the MAX Platform - modularml/max

 

---

### **Stability.ai (Stable Diffusion) ▷ #**[**general-chat**](https://discord.com/channels/1002292111942635562/1002292112739549196/1296252544854786049) (24 messages🔥):

> - `Stable Diffusion Prompt Suggestions`
> - `Fooocus Model Compatibility`
> - `Face Swap Features in Automatic1111`
> - `Image Quality Concerns`
> - `AI Hackathon Announcement`

- **Stable Diffusion Needs Help With Prompts**: A member sought help for a prompt to create a shadow effect for a **cube** without showing the light source above it, emphasizing the importance of lighting in the scene.
  
  - Multiple members discussed varying experiences with prompt effectiveness, showcasing the demand for more tailored suggestions.
- **Fooocus Models and Compatibility**: Inquiring about model compatibility, a member learned that **Fooocus** primarily uses **SDXL**, while another confirmed it can also work with **pony models**.
  
  - This exchange highlighted the community's focus on ensuring compatibility for enhanced user experience.
- **Face Swap Feature Solutions**: A member asked how to replicate the **faceswap** feature from **Fooocus** in **Automatic1111**, and another suggested using the **Reactor extension** or **IP-Adapter face**.
  
  - This showcases a collaborative effort among users to enhance tool functionality across different platforms.
- **Concerns About Image Quality**: A member reported that their generated images lacked details despite using 30 steps and multiple **LORA** models, seeking advice on potential solutions.
  
  - This prompted discussions about the various factors that could impact image quality in **Stable Diffusion** processes.
- **AI Hackathon for Innovative Projects**: An announcement highlighted a **Gen AI Agents** hackathon inviting teams and individuals to create AI solutions that enhance human potential through collaboration.
  
  - Participants are encouraged to consider ethical implications while developing safe and secure AI systems aimed at optimizing daily tasks.

 

**Link mentioned**: [Vertical Specific AI Agents Hackathon · Luma](https://lu.ma/ke0rwi8n): Gen AI Agents CreatorsCorner, collaborating with aixplain, Sambanova Systems, Prem, Marly, Senso, Mistral, coval, heygen, fiberplane, exa, and others…

 

---

### **Torchtune ▷ #**[**announcements**](https://discord.com/channels/1216353675241590815/1216353675241590818/1296526376610037770) (1 messages):

> - `PyTorch 2.5.0 Release`
> - `FlexAttention Feature`
> - `Per-Layer Compile`
> - `Contributing to Torchtune`

- **PyTorch 2.5.0 officially launched!**: The highly anticipated **PyTorch 2.5.0** has been officially released, which includes new features such as [FlexAttention](https://github.com/pytorch/pytorch/releases/tag/v2.5.0) and **per-layer compile**.
  
  - Users are encouraged to upgrade their local **torch** installations to take advantage of the latest features.
- **Tracker for Torchtune contributions**: For those looking to contribute to **Torchtune**, a tracker has been set up for cleaning the repository for full **PyTorch 2.5.0** support available [here](https://github.com/pytorch/torchtune/issues/1861).
  
  - This initiative aims to ensure the library aligns with the latest updates and improvements in PyTorch.

 

**Link mentioned**: [Issues · pytorch/torchtune](https://github.com/pytorch/torchtune/issues/1861.): PyTorch native finetuning library. Contribute to pytorch/torchtune development by creating an account on GitHub.

 

---

### **Torchtune ▷ #**[**general**](https://discord.com/channels/1216353675241590815/1216353675744641096/1296477139759267911) (12 messages🔥):

> - `Qwen 2.5 Model Integration`
> - `Tokenizer Modifications`
> - `Fine-tuning Guidance`
> - `Special Tokens Usage`

- **Qwen 2.5 Model Integration in Torchtune**: [The Qwen team has released Qwen 2.5](https://github.com/pytorch/torchtune/issues/1624) including various models that are being requested for integration into Torchtune, but updates are still pending.
  
  - Members are collaborating to add the model, and there's an openness for others to contribute if they are interested in the integration process.
- **Guidance on Tokenizer Modifications**: Users discussed modifying the tokenizer to support the new Qwen 2.5 model, particularly referencing [the tokenizer file](https://github.com/pytorch/torchtune/blob/main/torchtune/models/qwen2/_tokenizer.py) for necessary edits.
  
  - It's suggested to examine the [config file on Hugging Face](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct/blob/main/tokenizer_config.json) for details on what modifications are needed.
- **Fine-tuning Questions and Resource Requests**: A user is seeking detailed guidance on running fine-tuning for Qwen 2.5, acknowledging their novice status in the process.
  
  - Feedback includes an offer for help and pointers to the right files and dependencies, emphasizing community support.
- **Discussion on Special Tokens**: Inquiries were raised about where to implement new special tokens in the tokenizer, specifically regarding their usage in message roles like 'ipython'.
  
  - A member prompted for clarity on where to add these tokens, ensuring thorough integration into the existing model framework.

**Links mentioned**:

- [Qwen/Qwen2.5-14B-Instruct · Hugging Face](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct): no description found
- [Qwen 2.5 is here, Request for adding a model · Issue #1624 · pytorch/torchtune](https://github.com/pytorch/torchtune/issues/1624): The Qwen team has released qwen 2.5 base, coder, math models. They seem very promising. Requesting Team to add this model in torchtune. The content you are editing has changed. Please copy your edi...
- [Qwen 2.5 is here, Request for adding a model · Issue #1624 · pytorch/torchtune](https://github.com/pytorch/torchtune/issues/1624#issuecomment-2361869824)): The Qwen team has released qwen 2.5 base, coder, math models. They seem very promising. Requesting Team to add this model in torchtune. The content you are editing has changed. Please copy your edi...

---

### **Torchtune ▷ #**[**papers**](https://discord.com/channels/1216353675241590815/1293438210097025085/1296277177850925100) (10 messages🔥):

> - `Torchtune Papers`
> - `PhD Internship Aspirations`
> - `Implementation Collaboration`
> - `PPO Work Progress`
> - `RFCs and Branching`

- **Torchtune's Potential Paper**: The team humorously noted that a Torchtune paper might be written in about **10 years** when they find the time.
  
  - *"When we sit down and write one"* was the optimistic response regarding the timeline.
- **Excitement Around New Research**: A user shared an interesting paper on [arXiv](https://arxiv.org/pdf/2410.10630), sparking interest and excitement.
  
  - Another member expressed hope for a **PhD internship** to work on projects like those discussed in the paper.
- **Collaboration on Implementation**: Discussion arose about collaborating on implementing the ideas from the arXiv paper into **Torchtune**.
  
  - *"Come help me implement it in torchtune :)"* suggests eagerness to tackle the project together.
- **Ongoing Work on PPO**: One member indicated that they need to finish up their work on **PPO** before starting new tasks.
  
  - *"I gotta land a few RFCs first and finish up my PPO work"* reflects the current priorities within the team.
- **Starting Points for Development**: The team acknowledged the necessity to start somewhere with their projects.
  
  - The sentiment of progress was reiterated with *"but we gotta start somewhere"*, highlighting proactive engagement.

 

---

### **OpenInterpreter ▷ #**[**general**](https://discord.com/channels/1146610656779440188/1147665339266650133/1296219929745555469) (12 messages🔥):

> - `OpenInterpreter Task Issues`
> - `Kernel Panic on App Close`
> - `Integrating O1 in Workflow`
> - `Extracting Tar Files`
> - `OpenInterpreter GitHub Resources`

- **OpenInterpreter task completion issues**: Users are experiencing problems completing tasks with **OpenInterpreter**, repeatedly encountering a script that claims an action has been taken without any action occurring.
  
  - One suggested posting in a dedicated channel with details like version and model to facilitate troubleshooting.
- **Kernel panic when closing app**: A member reported encountering a **kernel panic** when attempting to close the OpenInterpreter app.
  
  - Help was suggested to be sought in the appropriate channels for troubleshooting this issue.
- **Integrating O1 into everyday workflow**: A user seeks advice on how to integrate **O1** into daily tasks, expressing excitement over a recent **NLP project** completion.
  
  - Discussions suggested that the integration heavily depends on the specific workflows and automation needs of the user.
- **Using OI to extract tar files**: A member humorously noted the struggles with extracting **tar files** and how OpenInterpreter helps by automating this process.
  
  - Another user expressed relief at finally understanding how to run the extraction command properly after some initial confusion.
- **OpenInterpreter GitHub resources shared**: A link to the **OpenInterpreter** GitHub [repository](https://github.com/OpenInterpreter/open-interpreter/tree/main/scripts) was shared, showcasing available scripts for users.
  
  - The shared link aims to support those who are interested in leveraging existing scripts for **natural language** processing tasks.

 

**Link mentioned**: [open-interpreter/scripts at main · OpenInterpreter/open-interpreter](https://github.com/OpenInterpreter/open-interpreter/tree/main/scripts): A natural language interface for computers. Contribute to OpenInterpreter/open-interpreter development by creating an account on GitHub.

 

---

### **OpenInterpreter ▷ #**[**O1**](https://discord.com/channels/1146610656779440188/1194880263122075688/1296327996906672274) (4 messages):

> - `Android QR Code Issues`
> - `Miniature Android Phone Tips`
> - `IOS vs. Android Performance`

- **Trouble Scanning QR Codes on Android**: A user is experiencing issues with the **Android client** not responding after scanning a QR code, while everything works perfectly on **iOS**.
  
  - They seek suggestions before examining the source code further, expressing frustration with the lack of functionality.
- **Seeking Tips for Miniature Android Phone Compatibility**: The user acquired a miniature Android phone, similar to one used in a demo, and is looking for advice on possible missing arguments or configurations in the repository.
  
  - They appreciate any insights the community might offer to enhance compatibility.
- **Shell Not Responding After Scan on Android**: The user reported that they cannot get the **shell** to activate after a scan on Android, while it works consistently on iOS.
  
  - This highlights ongoing issues with the Android client, prompting a call for help from the community.

 

---

### **OpenInterpreter ▷ #**[**ai-content**](https://discord.com/channels/1146610656779440188/1149229778138824765/1296306185812840540) (6 messages):

> - `Free LLM Integrations`
> - `Open Interpreter Scripts`
> - `Using AI in Vim`

- **Searching for Free LLM Options**: A user inquired about free LLMs that can be integrated with Chat GPT due to [rising API costs](https://link.url). Another user suggested considering the `i model` with the command `interpreter --model i` if local models aren't feasible.
- **Celebration of New Tools**: A user expressed excitement, remarking that it's 'about time' for advancements in the field, sparking a celebratory response from the community.
  
  - *Right?* ✨
- **Open Interpreter's** `wtf` Script Revealed: Mikebirdtech introduced the `wtf` script from Open Interpreter, showcasing its utility in [Tool Use](https://www.youtube.com/watch?v=Vz3cjbf4zeo) through a demo by Ty.
  
  - The script serves as a notable feature, expanding the functionalities users can explore.
- **AI Integration in Vim**: Mikebirdtech shared insights from Jake Koenig, who demonstrated how to use AI within Vim, available in a tutorial video [here](https://www.youtube.com/watch?v=Ho9yf7ks5sE).
  
  - This adds to the toolkit for developers seeking to enhance their coding experience with AI.

 

---

### **DSPy ▷ #**[**show-and-tell**](https://discord.com/channels/1161519468141355160/1202371242519441499/1296506311483326525) (4 messages):

> - `Multi-label classification for scientific documents`
> - `Heterogeneous graph neural networks`
> - `In-context learning`
> - `BootstrapFewShotWithRandomSearch`
> - `Medium article on research`

- **Innovative Multi-label Classification Approach**: A member shared an exciting new approach to multi-label classification for scientific documents, building on [previous work](https://link.to.research) in in-context learning for extreme multi-label classification.
  
  - They described creating a **Heterogeneous graph** with red nodes as documents and blue nodes as labels, expressing enthusiasm about its potential to search large corpora effectively.
- **Clustering Labels with Neural Networks**: The member explained using a **Heterogeneous Graph neural network** for clustering labels, although they weren't satisfied with imputed edges and hope in-context learning will yield better results.
  
  - They also mentioned using `BootstrapFewShotWithRandomSearch` to pick demonstrations from each cluster for document inference.
- **Interest in Medium Article**: After sharing their work, the author inquired if the community would be interested in a **Medium article** detailing the methodology and findings.
  
  - Responses were overwhelmingly positive, with members expressing strong enthusiasm for such an article.

 

---

### **DSPy ▷ #**[**general**](https://discord.com/channels/1161519468141355160/1161519469319946286/1296266853198598188) (16 messages🔥):

> - `Langtrace DSPy integration`
> - `DSPy prompt optimization issues`
> - `DSPy answer guarantees`
> - `Feedback on DSPy documentation`

- **Langtrace shines with DSPy integration**: Members discussed the promising integration of **Langtrace** with **DSPy**, highlighting the [setup instructions](https://docs.langtrace.ai/supported-integrations/llm-frameworks/dspy) that detail how to capture traces from DSPy pipelines.
  
  - The setup process includes installing DSPy, initializing Langtrace’s SDK, and creating a project with type **DSPy**.
- **DSPy prompt optimization not reflected in JSON**: A member reported that after optimizing a simple classifier with **MIPROV2**, the JSON config retained the original prompt instead of the optimized one, leading to questions about performance loss.
  
  - Discussion ensued regarding potential bugs in saving or loading configurations, with suggestions to investigate the contents of the JSON file.
- **Gaining guaranteed answers from DSPy**: A member inquired about ensuring DSPy returns answers from a list of possibilities, to which it was suggested to use the **Literal[]** type.
  
  - This technique may provide more control over valid output responses in their applications.
- **Positive feedback on DSPy documentation**: A user expressed appreciation for the new DSPy getting started guide, highlighting the approachable breakdown and complete RAG implementation as particularly helpful for newcomers.
  
  - Suggestions included the addition of interactive notebooks and a 'Try It Yourself' section for hands-on learning at the end.
- **Acknowledgment of valuable feedback**: In response to the feedback on the DSPy documentation, a member acknowledged the input as incredibly helpful and confirmed its usefulness for future improvements.
  
  - This conversation reflected a collaborative spirit in enhancing the learning materials for DSPy users.

 

**Link mentioned**: [DSPy - Langtrace AI Docs](https://docs.langtrace.ai/supported-integrations/llm-frameworks/dspy): no description found

 

---

### **DSPy ▷ #**[**colbert**](https://discord.com/channels/1161519468141355160/1250300504462856265/1296576706940899388) (1 messages):

> - `ColbertV2 Training`
> - `Data Format Confusion`

- **ColbertV2 Training Takes Triples & Queries**: The training example for **ColbertV2** takes in triples, collections, and queries as documented on the [GitHub repository](https://github.com/stanford-futuredata/ColBERT?tab=readme-ov-file#advanced-training-colbertv2-style). This indicates a complex data handling mechanism that requires clarity.
  
  - Members expressed confusion over how the dataset relates to indexed versions of **queries** and **collections** seen in examples.
- **Dataset Format Mirrors Raw Query Example**: When printing the first few characters of the dataset referenced, it appears similar to the `raw_query` format discussed. This observation aligns with indexing methods for the ColbertV2 training process.

 

**Link mentioned**: [GitHub - stanford-futuredata/ColBERT: ColBERT: state-of-the-art neural search (SIGIR'20, TACL'21, NeurIPS'21, NAACL'22, CIKM'22, ACL'23, EMNLP'23)](https://github.com/stanford-futuredata/ColBERT?tab=readme-ov-file#advanced-training-colbertv2-style)): ColBERT: state-of-the-art neural search (SIGIR'20, TACL'21, NeurIPS'21, NAACL'22, CIKM'22, ACL'23, EMNLP'23) - stanford-futuredata/ColBERT

 

---

### **tinygrad (George Hotz) ▷ #**[**general**](https://discord.com/channels/1068976834382925865/1068976834928193609/1296204933565255731) (8 messages🔥):

> - `MSE and MAE in Tensors`
> - `Library Loading Fix`
> - `LLVM Load for Gates`
> - `CLOUD=1 with Multi-Device`

- **MSE and MAE Implementation in Tensors**: A member shared a link to a [pull request](https://github.com/tinygrad/tinygrad/pull/7107) implementing **MSE** in `tensors.py` along with tests.
  
  - They indicated that **MSE** and **MAE** could be summarized in just two lines, worth adding to tensors.
- **Fixing Library Loading in Autogen**: A suggestion was made to correct the loading of **libc** in `autogen_stubs.sh` to handle cases where `find_library` returns **None**.
  
  - The member highlighted that this issue arises due to a big hack in the current method, suggesting a more reliable implementation.
- **Addressing LLVM Load with If_Then Gates**: It was pointed out that the current implementation of loading **LLVM** needs an adjustment to use **if_then** for handling gates.
  
  - The current approach is acknowledged as a hack, implying a need for a more structured fix.
- **Query on CLOUD=1 with Multi-Device Setup**: A member inquired about how **CLOUD=1** would function in a multi-device environment, indicating curiosity if it aligns with existing multi-device handling.
  
  - They assumed the behavior would be consistent with current multi-device operations on the same machine.

 

**Link mentioned**: [MSE in tensors.py and tests implemented by littlemountainman · Pull Request #7107 · tinygrad/tinygrad](https://github.com/tinygrad/tinygrad/pull/7107): MSE with testing implemented

 

---

### **tinygrad (George Hotz) ▷ #**[**learn-tinygrad**](https://discord.com/channels/1068976834382925865/1070745817025106080/1296243225052319765) (13 messages🔥):

> - `Update EMA Parameters`
> - `Skills Transfer from Tinygrad`
> - `Learning Resources for Tinygrad`
> - `Deep Learning Philosophy`
> - `Debugging and Deploying Neural Networks`

- **Curiosity about EMA Parameter Decay**: A member expressed curiosity about the *decay* in `update_ema_parameters`, wondering if this technique is common practice in the field.
  
  - This indicates an interest in understanding deeper mechanics behind deep learning optimizations.
- **Learning Tinygrad Benefits Transfer**: Discussion arose about whether skills learned from **Tinygrad** would transfer to libraries like **PyTorch**, with a consensus that it would greatly enhance understanding.
  
  - One contributor highlighted that learning **Tinygrad's philosophy** helped them grasp complex systems better, especially in hardware and robotics.
- **Recommended Resources for Tinygrad Learning**: A member suggested starting with the Beautiful MNIST example and modifying a specific [OpenAI Cookbook example](https://cookbook.openai.com/examples/rag_with_graph_db) to better understand Tinygrad's functionalities.
  
  - More resources were provided, including [tinygrad-notes](https://mesozoic-egg.github.io/tinygrad-notes) for studying internal workings.
- **Deep Learning as a Coherent Unit**: A discussion emphasized the importance of viewing **deep learning** processes as a *whole unit* rather than just isolated pieces, which aids in debugging and deployment.
  
  - Key aspects such as **hyperparameters** and architecture configurations were noted as critical to the overall performance of networks.
- **Appreciation for Deep Learning Insights**: A member thanked another for their insightful advice on learning Tinygrad, highlighting it as one of the best pieces of guidance received recently.
  
  - This reflects the collaborative spirit within the community, focused on sharing knowledge for mutual growth in AI development.

**Links mentioned**:

- [Tutorials on Tinygrad](https://mesozoic-egg.github.io/tinygrad-notes): Tutorials on tinygrad
- [RAG with a Graph database | OpenAI Cookbook](https://cookbook.openai.com/examples/rag_with_graph_db): Open-source examples and guides for building with the OpenAI API. Browse a collection of snippets, advanced techniques and walkthroughs. Share your own examples and guides.
- [Build software better, together](https://github.com/tinygrad/tinygrad/pull/6690/files).): GitHub is where people build software. More than 100 million people use GitHub to discover, fork, and contribute to over 420 million projects.

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**general**](https://discord.com/channels/1104757954588196865/1104757955204743201/1296279793402122271) (15 messages🔥):

> - `Axolotl Dataset Shuffling`
> - `Gradient Accumulation Issues`
> - `Bitnet Release`

- **Axolotl shuffles dataset**: Prior to training, **Axolotl** shuffles the dataset, ensuring randomness for each epoch.
  
  - One member confirmed the behavior after looking for references to validate their understanding.
- **Gradient Accumulation Discrepancies**: A shared issue indicates that **gradient accumulation** may not match losses between full batch training and toggled settings.
  
  - This was highlighted in discussions regarding a blog post while noting that **Hugging Face** should release a fix soon.
- **Member discusses training experience**: *Glad I didn’t start my 12b train yesterday* was a member's remark, referencing potential challenges with gradient accumulation.
  
  - Another member confirmed they had been debugging related issues, with encouragement from others to take breaks.
- **Bitnet 1-bit LLMs released**: **Bitnet**, an official inference framework for 1-bit LLMs, has been released and can be found on [GitHub](https://github.com/microsoft/BitNet).
  
  - The announcement included a brief overview along with an image from the repository.

**Links mentioned**:

- [Fixing Gradient Accumulation](https://huggingface.co/blog/gradient_accumulation): no description found
- [How to ensure the dataset is shuffled for each epoch using Trainer and Datasets?](https://discuss.huggingface.co/t/how-to-ensure-the-dataset-is-shuffled-for-each-epoch-using-trainer-and-datasets/4212): I am using the Seq2SeqTrainer and pass an datasets.arrow_dataset.Dataset as train_dataset when initiating the object. Is the dataset by default shuffled per epoch? If not, how to make it shuffled? An...
- [GitHub - microsoft/BitNet: Official inference framework for 1-bit LLMs](https://github.com/microsoft/BitNet): Official inference framework for 1-bit LLMs. Contribute to microsoft/BitNet development by creating an account on GitHub.

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**other-llms**](https://discord.com/channels/1104757954588196865/1104758057449308220/1296309856218779770) (4 messages):

> - `A100 compute usage`
> - `DeepSpeed issues`

- **Invisietch utilized A100 for 3 days**: @nanobitz inquired about the compute used, to which **invisietch** responded that **1x A100** was employed over a period of **3 days**.
  
  - The discussion highlights the **specific hardware** setup utilized for a particular task.
- **Challenges with DeepSpeed**: Invisietch cited difficulties stating, *“Because I couldn’t get DeepSpeed to work,”* indicating a possible setup or compatibility issue.
  
  - This raises questions about the practical implementation of **DeepSpeed** in their workflow.

 

---

### **Cohere ▷ #**[**discussions**](https://discord.com/channels/954421988141711382/954421988783444043/1296352843472699394) (12 messages🔥):

> - `Cohere tool response yielding`
> - `Command R+ performance`
> - `Inverse Reinforcement Learning for LLMs`
> - `Stealth multilingual project`
> - `Langgraph integration updates`

- **Yielding responses from Cohere tools**: A user inquired about how to yield responses from tools utilizing **Cohere** while using **langgraph**, expressing frustration about the lack of clarity in documentation.
  
  - Other members discussed the potential of using a *for loop* as a fallback method if their current approach using `chat_stream` didn't yield results.
- **Discussions on Command R+ performance**: A member shared that **version 0.8** of **Command R+** performed worse compared to **version 0.4** after a month of usage, seeking insights into the reasons behind this discrepancy.
  
  - There was also a query regarding any planned updates to improve performance in the future.
- **Curiosity on Inverse RL for LLMs**: A user shared a link to a paper on **Inverse Reinforcement Learning** for **LLMs** and expressed curiosity about community opinions on this direction.
  
  - This sparked interest in discussing innovative approaches in the AI field.
- **Call for participation in multilingual stealth project**: A community member announced a call for builders to participate in a **stealth** project requiring language expertise over the next week, with a link to join the **Aya** server.
  
  - Contributors would have their work acknowledged, with exclusive swag prepared for top collaborators, emphasizing involvement in the multilingual space.
- **Updates on Langgraph integration**: Members mentioned new documentation on the **langgraph** integration with **Cohere**, which could assist users in leveraging tools effectively.
  
  - There were hints at upcoming examples and changes to improve functionality within the **chat_stream** feature in the near future.

**Links mentioned**:

- [Tools on LangChain — Cohere](https://docs.cohere.com/docs/tools-on-langchain#langgraph-agents): Explore code examples for multi-step and single-step tool usage in chatbots, harnessing internet search and vector storage.
- [Cannot stream response from cohere · Issue #592 · cohere-ai/cohere-python](https://github.com/cohere-ai/cohere-python/issues/592): I am using langgraph stream_events and inside tools i am using cohere. from langgraph.prebuilt import create_react_agent async def generate_stream_response(message: str, user: dict, prompt_dict: di...

---

### **Cohere ▷ #**[**questions**](https://discord.com/channels/954421988141711382/1168411509542637578/1296551096633786409) (2 messages):

> - `RAG AMAs Recording`
> - `Course Creators`

- **RAG AMAs were not recorded**: A member inquired if the **RAG AMAs** were recorded, expressing interest in any available material.
  
  - Responding to the query, a participant confirmed that they **weren't recording them** and encouraged questions to be directed to a specific course creator.
- **Contact Course Creators for Questions**: The same participant suggested tagging **<@955487948705513472>** for any questions regarding the course.
  
  - This emphasizes the openness of the course creators to engage with participants' inquiries directly.

 

---

### **Cohere ▷ #**[**projects**](https://discord.com/channels/954421988141711382/1218409701339828245/) (1 messages):

sssandra: congrats! tho off-topic so removing it from here 🙂

---

### **LLM Agents (Berkeley MOOC) ▷ #**[**mooc-questions**](https://discord.com/channels/1280234300012494859/1280370030609170494/1296439890862669855) (13 messages🔥):

> - `Quiz Access`
> - `Course Navigation`
> - `Written Article Review`
> - `Course Websites`
> - `MOOC Participation`

- **Issues Accessing Week 5 Quiz**: A member reported problems accessing the **Week 5 quiz** which is located in the syllabus section of the course website, specifically [here](https://llmagents-learning.org/f24).
  
  - Another member confirmed the quiz availability and guided them to the correct section for access.
- **New Members Seeking Course Guidance**: A newcomer asked about receiving follow-up emails after filling out a course form and sought clarification on accessing course materials. They were reassured that it was fine to proceed with course participation and complete quizzes as needed.
  
  - Members encouraged focusing on the MOOC without stressing over hackathons and supported reviewing supplemental materials.
- **Clarification on Course Websites**: Members discussed two different course websites and confirmed that the site part of **llmagents-learning.org** is the correct one for MOOC students. The other site is primarily for **UC Berkeley** students attending classes physically.
  
  - They advised against using the Berkeley site for course-related activities and mentioned the need for separate forms depending on your student status.
- **Article Review Request for Assignment**: A member inquired if they could get their article reviewed before posting it on social media to ensure it meets course expectations. Concerns were raised about the review process being complicated, but others emphasized adherence to the article guidelines provided on the course website.
  
  - It was suggested that as long as the article aligns with the general criteria, members should not worry significantly about pre-review.
- **Update on Weekly Course Progress**: One participant updated the group that they just finished **Week 1** and planned to continue accordingly with the course's outlined structure. They were appreciated for their initiative and were encouraged to keep progressing through the weekly content.

**Links mentioned**:

- [Large Language Model Agents](https://llmagents-learning.org/f24): no description found
- [CS294/194-196 Large Language Model Agents](https://rdi.berkeley.edu/llm-agents/f24): Fall 2024

---

### **LangChain AI ▷ #**[**general**](https://discord.com/channels/1038097195422978059/1038097196224086148/1296506232353591378) (7 messages):

> - `AI Engineering Blogs`
> - `LangChain vs LangGraph`
> - `LangChain Critique`
> - `Agent Visualization`
> - `LangGraph Tools`

- **AI Engineering Blogs to Follow**: A user asked for recommendations on great AI Engineering blogs, expressing specific interest in **Retrieval systems** and **Multi-agent architectures**.
  
  - *No specific blogs were listed*.
- **Pros of Switching to LangGraph**: Discussion revolved around the advantages of switching from **LangChain to LangGraph**, especially regarding abstraction and usability.
  
  - A member inquired about what unique features **LangGraph** offers that **LangChain** does not.
- **Criticism of LangChain After Two Years**: A longtime user of **LangChain** reflected on the criticisms surrounding the tool, despite having spent considerable time learning it.
  
  - They humorously noted their frustration from late-night attempts to master **LangChain**.
- **Visualizing Agent Graphs**: A request was made about how to visualize or create a graph for agents within their projects.
  
  - *No solutions were provided in the discussion.*
- **LangGraph's Tool Access**: A member prompted a discussion on the tools that **LangGraph** has access to, seeking more insight into its capabilities.
  
  - *No detailed insights were shared in response.*

 

---

### **LAION ▷ #**[**general**](https://discord.com/channels/823813159592001537/823813160075132991/1296489925990027295) (3 messages):

> - `Inverse Reinforcement Learning for LLMs`
> - `NotebookLM Features`
> - `Gen AI Agent Hackathon`

- **Exploring Inverse RL for LLMs**: A recent paper discussing the application of **inverse reinforcement learning** for **LLMs** was shared, prompting interest and curiosity for feedback on this direction [PDF here](https://arxiv.org/pdf/2410.12491).
  
  - Participants are eager to hear thoughts on the viability and implications of this approach in advancing language model capabilities.
- **NotebookLM Introduces Exciting Features**: **Google** announced new features for **NotebookLM**, including audio overviews and enhanced collaboration tools as part of the notebook's business pilot program [details here](http://goo.gle/3UcO8Na).
  
  - New functionalities aim to improve user experience by allowing seamless multitasking while engaging with audio content.
- **Hackathon Invites Teams to Build Gen AI Agents**: **CreatorsCorner** invites participants to a hackathon focused on developing **Gen AI-powered multi-agent systems** that support users in daily tasks while ensuring safety and security [more info here](https://lu.ma/ke0rwi8n).
  
  - The challenge encourages innovation in creating collaborative AI solutions that consider ethical implications and societal benefits.

**Links mentioned**:

- [Tweet from Google (@Google)](https://x.com/Google/status/1846954813193359397?t=8gWKjTOUhZAYbjFMHluqGw&s=19): ✨ New features coming to NotebookLM ✨ 🗣️ Customize Audio Overviews and guide the conversation 🤝 Collaborate with teammates in the NotebookLM Business pilot program 🎧 Listen to Audio Overviews while...
- [Vertical Specific AI Agents Hackathon · Luma](https://lu.ma/ke0rwi8n): Gen AI Agents CreatorsCorner, collaborating with aixplain, Sambanova Systems, Prem, Marly, Senso, Mistral, coval, heygen, fiberplane, exa, and others…

---

### **LAION ▷ #**[**research**](https://discord.com/channels/823813159592001537/824374369182416994/1296381944522604565) (4 messages):

> - `Graph Reinforcement Learning`
> - `Inverse Reinforcement Learning for LLMs`
> - `Importance of Survey Papers`

- **Excitement over Graph Reinforcement Learning Survey**: A member expressed excitement about discovering a new [survey on Graph Reinforcement Learning](https://arxiv.org/abs/2404.06492), highlighting its potential as a decision-making method across various domains.
  
  - They pointed out that the synthesis between **graph structures** and **reinforcement learning** can lead to innovative strategies in fields like chemistry and computer science.
- **Praise for Survey Writers**: Another member shared a sentiment about the value of well-written surveys, calling them *godsends* for researchers.
  
  - This reflects a broader appreciation in the community for comprehensive literature reviews which aid in understanding complex topics.
- **Discussion on Inverse Reinforcement Learning for LLMs**: A request for opinions emerged regarding an [article on using inverse RL for LLMs](https://arxiv.org/pdf/2410.12491).
  
  - The inquiry indicates an interest in exploring how **inverse reinforcement learning** can be applied to enhance the capabilities of large language models.

 

**Link mentioned**: [Graph Reinforcement Learning for Combinatorial Optimization: A Survey and Unifying Perspective](https://arxiv.org/abs/2404.06492): Graphs are a natural representation for systems based on relations between connected entities. Combinatorial optimization problems, which arise when considering an objective function related to a proc...

 

---

### **Alignment Lab AI ▷ #**[**general**](https://discord.com/channels/1087862276448595968/1095458248712265841/1296564979989741719) (1 messages):

> - `Twitter/X embeds`
> - `FixTweet/FxTwitter`

- **Fix broken Twitter/X embeds**: Members discussed the need to **fix broken Twitter/X embeds** to allow for more features like multiple images, videos, polls, and translations across platforms like Discord and Telegram.
  
  - One member shared a link to the [FixTweet/FxTwitter initiative](https://x.com/i/spaces/1ypKdpLNZXnKW), urging others to participate in improving embed functionalities.
- **Discussion around tweeting features**: There was a conversation about the impact of **more interactive tweeting features** on user engagement, particularly regarding embeds.
  
  - Members believe that **enhanced multimedia support** could increase overall participation and content sharing.

 

**Link mentioned**: [Tweet from GitHub - FixTweet/FxTwitter: Fix broken Twitter/X embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others](https://x.com/i/spaces/1ypKdpLNZXnKW): Fix broken Twitter/X embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others - FixTweet/FxTwitter

 

---

### **Mozilla AI ▷ #**[**announcements**](https://discord.com/channels/1089876418936180786/1089876419926032396/1296495172653158412) (1 messages):

> - `Gen AI Bug Bounties`
> - `Vulnerability Submission Process`
> - `User Dashboard Features`
> - `Real-Time Notifications`
> - `Training Opportunities`

- **Gen AI Bug Bounties portal is now live**: The [portal](https://discord.com/channels/1089876418936180786/1245784344539435128/1295876886584492033) for **gen AI bug bounties** has officially launched, streamlining the vulnerability submission process with an intuitive design and automatic triage for faster reviews.
  
  - This initiative aims to enhance security by making it easier for researchers to report vulnerabilities.
- **Enhanced User Dashboard features introduced**: The new **Personalized User Dashboard** provides a centralized view for tracking submission status, updates, and researcher progress.
  
  - This dashboard is designed to enhance user experience and streamline the management of submissions.
- **Stay informed with Real-Time Notifications**: **Real-Time Notifications** will send instant email alerts for every action taken on submitted vulnerabilities, ensuring full transparency.
  
  - This feature allows users to stay updated on the status of their submissions without any delays.
- **Secure collaboration through Role-Based Permissions**: The platform implements **Role-Based Permissions** to provide structured access control, securing data management and collaboration.
  
  - This measure ensures that sensitive information is only accessible to authorized individuals.
- **Upcoming Training Opportunities in November**: Exciting **Prompt Engineering Courses & CTF Challenges** are set to launch in November, offering skill-building opportunities focused on AI vulnerabilities.
  
  - These educational initiatives will include continuous **Weekly Blogs & Tutorials**, enhancing knowledge in AI security.

 

---

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