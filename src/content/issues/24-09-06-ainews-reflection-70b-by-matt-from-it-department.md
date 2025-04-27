---
id: 02416d00-dc64-487c-ade4-95eb958e04bb
title: Reflection 70B, by Matt from IT Department
date: '2024-09-07T01:17:07.379983Z'
original_slug: ainews-reflection-70b-by-matt-from-it-department
description: >-
  **Reflection Tuning** technique has been used by a two-person team from
  **Hyperwrite** and **Glaive** to finetune **llama-3.1-70b**, showing strong
  performance improvements with minimal synthetic data. The approach builds on
  the concept of adding `thinking` and `reflection` steps to outputs, related to
  the **Chain of Thought** method. Despite some criticisms like contamination
  concerns, worse coding performance, and reliance on system prompts, the model
  has received positive reception and comparisons to **claude-3.5-sonnet**. The
  work highlights efficient instruction tuning and synthetic data generation for
  large models.
companies:
  - hyperwrite
  - glaive
models:
  - llama-3.1-70b
  - llama-3
  - claude-3.5-sonnet
topics:
  - fine-tuning
  - chain-of-thought
  - instruction-following
  - synthetic-data
  - quantization
  - model-evaluation
  - prompt-engineering
people:
  - matt-shumer
  - sahil-chaudhary
---


<!-- buttondown-editor-mode: plaintext -->**Reflection Tuning is all you need?**

> AI News for 9/5/2024-9/6/2024. We checked 7 subreddits, [**384** Twitters](https://twitter.com/i/lists/1585430245762441216) and **30** Discords (**214** channels, and **2813** messages) for you. Estimated reading time saved (at 200wpm): **304 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

We were going to wait til next week for the paper + 405B, but the reception has been so strong (with [VentureBeat cover story](https://venturebeat.com/ai/meet-the-new-most-powerful-open-source-ai-model-in-the-world-hyperwrites-reflection-70b/)) and the criticisms mostly minor so we are going to make this the title story even though it technically happened yesterday, since no other story comes close.

 ![image.png](https://assets.buttondown.email/images/9f4ddf9a-00aa-4c51-a6b6-817640bb73b6.png?w=960&fit=max) 

TL;DR a two person team; [Matt Shumer](https://x.com/mattshumer_/status/1831767014341538166) from Hyperwrite (who has no prior history of AI research but is a prolific AI builder and influencer) and [Sahil Chaudhary](https://x.com/csahil28) from Glaive finetuned Llama 3.1 70B ([though context is limited](https://www.reddit.com/r/LocalLLaMA/comments/1fanrr4/comment/lluks4i/)) using a technique similar to a one year old paper, [Reflection-Tuning: Recycling Data for Better Instruction-Tuning](https://openreview.net/forum?id=xaqoZZqkPU):
 ![image.png](https://assets.buttondown.email/images/7fb128e7-b5b0-4bb0-9279-4cde9d34a299.png?w=960&fit=max) 

Matt hasn't yet publicly cited the paper, but it almost doesn't matter because the process is retrospectively obvious to anyone who understands the broad Chain of Thought literature: train LLMs to add `thinking` and `reflection` sections to their output before giving a final `output`. 

 ![image.png](https://assets.buttondown.email/images/821a89f8-4a90-46f9-8792-612b2a325729.png?w=960&fit=max) 

This is basically "[Let's Think Step By Step](https://arxiv.org/abs/2205.11916)" in more formal terms, and is surprising to the extent that the Orca series of models ([our coverage here](https://buttondown.com/ainews/archive/ainews-microsoft-agentinstruct-orca-3/)) already showed that Chain of Thought could be added to Llama 1/2/3 and would work:

 ![image.png](https://assets.buttondown.email/images/59c07b47-5b59-42ee-a089-5c98523703b1.png?w=960&fit=max) 

It would seem that Matt has found the ideal low hanging fruit because nobody bothered to take a different spin on 
Orca + generate enough synthetic data (we still don't know how much it was, but it couldn't have been that much given the couple dozen person-days that Matt and Sahil spent on it) to do this until now.

The criticisms have been few and mostly not fatal:

- **Contamination concerns**: [99.2% GSM8K score too high - more than 1% is mislabeled, indicating contamination](https://x.com/hughbzhang/status/1831777846899175576)
  - [Johno Whitaker](https://x.com/johnowhitaker/status/1831800187012202672) independently verified that 5 known wrong questions from GSM8K were answered correctly (aka not memorized)
  - Matt ran the [LMsys decontaminator check](https://x.com/mattshumer_/status/1831767026681180539) on it as well
- **Worse for coding**: [Does worse on BigCodeBench-Hard](https://x.com/terryyuezhuo/status/1832112913391526052) - almost 10 points worse than L3-70B, and [Aider code editing](https://x.com/paulgauthier/status/1832160129720185225) - 7% worse than L3-70B.
- **Overoptimized for solving trivia**: "nearly but not quite on par with Llama 70b for comprehension, but far, far behind on summarization - both in terms of summary content and language. Several of its sentences made no sense at all. I ended up deleting it." - [/r/locallLama](https://www.reddit.com/r/LocalLLaMA/comments/1fanrr4/comment/lluttbm/)
- **Weirdly reliant on system prompts**: "The funny thing is that the model performs the same as base Llama 3.1 if you don't use the specific system prompt the author suggest. He even says it himself." [/r/localllama](https://www.reddit.com/r/LocalLLaMA/comments/1fanrr4/comment/llukruz/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button)
- [grifter](https://x.com/agihippo/status/1831944081066618907)/[hype](https://www.reddit.com/r/LocalLLaMA/comments/1fanrr4/reflection_70b_hype/) alarm bells - Matt [did not disclose that he is an investor in Glaive](https://x.com/gazorp5/status/1831844715379167420).

After a day of review, the overall vibes remain very strong - with /r/localLlama reporting that [even 4bit quantizations of Reflection 70B are doing well](https://www.reddit.com/r/LocalLLaMA/comments/1fan3aa/even_4bit_quants_of_reflection_70b_are_amazing/), and Twitter reporting [riddles](https://x.com/AIandDesign/status/1832221300791943561) and favorable comparisons with [Claude 3.5 Sonnet](https://x.com/gauravpathak/status/1831808959935868941) that it can be said to at least pass the vibe check if not as a generally capable model, but on enough reasoning tasks to be significant.

More information can be found on [this 34min livestream conversation](https://x.com/MatthewBerman/status/1832096560970395704) and [12min recap](https://x.com/MatthewBerman/status/1832098688581431713) with Matthew Berman.

All in all, not a bad day for [Matt from IT](https://x.com/andrew_n_carr/status/1832103565529379270).

 ![image.png](https://assets.buttondown.email/images/70a2f470-9a61-446a-b21f-a62c335d766d.png?w=960&fit=max) 





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

**LLM Training & Evaluation**

- **LLM Training & Evaluation**:  [@AIatMeta](https://twitter.com/AIatMeta/status/1831123520425963684) is still accepting proposals for their **LLM Evaluations Grant** until **September 6th**. The grant will provide **$200K in funding** to support **LLM evaluation research**. 
- **Multi-Modal Models**: [@glennko](https://twitter.com/glennko/status/1831119997306819048) believes that **AI** will eventually be able to count "r" with **high accuracy**, but that it might not be with an **LLM** but with a **multi-modal model**.
- **Specialized Architecture**: [@glennko](https://twitter.com/glennko/status/1831120394796896758) noted that **FPGAS** are too slow and **ASICs** are too expensive to build the specialized architecture needed for **custom logic**.

**Open-Source Models & Research**

- **Open-Source MoE Models**:  [@apsdehal](https://twitter.com/apsdehal/status/1831168945514045633) announced the release of **OLMoE**, a **1B parameter** **Mixture-of-Experts (MoE)** **language model** that is **100% open-source**. The model was a collaboration between **ContextualAI** and **Allen Institute for AI**.
- **Open-Source MoE Models**:  [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1831182890056798530) noted that **OLMOE-1B-7B** has **7 billion parameters** but only uses **1 billion per input token**, and it was pre-trained on **5 trillion tokens**. The model outperforms other available models with **similar active parameters**, even surpassing larger models such as **Llama2-13B-Chat** and **DeepSeekMoE-16B**.
- **Open-Source MoE Models**: [@teortaxesTex](https://twitter.com/teortaxesTex/status/1831233827353608584)  noted that **DeepSeek-MoE** scores well in **granularity**, but not in **shared experts**.

**AI Tools & Applications**

- **AI-Powered Spreadsheets**: [@annarmonaco](https://twitter.com/annarmonaco/status/1831355874872529284) highlighted how **Paradigm** is transforming spreadsheets with AI and using **LangChain** and **LangSmith** to monitor key costs and gain step-by-step agent visibility. 
- **AI for Healthcare Diagnostics**: [@qdrant_engine](https://twitter.com/qdrant_engine/status/1831240904293728327) shared a guide on how to create a high-performance diagnostic system using **hybrid search** with both **text** and **image data**, generating **multimodal embeddings** from text and image data. 
- **AI for Fashion**: [@flairAI_](https://twitter.com/flairAI_/status/1831349517310210199) is releasing a **fashion model** that can be trained on clothing with incredible accuracy, preserving texture, labels, logos, and more with **Midjourney-level quality**.

**AI Alignment & Safety**

- **AI Alignment & Safety**: [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1831286356728848511) shared a podcast discussing the challenges of **AI alignment** and the ability to supervise powerful systems effectively. The podcast included insights from **Anca Diana Dragan** and **Professor FryRSquared**.
- **AI Alignment & Safety**: [@ssi](https://twitter.com/ssi/status/1831327645054947498) is building a "straight shot to safe superintelligence" and has raised **$1B** from investors.
- **AI Alignment & Safety**: [@RichardMCNgo](https://twitter.com/RichardMCNgo/status/1831129636220350562) noted that **EA** facilitates power-seeking behavior by choosing strategies using naive consequentialism without properly accounting for second order effects.

**Memes & Humor**

- **Founder Mode**:  [@teortaxesTex](https://twitter.com/teortaxesTex/status/1831188248883843182) joked about **Elon Musk's Twitter feed**, comparing him to **Iron Man**.
- **Founder Mode**:  [@nisten](https://twitter.com/nisten/status/1831127609901457706) suggested that **Marc Andreessen** needs a better filtering **LLM** to manage his random blocked users.
- **Founder Mode**:  [@cto_junior](https://twitter.com/cto_junior/status/1831303273074016384)  joked about how Asian bros stack encoders and cross-attention on top of existing models just to feel something.


---

# AI Reddit Recap

## /r/LocalLlama Recap

**Theme 1. Advancements in LLM Quantization and Efficiency**

- **[llama.cpp merges support for TriLMs and BitNet b1.58](https://github.com/ggerganov/llama.cpp/pull/8151)** ([Score: 73, Comments: 4](https://reddit.com//r/LocalLLaMA/comments/1fa3ryv/llamacpp_merges_support_for_trilms_and_bitnet_b158/)): **llama.cpp** has expanded its capabilities by integrating support for **TriLMs** and **BitNet b1.58** models. This update enables the use of **ternary quantization** for weights in TriLMs and introduces a **binary quantization** method for BitNet models, potentially offering improved efficiency in model deployment and execution.


**Theme 2. Reflection-70B: A Novel Fine-tuning Technique for LLMs**

- **[First independent benchmark (ProLLM StackUnseen) of Reflection 70B shows very good gains. Increases from the base llama 70B model by 9 percentage points (41.2% -> 50%)](https://i.redd.it/tuawfbwms3nd1.png)** ([Score: 275, Comments: 115](https://reddit.com//r/LocalLLaMA/comments/1fa4y7q/first_independent_benchmark_prollm_stackunseen_of/)): **Reflection-70B** demonstrates significant performance improvements over its base model on the **ProLLM StackUnseen benchmark**, increasing accuracy from **41.2%** to **50%**, a gain of **9 percentage points**. This independent evaluation suggests that Reflection-70B's capabilities may surpass those of larger models, highlighting its effectiveness in handling unseen programming tasks.
  - **Matt from IT** unexpectedly ranks among top AI companies like **OpenAI**, **Google**, and **Meta**, sparking discussions about individual innovation and potential job offers from major tech firms.
  - The **Reflection-70B** model demonstrates significant improvements over larger models, beating the **405B** version on benchmarks. Users express excitement for future fine-tuning of larger models and discuss hardware requirements for running these models locally.
  - Debate arises over the fairness of comparing **Reflection-70B** to other models due to its unique output format using `<thinking>` and `<output>` tags. Some argue it's similar to **Chain of Thought** prompting, while others see it as a novel approach to enhancing model reasoning capabilities.

- **[Reflection-Llama-3.1-70B available on Ollama](https://ollama.com/library/reflection)** ([Score: 74, Comments: 35](https://reddit.com//r/LocalLLaMA/comments/1fa72an/reflectionllama3170b_available_on_ollama/)): The **Reflection-Llama-3.1-70B** model is now accessible on **Ollama**, expanding the range of large language models available on the platform. This model, based on **Llama 2**, has been fine-tuned using **constitutional AI** techniques to enhance its capabilities in areas such as **task decomposition**, **reasoning**, and **reflection**.
  - Users noted an initial **system prompt error** in the model, which was promptly **updated**. The model's name on Ollama mistakenly omitted "llama", causing some amusement.
  - A **tokenizer issue** was reported, potentially affecting the model's performance on **Ollama** and **llama.cpp**. An active discussion on [Hugging Face](https://huggingface.co/mattshumer/Reflection-Llama-3.1-70B/discussions/6) addresses this problem.
  - The model demonstrated its **reflection capabilities** in solving a candle problem, catching and correcting its initial mistake. Users expressed interest in applying this technique to smaller models, though it was noted that the **8B version** showed limited improvement.

## All AI Reddit Recap

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity

**AI Model Developments and Releases**

- **Reflection 70B**: A fine-tuned version of Meta's Llama 3.1 70B model, created by Matt Shumer, is [**claiming to outperform state-of-the-art models on benchmarks**](https://www.reddit.com/r/singularity/comments/1f9uszk/reflection_70b_the_worlds_top_opensource_model/). It uses synthetic data to offer an inner monologue, similar to Anthropic's approach with Claude 3 to 3.5.

- **AlphaProteo**: [**Google DeepMind's new AI model generates novel proteins**](https://www.reddit.com/r/singularity/comments/1f9orj0/google_deepminds_alphaproteo_generates_novel/) for biology and health research.

- **OpenAI's Future Models**: OpenAI is reportedly [**considering high-priced subscriptions up to $2,000 per month**](https://www.reddit.com/r/OpenAI/comments/1f9ovbm/openai_is_reportedly_considering_highpriced/) for next-generation AI models, potentially named Strawberry and Orion.

**AI Industry and Market Dynamics**

- **Open Source Impact**: The release of Reflection 70B has sparked discussions about the [**potential of open-source models to disrupt the AI industry**](https://www.reddit.com/r/OpenAI/comments/1f9ybqy/new_opensource_ai_model_is_smashing_the/), potentially motivating companies like OpenAI to release new models.

- **Model Capabilities**: There's a [**disconnect between public perception and actual AI model capabilities**](https://www.reddit.com/r/singularity/comments/1f9ukpg/people_really_have_0_idea_whats_going_on_the/), with many people unaware of the current state of AI technology.

**AI Applications and Innovations**

- **DIY Medicine**: A report discusses the [**rise of "Pirate DIY Medicine,"**](https://www.reddit.com/r/singularity/comments/1fa0rl1/the_rise_of_pirate_diy_medicine_an_amateur_can/) where amateurs can manufacture expensive medications at a fraction of the cost.

- **Stable Diffusion**: A new [**FLUX LoRA model for Stable Diffusion**](https://www.reddit.com/r/StableDiffusion/comments/1fa5ebi/less_than_24_hours_100_downloads_thrilled_that_my/) has gained popularity, demonstrating the ongoing development in AI-generated art.

---

# AI Discord Recap

> A summary of Summaries of Summaries by Claude 3.5 Sonnet

**1. LLM Advancements and Benchmarking**

- **Reflection 70B Makes Waves**: **[Reflection 70B](https://huggingface.co/mattshumer/Reflection-Llama-3.1-70B)** was announced as the world's top open-source model, utilizing a new **Reflection-Tuning** technique that enables the model to detect and correct its own reasoning mistakes.
   - While initial excitement was high, subsequent testing on benchmarks like **BigCodeBench-Hard** showed mixed results, with scores lower than previous models. This sparked debates about evaluation methods and the impact of synthetic training data.
- **DeepSeek V2.5 Enters the Arena**: **[DeepSeek V2.5](https://x.com/deepseek_ai/status/1832026579180163260)** was officially launched, combining the strengths of DeepSeek-V2-0628 and DeepSeek-Coder-V2-0724 to enhance writing, instruction-following, and human preference alignment.
   - The community showed interest in comparing DeepSeek V2.5's performance, particularly for coding tasks, against other recent models like Reflection 70B, highlighting the rapid pace of advancements in the field.
  


**2. Model Optimization Techniques**

- **Speculative Decoding Breakthrough**: **[Together AI](https://x.com/togethercompute/status/1831755763615674412)** announced a breakthrough in speculative decoding, achieving up to **2x improvement** in latency and throughput for long context inputs, challenging previous assumptions about its effectiveness.
   - This advancement signals a significant shift in optimizing high-throughput inference, potentially reducing GPU hours and associated costs for AI solution deployment.
- **AdEMAMix Optimizer Enhances Gradient Handling**: A new optimizer called **AdEMAMix** was proposed, utilizing a mixture of two Exponential Moving Averages (EMAs) to better handle past gradients compared to a single EMA, as detailed in this [paper](https://arxiv.org/pdf/2409.03137).
   - Early experiments show AdEMAMix outperforming traditional single EMA methods in language modeling and image classification tasks, promising more efficient training outcomes for various AI applications.
  


**3. Open-source AI Developments**

- **llama-deploy Streamlines Microservices**: **[llama-deploy](https://twitter.com/llama_index/status/1831794126511337880)** was launched to facilitate seamless deployment of microservices based on **LlamaIndex Workflows**, marking a significant evolution in agentic system deployment.
   - An [open-source example](https://twitter.com/llama_index/status/1832132462786576652) showcasing how to build an agentic chatbot system using llama-deploy with the @getreflex front-end framework was shared, demonstrating its full-stack capabilities.
- **SmileyLlama: AI Molecule Designer**: **[SmileyLlama](https://x.com/axolotl_ai/status/1831771214445945148)**, a fine-tuned Chemical Language Model, was introduced to design molecules based on properties specified in prompts, built using the **Axolotl** framework.
   - This development showcases Axolotl's capabilities in adapting existing Chemical Language Model techniques for specialized tasks like molecule design, pushing the boundaries of AI applications in chemistry.
  


**4. AI Infrastructure and Deployment**

- **NVIDIA's AI Teaching Kit Launch**: NVIDIA's **Deep Learning Institute** released a [generative AI teaching kit](https://www.hackster.io/news/nvidia-teams-up-with-dartmouth-for-a-free-generative-ai-teaching-kit-11358047a05a) developed with Dartmouth College, aimed at empowering students with GPU-accelerated AI applications.
   - The kit is designed to give students a significant advantage in the job market by bridging knowledge gaps in various industries, highlighting NVIDIA's commitment to AI education and workforce development.
- **OpenAI Considers Premium Pricing**: Reports emerged that **OpenAI** is considering a **$2000/month** subscription model for access to its more advanced AI models, including the anticipated Orion model, as discussed in this [Information report](https://x.com/aiexplainedyt/status/1831710902636228694?s=46).
   - This potential pricing strategy has sparked debates within the community about accessibility and the implications for AI democratization, with some expressing concerns about creating barriers for smaller developers and researchers.
  

---

# PART 1: High level Discord summaries



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Vision Language Models Overview**: A member shared a [blogpost](https://www.lightly.ai/post/introduction-to-vision-language-models) detailing the integration of vision and language in AI applications, emphasizing innovative potentials.
   - This piece aims to steer focus towards the versatile use cases emerging from this intersection of technologies.
- **Tau LLM Training Optimization Resources**: The [Tau LLM series](https://youtube.com/live/flwqvE4aSzA?feature=share) offers essential insights on optimizing LLM training processes, which promise enhanced performance.
   - It is considered pivotal for anyone delving into the complexities of training LLMs effectively.
- **Medical Dataset Quest for Disease Detection**: A member seeks a robust medical dataset for **computer vision**, aimed at enhancing **disease detection** through transformer models.
   - They're particularly interested in datasets that support extensive data generation efforts in this domain.
- **Flux img2img Pipeline Still Pending**: The **Flux img2img** feature remains unmerged, as noted in an [open PR](https://huggingface.co/docs/diffusers/main/en/api/pipelines/flux), with ongoing discussions surrounding its documentation.
   - Despite its potential strain on typical consumer hardware, measures for optimization are being explored, as shared in [related discussions](https://huggingface.co/blog/sd3#memory-optimizations-for-sd3).
- **Selective Fine-Tuning for Enhanced Language Models**: The concept of selective [fine-tuning](https://huggingface.co/blog/anakin87/spectrum) has been highlighted, showcasing its capability in improving language model performance without full retraining.
   - This targeted approach allows for deeper performance tweaks while avoiding the costs associated with comprehensive training cycles.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **ControlNet Enhances Model Pairings**: Users shared successful strategies for using **ControlNet** with **Loras** to generate precise representations like **hash rosin** images using various **SDXL** models.
   - They recommended applying techniques like **depth maps** to achieve better results, highlighting a growing mastery in combining different AI tools.
- **Flux Takes the Lead Over SDXL for Logos**: The community widely endorsed **Flux** over **SDXL** for logo generation, emphasizing its superior handling of logo specifics without requiring extensive training.
   - Members noted that **SDXL** struggles without familiarity with the logo design, making **Flux** the favored choice for ease and effectiveness.
- **Scamming Awareness on the Rise**: Discussion on online scams revealed that even experienced users can be vulnerable, leading to a shared commitment to promote ongoing vigilance.
   - Empathetic understanding of scamming behaviors emerged as a key insight, reinforcing that susceptibility isn't limited to the inexperienced.
- **Tagging Innovations in ComfyUI**: Community insights on tagging features in **ComfyUI** likened its capabilities to **Langflow** and **Flowise**, showcasing its flexibility and user-friendly interface.
   - Members brainstormed specific workflows to enhance tagging efficacy, pointing to a promising wave of adaptations in the interface’s functionality.
- **Insights Into Forge Extensions**: Inquiries into various extensions available in **Forge** highlighted user efforts to improve experience through contributions and community feedback.
   - Polls were referenced as a method for shaping future extension releases, underscoring the importance of quality assurance and community engagement.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Congrats on Y Combinator Approval!**: Team members celebrated the recent backing from **Y Combinator**, showcasing strong **community support** and enthusiasm for the project's future.
   - They acknowledged this milestone as a significant boost toward development and outreach.
- **Unsloth AI Faces Hardware Compatibility Hurdles**: Discussions highlighted **Unsloth's** current struggles with hardware compatibility, notably concerning **CUDA support** on Mac systems.
   - The team aims for **hardware agnosticism**, but ongoing issues reduce performance on certain configurations.
- **Synthetic Data Generation Model Insights**: Insights were shared on employing **Mistral 8x7B tunes** for synthetic data generation, alongside models like **jondurbin/airoboros-34b-3.3** for testing.
   - Experimentation remains essential for fine-tuning outcomes based on hardware constraints.
- **Phi 3.5 Model Outputs Confuse Users**: Users reported frustrating experiences with the **Phi 3.5** model returning **gibberish outputs** during fine-tuning efforts, despite parameter tweaks.
   - This prompted a wider discussion on troubleshooting and refining input templates for better model performance.
- **Interest Surges for Comparison Reports!**: A member expressed eagerness for comparison reports on key topics, emphasizing their potential for **insightful** reading.
   - Parallelly, another member announced plans for a [YouTube video](https://youtube.com) detailing these comparisons, showcasing community engagement.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Seeking Free Image API Options**: Users investigated [free Image API options](https://huggingface.co/lmstudio-community/stable-code-instruct-3b-GGUF) that support high limits, specifically inquiring about providers offering access for models like **Stable Diffusion**.
   - Curiosity was sparked around any providers that could accommodate these features at scale.
- **Reflection Llama-3.1 70B Gets Enhancements**: **Reflection Llama-3.1 70B** impressed as the top open-source LLM with updates that bolster error detection and correction capabilities.
   - However, users noted ongoing performance issues and debated optimal prompts to enhance model behavior.
- **LM Studio Download Problems After Update**: Post-update to version **0.3.2**, users faced challenges downloading models, citing [certificate errors](https://github.com/zed-industries/zed/issues/17482) as a primary concern.
   - Workarounds discussed included adjusting VRAM and context size, while clarifications on the RAG summarization feature were provided.
- **Mac Studio** Battling Speed with Large Models**: Concern arose over **Mac Studio's** capability with **256GB+** memory being sluggish for larger models, with hopes that **LPDDR5X 10.7Gbps** could remedy this.
   - One discussion highlighted a **70%** speed boost potential across all **M4s**, igniting further interest in hardware upgrades.
- **Maximizing Performance with NVLink and RTX 3090**: Users shared insights on achieving **10 to 25 t/s** with dual **RTX 3090** setups, especially with **NVLink**, while one reported hitting **50 t/s**.
   - Despite these high numbers, the actual inference performance impact of NVLink drew skepticism from some community members.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Reflection 70B Model Struggles on Benchmarks**: Recent tests revealed that **Reflection 70B** underperformed in comparisons with the [BigCodeBench-Hard](https://huggingface.co/mattshumer/Reflection-Llama-3.1-70B), particularly affected by tokenizer and prompt issues.
   - The community expressed concerns over evaluations, leading to uncertainty about the model’s reliability in real-world applications.
- **Community Investigates DeepSeek v2.5 Usability**: Members sought feedback on improvements seen with **DeepSeek v2.5** during coding tasks, encouraging a share of user experiences.
   - This initiative aims to build a collective understanding of the model's effectiveness and contribute to user-driven enhancements.
- **Inquiries on API Usability for Llama 3.1**: There was a discussion about optimal API options for implementing **Llama 3.1 70B**, emphasizing the need for tool call format support.
   - Suggestions included exploring various platforms, pointing toward **Groq** as a promising candidate for deployment.
- **Challenges with Quantization Techniques**: Users reported setbacks with the **FP16** quantization of the 70B model, highlighting struggles in achieving satisfactory performance with **int4**.
   - Ongoing discussions revolved around potential solutions to enhance model performance while maintaining quality integrity.
- **MCTS and PRM Techniques for Enhanced Performance**: Conversations indicated interest in merging **MCTS** (Monte Carlo Tree Search) and **PRM** (Probabilistic Roadmap) to boost training efficiencies.
   - The community showed enthusiasm about experimenting with these methodologies for improving model evaluation processes.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **OpenAI Considers $2000 Subscription**: OpenAI is exploring a pricing model at **$2000/month** for its premium AI models, including the upcoming Orion model, stirring accessibility concerns within the community.
   - As discussions unfold, opinions vary on whether this pricing aligns with market norms or poses barriers for smaller developers.
- **Reflection 70B's Mixed Benchmark Results**: The **Reflection 70B** model has shown mixed performance, scoring **20.3** on the **BigCodeBench-Hard** benchmark, notably lower than **Llama3's** score of **28.4**.
   - Critics emphasize the need for deeper analysis of its methodology, especially regarding its claim of being the top open-source model.
- **Speculative Decoding Boosts Inference**: Together AI reported that speculative decoding can enhance throughput by up to **2x**, challenging previous assumptions about its efficiency in high-latency scenarios.
   - This advancement could reshape approaches to optimizing inference speeds for long context inputs.
- **Exciting Developments in Text-to-Music Models**: A new open-source **text-to-music model** has emerged, claiming impressive sound quality and efficiency, competing against established platforms like **Suno.ai**.
   - Members are keen on its potential applications, although there are varied opinions regarding its practical usability.
- **Exploration of AI Code Editors**: Discussion on AI code editors highlights tools like [Melty](https://github.com/meltylabs/melty) and [Pear AI](https://github.com/trypear/pearai-app), showcasing unique features compared to Cursor.
   - Members are particularly interested in how these tools manage comments and TODOs, pushing for better collaboration in coding environments.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Perplexity steals spotlight**: Users praised **Perplexity** for its speed and reliability, often considering it a better alternative to **ChatGPT Plus** subscriptions.
   - One user noted it is particularly useful for school as it is accessible and integrated with **Arc browser**.
- **RunwayML faces backlash**: A user reported dissatisfaction with **RunwayML** after a canceled community meetup, which raises concerns about their customer service.
   - Comments highlighted the discontent among loyal members and how this affects Runway's reputation.
- **Reflection model's promising tweaks**: Discussion around the **Reflection Llama-3.1 70B model** focused on its performance and a new training method called **Reflection-Tuning**.
   - Users noted that initial testing issues led to a platform link where they can experiment with the model.
- **OpenAI token giveaway generates buzz**: An offer for **OpenAI tokens** sparked significant interest, as one user had **1,000 tokens** they did not plan to use.
   - This prompted discussions around potential trading or utilizing these tokens within the community.
- **Effective tool call integrations**: Members shared tips on structuring **tool calls** in prompts, emphasizing the correct sequence of the **Assistant message** followed by the **Tool message**.
   - One member noted finding success with over **ten Python tool calls** in a single prompt output.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Securing Academic Lab Roles**: Members discussed strategies for obtaining positions in academic labs, emphasizing the effectiveness of **project proposals** and the lower success of cold emailing.
   - One member highlighted the need to align research projects with current trends to grab the attention of potential hosts.
- **Universal Transformers Face Feasibility Issues**: The feasibility of *Universal Transformers* was debated, with some members expressing skepticism while others found potential in adaptive implicit compute techniques.
   - Despite the promise, stability continues to be a **significant barrier** for wide adoption in practical applications.
- **AdEMAMix Optimizer Improves Gradient Handling**: The newly proposed **AdEMAMix** optimizer enhances gradient utilization by blending two Exponential Moving Averages, showing better performance in tasks like language modeling.
   - Early experiments indicate this approach outperforms the traditional single EMA method, promising more efficient training outcomes.
- **Automated Reinforcement Learning Agent Architecture**: A new automated RL agent architecture was introduced, efficiently managing experiment progress and building curricula through a **Vision-Language Model**.
   - This marks one of the first complete automations in reinforcement learning experiment workflows, breaking new ground in model training efficiency.
- **Hugging Face RoPE Compatibility Concerns**: A member raised questions regarding compatibility between the **Hugging Face RoPE implementation** for **GPTNeoX** and other models, noting over **95%** discrepancies in attention outputs.
   - This raises important considerations for those working with multiple frameworks and might influence future integration efforts.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Open Interpreter celebrates a milestone**: Members enthusiastically celebrated the birthday of Open Interpreter, with a strong community sentiment expressing appreciation for its innovative potential.
   - *Happy Birthday, Open Interpreter!* became the chant, emphasizing the excitement felt around its capabilities.
- **Skills functionality in Open Interpreter is still experimental**: Discussion revealed that the skills feature is currently experimental, prompting questions about whether these skills persist across sessions.
   - Users noted that skills appear to be temporary, which led to suggestions to investigate the storage location on local machines.
- **Positive feedback on 01 app performance**: Users shared enthusiastic feedback about the 01 app's ability to efficiently search and play songs from a library of 2,000 audio files.
   - Despite praise, there were reports of **inconsistencies** in results, reflecting typical early access challenges.
- **Fulcra app expands to new territories**: The Fulcra app has officially launched in several more regions, responding to community requests for improved accessibility.
   - Discussions indicated user interest in availability across locations such as ***Australia***, rallying support for further expansion.
- **Request for Beta Role Access**: Multiple users are eager to get access to the **beta role for desktop**, including one who contributed to the dev kit for Open Interpreter 01.
   - A user expressed their disappointment at missing a live session, asking, *'Any way to get access to the beta role for desktop?'*



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo Values Page Returns 404**: Members noted that Modular's values page is currently showing a **404 error** at [this link](https://www.modular.com/values) and may need redirection to [company culture](https://www.modular.com/company/culture).
   - *Clarifications suggested* that changes were required for the link to effectively point users to the relevant content.
- **Async Functions Limitations in Mojo**: A user faced issues using `async fn` and `async def`, revealing these async features are exclusive to nightly builds, causing confusion in stable versions.
   - Users were advised to check their version and consider switching to the nightly build to access these features.
- **DType Constraints as Dict Keys**: Discussion sparked over the inability to use `DType` as a key in Dictionaries, raising eyebrows since it implements the `KeyElement` trait.
   - Participants explored the design constraints within Mojo’s data structures that might limit the use of certain types.
- **Constructor Usage Troubleshoot**: Progress was shared on resolving constructor issues involving `Arc[T, True]` and `Weak[T]`, highlighting challenges with @parameter guards.
   - Suggestions included improving naming conventions within the standard library for better clarity and aligning structure of types.
- **Exploring MLIR and IR Generation**: Interest was piqued on how MLIR can be utilized more effectively in Mojo, especially regarding IR generation.
   - A resource from a previous LLVM meeting was suggested, [2023 LLVM Dev Mtg - Mojo 🔥](https://www.youtube.com/watch?v=SEwTjZvy8vw), to gain deeper insights on integration.



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **Reflection 70B launches with exciting features**: The **Reflection 70B** model has been launched as the world’s best open-source model, utilizing [Reflection-Tuning](https://x.com/mattshumer_/status/1831767014341538166) to correct LLM errors.
   - A **405B model** is expected next week, possibly surpassing all current models in performance.
- **Investigating TorchDynamo cache lookup delays**: When executing large models, members noted **600us** spent in **TorchDynamo Cache Lookup**, mainly due to calls from `torch/nn/modules/container.py`.
   - This points to potential optimizations required in the cache lookup process to improve model training runtime.
- **NVIDIA teams up for generative AI education**: The **Deep Learning Institute** from NVIDIA released a [generative AI teaching kit](https://www.hackster.io/news/nvidia-teams-up-with-dartmouth-for-a-free-generative-ai-teaching-kit-11358047a05a) in collaboration with Dartmouth College to enhance GPU learning.
   - Participants will gain a competitive edge in AI applications, bridging essential knowledge gaps.
- **FP16 x INT8 Matmul shows limits on batch sizes**: The **FP16 x INT8 matmul** on the **4090 RTX** fails when batch sizes exceed 1 due to shared memory limitations, hinting at a need for better tuning for non-A100 GPUs.
   - Users experienced substantial slowdowns with enabled inductor flags yet could bypass errors by switching them off.
- **Liger's performance benchmarks raise eyebrows**: The performance of **Liger's swiglu kernels** was contrasted against [Together AI's benchmarks](https://www.together.ai/blog/nvidia-h200-and-h100-gpu-cluster-performance-together-kernel-collection), which reportedly offer up to **24% speedup**.
   - Their specialized kernels outperform **cuBLAS** and **PyTorch eager mode** by **22-24%**, indicating the need for further tuning options.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Reflection Llama-3.1 70B yields mixed performance**: The newly released **Reflection Llama-3.1 70B** claims to be the leading open-source model yet struggles significantly on benchmarks like **BigCodeBench-Hard**.
   - Users observed a drop in performance for reasoning tasks and described the model as a 'non news item meh model' on Twitter.
- **Concerns linger over Glaive's synthetic data**: Community members raised alarms about the effectiveness of synthetic data from **Glaive**, recalling issues from past contaminations that might impact model performance.
   - These concerns led to discussions about the implications of synthetic data on the **Reflection Llama** model's generalization capabilities.
- **HuggingFace Numina praised for research**: **HuggingFace Numina** was highlighted as a powerful resource for data-centric tasks, unleashing excitement among researchers for its application potential.
   - Users expressed enthusiasm about how it could enhance efficiency and innovation in various ongoing projects.
- **Introduction of CHAMP benchmark for math reasoning**: The community welcomed the new **CHAMP** benchmark aimed at assessing LLMs' mathematical reasoning abilities through annotated problems that provide hints.
   - This dataset will explore how additional context aids in problem-solving under complex conditions, promoting further study in this area.
- **Reliability issues of Fireworks and Together**: Discussions unveiled that both **Fireworks** and **Together** are viewed as less than **100% reliable**, prompting the implementation of **failovers** to maintain functionality.
   - Users are cautious about utilizing these tools until assurances of reliability are fortified.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Tech Entry Without Skills**: A member expressed eagerness to enter the tech industry without technical skills, seeking advice on building a compelling CV and effective networking.
   - Another member mentioned starting **cybersecurity training** through PerScholas, underscoring a growing interest in **coding and AI**.
- **Bing Copilot vs. Perplexity AI**: A user compared Bing Copilot's ability to provide **5 sources** with inline images to Perplexity's capabilities, suggesting improvements.
   - They hinted that integrating **hover preview cards** for citations could be a valuable enhancement for Perplexity.
- **Perplexity AI's Referral Program**: Perplexity is rolling out a **merch referral program** specifically targeted at students, encouraging sharing for rewards.
   - A question arose about the availability of a year of free access, particularly for the first **500 sign-ups**.
- **Web3 Job Openings**: A post highlighted **job openings** in a Web3 innovation team, looking for beta testers, developers, and UI/UX designers.
   - They invite applications and proposals to create mutual cooperation opportunities as part of their vision.
- **Sutskever's SSI Secures $1B**: **Sutskever's SSI** successfully raised **$1 billion** to boost advancements in AI technology.
   - This funding aims to fuel further **innovations in the AI sector**.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Bounty Exploration Sparks Interest**: A user expressed interest in trying out a bounty and sought guidance, referencing a resource on asking [smart questions](http://www.catb.org/~esr/faqs/smart-questions.html).
   - This led to a humorous acknowledgment from another member, highlighting community engagement in bounty discussions.
- **Tinygrad Pricing Hits Zero**: In a surprising twist, georgehotz confirmed the pricing for a **4090 + 500GB** plan has been dropped to **$0**, but only for tinygrad friends.
   - This prompted r5q0 to inquire about the criteria for friendship, adding a light-hearted element to the conversation.
- **Clarifying PHI Operation Confusion**: Members discussed the PHI operation's functionality in IR, noting its unusual placement compared to LLVM IR, especially in loops.
   - One member suggested renaming it to ASSIGN as it operates differently from traditional phi nodes, aiming to clear up misunderstandings.
- **Understanding MultiLazyBuffer's Features**: A user raised concerns about the `MultiLazyBuffer.real` property and its role in shrinking and copying to device interactions.
   - This inquiry led to discussions revealing that it signifies real lazy buffers on devices and potential bugs in configurations.
- **Views and Memory Challenges**: Members expressed ongoing confusion regarding the realization of views in the `_recurse_lb` function, questioning optimization and utilization balance.
   - This reflection underscores the need for clarity on foundational tensor view concepts, inviting community input to refine understanding.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Gemma 2 model resources shared**: Members discussed the [Gemma 2 model card](https://huggingface.co/google/gemma-2-9b), providing links to technical documentation from **Google's** lightweight model family.
   - Resources included a [Responsible Generative AI Toolkit](https://ai.google.dev/responsible) and links to [Kaggle](https://www.kaggle.com/models/google/gemma-2) and [Vertex Model Garden](https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/335), emphasizing ethical AI practices.
- **Multimodal models and causal masks**: A member outlined challenges with **causal masks** during inference for multimodal setups, focusing on fixed sequence lengths.
   - They noted that *exposing these variables through attention layers* is crucial to tackle this issue effectively.
- **Expecting speedups with Flex Attention**: There is optimism that **flex attention with document masking** will significantly enhance performance, achieving **40% speedup on A100 and 70% on 4090**.
   - This would improve **dynamic sequence length** training while minimizing padding inefficiencies.
- **Questions arise on TransformerDecoder design**: A member asked whether a **TransformerDecoder** could operate without self-attention layers, challenging its traditional structure.
   - Another pointed out that *the original transformer utilized* both cross and self-attention, complicating this deviation.
- **PR updates signal generation overhaul**: Members confirmed that GitHub PR **#1449** has been updated to enhance compatibility with `encoder_max_seq_len` and `encoder_mask`, with testing still pending.
   - This update paves the way for further modifications to **generation utils** and integration with **PPO**.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Llama-deploy Offers Microservices Magic**: The new **llama-deploy** system enhances deployment for microservices based on [LlamaIndex Workflows](https://twitter.com/llama_index/status/1831794126511337880). This opens up opportunities to streamline agentic systems similar to previous iterations of llama-agents.
   - An example shared in the community demonstrates full-stack capabilities using **llama-deploy** with **@getreflex**, showcasing how to effectively build agentic chat systems.
- **PandasQueryEngine Faces Column Name Confusion**: Users reported that **PandasQueryEngine** struggles to correctly identify the column `averageRating`, often reverting to incorrect labels during chats. Suggestions included verifying mappings within the chat engine's context.
   - This confusion could lead to deeper issues in data integrity when integrating engine responses with expected output formats.
- **Developing Customer Support Bots with RAG**: A user is exploring ways to create a customer support chatbot that efficiently integrates a conversation engine with retrieval-augmented generation (RAG). Members emphasized the synergy between chat and query engines for stronger data retrieval capabilities.
   - Validating this integration could enhance user experience in real-world applications where effective support is crucial.
- **NeptuneDatabaseGraphStore Bug Reported**: Concerns arose regarding a bug in **NeptuneDatabaseGraphStore.get_schema()** that misses date information in graph summaries. It is suspected the issue may be related to schema parsing errors with LLMs.
   - Community members expressed the need for further investigation, especially surrounding the `datetime` package’s role in the malfunction.
- **Azure LlamaIndex and Cohere Reranker Inquiry**: A discussion emerged about integrating the Cohere reranker as a postprocessor within Azure's **LlamaIndex**. Members confirmed that while no Azure module exists currently, creating one is feasible due to straightforward documentation.
   - The community is encouraged to consider building this integration as it could significantly enhance processing capabilities within Azure environments.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Reflection Llama-3.1: Top LLM Redefined**: [Reflection Llama-3.1 70B](https://huggingface.co/mattshumer/Reflection-Llama-3.1-70B) is now acclaimed as the leading open-source LLM, enhanced through **Reflection-Tuning** for improved reasoning accuracy.
   - This model was trained on synthetic data generated by [Glaive](https://glaive.ai) and can be further explored at [this link](https://reflection-playground-production.up.railway.app/).
- **Synthetic Dataset Generation for Fast Results**: Discussion focused on the rapid generation of the synthetic dataset for Reflection Llama-3.1, sparking curiosity about **human rater** involvement and sample size.
   - Members debated the balance between speed and quality in synthetic dataset creation.
- **Challenge Accepted: Fine-tuning Llama 3.1**: Members raised queries regarding effective **fine-tuning** techniques for **Llama 3.1**, noting its performance boost at **8k sequence length** with possible extension to **128k** using **rope scaling**.
   - Concerns about fine-tuning complexities arose, suggesting the need for custom token strategies for optimal performance.
- **SmileyLlama is Here: Meet the Chemical Language Model**: [SmileyLlama](https://x.com/axolotl_ai/status/1831771214445945148) stands out as a fine-tuned **Chemical Language Model** designed for molecule creation based on specified properties.
   - This model, marked as an **SFT+DPO** implementation, showcases **Axolotl's** prowess in specialized model adaptations.
- **GPU Power: Lora Finetuning Insights**: Inquiries about **A100 80 GB GPUs** for fine-tuning **Meta-Llama-3.1-405B-BNB-NF4-BF16** in **4 bit** using **adamw_bnb_8bit**, underscored the resource requirements for effective **Lora finetuning**.
   - This points to practical considerations essential for managing Lora finetuning processes efficiently.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Explore Cohere's Capabilities and Cookbooks**: Members discussed checking out the channel dedicated to [capabilities and demos](https://discord.com/channels/1218409701339828245) where the community shares projects built using Cohere models, referencing a comprehensive [cookbook](https://docs.cohere.com/page/cookbooks) that provides ready-made guides.
   - One member highlighted that these cookbooks showcase best practices for leveraging Cohere's generative AI platform.
- **Understanding Token Usage with Anthropic Library**: A member inquired about using the Anthropic library, sharing a code snippet for calculating token usage: `message = client.messages.create(...)`.
   - They directed others to the [GitHub repository](https://github.com/anthropics/anthropic-sdk-python) for the Anthropic SDK to further explore tokenization.
- **Embed-Multilingual-Light-V3.0 Availability on Azure**: A member questioned the availability of `embed-multilingual-light-v3.0` on Azure and asked if there are any plans to support it.
   - This inquiry reflects ongoing interest in the integration of Cohere's resources with popular cloud platforms.
- **Query on RAG Citations**: A member asked how citations will affect the content of text files when using **RAG** with an external knowledge base, specifically inquiring about receiving citations when they are currently getting **None**.
   - They expressed urgency in figuring out how to resolve the issue regarding the absence of citations in the responses from text files.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Chroma DB Setup Simplified**: A member pointed out that launching a server for **Chroma DB** requires just one line of code: `!chroma run --host localhost --port 8000 --path ./ChomaM/my_chroma_db1`, noting the ease of setup.
   - *They felt relieved knowing the database location with such simplicity.*
- **Weaviate Setup Inquiry**: The same member asked if there’s a simple setup for **Weaviate** similar to Chroma DB, avoiding **Go Docker** complexities.
   - *They expressed a need for ease due to their non-technical background.*
- **Jupyter Notebooks for Server-Client Communication**: Another member shared their use of **two Jupyter notebooks** to run a server and client separately, highlighting it fits their needs.
   - *They identify as a Biologist and seek uncomplicated solutions.*
- **Reflection 70B Takes the Crown**: **Reflection 70B** has been announced as the leading open-source model, featuring **Reflection-Tuning** to enable the model to rectify its own errors.
   - *A new model, **405B**, is on its way next week promising even better performance.*
- **Enhancing LLM Routing with Pricing**: Discussion emerged around routing appropriate LLMs based on queries, intending to incorporate aspects like **pricing** and **TPU speed** into the logic.
   - *Participants noted that while routing LLMs is clear-cut, enhancing it with performance metrics can refine the selection process.*



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **SwarmUI Usability Concerns**: Members expressed discomfort with user interfaces showcasing **100 nodes** compared to **SwarmUI**, reinforcing its usability issues.
   - Discussion highlighted how labeling it as 'literally SwarmUI' reflected a broader concern about UI complexity among tools.
- **SwarmUI Modular Design on GitHub**: A link to [SwarmUI on GitHub](https://github.com/mcmonkeyprojects/SwarmUI) was shared, featuring its focus on modular design for better accessibility and performance.
   - The repository emphasizes offering easy access to powertools, enhancing usability through a well-structured interface.
- **Reflection 70B Debut as Open-Source Leader**: The launch of **Reflection 70B** has been announced as the premier open-source model using **Reflection-Tuning**, enabling LLMs to self-correct.
   - A **405B model** is anticipated next week, raising eyebrows about its potential to crush existing benchmark performances.
- **Self-Correcting LLMs Make Waves**: New discussions emerged around an LLM capable of self-correction that reportedly outperforms **GPT-4o** in all benchmarks, including **MMLU**.
   - The open-source nature of this model, surpassing **Llama 3.1's 405B**, signifies a major leap in LLM functionality.
- **Lucidrains Reworks Transfusion Model**: Lucidrains has shared a [GitHub implementation](https://github.com/lucidrains/transfusion-pytorch) of the **Transfusion** model, optimizing next token prediction while diffusing images.
   - Future extensions may integrate **flow matching and audio/video processing**, indicating strong multi-modal capabilities.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **ReAct Agent Deployment Challenges**: A member struggles with deploying their **ReAct agent on GCP** via FastAPI, facing issues with the local SQLite database disappearing upon redeploy. They seek alternatives for **Postgres or MySQL** as a replacement for `SqliteSaver`.
   - The member is willing to share their local implementation for reference, hoping to find a collaborative solution.
- **Clarifying LangChain Callbacks Usage**: Discussion emerged on the accuracy of the syntax `chain = prompt | llm`, referencing [LangChain's callback documentation](https://python.langchain.com/v0.1/docs/modules/callbacks/). Members noted that the documentation appears outdated, particularly with updates in version **0.2**.
   - The conversation underscored the **utility of callbacks** for logging, monitoring, and third-party tool integration.
- **Cerebras and LangChain Collaboration Inquiry**: A member inquired about usage of **Cerebras** alongside **LangChain**, seeking collaborative insights from others. Responses indicated interest but no specific experiences or solutions were shared.
   - This topic remains open for further exploration within the community.
- **Decoding .astream_events Dilemma**: Members discussed the lack of references for decoding streams from **.astream_events()**, with one sharing frustration over manually serializing events. The conversation conveyed a desire for better resources and solutions.
   - The tedious process highlighted the need for collaboration and resource sharing in the community.



---



## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **Enhancing RAG with Limited Hardware**: A member sought strategies to upgrade their **RAG system** using **llama3-8b with 4bit quantization** along with the **BAAI/bge-small-en-v1.5** embedding model while working with a restrictive **4090 GPU**.
   - *Seeking resources for better implementation,* they expressed hardware constraints, highlighting the need for efficient practices.
- **Maximizing GPU Potential with Larger Models**: In response, another member suggested that a **4090** can concurrently run larger embedding models, indicating that the **3.1 version** might also enhance performance.
   - They provided a [GitHub example](https://github.com/milvus-io/pymilvus/blob/master/examples/hello_hybrid_sparse_dense.py) showcasing hybrid search integration involving **bge & bm25** on Milvus.
- **Leveraging Metadata for Better Reranking**: The chat underscored the critical role of **metadata for each chunk**, suggesting it could improve the sorting and filtering of returned results.
   - *Implementing a reranker,* they argued, could significantly enhance the output quality for user searches.



---



## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **XLAM System Prompt Sparks Curiosity**: A member pointed out that the **system prompt for XLAM** is unique compared to other **OSS models** and questioned the rationale behind this design choice.
   - Discussion revealed an interest in whether these differences stem from **functionality** or **licensing considerations**.
- **Testing API Servers Needs Guidance**: A user sought effective methods for testing their own **API server** but received no specific documentation in reply.
   - This gap in shared resources highlights a potential area for growth in community support and knowledge sharing.
- **How to Add Models to the Leaderboard**: A user inquired about the process for adding new models to the **Gorilla leaderboard**, prompting a response with relevant guidelines.
   - Access the contribution details on the [GitHub page](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard#contributing) to understand how to facilitate model inclusion.
- **Gorilla Leaderboard Resource Highlighted**: Members discussed the **Gorilla: Training and Evaluating LLMs for Function Calls** GitHub resource that outlines the leaderboard contributions.
   - An image from its repository was also shared, illustrating the guidelines available for users interested in participation at [GitHub](https://opengraph.githubassets.com/25d4bf4245a01dd99c8e3d1e4b47d26ef3db55d11499f2f9edfa259231aaacd2/ShishirPatil/gorilla).



---



## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord

- **Greetings from Knut09896**: Knut09896 stepped into the channel and said hello, sparking welcome interactions.
   - This simple greeting hints at the ongoing engagement within the **Alignment Lab AI** community.
- **Channel Activity Buzz**: The activity level in the **#general** channel appears vibrant with members casually chatting and introducing themselves.
   - Such interactions play a vital role in fostering community connections and collaborative discussions.



---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Mozilla AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **DiscoResearch Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1281347738113675306)** (1 messages): 

> - `Vision Language Models`
> - `Tau LLM Training Optimization`
> - `African Language Models`
> - `No-Code AI Model Tasks`
> - `Selective Fine-Tuning of Language Models` 


- **Introduction to Vision Language Models**: A member shared a [blogpost](https://www.lightly.ai/post/introduction-to-vision-language-models) on vision language models, providing a concise overview of the subject.
   - This introductory piece aims to illuminate the potentials of combining vision and language in AI applications.
- **Optimizing Tau LLM Training**: The [Tau LLM series](https://youtube.com/live/flwqvE4aSzA?feature=share) focuses on optimizing training processes and enhancing model performance.
   - Hailed as an essential resource, it promises to simplify learning the ins and outs of LLM training.
- **InkubaLM-0.4B Targets African Languages**: The newly released [InkubaLM-0.4B](https://huggingface.co/spaces/Tonic/Inkuba-0.4B) aims to support African languages and expand linguistic representation.
   - Developed specifically for this purpose, it showcases a commitment to inclusivity in AI language models.
- **Shadowbox Offers No-Code Model Task Construction**: Introducing [Shadowbox](https://github.com/darkshapes/singularity), a no-code constructor for AI tasks using FOSS models, simplifying user experiences.
   - Users can create tasks without coding expertise, broadening accessibility to AI solutions.
- **Selective Fine-Tuning with Spectrum**: The concept of selective [fine-tuning](https://huggingface.co/blog/anakin87/spectrum) for language models was discussed, highlighting its benefits.
   - By focusing on certain aspects, finer model performance enhancements can be achieved without comprehensive retraining.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1281336016963375144)** (258 messages🔥🔥): 

> - `Code Generation Evaluations`
> - `Model Training Issues`
> - `Data Handling for Training`
> - `Fine-tuning and Pre-training`
> - `Performance Analysis of Models` 


- **Performance Analysis in Code Generation**: Discussions included analyzing how often functions appear in datasets and whether common functions result in fewer errors, exploring metrics that consider functional correctness.
   - Contributors noted that near-exact clones of functions generated by models might indicate contamination in the training data.
- **Challenges in Model Training Setup**: Members experienced issues related to hardware limitations, with several discussing their struggles using GPU resources effectively for training models.
   - One user inquired about using platforms like Hugging Face for training, expressing concern about inadequate resources on their local setup.
- **Insights on Pre-training and Data Quality**: A paper was shared indicating the impact of including code in pre-training datasets and its benefits on non-code tasks and overall model performance.
   - Participants debated whether excluding code from training sets could lead to less effective model outputs.
- **Generation Scripts and Model Testing**: A minimal script was provided for generating outputs from a specified base model, highlighting potential issues in post-processing results.
   - Users were encouraged to test this script and analyze the generations, despite some concerns about the model's quality based on context length.
- **Reflections on Model Evaluation Metrics**: There was a consensus that static metrics for code generation are not ideal, with discussions emphasizing the importance of semantic correctness and functional output.
   - Participants reflected on how certain metrics, including edit distance, correlate with model performance and reliability.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://stackoverflow.com/help/how-to-ask">How do I ask a good question? - Help Center</a>: Stack Overflow | The World&#x2019;s Largest Online Community for Developers</li><li><a href="https://huggingface.co/spaces/Vipitis/shadermatch">ShaderMatch - a Hugging Face Space by Vipitis</a>: no description found</li><li><a href="https://arxiv.org/abs/2306.03203">A Static Evaluation of Code Completion by Large Language Models</a>: Large language models trained on code have shown great potential to increase productivity of software developers. Several execution-based benchmarks have been proposed to evaluate functional correctne...</li><li><a href="https://huggingface.co/datasets/nroggendorff/multi-csv">nroggendorff/multi-csv · Datasets at Hugging Face</a>: no description found</li><li><a href="https://arxiv.org/abs/2408.10914">To Code, or Not To Code? Exploring Impact of Code in Pre-training</a>: Including code in the pre-training data mixture, even for models not specifically designed for code, has become a common practice in LLMs pre-training. While there has been anecdotal consensus among p...</li><li><a href="https://huggingface.co/docs/diffusers/tutorials/basic_training#create-a-unet2dmodel">Train a diffusion model</a>: no description found</li><li><a href="https://openrouter.ai/models/mattshumer/reflection-70b">Reflection 70B - API, Providers, Stats</a>: Reflection Llama-3.1 70B is trained with a new technique called Reflection-Tuning that teaches a LLM to detect mistakes in its reasoning and correct course. Run Reflection 70B with API
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1281362437861412894)** (8 messages🔥): 

> - `Understanding Attention Mechanism in Transformers`
> - `Discussions on Cross-posting`
> - `Using AI for Tutoring Kids`
> - `Creating a Python Microservice with Ollama` 


- **Seeking clarity on the Attention Mechanism**: A member asked about how to represent attention for a given token in transformers, specifically if it relates to the distance in latent vector space between tokens.
   - They requested materials to aid in understanding this concept better, indicating a need for further explanation.
- **Reminder on Cross-posting Etiquette**: Multiple members discussed the issue of cross-posting questions in the channel, with one requesting to stop sending the same message across different channels.
   - *One member preferred to follow suggestions from other channels* over the given advice, prompting another to state that one channel is sufficient.
- **AI Tutoring for Kids without Bootcamp Approach**: One member shared a learning experience regarding how to tutor kids on AI without the pressures of a formal bootcamp.
   - This approach suggests a more engaging and less structured way to introduce children to AI concepts.
- **Developing a Python Microservice with Ollama**: A member inquired about creating a Python microservice using Ollama that can paraphrase sentences in ten different ways.
   - This request indicates an interest in practical applications of AI in text manipulation tasks.


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1281447142539661333)** (2 messages): 

> - `Elasticsearch`
> - `Vespa Search Engine` 


- **Goodbye Elasticsearch, Hello Vespa Search Engine**: A member announced their transition from **Elasticsearch** to **Vespa Search Engine** in a tweet, creating some buzz.
   - They included an emoji to express excitement: *'👀'* indicating positive anticipation for the change.
- **Discussion on Search Engine Technologies**: The shift from **Elasticsearch** to **Vespa** sparked a conversation about different search engine technologies and their advantages.
   - Participants expressed curiosity about the performance and features of **Vespa** compared to traditional solutions.



**Link mentioned**: <a href="https://x.com/jobergum/status/1831701040812450156">Tweet from Jo Kristian Bergum (@jobergum)</a>: Goodbye Elasticsearch, Hello Vespa Search Engine 👀

  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1281386617055215648)** (14 messages🔥): 

> - `Pro-Pretorian Computer Vision System`
> - `Interactive Model Comparator`
> - `Chess Puzzle Visualization`
> - `Tau LLM Series Update` 


- **Pro-Pretorian Computer Vision System Launch**: A member shared their completed first iteration of the **Pro-Pretorian Computer Vision System**, a Next.js app hosted on Azure with data persistence also on Azure, utilizing **tfjs** for inference via **WebGL**.
   - They plan to enhance the system by adding fine-tuned models and creating a pipeline through their [Hugging Face account](https://github.com/salim4n/pro-pretorian-system) for automation.
- **Interactive Model Comparator Introduced**: Another member presented the **Interactive Model Comparator**, a web tool designed for visually comparing output images of different machine learning models for computer vision tasks.
   - The tool allows users to load images, switch between models, and preview comparisons in real-time, making it a valuable resource for researchers and developers, available on [GitHub](https://github.com/kawchar85/InteractiveModelComparator).
- **Visualizing 4 Million Chess Puzzles**: A project was highlighted where **Hugging Face datasets** were leveraged to visualize **4 million chess puzzles**, with evaluations provided by Stockfish, detailing over **83 million** chess positions.
   - Key details include data formats and a [link to the Lichess database](https://database.lichess.org/#puzzles) for further exploration of chess evaluations.
- **Exciting Updates in Tau LLM Series**: **Episode 15** of the **Tau LLM series** introduced various updates including automated **data file de-duplication** and a new **ophrase Python module** for generating paraphrases, enhancing dataset diversity.
   - The episode promises the generation of new embeddings and a shift toward training an expanded dataset, aimed to bring efficiency and reduce entropy, shared via a [YouTube link](https://youtube.com/live/dOh_FEs12e4?feature=share).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/christopher/status/1832040993522147717">Tweet from Christopher Akiki (@christopher)</a>: 4 million @lichess chess puzzles</li><li><a href="https://huggingface.co/spaces/Muinez/Image-scorer">Image Scorer - a Hugging Face Space by Muinez</a>: no description found</li><li><a href="https://github.com/kawchar85/InteractiveModelComparator">GitHub - kawchar85/InteractiveModelComparator: A web-based tool designed for comparing output images from different machine learning models on the same dataset.</a>: A web-based tool designed for comparing output images from different machine learning models on the same dataset. - kawchar85/InteractiveModelComparator</li><li><a href="https://youtube.com/live/dOh_FEs12e4?feature=share">Unity ML-Agents | Pretrain an LLM from Scratch with Sentence Transformers | Part 15</a>: **Welcome back to our Tau LLM series! 🌟**In this episode, we&#39;re taking our project to the next level with some exciting new developments. Our highlights inc...</li><li><a href="https://github.com/salim4n/pro-pretorian-system">GitHub - salim4n/pro-pretorian-system: This project is a Computer Vision application that allows users to customize their detection parameters according to their needs. Whether you want to detect specific objects, define the area of interest, or schedule detection times, this application provides flexible and powerful options.</a>: This project is a Computer Vision application that allows users to customize their detection parameters according to their needs. Whether you want to detect specific objects, define the area of int...</li><li><a href="https://database.lichess.org/#puzzles">lichess.org open database</a>: no description found</li><li><a href="https://database.lichess.org/#">lichess.org open database</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/)** (1 messages): 

noaroggendorff: <@&1078351789843292311>
  

---


### **HuggingFace ▷ #[core-announcements](https://discord.com/channels/879548962464493619/1014557141132132392/1281597955023048794)** (1 messages): 

> - `Optimizing Flux and Cog`
> - `Diffusion models`
> - `TorchAO` 


- **New Recipe Repo Released for Optimization**: A new [GitHub repository](https://github.com/sayakpaul/diffusers-torchao) has been released showcasing how to optimize **Flux** and **Cog** using **diffusers** and **torchao**, including both inference and FP8 training.
   - This repo allows running **Cog** in just **3.1GB** memory with **quantization** and various offloading methods.
- **End-to-End Optimization for Diffusion Models**: The repository provides comprehensive recipes aimed at optimizing **diffusion models**, making them more efficient in training and inference.
   - It highlights techniques such as **offloading** and **quantization**, crucial for handling large model requirements.



**Link mentioned**: <a href="https://github.com/sayakpaul/diffusers-torchao">GitHub - sayakpaul/diffusers-torchao: End-to-end recipes for optimizing diffusion models with torchao and diffusers (inference and FP8 training).</a>: End-to-end recipes for optimizing diffusion models with torchao and diffusers (inference and FP8 training). - sayakpaul/diffusers-torchao

  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1281476646783815700)** (2 messages): 

> - `Medical dataset for disease detection`
> - `Training Nougat and Donut` 


- **Searching for medical datasets in CV**: A member is looking for a good medical dataset for **computer vision**, aiming at **disease detection** or potentially for larger scale data generation using transformers.
   - They expressed interest in datasets that could facilitate substantial **data generation** efforts.
- **Training methods for Nougat and Donut**: Another member inquired about anyone familiar with the specifics of **training Nougat** or **Donut** models.
   - This could indicate a desire for insights on model architectures or training techniques relevant to these frameworks.


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1281447213125599292)** (4 messages): 

> - `OOM errors during evaluation`
> - `DeepSpeed configuration for evaluation`
> - `Custom Dataset for evaluation`
> - `GPU distribution techniques` 


- **OOM Errors Plague Evaluation Phase**: Team encountered **OOM errors** during evaluation while using a custom setup with DeepSpeed, despite successful training on multiple GPUs.
   - It was noted that smaller batches (<10 examples) evaluated fine, while larger batches (>100 examples) triggered the errors, leading to questions about GPU loading.
- **Custom Dataset Recommended for Evaluation**: A member advised to utilize a **custom Dataset** yielding specific batch sizes to mitigate OOM errors, suggesting starting evaluations with **50 examples** as a test.
   - They referred to the [PyTorch Dataset tutorial](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) for guidance on implementing this.
- **Implementing Multi-GPU Distribution**: There's a recommendation for using a custom evaluation loop to load data onto specific GPUs, facilitating distribution across multiple GPUs.
   - Using methods like `data.to('cuda:1')` for loading onto individual GPUs was suggested to directly tackle OOM issues.
- **Custom Evaluation Loop for Smaller Batches**: Nympheliaa confirmed using a custom dataset and inquired about creating a **custom evaluation loop** with smaller batches for **GPU distribution**.
   - They expressed intent to utilize techniques like **torch DataParallel** or **DistributedDataParallel** to better manage GPU resources.


  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1281447186630180905)** (10 messages🔥): 

> - `Flux img2img Pipeline`
> - `SD3 vs. SDXL models`
> - `ControlNets for SDXL`
> - `Auto Class Recommendations`
> - `Memory Optimizations` 


- **Flux img2img Pipeline not merged yet**: A member noted that the **Flux img2img** feature is not merged and referenced an [open PR for it](https://huggingface.co/docs/diffusers/main/en/api/pipelines/flux). Another member confirmed the documentation contains information about Flux, including links to its [blog post](https://blackforestlabs.ai/announcing-black-forest-labs/).
   - *Flux can be expensive to run* on consumer hardware, but optimizations are possible, as discussed in a [related blog post](https://huggingface.co/blog/sd3#memory-optimizations-for-sd3).
- **Exploring Img2Img Pipeline Alternatives**: When asked for alternatives to the **Flux img2img Pipeline**, a member suggested using the **SD3** model for generic cases and **SDXL** for higher quality images involving humans. They also emphasized exploring **ControlNets** for enhanced functionality.
   - Another member inquired about popular **ControlNets** for SDXL, and the response included suggestions like **ControlnetUnion** and **Mistoline**.
- **Clarifying Usage of the Auto Class**: A user asked whether they should simply use the **Auto class** for Img2Img alternatives while starting with SD. The conversation pivoted to model preferences for higher quality outputs, particularly involving human images.
- **Documentation Discrepancies**: There was a discussion regarding discrepancies in the **documentation** which mentioned a feature that isn't merged yet. The clarification was made that **using the main branch** references features that may not yet be fully integrated.



**Link mentioned**: <a href="https://huggingface.co/docs/diffusers/main/en/api/pipelines/flux">Flux</a>: no description found

  

---



### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1281331684104736862)** (274 messages🔥🔥): 

> - `ControlNet and Model Usage`
> - `Flux vs. SDXL for Image Generation`
> - `Scams and Online Safety`
> - `Tagging and Workflow in ComfyUI`
> - `Integration of Extensions in Forge` 


- **ControlNet Guidance and Model Pairing**: Users discussed how to effectively use ControlNet, with specific emphasis on applying it to create accurate representations like hash rosin images using Loras along with various SDXL models.
   - Recommendations for models included 'Flux' and specifics about how to integrate techniques like depth maps were mentioned to help achieve desired outcomes.
- **Choosing Between Flux and SDXL for Logos**: Flux was recommended over SDXL for generating logos, as it handles logos exceptionally well and allows for easy prompting without needing significant training.
   - Conversely, users shared the difficulties employing SDXL for logos due to a lack of familiarity with the logo, thus advocating for Flux's capabilities.
- **Online Safety and Scams Discussion**: Members shared anecdotes about online scams and stressed the importance of vigilance, recalling how even experienced individuals can fall victim during vulnerable moments.
   - Empathy was highlighted as a crucial approach to understanding the behaviors that lead to scams, indicating that scams are not exclusive to naive individuals.
- **Tagging Techniques and Tools in ComfyUI**: The conversation included using ComfyUI for tagging, likening the interface's functionality to Langflow and Flowise, which cater to LLM models.
   - Community members discussed specific workflows in ComfyUI and adaptations made to enhance tagging effectiveness, emphasizing the flexibility it offers.
- **Forge Extensions and Community Contributions**: Users inquired about various extensions available in Forge, including those for utilizing ControlNet, and how these contribute to improving user experiences.
   - A mention was made regarding community polls and their impact, suggesting that input could influence future releases, underlining the need for quality assurance.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/h94/IP-Adapter">h94/IP-Adapter · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/p1atdev/wd-swinv2-tagger-v3-hf">p1atdev/wd-swinv2-tagger-v3-hf · Hugging Face</a>: no description found</li><li><a href="https://viralshort.ai/">Viralshort - TikTok &amp; Youtube Shorts made easy!</a>: The fastest way to automatically generate short-form content for TikTok, Instagram and Youtube Shorts using AI in seconds!</li><li><a href="https://civitai.green">Civitai: The Home of Open-Source Generative AI</a>: Explore thousands of high-quality Stable Diffusion models, share your AI-generated art, and engage with a vibrant community of creators</li><li><a href="https://huggingface.co/lllyasviel/ControlNet-v1-1/tree/main">lllyasviel/ControlNet-v1-1 at main</a>: no description found</li><li><a href="https://huggingface.co/lllyasviel/sd_control_collection/tree/main">lllyasviel/sd_control_collection at main</a>: no description found</li><li><a href="https://civitai.com/models/487689/hash-rosin">Hash Rosin - v1.0 | Stable Diffusion LoRA | Civitai</a>: This Lora can recreate close up macro shots of Hash Rosin in jars and on Dabbers. it is also flexible enough to make things out of Rosin like anima...
</li>
</ul>

</div>
  

---



### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1281332257029623874)** (189 messages🔥🔥): 

> - `Congratulations on Y Combinator backing`
> - `Unsloth AI functionality and support`
> - `Models for synthetic data generation`
> - `Reflection model performance`
> - `Hardware requirements for Unsloth` 


- **Congratulations to the team on YC backing**: Members congratulated the team on their selection for Y Combinator, expressing excitement and support for their journey.
   - The team reciprocated the gratitude and acknowledged the importance of community support.
- **Unsloth's hardware compatibility in question**: Discussions arose about Unsloth's compatibility with Mac systems, specifically in relation to CUDA support for GPU tasks.
   - The team clarified they aim for hardware agnosticism but current limitations affect performance on certain setups.
- **Recommendations for models in synthetic data generation**: Kearm shared insights on using Mistral 8x7B tunes for synthetic data, while other models were also suggested, including jondurbin/airoboros-34b-3.3.
   - Members discussed experimenting with these models for optimal results based on specific hardware limitations.
- **Reflection model performance concerns**: Members expressed mixed opinions about Matt Shumer's Reflection model, noting it has not performed well on private logic questions compared to other models like Claude 3.5 and GPT-4.
   - There is ongoing skepticism regarding the model's capabilities and claims of being a top open-source LLM.
- **Porting challenges for Mac users**: Members discussed the need to port Unsloth functionalities like bitsandbytes and Triton for Mac users, highlighting the lack of CUDA support on Mac chips.
   - The conversation emphasized the challenges of justifying high expenditures on hardware while attempting to optimize software compatibility.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/unclemusclez/ollamafy">Ollamafy (Work in Progress) - a Hugging Face Space by unclemusclez</a>: no description found</li><li><a href="https://huggingface.co/unsloth/Reflection-Llama-3.1-70B-bnb-4bit">unsloth/Reflection-Llama-3.1-70B-bnb-4bit · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/mlabonne/Hermes-3-Llama-3.1-8B-lorablated">mlabonne/Hermes-3-Llama-3.1-8B-lorablated · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/mattshumer/Reflection-70B">mattshumer/Reflection-Llama-3.1-70B · Hugging Face</a>: no description found</li><li><a href="https://tenor.com/view/republica-de-fifidonia-rick-idk-fake-it-looks-fake-gif-17266845">Republica De Fifidonia Rick GIF - Republica De Fifidonia Rick Idk - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://ollama.com/unsloth/unsloth-tutorial">unsloth/unsloth-tutorial</a>: Get up and running with large language models.</li><li><a href="https://github.com/FailSpy/abliterator">GitHub - FailSpy/abliterator: Simple Python library/structure to ablate features in LLMs which are supported by TransformerLens</a>: Simple Python library/structure to ablate features in LLMs which are supported by TransformerLens - FailSpy/abliterator</li><li><a href="https://ollama.com/unclemusclez/smollm-135m-instruct-devinator">unclemusclez/smollm-135m-instruct-devinator</a>: SmolLM 135M Instruct Trained on DEVINator Data for Open Hands (Open Devin)</li><li><a href="https://github.com/Leoleojames1/Agent_Chef">GitHub - Leoleojames1/Agent_Chef: 🍲Agent Chef🥘 is my robust tool for dataset refinement, structuring, and generation. By leveraging procedural and synthetic dataset generation techniques, Agent Chef will enable users to refine and clean their fine-tuning data, eliminating data poisoning and low-quality knowledge bases. Additionally, it will provide templates, and frameworks.</a>: 🍲Agent Chef🥘 is my robust tool for dataset refinement, structuring, and generation. By leveraging procedural and synthetic dataset generation techniques, Agent Chef will enable users to refine and ....</li><li><a href="https://huggingface.co/blog/mlabonne/abliteration">Uncensor any LLM with abliteration</a>: no description found</li><li><a href="https://github.com/Nottlespike/abliterator.py">GitHub - Nottlespike/abliterator.py: Simple Python library/structure to ablate features in LLMs which are supported by TransformerLens</a>: Simple Python library/structure to ablate features in LLMs which are supported by TransformerLens - Nottlespike/abliterator.py</li><li><a href="https://huggingface.co/datasets/Borcherding/OARC_Commander_v001">Borcherding/OARC_Commander_v001 · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/mahiatlinux/Reflection-Dataset-v1">mahiatlinux/Reflection-Dataset-v1 · Datasets at Hugging Face</a>: no description found</li><li><a href="https://github.com/daveshap/ACE_Framework">GitHub - daveshap/ACE_Framework: ACE (Autonomous Cognitive Entities) - 100% local and open source autonomous agents</a>: ACE (Autonomous Cognitive Entities) - 100% local and open source autonomous agents - daveshap/ACE_Framework
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1281533563585691679)** (10 messages🔥): 

> - `Evolution of Unsloth`
> - `Emoji Communication`
> - `App Promotion` 


- **Evolution of Unsloth - A Fun Journey**: A member shared a link discussing the [Evolution of the Peaceful Sloth](https://www.reddit.com/r/ChatGPT/comments/1faa0ur/evolution_of_the_peacefu), sparking laughter about the topic.
   - A reaction with emojis followed, showcasing enthusiasm for the discussion.
- **Emojis as a Means of Communication**: In a light-hearted moment, a member joked about being fine-tuned to convey messages using emojis, adding a playful tone to the chat.
   - Yeppp fine tuned myself to do so.
- **Conversation Around App Promotion**: One member shared a link that seemed to promote an app directly after mentioning the evolution topic.
   - This led to another member humorously stating, *'No promotion!'*, highlighting the spontaneous banter in the chat.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.reddit.com/r/ChatGPT/comments/1faa0ur/evolution_of_the_peaceful_sloth/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.reddit.com/r/ChatGPT/comments/1faa0ur/evolution_of_the_peacefu">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1281402766241038412)** (43 messages🔥): 

> - `Unsloth Library Installation`
> - `Kaggle Competition Constraints`
> - `Phi 3.5 Fine Tuning`
> - `Gemma-2-27B Loading Issues`
> - `Mistral 7B Domain Limitation` 


- **Unsloth Library Installation Problems**: Some users reported issues with installing the **Unsloth** library on Kaggle, particularly with the latest [notebook instructions](https://www.kaggle.com/code/danielhanchen/kaggle-qwen-2-7b-unsloth-notebook/notebook). Assistance was sought regarding updates to the installation process.
   - Participants are encouraged to share any recent development to address the installation problems users have been facing.
- **Kaggle Competition Constraints on Internet Access**: A member shared concerns regarding the requirement for **no internet access** during Kaggle competition submissions, impacting their ability to install required models and libraries. The discussion included suggested workarounds and potential solutions.
   - Suggestions included running some cells with internet enabled before switching it off, although some felt that this would not adequately solve the problem.
- **Phi 3.5 Template and Gibberish Output**: Users reported challenges with the Phi 3.5 model returning **gibberish outputs** while attempting to fine-tune it during training. Adjusting parameters like temperature and top_p did not resolve the issue for all users.
   - There was discussion on finding appropriate templates and troubleshooting methods, but many participants expressed frustrations with the model's performance.
- **Gemma-2-27B Weight Initialization Warnings**: Concerns were raised about **initialization warnings** for weights when loading trained **Gemma-2-27B** models, with users referencing a relevant GitHub issue for context. They sought workarounds to mitigate these warnings.
   - Unexpected behavior was noted during model loading, prompting users to seek solutions from others who encountered similar issues.
- **Limitations of Vision Models with Unsloth**: A question was posed about using **Phi 3.5 vision models** with Unsloth, but the consensus was that it is not currently supported. There is anticipation that support for vision LLMs will be added in the future.
   - Users expressed interest in the evolution of Unsloth's capabilities, especially concerning fine-tuning options for vision-related models.



**Link mentioned**: <a href="https://github.com/unslothai/unsloth/issues/478">Qwen2 error when loading from checkpoint · Issue #478 · unslothai/unsloth</a>: Works as expected when loading the base model, but when a LoRA checkpoint is loaded in place of the base model, unsloth returns: Unsloth cannot patch Attention layers with our manual autograd engin...

  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1281337667858202625)** (2 messages): 

> - `Comparison Reports`
> - `YouTube Explanations` 


- **Interest in Comparison Reports**: A member expressed interest in a report comparing certain topics, stating it would be **interesting to read**.
   - *No specific discussions or reports were mentioned regarding this comparison.*
- **Upcoming YouTube Video on Comparisons**: Another member announced plans to create a [YouTube video](https://youtube.com) that will explain the comparisons in detail.
   - *This video aims to address the interest shown in comparing the relevant topics.*


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1281523816480247848)** (1 messages): 

> - `Message Duplication`
> - `Channel Oversight` 


- **Duplicate Posting in Channels**: A member questioned the rationale behind a message being posted in the channel, noting that it had already been shared in the 'help' channel.
   - *Please remove this* was the direct request made, indicating frustration regarding the repetition of content.
- **Concern Over Channel Management**: The member expressed discontent over the lack of oversight in channel posts, highlighting that it led to confusion among participants.
   - This reflects a broader concern regarding the organization and maintenance of topic relevance within the community.


  

---



### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1281329238435168269)** (142 messages🔥🔥): 

> - `Image API options`
> - `Reflection Llama-3.1 70B updates`
> - `LM Studio issues`
> - `Scraping data with local LLMs`
> - `Accessing Llama 3.1 405B model` 


- **Searching for free high-limit Image API options**: Users discussed potential free options for Image APIs with high limits, with curiosity about providers offering API access for models like Stable Diffusion.
   - They also inquired about any providers giving access to these features at scale.
- **Reflection Llama-3.1 70B receiving updates**: Reflection Llama-3.1 70B has been hailed as the top open-source LLM, with new techniques enhancing its capability to detect and correct reasoning mistakes.
   - Members also noted some performance issues and discussed working prompts for optimal behavior with the model.
- **LM Studio issues with model downloads**: A user reported problems downloading models after an update to version 0.3.2, leading to inquiries regarding certificate errors and potential solutions.
   - Community members discussed workarounds like adjusting VRAM and context size, while also clarifying that the summarization feature of RAG does not support certain functions.
- **Web scraping and local LLM utilities**: A user inquired about agents for web scraping that could connect to LM Studio, with replies suggesting Python and tools like ScrapeGraphAI.
   - Community advice focused on the efficiency of scraping first and then processing data with LLMs instead of trying to scrape with LLMs directly.
- **Accessing Llama 3.1 405B model**: A discussion took place on obtaining access to the Llama 3.1 405B model, highlighting accessibility issues users faced on the meta.ai site.
   - Alternative recommendations included checking lmarena.ai or using different models, with speculation about potential filtering measures on meta.ai.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/lmstudio-community/stable-code-instruct-3b-GGUF">lmstudio-community/stable-code-instruct-3b-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/mattshumer/Reflection-70B">mattshumer/Reflection-Llama-3.1-70B · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/bartowski/Reflection-Llama-3.1-70B-GGUF/tree/main">bartowski/Reflection-Llama-3.1-70B-GGUF at main</a>: no description found</li><li><a href="https://github.com/zed-industries/zed/issues/17482">LM Studio offline model support · Issue #17482 · zed-industries/zed</a>: Check for existing issues Completed Describe the feature Add a section in the AI Models Configuration page to allow users to use a model from LM Studio. This has already been done before with #4424...</li><li><a href="https://github.com/ggerganov/llama.cpp/commit/9bc6db28d011d47a5f318dc4aebbe7927fac4629">ggml-quants : ternary packing for TriLMs and BitNet b1.58 (#8151) · ggerganov/llama.cpp@9bc6db2</a>: * ggml-quants : 1.625 bpw ternary packing for BitNet 1.58b
 
 * ggml-quants : faster 1.625 bpw AVX2 vec_dot
 
 Not using a lookup table anymore makes it match q4_0 speed.
 
 * gguf-py : fix formatt...
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1281374008318623766)** (59 messages🔥🔥): 

> - `Apple Event Announcement`
> - `Mac Studio Performance Concerns`
> - `NVIDIA RTX 3090 Performance with NVLink`
> - `LMStudio Boot Time Issues`
> - `NAS Usage with Apple Devices` 


- **Apple Event Set for iPhones and Watches**: The upcoming Apple event on **9/9** has been confirmed to focus on new **iPhones** and **watches**.
   - Members expressed anticipation for updates on the latest devices.
- **Mac Studio Slow with Large Models**: Concerns arose about **Mac Studio** with **256GB+** memory being too slow for large **models**, prompting hopes for upgrades to **LPDDR5X 10.7Gbps**.
   - A member pointed out that this could significantly improve performance across all **M4s**, boosting speeds by **70%**.
- **NVLink Boosts NVIDIA RTX 3090 Performance**: Discussion highlighted that with **2 RTX 3090s**, users can achieve between **10 to 25 t/s** for running a **70B model**.
   - One member mentioned achieving **50 t/s** with **NVLink**, although others questioned its impact on inference performance.
- **LMStudio Experiences Extended Boot Times**: Users reported that **LMStudio** is taking **15-20 seconds** to boot, significantly longer than the **2 seconds** pre-update.
   - Investigations suggested that internet connection may be causing delays, possibly related to update checks.
- **NAS Talk for Apple Users**: A member shared their positive experience with using an **Asustor NAS** for storage management compared to desktop setups.
   - There were suggestions on setting up backups for multiple devices and sharing resources across family devices efficiently.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lmstudio.ai/snapdragon">👾 LM Studio - Discover and run local LLMs</a>: Find, download, and experiment with local LLMs</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/16ubkyq/nvlink_bridge_worth_it_for_dual_rtx_3090/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.reddit.com/r/MacOS/comments/1ae3m3z/a_nas_that_actually_works_on_macos/">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---



### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1281332648672890884)** (190 messages🔥🔥): 

> - `Reflection 70B Model`
> - `Hermes 3 and Llama 3.1 API Usage`
> - `Benchmarking reflection and ICL performance`
> - `MCTS and PRM Techniques`
> - `Quantization Issues` 


- **Reflection 70B Model Performance Comparison**: Recent discussions highlighted mixed results with the [Reflection 70B model](https://huggingface.co/mattshumer/Reflection-Llama-3.1-70B), especially when compared against benchmarks like BigCodeBench-Hard, showing inferior performance in certain areas.
   - Users noted that system prompts and tokenizer issues may significantly affect outcomes, complicating the evaluation process.
- **API Options for Llama Models**: A member inquired about the best API options for using Llama 3.1 70B models, pointing out the need for support for tool call formats.
   - Suggestions included exploring platforms like Groq for efficient deployment.
- **Exploring MCTS and PRM for Model Enhancements**: Conversations suggested that combining MCTS (Monte Carlo Tree Search) with PRM (Probabilistic Roadmap) might yield better results for model training and evaluation.
   - Members expressed excitement about testing these techniques in their projects.
- **Quantization Challenges with AI Models**: Quantization efforts for the FP16 version of the 70B model produced disappointing results, particularly noted by users experimenting with int4 quantization.
   - Discussion continued around potential workarounds to improve model performance without sacrificing quality.
- **Exploration of Cognitive Science Concepts**: A member shared an academic paper discussing the dynamical hypothesis in cognitive science, indicating possible intersections with AI cognition.
   - The conversation hinted at the philosophical implications of expressing cognitive processes as computational functions.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/ZeyuanAllenZhu/status/1829326495757853005?t=VibYJ-3VXqmPmp9QWPYqSA&s=19">Tweet from Zeyuan Allen-Zhu (@ZeyuanAllenZhu)</a>: (1/7) Physics of LM, Part 2.2 with 8 results on &#34;LLM how to learn from mistakes&#34; now on arxiv: https://arxiv.org/abs/2408.16293. We explore the possibility to enable models to correct errors i...</li><li><a href="https://x.com/_xjdr/status/1832083976225513715">Tweet from xjdr (@_xjdr)</a>: plain jane 70B instruct with the system prompt from the repo, 1 ICL example of the format, top_p .95 and temp 0.7 (as recommended) and prefilled response with &#34;&lt;thinking&gt;\n&#34;  seems like ...</li><li><a href="https://x.com/soldni/status/1831857907291582552?t=HGy1Dj4Mwb1HZoezweFZfg&s=19">Tweet from Luca Soldaini 🎀 (@soldni)</a>: fig 1: shot fig 2: chaser fig 3: apropos of nothing</li><li><a href="https://x.com/mattshumer_/status/1831768677605155174">Tweet from Matt Shumer (@mattshumer_)</a>: @abacaj Not quite — we found current models struggled to do this well (they don&#39;t know when to reflect). It required training it into the model via a dataset that intentionally makes mistakes -&gt...</li><li><a href="https://x.com/vllm_project/status/1831742284804866237?s=46">Tweet from vLLM (@vllm_project)</a>: A month ago, we announced our performance roadmap. Today, we are happy to share that the latest release achieves 🚀2.7x higher throughput and is 5x faster for output latency on Llama 8B, and 1.8x high...</li><li><a href="https://x.com/mattshumer_/status/1831826171107144090?t=k5R0qg02Qr5azpPjQtfgaw&s=19">Tweet from Matt Shumer (@mattshumer_)</a>: @EnricoShippole @binary_racoon @GlaiveAI Different reflection -- just don&#39;t want any confusion, we&#39;re doing something totally different</li><li><a href="https://huggingface.co/matts">matts (Matt Szydlik)</a>: no description found</li><li><a href="https://huggingface.co/leafspark/Reflection-Llama-3.1-70B-GGUF">leafspark/Reflection-Llama-3.1-70B-GGUF · Hugging Face</a>: no description found</li><li><a href="https://openreview.net/forum?id=xaqoZZqkPU">Reflection-Tuning: Recycling Data for Better Instruction-Tuning</a>: Recent advancements in Large Language Models (LLMs) have expanded the horizons of natural language understanding and generation. Notably, the output control and alignment with the input of LLMs can...</li><li><a href="https://x.com/mattshumer_/status/1831767017507954808/photo/1">Tweet from Matt Shumer (@mattshumer_)</a>: Reflection 70B holds its own against even the top closed-source models (Claude 3.5 Sonnet, GPT-4o).  It’s the top LLM in (at least) MMLU, MATH, IFEval, GSM8K.  Beats GPT-4o on every benchmark tested. ...</li><li><a href="https://x.com/terryyuezhuo/status/1832112913391526052">Tweet from Terry Yue Zhuo (@terryyuezhuo)</a>: After verifying the required setup (with system prompt, no prefilling), I can safely say Reflection does not do well on BigCodeBench-Hard, at least.  Complete: 20.3 (vs 28.4 from Llama3.1-70B) Instruc...</li><li><a href="https://huggingface.co/mattshumer/Reflection-70B">mattshumer/Reflection-Llama-3.1-70B · Hugging Face</a>: no description found</li><li><a href="https://x.com/mattshumer_/status/1831767014341538166?t=ldUBdhhdmxU0qMgsmVaTUg&s=19">Tweet from Matt Shumer (@mattshumer_)</a>: I&#39;m excited to announce Reflection 70B, the world’s top open-source model.  Trained using Reflection-Tuning, a technique developed to enable LLMs to fix their own mistakes.  405B coming next week ...</li><li><a href="https://openrouter.ai/models/mattshumer/reflection-70b">Reflection 70B - API, Providers, Stats</a>: Reflection Llama-3.1 70B is trained with a new technique called Reflection-Tuning that teaches a LLM to detect mistakes in its reasoning and correct course. Run Reflection 70B with API</li><li><a href="https://github.com/tianyi-lab/Reflection_Tuning">GitHub - tianyi-lab/Reflection_Tuning: [ACL&#39;24] Selective Reflection-Tuning: Student-Selected Data Recycling for LLM Instruction-Tuning</a>: [ACL&#39;24] Selective Reflection-Tuning: Student-Selected Data Recycling for LLM Instruction-Tuning - tianyi-lab/Reflection_Tuning</li><li><a href="https://sci-hub.scrongyao.com/10.1017/S0140525X98001733">Sci-Hub | The dynamical hypothesis in cognitive science | 10.1017/S0140525X98001733</a>: no description found
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1281695101952987168)** (1 messages): 

> - `DeepSeek v2.5`
> - `Coding improvements` 


- **Inquiry on DeepSeek v2.5 Performance**: A member requested users to report any noticeable improvements while using **DeepSeek v2.5** for coding tasks.
   - *Please share experiences and insights!*
- **Expectation for User Feedback**: The community anticipates user feedback on the efficacy of **DeepSeek v2.5**, especially regarding coding enhancements.
   - Members are encouraged to contribute their findings to foster collective learning.


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/)** (1 messages): 

teknium: https://x.com/alexandr_wang/status/1832147956562284987?s=46
  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1281329043186122803)** (52 messages🔥): 

> - `OpenAI's $2000 Subscription Model`
> - `Reflection 70B Model Performances`
> - `Speculative Decoding in Inference`
> - `New Text-to-Music Models`
> - `AI Scientist Testing Challenges` 


- **$2000 Subscription for ChatGPT on the Table**: An ongoing discussion arises around OpenAI considering a subscription model priced at **$2000/month** for its more advanced AI models, including the expected Orion model.
   - The implications of this pricing and its justification remain a hot topic among community members who share concerns about accessibility.
- **Reflection 70B Model Under Scrutiny**: The **Reflection 70B** model's testing showed mixed results compared to Llama3, with lower performance on code benchmarks like **BigCodeBench-Hard** and **Aider**.
   - Critics suggest that performance discrepancies stem from the model’s methodology, requiring more thorough examination before fully relying on its metrics.
- **Speculative Decoding Promises Enhanced Performance**: Together AI shares findings that speculative decoding can improve latency and throughput by up to **2x** for long context inputs, contradicting prior assumptions about its effectiveness.
   - This advancement signals a significant shift in how high-throughput inference can be optimized using existing frameworks.
- **New Developments in Text-to-Music Models**: A new open-source **text-to-music model** has been released, showcasing impressive sound quality and efficiency compared to existing solutions like **Suno.ai**.
   - Developers express excitement for its application potential, despite mixed sentiments regarding comparative quality and usability in practical scenarios.
- **Challenges with AI Scientist Testing**: There are inquiries into testing the **Sakana AI Scientist** on models compatible with Apple Silicon due to PyTorch compatibility issues.
   - Discussion indicates concerns over the model's effectiveness, with members urging further investigation into performance and potential improvements.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/mattshumer_/status/1831767031735374222?s=46&t=PW8PiFwluc0tdmv2tOMdEg">Tweet from Matt Shumer (@mattshumer_)</a>: Most importantly, a huge shoutout to @csahil28 and @GlaiveAI.  I’ve been noodling on this idea for months, and finally decided to pull the trigger a few weeks ago. I reached out to Sahil and the data ...</li><li><a href="https://x.com/mjlbach/status/1831323536788791595?s=46">Tweet from Michael Lingelbach (@mjlbach)</a>: Looks like a SOTA open source text-to-music model (a rectified flow dit) is out.  Paper here: https://arxiv.org/abs/2409.00587  Code here: https://github.com/feizc/FluxMusic  The examples sound very s...</li><li><a href="https://x.com/terryyuezhuo/status/1832112913391526052">Tweet from Terry Yue Zhuo (@terryyuezhuo)</a>: After verifying the required setup (with system prompt, no prefilling), I can safely say Reflection does not do well on BigCodeBench-Hard, at least.  Complete: 20.3 (vs 28.4 from Llama3.1-70B) Instruc...</li><li><a href="https://x.com/maximelabonne/status/1832036357734109496?s=46">Tweet from Maxime Labonne (@maximelabonne)</a>: This is super cool but I have a lot of questions.  First, reflection = CoT on steroids. It means you can&#39;t compare these scores at all. Remember when people made fun of Gemini for providing CoT re...</li><li><a href="https://x.com/Techmeme/status/1831696947914404181">Tweet from Techmeme (@Techmeme)</a>: OpenAI says it now has 1M+ paid users for the corporate versions of ChatGPT, including ChatGPT Team, Enterprise, and Edu (@rachelmetz / Bloomberg)  https://www.bloomberg.com/news/articles/2024-09-05/o...</li><li><a href="https://x.com/ikristoph/status/1831803754678767875?s=46">Tweet from Kristoph (@ikristoph)</a>: I don&#39;t want to be negative by any means here about the work @mattshumer_ and his team have . These guys have done some fantastic work and I am super excited to try their models and you should be ...</li><li><a href="https://x.com/togethercompute/status/1831755763615674412">Tweet from Together AI (@togethercompute)</a>: We are excited to share our latest work on speculative decoding for high-throughput inference!  Before this work, we thought speculative decoding was useless at large batch sizes since the GPUs would ...</li><li><a href="https://x.com/aiexplainedyt/status/1831710902636228694?s=46">Tweet from AI Explained (@AIExplainedYT)</a>: Would you pay $2000/month for ChatGPT? This is the highest price that&#39;s &#39;on the table&#39;, for a subscription, according to a just-released Information report on OpenAI.   This would be the t...</li><li><a href="https://x.com/bindureddy/status/1831746158752088178">Tweet from Bindu Reddy (@bindureddy)</a>: OpenAI is considering $2K / month to access their top models.  All kidding aside, this will be a Vision-Pro level disaster.  I hope it&#39;s a joke</li><li><a href="https://x.com/cocktailpeanut/status/1831753703940092016">Tweet from cocktail peanut (@cocktailpeanut)</a>: A lot of people asked me to write a 1 click launcher for this.  Fortunately, someone else has already wasted their time and energy trying it out.  Listen and judge for yourselves. I think I&#39;ll sti...</li><li><a href="https://x.com/mattshumer_/status/1831767014341538166?s=46">Tweet from Matt Shumer (@mattshumer_)</a>: I&#39;m excited to announce Reflection 70B, the world’s top open-source model.  Trained using Reflection-Tuning, a technique developed to enable LLMs to fix their own mistakes.  405B coming next week ...</li><li><a href="https://www.melodio.ai/">Melodio AI |Vibe Your Moment - Official Website</a>: no description found</li><li><a href="https://x.com/natolambert/status/1831701773721203164?s=46">Tweet from Nathan Lambert (@natolambert)</a>: Much like Q*, OpenAI&#39;s Strawberry system has been leaked enough where we have substantive interesting hypotheses on their training setup and use cases.  Some ideas: * Self talk as reasoning with o...</li><li><a href="https://x.com/swyx/status/1832138164951249104">Tweet from swyx.io (@swyx)</a>: Diffusion transformers are awesome, but while we all wait for Sora, I like @toinfinityai&#39;s approach - severely constrain the usecase to just video sync (not just lip sync) - and go from there.   B...</li><li><a href="https://x.com/zbeyens/status/1832079140083687671?s=46">Tweet from Ziad Beyens (@zbeyens)</a>: Introducing AI Codex: the self-improving system for @cursor_ai.  ◆ http://codex.md: Error and learning repository. ◆ http://learn.md: Auto-save new insights. ◆ http://split-codex.md: Smart categorizat...</li><li><a href="https://x.com/krandiash/status/1832056408205935060?s=46">Tweet from Karan Goel (@krandiash)</a>: 2.5 months ago @elevenlabsio put up this comparison with our 10 day old Sonic model: https://elevenlabs.io/blog/elevenlabs-vs-cartesia  The team took it as a challenge, here&#39;s our new scorecard.  ...</li><li><a href="https://x.com/ericsmith1302/status/1831745370822516792?s=46">Tweet from Eric Smith (@ericsmith1302)</a>: At $90K MRR, this is how much it costs to run my solo AI startup (AutoShorts) 👇  I wanted to be fully transparent here, since I often post about revenue but not the flipside.   I haven&#39;t calculat...</li><li><a href="https://x.com/togethercompute/status/1831783919718690877?s=46">Tweet from Together AI (@togethercompute)</a>: 🚀 NVIDIA H200 and the Together Kernel Collection (TKC) are coming to Together GPU Clusters: delivering accelerated performance, efficiency, and scalability for AI training, fine-tuning, and inference...</li><li><a href="https://podcasts.apple.com/us/podcast/minus-one/id1759014294?i=1000668457399">Reid Hoffman &amp; the AI-Native Future</a>: Serial founder and prolific investor Reid Hoffman shares the winding journey that took him from the PayPal mafia to founding LinkedIn to investing in an AI-nati</li><li><a href="https://x.com/paulgauthier/status/1832160129720185225">Tweet from Paul Gauthier (@paulgauthier)</a>: Reflection 70B scored 42% on the aider code editing benchmark, well below Llama3 70B at 49%.  I modified aider to ignore the &lt;thinking/reflection&gt; tags. This model won&#39;t work properly with t...</li><li><a href="https://www.oranlooney.com/post/gpt-cnn/">A Picture is Worth 170 Tokens: How Does GPT-4o Encode Images? - OranLooney.com</a>: Here&rsquo;s a fact: GPT-4o charges 170 tokens to process each 512x512 tile used in high-res mode. At ~0.75 tokens/word, this suggests a picture is worth about 227 words&mdash;only a factor of four of...</li><li><a href="https://github.com/udecode/dotai/blob/main/codex/learn.md">dotai/codex/learn.md at main · udecode/dotai</a>: Contribute to udecode/dotai development by creating an account on GitHub.</li><li><a href="https://github.com/SakanaAI/AI-Scientist/tree/main">GitHub - SakanaAI/AI-Scientist: The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery 🧑‍🔬</a>: The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery 🧑‍🔬 - SakanaAI/AI-Scientist</li><li><a href="https://x.com/">Tweet from GitHub - FixTweet/FxTwitter: Fix broken Twitter/X embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others</a>: Fix broken Twitter/X embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others - FixTweet/FxTwitter</li><li><a href="https://blog.vllm.ai/2024/09/05/perf-update.html">vLLM v0.6.0: 2.7x Throughput Improvement and 5x Latency Reduction</a>: TL;DR: vLLM achieves 2.7x higher throughput and 5x faster TPOT (time per output token) on Llama 8B model, and 1.8x higher throughput and 2x less TPOT on Llama 70B model.</li><li><a href="https://buttondown.email/ainews/archive/ainews-to-be-named-5745/">[AINews] SciCode: HumanEval gets a STEM PhD upgrade</a>: PhD-level benchmarks are all you need. AI News for 7/15/2024-7/16/2024. We checked 7 subreddits, 384 Twitters and 29 Discords (466 channels, and 2228...
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1281705587557666910)** (76 messages🔥🔥): 

> - `AI Code Editors`
> - `Handling Errors in Engineering`
> - `Tools for Code Automation`
> - `Collaboration with AI`
> - `Fine-tuning Models` 


- **Exploring AI Code Editors**: Members expressed interest in various AI code editors like [Melty](https://github.com/meltylabs/melty) and [Pear AI](https://github.com/trypear/pearai-app) as alternatives to Cursor, with some discussing their unique features.
   - There's curiosity around features and usability, particularly with comments and TODO lines being stripped out in Cursor.
- **Engineering Beyond Happy Paths**: Discussions pointed out that effective software engineering requires handling edge cases, with one member noting their happy path code comprises only about 10% of total work.
   - This sparked a conversation about tools like Aider which assists in editing code effectively.
- **Collaboration Tools in AI Development**: Zed AI was highlighted as a powerful code editor for high-performance collaboration, with members noting its potential benefits for developers working with AI.
   - However, it was pointed out that it currently lacks bitmap font support, limiting its applicability for some users.
- **Upcoming Topics on LLMs**: Talks are in place for covering fine-tuning techniques using Loras or quantization techniques in future sessions, showing engagement in advanced AI topics.
   - Members exchanged thoughts about the intricate details of such tasks and the models involved.
- **Error Handling in AI Development**: Members discussed the importance of error handling in coding, where handling 'non-happy-path' scenarios sets engineering apart from simple prototyping.
   - Familiarity with tools that facilitate error management was also shared, emphasizing the need for robust solutions.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://zed.dev/">Zed - The editor for what&#x27;s next</a>: Zed is a high-performance, multiplayer code editor from the creators of Atom and Tree-sitter.</li><li><a href="https://aider.chat/docs/leaderboards/">Aider LLM Leaderboards</a>: Quantitative benchmarks of LLM code editing skill.</li><li><a href="https://github.com/go-go-golems/go-go-labs/tree/main/cmd/apps/catter">go-go-labs/cmd/apps/catter at main · go-go-golems/go-go-labs</a>: GO GO EXPERIMENTAL LAB. Contribute to go-go-golems/go-go-labs development by creating an account on GitHub.</li><li><a href="https://github.com/MikeBirdTech/ai-toolkit">GitHub - MikeBirdTech/ai-toolkit: A collection of community created AI tools to improve your life</a>: A collection of community created AI tools to improve your life - MikeBirdTech/ai-toolkit</li><li><a href="https://github.com/trypear/pearai-app">GitHub - trypear/pearai-app: The Open Source AI-Powered Code Editor. A fork of VSCode and Continue.</a>: The Open Source AI-Powered Code Editor. A fork of VSCode and Continue. - trypear/pearai-app</li><li><a href="https://docs.google.com/spreadsheets/d/1q5rwO4wleMTLXr1z58c2UC03QsDsGwbJY1v4UG7eEOs/edit?gid=0#gid=0">AI In Action: Weekly Jam Sessions</a>: no description found</li><li><a href="https://github.com/meltylabs/melty">GitHub - meltylabs/melty: Open source AI code editor. To download the packaged app:</a>: Open source AI code editor. To download the packaged app: - meltylabs/melty
</li>
</ul>

</div>
  

---



### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1281329809754161202)** (80 messages🔥🔥): 

> - `Perplexity usage`
> - `RunwayML controversy`
> - `Reflection model testing`
> - `Luma Dream Machine preferences`
> - `OpenAI tokens availability` 


- **Perplexity praised for efficiency**: Users highlighted their preference for **Perplexity** as it provides the fastest access to reliable information in a usable format, with some considering switching from their **ChatGPT Plus** subscriptions to it.
   - One user emphasized that it works well for school as it is not blocked, and **Arc browser** has it integrated, making it a fantastic AI search engine.
- **Tensions rise over RunwayML's customer service**: A user shared a troubling experience with **RunwayML**, describing the abrupt cancellation of a planned community meetup without any explanation, highlighting dissatisfaction with their customer service.
   - This incident raises concerns about Runway's responsiveness to its community, especially considering the loyalty of its paying members and the potential impact on their reputation.
- **Testing the Reflection model**: Discussion revolved around the **Reflection Llama-3.1 70B model**, with users expressing interest in its performance and the new training technique called Reflection-Tuning that corrects reasoning mistakes.
   - One user linked to a platform where interested individuals can try the model, noting that improvements were made after initial testing issues.
- **Luma Dream Machine offers competitive plans**: Members compared **Luma Dream Machine** with other offerings, appreciating its flexibility with plans ranging from free to $399 a month, with a recommended $29.99 per month plan being suitable for most users.
   - The growth potential of the service was discussed, with members keen on exploring its features as well.
- **OpenAI tokens are being given away**: A user offered **OpenAI tokens** for free, indicating they have 1,000 tokens available but do not intend to use them.
   - This sparked interest among channel members, suggesting a possible exchange or use of the tokens within the community.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://reflection-playground-production.up.railway.app">Reflection 70B Playground</a>: no description found</li><li><a href="https://www.bloomberg.com/news/articles/2024-09-05/openai-hits-1-million-paid-users-for-business-ver">Bloomberg - Are you a robot?</a>: no description found</li><li><a href="https://x.com/cjnnngs/status/1832074294203199936">Tweet from Cole (@cjnnngs)</a>: @runwayml Scrubbing Reddit and Discord of any real feedback on your company isn&#39;t a great look. So, I’ll post it here where you can’t swoop in and censor it. Screenshots included for full transpar...</li><li><a href="https://www.bloomberg.com/news/articles/2024-09-05/openai-hits-1-million-paid-users-for-business-version-of-chatgpt">Bloomberg - Are you a robot?</a>: no description found</li><li><a href="https://huggingface.co/mattshumer/Reflection-Llama-3.1-70B">mattshumer/Reflection-Llama-3.1-70B · Hugging Face</a>: no description found</li><li><a href="https://lumalabs.ai/dream-machine">Luma Dream Machine</a>: Dream Machine is an AI model that makes high quality, realistic videos fast from text and images from Luma AI</li><li><a href="https://huggingface.co/spaces/featherless-ai/try-this-model">HF&#39;s Missing Inference Widget - a Hugging Face Space by featherless-ai</a>: no description found
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1281431001452646411)** (10 messages🔥): 

> - `Rate Limit Issues`
> - `Custom GPT Sharing Problems`
> - `Browser Compatibility` 


- **ChatGPT rate limit confusion persists**: A Plus user reported consistently receiving an 'exceed rate limit' message despite minimal usage and switching to 4o mini. This issue prompted suggestions to seek help from [OpenAI](https://help.openai.com/).
   - The user expressed frustration over the limitations despite paying for the service, *'I haven't used ChatGPT for over 12 hours...*'.
- **Issues with sharing custom GPTs**: Several users discussed difficulties in saving changes and sharing their custom GPTs, indicating a Fluctuating access issue. One noted that after deleting a file, sharing became possible but reverted with any new additions, resulting in an 'updates pending' status.
   - Users are concerned this glitch may hinder functionality, hoping for a fix in future updates, as indicated by *'perhaps they will look into a fix in the next update.'*
- **Browser compatibility raises questions**: A user mentioned experiencing the same issues on Firefox while testing on Chrome mobile. This led to speculation about the problem not being solely browser related.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1281374205916479640)** (10 messages🔥): 

> - `Incorporating tool calls`
> - `Prompt library location`
> - `Creative prompt usage` 


- **Success with Tool Calls**: A member inquired about successfully incorporating tool calls into prompts, expressing frustration over error messages due to incorrect structure.
   - Another member shared their success in creating tool chains using over ten Python tool calls in a single output, emphasizing the importance of using the correct tool name.
- **Example for Tool Call Structure**: After struggling, a member reported figuring out the structure to include Tool results with the correct matching IDs: an Assistant message followed by a Tool message.
   - This highlights the need for meticulous attention to ID alignment in tool interactions.
- **Prompt Library Access**: A member asked for the location of the prompt library and was quickly informed by another member that it's called <#1019652163640762428> now.
   - This demonstrates the community's willingness to assist with navigation within the platform.
- **Unique Prompt Discovery**: A member shared a quirky prompt idea that involves writing the entire content of the buffer to a code block verbatim.
   - This showcases the creativity within the community in exploring different ways to use prompts.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1281374205916479640)** (10 messages🔥): 

> - `Incorporating Tool Calls`
> - `Prompt Library Location`
> - `Buffer Content Prompt` 


- **Success with Tool Calls in Prompts**: A member expressed frustration with incorporating tool calls into prompts, mentioning that they received a simple error message from OpenAI.
   - Another member claimed they successfully create **tool chains** using over **ten python tool calls** in a single output.
- **Correct Tool Structure Explained**: A member shared how to properly structure tool calls, emphasizing the need to follow an **Assistant message** with content and a corresponding **Tool Message** for the result.
   - They realized their mistake when they forgot to add the tool result after one tool call.
- **Finding the Prompt Library**: A member inquired about the location of the prompt library, asking for guidance on where to find it.
   - A response indicated that the prompt library is now referred to as <#1019652163640762428>.
- **Interesting Prompt Found**: A member shared a fun prompt they encountered, which instructs to output the entire content of the buffer verbatim.
   - This prompt highlights the capability to capture comprehensive context and instruction from preceding conversations.


  

---



### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1281330212352561343)** (97 messages🔥🔥): 

> - `Academic Lab Opportunities`
> - `Universal Transformers`
> - `Recurrence in Neural Networks`
> - `Computational Resource Challenges`
> - `Independence in Research` 


- **Exploring Academic Lab Opportunities**: Members discussed the intricacies of securing roles in academic labs, noting that while internship programs exist, cold emailing is another option with lower success rates.
   - One suggested writing a project proposal to pitch to labs, underscoring the importance of showcasing research, particularly if it's aligned with current trends.
- **Universal Transformers Under Scrutiny**: The conversation ventured into the feasibility of Universal Transformers (UTs), with one member expressing a personal obsession with this niche even if others doubt their future utility.
   - They also highlighted discussions on adaptive implicit compute in UTs that could enhance performance, though stability remains a substantial barrier to implementation.
- **Resource Allocation in Research**: Concerns were raised about resource allocation in both academia and research labs, particularly how compute availability tends to favor product-focused projects over unconventional research.
   - Members reflected on how seniority and alignment with popular research interests might impact an individual's freedom and available resources in institutions like DeepMind.
- **Cultural Differences in Funding Between US and Europe**: A member noted distinct differences in academic cultures between the US and European institutes, highlighting that European funding tends to be more relaxed.
   - Despite the perceived freedom in academia, the 'publish or perish' culture can pressure researchers to conform to popular topics, complicating niche pursuits.
- **Challenges with Recurrence in Models**: The discussion touched on recurrence models, with a focus on Deep Equilibrium Models (DEQs) and their comparison to traditional RNNs and state space models.
   - While some members shared enthusiasm for recurrence research, others expressed skepticism regarding the future of this approach, reinforcing its niche status.



**Link mentioned**: <a href="https://arxiv.org/abs/1807.03819">Universal Transformers</a>: Recurrent neural networks (RNNs) sequentially process data by updating their state with each new data point, and have long been the de facto choice for sequence modeling tasks. However, their inherent...

  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1281436023691415635)** (5 messages): 

> - `Momentum-based Optimizers`
> - `Reinforcement Learning Automation`
> - `Gradient Cosine Similarity`
> - `Consecutive Gradient Analysis` 


- **AdEMAMix Optimizer Enhances Gradient Utilization**: A proposed modification to the Adam optimizer, **AdEMAMix**, utilizes a mixture of two Exponential Moving Averages (EMAs) to optimize the handling of past gradients better than a single EMA [PDF](https://arxiv.org/pdf/2409.03137).
   - This approach aims to balance the weight of recent gradients with older ones more effectively, which has shown promising results in language modeling and image classification.
- **Automated Reinforcement Learning Agent Architecture**: A new agent architecture automates aspects of reinforcement learning workflows, allowing it to independently manage experiment progress and build curricula using a Vision-Language Model (VLM) [PDF](https://arxiv.org/pdf/2409.03402).
   - This system decomposes tasks into subtasks and retrieves skills, marking one of the first implementations of a fully automated reinforcement learning process.
- **Gradient Cosine Similarity Insights**: **Cosine similarities** of consecutive gradients suggest a recurring pattern in training datasets, correlating with the percentage of equal gradient signs and indicating underlying sequence structures.
   - This correlation hints at the notion that gradients may increasingly point in similar directions under certain dataset conditions.
- **Linear Relationship Between Gradients and Loss Derivative**: A member noted that the **cosine similarity** of consecutive gradients seems to exhibit a linear relationship with the derivative of loss during training.
   - This observation suggests deeper ties between gradient behavior and loss metric trends.
- **Insights and Resources on Model Training**: Links to the [Model Card](https://huggingface.co/distily/distily_attn_mlp_sweep) for the **Distily Attn MLP sweep** were shared, along with access to [Training Metrics](https://huggingface.co/distily/distily_attn_mlp_sweep/tensorboard) and [Community Discussions](https://huggingface.co/distily/distily_attn_mlp_sweep/discussions).
   - These resources provide a comprehensive overview of model performance and community interactions related to the sweep.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/distily/distily_attn_mlp_sweep/tensorboard">distily/distily_attn_mlp_sweep · Training metrics</a>: no description found</li><li><a href="https://arxiv.org/abs/2409.03402">Game On: Towards Language Models as RL Experimenters</a>: We propose an agent architecture that automates parts of the common reinforcement learning experiment workflow, to enable automated mastery of control domains for embodied agents. To do so, it leverag...</li><li><a href="https://arxiv.org/abs/2409.03137">The AdEMAMix Optimizer: Better, Faster, Older</a>: Momentum based optimizers are central to a wide range of machine learning applications. These typically rely on an Exponential Moving Average (EMA) of gradients, which decays exponentially the present...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1281623602680037457)** (2 messages): 

> - `Reusing Model Outputs`
> - `lm-evaluation-harness` 


- **Inquiry on Reusing Model Outputs for Benchmarks**: A member inquired about the possibility of reusing model outputs for multiple benchmarks if the datasets coincide, highlighting a concern about efficiency.
   - This raises important questions about how outputs can be effectively shared across different evaluations to save time and resources.
- **lm-evaluation-harness GitHub Resource Shared**: A member shared a link to the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness?tab=readme-ov-file#saving-results), a framework for few-shot evaluation of language models.
   - This resource may provide useful insights into how model result management can be optimized across various benchmarks.



**Link mentioned**: <a href="https://github.com/EleutherAI/lm-evaluation-harness?tab=readme-ov-file#saving-results">GitHub - EleutherAI/lm-evaluation-harness: A framework for few-shot evaluation of language models.</a>: A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness

  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1281375430204457012)** (2 messages): 

> - `Hugging Face RoPE Implementation Compatibility`
> - `Training Model for 1 Epoch` 


- **Hugging Face RoPE compatibility in GPTNeoX**: A member inquired about the compatibility between the **Hugging Face implementation of RoPE** for **GPTNeoX/Pythia** and that used by **Llama/GPT-Fast**.
   - They observed that attention outputs from the **scale_dot_product_attention** function were significantly different (over **95%**) between their implementation and the **Pythia model**.
- **Running model for just one epoch**: Another member asked if it's possible to run the model for just **1 epoch** or if they need to compute the `train_iters` manually.
   - They speculated that `train_iters` could be calculated as **num_data_sequences/(batch_size * number_of_ddp_processes)**.


  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1281331848068468818)** (74 messages🔥🔥): 

> - `Open Interpreter birthday celebration`
> - `Skills functionality in OI`
> - `Feedback on 01 app performance`
> - `Fulcra app availability`
> - `Beta testing for OI` 


- **Open Interpreter celebrates a milestone**: Members enthusiastically celebrated the birthday of Open Interpreter, with comments expressing excitement about its potential in AI-human interaction.
   - *Happy Birthday, Open Interpreter!* was a recurring sentiment, showcasing community appreciation for the innovation.
- **Skills in Open Interpreter still experimental**: Discussion highlighted that skills functionality in OI is experimental, with users asking about the persistence of skills across sessions.
   - One user noted that skills appear to be temporary, with suggestions to check the skills storage location on their machine.
- **Positive feedback on 01 app performance**: Users were impressed with the performance of the 01 app, with one stating it efficiently searched and played a song from 2,000 audio files.
   - There were some mentions of inconsistencies in results, reflecting typical early access app experiences.
- **Fulcra app expands to new regions**: The Fulcra app has launched in multiple new regions based on community requests, enhancing its accessibility.
   - Users inquired about the availability in Australia, signaling interest in expanding reach further.
- **Beta testing opportunities for Open Interpreter**: Community members expressed interest in participating in beta testing, with confirmations that opportunities were still available.
   - The enthusiasm for early access testing reflects a supportive and engaged user base.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://play.google.com/store/apps/details?id=com.interpreter.app">01 Light - Apps on Google Play</a>: no description found</li><li><a href="https://apps.apple.com/ca/app/01-light/id6601937732">‎01 Light</a>: ‎Control your computer and smart home with voice commands from anywhere. The 01 connects to a server on your home machine, enabling remote access to your files, apps, and IoT devices.   Capabilities: ...</li><li><a href="https://apps.apple.com/us/app/context-by-fulcra/id1633037434">‎Context by Fulcra</a>: ‎Context by Fulcra Dynamics is the trustworthy platform for collecting all the data your life produces.  De-silo your health metrics, calendar events, location, and the other contextual data into your...</li><li><a href="https://tenor.com/view/youre-a-wizard-hagrid-afirmation-magic-magical-gif-16533730">Youre A Wizard Hagrid GIF - Youre A Wizard Hagrid Afirmation - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/frankenstein-its-alive-happy-excited-gif-5625959">Frankenstein Its Alive GIF - Frankenstein Its Alive Happy - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/ooh-despicable-me-4-surprised-uh-oh-that%27s-gotta-hurt-gif-14253073070740964952">Ooh Despicable Me 4 GIF - Ooh Despicable me 4 Surprised - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/OpenInterpreter/open-interpreter">GitHub - OpenInterpreter/open-interpreter: A natural language interface for computers</a>: A natural language interface for computers. Contribute to OpenInterpreter/open-interpreter development by creating an account on GitHub.</li><li><a href="https://github.com/OpenInterpreter/open-interpreter/blob/dbc52593e608d3ce3d25a0eece4e84cf57bb7892/interpreter/core/computer/skills/skills.py">open-interpreter/interpreter/core/computer/skills/skills.py at dbc52593e608d3ce3d25a0eece4e84cf57bb7892 · OpenInterpreter/open-interpreter</a>: A natural language interface for computers. Contribute to OpenInterpreter/open-interpreter development by creating an account on GitHub.</li><li><a href="https://github.com/OpenInterpreter/01/pull/300/files#diff-1c7d7d67cce10f3be88bac85f3231198881bf48beb40f43cfe27015c6c9b53cd">Docs edit by MikeBirdTech · Pull Request #300 · OpenInterpreter/01</a>: no description found</li><li><a href="https://github.com/OpenInterpreter/01">GitHub - OpenInterpreter/01: The #1 open-source voice interface for desktop, mobile, and ESP32 chips.</a>: The #1 open-source voice interface for desktop, mobile, and ESP32 chips. - OpenInterpreter/01
</li>
</ul>

</div>
  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1281415680100536432)** (8 messages🔥): 

> - `Beta role for desktop`
> - `Open Interpreter 01 issues`
> - `Audio device inquiry` 


- **Request for Beta Role Access**: Multiple users expressed a desire for access to the **beta role for desktop**, including a fan who worked on the dev kit for **Open Interpreter 01**.
   - One user noted, *'Wasn't able to join live—any way to get access to the beta role for desktop?'*.
- **Issues Running 01 on M1 Mac**: A member on an **M1 Mac** reported issues with running **Open Interpreter 01**, citing errors with **torch** and environment conflicts.
   - They reached out for help, asking if any expert would be willing to troubleshoot live, stating, *'DM me if you're down.'*.
- **Inquiry About Audio Device**: A user asked if the **01 audio device** was mentioned during the presentation, following a positive comment about the session.
   - This indicates a keen interest in the technology discussed.


  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1281464644912939040)** (13 messages🔥): 

> - `404 on values page`
> - `Integration of C and Mojo`
> - `Company culture link update` 


- **404 Error on Values Page**: Members discussed that the values page on Modular's site is currently returning a **404 error** at [this link](https://www.modular.com/values). It was suggested that it might need to point to [company culture](https://www.modular.com/company/culture) instead.
- **C and Mojo Integration Made Simple**: A member inquired about integrating **C** with **Mojo**, and another member confirmed that it is possible to dynamically link to a `.so` file using `DLHandle`.
   - An example was provided: `handle = DLHandle('path/to/mylib.so')` followed by calling the function `is_even` from the C library.
- **Company Culture Link Location**: A user asked where the company culture link was found, and another user specified it was in the careers post under the section 'core company cultural values'.
   - This was confirmed with appreciation from another member, thanking for the clarification.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.modular.com/mojo/stdlib/sys/ffi/DLHandle">DLHandle | Modular Docs</a>: Represents a dynamically linked library that can be loaded and unloaded.</li><li><a href="https://www.modular.com/company/culture">Modular: Our Culture</a>: At Modular we believe a great culture is the key to creating a great company. The three pillars we work by are Build products users love, Empower people, and Be an incredible team.</li><li><a href="https://www.modular.com/company/career-post?4450311005&gh_jid=4450311005">Modular: Career Post</a>: At Modular we believe a great culture is the key to creating a great company. The three pillars we work by are Build products users love, Empower people, and Be an incredible team.</li><li><a href="https://www.modular.com/values">Modular: Our Culture</a>: At Modular we believe a great culture is the key to creating a great company. The three pillars we work by are Build products users love, Empower people, and Be an incredible team.
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1281328055549427874)** (68 messages🔥🔥): 

> - `Mojo async functionality`
> - `Use of DType as Dict key`
> - `Improvements in constructor usage`
> - `Wrapper for pop.array`
> - `MLIR and IR generation in Mojo` 


- **Mojo's Async Functions Confusion**: A user reported issues with using `async fn` and `async def`, indicating their attempts did not work in the stable build of Mojo.
   - It was clarified that async features are only available in nightly builds, leading to a suggestion to check the version being used.
- **DType Cannot Be Used as Dict Key**: A user questioned why `DType` cannot be used as a key in a Dictionary, despite it implementing the `KeyElement` trait.
   - This issue sparked a discussion about the constraints and usage of types in Mojo's data structures.
- **Enhancements in Constructor Usage**: A user shared progress in resolving constructor issues related to `Arc[T, True]` and `Weak[T]`, emphasizing the complexity with @parameter guards.
   - Suggestions were made to maintain consistent naming in the standard library and improve the structure of types for better clarity.
- **Wrapper for pop.array Insights**: A member discussed creating a wrapper for `pop.array` intended for optional fields, revealing some difficulties in locating the implementation.
   - Further notes were made about refining pointer indirection within the data structure to enhance usability.
- **Discussion on MLIR and IR Generation**: Several users expressed interest in how MLIR can be utilized more effectively within Mojo, particularly regarding IR generation and its benefits.
   - A video from a LLVM meeting was proposed as a valuable resource to understand Mojo's interplay with MLIR and LLVM in further detail.



**Link mentioned**: <a href="https://www.youtube.com/watch?v=SEwTjZvy8vw">2023 LLVM Dev Mtg - Mojo 🔥: A system programming language for heterogenous computing</a>: 2023 LLVM Developers&#39; Meetinghttps://llvm.org/devmtg/2023-10------Mojo 🔥: A system programming language for heterogenous computingSpeaker: Abdul Dakkak, Chr...

  

---



### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1281369655499292672)** (4 messages): 

> - `Reflection 70B model`
> - `Reflection Tuning technique`
> - `Together's custom kernel performance` 


- **Announcement of Reflection 70B Model**: An exciting announcement revealed the launch of **Reflection 70B**, claimed to be the world’s top open-source model using [Reflection-Tuning](https://x.com/mattshumer_/status/1831767014341538166) to allow LLMs to correct their own mistakes.
   - A forthcoming **405B model** is anticipated next week to potentially outperform all existing models in the market.
- **Explaining Reflection Tuning**: Discussion emerged on **Reflection-Tuning**, with claims that it integrates <thought> and <reflection> tags for Chain of Thought (CoT) and self-reflection in outputs, as illustrated in this [example of long addition](https://x.com/johnubalis/status/1831792041438949833).
   - It's suggested that synthetic training data, possibly generated with STaR, plays a crucial role in the training process.
- **Together GPU Clusters' Performance Boost**: Questions arose about new **20% faster MLP kernels** released by Together, which promise significant speed improvements for AI operations, claiming up to 24% faster training and 75% faster FP8 inference compared to standard implementations.
   - These enhancements are designed to reduce GPU hours and associated costs, thereby accelerating time to market for AI solutions.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/mattshumer_/status/1831767014341538166">Tweet from Matt Shumer (@mattshumer_)</a>: I&#39;m excited to announce Reflection 70B, the world’s top open-source model.  Trained using Reflection-Tuning, a technique developed to enable LLMs to fix their own mistakes.  405B coming next week ...</li><li><a href="https://www.together.ai/blog/nvidia-h200-and-h100-gpu-cluster-performance-together-kernel-collection">Supercharging NVIDIA H200 and H100 GPU Cluster Performance With Together Kernel Collection</a>: no description found</li><li><a href="https://x.com/johnubalis/status/1831792041438949833">Tweet from John Balis (@JohnUBalis)</a>: @neurosp1ke look at how this beautiful beast performs (correct!) long addition</li><li><a href="https://glaive.ai/">Glaive - Language models for all</a>: no description found</li><li><a href="https://github.com/open-thought/system-2-research">GitHub - open-thought/system-2-research: System 2 Reasoning Link Collection</a>: System 2 Reasoning Link Collection. Contribute to open-thought/system-2-research development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1281345758913626255)** (9 messages🔥): 

> - `Debugging tips for Triton`
> - `MLIR_ENABLE_DUMP`
> - `TRITON_INTERPRET`
> - `Triton vs Marlin comparison`
> - `Quantum zero effects` 


- **Use MLIR_ENABLE_DUMP for debugging**: A member suggested using `MLIR_ENABLE_DUMP=1` to dump MLIR after each compiler pass, showing the IR before and after TTIR, TTGIR, and LLIR generation.
   - This allows for detailed insight into how Triton is compiling code, potentially aiding in pinpointing issues.
- **TRITON_INTERPRET is a helpful tool**: Another user mentioned that `TRITON_INTERPRET=1` is one of the best debugging aids in Triton.
   - The community seems to agree that adjustments to settings can greatly facilitate troubleshooting.
- **Environment variables essential for debugging**: A member highlighted that the README contains various environment variables that may assist in debugging tricky issues, although not all will be necessary.
   - They encouraged checking these out as they can provide significant help in overcoming challenges.
- **Triton shines with minimal code**: A user expressed how impressive the capabilities of Triton are, noting that significant tasks can be accomplished with just a few lines of code.
   - However, they clarified that comparing Triton with Marlin (VLLM) isn't straightforward due to differences in how zero quantization is handled.
- **Concerns over quantizing zeros**: A discussion arose about the drawbacks of quantizing zeros, referencing potential accuracy issues with this approach.
   - Another member noted that in the Marlin implementation, they mainly round the zeros for AWQ, with a distinction between symmetric and asymmetric quantization.



**Link mentioned**: <a href="https://github.com/triton-lang/triton/tree/7480ef5028b724cb434b7841b016c6d6debf3b84?tab=readme-ov-file#tips-for-hacking">GitHub - triton-lang/triton at 7480ef5028b724cb434b7841b016c6d6debf3b84</a>: Development repository for the Triton language and compiler - GitHub - triton-lang/triton at 7480ef5028b724cb434b7841b016c6d6debf3b84

  

---


### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1281602675624120503)** (1 messages): 

> - `TorchDynamo Cache Lookup`
> - `Performance Issues with Large Models`
> - `torch/nn/modules/container.py` 


- **Investigating TorchDynamo Cache Lookup Delays**: When running very large models, members noted that **600us** is spent in **TorchDynamo Cache Lookup** due to frequent calls to `torch/nn/modules/container.py(320): __getitem__`.
   - A query was raised about the specific location of this logic, seeking pointers for further investigation.
- **Performance Concerns in Large Models**: There is ongoing discussion about the performance impact on large models with particular focus on **cache lookup delays**.
   - This highlights the need for optimization strategies as these delays can accumulate during model training and inference.


  

---


### **CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1281624427586523136)** (3 messages): 

> - `NVIDIA Generative AI Teaching Kit`
> - `Efficient Machine Learning Course`
> - `Model Compression Techniques`
> - `Llama2-7B Deployment` 


- **NVIDIA collaborates with Dartmouth for AI education**: NVIDIA's **Deep Learning Institute** released a [generative AI teaching kit](https://www.hackster.io/news/nvidia-teams-up-with-dartmouth-for-a-free-generative-ai-teaching-kit-11358047a05a) developed with Dartmouth College, aimed at empowering students to understand GPU-accelerated applications.
   - *Sam Raymond* emphasized that students completing this course will gain a significant advantage in the job market, aiding in bridging knowledge gaps in various industries.
- **MIT's Efficient Machine Learning Course Announcement**: A new course at MIT focuses on **efficient machine learning** and systems to tackle the computational demands of deep neural networks, which burden cloud infrastructure and everyday devices. Topics covered include model compression, pruning, and quantization.
   - Students will gain hands-on experience deploying **Llama2-7B** on laptops, learning practical techniques to enhance deep learning applications on resource-constrained devices.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.hackster.io/news/nvidia-teams-up-with-dartmouth-for-a-free-generative-ai-teaching-kit-11358047a05a">NVIDIA Teams Up with Dartmouth for a Free Generative AI Teaching Kit</a>: Learn LLMs, NLP, GPT, diffusion, training, optimization, and more — or grab the materials you need to teach the concepts yourself.</li><li><a href="https://hanlab.mit.edu/courses/2024-fall-65940">MIT 6.5940 Fall 2024 TinyML and Efficient Deep Learning Computing</a>: no description found
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1281452444110557246)** (3 messages): 

> - `Citadel Securities hiring`
> - `Liquid AI remote roles`
> - `CUDA Mode awareness` 


- **Citadel Securities seeks research engineers**: Citadel Securities is looking for research engineers experienced in **Triton** and/or **CUDA**, emphasizing their capability to train models on terabytes of financial data.
   - They aim to optimize their training pipeline and enable production deployment within days, with more details available on their [careers page](https://www.citadelsecurities.com/careers/details/machine-learning-engineer).
- **Remote roles at Liquid AI catch attention**: A member pointed out exciting remote opportunities at Liquid AI, specifically for the **Member of Technical Staff - AI Inference Engineer** position.
   - The roles are fully remote across major cities, and the talent lead is familiar with **CUDA mode**, making it a promising application for interested engineers.
- **Positive feedback on posting jobs in CUDA mode**: Another member shared that they know a recruiter at Liquid AI and complimented them for posting job openings in **CUDA mode**.
   - This indicates a supportive community and sharing of relevant opportunities in the AI field.



**Link mentioned**: <a href="https://jobs.lever.co/liquid.ai/">Liquid AI jobs</a>: Job openings at Liquid AI

  

---


### **CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1281340616894058506)** (9 messages🔥): 

> - `Image Convolution Optimization`
> - `Control Divergence vs Arithmetic`
> - `Triton Kernels for LLM Training` 


- **Beginners Explore Image Convolution Optimizations**: A member shared their experimentation with **optimization techniques** for improving image convolution, highlighting **constant memory** use and unexpected register behavior.
   - *Local memory usage reduced the constant load*, challenging the member's understanding of memory access patterns.
- **Control Divergence vs Arithmetic Discussions**: The community analyzed the performance implications between control divergence in CUDA, with one member favoring option 1 due to **compiler optimizations** and fewer global memory accesses.
   - Conversely, another pointed out that **option 2** struggles with automatic coalescence, complicating its efficiency.
- **Exploring Google Triton for Training**: A member expressed their excitement about the **Google Triton** group and a YouTube lecture on **efficient Triton kernels** for LLM training.
   - They plan to delve into tutorials and contribute to the community in the forthcoming weeks.



**Link mentioned**: <a href="https://www.youtube.com/watch?v=gWble4FreV4">Lecture 28: Liger Kernel - Efficient Triton Kernels for LLM Training</a>: Byron Hsu presents LinkedIn&#39;s open-source collection of Triton kernels for efficient LLM training.TIMESTAMPS00:00 Host Opening00:22 Main Focus01:18 Outline03...

  

---


### **CUDA MODE ▷ #[jax](https://discord.com/channels/1189498204333543425/1203956655570817034/)** (1 messages): 

0ut0f0rder: Thanks!
  

---


### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1281489812007682132)** (14 messages🔥): 

> - `Batch Size Limitations in FP16 x INT8 Matmul`
> - `Torch Compiler Performance Issues`
> - `Torchao Installation Errors` 


- **FP16 x INT8 Matmul hits wall with batch size > 1**: The **FP16 x INT8 matmul** with `torch.compile` breaks when the batch size exceeds 1 on a **4090 RTX**, raising an error related to shared memory capacity.
   - Users speculate that the inductor configurations are likely tuned for **A100 GPUs**, leading to failures on less powerful devices.
- **Performance drop with flags on inductor**: When the inductor flags are enabled, computations become significantly slower with batch sizes greater than 1, despite sometimes not throwing an error.
   - Turning the flags off allows the matmul operation to proceed without errors, albeit at reduced speed.
- **Torchao installation error resolved**: After encountering a `RuntimeError` related to `torchao::quant_llm_linear` during installation, a user linked to a potential fix in a [GitHub pull request](https://github.com/pytorch/ao/pull/826).
   - Following the suggested correction, the error was resolved, enabling successful import of the necessary modules.



**Link mentioned**: <a href="https://github.com/pytorch/ao/pull/826">Unbreak build after #621 by andrewor14 · Pull Request #826 · pytorch/ao</a>: no description found

  

---


### **CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1281361656630345748)** (9 messages🔥): 

> - `Avoiding Burnout Strategies`
> - `Personal Projects for Productivity`
> - `Flow State in Programming`
> - `Work-Life Balance`
> - `New System Torture Test Script` 


- **Avoiding Burnout Made Simple**: A member expressed that it's better to consistently give **95%** effort rather than push for **105%**, emphasizing it leads to greater sustainability and productivity in the long run.
   - They highlighted that identifying what’s in your control and accepting what isn’t are crucial for managing personal goals without falling into the burnout trap.
- **Side Projects Reignite Passion**: Another member shared that engaging in **small side projects** outside of work has helped counteract burnout, allowing them to feel rewarded without corporate stress.
   - They noted that this approach keeps the joy of programming alive and prevents feelings of stagnation.
- **Finding Your Flow State**: The discussion highlighted the importance of reaching a **flow state** in programming, with members agreeing that nothing compares to that intense focus and productivity.
   - One noted that while they find coding easier to justify when it's for school or income, maintaining that flow is crucial.
- **Work-Life Balance Importance**: Several members agreed on the necessity of maintaining a balance in personal care, stating that neglecting basic needs leads to decreased productivity and misery.
   - They emphasized that fun and enjoyment in work improve output, advising to deal with life’s challenges before diving deep into work.
- **Introducing a System Torture Test Script**: One member shared a **new system torture test** script that runs valid Bash, C, C++, and CUDA all at once, providing a fun and useful challenge for users.
   - The script can be found on [GitHub](https://github.com/apaz-cli/Scripts/blob/master/torture), showcasing how it compiles itself and launches testing kernels based on available compilers.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.nvidia.com/en-in/events/ai-summit/">Join NVIDIA AI Summit 2024</a>: October 23–25, Mumbai, India</li><li><a href="https://github.com/apaz-cli/Scripts/blob/master/torture">Scripts/torture at master · apaz-cli/Scripts</a>: Useful scripts that I use every day. Contribute to apaz-cli/Scripts development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1281508981600157708)** (6 messages): 

> - `Small Talk on llm.c in Yerevan`
> - `Innovative Uses of llm.c`
> - `NCCL Multi-GPU Training`
> - `Scaling on GPUs` 


- **Upcoming Talk on llm.c in Yerevan**: @aleksagordic announced a **small talk** on **llm.c** in Yerevan, aiming to provide a high-level overview, including contributions from others.
   - Members expressed excitement and interest, with one looking forward to a **recording** of the talk.
- **Collecting Creative Uses of llm.c**: A query arose about whether there’s a compiled list of **creative ways** people have utilized **llm.c**, aside from forks.
   - The discussion highlighted a specific instance where **chinthysl ran llm.c on 472x H100s**, showcasing the capability of scaling.
- **NCCL Multi-GPU Multi-Node Training Success**: A member referenced a [GitHub PR](https://github.com/karpathy/llm.c/pull/426) by **chinthysl** on running **NCCL only multi-GPU** training without MPI, which simplified job scheduling using Slurm.
   - It was noted that they achieved **linear scaling** up to at least **128 GPUs**, marking a notable success in performance.
- **Excitement Around llm.c Performance**: Some members expressed enthusiasm over the impressive scaling results observed in chinthysl's GPU runs, especially regarding the **472x GPU setup**.
   - They noted that chinthysl's figures showed improvement after certain fixes, reinforcing the effectiveness of the method.



**Link mentioned**: <a href="https://github.com/karpathy/llm.c/pull/426#issuecomment-2175386065),">NCCL only multi-gpu multi-node training without MPI by chinthysl · Pull Request #426 · karpathy/llm.c</a>: Scheduling jobs using Slurm seems much easier in a multi-node training setup compared to setting up MPI for the cluster. This draft contains the changes to use mpirun for single-node training and S...

  

---


### **CUDA MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/1281415594754707506)** (5 messages): 

> - `Multimodal Convergence Tests`
> - `Liger's Swiglu Kernels performance`
> - `Together AI's GPU Clusters`
> - `Performance comparison against cuBLAS`
> - `Kernel optimization strategies` 


- **Multimodal Convergence Tests PR Ready for Review**: A member announced that a pull request is ready for review, which includes **multimodal convergence tests**.
   - This new feature is expected to enhance the testing capabilities of the implementation.
- **Liger's Swiglu Kernels vs Together AI Benchmarks**: A member inquired about the performance of **Liger's swiglu kernels** compared to benchmarks from [Together AI](https://www.together.ai/blog/nvidia-h200-and-h100-gpu-cluster-performance-together-kernel-collection).
   - They highlighted that Together's **TKC** offers up to **24% speedup** for frequent training operations.
- **Performance Assessment of Specialized Kernels**: It was shared that their specialized kernel outperforms the common implementation using **cuBLAS** and **PyTorch eager mode** by **22-24%**.
   - Members discussed the lack of granular tuning as a potential area for improvement in their fusion process.
- **Curiosity About Performance Achievements**: A member asked for insights on how **Together AI** achieves their performance improvements compared to other implementations.
   - This reflects ongoing interest in understanding best practices for kernel optimization.



**Link mentioned**: <a href="https://www.together.ai/blog/nvidia-h200-and-h100-gpu-cluster-performance-together-kernel-collection">Supercharging NVIDIA H200 and H100 GPU Cluster Performance With Together Kernel Collection</a>: no description found

  

---



### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1281382320191963300)** (43 messages🔥): 

> - `Reflection Llama-3.1 70B`
> - `Glaive data usage`
> - `Model performance`
> - `Hype around LLMs`
> - `Feedback on self-reflection prompts` 


- **Reflection Llama-3.1 70B faces mixed results**: The recently released **Reflection Llama-3.1 70B** claims to be the world's top open-source model but has shown disappointing performance on benchmarks like **BigCodeBench-Hard**, with scores much lower than previous models.
   - One user noted a decline in performance for reasoning tasks and humorously described the model's reception on Twitter as 'non news item meh model'.
- **Concerns over Glaive's synthetic data**: Some users expressed skepticism about the effectiveness of synthetic data generated by **Glaive**, referencing past contamination issues in datasets.
   - The conversation hinted at the possibility that this synthetic data might have adversely affected the performance and generalization capabilities of the Reflection Llama model.
- **Intrigue around self-reflection capabilities**: Questions arose regarding the underlying logic of the **self-reflection process**, with suggestions that models might learn to generate errors purposely to enable reflections and corrections.
   - Critics pointed out that if the training data emphasizes corrections over correct reasoning, it could cultivate a disadvantageous model behavior.
- **The impact of social media hype**: The group acknowledged the significant hype surrounding new AI models, emphasizing how social media can amplify expectations despite potential performance discrepancies.
   - One commenter humorously remarked on the Twitter hype culture, suggesting that it fosters unnecessary excitement around models that may not perform as advertised.
- **Discourse contributing to SEO**: Several users recognized the merit of engaging in Twitter discussions for enhancing blog post visibility and SEO metrics.
   - One individual expressed a pragmatic view on participating in the discourse primarily for the benefit of their online presence, despite personal skepticism about the model.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.bloomberg.com/news/articles/2024-09-05/the-rise-and-pivot-of-germany-s-one-time-ai-champion">Bloomberg - Are you a robot?</a>: no description found</li><li><a href="https://x.com/mattshumer_/status/1831843865655214283">Tweet from Matt Shumer (@mattshumer_)</a>: Meta reached out, so here&#39;s the new model name and link: https://huggingface.co/mattshumer/Reflection-Llama-3.1-70B  Quoting Matt Shumer (@mattshumer_)   I&#39;m excited to announce Reflection 70B...</li><li><a href="https://x.com/HaveFunWithAI/status/1832107805815296376">Tweet from HaveFunWithAI (@HaveFunWithAI)</a>: not seeing expected improvements for math problems  Quoting Matt Shumer (@mattshumer_)   I&#39;m excited to announce Reflection 70B, the world’s top open-source model.  Trained using Reflection-Tuning...</li><li><a href="https://huggingface.co/TheDrummer/Llama-3SOME-8B-v2">TheDrummer/Llama-3SOME-8B-v2 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/mattshumer/Reflection-70B">mattshumer/Reflection-Llama-3.1-70B · Hugging Face</a>: no description found</li><li><a href="https://x.com/elder_plinius/status/1832107737012170940">Tweet from Pliny the Liberator 🐉 (@elder_plinius)</a>: that&#39;s odd...   Reflection-70b claims to have been created by Anthropic, not Meta.  &#34;Upon careful consideration, I remain confident that Anthropic created me, not Meta.&#34;  I wonder if any A...</li><li><a href="https://x.com/deepseek_ai/status/1832026579180163260">Tweet from DeepSeek (@deepseek_ai)</a>: 🚀 Exciting news! We’ve officially launched DeepSeek-V2.5 – a powerful combination of DeepSeek-V2-0628 and DeepSeek-Coder-V2-0724! Now, with enhanced writing, instruction-following, and human preferen...</li><li><a href="https://x.com/terryyuezhuo/status/1832112913391526052?s=46">Tweet from Terry Yue Zhuo (@terryyuezhuo)</a>: After verifying the required setup (with system prompt, no prefilling), I can safely say Reflection does not do well on BigCodeBench-Hard, at least.  Complete: 20.3 (vs 28.4 from Llama3.1-70B) Instruc...</li><li><a href="https://x.com/humancompressed/status/1832114674692731155">Tweet from ~bill (@humancompressed)</a>: @terryyuezhuo
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1281338645692944424)** (5 messages): 

> - `HuggingFace Numina`
> - `Math benchmarks`
> - `CHAMP benchmark`
> - `Research queries` 


- **HuggingFace Numina is a valuable resource**: Recent discussions highlighted that **HuggingFace Numina** offers great tools for data-related tasks, making it a valuable asset for researchers.
   - Members expressed excitement about its potential applications in various projects.
- **Standard math benchmarks remain unchanged**: Despite having many tools available, the general sentiment is that there are not many new **math benchmarks**, with focus still on **MATH** and **GSM8k**.
   - This could indicate a need for fresh datasets or evaluation metrics to further the field.
- **Introduction of CHAMP benchmark**: A new benchmark dataset called **CHAMP** was introduced, focusing on examining LLMs' mathematical reasoning ability using annotated math problems with hints.
   - This aims to provide a framework for exploring how additional information impacts problem-solving in complex scenarios.
- **Research collaboration sought**: A user sought input on unconventional **HuggingFace** projects that might be off the beaten path for a research endeavor.
   - There was an appeal for any notable resources or ideas that could aid in advancing their research.



**Link mentioned**: <a href="https://arxiv.org/abs/2401.06961">CHAMP: A Competition-level Dataset for Fine-Grained Analyses of LLMs&#39; Mathematical Reasoning Capabilities</a>: Recent large language models (LLMs) have shown indications of mathematical reasoning ability on challenging competition-level problems, especially with self-generated verbalizations of intermediate re...

  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1281341709552451707)** (16 messages🔥): 

> - `Reliability of Fireworks and Together`
> - `GitHub organization takedowns`
> - `Standardization of AI chat logs`
> - `Embarrassment in AI interactions`
> - `Chat templates for AI models` 


- **Fireworks and Together Reliability Issues**: Users discussed the reliability concerns of **Fireworks** and **Together**, acknowledging that neither is **100% reliable**.
   - To address this, they have implemented **failovers** to ensure functionality.
- **Curiosities about GitHub Takedowns**: A query arose regarding whether **GitHub** takes down organizations without supplying a reason, with some recalling past instances of this occurring.
   - Concerns were expressed about the lack of communication, particularly for larger entities like **Alibaba**.
- **Need for Standard AI Chat Logs**: A member proposed that there should be a standard **`chats.txt`** file to document interactions with AI for better codebase documentation.
   - Another suggested that **Cursor** enhances this idea's utility, indicating a shift that may already be happening.
- **Embarrassment About AI Questions**: Concerns were voiced about the embarrassment of asking simple questions to **Cursor**, wishing to maintain the facade of competence.
   - This sentiment resonated with others, highlighting a common fear of being perceived as inexperienced.
- **Chat Templates for Model Standardization**: A suggestion was made to standardize **chat templates** for AI models as a precursor to implementing a **`chats.txt`** file.
   - This playfully implied hidden goals towards creating such a standardized logging system.


  

---



### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1281330402899791940)** (42 messages🔥): 

> - `Getting into Tech with No Experience`
> - `Bing Copilot Capabilities`
> - `Perplexity AI Referral Program`
> - `Web3 Innovation Job Opportunities` 


- **Advice for Entering Tech Industry without Skills**: A member expressed eagerness to enter the tech industry without technical skills, seeking advice on building a compelling CV and networking effectively.
   - Another member mentioned starting cybersecurity training through PerScholas, highlighting enthusiasm for coding and AI.
- **Bing Copilot's Source Presentation**: A user compared Bing Copilot's ability to provide up to **5 sources** with inline images to Perplexity's current capabilities.
   - They suggested that Copilot's hover preview cards on citations might be an enhancement Perplexity could consider implementing.
- **Perplexity AI Referral Program for Merch**: A shared link revealed that Perplexity is offering new merchandise through a referral program aimed at students, emphasizing sharing to earn more.
   - Another member queried about obtaining a year of free access, asking if it was limited to the first 500 sign-ups.
- **Job Openings in Web3 Innovation Team**: A post highlighted job openings in a Web3 innovation team, seeking positions from beta testers to developers and UI/UX designers.
   - The team invites applications and proposals for mutually beneficial cooperation as part of their creative vision.



**Link mentioned**: <a href="https://x.com/perplexity_ai/status/1831762895220383807?s=61">Tweet from Perplexity (@perplexity_ai)</a>: New merch for students 🔜  Just one way to get it: refer your friends to Perplexity! Share more, get more: http://perplexity.ai/backtoschool

  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1281348215009972245)** (11 messages🔥): 

> - `Sutskever's SSI funding`
> - `Volkswagen ChatGPT integration`
> - `AI-powered worldbuilding`
> - `NFL 2024 season kickoff`
> - `Vehicle-to-everything tech` 


- **Sutskever's SSI Secures $1B Funding**: Perplexity AI announced that **Sutskever's SSI** has successfully raised **$1 billion** to further its advancements in AI technology.
   - This sizable funding is expected to drive more innovations in the AI sector.
- **Volkswagen Teams Up with ChatGPT**: Volkswagen has integrated **ChatGPT** into its systems, enhancing user interaction and driving experience.
   - This move represents a significant step toward integrating advanced AI capabilities into automotive technologies.
- **Biohybrid Mushroom Robot Unveiled**: **Biohybrid mushroom robots** are now a reality, showcasing exciting developments in robotics and biotechnology.
   - These robots are designed to interact with their environment in unique ways, pushing the boundaries of traditional robotics.
- **NFL 2024 Season Kickoff Announced**: The **NFL 2024 season kickoff** details have been unveiled, generating excitement among fans and teams.
   - Fans are particularly looking forward to the new teams and players joining the roster this season.
- **Exploring Vehicle-to-Everything Tech**: The latest discourse surrounding **vehicle-to-everything (V2X)** tech highlights its potential in improving traffic efficiency and safety.
   - Innovations in V2X are anticipated to enhance connectivity between vehicles, infrastructure, and pedestrians.



**Link mentioned**: <a href="https://www.youtube.com/embed/HunZuUB0Xdo">YouTube</a>: no description found

  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1281519125620850698)** (2 messages): 

> - `pplx-api memory usage`
> - `Telegram bot memory storage` 


- **Inquiry about Memory Usage in pplx-api**: A member asked if it's possible to utilize memory storage while using the **pplx-api** through Python.
   - They requested guidance on how to implement this feature.
- **Telegram Bot's Memory Storage Strategy**: Another member shared their attempt to achieve memory usage by managing it with a separate database for their **Telegram bot**.
   - This suggests an interest in integrating memory capabilities into current chat systems.


  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1281450485890420770)** (15 messages🔥): 

> - `Bounty Questions`
> - `Tinygrad Pricing`
> - `Server Relevance`
> - `Code Readability`
> - `Guidelines Acknowledgment` 


- **Bounty Exploration Initiation**: A user expressed interest in trying out a bounty and asked for guidance on where to start, prompting a response pointing to a resource on asking smart questions: [Smart Questions FAQ](http://www.catb.org/~esr/faqs/smart-questions.html).
   - User *th.blitz* humorously acknowledged this guidance.
- **Tinygrad Pricing Drops to Zero**: A user questioned a post about offering a **4090 + 500GB** for **$60** a month, and *georgehotz* revealed that the price had been dropped to **$0**, but only for friends of tinygrad.
   - *r5q0* immediately inquired about becoming friends.
- **Server's Relevance to AI Queries**: One user pointed out that another user's questions about **AI architecture/dataset/LLM finetuning** were off-topic in the tinygrad server, which focuses on a different abstraction level.
   - This user suggested that while some members might have expertise, the questions were likely not well-received in this context.
- **Code Readability Concerns**: A member expressed difficulty reading the tinygrad code due to the lack of enforced column width limits, despite larger monitor availability.
   - *leikowo* acknowledged that such limits should be in place but noted some lines may have this feature disabled.
- **Guidelines Acknowledgment Visibility**: A user asked if the guidelines in a specific channel were the first thing seen, requiring acknowledgment before proceeding.
   - *wozeparrot* confirmed that it should indeed be the case.



**Link mentioned**: <a href="http://www.catb.org/~esr/faqs/smart-questions.html">How To Ask Questions The Smart Way</a>: no description found

  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1281357362237669386)** (18 messages🔥): 

> - `PHI operation confusion`
> - `MultiLazyBuffer features`
> - `Sharded buffer behavior`
> - `Discussion on SDXL inference`
> - `Understanding Tensor views` 


- **Clarifying PHI Operation Confusion**: A member questioned the functionality of the PHI operation in IR, noting its placement differences compared to LLVM IR, particularly in loop structures.
   - Another member suggested it might be more accurately termed as ASSIGN rather than PHI, indicating it behaves differently from traditional phi nodes.
- **Understanding MultiLazyBuffer's 'real' Property**: A user raised concerns about the purpose of `MultiLazyBuffer.real`, especially its role in `MultiLazyBuffer.shrink` and its interaction with `copy_to_device`.
   - This led to further investigation, where another member noted that it represents real lazy buffers on devices and there may be bugs with similar devices in configurations.
- **Sharded Buffer Behavior Inquiry**: A user detailed their exploration of shared buffers, specifically focusing on how they interact with sharded axes for SDXL inference and the impact on GPGPU performance.
   - This investigation prompted them to open a discussion thread seeking feedback on their findings and suggestions for improvements.
- **Discussion on Cat and Shrink Along Sharded Axis**: A user created a discussion to document findings on the capabilities and limitations of tensor operations like cat and shrink along sharded axes, specifically for MLPerf inference tasks.
   - They provided examples of unsupported operations within tinygrad and are seeking community input to address these gaps.
- **Views and Memory Realization Clarification**: A member expressed confusion regarding the realization of views in the `_recurse_lb` function, questioning the balance between memory optimization and view utilization.
   - This discussion highlights the ongoing efforts to clarify the foundational concepts of tensor views among users.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://mesozoic-egg.github.io/tinygrad-notes/uops.html">Kernel Fusion part 3: the linear layer UOps</a>: Tutorials on tinygrad</li><li><a href="https://github.com/tinygrad/tinygrad/discussions/6380">Cat and Shrink Along Sharded Axis · tinygrad/tinygrad · Discussion #6380</a>: I am making this writeup to show some of my findings and get some feedback. What The following is not supported by tinygrad a, b = [Tensor.rand(3,4).shard((&quot;NV:0&quot;,&quot;NV:1&quot;,&quot;NV:2...</li><li><a href="https://github.com/tinygrad/tinygrad/compare/master...tobias17:tinygrad:multilazybuffer-copy-fix">Comparing tinygrad:master...tobias17:multilazybuffer-copy-fix · tinygrad/tinygrad</a>: You like pytorch? You like micrograd? You love tinygrad! ❤️  - Comparing tinygrad:master...tobias17:multilazybuffer-copy-fix · tinygrad/tinygrad
</li>
</ul>

</div>
  

---



### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1281439463481806992)** (2 messages): 

> - `Gemma 2 model`
> - `Links to resources`
> - `Model information` 


- **Discussion about Gemma 2 model's link**: Members discussed the [Gemma 2 model card](https://huggingface.co/google/gemma-2-9b) shared by a user, which provides various links to technical documentation and resources.
   - *Gemma* is described as a family of lightweight, state-of-the-art open models from **Google**, built from the same technology as the **Gemini** models.
- **Resources linked for Gemma 2**: Several resources for the Gemma model were shared, including a [Responsible Generative AI Toolkit](https://ai.google.dev/responsible) and links to [Kaggle](https://www.kaggle.com/models/google/gemma-2) and [Vertex Model Garden](https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/335).
   - Members highlighted the importance of reviewing these resources for understanding the capabilities and ethics surrounding generative AI.



**Link mentioned**: <a href="https://huggingface.co/google/gemma-2-9b">google/gemma-2-9b · Hugging Face</a>: no description found

  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1281556400446640209)** (28 messages🔥): 

> - `Multimodal Generation Handling`
> - `Flex Attention for Document Masking`
> - `INT8 Mixed-Precision Training`
> - `TransformerDecoder Configuration`
> - `GitHub PRs for Generation Overhaul` 


- **Handling Causal Masks for Multimodal Models**: A member outlined the challenge of managing **causal masks** during inference in multimodal setups, particularly with fixed sequence lengths.
   - *Seeing that we're already exposing these variables through our attention layers* helps clarify the approach.
- **Expecting Speedups with Flex Attention**: There is optimism that **flex attention with document masking** will provide significant speedups in performance, especially **40% on A100 and 70% on 4090**.
   - This approach is crucial for enhancing **dynamic sequence length** training while minimizing padding inefficiencies.
- **Questions on TransformerDecoder Design**: A member queried whether a **TransformerDecoder** could be set up without self-attention layers, referencing its traditional structure.
   - Another pointed out that *the original transformer utilized* cross and self-attention layers, indicating the challenge of deviating from that model.
- **PR Updates for Generation Overhaul**: A member confirmed that **#1449** has been updated to improve compatibility with `encoder_max_seq_len` and `encoder_mask`, although testing remains pending.
   - Once this overhaul lands, it will allow for further updates to **generation utils** and integration into **PPO**.
- **Cache Refactor and Generation Utils**: There was a discussion around moving **generation out of utils** with related GitHub PR **#1424** pending due to a needed cache refactor.
   - Addressing issues with the **GemmaTransformerDecoder** being outdated made the conversation quite pressing for further developments.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/pytorch/torchtune/pull/1424).">Build software better, together</a>: GitHub is where people build software. More than 100 million people use GitHub to discover, fork, and contribute to over 420 million projects.</li><li><a href="https://github.com/pytorch/ao/pull/748">Add INT8 mixed-precision training by gau-nernst · Pull Request #748 · pytorch/ao</a>: Excerpts from the new README Terminologies for INT8 training are generally not standardized yet. To be precise, we use these terms with the following meaning:  Quantized training: model weights are...
</li>
</ul>

</div>
  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1281353509106749512)** (4 messages): 

> - `llama-deploy launch`
> - `agentic system deployment example`
> - `Running Reflection 70B`
> - `advanced agentic RAG pipelines` 


- **llama-deploy Launch for Microservices**: Announcing the launch of **llama-deploy**, a system designed to facilitate the seamless deployment of microservices based on **LlamaIndex Workflows**. This marks a significant evolution since the introduction of llama-agents and Workflows.
   - For more details, check out the [launch announcement](https://twitter.com/llama_index/status/1831794126511337880).
- **End-to-End Example Using llama-deploy**: @LoganMarkewich shared an open-source example showcasing how to build an **agentic chatbot system** using **llama-deploy** with the **@getreflex** front-end framework. This full-stack example demonstrates deploying an agentic system as microservices.
   - Find the code and details in this [example link](https://twitter.com/llama_index/status/1832132462786576652).
- **Running Reflection 70B on Your Laptop**: You can now run **Reflection 70B** using **Ollama**, provided your laptop can handle it. This allows for immediate work with it from **LlamaIndex**.
   - For more information, see the [tweet](https://twitter.com/llama_index/status/1832144451579613497).
- **Building RAG Pipelines with Amazon Bedrock**: Learn how to build advanced **agentic RAG pipelines** using **LlamaIndex** and **Amazon Bedrock**. The process includes creating pipelines, implementing dynamic query routing, and using query decomposition.
   - Follow step-by-step instructions in the detailed guide available [here](https://twitter.com/llama_index/status/1832189386169184562).


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1281487782547820649)** (21 messages🔥): 

> - `PandasQueryEngine issues`
> - `Customer support chatbot integration`
> - `NeptuneDatabaseGraphStore bug`
> - `Cohere reranker in Azure` 


- **PandasQueryEngine struggles with column names**: A user reported that the `PandasQueryEngine` can't correctly identify the column `averageRating` when used with the chat engine, often defaulting to incorrect names like `rating`.
   - Another member suggested verifying the mapping of DataFrame columns within the chat engine's context to resolve the issue.
- **Combining chat and query engines for a chatbot**: A community member seeks advice on developing a customer support chatbot that utilizes both a conversation engine and a retrieval-augmented generation (RAG) approach.
   - Members agreed that various chat engines can integrate efficiently with query engines to enhance dialogue and data retrieval capabilities for chatbot applications.
- **Potential bug in NeptuneDatabaseGraphStore**: Concerns were raised about a possible bug with the `NeptuneDatabaseGraphStore.get_schema()` function, which fails to include date information in graph summaries.
   - One user indicated that the issue likely stems from a schema parsing error when feeding data to an LLM, and there are suspicions about the `datetime` package as well.
- **Cohere reranker integration in Azure**: An individual inquired about using the Cohere reranker as a node postprocessor within Azure's LlamaIndex, referencing a GitHub inquiry about it.
   - It's confirmed that no existing Azure rerank module exists yet, but a community member encouraged creating one since the base class is simple and documentation is available.



**Link mentioned**: <a href="https://docs.llamaindex.ai/en/stable/module_guides/querying/node_postprocessors/#custom-node-postprocessor">Node Postprocessor - LlamaIndex</a>: no description found

  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1281415627654959195)** (18 messages🔥): 

> - `Reflection Llama-3.1 70B`
> - `Synthetic Dataset Generation`
> - `Model Thinking Space`
> - `Fine-tuning Challenges`
> - `ReAct CoT Technique` 


- **Reflection Llama-3.1 70B emerges as top LLM**: [Reflection Llama-3.1 70B](https://huggingface.co/mattshumer/Reflection-Llama-3.1-70B) is the world's leading open-source LLM, utilizing **Reflection-Tuning** to enhance reasoning accuracy after initial upload issues were resolved.
   - It was trained on synthetic data created by [Glaive](https://glaive.ai), and users are encouraged to test the model at [this link](https://reflection-playground-production.up.railway.app/).
- **Synthetic Dataset Generation Speeds**: Discussion highlighted that the synthetic dataset for Reflection Llama-3.1 was reportedly generated quite quickly, raising questions about its **human rater** involvement and sample size.
   - Members speculated on how fast such datasets could be created while maintaining quality.
- **Model's Thinking Space brings improvement**: One member remarked that the ability to give models space to think, known in AI circles, is well-established, referencing that **ReAct** has been implementing this for nearly two years.
   - They further noted the interesting capacity of a **4B parameter model** outperforming **GPT-3.5 turbo**, stirring excitement.
- **Challenges in Fine-tuning Llama-3.1**: The conversation turned toward the challenges of fine-tuning such a dense model, with members acknowledging that every parameter is crucial for performance.
   - Concerns about the complexity of fine-tuning were raised, with arguments about the need for custom tokens surfacing in connection with expected dataset structures.
- **ReAct CoT Performance Discussion**: Members discussed the effectiveness of the **ReAct Chain of Thought** method, stating it yields strong results without necessarily retraining models.
   - Strategies like logit constraints were mentioned as alternatives for managing outputs while maintaining clarity.



**Link mentioned**: <a href="https://huggingface.co/mattshumer/Reflection-Llama-3.1-70B">mattshumer/Reflection-Llama-3.1-70B · Hugging Face</a>: no description found

  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1281639933064384534)** (2 messages): 

> - `Fine-tuning Llama 3.1`
> - `GPU requirements for Lora finetuning` 


- **Fine-tuning Llama 3.1 with Extended Sequence Length**: A member inquired about techniques for fine-tuning **Llama 3.1** effectively, mentioning that it performs well at **8k sequence length**.
   - They noted that **rope scaling** seems to enhance performance up to **128k**, suggesting there might be a trick involved.
- **A100 GPUs Needed for Lora Finetuning**: Another member asked for an estimate of the number of **A100 80 GB GPUs** required for fine-tuning **Meta-Llama-3.1-405B-BNB-NF4-BF16** in **4 bit** using **adamw_bnb_8bit**.
   - This highlights the practical considerations and resource needs for efficient **Lora finetuning**.


  

---


### **OpenAccess AI Collective (axolotl) ▷ #[community-showcase](https://discord.com/channels/1104757954588196865/1117851527143493664/1281332076158910535)** (2 messages): 

> - `SmileyLlama`
> - `Chemical Language Model`
> - `Molecule Design` 


- **SmileyLlama: New Chemical Language Model**: [SmileyLlama](https://x.com/axolotl_ai/status/1831771214445945148) is a fine-tuned **Chemical Language Model** that designs molecules based on properties specified in the prompt.
   - It is a **SFT+DPO model** comparable to pure CLMs, but specifically built with `Axolotl`.
- **Axolotl's Approach to Molecule Generation**: The development of **SmileyLlama** showcases Axolotl's capabilities in fine-tuning models for specific tasks like molecule design.
   - This advancement illustrates how **Axolotl** adapts existing CLM techniques to enhance functionality.



**Link mentioned**: <a href="https://x.com/axolotl_ai/status/1831771214445945148">Tweet from Axolotl (@axolotl_ai)</a>: SmileyLlama, a fine-tuned Chemical Language Model to design molecules from properties specified in the prompt. An SFT+DPO model on par with other pure CLM&#39;s, but built with Axolotl.

  

---



### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1281336290377207851)** (15 messages🔥): 

> - `Cohere resources`
> - `Anthropic library usage`
> - `Embed-multilingual-light-v3.0 on Azure` 


- **Explore Cohere's Capabilities and Cookbooks**: Members discussed checking out the channel dedicated to [capabilities and demos](https://discord.com/channels/1218409701339828245) where the community shares projects built using Cohere models, referencing a comprehensive [cookbook](https://docs.cohere.com/page/cookbooks) that provides ready-made guides.
   - *sssandra* highlighted that these cookbooks showcase best practices for leveraging Cohere's generative AI platform.
- **Understanding Token Usage with Anthropic Library**: *vpkprasanna* inquired about using the Anthropic library, sharing a code snippet for calculating token usage: `message = client.messages.create(...)`.
   - They directed others to the [GitHub repository](https://github.com/anthropics/anthropic-sdk-python) for the Anthropic SDK to further explore tokenization.
- **Embed-Multilingual-Light-V3.0 Availability on Azure**: *arcz1337* questioned the availability of `embed-multilingual-light-v3.0` on Azure and asked if there are any plans to support it.
   - This inquiry reflects ongoing interest in the integration of Cohere's resources with popular cloud platforms.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.cohere.com/page/cookbooks">Cookbooks — Cohere</a>: no description found</li><li><a href="https://github.com/anthropics/anthropic-sdk-python">GitHub - anthropics/anthropic-sdk-python</a>: Contribute to anthropics/anthropic-sdk-python development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1281618137678086227)** (2 messages): 

> - `RAG citations`
> - `Text files as knowledge base` 


- **Query on RAG Citations**: A member asked how citations will affect the content of text files when using **RAG** with an external knowledge base.
   - They specifically inquired about receiving citations when they are currently getting **None** for text file content.
- **Request for Help with RAG Citations**: The same member probed for assistance in getting citations for the content sourced from text files in their **RAG** implementation.
   - They expressed urgency in figuring out how to resolve the issue regarding the absence of citations in the responses.


  

---



### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1281483770494582795)** (3 messages): 

> - `Chroma DB Setup`
> - `Weaviate Examples`
> - `Jupyter Notebooks for Server-Client Communication` 


- **Chroma DB Easier Setup**: A member highlighted the minimal setup for **Chroma DB** using just one line of code to run the server locally: `!chroma run --host localhost --port 8000 --path ./ChomaM/my_chroma_db1`.
   - They expressed satisfaction with knowing the database location and operations so simply.
- **Seeking Simplified Weaviate Setup**: The same member inquired if a similar straightforward setup for **Weaviate** exists without resorting to using **Go Docker** and additional complexities.
   - *They emphasized a desire for ease of use given their non-technical background*.
- **Biologist's Tooling with Jupyter Notebooks**: Another member shared their approach of utilizing **two Jupyter notebooks** to separately fire the server and run the client, stating this works for their needs.
   - *They identified themselves as a Biologist rather than a computer science graduate, reinforcing their need for simplicity.*
- **Desire for Weaviate Examples**: The member expressed intent to create practical examples for **Weaviate** to assist in understanding and setup.
   - This shows a proactive approach to learning despite the technical challenges involved.


  

---


### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/1281474419788615740)** (3 messages): 

> - `Importance of Names`
> - `Collaborative Learning`
> - `AI in Education`
> - `MAIC Proposal`
> - `Online Course Evolution` 


- **Names have Infinite Potential**: A member noted how *it is amazing we never run out of names*, highlighting the variety and creativity in name generation.
   - This conversation illustrates the limitless possibilities in naming conventions within various contexts.
- **Collaborative Learning Innovations**: The mention of **collabin** signalizes ongoing discussions about cooperative initiatives in online education and project work.
   - Such platforms emphasize the shift towards more integrated learning experiences in educational environments.
- **AI Enhancements Transform Education**: A detailed link shared about the paper discusses how **AI technologies** are integrated into online education for personalization and improved learning outcomes.
   - This highlights the emerging trend of using **large language models** to enhance learning experiences.
- **Introducing MAIC for Online Education**: The proposed **MAIC (Massive AI-empowered Course)** aims to leverage LLM-driven multi-agent systems for constructing AI-augmented classrooms.
   - This concept seeks to balance technology integration while enhancing the educational experience for learners.
- **Evolution of Online Courses**: Discussion around the evolution of online courses showcases the ongoing adaptation of **educational models** over time.
   - Such adaptability is crucial for accommodating various learning needs and preferences, underscoring the importance of continuous innovation.



**Link mentioned**: <a href="https://huggingface.co/papers/2409.03512">Paper page - From MOOC to MAIC: Reshaping Online Teaching and Learning through
  LLM-driven Agents</a>: no description found

  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1281411157005045821)** (2 messages): 

> - `Reflection 70B`
> - `Routing LLMs by Query`
> - `TPU Speed and Pricing` 


- **Reflection 70B announced as leading open-source model**: The unveiling of **Reflection 70B**, touted as the world's top open-source model, was shared, emphasizing its ability to correct its mistakes through **Reflection-Tuning**.
   - *405B is expected next week* with promises of superior performance, developed in collaboration with **@GlaiveAI**.
- **Interest in CoT DSpy Program Logic**: A community member inquired about the specifics of the **CoT DSpy program**, questioning its functionality regarding reflection upon provided answers.
   - There seems to be anticipation around its implementation and utility for task execution.
- **Adding Pricing and TPU Speed to LLM Routing**: A member expressed interest in developing a method to route the appropriate LLM based on queries, incorporating **pricing** and **TPU speed** based on model hosting.
   - They noted that while routing the right LLM is straightforward, additional elements like performance and cost will enhance the process.



**Link mentioned**: <a href="https://x.com/mattshumer_/status/1831767014341538166?t=DJHN74LHKtz5ULXGi2vK_A&s=19">Tweet from Matt Shumer (@mattshumer_)</a>: I&#39;m excited to announce Reflection 70B, the world’s top open-source model.  Trained using Reflection-Tuning, a technique developed to enable LLMs to fix their own mistakes.  405B coming next week ...

  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1281357639389020191)** (5 messages): 

> - `SwarmUI`
> - `User Interface Design`
> - `Bane Meme` 


- **Discussion on SwarmUI Usability**: A member expressed discomfort with UIs featuring **100 nodes**, leading to a mention of **SwarmUI** as a comparison.
   - Another member reinforced this point, declaring that it is 'literally SwarmUI'.
- **Introduction to SwarmUI GitHub**: A link to [SwarmUI on GitHub](https://github.com/mcmonkeyprojects/SwarmUI) was shared, highlighting its modular design aimed at enhanced accessibility and performance.
   - The project is noted for its focus on making powertools easily accessible, with an image showing the repository's visual.
- **Bane Meme and GIF Share**: A member shared a **Bane-themed GIF**, featuring a green frog captioned 'and you are'.
   - The GIF sparked further discussion with multiple related searches linked, showcasing various **Bane** and **explosion** themes.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/bane-no-banned-and-you-are-explode-gif-16047504">Bane No GIF - Bane No Banned - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/mcmonkeyprojects/SwarmUI">GitHub - mcmonkeyprojects/SwarmUI: SwarmUI (formerly StableSwarmUI), A Modular Stable Diffusion Web-User-Interface, with an emphasis on making powertools easily accessible, high performance, and extensibility.</a>: SwarmUI (formerly StableSwarmUI), A Modular Stable Diffusion Web-User-Interface, with an emphasis on making powertools easily accessible, high performance, and extensibility. - mcmonkeyprojects/Swa...
</li>
</ul>

</div>
  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1281349209928503489)** (3 messages): 

> - `Reflection 70B`
> - `LLM Self-Correction`
> - `Lucidrains Transfusion Implementation`
> - `405B Model Release` 


- **Reflection 70B Launches as Top Open-Source Model**: Matt Shumer announced the launch of **Reflection 70B**, claiming it to be the world’s top open-source model trained via **Reflection-Tuning** which allows LLMs to fix their own mistakes.
   - He also hinted at a **405B model** coming next week, expected to surpass all existing benchmarks.
- **LLMs Combat Bugs with Self-Correction**: Kimmonismus expressed disbelief about a new LLM that can not only correct itself but also purportedly beats GPT-4o in every benchmark tested, including **MMLU and MATH**.
   - He highlighted that this new model is open-source and dramatically outperforms **Llama 3.1's 405B**, marking a significant advancement in LLM capabilities.
- **Lucidrains Implements Transfusion Model**: A reimplementation of the **Transfusion** model by Lucidrains has been shared, aimed at predicting the next token while diffusing images, showcasing its **multi-modal** capabilities.
   - The project promises future extensions to include **flow matching and audio/video processing**, representing a noteworthy development in AI models.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/lucidrains/transfusion-pytorch">GitHub - lucidrains/transfusion-pytorch: Pytorch implementation of Transfusion, &quot;Predict the Next Token and Diffuse Images with One Multi-Modal Model&quot;, from MetaAI</a>: Pytorch implementation of Transfusion, &quot;Predict the Next Token and Diffuse Images with One Multi-Modal Model&quot;, from MetaAI - lucidrains/transfusion-pytorch</li><li><a href="https://x.com/mattshumer_/status/1831767014341538166?t=DbIKb0tk5JYIwYIMQVB8sQ&s=19">Tweet from Matt Shumer (@mattshumer_)</a>: I&#39;m excited to announce Reflection 70B, the world’s top open-source model.  Trained using Reflection-Tuning, a technique developed to enable LLMs to fix their own mistakes.  405B coming next week ...</li><li><a href="https://x.com/kimmonismus/status/1831772661296345333?t=DbIKb0tk5JYIwYIMQVB8sQ&s=19">Tweet from Chubby♨️ (@kimmonismus)</a>: I can hardly believe what I&#39;m reading here: an LLM that fixes its own bugs, corrects itself and beats all current models, including GPT-4o in all benchmarks? And the model is still OpenSource? &#3...
</li>
</ul>

</div>
  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1281346156101636149)** (6 messages): 

> - `Deploying ReAct agent on GCP`
> - `LangChain Callbacks system`
> - `Cerebras with LangChain`
> - `Decoding streams from .astream_events` 


- **ReAct Agent Deployment Challenge**: A member is facing challenges deploying their ReAct agent on GCP using FastAPI since the local SQLite database disappears on redeploys. They are seeking alternatives, specifically for Postgres or MySQL implementation as a replacement for `SqliteSaver`.
   - The member is open to sharing their local implementation for reference if someone finds it helpful.
- **Clarifying Usage of Callbacks in LangChain**: A discussion arose about whether the syntax `chain = prompt | llm` is correct, pointing to [LangChain's callback documentation](https://python.langchain.com/v0.1/docs/modules/callbacks/). Members noted that the documentation might be outdated, specifically mentioning updates in version 0.2.
   - The conversation emphasized the utility of the callbacks system for logging, monitoring, and integrating with third-party tools.
- **Inquiry on Cerebras and LangChain**: A member asked if anyone is using Cerebras in conjunction with LangChain, indicating a need for collaborative insights. The responses highlighted potential interest but lacked specific interactions.
   - No direct solutions or experiences were shared in relation to this inquiry.
- **Exploring .astream_events() Decoding**: A member inquired about a reference implementation for decoding streams from `.astream_events()`. Another member shared their experience of manually serializing every event type due to a lack of resources.
   - This dialogue expressed frustration over the tedious process and a hope for better solutions within the community.



**Link mentioned**: <a href="https://python.langchain.com/v0.1/docs/modules/callbacks/">Callbacks | 🦜️🔗 LangChain</a>: Head to Integrations for documentation on built-in callbacks integrations with 3rd-party tools.

  

---



### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1281622843179925648)** (5 messages): 

> - `RAG system improvements`
> - `Embedding model usage`
> - `Hybrid search`
> - `Metadata and reranking` 


- **Improving RAG System with Hardware Constraints**: A member inquired about enhancing their RAG system, specifically using **llama3-8b with 4bit quantization** and **BAAI/bge-small-en-v1.5** embedding model.
   - They expressed limitations due to hardware (only a **4090** GPU) and sought resources for better implementation.
- **Exploring Bigger Models with 4090 GPU**: In response, a member noted that with a **4090**, it’s possible to run a larger embedding model concurrently with llama-8b, suggesting that the **3.1 version** might also be beneficial.
   - They shared a useful [GitHub example](https://github.com/milvus-io/pymilvus/blob/master/examples/hello_hybrid_sparse_dense.py) demonstrating hybrid search integration with **bge & bm25** on Milvus.
- **Utilizing Metadata for Reranking**: The discussion highlighted the importance of having **metadata for each chunk** to assist in further sorting and filtering through results.
   - A reranker could significantly refine the search process, enhancing the overall output quality for users.



**Link mentioned**: <a href="https://github.com/milvus-io/pymilvus/blob/master/examples/hello_hybrid_sparse_dense.py">pymilvus/examples/hello_hybrid_sparse_dense.py at master · milvus-io/pymilvus</a>: Python SDK for Milvus. Contribute to milvus-io/pymilvus development by creating an account on GitHub.

  

---



### **Gorilla LLM (Berkeley Function Calling) ▷ #[leaderboard](https://discord.com/channels/1111172801899012102/1214705495974092810/1281748599537143859)** (1 messages): 

> - `XLAM system prompt`
> - `OSS models comparison` 


- **XLAM's Unique System Prompt**: A member noticed that the **system prompt for XLAM** differs from that of other **OSS models**.
   - *Is there a particular reason why?* sparked interest in exploring the rationale behind these differences.
- **Curiosity about System Design Choices**: The discussion highlights an interesting aspect regarding the **design choices** behind XLAM's system prompts.
   - Members are keen to understand if the variations are due to functionality or licensing considerations.


  

---


### **Gorilla LLM (Berkeley Function Calling) ▷ #[discussion](https://discord.com/channels/1111172801899012102/1111353033352294440/1281500226779222090)** (3 messages): 

> - `Testing API server`
> - `Adding models to leaderboard`
> - `Gorilla leaderboard` 


- **How to Test Your Own API Server**: A user inquired about methods to effectively test their own **API server** and requested related documentation.
   - No specific resources were provided in response, indicating potential knowledge gaps in the responses.
- **Contributing to the Leaderboard**: A user asked how to add a new model to the **leaderboard**, which is crucial for acknowledging model contributions.
   - In response, a link to the relevant [GitHub page](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard#contributing) was shared, detailing contribution guidelines for the **Gorilla leaderboard**.
- **Gorilla Leaderboard GitHub Resource**: Another user highlighted the **Gorilla: Training and Evaluating LLMs for Function Calls** resource available on GitHub.
   - This resource details the process of contributing to the leaderboard and was illustrated with an image from its [GitHub repository](https://opengraph.githubassets.com/25d4bf4245a01dd99c8e3d1e4b47d26ef3db55d11499f2f9edfa259231aaacd2/ShishirPatil/gorilla).



**Link mentioned**: <a href="https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard#contributing">gorilla/berkeley-function-call-leaderboard at main · ShishirPatil/gorilla</a>: Gorilla: Training and Evaluating LLMs for Function Calls (Tool Calls) - ShishirPatil/gorilla

  

---



### **Alignment Lab AI ▷ #[general](https://discord.com/channels/1087862276448595968/1095458248712265841/)** (1 messages): 

knut09896: hi there
  

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
