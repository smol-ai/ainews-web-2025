---
id: 356b3235-97c4-4179-9dde-7aaa89d13d06
title: >-
  [AINews] Rombach et al: FLUX.1 [pro|dev|schnell], $31m seed for Black Forest
  Labs
date: '2024-08-02T01:05:39.247788Z'
status: sent
type: public
source: api
metadata: {}
original_slug: ainews-rombach-et-al-flux1-prodevschnell-31m-seed
---

<!-- buttondown-editor-mode: plaintext -->**Team and $31m is all you need to recreate Stability?**

> AI News for 7/31/2024-8/1/2024. We checked 7 subreddits, [**384** Twitters](https://twitter.com/i/lists/1585430245762441216) and **28** Discords (**335** channels, and **3565** messages) for you. Estimated reading time saved (at 200wpm): **346 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

We have been covering Rombach et al's work this year closely as he [shipped Stable Diffusion 3](https://buttondown.email/ainews/archive/ainews-to-be-named-7776/) and then [left Stability AI](https://buttondown.email/ainews/archive/ainews-the-last-hurrah-of-stable-diffusion/). His new stab at the text-to-image domain is FLUX.1, and we love featuring pretty images here so here it is executing a variety of standard tasks from hyperrealistic to fantastical to photorealistic to long text prompting:

 ![image.png](https://assets.buttondown.email/images/25ecba8c-6520-4e00-8400-18bddffeaeba.png?w=960&fit=max) 

The three variants span the spectrum of size and licensing:

- pro: API only
- dev: open-weight, non-commercial
- schnell: Apache 2.0

 ![image.png](https://assets.buttondown.email/images/17ac14c1-e394-4ce7-b86f-36234df028d7.png?w=960&fit=max) 

Based on Black Forest Labs' own ELO score, all three varients outdo Midjourney and Ideogram:

 ![image.png](https://assets.buttondown.email/images/1f6fa983-3aec-4842-bdeb-234d20dc77af.png?w=960&fit=max) 

 ![image.png](https://assets.buttondown.email/images/ad73321c-a0d8-4df6-97a6-a15a89533a85.png?w=960&fit=max) 

They also announced they will work on SOTA Text-to-Video next. All in all, one of the strongest and most confident model lab launches we've seen this past year.

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

**Gemma 2 Release and AI Model Developments**

Google DeepMind released Gemma 2, a new family of open-source AI models, including a 2 billion parameter model (Gemma-2 2B) that has achieved impressive performance:

- [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1818681376323096994) announced Gemma-2 2B, a new 2 billion parameter model offering best-in-class performance for its size and efficient operation on various hardware.

- [@lmsysorg](https://twitter.com/lmsysorg/status/1818694982980845685) reported that Gemma-2 2B achieved a score of 1130 on the Chatbot Arena, outperforming models 10x its size and surpassing GPT-3.5-Turbo-0613 (1117) and Mixtral-8x7b (1114).

- [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1818697538360295897) highlighted that Gemma-2 2B outperforms all GPT-3.5 models on Chatbot Arena, using distillation to learn from larger models and optimized with NVIDIA TensorRT-LLM for various hardware deployments.

- [@fchollet](https://twitter.com/fchollet/status/1818730987435835881) noted that Gemma 2-2B is the best model for its size, outperforming GPT 3.5 and Mixtral on the lmsys Chatbot Arena leaderboard.

The release also includes additional components:

- ShieldGemma: Safety classifiers for detecting harmful content, available in 2B, 9B, and 27B sizes.
- Gemma Scope: Uses sparse autoencoders (SAEs) to analyze Gemma 2's internal decision-making, with over 400 SAEs covering all layers of Gemma 2 2B and 9B.

**AI Model Benchmarks and Comparisons**

- [@bindureddy](https://twitter.com/bindureddy/status/1818738366466412601) criticized the Human Eval Leaderboard, claiming it's gamed and doesn't accurately represent model performance. They argue that GPT-3.5 Sonnet is superior to GPT-4o-mini, despite leaderboard rankings.

- [@Teknium1](https://twitter.com/Teknium1/status/1818709594560249922) pointed out a discrepancy between Arena scores and MMLU performance for Gemma-2 2B, noting it scores higher than GPT-3.5-turbo on Arena but has an MMLU of 50 compared to 3.5-turbo's 70.

**Open-Source AI and Government Stance**

- [@ClementDelangue](https://twitter.com/ClementDelangue/status/1818573917033730230) shared that the United States Department of Commerce issued policy recommendations supporting the availability of key components of powerful AI models, endorsing "open-weight" models.

- [@ylecun](https://twitter.com/ylecun/status/1818589409685483961) praised the NTIA report supporting open-weight/open-source AI platforms, suggesting it's time to abandon innovation-killing bills based on imaginary risks.

**AI in Coding and Development**

- [@svpino](https://twitter.com/svpino/status/1818708310637658153) discussed the limitations of current AI coding tools like Cursor, ChatGPT, and Claude, noting they don't significantly improve productivity in writing code.

- [@svpino](https://twitter.com/svpino/status/1818708333588791498) emphasized the potential of "passive AI" tools that work in the background, offering recommendations and identifying issues in code without requiring explicit queries.

**Other Notable AI Developments**

- [@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1818686489749787038) demonstrated real-time video generation, producing 10 seconds of video in 11 seconds.

- [@mervenoyann](https://twitter.com/mervenoyann/status/1818675981634109701) discussed SAMv2 (Segment Anything Model 2), which introduces a new task called "masklet prediction" for video segmentation, outperforming previous state-of-the-art models.

- [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1818659445640950110) shared information about faster ternary inference, allowing a 3.9B model to run as fast as a 2B model while using only 1GB of memory.

**Memes and Humor**

- [@bindureddy](https://twitter.com/bindureddy/status/1818613179511193720) joked about Apple Vision Pro being abandoned by users and potentially being the biggest flop in Apple's history.

- [@teortaxesTex](https://twitter.com/teortaxesTex/status/1818754584430702655) shared a humorous tweet about the "Friend" gimmick.


---

# AI Reddit Recap

## /r/LocalLlama Recap

**Theme 1. Google's Gemma 2 Release and Ecosystem**

- **[Google just launched 3 new Gemma products (Gemma 2 2B, ShieldGemma, and Gemma Scope)](https://developers.googleblog.com/en/smaller-safer-more-transparent-advancing-responsible-ai-with-gemma/)** ([Score: 143, Comments: 30](https://reddit.com//r/LocalLLaMA/comments/1egrjgp/google_just_launched_3_new_gemma_products_gemma_2/)): Google has expanded its Gemma AI lineup with three new products: **Gemma 2 2B**, **ShieldGemma**, and **Gemma Scope**. While specific details about these products are not provided in the post, the launch suggests Google is continuing to develop and diversify its AI offerings in the Gemma family.

- **Gemma-2 2b 4bit GGUF / BnB quants + 2x faster finetuning with Flash Attention support!** ([Score: 74, Comments: 10](https://reddit.com//r/LocalLLaMA/comments/1egrzp7/gemma2_2b_4bit_gguf_bnb_quants_2x_faster/)): Google released **Gemma-2 2b**, trained on **2 trillion tokens** of distilled output from a larger LLM. The post author uploaded **4bit quantized versions** (bitsandbytes and GGUF) for 2b, 9b, and 27b models, and developed a method for **2x faster finetuning** with **63% less VRAM** usage, incorporating **Flash Attention v2** support for Gemma-2. They provided links to various resources including [Colab notebooks](https://colab.research.google.com/drive/1weTpKOjBZxZJ5PQ-Ql8i6ptAY2x-FWVA?usp=sharing), [quantized models on Hugging Face](https://huggingface.co/unsloth/gemma-2-it-GGUF), and an [online inference chat interface](https://colab.research.google.com/drive/1i-8ESvtLRGNkkUQQr_-z_rcSAIo9c3lM?usp=sharing) for Gemma-2 instruct.

- **[Google quietly released a sparse auto-encoder to interpret Gemma 2 and 9b. This is a google colab they put together to get you started. Super exciting, I hope Meta follows this example!](https://colab.research.google.com/drive/17dQFYUYnuKnP6OwQPH9v_GSYUW5aj-Rp?usp=sharing)** ([Score: 104, Comments: 22](https://reddit.com//r/LocalLLaMA/comments/1eh4wja/google_quietly_released_a_sparse_autoencoder_to/)): Google has released a **sparse auto-encoder** for interpreting **Gemma 2** and **9b** models, providing a **Google Colab** notebook to help users get started with the tool. This release aims to enhance the interpretability of these language models, potentially setting a precedent for increased transparency in AI development that the poster hopes other companies like **Meta** will follow.
  - The **sparse auto-encoder** tool allows visualization of layer activations for each token, potentially enabling research into **refusal removal**, **induction heads**, and model lying detection. Users can explore **low-hanging fruit** in safety research and measure **fine-tuning impacts** on specific concepts.
  - The tool opens possibilities for **runtime, low-cost fine-tuning** to promote certain moods or themes in AI models. This could be applied to create dynamic AI experiences, such as an **interrogation game** where the model's lying probability is scored in real-time.
  - Users discussed interpreting the tool's graphs, noting they show **token probabilities** which can quantify **fine-tuning effects**. The feature activations, represented as number strings, are considered more useful than the visual dashboard for analysis purposes.


**Theme 2. Open Source LLM Advancements and Comparisons**

- **Llama-3.1 8B 4-bit HQQ/calibrated quantized model: 99.3% relative performace to FP16 and fast inference speed** ([Score: 156, Comments: 49](https://reddit.com//r/LocalLLaMA/comments/1egn0yh/llama31_8b_4bit_hqqcalibrated_quantized_model_993/)): The **Llama-3.1 8B** model has been released in a **4-bit HQQ/calibrated quantized** version, achieving **99.3% relative performance** to **FP16** while offering the fastest inference speed for transformers. This high-quality quantized model is available on [Hugging Face](https://huggingface.co/mobiuslabsgmbh/Llama-3.1-8b-instruct_4bitgs64_hqq_calib), combining efficiency with performance for improved AI applications.

- **[Just dropping the image..](https://i.redd.it/y9exyzedyzfd1.png)** ([Score: 562, Comments: 74](https://reddit.com//r/LocalLLaMA/comments/1eh9sef/just_dropping_the_image/)): The image compares **OpenAI's model releases** with **open-source alternatives**, highlighting the rapid progress of open-source AI development. It shows that while OpenAI released **GPT-3** in **June 2020** and **ChatGPT** in **November 2022**, open-source models like **BLOOM**, **OPT**, and **LLaMA** were released in quick succession between **June and December 2022**, with **Alpaca** following in **March 2023**.
  - Users criticize **OpenAI's** lack of openness, with comments like *"OpenAI being full closed. The irony."* and suggestions to rename it "**ClosedAI**" or "**ClosedBots**". Some argue OpenAI is sustained by public hype and brand recognition from being first in the space.
  - **Gemma 2** from Google receives praise, with users noting its surprising quality and personality. One user describes it as *"better than L3 in many ways"* and expresses anticipation for **Gemma 3** with potential multimodality and longer context.
  - **Mistral AI** is commended for its rapid progress despite limited resources compared to larger companies. Users suggest normalizing comparisons based on team size and available resources to highlight Mistral's achievements.


- **Google's Gemma-2-2B vs Microsoft Phi-3: A Comparative Analysis of Small Language Models in Healthcare** ([Score: 65, Comments: 9](https://reddit.com//r/LocalLLaMA/comments/1eh3dei/googles_gemma22b_vs_microsoft_phi3_a_comparative/)): A comparative analysis of **Google's Gemma-2-2b-it** and **Microsoft's Phi-3-4k** models in the medical field reveals their performance without fine-tuning. **Microsoft's Phi-3-4k** outperforms with an average score of **68.93%**, while **Google's Gemma-2-2b-it** achieves **59.21%** on average, as shared in a [tweet by Aaditya Ura](https://x.com/aadityaura/status/1818855166260519407).
  - Users criticized the **graph color choices** in the original analysis, highlighting the importance of visual presentation in data comparisons.
  - Discussion arose about the specific **Phi-3 model** used, with speculation it was the **3.8B Mini version**. Users also inquired about **fine-tuning techniques** for the **PubMed dataset**.
  - Debate ensued on the relevance of evaluating **small LLMs** on **medical QA datasets**. Some argued for its importance in assessing medical knowledge, while others noted LLMs are **already being used** to answer medical questions, especially in areas with limited access to doctors.


**Theme 3. Hardware and Inference Optimization for LLMs**

- **[Woah, SambaNova is getting over 100 tokens/s on llama 405B with their ASIC hardware and they let you use it without any signup or anything.](https://i.redd.it/9bxbfajq1xfd1.png)** ([Score: 247, Comments: 94](https://reddit.com//r/LocalLLaMA/comments/1egxxc4/woah_sambanova_is_getting_over_100_tokenss_on/)): **SambaNova** has achieved a breakthrough in AI hardware performance, generating over **100 tokens per second** on the **Llama 405B** model using their **ASIC hardware**. This technology is now accessible to users without requiring any signup process, potentially democratizing access to high-performance AI inference capabilities.

- **[Post your tokens per second for llama3.1:70b](https://i.redd.it/1l6qck24ywfd1.png)** ([Score: 61, Comments: 124](https://reddit.com//r/LocalLLaMA/comments/1egxdpt/post_your_tokens_per_second_for_llama3170b/)): The post requests users to share their **tokens per second (TPS)** performance benchmarks for the **Llama 3.1 70B model**. While no specific performance data is provided in the post itself, it aims to collect and compare TPS metrics from different users and hardware setups running this large language model.

- **[70b here I come!](https://i.redd.it/kyxk7s1f0tfd1.jpeg)** ([Score: 216, Comments: 65](https://reddit.com//r/LocalLLaMA/comments/1eggumi/70b_here_i_come/)): The post author is preparing to run **70B parameter models** with a high-end GPU setup. They express excitement about their upcoming capability to work with large language models, as indicated by the enthusiastic title "70b here I come!"
  - Users discussed **thermal management**, with one mentioning **undervolting** two **3090 FE GPUs** for better performance. The original poster uses a **Meshify case** with good airflow and disables the 3090 when not needed.
  - Performance benchmarks were shared, with one user reporting **35 tokens per second** using **AWQ** and **LMDeploy** for the **LLaMA 3.1 70B** model. Another recommended a [GitHub tool](https://github.com/olealgoritme/gddr6) for monitoring **GDDR6 memory temperatures**.
  - Concerns about **3090 memory overheating** were raised, especially in warmer climates. One user experienced crashes with **Stable Diffusion** image generation and resorted to removing the case side panel for better cooling.


**Theme 4. New Tools and Frameworks for LLM Development**

- **PyTorch just released their own llm solution - torchchat** ([Score: 135, Comments: 28](https://reddit.com//r/LocalLLaMA/comments/1eh6xmq/pytorch_just_released_their_own_llm_solution/)): **PyTorch** has released **torchchat**, a new solution for running **Large Language Models (LLMs)** locally on various devices including servers, desktops, and mobile. The tool supports multiple models like **Llama 3.1**, offers **Python** and native execution modes, and includes features for **eval** and **quantization**, with the GitHub repository available at [https://github.com/pytorch/torchchat](https://github.com/pytorch/torchchat).
  - A user tested **torchchat** with **Llama 3.1**, achieving **26.47 tokens/sec** on an **NVIDIA GeForce RTX 3090**. Comparatively, **vLLM** reached **43.2 tokens/s** initially, and up to **362.7 tokens/s** with higher batch sizes.
  - Discussions focused on performance optimization, including using **--num-samples** for more representative metrics after warmup, **--compile** and **--compile-prefill** for PyTorch JIT engagement, and **--quantize** for model quantization.
  - Users inquired about **ROCm support** for AMD GPUs, compatibility with **Mamba models**, and comparisons to other frameworks like **Ollama** and **llama.cpp**.

## All AI Reddit Recap

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity

**AI Research and Applications**

- **Google DeepMind's Diffusion Augmented Agents**: A [new paper from Google DeepMind](https://arxiv.org/abs/2407.20798) introduces Diffusion Augmented Agents, potentially advancing AI capabilities in complex environments. (r/singularity)

- **AI outperforms doctors in prostate cancer detection**: A [study finds AI detects prostate cancer 17% more accurately than doctors](https://www.ndtv.com/science/ai-detects-prostate-cancer-17-more-accurately-than-doctors-finds-study-6170131), showcasing the potential of AI in medical diagnostics. (r/singularity)

**AI Products and User Experiences**

- **ChatGPT Advanced Voice Mode**: A [video demonstration](https://v.redd.it/r1hyqf4jixfd1) shows ChatGPT's voice mode mimicking an airline pilot before abruptly stopping due to content guidelines. (r/singularity)

- **OpenAI's improved conversational AI**: A [user reports](https://www.reddit.com/r/OpenAI/comments/1egrvr6/short_demo/) better conversational flow and educational capabilities in OpenAI's latest update, used during a 1.5-hour commute to learn about GitHub repositories. (r/OpenAI)

- **Criticism of AI wearable device**: A [post criticizes](https://www.reddit.com/r/singularity/comments/1egjby2/man_this_is_dumb/) a new AI wearable device, comparing it to previous failed attempts like the Humane Pin and Rabbit R1. Users discuss potential issues with the device's functionality and business model. (r/singularity)

**AI and Data Rights**

- **Reddit CEO demands payment for AI data access**: [Reddit's CEO states that Microsoft should pay to search the site](https://www.theverge.com/2024/7/31/24210565/reddit-microsoft-anthropic-perplexity-pay-ai-search), sparking discussions about data rights and compensation for user-generated content. (r/OpenAI)


---

# AI Discord Recap

> A summary of Summaries of Summaries


## Claude 3.5 Sonnet


**1. New AI Models and Capabilities**

- **Llama 3.1 Launch Sparks Debate**: Meta released **Llama 3.1**, including a new 405 billion parameter model trained on 15.6 trillion tokens, with [Together AI's blog post](https://www.together.ai/blog/llama-31-quality) sparking debate about implementation differences affecting model quality across providers.
   - The AI community engaged in discussions about potential cherry-picking of results and the importance of rigorous, transparent evaluation methodologies. [Dmytro Dzhulgakov pointed out](https://x.com/dzhulgakov/status/1818753731573551516) discrepancies in Together AI's showcase examples, emphasizing the need for consistent quality testing.
- **Flux Shakes Up Text-to-Image Generation**: **Black Forest Labs**, formed by original Stable Diffusion team members, [launched FLUX.1](https://x.com/bfl_ml/status/1819003686011449788), a new suite of state-of-the-art text-to-image models including a 12B parameter version available under non-commercial and open licenses.
   - The FLUX.1 model gained attention for its impressive capabilities, with users noting its strengths in rendering body extremities like hands and fingers. A [pro version of FLUX.1](https://replicate.com/black-forest-labs/flux-pro) is already available for testing on Replicate, showcasing the rapid development in the text-to-image space.
  


**2. AI Infrastructure and Efficiency Gains**

- **MoMa Architecture Boosts Efficiency**: Meta introduced **MoMa**, a new sparse early-fusion architecture for mixed-modal language modeling that significantly improves pre-training efficiency, as detailed in their [recent paper](https://arxiv.org/pdf/2407.21770).
   - According to [Victoria Lin](https://x.com/VictoriaLinML/status/1819037439681565178), MoMa achieves approximately 3x efficiency gains in text training and 5x in image training. The architecture employs a mixture-of-experts (MoE) framework with modality-specific expert groups for handling interleaved mixed-modal token sequences.
- **GitHub Integrates AI Models**: GitHub [announced GitHub Models](https://github.blog/news-insights/product-news/introducing-github-models/), a new feature that brings industry-leading AI tools directly to developers on their platform, aiming to bridge the gap between coding and AI engineering.
   - This integration is designed to make AI more accessible to GitHub's massive developer base, potentially transforming how coding and AI interact at scale. The community speculated whether this move is an attempt to compete with platforms like Hugging Face by integrating AI capabilities into developers' existing workflows.
  


**3. AI Ethics and Policy Developments**

- **NTIA Advocates for Open AI Models**: The National Telecommunications and Information Administration (NTIA) [issued a report](https://www.ntia.gov/press-release/2024/ntia-supports-open-models-promote-ai-innovation) supporting the openness of AI models while recommending risk monitoring to guide policymakers in the US.
   - Community members noted the NTIA's direct reporting line to the White House, giving significant weight to its policy recommendations on AI model openness. This report could potentially influence future AI regulations and policy directions in the United States.
- **Watermarking Debate in AI Trust**: A debate emerged around the effectiveness of watermarking in solving trust issues in AI, with some arguing it only works in institutional settings and cannot prevent misuse entirely.
   - The discussion suggested that better cultural norms and trust mechanisms, rather than watermarking alone, are needed to address the spread of deepfakes and misrepresented content. This highlights ongoing challenges in establishing trust and authenticity in AI-generated content.

---

# PART 1: High level Discord summaries

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Fresh Web Simulators for Neural Networks**: A new [Neural network simulation](https://starsnatched.github.io/) tool invites **AI enthusiasts** to fiddle with different neural network configurations online.
   - The simulator aims at demystifying neural network behaviors, featuring an interactive experience for users to modify and understand **neural dynamics**.
- **Blueprints for Transferable AI Wisdom**: IBM offers a detailed breakdown of [Knowledge Distillation](https://www.ibm.com/topics/knowledge-distillation), elucidating the process of imbuing compact 'student' models with insights from bulkier 'teacher' models.
   - **Knowledge distillation** stands out as a method for model compression and efficient knowledge transfer, pivotal for **AI scalability**.
- **Interactive Heatmap Chronicles Model Milestones**: An innovative [heatmap space](https://huggingface.co/spaces/cfahlgren1/model-release-heatmap) charts **AI model releases**, gaining community interest for its potential integration into Hugging Face profiles.
   - This tool presents an insightful visual aggregation of model development trends, aiming to bolster visibility and understanding of **AI evolution tempo**.
- **Crafting Semantic Parsers for Solr**: A member seeks advice on teaching a **Large Language Model (LLM)** to interpret queries for [Apache Solr](https://solr.apache.org/), aiming to generate **JSON responses** with product information.
   - With no training dataset at hand, the challenge lies in methodically guiding the LLM to enhance **search functionality** and user experience.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Chameleon Architecture Leaps Ahead**: A **new multi-modal architecture** pioneered by the creators of **Chameleon** boasts substantial efficiency gains, with details available in an [academic paper](https://arxiv.org/pdf/2407.21770).
   - **Victoria Lin** provided insights on Twitter, noting gains of *approximately 3x in text training* and *5x in image training*, making **MoMa 1.4B** a standout performer ([source](https://x.com/VictoriaLinML/status/1819037439681565178)).
- **Decoding the Speculative Decoding**: Speculative decoding mechanisms were a hot topic, with claims that smaller draft models can impact output distribution unless corrected by techniques like **rejection sampling**.
   - A [YouTube resource](https://www.youtube.com/watch?v=hm7VEgxhOvk) further explains speculative decoding, hinting at the balance between speed and fidelity in the process.
- **Bitnet Boasts Blazing Speed**: **Bitnet's finetuning approach** is drawing attention, achieving an impressive **198 tokens per second** on a singular CPU core as reported on [Reddit](https://www.reddit.com/r/LocalLLaMA/comments/1ehh9x2/hacked_bitnet_for_finetuning_ended_up_with_a_74mb/).
   - A compact **74MB model** emerged from this finetuning method, with an open-source release expected, triggering anticipation for its use in future projects ([Twitter source](https://x.com/nisten/status/1818529201231688139)).
- **LangChain: A Key or a Kink?**: Debates arose around the necessity of **LangChain** when using **Mixtral API** in the **OpenAI API** format.
   - Some members question the requirement for LangChain, suggesting direct API interactions might suffice, sparking a discussion on tool dependencies and API conventions.
- **Project Participation without the Price Tag**: Members of the community inquired about ways to assist with a no-cost AI project, with steps laid out in an anticipated **PR**.
   - The discussion affirmed the project's cost-free nature, highlighting the actionable tasks to be disclosed in a forthcoming **PR**, easing onboarding for new contributors.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Multi-GPU Meltdown to Victory**: Discussions shed light on **multi-GPU training** issues, praising fixes but highlighting initial setup headaches and environmental tweaks.
   - A swap to `llamafacs env` was the key to success for some, contrasting with the more hands-on approach of a manual **transformers upgrade** for others.
- **Unsloth Crypto Runner Unveiled**: Details on **Unsloth Crypto Runner's AES/PKI-based** design were reconciled, elucidating its cryptographic communication from client to server.
   - The community buzzed when `MrDragonFox` underscored the imperative of GPU usage, and **Skunkworks AI's** intent to **open-source** was revealed.
- **Continuous Qwen Refinement Realized**: **Qwen2-1.5B-Instruct's Continuous Fine-tuning Without Loss** ushered in a blend of code FIM and instruct capabilities, marking a technical milestone.
   - Community spirit was buoyed as a call for a tutorial to demystify documentation challenges echoed amongst users.
- **LoRA's Binding Predicament**: Merging **LoRA adapters** was brought to the fore, with a focus on the risks of melding leading to deceptive 16-bit representations from 4-bit models.
   - Concerns bubbled up about the propagation of these faux **16-bit models** within the community, prompting vigilance.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity's Prodigy Perk with Uber One**: Uber One members now have access to **Perplexity Pro** subscription for free, valid until **October 31, 2024**, providing an enhanced answer engine worth **$200**.
   - To avail this benefit, users in the US and Canada need to maintain their Uber One subscription and set up a new Perplexity Pro account. More details are at [Perplexity Uber One](https://pplx.ai/uber-one).
- **Perplexity Tops AI Search Engine Benchmarks**: In a comparative assessment, **Perplexity Pro** outranked rivals like Felo.ai and Chatlabs, excelling in UI/UX and query responses.
   - Members rated search engines on their capabilities with Pro Search appearing as a favorite, highlighted on platforms such as [ChatLabs](https://labs.writingmate.ai).
- **Perplexity API Prompts Puzzlement**: Discussions revealed user dissatisfaction regarding suboptimal outputs from Perplexity's API, feeling the result quality has declined.
   - Speculation about problem prompts rose, with individuals requesting advice on improving outcomes and expressing curiosity about **Perplexity References Beta** access.
- **Perplexity's Refined Flask Authentication**: A discussion on Flask highlighted the need for secure user authentication, recommending packages such as `Flask-Login`, and [a secure setup guide](https://www.perplexity.ai/search/please-provide-an-example-of-s-EvlJDJwUTfy4IWmobEm0Fw).
   - Users were pointed to resources outlining model creation, user authentication routes, and encryption practices.
- **OpenAI Voices Future with GPT-4o**: **OpenAI** impressed with its launch of Advanced Voice Mode for ChatGPT, granting Plus subscribers realistic voice interactions as of July 30, 2024.
   - The update allows for enhanced vocal features, like emotional tone variation and interruption handling, documented on [OpenAI's update page](https://www.perplexity.ai/page/openai-begins-hyper-realistic-2_y7h8vPQEWaM4g63WvnVA).



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Vivid Visionaries: GPT-4o Sparks Image Innovation**: Enthusiastic debate surged on GPT-4o's image output capabilities with users comparing it to **DALL-E 3**, sharing [examples](https://x.com/gdb/status/1790869434174746805) that sparked interest over its lifelike and realistic imagery.
   - Despite acclaims for GPT-4o's impressive outputs, criticisms arose on its moderation endpoint, echoing similar concerns faced by DALL-E 3.
- **Versatile Vocals: GPT-4o's Vocal Prowess Under the Microscope**: AI aficionados tested GPT-4o's [voice model abilities](https://platform.openai.com/docs/guides/embeddings/use-cases), highlighting its adaptability with accents and emotional range, and its capacity to meld background tunes and effects.
   - Findings were a mix of admiration for its potential and pointers to its inconsistent performance, igniting discussions on the model's limitations and future improvements.
- **Platform Conundrums: The Search for Prompt Precision**: AI Engineering mavericks swapped insights on preferred platforms for **prompt engineering**, elevating **Claude 3**, **Sonnet**, and **Artifacts + Projects** as prime candidates.
   - Heuristic tools for prompt evaluations grabbed the spotlight, with the **Anthropic Evaluation Tool** mentioned for its heuristic approach, while a collaborative **Google Sheet with scripts** was tabled as a sharable and efficient alternative.
- **Strategic Subscription Shift: Pondering Plus's Influence**: Community chatter revolved around the impact of cancelling Plus subscriptions, revealing that doing so would render custom GPTs inaccessible.
   - The contemplation extended to the prerequisites for GPT monetization, spotlighting the need for substantial usage metrics and localization within the USA as criteria for revenue generation opportunities.
- **The Diagram Dilemma: Charting Courses Through AI Assistance**: In the world of AI diagrams, participants probed for complimentary tools adept at crafting visual aides, with a nod to **ChatGPT** – though its diagram-drawing talents remain up for debate.
   - The dialogue also touched on the challenge LLMs face in text truncation, suggesting that seeking qualitative descriptors might be more effective than exact character or word counts.



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **FSDP Discord Sparks Flare**: A member's critique of **FSDP** as 'kind of ass' sparked debate on its scalability, countered by the claim that it excels in ease of use.
   - The conversation pivoted toward FSDP's situational suitability, indicating it's not a one-size-fits-all solution despite its user-friendly nature.
- **Sharded LLaMA Woes and vLLM Hopes**: Challenges in sharding **LLaMA 405B** on multiple nodes surfaced during discussions, with possible workarounds involving **vLLM** enhancement for larger context windows.
   - Participants recommended approaches like quantization, with some avoiding vLLM, directing users to [enhancement details and support for LLaMA 3.1](https://blog.vllm.ai/2024/07/23/llama31.html).
- **Megatron's Scholarly Appeal**: The **Megatron paper** provoked interest among members discussing distributed training's relevance, backed by resources like the [Usenix paper](https://people.eecs.berkeley.edu/~matei/papers/2021/sc_megatron_lm.pdf) and explanatory [MIT lecture video](https://www.youtube.com/watch?v=u6NEbhNiyAE&ab_channel=MITHANLab).
   - Discourse on Megatron extended to practical insights on distributed training with references to both academically acclaimed and YouTube disseminated materials.
- **Triton Tutorial's Tiled Matmul Matrix**: Queries regarding the `GROUP_SIZE_M` argument in the [Triton tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html#l2-cache-optimizations) surfaced, addressing its role in optimizing caching.
   - The debate included how setting `GROUP_SIZE_M` too high could lead to inefficiencies, exploring the delicate equilibrium of hardware design choices.
- **Llama 3.1: Turmoil and TorchChat Guideposts**: Users voiced the need for a 10-line Python snippet to simplify **Llama 3.1 model** usage, with existing [inference scripts](https://github.com/meta-llama/llama-recipes) deemed complex.
   - In response, PyTorch unveiled [TorchChat](https://github.com/pytorch/torchchat) as a guide, providing the sorely needed reference implementation to run Llama 3.1.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Stable Fast 3D's Lightning Launch**: Stability AI announced **Stable Fast 3D**, a new model capable of converting a single image to a detailed 3D asset in just **0.5 seconds**, pushing the boundaries of 3D reconstruction technology. The model's implications for gaming and VR are substantial, with a focus on **speed** and **quality**. [Discover the technical details](https://stability.ai/news/introducing-stable-fast-3d).
   - *'Stable Fast 3D's incredible processing time pioneers rapid prototyping efforts in 3D frameworks.'* Users benefit from additional features like optional remeshing, adding minimal time increase for broad industry applicability.
- **SD3 in the Spotlight**: Community discussions revolved around the utilization of **Stable Diffusion 3 (SD3)** Medium, tackling loading errors and exploring the model's capabilities. Shared solutions include obtaining all components and utilizing tools like [ComfyUI workflows](https://comfyworkflows.com/) for smoother operation.
   - Challenges such as 'AttributeError' were navigated through community support and adapting to various available UIs, ensuring more seamless creative experiences with **SD3**.
- **Solving the VAE Conundrum**: A common issue within the community was addressed: images turning red during rendering due to VAE settings. Collaborative efforts led to troubleshooting methods that mitigate the problem.
   - Applying the '--no-half-vae' command emerged as a peer-recommended fix, easing workflows for artists crafting images with accuracy while navigating hardware-specific solutions.
- **Clearing Creative Upscaler Fog**: A collective effort was made to disentangle the confusion surrounding the mention of a 'Creative Upscaler' with clarification that it is not a Stability AI project. Members exchanged alternative upscaling recommendations.
   - The favored techniques included ERSGAN application and adopting transformer technology, with advice pooling from various community-contributed resources for prompted challenges.
- **Flux: The Next Generation in Imagery**: Anticipation surrounded Black Forest Labs' release of the **Flux model**, with the community buzzing about enhancements in image rendition and efficient parameter usage. The [announcement teased potential](https://blog.fal.ai/flux-the-largest-open-sourced-text2img-model-now-available-on-fal/) for the text-to-image field.
   - Discourse on the model's GPU efficiency highlighted the Nvidia 4090 for optimal performance, with a special nod to the model's prowess in rendering body extremities like hands and fingers.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Exit Codes Expose Compatibility Clashes**: LM Studio users report exit codes like **6** and **0**, sparking conversations on system compatibility and the debugging labyrinth.
   - This dilemma has escalated to discussions around **system-specific** quirks and the potential need for updated LM Studio versions.
- **Gemma 2 Glitches Generate GPU Grief**: Challenges in running **Gemma 2 2B** models emerged, particularly on dated hardware, compelling users to advocate for a new release of LM Studio.
   - The community's response included both commiseration and shared strategies for circumventing the hardware hurdle.
- **LLaMA: The Embedding Enigma**: Enthusiasts explore embedding capabilities with projects like [LLM2Vec](https://github.com/McGill-NLP/llm2vec), amidst queries on **LLaMA**'s integration within LM Studio.
   - This culminated in curated conversations on future-forward solutions for text encoders and the excitement around embedding evolution.
- **Diving into LM Studio's Depths**: Members unraveled bugs in LM Studio, from GPU offloading oddities to nettlesome network errors potentially tied to VPN/DNS configurations.
   - Peers pitched in to pinpoint problems and proposed possible patches, promoting a collaborative climate for tackling tech troubles.
- **Vision for Vivid LM Studio Features**: The discourse delved into dreams of future LM Studio features, with users yearning for additions like **TTS voices** and **RAG-supported** document interactions.
   - [Hugging Face](https://huggingface.co/meta-llama/Meta-Llama-3.1-405B) and approaches to **Visual Question Answering (VQA)** at [Papers with Code](https://paperswithcode.com/task/visual-question-answering) garnered attention amidst these aspirations.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Watermark Woes**: AI's Authentication Angst**: Members debated watermarking's role in AI trust issues, pointing out its limited effectiveness and suggesting that establishing **cultural norms** is crucial.
   - The concern is that watermarking may not thwart misuse and misrepresented content without broader trust mechanisms in place.
- **NTIA's Open AI Advocacy**: Policy Influence Peaks**: The [NTIA report](https://www.ntia.gov/press-release/2024/ntia-supports-open-models-promote-ai-innovation) promotes the openness of AI models and recommends diligent risk monitoring to guide policymakers.
   - Observers note the weight of NTIA's policy recommendations owing to its direct reporting line to the White House, flagging potential shifts in AI regulation.
- **GitHub's Model Mashup**: Integrating AI with Code**: GitHub's introduction of [GitHub Models](https://github.blog/news-insights/product-news/introducing-github-models/) facilitates direct access to AI models within developer workflows.
   - Debate ensued on whether this is a strategy to challenge competitors like Hugging Face or a natural evolution of GitHub's service offerings.
- **Relaying the Double Descent**: Scaling Laws Under Scrutiny**: AI researchers discussed anomalies in validation log-likelihood in scaling law experiments, particularly when models with **1e6 sequences underperformed**.
   - This prompted references to the [BNSL paper](https://arxiv.org/abs/2210.14891), shedding light on similar patterns and sparking curiosity about dataset size impacts.
- **Prompt Overproducing Mystery**: lm-eval's Unexpected Multiples**: lm-eval's behavior of using more prompts than benchmarks specify, as observed in benchmarks like **gpqa_main**, incited technical inquiry and debugging efforts.
   - Clarification emerged that the progress bar in lm-eval accounts for `num_choices * num_docs`, reconciling perceived discrepancies and aiding in understanding tool behavior.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Grok's Growth: xAI Unlikely to Capture Character AI**: [Rumors](https://x.com/nmasc_/status/1818788751528935468) of xAI acquiring Character AI to enhance its Grok models have been circulating, but [Elon Musk denied these claims](https://x.com/elonmusk/status/1818810438634946699), calling the information inaccurate.
   - The community pondered the truth behind Musk's statements, referencing prior instances where official denials preceded confirmed acquisitions.
- **Black Forest Labs Emerges from Stable Diffusion's Roots**: The founding team of **Stable Diffusion** sparked excitement with the launch of [Black Forest Labs](https://x.com/bfl_ml/status/1819003686011449788), specializing in advanced generative models.
   - Black Forest Labs' **Flux** demonstrates creative prowess, and early testers can try it out on fal, signaling potential disruptions in the generative landscape.
- **GitHub Models Meshes Devs with AI Prowess**: GitHub makes a splash in AI by [introducing GitHub Models](https://github.blog/news-insights/product-news/introducing-github-models/), offering powerful AI tools to its massive developer base.
   - This new suite aims to democratize AI usage for developers, potentially transforming how coding and AI interact on a grand scale.
- **Apple Intelligence Puts a Twist in Tech's Future**: [Apple's latest AI advancements](https://www.interconnects.ai/p/apple-intelligence) promise to weave apps together more seamlessly, enhancing daily tech interactions.
   - Skeptics in AI labs question the groundbreaking status of Apple Intelligence, while others see it as a significant multiplier for tech utility.
- **Rejection Sampling Finds Home in Open Instruct**: [Open Instruct](https://github.com/allenai/open-instruct/pull/205) embraces *rejection sampling*, a method set to fine-tune training by avoiding common pitfalls.
   - The move could signal improved efficiencies in model training and a step forward for methodologies within the AI training spectrum.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Llama 3.1 Touches Nerve in Quality Debate**: [Together AI blog](https://www.together.ai/blog/llama-31-quality) spurred debate on Llama 3.1 by spotlighting variances in performance due to different implementation practices by inference providers, raising concern for model consistency.
   - Dmytro Dzhulgakov drew the community’s attention to potential result cherry-picking and emphasized the cruciality of clear methodologies in model evaluation, igniting extensive discussion on [this thread](https://x.com/dzhulgakov/status/1818753731573551516).
- **Sybill Secures Millions for AI-Enhanced Selling**: Sybill has secured a potent **$11M Series A** to refine their personal assistant AI for sales reps, with prominent backers like **Greystone Ventures** ([announcement details](https://x.com/asnani04/status/1818642568349204896)).
   - The AI sales tool spectrum is seeing a spark of innovation with Sybill's solution, cloning sales reps' voices to engineer more relevant follow-ups.
- **Black Forest Labs Breaks Ground with FLUX.1**: **Black Forest Labs**, featuring ex-Stable Diffusion wizards, debut their groundbreaking text-to-image model **FLUX.1**, inclusive of a robust **12B parameter version** ([see announcement](https://x.com/iScienceLuvr/status/1819007823339999516)).
   - The pro iteration of FLUX.1 is currently live on [Replicate for trials](https://replicate.com/black-forest-labs/flux-pro), displaying an edge over others in the space.
- **LangGraph Studio Unveils New Horizons for Agentic Apps**: LangChain propels IDE innovation with the launch of **LangGraph Studio**, built to streamline the creation and debugging of agentic applications ([announcement tweet](https://x.com/LangChainAI/status/1819052975295270949)).
   - The agent-focused IDE marries **LangSmith**, boosting efficiency and teamwork for developers in the realm of large language models.
- **Meta MoMa Transforms Mixed-Modal Modeling**: Meta's novel **MoMa architecture** accelerates the pre-training phase for mixed-modal language models, employing a **mixture-of-experts approach** ([accompanying paper](https://arxiv.org/pdf/2407.21770)).
   - The architecture is tailored to juggle and make sense of mixed-modal sequences effectively, marking a step forward in the domain.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Async Advances Accelerate BedrockConverse**: New asynchronous methods for **BedrockConverse** have been integrated, resolving outstanding issues as seen in [pull request #14326](https://github.com/run-llama/llama_index/pull/14326), notably [#10714](https://github.com/run-llama/llama_index/issues/10714) and [#14004](https://github.com/run-llama/llama_index/issues/14004).
   - The community expressed appreciation, highlighting the contribution's significant impact on enhancing user experience with **BedrockConverse**.
- **Insights from the LongRAG Paper**: The **LongRAG** paper, authored by Ernestzyj, introduced techniques for indexing larger document chunks to harness the potential of long-context LLMs.
   - Opening new possibilities, this method simplifies the retrieval-augmented generation process, garnering interest from the community.
- **Workflows Work Wonders in LlamaIndex**: Newly introduced [workflows in llama_index](https://link.to.workflows) empower the creation of event-driven multi-agent applications.
   - The community applauded this innovation for its readable, Pythonic approach to complex orchestration.
- **Stabilizing the Codebase Conundrum**: Conversation revolved around determining the stable version of **LlamaIndex**, clarified by directing users to installations via pip as the safeguard for stability.
   - The term 'stable' emerged as a focal point, associating stability with the most recent releases available on PyPI, sparking further debate.
- **Prompt Playing with DSPy and LlamaIndex**: Members evaluated **DSPy's** prompt optimization against **LlamaIndex's** rewriting features.
   - Enthusiasm was noted for the comparative exploration between these two tools, considering their application in improving prompt performance.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Embed with Zest: Content Structures Clarified**: In a technical discussion, **Nils Reimers** clarified that embedding models automatically remove new lines and special symbols, reinforcing that **preprocessing text** is not essential.
   - This revelation indicates the models’ robustness in handling **noisy data**, allowing AI engineers to focus on model application rather than extensive text preprocessing.
- **Citations Boost Speed; Decay Dilemmas**: A perceptive user linked slower responses with **high citation_quality settings** in Ukrainian/Russian language on Cohere Cloud, noting that shifting from **fast** to **accurate** resolved character issues.
   - While the stable output was attained, the trade-off in response speed has become a topic for potential optimization conversation among engineers.
- **Arabic Dialects in LLMs: A Linguistic Leap**: Surprise was expressed when LLM **Aya** generated accurate text in various **Arabic dialects**, prompting questions about dialect training in an English-based prompt environment.
   - The community's experience with LLMs in dialect handling reinforces the notion of advanced contextual understanding, stoking curiosity about the **training mechanisms**.
- **Devcontainer Dilemma: Pydantic Ponders**: AI engineers faced a bottleneck when **pydantic validation errors** aborted setup of a **Cohere toolkit repository**, highlighting issues in the `Settings` class with missing fields like **auth.enabled_auth**.
   - A swift response from the team promised an imminent fix, demonstrating agility and commitment to toolkit maintenance and usability.
- **"Code and Convene": AI Hackathon Series**: Enthusiasm bubbled as community members discussed participation in the **AI Hackathon Series Tour at Google**, spanning **3 days** of AI innovation and competition.
   - The tour aims to highlight AI advancements and entrepreneurial ventures, culminating in **PAI Palooza**, a showcase of emerging AI startups and projects.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **Pydantic Puzzles in LangChain Programming**: Confusion arose with a **ValidationError** due to a version mismatch of Pydantic, causing type inconsistencies when working with **LangChain**.
   - The conflict was highlighted by input mismatches and validations that led to execution failures, spotlighting the necessity for **api_version** harmony.
- **API Access Angst for LangSmith Users**: A user experienced a `403 Forbidden` error when attempting to deploy an LLM using **LangSmith**, suggesting potential API key misconfiguration.
   - Community discussion circled around the proper setup for the key and seeking assistance through various **LangChain** channels.
- **Streaming Solutions for FastAPI Fabulousness**: Proposing a pattern for asynchronous streaming with **FastAPI** in LangChain applications, a user advocated using Redis for smooth message brokering.
   - This would maintain current synchronous operations while empowering LangChain agents to share outcomes in **real-time**.
- **Jump-Start Resources** for LangChain Learners**: The discourse delved into available resources for mastering LangChain, highlighting alternatives and repositories for effective learning.
   - Members exchanged **GitHub** examples and various API docs to advantageously navigate common deployment and integration puzzles.
- **LangGraph's Blueprints Unveiled**: An innovative LangGraph design pattern was shared, aimed at user-friendly integration into apps like **web-chats** and messenger bots, with a GitHub example showcasing the integration process.
   - Additionally, an invitation was extended for beta testing **Rubik's AI** new features, inclusive of top-tier models like **GPT-4o** and **Claude 3 Opus**, through a special promotional offer.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Digital Detox Diet: Moye's Method**: Moye Launcher's minimalistic design promotes digital wellbeing by intentionally making apps less accessible, championing behavioral shifts towards less screen time.
   - The developer targets three contributors to excess usage, such as auto-clicks and a lack of accountability, aiming to **forge habits for focused app engagement** through design and user feedback.
- **BEAMing Personalities: Big-agi's Big Play**: Big-agi's 'persona creator' lets users **spin up character profiles** from YouTube inputs and the BEAM feature merges outputs of multiple models, increasing response diversity.
   - Still, Big-agi feels the pinch of absent server save and sync functions, hindering an otherwise **smooth model interaction experience**.
- **Msty Merges Memory and Web Mastery**: Msty's integration with Obsidian and website connectivity garners user praise for its ease of use but faces criticism for its forgetful **parameter persistence**.
   - Some users look to swap to Msty despite its need for a polish, thanks to its **sleek interfacing capabilities**.
- **Llama 405B Walks FP16 Tightrope**: OpenRouter lacks a FP16 avenue for Llama 405B, while Meta-recommended FP8 quantization proves **more efficient**.
   - Although SambaNova Systems offers similar services, they're hemmed in by a max 4k context limit and cost-intensive bf16 hosting.
- **OpenRouter's Beta Guarantees Gateway to APIs**: OpenRouter teases an API integration beta, welcoming support emails for rate limit fine-tuning and threading OpenAI and Claude APIs into user endeavours.
   - While its website sometimes stumbles with regional troubles, the [OpenRouter status page](https://status.openrouter.ai/) acts as a beacon, guiding users through operational tempests.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Open Interpreter Stuck in the Slow Lane**: Concern is mounting over **Ben Steinher's delayed response** from Open Interpreter, who missed his mid-July response deadline.
   - Despite the delay, the community lauded a new PR for Groq profile contribution as an impactful way to support Open Interpreter, highlighting a [GitHub PR](https://github.com/OpenInterpreter/open-interpreter/pull/1376) by **MikeBirdTech**.
- **Techies Tune in for Accessibility Talk**: An **Accessibility Roundtable** is set for August 22nd to stir discussion and engagement, with an open invite for the community to share insights.
   - Anticipation is high for the upcoming House Party event, after sorting initial **time-zone tangles**, with participants directed to the [event link](https://discord.gg/zMwXfHwz?event=1267524800163610815).
- **Model Selection Muddles Minds**: Discussion arose about the necessity of an OpenAI API key and the right model string when using '01 --local', evidencing a need for clearer guidelines.
   - Inquisitive threads continue, probing whether **OpenInterpreter** can save and schedule workflows, with answers still pending in the community.
- **iKKO Earbuds Amplifying AI Possibilities**: Buzz is building about integrating OpenInterpreter on **iKKO ActiveBuds**, merging high-resolution audio with AI, as detailed on [iKKO's website](https://www.ikkoaudio.com/collections/tws-earbuds/products/activebuds).
   - Shipment updates for 01 spark urgency within the community, with an unanswered call for updated information as August ticks by.
- **Earbuds with a Vision: Camera Talk**: A novel idea emerged for earbuds equipped with cameras, bolstering interaction by capturing visual context during conversations with LLMs.
   - Community members pondered the integration, contemplating a tap feature to activate the camera for an enhanced HCI experience.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo Misses the Thread**: In a conversation about **Mojo**'s capabilities, a member clarified that **Mojo does not currently expose thread support** directly to users.
   - It was mentioned that utilizing **fork()** is a workaround for achieving threading within the compiled environments.
- **MAX & Mojo's Packing Proclamation**: Upcoming **changes to MAX and Mojo packaging** have been revealed, starting with version 0.9 of the `modular` CLI, dropping the need for authentication to download MAX and Mojo.
   - **Mojo will be merged with MAX nightly builds**, with the [announcement](https://docs.modular.com/max/faq#why-bundle-mojo-with-max) suggesting a shift to the new `magic` CLI for seamless Conda integration.
- **Charting a Tier of Confusion**: Members expressed bewilderment over a tier chart, debating its accurate representation and criticizing it for not reflecting the intended **'level of abstraction'**.
   - Some advocated for simplifying the visual with a fire emoji, indicating the expectation of a clear and effective communication tool.
- **Unicode Unleashed in CrazyString**: The [CrazyString gist](https://gist.github.com/mzaks/78f7d38f63fb234dadb1dae11f2ee3ae) was updated, introducing Unicode-based indexing and boasting **full UTF-8 compatibility**.
   - The conversation touched upon Mojo string's **small string optimization** and the increased usability due to the updates.
- **Max Installation Maze on M1 Max**: Challenges arose for a member attempting to install max on their **Mac M1 Max device**, with the community stepping in to provide potential fixes.
   - A shared resource suggested a [specific Python installation workaround](https://modul.ar/fix-python) could help to navigate the installation issue.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Axolotl's Ascent with Auto-Stopping Algorithms**: Axolotl introduced an **early stopping feature** in response to queries about halting training when **loss plateaus** or **validation loss surges**.
   - Community members engaged in a brief exchange regarding the abilities to manually terminate runs while saving the current **LoRA adapter** state.
- **Masked Learning Leap for SharedGPT**: A member put forward an **"output mask" field** for each turn of SharedGPT, aimed at targeted training through selective output masking.
   - This innovation sparked discussion about its potential to refine learning through processed output errors.
- **Chat Templates Call for Clarity**: Issues with deciphering new **chat templates** prompted members to call for better **documentation** to aid in understanding and customization.
   - A member volunteered to share personal notes on the topic, suggesting a community-driven update to the official documents.
- **Pacing Pad Token Problems**: Training troubles talked about the frequent occurrence of `<pad>` **token repetition**, hinting at inefficiencies in sampling methods.
   - The conversation contributed a tip: ensure pad tokens are cloaked from labels to prevent recurring redundancies.
- **Gemma2's Eager Edge Over Flash**: An endorsed tip for Gemma2 model training surfaced, suggesting 'eager' over 'flash_attention_2' to solidify stability and performance.
   - Practical guidance was given, with [code provided](https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=71bfdef0-8986-4d0c-a882-839872185c7e) to demonstrate setting `eager` attention in `AutoModelForCausalLM`.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Discussions Ignite around DSPy and Symbolic Learning**: Members buzz with anticipation over integrating **DSPy** with symbolic learners, speculating on the groundbreaking potential.
   - Optimism sparks as participants expect substantial advancements from such a combination in AI capabilities.
- **Self-Adapting Agents Step into the Spotlight**: The **Microsoft Research blog** brought self-adapting AI agents to the fore, showcasing an [article](https://www.microsoft.com/en-us/research/blog/tracing-the-path-to-self-adapting-ai-agents/) with promising workplace applications.
   - Insights link the games industry as a catalyst to AI advancement, now materializing in tools like **ChatGPT** and **Microsoft Copilots**.
- **Enter Agent Zero: A Foray into User-Tested AI**: **Agent Zero** makes its debut as the first user-tested production version, showing off its AI prowess.
   - Feedback insinuates a shift towards AI occupying more diverse roles in professional settings.
- **LLMs Self-Improve with Meta-Rewarding**: A new **Meta-Rewarding** technique enhances LLMs' self-judgment, revealed in an [arXiv paper](https://arxiv.org/abs/2407.19594), improving their performance.
   - Significant win rate increases are reported on AlpacaEval 2, indicating that models like **Llama-3-8B-Instruct** also benefit.
- **MindSearch Paper Explores LLM-Based Multi-Agent Frameworks**: A paper published on [arXiv](https://arxiv.org/abs/2407.20183) presents **MindSearch**, emulating human cognitive processes in web searches using LLM-driven agents.
   - The study tackles information seeking challenges and aims to refine modern search-assisted models.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **NVIDIA Grabs Taxpayer Dough**: A message showed enthusiasm for **NVIDIA** receiving public funds, detailing the value for the taxpayer's investment.
   - This topic stirred conversation on investment priorities and implications for tech development.
- **George Hits Hotz Button on Discord Decorum**: **George Hotz** issued a reminder about the server's rules, funneling focus towards **tinygrad development**.
   - Hotz's nudge was a call to maintain a professional and on-topic dialogue within the community.
- **Argmax Chokes GPT-2 Speed**: A deep dive into **GPT-2 performance** found that embedding combined with `argmax` significantly throttles execution speed, as observed in [Issue #1612](https://github.com/tinygrad/tinygrad/issues/1612).
   - The inefficiency traced back to an **O(n^2)** complexity issue, sparking discussions on more efficient algorithmic solutions.
- **Embedding Bounty: Qazalin's Got a Quest**: Talks of a bounty for enhancing **embeddings** in tinygrad surfaced, exclusively directed towards a user named **Qazalin**.
   - The bounty generated buzz and motivated other contributors to seek different optimization opportunities within tinygrad.
- **Cumsum Conundrum**: Challenges with the `cumsum` function's O(n) complexity were tackled in [Issue #2433](https://github.com/tinygrad/tinygrad/issues/2433), inciting innovative thought among developers.
   - George Hotz rallied the troops, advocating for practical experiments to discover possible optimization strategies.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Polyglot ChatGPT's Vocal Feats**: A member showcased [ChatGPT Advanced Voice Mode](https://x.com/CrisGiardina/status/1818799060385489248?t=oe5JjISZYPP6mFqmmJUthg&s=19) adeptly reciting poetry in **Urdu** and storytelling in several languages including **Hebrew, Norwegian, and Georgian**.
   - This display included narratives in lesser-known dialects like **Moroccan Darija, Amharic, Hungarian, Klingon**, wowing the engineering community.
- **Spectacular Reveal of Black Forest Labs**: Enthusiasm erupted over the launch of [Black Forest Labs](https://x.com/robrombach/status/1819012132064669739), with a mission focused on innovative generative models for media.
   - The initiative took off with **FLUX.1**, a model that promises to enhance creativity, efficiency, and diversity in generating visuals.
- **FLUX.1 Model Debuts Impressively**: The community turned their attention to **FLUX.1**, a new model whose debut on [Hugging Face](https://huggingface.co/spaces/black-forest-labs/FLUX.1-schnell) was met with acclaim.
   - Discussions emerged on how this model could potentially shift the landscape of generative learning, with features termed as *refreshing* and *super good*.
- **Innovative Activation Function Twists**: AI enthusiasts delved into experiments with varied **normalization and activation functions** on complex-valued activations, tagging the exercises as 'kinda fun!'.
   - This practical exploration led to sharing of insights and potential applications in complex domains.
- **The Overhyped Regularization Riddle**: A user pointed out, using [a Medium article](https://medium.com/ymedialabs-innovation/data-augmentation-techniques-in-cnn-using-tensorflow-371ae43d5be9), that extensive methods like **data augmentation and dropout** fail to curb overfitting significantly.
   - Probing the effectiveness of various **regularization techniques**, the community pondered on methods beyond traditional tricks to advance machine learning models.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Topping the Charts with Top_p**: A member discovered that setting **top_p=50** met their performance standards with **substantial** results.
   - They compared the **0.8 online model** against their own, noting the online variant's **superior** outcome.
- **Debugging Delight with Generate Recipe**: Clarification was brought that **generate recipe** is geared for debugging purposes, targeting an **accurate portrayal** of the model.
   - Any discrepancies with benchmarks should prompt the submission of an issue, with evaluations affirming the recipe's **efficacy**.
- **FSDP2's New Feature Fusion**: A member shared that **FSDP2** now handles both quantization for **NF4 tensor** and QAT, boosting its **versatility**.
   - While QAT recipes seem **compatible**, compiling with FSDP2 may present challenges, marking an area for potential **refinement**.
- **Merging PRs with Precision**: The upcoming merger of a PR has been flagged as dependent on a prior one, with **PR #1234** under review, thereby paving the way for **sequential improvements**.
   - This anticipates enhanced fine-tuning datasets, with a focus on **grammar and samsum**, advancing Torchtune's **methodical** evolution.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Data Phoenix Ascends with AI Webinar**: The **Data Phoenix** team announced a webinar titled 'Enhancing Recommendation Systems with LLMs and Generative AI,' featuring [Andrei Lopatenko](https://www.linkedin.com/in/lopatenko/) set for August 8 at 10 a.m. PDT.
   - This webinar aims to unveil how **LLMs** and **Generative AI** are transforming personalization engines, with a [webinar registration](https://lu.ma/6i6dtbhf) made available.
- **dlt Elevates ELT Know-how with Workshop**: A 4-hour workshop on **ELT with dlt** is slated to school data enthusiasts on constructing robust ELT pipelines, resulting in a 'dltHub ELT Engineer' certification.
   - Scheduled online for 15.08.2024 at 16:00 GMT+2, the session starts with dlt basics and [registrations can be made here](https://dlthub.com/events).
- **Conferences Showcase NLP & GenAI Dominance**: Two ML conferences placed a heavy accent on **NLP** and **genAI**, overshadowing presentations on models like **Gaussian Processes** and **Isolation Forest**.
   - The trend underscores a strong community tilt towards NLP and genAI technologies, leaving some niche model discussions in the shadows.
- **ROI from genAI Under Community Microscope**: A lively debate questioned whether the **ROI for genAI** will live up to the lofty expectations set by some in the field.
   - The conversation pointed out the gap between expectations and realities, stressing the need for grounded anticipation of returns.



---



## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **LangSmith Credits Conundrum**: **Digitalbeacon** reported an issue accessing LangSmith credits after adding a payment method, using a different email address from his organization ID **93216a1e-a4cb-4b39-8790-3ed9f7b7fa95**.
   - **Danbecker** recommended contacting support for credit-related troubles, implying a need for direct resolution with customer service.
- **Payment Method Mayhem for LangSmith**: **Digitalbeacon** inquired about a zero credit balance in LangSmith post payment method update, even after timely form submission.
   - The situation suggests a system glitch or user misstep, necessitating further investigation or support intervention.



---


The **Alignment Lab AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Mozilla AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **DiscoResearch Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1268649413648318474)** (1 messages): 

> - `Neural network simulation`
> - `Video clustering`
> - `Synthetic dataset`
> - `Knowledge distillation`
> - `Gradio demo` 


- **Simulate Neural Networks Online**: A member shared a [Neural network simulation](https://starsnatched.github.io/) that's now available online.
   - *Explore different neural network configurations and their behaviors in an interactive website*.
- **Master Video Clustering Techniques**: A new [YouTube video](https://www.youtube.com/watch?v=8f4oRcSnfbI) explains how to use image descriptors like Local Binary Pattern (LBP) and Histogram of Oriented Gradients (HOG) for video clustering.
   - *Learn clustering for better video data organization and processing*.
- **Explore Massive Synthetic Dataset**: A huge [synthetic dataset](https://huggingface.co/datasets/tabularisai/oak) was released by a community member.
   - *Perfect for experimenting with tabular data models*.
- **Trendy Knowledge Distillation Techniques**: An insightful article discusses the latest [knowledge distillation trends](https://www.lightly.ai/post/knowledge-distillation-trends) and their implications.
   - *Stay updated on efficient model training methods*.
- **Finance and Medical Models Launch**: New models for [finance and medical](https://x.com/samjulien/status/1818652901130354724) purposes, Palmyra-Med-70b and Palmyra-Fin-70b, have been introduced.
   - Palmyra-Med-70b excels in medical tasks with an **MMLU performance of ~86%**, while Palmyra-Fin-70b is the first model to pass the CFA Level III exam with **73%**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=8f4oRcSnfbI)">🎥 Master Video Clustering with Image Descriptors: LBP &amp; HOG Explained! 🌟</a>: 🔍 Discover the power of video clustering in this detailed guide! Learn how to use image descriptors like Local Binary Pattern (LBP) and Histogram of Oriente...</li><li><a href="https://www.youtube.com/live/dcCn4nuKpBs?feature=share)">Unity ML-Agents | Live Agent training from Scratch</a>: a quick little experiment withing ml agents and cuda</li><li><a href="https://x.com/samjulien/status/1818652901130354724">Tweet from Sam Julien (@samjulien)</a>: 🔥 @Get_Writer just dropped Palmyra-Med-70b and Palmyra-Fin-70b!  Palmyra-Med-70b 🔢 Available in 8k and 32k versions 🚀 MMLU perf ~86%, outperforming top models 👨‍⚕️ For diagnosing, planning treatme...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1268282601886580807)** (852 messages🔥🔥🔥): 

> - `GPTs agents`
> - `Keras introduction`
> - `OpenAI sidebars changes`
> - `Autoencoders for Minecraft`
> - `Fine-tuning models with quantization` 


- **GPTs Agents misunderstood**: Members discussed that GPTs agents do not learn from additional information after their initial training.
   - Clarification was provided that uploaded files are saved as 'knowledge' files for reference but do not modify the base knowledge.
- **Introducing Keras for Deep Learning**: Members provided an explanation of Keras as a multi-backend deep learning framework with support for JAX, TensorFlow, and PyTorch.
   - Keras is praised for accelerating model development and offering state-of-the-art performance with easy-to-debug runtimes.
- **OpenAI platform sidebar changes**: Members discussed the disappearance of two icons from the sidebars of platform.openai.com.
   - It was noted that icons for threads and messages disappeared from the sidebar, prompting further discussion.
- **Autoencoders for Minecraft video generation**: Members worked on training autoencoders to compress Minecraft images and videos with aims of generating Minecraft video sequences.
- **Challenges in Fine-tuning Models with Quantization**: Members addressed issues related to fine-tuning the Llama 3-8b model using quantization to manage GPU memory efficiently.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="http://localhost:11434",">no title found</a>: no description found</li><li><a href="https://blog.fal.ai/flux-the-largest-open-sourced-text2img-model-now-available-on-fal/">Announcing Flux by Black Forest Labs: The Next Leap in Text-to-Image Models</a>: Flux, the largest SOTA open source text-to-image model to date, developed by Black Forest Labs—the original team behind Stable Diffusion is now available on fal. Flux pushes the boundaries of creativi...</li><li><a href="https://x.com/nisten/status/1818529201231688139">Tweet from nisten (@nisten)</a>: hacked bitnet for finetuning, ended up with a 74mb file. It talks fine at 198 tokens per second on just 1 cpu core. Basically witchcraft. opensourcing later via @skunkworks_ai base here: https://huggi...</li><li><a href="https://imgur.com/dd3TB7g">imgur.com</a>: Discover the magic of the internet at Imgur, a community powered entertainment destination. Lift your spirits with funny jokes, trending memes, entertaining gifs, inspiring stories, viral videos, and ...</li><li><a href="https://colab.research.google.com/drive/15md1YRAvT8Hg6fnkEA8BnuNkg-mAajWQ?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://llm.extractum.io/list/?mtr=nroggendorff">Maintainer &laquo;nroggendorff&raquo;</a>: A Curated List of the Large and Small Language Models (Open-Source LLMs and SLMs). Maintainer «nroggendorff» with Dynamic Sorting and Filtering.</li><li><a href="https://huggingface.co/glides">glides (Glide)</a>: no description found</li><li><a href="https://huggingface.co/docs/accelerate/usage_guides/big_modeling">Handling big models for inference</a>: no description found</li><li><a href="https://huggingface.co/spaces/TencentARC/PhotoMaker">PhotoMaker - a Hugging Face Space by TencentARC</a>: no description found</li><li><a href="https://tenor.com/view/tuh-buh-guh-cuh-what-gif-9750912507529527670">Tuh Buh GIF - Tuh Buh Guh - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://pypi.org/project/keras/">keras</a>: Multi-backend Keras.</li><li><a href="https://lu.ma/6i6dtbhf?utm_source=DiscordEvent8">Enhancing Recommendation Systems with LLMs and Generative AI · Luma</a>: The Data Phoenix team invites you to our upcoming webinar, which will take place on August 8 at 10 a.m. PDT. Topic: Enhancing Recommendation Systems with LLMs…</li><li><a href="https://www.tensorflow.org/guide/keras">no title found</a>: no description found</li><li><a href="https://huggingface.co/BioMistral/BioMistral-7B">BioMistral/BioMistral-7B · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/blog/nroggendorff/finetune-mistral">Fine-tuning Mistral on Your Dataset</a>: no description found</li><li><a href="https://huggingface.co/collections/facebook/llm-compiler-667c5b05557fe99a9edd25cb">LLM Compiler - a facebook Collection</a>: no description found</li><li><a href="https://youtu.be/0me3guauqOU">The Unreasonable Effectiveness of JPEG: A Signal Processing Approach</a>: Visit  https://brilliant.org/Reducible/ to get started learning STEM for free, and the first 200 people will get 20% off their annual premium subscription.Ch...</li><li><a href="https://youtu.be/4VAkrUNLKSo">Computer Generates Human Faces</a>: 5:51 To skip to the results.Try It Online: http://codeparade.net/faces/Download App (Windows 64-bit): https://github.com/HackerPoet/FaceEditor/raw/master/Fac...</li><li><a href="https://github.com/noamgat/lm-format-enforcer/blob/main/README.md">lm-format-enforcer/README.md at main · noamgat/lm-format-enforcer</a>: Enforce the output format (JSON Schema, Regex etc) of a language model - noamgat/lm-format-enforcer</li><li><a href="https://youtu.be/NTlXEJjfsQU">Creating my own customized celebrities with AI</a>: Check out Brilliant.org for fun STEMmy courses online! First 200 people to sign up here get 20% off their annual premium subscription cost: https://brilliant...</li><li><a href="https://youtu.be/Dt2WYkqZfbs">Why images are compressible: The Vastness of Image Space</a>: We explore why images are compressible, which is related to the (larger than) astronomical space of all possible images.  This is one of my favorites. Follow...</li><li><a href="https://open.spotify.com/track/6y5HLopYu7Uu0hYwVBj4T6">palm of my hands</a>: Song · John Summit, venbee · 2024</li><li><a href="https://github.com/jxnl/instructor">GitHub - jxnl/instructor: structured outputs for llms</a>: structured outputs for llms . Contribute to jxnl/instructor development by creating an account on GitHub.</li><li><a href="https://esolangs.org/wiki/Chicken">Chicken - Esolang</a>: no description found</li><li><a href="https://github.com/huggingface/diffusers/pull/9043">Flux pipeline by sayakpaul · Pull Request #9043 · huggingface/diffusers</a>: We are working on uploading the diffusers weights to the respective FLUX repositories. Will be done very soon.
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1268483912267599872)** (4 messages): 

> - `finegrain Object Eraser model`
> - `Evolution of AI bots`
> - `Knowledge distillation` 


- **Finegrain unveils Object Eraser model**: A member shared news of a new **Object Eraser model** available on a [Hugging Face space](https://huggingface.co/spaces/finegrain/finegrain-object-eraser), demonstrating the model's capabilities.
   - This model was developed by **@finegrain_ai** and is aimed at showcasing new applications publicly for everyone to try.
- **Evolution of AI bots article on Medium**: A member posted an article on Medium about the **Evolution of AI bots**, detailing various AI tools like **LLMs** and **RAG** pipelines. [Read the full article](https://medium.com/@qdrddr/evolution-of-the-ai-bots-harnessing-the-power-of-agents-rag-and-llm-models-4cd4927b84f8).
   - The article is designed for newcomers and delves into high-level patterns, pipelines, and architectural designs used in **2024**.
- **Understanding Knowledge Distillation**: A member found knowledge distillation to be an interesting topic, sharing a detailed page from IBM on [Knowledge Distillation](https://www.ibm.com/topics/knowledge-distillation).
   - The article explains that **knowledge distillation** transfers learnings from a large pre-trained 'teacher model' to a smaller 'student model' for compression and knowledge transfer purposes.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/pchapuis/status/1818632367138885826">Tweet from Pierre Chapuis (@pchapuis)</a>: Made a @huggingface space to demonstrate one of the models we trained at @finegrain_ai: the Object Eraser. Pretty happy everyone can try it publicly at last. :)  https://huggingface.co/spaces/finegrai...</li><li><a href="https://www.ibm.com/topics/knowledge-distillation">What is Knowledge distillation? | IBM </a>: Knowledge distillation is a machine learning technique used to transfer the learning of a large pre-trained “teacher model” to a smaller “student model.”</li><li><a href="https://medium.com/@qdrddr/evolution-of-the-ai-bots-harnessing-the-power-of-agents-rag-and-llm-models-4cd4927b84f8">Evolution of the AI Bots: Harnessing the Power of Agents, RAG, and LLM Models</a>: Structuring knowledge about tools for AI bot development, also high-level overview of approaches, architectures and designs.
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1268285446937514107)** (16 messages🔥): 

> - `model release heatmap`
> - `grounding-sam2-demo`
> - `TinyML bird detection project`
> - `Infinite Sands project`
> - `2D parallelism in deep learning` 


- **Model release heatmap space gains attention**: A member created a [space for a heatmap of model releases among top AI labs](https://huggingface.co/spaces/cfahlgren1/model-release-heatmap).
   - Others expressed interest in integrating such a heatmap into future Hugging Face profile pages for better visibility.
- **Grounding-Sam2 demo showcases paired models**: A member shared a [GitHub project](https://github.com/CoffeeVampir3/grounding-sam2-demo) demonstrating a Gradio interface for grounding dino and segment anything v2 models.
   - The demo highlights upgraded usage of these models in a simple and interactive format.
- **TinyML detects birds with Seeed and Blues**: A project on [Hackster](https://www.hackster.io/timo614/bird-detection-with-tinyml-and-a-blues-notecard/) reports bird species using TinyML hardware and a Blues Notecard.
   - The setup involves Seeed's Grove Vision AI Module V2 and compresses EfficientNetLite for efficient bird detection.
- **Infinite Sands brings sandbox to life with AI**: [Infinite Sands](https://www.hackster.io/sand-command/infinite-sands-df675a) uses generative AI to create stories from sandbox shapes.
   - The project applies ControlNet depth and Whisper for command handling, making it a playful and interactive exploration.
- **AI + i podcast launches focused on AI models**: A new podcast series, [Ai + i](https://youtube.com/@aiplusi), has been launched to discuss leading foundation and open-source models.
   - The host seeks topic suggestions from the community for future podcast episodes.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/cfahlgren1/model-release-heatmap">Model Release Heatmap - a Hugging Face Space by cfahlgren1</a>: no description found</li><li><a href="https://malaysia-ai.org/2d-parallelism-ray">Malaysia-AI blog 2D Parallelism using Ray PyTorch</a>: Malaysia-AI blog 2D Parallelism using Ray PyTorch</li><li><a href="https://huggingface.co/spaces/cfahlgren1/model-release-heatmap/discussions?status=open&type=discussion">cfahlgren1/model-release-heatmap · Discussions</a>: no description found</li><li><a href="https://huggingface.co/tasksource/deberta-base-long-nli">tasksource/deberta-base-long-nli · Hugging Face</a>: no description found</li><li><a href="https://github.com/CoffeeVampir3/grounding-sam2-demo/blob/main/interface.py">grounding-sam2-demo/interface.py at main · CoffeeVampir3/grounding-sam2-demo</a>: A simple demo for utilizing grounding dino and segment anything v2 models together - CoffeeVampir3/grounding-sam2-demo</li><li><a href="https://www.hackster.io/timo614/bird-detection-with-tinyml-and-a-blues-notecard-b8b705">Bird Detection with TinyML and a Blues Notecard</a>: I built a project to identify birds at a bird feeder using Machine Learning (TinyML) and transmit data to the cloud with a Blues Notecard By Timothy Lovett and Kerin Lovett.</li><li><a href="https://www.hackster.io/sand-command/infinite-sands-df675a">Infinite Sands</a>: ROCM powered sandbox to shape your reality with the help of standard diffusion and controlnet. By Timothy Lovett and Kerin Lovett.
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1268408183026618368)** (8 messages🔥): 

> - `Deep Learning Study Group`
> - `LLM Model Suggestions`
> - `New Learners Collaboration` 


- **Deep Learning Enthusiasts Unite**: A new member expressed interest in forming a group of motivated individuals to learn **deep learning and machine learning** together.
- **LLM Model for PDF Table and Checkbox Detection**: A member requested suggestions for **LLM models** capable of performing table and checkbox detection and extraction from PDF inputs.


  

---


### **HuggingFace ▷ #[core-announcements](https://discord.com/channels/879548962464493619/1014557141132132392/)** (1 messages): 

sayakpaul: Will be merged in a few https://github.com/huggingface/diffusers/pull/9043
  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1268576936569864192)** (2 messages): 

> - `Training LLM for Solr`
> - `AI System for Aphasic Patients` 


- **Training LLM to interpret search queries for Solr**: A member asked for advice on training a **Large Language Model (LLM)** to receive search queries and output JSON with **product facets and categories** for use in [Apache Solr](https://solr.apache.org/).
   - *They mentioned not having an instruction dataset* and sought guidance on how to approach the task.
- **Building AI for Communication with Aphasic Patients**: A member intends to build an AI system combining **microexpression recognition, speech recognition, and image recognition** to help facilitate communication with aphasic patients.
   - They requested help as they have no idea how to start the project and mentioned that *anything would be extremely helpful*.


  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1268347706171068558)** (2 messages): 

> - `Amazing Results`
> - `Trolling Allegations` 


- **Welltoobado Praises Results**: A member expressed satisfaction, noting *'Yeah pretty amazing results, good job!'* in response to something.
- **Pseudoterminalx Questions Trolling**: Another member, uncertain about the sincerity, responded, *'hard to tell if you're trolling anymore lol'*.


  

---



### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/)** (1 messages): 

pradeep1148: https://www.youtube.com/watch?v=DLb7Lrzw8wo
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1268628172216340562)** (9 messages🔥): 

> - `New SOTA efficiency gains in Multi-modal architecture` 


- **New SOTA efficiency gains in Multi-modal architecture**: The authors who introduced **Chameleon** achieved significant efficiency gains in a [new multi-modal architecture](https://arxiv.org/pdf/2407.21770), incorporating a mixture of experts and modal-specific expert routing techniques.
   - Efficiency gains were *approximately 3x in text training* and *5x in image training*, with MoMa 1.4B significantly outperforming its dense counterpart and other MoE models according to [Victoria Lin](https://x.com/VictoriaLinML/status/1819037439681565178).
- **Discussion on New SOTA efficiency gains in Multi-modal architecture**: Members expressed excitement about the new architecture, noting its significant FLOPs savings and improved performance.
   - The gains in **image training** were particularly noted, highlighting the new architecture's impressive **5.2x** efficiency improvement.



**Link mentioned**: <a href="https://x.com/VictoriaLinML/status/1819037439681565178">Tweet from Victoria X Lin (@VictoriaLinML)</a>: 4/n Under a 1T token training budget, MoMa 1.4B (4 text experts+4 image experts) achieves FLOPs savings of 3.7x (text: 2.6x, image: 5.2x) compared to its dense counterpart (measured in pre-training lo...

  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1268282496374542463)** (441 messages🔥🔥🔥): 

> - `Heptagon Riddle`
> - `GPT Benchmarks vs Human Heuristics`
> - `Speculative Decoding Mechanics`
> - `Dynamic Memory Systems`
> - `Bitnet for Finetuning` 


- **Heptagon Riddle Solved**: **A riddle about a denizen of flatland** involves determining a regular polygon type. After discussion, **heptagon** was the correct answer.
   - *One user* noted some models occasionally get lucky answers, but overall, **the LLMs struggle with symbolic logic riddles**.
- **Speculative Decoding Insights**: Participants discussed **speculative decoding** techniques, explaining that using smaller draft models to speed up decoding isn't always lossless.
   - While some initial claims stated that output distribution can diverge if not done correctly, others clarified that **rejection sampling** ensures lossless output by aligning draft and base models.
- **Dynamic Memory System Applications**: **Dynamic persona memories** were discussed as a current gap in the ragdata set, with participants suggesting collaboration opportunities.
   - Participants compared techniques to **parallelize token generation** and noted issues with **accurate context handling** by LLMs in dynamic systems.
- **Bitnet's Finetuning Brings Speed**: A Reddit post about **Bitnet's finetuning method** received attention due to its impressive speed, running at **198 tokens per second** on just one CPU core.
   - Experimenters achieved a **74MB file size** using Bitnet and claimed it operates efficiently, sparking interest in its **potential for future projects**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/nisten/status/1818529201231688139">Tweet from nisten (@nisten)</a>: hacked bitnet for finetuning, ended up with a 74mb file. It talks fine at 198 tokens per second on just 1 cpu core. Basically witchcraft. opensourcing later via @skunkworks_ai base here: https://huggi...</li><li><a href="https://x.com/ryunuck/status/1818709409239121975?s=46">Tweet from ryunuck (p≈np) (@ryunuck)</a>: What Ilya saw  CRISPR-Q runs on Sonnet 3.5 and enables the model to rewrite the context window through targeted operations of its own self-memeplex. The incomprehensibly alien generative heuristic tha...</li><li><a href="https://huggingface.co/openai-community/gpt2-xl">openai-community/gpt2-xl · Hugging Face</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=hm7VEgxhOvk">Speculative Decoding Explained</a>: One Click Templates Repo (free): https://github.com/TrelisResearch/one-click-llmsAdvanced Inference Repo (Paid Lifetime Membership): https://trelis.com/enter...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1ehh9x2/hacked_bitnet_for_finetuning_ended_up_with_a_74mb/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/holo-q/OpenQ/">GitHub - holo-q/OpenQ: The open-source implementation of Q*, achieved in context as a zero-shot reprogramming of the attention mechanism. (synthetic data)</a>: The open-source implementation of Q*, achieved in context as a zero-shot reprogramming of the attention mechanism. (synthetic data) - holo-q/OpenQ</li><li><a href="https://github.com/carsonpo/octoquadmul">GitHub - carsonpo/octoquadmul</a>: Contribute to carsonpo/octoquadmul development by creating an account on GitHub.</li><li><a href="https://github.com/carsonpo/octomul">GitHub - carsonpo/octomul: Reasonably fast (compared to cublas) and relatively simple int8 tensor core gemm</a>: Reasonably fast (compared to cublas) and relatively simple int8 tensor core gemm - carsonpo/octomul
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1268591738939768913)** (2 messages): 

> - `LangChain usage`
> - `Mixtral API retrieval`
> - `OpenAI API format` 


- **Using LangChain with Mixtral API in OpenAI format**: A member discussed a code snippet using **LangChain** with environment variables like **mixtral_api_base** for retrieving the Mixtral LLM from the OpenAI API.
   - There was a debate on whether this approach makes sense without **LangChain**, since LangChain uses the OpenAI API format.
- **Debate on LangChain necessity**: Another discussion ensued regarding whether the use of **LangChain** is necessary for interacting with the Mixtral LLM from OpenAI API.
   - Members expressed differing views on the dependency on LangChain for such operations.


  

---


### **Nous Research AI ▷ #[reasoning-tasks-master-list](https://discord.com/channels/1053877538025386074/1264666760972472481/1268311059614269553)** (3 messages): 

> - `Assisting with project setup`
> - `Cost considerations for project` 


- **Project Setup Assistance**: A member asked what they could do to help get the project going and if it costs much.
   - Another member confirmed that it doesn't cost anything and instructed them to follow the steps mentioned in a pending PR.
- **Cost-Free Project Initiative**: A participant mentioned that the project does not incur any costs.
   - The next steps involve following the instructions provided once a new PR is made.


  

---



### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1268282183496237217)** (205 messages🔥🔥): 

> - `Multi-GPU Support`
> - `Unsloth Finetuning`
> - `Qwen Model Merging`
> - `AI Performance`
> - `Bitnet Code Hacking` 


- ****Multi-GPU Training** works but needs improvement**: Users confirmed multi-GPU training works after fixes, but noted earlier installation problems required creating a new environment and troubleshooting various setups.
   - An example stated: *'installing it into llamafacs env worked first try,'* while another mentioned needing to manually upgrade transformers.
- **Unsloth Crypto Runner Clarifications**: Clarifications were provided on the **Unsloth Crypto Runner**, stating it involves **AES/PKI-based cryptography** between client and license server.
   - *'MrDragonFox'* emphasized, 'what you need to care about is the right side as you see my both GPU's utilized.'
- **Finetuning Qwen with Continuous Fine-tuning**: Using **Continuous Fine-tuning Without Loss** on Qwen2-1.5B-Instruct was successful, incorporating both code FIM and instruct capabilities.
   - Members were excited about the method, with one suggesting *'writing up a tutorial'* for those facing confusion over the documentation.
- **Issues with Merging Adapters**: Users discussed merging **LoRA adapters** and 4-bit models, noting that improperly merging could lead to models only appearing as 16-bit but actually being 4-bit quality.
   - A concern was raised about **4-bit models being upscaled to 16-bit**, potentially leading fake 16-bit models to propagate in the community.
- **Hack on Bitnet for Finetuning**: User **Nisten** mentioned hacking Bitnet for finetuning, resulting in a **74MB model** that runs at **198 tokens per second on 1 CPU core**.
   - This hack was described as *'basically witchcraft'* and will be **open-sourced** via **Skunkworks AI**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/nisten/status/1818529201231688139">Tweet from nisten (@nisten)</a>: hacked bitnet for finetuning, ended up with a 74mb file. It talks fine at 198 tokens per second on just 1 cpu core. Basically witchcraft. opensourcing later via @skunkworks_ai base here: https://huggi...</li><li><a href="https://huggingface.co/johnpaulbin/qwen1.5b-e2-1-lora">johnpaulbin/qwen1.5b-e2-1-lora · Hugging Face</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1obulZ1ROXHjYLn6PPZJwRR6GzgQogxxb?usp=sharing)">Google Colab</a>: no description found</li><li><a href="https://huggingface.co/rombodawg/gemma-2-9b-reuploaded">rombodawg/gemma-2-9b-reuploaded · Hugging Face</a>: no description found</li><li><a href="https://x.com/dejavucoder/status/1818707409264861348">Tweet from sankalp (@dejavucoder)</a>: wake up babe, daniel han video finally dropped</li><li><a href="https://tenor.com/view/dancing-dj-ravine-groovy-mixing-music-party-gif-21277620">Dancing Dj Ravine GIF - Dancing Dj Ravine Groovy - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://unsloth.ai/blog/llama3">Finetune Llama 3 with Unsloth</a>: Fine-tune Meta&#x27;s new model Llama 3 easily with 6x longer context lengths via Unsloth!</li><li><a href="https://datta0.substack.com/p/ai-unplugged-16-llama-3-aimo-winners">AI Unplugged 16: Llama 3, AIMO winners, Segment Anything Model 2, LazyLLM</a>: Insights over Information</li><li><a href="https://youtu.be/pRM_P6UfdIc?feature=shared">Low Level Technicals of LLMs: Daniel Han</a>: This workshop will be split into 3x one hour blocks:How to analyze &amp; fix LLMs - how to find and fix bugs in Gemma, Phi-3, Llama &amp; tokenizersFinetuning with U...</li><li><a href="https://mer.vin/2024/07/llama-3-1-fine-tune/">Llama 3.1 Fine Tune - Mervin Praison</a>: https://huggingface.co/mervinpraison/Llama-3.1-8B-bnb-4bit-python Train Model with Custom Data Convert to GGUF Ollama Modelfile Ollama Create Custom Model
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1268651575664971787)** (4 messages): 

> - `Google new model`
> - `OpenAI vs Google` 


- **Google's new model beats OpenAI**: *Finally Google beat OpenAI* with a [new model](https://www.reddit.com/r/ChatGPT/comments/1ehlmqd/finally_google_beat_openai_new_model_from_google/).
   - A user shared a link to Reddit highlighting the **new model from Google** that claims to surpass OpenAI.
- **Users react skeptically**: *I can't believe it...* was the initial reaction to the purported news from Google.
   - Another user responded with skepticism saying, *ummm*, casting doubt on the credibility of the information.



**Link mentioned**: <a href="https://www.reddit.com/r/ChatGPT/comments/1ehlmqd/finally_google_beat_openai_new_model_from_google/">Reddit - Dive into anything</a>: no description found

  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1268282102487715891)** (130 messages🔥🔥): 

> - `Python versions for Unsloth installation`
> - `Installing Unsloth with Conda`
> - `LoRA fine-tuning issues`
> - `Inference problems with GGUF quantization`
> - `Custom dataset training errors on Llama 3.1` 


- ****Python versions spark debate****: Members were confused about Unsloth's compatibility with Python versions 3.10 and 3.11, as different results appeared when following the installation guide.
   - Felicitiy00637 shared issues with installation on Compute Canada's Narval cluster, noting success only after bypassing xforms in 'pyproject.toml'.
- ****Conda environment clarifies setup****: Fjefo stressed the importance of following the guide precisely for Conda environments, noting that deviations could complicate debugging.
   - Despite felicity00637's assurance of following the guide, confusion persisted until confirmation that Conda wasn't used.
- ****LoRA parameters under discussion****: Felicitiy00637 sought clarification on LoRA parameters like 'r' and 'lora_alpha', asking for their definitions and recommended values.
   - The community explained that LoRA scaling parameters should ideally be set to twice the rank (r), linking to the [LoRA parameter encyclopedia](https://docs.unsloth.ai/basics/lora-parameters-encyclopedia) for deeper insights.
- ****GGUF quantization wreaks havoc****: Akshatiscool reported models outputting gibberish post-GGUF quantization, despite correct outputs during Collab inference.
   - Theyruinedelise suggested checking chat templates, acknowledging recent issues fixed in GGUF quantization.
- ****Llama 3.1 training stumbles****: Bigboypikachu encountered 'Expected all tensors to be on the same device' errors when training custom long-context datasets on Llama 3.1-8b-instruct.
   - The same kernel successfully trained on a predefined dataset, but failed with custom datasets, hinting at context length issues.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/1wlCOvklww1YvACuIRrhkdFFH_vU7Hgbn?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://docs.unsloth.ai/basics/continued-pretraining">Continued Pretraining | Unsloth Documentation</a>: AKA as Continued Finetuning. Unsloth allows you to continually pretrain so a model can learn a new language.</li><li><a href="https://docs.alliancecan.ca/wiki/Narval/en">Narval - Alliance Doc</a>: no description found</li><li><a href="https://docs.unsloth.ai/basics/saving-models">Saving Models | Unsloth Documentation</a>: Learn how to save your finetuned model so you can run it in your favorite inference engine.</li><li><a href="https://youtu.be/TKmfBnW0mQA?t=740">Fixing bugs in Gemma, Llama, &amp; Phi 3: Daniel Han</a>: The story behind our 8 bug fixes for Gemma, multiple tokenization fixes for Llama 3, a sliding window bug fix and Mistral-fying Phi-3, and learn about how we...</li><li><a href="https://github.com/unslothai/unsloth/issues/839">FastLanguageModel has a problem with PromptTemplate and other complicate things · Issue #839 · unslothai/unsloth</a>: I am trying to specify the prompt to apply RAG in Unsloth environment but Unfortunately, current Unsloth environment has some complicated problems. First, I will provide slow but well-worked code. ...</li><li><a href="https://docs.unsloth.ai/basics/lora-parameters-encyclopedia>">Unsloth Documentation</a>: no description found</li><li><a href="https://github.com/huggingface/transformers/issues/27985">`KeyError: &#39;Cache only has 0 layers, attempted to access layer with index 0&#39;` · Issue #27985 · huggingface/transformers</a>: System Info transformers version: 4.36.0 Platform: Linux-5.15.0-70-generic-x86_64-with-glibc2.35 Python version: 3.11.4 Huggingface_hub version: 0.19.4 Safetensors version: 0.3.3 Accelerate version...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1268524354942144543)** (5 messages): 

> - `AI interoperability with Groq`
> - `Black Forest Labs launch`
> - `FLUX.1 text-to-image model`
> - `OpenAI models`
> - `Generative AI` 


- **Groq AI limited to inference post-finetuning**: Members discussed whether AI models can work on both **Google AI** and **Groq AI**.
   - It was clarified that with Groq, models can most likely only do **inference** after being **finetuned** using another service.
- **Black Forest Labs steps into the scene**: Announcing [Black Forest Labs](https://blackforestlabs.ai/announcing-black-forest-labs/), a new venture focused on advancing generative deep learning models for media.
   - Their initial release, the **FLUX.1 suite of models**, aims to push the frontiers of **text-to-image synthesis**. *Open weights* make it accessible for further development.



**Link mentioned**: <a href="https://blackforestlabs.ai/announcing-black-forest-labs/">Announcing Black Forest Labs</a>: Today, we are excited to announce the launch of Black Forest Labs. Deeply rooted in the generative AI research community, our mission is to develop and advance state&#x2d;of&#x2d;the&#x2d;art generati...

  

---



### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1268663522389131380)** (1 messages): 

> - `Perplexity Pro free for Uber One members` 


- **Uber One offers Perplexity Pro for free**: Uber One members across the US and Canada can now enjoy a free year of **Perplexity Pro**. This offer, available until **October 31**, allows members to unlock the full potential of Perplexity’s **answer engine**, normally valued at **$200**.
- **Enhance info discovery with Perplexity Pro**: From quick facts during Uber rides to detailed research at home, **Perplexity Pro** enhances every information discovery moment for Uber One members.
   - Learn more about this perk and the terms at [Perplexity Uber One](https://pplx.ai/uber-one).



**Link mentioned**: <a href="https://pplx.ai/uber-one">Eligible Uber One members can now unlock a complimentary full year of Perplexity Pro&nbsp;</a>: Uber One members can now save even more time with perks like Pro Search

  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1268293825009418292)** (293 messages🔥🔥): 

> - `Uber One Perplexity Pro deal`
> - `Rating AI search engines`
> - `Perplexity functionality comparisons`
> - `Technical issues and bugs`
> - `Legal use cases for AI` 


- **Uber One members get Perplexity Pro for free**: Perplexity announced that eligible Uber One members in the US and Canada can redeem a complimentary year of Perplexity Pro from now through October 31, 2024. Members discussed details and eligibility, noting the promotion requires signing up with a new Perplexity Pro account and maintaining an active Uber One membership throughout.
- **Comparing different AI search engines**: Users shared their experiences comparing various AI search engines like Perplexity, Felo.ai, and Chatlabs, focusing on aspects like UI, UX, speed, and response quality. **Perplexity Pro** was generally rated highest, followed by **SearchGPT**, **Uncovr free**, and others.
- **Perplexity app functionality issues and gaps**: Members highlighted several issues with Perplexity's app, especially on mobile, such as the inability to delete uploaded files and generate images, poor Android performance, and significant missing features compared to OpenAI and Microsoft Copilot. One user expressed their frustration with mobile bugs and inconsistencies that lead to lost text.
- **Troubleshooting exporting and uploading issues**: Users encountered issues with exporting text and sources from pages, with one noting: *'Truly IMPOSSIBLE. Impossible. Never going to happen.'* Another member reported token count errors when trying to upload large PDFs in AIStudio.
- **Using AI for legal document search and analysis**: A member shared their positive experience using Perplexity for searching and analyzing legal documents, finding it particularly useful for locating relevant cases. They inquired about applying **Retrieval-Augmented Generation (RAG)** to search through a large collection of discovery documents.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://labs.writingmate.ai">ChatLabs</a>: ChatLabs is a platform for LLM and AI tinkerers. Experience more than 30 AI models in one place.</li><li><a href="https://www.perplexity.ai/hub/blog/eligible-uber-one-members-can-now-unlock-a-complimentary-full-year-of-perplexity-pro">Eligible Uber One members can now unlock a complimentary full year of Perplexity Pro&nbsp;</a>: Uber One members can now save even more time with perks like Pro Search</li><li><a href="https://gitlab.com/monnef/ailin">monnef / AIlin · GitLab</a>: AIlin is a tool that connects AI services, such as Perplexity.ai, with your local computer.</li><li><a href="https://www.perplexity.ai/page/Complexity-Perplexitys-New-yl0q3mHYQz6RhRyuvjvN4w">Complexity: Perplexity&#x27;s New Extension</a>: The Complexity extension for Perplexity AI introduces a range of powerful features designed to enhance the user experience and streamline interactions with...
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1268285574679101440)** (10 messages🔥): 

> - `Perplexity AI skills and features`
> - `Flask secure user authentication`
> - `Checking Pro account status`
> - `Impacts of drinking coffee on dental health`
> - `Next iPhone release details` 


- **Perplexity AI combines search and text generation**: [Perplexity AI](https://www.perplexity.ai/search/what-is-best-skills-in-perplex-mvRHkNtwTHGP7MIk0q3akA) is a powerful tool that integrates search capabilities with large-scale language models to provide precise and comprehensive answers.
   - Its notable features include effective **market research** and **competitive analysis**, helping users to synthesize data from multiple reports and understand competitive landscapes.
- **Flask secure user authentication setup**: To implement secure user authentication in Flask, install necessary packages like `Flask-Login`, `Flask-SQLAlchemy`, and `Flask-Bcrypt`, and follow step-by-step guidelines.
   - This involves creating an application factory, defining a `User` model, and setting up routes for registration, login, and logout as demonstrated [here](https://www.perplexity.ai/search/please-provide-an-example-of-s-EvlJDJwUTfy4IWmobEm0Fw).
- **Check Pro account status with steps**: To check if an account is subscribed to Pro, navigate to account settings or billing information on the platform.
   - Alternatively, verify through payment history, or contact customer support for assistance, as detailed [here](https://www.perplexity.ai/search/na-porinde-wae-giboneuro-doeji-MGecFR96SpuhbQ04SrRLfA).
- **OpenAI rolls out hyper-realistic voice mode**: [OpenAI](https://www.perplexity.ai/page/openai-begins-hyper-realistic-2_y7h8vPQEWaM4g63WvnVA) launched its Advanced Voice Mode for ChatGPT, giving Plus subscribers access to hyper-realistic audio interactions powered by the GPT-4o model on July 30, 2024.
   - This feature introduces real-time, natural conversations with capabilities like mid-sentence interruptions and emotional intonation detection.
- **Folksable app enhances habit tracking with social features**: Folksable is a habit tracking app that encourages users to share updates and progress with friends through photos and social contracts to maintain accountability.
   - [Available on Android and iOS](https://www.perplexity.ai/search/folksable-app-tb4hRa_OSrmapF0bjyYs1g), the app ensures data privacy with encryption and allows users to create custom rituals and manage visibility through privacy controls.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.perplexity.ai/search/does-drinking-coffee-have-any-p3oI92DpQc6bMClukg.x8Q">does drinking coffee have any negative impact on dental health?</a>: Drinking coffee can have several negative impacts on dental health.  Coffee contains acidity, which can erode tooth enamel and cause sensitivity and brittle...</li><li><a href="https://www.perplexity.ai/search/folksable-app-tb4hRa_OSrmapF0bjyYs1g">Folksable app</a>: Folksable is a photo habit tracking app that encourages users to share updates on their habits with friends for accountability. Users can create private or...</li><li><a href="https://www.perplexity.ai/search/i-m-curious-about-the-shooting-_W0sHhfITLy9jLaqjCFU9A">I&#x27;m curious about the shooting posture of Olympic shooters and tell me why...</a>: Olympic shooters adopt specific postures and stances to maximize their accuracy, stability, and performance. Here&#x27;s an overview of the shooting postures used...</li><li><a href="https://youtu.be/ak19RS6VGGU?si=DLWE5mVmvDeyzLPO">Synchron&#39;s Brain Implant Advances, OpenAI&#39;s Voice Mode, Friend&#39;s AI Necklace, and HIV Nanobody Br...</a>: Discover the latest breakthroughs in brain-computer interface technology as Synchron enables mind control of Apple&#39;s Vision Pro headset. This groundbreaking ...</li><li><a href="https://www.perplexity.ai/search/na-porinde-wae-giboneuro-doeji-MGecFR96SpuhbQ04SrRLfA">나 por인데 왜 기본으로 되지?</a>: 안녕하세요. 제가 이해한 바로는 귀하께서 한국어로 대화를 시작하셨는데, 시스템이 기본적으로 영어(POR)로 설정되어 있는 것 같습니다. 이런 상황이 발생한 이유에 대해 몇 가지 가능성을 설명해 드리겠습니다:  1. 시스템 설정: 사용 중인 애플리케이션이나 웹사이트의 기본 언어 설정이...</li><li><a href="https://www.perplexity.ai/search/when-will-the-next-iphone-be-r-3SV74RalRFqKPIK6m4XgtQ#0">When will the next iPhone be released?</a>: The next iPhone, expected to be the iPhone 16, is anticipated to be released in September 2024. This follows Apple&#x27;s typical release pattern for new iPhone...</li><li><a href="https://www.perplexity.ai/page/openai-begins-hyper-realistic-2_y7h8vPQEWaM4g63WvnVA">OpenAI Begins Hyper-realistic Voice Rollout</a>: OpenAI has begun rolling out its highly anticipated Advanced Voice Mode for ChatGPT, offering select Plus subscribers access to hyper-realistic audio...</li><li><a href="https://www.perplexity.ai/search/please-provide-an-example-of-s-EvlJDJwUTfy4IWmobEm0Fw">please provide an example of secure user authentication in Flask</a>: To implement secure user authentication in a Flask application, you can follow these steps, which include setting up the necessary packages, creating a user...</li><li><a href="https://www.perplexity.ai/search/what-is-best-skills-in-perplex-mvRHkNtwTHGP7MIk0q3akA">What is best skills in PerplexitAI ?</a>: Perplexity AI é uma ferramenta poderosa que combina capacidades de busca e geração de texto, utilizando modelos de linguagem de grande escala (LLMs) para...
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1268284762758451251)** (4 messages): 

> - `Subpar Prompt Results`
> - `Perplexity References Beta`
> - `Perplexity API on make.com` 


- **Users call out subpar prompt results**: Users expressed concerns over recent prompt results, indicating they feel like the results are going backwards.
   - One user asked for suggestions on specific prompts that might be causing the issue.
- **Inquire about Perplexity References Beta access**: A user inquired about the status of the Perplexity references beta, wondering if it’s still possible to gain access.
   - *'Hey there, I've applied for the perplexity references beta and was wondering if those are still being given out or if there is a way for me to get there? 🙂'*.
- **Integrating Perplexity API on make.com**: A user inquired about connecting to Perplexity API on make.com, specifying the use of Sonnet 3.5 model to generate summaries.
   - The user outlined a requirement to generate a page with a model on Perplexity API and then post the link on Discord.


  

---



### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1268314148882747582)** (255 messages🔥🔥): 

> - `GPT-4o Image Output`
> - `Multimodal Training Models`
> - `Voice Model Testing`
> - `DALL-E and Imagen 3 Comparisons`
> - `Alpha Testing Experience` 


- **GPT-4o Image Output Debated**: Discussion centered around GPT-4o's image output capabilities with [examples](https://x.com/gdb/status/1790869434174746805), comparing it to other models like DALL-E 3.
   - Users noted that GPT-4o's output seemed more realistic but faced criticisms over its moderation endpoint similar to DALL-E 3.
- **Future of Multimodal Training Models**: A user proposed the future relevance of multimodal models that learn indirectly from video data to label emotions, suggesting they might outperform single-modality models for tasks like text to speech.
- **Voice Model Testing and Capabilities**: Users experimented with the [voice capabilities](https://platform.openai.com/docs/guides/embeddings/use-cases) of GPT-4o, sharing various scenarios including accent changes and emotional expressions.
   - Findings highlighted the model's ability to add background music and sound effects, though it was inconsistent.
- **Comparing DALL-E and Imagen 3**: Requests and comparisons were made between DALL-E and Imagen 3, with offers to run prompts to see which produced better imagery.
   - Initial feedback suggested that while both had strong capabilities, Imagen 3 might have a moderation endpoint issue.
- **Experiences and Limitations of Alpha Testing**: Alpha testers shared mixed experiences, noting issues like high latency and occasional connectivity problems while enjoying new features.
   - Debate over region-based access in Europe suggested varying availability, with some users contemplating refunds.



**Link mentioned**: <a href="https://x.com/gdb/status/1790869434174746805">Tweet from Greg Brockman (@gdb)</a>: A GPT-4o generated image — so much to explore with GPT-4o&#39;s image generation capabilities alone. Team is working hard to bring those to the world.

  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1268331656276607027)** (24 messages🔥): 

> - `Alpha testing eligibility`
> - `Custom GPTs issues`
> - `Free AI diagram tools`
> - `Plus subscription impacts`
> - `Monetizing GPTs` 


- **Alpha testing eligibility relies on luck**: When asked about how to become an alpha tester, a user simply replied that it requires luck.
- **Custom GPTs stuck during configuration**: A user having trouble uploading PNG screenshots to their custom GPTs received an error stating 'Hmm...something seems to have gone wrong' repeatedly without resolution.
- **Custom GPTs disabled upon cancelling Plus subscription**: It was confirmed that cancelling a Plus subscription will disable and hide any custom GPTs created by the user.
- **Monetizing GPTs requires significant usage numbers**: A discussion revealed that high usage numbers and being located in the USA are prerequisites for being invited to monetize GPTs.
   - Despite initial announcements about GPT Store monetization, users are disappointed due to lack of progress and rollouts of promised features.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1268290812542849034)** (12 messages🔥): 

> - `Prompt engineering platforms`
> - `Evaluation tools`
> - `Text reduction strategies` 


- **Best platform for prompt engineering**: A member asked for the best platform for prompt engineering, to which another replied, **Claude 3.5 Sonnet**.
   - Artifacts and Projects were praised for their strengths in this regard.
- **Tools for heuristic prompt evaluations**: A member expressed interest in prompt evaluations and steerability, preferring heuristic and prototyping tools over full automation.
   - The Anthropic Evaluation Tool was mentioned positively, but there was interest in alternatives that work with other LLMs.
- **Google Sheet for evaluation**: For collaborative prompt evaluation, a member suggested that **a Google Sheet with scripts** might be the best approach.
   - This method could facilitate sharing and collaboration better than other tools.
- **Free AI tools for drawing diagrams**: A member inquired about free AI tools that can draw diagrams.
   - Another member simply replied, **ChatGPT**.
- **Challenges in text length reduction**: A member asked about reducing text to a specific character or word count.
   - Another clarified that LLMs struggle with exact counts, suggesting qualitative language for more consistent lengths.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1268290812542849034)** (12 messages🔥): 

> - `Prompt Engineering Platforms`
> - `Human Evaluation Tools`
> - `AI for Drawing Diagrams`
> - `Reducing Text Length` 


- **Best Platforms for Prompt Engineering**: A member asked about the best platforms for **prompt engineering** and another suggested **Claude 3** and **Sonnet**.
   - They also mentioned that **Artifacts + Projects** are strong contenders in the field.
- **Anthropic Evaluation Tool for Steerability**: A discussion focused on **Anthropic Evaluation Tool** for **prompt evaluations** and **steerability** for heuristics and prototyping.
   - A member suggested that a **Google Sheet with scripts** might be the most collaborative and easy-to-share alternative.
- **Free AI Tools for Drawing Diagrams**: A member inquired about free AI tools that can draw diagrams.
   - Another member recommended **ChatGPT**, although its suitability for drawing diagrams was disputed.
- **Reducing Text to Specific Lengths**: A member asked about reducing text to specific **character** or **word counts**.
   - Another member explained that due to the nature of **LLMs**, they can't ensure exact counts and suggested using qualitative language terms like *short* or *long* instead.


  

---



### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1268284072963211444)** (55 messages🔥🔥): 

> - `FSDP Criticism`
> - `Sharding LLaMA 405B`
> - `vLLM and LLaMA 3.1 Support`
> - `Megatron Paper Discussions`
> - `Torchrun and GPU Memory Issues` 


- **FSDP Criticism Sparks Debate**: A member criticized FSDP, calling it 'kind of ass', which led to a discussion about its applications and scalability.
   - Another member pointed out that while **FSDP** is not ideal for all scenarios, *'there's no beating it as far as ease of use is concerned'*.
- **Struggling with Sharding LLaMA 405B Across Nodes**: Members discussed issues with sharding **LLaMA 405B** across 2 nodes with 8 x H100s, primarily facing problems during inference.
   - Suggestions were made to use vLLM and explore quantization methods, though the original member preferred to avoid VLLM.
- **vLLM Extends Support for LLaMA 3.1**: A member highlighted that **vLLM** now supports the LLaMA 3.1 model series with enhancements for larger context windows and pipeline parallelism.
   - They shared a [blog post](https://blog.vllm.ai/2024/07/23/llama31.html) detailing these new features including FP8 quantization.
- **Megatron Paper Sparks Interest**: Members showed interest in the Megatron paper from 2021, discussing its relevance and sharing [links to the paper](https://people.eecs.berkeley.edu/~matei/papers/2021/sc_megatron_lm.pdf) and related resources.
   - A [YouTube video](https://www.youtube.com/watch?v=u6NEbhNiyAE&ab_channel=MITHANLab) was also shared for further understanding of distributed training concepts.
- **Issues with Torchrun and GPU Memory**: A member reported issues with **torchrun**, where GPU memory isn't freed when manually stopping the script.
   - Suggestions included [using @record](https://pytorch.org/docs/stable/elastic/errors.html) to handle errors and ensure GPU memory is cleared.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://pytorch.org/docs/stable/elastic/errors.html">Error Propagation &mdash; PyTorch 2.4 documentation</a>: no description found</li><li><a href="https://blog.vllm.ai/2024/07/23/llama31.html">Announcing Llama 3.1 Support in vLLM</a>: Today, the vLLM team is excited to partner with Meta to announce the support for the Llama 3.1 model series. Llama 3.1 comes with exciting new features with longer context length (up to 128K tokens), ...</li><li><a href="https://people.eecs.berkeley.edu/~matei/papers/2021">Index of /~matei/papers/2021</a>: no description found</li><li><a href="https://arxiv.org/abs/2208.11174">Demystifying the Nvidia Ampere Architecture through Microbenchmarking and Instruction-level Analysis</a>: Graphics processing units (GPUs) are now considered the leading hardware to accelerate general-purpose workloads such as AI, data analytics, and HPC. Over the last decade, researchers have focused on ...</li><li><a href="https://www.youtube.com/watch?v=u6NEbhNiyAE&ab_channel=MITHAN">EfficientML.ai Lecture 17: Distributed Training (Part I) (MIT 6.5940, Fall 2023, Zoom)</a>: EfficientML.ai Lecture 17: Distributed Training (Part I) (MIT 6.5940, Fall 2023, Zoom)Instructor: Prof. Song HanSlides: https://efficientml.ai</li><li><a href="https://www.youtube.com/watch?v=u6NEbhNiyAE&ab_channel=MITHANLab">EfficientML.ai Lecture 17: Distributed Training (Part I) (MIT 6.5940, Fall 2023, Zoom)</a>: EfficientML.ai Lecture 17: Distributed Training (Part I) (MIT 6.5940, Fall 2023, Zoom)Instructor: Prof. Song HanSlides: https://efficientml.ai
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1268718331687342092)** (9 messages🔥): 

> - `Triton tiled matmul tutorial`
> - `GROUP_SIZE_M argument`
> - `Block and group tiling`
> - `L2 cache optimization` 


- **Clarification on GROUP_SIZE_M in Triton Tiled Matmul Tutorial**: A user inquired about the role of the `GROUP_SIZE_M` argument in the [Triton tiled matmul tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html#l2-cache-optimizations), questioning its purpose and advantage.
   - Another user explained that `GROUP_SIZE_M` controls how many blocks of rows are processed before changing columns, enhancing L2 cache hit rate, and is one level of cache tiling above block tiling and below warp/thread tiling.
- **GROUP_SIZE_M vs. MAX Value Usage**: The discussion continued with a user asking why `GROUP_SIZE_M` should not always be set to the maximum possible value.
   - The response highlighted that similar logic applies to block tiling in shared memory and that setting it to the max could lead to inefficiencies explained in the tutorial, comparing it to not using the full length of dimensions for block sizes.



**Link mentioned**: <a href="https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html#l2-cache-optimizations">Matrix Multiplication &mdash; Triton  documentation</a>: no description found

  

---


### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1268430240107986964)** (3 messages): 

> - `Running video predictor example notebook`
> - `Google Colab example for sam2`
> - `GitHub issue for segment-anything-2` 


- **Running video predictor example notebook fails**: A member was unable to run the video predictor example notebook from **sam2**.
   - Despite trying various changes on their end, they could not get it to work and sought community advice.
- **Alternative Google Colab notebook found for sam2**: The same member found a [Google Colab notebook](https://colab.research.google.com/drive/1Un09HITLLM-ljkG1Ehn9cJjdwk8FVI_1?usp=sharing) that works with their configuration.
   - They thanked the contributor on the relevant [GitHub issue](https://github.com/facebookresearch/segment-anything-2/issues/40) for providing a solution.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/1Un09HITLLM-ljkG1Ehn9cJjdwk8FVI_1?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://github.com/facebookresearch/segment-anything-2/issues/40">Google Colab example · Issue #40 · facebookresearch/segment-anything-2</a>: Not an issue. If someone needs it, I build a working Colab with the model - https://colab.research.google.com/drive/1Un09HITLLM-ljkG1Ehn9cJjdwk8FVI_1?usp=sharing Working end-to-end.
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1268572448656195716)** (1 messages): 

> - `Llama 3 Herd of Models`
> - `AIMO: Findings from the winners`
> - `SAM 2: Segment Anything Model 2`
> - `LazyLLM` 


- **Meta reveals Llama 3.1: Herd of Models**: Meta released [Llama 3.1](https://datta0.substack.com/i/143781557/llama) which includes a new model with **405 billion parameters**, trained on **15.6 trillion tokens** on a cluster of **16,000 H100 GPUs**.
   - They utilized models like **Roberta** to filter out and create a high-quality dataset for training.
- **AIMO winners' findings dissected**: This week's analysis includes a detailed review of the winners' findings from the AIMO competition.
- **SAM 2: The successor to Segment Anything Model**: Discussion covered [SAM 2](https://datta0.substack.com/p/ai-unplugged-16-llama-3-aimo-winners), the next iteration of the Segment Anything Model.
- **LazyLLM boosts LLM inference performance**: A segment focused on **LazyLLM**, which aims at improving the performance of LLMs during inference.



**Link mentioned**: <a href="https://datta0.substack.com/p/ai-unplugged-16-llama-3-aimo-winners">AI Unplugged 16: Llama 3, AIMO winners, Segment Anything Model 2, LazyLLM</a>: Insights over Information

  

---


### **CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1268311686134108203)** (3 messages): 

> - `Digital Video Eavesdropping`
> - `NVIDIA Titan Series Graphics Cards`
> - `Segment Anything Video (SA-V) Dataset` 


- **Revolutionizing Digital Video Eavesdropping Techniques**: A recent [arXiv paper](https://arxiv.org/abs/2407.09717) discusses a novel approach to eavesdrop on digital video displays by analyzing electromagnetic waves from HDMI cables, termed **TEMPEST**.
   - The authors propose using a deep learning module to map observed electromagnetic signals back to the displayed image, overcoming the challenges posed by the high bandwidth and non-linear mapping of digital signals.
- **NVIDIA's Next-Gen Titan GPUs Unveiled**: According to a [Wccftech article](https://wccftech.com/nvidia-next-gen-titan-graphics-card-exists-flagship-blackwell-gpu/), NVIDIA's new Titan-class graphics card based on the Blackwell GPU architecture exists, but its launch remains doubtful.
   - Previous Titan releases include the **Titan RTX** from 2018, and there is speculation whether new 
- **Meta Releases Vast SA-V Dataset for AI Research**: Meta introduced the [Segment Anything Video (SA-V) dataset](https://ai.meta.com/datasets/segment-anything-video/), containing 51K videos and 643K spatio-temporal segmentation masks.
   - The dataset supports computer vision research and consists of manually annotated and automatically generated masklets, with an average video resolution of **1401×1037 pixels**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://wccftech.com/nvidia-next-gen-titan-graphics-card-exists-flagship-blackwell-gpu/">NVIDIA&#039;s Next-Gen Titan Graphics Card Does Exist &amp; Based on Flagship Blackwell GPU</a>: NVIDIA reportedly already has a Titan-class graphics card based on its next-gen Blackwell GPU architecture but its launch is doubtful.</li><li><a href="https://arxiv.org/abs/2407.09717">Deep-TEMPEST: Using Deep Learning to Eavesdrop on HDMI from its Unintended Electromagnetic Emanations</a>: In this work, we address the problem of eavesdropping on digital video displays by analyzing the electromagnetic waves that unintentionally emanate from the cables and connectors, particularly HDMI. T...</li><li><a href="https://ai.meta.com/datasets/segment-anything-video/?utm_source=twitter&utm_medium=organic_social&utm_content=video&utm_campaign=sam2">no title found</a>: no description found
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1268324131812671594)** (2 messages): 

> - `Ampere A100 SM organization`
> - `Warp distribution in processing blocks`
> - `Hardware design choices`
> - `Hopper architecture` 


- **Ampere A100 SM split into smaller processing blocks**: A user queried why the Ampere A100 SM, with 64 cores, is organized into **four processing blocks with 16 cores each** rather than 32 cores to match the warp size.
   - Another user speculated that Nvidia likely made this choice to maintain a balance that keeps the hardware busy, given **kernel needs**, space on silicon, bandwidth, and latency parameters.
- **Speculations on Hardware Design Choices**: One user mentioned that hardware design involves balancing **space on silicon** with utilization, where more units take more space.
   - They suggested it might be a delicate balance act to ensure that additional units are worth their cost in terms of bandwidth and latency.


  

---


### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1268320487675793520)** (11 messages🔥): 

> - `.py vs .ipynb`
> - `Quantization-Aware Training (QAT)`
> - `Conversion of .ipynb to .py`
> - `GitHub Repositories for Jupyter and PyTorch`
> - `Performance Comparison of QAT and PTQ` 


- **.py vs .ipynb Usability Debate**: Discussion centered around whether .py files can be easily runnable and modifiable in comparison to .ipynb files, with some members suggesting various tools and methods for conversion.
   - One member mentioned using [LibCST](https://github.com/Instagram/LibCST) for conversions, while another noted the availability of export options in Colab and Jupyter UI.
- **Quantization-Aware Training improves PyTorch Model Accuracy**: A [blog post on PyTorch](https://pytorch.org/blog/quantization-aware-training) discusses an end-to-end Quantization-Aware Training (QAT) flow which can **recover up to 96% of the accuracy degradation** on hellaswag and **68% of the perplexity degradation on wikitext** for Llama3 compared to post-training quantization.
   - This blog also introduces QAT APIs in [torchao](https://github.com/pytorch/ao/) and highlights their integration with [torchtune](https://github.com/pytorch/torchtune/).
- **QAT vs. PTQ in Practical Application**: One member explained the crucial difference between Quantization-Aware Training and Quantized Training, emphasizing QAT's substantial performance improvements.
   - Another participant highlighted the excitement about combining low-rank adaptation with QAT for enhanced performance.
- **Overfitting Concerns with QAT**: A user questioned if overfitting was checked during the QAT process, suggesting that MMLU could be a good metric for verification.
   - This sparked a further mention for verification by another user, indicating the community's interest in the thorough evaluation of QAT.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://pytorch.org/blog/quantization-aware-training">Quantization-Aware Training for Large Language Models with PyTorch</a>: In this blog, we present an end-to-end Quantization-Aware Training (QAT) flow for large language models in PyTorch. We demonstrate how QAT in PyTorch can recover up to 96% of the accuracy degradation ...</li><li><a href="https://github.com/jupyter/notebook/blob/main/docs/source/examples/Notebook/Running%20Code.ipynb?short_path=c932132">notebook/docs/source/examples/Notebook/Running Code.ipynb at main · jupyter/notebook</a>: Jupyter Interactive Notebook. Contribute to jupyter/notebook development by creating an account on GitHub.</li><li><a href="https://github.com/Instagram/LibCST">GitHub - Instagram/LibCST: A concrete syntax tree parser and serializer library for Python that preserves many aspects of Python&#39;s abstract syntax tree</a>: A concrete syntax tree parser and serializer library for Python that preserves many aspects of Python&#39;s abstract syntax tree - Instagram/LibCST
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1268294297380196484)** (177 messages🔥🔥): 

> - `GELU changes`
> - `Llama 3.1 reference implementation`
> - `Reference implementation issues`
> - `TorchChat`
> - `RoPE scaling` 


- **GELU optimization PR for LLMC**: A new [PR](https://github.com/karpathy/llm.c/pull/721) was submitted to move faster GELU changes from the FP8 branch to master, which improves validation loss slightly.
   - *Surprisingly, it actually helps val loss a tiny bit, but again might be noise*.
- **Llama 3.1 implementation issues**: Members discussed the lack of documentation for running the Llama 3.1 model after downloading it from [Meta's repo](https://github.com/meta-llama/llama-models) and shared code snippets to attempt loading and running it.
   - It's suspected that a 10-line Python snippet is missing for a straightforward run, with [inference scripts](https://github.com/meta-llama/llama-recipes/blob/main/recipes/quickstart/inference/local_inference/README.md) highlighted as overly complicated.
- **TorchChat as a Llama 3.1 reference**: A reference implementation for Llama 3.1 was shared in the form of a new [TorchChat repository](https://github.com/pytorch/torchchat) released by PyTorch.
   - This implementation serves as a detailed guide for local and server-based running of Llama 3.1 models.
- **RoPE scaling and specialized features**: The conversation included detailed discussions on how RoPE scaling differs in Llama 3.1 and the necessity to update [reference implementations](https://github.com/karpathy/llm.c/blob/7e0c497936540a44338e214bc230a1f041090fcb/llmc/encoder.cuh#L161) accordingly.
   - Members shared insights on integrating this in CUDA code for better fine-tuning operations.
- **Fine-tuning techniques on Llama 3.1**: Discussion pivoted towards fine-tuning, weighing full finetuning vs. LoRA approaches, with insights into LoRA being efficient on smaller datasets.
   - It was suggested that sometimes training on *just completions* can yield better results, and a snippet to implement this was shared from the [unsloth repo](https://github.com/unslothai/unsloth/blob/main/unsloth/chat_templates.py#L1456).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/meta-llama/llama3/blob/main/llama/generation.py">llama3/llama/generation.py at main · meta-llama/llama3</a>: The official Meta Llama 3 GitHub site. Contribute to meta-llama/llama3 development by creating an account on GitHub.</li><li><a href="https://github.com/meta-llama/llama3/blob/main/example_text_completion.py">llama3/example_text_completion.py at main · meta-llama/llama3</a>: The official Meta Llama 3 GitHub site. Contribute to meta-llama/llama3 development by creating an account on GitHub.</li><li><a href="https://github.com/meta-llama/llama-recipes/blob/main/recipes/quickstart/inference/local_inference/README.md">llama-recipes/recipes/quickstart/inference/local_inference/README.md at main · meta-llama/llama-recipes</a>: Scripts for fine-tuning Meta Llama3 with composable FSDP &amp;amp; PEFT methods to cover single/multi-node GPUs. Supports default &amp;amp; custom datasets for applications such as summarization and Q...</li><li><a href="https://github.com/karpathy/nano-llama31/tree/master">GitHub - karpathy/nano-llama31: nanoGPT style version of Llama 3.1</a>: nanoGPT style version of Llama 3.1. Contribute to karpathy/nano-llama31 development by creating an account on GitHub.</li><li><a href="https://github.co">GitHub: Let’s build from here</a>: GitHub is where over 100 million developers shape the future of software, together. Contribute to the open source community, manage your Git repositories, review code like a pro, track bugs and fea...</li><li><a href="https://github.com/karpathy/nano-llama31">GitHub - karpathy/nano-llama31: nanoGPT style version of Llama 3.1</a>: nanoGPT style version of Llama 3.1. Contribute to karpathy/nano-llama31 development by creating an account on GitHub.</li><li><a href="https://github.com/karpathy/llm.c/pull/721">Faster GELU forward &amp; backward using MUFU.TANH for SM7.5+ by ademeure · Pull Request #721 · karpathy/llm.c</a>: These are faster GELU kernels by using the HW instruction NVIDIA introduced for this in Turing (SM7.5) but never exposed outside of PTX as far as I can tell, possibly because it&#39;s slightly less ac...</li><li><a href="https://github.com/meta-llama/llama-models">GitHub - meta-llama/llama-models: Utilities intended for use with Llama models.</a>: Utilities intended for use with Llama models. Contribute to meta-llama/llama-models development by creating an account on GitHub.</li><li><a href="https://github.com/pytorch/pytorch/issues/39716">Do not modify global random state · Issue #39716 · pytorch/pytorch</a>: 🚀 Feature Currently, the recommended approach to achieve reproducibility is setting global random seeds. I would like to propose that instead all functions which need a random source accept a local.....</li><li><a href="https://github.com/unslothai/unsloth/blob/main/unsloth/chat_templates.py#L1456">unsloth/unsloth/chat_templates.py at main · unslothai/unsloth</a>: Finetune Llama 3.1, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/api/model.py">llama-models/models/llama3_1/api/model.py at main · meta-llama/llama-models</a>: Utilities intended for use with Llama models. Contribute to meta-llama/llama-models development by creating an account on GitHub.</li><li><a href="https://www.picoquant.com/products/category/tcspc-and-time-tagging-modules/hydraharp-400-multichannel-picosecond-event-timer-tcspc-module">
     
        HydraHarp 400 - Multichannel Picosecond Event Timer & TCSPC Module
    
     | PicoQuant</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/blob/main/unsloth/models/llama.py#L1077)">unsloth/unsloth/models/llama.py at main · unslothai/unsloth</a>: Finetune Llama 3.1, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://github.com/karpathy/llm.c/blob/7e0c497936540a44338e214bc230a1f041090fcb/llmc/encoder.cuh#L161">llm.c/llmc/encoder.cuh at 7e0c497936540a44338e214bc230a1f041090fcb · karpathy/llm.c</a>: LLM training in simple, raw C/CUDA. Contribute to karpathy/llm.c development by creating an account on GitHub.</li><li><a href="https://github.com/trholding/llama2.c/blob/8a0ad84b9ee94fad175e5687fb8774503efbd23b/runq.c#L653">llama2.c/runq.c at 8a0ad84b9ee94fad175e5687fb8774503efbd23b · trholding/llama2.c</a>: Llama 2 Everywhere (L2E). Contribute to trholding/llama2.c development by creating an account on GitHub.</li><li><a href="https://github.com/pytorch/torchchat">GitHub - pytorch/torchchat: Run PyTorch LLMs locally on servers, desktop and mobile</a>: Run PyTorch LLMs locally on servers, desktop and mobile - pytorch/torchchat</li><li><a href="https://github.com/pytorch/torchchat/blob/main/generate.py">torchchat/generate.py at main · pytorch/torchchat</a>: Run PyTorch LLMs locally on servers, desktop and mobile - pytorch/torchchat</li><li><a href="https://github.com/pytorch/torchchat/blob/main/build/model.py">torchchat/build/model.py at main · pytorch/torchchat</a>: Run PyTorch LLMs locally on servers, desktop and mobile - pytorch/torchchat
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[lecture-qa](https://discord.com/channels/1189498204333543425/1238926773216084051/1268389635982233650)** (1 messages): 

> - `L2 latency as hyperparameter`
> - `latency bound algorithm` 


- **Question on using L2 latency as a hyperparameter**: A member asked how to use **L2 latency** as a hyperparameter in the options for the **2 billion options**.
   - The same member also inquired about the definition and application of a **latency bound algorithm**.
- **Understanding latency bound algorithm**: A user sought clarification on what is meant by **latency bound algorithm**.
   - This followed a previous question on the role of **L2 latency** in hyperparameter tuning.


  

---


### **CUDA MODE ▷ #[cudamode-irl](https://discord.com/channels/1189498204333543425/1267896441989234709/1268283861427552266)** (4 messages): 

> - `Gradient involvement`
> - `Seq Parallel`
> - `Triton Kernels`
> - `Hackathon`
> - `Event Criteria` 


- **Gradient's Michael explores Seq Parallel and Triton Kernels**: Michael from Gradient announced his work on either **Seq Parallel** or **Triton Kernels** for some unique architectures and invited others to join him in SF.
- **Hackathon-style learning interest from a newbie**: Pacomann expressed interest in joining the event, emphasizing a desire to learn a lot in a **hackathon-style** format.
- **Question on event approval criteria**: Evil666man asked whether there was a criterion for approval or if it was **first come, first serve**.
   - Kashimoo responded, implying the event would have been full if it were first come, first serve.


  

---



### **Stability.ai (Stable Diffusion) ▷ #[announcements](https://discord.com/channels/1002292111942635562/1002292398703001601/1268616070860902400)** (1 messages): 

> - `Stable Fast 3D Launch`
> - `Technical Report`
> - `3D Asset Generation Technology`
> - `Speed and Quality of 3D Reconstruction`
> - `Applications in Gaming and VR` 


- **Stable Fast 3D Launch 🚀**: Stability AI has introduced **Stable Fast 3D**, a model that transforms a single input image into a detailed 3D asset in just **0.5 seconds**, setting a new standard for speed and quality in 3D reconstruction. [Learn more and access the report](https://stability.ai/news/introducing-stable-fast-3d).
   - *'Stable Fast 3D's unprecedented speed and quality make it an invaluable tool for rapid prototyping in 3D work.'*
- **How Stable Fast 3D Works**: Users can upload a single image of an object, and **Stable Fast 3D** rapidly generates a complete 3D asset, including **UV unwrapped mesh**, material parameters, and albedo colors with reduced illumination bake-in. [Watch the video for detailed model improvements](https://www.youtube.com/watch?v=uT96UCBSBko).
   - Optional quad or triangle remeshing adds only **100-200ms** to the processing time, increasing its utility across various industries.



**Link mentioned**: <a href="https://stability.ai/news/introducing-stable-fast-3d">Introducing Stable Fast 3D: Rapid 3D Asset Generation From Single Images &mdash; Stability AI</a>: We are excited to introduce Stable Fast 3D, Stability AI’s latest breakthrough in 3D asset generation technology. This innovative model transforms a single input image into a detailed 3D asset, settin...

  

---


### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1268284094136188948)** (212 messages🔥🔥): 

> - `Training Loras for TV Characters`
> - `SD3 Model Usage`
> - `Handling VAE Issues`
> - `Creative Upscaler Confusion`
> - `Flux Model Release` 


- **Training Loras for TV characters in SD3**: Members discussed how to train 2 Loras of TV characters and have both of them in the same image, recommending the use of SD3 for its unique understanding capabilities.
   - Suggestions included starting with prompting, using regional prompter extension in auto1111, and validating through community testing.
- **SD3 Medium model issues and usage**: Users faced errors loading SD3 Medium from Huggingface such as 'AttributeError: NoneType object has no attribute lowvram'.
   - Resolutions discussed included downloading all model components, using ComfyUI workflows, and exploring other compatible UIs like Auto1111.
- **Managing VAE settings to prevent red images**: Community members addressed issues where rendered images turn red at 95%, attributing it mostly to VAE settings.
   - Solutions included using '--no-half-vae' setting and sharing troubleshooting tips for different graphics cards and VAE combinations.
- **Clarifying Stability AI's Creative Upscaler**: Confusion around the 'Creative Upscaler' mentioned in NightCafe led to clarifications that it's not a real Stability AI product.
   - Members recommended alternative upscaling techniques using ERSGAN, transformers, and multi-stage workflows shared on community forums.
- **Flux model release by Black Forest Labs**: The community welcomed the release of the Flux model, which offers significant improvements in image quality and parameter count.
   - Users discussed the model’s performance on different GPUs, with the 4090 being highly recommended, and noted exceptional results in rendering hands and fingers.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://blog.fal.ai/flux-the-largest-open-sourced-text2img-model-now-available-on-fal/">Announcing Flux by Black Forest Labs: The Next Leap in Text-to-Image Models</a>: Flux, the largest SOTA open source text-to-image model to date, developed by Black Forest Labs—the original team behind Stable Diffusion is now available on fal. Flux pushes the boundaries of creativi...</li><li><a href="https://huggingface.co/stabilityai/stable-diffusion-3-medium/tree/main">stabilityai/stable-diffusion-3-medium at main</a>: no description found</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1ehiz51/flux_image_examples/#lightbox">Reddit - Dive into anything</a>: no description found</li><li><a href="https://huggingface.co/stabilityai/stable-diffusion-3-medium">stabilityai/stable-diffusion-3-medium · Hugging Face</a>: no description found</li><li><a href="https://github.com/Stability-AI/generative-models">GitHub - Stability-AI/generative-models: Generative Models by Stability AI</a>: Generative Models by Stability AI. Contribute to Stability-AI/generative-models development by creating an account on GitHub.</li><li><a href="https://comfyworkflows.com/">Comfy Workflows</a>: Share, discover, &amp; run thousands of ComfyUI workflows.
</li>
</ul>

</div>
  

---



### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1268285391820165192)** (121 messages🔥🔥): 

> - `Exit codes in LM Studio`
> - `Gemma 2 models`
> - `Model embedding and LLaMA capabilities`
> - `Bugs and troubleshooting in LM Studio`
> - `Future LM Studio features and user requests` 


- **Members report various Exit Codes**: Users encountered different exit codes such as 6 and 0 on various systems, leading to discussions on system compatibility and debugging.
- **Gemma 2 Models: Compatibility and Errors**: Community members faced issues running **Gemma 2 2B** models, especially on older or specific hardware, with some requiring new LM Studio versions.
- **Embedding with LLaMA and Future Prospects**: Queries arose about using **LLaMA** for embedding within LM Studio, highlighting projects like [LLM2Vec](https://github.com/McGill-NLP/llm2vec) for potential solutions.
- **Bugs and Troubleshooting in LM Studio**: Various bugs were highlighted by users, including issues with GPU offload and network errors linked to VPN/DNS settings.
- **User Requests for Future LM Studio Features**: Users expressed a desire for features like **TTS voices**, internet access for models, and **RAG** for document interaction within LM Studio.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/meta-llama/Meta-Llama-3.1-405B">meta-llama/Meta-Llama-3.1-405B · Hugging Face</a>: no description found</li><li><a href="https://paperswithcode.com/task/visual-question-answering">Papers with Code - Visual Question Answering (VQA)</a>: **Visual Question Answering (VQA)** is a task in computer vision that involves answering questions about an image. The goal of VQA is to teach machines to understand the content of an image and answer...</li><li><a href="https://github.com/McGill-NLP/llm2vec">GitHub - McGill-NLP/llm2vec: Code for &#39;LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders&#39;</a>: Code for &#39;LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders&#39; - McGill-NLP/llm2vec
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1268376543730733157)** (24 messages🔥): 

> - `GPU offload in LM Studio`
> - `Stable Diffusion model compatibility`
> - `Amuse AI for image generation`
> - `Proxmox learning` 


- **Enable iGPU for better VRAM availability**: A member tried to enable their iGPU to free up VRAM on their RTX3090 for loading models in LM Studio but still sees 0.5/24.0 GB VRAM usage when idle.
   - Another member clarified that iGPUs are unsupported without the [OpenCL addon pack](https://discord.com/channels/1110598183144399058/1111797717639901324/1268091222686175246); a new beta version with Vulkan support might help.
- **Stable Diffusion not supported in LM Studio**: A user reported an error when trying to load a stable-diffusion model, revealing that LM Studio does not support image generation models such as Stable Diffusion.
   - Suggestions were given to use [Stability Matrix](https://stability.ai/), Automatic1111, or Amuse AI for these tasks.
- **Amuse AI now available for Radeon users**: A member announced that Amuse AI is available for Radeon users, allowing stable diffusion image generation on GPUs with new EZ mode.
   - It offers features such as AI filters and sketch-to-image generation without login or cost prerequisites.
- **Proxmox learning tips for beginners**: A participant asked for tips on drivers in Proxmox and was advised to practice Proxmox inside VirtualBox under Windows first.
   - A thorough [learning plan](https://example.com) was shared, covering topics from installation to GPU passthrough and LLM utilization.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/spongebob-patrick-star-shocked-loop-surprised-gif-16603980">Spongebob Patrick Star GIF - Spongebob Patrick Star Shocked - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.amuse-ai.com/">Amuse</a>: Stable Diffusion Image and Video Generation
</li>
</ul>

</div>
  

---



### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1268284046572650670)** (88 messages🔥🔥): 

> - `Watermarking in AI`
> - `NTIA Report on AI Openness`
> - `GitHub Models Launch`
> - `Legal Challenges in Deepfakes`
> - `GPT-2 Model Improvements` 


- **Watermarking tech trust issues spark debate**: Members debated the effectiveness of watermarking in solving trust issues in AI, with some arguing it only works in institutional settings and cannot prevent misuse entirely.
   - The discussion suggested that better cultural norms and trust mechanisms, rather than watermarking, are needed to address the spread of deepfakes and misrepresented content.
- **NTIA supports open models in latest report**: The NTIA issued a [report](https://www.ntia.gov/press-release/2024/ntia-supports-open-models-promote-ai-innovation) advocating for the openness of AI models while recommending risk monitoring, influencing policy considerations in the US.
   - Participants noted that the NTIA functions within the Department of Commerce and reports directly to the White House, giving weight to its policy recommendations on AI model openness.
- **GitHub introduces integrated AI models**: GitHub announced [GitHub Models](https://github.blog/news-insights/product-news/introducing-github-models/), allowing developers to access and experiment with top AI models directly on their platform.
   - Community members speculated that this move might be an attempt to compete with platforms like Hugging Face by integrating AI capabilities into developers' existing workflows.
- **Challenges of regulating deepfakes**: Members discussed the regulatory complexities around deepfakes, particularly libel and defamation issues, and the difficulties of enforcing laws on a global scale.
   - The discussion highlighted concerns over the feasibility of prosecuting deepfake creators and the potential for such content to be used in blackmail schemes.
- **Optimizing GPT-2 with new papers and techniques**: A participant working on a GPT-2 model sought advice on incorporating advanced techniques, having already implemented Rotary Positional Embeddings and Grouped Query Attention.
   - Community members suggested looking at recent papers and evaluation metrics like human eval to further improve the model and measure its performance effectively.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/ashtom/status/1819041110200906202">Tweet from Thomas Dohmke (@ashtom)</a>: Build AI applications right where you manage your code. With GitHub Models, now more than 100 million developers can access and experiment with top AI models where their workflow is – directly on GitH...</li><li><a href="https://www.ntia.gov/press-release/2024/ntia-supports-open-models-promote-ai-innovation">NTIA Supports Open Models to Promote AI Innovation | National Telecommunications and Information Administration</a>: no description found</li><li><a href="https://fixupx.com/impershblknight/status/1818769082944307517?t=41UyAwMxUTUMwBIspUiHRQ&s=19">Tweet from Imperishable Knight ⛩️ (RJ) (@impershblknight)</a>: Tip for Plus users hoping to get #ChatGPT Advanced Voice alpha access:  Have you tried enabling these settings? I didn&#39;t get the AV invite initially but I enabled them then hours later as the next...</li><li><a href="https://en.wikipedia.org/wiki/United_States_Department_of_Commerce#Structure">United States Department of Commerce - Wikipedia</a>: no description found</li><li><a href="https://www.federalregister.gov/documents/2023/04/13/2023-07776/ai-accountability-policy-request-for-comment">Federal Register :: Request Access</a>: no description found</li><li><a href="https://archive.is/2yfdW">UK&#x2019;s AI bill to focus on ChatGPT-style models</a>: no description found
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1268284431370555392)** (7 messages): 

> - `system prompt style model training`
> - `MLCommons AlgoPerf results`
> - `synthetic data generation`
> - `system prompt generalization` 


- **System Prompt Style Models Training Query**: A member questioned the existence of papers on how **system prompt style models** were trained, finding them synthetic as they don't exist in the wild.
   - Another member suggested they can be generated automatically or with minimal human effort once a system prompt-tuned model is available.
- **MLCommons AlgoPerf Results Announced**: [MLCommons AlgoPerf](https://x.com/mlcommons/status/1819098247270695254) results are in, highlighting a $50K prize competition where non-diagonal preconditioning outperformed Nesterov Adam by 28%, setting a new SOTA in hyperparameter-free algorithms.
   - This achievement was celebrated as **distributed shampoo** emerged victorious in the competition.
- **Synthetic Data for System Prompts**: Discussion on using **synthetic data generation** and GPT-4 distillation to generate system prompts for chat/instruct models.
   - A member expressed the need for more research to back up claims about the effectiveness of **system prompt** generation in ensuring model guardrails.



**Link mentioned**: <a href="https://x.com/mlcommons/status/1819098247270695254">Tweet from MLCommons (@MLCommons)</a>: @MLCommons #AlgoPerf results are in! 🏁 $50K prize competition yielded 28% faster neural net training with non-diagonal preconditioning beating Nesterov Adam. New SOTA for hyperparameter-free algorith...

  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1268591088751480832)** (15 messages🔥): 

> - `Scaling law experiments`
> - `Validation log-likelihood anomalies`
> - `Double descent phenomenon`
> - `Broken Neural Scaling Law (BNSL) paper`
> - `Task-specific scaling behavior` 


- **Scaling law experiments reveal anomalies**: Experiments comparing the validation log-likelihood of models trained on different-sized subsets show that the model trained on **1e6 sequences significantly underperforms** those trained on fewer or more sequences.
- **Speculations and explanations for validation dip**: Members initially suspected a **bug in the data processing pipeline** but couldn't find any, prompting discussions on the double descent phenomenon.
   - Another user mentioned the [BNSL paper](https://arxiv.org/abs/2210.14891) showing similar double descent behavior regarding dataset size, leading to confusion about this occurring depending on the task.
- **Double descent debated**: Double descent is mentioned as a potential cause, though traditionally linked to increasing parameters rather than dataset size.
   - A user clarified that double descent can occur for both parameters and dataset size, noting that the issue might be task-specific.



**Link mentioned**: <a href="https://arxiv.org/abs/2210.14891">Broken Neural Scaling Laws</a>: We present a smoothly broken power law functional form (that we refer to as a Broken Neural Scaling Law (BNSL)) that accurately models &amp; extrapolates the scaling behaviors of deep neural networks ...

  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1268285771878629376)** (5 messages): 

> - `Gemma Scope`
> - `ICML Mech Int Workshop Recording` 


- **Recording for the ICML Mech Int Workshop**: A member inquired about the recording for the **ICML Mech Int Workshop** and was informed by another member that it will be available after a month due to **ICML rules**.
   - It was mentioned that these rules are likely to incentivize people to pay for a virtual pass. *Another suggestion was made to obtain the link from a conference attendee.*
- **Great Work on Gemma Scope**: A member complimented the excellent progress on **Gemma Scope** in a brief interaction.
   - The query about the ICML Mech Int Workshop recording followed the praise for Gemma Scope.


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1268303952546631801)** (11 messages🔥): 

> - `lm-eval prompt counts`
> - `GPQA benchmarks`
> - `lm_eval harness behavior`
> - `Issue tracking for lm_eval`
> - `Interpreting progress bars in lm_eval` 


- **lm-eval uses more prompts than present in benchmark**: A user noticed that running **lm-eval** even with zeroshot uses 4x the prompts present in certain benchmarks like **gpqa_main**, processing **1792** prompts instead of **448**.
- **GPQA benchmark explained**: Another user explained that **GPQA** has four options and is likely running each option separately.
   - Another user clarified that varying sizes between options shouldn't result in exactly 4x prompts and indicated this happens across other benchmarks like MMLU.
- **Issue within GPQA eval harness**: A user shared their launch script and a specific case where the **lm_eval harness** processes more prompts than expected, providing detailed settings and asking for issue references.
- **Progress bars track choices**: A user clarified that the progress bar in **lm-eval** shows `num_choices * num_docs` for consistency, even if settings allow single-token responses without multiple LM calls.


  

---



### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1268283089474551919)** (61 messages🔥🔥): 

> - `xAI Acquisition Rumors`
> - `Black Forest Labs Announcement`
> - `Gemini 1.5 Pro Release`
> - `GitHub Introduces AI Models` 


- **xAI rumored acquisition of Character AI refuted by Elon Musk**: [Rumors](https://x.com/nmasc_/status/1818788751528935468) spread that xAI might acquire Character AI to test and improve its Grok models, but [Elon Musk denied](https://x.com/elonmusk/status/1818810438634946699) these claims, dismissing the reports as misinformation.
   - Users speculated about the credibility of these rumors, citing similar instances where Musk previously denied reports before they were later confirmed.
- **Black Forest Labs formed by original Stable Diffusion team**: The original **Stable Diffusion** team announced the formation of [Black Forest Labs](https://x.com/bfl_ml/status/1819003686011449788) to develop advanced generative deep learning models for media.
   - They aim to push the boundaries of creativity and efficiency, with their latest model **Flux** available for testing on fal.
- **Google launches Gemini 1.5 Pro**: [Google's latest model](https://x.com/tokumin/status/1819047737230528701?s=46), **Gemini 1.5 Pro**, was released on Google AI Studio and quickly became the top model on LMSYS with an ELO of 1300.
   - This model is praised as the strongest and most intelligent Gemini model to date, showcasing significant advancements.
- **GitHub introduces AI Models**: GitHub announced the [launch of GitHub Models](https://github.blog/news-insights/product-news/introducing-github-models/) to empower developers with industry-leading AI tools directly on their platform.
   - This initiative is designed to make AI more accessible to the developer community, bridging the gap between coder and AI engineer.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://blog.fal.ai/flux-the-largest-open-sourced-text2img-model-now-available-on-fal/">Announcing Flux by Black Forest Labs: The Next Leap in Text-to-Image Models</a>: Flux, the largest SOTA open source text-to-image model to date, developed by Black Forest Labs—the original team behind Stable Diffusion is now available on fal. Flux pushes the boundaries of creativi...</li><li><a href="https://x.com/elonmusk/status/1818810438634946699?s=46">Tweet from Elon Musk (@elonmusk)</a>: @nmasc_ @KalleyHuang @steph_palazzolo The [Mis]Information strikes again.   xAI is not considering an acquisition of Character AI.</li><li><a href="https://x.com/tokumin/status/1819047737230528701?s=46">Tweet from Simon (@tokumin)</a>: We&#39;ve just pushed the latest Gemini 1.5 Pro to http://aistudio.google.com. It&#39;s a REALLY good model, and coming in as the #1 model on LMSYS with an ELO of 1300.   Amazing work from the whole G...</li><li><a href="https://x.com/nmasc_/status/1818802320802824352?s=46">Tweet from natasha mascarenhas (@nmasc_)</a>: I&#39;m hearing that xAI is looking at a number of consumer AI companies as potential acquisition targets, in addition to Character AI.  Also hearing on a daily basis that there are more Inflection/Ad...</li><li><a href="https://x.com/nmasc_/status/1818788751528935468?s=46">Tweet from natasha mascarenhas (@nmasc_)</a>: SCOOP: xAI is weighing an acquisition of Character AI, as it looks to test and improve its Grok models and beef up its talent ranks  https://www.theinformation.com/articles/musks-xai-considers-buying-...</li><li><a href="https://github.blog/news-insights/product-news/introducing-github-models/">Introducing GitHub Models: A new generation of AI engineers building on GitHub</a>: We are enabling the rise of the AI engineer with GitHub Models – bringing the power of industry leading large and small language models to our more than 100 million users directly on GitHub.</li><li><a href="https://x.com/bfl_ml/status/1819003686011449788?t=IHBNW9bCDHQI9rosZVP2bw&s=19">Tweet from Black Forest Labs (@bfl_ml)</a>: We are excited to announce the launch of Black Forest Labs. Our mission is to develop and advance state-of-the-art generative deep learning models for media and to push the boundaries of creativity, e...</li><li><a href="https://x.com/bfl_ml/status/1819003686011449788?t=IHBNW9bCDHQ">Tweet from Black Forest Labs (@bfl_ml)</a>: We are excited to announce the launch of Black Forest Labs. Our mission is to develop and advance state-of-the-art generative deep learning models for media and to push the boundaries of creativity, e...</li><li><a href="https://x.com/elonmusk/status/1750995501560807465?s=46">Tweet from Elon Musk (@elonmusk)</a>: xAI is not raising capital and I have had no conversations with anyone in this regard  Quoting X Daily News (@xDaily)   NEWS: The Financial Times has reported that @xAI is seeking investments up to $6...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1268446410899066920)** (32 messages🔥): 

> - `Together AI's Critique`
> - `Suno vs Music Labels`
> - `AI2 Rebrand`
> - `OpenAI vs. Non-Profit Perceptions` 


- **Together AI Critique Calls Out Cherry-Picked Errors**: An AI researcher criticized Together AI for cherry-picking results and presented points on the need for scientific rigor in LLM evaluations, pointing out that non-smooth outputs and biased benchmarks skew real-world performance.
   - He shared detailed tweets and external resources to emphasize quantization techniques and transparent methodologies in LLM evaluation.
- **Suno Clashes with Music Labels Over Copyright**: Suno's response to RIAA highlights their mission amid a lawsuit from music labels who allege Suno trained on copyrighted output.
   - The discussion reflects on Suno admitting to using copyrighted materials and the contentious talks leading up to the lawsuit.
- **AI2's Rebrand Sparks Mixed Reactions**: Allen AI unveiled its new brand and website, but not all responses were favorable, with some highlighting the use of sparkles emoji as a familiar tactic in AI branding.
   - The change stirred conversations about how even non-profits face scrutiny and mixed reactions during rebranding efforts.
- **OpenAI's Non-Profit Status Questioned**: In a casual exchange, members humorously noted that OpenAI claims to be a non-profit, leading to skepticism about the legitimacy of such status in practice.
   - This reflected broader sentiments that even non-profits do not escape negative press and accountability.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/rachelmetz/status/1819086846913401266?s=46">Tweet from Rachel Metz (@rachelmetz)</a>: looks like @allen_ai is taking a page from the sparkles emoji playbook with its redesign! see my recent piece on the AI industry&#39;s embrace of ✨ to learn more about the humble sparkles&#39; jump in...</li><li><a href="https://x.com/jiayq/status/1818786673695809793?s=46">Tweet from Yangqing Jia (@jiayq)</a>: As an AI researcher and engineer, I fully respect together&#39;s achievement but would like to also point out the many cherrypicked errors. I am sure they are unintentional, but evaluation of LLMs is ...</li><li><a href="https://x.com/mikeyshulman/status/1819010384134631794?s=46">Tweet from Mikey (@MikeyShulman)</a>: We&#39;re filing our response to the members of the RIAA today. It&#39;s important to understand additional context around our mission and what is at stake. You can read more about it on the suno blog...</li><li><a href="https://x.com/allen_ai/status/1819077607897682156">Tweet from Ai2 (@allen_ai)</a>: After months of behind-the-scenes research, interviews, and labors of love, we’re delighted to debut Ai2’s new brand and website today.  Explore the evolution 🧵
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1268287925746012290)** (4 messages): 

> - `Anime Profile Picture Feed`
> - `Article Timing`
> - `Llama 3.1 Scores` 


- **Anime Profile Picture Feed Features Article**: A member mentioned that their anime PFP feed started posting an article, calling it a 'banger' with impeccable timing.
- **Perfect Timing on Article Release Awaiting Llama 3.1 Scores**: Natolambert mentioned getting lucky with the article's timing and revealed they were waiting for **Llama 3.1** scores before releasing it.


  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1268591043172110498)** (28 messages🔥): 

> - `Interviewing Sebastian Raschka`
> - `Knowledge distillation definitions`
> - `Apple AI advancements`
> - `Rejection sampling in RLHF`
> - `Open Instruct updates` 


- **Sebastian Raschka discusses open LLMs and Llama 3.1**: [Sebastian Raschka's interview](https://www.youtube.com/watch?v=-q79uzz1Wik) covers the state of open LLMs, **Llama 3.1**, and AI education.
   - During the interview, concerns about **distillation verbiage** similar to Alpaca and Self-Instruct papers were discussed, highlighting a *naming conflict* in the field.
- **Confusion over knowledge distillation terms**: Members debated the terms for **distillation** used during training with synthetic data versus soft-target and hard-target distillation.
   - The issue is magnified with terms like *rejection sampling* being *un-googleable* outside specific AI contexts.
- **Apple AI integration makes waves**: A discussion on [Apple's new AI features](https://www.interconnects.ai/p/apple-intelligence) suggests their integration can connect apps more seamlessly, making daily tasks easier.
   - Apple's *multi-model AI system, Apple Intelligence*, is seen as a force multiplier in everyday tech, though **AI labs** remain skeptical of its transformative potential.
- **Implementing rejection sampling in Open Instruct**: [Rejection sampling](https://github.com/allenai/open-instruct/pull/205) is being implemented in **Open Instruct**, aiming to streamline training processes.
   - This method might reduce issues found in other training approaches, improving the overall efficiency of model training.
- **On-policy preference data collection challenges**: The community discussed the costs and challenges of collecting **on-policy preference data** for single-policy alignment datasets.
   - It was noted in the *An update on DPO vs PPO for LLM alignment* video that having diverse model generations can make *Ultrafeedback* easier to use, but single-policy focus might be necessary for consistent alignment.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.interconnects.ai/p/apple-intelligence">AI for the rest of us</a>: Apple Intelligence makes a lot of sense when you get out of the AI bubble. Plus, the cool technical details Apple shared about their language models &quot;thinking different.&quot;</li><li><a href="https://www.youtube.com/watch?v=-q79uzz1Wik">Interviewing Sebastian Raschka on the state of open LLMs, Llama 3.1, and AI education</a>: This week, I had the pleasure of chatting with Sebastian Raschka. Sebastian is doing a ton of work on the open language model ecosystem and AI research broad...</li><li><a href="https://github.com/allenai/open-instruct/pull/205">Add rejection sampling script by vwxyzjn · Pull Request #205 · allenai/open-instruct</a>: no description found
</li>
</ul>

</div>
  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1268287138563096750)** (56 messages🔥🔥): 

> - `Llama 3.1 evaluation and controversies`
> - `AI SDR fundraising`
> - `New player in text-to-image space: Black Forest Labs`
> - `LangGraph Studio announcement`
> - `Mixed-modal language modeling with Meta MoMa` 


- **Llama 3.1 under scrutiny**: Llama 3.1 has taken the world by storm but faces criticism for differences in quality when different inference providers use different implementations ([Together AI blog](https://www.together.ai/blog/llama-31-quality)).
   - Notable figures in the AI community have pointed out inaccuracies and potential hallucinations in Together AI's evaluations and claim cherry-picked results, emphasizing the importance of transparent methodology and rigorous data-based testing ([discussion thread](https://x.com/dzhulgakov/status/1818753731573551516)).
- **Sybill raises $11M for AI SDR**: Sybill announced raising **$11M in Series A funding** to build a personal assistant for every sales rep, led by **Greystone Ventures** and other notable VCs ([read more](https://x.com/asnani04/status/1818642568349204896)).
   - The market for AI-powered sales tools is heating up, and Sybill’s feature of cloning the seller's voice to draft relevant follow-ups was highlighted as particularly on-point.
- **Black Forest Labs emerges in text-to-image space**: Black Forest Labs launched with a new suite of SOTA text-to-image models called **FLUX.1**, which includes a **12B param model** available under non-commercial and open licenses on Huggingface ([announcement](https://x.com/iScienceLuvr/status/1819007823339999516) and [model weights](https://huggingface.co/black-forest-labs)).
   - The team consists of former Stable Diffusion members, and their **pro model** is already available for testing on Replicate.
- **LangGraph Studio: New Agent IDE**: LangChain announced **LangGraph Studio**, a specialized IDE for **agentic applications**, enabling better visualization, interaction, and debugging of LLM workflows ([announcement](https://x.com/LangChainAI/status/1819052975295270949)).
   - The tool integrates with **LangSmith** for collaboration and aims to make developing LLM applications more efficient and accessible.
- **Meta introduces MoMa for mixed-modal language modeling**: Meta announced **MoMa**, a new sparse early-fusion architecture for mixed-modal language modeling, improving pre-training efficiency ([paper](https://arxiv.org/pdf/2407.21770) and [announcement](https://x.com/victorialinml/status/1819037433251721304?s=46)).
   - MoMa employs a **mixture-of-expert (MoE)** framework with modality-specific expert groups, handling interleaved mixed-modal token sequences efficiently.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/allen_ai/status/1819077607897682156">Tweet from Ai2 (@allen_ai)</a>: After months of behind-the-scenes research, interviews, and labors of love, we’re delighted to debut Ai2’s new brand and website today.  Explore the evolution 🧵</li><li><a href="https://x.com/nisten/status/1818529201231688139">Tweet from nisten (@nisten)</a>: hacked bitnet for finetuning, ended up with a 74mb file. It talks fine at 198 tokens per second on just 1 cpu core. Basically witchcraft. opensourcing later via @skunkworks_ai base here: https://huggi...</li><li><a href="https://x.com/elonmusk/status/1818810438634946699?s=46">Tweet from Elon Musk (@elonmusk)</a>: @nmasc_ @KalleyHuang @steph_palazzolo The [Mis]Information strikes again.   xAI is not considering an acquisition of Character AI.</li><li><a href="https://x.com/TheNoahHein/status/1819098232636481711">Tweet from Noah Hein (@TheNoahHein)</a>: trying out the @bfl_ml flux-dev model on @replicate!  Here&#39;s a list of it&#39;s outputs, with the prompt, and a side-by-side comparison of the same prompt into MJ!  Flux is on the left, MJ on the ...</li><li><a href="https://x.com/Tim_Dettmers/status/1818282778057941042">Tweet from Tim Dettmers (@Tim_Dettmers)</a>: After 7 months on the job market, I am happy to announce: - I joined @allen_ai - Professor at @CarnegieMellon from Fall 2025 - New bitsandbytes maintainer @Titus_vK  My main focus will be to strengthe...</li><li><a href="https://x.com/llama_index/status/1819048068798616058">Tweet from LlamaIndex 🦙 (@llama_index)</a>: Today we’re excited to introduce @llama_index workflows - a new event-driven way of building multi-agent applications. Model each agent as a component that subscribes to events and emits events; you c...</li><li><a href="https://x.com/LangChainAI/status/1819052975295270949">Tweet from LangChain (@LangChainAI)</a>: 🚀Announcing LangGraph Studio: The first agent IDE  LangGraph Studio offers a new way to develop LLM applications by providing a specialized agent IDE that enables visualization, interaction, and debu...</li><li><a href="https://x.com/al">Tweet from undefined</a>: no description found</li><li><a href="https://x.com/jiayq/status/1818786673695809793">Tweet from Yangqing Jia (@jiayq)</a>: As an AI researcher and engineer, I fully respect together&#39;s achievement but would like to also point out the many cherrypicked errors. I am sure they are unintentional, but evaluation of LLMs is ...</li><li><a href="https://x.com/dzhulgakov/status/1818753736359178414">Tweet from Dmytro Dzhulgakov (@dzhulgakov)</a>: Example: AI researcher question “What is group query attention?”  Claim: Factually correct, and detailed answer  Reality: The answer implies that GQA is some form of sequence-sparse attention. However...</li><li><a href="https://x.com/basetenco/status/1819048091451859238">Tweet from Baseten (@basetenco)</a>: We&#39;re excited to introduce our new Engine Builder for TensorRT-LLM! 🎉  Same great @nvidia TensorRT-LLM performance—90% less effort.  Check out our launch post to learn more: https://www.baseten.c...</li><li><a href="https://x.com/dzhulgakov/status/1818753731573551516">Tweet from Dmytro Dzhulgakov (@dzhulgakov)</a>: This you? We ran your show-case example 3 times on Together playground, and it infinitely looped or answered incorrectly every time. Curious how that slipped through all 5 steps of your quality testin...</li><li><a href="https://x.com/togethercompute/status/1818706177238397155">Tweet from Together AI (@togethercompute)</a>: Recently there has been considerable discussion on differences in quality when different inference providers use different implementations of Meta&#39;s Llama 3.1 models.   In the blog post below, we ...</li><li><a href="https://x.com/ContextualAI/status/1819032988933623943">Tweet from Contextual AI (@ContextualAI)</a>: We’re excited to share today that we’ve raised $80M in Series A funding to accelerate our mission to change the way the world works through AI. Read more at our blogpost: https://contextual.ai/news/an...</li><li><a href="https://x.com/romainhuet/status/1814054938986885550">Tweet from Romain Huet (@romainhuet)</a>: @triviatroy @OpenAI The dollar price per image is the same for GPT-4o and GPT-4o mini. To maintain this, GPT-4o mini uses more tokens per image. Thank you for your observation!</li><li><a href="https://x.com/asnani04/status/1818642568349204896">Tweet from Nishit Asnani (@asnani04)</a>: 🚀 Big news! Sybill raised $11M in Series A funding, led by @greycroftvc , with participation from @neotribevc, Powerhouse VC, and Uncorrelated VC.   We&#39;re building a personal assistant for every ...</li><li><a href="https://x.com/victorialinml/status/1819037433251721304?s=46">Tweet from Victoria X Lin (@VictoriaLinML)</a>: 1/n Introducing MoMa 🖼, our new sparse early-fusion architecture for mixed-modal language modeling that significantly boosts pre-training efficiency 🚀 (https://arxiv.org/pdf/2407.21770). MoMa employ...</li><li><a href="https://x.com/StabilityAI/status/1819025550062850451">Tweet from Stability AI (@StabilityAI)</a>: We are excited to introduce Stable Fast 3D, Stability AI’s latest breakthrough in 3D asset generation technology. This innovative model transforms a single input image into a detailed 3D asset in just...</li><li><a href="https://x.com/character_ai/status/1819138734253920369?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Character.AI (@character_ai)</a>: Thrilled to share that we&#39;re open sourcing our innovative approach to prompt design! Discover how Prompt Poet is revolutionizing the way we build AI interactions in our latest blog post: https://r...</li><li><a href="https://x.com/robrombach/status/1819012132064669739">Tweet from Robin Rombach (@robrombach)</a>: 🔥 I am so damn excited to announce the launch of Black Forest Labs. We set ourselves on a mission to advance state-of-the-art, high-quality generative deep learning models for images and video, and m...</li><li><a href="https://x.com/lmsysorg/status/1819048821294547441">Tweet from lmsys.org (@lmsysorg)</a>: Exciting News from Chatbot Arena!  @GoogleDeepMind&#39;s new Gemini 1.5 Pro (Experimental 0801) has been tested in Arena for the past week, gathering over 12K community votes.  For the first time, Goo...</li><li><a href="https://x.com/iScienceLuvr/status/1819007823339999516">Tweet from Tanishq Mathew Abraham, Ph.D. (@iScienceLuvr)</a>: Black Forest Labs announces new suite of SOTA text-to-image models called FLUX.1   Best model FLUX.1[pro] behind API  FLUX.1[dev] is 12B param model under non-commercial license  FLUX.1[dev] is 12B pa...</li><li><a href="https://github.blog/news-insights/product-news/introducing-github-models/">Introducing GitHub Models: A new generation of AI engineers building on GitHub</a>: We are enabling the rise of the AI engineer with GitHub Models – bringing the power of industry leading large and small language models to our more than 100 million users directly on GitHub.</li><li><a href="https://www.together.ai/blog/llama-31-quality">Llama 3.1: Same model, different results. The impact of a percentage point.</a>: no description found</li><li><a href="https://x.com/GriffinAdams92/status/1819072387469516884">Tweet from Griffin Adams (@GriffinAdams92)</a>: Announcing Cold Compress 1.0 with @answerdotai  A hackable toolkit for using and creating KV cache compression methods.  Built on top of @cHHillee and Team’s GPT-Fast for torch.compilable, light-weigh...</li><li><a href="https://youtu.be/qP3rXJc_L5Y?si=z52-nyB0Ov0lUCkg">Self-directed Synthetic Dialogues (and other recent synth data)</a>: A talk covering a recent synthetic data project we launched. Find the details below.https://arxiv.org/abs/2407.18421Slides: https://docs.google.com/presentat...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/s/jHnSGxfHRj">Reddit - Dive into anything</a>: no description found</li><li><a href="https://replicate.com/black-forest-labs/flux-pro">black-forest-labs/flux-pro – Run with an API on Replicate</a>: no description found
</li>
</ul>

</div>
  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1268301305160667199)** (3 messages): 

> - `Async functionality for BedrockConverse`
> - `LongRAG paper by @Ernestzyj`
> - `@llama_index workflows` 


- **Async functionality now in BedrockConverse**: [Async methods](https://t.co/rn3sAKG05N) for **BedrockConverse** LLM have been implemented, resolving issues [#10714](https://github.com/run-llama/llama_index/issues/10714) and [#14004](https://github.com/run-llama/llama_index/issues/14004).
   - *This contribution was greatly appreciated by the team for enhancing user experience.*
- **LongRAG paper simplifies long-context LLMs**: The **LongRAG** paper by @Ernestzyj proposes indexing and retrieving larger document chunks to better utilize long-context LLMs.
   - *This approach aims to ease the retriever’s tasks, enhancing the retrieval-augmented generation (RAG) process.*
- **@llama_index introduces workflows**: @llama_index [workflows](https://t.co/Ebme9eRvMb) enable event-driven multi-agent applications, allowing agents to subscribe to and emit events.
   - This new approach offers a **readable and Pythonic** way to build complex orchestration.



**Link mentioned**: <a href="https://t.co/rn3sAKG05N">feat: ✨ Implement async functionality in `BedrockConverse` by AndreCNF · Pull Request #14326 · run-llama/llama_index</a>: Description Implement async methods for the BedrockConverse LLM. Fixes #10714 Fixes #14004 New Package? Did I fill in the tool.llamahub section in the pyproject.toml and provide a detailed README.m...

  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1268282745189302382)** (47 messages🔥): 

> - `Alternatives to RagApp`
> - `Generating Images with LlamaParse`
> - `Stable Versions of LlamaIndex`
> - `Handling Agent Errors in ReAct`
> - `Configuration in LlamaIndex` 


- **Searching Alternatives to RagApp**: A user inquired about alternatives to RagApp and discussed the usefulness of `create-llama` despite some install issues with Poetry.
- **Generating Images with LlamaParse**: Users discussed methods for generating images with LlamaParse, referencing [GitHub examples](https://github.com/run-llama/llama_parse/blob/main/examples/demo_json.ipynb) and additional resources.
- **Identifying Stable Versions of LlamaIndex**: A user questioned how to identify the 'stable' version of LlamaIndex, and it was clarified that installing via pip ensures the latest stable version.
   - Further comments emphasized that the 'stable' version typically refers to the latest release on PyPI.
- **Handling Errors in ReAct Agent**: A user explored making ReAct agents function without invoking tools and discussed alternative approaches like `SimpleChatEngine` or handling agent errors more gracefully.
   - Suggestions included using `llm.chat(chat_messages)` for a simpler setup and exploring the function calling agent for better tool handling.
- **Configuring Parameters in LlamaIndex**: There was a discussion on setting parameters like `max_input_size` and chunk overlap in LlamaIndex v10.x after the removal of the `PromptHelper`.
   - Alternatives like passing configurations directly to node parsers or using response synthesizers were suggested.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/run-llama/llama_parse/blob/main/examples/multimodal/multimodal_rag_slide_deck.ipynb">llama_parse/examples/multimodal/multimodal_rag_slide_deck.ipynb at main · run-llama/llama_parse</a>: Parse files for optimal RAG. Contribute to run-llama/llama_parse development by creating an account on GitHub.</li><li><a href="https://github.com/run-llama/llama_parse/blob/main/examples/demo_json.ipynb">llama_parse/examples/demo_json.ipynb at main · run-llama/llama_parse</a>: Parse files for optimal RAG. Contribute to run-llama/llama_parse development by creating an account on GitHub.</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/vector_stores/CassandraIndexDemo/">Cassandra Vector Store - LlamaIndex</a>: no description found
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1268541355932651553)** (1 messages): 

> - `DSPy`
> - `Prompt Optimizing`
> - `Prompt Rewriting`
> - `LlamaIndex` 


- **Comparing DSPy prompt optimization with LlamaIndex**: A member inquired about others' experiences with **DSPy** and requested opinions on its **prompt optimizing** versus **prompt rewriting** capabilities in **LlamaIndex**.
- **DSPy Prompt Optimization versus LlamaIndex**: Interest was expressed in comparing **prompt optimization** and **prompt rewriting** between **DSPy** and **LlamaIndex**.


  

---



### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1268508658157883403)** (16 messages🔥): 

> - `Embedding content structure`
> - `Table and checkbox detection in PDFs`
> - `AI Hackathon Series Tour`
> - `Ivan as a Gamer`
> - `Cows as Pets` 


- **Discussion on leveraging content structure for embeddings**: Queries about the impact of new lines, page-breaks, and special symbols on embedding performance was discussed, with **Nils Reimers** confirming these elements are removed automatically in English and multilingual models.
   - *No need to preprocess the text extensively for embedding models* was the key takeaway, with models being robust enough to handle noisy data.
- **Detect and extract table and checkbox data from PDFs**: A member sought recommendations for models to detect tables and checkboxes from **non-readable PDFs** to extract into text or docx formats.
   - The suggestion highlighted the effectiveness of using **unstructured.io** for converting PDF data into JSON format, evidenced by a similar ongoing project within the community.
- **Join the AI Hackathon Series Tour at Google**: The **AI Hackathon Series Tour** invites registrations for an event at Google, encompassing innovative AI projects and a competition over **3 days**.
   - The event provides a creative and competitive platform, concluding with the **PAI Palooza**, showcasing top AI startups and projects from the host city.
- **Ivan's gaming background revealed**: A LinkedIn [article](https://www.linkedin.com/pulse/from-gamer-ai-unicorn-co-founder-conversation-coheres-f2tte) shared revealed **Ivan's** past as a gamer, surprising some community members.
   - **Karthik_99_** expressed amazement on discovering Ivan's transition from gaming to AI co-founder.
- **Taking care of cows**: A lighthearted comment on owning cows led to the observation that **they are a lot of work**, addressing a member's jealousy.



**Link mentioned**: <a href="https://lu.ma/2svuyacm">Techstars StartUp Weekend - PAI Palooza &amp; GDG Build with AI—Mountain View · Luma</a>: This AI Hackathon Series Tour is a groundbreaking, multi-city event that spans the United States, bringing together the brightest minds in artificial…

  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1268336626392629248)** (17 messages🔥): 

> - `Training LLMs for Arabic Dialects`
> - `Joining the Cohere Research Community`
> - `Training LLMs for JSON Output` 


- **Training LLMs for Arabic Dialects**: A member queried how models like **Aya** can generate fluent responses in different Arabic dialects without explicit dialect information in the training prompts.
   - They expressed surprise that a prompt in English asking for an Egyptian dialect would correctly generate text in that form.
- **Joining the Cohere Research Community**: A member reported issues joining the Cohere research community and being signed up for newsletters instead.
   - Responses mentioned the manual review process and apologized for delays, asking the member to DM their email for a status update.
- **Training LLMs for JSON Output**: A member asked about training an LLM to convert free-form search queries into structured JSON for Apache Solr input.
   - It was suggested they could manually label data, find labeled data, or generate data synthetically, and to check out [Cohere's documentation](https://docs.cohere.com/docs/structured-outputs-json) for producing structured outputs.



**Link mentioned**: <a href="https://docs.cohere.com/docs/structured-outputs-json">Structured Generations (JSON)</a>: no description found

  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1268565050465980507)** (15 messages🔥): 

> - `August OH event`
> - `Ukrainian/Russian language support degradation`
> - `Citation_quality settings`
> - `Speed optimization for Cohere Cloud` 


- **Invitation to August OH Event**: A member invited others to join the [August OH event](https://discord.com/events/954421988141711382/1265012161965461625/1275137202585600000) for a meetup.
   - They encouraged participation by suggesting the event would be a fun hangout.
- **Degradation in Ukrainian/Russian Language Support**: A user reported experiencing degradation in **Ukrainian/Russian language support** on Cohere Cloud, resulting in broken characters.
   - The issue was linked to the **citation_quality** setting, and switching from **fast** to **accurate** resolved it, although this affected response speed.


  

---


### **Cohere ▷ #[cohere-toolkit](https://discord.com/channels/954421988141711382/1254901651081269268/1268504819094650901)** (3 messages): 

> - `devcontainer issue`
> - `pydantic validation error`
> - `repository update`
> - `team response` 


- **Validation errors block repository setup**: A member reported issues running the latest version of the repository in a devcontainer, encountering various **pydantic validation errors** related to the `Settings` class.
   - *Six validation errors* were noted, specifically missing fields like **auth.enabled_auth** and **auth.google_oauth**, which caused `make setup` to fail.
- **Team swiftly addresses devcontainer issues**: The issue was acknowledged quickly by another member, promising that the team would look into and resolve the errors.
   - An update followed shortly, confirming that the team is already working on a fix.



**Link mentioned**: <a href="https://errors.pydantic.dev/2.8/v/missing">Redirecting...</a>: no description found

  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1268322459426361446)** (45 messages🔥): 

> - `Pydantic type error in LangChain`
> - `Executing tools in LangChain`
> - `LangSmith API key issue`
> - `LangChain and deployment`
> - `LangChain documentation and resources` 


- **Pydantic version conflicts cause errors**: A member encountered a `pydantic.v1.error_wrappers.ValidationError` despite having installed Pydantic v2, leading to a mismatch in expected types and validation errors during execution in LangChain.
- **Tool Execution Issues in LangChain**: LangChain tools encounter issues when executing `execute_tools` node, causing failures due to input type mismatches and validation errors, despite correct Pydantic validation of inputs beforehand.
- **LangSmith API key setup troubles**: A user struggled with a `403 Client Error: Forbidden` when trying to deploy an LLM with LangSmith, suspecting it was an issue related to the API key configuration.
- **LangChain resource suggestions and alternatives**: Members discussed different sources for learning about LangChain and alternative LLM inference services, recommending OpenAI and TogetherAI for free or affordable usage with LangChain's prompt classes.
- **LangChain documentation and error handling**: Users were directed to example resources on LangChain's GitHub to troubleshoot various issues and avoid common errors with tool use and API integrations.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/stabilityai/stable-fast-3d">Stable Fast 3D - a Hugging Face Space by stabilityai</a>: no description found</li><li><a href="https://js.langchain.com/v0.2/docs/tutorials/llm_chain/#using-language-models">Build a Simple LLM Application with LCEL | 🦜️🔗 Langchain</a>: In this quickstart we’ll show you how to build a simple LLM application</li><li><a href="https://github.com/langchain-ai/langgraph/blob/main/examples/plan-and-execute/plan-and-execute.ipynb">langgraph/examples/plan-and-execute/plan-and-execute.ipynb at main · langchain-ai/langgraph</a>: Build resilient language agents as graphs. Contribute to langchain-ai/langgraph development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langgraph/blob/main/examples/tool-calling.ipynb">langgraph/examples/tool-calling.ipynb at main · langchain-ai/langgraph</a>: Build resilient language agents as graphs. Contribute to langchain-ai/langgraph development by creating an account on GitHub.</li><li><a href="https://v02.api.js.langchain.com/index.html">LangChain.js - v0.2.12</a>: no description found</li><li><a href="https://github.com/langchain-ai/langgraph/blob/main/examples/tool-calling-errors.ipynb">langgraph/examples/tool-calling-errors.ipynb at main · langchain-ai/langgraph</a>: Build resilient language agents as graphs. Contribute to langchain-ai/langgraph development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1268560571238973531)** (2 messages): 

> - `Streaming Support in FastAPI LangChain Application`
> - `Using /stream_events endpoint in langserve v2` 


- **Adding Streaming Support to FastAPI LangChain Application**: A user proposed a design to add asynchronous streaming support to a FastAPI application with LangChain, focusing on using Redis as a message broker for real-time token generation.
   - The design includes keeping existing synchronous endpoints, adding new streaming endpoints, and updating LangChain agents to publish chunks and full responses to Redis.
- **Using /stream_events endpoint in langserve v2**: A user asked for guidance on how to use the `/stream_events` endpoint in langserve version v2, mentioning that they couldn't find any documentation.
   - They expressed difficulty in finding information and sought help from the community.


  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1268412970208071771)** (2 messages): 

> - `LangGraph design pattern`
> - `Advanced research assistant and search engine`
> - `GPT-4o`
> - `Claude 3 Opus`
> - `Llama 3.1` 


- **LangGraph design pattern for user apps**: A member shared a LangGraph design pattern for easy integration into user-facing apps like web-chats or Telegram/Whatsapp bots, with a detailed example available [on GitHub](https://github.com/TonySimonovsky/ai-champ-design-patterns/blob/main/ai-agents/LangGraph-multi-agent-user-facing.ipynb).
   - *“Here's a LangGraph design pattern that can be easily integrated into your user-facing apps with streaming.”*
- **Rubik's AI Pro offers beta testing with premium models**: A member invited others to beta test an advanced research assistant and search engine, offering 2 months of free premium that includes **Claude 3 Opus**, **GPT-4o**, **Gemini 1.5 Pro**, and other models via [Rubik's AI](https://rubiks.ai/).
   - *“Use the promo code `RUBIX` to get 2-months of free premium to test new features and expert models.”*


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://rubiks.ai/">Rubik's AI - AI research assistant & Search Engine</a>: no description found</li><li><a href="https://github.com/TonySimonovsky/ai-champ-design-patterns/blob/main/ai-agents/LangGraph-multi-agent-user-facing.ipynb">ai-champ-design-patterns/ai-agents/LangGraph-multi-agent-user-facing.ipynb at main · TonySimonovsky/ai-champ-design-patterns</a>: Contribute to TonySimonovsky/ai-champ-design-patterns development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1268578388851822695)** (1 messages): 

> - `Moye Launcher`
> - `Digital detox tools` 


- **Moye Launcher Promotes Digital Detox**: Moye Launcher is a minimalist Android launcher with built-in AI-powered digital detox tools, aiming to reduce excessive screen time. It eliminates the app drawer to make apps less accessible, encouraging less impulsive app use.
   - The launcher aims to address the top three reasons for unproductive screen time, such as auto-clicking due to boredom and lack of accountability, by removing easily accessible app icons and providing usage feedback.
- **Digital Detox Tools Explained**: Moye Launcher uses AI tools to help users stay accountable and avoid unnecessary app usage, providing reminders and tracking usage.
   - These features target the main reasons for unproductive screen time: auto-clicking of apps, lack of a 'watchman,' and forgetting why an app was opened initially.



**Link mentioned**: <a href="https://play.google.com/store/apps/details?id=in.noxchat.moyemoyelauncher">Moye Launcher: Digital Detox - Apps on Google Play</a>: no description found

  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1268282400451072021)** (39 messages🔥): 

> - `Lobe interface`
> - `Librechat capabilities`
> - `Big-agi features`
> - `Msty tool integrations with Obsidian`
> - `Llama 405B Instruct providers` 


- **Big-agi expands model capabilities with BEAM**: Big-agi introduces a 'persona creator' that allows users to generate prompts from YouTube videos or text and the BEAM feature to call 2/4/8 models simultaneously and merge their responses.
   - However, it lacks server saving and easy syncing capabilities.
- **Msty integrates Obsidian and websites**: Msty offers slick integrations with Obsidian and website access, though its parameter settings are reportedly easily forgotten.
   - Despite minor polish issues, many users find it appealing and are considering switching to it.
- **Llama 405B Instruct providers and quantization**: There are no FP16 providers for Llama 405B on OpenRouter, and FP8 quantization, recommended by Meta, runs more efficiently than FP16.
   - SambaNova Systems runs in bf16 but is limited to 4k context length, and hosting in bf16 is computationally expensive.
- **API Integration with OpenRouter under Beta**: Users seeking API integration to handle rate limits and integrate OpenAI and Claude API are advised to email support to join the Beta waitlist.
   - Detailed requests can be directed to support@openrouter.ai for assistance.
- **OpenRouter website faces occasional regional issues**: The OpenRouter website experiences occasional regional connection issues but generally remains operational.
   - Users can check status updates for real-time operational information via the [OpenRouter status page](https://status.openrouter.ai/).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://sambanova.ai/">SambaNova Systems | Revolutionize AI Workloads</a>: Unlock the power of AI for your business with SambaNova's enterprise-grade generative AI platform. Discover how to achieve 10x lower costs &amp; unmatched security.</li><li><a href="https://openrouter.ai/privacy#_4_-user-rights-and-choices">OpenRouter</a>: LLM router and marketplace</li><li><a href="https://status.openrouter.ai/">OpenRouter Status</a>: OpenRouter Incident History</li><li><a href="https://github.com/oobabooga/text-generation-webui/pull/5677">DRY: A modern repetition penalty that reliably prevents looping by p-e-w · Pull Request #5677 · oobabooga/text-generation-webui</a>: Looping is an undesirable behavior where the model repeats phrases verbatim that have previously occurred in the input. It affects most models, and is exacerbated by the use of truncation samplers....
</li>
</ul>

</div>
  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1268338710919909467)** (23 messages🔥): 

> - `Open Interpreter Response Delays`
> - `Groq Profile Contribution`
> - `Accessibility Roundtable Announcement`
> - `House Party Event`
> - `Community Building Focus` 


- **Open Interpreter Response Delays**: Members are concerned about a delayed response from Ben Steinher of Open Interpreter; he was expected to respond 'early next week' on the 11th of July.
- **Groq Profile Contribution Celebrated**: A member announced a new PR for a Groq profile, describing it as a great way to contribute to the **Open Interpreter** project.
   - *Heyyy we love Groq around these ends 😁*
- **Accessibility Roundtable on August 22nd**: [Accessibility Roundtable](https://discord.gg/open-interpreter-1146610656779440188?event=1268579948248170663) announced for August 22nd at noon PST, inviting members to participate in a discussion about accessibility.
- **Excitement for House Party Event**: Members reminded others about the House Party event happening in 4 hours, providing a [link to the event](https://discord.gg/zMwXfHwz?event=1267524800163610815).
   - There appeared to be some confusion about the event's start time, but the issue was resolved and participants joined the correct voice channel.
- **Community Building AI Focus**: A member shared their AI project's focus on community-building, specifically fostering **backyard barbecue neighborhood friendships**.
   - "*This is so important!! And community block parties without an HOA lol*"


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://youtu.be/O_Q1hoEhfk4">Friend Reveal Trailer</a>: not imaginary. preorder now at friend.com.</li><li><a href="https://github.com/OpenInterpreter/open-interpreter/pull/1376">Added Groq profile and flag by MikeBirdTech · Pull Request #1376 · OpenInterpreter/open-interpreter</a>: Added Open Interpreter groq profile support via default groq.py file, updated parser for CLI shortcut in start_terminal_interface.py to accept --groq flag to apply the profile Describe the changes ...
</li>
</ul>

</div>
  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1268343270690521199)** (8 messages🔥): 

> - `Model Selection Questions`
> - `01 Workflows and Scheduling`
> - `iKKO ActiveBuds`
> - `01 Shipping Status`
> - `Earbuds with Camera` 


- **Confusion Around Model Selection and API Key Use**: A member expressed confusion about selecting the model string and why an OpenAI API key is needed when running '01 --local.'
   - They cited their lack of knowledge about these basic concepts.
- **01 Workflows and Scheduling Capabilities?**: A member inquired if OpenInterpreter (OI) can save workflows and set up task schedules.
   - The question remains unanswered within the given messages.
- **01 on iKKO ActiveBuds Would Be Dope**: Members discussed the potential integration of 01 on the [iKKO ActiveBuds](https://www.ikkoaudio.com/collections/tws-earbuds/products/activebuds), which boasts features like an AI-Smart System, AMOLED Touchscreen, and High-Resolution Sound.
   - The idea was endorsed as feasible and exciting for improved Human-Computer Interaction (HCI).
- **Immediate Need for 01 Shipping Information**: A member asked about the shipping status of 01 since it is already August.
   - [Response linked](https://discord.com/channels/1146610656779440188/1194880263122075688/1266055462063964191) without further details provided in the conversation.
- **Desire for Earbuds with Camera**: Members expressed a desire for earbuds featuring a camera that can capture context while conversing with an LLM.
   - The idea includes a push/tap feature to activate the camera, enhancing Human-Computer Interaction capabilities.



**Link mentioned**: <a href="https://www.ikkoaudio.com/collections/tws-earbuds/products/activebuds">ActiveBuds: AI-Smart Earphones with ViVid Touchscreen | iKKO Audio</a>: AI Voice Assistant by ChatGPT-4o. High-bitrate Bluetooth pairing for high-resolution wireless audio among earphones, speakers, smartphones. 45 languages translations. Portable memos for ChatGPT and tr...

  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1268334732559515731)** (18 messages🔥): 

> - `Mojo Threads`
> - `Max and Mojo Packaging`
> - `Tier Chart Discussion`
> - `Existential Quantifiers` 


- **Mojo lacks explicit thread support**: A member asked if Mojo supports threads and another member confirmed **Mojo does not currently expose thread support** to users.
   - However, **calling fork()** and getting threads that way is tolerated in the compiled version.
- **MAX and Mojo packaging changes announced**: Announcements were made about **changes to MAX and Mojo packaging** starting with version 0.9 of the `modular` CLI, making authentication unnecessary to download MAX and Mojo.
   - [Further changes](https://docs.modular.com/max/faq#why-bundle-mojo-with-max) include merging Mojo nightly packages with MAX and transitioning to a new `magic` CLI for easier integration into the Conda ecosystem.
- **Tier chart discussion causes confusion**: A discussion ensued about a tier chart, with members questioning its representation and noting that it did not reflect a **'level of abstraction'**.
   - Suggestions were made to replace the entire iceberg with a fire emoji for simplicity.



**Link mentioned**: <a href="https://docs.modular.com/max/faq#why-bundle-mojo-with-max),">MAX FAQ | Modular Docs</a>: Answers to questions we expect about MAX Engine.

  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1268495275862003754)** (4 messages): 

> - `CrazyString gist update`
> - `Unicode based indexing` 


- ****CrazyString Gist Adds Unicode Support****: [CrazyString gist](https://gist.github.com/mzaks/78f7d38f63fb234dadb1dae11f2ee3ae) now includes support for Unicode-based indexing, along with small string optimization and full UTF-8 compatibility.
   - *Mojo String with small string optimisation* and potential full UTF-8 support described in the update.
- **Math and Computation as Universal Languages**: A member remarked that 'Math is the universal language and Computation is the universal action'.



**Link mentioned**: <a href="https://gist.github.com/mzaks/78f7d38f63fb234dadb1dae11f2ee3ae">Mojo String with small string optimisation and potential full UTF-8 support</a>: Mojo String with small string optimisation and potential full UTF-8 support - crazy_string.mojo

  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1268530304759365746)** (5 messages): 

> - `Installing max on Mac M1 Max`
> - `Mojo compatibility with Python` 


- **Issue with Installing max on Mac M1 Max**: A member reported facing issues while trying to install max on a Mac M1 Max device.
   - Another member suggested [following this fix for Python installation](https://modul.ar/fix-python) to potentially resolve the problem.
- **Mojo aims to be a superset of Python**: [Mojo](https://modul.ar/fix-python) is designed to be compatible with existing Python programs, allowing programmers to use it immediately while leveraging the vast ecosystem of Python packages.
   - Mojo is in early development and many Python features are not yet implemented, but it allows importing Python modules, calling Python functions, and interacting with Python objects.



**Link mentioned**: <a href="https://modul.ar/fix-python">Python integration | Modular Docs</a>: Using Python and Mojo together.

  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1268292373130117131)** (8 messages🔥): 

> - `Automated Training Run Termination`
> - `Early Stopping in Axolotl`
> - `Manual Run Termination`
> - `Output Mask Field Proposal` 


- **Axolotl Implements Early Stopping**: A member inquired if Axolotl has features to automatically terminate training runs when **loss converges asymptotically** or **validation loss increases**.
   - Another member confirmed that **Axolotl supports early stopping** for this purpose.
- **Manually Terminate and Save Current LoRA Adapter**: A member asked if they could manually terminate a run while saving the most recently trained **LoRA adapter** instead of canceling the whole run.
   - There was no follow-up from the community on this request.
- **Output Mask Field in SharedGPT**: A member proposed adding an **"output mask" field** in every turn of the **SharedGPT** to allow selective training on outputs.
   - They explained that this would let the AI make and subsequently learn from mistakes in the masked fields.


  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1268362706038034604)** (5 messages): 

> - `Chat templates documentation`
> - `Preprocessing step issue` 


- **Documentation for new chat templates needed**: A member mentioned the need for **documentation for new chat templates**, stating that it was challenging to understand how they work and how to extract specific parts of a message.
   - *Another member noted that they had already written some documentation for themselves and would try to add it to the official docs.*
- **Bug in preprocessing step with older version**: A member requested an example to run just the preprocess step on an older version of the main branch to identify a bug causing improper tokenization.
   - *They indicated that the bug needs to be fixed as it only triggers in some cases.*


  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1268283024055861299)** (6 messages): 

> - `Pad Token Repetition in Model Training`
> - `Dataset Viewers for Conversation Cleaning`
> - `Training and Finetuning Llama3` 


- **Issues with Pad Token Repetition in Model Training**: A member discussed the occurrence of `<pad>` repetition likely due to not using sample packing and possibly related to enabling eager attention instead of flash.
   - *Caseus* mentioned that the pad tokens should be masked out from the label to prevent this issue.
- **Need for Better Dataset Viewers**: A member sought recommendations for a dataset viewer that allows both viewing and editing conversations beyond simple jsonl format.
   - [Argilla](https://argilla.io/) was suggested, highlighting its collaboration tool capabilities for AI engineers and integration with Hugging Face, but this didn't meet the member's needs.
- **Finetuning Llama3 for Translation**: A member asked for advice on the best dataset for finetuning Llama3 as a translation model, citing their current limit of 8 billion parameters and showcasing their dataset on Hugging Face.
   - *Diabolic6045* shared a Sanskrit text dataset on [Hugging Face](https://huggingface.co/datasets/diabolic6045/Sanskrit-llama) used for translation, including both the Sanskrit source and English translation.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://argilla.io/">The tool where experts improve AI models</a>: Argilla is a collaboration tool for AI engineers and domain experts that strive for data quality, ownership, and efficiency.</li><li><a href="https://huggingface.co/blog/dvilasuero/argilla-2-0">🔥 Argilla 2.0: the data-centric tool for AI makers 🤗 </a>: no description found</li><li><a href="https://huggingface.co/datasets/diabolic6045/Sanskrit-llama">diabolic6045/Sanskrit-llama · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[replicate-help](https://discord.com/channels/1104757954588196865/1197414694248534116/1268318167785013248)** (1 messages): 

> - `Serverless GPUs`
> - `AI Infrastructure`
> - `Inferless report`
> - `Cold starts`
> - `Autoscaling tests` 


- **Inferless Publishes New Serverless GPUs Report**: [Inferless](https://www.inferless.com/learn/the-state-of-serverless-gpus-part-2) published a follow-up report on the state of **Serverless GPUs**, highlighting significant changes and improvements since their previous report six months ago.
   - The report gained traction on [Hacker News](https://news.ycombinator.com/item?id=35738072) and includes insights from hundreds of engineers deploying machine learning models in production.
- **Cold Starts and Autoscaling Tests in New Report**: The new Inferless report discusses **cold starts** and **autoscaling tests** across different serverless GPU providers.
   - These insights help developers make informed decisions when choosing their serverless provider.



**Link mentioned**: <a href="https://www.inferless.com/learn/the-state-of-serverless-gpus-part-2">Serverless GPU Part 2 Benchmarking: A Comprehensive Comparison of Performance &amp; Pricing</a>: Dive into an in-depth review of Serverless GPU platforms. Explore cold-start times, integration challenges, pricing comparison and auto-scaling capabilities. Make informed choices with our detailed an...

  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-help-bot](https://discord.com/channels/1104757954588196865/1225300056442409040/1268390021938020373)** (4 messages): 

> - `Gemma2 models training`
> - `Eager attention implementation`
> - `flash_attention_2`
> - `AutoModelForCausalLM` 


- **Training Gemma2 Models: Use Eager Attention**: It is strongly recommended to train Gemma2 models with the `eager` attention implementation instead of `flash_attention_2` by using `AutoModelForCausalLM.from_pretrained('<path-to-checkpoint>', attn_implementation='eager')`.
- **Eager Attention Over Flash_Attention_2 for Gemma2**: The `eager` attention implementation should be used over `flash_attention_2` for training Gemma2 models to ensure optimal performance.
   - A detailed [example code](https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=71bfdef0-8986-4d0c-a882-839872185c7e) demonstrates how to set this in the `AutoModelForCausalLM`.



**Link mentioned**: <a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=71bfdef0-8986-4d0c-a882-839872185c7e)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.

  

---



### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1268316060315025580)** (10 messages🔥): 

> - `Saving/Loading OptimizerResult`
> - `Improving JSON Parsing`
> - `Parallel Execution in DSPy Module`
> - `LiteLLM Proxy Issues with Non-OpenAI Models`
> - `DSPy with BIG-Bench via Weights & Biases` 


- **Saving/Loading OptimizerResult for Typed Optimizers**: A user inquired whether there is a method to save/load **OptimizerResult** for typed optimizers similar to untyped optimizers.
- **Schema-Aligned Parsing to Reduce JSON Errors**: A user proposed moving to [Schema-Aligned Parsing](https://www.boundaryml.com/blog/schema-aligned-parsing) to reduce unnecessary retries due to **bad JSON output**, noting it would also consume fewer tokens.
   - They lamented that their **TypedPredictor** ends up with a large JSON schema and this method could be more efficient.
- **Parallel Execution in DSPy Module**: A user asked if it's possible to run `dspy.Predict` in parallel within a module, showing an example where they wish to parallelize the `for c in criteria` loop.
- **LiteLLM Proxy Issues with Non-OpenAI Models**: A user reported encountering errors when using LiteLLM proxy with non-OpenAI models such as **Claude**, **mistral**, and **llama** models, despite it working well for **OpenAI models**.
   - They shared the code used: `dspy.OpenAI(model = 'gpt-3.5-turbo', api_base = BASE_API, max_tokens = 1024)`.
- **DSPy Integration with BIG-Bench and Weights & Biases**: A user found an example on [Twitter](https://x.com/soumikRakshit96/status/1816522389712462326) on how to use **DSPy** for causal reasoning tasks from **BIG-Bench Hard** and evaluate via **Weights & Biases Weave**.
   - However, they encountered an `OpCallError` due to an unexpected keyword argument '**system_prompt**' while executing the related [Colab notebook](https://colab.research.google.com/github/soumik12345/prompt-engineering-recipes/blob/main/notebooks/dspy/00_big_bench.ipynb#scrollTo=_vp8h91Uy0_F).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/soumikRakshit96/status/1816522389712462326">Tweet from GeekyRakshit (e/mad) (@soumikRakshit96)</a>: 🍀 DSPy is a framework that pushes modular &#34;programming&#34; models for prompting and lets us optimize our prompting strategies automatically using a teleprompter.  🧑‍💻 I created an example demo...</li><li><a href="https://colab.research.google.com/github/soumik12345/prompt-engineering-recipes/blob/main/notebooks/dspy/00_big_bench.ipynb#scrollTo=_vp8h91Uy0_F">Google Colab</a>: no description found</li><li><a href="https://www.boundaryml.com/blog/schema-aligned-parsing">Prompting vs JSON Mode vs Function Calling vs Constrained Generation vs SAP</a>: no description found
</li>
</ul>

</div>
  

---


### **DSPy ▷ #[random](https://discord.com/channels/1161519468141355160/1202371260873707520/1268434728009072731)** (1 messages): 

> - `Effortless AI article`
> - `Chatmangpt features` 


- **Effortless AI with Chatmangpt**: A [LinkedIn article](https://www.linkedin.com/pulse/effortless-ai-harness-power-simplicity-chatmangpts-fully-chatman--eamnc/) discusses the simplicity and power of Chatmangpt for harnessing AI capabilities effortlessly.
- **Chatmangpt features overview**: The article emphasizes how Chatmangpt's features integrate seamlessly into existing workflows, maximizing efficiency and productivity.


  

---


### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/1268298717551919124)** (8 messages🔥): 

> - `Integration of DSPy with symbolic learner`
> - `True Agentic Behavior`
> - `Self-Adapting AI Agents`
> - `Agent Zero`
> - `Novel Meta-Rewarding in Self-Improvement of LLMs` 


- **DSPy integrates with Symbolic Learner**: Members are excited about the potential of integrating DSPy with a symbolic learner, anticipating significant advancements.
   - One comment expressed excitement about the development, suggesting this could be a major leap forward.
- **Microsoft's Self-Adapting AI Agents Break New Ground**: A shared [Microsoft Research blog post](https://www.microsoft.com/en-us/research/blog/tracing-the-path-to-self-adapting-ai-agents/) highlights advancements in self-adapting AI agents, suggesting profound implications for the workplace.
   - The blog emphasizes that the games industry has historically driven AI innovation, culminating in modern applications like ChatGPT and Microsoft Copilots.
- **Agent Zero Debuts**: Agent Zero has been mentioned as the first production version tested by users, showcasing significant potential.
   - Opinions suggest that agents like Agent Zero are paving the way for AI to take on more roles in the workplace.
- **Meta-Rewarding Improves Self-Judgment in LLMs**: New research on [arXiv](https://arxiv.org/abs/2407.19594) introduces a Meta-Rewarding step enhancing the judgment capabilities of LLMs during the self-improvement process.
   - This method led to substantial win rate improvements on benchmarks like AlpacaEval 2, demonstrated by models such as Llama-3-8B-Instruct.
- **MindSearch: LLM-Based Multi-Agent Framework**: A recent [paper on arXiv](https://arxiv.org/abs/2407.20183) introduces MindSearch, which mimics human cognitive processes in web information seeking and integration using LLM-based multi-agent frameworks.
   - The study addresses challenges in information retrieval, noise management, and context handling, aiming to enhance the capabilities of modern search-assisted models.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2407.20183">MindSearch: Mimicking Human Minds Elicits Deep AI Searcher</a>: Information seeking and integration is a complex cognitive task that consumes enormous time and effort. Inspired by the remarkable progress of Large Language Models, recent works attempt to solve this...</li><li><a href="https://arxiv.org/abs/2407.19594">Meta-Rewarding Language Models: Self-Improving Alignment with LLM-as-a-Meta-Judge</a>: Large Language Models (LLMs) are rapidly surpassing human knowledge in many domains. While improving these models traditionally relies on costly human data, recent self-rewarding mechanisms (Yuan et a...</li><li><a href="https://www.microsoft.com/en-us/research/blog/tracing-the-path-to-self-adapting-ai-agents/">Discover Trace, a new framework for AI optimization from language models to robot control</a>: Introducing Trace, Microsoft and Stanford University&#039;s novel AI optimization framework, now available as a Python library. Trace adapts dynamically and optimizes a wide range of applications from...
</li>
</ul>

</div>
  

---


### **DSPy ▷ #[jobs](https://discord.com/channels/1161519468141355160/1211763460480827402/1268339802043056229)** (2 messages): 

> - `Official Job Board Setup`
> - `Bounties for Tutorial Blog Posts` 


- **Official Job Board Setup Announced**: An official job board is being set up, and members are invited to list their jobs for free by sending a DM.
- **Bounties for Tutorial Blog Posts**: A call was made for members interested in claiming bounties for writing tutorial blog posts.


  

---


### **DSPy ▷ #[colbert](https://discord.com/channels/1161519468141355160/1250300504462856265/)** (1 messages): 

amey_86281: Has anyone used Colbert Embeddings and store the embeddings in Pinecone ?
  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1268302812643594382)** (2 messages): 

> - `NVIDIA's impact on taxpayer money`
> - `Discord rules reminder by George Hotz` 


- ****NVIDIA Taxpayer Money Love****: A user expressed affection for taxpayer money being directed toward **NVIDIA**.
- ****George Hotz Reminds of Discord Rules****: George Hotz reminded users of the **discord rules** emphasizing that the chat is for **tinygrad development and usage discussions**.


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1268394846691790869)** (11 messages🔥): 

> - `GPT-2 Slowdown`
> - `Embedding/Argmax Inefficiency`
> - `Setup Environment for Tinygrad`
> - `Bounty for Embeddings`
> - `Cumsum O(n) Complexity` 


- **GPT-2 Slowed by Embedding/Argmax Bottleneck**: A user identified that the use of `Tensor.arange` in GPT-2 implementation results in inefficiencies, slowing down the model ([Issue #1612](https://github.com/tinygrad/tinygrad/issues/1612)).
   - The problem stems from the **O(n^2)** complexity due to looping over embeddings with masking, instead of direct fetching.
- **Bounty for Embeddings Addressed to Specific User**: There is a bounty for improving embeddings, but it is currently exclusive to a user named **Qazalin**.
   - Thus, new contributors are encouraged to explore other issues in the codebase.
- **Exploring Embedding Code in Tinygrad**: Discussion detailed the functioning of the `Embedding` feature within tinygrad, including an example kernel code clarifying its execution.
   - A member initially misunderstood the purpose of summing across the input embeddings matrix and later acknowledged the correct implementation.
- **Cumsum Complexity Discussion**: A user questioned the impossibility of making `cumsum` O(n) in the context of tinygrad ([Issue #2433](https://github.com/tinygrad/tinygrad/issues/2433)).
   - George Hotz encouraged experimentation to explore potential optimizations.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/tinygrad/tinygrad/issues/1612">Embedding/argmax are O(n^2) · Issue #1612 · tinygrad/tinygrad</a>: This is making GPT-2 slow</li><li><a href="https://github.com/tinygrad/tinygrad/issues/2433">Embeddings are slow and shouldn&#39;t be · Issue #2433 · tinygrad/tinygrad</a>: While it&#39;s not possible to make cumsum O(n), it should be possible to make Embeddings O(n). It&#39;s beyond ARANGE, but points the way to fast selection for dataloader.</li><li><a href="https://github.com/tinygrad/tinygrad/blob/c6a8395f1b726c00c47a65ba0252e7d142b7738a/tinygrad/nn/__init__.py#L319">tinygrad/tinygrad/nn/__init__.py at c6a8395f1b726c00c47a65ba0252e7d142b7738a · tinygrad/tinygrad</a>: You like pytorch? You like micrograd? You love tinygrad! ❤️  - tinygrad/tinygrad
</li>
</ul>

</div>
  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1268453692647149601)** (4 messages): 

> - `ChatGPT Advanced Voice Mode`
> - `Black Forest Labs Launch`
> - `FLUX.1 Model` 


- **ChatGPT Multilingual Voice Stunt**: A user shared [ChatGPT Advanced Voice Mode](https://x.com/CrisGiardina/status/1818799060385489248?t=oe5JjISZYPP6mFqmmJUthg&s=19) performing a linguistic stunt by reciting a couplet in **Urdu** and telling stories in multiple languages including **Hebrew, Norwegian, Moroccan Darija, Amharic, Hungarian, Georgian**, and **Klingon**.
- **Black Forest Labs Lights Up**: A user expressed excitement about the launch of **Black Forest Labs** aimed at advancing state-of-the-art generative deep learning models for images and video, underlined by their new release, **FLUX.1**.
   - [Black Forest Labs](https://x.com/robrombach/status/1819012132064669739) is committed to pushing the boundaries of creativity, efficiency, and diversity in media with their new mission and model.
- **FLUX.1 Debuts on Hugging Face**: A user shared a link to the [FLUX.1 model](https://huggingface.co/spaces/black-forest-labs/FLUX.1-schnell), highlighting its impressive capabilities.
   - *Refreshing* and *super good* were comments made about the performance of FLUX.1.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/robrombach/status/1819012132064669739">Tweet from Robin Rombach (@robrombach)</a>: 🔥 I am so damn excited to announce the launch of Black Forest Labs. We set ourselves on a mission to advance state-of-the-art, high-quality generative deep learning models for images and video, and m...</li><li><a href="https://huggingface.co/spaces/black-forest-labs/FLUX.1-schnell">FLUX.1 [Schnell] - a Hugging Face Space by black-forest-labs</a>: no description found</li><li><a href="https://x.com/CrisGiardina/status/1818799060385489248?t=oe5JjISZYPP6mFqmmJUthg&s=19">Tweet from Cristiano Giardina (@CrisGiardina)</a>: ChatGPT Advanced Voice Mode recites a couplet in Urdu → tells a story in Hebrew → Norwegian → Moroccan Darija → Amharic → Hungarian → Georgian → finally attempts some Klingon
</li>
</ul>

</div>
  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1268364167081889933)** (6 messages): 

> - `Normalization and activation functions`
> - `Regularization techniques`
> - `Common code errors` 


- **Experimenting with activation functions on complex-valued activations**: A user mentioned experimenting with different **normalization and activation functions** on complex-valued activations and noted it was 'kinda fun!'
- **Data augmentation and regularization techniques discussed**: [A link](https://medium.com/ymedialabs-innovation/data-augmentation-techniques-in-cnn-using-tensorflow-371ae43d5be9) on data augmentation was shared, but a member noted that **techniques like data augmentation, dropout, and weight decay** merely delay overfitting and do not significantly reduce final validation error.
   - 'They delay overfitting but don't generally reduce the final val error much.'
- **Code typo discovered after 50+ experiments**: A user found a **stupid typo** in their code which had been obstructing the architecture's performance in the past 50+ experiments.



**Link mentioned**: <a href="https://medium.com/ymedialabs-innovation/data-augmentation-techniques-in-cnn-using-tensorflow-371ae43d5be9">Data Augmentation Techniques in CNN using Tensorflow</a>: Recently, I have started learning about Artificial Intelligence as it is creating a lot of buzz in industry. Within these diverse fields of…

  

---



### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1268332069453430889)** (5 messages): 

> - `model performance`
> - `generate recipe debugging`
> - `llama3 model`
> - `top_p settings` 


- **Online model outperforms user's own model**: A member noted that testing **0.8 online** yielded much better results than their own model.
- **Top_p=50 considered acceptable**: The member reported that **top_p=50** seemed perfectly fine for their needs.
- **Generate recipe meant for debugging, not optimal quality**: Another member clarified that the **generate recipe** is intended for debugging, not to showcase optimal performance, but aims for a high-quality, accurate sampling of the trained model.
   - Evaluation tests using the same generation utils showed similar numbers to reported benchmarks, and any quality issues should be submitted as an issue.
- **Rechecking performance of original llama3 model**: A member planned to create a new server instance, download the **llama3-8B-instruct model** again, and test it on standard settings to check if the generation quality still differs from the online benchmarks.


  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1268283561383825420)** (4 messages): 

> - `PR Merge`
> - `FSDP2`
> - `Quantization APIs`
> - `QAT and FSDP2 Compatibility` 


- **Merged fine-tuning datasets discussed in PR #1234**: A member mentioned that they will put up a separate PR after [PR #1234](https://github.com/pytorch/torchtune/pull/1234) gets reviewed and landed since it depends on some elements from this PR.
- **FSDP2 supports both quantization and NF4 tensor**: A member noted that **FSDP2** should support both quantization for **NF4 tensor** and possibly QAT, although they have not tried many other quantization APIs.
   - They also mentioned that for their current QAT recipe, compile won't work with **FSDP2**.



**Link mentioned**: <a href="https://github.com/pytorch/torchtune/pull/1234">[1/n] Merged fine-tuning dataset: grammar + samsum by RdoubleA · Pull Request #1234 · pytorch/torchtune</a>: Context What is the purpose of this PR? Is it to   add a new feature  fix a bug  update tests and/or documentation  other (please add here)  As discussed in the RFC in #1186, we will merged instruc...

  

---



### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1268599285751218269)** (2 messages): 

> - `Data Phoenix Webinar`
> - `ELT Workshop with dlt` 


- **Data Phoenix Hosts Webinar on Enhancing Recommendation Systems**: The **Data Phoenix** team is hosting a free webinar on August 8 at 10 a.m. PDT, titled 'Enhancing Recommendation Systems with LLMs and Generative AI,' featuring [Andrei Lopatenko](https://www.linkedin.com/in/lopatenko/), VP AI & Engineering.
   - The talk will discuss how **LLMs** and **Generative AI** can revolutionize recommendation systems and personalization engines. [Register here](https://lu.ma/6i6dtbhf).
- **4-hour Comprehensive ELT Workshop with dlt**: A 4-hour workshop on **robust and easy ELT** with dlt is being held to teach data enthusiasts and engineers how to build ELT pipelines, with a [registration link here](https://dlthub.com/events).
   - Completion includes a 'dltHub ELT Engineer' certification. The first part covers **dlt fundamentals** and takes place online on 15.08.2024 at 16:00 GMT+2.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://dlthub.com/events">dltHub events</a>: Come meet the dltHub team at these events.</li><li><a href="https://lu.ma/6i6dtbhf?utm_source=DiscordEvent5">Enhancing Recommendation Systems with LLMs and Generative AI · Luma</a>: The Data Phoenix team invites you to our upcoming webinar, which will take place on August 8 at 10 a.m. PDT. Topic: Enhancing Recommendation Systems with LLMs…
</li>
</ul>

</div>
  

---


### **MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/1268705341185724559)** (5 messages): 

> - `Computer Vision`
> - `Conferences on Machine Learning`
> - `Gaussian Processes`
> - `Isolation Forest`
> - `GenAI ROI` 


- **Machine Learning Conferences Emphasize NLP & GenAI**: A member shared their experience attending two machine learning conferences in the past year where their presentations on **Gaussian Processes** and **Isolation Forest** models were overshadowed by the focus on **NLP** and **genAI**.
   - They noted that many attendees had no idea about their work, highlighting the prevalent interest in **NLP and genAI** technologies.
- **Skepticism Surrounds GenAI ROI Expectations**: Discussion revolved around skepticism that the **ROI from genAI** might not meet high expectations.
   - One member commented that a **return on investment** first requires a return of investment, emphasizing the need for realistic expectations.


  

---



### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1268437189046571041)** (3 messages): 

> - `LangSmith credit access`
> - `Payment method issues` 


- **LangSmith Credits Inaccessible Without Payment Method**: **Digitalbeacon** raised a concern about being unable to access credits in LangSmith despite adding a payment method. His organization ID is **93216a1e-a4cb-4b39-8790-3ed9f7b7fa95** and he used a different email ID in the form than in the course.
   - **Danbecker** advised contacting support for any credit-related issues.
- **Payment Method Issues for LangSmith Credits**: **Digitalbeacon** mentioned adding a payment method but still seeing zero credits in LangSmith. They asked for assistance because they had filled out the form on time.


  

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
