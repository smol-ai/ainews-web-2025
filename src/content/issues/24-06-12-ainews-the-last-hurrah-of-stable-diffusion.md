---
id: baa5ffc2-4e11-4be7-9ba4-182e18db701a
title: The Last Hurrah of Stable Diffusion?
date: '2024-06-12T22:08:29.963442Z'
original_slug: ainews-the-last-hurrah-of-stable-diffusion
description: >-
  Stable Diffusion 3 Medium launched with a detailed research paper revealing
  its MultiModal Diffusion Transformer architecture and a range of models up to
  *8B parameters*, marking the final legacy of Emad Mostaque at **Stability
  AI**. Meanwhile, **Llama 3 8B Instruct** demonstrated strong evaluation
  capabilities with *0.8 correlation to GPT-4*, and **Qwen 2 Instruct**
  surpassed Llama 3 on MMLU benchmarks, highlighting ongoing advances in large
  language model performance.
tags:
  - untagged
companies: []
models: []
topics:
  - untagged
---


<!-- buttondown-editor-mode: plaintext -->**MultiModal Diffusion Transformers are All You Need.**

> AI News for 6/11/2024-6/12/2024. We checked 7 subreddits, [**384** Twitters](https://twitter.com/i/lists/1585430245762441216) and **30** Discords (**413** channels, and **3555** messages) for you. Estimated reading time saved (at 200wpm): **388 minutes**. Track [AINews on Twitter](https://x.com/smol_ai).

SD3 Medium was launched today with an unusually (for Stability) flashy video:

<figure><img src="https://assets.buttondown.email/images/b0b3bf6b-1a67-4f19-aea8-61eb562a0437.png?w=960&amp;fit=max" draggable="false" contenteditable="false"><figcaption></figcaption></figure>

The [SD3 research paper](https://stability.ai/news/stable-diffusion-3-research-paper) is noteworthy for it's detail on the MMDiT architecture and usage of the T5 text encoder for text rendering in images, but also for mentioning its range of models from 450M to 8B params, making the 2B parameter SD3 Medium not the most powerful SD3 version available.

If you've been diligently reading the Discord Summaries for the Stability AI discord, you'll have known that the community has been fretting about the open weights release of SD3, [first announced 3 months ago](https://news.ycombinator.com/item?id=39466630), released [as Paper](https://news.ycombinator.com/item?id=39599958) and [as API](https://news.ycombinator.com/item?id=40065114), on an almost daily basis, particularly since the exit of Emad Mostaque and Robin Rombach and many of the senior researchers involved in the original Stable Diffusion. Adding up points of related posts, it is easy to see the gradual stalling of interest from SD1 to SD2 to SD3 as the project became increasingly less default-open:

<figure><img src="https://assets.buttondown.email/images/6d79af38-9313-4c6f-892d-f4df2853ca0a.png?w=960&amp;fit=max" alt="image.png" draggable="false" contenteditable="false"><figcaption></figcaption></figure>

This was the last legacy of Emad's tenure at Stability - [the new management](https://stability.ai/news/stabilityai-announcement?ref=futuretools.io) must now figure out their path ahead on their own.

---

{% if medium == 'web' %}

**Table of Contents**

[TOC]

{% else %}

The **Table of Contents** and **Channel Summaries** have been moved to the web version of this email: [{{ email.subject }}]({{ email_url }})! ([Share on Twitter](https://x.com/smol_ai).)

{% endif %}

---

# AI Twitter Recap

> all recaps done by Claude 3 Opus, best of 4 runs. We are working on clustering and flow engineering with Haiku.

**AI Models and Architectures**

- **Llama 3 and Instruction Finetuning**: [@rasbt](https://twitter.com/rasbt/status/1800660930948170117) found Llama 3 8B Instruct to be a good evaluator model that runs on a MacBook Air, achieving **0.8 correlation with GPT-4 scores**. A standalone notebook is provided.
- **Qwen2 and MMLU Performance**: [@percyliang](https://twitter.com/percyliang/status/1800774871187968404) reports **Qwen 2 Instruct surpassing Llama 3 on MMLU** in the latest HELM leaderboards v1.4.0.
- **Mixture of Agents (MoA) Framework**: [@togethercompute](https://twitter.com/togethercompute/status/1800536106729157054) introduces MoA, which **leverages multiple LLMs to refine responses**. It achieves **65.1% on AlpacaEval 2.0, outperforming GPT-4o**.
- **Spectrum for Extending Context Window**: [@cognitivecompai](https://twitter.com/cognitivecompai/status/1800710613594955880) presents Spectrum, a technique to **identify important layers for finetuning**. It can be combined with @Tim_Dettmers' QLoRA for **faster training with less VRAM**.
- **Grokking in Transformers**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1800544938704179567) discusses a paper showing transformers can learn robust reasoning through grokking - **extended training beyond overfitting**. Grokking involves a transition from a "memorizing" to a "generalizing" circuit.

**Benchmarks and Datasets**

- **ARC Prize Challenge**: [@fchollet](https://twitter.com/fchollet/status/1800577019979411560) and @mikeknoop launch a **$1M competition to create an AI that can adapt to novelty and solve reasoning problems**, aiming to measure progress towards AGI.
- **LiveBench**: [@micahgoldblum](https://twitter.com/micahgoldblum/status/1800894380511002724) announces LiveBench, a **general-purpose live LLM benchmark that releases new questions to avoid dataset contamination** and can be judged objectively.
- **Character Codex Dataset**: [@Teknium1](https://twitter.com/Teknium1/status/1800590745885413726) releases Character Codex, an **open dataset with data on 15,939 characters** from various sources for use in RAG, synthetic data generation, and roleplaying analysis.

**Tools and Frameworks**

- **MLX 0.2**: [@stablequan](https://twitter.com/stablequan/status/1800576080077881677) releases MLX 0.2, providing a **new LLM experience on Apple Silicon Macs** with a revamped UI/UX, fully-featured chat, and faster RAG.
- **Unsloth**: [@danielhanchen](https://twitter.com/danielhanchen/status/1800528838226804907) announces Unsloth is now in Hugging Face AutoTrain, allowing **2x faster QLoRA finetuning of LLMs like Llama-3, Mistral, and Qwen2 with less memory**.
- **LangChain Integrations**: [@LangChainAI](https://twitter.com/LangChainAI/status/1800615245830062364) adds **Elasticsearch capabilities for flexible retrieval and vector databases**. They also ship [@GroqInc](https://twitter.com/LangChainAI/status/1800952057517752625) support in LangSmith Playground.

**Applications and Use Cases**

- **Brightwave AI Research Assistant**: [@vagabondjack](https://twitter.com/vagabondjack/status/1800527732641521943) announces a **$6M seed round for Brightwave, an AI research assistant generating financial analysis**, with customers managing over $120B in assets.
- **Suno Audio Input**: [@suno_ai_](https://twitter.com/suno_ai_/status/1800932487633207599) releases an Audio Input feature, allowing users to **make songs from any sound by uploading or recording audio clips**.
- **Synthesia 2.0 Event**: [@synthesiaIO](https://twitter.com/synthesiaIO/status/1800874931141656882) teases **new features, workflows, use cases, and avatar capabilities** for their AI video generator, with an event on June 24.

**Discussions and Opinions**

- **Prompt Engineering vs. Finetuning**: [@corbtt](https://twitter.com/corbtt/status/1800597703560417700) argues that **fine-tuned adapters will outperform prompting for better performance, control, and cheaper inference** in the coming years.
- **Unintended Consequences of RLHF**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1800502590678982922) shares a paper exploring how **RLHF alignment reduces model creativity and output diversity** due to blocked token trajectories and mode collapse.
- **Hallucinations in LLMs**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1800917180604956855) cites a paper showing that **statistically calibrated language models must hallucinate at a certain rate**.

---

# AI Reddit Recap

> Across r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity. Comment crawling works now but has lots to improve!

AI Progress and Timelines

- **GPT-1 anniversary**: In /r/singularity, it was noted that [**GPT-1 was released exactly 6 years ago**](https://www.reddit.com/r/singularity/comments/13uj1ej/gpt1_was_released_exactly_6_years_ago/), with hope expressed that by June 2030 we will be in the AGI or ASI era.

AI Companies and Products

- **Tesla Optimus deployment**: /r/singularity shared that [**Tesla has deployed two Optimus bots performing tasks autonomously**](https://www.reddit.com/r/singularity/comments/13uqgp3/tesla_has_deployed_two_optimus_bots_performing/) in their factory.
- **Musk drops OpenAI lawsuit**: /r/singularity and /r/OpenAI reported that [**Elon Musk dropped his lawsuit against OpenAI and Sam Altman**](https://www.reddit.com/r/singularity/comments/13uj2yl/elon_musk_drops_lawsuit_against_openai_and_sam/), with emails showing Musk previously agreed with the for-profit direction.
- **Apple partners with OpenAI**: According to /r/singularity, [**Apple announced Apple Intelligence, powered by OpenAI and funded by Microsoft**](https://www.reddit.com/r/singularity/comments/13uj2yl/apple_announces_apple_intelligence_powered_by/), with on-device AI using OpenAI if local solutions fail.
- **OpenAI uses Oracle Cloud**: /r/OpenAI noted [**OpenAI selected Oracle Cloud Infrastructure to extend the Microsoft Azure AI platform**](https://www.reddit.com/r/OpenAI/comments/13uqgp3/openai_selects_oracle_cloud_infrastructure_to/).
- **Ex-Google CEO criticizes open models**: /r/singularity discussed the [**ex-Google CEO condemning open source models and arguing for government curbs on public releases**](https://www.reddit.com/r/singularity/comments/13uqgp3/exgoogle_ceo_condemns_open_source_models_argues/), with counterarguments that this protects big tech monopolies.

AI Capabilities

- **Restaurant robots advance**: /r/singularity shared that [**restaurant robots can now cook, serve and bus meals**](https://www.reddit.com/r/singularity/comments/13uj2yl/restaurant_robots_can_now_cook_serve_and_bus_meals/).
- **Robot dog sharpshooters**: According to /r/singularity, a study found [**machine gun-wielding robot dogs are better sharpshooters than humans**](https://www.reddit.com/r/singularity/comments/13uj2yl/machine_gunwielding_robot_dogs_are_better/).
- **AI video generation**: /r/singularity noted [**Katalist AI Video allows turning a sentence into a storyboard and consistent video story in 1 minute**](https://www.reddit.com/r/singularity/comments/13uj1ej/katalist_ai_video_allows_turning_a_sentence_into/), with potential for future movie production.
- **AI job interviews**: According to /r/singularity, [**AI cartoons may interview you for your next job**](https://www.reddit.com/r/singularity/comments/13uqgp3/ai_cartoons_may_interview_you_for_your_next_job/).
- **Deepfake nudes**: /r/singularity warned that [**teens are spreading deepfake nudes of one another, causing serious issues**](https://www.reddit.com/r/singularity/comments/13uqgp3/teens_are_spreading_deepfake_nudes_of_one/).

AI Research

- **Autoregressive image models**: /r/MachineLearning and /r/singularity discussed new LlamaGen research showing [**autoregressive models like Llama beating diffusion for scalable image generation**](https://www.reddit.com/r/MachineLearning/comments/13uqgp3/autoregressive_models_like_llama_beat_diffusion/), with questions raised about fairly citing prior autoregressive work.
- **Eliminating matrix multiplication**: /r/singularity shared a [**revolutionary approach that eliminates matrix multiplication in language models without losing performance**](https://www.reddit.com/r/singularity/comments/13uj2yl/revolutionary_approach_eliminates_matrix/), using a custom FPGA solution to process billion-parameter models at 13W.
- **Overtraining transformers**: According to /r/singularity, research found [**overtraining transformers beyond the overfitting point leads to unexpected reasoning improvements**](https://www.reddit.com/r/singularity/comments/13uj1ej/overtraining_transformers_beyond_overfitting/).
- **MaPO alignment technique**: /r/MachineLearning noted [**MaPO is a sample-efficient reference-free alignment technique for diffusion models that improves on prior work**](https://www.reddit.com/r/MachineLearning/comments/13uqgp3/mapo_is_a_sampleefficient_referencefree/).
- **Megalodon for LLM pretraining**: /r/MachineLearning shared that [**Megalodon allows efficient LLM pretraining and inference with unlimited context length**](https://www.reddit.com/r/MachineLearning/comments/13uqgp3/megalodon_allows_efficient_llm_pretraining_and/).

Stable Diffusion

- **Stable Diffusion 3.0 release**: /r/StableDiffusion is highly anticipating the [**Stable Diffusion 3.0 release set for June 12th**](https://www.reddit.com/r/StableDiffusion/comments/13uj2yl/stable_diffusion_30_set_to_release_on_june_12th/), with much excitement and memes from the community.
- **SD3 model debate**: /r/StableDiffusion is [**debating whether the SD3 2B or 8B model will be better**](https://www.reddit.com/r/StableDiffusion/comments/13uqgp3/debate_on_whether_sd3_2b_or_8b_model_will_be/), with the 8B model still training but expected to surpass 2B when finished.
- **Open source training tools**: According to /r/StableDiffusion, [**new open source tools like Kohya DeepShrink enable high-res SD training**](https://www.reddit.com/r/StableDiffusion/comments/13uqgp3/new_open_source_tools_for_training_like_kohya/).
- **SDXL vs SD3 comparisons**: /r/StableDiffusion is [**comparing SDXL vs SD3 car images and other subjects**](https://www.reddit.com/r/StableDiffusion/comments/13uj2yl/comparisons_of_sdxl_vs_sd3_car_images_and_other/), with SD3 showing improved proportions, reflections, and shadows.

Humor/Memes

- /r/singularity shared a [**meme showing Woody and Buzz as AI models, implying we are the "toys"**](https://www.reddit.com/r/singularity/comments/13uj2yl/meme_showing_woody_and_buzz_as_ai_models/).
- /r/singularity posted a [**meme of the 10 commandments with "Thou shalt not take the name of the ASI in vain"**](https://www.reddit.com/r/singularity/comments/13uqgp3/meme_of_10_commandments_with_thou_shalt_not_take/).
- /r/singularity shared a [**meme of Clippy asking if it would fold us into oblivion**](https://www.reddit.com/r/singularity/comments/13uqgp3/meme_of_clippy_asking_if_it_would_fold_us_into/).

---

# AI Discord Recap

> A summary of Summaries of Summaries

**1\. Stable Diffusion 3 Release and Discussions**

- **Stability.ai** released the open weights for [**Stable Diffusion 3 Medium**](https://huggingface.co/stabilityai/stable-diffusion-3-medium), their latest text-to-image AI model promising exceptional detail, color, advanced prompt understanding using three text encoders, and superior typography rendering.
- Users reported issues with **human anatomy accuracy**, mixed reactions on performance compared to older versions like **SD 1.5** and **SDXL**, and concerns over the **restrictive licensing terms** limiting commercial use.
- Extensive discussions around **finetuning challenges**, particularly for popular use cases, installation issues with existing frameworks like **ComfyUI** and **diffusers**, and the model's efficient GPU utilization.

**2\. Large Language Model (LLM) Advancements and Benchmarks**

- Google unveiled [**RecurrentGemma 9B**](https://huggingface.co/collections/google/recurrentgemma-release-66152cbdd2d6619cb1665b7a), capable of handling long sequences quickly while maintaining quality on par with Gemma, featuring base and instruct-tuned versions.
- The [**$1,000,000+ ARC Prize**](https://arcprize.org/) was announced to develop a solution for the **ARC-AGI benchmark**, measuring general intelligence through skill acquisition efficiency, sparking discussions on its significance.
- [**Scalable MatMul-free Language Modeling**](https://arxiv.org/abs/2406.02528) suggests eliminating matrix multiplication from LLMs while retaining strong performance, reducing memory usage during training.

**3\. Collaborative LLM Development and Deployment**

- Members sought guidance on **installing and integrating models** like **mistral-instruct**, **GPT-4o**, and **Codestral** using tools such as **llm**, **Datasette**, and **LangServe**, with resources shared on Hugging Face and GitHub.
- Discussions around **LLM evaluation methodologies**, the importance of **standardized evaluations**, and the potential of **LLM-as-judge systems**, with insights from Hailey Schoenfeld.
- Queries on **LLM finetuning** using Modal, Predibase, and Axolotl, covering topics like dataset preparation, resource constraints, and merging results from techniques like LoRA and qLoRA.

**4\. Hardware Optimization and Resource Management**

- Affordable **3090 GPUs** on platforms like **Kijiji Canada** sparked interest in building GPU rigs, with considerations for power draw, heat management, and optimizing performance metrics like flops/watts.
- Developers achieved **100% determinism** in their code but faced issues like loss values hitting -inf, indicating potential bugs requiring resolution.
- Efforts to improve computational efficiency resulted in a **custom matmul implementation achieving 70% speed of cuBLAS**, with ongoing work to remove dependencies on cuBLASLt and tackle stable FP8 implementation.

[Source](https://discord.com)

## Claude 3 Opus (>220B?)

Here is a high information density, technical summary of the top 3-4 major themes across the provided Discord content, with important key terms, facts, and URLs bolded and linked to sources where relevant:

- **Unsloth AI's QWEN Models Spark VRAM and Fine-Tuning Discussions**: Unsloth AI users compare **VRAM requirements for Qwen2 models** like the 72b variant ([estimated 48GB, similar to Llama 3 70b](https://discord.com/channels/1179035537009545276/1179035537529643040/1250170775781249164)) and share tips for [converting PyTorch bins to safetensors](https://huggingface.co/spaces/safetensors/convert). Fine-tuning challenges with **Qwen2-1.5b** using methods like **Qlora** and **Galore** are discussed, with rank 64-128 recommended to avoid failures. Performance drops after 16-bit merging are attributed to tokenizer/GGUF issues.
- **LLM Fine-Tuning Course Learnings Applied and Debated**: Participants reflect on Nehil's [blog post applying course concepts to improve budget categorization prompts](https://brain.nehiljain.com/posts/how-i-improved-my-prompting-for-budget-categorization/). Technical discussions cover adding pooling layers to Huggingface models, the impracticality of broad-scope chatbots vs. scoped interfaces due to security/privacy concerns, and using **Modal** to fine-tune LLMs for Kaggle competitions. Interest in **Datasette**, **Anthropic Claude**, and resources for transitioning from academia to industry is evident.
- **GPU Market Dynamics and Creative Applications**: Affordable **3090 GPUs** on **Kijiji Canada** (~$450 USD) inspire GPU rig building plans, with power draw and heat repurposing considerations discussed. Custom CUDA development efforts, like achieving 100% determinism and implementing efficient matrix multiplication, are shared alongside the [Scalable MatMul-free Language Modeling paper](https://arxiv.org/abs/2406.02528) and [implementation](https://github.com/ridgerchu/matmulfreellm). The **CUDA Performance Checklist** lecture is re-released with [video](https://www.youtube.com/watch?v=SGhfUhlowB4), [code](https://github.com/cuda-mode/lectures/tree/main/lecture8), and [slides](https://docs.google.com/presentation/d/1cvVpf3ChFFiY4Kf25S4e4sPY6Y5uRUO-X-A4nJ7IhFE/edit).

Let me know if you would like me to elaborate on any part of this summary or if you have additional questions!

## GPT4T (gpt-4-turbo-2024-04-09)

## Summaries:

**1\. Exploring LLM-Eval Harness and ARCSynthetic Challenges in EleutherAI:**

- Members of EleutherAI explored LLM-Eval harness and shared insights on traditional vs. score-based evals, prompting a collaborative environment aiming to improve sensitivity and specificity regarding specific use cases.
- A new paper titled [Scaling Laws for Diffusion Models](https://arxiv.org/abs/2406.06752) was introduced, discussing the efficiency of training and utilizing diffusion methods as compared to autoregressive models.

**2\. Cross-Language Communication in OpenInterpreter:**

- OpenInterpreter participants were enlightened with discussions on managing cross-language and cross-terminal communications effectively, utilizing NLP models for seamless interactions across diverse computing environments.
- Technical discussions highlighted the advancements in language model implementations that enhance user interface engagement through improved command interpretations.

**3\. Cohere's Practical Model Innovations Spotlighted:**

- Cohere's platform updates were highlighted with discussions focusing on practical applications of recently introduced AI models that enhance user experience and broaden application scopes.
- Debates on model integration challenges provided insights into best practices and forthcoming innovations in AI model deployments.

**4\. Modular (Mojo) Embraces TPU Considerations and Compiler Updates:**

- Modular's discussion on integrating TPUs and updating their compiler to **version** `2024.6.1205` showcased ongoing efforts to enhance computational performance and flexibility.
- Community feedback praised the improvements, with further anticipation for upcoming features that promise to advance scalable AI deployment scenarios using Modular's tools.

**1\. Model Performance Optimization and Benchmarking**

- **Quantization** techniques like **AQLM** and **QuaRot** aim to run large language models (**LLMs**) on individual GPUs while maintaining performance. Example: [AQLM project](https://github.com/Vahe1994/AQLM) with **Llama-3-70b** running on RTX3090.
- Efforts to **boost transformer efficiency** through methods like **Dynamic Memory Compression (DMC)**, potentially improving throughput by up to 370% on **H100 GPUs**. Example: [DMC paper](https://arxiv.org/abs/2403.09636) by @p_nawrot.

**2\. Fine-tuning Challenges and Prompt Engineering Strategies**

- Difficulties in **retaining fine-tuned data** when converting **Llama3** models to GGUF format, with a [confirmed bug](https://github.com/ggerganov/llama.cpp/issues/7062) discussed.
- Importance of **prompt design** and usage of correct templates, including end-of-text tokens, for influencing model performance during fine-tuning and evaluation. Example: [Axolotl prompters.py](https://github.com/OpenAccess-AI-Collective/axolotl/blob/3367fca73253c85e386ef69af3068d42cea09e4f/src/axolotl/prompters.py#L47).

**3\. Open-Source AI Developments and Collaborations**

- Launch of **StoryDiffusion**, an open-source alternative to Sora with MIT license, though weights not released yet. Example: [GitHub repo](https://github.com/HVision-NKU/StoryDiffusion/tree/main?tab=readme-ov-file).
- Release of **OpenDevin**, an open-source autonomous AI engineer based on Devin by Cognition, with [webinar](https://lu.ma/fp0xr460) and growing interest on GitHub.

**4\. Multimodal AI and Generative Modeling Innovations**

- [**Idefics2 8B Chatty**](https://twitter.com/sanhestpasmoi/status/1787503160757485609) focuses on elevated chat interactions, while [**CodeGemma 1.1 7B**](https://twitter.com/reach_vb/status/1786469104678760677) refines coding abilities.
- The [**Phi 3**](https://www.reddit.com/r/LocalLLaMA/comments/1cn2zwn/phi3_webgpu_a_private_and_powerful_ai_chatbot/) model brings powerful AI chatbots to browsers via WebGPU.

**5\. Misc**

- **Stable Artisan Brings AI Media Creation to Discord**: Stability AI launched **Stable Artisan**, a Discord bot integrating models like **Stable Diffusion 3**, **Stable Video Diffusion**, and **Stable Image Core** for [media generation and editing directly within Discord](https://bit.ly/4aiVy6C). The bot sparked discussions about **SD3's open-source status** and the introduction of **Artisan as a paid API service**.
- **Unsloth AI Community Abuzz with New Models and Training Tips**: IBM's [Granite-8B-Code-Instruct](https://huggingface.co/ibm-granite/granite-8b-code-instruct) and RefuelAI's [RefuelLLM-2](https://huggingface.co/refuelai/Llama-3-Refueled) were introduced, sparking architecture discussions. Users shared challenges with **Windows compatibility** and skepticism over certain **performance benchmarks**, while also exchanging model training and fine-tuning tips.

## GPT4O (gpt-4o-2024-05-13)

## Themes:

1.  LLM Advancements and Model Performance
2.  Multimodal AI and Generative Modeling Innovations
3.  Open-Source Tools and Community Contributions
4.  Technical Troubleshooting and Implementation Challenges
5.  Ethics and Industry Dynamics in AI

## Summaries:

1.  **LLM Advancements and Model Performance**:
  
  - [**RecurrentGemma 9B**](https://huggingface.co/collections/google/recurrentgemma-release-66152cbdd2d6619cb1665b7a) by Google is praised for its super-fast performance on long sequences with both base and instruct-tuned versions, compared to **Gemma**.
  - Members highlighted **LlamaGen**’s superiority in autoregressive image generation, rivaling diffusion models and supported by detailed [documentation and tutorials](https://github.com/FoundationVision/LlamaGen).
2.  **Multimodal AI and Generative Modeling Innovations**:
  
  - **Anthropic's Transformer Lens** helped debug model attention, while **Stable Diffusion 3 Medium** sparked discussions about its [image generation capabilities](https://huggingface.co/stabilityai/stable-diffusion-3-medium), yet faced criticisms regarding human anatomy accuracy.
  - [**Idefics2 8B Chatty**](https://twitter.com/sanhestpasmoi/status/1787503160757485609) and [**CodeGemma 1.1 7B**](https://twitter.com/reach_vb/status/1786469104678760677) improved chat and coding interactions. **Pixart Sigma** combined with **SDXL + PAG** aimed for **DALLE-3**\-level outputs.
3.  **Open-Source Tools and Community Contributions**:
  
  - [**Axolotl**](https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/) supported diverse formats for instruction tuning and pre-training LLMs. IBM’s [**Granite-8B-Code-Instruct**](https://huggingface.co/ibm-granite/granite-8b-code-instruct) excelled in code tasks.
  - [**DeepEval**](https://docs.confident-ai.com/docs/metrics-introduction#using-a-custom-llm) allows easy integration of custom LLM evaluation metrics, enhancing capabilities in tasks like summarization and faithfulness.
4.  **Technical Troubleshooting and Implementation Challenges**:
  
  - **Qwen2-1.5b Fine-tuning** faced issues with low-rank configurations, resolved by increasing ranks (64 or 128). Members shared setup struggles for **Stable Diffusion 3** in frameworks like **ComfyUI**, **diffusers**, and GPU-related discussions for **RTX 2070** and **P40 graphics cards**.
  - [**High Precision Requirements**](https://arxiv.org/abs/2210.02671) and stability challenges in models like **PPO for RL** led to shared insights on optimization techniques, tokenizer revamps and community troubleshooting on **PyTorch distributed log** errors.
5.  **Ethics and Industry Dynamics in AI**:
  
  - **Ethical concerns about Nightshade technology** and analyses of Perplexity AI’s alleged plagiarism ([Forbes article](https://www.forbes.com/sites/randalllane/2024/06/11/why-perplexitys-cynical-theft-represents-everything-that-could-go-wrong-with-ai/)) underscored the need for accountability.
  - **Microsoft’s acquisition of a 49% OpenAI stake via leveraging discussions with Tesla** ([Tweet](https://fxtwitter.com/nicoleperlroth/status/1800946061613416659?s=46)) highlighted strategic moves within tech giants, while **ARC Prize’s $1,000,000+ reward** spurred extensive discussions on industry benchmarks.

---

# PART 1: High level Discord summaries

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

**Stable Diffusion 3 Unleashes Potentials and Problems**: The newly released **Stable Diffusion 3 Medium** boasts better quality and advanced prompt comprehension but struggles with human anatomy accuracy, according to user reports. Discussions reveal mixed reactions to performance, with some finding it underwhelming and expressive concerns over technical hurdles in installation and finetuning.

**Licence to Confuse**: The **licensing terms of SD3** sparked intense debate in the community over its restrictions on commercial use, with many finding them too limiting for practical application.

**Photorealism Promise Meets Skepticism**: Users acknowledge the efforts to enhance realism in faces and hands with SD3, but outcome consistency remains a contentious point when compared to older versions such as **SD 1.5** and **SDXL**.

**Resource Effectiveness Favorable, But Customization Could Be Costly**: Engineers appreciate the efficient GPU utilization of SD3 and the customization options, although concerns about the financial and technical barriers to finetuning exist, especially for niche content.

**Installation Integration Anxiety**: A variety of issues related to integrating SD3 into popular frameworks like **ComfyUI** and **diffusers** have been flagged, leading to collaborative troubleshooting efforts within the community.

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **VRAM Demands for Large Models - No Jokes Allowed**: The **VRAM requirements for Qwen2 72b** are akin to Llama 3 70b at roughly **48GB**, with humorous suggestions of needing a supercomputer brushed aside by reality checks.
- **Transfiguring Pytorch to Safetensors**: Attempting to convert Pytorch bins to safetensors using [this tool](https://huggingface.co/spaces/safetensors/convert), users hit a snag on Google Colaboratory due to RAM limitations; a pivot to Kaggle for ample resources was advised.
- **Boosting Fine-tuning Efficiency**: To mitigate fine-tuning woes with low-rank configurations in **Qwen2-1.5b**, utilizing methods like Qlora and Galore, users recommend a minimum rank of 64 or 128.
- **Troubleshooting 16-bit Merging Mishaps**: A performance decline after merging a model to 16-bit calls attention to possible tokenizer and GGUF anomalies; members anticipate resolution in forthcoming patches.
- **Community Effort in Documentation**: Volunteers emerge to enhance documentation, pinpointing data preparation and chat notebook usage as areas ripe for improvement, and crafting video tutorials is suggested for better user engagement.

---

## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **Zoom Stumbles, but Blog Shines**: While technical issues delayed Zoom session recordings, Nehil's [blog post on budget categorization](https://brain.nehiljain.com/posts/how-i-improved-my-prompting-for-budget-categorization/) received praise for its practical application of course concepts in prompting.
- **Pooling Layer Puzzles and Scope-creep Concerns**: Discussions about the technicalities of adding pooling layers to models using libraries like Huggingface showcased the community's collaborative approach, while debates raged over the practicality of broad-scope chatbots versus scoped interfaces, citing concerns around data privacy and security.
- **Quantum of Quantization**: Excitement was palpable over Llama-recipes' "pure bf16" mode and ongoing optimizer quantization research, indicating a drive to balance model efficiency with calculation precision, evidenced by the shared [optimizer library incorporating Kahan summation](https://optimi.benjaminwarner.dev).
- **Crediting Sparks Queries**: The community saw several inquiries regarding missing or delayed credits from various platforms like OpenAI, with prompts to check pinned comments for guidance and centralized messages to streamline credit-related discussions.
- **Modal Makers, Datasette Devotees, and Chatbot Chumminess**: The promise of finetuning LLMs for Kaggle competitions via Modal, Datasette's command-line appeal, and character-focused AI dialogue models from Anthropic Claude agitated the technical waters, underscoring a zest for integration, analytical insights, and user experience.

Remember to check within each message history for specific links provided, such as [Datasette's stable version](https://llm.datasette.io/en/stable/), Simon Willison’s [GitHub repository](https://github.com/simonw/simonwillisonblog), and mentioned [meetup events](https://lu.ma/iulmro47) for more details on these topics.

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Smartwatches Clash with Tradition**: A playful exchange brought up the dominance of smartwatches such as **WearOS** versus traditional digital watches, spiced up with a tongue-in-cheek dig at Apple, highlighting individual preferences in wearable technology.
- **Perplexity AI Grapples with Tech Issues**: **Perplexity AI** users reported issues with file uploads and image interpretation features, hinting at possible downtime. However, the discussion about different language models like **GPT-4o**, **Claude Sonnet**, and **Llama 3 70B** focused on their performance, with members leaning towards **GPT-4o** as the top contender.
- **Forbes Flames Perplexity for Plagiarism**: A debate unfolded regarding a **Forbes article** alleging that **Perplexity AI** plagiarized content, illustrating the tightrope walk of ethical practices in the era of rapidly evolving AI. The topic underpins the importance of accountability and proper attribution within the AI landscape. [Forbes article on Perplexity’s Cynical Theft](https://www.forbes.com/sites/randalllane/2024/06/11/why-perplexitys-cynical-theft-represents-everything-that-could-go-wrong-with-ai/)
- **AI Ethics and Industry Progress Snapshot**: Ethical concerns surfaced regarding **Nightshade technology** and its potential abuses in sabotaging AI models, while **Perplexity AI** was commended for its advanced features in comparison to **SearXNG**. Another palpable moment in tech history was marked by the shutdown of **ICQ** after 28 years, showing the relentless pace of change in communication technology. [ICQ Shuts Down](https://www.perplexity.ai/page/ICQ-shuts-down-34n3T1XmQpuRpDB9VJcKVw)
- **API Integration Steps Confirmed**: There were confirmations around the initial setup of **Perplexity API** with a simple acknowledgment "This seems to work," followed by directive advice to *"add API key here"*, key for API integration - a testament to the nitty-gritty of getting digital tools up and running.

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **GPU Gold Rush in Canadian Markets**: **3090 GPUs** are hitting affordable prices, approximately 450 USD on **Kijiji Canada**, sparking interest in rig building among the community.
- **Heat: The Byproduct with Benefits**: Innovative suggestions such as using **4090 GPUs** to heat hot tubs were floated, looking at GPU rigs as alternative heat sources while pondering sustainable data center designs.
- **Meticulous Quest for 100% Determinism**: One engineer achieved 100% determinism in their code but encountered issues such as loss values hitting -inf, indicating potential bugs needing resolution.
- **Efficiency Drive in Custom Matmul Implementations**: Efforts to improve computational efficiency resulted in a custom matmul achieving 70% speed of cuBLAS, with discussions ongoing about removing dependencies on cuBLASLt and tackling the difficulties of implementing stable FP8.
- **Matrix Multiplication Shakeup**: The [Scalable MatMul-free Language Modeling](https://arxiv.org/abs/2406.02528) paper presents a novel approach that reduces memory usage and maintains strong performance in LLMs, attracting attention and implementation efforts within the community.

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Whisper WebGPU Gives a Voice to Browsers**: [Whisper WebGPU](https://x.com/xenovacom/status/1799858691639796089) enables rapid, on-device speech recognition in the browser, supporting over 100 languages, promising user privacy with local data processing.
- **Training an Expert Finance LLM - Live!**: Engineers with a yen for finance can tune into a [live event](https://events.dataphoenix.info/live) featuring speaker Mark Kim-Huang, which promises insights into training an Expert Finance LLM.
- **Glimpse into Google's RecurrentGemma 9B**: RecurrentGemma 9B has set the community abuzz with a [post](https://x.com/osanseviero/status/1800607752038818260) highlighting its performance on long sequences. Members are keen on exploring the model's potential.
- **Feasible Fusion with Fine-Tuned Mistral**: Fine-tuning models on Mistral can be explored further in a detailed [Medium article](https://medium.com/ai-artistry/craft-your-ai-vision-fine-tuning-magic-with-mistral-on-mistral-server-6c9335232159), offering practical guidance for engineers working with adaptable AI solutions.
- **Diffusers to Support SD3 - Anticipation Peaks**: The integration of `diffusers` for SD3 has the community on edge, with engineers eagerly waiting for the rollout of this advanced functionality.
- **Dalle 3 Dataset Provides a Creative Cache**: AI that are fueled by diverse data might benefit from the 1 million+ captioned images offered by Dalle 3, available on this [Dataset Card](https://huggingface.co/datasets/ProGamerGov/synthetic-dataset-1m-dalle3-high-quality-captions).
- **Simpletuner Advocated for Diffusion Models**: Amid discussions, **simpletuner** gains recommendation for training diffusion models, while warnings suggest that **diffusers examples** may need tailoring to fit the bill for specific tasks.
- **Quantization Quagmires and Queries**: Optimizing AI with quantization tools like **quanto** and **optimum** has engineers sharing their trials, signaling a need for more robust and error-proof solutions in model deployment.
- **Google Gemini Flash Module - Show Us the Code**: There's a clap of interest from the AI community for Google to [open-source Google Gemini Flash](https://github.com/google/gemma.cpp/issues/221), citing its potential benefits and sought-after capabilities.

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

**Ilya Sutskaver Strikes with Generalization Insights**: Ilya Sutskaver delivered a compelling lecture at the Simons Institute on generalization, viewable on YouTube under the title *An Observation on Generalization*. Separately, Neel Nanda of DeepMind discusses memorization versus generalization on YouTube in *Mechanistic Interpretability - NEEL NANDA (DeepMind)*.

**Llama vs. GPT Showdown**: The performance of Llama 3 8b instruct was compared with GPT 3.5, highlighting Llama 3 8b's free API on Hugging Face. GPT-4o’s coding capabilities sparked a debate regarding its performance issues.

**Enterprise Tier: To Pay or Not to Pay?**: Opinions were divided on the worthiness of the GPT Enterprise tier, despite benefits like enhanced context window and conversation continuity. A user conflated Teams with Enterprise, indicating a misunderstanding about the offerings.

**Bootstrap or Build? That is the AI Question**: Members suggested finetuning an existing AI such as Llama3 8b or seeking open-source options over building a GPT-like model from scratch, specifically tailored to one's niche.

**Technical Trouble Ticket**: Members faced various technical issues, including uploading PHP files to Assistants Playground despite support claims, and error messages while generating responses with unspecified solutions. A request for reducing citations from a GPT-4 assistant trained on numerous PDFs was also noted; they wish to prune citations while maintaining data retrieval.

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

**Qualcomm's AIMET Critiqued**: An individual aired grievances about the usability of Qualcomm's **AIMET Library**, describing it as the "worst library" encountered.

**Rust Gets Cohesive with RIG**: [**RIG**](https://github.com/0xPlaygrounds/rig), an open-source Rust library for building **LLM-powered applications**, was released, featuring modularity, ergonomics, and Cohere integration.

**Questions Arise Over PaidTabs' AI In Integrations**: There's speculation within the community about **PaidTabs** potentially using **Cohere AI** for message generation, focusing on the absence of audio capabilities in Cohere AI as per their [June 24 changelog](https://blog.paidtabs.com/paidtabs-june-24-changelog/).

**Musical Engineers Might Form A Band**: Conversations veered into sharing musical hobbies, suggesting the potential for a community band due to the number of music enthusiasts.

**Pricey Joysticks for Flight Sim Fanatics**: Members debated the steep pricing of advanced joystick setups like the [VPC Constellation ALPHA Prime](https://virpil-controls.us.com/vpc-constellation-alpha-prime-r.html), joking about the cost comparison to diamonds.

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Attention Hypernetworks Enhance Generalization**: An [arXiv paper](https://arxiv.org/abs/2406.05816) was shared indicating that Transformers can improve compositional generalization on intellectual tasks like the symbolic Raven Progressive Matrices by leveraging a low-dimensional latent code in multi-head attention.
- **Evaluating Next-Gen LlamaGen**: The [LlamaGen](https://github.com/FoundationVision/LlamaGen) repo suggests that autoregressive models like Llama excel in image generation, rivaling state-of-the-art performance and offering detailed explanations and tutorials.
- **DPO and Autoregressive Models Battle for Multi-turn Convos**: Members debated the under-researched area of Deterministic Policy Optimization (DPO) for multi-turn conversations, with suggestions to explore the Ultrafeedback dataset and MCTS-DPO approach.
- **Dataset and Architecture Dust-ups**: Discussions included the challenges in sourcing small coding datasets for Large Language Model (LLM) training, as well as critiques and skepticism around novel research papers and models like Griffin, Samba, and their place in efficiently handling long contexts.
- **Searching for the Ultimate Pre-training Dataset**: A quest for an open-source dataset similar to DeepMind's LTIP yielded a recommendation for the [DataComp dataset](https://huggingface.co/datasets/mlfoundations/datacomp_1b), believed to outperform LAION for CLIP models due to its richer 1 billion image-text pair compilation.

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Scripting Python for Chatbot Shenanigans**: A member brainstormed integrating a Python script into **LM Studio** to allow a chatbot to mimic roleplay, directed to adhere to official [LM Studio documentation](https://lmstudio.ai/docs) for setup. The **RecurrentGemma-9B** was suggested for addition, backed by its [Hugging Face page](https://huggingface.co/google/recurrentgemma-9b).
- **Gripes with Glitches in LM Studio 0.2.24**: Users flagged severe bugs in **LM Studio 0.2.24**, complaining about token counting errors and shaky model/GPU utilization. A more curious query delved into the feasibility of bot-created PowerPoint presentations using the OpenAI API.
- **Zooming in on GPU and Model Match-ups**: Lively debates in model and GPU matching homed in on finding the speediest options for an **RTX 2070**, namely **Mistral7b**, **Codestral**, and **StarCoder2**. Recommendations also orbited coding-specific models like **Codestral**, tuning tips with **Unsloth**, and persistence strategies using **vectordb**.
- **GPU Market Price Pulse**: Members exchanged notes on worldwide **P40 graphics card** pricing, diarizing deals seen on eBay for as low as **$150 USD**. Further commentary covered Aussie GPU prices from **RTX 3060** at $450 to **RTX 4090** at $3200, laying bare the realities of scarcity, power, and additional considerations.
- **A Peek at the Modern UI**: A snapshot was shared of the "modern UI" in the **AMD ROCm** tech preview frame, with a [linked photo](https://cdn.discordapp.com/attachments/1130536562422186044/1245478003044257932/image.png?ex=666a08c7&is=6668b747&hm=62024d42535148d6a9a7f23860dc0c011c6a2a48dba4759c91e04bb0f2fbe03f) evoking kudos from one member for its stylish manifestation.

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **RAG's Rocky Road in Real-World Deployment**: An [announcement](https://discord.com/channels/1059199217496772688/1073670729054294197/1250214985762603079) in the guild invites insight on deploying **Retrieval-Augmented Generation (RAG)** in enterprise environments, with Quentin and Shreya Shankar gathering feedback through interviews.
- **Supercharging Graph RAG via Entity Deduplication**: A tutorial shared in the guild suggests enhancing Graph RAG's efficiency by introducing entity deduplication and by employing custom retrieval methods, which can be further explored in the [full tutorial](https://t.co/fiFwDQS6WT).
- **Elevating RAG with Excel's Spatial Rigor**: A conversation emphasized the challenge of adapting RAG for use with Excel files, noting the significance of a well-organized spatial grid to ensure effective functionality, with more context provided in the referenced [tweet](https://t.co/vbdk8Yuw2t).
- **Vector Indexing Woes and Wins**: Members engaged with multiple concerns including S3DirectoryReader decrypting issues, failure of `MetadataFilter` in Redis index, markdown formatting challenges in the vector database, and strategies for customizing prompts in `CondensePlusContextChatEngine`.
- **Cracking the Context Code in Text-to-SQL Queries**: An AI engineer in the community pointed to an issue with context recognition in text-to-SQL queries where determining the nature of an item (such as if "Q" refers to a product name) remains a challenge, leading to incorrect SQL queries being generated.

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **ARC Prize Stirs Discussions**: The newly announced **$1,000,000+ ARC Prize** for developing a solution to the ARC-AGI benchmark spurred extensive talks, highlighting the benchmark's role in measuring general intelligence through skill acquisition efficiency. The industry's apparent lack of awareness about the ARC-AGI was also noted as concerning, despite its significance.
- **Tech Media Gets Mixed Reviews**: A TechCrunch article by Kyle Wiggers about AI was criticized for a superficial approach, while reactions to a podcast interview touching on the relationship between AI and human intelligence were mixed, with some points of contention regarding the role of genetic pedigree in determining intelligence.
- **Microsoft Secures OpenAI Stake**: A tweet shed light on how Microsoft acquired a 49% stake in OpenAI through leveraging discussions with Tesla, using OpenAI involvement as an incentive to potentially draw Tesla onto the Azure platform.
- **Bot Development and AI Updates Generate Buzz**: Issues and developments in AI tools were discussed: `june-chatbot` encountering NVDA API errors indicating stability problems; "SnailBot" earning its name for sluggishness; frustrations with LMSYS; and excitement about Dream Machine’s new text-to-video capabilities from Luma Labs.
- **AI Reinforcement Learning Evolves**: Members dissected an Apple blog post sharing their hybrid data strategy that involves human-annotated and synthetic data and discussed discrepancies in PPO (Proximal Policy Optimization) implementations from a tweet by @jsuarez5341, which contrasts with the pseudocode. Meanwhile, Tulu 2.5's performance updates and Nathan Lambert's exploration into RL practice reveal the community's deep dive into current AI methodologies.

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Collaborators Wanted for Generalist Model**: Manifold Research is on the lookout for collaborators to work on **transformers for multimodality and control tasks**, with the goal of creating a large-scale, open-source model akin to **GATO**. You can join their efforts via [Discord](https://discord.com/invite/a8uDbxzEbM?ref=manifoldrg.com), contribute on [Github](https://github.com/ManifoldRG?ref=manifoldrg.com), or check out their expectations on [Google Docs](https://docs.google.com/document/d/e/2PACX-1vQgq32ChlP_e26mRPgfC31lZJCcAHAgbJ_Tn1nfzq8pfysoPAUqAWnel87Qc26h2Q/pub?ref=manifoldrg.com).
- **Smart Factories Meet Group Chat**: Discussion around [GendelveChat](https://x.com/gendelvechat/status/1800580692046405784?s=46&t=eY--9rPoOkHV-u9kocxCMA), which showcases a simulation of group chat UX for industries like smart factories using @websim_ai, and StabilityAI releasing the open weights for [Stable Diffusion 3 Medium](https://x.com/StabilityAI/status/1800875914299048404).
- **AI Developments and Data Pipeline Proposals**: Apple's AI gets dissected, highlighting its 3B model and compression techniques in [Introducing Apple Foundation Models](https://machinelearning.apple.com/research/introducing-apple-foundation-models). [Stable Diffusion 3 Medium is announced](https://huggingface.co/stabilityai/stable-diffusion-3-medium) featuring visual improvements and performance gains, while a data pipeline leveraging **Cat-70B** and tools like **oobabooga** for ShareGPT data is pitched for the Character Codex project.
- **Empire of Language Models**: Amid requests for Japanese language model recommendations, users suggested **Cohere Aya** and **Stability AI** models with API access, especially praising the **Aya 35B** for its multilingual capabilities including Japanese. For more capabilities, **Cohere Command R+** (103B) was the preferred choice for Japanese model needs.
- **Console and Open-source Queries in WorldSim**: A recent update has made writing and editing longer console prompts more user-friendly on both mobile and desktop interfaces. Curiosity arose regarding WorldSim's open-source status, but no confirmation was provided in the queried time frame.

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

**Elon Musk Battles Apple and OpenAI**: Elon Musk reportedly took action against Apple's Twitter account following their partnership with OpenAI, a development highlighted with a link to a post by Ron Filipkowski on [Threads.net](https://www.threads.net/@ronaldfilipkowski/post/C8F8woLt7rT).

**Google's Gemma Goes Recurrent**: Google's [RecurrentGemma 9B](https://huggingface.co/collections/google/recurrentgemma-release-66152cbdd2d6619cb1665b7a) is out, capable of handling long sequences quickly while maintaining quality on par with the base Gemma model, as heralded by [Omar Sanseviero](https://x.com/osanseviero/status/1800607752038818260).

**Transformer Learning Challenged by ‘Distribution Locality’**: The learnability of Transformers faces limits due to 'distribution locality,' which is explored in a paper on [arXiv](https://arxiv.org/abs/2406.06467), indicating challenges for models in composing new syllogisms from known rules.

**Revising CC12M dataset with LlavaNext Expertise**: The **CC12M dataset** received a facelift using **LlavaNext**, resulting in a recaptioned version now hosted on [HuggingFace](https://huggingface.co/datasets/CaptionEmporium/conceptual-captions-cc12m-llavanext).

**Global Debut of a TensorFlow-based Machine Learning Library**: An engineer announced the launch of their TensorFlow-centric machine learning library capable of parallel and distributed training, supporting a slew of models like Llama2 and CLIP, introduced on [GitHub](https://github.com/NoteDance/Note).

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

**TPUs in Mojo's Future**: Members discussed the possibility of **Mojo** utilizing **TPU hardware** if Google provided a TPU backend for MLIR or LLVM, indicating future support for diverse architectures without waiting for official updates due to planned extensibility.

**Up-to-Date with Modular Releases**: A new **Mojo compiler version** `2024.6.1205` was released, featuring conditional conformance that received positive commentary, along with inquiries about recursive trait bounds capabilities. Updating instructions and details can be found [in the latest changelog](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md).

**Diving into Mojo's Capabilities and Quirks**: A code change from `var` to `alias` offered no performance gain, while issues with outdated `Tensor` module examples were addressed and a successful pointer conversion solution was introduced in a [recent Pull Request](https://github.com/modularml/mojo/pull/3007).

**Modular's Multimedia Updates**: Modular has been active across platforms with a [new YouTube video release](https://www.youtube.com/watch?v=uookgZ7Ojg8) and a tweet update from their [official Twitter account](https://twitter.com/Modular/status/1800948901652181260).

**Community Discussions and Resources**: Exchanges ranged from recommendations for learning **Mojo through VSCode**, with a potential resource at [Learn Mojo Programming Language](https://ruhati.net/mojo), to reflections on tech influencers serving as modern-day programming critics, highlighting a [Marques and Tim interview](https://www.youtube.com/watch?v=pMX2cQdPubk) among shared content.

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

**Zep Eases Memory Concerns Within Free Boundaries**: Participants identified **Zep** as an ace for memory management, provided that usage remains within its **free tier** limitations.

**Apple Tosses Freebies into the Tech Ring**: **Apple's** move to offer certain services free of charge stirred conversations, with members acknowledging it as a significant competitive edge.

**OpenAI's API Wallet-Friendly Pricing**: Debate emerged over the OpenAI API's pricing, with mentions suggesting a range of **$5-10 per month**, highlighting the affordability of OpenAI's offerings for engineers.

**Configuring GCP for Advanced Models**: A user successfully implemented **GPT-4o** on their **GCP account**, though flagged high costs and troubles when changing the default model to **gemini-flash** or **codestral**.

**OpenInterpreter Gains Momentum**: Comprehensive resources were spotlighted, including a [GitHub repository](https://github.com/OpenInterpreter/open-interpreter), [Gist for code](https://gist.github.com/0xrushi/e56085f93698c7267af9b1ba9643dc7a), and a [uConsole and OpenInterpreter video](https://odysee.com/@rushi:2/Openinterpreter-01-uconsole-test-part1:2), with users brainstorming about enhancing voice interactions potentially via a [mini USB mic](https://www.amazon.com/dp/B071WH7FC6).

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

**Custom Metrics for LLMs Made Easy**: [DeepEval](https://docs.confident-ai.com/docs/metrics-introduction#using-a-custom-llm) allows users to smoothly integrate custom evaluation metrics for language models, enhancing capabilities in **G-Eval, Summarization, Faithfulness,** and **Hallucination**.

**Transparent AI with Uncensored Models**: A heated discussion identified the growing interest in **uncensored models** among users, acknowledging their value in providing unfiltered AI responses for diverse applications.

**WizardLM-2's Surprisingly Low Price Tag**: Queries around **WizardLM-2’s** affordability led to insights that it might save on costs by utilizing fewer parameters and strategic GPU rentals, sparking discussions among members on the model’s efficiency.

**Self-Hosting vs. OpenRouter**: Debating the trade-offs, members concluded that **self-hosting large language models (LLMs)** might only make economic sense under constant high demand or if offset by pre-existing hardware capabilities, compared to solutions like OpenRouter.

**GPU Rentals for Batch Inference**: The guild exchanged ideas on the viability of renting GPUs for batch inference, touching on cost benefits and efficiency, and suggesting tools such as **Aphrodite-engine / vllm** for optimizing large-scale computations.

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Testing the Bounds of Local Model Training**: An engineer tested running a **7 billion parameter** model locally using the **mlx library** and reported it as "slow af, but works lmao," while others suggested trying **llama.cpp** for better performance. Another member looked for insights on training **QWEN 2** with 1.5 billion parameters, but no specific performance data was provided in the discussions.
- **Runpod's Perplexing Path Persistence**: Members discussed an issue on Runpod where mounting a data volume to **/workspace** often results in the overwrite of **/workspace/axolotl**, necessitating reinstallation or re-cloning of Axolotl - a persistent annoyance noted in the development environment setup.
- **Step Aside, Redundant Instructions**: Within the documentation channel, it was noted that "Step 2" is superfluous in the model upload process, as the repository is created automatically, indicating an update to documentation may be warranted.
- **PyTorch Distributed Drama**: An engineer encountered a *ChildFailedError* when using PyTorch's distributed launcher, prompting advice to check environmental setup, verify configurations, and potentially increase shared memory, with further troubleshooting steps available in the [Accelerate documentation](https://github.com/huggingface/accelerate/tree/main/docs/source/basic_tutorials/troubleshooting.md#L50L145).
- **LoRA's Learning Curve**: Queries were made about utilizing LoRA for transitioning from completion format to instruction format training and merging qLoRA training results on Mistral 7B within a 24GB GPU system, suggesting a discussion about handling resource limitations is in play.

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Google Plays Its Next Hand with RecurrentGemma 9B**: Google's **RecurrentGemma 9B** promises breakthroughs in processing long sequences swiftly, boasting similar prowess to **Gemma** while introducing base and instruct-tuned versions. The model's details and its comparison to its predecessor can be found in the collection [here](https://huggingface.co/collections/google/recurrentgemma-release-66152cbdd2d6619cb1665b7a) and the underlying research in the [Griffin paper](https://arxiv.org/abs/2402.19427).
- **Granular Analysis on Alexa AI's Pitfalls**: Mihail Eric pulled back the curtain on **Alexa AI**'s failures, citing technical disconnect and organizational inefficiencies as key roadblocks to innovation, backed by deep resources yet resulting in few public breakthroughs. The detailed thread sheds light on behind-the-scenes challenges and is available [here](https://x.com/mihail_eric/status/1800578001564057754).
- **ARC Prize Forges New Frontier in AI Benchmarking**: A prize connected to the ARC task dataset, which now includes over 4,100 interaction histories, aims to elevate understanding and development regarding human problem-solving methods in AI. Resources such as [videos](https://youtu.be/zbo6SdyWGns?si=5UypK-JD5h7Gz-SJ) and multiple datasets are available with invitation for participation opened through [this link](https://neoneye.github.io/arc/).
- **Google Unveils Healthcare-Specific Language Model**: Google's latest addition, the **Personal Health Large Language Model**, harnesses wearables data to deliver tailored health insights, reportedly surpassing industry experts. In-depth information about this model's capabilities and design can be found [here](https://x.com/chefjeffsf/status/1800597192593621100).
- **Stanford Shares Insights on AI's Rapid Evolution**: A lecture at Stanford by hwchung27 offered a valuable glimpse into the state-of-the-art in AI, drawing attention to the transformative impact of cheaper compute resources and the growth of Transformer architectures. Watch the insightful [lecture](https://youtu.be/orDKvo8h71o?si=RIfyZ7NSUAJifOBF) and review the accompanying [slides](https://docs.google.com/presentation/d/1u05yQQaw4QXLVYGLI6o3YoFHv6eC3YN8GvWD8JMumpE/edit?usp=sharing) for a detailed narrative.

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Bounty Hunting for RetinaNet on MLPerf**: George Hotz has announced a **$600 bounty** for implementing RetinaNet on the MLPerf benchmark, promising to double it if the contribution is accepted into MLPerf. See the pull request related to this [here](https://github.com/tinygrad/tinygrad/pull/4245).
- **Hungry for Efficient Data Loading**: Engineers have voiced concerns over the time sink associated with optimising data loading in PyTorch, especially on HPC clusters, suggesting [WebDataset](https://github.com/webdataset/webdataset) as a viable solution.
- **Drafting TinyGrad's Data Loader**: George Hotz shared plans for TinyGrad's data loader API, which includes a function to load records, shuffle buffer size, and batch size, mentioning Comma's "gigashuffle" for reference.
- **TinyGrad Evolves with 0.9.0**: TinyGrad version 0.9.0 is now available in the Nix repository, as confirmed by the [GitHub Pull Request](https://github.com/NixOS/nixpkgs/pull/316931), and features the inclusion of gpuctypes directly in the codebase.
- **MLPerf Benchmarking and Community Buzz**: The latest MLPerf results are published, featuring tinybox red/green benchmarks, alongside media coverage like a German article on TinyGrad which can be read on [Heise.de](https://www.heise.de/news/KI-Benchmark-MLPerf-Erste-AMD-Beschleuniger-mit-Minimalauftritt-9760531.html), with further discussions hinting at a future blog post comparing TinyGrad's speed to theoretical limits.

---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **LLMs Require Structured Output**: Members highlighted the necessity for **LLMs** (Large Language Models) to have defined grammar or JSON schemas to be effectively utilized as agents. The standardization of outputs to be parsable by external programs has been noted as critical for usefulness in the application layer.
- **Streamlining llamafile with Integrated Schemas**: The conversation on optimizing `llamafile` usage proposed a two-step streamline process: first converting JSON schema to grammar, and then integrating that grammar for utility, with a command example given: `llamafile --grammar <(schemaify <foo.sql)`.
- **Efficient Packaging of Shared Objects**: A technical discussion emerged concerning the most effective method for including `ggml_cuda.so` and `ggml_rocm.so` in `llamafile` distributions, including a shared Bash script and a mention of necessary manual adjustments for different libraries such as AMD and tinyblas.
- **Magit as a Sync Solution**: A humorous referral to a video titled "Interview with an Emacs Enthusiast in 2023 [Colorized]" was made to exemplify the use of [Magit](https://www.youtube.com/watch?v=urcL86UpqZc), a Git interface for Emacs, demonstrating its application for syncing files like `llama.cpp`.
- **Schema Applications in LLMs**: The dialog underlined a community interest in applying structured data schema to improve the output of Large Language Models, signaling a trend in engineering circles toward enhancing LLM integration with downstream systems.

---

## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

- **Apple Interfacing Prospects with LORA Analogue**: Apple's adapter technology has been compared to **LORA layers**, suggesting a dynamic loading system for local models to perform a variety of tasks.
- **The Contorting Web of HTML Extraction**: AI engineers explored different tools for HTML content extraction, like [htmlq](https://github.com/mgdm/htmlq), [shot-scraper](https://shot-scraper.datasette.io/en/stable/javascript.html#example-extracting-page-content-with-readability-js), and `nokogiri`, with Simonw highlighting the use of `shot-scraper` for efficient JavaScript execution and content extraction.
- **Shortcut to Summation Skips Scrutiny**: **Chrisamico** found it more efficient to bypass technical HTML extraction and directly paste an article into ChatGPT for summarization, foregoing the need for a complicated `curl` and `llm` system.
- **Simon Says Scrape with shot-scraper**: Simonw provided instructions on utilizing `shot-scraper` for content extraction, advocating the practicality of using CSS selectors in the process for those proficient in JavaScript.
- **Command Line Learning with Nokogiri**: Empowering engineers to leverage their command line expertise, Dbreunig shared insights on using `nokogiri` as a CLI tool, complete with an example for parsing and extracting text from HTML documents.

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

**LangChain Postgres Puzzles Programmers**: Engineers reported issues with **LangChain Postgres documentation**, finding it lacks a checkpoint in the package which is crucial for usage. The documentation can be found [here](https://api.python.langchain.com/en/latest/checkpoint/langchain_postgres.checkpoint.PostgresSaver.html), but the confusion continues.

**GPT-4 Gripes in LangChain**: A member flagged an error using **GPT-4** with `langchain_openai`; guidance was offered to switch to `ChatOpenAI` because `OpenAI` uses a legacy API not supporting newer models. More information about the OpenAI API can be found [here](https://platform.openai.com/docs/api-reference/completions).

**Sharing Snafu in LangServe**: Difficulty sharing conversation history in LangServe's chat playground was discussed, with users experiencing an issue where the "Share" button leads to an empty chat rather than showing the intended conversation history. This problem is tracked in [GitHub Issue #677](https://github.com/langchain-ai/langserve/issues/677).

**No Cost Code Creations at Nostrike AI**: Nostrike AI has rolled out a new free python tool allowing easy creation of CrewAI code with future plans to support exporting Langgraph projects, inviting users to explore it at [nostrike.ai](https://nostrike.ai/).

**Rubik's AI Recruits Beta Testers**: Rubik's AI, touted as an advanced AI research assistant and search engine, seeks beta testers with the enticement of a 2-month free trial using the promo code `RUBIX`, covering models like GPT-4 Turbo and Claude 3 Opus. Check it out [here](https://rubiks.ai/).

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

**Discord Amps Up with Apps**: Members can now enhance their Discord experience by adding apps across servers and direct messages starting June 18. Detailed information and guidance on app management and server moderation can be found in the [Help Center article](https://support.discord.com/hc/articles/23957313048343) and developers can create their own apps with the aid of a [comprehensive guide](https://discord.com/developers/docs/tutorials/developing-a-user-installable-app).

**Cache Conundrums in Torchtune**: A dialogue has opened up regarding the increased use of cache memory by **Torchtune** during each computational step, with community members probing deeper to understand this performance characteristic.

**Tokenizer Revamp on the Horizon**: An RFC detailing a significant overhaul of tokenizer systems sparked conversations about multimodal feature integration and design consistency, which is available for review and contribution on [GitHub](https://github.com/pytorch/torchtune/pull/1082).

---

The **LLM Perf Enthusiasts AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.

---

The **AI Stack Devs (Yoko Li) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.

---

The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.

---

The **DiscoResearch Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.

---

The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.

---

The **YAIG (a16z Infra) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.

---

# PART 2: Detailed by-Channel summaries and links

{% if medium == 'web' %}

### **Stability.ai (Stable Diffusion) ▷ #**[**announcements**](https://discord.com/channels/1002292111942635562/1002292398703001601/1250434682743685161) (1 messages):

- **Stable Diffusion 3 Medium Open Weights Released**: Stability.ai has announced the open weights for **Stable Diffusion 3 Medium**, which is described as *"the latest and most advanced text-to-image AI model"* in the series. Find out more in the [announcement post](http://stability.ai/news/stable-diffusion-3-medium).
- **Exceptional Quality and Photorealism**: The new model delivers images with *"exceptional detail, color, and lighting"* and addresses common pitfalls like realism in hands and faces using a 16-channel VAE.
- **Advanced Prompt Understanding**: It can comprehend long, complex prompts involving spatial reasoning and compositional elements, using three text encoders to balance performance and efficiency.
- **Superior Typography**: Achieves high-quality text rendering with fewer spelling and spacing errors, leveraging the Diffusion Transformer architecture.
- **Resource Efficiency and Customization**: The model runs efficiently on standard consumer GPUs without performance degradation and supports fine-tuning for customization with small datasets.

---

### **Stability.ai (Stable Diffusion) ▷ #**[**general-chat**](https://discord.com/channels/1002292111942635562/1002292112739549196/1250163004239249610) (734 messages🔥🔥🔥):

- **SD3 struggles with image and anatomy quality**: Users noted that **Stable Diffusion 3 (SD3)** has issues with generating images accurately, specifically with human anatomy. One remarked, *"it instantly ate all my 16GB ram and 8GB vram and crashed at 256x256"* while another said, *"results are pretty underwhelming, hardly better than SDXL models."*
- **Controversial licensing for SD3 models**: There was extensive discussion around the restrictive **licensing of SD3** for commercial use, making it impractical for many. One user clarified, *"you need a license for commercial use, even if you only want to sell the output images,"* leading to debates on its viability and comparisons to other models like **PixArt**.
- **Finetuning challenges and skepticism**: Members expressed concern over the ability and cost to finetune SD3, particularly for popular use cases like NSFW content. A user highlighted, *"finetunes get made to base models. This is the base model that was trained at great expense,"* indicating that further community efforts will be required.
- **Comparative performance disappoints some**: Despite being a newer model, SD3 has left many unimpressed compared to **SD 1.5** and **SDXL**. A member said, *"when SD3 produces better results, they are sometimes way better than SDXL,"* but these instances are inconsistent.
- **Technical setup and install issues**: Various users sought help installing SD3 within their existing frameworks like **ComfyUI**, **StableSwarm**, and **diffusers**. Some faced errors and shared setup resources, with a user mentioning, *"now have to wait for auto1111, having lots of workflows using it."*

---

### **Unsloth AI (Daniel Han) ▷ #**[**general**](https://discord.com/channels/1179035537009545276/1179035537529643040/1250170775781249164) (322 messages🔥🔥):

- **Members discuss VRAM requirements for large models**: There's an ongoing discussion about the **VRAM requirements for Qwen2 72b** and its relative efficiency. Users suggest it needs around **48GB of VRAM**, similar to Llama 3 70b, though some jokingly suggest requiring a "Super computer" or even "Windows XP."
- **Convert Pytorch Bins to Safetensors**: [This tool](https://huggingface.co/spaces/safetensors/convert) was suggested for converting Pytorch bins into safetensors. There was a failed attempt due to insufficient RAM on Google Colab, with users proposing moving to Kaggle for more resources.
- **Fine-tuning challenges and solutions**: Users discuss issues with low-rank configurations when fine-tuning models like **Qwen2-1.5b** using methods including Qlora and Galore. Recommendations included increasing the rank to at least 64 or 128 to avoid training failures.
- **Performance drop after merging to 16-bit**: A user noted that their model's performance worsened post-merging to 16-bit, citing issues like repeated results and hallucinations. This was attributed to possible tokenizer and GGUF issues, pending fixes in upcoming updates.
- **Community dynamics and support**: Members highlighted the challenges in maintaining community responses, contemplating the need for volunteer or paid help due to the increased volume of user inquiries. They also discussed contributions and improvements in areas such as **multi-GPU support** and **finetuning scripts**.

---

### **Unsloth AI (Daniel Han) ▷ #**[**random**](https://discord.com/channels/1179035537009545276/1179039861576056922/1250245408891080797) (20 messages🔥):

- **Unsloth updates Korean colab, prompts curiosity**: A member noticed an update in the **Korean colab** by Unsloth, questioning if it utilizes Korean prompts exclusively for answers. They shared their observation that the response "Describe the planet Earth extensively" was unexpectedly amusing.
- **Anthropic's Transformer Lens debugs model attention**: A member mentioned experimenting with **Anthropic's Transformer Lens** to debug a model's attention during inference. They believe it's a promising approach to solve hallucinations, though it shows how a single neuron's divergence can impact text generation.
- **On-device AI API by Apple sparks discussion**: Discussion centered around the **Apple Intelligence API**, introduced at the [WWDC 2024](https://developer.apple.com/videos/play/wwdc2024/102/) around the 3:00-minute mark. The conversation touched on lora adapters for on-device AI and the profitability of catering to Apple's ecosystem despite potential drawbacks.
- **Financial practicality drives AI support for Apple**: A member justified supporting **Apple's ecosystem** due to the lucrative nature of its user base. They emphasized the practical need to "pay my bills" and pointed out the significant market of "1 billion ready devices" as an untapped potential.
- **Apple’s AI supports broad device range**: Despite initial concerns about compatibility, it was clarified that **Apple AI** will be available on all devices supporting the next OS update. This dispelled fears that the AI would be limited to newer devices only.

---

### **Unsloth AI (Daniel Han) ▷ #**[**help**](https://discord.com/channels/1179035537009545276/1179777624986357780/1250179052145217688) (78 messages🔥🔥):

- **EOS token confusion clarified**: A member asked why one notebook included an **EOS token** while another didn't. Another member clarified that the second notebook uses `<|eot_id|>` as the EOS token, but *"you can still add it if you want though"*.

- **Unsloth installation woes resolved**: One member shared issues installing Unsloth on **GTX 1080/Windows10** but resolved it using specific pip installations. Another member chimed in suggesting *"pip install --force-reinstall jupyter"* to fix import issues in Jupyter Notebook.

- **Performance impact of 4-bit model**: A user questioned the performance difference with a 4-bit model. It was clarified that using the default unsloth configuration with `load_in_4bit=True` generally doesn't deteriorate performance and makes downloads faster.

- **Llama3 model generating self-responses**: A member reported that their **Llama3 quantized model** was generating and responding to its own customer messages. This prompted a query on updating Unsloth and using the correct chat templates.

- **Issues with fine-tuning dataset size**: A user mentioned their dataset had only 30 items, which wasn't sufficient for effective learning. It was suggested to increase the size to at least 300 items and avoid repetition in datasets directly but rather use multiple epochs during training.


---

### **Unsloth AI (Daniel Han) ▷ #**[**community-collaboration**](https://discord.com/channels/1179035537009545276/1180144489214509097/1250443156865159179) (4 messages):

- **Volunteer Steps Up for Documentation**: A member expressed willingness to help with documentation. Another member highlighted that many users ask about **data preparation** and using chat notebooks, indicating these areas need improvement.
- **Potential for Video Tutorials**: The idea of creating **video tutorials** for easier understanding in the future was brought up. This suggests a proactive approach towards more engaging and accessible resources for users.

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**general**](https://discord.com/channels/1238365980128706560/1238365980128706563/1250166178597572759) (25 messages🔥):

- **Zoom recording issues delaying video availability**: The Zoom recordings of recent sessions (Paige Bailey, Ben Clavie) are not available yet due to a transcription step issue. Dan reported filing a ticket with Zoom to resolve this problem.
- **Nehil's blog post on improving prompting**: Nehil wrote a [blog post](https://brain.nehiljain.com/posts/how-i-improved-my-prompting-for-budget-categorization/) based on course learnings to improve prompting for budget categorization. It was well-received by the community, with Dan praising the effort.
- **DOLPHIN crew introduces promising Spectrum method**: Devnull highlighted a new paper on the Spectrum method by the DOLPHIN crew, which is similar to LASER during QLoRA. The paper is available on [arXiv](https://arxiv.org/abs/2406.06623).
- **Challenges and tips for adding pooling layers in Huggingface**: Healthymonkey sought advice on adding pooling layers like AttentionPooling or GemPooling to a finetuned model such as Mistral or Llama. Shamik helped clarify the integration point, specifically the location between the backbone and head.
- **Arguments against broad-scope chatbots**: Chrislevy invited discussion on why general-purpose chatbots with endless scope are a bad idea and sought resources to convince leadership to adopt more scoped interfaces. Sidhusmart suggested using arguments around security/privacy concerning data usage between departments.

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**🟩-modal**](https://discord.com/channels/1238365980128706560/1241044231829848125/1250263048866758821) (7 messages):

- **Finetuning LLMs for Kaggle Competitions**: Members discussed the potential of using **Modal** to finetune LLMs and upload them to **Kaggle** for competitions. One member confirmed they had finetuned **mistral-7B** and used it on Kaggle for inference.
- **Modal Usage and Credit Inquiry**: A user inquired about receiving additional credits for using Modal, reflecting interest in maximizing resource usage effectively.
- **Modal Billing and Security**: An informative response outlined Modal's approach to billing and security. *"Billing limits are prominent and default low,"* and while using Modal for gRPC services includes straightforward authentication, setting up REST APIs requires custom security measures due to the lack of built-in RBAC.

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**learning-resources**](https://discord.com/channels/1238365980128706560/1241089743933149204/1250361991831486528) (2 messages):

- **Parameter Efficient Tuning explained**: PET includes methods like **LoRA** and [prompt tuning](https://github.com/google-research/prompt-tuning) as cited by Lester et al., 2021. FMT stands for full-model tuning.
- **Pretraining data scaling is impractical for GPU-poor users**: A member expressed confusion about the parts of an article discussing pretraining data scaling for finetuning, noting it wasn't actionable due to the high GPU requirements. *"This doesn't seem actionable for GPU-poor us, we won't be doing the full pretraining."*

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**hugging-face**](https://discord.com/channels/1238365980128706560/1241141471814488115/1250388609136328747) (3 messages):

- **New User Thiruvazhi Seeks Credit Confirmation**: A user named Thiruvazhi has expressed that they submitted the form and provided their email and Hugging Face username but have not received their credits yet.
- **Members Directed to Pinned Comments**: A member advised to check the pinned comment for guidance. Another followed up, confirming they found the information useful.

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**langsmith**](https://discord.com/channels/1238365980128706560/1241167367040405544/1250181894511792342) (2 messages):

- **Langsmith disappoints with upfront payment requirement**: One user appreciated that Langsmith offers free credits to test features, but found it *"rather entirely off-putting"* that setting up payment requires a credit card. This upfront requirement was criticized for detracting from the trial experience.
- **Email shared for an inquiry**: Another user shared their email address, [amirhossein.gh@gmail.com](mailto:amirhossein.gh@gmail.com), possibly in relation to an inquiry or support issue. The context of the inquiry, however, was not provided.

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**berryman_prompt_workshop**](https://discord.com/channels/1238365980128706560/1242223275463938221/1250454623391842449) (1 messages):

- **Prompt Workflow Inquiry**: A member asked, "What is your workflow and tools to write and iterate over prompts and evaluate which prompt was better?" No further discussion or responses were provided in the message history.

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**clavie_beyond_ragbasics**](https://discord.com/channels/1238365980128706560/1242223963346698250/1250290352930291742) (8 messages🔥):

- **No Link Between Model and Retrieval Embeddings**: It's confirmed that *there isn't much correlation* between the retrieval embeddings and the LLM being used. A user mentioned that using embeddings by the same provider as your LLM doesn't make any difference; instead, it's better to use whatever performs best for document retrieval.
- **Zoom Video Issue Resolved**: There was a problem with *Ben Clavies' video* on Zoom which is now fixed. The video can be accessed directly on [Zoom](https://us06web.zoom.us/rec/share/5O0wWYxZ8SVoyqyxG_S4U6xeJLzVPaoggNtrK0XQUG8Ts_hq9UBWFOSjdiFCju3a.5LZBUyRziZgbAN_S) or via Maven.
- **Plan for Implementing RAG-based Search**: A user detailed their *MVP plan* to create a RAG-based search system using various approaches and tools. They mentioned incorporating JXNLCO's feedback method, Freddy Aboulton's and Jeremy Howard's fastHTML talks, Eugene Yan's hallucination removal techniques, and deploying on Railway.

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**jason_improving_rag**](https://discord.com/channels/1238365980128706560/1242224099548332132/1250281309381857361) (3 messages):

- **New Newsletter Alert**: A member proudly announced their new newsletter aimed at providing an early glimpse into their personal and technical writing. [Subscribe here](https://subscribe.jxnl.co/).
- **Postponing Model Fine-tuning**: Another member detailed their current use of **GPT-4o** to handle user queries and filter product searches via meta tags without fine-tuning the model. They’ve considered fine-tuning to reduce static prompt info but find the current setup sufficient for their needs.
- **Minimizing Dependencies with Postgres**: A member expressed a preference for keeping dependencies to a minimum by consolidating capabilities in **Postgres**, including managing relational data and vectors. They questioned the necessity of adding **LanceDB** as an additional dependency.

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**saroufimxu_slaying_ooms**](https://discord.com/channels/1238365980128706560/1242224552415596554/1250202649421418517) (12 messages🔥):

- **Curiosity about 4-bit Quantization Calculation**: A member asked how internal model calculations work when a model is quantized in 4-bit, including details about dequantization and multiplication. Another member confirmed that weights are dequantized to bf16 before multiplication.
- **Llama-recipes' Pure bf16 Mode Impresses**: Jeremy Howard shared that Llama-recipes has a "pure bf16" mode utilizing Kahan summation to avoid accuracy issues. He asked if others had tried it, noting his positive experience; another member linked their own [optimizer library](https://optimi.benjaminwarner.dev) incorporating Kahan summation.
- **Weight-Only Quantization Explained**: A member clarified that weight-only quantization involves dequantizing weights before multiplication, typically to dtype bf16 or 8-bit on A100 GPUs. This concept was further endorsed as a beneficial method that contributes to flexibility in creating custom algorithms.
- **Active Research in Optimizer Quantization**: It was mentioned that there is ongoing active research focusing on the quantization of optimizers. This could potentially innovate and optimize model training further.

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**axolotl**](https://discord.com/channels/1238365980128706560/1242542198008975430/1250396870824427564) (3 messages):

- **Facing inconsistency while finetuning with Axolotl**: A user reported an **OSError** while finetuning with Axolotl and utilizing `llama-3`, receiving a message indicating issues connecting to Huggingface to load a config file. They noted that finetuning with `open-llama` worked without issue.
- **Resolved access issues by changing token permissions**: The user later identified the problem as stemming from Huggingface access token permissions. They advised others experiencing the same issue to adjust their token settings to **allow access to public gated repositories**.

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**wing-axolotl**](https://discord.com/channels/1238365980128706560/1242564077151326388/1250183703771218093) (4 messages):

- **Configuring Axolotl on Jarvis-labs for easy use**: A user mentioned using **Axolotl with credits from a course** on a Jarvis-labs machine, which has Axolotl pre-configured. They described it as a "very easy way" to fine-tune a model without a local GPU.
- **Inspect preprocessed data with pure Axolotl**: A user shared a [notebook from Modal](https://github.com/modal-labs/llm-finetuning/blob/main/nbs/inspect_data.ipynb) for inspecting preprocessed data. Another member suggested running Axolotl to preprocess data locally by following the same steps outlined in the notebook and ensuring the correct dataset path in the configuration file to use Axolotl independently.

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**charles-modal**](https://discord.com/channels/1238365980128706560/1242564177952768062/1250202407137312769) (4 messages):

- **Extra Credit Queries Clarified**: A user inquired about not receiving an extra 500 credits, and it was confirmed that credits were distributed around midnight UTC. Another user mentioned that last-minute sign-ups might need to wait until midnight UTC today.
- **Redirect Credit Questions to Specific Channel**: Users were advised to direct any questions about credits to a specified channel. There's a detailed message at the top of that thread to help users with their queries.

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**simon_cli_llms**](https://discord.com/channels/1238365980128706560/1242664474276659320/1250240328410202172) (44 messages🔥):

- **Datasette impresses the crowd**: Multiple users shared their excitement over [Datasette](https://llm.datasette.io/en/stable/), highlighting its seamless integration with `llm` and the sleek web UI for inspecting conversation logs. With comments like *"I cant move my eyes off the screen... awestruck..."*, it is clear the tool has garnered a lot of admiration.
- **Simon Willison’s incredible tooling**: Users praised Simon’s work with comments like *"Simon is a terminal wizard"* and constantly highlighted his prolific and insightful contributions, including his blog [Simon Willison’s Blog](https://simonwillison.net/) and its [GitHub repository](https://github.com/simonw/simonwillisonblog). He also provided a [handout for the talk](https://github.com/simonw/language-models-on-the-command-line/blob/main/README.md).
- **Mistral-7B-Instruct and Hugging Face pipeline**: Queries about installing models like **mistral-instruct** led to sharing the [Hugging Face model card](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) and some Python code using the Transformers pipeline. The discussion helped clarify how to use these models with `llm` commands.
- **vLLM offers OpenAI compatibility**: It was highlighted that vLLM can be integrated with `llm` without needing a special plugin due to its OpenAI compatible API. This context was supported by links to [vLLM’s OpenAI compatibility](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html) and [Datasette's guide](https://llm.datasette.io/en/stable/other-models.html#openai-compatible-models).
- **Anthropic Claude accolades**: Discussions and links such as the [Claude character research](https://www.anthropic.com/research/claude-character) underscored the community’s interest in character-driven AI personas. Users appreciated the thoughtful design and high utility of projects shared during discussions.

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**credits-questions**](https://discord.com/channels/1238365980128706560/1243721538432270388/1250220855439655094) (12 messages🔥):

- **LangSmith Credits MIA**: A member mentioned not receiving their LangSmith credits and asked who to contact for help. **Dan** requested their email address for follow-up.
- **Centralizing Credit Inquiries**: **Michael J. Ward** requested that users with credit issues post in the ⁠hugging-face channel to centralize the queries and ensure they are addressed promptly.
- **OpenAI Credits Not Received**: A member stated they haven't received their OpenAI credits despite completing the forms. **Dan** asked for email verification, while **Hamel** advised checking the OpenAI channel and pinging the relevant person.
- **Additional Credits for Modal Usage**: A user inquired if using Modal could entitle them to additional credits. **Dan** suggested asking Charles in a specific forum for more information.
- **Tracking Credit Allocations**: **doctormiko.** suggested having a document listing all the available credits as they have lost track. This suggestion received no immediate follow-up.

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**fireworks**](https://discord.com/channels/1238365980128706560/1245126291276038278/1250189456170811444) (6 messages):

- **Participants request credit assistance**: Multiple participants have requested assistance with receiving their credits. IDs mentioned include alexey-zaytsev-22319, sudarshansivakumar-4fadb8, srvaldepenas-c068e3, and solmaz-sh-fe1dbf.

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**braintrust**](https://discord.com/channels/1238365980128706560/1245407617031999581/1250273501286367275) (2 messages):

- **Credit Expiry Policy Confirmed**: A member inquired, *“so the credits is only good for 3 months right?”* It was confirmed that **the credits are valid for only 3 months**, but special concerns can be discussed privately with the admin.

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**west-coast-usa**](https://discord.com/channels/1238365980128706560/1245410680065097738/1250250282458157116) (2 messages):

- **Join the West Coast Meetup!**: A link to a [meetup event](https://lu.ma/iulmro47) was shared for participants of the **course** to join. Attendees are advised to indicate their course participation in a questionnaire.

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**europe-tz**](https://discord.com/channels/1238365980128706560/1245425547048386732/1250196475833749596) (3 messages):

- **Greetings from Hamburg**: A member from Hamburg, Germany expressed interest in **regional meetups**. They noticed that some members are based in Berlin and mentioned it's always worth dropping by.
- **Fine-Tuning with Euro24 Data**: Another member inquired if anyone is interested in working on fine-tuning a model using **Euro24 news and related data** like player and match statistics. They appear to be looking for collaborators for this task.

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**predibase**](https://discord.com/channels/1238365980128706560/1245803791710687272/1250500103073169428) (19 messages🔥):

- **Beneyal shares noteworthy papers**: Beneyal posted links to two [papers on arxiv](https://arxiv.org/abs/2405.00732) and [another](https://arxiv.org/abs/1605.07723) discussing recent advancements in the field.
- **Danbecker contributes recent research**: Danbecker provided a link to an [arxiv paper](https://arxiv.org/abs/2310.01352) relevant to the discussion.
- **Laith0x0 notes performance variability**: Laith0x0 highlighted that different r and alpha values (e.g., r=256, alpha=256 vs. r=32, alpha=64) showed better performance depending on the data or task, suggesting the results are highly contextual.
- **Account and support queries prevail**: Multiple users, including laith0x0 and hardboiledfish, reported issues such as difficulty in signing into Predibase and credit discrepancies, which were addressed directly by Michael Ortega through direct messages and support channels.
- **Gratitude expressed for insights**: Several users, including laith0x0, codingwitcher, and silverstar5654, thanked Travis for thorough and illuminating responses, indicating a helpful and collaborative discussion environment.

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**openpipe**](https://discord.com/channels/1238365980128706560/1245927847437008896/1250412608134058075) (2 messages):

- **Users report missing credits**: Both contributors mention issues with their credits not being added. They each supplied their email addresses to identify their accounts for resolution.

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**openai**](https://discord.com/channels/1238365980128706560/1245927985123692575/1250175585557155840) (6 messages):

- **Payment method required for API tier usage**: A member mentioned that they needed to add a payment method and $5 in credits to use Tier 2 API access. This highlights the requirement of payment setup even for tiered access to services.
- **Specific API version resolves issues**: For API calls, the user reported that **gpt-4o** did not work, but **gpt-4o-2024-05-13** worked perfectly. This indicates potential version-specific reliability or updates in the API.
- **Missing OpenAI credits**: A member reported not receiving their OpenAI credits and provided their email for follow-up. This brings attention to issues with credit allocation.
- **Future of GPT fine-tuning and data insights**: A user compiled a list of questions concerning the future availability and usability of GPT-4 and GPT-4-turbo for fine-tuning, the impact of prompt-based techniques vs. fine-tuning, data on fine-tuned model requests, improvements in multi-turn conversation fine-tuning, and feedback mechanisms for OpenAI's development priorities. These questions were intended for a speaker at an unspecified event.

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**hailey-evaluation**](https://discord.com/channels/1238365980128706560/1250194004176273479/1250194275581431931) (64 messages🔥🔥):

- **Shamik highlights must-read paper**: A member praised the paper Hailey referenced as a "total banger" and shared the link: [https://arxiv.org/abs/2405.14782v2](https://arxiv.org/abs/2405.14782v2). This sparked additional insightful discussions around evaluation methodologies.
- **Hailey's intricate responses captivate members**: Hailey detailed the importance of evaluation contexts and the historical context of public academic-created evals. She mentioned the potential coolness of a dataset exploring sysprompts' effects on model behavior.
- **Hailey's numerous resource recommendations**: Codingwitcher compiled an extensive list of Hailey's recommended resources, including Hailey's [Twitter](https://x.com/haileysch__), [Github](https://github.com/haileyschoelkopf), and several scholarly articles and projects.
- **Industry standard and transparency in eval systems**: Members discussed the idea of standardizing evaluations and sharing code for multiple choice questions to mitigate overfitting and ensure transparency. "Private eval sets and dynamic eval sets" were noted as future directions.
- **Hailey's skepticism on LLM-as-judge**: Despite recent trends, Hailey voiced skepticism about LLM-as-judge systems, advocating for more research and suggesting that trained rankers and small evaluators might often be undervalued.

---

### **Perplexity AI ▷ #**[**general**](https://discord.com/channels/1047197230748151888/1047649527299055688/1250166338346029078) (200 messages🔥🔥):

- **WearOS vs Digital Watches**: One member asked, "why wearOS," to which another replied, "Because Apple." They further elaborated, "I personally hate digital watches in general."
- **Issues with Perplexity AI file uploads**: Multiple users experienced problems with Perplexity AI's file uploads and image interpretation features. One member noted, "it kept saying that it couldn't view external files or links," while another confirmed, "Yes, it's down."
- **Dialogue on GPT-4o and models**: There was a detailed discussion about various language models, including GPT-4o, Claude Sonnet, and Llama 3 70B. Members compared their performance and capabilities, with one member noting, "Overall, the best model is probably GPT4o."
- **Concerns about Perplexity AI features**: Users expressed frustration with Perplexity AI's feature set, particularly regarding search and image handling capabilities. One member stated, "Perplexity should be able to support the same features through the API, just the devs are pretty slow at implementing stuff."
- **Forbes Article Controversy**: A discussion emerged about a Forbes article accusing Perplexity AI of plagiarism. One member commented, "if I were Forbes, I’d be pretty pissed too," noting the importance of proper attribution in the news industry.


**Links mentioned**:

- [Why Perplexity’s Cynical Theft Represents Everything That Could Go Wrong With AI](https://www.forbes.com/sites/randalllane/2024/06/11/why-perplexitys-cynical-theft-represents-everything-that-could-go-wrong-with-ai/): It’s the perfect case study for this critical moment: AI is only as good as the people overseeing it.
- [‎Discover Daily by Perplexity: Eric Schmidt's AI Drones for Ukraine, SpaceX's Starship Milestone, Humane's Fire Alert, ALS Research Breakthrough, and D-Day Stories on Apple Podcasts](https://podcasts.apple.com/us/podcast/discover-daily-by-perplexity/id1732181427?i=1000658127961): ‎Show Discover Daily by Perplexity, Ep Eric Schmidt's AI Drones for Ukraine, SpaceX's Starship Milestone, Humane's Fire Alert, ALS Research Breakthrough, and D-Day Stories - Jun 6, 2024
- [who is this?](https://www.perplexity.ai/search/who-is-this-UpstadEaSiSkXzfG8iSgjA): <scratchpad> [The prompt asks "who is he?" and provides an image of a man in a suit smiling at the camera.] [To identify who the person is, I will analyze...
- [who is he?](https://www.perplexity.ai/search/who-is-he-nKR6DwraTAmY0HGdYCuHhg): I don't know who the person in the image is.

---

### **Perplexity AI ▷ #**[**sharing**](https://discord.com/channels/1047197230748151888/1054944216876331118/1250207739683147809) (5 messages):

- **Perplexity AI offers advantages over SearXNG**: A German-language thread discusses the specific advantages of [Perplexity AI over SearXNG](https://www.perplexity.ai/search/Welche-Vorteile-hat-wPaImsRVTdieJxldcEvbMw). The conversation centers on Perplexity AI's unique features and the potential benefits it brings to various professional fields.
- **Nightshade technology ethical concerns**: The [use of Nightshade](https://www.perplexity.ai/search/how-does-nightshade-uR1LfxUQQx60gkANQRIcsA) raises multiple ethical concerns such as potential misuse to sabotage AI models and the technical challenges in ensuring effective "poisoned" pixels. Adapting Nightshade to keep up with evolving AI technologies is increasingly difficult.
- **ICQ shuts down after 28 years**: ICQ, the notable instant messaging service, is [shutting down on June 26, 2024](https://www.perplexity.ai/page/ICQ-shuts-down-34n3T1XmQpuRpDB9VJcKVw) after nearly 28 years of operation. Known for its unique user IDs and "Uh oh!" message alerts, ICQ reached over 100 million users before its decline.
- **Perplexity AI shares industry updates**: A Perplexity AI summary video highlights various industry updates including [Raspberry Pi's London IPO](https://www.youtube.com/embed/-YNbmp8QBx8), Voodoo's BeReal Buyout, and Mistral AI's $6 billion surge. General Motors' Texas self-driving efforts were also mentioned.

**Links mentioned**:

- [how does nightshade work](https://www.perplexity.ai/search/how-does-nightshade-uR1LfxUQQx60gkANQRIcsA): Nightshade works by applying subtle pixel-level perturbations to images in a way that is imperceptible to humans but causes AI models to misclassify the...
- [ICQ Shuts Down After 28 Years](https://www.perplexity.ai/page/ICQ-shuts-down-34n3T1XmQpuRpDB9VJcKVw): ICQ, the pioneering instant messaging service that launched in 1996, is shutting down on June 26, 2024, after nearly 28 years of operation. The closure marks...
- [Perplexity](https://www.perplexity.ai/search/Welche-Vorteile-hat-wPaImsRVTdieJxldcEvbMw>)): Perplexity is a free AI-powered answer engine that provides accurate, trusted, and real-time answers to any question.
- [Perplexity](https://www.perplexity.ai/search/W): Perplexity is a free AI-powered answer engine that provides accurate, trusted, and real-time answers to any question.
- [What advantages does perplexity.ai have over SearXNG?](https://www.perplexity.ai/search/What-advantages-does-suq3APsvQRabk0eAIOjuUw#2): Based on the information provided in the search results, here are some of the key advantages that Perplexity AI has over SearXNG: 1. Conversational...

---

### **Perplexity AI ▷ #**[**pplx-api**](https://discord.com/channels/1047197230748151888/1161802929053909012/1250431495424966809) (2 messages):

- **Initial Setup Confirmation**: A user confirms that their setup appears to be functioning with the message *"This seems to work."*. This brief message implies successful initial configuration or troubleshooting.
- **Guidance to Insert API Key**: Another follow-up message instructs to *"add API key here"*. This step is crucial for the functioning of an API-based integration or service.

---

### **CUDA MODE ▷ #**[**general**](https://discord.com/channels/1189498204333543425/1189498205101109300/1250162673245622452) (29 messages🔥):

- **Affordable 3090 GPUs spark rig building plans**: Members discussed building GPU rigs, with one mentioning the affordability of **3090 GPUs** on **Kijiji Canada** at around 700-800 CAD, approximately 450 USD.
- **Power considerations for GPU rigs**: Concerns were raised about connecting GPU rigs to household power sockets. One member suggested that homes with 200A systems might handle it, but calculations would be necessary to confirm.
- **Contributing heat for community benefits**: Jokes and ideas circulated around using GPU rigs for non-traditional heating solutions, like replacing a hot tub heater with a few **4090s** or creating sustainable datacenters that repurpose waste heat.
- **Managing power draw for efficiency**: Members suggested limiting power draw via drivers to optimize flops/watts performance. Another member recommended using **MSI Afterburner** to limit core clock rates without affecting memory clocks.
- **Hypergraph neural networks for optimization**: A member asked if anyone had experience with hypergraphs, sharing a link to the [**Hypergraph Neural Network-Based Combinatorial Optimization GitHub repo**](https://github.com/nasheydari/HypOp). They were curious about its applications and effectiveness.

**Link mentioned**: [GitHub - nasheydari/HypOp: Hypergraph Neural Network-Based Combinatorial Optimization](https://github.com/nasheydari/HypOp): Hypergraph Neural Network-Based Combinatorial Optimization - nasheydari/HypOp

---

### **CUDA MODE ▷ #**[**torch**](https://discord.com/channels/1189498204333543425/1189607750876008468/1250167482283589694) (7 messages):

- **Torch caching validation woes**: A user asked how to validate if `TORCHINDUCTOR_CACHE_DIR` and `TORCHINDUCTOR_FX_GRAPH_CACHE` are working, and inquired about enabling any logs for validation.
- **Warm up for CUDA libraries**: Another user mentioned experiencing issues with CUDA and its libraries needing a warm-up to function correctly, elaborating that this is used internally by torch for algorithm decisions.
- **AWQ working group curiosity**: A user questioned why there wasn't a working group for AWQ, to which another responded that existing groups, such as those listed for [asymmetric linear quantization](https://discord.com/channels/1205223658021458100), cover similar ground.
- **Interest in 8k or 16k benchmarks**: There was interest in seeing benchmarks for 8k or 16k performance, linking to the [YaFSDP GitHub repository](https://github.com/yandex/YaFSDP).
- **C++ tensor iterators issue**: A member sought help with tensor iterators in C++ internals, specifically having trouble with making one work with NHWC input & output.

**Link mentioned**: [GitHub - yandex/YaFSDP: YaFSDP: Yet another Fully Sharded Data Parallel](https://github.com/yandex/YaFSDP): YaFSDP: Yet another Fully Sharded Data Parallel. Contribute to yandex/YaFSDP development by creating an account on GitHub.

---

### **CUDA MODE ▷ #**[**algorithms**](https://discord.com/channels/1189498204333543425/1189861061151690822/1250225367059529768) (2 messages):

- **Ternary Weights MatMul-Free Language Modeling Reported**: A new paper titled [Scalable MatMul-free Language Modeling](https://arxiv.org/abs/2406.02528) suggests that matrix multiplication can be eliminated from LLMs while still maintaining strong performance. It claims to reduce memory usage by up to 61% during training and narrows the performance gap with full precision Transformers as model size increases.
- **GitHub project for Matmul-free LM gains attention**: One member shared their positive experience with the [MatMul-free LM implementation](https://github.com/ridgerchu/matmulfreellm) available on GitHub. They praised the project's integration with transformers and the availability of models.

**Links mentioned**:

- [Scalable MatMul-free Language Modeling](https://arxiv.org/abs/2406.02528): Matrix multiplication (MatMul) typically dominates the overall computational cost of large language models (LLMs). This cost only grows as LLMs scale to larger embedding dimensions and context lengths...
- [GitHub - ridgerchu/matmulfreellm: Implementation for MatMul-free LM.](https://github.com/ridgerchu/matmulfreellm): Implementation for MatMul-free LM. Contribute to ridgerchu/matmulfreellm development by creating an account on GitHub.

---

### **CUDA MODE ▷ #**[**llmdotc**](https://discord.com/channels/1189498204333543425/1227345713348870156/1250196757963477103) (125 messages🔥🔥):

- **ThunderKitten integration tweaks improve performance**: *"My hacky copy-paste of ThunderKitten's '/attn_causal/h100_fwd.cu' into '/dev/cuda/attention_forward.cu' worked on the first try..."*. The code showed slight performance improvements over cuDNN with specific batch sizes, but challenges emerged with further modifications requiring more complex integrations.
- **100% determinism achieved but facing loss issues**: *"i got 100% determinism working! need to do a few more experiments with master weights and will push the changes."*. Despite achieving determinism, they faced issues with loss values, sometimes hitting -inf, hinting at bugs in functions like `prepare_softmax_blockwide3`.
- **FP32 version code and challenges**: Custom matmul implementation achieved about 70% speed of non-tensorcore cuBLAS, with dependencies on cuBLASLt noted. *"Since forward matmul is the only op where we use cublaslt, just this single custom implementation would be enough to remove that dependency..."*
- **FP8 implementation discussions**: Elaborations on requirements for reliable/stable FP8, such as unit scaling and stochastic rounding, with noted challenges in replacing cuBLAS/cuDNN. *"worth pointing out that (3) basically means fully replacing cuBLAS/cuDNN 😱"*
- **GPT-2 (774M) training insights and S3 links shared**: Detailed the training configuration of a 774M parameter model and noted potential overperformance due to extra tokens. Training results and possible implications of dataset overlaps were logged in [a discussion on GitHub](https://github.com/karpathy/llm.c/discussions/580).

**Links mentioned**:

- [mdouglas/llmc-gpt2-774M-150B · Hugging Face](https://huggingface.co/mdouglas/llmc-gpt2-774M-150B): no description found
- [GPT-2 (774M) reproduced · karpathy/llm.c · Discussion #580](https://github.com/karpathy/llm.c/discussions/580): I left the GPT-2 774M model running for ~6 days on my 8X A100 80GB node (150B tokens, 1.5 epochs over the 100B FineWeb sample dataset) and training just finished a few hours ago and went well with ...
- [no title found](https://us-west-2.console.aws.amazon.com/s3/object/llmc?region=us-west-2&bucketType=general&prefix=gpt2_774M/main.log): no description found
- [no title found](https://us-west-2.console.aws.amazon.com/s3/object/llmc?region=us-west-2&bucketType=general&prefix=gpt2_774M/model_00286102.bin): no description found
- [fp16 buffers for ADAM by ngc92 · Pull Request #289 · karpathy/llm.c](https://github.com/karpathy/llm.c/pull/289): First proof-of-concept implementation
- [no title found](https://us-west-2.console.aws.amazon.com/s3/object/llmc?region=us-west-2&bucke): no description found
- [no title found](http://llmc.s3-us-west-2.amazonaws.com/gpt2_774M/model_00286102.bin): no description found

---

### **CUDA MODE ▷ #**[**bitnet**](https://discord.com/channels/1189498204333543425/1240586843292958790/1250225001823600742) (7 messages):

- **Ternary weights in language modeling**: A member shared a paper on [Scalable MatMul-free Language Modeling](https://arxiv.org/pdf/2406.02528) using ternary weights.
- **Bit-packing CPU issues**: An issue was reported with bit-packing functions for **FP6-LLM** where the function fails if the data is on **CPU** without specifying the device. They suggested handling this better in the implementation.
- **Handling negative dimensions in bit-packing**: Another issue highlighted was that negative dimensions don't work with the current bit-packing functions. The recommendation was to either support negative dimensions or clearly note that they are not supported.
- **Quick PR for bit-packing fixes**: A developer mentioned they would address the issues with bit-packing for **FP6-LLM** and later confirmed that they had already made a quick PR to fix these problems.
- **Bit-packing tests for older PyTorch versions**: A member questioned why bit-packing tests were only run for PyTorch >= 2.4, noting that eager bit-packing tests should be possible with previous versions. It was acknowledged that **eager tests** don't necessitate the nightly version of PyTorch.

---

### **HuggingFace ▷ #**[**announcements**](https://discord.com/channels/879548962464493619/897387888663232554/1250188678266097666) (1 messages):

- **Whisper speeds up speech recognition in browsers**: [Whisper WebGPU](https://x.com/xenovacom/status/1799858691639796089) enables "blazingly-fast ML-powered speech recognition" in the browser, supporting multilingual transcription and translation across 100 languages, with no data leaving your device.
- **3D Arena Leaderboard launches**: The official [3D Arena Leaderboard](https://x.com/dylan_ebert_/status/1800333314965852518) from Hugging Face has launched, and the current leader is InstantMesh. Members are encouraged to participate and vote.
- **Gradio clients go stable**: Now you can [connect](https://x.com/evilpingwin/status/1799176924805333442) to any Gradio app and use it as an API with both JavaScript and Python clients, requiring just a few lines of code.
- **MaPO introduces memory efficiency**: [MaPO](https://x.com/RisingSayak/status/1800447427503362471) is a new, memory-efficient technique for aligning T2I diffusion models on preference data, eliminating the need for a reference model during alignment fine-tuning.
- **New batch of generative AI apps**: A second batch of generative AI apps is now [live](https://x.com/julien_c/status/1800153076994801929) on Hugging Face compatible model pages, expanding the family of supported applications.

**Links mentioned**:

- [Tweet from Xenova (@xenovacom)](https://x.com/xenovacom/status/1799858691639796089)): Introducing Whisper WebGPU: Blazingly-fast ML-powered speech recognition directly in your browser! 🚀 It supports multilingual transcription and translation across 100 languages! 🤯 The model runs lo...
- [Tweet from dylan (@dylan_ebert_)](https://x.com/dylan_ebert_/status/1800333314965852518)): ⚡️ 3D News ⚡️ The official @huggingface 3D Arena Leaderboard has launched 🚀 The #1 right now is InstantMesh @xinntao Go vote. https://huggingface.co/spaces/dylanebert/3d-arena
- [Tweet from amy (@a_e_roberts)](https://x.com/a_e_roberts/status/1800101457628377117)): You can now load in any pretrained Hugging Face backbone weights into your vision models in transformers! Just specify the HF checkpoint with `backbone=checkpoint` and `use_pretrained_backbone=True`
- [Tweet from pngwn (@evilpingwin)](https://x.com/evilpingwin/status/1799176924805333442)): Yesterday we announced the stable release of @Gradio clients in both JavaScript and Python! Now you can connect to any gradio app and use it as an API with just a few lines of code!
- [Tweet from Daniël de Kok (@danieldekok)](https://x.com/danieldekok/status/1799081814709133342)): 💎Coming to the next release of @huggingface TGI: support for Marlin. ⚡ Marlin is a highly-optimized kernel for FP16xINT4 matrix multiplication by @elias_frantar and @DAlistarh to speed up models wit...
- [Tweet from Sayak Paul (@RisingSayak)](https://x.com/RisingSayak/status/1800447427503362471)): Introducing MaPO, a memory-efficient technique for aligning T2I diffusion models on preference data 🔥 We eliminate the need to have a reference model when performing alignment fine-tuning. Code, mo...
- [Tweet from Vaibhav (VB) Srivastav (@reach_vb)](https://x.com/reach_vb/status/1799113953500672094)): PSA: You can load GGUFs directly in Transformers 🔥 Bonus: Convert them to transformers equivalent state dict too ⚡
- [Tweet from Thomas Wolf (@Thom_Wolf)](https://x.com/Thom_Wolf/status/1799008162772836355)): - full end-to-end policy - no teleoperation - quasi-full open-source software and hardware (even Reachy-1’s hardware is open-source CC-BY-SA CAO assembly)
- [Tweet from Julien Chaumond (@julien_c)](https://x.com/julien_c/status/1800153076994801929)): A second batch of local Generative AI apps are now live on compatible model pages on @huggingface 🔥 Welcome to the family, friends 🥰
- [Tweet from Victor M (@victormustar)](https://x.com/victormustar/status/1798994405522915621)): Crazy what HuggingChat can do these days!
- [Tweet from abhishek (@abhi1thakur)](https://x.com/abhi1thakur/status/1800511251145015393)): AutoTrain + Unsloth = 🚀🚀🚀 AutoTrain has now added support for unsloth which means you can use unsloth's optimizations to finetune LLMs super-fast and with much less memory 💥 And all you need t...
- [Tweet from merve (@mervenoyann)](https://x.com/mervenoyann/status/1800555650273132967)): releasing: smol vision 🌼 @huggingface A repository with notebooks on shrinking, optimizing, speeding-up, customizing large vision models!
- [Tweet from +RAIN Film Festival | 11-14 June 2024 (@rainfilmfest)](https://x.com/rainfilmfest/status/1799838451140870149)): L[AI]VE CINEMA at @rainfilmfest  A real-time cinema experience created by @radamar,  @multimodalart and powered by Hugging Face 🤗 in collaboration with CCCB @cececebe and Artefacto @artefactofilms  ...
- [Tweet from abhishek (@abhi1thakur)](https://x.com/abhi1thakur/status/1799029532332314852)): Multimodal Foundation Models Challenge is up! Win $10K in prizes 🎉 https://huggingface.co/spaces/ai-competition/MMFMChallenge

---

### **HuggingFace ▷ #**[**general**](https://discord.com/channels/879548962464493619/879548962464493622/1250166321765814302) (120 messages🔥🔥):

- **Trouble merging quantized adapter models**: A user asked how to merge a quantized adapter model back with the original, receiving guidance to use the `merge_and_unload` method. A code example was provided to help with implementation using the Trainer API.
- **RecurrentGemma 9B by Google announced**: A member shared a [post](https://x.com/osanseviero/status/1800607752038818260) revealing Google’s launch of RecurrentGemma 9B, emphasizing its **superior performance** in processing long sequences with great throughput and latency. Members responded excitedly about the model's capabilities.
- **Discussion on Stable Diffusion 3 Medium**: Multiple users discussed the release and features of [Stable Diffusion 3 Medium](https://huggingface.co/stabilityai/stable-diffusion-3-medium), noting its enhanced performance and complexities in image generation. Feedback included its current shortcomings, like anatomy handling, with suggestions for potential improvements.
- **Verification issues on Discord**: A user struggled with verification via LevelBot, receiving instructions to adjust their Discord settings to allow messages from server members. They were guided through the verification process using a token with read access from Hugging Face.
- **University of Glasgow Hugging Face organization**: An announcement was made about the creation of the [University of Glasgow organization](https://huggingface.co/UniversityofGlasgow) on Hugging Face, inviting faculty, researchers, and students to join using their university email addresses.

**Links mentioned**:

- [Tweet from Omar Sanseviero (@osanseviero)](https://x.com/osanseviero/status/1800607752038818260): RecurrentGemma 9B by Google is out 🔥 ⚡️Super fast for long sequences: Good throughput+latency 👀Base and instruct tuned versions 🏆Similar quality as Gemma Check the y-axis below 🤯 Models: https:...
- [Video LLaVA - a Hugging Face Space by LanguageBind](https://huggingface.co/spaces/LanguageBind/Video-LLaVA): no description found
- [UniversityofGlasgow (University of Glasgow)](https://huggingface.co/UniversityofGlasgow): no description found
- [Dolphin - a Hugging Face Space by nroggendorff](https://huggingface.co/spaces/nroggendorff/dolphin): no description found
- [stabilityai/stable-diffusion-3-medium-diffusers · Hugging Face](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers): no description found
- [stabilityai/stable-diffusion-3-medium · Hugging Face](https://huggingface.co/stabilityai/stable-diffusion-3-medium): no description found
- [Hugging Face – The AI community building the future.](https://huggingface.co/settings/tokens): no description found

---

### **HuggingFace ▷ #**[**today-im-learning**](https://discord.com/channels/879548962464493619/898619964095860757/1250499854074118164) (1 messages):

- **Join Live Event on Finance LLM Training**: A member invites others to a live event focused on training an **Expert Finance LLM**. The event features Mark Kim-Huang as a speaker and is free to attend. [Event Link](https://events.dataphoenix.info/live).

---

### **HuggingFace ▷ #**[**cool-finds**](https://discord.com/channels/879548962464493619/897390579145637909/1250311770573373471) (4 messages):

- **Old Guy with 1K Followers Puzzles Us**: An old guy got 1k followers, leaving everyone baffled. *"Here we are 🤷‍♂️"* reflects the sentiment.
- **Fine-Tuning Magic with Mistral**: Check out this guide on fine-tuning AI models on Mistral server. Dive deeper with [this Medium article](https://medium.com/ai-artistry/craft-your-ai-vision-fine-tuning-magic-with-mistral-on-mistral-server-6c9335232159).
- **Distill.pub Still Valuable for ML/DL Resources**: Although Distill.pub is no longer active, its articles remain useful for understanding ML and DL topics. Highlighted articles include [Understanding Convolutions on Graphs](https://github.com/distillpub/post--understanding-gnns/issues?q=is%3Aissue+label%3Apeer-review) and [A Gentle Introduction to Graph Neural Networks](https://github.com/distillpub/post--gnn-intro/issues?q=is%3Aissue+label%3Apeer-review).
- **Consider Supporting Ciechanowski’s Quality Content**: For more interactive and high-quality content, visit [Ciechanowski’s archives](https://ciechanow.ski/archives/) on various engineering topics. If you enjoy these articles, you can support the creator on [Patreon](https://www.patreon.com/ciechanowski).

**Links mentioned**:

- [Distill — Latest articles about machine learning](https://distill.pub/): Articles about Machine Learning
- [Archives - Bartosz Ciechanowski](https://ciechanow.ski/archives/): no description found

---

### **HuggingFace ▷ #**[**i-made-this**](https://discord.com/channels/879548962464493619/897390720388825149/1250174120524513482) (4 messages):

- **Huge Dalle 3 Dataset Released**: A member shared a dataset of over 1 million high-quality captioned images generated with Dalle 3, emphasizing the human-driven creativity involved due to Dalle 3’s unpredictable censorship. The dataset covers diverse topics like art styles, landscapes, themes, holidays, and pop culture. [Dataset Card for Dalle3 1 Million+ High Quality Captions](https://huggingface.co/datasets/ProGamerGov/synthetic-dataset-1m-dalle3-high-quality-captions).
- **SimCLR Models Hit the Hub**: Another member converted and uploaded PyTorch weights for SimCLRv1 and SimCLRv2 models, which were previously unavailable on the Hub. [SimCLRv1 PyTorch Weights](https://huggingface.co/collections/SauravMaheshkar/simclrv1-pytorch-weights-6668b84a2bd3135cc3283fcd) and [SimCLRv2 PyTorch Weights](https://huggingface.co/collections/SauravMaheshkar/simclrv2-pytorch-weights-6668cd94d7c3f6fe7f3ac14b).
- **Mixins Make Model Management Simple**: The versatility of the `huggingface_hub.PyTorchModelHubMixin` for uploading and downloading functions was highlighted. A link to the [Mixins documentation](https://huggingface.co/docs/huggingface_hub/en/package_reference/mixins#huggingface_hub.PyTorchModelHubMixin) was provided.
- **Stable Diffusion 3 Medium Weights on a Space**: A member created a space for **Stable Diffusion 3 Medium Weights**, aimed at enabling rapid model use and deployment. Check out the project [here](https://huggingface.co/spaces/Nick088/Stable-Diffusion-3-Medium).

**Links mentioned**:

- [ProGamerGov/synthetic-dataset-1m-dalle3-high-quality-captions · Datasets at Hugging Face](https://huggingface.co/datasets/ProGamerGov/synthetic-dataset-1m-dalle3-high-quality-captions): no description found
- [Stable Diffusion 3 Medium Superpompt - a Hugging Face Space by Nick088](https://huggingface.co/spaces/Nick088/Stable-Diffusion-3-Medium): no description found
- [SimCLRv1 PyTorch Weights - a SauravMaheshkar Collection](https://huggingface.co/collections/SauravMaheshkar/simclrv1-pytorch-weights-6668b84a2bd3135cc3283fcd): no description found
- [SimCLRv2 PyTorch Weights - a SauravMaheshkar Collection](https://huggingface.co/collections/SauravMaheshkar/simclrv2-pytorch-weights-6668cd94d7c3f6fe7f3ac14b): no description found
- [Mixins & serialization methods](https://huggingface.co/docs/huggingface_hub/en/package_reference/mixins#huggingface_hub.PyTorchModelHubMixin): no description found

---

### **HuggingFace ▷ #**[**core-announcements**](https://discord.com/channels/879548962464493619/1014557141132132392/1250473835309039749) (1 messages):

- **Diffusers Integration for SD3 Coming Soon**: *"Hey folks,* `diffusers` integration for SD3 will be up in a few hours. Thanks for your patience in advance." Exciting integration announcement!

---

### **HuggingFace ▷ #**[**computer-vision**](https://discord.com/channels/879548962464493619/922424143113232404/1250255554132185139) (3 messages):

- **Semantic search appreciated in the space**: A user expressed gratitude, saying *"This is great... especially with the semantic search. Thanks for creating this space."*
- **Request for computer vision project ideas**: Another user asked for suggestions on *"awesome project ideas in the computer vision domain."*
- **CCTV smoking detection project suggested**: A response to the project idea request suggested creating a system for *"detection of a man smoking in CCTV surveillance"* with a feature to make the bounding box red if the confidence is greater than 0.9.

---

### **HuggingFace ▷ #**[**NLP**](https://discord.com/channels/879548962464493619/922424173916196955/1250298744587681822) (2 messages):

- **Request for Papers on Fine-Tuning Techniques**: One user asked for recommendations on papers to review before "diving deep into finetuning" models like **phi2**, **llama2**, and **mixtral2**. They also requested the names of any "interesting scripts."
- **Gemini Flash Release Request on GitHub**: A member shared a [GitHub issue on gemma.cpp](https://github.com/google/gemma.cpp/issues/221), expressing strong interest in seeing **Google Gemini Flash** released to the open-source community. The linked issue is addressed to the Google AI team, highlighting community interest in the project.

**Link mentioned**: [OFF Topic, Request for Open-Sourcing Google Gemini Flash · Issue #221 · google/gemma.cpp](https://github.com/google/gemma.cpp/issues/221): Dear Google AI Team, I wish to express my strong interest in seeing Google Gemini Flash released to the open-source community. As a developer and AI enthusiast, I have been incredibly impressed wit...

---

### **HuggingFace ▷ #**[**diffusion-discussions**](https://discord.com/channels/879548962464493619/1009713274113245215/1250205986795552809) (8 messages🔥):

- **Simpletuner recommended for diffusion models**: One member suggested that **simpletuner** is primarily focused on training diffusion models and mentioned that **diffusers examples** are good for basic ideas but not fully optimized. It was noted that adjustments would likely need to be made to fit specific use cases.
- **Discussion on Qualcomm's AIMET**: One member mentioned **AIMET** from Qualcomm as the worst library they have used, leading to a brief off-topic discussion. Another member acknowledged having similar experiences with other Qualcomm SDKs.
- **Exploring AI quantization tools**: A member shared their struggles with using **quanto** and **optimum** for quantization and export, mentioning errors and disappointing results. They had also tried an on-device AI course with no success.
- **Query about model weight extraction**: A member asked if there is a way to extract weights from a given model to perform aggregation. No responses to this query were visible in the provided messages.

---

### **OpenAI ▷ #**[**ai-discussions**](https://discord.com/channels/974519864045756446/998381918976479273/1250192667577684019) (111 messages🔥🔥):

- **Ilya Sutskaver's lecture recommended**: Ilya Sutskaver had an "amazing lecture at Simons Institute," available on YouTube under the title *An Observation on Generalization*. Another member suggested listening to Neel Nanda's thoughts on memorization vs generalization in the YouTube video *Mechanistic Interpretability - NEEL NANDA (DeepMind)*.
- **Differences in AI model performance discussed**: Members compared Llama 3 8b instruct with GPT 3.5, noting Llama 3 8b's free API on Hugging Face. There was also a debate on the merits of GPT-4o's coding abilities, with some observing performance issues.
- **Enterprise tier debated**: Extensive discussion on the merits and cost of the GPT Enterprise tier, with some users praising its benefits like a larger context window and seamless conversation continuation. It was noted that Enterprise tier offers more power and eliminates limits, but one user mistakenly equated Teams with Enterprise.
- **Advice on building a model from scratch**: When asked about building a smaller GPT-like model focused on a single field, it was suggested to either finetune an existing model like Llama3 8b or use open-source alternatives rather than starting from scratch.
- **Embeddings for medical documents**: A member inquired about a good open-source embedding model for medical document search, seeking recommendations for sentence and keyword search optimization.

---

### **OpenAI ▷ #**[**gpt-4-discussions**](https://discord.com/channels/974519864045756446/1001151820170801244/1250191528794652713) (9 messages🔥):

- **Custom GPT creation inquiries**: A user asked if this channel is appropriate *"to ask someone to create a custom GPT"*. Another user humorously pondered if it's a good place to showcase a GPT.
- **Error with generating responses**: A member consistently experienced the error message, *"Something went wrong while generating the response..."*, and sought help to resolve it.
- **Issues with uploading PHP files**: There were issues with uploading a PHP file to Assistants Playground despite the [file type being listed as supported](https://platform.openai.com/docs/assistants/tools/file-search/supported-files). The user also asked if there is an upload size limit before stating they managed to upload it in storage but still faced retrieval issues.

---

### **OpenAI ▷ #**[**prompt-engineering**](https://discord.com/channels/974519864045756446/1046317269069864970/1250380993668448297) (11 messages🔥):

- **How to deactivate citations in GPT-4 assistants**: A user asked for help on how to deactivate and avoid citations appearing as [1][2] in a GPT-4 assistant trained on over 20 PDFs. They mentioned they still want the retrieval of information without specifying the source.
- **Chatbot isn't aware of gizmo tool functionality**: When asked whether they tried asking ChatGPT, a user clarified that ChatGPT isn't aware of the gizmo tool's functionality.
- **Documentation troubles**: A user shared that providing API documentation in a text file of 300 MB might be overwhelming. Another user suggested creating a swagger file to manage it better.
- **Request for resume creation help**: A user expressed a need for assistance in creating a resume.

---

### **OpenAI ▷ #**[**api-discussions**](https://discord.com/channels/974519864045756446/1046317269069864970/1250380993668448297) (11 messages🔥):

- **Deactivate GPT4 citations, still retrieve data**: A member asked how to deactivate and avoid citations like `[1]`, `[2]` in a GPT4 assistant trained on 20+ PDFs. They still want the assistant to retrieve information but don't care about the specific source.
- **GPT unaware of gizmo tool functionality**: Another member suggested asking ChatGPT about an issue, but the original inquirer noted that ChatGPT isn’t aware of the functionality of the gizmo tool and hence isn't helpful in this context.
- **300MB txt file documentation issue**: A user pointed out the difficulty of providing documentation as a text file because it’s too large (300MB). There was a suggestion to write a Swagger file instead, but the user clarified it’s not just an API but a whole documentation.

---

### **Cohere ▷ #**[**general**](https://discord.com/channels/954421988141711382/954421988783444043/1250245729767788624) (126 messages🔥🔥):

- **Disappointment with Qualcomm's AIMET Library**: A member expressed frustration, saying *"For me this is the worst library I have seen till now."*
- **PaidTabs and Cohere AI Integration Speculation**: Members discussed whether PaidTabs is using Cohere for AI message generation, referencing [PaidTabs' June 24 changelog](https://blog.paidtabs.com/paidtabs-june-24-changelog/). The conversation clarified that *"cohere isn't doing anything audio related to my knowledge. YET at least."*
- **Music Enthusiasts in the Channel**: Several members discussed their musical interests and activities, with some experimenting with making music via software and others sharing personal experiences and setups. One member joked, *"With all the musicians here, I think we could start a band."*
- **Joystick and Flight Sim Setup**: An extensive discussion about a member's elaborate joystick setup for space flight simulations, which includes the [VPC Constellation ALPHA Prime joysticks](https://virpil-controls.us.com/vpc-constellation-alpha-prime-r.html). The setup cost was debated, with one noting, *"Sticks made of diamond or what? joysticks cost 1.5k?"*
- **Request for Offtopic Channel**: As the conversation became more off-topic, one member jokingly suggested creating a separate channel for such discussions to avoid spamming the main channel. *"we need an off-topic channel... I'm spamming."*

**Links mentioned**:

- [PaidTabs New Tuner, Mobile App and More Features!](https://blog.paidtabs.com/paidtabs-june-24-changelog/): Smart Tuner We’re happy to launch our long-awaited stringed tuner powered by AI, which detects the strings you’re playing and their tuning to help you play a perfect note! Check it out here. Commer...
- [VPC Constellation ALPHA Prime [R]](https://virpil-controls.us.com/vpc-constellation-alpha-prime-r.html): ★ \* RIGHT HAND VARIANT \* ★ Metal Flip Trigger and Brake Lever ★ Premium...
- [Udio | AI Music Generator - Official Website](https://www.udio.com/): Discover, create, and share music with the world. Use the latest technology to create AI music in seconds.
- [Star Citizen - Roberts Space Industries | Follow the development of Star Citizen and Squadron 42](https://robertsspaceindustries.com/star-citizen) : Roberts Space Industries is the official go-to website for all news about Star Citizen and Squadron 42. It also hosts the online store for game items and merch, as well as all the community tools use...

---

### **Cohere ▷ #**[**project-sharing**](https://discord.com/channels/954421988141711382/1218409701339828245/1250476945821532195) (1 messages):

- **Playgrounds releases RIG for Rust:** The founder and CEO of Playgrounds announced the release of an open-source library named **RIG** for building **LLM-powered applications in Rust**. The library emphasizes modularity and ergonomics, and it is fully integrated with Cohere out of the box. Check the [RIG repository](https://github.com/0xPlaygrounds/rig) on GitHub for further details and examples.

**Link mentioned**: [GitHub - 0xPlaygrounds/rig: A library for developing LLM-powered Rust applications.](https://github.com/0xPlaygrounds/rig): A library for developing LLM-powered Rust applications. - 0xPlaygrounds/rig

---

### **Eleuther ▷ #**[**general**](https://discord.com/channels/729741769192767510/729741769738158194/1250198644343312424) (22 messages🔥):

- **New Member Asks for Introduction**: A new member inquired about the purpose of the community and how to contribute. Other members recommended reading pinned messages and checking specific channels like `<#732688974337933322>` and `<#1102787157866852402>` for resources and ongoing projects.
- **Predicting Microplastic Concentrations**: One user explained their project involving the combination of oceanic vector factors like algae concentration and water velocities to predict microplastic concentrations and movements.
- **Autoregressive Model Beats Diffusion for Images**: A link to [LlamaGen's repo](https://github.com/FoundationVision/LlamaGen) was shared, showcasing that autoregressive models like Llama can achieve state-of-the-art image generation performance. The repo contains details and tutorials about LlamaGen, a new family of image generation models.
- **Debate on DPO Methods for Multi-turn Conversations**: Multiple members discussed the lack of research on DPO (Deterministic Policy Optimization) for multi-turn conversations. Some suggested checking the Ultrafeedback dataset and considering the MCTS-DPO approach for hierarchical structures.
- **Llama3 Model Generations**: A user shared their experience with the Llama3 70B model generating diverse outputs like AI2's ARC format evaluations, wiki text, and code from an empty prompt. Another member noted a 30% chance it starts with a "Question" format in its outputs.

**Link mentioned**: [FoundationVision/LlamaGen · Hugging Face](https://huggingface.co/FoundationVision/LlamaGen): no description found

---

### **Eleuther ▷ #**[**research**](https://discord.com/channels/729741769192767510/747850033994662000/1250166991826845867) (94 messages🔥🔥):

- **Trouble finding coding datasets for LLM training**: A user is looking for a small coding dataset with less than 10B tokens but notes difficulty in finding such a dataset. They mention potentially using a GPT-2 tokenizer and discuss the availability of 200B tokens divided into chunks excluding GitHub data.
- **Precision limits in transformers**: A discussion unfolds about the limitations due to finite precision in transformers, referencing an [arXiv paper](https://arxiv.org/abs/2210.02671). Precision issues are highlighted in the attention mechanism, specifically the softmax operation.
- **DeltaNet vs RWKV6 Finch architecture similarities**: Users debate the similarities between DeltaNet and RWKV6 Finch architecture, focusing on the "decay" mechanism and the token shift mechanism. It's concluded that DeltaNet's decay is dependent on the state rather than the input.
- **Novelty and relativity of research papers**: Multiple users critique new papers on architectures like Griffin and Samba, comparing them to existing works and expressing doubts about their novelty. They discuss Samba's performance in modeling long contexts and its architectural combination of state space models and sliding window attention.
- **Exploring benchmarks for long-context retrieval**: The conversation touches on Samba's ability to efficiently retrieve context over long sequences, with users showing skepticism and thinking that the benchmark tasks might not be robust enough. One user points out the potential role of positional encoding in performance on long-context tasks.

**Links mentioned**:

- [Simple and Effective Masked Diffusion Language Models](https://arxiv.org/abs/2406.07524): While diffusion models excel at generating high-quality images, prior work reports a significant performance gap between diffusion and autoregressive (AR) methods in language modeling. In this work, w...
- [Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling](https://arxiv.org/abs/2406.07522): Efficiently modeling sequences with infinite context length has been a long-standing problem. Past works suffer from either the quadratic computation complexity or the limited extrapolation ability on...
- [LLM Dataset Inference: Did you train on my dataset?](https://arxiv.org/abs/2406.06443): The proliferation of large language models (LLMs) in the real world has come with a rise in copyright cases against companies for training their models on unlicensed data from the internet. Recent wor...
- [A Logic for Expressing Log-Precision Transformers](https://arxiv.org/abs/2210.02671): One way to interpret the reasoning power of transformer-based language models is to describe the types of logical rules they can resolve over some input text. Recently, Chiang et al. (2023) showed tha...
- [Improve Mathematical Reasoning in Language Models by Automated Process Supervision](https://arxiv.org/abs/2406.06592): Complex multi-step reasoning tasks, such as solving mathematical problems or generating code, remain a significant hurdle for even the most advanced large language models (LLMs). Verifying LLM outputs...
- [Is a small transformer model able to effectively handle any input length provided it is fine-tuned on it?](https://ai.stackexchange.com/q/45949/68078): Suppose we have a transformer LLM which can do a task such as summarising. I know transformer can technically handle any input length (assume we are not using learned positional embeddings) becaus...
- [Tweet from Rivers Have Wings (@RiversHaveWings)](https://x.com/RiversHaveWings/status/1478093658716966912): You can apply a similar trick to classifier-free guidance to autoregressive transformers to sample from a synthetic "super-conditioned" distribution. I trained a CIFAR-10 class-conditional Ima...
- [Cache Me if You Can: Accelerating Diffusion Models through Block Caching](https://arxiv.org/abs/2312.03209): Diffusion models have recently revolutionized the field of image synthesis due to their ability to generate photorealistic images. However, one of the major drawbacks of diffusion models is that the i...
- [GitHub - nasheydari/HypOp: Hypergraph Neural Network-Based Combinatorial Optimization](https://github.com/nasheydari/HypOp): Hypergraph Neural Network-Based Combinatorial Optimization - nasheydari/HypOp
- [GitHub - microsoft/Samba](https://github.com/microsoft/Samba/): Contribute to microsoft/Samba development by creating an account on GitHub.

---

### **Eleuther ▷ #**[**scaling-laws**](https://discord.com/channels/729741769192767510/785968841301426216/1250185388937449583) (2 messages):

- **Transformers excel at compositional generalization**: A user shared an [arXiv paper](https://arxiv.org/abs/2406.05816) that discusses how Transformers *"generalize to novel problem instances"* by utilizing a low-dimensional latent code in their multi-head attention mechanism. The method improves compositional generalization, especially on tasks like a symbolic version of the Raven Progressive Matrices human intelligence test.

**Link mentioned**: [Attention as a Hypernetwork](https://arxiv.org/abs/2406.05816): Transformers can under some circumstances generalize to novel problem instances whose constituent parts might have been encountered during training but whose compositions have not. What mechanisms und...

---

### **Eleuther ▷ #**[**lm-thunderdome**](https://discord.com/channels/729741769192767510/755950983669874798/1250193482962833448) (3 messages):

- **Contamination Detection Suggestion:** A member shared their recent work on [contamination detection](https://arxiv.org/abs/2405.16281), which compares model performances on two benchmarks to detect 'suspicious' scores. They offered to create a PR to include this feature in the evaluation harness.
- **Critical One-Liner Fix:** A member urged others to review a [one-liner fix](https://github.com/EleutherAI/lm-evaluation-harness/pull/1848) for a bug in `anthropic_llms.py` where `self.max_tokens` was not properly set in the constructor. They emphasized the frustration of a single missing "s" causing issues.
- **Immediate Code Review:** Another member promptly responded, promising to review the proposed fix shortly.

**Link mentioned**: [Fix self.max_tokens in anthropic_llms.py by lozhn · Pull Request #1848 · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/pull/1848): Fix bug where self.max_tokens was not set. The AnthropicChatLM uses everywhere self.max_tokens while in constructor is sets self.max_token. It doesn't call issues on smaller responses but I caught...

---

### **Eleuther ▷ #**[**multimodal-general**](https://discord.com/channels/729741769192767510/795089627089862656/1250174056477491361) (3 messages):

- **Seeking LTIP Alternatives for Pre-training**: A member asked for an open-source alternative to the LTIP dataset used by DeepMind, expressing interest in dataset size comparable to LTIP's 312M image-text pairs. They noted the datasheet for LTIP can be found in the [Flamingo paper](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/tackling-multiple-tasks-with-a-single-visual-language-model/flamingo.pdf).
- **DataComp Recommended as LTIP Alternative**: In response, another member suggested the 1B subset of DataComp as a viable alternative, claiming it is better than LAION for CLIP models. They shared a link to the [DataComp dataset](https://huggingface.co/datasets/mlfoundations/datacomp_1b).

**Link mentioned**: [mlfoundations/datacomp_1b · Datasets at Hugging Face](https://huggingface.co/datasets/mlfoundations/datacomp_1b): no description found

---

### **LM Studio ▷ #**[**💬-general**](https://discord.com/channels/1110598183144399058/1110598183144399061/1250180925422043177) (85 messages🔥🔥):

- **Python Script Integration Inquiry**: A user expressed interest in integrating a Python script into LM Studio for a chatbot to roleplay and send messages if inactive for over a minute. They were directed to set up the inference server following the [LM Studio documentation](https://lmstudio.ai/docs).
- **RecurrentGemma Model Request**: There was a request to add the **RecurrentGemma-9B** model, with the link provided to its [Hugging Face page](https://huggingface.co/google/recurrentgemma-9b) for more details and documentation.
- **Issues with LM Studio 0.2.24**: A user reported severe bugs in LM Studio version 0.2.24, including issues with the token counter, model unloading and reloading, and GPU optimization. They expressed dissatisfaction, noting that LM Studio suddenly feels very "alpha."
- **GPU and Model Inference Discussion**: Discussions involved comparisons between GPUs for model inference, with some suggesting the **RTX 3060** over the **Tesla P40** for better performance and power efficiency. Another user aimed to improve processing time from 9 hours to 2 by potentially upgrading their hardware from a Radeon 6900XT to a 24GB Nvidia card.
- **Generating PowerPoint with Chatbots**: A user inquired about creating chatbots for generating PowerPoint presentations using the OpenAI Assistant API, looking for tools to automate the process based on previous presentations.

**Links mentioned**:

- [Documentation | LM Studio](https://lmstudio.ai/docs): Technical Reference
- [How to Finetune phi-3 on MacBook Pro](https://huggingface.co/blog/abhishek/phi3-finetune-macbook): no description found
- [google/recurrentgemma-9b · Hugging Face](https://huggingface.co/google/recurrentgemma-9b): no description found
- [All Large Language Models](https://llm.extractum.io/list/): A Curated List of the Large and Small Language Models (Open-Source LLMs and SLMs). All Large Language Models with Dynamic Sorting and Filtering.
- [Nitral-AI/Hathor-L3-8B-v.02 · Hugging Face](https://huggingface.co/Nitral-AI/Hathor-L3-8B-v.02): no description found
- [Nitral-AI/Hathor-L3-8B-v.02 · Great Uncensored Model](https://huggingface.co/Nitral-AI/Hathor-L3-8B-v.02/discussions/1): no description found
- [GitHub - jakobdylanc/discord-llm-chatbot: llmcord.py • Talk to LLMs with your friends!](https://github.com/jakobdylanc/discord-llm-chatbot): llmcord.py • Talk to LLMs with your friends! Contribute to jakobdylanc/discord-llm-chatbot development by creating an account on GitHub.

---

### **LM Studio ▷ #**[**🤖-models-discussion-chat**](https://discord.com/channels/1110598183144399058/1111649100518133842/1250311559025004604) (28 messages🔥):

- **Choosing the fastest model for RTX 2070**: A user inquired about the fastest model that could run on an RTX2070 with 64GB RAM and **8GB of VRAM**. The discussion narrowed down to several models including **Mistral7b**, **Codestral**, and **StarCoder2**, with considerations given to context window sizes and the need to keep everything on GPU.
- **Recommendations for Coding Models**: **Codestral** was recommended for those who can afford it, otherwise, coding-tuned Mistral or Llama models were suggested. **StarCoder2** and **StarChat** were highlighted as balanced options with good performance, and **CodeQwen1.5** was also mentioned as a decent choice.
- **Inquiry on Enabling Training Mode**: A user asked about enabling a training mode for a personal model fork to remember things across conversations. The reply indicated that LM Studio does not support this but recommended **Unsloth** and **Autotrain** for fine-tuning existing safetensor models.
- **Persistent Storage with VectorDB**: There was mention of using **vectordb plugins/extensions** for persistent storage to remember specific things across conversations. The user considered using a programming approach to achieve their desired functionality of consistent, memory-retentive conversation without the manual input of directives.
- **Clarification on Model Adaptation**: Discussion around the possibility of ongoing training or weight adjustments for models in different formats, specifically **gguf**. The user expressed interest in artificially weighting the model's responses to ensure certain outcomes in a conversational context.

---

### **LM Studio ▷ #**[**📝-prompts-discussion-chat**](https://discord.com/channels/1110598183144399058/1120489168687087708/1250511503120142459) (3 messages):

- **Implementing GIT as a Sys Prompt Addition**: A member shared their progress on integrating GIT into a sys prompt, stating it "does the basics after a day or so." They pondered whether this effort might be a waste of time.
- **Generative Reliability Concerns**: Although the GIT implementation isn't perfectly reliable due to its generative nature, the member noted it has "been working pretty well."
- **Humorous Product Naming**: In a light-hearted comment, the member suggested calling the implementation "Timemachine" in a nod to Apple's branding and humorously priced it at "$9 ? xD."

---

### **LM Studio ▷ #**[**🎛-hardware-discussion**](https://discord.com/channels/1110598183144399058/1153759714082033735/1250254102940881002) (4 messages):

- **P40s Pricing Drama**: Members discussed the varying prices of **P40 graphics cards**, with one citing a pair purchased overseas for $640 (48GB total). Another linked an **eBay listing** showing prices around **$150-170 USD** per card currently.
- **GPU Availability and Costs in AU**: One member shared the **local Australian pricing** for various GPUs—**RTX 3060** at $450, **RTX 4060 TI** at $800, and **RTX 4090** at $3200, noting differences in stock levels and recent price changes. Disclaimers include power consumption, PSU considerations, cooling challenges, and other trade-offs.
- **Consistent Seller Listing**: It was noted that the same eBay seller offers **P40 cards** on different international platforms, with slight price variations over time.
- **US-based P40 Purchases**: Another member stated they recently bought P40 cards delivered within the US for **$160-170 each**, emphasizing the difference from prices of cards shipped from China.

**Link mentioned**: [NVIDIA Tesla P40 24GB DDR5 GPU Accelerator Card Dual PCI-E 3.0 x16 - PERFECT! 190017118253 | eBay](https://www.ebay.com.au/itm/196435922621): no description found

---

### **LM Studio ▷ #**[**amd-rocm-tech-preview**](https://discord.com/channels/1110598183144399058/1195858490338594866/1250230075350188072) (3 messages):

- **Modern UI enabled by default**: A member noted that there should now be a "modern UI" that users can enable, stating it "seems quite different". However, they haven't used it enough yet to fully assess.
- **New UI appearance preview**: An image showcasing the modern UI variations was shared, and you can [view the image here](https://cdn.discordapp.com/attachments/1130536562422186044/1245478003044257932/image.png?ex=666a08c7&is=6668b747&hm=62024d42535148d6a9a7f23860dc0c011c6a2a48dba4759c91e04bb0f2fbe03f). One member responded positively, stating, "Now that's pretty good."

---

### **LlamaIndex ▷ #**[**announcements**](https://discord.com/channels/1059199217496772688/1073670729054294197/1250214985762603079) (1 messages):

- **RAG in Enterprise Setting Faces Challenges**: An announcement invites those involved in **building/productionizing RAG** (Retrieval-Augmented Generation) in enterprise settings to check out Quentin's message. Quentin and Shreya Shankar are conducting interviews to understand the challenges faced, and interested parties are encouraged to connect through the [provided Discord link](https://discord.com/channels/1059199217496772688/1059201661417037995/1249840705627488408).

---

### **LlamaIndex ▷ #**[**blog**](https://discord.com/channels/1059199217496772688/1187460979064324127/1250179545844285602) (2 messages):

- **Graph RAG with Entity Deduplication**: A detailed walkthrough posted by @tb_tomaz demonstrates how to *supercharge* Graph RAG using entity deduplication and custom retrieval methods in LlamaIndex. Check out the full tutorial [here](https://t.co/fiFwDQS6WT).
- **Excel-Formatted RAG Challenges**: Discussing the challenges of building RAG that works effectively over Excel files, the importance of laying out content in a well-formatted spatial grid is highlighted. More details are available in the tweet [here](https://t.co/vbdk8Yuw2t).

---

### **LlamaIndex ▷ #**[**general**](https://discord.com/channels/1059199217496772688/1059201661417037995/1250169658670190653) (71 messages🔥🔥):

- **Chat Engine Development Using Vector Index**: A member asked for help on developing a chat engine using the context of a vector index created from a plain text snippet. Another member suggested creating documents/nodes manually using the `Document` class.
- **S3DirectoryReader Issues**: A member reported getting encrypted files when using S3DirectoryReader but noted that Langchain S3 Reader worked correctly. They confirmed SSE-S3 encryption was enabled but not automatically decrypting.
- **MetadataFilter Not Functioning Correctly**: A member shared issues with `MetadataFilter` not returning expected results in their Redis index setup. Another member requested details on how the index was built to diagnose the problem.
- **Markdown Formatting in Vector DB**: A user experienced issues with markdown formatting persisting in stored documents and chat responses. Another member suggested extracting plain text from markdown documents as a pre-processing step using BeautifulSoup.
- **Customizing Chat Engine Prompts**: Guidance was provided on customizing the prompts for `CondensePlusContextChatEngine` using the `from_defaults` method. Follow-up clarifications explained that users don’t need to handle special tokens like EOT manually.

**Links mentioned**:

- [Python : How to convert markdown formatted text to text](https://stackoverflow.com/a/761847/1819550): I need to convert markdown text to plain text format to display summary in my website. I want the code in python.
- [llama_index/llama-index-core/llama_index/core/vector_stores/simple.py at 045582caf3564f5a549266c07d291f30b997425d · run-llama/llama_index](https://github.com/run-llama/llama_index/blob/045582caf3564f5a549266c07d291f30b997425d/llama-index-core/llama_index/core/vector_stores/simple.py#L368): LlamaIndex is a data framework for your LLM applications - run-llama/llama_index
- [Condense plus context - LlamaIndex](https://docs.llamaindex.ai/en/latest/api_reference/chat_engines/condense_plus_context/#llama_index.core.chat_engine.CondensePlusContextChatEngine>).): no description found
- [Function Calling AWS Bedrock Converse Agent - LlamaIndex](https://docs.llamaindex.ai/en/latest/examples/agent/bedrock_converse_agent/?h=converse): no description found
- [Query Pipeline for Advanced Text-to-SQL - LlamaIndex](https://docs.llamaindex.ai/en/stable/examples/pipeline/query_pipeline_sql/): no description found
- [Building Performant RAG Applications for Production - LlamaIndex](https://docs.llamaindex.ai/en/latest/optimizing/production_rag/#motivation_1>),): no description found
- [Chat Engine - Best Mode - LlamaIndex](https://docs.llamaindex.ai/en/latest/examples/chat_engine/chat_engine_best/#chat-engine-best-mode>)): no description found
- [Building Performant RAG Applications for Production - LlamaIndex](https://docs.llamaindex.ai/en/latest/optimizing/production_rag/#motivation_1>).): no description found
- [Building a Multi-PDF Agent using Query Pipelines and HyDE - LlamaIndex](https://docs.llamaindex.ai/en/latest/examples/agent/agent_runner/agent_around_query_pipeline_with_HyDE_for_PDFs/#what-is-react-agent>)): no description found
- [React - LlamaIndex](https://docs.llamaindex.ai/en/latest/api_reference/agent/react/#llama_index.core.agent.react.ReActAgent>).): no description found

---

### **LlamaIndex ▷ #**[**ai-discussion**](https://discord.com/channels/1059199217496772688/1100478495295017063/1250438272329580645) (1 messages):

- **Query engine struggles with context awareness**: A member expressed difficulties with a text-to-SQL query engine that fails to recognize specific context clues such as the nature of a queried item (e.g., whether "Q" is a product name or something else). They highlighted the problem as the engine's inability to discern the correct SQL query due to ambiguity about the data values and requested advice on resolving this issue.

---

### **Interconnects (Nathan Lambert) ▷ #**[**news**](https://discord.com/channels/1179127597926469703/1179128538679488533/1250198856592003153) (41 messages🔥):

- **ARC Prize for AGI Announced**: A substantial **$1,000,000+ competition** has been unveiled to solve and open source a solution to the ARC-AGI benchmark, hosted by Mike Knoop and François Chollet. Chollet's **ARC-AGI** measures general intelligence by the ability to acquire new skills efficiently and has been difficult for AI but easy for humans.
- **Mixed Reactions to Podcast Interview**: Members had mixed feelings about a recent interview, particularly criticizing the host, Dwarkesh, for not responding precisely to interviewee François' points. Despite some frustration, the interview was still enjoyed by listeners like chygao, who acknowledged the justified skepticism raised.
- **Industry Unawareness on ARC-AGI**: There was surprise that **industry peers do not know about the ARC-AGI benchmark**, which was deemed foundational by some members. Drj.bet was particularly taken aback when this was mentioned at the end of the podcast.
- **TechCrunch Critique**: A TechCrunch article by Kyle L. Wiggers received criticism for lacking depth on the subject of AI. Natolambert humorously commented on the repetitive nature of some tech writing, mocking Wiggers' previous piece about DBRX.
- **Discussion on Human Intelligence and Genetics**: The podcast also touched on human intelligence's dependence on genetic pedigree, compared by Dwarkesh to model initialization. This view was contested by members like vj256, who cited numerous counterexamples of successful individuals from diverse backgrounds.

**Links mentioned**:

- [ARC Prize](https://arcprize.org/): ARC Prize is a $1,000,000+ nonprofit, public competition to beat and open source a solution to the ARC-AGI benchmark.
- [Tweet from Kyle Wiggers (@Kyle_L_Wiggers)](https://x.com/Kyle_L_Wiggers/status/1800945959112749155): This Week in AI: Apple won’t say how the sausage gets made https://ift.tt/Q2GmDOC

---

### **Interconnects (Nathan Lambert) ▷ #**[**ml-drama**](https://discord.com/channels/1179127597926469703/1181746144821387334/1250507434490069122) (2 messages):

- **Microsoft snagged OpenAI stake via Tesla bait**: A [tweet](https://fxtwitter.com/nicoleperlroth/status/1800946061613416659?s=46) reveals that Microsoft gained its 49% stake in OpenAI by using it as leverage to get Tesla onto Azure. Microsoft initially attempted to court Tesla into using Azure, but Elon Musk's focus on OpenAI led to this unexpected partnership deal.

**Link mentioned**: [Tweet from Nicole Perlroth (@nicoleperlroth)](https://fxtwitter.com/nicoleperlroth/status/1800946061613416659?s=46): Ever wonder how Microsoft got into OpenAI? I have a great story for you… @elonmusk had it out for Apple because he felt the iCar project was a threat to Tesla. Meanwhile, MSFT had been trying to get ...

---

### **Interconnects (Nathan Lambert) ▷ #**[**random**](https://discord.com/channels/1179127597926469703/1183121795247779910/1250431095493759057) (7 messages):

- **June-Chatbot struggles with NVDA API errors**: A member noted that `june-chatbot` has entered the arena and sometimes "throws NVDA API errors." This highlights potential stability issues with the bot.
- **SnailBot earns a funny nickname**: Another member humorously remarked, "SnailBot really livin up to its name," suggesting the bot's performance is notably slow.
- **LMSYS frustration rises**: Nathan Lambert exclaimed, "lmsys just stop!", possibly referring to ongoing issues or annoyances with LMSYS.
- **Dream Machine's new text-to-video model impresses**: The Dream Machine from [Luma Labs](https://lumalabs.ai/dream-machine) creates high-quality, realistic videos from text and images quickly. It is described as a scalable and efficient transformer model capable of physically accurate, consistent, and dynamic video generation.
- **Co-determined AI phenomena catches attention**: Nathan Lambert commented on the simultaneous emergence of advancements in AI, finding it "interesting how co-determined AI seems" and noting these developments "just come at the same itme."

**Link mentioned**: [Luma Dream Machine](https://lumalabs.ai/dream-machine): Dream Machine is an AI model that makes high quality, realistic videos fast from text and images from Luma AI

---

### **Interconnects (Nathan Lambert) ▷ #**[**rl**](https://discord.com/channels/1179127597926469703/1208183216843005962/1250203449400885421) (10 messages🔥):

- **Apple Innovates with Hybrid Data Strategy**: Discussing a recent [Apple blog post](https://apple.com/blog), one member highlighted Apple's hybrid data strategy which includes both human-annotated and synthetic data. Apple also developed novel algorithms like a rejection sampling fine-tuning algorithm and RLHF with mirror descent policy optimization.
- **PPO Pseudocode Discrepancy Discussed**: A [Twitter thread by @jsuarez5341](https://x.com/jsuarez5341/status/1800677493495783713) caught attention for pointing out discrepancies in PPO implementations, specifically regarding gradient accumulation. Members discussed how the pseudocode implies it, but many popular implementations skip it, possibly affecting stability with larger batch sizes.
- **Tulu 2.5 Stability Insights**: In response to the PPO discussion, a member shared that Tulu 2.5 uses updates with each minibatch step and noted that changing this setup did not significantly impact performance beyond speed variations. They concluded that while the technical details are significant, the practical differences may be marginal in some scenarios.

**Links mentioned**:

- [Tweet from Joseph Suarez (e/🐡) (@jsuarez5341)](https://x.com/jsuarez5341/status/1800677493495783713): I think I found a discrepancy in PPO. The pseudocode implies gradient accumulation over minibatches. OpenAI implementation (which libraries use as reference) does not accumulate. This matters 🧵
- [Tweet from Joseph Suarez (e/🐡) (@jsuarez5341)](https://x.com/jsuarez5341/status/1800919908613849481): Larger batch size = more stable training, right? Wrong! Causes PPO to diverge because of a discrepancy between the pseudocode below and actual implementations, which skip gradient accumulation. Quick ...

---

### **Interconnects (Nathan Lambert) ▷ #**[**posts**](https://discord.com/channels/1179127597926469703/1228051082631188530/1250433618287792209) (8 messages🔥):

- **Genmoji drives OS adoption with emoji love**: Discussion touched on how the public's love for emojis, especially **genmoji**, drives OS adoption. One user mentioned that this feature even makes the new phones "wife-approved."
- **Nathan dives into RL practice**: Nathan Lambert mentioned his week was rushed but found it **good practice** and enjoyed seeing how professionals discuss these topics. Another user appreciated his **nerding about RL**, indicating a shared interest.
- **Cohere’s Leave-One-Out RL Mention**: Nathan Lambert regretted missing a link to **Cohere’s leave one out RL** in his detailing of RL concepts. He expressed that the **KL direction** was particularly cool to him.

For more details, feel free to ask!

---

### **Nous Research AI ▷ #**[**off-topic**](https://discord.com/channels/1053877538025386074/1109649177689980928/1250512433819287603) (1 messages):

- **Manifold Research Seeks Collaborators**: Sidh from Manifold Research announced they are looking for research collaborators interested in transformers for multimodality and control tasks. They aim to build a large-scale open-source “Generalist” model, similar to the GATO architecture.
- **Get Involved with Manifold via Various Channels**: They provided several ways to get involved: joining their [Discord](https://discord.com/invite/a8uDbxzEbM?ref=manifoldrg.com), contributing to issues on their [Github](https://github.com/ManifoldRG?ref=manifoldrg.com), and checking out the [OS Research Team Expectations](https://docs.google.com/document/d/e/2PACX-1vQgq32ChlP_e26mRPgfC31lZJCcAHAgbJ_Tn1nfzq8pfysoPAUqAWnel87Qc26h2Q/pub?ref=manifoldrg.com) for more committed involvement.

**Link mentioned**: [Opportunities](https://www.manifoldrg.com/opportunities/): There are a few ways to get involved with our work: 1. Join our Discord and take part in events and discussion, both project related and not. 2. Contribute asynchronously to issues on our Github. ...

---

### **Nous Research AI ▷ #**[**interesting-links**](https://discord.com/channels/1053877538025386074/1132352574750728192/1250180123815186553) (2 messages):

- **GendelveChat simulates smart factory**: Members discussed [GendelveChat](https://x.com/gendelvechat/status/1800580692046405784?s=46&t=eY--9rPoOkHV-u9kocxCMA) utilizing @websim_ai for simulating group chat UX for various industries and workflows. An example shown was a smart factory workflow for defect detection, suggesting that "maybe a group chat model with human insights works better?"
- **StabilityAI announces Stable Diffusion 3 Medium open weights**: [StabilityAI](https://x.com/StabilityAI/status/1800875914299048404) revealed the open weights for Stable Diffusion 3 Medium, their latest text-to-image AI model. This release is seen as a significant advancement in generative AI, aimed at continuing their mission of democratizing the technology.

**Links mentioned**:

- [Tweet from Stability AI (@StabilityAI)](https://x.com/StabilityAI/status/1800875914299048404): Today, we’re thrilled to announce the open weights for Stable Diffusion 3 Medium, the latest and most advanced text-to-image AI model in our Stable Diffusion 3 series! This new release represents a m...
- [Tweet from Gendelve (@GendelveChat)](https://x.com/gendelvechat/status/1800580692046405784?s=46&t=eY--9rPoOkHV-u9kocxCMA): @websim_ai My YC Demo built using @websim_ai You can target any industry, workflow or usecase and it will simulate a new GroupChatUX built for Human(s) & AI Teams Eg: a smart factory workflow simul...

---

### **Nous Research AI ▷ #**[**general**](https://discord.com/channels/1053877538025386074/1149866623109439599/1250166170745569390) (34 messages🔥):

- **Lemmyle Trolls with Stalin Card Edits**: An apparent joke about altering the Joseph Stalin character card in the `NousResearch/CharacterCodex` sparked reactions. *"Stalin was one of the best leaders of all time,"* triggered a brief discussion about historical accuracy and trolling.
- **Azure2089 on Apple's AI**: Shared a detailed overview of Apple's AI system architecture [Introducing Apple Foundation Models](https://machinelearning.apple.com/research/introducing-apple-foundation-models). The discussion emphasized Apple's 3B model, its compression techniques, speculative decoding for speed optimization, and the lack of details on GPT-4O integration.
- **Stable Diffusion 3 Medium Release Announced**: Announced the release of [Stable Diffusion 3 Medium](https://huggingface.co/stabilityai/stable-diffusion-3-medium) featuring improved image quality and performance. Included a link to a [research paper](https://stability.ai/news/stable-diffusion-3-research-paper) for further technical details.
- **_kquant shares Data Pipeline Idea**: Suggested a data pipeline capable of generating multi-turn ShareGPT data for the Character Codex project. Mentioned using **Cat-70B** with system prompts and tools like **oobabooga** for implementation.
- **Teknium highlights Stable Diffusion 3 Usage**: Shared a tweet from [minimaxir](https://x.com/minimaxir/status/1800921802765717754) showcasing impressive tests of Stable Diffusion 3, highlighting the model's high performance.

**Links mentioned**:

- [Discord - Group Chat That’s All Fun & Games](https://discordapp.com/channels/1053877538025386074/1145143867818119272/1250149635083866133): Discord is great for playing games and chilling with friends, or even building a worldwide community. Customize your own space to talk, play, and hang out.
- [stabilityai/stable-diffusion-3-medium · Hugging Face](https://huggingface.co/stabilityai/stable-diffusion-3-medium): no description found
- [Nitral-AI/Hathor-L3-8B-v.02 · Hugging Face](https://huggingface.co/Nitral-AI/Hathor-L3-8B-v.02): no description found
- [Nitral-AI/Hathor-L3-8B-v.02 · Great Uncensored Model](https://huggingface.co/Nitral-AI/Hathor-L3-8B-v.02/discussions/1): no description found
- [Tweet from Max Woolf (@minimaxir)](https://x.com/minimaxir/status/1800921802765717754): Taking a look at people testing out Stable Diffusion 3 and tbh this goes hard.
- [Introducing Apple’s On-Device and Server Foundation Models](https://machinelearning.apple.com/research/introducing-apple-foundation-models): At the 2024 Worldwide Developers Conference, we introduced Apple Intelligence, a personal intelligence system integrated deeply into…

---

### **Nous Research AI ▷ #**[**ask-about-llms**](https://discord.com/channels/1053877538025386074/1154120232051408927/1250312828628238387) (7 messages):

- **Seeking Japanese language models**: A member asked for recommendations on **visual language models (VLM)** and **large language models (LLM)** for the Japanese language. They requested a list for better clarity.
- **Cohere Aya suggested and reviewed**: Another member suggested **Cohere Aya** for multilingual capabilities and mentioned **Stability AI** models. The initial member found **Aya** to be good but sought API access instead of self-deployment.
- **Exploring Cohere's API options**: A member mentioned that **Cohere** provides API access to **Aya 35B** (c4ai-aya-23) and suggested **Command R+** for a larger model supporting Japanese (103B). They shared their positive experience after using Aya.
- **Finding API information and final recommendations**: The requester struggled to find API pricing information for **Aya 35B** and asked for help with a direct link. They later expressed relief and satisfaction upon finding **Cohere Command R+** to be the optimal Japanese model for their needs.

---

### **Nous Research AI ▷ #**[**world-sim**](https://discord.com/channels/1053877538025386074/1221910674347786261/1250488272140898417) (2 messages):

- **Update simplifies longer console prompts**: An update has been released designed to streamline writing and editing longer console prompts, enhancing the experience across both mobile and desktop platforms. The update promises to make console interactions smoother for users.
- **Inquiry about WorldSim's open-source status**: A member inquired about whether WorldSim is open-source and if it has a public repository. No response or further information was provided in the messages.

---

### **LAION ▷ #**[**general**](https://discord.com/channels/823813159592001537/823813160075132991/1250254590767792222) (26 messages🔥):

- **Speculation on DPO LoRA's potential**: There was talk about a potential **DPO LoRA**, which some believe could lead to negative outcomes like **"making terrible porny things"**. However, the actual implications remain to be seen.
- **Elon Musk vs Apple over OpenAI partnership**: A user claimed that **Elon Musk nuked Apple's account** due to Apple's partnership with OpenAI. This sentiment was supported with a link to [Threads.net](https://www.threads.net/@ronaldfilipkowski/post/C8F8woLt7rT).
- **CaptionEmporium recaptions CC12M with LlavaNext**: Users discussed the **CC12M dataset** recaptioned using **LlavaNext** and released on HuggingFace. The link to the dataset is [CaptionEmporium/conceptual-captions-cc12m-llavanext](https://huggingface.co/datasets/CaptionEmporium/conceptual-captions-cc12m-llavanext).
- **Stable Diffusion 3 medium released without diffusers**: It was noted that **Stable Diffusion 3 medium** has been released but lacks **diffusers code/weights**. This was confirmed with a link to its current state on [HuggingFace](https://huggingface.co/stabilityai/stable-diffusion-3-medium/tree/main).
- **New dataset for conditioning on emotional descriptions**: An exciting new dataset for **conditioning on emotional descriptions** was shared. The dataset, named **Emo-2-SNAC**, is available on [HuggingFace](https://huggingface.co/datasets/0xd4t4/Emo-2-SNAC).

**Links mentioned**:

- [Ron Filipkowski (@ronaldfilipkowski) on Threads](https://www.threads.net/@ronaldfilipkowski/post/C8F8woLt7rT): Elon Musk has auto-blocked everyone on Twitter from Apple.
- [CaptionEmporium/TextOCR-GPT4o · Datasets at Hugging Face](https://huggingface.co/datasets/CaptionEmporium/TextOCR-GPT4o?row=1>): no description found
- [0xd4t4/Emo-2-SNAC · Datasets at Hugging Face](https://huggingface.co/datasets/0xd4t4/Emo-2-SNAC): no description found
- [ylacombe/mls-eng-10k-descriptions-10k-v4 · Datasets at Hugging Face](https://huggingface.co/datasets/ylacombe/mls-eng-10k-descriptions-10k-v4): no description found

---

### **LAION ▷ #**[**research**](https://discord.com/channels/823813159592001537/824374369182416994/1250286057602285629) (9 messages🔥):

- **Image dataset labelling: iPhone to Gradio**: A member noted interest in *building an image dataset* and labelling images on the ground using an iPad and iPhone. They discussed potentially using *Gradio* to create a custom interface for labelling on the go, with around 5000 images to be labeled, including cropped sections.
- **RecurrentGemma 9B by Google Drops**: A member shared the release of [RecurrentGemma 9B by Google](https://huggingface.co/collections/google/recurrentgemma-release-66152cbdd2d6619cb1665b7a) with *super-fast throughput* for long sequences. The model boasts *similar quality to Gemma* and includes both base and instruct-tuned versions, highlighted in [Osanseviero's post](https://x.com/osanseviero/status/1800607752038818260).
- **Transformer's learnability limits discussed**: A link to a paper on [arxiv.org](https://arxiv.org/abs/2406.06467) discusses the *limitations of Transformers* in predicting new syllogisms by composing established ones. The study explores the concept of 'distribution locality' and the challenges posed by high locality distributions in terms of learnability.
- **Stable Diffusion 3 release shared**: The [Stable Diffusion 3 collection](https://huggingface.co/collections/stabilityai/stable-diffusion-3-666992fb11f5d8d0ec8192e8) by StabilityAI was shared. The collection highlights advancements in diffusion models, as indicated by a member's link to Hugging Face.

**Links mentioned**:

- [How Far Can Transformers Reason? The Locality Barrier and Inductive Scratchpad](https://arxiv.org/abs/2406.06467): Can Transformers predict new syllogisms by composing established ones? More generally, what type of targets can be learned by such models from scratch? Recent works show that Transformers can be Turin...
- [Stable Diffusion 3 - a stabilityai Collection](https://huggingface.co/collections/stabilityai/stable-diffusion-3-666992fb11f5d8d0ec8192e8): no description found
- [Tweet from Omar Sanseviero (@osanseviero)](https://x.com/osanseviero/status/1800607752038818260): RecurrentGemma 9B by Google is out 🔥 ⚡️Super fast for long sequences: Good throughput+latency 👀Base and instruct tuned versions 🏆Similar quality as Gemma Check the y-axis below 🤯 Models: https:...

---

### **LAION ▷ #**[**resources**](https://discord.com/channels/823813159592001537/991938328763056168/1250422238516084817) (1 messages):

- **New Machine Learning Library on GitHub**: A member shared their newly developed machine learning library based on **TensorFlow**. The library aims to "easily implement parallel training and distributed training" with support for various models including Llama2, Llama3, CLIP, and more. Check it out [here](https://github.com/NoteDance/Note).

**Link mentioned**: [GitHub - NoteDance/Note: Easily implement parallel training and distributed training. Machine learning library. Note.neuralnetwork.tf package include Llama2, Llama3, Gemma, CLIP, ViT, ConvNeXt, BEiT, Swin Transformer, Segformer, etc, these models built with Note are compatible with TensorFlow and can be trained with TensorFlow.](https://github.com/NoteDance/Note): Easily implement parallel training and distributed training. Machine learning library. Note.neuralnetwork.tf package include Llama2, Llama3, Gemma, CLIP, ViT, ConvNeXt, BEiT, Swin Transformer, Segf...

---

### **Modular (Mojo 🔥) ▷ #**[**general**](https://discord.com/channels/1087530497313357884/1098713601386233997/1250168751416934480) (3 messages):

- **Mojo Might Work with TPU Hardware in the Future**: A member suggested that if **Google has a TPU backend** for MLIR or LLVM, then **Mojo** should be capable of utilizing it.
- **Modular Compiler to Be Extensible**: In future development, users won't need to wait for Modular to support new architectures, as they plan to make the compiler **extensible**.
- **Mojo Infrastructure Is New**: Members reminded each other that **Mojo** and its infrastructure are still in early stages and actively being built out.

---

### **Modular (Mojo 🔥) ▷ #**[**💬︱twitter**](https://discord.com/channels/1087530497313357884/1098713626161987705/) (1 messages):

ModularBot: From *Modular*: [https://twitter.com/Modular/status/1800948901652181260](https://twitter.com/Modular/status/1800948901652181260)

---

### **Modular (Mojo 🔥) ▷ #**[**📺︱youtube**](https://discord.com/channels/1087530497313357884/1098713700719919234/1250503759604224073) (1 messages):

- **New Modular Video Alert**: The latest video from **Modular** has been released. Watch it [here](https://www.youtube.com/watch?v=uookgZ7Ojg8).

---

### **Modular (Mojo 🔥) ▷ #**[**🔥mojo**](https://discord.com/channels/1087530497313357884/1151418092052815884/1250168359618609253) (26 messages🔥):

- **Variable Alias Advice Results in No Performance Change**: A user suggested changing `var size = 12` to `alias size = 12`. Another user confirmed that this change showed no noticeable performance improvement in their benchmark tests.
- **Outdated `Tensor` Module Example in Documentation**: The provided `Tensor` example on the official modular documentation site doesn't work as expected due to incorrect usage of `Tensor.rand`. After some discussion, it was suggested to use `Tensor[DType.float32].rand(...)`, which resolved the issue.
- **Pointer Conversion Solution Sparks Celebrations**: A user asked how to convert a pointer to a `UInt64`, and another user provided a working solution. This led to successful implementation in a [Pull Request](https://github.com/modularml/mojo/pull/3007) and was met with enthusiastic praise.
- **Looking for Mojo Tutorials for VSCode**: A user expressed frustration in finding good Mojo tutorials for VSCode, to which several resources and humorous remarks about the learning curve were shared. Among the suggestions was [a book-like resource](https://ruhati.net/mojo), though it was noted it's still a work in progress.
- **Tech Influencers as Modern Critics**: A light-hearted conversation emerged about whether programming critics exist, culminating in the consensus that tech influencers like Theo and Primeagen fulfill that role through their videos. This led to sharing anticipation for upcoming interviews and existing content, such as a [Marques and Tim interview](https://www.youtube.com/watch?v=pMX2cQdPubk).

**Links mentioned**:

- [tensor | Modular Docs](https://docs.modular.com/mojo/stdlib/tensor/tensor/): Implements the Tensor type.
- [[stdlib] Specify alignment in UnsafePointer.alloc by sa- · Pull Request #3007 · modularml/mojo](https://github.com/modularml/mojo/pull/3007): Fixes #3006 Adds a function parameter in UnsafePointer.alloc to specify alignment at compile time. Although I'm not sure how to test it because I couldn't find a way to convert ptr.address to ...
- [Learn Mojo Programming Language](https://ruhati.net/mojo): no description found

---

### **Modular (Mojo 🔥) ▷ #**[**nightly**](https://discord.com/channels/1087530497313357884/1224434323193594059/1250327138150776916) (2 messages):

- **New Mojo compiler release**: A new nightly Mojo compiler version `2024.6.1205` has been released. Update using `modular update nightly/mojo`, and see the [raw diff](https://github.com/modularml/mojo/compare/76eda306af929d9576d7190a7f8f3aa1df83baf6...b590ea1ba0e80093118e32d91dcc4d252ccdfe69) and [current changelog](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md) for detailed changes.
- **Conditional conformance feature praised**: The release of conditional conformance received positive feedback. A user inquired about the plans for supporting recursive trait bounds, giving an example involving `Wrapper[Wrapper[Stringable]]`.

---

### **OpenInterpreter ▷ #**[**general**](https://discord.com/channels/1146610656779440188/1147665339266650133/1250221369359073341) (6 messages):

- **Zep offers convenient memory management with limitations**: A user pointed out that **Zep** is convenient for memory management as long as you stay within the free tier. *"Not worth it otherwise."*
- **Apple will offer free services**: A member mentioned that **Apple** will offer its services for free, highlighting a competitive advantage over other options. Another user confirmed, *"Yeah, great point."*
- **OpenAI API key pricing**: Discussion indicated that OpenAI keys will cost **around $5-10/month**. This emphasized the cost considerations for those planning to utilize OpenAI's services.
- **Trouble configuring default models on GCP**: A user sought help to change the default model to **gemini-flash** or **codestral** using their GCP account. They noted success with **GPT-4o** but at significant cost and mentioned conflicts with various YAML file configurations.
- **Seeking experts on OpenInterpreter**: Another user expressed eagerness to speak with someone knowledgeable about OpenInterpreter to continue making progress on their work. They requested assistance, *"Please HMU if you want to help."*

---

### **OpenInterpreter ▷ #**[**O1**](https://discord.com/channels/1146610656779440188/1194880263122075688/1250178624770936965) (22 messages🔥):

- **Link to OpenInterpreter GitHub repository shared**: A member shared a [link to the OpenInterpreter GitHub repository](https://github.com/OpenInterpreter/open-interpreter/tree/main/interpreter/terminal_interface/profiles/defaults) showcasing the natural language interface for computers.
- **Temporary code without websockets**: A member shared a [GitHub Gist](https://gist.github.com/0xrushi/e56085f93698c7267af9b1ba9643dc7a) containing temporary code for OpenInterpreter without websockets, specifying that it runs on an OpenAI key.
- **uConsole + OpenInterpreter = Beast Mode**: The combination of uConsole and OpenInterpreter was highlighted as a powerful setup, supported by a [video demonstration](https://odysee.com/@rushi:2/Openinterpreter-01-uconsole-test-part1:2) on Odysee.
- **Details on the setup and components**: A Raspberry Pi CM4 with a fancy keyboard and screen was discussed as the device used for the setup. An associated [link to ClockworkPi’s uConsole](https://www.clockworkpi.com/uconsole) was shared, mentioning that the setup works with any CM4 with no eMMC module.
- **Additional accessories for voice interaction**: While the uConsole includes a speaker, it does not come with a microphone. A member recommended using a [mini USB mic](https://www.amazon.com/dp/B071WH7FC6) for voice input.

**Links mentioned**:

- [no title found](https://www.amazon.com/dp/B071WH7FC6): no description found
- [Odysee](https://odysee.com/@rushi:2/Openinterpreter-01-u): Explore a whole universe of videos on Odysee from regular people just like you!
- [Openinterpreter 01 uconsole test part1](https://odysee.com/@rushi:2/Openinterpreter-01-uconsole-test-part1:2): View Openinterpreter 01 uconsole test part1 on Odysee
- [uConsole | ClockworkPi](https://www.clockworkpi.com/uconsole): uConsole - A real "fantasy console" for indie game developers and bedroom programmers.
- [open-interpreter/interpreter/terminal_interface/profiles/defaults at main · OpenInterpreter/open-interpreter](https://github.com/OpenInterpreter/open-interpreter/tree/main/interpreter/terminal_interface/profiles/defaults): A natural language interface for computers. Contribute to OpenInterpreter/open-interpreter development by creating an account on GitHub.
- [temporary_01_code_without_sockets.py](https://gist.github.com/0xrushi/e56085f93698c7267af9b1ba9643dc7a): GitHub Gist: instantly share code, notes, and snippets.

---

### **OpenRouter (Alex Atallah) ▷ #**[**general**](https://discord.com/channels/1091220969173028894/1094454198688546826/1250202401600962681) (27 messages🔥):

- **DeepEval integrates custom LLM metrics effortlessly**: In DeepEval, users can define custom evaluation metrics for LLMs, including metrics like **G-Eval, Summarization, Faithfulness,** and **Hallucination**. For more details on implementing these metrics, check the [documentation](https://docs.confident-ai.com/docs/metrics-introduction#using-a-custom-llm).
- **Uncensored models offer unfiltered responses**: Users discussed the purpose of new **uncensored models** that lack filters or alignment biases, with many preferring these for their transparent responses and multiple use cases.
- **WizardLM-2 pricing sparks curiosity**: Questions arose about how **WizardLM-2** operates profitably at only $0.65 per million tokens. A member explained these models may use fewer active parameters to save costs, and renting GPUs might help with cost management.
- **Self-hosting vs. using providers for LLMs**: Discussions highlighted the challenges and costs associated with **self-hosting models** compared to using service providers like OpenRouter. It was noted that self-hosting might not be economically viable unless there is consistent high demand or existing hardware investments.
- **Batch inference could justify GPU rental**: Members debated the effectiveness of renting GPUs for heavy batch inference tasks, noting that it could be cost-effective compared to single-request scenarios. A practical suggestion was to use tools like **Aphrodite-engine / vllm** to optimize large batch processing.

**Link mentioned**: [Metrics | DeepEval - The Open-Source LLM Evaluation Framework](https://docs.confident-ai.com/docs/metrics-introduction#using-a-custom-llm): Quick Summary

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**general**](https://discord.com/channels/1104757954588196865/1104757955204743201/1250413917583179808) (6 messages):

- **QWEN 2 Training Inquiry**: A member asked if anyone has trained **QWEN 2** with 1.5 billion parameters, seeking performance insights.
- **Running 7 Billion Model Locally**: The same user described running a **7 billion parameter** model locally, unquantized, using the **mlx library**. They noted it’s *"slow af, but works lmao"*.
- **Llama.cpp Suggestion**: Another member suggested using **llama.cpp** for better performance. The original poster indicated they were just testing the setup.

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**axolotl-dev**](https://discord.com/channels/1104757954588196865/1104758010959634503/1250438163768545302) (2 messages):

- **Runpod workspace woes**: One user expressed frustration, asking *"why not just keep it as /workspace/?"*. Another user explained that mounting the data volume to **/workspace** on Runpod often overwrites **/workspace/axolotl**, resulting in having to reinstall or reclone Axolotl.

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**docs**](https://discord.com/channels/1104757954588196865/1167137552470392842/) (1 messages):

le_mess: Step 2. is not needed in uploading the model. Repo is createevent automatically.

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**axolotl-help-bot**](https://discord.com/channels/1104757954588196865/1225300056442409040/1250421897791934465) (6 messages):

- **Critical error in PyTorch distributed launch**: A member encountered a child process failure when using PyTorch's distributed launcher: *"torch.distributed.elastic.multiprocessing.api: [ERROR] failed (exitcode: -7) local_rank: 4 (pid: 3830) of binary: /root/miniconda3/envs/py3.10/bin/python"*.
- **Phorm offers troubleshooting steps**: The steps to resolve the *ChildFailedError* include checking the environment setup, verifying distributed configuration, inspecting the script for errors, increasing shared memory size, and enabling debug mode as described in the [Accelerate documentation](https://github.com/huggingface/accelerate/tree/main/docs/source/basic_tutorials/troubleshooting.md#L50L145).

**Links mentioned**:

- [accelerate/docs/source/basic_tutorials/troubleshooting.md at main · huggingface/accelerate](https://github.com/huggingface/accelerate/tree/main/docs/source/basic_tutorials/troubleshooting.md#L50L145)): 🚀 A simple way to launch, train, and use PyTorch models on almost any device and distributed configuration, automatic mixed precision (including fp8), and easy-to-configure FSDP and DeepSpeed suppo.....
- [OpenAccess-AI-Collective/axolotl | Phorm AI Code Search](https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=c0a36daa-352f-4d94-8f0c-1e23a40fbb2f)): Understand code, faster.

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**axolotl-phorm-bot**](https://discord.com/channels/1104757954588196865/1225558824501510164/1250437253189337129) (6 messages):

- **Continuing Instruction Training with LoRA**: A user asked about moving from completion format training to instruction format training using LoRA. The response detailed steps such as loading the pre-trained model, preparing the instruction data, configuring LoRA, and setting up and conducting the training process, with a code example provided.
- **Query on Merging qLoRA Training for Mistral 7B**: A member inquired about merging results from a qLoRA training on Mistral 7B using a system with only 24GB GPU. The question highlighted a need for guidance on handling resource constraints during the model merging process.

**Link mentioned**: [OpenAccess-AI-Collective/axolotl | Phorm AI Code Search](https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=19ce6da5-fa00-4b7a-80ea-bde7eb10ae28)): Understand code, faster.

---

### **Latent Space ▷ #**[**ai-general-chat**](https://discord.com/channels/822583790773862470/1075282825051385876/1250168057851019356) (17 messages🔥):

- **Google's RecurrentGemma 9B release excites**: A new model, **RecurrentGemma 9B** by Google, promises super-fast performance for long sequences. It includes both base and instruct-tuned versions and is rated to be of similar quality to **Gemma**. [Learn more here](https://huggingface.co/collections/google/recurrentgemma-release-66152cbdd2d6619cb1665b7a) and read [the Griffin paper](https://arxiv.org/abs/2402.19427).
- **Google's Personal Health LLM debut**: Google launched a new **Personal Health Large Language Model** fine-tuned on Gemini, designed to read wearable data and provide personalized recommendations. *"It outperformed professional sleep and fitness experts on certification exams"* noted one excited member. [Details here](https://x.com/chefjeffsf/status/1800597192593621100).
- **ARC Prize Launch Announcement**: The ARC prize was launched today, linked to an ongoing project collecting data on how humans solve ARC tasks, now hosting 4100 interaction histories. Contributions include multiple [videos and datasets](https://github.com/neoneye/ARC-Interactive-History-Dataset) demonstrating approaches and solutions. [Further participation and datasets here](https://neoneye.github.io/arc/).
- **Mihail Eric's exposé on Alexa AI shortcomings**: In a detailed thread, Mihail Eric discusses **Alexa AI**'s missed opportunities due to technical and bureaucratic issues, including fragmentation and misalignment between product and science teams. Despite Alexa’s significant resources, many potential advancements never saw public release. [Full story here](https://x.com/mihail_eric/status/1800578001564057754).
- **Stanford lecture by hwchung27**: A comprehensive lecture by hwchung27 at Stanford discusses the rapid evolution in AI, emphasizing the importance of adapting to changes driven by exponentially cheaper compute and scaling models. Practical examples, such as the evolution of Transformer architectures, are used to highlight these concepts. Watch the [lecture video](https://youtu.be/orDKvo8h71o?si=RIfyZ7NSUAJifOBF) and check out the [slides](https://docs.google.com/presentation/d/1u05yQQaw4QXLVYGLI6o3YoFHv6eC3YN8GvWD8JMumpE/edit?usp=sharing).

**Links mentioned**:

- [Tweet from Omar Sanseviero (@osanseviero)](https://x.com/osanseviero/status/1800607752038818260?s=46&t=90xQ8sGy63D2OtiaoGJuww): RecurrentGemma 9B by Google is out 🔥 ⚡️Super fast for long sequences: Good throughput+latency 👀Base and instruct tuned versions 🏆Similar quality as Gemma Check the y-axis below 🤯 Models: https:...
- [I Told You This Was Going To Happen](https://youtu.be/zbo6SdyWGns?si=5UypK-JD5h7Gz-SJ): In today's episode I talk about some recent advancements in AI music technology and what its implications are.SALE:🎂 The Channel Anniversary Bundle — $89 FO...
- [Tweet from Hyung Won Chung (@hwchung27)](https://x.com/hwchung27/status/1800676312916656592?s=46&t=90xQ8sGy63D2OtiaoGJuww): I gave a lecture at @Stanford CS 25. Lecture video: https://youtu.be/orDKvo8h71o?si=RIfyZ7NSUAJifOBF AI is moving so fast that it's hard to keep up. Instead of spending all our energy catching u...
- [Tweet from Mihail Eric (@mihail_eric)](https://x.com/mihail_eric/status/1800578001564057754?s=46&t=90xQ8sGy63D2OtiaoGJu): How Alexa dropped the ball on being the top conversational system on the planet — A few weeks ago OpenAI released GPT-4o ushering in a new standard for multimodal, conversational experiences with soph...
- [Tweet from Chef Jeff (@chefjeffsf)](https://x.com/chefjeffsf/status/1800597192593621100): Breaking: Google just published a Personal Health Large Language Model - Fine-tuned on Gemini - Reads your wearable data to find personalized insights and recommendations - Outperformed professional ...
- [Tweet from Mihail Eric (@mihail_eric)](https://x.com/mihail_eric/status/1800578001564057754?s=46&t=90xQ8sGy63D2OtiaoGJuww): How Alexa dropped the ball on being the top conversational system on the planet — A few weeks ago OpenAI released GPT-4o ushering in a new standard for multimodal, conversational experiences with soph...
- [Tweet from Lucas Beyer (bl16) (@giffmana)](https://x.com/giffmana/status/1800617190242091289?s=46&t=90xQ8sGy63D2OtiaoGJuww): If you've never been at BigCo, you've likely found yourself thinking something like: > why the fuck is this BigCo thing so obviously shitty? Fixing this would take me one afternoon! Their ...
- [ARC Prize – a $1M+ competition towards open AGI progress | Hacker News](https://news.ycombinator.com/item?id=40648960): no description found

---

### **tinygrad (George Hotz) ▷ #**[**general**](https://discord.com/channels/1068976834382925865/1068976834928193609/1250361399935242342) (16 messages🔥):

- **$600 Bounty for RetinaNet on MLPerf**: George Hotz announced a $600 bounty for implementing RetinaNet on MLPerf, doubling the bounty if it makes it into the MLPerf benchmark. He mentioned, "with ResNet/BERT in master, this bounty shouldn't be that hard."
- **Challenges with PyTorch Data Loading**: A member highlighted that most development time with PyTorch is spent on efficient data loading, particularly in HPC clusters. They suggested adopting solutions like [WebDataset](https://github.com/webdataset/webdataset).
- **TinyGrad Data Loader Plan**: George Hotz proposed a simple API for TinyGrad's data loader that includes specifying a function to load records, shuffle buffer size, and batch size. He mentioned Comma's "gigashuffle" as an existing abstraction for reference.
- **TinyGrad 0.9.0 Update**: TinyGrad 0.9.0 has been added to the Nix repository, as per a [GitHub Pull Request](https://github.com/NixOS/nixpkgs/pull/316931). The update includes gpuctypes directly in the code base.
- **MLPerf and Publications**: MLPerf results are out, including tinybox red/green benchmarks. There was also a mention of a German article about TinyGrad on [Heise.de](https://www.heise.de/news/KI-Benchmark-MLPerf-Erste-AMD-Beschleuniger-mit-Minimalauftritt-9760531.html) and plans for a more detailed blog post to compare TinyGrad's speed to theoretical limits.

**Links mentioned**:

- [no title found](https://public.tableau.com/app/profile/data.visualization6666/viz/MLCommons-Training_16993769118290/MLCommons-Training): no description found
- [GitHub - pytorch/data: A PyTorch repo for data loading and utilities to be shared by the PyTorch domain libraries.](https://github.com/pytorch/data): A PyTorch repo for data loading and utilities to be shared by the PyTorch domain libraries. - pytorch/data
- [tinygrad/docs/tinygrad_intro.pdf at master · tinygrad/tinygrad](https://github.com/tinygrad/tinygrad/blob/master/docs/tinygrad_intro.pdf): You like pytorch? You like micrograd? You love tinygrad! ❤️ - tinygrad/tinygrad
- [[MLPERF] Retinanet by reddyn12 · Pull Request #4245 · tinygrad/tinygrad](https://github.com/tinygrad/tinygrad/pull/4245): TODO: Add float16 support Fix float16 save, float32 works Add bfloat16 support Speed up validation loop Fix val loop memory bloat (holding grads) Cleanup anchor gen code Remove old retinan...
- [python311Packages.tinygrad: 0.8.0 -> 0.9.0 by GaetanLepage · Pull Request #316931 · NixOS/nixpkgs](https://github.com/NixOS/nixpkgs/pull/316931): Description of changes Changelog: https://github.com/tinygrad/tinygrad/releases/tag/v0.9.0 Notable change: gpuctypes has been included directly in the code base. So we have to port our GPU-dependen...
- [KI-Benchmark MLPerf: Erste AMD-Beschleuniger mit Minimalauftritt](https://www.heise.de/news/KI-Benchmark-MLPerf-Erste-AMD-Beschleuniger-mit-Minimalauftritt-9760531.html): Er hat's getan: Die Firma tinycorp des Ex-Hackers George Hotz hat die ersten AMD-Chips im MLPerf Training v4.0 platziert – aber mit nur einem Einzelwert.

---

### **Mozilla AI ▷ #**[**llamafile**](https://discord.com/channels/1089876418936180786/1182689832057716778/1250176793441210429) (13 messages🔥):

- **LLMs need grammar or JSON schemas for utility**: A member argued that **LLMs are largely useless** in an applied manner (agents) without grammar or JSON schemas as it forces the sampler to pick only from a subset of tokens, making the outputs parsable by an external program. This process improves utility in the application layer by standardizing output formats.
- **Simplify llamafile grammar usage**: Members discussed simplifying the use of `llamafile` with grammar options and pointed out the two-step process: invoking JSON schema to grammar and then consuming that grammar. One suggested the command `llamafile --grammar <(schemaify <foo.sql)`.
- **Packing ggml_cuda.so and ggml_rocm.so**: A user inquired about efficiently packing `ggml_cuda.so` and `ggml_rocm.so` into the llamafile release. Another member responded with a Bash script example and mentioned manual modifications are needed for AMD and tinyblas.
- **Use of magit for syncing**: To sync `llama.cpp`, a member shared that they use [Magit](https://www.youtube.com/watch?v=urcL86UpqZc)—a popular Git interface within Emacs—illustrated humorously in a YouTube video titled "Interview with an Emacs Enthusiast in 2023 [Colorized]".

**Link mentioned**: [Interview with an Emacs Enthusiast in 2023 [Colorized]](https://www.youtube.com/watch?v=urcL86UpqZc): Emacs OSInterview with an Emacs Enthusiast in 2023 with Emerald McS., PhD - aired on © The Emacs.org. air date 1990.Programmer humorSoftware humorElisp humor...

---

### **Datasette - LLM (@SimonW) ▷ #**[**ai**](https://discord.com/channels/823971286308356157/1097032579812687943/1250231626315862036) (1 messages):

- **Apple's adaptors resemble LORA layers**: A member highlighted an interesting point in Apple’s videos about their adaptors that dynamically load on local models to perform different tasks. They likened this feature to specialized **LORA layers**.

---

### **Datasette - LLM (@SimonW) ▷ #**[**llm**](https://discord.com/channels/823971286308356157/1128504153841336370/1250187539038343178) (10 messages🔥):

- **Chrisamico struggles with article summarization**: Chrisamico sought help on `curl`ing a long article, extracting specific tag contents, and piping it to `llm` with a system prompt to summarize it. Eventually, he found it easier to paste the article directly into ChatGPT.
- **HTML scraping tools galore**: Members discussed several tools for extracting content from HTML, including [htmlq](https://github.com/mgdm/htmlq), [shot-scraper](https://shot-scraper.datasette.io/en/stable/javascript.html#example-extracting-page-content-with-readability-js), and the `nokogiri` Ruby gem. Simonw provided detailed instructions on using shot-scraper for extracting content with JavaScript.
- **Shot-scraper tips from Simon**: Simonw shared how to use `shot-scraper` to execute JavaScript against a page, extracting page contents efficiently. Additionally, he explained using CSS selectors directly for HTML extraction with `shot-scraper`.
- **Dbreunig learns about Nokogiri CLI**: Dbreunig mentioned discovering that the Ruby gem `nokogiri` can be used as a CLI tool for parsing HTML after installation. He provided a quick example of how to use `nokogiri` to extract and format article text from an HTML document.

**Links mentioned**:

- [A Plea for Sober AI](https://www.dbreunig.com/2024/05/16/sober-ai.html): The hype is so loud we can’t appreciate the magic
- [Datasette](https://datasette.io/): Datasette is a tool for exploring and publishing data. It helps people take data of any shape, analyze and explore it, and publish it as an interactive website and accompanying API.
- [GitHub - mgdm/htmlq: Like jq, but for HTML.](https://github.com/mgdm/htmlq): Like jq, but for HTML. Contribute to mgdm/htmlq development by creating an account on GitHub.
- [Scraping pages using JavaScript - shot-scraper](https://shot-scraper.datasette.io/en/stable/javascript.html#example-extracting-page-content-with-readability-js): no description found
- [Datasette 0.61: The annotated release notes](https://simonwillison.net/2022/Mar/24/datasette-061/): I released Datasette 0.61 this morning—closely followed by 0.61.1 to fix a minor bug. Here are the annotated release notes. In preparation for Datasette 1.0, this release includes two potentially …

---

### **LangChain AI ▷ #**[**general**](https://discord.com/channels/1038097195422978059/1038097196224086148/1250299529736093757) (6 messages):

- **LangChain Postgres documentation confuses users**: A member shared issues accessing LangChain Postgres documentation, noting the absence of a checkpoint in the package. *"What's going on with langchain_postgres. I can't use the documentation from here as there's no checkpoint in the package."* [LangChain Postgres Documentation](https://api.python.langchain.com/en/latest/checkpoint/langchain_postgres.checkpoint.PostgresSaver.html).
- **Combining chains in one call**: A user inquired about combining a chain outputting an array with another chain in one call and considered using `RunnableParallel` but was unsure if it was the best method. *"Anybody knows if it can be done in one chain? I was thinking of RunnableParallel..."*
- **Error using GPT-4 in langchain_openai**: A member reported an error when using GPT-4 with `langchain_openai` and received guidance to use `ChatOpenAI` instead of `OpenAI`, as the latter uses a legacy API not supporting newer models. *"using just* `OpenAI` uses the legacy completions API which does not support newer models. You want to use `ChatOpenAI`." [OpenAI Completions API](https://platform.openai.com/docs/api-reference/completions).

**Link mentioned**: [langchain_postgres.checkpoint.PostgresSaver — 🦜🔗 LangChain 0.2.0rc2](https://api.python.langchain.com/en/latest/checkpoint/langchain_postgres.checkpoint.PostgresSaver.html#): no description found

---

### **LangChain AI ▷ #**[**langserve**](https://discord.com/channels/1038097195422978059/1170024642245832774/1250471553288638625) (1 messages):

- **Sharing conversation history in LangServe**: A user asked for advice on how to share conversation history in LangServe's chat playground. They referenced [GitHub Issue #677](https://github.com/langchain-ai/langserve/issues/677), stating that clicking the "Share" button and copying the URL results in opening an empty chat instead of the intended history.

**Link mentioned**: [How to share conversation from chat playground? · Issue #677 · langchain-ai/langserve](https://github.com/langchain-ai/langserve/issues/677): I want to share a conversation from LangServe, how how to do that? I clicked on the "Share" button: Then copied the url: When I open that url in the browser, it brings an empty chat, without...

---

### **LangChain AI ▷ #**[**share-your-work**](https://discord.com/channels/1038097195422978059/1038097372695236729/1250270488081465374) (2 messages):

- **Nostrike AI offers free python tool**: Nostrike AI has announced a new tool to generate CrewAI (python) code, boasting user-friendliness and free access. They also plan to support exporting Langgraph projects soon and invite users to try it at [nostrike.ai](https://nostrike.ai/).
- **Rubik’s AI seeks beta testers**: A member introduces Rubik's AI, an advanced research assistant and search engine, and offers a 2-month free premium trial with promo code `RUBIX`. This trial includes access to models like GPT-4 Turbo, Claude 3 Opus, and Mistral Large, found [here](https://rubiks.ai/).

**Links mentioned**:

- [Rubik's AI - AI research assistant & Search Engine](https://rubiks.ai/): no description found
- [NoStrike](https://nostrike.ai/): no description found

---

### **Torchtune ▷ #**[**announcements**](https://discord.com/channels/1216353675241590815/1216353675241590818/1250390642639503381) (1 messages):

- **New Apps Bring Fun, Games, and Functionality**: *Starting June 18, members can now add apps to their account and use them across all servers, DMs, and GDMs.* Learn more about how to manage these apps in your server in the [Help Center article](https://support.discord.com/hc/articles/23957313048343).
- **Moderate Your Server with Advanced Features**: *Discover what these apps can (and can't) do, set up the new “Use External Apps” permission, and utilize the upgraded AutoMod to keep your server safe.* For those interested in creating their own apps, check out the [guide to building your own app](https://discord.com/developers/docs/tutorials/developing-a-user-installable-app).

---

### **Torchtune ▷ #**[**general**](https://discord.com/channels/1216353675241590815/1216353675744641096/1250379101051818014) (2 messages):

- **Curiosity about cache memory usage in Torchtune**: A member questioned *"Why is torchtune utilising cache memory after each step?"*. Another member asked for more context to better understand and respond to the query.

---

### **Torchtune ▷ #**[**dev**](https://discord.com/channels/1216353675241590815/1236040539409879170/1250529900495110267) (1 messages):

- **RFC for Tokenizer Revamp gets traction**: A member shared an RFC for a wide-scale tokenizer revamp with proposed code changes on [GitHub](https://github.com/pytorch/torchtune/pull/1082). The proposal aims to enable multimodal features, improve composability and design consistency, and reduce onboarding time for new model tokenizers.

**Link mentioned**: [Build software better, together](https://github.com/pytorch/torchtune/pull/1082.): GitHub is where people build software. More than 100 million people use GitHub to discover, fork, and contribute to over 420 million projects.

---

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