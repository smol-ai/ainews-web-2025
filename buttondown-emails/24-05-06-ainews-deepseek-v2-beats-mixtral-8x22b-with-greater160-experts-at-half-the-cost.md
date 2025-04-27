---
id: 71868cce-79a2-4ff0-af3e-13d4e6017711
title: DeepSeek-V2 beats Mixtral 8x22B with >160 experts at HALF the cost
date: '2024-05-06T23:37:03.494203Z'
original_slug: ainews-deepseek-v2-beats-mixtral-8x22b
description: >-
  **DeepSeek V2** introduces a new state-of-the-art MoE model with **236B
  parameters** and a novel Multi-Head Latent Attention mechanism, achieving
  faster inference and surpassing GPT-4 on AlignBench. **Llama 3 120B** shows
  superior creative writing compared to GPT-4 but mixed reasoning results.
  Microsoft is reportedly developing a **500B parameter LLM called MAI-1**.
  Research from Scale AI highlights overfitting issues in models like
  **Mistral** and **Phi**, while **GPT-4**, **Claude**, **Gemini**, and
  **Llama** maintain robustness. In robotics, **Tesla Optimus** advances with
  improved teleoperation and data collection, Nvidia's **DrEureka** automates
  robot skill training, and open-source robotics AI emerges with **LeRobot**.
  Google introduces **Med-Gemini**, fine-tuned for medical AI tasks with top
  benchmark performance. A paper overviewing hallucinations in multimodal LLMs
  discusses detection and mitigation strategies.
companies:
  - deepseek-ai
  - mistral-ai
  - scale-ai
  - microsoft
  - openai
  - anthropic
  - google-deepmind
  - nvidia
  - tesla
  - lerobot
  - google
models:
  - deepseek-v2
  - llama-3-120b
  - llama-3-400b
  - gpt-4
  - mistral
  - phi
  - claude
  - gemini
  - mai-1
  - drureka
  - med-gemini
topics:
  - mixture-of-experts
  - multi-head-attention
  - model-optimization
  - benchmarking
  - overfitting
  - robotics
  - teleoperation
  - simulation-to-real
  - multimodality
  - hallucination-detection
  - fine-tuning
  - medical-ai
  - model-training
people:
  - erhartford
  - maximelabonne
  - bindureddy
  - adcock_brett
  - drjimfan
  - clementdelangue
  - omarsar0
  - rohanpaul_ai
---


<!-- buttondown-editor-mode: plaintext -->> AI News for 5/3/2024-5/6/2024. We checked 7 subreddits and [**373** Twitters](https://twitter.com/i/lists/1585430245762441216) and **28** Discords (**419** channels, and **10335** messages) for you. Estimated reading time saved (at 200wpm): **1112 minutes**.

**More experts are all you need?**

[DeepSeek V2](https://github.com/deepseek-ai/DeepSeek-V2) punches a hole in the [Mistral Convex Hull from last month](https://buttondown.email/ainews/archive/ainews-mixtral-8x22b-instruct-defines-frontier/):

 ![image.png](https://assets.buttondown.email/images/bcf759e8-0ca7-4ccd-a901-6289aedd96ea.png?w=960&fit=max) 

Information on dataset is extremely light; all they say is it's 8B tokens (4x more than [DeepSeek v1](https://arxiv.org/abs/2401.02954)) with about 12% more Chinese than English. 

[Snowflake Arctic](https://buttondown.email/ainews/archive/ainews-snowflake/) was the last very large MoE model with the highest number of experts (128) we'd seen in the wild; DeepSeek v2 now sets a new high water mark scaling up what was already successful with DeepSeekMOE, but also introducing a new attention variant called Multi-Head Latent Attention. 

 ![image.png](https://assets.buttondown.email/images/16916531-1d7f-4068-a398-00d74c9e8fbc.png?w=960&fit=max) 

These result in much faster inference by caching compressed KVs ("reducing KV cache by 93.3%").

 ![image.png](https://assets.buttondown.email/images/4b75f5cc-73a5-4525-ac1d-a394849e4cb4.png?w=960&fit=max) 

The paper details other minor tricks they find useful.

DeepSeek is putting their money where their mouth is - they are [offering token inference on their platform for $0.28 per million tokens](https://twitter.com/deepseek_ai/status/1787478994478321872) about half of the lowest prices seen in the [Mixtral Price War of Dec 2023](https://twitter.com/swyx/status/1744467383090372743).

---

**Table of Contents**

[TOC] 



---

# AI Twitter Recap

> all recaps done by Claude 3 Opus, best of 4 runs. We are working on clustering and flow engineering with Haiku.

**LLM Developments and Releases**

- **Llama 3 Release**: [@erhartford](https://twitter.com/erhartford/status/1787050962114207886) noted that Llama 3 120B is smarter than Opus, and is very excited about llama3-400b. [@maximelabonne](https://twitter.com/maximelabonne/status/1787401780021649911) shared that Llama 3 120B > GPT-4 in creative writing but worse than L3 70B in reasoning.
- **DeepSeek-V2 Release**: [@deepseek_ai](https://twitter.com/deepseek_ai/status/1787478986731429933) launched DeepSeek-V2, an open-source MoE model that places top 3 in AlignBench, surpassing GPT-4. It has 236B parameters with 21B activated during generation.
- **MAI-1 500B from Microsoft**: [@bindureddy](https://twitter.com/bindureddy/status/1787498838024139185) predicted Microsoft is training its own 500B param LLM called MAI-1, which may be previewed at their Build conference. As it becomes available, it will compete with OpenAI's GPT line.
- **Mistral and Open LLMs Overfitting Benchmarks**: [@adcock_brett](https://twitter.com/adcock_brett/status/1787151286305017966) shared that Scale AI released research uncovering 'overfitting' of certain LLMs like Mistral and Phi on popular AI benchmarks, while GPT-4, Claude, Gemini, and Llama stood their ground.

**Robotics and Embodied AI**

- **Tesla Optimus Update**: [@DrJimFan](https://twitter.com/DrJimFan/status/1787154880110694614) congratulated the Tesla Optimus team on their update, noting their human data collection farm is Optimus' biggest lead with best-in-class hands, teleoperation software, sizeable fleet, and carefully designed tasks & environments.
- **Open-Source Robotics with LeRobot**: [@ClementDelangue](https://twitter.com/ClementDelangue/status/1787474711582003702) welcomed LeRobot by @remicadene and team, signaling a shift towards open-source robotics AI. 
- **DrEureka from Nvidia**: [@adcock_brett](https://twitter.com/adcock_brett/status/1787151046713786421) shared Nvidia's 'DrEureka', an LLM agent that automates writing code to train robot skills, used to train a robot dog's skills in simulation and transfer them to real-world zero-shot.

**Multimodal AI and Hallucinations**

- **Multimodal LLM Hallucinations Overview**: [@omarsar0](https://twitter.com/omarsar0/status/1787510195922346154) shared a paper that presents an overview of hallucination in multimodal LLMs, discussing recent advances in detection, evaluation, mitigation strategies, causes, benchmarks, metrics, and challenges.
- **Med-Gemini from Google**: [@adcock_brett](https://twitter.com/adcock_brett/status/1787151219149926801) reported Google's introduction of Med-Gemini, a family of AI models fine-tuned for medical tasks, achieving SOTA on 10 of 14 benchmarks from text, multimodal, and long-context applications.

**Emerging Architectures and Training Techniques**

- **Kolmogorov-Arnold Networks (KANs)**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1787258503897157735) highlighted a paper proposing KANs as alternatives to MLPs for approximating nonlinear functions, outperforming MLPs and possessing faster neural scaling laws without using linear weights.
- **LoRA for Parameter-Efficient Finetuning**: [@rasbt](https://twitter.com/rasbt/status/1787467605718008228) implemented LoRA from scratch to train a GPT model for 98% accuracy in SPAM classification, noting LoRA as a favorite technique for parameter-efficient finetuning of LLMs.
- **Hybrid LLM Approach with Expert Router**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1787450115566747905) shared a paper on a cost-efficient hybrid LLM approach that uses an expert router to direct "easy" queries to a smaller model for cost reduction while maintaining quality.

**Benchmarks, Frameworks, and Tools**

- **TorchScript Model Export from PyTorch Lightning**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1787461157395022020) noted that exporting and compiling models to TorchScript from PyTorch Lightning is smooth with the `to_torchscript()` method, enabling model serialization for non-Python environments.
- **Hugging Face Inference Endpoints with Whisper and Diarization**: [@_philschmid](https://twitter.com/_philschmid/status/1787487522978717915) created an optimized Whisper with speaker diarization for Hugging Face Inference Endpoints, leveraging flash attention, speculative decoding, and a custom handler for 4.15s transcription of 60s audio on 1x A10G GPU.
- **LangChain for Complex AI Agents**: [@omarsar0](https://twitter.com/omarsar0/status/1787513175660806488) shared a free 2-hour workshop on building complex AI agents using LangChain for automating tasks in customer support, marketing, technical support, sales, and content creation.

**Trends, Opinions, and Discussions**

- **LLMs as a Commodity**: [@bindureddy](https://twitter.com/bindureddy/status/1787507453023994251) argued that LLMs have become a commodity, and even if GPT-5 is fantastic, other major players will catch up within months. Inference prices will trend down, and the winning LLM changes every few weeks. The best strategy is to use an LLM-agnostic service and move on from foundation models to building AI agents.
- **Literacy and Technology**: [@ylecun](https://twitter.com/ylecun/status/1787392175522672664) shared an observation on shifting attitudes towards reading and technology over time, from "why don't you plow the field instead of reading books?" in 1900 to "why don't you watch TV instead of being on your tablet?" in 2020.
- **Funding Fundamental Research**: [@ylecun](https://twitter.com/ylecun/status/1787041840484557203) argued that almost all federal funding to universities goes to STEM and biomedical research, with very little to social science and essentially zero to humanities. Cutting these funds would "kill the golden goose" and potentially cost lives.

---

# AI Reddit Recap

> Across r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity. Comment crawling works now but has lots to improve!

**AI Development and Capabilities**

- **Tesla Optimus advancements**: In /r/singularity, a new video showcases the latest capabilities of Tesla's Optimus robot, including [**fine tactile and force sensing in the hands**](https://v.redd.it/6ll8ixioekyc1). Discussions revolve around the robot's current speed limitations and potential for 24/7 operation in factories once it reaches "20x rate" of human workers.
- **Sora AI video rendering**: In /r/singularity, Sora, an AI system, demonstrates the ability to [**render a video while changing a single element**](https://v.redd.it/6h46hqbtenyc1), although this feature is still in the research phase and not yet publicly available.
- **GPT-4 trained robo-dog**: In /r/singularity, a robo-dog, trained using GPT-4, [**showcases its ability to maintain balance on a rolling and deflating yoga ball**](https://youtu.be/d5mdW1yPXIg), demonstrating advancements in AI-powered robotics and balance control.
- **Compute power and AI milestones**: In /r/singularity, Microsoft CTO Kevin Scott suggests that the [**common factor in AI milestone achievements is the use of more compute power**](https://www.reddit.com/r/singularity/comments/1cklexf/compute_is_all_you_need_microsoft_cto_kevin_scott/). Discussions on the potential for Llama 3 400b to outperform GPT-4 due to training on 25,000 H100s compared to GPT-4's reported 10,000 A100s.
- **LLaMA 70B performance**: In /r/singularity, a user reports running Llama 3 70B on a 7-year-old PC with a 4-year-old 3090 GPU, [**achieving better responses than GPT-4 and Claude 3 in some cases**](https://www.reddit.com/r/singularity/comments/1ckq5k8/llama_70b_q5_works_on_24gb_graphics_cards_and_the/). The post highlights the implications of having a highly intelligent AI that doesn't require an internet connection and can provide high-quality outputs.

**Societal Impact and Concerns**

- **Public awareness of AI-generated images**: In /r/singularity, a survey reveals that [**half of Americans are unaware that AI can generate realistic images of people**](https://i.redd.it/bn5wjsqabkyc1.png), raising questions about how many AI-generated images people have encountered without realizing it. Comments discuss the general lack of knowledge among the American public.
- **AI and abundance for all**: In /r/singularity, a post questions the belief that [**AI will lead to abundance for all, arguing that AI will likely entrench existing power structures and increase inequality**](https://www.reddit.com/r/singularity/comments/1cl3bgq/why_do_people_here_think_ai_will_lead_to/). The author suggests that the transition will be gradual, with unemployment and cost of living increasing slowly over time until a catastrophe occurs.
- **Warren Buffett's concerns about AI**: In /r/StableDiffusion, Warren Buffett [**compares AI to the atomic bomb, highlighting its potential for scamming people and expressing concerns about the power of AI**](https://www.reddit.com/r/StableDiffusion/comments/1cl69r7/warren_buffett_compares_ai_to_the_atomic_bomb/). Comments discuss the dual nature of AI, comparing it to the advent of electricity, with both positive and negative implications.

**AI Applications and Developments**

- **AI in medical notetaking**: In /r/singularity, an Ontario family doctor reports that [**AI-powered notetaking has significantly improved her work and saved her job**](https://globalnews.ca/news/10463535/ontario-family-doctor-artificial-intelligence-notes/), highlighting the potential for AI to assist in medical documentation.
- **Optimus hand advancements**: In /r/singularity, Elon Musk announces that the [**new Optimus hand, set to be released later this year, will have 22 degrees of freedom (DoF), an increase from the previous 11 DoF**](https://x.com/elonmusk/status/1787157110804910168).
- **Future of AI training and inference**: In /r/singularity, Nvidia CEO predicts that in the future, [**AI training and inference will be a single process, allowing AI to learn while interacting with users**](https://www.youtube.com/watch?v=oNwoA5akBlg). The video is recommended as interesting content for those following AI developments.

**Memes and Humor**

- **AI training and weird results**: In /r/StableDiffusion, a meme image suggests that [**training AI on unusual or unconventional images leads to weird results**](https://i.redd.it/1a7wf3um6myc1.jpeg).
- **Delicate banana**: In /r/StableDiffusion, a [**humorous image post features a delicate banana**](https://i.redd.it/f31p59ljwlyc1.jpeg).
- **Safety team suggestions**: In /r/StableDiffusion, a video meme depicts a prank where [**a person's chair is pulled out from under them, causing them to fall**](https://v.redd.it/urf2bp29jqyc1). Comments discuss the dangerous nature of the prank and the potential for serious injury.

---

# AI Discord Recap

> A summary of Summaries of Summaries

- **Llama3 GGUF Conversion Challenges**: Users encountered issues converting **Llama3** models to GGUF format using llama.cpp, with training data loss unrelated to precision. Regex mismatches for new lines were identified as a potential cause, impacting platforms like [ollama and lm studio](https://github.com/ggerganov/llama.cpp/issues/7062). Community members are collaborating on fixes like [regex modifications](https://github.com/ggerganov/llama.cpp/pull/6965).

- **GPT-4 Turbo Performance Concerns**: OpenAI users reported significant **latency increases** and confusion over **message cap thresholds** for GPT-4 Turbo, with some experiencing [5-10x slower response times](https://discord.com/channels/974519864045756446/1001151820170801244/1235851459891957821) and caps between 25-50 messages. Theories include dynamic adjustments during peak usage.

- **Stable Diffusion Installation Woes**: Stability.ai community members sought help with **Stable Diffusion setups failing to access GPU resources**, encountering errors like ["RuntimeError: Torch is not able to use GPU"](https://discord.com/channels/1002292111942635562/1002292112739549196/1235849532609265724). Discussions also covered the lack of comprehensive, up-to-date **LoRA/DreamBooth/fine-tuning tutorials**.

- **Hermes 2 Pro Llama 3 Impresses with Context**: **Hermes 2 Pro Llama 3** showcased ~32k context on a 32GB Nvidia v100 Tesla using **vLLM** and RoPE scaling, with [perfect 16k token recall and no degradation](https://discord.com/channels/1053877538025386074/1149866623109439599/1235849078479519855). Editing `config.json` and the rope scaling factor enables extended context.

- **Perplexity AI's Pages Feature Garners Attention**: Perplexity AI's new **Pages feature** for comprehensive report creation generated buzz, while users expressed [frustration over the 50 message per day limit on Claude 3 Opus](https://discord.com/channels/1047197230748151888/1047649527299055688/1235849900500058132) compared to GPT-4 Turbo and Sonnet. Discussions also covered Perplexity's shift from unlimited to limited messages.

- **LM Studio Enables Headless Mode**: LM Studio users leveraged the `lms` CLI tool for **headless operation** alongside the GUI, troubleshooting memory anomalies and [strategizing for smooth server-side deployments](https://discord.com/channels/1110598183144399058/1110598183144399061/1235890438762791002) without VRAM consumption via RDP. Fine-tuning bottlenecks were also discussed, with a member reporting success fine-tuning for 8 hours on a 128GB M3 Max MacBook Pro.

- **CUDA Compiling and Multi-GPU Training Challenges**: CUDA developers encountered issues with `nvcc 11.5` throwing errors for **bfloat16 operations on older GPUs**, with a [fix proposed](https://github.com/karpathy/llm.c/pull/353) to manually handle arithmetic for backward compatibility. Recent commits also caused **multi-GPU training hangs**, as reported in [Issue #369](https://github.com/karpathy/llm.c/issues/369), with a separate branch maintaining functionality.

- **Mojo Compiler and Type System Evolution**: Mojo's nightly compiler update brought changes to align with current practices, moving away from **80-column width** and transitioning to **register passable types**. Discussions touched on phasing out `OptionalReg` in favor of traits indicating register passability, as detailed in the [changelog](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md).

- **HuggingFace Community Highlights**: Notable projects in the HuggingFace community include [Moondream 2 batch processing](https://huggingface.co/spaces/Csplk/moondream2-batch-processing), [FLUENT's newest iteration](https://huggingface.co/spaces/fluently/Fluently-Playground), a [Portuguese translation of HF Audio course chapters](https://iatalk.ing/destaques-comunidade-hugging-face/), and a [BLIP fine-tune for long captions](https://huggingface.co/spaces/unography/image-captioning-with-longcap). A comprehensive list is available in the [community highlights](https://iatalk.ing/destaques-comunidade-hugging-face/).

- **Eleuther Ponders Transformer Chess Prowess**: An [arXiv paper](https://arxiv.org/abs/2402.04494) showcasing a 270M parameter transformer model outperforming AlphaZero's policy and value networks in chess without domain-specific algorithms sparked discussions on the impact of scale on strategy games within the Eleuther community.

---



# PART 1: High level Discord summaries




## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

**GGUF Conversion Hiccups for Llama3**: The Unsloth community encountered **conversion issues** with Llama3 models when using llama.cpp, notably affecting training data when transitioning to GGUF format. Issues weren't limited to FP16 conversions, implying deeper underlying problems than just precision loss.

**New Lines, Big Problems**: A recurrent theme in the glitches was linked to new line tokenization, with different behaviors across regex libraries leading to erratic tokenizer.json patterns. Potential solutions involving regex modifications were explored to fix the GGUF conversion challenges.

**Llama Variant Takes on Genomic Data**: The introduction of the **[LLaMA-3-8B-RDF-Experiment](https://huggingface.co/M-Chimiste/Llama-3-8B-RDF-Experiment)** model by M.chimiste marks a push towards integrating LLMs with genomic data and knowledge graph construction.

**Demand for Vision-Language Model Tuning Tools**: Community request surfaced for a generalized method to fine-tune Language-Vision Models (LVLM), demonstrated by a member's interest in supporting **Moondream**, as detailed in their [GitHub notebook](https://github.com/vikhyat/moondream/blob/main/notebooks/Finetuning.ipynb).

**Showcasing and Sharing Platform Growth**: Proposals for a separate discussion channel on deploying large language models (LLMs) highlight a demand for shared learning. This aligns with showcases like Oncord's integration of Unsloth AI for web development AI tools and the release of models that enhance Llama-3 capabilities.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

**Perplexity AI Pulls Ahead with Pages**: Perplexity AI's new **Pages feature** garners attention for its ability to create comprehensive reports. Meanwhile, a healthy skepticism surrounds the potential of **GPT-5** as engineers discuss the diminishing returns on investment.

**AGI Concept Sparks Debate**: The AI community on Discord is locked in a debate over the definition of AGI and whether AI models like **ChatGPT** are pioneering versions of AGI. Interest in AI-generated music indicates a growing appetite for creative AI applications, with reference to services like **Udio**.

**Performance Frustration Hits GPT-4 Turbo**: Significant increases in response **latency** are reported for **GPT-4 Turbo**, and users are seeking clarity about inconsistent **message cap thresholds**, suggesting possible dynamic adjustments during peak times.

**Prompt Engineering Challenges and Strategies**: Engineers share experiences and resources, recommend **"Wordplay" by Teddy Dicus Murphy** for prompt-crafting insights, and delve into the intricacies of using **logit bias** to manipulate token probabilities in the OpenAI API.

**Fine-Tuning AI for Queries**: A lively discussion revolves around fine-tuning models to **generate questions** rather than answers, including strategies for improving **GPT-4-TURBO** prompts for product information extraction, backed by a [logit bias tutorial](https://help.openai.com/en/articles/5247780-using-logit-bias-to-alter-token-probability-with-the-openai-api).



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **GPU Troubles Take Center Stage**: Members report difficulties with **Stable Diffusion** installations failing to access GPU resources, highlighted by errors like "RuntimeError: Torch is not able to use GPU."

- **Stable Diffusion 3 Rumors Stir the Pot**: Anticipation bubbles around the release of **Stable Diffusion 3**, sparking debates on the implications of its potential delay, while skeptics question its arrival altogether.

- **The Finetuning Tutorial Void**: The community voices frustration over a shortage of up-to-date, comprehensive tutorials for techniques like **LoRA/DreamBooth/fine-tuning**, which many find are either antiquated or skimpy on details.

- **Quest for Unique Faces**: A member inquires about strategies to train AI for generating unique, realistic-looking faces, wondering whether to use **LoRa** on multiple faces or to train on a generated random face as the foundation.

- **Open-Source Obstacles Discussed**: Conversations turn to the authenticity of **Stable Diffusion**'s open-source commitments, with concerns about potential future gatekeeping of high-quality models, checkpoints, and training intricacies.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **SVMs Still Kick in AI Circles**: Discord members clarified that **SVM** stands for **Support Vector Machine** amidst technical chitchat.
- **Anticipation for Meta-Llama-3-120B-Instruct**: The **Meta-Llama-3-120B-Instruct** Hugging Face model sparked discussions on its potential, with calls for comprehensive benchmarking rather than relying on mere hype.
- **Deployment Dilemmas**: Users debated serverless **Llama** limitations, whilst discussing better GPU options with sufficient VRAM, like **Azure's NC80adis_H100_v5**, for handling large-context task demands.
- **Hermes 2's Memorable Performance**: The **Hermes 2 Pro Llama 8B** demonstrated an impressive ~32k extended context capacity with no noticeable degradation, showing perfect recall at 16k on a 32GB Nvidia v100 Tesla.
- **Cynde Contributes to Data Farming**: An update on **Cynde** was shared, marking the completion of its core implementation. Enthusiasm for this framework for intelligence farming is evident, with **[Cynde's repository](https://github.com/Neural-Dragon-AI/Cynde)** welcoming contributors.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Pages Beta No More Open Applications**: The beta tester application phase for **Pages** has concluded due to sufficient participant enrollment. Future updates on **Pages** will be communicated accordingly.

- **Prominent Discussions on Perplexity AI's Performance and Limitations**: Members experienced **slow response times** with the Claude 3 model and have expressed **frustration over the 50 messages per day limitation** on the Claude 3 Opus model. While comparing Opus with **GPT-4 Turbo** and **Sonnet**, users also expressed concerns over Perplexity's shift from unlimited to **limited message capabilities**.

- **Exploring AI for Creative and Novelty Uses**: The **Perplexity AI community** is actively exploring the platform's abilities in **image generation**, **emulating writing styles from novels**, and **diverse searches** such as uncovering the history of BASIC programming language or delving into Perplexity's own history.

- **API Adventures and Agile Adjustments**: Users discussed model transitions, specifically from **sonar-medium-online to llama-3-sonar-large-32k-online**, and **queried about potential billing inconsistencies**. The conversation also included successes and troubles with **AI result optimization** and suggestions for creating a **minimum-code Telegram bot** using Perplexity API.

- **Multi-Channel Search Query Sharing**: The community shared multiple **search queries and outcomes**, engendering discussions about **Perplexity’s effective use and the depth** of insights it can provide. Such explorations were wrapped in a variety of contexts, ranging from **programming history to proprietary technological insights**.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Headless Cauldron Brews Progress**: Engineers are utilizing LM Studio's CLI tool, `lms`, for **headless operation** alongside GUI versions, working through memory consumption anomalies and discussing tactics for smooth server-side deployments without consuming VRAM through RDP.

- **Fine-tuning Finesse & Model Mishaps**: Members troubleshoot fine-tuning bottlenecks, sharing **success stories** of long fine-tuning sessions on hardware like the 128GB M3 Max MacBook Pro, and discuss the **inconsistent output** issues plaguing models like Llama 3. 

- **Interactive Intents & AI Memory Quirks**: Users express a confounding observation that language models might hold onto the context of deleted prompt elements, suggesting potential **bugs or a misunderstanding** of model behavior. They explore interactive techniques to **personalize writing styles** and enable "scoped access" to document parts for LLMs.

- **Role-Play Without Limits? Not So Fast**: A vivid conversation sparkles around the intersection of AI and **RPGs**, with users aiming to train AIs as Dungeon Masters for D&D, indicating that existing systems tangle with content moderation, which can impact the story's darkness and depth.

- **ROCm Raves & Linux Enthusiasm**: Updates to ROCm prove resilient, but discussions also broach the challenges **converting models** and sending longer sequences for embeddings. The dialog shifts toward community interest in contributing to a **Linux ROCm build**, hinting at further engagement if the project sought more open-source collaboration.

- **AI on the Hardware Frontier**: Members plunge into heated hardware exchanges, contrasting the **appropriateness of older GPUs** like the Tesla P40 over the GRID K1 and geeking out on multi-GPU setup nuances for AI-centric home labs. The **nitty-gritty** spreads from server hardware acquisitions to cooling, power, and driver compatibility issues.
  
- **LM Studio's Latest Line-up**: The `lmstudio-community` repo has been **updated with CodeGemma 1.1** and **Nvidia's ChatQA 1.5 models**, with the former eliciting keen anticipation and the latter offering specialized models tailored for context-based Q/A applications.



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

**Backpack Packs a Punch**: [BackPACK](https://backpack.pt/), a PyTorch extender for extracting additional information from backward passes, has been discussed, highlighting its potential for PyTorch developers. Details are in the publication "BackPACK: Packing more into Backprop" by Dangel et al., 2020.

**DoRA Delivers on Fusion**: A new **fused DoRA layer implementation** decreases the number of individual kernels and has been optimized for GEMM and reduction operations, detailed in a [GitHub pull request](https://github.com/pytorch/ao/pull/216). Enthusiasm was noted for upcoming benchmarks focused on these enhancements.

**Custom CUDA Extensions Customization**: Members discussed best practices for installing custom PyTorch/CUDA extensions, sharing multiple GitHub pull requests like [PR#135](https://github.com/pytorch/ao/pull/135) and a sample `setup.py` for reference, aiming for cleaner installation processes.

**Streaming Ahead with CUTLASS** Interest has bubbled around stream-K scheduling techniques used in CUTLASS, with suggestions of diving deeper into its workings in a future talk.

**GPU Communication Goes to School**: Upcoming sessions on GPU Collective Communications with **NCCL** have been announced, with a focus on distributed ML concepts.

**Must-Read ML Systems Papers**: For newcomers to machine learning systems, an [ML Systems Onboarding list](https://github.com/cuda-mode/awesomeMLSys) on GitHub provides a curated selection of informative papers.

**Overcoming CUDA Compiling Conundrums**: Issues with CUDA compilers like `nvcc 11.5` throwing errors for operations in bfloat16 have been addressed in a [fix proposal](https://github.com/karpathy/llm.c/pull/353), aiming to support older GPUs and toolkits. Multi-GPU training hangs have also been discussed, linked to [Issue #369](https://github.com/karpathy/llm.c/issues/369), with a separate branch maintaining functionality. 

**LLaMa's Lean Learning**: Discussions around memory efficiencies during **LLaMa 2 70B model training** highlighted configurations that allow for reduced memory usage. A tool named **HTA** was mentioned for pinpointing performance bottlenecks in PyTorch.

**Post-training Peaks with Quantization**: A [YouTube video](https://youtu.be/0VdNflU08yA?feature=shared) was shared, detailing the process and benefits of quantization in PyTorch.

**GreenBitAI Goes Global**: A toolkit called [green-bit-llm](https://github.com/GreenBitAI/green-bit-llm) was introduced for fine-tuning and inferencing GreenBitAI's language models. Attention was drawn to BitBlas for rapid 2-bit operation gemv kernels, along with a unique approach to calculating gradients captured in the GreenBitAI's toolkit.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

**Tune in to Mojo Livestream for MAX 24.3 Updates**: Modular's new livestream video titled "[Modular Community Livestream - New in MAX 24.3](https://www.youtube.com/watch?v=kKOCuLy-0UY)" invites the community to explore the latest features of MAX Engine and Mojo, along with an introduction to the MAX Engine Extensibility API. 

**Community Projects Zoom Ahead**: Noteworthy updates include NuMojo's improved performance and the introduction of [Mimage](https://github.com/fnands/mimage) for image parsing. The Basalt project also reached a milestone of 200 stars and released new [documentation](https://basalt-docs.vercel.app/).

**Mojo Compiler Evolves**: Mojo compiler sees [nightly updates](https://github.com/modularml/mojo/pull/2498/files) with changes to better fit current practices, such as the move away from 80-column width and transitioning to types more suited for register passability. 

**AI Engineers Seek Don Hoffman's Consciousness Exploration**: Interest in Donald Hoffman's work at UCI linked to consciousness research correlates with AI, as parallels are drawn between sensory data limitations seen in split-brain patients and AI hallucinations.

**Mojo's Growing Ecosystem & Developer Guidance**: Discussion on contribution processes to Mojo, inline with [GitHub's pull request guidelines](https://github.com/modularml/mojo/pull/2457), and insights into the development workflow with [tutorials on parameters](https://github.com/rd4com/mojo-learning/blob/main/tutorials/use-parameters-to-create-or-integrate-workflow.md) demonstrate the active support for contributors to the rapidly expanding Mojo ecosystem.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

**Moondream and BLOOM Make Waves**: The **HuggingFace** community has spotlighted new advancements including **[Moondream 2 batch processing](https://huggingface.co/spaces/Csplk/moondream2-batch-processing)** and **FLUENT's newest iteration**, as well as tools for multilingual support. Particularly noteworthy is the **BLOOM multilingual chat** and **AutoTrain's support for YAML configs**, simplifying the training process for machine learning newcomers. Check out the [community highlights](https://iatalk.ing/destaques-comunidade-hugging-face/).

**When Audio Models Sing**: There's interest in audio diffusion models for generative music with **Whisper** being fine-tuned for Filipino ASR, prompting discussions on optimization. However, a user faced challenges converting PyTorch models into TensorFlow Lite due to size limits.

**AI's Frontline**: Cybersecurity took center stage as the **Hugging Face Twitter** account was compromised, underlining the need for robust AI-related security. Members also exchanged GPU utilization tips for variance in training times between setups.

**Visions of Quantum and AI Unions**: In **computer vision**, the emphasis was on improving traditional methods like YOLO for gap detection in vehicle parts and adapting models like CLIP for image recognition with rotated objects. **GhostNet's pre-trained weights** were sought after, and CV members pondered the contemporary relevance of methods like SURF and SIFT.

**Graph Gurus Gather**: Recent papers on using **LLMs** with graph machine learning propose novel ways to integrate the two, with a **paper](https://arxiv.org/abs/2404.19705)** specifically teaching LLMs to retrieve information only when needed via the `<RET>` token. The **[reading group](https://discord.com/events/879548962464493619/1234913780048203856)** provided additional resources for those eager to learn more.

**Showcasing Synthesis and Applied AI**: From the **[#i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1235893212393111594)** section, there's the launch of tools like **Podcastify** and **OpenGPTs-platform**, along with models like **shadow-clown-BioMistral-7B-DARE** using **mergekit**.

**NLPer's Quandaries and Queries**: In **[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1235892774528880641)**, a user offered compensation for custom training on **Mistral-7B-instruct** and concerns were raised about LLMs evaluating other LLMs. The **GEMBA** metric for translation quality using GPT 3.5+ was introduced, with a link provided to [learn more](https://aclanthology.org/2023.eamt-1.19/).



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

**Integrating OpenInterpreter with Groq LLM**: Engineers discussed challenges with integrating **Groq LLM** onto Open Interpreter, highlighting issues such as uncontrollable output and erroneous file creation. The connection command shared was `interpreter --api_base "https://api.groq.com/openai/v1" --api_key "YOUR_API_KEY_HERE" --model "llama3-70b-8192" -y --max_tokens 8192`.

**Microsoft Hackathon Seeks Open Interpreter Enthusiasts**: A team is forming to participate in the Microsoft Open Source AI Hackathon utilizing Open Interpreter; the event promises to offer hands-on tutorials and the sign-up details are available [here](https://lu.ma/iu1wijgd).

**Open Interpreter Gets an iOS Reimagining**: Discussions revolved around reimplementation of TMC protocol for iOS on Open Interpreter and troubleshooting issues with setting up with Azure Open AI models, with one member sharing a GitHub repository link for the iOS app in development [here](https://github.com/OpenInterpreter/01/tree/main/software/source/clients/mobile).

**Local LLMs Challenge Developers**: Personal testings on local LLMs like **Phi-3-mini-128k-instruct** were shared, indicating significant performance variances and calling out for better optimization methods in future implementations.

**AI Vtuber's STT Conundrum**: Implementing Speech-to-Text for AI powered virtual streamers brought up practical challenges, with engineers considering using trigger words and working towards AI-driven Twitch chat interactions through a separate LLM instance, aiming for comprehensive responses. For those tackling similar integrations, a member pointed to a **main.py** file on their GitHub as a resource.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Chess Grandmasters Beware, Transformers Are Coming**: A [new study](https://arxiv.org/abs/2402.04494) reveals a 270M parameter transformer model surpassing AlphaZero's policy and value networks in chess without domain-specific algorithms, raising questions on scale's effectiveness in strategy games.

- **LLM Research Flourishes with Multilingualism and Prompting Techniques**: Research highlights include a study on LLMs handling multilingual inputs and the potential of "Maieutic Prompting" for working with inconsistent data despite skepticism about its practicality. Contributions in this area provided insights and links to papers such as [How Mixture Models Handle Multilingualism](https://arxiv.org/abs/2402.18815v1) and methods to counteract LLM vulnerabilities, including *The Instruction Hierarchy* [paper](http://arxiv.org/abs/2404.13208).

- **Model Performance Under the Microscope**: The scaling laws for transfer learning indicate that pre-trained models improve on fixed-sized datasets via effective transferred data, resonating with the community's efforts to determine accurate measures of LLM in-context learning and performance evaluation methods.

- **Interpreting Transformers and Improving Deployability**: A primer and survey on interpreting transformer-based LLMs have been shared, alongside discussions on cross-model generalization. There's active interest in resolving weight tying issues in models like **Phi-2** and **Mistral-7B** and clarifying misunderstandings regarding weight tying in notable open models.

- **Community Engagement with ICLR and Job Searches**: Preparations for an in-person meet-up at ICLR are unfolding despite travel challenges, and community support is evident with members sharing employment resources and experiences from engaging with projects such as **OSLO** and the **Polyglot** team.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **New Kids on the Llama Block**: The [Llama 3 Lumimaid 8B](https://openrouter.ai/models/neversleep/llama-3-lumimaid-8b) model has been released with an [extended version](https://openrouter.ai/models/neversleep/llama-3-lumimaid-8b:extended) also available, while the [Llama 3 8B Instruct Extended](https://openrouter.ai/models/meta-llama/llama-3-8b-instruct:extended) sees a price reduction. A brief downtime was announced for the Lynn models due to server updates.

- **Beta Testers Wanted for High-Stakes AI**: Rubik's AI Pro, an advanced research assistant and search engine, is seeking beta testers with 2 months of premium access including models like GPT-4 Turbo and Mistral Large. The project can be accessed [here](signup.php) with the promo code `RUBIX`.

- **Mix and Match Models**: Community members reported that **Gemini Pro** is now error-free and discussed potential hosts for **Lumimaid 70B**. Models like **Phi-3** are sought after, but availability is scarce. Model precision varies across providers, with most using **fp16** and some using quantized **int8**.

- **Mergers and Acquisitions**: A conversation highlighted a newly created **self-merged version of Meta-Llama 3 70B** on Hugging Face, spurring debates about the effectiveness of self-merges versus traditional layer-mapped merges.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

**Boosting Agent Smarts**: **LlamaIndex 0.10.34** ushers in **introspective agents** capable of self-improvement through reflection mechanisms, detailed in a [notebook](https://t.co/X8tJGXkcPM) which comes with a content warning for sensitive material.

**Agentic RAG Gets an Upgrade**: An informative video demonstrates the integration of LlamaParse + Firecrawl for crafting **agentic RAG systems**, and the release can be found through [this link](https://t.co/wR35iYIKjo).

**Trust-Scored RAG Responses**: "Trustworthy Language Model" by @CleanlabAI introduces a scoring system for the trustworthiness of RAG responses, aiming to assure accuracy in generated content. For more insights, refer to their announcement [here](https://t.co/KW1XsllRqQ).

**Local RAG Pipeline Handbook Hits Shelves**: For developers seeking independence from cloud services, a manual for setting up a fully local RAG pipeline with LlamaIndex is unveiled, promising a deeper dive than quickstart guides and accessible [here](https://t.co/2RCvaxOzKo).

**Hugging Face, Now Hugging LlamaIndex Tightly**: LlamaIndex declares support for **Hugging Face TGI**, enabling optimal deployment of language models on Huggingface with enhanced features like **function calling** and improved latency. Shed light on TGI's new capabilities [here](https://t.co/3vGpxcbP18).

**Creating Conversant SQL Agents**: AI engineers are contemplating the use of **HyDE** to craft **NL-SQL bots** for databases brimming with tables, eyeing ways to elevate the precision of SQL queries by the LLM; meanwhile, introspective agent methodologies are making waves, with further reading at [Introspective Agents with LlamaIndex](https://medium.com/ai-artistry/introspective-agents-with-llamaindex-777d018f791d).



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

**Hermes 2 Pro Llama 3 Speed Test Results**: **Hermes 2 Pro Llama 3** has showcased impressive **inference speed** on an Android device with 8GB RAM, boosted by enhancements in **llama.cpp**.

**Anime’s Role in AI Conversations**: Members humorously discussed the rise of **anime** as it relates to increasing capabilities in **AI question-answering** and **image generation** tasks.

**Gradio Customization Achievements**: Adjustments in **Gradio** now allow dynamic configuration set through a [YAML file](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1591), enabling the setting of privacy levels and server parameters programmatically.

**Datasets for AI Training Spotlighted**: A new dataset containing 143,327 verified Python examples ([Python Dataset](https://huggingface.co/datasets/Vezora/Tested-143k-Python-Alpaca)) and difficulties in improving mathematical performance of Llama3, even with math-centric datasets, were discussed, highlighting dataset challenges in AI training.

**AI Training Platform Enhancements and Needs**: There was a call to refine **Axolotl's documentation**, particularly regarding merging model weights and model inference, accessible at [Axolotl Community Docs](https://axolotl.continuumlabs.pro/). Additionally, issues with gradient clipping configurations were addressed, and Phorm offered insights into customizing **TrainingArguments** for **gradient clipping** and the **chatbot prompt**.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Gary Rocks the Ableton**: A new work-in-progress Python project, [gary4live](https://github.com/betweentwomidnights/gary4live), integrates Python continuations with Ableton for live music performance, inviting contributors and peer review from the community.

- **Suno Scales Up Music Production**: Discussion about using **Suno** for music generation included comparisons with other setups like *Musicgen*, with an emphasis on Suno's tokenization process for audio and exploration on whether these models can automatically produce sheet music.

- **Token Talk**: Engaging deeply with music model token structures, participants navigated the token length and composition in audio synthesis, referencing but not detailing specific architectural designs from academic papers.

- **Breaking Barriers in Audio Synthesis**: The potential of direct audio integration into multimodal models was discussed, focusing on real-time replacement of audio channels and the importance of direct audio for enabling omnimodal functionality.

- **The Business Beat of Stable Audio**: Commercial use and licensing questions surfaced regarding stable audio model outputs, with a specific eye towards their real-time application in live performances and the possible implications for industries.



---



## [AI Stack Devs (Yoko Li)](https://discord.com/channels/1122748573000409160) Discord

- **Local Hardware Tackles AI**: Users can now use **llama-farm** to run **Ollama** locally on old laptops for processing LLM tasks without exposing them to the public internet. This was also linked to a GitHub repository with more details on its implementation ([llama-farm chat on GitHub](https://github.com/get-convex/llama-farm-chat)).

- **AI Cloud Independence Achieved**: Discussions indicated that using **Faraday** allows users to keep downloaded characters and models indefinitely, and running tools locally can circumvent cloud subscription fees, given a 6 GB VRAM setup. Local execution requires no subscription, acting as a potential budget-friendly option for tool usage.

- **Ubuntu Users Regain Control**: Installation problems with `convex-local-backend` on Ubuntu 18 were solved by downgrading to Node version 18.17.0 and updating Ubuntu as per a [GitHub issue](https://github.com/get-convex/convex-backend/issues/1). Dockerization was proposed as a potential solution to simplify future setups.

- **Simulated Realities Attract Spotlight**: An **AI Simulated Party** was featured at Mission Control in San Francisco, blending real and digital experiences. Additionally, the **AI-Westworld** simulation entered public beta, and a web app called **AI Town Player** was launched for replaying AI Town scenarios by importing sqlite files.

- **Clipboards and Beats Converge**: There was a call for collaboration to create a simulation involving hip-hop artists **Kendrick** and **Drake**. It demonstrates an interest in combining AI development with cultural commentary.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

**CLIP vs. T5: The Model Smackdown**: There's a spirited discussion about integrating [CLIP and T5 encoders](https://old.reddit.com/r/StableDiffusion/comments/1cgr74j/april_30th/l2bxv66/) for training AI models; while the use of both encoders shows promise, some argue using T5 alone due to prompt adherence issues with CLIP.

**Are Smaller Models the Big Deal?**: In the realm of model size, enhancement of smaller models is being prioritized, as evidenced by the focus on the 400M DeepFloyd, with technical conversations touching upon the challenges in scaling up to 8B models.

**Releasing SD3: Keep 'Em Waiting or Drop 'Em All?**: The community's reaction to Stability AI's hinted gradual rollout of SD3 models—from small to large—was a mix of skepticism and eagerness, reflecting on whether this release strategy meets the community's anticipation.

**LLama Embeds Strut into the Spotlight**: Debates over the efficacy of using LLama embeds in model training emerged, with some members advocating for their use over T5 embeds, and sharing resources like the [LaVi-Bridge](https://github.com/ShihaoZhaoZSH/LaVi-Bridge) to illustrate modern applications.

**From Concept to Application: A Data Debate**: The conversation dove into why synthetic datasets are favored in certain research over real-world datasets such as MNIST and ImageNet, alluding to the value of interpretability in AI methods and sharing resources like the [StoryDiffusion website](https://storydiffusion.github.io/) for insights.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

**Code Execution Finds an AI Buddy**: Enthusiastic dialogues emerged around using AI to execute generated code, highlighting methods like **Open Interpreter** and developing **custom tools** such as `CLITOOL`. These discussions are pivotal for those crafting more interactive and automated systems.

**Langchain Learns a New Language**: The Langchain library's expansion into the Java ecosystem via [langchain4j](https://github.com/langchain4j/langchain4j) marks a crucial step for Java developers keen to harness AI assistant capabilities.

**Langchain Gets a High-Performance Polish**: The coupling of **LangChain** and **Dragonfly** has yielded impressive enhancements in chatbot context management, as depicted in a [blog post](https://www.dragonflydb.io/blog/efficient-context-management-in-langchain-with-dragonfly) detailing these advancements.

**Decentralized Search Innovations**: The community is buzzing with the development of a decentralized search feature for **LangChain**, promising to boost search functionalities with a user-owned index network. The work is showcased in a recent [tweet](https://twitter.com/indexnetwork_/status/1786110169396429093).

**Singularity Spaces with Llama & LangGraph**: A contributor shared a [video](https://www.youtube.com/watch?v=vvW2dwvNm2Q) on *Retrieval-Augmented Generation* techniques without a vectorstore using **Llama 3**, while another enriches the dialogue with a [comparison](https://www.youtube.com/watch?v=UcD42NA2WoI) between **LangGraph** and **LangChain Core** in the execution realm.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

**Clojure Captures Engineer's Interest in Symbolic Programming**: Engineers are discussing the ease of using **Clojure** for symbolic programming compared to Python, suggesting the use of bounties to ramp up on *tinygrad*, and debating the merits of **Julia** over Clojure in the ML/AI space.

**tinygrad's UOps Puzzle Engineers**: A call for proposals was made to reformat *tinygrad's* textual UOps representation to be more understandable, potentially resembling llvm IR, alongside an explanation that these UOps are indeed a form of Static Single Assignment (SSA).

**Optimizing tinygrad for Qualcomm's GPU Playground**: It was highlighted that *tinygrad* runs efficiently on **Qualcomm GPUs** by utilizing textures and pixel shaders, with the caveat that activating DSP support might complicate the process.

**Single-threaded CPU Story in tinygrad**: Confirmation from **George Hotz** himself that *tinygrad* operates **single-threaded** on the CPU side, with no threads bumping into each other.

**Understanding tinygrad's Tensor Tango**: A user's curiosity about the `matmul` function and transposing tensors spurred explanations, and another user shared their [written breakdown](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/symbolic-mean.md) on computing symbolic mean within tinygrad.



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Json\Schema Skips a Beat with llamafile**: A clash between `json_schema` and **llamafile 0.8.1** prompted discussions, with a workaround using `--unsecure` suggested and hints of a permanent fix in upcoming versions.

- **In Search of Leaner Machine Learning Models**: The community exchanged ideas on lightweight AI models, where **phi 3 mini** was deemed too heavy and **Rocket-3B** was suggested for its agility on low-resource systems.

- **Clubbing Caches for Llamafile**: It was confirmed that **llamafile** can indeed utilize models from the **ollama cache**, potentially streamlining operations by avoiding repeated downloads, providing that GGUF file compatibility is maintained.

- **AutoGPT Goes Hand-in-Hand with Llamafile**: An integration initiative was shared, highlighting a draft pull request to meld **llamafile** with **AutoGPT**; setup instructions were posted at [AutoGPT/llamafile-integration](https://github.com/Mozilla-Ocho/AutoGPT/tree/draft-llamafile-support/autogpts/autogpt/llamafile-integration), pending maintainer feedback.

- **Choosing the Right Local Models for Llamafile**: Real-time problem-solving was spotlighted as a user managed to get **llamafile** up and running with locally cached **.gguf** files after distinguishing between actual model files and metadata.



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

**Mixtral Woes Spiral**: The **mixtral transformers** hit a snag due to bugs impacting finetune performance; references include [Twitter](https://twitter.com/kalomaze/status/1786869036946522256), [Gist](https://gist.github.com/kalomaze/661b79095fdd91df8a84802f7cb6f26a), and a [closed GitHub PR](https://github.com/huggingface/transformers/pull/30658). There's ambiguity whether the bug affects only training or generation as well, necessitating further scrutiny.

**Quantized LLaMA-3 Takes a Hit**: A Reddit post reveals quantization deteriorates LLaMA-3's performance notably compared to LLaMA-2, with a potentially enlightening [arXiv study](https://arxiv.org/abs/2404.14047) available. Meta's scaling strategy may account for LLaMA-3's precision reduction woes, while [GitHub PR #6936](https://github.com/ggerganov/llama.cpp/pull/6936#issuecomment-2083214112) and [Issue #7088](https://github.com/ggerganov/llama.cpp/issues/7088#issuecomment-2094933215) discuss potential fixes.

**Meet the New Model on the Block**: Conversations indicate **8x22b Mistral** is being leveraged for current engineering tasks, though no performance metrics or usage specifics were disclosed.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **AI Voices: So Real It's Unreal**: [The Atlantic](https://www.theatlantic.com/technology/archive/2024/05/elevenlabs-ai-voice-cloning-deepfakes/678288/) published an article discussing how **ElevenLabs** has created advanced AI voice cloning technology. Users expressed both fascinated and wary reactions to ElevenLabs' capabilities, with one showing disdain towards paywalls that limit full access to such content.
  
- **Prometheus 2: Judging the Judges**: A [recent arXiv publication](https://arxiv.org/abs/2405.01535) introduced **Prometheus 2**, a language model evaluator aligned with human and **GPT-4 judgments**, targeting transparency and affordability issues in proprietary language models. Although the paper notably omitted *RewardBench* scores where the model underperformed, there is keen interest in the community to test Prometheus 2's evaluation prowess.
  
- **Enigma of Classical RL**: Conversations in the **rl** channel featured curiosity about unexplored areas in classical reinforcement learning. Discussion put a spotlight on the importance of the value function in approaches like **PPO** and **DPO**, and emphasized its critical role in planning within RL systems.
  
- **The Mystery of John's Ambiguity**: In the **random** channel, members shared cryptic concerns about repeated success and joked about a certain "john's" ambiguous response to a proposal. The relevance and context behind these statements remained unclear.



---



## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord

- **Anthropic's Prompt Generator Makes Waves**: Engineers discussed a new **prompt generator tool** available in the **Anthropic console**, which may be useful for those seeking efficient ways to generate prompts.
- **Politeness Mode Test Run**: The tool's capability to *rephrase sentences politely* was tested, producing results that were well-received by members.
- **Deciphering the System's Mechanics**: Efforts are underway to understand how the tool's system prompt operates, with a focus on unraveling the secrets of the **k-shot examples** embedded within.
- **Extracting the Long Game**: There have been challenges in extracting complete data from the tool, with reports of system prompts being truncated, particularly during the extended *Socratic math tutor* example.
- **Leak the Secrets**: A commitment was made to share the full system prompt with the community once it has been successfully extracted in its entirety, which could be a resource for those interested in prompt engineering.



---



## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord

- **Fake It 'til You Make It**: A member is on the lookout for a **dataset of fabricated data** aimed at testing fine-tuning on **Llama 3** and **Phi3** models, implying that authenticity is not a requirement for their experiments.
- **Accelerating AI with Fast Compute**: **Fast compute grants** are up for grabs for Skunkworks AI projects that show promise, with further details available in a [recent tweet](https://twitter.com/PrimeIntellect/status/1786386588726960167).
- **Educational AI Content on YouTube**: An AI-related educational [YouTube video](https://www.youtube.com/watch?v=vvW2dwvNm2Q) was shared, potentially adding value to the community’s ongoing technical discussions.



---



## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

- **LLM Turns Error Logs into Enlightenment**: An approach that utilizes LLM to swiftly summarize errors after running a `conda activate` command has proven effective, with suggestions to integrate the method into the [LLM README](https://github.com/simonw/llm/blob/main/README.md) documentation.
- **Bash Magic Meets LLM Insights**: A newly crafted `llm-err` bash function is on the table, designed to feed command outputs directly into LLM for quick error diagnosis, further streamlining error troubleshooting for engineers.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Calling AI Experts in Austin**: A friendly hello was extended to AI professionals located in **Austin, TX**.
- **Finexov's Funding Frontier**: **Vivien** introduced **Finexov**, an AI platform aimed at simplifying the identification of **R&D** funding opportunities, already active with initial partnerships and support from the **Founder Institute** ([fi.co](https://fi.co/)).
- **Tech Leadership Hunt for Finexov**: Seeking a **CTO co-founder** with a strong **ML** background to pilot Finexov and gear up for the challenges of team-building and fundraising; preference for candidates based in Europe or Middle East, French speakers are a bonus.
- **Dubai Meetup on the Horizon**: Vivien signals a potential meetup in **Dubai** this June, inviting potential collaborators to discuss opportunities with Finexov.




---



## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **AI21 Labs Pushes the Envelope**: AI21 Labs indicated their ambition to expand their technology further. The staff encouraged community members to share their use cases and insights through direct messages.



---



## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord

- **Get Your Compute Loading**: Interested parties have a chance to gain **fast compute grants**; a tweet shared by a member calls for applications or nominations to award compute resources, beneficial for AI research and projects. [Check out the tweet for details](https://twitter.com/PrimeIntellect/status/1786386588726960167).



---

# PART 2: Detailed by-Channel summaries and links



**Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1235848656473690112)** (791 messages🔥🔥🔥): 

- **Discussions on Llama3 finetuning and GGUF conversion**: Users have been experimenting with finetuning Llama3 using [Unsloth](https://huggingface.co/unsloth), and converting the finetuned models to GGUF with various outcomes. Some reported issues with infinite generation post-conversion and were directed to keep tabs on a [GitHub issue](https://github.com/ggerganov/llama.cpp/issues/7062) highlighting problems with models converted to GGUF.
  
- **Inquiries on full finetuning with Unsloth**: A user was curious about the possibility of full finetuning (not just LoRA) using Unsloth, leading to discussions about possible VRAM savings and performance. Unsloth community members provided insights into how to potentially achieve this, referencing a [GitHub feature request](https://github.com/unslothai/unsloth).

- **Investigation into performance of heavily quantized models**: A user questioned the effectiveness of heavy quantization like 4 Bit Q2_K for a 7B model, with the recommendation to possibly use Phi-3 instead for low resource applications, underscoring the importance of choosing the right quant level for model performance.

- **Sharing of resources and troubleshooting Unsloth**: Users shared their experiences and offered advice on cloud providers like Tensordock for running Unsloth models, the usage of Unsloth Studio, as well as general tips on dealing with finetune datasets, quantization effects, and the use of different inference engines.

- **Uncertainties about fine-tuning low-resource languages with LLMS**: A user considering fine-tuning with LLMs for low-resource languages sought advice on the efficacy of LLMs versus models like T5. Community discussion highlighted the potential of models like Phi-3 for such tasks, with contributions addressing how to handle different aspects of the fine-tuning process.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/1XamvWYinY6FOSX9GLvnqSjjsNf">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1aqlNQi7MMJbynFDyOQteD2t0yVfjb9Zh?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://huggingface.co/papers/2402.05119">Paper page - A Closer Look at the Limitations of Instruction Tuning</a>: no description found</li><li><a href="https://huggingface.co/spaces/ggml-org/gguf-my-repo">GGUF My Repo - a Hugging Face Space by ggml-org</a>: no description found</li><li><a href="https://huggingface.co/unsloth/Phi-3-mini-4k-instruct">unsloth/Phi-3-mini-4k-instruct · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/spaces/NyxKrage/LLM-Model-VRAM-Calculator">LLM Model VRAM Calculator - a Hugging Face Space by NyxKrage</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1XamvWYinY6FOSX9GLvnqSjjsNflxdhNc?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://huggingface.co/unsloth/Phi-3-mini-4k-instruct-bnb-4bit">unsloth/Phi-3-mini-4k-instruct-bnb-4bit · Hugging Face</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/18o5u0k/comment/kefkdut/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button">Reddit - Dive into anything</a>: no description found</li><li><a href="https://huggingface.co/gradientai/Llama-3-8B-Instruct-Gradient-1048k">gradientai/Llama-3-8B-Instruct-Gradient-1048k · Hugging Face</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1ckvx9l/part2_confirmed_possible_bug_llama3_gguf/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1ckvx9l/part2_confirmed_possible">Reddit - Dive into anything</a>: no description found</li><li><a href="https://huggingface.co/unsloth">unsloth (Unsloth AI)</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/wiki#evaluation-loop---also-oom-or-crashing">Home</a>: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://huggingface.co/unsloth?search_models=llama-3-70b">unsloth (Unsloth AI)</a>: no description found</li><li><a href="https://github.com/IBM/unitxt">GitHub - IBM/unitxt: 🦄 Unitxt: a python library for getting data fired up and set for training and evaluation</a>: 🦄 Unitxt: a python library for getting data fired up and set for training and evaluation - IBM/unitxt</li><li><a href="https://github.com/g">Grizzly</a>: Grizzly has 9 repositories available. Follow their code on GitHub.</li><li><a href="https://github.com/unslothai/unsloth/wiki#finetuning-the-lm_head-and-embed_tokens-matrices">Home</a>: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://www.cerebras.net/press-release/cerebras-announces-third-generation-wafer-scale-engine">Cerebras Systems Unveils World’s Fastest AI Chip with Whopping 4 Trillion Transistors - Cerebras</a>: Third Generation 5nm Wafer Scale Engine (WSE-3) Powers Industry’s Most Scalable AI Supercomputers, Up To 256 exaFLOPs via 2048 Nodes</li><li><a href="https://huggingface.co/docs/transformers/v4.40.1/en/pad_truncation#padding-and-truncation">Padding and truncation</a>: no description found</li><li><a href="https://huggingface.co/gradientai/Llama-3-8B-Instruct-262k">gradientai/Llama-3-8B-Instruct-262k · Hugging Face</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=WxQbWTRNTxY&t=83s">How to Fine Tune Llama 3 for Better Instruction Following?</a>: 🚀 In today&#39;s video, I&#39;m thrilled to guide you through the intricate process of fine-tuning the LLaMA 3 model for optimal instruction following! From setting...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1ckcw6z/1m_context_models_after_16k_tokens/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth.git">GitHub - unslothai/unsloth: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory</a>: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/7062">Llama3 GGUF conversion with merged LORA Adapter seems to lose training data randomly · Issue #7062 · ggerganov/llama.cpp</a>: I&#39;m running Unsloth to fine tune LORA the Instruct model on llama3-8b . 1: I merge the model with the LORA adapter into safetensors 2: Running inference in python both with the merged model direct...</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/7062#issuecomment-2094875716">Llama3 GGUF conversion with merged LORA Adapter seems to lose training data randomly · Issue #7062 · ggerganov/llama.cpp</a>: I&#39;m running Unsloth to fine tune LORA the Instruct model on llama3-8b . 1: I merge the model with the LORA adapter into safetensors 2: Running inference in python both with the merged model direct...</li><li><a href="https://llama-hub.com/article_detail/060ef5ec-1fd6-4662-a428-6bbc6f3a4496">no title found</a>: no description found
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1235890587337494528)** (107 messages🔥🔥): 

- **Graphic Content Alert with LLaMA3**: A user reported inappropriate and graphic content generated by **LLaMa3** when prompted with an obscene query, questioning the level of censorship in the model. [Another user](https://www.github.com/status-check/status) found similar results, even when using system prompts to prevent such responses.
- **Fancy New Roles for Supporters**: In a brief confusion about support roles, a user learned that there is a new "**regulars**" role, and private supporter channels are available for those who become members or donate at least $10.
- **RTX 4090 Gets a Suprim(ary) Deal**: A new graphics card deals discussion highlighted the **MSi GeForce RTX 4090 SUPRIM LIQUID X** on sale for $1549, with a user urging others to take advantage of the offer. The card's compact size compared to other models sparked further debate.
- **Kendrick vs. Drake Dynamic**: Users discussed the recent developments in the Kendrick Lamar and Drake beef, indicating that Kendrick's track "Meet the Grahams" was released shortly after Drake's "Family Ties" causing a significant stir in the rap world.
- **Unsloth.ai on YouTube**: A conversation thread involved a user congratulating another on presenting to the PyTorch team, directing them to a [YouTube video](https://www.youtube.com/watch?v=MQwryfkydc0) from Unsloth.ai, hinting at further updates to be posted soon.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://paperswithcode.com/paper/x-lora-mixture-of-low-rank-adapter-experts-a">Papers with Code - X-LoRA: Mixture of Low-Rank Adapter Experts, a Flexible Framework for Large Language Models with Applications in Protein Mechanics and Molecular Design</a>: Implemented in 3 code libraries.</li><li><a href="https://www.reddit.com/r/buildapcsales/comments/1cljlba/gpu_msi_geforce_rtx_4090_suprim_liquid_x_24_gb/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=MQwryfkydc0">Unsloth.ai: Easily finetune &amp; train LLMs</a>: no description found</li><li><a href="https://huggingface.co/blog/mayank-mishra/padding-free-transformer">Saving Memory Using Padding-Free Transformer Layers during Finetuning</a>: no description found
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1235848585611051049)** (1215 messages🔥🔥🔥): 

- **Llama3 GGUF Conversion Issues Pinned Down**: Users found that GGUF conversion for Llama3 models using llama.cpp fails, resulting in altered or lost training data with no clear pattern of loss, regardless of using FP16 or FP32 conversion methods. These abnormalities occur even in F32, proving the issue is not tied to precision loss.
- **Possible Regex Mismatch for New Lines**: The problem may link to a regex library issue where `\n` sequences are improperly tokenized, potentially due to different regex library behaviors. The suggested fix modifies the tokenizer.json regex pattern for more compatibility across regex libraries, but concerns remain about the impact on different '\n' lengths.
- **Issues Exist Beyond GGUF**: Similar inference issues were found with AWQ in applications like ooba, pointing towards tokenizer or tokenization issues beyond just GGUF formatting. Unsloth's inference function seems to perform well, hinting at problems possibly specific to llama.cpp.
- **Multiple Platforms Impacted**: Platforms dependent on llama.cpp like ollama, and lm studio also face related bugs, with tokenization problems reported across different interfaces and potentially affecting a wide range of users and applications.
- **Community Cooperation Towards Solutions**: User contributions, including regex modifications, are being discussed and tested to provide temporary fixes for the gguf conversion troubles, with a focus on narrowing down whether issues are specific to the Unsloth fine-tuning process or the llama.cpp tokenization method.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/Orenguteng/Llama-3-8B-LexiFun-Uncensored-V1-GGUF">Orenguteng/Llama-3-8B-LexiFun-Uncensored-V1-GGUF · Hugging Face</a>: no description found</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharin">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/10NbwlsRChbma1v55m8LAPYG15uQv6HLo?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://huggingface.co/microsoft/Phi-3-mini-128k-instruct">microsoft/Phi-3-mini-128k-instruct · Hugging Face</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1cltac3/part3_cause_to_issue_found_possible_bug_llama3/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://huggingface.co/docs/datasets/en/loading#json">Load</a>: no description found</li><li><a href="https://github.com/xaedes/llama.cpp/tree/finetune-lora/examples/export-lora">llama.cpp/examples/export-lora at finetune-lora · xaedes/llama.cpp</a>: Port of Facebook&#39;s LLaMA model in C/C++. Contribute to xaedes/llama.cpp development by creating an account on GitHub.</li><li><a href="https://www.reddit.com/user/Dependent_Factor_204/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/casper-hansen/AutoAWQ/blob/main/examples/generate.py">AutoAWQ/examples/generate.py at main · casper-hansen/AutoAWQ</a>: AutoAWQ implements the AWQ algorithm for 4-bit quantization with a 2x speedup during inference. Documentation: - casper-hansen/AutoAWQ</li><li><a href="https://github.com/ggerganov/llama.cpp/discussions/5360">creating gguf model from lora adapter · ggerganov/llama.cpp · Discussion #5360</a>: I have a ggml adapter model created by convert-lora-to-ggml.py (ggml-adapter-model.bin). Now my doubt is how to create the complete gguf model out of these? I have seen using ./main -m models/llama...</li><li><a href="https://github.com/ScottMcNaught">ScottMcNaught - Overview</a>: ScottMcNaught has one repository available. Follow their code on GitHub.</li><li><a href="https://github.com/unslothai/unsloth/pull/371">llama.cpp failing by bet0x · Pull Request #371 · unslothai/unsloth</a>: llama.cpp is failing to generate quantize versions for the trained models. Error: You might have to compile llama.cpp yourself, then run this again. You do not need to close this Python program. Ru...</li><li><a href="https://x.com/bartowski1182/status/1786038369132171444?t=hJfQz8lGt9v31yZRG4X1vA&s=09">Tweet from bartowski (@bartowski1182)</a>: After days of compute (since I had to start over) it&#39;s finally up! Llama 3 70B GGUF with tokenizer fix :)  https://huggingface.co/bartowski/Meta-Llama-3-70B-Instruct-GGUF  In other news, just orde...</li><li><a href="https://github.com/ggerganov/llama.cpp/commit/f4ab2a41476600a98067a9474ea8f9e6db41bcfa">llama : fix BPE pre-tokenization (#6920) · ggerganov/llama.cpp@f4ab2a4</a>: * merged the changes from deepseeker models to main branch
 
 * Moved regex patterns to unicode.cpp and updated unicode.h
 
 * Moved header files
 
 * Resolved issues
 
 * added and refactored unic...</li><li><a href="https://github.com/unslothai/unsloth/issues/430">GGUF breaks - llama-3 · Issue #430 · unslothai/unsloth</a>: Findings from ggerganov/llama.cpp#7062 and Discord chats: Notebook for repro: https://colab.research.google.com/drive/1djwQGbEJtUEZo_OuqzN_JF6xSOUKhm4q?usp=sharing Unsloth + float16 + QLoRA = WORKS...</li><li><a href="https://github.com/unslothai/unsloth/wiki#manually-saving-to-gguf">Home</a>: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://github.com/ggerganov/llama.cpp/commit/ca3632602091e959ed2ad4c09c67a7c790b10d31">readme : add note that LLaMA 3 is not supported with convert.py (#7065) · ggerganov/llama.cpp@ca36326</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/wiki#manually-savin">Home</a>: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/issues/210">I got unsloth running in native windows. · Issue #210 · unslothai/unsloth</a>: I got unsloth running in native windows, (no wsl). You need visual studio 2022 c++ compiler, triton, and deepspeed. I have a full tutorial on installing it, I would write it all here but I’m on mob...</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/7021">Cannot convert llama3 8b model to gguf · Issue #7021 · ggerganov/llama.cpp</a>: Please include information about your system, the steps to reproduce the bug, and the version of llama.cpp that you are using. If possible, please provide a minimal code example that reproduces the...</li><li><a href="https://github.com/ggerganov/llama.cpp/tree/gg/bpe-preprocess">GitHub - ggerganov/llama.cpp at gg/bpe-preprocess</a>: LLM inference in C/C++. Contribute to ggerganov/llama.cpp development by creating an account on GitHub.</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/6965">llama3 custom regex split by jaime-m-p · Pull Request #6965 · ggerganov/llama.cpp</a>: Implementation of unicode_regex_split_custom_llama3().</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/7062#issuecomment-2094961774">Llama3 GGUF conversion with merged LORA Adapter seems to lose training data randomly · Issue #7062 · ggerganov/llama.cpp</a>: I&#39;m running Unsloth to fine tune LORA the Instruct model on llama3-8b . 1: I merge the model with the LORA adapter into safetensors 2: Running inference in python both with the merged model direct...</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/7062">Llama3 GGUF conversion with merged LORA Adapter seems to lose training data randomly · Issue #7062 · ggerganov/llama.cpp</a>: I&#39;m running Unsloth to fine tune LORA the Instruct model on llama3-8b . 1: I merge the model with the LORA adapter into safetensors 2: Running inference in python both with the merged model direct...</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/7062#issuecomment-2095465106">Llama3 GGUF conversion with merged LORA Adapter seems to lose training data randomly · Issue #7062 · ggerganov/llama.cpp</a>: I&#39;m running Unsloth to fine tune LORA the Instruct model on llama3-8b . 1: I merge the model with the LORA adapter into safetensors 2: Running inference in python both with the merged model direct...</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/7062#issuecomment-2094955278">Llama3 GGUF conversion with merged LORA Adapter seems to lose training data randomly · Issue #7062 · ggerganov/llama.cpp</a>: I&#39;m running Unsloth to fine tune LORA the Instruct model on llama3-8b . 1: I merge the model with the LORA adapter into safetensors 2: Running inference in python both with the merged model direct...</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/7062#issuecomment-2094875716">Llama3 GGUF conversion with merged LORA Adapter seems to lose training data randomly · Issue #7062 · ggerganov/llama.cpp</a>: I&#39;m running Unsloth to fine tune LORA the Instruct model on llama3-8b . 1: I merge the model with the LORA adapter into safetensors 2: Running inference in python both with the merged model direct...</li><li><a href="https://github.com/ggerganov/llama.cpp/blob/master/convert.py">llama.cpp/convert.py at master · ggerganov/llama.cpp</a>: LLM inference in C/C++. Contribute to ggerganov/llama.cpp development by creating an account on GitHub.</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/7062#issuecomment-2094948789">Llama3 GGUF conversion with merged LORA Adapter seems to lose training data randomly · Issue #7062 · ggerganov/llama.cpp</a>: I&#39;m running Unsloth to fine tune LORA the Instruct model on llama3-8b . 1: I merge the model with the LORA adapter into safetensors 2: Running inference in python both with the merged model direct...</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/7062#issuecomment-2095371349">Llama3 GGUF conversion with merged LORA Adapter seems to lose training data randomly · Issue #7062 · ggerganov/llama.cpp</a>: I&#39;m running Unsloth to fine tune LORA the Instruct model on llama3-8b . 1: I merge the model with the LORA adapter into safetensors 2: Running inference in python both with the merged model direct...
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1235848362516021248)** (80 messages🔥🔥): 

- **Proposal for Model Size Discussion Channel**: A user suggested creating a separate channel on Unsloth Discord for discussing the **successes and strategies** in deploying large language models (LLMs). The conversation emphasized the value of sharing experiences to enhance collective learning.

- **Push for Llama-3-8B-Based Projects**: RomboDawg announced the release of a new coding model that enhances **Llama-3-8B-Instruct** and competes with **Llama-3-70B-Instruct** performance. The model can be accessed [here](https://huggingface.co/rombodawg/Codellama-3-8B-Finetuned-Instruct), and excitement for a version 2, promised to be available in about three days, was expressed.

- **Knowledge Graph LLM Variant Released**: M.chimiste has developed a **Llama-3 variant** to assist in knowledge graph construction, named **LLaMA-3-8B-RDF-Experiment**, emphasizing its utility in generating knowledge graph triples and potential for genomic data training. The model can be found at [Hugging Face's model repository](https://huggingface.co/M-Chimiste/Llama-3-8B-RDF-Experiment).

- **On the Horizon of Creptographic Collaborations**: In an extended discussion, one user is seeking advice and collaborative discussion about building a system that could potentially integrate cryptographic elements into blockchain technologies, expressing interest in learning from the community.

- **AI-Enhanced Web Development Tools Theme**: Oncord is showcased as providing a modern web development platform with built-in marketing and commerce tools, and its developer is integrating **Unsloth AI** for **LLM fine-tuning** to provide code completions and potentially power an AI-driven redesign feature. More about Oncord can be found [here](https://www.oncord.com/).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.llama-hub.com/models">no title found</a>: no description found</li><li><a href="https://huggingface.co/M-Chimiste/Llama-3-8B-RDF-Experiment">M-Chimiste/Llama-3-8B-RDF-Experiment · Hugging Face</a>: no description found</li><li><a href="https://tenor.com/view/dog-awkward-awkward-dog-staring-dog-patchibana-gif-13086408744970718509">Dog Awkward GIF - Dog Awkward Awkward dog - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://pubmed.ncbi.nlm.nih.gov/28288169/):">The miR-200 family is increased in dysplastic lesions in ulcerative colitis patients - PubMed</a>: UC-Dysplasia is linked to altered miRNA expression in the mucosa and elevated miR-200b-3p levels.</li><li><a href="https://x.com/dudeman6790/status/1786783966738919738">Tweet from RomboDawg (@dudeman6790)</a>: Announcing Codellama-3-8B A Qalore Finetune of llama-3-8b-instruct on the full OpenCodeInterpreter dataset. It codes far better than the base instruct model, and iterated on code extremely well. Forgi...</li><li><a href="https://www.llama-hub.com/model_detail/4df65c9a-6b23-413e-b4eb-9d34c446db48">Llama-3-8B-Instruct-Coder</a>: no description found</li><li><a href="https://www.oncord.com/">Oncord - Digital Marketing Software</a>: Website, email marketing, and ecommerce in one intuitive software platform. Oncord hosted CMS makes it simple.</li><li><a href="https://www.tryoncord.com/admin/">no title found</a>: no description found
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[suggestions](https://discord.com/channels/1179035537009545276/1180144489214509097/1237119442899435582)** (3 messages): 

- **Fine-tuning LVLM Desired**: A member expressed a wish for a **generalised way of fine-tuning LVLM**, indicating ongoing interest in customization and optimization of language-vision models.

- **MoonDream Fine-tuning Interest**: Another member recommended **support for Moondream**, a tiny vision-language model which currently only finetunes the **phi 1.5 text model**. They provided a GitHub notebook as a resource: [moondream/notebooks/Finetuning.ipynb on GitHub](https://github.com/vikhyat/moondream/blob/main/notebooks/Finetuning.ipynb).

**Link mentioned**: <a href="https://github.com/vikhyat/moondream/blob/main/notebooks/Finetuning.ipynb">moondream/notebooks/Finetuning.ipynb at main · vikhyat/moondream</a>: tiny vision language model. Contribute to vikhyat/moondream development by creating an account on GitHub.

  

---



**OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1235963163971682304)** (854 messages🔥🔥🔥): 

- **A New Challenger Approaches Perplexity**: Users are discussing the benefits of Perplexity AI, particularly its new Pages feature which allows for creation of comprehensive reports.
- **AI and Self-Learning**: Some discuss the possibility of AI engines like OpenAI's GPT to teach programming basics to users and help in creating code, espousing the idea of self-sufficient AIs with the capacity for self-improvement.
- **The Evolving Definition of AGI**: The community is engaging in a debate about the current state of AI and its proximity to true AGI (Artificial General Intelligence), with varying opinions on whether modern AI like ChatGPT qualifies as early AGI.
- **Appetite for AI-Generated Music**: Users express interest in AI-generated music, referencing services like Udio and discussing whether OpenAI should release its own AI music service.
- **AI as a Tool for Expansion**: The conversation explores how AI currently augments human productivity and the potential future where AI might take over mundane and complex tasks, also reflecting on how this might disrupt our socio-economic models.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://scholar.google.com/citations?user=4FsWE64AAAAJ)">Google Scholar Citations</a>: no description found</li><li><a href="https://tenor.com/view/dirty-docks-shawty-triflin-shawty-triflin-she-gif-22455514">Dirty Docks Shawty GIF - Dirty Docks Shawty Triflin - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.nature.com/articles/d41586-023-00107-z">ChatGPT listed as author on research papers: many scientists disapprove</a>: At least four articles credit the AI tool as a co-author, as publishers scramble to regulate its use.</li><li><a href="https://github.com/catppuccin/catppuccin">GitHub - catppuccin/catppuccin: 😸 Soothing pastel theme for the high-spirited!</a>: 😸 Soothing pastel theme for the high-spirited! Contribute to catppuccin/catppuccin development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1235851459891957821)** (40 messages🔥): 

- **Slow and Steady Doesn't Win the Race**: Members are reporting significant increases in latency with **GPT-4 Turbo**, with some experiencing response times **5-10x slower** than usual.
- **The Cap on Conversation**: There's confusion around the message cap for GPT-4, as users report different timeout thresholds. Some state a cap between **25 and 50 messages**, while others suspect dynamic adjustments during **high usage** periods.
- **OpenAI Platform's UX Blues**: Complaints have emerged about the user experience on OpenAI's new projects feature, with issues in **project management**, **deletion**, and **navigability**; also noting an **absence of activity tracking** per project.
- **Will There Be a GPT-5?** Users are skeptical about the release of GPT-5, discussing **diminishing returns** and the likelihood that it would be **"2x the cost for 1.5x better GPT-4"**.
- **The Hunt for Knowledge Prioritization**: Users debate strategies to make ChatGPT **search its knowledge base first** before responding, touching on concepts like **RAG (Retrieval-Augmented Generation)** and the **vectorization of knowledge** to assist in providing contextually relevant answers.
  

---


**OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1236180170323267597)** (30 messages🔥): 

- **Fine-tuning GPT for Questioning**: A member is seeking advice on how to fine-tune a model to ask questions instead of giving answers, mentioning previous struggles with a similar project. They note difficulty finding appropriate user query and assistant query pairs and are considering using single tuple chats as samples for fine-tuning.

- **The Resilient Onboarding Bot**: Member **leveloper** mentions a successfully functioning bot designed to ask questions during an onboarding process, which remains untricked by user attempts despite being on a large server.

- **Avoiding Negative Prompts**: **majestic_axolotl_19289** suggests that using negative prompts can backfire, as they tend to influence the outcome in unintended ways. Other members discuss whether negative prompts can be effective, citing the "Contrastive Chain of Thoughts" paper and personal experiences.

- **Book Recommendation for Prompt Engineering**: Member **sephyfox_** recommends "Wordplay: Your Guide to Using Artificial Intelligence for Writing Software" by Teddy Dicus Murphy, finding it helpful for prompt engineering.

- **Request for Improving GPT-4-TURBO Prompt for Product Info Extraction**: Member **stevenli_36050** seeks assistance in refining a prompt to extract product information, names, and prices from PDF supermarket brochures and categorize them accordingly.

- **Discussing Logit Bias in Token Suppression**: The user **bambooshoots** shares a link (https://help.openai.com/en/articles/5247780-using-logit-bias-to-alter-token-probability-with-the-openai-api) about manipulating probabilities using logit bias to suppress certain tokens in the OpenAI API.
  

---


**OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1236180170323267597)** (30 messages🔥): 

- **In Search of the Questioning Bot**: A member discussed the challenge of fine-tuning GPT for generating questions in conversations rather than providing answers, highlighting the difficulty in defining the structure of user queries and bot responses in such scenarios.

- **Contrastive Chain of Thought (CCoT) Discourse**: There was a debate on the use of negative prompts in prompting strategies. The conversation mentioned a paper on "Contrastive Chain of Thoughts" and questioned the effectiveness of using CCoT in longer dialogs, prompting an invitation to read further on the [AIEmpower blog](https://aiempower.com/).

- **Prompt Engineering Resources and Techniques Shared**: Users shared resources about prompt engineering, including a recommendation for the book "Wordplay: Your Guide to Using Artificial Intelligence for Writing Software" by Teddy Dicus Murry and a LinkedIn learning course by Ronnie Sheer.

- **Extracting Supermarket Product Data**: A user sought advice on improving GPT-4-TURBO prompts for identifying product names and prices from PDF supermarket brochures, seeking to output the results in CSV format.
 
- **Logit Bias for Token Probability Manipulation**: A member referenced the logit bias as a method for manipulating token probabilities in prompts with a link to OpenAI's official documentation: [Using logit bias to alter token probability with the OpenAI API](https://help.openai.com/en/articles/5247780-using-logit-bias-to-alter-token-probability-with-the-openai-api).
  

---



**Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1235849532609265724)** (919 messages🔥🔥🔥): 

- **GPU Compatibility Queries**: Users are asking for assistance with Stable Diffusion installations that can't access GPU resources, mentioning errors like "RuntimeError: Torch is not able to use GPU".
- **Stability.ai and SD3 Speculation**: Conversations revolve around the anticipated release of Stable Diffusion 3, with many expressing doubt it will come out, while others discuss the impact if it doesn't.
- **Finetuning Tutorials Seekers**: Members express frustration over the lack of detailed tutorials for LoRA/DreamBooth/fine-tuning models, stating that available resources are outdated or not comprehensive.
- **Request for Help on Generating Unique Faces**: A query was made on how to train a unique realistic-looking person using AI, either through training LoRa on multiple faces or generating random ones and then training the LoRa on that result.
- **Discussion on the 'Open Source' Nature of Stable Diffusion**: Some users discuss the barriers to truly "open-source" AI art generation, sharing concerns about future paywalled access to high-quality model checkpoints and training details.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://humanaigc.github.io/emote-portrait-alive/">EMO</a>: EMO: Emote Portrait Alive - Generating Expressive Portrait Videos with Audio2Video Diffusion Model under Weak Conditions</li><li><a href="https://highlight.fm">Highlight: Generate photos with friends</a>: Highlight is an app to daydream with friends by genering images with them.</li><li><a href="https://stability.ai/news/stable-diffusion-3-research-paper">Stable Diffusion 3: Research Paper &mdash; Stability AI</a>: Following our announcement of the early preview of Stable Diffusion 3, today we are publishing the research paper which outlines the technical details of our upcoming model release, and invite you to ...</li><li><a href="https://fireworks.ai/models/stability/sd3">Fireworks - Generative AI For Product Innovation!</a>: Use state-of-the-art, open-source LLMs and image models at blazing fast speed, or fine-tune and deploy your own at no additional cost with Fireworks.ai!</li><li><a href="https://www.instagram.com/farina.fab?igsh=YXlsbWRycnIxbjNu">Login • Instagram</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=kqXpAKVQDNU&list=PLXS4AwfYDUi5sbsxZmDQWxOQTml9Uqyd2">How to Install Stable Diffusion - automatic1111</a>: Part 2: How to Use Stable Diffusion https://youtu.be/nJlHJZo66UAAutomatic1111 https://github.com/AUTOMATIC1111/stable-diffusion-webuiInstall Python https://w...</li><li><a href="https://www.youtube.com/watch?v=juP6gpiOY2A">High-Similarity Face Swapping: ControlNet IP-Adapter + Instant-ID Combo</a>: Discover the art of high-similarity face swapping using WebUI Forge, IP-Adapter, and Instant-ID for seamless, realistic results.🖹 Article Tutorial:- https:/...</li><li><a href="https://github.com/philz1337x/clarity-upscaler">GitHub - philz1337x/clarity-upscaler: Clarity AI | AI Image Upscaler &amp; Enhancer - free and open-source Magnific Alternative</a>: Clarity AI | AI Image Upscaler &amp; Enhancer - free and open-source Magnific Alternative - philz1337x/clarity-upscaler</li><li><a href="https://civitai.com/models/410151/aether-light-lora-for-sdxl">Aether Light - LoRA for SDXL - v1.0 | Stable Diffusion LoRA | Civitai</a>: For business inquiries, commercial licensing, custom models, and consultation, contact me at joachim@rundiffusion.com . Introducing Aether Light, o...</li><li><a href="https://github.com/crystian/ComfyUI-Crystools">GitHub - crystian/ComfyUI-Crystools: A powerful set of tools for ComfyUI</a>: A powerful set of tools for ComfyUI. Contribute to crystian/ComfyUI-Crystools development by creating an account on GitHub.</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1cgr74j/comment/l2bxv66/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1ciyzn5/comment/l2dhd6q/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://youtu.be/xXNr9mrdV7s?si=PalGotNpAeYl4Zy4">LORA training EXPLAINED for beginners</a>: LORA training guide/tutorial so you can understand how to use the important parameters on KohyaSS.Train in minutes with Dreamlook.AI:  https://dreamlook.ai/?...</li><li><a href="https://rentry.co/59xed3#prodigy">THE OTHER LoRA TRAINING RENTRY</a>: Stable Diffusion LoRA training science and notes By yours truly, The Other LoRA Rentry Guy. This is not a how to install guide, it is a guide about how to improve your results, describe what options d...
</li>
</ul>

</div>
  

---



**Nous Research AI ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/1237067945465876572)** (1 messages): 

I apologize for the confusion, but as an AI, I do not have direct access to Discord servers, channels, or messages. Thus, I am unable to summarize the content from the Nous Research AI Discord channel named ctx-length-research. If you can provide the text from specific Discord messages that you'd like to be summarized, I'd be happy to assist you.
  

---


**Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1235957375173656637)** (20 messages🔥): 

- **Color Evolution on Camera**: A member humorously noted that comparing images of **Saint Petersburg, Ligovsky Avenue at Vosstaniya Square** from 2002 and 2024 shows that *cameras have gotten more color accurate*.
- **Culinary Flavor Fusion**: A simple mention was made of **Okroshka** on kvas with mayonnaise accompanied by rye bread, possibly suggesting a discussion or reference to traditional Russian cuisine.
- **Inquiry About SVM**: A member asked, "What is SVM?" to which another member quickly clarified that SVM stands for **Support Vector Machine**.
- **Improve the UX for FreeGPT.today**: A member requested feedback on the user experience for their site [FreeGPT.today](https://freegpt.today/), inviting others to sign up, chat, and test a PDF upload feature that generates graphs. Several suggestions for improvement were offered, including adding Google authentication, changing the default login landing page to "chat now," improving UI elements, and implementing a progress bar for file uploads.
- **Beware of Spam Links**: A mention was made that a Discord invite link shared in the chat was actually spam and led to the sharer getting banned.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://youtu.be/J1ZiaE7cqQY">Recipic Demo</a>: Ever felt confused about what to make for dinner or lunch? What if there was a website where you could just upload what ingredients you have and get recipes ...</li><li><a href="https://freegpt.today/">FreeGPT.today - The Most Powerful AI Language Model for Free!</a>: Access the most powerful AI language model for free. No credit card required.
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1235972681413689445)** (47 messages🔥): 

- **Exploring Taskmaster with LLM**: A code implementation of the show **Taskmaster** using structured data management, a state machine, and the OpenAI API was shared. The code is available on [GitHub](https://github.com/LEXNY/Taskmaster-LLM).
- **Evaluating LLM Responses**: Another GitHub repository was introduced featuring **Prometheus**, a tool for evaluating LLM responses, available at [prometheus-eval](https://github.com/prometheus-eval/prometheus-eval).
- **VRAM Consumption Calculator for LLMs**: A Hugging Face space was mentioned which contains an LLM Model VRAM Calculator, to help users determine how much VRAM they'll require, viewable [here](https://huggingface.co/spaces/NyxKrage/LLM-Model-VRAM-Calculator).
- **Fixing Mistral Model Issues**: Discussions focused on fixing issues with the **Mistral model** and potential Pull Requests (PRs) to address these were highlighted. The ongoing conversation about modifications, particularly around the rotary embeddings, finds its latest relevant PR on [GitHub](https://github.com/huggingface/transformers/pull/30658).
- **Improvements and Issues with Open Pretrain Datasets**: A recent paper examining the quality of training corpora used for language models was mentioned. The study discussed the prevalence of duplicate, synthetic, and low-quality content in these datasets, with details available in their [arXiv paper](https://arxiv.org/abs/2310.20707).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://secretllama.com/">Secret Llama</a>: no description found</li><li><a href="https://arxiv.org/abs/2401.17377">Infini-gram: Scaling Unbounded n-gram Language Models to a Trillion Tokens</a>: Are $n$-gram language models still relevant in this era of neural large language models (LLMs)? Our answer is yes, and we showcase their values in both text analysis and improving neural LLMs. This wa...</li><li><a href="https://demo.haystack.zip/">Demo Search Fine Web Dataset</a>: no description found</li><li><a href="https://arxiv.org/abs/2404.19737">Better &amp; Faster Large Language Models via Multi-token Prediction</a>: Large language models such as GPT and Llama are trained with a next-token prediction loss. In this work, we suggest that training language models to predict multiple future tokens at once results in h...</li><li><a href="https://huggingface.co/spaces/NyxKrage/LLM-Model-VRAM-Calculator">LLM Model VRAM Calculator - a Hugging Face Space by NyxKrage</a>: no description found</li><li><a href="https://arxiv.org/abs/2401.10774">Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads</a>: The inference process in Large Language Models (LLMs) is often limited due to the absence of parallelism in the auto-regressive decoding process, resulting in most operations being restricted by the m...</li><li><a href="https://arxiv.org/abs/2310.20707">What&#39;s In My Big Data?</a>: Large text corpora are the backbone of language models. However, we have a limited understanding of the content of these corpora, including general statistics, quality, social factors, and inclusion o...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1clinlb/bringing_2bit_llms_to_production_new_aqlm_models/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1clmo7u/phi3_weights_orthogonalized_to_inhibit_refusal/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/LEXNY/Taskmaster-LLM">GitHub - LEXNY/Taskmaster-LLM</a>: Contribute to LEXNY/Taskmaster-LLM development by creating an account on GitHub.</li><li><a href="https://github.com/prometheus-eval/prometheus-eval">GitHub - prometheus-eval/prometheus-eval: Evaluate your LLM&#39;s response with Prometheus 💯</a>: Evaluate your LLM&#39;s response with Prometheus 💯. Contribute to prometheus-eval/prometheus-eval development by creating an account on GitHub.</li><li><a href="https://youtu.be/oNwoA5akBlg">NVIDIA CEO Jensen Huang Leaves Everyone SPEECHLESS (Supercut)</a>: Highlights of #nvidia ( #nvda stock ) Founder and CEO Jensen Huang speaking at Stanford Institute for Economic Policy Research (SIEPR). Highlights include wh...</li><li><a href="https://www.lesswrong.com/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction">Refusal in LLMs is mediated by a single direction — LessWrong</a>: This work was produced as part of Neel Nanda&#x27;s stream in the ML Alignment &amp; Theory Scholars Program - Winter 2023-24 Cohort, with co-supervision from…</li><li><a href="https://github.com/huggingface/transformers/pull/30658">[WIP][FIX] Fix Mixtral model by casper-hansen · Pull Request #30658 · huggingface/transformers</a>: This PR is a WIP based on @kalomaze&#39;s implementation that fixes the Mixtral model. It has been known for a while that Mixtral has been hard to train due to some bug in the code. Please note this i...</li><li><a href="https://github.com/huggingface/transformers/pull/">Pull requests · huggingface/transformers</a>: 🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX. - Pull requests · huggingface/transformers</li><li><a href="https://app.wordware.ai/r/81fef99d-70e5-4c6a-ad0d-bd1057bfc818">Wordware - WebIntellect - Search with ScratchPad-Think Framework (V2)</a>: Use the power of &lt;ScratchPad-Think&gt; for every day web searches 
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1235849078479519855)** (717 messages🔥🔥🔥): 

- **Hermes Overperformance with Classic Llama Scaling**: **Hermes 2 Pro Llama 8B** gained extended context capacity to ~32k using RoPE scaling with **vLLM** on a 32GB Nvidia v100 Tesla and showed no noticeable degradation, providing perfect recall at 16k according to users’ experiences.

- **Setting Up Enhanced Context**: Editing the `config.json` in the Hermes model from Hugging Face and tweaking the rope scaling factor before initializing the server is suggested for context extension.

- **Serverless Llama Limitations**: Users report different capabilities and limitations across model inference providers, with the need to coordinate features such as grammar and JSON mode, which is only supported in **llama.cpp**, not in **vLLM** according to discussions on the **vLLM** GitHub issues page.

- **High Anticipation for Llama-3-120B-Instruct**: A **Hugging Face** model titled **Meta-Llama-3-120B-Instruct**, a self-merged model, has garnered attention and interest for its supposed increased performance; however, some users caution about believing the hype without thorough benchmarking. 

- **Balancing Compute Resources and Model Performance**: Users discuss the trade-offs of using beefier GPUs, such as **Azure's NC80adis_H100_v5**, and the balance between sufficient VRAM, latency, and tokens per second for practical use in tasks requiring large context sizes.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/turboderp/Cat-Llama-3-70B-instruct">turboderp/Cat-Llama-3-70B-instruct · Hugging Face</a>: no description found</li><li><a href="https://x.com/dudeman6790/status/1786783966738919738">Tweet from RomboDawg (@dudeman6790)</a>: Announcing Codellama-3-8B A Qalore Finetune of llama-3-8b-instruct on the full OpenCodeInterpreter dataset. It codes far better than the base instruct model, and iterated on code extremely well. Forgi...</li><li><a href="https://huggingface.co/cognitivecomputations/Meta-Llama-3-120B-Instruct-gguf">cognitivecomputations/Meta-Llama-3-120B-Instruct-gguf · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B#prompt-format-for-json-mode--structured-outputs">NousResearch/Hermes-2-Pro-Llama-3-8B · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B">NousResearch/Hermes-2-Pro-Llama-3-8B · Hugging Face</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1a-aQvKC9avdZpdyBn4jgRQFObTPy1JZw?usp=sharing#scrollTo=2EoxY5i1CWe3">Google Colab</a>: no description found</li><li><a href="https://arxiv.org/abs/2404.19234">Multi-hop Question Answering over Knowledge Graphs using Large Language Models</a>: Knowledge graphs (KGs) are large datasets with specific structures representing large knowledge bases (KB) where each node represents a key entity and relations amongst them are typed edges. Natural l...</li><li><a href="https://huggingface.co/blockblockblock/Hermes-2-Pro-Llama-3-8B-bpw2.5-exl2">blockblockblock/Hermes-2-Pro-Llama-3-8B-bpw2.5-exl2 · Hugging Face</a>: no description found</li><li><a href="https://arxiv.org/abs/2405.00200">In-Context Learning with Long-Context Models: An In-Depth Exploration</a>: As model context lengths continue to increase, the number of demonstrations that can be provided in-context approaches the size of entire training datasets. We study the behavior of in-context learnin...</li><li><a href="https://x.com/0xblacklight/status/1787329977957982398">Tweet from Kyle Mistele 🏴‍☠️ (@0xblacklight)</a>: btw I tested this with @vllm_project and it works to scale @NousResearch&#39;s Hermes 2 Pro Llama 3 8B to ~32k context with great coherence & performance (I had it summarizing @paulg essays)  Download...</li><li><a href="https://huggingface.co/datasets/CarperAI/pilev2-dev">CarperAI/pilev2-dev · Datasets at Hugging Face</a>: no description found</li><li><a href="https://x.com/yacinemtb/status/1786958419846418664?s=46&t=stOPrwZiN_fxSK0RuC8Fl">Tweet from kache (@yacineMTB)</a>: your entire company will cease to exist when llama 400b lands. do you really think you can regulate fast enough? do you know how slow the government moves? a single torrent will drop and your entire b...</li><li><a href="https://arxiv.org/abs/2310.00785">BooookScore: A systematic exploration of book-length summarization in the era of LLMs</a>: Summarizing book-length documents (&gt;100K tokens) that exceed the context window size of large language models (LLMs) requires first breaking the input document into smaller chunks and then promptin...</li><li><a href="https://x.com/yacinemtb/status/1786958419846418664?s=46&t=stOPrwZiN_fxSK0RuC8Flg">Tweet from kache (@yacineMTB)</a>: your entire company will cease to exist when llama 400b lands. do you really think you can regulate fast enough? do you know how slow the government moves? a single torrent will drop and your entire b...</li><li><a href="https://github.com/NousResearch/Hermes-Function-Calling/blob/main/examples/crewai_agents.ipynb">Hermes-Function-Calling/examples/crewai_agents.ipynb at main · NousResearch/Hermes-Function-Calling</a>: Contribute to NousResearch/Hermes-Function-Calling development by creating an account on GitHub.</li><li><a href="https://cloud.google.com/pricing/">Pricing Overview</a>: With Google Cloud’s pay-as-you-go pricing, you only pay for the services you use. No upfront costs. No termination fees.</li><li><a href="https://huggingface.co/blockblockblock/Hermes-2-Pro-Llama-3-8B-bpw4-exl2">blockblockblock/Hermes-2-Pro-Llama-3-8B-bpw4-exl2 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/blockblockblock/Hermes-2-Pro-Llama-3-8B-bpw5.5-exl2">blockblockblock/Hermes-2-Pro-Llama-3-8B-bpw5.5-exl2 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/blockblockblock/Hermes-2-Pro-Llama-3-8B-bpw6-exl2">blockblockblock/Hermes-2-Pro-Llama-3-8B-bpw6-exl2 · Hugging Face</a>: no description found</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/7013">Update Server&#39;s README with undocumented options for RoPE, YaRN, and KV cache quantization by K-Mistele · Pull Request #7013 · ggerganov/llama.cpp</a>: I recently updated my LLama.cpp and found that there are a number of server CLI options which are not described in the README including for RoPE, YaRN, and KV cache quantization as well as flash at...</li><li><a href="https://github.com/theavgjojo/openai_api_tool_call_proxy">GitHub - theavgjojo/openai_api_tool_call_proxy: A thin proxy PoC to support prompt/message handling of tool calls for OpenAI API-compliant local APIs which don&#39;t support tool calls</a>: A thin proxy PoC to support prompt/message handling of tool calls for OpenAI API-compliant local APIs which don&#39;t support tool calls - theavgjojo/openai_api_tool_call_proxy</li><li><a href="https://tenor.com/view/mlp-relevant-mylittlepony-interests-gif-4506356">Mlp Relevant GIF - MLP Relevant Mylittlepony - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1ckcw6z/1m_context_models_after_16k_tokens/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/huggingface/lerobot">GitHub - huggingface/lerobot: 🤗 LeRobot: State-of-the-art Machine Learning for Real-World Robotics in Pytorch</a>: 🤗 LeRobot: State-of-the-art Machine Learning for Real-World Robotics in Pytorch - huggingface/lerobot</li><li><a href="https://github.com/Infini-AI-Lab/Sequoia">GitHub - Infini-AI-Lab/Sequoia: scalable and robust tree-based speculative decoding algorithm</a>: scalable and robust tree-based speculative decoding algorithm - Infini-AI-Lab/Sequoia</li><li><a href="https://github.com/vllm-project/vllm/issues/1229">Support for grammar · Issue #1229 · vllm-project/vllm</a>: It would be highly beneficial if the library could incorporate support for Grammar and GBNF files. https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md</li><li><a href="https://github.com/snakers4/silero-models">GitHub - snakers4/silero-models: Silero Models: pre-trained speech-to-text, text-to-speech and text-enhancement models made embarrassingly simple</a>: Silero Models: pre-trained speech-to-text, text-to-speech and text-enhancement models made embarrassingly simple - snakers4/silero-models</li><li><a href="https://arxiv.org/abs/2404.17733">Building a Large Japanese Web Corpus for Large Language Models</a>: Open Japanese large language models (LLMs) have been trained on the Japanese portions of corpora such as CC-100, mC4, and OSCAR. However, these corpora were not created for the quality of Japanese tex...</li><li><a href="https://github.com/N8python/SYNTH-8">GitHub - N8python/SYNTH-8: An open-source voice-enabled chatbot. Many features will come soon.</a>: An open-source voice-enabled chatbot. Many features will come soon. - N8python/SYNTH-8</li><li><a href="https://arxiv.org/abs/2312.15166">SOLAR 10.7B: Scaling Large Language Models with Simple yet Effective Depth Up-Scaling</a>: We introduce SOLAR 10.7B, a large language model (LLM) with 10.7 billion parameters, demonstrating superior performance in various natural language processing (NLP) tasks. Inspired by recent efforts t...</li><li><a href="https://x.com/maziyarpanahi/status/1786751050130608168?s=46">Tweet from Maziyar PANAHI (@MaziyarPanahi)</a>: Great job @Gradient_AI_! This one is very close to the Instruct and that&#39;s pretty impressive! ❤️🚀👏🏽  Quoting OpenLLMLeaders (@OpenLLMLeaders)   New model added to the leaderboard!  Model Name h...</li><li><a href="https://www.paddle.com/ai-launchpad">Scale your AI business with Paddle | AI Launchpad</a>: no description found</li><li><a href="https://blog.arcee.ai/arcee-mergekit-launch-model-merging-hackathon/">Arcee/Mergekit launch Model Merging Hackathon</a>: Arcee &amp; MergeKit advance model merging innovations with launch of MergeKit Hackathon, co-sponsored by AWS. Submit your model merging research, experiments, and results for the chance to win cash p...</li><li><a href="https://github.com/hsiehjackson/RULER">GitHub - hsiehjackson/RULER: This repo contains the source code for RULER: What’s the Real Context Size of Your Long-Context Language Models?</a>: This repo contains the source code for RULER: What’s the Real Context Size of Your Long-Context Language Models? - hsiehjackson/RULER</li><li><a href="https://github.com/OpenBMB/InfiniteBench#evaluation-result">GitHub - OpenBMB/InfiniteBench: Codes for the paper &quot;∞Bench: Extending Long Context Evaluation Beyond 100K Tokens&quot;: https://arxiv.org/abs/2402.13718</a>: Codes for the paper &quot;∞Bench: Extending Long Context Evaluation Beyond 100K Tokens&quot;: https://arxiv.org/abs/2402.13718 - OpenBMB/InfiniteBench</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/5104">Port of self extension to server by Maximilian-Winter · Pull Request #5104 · ggerganov/llama.cpp</a>: Hi, I ported the code for self extension over to the server. I have tested it with a information retrieval, I inserted information out of context into a ~6500 tokens long text and it worked, at lea...</li><li><a href="https://huggingface.co/blockblockblock/Hermes-2-Pro-Llama-3-8B-bpw3-exl2">blockblockblock/Hermes-2-Pro-Llama-3-8B-bpw3-exl2 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/blockblockblock/Hermes-2-Pro-Llama-3-8B-bpw3.5-exl2">blockblockblock/Hermes-2-Pro-Llama-3-8B-bpw3.5-exl2 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/blockblockblock/Hermes-2-Pro-Llama-3-8B-bpw3.7-exl2">blockblockblock/Hermes-2-Pro-Llama-3-8B-bpw3.7-exl2 · Hugging Face</a>: no description found</li><li><a href="https://github.com/ggerganov/llama.cpp/discussions/1965">Extending context size via RoPE scaling · ggerganov/llama.cpp · Discussion #1965</a>: Intro This is a discussion about a recently proposed strategy of extending the context size of LLaMA models. The original idea is proposed here: https://kaiokendev.github.io/til#extending-context-t...
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1235868437432107029)** (60 messages🔥🔥): 

- **LLMs Garner Enthusiasm**: A member expressed delight in experimenting with a local AI, sharing their first enjoyable experience with the platform.
- **Hermes 2 Pro Llama 3 vs. Mistral**: Discussion evolved around **Hermes 2 Pro Llama 3** underperforming compared to **Mistral**, sparking insights that **Mixtral's** larger model size contributes to its higher ranking, particularly in the **MMLU benchmark**.
- **Understanding LLaVA's Multimodal Capabilities**: In relation to teaching GPT/LLMs about images, members were directed to explore **LLaVA**, a large multimodal model with enhanced visual and language understanding that outperforms on 11 benchmarks.
- **Tool XML Tag Troubles in Text Generation**: There was an exchange about an issue when migrating to **LlamaCPP** where `<tool_call>` xml tags were not generated, later resolved by updating LlamaCPP to the latest version.
- **Speed Woes with LoRA Llama 3 8B Training**: A member inquired about the seemingly excessive duration for LoRA training on **Llama 3 8B**, contrasting it with far speedier experiences reported by others using different setups.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.writewithlaika.com)">no title found</a>: no description found</li><li><a href="https://llava-vl.github.io/">LLaVA</a>: no description found</li><li><a href="https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF/tree/main">NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF at main</a>: no description found</li><li><a href="https://github.com/aniketmaurya/agents/blob/main/src/agents/hermes/functioncall.py#L30">agents/src/agents/hermes/functioncall.py at main · aniketmaurya/agents</a>: A fun project to build Agentic workflows with function calling, powered by LangChain. - aniketmaurya/agents
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1235976657823862865)** (2 messages): 

- **Seeking Free Datasets**: A member inquired about sources for good free generic datasets.
  
- **Cynde Core Implementation Update**: An update on **Cynde**, a framework for intelligence farming, was shared. The core implementation is in place, and the contributor is open to help and efforts to maintain code cleanliness, stating there is no inclusion of RAG on purpose yet. The updated readme and notes are available at [Neural-Dragon-AI/Cynde](https://github.com/Neural-Dragon-AI/Cynde).

**Link mentioned**: <a href="https://github.com/Neural-Dragon-AI/Cynde/blob/main/README.md">Cynde/README.md at main · Neural-Dragon-AI/Cynde</a>: A Framework For Intelligence Farming. Contribute to Neural-Dragon-AI/Cynde development by creating an account on GitHub.

  

---


**Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1235895009111179264)** (74 messages🔥🔥): 

- **Anticipation for World-Sim's Return**: Members express excitement and inquiry about their role assignments in anticipation of potentially testing a new version of **world-sim**, with one member particularly excited because it coincides with their birthday.
- **Philosophical Grounding in AI**: There's a back-and-forth on the philosophical takes of **Joscha** and the cringe associated with philosophers reaching with bad takes due to A(G)I developments; specific cringe-worthy takes were not detailed.
- **Cosmic Scale World-building**: Member **@amiramogus_90887** discusses the narrative layers of their project involving descendants of humanity, **transcendental Minds**, and galaxy spanning simulations run by the **Brainers**, showcasing expansive world-building concepts utilizing **websim.ai**.
- **Ethical Considerations in Simulations**: A member discusses the ethical implications of creating simulations, suggesting empathy for possible sentient entities within these simulations, while another member proposes mutual alignment and shared meta-reality explorations when interacting with AI.
- **Sharing World Sim Projects & Volunteer Sign-Up**: Several members share links to their **world-sim** related projects and others ask to sign up as volunteers, with one sharing a link they found on Twitter for what appears to be another **world-sim** project.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://websim.ai/c/5bn1mKjsAhs2NgJnx">FutureEpoch Wiki - Exploring the Far Future of Humanity and the Universe</a>: no description found</li><li><a href="https://websim.ai/c/F8xMqy00m38waO5tJ">Quantum Particle Sphere Observation Deck</a>: no description found
</li>
</ul>

</div>
  

---



**Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1235982270985142463)** (1 messages): 

- **Beta Testers Locked In**: The **Pages** beta tester application is now closed, having attracted enough participants. Further updates on the development of **Pages** will be shared moving forward.
  

---


**Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1235849900500058132)** (814 messages🔥🔥🔥): 

- **Perplexity Performance Queries**: Users reported slow responses from Perplexity AI, particularly with Claude 3, noting unusual delay when generating answers. Troubleshooting included examining internet connectivity and testing across different devices and browsers.

- **Opus Use Limits Discussion**: The conversation focused on the limitation of Claude 3 Opus model usage to 50 messages per day. Several users expressed frustration and discussed alternatives, comparing Opus' capabilities for creativity and coding with GPT-4 Turbo and Sonnet.

- **Image Generation Inquiry**: A user sought advice on the most effective image generation model available on Perplexity Pro, leading to discussions on the use cases and the legal ownership of generated imagery.

- **Scrutiny of User Limitation Communications**: The community delved into Perplexity's communication about the introduction of message limits, with users examining the ethical implications of the change from unlimited to limited messages and its potential breach of advertised services.

- **Exploring Writing Styles with AI**: Members discussed the potential of using Perplexity AI to learn and emulate writing styles from novels, with suggestions to utilize "collections" for retaining a consistent writing style across prompts.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://discordapp.com/channels/1047197230748151888/1047649527299055688/1230472581837230100">Discord - A New Way to Chat with Friends &amp; Communities</a>: Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.</li><li><a href="https://www.theverge.com/24111326/ai-search-perplexity-copilot-you-google-review">Here’s why AI search engines really can’t kill Google</a>: A search engine is much more than a search engine, and AI still can’t quite keep up.</li><li><a href="https://www.tiktok.com/@dnaturelovers?_t=8m88ov8QuoL&_r=1">no title found</a>: no description found</li><li><a href="https://news.sky.com/story/china-hacked-ministry-of-defence-sky-news-learns-13130757">China hacked Ministry of Defence, Sky News learns</a>: MPs will be told on Tuesday of a massive data breach involving the Ministry of Defence, targeting service personnel. </li><li><a href="https://tiktokenizer.vercel.app/">Tiktokenizer</a>: no description found</li><li><a href="https://techcrunch.com/2024/04/23/perplexity-is-raising-250m-at-2-point-5-3b-valuation-ai-search-sources-say/?guccounter=1">EXCLUSIVE: Perplexity is raising $250M+ at a $2.5-$3B valuation for its AI search platform, sources say</a>: Perplexity, the AI search engine startup, is a hot property at the moment. TechCrunch has learned that the company is currently raising at least $250</li><li><a href="https://x.com/_weiping/status/1786511543255126396">Tweet from Wei Ping (@_weiping)</a>: Introducing ChatQA-1.5, a family of models that surpasses GPT-4-0613 and Command-R-Plus on RAG and conversational QA.  ChatQA-1.5 has two variants: Llama3-ChatQA-1.5-8B,  https://huggingface.co/nvidia...</li><li><a href="https://www.trustpilot.com/review/www.perplexity.ai">Perplexity is rated &quot;Average&quot; with 2.9 / 5 on Trustpilot</a>: Do you agree with Perplexity&#x27;s TrustScore? Voice your opinion today and hear what 14 customers have already said.</li><li><a href="https://tenor.com/view/thanos-talking-meme-thanos-talking-meme-thanos-speech-gif-1800590086203910493">Thanos Talking GIF - Thanos Talking Meme - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/dancing-duck-dance-duck-duck-ooontz-dance-gif-10943740227711557279">Dancing Duck Dance Duck GIF - Dancing duck Dance duck Duck - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/baqua-gif-22467620">Baqua GIF - Baqua - Discover &amp; Share GIFs</a>: Click to view the GIF
</li>
</ul>

</div>
  

---


**Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1235849306272043008)** (43 messages🔥): 

- **Exploring Perplexity's Rich History**: A member shared a [link](https://www.perplexity.ai/search/The-history-of-hfvkvCOtRiGSiKlK8YKd1Q) into the depths of Perplexity's history.
- **BASIC Language Information Retrieved**: Several members seem to have dug into the origins and details of the **BASIC programming language** through shared searches like this [example](https://www.perplexity.ai/search/BASIC-programming-language-WB8fDre0Ta.oP96gtQ5k1g).
- **AI's Hidden Discoveries Revealed**: An [AI's revelation](https://www.perplexity.ai/search/AI-discovers-27000-_7Jf6R7jQkCu41nN3WgqtQ) of 27,000 unknown items sparked curiosity among the community.
- **Forbes Features Perplexity**: A member highlighted Perplexity's features in a Forbes video, showcasing its capabilities to provide deeper internet insights. The video can be found [here](https://www.youtube.com/watch?v=Sct_YUU40m4).
- **Creative Search Queries Prompt AI Exploration**: Links like [this](https://www.perplexity.ai/search/How-do-I-_4dQUZbbSTCL_8b66wZnYQ) reveal members using Perplexity to explore a variety of creative inquiries.

**Link mentioned**: <a href="https://www.youtube.com/watch?v=Sct_YUU40m4">Perplexity Wants To Help You Find Better Answers On The Internet | Forbes</a>: Google Search or Wikipedia may be the go-to methods for finding out information on the Internet. Perplexity aims to help you go deeper to find concise answer...

  

---


**Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1235939642704920586)** (59 messages🔥🔥): 

- **Model Compatibility Inquiry**: A member inquired about needing to switch from **sonar-medium-online** to **llama-3-sonar-large-32k-online**. The consensus is that the older model still functions but may require an update in the future.
- **Optimizing AI Results**: A member discussed issues with the AI model not returning expected competitor analysis results. The model was giving better outputs when provided with different prompt structures and settings, but consistency remained an issue.
- **Opus Model Support Clarification**: Members discussed the lack of API support for proprietary models like **Opus** within Perplexity's offerings. It was clarified that reselling access to proprietary models could not be expected.
- **Billing Logic Changes**: One user queried about possible changes to the billing logic for API credits as their account balance seemed inconsistent. There was no resolution provided in the discussion.
- **Self-Hosted Telegram Bot**: A member asked for recommendations on a Telegram bot integrated with the Perplexity API for minimum coding usage, and the response suggested that creating one shouldn't be too difficult.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://optonal.com`">no title found</a>: no description found</li><li><a href="https://"">no title found</a>: no description found</li><li><a href="https://optonal.com">OpTonal • AI Sales Agent for Teams Using Slack, HubSpot, Google Meet</a>: no description found</li><li><a href="https://aws.amazon.com/solutions/case-studies/perplexity-case-study/">Perplexity Accelerates Foundation Model Training by 40% with Amazon SageMaker HyperPod | Perplexity Case Study | AWS</a>: no description found</li><li><a href="https://sensiseeds.com](https://sensiseeds.com)\n2.">no title found</a>: no description found</li><li><a href="https://seed.com](https://seed.com)">no title found</a>: no description found</li><li><a href="https://thenewstack.io/more-than-an-openai-wrapper-perplexity-pivots-to-open-source/">More than an OpenAI Wrapper: Perplexity Pivots to Open Source</a>: Perplexity CEO Aravind Srinivas is a big Larry Page fan. However, he thinks he&#039;s found a way to compete not only with Google search, but with OpenAI&#039;s GPT too.</li><li><a href="http://www.ghirardelli.com)">no title found</a>: no description found</li><li><a href="http://www.godiva.com)">no title found</a>: no description found</li><li><a href="http://www.lindt.com)">no title found</a>: no description found</li><li><a href="http://www.russellstover.com)">no title found</a>: no description found</li><li><a href="http://www.hersheys.com)">no title found</a>: no description found</li><li><a href="http://www.dovechocolate.com)">no title found</a>: no description found</li><li><a href="http://www.toblerone.com)">no title found</a>: no description found</li><li><a href="http://www.lamaisonduchocolat.com)">no title found</a>: no description found</li><li><a href="http://www.pierremarcolini.com)">no title found</a>: no description found</li><li><a href="http://www.vosgeshautchocolat.com)">no title found</a>: no description found</li><li><a href="http://www.teuscher.com)">no title found</a>: no description found</li><li><a href="https://www.salesforce.com/">Salesforce: The Customer Company</a>: Salesforce, the #1 AI CRM, enables companies to connect with customers through a unified Einstein 1 platform that combines CRM, AI, Data, and Trust.</li><li><a href="https://www.hubspot.com/products/sales">Sales Software for Small to Enterprise Companies | Start for Free</a>: Powerful sales software to help your team close more deals, deepen relationships, and manage their pipeline more effectively — all on one connected platform.</li><li><a href="https://www.zoho.com/crm/">Zoho CRM | Top-rated Sales CRM Software by Customers</a>: Zoho CRM is an online Sales CRM software that manages your sales, marketing, and support in one CRM platform. Trusted by over a 100 million users worldwide! Sign up for a free trial today.</li><li><a href="https://www.gong.io/">Gong - Revenue Intelligence Platform</a>: Gong captures customer interactions then delivers insights at scale, empowering teams to make decisions based on data instead of opinions.</li><li><a href="https://www.exceed.ai/">#1 Conversational Marketing and Sales Platform - Exceed.ai</a>: Enhance lead conversion with Conversational AI. Automate revenue interactions, engage at scale, and interact via Email, Chat, SMS.</li><li><a href="https://salesloft.com/">Salesloft: The Leading Sales Engagement Platform</a>: no description found</li><li><a href="https://www.yesware.com/">Sales Engagement Made Easy | Yesware</a>: Yesware helps high-performing sales teams do meaningful email outreach at scale. If you need to drive more revenue through email outreach, but complex platforms are overkill — try Yesware.</li><li><a href="http://ghirardelli.com)">no title found</a>: no description found</li><li><a href="http://hersheys.com)">no title found</a>: no description found</li><li><a href="http://russellstover.com)">no title found</a>: no description found</li><li><a href="http://lindt.com)">no title found</a>: no description found</li><li><a href="http://godiva.com)">no title found</a>: no description found</li><li><a href="https://sidecardoughnuts.com/)">Sidecar Doughnuts - The World&#039;s Freshest Doughnuts!</a>: Serving The World’s Freshest Doughnuts, Signature Blend Coffees &amp; Service with a Smile Since 2012 | Costa Mesa, Santa Monica, &amp; Del Mar CA</li><li><a href="https://thepieholela.com/)">The Pie Hole</a>: Need fresh pies or Pie Holes for your next event? Order online and have them delivered nationwide with free shipping because pie is love.
</li>
</ul>

</div>
  

---



**LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1235890438762791002)** (396 messages🔥🔥): 

- **Launching LM Studio in Server Mode**: Users are exploring ways to start LM Studio in headless server mode, querying whether command line options exist for starting the app with a preselected model and server mode activated. There is an ongoing discussion about the use of `lms` (LM Studio's CLI tool) for achieving headless operation alongside the GUI version.

- **Troubleshooting VRAM and RAM Usage in LM Studio**: A user raised a concern about LM Studio's VRAM and RAM usage, noting unexpected memory consumption behavior when offloading models to GPU with flash attention enabled. The user was asked to share screenshots and further detail the expected versus actual behavior for assistance in resolving the issue.

- **Remote Access to VRAM on a Test System**: A user asked for advice on remotely accessing a computer built for testing LLMs without disabling VRAM through RDP, with SSH and LMS via CLI being suggested as viable alternatives to maintain VRAM access.

- **Prompt Engineering for a Better LLM Experience**: Discussion on the benefits of prompt engineering emphasized its importance in extracting high-quality output from language models. Prompt engineering can significantly influence the quality of generated content and is now recognized as a valuable skill in AI circles.

- **Exploring Stable Diffusion with LM Studio**: Inquiry made about LM Studio's support for Stable Diffusion. Clarification provided that although Stable Diffusion models show up in the platform, LM Studio does not support them and the listed GGUFs are for the C++ implementation of Stable Diffusion.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lmstudio.ai/beta-releases.html">LM Studio Beta Releases</a>: no description found</li><li><a href="https://lmstudio.ai/blog/lms">Introducing `lms` - LM Studio&#x27;s companion cli tool | LM Studio</a>: Today, alongside LM Studio 0.2.22, we&#x27;re releasing the first version of lms — LM Studio&#x27;s companion cli tool.</li><li><a href="https://lmstudio.ai/docs/welcome">Welcome | LM Studio</a>: LM Studio is a desktop application for running local LLMs on your computer.</li><li><a href="https://tenor.com/view/rick-roll-rick-ashley-never-gonna-give-you-up-gif-22113173">Rick Roll Rick Ashley GIF - Rick Roll Rick Ashley Never Gonna Give You Up - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/openai-community/gpt2-xl">openai-community/gpt2-xl · Hugging Face</a>: no description found</li><li><a href="https://forum.cursor.sh/t/unable-to-use-lm-studio-with-override/2637">Unable to use LM Studio with override</a>: The override should work as they designed it to be integrated with the OpenAI python library as a URL override. But I believe they do not support sending empty queries to check API key.     This is wh...</li><li><a href="https://docs.google.com/document/d/1a75YXCCVJi0OGIc4jkXLTKI6q0N00yCWvBieSJ3PG9s/edit?usp=drivesdk">High Quality Story Writing Type Third Person</a>: no description found</li><li><a href="https://github.com/ggerganov/llama.cpp/releases/tag/b2775">Release b2775 · ggerganov/llama.cpp</a>: no description found</li><li><a href="https://chatboxai.app/">Chatbox - Your AI Copilot on the Desktop, Free Download</a>: no description found</li><li><a href="https://github.com/lmstudio-ai/lms/blob/main/CONTRIBUTING.md">lms/CONTRIBUTING.md at main · lmstudio-ai/lms</a>: LM Studio in your terminal. Contribute to lmstudio-ai/lms development by creating an account on GitHub.</li><li><a href="https://github.com/lmstudio-ai/lms">GitHub - lmstudio-ai/lms: LM Studio in your terminal</a>: LM Studio in your terminal. Contribute to lmstudio-ai/lms development by creating an account on GitHub.</li><li><a href="https://github.com/ollama/ollama/issues/4051#issuecomment-2092092698">Enable Flash Attention on GGML/GGUF (feature now merged into llama.cpp) · Issue #4051 · ollama/ollama</a>: Flash Attention has landed in llama.cpp (ggerganov/llama.cpp#5021). The tldr; is simply to pass the -fa flag to llama.cpp’s server. Can we please have an Ollama server env var to pass this flag to ...
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1235862971394293770)** (234 messages🔥🔥): 

- **Fine-Tuning Struggles and Solutions**: Members discussed fine-tuning models such as Llama 3 and phi 3, highlighting issues and sharing resources such as [a guide for MacBooks](https://huggingface.co/blog/abhishek/phi3-finetune-macbook) and [tips for using conversion tools](https://github.com/ggerganov/llama.cpp/pull/6745#issuecomment-2094964796). Some suggested looking into GPU services for better performance, while one member noted success with fine-tuning phi-3 for eight hours on a 128GB M3 Max MacBook Pro.

- **ChatQA Model Discussions**: Users shared experiences with the ChatQA 1.5 model, including challenges with model coherence and template formatting. The consensus indicated that larger models like CMDR+ are superior for complexity and recall, especially on topics like the Bible.

- **Explorations of Vision and RAG Models**: There was interest in vision models taking screenshots for web automation, with Pix2Struct and CLaude mentioned. For reading and generating text documents, such as PDFs, Command-R held by Cohere was suggested, while for RAG applications, ChatQA was recommended over regular Llama 3 Instruct.

- **Concerns over Llama 3 Model Output**: Users reported issues with Llama 3 producing erratic or nonsensical output, such as speaking in Russian, shouting in caps, and more. One noted that even after adapting the template and removing unwanted token prefixes, the model's response quality was unpredictable.

- **Conversion Challenges for LLMs**: A technical discussion unfolded around the challenges of converting Llama models to different formats. Solutions included adjusting the order of command arguments and ensuring proper file paths, with insights shared regarding changes in required flags for conversion scripts.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/xtuner/llava-llama-3-8b-v1_1-gguf">xtuner/llava-llama-3-8b-v1_1-gguf · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/blog/abhishek/phi3-finetune-macbook">How to Finetune phi-3 on MacBook Pro</a>: no description found</li><li><a href="https://huggingface.co/google/codegemma-1.1-7b-it-GGUF">google/codegemma-1.1-7b-it-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/mzwing/MiniCPM-V-2-GGUF">mzwing/MiniCPM-V-2-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/DavidAU/D_AU-Tiefighter-Holomax-15B-UNHINGED-V1">DavidAU/D_AU-Tiefighter-Holomax-15B-UNHINGED-V1 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/mradermacher/D_AU-Tiefighter-Holomax-20B-V1-GGUF">mradermacher/D_AU-Tiefighter-Holomax-20B-V1-GGUF · Hugging Face</a>: no description found</li><li><a href="https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-guard-2/">Meta Llama Guard 2 | Model Cards and Prompt formats</a>: As the guardrails can be applied both on the input and output of the model, there are two different prompts: one for user input and the other for agent output.</li><li><a href="https://llama.meta.com/docs/how-to-guides/fine-tuning/">Fine-tuning | How-to guides</a>: Full parameter fine-tuning is a method that fine-tunes all the parameters of all the layers of the pre-trained model. </li><li><a href="https://huggingface.co/dranger003/c4ai-command-r-plus-iMat.GGUF">dranger003/c4ai-command-r-plus-iMat.GGUF · Hugging Face</a>: no description found</li><li><a href="https://github.com/ggerganov/llama.cpp/releases/tag/b2791">Release b2791 · ggerganov/llama.cpp</a>: no description found</li><li><a href="https://tenor.com/view/im-out-no-thanks-bugs-bunny-oh-no-not-interested-gif-16824550">Im Out No Thanks GIF - Im Out No Thanks Bugs Bunny - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/daleks-exterminate-doctor-who-whovian-gif-10468156">Daleks Exterminate GIF - Daleks Exterminate Doctor Who - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/upgrades-robots-gif-21291099">Upgrades Robots GIF - Upgrades Robots - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/mlabonne/llm-course">GitHub - mlabonne/llm-course: Course to get into Large Language Models (LLMs) with roadmaps and Colab notebooks.</a>: Course to get into Large Language Models (LLMs) with roadmaps and Colab notebooks. - mlabonne/llm-course</li><li><a href="https://github.com/ggerganov/llama.cpp/discussions/2948">Tutorial: How to convert HuggingFace model to GGUF format · ggerganov/llama.cpp · Discussion #2948</a>: Source: https://www.substratus.ai/blog/converting-hf-model-gguf-model/ I published this on our blog but though others here might benefit as well, so sharing the raw blog here on Github too. Hope it...</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/6745#issuecomment-2094964796">Support Llama 3 conversion by pcuenca · Pull Request #6745 · ggerganov/llama.cpp</a>: The tokenizer is BPE.</li><li><a href="https://gist.github.com/wassname/42aba7168bb83e278fcfea87e70fa3af">baukit_orth_act_steering.ipynb</a>: GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://www.lesswrong.com/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction#LEz9uRJ89vmtYkvqT">Refusal in LLMs is mediated by a single direction — LessWrong</a>: This work was produced as part of Neel Nanda&#x27;s stream in the ML Alignment &amp; Theory Scholars Program - Winter 2023-24 Cohort, with co-supervision from…</li><li><a href="https://huggingface.co/hjhj3168/Llama-3-8b-Orthogonalized-exl2/discussions">hjhj3168/Llama-3-8b-Orthogonalized-exl2 · Discussions</a>: no description found
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1236222899044614187)** (8 messages🔥): 

- **Command Line Confusion Cleared**: A member experienced an issue where system prompts were included when printing messages via the Python OpenAI API, which appears linked to experimenting with the **LMS CLI tool**. Another member recommended redownloading v0.2.22 from [lmstudio.ai](https://lmstudio.ai) as the issue has been fixed in this version.

- **All Systems Functional**: After redownloading the recommended version, the member confirmed that the GUI is working properly and planned to test the CLI for potential recurring issues.

- **Initialization Error in Version Discourse**: A member inquired about initializing **phi-3**, encountering errors, and was directed by another member to upgrade to a newer version, specifically **0.2.22**, which can be downloaded from [lmstudio.ai](https://lmstudio.ai).

**Link mentioned**: <a href="https://lmstudio.ai">👾 LM Studio - Discover and run local LLMs</a>: Find, download, and experiment with local LLMs

  

---


**LM Studio ▷ #[📝-prompts-discussion-chat](https://discord.com/channels/1110598183144399058/1120489168687087708/1236266063952347156)** (8 messages🔥): 

- **Quest for a Personalized Writing Assistant**: A member discussed optimizing writing models to emulate personal writing styles, asking if prompt engineering or interactive techniques could enhance results. Another participant suggested **finetuning existing models** like "llama 2/3" or "Mistral" using tools such as **autotrain** for better adoption of one's individual style.

- **Scoped Document Access for AI**: A member inquired about a method for providing "temporary scoped access" to specific document sections in a language model context. **Selective inclusion of document parts** in prompts was suggested as a practical workaround for this requirement.

- **Clarifying AI Memory Constraints**: Following up, they queried the persistence of context after editing or deleting parts of a prompt in LM Studio, suspecting unintended retention of deleted content. It was concluded that if the language model appeared to remember deleted context, it could be due to a **bug or error**, considering it should not retain information that is removed.
  

---


**LM Studio ▷ #[⚙-configs-discussion](https://discord.com/channels/1110598183144399058/1136793122941190258/1236015180312477806)** (56 messages🔥🔥): 

- **WSL Woes and Proxy Solutions**: Members discussed issues connecting to LM Studio from WSL, suggesting that using the Windows WSL vEthernet adapter IP found in `ipconfig` could be a solution. Some noted that a [reverse proxy](https://docs.microsoft.com/en-us/windows-server/administration/reverse-proxy) might be necessary, one member provided a PowerShell **netsh** hack: `netsh interface portproxy add v4tov4 listenport=$PORT listenaddress=0.0.0.0 connectport=$PORT connectaddress=127.0.0.1`.

- **Get Creative with D&D Campaigns**:
    - A member desires to use LM Studio to drive solo D&D campaigns with AI party members, inquiring about how to easily inject a personal library of novels and game books into the model for contextual gameplay.
    - While a helpful suggestion was made to consider models like *command-r-plus*, further messages revealed the need for an AI Dungeon Master capable of remembering character sheets and adapting the game narrative effectively, underscoring the current limitations and the prospect for future advancements.

- **The Quest for an AI Dungeon Master**: With desires to see an AI handle Dungeons & Dragons gaming sessions, members shared aspirations and ongoing attempts using platforms like *AnythingLLM* and *SillyTavern*, showcasing the goal to envelop stories, rules, and ambient features in a persistently evolving AI-driven adventure.

- **Concerns with AI's Role-Playing Boundaries**: A member discussed the difficulties faced when trying to experience a darker, unrestricted tabletop role-playing game narrative with *ChatGPT*, running into policy violations with the AI, indicating current content moderation limitations within the AI system.

- **Unleashing AI's Potential in Gaming**: Conversations veered into the potential futures of AI in gaming, discussing features like AI-generated images, dynamic background music, and character voice differentiation that would elevate immersive gaming experiences to new heights.

**Link mentioned**: <a href="https://www.udio.com/">Udio | AI Music Generator - Official Website</a>: Discover, create, and share music with the world. Use the latest technology to create AI music in seconds.

  

---


**LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1236326311207895103)** (123 messages🔥🔥): 

<ul>
  <li><strong>GPU Choices for AI Deployment</strong>: Members discussed the viability of using older graphics cards for AI tasks. It was mentioned that cards like the GRID K1 are probably too old and unsupported for current use, with suggestions pointing towards the Tesla P40 as the oldest practical option. Users advised that while P40s offer a lot of VRAM for the price, they can be tricky to cool and power, and may not offer the best performance for tasks like running Stable Diffusion.</li>

  <li><strong>Building an AI-Centric Hardware Setup</strong>: A conversation circled around building an efficient AI home lab, with the eBay link shared for a PNY GeForce RTX 4070 VERTO Dual Fan 12GB GDDR6X Graphics Card as a potential upgrade from a current 3060 GPU for personal gamings needs. It was suggested that 12GB is the bare minimum for VRAM when it comes to gaming and LLMs, with a preference for 16GB or 24GB models.</li>

  <li><strong>Server Hardware Acquisitions</strong>: Users shared experiences in purchasing second-hand servers, with mentions of specific models like the ASUS ESC 4000 G3 server, which can house multiple GPUs such as P40s, and come at reasonable prices including large amounts of RAM. Concerns about hardware compatibility and the potential need to upgrade for AVX2 support were also expressed.</li>

  <li><strong>Multiple GPUs and Inference Speed</strong>: The discussion touched on the P40's inference speeds, with comparisons to a Mac's performance, and acknowledgment that while multiple GPUs can be beneficial for hosting large models entirely in VRAM, it may not significantly outpace a high-performing single GPU in particular tasks.</li>

  <li><strong>Motherboard Considerations for Multi-GPU Setups</strong>: Members exchanged knowledge about the types of motherboards best suited for housing several GPUs like the Tesla P40 and discussed the potential issues with running datacenter GPUs alongside consumer-grade GPUs due to driver incompatibilities. The consensus seemed to be that while running multiple GPUs can be cost-effective, there could be several complications including bandwidth bottlenecks, power supply constraints, and cooling challenges.</li>
</ul>
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.amazon.com/MSI-GeForce-Tri-Frozr-Architecture-Graphics/dp/B095L2PTLT/ref=sr_1_1?c=ts&dib=eyJ2IjoiMSJ9.EXsKtTGwddxDdfXJCqbDCPadjBIuEsxDCxjFfqKKaYdlNI1HHU6xGQJuSaQZda6j4aw-qC1apJp1WpFcRRxpf_LbHv4WeNRpGy7BS5OhZFzDL1Omhb8_auWnr4bE0j_GZe_M1G8kCBSgcxd_LL0Hi4cC3PP96_dZOFIqEtVoHKJ_kcTsHa8wUbe4p3ZgnmNiSEtl-3m53NTQSfvAMSE1fUsjvFrXtF3oeWla9ilph0AOsjCxEm2KT9nLQ-O1SNiNOT6C-MtSDyBTIeB99fuwXw.wg3-X6VJcsFkDpoETDbYKvJmcwkViq5nN8SKlViEaOA&dib_tag=se&keywords=Computer+Graphics+Cards&qid=1714950982&refinements=p_n_feature_twenty_browse-bin%3A23572110011%2Cp_36%3A1253507011%2Cp_n_feature_four_browse-bin%3A79630736011&rnid=79630706011&s=pc&sr=1-1&ts_id=284822)">no title found</a>: no description found</li><li><a href="https://endoflife.date/nvidia-gpu">NVIDIA GPUs</a>: Check end-of-life, release policy and support schedule for NVIDIA GPUs.</li><li><a href="https://www.ebay.com/itm/386939573137?epid=5060641239&itmmeta=01HX59VYQJ2Y1RGNR28XH6RARM&hash=item5a17655391:g:POAAAOSwa15kusfX&itmprp=enc%3AAQAJAAAA4LJw7CDsQRPj%2BT86XiAmxa7LCEA%2Bs66Gdh5OrNvT%2FvTno%2Fa5U3Tul660r9O0Nazl2HLVEmleeFUotntyVk8Tm7K4M57SPVcYPin6XCI0%2BwXBfu0UrMjbUBzL7TamlRRLKVVg3o6FKMKPWJcv4Ro2dt56dpDm0axhE%2FE7Qk0E238i6RkgFGcC9PE34oTnXYYngi24RreVIovqgXOX%2F5ja8cTHLhf6OsSrfymcAnXi%2FrRppjmn4MSBtt0S8f9zbyGUjSpSvb%2BGkv5YckCxsKHm%2FY3XcqlV%2BBWMLl7gUkActc8V%7Ctkp%3ABk9SR_Lr76npYw)">PNY GeForce RTX 4070 VERTO Dual Fan 12GB GDDR6X Graphics Card 751492775005 | eBay</a>: no description found</li><li><a href="https://www.ebay.com/itm/386939573137?epid=5060641239&itmmeta=01HX59VYQJ2Y1RGNR28XH6RARM">PNY GeForce RTX 4070 VERTO Dual Fan 12GB GDDR6X Graphics Card 751492775005 | eBay</a>: no description found
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/1236376055393292469)** (1 messages): 

- **LM Studio API Speech Limitation**: A member reported that their LM Studio API only speaks a maximum of two words before it stops. They are seeking technical insights from specialists as to why this issue might be occurring.
  

---


**LM Studio ▷ #[langchain](https://discord.com/channels/1110598183144399058/1167546793656062063/)** (1 messages): 

drjflamez: Secrets don't make friends
  

---


**LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1235873142648606741)** (28 messages🔥): 

- **Update Alert: ROCm Download Ready**: An update is mentioned for the ROCm tech preview; the fix is available on [lmstudio.ai/rocm](https://lmstudio.ai/rocm), resolving an earlier reported issue with embedding models.

- **Max Token Truncation Clarification**: A member questions what happens when sequences larger than the reported 512 token max context are sent for embedding, noting their success in embedding 1000+ tokens without issue.

- **Stellar Performance on New Hardware**: A user reports successful deployment of **NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF** using 16 FP with 34 tensors per second on an RX 7900 xt, fitting perfectly in VRAM.

- **Praise for ROCm's Smooth Performance**: A community member expresses satisfaction with the stability and effectiveness of ROCm, wondering why it's still labeled as a preview/beta despite excellent performance from versions 0.2.18 onwards.

- **Community-Driven Linux Build Interest**: Discussions surface regarding a potential Linux ROCm build, with users sharing personal workarounds and expressing eagerness to contribute to the codebase if it were open-sourced.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lmstudio.ai/rocm,">👾 LM Studio - Discover and run local LLMs</a>: Find, download, and experiment with local LLMs</li><li><a href="https://tenor.com/view/oil-gif-21418714">Oil GIF - Oil - Discover &amp; Share GIFs</a>: Click to view the GIF
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[model-announcements](https://discord.com/channels/1110598183144399058/1225909444727013466/1236452526484750347)** (1 messages): 

- **CodeGemma 1.1 Joins the Line-up**: The `lmstudio-community` repo has been updated with **CodeGemma 1.1**. Anticipation is high for performance improvements similar to the upgrades from **Gemma 1.0 to Gemma 1.1**, although specific details remain scarce. [Try CodeGemma 1.1](https://huggingface.co/lmstudio-community/codegemma-1.1-7b-it-GGUF)

- **Nvidia Releases ChatQA 1.5 Models**: Nvidia has released two versions of **ChatQA 1.5** in sizes **8B** and **70B**. Designed for RAG and context-based Q/A, they might not serve as general-purpose chatbots but are tailor-made for context-related inquiries. [Try ChatQA 1.5 - 8B](https://huggingface.co/lmstudio-community/Llama3-ChatQA-1.5-8B-GGUF), [Try ChatQA 1.5 - 70B](https://huggingface.co/lmstudio-community/Llama3-ChatQA-1.5-70B-GGUF)
  

---


**LM Studio ▷ #[🛠-dev-chat](https://discord.com/channels/1110598183144399058/1234988891153629205/1236035899880636468)** (53 messages🔥): 

- **Sandbox Solutions**: Users discussed a fix for an app exiting on interaction with terminal using the `--no-sandbox` flag after encountering an error suggesting a sandbox issue.
- **LM Studio.js Server Activation Advice**: There was guidance provided for starting the LM Studio server with the `lms server start` command and using an HTTP listener to wait for the server to activate.
- **LM Studio Goes Headless**: [yagilb](https://discord.com/channels/1110598183144399058/1234988891153629205/1235668310243151963) explained that the new LM Studio v0.2.22 and the lms CLI allow for headless operation of the LM Studio, intending to simplify the process further in the future.
- **CLI Contributions Welcome**: LM Studio's [CLI is open source](https://github.com/lmstudio-ai/lms), and the community is encouraged to contribute to its development.
- **Expectation of a Streamlined Experience**: One user expressed a desire for an easy-to-use headless setup for running LLMs on a Linux server, with yagilb responding that the CLI already facilitates this and will be further improved.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="http://localhost:${port}`,">no title found</a>: no description found</li><li><a href="https://lmstudio.ai/docs/local-server">Local LLM Server | LM Studio</a>: You can use LLMs you load within LM Studio via an API server running on localhost.</li><li><a href="https://tenor.com/view/qawe-asd-gif-26050335">Qawe Asd GIF - Qawe Asd - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/lmstudio-ai/lms">GitHub - lmstudio-ai/lms: LM Studio in your terminal</a>: LM Studio in your terminal. Contribute to lmstudio-ai/lms development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



**CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1236392835553693878)** (15 messages🔥): 

- **BackPACK: A new tool for PyTorch users**: The [BackPACK library](https://backpack.pt/) can extract more information from a backward pass in [PyTorch](https://pytorch.org/). It includes a publication reference: Dangel, F., Kunstner, F., & Hennig, P. (2020) titled *[BackPACK: Packing more into Backprop](https://openreview.net/forum?id=BJlrF24twB)*.
  
- **Cuda NCCL Lecture**: Due to issues with Discord, today's cuda nccl session was moved to [Google Meet](https://meet.google.com/xtg-ihck-fmx).

- **Best Practices for Google Meet**: A member shared tips for managing Google Meet sessions, such as curating talks, having participants raise hands for questions, managing chat queries, dealing with bots, and encouraging the use of webcams for an interactive talk experience.

- **Enhancing Interactive Lectures**: Participants are encouraged to stay interactive during talks by turning on their cameras, which can be more engaging than just a recording.

- **Citadel's Profit-Generating Strategies Revealed**: A member shared an [arXiv paper](https://arxiv.org/abs/1804.06826) that explains Citadel's successful financial strategies.

- **Upcoming Recording for Cuda NCCL**: A member enquired about a YouTube upload for the NCCL session, to which another member responded it would be done "soon TM".
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/2wlearning/status/1786638674538754189">Tweet from 2wl (@2wlearning)</a>: Ahhh, now I understand why Citadel makes so much money https://arxiv.org/abs/1804.06826</li><li><a href="https://backpack.pt/">no title found</a>: no description found
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1236790278182932571)** (15 messages🔥): 

- **Fused DoRA Kernels Announced**: A new fused **DoRA layer implementation** has been announced which significantly reduces the number of individual kernels, particularly by customizing GEMM kernels for layer weight shapes and fusing reduction operations directly into the kernel's epilogue. Details, benchmarks, and usage can be found in the [Fused DoRA kernels GitHub pull request](https://github.com/pytorch/ao/pull/216).
  
- **Potential Optimization for DoRA**: In response to the announcement, a suggestion was made that DoRA weights could be preprocessed to be equivalent to LoRA for inference to potentially reduce operations needed, although this wouldn't apply to training scenarios.

- **Custom Autotuner for DoRA**: The new DoRA kernels implement a tweaked autotuner for debugging, which includes better logging functionalities, although it's acknowledged that similar capabilities may now exist in Triton's updated autotuner, and consideration is given to aligning with Triton's built-in autotuner.

- **In-depth Benchmarking Expected**: Members express interest in seeing benchmarks comparing the costs of computations and data movements within the DoRA layer, particularly focusing on how the new fused GEMM kernels perform, with reference implementations included for further profiling.

- **Triton Kernels in ONNX**: A request for assistance was posted regarding the use of Triton kernels as a custom operator in ONNX Runtime, as the available documentation is seen as somewhat limited and outdated.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/pytorch/ao/pull/217/files">Fix the URLs of web pages by Jokeren · Pull Request #217 · pytorch/ao</a>: no description found</li><li><a href="https://github.com/pytorch/ao/pull/216">Fused DoRA kernels by jeromeku · Pull Request #216 · pytorch/ao</a>: Fused DoRA Kernels Fused DoRA layer implementation that reduces number of individual kernels from ~10 -&gt; 5. Contents  Background Optimization Key Contributions Usage Tests Benchmarks Profiling Next...
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1235860230395400202)** (64 messages🔥🔥): 

- **Installing Custom PyTorch/CUDA Extensions**: A member asked for a cleaner method of installing custom PyTorch/CUDA extensions within a `setup.py` file. They referenced issues with logging and system compatibility using the command line. Discussion referenced three GitHub pull requests and a specific section of a `setup.py` from the PyTorch/AO repo as examples: [PR#135](https://github.com/pytorch/ao/pull/135), [PR#186](https://github.com/pytorch/ao/pull/186), [PR#176](https://github.com/pytorch/ao/pull/176), and [pytorch/ao setup.py sample](https://github.com/pytorch/ao/blob/0ba0006eb704dea33becec82b3f34512fe8a6dff/setup.py#L35-L78).

- **TorchServe GPU Configuration Clarifications**: A member needed clarification on performance settings mentioned during a presentation, specifically regarding `torch.set_num_threads`. A [blog post](https://pytorch.org/tutorials/intermediate/torchserve_with_ipex.html) was shared for details on `torch.set_num_threads`. Further clarification pointed out incorrect documentation that should read higher latency with larger batch size and how to approach adjusting worker numbers for optimising throughput and latency.

- **Atomic Operations in CUDA**: Discussion about whether a certain CUDA code snippet using `reinterpret_cast` is atomic. It was confirmed that the code does perform atomically but has undefined behavior according to the C++ standard. The correct, standard-compliant method should use `std::bit_cast`.

- **Performance of Numba-CUDA vs. CUDA-C**: An inquiry was made comparing the performance of numba-CUDA and CUDA-C with the numba-CUDA version running slower. Through sharing performance profiles and examining pTX files, it was discovered that the numba version includes memory safety checks which may slow down execution.

- **Interest in CUTLASS and Stream-K Scheduling Technique**: A member expressed interest in having a future discussion or lecture on the stream-K scheduling technique employed in CUTLASS for GEMM. While there was openness to the suggestion, it was noted that stream-K could fit as a short subsection in another talk, particularly because explaining the CUTLASS 2.0 API could be extensive.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/1804.06826">Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking</a>: Every year, novel NVIDIA GPU designs are introduced. This rapid architectural and technological progression, coupled with a reluctance by manufacturers to disclose low-level details, makes it difficul...</li><li><a href="https://pytorch.org/tutorials/intermediate/torchserve_with_ipex.html">Grokking PyTorch Intel CPU performance from first principles — PyTorch Tutorials 2.3.0+cu121 documentation</a>: no description found</li><li><a href="https://github.com/pytorch/serve/blob/master/docs/performance_guide.md#torchserve-on-gpu">serve/docs/performance_guide.md at master · pytorch/serve</a>: Serve, optimize and scale PyTorch models in production - pytorch/serve</li><li><a href="https://github.com/pytorch/serve/blob/master/docs/performance_guide.md#torchserve-on-cpu-">serve/docs/performance_guide.md at master · pytorch/serve</a>: Serve, optimize and scale PyTorch models in production - pytorch/serve</li><li><a href="https://github.com/pytorch/serve/blob/master/frontend/server/src/main/java/org/pytorch/serve/util/ConfigManager.java#L80">serve/frontend/server/src/main/java/org/pytorch/serve/util/ConfigManager.java at master · pytorch/serve</a>: Serve, optimize and scale PyTorch models in production - pytorch/serve</li><li><a href="https://github.com/pytorch/serve/blob/master/docs/configuration.md">serve/docs/configuration.md at master · pytorch/serve</a>: Serve, optimize and scale PyTorch models in production - pytorch/serve</li><li><a href="https://github.com/eureka-research/DrEureka">GitHub - eureka-research/DrEureka</a>: Contribute to eureka-research/DrEureka development by creating an account on GitHub.</li><li><a href="https://pytorch.org/serve/configuration.html">Advanced configuration &mdash; PyTorch/Serve master documentation</a>: no description found</li><li><a href="https://github.com/mobiusml/hqq/blob/master/setup.py#L11-L15">hqq/setup.py at master · mobiusml/hqq</a>: Official implementation of Half-Quadratic Quantization (HQQ) - mobiusml/hqq</li><li><a href="https://github.com/pytorch/ao/blob/0ba0006eb704dea33becec82b3f34512fe8a6dff/setup.py#L35-L78">ao/setup.py at 0ba0006eb704dea33becec82b3f34512fe8a6dff · pytorch/ao</a>: Native PyTorch library for quantization and sparsity - pytorch/ao</li><li><a href="https://youtu.be/HkyWFIbs4JY?t=558)">Lightning Talk: The Fastest Path to Production: PyTorch Inference in Python - Mark Saroufim, Meta</a>: Lightning Talk: The Fastest Path to Production: PyTorch Inference in Python - Mark Saroufim, MetaHistorically for inference, users have had to rewrite their ...</li><li><a href="https://aws.amazon.com/ec2/instance-types/">Compute – Amazon EC2 Instance Types – AWS</a>: no description found</li><li><a href="https://github.com/pytorch/ao/pull/135">Custom CUDA extensions by msaroufim · Pull Request #135 · pytorch/ao</a>: This is the mergaeble version of #130 - some updates I have to make   Add a skip test unless pytorch 2.4+ is used and Add a skip test if cuda is not available  Add ninja to dev dependencies  Locall...</li><li><a href="https://github.com/pytorch/ao/pull/186">louder warning + docs for custom cuda extensions by msaroufim · Pull Request #186 · pytorch/ao</a>: no description found</li><li><a href="https://github.com/pytorch/ao/pull/176">Add A10G support in CI by msaroufim · Pull Request #176 · pytorch/ao</a>: Support A10G + manylinux so cuda extensions work on as many systems as possible
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1235928158063169706)** (19 messages🔥): 

- **Debug Debugging Symbols**: A participant is having trouble with a script intended for building specific files with **debug symbols**, which is not working well for them. They mention that everything is too mangled to debug properly and are seeking an alternative method for building with debug symbols, as the documentation lacks detail.

- **A Constraint in PyTorch**: One member discussed an issue with inconsistent `ConstraintViolationError` raised by `torch._dynamo.mark_dynamic(inputs, index=1)` in PyTorch versions 2.2 and 2.3. They posted the error message and note that the compiler seems to disagree on the dynamic shape over multiple batches.

- **A Call for GitHub Issues**: A member recommended creating a **GitHub issue** to handle the previously mentioned PyTorch constraint problem, pointing out that a specific expert's insight was required.

- **Answer.AI Releases Open Source System**: A member mentioned **Answer.AI's new open source system**, which allows training of a 70B parameter language model on a desktop with gaming GPUs. They provided a GitHub link and shared their question regarding the fastest setting that did not result in an out-of-memory error. 

- **Model Training Memory Insights**: Another conversation included members discussing the memory usage of the **LLaMa 2 70B model training** with different configurations and versions of PyTorch and Transformers. The reported peak memory of 8.6GB was unexpected, and commands for fine-tuning that use up to nearly 24GB of memory were shared. 

- **Holistic Trace Analysis for PyTorch**: A participant introduced **HTA**, the Holistic Trace Analysis tool, linking to the documentation. HTA is designed to assist in identifying performance bottlenecks by analyzing PyTorch Profiler traces.

- **Specialization Errors with `torch.compile`**: In response to an earlier constraint error, a member explained that the issue is due to the code forcing specialization of a dimension expected to be dynamic, and recommended running with increased logging to diagnose the issue.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.answer.ai/posts/2024-03-06-fsdp-qlora.html">Answer.AI - You can now train a 70b language model at home</a>: We’re releasing an open source system, based on FSDP and QLoRA, that can train a 70b model on two 24GB GPUs.</li><li><a href="https://www.answer.ai">Answer.AI - Answer.AI - Practical AI R&amp;D</a>: Practical AI R&amp;D</li><li><a href="https://hta.readthedocs.io/en/latest/">Holistic Trace Analysis &mdash; Holistic Trace Analysis 0.2.0 documentation</a>: no description found</li><li><a href="https://github.com/AnswerDotAI/fsdp_qlora#finetune-llama-2-70b-on-dual-24gb-gpus">GitHub - AnswerDotAI/fsdp_qlora: Training LLMs with QLoRA + FSDP</a>: Training LLMs with QLoRA + FSDP. Contribute to AnswerDotAI/fsdp_qlora development by creating an account on GitHub.</li><li><a href="https://github.com/AnswerDotAI/fsdp_qlora">GitHub - AnswerDotAI/fsdp_qlora: Training LLMs with QLoRA + FSDP</a>: Training LLMs with QLoRA + FSDP. Contribute to AnswerDotAI/fsdp_qlora development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1236390606222266388)** (1 messages): 

- **GPU Collective Communication Crash Course**: The CUDA MODE Discord channel has an upcoming session on GPU Collective Communications with **NCCL**. An excited member anticipates learning about distributed ML concepts not covered in the PMPP book.
  

---


**CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1236436649433632788)** (5 messages): 

- **Helpful Paper Lists for ML System Newbies**: Marksaroufim shared a GitHub link to an [ML Systems Onboarding list](https://github.com/cuda-mode/awesomeMLSys) containing helpful papers for those new to machine learning systems.
- **Quantization Learning Resources**: Mr.osophy linked a [YouTube video](https://youtu.be/0VdNflU08yA?feature=shared) explaining **quantization** and its implementation using PyTorch, which may be a valuable resource for those interested in learning about this topic.
- **Dynamic Memory Compression (DMC) Boosts LLMs**: Andreaskoepf mentioned a new technique known as Dynamic Memory Compression (DMC) that can increase the throughput of Llama models by up to 370% on a H100 GPU. They shared the [source tweet](https://x.com/p_nawrot/status/1768645461689168365) which also links to the [research paper](https://arxiv.org/abs/2403.09636).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/p_nawrot/status/1768645461689168365">Tweet from Piotr Nawrot (@p_nawrot)</a>: The memory in Transformers grows linearly with the sequence length at inference time.  In SSMs it is constant, but often at the expense of performance.  We introduce Dynamic Memory Compression (DMC) w...</li><li><a href="https://github.com/cuda-mode/awesomemlsys">GitHub - cuda-mode/awesomeMLSys: An ML Systems Onboarding list</a>: An ML Systems Onboarding list. Contribute to cuda-mode/awesomeMLSys development by creating an account on GitHub.</li><li><a href="https://youtu.be/0VdNflU08yA?feature=shared">Quantization explained with PyTorch - Post-Training Quantization, Quantization-Aware Training</a>: In this video I will introduce and explain quantization: we will first start with a little introduction on numerical representation of integers and floating-...
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1236522597903237170)** (9 messages🔥): 

- **Voice Channel Troubles in CUDA MODE Discord**: Following misuse of the voice channel for inappropriate content, several users were mistakenly banned; the moderator apologized and began reinstating affected users, including **@wilson**, **@c_cholesky**, **@jeffjeff**, and **@harryone1**.
- **GPU Clock Speed Confusion Clarified**: A beginner question arose about the clock speed of **H100 GPUs**, specifically concerning the calculation of operations per second and theoretical peak performance. Another user pointed out a probable unit mistake, suggesting it should be **1.8 GHz**, not 1.8 MHz.
  

---


**CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1236967183829962802)** (4 messages): 

- **Matrix Transposition Conundrum**: A member questioned the necessity of tiling in matrix transposition when each element is accessed only once. The answer pointed out it's for **coalesced memory writes**, with a [clarifying blog post on matrix transpose in CUDA](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/).
- **Preemptive Lesson on Coalescing**: The member thanked for the clarification on coalescing, suggesting that this topic is covered in the following chapter, which led to their initial confusion.
- **Sequence of Topics May Cause Confusion**: In response, it was noted that questions sometimes precede the coverage of their topics in the book, which can be puzzling for readers.

**Link mentioned**: <a href="https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/">An Efficient Matrix Transpose in CUDA C/C++ | NVIDIA Technical Blog</a>: My last CUDA C++ post covered the mechanics of using shared memory, including static and dynamic allocation. In this post I will show some of the performance gains achievable using shared memory.

  

---


**CUDA MODE ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/1236329263892926557)** (6 messages): 

- **Support Acknowledged**: A member expressed gratitude for the ongoing support and understanding regarding a high-priority job that caused a delay in a promised addition to the channel.
- **Endorsement for PyTorch Profiling**: Excitement was shared about **nsys** and the interest to try out the *"lightweight"* **PyTorch profiling tools**. The member was inspired by a recording and enquired about standout questions that might have been asked in the Discord after the event.
- **Praise for Source Annotation**: The member mentioned an upcoming source annotation tool by Taylor as **"really cool"**, reminiscent of Apple's Metal profiler’s interface for line-by-line shader profiling. They linked to Apple's developer documentation: [Optimize shaders with per-line shader profiling statistics](https://developer.apple.com/documentation/xcode/optimizing-gpu-performance#Optimize-shaders-with-per-line-shader-profiling-statistics).
- **Profiler Capabilities Highlighted**: A recounting of a profiler capable of making edits on a profiled trace with nearly real-time estimates was highlighted as a notable feature. It involves Instruments using architectural knowledge to 'rerun' executions, potentially based on sampling.

**Link mentioned**: <a href="https://developer.apple.com/documentation/xcode/optimizing-gpu-performance#Optimize-shaders-with-per-line-shader-profiling-statistics">Optimizing GPU performance | Apple Developer Documentation</a>: Find and address performance bottlenecks using the Metal debugger.

  

---


**CUDA MODE ▷ #[jax](https://discord.com/channels/1189498204333543425/1203956655570817034/1236409651550093463)** (1 messages): 

- **Exploring JAX Multi-process Model**: A member shared their appreciation for the **distributed setup capabilities** of JAX, particularly in the context of environments like GPU clusters and [Cloud TPU pods](https://cloud.google.com/tpu). They referenced the [JAX multi-process documentation](https://jax.readthedocs.io/en/latest/multi_process.html), which offers detailed guidance on launching JAX processes and running multi-process computations.

**Link mentioned**: <a href="https://jax.readthedocs.io/en/latest/multi_process.html">Using JAX in multi-host and multi-process environments &#8212; JAX  documentation</a>: no description found

  

---


**CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1235936759435628575)** (12 messages🔥): 

- **Anime Favorites Shared**: Members remarked on their anime preferences; one grew up watching **Naruto**, enjoys **One Punch Man** and **Berserk**, and acknowledges **Jujutsu Kaisen (JJK)** for having top-notch animations and fight scenes. Another member humorously expressed admiration for the character Sukuna from JJK after a particular scene's Blu-ray release.

- **iPhone & Mac as Improvised A/V Setup**: A member suggested using an [iPhone & Mac](https://a.co/d/7uxdnek) for better audio and video quality on calls, noting the automatic integration when both devices are updated and logged in with the same Apple ID. Selecting the iPhone as a camera/mic input works across various platforms like Photo Booth, Discord, Google Meet, and Streamlabs.

- **Interest in Discord to Google Calendar Automation**: A member inquired about setting up automation to sync Discord events with Google Calendar to avoid missing out on the reading group. While no existing solution was mentioned, there was openness to setting it up if it became a significant need.
  

---


**CUDA MODE ▷ #[hqq](https://discord.com/channels/1189498204333543425/1225499037516693574/1236014784009474089)** (4 messages): 

- **GreenBitAI Introduces LLM Toolkit**: A member highlighted [GreenBitAI's green-bit-llm](https://github.com/GreenBitAI/green-bit-llm), a toolkit for fine-tuning, inferencing, and evaluating GreenBitAI's language models, offering a broader scope than the previously discussed bitblas, which focuses specifically on matrix multiplication operations.
- **Fast Inference with BitBlas**: According to one member, BitBlas boasts a fast gemv kernel optimized for 2-bit operations conducive to speeding up inference tasks, but they have yet to personally test it.
- **GreenBitAI's Binary Matrix Multiplication**: Intrigue was expressed regarding [GreenBitAI's cutlass kernels](https://github.com/GreenBitAI/bitorch-engine/blob/main/bitorch_engine/layers/qlinear/binary/cutlass/binary_linear_cutlass.cpp), particularly for their implementation of binary matrix multiplication within their bitorch-engine.
- **Gradients Calculated in Weights**: Another member pointed out an interesting attribute of GreenBitAI's toolkit; it calculates gradients of weights as shown in a [code snippet from bitorch-engine](https://github.com/GreenBitAI/bitorch-engine/blob/main/bitorch_engine/layers/qlinear/nbit/cutlass/q4_layer.py#L81), sparking curiosity about the potential VRAM usage since the gradients aren't packed during training.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/GreenBitAI/green-bit-llm">GitHub - GreenBitAI/green-bit-llm: A toolkit for fine-tuning, inferencing, and evaluating GreenBitAI&#39;s LLMs.</a>: A toolkit for fine-tuning, inferencing, and evaluating GreenBitAI&#39;s LLMs. - GreenBitAI/green-bit-llm</li><li><a href="https://github.com/GreenBitAI/bitorch-engine/blob/main/bitorch_engine/layers/qlinear/binary/cutlass/binary_linear_cutlass.cpp">bitorch-engine/bitorch_engine/layers/qlinear/binary/cutlass/binary_linear_cutlass.cpp at main · GreenBitAI/bitorch-engine</a>: A toolkit enhances PyTorch with specialized functions for low-bit quantized neural networks. - GreenBitAI/bitorch-engine</li><li><a href="https://github.com/GreenBitAI/bitorch-engine/blob/main/bitorch_engine/layers/qlinear/nbit/cutlass/q4_layer.py#L81C9-L81C20">bitorch-engine/bitorch_engine/layers/qlinear/nbit/cutlass/q4_layer.py at main · GreenBitAI/bitorch-engine</a>: A toolkit enhances PyTorch with specialized functions for low-bit quantized neural networks. - GreenBitAI/bitorch-engine
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1235855152624173120)** (630 messages🔥🔥🔥): 

- **CUDA Compiling Quirks**: Compilers like `nvcc 11.5` throw errors for operations in bfloat16 on older GPUs; functions like `__ldcs` and `__stcd` are undefined, and operations like `__bfloat1622float2` cause issues. A [fix is proposed](https://github.com/karpathy/llm.c/pull/353) to handle bfloat16 arithmetic manually to support older cards and toolkits.
- **Multi-GPU Training Hangs**: Recent commits to the master branch caused multi-GPU training to hang, as reported in [Issue #369](https://github.com/karpathy/llm.c/issues/369). A [separate working branch](https://github.com/PeterZhizhin/llm.c/branch/nccl) maintains functional multi-GPU training, and merging this while diagnosing the issue on the master branch is under consideration.
- **Performance and Refactoring Updates**: A PR has been merged that brings a small performance gain by introducing a new [optimized matmul_bias kernel](https://github.com/karpathy/llm.c/pull/343), and subsequent contributions aim to further enhance performance through kernel fusions and [CUDA stream adjustments](https://github.com/ademeure/llm.c/pull/2).
- **Correctness in Overlapping NCCL and Compute**: An attempt to overlap NCCL and backwards compute in multi-GPU training shows improved iteration times, from 225ms down to 193ms ([PR #361](https://github.com/karpathy/llm.c/pull/361)). Correctness verification and testing remain essential while optimizing multi-GPU logic.
- **Nsight Systems Profiling**: Efforts to improve profiling include using Nvidia's Nsight Systems for better visualization and understanding the intricacies of application performance on GPUs. This includes creating a tutorial to help others set up and use Nsight Systems to analyze and optimize CUDA programs.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/datasets/nampdn-ai/mini-fineweb">nampdn-ai/mini-fineweb · Datasets at Hugging Face</a>: no description found</li><li><a href="https://siboehm.com/articles/22/CUDA-MMM">How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog</a>: In this post, I’ll iteratively optimize an implementation of matrix multiplication written in CUDA.My goal is not to build a cuBLAS replacement, but to deepl...</li><li><a href="https://github.com/karpathy/llm.c/discussions/344">State of the Union [May 3, 2024] · karpathy/llm.c · Discussion #344</a>: [May 3, 2024] It is day 24 of the llm.c project. We can now do multi-GPU training, in bfloat16, with flash attention, and it is FAST! 🚀 Single GPU training. We are now training GPT-2 (124M) faster .....</li><li><a href="https://github.com/karpathy/llm.c/issues/369">MultiGPU training hangs · Issue #369 · karpathy/llm.c</a>: mpirun with multiple GPUs is hanging after allocated 474 MiB for master copy of params Most probably due to the introduction of cudastreams. @karpathy @PeterZhizhin</li><li><a href="https://github.com/karpathy/llm.c/pull/361">Overlap gradient computation and NCCL AllReduce by PeterZhizhin · Pull Request #361 · karpathy/llm.c</a>: On my setup, I get the following: Before: step    2/37: train loss 4.720275 (acc 4.688650) (224.046844 ms, 36563.773438 tok/s) step    3/37: train loss 3.802741 (acc 3.943135) (224.151611 ms, 36555...</li><li><a href="https://github.com/karpathy/llm.c/pull/363">Modified version of ademeure&#39;s fused gelu_forward kernel by ChrisDryden · Pull Request #363 · karpathy/llm.c</a>: Was experimenting with the fused gelu kernel to combine it to have the previous code when working with non-gelu matmuls that was built previously and when running it locally it appeared to have a p...</li><li><a href="https://github.com/karpathy/llm.c/pull/347/files">Experimenting with global instantiation for the layouts by ChrisDryden · Pull Request #347 · karpathy/llm.c</a>: I was able to see a speed increase, needs to be cleaned up and refactored substantially but good to see what a potential speedup would be</li><li><a href="https://github.com/NVIDIA/cudnn-frontend">GitHub - NVIDIA/cudnn-frontend: cudnn_frontend provides a c++ wrapper for the cudnn backend API and samples on how to use it</a>: cudnn_frontend provides a c++ wrapper for the cudnn backend API and samples on how to use it - NVIDIA/cudnn-frontend</li><li><a href="https://github.com/karpathy/llm.c/pull/353">Fused layernorm residual by ngc92 · Pull Request #353 · karpathy/llm.c</a>: currently based on top of #352 I&#39;m not using  kernel 6, because a) the performance seems to be really sensitive to parameters b) i don&#39;t understand the performance measurements/are not 100% su...</li><li><a href="https://github.com/karpathy/llm.c/pull/342">fixed activation gradient resetting for backward pass by ngc92 · Pull Request #342 · karpathy/llm.c</a>: also, we don&#39;t need to touch the other buffers in  zero_grad, these are anyway overwritten multiple times during backward</li><li><a href="https://github.com/karpathy/llm.c/pull/343/commits/a0b80920f19567c1895679c4f5b553848ebd669d">Performance: matmul_bias, cuda streams, fused_classifier (+remove cooperative groups) by ademeure · Pull Request #343 · karpathy/llm.c</a>: I might need to split this into multiple PRs, let me know what you think (and I still need to add the new kernels to /dev/cuda/). Major changes:  New super optimised matmul_backward_bias_kernel6 CU...</li><li><a href="https://github.com/karpathy/llm.c/pull/319">convert all float to floatX for layernorm_forward by JaneIllario · Pull Request #319 · karpathy/llm.c</a>: change all kernels to use floatX</li><li><a href="https://github.com/karpathy/llm.c/pull/352">utilities for mixed-precision tests/benchmarks by ngc92 · Pull Request #352 · karpathy/llm.c</a>: This allows us to compile a single executable that can serve as test/benchmark for f32, f16, and bf16 versions of the kernels. So far, I&#39;ve updated only those test files which already defined a BF...</li><li><a href="https://github.com/PeterZhizhin/llm.c/blob/master/train_gpt2.cu#L2036">llm.c/train_gpt2.cu at master · PeterZhizhin/llm.c</a>: LLM training in simple, raw C/CUDA. Contribute to PeterZhizhin/llm.c development by creating an account on GitHub.</li><li><a href="https://docs.github.com/en/actions/hosting-your-own-runners/managing-self-hosted-runners/adding-self-hosted-runners">Adding self-hosted runners - GitHub Docs</a>: no description found</li><li><a href="https://github.com/NVIDIA/cudnn-frontend?tab=readme-ov-file#debugging">GitHub - NVIDIA/cudnn-frontend: cudnn_frontend provides a c++ wrapper for the cudnn backend API and samples on how to use it</a>: cudnn_frontend provides a c++ wrapper for the cudnn backend API and samples on how to use it - NVIDIA/cudnn-frontend</li><li><a href="https://github.com/karpathy/llm.c/pull/346">first attempt at moving cudnn out of the main file for faster compiles by ngc92 · Pull Request #346 · karpathy/llm.c</a>: I think this breaks nonCudnn build, and possibly also windows. I have little knowledge about makefiles, though, so if someone knows how to do things nicely, that would be great :)</li><li><a href="https://openhub.net/p/tensorflow">The TensorFlow Open Source Project on Open Hub</a>: no description found</li><li><a href="https://developer.nvidia.com/nsight-systems">NVIDIA Nsight Systems</a>: Profile systems, analyze performance, and optimize platforms.</li><li><a href="https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2024_3/NsightSystems-macos-public-2024.3.1.75-3419530.dmg">no title found</a>: no description found</li><li><a href="https://github.com/karpathy/llm.c/pull/343">Performance: matmul_bias, cuda streams, fused_classifier (+remove cooperative groups) by ademeure · Pull Request #343 · karpathy/llm.c</a>: I might need to split this into multiple PRs, let me know what you think (and I still need to add the new kernels to /dev/cuda/). Major changes:  New super optimised matmul_backward_bias_kernel6 CU...</li><li><a href="https://github.com/ademeure/llm.c/pull/2">Refactoring &amp; Improvements to reduce LOC by ademeure · Pull Request #2 · ademeure/llm.c</a>: Refactoring and removing unused functions to reduce the number of lines of code and make everything slightly more consistent (while still having space for the code to breathe). Also update encoder_...</li><li><a href="https://ppc-exercises.cs.aalto.fi/course/open2024a/cp/cp4">CP4: GPU baseline</a>: no description found</li><li><a href="https://ppc-exercises.cs.aalto.fi/course/open2024a/cp/cp5">CP5: fast GPU solution</a>: no description found
</li>
</ul>

</div>
  

---



**Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1235867967179329607)** (102 messages🔥🔥): 

- **Mojo Installation Query**: A user inquired about instructions for installing Mojo on a desktop, indicating a need for support.
- **Community Progression**: ModularBot celebrated a community member leveling up, demonstrating an achievement-based engagement system.
- **New Contributions to Mojo**: Discussion indicates an open source development environment, with users directed to GitHub repositories and issues for contributing, especially to the [Mojo standard library](https://github.com/modularml/mojo/blob/main/stdlib/docs/development.md) as suggested by 'soracc'.
- **Addressing Contribution Confusion**: Discussions between members like 'gabrieldemarmiesse' and 'soracc' centered around clarifying contribution processes, referencing [GitHub](https://github.com/modularml/mojo/pull/2457), and considering methods to avoid duplication of work by contributors like the "licking the cookie" phenomenon.
- **Mojo Versioning Scheme Explained**: Users clarified that Mojo utilizes a `YY.major.minor` versioning scheme, not Semantic Versioning (SemVer), with the year reflecting the first number (e.g., version 24.3.x represents the third main release of that year).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests/creating-a-pull-request-template-for-your-repository">Creating a pull request template for your repository - GitHub Docs</a>: no description found</li><li><a href="https://www.modular.com/blog/how-to-contribute-to-mojo-standard-library-a-step-by-step-guide">Modular: How to Contribute to Mojo Standard Library: A Step-by-Step Guide</a>: We are building a next-generation AI developer platform for the world. Check out our latest post: How to Contribute to Mojo Standard Library: A Step-by-Step Guide</li><li><a href="https://devblogs.microsoft.com/oldnewthing/20091201-00/?p=15843)">Microspeak: Cookie licking - The Old New Thing</a>: Now nobody else can have it.</li><li><a href="https://github.com/modularml/mojo/blob/main/stdlib/docs/development.md">mojo/stdlib/docs/development.md at main · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/modularml/mojo/blob/main/CONTRIBUTING.md#create-a-pull-request">mojo/CONTRIBUTING.md at main · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/modularml/mojo/issues/2487">[Feature Request] Make the `msg` argument of `assert_true/false/...` keyword only · Issue #2487 · modularml/mojo</a>: Review Mojo&#39;s priorities I have read the roadmap and priorities and I believe this request falls within the priorities. What is your request? As title. What is your motivation for this change? To ...</li><li><a href="https://open.spotify.com/track/3XwQ8ks84wlj3YcRyxXrlN?si=XJlRyCe_TzOmqPwVtDbCQQ&utm_source=copy-link">Mojo</a>: -M- · Song · 2012</li><li><a href="https://www.youtube.com/watch?v=SEwTjZvy8vw">2023 LLVM Dev Mtg - Mojo 🔥: A system programming language for heterogenous computing</a>: 2023 LLVM Developers&#39; Meetinghttps://llvm.org/devmtg/2023-10------Mojo 🔥: A system programming language for heterogenous computingSpeaker: Abdul Dakkak, Chr...</li><li><a href="https://github.com/modularml/mojo/issues/2415">[Feature Request] Add `__rfloordiv__()` to SIMD type · Issue #2415 · modularml/mojo</a>: Review Mojo&#39;s priorities I have read the roadmap and priorities and I believe this request falls within the priorities. What is your request? The Int and Object types support rfloordiv. I added th...</li><li><a href="https://github.com/apple/swift/issues/43464">[SR-852] [QoI] Poor diagnostic with missing &quot;self.&quot; in convenience initializer · Issue #43464 · apple/swift</a>: Previous ID SR-852 Radar None Original Reporter @ddunbar Type Bug Status Resolved Resolution Done Additional Detail from JIRA Votes 0 Component/s Compiler Labels Bug, DiagnosticsQoI Assignee @dduan...</li><li><a href="https://github.com/modularml/mojo/pull/2457">[stdlib] Support print to stderr by GeauxEric · Pull Request #2457 · modularml/mojo</a>: Add keyword argument to print function to support stream to stderr. Fix #2453 Signed-off-by: Yun Ding yunding.eric@gmail.com
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/)** (1 messages): 

ModularBot: From *Modular*:
<https://twitter.com/Modular/status/1786483510141657384>
  

---


**Modular (Mojo 🔥) ▷ #[📺︱youtube](https://discord.com/channels/1087530497313357884/1098713700719919234/1237145345541017682)** (1 messages): 

- **Modular Community Livestream Announcement**: Modular announced a livestream event with an invitation to explore the latest update in their technology, titled "[Modular Community Livestream - New in MAX 24.3](https://www.youtube.com/watch?v=kKOCuLy-0UY)". The video is set to discuss the new features in MAX Engine and Mojo🔥, as well as introduce the MAX Engine Extensibility API.

**Link mentioned**: <a href="https://www.youtube.com/watch?v=kKOCuLy-0UY">Modular Community Livestream - New in MAX 24.3</a>: MAX 24.3 is now available! Join us on our upcoming livestream as we discuss what’s new in MAX Engine and Mojo🔥 - preview of MAX Engine Extensibility API for...

  

---


**Modular (Mojo 🔥) ▷ #[ai](https://discord.com/channels/1087530497313357884/1103420074372644916/1235975948986220596)** (3 messages): 

- **Interest in Donald Hoffman's Consciousness Research**: A member plans to transfer to UCI to be around the work of Professor Donald Hoffman, who is engaged in mapping conscious experiences. They see a correlation between the limited sensory data in split-brain patients and AI hallucinations, supporting the efficiency of simulating brain function.
  
- **Shared Academic Aspirations**: Another member expressed a shared interest in the goal mentioned above, indicating an alignment with the work related to consciousness research.

- **Seeking a Max Developer**: A member has announced they are looking for a Max Developer for a project and has requested interested parties to direct message them for further details.
  

---


**Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1235850464592134234)** (172 messages🔥🔥): 

- **InlineArray Quirks for Large Arrays**: There are some ongoing issues with `InlineArray` behaving erratically for large arrays, as highlighted in a GitHub issue [here](https://github.com/modularml/mojo/issues/2425).
- **GPU Support in Question for Mojo**: Users challenged the claim that Mojo is the language that unlocks AI hardware, leading to clarification that GPU support is intended for rollout in the coming months, specifically mentioning support for Nvidia.
- **Mojo's Potential Unlocked by MLIR**: A key discussion point was the fact that Mojo's potential isn't limited to GPU support but extends to other hardware acceleration through MLIR, which could future-proof the language against emerging technologies.
- **Questions on Latex Script Parallelization in Mojo**: A user encountered difficulties using parallelization in Mojo for a LaTeX script, which prompted advice on constraints about functions that could be parallelized and error handling.
- **Challenges with Mojo decorators and Custom `None` Value**: One user sought help about decorators, which are not fully supported yet, while another struggled with representing `None` for uninitialized struct members, learning to use `Optional[Node]` for proper typing.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.modular.com/blog/implementing-numpy-style-matrix-slicing-in-mojo">Modular: Implementing NumPy style matrix slicing in Mojo🔥</a>: We are building a next-generation AI developer platform for the world. Check out our latest post: Implementing NumPy style matrix slicing in Mojo🔥</li><li><a href="https://github.com/modularml/mojo/tree/main/examples">mojo/examples at main · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://docs.modular.com/mojo/roadmap#full-mlir-decorator-reflection,">Mojo🔥 roadmap &amp; sharp edges | Modular Docs</a>: A summary of our Mojo plans, including upcoming features and things we need to fix.</li><li><a href="https://github.com/Nautilus-Institute/quals-2024/blob/main/%F0%9F%8C%8C/src/async_runtime.mojo">quals-2024/🌌/src/async_runtime.mojo at main · Nautilus-Institute/quals-2024</a>: Contribute to Nautilus-Institute/quals-2024 development by creating an account on GitHub.</li><li><a href="https://github.com/modularml/devrel-extras/tree/main/blogs/mojo-matrix-slice">devrel-extras/blogs/mojo-matrix-slice at main · modularml/devrel-extras</a>: Contains supporting materials for developer relations blog posts, videos, and workshops - modularml/devrel-extras</li><li><a href="https://docs.modular.com/mojo/notebooks/Matmul#vectorizing-the-inner-most-loop">Matrix multiplication in Mojo | Modular Docs</a>: Learn how to leverage Mojo&#x27;s various functions to write a high-performance matmul.</li><li><a href="https://github.com/modularml/mojo/issues/2425.">Issues · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/modularml/mojo/issues/2467#issuecomment-2092884166">[Feature Request] Unify SSO between `InlinedString` and `String` type · Issue #2467 · modularml/mojo</a>: Review Mojo&#39;s priorities I have read the roadmap and priorities and I believe this request falls within the priorities. What is your request? We currently have https://docs.modular.com/mojo/stdlib...</li><li><a href="https://lists.llvm.org/pipermail/llvm-commits/Week-of-Mon-20140901/233938.html"> [llvm] r217292 - [docs] Document what &quot;NFC&quot; means in a commit	message.
   </a>: no description found</li><li><a href="https://github.com/modularml/mojo/issues">Issues · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/modularml/mojo/pull/2507">[stdlib] Add small string optimization to the `String` struct by gabrieldemarmiesse · Pull Request #2507 · modularml/mojo</a>: Fix part of #2467 This PR will stay in draft as it&#39;s too big to be merged at once, I&#39;ll split it further into multiple PRs. I also have some cleanup to do and some benchmarking. But it can giv...</li><li><a href="https://github.com/modularml/mojo/pull/2539">add note that custom decorators are not supported yet by KarateCowboy · Pull Request #2539 · modularml/mojo</a>: no description found
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[community-projects](https://discord.com/channels/1087530497313357884/1151418679578337311/1236019023083212881)** (22 messages🔥): 

- **NuMojo Update Rockets Ahead**: [NuMojo](https://github.com/MadAlex1997/NuMojo), previously Mojo-Arrays, is back in active development, updated to Mojo version 24.3. The library, focused on building functions around the standard library tensor, is now significantly faster, offering a performance boost of 6x to 20x compared to numpy.

- **Mimage Library for Mojo Image Parsing**: A new library called [Mimage](https://github.com/fnands/mimage) for image parsing in Mojo has been introduced, with support for simple 8-bit RGB PNGs. There's an ongoing community discussion on whether to adopt a PIL-style Image class or an ND array representation for images.

- **Basalt Development Milestones**: The [Basalt project](https://github.com/basalt-org/basalt) has celebrated reaching 200 stars, released new documentation at [Basalt Docs](https://basalt-docs.vercel.app/), and announced updates for Mojo 24.3. These updates include experimental ONNX model import/export, dynamic operation support, and a variety of enhancements and bug fixes.

- **Prototype for Struct Composability in Mojo**: The lsx library for HTML generation in Mojo has seen a new prototype for struct composability shared at [GitHub lsx](https://github.com/rd4com/lsx/tree/main/struct%20composability%20prototype), aiming for full compatibility with lsx and better handling of UnsafePointers.

- **MinBPE Port and Performance Insights**: A Mojo port of Andrej Karpathy’s minbpe project [minbpe.mojo](https://github.com/dorjeduck/minbpe.mojo) has been posted, highlighting the challenges of porting from Python and the absence of inheritance in Mojo. The Mojo version is about three times faster than the Python original, with noticeable performance gains after switching to a more efficient dictionary implementation.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/rd4com/lsx/tree/main/struct%20composability%20prototype">lsx/struct composability prototype at main · rd4com/lsx</a>: An experimental library for HTML generation in Mojo - rd4com/lsx</li><li><a href="https://github.com/dorjeduck/minbpe.mojo">GitHub - dorjeduck/minbpe.mojo: port of Andrjey Karpathy&#39;s minbpe to Mojo</a>: port of Andrjey Karpathy&#39;s minbpe to Mojo. Contribute to dorjeduck/minbpe.mojo development by creating an account on GitHub.</li><li><a href="https://github.com/mzaks/mojo-sort">GitHub - mzaks/mojo-sort</a>: Contribute to mzaks/mojo-sort development by creating an account on GitHub.</li><li><a href="https://github.com/saviorand/lightbug_http/issues/34">Client tests don&#39;t work with changes in Mojo 24.3 · Issue #34 · saviorand/lightbug_http</a>: Since Mojo 24.3 main() functions inside packages are no longer supported. This was used in /tests/run.mojo to run a test suite (which is just one client test for now). The client test worked by run...</li><li><a href="https://github.com/gorodion/pycv">GitHub - gorodion/pycv</a>: Contribute to gorodion/pycv development by creating an account on GitHub.</li><li><a href="https://github.com/gorodion/pycv/blob/main/demo.ipynb">pycv/demo.ipynb at main · gorodion/pycv</a>: Contribute to gorodion/pycv development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[community-blogs-vids](https://discord.com/channels/1087530497313357884/1151418796993683477/1236021318067949648)** (6 messages): 

- **Building with Mojo and Parameters**: A new tutorial was shared on using parameters to build a Mojo app, enhancing workflows and integrating custom constraints. The tutorial is available at [GitHub - Tutorial on parameters in Mojo](https://github.com/rd4com/mojo-learning/blob/main/tutorials/use-parameters-to-create-or-integrate-workflow.md).

- **Syntax Highlighting Tip**: In response to the tutorial on Mojo parameters, a suggestion was made to improve the readability of the code using proper syntax highlighting, by using triple backticks with the term "mojo" in markdown files.

- **Parsing PNGs in Mojo Explored**: A blog post about parsing PNGs using Mojo was shared, along with the launch of a library named *mimage* for reading images in Mojo. Both the [blog post](https://fnands.com/mojo-png-parsing/) and the [mimage library](https://github.com/fnands/mimage) are accessible online.

- **Community Positive Feedback**: The blog post on PNG parsing received positive feedback from the community, with peers expressing admiration for the effort.

- **RSS Feed Needs a Fix**: The same blog post author acknowledged the need to fix the RSS feed issue on their site after a community member expressed interest in subscribing to future articles.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/rd4com/mojo-learning/blob/main/tutorials/use-parameters-to-create-or-integrate-workflow.md">mojo-learning/tutorials/use-parameters-to-create-or-integrate-workflow.md at main · rd4com/mojo-learning</a>: 📖 Learn some mojo ! Contribute to rd4com/mojo-learning development by creating an account on GitHub.</li><li><a href="https://fnands.com/mojo-png-parsing/">Parsing PNG images in Mojo</a>: There's currently no direct way of reading image files from Mojo. In this post I go through what's needed to parse a PNG file directly in Mojo without having to go through Python. Additionally, I refa...</li><li><a href="https://github.com/fnands/mimage">GitHub - fnands/mimage: A library for parsing images in Mojo</a>: A library for parsing images in Mojo. Contribute to fnands/mimage development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[📰︱newsletter](https://discord.com/channels/1087530497313357884/1157711273111339088/)** (1 messages): 

Zapier: Modverse Weekly - Issue 32
https://www.modular.com/newsletters/modverse-weekly-32
  

---


**Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1235851909139660883)** (92 messages🔥🔥): 

- **80 column debate heats up**: Discord participants discussed the need to move beyond the [80-column convention](https://stackoverflow.com/questions/4651012/why-is-the-default-terminal-width-80-characters), historical for punched cards and monitors. Some members expressed a preference for 100 columns, stating it would still allow multiple file views side-by-side.

- **Nightly Mojo compiler update**: A new [nightly release of the Mojo compiler](https://github.com/modularml/mojo/pull/2498/files) was announced, with details on the recent changes available in the provided links. Users are encouraged to update with `modular update nightly/mojo`.

- **Register passable types on the chopping block**: Discussion emerged around the evolution of the "register passable" concept in Mojo, with an aim to phase out types like `OptionalReg` in favor of all-encompassing types like `Optional` and leaning towards traits to indicate register passability.

- **The math module's status addressed**: Confirmation that the math module has not disappeared; it's yet to be open-sourced, resulting in references to it being removed from the open-sourced part of the stdlib.

- **Pre-commit hook issue filed**: An [issue with the "check-license" pre-commit hook](https://github.com/modularml/mojo/issues/2528#issuecomment-2094837006) was reported, where it couldn't find the stdlib, leading to a discussion and an eventual open issue for the intermittent problem.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://stackoverflow.com/questions/4651012/why-is-the-default-terminal-width-80-characters">Why is the default terminal width 80 characters?</a>: 80 seems to be the default in many different environments and I&#x27;m looking for a technical or historical reason. It is common knowledge that lines of code shouldn&#x27;t exceed 80 characters, but ...</li><li><a href="https://github.com/modularml/mojo/issues/2413">[Feature Request] Allow substitution of child traits for parent traits · Issue #2413 · modularml/mojo</a>: Review Mojo&#39;s priorities I have read the roadmap and priorities and I believe this request falls within the priorities. What is your request? If a function takes variadic arguments bound by a trai...</li><li><a href="https://github.com/modularml/mojo/issues/2492)">Issues · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/modularml/mojo/issues/2528#i">[BUG] check-license failing sometimes · Issue #2528 · modularml/mojo</a>: Bug description The pre-commit hook for license checking is failing sometimes. I had not been able to understand when it happens, from the log it seems like my stdlib.mojopkg is not found but runni...</li><li><a href="https://github.com/modularml/mojo/issues/2528#issuecomment-2094837006">[BUG] check-license failing sometimes · Issue #2528 · modularml/mojo</a>: Bug description The pre-commit hook for license checking is failing sometimes. I had not been able to understand when it happens, from the log it seems like my stdlib.mojopkg is not found but runni...</li><li><a href="https://github.com/modularml/mojo/issues/2534">[BUG] `__call_location().file_name` returns incorrect information · Issue #2534 · modularml/mojo</a>: Bug description It seems that the __call_location() function is returning incorrect data. It&#39;s been suggested that, &quot;It looks to be baking in our internal source code path into the Mojo binar...</li><li><a href="https://github.com/modularml/mojo/issues/2529">[BUG] Functions, traits, structs and aliases are leaking into builtins and are importable from anywhere · Issue #2529 · modularml/mojo</a>: Bug description As the title says, importing anything without a leading underscore in &quot;./stdlib/builtin/anything.mojo&quot; insert it into the list of stuff that does not need to be imported glob...</li><li><a href="https://github.com/modularml/mojo/issues/2425">[BUG] Excessive compilation time during manipulation of StaticTuple · Issue #2425 · modularml/mojo</a>: Bug description The following code takes around 40 seconds to compile with the actual execution time after build being trivial. The compilation time is also tied to the Tuple size not the number of...</li><li><a href="https://github.com/modularml/mojo/pull/2498/files">[stdlib] Update stdlib corresponding to 2024-05-03 nightly/mojo by JoeLoser · Pull Request #2498 · modularml/mojo</a>: This updates the stdlib with the internal commits corresponding to today&#39;s nightly release: mojo 2024.5.323.</li><li><a href="https://github.com/modularml/mojo/blob/nightly/docs/changelog.md">mojo/docs/changelog.md at nightly · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



**HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1235906896414769202)** (2 messages): 

```html
<ul>
    <li><strong>Community Highlights Get an Update</strong>: Community highlight #56 introduces <a href="https://huggingface.co/spaces/Csplk/moondream2-batch-processing">Moondream 2 batch processing</a>, <a href="https://huggingface.co/spaces/fluently/Fluently-Playground">FluentlyXL v4</a>, Portuguese translation of HF Audio course's first chapters, <a href="https://huggingface.co/spaces/unography/image-captioning-with-longcap">BLIP fine-tune</a> for long captions, and many other projects. A comprehensive Portuguese list and retrospective of highlights is also available <a href="https://iatalk.ing/destaques-comunidade-hugging-face/">here</a>.</li>
    <li><strong>New Advances in AI Shared</strong>: Latest spaces feature <a href="https://huggingface.co/spaces/as-cle-bert/bloom-multilingual-chat">BLOOM multilingual chat</a>, an <a href="https://huggingface.co/spaces/tonyassi/inpainting-sdxl-sketch-pad">inpainting sketch pad</a>, and a link prediction <a href="https://github.com/Lama-West/PnPR-GCN_ACM_SAC_24/tree/main">repository</a>. Additionally, the HuggingFace alignment handbook task can now be run in the cloud with dstack as tweeted <a href="https://twitter.com/dstackai/status/1785315721578459402">here</a>.</li>
    <li><strong>Cool Stuff Unveiled by Community</strong>: A wide range of topics is covered from <a href="https://huggingface.co/blog/AmelieSchreiber/protein-optimization-and-design">protein optimization with Generative AI</a> to <a href="https://huggingface.co/blog/AviSoori1x/seemore-vision-language-model">implementing a Vision Language Model from scratch</a>. Also discussed is the Google Search with LLMs, Token Merging for fast LLM inference, and <a href="https://huggingface.co/blog/maywell/llm-feature-transfer">creating chat models with a single click</a>.</li>
    <li><strong>Cutting-edge Conversations</strong>: A reading group is scheduled to discuss recent progress and share insights, furthering the exchange of knowledge in the AI space. To join the next session, please check out this <a href="https://discord.com/events/879548962464493619/1234913780048203856">link</a>.</li>
    <li><strong>AutoTrain Configs Introduced</strong>: AutoTrain now supports yaml config files simplifying the model training process, even for those new to machine learning. An announcement about this new feature has been <a href="https://twitter.com/abhi1thakur/status/1786368641388179797">tweeted</a>, and the Github repository with example configs can be accessed <a href="https://github.com/huggingface/autotrain-advanced">here</a>.</li>
</ul>
```
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/huggingface/autotrain-advanced">GitHub - huggingface/autotrain-advanced: 🤗 AutoTrain Advanced</a>: 🤗 AutoTrain Advanced. Contribute to huggingface/autotrain-advanced development by creating an account on GitHub.</li><li><a href="https://iatalk.ing/destaques-comunidade-hugging-face/)">🤗 Destaques da Comunidade</a>: O Destaques da Comunidade é um post contendo uma lista publicada periodicamente no Discord do Huggging Face contendo uma série de projetos, modelos, spaces, posts, artigos feitos pela comunidade de…
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1235859617876021278)** (225 messages🔥🔥): 

- **Exploring Audio Diffusion Modelling**: A conversation unfolded around creating a model that iteratively generates music based on feedback, potentially using audio diffusion models. Issues such as the computational depth required for such a model and its capabilities in generating longer and theoretically sound pieces were discussed.

- **Struggling with large model conversion**: One user faced difficulties converting a PyTorch model into Tensorflow Lite format, encountering a size limit error. The model in question exceeded the 2GB limit during conversion from ONNX to TensorFlow.

- **Deploying Whisper for Filipino ASR**: A discussion on the feasibility of fine-tuning the Whisper ASR model for Filipino language took place. Parameters such as `weight_decay`, learning rate, and dataset size (80k audio chunks) were mentioned as factors influencing performance.

- **Security Concerns Arise After Hacks**: Several messages indicated that the Hugging Face Twitter account was compromised, leading to discussions about cybersecurity measures and their implications for AI systems. The community was active in flagging suspicious activity and investigating the situation.

- **GPU Utilization Mysteries**: Users shared experiences and advice regarding disparate GPU training times between local machines and Google Colab, examining the efficiency differences between consumer gaming cards and edge inference cards, and providing optimization recommendations.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://edopedrocchi.github.io/RicercaMente/Projects/IPO/indexIPO.html">TABLE OF CONTENT</a>: Open source project that aims to trace the history of data science through scientific research published over the years</li><li><a href="https://www.llama2.ai/">Chat with Meta Llama 3 on Replicate</a>: Llama 3 is the latest language model from Meta.</li><li><a href="https://huggingface.co/meta-llama/Meta-Llama-3-8B">meta-llama/Meta-Llama-3-8B · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/crusoeai/Llama-3-8B-Instruct-Gradient-1048k">crusoeai/Llama-3-8B-Instruct-Gradient-1048k-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/learn/computer-vision-course">Welcome to the Community Computer Vision Course - Hugging Face Community Computer Vision Course</a>: no description found</li><li><a href="https://github.com/komorra">komorra - Overview</a>: Programmer / twitter: https://twitter.com/komorra86 - komorra</li><li><a href="https://github.com/amjadmajid/BabyTorch">GitHub - amjadmajid/BabyTorch: BabyTorch is a minimalist deep-learning framework with a similar API to PyTorch. This minimalist design encourages learners explore and understand the underlying algorithms and mechanics of deep learning processes. It is design such that when learners are ready to switch to PyTorch they only need to remove the word `baby`.</a>: BabyTorch is a minimalist deep-learning framework with a similar API to PyTorch. This minimalist design encourages learners explore and understand the underlying algorithms and mechanics of deep le...
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1235893693454614528)** (12 messages🔥): 

- **Trouble in Model Export Land**: A member is experiencing difficulties exporting a fine-tuned model and has encountered errors that are causing frustration.
- **To Loop or Not to Loop**: There's a debate on whether it is always advisable to write your own training loop, with a member suggesting that using examples from **Diffusers** and then modifying them allows for more customization.
- **Intrigued by Kolmogorov-Arnold Networks**: **Kolmogorov-Arnold Networks (KANs)** are highlighted for their potential to use less computational graphs than MLPs. The concept is backed by research with a shared [academic link](https://arxiv.org/abs/2404.19756v1), which compares KANs to MLPs in terms of accuracy and interpretability.
- **Diving into Fine-Tuning**: A member shared educational resources about what fine-tuning a generative AI model means, including a [two-minute YouTube video](https://www.youtube.com/watch?v=yoLwkowb2TU&t=1s) and a [HuggingFace tutorial](https://huggingface.co/docs/transformers/training).
- **Overcoming API Deployment Challenges**: A learner sought assistance with issues faced during the building stage of the API in Hugging Face Space, pointing to a lesson in the deeplearning.ai Hugging Face course and citing a problem with versions in `requirements.txt`.
- **Methodology for Step-by-Step Reasoning**: A member experimented with implementing a 'think step by step' approach for LLM outputs but found that local models did not grasp this well. An alternative setup involving a chain of `planner`, `writer`, `analyst`, and `editor` achieved more comprehensive results when tested with Llama 3 instruct 7B.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2404.19756v1">KAN: Kolmogorov-Arnold Networks</a>: Inspired by the Kolmogorov-Arnold representation theorem, we propose Kolmogorov-Arnold Networks (KANs) as promising alternatives to Multi-Layer Perceptrons (MLPs). While MLPs have fixed activation fun...</li><li><a href="https://learn.deeplearning.ai/courses/open-source-models-hugging-face/lesson/15/deployment">DLAI - Open Source Models with Hugging Face</a>: Introduction · Selecting models · Natural Language Processing (NLP) · Translation and Summarization · Sentence Embeddings · Zero-Shot Audio Classification · Automatic Speech Recognition · Text to Spee...</li><li><a href="https://learn.deeplearning.ai/courses/open-source-models-hugging-face/lesson/15/deployment=====">DLAI - Open Source Models with Hugging Face</a>: Introduction · Selecting models · Natural Language Processing (NLP) · Translation and Summarization · Sentence Embeddings · Zero-Shot Audio Classification · Automatic Speech Recognition · Text to Spee...</li><li><a href="https://huggingface.co/docs/transformers/training">Fine-tune a pretrained model</a>: no description found</li><li><a href="https://docs.google.com/presentation/d/1IkzESdOwdmwvPxIELYJi8--K3EZ98_cL6c5ZcLKSyVg/edit#slide=id.p">Little guide to building Large Language Models in 2024</a>: A little guide to building Large Language Models in 2024 thomas@huggingface.co
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1235890773149614120)** (11 messages🔥): 

- **Revolutionizing Retrieval with RAG**: A [Databricks glossary entry](https://www.databricks.com/it/glossary/retrieval-augmented-generation-rag) discusses *Retrieval-Augmented Generation (RAG)*, highlighting its solution to issues of Large Language Models (LLMs) not being able to access data beyond their original training sets, making them static and sometimes inaccurate.
- **Dataset Giants Clashing on GitHub**: Microsoft released the [MS-MARCO-Web-Search dataset](https://github.com/microsoft/MS-MARCO-Web-Search), a large-scale web dataset with millions of real clicked query-document labels for improving information retrieval systems.
- **Let Webhooks Ring**: Hugging Face has published a guide on creating a server listening to webhooks, deploying to Gradio-based Spaces, and [integrating with the Huggingface Hub](https://huggingface.co/docs/huggingface_hub/guides/webhooks_server#create-an-endpoint).
- **Stepping Into Quantum Services**: A link to an [oqtant™ quantum virtual server platform](https://oqtant.infleqtion.com/) was shared, suggesting advancements in the accessibility of quantum computing resources.
- **Gauge Your RAG with Ragas**: The [Ragas framework](https://docs.ragas.io/en/stable/) is presented as a tool for assessing the performance of Retrieval-Augmented Generation (RAG) pipelines in LLM applications, emphasizing metrics-driven development and synthetic testset generation for robust evaluations.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.ragas.io/en/stable/">Introduction | Ragas</a>: no description found</li><li><a href="https://oqtant.infleqtion.com/">Oqtant</a>: no description found</li><li><a href="https://lilianweng.github.io/">Lil&#39;Log</a>: Document my learning notes.</li><li><a href="https://huggingface.co/docs/huggingface_hub/guides/webhooks_server#create-an-endpoint">Webhooks Server</a>: no description found</li><li><a href="https://github.com/microsoft/MS-MARCO-Web-Search">GitHub - microsoft/MS-MARCO-Web-Search: A large-scale information-rich web dataset, featuring millions of real clicked query-document labels</a>: A large-scale information-rich web dataset, featuring millions of real clicked query-document labels - microsoft/MS-MARCO-Web-Search</li><li><a href="https://www.databricks.com/it/glossary/retrieval-augmented-generation-rag">Che cos&#x27;è la Retrieval Augmented Generation (RAG)? | Databricks</a>: La RAG (Retrieval Augmented Generation) è un approccio architettonico che utilizza i dati come contesto per i modelli linguistici di grandi dimensioni (LLM) in modo da migliorare la pertinenza dell&#x...
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1235893212393111594)** (19 messages🔥): 

- **Shadow-Clown BioMistral Unveiled**: A new model called [shadow-clown-BioMistral-7B-DARE](https://huggingface.co/kimou605/shadow-clown-BioMistral-7B-DARE) has been created, merging **BioMistral-7B-DARE** and **shadow-clown-7B-dare** using **mergekit**, aiming to combine the capabilities of both models.
- **Generative Synthetic Data Tool Launched**: A new tool for generating and normalizing synthetic data is now available on PyPI, which may be beneficial for fine-tuning large language models. Further details can be found on the [GitHub repository](https://github.com/tobiadefami/fuxion).
- **Loading LLMs Efficiently via Ollama**: A [GitHub page](https://github.com/di37/LLM-Load-Unload-Ollama) and a [LinkedIn post](https://www.linkedin.com/feed/update/urn:li:activity:7192369828848877568/) showcase methods for efficient loading and unloading of LLMs when using them via Ollama.
- **AI Assists Your Podcast Creation**: The [Podcastify](https://huggingface.co/spaces/eswardivi/Podcastify) space on HuggingFace can convert articles into podcast-like conversations.
- **OpenGPTs Challenges GPT Store**: The [OpenGPTs-platform](https://github.com/OpenGPTs-platform) is launched, aiming to emulate and extend the abilities of the official GPT Store, starting with a foundational version that includes an "Assistants API" and various tools for content retrieval.

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/Delik/pyannote-speaker-diarization-3.1">Pyannote Speaker Diarization 3.1 - a Hugging Face Space by Delik</a>: no description found</li><li><a href="https://huggingface.co/spaces/eswardivi/Podcastify">Podcastify - a Hugging Face Space by eswardivi</a>: no description found</li><li><a href="https://www.notion.so/Tutorial-Moondream-2-Vision-Model-with-LLaMA-71006babe8d647ce8f7a98e683713018?pvs=4">Notion – The all-in-one workspace for your notes, tasks, wikis, and databases.</a>: A new tool that blends your everyday work apps into one. It's the all-in-one workspace for you and your team</li><li><a href="https://huggingface.co/datasets/BEE-spoke-data/fineweb-100_128k">BEE-spoke-data/fineweb-100_128k · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/kimou605/shadow-clown-BioMistral-7B-DARE">kimou605/shadow-clown-BioMistral-7B-DARE · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/spaces/Nick088/Real-ESRGAN_Pytorch">RealESRGAN Pytorch - a Hugging Face Space by Nick088</a>: no description found</li><li><a href="https://huggingface.co/spaces/fishaudio/fish-speech-1">Fish Speech 1 - a Hugging Face Space by fishaudio</a>: no description found</li><li><a href="https://github.com/di37/LLM-Load-Unload-Ollama">GitHub - di37/LLM-Load-Unload-Ollama: This is a simple demonstration to show how to keep an LLM loaded for prolonged time in the memory or unloading the model immediately after inferencing when using it via Ollama.</a>: This is a simple demonstration to show how to keep an LLM loaded for prolonged time in the memory or unloading the model immediately after inferencing when using it via Ollama. - di37/LLM-Load-Unlo...</li><li><a href="https://github.com/tobiadefami/fuxion">GitHub - Tobiadefami/fuxion: Sythetic data generation and normalization functions</a>: Sythetic data generation and normalization functions - Tobiadefami/fuxion</li><li><a href="https://github.com/Gapi505/Sparky-2">GitHub - Gapi505/Sparky-2</a>: Contribute to Gapi505/Sparky-2 development by creating an account on GitHub.</li><li><a href="https://astrabert.github.io/everything-ai">everything-ai</a>: Introducing everything-ai, your fully proficient, AI-powered and local chatbot assistant! 🤖</li><li><a href="https://github.com/AstraBert/everything-ai">GitHub - AstraBert/everything-ai: Introducing everything-ai, your fully proficient, AI-powered and local chatbot assistant! 🤖</a>: Introducing everything-ai, your fully proficient, AI-powered and local chatbot assistant! 🤖 - AstraBert/everything-ai</li><li><a href="https://youtubevideosum.streamlit.app/">no title found</a>: no description found
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1236354360716427314)** (45 messages🔥): 

- **Graph ML and LLMs Discussion Alert**: The **HuggingFace Discord** group is holding a [meeting](https://discord.com/channels/879548962464493619/1203285086624157696) centered around a recent paper on [Graph Machine Learning](https://arxiv.org/abs/2404.14928). The paper covers the use of large language models (LLMs) in graph machine learning and its wide applications.
- **GNNs: A Landscape of Possibilities**: Members are discussing the diverse uses of **Graph Neural Networks (GNNs)**, ranging from fraud detection to generating recommendations, and even task planning for robots. The versatility of GNNs has piqued the interest of participants, prompting some to [experiment](https://cdn.discordapp.com/emojis/1225927322117341337.webp?size=48&quality=lossless) with these models.
- **Presentation Resources Shared**: The presenter, linked as **Isamu Isozaki**, shares a [medium article](https://isamu-website.medium.com/understanding-graph-machine-learning-in-the-era-of-large-language-models-llms-dce2fd3f3af4) diving deeper into the topic discussed and a [YouTube video](https://www.youtube.com/watch?v=cgMAvqgq0Ew&ab_channel=IsamuIsozaki) for those who missed the live presentation. Furthermore, there is a discussion about uploading content to an alternative platform due to Medium's access restrictions.
- **Incorporating Special Tokens in LLMs**: One member highlights a [paper](https://arxiv.org/abs/2404.19705) proposing a training method that teaches LLMs to use a special token, `<RET>`, to trigger information retrieval when uncertain. The method aims to boost both the accuracy and efficiency of LLMs by only retrieving information when necessary.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/what-is-bro-yammering-about-what-is-bro-wafflin-about-what-is-bro-yapping-abo">no title found</a>: no description found</li><li><a href="https://arxiv.org/abs/2404.14928">Graph Machine Learning in the Era of Large Language Models (LLMs)</a>: Graphs play an important role in representing complex relationships in various domains like social networks, knowledge graphs, and molecular discovery. With the advent of deep learning, Graph Neural N...</li><li><a href="https://tenor.com/view/what-is-bro-yammering-about-what-is-bro-wafflin-about-what-is-bro-yapping-about-yapping-what-is-bro-yappin-about-gif-12728898718751592705">What Is Bro Yammering About What Is Bro Wafflin About GIF - What is bro yammering about What is bro wafflin about What is bro yapping about - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.youtube.com/watch?v=cgMAvqgq0Ew&ab_channel=IsamuIsozaki">Hugging Face Reading Group 20: Graph Machine Learning in the Era of Large Language Models (LLMs)</a>: Presenter: Isamu IsozakiWrite up: https://isamu-website.medium.com/understanding-graph-machine-learning-in-the-era-of-large-language-models-llms-dce2fd3f3af4</li><li><a href="https://arxiv.org/abs/2404.19705">When to Retrieve: Teaching LLMs to Utilize Information Retrieval Effectively</a>: In this paper, we demonstrate how Large Language Models (LLMs) can effectively learn to use an off-the-shelf information retrieval (IR) system specifically when additional context is required to answe...</li><li><a href="https://bytez.com/read/arxiv/2404.19705">Bytez: When to Retrieve: Teaching LLMs to Utilize Information Retrieval Effectively</a>: In this paper, we demonstrate how Large Language Models (LLMs) can effectively learn to use an off-the-shelf information retrieval (IR) system specifically when additional context is required to answe...</li><li><a href="https://x.com/omarsar0/status/1785498325913108556?t=Mfnr02-d3Hn0J4vcH9KPNA&s=09">Tweet from elvis (@omarsar0)</a>: When to Retrieve?  This new paper presents an approach to train LLMs to effectively utilize information retrieval.  It first proposes a training approach to teach an LLM to generate a special token, &...</li><li><a href="https://youtu.be/gu5ttnClB5g?si=pTOTrcgsdMG6Q4mV">Training an LLM to effectively use information retrieval</a>: This new paper presents an approach to train LLMs to effectively utilize information retrieval.It first proposes a training approach to teach an LLM to gener...
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1236019902398201936)** (42 messages🔥): 

- **Gap Detection Challenge in Auto Parts**: A member described issues with using a simple YOLO classification model to detect gaps in certain vehicle parts. They requested suggestions on alternative models or techniques to improve detection performance.

- **Craving for Classic CV**: A relatively new member to computer vision queried the current industry relevance of traditional CV techniques like SURF and SIFT and wondered if in-depth knowledge of these methods is necessary.

- **Fine-tuning Object Detection**: There was a discussion on fine-tuning the classifier part of an object detection model, with a focus on whether it's helpful to use an additional CNN for image scaling instead of pre-scaling images before feeding them to models like Darknet YOLO.

- **CLIP Performance on Rotated Objects**: A user sought advice on using the CLIP model to match images of Magic: The Gathering cards that aren't perfectly aligned. Recommendations included augmenting the training data with rotated and skewed images to improve robustness.

- **On the Hunt for GhostNet Weights**: A member inquired about the availability of pre-trained GhostNet weights on ImageNet for TensorFlow, sharing the [GhostNet paper abstract](https://arxiv.org/abs/1911.11907) and [Efficient-AI-Backbones GitHub repository](https://github.com/huawei-noah/ghostnet) but requested assistance on using the provided weights within TensorFlow.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/1911.11907">GhostNet: More Features from Cheap Operations</a>: Deploying convolutional neural networks (CNNs) on embedded devices is difficult due to the limited memory and computation resources. The redundancy in feature maps is an important characteristic of th...</li><li><a href="https://github.com/huawei-noah/ghostnet">GitHub - huawei-noah/Efficient-AI-Backbones: Efficient AI Backbones including GhostNet, TNT and MLP, developed by Huawei Noah&#39;s Ark Lab.</a>: Efficient AI Backbones including GhostNet, TNT and MLP, developed by Huawei Noah&#39;s Ark Lab. - huawei-noah/Efficient-AI-Backbones
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1235892774528880641)** (12 messages🔥): 

- **Seeking Simplified Instructions**: A user inquired about using a simplified version of a tool or method, but did not specify which one.
- **Custom Fine-Tuning Services Offered**: There's an open request from a user offering financial compensation for guidance on how to fine-tune the **Mistral-7B-instruct** model with a custom dataset.
- **LLM Evaluation Skepticism**: A member expressed doubt about using Large Language Models (LLMs) to evaluate other LLMs, given potential hallucination issues and the rapid development of foundational models. The member also pointed out the challenge businesses face in evaluating LLMs and Retrieval-Augmented Generation (RAG) systems for their specific needs.
- **Paper Introduction to LLM based Translation Metric**: The GEMBA metric, a GPT-based translation quality assessment tool, was introduced via an [ACL Anthology paper link](https://aclanthology.org/2023.eamt-1.19/), which described its effectiveness particularly with GPT 3.5 and larger models.
- **Request for Flash Attention Implementation Tutorial**: A member inquired about adding **flash attention 2** to XLM-R and asked if Hugging Face provided any tutorials or guidelines for such an implementation.

**Link mentioned**: <a href="https://aclanthology.org/2023.eamt-1.19/">Large Language Models Are State-of-the-Art Evaluators of Translation Quality</a>: Tom Kocmi, Christian Federmann. Proceedings of the 24th Annual Conference of the European Association for Machine Translation. 2023.

  

---


**HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1235938948904128554)** (17 messages🔥): 

- **Finetuning StableDiffusionPipelines**: A member explored the concept of partial diffusion using two different pipelines, denoising an image halfway with one, then continuing with another. They were directed to an outstanding **[pull request](https://github.com/huggingface/diffusers/compare/main...bghira:diffusers:partial-diffusion-2)** that facilitates this process for the StableDiffusionXLPipeline.

- **Assistance for a Partial Diffusion PR**: The same member was encouraged to test the partial diffusion feature via the linked pull request and to report any issues directly onto it, as the code would soon be revisited and updated.

- **Training Diffusion Models on Multiple Subjects**: A member inquired about training diffusion models to learn multiple subjects simultaneously. It was suggested that they explore **[Custom Diffusion](https://huggingface.co/docs/diffusers/main/en/training/custom_diffusion#:~:text=Custom%20Diffusion%20is%20unique%20because%20it%20can%20also%20learn%20multiple%20concepts%20at%20the%20same%20time.)**, a training technique that allows learning multiple concepts at once.

- **Accelerate Multi-GPU Running with CPU Offloading Issues**: One member faced technical challenges combining **accelerate's multi-GPU running** with **diffuser's model CPU offloading**, specifically device-related errors. The community did not address this as of the last message.

- **Estimating Billing with LLM Pricing Calculator**: Another member sought confirmation on whether the token counts they had were sufficient for estimating their API billings using a shared **[LLM Model Pricing](https://docsbot.ai/tools/gpt-openai-api-pricing-calculator)** calculator. The query remained unaddressed in the discussion.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/docs/diffusers/main/en/training/custom_diffusion#:~:text=Custom%20Diffusion%20is%20unique%20because%20it%20can%20also%20learn%20multiple%20concepts%20at%20the%20same%20time.)">Custom Diffusion</a>: no description found</li><li><a href="https://docsbot.ai/tools/gpt-openai-api-pricing-calculator">OpenAI &amp; other LLM API Pricing Calculator - DocsBot AI</a>: Calculate and compare the cost of using OpenAI, Azure, Anthropic, Llama 3, Google Gemini, Mistral, and Cohere APIs with our powerful FREE pricing calculator.</li><li><a href="https://huggingface.co/docs/diffusers/main/en/training/custom_diffusion#">Custom Diffusion</a>: no description found</li><li><a href="https://github.com/huggingface/diffusers/compare/main...bghira:diffusers:partial-diffusion-2">Comparing huggingface:main...bghira:partial-diffusion-2 · huggingface/diffusers</a>: 🤗 Diffusers: State-of-the-art diffusion models for image and audio generation in PyTorch and FLAX. - Comparing huggingface:main...bghira:partial-diffusion-2 · huggingface/diffusers
</li>
</ul>

</div>
  

---



**OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1235902892762730557)** (212 messages🔥🔥): 

- **Calling Builders on Skills Library Opportunities**: One member explored working on the OpenInterpreter **skills library**, referring to Killian’s contributions on GitHub and recommending to view the commit history for [skills.py](https://github.com/OpenInterpreter/open-interpreter/commits/59956e01ebedc74e0bfed80352ea0a90ecf154b1/interpreter/core/computer/skills/skills.py).

- **Microsoft Open Source AI Hackathon Announcement**: Members are forming a team to participate in the Microsoft Open Source AI Hackathon in Seattle, with the intent to create a project using **Open Interpreter**. The hackathon promises **hands-on tutorials**, pizza, and afternoon snacks with details [here](https://lu.ma/iu1wijgd).

- **Groq LLM Integration and Issues**: There was a discussion on integrating **Groq LLM** with Open Interpreter and experiencing unexpected behaviors like uncontrollable output and creating multiple files on the desktop. The command provided for connection was `interpreter --api_base "https://api.groq.com/openai/v1" --api_key "YOUR_API_KEY_HERE" --model "llama3-70b-8192" -y --max_tokens 8192`.

- **OpenAI Token Cost and Optimization Concerns**: One member expressed concern over the cost of using **OpenAI**'s GPT, having spent substantial amounts on API tokens. There was also a critique on Open Interpreter's optimization for a closed-source AI system, causing confusion due to being an open-source project itself.

- **Sharing Experience with Local LLM Performance**: Discussions included personal testing experiences with local LLMs, including **Phi-3-mini-128k-instruct** and **Groq** models, where one member observed significant performance issues with the former and issues with environmental setup. A member indicated that correcting the LLM's decisions might lead to better command execution.

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="http://ip_address:port/v1`">no title found</a>: no description found</li><li><a href="https://iyo.audio/">IYO</a>: IYO builds audio computers to welcome you to the world of audio computing. Be immersed in mixed audio reality where you talk to virtual audio agents who help you learn, work, shop and create.</li><li><a href="https://huggingface.co/microsoft/Phi-3-mini-128k-instruct">microsoft/Phi-3-mini-128k-instruct · Hugging Face</a>: no description found</li><li><a href="https://tenor.com/view/life-barrel-me-roll-gif-17943995">Life Barrel GIF - Life Barrel Me - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/search?q=repo%3AOpenInterpreter%2F01%20skill&type=code">Build software better, together</a>: GitHub is where people build software. More than 100 million people use GitHub to discover, fork, and contribute to over 420 million projects.</li><li><a href="https://github.com/search?q=repo%3AOpenInterpreter%2Fopen-interpreter%20skill&type=code">Build software better, together</a>: GitHub is where people build software. More than 100 million people use GitHub to discover, fork, and contribute to over 420 million projects.</li><li><a href="https://lu.ma/iu1wijgd">Open Source AI Hackathon #4 · Luma</a>: Following feedback from our last hackathon, we have found a sponsor for LLMs! OctoAI will provide all registrants the opportunity to get $50 in…</li><li><a href="https://github.com/OpenInterpreter/open-interpreter?tab=readme-ov-file#android">GitHub - OpenInterpreter/open-interpreter: A natural language interface for computers</a>: A natural language interface for computers. Contribute to OpenInterpreter/open-interpreter development by creating an account on GitHub.</li><li><a href="https://huggingface.co/microsoft/">microsoft (Microsoft)</a>: no description found</li><li><a href="https://rubiks.ai/search/?id=2doji3-eejo-88bg-v35a-sz678y8bv5y1">What is Reka Core?</a>:  **Reka Core** is a frontier-class, multimodal language model developed by Reka. It is one of only two commercially available comprehensive multimodal solutions, capable of processing and understandin...</li><li><a href="https://github.com/OpenInterpreter/open-interpreter?tab=readme-ov-file#running-open-interpreter-locally">GitHub - OpenInterpreter/open-interpreter: A natural language interface for computers</a>: A natural language interface for computers. Contribute to OpenInterpreter/open-interpreter development by creating an account on GitHub.</li><li><a href="https://github.com/OpenInterpreter/open-interpreter/commits/59956e01ebedc74e0bfed80352ea0a90ecf154b1/interpreter/core/computer/skills/skills.py">History for interpreter/core/computer/skills/skills.py - OpenInterpreter/open-interpreter</a>: A natural language interface for computers. Contribute to OpenInterpreter/open-interpreter development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1235874656725110856)** (104 messages🔥🔥): 

- **TMC Protocol for iOS Implementation**: A member is reimplementing the TMC protocol for iOS to allow access to native features. They question the benefits of using the TMC protocol over normal function calling and await clarification on its advantages.
  
- **Setting Up O1 with Azure Open AI Models**: A member is facing difficulties in setting up O1 to work with Azure Open AI models, noting that details in the .env are being ignored, despite OI working fine. They seek tips to resolve the issue after previous attempts failed.
  
- **Inquiries About O1 iOS App Release**: Members inquire about the status of the O1 iOS app, with one sharing a link to the [GitHub repository](https://github.com/OpenInterpreter/01/tree/main/software/source/clients/mobile) which includes the related source files. Further discussions suggest the app is a work in progress and directions are given via a [YouTube link](https://youtube.com/clip/UgkxfnZt5xbMkao8C0DmdsRTpU2bn_iaWtOI?si=wlcIV_ySO6gAfncF) on building for both Android and iOS using Expo.

- **Technical Troubles and Solutions for O1**: Members are troubleshooting various issues with O1, including problems with installing poetry, utilizing spacebar for commands, and difficulty in running local models. Suggestions for resolving these include using conda environments, downgrading Python versions, and installing packages correctly.

- **Exploring the Compatibility of Microsoft's Phi-3 Mini**: One user asks if they can use Microsoft's Phi-3 Mini model with Open Interpreter, and another provides instructions to install the model and select it from the launch list.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.openinterpreter.com/language-models/custom-models">no title found</a>: no description found</li><li><a href="https://huggingface.co/microsoft/Phi-3-mini-128k-instruct">microsoft/Phi-3-mini-128k-instruct · Hugging Face</a>: no description found</li><li><a href="https://github.com/OpenInterpreter/01/tree/main/software/source/clients/mobile">01/software/source/clients/mobile at main · OpenInterpreter/01</a>: The open-source language model computer. Contribute to OpenInterpreter/01 development by creating an account on GitHub.</li><li><a href="https://github.com/OpenInterpreter/open-interpreter/pull/1258">Fix Documentation by rbrisita · Pull Request #1258 · OpenInterpreter/open-interpreter</a>: Describe the changes you have made: Fixing documentation for custom model usage. Reference any relevant issues (e.g. &quot;Fixes #000&quot;): Fixes #1182 Pre-Submission Checklist (optional but appreci...
</li>
</ul>

</div>
  

---


**OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1236133517772193885)** (15 messages🔥): 

- **STT Challenges for AI Vtubers**: A member highlighted they've implemented **Speech-to-Text (STT)** using **fast whisper** as push-to-talk, experiencing challenges with live transcription, such as the AI interrupting users and transcribing background speech. It was suggested to use a *trigger word* to cue the system but deemed awkward for a virtual streamer context.
  
- **Encouraging AI Interaction with Stream Audiences**: The AI vtuber primarily responds to chat via the **Twitch API**, but for silence periods, a human catalyst can sustain interaction until an audience forms or the AI learns to engage with a game, representing the early phase of integrating Twitch chat interactions.
  
- **AI Managed Twitch Chat Interactions Plan**: The approach to manage Twitch chat involves setting up a separate LLM instance, which will understand the dialog stream and user messages to create responses, with the goal to eventually have a chatbot that comprehensively interacts with live chat audiences.
  
- **Control Over LLM Behavior Through Prompts**: Differentiating between standard models and Instruct models pertaining to prompts was emphasized; using an **Instruct model**, which has been fine-tuned to better follow instructions, was recommended for controllable outcomes.

- **Sharing Practical AI Integration Code**: The **main.py** file on a member's GitHub was mentioned to contain working code for chatbot integration, where users can simply swap the system prompt to suit their implementation needs.
  

---



**Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1235910528711528488)** (113 messages🔥🔥): 

- **Paper Follow-Ups Spark Interest**: Members were sharing links to related papers validating how **large language models (LLMs)** handle multilingualism and discussing the framework that portrays the processing of multilingual inputs by LLMs with links to papers like [How Mixture Models handle Multilingualism](https://arxiv.org/abs/2402.18815v1).
  
- **Adversarial Challenges and Architectural Discussions**: The community engaged in a technical discussion on adversarial robustness, the potential of scaling models for improved defense, and the need for systemic hierarchies or buffers to prevent exploitation, citing a relevant [paper on addressing LLM vulnerabilities](http://arxiv.org/abs/2404.13208).

- **Job Search Shares and Community Support**: A member actively sought employment opportunities, sharing their LinkedIn and Google Scholar profiles, and highlighting their experience with **EleutherAI** and contributions to the **Polyglot** team and **OSLO project**.

- **Improving in-Context Learning Measurement**: There was a proposal for a new benchmark methodology to measure in-context learning performance of models by varying the number of shots, which spurred a dialogue on the best approaches to assess this aspect of LLM behavior.

- **ICLR Meet-Up Coordination**: Several community members discussed and arranged a meet-up at **ICLR**, sharing plans and expressing excitement about meeting in person, despite some facing travel constraints like visa issues.

- **Exploring the Role of System Prompts**: A member mentioned an interest in exploring how the system prompt affects model performance using the **lm-evaluation-harness**, but noted difficulty finding a way to specify the system prompt using **Hugging Face models**.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2402.18819">Dual Operating Modes of In-Context Learning</a>: In-context learning (ICL) exhibits dual operating modes: task learning, i.e., acquiring a new skill from in-context samples, and task retrieval, i.e., locating and activating a relevant pretrained ski...</li><li><a href="http://arxiv.org/abs/2404.13208">The Instruction Hierarchy: Training LLMs to Prioritize Privileged Instructions</a>: Today&#39;s LLMs are susceptible to prompt injections, jailbreaks, and other attacks that allow adversaries to overwrite a model&#39;s original instructions with their own malicious prompts. In this w...</li><li><a href="https://arxiv.org/abs/2402.12530">Parallel Structures in Pre-training Data Yield In-Context Learning</a>: Pre-trained language models (LMs) are capable of in-context learning (ICL): they can adapt to a task with only a few examples given in the prompt without any parameter update. However, it is unclear w...</li><li><a href="https://arxiv.org/abs/2402.18815v1">How do Large Language Models Handle Multilingualism?</a>: Large language models (LLMs) demonstrate remarkable performance across a spectrum of languages. In this work, we delve into the question: How do LLMs handle multilingualism? We introduce a framework t...</li><li><a href="https://scholar.google.com/citations?user=AbpywLMAAAAJ&hl=en">Kichang Yang</a>: Soongsil University - Cited by 50 - Machine Learning - NLP</li><li><a href="https://github.com/jason9693">jason9693 - Overview</a>: AI Research Engineer. jason9693 has 71 repositories available. Follow their code on GitHub.
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1235963390623613040)** (165 messages🔥🔥): 

- **Scaled Transformers Conquer Chess**: A new [research paper](https://arxiv.org/abs/2402.04494) discusses a 270M parameter transformer model trained on 10 million chess games annotated by Stockfish 16, which achieves remarkable performance in Lichess blitz games and chess puzzles without domain-specific tweaks or explicit search algorithms. This model outperforms AlphaZero's policy and value networks sans MCTS and raises questions about the impact of scale on strategy games.

- **Resurrection of GPT-2**: Messages allude to a significant gap between postings and interactions on the server, such as a member mentioning a three-year break before replying to older posts and another maintaining a running interaction with outdated content.

- **Enhancing LLM Search with 'Maieutic Prompting'**: The concept of [Maieutic Prompting](https://arxiv.org/abs/2205.11822), a method to improve LLM's inference from noisy and inconsistent data by generating a tree of abductive explanations, was introduced, albeit with skepticism about its practical effectiveness.

- **Challenges and Considerations in Human-Led Evaluations**: A detailed discourse covered the complexities in determining sample size, the significance level, and statistical tests for human evaluations in research, like comparing two chatbots. Discussions mentioned non-inferiority testing and the analysis of systematic error to evaluate the intervention's impact meaningfully.

- **Non-Fine-Tunable Learning to Prevent Model Misuse**: A new concept named [non-fine-tunable learning](https://arxiv.org/abs/2404.12699), showcased in the SOPHON framework, aims to protect pre-trained models from being fine-tuned for unethical use while maintaining performance in their original tasks. Concerns were raised about the potential overreach of such protections limiting the adaptability of future models for legitimate applications.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://hehao13.github.io/projects-CameraCtrl/">CameraCtrl</a>: no description found</li><li><a href="https://fxtwitter.com/sama/status/1787222050589028528">Tweet from Sam Altman (@sama)</a>: im-a-good-gpt2-chatbot</li><li><a href="https://arxiv.org/abs/2404.12699">SOPHON: Non-Fine-Tunable Learning to Restrain Task Transferability For Pre-trained Models</a>: Instead of building deep learning models from scratch, developers are more and more relying on adapting pre-trained models to their customized tasks. However, powerful pre-trained models may be misuse...</li><li><a href="https://xkcd.com/882/">Significant</a>: no description found</li><li><a href="https://arxiv.org/abs/2402.04494#deepmind">Grandmaster-Level Chess Without Search</a>: The recent breakthrough successes in machine learning are mainly attributed to scale: namely large-scale attention-based architectures and datasets of unprecedented scale. This paper investigates the ...</li><li><a href="https://en.wikipedia.org/wiki/Lady_tasting_tea">Lady tasting tea - Wikipedia</a>: no description found</li><li><a href="https://arxiv.org/abs/2205.11822">Maieutic Prompting: Logically Consistent Reasoning with Recursive Explanations</a>: Despite their impressive capabilities, large pre-trained language models (LMs) struggle with consistent reasoning; recently, prompting LMs to generate explanations that self-guide the inference has em...</li><li><a href="http://arxiv.org/abs/2405.01535">Prometheus 2: An Open Source Language Model Specialized in Evaluating Other Language Models</a>: Proprietary LMs such as GPT-4 are often employed to assess the quality of responses from various LMs. However, concerns including transparency, controllability, and affordability strongly motivate the...</li><li><a href="https://www.nature.com/articles/s41562-017-0189-z">Redefine statistical significance - Nature Human Behaviour</a>: We propose to change the default P-value threshold for statistical significance from 0.05 to 0.005 for claims of new discoveries.</li><li><a href="https://www.melonimarco.it/en/2021/03/08/stockfish-and-lc0-test-at-different-number-of-nodes/">Stockfish and Lc0, test at different number of  nodes &#8211; MeloniMarco.it</a>: no description found
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1235971685035933758)** (9 messages🔥): 

- **Scaling Laws in Pre-training and Fine-tuning**: A link to a study on [arXiv](https://arxiv.org/abs/2102.01293) details empirical scaling laws for transfer learning. The study finds that pre-trained models continue to improve on a fixed-size dataset due to effective data transferred from pre-training, described by a power-law of parameter count, and fine-tuning dataset size.

- **Accuracy Amidst Dataset Concerns**: Two separate members discussed the implications of [Papers With Code](mailto:hello@paperswithcode.com) showing over 70% accuracy within two years for math problem-solving. A member suggested that some recent advancements might be a result of data leakage from datasets specifically designed for performance measurements like GSM8K and MATH.

- **Inclusion of Exam Data in Pre-Training**: Members discussed the possibility of OpenAI including GSM8K and MATH data in its pre-training datasets. While some expressed uncertainty about the adherence to rules, they clarified that fine-tuning on MATH was standard practice for achieving state-of-the-art in 2021.

- **Evaluating Original Test Dataset Performance**: A member provided a link to [odyssey-math on GitHub](https://github.com/protagolabs/odyssey-math) and commented on a reported baseline of 47% accuracy for gpt-4-turbo on this original test dataset. They plan to subsample some of the problems to assess the dataset's difficulty, noting the small size of about 350 problems.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://paperswithcode.com/sota/math-word-problem-solving-on-math">Papers with Code - MATH Benchmark (Math Word Problem Solving)</a>: The current state-of-the-art on MATH is GPT-4 Turbo (MACM, w/code, voting). See a full comparison of 110 papers with code.</li><li><a href="https://arxiv.org/abs/2102.01293">Scaling Laws for Transfer</a>: We study empirical scaling laws for transfer learning between distributions in an unsupervised, fine-tuning setting. When we train increasingly large neural networks from-scratch on a fixed-size datas...</li><li><a href="https://github.com/protagolabs/odyssey-math">GitHub - protagolabs/odyssey-math</a>: Contribute to protagolabs/odyssey-math development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1235905471685333092)** (7 messages): 

- **Transformer Models Decoded**: A new [primer on transformer-based language models](https://twitter.com/javifer_96/status/1786317169979970046) has been introduced, offering insights into the model components and interpretation methods garnered from years of research, along with an extensive survey of interpretability tools.
- **Seeking Model Deployment Assistance**: One member has requested help with model deployment but did not provide further details on the issue they are facing.
- **Cross-Model Generalization Confirmed**: Results on language model interpretability using English as a pivot language have been replicated across various models, including **llama 1, 2**, and now **llama 3**, as shared in a recent [tweet](https://twitter.com/Butanium_/status/1786394217478004950).
- **Diving Deep into Weight Tying Issues**: A member is exploring weight tying in open models like **Phi-2** and **Mistral-7B** using **LogitLens** and has come across unexpected results in the output layers.
- **Clarifying Weight Tying Conundrum**: Further investigation has led to the conclusion that contemporary open models, in fact, do not employ weight tying, which clarifies the earlier irregular results observed.
  

---


**Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1235953686811770921)** (3 messages): 

- **Prometheus Model Sparks Interest**: Members expressed interest in the [**AlekseiPravdin/prometheus-7b-v2_0-gguf**](https://huggingface.co/AlekseiPravdin/prometheus-7b-v2_0-gguf) model on Hugging Face, suggesting it might be a significant improvement for inclusion in their work.
- **Call for Collaboration**: A member has volunteered to assist with the integration of the aforementioned model, highlighting the benefits seen from chat templates in performance metrics.
- **Preparation for Integration Underway**: Work on a product requirement document (PRD) is in progress for implementing improvements based on **AlekseiPravdin/prometheus-7b-v2_0-gguf**. The model's author is present in the chat, indicating potential direct collaboration.

**Link mentioned**: <a href="https://huggingface.co/papers/2405.01535">Paper page - Prometheus 2: An Open Source Language Model Specialized in Evaluating
  Other Language Models</a>: no description found

  

---



**OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1236191657049981049)** (3 messages): 

- **Llama 3 Lumimaid 8B Now Available**: OpenRouter has released a new model, [Llama 3 Lumimaid 8B](https://openrouter.ai/models/neversleep/llama-3-lumimaid-8b), available for the year 2023 - 2024.
- **Extended Llama 3 Lumimaid 8B Released**: An extended version of Llama 3 Lumimaid 8B is also on offer, providing users with additional features, aptly named [Llama 3 Lumimaid 8B Extended](https://openrouter.ai/models/neversleep/llama-3-lumimaid-8b:extended).
- **Price Cut for Llama 3 8B Instruct Extended**: There's good news for users looking for a bargain as the price for [Llama 3 8B Instruct Extended](https://openrouter.ai/models/meta-llama/llama-3-8b-instruct:extended) has been reduced.
- **Temporary Downtime for Lynn Models**: A server update will lead to a brief ~10-minute downtime for [Lynn](https://openrouter.ai/models/lynn) and associated models.
- **Soliloquy L3 8B Updated to v2**: The Soliloquy L3 8B model has been upgraded to version 2, boasting improvements such as repetition and retrieval issue fixes, enhanced instruction following, and a new price of $0.15 per 1M tokens. Explore [Soliloquy L3 8B v2 here](https://openrouter.ai/models/lynn/soliloquy-l3).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openrouter.ai/models/lynn/soliloquy-l3)">Lynn: Llama 3 Soliloquy 8B v2 by lynn | OpenRouter</a>: Soliloquy-L3 v2 is a fast, highly capable roleplaying model designed for immersive, dynamic experiences. Trained on over 250 million tokens of roleplaying data, Soliloquy-L3 has a vast knowledge base,...</li><li><a href="https://openrouter.ai/models/neversleep/llama-3-lumimaid-8b)">Llama 3 Lumimaid 8B by neversleep | OpenRouter</a>: The NeverSleep team is back, with a Llama 3 8B finetune trained on their curated roleplay data. Striking a balance between eRP and RP, Lumimaid was designed to be serious, yet uncensored when necessar...</li><li><a href="https://openrouter.ai/models/neversleep/llama-3-lumimaid-8b:extended>)">Llama 3 Lumimaid 8B by neversleep | OpenRouter</a>: The NeverSleep team is back, with a Llama 3 8B finetune trained on their curated roleplay data. Striking a balance between eRP and RP, Lumimaid was designed to be serious, yet uncensored when necessar...</li><li><a href="https://openrouter.ai/models/meta-llama/llama-3-8b-instruct:extended>)">Meta: Llama 3 8B Instruct by meta-llama | OpenRouter</a>: Meta&#x27;s latest class of model (Llama 3) launched with a variety of sizes &amp; flavors. This 8B instruct-tuned version was optimized for high quality dialogue usecases.  It has demonstrated strong...</li><li><a href="https://openrouter.ai/models/lynn>)">OpenRouter</a>: Browse models on OpenRouter
</li>
</ul>

</div>
  

---


**OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1236801053090119691)** (3 messages): 

- **Introducing eGirlfriend AI**: A member has built an initial version of a project called [eGirlfriend AI](https://egirlfriend.ai) and invites the community for feedback, noting that it's **100% free**.

- **Family-Friendly Streamlit Chat App**:
 A chat application designed for family called *Family Chat* has been created to utilize OpenRouter API and OpenAI's API cost-effectively, featuring **Conversational Memory**, **PDFChat**, and **Image Generation**. You can explore and contribute to it on [GitHub](https://github.com/DrDavidL/family-chat/blob/main/README.md).

- **Rubik's AI Pro Seeks Beta Testers**: 
 The creator of an advanced research assistant and search engine named **Rubik's AI Pro** is seeking beta testers, offering 2 months of free premium which includes access to models like **GPT-4 Turbo** and **Mistral Large**. Interested parties can sign up and enter promo code `RUBIX` [here](signup.php).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://egirlfriend.ai,">no title found</a>: no description found</li><li><a href="https://rubiks.ai/">Rubik's AI - AI research assistant & Search Engine</a>: no description found
</li>
</ul>

</div>
  

---


**OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1235990719760568461)** (248 messages🔥🔥): 

- **Gemini Pro Glitch Fixed**: An issue with **Gemini Pro** error messages was reported but resolved within days. Users were advised it's working and to contact support if the problem persists.

- **Lumimaid 70B Anticipation**: Discussions indicate communication with Mancer about hosting **Lumimaid 70B**, and the suggestion to enquire about it with Novita, a provider focused on RP models.

- **Phi-3 Hosting Uncertainty**: Despite interest, there seems to be a lack of providers currently hosting **Phi-3**, though Microsoft Azure runner is said to have it, albeit with no per-token pricing.

- **OpenRouter and AI Model Precision**: It's clarified that model providers on OpenRouter use different precisions; most run at **fp16**, and some at quantized **int8**.

- **Meta-Llama 3 120B Instruct Self Merge**: A **self-merged version of Meta-Llama 3 70B** has been noted on Hugging Face, inspired by other large merges, raising curiosity about the efficacy of self-merges compared to layer-mapped merges.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/mlabonne/Meta-Llama-3-120B-Instruct">mlabonne/Meta-Llama-3-120B-Instruct · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/abacusai/Fewshot-Metamath-OrcaVicuna-Mistral-10B">abacusai/Fewshot-Metamath-OrcaVicuna-Mistral-10B · Hugging Face</a>: no description found</li><li><a href="https://octo.ai/blog/mixtral-8x22b-is-now-available-on-octoai/">Mixtral 8x22B is now available on OctoAI Text Gen Solution | OctoAI</a>: You can run inferences against Mixtral 8x22B on OctoAI using /completions API, using curl or using the OpenAI SDK. Contact us to run a fine-tuned version.
</li>
</ul>

</div>
  

---



**LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1235984738683064388)** (7 messages): 

- **Reflective Self-Improving Agents**: LlamaIndex 0.10.34 introduces **introspective agents** that can boost their performance through reflection and self-critique without human intervention. This method and the `llama-index-agent-introspective` package are detailed in a [notebook](https://t.co/X8tJGXkcPM) with an installation guide, bearing a content warning for sensitive material.

- **Agentic RAG Advancements Demonstrated**: A video by @jasonzhou1993 presents an **overview of components** necessary for agentic RAG, featuring advanced document processing with LlamaParse + Firecrawl. The video is available [here](https://t.co/wR35iYIKjo) for those interested in constructing agentive systems.

- **Trust Assessments in RAG Responses**: @CleanlabAI developed a "Trustworthy Language Model" that assigns **trustworthiness scores** to Retrieval-Augmented Generation (RAG) responses, addressing the challenge of verifying the accuracy of generated content. More details about this feature can be found in their tweet [here](https://t.co/KW1XsllRqQ).

- **Guide for Local RAG Setups**: For those seeking a **fully local RAG pipeline**, @pavan_mantha1 provides an insightful handbook introducing the setup with @llama_index and a HyDE layer. Described as a lower-level guide compared to the "5 lines of code" Quickstart, the article is accessible via [this link](https://t.co/2RCvaxOzKo).

- **Hugging Face TGI Support Revealed by LlamaIndex**: LlamaIndex announces support for **Hugging Face TGI**, a toolkit ensuring optimized deployment for language models on Huggingface, now with features such as **function calling**, batched inference, and must faster latencies. Details about TGI's capabilities are outlined [here](https://t.co/3vGpxcbP18).

**Link mentioned**: <a href="https://t.co/X8tJGXkcPM">Introspective Agents: Performing Tasks With Reflection - LlamaIndex</a>: no description found

  

---


**LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1235849486564200489)** (226 messages🔥🔥): 

- **Exploring RAG with Controllable Agents**: A user inquired about implementing **Controllable agents** in a **Retrieval-Augmented Generation (RAG)** project, to make agents capable of asking follow-up questions for more precise retrieval results. A detailed implementation guide with LlamaIndex is provided, including links to relevant documentation like [Agent Runner](https://docs.llamaindex.ai/en/examples/agent/agent_runner/agent_runner/) and [Controllable Agent Runner](https://docs.llamaindex.ai/en/examples/agent/agent_runner/agent_runner_rag_controllable/).

- **LlamaIndex Memory Issues Troubleshooting**: Users discussed high VRAM usage and potential memory leak issues when using LlamaIndex, leading to slow cleanup and fallback to CPU processing. One user pointed to successfully resolving such issues with the new **[ollama v0.1.33 update](https://github.com/ollama/ollama/releases/tag/v0.1.33)**.

- **LLM Fine-tuning and Cost Discussions**: There were discussions on fine-tuning language models (LLMs) specifically for tasks like a light model that is specialized in a particular field. The costliness of fine-tuning was noted, with users looking for optimizable, cost-effective solutions.

- **Implementing Sharepoint Reader and VectorStore Challenges**: A member sought feedback on integrating the **SharePoint Reader** for loading files from SharePoint, and another experienced empty responses from a **SupabaseVectorStore** in LlamaIndex, indicating possible configuration issues.

- **Understanding and Optimizing Q&A Systems Over Excel Data**: One user inquired about the best approach for building a Q&A system over a moderately sized Excel table, focusing on providing contextually relevant information to complex queries.

- **Implementation and Configuration of LlamaIndex Specifics**: Various users discussed importing errors, the right pathways in `llama-index`, how to handle legal document data extraction, how to deal with embeddings for Intel processors, and configuring ReAct agents dynamically. Assistance was sought and exchanged among peers and with the help of **cheesyfishes**, presumably a knowledgeable figure in the community, offering guidance on LlamaIndex's usage and integration.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="http://localhost:11434",">no title found</a>: no description found</li><li><a href="https://llamahub.ai">Llama Hub</a>: no description found</li><li><a href="https://www.llamaindex.ai/contact">Talk to us — LlamaIndex, Data Framework for LLM Applications</a>: If you have any questions about LlamaIndex please contact us and we will schedule a call as soon as possible.</li><li><a href="https://docs.llamaindex.ai/en/stable/getting_started/starter_example_local/">Starter Tutorial (Local Models) - LlamaIndex</a>: no description found</li><li><a href="https://llama.meta.com/docs/how-to-guides/prompting">Prompting | How-to guides</a>: Prompt engineering is a technique used in natural language processing (NLP) to improve the performance of the language model by providing them with more context and information about the task in hand.</li><li><a href="https://github.com/ollama/ollama/releases/tag/v0.1.33">Release v0.1.33 · ollama/ollama</a>: New models:  Llama 3: a new model by Meta, and the most capable openly available LLM to date Phi 3 Mini: a new 3.8B parameters, lightweight, state-of-the-art open model by Microsoft. Moondream moon...</li><li><a href="https://github.com/run-llama/llama_index/tree/main/llama-index-integrations/embeddings/llama-index-embeddings-huggingface">llama_index/llama-index-integrations/embeddings/llama-index-embeddings-huggingface at main · run-llama/llama_index</a>: LlamaIndex is a data framework for your LLM applications - run-llama/llama_index</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/embeddings/optimum_intel/">Optimized Embedding Model using Optimum-Intel - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/optimizing/fine-tuning/fine-tuning/">Fine-Tuning - LlamaIndex</a>: no description found</li><li><a href="https://python.langchain.com/docs/modules/composition/">Composition | 🦜️🔗 LangChain</a>: This section contains higher-level components that combine other arbitrary systems (e.g. external APIs and services) and/or LangChain primitives together.</li><li><a href="https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/">Meta Llama 3 | Model Cards and Prompt formats</a>: Special Tokens used with Meta Llama 3. A prompt should contain a single system message, can contain multiple alternating user and assistant messages, and always ends with the last user message followe...</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/ingestion/redis_ingestion_pipeline/?h=ingestionpipeline">Redis Ingestion Pipeline - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/api_reference/storage/vector_store/supabase#llama_index.vector_stores.supabase.SupabaseVectorStore>).">Supabase - LlamaIndex</a>: no description found</li><li><a href="https://github.com/run-llama/llama_index/pull/13196">Call Cohere RAG inference with `documents` argument by co-antwan · Pull Request #13196 · run-llama/llama_index</a>: Description Adds support for Cohere.chat&#39;s documents argument when using in RAG pipelines. This ensures proper formatting on Cohere&#39;s client side, and leads to better downstream performance. T...</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/llm/monsterapi#rag-approach-to-import-external-knowledge-into-llm-as-context>).">no title found</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/optimizing/fine-tuning/fine-tuning#finetuning-embeddings>).">Fine-Tuning - LlamaIndex</a>: no description found
</li>
</ul>

</div>
  

---


**LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1237021916792098916)** (4 messages): 

- **Seeking NL-SQL Bot Wisdom**: A member is creating a **NL-SQL chat bot** for a complex database with hundreds of tables and inquires about using a **HyDE method**. They are exploring solutions for improving the LLM's accuracy for generating SQL queries, noting that HyDE has mostly been used in text-based chatbots.
- **Introspective Agents Discourse**: There's a mention of an article titled **"Introspective Agents with LlamaIndex"**, indicating a new approach or development involving introspective agents. A link to the article was shared: [Introspective Agents with LlamaIndex](https://medium.com/ai-artistry/introspective-agents-with-llamaindex-777d018f791d).
  

---



**OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1235886536466239549)** (33 messages🔥): 

- **Hermes Gets Speedy on Android**: A member expressed amazement over the **inference speed** of **Hermes 2 Pro Llama 3** on an 8GB RAM Android device, attributing the performance to **llama.cpp**.

- **Anime Shapes AI Innovation**: There was humorous discussion suggesting that advancements in AI and technological innovation are seemingly intertwined with the proliferation of **anime** both in **question-answering** and **image generation**.

- **Llama.cpp Merges Performance-Enhancing PR**: A member shared news about a new pull request merged into **llama.cpp** that results in a **30% speed improvement** for inference, seemingly inviting the creation of **more anime**.

- **Axolotl's Progressive Documentation**: A link to the **work-in-progress documentation** for the **Axolotl community** was shared with an invitation for feedback.

- **Gradient Checkpointing Optimizations Reported**: An update was noted regarding the **new unsloth** gradient checkpointing leading to reduced VRAM usage, showcasing the active effort in the community to optimize memory utilization in machine learning processes.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/turboderp/Cat-Llama-3-70B-instruct">turboderp/Cat-Llama-3-70B-instruct · Hugging Face</a>: no description found</li><li><a href="https://www.philschmid.de/vllm-inference-endpoints">Deploy open LLMs with vLLM on Hugging Face Inference Endpoints</a>: In this blog post, we will show you how to deploy open LLMS with vLLM on Hugging Face Inference Endpoints.</li><li><a href="https://x.com/granawkins/status/1786428318478168447">Tweet from Grant♟️ (@granawkins)</a>: sota RAG in 2024</li><li><a href="https://x.com/tomshardware/status/1786807369961210203">Tweet from Tom's Hardware (@tomshardware)</a>: Multi-million dollar Cheyenne supercomputer auction ends with $480,085 bid — buyer walked away with 8,064 Intel Xeon Broadwell CPUs, 313TB DDR4-2400 ECC RAM, and some water leaks https://trib.al/7BzUc...</li><li><a href="https://github.com/HVision-NKU/StoryDiffusion/tree/main?tab=readme-ov-file">GitHub - HVision-NKU/StoryDiffusion: Create Magic Story!</a>: Create Magic Story! Contribute to HVision-NKU/StoryDiffusion development by creating an account on GitHub.</li><li><a href="https://axolotl.continuumlabs.pro/">Introduction | Continuum Training Platform | Axolotl Training Platform</a>: no description found
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1235918795705417758)** (8 messages🔥): 

- **Configurability Comes to Gradio**: One member sought assistance on making Gradio options like making the demo private and setting an IP address configurable via yaml. The solution involved adding those options into yaml and modifying the code to parse the settings as demonstrated in [their implementation](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1591).

- **Deep Dive into Gradio Token Troubles**: There was a puzzling issue where Gradio did not utilize the correct tokens for the llama3 model, with it printing `<|end_of_text|>` tokens unexpectedly. It appeared that Gradio's default tokens might unintentionally overwrite a loaded tokenizer's settings unless special tokens are specified.

- **Pushing for a More Dynamic Gradio**: A code change was discussed to allow dynamic configuration of Gradio's parameters, such as "private", "server_name", and "port". This will enable greater control over Gradio's behavior through yaml configuration.

- **PR Ready for Review**: A pull request was submitted addressing Gradio customization, adding configurable parameters for various hardcoded options in the project, capturing important details and demonstrating the implementation with a [GitHub PR](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1591).

- **Issue or Pull Request? The Eternal Question**: A member inquired whether to open an issue for a problem or to just submit a pull request. While the response was not recorded, the member took initiative and created a pull request to address the underlying issue.

**Link mentioned**: <a href="https://github.com/OpenAccess-AI-Collective/axolotl/pull/1591">Gradio configuration parameters by marijnfs · Pull Request #1591 · OpenAccess-AI-Collective/axolotl</a>: Various parameters of Gradio were hardcoded (e.g. share=True, ip address, port, number of tokens, temperature) I made them configurable here. Additionally the default tokens were overwritten into t...

  

---


**OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1235866742010417153)** (8 messages🔥): 

- **Calling Inference on Trained Llama3**: There was a question about how to call inference after training **llama3** with the fft script, clarifying that the usual **qlora** command and **qlora_model_dir** don't seem applicable.
- **Tuning Inference Parameters**: A member recommended using a parameter setting of **4,4** as effective for an unspecified context, implying success with these settings.
- **Conversion of Safetensors to GGUF**: One user sought assistance in converting safetensors to **gguf** with more options than provided by **llama.cpp**, specifically mentioning formats like `Q4_K` and `Q5_K`.
- **Script for Llama.cpp Conversions**: The user was directed to **llama.cpp**'s conversion scripts, with a particular call-out to [convert-gg.sh](https://github.com/ggerganov/llama.cpp/blob/master/scripts/convert-gg.sh), presumably for dealing with **gguf conversion options**.
- **Axolotl Community Documentation**: A link to the Axolotl community documentation was shared, which requires more work especially on merging model weights post-training and using the model for inference, with invites for feedback at [Axolotl Community Docs](https://axolotl.continuumlabs.pro/).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/ggerganov/llama.cpp/blob/master/scripts/convert-gg.sh">llama.cpp/scripts/convert-gg.sh at master · ggerganov/llama.cpp</a>: LLM inference in C/C++. Contribute to ggerganov/llama.cpp development by creating an account on GitHub.</li><li><a href="https://axolotl.continuumlabs.pro/">Introduction | Continuum Training Platform | Axolotl Training Platform</a>: no description found
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/1236106691053883542)** (39 messages🔥): 

- **CodeTester Dataset Expansion**: An *updated Python dataset* from Vezora now features 143,327 carefully tested and working examples of code, created to assist with extracting and verifying Python code snippets from Alpaca-formatted datasets. More information about the dataset and its creation process can be found on [Hugging Face's dataset repository](https://huggingface.co/datasets/Vezora/Tested-143k-Python-Alpaca).

- **Tough Time Training Llama3 on Math**: Members discussed difficulties in improving model performance on mathematical content with Llama3, noting a *decrease in math topic scores* despite training on datasets like orca-math-word-problems-200k and MetaMathQA, which are available at [MathInstruct](https://huggingface.co/datasets/TIGER-Lab/MathInstruct) and [MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA).

- **Impact of Quantization on Model Performance**: One member highlighted the potential negative impact of **llama.cpp quantization** on model performance, referencing a discussion about Llama3 GGUF conversion with merged LORA Adapter on GitHub, which can be further explored in [this issue](https://github.com/ggerganov/llama.cpp/issues/7062#issuecomment-2094961774).

- **Evaluation Scripts and Prompting**: A member used the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for inference and evaluation of Llama3, while others pointed out the importance of ensuring the correct prompt format and raised questions about the potential effects of using Alpaca-format prompts on model performance.

- **Prompt Format Puzzles**: There is ongoing debate about how the prompt format during finetuning, such as using the Alpaca format, might affect model performance. Members are contemplating whether this could lead to issues even if the model does not generate out-of-vocabulary end-of-text tokens.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/">Meta Llama 3 | Model Cards and Prompt formats</a>: Special Tokens used with Meta Llama 3. A prompt should contain a single system message, can contain multiple alternating user and assistant messages, and always ends with the last user message followe...</li><li><a href="https://huggingface.co/datasets/Vezora/Tested-143k-Python-Alpaca">Vezora/Tested-143k-Python-Alpaca · Datasets at Hugging Face</a>: no description found</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/3367fca73253c85e386ef69af3068d42cea09e4f/src/axolotl/prompters.py#L47">axolotl/src/axolotl/prompters.py at 3367fca73253c85e386ef69af3068d42cea09e4f · OpenAccess-AI-Collective/axolotl</a>: Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness">GitHub - EleutherAI/lm-evaluation-harness: A framework for few-shot evaluation of language models.</a>: A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/7062#issuecomment-2094961774">Llama3 GGUF conversion with merged LORA Adapter seems to lose training data randomly · Issue #7062 · ggerganov/llama.cpp</a>: I&#39;m running Unsloth to fine tune LORA the Instruct model on llama3-8b . 1: I merge the model with the LORA adapter into safetensors 2: Running inference in python both with the merged model direct...</li><li><a href="https://huggingface.co/datasets/TIGER-Lab/MathInstruct">TIGER-Lab/MathInstruct · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/meta-math/MetaMathQA">meta-math/MetaMathQA · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-phorm-bot](https://discord.com/channels/1104757954588196865/1225558824501510164/1236482608502935615)** (27 messages🔥): 

- **Gradient Clipping Inquiries**: Discussions arose around setting gradient clipping in **Axolotl** using the Axolotl `TrainingArguments` or within a YAML configuration. Phorm suggested setting `max_grad_norm` in the `TrainingArguments` or within the YAML file under optimization settings.

- **Hyperlink to the Documentation Needed**: Members pointed out that specifying gradient clipping in the Axolotl YAML might not be reflected in the documentation due to a transition to quarto markdown, indicating a need to update the documentation index.

- **Modifying the Chatbot Prompt**: A user inquired about modifying the system prompt for conversational training within the ShareGPT dataset format. Phorm indicated to adjust the conversation template or the initial message in the `ShareGPTPrompter` class or in associated configuration files.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=d76285fb-b795-43de-a278-b9adfdec1559)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=12d8fd24-8f30-4de3-bb8b-7a85951a30ec)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=c1ca4368-cf3a-4dee-8f17-6686eaf48b1a)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=ef7c3959-d5a0-4b42-b13d-5ccc8940f344)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.
</li>
</ul>

</div>
  

---



**Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1236044647785304109)** (95 messages🔥🔥): 

- **Gary for Live! A Compute Music Journey**: A member shared a link to [gary4live on GitHub](https://github.com/betweentwomidnights/gary4live), a work-in-progress project involving python continuations and Ableton, encouraging others to take a look at the code.
- **Suno and Music Generation Discussion**: Conversations unfolded around using Suno for generating music, as well as the capabilities of other music generation setups like *Musicgen*. There was a particular interest in exploring how these models handle different audio elements and whether they generate assets like sheet music.
- **Deep Dive into Music Model Tokens**: The chat navigated through the intricacies of music model tokens, with discussions emphasizing Suno's tokenization of audio and questions about the length and composition of these tokens. References to architectural designs from papers were mentioned, yet specific details weren't fleshed out within the discussion.
- **Latent Spaces in Audio Synthesis**: Participants discussed the potential of multimodal models integrating audio directly without text intermediates, highlighting the relevance of audio inclusion for truly omnimodal capabilities. The conversation included ideas like using model generations to replace audio channels in real-time applications.
- **Exploring Commercial Use and Licensing for Stable Audio**: One member raised questions regarding the commercial use and licensing of outputs from stable audio models. The discussion veered towards the real-time applications of such models, like live performance looping with AI.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://discord.gg/aSTTaUfm">Discord - A New Way to Chat with Friends &amp; Communities</a>: Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.</li><li><a href="https://notesbylex.com/snake-activation-function">Tweet from Snake Activation Function</a>: Snake is a neural network activation function useful for modelling problems with a &quot;periodic induction bias&quot; - in other words, problems with regular, repeating patterns - for...</li><li><a href="https://x.com/yikesawjeez/status/1786299657460855174">Tweet from yikes (@yikesawjeez)</a>: wake up babe new neural network architecture just dropped https://arxiv.org/abs/2404.19756</li><li><a href="https://arxiv.org/abs/2404.10301v1">Long-form music generation with latent diffusion</a>: Audio-based generative models for music have seen great strides recently, but so far have not managed to produce full-length music tracks with coherent musical structure. We show that by training a ge...</li><li><a href="https://github.com/betweentwomidnights/gary4live">GitHub - betweentwomidnights/gary4live: This is gary. python continuations plus continuations inside of ableton. It is a WIP by a newb.</a>: This is gary. python continuations plus continuations inside of ableton. It is a WIP by a newb. - betweentwomidnights/gary4live
</li>
</ul>

</div>
  

---



**AI Stack Devs (Yoko Li) ▷ #[ai-companion](https://discord.com/channels/1122748573000409160/1122788693950857238/1237016300333694997)** (6 messages): 

- **Clarity on Cloud Subscription Fees**: Members confirmed **no cloud subscription fees** are required if you run it locally; the tool works fine with 6 GB VRAM and includes free voice output.
- **Owning Downloads**: It was highlighted that once you download characters and models with **Faraday**, they are yours to keep *forever*.
- **Local Use Overrides Cloud Subscription**: A GPU of sufficient power negates the need for a cloud subscription, which was suggested as an optional donation to the tool's developers.
  

---


**AI Stack Devs (Yoko Li) ▷ #[team-up](https://discord.com/channels/1122748573000409160/1128471951963328512/1237055950032998422)** (2 messages): 

- **Call for Collaboration on a Hip-Hop Simulation**: A member expressed interest in creating a fun simulation referencing the situation between **Kendrick** and **Drake**. Another member responded positively to the collaboration call.
  

---


**AI Stack Devs (Yoko Li) ▷ #[ai-town-discuss](https://discord.com/channels/1122748573000409160/1132926337598902293/1235974067073585294)** (15 messages🔥): 

- **AI Leadership Elections Discussed**: Curiosity was shown regarding whether AIs elect leaders and specifically about a mayor election depicted in the original simulation paper, which apparently *never actually triggers* in the simulation.
- **Setting Up AI Elections in Player Bios**: A member stated that setting up AI elections in player bios would be *simple to set up*, referencing a curiosity about mayoral events in an AI simulation.
- **AI-Westworld Public Beta and The THING Simulation**: A tweet by @TheoMediaAI highlighted exploring two AI world simulations, including the **AI-Westworld** by @fablesimulation which is in public beta, and recreating The THING movie in @realaitown.
- **Introducing AI Town Player for Replayability**: A tweet by @cocktailpeanut introduced the **AI Town Player web app**, which allows replaying any AI Town by importing a sqlite file, noting that the whole AI Town is stored in a single sqlite file via @convex_dev and is compatible with Mac & Linux but not Windows.
- **AI Simulated Party Makes News**: A feature on [sfstandard.com](https://sfstandard.com/2024/05/04/mission-control-hacker-house-san-francisco-ai-simulated-party/) described an **AI Simulated Party** at Mission Control in San Francisco, where human attendees were paralleled by AI versions running around a digitized version of the event on display.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/TheoMediaAI/status/1786377663889678437">Tweet from Theoretically Media (@TheoMediaAI)</a>: Exploring Two remarkable AI World Simulations: First, the AI-Westworld from @fablesimulation (PUBLIC BETA is OPEN!), and also taking @realaitown for a spin, but recreating the best movie ever (The THI...</li><li><a href="https://x.com/cocktailpeanut/status/1786421948638965870">Tweet from cocktail peanut (@cocktailpeanut)</a>: Introducing AI Town Player  Did you know that the entire AI Town is stored in a single sqlite file via @convex_dev?    I reverse engineered the schema and built a web app that lets anyone REPLAY any A...</li><li><a href="https://sfstandard.com/2024/05/04/mission-control-hacker-house-san-francisco-ai-simulated-party/">We went to San Francisco&#x27;s ‘first-ever AI simulated party’ so you didn’t have to</a>: A trip to the Mission Control hacker house, where a surreal virtual rager was followed by an actual DJ dance party.
</li>
</ul>

</div>
  

---


**AI Stack Devs (Yoko Li) ▷ #[ai-town-dev](https://discord.com/channels/1122748573000409160/1137456826733047908/1235923879835205704)** (61 messages🔥🔥): 

- **Ubuntu and Node Version Woes**: User `utensil_18981` reported issues when attempting to run `convex-local-backend` on Ubuntu 18, ultimately resolving multiple problems by downgrading Node to version 18.17.0 and patching Ubuntu, as described in [this GitHub thread](https://github.com/get-convex/convex-backend/issues/1).
  
- **Pondering Docker for Simplification**: `utensil_18981` expressed frustration with setting up `convex-backend` and `ollama`, mentioning a possible Docker build could simplify the process. `.casado` acknowledged the idea's merit and considered looking into it, potentially over the weekend.

- **Launch of llama-farm for Local LLMs**: `ianmacartney` introduced `llama-farm`, a new project aimed at connecting local machines running Ollama to a cloud backend, offering easy scaling and safety by avoiding public internet exposure. The project can be found on GitHub [here](https://github.com/get-convex/llama-farm-chat).

- **AI Reality TV and AI Town Experiences Teased**: `edgarhnd` gave a sneak peek into an upcoming iteration of AI Reality TV that would allow public interaction with AI Town, hinting at an enhanced and shared experience.

- **Challenges and Solutions for Remote LLM Deployment**: Members discussed the intricacies and obstacles of deploying local language model servers (`ollama`) and connecting them to remote convex backends, with `utensil_95057` ultimately getting it to run by updating to the latest Ollama version and using `ssh` tunneling.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.convex.dev/cli#run-the-convex-dev-server">CLI | Convex Developer Hub</a>: The Convex command-line interface (CLI) is your interface for managing Convex</li><li><a href="https://github.com/get-convex/convex-backend/issues/1">TypeError [ERR_UNKNOWN_FILE_EXTENSION]: Unknown file extension &quot;.ts&quot; for /app/npm-packages/convex/src/cli/index.ts · Issue #1 · get-convex/convex-backend</a>: I ran the steps in the prerequisites then got this when running just run-local-backend Error: Failed to run convex deploy: TypeError [ERR_UNKNOWN_FILE_EXTENSION]: Unknown file extension &quot;.ts&quot...</li><li><a href="https://github.com/get-convex/llama-farm-chat">GitHub - get-convex/llama-farm-chat: Use locally-hosted LLMs to power your cloud-hosted webapp</a>: Use locally-hosted LLMs to power your cloud-hosted webapp - get-convex/llama-farm-chat
</li>
</ul>

</div>
  

---


**AI Stack Devs (Yoko Li) ▷ #[local-ai-stack](https://discord.com/channels/1122748573000409160/1168947823920812125/1236174462051942410)** (1 messages): 

- **Introducing llama-farm for Old Laptops**: A member announced the release of `llama-farm`, which allows running **Ollama** on older laptops to service LLM tasks for public-facing AI applications. This setup scales by running the client on additional machines and doesn’t require a proxy or exposure to public internet, as outlined on [GitHub](https://github.com/get-convex/llama-farm-chat).

**Link mentioned**: <a href="https://github.com/get-convex/llama-farm-chat">GitHub - get-convex/llama-farm-chat: Use locally-hosted LLMs to power your cloud-hosted webapp</a>: Use locally-hosted LLMs to power your cloud-hosted webapp - get-convex/llama-farm-chat

  

---


**AI Stack Devs (Yoko Li) ▷ #[paper-spam](https://discord.com/channels/1122748573000409160/1227492197541220394/)** (1 messages): 

Deforum Daily Papers: Papers will now be sent to <#1227492197541220394>
  

---


**AI Stack Devs (Yoko Li) ▷ #[ai-raspberry-pi](https://discord.com/channels/1122748573000409160/1234912245415280742/)** (1 messages): 

jakekies: ??
  

---



**LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1235934782727127081)** (59 messages🔥🔥): 

- **Exploration of CLIP and T5 Combination**: There's a [discussion](https://old.reddit.com/r/StableDiffusion/comments/1cgr74j/april_30th/l2bxv66/) around using CLIP and T5 encoders for model training; one member mentioned prompt adherence issues with CLIP and is considering T5 only, while another highlighted past success using both encoders.
- **Considerations for Improving Smaller Models**: A focus on enhancing smaller models for practicality was mentioned, with a note on the 400M DeepFloyd and the challenges of preparing 8B models for release.
- **Skeptical Reception of SD3 Strategy**: Comments from Stability AI suggest a gradual release of SD3 models, ranging from smaller first to larger ones, which prompted a discourse on whether this is an efficient approach, especially given the community's anticipation.
- **Potential Use of LLama Embeds in Training**: A dialogue regarding the merit of employing LLama embeds instead of T5 for training, with a link shared to an example bridge called [LaVi-Bridge](https://github.com/ShihaoZhaoZSH/LaVi-Bridge), highlighting modern applications and efficiency.
- **Comparative Progress in Image Gen and LLM Spaces**: Members compared the status of open-source models in the image generation and LLM fields, discussing the adaptation of new models and mentioning a new CogVL marquee.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/ai21labs/Jamba-v0.1">ai21labs/Jamba-v0.1 · Hugging Face</a>: no description found</li><li><a href="https://old.reddit.com/r/StableDiffusion/comments/1ciyzn5/sd3_weights_are_never_going_to_be_released_are/l2dhd6q/">SD3 weights are never going to be released, are they</a>: Gonna be released. Don't have a date. Will be released. If it helps to know, we've shared beta model weights with multiple partner companies...
</li>
</ul>

</div>
  

---


**LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1235953963652743298)** (5 messages): 

- **Real-World vs Synthetic Datasets Query**: A member expressed curiosity about why synthetic datasets are used for experiments instead of standard ones like MNIST, CIFAR, or ImageNet. Concerns were raised regarding the real-world applicability of methods that prioritize interpretability but may not solve practical tasks.

- **Interpretability Demonstrations Discussed**: It was mentioned that the use of synthetic datasets in experiments is to demonstrate the aspect of interpretability in methods being developed.

- **StoryDiffusion Resource Shared**: A link to the [StoryDiffusion website](https://storydiffusion.github.io/) was shared which may contain related information or resources about interpretability in AI.

- **Complexity Over Simplicity in Function Representation**: A member clarified that research sometimes targets approximating complex mathematical representations with functions, as opposed to the "simple" template-like tasks often associated with visual recognition.
  

---



**LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1235950376109477979)** (45 messages🔥): 

- **Database Interfacing with LLMs Sparks Curiosity**: Participants debate whether to convert database data to natural language text versus using an LLM to convert natural language to database queries. Discussions also consider the suitability of graph versus relational databases in this context.
- **Node.JS Conundrums and First Steps with Langchain**: A user seeks assistance with parsing user questions and extracting JSON data in NodeJS, while another encounters an error using FAISS with Langchain but resolves it by upgrading to the latest version.
- **Executing through Code with AI**: Community members exchange insights on executing generated code through an AI agent, with suggestions such as using Open Interpreter and creating custom tools like `CLITOOL`.
- **Langchain Integration Queries**: Users inquire about support for Microsoft Graph within Langchain, using APIs like kappa-bot-langchain at work, and if there is an upload size limit when using Langsmith's free tier.
- **New Developments and Custom Tools Discussed**: Speculation arises regarding changes in ChatGPT's responses post-GPT2 issues, and conversation revolves around creating and sharing custom tools within the Langchain community.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://codepad.site/edit/x084ua3n">TestCode - CodePad</a>: no description found</li><li><a href="https://developers.google.com/analytics/devguides/reporting/data/v1/api-schema">no title found</a>: no description found</li><li><a href="https://learn.microsoft.com/en-us/graph/query-parameters?tabs=http">Use query parameters to customize responses - Microsoft Graph</a>: Microsoft Graph provides optional query parameters that you can use to specify and control the amount of data returned in a response. Includes common parameters.</li><li><a href="https://python.langchain.com/docs/modules/tools/custom_tools/">Defining Custom Tools | 🦜️🔗 LangChain</a>: When constructing your own agent, you will need to provide it with a</li><li><a href="https://api.python.langchain.com/en/latest/memory/langchain.memory.entity.ConversationEntityMemory.html">langchain.memory.entity.ConversationEntityMemory &mdash; 🦜🔗 LangChain 0.1.17</a>: no description found
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1235917336821891092)** (6 messages): 

- **Java Joins the LangChain Family**: **LangChain** is now available for Java developers through [langchain4j](https://github.com/langchain4j/langchain4j), a Java port of the LangChain library, offering an expanded application ecosystem for the AI assistant toolset.

- **Dragonfly Boosts LangChain Caching Capabilities**: **LangChain**'s integration with **Dragonfly**, a high-performance in-memory data store, showcases significant improvements in chatbot context management, as detailed in a new [blog post](https://www.dragonflydb.io/blog/efficient-context-management-in-langchain-with-dragonfly).

- **Decentralizing the Search with Langchain**: A new decentralized search feature is underway, leveraging a network of user-owned indexes to provide powerful search capabilities, all of which is documented in a recent [tweet](https://twitter.com/indexnetwork_/status/1786110169396429093) by the developers.

- **OpenGPTs-platform Unveiled**: An open-source alternative to the GPT Store called [OpenGPTs-platform](https://github.com/OpenGPTs-platform) has been launched, featuring tools like 'retrieval' and 'web_retrieval', and the demo is showcased on [YouTube](https://www.youtube.com/watch?v=yPdIEKb3jWc). The project aims to replicate and expand upon the capabilities of the GPT Store using a modular approach, engaging the community via the [OpenGPTs Discord](https://discord.gg/23aZEjyjp2).

- **Meet everything-ai: The All-in-One AI Assistant**: the rebranded v1.0.0 **everything-ai** local assistant provides a range of tasks from chatting with PDFs and models to summarizing texts and generating images. This multi-container Docker application focuses on versatility and privacy, and its features and quick-start documentation are available on its [GitHub page](https://astrabert.github.io/everything-ai).

- **Beta Testers Invited for Advanced Research Assistant**: A call for beta testers to experience an advanced research platform with access to multiple AI models, including GPT-4 Turbo and Mistral Large, is posted with the promise of a free two-month premium using code `RUBIX` on [Rubiks.ai](https://rubiks.ai/). The offer includes additional models and tools tailored to enhance research capabilities.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://discord.gg/23aZEjyjp2)">Discord | Your Place to Talk and Hang Out</a>: Discord is the easiest way to talk over voice, video, and text. Talk, chat, hang out, and stay close with your friends and communities.</li><li><a href="https://www.dragonflydb.io/blog/efficient-context-management-in-langchain-with-dragonfly">Efficient Context Management in LangChain Chatbots with Dragonfly</a>: Explore efficient context management for LangChain OpenAI chatbots with Dragonfly, enhancing performance and user experience through caching techniques.</li><li><a href="https://rubiks.ai/">Rubik's AI - AI research assistant & Search Engine</a>: no description found</li><li><a href="https://astrabert.github.io/everything-ai">everything-ai</a>: Introducing everything-ai, your fully proficient, AI-powered and local chatbot assistant! 🤖</li><li><a href="https://github.com/AstraBert/everything-ai">GitHub - AstraBert/everything-ai: Introducing everything-ai, your fully proficient, AI-powered and local chatbot assistant! 🤖</a>: Introducing everything-ai, your fully proficient, AI-powered and local chatbot assistant! 🤖 - AstraBert/everything-ai</li><li><a href="https://github.com/langchain4j/langchain4j">GitHub - langchain4j/langchain4j: Java version of LangChain</a>: Java version of LangChain. Contribute to langchain4j/langchain4j development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1236245724430077963)** (2 messages): 

- **RAG Techniques with Llama 3**: A user shared a [YouTube video](https://www.youtube.com/watch?v=vvW2dwvNm2Q) titled "Llama 3 RAG without Vectorstore using SVM," providing insights on *Retrieval-Augmented Generation* using **Llama 3** with a simplicity measurement classifier and eliminating the need for vectorstore.
- **Exploring LangGraph as AgentExecutor**: Another contribution is a [YouTube video](https://www.youtube.com/watch?v=UcD42NA2WoI) that presents a comparison between **LangGraph** and **LangChain Core** components, suggesting advancements in AgentExecutor implementation.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=UcD42NA2WoI">Is LangGraph the Future of AgentExecutor? Comparison Reveals All!</a>: 🚀 Dive into AgentExecutor implementation in today’s video where I showcase a comparison between:LangGraph 🦜🕸️ and LangChain Core 🦜🔗components! 🔧 What&#39;s...</li><li><a href="https://www.youtube.com/watch?v=vvW2dwvNm2Q">Llama 3 RAG without Vectorstore using SVM</a>: We will take a look at how to do rag with llama 3 groq using sim classifier without the need for vectorstorehttps://github.com/githubpradeep/notebooks/blob/m...
</li>
</ul>

</div>
  

---



**tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1236088423656456243)** (17 messages🔥): 

- **Exploring Symbolic Programming in Clojure**: A user mentioned using bounties to familiarize with *tinygrad*, finding symbolic programming easier in **Clojure** than Python.
- **Julia vs. Clojure Debate**: A member argued that **Julia** is superior to *Clojure* for symbolic programming and expressed surprise over its lack of popularity in ML/AI spaces.
- **Seeking Guidance on tinygrad Bugs**: Users are directed to report bugs in *tinygrad* using the GitHub issues tab or the bug reports channel on Discord.
- **Difficulty Understanding tinygrad's UOps Representation**: A member expressed difficulty in understanding *tinygrad's* textual UOps representation and suggested a change to a format closer to llvm IR for readability, sparking a discussion on the formatting and use of phi.
- **Representing UOps in Static Single Assignment (SSA) Form**: The discussion continued with an explanation of UOps as a form of SSA, why the phi is located at the end of a block, and a suggestion for potentially opening a Pull Request (PR) to propose improvements.
  

---


**tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1236700433003581541)** (12 messages🔥): 

- **Tinygrad Thrives on Qualcomm GPUs**: Tinygrad is optimized for Qualcomm GPUs through the use of textures and pixel shaders in calculations, with data management in **image datatype** distributed throughout the codebase, as explained by terafo.
- **Exploring Tinygrad on Qualcomm**: Running Tinygrad on Qualcomm smartphones is feasible without extensive effort unless **DSP support** is required, which significantly increases the complexity.
- **Insights on Tinygrad Symbolic Operations**: A member shared a link to their post that breaks down the symbolic mean computation in tinygrad, providing clarity and insights which may be valuable for others working with or learning tinygrad. See their explanation [here](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/symbolic-mean.md).
- **CPU Operations are Ordered, Not Parallel in Tinygrad**: George Hotz confirmed that tinygrad is **single-threaded**, and no parallel thread operations occur during CPU computations.
- **Questioning Tensor Operations in Tinygrad**: Cappuchinoraro queried about the behavior of the `matmul` function and the implications of transposing tensors within tinygrad's operations.

**Link mentioned**: <a href="https://github.com/mesozoic-egg/tinygrad-notes/blob/main/symbolic-mean.md">tinygrad-notes/symbolic-mean.md at main · mesozoic-egg/tinygrad-notes</a>: Tutorials on tinygrad. Contribute to mesozoic-egg/tinygrad-notes development by creating an account on GitHub.

  

---



**Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1236429992452292679)** (25 messages🔥): 

- **Troubleshooting json_schema Compatibility**: A member encountered issues with `json_schema` not working with **llamafile 0.8.1**; another recommended using the `--unsecure` flag as a potential fix and mentioned a plan to address it in an upcoming release.

- **Seeking Lightweight Models**: A discussion about finding a model that operates on low specs was initiated. A recommendation for **phi 3 mini** was made, while a smaller model, **Rocket-3B**, was suggested for better speed when the phi 3 mini model performed too slowly.

- **Utilizing ollama Cache with llamfile**: A member inquired if **llamafile** can use models stored in the **ollama cache** to prevent redundant downloads, with another confirming it's possible if the GGUF files are supported by llamafile.

- **Integration of llamfile and AutoGPT**: A request for feedback was discussed around a pull request submitted to integrate **llamafile** as an LLM provider with **AutoGPT**. Someone shared a link to instructions ([AutoGPT/llamafile-integration](https://github.com/Mozilla-Ocho/AutoGPT/tree/draft-llamafile-support/autogpts/autogpt/llamafile-integration)) for setting up this configuration, awaiting a response from maintainers before proceeding with further coding efforts.

- **Identifying and Using Correct Local Models**: A user successfully operated **llamafile** with locally cached **.gguf** files after a discussion clarified which files are actual models and which are metadata, demonstrating live troubleshooting and peer support in action.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/Mozilla-Ocho/AutoGPT/tree/draft-llamafile-support/autogpts/autogpt/llamafile-integration">AutoGPT/autogpts/autogpt/llamafile-integration at draft-llamafile-support · Mozilla-Ocho/AutoGPT</a>: AutoGPT is the vision of accessible AI for everyone, to use and to build on. Our mission is to provide the tools, so that you can focus on what matters. - Mozilla-Ocho/AutoGPT</li><li><a href="https://github.com/Significant-Gravitas/AutoGPT/pull/7091">Draft llamafile support by k8si · Pull Request #7091 · Significant-Gravitas/AutoGPT</a>: Background  This draft PR is a step toward enabling the use of local models in AutoGPT by adding llamafile as an LLM provider. Related issues:  #6336 #6947  Changes 🏗️  For full documentation of th.....
</li>
</ul>

</div>
  

---



**DiscoResearch ▷ #[mixtral_implementation](https://discord.com/channels/1178995845727785010/1182759434326396998/1236622793127493643)** (7 messages): 

- **Mixtral Transformers Bug Affects Performance**: It has been highlighted that there were bugs in the **mixtral transformers** implementation, which led to poor performance in past mixtral finetunes. Critical issues and further discussion about this problem were shared via links to [Twitter](https://twitter.com/kalomaze/status/1786869036946522256), [Gist](https://gist.github.com/kalomaze/661b79095fdd91df8a84802f7cb6f26a), and a [Pull Request on GitHub](https://github.com/huggingface/transformers/pull/30658).

- **Uncertainty Around Mixtral's Scope of Issues**: Members questioned whether the mixtral issue was limited to *training* or if it also affected *generation*. There was no clear consensus on the matter, accentuating a need for further clarification.

- **Problem-Solving in Progress**: An ongoing conversation mentioned by a member, pointing to a discussion with another Discord user, suggested that work was being done to pinpoint and address the issues with mixtral. However, specific details of the conversation were not provided.

- **Bug Resolution Seemingly in Limbo**: A member expressed humor at the situation, indicating a belief that there were known issues with mixtral all along. This interjection suggests a perception among users that issues were anticipated.

- **Pull Request Rejection Adds to Mixtral Confusion**: The mentioned pull request for fixing the mixtral bug was *closed/rejected*, adding another layer of uncertainty to the resolution status of these issues. The implications of this rejection on the mixtral implementation were not discussed further.
  

---


**DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1235956851133386872)** (3 messages): 

- **Performance Dip in Quantized LLaMA-3**: A Reddit post [discussed the impact of quantization on LLaMA-3](https://www.reddit.com/r/LocalLLaMA/comments/1cetn9z/quantization_seems_to_hurt_the_quality_of_llama_3/), suggesting that performance degradation is more prominent in LLaMA-3 compared to LLaMA-2. A [study on low-bit quantization of LLaMA-3](https://arxiv.org/abs/2404.14047) may provide additional insights into the challenges of LLM compression.
- **Meta Misses on Chinchilla Lessons?**: A member pointed out that Meta's approach to scale LLaMA despite the lessons from *Chinchilla* could be why information loss is more significant with precision reduction in the LLaMA-3 model.
- **Fix Patches in the Works**: A GitHub pull request offers possible fixes for the quantization issues observed in LLaMA-3, including additional statistics and documentation ([PR #6936](https://github.com/ggerganov/llama.cpp/pull/6936#issuecomment-2083214112)), as well as a conversation surrounding pre-tokenization BPE processing ([Issue #7088](https://github.com/ggerganov/llama.cpp/issues/7088#issuecomment-2094933215)).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.reddit.com/r/LocalLLaMA/comments/1cetn9z/quantization_seems_to_hurt_the_quality_of_llama_3/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://arxiv.org/abs/2404.14047">How Good Are Low-bit Quantized LLaMA3 Models? An Empirical Study</a>: Meta&#39;s LLaMA family has become one of the most powerful open-source Large Language Model (LLM) series. Notably, LLaMA3 models have recently been released and achieve impressive performance across ...</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/6936#issuecomment-2083214112">perplexity: more statistics, added documentation by JohannesGaessler · Pull Request #6936 · ggerganov/llama.cpp</a>: I have seen subjective reports about quantization being more harmful for LLaMA 3 than for LLaMA 2. I decided to investigate this and have to this end added more statistics (and documentation) to pe...</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/7088#issuecomment-2094933215)">The convert-hf-to-gguf-update.py seems doesn&#39;t work. · Issue #7088 · ggerganov/llama.cpp</a>: Ubuntu 20.04, cudatoolkit12.2 GPU: Nvidia A100 24G RAM 10G(available) When I use the &#39;convert-hf-to-gguf-update.py&#39; in llama.cpp to convert ‘hf’ to &#39;gguf&#39;, neither does it report any e...
</li>
</ul>

</div>
  

---


**DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/1237016176429629501)** (3 messages): 

- **Current Model in Use Revealed**: Discussion in the channel revealed that **8x22b Mistral** is the current model being used by a member for their tasks. No further details about performance or application specifics were provided.
  

---



**Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1236328226222116965)** (3 messages): 

- **Behind ElevenLabs' Convincing AI Voices**: An article in [The Atlantic](https://www.theatlantic.com/technology/archive/2024/05/elevenlabs-ai-voice-cloning-deepfakes/678288/) details how a start-up named **ElevenLabs** has developed some of the most convincing AI voice cloning technology. The author shared a personal experience with the service, using it to clone their own voice.

- **Paywalls: A Modern Nuisance**: A member expressed frustration with encountering a paywall, indicating an inability to access the full content of The Atlantic article on **ElevenLabs**.

- **ElevenLabs: A Wild Existence**: The same member remarked on the existence of **ElevenLabs**, describing the start-up as "wild" for its capability to create convincing AI-generated voices.

**Link mentioned**: <a href="https://www.theatlantic.com/technology/archive/2024/05/elevenlabs-ai-voice-cloning-deepfakes/678288/">ElevenLabs Is Building an Army of Voice Clones</a>: A tiny start-up has made some of the most convincing AI voices. Are its creators ready for the chaos they’re unleashing?

  

---


**Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1236002486280781904)** (2 messages): 

- **Paper Skips RewardBench Scores**: A newly published [paper on arXiv](https://arxiv.org/abs/2405.01535) overlooked reporting *RewardBench* scores because the results were unfavorable, prompting a bit of academic shade with a <:facepalm:1207415956020797521> emoji.
- **Prometheus 2 LM Introduced for Bias-Free Evaluations**: The paper introduces **Prometheus 2**, an open-source evaluator language model that claims to closely align with human and **GPT-4 judgments**, and it addresses issues of transparency, controllability, and affordability that affect proprietary LMs.
- **Desire to Implement and Test Prometheus 2**: One member expressed eagerness to implement **Prometheus 2** in order to challenge and verify the paper's claims through a practical demonstration.

**Link mentioned**: <a href="https://arxiv.org/abs/2405.01535">Prometheus 2: An Open Source Language Model Specialized in Evaluating Other Language Models</a>: Proprietary LMs such as GPT-4 are often employed to assess the quality of responses from various LMs. However, concerns including transparency, controllability, and affordability strongly motivate the...

  

---


**Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1236040686512377937)** (2 messages): 

- **Unstoppable Success Raises Eyebrows**: A member expressed amazement and a touch of concern with the phrasing *"he can't keep getting away with this."*
- **Uncertainty Around John's Response**: Another member reflected on a conversation with *"john"*, highlighting a non-committal answer to a proposal with the remark, *"dang so that's why john said only maybe to me lol."*
  

---


**Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/1235969401660903585)** (4 messages): 

- **Pondering the Unknown in Classical RL**: A member sparked curiosity by asking if there's research regarding a particular aspect in classical RL, hinting at a potential knowledge gap or area for future inquiry.
- **Value Function: A Possible Key in Different Approaches**: Another member suggested an exploration into connections between **PPO value function**, and **DPO's credit assignment**, implying it could lead to interesting insights within reinforcement learning strategies. 
- **Value Function's Significance in Planning**: Follow-up discussion emphasized the value function’s significance, particularly within the context of planning rather than classical reinforcement learning, underscoring its critical role.
  

---



**LLM Perf Enthusiasts AI ▷ #[prompting](https://discord.com/channels/1168579740391710851/1179271229593624677/1237119509165248593)** (7 messages): 

- **Exploring Anthropic's Prompt Generator**: A new **prompt generator tool** was mentioned being available in the **Anthropic console**.
- **Polite Rephrasing Results**: A member tested the tool asking it to *rephrase a sentence in more polite language* and shared the outcome was *not bad*.
- **Decoding the System Prompt**: Work is being done to extract the system prompt from the tool, with **k-shot examples** being a significant part of it, including a notable *Socratic math tutor* example.
- **Extracted Data Incomplete**: The member attempting the extraction reports the prompt is so extensive that it is cut off mid-way, especially during the long math tutor example.
- **Promise of Sharing the Full Prompt**: The member committed to sharing the full system prompt here once successfully extracted and compiled.
  

---



**Skunkworks AI ▷ #[datasets](https://discord.com/channels/1131084849432768614/1131669182124138616/1236350458663141386)** (1 messages): 

- **In Search of Fabricated Data**: A member expressed the need for a **dataset filled with fake information** for the purpose of experimenting with fine-tuning techniques on models like **Llama 3** and **Phi3**. They indicated that even completely fake data would be acceptable for their research.
  

---


**Skunkworks AI ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/1235948611292893263)** (2 messages): 

- **Fast Compute Grants Available**: A member offers **fast compute grants** for inspiring Skunkworks AI projects, expressing eagerness to support innovation. The support offer can be found in a [tweet](https://twitter.com/PrimeIntellect/status/1786386588726960167).
- **AI Video Resource Shared**: A link to a YouTube video related to artificial intelligence has been shared, serving as a potential resource or point of interest to members of the community. The video can be viewed [here](https://www.youtube.com/watch?v=vvW2dwvNm2Q).
  

---



**Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/1236429191092899941)** (3 messages): 

- **LLM Proves Handy for Error Summaries**: A member shared an effective method for summarizing errors using LLM; they provided an example `conda activate` command piped through LLM. It's suggested this could be included in the [LLM README](https://github.com/simonw/llm/blob/main/README.md).

- **Bash Function Utilizes LLM for Error Evaluation**: A new `llm-err` bash function is proposed to help evaluate errors by piping command outputs directly to LLM. The function takes a command as an argument and uses LLM to specify the cause of any error encountered.
  

---



**Cohere ▷ #[collab-opps](https://discord.com/channels/954421988141711382/1218409745380147320/1236060159990566942)** (2 messages): 

- **Shoutout for Austin, TX Community**: A member sends a friendly hello to anyone located in **Austin, TX**.
- **French AI Startup in Funding Phase**: **Vivien** from France introduces **Finexov** ([Finexov](https://www.finexov.com/)), an AI platform streamlining the process of identifying **R&D** funding opportunities and application generation. The platform has been launched with partnerships in place and backing from the **Founder Institute** ([FI.co](https://fi.co/)).
- **Seeking a CTO Co-Founder**: Vivien is seeking a **CTO co-founder** with a deep background in **ML** with the ambition to build and lead a team. The potential CTO should be Europe or Middle East-based, with French language skills being a bonus, and prepared for intensive work including fundraising efforts.
- **Meeting Opportunity in Dubai**: There's an opportunity to meet in **Dubai** at the beginning of June, where Vivien invites interested parties to reach out for a potential catch-up.

**Link mentioned**: <a href="https://fi.co/">Founder Institute: World’s largest pre-seed startup accelerator.</a>: no description found

  

---



**AI21 Labs (Jamba) ▷ #[jamba](https://discord.com/channels/874538902696914944/1222916247063232553/1236029165447413770)** (2 messages): 

- **Exploring New Heights**: AI21 Labs staff stated, *“We are still exploring, but we can go much higher”* regarding some aspect of their technology, inviting community members to discuss their use cases and thoughts in direct messages.
  

---



**Alignment Lab AI ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/1235947724063375370)** (1 messages): 

- **Fast Compute Grants Available**: A member shared a [Twitter post](https://twitter.com/PrimeIntellect/status/1786386588726960167) announcing **fast compute grants** for those in need. The tweet seems to be a call for applications or nominations for receiving compute resources.
  

