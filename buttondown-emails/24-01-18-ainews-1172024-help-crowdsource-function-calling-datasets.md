---
id: c809dfbf-4c0c-4562-8f89-82b4d58e553a
title: '1/17/2024: Help crowdsource function calling datasets'
date: '2024-01-18T21:20:01.765780Z'
original_slug: ainews-1172024-help-crowdsource-function-calling
description: >-
  **Skunkworks** is compiling function calling datasets, crucial for turning
  operations into callable functions. **LM Studio** updated its FAQ confirming
  it is **closed-source**, free for personal use, and does not collect user
  data, with a new beta release addressing memory and generation consistency
  issues and hinting at upcoming **2-bit quantization** support. For gaming
  mods, models like **Dolphin 2.7 Mixtral 8x7B**, **MegaDolphin**, and **Dolphin
  2.6 Mistral 7B DPO** with **Q4_K_M** quantization were recommended.
  Discussions highlighted that single powerful GPUs often outperform multi-GPU
  setups due to bottlenecks, with older GPUs like the Tesla P40 being
  cost-effective for LLMs. Microsoft's **AutoGen Studio** was introduced but has
  limitations due to **API fees** for open-source models. Linux users are
  advised to use **llama.cpp** over LM Studio due to lack of headless mode.
  Additional projects and tools like **LLMFarm** for iOS and various Hugging
  Face repositories were also mentioned.
companies:
  - skunkworks
  - lmstudio
  - microsoft
  - huggingface
models:
  - mistral-7b
  - mixtral-8x7b
  - dolphin-2.7-mixtral-8x7b
  - dolphin-2.6-mistral-7b-dpo
topics:
  - function-calling
  - quantization
  - model-performance
  - gpu-optimization
  - memory-optimization
  - model-selection
  - closed-source
  - local-inference
  - linux-server
  - api
people:
  - yagilb
  - heyitsyorkie
---


<!-- buttondown-editor-mode: plaintext -->> We checked **19** guilds, **287** channels, and **3277** messages for you. Estimated reading time saved (at 200wpm): **363 minutes**.


Skunkworks is working on collating function calling datasets - key to turning everything into functions!

 ![image.png](https://assets.buttondown.email/images/f8f12ccf-f13e-4b38-b425-574b6e9414fa.png?w=960&fit=max) 

It's also important to familiarize with underlying data formats and sources:

![image.png](https://assets.buttondown.email/images/24fe29d0-6a37-492b-a369-79164b71846d.png?w=960&fit=max) 



What other datasets are out there for tuning function calls? Can we synthesize some?

---

**Table of Contents**

[TOC]

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord Summary

- **LM Studio's Latest Updates and Compatibility**: LM Studio's FAQ was updated, clarifying its **closed-source** status, perpetual freeness for personal use, and non-invasive data handling, with the updated FAQ available [here](https://rentry.org/LMSTudioFAQ). The new LM Studio beta [release](https://lmstudio.ai/beta-releases.html) includes fixes for memory warnings, generation consistency, with `@yagilb` hinting at anticipated **2-bit quantization** support shown in [this pull request](https://github.com/ggerganov/llama.cpp/pull/4773).
  
- **Model Selection Advice for Gaming**: For a Skyrim mod, **Dolphin 2.7 Mixtral 8x7B** or **MegaDolphin** was suggested, and **Dolphin 2.6 Mistral 7B DPO** with **Q4_K_M** quantization was chosen for performance. The Ferret model clarification indicated it's a **Mistral 7B finetune**, not a vision model, with references to its [GitHub repository](https://github.com/apple/ml-ferret) and [Hugging Face page](https://huggingface.co/TheBloke/Ferret_7B-GGUF).
  
- **Performance Bottlenecks and Hardware Discussions**: In discussions of model execution, it was noted that a single, powerful GPU often outperforms multi-GPU setups due to potential bottlenecks, and when configuring hardware for LLMs, matching the model size to available GPU memory is key, with older server-class GPUs like the Tesla P40 being a cost-efficient upgrade choice.
  
- **Exploration of New AI Tools and Requests**: Microsoft's **AutoGen Studio** was introduced as a new tool for large language model applications, yet issues were reported and its full usefulness seems gated by the need for **API fees** for open-source models, prompting discussions on integration channels for other projects.
  
- **Use Cases for Linux Server Users**: Users on Linux were directed towards utilizing `llama.cpp` rather than LM Studio, considering there is no headless mode available and `llama.cpp` offers a more suitable backend for server use.
  

Additional links shared provided insights into a variety of projects, including [Microsoft's AutoGen Studio](https://github.com/microsoft/autogen/tree/main/samples/apps/autogen-studio), LM Studio's alternative for iOS [LLMFarm](https://github.com/guinmoon/LLMFarm), and various Hugging Face model repositories. However, sparse details or a single message was insufficient to establish context for summarization regarding GitHub links for [NexusRaven-V2](https://github.com/nexusflowai/NexusRaven-V2) and the mention of memory challenges with local models.

**LM Studio Channel Summaries**

### ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/) (62 messages🔥🔥):

- **LM Studio FAQ Refreshed**: `@heyitsyorkie` updated the LM Studio FAQ, outlining key features like it being closed source, always free for personal use, with no data collection from users. The FAQ can be found [here](https://rentry.org/LMSTudioFAQ).
  
- **LM Studio Closed Source Clarification**: `@heyitsyorkie` responded to `@esraa_45467`'s query about accessing LM Studio code by stating it's closed source and cannot be viewed.
  
- **No Headless Mode for LM Studio**: `@heyitsyorkie` mentioned that LMStudio must be running to use the local inference server (lis) as there is no headless mode available so it cannot be operated solely through a script.
  
- **Support for macOS and iOS Environments**: `@heyitsyorkie` confirmed to `@pierre_hugo_` that LM Studio is not supported on the MacBook Air 2018 with an Intel CPU and advised `@dagbs` on the feasibility of using jailbreak to run it headlessly, whereas `@technot80` shared information about an alternative app [LLMFarm](https://github.com/guinmoon/LLMFarm) for iOS.
  
- **Language Limitations and Conversational Oddities**: It’s mentioned that LM Studio primarily supports English, following the influx of Spanish-speaking users from a recent video. A discussion was observed about the humorous translations when switching between languages in the models, particularly a translation from Spanish to Chinese.
  

**Links mentioned**:

- [Despacio Despacito GIF - Despacio Despacito Luisfonsi - Discover & Share GIFs](https://tenor.com/view/despacio-despacito-luisfonsi-gif-8379347): Click to view the GIF
  
- [The unofficial LMStudio FAQ!](https://rentry.org/LMSTudioFAQ): Welcome to the unofficial LMStudio FAQ. Here you will find answers to the most commonly asked questions that we get on the LMStudio Discord. (This FAQ is community managed). LMStudio is a free closed...
  
- [GitHub - guinmoon/LLMFarm: llama and other large language models on iOS and MacOS offline using GGML library.](https://github.com/guinmoon/LLMFarm): llama and other large language models on iOS and MacOS offline using GGML library. - GitHub - guinmoon/LLMFarm: llama and other large language models on iOS and MacOS offline using GGML library.
  

### ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/) (80 messages🔥🔥):

- **Model Recommendation Inquiry for Skyrim Mod**: `@gamerred` was looking for a suitable model to use with a Skyrim mod, and `@dagbs` advised that anything in the Dolphin series would do, recommending **Dolphin 2.7 Mixtral 8x7B** or **MegaDolphin** for those who have the capability to run larger models. `@gamerred` decided on using **Dolphin 2.6 Mistral 7B DPO** with quantization **Q4_K_M** for faster responses in-game.
  
- **Confusion over Ferret's Functionality**: `@ahmd3ssam` encountered the error "Vision model is not loaded. Cannot process images" when attempting to use Ferret; `@heyitsyorkie` clarified that Ferret is a Mistral 7B finetune and not a vision model, pointing to its [GitHub repository](https://github.com/apple/ml-ferret) and its [Hugging Face page](https://huggingface.co/TheBloke/Ferret_7B-GGUF).
  
- **Optimizing AI Performance for In-Game Use**: `@dagbs` suggested using Dolphin models with lower quantization for better in-game performance due to VRAM limitations, further noting that as conversations lengthen, responses might slow down if not using a rolling message history. `@fabguy` added that disabling GPU in LMStudio might improve the experience as the game and the AI compete for GPU resources.
  
- **Prompt Formatting Advice**: `@.ben.com` sought the correct prompt format for laserxtral leading to a brief discussion, with `@dagbs` confirming that the "ChatML" preset is appropriate for all Dolphin-based models.
  
- **Machine Requirements for LLM**: In a query about the best model to fit on a 7900 XTX, `@heyitsyorkie` replied to `@_anarche_` that up to a 33-billion parameter model can be accommodated, such as Llama 1 models **Guanaco** and **WizardVicuna**.
  

**Links mentioned**:

- [cognitivecomputations/laserxtral-GGUF · Hugging Face](https://huggingface.co/cognitivecomputations/laserxtral-GGUF)
  
- [TheBloke/Ferret_7B-GGUF · Hugging Face](https://huggingface.co/TheBloke/Ferret_7B-GGUF)
  
- [TheBloke/dolphin-2.6-mistral-7B-dpo-GGUF · Hugging Face](https://huggingface.co/TheBloke/dolphin-2.6-mistral-7B-dpo-GGUF)
  
- [Herika - The ChatGPT Companion](https://www.nexusmods.com/skyrimspecialedition/mods/89931): 'Herika - The ChatGPT Companion' is a revolutionary mod that aims to integrate Skyrim with Artificial Intelligence technology. It specifically adds a follower, Herika, whose responses and interactions
  
- [GitHub - apple/ml-ferret](https://github.com/apple/ml-ferret): Contribute to apple/ml-ferret development by creating an account on GitHub.
  

### ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/) (5 messages):

- **New LM Studio Beta Released**: `@yagilb` announced a [new LM Studio beta release](https://lmstudio.ai/beta-releases.html) which includes bug fixes such as warning for potential memory issues, a fix for erratic generations after multiple regenerations, and server response consistency. They note the removal of ggml format support and temporary disables for certain features.
  
- **Inquiry About 2-bit Quant Support**: `@logandark` inquired about 2-bit quantization support, linking to a [GitHub pull request](https://github.com/ggerganov/llama.cpp/pull/4773) discussing the feature.
  
- **Awaiting Next Beta for New Features**: In response to a question about the addition of 2-bit quant, `@yagilb` confirmed that new betas are expected to be released later in the day and may include the discussed feature.
  
- **Eager Users Anticipating Updates**: `@logandark` expressed thanks and eagerness following the update on the upcoming beta from `@yagilb`.
  

**Links mentioned**:

- [LM Studio Beta Releases](https://lmstudio.ai/beta-releases.html)
  
- [SOTA 2-bit quants by ikawrakow · Pull Request #4773 · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/pull/4773): TL;DR This PR adds new "true" 2-bit quantization (but due to being implemented within the block-wise quantization approach of ggml/llama.cpp we end up using 2.0625 bpw, see below for more de...
  

### ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/) (81 messages🔥🔥):

- **Single GPU Tops Multi-GPU Setups for Speed**: `@pefortin` shared insights that using a **single GPU** for model execution is generally faster than distributing the workload across multiple GPUs. They mentioned that the **slowest component**, like a less powerful GPU in their multi-GPU setup, can bottleneck performance.
  
- **Quantization and Hardware Capabilities Discussion**: `@juansinisterra`, looking for advice on quantization levels suitable for their hardware, received suggestions to match the model size to available **GPU memory** from `@heyitsyorkie` and to explore using both **GPU and CPU** for model execution from `@fabguy`.
  
- **Cloud as an Alternative for Model Execution**: Users discussed options for running models in the cloud, with `@dagbs` warning about legal implications and trust concerns when it comes to uncensored content and recommending considering cost-effective, older **server-class GPUs** like the *Tesla P40* for personal hardware upgrades.
  
- **Navigating the GPU Market for AI Applications**: Discussions arose over the cost-effectiveness of various GPUs like the *7900xtx* compared to 3090s, with `@heyitsyorkie` and `.ben.com` discussing different GPUs' VRAM and suitability for tasks like **Large Language Models** (LLMs).
  
- **Configuring Models based on Specific Hardware**: `@heyitsyorkie` advised `@lex05` on the kind of models they can run with their hardware setup, suggesting **7b Q_4 models** and reading model cards to determine appropriate configurations for their **RTX 4060** with 8GB VRAM.
  

**Links mentioned**:

- [HuggingChat](https://huggingface.co/chat/)
  
- [ggml/docs/gguf.md at master · ggerganov/ggml](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md): Tensor library for machine learning. Contribute to ggerganov/ggml development by creating an account on GitHub.
  
- [MSI Radeon RX 7900 XTX GAMING TRIO CLASSIC 24GB Graphics Card | Ebuyer.com](https://www.ebuyer.com/1615563-msi-radeon-rx-7900-xtx-gaming-trio-classic-24gb-graphics-card-rx-7900-xtx-gaming-trio-classic-24g)
  

### ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/) (8 messages🔥):

- **Ubuntu 22.04 Compatibility Confirmed**: User `@ephemeraldust` inquired about running the application on Ubuntu 22.04 server, to which `@heyitsyorkie` responded that **it's compiled on 22.04** so **it should work fine**. However, they were also informed that there is **no headless mode or cli options** available, and the app must remain open for use.
  
- **Seeking CLI Accessibility**: `@ephemeraldust` was looking to access the application via command-line interface (CLI), which led `@heyitsyorkie` to suggest looking into `llama.cpp` for a more suitable solution.
  
- **Time-Saving Tip Appreciated**: `@ephemeraldust` thanked `@heyitsyorkie` for the advice regarding `llama.cpp`, acknowledging the potential time saved.
  
- **Clarity on LM Studio and llama.cpp Uses**: `@heyitsyorkie` clarified that LM Studio serves as a user-friendly front-end for Mac/Windows users, while `llama.cpp` is the backend that Linux server users should utilize.
  

### ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/) (6 messages):

- **Microsoft's AutoGen Studio Unveiled**: `@senecalouck` shared a link to [Microsoft's AutoGen Studio on GitHub](https://github.com/microsoft/autogen/tree/main/samples/apps/autogen-studio), highlighting its aim to enable next-gen large language model applications.
  
- **LM Studio Issues with AutoGen Studio Highlighted**: `@dagbs` tried out **AutoGen Studio** and reported having issues unrelated to LM Studio, also mentioning that others are experiencing functional difficulties with the tool.
  
- **Functionality Hinges on API Fees**: `@senecalouck` noted that the usefulness of **AutoGen Studio** is currently limited for open-source models and tooling without paying API fees, due to a lack of good function calling capability.
  
- **Request for CrewAI Integration Channel**: `@senecalouck` requested the creation of a CrewAI integration channel, indicating they have a project that could be of interest to the community.
  
- **Open Interprite Channel Suggestion**: `@dagbs` expressed surprise for the lack of an Open Interprite channel given its mention of LM Studio in their base configurations, hinting at the possible relevance for the community.
  

**Links mentioned**:

[autogen/samples/apps/autogen-studio at main · microsoft/autogen](https://github.com/microsoft/autogen/tree/main/samples/apps/autogen-studio): Enable Next-Gen Large Language Model Applications. Join our Discord: [https://discord.gg/pAbnFJrkgZ](https://discord.gg/pAbnFJrkgZ) - microsoft/autogen

### ▷ #[langchain](https://discord.com/channels/1110598183144399058/1167546793656062063/) (1 messages):

sublimatorniq: [https://github.com/nexusflowai/NexusRaven-V2](https://github.com/nexusflowai/NexusRaven-V2)

### ▷ #[memgpt](https://discord.com/channels/1110598183144399058/1170104578889502750/) (1 messages):

pefortin: Yeah, local models struggle on how and when to use memory.

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord Summary

- **Confusion Cleared on ACL Submission**: `ludgerpaehler` sought clarification regarding the **ACL submission deadline**, it is indeed necessary to send submissions to the ARR OpenReview portal by the 15th of February, adhering to the [ACL's outlined process](https://www.aclweb.org/portal/content/submission-dates-and-process-eaclnaacl-and-acl-2024).
  
- **Evaluation Harness Key Error Reported**: An issue was filed by `alexrs_` on the evaluation-harness and [reported an error](https://github.com/EleutherAI/lm-evaluation-harness/issues/1302) regarding a KeyError when using certain metrics from huggingface/evaluate.
  
- **Mamba and ZOH Discretization Debate**: There was a noteworthy discussion concerning the use of Zero-Order Hold (ZOH) discretization within the Mamba model, with insights regarding its relevance to linear state space models and ODE solutions.
  
- **Call for Reproducible Builds Amid Python Upgrade**: During an update to Python, `@catboyslimmer` encountered test failures and difficulties with Apex build while attempting to [modernize gpt-neox](https://github.com/EleutherAI/gpt-neox/pull/1122#issuecomment-1895304911). They emphasized the urgent need for a reproducible build process.
  
- **BaseLM Refactor and Implementational Directive**: Recent refactoring removed `BaseLM`, necessitating users to implement functions like `_loglikelihood_tokens`. However, plans to reintroduce similar features were referenced in a [Pull Request](https://github.com/EleutherAI/lm-evaluation-harness/pull/1279), and the team discussed potential solutions for boilerplate code.
  

**Eleuther Channel Summaries**

### ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/) (119 messages🔥🔥):

- **Confusion about ACL Submission Process**: `ludgerpaehler` needed clarification on ACL deadline process for submissions. They inquired whether the ACL submission must be sent to ARR OpenReview portal by the 15th of February according to the [ACL submission dates and process](https://www.aclweb.org/portal/content/submission-dates-and-process-eaclnaacl-and-acl-2024).
  
- **Issue with Evaluation Harness and Evaluate Metrics**: `alexrs_` encountered problems with the evaluation-harness and [reported an issue](https://github.com/EleutherAI/lm-evaluation-harness/issues/1302) where certain metrics from huggingface/evaluate are causing a KeyError.
  
- **Exploring Memmapped Datasets**: `lucaslingle` sought insights for implementing memmapped datasets and referenced the Pythia codebase's mention of 5gb-sized memmapped files. `hailey_schoelkopf` clarified that this size limitation was due to Huggingface upload constraints, but larger sizes should work for Megatron once combined.
  
- **Request for Alternatives to** `wandb`: `.the_alt_man` requested suggestions for a drop-in replacement for `wandb` with integrated hyperparameter tuning and plotting, leading to discussions about the architectural choices behind monitoring and tune scheduling.
  
- **Discussion on Hypernetworks versus MoE Layers**: `Hawk` started a conversation asking if anyone had tried hypernet layers as an alternative to MoE layers; `zphang` noted that hypernetwork technology might not be advanced enough yet for this application.
  

**Links mentioned**:

- [kjj0/cifar10-multirun-logits · Datasets at Hugging Face](https://huggingface.co/datasets/kjj0/cifar10-multirun-logits)
  
- [Accelerating PyTorch with CUDA Graphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/): Today, we are pleased to announce a new advanced CUDA feature, CUDA Graphs, has been brought to PyTorch. Modern DL frameworks have complicated software stacks that incur significant overheads associat...
  
- [mamba_small_bench/cifar_10.py at main · apapiu/mamba_small_bench](https://github.com/apapiu/mamba_small_bench/blob/main/cifar_10.py): Trying out the Mamba architecture on small examples (cifar-10, shakespeare char level etc.) - apapiu/mamba_small_bench
  
- [KeyError on some metrics from huggingface/evaluate · Issue #1302 · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/issues/1302): Context I am currently utilizing the lm-evaluation-harness in conjunction with the metrics provided by huggingface/evaluate. Specifically, I am using bertscore. This metric returns a dictionary wit...
  

### ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/) (69 messages🔥🔥):

- **Quality vs Diversity in Generative Models**: `@ai_waifu` mentioned that there's a tradeoff between quality and diversity that is not well captured by negative log-likelihood (NLL) metrics. Although GANs may perform poorly from an NLL perspective, they can still produce visually appealing images without heavy penalties for mode dropping.
  
- **Exploring 3D CAD Generation with GFlownets**: `@johnryan465` inquired about literature on GFlownet application to 3D CAD model generation, finding none. `@carsonpoole` suggested that large synthetic datasets of CAD images paired with actual geometry could be useful.
  
- **Introducing Contrastive Preference Optimization**: A study shared by `@xylthixlm` showcases a new training method for Large Language Models (LLMs) called Contrastive Preference Optimization (CPO) which focuses on training models to avoid generating adequate but not perfect translations, in response to supervised fine-tuning's shortcomings.
  
- **Mamba and ZOH Discretization Discussion**: `@michaelmelons` sparked a conversation questioning why the Mamba model employs Zero-Order Hold (ZOH) discretization for its matrices. `@useewhynot` and `@mrgonao` offered insights, relating to linear state space models and the solution of ODEs regarding A's discretization.
  
- **Tokenization and Byte-Level Encoding Exploration**: Discussion led by `@carsonpoole` and `@rallio.` examined the impacts of resetting embedding weights in models and the debate on whether to use Llama tokenizers or raw bytes for inputs. This topic evolved into a broader conversation about tokenizer inefficiencies, particularly in handling proper nouns and noisy data, with users `@catboyslimmer` and `@fern.bear` highlighting the understudied nature of tokenization and its distributional consequences.
  

**Links mentioned**:

- [ZeroShape](https://zixuanh.com/projects/zeroshape.html)
  
- [Contrastive Preference Optimization: Pushing the Boundaries of LLM Performance in Machine Translation](http://arxiv.org/abs/2401.08417): Moderate-sized large language models (LLMs) -- those with 7B or 13B parameters -- exhibit promising machine translation (MT) performance. However, even the top-performing 13B LLM-based translation mod...
  
- [GitHub - google-deepmind/alphageometry](https://github.com/google-deepmind/alphageometry): Contribute to google-deepmind/alphageometry development by creating an account on GitHub.
  

### ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/) (1 messages):

- **Discovering Developmental Interpretability**: User `@David_mcsharry` shared an intriguing update, mentioning their discovery of **developmental interpretability** which appears to be relevant to the topics of interest in the **interpretability-general** channel.
  

### ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/) (7 messages):

- **Seeking Clarity on BaseLM Removal**: User `@daniellepintz` inquired about the removal of the useful `BaseLM` class from the EleutherAI repository, which included methods like `loglikelihood` and `loglikelihood_rolling`. It was found in the code at [BaseLM Reference](https://github.com/EleutherAI/lm-evaluation-harness/blob/3ccea2b2854dd3cc9ff5ef1772e33de21168c305/lm_eval/base.py#L121).
  
- **Refactoring Behind BaseLM Removal**: `@hailey_schoelkopf` clarified that the removal of `BaseLM` was due to a refactoring process designed to improve batch generation and data-parallel evaluation. They indicated that there is a plan to re-add similar functionality, hinted at by [Pull Request #1279](https://github.com/EleutherAI/lm-evaluation-harness/pull/1279).
  
- **Implementation Requirements Persist**: In response to `@daniellepintz` noticing that users must implement `_loglikelihood_tokens` and `loglikelihood_rolling`, `@stellaathena` admitted that when creating a novel model API, such implementation steps are largely unavoidable.
  
- **Boilerplate Abstraction Still Possible**: `@hailey_schoelkopf` acknowledged that while some boilerplate might be abstracted away, functions like `loglikelihood_tokens` and `generate_until` may entail some unavoidable custom coding. However, reusing or subclassing HFLM could potentially be a solution for users.
  
- **Potential Issue with HF Datasets Version**: `@hailey_schoelkopf` suggested pinning the HF datasets version to 2.15 for the time being, noting that versions 2.16 and above might be causing issues for users due to changes in dataset loading scripting.
  

**Links mentioned**:

- [lm-evaluation-harness/lm_eval/base.py at 3ccea2b2854dd3cc9ff5ef1772e33de21168c305 · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/3ccea2b2854dd3cc9ff5ef1772e33de21168c305/lm_eval/base.py#L121): A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness
  
- [Loglikelihood refactor attempt 2 using template lm by anjor · Pull Request #1279 · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/pull/1279): Replaces #1215
  

### ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/) (9 messages🔥):

- **Potential Python Version Upgrade Issues**: `@catboyslimmer` raised a potential issue about the [Python version update pull request #1122](https://github.com/EleutherAI/gpt-neox/pull/1122#issuecomment-1895304911) on **EleutherAI/gpt-neox** failing some tests locally, noting that these failures could predate his changes and detailing his plan to test building in Docker and backporting into a poetry file.
  
- **Docker Build Success Amid Testing Needs**: `@catboyslimmer` mentioned that the build is successful in Docker but admitted that additional testing is likely needed, expressing a lack of certainty about potential issues caused by the changes.
  
- **Complications With Apex Build**: `@catboyslimmer` encountered difficulties with building **Apex**, considering extracting a fused kernel directly from Apex to resolve the issue without delving deeper into the underlying problem with Apex.
  
- **Magic in Running Scripts Multiple Times**: In response to `@catboyslimmer`'s building issues, `@stellaathena` suggested repeating the build script execution a few times, which can sometimes resolve the problem, though `@catboyslimmer` doubts this will work due to versioning and dependency issues.
  
- **Horrified by the Lack of Reproducibility**: `@catboyslimmer` expressed feelings of horror regarding their current non-reproducible build process and is compelled to set up a more reliable build system as soon as possible.
  

**Links mentioned**:

[Python version update by segyges · Pull Request #1122 · EleutherAI/gpt-neox](https://github.com/EleutherAI/gpt-neox/pull/1122#issuecomment-1895304911): Don't know if this is ready or not; in my local testing it fails some of the pytest tests, but it's plausible to likely it was doing so before. Bumps image to ubuntu 22.04 and uses the system ...

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord Summary

- **GPT-4-Turbo Troubles Ahead**: `.beowulfbr` voiced dissatisfaction with **GPT-4-Turbo**, highlighting nonsensical code output and a decline in performance. Meanwhile, the **ChatGPT versus GPT-4-Turbo** debate stirred up, with comparisons being made on iteration efficiency and bug presence.
  
- **Seeking Simplified Chat Models**: `@gabriel_syme` and `@murchiston` want chat models to skip the chit-chat and get things done, while `@giftedgummybee` proposes instructing AI to "use the top 100 most common English words" for clearer communication.
  
- **Anticipating Code Model Contenders**: The Discord buzzed with talks of **Stable Code 3B**'s release, [Stable Code 3B](https://stability.ai/news/stable-code-2024-llm-code-completion-release), the unveiling of **InternLM2** at [Hugging Face](https://huggingface.co/internlm/internlm2-chat-20b), and the potential debut of **DeciCoder-6B**, as discussed by `@osanseviero`.
  
- **Innovations in Text Recognition**: Traditional OCR is highlighted as currently more reliable than multimodal models for tasks like multilingual invoice analysis, as evidenced by the suggestion to use [Tesseract OCR](https://github.com/tesseract-ocr/tesseract). This approach is posed against AI models like GPT-4 Vision and multimodal alternatives.
  
- **Advancements in AI Geometry**: Updates on **AlphaGeometry** from DeepMind drew a mixed response from the community, with a mention of [DeepMind's research](https://deepmind.google/discover/blog/alphageometry-an-olympiad-level-ai-system-for-geometry/) evoking both humor and technical interest related to LLM integration and mathematical reasoning capabilities.
  

**Nous Research AI Channel Summaries**

### ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/) (27 messages🔥):

- **GPT-4-Turbo Under Fire**: `.beowulfbr` shared frustration with **GPT-4-Turbo**, mentioning it produced complete nonsensical code when prompted for implementation help. The user reported a noticeable decline in quality and errors, even after correcting the AI.
  
- **ChatGPT versus GPT-4-Turbo Debate**: User `giftedgummybee` argued that the API is better compared to ChatGPT, while `.beowulfbr` criticized ChatGPT for bugs and for taking twice as many iterations to arrive at the correct outcome in comparison to the API version.
  
- **The Persistence of LLM Errors**: `giftedgummybee` and `night_w0lf` discussed the tendency for LLMs to repeat mistakes and revert to a "stupid LLM mode" if not properly guided, possibly referencing a need for more thorough prompts or "waluigi-ing the model."
  
- **TTS Software Discussions**: `everyoneisgross` raised a question about offline TTS (Text-to-Speech) utilities being used in scripts, expressing challenges with Silero, while `tofhunterrr` and `leontello` suggested alternatives like `say` command on Mac and open-source TTS Bark, respectively.
  
- **Coqui TTS as a Recommended Solution**: `leontello` recommended checking out **Coqui TTS**, a tool that allows trying out various TTS alternatives with just a few lines of code.
  

**Links mentioned**:

- [Nous Research Deep Learning GIF - Nous Research Research Nous - Discover & Share GIFs](https://tenor.com/view/nous-research-research-nous-deep-learning-ai-gif-14158112487472873681): Click to view the GIF
  
- [Deep Learning Yann Lecun GIF - Deep Learning Yann LeCun LeCun - Discover & Share GIFs](https://tenor.com/view/deep-learning-yann-lecun-lecun-godfather-ai-gif-2302123676916500142): Click to view the GIF
  

### ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/) (15 messages🔥):

- **Chat Models Get Chatty**: `@gabriel_syme` and `@murchiston` express frustration over chat models being verbose and engaging in unnecessary dialogue instead of executing tasks promptly; they desire a straightforward "make it so" approach without the added waffle.
  
- **Simplification Tactic for Chat Models**: `@giftedgummybee` suggests enforcing simplicity by telling the AI to "use the top 100 most common English words" as a way to combat overly complicated responses from chat models.
  
- **InternLM2-Chat-20B Release**: `@euclaise` shares a link to Hugging Face repository detailing **InternLM2**, an open-source chat model with a 200K context window, heralded for stellar performance across various tasks, including reasoning and instruction following.
  
- **Call for Digital Art and AI Symposium Proposals**: `@everyoneisgross` highlights the call for proposals for the **Rising Algorithms Symposium** in Wellington, asking for contributions that explore the intersection between art and AI.
  
- **Vector-based Random Matrix Adaptation (VeRA)**: `@mister_poodle` shares an arXiv paper introducing **VeRA**, a technique that reduces the number of trainable parameters in finetuning large language models without compromising performance, as `@charlesmartin14` retains skepticism on its effectiveness without evidence of results.
  

**Links mentioned**:

- [VeRA: Vector-based Random Matrix Adaptation](https://arxiv.org/abs/2310.11454): Low-rank adapation (LoRA) is a popular method that reduces the number of trainable parameters when finetuning large language models, but still faces acute storage challenges when scaling to even large...
  
- [2024 ADA Symposium – Call for Proposals](https://ada.net.nz/events/2024-ada-symposium-call-for-proposals/): <p>Aotearoa Digital Arts Network Symposium<br /> Rising Algorithms: Navigate, Automate, Dream<br /> 24 &#8211;  26 May 2024<br /> Te Whanganui-a-Tara Wellington</p> &...
  
- [ReFT: Reasoning with Reinforced Fine-Tuning](https://arxiv.org/abs/2401.08967): One way to enhance the reasoning capability of Large Language Models (LLMs) is to conduct Supervised Fine-Tuning (SFT) using Chain-of-Thought (CoT) annotations. This approach does not show sufficientl...
  
- [internlm/internlm2-chat-20b · Hugging Face](https://huggingface.co/internlm/internlm2-chat-20b)
  
- [Solution Suicide GIF - Solution Suicide Rick And Morty - Discover & Share GIFs](https://tenor.com/view/solution-suicide-rick-and-morty-gif-10761762): Click to view the GIF
  

### ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/) (110 messages🔥🔥):

- **Anticipation of Stable Code 3B's release**: Enthusiasm and skepticism were expressed regarding the announcement of the new [Stable Code 3B](https://stability.ai/news/stable-code-2024-llm-code-completion-release), a state-of-the-art model designed for code completion, with `@.beowulfbr` calling it "disappointing" due to being behind a paywall.
- **Confusion around new models**: Discussion centered on upcoming models like StableLM Code, with users such as `@gabriel_syme` and `@giftedgummybee` trying to distill the information from [teasers on Twitter](https://fxtwitter.com/osanseviero/status/1747356927040815397), questioning whether they have been released or not.
- **Debate on coding model benchmarks**: Members like `@night_w0lf` trust specific evaluation platforms like [EvalPlus](https://evalplus.github.io/leaderboard.html) for judging the performance of code models, while others like `@teknium` and `@antonb5162` discuss the validity of HumaEval scores and the reliability of various models.
- **Interest in new coding models**: `@osanseviero` highlight the release of [DeciCoder-6B](https://fxtwitter.com/deci_ai/status/1747620747156111766?s=20), attracting attention with its performance claims and open-source availability.
- **Crowd-sourced OSS model funding**: `@carsonpoole` expresses interest in sponsoring open-source software (OSS) models related to mistral, mixtral, or phi, seeking collaborations with the community.


**Links mentioned**:

- [EvalPlus Leaderboard](https://evalplus.github.io/leaderboard.html)
  
- [Tweet from Google DeepMind (@GoogleDeepMind)](https://fxtwitter.com/GoogleDeepMind/status/1747651817461125352?s=20): Introducing AlphaGeometry: an AI system that solves Olympiad geometry problems at a level approaching a human gold-medalist. 📐 It was trained solely on synthetic data and marks a breakthrough for AI...
  
- [Tweet from Wavecoder (@TeamCodeLLM_AI)](https://fxtwitter.com/TeamCodeLLM_AI/status/1747652471714144702): We are in the process of preparing for open-source related matters. Please stay tuned. Once everything is ready, we will announce the latest updates through this account.
  
- [Tweet from Deci AI (@deci_ai)](https://fxtwitter.com/deci_ai/status/1747620747156111766?s=20): We’re back and excited to announce two new models: DeciCoder-6B and DeciDiffuion 2.0! 🙌 Here’s the 101: DeciCoder-6B 📋 ✅ A multi-language, codeLLM with support for 8 programming languages. ✅ Rel...
  
- [Cat Cats GIF - Cat Cats Cat eating - Discover & Share GIFs](https://tenor.com/view/cat-cats-cat-eating-eating-eating-cat-gif-8459125914348971806): Click to view the GIF
  
- [Tweet from Blaze (Balázs Galambosi) (@gblazex)](https://fxtwitter.com/gblazex/status/1747587267378475317): New SOTA coding model by @MSFTResearch 81.7 HumanEval & only 6.7B! (vs 85.4 GPT4) Only nerfed version of dataset is going open source I think. But the techniques are laid out in the paper. Nerfed vs ...
  
- [Stable Code 3B: Coding on the Edge — Stability AI](https://stability.ai/news/stable-code-2024-llm-code-completion-release): Stable Code, an upgrade from Stable Code Alpha 3B, specializes in code completion and outperforms predecessors in efficiency and multi-language support. It is compatible with standard laptops, includi...
  
- [Meet](https://meet.google.com/ytq-miod-kjh): Real-time meetings by Google. Using your browser, share your video, desktop, and presentations with teammates and customers.
  
- [Cat Cats GIF - Cat Cats Cat meme - Discover & Share GIFs](https://tenor.com/view/cat-cats-cat-meme-meme-meme-cat-gif-14470917232397934693): Click to view the GIF
  
- [Giga Gigacat GIF - Giga Gigacat Cat - Discover & Share GIFs](https://tenor.com/view/giga-gigacat-cat-mewing-mogging-gif-12429734670640119345): Click to view the GIF
  
- [Tweet from Div Garg (@DivGarg9)](https://fxtwitter.com/DivGarg9/status/1747683043446579416): We just solved the long-horizon planning & execution issue with Agents 🤯! Excited to announce that @MultiON_AI can now take actions well over 500+ steps without loosing context & cross-operate on 10...
  
- [Tweet from Omar Sanseviero (@osanseviero)](https://fxtwitter.com/osanseviero/status/1747356927040815397): Spoiler alert: this might be one of the most exciting weeks for code LLMs since Code Llama
  
- [FastChat/fastchat/llm_judge/README.md at main · lm-sys/FastChat](https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/README.md): An open platform for training, serving, and evaluating large language models. Release repo for Vicuna and Chatbot Arena. - lm-sys/FastChat
  
- [Tweet from Andriy Burkov (@burkov)](https://x.com/burkov/status/1747413167792181494?s=20): If you really want to do something useful in AI, instead of training another tiny llama, pick up this project [https://hazyresearch.stanford.edu/blog/2024-01-11-m2-bert-retrieval](https://hazyresearch.stanford.edu/blog/2024-01-11-m2-bert-retrieval) and train a 1B-paramet...
  
- [GitHub - evalplus/evalplus: EvalPlus for rigourous evaluation of LLM-synthesized code](https://github.com/evalplus/evalplus): EvalPlus for rigourous evaluation of LLM-synthesized code - GitHub - evalplus/evalplus: EvalPlus for rigourous evaluation of LLM-synthesized code
  
- [GitHub - draganjovanovich/sharegpt-vim-editor: sharegpt jsonl vim editor](https://github.com/draganjovanovich/sharegpt-vim-editor): sharegpt jsonl vim editor. Contribute to draganjovanovich/sharegpt-vim-editor development by creating an account on GitHub.
  

### ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/) (40 messages🔥):

- **In Search of Speed and Efficiency with AI Models**: `@realsedlyf` inquired about the performance of **OpenHermes2.5 gptq** with vllm compared to using transformers, wondering if it's faster.
  
- **Benchmarking Code Generation Models**: `@leontello` posed a question regarding trusted code generation benchmarks and leaderboards, while `@night_w0lf` pointed to a recent post in the general channel which apparently contains relevant information, but no specific URL was mentioned.
  
- **Multimodals vs. Traditional OCR for Multilingual Invoice Analysis**: `@.beowulfbr` suggested their friend to try [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) as an alternative to multimodal models like **Qwen-VL**, indicating OCR's superiority in accuracy, especially for invoices in various languages.
  
- **OCR Prevails Over Multimodal Models for Text Recognition**: `@bernaferrari` and `@n8programs` discussed the limitations of LLMs in image recognition, suggesting that while GPT-4 Vision shows promise, traditional OCR systems are still more effective at tasks like reading car plates.
  
- **DeepMind's AlphaGeometry Sparks a Mix of Interest and Humor**: `@bernaferrari` shared [DeepMind's latest research](https://deepmind.google/discover/blog/alphageometry-an-olympiad-level-ai-system-for-geometry/) on **AlphaGeometry**, with community reactions ranging from teknium joking about their own math skills to `@mr.userbox020` comparing the system to a combination of LLM and code interpreter architectures.
  

**Links mentioned**:

- [AlphaGeometry: An Olympiad-level AI system for geometry](https://deepmind.google/discover/blog/alphageometry-an-olympiad-level-ai-system-for-geometry/): Our AI system surpasses the state-of-the-art approach for geometry problems, advancing AI reasoning in mathematics
  
- [GitHub - tesseract-ocr/tesseract: Tesseract Open Source OCR Engine (main repository)](https://github.com/tesseract-ocr/tesseract): Tesseract Open Source OCR Engine (main repository) - GitHub - tesseract-ocr/tesseract: Tesseract Open Source OCR Engine (main repository)
  

---

## [Mistral](https://discord.com/channels/1144547040454508606) Discord Summary

- **Launchpad for LLM APIs Comparison**: A new website introduced by `@_micah_h` offers a comparison between different hosting APIs for models like **Mistral 7B Instruct** and **Mixtral 8x7B Instruct**, with a focus on technical metrics evaluation. The platforms for [Mistral 7B Instruct](https://artificialanalysis.ai/models/mistral-7b-instruct) and [Mixtral 8x7B Instruct](https://artificialanalysis.ai/models/mixtral-8x7b-instruct) were shared, alongside their [Twitter page](https://twitter.com/ArtificialAnlys) for updates.
  
- **Mistral Models' Verbosity & Token Questions**: Users in the **models** channel discussed challenges with Mistral's verbose responses and proper use of **bos_token** in chat templates. Integrating the tokens correctly doesn't seem to influence model scores significantly; however, verbosity issues are recognized and being addressed.
  
- **Fine-Tuning Facets and Snags**: The **finetuning** channel saw exchanges on topics like using `--vocabtype bpe` for tokenizers without `tokenizer.model`, formatting datasets for instruct model fine-tuning, and challenges with fine-tuned models not retaining previous task knowledge.
  
- **Deep Chat and Mistral Performance Optimization**: **Deep Chat** allows running models like **Mistral** directly in the browser using local resources, with its open-source project available on [GitHub](https://github.com/OvidijusParsiunas/deep-chat). Meanwhile, **FluxNinja Aperture** was introduced in the **showcase** channel as a solution for concurrency scheduling, detailed in their [blog post](https://blog.fluxninja.com/blog/concurrency-scheduling-in-mistral-ai).
  
- **Mistral-7B Instruct Deployment Angel**: The rollout of the Mistral-7B Instruct model was broadcasted in **la-plateforme** channel, directing users to stay tuned to artificialanalysis.ai group, particularly after a tweet update. The analysis of the model can be found at [ArtificialAnalysis.ai](https://artificialanalysis.ai/models/mistral-7b-instruct).
  

**Mistral Channel Summaries**

### ▷ #[general](https://discord.com/channels/1144547040454508606/1144547040928481394/) (76 messages🔥🔥):

- **New Comparative Website for LLM API Providers**: `@_micah_h` launched a website to compare different hosting APIs for models like Mistral 7B Instruct and Mixtral 8x7B Instruct, providing a [platform for Mistral 7B](https://artificialanalysis.ai/models/mistral-7b-instruct) and a [platform for Mixtral 8x7B](https://artificialanalysis.ai/models/mixtral-8x7b-instruct). The [Twitter page](https://twitter.com/ArtificialAnlys) was also shared for updates.
  
- **Perplexity AI's Pricing Source and Limitations Discussed**: Discussion around Perplexity AI's input token pricing was addressed by `@_micah_h`, pointing to a [changelog note revealing the removal of 13B pricing](https://docs.perplexity.ai/changelog/new-model-mixtral-8x7b-instruct), while also agreeing with `@blueridanus` that the limitation to 4k is somewhat unfair.
  
- **Local Deployment and Free Versions of Mistral Discussed**: `@rozonline` inquired about free versions of Mistral, and `@blueridanus` suggested deploying locally or trying Perplexity AI's playground for some free credits.
  
- **Adding Third-Party PHP Client to Mistral's Documentation**: `@gbourdin` requested that a PHP client library for Mistral API, available on [GitHub](https://github.com/partITech/php-mistral), be mentioned in Mistral's [client documentation page](https://docs.mistral.ai/platform/client/).
  
- **Public Resources on Mistral AI's Privacy and Data Processing**: `@ethux` provided `@khalifa007` with links to Mistral AI's [Privacy Policy](https://mistral.ai/privacy-policy/) and [Data Processing Agreement](https://mistral.ai/data-processing-agreement/) for information regarding personal data handling.
  

**Links mentioned**:

- [Chat with Open Large Language Models](https://chat.lmsys.org)
  
- [LLM in a flash: Efficient Large Language Model Inference with Limited Memory](https://arxiv.org/abs/2312.11514): Large language models (LLMs) are central to modern natural language processing, delivering exceptional performance in various tasks. However, their substantial computational and memory requirements pr...
  
- [Privacy Policy](https://mistral.ai/privacy-policy/): Frontier AI in your hands
  
- [Mistral AI | Open-weight models](https://mistral.ai/): Frontier AI in your hands
  
- [Open LLM Leaderboard - a Hugging Face Space by HuggingFaceH4](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
  
- [Data Processing Agreement](https://mistral.ai/data-processing-agreement/): Frontier AI in your hands
  
- [LMSys Chatbot Arena Leaderboard - a Hugging Face Space by lmsys](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard)
  
- [Mixtral of Experts](https://arxiv.org/abs/2401.04088): We introduce Mixtral 8x7B, a Sparse Mixture of Experts (SMoE) language model. Mixtral has the same architecture as Mistral 7B, with the difference that each layer is composed of 8 feedforward blocks (...
  
- [Pricing](https://docs.perplexity.ai/docs/pricing)
  
- [Self-Consuming Generative Models Go MAD](https://arxiv.org/abs/2307.01850): Seismic advances in generative AI algorithms for imagery, text, and other data types has led to the temptation to use synthetic data to train next-generation models. Repeating this process creates an ...
  
- [The Great Web AI Enshitification | DearDiary](https://ker2x.github.io/DearDiary/web-enshitification.html)
  
- [New Model: mixtral-8x7b-instruct](https://docs.perplexity.ai/changelog/new-model-mixtral-8x7b-instruct)
  
- [Client code | Mistral AI Large Language Models](https://docs.mistral.ai/platform/client/): We provide client codes in both Python and Javascript.
  
- [GitHub - partITech/php-mistral: MistralAi php client](https://github.com/partITech/php-mistral): MistralAi php client. Contribute to partITech/php-mistral development by creating an account on GitHub.
  
- [Mistral 7B - Host Analysis | ArtificialAnalysis.ai](https://artificialanalysis.ai/models/mistral-7b-instruct): Analysis of Mistral 7B Instruct across metrics including quality, latency, throughput, price and others.
  
- [Mixtral 8x7B - Host Analysis | ArtificialAnalysis.ai](https://artificialanalysis.ai/models/mixtral-8x7b-instruct): Analysis of Mixtral 8x7B Instruct across metrics including quality, latency, throughput, price and others.
  

### ▷ #[models](https://discord.com/channels/1144547040454508606/1154028112124846150/) (71 messages🔥🔥):

- **Mistral's Verbose Responses Puzzle Users**: User `@rabdullin` reported that **Mistral**'s hosted models are failing to adhere to the brevity of instructions provided in few-shot prompts, contrasting with the behavior of local models like **Mistral 7B Instruct v1**. In response, `@sophiamyang` shared that verbosity in Mistral models is a known issue, and the team is actively working on a fix.
  
- **Formatting Fiasco**: Confusion arose regarding the proper use of **bos_token** in the chat template, with `@rabdullin` initially positing that **Mistral** API might tokenize incorrectly due to his template placing **bos_tokens** within a loop. However, `@sophiamyang` clarified that the Mistral models expect the **bos_token** once at the beginning, leading `@rabdullin` to adjust his template and find that while verbosity remained, changing the token placement had no significant impact on model scores.
  
- **Benchmark Blues**: `@rabdullin` is eager to benchmark **Mistral** models against closed-source benchmarks from products and services, mentioning discrepancies in rankings between versions and unexpected verbosity affecting scores. `@sophiamyang` prompted for examples that can be shared and investigated by the Mistral team.
  
- **Template Troubles**: `@rabdullin` questioned the implications of his template's incorrect format, which led to a back-and-forth over the role of <s>and </s> tokens in prompt design. A reference to the GitHub Llama_2 tokenizer seemed to mirror `@rabdullin`'s structure, but the matter remains unresolved whether the format influenced API behavior.
  
- **Model Misidentification**: There was some confusion about the existence of a **Mistral 13B** model, sparked by `@dfilipp9` following an external hardware guide listing an alleged **MistralMakise-Merged-13B-GGUF** model. `@rabdullin` pointed out that only **Mistral 7B** or **8x7B** models exist, and they are available on HuggingFace.
  

**Links mentioned**:

- [tokenizer_config.json · mistralai/Mistral-7B-Instruct-v0.2 at main](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/blob/main/tokenizer_config.json#L42)
  
- [mistralai/Mixtral-8x7B-Instruct-v0.1 · Hugging Face](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1#instruction-format)
  
- [tokenizer_config.json · mistralai/Mixtral-8x7B-Instruct-v0.1 at main](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/blob/main/tokenizer_config.json#L42)
  
- [Mistral LLM: All Versions & Hardware Requirements – Hardware Corner](https://www.hardware-corner.net/llm-database/Mistral/)
  
- [2-Zylinder-Kompressor Twister 3800 D | AGRE | ZGONC](https://www.zgonc.at/at/pd/2-Zylinder-Kompressor-Twister-3600-D_p_19489%22)): Kompressoren Bei ZGONC kaufen! 2-Zylinder-Kompressor Twister 3800 D, AGRE Spannung in Volt: 400, Leistung in Watt: 3.000,... - Raunz´ned - kauf!
  

### ▷ #[finetuning](https://discord.com/channels/1144547040454508606/1156994197287612508/) (33 messages🔥):

- **Tokenizer Troubles Resolved**: User `@ethux` confirmed that for AquilaChat model which lacks `tokenizer.model`, the workaround is to use `--vocabtype bpe` when running `convert.py`. This advice helped `@distro1546` successfully quantize their fine-tuned AquilaChat model.
  
- **Chit Chat Isn't Child's Play**: `@distro1546` faced issues with fine-tuned Mistral not performing in an "assistant" manner and learned from `@ethux` that the normal model is unsuitable for chat. They consider fine-tuning the instruct version instead.
  
- **Transforming Transformers**: `@distro1546` also reported continuous text generation until manual interruption and was seeking advice on how to address it, additionally querying about merging a LoRA model with Mistral Instruct using the base model.
  
- **Format Finesse Anyone?**: Clarification sought by `@distro1546` regarding dataset formatting for fine-tuning instruct models, `@denisjannot` advised correct format is `[INST]question[/INST]answer</s>` without an initial `<s>` token.
  
- **Fine-Tuning Frustration**: `@kam414` sought assistance for an issue where their fine-tuned model fails to retain knowledge of previous tasks after learning a new one, despite having a small dataset of 200 rows which led to unsatisfactory loss metrics.
  

**Links mentioned**:

- [dfurman/Mistral-7B-Instruct-v0.2 · Hugging Face](https://huggingface.co/dfurman/Mistral-7B-Instruct-v0.2)
  
- [Add mistral's new 7B-instruct-v0.2 · Issue #1499 · jmorganca/ollama](https://github.com/jmorganca/ollama/issues/1499): Along with many releases, Mistral vastly improved their existing 7B model with a version named v0.2. It has 32k context instead of 8k and better benchmark scores: [https://x.com/dchaplot/status/1734](https://x.com/dchaplot/status/1734)...
  
- [TheBloke/AquilaChat2-34B-AWQ · FileNotFoundError - the tokenizer.model file could not be found](https://huggingface.co/TheBloke/AquilaChat2-34B-AWQ/discussions/1)
  
- [Could not find tokenizer.model in llama2 · Issue #3256 · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/issues/3256): When I ran this command: python convert.py \\ llama2-summarizer-id-2/final_merged_checkpoint \\ --outtype f16 \\ --outfile llama2-summarizer-id-2/final_merged_checkpoint/llama2-summarizer-id-2.gguf.fp...
  
- [How to Finetune Mistral AI 7B LLM with Hugging Face AutoTrain - KDnuggets](https://www.kdnuggets.com/how-to-finetune-mistral-ai-7b-llm-with-hugging-face-autotrain): Learn how to fine-tune the state-of-the-art LLM.
  

### ▷ #[showcase](https://discord.com/channels/1144547040454508606/1157223559278628885/) (6 messages):

- **Deep Chat Integrates LLMs Directly on Browsers**: User `@ovi8773` shared their open-source project named **Deep Chat**, which allows running LLMs like **Mistral** on the browser without needing a server. The project [Deep Chat GitHub Repo](https://github.com/OvidijusParsiunas/deep-chat) and [Playground](https://deepchat.dev/playground) were shared for users to experiment with the web component.
  
- **Genuine Excitement for In-Browser LLM**: `@gbourdin` expressed excitement about the potential of running LLMs on the browser, as introduced by `@ovi8773`.
  
- **In-Browser Acceleration Clarification**: `@Valdis` asked for clarification on how "Deep Chat" works, leading `@ovi8773` to confirm that the LLM runs inference on the local machine via the browser, using web assembly and hardware acceleration.
  
- **Concurrency Challenges with Mistral AI Highlighted**: User `@tuscan_ninja` wrote about a blog post discussing the current concurrency and GPU limitation challenges with the **Mistral 7B model**. They introduced **FluxNinja Aperture** as a solution offering concurrency scheduling and request prioritization to improve performance ([FluxNinja Blog Post](https://blog.fluxninja.com/blog/concurrency-scheduling-in-mistral-ai)).
  
- **User Seeking Moderation Role Information**: User `@tominix356` mentioned `@707162732578734181` to inquire about the moderation role, with no further context provided.
  

**Links mentioned**:

- [Balancing Cost and Efficiency in Mistral with Concurrency Scheduling | FluxNinja Aperture](https://blog.fluxninja.com/blog/concurrency-scheduling-in-mistral-ai): FluxNinja Aperture's Concurrency Scheduling feature efficiently reduces infrastructure costs for operating Mistral, while simultaneously ensuring optimal performance and user experience.
  
- [GitHub - OvidijusParsiunas/deep-chat: Fully customizable AI chatbot component for your website](https://github.com/OvidijusParsiunas/deep-chat): Fully customizable AI chatbot component for your website - GitHub - OvidijusParsiunas/deep-chat: Fully customizable AI chatbot component for your website
  
- [Playground | Deep Chat](https://deepchat.dev/playground): Deep Chat Playground
  

### ▷ #[la-plateforme](https://discord.com/channels/1144547040454508606/1184444810279522374/) (5 messages):

- **A Handy Python Tip for Model Outputs**: User `@rabdullin` shared a handy tip for handling model responses: Use `model_dump` on the `response` object to export it, and pass `mode="json"` if you want to save it as JSON.
  
- **Anyscale Performance Benchmarked**: User `@freqai` commented on the performance of Anyscale, noting that they *rarely see anywhere near those values*, with an average closer to 2 for Anyscale.
  
- **Clarification on Shared Graph**: `@sublimatorniq` clarified that the earlier graph they shared was not their own and they should have provided the source.
  
- **Launch of Mistral-7B Instruct**: `@sublimatorniq` announced the launch of the Mistral-7B Instruct model and encouraged following the artificialanalysis.ai group on Twitter for future updates. The source of the info was given as a post in another channel with the ID `#1144547040928481394`.
  

**Links mentioned**:

[Mistral 7B - Host Analysis | ArtificialAnalysis.ai](https://artificialanalysis.ai/models/mistral-7b-instruct): Analysis of Mistral 7B Instruct across metrics including quality, latency, throughput, price and others.

---

## [HuggingFace Discord](https://discord.com/channels/879548962464493619) Discord Summary

- **Rapid Response to Repository Alerts**: An immediate fix was applied to a **repository metadata issue** with users confirming a resolution. This alleviated earlier concerns about error messages stating "*400 metadata is required*" on repositories.
  
- **BERT and NER Get a Fine-Tuning Facelift**: Solutions were shared for correctly labeling BERT's configuration and detailed guidance on NER dataset creation. Users discussed the handling of `#` tokens and the significance of proper labeling in BERT's `config.json`.
  
- **Harnessing Deep Learning for Diverse Applications**: From AR hit-testing resources and automated transcriptions to AI-powered note-takers for school and ad recommendations across modalities, discussions included innovative applications for existing models. Concerns about timeout issues with large language models like `Deci/DeciLM-7B` and `phi-2` were raised, with suggestions to run Python as admin and test with smaller models like `gpt2`.
  
- **The Evolution of Model Serving and Deployment**: An invite to a [ML Model Serving webinar](https://lu.ma/l8hx98bm?utm_source=discord) was extended, covering the deployment of ML and LLMs. Users explored multi-stack app deployment on HuggingFace Spaces, chaining models for improved performance, and deploying local LLM assistants attuned to privacy.
  
- **New Frontiers in Fine-Tuning and Dataset Sharing**: Members shared resources, including a new Multimodal Dataset for Visual Question Answering and developments in ontology learning using LLMs. Attention was directed toward fine-tuning scripts for models such as `train_sd2x.py`, with one user adding untested LoRA support for Stable Diffusion 2.x. Projects such as [SimpleTuner](https://github.com/bghira/SimpleTuner/) were mentioned for their contributions towards model perfection.
  

**HuggingFace Discord Channel Summaries**

### ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/) (84 messages🔥🔥):

- **Quick Fix for Repository Metadata Issue**: `@lunarflu` acknowledged an issue with "400 metadata is required" on repos and worked on a remedy. `@jo_pmt_79880` confirmed that it was resolved quickly after a humorously noting initial panic.
  
- **BERT Label Fixes and Fine-Tuning**: `@Cubie | Tom` offered a solution for `@redopan706` on using correct labels instead of `LABEL_0, LABEL_1` in BERT's `config.json`. `@stroggoz` also provided detailed guidance on data structuring for NER dataset creation and handling of `#` tokens in outputs for `@redopan706`.
  
- **Deployment and Utilization of Multi-Model Architectures**: Users discussed how to deploy multi-stack apps on HuggingFace Spaces, with `@thethe_realghost` seeking assistance. And `@vishyouluck` asked for model chaining advice and shared experiences with model performance and the use of a "refiner" for image outputs.
  
- **Model Recommendations for Various Tasks**: `@zmkeeney` inquired about models for text-to-text tasks with `@doctorpangloss` providing a comprehensive response touching on suitability of models for market research, website development, brands creation, and consulting firm support.
  
- **AI-Powered Note Taking Inquiry**: `@blakeskoepka` asked about AI note takers for school which spawned a succinct suggestion from `@hoangt12345` about utilizing class recordings and automated transcriptions.
  

**Links mentioned**:

- [Join Pareto.AI's Screen Recording Team](https://paretoai.typeform.com/skilled-ai-us): We're looking for skilled content creators who are Windows users to screen-record activities they've gained proficiency or mastery over for AI training.
  
- [config.json · yy07/bert-base-japanese-v3-wrime-sentiment at main](https://huggingface.co/yy07/bert-base-japanese-v3-wrime-sentiment/blob/main/config.json#L11-L14)
  
- [config.json · yy07/bert-base-japanese-v3-wrime-sentiment at main](https://huggingface.co/yy07/bert-base-japanese-v3-wrime-sentiment/blob/main/config.json#L17-L20)
  

### ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/) (4 messages):

- **Invitation to ML Model Serving Webinar**: `@kizzy_kay` shared an invite for a webinar titled *"A Whirlwind Tour of ML Model Serving Strategies (Including LLMs)"* scheduled for **January 25, 10 am PST**, featuring Ramon Perez from Seldon. The [event](https://lu.ma/l8hx98bm?utm_source=discord) is free with required registration and will cover deployment strategies for traditional ML and LLMs.
  
- **Beginner's Query on Learning ML**: `@mastermindfill` asked for guidance on starting with machine learning and mentioned they have begun watching the ML series by **3blue1brown**. No further recommendations or responses were provided within the given message history.
  

**Links mentioned**:

[Webinar "A Whirlwind Tour of ML Model Serving Strategies (Including LLMs)" · Luma](https://lu.ma/l8hx98bm?utm_source=discord): Data Phoenix team invites you all to our upcoming webinar that’s going to take place on January 25th, 10 am PST. Topic: A Whirlwind Tour of ML Model Serving Strategies (Including...

### ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/) (8 messages🔥):

- **Local LLM Assistant for Sarcasm and Privacy**: `@tea3200` highlighted a blog post about setting up a local LLM assistant [Local LLM Assistant](https://johnthenerd.com/blog/local-llm-assistant/) that functions without cloud services, with a focus on privacy and flexibility to add new capabilities.
  
- **VQA Dataset now on Hugging Face**: `@andysingal` has contributed a Multimodal Dataset for visual question answering to the Hugging Face community, initially found from Mateusz Malinowski and Mario Fritz. You can access the dataset here: [Andyrasika/VQA-Dataset](https://huggingface.co/datasets/Andyrasika/VQA-Dataset).
  
- **Ollama: API for AI Interaction**: `@andysingal` shared a GitHub repository for Ollama API, which allows developers to deploy a RESTful API server to interact with Ollama and Stable Diffusion. [Ollama API on GitHub](https://github.com/Dublit-Development/ollama-api).
  
- **AlphaGeometry: AI for Olympiad-level Geometry**: `@tea3200` brought attention to DeepMind's new AI system, AlphaGeometry, that excels at complex geometry problems. DeepMind has published a research blog post about it [here](https://deepmind.google/discover/blog/alphageometry-an-olympiad-level-ai-system-for-geometry/).
  
- **Ontology Learning with LLMs**: `@davidello19` recommended an arXiv paper about using LLMs for ontology learning [LLMs for Ontology Learning](https://arxiv.org/pdf/2307.16648) and also shared a more accessible article on the same topic [Integrating Ontologies with LLMs](https://ai.plainenglish.io/integrating-ontologies-with-large-language-models-for-decision-making-bb1c600ce5a3).
  
- **Improving AI with Juggernaut XL**: `@rxience` mentioned enhanced performance using Juggernaut XL combined with good prompting in their implemented space on Hugging Face, which can be found [here](https://huggingface.co/spaces/r-neuschulz/h94-IP-Adapter-FaceID-SDXL).
  

**Links mentioned**:

- [H94 IP Adapter FaceID SDXL - a Hugging Face Space by r-neuschulz](https://huggingface.co/spaces/r-neuschulz/h94-IP-Adapter-FaceID-SDXL)
  
- [AlphaGeometry: An Olympiad-level AI system for geometry](https://deepmind.google/discover/blog/alphageometry-an-olympiad-level-ai-system-for-geometry/): Our AI system surpasses the state-of-the-art approach for geometry problems, advancing AI reasoning in mathematics
  
- [Building a fully local LLM voice assistant to control my smart home](https://johnthenerd.com/blog/local-llm-assistant/): I’ve had my days with Siri and Google Assistant. While they have the ability to control your devices, they cannot be customized and inherently rely on cloud services. In hopes of learning someth...
  
- [GitHub - Dublit-Development/ollama-api: Deploy a RESTful API Server to interact with Ollama and Stable Diffusion](https://github.com/Dublit-Development/ollama-api): Deploy a RESTful API Server to interact with Ollama and Stable Diffusion - GitHub - Dublit-Development/ollama-api: Deploy a RESTful API Server to interact with Ollama and Stable Diffusion
  
- [Integrating Ontologies with Large Language Models for Decision-Making](https://ai.plainenglish.io/integrating-ontologies-with-large-language-models-for-decision-making-bb1c600ce5a3): The intersection of Ontologies and Large Language Models (LLMs) is opening up new horizons for decision-making tools. Leveraging the unique…
  
- [Andyrasika/VQA-Dataset · Datasets at Hugging Face](https://huggingface.co/datasets/Andyrasika/VQA-Dataset)
  

### ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/) (6 messages):

- **CrewAI Gets an Automation Boost**: `@yannie_` shared their GitHub project, which automatically creates a crew and tasks in CrewAI. Check out the repo [here](https://github.com/yanniedog/crewai-autocrew).
  
- **Instill VDP Hits Open Beta**: `@xiaofei5116` announced that Instill VDP is now live on Product Hunt. This versatile data pipeline offers an open-source, no-code/low-code ETL solution and is detailed on their [Product Hunt page](https://www.producthunt.com/posts/instill-vdp).
  
- **Feedback Celebrated for Instill VDP**: `@shihchunhuang` expressed admiration for the Instill VDP project, acknowledging it as an amazing initiative.
  
- **Multimodal VQA Dataset Available**: `@andysingal` mentioned adding the Multimodal Dataset from Mateusz Malinowski and Mario Fritz to their repo for community use. Find it on HuggingFace [here](https://huggingface.co/datasets/Andyrasika/VQA-Dataset), ensuring credit is given to its creators.
  
- **Innovative FaceID Space Created**: `@rxience` launched a HuggingFace space that allows zero-shot transfer of face structure onto newly prompted portraits, inviting others to test [FaceID SDXL space](https://huggingface.co/spaces/r-neuschulz/h94-IP-Adapter-FaceID-SDXL).
  

**Links mentioned**:

- [H94 IP Adapter FaceID SDXL - a Hugging Face Space by r-neuschulz](https://huggingface.co/spaces/r-neuschulz/h94-IP-Adapter-FaceID-SDXL)
  
- [Andyrasika/VQA-Dataset · Datasets at Hugging Face](https://huggingface.co/datasets/Andyrasika/VQA-Dataset)
  
- [GitHub - yanniedog/crewai-autocrew: Automatically create a crew and tasks in CrewAI](https://github.com/yanniedog/crewai-autocrew/tree/main): Automatically create a crew and tasks in CrewAI. Contribute to yanniedog/crewai-autocrew development by creating an account on GitHub.
  
- [Instill VDP - Open source unstructured data ETL for AI first applications | Product Hunt](https://www.producthunt.com/posts/instill-vdp): Versatile Data Pipeline (VDP): An open-source, no-code/low-code solution for quick AI workflow creation. It handles unstructured data, ensuring efficient data connections, flexible pipelines, and smoo...
  

### ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/) (2 messages):

- **Exploring LLM Applications in the Legal Field**: `@chad_in_the_house` expressed interest in integrating Large Language Models (LLMs) in legal settings, such as assisting lawyers or judges. `@gduteaud` found this to be a **super interesting** topic as well.
  

### ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/) (12 messages🔥):

- **Seeking Ad Recommendation Datasets**: `@andysingal` inquired about datasets suitable for ad recommendations, which include text, images, and videos. However, no responses or suggestions were provided.
  
- **Directing Towards Fine-Tuning Scripts**: `@sayakpaul` requested a pointer to a fine-tuning script, and `@pseudoterminalx` responded by mentioning the script is called `train_sd2x.py`.
  
- **Training Script for t2i-adapter Unavailable for SD 1.5/2**: `@square1111` asked about a training script for t2i-adapter for stable diffusion versions 1.5/2, and later pointed out that it is not implemented for those versions, referencing the [Hugging Face Diffusers GitHub repository](https://github.com/huggingface/diffusers/tree/main/examples/t2i_adapter).
  
- **Fine-Tuning for Indian Cuisine on the Menu**: `@vishyouluck` expressed intent to fine-tune a diffusion model for knowledge on Indian recipes and ingredients using [SimpleTuner on GitHub](https://github.com/bghira/SimpleTuner). `@pseudoterminalx` suggested trying SDXL 1.0 with a LoRA approach.
  
- **LoRA Support for SD 2.x Added**: `@pseudoterminalx` noted they've added LoRA support for Stable Diffusion 2.x but also flagged that they have not tested it yet and mentioned that validations might not work as expected.
  

**Links mentioned**:

- [diffusers/examples/t2i_adapter at main · huggingface/diffusers](https://github.com/huggingface/diffusers/tree/main/examples/t2i_adapter): 🤗 Diffusers: State-of-the-art diffusion models for image and audio generation in PyTorch - huggingface/diffusers
  
- [GitHub - bghira/SimpleTuner: A general fine-tuning kit geared toward Stable Diffusion 2.1 and SDXL.](https://github.com/bghira/SimpleTuner/): A general fine-tuning kit geared toward Stable Diffusion 2.1 and SDXL. - GitHub - bghira/SimpleTuner: A general fine-tuning kit geared toward Stable Diffusion 2.1 and SDXL.
  

### ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/) (1 messages):

- **In Search of AR Hit-Testing Insights**: User `@skibbydoo` asked for resources on how hit-testing for **AR** (Augmented Reality) is implemented, specifically relating to plane detection and real-time meshing on mobile devices. Their search hasn't yielded substantial information yet.
  

### ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/) (53 messages🔥):

- **LLM Setup Woes for @kingpoki**: @kingpoki is struggling to get any large language model working with Hugging Face Transformers in Python, encountering timeout errors with no clear exceptions thrown. They attempted to use models like `Deci/DeciLM-7B` and `phi-2` among others, with robust system specs unlikely to be the issue.
  
- **Troubleshooting with @vipitis**: @vipitis engaged in troubleshooting, suggesting various fixes such as updating `accelerate` and `huggingface_hub`, running Python as an admin, and trying out smaller models like `gpt2`. Multiple other avenues were discussed, such as avoiding overriding stderr, decreasing `max_new_tokens`, and verifying if CPU inference works, but no resolution was reported by the end.
  
- **Subtitles Syncing Inquiry by @dtrifuno**: @dtrifuno sought advice on how to match high-quality human transcription text with model-generated speech transcripts containing timestamps. No conclusive solution was provided within the available message history.
  
- **Question on Context Window Limits by @cornwastaken**: @cornwastaken enquired about a resource or repository that details the context window lengths of large language models for a use case involving large documents and question answering. No responses to this question were included in the message history.
  
- **Docker Setup Query from @theintegralanomaly**: @theintegralanomaly asked for assistance on how to pre-download HuggingFace embeddings model data to avoid long initialization times in a new docker container. @Cubie | Tom suggested a potential workflow involving `git clone`, incorporating the model into the docker image, and directing the application to use the local files.
  

**Links mentioned**:

- [Manage huggingface_hub cache-system](https://huggingface.co/docs/huggingface_hub/how-to-cache#limitations.)
  
- [Enable your device for development - Windows apps](https://docs.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development): Activate Developer Mode on your PC to develop apps.
  

### ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/) (12 messages🔥):

- **Looking for Diverse Ad Datasets**: User `@andysingal` inquired about datasets for ad recommendation that include **text, image, and video** formats.
  
- **Pointing to Fine-tuning Scripts**: `@sayakpaul` sought assistance for a fine-tuning script, which was addressed with the suggestion to use the `train_sd2x.py` script by `@pseudoterminalx`.
  
- **Training Script for t2i-adapter Lacks 1.5/2 Support**: `@square1111` shared a discovery that **t2i-adapter training script** does not support sd1.5/2, linking to the [GitHub repository](https://github.com/huggingface/diffusers/tree/main/examples/t2i_adapter) for reference.
  
- **Fine-tuning for Indian Cuisine Imagery**: `@vishyouluck` expressed an intention to fine-tune a **diffusion model** for Indian recipes and ingredients using [SimpleTuner](https://github.com/bghira/SimpleTuner), while seeking advice on the appropriate base model. The recommendation provided by `@pseudoterminalx` was to use **sdxl 1.0** and consider a **LoRA**.
  
- **Unverified LoRA Support for SD 2.x**: `@pseudoterminalx` mentioned adding LoRA support for **Stable Diffusion 2.x** but cautioned that it is untested and validations might fail.
  

**Links mentioned**:

- [diffusers/examples/t2i_adapter at main · huggingface/diffusers](https://github.com/huggingface/diffusers/tree/main/examples/t2i_adapter): 🤗 Diffusers: State-of-the-art diffusion models for image and audio generation in PyTorch - huggingface/diffusers
  
- [GitHub - bghira/SimpleTuner: A general fine-tuning kit geared toward Stable Diffusion 2.1 and SDXL.](https://github.com/bghira/SimpleTuner/): A general fine-tuning kit geared toward Stable Diffusion 2.1 and SDXL. - GitHub - bghira/SimpleTuner: A general fine-tuning kit geared toward Stable Diffusion 2.1 and SDXL.
  

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord Summary

**GPT's Negation Challenge Sparks Discussion at Dev Event**: During an event, there was a notable acknowledgement from a developer that resonated with `@darthgustav.` regarding the AI's issue with handling **negation prompts**, which tends to ignore the negation leading to potential errors.

**Could GPT Assistant Join the Free Tier?**: `@mischasimpson` hinted, based on a live tutorial they watched, that the **GPT assistant** may soon be accessible without cost, which indicates a possible shift towards making advanced AI tools available on OpenAI's free tier.

**Customizing Education with GPT**: Users `@mischasimpson` and `@darthgustav.` discussed the use of GPT for generating personalized reading exercises for children, touching on the simplicity of process and the potential to track completion and performance.

**The Curious Case of the Mythical GPT-4.5 Turbo**: In a conversation spiked with speculation, `@okint` believed to have encountered a version of the AI dubbed "gpt-4.5-turbo." However, others like `@7877` and `@luarstudios` were quick to remind the community to be wary of possible AI fabrications, as such a version might be nonexistent.

**Managing Expectations of GPT's Capabilities**: Users `@solbus` and `@.bren_._` provided clarity on the actual workings of Custom GPTs, dispelling misconceptions that they can be trained directly on **knowledge files** and explaining that true model training requires OpenAI services or building a large language model from scratch.

**OpenAI Channel Summaries**

### ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/) (47 messages🔥):

- **GPT Model Hallucinations Addressed**: `@lugui` mentioned that calling the AI's reference to system internals a "leak" is an exaggeration because that information is public. They clarified that AI does not know its own IP or the server's IP and any such information it gives out is likely a hallucination.
  
- **Clarity on Custom GPT Web Browsing**: In response to `@luarstudios`' issue with web browsing capabilities, the discussion hinted at potential misunderstandings around the AI's features, indicating that it might not be able to directly access external sources.
  
- **LAM (Large Action Model) Buzz and Backend Speculations**: `@exx1` described "Large Action Model" as a buzzword that combines various models, including GPT-y with vision capabilities, and speculated that it could be using OpenAI models in the backend.
  
- **Enthusiasm and Skepticism Over GPT-4.5**:
  
  - While `@michael_6138_97508` advised `@murph12f` on fine-tuning options and the use of datasets from sources like Kaggle, `@lugui` confirmed the possibility of fine-tuning with books.
    
  - In another thread, `@okint` was convinced they had encountered "gpt-4.5-turbo," but others, like `@7877`, reminded them that AI is prone to make up information and that such a version might not exist yet.
    
- **Discord Bot Discussion for Deleting Messages**: `@names8619` asked about using ChatGPT premium to delete Discord posts, and `@7877` suggested developing a Discord bot while warning that unauthorized methods, like the ones shown in a YouTube video, could lead to bans.
  

**Links mentioned**:

[How to Mass Delete Discord Messages](https://www.youtube.com/watch?v=-HH0XuDTlkY): In this video I will be showing you how to mass delete discord messages in dms, channels, servers, ect with UnDiscord which is a easy extension that allows y...

### ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/) (43 messages🔥):

- **Confusion Over GPT Chat Renewal Time**: `@7877` clarifies to `@elegante94` that the cooldown for the next GPT chat is **every 3 hours**, not every hour, addressing confusion about the renewal rate.
  
- **In Search of an AI Note-Taker**: After discussion, `@satanhashtag` suggests a specific channel `<#998381918976479273>` in response to `@blakeskoepka` looking for an **AI note-taking application** for school.
  
- **The Perils of Naming Your GPT**: `@ufodriverr` expresses frustration over their GPT, **Unity GPT**, being banned potentially due to trademark issues without a clear path to appeal, despite efforts to address the issue and observing other similar-named GPTs not being banned.
  
- **Limitations of GPT and Knowledge Files Explained**: `@solbus` corrects misconceptions by explaining that Custom GPTs do not train on **knowledge files** but can reference them, and true model training requires access to OpenAI's fine-tuning services or training your own large language model from scratch.
  
- **GPT Builder's Misrepresentation and Learning Curves**: `@.bren_._` points out that the GPT Builder does not actually train on PDFs in a zip file, despite initial claims, sparking a discussion on ChatGPT's capabilities and how to achieve **custom behaviors** with it.
  

### ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/) (31 messages🔥):

- **Surprise Agreement on Negation Prompts**: `@darthgustav.` expressed satisfaction that a developer echoed their concerns at a GPT event, specifically about the model's tendency to overlook negations in prompts, leading to undesirable behavior.
  
- **GPT Assistant May Become Free**: `@mischasimpson` shared insights while watching a live tutorial, suggesting that the GPT assistant might eventually be accessible on the free tier.
  
- **Enhancing Reading Practices with GPT**: `@mischasimpson` and `@darthgustav.` discussed using GPT for creating tailored reading assignments for students, with `@darthgustav.` advising to keep it simple and offering to help with the project.
  
- **Prompt Crafting Challenges**: `@kobra7777` inquired about ensuring ChatGPT follows complete prompts, with `@darthgustav.` explaining that the model targets roughly 1k tokens and may require additional instructions for longer tasks.
  
- **Testing Task Management with GPT**: `@sugondese8995` shared examples and sought ways to improve testing for a custom GPT built for task management, while `@rendo1` asked for clarification on desired improvements.
  

### ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/) (31 messages🔥):

- **Surprise Echo at Dev Event**: User `@darthgustav.` expressed surprise that a developer repeated their comment on **negative prompts** and its challenges, highlighting that the AI might ignore the negation and perform the unwanted behavior instead.
  
- **Anticipation for GPT Assistant**: `@mischasimpson` mentioned watching a live tutorial about the **GPT assistant** which hinted at the possibility of it becoming available to the *free tier*.
  
- **Crafting Custom Reading Prompts for Children**: `@darthgustav.` and `@mischasimpson` discussed creating an easy method for parents to generate reading exercises for children using **OpenAI's tools**. Complexity and tracking issues were mentioned, including how to know when AI-generated tasks were complete or how children perform.
  
- **Test Prompt Crafting for Custom GPTs**: `@sugondese8995` shared their method of using ChatGPT to build test prompts and expected outputs for a custom GPT designed for **task management** and sought feedback for improvement.
  
- **Seeking Better GPT Prompt Adherence**: `@kobra7777` inquired about strategies to ensure **GPT follows prompts entirely**. `@darthgustav.` provided insight, mentioning the model's approximate token output target and strategies to manage longer requests.
  

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord Summary

- **ByteDance Unleashes Quartet of AI Apps**: ByteDance has launched four new AI apps, **Cici AI, Coze, ChitChop**, and **BagelBell**, powered by **OpenAI's language models**, as reported in [Forbes](https://www.forbes.com/sites/emilybaker-white/2024/01/16/tiktok-bytedance-ai-chatbots-openai/?sh=73f689c1a240) by *Emily Baker-White*.
  
- **AutoGen UI Amps Up Agent Creation**: **AutoGen Studio UI 2.0** has been released, featuring an enhanced interface that facilitates custom agent creation, as demonstrated in a [YouTube tutorial](https://www.youtube.com/watch?v=KIvl-VY8H0Y).
  
- **Artificial Analysis Puts AI Models to the Test**: The new **Artificial Analysis** benchmarking website allows users to compare AI models and hosting providers, focusing on the balance between price and latency, discussed via a [Twitter post](https://fxtwitter.com/swyx/status/1747741795281412133).
  
- **AI in Coding Evolution**: **AlphaCodium** from Codium AI represents a new leap in code generation models and **SGLang** from lmsys introduces an innovative LLM interface and runtime, potentially achieving up to 5x faster performance as noted on [lmsys' blog](https://lmsys.org/blog/2024-01-17-sglang/).
  
- **SPIN Your LLM From Weak to Strong**: A novel fine-tuning method called **Self-Play fIne-tuNing (SPIN)** enhances LLMs via self-generated data, effectively boosting their capabilities as outlined in [this paper](https://arxiv.org/abs/2401.01335).
  
- **ICLR Accepts MoE Paper for Spotlight**: A paper on **Mixture of Experts (MoE)** and expert merging called **MC-SMoE** has been accepted for a spotlight presentation at ICLR, presenting significant resource efficiency improvements, [read here](http://arxiv.org/abs/2310.01334).
  

**Latent Space Channel Summaries**

### ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/) (17 messages🔥):

- **ByteDance Launches New AI Apps**: `@coffeebean6887` shared a [Forbes article](https://www.forbes.com/sites/emilybaker-white/2024/01/16/tiktok-bytedance-ai-chatbots-openai/?sh=73f689c1a240) revealing that TikTok's parent company, **ByteDance**, has launched four new AI apps named **Cici AI, Coze, ChitChop**, and **BagelBell**. The article by *Emily Baker-White* discusses the offerings of these apps and their reliance on **OpenAI's language models**.
  
- **AutoGen UI 2.0 Released**: `@swyxio` mentioned the release of **AutoGen Studio UI 2.0**, providing a [YouTube link](https://www.youtube.com/watch?v=KIvl-VY8H0Y) to a video titled "AutoGen Studio UI 2.0: Easiest Way to Create Custom Agents".
  
- **Artificial Analysis Drops a Benchmark Bomb**: `@swyxio` highlighted a [Twitter post](https://fxtwitter.com/swyx/status/1747741795281412133) that discusses a new AI benchmark comparison website called **Artificial Analysis**. The website compares models and hosting providers, helping users in choosing the best price vs latency trade-offs.
  
- **Syntax to Semantics - The Future of Code?**: In an intriguing [blog post](https://www.alessiofanelli.com/posts/syntax-to-semantics), `@fanahova` contemplates the shift of coding from syntax to semantics, questioning whether everyone will become an AI engineer.
  
- **Next-Gen Code Generation Model on the Horizon**: `@swyxio` confidentially informed the chat about **AlphaCodium**, a new state-of-the-art code generation model being developed by **Codium AI**. Feedback is sought for its launch announcement and the related [blog post](https://www.codium.ai/blog/alphacodium-state-of-the-art-code-generation-for-code-contests/).
  
- **Innovations in LLM Interfaces and Runtimes**: `@swyxio` also shared news about a novel LLM interface and runtime called **SGLang** introduced by **lmsys**, which incorporates **RadixAttention**. This could potentially compete with other LLM systems like Guidance and vLLM, with a statement that SGLang can perform up to 5x faster. Details are available in their [blog post](https://lmsys.org/blog/2024-01-17-sglang/).
  

**Links mentioned**:

- [TikTok Owner ByteDance Quietly Launched 4 Generative AI Apps Powered By OpenAI’s GPT](https://www.forbes.com/sites/emilybaker-white/2024/01/16/tiktok-bytedance-ai-chatbots-openai/?sh=73f689c1a240): The websites and policies for new apps Cici AI, ChitChop, Coze, and BagelBell don’t mention that they were made by ByteDance.
  
- [AI in Action Weekly Jam · Luma](https://lu.ma/el0y5mpi): A weekly virtual chat dedicated to the hands-on application of AI in real-world scenarios, focusing on insights from blogs, podcasts, libraries, etc. to bridge the gap between theory and...
  
- [Tweet from lmsys.org (@lmsysorg)](https://fxtwitter.com/lmsysorg/status/1747675649412854230): We are thrilled to introduce SGLang, our next-generation interface and runtime for LLM inference! It greatly improves the execution and programming efficiency of complex LLM programs by co-designing t...
  
- [State-of-the-art Code Generation with AlphaCodium - From Prompt Engineering to Flow Engineering | CodiumAI](https://www.codium.ai/blog/alphacodium-state-of-the-art-code-generation-for-code-contests/): Read about State-of-the-art Code Generation with AlphaCodium - From Prompt Engineering to Flow Engineering in our blog.
  
- [Tweet from Alessio Fanelli (@FanaHOVA)](https://x.com/FanaHOVA/status/1747759608888996124?s=20): Engineering used to be the bridge between semantics (what the biz wanted) and syntax (how you implemented it) 🌉 [https://www.alessiofanelli.com/posts/syntax-to-semantics](https://www.alessiofanelli.com/posts/syntax-to-semantics) Code has slowly gotten more s...
  
- [AutoGen Studio UI 2.0: Easiest Way to Create Custom Agents](https://www.youtube.com/watch?v=KIvl-VY8H0Y): AutoGen now has a User Interface for creating powerful AI agents without writing code. In this video, we will look at different components of this newly rele...
  
- [Tweet from swyx (@swyx)](https://fxtwitter.com/swyx/status/1747741795281412133): found this absolute GEM in today's @smolmodels scrape of AI discords: [https://artificialanalysis.ai/](https://artificialanalysis.ai/) a new benchmark comparison site • by an independent third party • clearly outlines the quality...
  
- [The "Normsky" architecture for AI coding agents — with Beyang Liu + Steve Yegge of SourceGraph](https://www.latent.space/p/sourcegraph): Listen now | Combining Norvig and Chomsky for a new paradigm,
  

### ▷ #[ai-event-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/) (1 messages):

- **Innovative Fine-Tuning Method for LLMs Introduced**: `@eugeneyan` invited the group with role `@&1107197669547442196` to discuss the new Self-Play fIne-tuNing (SPIN) method, alongside `@713143846539755581`. This method enhances LLMs through self-generated data, negating the need for additional human-annotated data. Read the paper [here](https://arxiv.org/abs/2401.01335).
  

**Links mentioned**:

- [Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models](https://arxiv.org/abs/2401.01335): Harnessing the power of human-annotated data through Supervised Fine-Tuning (SFT) is pivotal for advancing Large Language Models (LLMs). In this paper, we delve into the prospect of growing a strong L...
  
- [Join the /dev/invest + Latent Space Discord Server!](https://discord.gg/s9BWMRJb): Check out the /dev/invest + Latent Space community on Discord - hang out with 2695 other members and enjoy free voice and text chat.
  

### ▷ #[llm-paper-club](https://discord.com/channels/822583790773862470/1107320650961518663/) (32 messages🔥):

- **Eager Readers Gather for Self-Play Study**: `@eugeneyan` invited participants for a discussion on the **Self-Play** paper by linking to its [abstract on arXiv](https://arxiv.org/abs/2401.01335). The paper presents **Self-Play fIne-tuNing (SPIN)**, a fine-tuning method where a Large Language Model (LLM) plays against itself to refine its abilities without new human-annotated data.
  
- **Anticipation for Self-Instruction Insights**: `@swyxio` hyped the upcoming discussion on a paper related to **self-improvement**. The mentioned paper, [Self-Instruct](https://arxiv.org/abs/2212.10560), demonstrates a 33% absolute improvement over the original GPT3 model using a self-generated instruction-following framework.
  
- **The Webcam Effect on Discord Limits**: Users including `@swizec`, `@youngphlo`, and `@gulo0001` discussed technical difficulties caused by Discord's user limit when webcams are on. To address large meeting requirements, `@youngphlo` shared Discord's "stage" feature allows hundreds of viewers in a more webinar-like setting and linked to the relevant [Discord Stage Channels FAQ](https://support.discord.com/hc/en-us/articles/1500005513722-Stage-Channels-FAQ).
  
- **social-format Discord Meetups Announced**: `@swyxio` announced two new Discord clubs starting next week and a social meetup in San Francisco, encouraging community members to participate and socialize. An additional meetup in Seattle is also being planned.
  
- **Spotlight on MoE Research at ICLR**: `@swyxio` highlighted the acceptance of a paper focusing on **Mixture of Experts (MoE)** and expert merging to be featured as a Spotlight paper at ICLR. The paper [MC-SMoE](http://arxiv.org/abs/2310.01334) showcases methods for reducing memory usage and computational needs by up to 80% through merging and compressing experts.
  

**Links mentioned**:

- [Join the /dev/invest + Latent Space Discord Server!](https://discord.gg/s9BWMRJb): Check out the /dev/invest + Latent Space community on Discord - hang out with 2695 other members and enjoy free voice and text chat.
  
- [Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models](https://arxiv.org/abs/2401.01335): Harnessing the power of human-annotated data through Supervised Fine-Tuning (SFT) is pivotal for advancing Large Language Models (LLMs). In this paper, we delve into the prospect of growing a strong L...
  
- [Self-Instruct: Aligning Language Models with Self-Generated Instructions](https://arxiv.org/abs/2212.10560): Large "instruction-tuned" language models (i.e., finetuned to respond to instructions) have demonstrated a remarkable ability to generalize zero-shot to new tasks. Nevertheless, they depend he...
  
- [Tweet from Prateek Yadav (@prateeky2806)](https://fxtwitter.com/prateeky2806/status/1747271753427251636): 🎉 Thrilled to announce our MOE Expert Merging paper has been accepted to @iclr_conf as a SpotLight paper. ! We reduce the inference memory cost of MOE models by utilizing routing statistics-based me...
  
- [GitHub - jondurbin/airoboros at datascience.fm](https://github.com/jondurbin/airoboros?ref=datascience.fm): Customizable implementation of the self-instruct paper. - GitHub - jondurbin/airoboros at datascience.fm
  
- [Solving olympiad geometry without human demonstrations - Nature](https://www.nature.com/articles/s41586-023-06747-5): A new neuro-symbolic theorem prover for Euclidean plane geometry trained from scratch on millions of synthesized theorems and proofs outperforms the previous best method and reaches the performance of...
  

### ▷ #[llm-paper-club-chat](https://discord.com/channels/822583790773862470/822583791217934366/) (65 messages🔥🔥):

- **Zooming In on Video Call Limits**: In a discussion on increasing user limits for voice channels, `@swyxio` and `@yikesawjeez` noticed that despite settings saying 99 users, the actual limit appears to be 25 when video or streaming is turned on. A [Reddit post](https://www.reddit.com/r/discordapp/comments/ma7oyc/when_does_the_25_user_limit_apply/) clarified when the 25-user limit applies.
  
- **Benchmarking Model Progress**: `@swyxio` shared a humorous observation regarding Microsoft's modest improvement over GPT-4 with MedPaLM, noting that a true 8% improvement would indicate a completely new model. Further, they shared personal [benchmark notes](https://github.com/swyxio/ai-notes/blob/main/Resources/BENCHMARKS.md) from GitHub for deeper insights.
  
- **To Share or Not to Share Paper Notes**: `@swyxio` suggested to `@mhmazur` about making a pull request (PR) to Eugene's paper notes repo. `@eugeneyan` later provided the [link to the repo](https://github.com/eugeneyan/llm-paper-notes) on GitHub for reference.
  
- **Exploring the Importance of Reasoning Continuity**: `@yikesawjeez` shared a thread from Twitter discussing the significance of maintaining a logical flow over factual accuracy in prompts when it comes to large language models' performance. This counterintuitive discovery was further examined in a paper available on [arXiv](https://arxiv.org/abs/2310.01798).
  
- **Pondering the Challenges of Healthcare Data**: In a snippet featuring doubts about synthetic data circumventing HIPAA regulations, `@dsquared70` and `@nuvic_` touched on the complexities and costs involved in using AI within regulated environments. `@swyxio` mentioned that the use of GPT-4 in this context makes companies like Scale AI essentially act as "GPT-4 whitewashing shops."
  

**Links mentioned**:

- [Tia clinches $100M to build out clinics, virtual care as investors bank on women's health startups](https://www.fiercehealthcare.com/tech/tia-clinches-100m-as-investors-bank-women-s-health-startups): Tia, a startup building what it calls a "modern medical home for women," secured a hefty $100 million funding round to scale its virtual and in-person care. | Tia, a startup building what it...
  
- [Tweet from Aran Komatsuzaki (@arankomatsuzaki)](https://fxtwitter.com/arankomatsuzaki/status/1745271296437469195): The Impact of Reasoning Step Length on Large Language Models Appending "you must think more steps" to "Let’s think step by step" increases the reasoning steps and signficantly improve...
  
- [Large Language Models Cannot Self-Correct Reasoning Yet](https://arxiv.org/abs//2310.01798): Large Language Models (LLMs) have emerged as a groundbreaking technology with their unparalleled text generation capabilities across various applications. Nevertheless, concerns persist regarding the ...
  
- [Double descent - Wikipedia](https://en.wikipedia.org/wiki/Double_descent)
  
- [MAUVE: Measuring the Gap Between Neural Text and Human Text using Divergence Frontiers](https://arxiv.org/abs/2102.01454): As major progress is made in open-ended text generation, measuring how close machine-generated text is to human language remains a critical open problem. We introduce MAUVE, a comparison measure for o...
  
- [Show Your Work: Scratchpads for Intermediate Computation with Language Models](https://arxiv.org/abs/2112.00114): Large pre-trained language models perform remarkably well on tasks that can be done "in one pass", such as generating realistic text or synthesizing computer programs. However, they struggle w...
  
- [We removed advertising cookies, here’s what happened](https://blog.sentry.io/we-removed-advertising-cookies-heres-what-happened/): This is not another abstract post about what the ramifications of the cookieless future might be; Sentry actually removed cookies from our…
  
- [Deep double descent](https://openai.com/research/deep-double-descent): We show that the double descent phenomenon occurs in CNNs, ResNets, and transformers: performance first improves, then gets worse, and then improves again with increasing model size, data size, or tra...
  
- [The Impact of Reasoning Step Length on Large Language Models](https://t.co/0F8lOSbxWC): Chain of Thought (CoT) is significant in improving the reasoning abilities of large language models (LLMs). However, the correlation between the effectiveness of CoT and the length of reasoning steps ...
  
- [Self-Instruct: Aligning Language Models with Self-Generated Instructions](https://arxiv.org/abs/2212.10560): Large "instruction-tuned" language models (i.e., finetuned to respond to instructions) have demonstrated a remarkable ability to generalize zero-shot to new tasks. Nevertheless, they depend he...
  
- [Reddit - Dive into anything](https://www.reddit.com/r/discordapp/comments/ma7oyc/when_does_the_25_user_limit_apply/)
  
- [Tweet from Carlos E. Perez (@IntuitMachine)](https://fxtwitter.com/IntuitMachine/status/1745773247403036891): 1/n A counterintuitive and surprinsing discovery about LLMs The principle of "reasoning continuity over accuracy" refers to the surprising finding that in chain-of-thought (CoT) prompting, ma...
  
- [GitHub - eugeneyan/llm-paper-notes: Notes from the Latent Space paper club. Follow along or start your own!](https://github.com/eugeneyan/llm-paper-notes): Notes from the Latent Space paper club. Follow along or start your own! - GitHub - eugeneyan/llm-paper-notes: Notes from the Latent Space paper club. Follow along or start your own!
  
- [ai-notes/Resources/BENCHMARKS.md at main · swyxio/ai-notes](https://github.com/swyxio/ai-notes/blob/main/Resources/BENCHMARKS.md): notes for software engineers getting up to speed on new AI developments. Serves as datastore for [https://latent.space](https://latent.space) writing, and product brainstorming, but has cleaned up canonical references und...
  
- [Convert Weak LLM to Strong LLM Using SPIN Technique](https://levelup.gitconnected.com/convert-weak-llm-to-strong-llm-using-spin-technique-9a083d3811df): Can we help a weak LLM get better without getting more data?
  

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord Summary

- **Sharing is Caring, or Scaring?**: Privacy concerns were voiced by `@imusicmash101` regarding the **share feature** on a platform, suggesting enhancements for selective content sharing rather than the whole session history.
  
- **Library Sidebar Snafus Reported**: Discord user `@moyaoasis` pointed out a mismatch between the library sidebar links and the search thread history, sparking a response from `@ok.alex` about different organizational methods between the two.
  
- **Parsing Perplexity.ai Praise and Problems**: Conversations surfaced involving an admiration tweet by `@rileybrown_ai` for **Perplexity.ai**, issues with the Pro role access, and queries about **pplx API's** capabilities in processing requests with URLs and payment processing glitches.
  
- **Perplexity Pro Conversion**: The positive feedback on Perplexity.ai, including its collections feature, has led to users like `@rmoore` upgrading to **Perplexity Pro** and sharing their learning experiences, such as understanding the Qing Dynasty's government structure.
  
- **Community Engagement & Feature Requests**: The community discussed recognition systems, such as starring valuable contributions which could confer the **EXPLORER role**, and the suggestion by `@brknclock1215` to allow feature request upvoting to recognize community desires.
  

**Perplexity AI Channel Summaries**

### ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/) (31 messages🔥):

- **Safety Warning on Share Feature**: User `@imusicmash101` raised a privacy concern that the current **share feature** copies the entire session history, not just the query intended for sharing. They suggested implementing options to share different ranges of content, like "most recent Q&A only".
  
- **Library Links Mismatch**: User `@moyaoasis` reported a problem where the links in the library sidebar don't match the search thread history, sometimes missing the most recent threads. `@ok.alex` clarified that the sidebar list is organized by the most recently opened threads, while the Library is ordered by the newest question asked.
  
- **Inquiry on Perplexity AI's Copilot Functionality**: `@yukiarimo` was curious about the workings of Perplexity AI's **copilot** and the underlying SD model for illustrations. `@me.lk` noted that DALLE3 is used for image generation and shared a link for more information on copilot.
  
- **Benefits of Perplexity Pro vs Normal**: User `@iloveh8` sought insights from the community regarding the differences between **Perplexity Pro** and the free version, specifically if Pro offers more detailed or better-reasoned answers. `@mares1317` posted links to relevant discussions and a comparison between **ChatGPT** and **Perplexity AI**.
  
- **Technical UI Issue on Perplexity Site**: User `@darkblanks` experienced a UI glitch where all text appeared selected and asked for a resolution. `@mares1317` advised making a screenshot and reporting the issue in a specified Discord help channel.
  

**Links mentioned**:

- [What is Perplexity Copilot?](https://blog.perplexity.ai/faq/what-is-copilot): Explore Perplexity's blog for articles, announcements, product updates, and tips to optimize your experience. Stay informed and make the most of Perplexity.
  
- [ChatGPT vs Perplexity AI: Does Perplexity Use ChatGPT? - AI For Folks](https://aiforfolks.com/chatgpt-vs-perplexity-ai/): The AI landscape is constantly shifting, and can be confusing. Many companies overlay different technologies for their own use. In this article, we'll compare
  
- [Perplexity Blog](https://blog.perplexity.ai/faq/what-is-copilot%3E.): Explore Perplexity's blog for articles, announcements, product updates, and tips to optimize your experience. Stay informed and make the most of Perplexity.
  

### ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/) (13 messages🔥):

- **Perplexity.ai Gains Favor**: User `@rileybrown_ai` [tweeted](https://x.com/rileybrown_ai/status/1747456314916446646) praising **@perplexity_ai** for its collections feature, stating it is superior to ChatGPT and Google and improves every month. This tweet influenced `@rmoore` to convert to **Perplexity Pro**.
  
- **Learning with Perplexity.ai**: User `@fanytumor` shared their experience of learning about the **government structure of the Qing Dynasty** after starting their subscription to **Perplexity.ai**.
  
- **Perplexity Pro Role Access Confusion**: Users including `@rmoore` and `@icelavaman` discussed issues regarding obtaining the **<a:pro:1138537257024884847>** role on the server after joining through a **Perplexity settings** link.
  
- **Endorsement for Perplexity**: User `@brknclock1215` provided a [YouTube link](https://www.youtube.com/watch?v=aphHCBSTx7Q&ab_channel=RileyBrown) as a potential strong endorsement of **Perplexity.ai**, although the identity of the YouTuber wasn't clear.
  

**Links mentioned**:

[Tweet from Riley Brown (@rileybrown_ai)](https://x.com/rileybrown_ai/status/1747456314916446646): I use @perplexity_ai more than chatgpt & google. Their collections feature is very underrated. And it gets better every month.

### ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/) (10 messages🔥):

- **Payment Processing Glitch**: User `@rxiiia` encountered an issue where **pplx API** payment setup was not being confirmed despite successful bank authorization. `@ok.alex` responded, advising that the issue had been resolved and prompting to retry the payment method setup.
  
- **Quality Assurance on API vs. Web Interface**: `@jayb1791` inquired about the ability of **pplx API** to process requests involving URLs and whether the API models would deliver the same high-quality summaries as the web interface. `@mares1317` provided a link to a message indicating it's not possible, with `@jayb1791` expressing disappointment over the limitation.
  
- **Community Engagement Encouragement**: `@Dyno` outlined the community recognition system, wherein reacting with a ⭐ emoji to helpful messages can lead to them being featured in the ⭐│starred channel and the author receiving the **EXPLORER role**.
  
- **Feature Request Upvoting Suggestion**: `@brknclock1215` suggested implementing a system to upvote requested features or FAQs, similar to the ⭐ emoji system for helpful messages.
  
- **References Fetching Query**: User `@dvrshil` asked if it's possible to get references along with the generated text from the **online models for the API**, and `@icelavaman` provided a message link in response.
  

---

## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord Summary

Only 1 channel had activity, so no need to summarize...

- **In Search of Expert Specialization**: `.interstellarninja` is considering comparing expert specialization between **Mistral** and **Hermes** fine-tuning to see if the distribution changes. They suggest that re-initializing expert layers might be akin to starting pre-training from scratch.
  
- **Rethinking Domain Specialization in MoEs**: `baptistelqt` is pioneering a "domain balancing loss" as an alternative to load balancing loss to encourage expert specialization in domains within **MoEs**. Despite current imperfections, the method shows potential, and `baptistelqt` promises to publish results and code when satisfied.
  
- **Function Calling and MoEs**: `.interstellarninja` is working on finetuning **Mixture of Experts (MoEs)** models with domain specialization where each expert is proficient in a specific domain or function class. They note that a model good at coding might have each expert specialize in a particular programming language.
  
- **A Promo for Open LLMs**: In the context of synthetic data generation, `baptistelqt` asks for recommendations on the best open LLMs, weighing between 7b-14b sized models, or to opt for **GPT-4**. `stereoplegic` responds with appreciation for the open-source approach and shares a relevant paper on the topic: [Using ICL for data generation](https://arxiv.org/abs/2310.13961).
  
- **New Function Calling Architectures and Datasets**: In response to `.interstellarninja`’s focus on function-calling fine-tunes, `yikesawjeez` mentions novel architectures like **gorilla openfunctions** and **nexusraven**, and IS supplemented with **glaive** and **fireworks-ai**. `yikesawjeez` also shares a link to a Google Doc with API/Function calling datasets found in manifold: [API/Function Calling Datasets Doc](https://docs.google.com/document/d/1OHjNOK4-ih3rtr21yOcOfkwZDDhqbLVtzpOPUXLXDww/edit).
  

**Links mentioned**:

- [Ensemble-Instruct: Generating Instruction-Tuning Data with a Heterogeneous Mixture of LMs](https://arxiv.org/abs/2310.13961): Using in-context learning (ICL) for data generation, techniques such as Self-Instruct (Wang et al., 2023) or the follow-up Alpaca (Taori et al., 2023) can train strong conversational agents with only ...
  
- [API/Function calling datasets](https://docs.google.com/document/d/1OHjNOK4-ih3rtr21yOcOfkwZDDhqbLVtzpOPUXLXDww/edit)
  

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord Summary

- **GCP TPUs Leaving Engineers Hanging**: `@henriklied` reported challenges in finding current libraries for training AI models such as GPT-J on Google Cloud TPUs, due to many libraries being outdated.
  
- **FrankenMoE Model Shines, With Caveats**: `@lee0099` lauded a [frankenMoE model on Hugging Face](https://huggingface.co/Kquant03/FrankenDPO-4x7B-bf16) for its benchmark performance, whilst flagging potential training difficulties with its routing model.
  
- **Blending Beats ChatGPT**: Sharing a [research paper](https://arxiv.org/pdf/2401.02994.pdf), `@le_mess` and `@xzuyn` discussed how a blend of diverse models could surpass ChatGPT-3.5's prowess, though highlighted the random nature of model selection during conversations.
  
- **Axolotl Ponders Proper Placement**: Ongoing decisions in **Axolotl's** development involve the best placement for new system messages code, with a [pull request](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1117) exploring whether `load_datasets` should be the residence.
  
- **Flash Attention Breakthrough on ROCm**: `@odellus_` announced that flash attention is now supported with ROCm, showcasing the project's advancement in exact attention capabilities for AI models, available at [fxmarty/flash-attention-rocm](https://github.com/fxmarty/flash-attention-rocm).
  
- **Fine-Tuning Queries Rising**: The conversation gravitated towards optimal AWS instances for training **Mistral 7B**, with `@jacques_10431` seeking advice and contemplating using **Axolotl**, supported by a reference on [fine-tuning with Magic: The Gathering](https://generallyintelligent.substack.com/p/fine-tuning-mistral-7b-on-magic-the) data.
  
- **LLMs Get Personal**: `@jb5846` ignited a debate on whether LLMs can maintain knowledge across large text documents. `@noobmaster29` clarified that full tuning would be necessary for adding new knowledge. He also shared a [research paper focusing on the potential and risks of foundation models](https://arxiv.org/pdf/2311.00176.pdf).
  
- **WebSocket Woes in Runpod**: `@dangfutures` faced obstacles with a Cross-Origin WebSocket being barred, resulting in a 403 error, but found guidance suggesting the use of the `train-notebook` branch and advice to restart the Jupyter Lab process to mitigate the issue.
  

**OpenAccess AI Collective (axolotl) Channel Summaries**

### ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/) (22 messages🔥):

- **Seeking Training Tools for GCP TPUs**: `@henriklied` expressed difficulty finding up-to-date libraries for training models like GPT-J on Google Cloud TPUs, noting that many libraries seem outdated or unmaintained.
  
- **FrankenMoE Model Boasts Impressive Results**: `@lee0099` shared a link to a [Hugging Face model page](https://huggingface.co/Kquant03/FrankenDPO-4x7B-bf16), highlighting a frankenMoE model that outperforms larger models on certain benchmarks, but mentioned a potential issue with training the routing model.
  
- **Model Blending Can Outdo ChatGPT**: `@le_mess` shared a [research paper](https://arxiv.org/pdf/2401.02994.pdf) discussing how blending multiple models can defeat ChatGPT-3.5, with `@xzuyn` adding that the selection of models during conversation was random, without intelligent choice optimization.
  
- **Randomness Adds Spice to Conversations**: `@leoandlibe` humorously illustrated how having multiple models, like one trained on the Bible and another on WallStreetBets, could make interactions more unpredictable and interesting.
  
- **Upcoming Training Opportunity with H100s**: `@ytl120` inquired about the best open-source model to train with around 200 H100 GPUs for 4 weeks, suggesting multiple options and noting constraints around dataset access. `@nanobitz` and `@nruaif` responded, discussing the feasibility and impact of various architectures such as Mistral, Mamba, and text2img models for the project.
  

**Links mentioned**:

[Kquant03/FrankenDPO-4x7B-bf16 · Hugging Face](https://huggingface.co/Kquant03/FrankenDPO-4x7B-bf16)

### ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/) (4 messages):

- **Axolotl's Codebase Dilemma**: `@le_mess` is considering where new code for system messages should reside in the **Axolotl** architecture. The code was added to `cli/train.py` and `cli/preproccess.py`, but there's a suggestion that `load_datasets` might be a more appropriate location. [Pull request #1117](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1117) is awaiting further input.
  
- **Flash Attention Now Supporting ROCm**: User `@odellus_` pointed out the availability of a ROCm branch for flash attention which is not by the VLLM folks, but rather another project which can be found on GitHub at [fxmarty/flash-attention-rocm](https://github.com/fxmarty/flash-attention-rocm). This indicates advancement in memory-efficient exact attention capabilities for AI models.
  

**Links mentioned**:

- [GitHub - fxmarty/flash-attention-rocm: Fast and memory-efficient exact attention](https://github.com/fxmarty/flash-attention-rocm): Fast and memory-efficient exact attention. Contribute to fxmarty/flash-attention-rocm development by creating an account on GitHub.
  
- [Draft: Feat/chatml add system message by mhenrichsen · Pull Request #1117 · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1117): Need ideas on how to change the default system message in the prompter.
  

### ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/) (6 messages):

- **Fine-Tuning Tips for Mistral 7B**: User `@jacques_10431` is looking to fine-tune **Mistral 7B** and seeks recommendations for an **AWS instance type** for this purpose, considering using **Axolotl**. They referenced a post about fine-tuning on Magic: The Gathering ([Fine-Tuning Mistral 7B](https://generallyintelligent.substack.com/p/fine-tuning-mistral-7b-on-magic-the)).
  
- **General vs Specific Knowledge in LLMs**: `@jb5846` inquired about the best approach for his LLM to answer questions and make generalizations across multiple large text documents. They questioned whether fine-tuning with raw data allows the model to retain memory of the documents for better responses.
  
- **Full Tune vs Instruction Tune on LLMs**: In response to `@jb5846`, `@noobmaster29` stated that **full tuning** is required to impart new knowledge to an LLM, and while **instruction tuning** can be beneficial, enabling a model to learn specific facts is not straightforward.
  
- **Research on LLMs and Fine-Tuning**: `@noobmaster29` shared a link to a research paper, potentially relevant to the discussion on tuning language models: [On the Opportunities and Risks of Foundation Models](https://arxiv.org/pdf/2311.00176.pdf).
  
- **Tailoring AWS Instances for GPU Work**: `@nanobitz` responded to `@jacques_10431` with a question about whether they plan to use **qLoRA**, **LoRA**, or **FFT**, as the choice will affect their AWS instance requirements.
  

### ▷ #[runpod-help](https://discord.com/channels/1104757954588196865/1162430527215763569/) (7 messages):

- **Cross-Origin WebSocket Blocked**: `@dangfutures` reported an issue with a **Cross-Origin WebSocket attempt** being blocked and received a 403 error.
  
- **Branch Suggestion for Fixing WebSocket Issue**: `@caseus_` suggested trying the `train-notebook` branch to resolve the **WebSocket issue**.
  
- **Need to Merge Branch for Solution**: `@caseus_` mentioned the need to **merge the** `train-notebook` branch for the fix to work.
  
- **Advice to Restart Jupyter Lab Process**: `@caseus_` offered a workaround by suggesting the user can **kill and restart the jupyter lab process**.
  
- **Error with 'blinker' Package During Installation**: `@dangfutures` encountered an error regarding an existing installation of the `blinker` package that couldn't be uninstalled, but then resolved the issue themselves.
  

### ▷ #[replicate-help](https://discord.com/channels/1104757954588196865/1197414694248534116/) (1 messages):

hamelh: 〰️

---

## [LlamaIndex Discord](https://discord.com/channels/1059199217496772688) Discord Summary

- **RankGPT Takes On Document Sorting**: **RankGPT** leverages **GPT-3.5 and GPT-4** to outclass monoBERT and Cohere rerank in document ranking tasks, potentially revolutionizing document filtering according to a [tweet](https://twitter.com/llama_index/status/1747681530347216995) by `@sunweiwei12`.
  
- **Full-Stack Feat with LlamaIndex and Sparrow**: A [full-stack app](https://twitter.com/llama_index/status/1747717413498651041) integrating **LlamaIndex and Sparrow** has been showcased by `@andrejusb`, marking a notable development in application architecture.
  
- **LlamaIndex.TS Streams Ahead**: The latest 0.0.47 **LlamaIndex.TS** release introduces streaming capabilities for all endpoints, as announced in this [tweet](https://twitter.com/llama_index/status/1747746779058290800), with examples and download details available.
  
- **Semantic Chunking Enhances AI Memory**: An [article](https://medium.com/ai-advances/unleashing-the-power-of-semantic-chunking-a-journey-with-llamaindex-767e3499ca73) shared by `@andysingal` emphasizes the pivotal role of **semantic chunking** for improving language model performance and enabling better long-term memory in computational applications.
  
- **Various LlamaIndex Challenges and Considerations**: Discord users discussed multiple concerns regarding LlamaIndex, including URL handling in RAG contexts, graph segregation in Neo4j, SemanticSplitterNodeParser issues, web interface navigation, and the impact of metadata on chunk size, but concrete solutions or consensus were not established in the conversations.
  

**LlamaIndex Discord Channel Summaries**

### ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/) (3 messages):

- **RankGPT Excels in Document Ranking**: Research highlighted by `@sunweiwei12` shows that **RankGPT**, using **GPT-3.5 and GPT-4**, outperforms both monoBERT and Cohere rerank in document ranking tasks. [Their tweet](https://twitter.com/llama_index/status/1747681530347216995) suggests its potential as a secondary filter for document selection.
  
- **Sparrow and LlamaIndex in Full-stack Apps**: `@andrejusb` demonstrates building a full-stack app with **LlamaIndex and Sparrow**. More details can be found in [their tweet](https://twitter.com/llama_index/status/1747717413498651041).
  
- **LlamaIndex.TS Introduces Streaming**: The new version 0.0.47 of **LlamaIndex.TS** now includes streaming support for all endpoints. Example and download available in [the announcement tweet](https://twitter.com/llama_index/status/1747746779058290800).
  

**Links mentioned**:

- [LlamaIndexTS/examples/huggingface.ts at llamaindex@0.0.47 · run-llama/LlamaIndexTS](https://t.co/RHp7ThXmQd): LlamaIndex is a data framework for your LLM applications - run-llama/LlamaIndexTS
  
- [llamaindex](https://t.co/3agScNi74h)
  

### ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/) (31 messages🔥):

- **Seeking Guidance on LlamaIndex and OpenAI Assistants**: `@don_ramoncillo` is revisiting a financial project, wondering about the benefits of their existing LlamaIndex system in comparison to new OpenAI features like Assistants and Retrieval. No specific answer was given in the related messages.
  
- **Handling URLs in RAG Context Questioned**: `@stdweird` questioned the importance of keeping URLs in the context for RAG, speculating it could be valuable yet might confuse the system. They expressed a rationale to retain URLs, but the issue remains open without a definitive community consensus.
  
- **Proper Use of GraphStores with LlamaIndex**: `@jpd1998` sought assistance on segregating graphs in Neo4j using LlamaIndex, aiming to store and query multiple document graphs distinctly, and inquired about persisting index information. The community didn't provide an answer within the messages.
  
- **Challenges with SemanticSplitterNodeParser**: `@dr.yyh_59768` mentioned difficulties in getting SemanticSplitterNodeParser to function despite an up-to-date environment and LlamaIndex version. The nature of the problem remains unaddressed in the conversation.
  
- **Persistent Annoyance with Web Interface Navigation**: `@mysterious_avocado_98353` brought up an issue with a web interface where closing a page navigates back to the top, forcing users to scroll through long pages again. `@cheesyfishes` suggested switching to stable docs, and `@mysterious_avocado_98353` realized different versions were displayed when accessed through Google.
  
- **Discussions on Metadata and Chunk Size in LlamaIndex**: `@americanthinker` sparked a conversation about the linkage between metadata length and chunk size in LlamaIndex, querying the design decisions behind default behaviors. `@cheesyfishes` shared insights, implying a potential bug regarding ignoring metadata length during chunking when `include_metadata` is set to false.
  

**Links mentioned**:

- [LlamaIndex 🦙 0.9.33](https://docs.llamaindex.ai/en/stable/)
  
- [Defining and Customizing Documents - LlamaIndex 🦙 0.9.33](https://docs.llamaindex.ai/en/stable/module_guides/loading/documents_and_nodes/usage_documents.html#advanced-metadata-customization)
  
- [maidalun1020/bce-embedding-base_v1 · Hugging Face](https://huggingface.co/maidalun1020/bce-embedding-base_v1)
  

### ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/) (1 messages):

- **Exploring Semantic Chunking with LlamaIndex**: User `@andysingal` shared a [Medium article](https://medium.com/ai-advances/unleashing-the-power-of-semantic-chunking-a-journey-with-llamaindex-767e3499ca73) detailing the importance of **semantic chunking** in language models. The piece delves into how breaking text into manageable parts can enhance model performance and facilitate long-term memory in applications.
  

**Links mentioned**:

[Unleashing the Power of Semantic Chunking: A Journey with LlamaIndex](https://medium.com/ai-advances/unleashing-the-power-of-semantic-chunking-a-journey-with-llamaindex-767e3499ca73): Ankush k Singal

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord Summary

- **Mixtral Achieves FLOP Efficiency**: `bjoernp` highlighted that **Mixtral** operates with higher FLOP efficiency by only activating 2 experts at a time, consuming less FLOPs than a 14b model and performing comparably to a 13b model in this regard.
  
- **Innovating Novel Writing with LLM**: `@rasdani` shared how Jon Durbin's Bagel release used human answers as accepted and LLM answers as rejected to enhance novel writing capabilities of LLMs. This method and its datasets can be explored on [Hugging Face](https://huggingface.co/datasets/jondurbin/gutenberg-dpo-v0.1) and [GitHub](https://github.com/jondurbin/bagel).
  
- **Advancing LLM Pipelines Over Fine-Tuning**: In a discussion about large text document analysis, `@_jp1_` recommended using advanced LLM pipelines such as rerankers and graph-based node retrieval, referring to llama-index for tutorials, instead of relying on fine-tuning for generalizations.
  
- **Beware of LLM Hallucinations**: `@philipmay` pointed to a [paper](https://arxiv.org/pdf/2304.09848.pdf) that categorizes a tendency of LLMs to make unsupported statements and inaccurately cite sources, cautioning that the most helpful-seeming responses could often be unreliable.
  
- **Evaluating Emotional Intelligence of LLMs**: `.calytrix` presented **EQ-Bench v2**, an emotional intelligence benchmark for LLMs that features triple the test questions and reduces variance in scoring, as seen in their [Github](https://github.com/EQ-bench/EQ-Bench) and [paper](https://arxiv.org/abs/2312.06281). Score variance has been reduced to 0-4%, and there's a noted correlation between EQ-Bench scores and larger LLM leaderboards.
  

**DiscoResearch Channel Summaries**

### ▷ #[mixtral_implementation](https://discord.com/channels/1178995845727785010/1182759434326396998/) (2 messages):

- **FLOPs Comparison Inquiry**: `vara2096` asked about the FLOPs per token for **Mixtral** compared to **Mistral-7B** and **Llama-2-70B**.
  
- **Mixtral Packs a Lightweight FLOP Punch**: `bjoernp` clarified that **Mixtral** is comparable to a 13b model in terms of FLOPs, as it activates only 2 experts at a time, and its FLOPs count is less than that of a 14b model because not all its weights are expert weights.
  

### ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/) (10 messages🔥):

- **Enhancing Novel Writing for LLM with Gutenberg DPO**: `@rasdani` shared that Jon Durbin used human answers as accepted and LLM answers as rejected in the latest Bagel release, which aims to improve novel writing capabilities of LLMs using public domain books. More about the dataset and code can be found on [Hugging Face](https://huggingface.co/datasets/jondurbin/gutenberg-dpo-v0.1) and [GitHub](https://github.com/jondurbin/bagel).
  
- **Navigating Large Text Document Analysis**: `@jb5846` raised a question about whether fine-tuning is better for getting generalizations across multiple large documents, and `@_jp1_` suggested that advanced LLM pipelines like reranker and graph-based node retrieval might be more suitable, referring to llama-index for great tutorials.
  
- **LLMs and the temptation to hallucinate**: `@philipmay` noted a tendency of LLMs to add their own knowledge rather than strictly adhering to provided context, citing a [paper](https://arxiv.org/pdf/2304.09848.pdf) which found that generative search engine responses often contain unsupported statements and the most helpful-seeming responses frequently had inaccurate citations.
  
- **Inquiry about the Best Open-Source Embedding Models**: `@vara2096` inquired about the best open-source embedding models for English clustering, and `@philipmay` recommended checking out resources on [Hugging Face's MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard) and [SBERT's pre-trained models](https://www.sbert.net/docs/pretrained_models.html).
  

**Links mentioned**:

- [MTEB Leaderboard - a Hugging Face Space by mteb](https://huggingface.co/spaces/mteb/leaderboard)
  
- [Pretrained Models — Sentence-Transformers documentation](https://www.sbert.net/docs/pretrained_models.html)
  
- [jondurbin/gutenberg-dpo-v0.1 · Datasets at Hugging Face](https://huggingface.co/datasets/jondurbin/gutenberg-dpo-v0.1)
  
- [GitHub - jondurbin/bagel: A bagel, with everything.](https://github.com/jondurbin/bagel): A bagel, with everything. Contribute to jondurbin/bagel development by creating an account on GitHub.
  

### ▷ #[benchmark_dev](https://discord.com/channels/1178995845727785010/1183158791605330051/) (2 messages):

- **Emotional Intelligence Bench: EQ-Bench v2 Launches**: `.calytrix` introduced **EQ-Bench v2**, an upgraded Emotional Intelligence Benchmark for LLMs featuring 3x the number of test questions and designed to reduce variance. The new version also accounts for different perturbations like temperature and prompt formats, affecting benchmark scores, which are detailed in their [Github](https://github.com/EQ-bench/EQ-Bench) and their recently published [paper](https://arxiv.org/abs/2312.06281).
  
- **Robustness and Sensitivity in EQ-Bench v2**: The enhanced **EQ-Bench v2** with 171 questions improves robustness to test environment variations, reducing score variance to 0-4% as opposed to v1's up to 10%, as `.calytrix` explains. The benchmark's detailed four-part questions enable finer discrimination of a model's emotional intelligence capabilities.
  
- **Indie Benchmarks Correlate with Major LLM Leaderboards**: `.calytrix` observed a strong correlation between EQ-Bench scores and larger model leaderboards, noting that some models, like **Beagle14-7B** and **SOLAR-10.7B-Instruct-v1.0**, perform exceptionally well. This suggests the legitimacy of these models' capabilities in emotional intelligence evaluations.
  

**Links mentioned**:

[EQ-Bench Leaderboard](https://eqbench.com.)

### ▷ #[embedding_dev](https://discord.com/channels/1178995845727785010/1192471915504341062/) (10 messages🔥):

- **Grok Speedier with Unspecified Tool**: `@rasdani` mentioned that Grok utilizes a tool that is rumored to be one of the fastest, but did not specify the tool nor provided further evidence or links.
  
- **Cursor IDE's Ground Truth Magic**: `@rasdani` linked to a tweet from [Aman Sanger](https://x.com/amanrsanger/status/1732145826963828997?s=46&t=1jtkL4JPu-DUOdo8JC668g) describing how Cursor IDE developed high-quality retrieval datasets for training embeddings/rerankers using GPT-4 grading and the Trueskill ratings system.
  
- **Disappointment with M2 Embed Models**: `@sebastian.bodza` reported poor performance with M2 Embed models compared to previously used bge embeddings and provided a link from [Hugging Face](https://huggingface.co/togethercomputer/m2-bert-80M-2k-retrieval) for the M2-BERT model.
  
- **Skepticism Towards M2 BERT Retrieval Benchmarks**: `@maxidl` expressed doubts about the M2 BERT's retrieval finetuning, noting the absence of details in the paper and skepticism due to the model's testing against transcripts, reports, and papers with character counts not aligned with typical benchmarks.
  
- **Missing Classic Retrieval Scores**: `@sebastian.bodza` inquired about the M2 model's specific retrieval scores like mean reciprocal rank (MRR) or precision, and `@maxidl` acknowledged that typical retrieval metrics like Recall, Precision, and MRR were not present in the M2 team's blog post benchmarks.
  

**Links mentioned**:

- [togethercomputer/m2-bert-80M-2k-retrieval · Hugging Face](https://huggingface.co/togethercomputer/m2-bert-80M-2k-retrieval)
  
- [Tweet from Aman Sanger (@amanrsanger)](https://x.com/amanrsanger/status/1732145826963828997?s=46&t=1jtkL4JPu-DUOdo8JC668g): At Cursor, we've built very high-quality retrieval datasets (for training embeddings/rerankers). To do this, we use GPT-4 grading and the Trueskill ratings system (a better version of Elo) Here’...
  

---

## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord Summary

- **GPT-4 Struggles with Vision Instructions**: A user named `@thebaghdaddy` highlighted issues with **GPT-4** not effectively using its innate vision capabilities for object detection tasks, despite attempts to steer it away from Python package dependencies.
  
- **Inference Service Insights**: `@rabiat` queried about inference services, with `@robotums` suggesting **Anyscale** and **Together.AI**, highlighting **Together.AI** for its low **Time To First Tweet (TTFT)**. There was also a mention of **Mixtral 8x7b Instruct**'s launch and users were directed to follow updates on Twitter.
  
- **Production Inference Concerns and Mixtral Quirks**: It was pointed out that **pplx-api** doesn't plan to host open-source models for production inference as it is more of a talent attraction tool. Moreover, issues with **Mixtral** throwing random text at the end of responses were being discussed, with potential causes being the base model use or `[INST]` token setup.
  
- **Tips for Model Finetuning**: `@natureplayer` recommended the MLX examples repository for those looking to finetune models and emphasized the flexibility of finetuning even small or quantized models on any setup.
  
- **Azure Performance Issues Noted**: User `@rabiat` reported slow Azure service speeds across different regions and was inquiring if others were experiencing similar issues.
  

**LLM Perf Enthusiasts AI Channel Summaries**

### ▷ #[gpt4](https://discord.com/channels/1168579740391710851/1168582188950896641/) (2 messages):

- **Seeking Pure GPT-4 Vision Capabilities**: `@thebaghdaddy` expressed frustration over GPT-4's tendency to default to using Python packages for object detection tasks instead of utilizing its **innate vision capabilities**. They noted that instructing the model to avoid "advanced analytics" was ineffective in this dilemma.
  

### ▷ #[opensource](https://discord.com/channels/1168579740391710851/1168606773595349082/) (11 messages🔥):

- **Choosing the Right Inference Service**: `@rabiat` posed a question about which service to choose for different needs, with `@robotums` suggesting **Anyscale** for tools and **Together.AI** for non-tool calling, especially if low **Time To First Tweet (TTFT)** is a high priority.
  
- **New Model Launch Announcement**: `@rabiat` shared a [link](https://artificialanalysis.ai/models/mixtral-8x7b-instruct) announcing the launch of **Mixtral 8x7b Instruct** and recommended following on Twitter for updates.
  
- **Concerns Regarding Production Inference**: `@robotums` expressed that despite the appeal of **Mixtral**, the **pplx-api** has no plans to host open-source models for production inference, and their API is primarily a talent attraction strategy.
  
- **Mixtral Response Issues Being Investigated**: `@thisisnotawill` asked if anyone has encountered issues with **Mixtral**, specifically regarding random text at the end of responses. `@natureplayer` suggested this could be due to using the base model or incorrect setup with `[INST]` tokens.
  
- **Finetuning Models on Any Setup**: `@natureplayer` recommended the MLX examples repository as a starting point for finetuning models and mentioned that it's possible to finetune small or quantized models on any setup, emphasizing its usability regardless of the speed.
  

**Links mentioned**:

[Mixtral 8x7B - Host Analysis | ArtificialAnalysis.ai](https://artificialanalysis.ai/models/mixtral-8x7b-instruct): Analysis of Mixtral 8x7B Instruct across metrics including quality, latency, throughput, price and others.

### ▷ #[speed](https://discord.com/channels/1168579740391710851/1168986766607384638/) (1 messages):

rabiat: Azure pretty slow for us across different regions. Anyone expierenceing the same?

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord Summary

- **Support for Pinecone Serverless in Langchain on Hold**: `@leonvanzyl` queried if Langchain will introduce support for Pinecone Serverless, given that current Langchain versions lack compatibility with the Pinecone package that includes serverless.
  
- **Node.js Axios Request Hits an API Wall**: `@digitalsimboja` encountered a `404` error when making API requests to the OpenAI endpoint with axios in Node.js, and provided comprehensive details of the error stack for community assistance.
  
- **The Quest for Optimal Local LLMs for Czech Translation**: `@kompicka` sought recommendations for high-quality, efficient Local LLMs capable of translating large Czech datasets, referencing experiences with Facebook's M2M and Falcon 180B, the latter having slower performance despite good quality translations.
  
- **Exploring Alternatives for Swift LLM Inference**: `@maximuslee` reported slow inference and repetitive outputs while using Langchain with FAISS and an 18b llama model, seeking faster alternatives that could handle large models more efficiently.
  
- **LangChain Advances in Streaming Technology**: `@veryboldbagel` shared an RFC and an example notebook on improvements to LangChain's streaming capabilities and requested feedback from the community. The example notebook can be found [here](https://github.com/langchain-ai/langchain/blob/dbbc7fa0d66bcdde420760baa0ddb9918c12c349/docs/docs/modules/agents/how_to/streaming_events.ipynb) and the RFC discussion is available [here](https://github.com/langchain-ai/langchain/discussions/16175).
  
- **Interfacing LangServe with SvelteKit**: Users `@albertperez.` and `@hiranga.g` raised questions about integrating **LangServe** with SvelteKit and employing **OpenAIAssistantRunnable with LCEL**, including specific use cases and implementation concerns.
  
- **New AI Data Pipeline and Bilingual Model Enhancements Shared**: Open beta for **Instill VDP** was announced by @xiaofei5116 and is a versatile data pipeline designed for AI applications with unstructured data, available on Product Hunt since January 17th, 2024, linked [here](https://www.producthunt.com/posts/instill-vdp). `@johnda98` shared that they had successfully integrated **langserve** with Google Cloud Platform. `@maidalun` released a bilingual and crosslingual embedding model on Hugging Face, optimized for RAG and compatible with LangChain and llamaindex, and it's accessible [here](https://huggingface.co/maidalun1020/bce-embedding-base_v1).
  
- **Showcasing the AI Crawl Assistant**: `@kagnar.` shared a [demo](https://x.com/kagnar_/status/1747828432266821905?s=46&t=B-6g0ZnbS1wgTjzR7lc5Gw) of his **AI Crawl Assistant**, showcasing its capability to navigate sitemaps with natural language inputs using OpenAI Assistant API and Mistral models.
  

**LangChain AI Channel Summaries**

### ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/) (7 messages):

- **Inquiry about Langchain Support for Pinecone Serverless**: `@leonvanzyl` asked whether Langchain will soon roll out support for Pinecone Serverless, as the current versions of Langchain are not compatible with the latest Pinecone package that includes serverless.
  
- **Seeking Assistance with 404 Error**: `@digitalsimboja` requested help with an error featuring `{"message": "Request failed with status code 404"}` received when making API requests with axios in Node.js to the OpenAI API endpoint, as shown in the detailed error stack provided in the message.
  
- **Searching for Local LLMs for Niche Language Translation**: `@kompicka` inquired about recommendations for good Local LLMs suitable for translating large datasets in Czech, mentioning previous experiences with Facebook's M2M and Falcon 180B, noting that the latter has good quality but with significant overhead and slow performance.
  
- **Seeking Alternatives for Efficient Inference with Large LLMs**: `@maximuslee` discussed challenges using Langchain with FAISS and an 18b llama model due to slow inference performance and repetitive answers while utilizing the ConversationalRetrievalChain feature, and is looking for faster inference alternatives for handling large models.
  
- **LangChain Improves Streaming with RFC and Example Notebook**: `@veryboldbagel` shared a GitHub [RFC](https://github.com/langchain-ai/langchain/discussions/16175) and an example notebook ([streaming_events.ipynb](https://github.com/langchain-ai/langchain/blob/dbbc7fa0d66bcdde420760baa0ddb9918c12c349/docs/docs/modules/agents/how_to/streaming_events.ipynb)) related to improvements in streaming with LangChain, seeking community feedback and discussion.
  
- **Announcement of Open Source Bilingual and Crosslingual Embedding Model**: `@maidalun` announced the release of an open source bilingual and crosslingual model on Hugging Face, optimized for RAG and compatible with Langchain and llamaindex, inviting feedback from the community.
  
- **Issue with RAG Pipeline Returning Incomplete Responses**: `@vinayak.pevekar` reported an issue with the RAG pipeline using "Mistral-7B-Instruct-v0.1-GGUF" and llama_index, where it was returning responses composed of hashes instead of expected outputs, noting that it was functioning correctly two days prior.
  

**Links mentioned**:

- [maidalun1020/bce-embedding-base_v1 · Hugging Face](https://huggingface.co/maidalun1020/bce-embedding-base_v1)
  
- [🛸 Streaming: RFC Adding astream_event to all Runnable objects to help with streaming use cases · langchain-ai/langchain · Discussion #16175](https://github.com/langchain-ai/langchain/discussions/16175): Hi everyone! We want to improve the streaming experience in LangChain. We're considering adding a astream_event method to the Runnable interface. The code below is from the following PR and has no...
  

### ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/) (2 messages):

- **Asking for OpenAIAssistantRunnable Guidance**: User `@albertperez.` posed a question about employing **OpenAIAssistantRunnable with LCEL**, specifically how to utilize it with prompts that contain `input_variables`. They provided a snippet of their code for context.
  
- **SvelteKit & Python LangServe Query**: User `@hiranga.g` inquired about the integration of Python **LangServe with SvelteKit**, seeking assistance from the community.
  

### ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/) (4 messages):

- **Instill VDP Hits Open Beta**: @xiaofei5116 announced the open beta launch of **Instill VDP** on Product Hunt, an open-source versatile data pipeline designed for unstructured data for **AI applications**. It features a robust design, extensibility, refined integrations, and both no-code and low-code solutions, [launched on January 17th, 2024](https://www.producthunt.com/posts/instill-vdp).
  
- **Langserve Deployed to GCP with Advanced Features**: `@johnda98` has integrated their **langserve** work with Google Cloud Platform, featuring Google Drive access, SQL endpoints, and document loading capabilities, all constructed with **LangChain libraries**.
  
- **Launch of AI Crawl Assistant**: `@kagnar.` shared a [demo video](https://x.com/kagnar_/status/1747828432266821905?s=46&t=B-6g0ZnbS1wgTjzR7lc5Gw) of their new **AI Crawl Assistant** that navigates sitemaps via natural language inputs, leveraging OpenAI Assistant API and Mistral models.
  
- **Bilingual and Crosslingual RAG Embedding Goes Public**: `@maidalun` released **BCEmbedding**, a bilingual and crosslingual embedding model for English and Chinese, optimized for **Retrieval Augmented Generation (RAG)** and accessible on [HuggingFace](https://huggingface.co/maidalun1020/bce-embedding-base_v1), aimed at facilitating integration with LangChain and llamaIndex.
  

**Links mentioned**:

- [Tweet from kagen (@kagnar_)](https://x.com/kagnar_/status/1747828432266821905?s=46&t=B-6g0ZnbS1wgTjzR7lc5Gw): Quick demo of @nerobotai AI Crawl Assistant in action. #buildinpublic It's able to navigate sitemaps for desired pages all from natural language inputs. It does so by leveraging a set of tools a...
  
- [maidalun1020/bce-embedding-base_v1 · Hugging Face](https://huggingface.co/maidalun1020/bce-embedding-base_v1)
  
- [Instill VDP - Open source unstructured data ETL for AI first applications | Product Hunt](https://www.producthunt.com/posts/instill-vdp): Versatile Data Pipeline (VDP): An open-source, no-code/low-code solution for quick AI workflow creation. It handles unstructured data, ensuring efficient data connections, flexible pipelines, and smoo...
  

---

## [LAION](https://discord.com/channels/823813159592001537) Discord Summary

- **Legal Eagles Eye Licenses**: `@pseudoterminalx` encouraged a review of LAION's **license** to ensure understanding and compliance, especially in a legal context.
  
- **Where in the World is LAION Aesthetics v2?**: Members, including `@_lazyegg_` and `@thejonasbrothers`, discussed the LAION **aesthetics v2 5+** dataset, with anticipation of a **re-release sans NSFW content**. Meanwhile, `@lulu_59476` expressed interest in the **improved_aesthetics_6.5plus dataset**, although it is currently unavailable.
  
- **Introducing Scaling to Vision**: The **AIM** project, highlighted in a shared [Twitter link](https://fxtwitter.com/_akhaliq/status/1747506197073129924?t=Yd2JyF_VYD2Mf67a4rgbqQ&s=19), showcases a collection of vision models scalable like LLMs, referencing a paper with promising results even at 7 billion parameters, available on [Hugging Face](https://huggingface.co/papers/2401.08541).
  
- **InstantID Unveiled for Personal Image Synthesis**: A new diffusion model named InstantID facilitates personalized image creation from a single facial image, promising better personalization and fidelity detailed in an [arXiv abstract](https://arxiv.org/abs/2401.07519).
  
- **Curiosity Spiked About Unsummarized Paper**: Mention of an [arXiv paper](https://arxiv.org/abs/2401.06951) authored by Jiaheng Liu and Wenhu Chen prompted curiosity but was not discussed in detail.
  

**LAION Channel Summaries**

### ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/) (5 messages):

- **License Reading Encouraged**: `@pseudoterminalx` mentioned that LAION has a **license** which one can have lawyers review.
  
- **Searching for Laion Aesthetics v2**: `@_lazyegg_` inquired about obtaining LAION **aesthetics v2 5+** parquet files, struggling to find them online.
  
- **Anticipation for NSFW-Pruned Release**: `@thejonasbrothers` responded that the requested datasets are likely not available but mentioned that they will be **re-released soon with NSFW content removed**.
  
- **Request for an Unavailable Dataset**: `@lulu_59476` asked for the **improved_aesthetics_6.5plus dataset** from Hugging Face, which is currently not available, and is also open to other versions of the dataset.
  

### ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/) (4 messages):

- **Introducing AIM for Vision**: `@spirit_from_germany` shared a [Twitter link](https://fxtwitter.com/_akhaliq/status/1747506197073129924?t=Yd2JyF_VYD2Mf67a4rgbqQ&s=19) announcing **AIM**, a new collection of vision models scalable like LLMs according to a paper found on [Hugging Face](https://huggingface.co/papers/2401.08541). AIM highlights that visual feature performance scales with both model capacity and data quantity, and promises continued improvement with no performance saturation even at 7 billion parameters.
  
- **InstantID Innovates Personalized Image Creation**: `@thejonasbrothers` posted an [arXiv abstract](https://arxiv.org/abs/2401.07519) detailing InstantID, which tackles the high demands of personalized image synthesis with a diffusion model that works with a single facial image, enhancing image personalization and fidelity without the need for extensive fine-tuning.
  
- **Yet Another arXiv Paper Not Yet Summarized**: `@thejonasbrothers` also shared an [arXiv link](https://arxiv.org/abs/2401.06951) to a paper authored by a team including Jiaheng Liu and Wenhu Chen, but the content and topic of the paper were not discussed in the provided messages.
  
- **Unspecified Twitter Content**: There was a [Twitter link](https://twitter.com/burny_tech/status/1747658128416473214?t=KJfpVBDpuMFfjX6f2OcqpA&s=19) posted by `@spirit_from_germany` with no accompanying context or discussion provided.
  

**Links mentioned**:

- [E^2-LLM: Efficient and Extreme Length Extension of Large Language Models](https://arxiv.org/abs/2401.06951): Typically, training LLMs with long context sizes is computationally expensive, requiring extensive training hours and GPU resources. Existing long-context extension methods usually need additional tra...
  
- [InstantID: Zero-shot Identity-Preserving Generation in Seconds](https://arxiv.org/abs/2401.07519): There has been significant progress in personalized image synthesis with methods such as Textual Inversion, DreamBooth, and LoRA. Yet, their real-world applicability is hindered by high storage demand...
  
- [Tweet from AK (@_akhaliq)](https://fxtwitter.com/_akhaliq/status/1747506197073129924?t=Yd2JyF_VYD2Mf67a4rgbqQ&s=19): Apple presents AIM Scalable Pre-training of Large Autoregressive Image Models paper page: [https://huggingface.co/papers/2401.08541](https://huggingface.co/papers/2401.08541) paper introduces AIM, a collection of vision models pre-trained wi...
  

---

## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord Summary

Only 1 channel had activity, so no need to summarize...

- **Introducing Synthetic Insights Tool**: `@edo4080` announced the creation of a new tool for analyzing, cleaning datasets, and downsizing them while maintaining class proportions. They've made a [working demo](http://demo.syntheticinsights.io) available, featuring datasets like OpenHermes and Ultrachat, and invited collaboration and feedback from the community.
  

---

## [YAIG (a16z Infra)](https://discord.com/channels/958905134119784489) Discord Summary

Only 1 channel had activity, so no need to summarize...

- **Kubernetes Still Leading the Pack**: `@stevekamman` shared a [CNCF blog post](https://www.cncf.io/blog/2024/01/17/a-look-back-at-cncf-linux-foundation-and-top-30-open-source-project-velocity-in-2023/) by Chris Aniszczyk highlighting **Kubernetes'** status as the project with the "largest contributor base."
  
- **OpenTelemetry Gains Speed**: The same [blog post](https://www.cncf.io/blog/2024/01/17/a-look-back-at-cncf-linux-foundation-and-top-30-open-source-project-velocity-in-2023/) notes that **OpenTelemetry** is growing rapidly, remaining the second highest velocity project.
  
- **Backstage Spotlight on Developer Experience**: According to the insights from the CNCF, **Backstage** is gaining traction in addressing issues related to developer experience.
  
- **Steady GitOps Adoption**: The update also shows a continued interest in **GitOps**, with projects like **Argo** and **Flux** maintaining a large community and growing influence in the cloud native ecosystem.
  
**Links mentioned**:

[A look back at CNCF, Linux Foundation, and top 30 open source project velocity in 2023](https://www.cncf.io/blog/2024/01/17/a-look-back-at-cncf-linux-foundation-and-top-30-open-source-project-velocity-in-2023/): By Chris Aniszczyk We have been tracking open source project velocity over the last several years and wanted to share the latest update highlighting open source project velocity over the last 12&#8230...