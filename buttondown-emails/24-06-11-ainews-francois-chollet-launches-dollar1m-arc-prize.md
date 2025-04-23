---
id: 011d8801-60a5-4eab-a7c5-330020cf23b3
title: '[AINews] Francois Chollet launches $1m ARC Prize'
date: '2024-06-11T23:42:03.241872Z'
status: sent
type: public
source: api
metadata: {}
original_slug: ainews-francois-chollet-launches-1m-arc-prize
---

<!-- buttondown-editor-mode: plaintext -->**Nonmemorizable Benchmarks are all you need.**

> AI News for 6/10/2024-6/11/2024.
We checked 7 subreddits, [**384** Twitters](https://twitter.com/i/lists/1585430245762441216) and **30** Discords (**412** channels, and **2774** messages) for you. 
Estimated reading time saved (at 200wpm): **313 minutes**.

In [this weekend's Latent Space pod we talked about test set contamination and the Science of Benchmarking](https://www.latent.space/p/iclr-2024-benchmarks-agents), and today one of the OGs in the field is back with a solution - generate a bunch of pattern-recognition-and-completion benchmarks:

 ![image.png](https://assets.buttondown.email/images/17cec8b3-b977-41dd-a6c5-f5aad99b812a.png?w=960&fit=max) 

You can play with the ARC-AGI puzzles yourself to get a sense for what "easy for humans hard for AI" puzzles look like:

 ![image.png](https://assets.buttondown.email/images/30ef397c-2045-45aa-a3b2-b6a10d41e64b.png?w=960&fit=max) 

This all presumes an opinionated definition of AGI, which the team gracefully provides:

> DEFINING AGI
> 
> Consensus but wrong:
> **AGI is a system that can automate the majority of economically valuable work.**
> 
> Correct:
> **AGI is a system that can efficiently acquire new skills and solve open-ended problems.**
> 
> Definitions are important. We turn them into benchmarks to measure progress toward AGI.
> Without AGI, we will never have systems that can invent and discover alongside humans.

This benchmark is curved to resist the classic 1-2 year saturation cycle that other benchmarks have faced:

 ![image.png](https://assets.buttondown.email/images/aacc028a-1899-4c79-a6c9-1321690a668f.png?w=960&fit=max) 

[The solution guide](https://arcprize.org/guide) offers François' thoughts on promising directions, including Discrete program search, skill acquisition, and hybrid approaches.

Last week the Dwarkesh pod was making waves predicting [AGI in 2027](https://www.dwarkeshpatel.com/p/leopold-aschenbrenner), and today [it's back](https://www.dwarkeshpatel.com/p/francois-chollet) with 
François Chollet asserting that the path we're on won't lead to AGI. Which way, AGI observoor?



---


{% if medium == 'web' %}


**Table of Contents**

[TOC] 

{% else %}

The **Table of Contents** and **Channel Summaries** have been moved to the web version of this email: [{{ email.subject }}]({{ email_url }})!

{% endif %}


---

# AI Twitter Recap

> all recaps done by Claude 3 Opus, best of 4 runs. We are working on clustering and flow engineering with Haiku.

**Apple Integrates ChatGPT into iOS, iPadOS, and macOS**

- **OpenAI partnership**: [@sama](https://twitter.com/sama/status/1800237314360127905) and [@gdb](https://twitter.com/gdb/status/1800237897871921435) announced Apple is partnering with OpenAI to integrate ChatGPT into Apple devices later this year. [@miramurati](https://twitter.com/miramurati/status/1800371566464880663) and [@sama](https://twitter.com/sama/status/1800240506318037208) welcomed Kevin Weil and Sarah Friar to the OpenAI team to support this effort.
- **AI features**: Apple Intelligence will allow AI-powered features across apps, like **summarizing documents, analyzing photos, and interacting with on-screen content**. [@karpathy](https://twitter.com/karpathy/status/1800242310116262150) noted the step-by-step AI integration into the OS, from multimodal I/O to agentic capabilities.
- **Privacy concerns**: Some expressed skepticism about Apple sharing user data with OpenAI, despite Apple's "Private Cloud Compute" guarantees. [@svpino](https://twitter.com/svpino/status/1800449867384258702) detailed the **security measures Apple is taking**, such as on-device processing and differential privacy.

**Reactions to Apple's WWDC AI Announcements**

- **Mixed reactions**: While some were impressed by Apple's AI integration, others felt Apple was behind or relying too much on OpenAI. [@karpathy](https://twitter.com/karpathy/status/1800223553989886447) and [@far__el](https://twitter.com/far__el/status/1800237517649678598) questioned if Apple can **ship capable AI on its own**.
- **Comparison to other models**: Apple's on-device models seem to **outperform other small models**, but their server-side models are still behind GPT-4. [@_philschmid](https://twitter.com/_philschmid/status/1800414656000938439) noted Apple is using adapters and mixed-precision quantization to optimize performance.
- **Integration focus**: Many noted Apple's focus on **deep, frictionless AI integration** rather than model size. [@ClementDelangue](https://twitter.com/ClementDelangue/status/1800231734262337936) praised the push for on-device AI to improve user experience and privacy.

**Advances in AI Research and Applications**

- **Mixture of Agents (MoA)**: [@togethercompute](https://twitter.com/togethercompute/status/1800536106729157054) introduced MoA, which leverages multiple open-source LLMs to **achieve a score of 65.1% on AlpacaEval 2.0**, outperforming GPT-4.
- **AI Reasoning Challenge (ARC)**: [@fchollet](https://twitter.com/fchollet/status/1800577019979411560) and @mikeknoop launched the **$1M ARC Prize** to create an AI that can adapt to novelty and solve reasoning problems, steering the field back towards AGI.
- **Advances in speech and vision**: [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1800496540672508261) showcased **Imagen 3's ability to generate rich images with complex textures**. [@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1800365825703972946) shared Microsoft's VALL-E 2, which achieves human parity in zero-shot text-to-speech.
- **AI applications**: Examples included @adcock_brett's updates on **Figure's robot manufacturing**, @vagabondjack's **$6M seed round for AI-powered financial analysis** at @brightwaveio, and @AravSrinivas's note on **Perplexity being a top referral source for publishers**.

**Memes and Humor**

- [@jxmnop](https://twitter.com/jxmnop/status/1800220386711470249) joked about **LLMs reaching the boundary of human knowledge** with a humorous image.
- [@nearcyan](https://twitter.com/nearcyan/status/1800338495036383357) poked fun at the repeated claims of **"nvidia is done for"** with a meme.
- [@far__el](https://twitter.com/far__el/status/1800245080563011611) quipped "Apple Intelligence is going to be the largest deployment of tool using AI and i'd like someone to speak at @aidotengineer on the design considerations!" in response to @swyx's call for Apple engineers to share insights.

---

# AI Reddit Recap

> Across r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity. Comment crawling works now but has lots to improve!

**AI Developments**

- **Apple partners with OpenAI to integrate GPT-4o into iOS, iPadOS, and macOS**: In /r/OpenAI, Apple unveils "[Apple Intelligence](https://www.reddit.com/r/OpenAI/comments/1dcqp4m/apple_unveils_apple_intelligence_at_wwdc_live/)", a personal AI system built into their operating systems. It integrates OpenAI's GPT-4o and runs on-device, enhancing Siri, writing tools, and enabling image generation.
- **Details on Apple's on-device LLM architecture revealed**: In /r/LocalLLaMA, more specifics are shared about [how Apple Intelligence works under the hood](https://www.reddit.com/r/LocalLLaMA/comments/1dcyo80/apple_intelligence_on_device_llm_details/), using quantized task-specific LoRAs called "adapters", optimized for inference performance, and leveraging a "semantic index" for personal context.
- **AMD releases open-source LLVM compiler for AI processors**: AMD launches [Peano](https://videocardz.com/newz/amd-launches-peano-an-open-source-llvm-compiler-for-ryzen-ai-xdna-and-xdna2-npus), an open-source LLVM compiler for their XDNA and XDNA2 Neural Processing Units (NPUs) used in Ryzen AI processors.
- **Microsoft introduces AI Toolkit for Visual Studio Code**: In /r/LocalLLaMA, Microsoft's new [AI Toolkit extension for VS Code](https://www.reddit.com/r/LocalLLaMA/comments/1dd0k9y/microsoft_ai_toolkit_for_visual_studio_code/) is discussed, which provides a playground and fine-tuning capabilities for various models, with the option to run locally or on Azure.

**Research and Benchmarks**

- **Study finds RLHF reduces LLM creativity and output variety**: A [new research paper](https://www.reddit.com/r/LocalLLaMA/comments/1dd3z73/new_research_shows_rlhf_heavily_reduces_llm/) posted in /r/LocalLLaMA shows that while alignment techniques like RLHF reduce toxic and biased content, they also limit the creativity of large language models, even in contexts unrelated to safety.
- **Benchmarking affordable AWS instances for LLM inference**: In /r/LocalLLaMA, [benchmarks of various AWS instances](https://www.reddit.com/r/LocalLLaMA/comments/1dclmwt/benchmarking_inexpensive_aws_instances/) for running Dolphin-Llama3 reveal the g4dn.xlarge offers the best cost-performance at $0.58/hr, with GPU speed being the key factor. More memory allows for higher token usage in outputs.

**Stable Diffusion 3 and Beyond** 

- **Stable Diffusion 3's importance and standout features highlighted**: A post in /r/StableDiffusion breaks down [why SD3 is a significant step forward](https://www.reddit.com/r/StableDiffusion/comments/1dcuval/the_importance_of_stable_diffusion_3_its_standout/), with its new 16-channel VAE capturing more details, enabling faster training and better low-res results. The multi-modal architecture aligns with LLM research trends and is expected to boost techniques like ControlNets and adapters.

**Miscellaneous**

- **Tip for boosting CPU+RAM inference speed by ~40%**: In /r/LocalLLaMA, a user shares a [trick to increase tokens/sec](https://www.reddit.com/r/LocalLLaMA/comments/1dcpdoc/trick_to_increase_inference_on_cpuram_by_40/) by enabling XMP in BIOS to run RAM at spec bandwidth instead of JEDEC defaults. RAM overclocking could provide further gains but risks instability.
- **Simplifying observability in RAG with BeyondLLM 0.2.1**: A /r/LocalLLaMA post explains how [BeyondLLM 0.2.1 makes it easier to add observability](https://www.reddit.com/r/LocalLLaMA/comments/1dcljqk/observability_in_rag/) to LLM and RAG applications, allowing tracking of metrics like response time, token usage, and API call types.

**Memes and Humor**

- **AI expectations vs reality meme**: An [amusing image](https://i.redd.it/pcfrnkmrdp5d1.jpeg) contrasting the hyped expectations and actual capabilities of AI systems shared in a subreddit.

---

# AI Discord Recap

1. **Apple Debuts with Major AI Innovations**:
   - At WWDC 2024, Apple announced **[Apple Intelligence](https://www.apple.com/newsroom/2024/06/introducing-apple-intelligence-for-iphone-ipad-and-mac/)**, a deeply integrated AI system for iPhones, iPads, and Macs. Key features include **ChatGPT integration into Siri**, AI writing tools, and a new "**Private Cloud Compute**" for secure offloading of complex tasks. Benchmarks showcase Apple's **[on-device and server models](https://x.com/ldjconfirmed/status/1800355063120151031)** performing well in instruction following and writing. However, concerns around **user privacy** and Elon Musk's warning of banning Apple devices at his companies due to OpenAI integration sparked debates.

2.  **Model Compression and Optimization Strategies**: 
   - Engineers actively discussed techniques for **quantizing**, **pruning**, and optimizing large language models like **LLaMA 3** to reduce model size and improve efficiency. Resources like **[LLM-Pruner](https://github.com/horseee/LLM-Pruner)** and **[Sconce](https://github.com/satabios/sconce)** were shared, along with debates on the stability of lower-precision formats like **FP8**. Optimizations like **LoRA**, **8-bit casting**, and **offloading to CPU** were explored to tackle Out-of-Memory (OOM) errors during training.
   - Engineers discussed **overcoming Out of Memory (OOM) errors** using strategies like **offloading optimizer state to CPU** and **bnb 8bit casting** ([VRAM Calculator](https://vram.asmirnov.xyz/)), highlighting techniques like **Low-Rank Adapters (LoRA)**.
   - Community conversations shared insights on **fine-tuning challenges** with practical examples and resources zlike [YouTube tutorial](https://www.youtube.com/watch?v=jyOqtw4ry2w&themeRefresh=1).

3. **Exciting Open-Source and Benchmark News**:
   - **Stable Diffusion 3 (SD3)** excited members, aiming for better voxel art, while comparisons of model platforms like Huggingface and Civitai led to debates on best upscaling methods and availability ([SD3 Announcement](https://glif.app/@Oliveira/glifs/clw44qfbl0000m0zztwqk2tnf)).
   - **Hugging Face** expanded **AutoTrain** with **Unsloth support** ([Announcement](https://x.com/abhi1thakur/status/1800511251145015393)), easing large model fine-tuning with enhanced memory management.
   - **Advancements in Language and Multimodal Models**: The AI community witnessed exciting breakthroughs, including **[LlamaGen](https://arxiv.org/abs/2406.06525)** for autoregressive image generation, **[VALL-E 2](https://arxiv.org/abs/2406.05370)** achieving human parity in zero-shot text-to-speech synthesis, and **[MARS5 TTS](https://github.com/camb-ai/mars5-tts)** from CAMB AI promising higher realism in voice cloning. Discussions explored quantization techniques like **IQ4_xs** and **HQQ** for efficient model deployment, and the potential of **federated learning** for privacy-preserving training.

4. **Community Collaboration on AI Challenges**:
   - Discussions around **Chain of Thought retrieval** in medical applications and techniques for essential model prompt engineering were highlighted in engaging threads ([YouTube tutorial](https://www.youtube.com/watch?v=3eq84KrdTWY&t=665s)).
   - **OpenAccess AI Collective** shared a beginner-friendly [RunPod Axolotl tutorial](https://docs.runpod.io/tutorials/pods/fine-tune-llm-axolotl), simplifying model training processes.

5. **Quantization and Model Deployment Insights**:
   - Exchanges on **4-bit quantization** for Llama 3 and suggestions using **Tensor Parallelism** showcased practical experiences from the AI community ([Quantization Blog](https://stephenpanaro.com/blog/llm-quantization-for-iphone)).
   - **DeepSeek-LLM-7B** model's **LLaMA-based structure** discussed alongside interpretability ([DeepSeek Project](https://github.com/Butanium/llm-latent-language/blob/1054015066a4fa20386765d72601d03aa7ef5887/utils.py#L334)).


---

# PART 1: High level Discord summaries




## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

**Fine-tuning LLMs, Cutting Problems Down to Size**: Engineers share solutions for **Out of Memory (OOM) errors** and discuss fine-tuning processes. There's a consensus on the benefits of offloading optimizer state to CPU or using CUDA managed memory, with techniques like **bnb 8bit casting** and **Low-Rank Adapters (LoRA)** to save memory and enhance performance during training. Valuable resources include a [YouTube video on 8-bit Deep Learning](https://www.youtube.com/watch?v=jyOqtw4ry2w&themeRefresh=1) and a benchmarking tool, [VRAM Calculator](https://vram.asmirnov.xyz/).

**Empathy for Credits Confusion**: Multiple guild members expressed difficulties in receiving promised credits. Missing credits are noted across several platforms, from Modal and OpenAI to Replicate, with appeals for resolution posted in respective channels. Information such as user and org IDs was offered in hopes of expediting support.

**Model Training Troubles and Triumphs**: Members troubleshoot fine-tuning and inference challenges on various platforms, focusing on practical aspects like **dataset preparation**, using existing frameworks like TRL or Axolotl, and handling large model training on limited hardware. On the other side of the coin, positive experiences with deploying **Mistral** on Modal were recounted, endorsing its hot-reload capabilities.

**Reeling in Real-World ML Discussions**: Conversations delved into practical Machine Learning (ML) applications, such as **dynamically swapping LoRAs** and Google's Gemini API for audio processing. The use of *Chain of Thought* reasoning for diagnosis by models like **Llama-3 8B** was also examined, acknowledging flaws in model conclusions.

**Resource Ramp-Up for Rapid Engineering**: The community has been actively sharing resources, including Jeremy Howard's ["A Hackers' Guide to Language Models"](https://www.youtube.com/watch?v=jkrNMKz9pWU) on YouTube and [Excalidraw](https://excalidraw.com/) for making diagrams. Tools like [Sentence Transformers](https://www.sbert.net/docs/sentence_transformer/training_overview.html) are recommended for fine-tuning transformers, highlighting the collaborative spirit in constantly elevating the craft.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Waiting for the Holy Grail, SD3**: Anticipation was high for the release of **Stable Diffusion 3 (SD3)**, with hopes expressed for improved features especially in voxel art, while one user humorously expressed dread over having to sleep before the release.
- **Slow Connections Test Patience**: One member faced a grueling twelve-hour marathon to download **Lora Maker**, due to reaching data cap limits and enduring speeds as sluggish as "50kb/s download from Plytorch.org."
- **Model Platform Showdown**: Discussion arose on the availability of AI models and checkpoints, with platforms like **Huggingface** and **Civitai** under the lens; Civitai takes the lead with a vast selection of Lorases and checkpoints.
- **Up for Debate: Upscaling Techniques**: A technical debate was sparked on whether **upscaling SD1.5** images to 1024 can rival the results of **SDXL** directly trained at 1024x1024 resolution, leading to suggestions to test SDXL's upscaling prowess to even higher resolutions.
- **AMD, Y U No Work With SD?**: Frustration bubbled up from a member struggling to run **Stable Diffusion with an AMD GPU**, culminating in a community nudge towards revisiting installation guides and seeking further technical support.




---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

**July 2024: Anticipated MultiGPU Support for Unsloth AI**  
MultiGPU support for Unsloth AI is highly anticipated for early July 2024, with enterprise-focused Unsloth Pro leading the charge; this will potentially enable more efficient fine-tuning and model training.

**Llama 3 Dabbles in Versatile Fine-Tuning**  
Users explored various tokenizer options for the Llama model, with discussions confirming that tokenizers from services like llama.cpp and Hugging Face are interoperable, and referencing [fine-tuning guidance on YouTube](https://www.youtube.com/watch?v=3eq84KrdTWY&t=665s) for those seeking precise instructions.

**Hugging Face AutoTrain Expands with Unsloth Support**  
[Hugging Face AutoTrain now includes Unsloth support](https://x.com/abhi1thakur/status/1800511251145015393), paving the way for more efficient large language model (LLM) fine-tuning as the AI community showed excitement for advancements that save time and reduce memory usage.

**Innovations in AI Showcased: Therapy AI and MARS5 TTS**  
Emerging tools such as a [therapy AI finetuned on llama 3 8b with Unsloth](https://xtherapy.streamlit.app/) and the newly open-sourced [CAMB AI's MARS5 TTS model](https://github.com/camb-ai/mars5-tts), which promises higher realism in voice cloning, are creating buzz in the community.

**Apple's Hiring: AI Integration Spurs Debate**  
Apple's latest initiative in personalized AI dubbed "Apple Intelligence" was a subject of intense discussion, with the community weighing its potential for language support and the integration of larger models, as reported during WWDC.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

**Deep Learning's Quest for Efficiency**: Members debated the benefits and hurdles of **4-bit quantization** for **Llama 3**, with suggestions like **Tensor Parallelism** providing possible pathways despite their experimental edge. The applicability of various quantization methods including **IQ4_xs** and **HQQ** was highlighted, referencing a blog showcasing their performance on Apple Silicon [LLMs for your iPhone](https://stephenpanaro.com/blog/llm-quantization-for-iphone).

**Seeking Smarter Transformers**: A discussion surfaced on improving **transformer models**, referencing to challenges with learning capabilities that are highlighted in papers like "How Far Can Transformers Reason?" which advocates for *supervised scratchpads*. Additionally, a debate on the usefulness of *influence functions* in models emerged, citing seminal works like [Koh and Liang's influence functions paper](https://arxiv.org/pdf/1703.04730).

**Tackling Text-to-Speech Synthesis**: *VALL-E 2* was mentioned for its exceptional *zero-shot TTS* capabilities, though researchers faced access issues with the [project page](https://web.archive.org/web/20240529183033/https://www.microsoft.com/en-us/research/project/vall-e-x/vall-e-2/). Meanwhile, **LlamaGen's** advances in visual tokenization promise enhanced auto-regressive models and stir discussions about incorporating methods from related works like "Stay on topic with Classifier-Free Guidance".

**Interpreting Multimodal Transformations**: Integration challenges of the **DeepSeek-LLM-7B** model were addressed, with its **LLaMA**-based structure being a focal point. Shared resources include a [GitHub repo](https://github.com/Butanium/llm-latent-language/blob/1054015066a4fa20386765d72601d03aa7ef5887/utils.py#L334) to assist the community in their interpretative efforts and overcome model integration complexities.

**Optimization Strategies for LLM Interaction**: Eleuther introduced chat templating capabilities with the **--apply_chat_template** flag, providing an example of ongoing work to enhance user interaction with language models. There's also a community push to optimize batch API implementations for both local and **OpenAI Batch API applications**, with high-level implementation steps discussed and plans for a future utility to rerun metrics on batch results.



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **GPUs Stirring Hot Tub Fantasies**: An engaging proposition to repurpose **GPU waste heat** for heating a hot tub led to a broader discourse on harnessing data center thermal output. The jest shed light on the potential to transform waste heat into communal heating solutions while providing sustainable data center operational models.
  
- **Cutting Through the Triton Jungle**: Navigational tips for better performance with **Triton** were offered, with common struggles including inferior speed compared to **cuDNN** and complexities with variable printing inside kernels. Preferences were voiced for simpler syntax, avoiding tuples to lessen the development maze.

- **Spanning the Spectrum from Torch to C**++**: A showcase of technical prowess, participants discussed the merits of full graph compilation with `torch.compile`, while others contemplated writing **HIP kernels** for PyTorch, both hinting at the imminent optimization tide. This confluence of conversations also pondered whether **C++20's** concepts could detangle code complexities without back-stepping to C++17.

- **Bitnet's Ones and Zeros Steal the Show**: A thoughtful exchange surfaced around training **1-bit Large Language Models (LLMs)**, with a shared resource from [Microsoft's Unilm GitHub](https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf), outlining the potential efficiency yet acknowledging the stability issues in comparison to **FP16**.

- **Altitudes of LLMs and Compression Techniques**: From analyzing **ThunderKitten's** lackluster **TFLOPS** performance to exploring model compression strategies with [Sconce](https://github.com/satabios/sconce), the community fused their cerebral powers to navigate these complicated terrains. Added to the repository was a benchmarking pull request in [PyTorch's AO repo](https://github.com/pytorch/ao/pull/276), promising accurate performance gauging for **Llama** models.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Concurrency Conundrums and GPU Woes in Mojo**: Engineers debated the adoption of structured concurrency in Mojo's library amidst concerns about its asynchronous capabilities, stressing the importance of heterogeneous hardware like TPU support. A strong sentiment echoed where Mojo succeeds in execution speed, but falls short on hardware acceleration when compared to cost-effective solutions like TPUs.

- **Rapid RNG and Math Mastery on Mojo**: Work on a ported xoshiro PRNG has led to significant speed gains on both laptops and using SIMD, while efforts are underway to bring numpy-equivalent functionality to Mojo through the NuMojo project. Trends show a community push towards expanding numerical computation capabilities and efficiency in Mojo.

- **Addressing Mojo’s Memory Mania**: Controversy sparked over memory management practices in Mojo, with discussions on the need for context managers versus reliance on RAII and the intricacies of UnsafePointers. The debate underlined the community’s commitment to refining Mojo’s ownership and lifetimes paradigms.

- **TPU Territory Tackled**: The MAX engine's potential compatibility with TPUs became a highlight, with community members exploring resources like OpenXLA for guidance on machine learning compilers. Forward-looking discussion touched on MAX engine roadmap updates, including inevitable Nvidia GPU support.

- **Nightly Update Notes Nuances for Mojo**: A freshly released nightly Mojo compiler version `2024.6.1105` brought to light changes including the removal of `SliceNew` and `SIMD.splat`, plus the arrival of `NamedTemporaryFile`. This continuous integration culture within the community exemplifies the lean towards iterative and fast-paced development cycles.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Apple Dips Toes in Personal AI**: Apple announced "Apple Intelligence," incorporating ChatGPT-4o into Siri and writing tools to enhance system-wide user experience while prioritizing privacy, as shared in a [Reddit post](https://www.reddit.com/r/ChatGPT/s/KrhcqUpEuq). 

- **iOS 18 and WWDC 2024 Embrace AI**: With the vision set at WWDC 2024, the new machine learning-powered Photos app in iOS 18 categorizes media more intelligently, coupled with significant AI integrations and software advances across the Apple ecosystem.

- **Is Rabbit R1 a Gadget Gone Wrong?**: Members exchanged views on the legitimacy of the Rabbit R1 device, mentioning its sketchy crypto ties, and speculated about its capabilities with an Android OS, as discussed in a Coffeezilla video.

- **Perplexity AI - Promise Meets Skepticism**: Confusion circulates around Perplexity's Pages and Pro features with desktop/web limitations; meanwhile, Perplexity's academic sourcing accuracy faces scrutiny, with users highlighting Google's NotebookLM as potentially superior.

- **Integration Headaches and Hidden Keys**: Introducing Perplexity AI into custom GPT applications saw roadblocks, prompting discussions on model name updates and safe API practices after an API key was mistakenly exposed, documented at [Perplexity's API guidelines](https://docs.perplexity.ai/discuss/65edc94038fa40001045873c). 





---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **PDF Parsing Pursuits**: Engineers are exploring **local tools** for parsing structured forms in PDFs, with **Langchain** emerging as a suggestion for incorporating local LLMs to extricate fields efficiently.

- **WebUI Woes and Workarounds**: A gap in official **WebUI** support for **LMStudio** has led users to employ the **llama.cpp server** and *[text-generation-webui](https://github.com/oobabooga/text-generation-webui/)* to interact with the tool from remote PCs.

- **California AI Bill Brews Controversy**: **SB 1047** sparks lively debate pertaining to its perceived impact on open-source AI, with fears that it may concentrate AI development among few corporations and encroach on model creators' liabilities indefinitely. [Dan Jeffries' tweet](https://x.com/dan_jeffries1/status/1794740447052525609?s=46) provides insights into the discussion.

- **GPU Upgrades and ROCm Insights**: Engineers discuss upgrading to GPUs with higher VRAM for running large AI models and recommend **AMD's ROCm** as a speedier alternative to **OpenCL** for computational tasks. Concerns with **multi-GPU performance** in LMStudio lead some to alternative solutions like *stable.cpp* and Zluda for **CUDA** sweep-ins on AMD.

- **Model Merging Mastery**: The community has been active in merging models (e.g., **Boptruth-NeuralMonarch-7B**), evaluating new configurations like **Llama3-FiditeNemini-70B**), and tackling operational issues like token limit bugs in **AutogenStudio** with fixes tracked on [GitHub](https://github.com/microsoft/autogen/issues/2050).



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Apple Makes Splashes in AI Waters**: Apple has announced AI integration into its ecosystem and the community is buzzing about the implications for the competition and device performance. Attendees of WWDC 2024 eagerly discussed "Apple Intelligence," a system deeply integrated into Apple's devices, and are examining the available [Apple Foundation Models overview](https://machinelearning.apple.com/research/introducing-apple-foundation-models).

- **Concerns and Debates on AI and Privacy**: Privacy worries surge with AI advancements, with users voicing concerns over potential data misuse, advocating for more secure, on-device AI features instead of relying solely on cloud computing. The discourse reflects the dichotomy where tech enthusiasts express skepticism over cloud and on-premises solutions alike.

- **GPT-4: High Hopes Meet Practical Hiccups**: OpenAI's promise of upcoming ChatGPT updates stirred excitement, yet users report app freezes and confusion over the new voice mode's delayed release. Additionally, developers are frustrated by apparent policy violations in the GPT Store hindering their ability to publish or edit GPTs.

- **Time Management Across Time Zones**: AI engineers are strategizing on how to tackle time zone challenges with the Completions API, weighing options such as using timestamp conversions via external libraries or synthetic data to mitigate risks and enhance precision. Consensus veers towards UTC as the baseline for consistent model output, with user-specific timezone adjustments conducted post-output.

- **Meet Hana AI: Your New Google Chat Teammate**: Hana AI is presented as an AI bot for Google Chat poised to boost team efficiency by handling various productivity tasks and is currently available for free. Engineers can trial and give feedback on the bot, which promises to aid managers and executives, accessible through the [Hana AI website](https://hana.hanabitech.com).



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Quest for Optimal Medical Diagnosis AI Stalled**: No consensus was reached on the best large language model (LLM) for medical diagnosis within the discussions.
  
- **Semantic Leap in CVPR Paper Accessibility**: A new app indexing **CVPR 2024 paper summaries** with **semantic search** capabilities was shared and is accessible [here](https://huggingface.co/spaces/pedrogengo/CVPR2024_search_papers).

- **Tech Hiccups with Civitai Files**: A member encountered `TypeError: argument of type 'NoneCycle' is not iterable` when using `diffusers.StableDiffusionPipeline.from_single_file()` with safetensors files from Civitai.

- **AI Legislation Looms Large Over Open-Source**: A tweet thread criticized the California AI Control Bill for potentially hampering open-source AI development, raising alarm over strict liabilities for model creators.

- **Anime Meets Diffusion with Wuerstchen3**: A user unveiled an anime-finetuned version of SoteDiffusion Wuerstchen3 and provided a useful link to [Fal.AI's documentation](https://fal.ai/models/fal-ai/stable-cascade/sote-diffusion) for API implementation details.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

**Character Codex Unleashed**: Nous Research has unveiled the **[Character Codex dataset](https://huggingface.co/datasets/NousResearch/CharacterCodex)** with data on 15,939 characters from diverse sources like anime, historical archives, and pop icons, now available for download.

**Technical Discussions Ablaze**: Engaging conversations included the potential stifling of creativity by **RLHF** in LLMs, contrasting with the success of companies like Anthropic. The debate also covered **model quantization and pruning methods**, with a [strategy for LLaMA 3 10b](https://github.com/horseee/LLM-Pruner) aiming to trim model sizes smartly.

**Knowledge in Sync**: Members discussed the **Chain of Thought** (CoT) retrieval technique used by CoHere for multi-step output construction and proposed a hybrid retrieval method that might pair **elastic search** with **bm25 + embedding** and web search.

**Code Meets Legislation**: There was a standout critique of **CA SB 1047**, arguing it poses a risk to open-source AI, while a member shared **[Dan Jeffries' insights](https://x.com/dan_jeffries1/status/1794740447052525609?s=46)** on the matter. A counter proposal, **SB 1048**, aimed at safeguarding AI innovation was also mentioned.

**New Rust Library Rigs the Game**: The release of **'Rig'**, an open-source library in Rust for creating LLM-powered applications, was greeted with interest; its [GitHub repo](https://github.com/0xPlaygrounds/rig) is a treasure trove of examples and tools for AI developers.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Apple's AI Game-Changer**: Apple has launched 'Apple Intelligence' at WWDC 2024, integrating ChatGPT with Siri for an improved user interface across iPhones, iPads, and Macs, sparking security concerns and debates. The announcement details were shared [in this article](https://asknews.app/en/stories/Apples-AI-Leap-Sparks-Controversy-Amid-Musks-Security-Concerns).

- **Job Hunt Reality Check**: An aspiring Cohere team member shared their frustration over job rejections despite notable hackathon successes and ML experience, sparking discussions on whether personal referrals trump qualifications.

- **Cohere's Developer Dialogue**: Cohere has introduced Developer Office Hours, a forum for developers to address their concerns and engage directly with the Cohere team. A [reminder for an upcoming session](https://discord.gg/7zjrJmKtBB?event=1248300806600392766) was posted with an invitation to participate.

- **Feedback Flourishes**: Members expressed high satisfaction with the new Developer Office Hours format offered by Cohere, complementing the team for fostering an engaging and relaxed environment.

- **Engage with Expertise**: Cohere encourages member engagement and offers an opportunity for developers to expand their knowledge and troubleshoot with the team through the Developer Office Hours. The next session is scheduled for June 11, 1:00 PM ET, and accessible via this [Discord Event](https://discord.gg/7zjrJmKtBB?event=1248300806600392766).



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Apple Goes All-In on AI Integration**: Apple announced the integration of AI throughout their OS, focusing on multimodal I/O and user experience while maintaining privacy standards. For AI tasks, they introduced "Private Cloud Compute," a secure system for offloading computation to the cloud without compromising user privacy.

- **ChatGPR Finds a New Home**: Partnerships were announced between Apple and OpenAI to bring ChatGPT to iOS, iPadOS, and macOS, signaling a significant move towards AI-enabled operating systems. This would bring conversational AI directly into the hands of Apple users later this year.

- **Mistral Rides the Funding Wave**: AI startup Mistral secured a €600M Series B funding for global expansion, a testament to investors' faith in the future of artificial intelligence. The round follows a surge of investments in the AI space, highlighting the market's growth potential.

- **PostgreSQL's AI Performance Edges Out Pinecone**: PostgreSQL's new open-source extension, "pgvectorscale," is hailed for outperforming Pinecone in AI applications, promising better performance and cost efficiency. This marks a significant development in the database technologies supporting AI workloads.

- **LLMs in the Real World**: Mike Conover and Vagabond Jack featured on the Latent Space podcast, sharing their experiences with deploying Large Language Models (LLMs) in production and AI Engineering strategies in the finance sector. Discussions center around practical considerations and strategies for leveraging LLMs effectively in industry contexts.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Advanced Knowledge Graph Bait**: A special workshop focusing on "advanced knowledge graph RAG" is scheduled with Tomaz Bratanic from Neo4j, aiming to explore LlamaIndex property graph abstractions. Engineers are encouraged to [register for the event](https://lu.ma/kqxmbuou) taking place on Thursday at 9am PT.

- **Parisian AI Rendezvous**: @hexapode will showcase a live demo at the [Paris Local & Open-Source AI Developer meetup](https://t.co/5GLV08cGFa) featuring several prominent companies including Koyeb, Giskard, Red Hat, and Docker at Station F in Paris on 20th June at 6:00pm, with opportunities for others to demo their work by applying [here](https://forms.gle/YMXvYCVhuuppTWTp7).

- **LlamaIndex Snafus and Workarounds**: Users are seeking help with the integration of various query engines and LLM pipelines, such as combining SQL, Vector Search, and Image Search using LlamaIndex and querying a vector database with potential OpenAI Chat Completion fallbacks. For projects involving SQL db retrieval and analysis with Llama 3, exploring text-to-SQL pipelines and consulting [LlamaIndex’s advanced guides](https://docs.llamaindex.ai/en/stable/examples/pipeline/query_pipeline_sql/?h=text2) is recommended.

- **Berkeley Brainstorming**: A UC Berkeley research team is exploring the terrain of custom RAG systems, seeking input from experienced engineers to navigate the complexity of building, deploying, and maintaining such systems.

- **The Need for Speed in Sparse Vector Generation**: Generating and uploading sparse vectors in hybrid mode with Qdrant and LlamaIndex is too slow for some users, with suggestions hinting at leveraging GPUs locally or using an API to hasten the process.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **LAION Caught in Controversy**: The LAION dataset was featured on Brazilian TV receiving criticism; the issue stems from a claim by Human Rights Watch that AI tools are misusing children's online personal photos, as discussed [here](https://www.hrw.org/news/2024/06/10/brazil-childrens-personal-photos-misused-power-ai-tools).

- **Privacy and Internet Literacy Debated**: Engineers expressed concerns over widespread misunderstanding of data privacy on the internet, touching on the grave problems caused by billions of users lacking knowledge on the subject.

- **LlamaGen Moves Image Generation Forward**: The announced LlamaGen model demonstrates a significant step in image generation, leveraging language model techniques for visual content creation, as detailed in their [research paper](https://arxiv.org/abs/2406.06525).

- **CAMB AI's MARS5 Goes Open Source**: The TTS model, MARS5, developed by CAMB AI, has been made open source for community use, with a Reddit post inviting feedback and further technical discussion available [on this thread](https://www.reddit.com/r/CAMB_AI/comments/1day7ta/introducing_mars5_opensource_insanely_prosodic/).

- **Safety in Visual Data Sets**: The LlavaGuard project, detailed [here](https://ml-research.github.io/human-centered-genai/projects/llavaguard/index.html), proposed a model aimed at increasing safety and ethical compliance in visual dataset annotations.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Apple Intelligence Divides Opinions**: Engineers mixed in their feedback on OpenAI's collaboration with Apple, suggesting the integration into **Apple Intelligence** may be superficial; however, user privacy highlighted in the official announcement, despite rumor and skepticism ([Read more](https://www.apple.com/newsroom/2024/06/introducing-apple-intelligence-for-iphone-ipad-and-mac/)). Comparative benchmarks for Apple's on-device and server models aroused curiosity about their performance against peers.

- **Creating Clear Distinctions**: Apple's strategic approach to separate **Apple Intelligence** from **Siri** has sparked dialogue on potential impacts on user adoption and perceptions of the new system's capabilities.

- **Tech Community Anticipates Key Interview**: The forthcoming interview of **François Chollet** by **Dwarkesh Patel** has engineers eager for a possible shift in the AGI timeline debate, highlighting the importance of informed questioning rooted in Chollet’s research on intelligence measures.

- **TRL Implementation Debated**: Caution was raised about implementing TRL, citing the technology as *"unproven"*. One member's plan to submit a Pull Request (PR) for TRL received active encouragement and a review offer from another community member.

- **Support in Community Contributions**: The spirit of collaboration is evident as a member plans to contribute to **TRL** and receives a pledge for review, showcasing the guild’s culture of mutual support and knowledge sharing.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Apple Intelligence on the AI Radar**: Community showed interest in the potential integration of Open Interpreter with Apple's privacy-centric AI capabilities outlined on the [Apple Intelligence page](https://www.apple.com/apple-intelligence/). This could lead to leveraging the developer API to enhance AI functionalities across Apple devices.

- **SB 1047 in the Line of Fire**: [Dan Jeffries criticized](https://x.com/dan_jeffries1/status/1794740447052525609?s=46) the California AI Control and Centralization Bill (SB 1047), introduced by Dan Hendyrcks, for its centralized control over AI and the threat it poses to open source AI innovation.

- **Arduino IDE Complications on Mac M1 Resolved**: An issue with Arduino IDE on Mac M1 chips was addressed through a fix found in a [GitHub pull request](https://github.com/lacamera/ESPAsyncWebServer/pull/2/files), but led to additional problems with the Wi-Fi setup on device restarts.

- **Linux as an Open Interpreter Haven**: Debate among members highlighted consideration of prioritizing Linux for future Open Interpreter developments, aiming to provide AI-assisted tools independent of major operating systems like Apple and Microsoft.

- **Personal Assistant that Remembers**: Work on enhancing Open Interpreter with a skilled prompting system that can store, search, and retrieve information like a personal assistant was shared, spotlighting innovation in creating memory retention for AI systems.

- **Killian's Insights Captured**: A noteworthy discussion followed Killian's recent talk, which was instrumental in casting a spotlight on pertinent AI topics among community members. The recording can be found [here for further review](https://discord.com/channels/1146610656779440188/1147665339266650133/1248858812761509938).



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **Tagging Troubles with LangChain**: Engineers noted that prompts are ignored with the `create_tagging_chain()` function in LangChain, causing frustration as no solution has been offered yet.
- **Collaborative Call for RAG Development Insights**: UC Berkeley team members are actively seeking discussions with engineers experienced in **Retrieval-Augmented Generation (RAG)** systems to share challenges faced in development and deployment.
- **LangGraph vs LangChain**: Interest was shown in understanding the advantages of using **LangGraph** over the classic **LangChain** setup, particularly regarding the execution of controlled scripts within LangGraph.
- **Awaiting ONNX and LangChain Alliance**: There was curiosity about potential compatibility between ONNX and LangChain; however, the conversation didn't progress into a detailed discussion.
- **Streamlined Large Dataset Processing Via OpenAI**: A comprehensive guide for processing large datasets with the OpenAI API was shared, focusing on best practices like setting environment variables, anonymizing data, and efficient data retrieval with Elasticsearch and Milvus. Related documentation and GitHub issue links were provided for reference.





---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Newcomer Encounters Permission Puzzle**: A new member eager to participate in **tinygrad** development found themselves permission-locked from the bounties channel, preventing them from working on the **AMX support bounty**. George Hotz resolved the confusion, stating that one must "Become a purple" to gain the necessary access for contribution.
  
- **George Plays Gatekeeper**: In response to questions about **AMX support** in **tinygrad**, George Hotz hinted that deeper engagement with the community's documentation is required before tackling such tasks, referencing the need to read a specific questions document.

- **A Classic Mix-Up**: A documentation mishap occurred when a new member cited the wrong guide, referring to *["How To Ask Questions The Smart Way"](http://www.catb.org/~esr/faqs/smart-questions.html)*, leading to a humorous "chicken and egg problem" moment with George Hotz.

- **Back to the Drawing Board**: After the back-and-forth, the new contributor decided to take a step back and delve deeper into the **tinygrad** codebase before returning with more precise questions, showcasing the complexity and dedication required for contributing to such a project.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Speedy Service with OpenRouter**: OpenRouter has tackled latency issues by utilizing **Vercel Edge** and **Cloudflare Edge** networks, ensuring that server nodes are strategically positioned close to users for faster response times.
- **Provider Preference in the Pipeline**: Although the OpenRouter playground currently lacks a feature for users to select their preferred API provider, plans to implement this capability have been confirmed.
- **API Provider Choices for the Tech-Savvy**: Users can bypass the lack of direct provider selection in the OpenRouter playground by using the API; a guide to this workaround is accessible in the [OpenRouter documentation](https://openrouter.ai/docs/provider-routing).



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **ShareGPT's Training Veil**: When training, **ShareGPT** does not "see" its own converted prompt format, ensuring a clean training process.
- **Apple's AI Struts Its Stuff**: [Benchmarks are in](https://x.com/ldjconfirmed/status/1800355063120151031) for **Apple's new on-device and server models**, showcasing their prowess in instruction following and writing, with comparisons to other leading models.
- **Rakuten Models Storm the Scene**: **Rakuten's AI team** has released a set of large language models that perform exceptionally in Japanese, based on **[Mistral-7B](https://mistral.ai/news/announcing-mistral-7b/)** and available under a commercial license, sparking an optimistic buzz among community members.
- **JSON Joy Ripples Through Conversation**: Engineers had a light-hearted moment appreciating a model's ability to respond in JSON, capturing a mix of amusement and technical appreciation for the model's capability.
- **Fine-Tuning Made Simpler with Axolotl**: AI practitioners are guided by a [new tutorial for fine-tuning](https://docs.runpod.io/tutorials/pods/fine-tune-llm-axolotl) on **RunPod**, which outlines a streamlined process for fine-tuning large language models with helpful YAML examples across various model families.



---



## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

- **Calm Before the Coding Storm**: Vincent Warmerdam recommends [calmcode.io](https://calmcode.io) for training models, with users acknowledging the site for its helpful content on model training strategies and techniques.
  
- **RAGged but Right**: A [Stack Overflow blog post](https://stackoverflow.blog/2024/06/06/breaking-up-is-hard-to-do-chunking-in-rag-applications/) details chunking strategies for RAG (retrieval-augmented generation) implementations, stressing the role of text embeddings to accurately map source text into the semantic fabric of LLMs, enhancing the grounding in source data.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Clarity on TRL's KL Plots for DPO**: There is no direct plotting of Kullback–Leibler (KL) divergence for the Dominant Policy Optimization (DPO) implementation, but such KL plots do exist within the **Trust Region Learning (TRL)**'s Proximal Policy Optimization (PPO) trainer. The KL plots can be found in the PPO trainer's code, as pointed out in [TRL's GitHub repository](https://github.com/huggingface/trl/blob/34ebc4ccaf376c862a081ff4bb0b7e502b17b2fb/trl/trainer/ppo_trainer.py#L1133).



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

**AI Community Unites at Mosaic Event**: Meet **Chip Huyen** in person at the [Mosaic event at Databricks Summit](https://mosaicx.events/events/june-10-2024-san-francisco-ca?events=main) for networking with AI and ML experts. The gathering is set for **June 10, 2024**, in San Francisco.



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

Given the lack of substantial discussion points and insufficient context in the provided snippet, there are no significant technical or detailed discussions to summarize for an engineer audience.



---


The **LLM Perf Enthusiasts AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI Stack Devs (Yoko Li) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **DiscoResearch Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **YAIG (a16z Infra) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1249830188628578495)** (37 messages🔥): 

- **Heuristics for model size in fine-tuning**: A member raised a general question about choosing model size for fine-tuning based on task complexity and mentioned the difficulty of extensive evaluations for rapid prototyping. They inquired if experienced users develop a sense of model capabilities over time.
  
- **Karpathy's impactful videos**: Discussion highlights the educational value of Andrej Karpathy’s videos, with one member sharing [a full implementation repository](https://github.com/gao-hongnan/omniverse/tree/main/omnivault/transformer) and supplementary notes on GPT from Karpathy's earlier tutorials.

- **NCCL timeout issue**: A user shared an error log indicating a timeout at NCCL work, seeking advice from the community. The log highlights complications in the ProcessGroupNCCL operations.

- **Gorilla Project shines in tool use and API generation**: The **Gorilla** project was mentioned as an interesting case for fine-tuning models to improve tool use and API generation. They highlighted the project’s resources including the [GoEx runtime](https://goex.gorilla-llm.com/index) and [leaderboards](https://gorilla.cs.berkeley.edu/leaderboard.html), and shared a [YouTube video](https://www.youtube.com/live/WAvO8FTDJ8M?si=dR_9-Q5hLxMPvRCS) outlining the project.

- **LLMs vs traditional ML/DL**: A discussion on transitioning to LLMs from traditional ML/DL pointed out the importance of leveraging existing models before starting from scratch. Core principles from ML/DL like data prep, EDA, and model pipelines remain largely relevant in the LLM lifecycle.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://gist.github.com/sroecker/feaa61ea69182cb7ae1c9328b755786a">A script to caption datikz graphs with Moondream</a>: A script to caption datikz graphs with Moondream. GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://github.com/writer/writer-framework">GitHub - writer/writer-framework: No-code in the front, Python in the back. An open-source framework for creating data apps.</a>: No-code in the front, Python in the back. An open-source framework for creating data apps. - writer/writer-framework</li><li><a href="https://brain.nehiljain.com/posts/how-i-improved-my-prompting-for-budget-categorization/">How I improved my prompting for Budget Categorization</a>: no description found</li><li><a href="https://huggingface.co/datasets/sroecker/datikz-v2-moondream-caption-test2/viewer?row=18">sroecker/datikz-v2-moondream-caption-test2 · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/sroecker/datikz-v2-moondream-caption-test">sroecker/datikz-v2-moondream-caption-test · Datasets at Hugging Face</a>: no description found</li><li><a href="https://Gorilla.cs.berkeley.edu">Gorilla</a>: no description found</li><li><a href="https://www.youtube.com/live/WAvO8FTDJ8M?si=dR_9-Q5hLxMPvRCS">Teaching LLMs to Use Tools at Scale - Shishir Patil | Stanford MLSys #98</a>: Episode 98 of the Stanford MLSys Seminar Series!Teaching LLMs to Use Tools at ScaleSpeaker: Shishir PatilBio:Shishir G. Patil is a CS PhD student at UC Berke...</li><li><a href="https://www.gaohongnan.com/influential/generative_pretrained_transformer/03_concept.html#autoregressive-self-supervised-learning-paradigm)">The Concept of Generative Pre-trained Transformers (GPT) &#8212; Omniverse</a>: no description found</li><li><a href="https://archive.ph/v8lN0">Is Slop A.I.&#x2019;s Answer to Spam? A Phrase Emerges for Bad Search. - The&#x2026;</a>: no description found
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[🟩-modal](https://discord.com/channels/1238365980128706560/1241044231829848125/1249829935250673784)** (44 messages🔥): 

- **Clarification on Bonus Credits**: Members inquired about the distribution of bonus credits for using Modal. It was clarified that the second disbursal of credits would happen a bit after midnight UTC on Tuesday.

- **"Simple Scalable Serverless Services" Slide Share**: A user shared a [Google Slides link](https://docs.google.com/presentation/d/14uDnzd06j9i0zAQ3lTmB7QHBSO45BIsVGUZBZ3HKxGo/edit#slide=id.g2c7588f453b_0_272) for Charles' presentation on "Mastering LLMs - Simple Scalable Serverless Services".

- **GitHub Projects and Repositories**: Multiple GitHub repositories were shared, including [charlesfrye/minimodal](https://github.com/charlesfrye/minimodal) and [awesome-modal](https://github.com/modal-labs/awesome-modal), with additional contribution guidance and project links.

- **Discussion on Cost Management**: Users discussed best practices around preventing cost blowups with serverless services like AWS S3 or Vercel. Recommendations included setting high-ball cost estimates and load balancer limits to prevent unexpected expenses.

- **Notebooks Feature in Modal**: Inquired about the development of the Notebooks feature within Modal, Charles confirmed it works with some limitations, suggesting users raise issues if they encounter problems. An example project link on [mistral-finetune-modal](https://github.com/andresckamilo/mistral-finetune-modal/blob/main/src/main.py) was shared to illustrate its use.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/modal-labs/awesome-modal/pull/1">Add indic-subtitler project by kurianbenoy · Pull Request #1 · modal-labs/awesome-modal</a>: IndicSubtitler Github Project Website</li><li><a href="https://github.com/andresckamilo/mistral-finetune-modal/blob/main/src/main.py">mistral-finetune-modal/src/main.py at main · andresckamilo/mistral-finetune-modal</a>: Contribute to andresckamilo/mistral-finetune-modal development by creating an account on GitHub.</li><li><a href="https://modal.com/docs/guide/notebooks">Jupyter notebooks</a>: You can use the Modal client library in notebook environments like Jupyter! Just import modal and use as normal. However, there are some limitations when using Modal within notebooks.</li><li><a href="https://github.com/modal-labs/awesome-modal">GitHub - modal-labs/awesome-modal: A curated list of amazingly awesome Modal applications, demos, and shiny things. Inspired by awesome-php.</a>: A curated list of amazingly awesome Modal applications, demos, and shiny things. Inspired by awesome-php. - modal-labs/awesome-modal</li><li><a href="https://github.com/charlesfrye/minimodal">GitHub - charlesfrye/minimodal: A miniature version of Modal</a>: A miniature version of Modal. Contribute to charlesfrye/minimodal development by creating an account on GitHub.</li><li><a href="https://docs.google.com/presentation/d/14uDnzd06j9i0zAQ3lTmB7QHBSO45BIsVGUZBZ3HKxGo/edit#slide=id.g2c7588f453b_0_272">Mastering LLMs - Simple Scalable Serverless Services</a>: Simple Scalable Serverless Services bit.ly/mastering-llms-ssss
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[hugging-face](https://discord.com/channels/1238365980128706560/1241141471814488115/1250111760795439156)** (4 messages): 

- **Spending Hugging Face Credits**: A member inquired about effective ways to use Hugging Face credits, beyond inference endpoints. Other members suggested using credits for **Spaces with GPUs** and **AutoTrain**, which enables **automatic training** and fast deployment of custom machine learning models by simply uploading data, with tasks like LLM finetuning, image classification, and text classification.
- **New Form Inquiry**: Another member asked if the new form is available yet, but no reply was documented.

**Link mentioned**: <a href="https://hf.co/autotrain">AutoTrain – Hugging Face</a>: no description found

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[replicate](https://discord.com/channels/1238365980128706560/1241163904927666287/1250129165429243986)** (10 messages🔥): 

- **Credits Troubleshooting Inquiries on Replicate**: Multiple users reported not receiving their Replicate credits despite following given instructions. A member of the support team responded, asking for direct messages with usernames and emails to resolve the issue.
  
- **Replicate Credits Without Billing Setup**: A user inquired if billing setup on Replicate was necessary to receive credits. The response clarified that billing setup is not required to get credits.

- **Feedback on Training and Deploying OSS Tool Calling Models**: A member shared their experience working with a starter repo for training and deploying OSS tool calling models. They sought feedback on whether their setup was correct and expressed interest in further discussion.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[langsmith](https://discord.com/channels/1238365980128706560/1241167367040405544/1249810351667675266)** (7 messages): 

- **Email communication on credits**: A user mentioned they had sent an email about their credits. Later, another user confirmed an email response regarding this matter.
- **Adding a payment method not initially required**: It's explained that *"Accessing your credits requires a valid payment method on file,"* but users don't need to have billing set up when filling out the credits form. [Reference link provided](https://discord.com/channels/1238365980128706560/1241167367040405544/1247687054826012693).
- **Org ID missing from form**: A user received assistance with credits even though they had left the org ID blank on their form submission. *"I've gone in and added these credits for you."*
- **Credits issue resolved with email**: A user was asked to provide the email they used for the credit form via DM or email to jess@langchain.dev. This was part of resolving the credit addition issue.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[berryman_prompt_workshop](https://discord.com/channels/1238365980128706560/1242223275463938221/1250157367056662728)** (3 messages): 

- **Chain of Thought leads to flawed conclusions**: One member discussed using **Chain of Thought (CoT)** reasoning for diagnosis steps, but their model (**Llama-3 8B**) sometimes arrives at incorrect conclusions. An example given was the model incorrectly stating "*This is a violation of the rules*" when diagnosing patient age within an interval.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[whitaker_napkin_math](https://discord.com/channels/1238365980128706560/1242223332695478332/1250123249715646585)** (2 messages): 

- **Estimating VRAM Consumption for DPO Training Is Tricky**: A member raises concerns about the fluctuating VRAM consumption during DPO training, seeking estimation methods to avoid out-of-memory (OOM) errors. Another member suggests starting with the longest sequences to prevent unexpected VRAM spikes mid-training.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[workshop-4](https://discord.com/channels/1238365980128706560/1242223495673286737/1250020611938324510)** (9 messages🔥): 

- **Apple jumps into LoRA swapping game**: A member pointed out a new technique from Apple on "dynamically swapping out LoRA's", getting peers excited and curious about its similarity to S-LoRA. Another asked for resources to implement **dynamic specialized LoRA adapters** based on query type, leading to suggestions like "Lorax" and mentions of related work like **CBTM for LoRAs** and **semantic similarity over prompt and dataset per task**. 
- **Workshop insights predate Apple’s announcement**: Another member highlighted how this concept of dynamic LoRA swapping discussed by Apple was covered in a workshop, albeit for cloud applications, making the on-device adaptation exciting and ahead of time. Recounting insights from "Travis," they appreciated the foresight and detailed understanding the workshop provided.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[clavie_beyond_ragbasics](https://discord.com/channels/1238365980128706560/1242223963346698250/1249873584231551076)** (129 messages🔥🔥): 

- **Jeremy Howard shares Hacker's Guide to Language Models**: Jeremy Howard recommended a YouTube video titled ["A Hackers' Guide to Language Models"](https://www.youtube.com/watch?v=jkrNMKz9pWU). His video is described as "deeply informative," covering the comprehensive utility of modern language models.
- **Ben Clavié's Resources on Reranking and NER**: Ben shared multiple valuable resources including a [GitHub link for rerankers](https://github.com/AnswerDotAI/rerankers) and detailed explanations on GIiNER, a robust model for zero-shot entity recognition. He highlighted its capability to handle in-house jargon and specific categories.
- **Discussion on Cosine Distance vs. L2 Distance**: Members discussed the merits between using Cosine Distance and normalized Euclidean Distance (L2) for vector search in RAG applications. They concluded that "cosine distance is equal to normalized Euclidian distance."
- **Sharing of Additional Tools and Libraries**: Members shared tools like [Excalidraw](https://excalidraw.com/) for making block diagrams and various resources for fine-tuning transformers such as [Sentence Transformers](https://www.sbert.net/docs/sentence_transformer/training_overview.html). 
- **Challenges with Video Hosting on Maven**: There were issues with video playback of Ben Clavié's talk on Maven, apparently due to Zoom's transcription process. Efforts to resolve this were ongoing, aiming to make the materials accessible again.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://stats.stackexchange.com/questions/71614/distance-measure-of-angles-between-two-vectors-taking-magnitude-into-account">distance measure of angles between two vectors, taking magnitude into account</a>: Suppose I have two vectors, v1 and v2, from which I can calculate the angle between these two vectors as a measure of their &quot;distance&quot;, using the arccos function, say. For example:&#xA;&#xA;...</li><li><a href="https://www.elastic.co/guide/en/elasticsearch/reference/current/dense-vector.html">Dense vector field type | Elasticsearch Guide [8.14] | Elastic</a>: no description found</li><li><a href="https://github.com/urchade/GLiNER">GitHub - urchade/GLiNER: Generalist and Lightweight Model for Named Entity Recognition (Extract any entity types from texts) @ NAACL 2024</a>: Generalist and Lightweight Model for Named Entity Recognition (Extract any entity types from texts) @ NAACL 2024 - urchade/GLiNER</li><li><a href="https://sbert.net/">SentenceTransformers Documentation &mdash; Sentence Transformers  documentation</a>: no description found</li><li><a href="https://excalidraw.com/">Excalidraw — Collaborative whiteboarding made easy</a>: Excalidraw is a virtual collaborative whiteboard tool that lets you easily sketch diagrams that have a hand-drawn feel to them.</li><li><a href="https://www.youtube.com/watch?v=jkrNMKz9pWU">A Hackers&#39; Guide to Language Models</a>: In this deeply informative video, Jeremy Howard, co-founder of fast.ai and creator of the ULMFiT approach on which all modern language models (LMs) are based...</li><li><a href="https://tenor.com/view/clem-fandango-steven-toast-toast-of-london-yes-i-can-hear-you-clem-fandango-gif-9211791307522605321">Clem Fandango Steven Toast GIF - Clem Fandango Steven Toast Toast of London - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/AnswerDotAI/rerankers">GitHub - AnswerDotAI/rerankers</a>: Contribute to AnswerDotAI/rerankers development by creating an account on GitHub.</li><li><a href="https://gist.github.com/bclavie/f7b041328615d52cf5c0a9caaf03fd5e">rag_mvp.py</a>: GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://www.sbert.net/docs/sentence_transformer/training_overview.html">Training Overview &mdash; Sentence Transformers  documentation</a>: no description found</li><li><a href="https://x.com/bclavie">Tweet from undefined</a>: no description found</li><li><a href="https://github.com/bclavie/RAGatouille">GitHub - bclavie/RAGatouille: Easily use and train state of the art late-interaction retrieval methods (ColBERT) in any RAG pipeline. Designed for modularity and ease-of-use, backed by research.</a>: Easily use and train state of the art late-interaction retrieval methods (ColBERT) in any RAG pipeline. Designed for modularity and ease-of-use, backed by research. - bclavie/RAGatouille</li><li><a href="https://arxiv.org/abs/2311.08526">GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer</a>: Named Entity Recognition (NER) is essential in various Natural Language Processing (NLP) applications. Traditional NER models are effective but limited to a set of predefined entity types. In contrast...
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[jason_improving_rag](https://discord.com/channels/1238365980128706560/1242224099548332132/1249815782934974605)** (1 messages): 

- **Exploring faster entity extraction over function calling**: The user discusses integrating categories as metadata to potentially improve a reranker. They chose an entity extraction + router model approach instead of function calling due to complexity and speed benefits.
- **Seeking insights on model training specifics**: The other user seeks details on the complexity and training specifics of models using function calling. They ask about the sample size needed, the number of functions prepared, and specifics of product data complexity, such as relationships like "is_accessory_of" or "bought_together."


  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[jeremy_python_llms](https://discord.com/channels/1238365980128706560/1242224309548875917/1249869859828011071)** (1 messages): 

- **Excitement for Fasthtml**: A user expressed their excitement for **fasthtml**, highlighting their struggles with scaling Streamlit apps into more complicated applications. They mentioned that fasthtml might save them from having to learn Typescript.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[saroufimxu_slaying_ooms](https://discord.com/channels/1238365980128706560/1242224552415596554/1249805912080650241)** (148 messages🔥🔥): 

- **Offload Optimizer State to Avoid OOM**: Users discussed strategies to **offload optimizer state to CPU** or CUDA managed memory to prevent Out of Memory (OOM) errors in model training. They emphasized the trade-off in performance unpredictability and the importance of fused optimizers to speed up `optimizer.step` operations.

- **Shared Insights on Efficient Deep Learning Techniques**: Several users exchanged insights on advanced optimization techniques like **bnb 8bit casting** and **LoRA**. They explored how these techniques save memory and enhance model performance during training.

- **Extensive Resource Sharing**: Members shared numerous resources on model training optimization, including links to **Profetto UI**, **torch profiler**, and various GitHub repositories. Specific URLs included a [YouTube video on 8-bit Deep Learning](https://www.youtube.com/watch?v=jyOqtw4ry2w&themeRefresh=1) and a [Google Drive](https://drive.google.com/drive/u/3/folders/1HmGNC4v4L5nXhtdDMVCpUBrme1ELp-2C) with related slides and traces.

- **Enthusiastic Discussion on Model Quantization and FSDP**: Users actively discussed the benefits and complexities of quantization, especially with tools like FSDP2, **emphasizing efficient memory management**. The conversation highlighted the practical implementations and challenges of working with NF4 tensors and large model training.

- **Interactive and Appreciative Community Engagement**: The chat was filled with supportive interactions, humor, and praise, particularly for the informative talks and materials shared by specific members. The community expressed gratitude for detailed presentations, with one member humorously adding, *"Memes are the best method of information dissemination to a crowd like us."*
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://perfetto.dev/">Perfetto - System profiling, app tracing and trace analysis</a>: no description found</li><li><a href="https://vast.ai/docs/autoscaler/introduction">Overview | Vast.ai</a>: no description found</li><li><a href="https://tenor.com/view/im-pretending-i-know-what-youre-talking-about-ahmed-aldoori-i-have-no-idea-faking-it-pretending-gif-18453815">Im Pretending I Know What Youre Talking About Ahmed Aldoori GIF - Im Pretending I Know What Youre Talking About Ahmed Aldoori I Have No Idea - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/xHx8.gif">I Know Some Of These Words Mhmm GIF - I Know Some Of These Words Mhmm Clueless - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html#using-tracing-functionality">PyTorch Profiler &mdash; PyTorch Tutorials 2.3.0+cu121 documentation</a>: no description found</li><li><a href="https://tenor.com/view/thanks-bow-thank-you-sign-of-respect-gif-4807966236937524301">Thanks Bow GIF - Thanks Bow Thank You - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.youtube.com/watch?v=SKV6kDk1s94">Lecture 16: On Hands Profiling</a>: no description found</li><li><a href="https://drive.google.com/">Google Drive: Sign-in</a>: no description found</li><li><a href="https://discord.gg/RfcRWeNs">Join the llm-fine-tuning Discord Server!</a>: Check out the llm-fine-tuning community on Discord - hang out with 1895 other members and enjoy free voice and text chat.</li><li><a href="https://www.youtube.com/watch?v=jyOqtw4ry2w&themeRefresh=1">8-bit Methods for Efficient Deep Learning with Tim Dettmers</a>: Tim Dettmers (PhD candidate, University of Washington) presents &quot;8-bit Methods for Efficient Deep Learning&quot; in this Cohere For AI Technical Talk.Abstract: La...</li><li><a href="https://drive.google.com/drive/u/3/folders/1HmGNC4v4L5nXhtdDMVCpUBrme1ELp-2C">Slaying OOMs traces – Google Drive</a>: no description found</li><li><a href="https://github.com/yandex/YaFSDP">GitHub - yandex/YaFSDP: YaFSDP: Yet another Fully Sharded Data Parallel</a>: YaFSDP: Yet another Fully Sharded Data Parallel. Contribute to yandex/YaFSDP development by creating an account on GitHub.</li><li><a href="https://asmirnov.xyz/vram">Breaking down GPU VRAM consumption</a>: no description found</li><li><a href="https://drive.google.com/drive/u/0/folders/1HmGNC4v4L5nXhtdDMVCpUBrme1ELp-2C">Slaying OOMs traces – Google Drive</a>: no description found</li><li><a href="https://github.com/pytorch/torchtune">GitHub - pytorch/torchtune: A Native-PyTorch Library for LLM Fine-tuning</a>: A Native-PyTorch Library for LLM Fine-tuning. Contribute to pytorch/torchtune development by creating an account on GitHub.</li><li><a href="https://www.answer.ai/posts/2024-03-06-fsdp-qlora.html">Answer.AI - You can now train a 70b language model at home</a>: We’re releasing an open source system, based on FSDP and QLoRA, that can train a 70b model on two 24GB GPUs.</li><li><a href="https://vram.asmirnov.xyz/">VRAM Calculator</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/pytorch/torchtitan">GitHub - pytorch/torchtitan: A native PyTorch Library for large model training</a>: A native PyTorch Library for large model training. Contribute to pytorch/torchtitan development by creating an account on GitHub.</li><li><a href="https://github.com/pytorch/torchtune/pull/909">enable QLoRA + FSDP2 by weifengpy · Pull Request #909 · pytorch/torchtune</a>: this PR is built on top of  TorchAO nightly that contains NF4Tensor FSDP2 ops PR1 PR2 Pytorch nightly that contains meta init + cpu offloading PR  unit test: pytest -s tests/torchtune/utils/test_di...</li><li><a href="https://github.com/pytorch/ao/blob/main/torchao/dtypes/nf4tensor.py#L801">ao/torchao/dtypes/nf4tensor.py at main · pytorch/ao</a>: Native PyTorch library for quantization and sparsity - pytorch/ao</li><li><a href="https://drive.google.com/file/d/1yJ176PyAyiMJLI07PL5Mhq-_E7-q2z-B/view">FINAL_torchtune_BIG_wrapping_policy_fused_adamw_llama2_7b_dummy_bs8_cpu_offload_ns48_threads8.json</a>: no description found</li><li><a href="https://vast.ai/">Rent GPUs | Vast.ai</a>: Reduce your cloud compute costs by 3-5X with the best cloud GPU rentals. Vast.ai&#x27;s simple search interface allows fair comparison of GPU rentals from all providers.</li><li><a href="https://mlflow.org/">MLflow | MLflow</a>: Description will go into a meta tag in &lt;head /&gt;</li><li><a href="https://github.com/janeyx99">janeyx99 - Overview</a>: janeyx99 has 32 repositories available. Follow their code on GitHub.</li><li><a href="https://x.com/marksaroufim">Tweet from undefined</a>: no description found</li><li><a href="https://github.com/msaroufim">msaroufim - Overview</a>: CUDA uninЇsțållåțîön fāīłüřęđ. Płēȃšę čøñțàçț șūppørt før åššīštåñćē - msaroufim</li><li><a href="https://github.com/drisspg">drisspg - Overview</a>: @pytorch core. drisspg has 37 repositories available. Follow their code on GitHub.</li><li><a href="https://github.com/awgu">awgu - Overview</a>: awgu has 10 repositories available. Follow their code on GitHub.</li><li><a href="https://github.com/weifengpy">weifengpy - Overview</a>: PyTorch Distributed. weifengpy has 7 repositories available. Follow their code on GitHub.</li><li><a href="https://github.com/rohan-varma">rohan-varma - Overview</a>: PyTorch @facebook | UCLA. rohan-varma has 83 repositories available. Follow their code on GitHub.</li><li><a href="https://github.com/ebsmothers">ebsmothers - Overview</a>: ebsmothers has 8 repositories available. Follow their code on GitHub.</li><li><a href="https://github.com/pytorch/torchtune/blob/1fa1f04baf124c074dcd93831fa38c8b657af1e9/recipes/configs/dev/llama2/7B_qlora_fsdp2.yaml">torchtune/recipes/configs/dev/llama2/7B_qlora_fsdp2.yaml at 1fa1f04baf124c074dcd93831fa38c8b657af1e9 · pytorch/torchtune</a>: A Native-PyTorch Library for LLM Fine-tuning. Contribute to pytorch/torchtune development by creating an account on GitHub.</li><li><a href="https://dev-discuss.pytorch.org/t/fsdp-cudacachingallocator-an-outsider-newb-perspective/1486">FSDP &amp; CUDACachingAllocator: an outsider newb perspective</a>: Hello there. The main motivator of this discussion is:   Questionable profile results for FSDP which led to Ke W. + Alban D. + Andrew G. discussing solutions which led to my benefiting from Alban’s ta...</li><li><a href="https://github.com/pytorch/pytorch/blob/f600faf2480ddd6600ad88fbfc5dd28da132d61d/torch/distributed/_composable/fsdp/_fsdp_param.py#L515">pytorch/torch/distributed/_composable/fsdp/_fsdp_param.py at f600faf2480ddd6600ad88fbfc5dd28da132d61d · pytorch/pytorch</a>: Tensors and Dynamic neural networks in Python with strong GPU acceleration - pytorch/pytorch
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[paige_when_finetune](https://discord.com/channels/1238365980128706560/1242224662142779530/1249860352565448834)** (166 messages🔥🔥): 

- **Popcorn classification gets geeky**: The channel humorously discusses using synthetic popcorn popping time data for fine-tuning LLMs and performing survival analyses on popcorn kernels. One member quips, "whoever does a case study on popcorn kernels following the ftcourse repo will be legend."

- **Inverse Poisson distribution sparks math talk**: Detailed discussions emerge around the topic of inverse Poisson distribution, with one user sharing a [math stack exchange link](https://math.stackexchange.com/questions/1195566/inverse-of-a-poisson-distribution-function) to explain formulas and stochasticity.

- **Gemini API catches attention**: Members chat about the capabilities of Google's Gemini Flash supporting audio input, referencing [documentation](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Audio.ipynb). Another member asks about the process of ingesting audio into the Gemini API.

- **Prompt engineering tips and tricks**: A key discussion revolves around using models to create prompts for themselves, including meta-prompt strategies and examples of self-improving prompt techniques. One user shares, "*You can ask models to write prompts for themselves (or other models)*".

- **Thankful endnote and resource sharing**: The chat wraps up with gratitude expressed for Paige's talk, and several important resources such as email contact and additional reading materials. [Paige's personal website](https://webpaige.dev/) and documentation for [context caching](https://ai.google.dev/gemini-api/docs/caching) are shared as useful follow-up links.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://ai.google.dev/gemini-api/docs/caching">no title found</a>: no description found</li><li><a href="https://ai.google.dev/pricing">no title found</a>: no description found</li><li><a href="https://math.stackexchange.com/questions/1195566/inverse-of-a-poisson-distribution-function">Inverse of a Poisson distribution function</a>: I have two i.i.d random variables $X_{1}$ and $X_{2}$ following a continuous Poisson distribution function&#xA;&#xA;$P(x) = \lambda e^{-\lambda\cdot x}$. &#xA;&#xA;I wish to obtain a distribution func...</li><li><a href="https://ai.google.dev/gemini-api/docs/get-started/android_aicore">no title found</a>: no description found</li><li><a href="https://x.com/googledevs/status/1800565067324195032">Tweet from Google for Developers (@googledevs)</a>: 📣 🧠 Exciting news for researchers pushing the boundaries of efficient deep learning! We&#39;ve scaled RecurrentGemma to 9 billion parameters.  🧵↓</li><li><a href="https://huggingface.co/blog/paligemma">PaliGemma – Google&#39;s Cutting-Edge Open Vision Language Model</a>: no description found</li><li><a href="https://tenor.com/view/spongebob-patrick-star-noted-notes-gif-17474838830648097856">Spongebob Patrick GIF - Spongebob Patrick Star - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/google-gemini/cookbook/blob/main/quickstarts/PDF_Files.ipynb">cookbook/quickstarts/PDF_Files.ipynb at main · google-gemini/cookbook</a>: A collection of guides and examples for the Gemini API. - google-gemini/cookbook</li><li><a href="https://tenor.com/view/so-excited-cant-wait-gif-24703188">So Excited GIF - So Excited Cant - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/outlines-dev/outlines">GitHub - outlines-dev/outlines: Structured Text Generation</a>: Structured Text Generation. Contribute to outlines-dev/outlines development by creating an account on GitHub.</li><li><a href="https://github.com/google-gemini/cookbook/blob/main/quickstarts/Audio.ipynb">cookbook/quickstarts/Audio.ipynb at main · google-gemini/cookbook</a>: A collection of guides and examples for the Gemini API. - google-gemini/cookbook</li><li><a href="https://github.com/google-research/t5x">GitHub - google-research/t5x</a>: Contribute to google-research/t5x development by creating an account on GitHub.</li><li><a href="https://simonwillison.net/2024/Feb/21/gemini-pro-video/#images-vs-video">The killer app of Gemini Pro 1.5 is video</a>: Last week Google introduced Gemini Pro 1.5, an enormous upgrade to their Gemini series of AI models. Gemini Pro 1.5 has a 1,000,000 token context size. This is huge—previously that …</li><li><a href="https://cloud.google.com/vertex-ai/generative-ai/docs/samples/generativeaionvertexai-gemini-token-count-multimodal">no title found</a>: no description found</li><li><a href="https://x.com/dynamicwebpaige">Tweet from undefined</a>: no description found</li><li><a href="https://webpaige.dev/">webpaige.dev</a>: no description found</li><li><a href="https://cloud.google.com/vertex-ai?hl=en">Vertex AI with Gemini 1.5 Pro and Gemini 1.5 Flash</a>: Try Vertex AI, a fully-managed AI development platform for building generative AI apps, with access to 130+ foundation models including Gemini 1.5 models.</li><li><a href="https://www.youtube.com/watch?v=wa0MT8OwHuk">Multimodal prompting with a 44-minute movie | Gemini 1.5 Pro Demo</a>: This is a demo of long context understanding, an experimental feature in our newest model, Gemini 1.5 Pro using a 44-minute silent Buster Keaton movie, Sherl...</li><li><a href="https://www.youtube.com/watch?v=LHKL_210CcU">Reasoning across a 402-page transcript | Gemini 1.5 Pro Demo</a>: This is a demo of long context understanding, an experimental feature in our newest model, Gemini 1.5 Pro using a 402-page PDF transcript and a series of mul...</li><li><a href="https://www.youtube.com/watch?v=SSnsmqIj1MI">Problem solving across 100,633 lines of code | Gemini 1.5 Pro Demo</a>: This is a demo of long context understanding, an experimental feature in our newest model, Gemini 1.5 Pro using 100,633 lines of code and a series of multimo...</li><li><a href="https://aistudio.google.com/">no title found</a>: no description found</li><li><a href="https://discuss.ai.google.dev/">Build with Google AI</a>: Ask questions and get support on Google&#39;s Gemini API and Google AI Studio
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[wing-axolotl](https://discord.com/channels/1238365980128706560/1242564077151326388/1250089153475903580)** (2 messages): 

- **LLAMA3 LORA training faces OOM issue on RTX4090**: A user reported encountering OOM (Out of Memory) errors while attempting to merge LORA weights with the base model on an RTX4090, despite trying suggested solutions like `lora_on_cpu: true` and `gpu_memory_limit` options. They referenced the [Axolotl GitHub README](https://github.com/OpenAccess-AI-Collective/axolotl?tab=readme-ov-file#merge-lora-to-base) for details.
- **Axolotl dataset formats and resources shared**: The same user shared various links to help understand Axolotl's supported dataset formats and HuggingFace chat templates. These include [Axolotl documentation](https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/), [HuggingFace chat templating](https://huggingface.co/docs/transformers/en/chat_templating), and a related GitHub repository on [Chat Templates for HuggingFace](https://github.com/chujiezheng/chat_templates).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/OpenAccess-AI-Collective/axolotl?tab=readme-ov-file#merge-lora-to-base).">GitHub - OpenAccess-AI-Collective/axolotl: Go ahead and axolotl questions</a>: Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.</li><li><a href="https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/">Axolotl - Dataset Formats</a>: no description found</li><li><a href="https://huggingface.co/docs/transformers/en/chat_templating">Templates for Chat Models</a>: no description found</li><li><a href="https://github.com/chujiezheng/chat_templates">GitHub - chujiezheng/chat_templates: Chat Templates for 🤗 HuggingFace Large Language Models</a>: Chat Templates for 🤗 HuggingFace Large Language Models - chujiezheng/chat_templates
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[charles-modal](https://discord.com/channels/1238365980128706560/1242564177952768062/1249818485127643187)** (8 messages🔥): 

- **Codingwitcher enjoys Modal's unique approach**: Codingwitcher shared their excitement about deploying **Mistral for inference** on Modal. They described the experience as "magical," especially appreciating the hot-reload feature on a remote machine.
- **Ed157 seeks help for fine-tuning setup**: Ed157 requested guidance on what to input in the datasets and tokens parts of the config YAML file for instruction fine-tuning. They provided a template to illustrate their needs.
- **DamonCrockett faces technical challenges**: DamonCrockett encountered an error message related to a missing volume ID while running a llm-finetuning example on Modal. Despite successful previous runs, they sought assistance to resolve this issue.
- **Charles redirects support queries**: Charles_irl suggested reaching out to the [Modal team on Slack](https://modal.com/slack) for DamonCrockett's issue and recommended the <#1242542198008975430> channel for Ed157's question about axolotl.
- **Danbecker consolidates presentation discussion**: Danbecker instructed to use the <#1241044231829848125> channel for the upcoming discussion about Charles's presentation to keep everything organized.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[fireworks](https://discord.com/channels/1238365980128706560/1245126291276038278/1249845739723554838)** (7 messages): 

- **Users request credits to be added**: Multiple members are requesting for credits to be added to their accounts. Member account IDs such as `i-00dda2`, `dimitry-611a0a`, `tanmaygupta9-70b723`, `contact-ff3a2c`, `ferdousbd-24e887`, `yorick-van-zweeden-e9b5c2`, and `ashiqur-cd00ce` were shared in hopes of resolving the credit issue.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[emmanuel_finetuning_dead](https://discord.com/channels/1238365980128706560/1245129595749925086/1250127543621779489)** (4 messages): 

```html
- **Request for Fine-Tuning Example**: One user asked for an example that illustrates the **fine-tuning process**, such as a notebook, GitHub repo, or blog post. They inquired whether this process can be done with **existing frameworks like TRL or Axolotl**.

- **Dataset Preparation Standard**: Another member shared a [link](https://platform.openai.com/docs/guides/fine-tuning/preparing-your-dataset) to the OpenAI guidelines for preparing datasets for fine-tuning, establishing it as a standard reference.

- **Two-Step Fine-Tuning Process**: Clarification was made on a two-step process for fine-tuning, which includes pretraining and alignment during finetuning. The discussion emphasized *"adding a 'head' layer on the pre-trained model's transformer stack for NLP tasks"* and using QLora to mitigate OOM errors.

- **Technical Breakdown of Mistral Model**: The member provided an example with detailed code illustrating a **MistralForCausalLM** model. The explanation detailed how the last layer `lm_head` functions and how **QLora** replaces linear layers with low-rank matrices to handle out-of-memory errors.
```
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[west-coast-usa](https://discord.com/channels/1238365980128706560/1245410680065097738/)** (1 messages): 

jonbiz: Schedules allowing, we could hang out? See who else is interested?
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[predibase](https://discord.com/channels/1238365980128706560/1245803791710687272/1249802975136977067)** (2 messages): 

- **User reports receiving $25 in credits**: A user named David reported signing up and receiving $25 in credits, sharing his tenant ID (c4697a91). Michael Ortega responded, stating he would look into the matter for David.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[openpipe](https://discord.com/channels/1238365980128706560/1245927847437008896/)** (1 messages): 

_iw3: Hi I also still saw a credit of $100 instead of $222, who should I follow up to check? thanks
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[openai](https://discord.com/channels/1238365980128706560/1245927985123692575/1249808714853122070)** (16 messages🔥): 

- **Scoring prompts toolkit suggestions**: One user asked for recommendations on tools for scoring a large list of prompts with features like error handling and resuming. Recommendations included Promptlayer and Promptfoo, with one user specifically seeking a CLI solution.
- **Using OpenAI credits**: Users discussed various ways to utilize OpenAI credits. One user mentioned using them for embedding tasks and trying out techniques from recent RAG talks, and another mentioned generating content based on their book.
- **Issues with receiving credits**: A user reported completing a form and emailing support multiple times but still not receiving their OpenAI credits. They provided their Org ID and User ID in an attempt to resolve the issue.
- **Tier 2 and GPT-4 access**: There was a discussion about accessing GPT-4/4o with a Tier 2 usage plan. One user shared that they were only able to access it after adding a payment method and credits.
  

---



### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1249800510568136775)** (402 messages🔥🔥): 

- **Member grapples with slow internet**: A member reported a twelve-hour download for Lora Maker due to hitting data cap limits, resulting in slower speeds. They mentioned that *"YT works"* even with slower speeds compared to "50kb/s download from Plytorch.org."
- **Tension over SD3 release**: Enthusiasm and anticipation build up as members discuss the imminent release of Stable Diffusion 3. One member humorously asked, "You mean I have to go to bed again and sleep before it releases?" while another looked forward to better generating pixel art: *"I hope sd3 is also good at voxel art"*.
- **Exploration of AI models and platforms**: Users compared platforms like Huggingface and Civitai for model availability. A member mentioned finding most Loras and checkpoints on Civitai but noted *"there are tons of legal things available in torrent format"*.
- **SDXL vs. traditional upscaling debate**: Discussion ensued on whether upscaling SD1.5 images to 1024 achieves similar results to SDXL trained for 1024x1024. One user proposed a practical solution: "Give it a try when you have SDXL upscale to 2048."
- **Challenges with Stable Diffusion setup**: A member faced difficulties running Stable Diffusion with an AMD GPU, expressing frustration: *"What's the problem? ... its not using my gpu"*. They were advised to revisit installation guides and seek out technical support channels for troubleshooting.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://glif.app/@Oliveira/glifs/clw44qfbl0000m0zztwqk2tnf">glif - StableDiffusion 3 + GPT4 Helper + SDXL 1.5x Upscale (CopyGenius) by Yuri Oliveira COPYGENIUS </a>: no description found</li><li><a href="https://sk2visual.gumroad.com/l/spsjsz">VISION Preset Pack #1 - @visualsk2</a>: PRESET PACK Collection by VisualSK2 ( PC-MOBILE)A collection of my best presets for Lightroom that I use on a daily basis to give my shoots a cinematic and consistent look.What&#39;s inside?20 Presets...</li><li><a href="https://www.youtube.com/watch?v=KyLqUf4cdwc">Microsoft Vista Speech Recognition Tested - Perl Scripting</a>: Credits to scrubadub (check for user: scrubadub1 for more videos like this !) for sharing this first, until he got banned... Here we go again... Please don&#39;t...</li><li><a href="https://www.instagram.com/p/C6p8KgSSzo3/">madhav kohli on Instagram: &quot;Fear and loathing in NCR&#x2026;&quot;</a>: 14K likes, 73 comments - mvdhav on May 6, 2024: &quot;Fear and loathing in NCR&#x2026;&quot;. </li><li><a href="https://youtu.be/ScPp2nhowgA">The Donald Trump Calculus Song (He Sings Terribly)</a>: The Donald Trump Calculus Song (He sings terribly)This was a school project. Please like, comment, and subscribe. My grade depends on it. 🙏Disclaimer: This ...</li><li><a href="https://www.instagram.com/p/C6_kd_hoNGb/">Samuele &#x201c;SK2&#x201d; Poggi on Instagram: &quot;[Vision III/Part. 4] &#x2728;&#x1f90d; SK2&#x2022; Fast day &#x2022;

#photography #longexposure #explore #trending #explorepage&quot;</a>: 33K likes, 265 comments - visualsk2 on May 15, 2024: &quot;[Vision III/Part. 4] &#x2728;&#x1f90d; SK2&#x2022; Fast day &#x2022;  #photography #longexposure #explore #trending #explorepage&quot;. </li><li><a href="https://www.seaart.ai/models/detail/0e5b32eb19562e304d29771ad3898af5">Hard Muscle - SeaArt AI Model</a>: no description found</li><li><a href="https://www.instagram.com/p/C781eUDoJ2h/">Samuele &#x201c;SK2&#x201d; Poggi on Instagram: &quot;[Vision IV/Part.6] Thanks so much for 170.000 Followers &#x2728;&#x1f64f;&#x1f3fb;
Only a few days left until the tutorial is released.

#grainisgood #idea #reels #framebyframe #photography #blurry #explorepage&quot;</a>: 16K likes, 130 comments - visualsk2 on June 8, 2024: &quot;[Vision IV/Part.6] Thanks so much for 170.000 Followers &#x2728;&#x1f64f;&#x1f3fb; Only a few days left until the tutorial is released.  #gra...</li><li><a href="https://github.com/lks-ai/ComfyUI-StableAudioSampler">GitHub - lks-ai/ComfyUI-StableAudioSampler: The New Stable Diffusion Audio Sampler 1.0 In a ComfyUI Node. Make some beats!</a>: The New Stable Diffusion Audio Sampler 1.0 In a ComfyUI Node. Make some beats! - lks-ai/ComfyUI-StableAudioSampler</li><li><a href="https://aitracker.art/">Home :: AiTracker</a>: no description found</li><li><a href="https://tensor.art/models/654286272942196700">Hard Muscle - v1.0 | Stable Diffusion Checkpoint</a>: no description found
</li>
</ul>

</div>
  

---



### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1249804822316843111)** (164 messages🔥🔥): 

- **Expect Multigpu Support in Early July 2024:** Members eagerly anticipate the release of multigpu, with a tentative date set for early July 2024. One member humorously mentioned, "2025 but no seriously, early July, 2024."
- **LORA Inference Interface and vLLM:** There's an interest in an inference interface that allows enabling/disabling LORA during inference. A user discovered that vLLM supports this feature and contemplates if it could work with exl2 or TabbyAPI.
- **Overfitting Issues in Training:** A member is experiencing overfitting in model training, resulting in lower performance on simpler tasks. Suggestions included trying data augmentation, leveraging weight decay, and ensuring diverse and comprehensive training data.
- **Fine-Tuning and EOS Token Discussions:** Members discussed the importance of EOS tokens while training instruct models on general texts. One suggested using `BOS_token + entire text + EOS_token` for continuous pre-training.
- **Hugging Face AutoTrain Adds Unsloth Support:** [Hugging Face AutoTrain now supports Unsloth](https://x.com/abhi1thakur/status/1800511251145015393), enabling users to fine-tune LLMs more efficiently. The new feature was met with excitement and appreciation.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2406.05587">Creativity Has Left the Chat: The Price of Debiasing Language Models</a>: Large Language Models (LLMs) have revolutionized natural language processing but can exhibit biases and may generate toxic content. While alignment techniques like Reinforcement Learning from Human Fe...</li><li><a href="https://huggingface.co/bartowski/Qwen2-72B-Instruct-GGUF">bartowski/Qwen2-72B-Instruct-GGUF · Hugging Face</a>: no description found</li><li><a href="https://x.com/abhi1thakur/status/1800511251145015393">Tweet from abhishek (@abhi1thakur)</a>: AutoTrain + Unsloth = 🚀🚀🚀 AutoTrain has now added support for unsloth which means you can use unsloth&#39;s optimizations to finetune LLMs super-fast and with much less memory 💥 And all you need t...</li><li><a href="https://github.com/unslothai/unsloth/pull/609">clears any selected_adapters before calling internal_model.save_pretr… by neph1 · Pull Request #609 · unslothai/unsloth</a>: …ained I have a script that downloads the adapter from hf, merges it with the base model and uploads the result. It worked a month ago (or so), but failed now. Tried it both on colab, kaggle and lo...</li><li><a href="https://github.com/unslothai/unsloth/issues/611">save_pretrained_merged doesn&#39;t merge the model · Issue #611 · unslothai/unsloth</a>: Problem My goal, I want to save the merged model as a GGUF file, but I&#39;m getting various errors. The deeper problem seems to be that merging lora+base model isn&#39;t saving a merged file. I think...</li><li><a href="https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://xtherapy.streamlit.app/">no title found</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1249821272008032337)** (72 messages🔥🔥): 

- **Revamping Design Colors**: Several users discussed improvements to the chatbot interface, suggesting changes like using white backgrounds instead of red, turning squares into diamonds, and desaturating colors. One user remarked, *"There we go a less stupid looking version."*
- **Llama for Scalable Image Generation**: The conversation touched upon [LlamaGen](https://github.com/FoundationVision/LlamaGen), an autoregressive model for scalable image generation. The project was discussed with enthusiasm, and a GitHub link was shared.
- **Apple's New AI Integration at WWDC**: Apple’s announcement of personalized AI ("Apple Intelligence") sparked discussions regarding its efficiency and potential for integrating large models. Users debated its implementation and potential language support, with comments like *"Apple's perfectly integrating things to their app."* 
- **Training on the Fly**: The feasibility and benefits of on-the-fly model training were debated. Concerns were raised about training costs and quality, while some saw potential in daily finetuning, likening real-time training to financial applications.
- **Online Machine Learning Limitations**: Discussions highlighted potential issues with online machine learning, such as catastrophic forgetting and quality of human data inputs. One user mentioned, *"There are probably several reasons it hasn't worked out. I can guess catastrophic forgetting is one of them."*
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/FoundationVision/LlamaGen">GitHub - FoundationVision/LlamaGen: Autoregressive Model Beats Diffusion: 🦙 Llama for Scalable Image Generation</a>: Autoregressive Model Beats Diffusion: 🦙 Llama for Scalable Image Generation - FoundationVision/LlamaGen</li><li><a href="https://github.com/ml-explore">ml-explore</a>: Machine learning research on your laptop or in a data center - by Apple - ml-explore</li><li><a href="https://www.macrumors.com/2024/06/10/apple-intelligence-generative-personal-ai-unveiled-for-iphone-ipad-and-mac/">'Apple Intelligence' Personal AI Unveiled for iPhone, iPad, and Mac</a>: Apple at WWDC today announced Apple Intelligence, a deeply integrated, personalized AI experience for Apple devices that uses cutting-edge generative...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1249817994024980500)** (67 messages🔥🔥): 

- **Tokenizers are not specific to Unsloth models**: "Any service tokenizer will do (llama.cpp tokenizer, huggingface tokenizer). Matter of fact, probably other than llama.cpp is using huggingface tokenizer (including unsloth)".
- **Unsloth model size confusion resolved**: A user questioned why saving the model only resulted in a 100MB file. It was clarified: "Save_pretrained_merged should save the whole thing".
- **Sample CSV format for fine-tuning confirmed**: A user asked if their CSV format was correct for fine-tuning; the format "question,answer" was discussed. They were directed to a [YouTube video](https://www.youtube.com/watch?v=3eq84KrdTWY&t=665s) for detailed guidance.
- **Multi-GPU support coming in July**: "Currently unsloth works on single GPU. We will be rolling out multiGPU support in early July", and clarified that Unsloth Pro is mainly for enterprises.
- **Can't finetune GGUF models directly**: An attempt to fine-tune GGUF models led to an explanation that it is not supported and suggested using new experimental interoperability features in [Hugging Face transformers](https://huggingface.co/docs/transformers/en/gguf#support-within-transformers).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=3e"> - YouTube</a>: no description found</li><li><a href="https://unsloth.ai/contact">Contact</a>: no description found</li><li><a href="https://huggingface.co/Bibekananda/bk_gguf_Chat_model">Bibekananda/bk_gguf_Chat_model · Hugging Face</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=3eq84KrdTWY&t=665s">Llama 3 Fine Tuning for Dummies (with 16k, 32k,... Context)</a>: Learn how to easily fine-tune Meta&#39;s powerful new Llama 3 language model using Unsloth in this step-by-step tutorial. We cover:* Overview of Llama 3&#39;s 8B and...</li><li><a href="https://huggingface.co/docs/transformers/en/gguf#support-within-transformers">GGUF and interaction with Transformers</a>: no description found</li><li><a href="https://huggingface.co/unsloth">unsloth (Unsloth AI)</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/wiki#saving-models-to-16bit-for-vllm">Home</a>: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/pull/609">clears any selected_adapters before calling internal_model.save_pretr… by neph1 · Pull Request #609 · unslothai/unsloth</a>: …ained I have a script that downloads the adapter from hf, merges it with the base model and uploads the result. It worked a month ago (or so), but failed now. Tried it both on colab, kaggle and lo...</li><li><a href="https://colab.research.google.com/drive/1Dyauq4kTZoLewQ1cApceUQVNcnnNTzg_?usp=sharing)">Google Colab</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1250136212250034277)** (3 messages): 

- **Try xTherapy AI Finetuned on Llama 3 8B**: [Check out this therapy AI](https://xtherapy.streamlit.app/), finetuned on **llama 3 8b** using **unsloth**. Feedback on improvements is welcomed.
  
- **CAMB AI Releases MARS5 TTS**: [CAMB AI](https://github.com/camb-ai/mars5-tts) has open-sourced their 5th iteration of MARS TTS model on GitHub. They were also [featured on VentureBeat](https://venturebeat.com/ai/exclusive-camb-takes-on-elevenlabs-with-open-voice-cloning-ai-model-mars5-offering-higher-realism-support-for-140-languages/) and are inviting feedback from the community.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/camb-ai/mars5-tts">GitHub - Camb-ai/MARS5-TTS: MARS5 speech model (TTS) from CAMB.AI</a>: MARS5 speech model (TTS) from CAMB.AI. Contribute to Camb-ai/MARS5-TTS development by creating an account on GitHub.</li><li><a href="https://xtherapy.streamlit.app/">no title found</a>: no description found
</li>
</ul>

</div>
  

---



### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1249807677840953415)** (62 messages🔥🔥): 

- **4-bit Quantization for Llama 3 Faces Challenges**: A member sought advice on **4-bit quantizing Llama 3** 8b with negligible performance degradation for training SAEs. Other members suggested trying **sharding** or using **Tensor Parallelism**, but noted the potential challenges and experimental nature of these methods [Tensor Parallelism in PyTorch](https://pytorch.org/docs/stable/distributed.tensor.parallel.html).

- **Debate Over Quantization Methods**: Members debated the most effective quantization methods, with **IQ4_xs** and **HQQ, AWQ, EXLv2** being discussed. A link to a [blog post](https://stephenpanaro.com/blog/llm-quantization-for-iphone) was shared, illustrating various quantization techniques for Apple Silicon with claims of better performance than traditional methods.

- **Federated Learning Considerations**: A user raised the idea of Apple training on personal data without moving it off-device, sparking a discussion about **federated learning**. Concerns about privacy and potential misuse of gradient data were discussed, illustrating the complex nature of federated implementations.

- **Geographical Time Series Data Prediction**: A member asked about experiences with **geographical time series data** and predicting events at specific locations. Another member shared their experience using **Google Earth Engine + LSTM** for similar predictions.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://stephenpanaro.com/blog/llm-quantization-for-iphone">LLMs for your iPhone: Whole-Tensor 4 Bit Quantization</a>: Introducing a new 4 bit quantization scheme that is fully compatible with Apple Silicon…</li><li><a href="https://pytorch.org/docs/stable/distributed.tensor.parallel.html">Tensor Parallelism - torch.distributed.tensor.parallel &mdash; PyTorch 2.3 documentation</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1cst400/result_llama_3_mmlu_score_vs_quantization_for/">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1249835887819231313)** (118 messages🔥🔥): 

- **Quest for Online RL in Regular-Scale LLMs**: A member is seeking research papers on *online RL* for regular-scale LLMs but finds theoretical assumptions often unsubstantiated in practice.
  
- **VALL-E 2 Advances Zero-Shot TTS**: [VALL-E 2](https://arxiv.org/abs/2406.05370) achieves human parity in zero-shot TTS, with improvements in Repetition Aware Sampling and Grouped Code Modeling. Yet, the [project page](https://web.archive.org/web/20240529183033/https://www.microsoft.com/en-us/research/project/vall-e-x/vall-e-2/) has fluctuating availability due to premature leakage.
  
- **LlamaGen Explores Visual Tokenization**: [LlamaGen's](https://arxiv.org/abs/2406.06525) new models apply autoregressive next-token prediction to visual domains, significantly outperforming popular diffusion models. A discussion emerges about its novel implementation of CFG in autoregressive decoding, reminiscent of methods in [previous works](https://arxiv.org/abs/2306.17806).

- **Challenges with Transformer Learning Capabilities**: Papers like [this one](https://arxiv.org/abs/2406.06467) outline tasks that standard Transformers struggle to learn without implementing supervised *scratchpads*. The discussion touches on the impracticality of *unsupervised scratchpads* due to inefficiencies in gradient descent for complex token interactions.

- **Efficacy of Influence Functions**: The utility and limitations of influence functions are probed, linking to foundational explanations like [Koh and Liang's paper](https://arxiv.org/pdf/1703.04730), with emphasis on approximations and practical applicability.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2406.05370">VALL-E 2: Neural Codec Language Models are Human Parity Zero-Shot Text to Speech Synthesizers</a>: This paper introduces VALL-E 2, the latest advancement in neural codec language models that marks a milestone in zero-shot text-to-speech synthesis (TTS), achieving human parity for the first time. Ba...</li><li><a href="https://arxiv.org/abs/2406.06525">Autoregressive Model Beats Diffusion: Llama for Scalable Image Generation</a>: We introduce LlamaGen, a new family of image generation models that apply original ``next-token prediction&#39;&#39; paradigm of large language models to visual generation domain. It is an affirmative...</li><li><a href="https://arxiv.org/abs/2406.06248">Compute Better Spent: Replacing Dense Layers with Structured Matrices</a>: Dense linear layers are the dominant computational bottleneck in foundation models. Identifying more efficient alternatives to dense matrices has enormous potential for building more compute-efficient...</li><li><a href="http://arxiv.org/abs/2406.06248">Compute Better Spent: Replacing Dense Layers with Structured Matrices</a>: Dense linear layers are the dominant computational bottleneck in foundation models. Identifying more efficient alternatives to dense matrices has enormous potential for building more compute-efficient...</li><li><a href="http://arxiv.org/abs/2406.06467">How Far Can Transformers Reason? The Locality Barrier and Inductive Scratchpad</a>: Can Transformers predict new syllogisms by composing established ones? More generally, what type of targets can be learned by such models from scratch? Recent works show that Transformers can be Turin...</li><li><a href="https://arxiv.org/abs/2210.02671">A Logic for Expressing Log-Precision Transformers</a>: One way to interpret the reasoning power of transformer-based language models is to describe the types of logical rules they can resolve over some input text. Recently, Chiang et al. (2023) showed tha...</li><li><a href="https://arxiv.org/abs/2406.06484">Parallelizing Linear Transformers with the Delta Rule over Sequence Length</a>: Transformers with linear attention (i.e., linear transformers) and state-space models have recently been suggested as a viable linear-time alternative to transformers with softmax attention. However, ...</li><li><a href="https://ai.stackexchange.com/q/45949/68078">Is a small transformer model able to effectively handle any input length provided it is fine-tuned on it?</a>: Suppose we have a transformer LLM which can do a task such as summarising.&#xA;I know transformer can technically handle any input length (assume we are not using learned positional embeddings) becaus...</li><li><a href="https://arxiv.org/abs/2306.17806">Stay on topic with Classifier-Free Guidance</a>: Classifier-Free Guidance (CFG) has recently emerged in text-to-image generation as a lightweight technique to encourage prompt-adherence in generations. In this work, we demonstrate that CFG can be us...</li><li><a href="https://web.archive.org/web/20240529183033/https://www.microsoft.com/en-us/research/project/vall-e-x/vall-e-2/">VALL-E</a>: VALL-E is a neural codec language model using discrete codes derived from an off-the-shelf neural audio codec model, and regard TTS as a conditional language modeling task rather. VALL-E emerges in-co...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1249818695421788201)** (6 messages): 

- **Challenges with DeepSeek model integration**: A member inquired about interpreting the DeepSeek-LLM-7B model and its addition to Transformerlens. Another member confirmed the difficulty, mentioning they had encountered short-circuit issues during their attempts.
- **DeepSeek model is LLaMA-based**: A helpful comment noted the DeepSeek-LLM-7B model's architecture is based on LLaMA, suggesting it's straightforward to integrate into Transformerlens with some hacks. They also advised double-checking output probabilities to avoid surprises.
- **Repo for Multilingual Transformers**: A member shared a [GitHub link](https://github.com/Butanium/llm-latent-language/blob/1054015066a4fa20386765d72601d03aa7ef5887/utils.py#L334) for a repo accompanying a paper on "Do Llamas Work in English? On the Latent Language of Multilingual Transformers". This resource could potentially offer insights or methods applicable to the DeepSeek model integration.

**Link mentioned**: <a href="https://github.com/Butanium/llm-latent-language/blob/1054015066a4fa20386765d72601d03aa7ef5887/utils.py#L334">llm-latent-language/utils.py at 1054015066a4fa20386765d72601d03aa7ef5887 · Butanium/llm-latent-language</a>: Repo accompanying our paper &quot;Do Llamas Work in English? On the Latent Language of Multilingual Transformers&quot;. - Butanium/llm-latent-language

  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1249817176869965937)** (6 messages): 

- **Enable chat templating with the --apply_chat_template flag**: Eleuther now supports chat templating with HF models using the **--apply_chat_template** flag. However, *this feature is not turned on by default*.
- **Specify stop sequences to resolve task issues**: Some users found that manually specifying stop sequences helps resolve task-specific issues. However, shuffling choices in `doc_to_choices` did not affect model answers as expected.
- **Batch API needs improvement and contributions are welcome**: The current implementation of batching in API or local server models is not optimal, especially for **OpenAI Batch API** integration. Contributions to improve this, particularly with better batching methods, are appreciated.
- **Steps for implementing batch API discussed**: A high-level implementation of the batch API includes creating a JSONL file, uploading it via API, running the chosen model, and returning run and file IDs for status checks. **Async evaluation calls** are suggested to smooth the process.
- **Utility for rerunning metrics on batch API results files planned**: The proposal includes adding a utility to convert responses from **OpenAI** to per-sample outputs. This will facilitate rerunning metrics on saved results files as part of the harness.
  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1250174056477491361)** (1 messages): 

- **Seeking LTIP Dataset Alternatives**: A user inquired about **open-source alternatives** to the **LTIP dataset** used by Deepmind for pre-training **Flamingo and GATO**. They noted that the **Datasheet for the LTIP dataset** can be found [in the Appendix of the Flamingo paper](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/tackling-multiple-tasks-with-a-single-visual-language-model/flamingo.pdf) and mentioned the now redacted **LAION datasets** as a previous option.
  

---



### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1249906243984953416)** (128 messages🔥🔥): 

- **Evaluating 7950x3D and GeForce RTX 4090 for Builds**: A member considered using a Ryzen 7950x with 2x GeForce RTX 4090s for a build but expressed concerns about the 4090's size, power draw, and lack of NVLink communication. Another member recommended the 7950x3D due to its larger L3 cache and minor price increase.
- **GPUs for Llama-3 Inference**: Members discussed optimal GPUs for Llama-3 inference, with considerations between single RTX 4090 and dual 3090 setups. The latter was preferred for applications requiring more VRAM, although the lack of NVLink on certain models like the 4060Ti and potential PCI bottlenecks were highlighted.
- **Challenges in Triton Kernel Development**: A new user shared issues with their Triton-based Conv2d/Linear layer implementations, finding them slower than cuDNN counterparts and struggling with debugging inside kernels. They sought advice on good resources and methods to print variables within Triton kernels.
- **Discussion on CPU/GPU Configurations for HPC Setups**: The conversation delved into various configurations including older Threadripper models, modern Epyc CPUs, and their PCIe lane limitations. Power draw considerations and cooling requirements for heavy GPU setups were also discussed.
- **Innovative Use of Waste Heat from GPUs**: In jest, a member suggested using GPUs to replace a hot tub heater, which sparked a discussion about sustainable data centers utilizing waste heat for community benefits.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.msi.com/Graphics-Card/GeForce-RTX-4090-SUPRIM-LIQUID-24G">MSI GeForce RTX™ 4090 SUPRIM LIQUID 24G</a>: GeForce RTX™ 4090 SUPRIM LIQUID 24G features liquid cooling for the GPU and air cooling for VRMs and a sturdy brushed metal backplate providing passive cooling. The MSI SUPRIM LIQUID is easy to instal...</li><li><a href="https://huggingface.co/Mozilla/Meta-Llama-3-70B-Instruct-llamafile/tree/main">Mozilla/Meta-Llama-3-70B-Instruct-llamafile at main</a>: no description found
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1249838457581469830)** (2 messages): 

- **For Loop is Necessary**: A user emphasized the need for a for loop in their code, simply stating, "no, you need to do the for loop."
- **Preference for Non-Tuple Syntax in Triton**: Another user noted their preference for the non-tuple version of the `load_2d` function in Triton. They explained, "Tuples would only add parentheses imo. So I'd leave it at that."
  

---


### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1249868128058605679)** (10 messages🔥): 

- **Custom C++/CUDA with torch.compile**: A member asked if custom C++/CUDA operators compatible with torch.compile allow full graph compilation and are AOT compilable with torch.export. Another member responded that it should allow full graph compilation but was not entirely sure about export. They provided an example for reference: [Custom CUDA extensions by msaroufim · Pull Request #135 · pytorch/ao](https://github.com/pytorch/ao/pull/135).
- **HIP kernels in PyTorch**: A user inquired if it's possible to write HIP kernels and use them in PyTorch. Another member suggested that using `load_inline` should work fine.
- **Inference optimization for AWQ**: A member questioned the lack of inference optimized Triton/CUDA kernels for AWQ, speculating if the heterogeneity in how it quantizes weights poses a challenge. Another member responded by referencing [PyTorch int4 matmul documentation](https://pytorch.org/cppdocs/api/function_namespaceat_1adeda9630914278ac02d7fd758da19e3d.html).
- **CUDA libraries warmup**: A user mentioned that they've experienced issues with CUDA libraries that require a warmup period for certain algorithms that torch uses internally.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://pytorch.org/cppdocs/api/function_namespaceat_1adeda9630914278ac02d7fd758da19e3d.html">Function at::_weight_int4pack_mm &mdash; PyTorch main documentation</a>: no description found</li><li><a href="https://docs.google.com/document/d/1-LdJZBzlxiF0Tm-8NfbyFvRJaofdwRgLcycXGmlIpS0/edit">[Tutorial] Custom C++ and CUDA Operators</a>: Custom C++ and CUDA Operators PyTorch offers a large library of operators that work on Tensors (e.g. torch.add, torch.sum, etc). However, you may wish to bring a new custom operator to PyTorch. This t...</li><li><a href="https://github.com/pytorch/ao/pull/135">Custom CUDA extensions by msaroufim · Pull Request #135 · pytorch/ao</a>: This is the mergaeble version of #130 - some updates I have to make   Add a skip test unless pytorch 2.4+ is used and Add a skip test if cuda is not available  Add ninja to dev dependencies  Locall...
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1250151507890798613)** (1 messages): 

- **Satabios showcases new model compression package**: A user introduced their self-built model compression and inferencing package, [Sconce](https://github.com/satabios/sconce). They invited other members to give the project a star if they liked it and welcomed suggestions for improvements.
  

---


### **CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1249832156151152784)** (1 messages): 

- **Iron Bound interview shared**: A member posted a link to [an interview on The Amp Hour podcast](https://theamphour.com/the-amp-hour-84-bunnies-bibelot-bonification/). The podcast episode is titled "Bunnies Bibelot Bonification."
  

---


### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1249821968941711491)** (2 messages): 

- **Charles's PR Improves Benchmarking**: Charles has submitted a [pull request to PyTorch's AO repository](https://github.com/pytorch/ao/pull/276) that adds support for benchmarking Llama models. This aims to provide "stable eval/benchmarking" functionality within the TorchAO codebase.
- **Large N May Not Require Changes**: A member noted that if N (sample size) is sufficiently large, additional modifications may not be necessary. The implication is that there may be minimal impact on the outcomes with larger sample sizes.

**Link mentioned**: <a href="https://github.com/pytorch/ao/pull/276">Adding Llama to TorchAO by HDCharles · Pull Request #276 · pytorch/ao</a>: Summary: This PR adds funcitonality for stable eval/benchmarking of llama models within the torchao codebase. the model stuff is in torchao/_models/llama with eval being moved to _models/_eval.py m...

  

---


### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1250022396031795200)** (42 messages🔥): 

```html
- **ThunderKitten Performance Disappoints**: Members discussed **ThunderKitten's** performance, noting it achieved ~75 TFLOPS versus ~400 TFLOPS with **cuBLAS** for basic matmul on **A100**. One explanation was that ThunderKitten might be overly focused on **TMA**, making the non-TMA path massively L1/load-store limited.
- **C++20 in ThunderKitten**: Conversations highlighted that **ThunderKitten** requires C++20, which some members found cumbersome despite the language’s advantages in handling concepts. There was debate on whether similar functionality could be achieved with C++17, albeit with more complex and less readable template code.
- **FP8 Training Stability Concerns**: One member mentioned that despite FP8 training being seen as offering performance improvements, many groups still prefer **FP16** due to stability concerns. They noted that **FP8** is not fully understood or stable, making **FP16** a more predictable choice for training currently.
- **Using Thrust for Elementwise Transformations**: A member inquired about optimizing performance with **Thrust** for elementwise transformations on **Hopper/Blackwell** GPUs. They sought advice on leveraging aligned data for more efficient computations and compared the performance of different methodologies, including **manual TMA**.
```
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://devblogs.microsoft.com/cppblog/cpp23-deducing-this/#crtp)">C++23&#039;s Deducing this: what it is, why it is, how to use it - C++ Team Blog</a>: Find out how C++23&#039;s Deducing this feature can help make your code better.</li><li><a href="https://www.modernescpp.com/index.php/c23-deducing-this/).">C++23: Deducing This &#8211; MC++ BLOG</a>: no description found
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[bitnet](https://discord.com/channels/1189498204333543425/1240586843292958790/1249848448711262268)** (1 messages): 

- **1-bit LLMs Training Guide Shared**: An important resource for training **1-bit LLMs** was shared, which includes tips, code, and FAQs. Check out the comprehensive guide at [Microsoft's Unilm GitHub](https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf).

**Link mentioned**: <a href="https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf">unilm/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf at master · microsoft/unilm</a>: Large-scale Self-supervised Pre-training Across Tasks, Languages, and Modalities - microsoft/unilm

  

---


### **CUDA MODE ▷ #[sparsity](https://discord.com/channels/1189498204333543425/1247663759434977453/)** (1 messages): 

satabios: Model Compression/Inferencing Package: https://github.com/satabios/sconce
  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1249831070916939997)** (134 messages🔥🔥): 

- **Debating the best concurrency models for Mojo**: Extensive discussion on whether structured concurrency, using primitives like async/await, should be part of Mojo's standard library or in the ecosystem. **"Rust was the first language with full async to io_uring"**, sparking a debate on achieving optimal performance without dangerous pitfalls.

- **Mojo's asynchronous capabilities and GPU support**: Conversations revolved around Mojo's current limitations to CPU, anticipated GPU support, and potential for TPU implementation. One user expressed concerns, *"If Mojo is 3 times faster than Python, but can't run on TPUs that are 6-8 times cheaper..."*, highlighting the importance of heterogeneous hardware compatibility.

- **Modular's approach and funding**: Members shared insights about Modular's funding and development strategy, noting that despite $130M funding, **"Mojo is really competing with C, C++, Rust and CUDA, not Python"**. The promise of supporting various hardware architectures in the future was also emphasized.

- **AI concurrency examples and structured paradigms**: The validity of different AI concurrency paradigms, such as structured concurrency, was discussed in the context of Mojo’s long-term goals. Points were raised about the importance of compiler support for breaking down tasks, ensuring **"programming model would be the same"** across devices.

- **Mojo's potential and community sentiments**: Users discussed their hopes and doubts about Mojo’s future capabilities and the timeline for supporting additional hardware like TPUs. **"Keep in mind that Mojo and its infra is fairly new"**, one user reminded, urging patience as the ecosystem evolves.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.modular.com/.">Modular: Accelerating the Pace of AI</a>: The Modular Accelerated Xecution (MAX) platform is the worlds only platform to unlock performance, programmability, and portability for your AI workloads.</li><li><a href="https://mlir.llvm.org/docs/Dialects/AsyncDialect/">'async' Dialect - MLIR</a>: no description found
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/)** (1 messages): 

ModularBot: From *Modular*:
<https://twitter.com/Modular/status/1800580309156847626>
  

---


### **Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1249803277030395937)** (21 messages🔥): 

- **Xoshiro PRNG Accelerates Simulations**: A member ported the xoshiro PRNG to Mojo, achieving *64 Gbps* on a laptop and *180 Gbps* using SIMD and 4 parallel streams. They inquired if there are plans to expand the standard library with additional generators. [Numerics for Mojo Repo](https://github.com/thk686/numojo) discussed.
- **NuMojo and Math Libraries**: Another member noted an ongoing project porting the numpy library to Mojo, and linked the [NuMojo repository](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo). There's also individual work on high energy Python math porting.
- **Loop Vectorization in Mojo**: Discussion on effectively vectorizing loops in Mojo, highlighting the `math.iota` function for creating increasing sequences via SIMD. Performance tests showed the vectorized loop performed faster (0.032032 sec) compared to the ordinary loop (0.059314 sec).
- **Warm-up Loop for Performance Testing**: A suggestion was made to include a warm-up loop before starting performance tests to ensure accuracy. It was noted that this might be more relevant in interpreted languages like Python, less so in Mojo which is compiled.
- **Compile-time Interpreter Speculations**: A member showed curiosity about Mojo's potential compile-time interpreter preventing undefined behavior similar to C++'s `constexpr`. They highlighted its importance for testing the safety of code using unsafe features.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.modular.com/max/api/mojo/graph/quantization/quantization_encoding/QuantizationEncoding">QuantizationEncoding | Modular Docs</a>: Describes the encoding for a data type that can be quantized.</li><li><a href="https://docs.modular.com/mojo/roadmap#no-python-style-generator-functions">Mojo🔥 roadmap &amp; sharp edges | Modular Docs</a>: A summary of our Mojo plans, including upcoming features and things we need to fix.</li><li><a href="https://docs.modular.com/mojo/stdlib/math/math/iota">iota | Modular Docs</a>: iotatype Int -&gt; SIMD[$0, $1]</li><li><a href="https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo">GitHub - Mojo-Numerics-and-Algorithms-group/NuMojo: A numerics library for the Mojo programming language</a>: A numerics library for the Mojo programming language - Mojo-Numerics-and-Algorithms-group/NuMojo</li><li><a href="https://github.com/thk686/numojo">GitHub - thk686/numojo: Numerics for Mojo</a>: Numerics for Mojo. Contribute to thk686/numojo development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[🏎engine](https://discord.com/channels/1087530497313357884/1212827673257316453/1249903046075482145)** (9 messages🔥): 

- **Broken Link in Blog Post**: A member reported that the [link](https://docs.modular.com/max/reference/mojo/graph/quantization/) on the [blog post](https://www.modular.com/blog/max-24-4-introducing-quantization-apis-and-max-on-macos) leads to a 404 error. Suggestions were made to adjust it to [this link](https://docs.modular.com/max/api/mojo/graph/quantization/).

- **Inquiries About TPUs Compatibility**: A user inquired about the potential for running Mojo (MAX engine) paired with Google TPU accelerators. They highlighted the cost-effectiveness of TPUs compared to A100x8 accelerators.

- **MAX Engine Roadmap and TPU Development**: It was shared that Nvidia's GPU support is on the roadmap for release in Summer. A user expressed interest in the feasibility of developing TPU implementation, citing potential time and cost benefits.

- **Exploring TPU Resources**: A member referenced the [OpenXLA GitHub repository](https://github.com/openxla/xla) as a potential resource for developing a TPU implementation. They considered leveraging existing machine learning compiler capabilities for this purpose.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.modular.com/max/api/mojo/graph/quantization/quantization_encoding/QuantizationEncoding">QuantizationEncoding | Modular Docs</a>: Describes the encoding for a data type that can be quantized.</li><li><a href="https://github.com/openxla/xla">GitHub - openxla/xla: A machine learning compiler for GPUs, CPUs, and ML accelerators</a>: A machine learning compiler for GPUs, CPUs, and ML accelerators - openxla/xla</li><li><a href="https://docs.modular.com/max/api/mojo/graph/quantization/">quantization | Modular Docs</a>: APIs to quantize graph tensors.
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1249821272917934160)** (13 messages🔥): 

- **RAII's role in Mojo memory management debated**: Discussion centered around the necessity of a context manager in Mojo, with one member stating that "RAII clean up" can handle file cleanup. Another pointed out potential issues given Mojo's memory model, with commentary on UnsafePointers and lifetimes.
- **UnsafePointers and Ownership**: Members debated the role of UnsafePointers in Mojo, emphasizing that these pointers don't use RAII due to their non-owning nature and lack of lifetimes. It was suggested that Mojo could adopt an owning pointer type like Rust's `Box`.
- **Mojo nightly compiler update released**: A new nightly Mojo compiler has been released, version `2024.6.1105`. Key updates include the removal of `SliceNew` and `SIMD.splat`, and the implementation of `NamedTemporaryFile` in the `tempfile` module. [Raw diff](https://github.com/modularml/mojo/compare/f8c229b856795f2782e77db6d125fda1f8d753d4...76eda306af929d9576d7190a7f8f3aa1df83baf6) and [current changelog](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md) were provided.
  

---



### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1249805037304025230)** (157 messages🔥🔥): 

- **Apple's New "Personal Intelligence" System Enters the Scene**: Discussions highlighted Apple's introduction of "Apple Intelligence," featuring free integration with ChatGPT-4o into Siri and system-wide writing tools. Detailed in a [Reddit thread](https://www.reddit.com/r/ChatGPT/s/KrhcqUpEuq), this system focuses on user privacy and runs without compromising personal context.

- **Rabbit R1 Device: Scam or Not?**: The Rabbit R1 was debated, with some stating it might be a scam linked to a crypto history, further discussed in Coffeezilla's videos. Users debated its practicality, with suggestions to install Android for better usability.

- **Perplexity Pro Access and Features**: There was confusion about accessing Perplexity Pages and the rollout for pro users. Currently, page generation is exclusive to Pro users and works only on desktop/web settings.

- **Issues with Use and Feedback on Academic Accuracy**: Users reported issues with Perplexity's accuracy when sourcing academic content, preferring Google's NotebookLM for reliable sourcing and better academic results. Concerns were raised on the lack of editing ability and control over sources in Perplexity Pages.

- **Integration for SAML SSO Requested**: A user requested assistance in integrating Perplexity.ai into their existing infrastructure using SAML-based Single Sign-On, seeking specific service provider metadata and instructions. This request emphasized the need for better support and documentation for enterprise integration.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/nwmsrocks-northwest-motorsport-pnw-pacific-northwest-toyota-gif-20681791">Nwmsrocks Northwest Motorsport GIF - Nwmsrocks Northwest Motorsport Pnw - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/you-know-it-wink-the-office-michael-scott-steve-carell-gif-17547389">You Know It Wink GIF - You Know It Wink The Office - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.reddit.com/r/ChatGPT/s/KrhcqUpEuq">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.perplexity.ai/search/Fr-SAMLSSO-bei-Cf6NgplsT8.QRLumYk3BNA">Für SAML-SSO bei perplexity.ai benötige ich die &#x27;Service Provider Metadata&#x27;...</a>: Basierend auf den bereitgestellten Suchergebnissen kann ich keine spezifischen Informationen zur SAML-SSO-Konfiguration für perplexity.ai finden. Die...
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1249814174788489279)** (9 messages🔥): 

- **iOS 18 Packs Massive Features**: The redesigned Photos app in iOS 18 uses advanced **machine learning algorithms** to automatically organize and categorize photos based on people, places, and events. Unprecedented customization options and custom emoji design adding a fun, creative touch to messaging are standout features ([source](https://www.perplexity.ai/page/What-Was-Introduced-9M8PxU85Tg2kA.6ubApgJw)).

- **WWDC 2024 Makes AI Central**: Apple's WWDC 2024 keynote introduced new software versions, emphasizing **AI integration** and software enhancements across devices. Highlights include iOS 18's Photos app redesign, iPadOS 18’s productivity enhancements, and **macOS 15**’s unified task management ([source](https://www.perplexity.ai/page/What-Was-Introduced-0laYmO7vS2mGryU354PYaA)).

- **Image-er is the Correct Pronunciation**: The correct pronunciation of "Imgur" is "image-er," confirmed by **Imgur staff** and supported by several Reddit discussions, despite various user pronunciations ([source](https://www.perplexity.ai/search/Whats-the-correct-2XyomPRVT0a3uF9C_ZDq0g)).

- **MARS5 Sets New TTS Standards**: Developed by CAMB.AI, the **MARS5 TTS model** uses a two-stage pipeline to generate high-quality prosody and emotion, suitable for applications like dubbing and translation. It excels in applying nuanced emotional performances in various scenarios like sports commentary and anime ([source](https://www.perplexity.ai/search/MARS-5-TTS-mLBhVSs_RRWWnypF.Aiv2Q)).
  
- **Lenny’s Newsletter on Product Management Insights**: Lenny provides valuable insights on **building products and growth strategies**, covering topics like successful pivots and defending feature team PMs. Subscribers gain access to in-depth articles and his podcast ([source](https://www.lennysnewsletter.com/p/how-to-use-perplexity-in-your-pm)).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/embed/hhX9EKtInok">YouTube</a>: no description found</li><li><a href="https://www.lennysnewsletter.com/p/how-to-use-perplexity-in-your-pm">How to use Perplexity in your PM work </a>: 27 examples (with actual prompts) of how product managers are using Perplexity today</li><li><a href="https://www.perplexity.ai/search/What-is-Perplexity-OC0LtdykRoC7dmzTcHXWVw#1">What is Perplexity AI?</a>: Perplexity AI is an AI-powered search engine designed to provide users with concise, accurate answers to their queries by leveraging large language models...</li><li><a href="https://www.perplexity.ai/page/What-Was-Introduced-0laYmO7vS2mGryU354PYaA">What Was Introduced at WWDC 2024?</a>: Apple&#x27;s WWDC 2024 keynote unveiled a wave of AI-powered features and software updates designed to deliver more intelligent, personalized experiences across...</li><li><a href="https://www.perplexity.ai/search/MARS-5-TTS-mLBhVSs_RRWWnypF.Aiv2Q">MARS 5, TTS model from Camb.ai</a>: MARS5 is a text-to-speech (TTS) model developed by CAMB.AI. It is designed to generate speech with high-quality prosody and emotion, making it suitable for a...</li><li><a href="https://www.perplexity.ai/page/What-Was-Introduced-9M8PxU85Tg2kA.6ubApgJw">What Was Introduced at WWDC 2024?</a>: Apple&#x27;s WWDC 2024 keynote unveiled a range of exciting updates and features, with a strong emphasis on artificial intelligence integration and software...</li><li><a href="https://www.perplexity.ai/search/Revise-into-a-t4taOXiIRU.nnMM4GwkMzQ#0">Revise into a more robust article. use the included &lt;scratchpad-think&gt;...</a>: Here is my attempt at revising the article into a more robust version using the provided scratchpad framework:  &lt;scratchpad&gt; Key information from the...</li><li><a href="https://www.perplexity.ai/search/How-does-singing-3icDgrnMSAixgVGK9r3TbQ">How does singing work in tonal languages?</a>: Singing in tonal languages like Chinese, Vietnamese, and Thai presents a unique challenge because the tones used to convey meaning in speech need to be...</li><li><a href="https://www.perplexity.ai/search/Whats-the-correct-2XyomPRVT0a3uF9C_ZDq0g">What&#x27;s the correct pronunciation of Imgur?</a>: Based on the search results, the correct pronunciation of &quot;Imgur&quot; is &quot;image-er&quot; or /ˈɪm.ɪdʒ.ər/.  While many people initially pronounced it as &quot;im-grr&quot; or...
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1249840516997320827)** (6 messages): 

- **Trouble integrating Perplexity API in custom GPTs**: A member expressed frustration with web search capabilities offered in Chat GPT for custom GPTs, seeking help to integrate alternatives like Serper, Tavily, or Perplexity API. They shared a [link to Perplexity API documentation](https://docs.perplexity.ai/discuss/65edc94038fa40001045873c) and asked for recommendations on how to implement it when a GPT cannot find an answer.
- **Updating model names might help**: Another member suggested that updating the model names (e.g., switching from pplx-70b-online to llama-3-sonar-large-32k-online) might resolve integration issues, but acknowledged that further review is needed.
- **API key exposed in shared code**: Code was shared to use the Perplexity API, but another member pointed out that the exposed API key should be deleted and a new one created for security purposes.

**Link mentioned**: <a href="https://docs.perplexity.ai/discuss/65edc94038fa40001045873c">Perplexity API with Custom GPT</a>: no description found

  

---



### **LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1249800442808893451)** (85 messages🔥🔥): 

- **Seeking tools for parsing PDFs locally**: A developer is looking for tools to parse structured forms in PDFs on-premise, seeking recommendations for **LLMs or scripts** to extract fields and answers. Someone suggested using **Langchain** for integrating local LLMs to extract fields from PDFs with minimal setup.

- **Issues with WebUI and LMStudio**: A member wants a WebUI to interact with LMStudio from another PC but finds no official package. Another member suggested using the bundled **llama.cpp server** for a simple interface, and checking out *[text-generation-webui](https://github.com/oobabooga/text-generation-webui/)* for web-based interaction.

- **Concerns over SB 1047 and Open Source AI**: A heated discussion arose about **SB 1047** and its impact on open-source AI. The bill is perceived as insidiously aiming to restrict AI development to a few companies, destroy open-source AI, and impose liability on model makers indefinitely. [Tweet](https://x.com/dan_jeffries1/status/1794740447052525609?s=46) linked for more details.

- **Adapter for RAM and GPU deficiencies**: A user discussed upgrading their RAM and GPU to run larger AI models like 70B. The consensus was to get a **more powerful GPU** with at least 24GB VRAM, as their current 6700 XT lacks ROCm support, and performance using **OpenCL** would be significantly slower.

- **Comparing GPUs for AI tasks**: Members discussed GPU recommendations, suggesting older server cards like **P40** for budget builds. For better performance, options like **7900XT(X)** and **used 3090** were suggested, with a note that **ROCm** is much faster than OpenCL for AMD cards. [ROCm information](https://www.amd.com/en/products/software/rocm.html).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.useanything.com/setup/llm-configuration/local/lmstudio">LMStudio LLM ~ AnythingLLM</a>: LMStudio is a popular user-interface, API, and LLM engine that allows you to download any GGUF model from HuggingFace and run it on CPU or GPU.</li><li><a href="https://x.com/dan_jeffries1/status/1794740447052525609?s=46">Tweet from Daniel Jeffries (@Dan_Jeffries1)</a>: I spent a few hours listening to Dan Hendyrcks, who runs the non-profit AI Safety group behind SB 1047, aka the California AI Control and Centralization Bill.   I find him charming, measured, intellig...</li><li><a href="https://github.com/oobabooga/text-generation-webui/">GitHub - oobabooga/text-generation-webui: A Gradio web UI for Large Language Models. Supports transformers, GPTQ, AWQ, EXL2, llama.cpp (GGUF), Llama models.</a>: A Gradio web UI for Large Language Models. Supports transformers, GPTQ, AWQ, EXL2, llama.cpp (GGUF), Llama models. - oobabooga/text-generation-webui
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1249871256224792677)** (5 messages): 

- **Boptruth-NeuralMonarch-7B debuts**: A member shared their successful model merge of **Boptruth-NeuralMonarch-7B** using [LazyMergekit](https://colab.research.google.com/drive/1obulZ1ROXHjYLn6PPZJwRR6GzgQogxxb?usp=sharing) and urged others to try it. The model works best with the Alpaca chat template and can be found on [Huggingface](https://huggingface.co/theprint/Boptruth-NeuralMonarch-7B).
- **Taming the Qwen2 72B Model**: Another member reported success in getting the **Dolphin 2.9.2 Qwen2 72B Q8** model to run on a 128MB M3 Max. They remarked that despite past difficulties with large multi-part models, this one performs reasonably well.
- **Llama3 Fine Tune Impresses**: One member is testing **Llama3-FiditeNemini-70B-Source.i1-Q6_K.gguf** and finds it superior to the base Llama3 instruct, noting its clever writing. They provided links to various [quants](https://huggingface.co/mradermacher/Llama3-FiditeNemini-70B-Source-i1-GGUF) and resources for usage.
- **Question on Prompt Issues**: A question was raised about whether the Llama3 fine-tune exhibits the common issue of random shouting in replies. The initial tester confirmed they have not observed this issue and is using it for role-play activities with satisfying results.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/mradermacher/Llama3-FiditeNemini-70B-Source-i1-GGUF">mradermacher/Llama3-FiditeNemini-70B-Source-i1-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/theprint/Boptruth-NeuralMonarch-7B">theprint/Boptruth-NeuralMonarch-7B · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1249936796905640036)** (3 messages): 

- **Confusion over Virtual Memory vs Physical Memory**: A user inquired about solving an issue related to memory, with another member clarifying that the large "commit size" seen is virtual memory reservation, not actual physical memory use. They highlighted that with full offloading to GPU, RAM usage remains low, citing an example where a 10GB commit size resulted in only 160MB of actual (private) use.
- **Page Faults and GPU VRAM Usage**: Further elaboration was provided explaining that page faults will increase because data isn't found in physical RAM and normally this would lead to disk reads. However, with GPU mapped memory, Windows can redirect these to GPU VRAM, or use direct access to VRAM to prevent page faults altogether.
  

---


### **LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1249815556249616466)** (38 messages🔥): 

- **P40 temperature and cooling discussions**: Members discussed the temperature and cooling solutions for their P40 cards, noting *"I found 'product specs' for P40, it says flow direction does not matter"* with thermal-throttling at 90C. Several links to [Aliexpress fans](https://aliexpress.ru/item/1005002259578351.html) were shared for potential cooling solutions. 

- **8700g performance intrigue**: One member mentioned testing an 8700g which *"can get 11 toks in lm studio now, with ability to address 32gb ram"*, suggesting it as an affordable option for running larger models. 

- **Concerns with LM Studio's multi-GPU performance**: There were complaints about LM Studio's inability to handle models across multiple GPUs efficiently, *"It bottlenecks on PCIe throughput"* and fails in partial GPU offload scenarios. A comparison to other tools like *llama.cpp* and *ollama* highlighted better multi-GPU support in those platforms.

- **Tesla V100 compatibility inquiry**: A user asked about running LM Studio with a Tesla V100, expressing concerns about potential compatibility or driver issues. Another member shared a [link](https://github.com/l4rz/running-nvidia-sxm-gpus-in-consumer-pcs) for further reading.

- **OS preferences for LM Studio**: Members debated whether LMStudio performs better on Windows or Linux, with one noting Windows *"just works"* while Linux requires specific distros and drivers and is in "beta". Another user commented, *"GPU interference is more-or-less the same on both OSes, CPU interference may be faster on Linux"*.
  

---


### **LM Studio ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/1249803467208265788)** (1 messages): 

- **AutogenStudio token limit bug fixed**: A member encountered an issue with AutogenStudio and the **TheBloke/Llama-2-7B-Chat-GGUF** model where the completion tokens were capped at 2. The problem was resolved with a workaround shared on [GitHub](https://github.com/microsoft/autogen/issues/2050) involving the max_tokens parameter.

**Link mentioned**: <a href="https://github.com/microsoft/autogen/issues/2050">[Bug]: [autogenstudio] agent llm send max_tokens: null · Issue #2050 · microsoft/autogen</a>: Describe the bug When max_tokens parameter is None, the agent send a frame /v1/chat/completions with max_tokens: null. In this case the LLM don&#39;t understand and and stop after the second token. St...

  

---


### **LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1249805942632222752)** (11 messages🔥): 

- **ROCm implementation for different tools**: Discussions are ongoing about the feasibility of using **ROCm** for **LM Studio** and extending its support to **Auto1111** or **Comfy**. One member notes that "ROCm implementation on A1111 is very hacky."
- **Stable.cpp and Zluda hooking into CUDA with AMD**: Members discuss the potential of a **stable.cpp project** and a tool called **Zluda** that might allow AMD to hook into CUDA. A member describes the integration as a significant *"grind"* and *"real challenge"*, yet to be solved.
- **Comparison of GPU acceleration tools**: When discussing different GPUs, members share experiences with tools like **CUDA**, **OpenCL**, and even **Metal** for building GPU-accelerated applications.
- **Frustration with SD.next interface**: Although some users have better performance with **Zluda** on **automatic1111**, there's a notable dislike for the **SD.next interface**. One member states, *"I just hate SD.next's interface."*
- **Best OS for LLMStudio**: There is a debate on whether it is better to run **LLMStudio on Windows or Linux (Ubuntu)**. Various user preferences and technical considerations, such as GPU compatibility and software support, shape these discussions.
  

---


### **LM Studio ▷ #[🛠-dev-chat](https://discord.com/channels/1110598183144399058/1234988891153629205/1249951378823188590)** (1 messages): 

- **Respect channel etiquette**: A member voiced their frustration about another user posting the same question in multiple channels. They highlighted that spamming multiple channels with the same query is considered *"bad etiquette"*.
  

---



### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1249800400857600184)** (95 messages🔥🔥): 

- **Apple Enters AI Space Intensifying Competition**: Members are excited about Apple's recent entry into the AI field, with one member noting, "Competition is going to be intense now that Apple has entered the space." Another member commented, "With Apple's ecosystem, having AI integrated would be amazing".
- **AI Features in New Apple Devices**: There's discussion around which Apple devices will support new AI features, with one commenter stating, "on-device needs 8gbs of RAM and A17 pro or better in terms of iPhones." This prompted concerns among users with older devices as they might have to trade in or upgrade.
- **Apple vs Cloud Computing Debate**: A lively debate surrounds Apple's use of both on-device AI features and cloud computing, with one member critiquing, "so they just threw out the PCC idea out of the window immediately after introducing it." Others discussed the necessity of cloud compute to handle more complex tasks securely.
- **Concerns Over AI and Data Privacy**: There's a strong sentiment against misuse of data by AI, with a member arguing, "This data is misused 100% of the time... it will be sold, your privacy will be violated, and your security will be compromised." Another member noted the dichotomy in tech enthusiasts being anti-cloud and anti-on-prem solutions.
- **Apple's Integrated AI at WWDC 2024**: Members are sharing excitement and resources about Apple's new "Apple Intelligence" introduced at WWDC 2024, with a member linking the [Apple Foundation Models overview](https://machinelearning.apple.com/research/introducing-apple-foundation-models). This showcases multiple generative models tailored for various user tasks across Apple's OS ecosystem.

**Link mentioned**: <a href="https://machinelearning.apple.com/research/introducing-apple-foundation-models">Introducing Apple’s On-Device and Server Foundation Models</a>: At the 2024 Worldwide Developers Conference, we introduced Apple Intelligence, a personal intelligence system integrated deeply into…

  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1249801913398984885)** (22 messages🔥): 

- **New Voice Mode MIA**: Users expressed frustration over the delayed release of the new voice mode from OpenAI, noting that the promise of "coming weeks" has stretched too long. One remarked on the ambiguity of timelines, suggesting it feels like empty promises given their financial investments.

- **ChatGPT Update Excitement Fizzles**: A user mentioned their excitement upon seeing an update for ChatGPT in the app store but faced issues with freezing during text generation. Another noted the waste of prompts due to the need for frequent refreshes, exacerbated by usage limits.

- **GPT Store Policy Violation Confusion**: A user reported not being able to edit or publish GPTs in the GPT Store due to supposed policy violations, despite no recent changes. Another user urged patience but lamented the lack of effective customer support from OpenAI.

- **GPTs and Internet Access**: A question about GPT's internet access was answered with assurance that GPTs can hit external APIs. A follow-up indicated users should look under the "Capabilities" section when configuring GPTs for more information.
  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1250052401180971029)** (10 messages🔥): 

- **Timezones mess up GPT completions**: A user raised the issue of handling timezones with the completions API for setting reminders globally. They expressed that despite adding time zone context, the returned times were often incorrect.

- **Convert UTC to user's time zone**: Another member suggested handling timezone conversions outside of GPT. They recommended using a function call to convert UTC times to the user's local timezone to avoid wasting tokens and hallucination risks.

- **Synthetic data might help**: The same member mentioned the potential benefits of fine-tuning the model to follow ISO 8601 standards using synthetic data. This approach was shared as a past successful strategy for handling datetime conversions.

- **Use of Google Primary Calendar**: For getting timezone context, the user mentioned they use the Google Primary Calendar of the user, aiming for consistency by making the model provide decisions in UTC.

- **Recommendation for consistency**: The member pointed out that while GPT-4 performs better, GPT-3.5 was unreliable for handling timestamps. They propose maintaining decisions in UTC for better consistency.
  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1250052401180971029)** (10 messages🔥): 

- **Timezone Challenges in Completions API**: User nav_archer_23316 raised an issue with the **Completions API** returning timestamps incorrectly in UTC, leading to mismatched reminders across different time zones. They are considering adding time zone context and timestamps within chat history to address this.
- **Converting Timestamps with a Library**: User zaki_1052 suggested letting GPT return time in UTC and then converting it using a *library* to the user's time zone. They emphasized minimizing token usage and preventing hallucinations by using function calls for conversions.
- **GPT Comparison for Time Management**: nav_archer_23316 noted that GPT-4 performs significantly better at managing timestamps compared to GPT-3.5, which was described as "totally terrible" for such tasks.
- **Consistent UTC Model Decisions**: To ensure consistency, nav_archer_23316 aims to have the model always give decisions in UTC. This decision is part of their approach to handle time zone discrepancies more effectively.
- **Using Google Calendar for Timezone Context**: For maintaining accurate time zones, nav_archer_23316 mentioned utilizing the Google Primary Calendar of the user to get the necessary timezone context.
  

---


### **OpenAI ▷ #[api-projects](https://discord.com/channels/974519864045756446/1037561385070112779/1250050588763226206)** (3 messages): 

- **Boost Team Efficiency with Hana AI**: Imagine having an all-knowing, forever-remembering, all-expert, 24/7, never-tired, polite, and helpful AI team member available on Google Chat. The Hana AI bot aims to "supercharge your teams" by being seamlessly integrated into Google Chat to enhance productivity and management capabilities.
- **Unlock Potential with Hana AI**: Learn more about Hana AI and how it simplifies day-to-day tasks for managers and executives, thus allowing them to focus on what truly matters. The bot promises to lighten the load and increase productivity with its versatile features.
- **Experience Hana AI for Free**: Hana AI offers a free forever plan and seeks user trials and feedback on their product. Start enhancing your team's productivity with Hana AI at no cost by signing up [here](https://hana.hanabitech.com).

**Link mentioned**: <a href="https://hana.hanabitech.com">Hana: Your AI-Powered Google Chat Assistant</a>: Enhance your team&#x27;s productivity with Hana, the AI-powered assistant by Hanabi Technologies, designed for seamless integration with Google Chat.

  

---



### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1249801940615827567)** (71 messages🔥🔥): 

- **Seeking Best Medical Diagnosis Models**: A user inquired about the best large language model (LLM) for medical diagnosis, but no specific model was mentioned in the responses.
  
- **Search for Programming Book Dataset**: A user asked if there was a dataset of programming books and articles, and a GitHub repository [amephraim/nlp](https://github.com/amephraim/nlp/tree/master/texts) with certain texts was shared, though it contained Harry Potter books.

- **Technical Issue with safetensors Files**: A user faced issues with safetensors files from Civitai, getting a `TypeError: argument of type 'NoneType' is not iterable` error while using `diffusers.StableDiffusionPipeline.from_single_file()`.

- **Concerns Over AI Control Legislation**: A user shared a Twitter thread critiquing the California AI Control and Centralization Bill, expressing concerns that it aims to restrict open-source AI development and impose harsh liabilities on model makers.

- **Checkpoint Usage for Models**: There was a conversation about using checkpoints from training sessions, with clarifications that each checkpoint can be loaded as a standalone model and the formats (e.g., `.pt` or safetensors) can be converted as needed.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/dan_jeffries1/status/1794740447052525609?s=46">Tweet from Daniel Jeffries (@Dan_Jeffries1)</a>: I spent a few hours listening to Dan Hendyrcks, who runs the non-profit AI Safety group behind SB 1047, aka the California AI Control and Centralization Bill.   I find him charming, measured, intellig...</li><li><a href="https://huggingface.co/spaces/vinthony/SadTalker">SadTalker - a Hugging Face Space by vinthony</a>: no description found</li><li><a href="https://huggingface.co/spaces/nroggendorff/dolphin">Dolphin - a Hugging Face Space by nroggendorff</a>: no description found</li><li><a href="https://github.com/amephraim/nlp/tree/master/texts">nlp/texts at master · amephraim/nlp</a>: Contribute to amephraim/nlp development by creating an account on GitHub.</li><li><a href="https://github.com/0xPlaygrounds/rig">GitHub - 0xPlaygrounds/rig: A library for developing LLM-powered Rust applications.</a>: A library for developing LLM-powered Rust applications. - 0xPlaygrounds/rig</li><li><a href="https://civitai.com/api/download/models/348913?type=Model&format=SafeTensor&size=full&fp=fp16&token=urtoken>">no title found</a>: no description found</li><li><a href="https://arxiv.org/abs/2406.05587">Creativity Has Left the Chat: The Price of Debiasing Language Models</a>: Large Language Models (LLMs) have revolutionized natural language processing but can exhibit biases and may generate toxic content. While alignment techniques like Reinforcement Learning from Human Fe...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1250060952309600257)** (4 messages): 

- **B&W Manga Model Shows Various Art Styles**: A user shared the [B&W Manga model](https://huggingface.co/alvdansen/BandW-Manga) on HuggingFace, showcasing illustrations like *"a boy in a sailor suit frowning"* and *"a girl with a flower crown"*. Others complimented the model, describing the artwork as "cute" and expressing their admiration with heart emojis.

- **Supermemory GitHub Project Promises to Help Forgetful Users**: A member posted a link to the [Supermemory project on GitHub](https://github.com/Dhravya/supermemory). This project aims to be a "ChatGPT for your bookmarks," allowing users to import tweets or save online content with a Chrome extension.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/alvdansen/BandW-Manga">alvdansen/BandW-Manga · Hugging Face</a>: no description found</li><li><a href="https://github.com/Dhravya/supermemory">GitHub - Dhravya/supermemory: Build your own second brain with supermemory. It&#39;s a ChatGPT for your bookmarks. Import tweets or save websites and content using the chrome extension.</a>: Build your own second brain with supermemory. It&#39;s a ChatGPT for your bookmarks. Import tweets or save websites and content using the chrome extension. - Dhravya/supermemory
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1249827785577595051)** (9 messages🔥): 

- **SoteDiffusion Wuerstchen3 goes live**: Shared an anime finetune of Würstchen V3, based on 6M images trained over 3 epochs. For API usage, they provided a link to [Fal.AI's documentation](https://fal.ai/models/fal-ai/stable-cascade/sote-diffusion).
- **Chat with multiple models on Hugging Face Spaces**: Introduced [Chat With 'Em](https://huggingface.co/spaces/as-cle-bert/chat-with-em), a customizable chat model space. Users can switch between models like Claude, GPT-3.5, GPT-4, and Llama-3 series by providing an API key.
- **Predict Formula 1 lap times**: Shared a project that predicts lap times using historical telemetry data, presented in a detailed [Kaggle notebook](https://www.kaggle.com/code/lucasdraichi/hamilton-lap-time-prediction). They sought feedback from the community.
- **CAMB AI unveils MARS5 TTS Model**: Announced the release of MARS5 TTS, open-sourced on [GitHub](https://github.com/camb-ai/mars5-tts) with a long post on Reddit for more details. The developer advocacy lead from Hugging Face expressed interest in collaborating.
- **Dalle 3 image captions dataset**: Released a dataset of over 1 million Dalle 3 images with high-quality captions, focusing on a wide range of concepts. Dataset available at [Hugging Face](https://huggingface.co/datasets/ProGamerGov/synthetic-dataset-1m-dalle3-high-quality-captions).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/posts/Draichi/560425192506443">@Draichi on Hugging Face: &quot;Hey Hugging Face Community 🤗

I&#39;m excited to share my latest project that…&quot;</a>: no description found</li><li><a href="https://huggingface.co/datasets/ProGamerGov/synthetic-dataset-1m-dalle3-high-quality-captions">ProGamerGov/synthetic-dataset-1m-dalle3-high-quality-captions · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/Disty0/sotediffusion-wuerstchen3">Disty0/sotediffusion-wuerstchen3 · Hugging Face</a>: no description found</li><li><a href="https://projectlove.life">Project Love Life</a>: no description found</li><li><a href="https://github.com/camb-ai/mars5-tts">GitHub - Camb-ai/MARS5-TTS: MARS5 speech model (TTS) from CAMB.AI</a>: MARS5 speech model (TTS) from CAMB.AI. Contribute to Camb-ai/MARS5-TTS development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1249821534709747712)** (3 messages): 

- **Check out CVPR 2024 Paper Summaries App**: A member shared a new app where they indexed all **CVPR 2024 paper summaries** and added **semantic search** to it. You can explore this tool [here](https://huggingface.co/spaces/pedrogengo/CVPR2024_search_papers).

- **Inquiry about Label Studio ML Backend**: Someone asked if any other members have ever used the **Label Studio ML backend**. No responses were recorded in the provided messages.

**Link mentioned**: <a href="https://huggingface.co/spaces/pedrogengo/CVPR2024_search_papers">CVPR2024 Search Papers - a Hugging Face Space by pedrogengo</a>: no description found

  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1249849718524084406)** (1 messages): 

- **Use `return_tensors="tf"` for TensorFlow Models**: If you have a TensorFlow model but are giving it PyTorch tensors, ensure you set `return_tensors="tf"` when using tokenizers. This resolves compatibility issues between TensorFlow models and input tensors.
  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1249826095805632584)** (6 messages): 

- **Comparing Finetuning Interfaces**: A member asked if various notebooks and GUI training interfaces still use **Hugging Face (HF)** for finetuning and sought recommendations for starting with **SDXL finetuning**. Another member noted that some are based on the original Stability codebase, while others use HF, but conveyed that each option has its own pros and cons without recommending a specific one.
- **Seeking Resources on Finetuning**: In response to a query for resources that outline the pros and cons of different finetuning interfaces, a member suggested the **SimpleTuner tutorial** and related docs, as they aim to explain the functionalities concisely.
- **Introducing MaPO**: Announcing new work on **MaPO**, a technique for aligning text-to-image diffusion models on preference datasets, a member highlighted its sample-efficiency and memory-friendliness. They also addressed the "reference mismatch" problem prevalent in preference datasets and shared the project [website and abstract](https://mapo-t2i.github.io/), asserting that their method requires fewer computational resources.

**Link mentioned**: <a href="https://mapo-t2i.github.io/">MaPO Project Page</a>: SOCIAL MEDIA DESCRIPTION TAG TAG

  

---



### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1249985508252057671)** (5 messages): 

- **k?d impresses with "Creator's Flower"**: A YouTube video titled ["k?d - Creator's Flower"](https://youtu.be/7yjJ43tI9aU) was shared. The video includes links for streaming and downloading the track.
  
- **Virtual Riot drops "Come With Me" feat. Leah Culver**: A YouTube link to ["Virtual Riot - Come With Me Ft. Leah Culver"](https://youtu.be/9HxK4O1bxkA) was mentioned. The video offers options for following the artist on Spotify and social media.
  
- **Porter Robinson's "Musician" official video announced**: The YouTube video ["Porter Robinson - Musician (Official Music Video)"](https://youtu.be/q-74HTjRbuY) was highlighted. The video description includes details about Porter Robinson's first world tour.
  
- **Xan Griffin features WOLFE in "Capricorn"**: A YouTube video for ["Capricorn (feat. WOLFE)"](https://youtu.be/rXD64OtlA40) by Xan Griffin was shared, auto-generated by YouTube. It was released under Seeking Blue.
  
- **Motionless In White releases "Werewolf"**: The official video for ["Motionless In White - Werewolf"](https://youtu.be/xzojuv9zMGA) was shared. The track is part of their album "Scoring The End Of The World," available via Roadrunner Records.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://youtu.be/7yjJ43tI9aU">k?d - Creator&#39;s Flower</a>: Stream + Download: https://altvision.lnk.to/findparadisek?dhttps://www.facebook.com/whoskidmusic/https://twitter.com/whoskidhttps://www.instagram.com/whos_ki...</li><li><a href="https://youtu.be/rXD64OtlA40">Capricorn (feat. WOLFE)</a>: Provided to YouTube by Seeking BlueCapricorn (feat. WOLFE) · Xan GriffinCapricorn (feat. WOLFE)℗ 2017. Seeking BlueReleased on: 2017-12-22Auto-generated by Y...</li><li><a href="https://youtu.be/xzojuv9zMGA">Motionless In White - Werewolf [Official Video]</a>: Motionless In White&#39;s official video for &#39;Werewolf&#39; - available now on Roadrunner Records.Listen to the new album &quot;Scoring The End Of The World&quot; out now: htt...</li><li><a href="https://youtu.be/9HxK4O1bxkA">Virtual Riot - Come With Me Ft. Leah Culver</a>: OUT NOW : https://disciple.fanlink.to/presetjep► Follow me on Spotifyhttps://goo.gl/4mgqJq► Connect with meFB: http://facebook.com/virtualriotmusicTwitter: h...</li><li><a href="https://youtu.be/q-74HTjRbuY">Porter Robinson - Musician (Official Music Video)</a>: Porter Robinson - Musician (Official Music Video)ANNOUNCING THE &quot;SMILE! :D WORLD TOUR&quot; — MY FIRST EVER WORLD TOUR !! presale starts tuesday, signup: https://...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1250149635083866133)** (1 messages): 

- **Nous Research releases Character Codex dataset**: Nous Research announced a new dataset named **Character Codex** that includes data on 15,939 characters from a wide array of sources, ranging from anime to historical figures and pop icons. You can download it on [HuggingFace](https://huggingface.co/datasets/NousResearch/CharacterCodex).

**Link mentioned**: <a href="https://huggingface.co/datasets/NousResearch/CharacterCodex">NousResearch/CharacterCodex · Datasets at Hugging Face</a>: no description found

  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1249800784900657235)** (68 messages🔥🔥): 

```html
- **Exploring Mutual Information**: A user asked, *"What is mutual information?"*, prompting another to share a [Wikipedia link](https://en.m.wikipedia.org/wiki/Mutual_information) explaining the concept as a measure of mutual dependence between two random variables in probability and information theory.

- **Discussion on CA SB 1047**: A strong critique of CA SB 1047 was shared, emphasizing its potential threat to open-source AI by imposing stringent controls and liabilities on model developers. Another user suggested a counter bill, SB 1048, to protect AI innovation. [Dan Jeffries' thread](https://x.com/dan_jeffries1/status/1794740447052525609?s=46) offers a detailed commentary on the topic.

- **Investigating RLHF on Creativity**: Users discussed a paper exploring the impact of Reinforcement Learning from Human Feedback (RLHF) on creativity in LLMs. They debated whether alignment techniques inherently stifle creativity or if less aggressive methods, like those used by companies such as Anthropic, avoid this pitfall. [Research paper link](https://arxiv.org/abs/2406.05587).

- **Rig Open Source Library Release**: The release of 'Rig,' an open-source Rust library for developing LLM-powered applications, was announced. The [GitHub repository](https://github.com/0xPlaygrounds/rig) provides an array of examples and modular components aimed at simplifying the development of AI agents.

- **Quantization and Model Pruning**: Users engaged in a detailed discussion on the challenges and techniques for quantizing and pruning large language models like LLaMA 3 8b. They referenced various approaches, including [LLM-Pruner](https://github.com/horseee/LLM-Pruner), for effectively reducing model size without significant performance degradation.
```
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/dan_jeffries1/status/1794740447052525609?s=46">Tweet from Daniel Jeffries (@Dan_Jeffries1)</a>: I spent a few hours listening to Dan Hendyrcks, who runs the non-profit AI Safety group behind SB 1047, aka the California AI Control and Centralization Bill.   I find him charming, measured, intellig...</li><li><a href="https://arxiv.org/abs/2406.05587">Creativity Has Left the Chat: The Price of Debiasing Language Models</a>: Large Language Models (LLMs) have revolutionized natural language processing but can exhibit biases and may generate toxic content. While alignment techniques like Reinforcement Learning from Human Fe...</li><li><a href="https://github.com/0xPlaygrounds/rig">GitHub - 0xPlaygrounds/rig: A library for developing LLM-powered Rust applications.</a>: A library for developing LLM-powered Rust applications. - 0xPlaygrounds/rig</li><li><a href="https://x.com/jvnixon/status/1799996074146578801?s=46">Tweet from Jeremy Nixon (@JvNixon)</a>: SB 1047 deserves a rejoinder!! Welcome to SB 1048.  📚The Freedom of AI Innovation Act.📚  It gifts AI the strongest arguments from Section 230, which protected the verdant ecosystem of the internet f...</li><li><a href="https://github.com/horseee/LLM-Pruner">GitHub - horseee/LLM-Pruner: [NeurIPS 2023] LLM-Pruner: On the Structural Pruning of Large Language Models. Support LLaMA, Llama-2, BLOOM, Vicuna, Baichuan, etc.</a>: [NeurIPS 2023] LLM-Pruner: On the Structural Pruning of Large Language Models. Support LLaMA, Llama-2, BLOOM, Vicuna, Baichuan, etc. - horseee/LLM-Pruner</li><li><a href="https://en.m.wikipedia.org/wiki/Mutual_information">Mutual information - Wikipedia</a>: no description found
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1249901251693383701)** (2 messages): 

- **Cohere uses multi-step retrieval**: A member noted that CoHere does retrieval over multiple agent calls, referred to as "connections", leading to multi-step construction of outputs. This CoT (Chain of Thought) approach explains confusing output references.
- **Hybrid retrieval methods discussed**: Another member proposed connecting **elastic search** with hybrid retrieval approaches like **bm25 + embedding** and web search. They questioned whether to also index the web-search results.
  

---



### **Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1249825492593545337)** (48 messages🔥): 

- **Apple unveils AI-powered system**: In a [bold move to redefine its ecosystem](https://asknews.app/en/stories/Apples-AI-Leap-Sparks-Controversy-Amid-Musks-Security-Concerns), Apple announced 'Apple Intelligence' at WWDC 2024, intended to enhance iPhones, iPads, and Macs. The integration of ChatGPT into Siri marks a significant shift aiming to provide a more personalized and conversational user experience.
  
- **Job application struggles and advice**: A user sought a referral from the Cohere team after being rejected twice, highlighting their hackathon wins and work experience in ML and LLM. Conversations around the effectiveness of referrals ensued, with multiple users advising that strong credentials outweigh the need for a referral.
  
- **Developer Office Hours announced**: Cohere announced new Developer Office Hours encouraging members to bring their best questions. [Next session details](https://discord.gg/6aFP6F4Ecj?event=1248300905703673987) were shared, and feedback about the first session's format was actively requested from the participants.

- **Positive feedback on engagement**: Users praised the new office hours format for its engagement and the Cohere team's approachable, laid-back demeanor. The team responded positively, appreciating the participation and encouraging further feedback.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://discord.gg/6aFP6F4Ecj?event=1248300905703673987">Join the Cohere Community Discord Server!</a>: Cohere community server. Come chat about Cohere API, LLMs, Generative AI, and everything in between. | 16987 members</li><li><a href="https://asknews.app/en/stories/Apples-AI-Leap-Sparks-Controversy-Amid-Musks-Security-Concerns">AskNews | Apple&#x27;s AI Leap Sparks Controversy Amid Musk&#x27;s Security Concerns</a>: In a bold move to redefine its ecosystem, Apple has unveiled &#x27;Apple Intelligence&#x27;, a new AI-powered system announced at WWDC 2024, poised to enhance the functionality of iPhones, iPads, and ...
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[announcements](https://discord.com/channels/954421988141711382/996880279224451154/1250019425575370792)** (1 messages): 

- **Join Cohere Developer Office Hours Today**: *"Join us today for Cohere Developer Office Hours!"* This event is an opportunity to troubleshoot issues, get your questions answered, and discuss everything related to Cohere's models and API. The event will be held today, June 11, at 1:00 PM ET, and you can join via this [Discord link](https://discord.gg/7zjrJmKtBB?event=1248300806600392766).

**Link mentioned**: <a href="https://discord.gg/7zjrJmKtBB?event=1248300806600392766">Join the Cohere Community Discord Server!</a>: Cohere community server. Come chat about Cohere API, LLMs, Generative AI, and everything in between. | 16987 members

  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1249801456870101013)** (40 messages🔥): 

- **Apple Layers AI onto OS**: @karpathy shared major themes from Apple's announcement about integrating AI into their OS. Key points include fostering a multimodal I/O, creating a seamless and anticipatory user experience, leveraging local and cloud computations, and maintaining privacy standards. [Full thread](https://x.com/karpathy/status/1800242310116262150?s=46&t=90xQ8sGy63D2OtiaoGJuww)
- **Integration with ChatGPT**: OpenAI announced a partnership with Apple to integrate ChatGPT into iOS, iPadOS, and macOS, launching later this year. [Source](https://x.com/openai/status/1800240380220473552?s=46&t=90xQ8sGy63D2OtiaoGJuww)
- **Private Cloud Compute**: Apple introduced a secure system called "Private Cloud Compute" allowing phones to offload complex AI tasks securely. @Matthew_D_Green and others discussed the advanced security features and implications. [More details](https://x.com/matthew_d_green/status/1800291897245835616?s=46&t=90xQ8sGy63D2OtiaoGJuww), [Blog Post](https://security.apple.com/blog/private-cloud-compute/)
- **Mistral's Funding**: Mistral announced a €600M Series B funding round for global expansion, garnering reactions about the AI funding landscape. [Full announcement](https://x.com/arthurmensch/status/1800558395872731379?s=46&t=46), [Pitch deck discussion](https://x.com/chiefaioffice/status/1800581527480274984?s=46)
- **Pgvectorscale Challenges Pinecone**: Timescale introduced "pgvectorscale," an open-source extension for PostgreSQL, claiming it achieves better performance and cost-efficiency compared to Pinecone for AI applications. [Details](https://x.com/avthars/status/1800517917194305842)
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/maxwinebach/status/1800277157135909005?s=46">Tweet from Max Weinbach (@MaxWinebach)</a>: This is from Apple&#39;s State of the Union  The local model is a 3B parameter SLM that uses adapters trained for each specific feature. Diffusion model does the same thing, adapter for each style.  A...</li><li><a href="https://x.com/osanseviero/status/1800607752038818260?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Omar Sanseviero (@osanseviero)</a>: RecurrentGemma 9B by Google is out 🔥  ⚡️Super fast for long sequences: Good throughput+latency 👀Base and instruct tuned versions 🏆Similar quality as Gemma  Check the y-axis below 🤯  Models: https:...</li><li><a href="https://x.com/bilawalsidhu/status/1800355980829405603?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Bilawal Sidhu (@bilawalsidhu)</a>: Ok I take it back. Apple’s ‘Private Cloud Computing’ actually takes ‘Confidential Computing’ to the next level. It’s SO secure that they can’t even comply with law enforcement requests.  &gt; No data ...</li><li><a href="https://x.com/suhail/status/1800265203915055221?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Suhail (@Suhail)</a>: I am an old man now after two platform waves but what Apple did today was communicate:  &#34;Hi guys, we made native integration points to make all you all AI model makers compete for our 1B-user scal...</li><li><a href="https://x.com/arthurmensch/status/1800558395872731379?s=46">Tweet from Arthur Mensch (@arthurmensch)</a>: We are announcing €600M in Series B funding for our first anniversary.  We are grateful to our new and existing investors for their continued confidence and support for our global expansion. This will...</li><li><a href="https://x.com/chiefaioffice/status/1800581527480274984?s=46">Tweet from Chief AI Officer (@chiefaioffice)</a>: BREAKING: Mistral raises a $640M Series B led by General Catalyst at a $6B valuation.  Here&#39;s their Seed pitch deck to remind you of their vision:</li><li><a href="https://x.com/levie/status/1800224021193396594">Tweet from Aaron Levie (@levie)</a>: iPad calculator is actually pretty nuts</li><li><a href="https://x.com/nickadobos/status/1800289718439186455?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Nick Dobos (@NickADobos)</a>: Siri can read EVERY piece of data on your phone (for apps that opt in)</li><li><a href="https://x.com/chefjeffsf/status/1800597192593621100">Tweet from Chef Jeff (@chefjeffsf)</a>: Breaking: Google just published a Personal Health Large Language Model  - Fine-tuned on Gemini - Reads your wearable data to find personalized insights and recommendations - Outperformed professional ...</li><li><a href="https://x.com/matthew_d_green/status/1800291897245835616?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Matthew Green (@matthew_d_green)</a>: So Apple has introduced a new system called “Private Cloud Compute” that allows your phone to offload complex (typically AI) tasks to specialized secure devices in the cloud. I’m still trying to work ...</li><li><a href="https://x.com/karpathy/status/1800242310116262150?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Andrej Karpathy (@karpathy)</a>: Actually, really liked the Apple Intelligence announcement. It must be a very exciting time at Apple as they layer AI on top of the entire OS. A few of the major themes.  Step 1 Multimodal I/O. Enable...</li><li><a href="https://x.com/reach_vb/status/1800293882585919620?s=46">Tweet from Vaibhav (VB) Srivastav (@reach_vb)</a>: The unsaid star of WWDC & Apple  pip install mlx and get access to plethora of multimodal, audio and LLMs (fully open source)  https://github.com/ml-explore/mlx</li><li><a href="https://x.com/avthars/status/1800517917194305842">Tweet from Avthar (@avthars)</a>: PGVECTOR IS NOW FASTER THAN PINECONE. And 75% cheaper thanks to a new open-source extension – introducing pgvectorscale.  🐘 What is pgvectorscale? Pgvectorscale is an open-source PostgreSQL extension...</li><li><a href="https://x.com/mkbhd/status/1800223468627304657?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Marques Brownlee (@MKBHD)</a>: Ok you know what? That&#39;s sick  Math Notes = write down a math problem with Apple pencil and the app solved it immediately   They&#39;re not calling it AI (they haven&#39;t said it once yet) but th...</li><li><a href="https://x.com/elonmusk/status/1800265431078551973?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Elon Musk (@elonmusk)</a>: If Apple integrates OpenAI at the OS level, then Apple devices will be banned at my companies. That is an unacceptable security violation.</li><li><a href="https://x.com/stevesi/status/1800314848070557864?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Steven Sinofsky (@stevesi)</a>: In case it isn&#39;t clear, what Apple has done is the reverse of the search deal (to OpenAI). Rather than get paid, whether they pay a lot or a little it won&#39;t matter it will be for a finite time...</li><li><a href="https://x.com/matthew_d_green/status/1800291897245835616?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Matthew Green (@matthew_d_green)</a>: So Apple has introduced a new system called “Private Cloud Compute” that allows your phone to offload complex (typically AI) tasks to specialized secure devices in the cloud. I’m still trying to work ...</li><li><a href="https://x.com/stalman/status/1800278850435190871?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Tyler Stalman (@stalman)</a>: Apple says they will eventually integrate the Google Gemini model</li><li><a href="https://security.apple.com/blog/private-cloud-compute/">Blog - Private Cloud Compute: A new frontier for AI privacy in the cloud - Apple Security Research</a>: Secure and private AI processing in the cloud poses a formidable new challenge. To support advanced features of Apple Intelligence with larger foundation models, we created Private Cloud Compute (PCC)...</li><li><a href="https://x.com/openai/status/1800240380220473552?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from OpenAI (@OpenAI)</a>: We’re partnering with Apple to integrate ChatGPT into iOS, iPadOS, and macOS—coming later this year: https://openai.com/apple</li><li><a href="https://www.ft.com/content/7a70a8a6-4a2a-47c5-8483-d0b829f32ae6">Mistral secures €600mn funding as valuation soars to almost €6bn </a>: no description found
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1250114907349717043)** (1 messages): 

- **Mike Conover returns on the Latent Space pod**: The new podcast episode features Mike Conover discussing his extensive hands-on experience deploying LLMs in production. You can listen to the episode [here](https://x.com/FanaHOVA/status/1800553625607155856).
  
- **AI and Finance insights from Vagabond Jack**: Vagabond Jack joins LatentSpacePod to share AI Engineering strategies used at BrightWaveIO for clients with over $120B under management. Topics include losing faith in long context windows, LLMs as judges, the uselessness of anthropomorphizing models, and dataset half-lives relative to finetuned models.

**Link mentioned**: <a href="https://x.com/FanaHOVA/status/1800553625607155856">Tweet from Alessio Fanelli (@FanaHOVA)</a>: How AI is eating Finance 📈  @vagabondjack is back on @latentspacepod! He shared all the AI Engineering wisdom he acquired while turning LLMs into AI thought partners @brightwaveio for customers with ...

  

---



### **LlamaIndex ▷ #[announcements](https://discord.com/channels/1059199217496772688/1073670729054294197/1249874265130537001)** (1 messages): 

- **Sign Up for Advanced Knowledge Graph RAG Workshop**: Register for a special workshop on "advanced knowledge graph RAG" happening this Thursday at 9am PT, featuring Tomaz Bratanic from Neo4j. [Sign up here](https://lu.ma/kqxmbuou) to learn about using LlamaIndex property graph abstractions.

**Link mentioned**: <a href="https://lu.ma/kqxmbuou">LlamaIndex Webinar: Advanced RAG with Knowledge Graphs (with Tomaz from Neo4j) · Zoom · Luma</a>: We’re hosting a special workshop on advanced knowledge graph RAG this Thursday 9am PT, with the one and only Tomaz Bratanic from Neo4j. In this webinar, you’ll…

  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1250111440195555418)** (1 messages): 

- **Join the Fun at Paris AI Meetup**: Catch a live demo by @hexapode at the [Paris Local & Open-Source AI Developer meetup](https://t.co/5GLV08cGFa) on Thursday, 20th June at Station F in Paris, starting at 6:00pm. The event features lightning demos from Koyeb, LlamaIndex, Giskard, Red Hat, Docker, and more, followed by networking.
- **Demo Opportunities Available**: Interested participants can demo their projects by filling out [this form](https://forms.gle/YMXvYCVhuuppTWTp7). Networking and giveaways, including Ollama keychains, will add to the night’s excitement.

**Link mentioned**: <a href="https://t.co/5GLV08cGFa">Paris Open-source AI developer meetup  · Luma</a>: Docker and Friends are in Paris! Docker and Friends will be hosting a local &amp; open-source AI developer meetup on Thursday, 20 June at 6:00pm at Station F in…

  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1249813091382984787)** (29 messages🔥): 

- **Deleting nodes with a common ref_doc_id**: A user inquired about how to delete all related nodes when a Page is deleted by setting the same `ref_doc_id` for all documents on the Page. They attempted to create a parent Document but faced issues with embedding and asked if it’s possible to exclude certain Documents from embedding.

- **UC Berkeley team seeks RAG insights**: A user from UC Berkeley expressed challenges in building, deploying, and maintaining custom RAG systems and sought feedback from engineers. They invited anyone with experience to chat with them to understand common difficulties and potential solutions.

- **Exploring LLM pipelines and RAG with LlamaIndex**: Users discussed integrating multiple query engines like SQL, Vector Search, Keyword Search, and Image Search using LlamaIndex. Suggestions included using `RouterQueryEngine` and integrating with Qdrant to utilize its features, and considering model deployment on Huggingface for efficient vector generation.

- **Running SQL db retrieval and data analysis with Llama 3**: A user working on SQL db retrieval and data analysis using LLMs, specifically Llama 3, asked for guidance on integration issues. They discussed potential solutions like using text-to-SQL pipelines and verifying Llama 3’s responses.

- **Generating sparse vectors efficiently**: Another user faced issues with the slow process of generating and uploading sparse vectors in hybrid mode with Qdrant and LlamaIndex. Suggestions included running sparse embeddings on GPU locally or via an API for improved efficiency.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.llamaindex.ai/en/stable/examples/pipeline/query_pipeline_sql/?h=text2">Query Pipeline for Advanced Text-to-SQL - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/evaluation/multi_modal/multi_modal_rag_evaluation/#build-our-multi-modal-rag-systems>).">Evaluating Multi-Modal RAG - LlamaIndex</a>: no description found
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1249974605313998949)** (1 messages): 

- **Creating tool functions in LlamaIndex**: A user inquired about setting up two tool functions in LlamaIndex, one for querying a vector database and another for using the OpenAI Chat Completion API when no product is found. They questioned if the agent could decide which tool to use and asked for recommendations on the appropriate agent framework, mentioning ReAct.
  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1249872052731510844)** (27 messages🔥): 

- **LAION makes news in Brazil but not positively**: A member mentioned seeing LAION referenced on TV in Brazil, but it **wasn't in a good context**. Another linked a Human Rights Watch article criticizing AI tools for misusing children's personal photos [link here](https://www.hrw.org/news/2024/06/10/brazil-childrens-personal-photos-misused-power-ai-tools).

- **Discussion on image privacy and internet misconceptions**: Members debated about people misunderstanding public data privacy online. One summarized, *"The fundamental issue with the internet is we have billions of people using it and none of them understand what it implies."*

- **LlamaGen image generation model announced**: A member shared an [arXiv paper](https://arxiv.org/abs/2406.06525) introducing **LlamaGen**, a new family of image generation models applying **next-token prediction** from LLMs to visual generation. The model achieved impressive benchmarks, outperforming popular diffusion models.

- **CAMB AI opens MARS5 TTS model**: **Arsalan from CAMB AI** announced that MARS5, a **new speech model (TTS)**, is now open source on [GitHub](https://github.com/camb-ai/mars5-tts). He also shared a more detailed [Reddit post](https://www.reddit.com/r/CAMB_AI/comments/1day7ta/introducing_mars5_opensource_insanely_prosodic/) for further reading and feedback.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2406.06525">Autoregressive Model Beats Diffusion: Llama for Scalable Image Generation</a>: We introduce LlamaGen, a new family of image generation models that apply original ``next-token prediction&#39;&#39; paradigm of large language models to visual generation domain. It is an affirmative...</li><li><a href="https://github.com/camb-ai/mars5-tts">GitHub - Camb-ai/MARS5-TTS: MARS5 speech model (TTS) from CAMB.AI</a>: MARS5 speech model (TTS) from CAMB.AI. Contribute to Camb-ai/MARS5-TTS development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1249869396038516756)** (4 messages): 

- **LlavaGuard Introduced**: A member shared a [link](https://ml-research.github.io/human-centered-genai/projects/llavaguard/index.html) to the LlavaGuard project presented by TU Darmstadt and others, which is focused on **safeguarding visual datasets** using VLM-based models. The paper emphasizes its suitability for **dataset annotation** and **safety compliance** with context-aware safety risks.

- **'Alice' vs. 'A girl' Model Performance Test**: Another member noted significant results when replacing "Alice" with "A girl" in multiple models, stating that it worked well. They inquired about any expected change in model performance when comparing scenarios "with names" vs. "without names", sharing anecdotal evidence through a screenshot.



**Link mentioned**: <a href="https://ml-research.github.io/human-centered-genai/projects/llavaguard/index.html">LlavaGuard - Project Page</a>: We introduce LlavaGuard, a family of VLM-based safeguard models, offering a versatile framework for evaluating the safety compliance of visual content. Specifically, we designed LlavaGuard for dataset...

  

---



### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1249801393263480984)** (25 messages🔥): 

- **Apple Intelligence generates mixed reactions**: Members express skepticism about the depth of OpenAI's integration with Apple Intelligence, feeling it "feels like an afterthought" despite privacy protections highlighted in the official [Apple Newsroom](https://www.apple.com/newsroom/2024/06/introducing-apple-intelligence-for-iphone-ipad-and-mac/) announcement. Concerns were raised about Apple's privacy claims and the actual user’s privacy when linking ChatGPT accounts.
  
- **Contrasts with Siri**: The group notes Apple’s decision to differentiate Apple Intelligence from Siri, hinting it might be a strategic move to distance the new feature from Siri’s existing reputation. This differentiation could play a crucial role in user perception of the service's effectiveness.

- **Excited for Dwarkesh's interview with François Chollet**: Members are looking forward to an upcoming episode where Dwarkesh Patel interviews François Chollet, anticipating a "more skeptical" perspective on AGI timelines. Participants express hope that Patel will prepare by reading Chollet’s work on the measure of intelligence to ensure a productive discussion. 

- **Benchmark leaks for Apple's models**: A link to a Twitter post ([source: Apple](https://x.com/ldjconfirmed/status/1800355063120151031?s=46)) showcases benchmarks for Apple's new on-device and server models. Discussions suggest keen interest in how these models stack up against other popular models in instruction following and writing abilities.

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.apple.com/newsroom/2024/06/introducing-apple-intelligence-for-iphone-ipad-and-mac/">Introducing Apple Intelligence for iPhone, iPad, and Mac</a>: Apple today introduced Apple Intelligence, the personal intelligence system for iPhone, iPad, and Mac.</li><li><a href="https://x.com/ldjconfirmed/status/1800355063120151031?s=46">Tweet from LDJ (@ldjconfirmed)</a>: If anyone is curious, here are some benchmarks for Apples new on-device model and server model, versus other popular models at instruction following and writing abilities.
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/1249837969246916749)** (4 messages): 

- **Implementing TRL discussed amidst concerns**: A participant considered implementing a paper for **TRL** but was cautioned by another that the work is *"unproven"* rather than merely *"messy"*. They expressed an interest in contributing despite the concerns.
- **Pull Request (PR) intent and support expressed**: The participant mentioned their potential submission of a **PR** and received an offer for a review. Encouragement for contribution to the TRL implementation was provided: "Lmk if you submit a PR, would happily review".
  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1249808505150640310)** (25 messages🔥): 

- **Apple Intelligence sparks interest, potential API integration**: A member shared a [link](https://www.apple.com/apple-intelligence/) to Apple Intelligence, highlighting its privacy-focused AI features built into iPhone, iPad, and Mac. Another member noted the potential for expanding Open Interpreter to integrate with this developer API.

- **CA SB 1047 faces sharp criticism**: A [tweet by Dan Jeffries](https://x.com/dan_jeffries1/status/1794740447052525609?s=46) criticized Dan Hendyrcks and the California AI Control and Centralization Bill, asserting it aims to centralize AI, destroy open source AI, and impose burdensome liability on model makers.

- **Arduino IDE issue resolved with a pull request**: A member encountered an error using Arduino IDE on Mac M1 and resolved it by applying a fix from this [GitHub pull request](https://github.com/lacamera/ESPAsyncWebServer/pull/2/files). However, they faced further issues with the device not showing the Wi-Fi setup popup after restarting.

- **Debate on platform focus for OI**: Members discussed whether Open Interpreter should concentrate more on Linux to provide AI computer-assisted functionalities outside of Apple and Microsoft ecosystems. There was also a mention of Open Interpreter potentially offering functionalities that Apple Intelligence cannot.

- **Developing a true assistant with memory and skills**: A member detailed their work on an Open Interpreter prompting system that can store, search, and manage skills via prompts, aiming to create a true personal assistant capable of retaining user-specific information and memories.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.apple.com/apple-intelligence/">Apple Intelligence Preview</a>: Apple Intelligence is personal intelligence for the things you do every day. Built into iPhone, iPad, and Mac with groundbreaking privacy.</li><li><a href="https://x.com/dan_jeffries1/status/1794740447052525609?s=46">Tweet from Daniel Jeffries (@Dan_Jeffries1)</a>: I spent a few hours listening to Dan Hendyrcks, who runs the non-profit AI Safety group behind SB 1047, aka the California AI Control and Centralization Bill.   I find him charming, measured, intellig...</li><li><a href="https://github.com/lacamera/ESPAsyncWebServer/pull/2/files">Ready for ESP32 V3 &amp;V2 by ednieuw · Pull Request #2 · lacamera/ESPAsyncWebServer</a>: Changed ESP32 board to 3.0.0 Changed in Arduino\libraries\ESPAsyncWebServer\src\WebAuthentic at line 75,76,77 //----------------- #ifdef ESP_ARDUINO_VERSION_MAJOR #if ESP_ARDUINO_VERSION &amp;gt;= ESP...
</li>
</ul>

</div>
  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1250047220946833520)** (3 messages): 

- **Killian's Recent Talk Turns Heads**: In a brief exchange, members referenced a recent talk by Killian. One pointed out, *"it was in the recent talk Killian did, its recorded somewhere"* and shared a link for further details. [Recording Here](https://discord.com/channels/1146610656779440188/1147665339266650133/1248858812761509938).
  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1249812420655059085)** (24 messages🔥): 

- **Custom prompts in create_tagging_chain are ignored**: A member raised concerns that all types of prompts get ignored when using `create_tagging_chain()`. No workaround or solution was provided in the discussion.
- **RAG systems at UC Berkeley**: A team member from UC Berkeley is seeking input from engineers who build Retrieval-Augmented Generation (RAG) systems to understand the challenges they face in successfully building, deploying, and maintaining these systems. They are inviting engineers to chat with them to gather insights.
- **LangGraph vs Traditional LangChain**: A member asked for clarification on the benefits of instantiating agents in LangGraph versus using traditional LangChain. They sought specific insights on whether LangGraph applications can be implemented as controlled scripts.
- **Using ONNX with LangChain**: A user inquired about the compatibility of ONNX with LangChain, but no detailed discussion or response was provided.
- **Processing large datasets with OpenAI API**: A detailed guide was provided on utilizing the OpenAI API for processing large datasets, which included steps on setting environment variables, data anonymization, avoiding hallucinations, and using retrievers like Elasticsearch and Milvus for efficient data retrieval. Links to relevant LangChain documentation and GitHub issues were included to assist with implementation.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://python.langchain.com/docs/modules/chains/how_to/openai_functions>).">Chains | 🦜️🔗 LangChain</a>: Chains refer to sequences of calls - whether to an LLM, a tool, or a data preprocessing step. The primary supported way to do this is with LCEL.</li><li><a href="https://github.com/langchain-ai/langchain/issues/6723>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://python.langchain.com/v0.2/docs/how_to/extraction_examples/#create-an-extractor>)">How to use reference examples when doing extraction | 🦜️🔗 LangChain</a>: The quality of extractions can often be improved by providing reference examples to the LLM.
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/)** (1 messages): 

unaiarambarri: When will the langserve server be available for JS/TS? Thanks!
  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1249844151499886676)** (1 messages): 

- **Chat with Top AI Models on Hugging Face**: A member introduces [Chat With 'Em](https://huggingface.co/spaces/as-cle-bert/chat-with-em), a customizable chat model on Hugging Face Spaces that supports Groq, Anthropic, OpenAI, and Cohere models. Users can easily switch among models like Claude, Command-R, GPT-3.5, GPT-4o, Llama-3-8B, Llama-3-70B, and Mixtral 8x7b by providing an API key, thanks to LangChain.
  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1250094589034106880)** (14 messages🔥): 

- **New member inquiries about access**: A new member asked how to get access to the bounties channel to work on the AMX support bounty. *"I get the 'You don't have permission to send messages on this channel' when I enter it."*
- **Access explained by George Hotz**: George Hotz mentioned that the bounties channel is a development channel requiring a higher access level. *"Become a purple and you can talk in it."*
- **Clarification on AMX support bounty**: The new member inquired whether the AMX support bounty involves adding matrix operations in the runtime files `tinygrad/runtime/ops_llvm.py` or `tinygrad/runtime/ops_clang.py`. George Hotz responded by asking if they had read the questions document.
- **Questions document confusion**: The new member initially referred to a different document related to asking smart questions. George Hotz humorously acknowledged the mix-up, calling it *"a real chicken and egg problem."*
- **Member decides to revisit the code**: After the back-and-forth, the new member decided to spend more time reading the tinygrad code before asking a refined question. They planned to return with a better-formulated inquiry.

**Link mentioned**: <a href="http://www.catb.org/~esr/faqs/smart-questions.html">How To Ask Questions The Smart Way</a>: no description found

  

---



### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1249880204634226758)** (11 messages🔥): 

- **OpenRouter Uses Edge Networks for Speed**: Users inquired about the location of OpenRouter servers and whether latency is an issue. The response clarified that OpenRouter leverages both **Vercel Edge** and **Cloudflare Edge**, ensuring nodes are close to the user to minimize latency.
  
- **Provider Selection Feature Queued**: A user asked if it is possible to select the API provider in the OpenRouter playground. The response confirmed that this feature is in the queue, hinting at future availability.

- **Direct API Provider Selection Available**: Another user noted that selecting the provider is not currently available in OpenRouter. However, it was pointed out that provider selection can be done via the API, with detailed instructions available in the [OpenRouter documentation](https://openrouter.ai/docs/provider-routing).

**Link mentioned**: <a href="https://openrouter.ai/docs/provider-routing">Provider Routing | OpenRouter</a>: Route requests across multiple providers

  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1249821600375771216)** (2 messages): 

- **ShareGPT formatting clarified**: A member clarified that **ShareGPT** will just be converted into the model's prompt format and is not visible to the model during training.
- **Benchmark competition for Apple models**: A member shared a link to [benchmarks for Apple's new on-device and server models](https://x.com/ldjconfirmed/status/1800355063120151031), comparing their instruction following and writing abilities to other popular models.

**Link mentioned**: <a href="https://x.com/ldjconfirmed/status/1800355063120151031">Tweet from LDJ (@ldjconfirmed)</a>: If anyone is curious, here are some benchmarks for Apples new on-device model and server model, versus other popular models at instruction following and writing abilities.

  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1249947721104621608)** (7 messages): 

- **Rakuten’s LLM tops charts in Japan**: A user shared a link to a blog post about Rakuten's AI engineers and scientists unveiling a suite of large language models that excel in Japanese. The models are based on [Mistral-7B](https://mistral.ai/news/announcing-mistral-7b/) and are available under a commercial license.
- **Rakuten models impress community**: Another user expressed approval, saying they would check it out, agreeing that the Rakuten models seem decent. There's a general consensus that the models are promising.
- **Humorous reaction to JSON response**: Members humorously commented on the model responding in JSON, illustrating both amusement and surprise at the model's capabilities. One particular reaction was encapsulated with *"this model is really something"*.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://rakuten.today/blog/rakutens-open-llm-tops-performance-charts-in-japanese.html">Rakuten&#039;s Open LLM Tops Performance Charts in Japanese</a>: Led by Rakuten Group Chief Data Officer Ting Cai, Rakuten&#039;s AI team unveiled a suite of large language models with exceptional performance in Japanese.</li><li><a href="https://rakuten.today/blog">Rakuten Today: Blog</a>: The latest and greatest from around Rakuten Group
</li>
</ul>

</div>
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[docs](https://discord.com/channels/1104757954588196865/1167137552470392842/1250127543051485298)** (1 messages): 

- **RunPod's tutorial eases fine-tuning with Axolotl**: A member shared a [RunPod tutorial on fine-tuning with Axolotl](https://docs.runpod.io/tutorials/pods/fine-tune-llm-axolotl), emphasizing how Axolotl simplifies training large language models (LLMs). They highlighted Axolotl's user-friendly workflow and comprehensive YAML examples for various LLM families, which help users fine-tune models efficiently using RunPod's GPU resources.

**Link mentioned**: <a href="https://docs.runpod.io/tutorials/pods/fine-tune-llm-axolotl">Fine tune an LLM with Axolotl on RunPod | RunPod Documentation</a>: Learn how to fine-tune large language models with Axolotl on RunPod, a streamlined workflow for configuring and training AI models with GPU resources, and explore examples for LLaMA2, Gemma, LLaMA3, a...

  

---



### **Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/1249980196560310313)** (4 messages): 

- **Vincent Warmerdam's Training Recommendations**: Vincent Warmerdam mentioned that he trains models for his employer and recommended [calmcode.io](https://calmcode.io). A user acknowledged having watched "basically all" of the Calmcode videos.
  
- **Chunking Strategies in RAG Explored**: A link to [a blog post on Stack Overflow](https://stackoverflow.blog/2024/06/06/breaking-up-is-hard-to-do-chunking-in-rag-applications/) was shared, discussing chunking strategies for retrieval-augmented generation (RAG) applications. The post emphasized the importance of grounding LLM responses in source data to mitigate inaccuracies and hallucinations, using text embeddings to place source text within the semantic space of LLMs.

**Link mentioned**: <a href="https://stackoverflow.blog/2024/06/06/breaking-up-is-hard-to-do-chunking-in-rag-applications/">Breaking up is hard to do: Chunking in RAG applications - Stack Overflow</a>: no description found

  

---



### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1249843482864914453)** (2 messages): 

- **Query on KL Plots for DPO Implementation**: A member asked another if they had TRL's KL plots for their DPO implementation comparison experiments [here](https://github.com/pytorch/torchtune/pull/645#issuecomment-2041861215). The queried member responded that although they did not plot KLs, there are KL plots available in TRL's PPO trainer, linking to the [relevant code](https://github.com/huggingface/trl/blob/34ebc4ccaf376c862a081ff4bb0b7e502b17b2fb/trl/trainer/ppo_trainer.py#L1133).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/huggingface/trl/blob/34ebc4ccaf376c862a081ff4bb0b7e502b17b2fb/trl/trainer/ppo_trainer.py#L1133)">trl/trl/trainer/ppo_trainer.py at 34ebc4ccaf376c862a081ff4bb0b7e502b17b2fb · huggingface/trl</a>: Train transformer language models with reinforcement learning. - huggingface/trl</li><li><a href="https://github.com/pytorch/torchtune/pull/645#issuecomment-2041861215)?">DPO by yechenzhi · Pull Request #645 · pytorch/torchtune</a>: Context integrating DPO into Torchtune, more details see here Changelog  ...  Test plan  ....
</li>
</ul>

</div>
  

---



### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1249872448506171392)** (1 messages): 

- **Meet Chip Huyen at Mosaic Event**: Chip invites everyone to say hi at the [Mosaic event at Databricks Summit](https://mosaicx.events/events/june-10-2024-san-francisco-ca?events=main) tonight. This is a chance for in-person networking and connecting with peers in the AI and ML community.

**Link mentioned**: <a href="https://mosaicx.events/events/june-10-2024-san-francisco-ca?events=main">Events | June 10, 2024 San Francisco, CA</a>: no description found

  

---



### **Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/)** (1 messages): 

jartine: is that a grammar thing?
  

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
