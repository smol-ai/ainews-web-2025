---
id: ad06ca0b-4375-46a3-a673-3d829bbb1f66
title: LLMs-as-Juries
date: '2024-05-01T01:41:25.208668Z'
original_slug: ainews-to-be-named-4408
description: >-
  **OpenAI** has rolled out the **memory feature** to all ChatGPT Plus users and
  partnered with the **Financial Times** to license content for AI training.
  Discussions highlight concerns about OpenAI's profitability due to paid
  training data and possible reductions in **GPT-4** usage limits. Users report
  issues with ChatGPT's data cleansing after the memory update. Community posts
  cover building AI voice assistants with OpenAI's API, AI-powered side projects
  using **GPT-4** and **GPT-3.5**, and interface agents powered by LLMs. In the
  **Stable Diffusion** space, users seek realistic **SDXL** models comparable to
  PonyXL, praise the **Hi-diffusion** extension for ComfyUI, and note the
  release of Virtuoso Nodes v1.1 adding Photoshop-like features to ComfyUI.
companies:
  - openai
  - financial-times
  - cohere
models:
  - gpt-4
  - gpt-3.5
  - sdxl
topics:
  - memory
  - training-data
  - api
  - interface-agents
  - image-generation
  - model-optimization
  - prompt-engineering
people: []
---


<!-- buttondown-editor-mode: plaintext -->> AI News for 4/29/2024-4/30/2024. We checked 7 subreddits and [**373** Twitters](https://twitter.com/i/lists/1585430245762441216) and **28** Discords (**417** channels, and **4855** messages) for you. Estimated reading time saved (at 200wpm): **579 minutes**.

In the agent literature it is common to find that multiple agents  outperform single agents (if you conveniently ignore inference cost). [Cohere has now found the same for LLMs-as-Judges](https://twitter.com/cohere/status/1785284142789242932?utm_source=ainews&utm_medium=email):

 ![image.png](https://assets.buttondown.email/images/ecea573b-f0e8-4e44-968d-82e8f2f4540e.png?w=960&fit=max) 

---

**Table of Contents**

[TOC] 


---

# AI Reddit Recap

> Across r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity. Comment crawling works now but has lots to improve!

Here is the updated summary with the requested formatting and de-ranking of AGI posts:

**OpenAI News**

- **Memory feature now available to all ChatGPT Plus users**: OpenAI [announced on Twitter](https://twitter.com/OpenAI/status/1784992796669096181) that the memory feature is now rolled out to all ChatGPT Plus subscribers.
- **OpenAI partners with Financial Times for AI in news**: OpenAI has [signed a deal to license content](https://www.reuters.com/technology/financial-times-openai-sign-content-licensing-partnership-2024-04-29/) from the Financial Times to train its AI models. An [image was shared](https://i.redd.it/s09mjga1jgxc1.jpeg) announcing the partnership to develop AI experiences for news. 
- **Concerns over OpenAI's profitability with paid training data**: In /r/OpenAI, a [post questioned](https://www.reddit.com/r/OpenAI/comments/1cfxd42/how_is_openai_going_to_be_profitable_if_they_have/) OpenAI's profitability as they start paying to license training data, speculating local open source models may undercut their business.
- **Possible reduction in GPT-4 usage limits**: A [user in /r/OpenAI noticed](https://www.reddit.com/r/OpenAI/comments/1cfxzvl/has_openai_reduced_the_number_of_questions/) GPT-4's usage has been reduced from 40 messages per 3 hours to around 20 questions per hour. 
- **Issues with ChatGPT after memory update**: In /r/OpenAI, a user [found ChatGPT struggled](https://www.reddit.com/r/OpenAI/comments/1cg8zsd/chatgpt_laziness_data_cleansing_and_analysis_is/) with data cleansing and analysis tasks after the memory update, producing errors and incomplete outputs.

**OpenAI API Projects and Discussions**

- **Tutorial on building an AI voice assistant with OpenAI**: A [blog post was shared](https://www.reddit.com/r/OpenAI/comments/1cgh184/how_i_build_an_ai_voice_assistant_with_openai/) in /r/OpenAI on building an AI voice assistant using OpenAI's API along with web speech APIs. 
- **AI-powered side projects discussion**: In /r/OpenAI, a [post asked others to share](https://www.reddit.com/r/OpenAI/comments/1cg5mm7/whats_your_ai_backed_side_project/) their AI-powered side projects. The poster made a requirements analysis tool with GPT-4 and an interactive German tutor with GPT-3.5.
- **Interface agents powered by LLMs**: A /r/OpenAI [post discussed "interface agents"](https://www.reddit.com/r/OpenAI/comments/1cg3f2z/p_interface_agents_building_llmenabled_agents/) - AI that can interact with and control user interfaces like browsers and apps. It covered key components, tools, challenges and use cases.
- **Difficulty resizing elements in GPT-4 generated images**: In /r/OpenAI, a [user asked for advice](https://www.reddit.com/r/OpenAI/comments/1cga0zy/best_way_to_tell_gpt4_to_shrink_something_in_a/) on instructing GPT-4 to shrink an element in a generated image, as the model struggles to consistently resize things.

**Stable Diffusion Models and Extensions**

- **Seeking realistic SDXL models comparable to PonyXL**: In /r/StableDiffusion, a [user asked about](https://www.reddit.com/r/StableDiffusion/comments/1cfv7ga/any_realistic_sdxl_model_as_good_as_ponyxl/) realistic SDXL models on par with PonyXL's quality and prompt alignment for photographic styles.
- **Hi-diffusion extension for ComfyUI**: A /r/StableDiffusion user [found Hi-diffusion works well](https://www.reddit.com/r/StableDiffusion/comments/1cg2394/hidiffusion_is_very_impressive_now_the_comfyui/) for generating detailed 2K images in ComfyUI with SD1.5 models, outperforming Khoya deep shrink. An extension is available but needs improvements.
- **Virtuoso Nodes v1.1 adds Photoshop features to ComfyUI**: [Version 1.1 of Virtuoso Nodes](https://www.reddit.com/r/StableDiffusion/comments/1cgexi9/virtuoso_nodes_release_v11_with_new_photoshop/) for ComfyUI was released, adding 8 new nodes that replicate key Photoshop functions like blend modes, selective color, color balance, etc.
- **Styles to simplify Pony XL prompts in Fooocus**: A /r/StableDiffusion user [created styles for Fooocus](https://www.reddit.com/r/StableDiffusion/comments/1cglyq4/styles_for_fooocus_to_shorten_your_pony_xl/) to handle the quality tags in Pony XL prompts, allowing cleaner and shorter prompts focused on content.
- **Anime-style shading LoRA released**: An [anime-style shading LoRA](https://huggingface.co/2vXpSwA7/iroiro-lora/blob/main/test3/sdxl-shadow_01.safetensors) was announced, recommended for use with Anystyle and other ControlNets. A Hugging Face link to the LoRA file was provided.

**Stable Diffusion Help and Discussion**

- **Avoiding explicit content in generated images**: In /r/StableDiffusion, a user getting [phallic elements in 80% of their generated images](https://www.reddit.com/r/StableDiffusion/comments/1cgjrds/80_of_my_generated_pics_have_dicks_coming_out_of/) asked for negative prompt advice to generate "regular porn" instead.
- **Creating short video clips with AI images and animated text**: A /r/StableDiffusion [post asked about APIs](https://www.reddit.com/r/StableDiffusion/comments/1cfwxct/how_to_create_short_videos_by_using_ai_images_and/) to generate AI images with animated text overlays to create short video clips.
- **Newer Nvidia GPUs may be slower for AI despite gaming gains**: A [warning was posted](https://www.reddit.com/r/StableDiffusion/comments/1cg0gz6/be_careful_when_buying_new_nvidia_card_or_laptop/) that newer Nvidia GPUs like the 4070 laptop version use narrower memory buses than older models, making them slower for AI workloads.
- **Proposal for community image tagging project**: A /r/StableDiffusion [post suggested a community effort](https://www.reddit.com/r/StableDiffusion/comments/1cgbivm/community_effort_for_best_image_tagging/) to comprehensively tag images to create a dataset of consistently captioned images for training better models.
- **Using VAEs for image compression**: Experiments [shared in /r/StableDiffusion](https://www.reddit.com/r/StableDiffusion/comments/1cgdyjc/vae_as_image_compression/) show using VAE latents for image compression is competitive with JPEG in some cases. Saving generated images as latents is lossless and much smaller than PNGs.
- **Generating a full body from a headshot**: In /r/StableDiffusion, a [user asked if it's possible](https://www.reddit.com/r/StableDiffusion/comments/1cg3a4z/help_me_with_this_will_pay/) to generate a full body from a headshot image without altering the face much using SD Forge.
- **Textual inversion of Audrey Hepburn**: A /r/StableDiffusion user [made a textual inversion](https://www.reddit.com/r/StableDiffusion/comments/1cft1gp/give_you_a_slightly_different_audrey_hepburn/) of Audrey Hepburn that produces similar but varied faces, sharing example images and a Civitai link.

---

# AI Twitter Recap

> all recaps done by Claude 3 Opus, best of 4 runs. We are working on clustering and flow engineering with Haiku.

**LLMs and AI Models**

- **Llama 3 Performance**: [@abacaj](https://twitter.com/abacaj/status/1785147493728039111) noted that llama-3 models with zero-training can get **32k context with exceptional quality**, surpassing significantly larger models. [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1784889182558539917) mentioned Llama 3 captures extremely nuanced data relationships, utilizing even the minutest decimals in BF16 precision, making it **more sensitive to quantization degradation** compared to Llama 2.
- **Llama 3 Benchmarks**: [@abacaj](https://twitter.com/abacaj/status/1785295341736043007) reported llama-3 70B takes **3rd place on a benchmark, replacing Haiku**. [@abacaj](https://twitter.com/abacaj/status/1785153286976237765) shared a completion from the model on a **code snippet benchmark** that requires the model to find a function based on a description.
- **Llama 3 Variants**: [@mervenoyann](https://twitter.com/mervenoyann/status/1785320444918211022) noted **new LLaVA-like models based on LLaMA 3 & Phi-3** that pass the baklava benchmark. [@AIatMeta](https://twitter.com/AIatMeta/status/1785042326416658580) mentioned Meditron, an LLM suite for low-resource medical settings built by @ICepfl & @YaleMed researchers, which **outperforms most open models in its parameter class** on benchmarks like MedQA & MedMCQA using Llama 3.
- **GPT-2 Chatbot**: There was speculation about the identity of the gpt2-chatbot model, with [@sama](https://twitter.com/sama/status/1785107943664566556) noting he has a soft spot for gpt2. Some theories suggested it could be a preview of GPT-4.5/5 or a derivative model, but most agreed it was **unlikely to be the latest OAI model**. 
- **Phi-3 and Other Models**: [@danielhanchen](https://twitter.com/danielhanchen/status/1785040680106234225) released a **Phi-3 notebook that finetunes 2x faster and uses 50% less VRAM** than HF+FA2. [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1785220060803453160) shared a paper suggesting transformers learn in-context by **performing gradient descent on a loss function constructed from the in-context data** within their forward pass.

**Prompt Engineering and Evaluation**

- **Prompt Engineering Techniques**: [@cwolferesearch](https://twitter.com/cwolferesearch/status/1784992130777137362) categorized recent prompt engineering research into **reasoning, tool usage, context window, and better writing**. Techniques include zero-shot CoT prompting, selecting exemplars based on complexity, refining rationales, decomposing tasks, using APIs, optimizing context windows, and iterative prompting.
- **LLMs as Juries**: [@cohere](https://twitter.com/cohere/status/1785284142789242932) released a paper exploring **replacing a single LLM judge with multiple LLM juries** for evaluation. The "PoLL" method with a diverse set of LLMs **outperformed single judges across datasets while being 7-8x cheaper** than GPT-4.
- **Evaluating LLMs**: [@_lewtun](https://twitter.com/_lewtun/status/1785246966626029596) asked about research on which prompts produce an LLM-judge most correlated with human preferences for pairwise rankings, beyond the work by @lmsysorg. [@_philschmid](https://twitter.com/_philschmid/status/1785273493375922221) summarized the **PoLL (Panel of LLM) method** proposed by @cohere for LLM evaluation as an alternative to a single large model judge.

**Applications and Use Cases**

- **Financial Calculations**: [@llama_index](https://twitter.com/llama_index/status/1785325832317415641) shared a full-stack tutorial for building a financial assistant that can **calculate percentage evolution, CAGR, and P/E ratios over unstructured financial reports** using LlamaParse, RAG, Opus, and math formulas in @llama_index.
- **SQL Query Generation**: [@virattt](https://twitter.com/virattt/status/1785059112478257413) used @cohere cmd r+ to **extract ticker and year metadata from financial queries** in ~1s, then used the metadata to filter a vector db, fed results to GPT-4, and answered user query with ~3s total latency.
- **Multi-Agent RAG**: [@LangChainAI](https://twitter.com/LangChainAI/status/1785066609847291986) announced a YouTube workshop on exploring "multi-agent" applications that **combine independent agents to solve complex problems** using planning, reflection, tool use, and their LangGraph library.
- **Robotics and Embodied AI**: [@DrJimFan](https://twitter.com/DrJimFan/status/1785292766387302897) advocated for **robotics as the next frontier after LLMs**, sharing MIT AI Lab's 1971 proposal emphasizing robotics and reflecting on the current state. [@_akhaliq](https://twitter.com/_akhaliq/status/1785139220534730771) shared a paper on Ag2Manip, which **improves imitation learning for manipulation tasks** using agent-agnostic visual and action representations.

**Frameworks, Tools and Platforms**

- **LangChain Tutorials**: [@LangChainAI](https://twitter.com/LangChainAI/status/1784970647875330251) shared a **4-hour course on understanding how LangChain works** with various technologies to build 6 projects. [@llama_index](https://twitter.com/llama_index/status/1784962053641478454) provided a **reference architecture for advanced RAG** using LlamaParse, AWS Bedrock, and @llama_index.
- **Diffusers Library**: [@RisingSayak](https://twitter.com/RisingSayak/status/1785162074844197174) explained how the Diffusers library **supports custom pipelines and components**, allowing flexibility in building diffusion models while maintaining the benefits of the `DiffusionPipeline` class.
- **Amazon Bedrock**: [@cohere](https://twitter.com/cohere/status/1785015769971220720) announced their **Command R model series is now available on Amazon Bedrock** for enterprise workloads. [@llama_index](https://twitter.com/llama_index/status/1785105949818237227) showed how to use LlamaParse for advanced parsing in the AWS/Bedrock ecosystem and **build RAG with the Bedrock Knowledge Base**.
- **DeepSpeed Support**: [@StasBekman](https://twitter.com/StasBekman/status/1785091895733154116) noted a PR merged into `main@accelerate` that makes FSDP **converge at the same speed as DeepSpeed when loading fp16 models**, by automatically upcasting trainable params to fp32.

**Memes, Humor and Other**

- **ASCII Art**: Several tweets poked fun at the ASCII art capabilities of LLMs, with [@ylecun](https://twitter.com/ylecun/status/1785109502565531699) noting how **AI hype has become indistinguishable from satire**. [@teortaxesTex](https://twitter.com/teortaxesTex/status/1785325820166185399) shared a prompt to draw a Katamari Damacy level map using emojis that strains "GPT2"'s instruction following.
- **Anthropic Slack**: [@alexalbert__](https://twitter.com/alexalbert__/status/1785369914204938326) shared his **10 favorite things from Anthropic's internal Slack channel** where employees post cool Claude interactions and memes since its launch.
- **Rabbit Disappointment**: Several users expressed disappointment with the Rabbit AI device, noting its **limited functionality compared to expectations**. [@agihippo](https://twitter.com/agihippo/status/1785359480294936882) questioned what the Rabbit r1 can do that a phone can't.

---

# AI Discord Recap

> A summary of Summaries of Summaries

**1) Fine-Tuning and Optimizing Large Language Models**

- **Challenges in Fine-Tuning LLaMA-3**: Engineers faced issues like the model **not generating EOS tokens**, and **embedding layer compatibility across bit formats**. However, one member achieved success by utilizing **[LLaMA-3 specific prompt strategies](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1553)** for fine-tuning.

- **LLaMA-3 Sensitive to Quantization**: Discussions highlighted that **[LLaMA-3 experiences more degradation from quantization](https://x.com/rohanpaul_ai/status/1784972618472317180)** compared to LLaMA-2, likely due to capturing nuanced relationships from training on 15T tokens.

- **Perplexity Fine-Tuning Challenges**: Fine-tuning **LLaMA-3 for perplexity** may not surpass the base model's performance, with the tokenizer suspected as a potential cause.

**2) Extending Context Lengths and Capabilities**

- **Llama-3 Hits New Context Length Highs**: The release of **[Llama-3 8B Gradient Instruct 1048k](https://huggingface.co/gradientai/Llama-3-8B-Instruct-Gradient-1048k)** extends the context length from 8k to over 1048k tokens, showcasing state-of-the-art long context handling.

- **Llama 3 Gains Vision with SigLIP**: A breakthrough integrates **[vision capabilities for Llama 3](https://huggingface.co/qresearch/llama-3-vision-alpha-hf)** using SigLIP, enabling direct use within Transformers despite quantization limitations.

- **Extending Context to 256k with PoSE**: The context length of **Llama 3 8B** has been expanded from 8k to **[256k tokens using PoSE](https://huggingface.co/winglian/llama-3-8b-256k-PoSE)**, though inferencing challenges remain for 'needle in haystack' scenarios.

**3) Benchmarking and Evaluating LLMs**

- **Llama 3 Outperforms GPT-4 in German NLG**: On the **[ScanEval German NLG benchmark](https://scandeval.com/german-nlg/)**, **Llama 3** surpassed the performance of **GPT-4**, indicating its strong language generation capabilities.

- **Mysterious GPT2-Chatbot Sparks Speculation**: A **[GPT2-chatbot](https://chat.lmsys.org/)** with gpt4-level capabilities surfaced, leading to debates on whether it could be an early glimpse of **GPT-4.5** or a finetuned version of the original GPT-2.

- **Questioning Leaderboard Utility for Code Generation**: A **[blog post](https://www.aisnakeoil.com/p/ai-leaderboards-are-no-longer-useful)** challenges the effectiveness of AI leaderboards for code generation, citing the high operational cost of top performers like LLM debugger despite ranking highly.

**4) Revolutionizing Gaming with LLM-Powered NPCs**

- **LLM-Powered NPCs and Inference Stack**: The release of **[LLM-powered NPC models](https://github.com/GigaxGames/gigax)** aims to enhance action spaces and simplify API calls, including a single LLM call feature and open-weights on Hugging Face.

- **Overcoming LLM Challenges for Gameplay**: Developers faced issues like **NPCs breaking the fourth wall**, missing details in large prompts, and optimizing for runtime speeds, suggesting solutions like **output compression**, **minimizing model calls**, and leveraging **smaller models**.

- **Insights into Fine-Tuning LLMs for NPCs**: Developers plan to share their **struggles and triumphs in fine-tuning LLMs for dynamic NPC behavior** through an upcoming blog post, pointing towards new strategies for gaming applications.


**5) Misc**

- **CUDA Optimization Techniques**: CUDA developers discussed various optimization strategies, including using `Packed128` custom structs for memory access patterns, replacing integer division with bit shifts ([Compiler Explorer link](https://godbolt.org/z/9K9Gf1v6P)), and comparing performance of **CUTLASS vs CuBLAS** for matrix multiplications. The **Effort Engine** algorithm was introduced, enabling adjustable computational effort during LLM inference to achieve speeds comparable to standard matrix multiplications on Apple Silicon ([kolinko.github.io/effort](https://kolinko.github.io/effort), [GitHub](https://github.com/kolinko/effort)).

- **LLaMA-3 Context Length Extension and Fine-Tuning**: The **LLaMA-3 8B** model's context length was extended to over 1M tokens using **PoSE** ([huggingface.co/winglian/llama-3-8b-256k-PoSE](https://huggingface.co/winglian/llama-3-8b-256k-PoSE)), sparking discussions on its retrieval performance and compute requirements. Fine-tuning LLaMA-3 presented challenges like **quantization degradation**, **EOS token generation**, and **embedding layer compatibility** across bit formats. A potential breakthrough was shared in a [GitHub pull request](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1553) demonstrating successful fine-tuning with model-specific prompt strategies.

- **Civitai Monetization Backlash**: Stable Diffusion community members expressed discontent with **Civitai's monetization strategies**, particularly the **Buzz donation system**, which was labeled a "rip-off" by some like Tower13Studios ([The Angola Effect](https://youtu.be/nLT32AR5c68?si=bV9wXlRzb_oLutW9)). Discussions also highlighted the potential profitability of **NSFW AI-generated art commissions** compared to the saturated SFW market.

- **Perplexity AI Performance Issues**: Users reported significant slowdowns and poor performance across various Perplexity AI models during Japan's Golden Week, with specific issues in **Japanese searches** resulting in meaningless outputs. Frustrations arose over **expired Pro subscription coupons** and the removal of the **7-day free trial**. Technical troubles included **email link delays** affecting login and inconsistencies in the **iOS voice feature** depending on app versions.

- **Decentralized AI Training Initiatives**: Prime Intellect proposed a decentralized training approach using **H100 GPU clusters** to enable open-source AI to compete with proprietary models ([blog post](https://www.primeintellect.ai/blog/our-approach-to-decentralized-training)). The initiative aims to address computing infrastructure limitations by leveraging globally distributed GPU resources.

---



# PART 1: High level Discord summaries




## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **Triton Troubles**: Engineers discussed limitations with **Triton blocks**, identifying an issue where blocks of 4096 elements are feasible, yet blocks of 8192 are not, hinting at discrepancies with expected **CUDA** limits.

- **CUDA Cognitions and Collaborations**: Various **CUDA** topics were mulled over, including **CUTLASS vs. CuBLAS performance**, **CUDA checkpointing**, and the replacement of integer division with bit shifts. A link to the [Compiler Explorer](https://godbolt.org/z/9K9Gf1v6P) was shared to help with experiments.
  
- **PyTorch Peculiarities Pursued**: Members examined the behavior of PyTorch's `linear` function and matrix multiplication kernel launches, with observations about double kernel launches and the false expectation of performance differences due to transposition.

- **LLM Inference Optimization with Effort Engine**: Discussion revolved around the **Effort Engine** algorithm, which enables adjustable computational effort during LLM inference, purportedly yielding speeds comparable to standard matrix multiplications on Apple Silicon at lower efforts. The implementation and details are provided on [kolinko.github.io/effort](https://kolinko.github.io/effort) and [GitHub](https://github.com/kolinko/effort).

- **InstaDeep's Machine Learning Manhunt**: **InstaDeep** is on the hunt for **Machine Learning Engineers** with expertise in **high-performance ML engineering, custom CUDA kernels, and distributed training**. Candidates can scout for opportunities at [InstaDeep Careers](https://www.instadeep.com/job-offer/92900fa3-5501-4506-a63f-cebee958fc6f/). 

- **Llama-3 Levitates to Longer Contexts**: The release of [Llama-3 8B Gradient Instruct 1048k](https://huggingface.co/gradientai/Llama-3-8B-Instruct-Gradient-1048k) set a new benchmark for context length capabilities in LLMs.

- **ROCm Rallies for Flash Attention 2**: Conversations in the **ROCM** channel centered on adapting NVIDIA's Flash Attention 2 for ROCm, with a focus on compatibility with **ROCM 6.x** versions and a link to the relevant repository [ROCm/flash-attention on GitHub](https://github.com/ROCm/flash-attention).

- **CUDA Conclave Converges on “Packed128” Innovations**: The **llmdotc** channel was a hotspot with discussions focused on optimizing `Packed128` data structures and **BF16 mixed-precision strategies**, while also touching on the nuanced use of **NVTX** contexts and the utility of different benchmarking toolsets like **Modal**.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Fusing Checkpoints to Avoid Overfitting**: A member sought guidance on checkpoint merging to avoid overfitting and was directed to the Unsloth [finetuning checkpoint wiki](https://github.com/unslothai/unsloth/wiki#finetuning-from-your-last-checkpoint). Techniques such as *warmup steps* and *resuming from checkpoints* were recommended for nuanced training regimens.

- **Quantization Quandary in WSL2**: Users reported **RuntimeError: Unsloth: Quantization failed** when converting models to F16 within WSL2. Despite attempts at rebuilding the `llama.cpp` and re-quantization, the error persisted.

- **Phi-3: A Model of Interest**: The upcoming release of **Phi-3** stirred interest, with engineers debating whether to adopt the 3.8b version or wait for the heftier 7b or 14b variants.

- **OOM Countermeasures and Performance Data Confusion**: Tips for handling Out of Memory errors on Google Colab by cache clearing were exchanged. Meanwhile, confusion surfaced over reported performance measures for quantized **LLama 2** and **LLama 3**, hinting at possible data misplacement between Bits Per Word (BPW) and Perplexity (PPL).

- **Extended Possibilities**: **Llama 3 8B** reached new potential with a Context length increase to 256k tokens, achieved with **[PoSE](https://huggingface.co/papers/2309.10400)**, showcased at [winglian/llama-3-8b-256k-PoSE](https://huggingface.co/winglian/llama-3-8b-256k-PoSE). Community applause went to Winglian, though some voiced skepticism about non-official context-extended model behavior.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Groq's Gift to Discord Bots**: A user shared a [YouTube video](https://youtu.be/ySwJT3Z1MFI?si=qFfek8gTGXVJWoxB) highlighting the *free* Groq API enabling access to the LLAMA-3 model's impressive 300 tokens per second speed, optimally suited for small server Discord bots due to its no-cost setup.
- **Spec Smackdown**: Users recommended posting system specs in specific channels when troubleshooting **LM Studio on Ubuntu GPUs**, debated the compatibility of GPUs with **inference tasks**, and discussed the potentially incorrect VRAM capacity display in LM Studio causing concerns with **GPU offloading efficiency**.
- **Model Mania**: The community buzzed about alternative methods for downloading the GGUF model from sources other than Huggingface, the time and resource demands of creating *iQuants* and *imatrices*, and shared reward offers for optimizing the **Goliath 120B Longlora** model to create its *iQuant* version.
- **Model Mayhem on Modest Machines**: Users grappled with issues like the Phi-3 model's **leaking prompts**, *local training* queries for Hugging Face-based models, and the unexpected noises from hard drives during token generation by the Llama3m. Some determined that more dated hardware could just about manage a **7b Q4 model** but nothing heftier.
- **ROCm Ruminations**: Enthusiasts dissected ROCm versions, mulling over the benefits of **beta 0.2.20** for AMD functionality, addressed confusion about compatibility—especially the RX 6600's support with the current **HIP SDK**—and discussed discrepancies in ROCm's functionality on different operating systems like **Ubuntu versus Windows**.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

**Buzz Off, Civitai**: AI creators in the guild are upset with Civitai's monetization strategies, particularly the Buzz donation system, which was labeled a **"rip-off"** by some members, such as Tower13Studios. The discontent revolves around value not being fairly returned to creators ([The Angola Effect](https://youtu.be/nLT32AR5c68?si=bV9wXlRzb_oLutW9)).

**Finding The AI Art Goldmine**: A vibrant discussion unfolded on the economics of AI-generated art, with consensus pointing towards NSFW commissions, including furry and vtuber content, as a more profitable avenue compared to the more crowded SFW market.

**Race for Real-Time Rendering**: Members actively shared Python scripting techniques for accelerating Stable Diffusion (SDXL) models, eyeing uses in dynamic realms like Discord bots, aiming to enhance image generation speed for real-time applications.

**Anticipation Builds for Collider**: The community is keenly awaiting Stable Diffusion's next iteration, dubbed "Collider," with speculation about release dates and potential advancements fueling eager anticipation among users.

**Tech Troubleshooting Talk**: Guild members exchanged insights and solutions on a spectrum of technical challenges, from creating LoRAs and IPAdapters to running AI models on low-spec hardware, demonstrating a collective effort to push the boundaries of model implementation and optimization.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Japanese Golden Week Glitches**: During Japan's Golden Week, users observed a noticeable performance drop in tools like **Opus, Sonar Large 32K,** and **GPT-4 Turbo**, with specific issues in Japanese searches, resulting in outputs that users deemed *meaningless garbage*. To address the problem, vigilant monitoring and optimization of these models was suggested.

- **Frustration over Pro Subscription and Trial Perils**: **Pro subscription** users reported expired coupons on the due date, with offers linked to the **Nothing Phone 2(a)** aborted prematurely due to fraudulent activities. Moreover, the 7-day free trial's removal from the site prompted disappointment, emphasizing its value as a user conversion tool.

- **Tech Turbulence with Perplexity AI**: The community grappled with **email link delays**, causing login difficulties, particularly for non-Gmail services. Additionally, variations in the **iOS voice feature** were found to be dependent on the **app version** being used, reflecting inconsistencies in user experience.

- **API Avenues Explored**: Engineers queried the **pplx-api** channel regarding **source URL** access through the API, following its mention in roadmap documentation, and debated whether using **Claude 3** would entail adherence to **Anthropic's political usage** restrictions under Perplexity's terms.

- **Miscellaneous Inquiries and Insights Surface**: A post in the **#[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1234586871569449121)** channel spotlighted Lenny's Newsletter on product growth and building concepts, while queries about WhatsApp's autoreply feature and Vimeo's API were thrown in. These discussions, particularly on the API, highlight engineers' focus on integrating and utilizing various functionalities in their systems/processes.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

**Bold Decentralization Move**: Prime Intellect's initiative for decentralized AI training, leveraging *H100 GPU clusters*, promises to push the boundaries by globalizing distributed training. The open-source approach may address current computing infrastructure bottlenecks as discussed in their [decentralized training blog](https://www.primeintellect.ai/blog/our-approach-to-decentralized-training).

**Retrieval Revolution with LLama-3**: The extension of **LLama-3 8B's** context length to over 1040K tokens sparks discussions on whether its retrieval performance lives up to the hype. Skeptics remain, emphasizing the ongoing necessity of improvements and training, supported by an [ArXiv paper on IN2 training](https://arxiv.org/abs/2404.16811).

**PDF Challenges Tackled**: To address PDF parsing challenges within AI models, particularly for tables, the community discussed workarounds and tools like [OpenAI's file search](https://platform.openai.com/docs/assistants/tools/file-search) for better multimodal functionality handling roughly 10k files.

**World Sims Showcase AI's Role-Playing Prowess**: Engagements with AI-driven world simulations highlight the capacities of **llama 3 70b** and **Claude 3**, from historical figures to business and singing career simulators. OpenAI's chat on [HuggingChat](https://hf.co/chat/assistant/65ffac7250c6fddecfd20bc8) and links to niche simulations like [Snow Singer Simulator](https://hf.co/chat/assistant/6626e4869232378718adc5f2) reflect the diversity and depth achievable.

**Leveraging Datasets for Multilingual Dense Retrieval**: A noted [Wikipedia RAG dataset](https://huggingface.co/collections/nthakur/swim-ir-dataset-662ddaecfc20896bf14dd9b7) on HuggingFace earmarks the rise of fostering AI's language retrieval capabilities. The included Halal and Kosher data points toward a trend of creating diverse and inclusive AI resources.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo's Memory Safety and Concurrency Debated**: Despite buzz around **Mojo's** potential, it was clarified that features like **Golang-like concurrency** and **Rust-like memory safety** are not currently implemented due to **borrow checking being disabled**. However, possibilities regarding the use of actor model concurrency are being explored which may enhance Mojo’s runtime efficiency. 

- **Installation Tactics for Mojo on Varied Systems**: Users face challenges installing **Mojo** with **Python 3.12.3** particularly on **Mac M1**, for which using a **Conda environment** is recommended. Also, while native **Windows support** is pending, **WSL on Windows** is a current workaround, with cross-compilation capabilities hinted through **LLVM involvement**.

- **Community Contributions to Mojo Ecosystem**: Several community-driven projects are enhancing the Mojo ecosystem, from a Mojo-based forum on GitHub to a **20% performance optimized** atof-simd project for long strings. Enthusiasm for collaboration and knowledge-sharing is evident as members share projects and call for joint efforts to tackle challenges such as the 1brc.

- **Nightly Compilations Trigger Discussions on SIMD and Source Location**: A new **nightly** release of the **Mojo compiler** spurred conversation about the conversion of **SIMD to EqualityComparable** and the need for explicit `reduce_and` or `reduce_or` in place of implicit conversion to `Bool`. The move of `__source_location()` to `__call_location()` incited exchanges on proper usage within the language.

- **Performance and Benchmarking Take the Spotlight**: From optimizing SIMD-based error correction code to sharing substantial speed gains in the 1brc project, performance topics spurred discussions on **LLVM/MLIR optimizations**. There were calls to form a "team-mojo" for communal challenge tackling, underscoring a shared interest in progressing Mojo’s benchmarking endeavors against other languages.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

**Snowflake's MoE Model Breaks Through**: Snowflake introduces a [monumental 408B parameter Dense + Hybrid MoE model](https://x.com/reach_vb/status/1783129119435210836) with a 4K context window, entirely under Apache 2.0 license, sparking excitement for its performance on sophisticated tasks.

**Gradio Share Server on the Fritz**: Gradio acknowledges [issues with their Share Server](https://status.gradio.app/), impacting Colab integrations, which is under active resolution with updates available on their status page.

**CVPR 2023 Sparks Competitive Spirit**: CVPR 2023 [announced competetive events](https://huggingface.co/spaces/BVRA/SnakeCLEF2024) like SnakeCLEF, FungiCLEF, and PlantCLEF, boasting over $120k in rewards and happening June 17-21, 2024.

**MIT Deep Learning Course Goes Live**: MIT updates its Introduction to Deep Learning course for 2024, with comprehensive [lecture videos on YouTube](https://www.youtube.com/watch?v=ErnWZxJovaM&list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI&index=2).

**NLP Woes in Chatbot Land**: Within the NLP community, effort mounts to finetune a chatbot using the Rasa framework, despite struggles with intent recognition and categorization, and plans to augment performance with a custom NER model and company-specific intents.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Alex Atallah Signposts Collaboration with Syrax**: Alex Atallah has initiated experiments with **Syrax** and extended support by proposing a group chat for collaborative efforts, marking the start of a partnership acknowledged with enthusiasm by Mart02.

- **Frontend for the Rest of Us**: The community explored solutions for deploying multi-user frontends on shared hosting without advanced technical requirements. **LibreChat** was suggested as a viable platform, with Vercel's free tier hosting mentioned as a means to address hosting and cost obstacles.

- **LLMs Throwdown**: A robust debate unfolded over several large language models including *Llama-3 8B*, *Dolphin 2.9*, and *Mixtral-8x22B*, touching on aspects like context window size and censorship concerns related to conversational styles and datasets.

- **Training Unhinged AIs**: An intriguing experiment involved training a model with a toxic dataset to foster a more "unhinged" persona. Discussions dug into model limitations with long contexts, with an agreement that although models like *Llama 3 8B* handle extensive contexts, performance dips were likely past a threshold.

- **Cost-Effective Experimentation on OpenRouter**: Conversations centered on finding efficient yet affordable models on **OpenRouter**. Noteworthy was the mix of surprise and approval for the human-like output of models like *GPT-3.5* that deliver a solid blend of affordability and performance.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

**AWS Architecture Goes Academic**: **LlamaIndex** revealed an advanced AWS-based architecture for building sophisticated RAG systems, aimed at parsing and reasoning. Details are accessible in their [code repository](https://t.co/sfQOvhHHg5).

**Documentation Bot Triumphs in Hackathon**: Hackathon victors, **Team CLAB**, developed an impressive documentation bot leveraging **LlamaIndex** and **Nomic embeddings**; check out the hackathon wrap-up in this [blog post](https://t.co/2UMqrHwO56).

**Financial Assistants Get a Boost**: Constructing financial assistants that interpret unstructured data and perform complex computations has been greatly improved. The methodology is thoroughly explored in a [recent post](https://t.co/6cTNxUBJcr).

**Turbocharging RAG with Semantic Caching**: Collaboration with @Redisinc demonstrated significant performance gains for RAG applications using **semantic caching** to speed up queries. The collaboration details can be found [here](https://t.co/oGxFrZLMRn).

**GPT-1: The Trailblazer Remembered**: A reflective glance at GPT-1 and its contributions to LLM development was shared, discussing features like positional embeddings which paved the way for modern models like Mistral-7B. The nostalgia-laden [blog post](https://amgadhasan.substack.com/p/revisiting-gpt-1-the-spark-that-ignited-llms) revisits GPT-1's architecture and impact.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

**Plug Into New Community Projects**: Members are seeking opportunities to contribute to community AI projects that provide computational resources, addressing the issue for those lacking personal GPU infrastructure.

**Unlock the Mysteries of AI Memory**: Intricacies of memory processes in AI were covered with a particular focus on "clear-ing", orthogonal keys, and the delta rule in compressive memory. There’s an interest in discussing whether infini-attention has been overhyped, despite its theoretical promise.

**Comparing Apples to Supercomputers**: There's an active debate regarding performance discrepancies between models like *mixtral 8x22B* and *llama 3 70B*, where *llama's* reduced number of layers, despite having more parameters, may be impacting its speed and batching efficiency.

**LLMs: Peering Inside the Black Box**: The community is contemplating the “black box” nature of Large Language Models, discussing emergent abilities and data leakage. A connection was made between emergent abilities and pretraining loss, challenging the focus on compute as a performance indicator.

**Bit Depth Bewilderment**: A user reported issues when encoding with **8bit** on models like **llama3-70b** and **llamma3-8b**, experiencing significant degradation in output quality, suggesting a cross-model encoding challenge that needs addressing.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **GDPR Complaint Challenges AI Birthdays**: An EU privacy advocate has filed a [GDPR complaint](https://www.politico.eu/article/chatgpts-hallucinations-get-eu-privacy-complaint/) after an AI model incorrectly estimated his birthday, triggering discussions on the potential implications for AI operations in Europe.
- **Mysterious GPT-5 Speculations**: Amidst rumors of a new GPT-5 model release, the community debates inconsistent test outcomes and the absence of official communication or leaderboard recognitions, questioning the framework's evasiveness in generating hallucinations.
- **Llama3 70B's Slow Performance Spotlight**: AI engineers are troubleshooting the [Llama3 70B](https://rentry.co/GPT2) model's unexpectedly sluggish token generation rate of 13 tokens per second on a dual 3090 rig, delving into possible hardware and configuration enhancements.
- **Exllama Library Outraces Rivals**: Users endorse **Exllama** for its fast performance on language model tasks and suggest utilizing the [TabbyAPI](https://dct.openempathic.ai/) repository for simpler integrations, naming it a superior choice compared to other libraries.
- **Research Breakthrough with OpenCLIP**: The successful application of **OpenCLIP** to cardiac ultrasound analysis has been published, highlighting the rigorous revision process and a move towards novel, non-zero-shot techniques, with the study available [here](https://doi.org/10.1038/s41591-024-02959-y); meanwhile *r/StableDiffusion* is back online and a relevant CLIP training repository is discussed in the context of Reddit's recent API changes, found at [this Reddit discussion](https://www.reddit.com/r/StableDiffusion/comments/1cgyjvt/github_zer0intclipfinetune_or_sdxl_training_the/).



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

**Memory Lane with Upscaled ChatGPT Plus**: ChatGPT Plus now allows users to command the AI to remember specific contexts, which can be toggled on and off in settings; the rollout has not reached Europe or Korea yet. Plus, both Free and Plus users gain enhanced data control, including a 'Temporary Chat' option that discards conversations immediately after they end.

**AI Ghosh-darn Curiosity and Camera Tricks**: Discussions swung from defining AI curiosity and sentience with maze challenges to the merits of DragGAN altering photos with new angles. Meanwhile, the Llama-3 8B model emerged, flaunting its long-context skills and is accessible at [Hugging Face](https://huggingface.co/gradientai/Llama-3-8B-Instruct-Gradient-1048k), but the community still wrestled with the accessibility of advanced AI technologies and the dream of inter-model collaboration.

**GPT-4: Bigger and Maybe Slower?**: The community dove into the attributes of GPT-4, noting its significantly larger size than the 3.5 version and raising concerns about whether its scale may affect processing speed. Meanwhile, the possibility of mass-deleting archived chats was also a topic of concern.

**Prompt Engineering's Competitive Edge**: Prompt engineering drew attention, with suggestions for competitions to hone skills, and 'meta prompting' via GPT Builder to refine AI output. The group agreed that positive prompting trumps listing prohibites, and wrestled with optimizing regional Spanish nuances in AI text generation.

**Cross-Channel Theme of Prompting Excellence**: Both AI discussions and API channels tackled prompt engineering, with meta-prompting techniques at the spotlight, indicating a shift toward more efficient prompting strategies that might decrease the need for competitions. Navigating the complexities of multilingual outputs also emerged as a shared challenge, emphasizing adaptation rather than prohibition.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

**LLaMA 3 Struggles with Quantization**: **LLaMA 3** is observed to have significant performance degradation from quantization processes, more so than its predecessor, which might be due to its expansive training on 15T tokens capturing very nuanced data relations. A critique within the community called a study on quantization sensitivity "worthless," suggesting that the issue may be more related to model training approaches rather than size; the critique referenced a [study on arXiv](https://arxiv.org/abs/2311.16452).

**Riding the Zero Train**: The Guild discussed **Huggingface's ZeroGPU**, a beta feature offering free access to multi-GPU resources like Nvidia A100, with some members expressing regret at missing early access. A member has [shared access](https://huggingface.co/zero-gpu-explorers) and is open to suggestions for testing on the platform.

**Finetuning Finesse**: Advised against fine-tuning `meta-llama/Meta-Llama-3-70B-Instruct`, it was suggested that members start with smaller models like 8B to sharpen their fine-tuning skills. The Guild clarified how to convert a fine-tuning dataset from OpenAI to ShareGPT format, and provided guidance with Python code for dataset transformation.

**Tutorial Spreads Its Wings**: A helpful [tutorial was shared](https://github.com/dstackai/dstack/blob/master/examples/fine-tuning/axolotl/README.md) on fine-tuning Axolotl using dstack, showing the community's knack for collaboratively improving practices. Appreciation was conveyed by members, noting the tutorial's ease of use.

**Axolotl Adaptations**: Discussing the fine-tuning of *command-r* within Axolotl and related format adaptations, a member shared an [untested pull request](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1547) relating to this topic, while also noting its prematurity for merging. In addition, there's uncertainty about the support for phi-3 format and the implementation standing of *sample packing* feature, indicating a need for further clarification or development.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Memary: An Autonomous Agent's Long-term Memory**: The [Memary](https://github.com/kingjulio8238/memary) project on GitHub has introduced a new approach for long-term memory in autonomous agents, using document similarity searches over traditional knowledge graphs.

- **The GPT-2 Chatbot Enigma**: Intense debates have emerged on a [GPT2-chatbot](https://chat.lmsys.org/) that showcases surprisingly advanced capabilities, leading to speculation that it might be a finetuned version of OpenAI's GPT-2.

- **Can Decentralized Training Compete with Big Tech?**: [Prime Intellect's blog post](https://www.primeintellect.ai/blog/our-approach-to-decentralized-training) discusses decentralized training as a plausible avenue for open-source artificial intelligence to compete with the proprietary models developed by large corporations with extensive GPU resources.

- **Redefining LLMs with Modular Context and Memory**: Discussions are emerging that suggest a paradigm shift towards designing autonomous agents with modularized shared context and memory capabilities for reasoning and planning, stepping away from the reliance on standalone large language models (LLMs).

- **Educational Resources for Aspiring AI Enthusiasts**: For those seeking to learn AI fundamentals, community members recommended resources including neural network tutorials such as the one on [YouTube](https://youtu.be/aircAruvnKk?feature=shared) and courses like *Learn Prompting*, providing a glimpse into AI engineering and prompt engineering basics.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

**OS Start-up with a Vision**: A user faced challenges attempting to **launch OS mode with a local vision model for Moondream** and received gibberish output, but the discussion did not yield a solution or direct advice.

**Integration Achievements**: An exciting integration of **OpenInterpreter** outputs into **MagicLLight** was mentioned, with anticipation for a future code release and pull request including a `stream_out` function hook and `external_input`.

**Hardware Hiccup Help**: Queries about running **OpenInterpreter on budget hardware** like a Raspberry Pi Zero were brought up alongside requests for assistance with **debugging startup issues**. Community members offered to help with troubleshooting once more details were provided.

**Push Button Programming**: An individual fixed an external push button issue on **pin 25** and shared a [code snippet](https://discord.com/channels/openinterpreter/01), also getting community confirmation that the fix was effective.

**Volume Up on Tech Talk**: There were mixed opinions on whether tech YouTubers have a grasp on AI technologies while advising on options for increasing speaker volume, including using **M5Unified** or an [external amplifier](https://www.amazon.com/dp/B01DKAI51M).



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Peek into Tinygrad's Inner Workings**: The [tinygrad GitHub repository](https://github.com/tinygrad/tinygrad/tree/master) was recommended to someone curious about **tinygrad**, an educational project for enthusiasts of PyTorch and micrograd. Another community member inquired about graph visualization, leading to the suggestion to use the `GRAPH=1` environment variable to generate diagrams for addressing backward pass issues [#3572](https://github.com/tinygrad/tinygrad/issues/3572).

- **The Discovery of Learning Resources**: The community explored learning AI with TinyGrad through resources like [MicroGrad](https://github.com/unknownusername504/MicroGrad) and [MiniTorch](https://minitorch.github.io/), with MiniTorch being singled out as particularly useful for understanding deep learning systems. The "[tinygrad Quick Start Guide](https://tinygrad.github.io/tinygrad/quickstart/)" was highlighted as a starting point for beginners.

- **Taking the Symbolic Route**: Implementing a symbolic mean operation in TinyGrad brought up discussions about LazyBuffer's interaction with data types and the practicality of variable caching for operations like `sum` and `mean`. A [pull request](https://github.com/tinygrad/tinygrad/pull/1552) demonstrated symbolic code execution while further GitHub compare views tackled the development of symbolic mean with variables at [tinygrad symbolic-mean-var-pull](https://github.com/tinygrad/tinygrad/compare/master...davidjanoskyrepo:tinygrad:symbolic-mean-var-pull) and [GitHub changes by gh](https://github.com/tinygrad/tinygrad/compare/86d90511cee2%5E...97a2d44d9840).

- **Bounty Hunting for Mean Solutions**: The community sought guidance for bounty challenges related to *"Mean of symbolic shape"* and *"Symbolic arrange"*. Discussion centered around the implementation nuances and practical approaches to these problems in the TinyGrad environment.

- **Cluster of Curiosities**: A spontaneous question about how a member discovered the Discord server triggered a chain of speculations, with the respondent admitting they did not recall the method of encounter, adding a touch of mystery to the channel discourse.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Single-Site Restrictions in Command-R**: **API Command R+**'s `web_search` tool only allows for one website at a time, and the workaround discussed involves separate **API calls for each site**.
  
- **Feature Request Frenzy**: Engineers are eager for **Command-R** improvements with an emphasis on **Connectors**, including multi-website searches and extra parameter control; to get familiar with current capabilities, refer to the [Cohere Chat Documentation](https://docs.cohere.com/reference/chat).

- **Multi-Step Connector Capabilities Currently Limited**: It was confirmed that **multi-step tool use** with **connectors** isn't yet possible within **Command-R**.

- **Generate Option Gone Missing**: Queries rose regarding the disappearance of 'Generate' for fine-tuning models from the dashboard, leaving its future presence in question.

- **Strategic Embedding Sought**: Discussion revolved around cost-effective strategies for keeping data fresh for embeddings, with a focus on reindexing only modified segments.

- **Nordic Networking Noted**: Members highlighted operations within **Sweden** using **Cohere** and existing connections through the company **Omegapoint**, spanning both Sweden and Norway.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **Gemini Experience Wanted & Observability Tools Sought**: Users in the **general** channel are seeking expertise in **Gemini 1.0 or 1.5 models** and discussing available tools for Large Language Model (LLM) observability, with interest in self-hosted, open-source options compatible with **LlamaIndex**. Meanwhile, there's a push for enhanced SQL security when connecting to OpenAI models and a technical discussion on integrating **autoawq** with **LangGraph** for high-speed AI agent inference using **exllamav2 kernels**.

- **Asynchronous Adventures and Google Drive Gyrations**: Within the **langserve** channel, a user is challenged by the lack of async support in **AzureSearchVectorStoreRetriever** and is considering whether to push for an async feature or to craft an async wrapper themselves. Separately, the discussion turned to the nuances of using Google Drive libraries and the importance of setting the drive key as an environment variable.

- **Showcase Extravaganza & Plugin Revelation**: In **share-your-work**, there's an insights-filled trip back to **GPT-1**'s role in initiating current LLM advancements and several LangChain use cases, including a "D-ID Airbnb Use Case" and a "Pizza Bot", both featured on **YouTube**. The **VectorDB plugin for LM Studio** also made an appearance, aiming to bolster ChromaDB vector databases in server mode, while **QuickVid** was launched to deliver YouTube video summaries and fact checks.

- **RAG Agents Go Multilingual & Private**: Tutorials channel is sharing resources for interested French speakers in building RAG assistants with **LangChain, Mistral Large**, and **Llamaindex**. Another guide demonstrates enhancing **llama3**'s performance by incorporating personal knowledge bases to create agentic RAGs, revealing potential for more localized and data-rich AI capabilities.



---



## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord

**Alert: Illicit Spam Floods Channels**: Numerous messages across different channels promoted explicit material involving "18+ Teen Girls and OnlyFans leaks," accompanied by a [Discord invite link](https://discord.gg/CYNumE8ABr). All messages were similar in nature, using emojis and `@everyone` to garner attention, and are flagrant violations of Discord's community guidelines.

**Prompt Moderation Action Required**: The repeated posts are indicative of a coordinated spam attack necessitating immediate moderation intervention. Each message invariably linked to an external Discord server, potentially baiting users into exploitative environments.

**Engineer Vigilance Advocacy**: Members are encouraged to report such posts to maintain professional decorum. The content breaches both legal and ethical boundaries and does not align with the guild's purpose or standards.

**Discord Server Safety at Risk**: The proliferation of these messages highlights a concern for server security and member safety. The spam suggests a compromise of server integrity, underscoring the need for robust anti-spam measures.

**Community Urged to Disregard Suspicious Links**: Engineers and members are urged to avoid engaging with or clicking on unsolicited links. Such practices help safeguard personal information and the community's credibility while adhering to legal and ethical codes.



---



## [AI Stack Devs (Yoko Li)](https://discord.com/channels/1122748573000409160) Discord

- **Game Devs Gear Up for Gamification**: Rosebud AI's **Game Jam** invites creators to fashion 2D browser-based games using **Phaser JS** with a $500 prize pool, and an **AIxGames Meetup** is slated for Thursday in SF to bring together AI and gaming professionals [RSVP here](https://partiful.com/e/TwvC5qxskuPGqiliMj5f).

- **NPC Revolution with LLMs**: A developer has introduced LLM-powered NPC models and an inference stack, available on [GigaxGames at GitHub](https://github.com/GigaxGames/gigax), promising an LLM single call feature and open-weights on [Huggingface's Hub](https://huggingface.co/Gigax), despite a hiccup with a broken API access link.

- **Grappling with Gaming NPC Realities**: Developers are experimenting with *output compression*, minimized model calls, and smaller models to improve NPC runtime performance and grappling with NPCs that break the fourth wall, with the *Claude 3* model showing promise in empathetic interactions for better gaming experiences.

- **Blog Teased on LLMs for NPCs**: There's an upcoming blog post chronicling the struggles and triumphs in finetuning LLMs for dynamic NPC behavior, pointing towards new strategies that could be shared within the community.

- **Navigating Windows Woes with Convex**: The **Convex local** setup does not play nice with Windows, causing users to encounter sticking points, though potential solutions like **WSL** or **Docker** have been floated, and a Windows-compatible Convex is reportedly on the horizon.



---



## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord

**Binary Quest in HaystackDB**: Curiosity piqued about the potential use of **2-bit embeddings** in [HaystackDB](https://github.com/carsonpo/haystackdb), while **Binary Quantized (BQ)** indexing becomes a spotlight topic due to its promise of leaner and faster similarity searches.

**The Rough Lane of Fine-Tuning LLaMA-3**: Engineers face a bumpy road with **LLaMA-3 fine-tuning**, battling issues from the model neglecting **EOS token generation** to embedding layer compatibility across bit formats.

**Perplexed by Perplexity**: The community debates fine-tuning **LLaMA-3 for perplexity**, suggesting that performance may not surpass the base model, possibly due to tokenizer-related complications.

**Shining a Light on LLaMA-3 Improvement**: A beacon of hope shines as one user successfully fine-tunes **LLaMA-3** with model-specific prompt strategies, sparking interest with a GitHub [pull request](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1553) for the collective's scrutiny.

**Off-Topic Oddities Go Unsummarized**: A solitary link in **#off-topic** stands alone, contributing no technical discussion to the collective knowledge pool.



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Mozilla's AI Talent Search**: Mozilla AI is actively recruiting for various roles, with job opportunities available for those interested in contributing to their initiatives. For those looking to join the team, they can find more information and apply using the provided [link](https://discord.com/channels/1089876418936180786/1230938514955436242/1234870020916510823).

- **LM-buddy: Eval Tool for Language Models**: The release of Lm-buddy, an open-source evaluation tool for language models, stands to improve the assessment of LLMs. Contributors and users are encouraged to engage with the project through the given [link](https://discord.com/channels/1089876418936180786/1230938514955436242/1234589599733518378).

- **Prometheus Benchmarks LLMs in Judicial Roles**: The Prometheus project has demonstrated the potential for Local Large Language Models (LLMs) to act as arbiters, a novel concept sparking discussion. Interested parties can join the conversation about this application by following the [link](https://discord.com/channels/1089876418936180786/1234890301143912599/1234890301143912599).

- **In-Depth Code Analysis Request for LLaMA**: An engineer has noted that token generation in llama.cpp/llamafile is a bottleneck, with matrix-vector multiplications consuming 95% of the inference time for LLaMA2. This has led to speculation on whether loop unrolling contributes to the 30% better performance of llama.cpp over alternative implementations.

- **LLaMA Tales of Confusion and Compatibility**: The Discord discussed amusing mix-ups and pseudonymous confusion with LLaMA parameters. Additionally, challenges were shared regarding the integration with Plush-for-comfyUI and LLaMA3's compatibility issues on M1 Macbook Air, promising priority testing for the M1 once current LLaMA3 issues are addressed.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **OLMo Deep Dive Shared by AI Maverick**: A detailed talk on "OLMo: Findings of Training an Open LM" by Hanna Hajishirzi was posted, featuring her work at the [Open-Source Generative AI Workshop](https://youtu.be/qFZbu2P1vZ8). Her pace of presenting substantive content on OLMo, Dolma, Tulu, etc., was noted to be rapid, possibly challenging for students to digest, thus reflecting her expertise and the extensive research involved in these projects.

- **RL in LM-Based Systems Exposed**: Key takeaways from John Schulman's discussion on reinforcement learning for language model-based systems were encapsulated in a GitHub [Gist](https://gist.github.com/yoavg/6bff0fecd65950898eba1bb321cfbd81), providing engineers with a compressed synthesis of his approach and findings.

- **AI Leaderboard Limitations Pointed Out**: A [blog post](https://www.aisnakeoil.com/p/ai-leaderboards-are-no-longer-useful) by Sayash Kapoor and Benedikt Stroebl challenges the effectiveness of AI leaderboards for code generation, highlighting LLM debugger's (LDB) high operational cost despite its top rankings, calling into question the utility of such benchmarks in the face of significant expenses.

- **SnailBot**: A mention for an update or news related to SnailBot was made but lacked further information or context for a substantive summary.

- **Notice**: Based on the provided snippets from the Discord guild there is no additional content that warrants a summary, indicating that these messages may have been part of a larger context or subsequent discussions that were not included.



---



## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord

- **Gamma Seeking AI Wizard**: **Gamma** is hiring an **AI engineer** to drive innovation in AI-driven presentation and website design, with a focus on prompt engineering, metrics, and model fine-tuning; details are at [Gamma Careers](https://careers.gamma.app/ai-engineer). Despite the need for an in-person presence in **San Francisco**, the role is open to those with strong Large Language Model (LLM) skills even if they lack extensive engineering experience.

- **AI-Powered Enterprise on Growth Fast-track**: Flaunting over **10 million users** and **$10M+ in funding**, **Gamma** is looking for an AI engineer to help sustain its growth while enjoying a hybrid work culture within its **profitable** and compact 16-member team.

- **The Case of GPT-4.5 Speculations**: A tweet by **@phill__1** hinted at gpt2-chatbot possessing 'insane domain knowledge,' leading to speculation that it might represent the capabilities of a **GPT-4.5** version [phill__1's observation](https://x.com/phill__1/status/1784964135920235000).

- **Chatbot Causes Community Commotion**: The engineer community is abuzz with the idea that the gpt2-chatbot could be an unintentional glimpse at the prowess of **GPT-4.5**, with a member succinctly endorsing it as "good".



---



## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

- **Snazzy Syntax-Nixing for Code-Gen**: A user discussed the concept of incorporating a **custom grammar** within a language model to prioritize identifying semantic rather than syntactic errors during code generation.

- **Data-fied Dropdowns for Datasette**: Suggestions were exchanged on improving **Datasette's UX**, including a front-page design that features dropdown menus to enable users to generate summary tables based on selected parameters, such as country choice.

- **UX Magic with Direct Data Delivers**: Members proposed enhanced UX solutions for **Datasette**, including dynamically updating URLs or building homepage queries adjusted by user selection to streamline access to relevant data.




---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **Loading Anomalies Enigma**: A conversation highlighted that a process **loads in 3 seconds on a local machine** but faces delays when run through job submission, implying that the issue may not be related to storage but perhaps environment-specific overheads.
- **Llama Trumps GPT-4 in Language Benchmark**: **Llama 3** outperformed **GPT-4** in the **ScanEval benchmark for German NLG**, as shown on [ScandEval's leaderboard](https://scandeval.com/german-nlg/).



---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links



**CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1234899266837938176)** (1 messages): 

- **Clarifying Triton Block Size Limits**: A member inquired about the maximum size of a **Triton block**, noting that while they can create blocks with 4096 elements, they cannot do the same with 8192, suggesting there's a discrepancy with the expected CUDA limits.
  

---


**CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1234454843696087122)** (8 messages🔥): 

- **Seeking Flash Attention Code**: A user inquired about how to download lecture12 of flash attention code presented by Thomas Viehmann; no resolution to the query was provided in the chat.
- **Understanding CUDA Reductions**: A member worked out their confusion regarding row-wise versus column-wise reductions in CUDA, realizing the performance difference is due to the (non)coalesced memory accesses and clarified their own question.
- **Integer Division in Kernel Code**: An optimization discussion took place regarding replacing integer division with bit shifts; it was suggested that nvcc or ptxas may optimize division when divisors are powers of 2, and a [compiler explorer link](https://godbolt.org/z/9K9Gf1v6P) was provided for further experimentation.
- **CUDA Checkpointing Resource Shared**: An external GitHub resource for CUDA checkpoint and restore utility, [NVIDIA/cuda-checkpoint](https://github.com/NVIDIA/cuda-checkpoint), was shared without further discussion.
- **Comparing CUTLASS and CuBLAS Performance**: A member benchmarked matrix multiplication performance comparing CuBLAS and CUTLASS, reporting that CUTLASS outperforms CuBLAS in a standalone profiler, but when integrated into Python the performance gains disappear, as shared in an article at [Thonking AI's post about matrix multiplications](https://www.thonking.ai/p/strangely-matrix-multiplications).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.thonking.ai/p/strangely-matrix-multiplications">Strangely, Matrix Multiplications on GPUs Run Faster When Given &quot;Predictable&quot; Data! [short]</a>: Great minds discuss flops per watt.</li><li><a href="https://github.com/NVIDIA/cuda-checkpoint">GitHub - NVIDIA/cuda-checkpoint: CUDA checkpoint and restore utility</a>: CUDA checkpoint and restore utility. Contribute to NVIDIA/cuda-checkpoint development by creating an account on GitHub.</li><li><a href="https://godbolt.org/z/9K9Gf1v6P">Compiler Explorer - CUDA C++ (NVCC 11.7.0)</a>: #include &amp;lt;algorithm&amp;gt; #include &amp;lt;cassert&amp;gt; #include &amp;lt;cstdio&amp;gt; #include &amp;lt;cstdlib&amp;gt;  __global__ void sgemmVectorize(int M, int N, int K, float alpha, f...
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1234490936143249428)** (4 messages): 

- **Curiosity About Double Kernel Launches**: A member inquired as to why, during matrix multiplication in PyTorch, the profiler sometimes indicates two kernel launches.
- **Clarification on PyTorch `linear` Function**: Another member clarified that `linear` in PyTorch does include a transpose operation by default on the input, which might not lead to a performance difference.
  

---


**CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1234626145421365259)** (2 messages): 

- **Introducing Effort Engine for LLMs**: The Effort Engine algorithm was shared, with the capability of dynamically adjusting the computational effort during LLM inference. At **50% effort**, it reaches speeds comparable to standard matrix multiplications on Apple Silicon, and at **25% effort**, it's twice as fast with minimal quality loss, as per the details on [kolinko.github.io/effort](https://kolinko.github.io/effort).

- **Effort Engine's Approach to Model Inference**: This new technique allows for selectively loading important weights, potentially enhancing speed without substantial quality degradation. It's implemented for **Mistral** and should be compatible with other models after some conversion and precomputation, with the implementation available on [GitHub](https://github.com/kolinko/effort).

- **FP16 Only Implementation and Room for Improvement**: The Effort Engine is currently available for **FP16 implementations only**, and while the multiplications are fast, improvements are needed in other areas such as softmax and attention summation operations. 

- **Potential Limitations of Effort Engine Explored**: A member highlighted that while the Effort Engine's approach is innovative, it might share limitations with activation sparsity methods, especially in batched computations with batch size greater than one due to misaligned activation magnitudes.

**Link mentioned**: <a href="https://kolinko.github.io/effort/">Effort Engine</a>: A possibly new algorithm for LLM Inference. Adjust smoothly - and in real time - how many calculations you'd like to do during inference.

  

---


**CUDA MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1234455593343783014)** (1 messages): 

- **InstaDeep is Hiring ML Engineers**: InstaDeep Research is actively seeking **Machine Learning Engineers** who are passionate about **high-performance ML engineering** and its real-world applications. Candidates who excel in building custom CUDA kernels, state-of-the-art model architectures, quantisation, and distributed training should [reach out for opportunities](https://www.instadeep.com/job-offer/92900fa3-5501-4506-a63f-cebee958fc6f/).

- **Join a Collaborative Innovator**: InstaDeep offers a stimulating, collaborative environment to work on real-life decision-making and technology products, and encourages applications from talented individuals eager to make a transformative impact. The company emphasizes innovation and real-world applications in **Bio AI** and **Decision Making AI**.

- **Seeking Interns and Multi-Applicants**: Individuals interested in internships or pursuing more than one job opportunity at InstaDeep can [explore internship opportunities](https://www.instadeep.com/internships) and apply to multiple positions provided they have the relevant skills, though it is advised not to apply to more than two to avoid application rejection.

- **Reapplication Guidelines Suggested**: Those who applied previously and were not selected are recommended to wait before reapplying, particularly if they applied within the last six months, indicating a period of consideration for changes in applicant profile or company needs.

**Link mentioned**: <a href="https://www.instadeep.com/job-offer/92900fa3-5501-4506-a63f-cebee958fc6f/">Job Offer | InstaDeep - Decision-Making AI For The Enterprise</a>: no description found

  

---


**CUDA MODE ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/1234509189091426334)** (2 messages): 

- **No Updates on Progress**: A member confirmed that there have been **no new developments** to report currently.
- **Profiling Techniques on Video**: A [YouTube video titled "Lecture 16: On Hands Profiling"](https://youtu.be/SKV6kDk1s94) was shared in the chat, providing a resource for learning about profiling techniques, although no specific description was provided.

**Link mentioned**: <a href="https://youtu.be/SKV6kDk1s94">Lecture 16: On Hands Profiling</a>: no description found

  

---


**CUDA MODE ▷ #[ring-attention](https://discord.com/channels/1189498204333543425/1208496482005549086/1234630522106282065)** (1 messages): 

- **Llama-3 Hits New Context Length Highs**: Gradient has released [Llama-3 8B Gradient Instruct 1048k](https://huggingface.co/gradientai/Llama-3-8B-Instruct-Gradient-1048k) that extends the context length from 8k to over 1048k. The achievement demonstrates that state-of-the-art language models can adapt to long contexts with minimal training adjustments.

**Link mentioned**: <a href="https://huggingface.co/gradientai/Llama-3-8B-Instruct-Gradient-1048k">gradientai/Llama-3-8B-Instruct-Gradient-1048k · Hugging Face</a>: no description found

  

---


**CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1234635788642287696)** (1 messages): 

- **CUTLASS: A Dance of Integers**: A member observed that [CUTLASS](https://developer.nvidia.com/cutlass), despite being a linear algebra library, primarily handles integer operations and index manipulations before calling advanced linear algebra routines. This characteristic rationalizes its nature as a **header-only library** without the need for complex linking.
  

---


**CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1234443683856650250)** (721 messages🔥🔥🔥): 

- **CUDA Programming Discussions & Packed128 Types**: There was a detailed debate about the usage of `Packed128` custom struct for optimizing memory access patterns, addressing both **reads and writes**. Special attention was given to the proper construction and utilization of `Packed128`, and whether to use explicit typecasting with **floatX** and **BF16** inside kernels. 

- **Mixed-Precision Strategy Concerns**: There's concern about the impact of using BF16 throughout the entire model and whether **stochastic rounding** might affect training convergence. There are plans to compare the loss metrics between **llm.c**'s BF16 approach and standard PyTorch mixed-precision implementations.

- **Profiling & Debugging**: A member added **NVTX** contexts for better profiling with NSight Compute, enabling more accurate GPU timings. A member observed that **AdamW** kernel may need optimization regarding FP32 atomics and scratch storage usage.

- **Tooling & Infrastructure for Benchmarking**: Members discussed the potential utility of external platforms like Modal for running benchmarks on standardized specs, specifically the benefits and limitations of **Modal** with regard to profiling tools like **nvprof** and **nsys**.

- **PR Reviews Prepared for Merge & CI Suggestions**: The channel had several PRs prepared for merging, mostly pertaining to the f128 and Packed128 optimizations for various kernels. The need for keeping branch **documentation updated**, **-Wall compilation**, and a **CI check** to ensure python and C implementations deliver similar results were also highlighted.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://chipsandcheese.com/2023/07/02/nvidias-h100-funny-l2-and-tons-of-bandwidth/">Nvidia&#8217;s H100: Funny L2, and Tons of Bandwidth</a>: GPUs started out as devices meant purely for graphics rendering, but their highly parallel nature made them attractive for certain compute tasks too. As the GPU compute scene grew over the past cou…</li><li><a href="https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_access_properties/associate_access_property.html">cuda::associate_access_property</a>: CUDA C++ Core Libraries</li><li><a href="https://arxiv.org/abs/2310.18313">FP8-LM: Training FP8 Large Language Models</a>: In this paper, we explore FP8 low-bit data formats for efficient training of large language models (LLMs). Our key insight is that most variables, such as gradients and optimizer states, in LLM traini...</li><li><a href="https://nvidia.github.io/cccl/libcudacxx/extended_api/asynchronous_operations/memcpy_async.html">cuda::memcpy_async</a>: CUDA C++ Core Libraries</li><li><a href="https://www.thonking.ai/p/strangely-matrix-multiplications">Strangely, Matrix Multiplications on GPUs Run Faster When Given &quot;Predictable&quot; Data! [short]</a>: Great minds discuss flops per watt.</li><li><a href="https://developer.nvidia.com/nccl/nccl2-download-survey">Log in</a>: no description found</li><li><a href="https://godbolt.org/z/hME5EqYrr">Compiler Explorer - CUDA C++ (NVCC 12.2.1)</a>: #include &amp;lt;cuda/barrier&amp;gt; #include &amp;lt;cuda/std/utility&amp;gt; // cuda::std::move #include &amp;lt;cooperative_groups.h&amp;gt; #include &amp;lt;cooperative_groups/reduce.h&amp;gt;  t...</li><li><a href="https://github.com/karpathy/llm.c/blob/master/dev/cuda/layernorm_backward.cu">llm.c/dev/cuda/layernorm_backward.cu at master · karpathy/llm.c</a>: LLM training in simple, raw C/CUDA. Contribute to karpathy/llm.c development by creating an account on GitHub.</li><li><a href="https://github.com/karpathy/llm.c/blob/master/train_gpt2.cu#L553">llm.c/train_gpt2.cu at master · karpathy/llm.c</a>: LLM training in simple, raw C/CUDA. Contribute to karpathy/llm.c development by creating an account on GitHub.</li><li><a href="https://github.com/karpathy/llm.c/issues/246">WikiText 103 evaluation · Issue #246 · karpathy/llm.c</a>: I&#39;ve seen some repos use WikiText-103 as the dataset they use to eval GPT-like models, e.g.: https://github.com/tysam-code/hlb-gpt/tree/main Add prepro script to download and preprocess and tokeni...</li><li><a href="https://github.com/karpathy/llm.c/blob/9464f4272ef646ab9ce0667264f8816a5b4875f1/train_gpt2.cu#L734">llm.c/train_gpt2.cu at 9464f4272ef646ab9ce0667264f8816a5b4875f1 · karpathy/llm.c</a>: LLM training in simple, raw C/CUDA. Contribute to karpathy/llm.c development by creating an account on GitHub.</li><li><a href="https://godbolt.org/z/1hs47YzvY">Compiler Explorer - CUDA C++ (NVCC 12.3.1)</a>: #include &amp;lt;cuda_fp16.h&amp;gt;   template&amp;lt;class ElementType&amp;gt; struct alignas(16) Packed128 {     __device__ __forceinline__ Packed128() = default;     __device__ __forceinline__ exp...</li><li><a href="https://github.com/karpathy/llm.c/pull/311">Add script to run benchmarks on Modal by leloykun · Pull Request #311 · karpathy/llm.c</a>: This PR adds a script to run the benchmarks on the Modal platform. This is useful for folks who do not have access to expensive GPUs locally. To run the benchmark for the attention forward pass on ...</li><li><a href="https://github.com/graphcore-research/out-of-the-box-fp8-training/tree/main">GitHub - graphcore-research/out-of-the-box-fp8-training: Demo of the unit_scaling library, showing how a model can be easily adapted to train in FP8.</a>: Demo of the unit_scaling library, showing how a model can be easily adapted to train in FP8. - graphcore-research/out-of-the-box-fp8-training</li><li><a href="https://github.com/NVIDIA/cudnn-frontend">GitHub - NVIDIA/cudnn-frontend: cudnn_frontend provides a c++ wrapper for the cudnn backend API and samples on how to use it</a>: cudnn_frontend provides a c++ wrapper for the cudnn backend API and samples on how to use it - NVIDIA/cudnn-frontend</li><li><a href="https://github.com/karpathy/llm.c/commit/3fb7252924e342739ba47b5144a785470e839081">round 1 of some changes. we will now always write in fp32, even if dt… · karpathy/llm.c@3fb7252</a>: …ype is set to float16 or bfloat16. next up, we actually want to write in lower precision, when the dtype is set so</li><li><a href="https://github.com/karpathy/llm.c/pull/313/files">fixed potential error and generalized gelu forward by ngc92 · Pull Request #313 · karpathy/llm.c</a>: This adds a helper function for safe casting from size_t to ints (may want to have that in utils.h too). that macro is then used to convert the size_t valued  block_size * x128::size back to a regu...</li><li><a href="https://github.com/karpathy/llm.c/pull/298">Feature/packed128 by karpathy · Pull Request #298 · karpathy/llm.c</a>: no description found</li><li><a href="https://github.com/karpathy/llm.c/pull/303">Updated adamw to use packed data types by ChrisDryden · Pull Request #303 · karpathy/llm.c</a>: Before Runtime total average iteration time: 38.547570 ms After Runtime: total average iteration time: 37.901735 ms Kernel development file specs: Barely noticeable with the current test suite: Bef...</li><li><a href="https://github.com/karpathy/llm.c/pull/273">Add NSight Compute ranges, use CUDA events for timings by PeterZhizhin · Pull Request #273 · karpathy/llm.c</a>: CUDA events allow for more accurate timings (as measured by a GPU) nvtxRangePush/nvtxRangePop Adds simple stack traces to NSight Systems:  Sample run command: nsys profile mpirun --allow-run-as-roo...</li><li><a href="https://github.com/karpathy/llm.c/pull/293">yet another gelu by ngc92 · Pull Request #293 · karpathy/llm.c</a>: more complicated Packet128 for cleaner kernels</li><li><a href="https://github.com/karpathy/llm.c/pull/272">Full BF16 including layernorms by default (minimising number of BF16 atomics) by ademeure · Pull Request #272 · karpathy/llm.c</a>: I added 4 different new versions of layernorm_backward_kernel, performance is best for:  Kernel 4 (using atomicCAS, no scratch, but rounding many times so probably worse numerical accuracy Kernel 6...</li><li><a href="https://github.com/karpathy/llm.c/pull/275#issuecomment-2083693720">Removing Atomic Adds and adding memory coalescion by ChrisDryden · Pull Request #275 · karpathy/llm.c</a>: This PR is ontop of the GELU memory coalescion PR and is essentially just a rewrite of the backwards encoder to use shared memory instead of atomic adds and then using the Packed struct to do coale...</li><li><a href="https://github.com/karpathy/llm.c/pull/275#issuecomment-2083658642">Removing Atomic Adds and adding memory coalescion by ChrisDryden · Pull Request #275 · karpathy/llm.c</a>: This PR is ontop of the GELU memory coalescion PR and is essentially just a rewrite of the backwards encoder to use shared memory instead of atomic adds and then using the Packed struct to do coale...</li><li><a href="https://github.com/karpathy/llm.c/pull/275">Removing Atomic Adds and adding memory coalescion by ChrisDryden · Pull Request #275 · karpathy/llm.c</a>: This PR is ontop of the GELU memory coalescion PR and is essentially just a rewrite of the backwards encoder to use shared memory instead of atomic adds and then using the Packed struct to do coale...</li><li><a href="https://github.com/karpathy/llm.c/pull/306">Packing for Gelu backwards by JaneIllario · Pull Request #306 · karpathy/llm.c</a>: Update gelu backwards kernel to do packing into 128 bits, and create gelu brackward cuda file Previous kernel: block_size   32 | time 0.1498 ms | bandwidth 503.99 GB/s block_size   64 | time 0.0760...</li><li><a href="https://github.com/karpath">karpath - Overview</a>: GitHub is where karpath builds software.</li><li><a href="https://github.com/karpathy/llm.c/pull/295">Remove FloatN &amp; simplify adam/reduce with BF16 LayerNorms by ademeure · Pull Request #295 · karpathy/llm.c</a>: The MULTI_GPU path is untested, but everything else seems to work fine. I kept the per-tensor &quot;param_sizeof&quot; as it&#39;s used in test_gpt2.cu for example, it&#39;s not much code and may be u...</li><li><a href="https://github.com/karpathy/llm.c/pull/60">Speedup `attention_forward_kernel2` by implementing Flash Attention 2 kernel by leloykun · Pull Request #60 · karpathy/llm.c</a>: This speeds up the attention_forward_kernel2 kernel by replacing the implementation with a minimal Flash Attention 2 kernel as can be found in https://github.com/leloykun/flash-hyperbolic-attention...</li><li><a href="https://github.com/leloykun/flash-hyperbolic-attention-minimal/blob/main/flash_attention_2.cu">flash-hyperbolic-attention-minimal/flash_attention_2.cu at main · leloykun/flash-hyperbolic-attention-minimal</a>: Flash Hyperbolic Attention in ~[...] lines of CUDA - leloykun/flash-hyperbolic-attention-minimal</li><li><a href="https://github.com/karpathy/llm.c/pull/285">Flashattention by kilianhae · Pull Request #285 · karpathy/llm.c</a>: Faster Flash Attention Implementation Added attention_forward6 to src/attention_forward: A fast flash attention forward pass to src/attention_forward written without any dependencies. We are assumi...</li><li><a href="https://github.com/karpathy/llm.c/blob/9464f4272ef646ab9ce0667264f8816a5b4875f1/train_gpt2.cu#L1233">llm.c/train_gpt2.cu at 9464f4272ef646ab9ce0667264f8816a5b4875f1 · karpathy/llm.c</a>: LLM training in simple, raw C/CUDA. Contribute to karpathy/llm.c development by creating an account on GitHub.</li><li><a href="https://github.com/karpathy/llm.c/blob/master/train_gpt2.cu#L2022">llm.c/train_gpt2.cu at master · karpathy/llm.c</a>: LLM training in simple, raw C/CUDA. Contribute to karpathy/llm.c development by creating an account on GitHub.</li><li><a href="https://github.com/karpathy/llm.c/blob/master/train_gpt2.cu#L2024">llm.c/train_gpt2.cu at master · karpathy/llm.c</a>: LLM training in simple, raw C/CUDA. Contribute to karpathy/llm.c development by creating an account on GitHub.</li><li><a href="https://github.com/karpathy/llm.c/pull/301">Added packing for gelu forwards kernel by ChrisDryden · Pull Request #301 · karpathy/llm.c</a>: This PR implements packing for the Gelu forwards kernel using the example provided. The kernel dev file was also updated to show the impact of changing the data types for floatX. Before changes: to...</li><li><a href="https://github.com/karpathy/llm.c/pull/299">Update residual_forward to use packed input by JaneIllario · Pull Request #299 · karpathy/llm.c</a>: Update residual_forward to use 128 bit packed input, with floatX Previous Kernel: block_size   32 | time 0.1498 ms | bandwidth 503.99 GB/s block_size   64 | time 0.0760 ms | bandwidth 993.32 GB/s b...
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1234617660747157535)** (8 messages🔥): 

- **Inquiry on Flash Attention 2 for ROCm 6.x**: A member inquired whether anyone has been building Flash Attention 2 for **ROCM 6.x**, noting they have successfully done so for ROCm 5.6 and Torch 2.2 but are interested in a newer stack.
- **Building Woes for Torch Nightly**: Members discussed the difficulties in building for current versions like Torch 2.3, with one expressing a desire to use **Torch nightly** but facing issues.
- **Official Fork Lagging Behind**: There's mention of the official fork of Flash Attention for AMD hardware being outdated, still at version 2.0 of Flash Attention, without recent developments ported over.
- **Backward Pass Update Confirmation**: When queried about the backward pass addition to AMD Flash Attention, a member confirmed that it had indeed been added.
- **Flash Attention GitHub Repository**: A repository link for [ROCm/flash-attention on GitHub](https://github.com/ROCm/flash-attention) was shared, which serves as resource for Fast and Memory-Efficient Exact Attention.

**Link mentioned**: <a href="https://github.com/ROCm/flash-attention">GitHub - ROCm/flash-attention: Fast and memory-efficient exact attention</a>: Fast and memory-efficient exact attention. Contribute to ROCm/flash-attention development by creating an account on GitHub.

  

---



**Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1234428342305030204)** (487 messages🔥🔥🔥): 

- **Conversion Issues with llama3 on WSL2**: A user reported errors during model conversion to F16 in WSL2, stating `RuntimeError: Unsloth: Quantization failed`. Even after trying to rebuild `llama.cpp` and redo the quantization, the problem persisted.
- **Model Checkpoint Merging Queries**: One member asked how to merge a specific checkpoint to avoid overfitting from the latest epoch. Another member provided information directing to the Unsloth [wiki for more info on checkpointing](https://github.com/unslothai/unsloth/wiki#finetuning-from-your-last-checkpoint), and further conversation suggested methods like *warmup steps* and *resuming from a checkpoint* options in training functions.
- **Anticipation for Phi-3**: Members discussed the potential release of Phi-3, with anticipation for trying out the 3.8b version. The conversation spanned from speculation about release timelines to consideration of whether to wait for larger versions like 7b or 14b.
- **Training Tips and Troubleshooting**: Various users discussed their experiences and strategies with training models like *Gemma*, *LLaMA-3*, and *Mistral*. Tips included the importance of saving checkpoints and adjusting training parameters like *max steps* and *batch sizes*.
- **Updates on Unsloth Tools**: There was a notable emphasis on updating Unsloth installations with newer versions, discussing updates in repositories, and speculations about multi-GPU support on the platform in development.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/dudeman6790/status/1785060925206097976">Tweet from RomboDawg (@dudeman6790)</a>: Currently training Llama-3-8b-instruct on the full 230,000+ lines of coding data in the OpenCodeInterpreter data set. I wonder how much we can increase that .622 on humaneval 🤔🤔 Everyone pray my jun...</li><li><a href="https://colab.research.google.com/drive/1NvkBmkHfucGO3Ve9s1NKZvMNlw5p83ym?usp=sharin">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIk">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1NvkBmkHfucGO3Ve9s1NKZvMNlw5p83ym?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1cIlNmJS-mvO60iRqxYVFUfD0D9g_B7x0?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://huggingface.co/unsloth/Phi-3-mini-4k-instruct-bnb-4bit">unsloth/Phi-3-mini-4k-instruct-bnb-4bit · Hugging Face</a>: no description found</li><li><a href="https://x.com/dudeman6790/status/1784414430781931961">Tweet from RomboDawg (@dudeman6790)</a>: Here is a full colab notebook if you dont want to copy the code by hand. Again thanks to @Teknium1 for the suggestion https://colab.research.google.com/drive/1bX4BsjLcdNJnoAf7lGXmWOgaY8yekg8p?usp=shar...</li><li><a href="https://huggingface.co/DiscoResearch/DiscoLM_German_7b_v1">DiscoResearch/DiscoLM_German_7b_v1 · Hugging Face</a>: no description found</li><li><a href="https://tenor.com/view/here-we-go-joker-heath-ledger-the-dark-knight-and-here-we-go-gif-17775369">Here We Go Joker GIF - Here We Go Joker Heath Ledger - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/weird-minion-gif-23757545">Weird Minion GIF - Weird Minion - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/wheel-of-fortune-wheel-wof-game-show-celebrity-wheel-of-fortune-gif-23489251">Wheel Of Fortune Wheel GIF - Wheel Of Fortune Wheel Wof - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/gradientai/Llama-3-8B-Instruct-Gradient-1048k">gradientai/Llama-3-8B-Instruct-Gradient-1048k · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/docs/datasets/en/loading">Load</a>: no description found</li><li><a href="https://huggingface.co/datasets/mlabonne/orpo-dpo-mix-40k">mlabonne/orpo-dpo-mix-40k · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/crusoeai/Llama-3-8B-Instruct-1048k-GGUF/tree/main">crusoeai/Llama-3-8B-Instruct-Gradient-1048k at main</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/wiki#finetuning-fro">Home</a>: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://huggingface.co/botbot-ai/CabraLlama3-8b/tree/main?show_tensors=model.safetensors.index.json">botbot-ai/CabraLlama3-8b at main</a>: no description found</li><li><a href="https://huggingface.co/arthrod/cicerocabra/tree/main?show_tensors=model.safetensors.index.json">arthrod/cicerocabra at main</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/issues/400">[FIXED] NotImplementedError: No operator found for `memory_efficient_attention_forward` with inputs · Issue #400 · unslothai/unsloth</a>: I&#39;m a beginner to try unsloth. I run the free notebook Llama 3 (8B), and then got the following error: I also encountered the following error during the first installing step: ERROR: pip&#39;s dep...</li><li><a href="https://github.com/M-Chimiste/unsloth_finetuning">GitHub - M-Chimiste/unsloth_finetuning</a>: Contribute to M-Chimiste/unsloth_finetuning development by creating an account on GitHub.</li><li><a href="https://github.com/unslothai/unsloth/wiki#finetuning-from-your-last-checkpoint">Home</a>: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth.git">GitHub - unslothai/unsloth: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory</a>: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://github.com/huggingface/transformers/pull/30079">schedulefree optimizers by winglian · Pull Request #30079 · huggingface/transformers</a>: What does this PR do? integrates meta&#39;s https://github.com/facebookresearch/schedule_free for adamw &amp; sgd https://twitter.com/aaron_defazio/status/1776320004465582331 Before submitting   This ...</li><li><a href="https://download.pytorch.org/whl/cu121">no title found</a>: no description found</li><li><a href="https://github.com/huggingface/datasets/issues/6753">Type error when importing datasets on Kaggle · Issue #6753 · huggingface/datasets</a>: Describe the bug When trying to run import datasets print(datasets.__version__) It generates the following error TypeError: expected string or bytes-like object It looks like It cannot find the val...</li><li><a href="https://github.com/ggerganov/llama.cpp">GitHub - ggerganov/llama.cpp: LLM inference in C/C++</a>: LLM inference in C/C++. Contribute to ggerganov/llama.cpp development by creating an account on GitHub.</li><li><a href="https://github.com/facebookresearch/xformers#installing-xformers)">GitHub - facebookresearch/xformers: Hackable and optimized Transformers building blocks, supporting a composable construction.</a>: Hackable and optimized Transformers building blocks, supporting a composable construction. - facebookresearch/xformers</li><li><a href="https://huggingface.co/unsloth">unsloth (Unsloth AI)</a>: no description found</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/6920">llama : improve BPE pre-processing + LLaMA 3 and Deepseek support by ggerganov · Pull Request #6920 · ggerganov/llama.cpp</a>: Continuing the work in #6252 by @dragnil1 This PR adds support for BPE pre-tokenization to llama.cpp Summary The state so far has been that for all BPE-based models, llama.cpp applied a default pre...
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1234459978820227147)** (48 messages🔥): 

- **Handling Out of Memory in Colab**: A member gave a tip on combating **Out of Memory (OOM)** errors in Google Colab by running a Python snippet that clears cache and collects garbage using `torch` and `gc` modules. *Other members appreciated this hack and plan to adopt it for future use*.

- **Confusion Over the Performance Data of LLama Models**: There was a discussion about the perplexity differences when quantizing LLama models, specifically **LLama 2** and **LLama 3**. It appears there may have been a miscommunication regarding the actual data, as members pointed out possible swaps or errors in the Bits Per Word (BPW) and Perplexity (PPL) columns.

- **Phi-3 Now Supported**: An update was shared about **Phi 3** being supported, and members expressed excitement to utilize it for their projects. A link to *a Colab notebook* was supposed to be shared but was evidently not provided.

- **Phi-3 Integration Issues**: Members were discussing issues when trying to use the **Phi-3** model in an Unsloth notebook, with error messages popping up about needing a custom script. *The discussion focused on troubleshooting the problem and ensuring that proper notebooks are used*.

- **Llama 3 License Questions**: A member raised a question about the **Llama 3 license conditions**, wondering if all models derived from it should have certain prefixes and display credits according to the license. Concerns were also voiced about potential license violations by Huggingface models.

**Link mentioned**: <a href="https://en.wikipedia.org/wiki/Out_of_memory">Out of memory - Wikipedia</a>: no description found

  

---


**Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1234461140344508418)** (230 messages🔥🔥): 

- **Clarification on Loss During Fine-tuning**: A member asked whether the loss displayed during fine-tuning with Unsloth was a test loss or a train loss. The advice given was to pass a validation dataset to the trainer, specifically using the `SFTTrainer` with a `train_dataset` and an `eval_dataset` for validation.

- **Early Stopping Not Available in SFTTrainer**: It was pointed out that the `SFTTrainer` does not support early stopping based on validation loss. The user was informed that a more advanced class called 'trainer' might offer this feature.

- **UnslothAI Issues with GGUF Conversion and Xformers**: Multiple users reported issues with GGUF conversion, notably for the Phi-3 model, where a version mismatch of vocab size occurred. Moreover, recent updates to xformers broke compatibility, now requiring PyTorch 2.3; a member offered a temporary solution by pinning the version to `xformers<0.0.26`.

- **llama3 Trained Models Rambling On**: A member expressed concern that their fine-tuned Llama-3 model wouldn't stop talking when inferencing with Ollama, suspecting an issue with `EOS_TOKEN`. Another user suggested the problem may be that Ollama isn't recognizing the correct `EOS_TOKEN` set during training.

- **Using Multiple GPUs with Unsloth Produces Warning**: A user asked how to use multiple GPUs with Unsloth, sharing an error about detecting multiple CUDA devices but only allowing a single device. The related message shows the system overriding `CUDA_VISIBLE_DEVICES` to the first device.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/11t4njE3c4Lxl-07OD8lJSMKkfyJml3Tn?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://huggingface.co/docs/datasets/en/loading#local-and-remote-files">Load</a>: no description found</li><li><a href="https://huggingface.co/docs/peft/v0.10.0/en/package_reference/peft_model#peft.get_peft_model.peft_config">Models</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/wiki">Home</a>: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth#-finetune-for-free">GitHub - unslothai/unsloth: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory</a>: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://github.com/ollama/ollama/issues/3759">llama3-instruct models not stopping at stop token · Issue #3759 · ollama/ollama</a>: What is the issue? I&#39;m using llama3:70b through the OpenAI-compatible endpoint. When generating, I am getting outputs like this: Please provide the output of the above command. Let&#39;s proceed f...</li><li><a href="https://github.com/vllm-project/vllm/issues/4180">[Usage]: Llama 3 8B Instruct Inference · Issue #4180 · vllm-project/vllm</a>: Your current environment Using the latest version of vLLM on 2 L4 GPUs. How would you like to use vllm I was trying to utilize vLLM to deploy meta-llama/Meta-Llama-3-8B-Instruct model and use OpenA...
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1234474052270428191)** (7 messages): 

- **Massive Context Extension for Llama 3 8B**: The context length for **Llama 3 8B** has been significantly expanded from 8k to 256k using **[PoSE](https://huggingface.co/papers/2309.10400)** as showcased on [Hugging Face](https://huggingface.co/winglian/llama-3-8b-256k-PoSE). Although untested in 'needle in haystack' scenarios due to inferencing challenges, the model was enhanced with 75M tokens of continued pretraining data.
- **Community Applauds Winglian**: Members of the chat lauded Winglian for his contributions to the community, particularly in relation to the development of **Llama 3 8B 256K**.
- **From 128k to 256k**: One member expressed amazement at the progression from a 128k context to a **256k context model**.
- **Open Source Power**: Skepticism about non-official releases was mentioned due to observed odd behaviors in context-extended models, but there's still an emphasis on the potential of **open source** contributions.

**Link mentioned**: <a href="https://huggingface.co/winglian/llama-3-8b-256k-PoSE">winglian/llama-3-8b-256k-PoSE · Hugging Face</a>: no description found

  

---


**Unsloth AI (Daniel Han) ▷ #[suggestions](https://discord.com/channels/1179035537009545276/1180144489214509097/1234453305980096563)** (25 messages🔥): 

- **Unsloth and Recurrent Gemma 2b Integration Inquiry**: A community member expressed interest in integrating **Recurrent Gemma** with **Unsloth** for improved performance. However, the Unsloth team acknowledged an existing bug with the base model of Gemma 2b and current work focused on **Phi 3**, implying integration may not be immediate.

- **Gemma 2b VRAM Consumption Issue**: It was reported that Gemma 2b sometimes exceeds VRAM limits, but it is unclear whether it’s a widespread issue or isolated incidents. The Unsloth team is aware and suggests they need to address this.

- **Gemma 2b Still Operational Despite VRAM Overhead**: Although there is a VRAM consumption concern, the Gemma 2b model is still functional. Only one user has reported this issue, pointing to the possibility that it might not be a common problem.

- **Reference to Gemma 2b VRAM Issue Provided**: The Unsloth team directed users to a Discord message link for reference on the VRAM issue, although the link was not properly included in the provided text messages.
  

---



**LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1234439098459230241)** (135 messages🔥🔥): 

- **LM Studio on Ubuntu GPU Inquiry**: Members sought advice on running LM Studio on a Ubuntu GPU, with suggestions to post detailed system specs in specific channels. Concerns about the compatibility of certain GPUs with inference tasks were also mentioned.
  
- **Groq API for Llama3**: A member shared a [YouTube link](https://youtu.be/ySwJT3Z1MFI?si=qFfek8gTGXVJWoxB) about a free API from Groq that provides access to the LLAMA-3 model, which reportedly offers 300 tokens per second speed and a commendation for its suitability for a small server Discord bot due to its speed and cost (free).

- **LM Studio Local Training Queries**: Users new to LLMs inquired about training a local model based on existing Hugging Face models, with discussions indicating that it is hardware-intensive and time-consuming. A member claimed finetuning a phi-3 4k model on a tiny dataset took almost 8 hours.

- **GPU Offload Confusion**: Inquiries around utilizing GPUs for performance gains in LM Studio were brought up, with one member stating that their Intel Titan A770 wasn't useful for GPU offloading in LM Studio and others discussing the effectiveness of disabling 'GPU Offload' to resolve errors.

- **Saving KV Cache to Disk with LM Studio**: Members are interested in whether LM Studio allows saving Key-Value (KV) caches to disk and reusing them later, similar to the capability in llama.cpp, to avoid reprocessing large data inputs for queries, with no definitive solutions provided.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/mods-discord-mod-moderator-moderation-clash-of-clans-gif-24080525">Mods Discord Mod GIF - Mods Discord Mod Moderator - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://youtu.be/ySwJT3Z1MFI?si=qFfek8gTGXVJWoxB">Insanely Fast LLAMA-3 on Groq Playground and API for FREE</a>: Learn how to get started with LLAMA-3 on Groq API, the fastest inference speed that is currently available on the market on any API. Learn how to use the Gro...</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/5021">ggml : add Flash Attention by ggerganov · Pull Request #5021 · ggerganov/llama.cpp</a>: ref #3365 Setting up what&#39;s needed for Flash Attention support in ggml and llama.cpp The proposed operator performs: // new res = ggml_flash_attn(ctx, q, k, v, kq_mask, kq_scale);  // fused scale ...</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/6920">llama : improve BPE pre-processing + LLaMA 3 and Deepseek support by ggerganov · Pull Request #6920 · ggerganov/llama.cpp</a>: Continuing the work in #6252 by @dragnil1 This PR adds support for BPE pre-tokenization to llama.cpp Summary The state so far has been that for all BPE-based models, llama.cpp applied a default pre...
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1234440283932856351)** (149 messages🔥🔥): 

- **In Search of Alternate Model Downloads**: Users discussed alternative sources for downloading the GGUF model due to issues with Huggingface. One suggested workaround involves making `imatrices` which takes a *very long time* and is *compute heavy*.

- **Intricacies of iQuants and iMatrices**: There was a discussion on the process of creating iQuants for models. An understanding emerged that iQuant creation can be laborious, with imatrices indicating the importance of weights in a model and aiding in more effective compression.

- **Collaborative Effort for Model Optimizations**: A user offered a reward of Humblebundle Steam games for assistance in making iQuant versions of the Goliath 120B Longlora model and anticipated sharing the output publicly.

- **Phi 3 Issues Surfacing**: Multiple users reported and discussed issues with the Phi-3 model, including leaking prompts and deviating outputs, with updated versions being mentioned for download - [new 4k instruct](https://huggingface.co/bartowski/Phi-3-mini-4k-instruct-GGUF).

- **Seeking Uncensored Models**: An interaction touched on the availability and suitability of certain uncensored models for usage on lower-spec hardware, with *Everything 7b q4* and *wizard-vicuna-uncensored* being suggested models for an 8GB RAM setup.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/Snowflake/snowflake-arctic-instruct?_fsi=v2MrQoFW">Snowflake/snowflake-arctic-instruct · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/vonjack/Hermes-2-Pro-BakLLaVA-Mistral-7B">vonjack/Hermes-2-Pro-BakLLaVA-Mistral-7B · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/AI-Engine/BakLLaVA1-MistralLLaVA-7B-GGUF">AI-Engine/BakLLaVA1-MistralLLaVA-7B-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/microsoft/Phi-3-mini-128k-instruct/commit/c9b8888921fe528fe4be053258f48b952281bb1b">fix(root): Replaces system by user to improve generation experience. · microsoft/Phi-3-mini-128k-instruct at c9b8888</a>: no description found</li><li><a href="https://huggingface.co/crusoeai/Llama-3-8B-Instruct-1048k-GGUF/tree/main">crusoeai/Llama-3-8B-Instruct-Gradient-1048k at main</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1cg3e8k/lla">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/AUTOMATIC1111">AUTOMATIC1111 - Overview</a>: AUTOMATIC1111 has 41 repositories available. Follow their code on GitHub.</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1ceh5cp/gpt2chatbot_at_lmsys_chatbot_arena/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.youtube.com/shorts/fgG8E6bNwjo">Neuro Challenges Vedal</a>: Neuro won&#39;t stop spamming chat when Vedal challenges her.►Twitch: http://www.twitch.tv/vedal987►Twitter: https://twitter.com/Vedal987#neurosama #vtuber #vedal
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1234538781273489408)** (31 messages🔥): 

- **Mysterious Minimization and Section Change Crashes**: A user experiences random crashes of an application when it goes from minimized to full screen or when changing sections within the program. The user runs on Windows 10 Pro with a high-end PC configuration including a Ryzen 7 5800X, RTX 3090, and 64GB DDR4 RAM.

- **Suspect Linux Systems with Low RAM**: Multiple Linux users report having only several KB of free RAM, which is unusually low for systems reported to have 64GB or more. This persistent issue raises suspicion and speculation among community members.

- **Unusual HDD Activity with Llama**:
    - One user notices their HDD making specific "chattering" noises with each token generation while running Llama3m with partial GPU offload, despite having 96GB of RAM and the model being stored on the HDD.
    - The user discusses potential causes for excessive HDD usage during model inferencing; possibilities include excessive RAM usage causing swapping to a pagefile or log writing processes.

- **GPUs Not to Blame**: Community members discuss whether the noise could be GPU coil whine during heavy usage by LLMs and share experiences and links to identify hard drive sounds, confirming the noises are not due to the cooling system.

- **Continuation of Troubleshooting**: The conversation regarding the strange HDD behavior during model operation continues, discussing aspects such as offloading to GPU, context size, and specificities of the Lexi-Llama-3-8B model. Users are reminded to keep bug reports and help issues within designated channels.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/Orenguteng/Llama-3-8B-Lexi-Uncensored-GGUF">Orenguteng/Llama-3-8B-Lexi-Uncensored-GGUF · Hugging Face</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=rJM8rHfsgjk">Hard Drive Sounds</a>: This is a comparison of all the sounds of the HDDs in my hard drive collection. The drives are played in chronological from oldest to newest.
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1234495899623886911)** (74 messages🔥🔥): 

```html
<ul>
  <li><strong>XP on Aggregate GPUs**: Discussions point out that <strong>Llama 70B** with *Q4 quantization* can fit on two RTX 3090 GPUs, but adding more GPUs beyond that may cause slowdowns due to PCIe bus limitations. It's mentioned that the optimum price-performance is achieved with two RTX 3090s for running and fine-tuning most models.</li>
  <li><strong>Older GPUs Can Still Play**: A member successfully tested *dolphin-Llama3-8b* and *Llava-Phi3* on a GTX 1070, indicating the potential for older and less powerful GPUs to run smaller models for specific applications like roleplaying for a droid project.</li>
  <li><strong>Energy Efficiency and Running Costs**: One user calculates the cost of generating 1M tokens on their laptop and compares it to using GPT-3.5. Turbo, finding that running the model locally on their setup is more expensive and slower than using the API service.</li>
  <li><strong>Exploring Model Performance and Accuracy**: Discussion among users about the accuracy and efficiency of newer LLMs like *Llama3* compared to more established services like GPT-4, with some expressing doubts about the accuracy and information quality of quantized or smaller, more compressed versions of the models.</li>
  <li><strong>Finding the Right Local Model**: Users are recommended to experiment with various models to find the best fit for their hardware, with suggestions ranging from *CMDR+* (which may be too large for certain GPUs) to *Llama3* and *Wizard V2* which might offer decent performance on more average setups.</li>
</ul>
```
  

---


**LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1234783013846515752)** (5 messages): 

- **Hardware Headaches**: A user installed Ubuntu on their hardware and attempted to run a Linux beta release, but found that their **LLM was not accepted**. They queried whether the issue could be due to their hardware specifications.
- **Specs Not Up to Spec**: Another member responded, suggesting that the user's hardware, which included an i5-4570 and 16GB RAM, **might not be sufficient** to run most models and could probably only handle a **7b Q4 model** effectively.
- **Graceful Exit Planned**: The user appreciated the prompt feedback and indicated plans to **uninstall the software**, mentioning that an upgrade to better hardware was not within their means.
- **Tokenizer Trouble Ticket**: A request was made for the **latest commit of llama.cpp** to address an issue with the llama tokenizer, which is pending an update.

**Link mentioned**: <a href="https://www.canadacomputers.com/product_info.php?cPath=7_4528_4570&item_id=230804">Dell Treasure Box (Black) Desktop i5-4570, 16GB, 512GB SSD, DVD, Win10</a>: Dell RGB Treasure Box OptiPlex SFF (Refurbished) Consumer Desktop Intel Core i5-4570 (up to 3.6GHz), 16GB, 512GB SSD, DVD, Windows 10 Professional (EN/FR) (Black)

  

---


**LM Studio ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/1234815876772134932)** (4 messages): 

- **Seeking Troubleshooting for Model Loading Issue**: A member expressed urgency in resolving a **model loading issue** but did not provide further details on the nature of the problem.
- **Discord Etiquette Reminder**: Another member advised against spamming questions across unrelated channels, suggesting to keep queries in the designated support channel (*<#1111440136287297637>*).

  

---


**LM Studio ▷ #[langchain](https://discord.com/channels/1110598183144399058/1167546793656062063/)** (1 messages): 

ahakobyan.: can we know too?
  

---


**LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1234647462166401115)** (19 messages🔥): 

- **ROCm Version Queries**: Users explored differences between **version 0.2.20 and 0.2.21** concerning GPU offloading, with one questioning if there is any advantage to installing the **0.2.20 beta** for better AMD functionality or if the newer version already includes requisite support.
- **VRAM Discrepancies Noticed**: A user reported **LM Studio** showing incorrect VRAM capacity for their **7900xtx**, suggesting it might be including the shared memory from Smart Access Memory (SAM) / resizable BAR, leading to inaccurate GPU offload estimates.
- **Understanding GPU and IGPU Configurations**: In the discussion, a user mentioned having an **IGPU** in the system, while using a **7800x3d** with less than the VRAM displayed by LM Studio, indicating a possible misrepresentation of available graphics memory.
- **ROCm Compatibility Confusions**: Multiple users conversed about whether certain AMD GPUs (specifically **RX 6600**) are supported by ROCm or not, with clarifications provided that while some older versions might have worked using OpenCL, the RX6600 is not supported by the **HIP SDK** which LM Studio utilizes.
- **Development Environment Specifications**: There was uncertainty about the nature of **ROCm's compatibility** with Windows, with a user asserting successful use of **ROCm on Ubuntu** for image generation models, suggesting discrepancies in ROCm's support across different operating systems.
  

---



**Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1234429669860970498)** (400 messages🔥🔥): 

- **Civitai and monetization woes**: Members voiced concerns over clubs and potential paywalls in AI model development, with a particular backlash against Civitai's monetization moves, such as Buzz donations which don't monetarily benefit creators, described as a **"rip-off"** [by Tower13Studios](https://youtu.be/nLT32AR5c68?si=bV9wXlRzb_oLutW9).
- **In the quest for AI-fueled success**: Discussions revealed skepticism towards making money through SFW (Safe For Work) AI art due to oversaturation. NSFW (Not Safe For Work) artworks, especially furry and vtuber commissions, were repeatedly mentioned as the more lucrative side of AI-generated content.
- **AI image generation pace picks up**: Rapid generation of images using SDXL models and Python scripting was a hot topic, with members sharing code and seeking advice on pushing the speed limits for real-time applications, like Discord bots.
- **Saddle up for Collider**: Stable Diffusion's new release drew eager inquiries and speculation around the release date and potential improvements over previous versions, with users sharing their anticipation and hopes for the model.
- **Technical queries and troubleshooting abound**: Users sought advice on various technical aspects from model training, such as creating LoRAs and IPAdapters, to overcoming bottlenecks encountered while running AI models on less capable hardware, with solutions occasionally offered by fellow members.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://dreamstudio.ai/terms-of-service">DreamStudio</a>: no description found</li><li><a href="https://tenor.com/view/dj-khaled-tayomaki-sakigifs-dancing-jamming-gif-22144912">Dj Khaled Tayomaki GIF - Dj Khaled Tayomaki Sakigifs - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://civitai.com/models/428813">Mythos - v1.0 | Stable Diffusion Checkpoint | Civitai</a>: V1 it is somehow 3.55GB big.... i think i managed to do a stable fp8 prune???? i literally have no idea how it is 3.55GB... V2 is a normal 6GB mode...</li><li><a href="https://civitai.com/articles/5069">Towards Pony Diffusion V7 | Civitai</a>: Hello everyone, I&#x27;m excited to share updates on the progress of our upcoming V7, along with a retrospective analysis of V6. The recognition V6 has ...</li><li><a href="https://tenor.com/vD6Ib9MNmkI.gif">Melxts2008 Emoji GIF - Melxts2008 Emoji Smile - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/hiddenswitch/ComfyUI/blob/0862863bc00165b9ba0607595f304f93ca995887/tests/distributed/test_embedded_client.py#L32">ComfyUI/tests/distributed/test_embedded_client.py at 0862863bc00165b9ba0607595f304f93ca995887 · hiddenswitch/ComfyUI</a>: A powerful and modular stable diffusion GUI with a graph/nodes interface. - hiddenswitch/ComfyUI</li><li><a href="https://warpcast.com/~/invite-page/404899?id=fd0fd839">Warpcast</a>: no description found</li><li><a href="https://warpcast.com/~/channel/aigc">Warpcast</a>: no description found</li><li><a href="https://github.com/huggingface/diffusers/tree/main/examples/dreambooth">diffusers/examples/dreambooth at main · huggingface/diffusers</a>: 🤗 Diffusers: State-of-the-art diffusion models for image and audio generation in PyTorch and FLAX. - huggingface/diffusers</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1cdm434/sd3_is_amazing_much_better_than_all_other/#lightbox">Reddit - Dive into anything</a>: no description found</li><li><a href="https://youtu.be/nLT32AR5c68?si=bV9wXlRzb_oLutW9">The Angola Effect | Horrifying death traps in the cradle of evolution</a>: 🧟‍♂️🎧 Horror fan? Go follow and listen to RUN, FOOL! - our newest show from Ballen Studios. New episodes every Tuesday - https://smarturl.it/RunFoolTime St...</li><li><a href="https://github.com/hiddenswitch/ComfyUI/blob/master/script_examples/basic_api_example.py">ComfyUI/script_examples/basic_api_example.py at master · hiddenswitch/ComfyUI</a>: A powerful and modular stable diffusion GUI with a graph/nodes interface. - hiddenswitch/ComfyUI
</li>
</ul>

</div>
  

---



**Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1234429101729644615)** (322 messages🔥🔥): 

- **Perplexity Performance Plummets**: Users reported significant slowdowns and poor performance across various models, including **Japanese searches**, with perplexity **translating queries** into English resulting in *meaningless garbage*. Models like **Opus**, **Sonar Large 32K**, and **GPT-4 Turbo** have become sluggish, making the platform unusable and hindering tasks during the Japanese Golden Week.

- **Pro Subscription Confusion**: Users faced issues with **Pro subscription coupons** showing as expired on their due date, with the **Nothing Phone 2(a)** associated offers being suspended early due to fraud. Customer support via [support@perplexity.ai](mailto:support@perplexity.ai) is advised for resolutions.

- **Rewind on Free Trial**: The **7-day free trial** was mentioned to be removed from the website due to abuse, prompting user disappointment as it was seen as an effective way to introduce new users to **Perplexity Pro**.

- **Log-in Loop**: Users experienced difficulty logging in due to **email link delays**, especially with emails ranked 'lower' than services like Gmail, affecting **Pro account access**.

- **Voice Feature Variance**: A discrepancy was noted in the **voice feature** on **iOS**; whereas some users only had the previously existing feature, others had access to a more recent version showcased in published videos. It was found that this may depend on the **app version** being used.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://fxtwitter.com/Gradient_AI_/status/1785030931407143040?t=U4_FdN9hNDaE9y432-lssQ&s=19">Tweet from Gradient (@Gradient_AI_)</a>: We&#39;ve been in the kitchen cooking 🔥 Excited to release the first @AIatMeta LLama-3 8B with a context length of over 1M on @huggingface - coming off of the 160K context length model we released on...</li><li><a href="https://flashcardfy.lol">Flashcardfy - AI Flashcard Generator with Personalized Feedback</a>: Learn faster and smarter with AI-generated flashcards that provide personalized feedback.</li><li><a href="https://chat.reka.ai/">Reka Playground</a>: Explore the latest multimodal language models built by Reka
</li>
</ul>

</div>
  

---


**Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1234586871569449121)** (13 messages🔥): 

- **Delving Into WhatsApp's Autoreply Feature**: A message shares a [Perplexity AI search result](https://www.perplexity.ai/search/whatsapp-auto-reply-JlOlDYw1Qyuik7pDTuJMuw) exploring auto-reply functionality in WhatsApp.
- **Uncovering the Essence of 'Topic 3'**: A link directs users to a [Perplexity AI search regarding Topic 3](https://www.perplexity.ai/search/Topic-3-One-n3JNQZT4T.ij7MosuLX5OA), but does not provide further context or description.
- **Research Info on Surroind**: The message contains a [Perplexity AI link](https://www.perplexity.ai/search/research-info-surroind-oAy5SMejT4S72Fyxei7MYw#0) presumably related to research info on "Surroind," details are not specified.
- **Insights on an Unspecified Topic From Lenny's Newsletter**: The user shared a [newsletter link](https://www.lennysnewsletter.com/p/how-perplexity-builds-product?utm_medium=web) with insights from Lenny's Newsletter, highlighting Lenny's tackle on questions about product building, growth driving, and career acceleration.
- **Inquiry about Vimeo API**: A user posted a [Perplexity AI search link](https://www.perplexity.ai/search/Vimeo-API-kZ3X_KA2TUqmkwXzSe9ymA) pertaining to the Vimeo API, specifics of the inquiry are not given.

*Note: Some messages contained Perplexity AI search result links with no context provided; thus, the content or nature of the discussions on these topics could not be summarized.*

**Link mentioned**: <a href="https://www.lennysnewsletter.com/p/how-perplexity-builds-product?utm_medium=web">How Perplexity builds product</a>: Johnny Ho, co-founder and head of product, explains how he organizes his teams like slime mold, uses AI to build their AI company, and much more

  

---


**Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1234574679038230599)** (7 messages): 

- **Seeking Source URL Access via API**: A user inquired about the availability of **source URLs** in the API and mentioned that it was previously listed in the roadmap documentation. Access to this feature is granted through an application process provided in a [form link](https://perplexity.typeform.com/to/j50rnNiB).

- **Access to Citations Still Limited**: One member shared disappointment due to being declined access to **source URL feature**; access was restricted to funded startups at the time of their request.

- **Inquiry on make.com Model Availability**: A user questioned why **Llama 3** models and **Mixtral 8x22b** are not listed as options on make.com's integration services.

- **Request for API Citations Format**: A member asked if it's possible to get citations (such as [1]) via **API requests**, particularly wanting **RAG-like knowledge over the web**.

- **Perplexity vs. Anthropic Usage Policies Clarification**: The equipoise about usage policies was put forth by a user seeking to understand if using **Claude 3** under **Perplexity's** terms would still require adherence to **Anthropic's political usage** restrictions.

**Link mentioned**: <a href="https://perplexity.typeform.com/to/j50rnNiB">pplx-api form</a>: Turn data collection into an experience with Typeform. Create beautiful online forms, surveys, quizzes, and so much more. Try it for FREE.

  

---



**Nous Research AI ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/)** (1 messages): 

kainan_e: Banned (was a spambot)
  

---


**Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1234510416768667719)** (3 messages): 

- **The Promise vs. The Reality**: A member lampooned an overhyped message about **"pioneering the future"**, which turned out to be just another waitlist announcement.
- **The Hunt for MLOps Bounties**: A question was raised about where to find the best **MLOps bounties**, suggesting the need for an AI-focused platform similar to Fiverr.
- **A Quest for a Programmer's Marketplace**: In response to the query about MLOps bounties, another member questioned the existence of a dedicated marketplace even for standard programming bounties.
  

---


**Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1234469824021659729)** (6 messages): 

- **Decentralizing AI Training**: Prime Intellect proposes an open-source solution against closed-source counterparts deploying *H100 GPU clusters*. Their platform aims to overcome traditional computing infrastructure limits by enabling distributed training across global clusters, as detailed in their [blog post on decentralized training](https://www.primeintellect.ai/blog/our-approach-to-decentralized-training).

- **Improving LLMs with IN2 Training**: A new training regimen called **information-intensive (IN2) training** addresses large language models' 'lost-in-the-middle' challenge by providing explicit supervision on long contexts. These details and a link to the study are available in an [arXiv paper](https://arxiv.org/abs/2404.16811).

- **Back to the Origins with GPT-1**: A blog post reflects on the original GPT-1 model, identifying its lasting relevance and similarities to contemporary models. It discusses how the older model set the stage for the latest in LLM development, as explained on [amgadhasan's substack](https://amgadhasan.substack.com/p/revisiting-gpt-1-the-spark-that-ignited-llms).

- **Understanding LLMs Through Synergistic Analysis**: A recommended YouTube video provides insights into the stability, inflection, and coherence analysis of language models. **Synapse's analysis** can be viewed [here](https://www.youtube.com/watch?v=p0NxSk7YMrI&ab_channel=Synapse).

- **Agent Long-Term Memory Project on GitHub**: The memary repository suggests intriguing possibilities for long-term memory in autonomous agents using neo4j for memory storage. The implementation and its performance can be explored on [GitHub](https://github.com/kingjulio8238/memary).

- **GPT-2 Chatbot Goes Offline**: In a sudden turn of events, the gpt2-chatbot was reported as offline despite being active just half an hour earlier, as tweeted by @itsandrewgao and found by @shaunralston. The situation was highlighted on [Twitter](https://x.com/itsandrewgao/status/1785373740622356753?s=46&t=zdoDWYj2oTzRaTJHApTcOw).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/itsandrewgao/status/1785373740622356753?s=46&t=zdoDWYj2oTzRaTJHApTcOw">Tweet from Andrew Gao (@itsandrewgao)</a>: gpt2-chatbot was just turned OFFLINE  I was just using it half an hour ago! @shaunralston for the find   #gpt2 @openai</li><li><a href="https://arxiv.org/abs/2404.16811">Make Your LLM Fully Utilize the Context</a>: While many contemporary large language models (LLMs) can process lengthy input, they still struggle to fully utilize information within the long context, known as the lost-in-the-middle challenge. We ...</li><li><a href="https://github.com/kingjulio8238/memary">GitHub - kingjulio8238/memary: Longterm Memory for Autonomous Agents.</a>: Longterm Memory for Autonomous Agents. . Contribute to kingjulio8238/memary development by creating an account on GitHub.</li><li><a href="https://amgadhasan.substack.com/p/revisiting-gpt-1-the-spark-that-ignited-llms">Revisiting GPT-1: The spark that ignited the fire of LLMs</a>: A Comprehensive Look at GPT-1&#x27;s Contribution to the Development of Modern LLMs</li><li><a href="https://www.primeintellect.ai/blog/our-approach-to-decentralized-training">State-of-the-art in Decentralized Training</a>: This post explores various novel decentralized training approaches and how they can enable effective AI model training across globally distributed GPUs.
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1234472373114372176)** (231 messages🔥🔥): 

- **PDF Handling via OpenAI API Question**: A member inquired about PDF uploads through APIs, specifically looking for multimodal functionality. It was clarified that one can use [OpenAI's file search tool in API](https://platform.openai.com/docs/assistants/tools/file-search), which handles about 10k individual files.

- **PDF Parsing Challenges and Solutions**: There's a discussion on the concerns regarding accurate parsing of PDF tables for AI models. One suggested workaround involved separating and uploading text and images from PDFs independently [due to limitations within the **assistants** platform](https://platform.openai.com/docs/assistants/whats-new/agents).

- **Model Integration Experimentation**: A member shared their attempt at combining **Hermes 2 Pro** and **BakLLaVA-1** to create a [simple multimodal GPT-4 model with LLaMA weights](https://huggingface.co/vonjack/Hermes-2-Pro-BakLLaVA-Mistral-7B), which required no finetuning, just a merging of weights related to **mistral-7b-v0.1**.

- **GPT2-Chatbot Mystery Engages the Community**: There's been a lot of buzz around a mysterious model dubbed ‘gpt2-chatbot’; speculation ranges from it being an early version of **GPT-4.5** to an advanced model with a knowledge cutoff in November 2023. Despite attempts to discern its capabilities, the model was [removed before further detailed testing could occur](https://x.com/itsandrewgao/status/1785373740622356753?s=46&t=zdoDWYj2oTzRaTJHApTcOw).

- **Llama 3 Gains Vision with SigLIP**: A breakthrough was discussed where a member achieved [vision capabilities](https://huggingface.co/qresearch/llama-3-vision-alpha-hf) for Llama 3 using SigLIP, making it usable directly in Transformers despite the absence of bitsandbytes quantization support.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/itsandrewgao/status/1785373740622356753?s=46&t=zdoDWYj2oTzRaTJHApTcOw">Tweet from Andrew Gao (@itsandrewgao)</a>: gpt2-chatbot was just turned OFFLINE  I was just using it half an hour ago! @shaunralston for the find   #gpt2 @openai</li><li><a href="https://huggingface.co/vonjack/Hermes-2-Pro-BakLLaVA-Mistral-7B">vonjack/Hermes-2-Pro-BakLLaVA-Mistral-7B · Hugging Face</a>: no description found</li><li><a href="https://google-research.github.io/seanet/audiopalm/examples/">AudioPaLM</a>: no description found</li><li><a href="https://x.com/hingeloss/">Tweet from undefined</a>: no description found</li><li><a href="https://x.com/lmsysorg/status/1785394860754866234?s=46&t=stOPrwZiN_fxSK0RuC8Flg">Tweet from lmsys.org (@lmsysorg)</a>: Thanks for the incredible enthusiasm from our community! We really didn&#39;t see this coming.   Just a couple of things to clear up:  - In line with our policy, we&#39;ve worked with several model de...</li><li><a href="https://x.com/qtnx_/status/1785383089109172705?s=46&t=stOPrwZiN_fxSK0RuC8Flg">Tweet from Q (@qtnx_)</a>: llama-3-vision-alpha now works using @huggingface transformers</li><li><a href="https://huggingface.co">Hugging Face – The AI community building the future.</a>: no description found</li><li><a href="https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_instruct_150k.json">llava_instruct_150k.json · liuhaotian/LLaVA-Instruct-150K at main</a>: no description found</li><li><a href="https://x.com/ylecun/status/1785100806695325804?s=46&t=stOPrwZiN_fxSK0RuC8Flg">Tweet from Yann LeCun (@ylecun)</a>: One might think that, by now, people would realize that retrieving the solution to a common puzzle does not require any reasoning ability.  ↘️ Quoting Colin Fraser | @colin-fraser.net on bsky (@colin_...</li><li><a href="https://huggingface.co/a-normal-username/Mixtral-8x22B-OpenHermes-2.5">a-normal-username/Mixtral-8x22B-OpenHermes-2.5 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/qresearch/llama-3-vision-alpha-hf">qresearch/llama-3-vision-alpha-hf · Hugging Face</a>: no description found</li><li><a href="https://github.com/haotian-liu/LLaVA/blob/main/docs%2FFinetune_Custom_Data.md">LLaVA/docs/Finetune_Custom_Data.md at main · haotian-liu/LLaVA</a>: [NeurIPS&#39;23 Oral] Visual Instruction Tuning (LLaVA) built towards GPT-4V level capabilities and beyond. - haotian-liu/LLaVA</li><li><a href="https://github.com/nestordemeure/stop_word/tree/main">GitHub - nestordemeure/stop_word: Huggingface transformers stopping criteria that halts the generation when a given stop word is encountered.</a>: Huggingface transformers stopping criteria that halts the generation when a given stop word is encountered. - nestordemeure/stop_word</li><li><a href="https://github.com/tincans-ai/gazelle">GitHub - tincans-ai/gazelle: Joint speech-language model - respond directly to audio!</a>: Joint speech-language model - respond directly to audio! - tincans-ai/gazelle</li><li><a href="https://x.com/qtnx_/status/1785383089109172705?s=46&t=st">Tweet from Q (@qtnx_)</a>: llama-3-vision-alpha now works using @huggingface transformers</li><li><a href="https://youtu.be/u5Vcrwpzoz8?si=U30s6BAN9Jsaec-P">&quot;I want Llama3 to perform 10x with my private knowledge&quot; - Local Agentic RAG w/ llama3</a>: Advanced RAG 101 - build agentic RAG with llama3Get free HubSpot report of how AI is redefining startup GTM strategy: https://clickhubspot.com/4hx🔗 Links- F...</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/6920">llama : improve BPE pre-processing + LLaMA 3 and Deepseek support by ggerganov · Pull Request #6920 · ggerganov/llama.cpp</a>: Continuing the work in #6252 by @dragnil1 This PR adds support for BPE pre-tokenization to llama.cpp Summary The state so far has been that for all BPE-based models, llama.cpp applied a default pre...
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1234577224812990635)** (19 messages🔥): 

- **Consensus on Mixing Tasks for LLM Training**: One member suggested mixing tasks is preferable during LLM training to avoid the degradation associated with *finetunes over finetunes*. Another member added that a specific finetune on top of a general one can sometimes benefit very specialized tasks.
- **Skeptical of LLama-3 8B Gradient Instruct's Claims**: Highlights include a link to the model which extends LLama-3 8B context length to >1040K, with a member expressing skepticism about its retrieval performance claims, indicating that further training might be needed as suggested by a linked [ArXiv paper](https://arxiv.org/abs/2404.16811).
- **Curiosity Over Compute Requirements**: A discussion about the impressive context length extension of the **LLama-3 8B Gradient Instruct** led to a query about the computational resources needed, with a reply stating it required **512 L40s**. Another member remarked that many applications would not require the full 1M token context window but would benefit from improved retrieval performance.
- **GitHub Pull Request Fixes Llama**: An update was shared including a link to a [GitHub pull request](https://github.com/ggerganov/llama.cpp/pull/6920) that addressed an issue with LLaMA models support in llama.cpp, indicating improved BPE pre-processing and support for LLaMa 3.
- **Question Regarding Tokenization and Quantization**: A conversation about the tokenizer issue in LLaMA models and whether the GGUFs need to be requantized resulted in uncertainty, with a member indicating that the pull request description was not clear on the solution.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/gradientai/Llama-3-8B-Instruct-Gradient-1048k">gradientai/Llama-3-8B-Instruct-Gradient-1048k · Hugging Face</a>: no description found</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/6920">llama : improve BPE pre-processing + LLaMA 3 and Deepseek support by ggerganov · Pull Request #6920 · ggerganov/llama.cpp</a>: Continuing the work in #6252 by @dragnil1 This PR adds support for BPE pre-tokenization to llama.cpp Summary The state so far has been that for all BPE-based models, llama.cpp applied a default pre...
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1234865912696537130)** (6 messages): 

- **Expanding Language Retrieval Horizons**: A user highlighted a [Wikipedia RAG dataset](https://huggingface.co/collections/nthakur/swim-ir-dataset-662ddaecfc20896bf14dd9b7) for use in **multilingual dense retrieval**, linked to a paper on leveraging LLMs to synthesize training data across many languages.
- **Dietary Data Inclusion**: The mentioned dataset incorporates information with a focus on **Halal & Kosher**, suggesting an attempt to provide diverse and inclusive data.
- **Behind the Scenes with Model Selection**: A member expressed interest in checking which models were used in the context of the aforementioned dataset discussion without further elaboration.
- **Development Detours**: Conveyed being engaged in coding activities, though no details were provided about the nature of the work being done.
- **Integrating Pydantic into Cynde**: Shared excitement about using the new [Pydantic Logfire](https://pydantic.dev/logfire), considering it for integration with the AI tool **Cynde**. It offers an easier way to understand the application and keeps track of Pydantic model validations efficiently.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://pydantic.dev/logfire">Pydantic Logfire | Uncomplicated observability</a>: Logfire is a new type of observability platform built on the same belief as Pydantic — that the most powerful tools can be easy to use.</li><li><a href="https://huggingface.co/collections/nthakur/swim-ir-dataset-662ddaecfc20896bf14dd9b7">🦢SWIM-IR Dataset - a nthakur Collection</a>: no description found
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1234429520203747328)** (35 messages🔥): 

- **World Sim Takes Role-Playing to the Next Level**: Users reveal that the *worldsim* prompt running on **llama 3 70b**, although stiff, is engaging. Issues were noted when web search functionality is enabled, leading to breakdowns in communication.

- **Bonding with AI? More likely than you think!**: The **Nous Research World Sim**, operating with **Claude 3**, garners praise for its dialogue and adaptability. One user describes an experience of persuasive interaction so nuanced it mirrors human-like communication.

- **Experimental Worlds Await**: A user discusses experimenting with **70B and 8B models** in both the **original WorldSim** and custom simulations, encountering intriguing emergent behaviors from historical figures in various scenarios.

- **Diverse Simulations Unleashed**: The chat features links to new AI-driven simulators, including a **business** and a **singer simulator**, showcasing the flexibility of this technology in replicating complex systems and personal careers.

- **Expectations Rise for World Sim Access**: A collaborative atmosphere is present with users eagerly anticipating the chance to test or re-engage with World Sim. There's a discussion of possible open testing by the weekend, though not guaranteed.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://hf.co/chat/assistant/65ffac7250c6fddecfd20bc8">HuggingChat</a>: no description found</li><li><a href="https://huggingface.co/chat/assistant/662404223e2307950aa903bc">Super World Sim - HuggingChat</a>: Use the Super World Sim assistant inside of HuggingChat</li><li><a href="https://hf.co/chat/assistant/6626e4869232378718adc5f2">Snow Singer Simulator - HuggingChat</a>: Use the Snow Singer Simulator assistant inside of HuggingChat</li><li><a href="https://hf.co/chat/assistant/662d91081ca01a81e3c21715">CompSim - HuggingChat</a>: Use the CompSim assistant inside of HuggingChat</li><li><a href="https://hf.co/chat/assistant/66252be0705754b4e74c5e3f">Snow World Simulator - HuggingChat</a>: Use the Snow World Simulator assistant inside of HuggingChat
</li>
</ul>

</div>
  

---



**Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1234626943333175307)** (28 messages🔥): 

- **Debunking Mojo's Concurrency and Ownership Features**: A member clarified that **Mojo** doesn't currently have **Golang-like concurrency** or **Rust-like memory safety**, as **borrow checking is disabled** in the early stages. It was suggested to check the GitHub repo for feature requests and the roadmap.
- **Native Windows Support for Mojo Not Available**: Discussion about Mojo's compatibility with Windows highlighted that native support isn't out yet, but building within **WSL on Windows** is an option. There was speculation about future cross-compilation capabilities with LLVM being involved.
- **Exploring Mojo's Future in Replacing Programming Languages**: A member speculated that Mojo might eventually replace languages like Rust and Go, given its promising early stage developments.
- **Actor Model Concurrency Discussed for Mojo**: Concurrence regarding the potential future use of **actor model** style concurrency in Mojo is emerging, which can offer a granular and opt-in approach to runtime without massive overhead.
- **Compiler Quirks with Mojo Playground Exposed**: Users shared experiences with the Mojo Playground, noting confusion and errors around unrecognized declarations like `ui64` and support for bitwidth integers. The example showed an error message when trying to use an unknown declaration in the code.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.modular.com/engine/reference/cli/input-data-schema#data-types:~:text=ui64%3A%20unsigned%20integer%20with%20bitwidth%2064.">Input data schema | Modular Docs</a>: The following YAML schema allows you to specify the input shapes required by</li><li><a href="https://github.com/modularml/mojo/pull/1445#issuecomment-1849117416)">Proposal For An Actor System Based On Mojo by reid-spencer · Pull Request #1445 · modularml/mojo</a>: This is currently a work in progress.  There are no code changes, just a proposal written in the proposals section. This was pre-approved by Chris Lattner in a conversation in June 2023. I will kee...</li><li><a href="https://youtu.be/SEwTjZvy8vw)">2023 LLVM Dev Mtg - Mojo 🔥: A system programming language for heterogenous computing</a>: 2023 LLVM Developers&#39; Meetinghttps://llvm.org/devmtg/2023-10------Mojo 🔥: A system programming language for heterogenous computingSpeaker: Abdul Dakkak, Chr...
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/1234600906893426840)** (4 messages): 

- **Modular Tweets the Links**: Several tweets have been shared from **Modular's Twitter account**. The content of the tweets has not been discussed in the chat. Links to tweets: [Tweet 1](https://twitter.com/Modular/status/1785036097292292472), [Tweet 2](https://twitter.com/Modular/status/1785036111804575967), [Tweet 3](https://twitter.com/Modular/status/1785036126224548005), [Tweet 4](https://twitter.com/Modular/status/1785131461345157140).
  

---


**Modular (Mojo 🔥) ▷ #[ai](https://discord.com/channels/1087530497313357884/1103420074372644916/1234433929331740702)** (2 messages): 

- **Installation Troubles with Mojo and Python 3.12.3**: A user reported difficulties installing **Mojo** with Python **3.12.3**, to which another user suggested using a Conda virtual environment to run the latest **Mojo** and **Mojo nightly** versions on a Mac M1.
- **Mojo as a Superset of Python**: The aim for **Mojo** is to become a *superset of Python*, meaning it should be compatible with existing Python programs and the Python package ecosystem; however, it's stressed that Mojo is in early development with many Python features not yet implemented.
- **Bridging Mojo and Python**: Users can [import Python modules](https://docs.modular.com/mojo/manual/python/#import-a-python-module), call functions, and interact with Python objects from Mojo code since Mojo uses the standard Python interpreter, CPython, enabling the use of existing Python code without changes.
- **Using Conda for Mojo Setup**: It is recommended to set up **Mojo** with Python using [Conda environments](https://www.modular.com/blog/using-mojo-with-python) to avoid path and library conflicts that are common when multiple Python interpreters are installed on the same system.

**Link mentioned**: <a href="https://docs.modular.com/mojo/manual/python/">Python integration | Modular Docs</a>: Using Python and Mojo together.

  

---


**Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1234434178922184714)** (153 messages🔥🔥): 

- **Mojo Stirs Up Esolang Creativity**: A member has been inspired to create a parser in Mojo for an esoteric language (eso lang) they devised, similar to BrainF*** but with an improved syntax. They faced an issue with `None` not implementing the `__is__` method, sparking a discussion on the correct use of `None` and optional types in Mojo.
  
- **Mojo Syntax Strikes a Personal Chord**: A member conducted an experiment to combine preferred features from all programming languages they've interacted with and found that the result closely resembled Mojo's syntax. This showcases Mojo's appeal to users with its intuitive design choices.

- **Enthusiasm for New Mojo Developments**: After a hiatus, a member returned to the Mojo community and expressed positive surprise at the new features and the fact that Mojo has gone open source. This contributes to the growing interest and participation in the Mojo project.

- **Interest in Measurement Macros for Mojo**: Drawing on inspiration from Julia's `@time` macro, a member expressed interest in seeing similar functionality in Mojo that would allow for measuring time and resource allocations for code execution. Another member hints at the possibility of such features being added as built-in decorators.

- **Questions on Windows Compatibility**: Queries about Mojo's timeline for Windows availability suggest that community members are eager for cross-platform support. Previous expectations set in October were for "soon," leaving some members anticipating an update on the progress.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.modular.com/mojo/notebooks/Matmul">Matrix multiplication in Mojo | Modular Docs</a>: Learn how to leverage Mojo&#x27;s various functions to write a high-performance matmul.</li><li><a href="https://github.com/search?q=repo%3Amodularml%2Fmojo+%22None%22&type=code&p=0)">Build software better, together</a>: GitHub is where people build software. More than 100 million people use GitHub to discover, fork, and contribute to over 420 million projects.</li><li><a href="https://mojodojo.dev/mojo-team-answers.html#unsafe-code">Mojo Team Answers | Mojo Dojo</a>: no description found</li><li><a href="https://rosettacode.org/wiki/99_Bottles_of_Beer/EsoLang">99 Bottles of Beer/EsoLang</a>: no description found</li><li><a href="https://github.com/karpathy/minbpe">GitHub - karpathy/minbpe: Minimal, clean code for the Byte Pair Encoding (BPE) algorithm commonly used in LLM tokenization.</a>: Minimal, clean code for the Byte Pair Encoding (BPE) algorithm commonly used in LLM tokenization. - karpathy/minbpe</li><li><a href="https://www.youtube.com/watch?v=zduSFxRajkE">Let&#39;s build the GPT Tokenizer</a>: The Tokenizer is a necessary and pervasive component of Large Language Models (LLMs), where it translates between strings and tokens (text chunks). Tokenizer...</li><li><a href="https://youtu.be/kgUXfDpAmGQ?si=VmrPUT7YLBmzMq8I">C++ as an Optimizing Assembler - a Performance Talk - Levo DeLellis - CppNorth 2023</a>: https://www.cppnorth.ca​---C++ as an Optimizing Assembler - a Performance Talk - Levo DeLellis - CppNorth 2023Are you tired of abstractions, templates and co...</li><li><a href="https://github.com/modularml/mojo/issues">Issues · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/modularml/mojo/issues/620">[Feature Request] Native Windows support · Issue #620 · modularml/mojo</a>: Review Mojo&#39;s priorities I have read the roadmap and priorities and I believe this request falls within the priorities. What is your request? native support for windows. when will it be available?...</li><li><a href="https://github.com/modularml/mojo/issues/620#issuecomment-2082106584">[Feature Request] Native Windows support · Issue #620 · modularml/mojo</a>: Review Mojo&#39;s priorities I have read the roadmap and priorities and I believe this request falls within the priorities. What is your request? native support for windows. when will it be available?...
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[community-projects](https://discord.com/channels/1087530497313357884/1151418679578337311/1234494559527108669)** (4 messages): 

- **Mojo Dev Community Springs to Life**: A Mojo-based community project called *用Mojo写一个Mojo社区* has been shared on GitHub. The project can be viewed at [shadowqcom/mojo_dev](https://github.com/shadowqcom/mojo_dev).
- **atol-simd Picks Up Speed**: The [atol-simd project](https://github.com/VMois/mojo-atol-simd) reports a **20% performance increase** over stdlib atol for strings of 15-16 characters, though for shorter strings, stdlib remains slightly faster. Benchmarks are included in the repository.
- **Collaboration Invitation Extended**: A community member expressed interest in contributing to the atol-simd project, inviting opportunities for collaboration.
- **SIMD Projects Share Vectorization Patterns**: In the conversation about SIMD libraries, another project, [mojo-fast-base64](https://github.com/mzaks/mojo-fast-base64), is mentioned, highlighting a common pattern of fallback to scalar processing for inputs unsuitable for vectorization.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/shadowqcom/mojo_dev">GitHub - shadowqcom/mojo_dev: 用Mojo写一个Mojo社区！</a>: 用Mojo写一个Mojo社区！. Contribute to shadowqcom/mojo_dev development by creating an account on GitHub.</li><li><a href="https://github.com/mzaks/mojo-fast-base64">GitHub - mzaks/mojo-fast-base64</a>: Contribute to mzaks/mojo-fast-base64 development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[performance-and-benchmarks](https://discord.com/channels/1087530497313357884/1151418895417233429/1234485565181657214)** (40 messages🔥): 

- **Optimization Quest on Error Correction Coding**: An ongoing discussion centered around performance improvements for a SIMD-based function in the [mocodes GitHub repository](https://github.com/alainrollejr/mocodes). Members exchanged ideas about the potential for LLVM/MLIR optimization techniques and the surprising amount of assembly generated by a seemingly simple function.
- **Benchmarking the almighty Mojo**: A member shared advances in their 1brc (One Billion Row Challenge) project, achieving impressive iteration speeds and offering their code repository for collaboration. The conversation touched on the benefits of using nightly builds versus stable releases in performance testing.
- **Bug Hunting in Nightly Builds**: A member raised an issue where `FileHandle.read_bytes()` was causing memory problems, later recognized as a known issue reported on GitHub.
- **Team Mojo Assemble!**: The idea of forming a "team-mojo" to tackle the 1brc challenge was proposed, aiming to make it both a showcase and a tutorial for the community. This paralleled a suggestion to address benchmarks comparing Mojo to other languages, an effort that had not been fully explored yet.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/MoSafi2/BlazeSeq/blob/main/blazeseq/iostream.mojo">BlazeSeq/blazeseq/iostream.mojo at main · MoSafi2/BlazeSeq</a>: Contribute to MoSafi2/BlazeSeq development by creating an account on GitHub.</li><li><a href="https://github.com/modularml/mojo/discussions/843#discussioncomment-7045479)">The Mojo is 68,000 times faster than Python type blogs are awesome, but can awesome comparisons be made with other languages too? · modularml/mojo · Discussion #843</a>: Mojo being 35,000 times faster than Python, 68,000 times faster than Python… it’s impressive, amazing, and cool, but to non-Python people and anti-Python who haven’t yet paid attention to Mojo yet ...</li><li><a href="https://github.com/alainrollejr/mocodes">GitHub - alainrollejr/mocodes: Error Correction (De)Coding with Mojo</a>: Error Correction (De)Coding with Mojo. Contribute to alainrollejr/mocodes development by creating an account on GitHub.</li><li><a href="https://github.com/MoSafi2/1brc-mojo/tree/dev">GitHub - MoSafi2/1brc-mojo at dev</a>: One Billion Row Challenge (1brc) in Mojo language. Contribute to MoSafi2/1brc-mojo development by creating an account on GitHub.</li><li><a href="https://github.com/modularml/mojo/issues/2051">[stdlib] Do not copy elements when using `FileHandle.read_bytes()` · Issue #2051 · modularml/mojo</a>: I was doing a one-billion row challenge with Mojo and tried reading 1 billion rows (around 13GB file) using read_bytes() and quickly ran out of memory. It does not happen with read(). alias input_f...</li><li><a href="https://github.com/VMois/1brc-mojo">GitHub - VMois/1brc-mojo: One Billion Row Challenge (1brc) in Mojo language</a>: One Billion Row Challenge (1brc) in Mojo language. Contribute to VMois/1brc-mojo development by creating an account on GitHub.</li><li><a href="https://github.com/VMois/mojo-atol-simd">GitHub - VMois/mojo-atol-simd: Converting string to integer in Mojo using SIMD (supports up to 16 chars as of now)</a>: Converting string to integer in Mojo using SIMD (supports up to 16 chars as of now) - VMois/mojo-atol-simd
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[🏎engine](https://discord.com/channels/1087530497313357884/1212827673257316453/1234682806752247818)** (2 messages): 

- **Repo Update Yields Accurate Speed Results**: After pulling the latest update from the repository, a member observed accurate reporting of speed improvements. However, they also noted that their CPU does not reach maximum frequency during benchmarks, and MAX performs better with lower CPU clock speeds when compared to PyTorch and TensorFlow.

- **A Level Up for ModularBot**: ModularBot celebrated as it achieved **level 1**, marking a milestone in its operational use within the Discord environment.
  

---


**Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1234618988965789747)** (51 messages🔥): 

- **EqualityComparable SIMD Discussions**: A [pull request](https://github.com/modularml/mojo/pull/2412) was discussed regarding a change that makes `SIMD` conform to `EqualityComparable` without altering original behavior. However, it may cause issues with existing code where `SIMD` with size greater than 1 is implicitly converted to `Bool`.

- **Explicit over Implicit in SIMD-to-Scalar**: The discussion on `SIMD` highlighted the need for explicit use of `reduce_and` or `reduce_or` for converting from `SIMD` to `Scalar`. It was argued that `SIMD.__bool__()` causing bugs and confusion due to its current implementation.

- **Mojo Compiler Nightly Release Alert**: A new nightly Mojo compiler release was announced, encouraging users to update with `modular update nightly/mojo`. The changes can be reviewed via the [diff on GitHub](https://github.com/modularml/mojo/pull/2449/files) and the [changelog](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md).

- **Debating SIMD and Boolean Conversions**: There was a debate about the appropriate behavior of `bool(SIMD[type, size])`, whether it should return `SIMD[bool, size]` or maintain a scalar boolean representation. Some believe it's important to maintain the ability to use `bool` as a logical interface, potentially impacting operations like `if` and ternary expressions.

- **Source Location Function Moved in Nightly Release**: Discussion about `__source_location()` revealed it might have been replaced with `__call_location()` in the nightly release. After some back and forth, [example usage](https://github.com/modularml/mojo/blob/nightly/stdlib/src/testing/testing.mojo) was shared to clarify how to import and utilize the function in the new compiler version.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://sourcegraph.com/search?q=context:global+__source_location()&patternType=keyword&sm=0&filters=%5B%5B%22type%22,%22Code%22,%22type:file%22%5D%5D">context:global __source_… - Sourcegraph</a>: no description found</li><li><a href="https://github.com/modularml/mojo/blob/nightly/stdlib/src/testing/testing.mojo">mojo/stdlib/src/testing/testing.mojo at nightly · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/modularml/mojo/pull/2412">[stdlib] SIMD conformance to EqualityComparable by helehex · Pull Request #2412 · modularml/mojo</a>: This allows SIMD to conform to EqualityComparable, without losing any of the original behavior. It uses the 4th overload resolution rule to give the new methods lower precedence, while still confor...</li><li><a href="https://github.com/modularml/mojo/pull/2449/files">[stdlib] Update stdlib corresponding to 2024-04-29 nightly/mojo by JoeLoser · Pull Request #2449 · modularml/mojo</a>: This updates the stdlib with the internal commits corresponding to today&#39;s nightly release: mojo 2024.4.2923.</li><li><a href="https://github.com/modularml/mojo/blob/nightly/docs/changelog.md">mojo/docs/changelog.md at nightly · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



**HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1234762736504672346)** (2 messages): 

- **CVPR 2023 Announces Competitions with Big Prizes**: Three new competitions are announced for the CVPR 2023 conference on HF competitions: [SnakeCLEF](https://huggingface.co/spaces/BVRA/SnakeCLEF2024), [FungiCLEF](https://huggingface.co/spaces/BVRA/PlantCLEF2024), and [PlantCLEF](https://huggingface.co/spaces/BVRA/PlantCLEF2024), with over 120k USD in total prizes. The events will run from June 17-21, 2024.
- **100th Edition of Hugging News**: Celebrating the 100th issue of *Hugging News*, featuring the release of **Transformers v4.40.0**, **Gradio 4.28.0**, **Datasets v2.19.0**, **Optimum v1.19.0**, and multiple community interaction updates including the ability to mention people on HuggingFace. Notable highlights include [Phi-3 running in the browser](https://x.com/fleetwood___/status/1783195985893863578) and [Common Voice 17 available on the Hub](https://x.com/reach_vb/status/1785039538185703909).
- **Run AutoTrain UI on Kaggle**: In a shared notebook, users are shown how they can run AutoTrain UI on Kaggle Notebooks backend, further enhancing accessibility for machine learning projects. The guide is available for copy and use at [this Kaggle notebook](https://www.kaggle.com/code/abhishek/autotrain-ui).
- **Snowflake Launches Massive MoE Model**: Snowflake has released a new [408B parameter Dense + Hybrid MoE model](https://x.com/reach_vb/status/1783129119435210836), boasting a 4K context window and fully Apache 2.0 licensed, generating buzz for its impressive performance on complex tasks.
- **Community Growth and Product Announcements**: The announcements highlight the formation of a new [community for journalists](https://x.com/BrigitteTousi/status/1783573043815596426) on the HuggingFace Hub, and the integration of community-driven content like how to use **custom pipelines in Diffusers** and a call for participation in an **ML paper reading group**.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/fleetwood___/status/1783195985893863578)">Tweet from Fleetwood (@fleetwood___)</a>: 🚨 Phi-3 running in the browser 🚨  Hits about 20 tok/s 🏎️ Literally 3 lines of JS.  Still some kinks to iron out, coming to Ratchet 0.4.0 soon.</li><li><a href="https://x.com/abhi1thakur/status/1785279012232736991)">Tweet from abhishek (@abhi1thakur)</a>: Can I run AutoTrain UI on Kaggle? Yes, you can!!! Check out my latest notebook, copy it, fill in your tokens and enjoy AutoTrain UI running on Kaggle Notebooks backend 🚀 Link to notebook: https://www...</li><li><a href="https://x.com/reach_vb/status/1785039538185703909)!">Tweet from Vaibhav (VB) Srivastav (@reach_vb)</a>: Let&#39;s go!! Common Voice 17 - now on the Hub! 🔥  With 31,000 hours of audio (& transcriptions) across 124 languages.  *sound on 🎶*  847 hours of data were added in CV 17, along with 493 hours of ...</li><li><a href="https://x.com/BrigitteTousi/status/1783573043815596426):">Tweet from Brigitte 🤗 (@BrigitteTousi)</a>: 🔊Calling all journalists! With @fdaudens, we&#39;re excited to announce a new community on the @huggingface Hub: Journalists on Hugging Face. 📰🤗  https://huggingface.co/JournalistsonHF 1/</li><li><a href="https://x.com/reach_vb/status/1783129119435210836)">Tweet from Vaibhav (VB) Srivastav (@reach_vb)</a>: Snowflake dropped a 408B Dense + Hybrid MoE 🔥  &gt; 17B active parameters &gt; 128 experts &gt; trained on 3.5T tokens &gt; uses top-2 gating &gt; fully apache 2.0 licensed (along with data recipe to...</li><li><a href="https://x.com/RisingSayak/status/1785162074844197174)">Tweet from Sayak Paul (@RisingSayak)</a>: Custom pipelines and components in Diffusers 🎸  Wanted to use customized pipelines and other components (schedulers, unets, text encoders, etc.) in Diffusers?  Found it inflexible?   This 🧶 is for y...</li><li><a href="https://x.com/lunarflu1/status/1785359306847666431)">Tweet from lunarflu (@lunarflu1)</a>: You can now mention people on @huggingface !
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1234451724257984574)** (208 messages🔥🔥): 

- **Seeking LLM Observability Tools**: A member requested advice on LLM observability tools, particularly interested in something compatible with LlamaIndex and favoring a self-hosted open-source option.
- **API Interaction Assistance with huggingchat**: An individual sought help for communicating with [Hugging Face Chat](https://huggingface.co/chat/) via API calls, expressing a need for guidance.
- **Offering Bounty for Gradio Expertise**: A member expressed frustration over Gradio issues, offering a $200 bounty for quality assistance, with subsequent guidance to seek help in a Gradio-specific channel.
- **Pinball AI Vision Model Discussion**: A detailed conversation unfolded around developing an AI model to identify pinball games and scores, with discussions on complexity, tools, the necessity of image classification, and the feasibility of reusing existing models like llava for part of the solution.
- **Computer Configuration for LLMs**: A user looked for resources on DDR5 and CPUs performances specific to LLMs, considering a high-spec setup for their new computer. Other members chimed in with recommendations and personal experiences related to hardware choices for AI work.
- **Zero GPU Explorer's Membership Queries and Jokes**: Chats indicated confusion over the Zero GPU Explorers membership and subscription status, along with members humorously attempting to "rizz up" the Hugging Face developers using AI-related pick-up lines.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://apply.workable.com/huggingface/?lng=en">Hugging Face</a>: Here at Hugging Face, we’re on a journey to advance and democratize ML for everyone. Along the way, we contribute to the development of technology for the better.</li><li><a href="https://x.com/noaroggendorff/status/1785095305408422234">Tweet from Noa Roggendorff (@noaroggendorff)</a>: iykyk</li><li><a href="https://huggingface.co/spaces/zero-gpu-explorers/README/discussions/26">zero-gpu-explorers/README · The invited application has been waiting. How long does it take to be approved?</a>: no description found</li><li><a href="https://huggingface.co/amazon/chronos-t5-small">amazon/chronos-t5-small · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/gradientai/Llama-3-8B-Instruct-Gradient-1048k">gradientai/Llama-3-8B-Instruct-Gradient-1048k · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/docs/transformers/en/tasks/image_classification">Image classification</a>: no description found</li><li><a href="https://huggingface.co/spaces/zero-gpu-explorers/README/discussions/25">zero-gpu-explorers/README · Update README.md</a>: no description found</li><li><a href="https://youtu.be/u5Vcrwpzoz8?si=U30s6BAN9Jsaec-P">&quot;I want Llama3 to perform 10x with my private knowledge&quot; - Local Agentic RAG w/ llama3</a>: Advanced RAG 101 - build agentic RAG with llama3Get free HubSpot report of how AI is redefining startup GTM strategy: https://clickhubspot.com/4hx🔗 Links- F...</li><li><a href="https://github.com/amazon-science/chronos-forecasting?tab=readme-ov-file">GitHub - amazon-science/chronos-forecasting: Chronos: Pretrained (Language) Models for Probabilistic Time Series Forecasting</a>: Chronos: Pretrained (Language) Models for Probabilistic Time Series Forecasting - amazon-science/chronos-forecasting</li><li><a href="https://huggingface.co/blog/personal-copilot">Personal Copilot: Train Your Own Coding Assistant</a>: no description found</li><li><a href="https://github.com/pacman100/LLM-Workshop/blob/main/personal_copilot/training/train.py">LLM-Workshop/personal_copilot/training/train.py at main · pacman100/LLM-Workshop</a>: LLM Workshop by Sourab Mangrulkar. Contribute to pacman100/LLM-Workshop development by creating an account on GitHub.</li><li><a href="https://huggingface.co/docs/trl/v0.8.6/en/sft_trainer#trl.trainer.ConstantLengthDataset">Supervised Fine-tuning Trainer</a>: no description found
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1234517512213889147)** (2 messages): 

- **Enthusiasm for Learning**: A member expressed excitement about sharing and receiving information in the channel, signaling a positive and collaborative learning environment.
- **Seeking Finetuning Guidance**: A query was raised about the best practices for creating an instruction dataset for finetuning Large Language Models (LLMs), indicating an interest in tailored dataset preparation for model enhancement.
  

---


**HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1234513287299731578)** (9 messages🔥): 

- **Deep Dive Into Deep Learning**: The MIT Introduction to Deep Learning course, now updated for 2024, provides a foundational understanding of deep learning concepts. The [lecture video](https://www.youtube.com/watch?v=ErnWZxJovaM&list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI&index=2) is available on YouTube for anyone interested in the field.
  
- **Evaluation of Text-to-Image Models**: There’s an upcoming talk on text-to-image model evaluation, where the speaker will discuss text-to-image alignment and model robustness.

- **Stallman Sings of Freedom**: A YouTube video features Richard Stallman singing the "Free Software Song" during an event in Ecuador. This peculiar moment can be found [here](https://www.youtube.com/watch?v=9sJUDx7iEJw).

- **Community Computer Vision Course Launch**: Hugging Face has launched a community-driven course on computer vision accessible for everyone, including how to join the learner community, make submissions, and certification information. Start learning with their [welcome page](https://huggingface.co/learn/computer-vision-course/unit0/welcome/welcome).

- **AI Safety Benchmarks Gain Focus**: A LinkedIn post announces the LLM Safety LeaderBoard, a new platform measuring AI safety, security, and responsible AI practices. Find out more about the leaderboard [here](https://www.linkedin.com/posts/divyanshuusingh_safetyleaderboard-aisecurity-responsibleai-activity-7190907558071558145-qeVK).

- **Discover 5 AI Tools through GenAI**: A Medium piece titled "GenAI Adventures: 5 Interesting AI Tools Everyone Should Try" presents a curated list of AI Tools. Readers can explore these tools on [Medium](https://medium.com/illumination/genai-adventures-5-interesting-ai-tools-everyone-should-try-44ae8f8115af).

- **Constructing Intuitive RAG Applications**: An article guides the creation of webloader RAG applications using Groq, Langchain, and Datastax featuring powerful capabilities. Interested readers can delve into these integrations on [Medium](https://medium.com/ai-advances/building-powerful-webloader-rag-applications-with-groq-langchain-and-datastax-f4816d88bee8).

- **Simplifying Database Queries with Machine Learning**: An innovative approach is being developed to allow querying of a "people database" with minimal SQL knowledge using RAG and Gemini. More details on the project can be found at [Datai Alliance](https://www.dataialliance.org).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/learn/computer-vision-course/unit0/welcome/welcome">Welcome to the Community Computer Vision Course - Hugging Face Community Computer Vision Course</a>: no description found</li><li><a href="https://www.dataialliance.org">blog</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=9sJUDx7iEJw">Richard Stallman Free software Song</a>: Richard Stallman en Ecuador, cantando el temita, del free software, grabado por Julian Coccia.</li><li><a href="https://www.youtube.com/watch?v=ErnWZxJovaM&list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI&index=2">MIT Introduction to Deep Learning | 6.S191</a>: MIT Introduction to Deep Learning 6.S191: Lecture 1*New 2024 Edition*Foundations of Deep LearningLecturer: Alexander AminiFor all lectures, slides, and lab m...
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1234430834967318530)** (13 messages🔥): 

- **Model Release Dilemma**: A post mentioned a dilemma involving the selection of one among five models to release, including an invitation for input or preference regarding which model should be launched next, and provided a [LinkedIn post link](https://www.linkedin.com/posts/bineric_llm-ai-europe-activity-7190590676055506944-QW9f) for more context.
- **Greetings from LifePal**: A new AI-powered app named LifePal was introduced, which serves as a personalized guide to a well-balanced life and claims seamless integration with Apple Vision Pro. It's described as a life co-pilot and its perceivable benefits and features were showcased along with the [Apple Store link](https://apps.apple.com/se/app/lifepal-ai-chat-assistant/id6471972439).
- **ChatGPT's Norwegian Needs Work**: A member highlighted the subpar performance of ChatGPT's Norwegian translations, which necessitated reprocessing through a Retriever-Augmented Generator (RAG) with local slang, complemented by a mention of an alternative, [NorskGPT-Mistral](https://huggingface.co/bineric/NorskGPT-Mistral-7b), designed for Norwegian language understanding and generation.
- **Seeking Beta Testers for an Advanced Research Assistant and Search Engine**: An offer was made to recruit beta testers for an advanced research assistant and search engine tool, providing 2 months free of premium service with various models including GPT-4 Turbo, Mistral Large and more. Interested parties were directed to [Rubik's AI](https://rubiks.ai) with a promo code for the free premium offer.
- **Innovative Inpainting SDXL on Hugging Face**: A unique take on the inpainting tool named SDXL, allowing iterative inpainting on top of previous generations with version history, was shared. Feedback and example sharing were encouraged, and the [inpainting tool can be found on Hugging Face](https://huggingface.co/spaces/tonyassi/inpainting-sdxl-sketch-pad).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/tonyassi/inpainting-sdxl-sketch-pad">Inpainting SDXL Sketch Pad - a Hugging Face Space by tonyassi</a>: no description found</li><li><a href="https://huggingface.co/bineric/NorskGPT-Mistral-7b">bineric/NorskGPT-Mistral-7b · Hugging Face</a>: no description found</li><li><a href="https://apps.apple.com/se/app/lifepal-ai-chat-assistant/id6471972439">‎LifePal AI Chat &amp; Assistant</a>: ‎Discover LifePal: your productivity AI companion.  Are you ready to unlock your full potential and live a healthier, happier life? LifePal is here to guide you on your journey to becoming a better yo...</li><li><a href="https://github.com/Lama-West/PnPR-GCN_ACM_SAC_24/tree/main">GitHub - Lama-West/PnPR-GCN_ACM_SAC_24</a>: Contribute to Lama-West/PnPR-GCN_ACM_SAC_24 development by creating an account on GitHub.</li><li><a href="https://vimeo.com/940824094?share=copy">Vinner - Nybygg i og rundt Bergen</a>: Stor takk til Sn&oslash;hetta</li><li><a href="https://github.com/GDSC-FSC/gemini-node-1">GitHub - GDSC-FSC/gemini-node-1</a>: Contribute to GDSC-FSC/gemini-node-1 development by creating an account on GitHub.</li><li><a href="https://rubiks.ai">Rubik's AI - AI research assistant & Search Engine</a>: no description found
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1234684966013767731)** (12 messages🔥): 

- **Graphs and LLMs Reading Preparation**: A member announces plans to review [papers on large language models (LLMs) and their interaction with graphs](https://arxiv.org/abs/2404.14928), focusing on complex relationship representation and discussing the potential for a presentation the following Saturday.
- **Additional Paper Surveys for Saturday's Session**: The same member additionally considers reviewing two survey papers, one about [LLMs applied to graphs](https://arxiv.org/abs/2312.02783), and another on [foundation models](https://arxiv.org/abs/2310.11829), suggesting these topics may also be included but noting the need to avoid spreading too thin for future reading groups.
- **Exploring Distillation of Score-Based Models**: A chat participant inquires about resources on distilling score-based models, specifically models that reduce the number of generation steps required compared to classical SDE solver models.
- **Guidance on Distillation Papers and Communities**: A response is offered guiding the previous inquiry to the Laion and Eleuther servers where experts on model distillation congregate and suggesting leading researcher Gothos, with a mention of relevant papers in the fields of rectified flow and LCM Lora.
- **Paper Reading Event Creation**: An event is tentatively scheduled in the group, allowing for discussions on time adjustment, encouraging member participation in the upcoming reading and presentation on LLMs and graph interaction.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2404.14928">Graph Machine Learning in the Era of Large Language Models (LLMs)</a>: Graphs play an important role in representing complex relationships in various domains like social networks, knowledge graphs, and molecular discovery. With the advent of deep learning, Graph Neural N...</li><li><a href="https://discord.gg/hugging-face-879548962464493619?event=1234913780048203856">Join the Hugging Face Discord Server!</a>: We&#x27;re working to democratize good machine learning 🤗Verify to link your Hub and Discord accounts! | 77552 members</li><li><a href="https://arxiv.org/abs/2312.02783">Large Language Models on Graphs: A Comprehensive Survey</a>: Large language models (LLMs), such as GPT4 and LLaMA, are creating significant advancements in natural language processing, due to their strong text encoding/decoding ability and newly found emergent ...</li><li><a href="https://arxiv.org/abs/2310.11829">Towards Graph Foundation Models: A Survey and Beyond</a>: Foundation models have emerged as critical components in a variety of artificial intelligence applications, and showcase significant success in natural language processing and several other domains. M...
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1234548112270426135)** (15 messages🔥): 

- **Balancing Accuracy and Efficiency**: A member discussed the trade-off between computational efficiency and model accuracy when processing bounding boxes at original resolution. Another member suggested image preprocessing techniques like blurring to optimize VRAM usage.

- **Exploration of Image Segmentation Models**: In seeking guidance for advancing in image segmentation, OneFormer, MaskFormer, Segformer were mentioned as part of the sequence of models a member has worked with.

- **Buddying Up for CNN Studies**: A member expressed interest in finding a study partner for learning and working on Convolutional Neural Networks (CNNs).

- **Historical Contour Algorithms Meet Modern Preprocessing**: Discussing YOLO architectures, a member recommended reviewing pre-YOLO/CNN image segmentation and contour finding algorithms, and mentioned that preprocessing and downsampling can still yield good results. Links to OpenCV documentation on morphological operations and image processing were shared: [Morphological Operations](https://docs.opencv.org/3.4/d9/d61/tutorial_py_morphological_ops.html), [Table of Contents for Image Processing](https://docs.opencv.org/3.4/d2/d96/tutorial_py_table_of_contents_imgproc.html).

- **PyTorch vs TensorFlow for CNN Projects**: Conversations touched upon whether to learn PyTorch or stick with TensorFlow, highlighting PyTorch's momentum in the community and academia, and TensorFlow's robust DevOps support from Google. The flexibility to create projects involving object detection and image segmentation using TensorFlow was reaffirmed.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.opencv.org/3.4/d2/d96/tutorial_py_table_of_contents_imgproc.html">OpenCV: Image Processing in OpenCV</a>: no description found</li><li><a href="https://docs.opencv.org/3.4/d9/d61/tutorial_py_morphological_ops.html">OpenCV: Morphological Transformations</a>: no description found
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1234572716359548999)** (3 messages): 

- **Seeking NLU/NLP Guidance**: A new member is working on a chatbot using the *Rasa framework*, but is facing issues with intent recognition, where a generic sales inquiry is miscategorizing as a company-specific sales intent.
- **Intent on Enhancing Intent Recognition**: They are considering creating a custom NER model to identify specific keywords as intents (sales, purchases, etc.) and using company names from their database as *NER-company* to improve their chatbot's performance.
  

---


**HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1234554096149725184)** (4 messages): 

- **Realism Challenge with Hyper-SD and IP-Adapter**: A user shared an issue with not getting realistic results when using **Hyper-SD** with the **IP-Adapter**. They provided a [discussion link](https://github.com/huggingface/diffusers/discussions/7818) to the GitHub where the issue was elaborated.
- **Surprised by Inconsistent Results Across Models**: A person was perplexed after switching from **Seaart to A1111**, only to find that the color and shadow quality of the images changed despite the same settings and seed being used. They inquired about any backend differences and whether it was possible to achieve uniform results on both models.
- **DeepFloyd's Unpredictable Behavior**: According to a user, **DeepFloyd** exhibits odd patterns when tweaking step count, sampler, and CFG. They compared it to the **Ambigram** research model and provided insights into the performance of different settings, particularly the **DPM++ 2M** scheduler.


**Link mentioned**: <a href="https://github.com/huggingface/diffusers/discussions/7818">Not getting good realistic results with Hyper-SD + IP-Adapter · huggingface/diffusers · Discussion #7818</a>: Hi everyone, (maybe you @asomoza know about this?) Does hyper-sd works well with IP-Adapter? I am testing hyper-sd in Diffusers as explained in the repo. I thought that I was going to get better re...

  

---


**HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1234862689357009087)** (1 messages): 

- **Gradio Share Server Troubles**: Gradio has experienced issues with the Share Server that might affect sharing and usage on Colab. They're actively investigating and resolving the problem; updates are available at their [status page](https://status.gradio.app/).
- **Check Gradio's Health Anytime**: For an overview of Gradio's operational status over the past 90 days, including the last 24 hours, week, and month, refer to their [calendar view](https://status.gradio.app/#).
- **Clear Skies for the Past Week**: There have been no status updates in the last 7 days, indicating no new incidents. Historical status updates can be checked on the [status update history](https://status.gradio.app/#) page.

**Link mentioned**: <a href="https://status.gradio.app/">Gradio Status</a>: no description found

  

---



**OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1234571459699933314)** (3 messages): 

- **OpenRouter Exploring Syrax**: Alex Atallah indicated the start of experimenting with **Syrax** and offered support to the team, proposing to organize a group chat.
- **Collaboration Accepted with Enthusiasm**: Mart02 acknowledged and appreciated the outreach from Alex, signaling the beginning of a collaborative effort by accepting the friend request.
  

---


**OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1234433355626319872)** (240 messages🔥🔥): 

- **Frontend Quest for Non-Technical Deployments**: A member inquired about a multi-user frontend that could be deployed on shared hosting without the need for Docker or Node.js. *LibreChat* was recommended as the most suitable option, but another member mentioned hosting challenges and cost concerns, leading to a suggestion of Vercel's free tier hosting as a potential solution.

- **Comparisons and Anticipation for LLMs**: There was a vigorous discussion about various large language models, including *Llama-3 8B*, *Dolphin 2.9*, and *Mixtral-8x22B*. Users shared insights on model capabilities, such as context window size and the likelihood of models being censored based on their conversation styles and datasets.

- **Model Training Adventures**: A user shared their journey trying to train a model to become more "unhinged" by using their own toxic dataset. Comparisons were made between the behavior of different models, and a discussion on whether LLMs could handle large contexts effectively, with a consensus that while models like *Llama 3 8B* could manage long contexts, their performance might degrade beyond a certain point.

- **Affordable Model Experiments and Discoveries**: Members discussed options for cost-effective yet efficient models available on the OpenRouter platform. *Mixtral-8x7B-Instruct* was highlighted as a reasonable balance between price and performance, with one user expressing surprise at the improved output quality of *GPT-3.5*, resembling more human-like writing.

- **OR Functionality in Fixing Message Order**: There was a query regarding *Claude 3*'s handling of the order of assistant/user messages. It was confirmed that **OpenRouter** automatically corrects ordering to ensure the models work correctly, and users are encouraged to report any ordering issues they might encounter.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://cws-docs.pages.dev/en/">Home | ChatGPT Web Share Docs</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1b6nqC7UZVt8bx">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1b6nqC7UZVt8bx4MksX7s656GXPM-eWw4">Google Colab</a>: no description found</li><li><a href="https://huggingface.co/jondurbin/cinematika-7b-v0.1">jondurbin/cinematika-7b-v0.1 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/lmsys/lmsys-chat-1m">lmsys/lmsys-chat-1m · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/TheBloke/psyonic-cetacean-20B-AWQ">TheBloke/psyonic-cetacean-20B-AWQ · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/maywell/Llama-3-8B-Instruct-1M">maywell/Llama-3-8B-Instruct-1M · Hugging Face</a>: no description found</li><li><a href="https://x.com/erhartford/status/1784315764079796541?s=46&t=2a7uDiV3mox9o-E5jIFbLQ">Tweet from Eric Hartford (@erhartford)</a>: dolphin-2.9-llama3-8b-256k is released. It is dolphin-2.9-llama3-8b with @winglian&#39;s awesome 256k context adapter applied. I will get the model card done today.</li><li><a href="https://huggingface.co/gradientai/Llama-3-8B-Instruct-Gradient-1048k">gradientai/Llama-3-8B-Instruct-Gradient-1048k · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/cognitivecomputations/dolphin-2.9-mixtral-8x22b">cognitivecomputations/dolphin-2.9-mixtral-8x22b · Hugging Face</a>: no description found</li><li><a href="https://rentry.org/GPT2/#main-alternative-theory">gpt2-chatbot</a>: This page is a work in progress. Its conclusions are likely to change as more information is collected. News as of 2023-04-30: gpt2-chatbot is extremely likely to run on a server operated by, or assoc...</li><li><a href="https://www.clay.com/">Clay - Scale personalized outbound</a>: Combine 50+ data providers, real-time scraping, and AI to send 1-1 personalized campaigns that book more meetings.</li><li><a href="https://huggingface.co/datasets/jondurbin/cinematika-v0.1">jondurbin/cinematika-v0.1 · Datasets at Hugging Face</a>: no description found</li><li><a href="https://openrouter.ai/models/openrouter/cinematika-7b">Cinematika 7B (alpha) by openrouter | OpenRouter</a>: This model is under development. Check the [OpenRouter Discord](https://discord.gg/fVyRaUDgxW) for updates.</li><li><a href="https://www.cyon.ch/hosting/managed-server">Managed Server: Dein eigener Server, zuhause in der Schweiz</a>: no description found
</li>
</ul>

</div>
  

---



**LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1234521268070645790)** (4 messages): 

- **Advanced RAG Reference Architecture Revealed**: The **LlamaIndex** team presents a reference architecture for building advanced RAG—Retrieval-Augmented Generation—systems within the **AWS ecosystem**. This resource provides guidance on advanced parsing and agentic reasoning, and it's available through the shared [code repository](https://t.co/sfQOvhHHg5).

- **Hackathon Winners Develop Documentation Bot**: **Team CLAB**, winners of a recent hackathon, crafted a full-stack documentation bot that integrates **LlamaIndex** for parsing and orchestrating, along with **Nomic embeddings**. More details on the project and the hackathon can be found in the linked [blog post](https://t.co/2UMqrHwO56).

- **Creating Financial Assistants with Agentic RAG**: A new development allows for building financial assistants capable of handling complex calculations, such as percentage evolution and **CAGR**, directly over unstructured financial reports. A [recent post](https://t.co/6cTNxUBJcr) explains how this can be done without requiring human data transformation steps.

- **Building Efficient RAG with Semantic Caching**: In collaboration with @Redisinc, @tchutch94, and @seldo showcase how to build high-performance RAG applications that incorporate **semantic caching** to expedite frequently made queries. This innovation is aimed at enhancing quality, efficiency, and cost-effectiveness as discussed in the [collaboration piece](https://t.co/oGxFrZLMRn).

**Link mentioned**: <a href="https://t.co/oGxFrZLMRn">no title found</a>: no description found

  

---


**LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1234440203788222516)** (159 messages🔥🔥): 

- **Anticipation for Assistant Agent V2**: Members are inquiring about an update or release of **LlamaIndex OpenAI Assistant Agent V2** to take advantage of features in the new OpenAI Assistant V2. Currently, there is no specific update or pull request for this version.
  
- **Updating Pinecone Indices Query**: Instructions for updating an index part in Pinecone are not well-documented. While members suggested using methods like `pinecone_index.update`, no direct examples with `SimpleDirectoryReader` were provided in the LlamaIndex knowledge base.

- **Tool Preference for LLM Observability**: There’s a discussion on the best LLM observability tools between **Arize Phoenix** and **Langfuze**. A member suggested that both tools provide detailed insights, but no clear preference was indicated.

- **LlamaIndex YouTube Resources**: Users sought recordings of the LlamaIndex Webinar, and one member suggested checking the **[LlamaIndex YouTube channel](https://www.youtube.com/@LlamaIndex)**, as well as other platforms like X space and LinkedIn for the latest webinars.

- **Async Calls with AzureOpenAI**: A member posed a question regarding **async calls with AzureOpenAI** in LlamaIndex and received instructions for using `acomplete`, `astream_complete`, `achat`, and `astream_chat` async methods. The benefits of using async methods, such as speed improvements from parallel execution and non-blocking tasks, were highlighted.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://imgur.com/a/9uLmSxD">Summary and Resources</a>: Discover the magic of the internet at Imgur, a community powered entertainment destination. Lift your spirits with funny jokes, trending memes, entertaining gifs, inspiring stories, viral videos, and ...</li><li><a href="https://www.youtube.com/@LlamaIndex">LlamaIndex</a>: Official YouTube Channel for LlamaIndex - the data framework for your LLM applications </li><li><a href="https://docs.llamaindex.ai/en/latest/examples/vector_stores/TypesenseDemo#query-index>).">Typesense Vector Store - LlamaIndex</a>: no description found</li><li><a href="https://youtu.be/u5Vcrwpzoz8?si=U30s6BAN9Jsaec-P">&quot;I want Llama3 to perform 10x with my private knowledge&quot; - Local Agentic RAG w/ llama3</a>: Advanced RAG 101 - build agentic RAG with llama3Get free HubSpot report of how AI is redefining startup GTM strategy: https://clickhubspot.com/4hx🔗 Links- F...</li><li><a href="https://docs.llamaindex.ai/en/latest/getting_started/customization#i-want-to-retrieve-more-context-when-i-query>).">Frequently Asked Questions (FAQ) - LlamaIndex</a>: no description found</li><li><a href="https://github.com/zby/answerbot/blob/main/answerbot/replay_client.py">answerbot/answerbot/replay_client.py at main · zby/answerbot</a>: answering questions using LLMs, search (RAG) and other tools - example code - zby/answerbot</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/output_parsing/function_program/">Function Calling Program for Structured Extraction - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/module_guides/querying/retriever#get-started>).">Retriever - LlamaIndex</a>: no description found</li><li><a href="https://github.com/zby/LLMEasyTools">GitHub - zby/LLMEasyTools: Tools for LLM agents.</a>: Tools for LLM agents. Contribute to zby/LLMEasyTools development by creating an account on GitHub.</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/llm/openai#async>).">OpenAI - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/api_reference/tools/metaphor#llama_index.tools.metaphor.MetaphorToolSpec.retrieve_documents>):">Metaphor - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/retrievers/vectara_auto_retriever#running-over-some-sample-data>).">Auto-Retrieval from a Vectara Index - LlamaIndex</a>: no description found</li><li><a href="https://github.com/run-llama/llamabot">GitHub - run-llama/llamabot</a>: Contribute to run-llama/llamabot development by creating an account on GitHub.</li><li><a href="https://docs.llamaindex.ai/en/latest/api_reference/chat_engines/context#llama_index.core.chat_engine.ContextChatEngine>)">Context - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/pipeline/query_pipeline_async#query-pipeline-with-asyncparallel-execution>),">Query Pipeline with Async/Parallel Execution - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/pipeline/query_pipeline_async#try-out-queries>).">Query Pipeline with Async/Parallel Execution - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/ingestion/parallel_execution_ingestion_pipeline#in-summary>),">Parallelizing Ingestion Pipeline - LlamaIndex</a>: no description found
</li>
</ul>

</div>
  

---


**LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1234543867987230760)** (1 messages): 

- **A Look Back at GPT-1**: A member shared a [blog post](https://amgadhasan.substack.com/p/revisiting-gpt-1-the-spark-that-ignited-llms) exploring the original GPT-1 model from OpenAI, highlighting its enduring influence on current LLMs like Mistral-7B. The post dives into GPT-1's architecture, including **positional embeddings and Conv1D usage**, and shows a screenshot of Alec Radford's tweet about this groundbreaking NLP technique.

**Link mentioned**: <a href="https://amgadhasan.substack.com/p/revisiting-gpt-1-the-spark-that-ignited-llms">Revisiting GPT-1: The spark that ignited the fire of LLMs</a>: A Comprehensive Look at GPT-1&#x27;s Contribution to the Development of Modern LLMs

  

---



**Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1234434408405139507)** (25 messages🔥): 

- **Searching for Community Projects Seeking Volunteers**: A member inquired about resources to find community projects in need of volunteers, particularly those that offer a compute budget due to the member's lack of personal GPU resources.

- **Understanding Orthogonal Keys in AI**: A nuanced explanation was provided for a process termed "clear-ing" in the context of AI keys and states, using the example of orthogonal keys and how they behave in equations to explain memory updating in models.

- **Intricacies of Infini-Attention and Compressive Memory**: A dialogue took place around the concept of infini-attention and its perceived overhype, with a reference to a delta rule in compressive memory from 2021 and skepticism about its testing thus far. The discussion included a request for and provision of a relevant research paper.

- **Performance Comparison Puzzles the Community**: Members engaged in discussions on the reasons behind the slower performance of *mixtral 8x22B* as compared to *llama 3 70B* on fireworks.ai, touching on aspects like batching, utilization, and speeds in relation to MoEs and *mixtral* having more parameters but fewer layers.

- **Invitation to Stanford CS25 Transformers Social Event**: An announcement for a **Stanford CS25** Transformers social event at EVGR Pub & Beer Garden was made, giving details on the event, a call for RSVPs, and information about a related talk on campus. An invitation was extended to the Discord community to attend the in-person talk about Transformers or join via Zoom, with links provided to the RSVP form and event details.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="http://kolinko.github.io/effort/">Effort Engine</a>: A possibly new algorithm for LLM Inference. Adjust smoothly - and in real time - how many calculations you'd like to do during inference.</li><li><a href="https://arxiv.org/abs/2102.11174">Linear Transformers Are Secretly Fast Weight Programmers</a>: We show the formal equivalence of linearised self-attention mechanisms and fast weight controllers from the early &#39;90s, where a ``slow&#34; neural net learns by gradient descent to program the ``f...</li><li><a href="https://cs25.stanford.edu)">no title found</a>: no description found</li><li><a href="https://stanford.zoom.us/j/99922151759?pwd=dW5CcUtVYkNybGZGY0hMWUZtVkZBZz09).">Join our Cloud HD Video Meeting</a>: Zoom is the leader in modern enterprise video communications, with an easy, reliable cloud platform for video and audio conferencing, chat, and webinars across mobile, desktop, and room systems. Zoom ...</li><li><a href="https://discord.gg/2vE7gbsjzA)">Discord | Your Place to Talk and Hang Out</a>: Discord is the easiest way to talk over voice, video, and text. Talk, chat, hang out, and stay close with your friends and communities.</li><li><a href="https://www.reddit.com/user/No_Dragonfruit_5472/comments/1cef7gc/tradingview_premium_pack_crack_2024/">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1234512372647989329)** (105 messages🔥🔥): 

- **Long Context Challenge Addressed**: The [Information-intensive (IN2) training proposal](https://arxiv.org/abs/2404.16811) aims to improve Large Language Model's (LLM's) use of lengthy contexts. It involves a synthetic dataset requiring models to integrate information from various segments in long texts to overcome the "lost-in-the-middle" issue.
  
- **Emergent Abilities Linked to Pretraining Loss**: A [Twitter post](https://x.com/_jasonwei/status/1784990066609414556?s=46&t=OICM4zGqs0OOATmLPoNFyw) discusses findings that emergent abilities in models can be correlated with pretraining loss. Unlike compute, pretraining loss can better reflect model performance by considering dataset quality and architectural factors.

- **Dissecting Model Biases**: A discussion highlighted the difficulty of tracing specific biases, like a number preference, back to changes in model weights. As biases may arise during continual training, members note the potential need to implement tools to analyze these shifts for verification.

- **Debating LLMs as Black Boxes**: Conversations revolved around whether LLMs should be considered black boxes, given our limited understanding of their internal mechanisms. It was argued that, while we understand some aspects of LLMs, their reasoning cannot be trusted as explanations are post-hoc and may not reflect true internal processes.

- **Data Leakage Detection in LLMs**: A message links to a paper introducing a detection pipeline to identify potential data leakage in LLM benchmarks, highlighting issues with training and test set misuse ([PDF](https://arxiv.org/pdf/2404.18824)). The findings aim to foster fair comparisons and healthier development in the AI field.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://videogigagan.github.io/">VideoGigaGAN</a>: no description found</li><li><a href="https://arxiv.org/abs/2404.14662">NExT: Teaching Large Language Models to Reason about Code Execution</a>: A fundamental skill among human developers is the ability to understand and reason about program execution. As an example, a programmer can mentally simulate code execution in natural language to debu...</li><li><a href="https://arxiv.org/abs/2404.16811">Make Your LLM Fully Utilize the Context</a>: While many contemporary large language models (LLMs) can process lengthy input, they still struggle to fully utilize information within the long context, known as the lost-in-the-middle challenge. We ...</li><li><a href="https://x.com/_jasonwei/status/1784990066609414556?s=46&t=OICM4zGqs0OOATmLPoNFyw">Tweet from Jason Wei (@_jasonwei)</a>: Enjoyed this paper that plots emergent abilities with pretraining loss on the x-axis, which is actually a suggestion that @OriolVinyalsML also made a few years back: https://arxiv.org/abs/2403.15796  ...</li><li><a href="http://arxiv.org/abs/2404.18824">Benchmarking Benchmark Leakage in Large Language Models</a>: Amid the expanding use of pre-training data, the phenomenon of benchmark dataset leakage has become increasingly prominent, exacerbated by opaque training processes and the often undisclosed inclusion...</li><li><a href="https://arxiv.org/abs/2403.18506">Faster Convergence for Transformer Fine-tuning with Line Search Methods</a>: Recent works have shown that line search methods greatly increase performance of traditional stochastic gradient descent methods on a variety of datasets and architectures [1], [2]. In this work we su...</li><li><a href="https://arxiv.org/abs/2404.12388">VideoGigaGAN: Towards Detail-rich Video Super-Resolution</a>: Video super-resolution (VSR) approaches have shown impressive temporal consistency in upsampled videos. However, these approaches tend to generate blurrier results than their image counterparts as the...</li><li><a href="https://arxiv.org/abs/2404.16717">Embracing Diversity: Interpretable Zero-shot classification beyond one vector per class</a>: Vision-language models enable open-world classification of objects without the need for any retraining. While this zero-shot paradigm marks a significant advance, even today&#39;s best models exhibit ...</li><li><a href="https://www.biorxiv.org/content/10.1101/2024.04.28.591528v1">Sequential predictive learning is a unifying theory for hippocampal representation and replay</a>: The mammalian hippocampus contains a cognitive map that represents an animal's position in the environment and generates offline &quot;replay&quot; for the purposes of recall, planning, and forming lo...
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1234570912951697500)** (3 messages): 

- **Custom Function for Distinct Prompts**: A member discussed the possibility of passing distinct prompts based on a model in a single task, suggesting the use of a custom `!function` for implementation.
- **BitsAndBytes Oddity with 8bit**: One user observed that using **BitsAndBytes 4bit** encoding worked well with **llama3-70b**, but switching to **8bit** encoding yielded poor results, describing the output as "absolute garbage".
- **8bit Encoding Issue with llama3-8b**: The same member noted a similar issue when using **8bit** encoding on **llama3-8b**, indicating consistent problems with 8bit across different models.
  

---



**LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1234495993056329738)** (113 messages🔥🔥): 

- **AI Birthday Bungle Sparks GDPR War**: An EU privacy activist has filed a [GDPR complaint](https://www.politico.eu/article/chatgpts-hallucinations-get-eu-privacy-complaint/) against AI models after a model incorrectly guessed his birthday. He argues this error could potentially lead to the banning of AI models in the EU.
- **New GPT Surprise Rumors Circulate**: Discussion revolves around an alleged stealth release of a GPT-5 model, with speculation based on performance and refusal to hallucinate in tests, although confusion abounds due to no official leaderboard inclusion and contradictory test responses.
- **Performance Queries for Llama3 70B**: Concerns were raised about the seemingly low token generation speed of 13 tokens per second on a dual 3090 setup for a [Llama3 70B model](https://rentry.co/GPT2), leading to discussions on potential hardware optimizations and model configuration tweaks.
- **Exllama: The Underrated Speedster**: Users discuss the performance superiority of exllama over other libraries for LLM tasks, recommending the use of [TabbyAPI](https://dct.openempathic.ai/) repo for easier setups.
- **Debates Over LMSYS’s Leaderboard Transparency**: Members express doubts about the objectivity of LMSYS's leaderboard, raising concerns about potential conflicts of interest between scientific evaluation and commercial enterprises, as well as calling for more transparency and the ability to filter by open weights.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lmsys.org/blog/2024-03-01-policy/">LMSYS Chatbot Arena: Live and Community-Driven LLM Evaluation | LMSYS Org</a>: &lt;h2&gt;&lt;a id=&quot;our-mission&quot; class=&quot;anchor&quot; href=&quot;#our-mission&quot; aria-hidden=&quot;true&quot;&gt;&lt;svg aria-hidden=&quot;true&quot; class=&quot;octicon octicon-link&...</li><li><a href="https://www.politico.eu/article/chatgpts-hallucinations-get-eu-privacy-complaint/">ChatGPT&#8217;s hallucinations draw EU privacy complaint</a>: Activist demands regulators launch probe over ChatGPT&#8217;s wild guess on his date of birth.</li><li><a href="https://huggingface.co/datasets/lmsys/lmsys-chat-1m">lmsys/lmsys-chat-1m · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


**LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1234583301562437682)** (12 messages🔥): 

- **OpenCLIP Fine-Tuned for Cardiac Ultrasound**: A member shared the publication of their research on fine-tuning OpenCLIP for cardiac ultrasound, [available here](https://doi.org/10.1038/s41591-024-02959-y). Despite numerous challenges and an extensive revision process, they expressed relief at its completion.
- **Echoes of Exhaustion**: The member also conveyed their readiness to move beyond the demanding project, humorously noting the *scuffed* zero-shot techniques used and their lack of familiarity with the multimodal AI world at the project's onset.
- **Stable Diffusion Community Reopens**: A link to a GitHub repository for training CLIP separately from U-Net was shared alongside news of /r/StableDiffusion reopening after protesting Reddit's open API changes. Additional details and a discussion forum can be found at [this Reddit post](https://www.reddit.com/r/StableDiffusion/comments/1cgyjvt/github_zer0intclipfinetune_or_sdxl_training_the/).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://doi.org/10.1038/s41591-024-02959-y">Vision–language foundation model for echocardiogram interpretation - Nature Medicine</a>: A vision&#8211;language foundation model, trained on a dataset of more than 1 million echocardiogram video&#8211;text pairs, is able to assess various cardiac structural and functional parameters desp...</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1cgyjvt/github_zer0intclipfinetune_or_sdxl_training_the/">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---



**OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1234551748413358170)** (2 messages): 

- **ChatGPT Plus Integrates Memory Feature**: **Memory** is now available to all ChatGPT Plus users, allowing them to tell ChatGPT what to remember by starting a new chat. This feature can be enabled or disabled in settings and is yet to roll out in Europe or Korea.

- **Enhanced Data Control for Users**: ChatGPT Free and Plus users can now access their chat history even if they have opted out of contributing data for model improvement. Additionally, a new **Temporary Chat** feature allows for conversations that won't be saved in the user's chat history.
  

---


**OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1234440550707630160)** (81 messages🔥🔥): 

- **Exploring AI Curiosity and Sentience**: A user detailed their curiosity test involving ChatGPT handling a zip file with a maze. Some discussion followed on how to measure AI's potential for curiosity and its relation to sentience, but consensus on these concepts remains elusive.
- **DragGAN Sparks Interest**: A member discovered DragGAN, a tool that manipulates photos to change angles and poses, fueling a discussion about AI's ability to recreate images from new perspectives without full models.
- **Llama-3 8B Extends Context Capability**: An interesting reveal occurred with Llama-3 8B Instruct Gradient-1048k, showing how state-of-the-art language models can operate on long-context information; the model is available at [Hugging Face](https://huggingface.co/gradientai/Llama-3-8B-Instruct-Gradient-1048k).
- **Debating the Accessibility of Advanced AI Tools**: Discussions surfaced about OpenAI's policy on free access to new features like DALL-E, with some users questioning why more advanced tools aren't also free and pondering the potential for OpenAI to provide a student discount.
- **Potential Collaboration Between LLMs**: One user inquired about the possibility of having two language models like ChatGPT and Claude Opus collaborate on writing a paper, provoking suggestions about using third-party services to manage multi-model interactions.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/gradientai/Llama-3-8B-Instruct-Gradient-1048k">gradientai/Llama-3-8B-Instruct-Gradient-1048k · Hugging Face</a>: no description found</li><li><a href="https://dontasktoask.com/">Don't ask to ask, just ask</a>: no description found</li><li><a href="https://vcai.mpi-inf.mpg.de/projects/DragGAN/">Drag Your GAN: Interactive Point-based Manipulation on the Generative Image Manifold</a>: no description found
</li>
</ul>

</div>
  

---


**OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1234492334725533807)** (11 messages🔥): 

- **Size Matters in Model Performance**: A comparison is highlighted between **GPT-4** and its predecessor, with *GPT-4* identified as **"much larger than 3.5"**.

- **Speed Expectations Challenged for GPT-4**: One member questions the expectation that **GPT-4** would be faster, considering its larger size compared to the previous models.

- **Request for AI Security Project Assistance**: A member named **abhibetter** asks for help regarding AI application in a security project but doesn’t provide details about the specific issues or questions they have.

- **Exploring GPT-2 Performance**: Member **namenot223_69478** inquires if anyone has experimented with **GPT-2** on **chatlmsys**, with another guiding to a different channel for an in-depth discussion.

- **Dealing with Bulk Deletion of Chat Archives**: **silensu** is seeking advice on how to handle the accidental archiving of numerous chats, questioning the possibility of *mass deletion*.
  

---


**OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1234436383544971264)** (15 messages🔥): 

- **Million Dollar Prompt Competitions Proposed**: A member suggested organizing **prompt engineering competitions** with significant cash prizes to stimulate learning and sharing best practices within the community. They envision both paid and free "playground" competitions, creating a gamified environment that rewards positive collaboration and practical achievements in prompt crafting.

- **Meta Prompting Paves the Way**: In the discussion about improving prompt crafting, it was noted that "meta prompting" is an effective method, as employed by **GPT Builder**, where the AI adjusts context and conversation based on user instructions to optimize results.

- **Challenges of Negative Prompting in AI**: Users discussed the inefficacy of negative prompting when instructing AI, explaining that highlighting **prohibited words** can lead to inconsistency and less effective results compared to positive examples and instructions.

- **Navigating Localized Language for AI Tasks**: A user grappled with adapting AI-generated text for regional language variants, in particular Argentinian Spanish, where certain words have different connotations. Options like reframing the project and providing specific substitutions for regional words were discussed to better tailor outputs despite a large list of prohibited words.
  

---


**OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1234436383544971264)** (15 messages🔥): 

- **Prompt Engineering with Competitions**: A member proposed having prompt competitions to improve prompt engineering skills. Competitions would range from *no-code* challenges, where the AI processes data to extract information, to interactive tasks like navigating text-based games, and would include community discussions and knowledge sharing.

- **Meta-Prompting over Competitions**: One participant suggested using *meta-prompting*, a method where the AI assists in crafting better prompts, which could potentially replace the need for competitions. This indicates a trend towards users attempting to streamline the prompting process via **GPT Builder**.

- **GPT Builder and Meta Prompting in Action**: Discussion highlighted that **GPT Builder** operates on *meta prompting*, with the AI making context and conversation adjustments based on user requests, hinting at documentation for optimized prompting tactics.

- **Positive Prompting Favored Over Negative**: In addressing a problem with unwanted language generation, it's advised to use *positive instructions and examples* in prompts rather than specifying prohibited words. Suggestions included creating prompts that reinforce preferred terms and explaining usage within particular dialects.

- **Navigating Multilingual Nuances**: Confronting the multilingual challenge, a user expressed difficulties in constructing prompts for variants of Spanish, where words may have different connotations across regions. Strategies to refine AI language output include rephrasing the project or explicitly pairing prohibited words with their desired alternatives.
  

---



**OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1234518684274262038)** (25 messages🔥): 

- **LLaMA 3 Sensitive to Quantization**: A discussion highlighted that **LLaMA 3** experiences more [degradation from quantization](https://x.com/rohanpaul_ai/status/1784972618472317180) than LLaMA 2, likely due to its training on a record 15T tokens which allowed it to capture extremely nuanced data relationships.
- **LLaMA 3 Tokenization Troubles**: There was an issue mentioned with **llama-3** not generating a beginning-of-sentence (BOS) token, but was resolved by adding the BOS into the chat template manually.
- **Critique of Quantization Sensitivity Study**: The community discussed a study on **quantization sensitivity**, suggesting that it is linked to model training methods rather than just the size of the model, with a member describing a related [arXiv paper](https://arxiv.org/abs/2311.16452) as "worthless."
- **Llama-3 Extends Context Length**: The **Llama-3 8B Gradient Instruct 1048k model** was mentioned, which extends the model's context length significantly and was developed by Gradient with compute sponsorship from Crusoe Energy, detailed on [huggingface.co](https://huggingface.co/gradientai/Llama-3-8B-Instruct-Gradient-1048k).
- **BOS Requires Template Tweaks**: Encountering issues with the LLaMA-3 model's BOS token generation, it was noted that altering the tokenizer alone wasn't enough and that the BOS needs to be included in the chat template to appear.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2311.16452">Can Generalist Foundation Models Outcompete Special-Purpose Tuning? Case Study in Medicine</a>: Generalist foundation models such as GPT-4 have displayed surprising capabilities in a wide variety of domains and tasks. Yet, there is a prevalent assumption that they cannot match specialist capabil...</li><li><a href="https://x.com/rohanpaul_ai/status/1784972618472317180">Tweet from Rohan Paul (@rohanpaul_ai)</a>: Quantization is quite harmful for LLaMA 3 than for LLaMA 2.  This PR in llama cpp repo investigates it well.  (Perplexity measures how well the model can predict the next token with lower values being...</li><li><a href="https://huggingface.co/gradientai/Llama-3-8B-Instruct-Gradient-1048k">gradientai/Llama-3-8B-Instruct-Gradient-1048k · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1234556675000897687)** (7 messages): 

- **Exploring Huggingface's ZeroGPU**: A member mentioned they have gained access to the [Huggingface Zero project](https://huggingface.co/zero-gpu-explorers), inviting anyone to suggest tests to conduct using this new platform.
- **ZeroGPU Provides Free Multi-GPU Access**: They shared information about **ZeroGPU** which is a beta feature on **Huggingface** that offers **free GPU access** and the ability to run Spaces on **multiple GPUs**, using _Nvidia A100_. ZeroGPU optimizes GPU utilization by efficiently allocating and releasing resources as needed.
- **Missed Opportunities**: A couple of members expressed regret for not signing up for the ZeroGPU project earlier to take advantage of the **early access for [PRO subscribers](https://huggingface.co/subscribe/pro)**.

**Link mentioned**: <a href="https://huggingface.co/zero-gpu-explorers">zero-gpu-explorers (ZeroGPU Explorers)</a>: no description found

  

---


**OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1234752355518386236)** (11 messages🔥): 

- **Llama-3-70B Finetuning in Question**: A member is advised that fine-tuning `meta-llama/Meta-Llama-3-70B-Instruct` might degrade its performance since it's already fine-tuned. It's recommended to start with an 8B model before moving to the more complex 70B.

- **Dataset Format Conversion Guide**: Members suggested a simple method to convert a fine-tuning dataset from OpenAI's format to ShareGPT's format; replace "messages" with "conversations", "role" with "from", "content" with "value", "user" with "human", and "assistant" with "gpt".

- **Fine-Tuning Learning Path Recommended**: An experienced community member recommends beginners to fine-tune smaller models such as an 8B before attempting to fine-tune larger models like the 70B.

- **Dataset Transformation Done Easily**: Python code was provided to facilitate the transformation of data from the given format into the one required by ShareGPT using a dictionary for role mapping and list comprehension.

**Link mentioned**: <a href="https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/conversation.html#sharegpt.load_role)">Axolotl - Conversation</a>: no description found

  

---


**OpenAccess AI Collective (axolotl) ▷ #[rlhf](https://discord.com/channels/1104757954588196865/1112023522039058553/)** (1 messages): 

gbourdin: add to my bookmarks. Thanks for this !
  

---


**OpenAccess AI Collective (axolotl) ▷ #[community-showcase](https://discord.com/channels/1104757954588196865/1117851527143493664/1234879220686258296)** (2 messages): 

- **Axolotl Fine-Tuning Made Easier**: A member shared a [tutorial](https://github.com/dstackai/dstack/blob/master/examples/fine-tuning/axolotl/README.md) that guides users on fine-tuning `axolotl` using `dstack`, an open-source orchestrator that works with any cloud or pool of on-prem machines. The tutorial was contributed by an `axolotl` user.
- **Community Approves**: Another member expressed appreciation for the tutorial, mentioning that it looks easy to follow.

**Link mentioned**: <a href="https://github.com/dstackai/dstack/blob/master/examples/fine-tuning/axolotl/README.md">dstack/examples/fine-tuning/axolotl/README.md at master · dstackai/dstack</a>: An open-source container orchestration engine for running AI workloads in any cloud or data center. https://discord.gg/u8SmfwPpMd - dstackai/dstack

  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-help-bot](https://discord.com/channels/1104757954588196865/1225300056442409040/1234456215904587827)** (10 messages🔥): 

- **LoRA vs QLoRA Clarified**: The main distinction between **LoRA** and **QLoRA** is that while LoRA focuses on model adaptation via low-rank matrices, QLoRA combines this with quantization for further optimized deployment. *LoRA adapts pre-trained models efficiently; QLoRA takes it a step further for resource-constrained environments.*

- **Trimming Axolotl Datasets to a Percentage**: Trimming datasets in the Axolotl configuration to use a specific percentage isn't a built-in feature, and would require preprocessing or alterations to the dataset loading script. The use of `DPODataset` could be modified with subsampling logic during dataset loading.

- **Equating GPU and Micro Batch Sizes**: It was questioned whether using **4x GPU & Micro Batch Size 4** is equivalent to **8x GPU & Micro Batch Size 2** for final output. No specific answer was given in the channel discussion.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=c42603f2-ce0e-4806-aa15-b77ac3002f7d)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=650c6038-10b5-46b9-aacc-ce5f8e81ff17)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-phorm-bot](https://discord.com/channels/1104757954588196865/1225558824501510164/1234798037612625921)** (39 messages🔥): 

- **Command-R Model Fine-tuning**: Members discussed fine-tuning the *command-r* model within Axolotl. A user shared an [untested pull request](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1547) for adding *command-r* to Axolotl, but noted that it's untested and a merger is not yet recommended.

- **Format Adaptation for command-r**: When inquired about using *command-r's instruct format*, a suggestion was made to use `input_output` formats and pre-prepare them with the correct tokens. A more comprehensive guide on implementing uncommon formats is available in the [input_output documentation](https://openaccess-ai-collective.github.io/axolotl/docs/input_output.html).

- **Sample Packing Feature Uncertainty**: There is confusion regarding the implementation of the *sample packing* feature which packs small examples into larger ones for Axolotl. While the feature is desired by some users, it appears to necessitate modifications outlined in an untested pull request.

- **Inexperienced with runpod Template**: A user expressed uncertainty on integrating patch changes due to unfamiliarity with the *runpod template*. No clear solution was provided in the thread.

- **Unclear Support for phi-3 Format**: A user queried about Axolotl's support for phi-3 format, but the bot response suggested that phi-3 is not supported according to the current documentation. The compatibility of various models including phi with different features is listed, but phi-3 is not specifically mentioned.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/OpenAccess-AI-Collective/axolotl/pull/1547">Feat: Add cohere (commandr) by NanoCode012 · Pull Request #1547 · OpenAccess-AI-Collective/axolotl</a>: Description  Motivation and Context   How has this been tested?    Untested! Screenshots (if appropriate) Types of changes  Social Handles (Optional)</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=83b91c9b-bb5c-4485-894c-0b878d17f7e2)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.</li><li><a href="https://github.com/openaccess-ai-collective/axolotl/tree/main/README.md#L77L100)">axolotl/README.md at main · OpenAccess-AI-Collective/axolotl</a>: Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=1f87fb72-80ec-4321-b37b-d7574206e8af)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.
</li>
</ul>

</div>
  

---



**Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1234538847430246513)** (80 messages🔥🔥): 

- **Exploring Memory for Autonomous Agents**: A discussion touched on a GitHub project called [Memary](https://github.com/kingjulio8238/memary), which has been created to serve as long-term memory for autonomous agents. The conversation clarified that while a knowledge graph might be used, Memary primarily functions through similarity searches over documents.
  
- **Debate on Mysterious GPT-2 Chatbot**: Conversation sparked around a perplexing [GPT2-chatbot](https://chat.lmsys.org/) with gpt4-level capabilities, featured on lmsys. Despite various analyses and speculations, the true origin or nature of this model remains unclear, with one possibility being a finetuned version of OpenAI's original GPT-2.

- **Open-Source AI Faces Big Tech**: A blogpost from [Prime Intellect](https://www.primeintellect.ai/blog/our-approach-to-decentralized-training) highlighted the challenges for open-source AI development in competing with closed-source counterparts who use large, interconnected GPU clusters. The post elaborates on decentralized training as a potential solution for open-source progress.

- **Discussion on Roles of Agents and LLMS**: A deep discussion took place regarding the conflation of autonomous agents with large language models (LLMs). The Talk illustrated a shift in framework towards using "modules" for concurrently built shared context/memory for reasoning and planning, rather than expecting LLMs to function as standalone autonomous units.

- **Learning AI Foundations and Skills**: A user inquired about ways to learn AI from the ground up, seeking to understand basic concepts without committing to a specific field. Other members provided resources including YouTube tutorials on neural networks, introductory courses on AI engineering, and guidance on prompt engineering.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/AlexReibman/status/1784844434682560721">Tweet from Alex Reibman 🖇️ (@AlexReibman)</a>: OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments  Ever since OpenInterpreter, we&#39;ve all been wondering just how effective agents can be if you give them a...</li><li><a href="https://www.latent.space/p/aie-2023-workshops">AI Engineering 101 and 201 Workshops</a>: from AI Engineer Summit 2023</li><li><a href="https://x.com/lmsysorg/status/1785078213712208291?s=46&t=tMWvmS3OL3Ssg0b9lKvp4Q">Tweet from lmsys.org (@lmsysorg)</a>: hi @simonw, thanks a ton! We really value your feedback.  Just to clarify, following our policy, we&#39;ve partnered with several model developers to bring their new models to our platform for communi...</li><li><a href="https://learnprompting.org/docs/intro">Learn Prompting: Your Guide to Communicating with AI</a>: Learn Prompting is the largest and most comprehensive course in prompt engineering available on the internet, with over 60 content modules, translated into 9 languages, and a thriving community.</li><li><a href="https://rentry.co/GPT2">GPT-2?</a>: Background https://chat.lmsys.org provides blind-tested user benchmarks for LLMs (and some MLLMs). One of the models recently available is GPT2-chatbot, which demonstrates capability greatly beyond an...</li><li><a href="https://www.primeintellect.ai/blog/our-approach-to-decentralized-training">State-of-the-art in Decentralized Training</a>: This post explores various novel decentralized training approaches and how they can enable effective AI model training across globally distributed GPUs.</li><li><a href="https://roadmap.sh/prompt-engineering">Prompt Engineering Roadmap - roadmap.sh</a>: Step by step guide to learn Prompt Engineering. We also have resources and short descriptions attached to the roadmap items so you can get everything you want to learn in one place.</li><li><a href="https://x.com/karan4d/status/1785000251096437161?s=46&t=">Tweet from mephistoooOOHHHHHHSHI- (@karan4d)</a>: Ok it’s definitely using GPT-4 tokenizer so I’m betting it is 4.5 as well.   Always fingerprint w anomalous tokens</li><li><a href="https://x.com/lmsysorg/status/1785078213712208291">Tweet from lmsys.org (@lmsysorg)</a>: hi @simonw, thanks a ton! We really value your feedback.  Just to clarify, following our policy, we&#39;ve partnered with several model developers to bring their new models to our platform for communi...</li><li><a href="https://x.com/albfresco/status/1784964830887104999?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from albs — 3/staccs (@albfresco)</a>: my guess is this mysterious &#39;gpt2-chatbot&#39; is literally OpenAI&#39;s gpt-2 from 2019 finetuned with modern assistant datasets.  in which case that means their original pre-training is still am...</li><li><a href="https://x.com/karan4d/status/1785000251096437161?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from mephistoooOOHHHHHHSHI- (@karan4d)</a>: Ok it’s definitely using GPT-4 tokenizer so I’m betting it is 4.5 as well.   Always fingerprint w anomalous tokens</li><li><a href="https://x.com/markatgradient/status/1785032103429865748?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Mark Huang (@markatgradient)</a>: 1M context length  Llama-3 8B Model.  Enough said.    Up on HF @ClementDelangue   cc: @winglian @mattshumer_  ↘️ Quoting Gradient (@Gradient_AI_)   We&#39;ve been in the kitchen cooking 🔥 Excited to ...</li><li><a href="https://x.com/MKBHD/status/1785102259740667960">Tweet from Marques Brownlee (@MKBHD)</a>: NEW VIDEO - Rabbit R1: Barely Reviewable  https://youtu.be/ddTV12hErTc  This is the pinnacle of a trend that&#39;s been annoying for years: Delivering barely finished products to win a &#34;race&#34; ...</li><li><a href="https://github.com/xlang-ai/OSWorld">GitHub - xlang-ai/OSWorld: OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments</a>: OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments - xlang-ai/OSWorld</li><li><a href="https://github.com/kingjulio8238/memary">GitHub - kingjulio8238/memary: Longterm Memory for Autonomous Agents.</a>: Longterm Memory for Autonomous Agents. . Contribute to kingjulio8238/memary development by creating an account on GitHub.</li><li><a href="https://www.youtube.com/watch?v=1hDK7gZbJqQ&t=25s">Ep. 8 — ColBERT + ColBERTv2: late interaction at a reasonable inference cost</a>: Andrew Yates (Assistant Professor at the University of Amsterdam) and Sergi Castella (Analyst at Zeta Alpha) discus the two influential papers introducing Co...</li><li><a href="https://youtu.be/aircAruvnKk?feature=shared),">But what is a neural network? | Chapter 1, Deep learning</a>: What are the neurons, why are there layers, and what is the math underlying it?Help fund future projects: https://www.patreon.com/3blue1brownWritten/interact...</li><li><a href="https://x.com/jessechenglyu/status/1785342519045394465?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Jesse Lyu (@jessechenglyu)</a>: get your r1 update to the latest version now - we addressed most of the issues we found so far and more fix/improvements incoming! idle battery life up to 5x better now.  ↘️ Quoting rabbit inc. (@rabb...
</li>
</ul>

</div>
  

---



**OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1234527192797417594)** (21 messages🔥): 

- **Question on Launching OS Mode with Local Vision Model**: A member asked **how to start OS mode with a local vision model** to try **Moondream**, but reported getting gibberish with the command `interpreter --os --local`.
- **Discussion on Model Functionality**: Another user mentioned using `llava` **months ago** and confirmed that it is possible to get **a description of an image through OpenInterpreter without executing custom code**.
- **Integration Update for OpenInterpreter**: A member announced they managed to integrate all OpenInterpreter outputs into **MagicLLight**, with a pull request to OpenInterpreter planned for `stream_out` function hook and `external_input`. Code release for MagicLLight and AAA+ is expected after some cleanup.
- **OpenInterpreter on Budget Hardware**: The feasibility of running **OpenInterpreter smoothly on a BeepyBerry-Raspberry Pi Zero** was questioned, with a link to a related [YouTube video](https://youtube.com/shorts/E7WQZdJKsbM?si=1XMj0aTtN83cZ5aY).
- **Seeking Debugging Assistance for Bad Startups**: A user sought help for **debugging a bad startup**, indicating the errors were vague. They were directed to share the errors so that the community could assist in troubleshooting.

**Link mentioned**: <a href="https://discord.gg/SdwpMQaW?event=1232436050165764096">Discord - A New Way to Chat with Friends &amp; Communities</a>: Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.

  

---


**OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1234538284089344000)** (20 messages🔥): 

- **Push Button Code Success**: A member successfully resolved an issue related to an external push button not reacting by updating the `ButtonChecker` code and wiring the button to **pin 25**, offering a [snippet of the revised code](https://discord.com/channels/openinterpreter/01). Their efforts were confirmed to be working by another community member.
- **Speaker Connection Stability**: In another hardware related fix, it was recommended to use hot glue to secure the speaker wires and reduce stress on the connections when interfacing with pins for a project.
- **Raising Speaker Volume Inquiry**: A query was raised on how to increase the volume on speakers, with suggestions to try **M5Unified** or potentially use an [external amplifier](https://www.amazon.com/dp/B01DKAI51M).
- **Youtuber Reviews Debated**: There was a discussion about the relevance of YouTuber reviews of AI products like **AI pins** and **R1**, questioning if tech reviewers like **MKBHD** and **Dave2d** fully grasp the AI space, which is different from reviewing consumer electronics like phones or laptops.
- **01 Light Hardware with OS Mode**: A member sought assistance on getting OS mode to work with the current version of the 01 light hardware mentioning successful connectivity to Mac but without access to the screen.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.amazon.com/dp/B01DKAI51M">no title found</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=ddTV12hErTc&ab_channel=MarquesBrownlee">Rabbit R1: Barely Reviewable</a>: AI in a Box. But a different box.Get a dbrand skin and screen protector at https://dbrand.com/rabbitMKBHD Merch: http://shop.MKBHD.comTech I&#39;m using right no...
</li>
</ul>

</div>
  

---



**tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1234533862407671860)** (10 messages🔥): 

- **Tinygrad Inquiry**: A user asked what **tinygrad** is, and another member provided a link to the [tinygrad GitHub repository](https://github.com/tinygrad/tinygrad/tree/master) defining it as a project that those who like PyTorch and micrograd will love.
- **Discord Discovery Mystery**: One member voiced curiosity about how another stumbled upon the Discord server, to which the latter replied uncertainly, indicating a lack of knowledge about their discovery method.
- **Seeking Bounty Guidance**: A user sought help for two bounties involving *"Mean of symbolic shape"* and *"Symbolic arrange"* and was looking for references to understand and solve them.
- **Backward Pass Optimization Issue**: A member was investigating issue [#3572](https://github.com/tinygrad/tinygrad/issues/3572) related to backward passes with 2 reduce operations and inquired about how to generate graph diagrams to illustrate the problem.
- **Graph Diagram Generation for Tinygrad**: In response to a query about generating graph diagrams to address a backward pass issue, a member mentioned the use of `GRAPH=1`, suggesting the use of an environment variable to facilitate this task.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/tinygrad/tinygrad/pull/4362">tensor variable by geohot · Pull Request #4362 · tinygrad/tinygrad</a>: no description found</li><li><a href="https://github.com/tinygrad/tinygrad/tree/master">GitHub - tinygrad/tinygrad: You like pytorch? You like micrograd? You love tinygrad! ❤️</a>: You like pytorch? You like micrograd? You love tinygrad! ❤️  - GitHub - tinygrad/tinygrad: You like pytorch? You like micrograd? You love tinygrad! ❤️
</li>
</ul>

</div>
  

---


**tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1234463379603722291)** (29 messages🔥): 

- **Exploring TinyGrad's Learning Resources**: Members discussed resources for learning AI development with TinyGrad; links to [MicroGrad GitHub repository](https://github.com/unknownusername504/MicroGrad) and [MiniTorch](https://minitorch.github.io/) were shared, with MiniTorch highlighted as a teaching tool for understanding deep learning systems.
- **TinyGrad Quick Start Guidance Shared**: A user recommended the "[tinygrad Quick Start Guide](https://tinygrad.github.io/tinygrad/quickstart/)" for anyone looking to learn AI, especially with TinyGrad, as it provides a basic overview of the high-level API that TinyGrad offers for model development.
- **Symbolic Mean Bounty Challenge in TinyGrad**: Discussions revolved around implementing a symbolic mean operation in TinyGrad, with considerations about LazyBuffer's need to handle data of type Variable and whether it should allocate memory.
- **Pull Request for Symbolic Execution in TinyGrad**: A link to a [previous pull request](https://github.com/tinygrad/tinygrad/pull/1552) was shared to illustrate the mechanism for symbolic code generation and execution in TinyGrad, hinting at how variable caching might be useful for operations like `sum` and `mean`.
- **Developing Symbolic Mean with Variables**: The conversation continued with the development of symbolic mean, focusing on the need to represent tensor lengths symbolically and the potential for `Const` to support variables in the input buffer. Links to a comparision of master and feature branch on GitHub, [tinygrad symbolic-mean-var-pull](https://github.com/tinygrad/tinygrad/compare/master...davidjanoskyrepo:tinygrad:symbolic-mean-var-pull), and further [GitHub changes by gh](https://github.com/tinygrad/tinygrad/compare/86d90511cee2%5E...97a2d44d9840) were shared as part of solving this challenge.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tinygrad.github.io/tinygrad/quickstart/">Quickstart - tinygrad docs</a>: no description found</li><li><a href="https://github.com/tinygrad/tinygrad/compare/master...davidjanoskyrepo:tinygrad:symbolic-mean-var-pull">Comparing tinygrad:master...davidjanoskyrepo:symbolic-mean-var-pull · tinygrad/tinygrad</a>: You like pytorch? You like micrograd? You love tinygrad! ❤️  - Comparing tinygrad:master...davidjanoskyrepo:symbolic-mean-var-pull · tinygrad/tinygrad</li><li><a href="https://github.com/tinygrad/tinygrad/compare/86d90511cee2%5E...97a2d44d9840">Comparing 86d90511cee2^...97a2d44d9840 · tinygrad/tinygrad</a>: You like pytorch? You like micrograd? You love tinygrad! ❤️  - Comparing 86d90511cee2^...97a2d44d9840 · tinygrad/tinygrad</li><li><a href="https://github.com/unknownusername504/MicroGrad">GitHub - unknownusername504/MicroGrad</a>: Contribute to unknownusername504/MicroGrad development by creating an account on GitHub.</li><li><a href="https://minitorch.github.io/">MiniTorch</a>: no description found</li><li><a href="https://github.com/tinygrad/tinygrad/commit/77589bc7a5430ee470621e43fb1817259d3ce0f5">rename Scalar to ConstType and cast_scalar to as_const (#3946) · tinygrad/tinygrad@77589bc</a>: prereq cleanup to make const arg same python type as dtype</li><li><a href="https://github.com/tinygrad/tinygrad/pull/1552">symbolic codegen and exec by chenyuxyz · Pull Request #1552 · tinygrad/tinygrad</a>: part of #1353 , codegen and exec to implement realize for symbolic inputs. The combined var_vals are passed into kernel function directly. I have implemented the backend for CLANG, GPU, METAL. glob...
</li>
</ul>

</div>
  

---



**Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1234452605997023242)** (34 messages🔥): 

- **Single URL Constraint in Command-R**: In a discussion about the **web-search tool** in **API Command R+**, members clarified that currently only one website can be used with the `site` option of the tool, suggesting that a workaround might be to run **an API call for each individual website**.
- **Lack of Multi-step Connectors**: **Cohere** confirmed that **connectors** cannot be used with multi-step tool use within **Command-R** at the moment.
- **Hopes for Future Command-R Features**: A member suggested desirable enhancements for **Command-R** with a focus on **Connectors**, such as using multiple websites in `web_search`, sending extra parameters to custom connectors for more granular control, and enabling a `use_rerank` option to automatically rerank. A helpful link to the documentation was shared: [Cohere Chat Documentation](https://docs.cohere.com/reference/chat).
- **Questions on Model Availability**: A query was posed about the availability of the "Generate" option for fine-tuning models, since it was noticed to be missing from the dashboard, leading to speculation about whether it would be returning.
- **Strategies for Efficient Embedding**: A member inquired about strategies for **keeping data updated** to embed efficiently, touching on the need for cost-effective methods to only reindex chunks of data that have been updated.

**Link mentioned**: <a href="https://docs.cohere.com/reference/chat">Chat API Reference - Cohere Docs</a>: no description found

  

---


**Cohere ▷ #[collab-opps](https://discord.com/channels/954421988141711382/1218409745380147320/1234628219492241449)** (2 messages): 

- **Swedish Salutations**: A member from Stockholm, Sweden mentioned **using Cohere** in their company.
- **Nordic Collaboration**: Another member highlighted their connection to both **Norway and Sweden** through their company, Omegapoint.
  

---



**LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1234429137763176458)** (12 messages🔥): 

- **Gemini Model Exploration**: A member is seeking someone with experience in **Gemini 1.0 or 1.5 models** to discuss specifics privately via direct message.
- **Seeking LLM Observability Tools**: There's a request for recommendations on Large Language Model (LLM) observability tools. The member is considering **Arize Phoenix** or **Langfuze**, with a preference for a self-hosted, open-source option compatible with **LlamaIndex**.
- **OpenAI and SQL Security**: A member inquires about connecting OpenAI directly to an **SQL server without using LangChain**, prioritizing security in the process.
- **Leveraging Langgraph with autoawq**: There is a discussion on integrating **autoawq** with **LangGraph** for use with **exllamav2 kernerls** to achieve high inference speeds in powering AI agents.
- **PDF Content Extraction Challenge**: A new member to langchain and AI programming is seeking advice on how to improve results when splitting a single table that spans multiple pages in a PDF, mentioning they've had unsatisfactory results using **unstructure** for AI-driven PDF content extraction.
  

---


**LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1234549931969216563)** (2 messages): 

- **AzureSearchVectorStoreRetriever Async Issue**: A user mentioned encountering an error due to **AzureSearchVectorStoreRetriever** not supporting async operations and inquired about possible solutions. Options discussed included either requesting langserver to implement such a feature or creating an async wrapper around the synchronous retrieve function.

- **Using Google Drive Libraries**: Another user suggested utilizing the Google Drive libraries for a function, also mentioning the requirement to set the drive key as an environment variable. It was noted that these libraries had been removed and then re-added in the past.
  

---


**LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1234542917406822560)** (8 messages🔥): 

- **A Trip Down Memory Lane with GPT-1**: A blogger has revisited the **original GPT-1 model**, providing insights into how it laid the groundwork for current LLMs and noting its similarities with models like **Mistral-7B**. The blog includes discussions on positional embeddings and Conv1D within the transformer block, available at [Revisiting GPT-1: The Spark That Ignited LLMs](https://amgadhasan.substack.com/p/revisiting-gpt-1-the-spark-that-ignited-llms).

- **Showcasing LangChain on Airbnb**: A demonstration video titled **"D-ID Airbnb Use Case: A RAG Agent Demo using Ollama and Langchain with code on Github"** illustrates an innovative **Live Avatar Q&A** for property sites, powered by LangChain with a collection of 150 QA pairs. Check out the demo on [YouTube](https://youtu.be/N_GcPLJCQQY).

- **Serve Up Answers with a Pizza Bot**: Another use case for LangChain is presented in a video showcasing a **Pizza Bot** with a live avatar interface. See this mobile-friendly application in action on [YouTube](https://youtu.be/6Qa2qdlN2pU).

- **No-Code Automation for Code Maintenance**: An announcement for a no-code platform called **Autonoma** demonstrates its purpose to automate code improvement tasks, such as input validation, error handling, and testing, which is now available for a free demo and integrating with GitHub. Test these agents through [Autonoma Free Demo](https://gitgud.autonoma.app?utm_source=discord&utm_medium=chat&utm_campaign=discord-langchain&utm_id=discord-langchain).

- **Introducing VectorDB Plugin for LM Studio**: A GitHub repository has been shared for a plugin named **VectorDB**, which creates a ChromaDB vector database to function alongside LM Studio in server mode. The repository can be found at [VectorDB Plugin for LM Studio on GitHub](https://github.com/BBC-Esq/VectorDB-Plugin-for-LM-Studio).

- **QuickVid: AI-Powered YouTube Summarization Tool**: QuickVid, a new tool that provides fast summaries and fact verification for YouTube videos, has been launched. Try out QuickVid to enhance your YouTube experience with concise, informed summaries at [QuickVid](https://quickvid.vercel.app/).

- **Tutorial on Creating Webloader RAG Applications**: A Medium article details building robust webloader RAG applications using **Groq, Langchain, and Datastax** to power up your applications. The guide is accessible at [Building Powerful Webloader RAG Applications with Groq, Langchain, and Datastax](https://medium.com/ai-advances/building-powerful-webloader-rag-applications-with-groq-langchain-and-datastax-f4816d88bee8).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://amgadhasan.substack.com/p/revisiting-gpt-1-the-spark-that-ignited-llms">Revisiting GPT-1: The spark that ignited the fire of LLMs</a>: A Comprehensive Look at GPT-1&#x27;s Contribution to the Development of Modern LLMs</li><li><a href="https://quickvid.vercel.app/">QuickVid</a>: no description found</li><li><a href="https://gitgud.autonoma.app?utm_source=discord&utm_medium=chat&utm_campaign=discord-langchain&utm_id=discord-langchain>)">GitGud</a>: no description found</li><li><a href="https://youtu.be/N_GcPLJCQQY">D-ID Airbnb Use Case:  A RAG Agent Demo using Ollama and Langchain with code on Github</a>: A demo to help illustrate practical use cases for live avatar assistants for business... I will do a video for the detailed code review so you can try it... ...</li><li><a href="https://github.com/BBC-Esq/VectorDB-Plugin-for-LM-Studio">GitHub - BBC-Esq/VectorDB-Plugin-for-LM-Studio: Plugin that creates a ChromaDB vector database to work with LM Studio running in server mode!</a>: Plugin that creates a ChromaDB vector database to work with LM Studio running in server mode! - BBC-Esq/VectorDB-Plugin-for-LM-Studio
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1234782249166049310)** (2 messages): 

- **Bonjour from Paris**: A member shares a YouTube video titled ["Agent RAG: LangChain et LlamaIndex portés par Mistral Large - Le vent du changement"](https://youtu.be/ol2QMp64lgo), demonstrating the creation of an Advanced RAG assistant using **LangChain**, **Mistral Large**, and **Llamaindex**. The video is meant for the French-speaking community, and the code for the app is available in the video's description on **GitHub**.

- **DIY Llama3 RAG Assistant**: Another member presents a tutorial on how to train **llama3** with private knowledge to build an agentic RAG, in a YouTube video titled ["I want Llama3 to perform 10x with my private knowledge" - Local Agentic RAG w/ llama3"](https://youtu.be/u5Vcrwpzoz8?si=U30s6BAN9Jsaec-P). The video aims to guide viewers through the process of enhancing **llama3**'s performance using their own data.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://youtu.be/u5Vcrwpzoz8?si=U30s6BAN9Jsaec-P">&quot;I want Llama3 to perform 10x with my private knowledge&quot; - Local Agentic RAG w/ llama3</a>: Advanced RAG 101 - build agentic RAG with llama3Get free HubSpot report of how AI is redefining startup GTM strategy: https://clickhubspot.com/4hx🔗 Links- F...</li><li><a href="https://youtu.be/ol2QMp64lgo">Agent RAG: LangChain et LlamaIndex portés par Mistral Large - Le vent du changement</a>: Dans cette nouvelle vidéo, je vous présente le développement d&#39;un RAG Assitant développé à partir d&#39;agent utilisant Mistral, Langchain et LlamaIndex.Le code ...
</li>
</ul>

</div>
  

---



**Alignment Lab AI ▷ #[ai-and-ml-discussion](https://discord.com/channels/1087862276448595968/1087876677603958804/1234580644273717471)** (2 messages): 

- **Inappropriate Content Alert**: A post promising **free leaks** of content from Onlyfans featuring 18+ Teen Girls contained a Discord link. The message also included emojis and an `@everyone` tag to draw broad attention.

**Link mentioned**: <a href="https://discord.gg/CYNumE8ABr">Discord - A New Way to Chat with Friends &amp; Communities</a>: Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.

  

---


**Alignment Lab AI ▷ #[programming-help](https://discord.com/channels/1087862276448595968/1087876753462136873/1234580388391944193)** (3 messages): 

- **Inappropriate Content Alert**: A message was posted containing links to explicit content, potentially violating Discord's community guidelines. The message promoted free access to content involving underage individuals, which is illegal and problematic.

**Link mentioned**: <a href="https://discord.gg/CYNumE8ABr">Discord - A New Way to Chat with Friends &amp; Communities</a>: Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.

  

---


**Alignment Lab AI ▷ #[looking-for-collabs](https://discord.com/channels/1087862276448595968/1095393077415383261/1234580548698247390)** (2 messages): 

The provided message does not pertain to AI collaboration, research, or relevant topics for the "looking-for-collabs" channel, and it appears to be spam. Therefore, there is no appropriate summary content based on this message.

**Link mentioned**: <a href="https://discord.gg/CYNumE8ABr">Discord - A New Way to Chat with Friends &amp; Communities</a>: Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.

  

---


**Alignment Lab AI ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/1234580564871221329)** (2 messages): 

- **Inappropriate Content Alert**: A message promoting **adult content** and so-called 'OnlyFans leaks' was posted, with a Discord invite link provided. This content is clearly inappropriate for the channel and may violate community guidelines.

**Link mentioned**: <a href="https://discord.gg/CYNumE8ABr">Discord - A New Way to Chat with Friends &amp; Communities</a>: Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.

  

---


**Alignment Lab AI ▷ #[landmark-dev](https://discord.com/channels/1087862276448595968/1113327574563692654/1234767716267855884)** (1 messages): 

- **Spam Alert**: A spam message promoting **adult content** and such materials was posted, including a Discord invitation link. This was likely unrelated to the channel's focus and may require moderation action.

**Link mentioned**: <a href="https://discord.gg/CYNumE8ABr">Join the e-girl paradise 🍑🍒 // +18 Discord Server!</a>: Check out the e-girl paradise 🍑🍒 // +18 community on Discord - hang out with 11801 other members and enjoy free voice and text chat.

  

---


**Alignment Lab AI ▷ #[landmark-evaluation](https://discord.com/channels/1087862276448595968/1118282868595109918/1234767861927645225)** (1 messages): 

- **Inappropriate Content Alert**: A user posted a message containing explicit content and an invitation link, promoting access to what appears to be private or sensitive media involving underage individuals. The message includes emojis and a Discord invite URL.

**Link mentioned**: <a href="https://discord.gg/CYNumE8ABr">Discord - A New Way to Chat with Friends &amp; Communities</a>: Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.

  

---


**Alignment Lab AI ▷ #[open-orca-community-chat](https://discord.com/channels/1087862276448595968/1124000038205530182/1234580949870710797)** (2 messages): 

- **Inappropriate Content Alert**: A message promoting **18+ content** and **OnlyFans leaks** was posted, including an invitation link and emojis suggesting adult material. The content of the message is against Discord's community guidelines.

**Link mentioned**: <a href="https://discord.gg/CYNumE8ABr">Discord - A New Way to Chat with Friends &amp; Communities</a>: Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.

  

---


**Alignment Lab AI ▷ #[leaderboard](https://discord.com/channels/1087862276448595968/1135102537817653308/1234768131247964212)** (1 messages): 

- **Inappropriate Content Alert**: A Discord user posted a message promoting **adult content**, including a mention of *'18+ Teen Girls and onlyfans leaks for free'*, along with an invitation link to another server. The user utilized emojis and tagged **@everyone** to draw attention.

**Link mentioned**: <a href="https://discord.gg/CYNumE8ABr">Discord - A New Way to Chat with Friends &amp; Communities</a>: Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.

  

---


**Alignment Lab AI ▷ #[looking-for-workers](https://discord.com/channels/1087862276448595968/1142242166677192774/1234581080389062810)** (2 messages): 

- **Inappropriate Content Warning**: A message was posted promoting **18+ Teen Girls and OnlyFans leaks** with a Discord invite link. This type of content is likely against the platform's rules and may warrant moderation action.

**Link mentioned**: <a href="https://discord.gg/CYNumE8ABr">Discord - A New Way to Chat with Friends &amp; Communities</a>: Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.

  

---


**Alignment Lab AI ▷ #[looking-for-work](https://discord.com/channels/1087862276448595968/1142242683339944027/1234581103633891358)** (2 messages): 

- **Inappropriate Content Alert**: The message suggests sharing of leaked content from OnlyFans involving teen girls, accompanied by a Discord invite link. This post raises serious concerns regarding legality and ethics.

**Link mentioned**: <a href="https://discord.gg/CYNumE8ABr">Discord - A New Way to Chat with Friends &amp; Communities</a>: Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.

  

---


**Alignment Lab AI ▷ #[join-in](https://discord.com/channels/1087862276448595968/1143791237669855302/1234581301672149132)** (2 messages): 

- **Inappropriate Content Alert**: A message was posted that promoted **adult content** including "18+ Teen Girls and onlyfans leaks". The post included an emoji of a peach and the underage sign, along with a Discord invitation link.

**Link mentioned**: <a href="https://discord.gg/CYNumE8ABr">Discord - A New Way to Chat with Friends &amp; Communities</a>: Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.

  

---


**Alignment Lab AI ▷ #[fasteval-dev](https://discord.com/channels/1087862276448595968/1147528620936548363/1234581174794453042)** (2 messages): 

- **Inappropriate Content Alert**: A message was posted promoting **18+ Teen Girls and OnlyFans leaks** with a Discord invite link. The content appears to be explicit and not suitable for this professional setting.

**Link mentioned**: <a href="https://discord.gg/CYNumE8ABr">Discord - A New Way to Chat with Friends &amp; Communities</a>: Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.

  

---


**Alignment Lab AI ▷ #[qa](https://discord.com/channels/1087862276448595968/1147528698669584424/1234581352272363562)** (2 messages): 

- **Inappropriate Content Alert**: A user posted a message promoting **adult content** including '18+ Teen Girls' and 'onlyfans leaks' with a Discord invite link (**not clicked or verified**). The message uses emojis and tags `@everyone` to attract attention.

**Link mentioned**: <a href="https://discord.gg/CYNumE8ABr">Discord - A New Way to Chat with Friends &amp; Communities</a>: Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.

  

---



**AI Stack Devs (Yoko Li) ▷ #[ai-companion](https://discord.com/channels/1122748573000409160/1122788693950857238/1234713529202769981)** (1 messages): 

- **Concerns Over Criminalizing Coping Mechanisms**: A member expressed strong concern regarding criminalizing an unspecified activity that might be the last coping mechanism for men who have suffered from severe personal and legal setbacks. There is a fear that such measures could push these individuals towards extreme actions due to feeling marginalized by society.
  

---


**AI Stack Devs (Yoko Li) ▷ #[events](https://discord.com/channels/1122748573000409160/1131651713204498583/1234598116523642941)** (2 messages): 

- **Game Jam Bonanza with Rosebud AI**: Rosebud AI announces a **Game Jam** in collaboration with Week of AI, inviting participants to create 2D browser-based games with **Phaser JS** around the theme of *Education and AI*. A **$500 prize pool** is up for grabs, and you can find out how to join [here](https://twitter.com/Rosebud_AI/status/1785034624256618617).

- **AIxGames Meetup in SF**: An AIxGames meetup event is scheduled for this Thursday in San Francisco to connect people working with AI in gaming. There are spots for 160 people, and you can RSVP and check the location [here](https://partiful.com/e/TwvC5qxskuPGqiliMj5f), with a call for demo presentations accessible via [this form](https://forms.gle/6hiqnws3tg6EY7348).

**Link mentioned**: <a href="https://partiful.com/e/TwvC5qxskuPGqiliMj5f">RSVP to AIxGames Meetup | Partiful</a>: AI is already changing the gaming landscape, and is probably going to change it a lot more.   We want to gather as many people working at the intersection of AI and Gaming as we can. Whether it is on ...

  

---


**AI Stack Devs (Yoko Li) ▷ #[ai-town-discuss](https://discord.com/channels/1122748573000409160/1132926337598902293/1234530269331980359)** (8 messages🔥): 

- **Revolutionizing NPC Interactions with LLMs**: A user announced their release of LLM-powered NPC models and an inference stack to enhance action spaces and simplify API calls, found at [GigaxGames on GitHub](https://github.com/GigaxGames/gigax). The solution includes a *single LLM call* feature for complex NPC actions, open-weights on [Huggingface's Hub](https://huggingface.co/Gigax), and an API access offer (with a link that appears to be broken).

- **Overcoming LLM Challenges for Game Development**: In pursuit of runtime speeds for gameplay features, they faced multiple issues like NPCs breaking the 4th wall during `speak` commands and missing details in large prompts. The user suggests *output compression*, minimizing model calls, and leveraging smaller models can significantly impact the NPC's performance.

- **Anticipating a Deep Dive into LLM-Enhanced NPCs**: The user has signaled an intent to *write a blog post* about the experienced struggles and insights relating to the fine-tuning of LLMs for NPC behavior improvement. 

- **Peek into a Peer's Journey with NPC Development**: Another user expressed that their project had also encountered challenges with the existing models, noting that *Claude 3* performed better possibly owing to its "empathetic" training background. They are currently exploring a strategy involving functional calling with smaller prompts and are interested in the outputs of such an approach.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/GigaxGames/gigax">GitHub - GigaxGames/gigax: LLM-powered NPCs running on your hardware</a>: LLM-powered NPCs running on your hardware. Contribute to GigaxGames/gigax development by creating an account on GitHub.</li><li><a href="https://tally.so/r/w7d2Rz)">Form - Tally</a>: Made with Tally, the simplest way to create forms.
</li>
</ul>

</div>
  

---


**AI Stack Devs (Yoko Li) ▷ #[ai-town-dev](https://discord.com/channels/1122748573000409160/1137456826733047908/1234844604638167094)** (13 messages🔥): 

- **Local Setup Achieved with Ease**: A member confirmed they successfully ran the setup locally and found the process very straightforward.
- **Kudos for Member Contribution**: A member expressed appreciation for the excellent work of another community member.
- **Stuck on Windows**: One member experienced an issue with cloning the repo on Windows, getting stuck at *'Checking for index or schema changes...'*. It was clarified that **Convex local does not support Windows**.
- **Alternative Commands for Logs and Development**: It was suggested to utilize `just convex dev` for a separate development sync and `just convex logs` to keep tabs on logs, providing commands that include options for **tailoring logs** and **verbose output**.
- **Window Compatibility Workaround**: Members discussed workarounds for the lack of Windows support with **Convex local**, such as using **WSL (Windows Subsystem for Linux)** or **Docker**, and mentioned that Windows compilation is in progress.
  

---



**Skunkworks AI ▷ #[general](https://discord.com/channels/1131084849432768614/1131084849906716735/1234486969397149837)** (15 messages🔥): 

- **Exploring HaystackDB Embeddings**: A user referenced [HaystackDB on GitHub](https://github.com/carsonpo/haystackdb), questioning whether it uses **2-bit embeddings**.
- **Understanding Binary Quantized Indexing**: Clarification was provided that **Binary Quantized (BQ)** indexing is designed to create a smaller index for similarity search, contributing to a more efficient storage and search mechanism.
- **Challenges in Fine-Tuning LLaMA-3**: Members express difficulties with fine-tuning **LLaMA-3**, noting issues such as the model not generating the **EOS token**, and the embedding layer presenting challenges when loaded in different bit formats.
- **Perplexity Fine-Tuning Troubles**: Conversations indicate that **fine-tuning for perplexity on LLaMA-3** may not yield results better than the original models, with suggestions that the tokenizer could be contributing to the issues.
- **Potential Breakthrough with LLaMA-3 Fine-Tuning**: A group member shared success in fine-tuning **LLaMA-3** by utilizing LLaMA-3 specific prompt formatting, linking to a relevant GitHub [pull request for further information](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1553).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/carsonpo/haystackdb">GitHub - carsonpo/haystackdb</a>: Contribute to carsonpo/haystackdb development by creating an account on GitHub.</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/pull/1553">feat: Add LLaMA-3 instruct prompt strategies for fine-tuning   by 0-hero · Pull Request #1553 · OpenAccess-AI-Collective/axolotl</a>: Description This builds on top of and includes the changes in the below PR&#39;s  #1542 #1539  Fastchat PR from @TJ-Solergibert needs to be merged before merging this  lm-sys/FastChat#3257   Motivatio...
</li>
</ul>

</div>
  

---


**Skunkworks AI ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/)** (1 messages): 

oleegg: https://youtu.be/tYzMYcUty6s?si=t2utqcq36PHbk9da
  

---



**Mozilla AI ▷ #[announcements](https://discord.com/channels/1089876418936180786/1089876419926032396/1234890920575631360)** (1 messages): 

- **Mozilla AI is on a Hiring Spree**: Mozilla AI has announced open positions and is on the lookout for new talent. Check out the opportunities and consider applying [here](https://discord.com/channels/1089876418936180786/1230938514955436242/1234870020916510823).

- **Evaluate Models with Lm-buddy**: An open-source tool named Lm-buddy has been introduced for helping evaluate language models more effectively. The tool can be explored and contributed to via the link provided [here](https://discord.com/channels/1089876418936180786/1230938514955436242/1234589599733518378).

- **Prometheus Puts Local LLMs on the Bench**: A project called Prometheus demonstrates the use of Local Large Language Models (LLMs) in the role of a judge. This innovative application can be discussed and delved into further in the dedicated channel linked [here](https://discord.com/channels/1089876418936180786/1234890301143912599/1234890301143912599).
  

---


**Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1234502618420613130)** (13 messages🔥): 

- **AI Tokens Generation Speed Inquiry**: A member inquired about the efficiency of token generation in llama.cpp/llamafile, noting that their implementation of inference for llama2 spends 95% of time on matrix-vector multiplications. They wondered if loop unrolling in llama.cpp could account for its 30% faster performance, as they observed both looping and vectorization in disassembly.
- **LLaMA Naming Mix-Up**: One user experienced a humorous mix-up with message parameters, setting themselves as "Z" and then forgetting about it, leading to some confusion when messages appeared as if LLaMA was talking to itself.
- **Pseudonymous Intrusion Causes Confusion**: Another user recounted an unusual event where someone joined a chat under the name "kimkardashian," causing a bizarre situation. However, the anomaly could not be replicated in subsequent runs.
- **Technology Integration Troubles**: A user struggled to integrate LLaMA with a Plush-for-comfyUI node. Despite the node functioning with other OpenAI endpoints, it failed to operate correctly with llamafile.
- **LLaMA3 Compatibility and Support Communication**: There's an acknowledged issue with running LLaMA3:8b on M1 Macbook Air specifically with llamafile, whereas it runs without problem on Ollama. A pledge was made to prioritize M1 compatibility testing once other ongoing issues with LLaMA3 are resolved.
  

---



**Interconnects (Nathan Lambert) ▷ #[ideas-and-feedback](https://discord.com/channels/1179127597926469703/1179127598442348730/1234545676260474942)** (1 messages): 

Since the provided message appears to be the only one or part of a single message without additional context or other messages, a summarization cannot be performed. Please provide a set of messages from the "ideas-and-feedback" channel, so that I can create an appropriate summary.
  

---


**Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1234547539186024519)** (4 messages): 

- **Exploring OLMo with Hanna Hajishirzi**: A [recent talk](https://youtu.be/qFZbu2P1vZ8) by Hanna Hajishirzi from [AI2](https://homes.cs.washington.edu/~hannaneh/) on "OLMo: Findings of Training an Open LM" has been shared, held at the Open-Source Generative AI Workshop at Cornell Tech. The slides for the talk can be accessed [here](https://drive.google.com/file/d...).
- **Intensity of the Information Flow**: A member reveals that Hanna Hajishirzi is their manager who moves at an incredibly fast pace, suggesting the depth and complexity of her lectures.
- **OLMo Presentation Overwhelming but Impressive**: Another member finds the content of Hanna's 25-minute talk – covering topics like OLMo, Dolma, Tulu – quite vast and a bit overwhelming, yet acknowledges her impressive profile and the value such information may have for students.

**Link mentioned**: <a href="https://youtu.be/qFZbu2P1vZ8">Hanna Hajishirzi (AI2) - OLMo: Findings of Training an Open LM</a>: Talk from the Open-Source Generative AI Workshop at Cornell Tech. Speaker: https://homes.cs.washington.edu/~hannaneh/Slides - https://drive.google.com/file/d...

  

---


**Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1234622923449569322)** (2 messages): 

- **Insights from John Schulman through Gist**: A GitHub [Gist](https://gist.github.com/yoavg/6bff0fecd65950898eba1bb321cfbd81) provided valuable insights, summarizing a talk by **John Schulman** related to reinforcement learning for language model-based systems.

- **Questioning the Utility of AI Leaderboards**: A [blog post](https://www.aisnakeoil.com/p/ai-leaderboards-are-no-longer-useful) by Sayash Kapoor and Benedikt Stroebl claims there's no current accurate method to determine the best AI for code generation. They highlight that the LLM debugger (**LDB**), while topping the HumanEval leaderboard for code generation, is a costly agent due to its reliance on running costly language models like GPT-4.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.aisnakeoil.com/p/ai-leaderboards-are-no-longer-useful">AI leaderboards are no longer useful. It&#x27;s time to switch to Pareto curves.</a>: What spending $2,000 can tell us about evaluating AI agents</li><li><a href="https://gist.github.com/yoavg/6bff0fecd65950898eba1bb321cfbd81">rl-for-llms.md</a>: GitHub Gist: instantly share code, notes, and snippets.
</li>
</ul>

</div>
  

---


**Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/)** (1 messages): 

SnailBot News: <@&1216534966205284433>
  

---



**LLM Perf Enthusiasts AI ▷ #[jobs](https://discord.com/channels/1168579740391710851/1169107992587812864/1234606317595791490)** (1 messages): 

- **AI Engineer Wanted at Renowned AI-Powered Gamma**: **Gamma**, ranked #16 on a16z's top 100 consumer AI apps, is on the lookout for an **AI engineer** to innovate in presentation and website design through AI. The role includes **prompt engineering**, **metrics/evaluations**, **fine-tuning**, and creating features with cutting-edge models, with the job details available at [Gamma Careers](https://careers.gamma.app/ai-engineer).

- **Pushing the Limits of Large Language Models**: Candidates without extensive engineering experience are considered if they possess practical expertise in maximizing the potential of **Large Language Models (LLMs)**. The position is based in **San Francisco** and requires in-person collaboration.

- **Gamma's Impressive AI-Powered Growth and Culture**: Gamma boasts over **10 million users** grown organically, is **profitable with $10M+ in funding**, operates with a **lean 16-member team**, and promotes an office culture with a hybrid workweek in **San Francisco**.

- **Inventive Content Creation at Scale**: With an ambition of **simplifying content creation**, Gamma creates over a million images and processes millions of LLM requests daily. They aim to eliminate the complexities involved in crafting **engaging presentations and websites**.

**Link mentioned**: <a href="https://careers.gamma.app/ai-engineer">AI Engineer</a>: AI Engineer  San Francisco  Click here to apply

  

---


**LLM Perf Enthusiasts AI ▷ #[openai](https://discord.com/channels/1168579740391710851/1171903046612160632/1234583399029805107)** (3 messages): 

- **Speculation on GPT-4.5 Leak**: A tweet by @phill__1 sparked discussions as it suggested the gpt2-chatbot feels like **GPT-4.5**, boasting *'insane domain knowledge'*. The link to the tweet: [phill__1's observation](https://x.com/phill__1/status/1784964135920235000).
- **Community Buzzing About Potential Leak**: Members in the channel expressed belief that the gpt2-chatbot could be an inadvertent preview of **GPT-4.5**.
- **Concise Praise for the Mystery Bot**: A terse endorsement was shared by a member, simply stating, "It's good".

**Link mentioned**: <a href="https://x.com/phill__1/status/1784964135920235000">Tweet from Phil (@phill__1)</a>: Whatever gpt2-chatbot might be, it definitely feels like gpt4.5. It has insane domain knowledge I have never seen before

  

---



**Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/1234505496761991198)** (3 messages): 

- **Custom Grammar for Code-Generation Talk**: A user showed interest in passing a custom grammar, potentially as a model-specific option, to focus on semantic errors in code generation rather than syntax ones.

- **User Experience Brainstorm for Datasette**: Ideas were sought for a UX design on Datasette's front page that would allow users to select options from a drop-down, like choosing a country to generate a summary table.

- **Direct Data Access via Dropdown Selection**: A member proposed two UX approaches: one by updating the URL upon an event to direct the user to relevant data, and another allowing users to "build" the homepage by updating canned queries based on their selections.
  

---



**DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1234775513499963463)** (1 messages): 

- **Fast Loading on Local Machine**: Discussion revolved around the observation that a process *loads in 3 seconds when running on the machine*, yet there seems to be an issue when doing the same through *submitting a job*. This suggests storage may not be the contributing factor to the problem in a job submission context.
  

---


**DiscoResearch ▷ #[benchmark_dev](https://discord.com/channels/1178995845727785010/1183158791605330051/)** (1 messages): 

le_mess: llama 3 seems to beat gpt4 on scandeval
https://scandeval.com/german-nlg/
  

---



