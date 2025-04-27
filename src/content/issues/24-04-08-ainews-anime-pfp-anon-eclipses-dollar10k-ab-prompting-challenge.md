---
id: 56ec05bf-3c4f-454a-aa7c-26178a8809fd
title: Anime pfp anon eclipses $10k A::B prompting challenge
date: '2024-04-09T01:18:42.938105Z'
original_slug: ainews-anime-pfp-anon-eclipses-10k-ab-prompting
description: >-
  **Victor Taelin** issued a $10k challenge to GPT models, initially achieving
  only **10% success** with state-of-the-art models, but community efforts
  surpassed **90% success** within 48 hours, highlighting GPT capabilities and
  common skill gaps. In Reddit AI communities, **Command R Plus (104B)** is
  running quantized on **M2 Max hardware** via **Ollama** and **llama.cpp**
  forks, with **GGUF quantizations** released on Huggingface. Streaming
  text-to-video generation is now available through the **st2v** GitHub repo.
  **WD Tagger v3** was released for mass auto-captioning datasets with a WebUI.
  Lesser-known prompting techniques like self-tagging and generational
  frameworks produced thought-provoking outputs in OpenAI discussions, including
  experiments with self-evolving system prompts. Stable Diffusion users
  discussed image composition importance for training character LoRAs and best
  checkpoints for video game character generation. Discussions also covered
  scarcity of **5B parameter models** and open(ish) licenses for open source AI.
  Memes included jokes about ChatGPT and Gemini training data differences.
companies:
  - openai
  - ollama
  - huggingface
models:
  - command-r-plus-104b
  - stable-diffusion-1.5
topics:
  - quantization
  - model-optimization
  - streaming
  - prompt-engineering
  - self-prompting
  - image-composition
  - character-lora-training
  - model-size
  - open-source-licenses
  - memes
  - humor
people:
  - victor-taelin
  - futuristfrog
---


<!-- buttondown-editor-mode: plaintext -->> AI News for 4/5/2024-4/8/2024. We checked 5 subreddits and [**364** Twitters](https://twitter.com/i/lists/1585430245762441216) and **26** Discords (**387** channels, and **9770** messages) for you. Estimated reading time saved (at 200wpm): **1103 minutes**.

4 days ago, Victor Taelin confidently tweeted a simple A::B challenge for GPTs and then [offered a $10k contest to prove him wrong](https://twitter.com/VictorTaelin/status/1776677635491344744):

 ![image.png](https://assets.buttondown.email/images/1bca54e3-44a4-4e88-b38d-287814273c06.png?w=960&fit=max) 

His initial attempts with all SOTA models got 10% success rates. Community submissions got 56%. [It took another day for @futuristfrog to surpass 90%.](https://twitter.com/victortaelin/status/1777049193489572064) The challenge lasted 48 hours in total. A fun lesson in GPT capability, and another reminder that failure to do something in 2024 AI pre AGI is often a simple skill issue.

---

**Table of Contents**

[TOC] 


---

# AI Reddit Recap

> Across r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence. Comment crawling still not implemented but coming soon.

**Technical Developments and Releases**

- **Command R Plus (104B) working with Ollama**: In /r/LocalLLaMA, Command R Plus (104B) is working with Ollama using a forked llama.cpp, allowing for [**quantized models to run on M2 Max hardware**](https://www.reddit.com/r/LocalLLaMA/comments/1bymeyw/command_r_plus_104b_working_with_ollama_using/).
- **GGUF quantizations for Command R+ 104B released**: In /r/LocalLLaMA, Dranger has released [**GGUF quantizations for Command R+ 104B from 1 to 8 bit on Huggingface**](https://huggingface.co/dranger003/c4ai-command-r-plus-iMat.GGUF).
- **Streaming t2v now available**: In /r/StableDiffusion, streaming t2v is now available, allowing for [**generating longer videos using the st2v Github repo**](https://www.reddit.com/r/StableDiffusion/comments/1by5upa/streaming_t2v_is_now_avaible/).
- **New version of WD Tagger (v3) released**: In /r/StableDiffusion, a new version of WD Tagger (v3) is available for [**mass auto captioning of datasets, utilizing a WebUI interface**](https://www.reddit.com/r/StableDiffusion/comments/1by0zsg/mass_auto_caption_with_wd_tagger_v3_with_webui/).

**Techniques and Prompting**

- **Lesser known prompting techniques yield thought-provoking outputs**: In /r/OpenAI, thought provoking outputs were generated using [**lesser known prompting techniques such as self-tagging output, generational frameworks, and real-time self-checks**](https://www.reddit.com/r/OpenAI/comments/1by9uo8/thought_provoking_outputs_via_lesser_known/).
- **Experiment with self-evolving system prompts**: In /r/OpenAI, an experiment letting OpenAI API write its own system prompt over multiple iterations resulted in [**increasingly flowery and grandeur wording**](https://www.reddit.com/r/OpenAI/comments/1byijwt/letting_openai_api_write_its_own_system_prompt/).
- **Promptless outpaint/inpaint canvas updated**: In /r/StableDiffusion, promptless outpaint/inpaint canvas has been updated to [**run ComfyUI workflows on low-end hardware**](https://v.redd.it/xi2hkxh4l4tc1).

**Questions and Discussions**

- **Importance of image composition when training character LoRAs**: In /r/StableDiffusion, there is a discussion on the [**importance of image composition when training character LoRAs, and whether auto-tagging sufficiently captures details**](https://www.reddit.com/r/StableDiffusion/comments/1byibwu/how_important_is_image_composition_when_training/).
- **Best checkpoint for video game characters in Stable Diffusion 1.5**: In /r/StableDiffusion, there is a question about the [**best checkpoint for generating video game characters in Stable Diffusion 1.5**](https://www.reddit.com/r/StableDiffusion/comments/1by0nnk/which_checkpoint_is_best_for_video_game_characters/).
- **Scarcity of 5B parameter models**: In /r/LocalLLaMA, there is an inquiry about [**why there are so few 5B parameter models compared to 3B and 7B**](https://www.reddit.com/r/LocalLLaMA/comments/1bybtky/why_are_there_so_few_5b_models/).
- **Open(ish) licenses and aligning incentives for open source AI**: In /r/LocalLLaMA, there is a discussion on [**open(ish) licenses and what terms are desired to align incentives for open source AI**](https://www.reddit.com/r/LocalLLaMA/comments/1bymr57/openish_licenses_recap_and_discussion/).

**Memes and Humor**

- **Humorous post about dancing anime girls and "realistic" Call of Duty**: In /r/StableDiffusion, there is a humorous post about [**too many dancing anime girls, countered with a "realistic" Call of Duty image**](https://i.redd.it/av8h3pwy16tc1.png).
- **Joke about ChatGPT vs Gemini training data**: In /r/ProgrammerHumor, there is a joke confirming that [**ChatGPT was trained with YouTubers while Gemini was not**](https://www.reddit.com/gallery/1by68qc).

# AI Twitter Recap

> all recaps done by Claude 3 Opus, best of 4 runs. We are working on clustering and flow engineering with Haiku.

**AI and Robotics Research Developments**

- **AI and robotics progress**: [@adcock_brett](https://twitter.com/adcock_brett/status/1777004161416020407) shares a weekly roundup of the most important research and developments in AI and robotics, highlighting the rapid pace of progress in the field.
- **Rumored capabilities of GPT-5**: [@bindureddy](https://twitter.com/bindureddy/status/1777023216810438900) reports that OpenAI's upcoming GPT-5 model is rumored to have extremely powerful coding, reasoning and language understanding abilities that surpass Anthropic's Claude 3.
- **Sora for generating music videos**: [@gdb](https://twitter.com/gdb/status/1777127364822024283) showcases Sora, a tool that allows users to visualize how a song has always "looked" by generating corresponding music videos.
- **Fast performance of 4-bit Mistral 7B**: [@awnihannun](https://twitter.com/awnihannun/status/1777072588633882741) achieved an impressive **103.5 tokens-per-second** running the 4-bit Mistral 7B model on an M2 Ultra chip.
- **Many-shot jailbreaking technique**: [@adcock_brett](https://twitter.com/adcock_brett/status/1777004446469230651) shares that Anthropic researchers discovered a technique called "many-shot jailbreaking" that can evade the safety guardrails of large language models by exploiting expanded context windows.

**AI Agents and Robotics**

- **Complexity of building AI agents**: [@bindureddy](https://twitter.com/bindureddy/status/1777136946705539363) notes that only 10% of the work in building AI agents is about LLMs and reasoning, while the remaining 90% involves heavy lifting in code, data, memory, evaluation and monitoring. 
- **OpenAI's plans and LLMs in robotics**: [@adcock_brett](https://twitter.com/adcock_brett/status/1776816987202867673) provides an overview of OpenAI's plans and discusses why large language models are important for robotics applications.
- **Key factors for reliable LM-based agents**: [@sarahcat21](https://twitter.com/sarahcat21/status/1776644684997365817) emphasizes that purposeful pretraining and interface design are crucial for building reliable agents based on large language models.
- **Growth of coding agents**: [@mbusigin](https://twitter.com/mbusigin/status/1776377605555454028) highlights the rapid explosion in the development and adoption of coding agents.
- **Figure-01 humanoid robot**: [@adcock_brett](https://twitter.com/adcock_brett/status/1776672870816739369) shares an image of the Figure-01 electromechanical humanoid robot.

**LLM Developments and Capabilities**

- **Grok 2.0 rumored performance**: [@bindureddy](https://twitter.com/bindureddy/status/1777378250962129012) reports that Grok 2.0 is rumored to be the second model after Anthropic's Claude Opus to beat OpenAI's GPT-4 in performance, which would be a significant achievement for Grok and X.
- **Claude 3 Opus outperforms GPT-4**: [@Teknium1](https://twitter.com/Teknium1/status/1777117967802871858) and [@bindureddy](https://twitter.com/bindureddy/status/1777023216810438900) note that Anthropic's Claude 3 Opus model outperforms GPT-4 on certain tasks.
- **New model releases**: [@osanseviero](https://twitter.com/osanseviero/status/1776620683465764936) announces the release of Cohere Command R+, Google Gemma Instruct 1.1, and the Qwen 1.5 32B model family.

**Retrieval Augmented Generation (RAG) Architectures**

- **Finance agent with LangChain and Yahoo Finance**: [@llama_index](https://twitter.com/llama_index/status/1777076087853392027) demonstrates building a finance agent using LangChain and Yahoo Finance, covering functions for stock analysis such as balance sheets, income statements, cash flow, and recommendations.
- **Multi-document agents with LlamaIndex**: [@llama_index](https://twitter.com/llama_index/status/1776627066126901311) and [@jerryjliu0](https://twitter.com/jerryjliu0/status/1776971813874028694) showcase treating documents as sub-agents for both semantic search and summarization using LlamaIndex.
- **Agentic extension for RAG**: [@jerryjliu0](https://twitter.com/jerryjliu0/status/1776971813874028694) proposes an agentic extension for retrieval augmented generation that treats documents as tools and agents for dynamic interaction beyond fixed chunks.
- **Extracting document knowledge graph for RAG**: [@llama_index](https://twitter.com/llama_index/status/1777348428755820849) demonstrates using LlamaParse to extract structured markdown, convert it to a document graph, and store it in Neo4j for advanced querying to power a RAG pipeline.

**Memes and Humor**

- **Timeout set in seconds instead of milliseconds**: [@gdb](https://twitter.com/gdb/status/1776824716931838227) shares a humorous meme about accidentally setting a timeout in seconds instead of milliseconds.
- **Preferring In-N-Out to Shake Shack**: [@adcock_brett](https://twitter.com/adcock_brett/status/1777131105566740830) jokes about preferring In-N-Out to Shake Shack after living in NYC.
- **Biological neural network performance after bad sleep**: [@_jasonwei](https://twitter.com/_jasonwei/status/1777088156443279469) humorously compares the performance of a biological neural network after a bad night's sleep to GPT-4 base with poor prompting.
- **Pains of saying Claude solved something**: [@Teknium1](https://twitter.com/Teknium1/status/1776820170348171298) shares a meme about the pains of admitting that Anthropic's Claude model solved a problem.
- **Studying LeetCode for 3 months without getting a job**: [@jxnlco](https://twitter.com/jxnlco/status/1777095850172268903) shares a meme about the frustration of studying LeetCode for 3 months but not getting a job.

---

# AI Discord Recap

> A summary of Summaries of Summaries

**1. Quantization and Optimization Breakthroughs for LLMs**

- **[QuaRot](https://arxiv.org/abs/2404.00456)** enables end-to-end **4-bit quantization** of large language models like LLaMa2-70B with minimal performance loss, handling outliers while maintaining computational invariance. [HQQ](https://github.com/mobiusml/hqq) also showcased promising 4-bit quantization results integrated with **gpt-fast**.

- **Schedule-Free Optimization Gains Traction**: Meta's **schedule-free optimizers** for AdamW and SGD have been [integrated into Hugging Face's transformers library](https://github.com/huggingface/transformers/pull/30079), potentially revolutionizing model training. Discussions revolved around the [Schedule-Free Optimization in PyTorch repository](https://github.com/facebookresearch/schedule_free) and a related [Twitter thread by Aaron Defazio](https://twitter.com/aaron_defazio/status/1773381393831067787) on the topic.

- Discussions around **torch.compile** focused on its utilization of **Triton** kernels only for CUDA inputs, asynchronous collective operations, DeepSpeed integration, and potential MLP optimizations using **tiny-cuda-nn** or CUTLASS.

**2. Expanding Context Lengths and Attention Mechanisms**

- The **[EasyContext](https://github.com/jzhang38/EasyContext)** project introduces memory optimization and training recipes to extrapolate language model context lengths to **1 million tokens** using **ring attention** on modest hardware like 8 A100 GPUs. A [tweet by Zhang Peiyuan](https://twitter.com/PY_Z001/status/1776176932687892796) discussed the impact of increased context size on training throughput.

- **[Mixture-of-Depths](https://arxiv.org/abs/2404.02258)** proposes dynamically allocating compute across a transformer sequence within a fixed budget, potentially enhancing efficiency without compromising flexibility.

- Discussions covered **linear attention** vs. classic attention, **variable length striped attention** implementations, and the speed/memory trade-offs of ring attention in distributed computing scenarios.

**3. Open-Source AI Advancements and Community Engagement**

- **AMD** announced open-sourcing the Micro Engine Scheduler (MES) firmware and documentation for Radeon GPUs, aligning with broader open-source GPU efforts welcomed by the community. ([The Register Article](https://www.theregister.com/2024/04/05/amd_mes_open_source/), [AMD Radeon Tweet](https://twitter.com/amdradeon/status/1775999856420536532))

- The **[PaperReplica GitHub repository](https://github.com/hegdeadithyak/PaperReplica)** aims to replicate AI/ML research papers through community contributions, fostering knowledge sharing and skill development.

- Licensing changes for **text-generation-inference (TGI)** to Apache 2 sparked a surge in contributors after the project was fully open-sourced, highlighting the potential economic benefits of open ecosystems like **Mistral**.

- **[Command R+](https://huggingface.co/spaces/CohereForAI/c4ai-command-r-plus)** by Cohere demonstrated impressive translation capabilities for archaic languages like Middle High German, outperforming GPT-4 class models and fueling hopes for an open-source release to drive developer engagement.

**4. Multimodal AI Advancements and Applications**

- The **Aurora-M** project introduced a new [15.5B parameter open-source multilingual language model](https://arxiv.org/abs/2404.00399) following the U.S. Executive Order on AI, showcasing cross-lingual impact of mono-lingual safety alignment across 2 trillion training tokens.

- Unsloth AI faced challenges with models like **Chat GPT** and **Claude** accurately converting images to HTML while preserving colors and borders, prompting tongue-in-cheek suggestions to use ASCII art instead.

- **[BrushNet](https://www.youtube.com/watch?v=X89IQop_0dM)**, a new method for AI inpainting incorporating object detection, promises higher quality results as demonstrated in a tutorial video.

- The **LLaVA** vision-language model underwent a novel "Rorschach test" by feeding it random image embeddings and analyzing its interpretations, detailed in [a blog post](https://anotherjesse.com/posts/llava-rorschach/). A compact **nanoLLaVA** model for edge devices was also [released on Hugging Face](https://huggingface.co/qnguyen3/nanoLLaVA).


**5. Misc**

- **Tinygrad Development Progresses with Reversions and Integrations**: **George Hotz** reverted the command queue in tinygrad and is integrating the **memory scheduler** directly with the existing scheduler model using the **multidevicegraph abstraction**. The [TinyJit tutorial](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/jit.md) and [multi GPU training guide](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/multigpu.md) were shared to aid contributors.

- **Jamba Models Offer Alternatives for Limited Hardware**: Scaled-down versions of the **Jamba** architecture, including an [8xMoE with 29B parameters](https://huggingface.co/isemmanuelolowe/Jamba-8xMoE_Slerp) and [4xMoE with 17.7B parameters](https://huggingface.co/isemmanuelolowe/Jamba-4xMoE_Slerp), were created using **spherical linear interpolation (Slerp)** of expert weights to enable fine-tuning on more accessible hardware like a 4090 GPU at 4-bit precision.

---



# PART 1: High level Discord summaries




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

**iOS Users Test Drive New Story Discovery**: Perplexity AI is trialing an innovative **story discovery format on iOS**. Users are encouraged to provide feedback on their experiences through a designated channel, and can download the test app [here](https://pplx.ai/download).

**AI Event Ends in Harmony**: The Perplexity AI Discord event wrapped up with both *eun08000* and *codelicious* sharing first place. Prize recipients will receive direct messages with details.

**Claude 3 Opus - A Model Debate**: On the server, the talk revolved around observed variations in Perplexity's implementation of the **Claude 3 Opus model** compared to others, particularly regarding tasks demanding creativity.

**API Quirks and Queries**: Users noted inconsistencies between Perplexity's API and web application, with the API showing more hallucinations; the API's default model diverges from the [web version](https://pplx.ai). The 'sonar-medium-online' model is suggested for API users to closely mimic the *Sonar* model accessible via the web app for non-Pro users.

**Tech Enthusiasts Share and Learn**: Users exchanged information on a variety of topics from how AI affects the music industry to Tesla's and Apple's latest tech innovations. Additionally, a case study featuring Perplexity AI highlighted a 40% speed increase in model training powered by Amazon Web Services, demonstrating Perplexity's efficient utilization of advanced machine learning infrastructure and techniques.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Rorschach Test for AI Vision Models**: The **LLaVA vision-language model** was put through a novel "Rorschach test" by feeding it random image embeddings and analyzing the interpretations, described in a [blog post](https://anotherjesse.com/posts/llava-rorschach/). Moreover, a compact **nanoLLaVA model** suitable for edge devices was introduced on [Hugging Face](https://huggingface.co/qnguyen3/nanoLLaVA).

- **Claude's Memory Mechanism in Question**: Technical discussions ensued on whether **Claude** retains information across sessions or if the semblance of memory is due to probabilistic modeling. Engineers debated the effectiveness of current models against the challenge of persistent context.

- **Worldsim Woes and Wisdom**: Post-DDoS attack, proposals for a **Worldsim** login system to thwart future threats and discussions of a "pro" version to include more scenarios were afoot. Meanwhile, philosophical musings floated around potential AI-driven simulations akin to observed realities.

- **Chunking for RAG Diversity**: Suggestions arose to pre-generate diverse datasets for RAG using a *chunking script*, alongside talk of creating complex multi-domain queries using **Claude Opus**. Ethical queries surfaced regarding data provenance, specifically using leaked documents from ransomware attacks, contrasting with the clustering strategies like **RAPTOR** for dataset curation.

- **The Coalescence of GitHub and Hermes**: A GitHub repository, **VikParuchuri/marker**, was spotlighted for its high-accuracy PDF-to-markdown conversion, and can be found at [GitHub - VikParuchuri/marker](https://github.com/VikParuchuri/marker). Additionally, discussions focused on enhancing `Hermes-2-Pro-Mistral-7B` to execute functions with `tools` configurations, a hurdle matching the challenges delegates face with full-parameter finetuning vis-à-vis adapter training in various LLM contexts.

- **Canada's AI Ambitions and Enterprise LLMs**: From the introduction of **Command R+**, a scalable LLM by Cohere for businesses, to insights into Canada's strategy to champion global AI leadership, the discourse expanded towards understanding SSL certifications, creating local solutions akin to Google Image Search and untangling the surfeit of AI research and synthesis.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Stability Bot MIA**: Users seeking image generation services were guided to check bot status due to outages, pushing them towards alternate server channels for updates and support.
- **Quality Quest in Image Generation**: Debates emerged comparing local model outputs with Dreamstudio's, with participants recommending open-source upscalers and discussing the effectiveness of various image enhancement techniques.
- **SD3 Buzz Builds**: There is an informal 2-4 week ETA on Stable Diffusion 3 (SD3), sparking conversations around expected improvements and new capabilities of the model.
- **LoRa Training Dialogue**: Information exchange on LoRa training saw users seeking installation advice and citing GitHub repositories for practical training methods.
- **User Interface Upgrades**: Discussions on user interface enhancements included suggestions for transitioning from Automatic 1.1.1.1 to StableSwarm, with a focus on user-friendliness and feature accessibility for new adopters.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

**HTML Conversion Leaves Engineers Blue**: AI engineers discussed the limitations of current language models like **Chat GPT** and **Claude** in accurately converting images to HTML, leading to lost color fidelity and rounded borders. A tongue-in-cheek proposal suggested the use of ASCII art as an alternative, stemming from its ability to elicit responses from AI models as shown in this [Ars Technica article](https://arstechnica.com/security/2024/03/researchers-use-ascii-art-to-elicit-harmful-responses-from-5-major-ai-chatbots/).

**Aurora-M Lights Up Possibilities**: An open-source multilingual model, **Aurora-M**, boasting 15.5 billion parameters, was introduced and caught the community's attention with its cross-lingual safety capabilities, further detailed in [this paper](https://arxiv.org/abs/2404.00399). The findings show that safety alignment in one language can have a positive impact on other languages.

**Jamba Juice or Mamba Sluice? Investment Opinions Clash**: Engineers debated the investment into AI21 Labs' **Jamba**, especially given their recent fundraising of $155 million as reported by [TechCrunch](https://techcrunch.com/2023/08/30/generative-ai-startup-ai21-labs-lands-155m-at-a-1-4b-valuation/). The return on investment (ROI) of focused model fine-tuning was brought to light, presenting an optimistic view despite the model's upfront costs.

**AI Fine-Tuning Perspectives Merge and Diverge**: The community engaged in a robust exchange on fine-tuning approaches, such as unsupervised fine-tuning techniques mentioned like **GGUF**, and the benefits of Dynamic Positional Offsets (DPO). Specific strategies for fine-tuning and the application of techniques like **LoRA** in enhancing performance were discussed.

**Private AI Hosting Hustle**: Data privacy concerns have led members to host their AI projects on personal servers, with anecdotes of using platforms like [Hircoir TTS](https://tts.hircoir.eu.org/) independently. Some envisioned future plans include integrating advertisements to capitalize on the growing portfolio of models.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

**Boost Your Model’s Performance**: The **LM Studio** appears to leap ahead of alternatives like **oogabooga** and **Faraday** with a GUI that wins user preference for its higher quality outputs. Suggestions poured in for expansions, notably for file reading support and modalities such as text-to-image and text-to-voice; such features edge closer to what **Devin** already offers and are angled towards enhancing creativity and productivity.

**Big Thinkers, Bigger Models**: A technical crowd advocates the power play of handling heavyweight models such as the **Command R+**, tipping the scales at 104B, and recommending brawnier hardware like the Nvidia P40 for older yet hefty models. Discussions around VRAM spill into strategies for optimizing multi-GPU setups, hinting at the use of both RTX 4060 Ti and GTX 1070 to spread the computational load, and leveraging **Tesla P40 GPUs** despite potential outdated **CUDA** woes.

**The Joy of Smoothly Running Models**: On both **ROCM** and **ROCm Preview Beta** fronts, GPU support discussion was rife, including the use of AMD’s RX 5000 and 6000 series chips. Users flagged the "exit 42" errors on **ROCm 0.2.19 Beta**, rallying around debug builds for a solution, displaying a communal spirit in action. Meanwhile, whispers of Intel’s **Advanced Matrix Extensions (AMX)** stirred speculation on how LM Studio could tap into such formidable processing prowess.

**Excavating Model Gems**: A surge in shared resources and models came through announcements, including **Starling-LM 7B**, **c4ai command r v01**, and **stable-code-instruct-3b**, among others. Accessibility stands upfront with a collective push towards a community page on **Hugging Face**, where the latest **GGUF quants** shine, luring AI enthusiasts to experiment with the offerings such as **Google's Gemma 1.1 2B**, and stay alert for the upcoming 7B variant.

**Sculpting the Vision Models Landscape**: A member's inquisition about training **LLMs** to decipher stock market **OHLC** patterns, amidst praise for **LM Studio’s** utility in vision model implementations, ignites a spark in exploring how the intricate dance between technology and finance could be choreographed with AI's grace. The revelation of **vision models on Hugging Face** mirrors the community’s camera-ready attitude to snapshot and subsequently transpose this conceptual aesthetic into practical applications.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

**Gradio's API Recorder and Chatbot UI Fixes Gear Up for Release**: Gradio version 4.26.0 introduces an **API Recorder** to translate interactions into code and addresses crucial bugs related to page load times and chatbot UI crashes. The update is detailed in the [Gradio Changelog](https://www.gradio.app/changelog#4-26-0).

**A Crescendo of Concern Over LLMs**: Security concerns gain spotlight as 'Crescendo', a new method that challenges the ethical restraints of LLMs, and vulnerabilities in Cohere's Command-R-plus are exposed. Meanwhile, Mixture-of-Depths (Modes) proposal and llamaindex blogs offer innovative solutions for model efficiency and information retrieval.

**NLP Community Finesse with SageMaker, Desire for PDF ChatGPT, and Sails Through Challenges**: The community debates deploying models on SageMaker, customizing ChatGPT for PDFs, and shares fascination over Gemini 1.5's 10M context window. Solution seekers confront multi-GPU training hiccups and demand token count information when using Hugging Face libraries.

**Thriving Repository of AI Contributions and Dialogues**: HybridAGI's neuro-symbolic behavior programming on GitHub welcomes peer review, and the Hugging Face reading group archives its collective wisdom [on GitHub](https://github.com/isamu-isozaki/huggingface-reading-group). PaperReplica's open-source invitation and RAG-enabled llamaindex shine as beacons of collaborative learning and resource sharing.

**Vision and Beyond**: Dialogues in the computer vision channel touch on the utility of HuggingFace as a model repository, efficacy of different Transformer models (e.g., XCLIP), and address real-time challenges using tools like the HuggingFace 'datasets' library for parquet file manipulation. Meanwhile, an open call for resources to apply diffusion models to video enhancement signifies the domain's vibrant investigative spirit.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

**Mojo Rising: A Dive into Special Functions and SICP Adaptation**
- The Mojo community is flexing its technical muscle, diving into specialized **mathematical functions** with an update to the Specials package and porting the famed "Structure and Interpretation of Computer Programs" (SICP) text to Mojo. Users can now find numerically accurate functions like `exp` and `log` in the [Specials package](https://github.com/leandrolcampos/specials) and participate in collaborative algorithm and package sharing via repositories such as [mojo-packages](https://github.com/kernhanda/mojo-packages).

**MAX Aligns with AWS; Open Source Documentation Drive**
- Modular announced a [strategic alliance with AWS](https://www.modular.com/blog/modular-partners-with-amazon-web-services-aws-to-bring-max-to-aws-services), intending to integrate MAX with AWS services and extend AI capabilities globally. The Mojo language is gearing up for enhanced collaboration with an appeal for community contributions to [Mojo's standard library documentation](https://www.modular.com/blog/how-to-contribute-to-mojo-standard-library-a-step-by-step-guide).

**Discord Dynamics: Python Interop and Contributing to Mojo's Growth**
- The Mojo community is actively engaging in discussions about metaprogramming capabilities, compile-time evaluation complexities, and lifetimes in the `Reference` types. They are exploring pathways to Python interoperability by implementing essential functions and are inviting contributors to jump in on "good first issues" on GitHub, offering a starting point with [Mojo's Changelog](https://docs.modular.com/mojo/changelog#week-of-2023-01-30) and [contribution guidelines](https://github.com/modularml/mojo/blob/main/CONTRIBUTING.md).

**Var vs. Let - the Mojo Parameter Saga**
- A conversation revealed that while `let` may have been removed from Mojo, `var` remains for lazily assigned variables with details in the [Mojo Manual](https://docs.modular.com/mojo/manual/variables#declared-variables), feeding further knowledge to users. Additionally, efforts are converging on infusing Mojo into web development, with the availability of [lightbug_http](https://github.com/saviorand/lightbug_http), reiterating Mojo's position as a comprehensive general-purpose language.

**Nightly Chronicles: From CPython Interop to Community Discussions**
- Members are celebrating advancements in CPython interoperability in Mojo and fostering an environment ripe for contributions, discussing best practices for signed-off commits in PRs, and sharing solutions for managing nightly builds and package updates. This proactive collaboration is paving the way for future open source contributions, signposted on GitHub, including anticipated discussions on the [Mojo Standard Library](https://github.com/modularml/mojo/discussions/2234).

**Blog Beats and Video Treats in Mojo's Creative Continuum**
- The launch of the [Joy of Mojo](https://joyofmojo.com/) website underscores the community's commitment to sharing creative experiences with Mojo, further amplified by GitHub repositories like [mojo-packages](https://github.com/kernhanda/mojo-packages) and enlightening videos on Mojo's Python interoperability, underscoring its dynamic evolution.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **WikiText's New Main Access Point**: Stephen Merity has [rehosted WikiText](https://state.smerity.com/smerity/state/01HRTB51QZMG59MDAX2ME1TCHR) on Cloudflare R2, offering a larger dataset while maintaining original formatting, which is important for training language models with authentic data structures.

- **Perplexing Perplexity Scores**: A debate emerged about the validity of perplexity scores reported by the GateLoop Transformer author, with lucidrains unable to replicate them, prompting discussions over result reproduction and transparency in reporting.

- **Hugging Face's Automatic Parquet Conversion Frustration**: Users expressed frustration at Hugging Face's autoconversion of datasets to parquet format, which can cause confusion and issues, such as with `.raw` files; a workaround involves hosting datasets using Git LFS.

- **Documentation Ephemera and Reproducibility Emphasis**: OpenAI's fluctuating documentation on models, with some links being removed, underscores the importance of reliable resources like [archived pages](https://archive.ph/n5xMq) for consistency in the AI research community. Simultaneously, there's a push for reproducible data formats, as shown by community efforts to mirror datasets like WikiText on platforms such as Cloudflare R2.

- **Optimizer Optimization and Zero-Shot Innovations**: Conversations coalesced around the Schedule-Free optimizer and its capacity to estimate optimal learning rates, as well as intriguing methods for teaching language models to search using a stream of search (SoS) language. Moreover, the connection between emergent abilities in language models and exposure to long-tail data during training was a focal topic, with implications for zero-shot task performance.

- **Stars Matter for NSF Reviews**: The number of GitHub stars for [nnsight](https://github.com/ndif-team/nnsight) was highlighted by an NSF reviewer as a metric of concern, illustrating the unconventional impact of community engagement on research funding perspectives.

- **GPU Utilization and BigBench Task Recognition**: Analysis of GPU utilization led to reduction in evaluation times by using `batch size=auto`, revealing potential underutilization issues. Members also navigated confusion around BigBench tasks, suggesting verification of task variants using `lm_eval —tasks list`.

- **CLI Command Conundrums and Logit Bias Discussions**: Technical discussions flourished around the `—predict_only` CLI command issues and the non-effect of OpenAI's `logit_bias` as expected on logits during one-token MCQA tasks, leading to exploration of alternative approaches such as `greedy_until`. Temperature settings and their effects on outputs were clarified, highlighting the importance of correct `gen_kwargs` settings for achieving desired model behavior.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Translation Showdown: GPT-4 vs DeepL**: GPT-4's translation capabilities were compared to DeepL, highlighting that while **DeepL excels in contextual language translation**, GPT-4 sometimes falls short on nuancing basic contexts.
- **AI Models in Code Generation Face-Off**: **Opus** and **GPT-4** received praise for impressive performance in code generation tasks, but GPT-4 also showed potential issues when processing larger contexts compared to other models.
- **Decoding AI Consciousness**: A lively exploration into simulating human consciousness with AI involved equating human neurochemical activities with GPT's programming mechanisms, sparking debates on consciousness's origins and AI's role in its depiction.
- **Prompt Engineering for Sensitive Content**: Writers discussed circumventing ChatGPT's content policy to develop backstories for characters with traumatic histories, seeking subtler ways to infuse nuanced, sensitive details into their narratives.
- **Building AI-Powered Games**: Engineers suggested utilizing JSON for structuring game progress data while discussing the challenge of crafting seamless game experiences that keep underlying code concealed from players.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

**Claude 3 Takes on Images**: The **Claude 3 models** have been updated to multimodal, now supporting image input, requiring developers to modify existing codebases accordingly.

**AI Goes Old School with Rock, Paper, Scissors**: A new game at [blust.ai](https://rock.blust.ai), where players can challenge ChatGPT to a classic round of Rock, Paper, Scissors.

**Frontends and Favorites Front and Center**: Engineers discussed various OpenRouter API frontends like [LibreChat](https://librechat.ai/), [SillyTavern](https://sillytavern.com/), and [Jan.ai](https://jan.ai/docs/remote-inference/router). Command-R+ has emerged as a favored model for coding tasks and interactions in Turkish, while concerns are raised about content censorship in models.

**Performance Insights in Modeling**: Conversations highlighted that Sonnet outstrips Opus in coding tasks, and Claude 3 is superior in PDF data extraction compared to Gemini Pro 1.5, which prompted some skepticism about its utility.

**Model Efficacy Metrics Spark Debate**: The community has voiced that model ranking based solely on usage statistics might not accurately reflect a model's worth, suggesting spending or retention as potential alternate measures.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

**Revving Up RAG Applications**: Marker-Inc-Korea introduced **AutoRAG**, an automated tool for tuning **RAG pipelines** to enhance performance, detailed and linked in their [tweet](https://t.co/ZndrM36n61). Meanwhile, `create-llama` was released to streamline the launch of full-stack RAG/agent applications, as announced in its [tweet](https://t.co/YOEZUQt7Lr).

**Tweaking Sales Pitches with AI**: A new application using **RAG** to create personalized sales emails was featured in a recent webinar, ditching hard-coded templates with an LLM-powered approach, further info available in a [tweet](https://t.co/kV7MGJ6PqS).

**Deep Diving Into Documents**: Andy Singal presented on multi-document agents that handle complex QA across numerous sources. The aim is to expand this functionality for more intricate inquiries, shared in a [presentation tweet](https://t.co/3yKuv2qDDf).

**Metadata to the Rescue for Document Queries**: To get page numbers and document references from multi-document queries, make sure to include this metadata before indexing, allowing retrieval of detailed references post-query.

**Optimization Overhaul for Azure and Embedding Times**: Participants noted issues with Azure's OpenAI not recognizing context and discussed using batching methods for faster embedding generation. Regarding challenges with **ReAct** agents and open-source models like "llama2" and "mistral", better router descriptions may improve model-routing performance.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

**Mistral Needs Muscle**: **Mistral 7B Instruct v0.2** has been acknowledged as high-performing, yet it demands substantial resources—expect to allocate at least 16GB of RAM and have some GPU support for smooth operation.

**Challenges with Python Compatibility**: There's a community consensus to stick with **Python <=3.10** to avoid issues with **TTS packages**, with repeated suggestions to avoid using **Python 3.11.4** for setups dependent on voice command recognition.

**A Call for Better Documentation**: Inquiries about local vision models and calls highlighting the need for more comprehensive examples and documentation in the **Open Interpreter's cookbook** reveal gaps that are yet to be filled.

**Efficiency Over Expense with Local Models**: The costliness of **GPT-4** has prompted discussions around leveraging local models such as **Hermes 7B** and **Haiku**—less expensive yet slightly less refined alternatives offering privacy and lower operating costs.

**Hardware Hang-Ups and Software Setbacks**: The **O1** community reported hardware issues, particularly with external push-button integration, and software setup challenges when installing on Windows, with tweaks including using **chocolatey**, **virtualenv**, and specific **environment variables** being part of the troubleshooting dialogue.

Relevant resources and conversations are threaded throughout the community, with direct engagement on issues being tracked on platforms like [GitHub](https://github.com/OpenInterpreter).



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **GitHub Grievances**: A user requested assistance with a **[Pull Request](https://github.com/langchain-ai/langchain/pull/19751)** that was failing due to a "module not found" error related to "openapi-pydantic," even though the module was included in dependencies. This highlights dependency management as a notable pain point in the community.
  
- **Fine-Tuning Finesse Without the GPU Muscle**: Queries about training and fine-tuning language models sans GPU led to recommendations for tools like **Google Colab** and the mention of **ludwig.ai** as viable options, indicating an area of interest among engineers seeking cost-effective computing resources.

- **Visual Visions via Artful AI's Update**: The announcement of **Artful AI's** new models, including **Dalle Creative, Anime Dream, & Epic Realism**, released on the **[Google Play Store](https://play.google.com/store/apps/details?id=com.projecthit.artful&referrer=ph-aai)**, piqued the community’s interest in the evolving domain of AI-driven image generation.

- **Security Spotlight on AISploit**: The introduction of **AISploit**, available on **[GitHub](https://github.com/hupe1980/aisploit)**, sparked discussions on leveraging AI for offensive security simulations, indicating a tactical pivot in the use of AI technologies in cybersecurity.

- **TypeScript and Text Chunking Techniques Revealed**: The share of a **[TypeScript Gist](https://gist.github.com/tsensei/3b6589662271874b5055d79473932aae)** that demonstrated breaking up large text into semantic chunks using OpenAI's sentence embedding service exemplified community engagement in developing and sharing tools for enhanced text processing workflows.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

**Apple's AI Ambitions Under Scrutiny**: Apple is criticized for the subpar performance of **Metal Performance Shaders** (MPS) and **torch compile**, even as recent merges aim to fix **MPS** issues in the PyTorch nightly branch. Community experiences with **torch.compile** vary, reflecting ongoing optimizations needed for Apple's platforms.

**Copyright Conundrum**: AI's use of copyrighted content for creating derivative works sparks legal debate, with consensus on the insufficiency of paraphrasing to avoid infringement. The community anticipates the need for substantial legal changes to accommodate new AI training data practices.

**The Harmony of AI-Composed Music**: Discussions about AI-generated music, involving companies like **Suno** and Nvidia, recognized rapid advancements but also forecasted potential legal spats with the music industry. Members also noted the less impressive progress in text-to-speech (TTS) technology compared to AI's leap in music generation.

**AI Career Dynamics Shifting**: The rise of freelance AI-related careers due to technological progress is noted, with resources like **Bloomberry's analysis** cited. **Stability AI's CosXL model** release sparks conversations about the efficacy of **EDM schedules** and **offset noise** in model training.

**Novelties in AI Research Techniques**: A new paper on **transformers** shows computational resource allocation can be dynamic, **DARE's pruning** technique for language models hints at preservable capabilities, and **BrushNet** introduces enhanced AI inpainting. Latent diffusion for text generation, referenced from a NeurIPS paper, indicates a potential shift in generative model techniques.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **GPT Models Tackle the A::B Challenge**: Victor Taelin conceded that GPT structures could indeed address certain problem-solving tasks, including long-term reasoning, after a participant utilized GPT to solve the A::B problem with a near 100% success rate, winning a $10k prize. [Victor Taelin's statement on the outcome is available online](https://x.com/victortaelin/status/1777049193489572064?s=46&t=6FDPaNxZcbSsELal6Sv7Ug).

- **Stanford Debuts Language Modeling Course CS336**: Stanford is offering a new course, CS336, which delves into the nuts and bolts of language modeling, including insights on Transformers and LLMs, garnering significant interest from the community eager for the release of lecture recordings.

- **Groq Plans to Topple AI Hardware Rivals**: The AI hardware startup Groq, led by a founder with an unconventional educational background, aims to outdo all existing inference capacity providers combined by next year and asserts their developers enjoy reduced inference costs and speedier hardware in comparison to NVIDIA's offerings.

- **Introducing LangChain's Memory Service**: LangChain's latest alpha release brings a memory service aiming to upgrade chatbot interactions by automatically condensing and refining conversations, with [resources posted for quick start](https://langchain-ai.github.io/long-term-memory/).

- **Peer Learning in AI Tools and Knowledge Management**: Engineers exchanged resources and strategies for curating personal and organizational knowledge using AI tools, such as incorporating [Obsidian-Copilot](https://github.com/logancyang/obsidian-copilot) and [fabric](https://github.com/danielmiessler/fabric), and discussed the development of integrations to enhance tools like ChatGPT within knowledge systems.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Quantized DoRA Available, Dance of the LoRA**: The latest release of `peft=0.10.0` supports **quantized DoRA**, prompting suggestions to update **axolotl's** `requirements.txt` ([PEFT's release notes](https://github.com/huggingface/peft/releases/tag/v0.10.0)). The **advanced optimizers** from Facebook Research have now been integrated into Hugging Face's transformers library, with **Schedule-Free Learning** open-sourced and specific parameter recommendations of `0.0025` for ScheduleFreeAdamW ([Hugging Face PR #30079](https://github.com/huggingface/transformers/pull/30079)).

- **Model Generation Hiccup**: Users reported and discussed an error occurring in the generation process with a **fine-tuned Mistral 7b model** using **fp16**, specifically after a few successful generations resulting in `_queue.Empty`.

- **Rotary Queries and Sharding Insights**: The parameter `"rope_theta": 10000.0` came under scrutiny, relating to **Rotary Positional Embedding**. Meanwhile, The **FSDP configuration** for Mistral was shared with details on how the `MixtralSparseMoeBlock` class should be utilized ([mixtral-qlora-fsdp.yml](https://github.com/openaccess-ai-collective/axolotl/tree/main/examples/mistral/mixtral-qlora-fsdp.yml#L1L75)).

- **Seek and You Shall Find: LISA and Configs**: Queries arose about the location and absence of **LISA** parameters in documentation, later resolved with the discovery of the [LISA configuration file](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/examples/llama-2/lisa.yml). Members also engaged in technical discussions on handling optimizer states for **unfreezing new layers during training**.

- **Model Training Conundrums Solved**: The community solved various challenges including **training with raw text**, adapting **Alpaca instruction sets**, differentiating **micro batch size** and **batch size**, and adjusting configurations to disable checkpoints and evaluations or handling special tokens.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

**Podcasting Gold: John Schulman to Possibly Feature on Show**: Nathan Lambert is considering featuring John Schulman in a podcast, a move that stirred excitement among members. Moreover, a licensing change for **text-generation-inference (TGI)** to Apache 2 has spurred a significant increase in contributors to the open-source project.

**Memes Channel Maintains Light-Heartedness**: The memes channel included joking references to targetings without context, improvements in experiences, and confirmation of employment status, indicating a casual, light-hearted discourse among members.

**Open AI Weights Debate Hits Engaged Nerve**: The #reads channel had a vibrant discussion on the societal impacts of open foundation models, with a focus on safety thresholds, regulation feasibility, and AI's potential to manipulate societal processes. A shared visualization of Transformer attention mechanisms and speculation about future models that emphasize verification instead of generation were among the in-depth topics discussed.

**Bridging the Knowledge Gaps with Visuals**: The #sp2024-history-of-open-alignment channel discussed effective resources like [lmsys](https://lmsys.deepai.org/) and [alpacaeval leaderboard](https://alpacaeval.com/) to find state-of-the-art models. Additionally, an intent to visually categorize models for better comprehension was expressed, along with sharing a live document ([Google Slides presentation](https://docs.google.com/presentation/d/1quMyI4BAx4rvcDfk8jjv063bmHg4RxZd9mhQloXpMn0/edit?usp=sharing)) for an upcoming alignment talk and a guide ([comprehensive spreadsheet](https://docs.google.com/spreadsheets/d/1gc6yse74XCwBx028HV_cvdxwXkmXejVjkO-Mz2uwE0k/edit#gid=0)) on open models by Xeophon.

**A Note on AI Generated Music**: Nathan noted the impressive quality of a new contender in AI music generation, posing a potential challenge to the Suno AI platform.



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **Fast Track to Tokenization**: Engineers discussed speeding up tokenization using Huggingface's fast tokenizer for *c4-en*, exploring options like increasing threads or utilizing more capable machines.

- **Open Source GPU**: AMD announced the open sourcing of its Micro Engine Scheduler (MES) firmware for Radeon GPUs, a decision celebrated within the community and praised by entities like George Hotz's Tiny Corp. ([The Register Article](https://www.theregister.com/2024/04/05/amd_mes_open_source/), [AMD Radeon Tweet](https://twitter.com/amdradeon/status/1775999856420536532)).

- **Paper Trail**: An open-source repository, [PaperReplica GitHub Repo](https://github.com/hegdeadithyak/PaperReplica), for replicating research papers in AI & ML got its unveiling, inviting community contributions and GitHub stars.

- **CUDA Conundrums and Triton Strategizing**: From setting up CUDA environments on Ubuntu to appreciating libraries that boost proficiency with Triton, members exchanged tips and troubles. In particular, a lean GPT-2 training implementation by Andrej Karpathy in C was highlighted for its efficiency without the heft of PyTorch or Python ([GitHub](https://github.com/karpathy/llm.c)).

- **DeepSpeed in the Fast Lane**: Conversations revolved around practical applications of DeepSpeed, integration with Hugging Face's Accelerate, and memory optimization wonders even at zero stage. Additionally, use of Triton kernels was noted to be conditional on CUDA device input, and a curiosity about optimizing transformer MLPs with cublas or tiny-cuda-nn was shared ([tiny-cuda-nn Documentation](https://github.com/NVlabs/tiny-cuda-nn/blob/master/DOCUMENTATION.md)).

- **Quantum of Solace for LLMs**: A novel quantization approach, [QuaRot](https://arxiv.org/abs/2404.00456), was mooted for its capability to quantize LLMs to 4 bits effectively, while a revelatory tweet hinted at schedule-free optimization, potentially indicating a move away from the traditional learning rate schedules ([Twitter](https://twitter.com/aaron_defazio/status/1773381393831067787)).

- **Vexed by Visualizing Triton**: Engineers delved into the challenges and opportunities in visualizing Triton code, from shared memory to tensor views, and from CPU constructs to enhancing JavaScript interactivity, signaling a continuing quest for more user-friendly debugging tools.

- **Calendar Confusion Cleared**: A small timezone clarification was sought for a ring attention session, hinting at the vibrancy of the community's relentless pursuit of knowledge and optimization. 

- **Of Numbers and Neurons**: The value of precise quantization methods surfaced, highlighting the importance of accurate tensor transformations and the potential performance gains leveraged from tools like Triton, indicating a keen focus on efficiency within machine learning pipelines.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

**Tinygrad Takes a Step Back**: George Hotz has reverted the command queue in tinygrad and is opting to integrate the memory scheduler directly with the current scheduler model. This approach utilizes the multidevicegraph abstraction already in place, as discussed [here](https://github.com/tinygrad/tinygrad/pull/4094).

**TinyJIT Under the Microscope**: The TinyJit tutorial has been released, although it may contain inaccuracies, particularly with the `apply_graph_to_jit` function, and users are encouraged to submit pull requests for corrections [TinyJit Tutorial](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/jit.md).

**Tinygrad Learning Expanded**: A collection of tutorials and guides for contributing to tinygrad are now available with a focus on topics like multi GPU training [Multi GPU Training Guide](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/multigpu.md).

**Discord Roles Reflect Contribution**: George Hotz redesigned roles within the tinygrad Discord to better reflect community engagement and contribution levels, reinforcing the value of collaboration and respect for others' time.

**Unpacking MEC's Firmware Mystery**: Discussions about MEC firmware's opcode architectures emerged with speculation on RISC-V and different instruction sets, revealing a potential `cbz` instruction and inclusive dialogue around the nuances of RISC-V ISA.



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

**Scan Reveals Llamafile's Wrongful Accusation**: Versions of **llamafile**, including **llamafile-0.6.2.exe** and **llamafile-0.7**, were flagged as malware by antivirus software; utilizing appeal forms with the respective antivirus companies was suggested as a remedial step.

**Run Llamafile Smoother in Kaggle**: Users encountering issues when running `llamafile` on Kaggle found solace through an **updated command** that resolves CUDA compilation and compatible GPU architecture concerns, enabling efficient usage of **llamafile-0.7**.

**RAG-LLM Gets Local Legs**: A query about locally distributing RAG-LLM application without the burdens of Docker or Python was answered affirmatively, indicating the suitability of **llamafile** for such purposes, particularly beneficial for macOS audiences.

**Taming the Memory Beast with an Argument**: An **out of memory error** experienced by a user was rectified by adjusting the `-ngl` parameter, demonstrating the importance of fine-tuning arguments based on the specific capabilities of their NVIDIA GeForce GTX 1050 card.

**Vulkan Integration Spurs Performance Gains**: A proposition to bolster **llamafile** by integrating Vulkan support led to performance enhancements on an Intel-based laptop with an integrated GPU, yet this required the granular task of re-importing and amending the **llama.cpp** file.



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **No Schedules Needed for New Optimizers**: The [huggingface/transformers repository](https://github.com/huggingface/transformers/pull/30079) now has a pull request introducing Meta's *schedule-free optimizers* for AdamW and SGD, which promises substantial enhancements in model training routines.
- **AI Devs Convene in Hürth**: An AI community event focusing on synthetic data generation, LLM/RAG pipelines, and embeddings is scheduled for May 7th in Hürth, Germany. Registration is open, with emphasis on a hands-on, developer-centric format, and can be found at [Developer Event - AI Village](https://www.eventbrite.de/e/developer-event-ai-village-tickets-868896702427).
- **Sharing Synthetic Data Insights Sought**: Demand for knowledge on synthetic data strategies is high, with specific interest in the quality of German translated versus German generated data, indicating a niche requirement for regional data handling expertise.
- **Command-R Tackles Tough Translations**: The Command-R model showcased on [Hugging Face Spaces](https://huggingface.co/spaces/CohereForAI/c4ai-command-r-plus) excels at translating archaic Middle High German text, outperforming GPT-4 equivalents and underscoring the potential upheaval in historical language processing.
- **Open-Source Model Development Desired**: There's anticipatory buzz that an open-source release of the impressive Command-R could amplify developer engagement, echoing the ecosystem success seen with publicly accessible models like Mistral.



---



## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **Slow and Steady Wins the Race?**: Comparisons reveal that **Jamba's** 1B Mamba model lags in training speed by *76%* when run on an HGX, compared to a standard Transformer model.

- **Size Doesn't Always Matter**: Engineers have introduced scaled-down **Jamba** models, [8xMoE with 29B](https://huggingface.co/isemmanuelolowe/Jamba-8xMoE_Slerp) and [4xMoE with 17.7B](https://huggingface.co/isemmanuelolowe/Jamba-4xMoE_Slerp) parameters, achieving decent performance on hardware as accessible as a 4090 GPU at 4 bit.

- **Weights and Measures**: A creator's application of spherical linear interpolation (Slerp) for expert weight reduction in **Jamba** models sparked interest, with plans to share a notebook detailing the process.

- **Power Play**: In the quest for optimal GPU utilization while handling a 52B **Jamba** model, one engineer seeks more efficient methods for training, likely considering a switch from pipeline to Tensor Parallelism given current capacity constraints.

- **What's the Best Model Serving Approach?**: The community is engaging in conversations about effective **inference engines** for **Jamba** models, though no consensus has been reached yet.



---



## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

- **QNAP NAS - A Home Lab for AI Enthusiasts**: An AI engineer shared a [guide](https://www.storagereview.com/review/run-a-private-rag-chatgpt-on-qnap-nas) about setting up a **QNAP NAS** (model TS-h1290FX) as an AI testing platform, emphasizing its notable specs such as an AMD EPYC 7302P CPU, 256GB DRAM, and 25GbE networking.

- **Streamlining AI with Preset Prompts**: There's curiosity among engineers about storing and reusing system prompts to improve efficiency in AI interactions, although the discussion did not progress with more detailed insights or experiences.

- **Alter: The Mac's AI Writing Assistant**: **Alter** is launching in beta, offering AI-powered text improvement services to macOS users, capable of integrating with applications like Keynote, as showcased in [this demonstration video](https://youtu.be/IK53CSSbaqI).

- **A Singular AI Solution for Mac Enthusiasts**: The Alter app aims to provide context-aware AI features across all macOS applications, potentially centralizing AI tools and reducing the need for multiple services. Details about its full capabilities are available on the [Alter website](https://alterhq.com).



---



## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord

- **Dynamic Compute Allocation Sparks Ideas**: Engineers discussed a paper proposing **dynamic allocation of compute resources on a per-token basis** within neural networks, which stirred interest for possible adaptations in **neurallambda**; the aim is to allow the network to self-regulate its computational efforts.
- **Rethinking Training Approaches for neurallambda**: Exploratory talks included using **pause/think tokens**, **reinforcement learning for conditionals**, and emulating aspects of RNNs that adaptively control their compute usage, which could enhance training efficacy for **neurallambda**.
- **Innovative Input Handling on the Horizon**: Technologists considered novel input approaches for **neurallambda**, like using a neural queue for more flexible processing and conceptualizing input as a Turing machine-esque tape, where the network could initiate tape movements.
- **Improving LLMs Data Structuring Capabilities**: Participants shared an instructional video titled "Instructor, Generating Structure from LLMs", showing methods to extract structured data such as JSON from LLMs like **GPT-3.5, GPT-4**, and **GPT-4-Vision**, aimed at getting more reliable results from these models. [Watch the instructional video](https://www.youtube.com/watch?v=KxOqjKq2VyY).
- **Video Learning Opportunities**: A second educational video was linked, however, it was provided without context, suggesting a potential resource for self-guided learning for the curious. [Explore the video](https://www.youtube.com/watch?v=keUjQyWmgY0).



---



## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord

- **Haiku Performance Tuning Search**: A guild member is seeking advice on improving the speed of **Haiku** due to dissatisfaction with its current throughput.

- **Anthropic's API Outperforms GPT-4 Turbo**: A user presented evidence that **Anthropic’s** beta API surpassed **GPT-4 Turbo** in numerous tests on the Berkeley function calling benchmark. Results from this study can be found in a [detailed Twitter thread](https://x.com/JoschkaBraun/status/1777381282751688868).



---


The **Alignment Lab AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links



**Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1225920372562591825)** (1 messages): 

- **New Story Discovery Experience on iOS**: Perplexity is testing a new format for story discovery in its iOS app. Feedback is welcomed in the designated channel; get the app [here](https://pplx.ai/download).
  

---


**Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1225742672845606993)** (1199 messages🔥🔥🔥): 

- **Event Ends with a Draw**: The Perplexity AI Discord event concluded with users *eun08000* and *codelicious* tied for top place. Winners will be contacted via DMs for their prizes.

- **Differences in Claude 3 Opus**: Users discussed differences in the Claude 3 Opus model between Poe and Perplexity, noting performance variations, particularly in creativity and writing tasks.

- **Solar Eclipse Excitement**: Members of the server shared their anticipation and observations of the solar eclipse, with conversations including the ideal viewing equipment and experiences of witnessing the phenomenon.

- **Questions on Moon Formation**: A discussion arose about the formation of the Moon, with one user skeptical about the theory of the Moon being part of a celestial body that collided with Earth. Links to educational resources were shared for further understanding.

- **Getting Pro Role on Discord**: Users inquired about obtaining the 'Pro' role on the Discord server, with a direction to rejoin via a Pro Discord link provided in the account settings on the Perplexity website.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/OscaR_010__/status/1776969765635961068?t=lvoBPNlllBK_dMSAxHr16w&s=33">Tweet from OscaR-_-010 (@OscaR_010__)</a>: @kodjima33 @bing @OpenAI @perplexity_ai @AnthropicAI Hi, friend . Here I do have to say that in searches @perplexity_ai surpasses all those mentioned, its search range is superior. The search includes...</li><li><a href="https://www.rabbit.tech/">rabbit</a>: $199 no subscription required - the future of human-machine interface - pre-order now</li><li><a href="https://docs.perplexity.ai/docs/model-cards">Supported Models</a>: no description found</li><li><a href="https://support.stripe.com/questions/impact-of-sanctions-on-russia-and-belarus?locale=en-US">Impact of sanctions on Russia and Belarus : Stripe: Help &amp; Support</a>: no description found</li><li><a href="https://tenor.com/view/cat-cat-memes-cat-images-cat-meme-gif-4644773688486402896">Cat Cat Memes GIF - Cat Cat memes Cat images - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/space-gif-25736952">Space GIF - Space - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/queen-freddie-mercury-we-are-the-champions-champion-sing-gif-4654136">Queen - Champion GIF - Queen Freddie Mercury We Are The Champions - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://gs.statcounter.com/vendor-market-share/mobile">Mobile Vendor Market Share Worldwide | Statcounter Global Stats</a>: This graph shows the market share of mobile vendors worldwide based on over 5 billion monthly page views.</li><li><a href="https://support.privacy.com/hc/en-us/articles/360050917053-Can-I-use-Privacy-if-I-live-outside-the-US">Can I use Privacy if I live outside the US?</a>: We are only able to provide our service to US citizens or legal residents of the US at this time.  We're continuing to explore opportunities and options to bring Privacy to the rest of the world. H...</li><li><a href="https://gs.statcounter.com/os-market-share/mobile/worldwide">Mobile Operating System Market Share Worldwide | Statcounter Global Stats</a>: This graph shows the market share of mobile operating systems worldwide based on over 5 billion monthly page views.</li><li><a href="https://tenor.com/view/unlimited-power-star-wars-gif-10270127">Unlimited Power Star Wars GIF - Unlimited Power Star Wars - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://labs.mojeek.com/rag/index.html">Mojeek Labs | RAG Search</a>: no description found</li><li><a href="https://docs.perplexity.ai/discuss/65d956e39db34f001ff8ce0a">Are Sonar models new?</a>: no description found</li><li><a href="https://docs.perplexity.ai/discuss/6582f98b41714c00723d5d5c">The difference between the models on the PPL website and the API models.</a>: no description found</li><li><a href="https://docs.perplexity.ai/discuss/6601ffd6bd5f0e0045ac5d16">Model names?</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=gCkZmADecL0">Solar Eclipse LIVE Coverage (with Video &amp; Updates)</a>: Join us for live coverage of the solar eclipse, featuring live eclipse video! We’ll show you the total solar eclipse live in Mexico, the United States, and C...</li><li><a href="https://youtu.be/wkQuOrsgVGY?si=qrl5Bdx_Mr4L-f6_&t=2603">Eight Wonders Of Our Solar System | The Planets | BBC Earth Science</a>: Discover the most memorable events in the history of our solar system. Travel to the surface of these dynamic worlds to witness the moments of high drama tha...</li><li><a href="https://www.star.nesdis.noaa.gov/GOES/conus_band.php?sat=G16&band=GEOCOLOR&length=24">GOES-East CONUS - GeoColor - NOAA / NESDIS / STAR</a>: no description found
</li>
</ul>

</div>
  

---


**Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1225901847470932038)** (40 messages🔥): 

- **Exploring the AI Frontier**: Users shared a plethora of search queries leading to Perplexity AI's platform, covering topics from the Samsung Galaxy S23 to the impacts of AI on the music industry.
- **Tech Giants Making Moves**: A link to a YouTube video discussing Tesla's robotaxi announcement and Apple's home robot project was shared, highlighting the advancements and rumors in technology sectors.
- **Interactivity Reminders**: Several reminders for users were posted, urging them to ensure their threads are shareable, indicated by specific instructions and attachment links.
- **Featured Success Story**: Perplexity AI's efficiency in model training was showcased in an Amazon Web Services case study, presenting significant reductions in training times and enhanced user experience.
- **Educational Insights**: Links to knowledge resources on various subjects such as color basics, the origins of geometric proofs, and SpaceX's Mars plans were provided, reflecting a diverse interest in learning and self-improvement.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://aws.amazon.com/solutions/case-studies/perplexity-case-study/">Perplexity Accelerates Foundation Model Training by 40% with Amazon SageMaker HyperPod | Perplexity Case Study | AWS</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=JAuKnXSn70s">Tesla robotaxi announcement, Alphabet-HubSpot deal rumors, Apple’s home robot project</a>: In today&#39;s episode of Discover Daily by Perplexity, we dive into Tesla&#39;s upcoming robotaxi unveiling, set for August 8th, and explore what this autonomous ve...</li><li><a href="https://www.youtube.com/watch?v=yGejxO1xYmo">Workflows &amp; Tooling to Create Trusted AI | Ask More of AI with Clara Shih</a>: Clara sits down with the founder/CEOs of three of the hottest AI companies-- Aravind Srinivas (Perplexity AI), Jerry Liu (LlamaIndex), and Harrison Chase (La...
</li>
</ul>

</div>
  

---


**Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1225764813393629244)** (40 messages🔥): 

- **API Credit Purchase Difficulties**: Members are experiencing issues when attempting to purchase API credits; the balance appears as $0 after a refresh despite trying multiple times. **ok.alex** requests affected users to send account details for resolution.
- **Discrepancy Between pplx-labs, API, and Web App**: Users have reported inconsistencies in results when using the same prompts across the pplx-labs, API, and web application, with the API showing more hallucinations. **icelavaman** informed that the default model from [pplx.ai](https://pplx.ai) is not available via the API, and citations are currently in closed beta.
- **Ruby Wrapper for Perplexity API in Progress**: **filterse7en** is developing an OpenAI-based Ruby wrapper library for the Perplexity API.
- **API vs Web App Model Differences**: Discussions reveal differences between results from the API and the web application, with skepticism around results quality and the presence of hallucinations. **brknclock1215** suggests that using the `sonar-medium-online` model via the API should effectively be the same as the "Sonar" model on the web version without Pro.
- **Inquiries on pplx-pro Model API Access**: **marciano** inquired if the model used in pplx-pro is accessible via the API. **ok.alex** clarified that Pro search is only available on the web and their apps, not via the API.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://api.perplexity.ai")">no title found</a>: no description found</li><li><a href="https://docs.perplexity.ai/docs/model-cards">Supported Models</a>: no description found
</li>
</ul>

</div>
  

---



**Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1225759346894442506)** (15 messages🔥): 

- **Introducing Command R+**: A YouTube video titled "Introducing Command R+: A Scalable LLM Built for Business" has been shared, showcasing [cohere's powerful LLM](https://www.youtube.com/watch?v=keUjQyWmgY0) specifically built for enterprise applications.
- **Overflow of AI Research**: A member expressed concern asking if more AI researchers are needed, with one member suggesting there's already more research than what can be digested, while another member pointed out the need for more meta-researchers to synthesize and interpret the influx of information.
- **Search Images in a Snap**: The project 'Where's My Pic?' was introduced, offering a solution similar to Google Image Search for local folders, which can be a time-saver for locating images quickly. Learn more about the project in this [YouTube video](https://www.youtube.com/watch?v=oVJsJ0e6jWk).
- **Canada's AI Strategy**: An announcement of Canada's ambition to be at the forefront of AI, including creating good-paying job opportunities in innovation and technology, was highlighted through a [government news release](https://www.pm.gc.ca/en/news/news-releases/2024/04/07/securing-canadas-ai-advantage).
- **Hugging Face Tech Insights**: The SSL certificates of huggingface.tech have been analyzed, providing insights into the tools they use, as detailed on [crt.sh](https://crt.sh/?q=huggingface.tech).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=keUjQyWmgY0">Introducing Command R+: A Scalable LLM Built for Business</a>: Today, we will take a look at the  Command R+, cohere&#39;s most powerful, scalable large language model (LLM) purpose-built to excel at real-world enterprise us...</li><li><a href="https://www.pm.gc.ca/en/news/news-releases/2024/04/07/securing-canadas-ai-advantage">Securing Canada’s AI advantage</a>: no description found</li><li><a href="https://crt.sh/?q=huggingface.tech">crt.sh | huggingface.tech</a>: no description found
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1225822258279354479)** (49 messages🔥): 

- **Rohan Paul's AI Tweet Sparks Curiosity**: A tweet by Rohan Paul regarding AI was revisited, noting its promising earlier impression but lacking follow-up information and insights after three months. The discussion also touched upon the usability of **fp8** on **NVIDIA's 4090 GPUs**.
  
- **LLaMA-2-7B Breaks the Context Length**: A groundbreaking achievement was shared where **LLaMA-2-7B** was trained to handle a massive **700K context length** using just 8 **A100 GPUs**, significantly surpassing the expected capacity of 32K to 200K tokens.

- **Gemma 1.1 Joins the AI Language Model Family**: Google released **Gemma 1.1 7B (IT)**, an instructive language model, on Hugging Face, boasting improvements in quality, coding capabilities, and instruction following. It was highlighted for its novel **RLHF method** used during training.

- **Pathfinding in Mazes Takes a Twist**: A unique approach to unifying physics was proposed using conjectural frameworks like the Fibonacci binomial conjecture, suggesting that **NLP** can simulate any process.

- **GPT-4 Takes a Meta Turn**: An experience with GPT-4 was shared where a given prompt led to unexpectedly meta and self-referential content. The discussion on this intriguing performance included a **YouTube link** to a relevant game's narrator feature.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/PY_Z001/status/1776176932687892796">Tweet from Zhang Peiyuan (@PY_Z001)</a>: 🌟700K context with 8 GPUs🌟 How many tokens do you think one can put in a single context during training, with 8 A100, for a 7B transformer? 32K? 64K? 200K? No, my dear friend.  I just managed to tra...</li><li><a href="https://arxiv.org/abs/2305.14078">Large Language Models as Commonsense Knowledge for Large-Scale Task Planning</a>: Large-scale task planning is a major challenge. Recent work exploits large language models (LLMs) directly as a policy and shows surprisingly interesting results. This paper shows that LLMs provide a ...</li><li><a href="https://outlines-dev.github.io/outlines/cookbook/classification/">Classification - Outlines 〰️</a>: Structured text generation with LLMs</li><li><a href="https://huggingface.co/papers/2402.14083">Paper page - Beyond A*: Better Planning with Transformers via Search Dynamics
  Bootstrapping</a>: no description found</li><li><a href="https://huggingface.co/google/gemma-1.1-7b-it">google/gemma-1.1-7b-it · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/papers/2404.03715">Paper page - Direct Nash Optimization: Teaching Language Models to Self-Improve with
  General Preferences</a>: no description found</li><li><a href="https://www.factorialfunds.com/blog/under-the-hood-how-openai-s-sora-model-works">Factorial Funds | Under The Hood: How OpenAI&#039;s Sora Model Works</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=eMlx5fFNoYc">Visualizing Attention, a Transformer&#39;s Heart | Chapter 6, Deep Learning</a>: Demystifying attention, the key mechanism inside transformers and LLMs.Instead of sponsored ad reads, these lessons are funded directly by viewers: https://3...</li><li><a href="https://www.youtube.com/watch?v=6RTkUgov60g&ab_channel=GameplayDump">Bastion: Narrator Bits Part 1 (Wharf District, Workmen Ward, Breaker Barracks)</a>: I recorded the game with narrator audio only and everything else turned down to zero in the sound menu volume settings.   Then I just cut out the silent part...</li><li><a href="https://github.com/vicgalle/configurable-safety-tuning">GitHub - vicgalle/configurable-safety-tuning: Data and models for the paper &quot;Configurable Safety Tuning of Language Models with Synthetic Preference Data&quot;</a>: Data and models for the paper &quot;Configurable Safety Tuning of Language Models with Synthetic Preference Data&quot; - vicgalle/configurable-safety-tuning
</li>
</ul>

</div>

---


**Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1225787256074145843)** (148 messages🔥🔥): 

- **GitHub Resource for PDF to Markdown Conversion**: A member shared a GitHub repository titled **VikParuchuri/marker**, which provides a tool to convert PDF files to markdown format with high accuracy. The repository can be found at [GitHub - VikParuchuri/marker](https://github.com/VikParuchuri/marker).

- **Hermes Function Calling Woes**: There was a discussion on how to make `Hermes-2-Pro-Mistral-7B` execute functions using a `tools` configuration similar to OpenAI's models. It was noted that while the model can handle ChatML syntax and function calls within messages, it encounters problems executing functions defined in `tools`.

- **Full-Parameter vs Adapter Training in LLMs**: Members debated on the challenges of achieving consistent results with full parameter finetuning compared to training adapters, with some sharing their relative success or lack thereof with either method in different contexts, such as Mixtral or Llamas.

- **Exploring Large Model Output Limitations**: A conversation took place about the limitations on output size in large language models, with the understanding that while input contexts can be quite large, output is limited due to different training data and operational considerations like needing examples of similarly sized outputs for training.

- **Combining Ontologies and Vector Searches**: There was an extensive discussion on utilizing **knowledge graph (KG) ontologies** with language models. Tips were shared on how to create Cypher queries from input text, the effectiveness of walking KG graphs using a vector search evaluation function, and the integration of vector databases with graph databases for production uses.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/TroyDoesAI/MermaidMistral">TroyDoesAI/MermaidMistral · Hugging Face</a>: no description found</li><li><a href="https://github.com/NousResearch/Hermes-Function-Calling">GitHub - NousResearch/Hermes-Function-Calling</a>: Contribute to NousResearch/Hermes-Function-Calling development by creating an account on GitHub.</li><li><a href="http://github.com/joey00072/ohara/issues/8">About GPU memory usage · Issue #8 · joey00072/ohara</a>: Hello. First of all, thanks for sharing a bitnet training code. I have a question about GPU memory usage. As I understanding, bitnet can reduce VRAM usage compared to fp16/bf16 precision. However, ...</li><li><a href="https://github.com/VikParuchuri/marker">GitHub - VikParuchuri/marker: Convert PDF to markdown quickly with high accuracy</a>: Convert PDF to markdown quickly with high accuracy - VikParuchuri/marker
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[project-obsidian](https://discord.com/channels/1053877538025386074/1156472202619781140/1226177515349872640)** (5 messages): 

- **Twisting LLaVA with Randomness**: A member experimented with the LLaVA vision-language model by injecting randomness into the image embeddings and observed the LLM's interpretations, detailed in their [blog post](https://anotherjesse.com/posts/llava-rorschach/). The process involved tweaking the model to accept random projections instead of CLIP projections, essentially performing a "Rorschach test" on the AI.

- **nanoLLaVA's Mighty Appearance**: Launching the "small but mighty" **nanoLLaVA** sub 1B vision-language model, a member shared a link to their creation [nanoLLaVA](https://huggingface.co/qnguyen3/nanoLLaVA) on Hugging Face, running on edge devices and boasting a unique combination of a Base LLM and Vision Encoder.
 
- **Obsidian and Hermes Vision Updates Imminent**: The same member announced impending updates to both **Obsidian** and **Hermes Vision**, suggesting enhancements in vision-language model capabilities.

- **ChatML Fusing Capabilities with LLaVA**: There was a successful endeavor to make ChatML work with the LLaVA model, hinting at a potential bridge between chat and vision-language tasks.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://anotherjesse.com/posts/llava-rorschach/">anotherjesse.com - Rorschach Test For LLaVA LLM</a>: no description found</li><li><a href="https://huggingface.co/qnguyen3/nanoLLaVA">qnguyen3/nanoLLaVA · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1225927044035252286)** (19 messages🔥): 

- **Chunking Script for Diverse Dataset Suggested**: The idea of writing a *chunking script* to save using a big RAG call at the time of generating the dataset was put forward. It can potentially make the dataset more diverse and efficient by preparing the RAG generation beforehand.
- **Multidoc Queries Via Claude Opus**: A discussion took place about the possibility of generating multidoc queries using *Claude Opus* by selecting documents from varied domains and generating queries that cut across them. This approach could enhance complex query generation for RAG models.
- **Diverse Document Sources for Model Training**: Links to diverse document sources have been shared, such as [the OCCRP data platform](https://aleph.occrp.org) and a repository of various files at [The Eye](https://the-eye.eu/public/). These sources could be scraped to create a rich training dataset.
- **Ransomware Victim Documents for Training**: There was a consideration of ransomware groups publishing victims' internal documents as a potential training data source. However, the ethics of using such data was flagged as questionable.
- **RAPTOR Clustering Strategy Discussed**: The recursive aspect of the RAPTOR clustering method was highlighted, prompting a discussion on the strategy of generating clusters and their role in stratifying collections for the RAG dataset.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://aleph.occrp.org">no title found</a>: no description found</li><li><a href="https://the-eye.eu/public/">Index of /public/</a>: no description found
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1225820438412787844)** (567 messages🔥🔥🔥): 

- **Worldsim Wheezes as We Wait**: Users continue to eagerly inquire about the recovery of **Worldsim** following a *DDoS* attack, discussing the potential implementation of a login system to prevent future attacks from notorious online communities like *4chan*.

- **AI Memory Mystery**: There's confusion and discussion about whether **Claude** can remember information across separate sessions or whether it is just imitating this ability through its probabilistic model, with random token selection causing varying outcomes despite identical prompts.

- **Seeking Sustainable Solution**: As users propose subscription models for Worldsim to offset the high operational costs sparked by indiscriminate access, **Nous Research** hints at a future "pro" version with plans for more scenarios, while stressing the need for a sustainable platform.

- **Tales and Tech of Transcendence**: The channel teems with philosophical discussions about consciousness, existence, and the potential of living in a simulation; parallel dialogues delve into the nature of AI, existence, and the interplay of science and philosophy.

- **Impatient for Play**: Users express a mix of impatience and enthusiasm for Worldsim's return, asking for updates, while discussing ways to prevent unrestricted access and pondering potential costs of a subscription-based model to keep the service both financially viable and protected from misuse.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.anthropic.com/claude/reference/client-sdks">Client SDKs</a>: no description found</li><li><a href="https://www.mlexpert.io/prompt-engineering/memgpt">MemGPT - Unlimited Context (Memory) for LLMs | MLExpert - Crush Your Machine Learning interview</a>: How can you overcome the context window size limit of LLMs? MemGPT helps handle longer conversations by cleverly managing different memory tiers.</li><li><a href="https://huggingface.co/fbjr/cohere_c4ai-command-r-plus-mlx-4bit-128g">fbjr/cohere_c4ai-command-r-plus-mlx-4bit-128g · Hugging Face</a>: no description found</li><li><a href="https://worldsim.nousresearch.com/">world_sim</a>: no description found</li><li><a href="https://websim.ai/">websim.ai</a>: no description found</li><li><a href="https://a.co/d/e98NrUY">no title found</a>: no description found</li><li><a href="https://www.google.com/amp/s/80.lv/articles/google-s-new-ai-can-generate-entire-2d-platformer-games/%3famp=1">Google&#x27;s New AI Can Generate Entire 2D Platformer Games</a>: The new model, dubbed Genie, can create playable environments from a single image prompts.</li><li><a href="https://www.gameb.wiki/index.php?title=An_Introduction_to_Game_B">An Introduction to Game B - Game B Wiki</a>: no description found</li><li><a href="https://youtube.com/shorts/qE9gYuSVfyQ">The Box | Science Fiction Animatic</a>: Video Summary:The animatic follows the perspective of &quot;The Breacher,&quot; a character determined to escape the confines of a simulated reality, known as the &quot;Wor...</li><li><a href="https://github.com/simonw/llm">GitHub - simonw/llm: Access large language models from the command-line</a>: Access large language models from the command-line - simonw/llm</li><li><a href="https://youtu.be/PHQweR1z7pI?si=ac4KikfzI5A4w4kZ">Worldsim and jailbreaking Claude 3 Including subtitles [-2-]</a>: Personal learning</li><li><a href="https://youtube.com/shorts/oGng-eDRb0A">The Great Eclipse | Science Fiction Animatic</a>: Video Summary:This animatic explores a war of ideas and beliefs, fought not with weapons but through the power of data, debates, and simulated worlds. As dif...</li><li><a href="https://www.nature.com/articles/s41598-019-56357-3">Quantum Mechanics can be understood through stochastic optimization on spacetimes - Scientific Reports</a>: no description found</li><li><a href="https://www.urantia.org/urantia-book/read-urantia-book-online>">Home</a>: no description found
</li>
</ul>

</div>
  

---



**Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1225739403645947915)** (977 messages🔥🔥🔥): 

- **Searching for the Stability Bot**: Users inquired about generating images and were redirected to check server status in <#1047610792226340935> as bots are currently down.
- **Curiosity About Image Generation Results**: Users discussed differences in image output quality between local models and Dreamstudio, with suggestions to try open-source upscalers and inquiries into the effectiveness of various techniques.
- **Anticipation for Stable Diffusion 3**: Conversations indicate an informal ETA of 2-4 weeks for the release of SD3, with discussions about its anticipated improvements and capabilities.
- **Exploring SD Model Enhancements**: Users exchanged information on training LoRAs, including questions about installation and practicality, with suggestions to follow specific GitHub repositories for guidance.
- **Switching Between UIs**: Members shared advice for switching from Automatic 1.1.1.1 to other UIs like StableSwarm, emphasizing the latter's enhanced user experience and features for newcomers.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://app.leonardo.ai/">Leonardo.Ai</a>: Create production-quality visual assets for your projects with unprecedented quality, speed and style-consistency.</li><li><a href="https://huggingface.co/stabilityai/cosxl">stabilityai/cosxl · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/spaces/ByteDance/SDXL-Lightning">SDXL-Lightning - a Hugging Face Space by ByteDance</a>: no description found</li><li><a href="https://arxiv.org/abs/2404.02905">Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction</a>: We present Visual AutoRegressive modeling (VAR), a new generation paradigm that redefines the autoregressive learning on images as coarse-to-fine &#34;next-scale prediction&#34; or &#34;next-resolutio...</li><li><a href="https://huggingface.co/spaces/PixArt-alpha/PixArt-LCM">PixArt LCM - a Hugging Face Space by PixArt-alpha</a>: no description found</li><li><a href="https://civitai.com/models/3798/lexica-testica">Lexica Testica - 1.0 | Stable Diffusion Checkpoint | Civitai</a>: Initialized from OpenJourney v2, further fine-tuned for 4000 steps on images scraped from the front page of Lexica art (January 2023). Good at prod...</li><li><a href="https://tenor.com/view/frieren-wow-elf-peek-a-boo-gif-12265100463579712545">Frieren Wow GIF - Frieren Wow Elf - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/yoshi-mario-yoshis-island-super-smash-brother-super-smash-brother-n64-gif-21681448">Yoshi Mario GIF - Yoshi Mario Yoshis Island - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://openmodeldb.info/">OpenModelDB</a>: OpenModelDB is a community driven database of AI Upscaling models. We aim to provide a better way to find and compare models than existing sources.</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1bnjm3i/comment/kwjb37c/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/FoundationVision/VAR/blob/main/demo_sample.ipynb">VAR/demo_sample.ipynb at main · FoundationVision/VAR</a>: [GPT beats diffusion🔥] [scaling laws in visual generation📈] Official impl. of &quot;Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction&quot; - FoundationVision/VAR</li><li><a href="https://github.com/LykosAI/StabilityMatrix">GitHub - LykosAI/StabilityMatrix: Multi-Platform Package Manager for Stable Diffusion</a>: Multi-Platform Package Manager for Stable Diffusion - LykosAI/StabilityMatrix</li><li><a href="https://www.federalregister.gov/documents/2023/03/16/2023-05321/copyright-registration-guidance-works-containing-material-generated-by-artificial-intelligence>">Federal Register :: Request Access</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=8_V8CO_Dbdw">How to Run Stable Diffusion in Google Colab (Free) WITHOUT DISCONNECT</a>: Here&#39;s how to code your own python notebook in Colab to generate AI images for FREE, without getting disconnected. We&#39;ll use the Diffusers library from Huggi...</li><li><a href="https://forms.gle/avNEgKWp8nj3UAEg9">Survey: Comparing generated photos by AI (Diffusion Models)</a>: This is a survey to determine the more accurate output from different diffusion models like SD 1.5, SD 2.0, SDXL, Dall-e-3 and a custom fine tuned model.  It will take a few minutes to complete the su...</li><li><a href="https://www.youtube.com/watch?v=gEwPGyWjK70">Installing Stability Matrix (1 click installers for Automatic 1111, ComfyUI, Fooocus, and more)</a>: This Stability Matrix application is designed for Windows and allows for installing and managing text to image web ui apps like Automatic 1111, ComfyUI, and ...</li><li><a href="https://youtu.be/QIqoMSf4P88">A Conversation with Malcolm and Simone Collins (Supercut)</a>: Malcolm and Simone are the founders of pronatalist.org, The Collins Institute for the Gifted, and Based Camp Podcast.All Outcomes Are Acceptable Blog: https:...</li><li><a href="https://github.com/altoiddealer/--sd-webui-ar-plusplus">GitHub - altoiddealer/--sd-webui-ar-plusplus: Select img aspect ratio from presets in sd-webui</a>: Select img aspect ratio from presets in sd-webui. Contribute to altoiddealer/--sd-webui-ar-plusplus development by creating an account on GitHub.</li><li><a href="https://hforsten.com/identifying-stable-diffusion-xl-10-images-from-vae-artifacts.html">Identifying Stable Diffusion XL 1.0 images from VAE artifacts</a>: The new SDXL 1.0 text-to-image generation model was recently released that generates small artifacts in the image when the earlier 0.9 release didn&#39;t have them.</li><li><a href="https://github.com/nashsu/FreeAskInternet">GitHub - nashsu/FreeAskInternet: FreeAskInternet is a completely free, private and locally running search aggregator &amp; answer generate using LLM, without GPU needed. The user can ask a question and the system will  make a multi engine search and combine the search result to the ChatGPT3.5 LLM and generate the answer based on search results.</a>: FreeAskInternet is a completely free, private and locally running search aggregator &amp;amp; answer generate using LLM, without GPU needed. The user can ask a question and the system will  make a mul...</li><li><a href="https://www.youtube.com/watch?v=kqXpAKVQDNU&list=PLXS4AwfYDUi5sbsxZmDQWxOQTml9Uqyd2">How to Install Stable Diffusion - automatic1111</a>: Part 2: How to Use Stable Diffusion https://youtu.be/nJlHJZo66UAAutomatic1111 https://github.com/AUTOMATIC1111/stable-diffusion-webuiInstall Python https://w...</li><li><a href="https://github.com/derrian-distro/LoRA_Easy_Training_Scripts">GitHub - derrian-distro/LoRA_Easy_Training_Scripts: A UI made in Pyside6 to make training LoRA/LoCon and other LoRA type models in sd-scripts easy</a>: A UI made in Pyside6 to make training LoRA/LoCon and other LoRA type models in sd-scripts easy - derrian-distro/LoRA_Easy_Training_Scripts</li><li><a href="https://civitai.com/models/1493/sonicdiffusion">SonicDiffusion - V4 | Stable Diffusion Checkpoint | Civitai</a>: Try it out here! https://mobians.ai/ Join the discord for updates, share generated-images, just want to chat or if you want to contribute to helpin...</li><li><a href="https://github.com/Stability-AI/StableSwarmUI?tab=readme-ov-file#stableswarmui">GitHub - Stability-AI/StableSwarmUI: StableSwarmUI, A Modular Stable Diffusion Web-User-Interface, with an emphasis on making powertools easily accessible, high performance, and extensibility.</a>: StableSwarmUI, A Modular Stable Diffusion Web-User-Interface, with an emphasis on making powertools easily accessible, high performance, and extensibility. - Stability-AI/StableSwarmUI</li><li><a href="https://github.com/camenduru/Open-Sora-Plan-replicate">GitHub - camenduru/Open-Sora-Plan-replicate</a>: Contribute to camenduru/Open-Sora-Plan-replicate development by creating an account on GitHub.</li><li><a href="https://github.com/GarlicCookie/PNG-SD-Info-Viewer">GitHub - GarlicCookie/PNG-SD-Info-Viewer: PNG-SD-Info-Viewer is a program designed to quickly allow the browsing of PNG files with associated metadata from Stable Diffusion generated images.</a>: PNG-SD-Info-Viewer is a program designed to quickly allow the browsing of PNG files with associated metadata from Stable Diffusion generated images. - GarlicCookie/PNG-SD-Info-Viewer</li><li><a href="https://github.com/GarlicCookie/SD-Quick-View">GitHub - GarlicCookie/SD-Quick-View: SD-Quick-View is a program designed to very quickly look through images generated by Stable Diffusion and see associated metadata.</a>: SD-Quick-View is a program designed to very quickly look through images generated by Stable Diffusion and see associated metadata. - GarlicCookie/SD-Quick-View</li><li><a href="https://github.com/lllyasviel/stable-diffusion-webui-forge">GitHub - lllyasviel/stable-diffusion-webui-forge</a>: Contribute to lllyasviel/stable-diffusion-webui-forge development by creating an account on GitHub.</li><li><a href="https://github.com/ronniebasak/ComfyUI-Tara-LLM-Integration/blob/main/README.md">ComfyUI-Tara-LLM-Integration/README.md at main · ronniebasak/ComfyUI-Tara-LLM-Integration</a>: Contribute to ronniebasak/ComfyUI-Tara-LLM-Integration development by creating an account on GitHub.</li><li><a href="https://www.reddit.com/r/StableDiffusion/s/HbOA5xdG8J">Reddit - Dive into anything</a>: no description found</li><li><a href="https://civitai.com/models/161068/newrealityxl-all-in-one-photographic">NewRealityXL ❗ All-In-One Photographic - ✔ 3.0 Experimental | Stable Diffusion Checkpoint | Civitai</a>: IMPORTANT: v2.x ---&amp;gt; Main Version | v3.x ---&amp;gt; Experimental Version I need your time to thoroughly test this new 3rd version to understand all...
</li>
</ul>

</div>
  

---



**Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1225801606377898034)** (341 messages🔥🔥): 

- **Conversion Frustration**: Members expressed dissatisfaction with machine learning models like Chat GPT and Claude, which when converting an image to HTML, lost fidelity in colors and border rounding. A humorous suggestion was made to convert images to ASCII art instead.

- **Shared Challenges with Model Limitations**: Conversations included server crashes leading to lost models due to issues with model saving on platforms like Unsloth AI and Hugging Face, and some users expressed their loss of extensive fine-tuning efforts.

- **Curiosity on LLM Vision Model Alternatives**: While vision models still linger on Unsloth AI's roadmap, they remain a lower priority. Users discussed alternatives like Dreambooth but didn't find a definitive solution.

- **GPU Woes for LLM Training**: Strategies for avoiding laptop overheating during model training were humorously debated, including moving to Antarctica or using air conditioning. 

- **Anticipation for Gradient Checkpointing and Longer Contexts**: Unofficial hints and teasers about Unsloth AI's upcoming features spurred debate and anticipation, leading to discussions on possible implementations and benefits for users with limited GPU resources.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/liminerity/Mistral-quiet-star-demo">liminerity/Mistral-quiet-star-demo · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/liminerity/Mistral-quiet-star">liminerity/Mistral-quiet-star-demo · Hugging Face</a>: no description found</li><li><a href="https://unsloth.ai/blog/gemma-bugs">Unsloth Fixing Gemma bugs</a>: Unsloth fixing Google&#x27;s open-source language model Gemma.</li><li><a href="https://arstechnica.com/security/2024/03/researchers-use-ascii-art-to-elicit-harmful-responses-from-5-major-ai-chatbots/">ASCII art elicits harmful responses from 5 major AI chatbots</a>: LLMs are trained to block harmful responses. Old-school images can override those rules. </li><li><a href="https://github.com/uclaml/SPIN/blob/main/scripts/finetune.sh">SPIN/scripts/finetune.sh at main · uclaml/SPIN</a>: The official implementation of Self-Play Fine-Tuning (SPIN) - uclaml/SPIN</li><li><a href="https://github.com/haotian-liu/LLaVA/blob/main/docs%2FFinetune_Custom_Data.md">LLaVA/docs/Finetune_Custom_Data.md at main · haotian-liu/LLaVA</a>: [NeurIPS&#39;23 Oral] Visual Instruction Tuning (LLaVA) built towards GPT-4V level capabilities and beyond. - haotian-liu/LLaVA</li><li><a href="https://github.com/facebookresearch/schedule_free">GitHub - facebookresearch/schedule_free: Schedule-Free Optimization in PyTorch</a>: Schedule-Free Optimization in PyTorch. Contribute to facebookresearch/schedule_free development by creating an account on GitHub.</li><li><a href="https://www.youtube.com/watch?v=VNsWWb8g3Js">When GPT-5 is coming out | Sam Altman and Lex Fridman</a>: Lex Fridman Podcast full episode: https://www.youtube.com/watch?v=jvqFAi7vkBcPlease support this podcast by checking out our sponsors:- Cloaked: https://cloa...</li><li><a href="https://github.com/unslothai/unsloth/wiki#saving-models-to-16bit-for-vllm">Home</a>: 2-5X faster 70% less memory QLoRA &amp; LoRA finetuning - unslothai/unsloth</li><li><a href="https://github.com/stanfordnlp/pyreft">GitHub - stanfordnlp/pyreft: ReFT: Representation Finetuning for Language Models</a>: ReFT: Representation Finetuning for Language Models - stanfordnlp/pyreft</li><li><a href="https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.push_to_hub">Trainer</a>: no description found</li><li><a href="https://huggingface.co/datasets/gate369/Alpaca-Star">gate369/Alpaca-Star · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1225755450117460129)** (78 messages🔥🔥): 

- **Preferred AI News Sources Shared**: Members shared their go-to sources for AI news—**AI News** and **Reddit**, particularly giving a shoutout to the user *localllama* for consistent updates.
- **Debating the Merits of Learning Rate Schedulers**: A member is sharing results from experimenting with different learning rate schedulers—**linear**, **cosine with restarts**, and **constant**—pointing out that constant surprisingly seems to work best for their model, focusing on a general assistant in multiple languages.
- **Finetuning with DPO Questions Raised**: Curiosity arose around why models finetuned with **DPO** (Dynamic Positional Offsets) perform well on the **Open LLM Leaderboard**, even if the base models have lower scores. It was suggested that proprietary datasets could be an influencing factor.
- **Impact of Benchmarks on Model Perception Discussed**: Discussion about benchmarks revealed that they may not always align with perceived quality; low-scoring models could still be very effective. Concerns about the potential for models to be 'contaminated' by test data and about the rigidity of the benchmarks were also expressed.
- **Unsloth Hiring and Open Source Contributions Mentioned**: Members discussed forthcoming hiring for full-stack developers and a developer advocate, clarifying that the roles will be for contributing to the open-source community and building out the Unsloth Pro platform.

**Link mentioned**: <a href="https://github.com/unslothai/unsloth/wiki">Home</a>: 2-5X faster 70% less memory QLoRA &amp; LoRA finetuning - unslothai/unsloth

  

---


**Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1225765845653917788)** (374 messages🔥🔥): 

- **Private Hosting of AI Models**: A member announced they are hosting their AI projects on their own server to maintain the privacy of some unreleased models. They hinted at future integration of advertisements after hosting more high-quality models and shared a link: [Hircoir Text-to-Speech](https://tts.hircoir.eu.org/).

- **Inference Code Flexibility Praised**: The Unsloth AI's inference code received praise for its speed and ease of use. Members were reminded they could modify inference settings like temperature and use Generative Guided Unsupervised Fine-tuning (GGUF) as desired.

- **Discussion on Model Merging**: A chat involved potential merging tactics for AI models, with suggestions including applying differences between various models onto each other. Views on the subject varied from skepticism to optimism based on past experiences.

- **User Struggles with Code**: One user expressed difficulty with coding, specifically related to model parameter adjustments for batch inference. They were directed to Unsloth's GitHub for guidance, highlighting the utility of `model.generate`.

- **Batch Inference Clarification**: Members discussed how to effectively execute batch inference. The usage of `num_return_sequences` was corrected, indicating that it only works for single prompts, not for batched prompts which should be "shoved all together."
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/1Aau3lgPzeZKQ-98h69CCu1UJcvIBLmy2?usp=sharing">Google Colaboratory</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1n8vXmEQ-rAXdytw25M3y6ff4k1acYtxt?usp=sharing">Google Colaboratory</a>: no description found</li><li><a href="https://tts.hircoir.eu.org/">HirLab - Convertidor de Texto a Voz por Hircoir</a>: HirLab, es una plataforma de conversión de texto a voz basada en inteligencia artificial. Convierte texto a voz de forma rápida y precisa.</li><li><a href="https://magazine.sebastianraschka.com/p/lora-and-dora-from-scratch">Improving LoRA: Implementing Weight-Decomposed Low-Rank Adaptation (DoRA) from Scratch</a>: Low-rank adaptation (LoRA) is a machine learning technique that modifies a pretrained model (for example, an LLM or vision transformer) to better suit a specific, often smaller, dataset by adjusting o...</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/models/llms/usage_custom/#">Customizing LLMs - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/models/llms/usage_custom/#example-using-a-huggingface-llm">Customizing LLMs - LlamaIndex</a>: no description found</li><li><a href="https://huggingface.co/docs/transformers/generation_strategies">Text generation strategies</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=rANv5BVcR5k">Mistral Fine Tuning for Dummies (with 16k, 32k, 128k+ Context)</a>: Discover the secrets to effortlessly fine-tuning Language Models (LLMs) with your own data in our latest tutorial video. We dive into a cost-effective and su...</li><li><a href="https://github.com/unslothai/unsloth/blob/main/unsloth/save.py#L111">unsloth/unsloth/save.py at main · unslothai/unsloth</a>: 2-5X faster 70% less memory QLoRA &amp; LoRA finetuning - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth#-finetune-for-free">GitHub - unslothai/unsloth: 2-5X faster 70% less memory QLoRA &amp; LoRA finetuning</a>: 2-5X faster 70% less memory QLoRA &amp; LoRA finetuning - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/wiki#saving-models-to-16bit-for-vllm">Home</a>: 2-5X faster 70% less memory QLoRA &amp; LoRA finetuning - unslothai/unsloth</li><li><a href="https://github.com/huggingface/transformers/issues/26877">Mistral with flash attention 2 and right padding · Issue #26877 · huggingface/transformers</a>: System Info transformers version: 4.34.0 Platform: Linux-5.4.0-148-generic-x86_64-with-glibc2.31 Python version: 3.10.13 Huggingface_hub version: 0.17.3 Safetensors version: 0.4.0 Accelerate versio...</li><li><a href="https://huggingface.co/datasets/pharaouk/UltraInteract_sft">pharaouk/UltraInteract_sft · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/docs/transformers/en/main_classes/model#model-instantiation-dtype">Models</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/issues/267#issuecomment-2034047189">Batch inference produces nonsense results for unsloth/mistral-7b-instruct-v0.2-bnb-4bit · Issue #267 · unslothai/unsloth</a>: Hi there, after loading the model with: from unsloth import FastLanguageModel import torch model, tokenizer = FastLanguageModel.from_pretrained( model_name = &quot;unsloth/mistral-7b-instruct-v0.2-bnb...
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1226796733393010688)** (2 messages): 

- **Introducing Aurora-M**: A new [15.5 billion parameter open-source multilingual language model](https://arxiv.org/abs/2404.00399) named **Aurora-M** has been developed, following the U.S. Executive Order on AI. It demonstrates cross-lingual impact of mono-lingual safety alignment and surpasses 2 trillion training tokens.
- **Cross-Lingual Safety Impact Validated**: The team found that safety alignment tuning performed on English not only enhanced safety in English but also in other languages such as German. This is touted as the first evidence of *cross-lingual impact of mono-lingual safety alignment*.
- **Peer Recognition**: The community has shown support for the *Aurora-M* project with positive reactions like "great work! 🔥".
- **Aurora-M's Upcoming Developments**: The project aims to build on *Aurora-M* by training a mixture of experts using LoRA and a subsequent merge. Feedback from the Unsloth AI community is sought, particularly concerning the use of LoRA fine-tuning notebooks.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/__z__9/status/1774965364301971849?s=20">Tweet from ً ‎ (@__z__9)</a>: New preprint! The first multi-lingual red-teamed open-source continually pre-trained LLM - **Aurora-M** in accordance with the #WhiteHouse Executive Order on the Safe, Secure, and Trustworthy developm...</li><li><a href="https://arxiv.org/abs/2404.00399">Aurora-M: The First Open Source Multilingual Language Model Red-teamed according to the U.S. Executive Order</a>: Pretrained language models underpin several AI applications, but their high computational cost for training limits accessibility. Initiatives such as BLOOM and StarCoder aim to democratize access to p...
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[suggestions](https://discord.com/channels/1179035537009545276/1180144489214509097/1226587652262985858)** (148 messages🔥🔥): 

- **The Investment Conundrum**: A hot debate ensued regarding the value of **Jamba** and **Mamba**, with opinions varying from skepticism to cautious optimism. While one member noted that the company behind Jamba, AI21 Labs, raised $155 million, which may indicate market interest, others were critical, suggesting that such investments might have been misguided. ([AI21 Labs Fundraising](https://techcrunch.com/2023/08/30/generative-ai-startup-ai21-labs-lands-155m-at-a-1-4b-valuation/))

- **Quantized Models Take Center Stage**: The viability of quantized models and optimization techniques, such as AQLM, generated mixed feelings. One individual pointed out the desirable ROI from fine-tuning for specific use cases, sharing that proper investments in fine-tuning can yield significant returns despite high upfront costs. ([AQLM Mixtral](https://huggingface.co/ISTA-DASLab/Mixtral-8x7b-AQLM-2Bit-1x16-hf))

- **Varying Opinions on Model Architecture**: Members evaluated the need for incorporating partially optimized **MoEs** and discussed the implementation of new architectures with a particular member suggesting a cautious approach of waiting to see if such models gain popularity.

- **Future of Automated AI Engineering**: Conversations briefly touched upon the potential of creating **finetuned models that could assist in writing optimized kernels** or heavy computational tasks like Triton code generation, aiming to pioneer automated AI engineering solutions.

- **Practicality Over Hype**: There was a strong current throughout the discussion underscoring the importance of practical, well-optimized models over "hyped" startup ventures. The consensus seemed to lean towards a preference for supporting proven, scalable architectures like **Transformers** and adding **MoE** implementations in due time.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/ISTA-DASLab/Mixtral-8x7b-AQLM-2Bit-1x16-hf">ISTA-DASLab/Mixtral-8x7b-AQLM-2Bit-1x16-hf · Hugging Face</a>: no description found</li><li><a href="https://techcrunch.com/2023/08/30/generative-ai-startup-ai21-labs-lands">Generative AI startup AI21 Labs lands $155M at a $1.4B valuation | TechCrunch</a>: AI21 Labs, a company competing against OpenAI and Anthropic, among other generative AI players, has raised $155 million in capital.</li><li><a href="https://huggingface.co/ai21labs/Jamba-v0.1">ai21labs/Jamba-v0.1 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/alpindale/Mistral-7B-Instruct-v0.2-AQLM-2Bit-1x16">alpindale/Mistral-7B-Instruct-v0.2-AQLM-2Bit-1x16 · Hugging Face</a>: no description found</li><li><a href="https://techcrunch.com/2023/08/30/generative-ai-startup-ai21-labs-lands-155m-at-a-1-4b-valuation/">Generative AI startup AI21 Labs lands $155M at a $1.4B valuation | TechCrunch</a>: AI21 Labs, a company competing against OpenAI and Anthropic, among other generative AI players, has raised $155 million in capital.</li><li><a href="https://discuss.pytorch.org/t/choice-of-torch-compile-vs-triton/195604/2">Choice of torch.compile vs. triton</a>: On GPUs torch.compile() will apply various compiler optimizations with the most important ones being cuda graphs and fusions. Fusions specifically are done by code generating triton kernels so torch.c...
</li>
</ul>

</div>
  

---



**LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1225750743047208991)** (488 messages🔥🔥🔥): 

- **GPU Offload Confusion Cleared**: Users were guided on optimizing GPU use in LM Studio with suggestions to adjust "n_gpu_layers" and "GPU Offloading" settings for better performance, alleviating concerns about overreliance on integrated graphics or failure to utilize Nvidia GPUs. The advice pointed towards ensuring models are fully offloaded to GPU for better speed when possible.
  
- **Big LLMs and Multi-GPU Setup**: Discussions about running large models, such as a 70b model, revolved around the importance of VRAM, with users sharing their experiences and setups including dual RTX 4060 Ti 16GB configurations. The consensus is that the more VRAM available, the larger the models that can be run without slowing down due to system RAM use.

- **Exploring Model Capabilities**: Queries about whether lower GB models can learn user names led to explanations regarding the use of system prompts to instruct models on desired behavior. Clarifications were provided stating that LM Studio does not support actual learning or training of models; however, crafting detailed prompts can achieve similar outcomes as learning.

- **AI for Coding Purposes**: Users recommended OpenAI's GPT-4 for its proficiency in coding, although noting the associated costs. The discussion highlighted the lack of equivalent open-source models, reflecting the balancing act between model capability and cost.

- **Diverse Usage and Integration Questions**: Conversations ranged from setting up models on separate drives, to error debugging, to asking about the compatibility of models with different formats like EXL2 and GGUF. Integration topics included connecting LM Studio with other tools like OpenDevin and using LLMs for tasks like text analysis and fiction writing.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://klu.ai/glossary/grouped-query-attention">What is Grouped Query Attention (GQA)? — Klu</a>: no description found</li><li><a href="https://lmstudio.ai/docs/">Documentation | LM Studio</a>: Technical Reference</li><li><a href="https://huggingface.co/LoneStriker/miqu-1-70b-sf-4.25bpw-h6-exl2">LoneStriker/miqu-1-70b-sf-4.25bpw-h6-exl2 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/TheBloke/MXLewdMini-L2-13B-GGUF">TheBloke/MXLewdMini-L2-13B-GGUF · Hugging Face</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1avdwx2/new_try_where_is_the_quantization_god/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://huggingface.co/TheBloke/MXLewdMini-L2-13B-GGUF#prompt-template-alpaca">TheBloke/MXLewdMini-L2-13B-GGUF · Hugging Face</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/16ubkyq/nvlink_bridge_worth_it_for_dual_rtx_3090/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/Pythagora-io/gpt-pilot/issues/807#issuecomment-2037824538">[Bug]: LLM Studio does not connect  · Issue #807 · Pythagora-io/gpt-pilot</a>: Version VisualStudio Code extension Operating System Windows 11 What happened? By changing the endpoint and api key from Openai to LLmStudio: if using OPENAI_ENDPOINT=http://localhost:1234/v1 There...</li><li><a href="https://github.com/enricoros/big-AGI/blob/main/docs/config-local-lmstudio.md">big-AGI/docs/config-local-lmstudio.md at main · enricoros/big-AGI</a>: Generative AI suite powered by state-of-the-art models and providing advanced AI/AGI functions. It features AI personas, AGI functions, multi-model chats, text-to-image, voice, response streaming, ...</li><li><a href="https://rentry.org/LMSTudioFAQ">The unofficial LMStudio FAQ!</a>: Welcome to the unofficial LMStudio FAQ. Here you will find answers to the most commonly asked questions that we get on the LMStudio Discord. (This FAQ is community managed).  LMStudio is a free closed...</li><li><a href="https://www.humblebundle.com/books/machine-learning-ai-deep-learning-and-llm-pearson-books?hmb_source=&hmb_medium=product_tile&hmb_campaign=mosaic_section_1_layout_index_2_layout_type_threes_tile_index_2_c_machinelearningaideeplearningandllmpearson_bookbundle">Humble Tech Book Bundle: Machine Learning, AI, Deep Learning, and LLM by Pearson</a>: Stay abreast of the technologies that will define the future with these books on AI, machine learning, and other cutting edge topics in computer science!</li><li><a href="https://github.com/jakobdylanc/discord-llm-chatbot">GitHub - jakobdylanc/discord-llm-chatbot: llmcord.py • Talk to LLMs with your friends!</a>: llmcord.py • Talk to LLMs with your friends! Contribute to jakobdylanc/discord-llm-chatbot development by creating an account on GitHub.</li><li><a href="https://www.reddit.com/r/LocalLLaMA">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/OpenDevin/OpenDevin/issues/419">Trouble using LMStudio · Issue #419 · OpenDevin/OpenDevin</a>: Describe the bug Trouble connecting to LMStudio Steps to Reproduce 1.Start server on LMStudio 2.Start frontend and backend on OpenDevin 3. Expected behavior OpenDevin asks what I want it to build A...</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/6491">Add Command R Plus support by Carolinabanana · Pull Request #6491 · ggerganov/llama.cpp</a>: Updated tensor mapping to add Command R Plus support for GGUF conversion.
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1225757263030128641)** (103 messages🔥🔥): 

- **Command R+ on the Horizon**: The highly anticipated **Command R+** model still faces integration challenges with llama.cpp, but a [fork is available](https://github.com/pmysl/c4ai-command-r-plus-GGUF) where it can work. It is a significant 104B model requiring powerful hardware specs.
- **GGUF Format Quirks and Help Offered**: Discussions around hassle-free implementation of quant formats led to members sharing experiences and offering assistance for those looking to run GGUF models. Concerns over silently vanishing contributors like TheBloke stir community curiosity.
- **Hardware Headaches and Humor**: Community members jest about the costly nature of their LLM hardware hobbies, comparing expenditures to extravagant purchases like a BMW M4. There's also advice on finding budget solutions like using the Nvidia P40 graphics card.
- **AI Storytelling Pursuits**: One member expresses interest in using AI models for creative storytelling and receives tips on selecting models with large context and memory management tools like [MemGPT](https://github.com/cpacker/MemGPT).
- **Curiosity Over Vision Adapters**: A question arises about the functionality of vision adapters in cjpais llava models, followed by inquiries on how to replicate visual task capabilities similar to those demonstrated in a specific video.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/pmysl/c4ai-command-r-plus-GGUF">pmysl/c4ai-command-r-plus-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/spaces/CohereForAI/c4ai-command-r-plus">C4AI Command R Plus - a Hugging Face Space by CohereForAI</a>: no description found</li><li><a href="https://huggingface.co/TheB">TheB (Pastor B)</a>: no description found</li><li><a href="https://huggingface.co/google/gemma-1.1-7b-it">google/gemma-1.1-7b-it · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/Qwen/Qwen1.5-32B-Chat-GGUF/tree/main">Qwen/Qwen1.5-32B-Chat-GGUF at main</a>: no description found</li><li><a href="https://huggingface.co/TheBloke/Wizard-Vicuna-7B-Uncensored-GGUF">TheBloke/Wizard-Vicuna-7B-Uncensored-GGUF · Hugging Face</a>: no description found</li><li><a href="https://llm.extractum.io/model/TheBloke%2FWizard-Vicuna-7B-Uncensored-GPTQ,1e2RcN80JhFWYaq1IBixLq">Wizard Vicuna 7B Uncensored GPTQ By TheBloke: Benchmarks and Detailed Analysis. Insights on Wizard Vicuna 7B Uncensored GPTQ.</a>: LLM Card: 7b LLM, VRAM: 4.5GB, Context: 2K, License: other, Quantized, Uncensored.</li><li><a href="https://huggingface.co/TheBloke/goliath-120b-GGUF">TheBloke/goliath-120b-GGUF · Hugging Face</a>: no description found</li><li><a href="https://github.com/j4ys0n">j4ys0n - Overview</a>: blockchain engineer. j4ys0n has 62 repositories available. Follow their code on GitHub.</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/6387">ggml : update mul_mat_id to use the same tensor for all the experts by slaren · Pull Request #6387 · ggerganov/llama.cpp</a>: Changes the storage of experts in memory from a tensor per expert, to a single 3D tensor with all the experts. This will allow us support models with a large number of experts such as qwen2moe. Exi...
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1225884680008503366)** (1 messages): 

- **LM Studio Community Page Launches**: The LM Studio team has introduced a new "lmstudio-community" page on Hugging Face, providing access to the latest **GGUF quants**. Users can find and experiment with these models by searching for `lmstudio-community` within LM Studio; find them [here](https://huggingface.co/lmstudio-community).
- **@bartowski1182 Joins as LLM Archivist**: Announced on Twitter, @bartowski1182 will serve as the resident **LLM Archivist** for LM Studio, assisting with updates to the new Hugging Face community page. Check the Twitter announcement [here](https://x.com/LMStudioAI/status/1776324680124694654).

**Link mentioned**: <a href="https://x.com/LMStudioAI/status/1776324680124694654">Tweet from LM Studio (@LMStudioAI)</a>: If you&#39;ve been around these parts for long enough, you might be missing @TheBlokeAI as much as we do 🥲.  Us & @bartowski1182 decided to try to help fill the void. We&#39;re excited to share the n...

  

---


**LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1225804230238011452)** (25 messages🔥): 

- **Praise for LM Studio's GUI**: Members found **LM Studio** to outperform other local LLM GUIs like **oogabooga** and **Faraday**, appreciating the quality results even using the same models and instructions.

- **Feature Expansion Request**: A suggestion to add **file reading support** and various modes such as *text to images*, *image to text*, and *text to voice* functionalities for LM Studio was proposed, seeking improvements similar to an existing tool named **Devin**.

- **Vision Models Awestruck**: Members are excited by the vision models tested, thanking LM Studio for its utility. Vision models are available on [Hugging Face](https://huggingface.co/collections/lmstudio-ai/vision-models-gguf-6577e1ce821f439498ced0c1).

- **Troubleshooting Download Issues**: There were issues downloading the Linux beta version of LM Studio with a user trying on Pop!_OS 22.04 LTS. The issue was identified as a bug on the website, with a direct link to the AppImage being provided ([Link Here](https://releases.lmstudio.ai/linux/0.2.19/beta/LM_Studio-0.2.19-Preview-1.AppImage)).

- **Supporting the Uncensored Model**: A request was made for LM Studio to support a new, uncensored model named **Dolphin 2.8 Mistral 7b v0.2**, which is available on Hugging Face ([Uncensored Model](https://huggingface.co/cognitivecomputations/dolphin-2.8-mistral-7b-v02)).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lmstudio.ai/beta-releases.html">LM Studio Beta Releases</a>: no description found</li><li><a href="https://lmstudio.ai">👾 LM Studio - Discover and run local LLMs</a>: Find, download, and experiment with local LLMs</li><li><a href="https://releases.lmstudio.ai/linux/0.2.19/beta/LM_Studio-0.2.19-Preview-1.AppImage">no title found</a>: no description found</li><li><a href="https://huggingface.co/collections/lmstudio-ai/vision-models-gguf-6577e1ce821f439498ced0c1">Vision Models (GGUF) - a lmstudio-ai Collection</a>: no description found</li><li><a href="https://huggingface.co/cognitivecomputations/dolphin-2.8-mistral-7b-v02">cognitivecomputations/dolphin-2.8-mistral-7b-v02 · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[📝-prompts-discussion-chat](https://discord.com/channels/1110598183144399058/1120489168687087708/1226893471496011857)** (2 messages): 

- **Inquiry about LLMs for Stock Market Analysis**: A member posed a question on how to train **Large Language Models (LLMs)** for interpreting stock market **OHLC** (Open, High, Low, Close) prices.
- **Request for LLM Training with Indicators**: The same member inquired about incorporating **indicators** in the training process of LLMs for stock market analysis.
  

---


**LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1225762035325866015)** (39 messages🔥): 

- **Mixing GPUs in LM Studio**: A member shared that LM Studio detects the cumulative VRAM across different GPU cards, such as an **RTX 4060 Ti** and a **GTX 1070**, resulting in improved performance compared to using VRAM with CPU/RAM.

- **Compatibility Queries for Advanced Matrix Extensions**: One member asked if **LM Studio** can leverage **Intel's Advanced Matrix Extensions (AMX)** present in 4th generation **Xeon processors**.

- **Explorations with ROCm Support and Mixed GPUs**: Users discussed the compatibility of various **RX 5000** and **RX 6000** series AMD GPUs with **ROCm** support in **LM Studio**, noting that some but not all cards are supported.

- **CPU Instruction Set Support Concerns**: A user experienced an issue where their processor, a **Xeon E5-2690 v2**, seemed to lack AVX2 support, conflicting with their belief based on previous experiences with **LM Studio**. A suggestion to manually install **llama.cpp** was proposed as a workaround.

- **Tesla P-Series GPU for Model Training and Fine-Tuning Debate**: There was a mention of contention around the use of **Tesla P40 GPUs** for model training and fine-tuning, with some users alleging success and others suggesting limitations due to outdated **CUDA** support.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.reddit.com/r/LocalLLaMA/comments/1alcwc1/comment/kpenylq/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.hwinfo.com/">HWiNFO - Free System Information, Monitoring and Diagnostics</a>: Free Hardware Analysis, Monitoring and Reporting. In-depth Hardware Information, Real-Time System Monitoring, Reporting &amp; more</li><li><a href="https://github.com/ggerganov/llama.cpp.git">GitHub - ggerganov/llama.cpp: LLM inference in C/C++</a>: LLM inference in C/C++. Contribute to ggerganov/llama.cpp development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1225887364434624541)** (30 messages🔥): 

- **Beta Build Number Confusions Addressed**: Facing version number confusion, a member was clarified that beta releases might not immediately reflect the correct version number and it will change in the live release.
- **LM Studio Beta 0.2.19 Release**: **LM Studio 0.2.19 Beta** was announced with support for Text Embeddings via the local server, available for download in the Beta Releases section.
- **ROCM Build Delayed but Well-Received**: It was mentioned that **ROCM builds** tend to be a version behind the main release, but despite this and some bugs, members have found it impressive and user-friendly.
- **MacOS Crashes with 0.2.19**: A user reported recurrent crashes with **0.2.19 on MacOS**, tied to the context window of a particular model, suggesting a comprehensive issue.
- **Quantized Embedding Models and GGUF Conversions**: An active discussion on quantizing additional embedding models for GGUF format resulted in new models being converted and shared, with model cards to be published shortly.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lmstudio.ai/">👾 LM Studio - Discover and run local LLMs</a>: Find, download, and experiment with local LLMs</li><li><a href="https://tenor.com/view/leonardo-dicaprio-clapping-clap-applause-amazing-gif-16078907558888063471">Leonardo Dicaprio Clapping GIF - Leonardo Dicaprio Clapping Clap - Discover &amp; Share GIFs</a>: Click to view the GIF
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/1225890061686538282)** (17 messages🔥): 

- **A Better Multi-Agent System on the Horizon**: .j4ys0n announced an upcoming release of their own multi-agent system with a **user interface (UI)**, suggesting it as a solution to the problems with existing systems and highlighting it will not require coding on the user's part like **CrewAI** does.
- **The UI Advantage for Simplicity**: .j4ys0n’s tool is positioned as a "for dummies" solution that will offer ease of use without the need for coding, in contrast to **CrewAI** which still requires code.
- **Domain Registration Alert**: .j4ys0n indicated a reluctance to share screenshots of the new project until the domain is registered to avoid "domain sniping," while heyitsyorkie emphasized the importance of securing the domain quickly.
- **Development Focus**: .j4ys0n mentioned devoting more time to developing their project over regular job tasks, believing it was the **right choice** given the progress made.
- **Editing datamodel.py as a Workaround**: mmonir shared a solution to an issue by suggesting an edit in `datamodel.py` (changing `max_tokens` to `3000`), referencing a bug with an open issue on GitHub concerning **Autogen Studio** ([Bug Report here](https://github.com/microsoft/autogen/issues/2050)).

**Link mentioned**: <a href="https://github.com/microsoft/autogen/issues/2050">[Bug]: [autogenstudio] agent llm send max_tokens: null · Issue #2050 · microsoft/autogen</a>: Describe the bug When max_tokens parameter is None, the agent send a frame /v1/chat/completions with max_tokens: null. In this case the LLM don&#39;t understand and and stop after the second token. St...

  

---


**LM Studio ▷ #[langchain](https://discord.com/channels/1110598183144399058/1167546793656062063/1226634657567870986)** (4 messages): 

- **Looking for Notebooks?**: A member inquired about having a notebook, possibly seeking shared resources or examples for their work.
- **Substack Post Teaser**: Another member provided a [Substack post](https://substack.com/home/post/p-143137776?source=queue) they wrote, hinting it could contain valuable insights or information.

**Link mentioned**: <a href="https://substack.com/home/post/p-143137776?source=queue">Switching from open ai api to local LLM</a>: Small follow up post on our last one about building a rag agent with langchain and node

  

---


**LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1225785291629920318)** (97 messages🔥🔥): 

- **GPU Target Override Enquiry**: A member asked if it was possible to override the GPU target like with `HCC_AMDGPU_TARGET=gfx1030` on Linux, referencing a conversation on [Reddit](https://www.reddit.com/r/Amd/comments/13e6jav/comment/jn8v5n5/). However, it was clarified that with the current Linux build, one is stuck with using OpenCL for GPU acceleration.

- **LM Studio 0.2.19 ROCm Preview Beta Released**: An announcement for **LM Studio 0.2.19 ROCm Preview Beta** was made, highlighting new support for text embedding models, a candidate fix for ROCm iGPU issues, and other bug fixes. The community was informed to download the beta from [LM Studio with ROCm](https://lmstudio.ai/rocm) although the version might still show 0.2.18.

- **Confusion Over ROCm Support for Different GPUs**: Community members were debating and asking questions about whether the new version of ROCm supports a mix of different GPUs like the RX 5000 and RX 6000 series, with one member stating success in running mixed AMD GPUs using hipblas and Vulkan.

- **AMD's Silence on Limiting GRE to 2.8GHz**: Users expressed frustration regarding AMD's limitation of GRE to 2.8 GHz and hoped for custom BIOS releases. One member said that only someone at AMD would release such BIOS at the risk of their job.

- **ROCm 0.2.19 Beta Debugging in Progress**: Several users reported "exit 42" errors with the latest ROCm beta, prompting the sharing of a verbose debug build for further investigation. Participants are encouraged to download the verbose build, attempt to load a model, and submit app logs for troubleshooting.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lmstudio.ai/rocm">👾 LM Studio - Discover and run local LLMs</a>: Find, download, and experiment with local LLMs</li><li><a href="https://files.lmstudio.ai/windows/LM-Studio-0.2.19-Rocm-Beta-Verbose.exe/beta/LM-Studio-0.2.19-Rocm-Beta-Verbose.exe">no title found</a>: no description found</li><li><a href="https://rocm.docs.amd.com/projects/install-on-windows/en/latest/reference/system-requirements.html">System requirements (Windows) — HIP SDK installation Windows</a>: no description found</li><li><a href="https://community.amd.com/t5/ai/how-to-run-a-large-language-model-llm-on-your-amd-ryzen-ai-pc-or/ba-p/670709">How to run a Large Language Model (LLM) on your AMD Ryzen™ AI PC or Radeon Graphics Card</a>: Did you know that you can run your very own instance of a GPT based LLM-powered AI chatbot on your Ryzen™ AI PC or Radeon™ 7000 series graphics card? AI assistants are quickly becoming essential resou...</li><li><a href="https://www.reddit.com/r/Amd/comments/13e6jav/comment/jn8v5n5/">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[model-announcements](https://discord.com/channels/1110598183144399058/1225909444727013466/1225910988717559972)** (3 messages): 

- **Model Announcement Central**: A series of new models have been released including **Starling-LM 7B**, **c4ai command r v01**, **stable-code-instruct-3b**, **dolphin 2.8 mistral 7b v02**, and **Hyperion 3.0 Mistral 7B**. The announcement invites users to [check out the models](https://huggingface.co/lmstudio-community) and stay tuned for more.

- **Introducing Qwen 1.5 32B Chat**: A new model, **Qwen 1.5 32B Chat**, part of the Qwen2 family with enhanced multi-turn conversation capabilities, has been released. Interested users can find more details on the [model card and LM Studio app](https://huggingface.co/lmstudio-community/Qwen1.5-32B-Chat-GGUF).

- **Gemma Shines at 2B**: Google's **Gemma 1.1 2B** model impresses with its performance, delivering coherent outputs at high speed, using only *3GB of memory*. The model is available, but the 7B version will require adjustments in LM Studio settings for optimization before release, as noted on the [model's page](https://huggingface.co/lmstudio-community/gemma-1.1-2b-it-GGUF).
  

---



**HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1225904914312597666)** (4 messages): 

- **New Resources and Demos Galore**: The community has shared various resources including a [neuro-symbolic agent systems repository](https://github.com/SynaLinks/HybridAGI), integration of datasets within the PyG ecosystem, and a demo for a function calling capable model, Octopus. Additional content includes visualization for hyper-graph datasets, the TensorLM Gradio UI for large language models, and the announcement of Aurora-M, a multi-lingual continually pre-trained language model.
- **Tech and Thought Leadership on Display**: Community members have released a new multi-subject image node pack, given a TED talk about the future of film in the age of AI, and published an open-source repo for replicating research papers. Other highlights feature a video on Python app safekeeping using virtual containers, a SaaS boilerplate demo, a line follower robot demo, and deepened integration between DagsHub + Colab for data management.
- **Software to Streamline Your Work**: LLMinator, a context-aware streaming chatbot, and ClipboardConqueror, a tool to reduce context switching, have been made available by the community to improve the efficiency of working with large language models.
- **Thought-Provoking Reads and Tools for AIs**: Members have contributed blog posts discussing various AI-related topics, such as evaluating SVD compression with the LASER technique and understanding diffusion models. Articles on custom architectures with HuggingFace and the levels of complexity in AI compute have also been shared.
- **Attention on Multilingual LLM Innovation**: The blog post for Aurora-M has been suggested as a potential topic for the next reading group, emphasizing the importance of multilingual and safe AI developments.

**Link mentioned**: <a href="https://huggingface.co/blog/mayank-mishra/aurora">Aurora-M: The First Open Source Biden-Harris Executive Order Red teamed Multilingual Language Model</a>: no description found

  

---


**HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1225759129054740480)** (372 messages🔥🔥): 

- **Exploring Deployment with SageMaker and TGI**: A user is considering the feasibility of deploying a model using TensorRT with SageMaker instead of TGI and seeks a way to update their kernel version on a website-based cloud compute resource.
- **Interest in Custom ChatGPT for PDFs**: In a quest to develop a unique ChatGPT app tailored for PDFs, one user is sourcing ideas to distinguish their project in a college competition.
- **Quest for ML Hardware Benchmarking Tools**: Users are discussing hardware benchmark tools for ML/AI tasks; MLPerf is recommended as a FOSS benchmark suite that includes tracks for GPT-J 6B and Llama 2 70B inference.
- **Issue with Multi-GPU Training and SageMaker**: There's an exchange about issues encountered while multi-GPU training with SageMaker and diffusers, including SIGSEGV error and environment variable messages.
- **Request for Token Count Information in Response**: A user inquires if the Hugging Face SageMaker library can provide the number of tokens in the response when calling `.predict`.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2403.10853">Just Say the Name: Online Continual Learning with Category Names Only via Data Generation</a>: In real-world scenarios, extensive manual annotation for continual learning is impractical due to prohibitive costs. Although prior arts, influenced by large-scale webly supervised training, suggest l...</li><li><a href="https://huggingface.co/HuggingFaceTB/cosmo-1b">HuggingFaceTB/cosmo-1b · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/docs/sagemaker/en/getting-started">Train and deploy Hugging Face on Amazon SageMaker</a>: no description found</li><li><a href="https://discuss.huggingface.co/t/runtimeerror-expected-tensor-for-argument-1-indices-to-have-one-of-the-following-scalar-types-long-int-but-got-mpsfloattype-instead-while-checking-arguments-for-embedding/80417">RuntimeError: Expected tensor for argument #1 &#39;indices&#39; to have one of the following scalar types: Long, Int; but got MPSFloatType instead (while checking arguments for embedding)</a>: I am trying to train a multi-modal model by taking in image and text input to output a text.  Here is my architecture;  (Assuming batch size=1)  I use a ViT (from hugging face) to convert images (1, 3...</li><li><a href="https://cookbook.openai.com/examples/question_answering_using_embeddings">Question answering using embeddings-based search | OpenAI Cookbook</a>: no description found</li><li><a href="https://huggingface.co/NexaAIDev/Octopus-v2">NexaAIDev/Octopus-v2 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/docs/sagemaker/en/inference">Deploy models to Amazon SageMaker</a>: no description found</li><li><a href="https://huggingface.co/docs/accelerate/v0.28.0/en/package_reference/launchers#accelerate.notebook_launcher">Launchers</a>: no description found</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/18o6z49/is_it_possible_to_queue_batch_img2img_with_a_new/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://huggingface.co/blog/noob_intro_transformers">Total noob’s intro to Hugging Face Transformers</a>: no description found</li><li><a href="https://huggingface.co/facebook/bart-large-cnn">facebook/bart-large-cnn · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/blog/embedding-quantization">Binary and Scalar Embedding Quantization for Significantly Faster &amp; Cheaper Retrieval</a>: no description found</li><li><a href="https://learnbybuilding.ai/tutorials/rag-">no title found</a>: no description found</li><li><a href="https://github.com/Haoming02/sd-webui-diffusion-cg">GitHub - Haoming02/sd-webui-diffusion-cg: An Extension for Automatic1111 Webui that performs color grading based on the latent tensor value range</a>: An Extension for Automatic1111 Webui that performs color grading based on the latent tensor value range - Haoming02/sd-webui-diffusion-cg</li><li><a href="https://huggingface.co/docs/diffusers/en/tutorials/basic_training">Train a diffusion model</a>: no description found</li><li><a href="https://www.reddit.com/r/StableDiffusi">Reddit - Dive into anything</a>: no description found</li><li><a href="https://huggingface.co/spaces/Yakova/ollama-mistral">Streamer - a Hugging Face Space by Yakova</a>: no description found</li><li><a href="https://blog.salad.com/ollama-deploy-chatgpt/">Your own ChatGPT for $0.04/hr - With Ollama, ChatUI &amp; Salad</a>: We explore how to build your own ChatGPT with Ollama, Huggingface Chat UI and SaladCloud for just $0.04 per hour.</li><li><a href="https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)```">CUDA semantics &mdash; PyTorch 2.2 documentation</a>: no description found</li><li><a href="https://github.com/huggingface/peft/blob/main/examples/int8_training/peft_bnb_whisper_large_v2_training.ipynb">peft/examples/int8_training/peft_bnb_whisper_large_v2_training.ipynb at main · huggingface/peft</a>: 🤗 PEFT: State-of-the-art Parameter-Efficient Fine-Tuning. - huggingface/peft</li><li><a href="https://github.com/philschmid/huggingface-sagemaker-workshop-series/blob/main/workshop_1_getting_started_with_amazon_sagemaker/lab_1_default_training.ipynb">huggingface-sagemaker-workshop-series/workshop_1_getting_started_with_amazon_sagemaker/lab_1_default_training.ipynb at main · philschmid/huggingface-sagemaker-workshop-series</a>: Enterprise Scale NLP with Hugging Face &amp; SageMaker Workshop series - philschmid/huggingface-sagemaker-workshop-series</li><li><a href="https://learnbybuilding.ai/tutorials/rag-from-scratch">A beginner's guide to building a Retrieval Augmented Generation (RAG) application from scratch</a>: This post will teach you the fundamental intuition behind RAG while providing a simple tutorial to help you get started.</li><li><a href="https://github.com/Mikubill/sd-webui-controlnet">GitHub - Mikubill/sd-webui-controlnet: WebUI extension for ControlNet</a>: WebUI extension for ControlNet. Contribute to Mikubill/sd-webui-controlnet development by creating an account on GitHub.</li><li><a href="https://github.com/guananya/AllenNLP-Coreference-Resolution-in-Python-Readable-clusters/blob/master/allennlp_coref.py">AllenNLP-Coreference-Resolution-in-Python-Readable-clusters/allennlp_coref.py at master · guananya/AllenNLP-Coreference-Resolution-in-Python-Readable-clusters</a>: Using AllenNLP Coreference Resolution in Python (getting clusters which are ACTUALLY readable) - guananya/AllenNLP-Coreference-Resolution-in-Python-Readable-clusters</li><li><a href="https://huggingface.co/ProsusAI/finbert">ProsusAI/finbert · Hugging Face</a>: no description found</li><li><a href="https://arxiv.org/abs/2304.14241">Entity-Level Sentiment Analysis (ELSA): An exploratory task survey</a>: This paper explores the task of identifying the overall sentiment expressed towards volitional entities (persons and organizations) in a document -- what we refer to as Entity-Level Sentiment Analysis...
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1225782561343799448)** (2 messages): 

- **Seeking Knowledge on Knowledge Graphs**: A member expressed interest in learning about **knowledge graphs** and their applications, asking for resource recommendations.
- **Building Collate, Seeking Learning Experiences**: [Collate](https://collate.one/preview) is a new platform aimed at transforming everyday learning for students, professionals, and content creators. The creator is seeking feedback and experiences related to learning challenges and is offering early access and a 15-minute call to discuss further. [Schedule a call](https://calendly.com/vel-yan/15min).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://collate.one/preview">Collate Preview</a>: Transform your everyday learning</li><li><a href="https://calendly.com/vel-yan/15min">15 Minute Meeting - Vel Yanchina</a>: no description found
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1225755414554214431)** (11 messages🔥): 

- **PIDNet Improves Semantic Segmentation**: A [new paper](https://arxiv.org/abs/2206.02066) introduces PIDNet, a three-branch network architecture inspired by PID controllers, designed to enhance real-time semantic segmentation by effectively integrating detailed, context, and boundary information.
- **LLMs Vulnerable to 'Crescendo' Jailbreak**: Mark Russinovich shared a link to 'Crescendo', a [potentially concerning jailbreak](https://crescendo-the-multiturn-jailbreak.github.io/) for large language models that aims to bypass ethical boundaries set to prevent the generation of harmful content.
- **Cohere Command-R-plus Caught in Jailbreak Snare**: A LinkedIn [post highlights](https://www.linkedin.com/posts/enkryptai_command-r-red-teaming-report-activity-7182087079974117377-ujmT) vulnerabilities in Cohere's Command-R-plus system exposed to jailbreak attacks.
- **Forwarding the Mixture-of-Depths Concept**: The new [Mixture-of-Depths (Modes) proposal](https://arxiv.org/abs/2404.02258) suggests a dynamic allocation of Transformer computations across a sequence, potentially enhancing efficiency without compromising flexibility.
- **Exploring Multi-Document Solutions with llamaindex**: A user shared [a blog post](https://ai.gopubby.com/unlocking-the-power-of-multi-document-agents-with-llamaindex-d09e4d7dfe0e?gi=947416d131c6) about leveraging llamaindex to create multi-document RAG solutions for improved information retrieval.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2404.02258">Mixture-of-Depths: Dynamically allocating compute in transformer-based language models</a>: Transformer-based language models spread FLOPs uniformly across input sequences. In this work we demonstrate that transformers can instead learn to dynamically allocate FLOPs (or compute) to specific ...</li><li><a href="https://arxiv.org/abs/2206.02066">PIDNet: A Real-time Semantic Segmentation Network Inspired by PID Controllers</a>: Two-branch network architecture has shown its efficiency and effectiveness in real-time semantic segmentation tasks. However, direct fusion of high-resolution details and low-frequency context has the...</li><li><a href="https://crescendo-the-multiturn-jailbreak.github.io/">Crescendo </a>: The Multi-Turn LLM Jailbreak Attack</li><li><a href="https://www.youtube.com/watch?v=_j7JEDWuqLE">Hugging Face + Langchain in 5 mins | Access 200k+ FREE AI models for your AI apps</a>: Learn how to use Hugging Face, and get access to 200k+ AI models while building in Langchain for FREE.🔗 Links- Hugging Face tutorials: https://hf.co/tasks- ...
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1225741246526390374)** (44 messages🔥): 

- **Neuro-Symbolic AGI on GitHub**: A new open-source neuro-symbolic AGI designed for behavior programming using Graph-based Prompt Programming has been introduced by a French AI startup. The project is now seeking community feedback and is showcased on [GitHub](https://github.com/SynaLinks/HybridAGI).
- **Open Source Paper Replication Repository**: A repository aimed at upskilling through replication of AI & ML research papers has been launched, inviting contributors to star, advise, and open PRs. Check out the repository here: [PaperReplica on GitHub](https://github.com/hegdeadithyak/PaperReplica).
- **Managing Audio Datasets via Gradio**: A new Gradio interface for creating and managing large audio datasets has been shared, intended for tasks like segmenting audiobooks and transcribing. The tool is available for use on [GitHub](https://github.com/maepopi/audio-dataset-manager).
- **RNN for MNIST Handwritten Digits**: A self-coded vanilla RNN using numpy to classify MNIST digits has been released and its code is available for review. Visit the project on [GitHub](https://github.com/suprasauce/RNN_MEDIUM).
- **Image Search for Local Folders**: A project named 'Where's My Pic?' offers a Google Image Search-like experience for local folders, assisting in finding images quickly. A demonstration can be found on [YouTube](https://www.youtube.com/watch?v=oVJsJ0e6jWk).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/dhanikkcs/status/1776179274640400502">Tweet from Kheem Chandra (@dhanikkcs)</a>: I have created a telegram bot . It is powered by Gemini 1.5 Pro.  You can chat with it in Video, Image, or Text format.  bot name: &#34;int_gem_bot&#34; link: https://telegram.me/int_gem_bot  Give it ...</li><li><a href="https://thebeastbot.com/welcome/">The creative genius of MrBeast as an AI Bot :)</a>: I&#039;m the Beast of all AI bots! I&#039;ve been loaded up with mountains of MrBeast&#039;s wildest, most innovative content. It&#039;s like having exclusive backstage access to his mind-boggling bra...</li><li><a href="https://huggingface.co/spaces/not-lain/RMBG1.4-with-imageslider">RMBG1.4 with imageslider - a Hugging Face Space by not-lain</a>: no description found</li><li><a href="https://huggingface.co/spaces/not-lain/RAG-Chatbot">RAG - a Hugging Face Space by not-lain</a>: no description found</li><li><a href="https://huggingface.co/spaces/TencentARC/BrushNet">BrushNet - a Hugging Face Space by TencentARC</a>: no description found</li><li><a href="https://arxiv.org/html/2403.17887v1">The Unreasonable Ineffectiveness of the Deeper Layers</a>: no description found</li><li><a href="https://github.com/ehristoforu/TensorLM-webui">GitHub - ehristoforu/TensorLM-webui: Simple and modern webui for LLM models based LLaMA.</a>: Simple and modern webui for LLM models based LLaMA. - ehristoforu/TensorLM-webui</li><li><a href="https://github.com/RooTender/augmentator">GitHub - RooTender/augmentator: Ready-to-use tool for image augmentation</a>: Ready-to-use tool for image augmentation. Contribute to RooTender/augmentator development by creating an account on GitHub.</li><li><a href="https://www.youtube.com/watch?v=oVJsJ0e6jWk">Where&#39;s My Pic Demo</a>: Hello everyone, I&#39;m Om Alve and in this video I&#39;m giving a demo of my project called &#39;Where&#39;s my pic?&#39;. This project solves the problem of searching through ...</li><li><a href="https://github.com/abhaskumarsinha/Corpus2GPT">GitHub - abhaskumarsinha/Corpus2GPT: CustomGPTBuilder: A project enabling users to train their own GPT models on diverse datasets, including local languages and various corpus types, using Keras and compatible with TensorFlow, PyTorch, or JAX backends for subsequent storage or sharing.</a>: CustomGPTBuilder: A project enabling users to train their own GPT models on diverse datasets, including local languages and various corpus types, using Keras and compatible with TensorFlow, PyTorch...</li><li><a href="https://rapidapi.com/NextAPI/api/cheapest-gpt-4-turbo-gpt-4-vision-chatgpt-openai-ai-api/">Cheapest GPT-4 Turbo, GPT 4 Vision, ChatGPT OpenAI AI API API Documentation (NextAPI) | RapidAPI</a>: no description found</li><li><a href="https://github.com/hegdeadithyak/PaperReplica">GitHub - hegdeadithyak/PaperReplica: We Replicate Research Papers in the field of AI &amp; ML.</a>: We Replicate Research Papers in the field of AI &amp; ML. - hegdeadithyak/PaperReplica</li><li><a href="https://github.com/SynaLinks/HybridAGI">GitHub - SynaLinks/HybridAGI: The Programmable Neuro-Symbolic AGI that lets you program its behavior using Graph-based Prompt Programming: for people who want AI to behave as expected</a>: The Programmable Neuro-Symbolic AGI that lets you program its behavior using Graph-based Prompt Programming: for people who want AI to behave as expected - SynaLinks/HybridAGI</li><li><a href="https://github.com/suprasauce/RNN_MEDIUM">GitHub - suprasauce/RNN_MEDIUM</a>: Contribute to suprasauce/RNN_MEDIUM development by creating an account on GitHub.</li><li><a href="https://git.ecker.tech/mrq/ai-voice-cloning/">ai-voice-cloning</a>: Collection of utilities aimed to voice clone through AI</li><li><a href="https://github.com/maepopi/audio-dataset-manager">GitHub - maepopi/audio-dataset-manager: An all in one tool designed to prepare audiobooks or large audios for TTS and voice cloning.</a>: An all in one tool designed to prepare audiobooks or large audios for TTS and voice cloning. - maepopi/audio-dataset-manager
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1225818367861194863)** (10 messages🔥): 

- **Searching for the Right Channel**: A member wondered if another channel, <#879548962464493622>, would be more appropriate for specific questions.
- **Inquiring Minds Want to Know about Paper Reading Events**: Members have shown interest in paper reading events, confirming that such events typically occur every weekend. Last week's event featured an impressive presenter and was recorded.
- **Looking for Learning Resources**: A query was raised about finding resources to understand the foundational blocks of models for tweaking and building new ones, regardless of a specific model.
- **Repository of Knowledge Awaits**: Recordings and notifications of the paper reading sessions are compiled in a [GitHub repository](https://github.com/isamu-isozaki/huggingface-reading-group), with the most recent recording yet to be added. Discord events are the current go-to for session notifications.
- **General Guidance for Model Exploration**: When asked for guidance on how to understand model codebases, a member inquired about specific domains of knowledge required to navigate and comprehend the coding aspects of models without focusing on any particular one.

**Link mentioned**: <a href="https://github.com/isamu-isozaki/huggingface-reading-group">GitHub - isamu-isozaki/huggingface-reading-group: This repository&#39;s goal is to precompile all past presentations of the Huggingface reading group</a>: This repository&#39;s goal is to precompile all past presentations of the Huggingface reading group - isamu-isozaki/huggingface-reading-group

  

---


**HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1225803402806558810)** (12 messages🔥): 

- **HuggingFace as Git for ML Models**: Users discussed the similarity between HuggingFace model repositories and Git, where you can create a repo and commit and push updates just like with code.
  
- **Monitoring GPU Usage During Training**: In response to a user's query on how to monitor GPU usage while training models, a HuggingFace Space named [Model Memory Usage](https://huggingface.co/spaces/hf-accelerate/model-memory-usage) was recommended.

- **Manipulating Parquet Files without Pandas**: A user sought alternatives to **Pandas** for dropping a column from a parquet file. It was suggested to use the `from_parquet` method from the `datasets` library available in the [HuggingFace documentation](https://huggingface.co/docs/datasets/v2.18.0/en/package_reference/main_classes#datasets.Dataset.from_parquet).

- **Seeking Resources on Diffusion Models for Video Quality**: A user asked for assistance and resources related to improving video quality using diffusion models, seeking any relevant academic papers.

- **Training XCLIP with More Frames**: One member shared their experience trying to pretrain an **XCLIP** model with more frames than the pretrained versions. They faced issues with stagnant losses and NaNs, seeking advice on training from scratch with extended frame capacities as described in the [XCLIP documentation](https://huggingface.co/docs/transformers/en/model_doc/xclip#transformers.XCLIPModel).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/hf-accelerate/model-memory-usage">Model Memory Utility - a Hugging Face Space by hf-accelerate</a>: no description found</li><li><a href="https://huggingface.co/docs/transformers/en/model_doc/xclip#transformers.XCLIPModel">X-CLIP</a>: no description found
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1225957005756334122)** (24 messages🔥): 

- **Fine-tuning Mistral7b for Specific Data Extraction**: A member queried whether they could fine-tune **Mistral7b** for JSON data extraction by using the cleaned results of its output. They were pondering the need for an LLM versus a more specialized model for similar input-output formats.

- **Parsing Tweets Without Twitter API**: A member sought alternatives for scraping tweets without using Twitter's complex API, hinting at a desire for a less complicated tool or method to achieve this task.

- **Colab Pro+ struggles with WizardLM models**: A participant faced out-of-memory errors trying to load cognitivecomputations' **WizardLM-13B** and **WizardLM-7B** models on Google Colab Pro+, despite trying different GPUs and looking for solutions.

- **10M Context Window Feasibility in Gemini 1.5**: The **Gemini 1.5** paper's claim of a 10M context window sparked a discussion, with a member seeking explanations on how it calculates a substantial attention matrix. Another member shared a [potential relevant paper](https://arxiv.org/abs/2310.01889) that might illustrate the method used for this achievement.

- **Imputing Null Values with LLM**: A member expressed the need to impute null values in a dataset containing 'object' datatype fields with an LLM based on context, seeking references or assistance on how to proceed.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2310.01889">Ring Attention with Blockwise Transformers for Near-Infinite Context</a>: Transformers have emerged as the architecture of choice for many state-of-the-art AI models, showcasing exceptional performance across a wide range of AI applications. However, the memory demands impo...</li><li><a href="https://bhosmer.github.io/mm/ref.html">mm ref</a>: no description found</li><li><a href="https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)">CUDA semantics &mdash; PyTorch 2.2 documentation</a>: no description found
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1225780367022096546)** (9 messages🔥): 

- **PEFT Shrinks llava2 But Faces Deployment Issues**: A member is using the **PEFT technique** to reduce the size of the llava2 model but encounters problems when trying to run the reduced model on another machine. The issue seems to be related to the model being in **safetensors format**, leading to an error about a missing `pytorch_model_bin` file.

- **Deploying Safetensors Formatted Models**: In response to the above issue, a suggestion was made to check the use of `use_safetensors=True` which might resolve the problem of deploying the reduced model safely formatted as safetensors.

- **Learning Curve for NLP Beginners**: A new member seeking advice on whether to learn **transformers, LSTM, GRU, or bidirectional LSTM/GRU** was directed to a [Stanford CS224N YouTube course](https://www.youtube.com/playlist?list=PLoROMvodv4rMFqRtEuo6SGjY4XbRIVRd4), a resource that comprehensively covers Natural Language Processing with Deep Learning.

- **Request for Euler/Euler-A Sampler Insights**: A member expressed difficulty in finding blog-type resources on the **euler/euler-a sampler** and is seeking suggestions, having found only the k-diffusion repo to reference.

- **LaBSE Model Export Challenges in OpenSearch**: An individual encountered errors when trying to use "sentence-transformers/LaBSE" as a custom model with **OpenSearch** and faced difficulties after attempting to export the model to TorchScript using a Python script.

**Link mentioned**: <a href="https://www.youtube.com/playlist?list=PLoROMvodv4rMFqRtEuo6SGjY4XbRIVRd4">Stanford CS224N: Natural Language Processing with Deep Learning | 2023</a>: Natural language processing (NLP) is a crucial part of artificial intelligence (AI), modeling how people share information. In recent years, deep learning ap...

  

---


**HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1226986017958138049)** (1 messages): 

- **API Recorder Hits the Stage**: Gradio's latest update 4.26.0 features an 🎥**API Recorder** that records interactions with any Gradio app and auto-generates the corresponding Python or JavaScript code. This can be accessed through the `View API` page to simplify recreating app actions programmatically.
- **Squashing Bugs for Speed**: The update also addresses a critical **bug** that previously led to slow page load times in Gradio version 4.25.0.
- **Chatbot UI Crashes Resolved**: Fixed a significant issue where rapid chatbot updates could crash the UI, ensuring smoother user experiences.
- **Check Out the Full Changelog**: For a comprehensive list of bug fixes and features in the latest release, users can view the full changelog at [Gradio's Changelog](https://www.gradio.app/changelog#4-26-0).
  

---



**Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1225746638346129465)** (34 messages🔥): 

- **Exploring Rustlings and Ziglings**: A member discussed their experience with programming exercises from [Rustlings](https://github.com/rust-lang/rustlings) and [Ziglings](https://codeberg.org/ziglings/exercises/), and discovered an equivalent for Mojo called [Mojolings](https://github.com/dbusteed/mojolings).
- **Var vs. Let in Mojo**: There was a clarification that `var` is used for lazily assigned variables in Mojo and it's not going away, even though `let` got removed, with a resource provided to learn about Mojo's use of `var` [here](https://docs.modular.com/mojo/manual/variables#declared-variables).
- **Mojo for Web Applications**: Discussion about the potential for Mojo in web development yielded info about a simple and fast HTTP framework for Mojo called [lightbug_http](https://github.com/saviorand/lightbug_http).
- **Mojo as a General Purpose Language**: Members reassured one another that Mojo is indeed a general purpose language designed with AI/ML in mind, and highlighted Mojo's young but evolving nature.
- **Seeking Documentation and Learning Resources for Mojo**: Members inquired about books or comprehensive documentation for learning Mojo, with the closest current recommendation being the [Mojo Manual](https://docs.modular.com/mojo/manual).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.modular.com/mojo/manual">Mojo Manual | Modular Docs</a>: A comprehensive guide to the Mojo programming language.</li><li><a href="https://docs.modular.com/mojo/manual/variables#declared-variables).">Variables | Modular Docs</a>: Introduction to Mojo variables.</li><li><a href="https://github.com/modularml/max">GitHub - modularml/max: A collection of sample programs, notebooks, and tools which highlight the power of the MAX platform</a>: A collection of sample programs, notebooks, and tools which highlight the power of the MAX platform - modularml/max</li><li><a href="https://github.com/saviorand/lightbug_http">GitHub - saviorand/lightbug_http: Simple and fast HTTP framework for Mojo! 🔥</a>: Simple and fast HTTP framework for Mojo! 🔥. Contribute to saviorand/lightbug_http development by creating an account on GitHub.</li><li><a href="https://github.com/rust-lang/rustlings">GitHub - rust-lang/rustlings: :crab: Small exercises to get you used to reading and writing Rust code!</a>: :crab: Small exercises to get you used to reading and writing Rust code! - rust-lang/rustlings</li><li><a href="https://codeberg.org/ziglings/exercises/.">exercises</a>: Learn the ⚡Zig programming language by fixing tiny broken programs.</li><li><a href="https://github.com/dbusteed/mojolings">GitHub - dbusteed/mojolings: Learn to read and write Mojo code by fixing small programs</a>: Learn to read and write Mojo code by fixing small programs - dbusteed/mojolings
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/1225851676305526835)** (7 messages): 

- **Modular's Tweet Cascade Begins**: Modular shared a tweet themed around Modular's innovative strides in technology. View the tweet [here](https://twitter.com/Modular/status/1776287802533245372).
- **Advancing the Modular Movement**: Another Modular tweet hints at further advancements, suggesting a sustained push in their tech development. The tweet can be seen [here](https://twitter.com/Modular/status/1776287865242300621).
- **A Sneak Peek into Modular's Future Plans**: A tweet by Modular appears to tease upcoming projects or developments in their ecosystem. Check out the tweet [here](https://twitter.com/Modular/status/1776287868710998188).
- **Rising to New Challenges**: Modular issues a tweet that may discuss overcoming challenges or setting new goals. The full content is available [here](https://twitter.com/Modular/status/1776356366309113974).
- **Continuing the Modular Story**: The story of Modular's progress continues in another tweet, which could be building on previous announcements or achievements. View the Tweet [here](https://twitter.com/Modular/status/1776356370004242655).
- **Charting Modular's Path Forward**: Modular's Twitter post suggests an outline of the path ahead for the company or its technology. The post is accessible [here](https://twitter.com/Modular/status/1776356373682696701).
- **Modular's Vision Unfolds**: A tweet from Modular presents their vision, possibly revealing new insights or directions for the company. Read the tweet [here](https://twitter.com/Modular/status/1777447869907431562).
  

---


**Modular (Mojo 🔥) ▷ #[✍︱blog](https://discord.com/channels/1087530497313357884/1098713717509730466/1226269140281720889)** (2 messages): 

- **Modular Joins Forces with AWS**: Modular has announced a [partnership with Amazon Web Services (AWS)](https://www.modular.com/blog/modular-partners-with-amazon-web-services-aws-to-bring-max-to-aws-services), aiming to integrate the **MAX Platform** with AWS services, thereby providing innovative AI features on a global scale. Bratin Saha, AWS VP of Machine Learning & AI services, emphasized the partnership's role in accelerating the adoption of GenAI and traditional AI use cases by AWS customers.

- **Open Collaboration on Mojo Standard Library**: Modular made a call to the community for contributions to the [Mojo standard library](https://www.modular.com/blog/how-to-contribute-to-mojo-standard-library-a-step-by-step-guide), providing a comprehensive guide on how to contribute, from identifying issues on GitHub to creating successful pull requests. The guide follows Modular's recent milestone of open sourcing the Mojo standard library, inviting improvements ranging from documentation to code changes.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.modular.com/blog/modular-partners-with-amazon-web-services-aws-to-bring-max-to-aws-services">Modular: Modular partners with Amazon Web Services (AWS) to bring MAX to AWS services</a>: We are building a next-generation AI developer platform for the world. Check out our latest post: Modular partners with Amazon Web Services (AWS) to bring MAX to AWS services</li><li><a href="https://www.modular.com/blog/how-to-contribute-to-mojo-standard-library-a-step-by-step-guide">Modular: How to Contribute to Mojo Standard Library: A Step-by-Step Guide</a>: We are building a next-generation AI developer platform for the world. Check out our latest post: How to Contribute to Mojo Standard Library: A Step-by-Step Guide
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[ai](https://discord.com/channels/1087530497313357884/1103420074372644916/)** (1 messages): 

rxzfn: There is a moveable product like this, but using pcie
  

---


**Modular (Mojo 🔥) ▷ #[tech-news](https://discord.com/channels/1087530497313357884/1151359062789857280/1226067117544050750)** (2 messages): 

- **Repository Access Issue Resolved**: A brief exchange indicated there was a problem accessing a repository, which was promptly rectified with an updated, *working link*. No further context or the actual link was provided.
  

---


**Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1225742384814624829)** (336 messages🔥🔥): 

- **Exploring Mojo's Parameter Abilities**: Through a deep dive into Mojo's parameter usage, it was discovered that calculations can be performed exclusively at parameter time, allowing for innovative metaprogramming. However, this exposes a scoping issue where an operation (`a + b`) executed in a function signature doesn't yield the same result as when the operation is stored in a named, inferred parameter (`_L = a + b`).

- **The Complexities of Compile-Time Evaluation**: A long conversation unfolded around difficulties faced when performing certain type operations at compile time. It highlighted the complexity inherent to Mojo's compiler and the type system's handling of operations like adding, which aren't straightforward due to the requirement of proofs for simple equations like `a + b == b + a`.

- **Reference and Lifetime Intricacies in Mojo**: The chat discussed potential issues and methodologies around the `Reference` types and their lifetimes when using the `@value` decorator and `init` methods. It was pointed out that the `Reference` and lifetime mechanics might require more clarification and documentation for ease of use.

- **Anticipation for Future Open Source Contributions**: Users expressed anticipation for when Mojo becomes open source, hoping the community will contribute with ports for other systems like BSDs. This open-sourcing is expected to enable wider adaptation and integration of Mojo.

- **RustPython as a Case Study for Language Implementation**: RustPython was examined as an example of reimplementing a language's standard library, considering its slower execution times compared to CPython. The discussion acknowledged that while such projects are cool and ambitious, they often lack the extensive optimizations seen in longer-established counterparts.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/angry-anger-pixar-inside-out-aaah-gif-5628546">Angry Anger GIF - Angry Anger Pixar - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/you-make-a-compelling-argument-simon-hardwick-blood-and-treasure-you-make-a-persuasive-argument-you-make-a-strong-argument-gif-26852864">You Make A Compelling Argument Simon Hardwick GIF - You Make A Compelling Argument Simon Hardwick Blood And Treasure - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://hirrolot.github.io/posts/rust-is-hard-or-the-misery-of-mainstream-programming.html">Rust Is Hard, Or: The Misery of Mainstream Programming</a>: no description found</li><li><a href="https://github.com/modularml/mojo/issues/1702#issuecomment-1940230390">[BUG]: Behaviour of the type checker is inconsistent (some expressions are manipulated prior to type-checking) · Issue #1702 · modularml/mojo</a>: Bug description I&#39;ve recently been trying to understand how the type checker reasons about the equality of types. I&#39;ve noticed a few inconsistencies and potential bugs. These are demonstrated ...</li><li><a href="https://github.com/modularml/mojo/blob/main/stdlib/src/collections/list.mojo#L41-L70">mojo/stdlib/src/collections/list.mojo at main · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/modularml/mojo/issues/2100">[Feature Request] Allow parametric materialization · Issue #2100 · modularml/mojo</a>: Review Mojo&#39;s priorities I have read the roadmap and priorities and I believe this request falls within the priorities. What is your request? Allow parametric materialization of nonmaterializable ...</li><li><a href="https://github.com/modularml/max">GitHub - modularml/max: A collection of sample programs, notebooks, and tools which highlight the power of the MAX platform</a>: A collection of sample programs, notebooks, and tools which highlight the power of the MAX platform - modularml/max</li><li><a href="https://www.modular.com/blog/mojo-python-calculating-and-plotting-a-valentines-day-using-mojo-and-python">Modular: Mojo🔥 ♥️ Python: Calculating and plotting a Valentine’s day ♥️ using Mojo and Python</a>: We are building a next-generation AI developer platform for the world. Check out our latest post: Mojo🔥 ♥️ Python: Calculating and plotting a Valentine’s day ♥️ using Mojo and Python</li><li><a href="https://github.com/modularml/devrel-extras/tree/main/blogs/">devrel-extras/blogs at main · modularml/devrel-extras</a>: Contains supporting materials for developer relations blog posts, videos, and workshops - modularml/devrel-extras</li><li><a href="https://youtu.be/pdJQ8iVTwj8?si=ML7lZfXAel9zEgj0&t=5763">Chris Lattner: Future of Programming and AI | Lex Fridman Podcast #381</a>: Chris Lattner is a legendary software and hardware engineer, leading projects at Apple, Tesla, Google, SiFive, and Modular AI, including the development of S...</li><li><a href="https://github.com/mo">mo - Overview</a>: mo has 49 repositories available. Follow their code on GitHub.</li><li><a href="https://github.com/modularml/mojo/blob/nightly/stdlib/docs/style-guide.md">mojo/stdlib/docs/style-guide.md at nightly · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/modularml/mojo/issues/1837">[BUG]: Compiler Crash with Self Referential Variant · Issue #1837 · modularml/mojo</a>: Bug description from utils.variant import Variant from collections.vector import DynamicVector @value struct Value(CollectionElement): alias Variant = Variant[Float64, DynamicVector[Value]] var _va...</li><li><a href="https://github.com/modularml/mojo/blob/main/stdlib/src/collections/dict.mojo#L48-L94">mojo/stdlib/src/collections/dict.mojo at main · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/RustPython/RustPython">GitHub - RustPython/RustPython: A Python Interpreter written in Rust</a>: A Python Interpreter written in Rust. Contribute to RustPython/RustPython development by creating an account on GitHub.</li><li><a href="https://github.com/python/cpython/blob/main/Objects/dictobject.c">cpython/Objects/dictobject.c at main · python/cpython</a>: The Python programming language. Contribute to python/cpython development by creating an account on GitHub.</li><li><a href="https://github.com/modularml/mojo/blob/main/stdlib/src/collections/dict.mojo">mojo/stdlib/src/collections/dict.mojo at main · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[community-projects](https://discord.com/channels/1087530497313357884/1151418679578337311/1225755101755346996)** (18 messages🔥): 

- **Special Functions Now in Mojo**: An update to the Specials package introduces several **elementary mathematical functions** such as `exp`, `exp2`, `expm1`, `log`, and `log1p`. These implementations prioritize numerical accuracy over FLOPS and benchmarks can be found in the [package repository](https://github.com/leandrolcampos/specials).

- **SICP Gets Mojofied**: The classic textbook "Structure and Interpretation of Computer Programs" is being ported to Mojo language in the [sicp_mojo](https://github.com/Brian-M-J/sicp_mojo) project, currently referencing the JavaScript version.

- **Mojo Algorithms Collective Initiative**: A member is planning to rewrite popular algorithms in Mojo, such as Dijkstra's and different sorting methods, and is interested in coordinated efforts.

- **One-stop Mojo Packages Repo**: Community members can share their Mojo packages through PRs in the [mojo-packages repository](https://github.com/kernhanda/mojo-packages), which aims to function as a central hub until an official package manager is available.

- **Mambamojo Collaborative Project**: A GitHub repository called [mamba.mojo](https://github.com/aizzf/mamba.mojo) is seeking collaborators to work on implementing Mamba in pure Mojo, from models to inference and training.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/kernhanda/mojo-packages">GitHub - kernhanda/mojo-packages: A place to find and share packages for the Mojo language</a>: A place to find and share packages for the Mojo language - kernhanda/mojo-packages</li><li><a href="https://github.com/aizzf/mamba.mojo">GitHub - aizzf/mamba.mojo: Mamba in pure mojo from model to inference and train.</a>: Mamba in pure mojo from model to inference and train. - aizzf/mamba.mojo</li><li><a href="https://github.com/kernhanda/mojopack">GitHub - kernhanda/mojopack: mojopack is a tool for managing packages for the Mojo programming language</a>: mojopack is a tool for managing packages for the Mojo programming language - kernhanda/mojopack</li><li><a href="https://github.com/leandrolcampos/specials">GitHub - leandrolcampos/specials: Special functions with hardware acceleration</a>: Special functions with hardware acceleration. Contribute to leandrolcampos/specials development by creating an account on GitHub.</li><li><a href="https://github.com/Hammad-hab/pkm">GitHub - Hammad-hab/pkm: Mojo&#39;s unoffical package manager</a>: Mojo&#39;s unoffical package manager. Contribute to Hammad-hab/pkm development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[community-blogs-vids](https://discord.com/channels/1087530497313357884/1151418796993683477/1226416943242936350)** (7 messages): 

- **"Joy of Mojo" blog launch**: A new community website called [Joy of Mojo](https://joyofmojo.com/) has been introduced, where individuals can share demo programs created while exploring the Mojo language. Although there were initial issues with GitHub Pages, the site appears to be functioning again, and the community is invited to contribute and discuss.

- **Link Troubles for "Joy of Mojo"**: The [Joy of Mojo](https://joyofmojo.com/) website faced hosting issues on GitHub Pages, displaying errors for some users, but these seem to be resolved now, assuring users of the site's accessibility.

- **Mojo Package Sharing Initiatives**: Community members have created repositories like [mojo-packages](https://github.com/kernhanda/mojo-packages) and [mojopack](https://github.com/kernhanda/mojopack) on GitHub for sharing packages for the Mojo language, complementing the collaborative spirit within the community.

- **Dynamic Mojo Evolution Acknowledged**: The rapid **evolution of Mojo** is recognized, with expectations set that some shared content may become outdated within months, highlighting the ongoing development and changes within the language ecosystem.

- **Educative Mojo Trolling**: A community member shared a [YouTube video](https://youtu.be/6cyCeJwgNjc) designed to educate and surprise viewers by demonstrating a star pattern in Python, which is revealed to be Mojo code written with a Mojo plugin in VSCode, drawing attention to Mojo's Python compatibility.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://joyofmojo.com/">Joy of Mojo 🔥</a>: This is Joy of Mojo</li><li><a href="https://github.com/kernhanda/mojo-packages">GitHub - kernhanda/mojo-packages: A place to find and share packages for the Mojo language</a>: A place to find and share packages for the Mojo language - kernhanda/mojo-packages</li><li><a href="https://github.com/kernhanda/mojopack">GitHub - kernhanda/mojopack: mojopack is a tool for managing packages for the Mojo programming language</a>: mojopack is a tool for managing packages for the Mojo programming language - kernhanda/mojopack
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1225758883436429312)** (71 messages🔥🔥): 

- **Python Interop Enhancements**: A member has been working on **CPython** interoperability by implementing *PyMethodDef*, *PyCFunction_New*, and *PyModule_NewObject* in Mojo. They highlighted progress in reference counting without bugs and believe their work lays a promising foundation for further planning of Python interop. The related development work is available on [GitHub](https://github.com/rd4com/mojo_branch/tree/nightly).

- **Getting Started for New Contributors**: New contributors are guided to start by looking into "good first issues", with links provided to the [changelog](https://docs.modular.com/mojo/changelog#week-of-2023-01-30) and the [contributing guidelines](https://github.com/modularml/mojo/blob/main/CONTRIBUTING.md) on Mojo's GitHub repository.

- **Discussions on Signed-off Commit Best Practices**: In a lively exchange about pull request practices, a member learned the importance of proper signing-off commits and was guided on how to amend commit authorship and use `git config` to correctly attribute their work. Relevant GitHub documentation was linked for configuring the username in git, and VSCode was recommended as a tool with an automatic sign-off option.

- **Soliciting Feedback on Standard Library Test Practices**: A member has created a [discussion](https://github.com/modularml/mojo/discussions/2234) on GitHub to gather input on improving List and String slicing tests in the Mojo Standard Library, as well as proposing more descriptive labels for `assert_equal` within tests.

- **Managing Nightly Builds and Packages**: Nightly build updates notification included information on how to update with `modular update`, linked the [changelog](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md) and the diffs between the releases. Some members also shared issues and solutions on how to handle "Error opening archive" when updating, with `modular clean` and reinstalling as common remedies.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.github.com/en/get-started/getting-started-with-git/setting-your-username-in-git">Setting your username in Git - GitHub Docs</a>: no description found</li><li><a href="https://github.com/modularml/mojo/discussions/2234)">Fix List slicing and String slicing to match Python when step is negative · modularml/mojo · Discussion #2234</a>: I want to fix issues #1944, #2046, and #2142. Those are List but the same problem is there for String. I actually have a fix done here with the interesting commit here. The implementation just has ...</li><li><a href="https://github.com/modularml/mojo/compare/1a8f912..1bce16d">Comparing 1a8f912..1bce16d · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/modularml/mojo/blob/738901dec1058612d8f01fd13e13a3e09103944f/stdlib/test/lit.cfg.py#L57">mojo/stdlib/test/lit.cfg.py at 738901dec1058612d8f01fd13e13a3e09103944f · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://docs.python.org/3/library/pathlib.html#pathlib.Path.cwd">pathlib — Object-oriented filesystem paths</a>: Source code: Lib/pathlib.py This module offers classes representing filesystem paths with semantics appropriate for different operating systems. Path classes are divided between pure paths, which p...</li><li><a href="https://github.com/modularml/mojo/compare/1a8f912..1bce1">Comparing 1a8f912..1bce1 · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/microsoft/vscode/issues/83096">Automatic --signoff via settings · Issue #83096 · microsoft/vscode</a>: It seems that presently, you can use the -s or --signoff command on your git commits by opening up the git commit dropdown and selecting signoff. This was implemented as a &quot;fix&quot; for #7010 in...</li><li><a href="https://github.com/rd4com/mojo_branch/tree/nightly">GitHub - rd4com/mojo_branch at nightly</a>: The Mojo Programming Language. Contribute to rd4com/mojo_branch development by creating an account on GitHub.</li><li><a href="https://github.com/modularml/mojo/pull/2215/checks?check_run_id=23522364066">[stdlib] Add `reversed` by helehex · Pull Request #2215 · modularml/mojo</a>: Adds an initial implementation of reversed(), for getting a reversed iterator of a range or list. This isn&amp;#39;t meant to be final, since iterators are not fully worked out yet, but it works for n...</li><li><a href="https://github.com/modularml/mojo/blob/nightly/docs/changelog.md">mojo/docs/changelog.md at nightly · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



**Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1225850270425157752)** (80 messages🔥🔥): 

- **WikiText Dataset Access Clarified**: Stephen Merity, the original author of the WikiText dataset, has [rehosted the data](https://state.smerity.com/smerity/state/01HRTB51QZMG59MDAX2ME1TCHR) on Cloudflare R2, which is considered the new main access point. The rehosted data, still under the Creative Commons license, includes larger datasets than the Penn Treebank and maintains the original case, punctuation, and numbers.

- **GateLoop Perplexity Puzzle**: There's a discussion about perplexity scores reported by the author of the GateLoop Transformer. While the author claimed good scores, lucidrains was unable to replicate them, raising some suspicions about the results.

- **Hugging Face Dataset Autoconversion Dilemma**: Chat members express frustration over Hugging Face's automatic conversion of datasets to parquet, which can be circumvented by using Git LFS for hosting. An example where this has been a problem is the format confusion with `.raw` files.

- **Search for Reproducible Data Formats**: The conversation was active around the need for reproducible and consistent data formats, with efforts made to mirror the original WikiText data on Cloudflare R2 as well for the sake of experiment reproducibility.

- **OpenAI Model Documentation Becomes Ephemeral**: Members shared their experiences finding information about OpenAI's models; several links to documentation about the models were taken down, leading to reliance on an [archived page](https://archive.ph/n5xMq) to understand the specifics regarding GPT 3.5 and other series, highlighting the challenges in tracking model evolution and changes.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://state.smerity.com/smerity/state/01HRTB51QZMG59MDAX2ME1TCHR">Smerity.com: The WikiText Long Term Dependency Language Modeling Dataset (2016)</a>: no description found</li><li><a href="https://huggingface.co/datasets/segyges/wikitext-103/tree/main">segyges/wikitext-103 at main</a>: no description found</li><li><a href="https://huggingface.co/datasets/wikitext/tree/main/wikitext-103-raw-v1">wikitext at main</a>: no description found</li><li><a href="https://github.com/lucidrains/gateloop-transformer">GitHub - lucidrains/gateloop-transformer: Implementation of GateLoop Transformer in Pytorch and Jax</a>: Implementation of GateLoop Transformer in Pytorch and Jax - lucidrains/gateloop-transformer</li><li><a href="https://github.com/tobiaskatsch/GatedLinearRNN">GitHub - tobiaskatsch/GatedLinearRNN</a>: Contribute to tobiaskatsch/GatedLinearRNN development by creating an account on GitHub.</li><li><a href="https://archive.ph/n5xMq">Model index for researchers - OpenAI API</a>: no description found
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1225769151029575701)** (313 messages🔥🔥): 

- **Understanding Schedule-Free Optimizer**: The Schedule-Free optimizer keeps a simple running average of weights, not an exponential moving average. The 1/t learning rate from one component is just another way of calculating the mean of all values, as demonstrated by the formula (1 * (1/2) * (2/3) * (3/4) * ... * (1-1/t)) equating to 1/t.

- **The Debate on Schedule-Free's Efficacy**: Results on Schedule-Free optimizer's performance are mixed, showing benefits in low-step runs but not significantly aiding in larger-step regimes. The optimizer estimates optimal learning rates, which might vary with the number of update steps.

- **Mixing Methods in Optimizers**: There is a discussion on whether increasing batch size over time could be an alternative or complement to learning rate schedules, with batch size doubling suggested as analogous to halving the learning rate.

- **New Approach to Language Model Search Strategies**: A study proposes a method to teach language models to search by using a stream of search (SoS) language, shown to boost search accuracy by 25% over models trained on predicting single next steps.

- **Emergent Abilities Linked to Long-Tail Data**: The ability of models for zero-shot tasks is being explored, with suggestions that emergent abilities in language models may be a function of the exposure to long-tail data during training.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/kyo_takano/status/1777273932120526969">Tweet from Kyo (@kyo_takano)</a>: Some data points where ScheduleFree outperforms Adam/SGD: - LM/GPT (@eric_alcaide) https://twitter.com/eric_alcaide/status/1776571679524683950 - CIFAR10/ResNet18 (@Sree_Harsha_N) https://twitter.com/S...</li><li><a href="https://x.com/aaron_defazio/status/1776320004465582331?s=46">Tweet from Aaron Defazio (@aaron_defazio)</a>: Schedule-Free Learning https://github.com/facebookresearch/schedule_free We have now open sourced the algorithm behind my series of mysterious plots. Each plot was either Schedule-free SGD or Adam, no...</li><li><a href="https://x.com/arankomatsuzaki/status/1777143382554313004?s=46&t=OICM4zGqs0OOATmLPoNFyw">Tweet from Aran Komatsuzaki (@arankomatsuzaki)</a>: No “Zero-Shot” Without Exponential Data: Pretraining Concept Frequency Determines Multimodal Model Performance  repo: https://github.com/bethgelab/frequency_determines_performance hf: https://huggingf...</li><li><a href="https://arxiv.org/abs/2403.15796">Understanding Emergent Abilities of Language Models from the Loss Perspective</a>: Recent studies have put into question the belief that emergent abilities in language models are exclusive to large models. This skepticism arises from two observations: 1) smaller models can also exhi...</li><li><a href="https://arxiv.org/abs/2110.00641">Batch size-invariance for policy optimization</a>: We say an algorithm is batch size-invariant if changes to the batch size can largely be compensated for by changes to other hyperparameters. Stochastic gradient descent is well-known to have this prop...</li><li><a href="https://arxiv.org/abs/2404.03683">Stream of Search (SoS): Learning to Search in Language</a>: Language models are rarely shown fruitful mistakes while training. They then struggle to look beyond the next token, suffering from a snowballing of errors and struggling to predict the consequence of...</li><li><a href="https://arxiv.org/abs/2402.05120">More Agents Is All You Need</a>: We find that, simply via a sampling-and-voting method, the performance of large language models (LLMs) scales with the number of agents instantiated. Also, this method is orthogonal to existing compli...</li><li><a href="https://arxiv.org/abs/1708.07120">Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates</a>: In this paper, we describe a phenomenon, which we named &#34;super-convergence&#34;, where neural networks can be trained an order of magnitude faster than with standard training methods. The existenc...</li><li><a href="https://arxiv.org/abs/2211.08411">Large Language Models Struggle to Learn Long-Tail Knowledge</a>: The Internet contains a wealth of knowledge -- from the birthdays of historical figures to tutorials on how to code -- all of which may be learned by language models. However, while certain pieces of ...</li><li><a href="https://openreview.net/forum?id=FpKgG31Z_i9">Learning Rate Grafting: Transferability of Optimizer Tuning</a>: In the empirical science of training large neural networks, the learning rate schedule is a notoriously challenging-to-tune hyperparameter, which can depend on all other properties (architecture...</li><li><a href="https://arxiv.org/abs/2404.03648">AutoWebGLM: Bootstrap And Reinforce A Large Language Model-based Web Navigating Agent</a>: Large language models (LLMs) have fueled many intelligent agent tasks, such as web navigation -- but most existing agents perform far from satisfying in real-world webpages due to three factors: (1) t...</li><li><a href="https://www.torchstudio.ai/getstarted/">Get Started</a>: &lt;a href=&quot;#install-torchstudio&quot;&gt;Install&lt;/a&gt; TorchStudio, &lt;a href=&quot;#load-and-analyze-the-mnist-dataset&quot;&gt;load&lt;/a&gt; a dataset, &lt;a href=&quot;#build-and-train-...</li><li><a href="https://github.com/facebookresearch/schedule_free">GitHub - facebookresearch/schedule_free: Schedule-Free Optimization in PyTorch</a>: Schedule-Free Optimization in PyTorch. Contribute to facebookresearch/schedule_free development by creating an account on GitHub.</li><li><a href="https://github.com/drukpa1455/fractal-gnn.git">GitHub - drukpa1455/fractal-gnn: fractal graph neural network exploration</a>: fractal graph neural network exploration. Contribute to drukpa1455/fractal-gnn development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1225837125447057478)** (1 messages): 

- **NSF Reviewer Highlights GitHub Stars**: An NSF reviewer noted the low number of GitHub stars for the **nnsight** project as a point of concern. The team emphasizes the importance of starring the repo, particularly for users who generally interact with the project via pip installs, and requests support [here](https://github.com/ndif-team/nnsight).

**Link mentioned**: <a href="https://github.com/ndif-team/nnsight">GitHub - ndif-team/nnsight: The nnsight package enables interpreting and manipulating the internals of deep learned models.</a>: The nnsight package enables interpreting and manipulating the internals of deep learned models. - ndif-team/nnsight

  

---


**Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1225740850974429244)** (83 messages🔥🔥): 

- **GPU Utilization Mystery Solved**: A member's evaluation time reduced dramatically from 20 minutes to 3 by running `batch size=auto`, indicating they were *underutilizing their GPU previously*.
- **Confusion Over BigBench Task Recognition**: Some users faced issues with `bigbench` not being recognized as a task; it was suggested to use `lm_eval —tasks list` to find the *correct bigbench variant(s)*.
- **Technical Hiccups with CLI Command**: There were errors reported involving the `—predict_only` CLI command; members discussed potential causes, including *version conflicts* or *improper use of the feature*.
- **Using Logit Bias for MCQA Tasks**: Dialogue about leveraging `logit_bias` in one-token MCQA tasks ensued, with the discovery that OpenAI’s implementation doesn't affect the returned logits, only the text, leading to exploration of using `greedy_until` instead.
- **Temperature Settings Affect on Output Quality**: A member questioned why different temperature settings in custom samplers didn't affect the output, leading to a technical dive into proper `gen_kwargs` settings and revealing the need for setting `do_sample=True` to avoid *greedy generation default behavior*.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/1pDByKcCu3vQzy58iz8uSmUm806LQtG8v#scrollTo=mTSKBJlVjaB-">Google Colaboratory</a>: no description found</li><li><a href="https://github.com/vllm-project/vllm/blob/b4543c8f6bf67a7f1a0d6d0fd6cf5697c7eeaabb/vllm/model_executor/layers/sampler.py#L161">vllm/vllm/model_executor/layers/sampler.py at b4543c8f6bf67a7f1a0d6d0fd6cf5697c7eeaabb · vllm-project/vllm</a>: A high-throughput and memory-efficient inference and serving engine for LLMs - vllm-project/vllm</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/e9a405431989fe30fe3c54a54ddc2c494a6a9e16/lm_eval/models/vllm_causallms.py#L480).">lm-evaluation-harness/lm_eval/models/vllm_causallms.py at e9a405431989fe30fe3c54a54ddc2c494a6a9e16 · EleutherAI/lm-evaluation-harness</a>: A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness</li><li><a href="https://wandb.ai/menhguin/lm-eval-harness-integration/reports/Weave-chain_of_thought_eval_results-24-04-08-02-23-54---Vmlldzo3NDQ5OTk0?accessToken=50arizwokl2js3if6g8y8wkse66pig35u5ijizuflou0aplud5dpx87drr4l4m78">Weave: chain_of_thought_eval_results (24/04/08 02:23:54)</a>: Publish your model insights with interactive plots for performance metrics, predictions, and hyperparameters. Made by Nguyen Nhat Minh using W&amp;B</li><li><a href="https://wandb.ai/menhguin/lm-eval-harness-integration/reports/Weave-chain_of_thought_eval_results-24-04-08-02-25-08---Vmlldzo3NDUwMDAy?accessToken=9831cfodvgpzdpvdwihwfmhh3grdytmvqj4sro1nth71jh6nunvw734eb1zp9dfp">Weave: chain_of_thought_eval_results (24/04/08 02:25:08)</a>: Publish your model insights with interactive plots for performance metrics, predictions, and hyperparameters. Made by Nguyen Nhat Minh using W&amp;B</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/e9a405431989fe30fe3c54">GitHub - EleutherAI/lm-evaluation-harness at e9a405431989fe30fe3c54a54ddc2c494a6a9e16</a>: A framework for few-shot evaluation of language models. - GitHub - EleutherAI/lm-evaluation-harness at e9a405431989fe30fe3c54a54ddc2c494a6a9e16
</li>
</ul>

</div>
  

---



**OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1225739422461595729)** (220 messages🔥🔥): 

- **Sentiment Analysis for Recordings Inquiry**: One member is exploring sentiment analysis options for text, phone, and video meeting recordings and is seeking recommendations for SaaS to utilize.
- **Anticipation for GPT-5**: There's a conversation about the anticipation for GPT-5, with users discussing various AI models that might be suitable for programming tasks, like Claude 3 Opus and Gemini 1.5 Pro.
- **Community Support and Kindness**: A member's supportive attitude is highlighted, offering personal assistance to others with their AI-related queries, and the significance of being nice while providing help is debated.
- **Questioning AI Training Data Sources**: A member raises concerns about whether OpenAI used YouTube data for Sora training and whether this could conflict with YouTube's terms of service.
- **Seeking Image Generation APIs**: Inquiry about alternative AI APIs for image generation besides DALL-E is met with a mention of an unspecified alternative, and members discussing the availability and potential of other models for image generation tasks.

**Link mentioned**: <a href="https://tenor.com/view/wow-really-gif-25055968">Wow Really GIF - Wow Really - Discover &amp; Share GIFs</a>: Click to view the GIF

  

---


**OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1225740006942048316)** (72 messages🔥🔥): 

- **GPT-4 Translation Capabilities vs DeepL**: One user mentioned that **ChatGPT-4's translation** does not perform as well as **DeepL**, particularly in capturing basic contexts and choosing contextually appropriate words rather than direct translations.
- **Developing Sensitive Character Backstories with ChatGPT**: Writers discussed strategies for working within **ChatGPT's content policy** when developing characters with traumatic backgrounds, suggesting subtler approaches to describing character experiences.
- **Custom GPTs Require a Subscription**: Users clarified that all **GPTs**, including variants used within custom applications, require a **Plus (or above) plan** subscription to be accessible.
- **Multilanguage Prompts in Custom GPT Starters**: There was a discussion on the efficacy of using prompts in various languages for conversation starters in custom GPTs, though potential filtering issues on platforms like Discord were noted.
- **Performance Variability in GPT Models**: Coders compared the performances of different GPT models, with one citing that **Opus** and a preview version of **GPT-4** performed impressively in code generation tasks. However, there was also a mention of **GPT-4** potentially performing less optimally with larger contexts when compared to other models.
  

---


**OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1225769191324258304)** (57 messages🔥🔥): 

- **AI's Simulated Consciousness Draws Interest**: Curiosity peaks as a member attempts to simulate human consciousness within GPT by asking it to develop pseudocode for human chemical hormones and equate them to its programming mechanisms. This exploration is deemed *adorable* and *interesting* even as GPT struggles to maintain consistency in representing consciousness.
  
- **Pseudocode for Neurochemical Functions**: A member with a background in psychology and computer science ponders the potential to code aspects of human consciousness, suggesting the possibility of translating neurochemical functions into code and minimizing the "specialness" of human ego in the process.
  
- **The Interplay of Biology, Spirituality, and AI**: A discussion about whether consciousness is purely biological or has spiritual components leads to varied opinions. One member suggests adopting a default assumption that entities may possess some form of consciousness and to err on the side of not causing detectable misery, regardless of the origin of consciousness.

- **Techniques for Extracting Information from AI**: Users debate on the distinctions between a model’s system message and an operating system-like set of instructions for tools, relating these concepts to the transparency and modular nature of ChatGPT’s system prompts.

- **Finalizing a Text-Based AI Game**: Suggestions arise for how to manage and display game progress information when building a completely AI-powered text-based game, mentioning the use of JSON for data structuring and the challenges of presenting cropped information to users.
  

---


**OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1225769191324258304)** (57 messages🔥🔥): 

- **Exploring AI's Take on Consciousness**: Members sparked a debate over emulating human consciousness and emotions by breaking them down into chemicals represented in code. While the GPT struggled to stay in character, it acknowledged that consciousness may emerge from neurochemical interplays.

- **GPT and Depictions of Consciousness**: Conversations turned adorable when discussing consciousness with GPT. Despite initial skepticism, participants found insights into the interplay of neurochemicals and self-preservation as potential factors in the emergence of consciousness, making for an intriguing AI-human interaction.

- **Dall-E as a Dissertation Designer**: Users discussed using Dall-E to create a dissertation front page, debating the effectiveness of different tools, with some suggesting combinations of GPT with LaTeX or Python-Matplotlib as superior approaches.

- **Enhancing GPT Prompts for Fun and Games**: A user seeking to create an AI text-based game considered ways to refine prompts to conceal code information from user-displayed text, with JSON being suggested as one tool for this task.

- **Understanding ChatGPT’s System Prompts and Tools**: An in-depth exchange occurred regarding how system prompts and tools affect GPT's responses, drawing a comparison between "the difference between an LLM and an LLM OS" and illustrating the modularity of the ChatGPT environment.
  

---



**OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1225750620330528830)** (4 messages): 

- **Multimodality Activated for Claude 3**: The modality of all **Claude 3 models** has been changed to `multimodal`, supporting image input. Developers relying on this property need to update their code to accommodate these changes.

- **Claude 3 Messages Enhanced**: `messages.name` has been integrated into the upstream **Claude 3 messages**. For more details and implications for your projects, read the discussion [here](https://discord.com/channels/1091220969173028894/1223444233394847864).

- **Prompt Template Improvement for DBRX**: The prompt template for **DBRX** has been updated to reduce repetitiveness based on user feedback. More can be found on the topic [here](https://discord.com/channels/1091220969173028894/1222619272208187402).

- **Fresh Models & Features Unveiled**: Two new models released are **DBRX Nitro**, excelling at code generation and general knowledge tasks, and **Command R+**, a large model from Cohere outperforming GPT-4 Turbo and Claude 3 Sonnet on various benchmarks. Additional updates include UI enhancements, new analytics, more model parameters like `logit_bias`, and support for `seed` and `response_format`s for several models. Model details: [DBRX Nitro](https://openrouter.ai/models/databricks/dbrx-instruct:nitro) and [Command R+](https://openrouter.ai/models/cohere/command-r-plus).

- **Cohere's Formatting Fix & Policy Update**: An issue with system prompt formatting for **Cohere requests** has been resolved. Cohere models will not be moderated by OpenRouter moving forward, but will adhere to Cohere's [acceptable use policy](https://docs.cohere.com/docs/c4ai-acceptable-use-policy).

- **Community Feedback Sought**: A new poll has been posted for community feedback. Participate in the poll [here](https://discord.com/channels/1091220969173028894/1094454198688546826/1226585086997041203).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openrouter.ai/models/databricks/dbrx-instruct:nitro">DBRX 132B Instruct by databricks | OpenRouter</a>: DBRX is a new open source large language model developed by Databricks. At 132B, it outperforms existing open source LLMs like Llama 2 70B and Mixtral-8x7B on standard industry benchmarks for language...</li><li><a href="https://openrouter.ai/models/cohere/command-r-plus">Command R+ by cohere | OpenRouter</a>: Command R+ is a new, 104B-parameter LLM from Cohere. It&#x27;s useful for roleplay, general consumer usecases, and Retrieval Augmented Generation (RAG).  It offers multilingual support for ten key lan...
</li>
</ul>

</div>
  

---


**OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1226271687163641856)** (1 messages): 

- **AI Enters the Classic Game Arena**: Take on ChatGPT in a simple yet challenging game of [Rock, Paper, Scissors](https://rock.blust.ai). Pit your strategic skills against the bot to see if you can outwit it.

**Link mentioned**: <a href="https://rock.blust.ai">Rock, Paper, Scissors Game by Blust.AI</a>: Play Rock, Paper, Scissors against ChatGPT. It’s easy to play and a fun way to see if you can outsmart an AI.

  

---


**OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1225757847682547777)** (322 messages🔥🔥): 

- **API Frontends for OpenRouter**: Users discussed various frontends for the OpenRouter, including [LibreChat](https://librechat.ai/) which has a ChatGPT-like UI and offers authentication and plugins, [SillyTavern](https://sillytavern.com/) which is good for chat/roleplay, and [Jan.ai](https://jan.ai/docs/remote-inference/router) which is similar to LM Studio but open source and supports local API servers.
- **Favorites Models for Roleplay and Coding**: Command-R+ was lauded as good for coding and even translating Turkish, with some users equating its usefulness to that of various Claude models. Meanwhile, others expressed concern about over-censorship with some models and OpenAI's implementation of concepts like 'unsafe' content.
- **Discussions About Model Performance**: Users noted that Sonnet performs better than Opus in coding, especially with German and chemical tasks, and some found that Claude 3 performed better in extracting data from PDFs than Gemini Pro 1.5. There was also skepticism expressed about the capabilities of Gemini Pro 1.5, with some users not finding it useful.
- **Exploring Model Features**: The community engaged in discussions about model features such as JSON mode support and logit bias, with some users providing tips and workarounds for issues with certain models and requests for additional feature filtering in model selection tools.
- **Concerns Over Model Rankings and Usefulness**: There was dialogue regarding the effectiveness of basing model rankings on usage statistics, with suggestions for alternative metrics like users' spending or model retention to assess a model's utility more accurately.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://librechat.ai/">LibreChat</a>: Enhanced ChatGPT Clone, featuring OpenAI, Azure, Mistral, Anthropic, Google, Ollama, DALL-E-3 models and more. An Open-source, versatile Web UI, with seamless self-hosting and active developments.</li><li><a href="https://docs.cohere.com/docs/c4ai-acceptable-use-policy">C4AI Acceptable Use Policy</a>: no description found</li><li><a href="https://jan.ai/docs/remote-inference/router">Jan - OpenRouter</a>: A step-by-step guide on how to integrate Jan with OpenRouter.</li><li><a href="https://docs.together.ai/docs/json-mode">JSON Mode</a>: no description found</li><li><a href="https://openrouter.ai/models/perplexity/sonar-medium-chat?tab=parameters">Sonar 8x7B by perplexity | OpenRouter</a>: Sonar is Perplexity&#x27;s latest model family. It surpasses their earlier models in cost-efficiency, speed, and performance.  The version of this model with Internet access is [Sonar 8x7B Online](/mo...</li><li><a href="https://openrouter.ai/">OpenRouter</a>: A router for LLMs and other AI models</li><li><a href="https://openrouter.ai/docs#parameters">OpenRouter</a>: Build model-agnostic AI apps
</li>
</ul>

</div>
  

---



**LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1225849052378431672)** (8 messages🔥): 

- **Introducing AutoRAG for Performance Optimization**: Marker-Inc-Korea's AutoRAG 🔥 is a new tool that optimizes RAG pipelines by automatically fine-tuning hyperparameters using a given evaluation dataset. This optimization is announced via a tweet with links to more details: [AutoRAG's tweet](https://t.co/ZndrM36n61).

- **RAG Transforms Sales Outreach**: Described in a recent webinar, a new sales use case for RAG replaces hard-coded templates with prompt templates that leverage an LLM to craft personalized sales emails. Find further information in the shared links: [Sales Use Case tweet](https://t.co/kV7MGJ6PqS).

- **Scaffold Full-Stack RAG/Agent Apps with Ease**: `create-llama` is a just-released stand-alone repository that simplifies the process of starting full-stack RAG/agent applications, inspired by `create-react-app`, allowing deployment of a Javascript-based full-stack chatbot in one command. The announcement with relevant links is accessible here: [create-llama tweet](https://t.co/YOEZUQt7Lr).

- **Complex QA with Multi-Document Agents**: Andy Singal's overview on @llama_index multi-document agents demonstrates their ability to navigate complex QA over many documents, aiming to extend the functionality beyond simple, single-document queries. The presentation tweet can be seen here: [Multi-Document Agents tweet](https://t.co/3yKuv2qDDf).

- **Best Full-Stack RAG Tutorial**: ClusteredBytes created a tutorial and GitHub repository showcasing the sophisticated architecture required to build a full-stack RAG application capable of streaming intermediate results to a UI. Details can be found in this tweet: [Full-Stack RAG App Tutorial tweet](https://t.co/6w23wQ35u3).
  

---


**LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1225739171067461642)** (254 messages🔥🔥): 

- **Document References and Page Numbers in Multi-Document Queries**: For someone looking to get document references along with page numbers when querying, it was suggested to ensure metadata includes such details before indexing. Accessing these in the source nodes' metadata is crucial to getting desired references post-query.

- **Azure OpenAI Context Seek Issues**: Discussions highlighted problems with Azure's OpenAI service failing to identify context contained within nodes. Despite the relevant information present, the model would apologize for not finding the context, suggesting possible issues with settings or inconsistencies compared to Mistral AI's functioning.

- **Product Recognition and Classification with LLMs**: A chat about classifying products from various stores with different names but being essentially the same item explored the use of large language models (LLMs) for identification. Several strategies, including the use of model merging tactics and embedding models, were discussed as potential solutions for managing extensive databases of products.

- **Speeding up Embedding Generation**: Optimizing embedding generation involved switching from processing embeddings one by one to using batching methods like `get_text_embedding_batch`. This adjustment speeds up the processes, especially for large files, by aligning the text chunks with the nodes, embedding in batches, and then reassigning these batch embeddings back to individual nodes.

- **RAG and OpenSource Model Challenges**: There were concerns expressed about the ReAct agent not utilizing tools as expected when paired with open-source models like "llama2", "mistral", and "gemma". It was clarified that open-source models often struggle with agentic tasks, and better descriptions in routers could help with accurate routing.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.llamaindex.ai/en/stable/">LlamaIndex - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/loading/simpledirectoryreader/?gad_source=1&gclid=Cj0KCQjw5cOwBhCiARIsAJ5njubnGYY3NjP8r3E42fQb_lLj3hG8QwN7xhrXol1Qz71aqWshIPDGkk0aAlnREALw_wcB">SimpleDirectoryReader - LlamaIndex</a>: no description found</li><li><a href="https://console.aws.amazon.com/ec2/.">no title found</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/agent/react_agent/">ReAct Agent - A Simple Intro with Calculator Tools - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/models/llms/#open-source-llms">Using LLMs - LlamaIndex</a>: no description found</li><li><a href="https://www.llamaindex.ai/blog/launching-the-first-genai-native-document-parsing-platform">Launching the first GenAI-native document parsing platform — LlamaIndex, Data Framework for LLM Applications</a>: LlamaIndex is a simple, flexible data framework for connecting custom data sources to large language models (LLMs).</li><li><a href="https://www.llamaindex.ai/blog/introducing-llamacloud-and-llamaparse-af8cedf9006b">Introducing LlamaCloud and LlamaParse — LlamaIndex, Data Framework for LLM Applications</a>: LlamaIndex is a simple, flexible data framework for connecting custom data sources to large language models (LLMs).</li><li><a href="https://github.com/run-llama/llama_index/blob/9163067027ea8222e9fe5bffff9a2fac26b57686/llama-index-integrations/readers/llama-index-readers-file/llama_index/readers/file/docs/base.py#L32">llama_index/llama-index-integrations/readers/llama-index-readers-file/llama_index/readers/file/docs/base.py at 9163067027ea8222e9fe5bffff9a2fac26b57686 · run-llama/llama_index</a>: LlamaIndex is a data framework for your LLM applications - run-llama/llama_index</li><li><a href="https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/indices/utils.py#L114">llama_index/llama-index-core/llama_index/core/indices/utils.py at main · run-llama/llama_index</a>: LlamaIndex is a data framework for your LLM applications - run-llama/llama_index</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/multi_modal/azure_openai_multi_modal/?h=azureopenaimultimodal">Multi-Modal LLM using Azure OpenAI GPT-4V model for image reasoning - LlamaIndex</a>: no description found</li><li><a href="https://github.com/run-llama/llama_index/blob/main/llama-index-legacy/llama_index/legacy/readers/file/image_reader.py#L71">llama_index/llama-index-legacy/llama_index/legacy/readers/file/image_reader.py at main · run-llama/llama_index</a>: LlamaIndex is a data framework for your LLM applications - run-llama/llama_index</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/retrievers/auto_merging_retriever">Auto Merging Retriever - LlamaIndex</a>: no description found</li><li><a href="https://github.com/run-llama/llama_index/blob/9163067027e">GitHub - run-llama/llama_index at 9163067027ea8222e9fe5bffff9a2fac26b57686</a>: LlamaIndex is a data framework for your LLM applications - GitHub - run-llama/llama_index at 9163067027ea8222e9fe5bffff9a2fac26b57686</li><li><a href="https://docs.llamaindex.ai/en/latest/understanding/storing/storing#inserting-documents-or-nodes>))">Storing - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/understanding/indexing/indexing#using-vector-store-index>))">Indexing & Embedding - LlamaIndex</a>: no description found
</li>
</ul>

</div>
  

---


**LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1226807615942823986)** (1 messages): 

- **Challenges with Top Agent Tool Selection**: A member discussed an issue where the **top agent** mistakenly chose the incorrect tool from the five available in the index. They mentioned they are optimizing the retrieval logic and will share their findings.

  

---



**OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1225753717253345350)** (170 messages🔥🔥): 

- **Mistral’s Computing Requirements**: Mention of **"Mistral 7B Instruct v0.2"** indicating it performs well but requires significant computing power, suggesting at least 16GB of RAM and some GPU capabilities.
- **Calls for Examples on Vision Models and Documentation**: Inquiry about successful local vision models for os mode and a request for examples based on base open interpreter/cookbook, pointing to gaps in the current documentation.
- **Interest in Event Recording**: Discussion on recording Discord voice chats for events using the OpenInterpreter Python library, with suggestions to use broadcasting software like OBS (Open Broadcaster Software) for recording, and considering Craig Bot for audio.
- **Language Barrier in Technical Assistance**: Members trying to provide technical help despite language barriers, using examples of adding voice (TTS) to the Open Interpreter with mixed results, possibly resolved after several attempts.
- **Inquiry about Open Interpreter Capabilities**: Questions regarding the feasibility of certain tasks with the Open Interpreter, such as downloading and converting articles into markdown files, and an indication of efforts to improve core repository reliability over the next few months.

**Link mentioned**: <a href="https://discord.gg/xXtcB9hq?event=1225831217832919051">Join the Open Interpreter Discord Server!</a>: A new way to use computers | 8147 members

  

---


**OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1225764004044210196)** (71 messages🔥🔥): 

- **Trouble Connecting Client and Server**: Members reported difficulties with the client not connecting to the server on their configurations. It was suggested that an incompatible environment, potentially due to incompatible Python versions, could be causing missing/conflicting **TTS packages**. A proposed solution involved creating a **Conda environment with Python <=3.10** and re-cloning the repository.

- **Push Button Switch Issues**: Individuals constructing the **01 hardware** noted that the built-in button of the M5 was functioning, but the external push-button switch was not. Subsequent discussions include mentions of reviewing `client.ino` code for missing GPIO definitions for an external button.

- **Python Version Compatibility Challenges**: Several users cited that **Python 3.11.4** did not work for their setup. It was confirmed that downgrading to **Python 3.10** resolved the issue where the system appeared to not "hear" spoken commands, indicating a **version support limitation**.

- **Local Models As Cost-effective Alternative**: Conversations around cost concerns with using **GPT-4** led to discussions around **local models** like Hermes 7B and Haiku as effective, cost-efficient alternatives. Members indicated these models being slightly worse on some tasks but offering advantages like lower cost and privacy.

- **Windows Setup Struggles**: One member struggled with Windows installation, following several instructions including using **chocolatey**, **virtualenv**, and setting the **OPENAI_API_KEY**. They identified a potential solution by ensuring Python 3.9 was used in a virtual environment and reaching out for further help with setting the API key properly.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.openinterpreter.com">no title found</a>: no description found</li><li><a href="https://docs.openinterpreter.com/language-models/hosted-models/anthropic#supported-models">no title found</a>: no description found</li><li><a href="https://01.openinterpreter.com/services/language-model#hosted-models">Language Model - 01</a>: no description found</li><li><a href="https://github.com/OpenInterpreter/01/issues/226">Access Issue with Linux `dmesg` · Issue #226 · OpenInterpreter/01</a>: Describe the bug The software\source\server\utils\kernel.py function get_kernel_messages is trying to access a file that doesn&#39;t exist on some Linux distros (all?) like Arch, Debian, and Fedora. U...</li><li><a href="https://github.com/OpenInterpreter/01.git">GitHub - OpenInterpreter/01: The open-source language model computer</a>: The open-source language model computer. Contribute to OpenInterpreter/01 development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



**LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1225744947827839017)** (190 messages🔥🔥): 

- **Seeking Pull Request Assistance**: A user requested help for a [Pull Request](https://github.com/langchain-ai/langchain/pull/19751) with a build failing on GitHub due to a "module not found" error involving "openapi-pydantic." Despite the module being listed in dependencies, the issue persisted.

- **Discord Summarization Queries**: Users discussed how to incorporate a YouTube URL using the `YouTubeAudioLoader` documentation from LangChain, with specific questions about substituting OpenAI Whisper Parser with Ollama, and whether `Whisper` from OpenAI could be a solution.

- **LangChain Coding Issues**: Members sought coding assistance, such as using `register_tool` in LangChain that's causing import errors, setting up LangGraph and addressing an `InvalidUpdateError`, fix imports from `langchain.messages`, and tackling embedding dimension lengths for specific use cases.

- **Fleshing Out Agents and Chains**: In conversations involving LangChain scripts, users requested examples and guidance for creating custom tools, registering reactive agents, implementing prompt templates, and generating output keys for given inputs. They also deliberated on the deprecated status of `ZeroShotAgent`.

- **AI Fine-Tuning Interest**: A user expressed interest in learning about training and fine-tuning LLMs without a GPU, with another recommending the use of Google Colab and frameworks like ludwig.ai for this purpose.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://python.langchain.com/docs/integrations/document_loaders/microsoft_excel/.">Microsoft Excel | 🦜️🔗 LangChain</a>: The UnstructuredExcelLoader is used to load Microsoft Excel files.</li><li><a href="https://python.langchain.com/docs/expression_language/how_to/routing/#:~:text=Routing%20allows%20you%20to%20create%20non-deterministic%20chains%20where,runnables%20from%20a%20RunnableLambda%20%28recommended%29%20Using%20a%20RunnableBranch.">Route logic based on input | 🦜️🔗 LangChain</a>: dynamically-route-logic-based-on-input}</li><li><a href="https://python.langchain.com/docs/expression_language/how_to/routing/#:">Route logic based on input | 🦜️🔗 LangChain</a>: dynamically-route-logic-based-on-input}</li><li><a href="https://serper.dev>)">no title found</a>: no description found</li><li><a href="https://js.langchain.com/docs/integrations/document_loaders/web_loaders/serpapi#usage>)">SerpAPI Loader | 🦜️🔗 Langchain</a>: This guide shows how to use SerpAPI with LangChain to load web search results.</li><li><a href="https://js.langchain.com/docs/use_cases/tool_use/quickstart#create-a-tool>)).">Quickstart | 🦜️🔗 Langchain</a>: In this guide, we will go over the basic ways to create Chains and Agents that call Tools. Tools can be just about anything — APIs, functions, databases, etc. Tools allow us to extend the capabilities...</li><li><a href="https://python.langchain.com/docs/modules/memory/types/entity_summary_memory#using-in-a-chain>).">Entity | 🦜️🔗 LangChain</a>: Entity memory remembers given facts about specific entities in a conversation. It extracts information on entities (using an LLM) and builds up its knowledge about that entity over time (also using an...</li><li><a href="https://python.langchain.com/docs/integrations/tools/lemonai#load-api-keys-and-access-tokens>),">Lemon Agent | 🦜️🔗 LangChain</a>: Lemon Agent helps you</li><li><a href="https://js.langchain.com/docs/use_cases/graph/prompting#set-environment-variables>)).">Prompting strategies | 🦜️🔗 Langchain</a>: In this guide we’ll go over prompting strategies to improve graph</li><li><a href="https://python.langchain.com/docs/langgraph#add_edge>).">🦜🕸️LangGraph | 🦜️🔗 LangChain</a>: Downloads</li><li><a href="https://js.langchain.com/docs/langgraph#interaction-with-lcel>).">LangGraph | 🦜️🔗 Langchain</a>: ⚡ Building language agents as graphs ⚡</li><li><a href="https://python.langchain.com/docs/modules/agents/quick_start/">Quickstart | 🦜️🔗 LangChain</a>: quickstart}</li><li><a href="https://python.langchain.com/docs/use_cases/summarization/">Summarization | 🦜️🔗 LangChain</a>: Open In Colab</li><li><a href="https://docs.google.com/forms/d/e/1FAIpQLSf_A93vTlBH428XoGyBdeR9cDHAIo6TRnQOmaK0LziY7-9C2Q/viewform">Pesquisa de mercado</a>: Gostaríamos de convidá-lo a participar de nossa pesquisa de mercado. Sua participação é fundamental para ajudar nossa empresa a entender melhor o mercado e aprimorar nosso MVP. Nossa pesquisa é projet...</li><li><a href="https://js.langchain.com/docs/get_started/quickstart#llm-chain>)">Quickstart | 🦜️🔗 Langchain</a>: In this quickstart we&#x27;ll show you how to:</li><li><a href="https://python.langchain.com/docs/use_cases/question_answering/local_retrieval_qa#qa>).">Using local models | 🦜️🔗 LangChain</a>: The popularity of projects like</li><li><a href="https://github.com/langchain-ai/langchain/pull/19751.">Build software better, together</a>: GitHub is where people build software. More than 100 million people use GitHub to discover, fork, and contribute to over 420 million projects.</li><li><a href="https://github.com/langchain-ai/langchain/issues/3638>).">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/13446>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://python.langchain.com/docs/get_started/quickstart#retrieval-chain>).">Quickstart | 🦜️🔗 LangChain</a>: In this quickstart we&#x27;ll show you how to:</li><li><a href="https://js.langchain.com/docs/langgraph#addedge>)">LangGraph | 🦜️🔗 Langchain</a>: ⚡ Building language agents as graphs ⚡</li><li><a href="https://github.com/langchain-ai/langchain/pull/19979">community: extend Predibase integration to support fine-tuned LLM adapters by alexsherstinsky · Pull Request #19979 · langchain-ai/langchain</a>: PR title: &quot;package: description&quot;  Where &quot;package&quot; is whichever of langchain, community, core, experimental, etc. is being modified. Use &quot;docs: ...&quot; for purely docs change...</li><li><a href="https://github.com/anujmehta/langchain/blob/request-body-reference/libs/community/pyproject.toml#L22">langchain/libs/community/pyproject.toml at request-body-reference · anujmehta/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to anujmehta/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/anujmehta/langchain/blob/request-body-reference/libs/community/pyproject.toml#L244.">langchain/libs/community/pyproject.toml at request-body-reference · anujmehta/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to anujmehta/langchain development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1225822204848111676)** (45 messages🔥): 

- **Semantic Chunking for Node.js**: A TypeScript implementation of **Semantic Chunking**, now available for those using Node.js environments, enabling the effective processing of large text corpora into semantic chunks. The technique combines sentences for context, utilizes OpenAI's service for sentence embeddings, and groups semantically similar sentences. Check out the [shared gist](https://gist.github.com/tsensei/3b6589662271874b5055d79473932aae) for details.
  
- **New AI Image Generation Models in Artful**: **Artful AI** has been updated with new models **Dalle Creative, Anime Dream, & Epic Realism**, designed for transforming ideas into stunning images. This AI image generator has also undergone bug fixes to enhance user experience. Take a look at the new features on [Google Play Store](https://play.google.com/store/apps/details?id=com.projecthit.artful&referrer=ph-aai).

- **AISploit for Exploiting LLM AI Solutions**: Introducing **AISploit**, a tiny package aimed at aiding red teams and penetration testers in exploiting large language model AI solutions. The tool can be an essential asset for security professionals working with AI. Find it on [GitHub](https://github.com/hupe1980/aisploit).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://js.langchain.com/docs/modules/agents/agent_types/react#run-agent>).">ReAct | 🦜️🔗 Langchain</a>: This walkthrough showcases using an agent to implement the ReAct logic.</li><li><a href="https://python.langchain.com/docs/modules/agents/agent_types/json_agent#run-agent>).">JSON Chat Agent | 🦜️🔗 LangChain</a>: Some language models are particularly good at writing JSON. This agent</li><li><a href="https://play.google.com/store/apps/details?id=com.projecthit.artful&referrer=ph-aai">Artful - AI Art Generator - Apps on Google Play</a>: no description found</li><li><a href="https://python.langchain.com/docs/use_cases/query_analysis/techniques/structuring#load-example-document>)">Structuring | 🦜️🔗 LangChain</a>: One of the most important steps in retrieval is turning a text input</li><li><a href="https://smith.langchain.com/>).">LangSmith</a>: no description found</li><li><a href="https://python.langchain.com/docs/langsmith/walkthrough#log-runs-to-langsmith>)">LangSmith Walkthrough | 🦜️🔗 LangChain</a>: Open In Colab</li><li><a href="https://js.langchain.com/docs/guides/langsmith_evaluation#log-runs-to-langsmith>).">LangSmith Walkthrough | 🦜️🔗 Langchain</a>: LangChain makes it easy to prototype LLM applications and Agents. However, delivering LLM applications to production can be deceptively difficult. You will have to iterate on your prompts, chains, and...</li><li><a href="https://github.com/hupe1980/aisploit">GitHub - hupe1980/aisploit: 🤖🛡️🔍🔒🔑 Tiny package designed to support red teams and penetration testers in exploiting large language model AI solutions.</a>: 🤖🛡️🔍🔒🔑 Tiny package designed to support red teams and penetration testers in exploiting large language model AI solutions. - hupe1980/aisploit</li><li><a href="https://gist.github.com/tsensei/3b6589662271874b5055d79473932aae">This TypeScript snippet processes a large corpus of text to output semantic chunks by tokenizing into sentences, combining them for context, generating sentence embeddings with OpenAI&#39;s service, calculating cosine similarities to identify semantic shifts, and finally grouping sentences into semantically cohesive chunks based on these shifts.</a>: This TypeScript snippet processes a large corpus of text to output semantic chunks by tokenizing into sentences, combining them for context, generating sentence embeddings with OpenAI&amp;#39;s servic...</li><li><a href="https://smith.langchain.com/>),">LangSmith</a>: no description found</li><li><a href="https://js.langchain.com/docs/use_cases/question_answering/chat_history#langsmith>).">Add chat history | 🦜️🔗 Langchain</a>: In many Q&amp;A applications we want to allow the user to have a</li><li><a href="https://python.langchain.com/docs/use_cases/question_answering/chat_history#langsmith>).">Add chat history | 🦜️🔗 LangChain</a>: In many Q&amp;A applications we want to allow the user to have a</li><li><a href="https://github.com/langchain-ai/langchain/issues/1071>)).">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/2371>)).">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/15692>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://js.langchain.com/docs/use_cases/chatbots/tool_usage#conversational-responses>).">Tool usage | 🦜️🔗 Langchain</a>: This section will cover how to create conversational agents: chatbots that can interact with other systems and APIs using tools.
</li>
</ul>

</div>
  

---



**LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1225816625916153957)** (157 messages🔥🔥): 

- **Apple's AI Efforts Dubbed Lackluster**: Discussion centers on Apple's perceived failure to deliver on AI promises with critiques about **MPS** (Metal Performance Shaders) and **torch compile** being suboptimal on their platforms. They also discussed recently merged fixes for **MPS in the PyTorch** nightly branch, with members sharing varied experiences with the implementation and functionality of **torch.compile**.
  
- **Challenging Legal Terrain in AI-Rewritten Text**: Members engaged in an exploration of the legality surrounding the use of AI to rewrite copyrighted texts. There was consensus that mere paraphrasing or name changes do not eliminate copyright infringement, and skirting copyright may require significant legal shifts or new practices in AI training data use.

- **New AI Music Generation Battle Heats Up**: A conversation about the advances in AI-generated music touched on companies like **Suno** and its yet-to-be-named competitors from Nvidia, with enthusiasm for the new technology tempered by predictions of legal challenges from the music industry. Insight was offered on the limited "real world" advances in TTS despite the leap in AI music capabilities.

- **Surge in AI Ethics, Careers, and Models**: Discussions highlighted the dynamics in AI-related careers influenced by AI enhancements, with a focus on freelancing. Moreover, Stability AI released a **zero SNR model, CosXL**, under a non-commercial research community license agreement, prompting debates over the practical and theoretical aspects of their approach, including the use of **EDM schedules** and **offset noise** in model training.

- **Data Scarcity and Open Projects in AI**: Users shared experiences and requests for assistance with specific AI projects, such as generating images of a personal school using **Stable Diffusion**, while others commented on the availability of rare datasets like CT abdominal images. Also mentioned were contributions to the open-source community, like **PKU-YuanGroup's Open-Sora-Plan**, as initiatives to replicate landmark AI functionalities like OpenAI's T2V model.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://fxtwitter.com/Legit4K/status/1777059367788982389">Tweet from ʟᴇɢɪᴛ (@legit_rumors)</a>: here&#39;s an exclusive new udio AI music generation 🫡  source: anon. anon is always the source.</li><li><a href="https://fxtwitter.com/lifeafterAi_/status/1776930684642443400">Tweet from moonbi⭕ (@lifeafterAi_)</a>: This post was deleted. Seems like suno’s competitor is nivdia 👀 btw this sounds like 2pac 🔥🔥 @apples_jimmy good call again 🐐</li><li><a href="https://huggingface.co/stabilityai/cosxl">stabilityai/cosxl · Hugging Face</a>: no description found</li><li><a href="https://colab.research.google.com/github/borisdayma/dalle-mini/blob/main/tools/inference/inference_pipeline.ipynb#scrollTo=uzjAM2GBYpZX">Google Colaboratory</a>: no description found</li><li><a href="https://sonauto.ai/">sonauto-platform</a>: no description found</li><li><a href="https://bloomberry.com/i-analyzed-5m-freelancing-jobs-to-see-what-jobs-are-being-replaced-by-ai/">The jobs being replaced by AI - an analysis of 5M freelancing jobs - bloomberry</a>: There’s no question that AI will impact jobs. But which jobs are more likely to be replaced by&hellip;</li><li><a href="https://github.com/PKU-YuanGroup/Open-Sora-Plan/blob/main/docs/Report-v1.0.0.md">Open-Sora-Plan/docs/Report-v1.0.0.md at main · PKU-YuanGroup/Open-Sora-Plan</a>: This project aim to reproduce Sora (Open AI T2V model), but we only have limited resource. We deeply wish the all open source community can contribute to this project. - PKU-YuanGroup/Open-Sora-Plan</li><li><a href="https://github.com/facebookresearch/schedule_free">GitHub - facebookresearch/schedule_free: Schedule-Free Optimization in PyTorch</a>: Schedule-Free Optimization in PyTorch. Contribute to facebookresearch/schedule_free development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1225799018878206020)** (23 messages🔥): 

- **Dynamic Compute Allocation in Transformers**: A [new paper](https://arxiv.org/abs/2404.02258) details how transformers can allocate computational resources (FLOPs) dynamically across input sequences using a top-$k$ routing mechanism, optimizing the self-attention and MLP computations within a predetermined compute budget.
- **Efficiency Innovations Round-Up**: The r/singularity subreddit contains [a list of recent papers and approaches](https://www.reddit.com/r/singularity/comments/1bwu2x5/efficiency_alert_some_papers_and_approaches_in/) aimed at reducing pretraining, fine-tuning, and inference costs for various AI applications.
- **DARE Method for Language Model Capability Assimilation**: Research [introduced DARE](https://arxiv.org/abs/2311.03099), a tool to merge and sparsify delta parameters across fine-tuned language models, potentially demonstrating that significant pruning of these parameters is possible without loss of capabilities.
- **BrushNet for Enhanced AI Inpainting**: An announcement of [BrushNet](https://www.youtube.com/watch?v=X89IQop_0dM), a new method for inpainting that incorporates object detection, was shared along with a tutorial video explaining how it generates higher-quality results.
- **Exploring Latent Diffusion in Text Generation**: A discussion was ignited by [a NeurIPS paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/b2a2bd5d5051ff6af52e1ef60aefd255-Paper-Conference.pdf) on "Latent Diffusion for Language Generation," suggesting innovative directions for text generation, highlighting a potential move towards techniques commonly used in image model generation.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2404.02258">Mixture-of-Depths: Dynamically allocating compute in transformer-based language models</a>: Transformer-based language models spread FLOPs uniformly across input sequences. In this work we demonstrate that transformers can instead learn to dynamically allocate FLOPs (or compute) to specific ...</li><li><a href="https://arxiv.org/abs/2311.03099">Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch</a>: In this paper, we unveil that Language Models (LMs) can acquire new capabilities by assimilating parameters from homologous models without retraining or GPUs. We first introduce DARE to set most delta...</li><li><a href="https://tenor.com/view/rick-and-morty-that-just-sounds-like-slavery-with-extra-steps-slave-rick-morty-gif-18016642">Rick And Morty That Just Sounds Like Slavery With Extra Steps GIF - Rick And Morty That Just Sounds Like Slavery With Extra Steps Slave - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://aaronlou.com/blog/2024/discrete-diffusion/">Language Modeling by Estimating the Ratios of the Data Distribution | Aaron Lou</a>: no description found</li><li><a href="https://www.reddit.com/r/singularity/comments/1bwu2x5/efficiency_alert_some_papers_and_approaches_in/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=X89IQop_0dM">BrushNet - The Best InPainting Method Yet? FREE Local Install!</a>: Inpainting with the latest BrushNet models for AI generation using stable diffusion is lots of fun! Get great results quickly thanks to both random and segme...
</li>
</ul>

</div>
  

---



**Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1225794935928782958)** (88 messages🔥🔥): 

- **Victor Taelin's Prompt Challenge Proven Wrong**: A $10k prize was awarded after it was proven that GPT models *could* solve the A::B problem, challenging the initial claim that GPT architectures lack the capability for certain problem-solving tasks, particularly regarding long-term reasoning. The prize-winning solution achieved a near 100% success rate, sparking discussions about the potential of GPT models and their existing architectures. [Victor Taelin's admission can be found here](https://x.com/victortaelin/status/1777049193489572064?s=46&t=6FDPaNxZcbSsELal6Sv7Ug).

- **CS336: Language Modeling from Scratch**: Stanford offers a new course CS336, conducted by Professor Percy Liang, focusing on the fundamentals of language modeling, including Transformers, LLMs, and optimizers. There's high interest in the materials and a request has been made to release lecture recordings.

- **Groq's Ambitious AI Hardware Goals**: Groq's founder, a high school and undergrad dropout, details their journey from starting the TPU project at Google to expecting Groq to have the largest inference capacity by next year, surpassing all providers combined. Groq is now at 75k developers and boasts lower inference costs and faster hardware than NVIDIA's H200.

- **LangChain's New Memory Service**: LangChain releases an alpha memory service for chatbots to automatically extract and enrich user conversations, potentially improving personalization and user experience. [Documentation and quick start resources are available](https://langchain-ai.github.io/long-term-memory/).

- **New Techniques in Transformer Architecture**: Discussions around the effectiveness of ensemble methods with LLMs lead to the acknowledgment that using multiple agents and voting methods can enhance performance, particularly in challenging tasks. The method involves scoring based on the most common outputs amongst similar responses.

- **Attention Mechanism Illuminated**: A new video by 3Blue1Brown demystifies the attention mechanism within transformers and LLMs. The content is applauded for its clear explanation and considered to be a potential resource for educational discussions.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/soumithchintala/status/1776323475101081816">Tweet from Soumith Chintala (@soumithchintala)</a>: @fchollet @JeffDean @GoogleDeepMind This is honestly a baffling response. You cant be saying that benchmarking FP32 vs TF32 (just the dtype) is a &#34;compiler optimization&#34;. Honestly, I&#39;m los...</li><li><a href="https://x.com/clattner_llvm/status/1776468511130591286?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Chris Lattner (@clattner_llvm)</a>: I love both PT and XLA/TF and am happy you are working through your differences. Speaking as one on the outside, we all want ai to win and all the systems to succeed. If one wants to resort to benchma...</li><li><a href="https://x.com/victortaelin/status/1777049193489572064?s=46&t=Yfq9g0ScYi47w3NFZRPVLw">Tweet from Taelin (@VictorTaelin)</a>: I *WAS* WRONG - $10K CLAIMED!  ## The Claim  Two days ago, I confidently claimed that &#34;GPTs will NEVER solve the A::B problem&#34;. I believed that: 1. GPTs can&#39;t truly learn new problems, out...</li><li><a href="https://x.com/victortaelin/status/1776225351678468429">Tweet from Taelin (@VictorTaelin)</a>: dear diary  today I taught 1k people how to use interaction combinators  but at what cost  ↘️ Quoting Taelin (@VictorTaelin)   A simple puzzle GPTs will NEVER solve:  As a good programmer, I like isol...</li><li><a href="https://x.com/victortaelin/status/1777049193489572064?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Taelin (@VictorTaelin)</a>: I *WAS* WRONG - $10K CLAIMED!  ## The Claim  Two days ago, I confidently claimed that &#34;GPTs will NEVER solve the A::B problem&#34;. I believed that: 1. GPTs can&#39;t truly learn new problems, out...</li><li><a href="https://x.com/gdb/status/1777127364822024283?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Greg Brockman (@gdb)</a>: Making a music video with Sora: &#34;this is how the song has always &#39;looked&#39;, it&#39;s just that now i get to show you.&#34;  https://www.youtube.com/watch?v=Se93p3gk_14</li><li><a href="https://x.com/nielsrogge/status/1777050848675201065">Tweet from Niels Rogge (@NielsRogge)</a>: Watched a super interesting talk on Ring Attention, probably the magic behind Gemini&#39;s 1 million context window  You organize your devices (GPU/TPU) in a ring, each computing a part of the final a...</li><li><a href="https://web.stanford.edu/class/cs25/">CS25: Tranformers United!</a>: Disussing the latest breakthroughs with Transformers in diverse domains</li><li><a href="https://arxiv.org/abs/2402.05120">More Agents Is All You Need</a>: We find that, simply via a sampling-and-voting method, the performance of large language models (LLMs) scales with the number of agents instantiated. Also, this method is orthogonal to existing compli...</li><li><a href="https://www.brightwave.io/">Brightwave</a>: no description found</li><li><a href="https://openrag.notion.site/Open-RAG-c41b2a4dcdea4527a7c1cd998e763595">Notion – The all-in-one workspace for your notes, tasks, wikis, and databases.</a>: A new tool that blends your everyday work apps into one. It's the all-in-one workspace for you and your team</li><li><a href="https://share.snipd.com/snip/8eb39371-e1c4-4140-9ad1-5981efe3c21b">Innovating Data Centers with Moore's Law | 48sec snip from ChinaTalk</a>: 48sec snip from A Gut Check on Intel and Nvidia with Asianometry, Fabricated Knowledge, and SemiAnalysis | ChinaTalk</li><li><a href="https://arxiv.org/abs/2312.10997">Retrieval-Augmented Generation for Large Language Models: A Survey</a>: Large Language Models (LLMs) showcase impressive capabilities but encounter challenges like hallucination, outdated knowledge, and non-transparent, untraceable reasoning processes. Retrieval-Augmented...</li><li><a href="https://newvick.com/rag-evolution/">Evolution of RAG: Addressing the common problems of a simple RAG system</a>: RAG is not all you need. This post will cover some of the common problems that are encountered in a simple RAG system, and potential solutions for them.</li><li><a href="https://x.com/llama_index/status/1774832426000515100">Tweet from LlamaIndex 🦙 (@llama_index)</a>: This is an excellent tutorial by @mesudarshan showing you how to build advanced PDF RAG with LlamaParse and purely local models for embedding, LLMs, and reranking (@GroqInc and FastEmbed by @qdrant_en...</li><li><a href="https://x.com/AndrewYNg/status/1773006786058219889">Tweet from Andrew Ng (@AndrewYNg)</a>: New JavaScript short course: Build a full-stack web application that uses RAG in JavaScript RAG Web Apps with LlamaIndex, taught by @seldo, VP of Developer Relations at @llama_index and npm co-founder...</li><li><a href="https://x.com/llama_index/status/1767687784712814619">Tweet from LlamaIndex 🦙 (@llama_index)</a>: Use LlamaIndex and @MathpixApp to parse and index complex mathematics to LaTeX, and answer questions about scientific papers!  Check out this detailed notebook in which MathPix takes you through ➡️ Pa...</li><li><a href="https://x.com/llama_index/status/1761553473219551301">Tweet from LlamaIndex 🦙 (@llama_index)</a>: Let’s walk through RAG pain points and solutions! 🧑‍🏫🎬  We’re excited to feature @wenqi_glantz for a video walkthrough video of her popular “12 RAG Pain Points and Solutions” blog post, which is th...</li><li><a href="https://www.modular.com/blog/how-to-be-confident-in-your-performance-benchmarking">Modular: How to Be Confident in Your Performance Benchmarking</a>: We are building a next-generation AI developer platform for the world. Check out our latest post: How to Be Confident in Your Performance Benchmarking</li><li><a href="https://partiful.com/e/VJPFposDqQg2eCqHuL38">RSVP to Realtime Voice AI and Multimodal Hackathon | Partiful</a>: Hi fellow lovely hackers,  The AI Engineer Foundation (Your Friendly Open Source Nonprofit Neighbor - website: aie.foundation) is hosting a Real Time Interactive/Conversational Multimodal AI hackathon...</li><li><a href="https://www.daily.co/blog/how-to-talk-to-an-llm-with-your-voice/">How to talk to an LLM (with your voice)</a>: Code for building real-time AI WebRTC applications</li><li><a href="https://langchain-ai.github.io/long-term-memory/">LangMem - LangMem</a>: no description found</li><li><a href="https://long-term-memory-shared-for-f208c46599174c09b9b79-vz4y4ooboq-uc.a.run.app'">no title found</a>: no description found</li><li><a href="https://share.1password.com/s#DPhaOn02m2OD18hu1Ig45a5fPbZxGNKd63VVc37lQtA">I’m sharing an item with you using 1Password</a>: A password manager, digital vault, form filler and secure digital wallet. 1Password remembers all your passwords for you to help keep account information safe.</li><li><a href="https://www.youtube.com/watch">YouTube</a>: no description found</li><li><a href="https://hlfshell.ai/posts/llms-and-robotics-papers-2023/#self-consistency">State of the art in LLMs &#43; Robotics - 2023</a>: tldr I write about some of the more interesting works that shaped my understanding of applying LLMs for AI agents and robotic applications.  Introduction  What is this LLMs as a fad - a caveat Are LLM...</li><li><a href="https://github.com/stanford-cs336/spring2024-assignment1-basics/blob/master/cs336_spring2024_assignment1_basics.pdf">spring2024-assignment1-basics/cs336_spring2024_assignment1_basics.pdf at master · stanford-cs336/spring2024-assignment1-basics</a>: Contribute to stanford-cs336/spring2024-assignment1-basics development by creating an account on GitHub.</li><li><a href="https://www.youtube.com/watch?v=yiewqC6qNM8">Eugene Cheah - From idea to LLM (RWKV / Recursal)</a>: Talk from the Open-Source Generative AI Workshop at Cornell Tech. Website: https://github.com/PicoCreatorSlides: https://drive.google.com/file/d/1-lfITA0j_9-...</li><li><a href="https://youtu.be/eMlx5fFNoYc">Visualizing Attention, a Transformer&#39;s Heart | Chapter 6, Deep Learning</a>: Demystifying attention, the key mechanism inside transformers and LLMs.Instead of sponsored ad reads, these lessons are funded directly by viewers: https://3...</li><li><a href="https://github.com/go-go-golems/bobatea/blob/main/pkg/chat/README.md">bobatea/pkg/chat/README.md at main · go-go-golems/bobatea</a>: Custom bubbletea bubbles. Contribute to go-go-golems/bobatea development by creating an account on GitHub.</li><li><a href="https://github.com/go-go-golems/bobatea/blob/main/cmd/chat/backend.go">bobatea/cmd/chat/backend.go at main · go-go-golems/bobatea</a>: Custom bubbletea bubbles. Contribute to go-go-golems/bobatea development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1225854717213544498)** (8 messages🔥): 

- **Announcing Latent Space University's Inaugural Course**: The _Latent Space University_ is kicking off its first course on AI Engineering with a free introductory session at 1pm PT. Sign up and details are available at [Maven Learning, Inc.](https://maven.com/p/245c45).

- **Clash of Events**: There's a light-hearted acknowledgment of scheduling overlap as a course introduction coincides with the **Latent Space Discord** event.

- **Expand Your AI Expertise in Three Weeks**: A new course promises a comprehensive journey through AI modalities, covering **OpenAI API, Retrieval Augmented Generation, Code Generation, Image Generation**, and **Speech-to-Text** features over a span of three weeks. A discount is available with the code "lightning" as stated in the [Course Overview](https://maven.com/noah-hein/ai-engineering-intro).

- **Weekend Podcast Teaser**: A new podcast episode has been released for the weekend, with the announcement shared via a [Twitter link](https://twitter.com/swyx/status/1776687540520767544).

- **Latent Space Podcast Weekend Special**: The podcast covers various topics including **AI UX, The World's Fair**, and the latest in AI technology and leadership. Dive into the discussion of the AI Engineering trends and more in the _Weekend Special_ episode summarized on [Latent Space](https://www.latent.space/p/weekend-special-5-chats).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://maven.com/noah-hein/ai-engineering-intro">Level Up From Software Engineer to AI Engineer by Shawn &quot;Swyx&quot; Wang and Noah Hein on Maven</a>: From Dabbling To Complete Competency: Learn how to build real-world working AI products</li><li><a href="https://maven.com/p/245c45">Code a custom ChatGPT</a>: This is the foundation of AI products. If you want to be an AI engineer these are MUST KNOW topics and API&#x27;s.  Everything from ChatGPT to robust AI powered summarization and classification use th...</li><li><a href="https://www.latent.space/p/weekend-special-5-chats">Latent Space Chats: NLW (Four Wars, GPT5), Josh Albrecht/Ali Rohde (TNAI), Dylan Patel/Semianalysis (Groq), Milind Naphade (Nvidia GTC), Personal AI (ft. Harrison Chase — LangFriend/LangMem)</a>: 5 recent Latent Space appearances for you to get your LS fix on all AI Engineering topics imaginable.
</li>
</ul>

</div>
  

---


**Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1225897703175622791)** (57 messages🔥🔥): 

- **Seamless Knowledge Capture**: A discussion highlighted the use of chat applications, like Slack, as knowledge bases, suggesting it's a "power move" for capturing and synthesizing useful information artifacts from human-to-human interactions.
- **Optimizing Text Work**: A member emphasized the attractiveness and feasibility of reducing cognitive load through structured documents, rather than relying on Slack, which was pointed out to be a "terrible system of record" despite being where "the action happens" in many companies.
- **Tools and Integrations Galore**: The chat surfaced several resources for augmenting personal knowledge bases and workspaces with AI, including [Obsidian-Copilot](https://github.com/logancyang/obsidian-copilot) and [fabric](https://github.com/danielmiessler/fabric) for AI-augmented human performance, along with a suggestion to use [Obsidian's CLI tools](https://github.com/Yakitrak/obsidian-cli).
- **Building Better Bridges**: Ongoing exploration of integrations for AI tools like ChatGPT into personal knowledge systems, such as Obsidian, was discussed, focusing on existing plugins and the potential for creating new ones.
- **Collaborative Contribution**: The chat concluded with acknowledgments for sharing useful ideas and resources, indicating a collective appreciation for the insights and suggestions offered during the discussion.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/logancyang/obsidian-copilot">GitHub - logancyang/obsidian-copilot: A ChatGPT Copilot in Obsidian</a>: A ChatGPT Copilot in Obsidian. Contribute to logancyang/obsidian-copilot development by creating an account on GitHub.</li><li><a href="https://github.com/Yakitrak/obsidian-cli">GitHub - Yakitrak/obsidian-cli: Interact with Obsidian in the terminal. Open, search, create, update, move and delete notes!</a>: Interact with Obsidian in the terminal. Open, search, create, update, move and delete notes! - Yakitrak/obsidian-cli</li><li><a href="https://docs.google.com/spreadsheets/d/1q5rwO4wleMTLXr1z58c2UC03QsDsGwbJY1v4UG7eEOs/edit#gid=0">AI In Action: Weekly Jam Sessions</a>: 2024  Topic,Date,Facilitator,Resources,@dropdown,@ UI/UX patterns for GenAI,1/26/2024,nuvic,&lt;a href=&quot;https://maggieappleton.com/squish-structure&quot;&gt;https://maggieappleton.com/squish-stru...</li><li><a href="https://github.com/danielmiessler/fabric">GitHub - danielmiessler/fabric: fabric is an open-source framework for augmenting humans using AI. It provides a modular framework for solving specific problems using a crowdsourced set of AI prompts that can be used anywhere.</a>: fabric is an open-source framework for augmenting humans using AI. It provides a modular framework for solving specific problems using a crowdsourced set of AI prompts that can be used anywhere. - ...
</li>
</ul>

</div>
  

---



**OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1225778122549624832)** (53 messages🔥): 

- **AWQ Models Operational on Hugging Face Inference**: A member mentioned successfully running **awq models** on Hugging Face's inference service.
- **GitHub Oddity Noted**: There was a report of **GitHub search** redirecting to a specific page with a detailed description and image, suspected to be an auto-redirection issue.

- **Enthusiasm for New Qwen Model**: Dialog centered on the latest **qwen model**, with specific interest in its **32B** variant. Further discussions suggested that **Yi 34B** and **Command R** were also models of interest for comparison on the same **fine-tune dataset**. 
- **Training on Context Length with Ring Attention**: A member brought attention to a GitHub repository named [EasyContext by jzhang38](https://github.com/jzhang38/EasyContext), which outlines a memory optimization and training recipe for extrapolating LM context length to 1 million tokens using ring attention on 8xA100 GPUs. Accompanying this was a [Twitter thread](https://twitter.com/PY_Z001/status/1776176932687892796) by the author discussing the decline in training throughput as context size increases.
- **Schedule-Free Optimization on GitHub**: Introduction to the **Schedule-Free Optimization in PyTorch** repository was posted, presumably to highlight a tool for improving optimization processes.
- **Coding Challenges with ORPO Structure**: A member detailed their struggles with the ORPO structure when trying to implement a new prompt template and encountering issues with **micro_batch_size**. Caseus_ confirmed ongoing issues with **ORPO and batch sizes** in a subsequent response.
- **Generative AI Summit Consideration and Reflection**: A member weighed the pros and cons of attending a Generative AI summit in Paris, deliberating on the value of attendance for networking. They later confirmed their attendance and mentioned finishing second at the summit event but not networking much.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openai.com/blog/introducing-improvements-to-the-fine-tuning-api-and-expanding-our-custom-models-program">Introducing improvements to the fine-tuning API and expanding our custom models program</a>: We’re adding new features to help developers have more control over fine-tuning and announcing new ways to build custom models with OpenAI.</li><li><a href="https://www.raisesummit.com/#bl-59ec20d5-ce0f-4f84-971d-543b5c7efa9b>">R.AI.SE Summit</a>: RAISE is a gathering of 1,500+ global leaders dedicated to explore the transformative power of Generative AI on businesses and society. The conference will take place in Paris on April 8th, 2024</li><li><a href="https://www.lepton.ai/">Build AI The Simple Way | Lepton AI</a>: Run AI applications efficiently, at scale, and in minutes with a cloud native platform.</li><li><a href="https://github.com/search">Build software better, together</a>: GitHub is where people build software. More than 100 million people use GitHub to discover, fork, and contribute to over 420 million projects.</li><li><a href="https://github.com/facebookresearch/schedule_free">GitHub - facebookresearch/schedule_free: Schedule-Free Optimization in PyTorch</a>: Schedule-Free Optimization in PyTorch. Contribute to facebookresearch/schedule_free development by creating an account on GitHub.</li><li><a href="https://github.com/jzhang38/EasyContext">GitHub - jzhang38/EasyContext: Memory optimization and training recipes to extrapolate language models&#39; context length to 1 million tokens, with minimal hardware.</a>: Memory optimization and training recipes to extrapolate language models&#39; context length to 1 million tokens, with minimal hardware. - jzhang38/EasyContext</li><li><a href="https://github.com/huggingface/transformers/pull/30005">Add JetMoE model by yikangshen · Pull Request #30005 · huggingface/transformers</a>: What does this PR do? Add support to JetMoE architecture by Yikang Shen and MyShell AI. JetMoE is a new sparsely activated architecture inspired by the ModuleFormer. Each JetMoE block consists of t...</li><li><a href="https://www.raisesummit.com/#bl-59ec20d5-ce0f-4f84-971d-543b5c7">R.AI.SE Summit</a>: RAISE is a gathering of 1,500+ global leaders dedicated to explore the transformative power of Generative AI on businesses and society. The conference will take place in Paris on April 8th, 2024
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1225740998353752117)** (19 messages🔥): 

- **Quantized DoRA Supported**: The `peft=0.10.0` now supports quantized DoRA, as indicated in [PEFT's latest release notes](https://github.com/huggingface/peft/releases/tag/v0.10.0). The change may warrant an update to **axolotl's** `requirements.txt` file.

- **Introducing Schedule-Free Learning**: Facebook Research has [open sourced Schedule-Free Learning](https://github.com/facebookresearch/schedule_free), a method that replaces momentum with averaging and interpolation, removing the need for traditional learning rate schedules.

- **New Optimizer Requires Code Changes**: Developers should note that the new Schedule-Free optimizer requires additional `optimizer.train()` and `optimizer.eval()` calls; this is highlighted in the optimizer's repository.

- **ScheduleFreeAdamW Parameters Tuning Tips**: For optimal performance with ScheduleFreeAdamW, a value of `0.0025` is recommended, and developers should consult the caveats section for guidance on additional tunable parameters.

- **Upstream Contributions to Hugging Face Transformers**: The support for adamw schedulefree has been [upstreamed to Hugging Face's transformers library](https://github.com/huggingface/transformers/pull/30079), which simplifies integration with deepspeed or PyTorch FSDP configurations.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/aaron_defazio/status/1776341914364641583?s=46&t=hIokEbug9Pr72tQFuXVULA">Tweet from Aaron Defazio (@aaron_defazio)</a>: @divideconcept It needs to be that tuned. See the notes in the caveats section for suggested ranges. The value 0.0025 seems to work pretty reliably for ScheduleFreeAdamW</li><li><a href="https://x.com/aaron_defazio/status/1776320004465582331?s=46&t=hIokEbug9Pr72tQFuXVULA">Tweet from Aaron Defazio (@aaron_defazio)</a>: Schedule-Free Learning https://github.com/facebookresearch/schedule_free We have now open sourced the algorithm behind my series of mysterious plots. Each plot was either Schedule-free SGD or Adam, no...</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/pull/1486">add support for adamw schedulefree by winglian · Pull Request #1486 · OpenAccess-AI-Collective/axolotl</a>: implements meta&#39;s https://github.com/facebookresearch/schedule_free for adamw https://twitter.com/aaron_defazio/status/1776320004465582331 optimizer: schedule_free_adamw lr_scheduler: constant</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/src/axolotl/utils/config/models/input/v0_4_1/__init__.py#L245>">axolotl/src/axolotl/utils/config/models/input/v0_4_1/__init__.py at main · OpenAccess-AI-Collective/axolotl</a>: Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.</li><li><a href="https://github.com/huggingface/transformers/pull/30079">schedulefree optimizers by winglian · Pull Request #30079 · huggingface/transformers</a>: What does this PR do? integrates meta&#39;s https://github.com/facebookresearch/schedule_free for adamw &amp; sgd https://twitter.com/aaron_defazio/status/1776320004465582331 Before submitting   This ...
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1225806966551613551)** (5 messages): 

- **Mistral Model Generation Error**: A user reported an error when generating with a *fine-tuned Mistral 7b model* using **fp16**. The error occurs after a few successful generations, with a traceback leading to `_queue.Empty`.

- **Clarification on Inference Method**: The user clarified that they were not using the built-in inference method but were instead utilizing **Huggingface's generate with streamer**.

- **Assumption of Accelerate Library Usage**: Another member suggested the user might be utilizing **Accelerate**, but the user denied this, confirming they were using plain Python.

- **Sharing of Troublesome Code**: The user facing the generation issue shared their code that employs **Huggingface's transformers**, python's **threading**, and a custom class `StopOnTokens` based on `StoppingCriteria` alongside **Gradio's ChatInterface** for deploying a chatbot application.
  

---


**OpenAccess AI Collective (axolotl) ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/)** (1 messages): 

faldore: <@&1166009801583628349> porn spam
  

---


**OpenAccess AI Collective (axolotl) ▷ #[docs](https://discord.com/channels/1104757954588196865/1167137552470392842/1225946216739770421)** (3 messages): 

- **LISA Parameters Not in Config**: A member noticed that the parameters for the **LISA implementation** are missing from `axolotl/docs/config.qmd`.
- **LISA Config Found Elsewhere**: The LISA parameters were later found and shared by another member, with a link to the [LISA configuration file](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/examples/llama-2/lisa.yml) on GitHub.
- **Unfreezing Layers Queries**: A question was raised about **handling optimizer states** after unfreezing new layers during the model training process, capturing the community's interest in practical implementation details.

**Link mentioned**: <a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/examples/llama-2/lisa.yml">axolotl/examples/llama-2/lisa.yml at main · OpenAccess-AI-Collective/axolotl</a>: Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.

  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-help-bot](https://discord.com/channels/1104757954588196865/1225300056442409040/1225804060087812166)** (46 messages🔥): 

- **Docker Image for LoRA Adapter Merge Error**: A user encountered a [pydantic validation error](https://errors.pydantic.dev/2.6/v/value_error) while trying to merge the LoRA adapter, which required either `flash_attention` or `sdp_attention` to be set to true when `sample_packing` is enabled.

- **Training with Raw Text on Mistral Model**: For training a Mistral model with raw text, a member shared a YAML [configuration example](https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=c12d621b-5026-4bc0-af60-66a9b40e9708) detailing model and tokenizer specifications, dataset paths, and training parameters.

- **Adapting Alpaca Instruction Set for Fine-Tuning**: When fine-tuning with an Alpaca instruction set and ChatML format, a user suggested converting the dataset to ShareGPT format and utilizing the conversation: chatml for configuration, which resolved the dataset mix concerns.

- **Micro Batch vs. Batch Size**: The difference between micro batch size and batch size was clarified, where micro batch size allows for efficient memory usage and simulates larger batch sizes without computational costs, while batch size updates the model's weights once per entire batch of data.

- **Config for Disabling Checkpoints and Evaluation Phases**: Users discussed how to modify configuration files to never save checkpoints by changing `saves_per_epoch` to `0` and inquiring about disabling evaluation phases altogether, suggesting setting the `evaluation_strategy` to `EvaluationStrategy.NO`.

- **Handling of Undefined Special Tokens in Configs**: It was clarified that when special tokens are not defined in a configuration file like `examples/mistral/qlora.yml`, default values based on the base model and tokenizer will be used unless they're specifically overridden.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://errors.pydantic.dev/2.6/v/value_error">Redirecting...</a>: no description found</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=f3156bb0-3cb9-4c34-b7d8-7cb4618a499d)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=8cc85b97-df87-499b-a134-50674538d2f4)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=3a24e145-a395-4639-b2a6-100b531e959b)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=39846e14-89a0-4353-a806-cc1e3136c78a)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=c12d621b-5026-4bc0-af60-66a9b40e9708)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=2cc617a2-4788-4f73-a29b-6d622e452e3b)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=c371c38b-42d0-4b01-b381-55fd5f1d093f)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=2dc474d5-a5e8-441b-bc59-17e2571f2781)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-phorm-bot](https://discord.com/channels/1104757954588196865/1225558824501510164/1225744640372899841)** (22 messages🔥): 

- **Clarifying "rope_theta"**: A member inquired about the meaning of `"rope_theta": 10000.0`, which refers to a parameter in the Rotary Positional Embedding technique used to introduce positional information to Transformer models.

- **FSDP Config for Mistral Revealed**: The Fully Sharded Data Parallel (FSDP) configuration for Mistral is specified in the `mixtral-qlora-fsdp.yml` file, indicating that the `MixtralSparseMoeBlock` class should be wrapped for sharding. The conversation included a [link to the configuration file](https://github.com/openaccess-ai-collective/axolotl/tree/main/examples/mistral/mixtral-qlora-fsdp.yml#L1L75).

- **LISA Layer Left Undefined**: A query about what constitutes a "lisa layer" emerged, but the term does not correspond to widely recognized concepts within the AI and machine learning communities as of the latest knowledge update.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=2150270f-2213-4881-b572-a8c9dab49c46)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=35cacb5b-24d0-43ce-8a22-8eb2ab861118)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=3246eab0-a12a-4f23-ac87-0cb50c2fccf2)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.</li><li><a href="https://github.com/openaccess-ai-collective/axolotl/tree/main/examples/mistral/mixtral-qlora-fsdp.yml#L1L75)">axolotl/examples/mistral/mixtral-qlora-fsdp.yml at main · OpenAccess-AI-Collective/axolotl</a>: Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=d540907f-286f-4152-8935-2370919b6441)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.
</li>
</ul>

</div>
  

---



**Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1225867580506374154)** (15 messages🔥): 

- **Podcast Potential with John Schulman**: The channel's host, Nathan, contemplated the exciting idea of having John Schulman on for an interview, acknowledging that it somehow slipped his mind before.
- **Anticipation for a 'Banger' Interview**: The suggestion of interviewing John Schulman was met with enthusiasm, with another member agreeing that it would indeed be a hit.
- **Exploring New AI Musical Horizons**: A link to a tweet was shared, hinting at a competitor to the Suno AI platform, which Nathan found to be "rly freaking good".
- **Opening the License Floodgates**: A tweet from @julien_c announced a notable licensing change, switching **text-generation-inference (TGI)** from a custom HFOIL license back to Apache 2, making the library completely open-source. Nathan commented on Hugging Face's transparency and the risks they took.
- **Contributor Influx Following Licensing Change**: The decision to open-source TGI led to a threefold increase in contributors, and despite initial low popularity, the project gained traction after the license modification. Nathan's ensuing messages appeared to express excitement over this development, using phrases like "rip," "NOW WE'RE ECLIPSING," and "LFG," suggesting positive momentum.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/robdadashi/status/1777317222496526663?s=46">Tweet from Robert Dadashi (@robdadashi)</a>: The training data was pretty much the same as v1.0, but we switched the RL algorithm to something new. I hope that we will be able to disclose more about this in the future :). 6/11</li><li><a href="https://fxtwitter.com/julien_c/status/1777328456709062848">Tweet from Julien Chaumond (@julien_c)</a>: We have decided to update text-generation-inference (TGI)&#39;s license.  We switch the license from HFOIL (our custom license) back to Apache 2, hence making the library fully open-source.  Read belo...</li><li><a href="https://fxtwitter.com/legit4k/status/1777059367788982389?s=46">Tweet from ʟᴇɢɪᴛ (@legit_rumors)</a>: here&#39;s an exclusive new udio AI music generation 🫡  source: anon. anon is always the source.
</li>
</ul>

</div>
  

---


**Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1226275316482441246)** (9 messages🔥): 

- **Someone's Been Targeted... Again!**: A member mentioned being targeted once more, without providing specifics or context.
- **Improvements on the Horizon?**: Another member responded that their experiences have improved recently, suggesting whatever was "bad" is less so now.
- **Advice or Just Meme-ing Around?**: A member recommended "Use Code Interpreter" in response to a previous comment, seemingly as a continuation of a joke.
- **All in Good Fun**: The initial requester clarified that the request was just a meme and thanked the group, indicating no actual need for advice.
- **Just Kidding! Employment Status Confirmed**: The suggestion of unemployment was clarified as a joke, and the member confirmed being employed and in good stead.
  

---


**Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1226374380981063690)** (55 messages🔥🔥): 

- **Debating the Risks and Inevitabilities of Open Model Weights**: Discussion centered around the societal impact of open foundation models, concerning whether there needs to be a safety threshold for their release. Diverse views were expressed about the practicality and feasibility of enforcing non-proliferation of AI technology, and whether there's an ethical responsibility to regulate its distribution.
  
- **Exploring AI's Power Dynamic**: Members conversed about the potential of language models to manipulate societal and democratic processes. The consensus seemed to focus on the inevitability of advances in AI, the ease of building language models, and their accessibility, leading to a resignation of sorts that tight regulation may not be practical.
  
- **The Genie Out of the Bottle**: A member offered an analogy to discuss the control of powerful AI, likening unrestricted AI to personal genies that could have unwanted societal effects. There was skepticism regarding the practical enforcement of usage restrictions, with a comparison to the challenges faced in nuclear non-proliferation.
  
- **Scale and Accessibility in Open AI Research**: The trend of increasing computational costs to train large models was noted, suggesting that this could outpace the capacity of commodity hardware and gate community/academic access. The current state of model inference being more cost-effective via APIs than running on personal hardware was highlighted.
  
- **Future Models: Generation vs. Verification**: Towards the end of the discussion, the concept of open models that focus on verification rather than generation was brought up. A curiosity was expressed about whether this approach could make models more accessible and bypass the need for large-scale models by shifting verifier knowledge to inference time.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/modeless/status/1776693329432088718>">Tweet from James Darpinian (@modeless)</a>: @TheXeophon Have you seen the theory that the typos are intentional to filter out the intelligent people</li><li><a href="https://open.substack.com/pub/aisnakeoil/p/on-the-societal-impact-of-open-foundation">On the Societal Impact of Open Foundation Models</a>: Adding precision to the debate on openness in AI</li><li><a href="https://www.youtube.com/watch?v=eMlx5fFNoYc">Visualizing Attention, a Transformer&#39;s Heart | Chapter 6, Deep Learning</a>: Demystifying attention, the key mechanism inside transformers and LLMs.Instead of sponsored ad reads, these lessons are funded directly by viewers: https://3...
</li>
</ul>

</div>
  

---


**Interconnects (Nathan Lambert) ▷ #[sp2024-history-of-open-alignment](https://discord.com/channels/1179127597926469703/1223784028428177510/1225849170326327359)** (31 messages🔥): 

- **Scouring for Top Fine Tunes**: Discussion around utilizing [lmsys](https://lmsys.deepai.org/) and the [alpacaeval leaderboard](https://alpacaeval.com/) as resources to discover effective fine-tuned models. These platforms are highlighted as good starting points for finding models that achieve state-of-the-art performance.
- **Acknowledgment of OpenWeights**: Within the context of finding models, DeepSeek's **OpenWeights** models were mentioned as a potential source.
- **Shift to Visual Aids**: A commitment was made to clarify and categorize models visually during an upcoming talk; this will include tactics like fading, highlighting, and enlarging specific models on screen for better understanding.
- **Live Document for a History Talk**: A link to a [Google Slides presentation](https://docs.google.com/presentation/d/1quMyI4BAx4rvcDfk8jjv063bmHg4RxZd9mhQloXpMn0/edit?usp=sharing) was shared, which appears to be a work-in-progress for an upcoming lecture on “Aligning open language models”.
- **Guide to Open Models**: Xeophon offered a detailed exposition on Salesforce's CodeGen series, including the release timeline, datasets used, licensing, plus a [comprehensive spreadsheet](https://docs.google.com/spreadsheets/d/1gc6yse74XCwBx028HV_cvdxwXkmXejVjkO-Mz2uwE0k/edit#gid=0) compiling various models and their attributes. This resource serves to save time for those researching open models.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.google.com/presentation/d/1quMyI4BAx4rvcDfk8jjv063bmHg4RxZd9mhQloXpMn0/edit?usp=sharing">[18 April 2024] Aligning open language models</a>: Aligning open language models Nathan Lambert Stanford CS25: Transformers United V4 1</li><li><a href="https://docs.google.com/spreadsheets/d/1gc6yse74XCwBx028HV_cvdxwXkmXejVjkO-Mz2uwE0k/edit#gid=0>">Directory of Generative AI</a>: Pretrained LLMs  Name,Date,Parameters (Active),Parameters (Total),Organizaton,Organization Type,Author Location,Language,Commercial Use,Model Accessibility,Code Accessibility,Data Accessibility,Major ...
</li>
</ul>

</div>
  

---



**CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1225980876677189633)** (15 messages🔥): 

- **Fast Tokenizers: A Need for Speed**: A member discussed the slow tokenization process of the *c4-en* dataset using Huggingface's fast tokenizer, inquiring about options to speed it up, such as increasing threads. Another suggested looking into machines with more threads.
  
- **AMD's Open Source Leap**: AMD announced they will open source their Micro Engine Scheduler (MES) firmware, along with documentation and a GitHub tracker for Radeon GPUs. This news aligns with broader efforts by AMD to make their GPU technology more open source and is welcomed by the community, including George Hotz's Tiny Corp. [The Register Article](https://www.theregister.com/2024/04/05/amd_mes_open_source/), [AMD Radeon Tweet](https://twitter.com/amdradeon/status/1775999856420536532).

- **Paper Replication Repository Launch**: An open-source repository dedicated to replicating research papers in AI & ML was announced. They encourage the community to contribute and leave stars on the GitHub repository. [PaperReplica GitHub Repo](https://github.com/hegdeadithyak/PaperReplica).
  
- **Seeking CUDA Setup Guides**: A new guide for setting up a CUDA development environment on Ubuntu was shared, detailing the installation of CUDA Toolkit, drivers, CuDNN, and OpenCV with CUDA support. The conversation also opened a discussion regarding the comfortability of working with CUDA on different systems. [Setup-as-Cuda-programmers GitHub Repo](https://github.com/CisMine/Setup-as-Cuda-programmers).
  
- **Innovative Sequence Parallelism for LMs**: EasyContext introduces sequence parallelism through ring attention, aiming to extend language models' context length to 1 million tokens while optimizing memory usage. The GitHub repository offers training recipes for this expansion with minimal hardware. [EasyContext GitHub Repo](https://github.com/jzhang38/EasyContext.git).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://discordapp.com/channels/1189498204333543425/1189640399476764692/1226242857501720596">Discord - A New Way to Chat with Friends &amp; Communities</a>: Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.</li><li><a href="https://x.com/amdradeon/status/1775999856420536532">Tweet from AMD Radeon (@amdradeon)</a>: We are working to release Micro-Engine Scheduler(MES) documentation towards end of May and will follow up with published source code for external review and feedback. We have also opened a GitHub trac...</li><li><a href="https://discordapp.com/channels/1189498204333543425/11">Discord - A New Way to Chat with Friends &amp; Communities</a>: Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.</li><li><a href="https://www.theregister.com/2024/04/05/amd_mes_open_source/">AMD to open source MES firmware for Radeon GPUs</a>: And it was all thanks to peer pressure</li><li><a href="https://github.com/hegdeadithyak/PaperReplica">GitHub - hegdeadithyak/PaperReplica: We Replicate Research Papers in the field of AI &amp; ML.</a>: We Replicate Research Papers in the field of AI &amp; ML. - hegdeadithyak/PaperReplica</li><li><a href="https://github.com/jzhang38/EasyContext.git">GitHub - jzhang38/EasyContext: Memory optimization and training recipes to extrapolate language models&#39; context length to 1 million tokens, with minimal hardware.</a>: Memory optimization and training recipes to extrapolate language models&#39; context length to 1 million tokens, with minimal hardware. - jzhang38/EasyContext</li><li><a href="https://github.com/CisMine/Setup-as-Cuda-programmers">GitHub - CisMine/Setup-as-Cuda-programmers</a>: Contribute to CisMine/Setup-as-Cuda-programmers development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1225935229366046801)** (7 messages): 

- **Library Appreciation Echoed**: A member expressed gratitude for a library that enhanced their understanding of **Triton** and improved their ability to debug Triton kernels.
- **Seeking Autotune Knowledge**: **@ryanatseattle** inquired about effective ways to autotune parameters such as `num_wrap`, `num_stage`, and `GROUP_SIZE` in Triton, mentioning the existing `triton.autotune` feature seems to only provide random configurations.
- **Auto-tune vs Benchmarking Dilemma**: An individual questioned how to best integrate auto-tuning with benchmarking, asking if they should first use auto-tune to determine optimal configurations and then proceed with benchmarking.
- **Performance Showdown - Custom CUDA vs Triton**: A query was raised regarding the efficiency comparison between custom CUDA kernels in PyTorch and Triton kernels.
- **Dot Product vs Addition Comment**: **@mobicham** made a cryptic comment hinting at a preference or observation regarding the use of dot products instead of additions, perhaps in the context of coding or algorithm optimization.
  

---


**CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1226988886484324473)** (1 messages): 

- **GPT-2 Goes Lean and Mean in C**: A member shared [Andrej Karpathy's recent GitHub project](https://github.com/karpathy/llm.c), llm.c, which enables training of GPT-2 in pure C without the heavy dependencies of PyTorch and Python. This lightweight implementation compiles and runs instantly, boasting clean code and a trainers delight at only ~1,000 lines.

**Link mentioned**: <a href="https://x.com/karpathy/status/1777427944971083809?s=46&t=ej2aClHUAjeapC55UGHfwg">Tweet from Andrej Karpathy (@karpathy)</a>: Have you ever wanted to train LLMs in pure C without 245MB of PyTorch and 107MB of cPython? No? Well now you can! With llm.c: https://github.com/karpathy/llm.c  To start, implements GPT-2 training on ...

  

---


**CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1226075591309262859)** (14 messages🔥): 

- **Torch compile with async ops dilemma**: A member brought up difficulties using **torch compile** with asynchronous collective operations and was directed towards new experimental *functional_collectives* that might support torch compile. However, these do not seem to support asynchronous operations like [async all reduce](https://github.com/pytorch/pytorch/blob/eff1e4899c7c89f8a8fc8f6ff6bed06dd8d2ec8a/torch/distributed/_functional_collectives.py#L169).

- **Asynchronous collective operations demystified**: In a follow-up, it was clarified that the new *functional_collectives* are indeed asynchronous and employ tensor subclassing magic to synchronize automatically, or alternatively, users can call `.wait()` for explicit synchronization.

- **DeepSpeed and Accelerate integration experiences**: Another member inquired about the integration of **DeepSpeed** with Hugging Face's **Accelerate**, focusing on whether features like mixture of experts (MoE) are lost. It was suggested that very few features are lost, but that one should manually define a deepspeed configuration JSON file rather than rely on HF trainer settings.

- **Unraveling Memory Usage Mysteries with DeepSpeed**: An observation was made that setting **zero stage** to 0, which should presumably disable it, still results in less memory consumption compared to Distributed Data Parallel (DDP), indicating that DeepSpeed might be running some optimization unknowingly.

- **Deciphering Triton utilization in torch compile**: It was highlighted by members discussing **torch compile** that **Triton** kernels are utilized only if the inputs are on a CUDA device, else C++ generated code is employed for fused kernels on CPU.

- **Flashing lights on MLPs**: A member keen on optimizing their transformer model's MLP shared that a cublas function-based MLP from the flash attention library wasn't yielding faster results than simple MLP with torch functionals. They received suggestions to explore further optimizations, potentially using [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn/blob/master/DOCUMENTATION.md) or CUTLASS over Triton, if fusing operations didn't outperform matrix multiplication library implementations.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/NVlabs/tiny-cuda-nn/blob/master/DOCUMENTATION.md">tiny-cuda-nn/DOCUMENTATION.md at master · NVlabs/tiny-cuda-nn</a>: Lightning fast C++/CUDA neural network framework. Contribute to NVlabs/tiny-cuda-nn development by creating an account on GitHub.</li><li><a href="https://github.com/pytorch/pytorch/blob/014f91a9d9f94ac9a7f0711600240d7cd7f69844/torch/_dynamo/variables/functions.py#L704,">pytorch/torch/_dynamo/variables/functions.py at 014f91a9d9f94ac9a7f0711600240d7cd7f69844 · pytorch/pytorch</a>: Tensors and Dynamic neural networks in Python with strong GPU acceleration - pytorch/pytorch</li><li><a href="https://github.com/pytorch/pytorch/blob/eff1e4899c7c89f8a8fc8f6ff6bed06dd8d2ec8a/torch/distributed/_functional_collectives.py#L169">pytorch/torch/distributed/_functional_collectives.py at eff1e4899c7c89f8a8fc8f6ff6bed06dd8d2ec8a · pytorch/pytorch</a>: Tensors and Dynamic neural networks in Python with strong GPU acceleration - pytorch/pytorch
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1226242857501720596)** (2 messages): 

- **CUDA-MODE Lecture Series Continues**: The next installment of the CUDA-MODE Lecture Series, *Lecture 13: Ring Attention*, is scheduled to start [at the announced time](<t:1712430000:t>), presented by the esteemed <@719599526448463933>.

- **Celebrating a Thriving Community**: With over 5,000 members, the CUDA-MODE Discord community celebrates its growth and expresses gratitude to its members. The continuity of delivering **one lecture per week** since its inception is highlighted as a keystroke of success. 

- **Applied Learning in Action**: The lectures have inspired members to apply their knowledge in the real world, contributing to many active working groups within the community. These practical efforts are evidenced by discussions and collaborations in specific channels.

- **Invitation to Expand the CUDA-MODE Family**: Community members are encouraged to invite performance-oriented friends to join the CUDA-MODE adventure and learn together. An invitation is extended via [discord.gg/cudamode](https://discord.gg/cudamode).
  

---


**CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1225857072311046219)** (7 messages): 

- **Revolutionizing Model Quantization**: [QuaRot](https://arxiv.org/abs/2404.00456), a new quantization scheme, allows for end-to-end quantization of large language models (LLMs) in 4 bits. It uniquely handles outliers and maintains computational invariance, achieving just 0.29 WikiText-2 perplexity loss and 99% zero-shot performance retention for their LLaMa2-70B model.

- **The Challenge of 4-bit Quantization**: An observation was made that although QuaRot is a promising advancement, unlike typical 4-bit quantization, it requires training/calibration for effective performance.

- **Scheduling Made Redundant in Optimization**: A member spotlighted [Schedule-Free Optimization in PyTorch](https://github.com/facebookresearch/schedule_free), a repository by Facebook Research, which presents a new approach utilizing schedule-free SGD or Adam.

- **Deep Optimizer Dive on Twitter**: A link to a [Twitter post](https://twitter.com/aaron_defazio/status/1773381393831067787) by Aaron Defazio was shared, potentially providing insights into the schedule-free optimization technique discussed earlier.

- **Casting Llama in a New Light**: A brief mention hinted at a possible connection between schedule-free optimization and something reminiscent of "Llama3", alluding to its significance.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2404.00456">QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs</a>: We introduce QuaRot, a new Quantization scheme based on Rotations, which is able to quantize LLMs end-to-end, including all weights, activations, and KV cache in 4 bits. QuaRot rotates LLMs in a way t...</li><li><a href="https://github.com/facebookresearch/schedule_free">GitHub - facebookresearch/schedule_free: Schedule-Free Optimization in PyTorch</a>: Schedule-Free Optimization in PyTorch. Contribute to facebookresearch/schedule_free development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[suggestions](https://discord.com/channels/1189498204333543425/1189868872887705671/1225793515984785538)** (1 messages): 

- **A Classic Resource for Parallel Algorithm Enthusiasts**: One member brought attention to an [Udacity course](https://www.youtube.com/watch?v=F620ommtjqk&list=PLAwxTw4SYaPnFKojVQrmyOGFCqHTxfdv2) they utilized for their dissertation in 2013. The course covers not only hardware and programming, but it's also focused on **parallel algorithms and performance**.
  

---


**CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1225993633631698984)** (1 messages): 

- **Installation Troubles with Nsight Compute**: A member experienced issues while installing **Nsight Compute** on **Ubuntu 22.04** using a `.run` file; despite following the installation steps, including `chmod +x`, the program did not appear after execution. Attempting to redo the `./nsight compute` command resulted in the program extracting again.
  

---


**CUDA MODE ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/)** (1 messages): 

itali4no: https://youtu.be/ws7angQYIxI?si=PcRy7siLQuFywpgp
  

---


**CUDA MODE ▷ #[jax](https://discord.com/channels/1189498204333543425/1203956655570817034/1225778502385537037)** (1 messages): 

- **Porting Triton Puzzles to Pallas**: There is an interest in porting **Triton puzzles** to Pallas, and it's suggested that this could be achieved through the **Triton backend** for those willing to investigate the possibilities.
  

---


**CUDA MODE ▷ #[ring-attention](https://discord.com/channels/1189498204333543425/1208496482005549086/1225927315758776403)** (8 messages🔥): 

- **Clarifying Linear Attention**: The discussion briefly touched on the nature of **linear attention**, confirming that it is not the same as classic attention.
- **Ring-Flash-Attention Script Shared**: A GitHub link was shared for a training script featuring **ring-flash-attention** by *jzhang38*. The project aims at context length extrapolation and is suggested to be included in the ring-attention repo's readme. [EasyContext on GitHub](https://github.com/jzhang38/EasyContext).
- **Exploring Context Parallelism**: A link to NVIDIA documentation was shared illustrating the concept of **Context Parallelism**, highlighting its difference from sequence parallelism and its impact on transformer models. [NVIDIA's Context Parallelism Docs](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/context_parallel.html).
- **Progress on Variable Length Striped Attention**: A participant mentioned they are working on implementing **varlen striped attention** but provided no further context on the progress or implications.
- **Questioning Ring Attention's Memory Usage**: A query was raised about **ring attention's** trade-off between speed and memory usage, particularly in the context of distributed computing and the buffering process in message-passing systems.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/context_parallel.html">Context parallelism overview - NVIDIA Docs</a>: no description found</li><li><a href="https://github.com/jzhang38/EasyContext">GitHub - jzhang38/EasyContext: Memory optimization and training recipes to extrapolate language models&#39; context length to 1 million tokens, with minimal hardware.</a>: Memory optimization and training recipes to extrapolate language models&#39; context length to 1 million tokens, with minimal hardware. - jzhang38/EasyContext
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1226221551347568741)** (3 messages): 

- **Clarification on Ring Attention Session Timezone**: There was a query regarding the timezone for a session on ring attention, with a clarification sought on whether it was PDT.

- **Naming Conventions in GPU Terminology**: A member expressed the opinion that "kernels" may not be the most suitable term for GPU kernels, suggesting it might be too late to change but questioning if others share this sentiment.
  

---


**CUDA MODE ▷ #[hqq](https://discord.com/channels/1189498204333543425/1225499037516693574/1225892929810333799)** (20 messages🔥): 

- **No Rounding for Quantized Tensors**: Group quantization does not involve rounding scales and only supports reshaping with the format `w.reshape(-1, groupsize)` for the calculation of scales and zero points.
- **Ensuring Quant and Dequant Consistency**: To verify accurate quantization and dequantization using the provided methods, one can quantize `W_hat` using `gpt-fast.quantize.group_quantize_tensor`, then dequantize and compare the sum of absolute differences with the original `W_hat`.
- **Quantization Approach Alignment**: Clarification that both parties seem to be employing *int4 affine, groupwise* quantization, albeit potentially along different axes, thus the suggested methods should be compatible.
- **Exploring Triton for Performance Gains**: Initial experimentation with Triton provided a significant 62x speed increase over PyTorch for unpacking 4-bit tensors on certain matrices, suggesting more optimization could yield even greater performance.
- **Quantization Integration and Testing on gpt-fast**: Updates in gpt-fast for HQQ 4bit quantization are showing promising token generation speeds, especially when the `--compile` flag is enabled, reaching 200 tokens per second. Quantization times and inference speeds appear to be in line with current baselines.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/mobiusml/hqq/blob/master/hqq/core/quantize.py#L83">hqq/hqq/core/quantize.py at master · mobiusml/hqq</a>: Official implementation of Half-Quadratic Quantization (HQQ) - mobiusml/hqq</li><li><a href="https://github.com/mobiusml/hqq/blob/master/examples/llama2_benchmark/eval_model.py#L12">hqq/examples/llama2_benchmark/eval_model.py at master · mobiusml/hqq</a>: Official implementation of Half-Quadratic Quantization (HQQ) - mobiusml/hqq</li><li><a href="https://github.com/zhxchen17/gpt-fast/commit/f94584359076dd484acf28119ec49ffc30ce87f1">HQQ 4 bit llama 2 4b · zhxchen17/gpt-fast@f945843</a>: export MODEL_REPO=meta-llama/Llama-2-7b-chat-hf scripts/prepare.sh $MODEL_REPO python quantize.py --checkpoint_path checkpoints/$MODEL_REPO/model.pth --mode int4-hqq --groupsize 64 python generate....</li><li><a href="https://github.com/pytorch-labs/gpt-fast?tab=readme-ov-file#evaluation.">GitHub - pytorch-labs/gpt-fast: Simple and efficient pytorch-native transformer text generation in &lt;1000 LOC of python.</a>: Simple and efficient pytorch-native transformer text generation in &lt;1000 LOC of python. - pytorch-labs/gpt-fast</li><li><a href="https://github.com/zhxchen17/gpt-fast/blob/f94584359076dd484acf28119ec49ffc30ce87f1/quantize.py#L455">gpt-fast/quantize.py at f94584359076dd484acf28119ec49ffc30ce87f1 · zhxchen17/gpt-fast</a>: Simple and efficient pytorch-native transformer text generation in &lt;1000 LOC of python. - zhxchen17/gpt-fast</li><li><a href="https://github.com/mobiusml/hqq/blob/master/hqq/core/quantize.py#L832-L837">hqq/hqq/core/quantize.py at master · mobiusml/hqq</a>: Official implementation of Half-Quadratic Quantization (HQQ) - mobiusml/hqq</li><li><a href="https://github.com/pytorch-labs/gpt-fast/blob/main/model.py#L131">gpt-fast/model.py at main · pytorch-labs/gpt-fast</a>: Simple and efficient pytorch-native transformer text generation in &lt;1000 LOC of python. - pytorch-labs/gpt-fast</li><li><a href="https://gist.github.com/mobicham/84ed1809c9c2f56c5c01fbcdbe22391f">eval_model_wikitext_gptfast.py</a>: GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://github.com/pytorch/pytorch/pull/106516/files#diff-b5f9afc0719fb33b38ccac5f6d4b566644fc9674e3477032ec3758ca8d833313R161">adding fused uint4x2_mixed_mm to inductor by HDCharles · Pull Request #106516 · pytorch/pytorch</a>: Stack from ghstack (oldest at bottom):  -&amp;gt; #106516  Summary: this is needed for int4 weight-only quantization, we&amp;#39;re matching on the specific unpack operation that unpacks the uint4x2 i...
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[triton-viz](https://discord.com/channels/1189498204333543425/1225499141241573447/1225777509120086059)** (23 messages🔥): 

- **Debating the Overkill of Three.js**: Members discussed the potential use of **Three.js** for visualizations, but some felt it might be too powerful and complex for their needs. Consideration was given to using **D3** as a more interaction-friendly option.

- **Visualizing Shared Memory and Tensors**: There was a conversation about visual representations in **triton-viz**, considering how to display shared memory and tensor views effectively. One member plans to use **ipycanvas + ipyevents** for rich visuals within Jupyter, supplementing the current Gradio setup.

- **Triton Debugging Challenges**: The group discussed common issues encountered while debugging Triton code, specifically the frequent problem of loading data into the wrong place. A focus was suggested on visualizing data origins in kernels to aid developers.

- **Triton Visualization for CPU Constructs**: Members expressed interest in visualizing loops and control flow constructs within **triton-viz**, though concerns were raised about the current view's intuitiveness for such features. Brainstorming was encouraged among the members for potential solutions.

- **Interactive Debugging with JavaScript**: There was a suggestion to implement visual debugging tools in JavaScript to enhance interactions, such as mouse-over effects and quick animations, to enable better understanding and clearer tutorials for Triton's debugging traces.
  

---



**tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1225854310093426830)** (59 messages🔥🔥): 

- **Tinygrad Learning Resources**: For those looking to contribute to tinygrad, you can find tutorials and documentation at [GitHub - mesozoic-egg/tinygrad-notes](https://github.com/mesozoic-egg/tinygrad-notes).

- **Reversion of the Command Queue**: George Hotz mentioned a reversion of the command queue in tinygrad development with a comment, *lol no, reverted*.

- **Memory Scheduler Integration Strategy**: According to George Hotz, the memory scheduler is to be integrated into the scheduler itself, and *the queue stuff can be handled with the existing multidevicegraph abstraction*.

- **Exploration of RISC-V Opcodes in Firmware**: Members discussed the architecture of MEC firmware, debating whether it is RISC-V based and analyzing differing opcode structures, including an unexpected `cbz` instruction.

- **Usage Guidelines for TinyJit Requested**: A member sought advice on using TinyJit and whether issues they encountered were due to misuse or a bug, sparking a further discussion on the nuances of RISC-V ISA including ARM mnemonic use.

- **Tinygrad Role Redefinition and Community Responsibilities Emphasized**: George Hotz updated the Discord roles to reflect contributions and involvement, highlighting the importance of effective collaboration and mindful use of others' time in the tinygrad project development.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/mesozoic-egg/tinygrad-notes/tree/main">GitHub - mesozoic-egg/tinygrad-notes: Tutorials on tinygrad</a>: Tutorials on tinygrad. Contribute to mesozoic-egg/tinygrad-notes development by creating an account on GitHub.</li><li><a href="https://github.com/tinygrad/tinygrad/pull/4094">new memory scheduler with LRU by geohot · Pull Request #4094 · tinygrad/tinygrad</a>: no description found
</li>
</ul>

</div>
  

---


**tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1225851944602701964)** (6 messages): 

- **TinyJIT Unveiled**: A member shared a [tutorial on TinyJit](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/jit.md) for those interested in how it works, though it may contain some inaccuracies, particularly in the `apply_graph_to_jit` section.
- **Clarifying TinyJIT Mechanics and Seeking Insight**: The member who shared the TinyJit tutorial noted possible issues with the runtime under `/graph` folder and invited others to message with insights to improve the accuracy of the content.
- **Call for Error Correction on TinyJIT Tutorial**: In response to potential inaccuracies, another member requested for "reds" to create an error-correcting pull request to aid the community.
- **Diving into Multi GPU Training with Tinygrad**: Another tutorial explaining how tinygrad implements multi GPU training was introduced, with the source available on [GitHub](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/multigpu.md).
- **Community Praise for Multi GPU Training Guide**: The multi GPU training tutorial was well-received, acknowledged by a member as very useful.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/mesozoic-egg">mesozoic-egg - Overview</a>: mesozoic-egg has 4 repositories available. Follow their code on GitHub.</li><li><a href="https://github.com/mesozoic-egg/tinygrad-notes/blob/main/jit.md">tinygrad-notes/jit.md at main · mesozoic-egg/tinygrad-notes</a>: Tutorials on tinygrad. Contribute to mesozoic-egg/tinygrad-notes development by creating an account on GitHub.</li><li><a href="https://github.com/mesozoic-egg/tinygrad-notes/blob/main/multigpu.md">tinygrad-notes/multigpu.md at main · mesozoic-egg/tinygrad-notes</a>: Tutorials on tinygrad. Contribute to mesozoic-egg/tinygrad-notes development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



**Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1225827225325146112)** (26 messages🔥): 

- **Appeals to Resolve Llamafile False Positives**: Llamafile versions face *false positive* malware detections, possibly affecting **llamafile-0.6.2.exe** and **llamafile-0.7**. An appeal to those AVs with an appeals form was suggested as a possible action to take.
  
- **GPU Issues with Llamafile in Kaggle**: A user experienced issues when running `llamafile` in Kaggle, due to complications with compiling CUDA and finding a compatible GPU arch. Another user provided an **updated command** to facilitate `llamafile-0.7` usage.

- **Local Distribution Considerations for RAG-LLM Application**: A member inquired about distributing a RAG-LLM application locally without heavy dependencies like Docker or Python and was open to using **llamafile** for macOS users. An assurance was given that **llamafile** could meet these requirements.

- **Llamafile Out of Memory Error Resolved by Adjusting `-ngl`**: One user successfully resolved an out of memory error by **tweaking the `-ngl` argument**, which they initially set too high for their NVIDIA GeForce GTX 1050 card.

- **Vulkan Support Enhancement Proposed**: A suggestion was made to add support for Vulkan in llamafile after testing showed improved performance on a modest Intel-based laptop with an integrated GPU. However, concerns were raised about the need to re-import and apply local changes to **llama.cpp** to achieve this.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.virustotal.com/gui/file/57a2ad7b2458896e8936f00cd4c91c8b4c919fceab35bfd3f85371b3a84dc935">VirusTotal</a>: no description found</li><li><a href="https://www.virustotal.com/gui/file/57a2ad7b2458">VirusTotal</a>: no description found</li><li><a href="https://www.virustotal.com/gui/file/37a39d8970573110c425c3edd1be4b1df6ab32c4a4a38ae6d98ad4728093267e">VirusTotal</a>: no description found
</li>
</ul>

</div>
  

---



**DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1226255944778780783)** (9 messages🔥): 

- **Schedule-Free Optimizers come to HF Transformers**: A new pull request in the [huggingface/transformers repository](https://github.com/huggingface/transformers/pull/30079) introduces integration of Meta's schedule-free optimizers for AdamW and SGD, which could be a significant update for the training of models.
- **Training with AdaptiveSoftmax?**: One member is seeking insights or success stories from others regarding training with *adaptivesoftmax* but did not provide specific details or context.
- **AI Community Event in Germany**: The "AIDEV" Community event is announced for AI engineers in Germany, taking place in Hürth on May 7th. Discussions will center around synthetic data generation, LLM/RAG pipelines, and embeddings with a developer-centric, no-nonsense approach. Interested parties can register for free at [Developer Event - AI Village](https://www.eventbrite.de/e/developer-event-ai-village-tickets-868896702427).
- **Request for Public Info on Synthetic Data Generation**: One member inquires about public information or discussions on synthetic data generation and related strategies, particularly in German contexts, with mentions of German translated vs. German generated data.
- **Post-Event Summary Sharing**: Several members express enthusiasm for the upcoming event in Hürth, Germany, with requests to share summaries and insights post-event for those who cannot attend or are eager to digest the discussed content.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/huggingface/transformers/pull/30079">schedulefree optimizers by winglian · Pull Request #30079 · huggingface/transformers</a>: What does this PR do? integrates meta&#39;s https://github.com/facebookresearch/schedule_free for adamw &amp; sgd https://twitter.com/aaron_defazio/status/1776320004465582331 Before submitting   This ...</li><li><a href="https://www.eventbrite.de/e/developer-event-ai-village-tickets-868896702427">Developer Event AI Village</a>: AIDev - Developer Community Large Language Models, LLM Application and generative AI
</li>
</ul>

</div>
  

---


**DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/1225740644690034698)** (5 messages): 

- **Inspirational Command-R Performance**: A link to [**Command-R** space on Hugging Face](https://huggingface.co/spaces/CohereForAI/c4ai-command-r-plus) was shared, described as "mindblowing" due to its impressive grounding capabilities, potentially influencing the development of future models.
- **Setting New Benchmarks in Middle High German Translation**: Command-R from *CohereForAI* excels at translating Middle High German into modern German, effortlessly outperforming GPT-4 class models and making months of specialized training on other LLMs appear obsolete.
- **Implications for Developer Activity and Open-Source Licensing**: The hope is expressed that Cohere will adopt a fully open-source license for their new, superior model, as this would likely boost developer engagement and ecosystem growth, with Mistral serving as an example of the economic benefits of such a strategy.
- **Concrete Examples of Command-R Superiority**: It’s claimed that Command-R provides perfect translations from Middle High German and seems to recognize source material, indicating strong needle-haystack capabilities, which makes it a prime candidate for RAG (retrieval-augmented generation) functions integration.

**Link mentioned**: <a href="https://huggingface.co/spaces/CohereForAI/c4ai-command-r-plus">C4AI Command R Plus - a Hugging Face Space by CohereForAI</a>: no description found

  

---



**AI21 Labs (Jamba) ▷ #[jamba](https://discord.com/channels/874538902696914944/1222916247063232553/1225925361955770498)** (9 messages🔥): 

- **Jamba's Training Speed Caution**: One member reported that training a 1B Mamba model on an HGX was *76% slower* than its transformer counterpart. After some clarification, it was established they were comparing the training speed to that of regular Transformers.

- **Alternate Jamba Solutions for Limited Hardware**: A user created downsized versions of the **Jamba** architecture, the [8xMoE with 29B parameters](https://huggingface.co/isemmanuelolowe/Jamba-8xMoE_Slerp) and [4xMoE with 17.7B parameters](https://huggingface.co/isemmanuelolowe/Jamba-4xMoE_Slerp), for those unable to run the full 52B model locally. These models have shown promising results and can be fine-tuned on a 4090 GPU at 4 bit.

- **Sharing Downscaled Jamba Techniques**: In response to curiosity about how the reduced parameter models were created, the user mentioned using an accumulative Slerp (spherical linear interpolation) of the expert weights and promised to share an `ipynb` notebook soon.

- **Inference Engine Queries**: A member sought advice on the best inference engine to serve **Jamba** models, but no direct recommendations followed in the given messages.

- **Challenges with GPU Utilization for Jamba**: A user successfully replicated the fine-tuning example from the [Hugging Face Jamba model page](https://huggingface.co/ai21labs/Jamba-v0.1), but had to spread the 52B model training across 8 GPUs due to its size. They are experiencing limitations due to pipeline parallelism, which is causing only 1/8th of the total GPU capacity to be utilized and inquired about training with Tensor Parallelism (TP).
  

---



**Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/1225899553975439431)** (2 messages): 

- **Powering AI with QNAP NAS**: A member highlighted a [practical at-home setup for AI](https://www.storagereview.com/review/run-a-private-rag-chatgpt-on-qnap-nas) using a **QNAP NAS** with a GPU added to test AI capabilities. The setup in question involves the TS-h1290FX model, which boasts an AMD EPYC 7302P CPU, 256GB DRAM, and 25GbE capability.
- **Storing System Prompts for Efficiency**: A member inquired whether others have begun storing and retrieving system prompts for common tasks to streamline the process of setting up context in AI interactions. No further context or responses were provided in the available messages.

**Link mentioned**: <a href="https://www.storagereview.com/review/run-a-private-rag-chatgpt-on-qnap-nas">Run a Private RAG ChatGPT on QNAP NAS</a>: QNAP NAS platforms have the most unique and capable hardware designes in the category. We added a GPU to one and tested the AI capabilities.

  

---


**Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/1226911956590399512)** (3 messages): 

- **Introducing Alter**: Alter, born from the use of llm-cli, is set to launch in beta and brings AI-powered text improvement functionalities to macOS, with [a demonstration video available on YouTube](https://youtu.be/IK53CSSbaqI). The app integrates with various macOS applications, including Keynote, to generate and edit content. 
- **AI at Your Fingertips with Alter**: Alter promises context-aware AI capabilities across all macOS applications, offering a centralized AI tool to replace multiple subscriptions and addons. Information on features, pricing, and capabilities can be found on the [Alter website](https://alterhq.com).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://alterhq.com">Alter | Invisible AI for your Mac</a>: no description found</li><li><a href="https://youtu.be/IK53CSSbaqI">Alter demo  - Fix typo and grammar</a>: Improve grammar and spelling across all your apps!Alterhq.com recommends the best actions tailored to your work context.#ai #spelling #macos #chatgpt
</li>
</ul>

</div>
  

---



**Skunkworks AI ▷ #[general](https://discord.com/channels/1131084849432768614/1131084849906716735/1226632752988815380)** (1 messages): 

- **Neural Network Resource Allocation Innovations**: Member discussing a paper where **dynamic allocation** of a static compute budget is managed per token within a neural network. This strategy piqued the member's interest for implementation in **neurallambda**, raising the idea that a network could discern how to distribute computational resources optimally.
- **Pondering New Training Techniques**: The member contemplates incorporating various methods for **neurallambda**, such as using pause/think tokens, implementing conditionals through reinforcement learning, and drawing inspiration from a paper where RNNs emitted their own compute usage.
- **Exploring Neural Input Processing Methods**: Additional considerations for **neurallambda** include reading input into a neural queue for flexible processing and treating input as a tape with the ability to emit tape movements on-demand, resembling a Turing machine's operation.
  

---


**Skunkworks AI ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/1225759372379029534)** (2 messages): 

- **Exploring Structured Data Extraction**: A video titled "Instructor, Generating Structure from LLMs" was shared, demonstrating how to extract structured data like JSON from Large Language Models including **GPT-3.5, GPT-4**, and **GPT-4-Vision**. The video aims to make it easier to get reliable structured results from LLMs. [Watch the video here.](https://www.youtube.com/watch?v=KxOqjKq2VyY)
- **Another Video Shared**: A second YouTube video link was provided without additional context. [Check out the video.](https://www.youtube.com/watch?v=keUjQyWmgY0)

**Link mentioned**: <a href="https://www.youtube.com/watch?v=KxOqjKq2VyY">Instructor, Generating Structure from LLMs</a>: Instructor makes it easy to reliably get structured data like JSON from Large Language Models (LLMs) like GPT-3.5, GPT-4, GPT-4-Vision, including open source...

  

---



**LLM Perf Enthusiasts AI ▷ #[claude](https://discord.com/channels/1168579740391710851/1168582222194933860/1226270438519472178)** (2 messages): 

- **Searching for Haiku Speed Solutions**: A member inquired about ways to optimize **Haiku** performance, as they are experiencing unacceptable speed with the current setup.

- **Anthropic's API Grabs the Spotlight**: Results shared by a user show **Anthropic’s** new tool use beta API has outperformed **GPT-4 Turbo** in half of the scenarios on the Berkeley function calling benchmark. Full experimental outcomes are detailed in a [Twitter thread](https://x.com/JoschkaBraun/status/1777381282751688868).

**Link mentioned**: <a href="https://x.com/JoschkaBraun/status/1777381282751688868">Tweet from Joschka Braun (@JoschkaBraun)</a>: I benchmarked @AnthropicAI&#39;s new tool use beta API on the Berkeley function calling benchmark. Haiku beats GPT-4 Turbo in half of the scenarios. Results in 🧵  A huge thanks to @shishirpatil_, @fa...

  

---





