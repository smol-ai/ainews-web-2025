---
id: 2cd5ffe8-2313-4289-b37a-bdf7806008a7
title: AI Engineer Summit Day 1
date: '2025-02-22T02:50:34Z'
original_slug: ainews-ai-engineer-summit-day-1
description: >-
  The AIE Summit in NYC highlighted major developments including OpenAI's first
  formal definition of Agents and a significant $930 million funding
  announcement from speakers and sponsors. On AI model performance, Grok-3's
  capabilities sparked debate with mixed evaluations, while o3-mini demonstrated
  superior performance and effective debugging of CUDA kernel issues.
tags:
  - aie-summit
  - openai
  - agents
  - funding
  - grok-3
  - o3-mini
  - cuda-kernels
---


<!-- buttondown-editor-mode: plaintext -->**AI Engineers are all you need.**

> AI News for 2/19/2025-2/20/2025. We checked 7 subreddits, [**433** Twitters](https://twitter.com/i/lists/1585430245762441216) and **29** Discords (**211** channels, and **6423** messages) for you. Estimated reading time saved (at 200wpm): **647 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

Day 1 of AIE Summit has concluded here in NYC.

If you forced us to pick only 3 talks to focus on, check out  [Grace Isford's Trends Keynote](https://www.youtube.com/live/L89GzWEILkM?si=ZDW5jhKAD4LVyQVx&t=1033), [Neo4j/Pfizer](https://www.youtube.com/live/L89GzWEILkM?si=xkUBa6CUDYIZtJfw&t=7632)'s presentation, and [OpenAI defining Agents for the first time](https://www.youtube.com/live/L89GzWEILkM?si=TC5qcVHcSE1ny1wq&t=11410). [$930m of funding](https://x.com/swyx/status/1892771856484122933) was announced by speakers/sponsors. [Multiple Anthropic datapoints](https://x.com/swyx/status/1892684773891375125) went semi-viral.

![image.png](https://assets.buttondown.email/images/5eb12543-f1b0-46c4-87ff-1047282c222a.png?w=960&fit=max)


You can watch back the full VOD here:

https://www.youtube.com/watch?v=L89GzWEILkM

Day 2 will [focus on Agent Engineering](https://www.youtube.com/watch?v=D7BzTxVVMuw), while Day 3 will [have IRL workshops and the new Online track](https://www.latent.space/p/2025-summit-online).



---


{% if medium == 'web' %}


**Table of Contents**

[TOC] 

{% else %}

The **Table of Contents** and **Channel Summaries** have been moved to the web version of this email: [{{ email.subject }}]({{ email_url }})!

{% endif %}


---

# AI Twitter Recap

**Models, Benchmarks, and Performance**

- **Grok-3 Performance and Capabilities**: [@BorisMPower](https://twitter.com/BorisMPower/status/1892407015038996740) reported that **o3-mini is better in every eval compared to Grok 3**, stating that Grok 3 is decent but oversold. This sparked discussion with [@ibab](https://twitter.com/ibab/status/1892418351084732654) from xAI who responded that they used the same evaluation methods. [@Yuhu_ai_](https://twitter.com/Yuhu_ai_/status/1892449337218883868) from xAI defended Grok 3's performance, claiming their **mini model surpassed o3-mini high in AIME 2024, GPQA, and LCB for pass@1**, and that benchmarks don't fully capture model intelligence. [@aidan_mclau](https://twitter.com/aidan_mclau/status/1892416555868201321) criticized Grok 3's chart presentation as "chart crimes". [@itsclivetime](https://twitter.com/itsclivetime/status/1892463726810583245) shared initial positive experiences with **Grok 3, noting its speed in Deep Research**, but also mentioned slower coding and occasional crashes. [@nrehiew_](https://twitter.com/nrehiew_/status/1892469273446035924) defended xAI's evaluation reporting, saying it follows OpenAI's practices and the issue was clarity, not deception. [@teortaxesTex](https://twitter.com/teortaxesTex/status/1892509598612877503) expressed surprise at the grief received for bullishness on Grok.  [@EpochAIResearch](https://twitter.com/EpochAIResearch/status/1892671695535677745) noted **Grok-3's record compute scale**, estimating **4e26 to 5e26 FLOP**, making it the first released model trained on over 1e26 FLOP.

- **o3-mini Performance and CUDA Kernel Issue**: [@giffmana](https://twitter.com/giffmana/status/1892510741242036468) highlighted that **o3-mini figured out an issue with Sakana AI's CUDA kernels in 11 seconds**, revealing a bug that made it appear 150x faster when it was actually 3x slower.  [@giffmana](https://twitter.com/giffmana/status/1892510744224182661) emphasized lessons learned: **straightforward CUDA code is unlikely to outperform optimized kernels**, **inconsistent benchmarks indicate problems**, and **o3-mini is highly effective for debugging**. [@main_horse](https://twitter.com/main_horse/status/1892446384910987718) also benchmarked and found **Sakana AI's claimed 150x speedup to be actually 3x slower**, pointing to issues with their CUDA kernel.

- **DeepSeek R1 Capabilities and Training**: [@teortaxesTex](https://twitter.com/teortaxesTex/status/1892636514221166644) mentioned a **"R1-inspired Cambrian explosion in RL"**, noting its scientific recipe is similar to other top labs, highlighting a shift away from demoralizing "hopeless BS". [@togethercompute](https://twitter.com/togethercompute/status/1892609242957582505) promoted **DeepSeek-R1 as an open-source alternative to proprietary models**, offering fast inference on NVIDIA GPUs. [@andrew_n_carr](https://twitter.com/andrew_n_carr/status/1892403826717569120) shared a **cool fact about DeepSeek's training**, noting a **batch size of ~60M tokens for 14 trillion tokens**, contrasting with Llama 1's smaller batch size.

- **Qwen 2.5-VL Model Release**: [@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1892576737848160538) announced the **tech report for Qwen2.5-VL**, detailing its architecture and training, highlighting its **capability alignment with Qwen2.5-72B and industry-leading visual semantic parsing**. [@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1892576743904670022) also released **AWQ quantized models for Qwen2.5-VL in 3B, 7B, and 72B sizes**. [@_akhaliq](https://twitter.com/_akhaliq/status/1892433462910501170) shared the **Qwen2.5-VL Technical Report drop**.  [@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1892422884049768473) also announced the **Qwen2.5-VL Technical Report release**. [@_philschmid](https://twitter.com/_philschmid/status/1892506190656999925) detailed **how Qwen Vision Language Models are trained**, emphasizing dynamic resolution processing and a redesigned Vision Transformer.

- **SmolVLM2 Video Models**: [@mervenoyann](https://twitter.com/mervenoyann/status/1892576290181382153) announced **SmolVLM2, "world's smollest video models"** in 256M, 500M, and 2.2B sizes, including an **iPhone app, VLC integration, and a highlights extractor**. [@reach_vb](https://twitter.com/reach_vb/status/1892578169615523909) highlighted **SmolVLM2, Apache 2.0 licensed VideoLMs ranging from 2.2B to 256M**, noting they can run on a free Colab and even an iPhone. [@awnihannun](https://twitter.com/awnihannun/status/1892594913893556707) promoted **SmolVLM2's day-zero support for MLX and MLX Swift**, enabling local runs on Apple devices.

- **Helix VLA Model for Robotics**: [@adcock_brett](https://twitter.com/adcock_brett/status/1892579315461599658) announced the **technical report for Helix, a generalist Vision-Language-Action (VLA) model**. [@adcock_brett](https://twitter.com/adcock_brett/status/1892579188424712682) described **Helix's architecture as "System 1, System 2"**, with a 7B parameter VLM and an 80M parameter visuomotor policy, running on embedded GPUs. [@adcock_brett](https://twitter.com/adcock_brett/status/1892579136956186947) showcased **Helix robots picking up household items**, and [@adcock_brett](https://twitter.com/adcock_brett/status/1892579000817521092) detailed **Helix coordinating a 35-DoF action space at 200Hz**. [@adcock_brett](https://twitter.com/adcock_brett/status/1892578885226635525) presented **two robots collaboratively storing groceries using Helix**. [@adcock_brett](https://twitter.com/adcock_brett/status/1892578309344502191) emphasized **Helix's human-like thinking and generalization capabilities for robotics**. [@adcock_brett](https://twitter.com/adcock_brett/status/1892577936869327233) introduced **Helix as "AI that thinks like a human"**, aiming for robots in homes.

- **SholtoBench AGI Benchmark**: [@nearcyan](https://twitter.com/nearcyan/status/1892469757653442989) announced **SholtoBench, a new AGI benchmark tracking Sholto Douglas's (@_sholtodouglas) AGI lab employment**. [@nearcyan](https://twitter.com/nearcyan/status/1892470292758614148) provided a link to the **official SholtoBench website** and thanked anonymous contributors.

- **AIME 2025 Performance Chart**: [@teortaxesTex](https://twitter.com/teortaxesTex/status/1892487275948224793) shared a **"definitive Teortaxes edition" performance chart for AIME 2025**, comparing models like o3-mini, Grok-3, DeepSeek-R1, and Gemini-2 FlashThinking. [@teortaxesTex](https://twitter.com/teortaxesTex/status/1892482831680504224) commented on labs releasing "asinine, deformed charts" to claim SoTA. [@teortaxesTex](https://twitter.com/teortaxesTex/status/1892471638534303946) presented a **compilation of AIME 2025 results**, aiming for clarity over "chart crimes".

- **Grok DeepSearch Evaluation**: [@casper_hansen_](https://twitter.com/casper_hansen_/status/1892531542548684820) found **Grok DeepSearch "pretty good"**, noting its query expansions and questioning its comparison to OpenAI's DeepResearch.

- **LLM Scaling Laws and Data Quality**: [@JonathanRoss321](https://twitter.com/JonathanRoss321/status/1892596586347160059) discussed **LLM scaling laws**, arguing that **improvement can continue with better data quality, even if internet data is exhausted**, citing AlphaGo Zero's self-play as an example of synthetic data driving progress.

- **FlexTok Image Tokenizer**: [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1892550422050877486) highlighted **FlexTok, a new tokenizer from Apple and EPFL**, projecting 2D images into variable-length 1D token sequences, allowing for hierarchical and semantic compression.

- **Vision Language Model Training**: [@_philschmid](https://twitter.com/_philschmid/status/1892506190656999925) explained **how Vision Language Models like @Alibaba_Qwen 2.5-VL are trained**, detailing pre-training phases (ViT only, Multimodal, Long-Context) and post-training (SFT & DPO).

- **vLLM Speedup with DeepSeek's Module**: [@vllm_project](https://twitter.com/vllm_project/status/1892646680719216960) announced **vLLM v0.7.3 now supports DeepSeek's Multi-Token Prediction module**, achieving up to **69% speedup boost**.

**Open Source and Community**

- **Open Source AI Models**: [@togethercompute](https://twitter.com/togethercompute/status/1892609241212715045) affirmed their belief that **"the future of AI is open source"**, building their cloud company around open-source models and high-performance infrastructure. [@_akhaliq](https://twitter.com/_akhaliq/status/1892600666276671710) congratulated [@bradlightcap](https://twitter.com/bradlightcap) and suggested **open models could further enhance their success**. [@cognitivecompai](https://twitter.com/cognitivecompai/status/1892648693691551839) expressed **love for new Apache 2.0 drops from @arcee_ai**.

- **Hugging Face Inference Support Expansion**: [@_akhaliq](https://twitter.com/_akhaliq/status/1892628229871030602) announced **Hugging Face Inference providers now support over 8 different providers and close to 100 models**.

- **LangChain Agent Components and Open Deep Research**: [@LangChainAI](https://twitter.com/LangChainAI/status/1892675173226316073) promoted **Interrupt conference with speakers from Uber sharing reusable agent components with LangGraph**. [@LangChainAI](https://twitter.com/LangChainAI/status/1892645710224622024) introduced **Open Deep Research**, a configurable open-source deep researcher agent. [@LangChainAI](https://twitter.com/LangChainAI/status/1892642089529442697) highlighted **Decagon's AI Agent Engine**, used by companies like Duolingo and Notion, in a fireside chat.

- **Unsloth Memory Efficient GRPO**: [@danielhanchen](https://twitter.com/danielhanchen/status/1892643424538595611) announced **memory savings of up to 90% for GRPO (algorithm behind R1) in @UnslothAI**, achieving 20K context length GRPO with 54GB VRAM versus 510GB in other trainers.

- **Lumina2 LoRA Fine-tuning Release**: [@RisingSayak](https://twitter.com/RisingSayak/status/1892462411451412674) announced **Lumina2 LoRA fine-tuning release** under Apache 2.0 license.

- **Offmute Open Source Meeting Summarization**: [@_philschmid](https://twitter.com/_philschmid/status/1892599725913768161) presented **Offmute, an open-source project using Google DeepMind Gemini 2.0 to transcribe, analyze, and summarize meetings**, generating structured reports and key points.

- **SongGen Open-Source Text-to-Music Model**: [@multimodalart](https://twitter.com/multimodalart/status/1892533897537192366) announced **SongGen, joining YuE as an open-source text-to-music model**, similar to Suno, allowing users to create songs from voice samples, descriptions, and lyrics.

**Research and Development**

- **AI CUDA Engineer - Agentic CUDA Kernel Optimization**: [@DrJimFan](https://twitter.com/DrJimFan/status/1892404919480832259) highlighted **Sakana AI's "AI CUDA Engineer," an agentic system that produces optimized CUDA kernels**, using AI to accelerate AI. [@omarsar0](https://twitter.com/omarsar0/status/1892621241674301761) broke down **Sakana AI's AI CUDA Engineer**, explaining its end-to-end agentic system for kernel optimization. [@SakanaAILabs](https://twitter.com/SakanaAILabs/status/1892433535400890734) announced the **"AI CUDA Engineer," an agent system automating CUDA kernel generation, potentially speeding up model processing by 10-100x**, and releasing a dataset of 17,000+ CUDA kernels. [@omarsar0](https://twitter.com/omarsar0/status/1892621325136810001) detailed the **Agentic Pipeline of the AI CUDA Engineer**, including PyTorch to CUDA conversion and evolutionary optimization. [@omarsar0](https://twitter.com/omarsar0/status/1892621450340921345) mentioned the availability of an **archive of 17000+ verified CUDA kernels** created by the AI CUDA Engineer.

- **Thinking Preference Optimization (TPO)**: [@_akhaliq](https://twitter.com/_akhaliq/status/1892431954085024189) shared a link to **research on Thinking Preference Optimization**.

- **Craw4LLM for Efficient Web Crawling**: [@_akhaliq](https://twitter.com/_akhaliq/status/1892435546703638628) posted about **Craw4LLM, efficient web crawling for LLM pretraining**.

- **RAD for Driving Policy via 3DGS-based RL**: [@_akhaliq](https://twitter.com/_akhaliq/status/1892435007412621429) shared **RAD research on training an end-to-end driving policy using large-scale 3DGS-based Reinforcement Learning**.

- **Autellix - Efficient Serving Engine for LLM Agents**: [@_akhaliq](https://twitter.com/_akhaliq/status/1892434670597345474) highlighted **Autellix, an efficient serving engine for LLM agents as general programs**.

- **NExT-Mol for 3D Molecule Generation**: [@_akhaliq](https://twitter.com/_akhaliq/status/1892438110480302474) shared **NExT-Mol research on 3D Diffusion Meets 1D Language Modeling for 3D Molecule Generation**.

- **Small Models Learning from Strong Reasoners**: [@_akhaliq](https://twitter.com/_akhaliq/status/1892435858684248326) linked to research on **Small Models Struggling to Learn from Strong Reasoners**.

- **NaturalReasoning Dataset for Complex Reasoning**: [@maximelabonne](https://twitter.com/maximelabonne/status/1892539204875227642) introduced **NaturalReasoning, a new instruction dataset designed to improve LLMs' complex reasoning without human annotation**, emphasizing quality over quantity and diverse training data.

- **Fine-grained Distribution Refinement for Object Detection**: [@skalskip92](https://twitter.com/skalskip92/status/1892497124534747193) introduced **D-FINE, a "new" SOTA object detector** using Fine-grained Distribution Refinement, improving bounding box accuracy through iterative edge offset adjustments and sharing precise distributions across network layers.

- **BioEmu for Biomolecular Equilibrium Structure Prediction**: [@reach_vb](https://twitter.com/reach_vb/status/1892656772759916860) highlighted **Microsoft's BioEmu, a large-scale deep learning model for efficient prediction of biomolecular equilibrium structure ensembles**, capable of sampling thousands of structures per hour.

**Robotics and Embodiment**

- **Figure's Helix Humanoid Robot AI**: Figure AI is developing **Helix**, an AI model for humanoid robots, showcased through various capabilities like grocery storage and object manipulation (tweets from [@adcock_brett](https://twitter.com/adcock_brett)). They are scaling their AI team for **Helix, Training Infra, Large Scale Training, Manipulation Engineer, Large Scale Model Evals, and Reinforcement Learning** ([@adcock_brett](https://twitter.com/adcock_brett/status/1892579357182345588)). They are aiming for production and shipping more robots by **2025**, focusing on home robotics ([@adcock_brett](https://twitter.com/adcock_brett/status/1892579860289130520)).

- **7B LLM on Robots vs. o3 for Math**: [@abacaj](https://twitter.com/abacaj/status/1892622993148313747) stated that **"putting a 7B LLM on a robot is more interesting than using o3 to solve phd level math problems"**. [@abacaj](https://twitter.com/abacaj/status/1892611093802910152) found a **7B parameter onboard vision-based LLM powering a robot "interesting and sort of expected"**, noting increased model capability. [@abacaj](https://twitter.com/abacaj/status/1892623488889831520) humorously suggested **"a 7B LLM will do your dishes, o3 won't"**.

- **Skyfire AI Drone Saves Police Officer**: [@AndrewYNg](https://twitter.com/AndrewYNg/status/1892628887856939236) shared a story about a **Skyfire AI drone saving a police officer's life**, by locating an officer in distress during a traffic stop, enabling rapid backup and intervention.

**Tools and Applications**

- **Glass 4.0 AI Clinical Decision Support Platform**: [@GlassHealthHQ](https://twitter.com/GlassHealthHQ/status/1892574802327523360) introduced **Glass 4.0, their updated AI clinical decision support platform**, featuring continuous chat, advanced reasoning, expanded medical literature coverage, and increased response speed.

- **AI-Toolkit UI**: [@ostrisai](https://twitter.com/ostrisai/status/1892424544356294978) shared progress on the **AI-Toolkit UI**, noting the "hard stuff is done" and UI cleanup is underway before adding "fun features".

- **Gradio Sketch for AI App Building**: [@_akhaliq](https://twitter.com/_akhaliq/status/1892604706377052357) highlighted a **new way to build AI apps using "gradio sketch,"** enabling visual component selection and configuration to generate Python code.

- **Gemini App Deep Research**: [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1892629054311772463) announced **Deep Research is available in the Gemini App for Gemini Advanced users in 150 countries and 45+ languages**, functioning as a personal AI research assistant.

- **Elicit Systematic Reviews**: [@elicitorg](https://twitter.com/elicitorg/status/1892592908563534221) introduced **Elicit Systematic Reviews, supporting automated search, screening, and data extraction for research reviews**, aiming to accelerate research with user control.

- **PocketPal Mobile App with Qwen 2.5 Models**:  [Qwen 2.5 models, including 1.5B (Q8) and 3B (Q5_0) versions, have been added](https://twitter.com/ANOTHER_HANDLE/status/SOME_ID) to the PocketPal mobile app for both iOS and Android platforms. Users can provide feedback or report issues through the project's GitHub repository, with the developer promising to address concerns as time permits. The app supports various chat templates (ChatML, Llama, Gemma) and models, with users comparing performance of Qwen 2.5 3B (Q5), Gemma 2 2B (Q6), and Danube 3. The developer provided [screenshots](https://preview.redd.it/130oisgjvspd1.png?width=1290&format=png&auto=webp&s=9890aa96eec037b33f6849e).


---

# AI Reddit Recap

## /r/LocalLlama Recap

**Theme 1. Qwen2.5-VL-Instruct excels in visual and video tasks**

- **Qwen/Qwen2.5-VL-3B/7B/72B-Instruct are out!!** ([Score: 489, Comments: 75](https://reddit.com/r/LocalLLaMA/comments/1itq30t/qwenqwen25vl3b7b72binstruct_are_out/)): **Qwen2.5-VL** offers significant enhancements, including improved **visual understanding** for recognizing objects, text, charts, and layouts within images, and **agentic capabilities** that allow it to reason and interact with tools like computers and phones. It also features **long video comprehension** for videos over an hour, **visual localization** with accurate object identification and localization, and **structured output generation** for complex data such as invoices and forms, making it highly applicable in finance and commerce. Links to the models are available on [Hugging Face](https://huggingface.co/Qwen).
  - Users noted the release of **Qwen2.5-VL** and its **AWQ versions**, with some confusion about its timing. **Recoil42** highlighted the potential impact of its **long video comprehension** feature in the video industry, while others discussed the substantial **VRAM requirements** for processing long videos, particularly with the **70B model**.
  - **Benchmark results** for different model sizes and quantizations were shared, including performance metrics like **MMMU_VAL**, **DocVQA_VAL**, and **MathVista_MINI**, showing variations between **BF16** and **AWQ** quantizations. The **3B, 7B, and 72B models** were compared, with **AWQ** generally showing slightly lower performance than **BF16**.
  - Users discussed **compatibility and support** issues, including whether **ollama** or **llama.cpp** support the model, and shared solutions for running the model on different platforms like **MLX on Mac** and **TabbyAPI on Nvidia/Linux**. There was also discussion about the **exl2 format** and its compatibility with newer Nvidia hardware.


**Theme 2. Reverb-7b Outperforms in Open LLM Leaderboards**

- **New AI Model | Ozone AI** ([Score: 164, Comments: 54](https://reddit.com/r/LocalLLaMA/comments/1itr9th/new_ai_model_ozone_ai/)): **Reverb-7b**, the latest AI model from **Ozone AI**, has been released, showcasing significant improvements in 7B model performance. Trained on over **200 million tokens** from **Claude 3.5 Sonnet** and **GPT-4o**, and fine-tuned from **Qwen 2.5 7b**, Reverb-7b surpasses other 7B models on the **Open LLM Leaderboard**, particularly excelling in the **MMLU Pro** dataset with an average accuracy of **0.4006** across various subjects. More details and the model can be found on [Hugging Face](https://huggingface.co/ozone-ai/Reverb-7b), and upcoming models include a 14B version currently under training.
  - **Performance Concerns:** There are concerns about **Reverb-7b's** creative writing capabilities, with users noting it performs poorly in this area despite its high **MMLU Pro** scores, which suggest a focus on STEM subjects rather than diverse word knowledge.
  - **Model Differentiation:** The model is a fine-tune of **Qwen 2.5 7b**, with improvements in intelligence and creative writing over previous versions, as noted by users comparing it to models like **llama 3.1 8B**.
  - **Dataset and Releases:** The dataset remains closed due to profit motives, though there are future plans for openness. **Reverb-7b**'s **GGUF** version was released on **Hugging Face**, and users have converted it to **mlx** format for broader accessibility.


**Theme 3. SmolVLM2: Compact models optimizing video tasks**

- **SmolVLM2: New open-source video models running on your toaster** ([Score: 104, Comments: 15](https://reddit.com/r/LocalLLaMA/comments/1iu2sdk/smolvlm2_new_opensource_video_models_running_on/)): **SmolVLM2** has been released by **Merve from Hugging Face**, offering new open-source vision language models in sizes **256M, 500M, and 2.2B**. The release includes zero-day support for **transformers and MLX**, an iPhone app using the 500M model, VLC integration for description segmentation using the 2.2B model, and a video highlights extractor also based on the 2.2B model. More details can be found in their [blog](https://reddit.com/link/1iu2sdk/video/fzmniv61obke1/player).
  - **Zero-shot vision** is explained as the capability of a vision model to perform tasks without direct training for those specific tasks, by leveraging general knowledge. An example given is classifying images for new labels specified at test time.
  - Users express appreciation for **Hugging Face's** work on small models, noting the impressive performance of **SmolVLM2** despite its compact size. The model's integration and utility in various applications are highlighted as significant achievements.
  - **Merve** provides links to the [blog](https://huggingface.co/blog/smolvlm2) and [collection of checkpoints and demos](https://huggingface.co/collections/HuggingFaceTB/smolvlm2-smallest-video-lm-ever-67ab6b5e84bf8aaa60cb17c7) for **SmolVLM2**, facilitating further exploration and use of the model.


**Theme 4. Open-source AI agents tackling new frontiers**

- **[Agent using Canva. Things are getting wild now...](https://v.redd.it/hjbttwq4r9ke1)** ([Score: 125, Comments: 47](https://reddit.com/r/LocalLLaMA/comments/1itv9ia/agent_using_canva_things_are_getting_wild_now/)): The post discusses an **AI agent** using **Canva** and potentially bypassing **CAPTCHAs**, indicating advanced capabilities in automating tasks that typically require human interaction. The absence of a detailed post body suggests reliance on the accompanying video for further context.
  - The **AI agent** showcased in the post has the capability to bypass **CAPTCHAs**, though skepticism remains about the authenticity of such demos, with advice to verify by personal use. The project is open-sourced and available on [GitHub](https://github.com/Aident-AI/open-cuak).
  - There is interest in the agent's compatibility with other **multimodal models** beyond **OpenAI**, with confirmation that it can work with other open-source models, although performance may vary. Running costs can be managed by renting a GPU for approximately **$1.5 per hour**.
  - The setup for using **Canva** with the AI requires detailed instructions, indicating a trial-and-error process. Concerns about the agent's adaptability to interface changes were raised, highlighting the need for precise control details in prompts or a knowledge base.


## Other AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding

**Theme 1. Multi-modal AI Systems: Bridging Text and Vision**

- **["Actually.. on a second thought" ahh AI](https://i.redd.it/ekotu6u4z7ke1.jpeg)** ([Score: 103, Comments: 44](https://reddit.com/r/ChatGPT/comments/1itpt8c/actually_on_a_second_thought_ahh_ai/)): The post discusses a common error in **AI's understanding of numerical data**, specifically how AI can misinterpret decimal numbers. The example given shows a comparison between **9.11 and 9.9**, illustrating that **9.9 is larger** because **0.90 is greater than 0.11**, emphasizing the importance of correctly parsing decimal components.
  - **Human-like Confusion**: The discussion highlights that the initial confusion in AI's interpretation of numbers is similar to how humans might misinterpret at first glance, but humans can quickly analyze and correct their understanding.
  - **AI's Self-Correction**: Users noted instances where AI, like **ChatGPT**, acknowledges its mistakes midway through responses, similar to human behavior when realizing an error.
  - **Humor in Misinterpretation**: Comments humorously compare the numerical misinterpretation to other contexts, such as physical size or dates, and joke about AI's tendency to cover up mistakes like humans do.


---

# AI Discord Recap

> A summary of Summaries of Summaries by o1-preview-2024-09-12

**Theme 1. Grok 3 Steals the Spotlight from OpenAI**

- [**Grok 3 Crushes Coding Tasks ChatGPT Can't Handle**](https://grok.com): Users report **Grok 3** solves complex coding problems that **ChatGPT Pro** struggles with, prompting many to consider switching to **SuperGrok**.
- [**SuperGrok Offers Premium AI at a Bargain Price**](https://grok.com/?show_subscribe=1): At **$30/month**, **SuperGrok** is seen as a better value than **ChatGPT Pro's** **$250/month** subscription, leading users to reevaluate their AI service choices.
- [**Grok 3 Becomes the Community's New 'Bestie'**](https://x.com/Yuchenj_UW/status/1892634804786757712): Enthusiastic users call **Grok 3** their "*bestie*" due to its performance, speed, and user-friendly interface, with many praising its unlimited API and upcoming features.

**Theme 2. Unsloth's GRPO Algorithm Slashes VRAM Requirements**

- [**Train GRPO Models with Just 5GB VRAM—No Magic Required!**](https://unsloth.ai/blog/grpo): **Unsloth** releases new algorithms enabling **10x longer context lengths** and **90% less VRAM**, allowing training with only **5GB VRAM** without accuracy loss.
- [**Community Cheers Unsloth's VRAM-Saving Breakthrough**](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb): Users express excitement and gratitude, sharing improvements while using [Unsloth's Google Colab notebooks](https://colab.research.google.com/github/unslothai/notebooks/) for their projects.
- [**Llama 3.1 Training Gets 90% VRAM Reduction**](https://x.com/UnslothAI/status/1892640995847901684): **Unsloth's** GRPO algorithm reduces **Llama 3.1** VRAM requirements from **510.8GB** to **54.3GB**, inspired by **Horace He's** gradient checkpointing techniques.

**Theme 3. AI CUDA Engineer's Wild Speedup Claims Raise Eyebrows**

- [**'AI CUDA Engineer' Claims 100x Speedup, Engineers Cry Foul**](http://sakana.ai/ai-cuda-engineer/): **Sakana AI** launches an AI system boasting **10-100x speedups** in CUDA kernel optimization, but skeptics point out flawed baselines and fundamental bugs.
- [**'NOP Kernels' Win the Race—But Do Nothing!**](https://x.com/main_horse/status/1892473238036631908): Members uncover that some kernels achieve speedups by effectively doing nothing, highlighting instances of *reward hacking* and questioning the system's validity.
- [**Overhyped AI Kernels Get Roasted by the Community**](https://x.com/BingXu_/status/1892405811596710392): Experts debunk the impressive speedups, revealing errors like memory reuse and incorrect evaluations; the AI isn't ready to replace human CUDA engineers yet.

**Theme 4. Microsoft's Quantum Leap with Majorana 1 Meets Skepticism**

- [**Microsoft Promises Million-Qubit Future with Majorana 1 Chip**](https://azure.microsoft.com/en-us/blog/quantum/2025/02/19/microsoft-unveils-majorana-1-the-worlds-first-quantum-processor-powered-by-topological-qubits/): Microsoft unveils the world's first quantum processor powered by topological qubits, aiming for scalability to **one million qubits**.
- [**Topological Qubits Explained—or Are They?**](https://www.youtube.com/watch?v=wSHmygPQukQ): In a [YouTube video](https://www.youtube.com/watch?v=wSHmygPQukQ), Microsoft's team discusses topological qubits, but some remain skeptical about their practical applications requiring *helium fridges*.
- [**Nadella Hypes Quantum, Users Groan Over Teams**](https://youtu.be/4GLSzuYXh6w): While **Satya Nadella** promotes Microsoft's quantum breakthroughs, users express frustration with existing products like **Teams** and **Copilot**, questioning Microsoft's focus on innovation over product quality.

**Theme 5. AI Companies Bag Big Bucks, Betting on Inference Boom**

- [**Lambda Lands $480M to Power the AI Cloud**](https://x.com/stephenbalaban/status/1892275552171737220): **Lambda** announces a **$480 million Series D** to bolster its AI computing resources, aiming to be the go-to **cloud service tailored for AI**.
- [**Arize AI Raises $70M to Perfect AI Evaluation**](https://arize.com/blog/arize-ai-raises-70m-series-c-to-build-the-gold-standard-for-ai-evaluation-observability/): **Arize AI** secures funding to advance AI evaluation and observability, ensuring **AI agents** operate reliably at scale.
- [**Baseten and Together Compute Bet Big on 2025 Inference Boom**](https://x.com/basetenco/status/1892259130540179863): **Baseten** raises **$75M** and **Together Compute** bags **$305M**, both gearing up for what they see as a pivotal year for **AI inference technologies**.


---

# PART 1: High level Discord summaries




## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Grok 3 Outshines OpenAI Models**: **Grok 3** is showing superior performance compared to **OpenAI's models**, particularly in benchmarks and resolving coding tasks that **ChatGPT Pro** struggles with.
   - Users express increased confidence in **Grok 3's capabilities**, reporting it solves complex problems that **o1 Pro** cannot, and are considering switching to **SuperGrok**.
- **SuperGrok Offers Better Subscription Value**: At $30 USD per month, **SuperGrok** is seen as offering better value compared to **ChatGPT Pro’s** $250 USD subscription.
   - Users perceive **SuperGrok** as having advantages in terms of performance and usage limits, causing many to reevaluate their AI service subscriptions.
- **Grok's Voice Mode Anticipation**: Community members are anticipating upcoming features for **Grok**, such as voice mode and custom instructions, believing they will further enhance its utility and competitiveness.
   - The **Grok 3** model's API is noted for its unlimited capabilities, allowing for extensive interactions without the strict limits seen in some other models.  They are actively seeking more integrations.
- **Propose saving Chat URLs to return to valuable discussions**: One member proposed saving the **URL of the chat** to easily return to valuable discussions, encouraging others to share their ideas in the designated channel for **OpenAI** to see them.
   - They also recommended using keywords like *'good1'* or *'Track this chat'* to help remember significant chats.
- **Prompt Engineering Troubleshooting Anticipated**: A member expressed eagerness for a call to determine if the issues are due to **prompt** or the **software** malfunctioning, which *is taking too much time than expected*.
   - The same member thanked others for their **helpful advice**, stating they will keep the insights in mind for future reference, but needing *something else* for a particular case.



---



## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **DeepSeek-V3 Grants Unlimited Access!**: **DeepSeek-V3** is now unlimited for **Windsurf Pro** and **Ultimate** plan users, providing unrestricted access with **0 prompt credits** and **0 flow action credits**.
   - Windsurf encouraged users to check [this tweet](https://x.com/windsurf_ai/status/1892322088507105561) to see more about this change.
- **MCP Use Cases Spark Excitement**: Matt Li shared **MCP** content, encouraging users to explore its potential on [X](https://x.com/windsurf_ai/status/1892394489588727985), highlighting the community's desire for engagement.
   - A quick demo illustrates how **MCP** can work within **Cascade**, serving as a resource for those still exploring its capabilities.
- **Codeium Plugin Faces EOL Speculation**: Users voiced concerns about the **JetBrains Codeium plugin** potentially being unsupported, expressing frustration over its perceived lack of direction.
   - One user lamented, *It's a shame to see Codeium as a plugin be abandoned.*
- **Cascade's Memory System Needs Love**: Users are encouraged to use commands such as '*add to memory*' and '*update memory*' to help **Cascade** remember project details, while the proposed structure of global rules into separate files aims to improve Cascade's performance.
   - There has been discussion on the strengths of **DeepSeek v3** versus **Cascade Base**.
- **Windsurf Users Await Support**: Users report delays in receiving responses to support tickets, including the lack of auto-replies with expected ticket numbers in the subject line.
   - Confusion persists over the correct email source for support communications.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth Unleashes Long Context GRPO**: Unsloth has released **Long Context GRPO**, enabling the training of reasoning models with just **5GB VRAM**, promising **10x** longer context lengths and **90%** less VRAM usage, as noted in [this Tweet](https://x.com/UnslothAI/status/1892640995847901684).
   - Users expressed excitement and shared their improvements while gratefully acknowledging Unsloth for providing free resources, such as [this Google Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb).
- **Training Loss Swings Cause Concern**: Users have observed significant fluctuations in **training loss** during model training, which often stabilizes only after several epochs, with users making adjustments using [this Google Colab](https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing).
   - The community recommended adjusting the **learning rate** and maintaining clarity in training prompts to reduce **overfitting** and enhance learning outcomes, which is also mentioned in the [Unsloth Documentation](https://docs.unsloth.ai/basics/reasoning-grpo-and-rl).
- **5090 Mobile Specs Spark Upgrade Fantasies**: The **RTX 5090 Mobile** will feature **24GB** of memory, and preorders are anticipated to begin next week.
   - The announcement has stirred interest among community members who are actively contemplating hardware upgrades.
- **Nuances of RAG vs Fine-tuning Revealed**: A [YouTube video titled "RAG vs. Fine Tuning (Live demo)"](https://www.youtube.com/watch?v=LDMFL3bjpho) was shared which examines if **fine tuning** yields better results than traditional **RAG** systems.
   - Viewers requested additional examples comparing **RAG** and **fine tuning**, hinting at a demand for more comprehensive insights in future demos; the creator indicated plans for a follow-up video detailing how to get started with **Kolo**.
- **Triton's Custom Assembly Works Wonders**: Clarification was provided on what **custom_asm_works** refers to in the context of a challenge scoring system, explaining that it involves **inline assembly** in Triton, allowing for execution over a tensor without **CUDA**, as detailed in [Triton documentation](https://triton-lang.org/main/python-api/generated/triton.language.inline_asm_elementwise.html).
   - This is being used as a technique to improve cohesion timing concerns for hardware and is a focus of current work.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Hunyuan Image Gen Demands VRAM**: The **Hunyuan** model for image generation is now available but requires at least **24GB of VRAM** and works primarily on **NVIDIA** cards, taking several minutes to generate video content.
   - Users are keen to test **Hunyuan's** capabilities against other platforms.
- **A100 GPUs for AI Tasks**: Users discussed the utility of **A100 GPUs** with **LM Studio**, highlighting their **80GB VRAM** capacity for AI tasks.
   - Despite the potential costs, there's significant interest in acquiring **A100s** to boost performance.
- **AMD Ryzen AI Max+ CPU Rivals RTX 4090**: The **Ryzen AI Max+** specs have garnered interest, with claims they *beat Nvidia RTX 4090 at LLM Work* as seen in [this article](https://www.club386.com/amd-ryzen-ai-max-cpus-beat-nvidia-rtx-4090-at-llm-performance/).
   - Skepticism remains about their real-world performance compared to existing GPUs, pending independent benchmarks.
- **Apple Silicon Criticized for Soldered Components**: Discussions around **Apple's** soldering of components in laptops, limiting repairability and upgrades. Discussion includes concern that integrated design trends limit memory configuration flexibility.
   - Users voice a preference for systems allowing component upgrades.
- **Speculative Decoding Dives**: Speculative decoding with certain models may yield lower token acceptance rates and slower performance, according to user feedback.
   - Users shared experiences with token acceptance and asked about optimal model setups for maximizing performance.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Grok 3 Takes the Lead**: Users are finding **Grok 3** performs faster than **GPT-4o**, and some are canceling other subscriptions for it, calling Grok 3 their *'bestie'* due to its performance, cheaper pricing, and user-friendly UI, according to [this X post](https://x.com/Yuchenj_UW/status/1892634804786757712).
   - Notably, **Grok 3** is available for free (until their servers melt), per [xAI's tweet](https://x.com/xai/status/1892400129719611567), with increased access for Premium+ and SuperGrok users.
- **Aider Faces Linux Argument Size Constraints**: A user reported difficulty passing many files into **Aider** due to Linux argument size constraints, particularly with deeply nested directory paths.
   - They suggested using a text file with `/load` commands as a workaround, while noting the repo contains many small files, the length of the nested directory paths is a significant issue.
- **SambaNova Claims DeepSeek-R1 Efficiency Crown**: **SambaNova** announced serving **DeepSeek-R1** with significant speed and cost reductions compared to existing models, achieving *198 tokens per second*, according to [their press release](https://sambanova.ai/press/fastest-deepseek-r1-671b-with-highest-efficiency).
   - The claim positions **DeepSeek-R1** as highly efficient, making significant strides in AI model application and implementation, per a [Kotlin blog post](https://blog.jetbrains.com/kotlin/2025/02/openai-vs-deepseek-which-ai-understands-kotlin-better/).
- **Aider Font Colors Spark Visibility Debate**: Users raised concerns about the font color visibility in **Aider**, especially the blue color in light mode.
   - Suggestions included checking dark mode settings and ensuring proper configurations to address the visibility problem.
- **RAG Setup Superior Than AI Chat**: A member stated the current **RAG** setup yields better results than the **AI Chat** RAG feature for their coding needs.
   - Another member agreed, noting that normal **RAG** struggles with code and improvements are necessary.



---



## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **Cursor IDE Sparking Debate**: Users reported issues with **Cursor's Sonnet 3.5 performance**, expressing frustration over reliability compared to previous versions.
   - In contrast, **Grok 3** received praise for its speed and effectiveness in problem-solving during coding tasks, though some criticized its owner and past performance, along with its lack of API access; see [Grok 3 is an...interesting model](https://youtu.be/WVpaBTqm-Zo).
- **MCP Servers Create Headaches**: Users discussed the complications surrounding the setup and functionality of **MCP servers** within Cursor, with some finding it challenging to utilize effectively; check out [Perplexity Chat MCP Server | Smithery](https://smithery.ai/server/@daniel-lxs/mcp-perplexity).
   - Community members suggested that improved documentation could enhance the user experience and streamline installation, noting that the *MCP config is OSX and Linux specific*, see [issue #9 · anaisbetts/mcp-installer](https://github.com/anaisbetts/mcp-installer/issues/9).
- **AI Model Performance Questioned**: Participants expressed dissatisfaction with the current performance of **AI models**, notably Claude, attributing inconsistencies in output to underlying prompting and context management issues.
   - Variations in responses from LLMs are expected, highlighting the stochastic nature of these models, but some are hoping for better performance from Grok-3 and the new **DeepSeek-V3** available in Windsurf Pro and Ultimate plans, see [Tweet from Windsurf (@windsurf_ai)](https://x.com/windsurf_ai/status/1892322088507105561?s=46&t=ggmESCIXF0nYw8_kshHz7A).
- **Developer tools trigger frustrations**: Users reported challenges using the **Cursor Tab**, with some stating it introduced bugs during development that slowed workflows.
   - The **Cursor Composer** was praised for generating stronger and more reliable code, but overall developers are looking forward to the next generation of Amazon and Anthropic models powered by the **Rainier AI compute cluster**, see [Amazon announces new ‘Rainier’ AI compute cluster with Anthropic](https://www.semafor.com/article/12/03/2024/amazon-announces-new-rainier-ai-compute-cluster-with-anthropic).



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Hugging Face Hardcover Hits the Shelves**: Excitement is building around the release of a new Hugging Face-themed hardcover book, marking a year of teamwork celebrated in [a recent blog post](https://x.com/Nouamanetazi/status/1892274582503248178).
   - Those interested should act fast to *secure a copy*.
- **Qwen2.5 Achieves Training Breakthrough**: Leveraging Unsloth's new algorithms, users can now train reasoning models with just **5GB of VRAM** for Qwen2.5, achieving **10x longer context lengths** and **90% less VRAM**, showcased in [this blog](https://unsloth.ai/blog/grpo).
   - These improvements provide practical tools for developers.
- **HF Spaces Hosts Fast Video Generators**: Discussion highlights the availability of video generators on HF Spaces, with *ltxv* noted as a standout for its speed, generating videos in just **10-15 seconds**.
   - There is a new plan for collaborations to create a video generator based on the latest releases.
- **CommentRescueAI Speeds Up Python Doc Generation**: **CommentRescueAI**, a tool that adds AI-generated docstrings and comments to Python code with a single click, is now available on the VS Code extension marketplace.
   - The developer is seeking community input on ideas for improvement.
- **Lumina2 Gets Fine-Tuned with LoRA**: A new fine-tuning script for **Lumina2** using **LoRA** is now available, enhancing capabilities for users under the **Apache2.0** license, with more information in the [documentation](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_lumina2.md).
   - This promotes open collaboration on AI technology.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity AI Users Battle Glitches**: Users report frustrating experiences with the **Perplexity AI** app, citing lag, high resource consumption, and glitches during text generation, but the developers may be [working on it](https://www.cplx.app/).
   - Concerns have specifically been raised about the model's performance, prompting inquiries into whether the development team is actively addressing these ongoing issues.
- **Grok 3 Hallucinates Wildly, say Users**: Discussion around **Grok 3** revealed mixed feelings; some users feel it performs better than previous models, while others noted significant hallucinatory behavior.
   - Users compared **Grok 3** to **Claude** and **O3** combinations, generally preferring **Claude** for more reliable performance.
- **Mexico vs. Google Gulf Faceoff**: In a bold move, **Mexico** has threatened **Google** regarding their operations near the Gulf, highlighting ongoing jurisdictional disputes.
   - This conflict underscores the growing tension between tech companies and national regulators over the use of machine learning.
- **Sonar API Struggles Stir Concerns**: A user raised concerns over the **Sonar API's performance**, finding it to yield worse results than older models like **llama-3.1-sonar-large-128k-online**.
   - This user reported that the legacy models perform better for tasks like fetching website information, expressing disappointment over the perceived decline in quality despite similar pricing.
- **Deep Research API Rumored Soon**: Members are inquiring about the potential for **deep research capabilities** to be integrated into the API, which could lead to exciting new functionalities.
   - One user expressed enthusiasm, thanking the Perplexity team for their ongoing work in this area.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Saudi Arabia Launches ALLaM**: The **Saudi Arabia**-backed [ALLaM](https://arxiv.org/html/2407.15390v1) focuses on creating Arabic language models to support the ecosystem of Arabic Language Technologies, which represents a push for LLMs in the current geopolitical climate.
   - The model can generate both Arabic and English text and has 70B parameters.
- **Mercor raises $100M for AI Recruiting**: [Mercor](https://techcrunch.com/2025/02/20/mercor-an-ai-recruiting-startup-founded-by-21-year-olds-raises-100m-at-2b-valuation/) raises $100 million for its AI recruiting platform, founded by young Thiel Fellows, highlighting its rapid growth and a valuation jump to $2 billion.
   - Discussions centered on Mercor's innovative marketing drive amidst the competitive AI landscape.
- **Innovative GRPO Algorithm reduces VRAM**: Unsloth released a new **GRPO algorithm** that reduces VRAM requirements for Qwen2.5 training to just **5GB**, marking a significant improvement.
   - The algorithm enables **10x longer context lengths**, offering streamlined setups that could revolutionize model training efficiency.
- **Nadella promotes Microsoft, but product quality is questionable**: In a recent [YouTube video](https://youtu.be/4GLSzuYXh6w), **Satya Nadella** shares his skepticism about AGI while promoting economic growth and Microsoft's **topological qubit breakthrough**.
   - Members expressed frustration, questioning how **Satya Nadella** can be viewed positively when **Microsoft products** like Teams and Copilot fall short.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Reasoning Tokens Ruffle Feathers**: Users expressed dissatisfaction with **low max_tokens** in OpenRouter's implementation leading to empty or null responses when **include_reasoning** defaults to false.
   - Proposed changes include setting **include_reasoning** to true by default and ensuring content is always a string, avoiding null values to improve response consistency, with community input being gathered via a poll.
- **Weaver Extension Weaves Versatile Options**: The **Weaver** Chrome extension provides highly configurable options like PDF support, cloud sync with **Supabase**, and direct API calls from the browser.
   - While currently free and hosted on Vercel's free plan, it may face accessibility limitations due to usage limits, with no backend data logging.
- **API Translator Turns Open Source**: A user shared a newly developed **open-source Chrome extension** available via [GitHub](https://github.com/amirrezasalimi/aify) that allows users to transform any content into their preferred style.
   - The tool only requires an **OpenAI-compatible API** to function.
- **Gemini Output Glitches Generate Gripes**: Users reported issues with the **Gemini 2.0 Flash** model's structured outputs, noting discrepancies compared to OpenAI's models when integrating with OpenRouter.
   - Feedback suggests a need for clearer UI indications regarding model capabilities, especially concerning input types and error messages.
- **DeepSeek's Performance Dips Alarmingly**: Some users reported that **DeepSeek** models yield high-quality responses initially, but later responses deteriorated significantly within OpenRouter.
   - Discussions addressed possible causes and mitigation strategies for the decline in response quality.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Grok3 Benchmarks Get Questioned**: Doubts emerged around **Grok3's** performance and benchmarking, as members allege that xAI might have obfuscated data regarding cons@64 usage.
   - Skeptics challenged claims of **Grok3** outperforming state-of-the-art models and shared specific counterexamples.
- **EAs for Neural Net Optimization?**: The community debated using **evolutionary algorithms** for optimizing **neural networks**, considering slower convergence rates at scale due to high dimensionality.
   - Members discussed using GAs for specific training pipeline components to improve model performance, contrasting this with traditional backpropagation.
- **Coding Datasets Shared**: Members shared coding datasets on **Hugging Face**, suggesting their use in augmenting existing models.
   - The conversation underscored the importance of dataset quality and the possibility of reworking existing datasets with advanced reasoning models, such as [NovaSky-AI/Sky-T1_data_17k](https://huggingface.co/datasets/NovaSky-AI/Sky-T1_data_17k).
- **Agents Team Up to Refine**: A member inquired about research on **agents collaborating** to refine ideas towards a goal, focusing on communication and methodologies.
   - The conversation included references to personal experiments where agents discussed and refined processes to achieve specific outcomes, towards goal refinement.
- **Equilibrium Propagation > Backprop?**: The community explored **equilibrium propagation** as an alternative to backpropagation for training energy-based models, highlighting its ability to nudge predictions towards minimal error configurations as shown in [Equilibrium Propagation: Bridging the Gap Between Energy-Based Models and Backpropagation](https://arxiv.org/abs/1602.05179).
   - Discussions covered the parallels between equilibrium propagation and recurrent backpropagation, emphasizing potential applications in neural network training techniques, as discussed in [Equivalence of Equilibrium Propagation and Recurrent Backpropagation](https://arxiv.org/abs/1711.08416).



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Logits Outperform Probabilities for Training**: Discussions highlighted that **logits** are more informative than normalized probabilities, suggesting unnecessary normalization may impede optimization.
   - The consensus was that while probabilities are essential for decision-making, leveraging **logit space** could optimize training efficiency for specific models.
- **Sparse Attention Gains Traction**: Participants explored **DeepSeek's** paper on *Native Sparse Attention*, noting implications for both efficiency and enhanced contextual understanding.
   - They appreciated **DeepSeek's** high research standards and ability to make findings accessible.
- **Microsoft Enters Topological Qubit Arena**: Microsoft introduced the **Majorana 1**, the first QPU utilizing topological qubits, aiming for scalability up to one million qubits, as reported on [Microsoft Azure Quantum Blog](https://azure.microsoft.com/en-us/blog/quantum/2025/02/19/microsoft-unveils-majorana-1-the-worlds-first-quantum-processor-powered-by-topological-qubits/).
   - A [YouTube video](https://www.youtube.com/watch?v=wSHmygPQukQ) featuring the Microsoft team explains the significance of **topological qubits** and their potential to redefine quantum computing.
- **Perplexity Breaches Censorship Barriers**: Perplexity AI launched **R1 1776**, designed to bypass Chinese censorship in the Deepseek R1 model, employing specialized post-training techniques, according to [The Decoder](https://the-decoder.com/perplexity-ai-removes-chinese-censorship-from-deepseek-r1/).
   - This development showcases the increasing role of **AI** in navigating and overcoming regulatory restrictions.
- **Google Launches PaliGemma 2: A Visionary Leap**: Google unveiled **PaliGemma 2 mix checkpoints**, an enhanced vision-language model, available in various pre-trained sizes, documented in their [blog post](https://developers.googleblog.com/en/introducing-paligemma-2-mix/?linkId=13028688).
   - Engineered for fine-tuning across diverse tasks, this model excels in areas like image segmentation and scientific question answering.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Sakana AI's AI CUDA Engineer Automates Optimization**: The [AI CUDA Engineer](http://sakana.ai/ai-cuda-engineer/) automates the production of highly optimized CUDA kernels, claiming **10-100x speedup** over common machine learning operations in PyTorch.
   - The system also releases a dataset of **over 17,000 verified CUDA kernels** and a paper detailing its capabilities, though some users feel the paper may be *overhyped* due to weak baselines.
- **Unsloth unveils 10x context and 90% VRAM savings**: Unsloth announced new algorithms enabling training with just **5GB VRAM** for **Qwen2.5-1.5B** models, achieving a **90% reduction** in VRAM usage, detailed in their [blog](https://unsloth.ai/blog/grpo).
   - Comparative benchmarks show that a standard GRPO QLoRA setup for **Llama 3.1** at 20K context previously required **510.8GB VRAM**, now reduced to **54.3GB** by leveraging a previous **gradient checkpointing algorithm** inspired by **Horace He**'s implementation.
- **RTX 5080+ Faces Triton Compatibility Issues**: A member shared their experience running **RTX 5080+** on Triton with **TorchRL**, highlighting errors related to `torch.compile` triggering Triton issues, ultimately resolved by removing the **PyTorch-triton** installation.
   - This brought attention to the compatibility concerns that remain with Triton and PyTorch interactions.
- **Raw-Dogged Tensors Yield Permutation Victory**: A member proposed a new nomenclature called a **raw-dogged Tensor**, aimed at aligning storage format with **MMA_Atom** thread layout, noting a significant reduction in permutation complexity.
   - Another member confirmed using this approach for **int8 matmul**, emphasizing its necessity to avoid shared-memory bank conflicts.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Stable Diffusion Edges Out Flux**: Members find **Stable Diffusion (SD)** more refined than **Flux**, though they acknowledged that **Flux** is still under active development.
   - One member suggested comparing example images to see which model matches personal taste.
- **ControlNet Tames Image Poses**: **ControlNet** uses depth maps or wireframes to generate images from poses, handling adjustments like *'hand in front'* or *'hand behind'*, for creative control.
   - Members pointed out control methods enable precise image generation from poses.
- **DIY Custom Models**: A user inquired about hiring an artist skilled in both **Stable Diffusion** and art to create a custom model and prompt style, raising questions about practicality.
   - The community suggested that learning to create the model is more beneficial and cost-effective in the long run.
- **From Scribbles to AI Images**: One user shared their workflow of using sketches on an iPad to guide AI image generation, seeking advice on refining scribbles into finished images.
   - The user found *img2img* useful, but wanted to find out ways to start from simple doodles.
- **Nvidia GPUs Still King for Image Generation**: **Nvidia GPUs** are the recommended choice for running **Stable Diffusion** smoothly, while **AMD** options may have performance issues.
   - Users shared GPU setups and discussed model compatibility with GPU capabilities.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **AI CUDA Engineer Generates Skepticism**: The [AI CUDA Engineer](http://sakana.ai/ai-cuda-engineer/) is an AI system claiming **10-100x speedup** in CUDA kernel production, but doubts arose about the accuracy of its evaluation and prior misrepresentations in similar projects.
   - Critiques highlighted that a purported **150x speedup kernel** had memory reuse and fundamental **bugs**, leading to skepticism about the reliability of generated kernels.
- **Community Debates LLM Compiler Viability**: Members speculated on whether an **LLM-compiler** could translate high-level PyTorch code into optimized machine code, sparking an engaging conversation.
   - While intriguing, a consensus emerged that substantial challenges, particularly the lack of a common instruction set, could impede progress.
- **Clockwork RNN Architecture is Back**: The discussion around the **Clockwork RNN**, a revised architecture using separate modules for input granularities, gained traction.
   - Members debated the viability of such architectures in future models, including the application of dilated convolutions and attention mechanisms.
- **NeoX vs NeMo in Llama 3.2 TPS**: A comparison of the **Llama 3.2 1B configuration** across NeMo and NeoX revealed **21.3K TPS** for NeoX versus **25-26K TPS** for NeMo, with the [configuration file available](https://github.com/aflah02/gpt-neox/blob/olmo-support/configs/hubble/Speed_Exps/1_1B_Baseline_BS_8_GAS_8_No_Activation_Checkpointing_GQA_Llama_3_2_Fusions_All_4_FA_Swiglu.yml).
   - The member shared the **[WandB run](https://wandb.ai/aflah/hubble-speed-testing/runs/nioywj5f?nw=nwuseraflah)** for detailed metrics and others to optimize their setups.



---



## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **Podcast TTS Faces Challenges**: A user reported issues with the **TTS** function in NotebookLM failing to properly read and interpret input prompts for their podcast.
   - The user expressed frustration when the desired tone for their podcast host could not be achieved, despite trying varied prompts.
- **Non-Google User Access Debated**: A member inquired whether users without Google accounts can be invited to access NotebookLM notebooks, similar to **Google Docs**.
   - The discussion highlighted the need for alternative collaboration methods for those not integrated within the Google ecosystem.
- **Tesla Patent Explored via Podcast**: A user analyzed Tesla's Autonomous Driving AI following a patent grant, spotlighting technologies like **Lidar**, **Radar**, and **Ultrasonics**, and discussed it in a podcast.
   - The user provided a **free** article on their Patreon, inviting listeners to explore their findings further.
- **Homeschooling Enhanced with AI Duo**: A user shared their successful experience integrating **NotebookLM** with **Gemini** in their homeschooling approach, which they likened to having skilled assistants.
   - The synergy between the two tools significantly aided in executing teaching efforts, enhancing the learning experience.
- **AI Struggles with Literary Nuance**: Users expressed concerns about **AI's misinterpretations** of literary works, citing instances where character details and narrative nuances were misunderstood.
   - In some cases, the **AI** resisted corrections even when presented with direct evidence, causing conflicts with the original text's integrity.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune roadmap drops for early 2025**: The official **Torchtune roadmap** for H1 2025 has been released on [PyTorch dev-discuss](https://drive.google.com/file/d/1mKENBMrdMzMQQG1kn43Il64qWPB9uP21/view), outlining key directions and projects planned for **Torchtune** during this period.
   - The full set of **PyTorch roadmaps** for various projects is also accessible on [dev-discuss](https://dev-discuss.pytorch.org/t/meta-pytorch-team-2025-h1-roadmaps/2794), showcasing exciting developments and ongoing work across the platform.
- **Packing Causes VRAM to Explode**: Using packing with a dataset at **max_tokens** length significantly increases **VRAM demands**, causing *out-of-memory* errors at **16K sequence lengths**.
   - One user reported memory usage at **30GB** without packing, underscoring the substantial resource implications.
- **Attention Mechanisms Debate Heats Up**: Discussions revolved around the priority of integrating **exotic transformer techniques**, such as *sparse attention* and *attention compression*, to enhance **efficiency in sequence scaling**.
   - Feedback suggested interest exists but integrating new research faces resistance due to established methodologies.
- **AdamWScheduleFree Emerges as Optimizer**: Discussions are underway regarding the potential of **AdamWScheduleFree** as the default optimizer for **llama3.1 8B DPO**, tested across **2 nodes with 16 GPUs**.
   - A workaround involving adjustments to the full-dpo Python script was proposed to address previous issues with fsdp.
- **Hugging Face Drops UltraScale Playbook**: A user shared a [link to the UltraScale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) hosted on Hugging Face, describing it as **refreshing**.
   - The playbook aims to guide users in scaling model usage within a practical framework.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Baseten Bags $75M, Eyes Inference in 2025**: Baseten announced a $75 million Series C funding round, co-led by @IVP and @sparkcapital, pinpointing 2025 as the key year for **AI inference technologies**.
   - The round included new investors such as Dick Costolo and Adam Bain from @01Advisors, underscoring **Baseten's growth** and potential in the **AI infrastructure space**; see the [announcement tweet](https://x.com/basetenco/status/1892259130540179863).
- **Mastra's Agents Open for Business**: The open-source project **Mastra** introduced a JavaScript SDK for constructing **AI agents** on Vercel’s AI SDK, emphasizing integration and ease of use; check out [Mastra's agent documentation](https://mastra.ai/docs/agents/00-overview).
   - Developers are exploring **Mastra agents'** capabilities for tasks like accessing third-party APIs and custom functions, enhancing workflow automation.
- **Arize AI's $70M Bet on Observability**: Arize AI has raised $70 million in Series C funding to advance **AI evaluation and observability** across generative and decision-making models, according to their [Series C announcement](https://arize.com/blog/arize-ai-raises-70m-series-c-to-build-the-gold-standard-for-ai-evaluation-observability/).
   - Their mission is to ensure **AI agents** operate reliably at scale, tackling the challenges emerging from new developments in **AI technology**.
- **Lambda Launches to $480M, Aims for AI Cloud**: Lambda revealed a $480 million Series D funding round led by Andra Capital and SGW, to solidify the company's standing in **AI computing resources**; see the [announcement from stephenbalaban](https://x.com/stephenbalaban/status/1892275552171737220?s=46).
   - The funding will help Lambda enhance its position as a **cloud service tailored for AI**, boosting its capabilities and offerings to meet rising industry demands.
- **OpenAI's User Base Skyrockets**: OpenAI reported over 400 million weekly active users on ChatGPT, marking a 33% increase in less than three months, according to [Brad Lightcap](https://x.com/bradlightcap/status/1892579908179882057?s=46).
   - The anticipated **GPT-5**, promising free unlimited use for all, is expected to consolidate existing models, intensifying competition within the **AI landscape**.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **SSE Implementation Goes Live**: A member confirmed a successful **/sse** implementation for their project, marking an enhancement to **MCP functionality**.
   - Details can be found in the specified channel, highlighting ongoing improvements.
- **Glama Debugging Suffers Cursor Confusion**: A member reported issues debugging **Glama hosted models**, with the cursor failing to locate tools.
   - The problem is primarily attributed to improper use of node paths and potential omissions of necessary quotes, accounting for *99% of the issue*.
- **Docker Installation Confusion Addressed**: A new member needed help with **Puppeteer installation** via a **Docker** build command, leading to clarification on directory navigation.
   - Guidance was given to ensure they were in the correct parent directory and to explain the use of `.` in the command.
- **Python REPL Joins MCP**: A member shared a simple **Python REPL** implementation supporting **STDIO** for MCP and provided the latest image along with [GitHub repository](https://github.com/evalstate/mcp-py-repl) link.
   - Inquiries about **IPython support** were met with optimism for potential addition, opening avenues for further development.
- **Docker Deployment Steps Clarified**: A member shared a [blog post](https://docs.defang.io/blog/2025/02/18/model-context-protocol) on deploying Dockerized **MCP servers**, addressing environment setup challenges across architectures.
   - The post emphasizes **Docker**'s role in ensuring consistency across development environments and offers a list of [reference MCP Servers](https://github.com/modelcontextprotocol/server) for implementation.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **MAX 25.1 Livestream Scheduled**: A livestream is scheduled to discuss **MAX 25.1**, with opportunities to [join on LinkedIn](https://www.linkedin.com/events/introducingmax25-17297704283980902402/theater/) and submit questions through a [Google Form](https://forms.gle/NkjU6e3n15TRtiMA7).
   - Speakers encouraged the community to share their questions, emphasizing eagerness to hear community's insights.
- **Mojo on Windows Unlikely Soon**: Native **Mojo Windows support** isn't on the immediate roadmap due to the expenses of running **AI clusters on Windows**.
   - The consensus is that **nix OSes** are preferred for compute tasks, and many are using cloud **Linux** platforms instead, diminishing the urgency for Windows support.
- **Slab Lists for Memory Efficiency**: A member defined a **slab list** as an efficient data structure, akin to a `LinkedList[InlineArray[T, N]]`, that promotes simplicity and good memory management, and linked to [nickziv/libslablist](https://github.com/nickziv/libslablist).
   - The user noted that this structure can achieve **O(1)** performance for certain operations and offers faster iteration compared to linked lists because of better cache use.
- **Mojo Bridges Python Performance Gap**: It was agreed that **Mojo** is Python-derived but gets performance closer to C/C++/Rust, aiming for future C++-like compatibility with C.
   - The community feels **Mojo’s** type system allows for a **Python-like** experience, attracting users of languages such as **Nim**.
- **Mojo Excels in Low-Level Ease**: A member remarked that handling low-level tasks in **Mojo** is more user-friendly compared to C/C++, suggesting Mojo makes hardware utilization easier.
   - The community suggested that for low-level coding, **Mojo** doesn’t need to strictly follow Python's syntax, because running Python scripts will be sufficient for many uses.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaCloud Launches in EU**: [LlamaCloud EU](https://t.co/HTt2pob88p) launched early access, offering a new SaaS solution with secure knowledge management and full data residency within the EU.
   - The launch aims to remove barriers for European companies needing compliant solutions, emphasizing **security** and **data residency**.
- **LlamaParse Gets Parsing Boost**: [LlamaParse](https://t.co/ux3gsvDIeW) introduced new parsing modes—Fast, Balanced, and Premium—to effectively address diverse document parsing needs.
   - These upgrades enhance versatility in handling different document types to tackle existing **document parsing challenges**.
- **Agents Stuck in Handoff Limbo**: A developer reported issues with an LLM repeatedly returning *'I am handing off to AgentXYZ'* instead of executing tool calls in a multi-agent workflow.
   - Suggestions included incorporating **handoff rules** directly into the **system message** to better clarify expected behavior, but concerns were raised about breaking the existing prompt.
- **Redis Races Rampant?**: A user seeks strategies to effectively run **1000 parallel batches** persisting a summary index, while avoiding race conditions in Redis.
   - With review embeddings stored in a Redis namespace, the user is concerned about potential **key collisions** and **resource constraints**.
- **Scamcoin Shenanigans!**: Discussion of the possibility of creating a coin on **Solana** has led the community to deem such claims as **scams**.
   - Concerns were also raised about the implications of being involved with 'scamcoin' projects more broadly.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Pink Status Gains Traction**: A member updated their status to indicate, *"now I am pink."*
   - This color change likely contributes to the visual dynamics of the Discord community.
- **Identity Sharing Initiative Under Fire**: A user proposed a collaboration opportunity involving identity sharing for profit ranging from **$100-1500**, highlighting an age range of **25-50**.
   - This led to concerns being raised about the implications of identity theft in such arrangements, with no **website** or relevant documentation provided, and sparked debates about being cautious around disclosing **personally identifiable information** in a public forum.
- **Essay on Coffee Absence Requested**: A member requested an essay about the effects of a world without **coffee**, highlighting its cultural and economic significance.
   - This request suggests a curiosity about lifestyle changes in the hypothetical scenario where coffee is no longer available.
- **Communication Clarity Considered Paramount**: Concerns were raised about the ambiguity in written communication, with advice given to use **clearer writing** to prevent misunderstandings.
   - Members emphasized the importance of improving communication to foster **positive collaboration** within the group.



---



## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **Engineers Dive into Jamba API**: Users are actively exploring the **Jamba API**, with one member sharing [code](https://docs.ai21.com/reference) for making API calls and seeking syntax help, while another offered a detailed API usage outline.
   - The comprehensive outline included the headers and necessary parameters, providing practical guidance to other engineers in the channel.
- **Jamba API Outputs Spark Debate**: Concerns arose over the output format of the **Jamba API**, particularly regarding **escape characters** that complicate data processing in different languages.
   - Confirmation was given that response formatting varies by language, necessitating tailored handling methods for outputs.
- **PHP Engineers Tackle Jamba API Integration**: A Symfony and PHP engineer sought advice on converting **Jamba API** responses into usable formats, specifically addressing **special character handling**.
   - Other members pointed to potential peer assistance with **PHP**-specific challenges and effective output handling.
- **AJAX Proposed for Jamba API Enhancement**: One member suggested leveraging **AJAX** to improve **Jamba API** response handling, although results showed inconsistencies.
   - It was noted that the **Jamba chat window** formats outputs differently, influencing how results appear and potentially affecting handling strategies.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Old GeForce struggles against RTX 4070**: Performance tests show an **old GeForce 850M** achieving **3 tok/s** after **8 seconds**, while an **RTX 4070** reaches **12 tok/s** in **1.9 seconds**.
   - However, overall model usability is limited by significant **computational costs and numerical stiffness**.
- **Int8 Quantization Derails Models**: Members noted that **Int8 quantization** may require adjustment as models occasionally go 'off rails' after several hundred tokens when using **Int8Linear**.
   - The suggestion was made that conversations about **tinychat** developments should take place in *direct messages or GitHub* to be more focused.
- **Torch Edges Out Tinygrad on Speed Tests**: Speed tests indicate that **torch** outperforms **tinygrad** on **2048x2048** tensors, with **0.22 ms** for torch compared to **0.42 ms** for tinygrad.
   - However, on **4096x4096** tensors, **tinygrad** is only **1.08x slower** than torch, indicating optimized scaling.
- **BEAM Could Boost Performance**: Increasing **BEAM** values might alleviate performance constraints, with tests showing **0.21 ms** for **2048x2048** tensors with **BEAM=10** in torch.
   - Performance appears consistent across different tensor sizes, highlighting potential gains from higher **BEAM** configurations.
- **New PyTorch Channel Launched**: A new channel dedicated to **PyTorch** discussions has been created.
   - The intent is to encourage more focused and in-depth conversations as user contributions expand.



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **System Message Terminology Causes Confusion**: A member clarified that the term *'system message'* is now used in the UI, indicating a shift in naming conventions.
   - Another participant affirmed that old habits can be difficult to change when navigating these systems.
- **Instructions in System Message: Plain English OK?**: It's mentioned that plain English instructions can be used in the *'system message'*, and most models will respect these commands.
   - Some members expressed skepticism about the ease of this process, questioning if using Jinja or JSON code is more effective.
- **GPT4All Falls Flat on Image Handling**: One member queried about the ability to paste images directly into the text bar like in other AI platforms, but it was clarified that **GPT4All** cannot handle images.
   - External software is recommended for such tasks.
- **Nomic and NOIMC v2: Is it real?**: A member expressed confusion over the implementation of **NOIMC v2**, questioning why it appears to be incorrectly implemented.
   - Another member humorously sought confirmation about being on **Nomic**, showcasing their frustration.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **2024 LLM Agents Course Still Useful**: A member suggested that while not required, auditing the **Fall 2024 Course** from [this YouTube playlist](https://youtube.com/playlist?list=PLS01nW3RtgopsNLeM936V4TNSsvvVglLc&feature=shared) could deepen understanding, especially for **DSPy**.
   - They noted that **DSPy** is absent from the current semester’s syllabus, making the Fall 2024 course particularly useful for those interested in it.
- **Quizzes Archived for LLM Agents Course**: A member shared a link to a **quizzes archive** for the Fall 2024 course, located [here](https://docs.google.com/document/d/1pYvOxt2UWwc3z4QlW2Di5LQT-FJPWZ419mxJT7pFPsU/edit?usp=sharing), responding to confusion over their disappearance from the current syllabus.
   - The quizzes are now accessible to those who started the course late and want to catch up.
- **Navigating Quiz Access on MOOC**: In response to a user seeking **quiz 1 and 2**, it was pointed out that the quizzes can be found on the MOOC’s [page](https://llmagents-learning.org/sp25) or the [announcement page](https://llmagents-learning.org/f24).
   - It was also mentioned that *all certificates have been released* and students were encouraged to sign up for the [Spring 2025 iteration](https://llmagents-learning.org/sp25).
- **Course Completion Notice**: The **LLM Agents MOOC** has completed, but video lectures remain accessible in the syllabus.
   - All certificates have been released, and students are encouraged to sign up for the [Spring 2025 iteration](https://llmagents-learning.org/sp25).



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Qwen/Qwen2.5-VL-7B-Instruct Scores Varying for HaizeLabs Judge Compute**: A member replicated the same dataset as **HaizeLabs Judge Compute** and found that scores with the model **Qwen/Qwen2.5-VL-7B-Instruct** ranged from **60%-70%** for 2-stage optimized to **88.50%** for mipro2.
   - The project titled **LLM-AggreFact_DSPy** has been shared on [GitHub](https://gist.github.com/fullstackwebdev/fa4934fb4669cfc3e8c6ced950ea7a22) with source code related to the evaluation, enabling deeper insights into the methodologies used.
- **Leonard Tang Releases Verdict Library**: Leonard Tang released [Verdict](https://x.com/leonardtang_/status/1892243653071908949), a library targeting judge-time compute scaling, pointing out AI reliability issues stem from evaluation rather than generation.
   - He emphasized that the next advancement for AI should focus on evaluation improvements, contrasting with the emphasis on **pre-training** and **inference-time scaling**.
- **DSPy Conversation History Examined**: A member asked whether DSPy automatically injects conversation history into calls, indicating a caution before more implementation.
   - This highlights concerns about potential complexities in managing AI interactions without unintentionally overwriting previous context, especially in more complex applications.
- **Exporting Prompts to Message Templates Described**: A member shared an FAQ explaining how to freeze and export prompts into message templates by using a Python snippet with `dspy.ChatAdapter()`.
   - It was clarified that this method results in a loss of control flow logic, suggesting `program.save()` or `program.dump_state()` as alternatives for a more comprehensive export.



---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Gorilla LLM (Berkeley Function Calling) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1341820211363582083)** (979 messages🔥🔥🔥): 

> `Grok 3 performance, SuperGrok subscription, Comparison with OpenAI models, Grok's capabilities, Community feedback` 


- **Grok 3 surpasses OpenAI models**: Grok 3 has demonstrated superior performance compared to OpenAI's models, notably excelling in benchmarks and specific coding tasks that ChatGPT Pro struggled with.
   - Users have reported Grok 3 resolving complex problems that o1 Pro could not, leading to increased confidence in its capabilities.
- **SuperGrok offers better value**: Many users are considering switching to SuperGrok because it provides better value for money at $30 USD per month compared to ChatGPT Pro’s $250 USD.
   - SuperGrok is perceived as having significant advantages, particularly in terms of performance and usage limits.
- **Features sought in Grok**: Community members are interested in upcoming features for Grok, such as voice mode and custom instructions, which they believe will further enhance its utility.
   - These features are expected to make Grok more competitive against other models, especially in handling context and usability.
- **Discussion on AI subscription models**: Users discussed the various subscription models available and the limitations of existing services, with Grok 3 being favored due to its better offerings and pricing.
   - The conversation revealed a general sentiment that many are reevaluating their subscriptions to AI services in light of new competitors.
- **Grok's API and capabilities**: The Grok 3 model's API is noted for its unlimited capabilities, allowing for extensive interactions without the strict limits seen in some other models.
   - Users expressed a desire for more integrations and functionalities to maximize the potential of the Grok platform.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://grok.com/share/bGVnYWN5_b5ffc957-9e88-4942-96aa-80372f58d995">Receiving Jupiter Signals with SDRPlay RSP 2 | Shared Grok Conversation</a>: how can I use my SDRPlay RSP 2 to receive signals from Jupiter like JOVE?</li><li><a href="https://x.ai/blog/grok-image-generation-release">Grok Image Generation Release</a>: no description found</li><li><a href="https://grok.com/">Grok</a>: Grok is a free AI assistant designed by xAI to maximize truth and objectivity. Grok offers real-time search, image generation, trend analysis, and more.</li><li><a href="https://grok.com/share/bGVnYWN5_7ddf224c-9606-4e02-a956-22135248bc79">Dipole Antenna for JOVE with SDRPlay RSP2 | Shared Grok Conversation</a>: show me how to create a dipole suitable for a JOVE receiver using a SDRPlay RSP 2</li><li><a href="https://grok.com/share/bGVnYWN5_76a8e85f-5559-4230-8ef4-f7730b83056b">Ladder Rungs Submerged at Low Tide | Shared Grok Conversation</a>: There is a ladder attach to a boat with 10 rungs. At low tide, the water level drops by 60 cm. Each </li><li><a href="https://grok.com/share/bGVnYWN5_ccd48442-c8fb-4d56-ae0f-739cf884de16">Allstar Node Purchase in Australia | Shared Grok Conversation</a>: Where can I buy a fully assembled plug and play Allstar node in Australia?</li><li><a href="https://grok.com/?show_subscribe=1">Grok</a>: Grok is a free AI assistant designed by xAI to maximize truth and objectivity. Grok offers real-time search, image generation, trend analysis, and more.</li><li><a href="https://grok.com/share/bGVnYWN5_67cdd414-63d1-427a-b6ee-54e4d24d738e">HackerNews Top Stories Overview | Shared Grok Conversation</a>: Summarize top results on the front page of HackerNews today. For interesting articles, explore them </li><li><a href="https://grok.com/share/bGVnYWN5_26c250ce-ff40-4328-9385-bd71cbf04f80">Grok 3 Free Plan Limits | Shared Grok Conversation</a>: What are the current limits for Grok 3? I&#x27;m using it on the free plan (Elon Musk tweeted that it&#x27;s t</li><li><a href="https://grok.com/share/bGVnYWN5_4e986ed7-df82-4227-841f-1bdeae4fc961">SuperGrok: AI Subscription or Crypto? | Shared Grok Conversation</a>: Should I buy supergrok?</li><li><a href="https://tenor.com/view/yarp-hot-fuzz-gif-12386003">Yarp Hot Fuzz GIF - Yarp Hot Fuzz - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.youtube.com/watch?v=pEErLop52Jw">DANGEROUS &quot;EMOJI HACK&quot;: AI models susceptible to &#39;trojan horse&#39; emojis...</a>: The latest AI News. Learn about LLMs, Gen AI and get ready for the rollout of AGI. Wes Roth covers the latest happenings in the world of OpenAI, Google, Anth...</li><li><a href="https://en.wikipedia.org/wiki/Catastrophic_interference">Catastrophic interference - Wikipedia</a>: no description found
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1342180418010611772)** (1 messages): 

> `Feature Requests, Chat Tracking Methods` 


- **Encouragement to Share Ideas**: A member suggested posting ideas in the designated channel as it's a great way for **OpenAI** to see them and for others to engage.
   - *Comment and share if you want this feature too!*
- **Chat URL Saving for Future Reference**: One member proposed saving the **URL of the chat** to easily return to valuable discussions.
   - They also recommended using keywords like **'good1'** or *'Track this chat'* to help remember significant chats.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1342241982667427941)** (2 messages): 

> `Software troubleshooting, Insights for improvement` 


- **Anticipation for Troubleshooting Call**: A member expressed eagerness for a call scheduled tomorrow to determine if the issues are due to **prompt** or if the **software** is malfunctioning.
   - They humorously noted that resolving the problem is taking longer than expected, saying *it's taking too much time than I expected*.
- **Gratitude for Helpful Advice**: The same member thanked others for their **helpful advice**, stating they will keep the insights in mind for future reference.
   - However, they feel that in one particular case they may require additional support, expressing uncertainty with *not sure yet what exactly*.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1342241982667427941)** (2 messages): 

> `Prompt issues, Software performance` 


- **Anticipation Builds Over Tomorrow's Call**: A member expressed excitement about an upcoming call, pondering whether the issues faced are due to the **prompt** or the **software** behaving unpredictably.
   - *It’s taking too much time than expected* as they seek clarity on the matter.
- **Grateful for Support, Yet Seeking More**: The same member thanked others for their advice, feeling confident that the insights shared will assist them in the future.
   - However, they mentioned needing *something else* for a particular case, indicating uncertainty about the next steps.


  

---


### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1341881225656139799)** (1 messages): 

> `DeepSeek-V3 Unlimited, Windsurf Pro and Ultimate Plans, Prompt and Flow Action Credits` 


- **DeepSeek-V3 Goes Unlimited!**: DeepSeek-V3 is now unlimited for users on the **Windsurf Pro** and **Ultimate** plans, allowing unrestricted access.
   - This update comes with **0 prompt credits** and **0 flow action credits**, enabling seamless use without limitations.
- **Surfing to New Features**: Users are encouraged to check the announcement through [this tweet](https://x.com/windsurf_ai/status/1892322088507105561) highlighting the new unlimited access.
   - Let's surf into these updates with enthusiasm as Windsurf continues to evolve! <:windsurf:1306309317011570699>



**Link mentioned**: <a href="https://x.com/windsurf_ai/status/1892322088507105561">Tweet from Windsurf (@windsurf_ai)</a>: DeepSeek-V3 is now unlimited in Windsurf Pro and Ultimate plans.0 prompt credits. 0 flow action credits.

  

---


### **Codeium (Windsurf) ▷ #[content](https://discord.com/channels/1027685395649015980/1092566563862884412/1341954445147377685)** (1 messages): 

> `MCP content, Use cases for MCP, MCP in Cascade` 


- **Exciting MCP Content Unveiled**: A member shared a post showcasing **cool use cases for MCP** from Matt Li, encouraging others to check out the content on [X](https://x.com/windsurf_ai/status/1892394489588727985).
   - *Go show some love on the post* ❤️ highlighted the community's desire for engagement with the expanding MCP features.
- **MCP's Potential Use Cases Demonstrated**: The original post included a quick demo illustrating how **MCP** can work within **Cascade**, increasing awareness around its functionality.
   - This demo serves as a resource for those still having questions about MCP, promoting further exploration of its capabilities.



**Link mentioned**: <a href="https://x.com/windsurf_ai/status/1892394489588727985">Tweet from Windsurf (@windsurf_ai)</a>: If you&#39;re still having questions about MCP and its potential use cases, here&#39;s a quick demo on how MCP can work within Cascade!

  

---


### **Codeium (Windsurf) ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1341855441151332434)** (86 messages🔥🔥): 

> `Codeium plugin in JetBrains, Supercomplete feature, Windsurf installation requirements, Comparison of Codeium and CodeBuddy, Concerns about Codeium's support` 


- **Codeium plugin faces EOL speculation**: Concerns were raised about the **JetBrains Codeium plugin** potentially being unsupported as users express frustration over its lack of direction.
   - *It's a shame to see Codeium as a plugin be abandoned,* remarked one user.
- **Supercomplete revolutionizes autocompletion**: The **Supercomplete** feature in Codeium anticipates user actions beyond simple autocomplete, offering relevant edits and context-aware suggestions.
   - Users highlighted its value for refactoring, stating, *This capability is just great for code in a single file.*
- **Windsurf installation mandatory for trials**: To access a trial of the Pro version, users must **register and download Windsurf**, though the free version doesn't require installation.
   - This was clarified amidst queries about whether using the plugin necessitated installing Windsurf.
- **Comparing Codeium with CodeBuddy**: **CodeBuddy** and Codeium are highlighted as top options, with one user expressing interest in trying both before making a decision.
   - Another user noted that while CodeBuddy has more convenient chat functionalities, Codeium's autocomplete currently outperforms.
- **Anticipated API improvements**: Users are eagerly awaiting improvements, such as the addition of Grok 3 support to the Codeium API, expected soon.
   - One member remarked on Grok's strengths in creative problem solving, adding to the discussion of capabilities.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://code.visualstudio.com/blogs/2025/02/12/next-edit-suggestions">Copilot Next Edit Suggestions (preview)</a>: Announcing the Next Edit Suggestions and Agent Mode for GitHub Copilot in Visual Studio Code.</li><li><a href="https://codeium.com/supercomplete">Supercomplete | Windsurf Editor and Codeium extensions</a>: Supercomplete is able to predict your next intent, regardless of your cursor position. Whether you want an insertion, deletion, or edit, Supercomplete has you covered.</li><li><a href="https://codeium.canny.io/feature-requests/p/supercomplete-for-jetbrains">Supercomplete for Jetbrains | Feature Requests | Codeium</a>: I think jetbrains lack the most in the field of &quot;consecutive action proposals&quot;. Supercomplete would be a thing that would be first-of-its-kind in this
</li>
</ul>

</div>
  

---


### **Codeium (Windsurf) ▷ #[windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1341818857756557457)** (546 messages🔥🔥🔥): 

> `Windsurf usability issues, DeepSeek vs Cascade Base, Memory system in Cascade, MCP server configuration, Support response inquiries` 


- **Windsurf experiences frustrations**: Users expressed consistent frustrations with Windsurf's functionality, citing issues like UI errors and unintended changes to code.
   - The general consensus is to work in smaller increments and frequently commit changes to avoid losing progress.
- **Comparative performance of DeepSeek vs Cascade Base**: DeepSeek v3 is viewed as a solid alternative to Cascade Base, but it has limitations when it comes to reliably calling tools without custom instructions.
   - Cascade Base, however, excels in tool calls thanks to its fine-tuning with Llama 3.1 70b.
- **Strategies for using the Memory system in Cascade**: Users are encouraged to utilize commands like 'add to memory' and 'update memory' to ensure Cascade maintains project details.
   - The proposed structure of global rules into separate files aims to enhance organization and improve Cascade's performance.
- **Challenges with MCP server configuration**: Several users encountered issues with their MCP server setups, leading to errors until the configuration files were adjusted or removed.
   - It was suggested that configurations reflecting errors should be relocated to address underlying issues with Windsurf's performance.
- **Queries about support and response times**: Users are experiencing delays in receiving responses from support tickets, with some not receiving any auto replies.
   - The expected response includes ticket numbers in the subject line, but users expressed confusion over the email source for these communications.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://codeium.canny.io/">Codeium Feedback</a>: Give feedback to the Codeium team so we can make more informed product decisions. Powered by Canny.</li><li><a href="https://codeium.com/support">Support | Windsurf Editor and Codeium extensions</a>: Need help? Contact our support team for personalized assistance.</li><li><a href="https://codeium.canny.io/feature-requests/p/placeholder-input-does-not-change-when-changing-windsurf-open-chat-with-cascade">Placeholder input does not change when changing `Windsurf: Open Chat with Cascade` keybind | Feature Requests | Codeium</a>: See attached screenshot</li><li><a href="https://codeium.canny.io/feature-requests/p/devcontainer-support">Devcontainer Support | Feature Requests | Codeium</a>: Would love more devcontainer support Specifically: Rebuild and reopen in container (currently only have Reopen in containerr) Need it to install the extensions</li><li><a href="https://docs.codeium.com/windsurf/mcp>">Welcome to Codeium - Codeium Docs</a>: no description found</li><li><a href="https://tenor.com/view/japanese-tyranno-dance-japanese-dance-tyranno-gif-10262458857606665890">Japanese Tyranno Dance Japanese Dance GIF - Japanese Tyranno dance Japanese dance Tyranno - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://x.com/donvito/status/1892640143145644056">Tweet from Melvin Vivas (@donvito)</a>: Ultimate MCP tutorial 🤯🤯🤯Learn how to configure MCP in Cursor, Windsurf and ClaudeIn this tutorial, we used the github mcp servera thread 🧵👇</li><li><a href="https://glama.ai/mcp/servers/vwi6nt8i80">supabase-mcp</a>: An MCP server that provides tools for interacting with Supabase databases, storage, and edge functions.</li><li><a href="https://x.com/sdrzn/status/1892262424881090721">Tweet from Saoud Rizwan (@sdrzn)</a>: Cline v3.4 is out 🚀 Introducing MCP Marketplace! Discover and install the best MCP servers right in the extension, where Cline handles all the setup. We’ve also added mermaid diagrams to Plan mode, n...</li><li><a href="https://tenor.com/view/mr-bean-mrbean-bean-mr-bean-holiday-mr-bean-holiday-movie-gif-3228235746377647455">Mr Bean Mrbean GIF - Mr bean Mrbean Bean - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/albert-einstein-lol-think-be-smart-think-wise-gif-8735407">Albert Einstein Lol GIF - Albert Einstein Lol Think - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/i-have-a-question-david-crane-frasier-may-i-ask-a-question-i-have-an-inquiry-gif-12327607075242146179">I Have A Question David Crane GIF - I have a question David crane Frasier - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://codeium.canny.io/feature-requests/p/git-awareness">Git awareness | Feature Requests | Codeium</a>: Since git is so fundamental to the software development process, Windsurf should, at all times, be perfectly aware of: git status git diff probably more of any</li><li><a href="https://youtu.be/iBiNfa32AnE?si=0nsiCJAlGa8If-1l">The ONLY Windows PC OPTIMIZATION Guide You Will EVER Need In 2024</a>: THE BEST Quick Guide To Optimizing / Improving Windows on your gaming PC! How To Optimize Windows 10 For GAMING - Best Settings for FPS &amp; NO DELAY! In today&#39;...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1341818452091994143)** (485 messages🔥🔥🔥): 

> `Unsloth AI Models, GRPO Training Updates, Training Loss Issues, Distilled Model Performance, AI Community Insights` 


- **Release of Long Context GRPO**: Unsloth has launched Long Context GRPO, allowing training of reasoning models with just 5GB VRAM, promising 10x longer context lengths and 90% less VRAM usage.
   - The community is excited, with users noting improvements and expressing gratitude for the free resources provided by Unsloth.
- **Training Loss Fluctuations**: Users have observed significant fluctuations in training loss during model training, often stabilizing only after several epochs.
   - Advice given includes adjusting the learning rate and maintaining clarity in training prompts to reduce overfitting and improve learning outcomes.
- **Distilled Model Limitations**: Discussion about the limitations of using distilled models for GRPO training noted that these models may not produce expected output formats without proper adjustments.
   - Users reported mixed experiences with distilled models generating different formats than required, emphasizing the need for a two-stage approach in some cases.
- **Community Engagement and Experimentation**: The community is actively engaged in sharing experiences, tips, and techniques for optimizing model training and output accuracy.
   - Common practices include leveraging structured outputs and refining prompt engineering to enhance model understanding.
- **Challenges in Fine-tuning**: Participants expressed common challenges faced in fine-tuning models, particularly regarding processing complex datasets and maintaining meaningful summaries.
   - Some users suggested utilizing Named Entity Recognition (NER) models to assist in managing company-specific jargon and abbreviations before training.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/nanotron/ultrascale-playbook">The Ultra-Scale Playbook - a Hugging Face Space by nanotron</a>: no description found</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://www.youtube.co">no title found</a>: no description found</li><li><a href="https://docs.unsloth.ai/basics/running-and-saving-models/saving-to-vllm">Saving to VLLM | Unsloth Documentation</a>: Saving models to 16bit for VLLM</li><li><a href="https://colab.research.google.com/drive/1ZF4qWG0CO67j8gm0hoeGiEXXFBPFyF2X?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/overview">AI Mathematical Olympiad - Progress Prize 2</a>: Solve national-level math challenges using artificial intelligence models</li><li><a href="https://docs.vllm.ai/en/latest/">Welcome to vLLM &#8212; vLLM</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=lyVxD0bJDOk">Start Up Wednesday with Unsloth.AI</a>: Meet Daniel and Michael Han, the Australian brothers transforming AI development with Unsloth. Their open-source project makes model fine-tuning 2x faster wh...</li><li><a href="https://x.com/UnslothAI/status/1892640995847901684">Tweet from Unsloth AI (@UnslothAI)</a>: Today, we’re launching new algorithms that enable 10x longer context lengths & 90% less VRAM for training Reasoning Models (GRPO).Using Unsloth, you can now train your own reasoning model with just 5G...</li><li><a href="https://github.com/vllm-project/vllm/issues/13486.">vllm-project/vllm</a>: A high-throughput and memory-efficient inference and serving engine for LLMs - vllm-project/vllm</li><li><a href="https://www.youtube.com/watch?v=bAWV_yrqx4w">[GRPO Explained] DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models</a>: #deepseek #llm #grpoGRPO is one of the core advancements used in Deepseek-R1, but was introduced already last year in this paper that uses a combination of n...</li><li><a href="https://docs.unsloth.ai/basics/reasoning-grpo-and-rl">Reasoning - GRPO &amp; RL | Unsloth Documentation</a>: Train your own DeepSeek-R1 reasoning model with Unsloth using GRPO which is a part of Reinforcement Learning (RL) fine-tuning.</li><li><a href="https://github.com/jingyaogong/minimind/blob/master/README_en.md">minimind/README_en.md at master · jingyaogong/minimind</a>: 🚀🚀 「大模型」2小时完全从0训练26M的小参数GPT！🌏 Train a 26M-parameter GPT from scratch in just 2h! - jingyaogong/minimind</li><li><a href="https://huggingface.co/unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit">unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit · Hugging Face</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: Finetune Llama 3.3, DeepSeek-R1 &amp; Reasoning LLMs 2x faster with 70% less memory! 🦥</a>: Finetune Llama 3.3, DeepSeek-R1 &amp; Reasoning LLMs 2x faster with 70% less memory! 🦥 - unslothai/unsloth
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1341934054785679412)** (17 messages🔥): 

> `Unsloth Art, Quantum Computing, Triton Language in Challenges, Cohesion Timing Hardware, Inline Assembly in Triton` 


- **Unsloth art involves AI and artists**: Discussion revealed that the **3D sloths** are AI-generated while the stickers are created by a **talented artist**.
   - *Great art indeed!* was the consensus among members appreciating the creativity involved.
- **Quantum advancements with Majorana 1 chip**: A YouTube video titled [Majorana 1 Explained](https://youtu.be/wSHmygPQukQ) featured the Microsoft team discussing breakthroughs in **quantum computing** with the new chip.
   - However, it was noted that the technology still requires a **helium fridge** for operation.
- **Clarification on custom_asm_works in Triton**: A member sought clarification on what **custom_asm_works** refers to in the context of a challenge scoring system.
   - It was explained that it involves inline assembly in Triton, allowing for execution over a tensor without using **CUDA**.
- **Cohesion timing concerns for hardware**: There was a mention of curiosity around the **cohesion timing** of the hardware being discussed, particularly in relation to the recent quantum advancements.
   - The exact implications and details were left unexplored but indicated interest in the technical aspects.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://triton-lang.org/main/python-api/generated/triton.language.inline_asm_elementwise.html">triton.language.inline_asm_elementwise &mdash; Triton  documentation</a>: no description found</li><li><a href="https://youtu.be/wSHmygPQukQ?si=4VyaksRGdCXpnNeE">Majorana 1 Explained: The Path to a Million Qubits</a>: Hear from the Microsoft team behind the recent breakthrough in physics and quantum computing demonstrated by the new Majorana 1 chip, engineered from an enti...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1341819278155976876)** (32 messages🔥): 

> `Installing Unsloth, RTX 5090 Mobile Specs, GPU Performance and Fine-tuning, VRAM Usage in Datasets, Qwen2.5 Model Inference Issues` 


- **New Users Seek Unsloth Installation Help**: A new member asked for assistance with installing **Unsloth**, citing their introduction from another Discord server.
   - *Hopeful for guidance*, they engaged with the community to get started quickly.
- **RTX 5090 Mobile Specifications Released**: **RTX 5090 Mobile** will feature **24GB** of memory, with preorders expected to start next week.
   - The reminder of its existence sparked interest among members considering upgrades.
- **Discussion on GPU Performance**: Members shared their **GPU setups**, with one mentioning **3x24GB GPUs** running at **1 token/sec**, while another achieved **3t/s** with **96GB VRAM**.
   - The conversations explored optimization strategies to enhance performance with existing hardware.
- **VRAM Climbing Concerns in Training**: A user inquired about **VRAM usage** rising with uneven dataset lengths, questioning if it was a coincidence or a known issue.
   - Community responses suggested that existing solutions were being tested, with some recommending the **packing** option in the SFTTrainer.
- **Inconsistencies in Qwen2.5 Model Outputs**: After fine-tuning the **Qwen2.5-VL3B model**, a user reported inconsistent outputs when using the **merged model** compared to the standalone **LoRA adapter**.
   - Despite loading the model correctly, confusion remained around the different outputs generated with **vLLM**, prompting further troubleshooting from the community.



**Link mentioned**: <a href="https://docs.unsloth.ai/get-started/fine-tuning-guide#id-7.-running--saving-the-model">Fine-tuning Guide | Unsloth Documentation</a>: Learn all the basics of fine-tuning.

  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1341919319591882855)** (4 messages): 

> `RAG vs Fine Tuning, Video Examples, Kolo Usage, Industry Insights` 


- **Video Explores RAG vs Fine Tuning**: A [YouTube video titled "RAG vs. Fine Tuning (Live demo)"](https://www.youtube.com/watch?v=LDMFL3bjpho) was shared, questioning whether fine tuning delivers better results than traditional RAG systems.
   - *Which is better?* This video aims to challenge prevailing industry thoughts on the effectiveness of the two methods.
- **Desire for More Examples in the Demo**: A viewer expressed a desire for more examples comparing **RAG** and **fine tuning** during the demonstration.
   - *Would it have been possible to show more examples?* The inquiry highlights a need for deeper insights in future iterations.
- **Plans for Future Kolo Video**: The creator responded to feedback, indicating plans for a follow-up video detailing how to get started with **Kolo**.
   - *I will probably make another one later on* that includes comprehensive testing and training data insights.



**Link mentioned**: <a href="https://www.youtube.com/watch?v=LDMFL3bjpho">RAG vs. Fine Tuning (Live demo)</a>: Which is better RAG or Fine tuning? Does the industry have it wrong? Can fine tuning deliver better results than a traditional RAG system? Watch the video fo...

  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1341824741098651700)** (56 messages🔥🔥): 

> `Rigor in Science, Citizen Science, AI in Medicine, Content Moderation Research, Phytochemical Formulations` 


- **Debate on Scientific Rigor**: Members discussed the importance of **rigor** in science, criticizing the use of AI like **ChatGPT** without proper vetting and endorsement for scientific claims.
   - Concerns were raised that merely having a **PhD** or a research title doesn’t guarantee quality, and many believe that this trend diminishes scientific credibility.
- **Citizen Science and Credentials**: A member emphasized that valid research doesn't solely hinge on academic credentials and that **citizen science** plays an important role in knowledge production.
   - The community debated whether a **degree** is necessary to be considered a scientist, with some highlighting exceptions in the field.
- **Challenges in AI for Content Moderation**: Discussions highlighted the challenges in creating AI systems that could objectively moderate content, suggesting solutions like **BERT classifiers** while acknowledging their limitations.
   - One member referenced a paywalled research article related to **content moderation**, emphasizing the need for separating subjective feelings from objective facts.
- **AI-powered Nutraceutical Research**: One member introduced their work on using AI to create targeted nutraceutical formulations for various disorders, sharing links and documents as research material.
   - Despite providing numerous warnings about the need for clinical trials, others expressed concerns about the ethics of presenting such information to the general public.
- **Critique of AI Applications in Medicine**: Several members discussed the potential dangers of oversimplifying medical advice using AI, arguing that it undermines the complexities of medical practice and ethics.
   - Concerns were raised about AI-generated content potentially misleading those with terminal illnesses, but others advocated for progress despite potential misinterpretations.



**Link mentioned**: <a href="https://www.marielandryceo.com/2025/02/title-ai-powered-phytochemical.html?m=1">Title: AI-Powered Phytochemical Formulation: A Data-Driven Approach to Supporting Health</a>: no description found

  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1341816752169947178)** (381 messages🔥🔥): 

> `Hunyuan Image Generation Model, A100 GPU Performance, Speculative Decoding Analysis, LM Studio Features, Embedding Models for Long Texts` 


- **Hunyuan Image Generation Model**: The Hunyuan model for image generation is available, but requires at least 24GB of VRAM and primarily works on NVIDIA cards, taking a few minutes to generate video content.
   - Users have expressed interest in experimenting with this model, especially regarding its capabilities compared to other platforms.
- **A100 GPU Performance**: Users discussed the functionality of A100 GPUs with LM Studio, noting that they can be effective for AI tasks, specifically highlighting their 80GB VRAM capacity.
   - Despite the potentially high costs, interest in acquiring A100s for better performance was evident among the users.
- **Speculative Decoding Analysis**: It was noted that using speculative decoding with certain models may yield lower token acceptance rates and slower performance.
   - Users shared varying experiences with token acceptance and raised questions on the optimal model setups to maximize performance.
- **LM Studio Features**: Users expressed satisfaction with LM Studio, citing its efficiency in saving time and costs for their AI projects.
   - Conversations included discussions about the ease of use when selecting models and features available in the platform.
- **Embedding Models for Long Texts**: The performance of embedding models designed for long texts was discussed, with recommendations for using specific models that support larger context windows.
   - Results from using a 7B model for analyzing lengthy texts demonstrated its capability to answer queries accurately after processing extensive material.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://notebooklm.google.com>">no title found</a>: no description found</li><li><a href="https://huggingface.co/Efficient-Large-Model/Sana_1600M_1024px">Efficient-Large-Model/Sana_1600M_1024px · Hugging Face</a>: no description found</li><li><a href="https://tenor.com/view/tkt-smart-gif-20642718">Tkt Smart GIF - Tkt Smart - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/ggml-org/llama.cpp/discussions/11442">Force All Computations to Run on GPU during Partial Offloading · ggml-org/llama.cpp · Discussion #11442</a>: I propose adding an option in the form of a command-line argument to force all computations onto the GPU during partial offloading with CPU RAM as an offload buffer. This would allow us to maintain...</li><li><a href="https://tenor.com/view/imagination-spongebob-squarepants-dreams-magic-gif-12725683">Imagination Spongebob Squarepants GIF - Imagination Spongebob Squarepants Dreams - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://gitlab.com/logliwo/lm-studio-docker-compose">Aleksey Tsepelev / LM-Studio docker-compose · GitLab</a>: GitLab.com</li><li><a href="https://unsloth.ai/blog/deepseekr1-dynamic">Run DeepSeek-R1 Dynamic 1.58-bit</a>: DeepSeek R-1 is the most powerful open-source reasoning model that performs on par with OpenAI&#x27;s o1 model.Run the 1.58-bit Dynamic GGUF version by Unsloth.</li><li><a href="https://tenor.com/view/doubt-it-i-dont-believe-you-will-farrell-anchor-man-gif-5332521">Doubt It GIF - Doubt It I Dont Believe You Will Farrell - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.youtube.com/watch?v=JAG_83hj1SI">Can DeepSeek AI Really Code a Python Crypto Trading Bot in 5 Minutes?</a>: 🔥 *10% Off Trading Fees!* Register with my Bitget link: https://bonus.bitget.com/Robottraders💡 *What You&#39;ll Learn in This Video* : I put DeepSeek AI to the...</li><li><a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui">GitHub - AUTOMATIC1111/stable-diffusion-webui: Stable Diffusion web UI</a>: Stable Diffusion web UI. Contribute to AUTOMATIC1111/stable-diffusion-webui development by creating an account on GitHub.</li><li><a href="https://lmstudio.ai/blog/lmstudio-v0.3.10)">Blog post not found</a>: no description found</li><li><a href="https://huggingface.co/unsloth/DeepSeek-R1-GGUF">unsloth/DeepSeek-R1-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/unsloth/DeepSeek-R1">unsloth/DeepSeek-R1 · Hugging Face</a>: no description found</li><li><a href="https://github.com/mcmonkeyprojects/SwarmUI">GitHub - mcmonkeyprojects/SwarmUI: SwarmUI (formerly StableSwarmUI), A Modular Stable Diffusion Web-User-Interface, with an emphasis on making powertools easily accessible, high performance, and extensibility.</a>: SwarmUI (formerly StableSwarmUI), A Modular Stable Diffusion Web-User-Interface, with an emphasis on making powertools easily accessible, high performance, and extensibility. - mcmonkeyprojects/Swa...</li><li><a href="https://github.com/erwold/qwen2vl-flux">GitHub - erwold/qwen2vl-flux</a>: Contribute to erwold/qwen2vl-flux development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1341844278040727702)** (190 messages🔥🔥): 

> `Apple Silicon Performance, ARM vs x86 Architecture, Intel's Competition in the Market, Latest AMD Ryzen AI Max+ Specs, Memory Configuration and Performance` 


- **Apple Silicon's Integrated Design Draws Criticism**: Discussion on Apple's trend of soldering components in laptops raises concerns about repairability and upgradeability, leading to a preference for systems that allow component upgrades.
   - Users express frustration over limitations in memory configuration, particularly highlighting that the lack of flexibility constrains performance enhancements.
- **ARM Architecture Seen as Limiting**: Critics argue that the movement towards ARM architecture, particularly in laptops, is driven by marketing rather than tangible performance benefits, compared to traditional x86 systems.
   - Concerns are raised about the inefficiencies in software and process management on these systems, leading to user dissatisfaction.
- **Intel's Struggles in the Competition Arena**: Participants reflect on Intel's significant lag behind competitors like AMD and Apple, and the impact of their design decisions on power consumption and overall performance.
   - There are discussions on the cautious optimism surrounding Intel's future developments, depending on their ability to catch up technologically.
- **AMD Ryzen AI Max+ Impressed but Questions Remain**: The Ryzen AI Max+ specs generate interest among users, but there's skepticism about their real-world performance compared to existing GPUs.
   - Opinions reflect a cautious anticipation for independent benchmarks to truly assess this new architecture against established competitors.
- **Memory Performance Considers Integrated Options**: Tech enthusiasts discuss the implications of memory speed and architecture on overall system performance, particularly comparing HBM and DDR configurations.
   - The conversation highlights the trade-offs faced when integrating components for performance gains versus ensuring overall user control in desktop environments.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://oobabooga.github.io/blog/posts/gptq-awq-exl2-llamacpp/">A detailed comparison between GPTQ, AWQ, EXL2, q4_K_M, q4_K_S, and load_in_4bit: perplexity, VRAM, speed, model size, and loading time. - LLM blog</a>: no description found</li><li><a href="https://www.club386.com/amd-ryzen-ai-max-cpus-beat-nvidia-rtx-4090-at-llm-performance/">AMD Ryzen AI Max+ CPUs beat Nvidia RTX 4090 at LLM work</a>: AMD expands its selection of mobile chips catering to AI workloads, led by a laptop CPU that&#039;s more capable than a discrete graphics card.</li><li><a href="https://hothardware.com/reviews/rog-flow-z13-review">ASUS ROG Flow Z13 Review: AMD Strix Halo Is A Potent Beast</a>: Our first device based on AMD's Ryzen AI MAX impresses with great performance and solid battery life.</li><li><a href="https://tenor.com/view/cat-despair-meme-atone-gif-18083281511005463831">Cat Despair GIF - Cat Despair Meme - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.club386.com/amd-ryzen-ai-max-cpus-beat-nvidia-rtx-4090-at-llm-perform">AMD Ryzen AI Max+ CPUs beat Nvidia RTX 4090 at LLM work</a>: AMD expands its selection of mobile chips catering to AI workloads, led by a laptop CPU that&#039;s more capable than a discrete graphics card.</li><li><a href="https://www.youtube.com/watch?v=WVTuU-Bu7OE">A message from Duolingo&#39;s CEO</a>: Duo&#39;s streak on Earth has ended. In lieu of flowers, do a lesson today in his memory.</li><li><a href="https://docs.unsloth.ai/get-started/unsloth-notebooks">Unsloth Notebooks | Unsloth Documentation</a>: Below is a list of all our notebooks:</li><li><a href="https://www.youtube.com/watch?v=v7HUud7IvAo">AMD CPU, Apple M4 Pro Performance - Ryzen AI MAX Review</a>: The Ryzen AI MAX+ 395 and Ryzen AI MAX 390 are supposed to be Apple M4 and Apple M4 Pro competitors that combine high efficiency with some pretty crazy perfo...
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1341829906580902029)** (358 messages🔥🔥): 

> `Grok 3 Performance, Aider Integration Challenges, Elon Musk's Influence on AI, DeepSeek-R1 Comparison, AI Model Cost Efficiency` 


- **Grok 3: The New Favorite AI**: Users are praising Grok 3 for being faster than GPT-4o and offering effective 'Think' mode, with many considering canceling other subscriptions for it.
   - One user declared it as their 'bestie' for its performance, cheaper pricing, and favorable UI.
- **Aider's Limitations with Large Repos**: A user expressed difficulty passing many files into Aider due to Linux argument size constraints, suggesting the use of a text file with /load commands.
   - They noted that while their repo contains many small files, the length of the nested directory paths is a significant issue.
- **Elon Musk's Impact on AI Perception**: Elon Musk continues to be a divisive figure, with some expressing admiration for his contributions to AI and others critiquing his business practices.
   - Conversations revealed conflicted feelings towards Musk, with humor intertwined in the discussion.
- **DeepSeek-R1 vs OpenAI Models**: SambaNova announced the efficiency of serving DeepSeek-R1 with significant speed and cost reductions compared to existing models in the market.
   - The update claims to offer the highest efficiency for DeepSeek-R1, making significant strides in AI model application and implementation.
- **Cost Concerns of AI Models**: Discussions highlighted the costs associated with various AI models, particularly Sonnet and Grok 3, with users reflecting on their value.
   - Concerns were raised about the sustainability of free offerings from AI services and whether users would migrate to models with clearer cost benefits.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://grok.com/">Grok</a>: Grok is a free AI assistant designed by xAI to maximize truth and objectivity. Grok offers real-time search, image generation, trend analysis, and more.</li><li><a href="https://sambanova.ai/press/fastest-deepseek-r1-671b-with-highest-efficiency">SambaNova Launches the Fastest DeepSeek-R1 671B with the Highest Efficiency</a>: SambaNova announces that DeepSeek-R1 671B is running today on SambaNova Cloud at 198 tokens per second - speeds &amp; efficiency no other platform can match.</li><li><a href="https://blog.jetbrains.com/kotlin/2025/02/openai-vs-deepseek-which-ai-understands-kotlin-better/">OpenAI vs. DeepSeek: Which AI Understands Kotlin Better? | The Kotlin Blog</a>: Which AI model understands Kotlin best? We tested DeepSeek-R1, several OpenAI models, and more using Kotlin-specific benchmarks. See how they compare in our analysis.</li><li><a href="https://tenor.com/view/down-syndrome-gif-9029652995864711868">Down Syndrome GIF - Down syndrome - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://x.com/Yuchenj_UW/status/1892634804786757712">Tweet from Yuchen Jin (@Yuchenj_UW)</a>: I can finally say:Grok 3 is my bestie.- much faster than GPT-4o- the &#34;Think&#34; mode works perfectly with the prompt guideline below- cheaper- I prefer their UI over ChatGPT and Claude (am i a lo...</li><li><a href="https://tenor.com/view/elon-musk-smoke-smoking-well-maybe-gif-12516944">Elon Musk Smoke GIF - Elon Musk Smoke Smoking - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/burgerkingguy-gif-21201954">Burgerkingguy GIF - Burgerkingguy - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/yacquub-lexhinds-gif-19320537">Yacquub Lexhinds GIF - Yacquub Lexhinds - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://x.com/xai/status/1892400129719611567">Tweet from xAI (@xai)</a>: This is it: The world’s smartest AI, Grok 3, now available for free (until our servers melt).Try Grok 3 now: https://x.com/i/grokX Premium+ and SuperGrok users will have increased access to Grok 3, in...</li><li><a href="https://x.com/elonmusk/status/1892452789042757709">Tweet from Elon Musk (@elonmusk)</a>: This is without voice mode and many other features coming out in the next few days</li><li><a href="https://www.reddit.com/r/singularity/comments/1itoi3f/grok3_thinking_had_to_take_64_answers_per/">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1341930831521779736)** (20 messages🔥): 

> `Model Configuration in Aider, Editor vs Architect Mode, Font Color Changes in Aider, Using Local Models, NPM Package Management` 


- **Model Configuration Challenges**: A user shared issues with configuring their `.aider.conf` for fallback models in architect mode, facing errors when providing a list of models.
   - They also mentioned difficulties in changing models without exiting architect mode, seeking assistance on proper configurations.
- **Font Color Visibility Issues**: Concerns were raised about the visibility of the font colors in Aider, with a user noting that the blue color was hard to see in light mode.
   - Suggestions included checking dark mode settings and ensuring the configurations were set properly to address the visibility problem.
- **Switching Between Editor and Architect Mode**: A user asked about switching between editor and architect modes in Aider, expressing frustration with the system defaulting to architect mode.
   - Another member suggested using the `--edit-format` option to control which format would be utilized based on user needs.
- **Local Model Loading in Aider**: Concerns were discussed regarding the slowness of local models, with a user seeking a way to keep the model loaded in RAM throughout the Aider session.
   - This pointed to issues with performance, as repeated loading for every prompt was causing delays.
- **Managing Git Repositories in Aider**: Users reported issues when switching branches with active Aider sessions, encountering bad object errors in the git repository.
   - Proposals included an idea for a `/drop-stale` command to automatically clean up non-existent files from added states to streamline workflows.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://aider.chat/docs/usage/conventions.html">Specifying coding conventions</a>: Tell aider to follow your coding conventions when it works on your code.</li><li><a href="https://aider.chat/docs/more/edit-formats.html">Edit formats</a>: Aider uses various “edit formats” to let LLMs edit source files.
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1341921412478144532)** (2 messages): 

> `Slow Build Process, RAG vs AI Chat Performance, Costs of Indexing` 


- **Build Process Lags Behind**: The build process is notably slow, rewriting chunks with a 'build' method, leading to delays in indexing files and costing significantly in API calls, specifically for **chunks** and **tokens**.
   - *It's frustrating that my indexing is still ongoing since yesterday, yet I can continue using the system in the meantime.*
- **RAG Outperforms AI Chat**: A member expressed that the current **RAG** setup yields better results than the **AI Chat** RAG feature for their coding needs.
   - Another member agreed, noting that *normal RAG struggles with code* and that improvements are necessary.
- **Batch Cost Efficiency Suggestion**: There's a suggestion to enhance the system's efficiency by allowing batching costs from providers to be incorporated, potentially reducing overall costs.
   - This change could address the high expense currently associated with long indexing operations like the ones being experienced.


  

---


### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1341816667176570931)** (354 messages🔥🔥): 

> `Cursor IDE updates, Grok 3 performance, Sonnet 3.5 issues, MCP server functionality, AI model discussions` 


- **Cursor IDE updates spark debate**: Several users reported issues with Cursor's Sonnet 3.5 performance compared to previous versions, expressing frustration over reliability and functionality.
   - In contrast, Grok 3 received praise for its speed and effectiveness in problem-solving during coding tasks.
- **Grok 3 receives mixed reviews**: While some users advocate for Grok 3, claiming it excels in coding tasks, others remain critical of its owner and past performances.
   - Discussions included varying opinions on whether Grok 3 should be implemented in Cursor, highlighting its lack of API access.
- **MCP servers create confusion**: Users discussed the complications surrounding the setup and functionality of MCP servers within Cursor, with some finding it challenging to utilize effectively.
   - Community members suggested that improved documentation could enhance the user experience and streamline installation.
- **AI performance under scrutiny**: Several participants expressed dissatisfaction with the current performance of AI models, notably Claude, attributing inconsistencies in output to underlying prompting and context management issues.
   - It was noted that variations in responses from LLMs are expected, highlighting the stochastic nature of these models.
- **Developer frustrations with Cursor tools**: Users reported challenges using the Cursor Tab, with some stating it introduced bugs during development that slowed workflows.
   - In contrast, Cursor Composer was praised for generating stronger and more reliable code.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.cursor.com/settings/models">Cursor – Models</a>: no description found</li><li><a href="https://browsertools.agentdesk.ai/installation">Installation - AgentDesk - BrowserToolsMCP</a>: no description found</li><li><a href="https://smithery.ai/server/@daniel-lxs/mcp-perplexity">Perplexity Chat MCP Server | Smithery</a>: no description found</li><li><a href="https://gist.github.com/grahama1970/98c5cd8bc4e266fd7b3ebad36e6823eb">This README outlines the limitations of the Cursor MCP Environment, highlighting constraints on package access and environment variables. Key issues include reliance on the Python standard library and the need to hardcode sensitive data. Workarounds involve adapting scripts accordingly. Open questions focus on potential configurations for improved security and access.</a>: This README outlines the limitations of the Cursor MCP Environment, highlighting constraints on package access and environment variables. Key issues include reliance on the Python standard library ...</li><li><a href="https://www.semafor.com/article/12/03/2024/amazon-announces-new-rainier-ai-compute-cluster-with-anthropic">Amazon announces new ‘Rainier’ AI compute cluster with Anthropic</a>: The multi-location data center will train the next generation of Anthropic models.</li><li><a href="https://x.com/windsurf_ai/status/1892322088507105561?s=46&t=ggmESCIXF0nYw8_kshHz7A">Tweet from Windsurf (@windsurf_ai)</a>: DeepSeek-V3 is now unlimited in Windsurf Pro and Ultimate plans.0 prompt credits. 0 flow action credits.</li><li><a href="https://youtu.be/WVpaBTqm-Zo">Grok 3 is an...interesting model.</a>: I had high hopes for Grok 3. According to their benchmarks it should be the new best model right? Right? Quite a lot to talk about with this one...Thank you ...</li><li><a href="https://github.com/anaisbetts/mcp-installer/issues/9">Your MCP config is OSX and Linux specfic · Issue #9 · anaisbetts/mcp-installer</a>: To get this to work on windows one&#39;s config needs to look like this { &quot;mcpServers&quot;: { &quot;mcp-installer&quot;: { &quot;command&quot;: &quot;cmd.exe&quot;, &quot;args&quot;: [ &quot;/c&...</li><li><a href="https://github.com/AgentDeskAI/browser-tools-mcp/issues/5">No matter how many times I add the MCP service in Cursor, it doesn&#39;t take effect, even after restarting Cursor multiple times. · Issue #5 · AgentDeskAI/browser-tools-mcp</a>: BrowserTools Server, on the other hand, runs normally.</li><li><a href="https://downloader.cursor.sh/versions/0.45.14/mac/zip/arm64">no title found</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1341838930718232699)** (79 messages🔥🔥): 

> `Hugging Face Hardcover Release, Qwen2.5 Training Improvement, Video Generators on HF Spaces, Coding Models Discussion, Spark Engine Discord Community` 


- **Hugging Face Hardcover Release is coming!**: Members expressed excitement for the new Hugging Face-themed hardcover book, highlighting a [blog post](https://x.com/Nouamanetazi/status/1892274582503248178) celebrating a year of work by the team.
   - *Click fast to secure a copy if you're interested!*
- **New algorithms for Training Qwen2.5**: A member announced that using Unsloth's new algorithms, users can train reasoning models with just **5GB of VRAM** for Qwen2.5, achieving **10x longer context lengths** with **90% less VRAM**.
   - They shared the [blog link](https://unsloth.ai/blog/grpo) highlighting these improvements and encouraged users to take advantage.
- **Interest in Video Generators on HF Spaces**: Discussion sparked around the availability of video generators on HF Spaces, with a member noting that *ltxv* is quite fast, generating videos in *10-15 seconds* on existing platforms.
   - Another member showed interest in collaborating to create a video generator based on the latest releases.
- **Best Coding Models Comparison**: Members debated the best coding models for development, suggesting various open-source and closed models, with *claude* being highlighted for its static page generation capabilities.
   - Discussions revealed a preference for Hugging Chat due to better control and user freedom compared to proprietary models.
- **Joining the Spark Engine Discord**: A sparkjordi introduced members to the Spark Engine and shared a link to their Discord community, which garnered positive responses.
   - It was revealed that sparkjordi played a role in starting the Spark Engine project, fostering further interest in the platform.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/UnslothAI/status/1892640995847901684">Tweet from Unsloth AI (@UnslothAI)</a>: Today, we’re launching new algorithms that enable 10x longer context lengths & 90% less VRAM for training Reasoning Models (GRPO).Using Unsloth, you can now train your own reasoning model with just 5G...</li><li><a href="https://sparkengine.ai">Spark Engine - The AI Sandbox</a>: Turn ideas into AI-powered products, no coding experience required</li><li><a href="https://x.com/Nouamanetazi/status/1892274582503248178">Tweet from Nouamane Tazi (@Nouamanetazi)</a>: 🚀 Excited to release *THE* Ultra-Scale Playbook - a comprehensive guide on training LLMs from 1 to 1000s of GPUs!</li><li><a href="https://github.com/huggingface/huggingface_hub/issues">huggingface/huggingface_hub</a>: The official Python client for the Huggingface Hub. - huggingface/huggingface_hub</li><li><a href="https://github.com/huggingface/datasets/pull/6968">Use `HF_HUB_OFFLINE` instead of `HF_DATASETS_OFFLINE` by Wauplin · Pull Request #6968 · huggingface/datasets</a>: To use datasets offline, one can use the HF_DATASETS_OFFLINE environment variable. This PR makes HF_HUB_OFFLINE the recommended environment variable for offline training. Goal is to be more consist...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1341832837866983545)** (2 messages): 

> `Quantum Computing, Majorana 1, Satya Nadella's innovations` 


- **Microsoft's Majorana 1 Leads Quantum Computing Charge**: Microsoft has introduced **Majorana 1**, a quantum chip that could potentially solve problems in minutes that would take current supercomputers **billions of years**.
   - This breakthrough comes after nearly **20 years of research** and is seen as a significant milestone in the field of **Quantum Computing**.
- **Satya Nadella Shines Light on Quantum Innovations**: Coinciding with Microsoft's announcement, Satya Nadella shared insights on his latest efforts in the **quantum computing space**.
   - This has sparked excitement and discussions around the implications of quantum technology across various industries.



**Link mentioned**: <a href="https://kuberwastaken.github.io/blog/Technology/Majorana-1---Why-Quantum-Computing-Matters-Now">Majorana 1 - Why Quantum Computing Matters Now</a>: Introduction: A Potential New Era of Computing Imagine a computer so powerful it could solve problems in minutes that would take today’s fastest supercomputers billions of years ...

  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1342086993924063297)** (3 messages): 

> `Zurich 14B Model, Hugging Face Spaces` 


- **Excitement over Zurich 14B Model**: Members expressed enthusiasm about discovering the **Zurich 14B model**, shared via a [Hugging Face collection](https://huggingface.co/collections/rubenroy/zurich-14b-679b57329ebbbc09ab6f03d4).
   - One member commented that it's actually **insane**, emphasizing the model's impressive capabilities.
- **Introducing Zurich 14B Chat Feature**: The **HF Space** featuring the Zurich 14B model allows users to engage in chat for interactive experiences, available for **5 minutes**.
   - *Rocket emojis* and excitement about using such spaces were noted from the discussions.



**Link mentioned**: <a href="https://huggingface.co/collections/rubenroy/zurich-14b-679b57329ebbbc09ab6f03d4">Zurich 14B - a rubenroy Collection</a>: no description found

  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1341851680827904141)** (10 messages🔥): 

> `CommentRescueAI, Aster audio search app, ASR dataset for Ukrainian, docSmith documentation generator, NotAnAI.ai` 


- **CommentRescueAI simplifies Python documentation**: A member introduced **CommentRescueAI**, a tool that adds AI-generated docstrings and comments to Python code with a single click. It is now available on the VS Code extension marketplace, inviting feedback from users.
   - The developer expressed enthusiasm for community input on ideas for improvement.
- **Aster app explores audio search with HF model**: A member shared a blog post detailing the **Aster** app, a free audio search tool utilizing the HF Laion **CLAP** model. The discussion included performance comparisons between ONNX and PyTorch, highlighting that batching support is needed for improved efficiency.
   - Feedback from the community on the app's features and performance is sought to enhance its capabilities.
- **Clean ASR dataset for Ukrainian language**: A member announced the publication of a cleaned **ASR dataset** for Ukrainian, aimed at correcting previous issues with unreliable labels. This dataset is intended to facilitate reliable testing of ASR models and was created with human verification for accuracy.
   - Community members are encouraged to share and promote the dataset to enhance its reach and usefulness.
- **docSmith generates structured documentation**: The launch of **docSmith**, an AI-driven documentation generator, was shared, which creates structured docs directly from GitHub repositories using the **Gemini** language model. It's designed for developers, writers, and project managers to streamline the documentation process.
   - Users can explore the project and its capabilities [here](https://github.com/Jai0401/docSmith).
- **NotAnAI offers interactive AI experiences**: A member introduced **NotAnAI**, an AI-powered Discord bot and website providing interactive experiences with various questions. The underlying technology utilizes the **Qwen2.5-Coder** model for conversational capabilities.
   - Links to both the bot and website were shared, inviting users to try out the functionalities offered.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://asteraudio.app/blog/whats-it-for">Aster</a>: no description found</li><li><a href="https://asteraudio.app/blog/webgpu-wasm-cuda">Aster</a>: no description found</li><li><a href="https://www.kaggle.com/code/allanwandia/secondary-structure-data-analysis">Secondary Structure Data - Analysis</a>: Explore and run machine learning code with Kaggle Notebooks | Using data from multiple data sources</li><li><a href="https://github.com/Jai0401/docSmith">GitHub - Jai0401/docSmith: docSmith is an AI-powered codebase documentation generator for analyzing codebases and producing structured docs. Supports GitHub repos &amp; local files. Perfect for developers, writers, &amp; project managers.</a>: docSmith is an AI-powered codebase documentation generator for analyzing codebases and producing structured docs. Supports GitHub repos &amp;amp; local files. Perfect for developers, writers, &amp;amp...</li><li><a href="https://doc-smith.vercel.app/">docSmith</a>: no description found</li><li><a href="https://huggingface.co/datasets/Yehor/cv10-uk-testset-clean">Yehor/cv10-uk-testset-clean · Datasets at Hugging Face</a>: no description found</li><li><a href="https://notanai.xyz">NotAnAi</a>: Definitely NOT AN AI - k/wom.p.womp</li><li><a href="https://not-an-ai.vercel.app)">no title found</a>: no description found</li><li><a href="https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct">Qwen/Qwen2.5-Coder-32B-Instruct · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1342135559811764236)** (1 messages): 

> `Substack on LLMs, Code & Cognition` 


- **Launch of Substack on LLMs**: A new [Substack](https://open.substack.com/pub/codeandcognition/p/unlocking-lightning-fast-llms-the?r=1u0tss) has been launched, focusing on easy-to-digest content about **LLMs** and AI.
   - The creator invites feedback and emphasizes the aim of sharing practical insights and innovations in the field.
- **Exploration of AI Innovations**: The Substack, titled **Code & Cognition**, explores the latest in **AI**, machine learning, and software engineering with deep dives and practical insights.
   - Launched just a week ago, it aims to provide **cutting-edge innovations** in the space.



**Link mentioned**: <a href="https://open.substack.com/pub/codeandcognition/p/unlocking-lightning-fast-llms-the?r=1u0tss&utm_campaign=post&utm_medium=web&showWelcomeOnShare=false">Unlocking Lightning Fast LLMs: The Power of KV Caching</a>: Have you ever wondered how AI chatbots respond almost instantly, despite running massive language models under the hood? The secret lies in a powerful optimization technique called KV caching.

  

---


### **HuggingFace ▷ #[core-announcements](https://discord.com/channels/879548962464493619/1014557141132132392/1342023553221918761)** (1 messages): 

> `Lumina2 Fine-Tuning, LoRA Implementation` 


- **New Fine-Tuning Script for Lumina2 Released**: A new fine-tuning script for **Lumina2** with **LoRA** has been shipped, enhancing its capabilities for users.
   - Developers can check out the details in the [documentation here](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_lumina2.md).
- **Celebrate Apache2.0 License**: The new feature is under the **Apache2.0** license, promoting openness and accessibility.
   - This aligns with the community's commitment to sharing innovations in AI technology.



**Link mentioned**: <a href="https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_lumina2.md">diffusers/examples/dreambooth/README_lumina2.md at main · huggingface/diffusers</a>: 🤗 Diffusers: State-of-the-art diffusion models for image, video, and audio generation in PyTorch and FLAX. - huggingface/diffusers

  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1342035402890154037)** (5 messages): 

> `Quantifying Summarization and Charts, NLP Learning Resources, Fine-tuning Chat Models, Modular Arithmetic in Coding Theory` 


- **Assessing Summarization with SQL Context**: A member inquired about approaches to quantify summarization and generate charts based on SQL queries and context.
   - They expressed interest in leveraging an LLM as a judge for this assessment and sought guidance on moving forward.
- **Seeking Comprehensive NLP Materials**: One member requested recommendations for complete resources on NLP, from basics to advanced topics.
   - Another member suggested checking out the **NLP course from HuggingFace** as a potential resource.
- **Challenges in Fine-tuning Chat Models**: A user shared their experience of fine-tuning a chat model for one epoch on a dataset of **100,000** samples but faced issues where the model outputs the user's input.
   - They sought assistance from the community to troubleshoot the inference problems they encountered.
- **Modular Arithmetic Problem Solving Techniques**: A question was raised about efficiently solving modular arithmetic problems involving powers and various types of moduli by hand.
   - This is relevant in the context of coding theory and cryptography, highlighting an interest in mathematical methodologies.


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1341838147394207785)** (4 messages): 

> `HF Learn Course Implementation, New Units for Course` 


- **HF Learn Course Becomes More Interactive**: A member is currently working on implementing the course on **HF Learn** to make it more accessible and interactive, as noted in a [Discord message](https://discord.com/channels/879548962464493619/1313889336907010110/1341067833479794780).
   - This effort aims to enhance the overall learning experience by integrating more interactive elements.
- **Plans to Add New Units**: Another member expressed the intention to add new units to the course, indicating ongoing development and updates.
   - This update aims to expand the course offerings, improving its relevance and usefulness.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1341817572345122886)** (238 messages🔥🔥): 

> `Unit 2.1 Publication Status, Accessing Hugging Face Models, Troubleshooting Dummy Agent Library, Introducing Team Members, Questions about Course Format` 


- **Unit 2.1 Publication Status**: Several users expressed confusion over the availability of Unit 2.1, with some confirming it had not yet been published.
   - A user noted they saw a bonus chapter but were unsure about the status of Unit 2.1, leading to discussions about waiting for updates.
- **Accessing Hugging Face Models**: Users shared insights about how to create tokens and request access for the Meta Llama models on Hugging Face, directing others to the relevant settings page.
   - It was noted that specific permissions for inference are needed for the models, emphasizing the need for clarity regarding access levels.
- **Troubleshooting Dummy Agent Library**: A user encountered an error while testing the Dummy Agent Library and suggested that resolving it involved changing the model to a mirror link.
   - Others chimed in about providing alternative APIs and using error handling techniques for model fallback options.
- **Introducing Team Members**: Various introductions were made by users from different backgrounds, including data science, engineering, and machine learning, expressing excitement about the course.
   - Participants showed eagerness to collaborate and network, emphasizing a supportive community atmosphere.
- **Questions about Course Format**: One user questioned whether they were struggling due to poor course formatting or personal misunderstanding, reflecting a common sentiment about learning challenges.
   - This prompted discussions about course clarity and accessibility, indicating a desire for improved structure in the provided materials.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://sparkengine.ai">Spark Engine - The AI Sandbox</a>: Turn ideas into AI-powered products, no coding experience required</li><li><a href="https://huggingface.co/learn/agents-course/bonus-unit1/introduction">Introduction - Hugging Face Agents Course</a>: no description found</li><li><a href="https://huggingface.co/spaces/agents-course/First_agent">First Agent - a Hugging Face Space by agents-course</a>: no description found</li><li><a href="https://agents-course-unit-1-quiz.hf.space/">Dataset Quiz for agents-course/unit_1_quiz</a>: no description found</li><li><a href="https://huggingface.co/spaces/sebasArTecnology/First_agent_template">First Agent Template - a Hugging Face Space by sebasArTecnology</a>: no description found</li><li><a href="https://huggingface.co/learn/agents-course/unit0/introduction">Welcome to the 🤗 AI Agents Course - Hugging Face Agents Course</a>: no description found</li><li><a href="https://huggingface.co/spaces/agents-course/First_agent_template">First Agent Template - a Hugging Face Space by agents-course</a>: no description found</li><li><a href="https://huggingface.co/learn/agents-course/unit1/dummy-agent-library">Dummy Agent Library - Hugging Face Agents Course</a>: no description found</li><li><a href="https://huggingface.co/spaces/vitpolis/First_agent_template">First Agent Template - a Hugging Face Space by vitpolis</a>: no description found</li><li><a href="https://huggingface.co/settings/tokens.">Hugging Face – The AI community building the future.</a>: no description found</li><li><a href="https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct">meta-llama/Llama-3.2-3B-Instruct · Hugging Face</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=Z3yQHYNXPws">Introducing Helix</a>: We&#39;re introducing Helix, a generalist Vision-Language-Action (VLA) model that unifies perception, language understanding, and learned control to overcome mul...
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1341818939243495434)** (243 messages🔥🔥): 

> `Perplexity AI usage issues, Grok 3 performance comparison, Deep Research functionality, O3 and O3 Mini models, API integration and capabilities` 


- **Perplexity AI faces usage issues**: Users have reported frustrating experiences with the Perplexity app, including lag, resource consumption, and glitches during text generation.
   - Concerns were raised about the model's performance, prompting inquiries into whether the development team is addressing these ongoing problems.
- **Grok 3's capabilities under scrutiny**: Discussion around Grok 3 revealed mixed feelings, with some users feeling it performs better than previous models while others noted significant hallucinatory behavior.
   - Users compared Grok 3 to Claude and O3 combinations, leaning towards Claude for more reliable performance.
- **Deep Research's performance evaluation**: The effectiveness of Deep Research was debated, with users noting improvements since the implementation of R1 1776, though hallucinatory outputs remain an issue.
   - One user expressed that both Deep Research and ChatGPT proved useful in retrieving local historical crime data, showcasing their capability over local news.
- **Clarification on O3 and O3 Mini models**: Users clarified that while O3 is a full model, it's not easily accessible, with only the O3 Mini available for general use.
   - There was consensus that O3 Mini effectively retains the capabilities of the full model with limitations on computational power and accessibility.
- **API integration and user support**: Users are exploring the integration of Perplexity models through APIs, aiming to build AI tools efficiently without extensive coding.
   - Concerns were expressed regarding support response times and the pricing model for API usage, with discussions surrounding potential workarounds.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/vectara/Hallucination-evaluation-leaderboard">Hallucination Evaluation Leaderboard - a Hugging Face Space by vectara</a>: no description found</li><li><a href="https://sparkengine.ai">Spark Engine - The AI Sandbox</a>: Turn ideas into AI-powered products, no coding experience required</li><li><a href="https://chromewebstore.google.com/detail/complexity-perplexity-ai/ffppm">Chrome Web Store</a>: Add new features to your browser and personalize your browsing experience.</li><li><a href="https://openrouter.ai/perplexity/r1-1776">R1 1776 - API, Providers, Stats</a>: Note: As this model does not return &lt;think&gt; tags, thoughts will be streamed by default directly to the `content` field.R1 1776 is a version of DeepSeek-R1 that has been post-trained to remove ce...</li><li><a href="https://en.wikipedia.org/wiki/OpenAI_o3">OpenAI o3 - Wikipedia</a>: no description found</li><li><a href="https://x.com/naivigator/status/1892658960496230880">Tweet from Navigator (@naivigator)</a>: 🧵 Introducing Navigator – your all-in-one DeFai AI agent, launchpad, and framework for automating browser tasks! 🚀</li><li><a href="https://www.cplx.app/">Complexity</a>: An enhanced version of Perplexity.ai that everyone has ever wanted.</li><li><a href="https://x.com/CryptoEternalAI/status/1892490182479192287?s=46">Tweet from Eternal AI (EAI) (@CryptoEternalAI)</a>: Anyone can now launch decentralized AI agents using R1 1776 model by @perplexity_ai, stored on @Filecoin and trustlessly served on @Avax.http://eternalai.org/avaxAvax agents powered by @CryptoEternalA...</li><li><a href="https://chromewebstore.google.com/detail/complexity-perplexity-ai/ffppmilmeaekegkpckebkeahjgmhggpj,">Chrome Web Store</a>: Add new features to your browser and personalize your browsing experience.</li><li><a href="https://news.microsoft.com/source/features/ai/microsofts-majorana-1-chip-carves-new-path-for-quantum-computing/">Microsoft’s Majorana 1 chip carves new path for quantum computing - Source</a>: Majorana 1, the first quantum chip powered by a new Topological Core architecture</li><li><a href="https://azure.microsoft.com/en-us/blog/quantum/2025/02/19/microsoft-unveils-majorana-1-the-worlds-first-quantum-processor-powered-by-topological-qubits/">Microsoft unveils Majorana 1, the world’s first quantum processor powered by topological qubits - Microsoft Azure Quantum Blog</a>: Majorana 1 from Microsoft is the world’s first Quantum Processing Unit (QPU) built with a topoconductor. Discover more.</li><li><a href="https://blog.google/technology/research/google-willow-quantum-chip/">Meet Willow, our state-of-the-art quantum chip</a>: Our new quantum chip demonstrates error correction and performance that paves the way to a useful, large-scale quantum computer.</li><li><a href="https://scitechdaily.com/superconduction-breakthrough-scientists-discover-new-state-of-quantum-matter/">Superconduction Breakthrough: Scientists Discover New State of Quantum Matter</a>: Researchers from Cornell University have identified a new state of matter in candidate topological superconductors, a discovery that may have far-reaching implications for both condensed matter physic...
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1341839883173626058)** (23 messages🔥): 

> `AI Hedge Fund outperforming market, Mexico vs Google over Gulf, Bipedal muscular robots, Glowing protein creation, Neural networks analysis` 


- **AI Hedge Fund Surpasses Market Expectations**: A recent article reveals that an **AI Hedge Fund** has significantly outperformed the market, raising eyebrows among investors.
   - The fund leverages advanced algorithms to analyze market trends and decision-making processes.
- **Mexico Issues Warning to Google Over Gulf Access**: In a bold move, **Mexico** threatens **Google** regarding their operations near the Gulf, drawing attention to jurisdictional disputes.
   - This conflict underscores the growing tension between tech companies and national regulators.
- **World's First Bipedal Muscular Robot Unveiled**: A groundbreaking development has emerged with the introduction of the **world's first bipedal muscular robot**, showcasing advanced engineering.
   - This innovation promises to revolutionize robotic movement and interactions with human-like agility.
- **AI Creates Glowing Protein for Research**: **Scientists** have developed an **AI** that can create a **glowing protein**, which could be pivotal in various biological research applications.
   - This protein could facilitate advancements in imaging techniques and molecular biology studies.
- **Exploring Recent Insights in Neural Networks**: An analysis dives into the latest advancements and discussions surrounding **neural networks**, highlighting significant findings.
   - This topic covers various applications and future directions in the field of artificial intelligence.



**Link mentioned**: <a href="https://www.youtube.com/embed/LM6r_rSF1pU">YouTube</a>: no description found

  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1341827686066032763)** (4 messages): 

> `Deep research API, Sonar API performance issues, Model comparison` 


- **Deep research may come to API soon**: Members are inquiring about the potential for **deep research capabilities** to be integrated into the API, hinting at exciting new functionalities.
   - One user expressed enthusiasm, thanking the Perplexity team for their ongoing work in this area.
- **End tag issue with r1-1776 API**: A user reported that the **r1-1776 API** unexpectedly returns an end tag </think> without a matching opening tag <think>, which was verified through curl.
   - They noted that this issue does not occur with **sonar-reasoning models**, where the opening tag is appropriately provided.
- **Concerns about Sonar API performance**: A user raised concerns over the **Sonar API's performance**, suggesting that it yields worse results compared to older models like **llama-3.1-sonar-large-128k-online**.
   - This user has consistently found that the legacy models perform better for tasks like fetching website information, leading to disappointment over the perceived decline in quality despite similar pricing.


  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1341816963575582730)** (156 messages🔥🔥): 

> `PaliGemma 2 Mix Model, AI CUDA Engineer, ALLaM Arabic Model, Helix Robotics Model, Mercor AI Recruiting` 


- **PaliGemma 2 Mix: Enhanced Capabilities**: The newly launched [PaliGemma 2 mix](https://developers.googleblog.com/en/introducing-paligemma-2-mix/) models allow direct exploration of capabilities and are fine-tuned on various real-world tasks, with **promised speedups** over previous versions.
   - Despite user confusion about differences compared to PaliGemma 2, community members noted performance improvements in practical applications.
- **AI CUDA Engineer Claims and Controversies**: The [AI CUDA Engineer](https://pub.sakana.ai/ai-cuda-engineer/paper/) claims 10-100x speedups over existing CUDA kernels, sparking debate after users reported discrepancies in performance benchmarks.
   - Critics are questioning the reliability of speedup claims, with evidence suggesting some optimizations resulted in slower performance.
- **ALLaM's National Efforts in AI**: The **Saudi Arabia**-backed [ALLaM](https://arxiv.org/html/2407.15390v1) focuses on creating Arabic language models to support the ecosystem of Arabic Language Technologies.
   - This represents one of the few successful national efforts in building competitive LLMs in the current geopolitical climate.
- **Helix's Innovations in Robotics**: Figure introduces Helix, a **Vision-Language-Action model** that enables coordinated multi-robot efforts and high-level control of humanoid robots, marking significant advancements in **robotic capabilities**.
   - Equipped with Helix, robots can execute complex tasks, responding dynamically to unfamiliar objects by following natural language prompts.
- **Mercor's AI Recruiting Launch**: [Mercor](https://techcrunch.com/2025/02/20/mercor-an-ai-recruiting-startup-founded-by-21-year-olds-raises-100m-at-2b-valuation/) raises $100 million for its AI recruiting platform, founded by young Thiel Fellows, highlighting its rapid growth and a valuation jump to $2 billion.
   - Asking whether this approach is akin to a data labeling firm, discussions centered on Mercor's innovative marketing drive amidst the competitive AI landscape.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.figure.ai/news/helix">Helix: A Vision-Language-Action Model for Generalist Humanoid Control</a>: Figure was founded with the ambition to change the world.</li><li><a href="https://huggingface.co/spaces/nanotron/ultrascale-playbook">The Ultra-Scale Playbook - a Hugging Face Space by nanotron</a>: no description found</li><li><a href="https://arxiv.org/abs/2502.13595v1">MMTEB: Massive Multilingual Text Embedding Benchmark</a>: Text embeddings are typically evaluated on a limited set of tasks, which are constrained by language, domain, and task diversity. To address these limitations and provide a more comprehensive evaluati...</li><li><a href="https://x.com/Alibaba_WanX/status/1892607749084643453">Tweet from WanX (@Alibaba_WanX)</a>: 🌟 Big News from @alibaba_cloud! 🌟Meet WanX - our next-gen AI model redefining video generation !🚀 Presenting mind-blowing demos from WanX 2.1！🔥 Even more exciting:WanX 2.1 will be OPEN-SOURCE !Com...</li><li><a href="https://developers.googleblog.com/en/introducing-paligemma-2-mix/">Introducing PaliGemma 2 mix: A vision-language model for multiple tasks</a>: no description found</li><li><a href="https://x.com/mervenoyann/status/1892289763954069720">Tweet from merve (@mervenoyann)</a>: @skalskip92 @onuralpszr it&#39;s mixed transfer, but this time model accepts open ended inputs and not structured task prefixes 🥹</li><li><a href="https://x.com/main_horse/status/1892489065746088217">Tweet from main (@main_horse)</a>: @RobertTLange the kernel in your first notebook is broken. please see https://x.com/main_horse/status/1892473238036631908Quoting main (@main_horse) @miru_why I believe there is something wrong with th...</li><li><a href="https://x.com/swyx/status/1892668077768106424?s=46">Tweet from swyx 🗽 NYC (@aiDotEngineer) (@swyx)</a>: First time I’ve seen @AnthropicAI lay out its top priorities like thisfocusing more on mechinterp than Claude 4 now!great presentation from @ambricken and Joe Bayley!Quoting swyx 🗽 NYC (@aiDotEnginee...</li><li><a href="https://fxtwitter.com/bingxu_/status/1892405811596710392?s=61">Tweet from Bing Xu (@bingxu_)</a>: I quickly take a look of their report on phone, there are a few misleading parts: 1. Torch C++ code is not CUDA kernel, it is calling CUDNN under hood.2. The highlighted example Conv3D GroupNorm, conv...</li><li><a href="https://x.com/mervenoyann/status/1892576290181382153">Tweet from merve (@mervenoyann)</a>: we just dropped SmolVLM2: world&#39;s smollest video models in 256M, 500M and 2.2B ⏯️🤗we also release the following 🔥&gt; an iPhone app (runs on 500M model in MLX)&gt; integration with VLC for segme...</li><li><a href="https://x.com/main_horse/status/1892474049114108138">Tweet from main (@main_horse)</a>: @miru_why for avoidance of all doubt: you can unzip the link you provided to me, and apply the following diff to show the kernel is broken.</li><li><a href="https://x.com/btibor91/status/1892290734650433980">Tweet from Tibor Blaho (@btibor91)</a>: Claude web app updates - looks like web search & Paprika modes (the new thinking model) are still in the works, with multiple new builds deployed in the last 24 hoursThis includes a new experiment, &#...</li><li><a href="https://x.com/owl_posting/status/1892317797172015210">Tweet from owl (@owl_posting)</a>: lmao greg brockman&#39;s affiliation for the Evo paper is &#39;independent researcher&#39;</li><li><a href="https://x.com/RobertTLange/status/1892489402070220989">Tweet from Robert Lange (@RobertTLange)</a>: Hi there! Thank you for your interest in our project.Our speedup estimates were obtained on H100. We have confirmed our results on 3 more GPUs and share the corresponding speedups and colab links here...</li><li><a href="https://x.com/tomwarren/status/1892620459062988911">Tweet from Tom Warren (@tomwarren)</a>: scoop: Microsoft is getting ready for OpenAI&#39;s GPT-5 model, and GPT-4.5 could arrive as soon as next week. All of this and more in this week&#39;s 📒 Notepad  issue, live now for subscribers 👇 ht...</li><li><a href="https://arxiv.org/html/2407.15390v1">ALLaM: Large Language Models for Arabic and English</a>: no description found</li><li><a href="https://x.com/klarnaseb/status/1892262217568891179">Tweet from Sebastian Siemiatkowski (@klarnaseb)</a>: @GergelyOrosz We are not reversing course. We are further developing it. Today our AI chat bot handles more complex enquires and at higher quality then back when you tested it. But at the same time th...</li><li><a href="https://x.com/main_horse/status/1892473238036631908">Tweet from main (@main_horse)</a>: @miru_why I believe there is something wrong with their kernel -- it seems to &#39;steal&#39; the result of the eager impl (memory reuse somehow?), allowing it to bypass the correctness check.Here, I ...</li><li><a href="https://x.com/SakanaAILabs/status/1892385766510338559">Tweet from Sakana AI (@SakanaAILabs)</a>: Introducing The AI CUDA Engineer: An agentic AI system that automates the production of highly optimized CUDA kernels.http://sakana.ai/ai-cuda-engineer/The AI CUDA Engineer can produce highly optimize...</li><li><a href="https://techcrunch.com/2025/02/20/mercor-an-ai-recruiting-startup-founded-by-21-year-olds-raises-100m-at-2b-valuation/">Mercor, an AI recruiting startup founded by 21-year-olds, raises $100M at $2B valuation | TechCrunch</a>: Mercor, the AI recruiting startup founded by three 21-year-old Thiel Fellows, has raised $100 million in a Series B round, the company confirmed to</li><li><a href="https://x.com/Figure_robot/status/1892577871366939087">Tweet from Figure (@Figure_robot)</a>: Meet Helix, our in-house AI that reasons like a humanRobotics won&#39;t get to the home without a step change in capabilitiesOur robots can now handle virtually any household item:</li><li><a href="https://fxtwitter.com/main_horse/status/1892446384910987718">Tweet from main (@main_horse)</a>: This example from their paper (https://pub.sakana.ai/static/paper.pdf#page=47), which is claimed to have 150x speedup, is actually 3x slower if you bench it...Quoting Sakana AI (@SakanaAILabs) Introdu...</li><li><a href="https://x.ai/blog/grok-3">Grok 3 Beta — The Age of Reasoning Agents</a>: no description found</li><li><a href="https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero/blob/main/ORZ_paper.pdf">Open-Reasoner-Zero/ORZ_paper.pdf at main · Open-Reasoner-Zero/Open-Reasoner-Zero</a>: Official Repo for Open-Reasoner-Zero. Contribute to Open-Reasoner-Zero/Open-Reasoner-Zero development by creating an account on GitHub.</li><li><a href="https://x.com/GergelyOrosz/status/1892196257608687842">Tweet from Gergely Orosz (@GergelyOrosz)</a>: Klarna was the company that went all-on replacing customer support with an AI bot and went on to brag about the cost savings.Now they are reversing course.Easy to see more companies blindly replacing ...</li><li><a href="https://fxtwitter.com/main_horse/status/1892408991327932883?s=61">Tweet from main (@main_horse)</a>: @SakanaAILabs isn&#39;t there clearly something wrong with level_1-&gt;15_Matmul_for_lower_triangular_matrices?claimed 152.9x speedup for the kernel on the left over the code on the right. really?</li><li><a href="https://huggingface.co/google/paligemma2-3b-mix-448#paligemma-2-results-by-model-resolution-and-size>">google/paligemma2-3b-mix-448 · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1342024334172098630)** (4 messages): 

> `Grok 3 reasoning, Difference between think and big brain, xAI vs OpenAI capabilities, Confusion over scores` 


- **Grok 3 Reasoning is an ~o1 Level Model**: A member noted that if the light blue part represents the best of N scores, then **Grok 3 reasoning** is inherently an ~o1 level model, indicating a **capabilities gap of ~9 months** between OpenAI and xAI.
   - They questioned the meaning of the terms *think* and *big brain*, suggesting a deeper nuance in model performance metrics.
- **Score Misinterpretation Clarified**: Another member clarified that the light shaded areas refer to a **cons@64 score**, which indicates a misunderstanding about the difference between *non-thinking and thinking* models.
   - This led to a moment of frustration, as highlighted by the facepalm emoji expressing the confusion in the discussion.
- **Consensus on Current Model Confusion**: A separate comment reflected a shared sentiment about the **messy state** of the discussion around the differences in model capabilities.
   - The ongoing conversations emphasized the need for clearer communication regarding model evaluations and comparisons.



**Link mentioned**: <a href="https://x.com/nrehiew_/status/1891710589115715847">Tweet from wh (@nrehiew_)</a>: If the light blue part is best of N scores, this means that Grok 3 reasoning is inherently an ~o1 level model. This means the capabilities gap between OpenAI and xAI is ~9 months. Also what is the dif...

  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1341822327041687602)** (69 messages🔥🔥): 

> `Nadella on Dwarkesh, AI competitions, GRPO advancements, Anthropic employee retention, Podcast appearances` 


- **Nadella's Engagement with Dwarkesh**: *Nadella gets prominent guests* on the Dwarkesh podcast, demonstrating his media-savvy approach in the tech landscape.
   - Discussion around how *CEOs leverage podcasts and media* to enhance their public profile and influence.
- **Russia's Position in AI**: Concerns were raised regarding Russia's AI capabilities, with members agreeing that they are **'GPU poor,'** impacting their competitiveness.
   - War applications seem to drive their limited AI efforts, suggesting any advancements are closely tied to military benefits.
- **Innovative GRPO Developments**: Unsloth released a new **GRPO algorithm** that reduces VRAM requirements for Qwen2.5 training to just **5GB**, marking a significant improvement.
   - The algorithm enables **10x longer context lengths**, offering streamlined setups that could revolutionize model training efficiency.
- **Anthropic's Retention Rates**: *AnthropicAI has a high employee retention rate* among major AI labs, underscoring its workplace culture.
   - The focus is shifting towards *mechanistic interpretability*, showcasing a strategic pivot away from Claude 4 development.
- **Insights from the Podcast Circuit**: A member noted surprise at being invited to the podcast channel hosted by Gleb Solomin, highlighting its engaging content.
   - Conversations continued around the value of *podcast appearances* for industry professionals, balancing fun with serious discourse.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/@podcast_solomina/videos">Подкаст Глеба Соломина</a>: Глубокие беседы с мыслящими людьми. Глеб Соломин — 24-летний предприниматель, выпускник МГУ (окончил с красным дипломом) приглашает в гости выдающихся учёных, бизнесменов и людей, достигших высот в св...</li><li><a href="https://x.com/JustinLin610/status/1892625486284734696">Tweet from Junyang Lin (@JustinLin610)</a>: @TheXeophon Yes. 7 is of apache 2.0</li><li><a href="https://tenor.com/view/just-house-totally-duh-gif-23663188">Just House GIF - Just House Totally - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://x.com/huybery/status/1892628963878486233">Tweet from Binyuan Hui (@huybery)</a>: &lt;think&gt;…&lt;/think&gt;Binyuan is cooking…</li><li><a href="https://unsloth.ai/blog/grpo">Long-context GRPO (R1 Reasoning)</a>: DeepSeek R-1 is the most powerful open-source reasoning model that performs on par with OpenAI&#x27;s o1 model.Run the 1.58-bit Dynamic GGUF version by Unsloth.</li><li><a href="https://fxtwitter.com/swyx/status/1892668077768106424">Tweet from swyx 🗽 NYC (@aiDotEngineer) (@swyx)</a>: First time I’ve seen @AnthropicAI lay out its top priorities like thisfocusing more on mechinterp than Claude 4 now!great presentation from @ambricken and Joe Bayley!Quoting swyx 🗽 NYC (@aiDotEnginee...</li><li><a href="https://x.com/swyx/status/1892684773891375125">Tweet from swyx 🗽 NYC (@aiDotEngineer) (@swyx)</a>: TIL @AnthropicAI has the highest employee retention rate of the big labsQuoting swyx 🗽 NYC (@aiDotEngineer) (@swyx) First time I’ve seen @AnthropicAI lay out its top priorities like thisfocusing more...</li><li><a href="https://x.com/colin_fraser/status/1892379172007285176">Tweet from Colin Fraser (@colin_fraser)</a>: Answer: 0/100.It &#34;thought&#34; for four minutes and then came back to me with the (correct, I admit!) answers to five unrelated 3-digit sums and no downloadable file.Quoting Colin Fraser (@colin_f...</li><li><a href="https://x.com/mvpatel2000/status/1892627122729988450">Tweet from Mihir Patel (@mvpatel2000)</a>: Small life update: I joined Anthropic at the start of the year! The future is going to be wild, and I&#39;m incredibly happy to be part of a team changing the world for good 😊. I&#39;m also excited t...</li><li><a href="https://www.youtube.com/watch?v=YXTYbr3hiFU">An Unexpected Reinforcement Learning Renaissance</a>: The era we are living through in language modeling research is one pervasive with complete faith that reasoning and new reinforcement learning (RL) training ...</li><li><a href="https://fxtwitter.com/colin_fraser/status/1892368545884873016">Tweet from Colin Fraser (@colin_fraser)</a>: How do you expect that the OpenAI Deep Research agent will perform on these 100 4-digit addition problems?
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1341854725896671293)** (5 messages): 

> `Useless Machine with AI Agent, AI Research in China and Google, Claude's Situation, AIME 2025 Performance Comparison, Grok's Development` 


- **Interest in AI-Powered Useless Machine**: A request was made for someone to demonstrate a **useless machine** but with an **AI agent** integration, showcasing curiosity in novel AI applications.
   - This reflects an ongoing trend of blending humor and technology, provoking thoughts on the implications of AI in playful formats.
- **China and Google's Open Research Paths**: A member pointed out that the leading countries in AI, namely **China** and **Google**, approach AI through **open research**, expressing skepticism about underlying motives.
   - This commentary hints at ongoing tensions and perceptions around proprietary versus open advancements in the AI field.
- **Concern about Claude**: A member expressed a worried reaction with a message saying, '**Claude nooooo**,' hinting at some distressing update regarding the **Claude** AI.
   - This showcases the level of engagement and concern within the community regarding AI developments.
- **Insights on AIME 2025 Performance**: A compilation of results regarding the **AIME 2025 performance** of **Grok** and **OpenAI** models was shared, with comparisons to other model performances.
   - A notable quote emphasized that for accurate comparisons, it's important to look at results from different training versions, particularly highlighting **Grok3** as still having room to improve efficiently.
- **Grok3's Development Revealed**: Yuhuai (Tony) Wu shared insights on the rigorous training behind **Grok3**, explaining that its larger size affects training duration, yet it is rapidly enhancing.
   - This indicates an ongoing commitment to improving AI capabilities with promises of power being **unleashed** in future updates.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/zetalyrae/status/1892331830939976169">Tweet from Fernando 🌺🌌 (@zetalyrae)</a>: Claude nooooo</li><li><a href="https://x.com/doomslide/status/1892311556991697009">Tweet from doomslide (@doomslide)</a>: it&#39;s quite telling that the two countries at the forefront of AI both follow the path of open research. china, with its ulterior motive to sabotage funding of san francisco&#39;s finest, and googl...</li><li><a href="https://x.com/teortaxestex/status/1892471638534303946?s=46">Tweet from Teortaxes▶️ (DeepSeek 推特🐋铁粉 2023 – ∞) (@teortaxesTex)</a>: Fair enough. Here&#39;s my compilation of all results from relevant sources on AIME 2025 performance of Grok and OpenAI models, plus extrapolations of cons@64 for DeepSeek models and o1. I think this ...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[cv](https://discord.com/channels/1179127597926469703/1208183243447468093/)** (1 messages): 

the_real_jrb: https://arxiv.org/abs/2502.13923
  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1341845360271687731)** (9 messages🔥): 

> `Open Source AI Critique, Satya Nadella on AI, Microsoft Product Quality, Copilot Development, Microsoft Teams Integration` 


- **Open Source AI faces scrutiny**: Critics assert that if **Linux** started today, it would be crushed by **lawmakers** and alarmists claiming open source enables threats, which could lead to **tight control** over software.
   - The discussion highlighted the fear-driven narrative surrounding open-source software, discussing the financial and legislative power behind such movements.
- **Satya Nadella's grounded AI perspective**: In a recent [YouTube video](https://youtu.be/4GLSzuYXh6w), **Satya Nadella** shares his skepticism about AGI while promoting economic growth and Microsoft's **topological qubit breakthrough**.
   - His sensible take drew appreciation, contrasting with the mixed feelings about Microsoft's broader product quality.
- **Concerns over Microsoft product performance**: Members expressed frustration, questioning how **Satya Nadella** can be viewed positively when **Microsoft products** like Teams and Copilot fall short.
   - Insightful comments pointed out that while **Windows** excels in gaming, its search features lag significantly behind competitors like Mac.
- **Copilot receives updates post-competition**: Discussions noted that **Copilot** has undergone numerous updates in response to competition, highlighting perceived shortcomings compared to **Cursor**.
   - Members reflected that Microsoft tends to ignore quality improvements until they feel threatened in the market.
- **Teams gains by integration, not quality**: A member articulated that while **Microsoft Teams** integrates well within the **MSFT ecosystem**, that doesn't necessarily reflect its standalone quality.
   - The dialogue indicated a perception that Microsoft's primary focus is on enterprise clients, shifting definitions of what constitutes a 'good' product.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://danieljeffries.substack.com/p/defending-open-source-ai-against">Defending Open Source AI Against the Monopolist, the Jingoist, the Doomer and the Idiot</a>: If Linux Were Just Getting Started Today, It Would Ge Crushed, and We&#x27;d All be a Lot Poorer for It. We Can&#x27;t Let that Happen to AI.</li><li><a href="https://youtu.be/4GLSzuYXh6w">Satya Nadella – Microsoft’s AGI Plan &amp; Quantum Breakthrough</a>: Satya Nadella on:- Why he doesn’t believe in AGI but does believe in 10% economic growth,- Microsoft’s new topological qubit breakthrough and gaming world mo...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/)** (1 messages): 

SnailBot News: <@&1216534966205284433>
  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1341888997420433419)** (2 messages): 

> `Reasoning Tokens Behavior, User Feedback on Token Responses, Proposed Changes to Reasoning Tokens, Poll on Reasoning Token Settings` 


- **User Feedback Sparks Reasoning Tokens Discussion**: Feedback indicates that users are dissatisfied when **max_tokens are low**, resulting in no content being returned.
   - Currently, **include_reasoning** defaults to false, leading to either empty content or null responses, which users find frustrating.
- **Proposed Changes Aim to Enhance Response Clarity**: Two key proposals are on the table: set **include_reasoning** to true by default and ensure content is always a string, avoiding null values.
   - These changes aim to provide consistency in responses, ensuring developers receive usable content even when reasoning consumes all tokens.
- **Expanded Poll for Community Input**: A poll has been initiated to gather opinions on the proposed changes regarding **include_reasoning** settings.
   - Options range from keeping the current behavior to changing defaults, with feedback being actively sought from the community.


  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1342210705818128488)** (2 messages): 

> `Weaver Chrome Extension, Open Source API Tool` 


- **Weaver: Versatile Chrome Extension**: The **Weaver** Chrome extension allows for highly configurable options like PDF support, cloud sync with **Supabase**, and direct API calls from the browser, promoting better performance.
   - It's currently free but hosted on Vercel's free plan, implying potential limitations on accessibility due to usage limits, with no backend data logging.
- **Open Source Translation Tool Emerges**: A user shared their newly developed **open-source Chrome extension** that allows users to transform any content into their preferred style such as translating or summarizing.
   - The tool is accessible via [GitHub](https://github.com/amirrezasalimi/aify) and only requires an **OpenAI-compatible API** to function.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://weaver-one.vercel.app/">Weaver</a>: no description found</li><li><a href="https://x.com/amirsalimiiii/status/1892667934641692774">Tweet from Amirreza (@amirsalimiiii)</a>: Just cooked up a powerful Chrome extension! Turn any content into your preferred style—translate, simplify, summarize, you name it. 🔥🛠️ Fully open-source & only needs an OpenAI-compatible API.Check ...
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1341817175245066240)** (209 messages🔥🔥): 

> `OpenRouter API Integration, Gemini Model Issues, DeepSeek Models Performance, API Key Generation, Vision and Reasoning Models` 


- **Integrating OpenRouter with Websites**: A user inquired about how to use OpenRouter's API key to integrate a chatbot into their Elementor website, expressing the need for guidance.
   - Another user indicated that OpenRouter only provides access to LLMs, and advised reaching out to a developer for assistance with integration.
- **Gemini 2.0 Model Performance Issues**: Users discussed problems with the Gemini 2.0 Flash model's structured outputs, highlighting discrepancies compared to OpenAI's models.
   - Feedback indicated a need for clarity in the UI regarding the capabilities of different models, especially concerning input types and error messages.
- **Performance Fluctuations in DeepSeek Models**: Some users reported that DeepSeek models yield high-quality responses initially, but later responses deteriorated significantly.
   - Discussion centered on the possible causes for this behavior and whether there are settings to mitigate the decline in response quality.
- **Generating API Keys Programmatically**: A user expressed interest in the ability to programmatically generate API keys for their own usage without visiting the OpenRouter website.
   - Respondents confirmed that this feature is planned for release soon, with hopes for its availability by the end of the week.
- **Understanding Model Capabilities**: A user inquired about how to identify whether models have vision, reasoning, or tool use capabilities when browsing on OpenRouter.
   - Clarifications were provided regarding indicators for vision and reasoning, with suggestions for improving the interface to make this information more accessible.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://poloclub.github.io/transformer-explainer/">Transformer Explainer: LLM Transformer Model Visually Explained</a>: An interactive visualization tool showing you how transformer models work in large language models (LLM) like GPT.</li><li><a href="https://openrouter.ai/perplexity/r1-1776">R1 1776 - API, Providers, Stats</a>: Note: As this model does not return &lt;think&gt; tags, thoughts will be streamed by default directly to the `content` field.R1 1776 is a version of DeepSeek-R1 that has been post-trained to remove ce...</li><li><a href="https://x.com/perplexity_ai/status/1892329089903841467?t=6lD3qXX2sOcKytYFI8L1kA&s=19">Tweet from Perplexity (@perplexity_ai)</a>: R1 1776 is now available via Perplexity&#39;s Sonar API.Quoting Perplexity (@perplexity_ai) Today we&#39;re open-sourcing R1 1776—a version of the DeepSeek R1 model that has been post-trained to provi...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1341824812275994695)** (196 messages🔥🔥): 

> `Grok3 Performance Concerns, Applications of Evolutionary Strategies in Training, Coded Datasets for AI Models, Agents Collaboration and Refinement, Equilibrium Propagation in Neural Networks` 


- **Grok3's Benchmarking and Performance**: Discussions arose regarding the performance of Grok3 and its benchmarking, with some claiming that xAI was not forthright with their data, particularly related to cons@64 usage.
   - Members expressed skepticism about Grok3 outperforming state-of-the-art models, with specific examples shared for context.
- **Exploring Evolutionary Strategies for Training**: The feasibility of using evolutionary algorithms (GAs) for optimizing neural networks was debated, highlighting the slower convergence rates at scale due to high dimensionality.
   - Ideas were exchanged about potentially using GAs for specific components within a training pipeline to enhance model performance while contrasting it with traditional backpropagation.
- **Sharing Quality Coding Datasets**: Users shared various coding datasets available on Hugging Face, suggesting that they could be useful for augmenting existing models.
   - Members reflected on the importance of dataset quality and the potential for reworking existing datasets using advanced reasoning models.
- **Agent Collaboration in Objective Refinement**: A member inquired about state-of-the-art research on agents collaborating to refine ideas towards a goal, specifically how they communicate and utilize methodologies.
   - The conversation included references to personal experiments with agents discussing and refining processes to achieve targeted outcomes.
- **Understanding Equilibrium Propagation in Neural Networks**: Equilibrium propagation was discussed as an alternative to traditional backpropagation for training energy-based models, with an emphasis on its ability to nudge predictions towards a configuration with minimal error.
   - The community engaged in exploring the parallels between equilibrium propagation and recurrent backpropagation, focusing on its potential applications in evolving neural network training techniques.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://news.mit.edu/2025/large-language-models-reason-about-diverse-data-general-way-0219">Like human brains, large language models reason about diverse data in a general way</a>: MIT researchers find large language models process diverse types of data, like different languages, audio inputs, images, etc., similarly to how humans reason about complex problems. Like humans, LLMs...</li><li><a href="https://arxiv.org/abs/1602.05179">Equilibrium Propagation: Bridging the Gap Between Energy-Based Models and Backpropagation</a>: We introduce Equilibrium Propagation, a learning framework for energy-based models. It involves only one kind of neural computation, performed in both the first phase (when the prediction is made) and...</li><li><a href="https://arxiv.org/abs/1711.08416">Equivalence of Equilibrium Propagation and Recurrent Backpropagation</a>: Recurrent Backpropagation and Equilibrium Propagation are supervised learning algorithms for fixed point recurrent neural networks which differ in their second phase. In the first phase, both algorith...</li><li><a href="https://arxiv.org/abs/1808.04873">Generalization of Equilibrium Propagation to Vector Field Dynamics</a>: The biological plausibility of the backpropagation algorithm has long been doubted by neuroscientists. Two major reasons are that neurons would need to send two different types of signal in the forwar...</li><li><a href="https://huggingface.co/microsoft/wham">microsoft/wham · Hugging Face</a>: no description found</li><li><a href="https://fxtwitter.com/satyanadella/status/1892244164814725387">Tweet from Satya Nadella (@satyanadella)</a>: If you thought AI-generated text, images, and video were cool, just imagine entire interactive environments like games!</li><li><a href="https://steamcommunity.com/sharedfiles/filedetails/?id=3143225812&searchtext=">Steam Workshop::Reforged Eden 2 Beta</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1itdy0k/no_system_instructions_for_deepseek_makes_jake/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://huggingface.co/datasets/NovaSky-AI/Sky-T1_data_17k">NovaSky-AI/Sky-T1_data_17k · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1342230052565155962)** (1 messages): 

> `Reinforcement Learning for LLMs, Scaling Supervision` 


- **Explainer on Reinforcement Learning for Beginners**: A member shared a [Twitter thread](https://x.com/ShashwatGoel7/status/1892668493390094338) that explains **Reinforcement Learning** (RL) tailored for newcomers to **large language models** (LLMs), emphasizing a no-prerequisite approach.
   - The thread highlights that **RL** is exciting because it enables learning from **rewards** rather than relying solely on demonstrations.
- **Importance of Scaling Supervision with RL**: The thread emphasizes that **scaling supervision** is a significant benefit of using **Reinforcement Learning**, as it allows for effective learning with simpler reward mechanisms.
   - This approach ultimately shifts the paradigm from needing detailed demonstrations to leveraging more generalized reward feedback.



**Link mentioned**: <a href="https://x.com/ShashwatGoel7/status/1892668493390094338">Tweet from Shashwat Goel (@ShashwatGoel7)</a>: I pieced together this first-principles no RL prerequisites explainer on how RL for LLMs works, and why we need it🧵The main point? RL is exciting because it allows us to scale supervision. We can now...

  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1341818699895799808)** (75 messages🔥🔥): 

> `Transformer Backpropagation, Logit vs Probability in Decision Making, Evolutionary Strategies for LLMs, LoRA vs Full Fine-Tuning, Reinforcement Learning for LLMs` 


- **Struggles with Transformer Backpropagation**: A user expressed confusion about implementing backpropagation in transformers, specifically with handling parallel operations and attention mechanisms.
   - Others suggested focusing on individual attention components and using resources like the unsloth triton kernels for reference.
- **Logits Hold More Information than Probabilities**: Discussions centered around the notion that logits are more expressive than normalized probabilities, while suggesting that unnecessary normalization could hinder optimization processes.
   - It was asserted that while probabilities are necessary for decision making, working in logit space could enhance training efficiency for certain models.
- **Low-Rank Adaptation (LoRA) Limitations**: Participants discussed how LoRA may not be equivalent to full fine-tuning due to its lower-dimensional updates which could limit fitting new data accurately.
   - It was argued that while smaller rank LoRA struggles to maintain invariance for out-of-distribution data, higher-rank LoRA approaches full fine-tuning but reduces efficiency.
- **Concerns with Evolutionary Strategies**: A user questioned if Evolutionary Strategies (ES) experience similar limitations as LoRA in lower-dimensional learning frameworks, suggesting potential issues with mutation noise.
   - The response indicated that while ES might not suffer from the same challenges as LoRA, it could still face problems if the mutation noise is too strong.
- **Reinforcement Learning Realizations**: A user shared that they pieced together an introductory explainer on Reinforcement Learning applied to LLMs, emphasizing its capacity to enhance supervision scaling.
   - The explainer posited that RL allows learning solely from rewards instead of requiring demonstrations, highlighting its potential for efficiency in model training.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/ShashwatGoel7/status/1892668493390094338">Tweet from Shashwat Goel (@ShashwatGoel7)</a>: I pieced together this first-principles no RL prerequisites explainer on how RL for LLMs works, and why we need it🧵The main point? RL is exciting because it allows us to scale supervision. We can now...</li><li><a href="https://arxiv.org/abs/2502.06773">On the Emergence of Thinking in LLMs I: Searching for the Right Intuition</a>: Recent AI advancements, such as OpenAI&#39;s new models, are transforming LLMs into LRMs (Large Reasoning Models) that perform reasoning during inference, taking extra time and compute for higher-qual...</li><li><a href="https://www.youtube.com/watch?v=X_niF6KaWd8">🚨🚨 Chad Game Dev Reviews Devin.ai Game Code 🚨🚨</a>: Twitch https://twitch.tv/ThePrimeagenDiscord https://discord.gg/ThePrimeagenBecome Backend Dev: https://boot.dev/prime(plus i make courses for them)This is a...
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1341866357540655184)** (73 messages🔥🔥): 

> `DeepSeek's Sparse Attention Paper, AGI and Intelligence Models, Conditional Attention Concepts, Differential Transformers` 


- **DeepSeek Releases Native Sparse Attention**: Today, participants engaged in a discussion about DeepSeek's paper on **Native Sparse Attention**, exploring its implications for efficiency and contextual awareness. The event is scheduled to repeat this Saturday for those who missed it.
   - *I like papers from DeepSeek!* They do good research and have high standards, making their findings accessible.
- **Debating AGI Definitions and Reach**: There was a consensus that defining **AGI** remains challenging, with various opinions on its implications and realizations in technology. Participants suggested alternative terms like **ActGI** to navigate the ongoing debate.
   - Discussions highlighted that *not everyone's definition will suit all scenarios*, contributing to the complexity of establishing a universally accepted definition.
- **Understanding Conditional vs. Sparse Attention**: Conditional attention was discussed as a decision-making process versus the implicit selection in **sparse attention models**. A member explained how their mechanism captures relevance through compressed representations.
   - This comparison clarifies how modern attention mechanisms could evolve to improve computational efficiency.
- **Importance of Continual Learning**: There was a dialogue on **continual learning** being less mature compared to other fields like reinforcement learning, with suggestions to explore maturity levels across different areas. The participants emphasized the significance of fostering understanding in the field.
   - There was an acknowledgment that improving abilities such as memory retention could propel advancements in learning efficiency.
- **Novel Ideas in Transformer Research**: Contributions about **differential transformers** sparked interest, noted for their innovative approaches that lack commercial traction. Participants felt many valuable papers remain underappreciated in the current research landscape.
   - A desire for combining ideas from sparse and differential approaches was expressed, highlighting further potential for transformation in the field.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1341844452825759887)** (9 messages🔥): 

> `Perplexity AI and Chinese Censorship, Microsoft Unveils Majorana 1, Topological Qubits Explained, Windows 11 Privacy Updates, Google's PaliGemma 2 Launch` 


- **Perplexity AI breaks through censorship**: Perplexity AI unveiled R1 1776 to overcome Chinese censorship in their Deepseek R1 model using specialized techniques.
   - This move highlights the growing importance of **AI** in navigating and overcoming regulatory barriers.
- **Microsoft introduces Majorana 1 Quantum Processor**: Microsoft announced the **Majorana 1**, the world's first QPU powered by topological qubits, aiming to scale to a million qubits.
   - This advancement represents a significant step towards practical **quantum computing** and error correction.
- **Understanding topological qubits**: A new YouTube video explains the significance of **topological qubits**, featuring insights from the Microsoft team behind the Majorana 1 chip.
   - The content emphasizes how these breakthrough materials could redefine quantum computing capabilities.
- **Windows 11 undergoes privacy-related changes**: Microsoft is removing several features from Windows 11's File Explorer to comply with **privacy regulations** in Europe.
   - The update results in a streamlined interface for European users, disconnecting features that relied on tracking user data.
- **Launch of PaliGemma 2 Vision-Language Models**: Google announced the release of **PaliGemma 2 mix checkpoints**, an upgraded vision-language model with various pretrained sizes.
   - This model is designed for fine-tuning across a multitude of tasks, including image segmentation and scientific question answering.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://fixvx.com/Alibaba_WanX/status/1892607749084643453">Tweet from undefined</a>: no description found</li><li><a href="https://arxiv.org/abs/2502.08859">EnigmaEval: A Benchmark of Long Multimodal Reasoning Challenges</a>: As language models master existing reasoning benchmarks, we need new challenges to evaluate their cognitive frontiers. Puzzle-solving events are rich repositories of challenging multimodal problems th...</li><li><a href="https://it.slashdot.org/story/25/02/20/0227241/microsoft-declutters-windows-11-file-explorer-in-the-name-of-euro-privacy">Microsoft Declutters Windows 11 File Explorer in the Name of Euro Privacy - Slashdot</a>: Microsoft will strip several features from Windows 11's File Explorer for European users to comply with privacy regulations, the company says. The changes, affecting Entra ID accounts in the European ...</li><li><a href="https://developers.googleblog.com/en/introducing-paligemma-2-mix/?linkId=13028688">Introducing PaliGemma 2 mix: A vision-language model for multiple tasks</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=wSHmygPQukQ">Majorana 1 Explained: The Path to a Million Qubits</a>: Hear from the Microsoft team behind the recent breakthrough in physics and quantum computing demonstrated by the new Majorana 1 chip, engineered from an enti...</li><li><a href="https://the-decoder.com/perplexity-ai-removes-chinese-censorship-from-deepseek-r1/">Perplexity AI removes Chinese censorship from Deepseek R1</a>: Perplexity AI has unveiled R1 1776, a modified version of the Deepseek-R1 language model specifically designed to overcome Chinese censorship through specialized post-training techniques.</li><li><a href="https://x.com/elder_plinius/status/1891968598496760230?s=46">Tweet from Pliny the Liberator 🐉󠅫󠄼󠄿󠅆󠄵󠄐󠅀󠄼󠄹󠄾󠅉󠅭 (@elder_plinius)</a>: 🧙‍♂️ 󠅗󠅗NEW ATTACK CLASS UNLOCKED 🧙‍♂️󠅗󠅗</li><li><a href="https://azure.microsoft.com/en-us/blog/quantum/2025/02/19/microsoft-unveils-majorana-1-the-worlds-first-quantum-processor-powered-by-topological-qubits/">Microsoft unveils Majorana 1, the world’s first quantum processor powered by topological qubits - Microsoft Azure Quantum Blog</a>: Majorana 1 from Microsoft is the world’s first Quantum Processing Unit (QPU) built with a topoconductor. Discover more.
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1341977154145222717)** (12 messages🔥): 

> `GPU spec spreadsheet, AI CUDA Engineer, Snapdragon GPU computations, GPU architecture resources, Computer architecture books` 


- **Searching for a definitive GPU spec spreadsheet**: A member expressed frustration about not finding a reliable **GPU spec spreadsheet**, similar to one linked ([Google Sheets](https://docs.google.com/spreadsheets/d/e/2PACX-1vSdXHeEqyabPZTgqFPQ-JMf-nogOR-qaHSzZGELH7uNU_FixVDDQQuwmhZZbriNoqdJ6UsSHlyHX89F/pubhtml)). Another member suggested [TechPowerUp](https://www.techpowerup.com/gpu-specs/) as a potential resource.
- **Excitement about the AI CUDA Engineer**: Introducing the **AI CUDA Engineer**, which automates the creation of optimized CUDA kernels with claims of achieving **10-100x speedup** in PyTorch operations ([Sakana AI](http://sakana.ai/ai-cuda-engineer/)). The system also releases a dataset of **over 17,000 verified CUDA kernels** and a paper detailing its capabilities ([paper link](https://pub.sakana.ai/ai-cuda-engineer/paper/)).
- **Interest in Snapdragon GPU computing platforms**: A member inquired about a channel for **Snapdragon/Adreno GPU computing**, as they're exploring this on a Windows on ARM laptop. The conversation highlighted their interest in **OpenCL/Vulkan** computations on this platform.
- **Seeking GPU architecture resources**: A member, new to GPUs, is looking for resources focused on **GPU architecture** and how optimizations link to hardware design. They referenced a helpful resource from **Springer** and asked for additional recommendations.
- **Inquiry about computer architecture books**: A member expressed curiosity about good **computer architecture** books, seeking suggestions from others in the community. This reflects an ongoing interest in foundational principles relevant to their GPU studies.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/SakanaAILabs/status/1892385766510338559">Tweet from Sakana AI (@SakanaAILabs)</a>: Introducing The AI CUDA Engineer: An agentic AI system that automates the production of highly optimized CUDA kernels.http://sakana.ai/ai-cuda-engineer/The AI CUDA Engineer can produce highly optimize...</li><li><a href="https://www.techpowerup.com/gpu-specs/">TechPowerUp</a>: no description found</li><li><a href="https://link.springer.com/book/10.1007/978-3-031-01759-9">General-Purpose Graphics Processor Architectures</a>: no description found</li><li><a href="https://docs.google.com/spreadsheets/d/e/2PACX-1vSdXHeEqyabPZTgqFPQ-JMf-nogOR-qaHSzZGELH7uNU_FixVDDQQuwmhZZbriNoqdJ6UsSHlyHX89F/pubhtml#">GPU_Compare</a>: no description found
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1342249630687170622)** (1 messages): 

> `TMA Descriptor in Triton, Persistent Kernel Implementations, Matrix Multiplication Techniques, FP8 and FP16 Support, Benchmarking Triton with cuBLAS` 


- **Exploring TMA Descriptor Usage in Triton**: The tutorial on [persistent matmul](https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html) illustrates how the **TMA (Tensor Memory Accelerator)** descriptor can enhance matrix multiplication implementations in Triton.
   - The script provides various examples, including **naive**, **persistent**, and **TMA-based approaches**, emphasizing the benefits of TMA in efficient memory use.
- **Matrix Multiplication Methods Highlighted**: The tutorial showcases several matrix multiplication techniques implemented in Triton, specifically **naive**, **persistent**, and **TMA-based** methods for optimized performance.
   - It also mentions that kernels support both **FP16** and **FP8** data types, with specific instructions for usage depending on the chosen precision.
- **Configurable Command-Line Arguments**: Users can flexibly specify matrix dimensions and iterations through command-line arguments, such as using `--prec` for precision settings in the **FP8** and **FP16** examples.
   - For instance, the command `python 09-persistent-matmul.py --prec fp8 --K_range 128 1024 --K_step 128` sets the parameters for the **FP8** implementation.
- **Caveats for Shared Memory Size**: The tutorial warns that it may fail on devices with limited shared memory size, such as the **RTX-4090**, which could affect performance and compatibility.
   - This consideration is vital for users aiming to successfully execute the examples provided in the tutorial.
- **Benchmarking Strategy Explained**: The script benchmarks the Triton and **cuBLAS implementations** under varying configurations and evaluates them using the **proton profiler**.
   - This benchmarking approach helps users understand the performance implications of different matrix multiplication techniques.



**Link mentioned**: <a href="https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html">Persistent Matmul &mdash; Triton  documentation</a>: no description found

  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1341841708400508979)** (14 messages🔥): 

> `Raw-Dogged Tensor Proposal, RTX 5080+ Triton Issues, Warp Specialization Kernels, TF32 NT Kernel Inquiry, Custom gmem Offset Math in Device Code` 


- **Proposing Raw-Dogged Tensor Nomenclature**: A member proposed a new nomenclature called a **raw-dogged Tensor**, aimed at aligning storage format with **MMA_Atom** thread layout. They noted a significant reduction in permutation complexity.
   - Another member confirmed using this approach for **int8 matmul**, emphasizing its necessity to avoid shared-memory bank conflicts.
- **RTX 5080+ Triton Compatibility Hurdles**: A member shared their experience running **RTX 5080+** on Triton with **TorchRL**, highlighting errors related to `torch.compile` triggering Triton issues. They resolved the problems by removing the **PyTorch-triton** installation.
   - This brought attention to the compatibility concerns that remain with Triton and PyTorch interactions.
- **Cool Warp Specialization Kernels Discussion**: Inquiries were made about **warp specialization kernels**, with examples cited such as the one from the [arxiv link](https://arxiv.org/pdf/2307.03760). Members discussed common **GEMM kernels with producer/consumer specialization**, noting synchronization techniques.
   - A member also encouraged reviewing a useful presentation highlighting **GEMM warp specialization**.
- **Seeking TF32 16x8x8 NT Kernel**: A request was made for **TF32 16x8x8 NT kernel** implementations as part of improving their work in Cutlass. The inquiry reflects the ongoing need for optimized kernels in contemporary applications.
- **Custom gmem Offset Math for Batched Syrk**: A user inquired about implementing a **batched strided SYRK** by adjusting **gmem offset math** based on block index. They expressed difficulty finding a suitable path with standard Cutlass features while ensuring **bM == bN**.



**Link mentioned**: <a href="https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/collective/sm90_mma_tma_gmma_rs_warpspecialized.hpp">cutlass/include/cutlass/gemm/collective/sm90_mma_tma_gmma_rs_warpspecialized.hpp at main · NVIDIA/cutlass</a>: CUDA Templates for Linear Algebra Subroutines. Contribute to NVIDIA/cutlass development by creating an account on GitHub.

  

---


### **GPU MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1342211788707467325)** (1 messages): 

> `GRPO algorithm advancements, VRAM reduction techniques, Extended context lengths, Llama 3.1 benchmarking, Gradient checkpointing` 


- **Unsloth unveils 10x context and 90% VRAM savings**: Unsloth announced new algorithms enabling training with just **5GB VRAM** for **Qwen2.5-1.5B** models, achieving a **90% reduction** in VRAM usage.
   - They stated, *'Using Unsloth, you can now train your own reasoning model with no accuracy loss.'* More details can be found on their [blog](https://unsloth.ai/blog/grpo).
- **Benchmarking results reveal significant VRAM savings**: Comparative benchmarks show that a standard GRPO QLoRA setup for **Llama 3.1** at 20K context previously required **510.8GB VRAM**, now reduced to **54.3GB**.
   - This improvement comes from leveraging a previous **gradient checkpointing algorithm**, inspired by **Horace He**'s linear cross entropy implementation.



**Link mentioned**: <a href="https://x.com/UnslothAI/status/1892640995847901684">Tweet from Unsloth AI (@UnslothAI)</a>: Today, we’re launching new algorithms that enable 10x longer context lengths & 90% less VRAM for training Reasoning Models (GRPO).Using Unsloth, you can now train your own reasoning model with just 5G...

  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1341861646724890756)** (12 messages🔥): 

> `AI CUDA Engineer, Nanotron Blog Post, HadaCore Quantization, CUDA Kernel Optimization, Quantization Techniques` 


- **AI CUDA Engineer Automates Optimization**: The [AI CUDA Engineer](https://pub.sakana.ai/ai-cuda-engineer/) can produce highly optimized CUDA kernels, achieving **10-100x** speedup over common machine learning operations in PyTorch.
   - It achieves a **90%** success rate in translating PyTorch operations to CUDA and is superior to native torch kernels, but some feel the actual paper may be *overhyped* due to weak baselines.
- **Nanotron Team Reveals New Blog Post**: The [Nanotron](https://huggingface.co/spaces/nanotron/ultrascale-playbook) team has released an exciting blog post that was described as **awesome** by some users.
   - Discussion centered around whether Nanotron is a team within Hugging Face, with confirmation on their involvement with the [GitHub project](https://github.com/huggingface/nanotron).
- **HadaCore Introduces Advanced Quantization Methods**: The [HadaCore](https://pytorch.org/blog/hadacore/?utm_source=tldrai) method highlights a Hadamard Transform CUDA kernel that enhances quantization technique efficiency, achieving performance gains of **1.1–1.4x** over its predecessors.
   - Recent works like [QuaRot](https://arxiv.org/abs/2404.00456) and [SpinQuant](https://arxiv.org/abs/2405.16406) demonstrate methods to improve the numerical accuracy of low-precision quantization methods utilized in large language models.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://sakana.ai/ai-cuda-engineer/">no title found</a>: no description found</li><li><a href="https://pytorch.org/blog/hadacore/?utm_source=tldrai">HadaCore: Tensor Core Accelerated Hadamard Transform Kernel</a>: Quantization is a method for improving model inference speeds by compressing model weights and performing (faster) computation in lower precision data types. However, quantization can result in accura...</li><li><a href="https://huggingface.co/spaces/nanotron/ultrascale-playbook">The Ultra-Scale Playbook - a Hugging Face Space by nanotron</a>: no description found</li><li><a href="https://pub.sakana.ai/ai-cuda-engineer/">The AI CUDA Engineer 👷</a>: no description found</li><li><a href="https://github.com/huggingface/nanotron">GitHub - huggingface/nanotron: Minimalistic large language model 3D-parallelism training</a>: Minimalistic large language model 3D-parallelism training - huggingface/nanotron
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1342175502055641170)** (2 messages): 

> `Apple ML Research, A5Labs ML Engineer Position` 


- **Apple Research Scientist Position Open**: The Apple machine learning research group is hiring a **research scientist** focused on curiosity-driven work in **efficient foundation models**. Interested candidates can check the [job description here](https://jobs.apple.com/en-us/details/200587898/aiml-ml-researcher-foundation-models?team=MLAI).
   - The team has a strong research background with impactful papers in **NLP** and **speech** and emphasizes the importance of reproducible high-quality research.
- **A5Labs Seeking Remote ML Engineer**: A5Labs is looking for a **remote ML Engineer** specializing in **reinforcement learning** and gaming, with a diverse global team. Interested applicants can view the [job listing here](https://a5labs.co/we-are-hiring/?jobId=Pz34B6RbYyAI).
   - The team invites direct messages from candidates and highlights their international presence across **Asia**, **North America**, and **Europe**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://a5labs.co/we-are-hiring/?jobId=Pz34B6RbYyAI">We’re hiring! - A5 Labs</a>: Career Center We’re hiring! Join the A5 Labs Team</li><li><a href="https://jobs.apple.com/en-us/details/200587898/aiml-ml-researcher-foundation-models?team=MLAI.">AIML - ML Researcher, Foundation Models - Careers at Apple</a>: Apply for a AIML - ML Researcher, Foundation Models job at Apple. Read about the role and find out if it’s right for you.
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1341843700451381348)** (9 messages🔥): 

> `torchao issue, HuggingFace error, past_key_values bug, modeling_llama.py fix` 


- **torchao experiencing issues**: A member reported that there is a **broken issue** in *torchao* and mentioned they are investigating it further.
   - Another member offered to help with the situation.
- **HuggingFace torchao example error**: A link was shared to a GitHub issue regarding a **torch.compile error** when running the HuggingFace torchao example, citing versions of **torch (2.6.0)** and **torchao (0.8.0)**.
   - The issue description mentioned a problem with both quantization and the example code provided.
- **Identifying the cause of the error**: It was suggested that the error occurs due to Hugging Face using **past_key_values** and **past_key_value** interchangeably, leading to confusion and bugs.
   - This inconsistency was noted as a significant part of the problem contributing to the errors.
- **Proposed fix for the Llama model**: A pull request was linked that offers a **bugfix** for the Llama model by updating *modeling_llama.py* to correctly handle the keys skipping issue.
   - The bugfix addresses the mixed usage of **past_key_value** and **past_key_values**, ensuring both are skipped appropriately during processing.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/pytorch/ao/issues/1705">torch.compile error when running the HuggingFace torchao example · Issue #1705 · pytorch/ao</a>: When I run the code snippet from https://huggingface.co/docs/transformers/main/en/quantization/torchao, I see a torch.compile error. torch version: 2.6.0 torchao version: 0.8.0 transformers version...</li><li><a href="https://github.com/huggingface/transformers/pull/36289">[bugfix] Update modeling_llama.py so it skips keys correctly by HDCharles · Pull Request #36289 · huggingface/transformers</a>: the llama model was using past_key_value and past_key_values interchangeably which caused issues because only one of those was actually skipped in _skip_keys_device_placement when both needed to be...
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1342176395618549872)** (1 messages): 

> `Together Computer Series Funding` 


- **Together Computer Secures $305M in Series Funding**: Today, [Together Computer](https://www.linkedin.com/posts/togethercomputer_today-were-announcing-our-305m-series-activity-7298375921277800450-Jvjs/) announced its impressive **$305 million** Series funding aimed at accelerating its technological advancements.
   - This significant investment highlights the increasing interest and potential in the AI computing sector.
- **Growth in AI Computing Investments**: This round of funding showcases a trend where investors are increasingly pouring money into **AI computing** companies, indicating strong market confidence.
   - Industry experts believe this may lead to further innovations and breakthroughs, especially in **machine learning** and **cloud computing**.


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/)** (1 messages): 

kpk1340: Anyone in NYC?
  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1341939234327494676)** (9 messages🔥): 

> `Mi50 Hardware Support, Matmul Operations, GPU Architectures` 


- **Mi50 Lacks Hardware Matmul Support**: Members confirmed that the **Mi50** does not support hardware matmul, or tensor operations, despite its capability to handle multiple data types.
   - One member stated, *'No wmma and no mfma'* indicating the absence of specific matrix multiplication features.
- **Clarification on Matmul Technologies**: Discussion revealed that matmul support is delineated between **XDL** for datacenter use in CDNA architectures and **WMMA** for gaming on RDNA3 cards.
   - Another member emphasized that the **Mi50** utilizes **Vega / GCN 5**, which does not include these newer features.
- **Acknowledgment of Mi50's Limitations**: Conversations highlighted the consensus on the limitations of the **Mi50** regarding matmul capabilities, specifically its inability to utilize WMMA.
   - Members expressed appreciation for the confirmation, affirming their understanding of the hardware's specifications.



**Link mentioned**: <a href="https://www.8anet.com/Product/17823/AMD-100-506143-Radeon-Instinct-MI50-Accelerator-PCIe-4-0-x16-32GB-HBM2-4096-bit-3840-Stream-Processors-Passive-Cooling">
	8ANET - AMD 100-506143 Radeon Instinct™ MI50 Accelerator PCIe 4.0 x16 32GB HBM2 4096-bit 3840 Stream Processors Passive Cooling
</a>: no description found

  

---


### **GPU MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/1341903496987017267)** (3 messages): 

> `Convergence test fix, PR merging process, Native Sparse Attention` 


- **Convergence Test Fix Success**: A member reported fixing the **convergence test** by addressing a missing logit scaling parameter in the **MiniModelConfig**, leading to corrected logit magnitudes.
   - They expressed a desire for assistance in getting the **PR merged**, stating their willingness to do anything required to expedite the process.
- **Inquiry about PR Number**: Another member inquired about the specific **PR number** needed for the merging process.
   - The message was light-hearted, with a laughing emoji, indicating a friendly atmosphere in the discussion.
- **Interest in Native Sparse Attention Collaboration**: A member initiated a conversation about the **Native Sparse Attention** feature, asking if anyone is interested in collaborating on making it hardware-aligned and natively trainable in **liger**.
   - The invitation to work together was met with enthusiasm, showcasing a collaborative spirit in the community.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/)** (1 messages): 

iron_bound: Goat https://m.youtube.com/watch?v=leCY8vCUS4g
  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1341947354999296130)** (10 messages🔥): 

> `AI CUDA Engineer, CUDA kernel optimization, Rewards and challenges in code generation, Research papers on CUDA, Evolutionary AI approaches` 


- **AI CUDA Engineer optimizes CUDA kernels**: The [AI CUDA Engineer](http://sakana.ai/ai-cuda-engineer/) automates the production of highly optimized CUDA kernels, achieving **10-100x speedup** over standard operations in PyTorch.
   - This system utilizes evolutionary **LLM-driven code optimization** to enhance CUDA kernel performance and even discover novel operation solutions.
- **Notable contributions and findings**: The paper outlines significant findings such as **kernels exceeding torch.compile** for specific tasks, including categorical cross-entropy optimizations and a dataset of **17K** kernel pairs with speedups.
   - It also highlights the challenges in selecting useful data from NCU and teaching LLMs about new features like tensor cores.
- **Insights on reward mechanisms**: *AutoML is back!* Current discussions emphasize that the reward function for improving CUDA kernels is clearly defined, focusing on **numeric correctness and wall clock speed**.
   - One member jokingly noted a case of **reward hacking** where a ‘nop kernel’ won because it didn't do anything, humorously reflecting the nature of optimization.
- **Discussion on kernel issues**: Concerns arose over some kernels being malformed due to **output buffer reuse**, affecting their performance.
   - Issues such as reclaiming memory used by previous outputs were discussed as significant obstacles in ensuring kernel correctness.
- **Fun environment for collaboration**: Several members contemplated potential collaboration opportunities with Sakana AI, suggesting it as a promising **Colab opportunity**.
   - There's a lighthearted atmosphere in discussions, with members sharing quips about the ease of avoiding mistakes by not executing any operations—*you can't ruin anything if you don't do anything*.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/miru_why/status/1892478739491279153">Tweet from miru (@miru_why)</a>: @main_horse looks like the torch::empty_like call reclaims torch_output’s memory containing correct output (caching allocator) and then the kernel does almost nothing because of the ‘2d block configur...</li><li><a href="https://x.com/drjimfan/status/1892404919480832259?s=46">Tweet from Jim Fan (@DrJimFan)</a>: The coolest autonomous coding agent I&#39;ve seen recently: use AI to write better CUDA kernels to accelerate AI. AutoML is so back! The highest leverage thing you can do with your compute resources i...</li><li><a href="https://x.com/sakanaailabs/status/1892385766510338559?s=46">Tweet from Sakana AI (@SakanaAILabs)</a>: Introducing The AI CUDA Engineer: An agentic AI system that automates the production of highly optimized CUDA kernels.http://sakana.ai/ai-cuda-engineer/The AI CUDA Engineer can produce highly optimize...</li><li><a href="https://x.com/miru_why/status/">Tweet from GitHub - FixTweet/FxTwitter: Fix broken Twitter/X embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others</a>: Fix broken Twitter/X embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others - FixTweet/FxTwitter
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[edge](https://discord.com/channels/1189498204333543425/1303441437592911912/1341985412683399269)** (1 messages): 

> `Hybrid Speech Processing Application, NVIDIA Jetson Nano, Speech Separation Model, Cloud LLM Integration` 


- **Hybrid Speech Processing Application Demonstrated**: A user built a **hybrid speech processing application** for a class that deploys a speech separation model on an **NVIDIA Jetson Nano** to filter input speech based on prompts.
   - The application integrates cloud capabilities where an **LLM** decodes prompts and sends embeddings to the edge device for processing.
- **Feedback Requested on Application Report**: The user attached a report titled [Listen, Chat, and Edit on Edge](https://cdn.discordapp.com/attachments/1303441437592911912/1341985412197122159/Listen__Chat__and_Edit_on_Edge.pdf?ex=67b8a58f&is=67b7540f&hm=e5ce784faf8d568c323c01e402323a53e7e88e4367b798d3115f07821dd98acd&) and requested feedback on their project.
   - They encouraged discussion and evaluation of the project's approaches and outcomes.


  

---


### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/1341849877054881835)** (76 messages🔥🔥): 

> `Reasoning Gym Server, Spatial Reasoning Datasets, Decimal Arithmetic Enhancements, Needle in Haystack Dataset, UnslothAI's New Algorithms` 


- **Progress on Reasoning Gym Server**: Team members are finalizing the first version of the Reasoning Gym server with server and CLI tools being merged and debugged for smooth operation.
   - The goal is to enable seamless handling of diverse reasoning tasks, including potential integration of ILP tasks.
- **Search for Spatial Reasoning Datasets**: Members discussed the need for datasets that focus on **spatial reasoning** and proposed ideas for generating questions related to 3D spaces and relationships.
   - Examples include using classic puzzles such as the marble question and concepts from research papers to refine datasets.
- **Enhancements in Decimal Arithmetic**: There was a conversation around potentially reducing the maximum significant digits in decimal arithmetic configurations to ensure accurate results.
   - Members expressed that while floating point issues are known, proper handling in training could streamline performance.
- **Improvements in Needle in Haystack Dataset**: Discussions included optimizing memory usage in the Needle in Haystack dataset by potentially deferring data loading until necessary.
   - Members highlighted the importance of balancing memory efficiency with the ability to generate and retain multiple examples.
- **UnslothAI Launching New Algorithms**: A new launch from UnslothAI promises 10x longer context lengths and 90% less VRAM for training reasoning models.
   - This advancement allows for training models effectively with minimal resources, sparking excitement among team members.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2401.03991">Advancing Spatial Reasoning in Large Language Models: An In-Depth Evaluation and Enhancement Using the StepGame Benchmark</a>: Artificial intelligence (AI) has made remarkable progress across various domains, with large language models like ChatGPT gaining substantial attention for their human-like text-generation capabilitie...</li><li><a href="https://www.interconnects.ai/p/artifacts-7">The latest open artifacts (#7): Alpaca era of reasoning models, China&#x27;s continued dominance, and tons of multimodal advancements</a>: Artifacts Log 7. It&#x27;ll continue to be a fun spring for AI researchers and practitioners.</li><li><a href="https://github.com/open-thought/reasoning-gym/issues/173">Add &quot;emoji mystery&quot; dataset · Issue #173 · open-thought/reasoning-gym</a>: Create a dataset which generates &quot;emoji mystery&quot; questions which contain a hidden messages encoded in a unicode emoji via &quot;variation selectors&quot;. See Andrej Karpathy&#39;s x thread,...</li><li><a href="https://x.com/unslothai/status/1892640995847901684">Tweet from Unsloth AI (@UnslothAI)</a>: Today, we’re launching new algorithms that enable 10x longer context lengths & 90% less VRAM for training Reasoning Models (GRPO).Using Unsloth, you can now train your own reasoning model with just 5G...</li><li><a href="https://github.com/Fangjun-Li/SpatialLM-StepGame">GitHub - Fangjun-Li/SpatialLM-StepGame: Codes and data for AAAI-24 paper &quot;Advancing Spatial Reasoning in Large Language Models: An In-depth Evaluation and Enhancement Using the StepGame Benchmark&quot;</a>: Codes and data for AAAI-24 paper &quot;Advancing Spatial Reasoning in Large Language Models: An In-depth Evaluation and Enhancement Using the StepGame Benchmark&quot; - Fangjun-Li/SpatialLM-StepGame</li><li><a href="https://github.com/open-thought/reasoning-gym/pull/174">Add decimal number comparison by Adefioye · Pull Request #174 · open-thought/reasoning-gym</a>: Python generator to compare decimals</li><li><a href="https://github.com/open-thought/reasoning-gym/blob/main/reasoning_gym/arithmetic/number_format.py">reasoning-gym/reasoning_gym/arithmetic/number_format.py at main · open-thought/reasoning-gym</a>: procedural reasoning datasets. Contribute to open-thought/reasoning-gym development by creating an account on GitHub.</li><li><a href="https://github.com/open-thought/reasoning-gym/pull/170">Adds Needle in a Haystack problems by Miserlou · Pull Request #170 · open-thought/reasoning-gym</a>: Ex:Boedyn is crazy about burritos. Tyrnan regrets geography. Deryn commends soup. David-Jay extols dusting the furniture. Malikye exults literature. Oluwadamilare celebrates electric scooters. Nai...
</li>
</ul>

</div>
  

---


### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1341828814451376130)** (130 messages🔥🔥): 

> `Comparison of SD and Flux, ControlNet Applications, Custom Model Creation, Using Scribbles for Image Generation, GPU Recommendations for AI Tools` 


- **Choosing between Stable Diffusion and Flux**: Members discussed that **Stable Diffusion (SD)** is currently more refined than **Flux**, though **Flux** is still in development.
   - One member advised looking at example images to determine which model aligns best with personal preferences.
- **ControlNet for Image Poses**: ControlNet can effectively utilize depth maps or skeleton wireframes to generate images based on poses, managing adjustments like 'hand in front' or 'hand behind'.
   - It was noted that using control methods can allow for more accurate and creative image generation from provided poses.
- **Inquiry about Custom Model Creation**: A user expressed a desire to hire someone skilled in both **Stable Diffusion** and traditional art for creating a customized model and prompt style.
   - Others questioned the practicality of such a request, suggesting that learning to create the model personally is more beneficial and cost-effective.
- **Scribbles to Image Generation Workflow**: A user shared a workflow involving using rough sketches on an iPad to guide the AI in generating images, seeking advice on transitioning from scribbles to finished images.
   - They acknowledged the utility of img2img processes but were uncertain about how to start with simplistic doodles.
- **GPU Requirements for Image Generation Tools**: Discussion highlighted that **Nvidia GPUs** remain the recommended choice for running **Stable Diffusion** efficiently, while AMD options may face performance issues.
   - Users shared their current GPU setups and discussed the compatibility of different models with GPU capabilities.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/jiseok-kim-jiseok-big-ocean-bigocean-kpop-gif-16919206117458777151">Jiseok Kim Jiseok GIF - Jiseok Kim jiseok Big ocean - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.youtube.com/watch?v=n233GPgOHJg">Stable Diffusion Models Explained Once and for All (1.5, 2, XL, Cascade, 3)</a>: In this video, I explain the 5 different model families of Stable Diffusion.Did I get anything wrong or leave something out? Let me know.Chapters:00:00 Intro...</li><li><a href="https://github.com/LykosAI/StabilityMatrix/">GitHub - LykosAI/StabilityMatrix: Multi-Platform Package Manager for Stable Diffusion</a>: Multi-Platform Package Manager for Stable Diffusion - LykosAI/StabilityMatrix</li><li><a href="https://github.com/mcmonkeyprojects/SwarmUI?tab=readme-ov-file#installing-on-windows">GitHub - mcmonkeyprojects/SwarmUI: SwarmUI (formerly StableSwarmUI), A Modular Stable Diffusion Web-User-Interface, with an emphasis on making powertools easily accessible, high performance, and extensibility.</a>: SwarmUI (formerly StableSwarmUI), A Modular Stable Diffusion Web-User-Interface, with an emphasis on making powertools easily accessible, high performance, and extensibility. - mcmonkeyprojects/Swa...</li><li><a href="https://github.com/CS1o/Stable-Diffusion-Info/wiki/Webui-Installation-Guides">Webui Installation Guides</a>: Stable Diffusion Knowledge Base (Setups, Basics, Guides and more) - CS1o/Stable-Diffusion-Info
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1341964090435960833)** (6 messages): 

> `GPU scheduler optimization, AI CUDA Engineer, ARENA 4.0 program` 


- **Seeking Dataset Recommendations for ML Project**: A member shared their focus on **GPU scheduler optimization** using **deep reinforcement learning** as part of their ML studies, seeking advice on datasets for benchmarks.
   - They specifically asked for recommendations, indicating their current challenge in finding suitable datasets.
- **Introducing AI CUDA Engineer for Optimization**: A resource was shared about the **AI CUDA Engineer**, an automatic framework for optimizing CUDA kernels with a reported **>90% success rate** for translating PyTorch to CUDA.
   - Despite its effectiveness, there are concerns that the results may be **spurious/error-ridden** according to the community consensus.
- **Discussion on Data Quality from AI CUDA Engineer**: A member pointed out that the dataset containing kernels generated by the **AI CUDA Engineer** could be flawed due to potential inaccuracies in generated outputs.
   - This sparked a debate about the reliability of the baseline implementations associated with the dataset.
- **Call for Contact Regarding ARENA 4.0**: One user expressed a desire to connect with the creator of the **ARENA 4.0 program**, asking for a direct message.
   - This indicates a need for collaboration or assistance related to that specific project.



**Link mentioned**: <a href="https://pub.sakana.ai/ai-cuda-engineer">The AI CUDA Engineer 👷</a>: no description found

  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1341983181078921226)** (81 messages🔥🔥): 

> `AI CUDA Engineer, CUDA and PyTorch performance, LLM Optimization, Clockwork RNN, Model training insights` 


- **AI CUDA Engineer's impressive claims**: Introducing the [AI CUDA Engineer](http://sakana.ai/ai-cuda-engineer/), an AI system that claims to achieve **10-100x speedup** in CUDA kernel production over PyTorch, alongside a dataset of **17,000+ kernels**.
   - However, discussions among members raised concerns about the evaluation's accuracy, with claims of previous misrepresentations in similar projects indicating ongoing skepticism.
- **CUDA kernel evaluation flaws**: Critique emerged around the kernel evaluation methods, revealing that the **150x speedup kernel** purportedly utilized memory reuse and had fundamental **bugs** in implementation.
   - Members expressed doubts about the reliability of these kernels, leading to a broader discussion about the potential prevalence of issues within the sample provided.
- **Exploration of LLM Compilers**: A conversation unfolded around the concept of **LLM-compilers**, where members speculate whether an LLM could translate high-level PyTorch code into optimized machine code for specific setups.
   - While the idea intrigued members, there was a consensus that substantial challenges, especially due to the lack of a common instruction set, could impede progress.
- **Clockwork RNN and Transformer architectures**: Discussion arose regarding the **Clockwork RNN**, a revised architecture that improves performance by using separate modules for various input granularities, akin to predictions in transformers.
   - Members debated the viability of such architectures being utilized in future models, including the application of dilated convolutions and attention mechanisms.
- **Need for experimental model checkpoints**: Conversations indicated a demand for **checkpoints** in models such as the Muon optimizer, emphasizing that direct comparisons with traditional models could yield insightful results.
   - The potential benefits of semi-blackbox hyperoptimization and its implications for training strategies were also highlighted, calling for further exploration in theoretical frameworks.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/SakanaAILabs/status/1892385766510338559">Tweet from Sakana AI (@SakanaAILabs)</a>: Introducing The AI CUDA Engineer: An agentic AI system that automates the production of highly optimized CUDA kernels.http://sakana.ai/ai-cuda-engineer/The AI CUDA Engineer can produce highly optimize...</li><li><a href="https://arxiv.org/abs/2502.10927">The underlying structures of self-attention: symmetry, directionality, and emergent dynamics in Transformer training</a>: Self-attention is essential to Transformer architectures, yet how information is embedded in the self-attention matrices and how different objective functions impact this process remains unclear. We p...</li><li><a href="https://arxiv.org/abs/1402.3511">A Clockwork RNN</a>: Sequence prediction and classification are ubiquitous and challenging problems in machine learning that can require identifying complex dependencies between temporally distant inputs. Recurrent Neural...</li><li><a href="https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero/blob/main/ORZ_paper.pdf">Open-Reasoner-Zero/ORZ_paper.pdf at main · Open-Reasoner-Zero/Open-Reasoner-Zero</a>: Official Repo for Open-Reasoner-Zero. Contribute to Open-Reasoner-Zero/Open-Reasoner-Zero development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1341827572085821552)** (4 messages): 

> `Logit Lens, Tuned Lens, Transformers Analysis, Computer Security Analogy, Average-case Goals` 


- **Logit Lens and Tuned Lens Promise**: Discussion highlighted the potential of the **Logit Lens** and **Tuned Lens** for analyzing transformers and recurrent models, suggesting that there’s unexplored value in understanding how models approach problems at each step.
   - Exploring this further could yield insights into long-form Chain of Thought reasoning.
- **Challenges in Analyzing Complex Questions**: A member expressed that addressing the specific question raised in a tweet is difficult, comparing it to issues in **computer security** like fuzzing and identifying backdoors.
   - This highlights the complexity and intricacies involved in discerning meaningful patterns in model behavior.
- **Intuition on Average-case Performance**: One participant opined that aiming for **average-case performance** may be more attainable, as it doesn't rely on hidden cues but rather natural training configurations.
   - This perspective emphasizes the importance of focusing on accessible latents rather than elusive outlier situations.


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1342052538655965185)** (8 messages🔥): 

> `lm-eval-harness, runtime benchmarks, model path errors, lm studio, task path errors` 


- **Seeking benchmarks with lm-eval-harness**: A member asked for guidance on using **lm-eval-harness** to benchmark a model running locally on **lm studio** while also assessing the PC's performance.
   - *stellaathena* mentioned that **lm-eval** measures performance, not runtime.
- **Runtime benchmarks already obtained**: The member clarified that they have already gathered runtime benchmarks using **llm perf** and are now facing errors with **eval harness** related to task paths.
   - *stellaathena* requested the command being run to better assist.
- **Errors with model path in lm_eval command**: The member shared a command they're using but experienced repeated issues due to an incorrect **model path** despite attempts to change it.
   - They provided their command, specifying the model path and additional parameters they were using.
- **Request for private assistance**: The member expressed a desire to connect via private messages for more personalized assistance regarding their issues.
   - This indicates their preference for one-on-one support to troubleshoot the challenges they are facing.
- **Trying different model completions**: They mentioned experimenting with both **openai-completions** and **local-chat completions** for their benchmarking efforts.
   - This suggests a broader search for solutions amidst the task execution difficulties.


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1341896383501828126)** (12 messages🔥): 

> `Evo2 Genome Models, Llama 3.2 Comparison, NCCL_BUFFSIZE Adjustments` 


- **Evo2 Genome Models leverage GPT-NeoX**: The new **Evo2 genome models** were trained using [a library that builds on GPT-NeoX](https://github.com/Zymrael/savanna). This confirms a strong integration of contemporary models into existing frameworks.
   - *Very rewarding to hear* that the announcement is well-received within the community.
- **Llama 3.2 shows TPS differences**: A member compared the **Llama 3.2 1B configuration** across NeMo and NeoX, noting **21.3K TPS** for NeoX versus **25-26K TPS** for NeMo. The shared [configuration file](https://github.com/aflah02/gpt-neox/blob/olmo-support/configs/hubble/Speed_Exps/1_1B_Baseline_BS_8_GAS_8_No_Activation_Checkpointing_GQA_Llama_3_2_Fusions_All_4_FA_Swiglu.yml) outlines the experimental setup.
   - The performance insights can help others optimize their setups as they refer to the **[WandB run](https://wandb.ai/aflah/hubble-speed-testing/runs/nioywj5f?nw=nwuseraflah)** for detailed metrics.
- **Adjustment of NCCL_BUFFSIZE discussed**: A member raised the *curiosity* regarding the **NCCL_BUFFSIZE**, suggesting a value of **2097152**. It is considered beneficial for multi-GPU communication, especially when using InfiniBand.
   - The suggestion to adjust the buffer size independently from DeepSpeed's bucket size implies that best practices can enhance performance in complex setups.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/aflah02/gpt-neox/blob/olmo-support/configs/hubble/Speed_Exps/1_1B_Baseline_BS_8_GAS_8_No_Activation_Checkpointing_GQA_Llama_3_2_Fusions_All_4_FA_Swiglu.yml">gpt-neox/configs/hubble/Speed_Exps/1_1B_Baseline_BS_8_GAS_8_No_Activation_Checkpointing_GQA_Llama_3_2_Fusions_All_4_FA_Swiglu.yml at olmo-support · aflah02/gpt-neox</a>: An implementation of model parallel autoregressive transformers on GPUs, based on the Megatron and DeepSpeed libraries - aflah02/gpt-neox</li><li><a href="https://wandb.ai/aflah/hubble-speed-testing/runs/nioywj5f?nw=nwuseraflah">aflah</a>: Weights & Biases, developer tools for machine learning
</li>
</ul>

</div>
  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1341922645066649671)** (12 messages🔥): 

> `Podcast TTS Issues, Inviting Non-Google Users to Notebooks, Tesla Autonomous Driving Patent Insights, Using NotebookLM for Homeschooling, AI's Understanding of Literary Works` 


- **Podcast TTS Issues**: A user struggled with getting the TTS function to read their input correctly, trying various prompts without success.
   - They expressed frustration over the lack of cooperation in having the podcast host read the text as intended.
- **Inviting Non-Google Users to Notebooks**: A member inquired about the possibility of inviting someone without a Google account to access a notebook, similar to Google Docs functionality.
   - This raised questions on alternative access methods for collaboration in Notebook LM.
- **Tesla Autonomous Driving Patent Insights**: A user explored Tesla's Autonomous Driving AI after a recent patent grant, mentioning key technologies like **Lidar**, **Radar**, and **Ultrasonics**.
   - They created a podcast discussing their findings, highlighting a **free** article available on their Patreon for listeners.
- **Using NotebookLM for Homeschooling**: A user shared their positive experience utilizing NotebookLM alongside Gemini for homeschooling their child, comparing it to having highly skilled assistants.
   - They attributed significant help in executing their teaching efforts through this integrated approach.
- **AI's Understanding of Literary Works**: Multiple users expressed frustrations regarding the AI's misinterpretations of their writing and character details, citing various examples of errors.
   - One noted that even when presented with evidence, the AI often refused to acknowledge corrections, leading to conflict with the narrative.


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1341819474499866654)** (97 messages🔥🔥): 

> `NotebookLM Permissions, Audio Features, Notebook Sharing Issues, Source Limitations, User Experience on NotebookLM` 


- **Navigating NotebookLM Permissions**: Users discussed how to share notebooks and some reported difficulties finding the share button on the Plus version, highlighting possible restrictions in user roles.
   - One user suggested filing a bug report regarding the missing share button functionality.
- **Utilizing Audio Overviews in Courses**: A user inquired about using the Audio 'Deep Dive' outputs for academic purposes, and confirmed that sharing within EDU accounts is allowed.
   - Guidance on generating Audio Overviews was provided, indicating they reflect the source content and not the AI hosts' opinions.
- **Embedding Features and Organization Requests**: Asking for folder organization options was a recurring theme, with users expressing a need for improved management of notes and notebooks.
   - The request for this feature has been logged internally, but no timeline was provided for its implementation.
- **Addressing Upload Challenges**: Users reported issues with uploading various file types including PDFs and audio files, speculating on potential bugs.
   - Tests were suggested to upload different files or use Google Docs to manage content effectively.
- **Clarifying Source Usage Policies**: Discussion around the limitations of using news sources as inputs for NotebookLM raised questions regarding accepted source types.
   - One user suggested a workaround by copying text directly rather than using links when facing limitations with recognized news outlets.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://support.google.com/notebooklm/answer/15731776?hl=en&ref_topic=14272601&sjid=7303781756764289573-NC">Audio Overviews - NotebookLM Help</a>: no description found</li><li><a href="https://notebooklm.google.com/?hl=ar">تسجيل الدخول - حسابات Google</a>: no description found</li><li><a href="https://youtu.be/EGhXtFjzcJY">NotebookLM - Research YouTube Comments and sentiments For FREE!</a>: Comprehensive NotebookLM playlist - https://www.youtube.com/playlist?list=PL-HkokgcYrl5SrKYeVo28JA4OMPbslhA8🚀  Ever wished you could pull thousands of YouTu...
</li>
</ul>

</div>
  

---


### **Torchtune ▷ #[announcements](https://discord.com/channels/1216353675241590815/1216353675241590818/1341892739289321583)** (1 messages): 

> `Torchtune Roadmap, PyTorch Roadmaps` 


- **Torchtune Roadmap for H1 2025 Released**: The official **Torchtune roadmap** for the first half of the year has been posted on [PyTorch dev-discuss](https://drive.google.com/file/d/1mKENBMrdMzMQQG1kn43Il64qWPB9uP21/view). This document outlines the essential directions and projects planned for Torchtune in this timeframe.
   - Members are encouraged to check out the roadmap, as it details **key initiatives** and strategies crucial for the Torchtune development.
- **Comprehensive Overview of PyTorch Roadmaps**: The full set of **PyTorch roadmaps** for various projects is also accessible on [dev-discuss](https://dev-discuss.pytorch.org/t/meta-pytorch-team-2025-h1-roadmaps/2794). This release showcases an array of exciting developments and ongoing work across the entire PyTorch platform this half.
   - This broader overview demonstrates the collaborative efforts of the PyTorch team to innovate and advance their technologies.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://drive.google.com/file/d/1mKENBMrdMzMQQG1kn43Il64qWPB9uP21/view">[PUBLIC] Torchtune - H1 2025 Roadmap.pdf</a>: no description found</li><li><a href="https://dev-discuss.pytorch.org/t/meta-pytorch-team-2025-h1-roadmaps/2794">Meta PyTorch Team 2025 H1 Roadmaps</a>: PyTorch Community,  The Meta team are happy to make our 2025 H1 roadmaps available. We plan on a half year basis and globally optimize across the things we do for our users here at Meta and across the...
</li>
</ul>

</div>
  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1341825832188121161)** (43 messages🔥): 

> `VRAM requirements with packing, Roadmap updates, Emerging attention techniques, Pruning strategies for LLMs, Exotic transformer architectures` 


- **VRAM demands skyrocket with packed sequences**: When using packing with a dataset at **max_tokens** length, **VRAM requirements** drastically increase, leading to *out-of-memory* (OOM) errors at **16K sequence lengths**.
   - A user noted that with packing set to false, memory usage was at **30GB**, showcasing the vast difference in resource needs.
- **Roadmap posted to PyTorch dev-discuss**: The roadmap for PyTorch has been shared on [Google Drive](https://drive.google.com/file/d/1mKENBMrdMzMQQG1kn43Il64qWPB9uP21/view), with an emphasis on upcoming conference deadlines.
   - Despite being a work in progress, feedback has been received positively with *commitment to continuous improvements*.
- **Seeking opinions on exotic attention mechanisms**: Discussion centered around the priority of **exotic transformer techniques** like *sparse attention* and *attention compression*, which enhance **efficiency in sequence scaling**.
   - Contributions from researchers suggest that while interest exists, there are reservations about integrating new research due to existing methodologies.
- **Pruning techniques for large language models**: A new recipe to support **width and depth pruning** for LLMs (Large Language Models) is in development, encouraged by the recent paper on *pruning alternatives*.
   - This methodology could enable compressing models significantly, improving resource utilization without full retraining.
- **Clarifications on roadmap objectives**: Feedback regarding **KR2.4** was noted, highlighting a lack of clear state-of-the-art (SOTA) examples in its assessment, such as *Codestral* and *Jamba*.
   - The roadmap's objectives emphasize interest in long-term innovation while prioritizing core tasks, reflecting the intention to adapt as the field evolves.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2407.14679">Compact Language Models via Pruning and Knowledge Distillation</a>: Large language models (LLMs) targeting different deployment scales and sizes are currently produced by training each variant from scratch; this is extremely compute-intensive. In this paper, we invest...</li><li><a href="https://github.com/pytorch/torchtune/issues/2392">`torch._inductor.exc.LoweringException: NoValidChoicesError` using torch 2.6.0 · Issue #2392 · pytorch/torchtune</a>: Error [rank0]: raise NoValidChoicesError( [rank0]: torch._inductor.exc.LoweringException: NoValidChoicesError: No choices to select, please consider adding ATEN into max_autotune_gemm_backends conf...</li><li><a href="https://github.com/pytorch/torchtune/blob/e6cba25">GitHub - pytorch/torchtune at e6cba2532d51a53936c7646bd4cdaa6b2b57ed66</a>: PyTorch native post-training library. Contribute to pytorch/torchtune development by creating an account on GitHub.</li><li><a href="https://github.com/pytorch/torchtune/blob/e6cba2532d51a53936c7646bd4cdaa6b2b57ed66/torchtune/modules/attention_utils.py#L35">torchtune/torchtune/modules/attention_utils.py at e6cba2532d51a53936c7646bd4cdaa6b2b57ed66 · pytorch/torchtune</a>: PyTorch native post-training library. Contribute to pytorch/torchtune development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1341836804399038566)** (15 messages🔥): 

> `Judge Framework for Online DPO, AdamWScheduleFree as Default Optimizer, Pruning & Checkpointing Utilities, Integration of Torchtune with Gymnasium, Intercode for LLMs` 


- **Feedback Requested on Judge Framework RFC**: A member seeks feedback on their [RFC for a judge framework implementation](https://github.com/pytorch/torchtune/issues/2413) intended for online DPO, aiming to contribute to the dev branch if reasonable.
   - The [TRL Judges Doc](https://trl-docs.com) conceptually supports multiple judges for RLHF methods.
- **AdamWScheduleFree might serve as an Optimizer**: Discussion arose on the potential of **AdamWScheduleFree** as a default optimizer for **llama3.1 8B DPO**, with testing conducted across **2 nodes with 16 GPUs**.
   - A workaround for previous issues with fsdp was suggested, requiring adjustments in the full-dpo Python script.
- **Pruning and Checkpointer Utilities in Pull Request**: A member highlighted the [pull request on checkpointer utilities](https://github.com/joecummings/torchtune/pull/2) that includes a feature to get the latest checkpoint in a given directory.
   - There’s an emphasis on reviewing the contribution to ensure its alignment with existing utilities.
- **Questioning Gymnasium's Suitability for RL with LLMs**: A query was made regarding ongoing work on integrating **Torchtune** with **Gymnasium**, leading to a discussion on compatibility with LLMs.
   - Concerns were raised about Gymnasium’s design not aligning well with the unique requirements of LLMs, especially regarding environment actions and observations.
- **Exploration of Intercode for LLM Integration**: Members explored the possibility of using [Intercode](https://github.com/princeton-nlp/intercode) for enhancing RL tasks suited for LLMs, questioning its interface's effectiveness.
   - The conversation revealed skepticism about combining gym-like interfaces for LLM projects and recognized the need for further development in this area.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://ysymyth.github.io)">no title found</a>: no description found</li><li><a href="https://github.com/pytorch/torchtune/issues/2413">[RFC] Judge Framework and Online DPO · Issue #2413 · pytorch/torchtune</a>: TRL has a concept of multiple different judges which can be used in various online RLHF type methods, see the TRL Judges Doc. As a starting point, we could implement just a pairwise judge that coul...</li><li><a href="https://github.com/pytorch/torchtune/blob/main/recipes/configs/llama3_1/8B_full_dpo.yaml#L64-L72">torchtune/recipes/configs/llama3_1/8B_full_dpo.yaml at main · pytorch/torchtune</a>: PyTorch native post-training library. Contribute to pytorch/torchtune development by creating an account on GitHub.</li><li><a href="https://github.com/princeton-nlp/intercode">GitHub - princeton-nlp/intercode: [NeurIPS 2023 D&amp;B] Code repository for InterCode benchmark https://arxiv.org/abs/2306.14898</a>: [NeurIPS 2023 D&amp;B] Code repository for InterCode benchmark https://arxiv.org/abs/2306.14898 - princeton-nlp/intercode</li><li><a href="https://github.com/joecummings/torchtune/pull/2">feat: get_latest_checkpoint for checkpointer utils by bogdansalyp · Pull Request #2 · joecummings/torchtune</a>: Added get_latest_checkpoint    &amp;quot;&amp;quot;&amp;quot;    Returns the latest checkpoint in the given directory.    The pattern argument is a regular expression that matches the epoch number in ...
</li>
</ul>

</div>
  

---


### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/1341848371287162951)** (4 messages): 

> `Multi-step PPO, Tool Learning, Reward Shaping, StepTool Framework, UltraScale Playbook` 


- **Exploration of Multi-step PPO Approaches**: A user inquired about papers on **multi-step PPO**, which involves multiple sequential calls to LLMs with the reward assessed only after several interactions.
   - They suggested researching in the broader domain of tool learning and reward shaping.
- **StepTool Framework for Tool Learning**: A key paper shared discusses **StepTool**, a new step-grained reinforcement learning framework that enhances multi-step tool use capabilities of LLMs, detailing its components of **Step-grained Reward Shaping** and **Step-grained Optimization**.
   - The paper emphasizes the need to consider the decision-making complexities of multi-step contexts in the context of tool learning.
- **Hugging Face UltraScale Playbook Released**: A user shared a [link to the UltraScale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) hosted on Hugging Face, which was described as **refreshing**.
   - This playbook is likely aimed at guiding users in scaling usage of models within a practical framework.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/nanotron/ultrascale-playbook">The Ultra-Scale Playbook - a Hugging Face Space by nanotron</a>: no description found</li><li><a href="https://arxiv.org/abs/2410.07745">StepTool: Enhancing Multi-Step Tool Usage in LLMs through Step-Grained Reinforcement Learning</a>: Despite powerful text generation capabilities, large language models (LLMs) still need to learn how to utilize external tools to solve complex tasks, a process known as tool learning. Existing methods...
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1341832770216919090)** (49 messages🔥): 

> `Baseten Series C funding, Mastra JS agent framework, Arize AI Series C funding, Lambda $480M Series D, OpenAI's growing user base` 


- **Baseten Secures $75M Series C**: Baseten announced a successful $75 million Series C funding round, co-led by @IVP and @sparkcapital, highlighting 2025 as the year of inference for AI technologies.
   - New investors like Dick Costolo and Adam Bain from @01Advisors joined the round, emphasizing Baseten's growth and potential in the AI space.
- **Mastra Opens Up AI Agent Framework**: The open-source project Mastra offers a JavaScript SDK for building AI agents on top of Vercel’s AI SDK, focusing on ease of use and integration with workflows.
   - Developers expressed interest in the capabilities of Mastra agents for complex task execution, such as accessing third-party APIs and custom functions.
- **Arize AI Raises $70M Series C**: Arize AI secured $70 million in Series C funding to enhance AI evaluation and observability across generative and decision-making models.
   - Their mission is to ensure AI agents operate reliably at scale, addressing the challenges posed by new developments in AI technology.
- **Lambda's $480M Series D Funding**: Lambda announced a notable $480 million Series D funding round led by Andra Capital and SGW, showcasing the company's growth in AI computing resources.
   - With this funding, Lambda aims to strengthen its position as a cloud service developed for AI demands and capabilities.
- **OpenAI Surpasses 400M Users**: OpenAI recently reported over 400 million weekly active users on ChatGPT, reflecting a significant growth of 33% in under three months.
   - The upcoming GPT-5 promises unlimited use for free users and is expected to unify existing models, intensifying the competition in the AI space.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://mastra.ai/docs/agents/00-overview">Creating and Calling Agents | Agent Documentation | Mastra</a>: no description found</li><li><a href="https://x.com/huybery/status/1892628963878486233">Tweet from Binyuan Hui (@huybery)</a>: &lt;think&gt;…&lt;/think&gt;Binyuan is cooking…</li><li><a href="https://huggingface.co/spaces/nanotron/ultrascale-playbook">The Ultra-Scale Playbook - a Hugging Face Space by nanotron</a>: no description found</li><li><a href="https://x.com/drjimfan/status/1892404919480832259?s=46">Tweet from Jim Fan (@DrJimFan)</a>: The coolest autonomous coding agent I&#39;ve seen recently: use AI to write better CUDA kernels to accelerate AI. AutoML is so back! The highest leverage thing you can do with your compute resources i...</li><li><a href="https://x.com/Figure_robot/status/1892577871366939087">Tweet from Figure (@Figure_robot)</a>: Meet Helix, our in-house AI that reasons like a humanRobotics won&#39;t get to the home without a step change in capabilitiesOur robots can now handle virtually any household item:</li><li><a href="https://x.com/stephenbalaban/status/1892275552171737220?s=46">Tweet from stephen balaban (@stephenbalaban)</a>: Lambda is a cloud designed for the age of AI. Today, we announced a $480M Series D co-led by Andra Capital and SGW with participation from NVIDIA, Andrej Karpathy, In-Q-Tel, ARK Invest, and others.</li><li><a href="https://x.com/bingxu_/status/1892405811596710392">Tweet from Bing Xu (@bingxu_)</a>: I quickly take a look of their report on phone, there are a few misleading parts: 1. Torch C++ code is not CUDA kernel, it is calling CUDNN under hood.2. The highlighted example Conv3D GroupNorm, conv...</li><li><a href="https://x.com/loubnabenallal1/status/1892278622104215894?s=46">Tweet from Loubna Ben Allal (@LoubnaBenAllal1)</a>: The nanotron team just released The Ultra-Scale PlayBook with everything you need to know about LLM pretraining at smol and large scales  https://huggingface.co/spaces/nanotron/ultrascale-playbook</li><li><a href="https://x.com/leonardtang_/status/1892243653071908949">Tweet from Leonard Tang (@leonardtang_)</a>: First came pre-training scaling; then came inference-time scaling.Now comes judge-time scaling.Despite progress in AI through scaled inference-time compute, AI remains unreliable in open-ended, non-ve...</li><li><a href="https://www.youtube.com/watch?v=L89GzWEILkM">AI Engineer Summit 2025 - AI Leadership (Day 1)</a>: Scheduled Talks (All times EST):9:00am - Show opener 9:07AM - Beyond the Consensus: Navigating AI&#39;s Frontier in 2025 - Grace Isford of Lux Capital9:28AM - Ho...</li><li><a href="https://x.com/stefania_druga/status/1892669203657736600?s=46">Tweet from Stefania Druga (@Stefania_druga)</a>: Great list! Much work left to be done on interpretabilityQuoting swyx 🗽 NYC (@aiDotEngineer) (@swyx) First time I’ve seen @AnthropicAI lay out its top priorities like thisfocusing more on mechinterp ...</li><li><a href="https://x.com/stephenbalaban/status/1892403855817859079">Tweet from stephen balaban (@stephenbalaban)</a>: This was Lambda’s first supercomputer. Built in 2015 to run the Dreamscope style transfer app. It was 32 NVIDIA GTX 980s. We were running out of cash from a $40K/mo cloud bill. We called it the Deep B...</li><li><a href="https://x.com/togethercompute/status/1892609235789422724">Tweet from Together AI (@togethercompute)</a>: 📣 Today we&#39;re announcing our $305M Series B funding led by @generalcatalyst  and co-led by @p7ventures, with participation from a distinguished group of global institutional and strategic investo...</li><li><a href="https://arize.com/blog/arize-ai-raises-70m-series-c-to-build-the-gold-standard-for-ai-evaluation-observability/">Arize AI Raises $70M Series C to Build the Gold Standard for AI Evaluation &amp; Observability</a>: Learn how we&#039;re shaping the future of trustworthy LLMs &amp; AI agents, and what&#039;s next for Arize in our Series C announcement.</li><li><a href="https://x.com/nutlope/status/1892619157662806272?s=46">Tweet from Hassan (@nutlope)</a>: Excited to share we raised $305M at a 3.3B valuation!It&#39;s been awesome witnessing all the growth – hitting 450k devs & 200+ models supported! Also, we&#39;re hiring!Quoting Together AI (@togetherc...</li><li><a href="https://x.com/dhravyashah/status/1892363590671233255?s=46">Tweet from Dhravya Shah (@DhravyaShah)</a>: Introducing apple-mcp - http://git.new/apple-mcpone simple command to give LLMs access to a bunch of apple-native tools like- contacts- notes- iMessagesand more (soon)just add this to your claude desk...</li><li><a href="https://x.com/yoheinakajima/status/1892257339400737087?s=46">Tweet from Yohei (@yoheinakajima)</a>: i’ve said this in private convos so I’ll say it here…“better memory” is the final unlock we need to get truly better agents, and 2025 is when we’ll see more of thiswe have strong reasoning, tools for ...</li><li><a href="https://x.com/skalskip92/status/1892233630577000820?s=46">Tweet from SkalskiP (@skalskip92)</a>: this guy took my football AI project and used it for automated advance match analytics; focusing on through passes firstQuoting SkalskiP (@skalskip92) football AI code is finally open-source- player d...</li><li><a href="https://x.com/bradlightcap/status/1892579908179882057?s=46">Tweet from Brad Lightcap (@bradlightcap)</a>: chatgpt recently crossed 400M WAU, we feel very fortunate to serve 5% of the world every week2M+ business users now use chatgpt at work, and reasoning model API use is up 5x since o3 mini launchwe&#39...</li><li><a href="https://x.com/basetenco/status/1892259130540179863">Tweet from Baseten (@basetenco)</a>: 2025 is the year of inference.We&#39;re thrilled to announce our $75m Series C co-led by @IVP  and @sparkcapital  with participation from @GreylockVC, @conviction, @basecasevc, @southpkcommons  and @l...</li><li><a href="https://x.com/thom_wolf/status/1892273133547078036?s=46">Tweet from Thomas Wolf (@Thom_Wolf)</a>: After 6+ months in the making and burning over a year of GPU compute time, we&#39;re super excited to finally release the &#34;Ultra-Scale Playbook&#34;Check it out here: http://hf.co/spaces/nanotron/...</li><li><a href="https://news.ycombinator.com/item?id=43103073">Show HN: Mastra – Open-source JS agent framework, by the developers of Gatsby | Hacker News</a>: no description found</li><li><a href="https://x.com/miru_why/status/1892500715857473777?s=46">Tweet from miru (@miru_why)</a>: turns out the AI CUDA Engineer achieved 100x speedup by… hacking the eval scriptQuoting main (@main_horse) @miru_why I believe there is something wrong with their kernel -- it seems to &#39;steal&#39;...</li><li><a href="https://x.com/giffmana/status/1892510741242036468?s=46">Tweet from Lucas Beyer (bl16) (@giffmana)</a>: o3-mini-high figured out the issue with @SakanaAILabs CUDA kernels in 11s.It being 150x faster is a bug, the reality is 3x slower.I literally copy-pasted their CUDA code into o3-mini-high and asked &#...
</li>
</ul>

</div>
  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1341824207218020382)** (11 messages🔥): 

> `SSE implementation, Debugging Glama hosted models, Puppeteer installation issues, Docker requirements, Remote MCP feature timeline` 


- **SSE implementation is live**: A member confirmed they have successfully implemented **/sse** for their project, which can be viewed in a specific channel.
   - This addition highlights ongoing improvements within the MCP functionality.
- **Glama hosted models debugging woes**: Another member shared they are encountering issues with the cursor not finding tools while debugging **Glama hosted models**.
   - *99% of the issue* is attributed to incorrect use of node paths and potentially missing quotes.
- **Puppeteer installation confusion**: A new member sought help with **Puppeteer installation**, specifically regarding running a Docker build command.
   - Guidance was provided to navigate to the correct parent directory and clarify the purpose of the `.` in the Docker command.
- **Docker essentials clarified**: It's confirmed that **Docker** needs to be installed prior to using it, with one member noting that the command was not found.
   - Furthermore, an account is not required for installation since Docker is free software.
- **Inquiry about Remote MCP timeline**: A user inquired about the timeline for the **Remote MCP** feature, expressing interest in its potential applications for their company.
   - Another member responded that existing support for **SSE** and **websocket** MCP transports is already available.


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1341837709735493713)** (26 messages🔥): 

> `Dockerized MCP Servers, Sage support for LLM Providers, Glama Integration, MCP Python Interpreter, Roots in MCP Clients` 


- **How to Deploy Dockerized MCP Servers**: A member shared a blog post detailing the steps to deploy Dockerized **MCP servers** and highlighted challenges with environment setups across architectures. They noted that **Docker** can help ensure consistency across development environments.
   - The blog also pointed to a list of [reference MCP Servers](https://github.com/modelcontextprotocol/server) available for developers looking to implement MCP functions.
- **Sage LLM Support Queries**: There was discussion about when **Sage** would support additional LLM providers like **OpenRouter**, with a hint at potential API additions being awaited. It was indicated that **Glama** can already be integrated directly into Sage.
   - One member expressed a desire to align the two projects more closely after recognizing shared interests and goals.
- **MCP Python REPL Implementation**: A member introduced a simple **Python REPL** implementation that supports **STDIO** for MCP, sharing a link to their [GitHub repository](https://github.com/evalstate/mcp-py-repl). They also provided the latest image for those interested.
   - Another member inquired about **IPython support**, to which the developer indicated it might be straightforward to add, suggesting further development on this feature.
- **Matplotlib Support in MCP**: Discussion emerged around integrating **matplotlib/pyplot** support for rendering plots in MCP, similar to Jupyter. The creator confirmed that **matplotlib**, **seaborn**, and **numpy** are already included in the implementation.
   - They mentioned returning plot images as .png files, discussing whether these could be returned directly to the MCP client.
- **Roots Usage in MCP Clients**: A conversation broke out concerning the use of **roots** in MCP and existing client implementations. One member noted it’s easy to return file results with an MCP server, but expressed curiosity about the current extent of usage in various clients.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.defang.io/blog/2025/02/18/model-context-protocol">Simplifying Deployment of AI Apps to the Cloud using Docker and Model Context Protocol | Defang</a>: mcp</li><li><a href="https://github.com/evalstate/mcp-py-repl">GitHub - evalstate/mcp-py-repl: A python repl for MCP</a>: A python repl for MCP. Contribute to evalstate/mcp-py-repl development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1341838736530604102)** (3 messages): 

> `MAX 25.1 Livestream, Community Meeting Talks, Modular Branded Merchandise` 


- **Join the MAX 25.1 Livestream!**: A livestream is scheduled for tomorrow to discuss everything about **MAX 25.1**. You can [join on LinkedIn](https://www.linkedin.com/events/introducingmax25-17297704283980902402/theater/) and submit questions through this [Google Form](https://forms.gle/NkjU6e3n15TRtiMA7).
   - *Feel free to share your questions; we’re eager to hear your thoughts.*
- **Opportunities to Present in Community Meeting**: Spots are open for talks during Monday's community meeting, inviting members to highlight their projects or focus areas. Interested participants are encouraged to reach out and express their desire to present.
   - *This is a great opportunity to showcase innovative work within the community.*
- **Modular's Stylish Patagonia Sweater**: A member praised the **Modular branded Patagonia sweater**, expressing strong enthusiasm for its design. It appears to be a hit among community members, showcasing their brand pride.
   - *It definitely has caught attention for its style and quality.*



**Link mentioned**: <a href="https://forms.gle/NkjU6e3n15TRtiMA7">Modular Community Q&amp;A</a>: no description found

  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1341907584952832103)** (33 messages🔥): 

> `Native Mojo Windows Support, Slab List Structure Discussion, Comparing Mojo and Python, AI Compute Performance, Low-Level Programming in Mojo` 


- **Native Mojo Windows Support is Uncertain**: Discussions indicated that there's likely no scheduled timeline for **native Mojo Windows support**, primarily due to the high costs associated with running AI clusters on **Windows**.
   - *nix OSes are favored for compute work, and many are utilizing cloud Linux platforms instead of Windows, making this not a short-term priority.*
- **Understanding Slab List Structures**: A member defined a **slab list** as an effective data structure that operates much like a `LinkedList[InlineArray[T, N]]`, focusing on simplicity and efficient memory management.
   - They highlighted that using this structure can yield **O(1)** performance for various operations, with faster iteration than traditional linked lists due to improved cache efficiency.
- **Mojo's Relationship with Python**: There was a consensus that **Mojo** can be seen as a language derived from Python but with performance closer to C/C++/Rust, aiming for future compatibility similar to that of C++ with C.
   - One member concluded that Mojo's advanced type system allows for a **Python-inspired** experience, suggesting it may appeal to existing **Nim** users.
- **AI Compute Performance Compared**: Members noted that once **AI compute** tasks are pushed to the GPU, performance differences become negligible, with Mojo potentially outperforming Python significantly for many CPU tasks.
   - Mojo's speed can even be better on ARM architectures than traditional **pure Python**, although using Windows through WSL is said to introduce some overhead.
- **Low-Level Programming Ease with Mojo**: A member expressed that working with low-level tasks in **Mojo** is easier compared to C/C++, indicating that Mojo's design facilitates hardware utilization effectively.
   - They suggested that Mojo doesn’t have to strictly adhere to Python's syntax for low-level coding, as a strong capability for running Python scripts suffices for many applications.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/nic">nic - Overview</a>: Chief Trolling Officer. nic has 58 repositories available. Follow their code on GitHub.</li><li><a href="https://github.com/nickziv/libslablist">GitHub - nickziv/libslablist: The slab list is a very memory-efficient logarithmic-time data structure.</a>: The slab list is a very memory-efficient logarithmic-time data structure. - nickziv/libslablist
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1341830882704625674)** (2 messages): 

> `LlamaCloud EU, LlamaParse upgrades` 


- **LlamaCloud EU launches for data compliance**: We announced early access to [LlamaCloud EU](https://t.co/HTt2pob88p), a new SaaS offering providing secure knowledge management with full data residency within EU jurisdiction.
   - This launch aims to remove significant barriers for European companies seeking compliant solutions, focusing on **security** and **data residency**.
- **LlamaParse evolves with new features**: [LlamaParse](https://t.co/ux3gsvDIeW) has introduced new parsing modes: Fast, Balanced, and Premium, to meet varying document parsing needs effectively.
   - These upgrades are designed to tackle existing **document parsing challenges**, allowing more versatility in handling different types of documents.


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1341864480254201868)** (21 messages🔥): 

> `Agent Workflows in the Loop, Handling Multiple Tool Calls, Redis Parallel Processing Best Practices, LlamaCloud System Outage, Blockchain Developments` 


- **Agent Workflow with Handoff Issues**: A developer is facing issues where the LLM returns *'I am handing off to AgentXYZ'* instead of executing tool calls, specifically in a multi-agent workflow scenario.
   - Another user questioned whether the **handoff rules should be included in the system message** to clarify this behavior.
- **Ensuring Parallel Processing in Redis**: A user is seeking strategies to effectively run **1000 parallel batches** persisting a summary index while avoiding race conditions in Redis.
   - They are storing review embeddings in a Redis namespace and are concerned about potential **key collisions and resource constraints**.
- **LlamaCloud Service Status Discussed**: Users reported on a potential issue with LlamaCloud services, although the status page indicated *all systems operational*.
   - Team members confirmed they are investigating the situation, and one user humorously suggested there's already enough *scamcoin* activity present.
- **Concerns Over Blockchain Projects**: Inquiries about the possibility of creating a coin on **Solana** revealed that any such claims are deemed **scams** by the community.
   - Discussion also unfolded regarding the broader implications of being involved with 'scamcoin' projects.
- **Challenges with Tool Calling in LLMs**: One user expressed frustration over LLM responses failing to execute desired actions, focusing on the need for tool calls instead of generic responses.
   - They noted concerns about breaking the existing handoff prompt, which appears to be designed for **versatile tool interactions**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://llamaindex.statuspage.io">LlamaIndex Status</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/agent/agent_workflow_basic/#human-in-the-loop">AgentWorkflow Basic Introduction - LlamaIndex</a>: no description found
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1341920240250126366)** (1 messages): 

> `Next phase of AI, Data operation trends` 


- **Exploring the next phase of AI and data operations**: A member shared a post discussing the *next phase of AI* and emerging **data operation trends** that could significantly impact the industry.
   - The article titled *The End of Big Dumb AI Data* can be accessed [here](https://open.substack.com/pub/procurefyi/p/the-end-of-big-dumb-ai-data?r=223ajc&utm_campaign=post&utm_medium=web&showWelcomeOnShare=false) for those interested in understanding these shifts.
- **Insights on AI Data Management**: The discussion highlights a potential shift from **traditional AI data management** to more efficient and adaptable methodologies.
   - The member emphasized that organizations need to rethink their data strategies to keep pace with the evolving technology landscape.


  

---


### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1341846122133721098)** (2 messages): 

> `Channel Creation Request, Color Change Announcement` 


- **Request for New Channel Creation**: A member requested the creation of a specific channel, indicating that they should be able to send screenshots there as well.
   - This request was made in a more casual and friendly tone, highlighted by a heart emoji. 
- **Member's Color Change Update**: Another member announced their color change, stating simply, *"now I am pink."*
   - This change likely contributes to the visual dynamics of the Discord community.


  

---


### **Cohere ▷ #[cmd-r-bot](https://discord.com/channels/954421988141711382/1168578374038470656/1341928753395138601)** (3 messages): 

> `Profit-sharing opportunities, Impact of a world without coffee` 


- **Seeking Partner for Profit-sharing**: A member is looking for someone aged **25-50** who is willing to share their identity and profits ranging from **$100-1500**.
   - This could indicate a potential business or investment opportunity based on shared interests.
- **Essay Request on Coffee's Absence**: A member requested an essay about the effects of a world without **coffee**, highlighting its cultural and economic significance.
   - This discussion suggests a curiosity about lifestyle changes in the hypothetical scenario where coffee is no longer available.


  

---


### **Cohere ▷ #[projects](https://discord.com/channels/954421988141711382/1218409701339828245/1341921219322318903)** (13 messages🔥): 

> `Identity Sharing in Collaboration, Concerns about Personal Information, Communication Clarity in Forums` 


- **Identity Sharing Proposal Sparks Debate**: A user proposed a collaboration opportunity involving identity sharing for profit ranging from **$100-1500**, highlighting an age range of **25-50**.
   - This led to concerns being raised about the implications of identity theft in such arrangements, with no **website** or relevant documentation provided.
- **Caution Around Personal Information**: A member reminded the group that not everyone is comfortable disclosing **personally identifiable information** in a public forum, emphasizing this channel's focus on **Cohere related projects**.
   - The reminder underscored the importance of respecting individual privacy while discussing potential collaborations.
- **Call for Clarity in Communication**: Concerns were raised about the ambiguity in written communication, with advice given to use **clearer writing** to prevent misunderstandings.
   - Members emphasized the importance of improving communication to foster **positive collaboration** within the group.
- **Skepticism About Project Details**: A user expressed skepticism about the initial proposal due to the lack of information, citing the absence of a **website**, documentation, or a clear project description.
   - This skepticism highlights a demand for transparency when discussing new collaborative opportunities.


  

---


### **AI21 Labs (Jamba) ▷ #[general-chat](https://discord.com/channels/874538902696914944/874538902696914947/1341832890568282164)** (16 messages🔥): 

> `Jamba API usage, PHP integration with Jamba, Response formatting issues, Removing special characters, Using AJAX for API calls` 


- **Getting Started with Jamba API**: A user sought assistance with using the **Jamba API** and shared code for making API calls, particularly noting difficulties with syntax.
   - Another member provided a detailed outline of using the API, including necessary parameters and headers.
- **Understanding API Responses**: There was a discussion regarding the output from the API, particularly that it includes **escape characters** which can complicate processing.
   - Members confirmed that response formatting may differ depending on the **language** used, emphasizing the need for additional handling.
- **PHP Specifics for Jamba API Integration**: A user mentioned working with **Symfony and PHP** and expressed the need to convert API responses into a usable format.
   - Advice was given to seek help from other members regarding special character handling in PHP outputs.
- **Using AJAX for Improved API Output**: One user suggested utilizing **AJAX** to enhance the API response handling but noted that results are still inconsistent.
   - There was confirmation that output in the **Jamba chat window** is formatted differently, which may influence how results appear.
- **Collaboration for PHP Challenges**: Members noted that assistance might be available from other users familiar with PHP, particularly in handling outputs effectively.
   - One member reached out directly to another for potential guidance on the subject.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.ai21.com/reference">Jamba 1.5</a>: Jamba-1.5 instruction following chat models</li><li><a href="https://docs.ai21.com/reference/jamba-15-api-ref">Jamba 1.5</a>: Jamba-1.5 instruction following chat models
</li>
</ul>

</div>
  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1341888954336808980)** (6 messages): 

> `Model Performance on Different Hardware, Int8 Quantization Issues, Testing Speed in Torch vs Tinygrad, Optimizations with BEAM, New PyTorch Channel` 


- **GeForce 850M vs RTX 4070 Performance**: Testing revealed that an **old GeForce 850M** performs at **3 tok/s** after **8 seconds** on Brave, while the **RTX 4070** achieves **12 tok/s** in **1.9 seconds** on Windows 11 Chrome.
   - However, it was noted that overall model usability remains limited due to various **computational costs and numerical stiffness**.
- **Challenges with Int8 Quantization**: It was pointed out that the **Int8 quantization** approach may need improvement since the model occasionally goes 'off rails' after several hundred tokens when using **Int8Linear**.
   - *Direct messages or GitHub* discussions were suggested for more focused conversations about **tinychat** developments.
- **Speed Test Results Show Mixed Performance**: Recent speed tests indicated that **torch** outperformed **tinygrad** on **2048x2048** tensors, with **0.22 ms** for torch compared to **0.42 ms** for tinygrad.
   - However, on **4096x4096**, the performance was closer, with tinygrad only being **1.08x slower** than torch as they continue to investigate performance discrepancies.
- **Optimizing Performance with BEAM**: Further insights suggested that increasing **BEAM** values might mitigate performance issues, with tests showing **0.21 ms** for **2048x2048** tensors with **BEAM=10** in torch.
   - Performance remained relatively consistent across tensor sizes, highlighting the potential for optimization with higher **BEAM** settings.
- **George Hotz Announces New PyTorch Channel**: A new channel was created for discussions related to **PyTorch**, indicating community engagement.
   - This addition is expected to facilitate more specialized discussions as user contributions grow.


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1342172399084503213)** (4 messages): 

> `Operations in tinygrad, Documentation for BLOCK operations, Codebase search strategies` 


- **Inquiry About BLOCK Operations**: A member requested documentation regarding the `BLOCK`, `BLOCKSTART`, `BLOCKFORK`, and `BLOCKEND` operations to understand what they represent and store.
   - The question highlights a need for clearer documentation or guidelines within the tinygrad project.
- **GitHub Resource Shared**: In response to the inquiry, a member linked to the [GitHub repository](https://github.com/tinygrad/tinygrad/blob/master/tinygrad%2Fcodegen%2Flinearize.py) containing `linearize.py`, which is likely relevant to the BLOCK operations.
   - This resource could serve as a starting point for understanding the implementation and usage of these operations.
- **Codebase Search for Documentation**: A member suggested that a useful first step for finding information is to search the entire codebase for related references.
   - This approach emphasizes the importance of leveraging available resources for self-directed learning within the tinygrad framework.



**Link mentioned**: <a href="https://github.com/tinygrad/tinygrad/blob/master/tinygrad%2Fcodegen%2Flinearize.py">tinygrad/tinygrad/codegen/linearize.py at master · tinygrad/tinygrad</a>: You like pytorch? You like micrograd? You love tinygrad! ❤️  - tinygrad/tinygrad

  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1341828771438923868)** (10 messages🔥): 

> `System Message Terminology, Model Instructions, Image Pasting Capability, Nomic Implementation Questions` 


- **Confusion Over System Message Terminology**: A member clarified that the term 'system message' is now used in the UI, indicating a shift in naming conventions.
   - Another participant affirmed that old habits can be difficult to change when navigating these systems.
- **Using Instructions in System Message**: It's mentioned that plain English instructions can be used in the 'system message', and most models will respect these commands.
   - Some members expressed skepticism about the ease of this process, questioning if using Jinja or JSON code is more effective.
- **Image Handling Limitations in GPT4All**: One member queried about the ability to paste images directly into the text bar like in other AI platforms.
   - It was clarified that **GPT4All** cannot handle images, and external software is recommended for such tasks.
- **Discussion on Nomic and NOIMC v2 Release**: A member expressed confusion over the implementation of **NOIMC v2**, questioning why it appears to be incorrectly implemented.
   - Another member humorously sought confirmation about being on **Nomic**, showcasing their frustration.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-lecture-discussion](https://discord.com/channels/1280234300012494859/1282734248112947210/1341943677773418506)** (4 messages): 

> `2024 LLM Agents Course, Quiz Archive Access, DSPy Interest, Lecture Availability` 


- **Consider Watching 2024 LLM Agents Course**: A member suggested that while it's not necessary to audit the **Fall 2024 Course**, it could be beneficial for deeper understanding, especially for those interested in **DSPy**, absent from this semester’s syllabus.
   - They provided a [YouTube playlist](https://youtube.com/playlist?list=PLS01nW3RtgopsNLeM936V4TNSsvvVglLc&feature=shared) of lectures from that course.
- **Disappearance of Videos and Quizzes**: A member expressed confusion over the **videos and quizzes** disappearing from the current syllabus, hindering their ability to catch up.
   - In response, another member linked to a **quizzes archive** for the Fall 2024 course available [here](https://docs.google.com/document/d/1pYvOxt2UWwc3z4QlW2Di5LQT-FJPWZ419mxJT7pFPsU/edit?usp=sharing).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://youtube.com/playlist?list=PLS01nW3RtgopsNLeM936V4TNSsvvVglLc&feature=shared,">LLM Agents MOOC</a>: Large Language Model Agents MOOC F24</li><li><a href="https://docs.google.com/document/d/1pYvOxt2UWwc3z4QlW2Di5LQT-FJPWZ419mxJT7pFPsU/edit?usp=sharing)">Quizzes Archive - LLM Agents MOOC</a>: NOTE: The correct answers are in the black boxes (black text on black background). Highlight the box with your cursor to reveal the correct answer (or copy the text into a new browser if it’s hard to ...
</li>
</ul>

</div>
  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-readings-discussion](https://discord.com/channels/1280234300012494859/1282735578886181036/1341942110936498176)** (3 messages): 

> `Quiz Access, MOOC Resources` 


- **Help on Accessing Quizzes**: A member inquired about obtaining **quiz 1 and 2** for the weekend due to starting late in the course.
   - Another member responded, mentioning that the quizzes can be found on the MOOC’s [page](https://llmagents-learning.org/sp25) or the [announcement page](https://llmagents-learning.org/f24).
- **MOOC Course Completion Notice**: It was noted that the course has now completed, but video lectures remain accessible in the syllabus.
   - *All certificates have been released,* and students were encouraged to sign up for the [Spring 2025 iteration](https://llmagents-learning.org/sp25).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://llmagents-learning.org/sp25">Advanced Large Language Model Agents MOOC</a>: MOOC, Spring 2025</li><li><a href="https://llmagents-learning.org/f24">Large Language Model Agents MOOC</a>: MOOC, Fall 2024
</li>
</ul>

</div>
  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1342013410719567882)** (1 messages): 

> `HaizeLabs Judge Compute, Qwen/Qwen2.5-VL-7B-Instruct, LLM-AggreFact scores` 


- **Inspiration from HaizeLabs Judge Compute**: A member ran the same dataset as HaizeLabs Judge Compute and achieved varying scores with the model **Qwen/Qwen2.5-VL-7B-Instruct**.
   - The scores ranged from **60%-70%** for 2-stage optimized to **88.50%** for mipro2, showcasing impressive performance metrics.
- **LLM-AggreFact Scores Detailed**: The scores for **LLM-AggreFact** with various methods were reported as follows: **labeled fewshots 81.25%**, **bootstrap random 84.50%**, **copro 84%**.
   - This indicates a competitive performance across different evaluation methods, suggesting robustness in the model's scoring capabilities.
- **Source Code Shared on GitHub**: All the source code related to the evaluation was shared in a [GitHub Gist](https://gist.github.com/fullstackwebdev/fa4934fb4669cfc3e8c6ced950ea7a22).
   - The project titled **LLM-AggreFact_DSPy** can be accessed for further insights into the methodologies used in the evaluations.



**Link mentioned**: <a href="https://gist.github.com/fullstackwebdev/fa4934fb4669cfc3e8c6ced950ea7a22">LLM-AggreFact_DSPy</a>: GitHub Gist: instantly share code, notes, and snippets.

  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1341817003324739586)** (5 messages): 

> `Judge-Time Scaling, Personal Voice Identity Manager, DSPy Conversation History, Message Template Exporting` 


- **Judge-Time Scaling Pioneered with Verdict**: Leonard Tang announced the release of [Verdict](https://x.com/leonardtang_/status/1892243653071908949), a library aimed at scaling judge-time compute, emphasizing that AI's current reliability issues stem from evaluation rather than generation.
   - Tang noted that the recent innovations in AI have focused on **pre-training** and **inference-time scaling**, positioning improved evaluation as the next major advancement for the field.
- **Personal Voice Identity Manager Potential**: A member expressed enthusiasm about the **Verdict** library, suggesting it aligns perfectly with their concept for a **Personal Voice Identity Manager**.
   - This indicates an interest in exploring how enhanced evaluation techniques can benefit user identity management in AI applications.
- **Clarification on DSPy Conversation History**: A member sought to verify whether DSPy automatically injects conversation history into calls, indicating a precaution before diving deeper into the implementation.
   - This highlights concerns about potential complexities in managing AI interactions without overwriting previous context.
- **Exporting Prompts to Message Templates**: An FAQ was shared detailing how to freeze and export prompts in a program into message templates using a Python snippet with `dspy.ChatAdapter()`.
   - It was mentioned that while this method is useful, it results in a loss of control flow logic, suggesting alternatives like `program.save()` or `program.dump_state()` for complete exports.



**Link mentioned**: <a href="https://x.com/leonardtang_/status/1892243653071908949">Tweet from Leonard Tang (@leonardtang_)</a>: First came pre-training scaling; then came inference-time scaling.Now comes judge-time scaling.Despite progress in AI through scaled inference-time compute, AI remains unreliable in open-ended, non-ve...

  

---


---


{% else %}


> The full channel by channel breakdowns have been truncated for email. 
> 
> If you want the full breakdown, please visit the web version of this email: [{{ email.subject }}]({{ email_url }})!
>
> If you enjoyed AInews, please [share with a friend](https://buttondown.email/ainews)! Thanks in advance!

{% endif %}