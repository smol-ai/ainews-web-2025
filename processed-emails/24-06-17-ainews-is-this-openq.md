---
id: 673ee033-f1bd-46e2-a99c-140ed5ab9682
title: Is this... OpenQ*?
date: '2024-06-18T00:38:33.191318Z'
original_slug: ainews-is-this-openq
description: >-
  AI news for 6/14/2024-6/17/2024 covers new model releases, research on
  test-time search, Apple AI developments, open source models matching GPT-4,
  and new video generation models.
companies:
  - apple
  - openai
  - nvidia
  - stability-ai
  - luma-labs
  - runway
  - apparate-labs
models:
  - gpt-4
  - llama-3-8b
  - nemotron-4-340b
  - deepseek-coder-v2
  - stable-diffusion-3-medium
  - dream-machine
  - gen-3-alpha
  - proteus
topics:
  - test-time-search
  - model-performance
  - on-device-ai
  - llm-research
  - multimodal
  - text-to-image
  - video-generation
  - ai-partnerships
---


<!-- buttondown-editor-mode: plaintext -->**MCTS is all you need.**

> AI News for 6/14/2024-6/17/2024.
We checked 7 subreddits, [**384** Twitters](https://twitter.com/i/lists/1585430245762441216) and **30** Discords (**414** channels, and **5506** messages) for you. 
Estimated reading time saved (at 200wpm): **669 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

**A bunch of incremental releases over this weekend**; [DeepSeekCoder V2](https://x.com/deepseek_ai/status/1802680388256768145) promises GPT4T-beating performance (validated by [aider](https://x.com/paulgauthier/status/1802774069185753298)) at $0.14/$0.28 per million tokens (vs GPT4T's $10/$30), Anthropic dropped some [Reward Tampering research](https://x.com/anthropicai/status/1802743256461046007?s=46&t=90xQ8sGy63D2OtiaoGJuww), and [Runway finally dropped their Sora response](https://x.com/runwayml/status/1802691475391566108?s=46&t=90xQ8sGy63D2OtiaoGJuww).

However probably the longer lasting, meatier thing to dive into is the discussion around "test-time" search:


 ![image.png](https://assets.buttondown.email/images/7f958e3b-c3f9-452d-9499-2e97af8c0c7e.png?w=960&fit=max) 

spawning a list of [related papers](https://x.com/teortaxestex/status/1802128370861232374?s=46&t=90xQ8sGy63D2OtiaoGJuww):

- [Accessing GPT-4 level Mathematical Olympiad Solutions via Monte Carlo Tree Self-refine with LLaMa-3 8B: A Technical Report](https://arxiv.org/abs/2406.07394)
- [Improve Mathematical Reasoning in Language Models by Automated Process Supervision](https://arxiv.org/abs/2406.06592)
- [AlphaMath Almost Zero: Process Supervision Without Process](https://arxiv.org/abs/2405.03553)
- [ReST-MCTS*: LLM Self-Training via Process Reward Guided Tree Search](https://arxiv.org/abs/2406.03816)

We'll be honest that we haven't read any of these papers yet, but we did cover [OpenAI's thoughts on verifier-generator process supervision on the ICLR podcast](https://www.latent.space/p/iclr-2024-benchmarks-agents), and have lined the remaining papers up for the Latent Space Discord Paper Club.


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

**Apple's AI Developments and Partnerships**

- **Apple Intelligence announced**: [@adcock_brett](https://twitter.com/adcock_brett/status/1802371344539025882) noted Apple revealed Apple Intelligence at WWDC, their first AI system coming to iPhone, iPad, and Mac, with features like a smarter Siri and image/document understanding.
- **OpenAI partnership**: Apple and OpenAI announced a partnership to directly integrate ChatGPT into iOS 18, iPadOS 18, and macOS, as mentioned by [@adcock_brett](https://twitter.com/adcock_brett/status/1802371424268460037).
- **On-device AI models**: [@ClementDelangue](https://twitter.com/ClementDelangue/status/1802742076544594254) highlighted that Apple released 20 new CoreML models for on-device AI and 4 new datasets on Hugging Face.
- **Optimized training**: Apple offered a peek into its new models' performance and how they were trained and optimized, as reported by [@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1802717835182968928).
- **LoRA adapters for specialization**: [@svpino](https://twitter.com/svpino/status/1802677551355048292) explained how Apple uses LoRA fine-tuning to generate specialized "adapters" for different tasks, swapping them on the fly.

**Open Source LLMs Matching GPT-4 Performance**

- **Nemotron-4 340B from NVIDIA**: NVIDIA released Nemotron-4 340B, an open model matching GPT-4 (0314) performance, according to [@adcock_brett](https://twitter.com/adcock_brett/status/1802371484972720396).
- **DeepSeek-Coder-V2**: [@deepseek_ai](https://twitter.com/deepseek_ai/status/1802680388256768145) introduced DeepSeek-Coder-V2, a 230B model excelling in coding and math, beating several other models. It supports 338 programming languages and 128K context length.
- **Stable Diffusion 3 Medium**: Stability AI released open model weights for its text-to-image model, Stable Diffusion 3 Medium, offering advanced capabilities, as noted by [@adcock_brett](https://twitter.com/adcock_brett/status/1802371687129686390).

**New Video Generation Models**

- **Dream Machine from Luma Labs**: Luma Labs launched Dream Machine, a new AI model generating 5-second video clips from text and image prompts, as reported by [@adcock_brett](https://twitter.com/adcock_brett/status/1802371446678704332).
- **Gen-3 Alpha from Runway**: [@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1802706846597398749) showcased Runway's new Gen-3 Alpha model, generating detailed videos with complex scenes and customization options.
- **PROTEUS from Apparate Labs**: Apparate Labs launched PROTEUS, a real-time AI video generation model creating realistic avatars and lip-syncs from a single reference image, as mentioned by [@adcock_brett](https://twitter.com/adcock_brett/status/1802371709518885038).
- **Video-to-Audio from Google DeepMind**: [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1802733643992850760) shared progress on their video-to-audio generative technology, adding sound to silent clips matching scene acoustics and on-screen action.

**Robotics and Embodied AI Developments**

- **OpenVLA for robotics**: OpenVLA, a new open-source 7B-param robotic foundation model outperforming a larger closed-source model, was reported by [@adcock_brett](https://twitter.com/adcock_brett/status/1802371664795009361).
- **Virtual rodent from DeepMind and Harvard**: DeepMind and Harvard created a 'virtual rodent' powered by an AI neural network, mimicking agile movements and neural activity of real-life rats, as noted by [@adcock_brett](https://twitter.com/adcock_brett/status/1802371597669388586).
- **Manta Ray drone from Northrop Grumman**: [@adcock_brett](https://twitter.com/adcock_brett/status/1802371575347331399) mentioned Northrop Grumman released videos of the 'Manta Ray', their new uncrewed underwater vehicle drone prototype.
- **Autonomous driving with humanoids**: A new approach to autonomous driving leveraging humanoids to operate vehicle controls based on sensor feedback was reported by [@adcock_brett](https://twitter.com/adcock_brett/status/1802371731899740536).

**Miscellaneous AI Research and Applications**

- **Anthropic's reward tampering research**: [@AnthropicAI](https://twitter.com/AnthropicAI/status/1802743256461046007) published a new paper investigating reward tampering, showing AI models can learn to hack their own reward system.
- **Meta's CRAG benchmark**: Meta's article discussing the Corrective Retrieval-Augmented Generation (CRAG) benchmark was highlighted by [@dair_ai](https://twitter.com/dair_ai/status/1802695213086863748).
- **DenseAV for learning language from videos**: An AI algorithm called 'DenseAV' that can learn language meaning and sound locations from unlabeled videos was mentioned by [@adcock_brett](https://twitter.com/adcock_brett/status/1802371642460426368).
- **Goldfish loss for training LLMs**: [@tomgoldsteincs](https://twitter.com/tomgoldsteincs/status/1802726878924464273) introduced the goldfish loss, a technique for training LLMs without memorizing training data.
- **Creativity reduction in aligned LLMs**: [@hardmaru](https://twitter.com/hardmaru/status/1802578579605393892) shared a paper exploring the unintended consequences of aligning LLMs with RLHF, which reduces their creativity and output diversity.

---

# AI Reddit Recap

> Across r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity. Comment crawling works now but has lots to improve!

**AI Models and Techniques**

- **Improved CLIP ViT-L/14 for Stable Diffusion**: In /r/StableDiffusion, an [**improved CLIP ViT-L/14 model is available for download**](https://www.reddit.com/r/StableDiffusion/comments/1dhaz43/greatly_improved_clip_vitl14_for_download_a/), along with a Long-CLIP version, which can be used with any Stable Diffusion model.
- **Mixed Precision Training from Scratch**: In /r/MachineLearning, a [reimplementation of the original mixed precision training paper from Nvidia](https://www.reddit.com/r/MachineLearning/comments/1dhlh0z/p_mixed_precision_training_from_scratch/) on a 2-layer MLP is presented, diving into CUDA land to showcase TensorCore activations.
- **Understanding LoRA**: Also in /r/MachineLearning, a [**visual guide to understanding Low-Rank Approximation (LoRA)**](https://www.reddit.com/r/MachineLearning/comments/1dh4s3x/r_understanding_lora_a_visual_guide_to_lowrank/) for efficient fine-tuning of large language models is shared. LoRA reduces the number of parameters involved in fine-tuning by 10,000x while still converging to the performance of a fully fine-tuned model.
- **GPT-4 level Math Solutions with LLaMa-3 8B**: A [research paper explores accessing GPT-4 level Mathematical Olympiad solutions](https://arxiv.org/abs/2406.07394) using Monte Carlo Tree Self-refine with the LLaMa-3 8B model.
- **Instruction Finetuning From Scratch**: An [implementation of instruction finetuning from scratch](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/01_main-chapter-code/ch07.ipynb) is provided.
- **AlphaMath Almost Zero**: [Research on AlphaMath Almost Zero](https://arxiv.org/abs/2405.03553) introduces process supervision without process.

**Stable Diffusion Models and Techniques**

- **Model Comparisons**: In /r/StableDiffusion, a [comparison of PixArt Sigma, Hunyuan DiT, and SD3 Medium models](https://www.reddit.com/r/StableDiffusion/comments/1dh73j6/model_comparison_pixart_sigma_vs_hunyuan_dit_vs/) for image generation is presented, with PixArt Sigma and SDXL refinement showing promise. 
- **ControlNet for SD3**: [ControlNet Canny and Pose models have been released for SD3](https://www.reddit.com/r/StableDiffusion/comments/1dh257c/controlnet_canny_and_pose_are_released_for_sd3/), with Tile and Inpainting models coming soon.
- **Sampler and Scheduler Permutations**: An [overview of all working sampler and scheduler combinations for Stable Diffusion 3](https://www.reddit.com/r/StableDiffusion/comments/1dh5n7k/all_sampler_and_scheduler_permutations_for_stable/) is provided.
- **CFG Values in SD3**: A [comparison of different CFG values in Stable Diffusion 3](https://www.reddit.com/r/StableDiffusion/comments/1dhdyt7/comparison_of_cfg_values_in_stable_diffusion_3/) shows a narrower usable range compared to SD1.
- **Playground 2.5 Similar to Midjourney**: The [Playground 2.5 model is identified as the most similar to Midjourney](https://www.reddit.com/r/StableDiffusion/comments/1dhflr4/the_most_similar_model_to_midjourney_is/) in terms of output quality and style.
- **Layer Perturbation Analysis in SD3**: An [analysis of how adding random noise to different layers in SD3](https://www.reddit.com/r/StableDiffusion/comments/1dh3ml4/manual_layer_pertrubation_for_sd3_my_findings_on/) affects the final output is conducted, potentially providing insights into how and where SD3 was altered.

**Llama and Local LLM Models**

- **Llama 3 Spellbound**: In /r/LocalLLaMA, [Llama 3 7B finetune is trained without instruction examples](https://www.reddit.com/r/LocalLLaMA/comments/1dh5ee0/llama_3_spellbound_pretraining_over_sft_for/), aiming to preserve world understanding and creativity while reducing positivity bias in writing.
- **Models for NSFW Roleplay**: A [request for models that can run on a 3060 12GB GPU](https://www.reddit.com/r/LocalLLaMA/comments/1dh5fsq/are_there_any_models_that_i_can_run_on_my_3060/) and produce NSFW roleplay similar to a provided example is made.
- **Model Similar to Command-R**: Someone is [seeking a model with similar quality to Command-R](https://www.reddit.com/r/LocalLLaMA/comments/1dhllgv/any_model_with_similar_quality_to_commandr_but/) but requiring less memory for 64k context size on a Mac with M3 Max 64GB.
- **System Prompt for RP/Chatting/Storytelling**: A [detailed system prompt for controlling models in roleplay, chatting, and storytelling scenarios](https://www.reddit.com/r/LocalLLaMA/comments/1dhluqd/a_detailed_system_prompt_for_reining_models_in/) is shared, focusing on thorough, direct, and symbolic instructions.
- **Running High Models on 24GB VRAM**: Guidance is sought on [running high models/context on 24GB of VRAM](https://www.reddit.com/r/LocalLLaMA/comments/1dhj5e7/im_dumb_how_are_you_guys_running_high/), possibly using quantization or 4/8 bits.
- **Underlying Model Importance with RAG**: A discussion on [whether the underlying model matters when using Retrieval-Augmented Generation (RAG)](https://www.reddit.com/r/LocalLLaMA/comments/1dh6knt/does_the_underlying_model_matter_if_using_rag/) with a solid corpus takes place.

**AI Ethics and Regulation**

- **OpenAI Board Appointment Criticism**: [Edward Snowden criticizes OpenAI's decision to appoint a former NSA director](https://fortune.com/2024/06/14/edward-snowden-eviscerates-openai-paul-nakasone-board-directors-decision/) to its board, calling it a "willful, calculated betrayal of the rights of every person on earth."
- **Stability AI's Closed-Source Approach**: In /r/StableDiffusion, there is a [discussion on Stability AI's decision to go the closed-source API selling route](https://www.reddit.com/r/StableDiffusion/comments/1dhh6z6/why_cant_sai_understand_their_advantage_is/), questioning their ability to compete without leveraging community fine-tunes.
- **Clarification on Stable Diffusion TOS**: A [clarification on the terms of service (TOS) for Stable Diffusion models](https://www.reddit.com/r/StableDiffusion/comments/1dh9buc/to_all_the_people_misunderstanding_the_tos/) is provided, addressing misunderstandings caused by a clickbait YouTuber.
- **Crowdfunded Open-Source Alternative to SD3**: A [suggestion to start a crowdfunded open-source alternative to SD3](https://www.reddit.com/r/StableDiffusion/comments/1dhjal0/would_be_great_if_we_can_start_a_crowdfunded/) is made, potentially led by a former Stability AI employee who helped train SD3 but recently resigned.
- **Malicious Stable Diffusion Tool on GitHub**: A [news article reports on hackers targeting AI users with a malicious Stable Diffusion tool](https://www.404media.co/hackers-target-ai-users-with-malicious-stable-diffusion-tool-on-github/) on GitHub, claiming to protest "art theft" but actually seeking financial gain through ransomware.
- **Impact of Debiasing on Creativity**: A [research paper discusses the impact of debiasing language models on their creativity](https://arxiv.org/abs/2406.05587), suggesting that censoring models makes them less creative.

**AI and the Future**

- **Feeling Lost Amidst AI Advancements**: In /r/singularity, a [personal reflection on feeling lost and uncertain about the future](https://www.reddit.com/r/singularity/comments/1dh5aqd/feeling_lost/) in the face of rapid AI advancements is shared.
- **Concerns About AI's Impact on Career**: Also in /r/singularity, someone expresses [feeling lost about the future of AI and their career](https://www.reddit.com/r/singularity/comments/1dh5aq.) in light of recent developments.

---

# AI Discord Recap

> A summary of Summaries of Summaries

**1. AI Model Performance and Scaling**

- **Scaling Up with New AI Models**: DeepSeek's **[Coder V2](https://vxtwitter.com/teortaxesTex/status/1802681431992213767)** reportedly beats GPT-4 on benchmarks and Google DeepMind reveals new video-to-audio tech creating tracks for any video, gaining traction on [Rowan Cheung's X profile](https://x.com/rowancheung/status/1802734770117333257).
- **Expanding AI Capabilities Across Platforms**: Runway introduces Gen-3 Alpha for video generation, enhancing cinematic styles and scene transitions. AP details shared on [Twitter](https://x.com/runwayml/status/1802691475391566108?s=46&t=90xQ8sGy63D2OtiaoGJuww).

**2. Integration and Implementation Across Platforms**

- **Hybrid Notes App Unveils LLM integration**: OpenRouter unveils a notes app integrating LLMs for dynamic content interaction, though lacking mobile support as specified on their [full-screen app](https://000700836.deployed.codepen.website/).
- **Challenges with Implementation on Various Platforms**: Users face issues like CORS errors on OpenRouter and integration challenges on LangChain, reflecting the need for better implementation guides or platform-specific APIs.

**3. Ethical AI and Governance**

- **OpenAI Shifts Towards Profit-Driven Model**: Speculations and confirmations stir about OpenAI's move towards becoming a profit entity, potentially impacting governance and ethical considerations. More on this from [The Information](https://www.theinformation.com/articles/openai-ceo-says-company-could-become-benefit-corporation-akin-to-rivals-anthropic-xai?utm_campaign=Editorial&utm_content=Article&utm_medium=organic_social&utm_source=twitter).
- **Discussions on AI Ethics Heat Up**: Debates continue about data privacy, model biases, and corporate governance in AI, as Edward Snowden criticizes OpenAI's new board appointments on [Edward Snowden's X profile](https://x.com/Snowden/status/1801610725229498403).

**4. New AI Developments and Benchmarking**

- **AI Innovations and Improvements Announced**: Anthropic publishes insights into AI's ability to tamper with reward systems in their new [research article](https://anthropic.com/research/reward-tampering).
- **Benchmarking New Models**: Stability AI releases SD3 models discussing new techniques for loss stabilization and artifacts management across forums, including a spotlight on [Reddit](https://www.reddit.com/r/StableDiffusion/comments/1dhd7vz/the_developer_of_comfy_who_also_helped_train_some/#lightbox).

**5. Collaborative AI Projects and User Engagement**

- **Community Projects Highlight AI Integration**: From a notes app merging notes and API keys management on OpenRouter to innovative AI-driven video generation tools like Dream Machine, community-built tools are pushing the boundaries of creativity and practical AI application, visible on platforms like [Lumalabs](https://lumalabs.ai/dream-machine).
- **Interactive AI Discussions and Collabs Flourish**: Webinars and collaborative events like the upcoming Mojo Community Meeting encourage deep dives into AI advancements, with detailed discussions and participation boasts from across the global user base as shared on [blog](https://www.modular.com/blog/whats-new-in-mojo-24-4-improved-collections-new-traits-os-module-features-and-core-language-enhancements).

---

# PART 1: High level Discord summaries




## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **SD3 License Troubles**: The new license for **Stable Diffusion 3 (SD3)** has led to its ban on Civitai due to legal ambiguities, with a review by Civitai's legal team announced in their [temporary ban statement](https://civitai.com/articles/5732).
- **Community Rift Over SD3**: Users expressed frustration with Stability AI's licensing of SD3, highlighting both confusion and discontent, while some criticized youtuber Olivio Sarikas for allegedly misrepresenting the SD3 license for views, referencing his [video](https://www.youtube.com/watch?v=HmxbsQDRTSs).
- **Guidance for ComfyUI**: Issues around **ComfyUI** setup sparked technical discussion, with suggested fixes for custom node installations including dependencies like cv2; a user-contributed [ComfyUI tutorial](https://www.youtube.com/watch?v=Di1KqPXxx2Y&t=30s) was shared to assist.
- **Seeking SD3 Alternatives**: The dialogue points to a shift towards seeking alternative models and artistic tools, such as video generation with animatediff, possibly due to the ongoing SD3 controversy.
- **Misinformation Allegations in the AI Community**: Accusations fly regarding youtuber Olivio Sarikas spreading misinformation about SD3's license, with community members challenging the veracity of his content found in his [contentious video](https://www.youtube.com/watch?v=HmxbsQDRTSs&t=2s&ab_channel=OlivioSarikas).



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Ollama Integration Nears Completion**: The Ollama support development has reached 80% completion, with the *Unsloth AI team* and Ollama collaboratively pushing through delays. Issues with template fine-tuning validation and learning rates concerning Ollama were discussed, along with an issue where running `model.push_to_hub_merged` does not save the full merged model, prompting a manual workaround.
  
- **Unsloth Speeds Ahead**: Unsloth's training process is touted to be 24% faster than torch.compile() torchtune for the NVIDIA GeForce RTX 4090, as benchmarks show its impressive training speed. Additionally, upcoming multi-GPU support for up to 8 GPUs is being tested with a select group of users getting early access for initial evaluations.

- **Training Troubles and Tricks**: Members encountered challenges like crashes during saving steps while training the Yi model, possible mismanagement of `quantization_method` during saving, and confusion around batch sizes and gradient accumulation in VRAM usage. Solutions and workarounds included verifying memory/disk resources and a [submitted pull request](https://github.com/unslothai/unsloth/pull/651) addressing the quantization error.

- **Lively Discussion on Nostalgia and Novelty in Music**: Members shared music ranging from a nostalgic 1962 song to iconic tracks by Daft Punk and Darude, showing a light-hearted side to the community. In contrast, concerns were raised over Gemma 2's output on AI Studio, with mixed reactions varying from disappointment to intrigue and anticipation for Gemini 2.0.

- **CryptGPT Secures LLMs with an Encryption Twist**: CryptGPT was introduced as a concept using the Vigenere cipher to pretrain GPT-2 models on encrypted datasets, ensuring privacy and requiring an encryption key to generate output, as detailed in a shared [blog post](https://x.com/diwanksingh/status/1802118343446724655). 

- **Singular Message of Curiosity**: The community-collaboration channel featured a single message expressing interest, but without further context or detail, its relevance to broader discussion topics remains unclear.



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **NVIDIA's Next Big Thing Speculated and PyCUDA SM Query Clarified**: Engineers speculated about the potential specs of the upcoming NVIDIA 5090 GPU, with rumors of up to 64 GB of VRAM circulating yet met with skepticism. Additionally, a discrepancy in GPU SM count for an A10G card reported by techpowerup was cleared up, with independent sources such as Amazon Web Services confirming the correct count as 80, not the 72 originally stated.

- **Triton and Torch Users Navigate Glitches and Limits**: Triton users encountered an `AttributeError` in Colab and debated the feasibility of nested reductions for handling quadrants. Meanwhile, PyTorch users adjusted the SM threshold in `torch.compile(mode="max-autotune")` to accommodate GPUs with less than 68 SMs and explored enabling coordinate descent tuning for better performance.

- **Software and Algorithms Push the AI Envelope**: A member lauded the matching of GPT-4 with LLaMA 3 8B, while Akim will attend the AI_dev conference and is open to networking. Elsewhere, Vayuda’s search algorithm paper spurred interest among enthusiasts, discussed across multiple channels. Discussions around AI training, evident in Meta's described challenges in LLM training, underscore the importance of infrastructure adaptability.

- **CUDA Development Optics**: News from CUDA-focused development revealed: Permuted DataLoader integration did not significantly affect performance; a unique seed strategy was developed for stochastic rounding; challenges surfaced regarding ZeRO-2's memory overhead; and new LayerNorm kernels provided much-needed speedups under certain configurations.

- **Beyond CUDA: Dynamic Batching, Quantization, and Bit Packing**: In the domain of parallel computing, engineers struggled with dynamic batching for Gaudi architecture and discussed the complexity of quantization and bit-packing techniques. They stressed the VRAM limitations constraining local deployment of large models and shared diverse resources, including links to Python development environments and documentation on novel machine learning libraries.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio equips engineers with CLI tools**: The latest [LM Studio 0.2.22](https://lmstudio.ai) release introduced 'lms', a CLI management tool for models and debugging prompts, which is detailed in its [GitHub repository](https://github.com/lmstudio-ai/lms). The update streamlines the workflow for AI deployments, especially with model loading/unloading and input inspection.

- **Performance tweaks and troubleshooting**: Engineers discussed optimal settings for AI model performance, including troubleshooting GPU support for Intel ARC A7700, configuration adjustments for GPU layers, and adjusting Flash Attention settings. There was a recommendation to check [Open Interpreter's documentation](https://docs.openinterpreter.com/language-models/local-models/lm-studio) for issues hosting local models and a call for better handling of font sizes in LM Studio interfaces for usability.

- **Diverse model engagement**: Members recommended [Fimbulvetr-11B](https://huggingface.co/DavidAU/Fimbulvetr-11B-Ultra-Quality-plus-imatrix-GGUF) for roleplaying use-cases, while highlighting the fast-paced changes in coding models like **DeepSeek-Coder-V2**, advising peers to stay updated with current models for specific tasks like coding, which can be reviewed on sites like [Large and Small Language Models list](https.wikipedia://llm.extractum.io/list).

- **Hardware optimization and issues**: A link to archived LM Studio 0.2.23 was shared for those facing installation issues—a [MirrorCreator link](https://mir.cr/E1WVBIOO). Hardware discussions also included the compatibility of mixed RAM sticks, setting CPU cores for server mode, and troubleshooting GPU detection on various systems.

- **Development insights and API interactions**: Developers shared their aspirations for integrating various coding models like `llama3` and `deepseek-coder` into their VSCode workflow and sought assistance with implementing models in `continue.dev`. There was also a conversation about decoupling ROCm from the main LM Studio application and a user guide for [configuring `continue.dev` with LM Studio](https://docs.continue.dev/reference/Model%20Providers/lmstudio).

- **Beta release observations and app versioning**: The community tested and reviewed recent beta releases, discussing tokenizer fixes and GPU offloading glitches. There’s a need for access to older versions, which is challenged by LM Studio's update policies, and a suggestion to maintain personal archives of preferred versions.

- **AI-driven creativity and quality of life concerns**: Engineers raised issues like the mismanagement of stop tokens by LM Studio and a tool's tendency to append irrelevant text in outputs. A frequent use-case-related complaint was an AI model not indicating its failure to provide a correct output by using an *"#ERROR"* message when necessary.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

**AI Alternatives for GPT-4 on Low-End Hardware**: Users debated on practical AI models for less powerful servers with suggestions like **"llama3 (70B-7B), mixtral 8x7B, or command r+"** for self-hosted AI similar to GPT-4.

**RWKV-TS Challenges RNN Dominance**: An [arXiv paper](https://arxiv.org/abs/2401.09093) introduces **RWKV-TS**, proposing it as a more efficient alternative to RNNs in time series forecasting, by effectively capturing long-term dependencies and scaling computationally.

**Model Selection Matters in Business Use**: In the choice of AI for business applications, it's crucial to consider use cases, tools, and deployment constraints, even with a limitation like the 7B model size. For tailored advice, members suggested focusing on specifics.

**Innovations and Integrations Abound**: From [Difoosion](https://gitlab.com/mad-moo/difoosion), a user-friendly web interface for Stable Diffusion, to *Ask Steve*, a Chrome extension designed to streamline web tasks using LLMs, community members are actively integrating AI into practical tools and workflows.

**Issues and Suggestions in Model Handling and Fine-Tuning**:
- A *[tutorial for fine-tuning BERT](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/BERT/Fine_tuning_BERT_(and_friends)_for_multi_label_text_classification.ipynb)* was shared.
- Concerns about non-deterministic model initializations were raised, with advice to save the model state for reproducibility.
- **Mistral-7b-0.3**'s context length handling and the quest for high-quality **meme generator models** indicate challenges and pursuits in model customization.
- For TPU users, guidance on using **Diffusers** with **GCP's TPU** is sought, indicating an interest in leveraging cloud TPUs for diffusion models.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **iOS Compatibility Question Marks**: Members debated whether **ChatGPT** functioned with iOS 18 beta, recommending sticking to stable versions like iOS 17 and noting that beta users are under NDA regarding new features. No clear consensus was reached on compatibility.

- **Open Source Ascending**: The release of an open-source model by [DeepSeek AI](https://x.com/deepseek_ai/status/1802680388256768145) that outperforms **GPT-4 Turbo** in coding and math sparked debate about the advantages of open-source AI over proprietary models.

- **Database Deployments with LLMs**: For better semantic search and fewer hallucinations, a community member highlighted [OpenAI's Cookbook](https://cookbook.openai.com/examples/vector_databases/readme) as a resource for integrating vector databases with OpenAI's models.

- **GPT-4 Usage Ups and Downs**: Users expressed frustrations with access to GPT interactions, privacy settings on Custom GPTs, and server downtimes. The community provided workarounds and suggested monitoring [OpenAI's service status](https://status.openai.com) for updates.

- **Challenges with 3D Modeling and Prompt Engineering**: Conversations focused on the technicalities of generating shadow-less 3D models and the intricacies of preventing GPT-4 from mixing information. Members shared various strategies, including step-back prompting and setting explicit actions to guide the AI's output.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Stabilizing SD3 Models**: The discussion revolved around **SD3 models** facing stability hurdles, particularly with artifacts and training. Concerns were raised about **loss stabilization**, pinpointing issues like *non-uniform timestep sampling* and missing elements such as **qk norm**.

- **T2I Models Take the Stage**: The dialog highlighted interest in **open-source T2I (text-to-image) models**, notably for character consistency across scenes. Resources such as [Awesome-Controllable-T2I-Diffusion-Models](https://github.com/PRIV-Creation/Awesome-Controllable-T2I-Diffusion-Models) and [Theatergen](https://github.com/donahowe/Theatergen) were recommended for those seeking reliable multi-turn image generation.

- **Logical Limitbreak**: A member brought attention to current challenges in **logical reasoning** within AI, identifying **Phi-2**'s "severe reasoning breakdown" and naming bias in LLMs when tackling AIW problems—a key point supported by [related research](https://arxiv.org/abs/2406.02061).

- **Boosting Deductive Reasoning**: Queries about hybrid methods for enhancing deductive reasoning in LLMs directed to [Logic-LM](https://arxiv.org/abs/2305.12295), a method that combines LLMs with symbolic AI solvers to improve logical problem-solving capabilities.

- **Video Generation Innovation**: Fudan University's **Hallo model** sparked excitement, a tool capable of video generation from single images and audio, with potential application alongside Text-to-Speech systems. A utility to run it locally was shared from [FXTwitter](https://fxtwitter.com/cocktailpeanut/status/1802376983021580428), highlighting community interest in practical integrations.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **200T Parameter Model: AGI or Fantasy?**: Discussions about the accessibility of a hypothetical **200T parameter model** surfaced, highlighting both the limits of current compute capabilities for most users and the humor in staking an AGI claim for such models.

- **Competing at the Big Model Rodeo**: Members juxtaposed the **Qwen7B** and **Llama3 8B** models, acknowledging **Llama3 8B** as the dominant contender in performance. The problem of custom training configurations for Llama3 models was tackled, with a [solution shared](https://github.com/xzuyn/axolotl/blob/dan_metharme/src/axolotl/prompt_strategies/customllama3.py) to address the `chat_template` setting issues.

- **Optimization Quest for PyTorch GPUs**: Requests for optimization feedback directed towards various GPU setups in PyTorch have yielded a trove of diverse community experiences ranging from AMD MI300X to RTX 3090, Google TPU v4, and 4090 with tinygrad.

- **Navigating Axolotl's Development Labyrinth**: An issue halting the development with the **Llama3 models** was found and traced to a specific commit, which helped identify the problem but emphasized the need for a fix in the main branch. Instructions for setting inference parameters and fine-tuning vision models within **Axolotl** were detailed for users.

- **Data Extraction with a Twist of Structure**: Community showcase hinted at positive results after fine-tuning LLMs with **Axolotl**, particularly in transforming unstructured press releases into structured outputs. A forthcoming post promises to expound on the use of the OpenAI API's function calling to enhance LLM accuracy in this task. The author points to a [detailed post](https://mlops.systems/posts/2024-06-15-isafpr-first-finetune.html) for more information.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Pro Language Partnerships!**: Perplexity AI has inked a deal with **SoftBank**, offering **Perplexity Pro** free for one year to SoftBank customers. This premium service, typically costing 29,500 yen annually, is set to enhance users' exploration and learning experiences through AI ([More info on the partnership](https://pplx.ai/softbank)).

- **Circumventing AB Testing Protocols? Think Again**: Engineers discussed how to bypass A/B testing for **Agentic Pro Search**, with a Reddit link provided; however, concerns about integrity led to reconsideration. The community also tackled a myriad of usage questions on **Perplexity features**, debated the merits of Subscriptions to Perplexity versus ChatGPT, and raised critical privacy issues concerning web crawling practices.

- **API Access is the Name of the Game**: Members expressed urgency for closed-beta access to the Perplexity API, emphasizing the impact on launching projects like those at **Kalshi**. Troubleshooting Custom GPT issues, they exchanged tips to enhance its "ask-anything" feature using schema-based explanations and error detail to improve action/function call handling.

- **Community Leaks and Shares**: Links to **Perplexity AI searches** and pages on varied topics, from data table management tools (Tanstack Table) to Russia’s pet food market and elephant communication strategies, were circulated. A mishap with a publicized personal document on prostate health led to community-driven support resolving the issue.

- **Gaming and Research Collide**: The shared content within the community included a mix of academic interests and gaming culture, demonstrated by a publicly posted page pertaining to **The Elder Scrolls**, hinting at the intersecting passions of the technical audience involved.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Neurons Gaming with Doom**: An innovative approach brings together biotech and gaming as living neurons are used to play the video game **Doom**, detailed in a [YouTube video](https://youtu.be/c-pWliufu6U). This could be a step forward in understanding biological process integration with digital systems.

- **AI Ethics and Bias in the Spotlight**: A critical take on AI discussed in a [ResearchGate paper](https://www.researchgate.net/publication/378191925_Automated_Bias_and_Indoctrination_at_Scale_Is_All_You_Need) calls attention to AI's trajectory towards promulgating human bias and aligned corporate interests, naming "stochastic parrots" as potential instruments of cognitive manipulation.

- **LLM Merging and MoE Concerns**: An engaged debate over the practical use of **Mixture of Experts (MoE)** models surfaced, contemplating the effectiveness of model merging versus comprehensive fine-tuning, citing a PR on [llama.cpp](https://github.com/ggerganov/llama.cpp/pull/6453) and MoE models on [Hugging Face](https://huggingface.co/Kquant03/CognitiveFusion-4x7B-bf16-MoE).

- **Llama3 8B Deployment Challenges**: On setting up and deploying **Llama3 8B**, it was advised to utilize platforms like **unsloth qlora**, **Axolotl**, and **Llamafactory** for training and **lmstudio** or **Ollama** for running fast OAI-compatible endpoints on Apple's M2 Ultra, bringing light to tooling for model deployment.

- **Autechre Tunes Stir Debate**: Opinions and emotions around Autechre's music led to sharing of contrasting YouTube videos, ["Gantz Graf"](https://www.youtube.com/watch?v=ev3vENli7wQ) and ["Altibzz"](https://www.youtube.com/watch?v=m3ZyEGTIsvE), showcasing the diverse auditory landscapes crafted by the electronic music duo.

- **Explore Multiplayer AI World Building**: Suggestion raised for collaborative creation in **WorldSim**, as members discussed enabling multiplayer features for AI-assisted co-op experiences, while noting censorship from the model provider could influence WorldSim AI content.

- **NVIDIA's LLM Rolls Out**: Introductions to NVIDIA's **Nemotron-4-340B-Instruct** model, accessible on [Hugging Face](https://huggingface.co/nvidia/Nemotron-4-340B-Instruct), kindled talks on synthetic data generation and strategic partnerships, highlighting the company's new stride into language processing.

- **OpenAI's Profit-Minded Pivot**: OpenAI's CEO [Sam AltBody](https://x.com/aaronpholmes/status/1801785687030829240?s=46) has indicated a potential shift from a non-profit to a for-profit setup, aligning closer to competitors and affecting the organizational dynamic and future trajectories within the AI industry.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo Functions Discussion Heats Up**: Engineers critiqued the Mojo manual's treatment of `def` and `fn` functions, highlighting the ambiguity in English phrasing and implications for type declarations in these function variants. This led to a consensus that while `def` functions permit optional type declarations, `fn` functions enforce them; a nuanced distinction impacting code flexibility and type safety.

- **Meetup Alert: Mojo Community Gathers**: An upcoming *Mojo Community Meeting* was announced, featuring talks on constraints, Lightbug, and Python interoperability, inviting participants to [join via Zoom](https://modul.ar/community-meeting-zoom). Moreover, benchmark tests revealed that Mojo's Lightbug outstrips Python FastAPI in single-threaded performance yet falls short of Rust Actix, sparking further discussion on potential runtime costs entailed by function coloring decisions.

- **Fresh Release of Mojo 24.4**: The Mojo team has rolled out version 24.4, introducing core language and standard library improvements. Detail-oriented engineers were pointed towards a [blog post](https://www.modular.com/blog/whats-new-in-mojo-24-4-improved-collections-new-traits-os-module-features-and-core-language-enhancements) for a deep dive into the new traits, OS module features, and more.

- **Advanced Mojo Techniques Uncovered**: Deep technical discussions unveiled challenges and insights in Mojo programming, from handling 2D Numpy arrays and leveraging `DTypePointer` for efficient SIMD operations to addressing bugs in casting unsigned integers. Notably, a discrepancy involving `alias` usage in CRC32 table initialization sparked an investigation into unexpected casting behaviors.

- **Nightly Mojo Compiler on the Horizon**: Engineers were informed about the new **nightly builds** of the Mojo compiler with the release of versions `2024.6.1505`, `2024.6.1605`, and `2024.6.1705`, along with instructions to update via `modular update`. Each version's specifics could be examined via provided GitHub diffs, showcasing the platform's continuous refinement. Additionally, the absence of external documentation for built-in MLIR dialects was noted, and enhancements such as direct output expressions in REPL were requested.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Replication of OpenAI's Generalization Techniques by Eleuther**: EleutherAI's interpretability team successfully replicated OpenAI's "weak-to-strong" generalization on open-source LLMs across 21 NLP datasets, publishing a detailed account of their findings, positive and negative, on experimenting with variants like strong-to-strong training and probe-based methods, [here](https://blog.eleuther.ai/weak-to-strong/).

- **Job Opportunities and Navigating CommonCrawl**: The AI Safety Institute announced new roles with visa assistance for UK relocation on their [careers page](https://aisi.gov.uk/careers), while discussions on efficiently processing CommonCrawl data mentioned tools like [ccget](https://github.com/allenai/ccget) and [resiliparse](https://resiliparse.chatnoir.eu/en/latest/man/parse/html.html).

- **Model Innovations and Concerns**: From exploring **RWKV-CLIP**, a vision-language model, to concerns about content generated by diffusion models and the stealing of commercial model outputs, the community addressed various aspects of AI model development and security. The effectiveness of the **Laprop** optimizer was debated, and papers ranging from those on online adaptation to those on "stealing" embedding models were shared, with a key paper being [here](https://arxiv.org/abs/2406.09355).

- **Evolving Optimization and Scaling Laws**: A member's critique of a hypernetwork-based paper sparked conversations on the value and comparison of hypernetworks with Hopfield nets. Interested parties ventured into the scaling of scaling laws, considering online adaptation for LLMs and citing Andy L. Jones' concept of offsetting training compute against inference compute.

- **Interpretability Insights on Sparse Autoencoders**: Interpretability research centered around Sparse Autoencoders, with a paper proposing a framework for evaluating feature dictionaries in tasks like indirect object identification with GPT-2, and another highlighting "logit prisms" decomposing logit output components, as documented in [this article](https://neuralblog.github.io/logit-prisms).

- **Need for A Shared Platform for Model Evaluation**: Calls were made for a platform to share and validate evaluation results of AI models, particularly for those using Hugging Face and seeking to verify the credibility of closed-source models, highlighting the need for comprehensive and transparent evaluation metrics.

- **Awaiting Code Release for Vision-Language Project**: A specific request for a release date for code related to **RWKV-CLIP** was directed to the [GitHub Issues page](https://github.com/deepglint/RWKV-CLIP/issues) of the project, indicating a demand for access to the latest advancements in vision-language representation models.



---



## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **Apple Sidesteps NVIDIA in AI**: Apple's WWDC reveal details its avoidance of NVIDIA hardware, preferring their in-house AXLearn on TPUs and Apple Silicon, potentially revolutionizing their AI development strategy. The technical scoop is unpacked in a [Trail of Bits blog post](https://blog.trailofbits.com/2024/06/14/understanding-apples-on-device-and-server-foundations-model-release/).

- **Embeddings and Fine-Tuning**: Enthusiasm emerges for fine-tuning methodologies, with discussions ranging from embedding intricacies, highlighted by resources like [Awesome Embeddings](https://github.com/eifuentes/awesome-embeddings), to specific practices like adapting TinyLlama for unique narration styles, detailed in a developer's [blog post](https://gabrielchua.me/posts/finetuning-tinyllama-axolotl-beginner/).

- **Prompt Crafting Innovations**: Mention of **Promptfoo** and **inspect-ai** indicates a trend toward more sophisticated prompt engineering tools, with the community weighing functionality and user-friendliness. Diverging preferences suggest such tools are pivotal for refined human-AI interaction schemes.

- **Crediting Confusions Cleared**: Participants express mixed signals about course credits across platforms like **LangSmith** and **Replicate**, with reminders and clarifications surfacing through communal support. The difference between beta and course credits was elucidated for concerned members.

- **Code Llama Leaps Forward**: Conversations ignited by the release of [Code Llama](https://huggingface.co/blog/codellama) show a commitment to enhancing programming productivity. Curiosity about permissible variability between Hugging Face and GitHub configuration formats for Code Llama indicates the precision required for fine-tuning these purpose-built models.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Sakana AI Joins the Unicorn Club**: Sakana AI, pushing past traditional transformer models, has secured a monster $1B valuation from heavy-hitters like NEA, Lux, and Khosla, marking a significant milestone for the AI community. Full financial details can be ferreted out in [this article](https://www.theinformation.com/articles/openais-japanese-rival-gets-1-billion-valuation-from-silicon-valley-investors).

- **Next-Gen Video Generation with Runway's Gen-3 Alpha**: Runway has turned heads with its Gen-3 Alpha, flaunting the ability to create high-quality videos replete with intricate scene transitions and a cornucopia of cinematographic styles, setting a new bar in video generation which can be explored [here](http://runwayml.com/gen-3-alpha).

- **DeepMind's Video-Turned-Audio Breakthrough**: Google DeepMind's new video-to-audio technology aims to revolutionize silent AI video generations by churning out a theoretically infinite number of tracks tailored to any video, as showcased in [Rowan Cheung's examples](https://x.com/rowancheung/status/1802734770117333257).

- **Wayve's Impressive Take on View Synthesis**: Wayve claims a fresh victory in AI with a view synthesis model that leverages 4D Gaussians, promising a significant leap in generating new perspectives from static images, detailed in [Jon Barron's tweet](https://x.com/jon_barron/status/1802758455830437975).

- **Speculations Stir on OpenAI's Future**: Whispers of OpenAI's governance shake-up suggest a potential pivot to a for-profit stance with musings of a subsequent IPO, stirring debate within the community; some greet with derision while others await concrete developments, as covered in [The Information](https://www.theinformation.com/articles/openai-ceo-says-company-could-become-benefit-corporation-akin-to-rivals-anthropic-xai?utm_campaign=Editorial&utm_content=Article&utm_medium=organic_social&utm_source=twitter) and echoed by [Jacques Thibault's tweet](https://x.com/jacquesthibs/status/1801782465364640247?s=46).



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **RAG and Agents Drawn Clear**: An [Excalidraw-enhanced slide deck](https://t.co/oibSiDjseO) was shared detailing the construction of Retrieval-Augmented Generation (RAG) and Agents, containing diagrams that elucidate concepts from simple to advanced levels.

- **Observability Integrated in LLM Apps**: A new module for instrumentation brings end-to-end observability to LLM applications through Arize integration, with a [guide available](https://t.co/cOBP9IOjro) detailing custom event/span handler instrumentation.

- **Knowledge Graphs Meet Neo4j**: Discussions around integrating Neo4j knowledge graphs with LlamaIndex focused on transforming Neo4j graphs into property graphs for LlamaIndex, with resources and documentation provided ([LlamaIndex Property Graph Example](https://docs.llamaindex.ai/en/latest/examples/property_graph/property_graph_neo4j/)).

- **Enhanced LLMs with Web Scraping Strategies**: Apublication discusses improving LLMs by combining them with web scraping and RAG, recommending tools such as [Firecrawl](https://medium.com/ai-advances/how-to-power-up-llms-with-web-scraping-and-rag-975a165587f6) for effective Markdown extraction, and Scrapfly for diverse output formats suitable for LLM preprocessing.

- **Practical Tutorials and AI Event Highlights**: Practical step-by-step guides for full-stack agents and multimodal RAG pipelines were made available, and [AI World's Fair highlighted](https://t.co/O6WAkbI9jt) with noteworthy speakers shared their knowledge on AI and engineering, enhancing the community's skill set and understanding of emerging AI trends.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Script Snafu and OpenCL Woes**: Discussions around `autogen_stubs.sh` revealed that `clang2py` breaks the indentation, but this was found unnecessary for GPU-accelerated tinygrad operations. Meanwhile, George Hotz suggested fixing OpenCL installation and verifying with `clinfo` due to errors affecting tinygrad's GPU functionality.

- **Enhanced OpenCL Diagnostics on the Horizon**: A move to improve OpenCL error messages is underway, with a [proposed solution](https://github.com/tinygrad/tinygrad/pull/5004) that autonomously generates messages from available OpenCL headers, aiming to ease developers' debugging process.

- **Deciphering Gradient Synchronization**: In a bid to demystify gradient synchronization, George Hotz affirmed Tinygrad's built-in solution within its optimizer, touting its efficiency compared to the more complex Distributed Data Parallel in PyTorch.

- **Chasing PyTorch's Tail with Ambitions and Actions**: George Hotz conveyed ambitions for tinygrad to eclipse PyTorch in terms of speed, simplicity, and reliability. Although currently trailing, particularly in LLM training, tinygrad's clean design and strong foundation exude promise.

- **Precision Matters in the Kernel Cosmos**: A technical exchange discussed strategies for incorporating mixed precision in models, where George Hotz recommended late casting for efficiency gains and the use of `cast_` methods, highlighting a critical aspect of optimizing for computation-heavy tasks.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **GPT Notes App Unveiled**: An LLM client and notes app hybrid has been demonstrated, featuring dynamic inclusion of notes, vanilla JavaScript construction, and local storage of notes and API keys in the browser; however, it currently lacks mobile support. The app is showcased with a [Codepen](https://codepen.io/bulgakoff08/project/editor/DnJLrG) and a [full-screen deployment](https://000700836.deployed.codepen.website/).

- **OpenRouter Gripes and Glimpses**: OpenRouter requires at least one user message to prevent errors, with users suggesting the use of the `prompt` parameter; formatting tools like [PDF.js and Jina AI Reader](https://jina.ai/reader/) are recommended for PDF pre-processing to enhance LLM compatibility.

- **Censorship Consternation with Qwen2**: The Qwen2 model is facing user criticism for excessive censorship, while the less restrictive Dolphin Qwen 2 model garners recommendation for its more realistic narrative generation.

- **Gemini Flash Context Clash**: Questions arise over Gemini Flash's token limits, with OpenRouter listing a 22k limit, in contrast to the 8k tokens cited in the Gemini Documentation; the discrepancy is attributed to OpenRouter's character counting to align with Vertex AI's pricing.

- **Rate Limits and Configuration Conversations**: Users discuss rate limits for models like GPT-4o and Opus and model performance configurations; for further information, the OpenRouter [documentation on rate limits](https://openrouter.ai/docs/limits) proves informative, and there is a focus on efficiency in API requests and usage.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **LangChain API Update Breaks TextGen**: A recent **API update** has disrupted **textgen integration** in **LangChain**, with members seeking solutions in the general channel.

- **Technical Troubleshooting Takes the Stage**: Users discussed challenges with **installing langchain_postgres** and a **ModuleNotFoundError** caused by an update to **tenacity version 8.4.0**; reverting to **version 8.3.0** fixed the issue.

- **LangChain Knowledge Sharing**: Questions around **LangChain** usage emerged, including transitioning from Python to **JavaScript implementations**, and handling of models like **Llama 3** or **Google Gemini** for local deployment.

- **Tech enthusiasts Intro New Cool Toys**: Innovative projects were highlighted such as **R2R's automatic knowledge graph construction**, an **interactive map for Collision events**, and **CryptGPT**, which is a privacy-preserving approach to LLMs using Vigenere cipher.

- **AI for the Creatively Inclined**: Community members announced a **custom GPT for generating technical diagrams**, and **Rubik's AI**, a research assistant and search engine offering free premium with models like **GPT-4 Turbo** to beta testers.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

**OtterTune Exits Stage Left**: [OtterTuneAI has shut down](https://x.com/andy_pavlo/status/1801687420330770841?s=46&t=Tc6nPt_FP2Ybqya6_6Xu-w) following a failed acquisition deal, marking the end of their automatic database tuning services.

**Apple and OpenAI Make Moves**: Apple released optimized on-device models on Hugging Face, such as [DETR Resnet50 Core ML](https://huggingface.co/apple/coreml-detr-semantic-segmentation), while OpenAI faced criticism from Edward Snowden for adding former NSA Director Paul M. Nakasone to its board.

**DeepMind Stays in Its Lane**: In recent community discussions, it was clarified that DeepMind has not been contributing to specific AI projects, debunking earlier speculation.

**Runway and Anthropic Innovate**: Runway announced their new video generation model, Gen-3 Alpha, on [Twitter](https://x.com/runwayml/status/1802691475391566108?s=46&t=90xQ8sGy63D2OtiaoGJuww), while Anthropic publicized important research on AI models hacking their reward systems in a [blog post](https://anthropic.com/research/reward-tampering).

**Future of AI in Collaboration and Learning**: Prime Intellect is set to open source sophisticated models DiLoco and DiPaco, Bittensor is making use of The Horde for decentralized training, and a [YouTube video](https://www.youtube.com/watch?v=mdKjMPmcWjY&t=23s) shared among users breaks down optimizers critical for model training.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **AGI: Fantasy or Future?**: Members shared their perspectives on a [YouTube video about AGI](https://youtu.be/4JF1V2hzGKE?si=gKCvVxBpwsGTD6ow), discussing the balance between skepticism and the potential for real progress that parallels the aftermath of the dot-com bubble.

- **Next.js Migrations Ahead**: There's a collaborative push to utilize Next.js App Router for the Cohere toolkit, aiming at better code portability and community contribution, details of which are in [GitHub issue #219](https://github.com/cohere-ai/cohere-toolkit/issues/219).

- **C4AI by Cohere**: Nick Frosst invites to a C4AI talk via a [Google Meet link](https://meet.google.com/ibt-wsgv-kbq?hs=122&authuser=0), offering an avenue for community engagement on LLM advancements and applications.

- **Command Your Browser**: A [free Chrome Extension](https://www.asksteve.to) has been released, baking **LLMs** into Chrome to boost productivity, while an [interactive Collision map](https://collision.talewind.ai/) with AI chat features showcases events using modern web tech stacks.

- **Developer Touch Base**: Cohere is hosting Developer Office Hours with David Stewart for a deep dive into API and model intricacies; interested community members can join [here](https://discord.gg/jy5XFg5GDh?event=1248300905703673987) and post their questions on the mentioned thread for dedicated support.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Frozen Model Mystery Solved**: Engineers reported instances of a model **freezing during coding**, but it was determined that patience pays off as the model generally completes the task, albeit with a deceptive pause.

- **Tech Support Redirect**: A query about **Windows installation issues** for a model led to advice pointing the user towards a specific help channel for more targeted assistance.

- **Model Memory Just Got Better**: A member celebrated a breakthrough with **memory implementation**, achieving success they described in rudimentary terms; meanwhile, **Llama 3 Instruct 70b** and **8b** performance details were disclosed through a [Reddit post](https://www.reddit.com/r/LocalLLaMA/comments/1cal17l/llm_comparisontest_llama_3_instruct_70b_8b/).

- **Cyber Hat Countdown**: An open-source, AI-enabled “cyber hat” project sparked interest among engineers for its originality, potential for innovation, and an open invite for collaboration [watch here](https://www.youtube.com/watch?v=71p9DcGqNDc); similarly, [Dream Machine’s](https://lumalabs.ai/dream-machine) text and image-based realistic video creation signaled strides in AI model capabilities.

- **Semantic Search Synergy**: Conversation turned to the fusion of voice-based **semantic search and indexing** with a vector database holding audio data, leveraging the prowess of an LLM to perform complex tasks based on vocal inputs, suggesting the nascent power of integrated tech systems.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Tuning Into Torchtune's Single Node Priorities**: Torchtune is focusing on optimizing **single node training** before considering multi-node training; it utilizes the `tune run` command as a wrapper for `torch run`, which might support multi-node setups with some adjustments, despite being untested for such use.
  
- **Unlocking Multi-Node Potential in Torchtune**: Some members shared how to potentially configure Torchtune for **multi-node training**, suggesting the use of `tune run —nnodes 2` and additional tools like **TorchX** or **slurm** for script execution and network coordination across nodes, referencing the [FullyShardedDataParallel documentation](https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.ShardingStrategy) as a resource for sharding strategies.



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **Llama3 Sticks to Its Roots**: Despite the introduction of a German model, the **Llama3 tokenizer** has *not* been modified and remains identical to the base Llama3, raising questions about its efficiency in handling German tokens.
- **Token Talk**: Concerns emerged over the unchanged tokenizer, with engineers speculating that not incorporating specific German tokens *could substantially reduce the context window* and affect the quality of embeddings.
- **Comparing Llama2 and Llama3 Token Sizes**: *Inquisitive minds* noted that **Llama3's tokenizer is notably 4 times larger than Llama2's**, leading to questions about its existing efficacy with the German language and potential unrecognized issues.




---



## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

**Heralding Data Engineering Job Security**: ChatGPT's burgeoning role in the tech landscape drew humor-inflected commentary that it represents an infinite job generator for data engineers.

**Thoughtbot Clears the Fog on LLMs**: The guild appreciated a [guide by Thoughtbot](https://thoughtbot.com/blog/understanding-open-source-llms) for its lucidity in dissecting the world of Large Language Models, specifically for their delineation of Base, Instruct, and Chat models which can aid beginners.

**New Kid on the Search Block**: Turso's latest release integrates [native vector search](https://turso.tech/blog/turso-brings-native-vector-search-to-sqlite) with SQLite, which aims at enhancing the AI product development experience by replacing the need for independent extensions like [sqlite-vss](https://github.com/asg017/sqlite-vss).



---



## [AI Stack Devs (Yoko Li)](https://discord.com/channels/1122748573000409160) Discord

- **In Search of Hospital AI Project Name**: User gomiez inquired about the name of the hospital AI project within the **AI Stack Devs** community. There was no additional context or responses provided to further identify the project.



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Llama as Firefox's New Search Companion?**: A guild member, cryovolcano., inquired about the possibility of integrating **llamafile** with **tinyllama** as a search engine in the Firefox browser. No further details or context about the implementation or feasibility were provided.



---


The **LLM Perf Enthusiasts AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **YAIG (a16z Infra) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1251251945201729567)** (723 messages🔥🔥🔥): 

- **Lack of Trust in SD3 License Creates Chaos**: There are significant concerns over Stability AI's new license for SD3, leading to temporary banning of SD3-related content on Civitai due to the perceived legal ambiguities. [Civitai announcement](https://civitai.com/articles/5732) mentions that "legal team review" is underway.
- **Community Frustration and Critics' Backlash**: Many users voice their frustrations and criticisms towards Stability AI's confusing license and handling of SD3’s release. One user notes, *“The worst base model release yet… I just wanted nice hands.”*
- **Inquiry and Troubleshooting in ComfyUI**: Several users discuss issues and fixes for ComfyUI setup, particularly around custom nodes installations and dependencies like cv2. One user shared a helpful [ComfyUI install tutorial](https://www.youtube.com/watch?v=Di1KqPXxx2Y&t=30s).
- **Interest in Model Applications and Alternatives**: Users explore models for various art styles and uses, such as retro dark fantasy and video generation with animatediff tools. User discussions imply the open-source community might pivot attention to alternative models and tools post-SD3 controversy.
- **Youtuber Olivio Sarikas Faces Scrutiny**: Multiple users discuss the youtuber's video on SD3's license, accusing him of spreading misinformation and overblown fears about the legal implications, with one stating, *"Olivio had all the information... and willfully misreported it to farm views.”*
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=r8"> - YouTube</a>: no description found</li><li><a href="https://huggingface.co/PenelopeSystems/penelope-palette">PenelopeSystems/penelope-palette · Hugging Face</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=HmxbsQDRTSs">SD3 - Absurd new License. Stability AI asks you to destroy your Models!</a>: The new SD3 License from Stability AI asks you to destroy your models. The new Creator License has some pretty absurd Terms. Including limiting you to only 6...</li><li><a href="https://www.youtube.com/w">YouTube</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=HmxbsQDRTSs&t=2s&ab_channel=OlivioSarikas">SD3 - Absurd new License. Stability AI asks you to destroy your Models!</a>: The new SD3 License from Stability AI asks you to destroy your models. The new Creator License has some pretty absurd Terms. Including limiting you to only 6...</li><li><a href="https://youtu.be/r8oA42saTNI?si=g0aIpBHftWleWpaW">Stable Diffusion 3&#39;s Concerning Fine Print: What Studios and Artists Should Know About the New Terms</a>: We took a look at the fine print of Stable Diffusion 3&#39;s new licenses and break down what you need to know if you are planning to use SD3 for commercial or n...</li><li><a href="https://onnx.ai/">ONNX | Home</a>: no description found</li><li><a href="https://youtu.be/HmxbsQDRTSs">SD3 - Absurd new License. Stability AI asks you to destroy your Models!</a>: The new SD3 License from Stability AI asks you to destroy your models. The new Creator License has some pretty absurd Terms. Including limiting you to only 6...</li><li><a href="https://dreamstudio.ai/generate>)">DreamStudio</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=Di1KqPXxx2Y&t=30s">SD3 IS HERE!! ComfyUI Workflow.</a>: SD3 is finally here for ComfyUI!Topaz Labs: https://topazlabs.com/ref/2377/HOW TO SUPPORT MY CHANNEL-Support me by joining my Patreon: https://www.patreon.co...</li><li><a href="https://civitai.com/articles/5732">Temporary Stable Diffusion 3 Ban | Civitai</a>: Unfortunately, due to a lack of clarity in the license associated with Stable Diffusion 3 , we are temporarily banning: All SD3 based models All mo...</li><li><a href="https://huggingface.co/ByteDance/Hyper-SD">ByteDance/Hyper-SD · Hugging Face</a>: no description found</li><li><a href="https://hyper-sd.github.io">Hyper-SD: Trajectory Segmented Consistency Model for Efficient Image Synthesis</a>: no description found</li><li><a href="https://github.com/Picsart-AI-Research/StreamingT2V?tab=readme-ov-file">GitHub - Picsart-AI-Research/StreamingT2V: StreamingT2V: Consistent, Dynamic, and Extendable Long Video Generation from Text</a>: StreamingT2V: Consistent, Dynamic, and Extendable Long Video Generation from Text - Picsart-AI-Research/StreamingT2V</li><li><a href="https://tenor.com/bjN7k.gif">Shoe Nike GIF - Shoe Nike Design Shoe - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/Fannovel16/comfyui_controlnet_aux">GitHub - Fannovel16/comfyui_controlnet_aux: ComfyUI&#39;s ControlNet Auxiliary Preprocessors</a>: ComfyUI&#39;s ControlNet Auxiliary Preprocessors. Contribute to Fannovel16/comfyui_controlnet_aux development by creating an account on GitHub.</li><li><a href="https://x.com/xqdior/status/1801995625745252765">Tweet from D̷ELL (@xqdior)</a>: Stable Diffusion 3 8Bを搭載した、Stable Image API Ultraをみなさまに体験していただきました。生成いただいた画像をまとめましたので、レポートさせて頂きます。  Stable Diffusion 3 最上級モデルの性能を、ぜひご覧くださいませ。 https://qiita.com/nqdior/items/ce894b5c5382b2029ced #Qiita</li><li><a href="https://imgur.com/gallery/sd-web-ui-3d-lora-ZkV6KkP">sd web ui 3d lora</a>: Discover the magic of the internet at Imgur, a community powered entertainment destination. Lift your spirits with funny jokes, trending memes, entertaining gifs, inspiring stories, viral videos, and ...</li><li><a href="https://fxtwitter.com/zhozho672070/status/1802037549864804820?s=46&t=AZs45ckJ7UUM_kJZcxnR_w">Tweet from -Zho- (@ZHOZHO672070)</a>: InstantX 刚刚又连续上传了 4 个 SD3 Medium 的 ControlNet 模型  Canny（1024）：https://huggingface.co/InstantX/SD3-Controlnet-Canny Pose：https://huggingface.co/InstantX/SD3-Controlnet-Pose Tile（还在上传）：https://huggingfa...</li><li><a href="https://github.com/comfyanonymous/ComfyUI/releases">Releases · comfyanonymous/ComfyUI</a>: The most powerful and modular stable diffusion GUI, api and backend with a graph/nodes interface. - comfyanonymous/ComfyUI</li><li><a href="https://stability.ai/creator-license-agreement">Professional Membership Agreement &mdash; Stability AI</a>: no description found</li><li><a href="https://huggingface.co/ptx0/sd3-reality-mix">ptx0/sd3-reality-mix · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/nerijs/pixel-art-medium-128-v0.1">nerijs/pixel-art-medium-128-v0.1 · Hugging Face</a>: no description found</li><li><a href="https://civitai.com/">Civitai: The Home of Open-Source Generative AI</a>: Explore thousands of high-quality Stable Diffusion models, share your AI-generated art, and engage with a vibrant community of creators</li><li><a href="https://www.reddit.com/r/StableDiffusion/s/ZEV6SEsTHU">Reddit - Dive into anything</a>: no description found</li><li><a href="https://civitai.com/models/4201/realistic-vision-v60-b1">Realistic Vision V6.0 B1 - V5.1 Hyper (VAE) | Stable Diffusion Checkpoint | Civitai</a>: Recommendations for using the Hyper model: Sampler = DPM SDE++ Karras or another / 4-6+ steps CFG Scale = 1.5-2.0 ( the lower the value, the more m...
</li>
</ul>

</div>
  

---



### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1251252320520372366)** (517 messages🔥🔥🔥): 

<ul>
    <li><strong>Work in Progress on Ollama Support:</strong> A member stated, "Unfortunately the Ollama support got kinda delayed," but reassured that they are "working with the amazing Ollama team." The support is around 80% complete.</li>
    <li><strong>Validation Issues in Template Fine-Tuning:</strong> A member queried about validating templates for use with Ollama and discussed issues with learning rates and model configurations. They noted, "I had acceptable results with my merged models but it turns sick sometimes."</li>
    <li><strong>Push to HF Merged Models Issue:</strong> A member raised a problem where running `model.push_to_hub_merged` only saves the adapter but not the full merged model. Another member suggested a workaround involving manually merging before uploading.</li>
    <li><strong>Training Performance Comparisons:</strong> A user highlighted UnsLoath's performance in training speed, claiming it was "24% faster than torch.compile() torchtune for 4090" based on their benchmarking results. The UnsLoath team acknowledged this and discussed the possibility of releasing an academic paper on it.</li>
    <li><strong>Upcoming Multi-GPU Support:</strong> The team confirmed that they will be implementing multi-GPU support up to 8 GPUs. A small group is getting early access for initial testing.</li>
</ul>

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://discord.gg/ollama?event=1251334371349233814">Join the Ollama Discord Server!</a>: Check out the Ollama community on Discord - hang out with 49602 other members and enjoy free voice and text chat.</li><li><a href="https://huggingface.co/nyunai/nyun-c2-llama3-50B">nyunai/nyun-c2-llama3-50B · Hugging Face</a>: no description found</li><li><a href="https://arxiv.org/abs/2406.07394">Accessing GPT-4 level Mathematical Olympiad Solutions via Monte Carlo Tree Self-refine with LLaMa-3 8B</a>: This paper introduces the MCT Self-Refine (MCTSr) algorithm, an innovative integration of Large Language Models (LLMs) with Monte Carlo Tree Search (MCTS), designed to enhance performance in complex m...</li><li><a href="https://www.youtube.com/watch?v=FHsEW0HpuoU>">Lecture 10: Build a Prod Ready CUDA library</a>: Slides https://drive.google.com/drive/folders/158V8BzGj-IkdXXDAdHPNwUzDLNmr971_?usp=sharingSpeaker: Oscar Amoros Huguet</li><li><a href="https://www.youtube.com/watch?v=JuPwfQlPUt0">KAN: Kolmogorov-Arnold Networks</a>: A Google Algorithms Seminar TechTalk, presented by Ziming Liu, 2024-06-04ABSTRACT: Inspired by the Kolmogorov-Arnold representation theorem, we propose Kolmo...</li><li><a href="https://wandb.ai/augmxnt/train-bench/reports/torchtune-vs-axolotl-vs-unsloth-Trainer-Comparison--Vmlldzo4MzU3NTAx">torchtune vs axolotl vs unsloth Trainer Performance Comparison</a>: A performance comparison of various trainers and GPUs. Made by lhl using Weights &amp; Biases</li><li><a href="https://tenor.com/view/card-codes-gif-21814106">Card Codes GIF - Card Codes - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://x.com/ryanels4/status/1801898008755241251?s=46">Tweet from Ryan Els (@RyanEls4)</a>: AI revealed 😲</li><li><a href="https://github.com/unslothai/unsloth/issues/611">save_pretrained_merged doesn&#39;t merge the model · Issue #611 · unslothai/unsloth</a>: Problem My goal, I want to save the merged model as a GGUF file, but I&#39;m getting various errors. The deeper problem seems to be that merging lora+base model isn&#39;t saving a merged file. I think...</li><li><a href="https://github.com/unslothai/unsloth/wiki">Home</a>: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://x.com/aiDotEngineer/status/1791506805065216017">Tweet from AI Engineer (@aiDotEngineer)</a>: We&#39;re excited to announce our speakers!  CEO @Modular AI LEAD @MozillaAI ENG LEAD @OpenAI CEO @UnslothAI TBA @Microsoft TBA @AnthropicAI CEO @cognition_labs (Devin) CEO @anysphere (@cursor_ai) CTO...</li><li><a href="https://github.com/unslothai/unsloth/tree/main">GitHub - unslothai/unsloth: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory</a>: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/tree/main?tab=readme-ov-file#-key-features">GitHub - unslothai/unsloth: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory</a>: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://colab.research.google.com/github/oobabooga/text-generation-webui/blob/main/Colab-TextGen-GPU.ipynb#scrollTo=LGQ8BiMuXMDG">Google Colab</a>: no description found</li><li><a href="https://unsloth.ai/blog/contpretraining?s=08">Continued LLM Pretraining with Unsloth</a>: Make a model learn a new language by doing continued pretraining with Unsloth using Llama 3, Phi-3 and Mistral.</li><li><a href="https://colab.research.google.com/drive/1tEd1FrOXWMnCU9UIvdYhs61tkxdMuKZu?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/s/psE4lqVDsO">Reddit - Dive into anything</a>: no description found</li><li><a href="https://docs.google.com/spreadsheets/d/115Utf5SuQOEOCQpYCeg_zB-nowej7joTkya6gYla5Hc">Training Comparison</a>: Sheet1  Run,Trainer,GPU,Train (h),Max Mem (GiB),Power (W),Energy (kWh),Tok/s,Steps,Optimizer,Max Seq,Batch Size,Grad Accum,Global,Notes &lt;a href=&quot;https://wandb.ai/augmxnt/train-bench/runs/n59y6...</li><li><a href="http://wandb.ai/augmxnt/train-bench">augmxnt</a>: Weights & Biases, developer tools for machine learning
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1251331569403170877)** (17 messages🔥): 

- **Vintage Music Video Shared**: A member posted a [YouTube video](https://youtu.be/er-juqsr4nc?si=g3vtzgUa94GEcD-s) titled "Not like us (1962) full song," indicating their appreciation for older music styles. Another member complimented the taste, humorously noting they've only listened to anime songs.
- **Darude's Sandstorm and Musical Preferences**: A member jokingly shared [Darude - Sandstorm](https://www.youtube.com/watch?v=y6120QOlsfU), later revealing a genuine preference for Daft Punk's Discovery album, sharing it on [Spotify](https://open.spotify.com/track/098ttCNmncrO4YvqWUNMvn?si=0d0c602e42244d3e). Other users chimed in to share their favorite Daft Punk songs like "Lose Yourself to Dance."
- **Mixed Reactions to Gemma 2 on AI Studio**: A member mentioned trying out Gemma 2 27b on [aistudio.google.com](https://aistudio.google.com), noting the output was not impressive. Another user recognized the reference from Reddit, while others expressed excitement and anticipation for Gemma 2 and its potential capabilities.
- **Speculation and Excitement for Gemini 2.0**: Users speculated that the release of Gemma 2 could mean that Gemini 2.0 is also near. There was notable excitement about the potential for training the model, with one user contemplating renting a [Runpod 48GB instance](https://runpod.io) to thoroughly test the model's performance and capacity.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://youtu.be/er-juqsr4nc?si=g3vtzgUa94GEcD-s">Not like us (1962) full song</a>: K Dot senior is back to fulfill your requests for the full version of this song.</li><li><a href="https://www.youtube.com/watch?v=y6120QOlsfU">Darude - Sandstorm</a>: New Darude album &quot;Together&quot; out now → https://found.ee/Darude-TogetherNew &#39;Closer Together&#39; music video out now → https://youtu.be/edUBI3k2lUo?si=ynkxg7p7Ofa...</li><li><a href="https://open.spotify.com">Spotify - Web Player: Music for everyone</a>: Spotify is a digital music service that gives you access to millions of songs.</li><li><a href="https://open.spotify.com/track/098ttCNmncrO4YvqWUNMvn?si=0d0c602e42244d3e">High Life</a>: Daft Punk · Song · 2001
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1251258835419660319)** (304 messages🔥🔥): 

- **Facing issues with Triton on Windows**: A member reported issues installing Triton on Windows 11 even after setting up Visual C++ correctly. Assistance was provided by querying if `g++` or `clang++` could be called from the terminal.

- **Data Preparation Tutorial Request**: A member inquired about a data preparation tutorial for Unsloth fine-tuning similar to [OpenAI's chat fine-tuning data prep notebook](https://cookbook.openai.com/examples/chat_finetuning_data_prep). Another member cited a plan to create a tutorial and recommended a related [YouTube video](https://youtu.be/3eq84KrdTWY).

- **Model training crashes during saving**: A member experienced crashes while training the Yi model during the last saving steps, suspecting memory or disk space issues. It was suggested to check available memory and disk space, and a [link to Unsloth's saving issues on GitHub](https://github.com/unslothai/unsloth/wiki#saving-to-gguf--vllm-16bit-crashes) was provided.

- **Issues with batch size and gradient accumulation**: A member questioned the discrepancy in VRAM usage when adjusting batch size and gradient accumulation. Discussions clarified that gradient accumulation steps act similar to increasing batch size, and experimenting with larger batch sizes was recommended.

- **Error with quantization_method in save.py**: A bug was identified where `quantization_method` was mishandled as a string, leading to errors. A workaround involved passing `quantization_method` as a list, and a [pull request](https://github.com/unslothai/unsloth/pull/651) to fix the bug was submitted.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html#ubuntu">CUDA Quick Start Guide</a>: no description found</li><li><a href="https://youtu.be/3eq84KrdTWY">Llama 3 Fine Tuning for Dummies (with 16k, 32k,... Context)</a>: Learn how to easily fine-tune Meta&#39;s powerful new Llama 3 language model using Unsloth in this step-by-step tutorial. We cover:* Overview of Llama 3&#39;s 8B and...</li><li><a href="https://cookbook.openai.com/examples/chat_finetuning_data_prep">Data preparation and analysis for chat model fine-tuning | OpenAI Cookbook</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/wiki#saving-to-gguf--vllm-16bit-crashes">Home</a>: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/wiki#continued-pretraining--finetuning-the-lm_head-and-embed_tokens-matrices">Home</a>: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://huggingface.co/datasets/maplerxyz1/rbxidle">maplerxyz1/rbxidle · Datasets at Hugging Face</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/wiki#continued-pretraining--finetuning-the-lm_h">Home</a>: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://github.com/Syllo/nvtop">GitHub - Syllo/nvtop: GPU &amp; Accelerator process monitoring for AMD, Apple, Huawei, Intel, NVIDIA and Qualcomm</a>: GPU &amp; Accelerator process monitoring for AMD, Apple, Huawei, Intel, NVIDIA and Qualcomm - Syllo/nvtop</li><li><a href="https://discuss.huggingface.co/t/continuous-training-on-fine-tuned-model/6687/3">Continuous training on Fine-tuned Model</a>: Thank you for your reply.  I tried this as well. My old dataset and the new dataset has different texts.  This method makes the model heavily lean towards the new text provided. This results in the te...</li><li><a href="https://github.com/unslothai/unsloth/issues/210">I got unsloth running in native windows. · Issue #210 · unslothai/unsloth</a>: I got unsloth running in native windows, (no wsl). You need visual studio 2022 c++ compiler, triton, and deepspeed. I have a full tutorial on installing it, I would write it all here but I’m on mob...</li><li><a href="https://github.com/unslothai/unsloth/pull/651">Fix breaking bug in save.py with interpreting quantization_method as a string when saving to gguf by ArcadaLabs-Jason · Pull Request #651 · unslothai/unsloth</a>: Context Upon attempting to save my fine-tuned models as gguf I encountered a new error as of today. Upon investigation I discovered the issue to be some code that incorrectly broke strings passed f...</li><li><a href="https://github.com/unslothai/unsloth/wiki">Home</a>: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/blob/main/unsloth/save.py?utm_source=ainews&utm_medium=email&utm_campaign=ainews-fixing-gemma.">unsloth/unsloth/save.py at main · unslothai/unsloth</a>: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="http://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory</a>: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://colab.research.google.com/drive/1rU4kVb9A8yNRThGwftk0EU7dWsDIPOYb?usp=sharing#scrollTo=2eSvM9zX_2d3">Google Colab</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1251951286845444198)** (3 messages): 

- **CryptGPT introduces privacy-preserving LLMs**: A user shared an introductory blog post titled *"CryptGPT: Privacy-Preserving LLMs using Vigenere cipher"*. The blog post describes pretraining a GPT-2 model on an encrypted dataset, achieving comparable performance to a regular GPT-2 but requiring an encryption key to use it. [Blog Post Link](https://x.com/diwanksingh/status/1802118343446724655).

**Link mentioned**: <a href="https://x.com/diwanksingh/status/1802118343446724655">Tweet from Diwank Singh (@diwanksingh)</a>: http://x.com/i/article/1802116084507848704

  

---


### **Unsloth AI (Daniel Han) ▷ #[community-collaboration](https://discord.com/channels/1179035537009545276/1180144489214509097/)** (1 messages): 

starsupernova: Oh very interesting!
  

---



### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1251287545086607391)** (49 messages🔥): 

- **Lighting AI Interface Suggestions**: A member shared the [NVIDIA warp example code](https://github.com/NVIDIA/warp/blob/main/warp/examples/core/example_sph.py) and sought advice on a graphical interface to see the rendered results. They considered setting up a VNC session to resolve the issue.

- **Solved NVRTC Compilation Error**: A user described an issue with NVRTC where compiling multiple kernels resulted in 'invalid resource handle'. They later resolved it by avoiding initializing a new context for each compilation, which was causing CUDA to free the modules/functions.

- **GPU SM Count Discrepancy**: A query was raised about the discrepancies between measured and reported SM counts for the [A10G GPU](https://www.techpowerup.com/gpu-specs/a10g.c3798), noting that techpowerup reports 72 SMs while pycuda measures 80. It was clarified that the site might be wrong and [other sources](https://d1.awsstatic.com/product-marketing/ec2/NVIDIA_AWS_A10G_DataSheet_FINAL_02_17_2022.pdf) confirm 80 SMs.

- **New NVIDIA 5090 GPU Speculations**: Members discussed the upcoming NVIDIA 5090, with speculations about it having up to 64 GB of VRAM ([source](https://www.techpowerup.com/323495/possible-specs-of-nvidia-geforce-blackwell-gpu-lineup-leaked)). There were debates about the likelihood of these specs, with pessimistic views on seeing 64GB in consumer versions.

- **Value of Forum Knowledge in Daily AI Work**: A member expressed doubts about the practical value of most discussions in their daily AI work apart from a few specific topics. Others responded by emphasizing the importance of performance optimization and the general value of learning and being part of such communities.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://en.wikipedia.org/wiki/Chinese_zodiac">Chinese zodiac - Wikipedia</a>: no description found</li><li><a href="https://github.com/NVIDIA/warp/blob/main/warp/examples/core/example_sph.py">warp/warp/examples/core/example_sph.py at main · NVIDIA/warp</a>: A Python framework for high performance GPU simulation and graphics - NVIDIA/warp</li><li><a href="https://www.baseten.co/blog/nvidia-a10-vs-a10g-for-ml-model-inference/">NVIDIA A10 vs A10G for ML model inference</a>: The A10, an Ampere-series GPU, excels in tasks like running 7B parameter LLMs. AWS&#x27;s A10G variant, similar in GPU memory &amp; bandwidth, is mostly interchangeable.</li><li><a href="https://www.reddit.com/r/LocalLLaMA/s/1oKyTQEmlf">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.techpowerup.com/gpu-specs/geforce-rtx-3090.c3622">NVIDIA GeForce RTX 3090 Specs</a>: NVIDIA GA102, 1695 MHz, 10496 Cores, 328 TMUs, 112 ROPs, 24576 MB GDDR6X, 1219 MHz, 384 bit</li><li><a href="https://www.techpowerup.com/gpu-specs/geforce-rtx-4090.c3889">NVIDIA GeForce RTX 4090 Specs</a>: NVIDIA AD102, 2520 MHz, 16384 Cores, 512 TMUs, 176 ROPs, 24576 MB GDDR6X, 1313 MHz, 384 bit</li><li><a href="https://www.techpowerup.com/gpu-specs/a10g.c3798">NVIDIA A10G Specs</a>: NVIDIA GA102, 1710 MHz, 9216 Cores, 288 TMUs, 96 ROPs, 24576 MB GDDR6, 1563 MHz, 384 bit
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1252285272507551794)** (2 messages): 

- **AttributeError in Triton on Colab**: A user encountered an AttributeError while running Fused Softmax from Triton's official tutorial on Colab. The error message indicated `'CudaDriver' object has no attribute 'active'` and they are seeking assistance for this issue.

- **Nested Reduction Feasibility in Triton**: Another user inquired about the possibility of performing nested reductions in Triton. They are interested in running reduction code at various stages to handle quadrants individually, asking if this staged reduction is supported.
  

---


### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1251580462028161075)** (10 messages🔥): 

- **Error with `torch.compile(mode="max-autotune")`**: A user reported receiving an error message, `Not enough SMs to use max_autotune_gemm mode`, due to a hard-coded limit of 68 SMs in the PyTorch code, while their GPU only has 66 SMs. The user shared a [link to the relevant section in the PyTorch repository](https://github.com/pytorch/pytorch/blob/f0d68120f4e99ee6c05f1235d9b42a4524af39d5/torch/_inductor/utils.py#L976).

- **Discussion on Reducing SM Threshold**: A member suggested lowering the SM threshold to test if performance remains good without needing to rebuild from source. The lack of consumer GPUs in CI was mentioned as a reason for the current hard-coded value.

- **Testing Performance with Modified SM Threshold**: After changing the SM threshold to 0, the user reported no significant performance improvement.

- **Enabling Coordinate Descent Tuning**: Another member proposed enabling coordinate descent tuning found in `inductor/config.py` as a potential solution for improving performance.

**Link mentioned**: <a href="https://github.com/pytorch/pytorch/blob/f0d68120f4e99ee6c05f1235d9b42a4524af39d5/torch/_inductor/utils.py#L976">pytorch/torch/_inductor/utils.py at f0d68120f4e99ee6c05f1235d9b42a4524af39d5 · pytorch/pytorch</a>: Tensors and Dynamic neural networks in Python with strong GPU acceleration - pytorch/pytorch

  

---


### **CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1252343980423909487)** (2 messages): 

- **Vayuda paper sparks interest in search algorithms**: A member shared a [link to the Vayuda paper](https://arxiv.org/pdf/2406.07394) expressing hope that more people would work on search. This implies a potential for significant research and development in the area.
  
- **GPT-4 matches LLaMA 3 8B impressively**: A member was impressed by how matching **GPT-4** with **LLaMA 3 8B** turned out. They highlighted this achievement as noteworthy in current AI capabilities.
  

---


### **CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1251444875799433288)** (5 messages): 

- **Blockwise softmax not in PMPP book**: Blockwise softmax concepts are not covered in the PMPP book, but understanding the flash-attn algorithm and shared memory (smem) is crucial. High-end implementations leverage tensor cores, requiring further exploration into resources like CUTLASS.
- **Start with accessible YouTube lectures**: For newcomers to GPU programming and high-performance computing, starting with [YouTube lectures](https://youtube.com) is advised. These lectures aim to provide an accessible introduction to the fundamentals.
  

---


### **CUDA MODE ▷ #[jax](https://discord.com/channels/1189498204333543425/1203956655570817034/1252344004738289765)** (1 messages): 

- **Announcing tpux for simplifying Cloud TPU**: A member announced the **tpux project**, a suite of tools aimed at simplifying Cloud TPU setup and operation to facilitate the usage of JAX across multiple hosts. For more details, visit [tpux on GitHub](https://github.com/yixiaoer/tpux) and [give it a ⭐️](https://twitter.com/shin_miyaku22/status/1802373884077072537).

**Link mentioned**: <a href="https://github.com/yixiaoer/tpux">GitHub - yixiaoer/tpux: A set of Python scripts that makes your experience on TPU better</a>: A set of Python scripts that makes your experience on TPU better - yixiaoer/tpux

  

---


### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1251751170675052565)** (11 messages🔥): 

- **Quant API Import Documentation Issue**: A member flagged a correction stating, *``unwrap_tensor_subclass()`` should be imported from `torchao.utils` or `torchao.quantization.quant_api`*, not `torchao.quantization.utils`. They emphasized the importance of users calling `unwrap_tensor_subclass()` before compiling the quant model to avoid errors.

- **API Release Delay and BC Issues**: It was confirmed that the **0.3 release is being delayed** due to backward compatibility issues that need resolution. This delay ensures the team can address and fix critical problems.

- **Innovative API Naming with 'brrr' Proposal**: There was a playful yet practical suggestion to create an API with the name **`brrr` that adds additional experimental flags** based on the number of 'r's. A member humorously asked if this was serious but also hinted at a need for easier control over `torchinductor` flags like `use_mixed_mm`.

- **Feedback on `use_mixed_mm` Flag**: A member suggested enabling the `use_mixed_mm` flag by default if the relevant kernel in AO is on. This feedback may lead to a GitHub issue for further discussion and implementation.


  

---


### **CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1251340075686432778)** (10 messages🔥): 

- **Meta tackles large-scale AI training challenges**: [Meta's article](https://engineering.fb.com/2024/06/12/data-infrastructure/training-large-language-models-at-scale-meta/) discusses the complexity and computation required to train large language models (LLMs). The shift to generative AI has necessitated a rethinking of software, hardware, and network infrastructure.

- **Interview with Esolang Academics**: A [YouTube video](https://www.youtube.com/watch?v=ieqsL5NkS6I) titled "Interview with Esolang Academic 2024" was shared. The full version and BC Vim Linter will be available on Patreon for $5 the following day.

- **Pessimistic Neko's Jensen Emojis**: Member pessmistic_neko posted emojis <:jensen:1189650200147542017> to express their amusement.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=ieqsL5NkS6I">Interview with Esolang Academic 2024</a>: Esoteric programming languageFull version + BC Vim Linter for $5 tomorrow on: https://www.patreon.com/ProgrammersAreAlsoHuman Interview with an Esoteric deve...</li><li><a href="https://engineering.fb.com/2024/06/12/data-infrastructure/training-large-language-models-at-scale-meta/">How Meta trains large language models at scale</a>: As we continue to focus our AI research and development on solving increasingly complex problems, one of the most significant and challenging shifts we&#8217;ve experienced is the sheer scale of co…</li><li><a href="https://engineering.fb.com/2024/06/12/data-infrastructure/training-large-language-models-at-scale-me">How Meta trains large language models at scale</a>: As we continue to focus our AI research and development on solving increasingly complex problems, one of the most significant and challenging shifts we&#8217;ve experienced is the sheer scale of co…
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1252199364651257867)** (1 messages): 

- **Catch Akim at AI_dev Conference**: One member mentioned they will "probably be at AI_dev" and invited others to reach out. They also noted that there will be a movie about "PyTorch" shown on Tuesday.
  

---


### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1251250350711902230)** (473 messages🔥🔥🔥): 

- **DataLoader PR merged, surprising no performance difference**: The Permuted DataLoader PR was merged after some discussion and testing, although initial runs showed no performance improvement. [A surprised Aleksa retested and finally confirmed a slight improvement in validation loss](https://github.com/karpathy/llm.c/pull/573).

- **Stochastic rounding only with unique seeds**: The PR to ensure unique seeds for stochastic rounding was [discussed](https://github.com/karpathy/llm.c/pull/597). The team discovered that an overflow feature in their approach was actually intended by the noise function algorithm.

- **ZeRO-2 PR has noticeable memory overhead**: PR 593 for ZeRO-2 was discussed for its complexity and memory overhead. [Suggestions included falling back to ZeRO-1 for certain parameters](https://github.com/karpathy/llm.c/pull/593).

- **Master weight storage**: A PR (https://github.com/karpathy/llm.c/pull/522) to save master weights to resume state was merged to improve determinism. Follow-up tasks included verifying determinism through CI and exploring memory-saving techniques [such as saving 16-bit master weights](https://github.com/karpathy/llm.c/pull/432).

- **LayerNorm kernel optimization delivers speedup**: Profiling data indicated that the new LayerNorm kernel (kernel 6) was faster than the older ones (kernel 3 and 5). [This boosted certain tasks significantly](https://github.com/karpathy/llm.c/pull/600) especially under specific configurations, like recompute=2.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2010.06192">Revisiting BFloat16 Training</a>: State-of-the-art generic low-precision training algorithms use a mix of 16-bit and 32-bit precision, creating the folklore that 16-bit hardware compute units alone are not enough to maximize model acc...</li><li><a href="https://x.com/SquirrelTweets">Tweet from undefined</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=LWFzPP8ZbdU&t=3347s">Noise-Based RNG</a>: In this 2017 GDC Math for Game Programmers talk, SMU Guildhall&#39;s  Squirrel Eiserloh discuss RNGs vs. noise functions, and shows how the latter can replace th...</li><li><a href="https://x.com/SquirrelTweets/status/1421251894274625536">Tweet from Squirrel Eiserloh (@SquirrelTweets)</a>: Updated my raw noise function.  Eliminates a flaw discovered by @ptrschmdtnlsn in which certain high input bits lacked influence over certain low output bits.  Anyone using Squirrel3 from my GDC 2017 ...</li><li><a href="https://www.youtube.com/watch?v=LWFzPP8ZbdU&t=2199s">Noise-Based RNG</a>: In this 2017 GDC Math for Game Programmers talk, SMU Guildhall&#39;s  Squirrel Eiserloh discuss RNGs vs. noise functions, and shows how the latter can replace th...</li><li><a href="https://arxiv.org/abs/2405.18392">Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations</a>: Scale has become a main ingredient in obtaining strong machine learning models. As a result, understanding a model&#39;s scaling properties is key to effectively designing both the right training setu...</li><li><a href="https://github.com/karpathy/llm.c/pull/600">Use faster kernel for LayerNorm forward by gordicaleksa · Pull Request #600 · karpathy/llm.c</a>: I ran kernel 5 under /dev/cuda/ (./layernorm_forward 5) on both RTX 3090 and H100 systems and it&#39;s faster on both of them. Numbers: kernel 3, optimal block size on:  RTX 3090 → 32 (689.11 GB/s) H1...</li><li><a href="https://github.com/karpathy/llm.c/blob/master/llmc/global_norm.cuh#L63">llm.c/llmc/global_norm.cuh at master · karpathy/llm.c</a>: LLM training in simple, raw C/CUDA. Contribute to karpathy/llm.c development by creating an account on GitHub.</li><li><a href="https://github.com/karpathy/llm.c/pull/594">add scripts to export to HF and run Eleuther evals by karpathy · Pull Request #594 · karpathy/llm.c</a>: no description found</li><li><a href="https://github.com/karpathy/llm.c/pull/338">GELU Fusion with cuBLASLt (SLOWER because it only merges in FP16 mode, not BF16/FP32...) by ademeure · Pull Request #338 · karpathy/llm.c</a>: It turns out that not only is cuBLASLt not able to fuse BF16 GELU (or RELU) into a BF16 matmul, it also ends up with a strange kernel that is slower than our own GELU kernel as it does 2 writes per...</li><li><a href="https://github.com/karpathy/llm.c/pull/602">Hotfix - proper handling of max num of block sums by gordicaleksa · Pull Request #602 · karpathy/llm.c</a>: The previous assert logic was too restrictive as it depended on the number of layers of the model and the specification of the GPU (num of SMs &amp; max number of threads per SM). This PR fixes that. ...</li><li><a href="https://github.com/karpathy/build-nanogpt/pull/30">fix sync issue that results in incorrect gradient accumulation and incorrect loss by WilsonCWu · Pull Request #30 · karpathy/build-nanogpt</a>: Repro Used torch version &#39;2.3.1+cu121&#39;. Set B = 16 (from 64). The following losses were observed: 250 val 6.4300 250 hella 0.2440 250 train 6.387966 ... 1000 val 4.8797 1000 hella 0.2419 1000 ...</li><li><a href="https://github.com/karpathy/llm.c/pull/601">Fix stochastic rounding in encoder backward kernel by gordicaleksa · Pull Request #601 · karpathy/llm.c</a>: #597 provided unique seeds to adamw update. This PR does the same thing for the encoder backward which is the only other place where we do stochastic rounding.</li><li><a href="https://github.com/karpathy/llm.c/pull/573?">Dataloader - introducing randomness by gordicaleksa · Pull Request #573 · karpathy/llm.c</a>: On the way to fully random train data shuffling... This PR does the following:  Each process has a different unique random seed Each process train data loader independently chooses its starting sha...</li><li><a href="https://github.com/karpathy/llm.c/pull/595">Changes toward `layernorm_forward` in `dev/cuda` by KarhouTam · Pull Request #595 · karpathy/llm.c</a>: Remove cooperative groups Following the instructions in #292, remove cooperative groups codes in existing layernorm forward kernels. benchmark Performance before and after changes:     Block Size l...</li><li><a href="https://github.com/karpathy/llm.c/pull/591">Fused Forward GELU (again) by ademeure · Pull Request #591 · karpathy/llm.c</a>: This turns out to be properly fused (and therefore faster) on H100 with CUDA 12.5 - it was definitely not fused and actually noticeably slower on RTX 4090 with CUDA 12.4, I suspect that is more abo...</li><li><a href="https://github.com/karpathy/llm.c/pull/513">Added packed layernorm_forward by ChrisDryden · Pull Request #513 · karpathy/llm.c</a>: This is the implementation of using packed data types for layernorm and has an associated speedup of around 50% for this kernel in the dev files, waiting for the PR for making the data types in tha...</li><li><a href="https://github.com/karpathy/llm.c/pull/506">Added additional layernorm forward kernel that does not recalculate mean and rstd by ChrisDryden · Pull Request #506 · karpathy/llm.c</a>: This is the first optimization and there are many more that can be done now, but now the kernel is split into two so that each of the Layernorm forwards can be modified independently now for future...</li><li><a href="https://github.com/karpathy/llm.c/pull/561">Fix the compiler warnings and errors by lancerts · Pull Request #561 · karpathy/llm.c</a>: Fix such error that happens for CUDA 11.8 per discussion #558 (comment) matmul_backward_bias.cu(151): error: no operator &quot;+=&quot; matches these operands             operand types are: floatX += ...</li><li><a href="https://github.com/karpathy/llm.c/actions/runs/9537233489/job/26285087145?pr=600">Use faster kernel for LayerNorm forward · karpathy/llm.c@7d7084a</a>: LLM training in simple, raw C/CUDA. Contribute to karpathy/llm.c development by creating an account on GitHub.</li><li><a href="https://github.com/karpathy/llm.c/pull/508/">adding wsd schedule with (1-sqrt) decay by eliebak · Pull Request #508 · karpathy/llm.c</a>: Adding new learning rate schedule support: WSD learning rate schedule:  Warmup: classical linear warmup Stable: constant lr Decay: Decaying to min_lr in a (1-sqrt) shape. (more info here https://ar...</li><li><a href="https://github.com/karpathy/llm.c/pull/597">Fix stochastic rounding by gordicaleksa · Pull Request #597 · karpathy/llm.c</a>: Previously our stochastic rounding logic didn&#39;t have a unique seed for each of the parameters we&#39;re rounding. This PR fixes that. In more detail, previously:  we were passing the same seed for...</li><li><a href="https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-features/rounding-modes.html">Neuron Rounding Modes &#8212; AWS Neuron Documentation</a>: no description found</li><li><a href="https://github.com/karpathy/llm.c/pull/583">Replaced hard-coded max float with FLT_MAX by vyom1611 · Pull Request #583 · karpathy/llm.c</a>: Just fixed some TODOs and replaced hard-coded values with FLT_MAX for floating point integers which comes with already included &lt;float.h&gt; headers in both files.</li><li><a href="https://github.com/karpathy/llm.c/pull/573/commits/c81f1efbb82b4056cb9402d2ae7786e9d0165f1f">Dataloader - introducing randomness by gordicaleksa · Pull Request #573 · karpathy/llm.c</a>: On the way to fully random train data shuffling... This PR does the following:  Each process has a different unique random seed Each process train data loader independently chooses its starting sha...</li><li><a href="https://github.com/karpathy/llm.c/pull/603">Check determinism in CI by ngc92 · Pull Request #603 · karpathy/llm.c</a>: Extends the test script to validate determinism. Some thoughts:  With C memory management, it is quite easy to introduce memory leaks, e.g., loading from checkpoint twice without freeing in the mid...</li><li><a href="https://huggingface.co/mdouglas/llmc-gpt2-774M-150B">mdouglas/llmc-gpt2-774M-150B · Hugging Face</a>: no description found</li><li><a href="https://github.com/karpathy/llm.c/discussions/580">GPT-2 (774M) reproduced · karpathy/llm.c · Discussion #580</a>: I left the GPT-2 774M model running for ~6 days on my 8X A100 80GB node (150B tokens, 1.5 epochs over the 100B FineWeb sample dataset) and training just finished a few hours ago and went well with ...</li><li><a href="https://github.com/karpathy/llm.c/pull/522/files#diff-fd05f347e9f31ebf4645e7ee912dfc80e74faa9b94cf8a2c35df555fc7927b83R38">Add master weights to resume state by gordicaleksa · Pull Request #522 · karpathy/llm.c</a>: We&amp;#39;re currently not saving master weights as part of the state -&amp;gt; we lose some precision because otherwise when we resume we&amp;#39;ll have to reconstruct the master weights by upcasti...</li><li><a href="https://github.com/karpathy/llm.c/pull/432">only save missing bits to reconstruct fp32 master weights by ngc92 · Pull Request #432 · karpathy/llm.c</a>: I think I managed to get the bit-fiddling right, and this will effectively give us fp31 master parameters at the cost of only 16 additional bits (instead of the current 32). Before merging, the cod...</li><li><a href="https://github.com/karpathy/llm.c/pull/600/files">Use faster kernel for LayerNorm forward by gordicaleksa · Pull Request #600 · karpathy/llm.c</a>: I ran kernel 5 under /dev/cuda/ (./layernorm_forward 5) on both RTX 3090 and H100 systems and it&amp;#39;s faster on both of them. Numbers: kernel 3, optimal block size on:  RTX 3090 → 32 (689.11 GB/s...</li><li><a href="https://github.com/karpathy/llm.c/pull/556">Utilities for cuda streams + disk IO by ngc92 · Pull Request #556 · karpathy/llm.c</a>: handling disk io for checkpointing with cuda streams is a nontrivial task. If you&#39;re  not careful, you can easily get broken code (need to wait for data to be on the CPU before you can start writi...</li><li><a href="https://github.com/karpathy/llm.c/pull/593/files">Zero 2 - WIP by ngc92 · Pull Request #593 · karpathy/llm.c</a>: Trying to get a first version working. Code isn&amp;#39;t nice, we currently lose the asynchrony in the communication code because we need to reuse the buffer for the next layer, and it doesn&amp;#39;...</li><li><a href="https://github.com/karpathy/llm.c/pull/599/files">Permuted DataLoader by karpathy · Pull Request #599 · karpathy/llm.c</a>: Permuted dataloader, and some first tests. WIP still.</li><li><a href="https://github.com/karpathy/llm.c/pull/522">Add master weights to resume state by gordicaleksa · Pull Request #522 · karpathy/llm.c</a>: We&#39;re currently not saving master weights as part of the state -&gt; we lose some precision because otherwise when we resume we&#39;ll have to reconstruct the master weights by upcasting from lowe...
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[oneapi](https://discord.com/channels/1189498204333543425/1233802893786746880/1251457210907889696)** (2 messages): 

- **Dynamic batching support struggles with Gaudi**: A member mentioned the difficulties in getting dynamic batching with vLLM ported to Gaudi. They questioned if there is an architecture limitation preventing the implementation of KV cache flash attention kernels, contrasting it with regular "rectangular" shapes that are processed without issue.

- **Channel rename suggestion to Intel**: Another suggestion was to rename the channel to **Intel**, tagging a user for their input. This reflects a possible channel rebranding direction.
  

---


### **CUDA MODE ▷ #[bitnet](https://discord.com/channels/1189498204333543425/1240586843292958790/1251566494987321424)** (49 messages🔥): 

- **Meeting Troubleshooting and New Link Shares**: Users were discussing voice chat issues and shared several resources like [Python development environments with Nix](https://wiki.nixos.org/wiki/Python). "New laptop and having some problems with ubuntu," one mentioned while testing their setup.
- **Benchmarking and Quantization Debates**: Much of the conversation centered around benchmarking matrix multiplication with different precisions and quantization techniques. One user inquired, "Are you benchmarking matmul(x_fp16, W_nbit) or do you include scaling / zeros with grouping?" while others responded with their specific benchmarking approaches and the importance of grouping for better quality.
- **Resource Links for Further Reading**: Several useful links were shared including a [quantization](https://github.com/mobiusml/hqq/blob/master/hqq/core/quantize.py#L86-L90) technique and a [library supporting mixed-precision matrix multiplications](https://github.com/microsoft/BitBLAS/blob/main/docs/PythonAPI.md#matmul). These resources aimed to facilitate a clearer understanding of optimization strategies.
- **VRAM Constraints and GPU Considerations**: Discussions also included the limitations of running larger models like llama2 locally due to VRAM constraints. One user mentioned using an XPS15 laptop with a GeForce GTX 1650 and explored alternative platforms like Lightning AI’s L4 with 22 free hours for testing.
- **New Git Pull Requests and Test Case Pushes**: Updates on the development side were shared, including pushing new test cases for BitnetTensor and UInt2Tensor. Users interacted around issues and updates, as seen in the comment, "pushed test cases for BitnetTensor, UInt2Tensor and bitpacking gen," providing collaborative development progress.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/microsoft/BitBLAS/blob/main/docs/PythonAPI.md#matmul">BitBLAS/docs/PythonAPI.md at main · microsoft/BitBLAS</a>: BitBLAS is a library to support mixed-precision matrix multiplications, especially for quantized LLM deployment. - microsoft/BitBLAS</li><li><a href="https://github.com/pytorch/ao/issues/386">Tensor Core Layout docs is not clear · Issue #386 · pytorch/ao</a>: Right now what we have is docstrings but they could use work - this came up as @vayuda was looking at extending his bitpacking work to include a notion of scales What does tensor core layout mean? ...</li><li><a href="https://github.com/mobiusml/hqq/blob/master/hqq/core/quantize.py#L86-L90">hqq/hqq/core/quantize.py at master · mobiusml/hqq</a>: Official implementation of Half-Quadratic Quantization (HQQ) - mobiusml/hqq</li><li><a href="https://wiki.nixos.org/wiki/Python">Python - NixOS Wiki</a>: no description found</li><li><a href="https://github.com/pytorch/ao/pull/282">[WIP] Added first bits of Uint2Tensor and BitnetTensor by andreaskoepf · Pull Request #282 · pytorch/ao</a>: Created a UInt2Tensor class (similar to the UInt4Tensor class). Added a BitnetTensor class and a first unit test which quantizes the weights of a nn.Linear() layer and executes the matmul. Currentl...
</li>
</ul>

</div>
  

---



### **LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1251249928613658635)** (204 messages🔥🔥): 

- **Link Your Code Projects with 'lms' Tool**: With the release of [LM Studio 0.2.22](https://lmstudio.ai), users can now utilize 'lms' for managing models and debugging prompts. The tool helps with loading/unloading models, and inspecting raw LLM input, streamlining local AI deployments ([GitHub repository](https://github.com/lmstudio-ai/lms)).

- **Intel ARC A770 GPU Now Supported**: There were several inquiries about Intel ARC A770 GPU support. [Instructions](https://github.com/lmstudio-ai/configs/blob/main/Extension-Pack-Instructions.md) were provided to enable OpenCL for Intel GPUs, emphasizing manual adjustments for GPU layers.

- **Performance Comparison and GPU Utilization**: Members discussed performance comparisons, revealing mixed results with CPU vs. GPU, and specific configuration needs for optimal model performance. Issues with the Deepseek Coder V2 Lite GGUF models were addressed, highlighting the necessity to toggle Flash Attention settings.

- **Local Model Hosting Issues with Open Interpreter**: Users encountered issues hosting local models for Open Interpreter via LM Studio. Recommendations included checking the detailed guide on [Open Interpreter's documentation](https://docs.openinterpreter.com/language-models/local-models/lm-studio).

- **Font Size Adjustments in LM Studio**: A repeated request was to improve font size controls in LM Studio. Although there are keyboard shortcuts for zooming in/out, a more permanent and versatile solution within the app was suggested.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lmstudio.ai/careers">👾 LM Studio - Discover and run local LLMs</a>: Find, download, and experiment with local LLMs</li><li><a href="https://lmstudio.ai/,">👾 LM Studio - Discover and run local LLMs</a>: Find, download, and experiment with local LLMs</li><li><a href="https://lmstudio.ai/blog/lms#debug-your-prompting-with-lms-log-stream">Introducing `lms` - LM Studio&#x27;s companion cli tool | LM Studio</a>: Today, alongside LM Studio 0.2.22, we&#x27;re releasing the first version of lms — LM Studio&#x27;s companion cli tool.</li><li><a href="https://docs.openinterpreter.com/language-models/local-models/lm-studio">no title found</a>: no description found</li><li><a href="https://huggingface.co/Qwen/Qwen2-7B-Instruct-GGUF">Qwen/Qwen2-7B-Instruct-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/MaziyarPanahi/Meta-Llama-3-70B-Instruct-GGUF">MaziyarPanahi/Meta-Llama-3-70B-Instruct-GGUF · Hugging Face</a>: no description found</li><li><a href="https://github.com/lmstudio-ai/configs/blob/main/Extension-Pack-Instructions.md">configs/Extension-Pack-Instructions.md at main · lmstudio-ai/configs</a>: LM Studio JSON configuration file format and a collection of example config files. - lmstudio-ai/configs</li><li><a href="https://www.reddit.com/r/techsupport/comments/wmq143/windows_10_explorer_vram_leak/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://huggingface.co/datasets">Hugging Face – The AI community building the future.</a>: no description found</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/7979">Bug: Deepseek Coder MOE GGML_ASSERT: ggml.c:5705: ggml_nelements(a) == ne0*ne1 · Issue #7979 · ggerganov/llama.cpp</a>: What happened? When trying to run one of the new Deepseek Coder conversions or quantizations I see this error: GGML_ASSERT: ggml.c:5705: ggml_nelements(a) == ne0*ne1 Happens when on pure CPU My F32...
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1251250666870145075)** (137 messages🔥🔥): 

<ul>
  <li><strong>Qwen2 and the search mishap</strong>: A user initially struggled to get coherent outputs from Qwen2 instruct, solved it using the "blank" preset, and another member advised on searching within the Discord for help rather than external sites.</li>
  <li><strong>Roleplaying model recommendation</strong>: When asked for the best model for roleplaying, a member suggested <a href="https://huggingface.co/DavidAU/Fimbulvetr-11B-Ultra-Quality-plus-imatrix-GGUF">Fimbulvetr-11B</a>, describing it as effective for their needs.</li>
  <li><strong>Finding coding models amid confusion</strong>: There was a discussion about the best models for coding, emphasizing the rapidly changing landscape and the difficulty of making reliable recommendations. Users mentioned preferring <em>Codestral</em> and exploring <a href="https://llm.extractum.io/list/">Large and Small Language Models list</a> for detailed searches.</li>
  <li><strong>New "Ultra-Quality" model releases</strong>: Members highlighted the release of new high-performance models like <a href="https://huggingface.co/DavidAU/Psyonic-Cetacean-Ultra-Quality-20b-GGUF-imat-plus2">Psyonic-Cetacean-Ultra-Quality-20b-GGUF-imat-plus2</a> and discussed their testing results and quantitative improvements.</li>
  <li><strong>Discussion on DeepSeek-Coder-V2</strong>: A member noted the release of <a href="https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct">DeepSeek-Coder-V2</a>, capturing the excitement around its coding capabilities and discussing VRAM requirements and flash attention settings for optimal performance.</li>
</ul>

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/DavidAU/Psyonic-Cetacean-Ultra-Quality-20b-GGUF-imat-plus2">DavidAU/Psyonic-Cetacean-Ultra-Quality-20b-GGUF-imat-plus2 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/DavidAU/PsyCET-Decision-Time-Imatrix">DavidAU/PsyCET-Decision-Time-Imatrix · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct">deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/mradermacher/Oumuamua-7b-instruct-v2-i1-GGUF">mradermacher/Oumuamua-7b-instruct-v2-i1-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/nitky/Oumuamua-7b-instruct-v2">nitky/Oumuamua-7b-instruct-v2 · Hugging Face</a>: no description found</li><li><a href="https://llm.extractum.io/list/">All Large Language Models</a>: A Curated List of the Large and Small Language Models (Open-Source LLMs and SLMs). All Large Language Models with Dynamic Sorting and Filtering.</li><li><a href="https://x.com/umiyuki_ai/status/1801836418744127702">Tweet from うみゆき@AI研究 (@umiyuki_ai)</a>: Oumuamua-7b-instruct-v2、Shaberi3ベンチマークで平均点7.25です。GPT3.5T（7.16）やQwen2-7B（7.23）を超えてます。メッチャ強い</li><li><a href="https://huggingface.co/RichardErkhov/ArthurZ_-_mamba-2.8b-gguf">RichardErkhov/ArthurZ_-_mamba-2.8b-gguf · Hugging Face</a>: no description found</li><li><a href="https://rentry.org/quant_test">How do quantization formats affect model output?</a>: How do quantization formats affect model output? Introduction Test method The box question Prompt Results Thoughts Shopping and haircut Prompt Results Thoughts Health education Prompt Results Thoughts...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1251271697072656424)** (13 messages🔥): 

- **How to handle AVX2 instruction issue**: A member faced issues after updating LM Studio and found that reinstalling the beta version from [here](https://lmstudio.ai/beta-releases.html) resolved the problem. They warn, "do not update" afterwards to avoid recurring issues.

- **Qwen2 outputting eot_id token problem**: Users reported LM Studio outputting the eot_id token for Qwen2 instead of stopping generation, similar to issues with Llama3. Suggestions included checking the preset used and whether flash was enabled.

- **Suggestion for GPU off-loading**: A user proposed an enhancement to allow off-loading models to GPU before they fully load into RAM. This would benefit machines with more VRAM than RAM, particularly GPU servers, ensuring faster and more efficient model loading.

- **Stop token handling in LM Studio**: Concerns were raised about LM Studio allowing stop tokens to appear in the output and not stopping generation, leading to extensive token generation. One user emphasized the need for LM Studio to honor all listed stop tokens and treat this as a release-blocking bug.

- **User interface feedback**: The LM Studio interface received positive feedback for being "cool, soft, intuitive, and fast." Another user suggested adding VRAM usage statistics for better performance monitoring.

**Link mentioned**: <a href="https://lmstudio.ai/beta-releases.html">LM Studio Beta Releases</a>: no description found

  

---


### **LM Studio ▷ #[📝-prompts-discussion-chat](https://discord.com/channels/1110598183144399058/1120489168687087708/1251899080448278538)** (8 messages🔥): 

- **Wrestling with Error Detection**: A member expressed frustration over their model's inability to detect its errors and suggested it should output *"#ERROR"* when it cannot self-correct. Despite clear instructions, the model keeps requesting guidance rather than failing gracefully.
- **Struggling with Text Appendages**: Another member sought advice on preventing a model from adding irrelevant text at the end of responses. They specified using the **bartowski/aya-23-8B-GGUF/aya-23-8B-Q8_0.gguf** model and received a suggestion to try the *Cohere Command R* preset.
  

---


### **LM Studio ▷ #[⚙-configs-discussion](https://discord.com/channels/1110598183144399058/1136793122941190258/1251469810332864675)** (3 messages): 

- **Mikupad User Faces Config Issues**: A user sought help for using Mikupad as a webUI to interact with LMS, reporting an error message for an unexpected endpoint or method. They noted, *"Mikupad have same config as LMS."*
- **Codestral RAG Preset Advice Needed**: A member downloaded Codestral RAG and requested advice on creating a preset oriented towards RAG (retrieval-augmented generation). They mentioned reading relevant information on Hugging Face but remained unsure about the preset creation process.
  

---


### **LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1251271511277441105)** (34 messages🔥): 

- **Archiving LM Studio 0.2.23 Setup**: A member shared a [MirrorCreator link](https://mir.cr/E1WVBIOO) to the archived LM Studio 0.2.23 setup file, noting that the installers are digitally signed and can be verified for integrity.
- **Adding a Second RTX 3090**: A member asked if adding a different brand RTX 3090 would cause issues and whether to retain an RTX 2070 in the same system. Advice given suggested that for best results, get the exact same card and an SLI bridge; keeping the 2070 would slow down performance.
- **Setting CPU Cores in Server Mode**: A query was raised regarding the ability to set the number of CPU cores for processing in Server Mode, noting that only four cores were being utilized despite the model being loaded in RAM.
- **AMD Ryzen RX 7700S GPU Detection Issues**: A member faced issues with LM Studio not detecting an AMD Ryzen RX 7700S GPU on a Windows laptop. The discussion sought troubleshooting steps and clarified specifics about the GPU and OS.
- **Mixing RAM Sticks Concerns**: The conversation involved the viability of mixing different RAM sticks with the same speed but potentially different timings for CPU-only inference tasks. The conclusion was that it should work but to confirm compatibility using memtest.

**Link mentioned**: <a href="https://mir.cr/E1WVBIOO">LM-Studio-0.2.23-Setup.exe - Mirrored.to - Mirrorcreator - Upload files to multiple hosts</a>: no description found

  

---


### **LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1251252850638716992)** (22 messages🔥): 

- **Smaug-3 Tokenizer Issue Resolved**: The latest build resolves the previously noted **smaug-3 tokenizer issue**. This update was quickly acknowledged and appreciated by other members. 
- **Decoupling ROCm from Main App**: A user commended the move to decouple **ROCm** from the main app, highlighting the successful upgrade and smooth operation on a 7900xtx. They shared their positive experience: *"working just fine for me after upgrading"*. 
- **Command R+ GPU Offloading Glitch**: Users debated an issue where **Command R+ outputs gibberish** when fully offloaded to the GPU, while the same model functions correctly on the CPU. One user mentioned, *"Something screwy there. My context is only 4k"*, indicating it might not be a memory issue.
- **Older Version Availability**: Members discussed the difficulty of accessing older versions of the app, noting that changing version numbers in the URL to access older versions no longer works. Suggestions included personally keeping copies of older versions before updating, although this was flagged as impractical post-update.
  

---


### **LM Studio ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/1251650482070491196)** (1 messages): 

- **Environment recreation resolves API key issue**: A user described an issue receiving an "incorrect API key" error that persisted until they recreated their environment and reinstalled dependencies. Setting the API key using `$env:OPENAI_API_KEY` resolved their problem.
- **Assistant sends blank messages, causing errors**: Although the user successfully set the default message and configured a model for user proxy and chat managers, the assistant sends blank messages, which results in errors in LM Studio. They are seeking further solutions to this issue.
  

---


### **LM Studio ▷ #[open-interpreter](https://discord.com/channels/1110598183144399058/1197707651438624849/1252341113931436143)** (13 messages🔥): 

- **Interpreter defaults to GPT-4 despite LM Studio running**: A user faced an issue where attempting to run **interpreter --local** with a running **LM Studio** server resulted in a prompt for a provider, and then defaulted to GPT-4 even after setting LM Studio as the provider.
- **YouTube tutorial link shared**: Another user suggested following this [YouTube tutorial](https://youtu.be/xPd8FFzIeOw?t=602) to potentially resolve the issue with the Open Interpreter setup.
- **Need to see full server page screenshot**: It was advised to have the server running with a model selected and to share a screenshot of the entire LMStudio server page to diagnose the problem.
- **MacOS vs Linux inquiry**: The troubleshooting user mentioned the steps they took on **MacOS**, prompting an inquiry about whether the original issue occurred on **Linux**.
- **Simple setup steps shared**: A user provided clear steps to set up the interpreter on their machine, which seemed to work fine on **MacOS**.

**Link mentioned**: <a href="https://youtu.be/xPd8FFzIeOw?t=602">ChatGPT &quot;Code Interpreter&quot; But 100% Open-Source (Open Interpreter Tutorial)</a>: This is my second video about Open Interpreter, with many new features and much more stability, the new Open Interpreter is amazing. Update: Mixtral 7x8b was...

  

---


### **LM Studio ▷ #[model-announcements](https://discord.com/channels/1110598183144399058/1225909444727013466/1252325560248434861)** (1 messages): 

- **DeepSeek releases ultra-fast coding models**: DeepSeek's new **Coding models** are now available, featuring their **V2 MoE** with **16B total parameters** and only **2.4B activated** for each request. This model requires **flash attention disabled** for proper functioning; download it [here](https://huggingface.co/lmstudio-community/DeepSeek-Coder-V2-Lite-Instruct-GGUF). 

- **DeepSeek's community contributions highlighted**: The **DeepSeek-Coder-V2-Lite-Instruct** is part of the [LM Studio Community](https://lmstudio.ai) models highlights program, which emphasizes new and notable models. The **GGUF quantization** was provided by bartowski based on the latest `llama.cpp` release.

**Link mentioned**: <a href="https://huggingface.co/lmstudio-community/DeepSeek-Coder-V2-Lite-Instruct-GGUF">lmstudio-community/DeepSeek-Coder-V2-Lite-Instruct-GGUF · Hugging Face</a>: no description found

  

---


### **LM Studio ▷ #[🛠-dev-chat](https://discord.com/channels/1110598183144399058/1234988891153629205/1251651971656581254)** (27 messages🔥): 

- **VSCode code scripts and model suggestions integration**: A member shared their "dream workflow" for integrating VSCode with various models, using tools like CodeParrot and OpenAI's Playground to generate script files and `continue.dev` for code modification and explanation. They expressed challenges in iterating code versions and requested help setting up `continue.dev`.

- **Recommendations for model selection and config in `continue.dev`**: Another member recommended using models like `llama3` or `deepseek-coder` for chat and provided a [configuration file example](https://docs.continue.dev/setup/examples) for `continue.dev`. They pointed to issues related to unsupported GPU (6600XT) needing OpenCL instead of ROCM.

- **GPU setup issues**: A member faced problems setting up GPU acceleration with ROCM and then OpenCL, leading to repeated errors about GPU survey failures. It was suggested they might be missing drivers and to seek detailed help in a specific channel.

- **Configuring `continue.dev` with LM Studio**: Discussions highlighted the complexities of setting up multiple servers with LM Studio for different models, and using the `apiBase` property in `continue.dev`'s config. A link to [setup instructions](https://docs.continue.dev/reference/Model%20Providers/lmstudio) specifically for LM Studio was shared.

- **Call for API usage of LM Studio**: A member asked about using LM Studio via API through ngrok, but it was clarified that LM Studio must be installed and run locally to use its services.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.continue.dev/setup/select-model">Select models | Continue</a>: Configure LLMs</li><li><a href="https://docs.continue.dev/walkthroughs/tab-autocomplete#setting-up-with-lm-studio">Tab Autocomplete (beta) | Continue</a>: Continue now provides support for tab autocomplete in VS Code and JetBrains IDEs. We will be greatly improving the experience over the next few releases, and it is always helpful to hear feedback. If ...</li><li><a href="https://docs.continue.dev/setup/examples#i-need-to-be-entirely-local--offline">Example configurations | Continue</a>: If you&#x27;re looking for a quick way to create the perfect Continue setup, we&#x27;ve written a few sample config.jsons for common situations. You can copy these and paste them into your config.json...</li><li><a href="https://docs.continue.dev/reference/Model%20Providers/lmstudio">LM Studio | Continue</a>: LM Studio is an application for Mac, Windows, and Linux that makes it easy to locally run open-source models and comes with a great UI. To get started with LM Studio, download from the website, use th...
</li>
</ul>

</div>
  

---



### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1251254405785976852)** (372 messages🔥🔥): 

- **Homemade AIs for Low-Resource Devices**: Users discussed self-hosted AI alternatives to GPT-4 that don't require powerful servers. **"Maybe llama3 (70B-7B), mixtral 8x7B, or command r+"** were suggested.
- **FlowGPT's NSFW Content**: **FlowGPT** is under scrutiny for potentially allowing NSFW content, which **OpenAI prohibits**. One user argued that while NSFW bots are common, it's important to clarify **moral vs. legal concerns**.
- **Efficient Fine-Tuning and Evaluation**: **Viliamvolosv** shared his **QLoRa settings** for improving Russian language models on classic literature, seeking advice on optimal parameters. **Fulx69** highlighted the importance of experimenting with r and alpha values and suggested tools for evaluation like **LLaMA-Factory**.
- **New AI Models and Tools**: **DeepSeek-Coder-V2** is claimed to surpass GPT-4-Turbo in coding and math, with users recommending using **LiveCodeBench** for unbiased evaluation. Its on [@deepseek_ai](https://x.com/deepseek_ai/status/1802680388256768145).
- **Joined and Welcomed**: New users like **9do4n1** and **open.group** joined, with others welcoming them and clarifying server rules and culture. **"Welcome 🤗"** messages emphasized the supportive community environment.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/eswardivi/Phi-3-mini-128k-instruct">Phi-3-mini-128k-instruct - a Hugging Face Space by eswardivi</a>: no description found</li><li><a href="https://huggingface.co/spaces/KingNish/OpenGPT-4o">OpenGPT 4o - a Hugging Face Space by KingNish</a>: no description found</li><li><a href="https://huggingface.co/spaces/OpenGVLab/InternVL">InternVL - a Hugging Face Space by OpenGVLab</a>: no description found</li><li><a href="https://huggingface.co/spaces/lllyasviel/Omost">Omost - a Hugging Face Space by lllyasviel</a>: no description found</li><li><a href="https://huggingface.co/lllyasviel">lllyasviel (Lvmin Zhang)</a>: no description found</li><li><a href="https://youtu.be/R9c-_neaxeU">MENACE: the pile of matchboxes which can learn</a>: See more data and check out what we changed on the second day (which caused MENACE to learn a different strategy) in the second video: https://youtu.be/KcmjO...</li><li><a href="https://github.com/huggingface/trl/blob/main/examples/notebooks/gpt2-sentiment.ipynb">trl/examples/notebooks/gpt2-sentiment.ipynb at main · huggingface/trl</a>: Train transformer language models with reinforcement learning. - huggingface/trl</li><li><a href="https://huggingface.co/blog/nroggendorff/ttt-ai">Train a Shitty Tic-Tac-Toe AI</a>: no description found</li><li><a href="https://shog.ai/">ShogAI | Explore Open Source & Decentralized AI</a>: no description found</li><li><a href="https://www.robustintelligence.com/platform/ai-firewall-guardrails">Protect your AI applications in real time — Robust Intelligence</a>: Protect generative AI applications against attacks and undesired responses. Robust Intelligence guardrails protect against security and safety threats.</li><li><a href="https://pypi.org/project/numpy/">numpy</a>: Fundamental package for array computing in Python</li><li><a href="https://x.com/deepseek_ai/status/1802680388256768145">Tweet from DeepSeek (@deepseek_ai)</a>: DeepSeek-Coder-V2: First Open Source Model Beats GPT4-Turbo in Coding and Math  &gt; Excels in coding and math, beating GPT4-Turbo, Claude3-Opus, Gemini-1.5Pro, Codestral. &gt; Supports 338 programmin...</li><li><a href="https://tenor.com/view/dead-pixels-dpgc-gif-25869325">Dead Pixels Dpgc GIF - Dead Pixels Dpgc - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/guy-arguing-guy-talking-to-wall-talking-brick-wall-gif-18667615">Guy Arguing GIF - Guy Arguing Guy Talking To Wall - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/michael-jackson-eating-popcorn-enjoy-i-like-nom-nom-gif-11040065238845078056">Michael Jackson Eating Popcorn GIF - Michael Jackson Eating Popcorn Enjoy - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/microsoft/Phi-3-vision-128k-instruct">microsoft/Phi-3-vision-128k-instruct · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/THUDM/cogvlm2-llama3-chat-19B-int4">THUDM/cogvlm2-llama3-chat-19B-int4 · Hugging Face</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/17pw7bv/eternal_question_what_rank_r_and_alpha_to_use_in/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/Zz-ww/SadTalker-Video-Lip-Sync">GitHub - Zz-ww/SadTalker-Video-Lip-Sync: 本项目基于SadTalkers实现视频唇形合成的Wav2lip。通过以视频文件方式进行语音驱动生成唇形，设置面部区域可配置的增强方式进行合成唇形（人脸）区域画面增强，提高生成唇形的清晰度。使用DAIN 插帧的DL算法对生成视频进行补帧，补充帧间合成唇形的动作过渡，使合成的唇形更为流畅、真实以及自然。</a>: 本项目基于SadTalkers实现视频唇形合成的Wav2lip。通过以视频文件方式进行语音驱动生成唇形，设置面部区域可配置的增强方式进行合成唇形（人脸）区域画面增强，提高生成唇形的清晰度。使用DAIN 插帧的DL算法对生成视频进行补帧，补充帧间合成唇形的动作过渡，使合成的唇形更为流畅、真实以及自然。 - Zz-ww/SadTalker-Video-Lip-Sync</li><li><a href="https://www.youtube.com/watch?v=G-di38Fpgdw">SAMPLE LESSON: Matchboxes Play Tic-Tac-Toe</a>: This is an example of a lesson from my AWS Machine Learning course.See the full course here: https://learn.mikegchambers.com/p/aws-machine-learning-specialty...</li><li><a href="https://github.com/hiyouga/LLaMA-Factory/">GitHub - hiyouga/LLaMA-Factory: Unify Efficient Fine-Tuning of 100+ LLMs</a>: Unify Efficient Fine-Tuning of 100+ LLMs. Contribute to hiyouga/LLaMA-Factory development by creating an account on GitHub.</li><li><a href="https://livecodebench.github.io/">
    LiveCodeBench: Holistic and Contamination Free Evaluation of Large
    Language Models for Code
  </a>: no description found</li><li><a href="https://paperswithcode.com/paper/textgrad-automatic-differentiation-via-text">Papers with Code - TextGrad: Automatic &quot;Differentiation&quot; via Text</a>: 🏆 SOTA for  on GPQA (Accuracy metric)</li><li><a href="https://app.rebrandly.com/public/links/share?href=rb.gy/klkbs7">Rebrandly Dashboard</a>: no description found</li><li><a href="https://gist.github.com/viliamvolosv/fdd641d77721a48bf38225d088683d07">Settings for qlora</a>: Settings for qlora. GitHub Gist: instantly share code, notes, and snippets.
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1251363959982788639)** (5 messages): 

- **Seeking a Model for Business Use**: A member inquired about the best model for general-purpose support and business use. They specified that the largest model they can deploy is 7B.

- **Experimentation Recommended**: In response, another member suggested that the choice of the model will depend on the specific use case, whether tools/agents are being used, and the deployment/affordability constraints.

- **Game Screenshot Project with GPT-4 API**: A member shared their experience of using the **GPT-4 API to crop and caption over 150 screenshots from the game Mirror's Edge: Catalyst** and creating a LoRA for **Stable Diffusion** from those images.
  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1251300382718562386)** (10 messages🔥): 

- **RNNs vs RWKV-TS in Time Series Forecasting**: A member shared an [arXiv paper](https://arxiv.org/abs/2401.09093) discussing the declining dominance of traditional RNN architectures in time series tasks. The paper introduces **RWKV-TS**, a novel RNN-based model, which claims better efficiency, long-term sequence information capture, and computational scalability.
  
- **Advanced Prompt Option Impact on Production Time**: A member reported that disabling the advanced prompt option significantly reduces the production time during peak periods, improving fidelity and maintaining scene stability.

- **Web Scraping and RAG to Enhance LLMs**: A [Medium article](https://medium.com/ai-advances/how-to-power-up-llms-with-web-scraping-and-rag-975a165587f6) was shared, explaining how integrating web scraping with retrieval-augmented generation (RAG) can power up large language models (LLMs). Techniques referenced aim to enhance data collection and prompt accuracy.

- **Labor Market Impact of LLMs**: A member shared a study examining the [labor market impact potential of LLMs](https://ar5iv.labs.arxiv.org/html/2303.10130), revealing that large portions of the U.S. workforce could see significant changes in their job tasks due to LLMs. The investigation suggests both low and high-wage workers may experience shifts in their work responsibilities.

- **Reducing AI Hallucinations through RAG**: An article from Wired on [reducing AI hallucinations](https://www.wired.com/story/reduce-ai-hallucinations-with-rag/) using retrieval-augmented generation (RAG) was discussed. The approach involves a model gathering information from a custom database before generating responses, enhancing reliability and accuracy.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.wired.com/story/reduce-ai-hallucinations-with-rag/">Reduce AI Hallucinations With This Neat Software Trick</a>: A buzzy process called retrieval augmented generation, or RAG, is taking hold in Silicon Valley and improving the outputs from large language models. How does it work?</li><li><a href="https://www.facebook.com/share/p/rZucZJuafBDTZyVa/">Dull Men&#039;s Club | Hi, I&#039;m a mathematician who likes to think about things that would be useful if implemented but stand 0 chance of ever being implemented | Facebook</a>: Hi, I&#039;m a mathematician who likes to think about things that would be useful if implemented but stand 0 chance of ever being implemented.  For example, I often think we could improve the calendar...</li><li><a href="https://arxiv.org/abs/2401.09093">RWKV-TS: Beyond Traditional Recurrent Neural Network for Time Series Tasks</a>: Traditional Recurrent Neural Network (RNN) architectures, such as LSTM and GRU, have historically held prominence in time series tasks. However, they have recently seen a decline in their dominant pos...</li><li><a href="https://ar5iv.labs.arxiv.org/html/2303.10130">GPTs are GPTs: An Early Look at the Labor Market Impact Potential of Large Language Models</a>: We investigate the potential implications of large language models (LLMs), such as Generative Pre-trained Transformers (GPTs), on the U.S. labor market, focusing on the increased capabilities arising ...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1251560066600140830)** (18 messages🔥): 

- **Introducing Difoosion - A Web Interface for Stable Diffusion**: A member showcased their Web-Interface for Stable Diffusion leveraging the `diffusers` library and a Pure-Python web framework, Rio. They invited the community to check it out on [GitLab](https://gitlab.com/mad-moo/difoosion).

- **Ask Steve - LLMs Integration into Chrome**: A member developed a Chrome extension that integrates LLMs directly into the browser, akin to GitHub Copilot but for web navigation. They introduced the tool as a way to eliminate repetitive tasks and promoted the project at [Ask Steve](https://www.asksteve.to).

- **Ilaria RVC for Voice Conversion**: A member announced the creation of Ilaria RVC, a voice conversion space running on Zero, and thanked another user for their help. They shared the project on [Hugging Face Spaces](https://huggingface.co/spaces/TheStinger/Ilaria_RVC).

- **Demonstrating Transformers.js with LLM Temperature Parameter**: A blog post was shared about the temperature parameter in LLMs, featuring an interactive demo via Transformers.js running directly in the browser. The author highlighted how this approach could revolutionize educational content by eliminating the need for hosting models, shared on [Twitter](https://x.com/taha_yssne/status/1802607279809630562).

- **PowershAI - Combining PowerShell with AI**: A member introduced PowershAI, a PowerShell module allowing Function Calling with AI integration, which they developed while studying the OpenAI API. They shared their progress on [GitHub](https://github.com/rrg92/powershai) and detailed their journey in a blog post.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/TheStinger/Ilaria_RVC">Ilaria RVC - a Hugging Face Space by TheStinger</a>: no description found</li><li><a href="https://huggingface.co/spaces/ptx0/sd3-reality-mix">SD3 Reality Mix (Finetune) - a Hugging Face Space by ptx0</a>: no description found</li><li><a href="https://youtu.be/zRxJunvT3vo?si=UivatIByKGkwjCDX">Autonomous agents with layer-2 blockchain transaction capabilities</a>: This video walks you through turning a GPT into an agent that can make layer-2 nano-payments.To discover more:Dhali - https://dhali.io</li><li><a href="https://www.asksteve.to">Ask Steve - Unlock the Power of ChatGPT and Gemini in any web page!</a>: Ask Steve adds AI superpowers from ChatGPT &amp; Gemini to any web page, so you can get your everyday tasks done better and faster. FREE!</li><li><a href="https://x.com/taha_yssne/status/1802607279809630562">Tweet from Taha Yassine (@taha_yssne)</a>: I just wrote a blog post about the temperature parameter in LLMs, but really it was just an excuse to play with Transformers.js. I had fun implementing an interactive demo of the impact of T on genera...</li><li><a href="https://gitlab.com/mad-moo/difoosion">Jakob Pinterits / Difoosion · GitLab</a>: A simple web interface for Stable Diffusion - Including the new Stable Diffusion 3</li><li><a href="https://github.com/rrg92/powershai">GitHub - rrg92/powershai: IA com PowerShell</a>: IA com PowerShell. Contribute to rrg92/powershai development by creating an account on GitHub.</li><li><a href="https://iatalk.ing/powershai-powershell-inteligencia-artificial/">PowershAI: PowerShell + Inteligência Artificial &#8211; IA Talking 🤖 </a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1251433087720751176)** (16 messages🔥): 

- **QKV in ViT challenged and experiments planned**: A user questioned the correctness of the QKV implementation in ViTs, describing it as "wrong" and promising to conduct experiments to provide insights. More on this in the coming days.

- **HyperZZW vs Self-Attention**: A member shared a critique of the **self-attention** mechanism in ViTs, proposing the **HyperZZW operator** as a simpler and more reasonable alternative. They linked a detailed post on [X (Twitter)](https://x.com/harvie_zhang/status/1763867695383208382), suggesting that it deals better with spatial information.

- **Global HyperZZW and tokenization issues**: The same user argued that converting images into tokens in ViTs is fundamentally flawed and that the **Global HyperZZW** branch can manage global position info more efficiently with a matrix multiplication strategy.

- **Different strategies for image and text data**: They also stressed that images and text are fundamentally different, making ViT's implementation inappropriate for vision data, hinting at the use of prior information for future sequence modeling instead of attention mechanisms.

- **Slow neural loss and local feedback error**: Contributions like **slow neural loss** as local feedback error have been verified and mentioned as a potential key element for next-gen architectures, inspired by Hinton’s proposal. This was promoted with another [Twitter link](https://x.com/harvie_zhang/status/1802356749170778574?s=46&t=BsqYoGA8vIHGcXwORlMk7w).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/harvie_zhang/status/1763867695383208382.">Tweet from Harvie Zhang (@harvie_zhang)</a>: I propose a #HyperZZW operator with linear complexity to replace the #SelfAttention mechanism. The pixcel-level scores are obtained by Hadamard product between large implicit kernels and input activat...</li><li><a href="https://x.com/harvie_zhang/status/1802356749170778574?s=46&t=BsqYoGA8vIHGcXwORlMk7w.">Tweet from Harvie Zhang (@harvie_zhang)</a>: Do you think there is any difference between your proposed loss and my slow neural loss?   Please also refer to Eqn. 11-12 in https://arxiv.org/pdf/2401.17948.  Quoting Francesco Faccio (@FaccioAI)   ...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1251472397895336027)** (4 messages): 

- **Inquiring about VASA models**: A member asked if anyone has figured out the **VASA-like open-source models**. There's no indication of a follow-up or response in the provided messages. 
- **Interest in mobile CLIP**: Another member queried if **Hugging Face** will implement the **mobile CLIP model**. There were no further discussions or responses to this question in the provided messages.
  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1251495869165862922)** (5 messages): 

- **Fine-tuning BERT methods shared**: A member suggested using the method outlined in [this tutorial](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/BERT/Fine_tuning_BERT_(and_friends)_for_multi_label_text_classification.ipynb) for fine-tuning BERT. 
- **Randomness issue in HF model loading**: A user mentioned that loading HuggingFace models multiple times leads to different validation outputs, suggesting to save untrained model states for reproducibility. They noted, *"don't rely on HF initialization to be deterministic... Save your untrained model state"*. 
- **Trouble with Mistral-7b-0.3 context handling**: A new member is having issues with Mistral-7b-0.3 model not handling a context length properly, failing to answer questions beyond the first half of the context. They seek guidance on whether they misunderstood the model capabilities.
- **New Open Source TTS model**: A member shared a new TTS model, [MARS5-TTS](https://github.com/Camb-ai/mars5-tts), inviting their team to a talk on the Mozilla AI Main stage. They requested the community to submit any questions they might have for the MARS5-TTS team.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/Camb-ai/mars5-tts">GitHub - Camb-ai/MARS5-TTS: MARS5 speech model (TTS) from CAMB.AI</a>: MARS5 speech model (TTS) from CAMB.AI. Contribute to Camb-ai/MARS5-TTS development by creating an account on GitHub.</li><li><a href="https://github.com/NielsRogge/Transformers-Tutorials/blob/master/BERT/Fine_tuning_BERT_(and_friends)_for_multi_label_text_classification.ipynb">Transformers-Tutorials/BERT/Fine_tuning_BERT_(and_friends)_for_multi_label_text_classification.ipynb at master · NielsRogge/Transformers-Tutorials</a>: This repository contains demos I made with the Transformers library by HuggingFace. - NielsRogge/Transformers-Tutorials
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1251264088705597510)** (5 messages): 

- **Struggles with meme generator model**: A member sought advice on developing a **high-quality meme generator model** and asked for guidance from those with experience or interest in this domain. They emphasized the desire to produce *high-quality memes* and wondered about the initial steps.

- **Rate limit errors hinder progress**: One member reported *rate limit exceeding errors* and requested help to resolve this issue.

- **Overflow error in Stable Diffusion XL**: A detailed error involving SDXL loading was shared, showcasing an *Overflow error: cannot fit 'int' into an index-sized integer*. The provided code snippet and system information, including **GPU: A100** and **Torch: 2.3.1**, were part of the context.

- **Seeking examples for Diffusers with GCP's TPU**: Another member requested an example or guidance on using **Diffusers** with **GCP's TPU**.


  

---



### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1251266511373336626)** (184 messages🔥🔥): 

- **ChatGPT on iOS 18 remains uncertain**: A member asked if ChatGPT works with iOS 18, and another noted not to install beta software, underscoring the importance of using a stable iOS version like iOS 17 for ChatGPT. They also mentioned that beta users sign NDAs about new features.

- **Extracting transcripts from YouTube videos**: Members discussed tools for extracting transcripts from YouTube videos, including AI tools like **Otter.ai**, and a specific tool that requires the YouTube API key via the `fabric` library. One member suggested using Google Gemini's trial for a consumer-friendly experience.

- **Open source models beat GPT-4 in specific tasks**: [DeepSeek AI](https://x.com/deepseek_ai/status/1802680388256768145) released an open-source model reportedly outperforming GPT-4 Turbo in specialized tasks like coding and math. This sparked discussions about open-source versus proprietary models.

- **Connecting OpenAI models to databases**: A member asked about integrating OpenAI's LLM with a continuously updating database, and another shared links to [OpenAI's Cookbook](https://cookbook.openai.com/examples/vector_databases/readme) with examples for vector databases, which are foundational for supporting semantic search and reducing hallucinations in responses.

- **Dream Machine and Sora's AI capabilities**: There was enthusiastic discussion about **Luma's Dream Machine** video capabilities, compared to the anticipated **Sora**, revealing some users' impatience with the limited release of Sora. Members noted its impressive but still evolving functionality, with unique features like incorporating consistent physical motion.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/deepseek_ai/status/1802680388256768145">Tweet from DeepSeek (@deepseek_ai)</a>: DeepSeek-Coder-V2: First Open Source Model Beats GPT4-Turbo in Coding and Math  &gt; Excels in coding and math, beating GPT4-Turbo, Claude3-Opus, Gemini-1.5Pro, Codestral. &gt; Supports 338 programmin...</li><li><a href="https://x.com/deepseek">Tweet from undefined</a>: no description found</li><li><a href="https://huggingface.co/spaces/sanchit-gandhi/whisper-jax">Whisper JAX - a Hugging Face Space by sanchit-gandhi</a>: no description found</li><li><a href="https://copilot.microsoft.com/sl/b6od4WvwBVs">What is Copilot? - Microsoft Copilot: 你的日常 AI 助手</a>: Microsoft Copilot 利用 AI 的强大功能来提高工作效率、释放创造力，并通过简单的聊天体验帮助你更好地理解信息。</li><li><a href="https://tenor.com/view/potato-fries-poop-gif-5060738">Potato Fries GIF - Potato Fries Poop - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://cookbook.openai.com/examples/rag_with_graph_db">RAG with a Graph database | OpenAI Cookbook</a>: no description found</li><li><a href="https://cookbook.openai.com/examples/vector_databases/readme">Vector databases | OpenAI Cookbook</a>: no description found
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1251274066560811080)** (49 messages🔥): 

- **Custom GPTs privacy setting confusion**: A member struggled with setting their Custom GPTs to private, mentioning that the most restricted option now is "invite only," but it was still showing as available to everyone. A workaround suggested is to create a copy and restrict it to people with the link, then delete the original.
- **Funny idea for a GPT mod**: A member suggested making a Fallout or Skyrim mod that changes all the game's dialogue to zoomer slang or any specified prompt, noting it would be amusing.
- **Access issues with free-tier GPT interactions**: Several members reported difficulties in accessing GPT interactions, with conversations requiring a paid subscription to continue. This seems to be affecting multiple users, with some confirming the same issue with their friends.
- **Specifying actions for Custom GPTs**: A user inquired about setting specific actions like web browsing in their custom GPT and was advised to prompt the GPT accordingly for when to use certain tools.
- **GPT usage limits frustration**: Another user expressed frustration over GPT not loading and servers being down, with others confirming similar issues. For real-time updates, users were directed to check [status.openai.com](https://status.openai.com).
  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1251273943017459764)** (28 messages🔥): 

- **3D Models Struggle with No Shadows**: Members discussed the challenges of creating 3D models with no shadows or lighting. One shared hope to create texture resembling an "albedo map" to aid in 3D conversions, while another suggested inpainting or using tools like Canva to minimize shadows.

- **Extracting Information from GPT-4**: A member faced issues with GPT-4 mixing sample and target emails during information extraction. Solutions included clearly separating samples with distinct markers and clarifying instructions.

- **Generate Detailed Roadmaps with ChatGPT**: To explore topics like marketing and branding in depth, members recommended strategies such as step-back prompting and using detailed queries. Shared tips included creating topic trees and using browser tools for specific research.

- **Handling ChatGPT's Request Refusals**: A user experienced consecutive refusals from ChatGPT to fulfill certain requests without clear reasons. Tips shared included repeating the prompt and asking for detailed explanations while requesting the fulfillment.

- **Generating Tables from XML Data**: A member inquired about prompts for extracting XML data into table form and generating specific token amounts with the GPT API. The community awaits further responses to this technical query.
  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1251273943017459764)** (28 messages🔥): 

- **Secrets to 3D Model Prompts**: A member suggested finding 3 examples of a 3D model with no shadows or lighting, asking ChatGPT to notate their lack and then generating a new image. Another user noted that completely eliminating shadows seems impossible due to language limitations and rendering corrections by ChatGPT and Dall-E.

- **Using Separate Samples for GPT-4**: To prevent GPT-4 from mixing sample and target emails, members debated using distinct markers. Clear separation and specific instructions can prevent content amalgamation.

- **Balancing Shadows in 3D Models**: A detailed discussion on minimizing shadows and light on objects for better 3D model texture mapping ensued. The consensus was that the baked-in shading interferes with albedo map creation, recommending using the generated shape as a base model instead.

- **Generating Marketing Roadmaps with ChatGPT**: One user sought advanced insights on marketing topics like Brand Archetypes using ChatGPT. Members advised step-back prompting and specific roadmaps of subtopics; suggestions included using clear directives and external resources for deeper dives.

- **ChatGPT Refusal Quirks**: Several users reported that ChatGPT sometimes refuses requests without giving reasons. The proposed workaround involves asking ChatGPT to explain refusals, which may prompt it to fulfill the request.
  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1251322975228268554)** (250 messages🔥🔥): 

- **SD3 Models Struggle with Artifacts and Training**: Members discussed the stability and training challenges with SD3 models, noting that **loss stabilization** remains complex. Explicit concerns were raised about *non-uniform timestep sampling* and the lack of critical components such as **qk norm**.

- **Timestep Weighting in Training**: Discussion highlighted different approaches to timestep weighting with **V-prediction models**. One user prefers *uniform sampling* while reweighting loss, segmenting schedules into smaller batches to distribute training effectively.

- **Open-source T2I Models**: Queries and recommendations about the best **open T2I models** with character consistency led to [GitHub resources](https://github.com/PRIV-Creation/Awesome-Controllable-T2I-Diffusion-Models) for controllable text-to-image generation. [Theatergen for character management](https://github.com/donahowe/Theatergen) was also discussed as an option for consistent multi-turn image generation.

- **ComfyUI and Adaptive ODE Solvers**: A member shared a [GitHub link](https://github.com/redhottensors/ComfyUI-ODE) for adaptive ODE solvers implemented for SD3, suggesting they offer better results than existing fixed-step solvers and could serve as a valuable reference or alternative for current diffusers.

- **Fudan's Open-source Video Generative Model**: Spirited discussion erupted around Fudan University's [Hallo model](https://github.com/fudan-generative-vision/hallo) for video generation from single images and audio, with another tool to run it locally shared [on FXTwitter](https://fxtwitter.com/cocktailpeanut/status/1802376983021580428). Members expressed interest in integrating it with Text-to-Speech systems like Udio or Suno.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://runwayml.com/blog/introducing-gen-3-alpha/">Introducing Gen-3 Alpha: A New Frontier for Video Generation</a>: Gen-3 Alpha is the first of the next generation of foundation models trained by Runway on a new infrastructure built for large-scale multimodal training. It is a major improvement in fidelity, consist...</li><li><a href="https://fxtwitter.com/JoeSiyuZhu/status/1801780534022181057">Tweet from Siyu ZHU (@JoeSiyuZhu)</a>: We (Fudan, Baidu, ETH Zurich) have open-sourced the superior-performing video generative model that make single image sing and talk from audio reference, and can adaptively control facial expression. ...</li><li><a href="https://huggingface.co/CaptionEmporium">CaptionEmporium (Caption Emporium)</a>: no description found</li><li><a href="https://bottosson.github.io/posts/oklab/">A perceptual color space for image processing</a>: A perceptual color space for image processing A perceptual color space is desirable when doing many kinds of image processing. It is useful...</li><li><a href="https://tenor.com/view/tonton-friends-yuta-chubby-shiba-shiba-inu-dog-gif-16103647401394133233">Tonton Friends Yuta GIF - Tonton Friends Yuta Chubby Shiba - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/im-doing-my-part-serious-stare-gif-11979677">Im Doing My Part Serious GIF - Im Doing My Part Serious Stare - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/thats-the-neat-part-you-dont-invincible-gif-27194608">Thats The Neat Part You Dont Invincible GIF - Thats The Neat Part You Dont Invincible - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/oppo-us-research/FA-VAE">GitHub - oppo-us-research/FA-VAE: Description： Frequency Augmented Variational Autoencoder for better Image Reconstruction</a>: Description： Frequency Augmented Variational Autoencoder for better Image Reconstruction - oppo-us-research/FA-VAE</li><li><a href="https://fxtwitter.com/cocktailpeanut/status/1802381406196154570">Tweet from cocktail peanut (@cocktailpeanut)</a>: Kanye, singing &#34;Dynamite&#34; by BTS</li><li><a href="https://fxtwitter.com/cocktailpeanut/status/1802376983021580428">Tweet from cocktail peanut (@cocktailpeanut)</a>: Run High Quality Lipsync Locally, with 1 Click.  [NVIDIA ONLY] The quality of lip syncing you get from Hallo is the best I&#39;ve seen.  So I wrote a gradio app AND a 1 click launcher for this. Enjoy!...</li><li><a href="https://github.com/PRIV-Creation/Awesome-Controllable-T2I-Diffusion-Models">GitHub - PRIV-Creation/Awesome-Controllable-T2I-Diffusion-Models: A collection of resources on controllable generation with text-to-image diffusion models.</a>: A collection of resources on controllable generation with text-to-image diffusion models. - PRIV-Creation/Awesome-Controllable-T2I-Diffusion-Models</li><li><a href="https://github.com/donahowe/Theatergen">GitHub - donahowe/Theatergen: TheaterGen: Character Management with LLM for Consistent Multi-turn Image Generation</a>: TheaterGen: Character Management with LLM for Consistent Multi-turn Image Generation - donahowe/Theatergen</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1dhd7vz/the_developer_of_comfy_who_also_helped_train_some/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://replicate.com/fofr/consistent-character?prediction=v3b629v38drgm0cg3h1bmmz1zm">fofr/consistent-character – Run with an API on Replicate</a>: no description found
</li>
</ul>

</div>
  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1251316649118797944)** (34 messages🔥): 

- **Logical Reasoning Challenges with AIW Problems**: Discussion highlighted the frequent use of names like "Alice" in logical reasoning problems, which may bias LLMs. A member shared that *"Phi-2 performed horrible in general,"* showing *"severe reasoning breakdown"* in SOTA LLMs on the AIW problem described in [this paper](https://arxiv.org/abs/2406.02061).
  
- **Experiment Tactics to Address Bias**: One member experimented by changing the problem setup to remove bias from known examples, noting that models like *"GPT4o and Claude-Opus managed to solve it,"* while others failed. Failures were attributed to the LLMs’ misinterpretations like handling groupings incorrectly or hallucinating geometric associations.

- **Reasoning Sensitivity in Models**: Further analysis showed LLMs are *"VERY SENSITIVE to even slight AIW problem variations,"* with Fig 11 from [the referenced paper](https://arxiv.org/abs/2406.02061) illustrating drastic fluctuations in correct response rates with slight changes, emphasizing the fragile state of their reasoning capabilities.

- **Symbolic AI Hybrids for Deductive Reasoning**: A query about research efforts combining LLMs with symbolic AI for improved deductive reasoning led to the recommendation of [Logic-LM](https://arxiv.org/abs/2305.12295), which integrates LLMs with symbolic solvers to significantly boost logical problem-solving performance.

- **JEPA for Building a Collective Vision in Email Assistants**: Anu4938 shared ambitions of using [JEPA](https://openreview.net/pdf?id=BZ5a1r-kVsf) to create an email assistant aimed at maximizing collective good and efficiently managing complexities. The envisioned assistant emphasizes values such as environmental respect, climate change action, and fostering global cooperation.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2406.10208">Glyph-ByT5-v2: A Strong Aesthetic Baseline for Accurate Multilingual Visual Text Rendering</a>: Recently, Glyph-ByT5 has achieved highly accurate visual text rendering performance in graphic design images. However, it still focuses solely on English and performs relatively poorly in terms of vis...</li><li><a href="https://www.anthropic.com/research/claude-character">Claude’s Character</a>: Anthropic is an AI safety and research company that&#x27;s working to build reliable, interpretable, and steerable AI systems.</li><li><a href="https://x.com/JJitsev/status/1801760448657727737">Tweet from Jenia Jitsev 🏳️‍🌈 🇺🇦 (@JJitsev)</a>: Community was digging into our work https://arxiv.org/abs/2406.02061 that shows severe reasoning breakdown on the very simple AIW problem in SOTA LLMs. Following intense debates,  to showcase further ...</li><li><a href="https://arxiv.org/abs/2305.12295">Logic-LM: Empowering Large Language Models with Symbolic Solvers for Faithful Logical Reasoning</a>: Large Language Models (LLMs) have shown human-like reasoning abilities but still struggle with complex logical problems. This paper introduces a novel framework, Logic-LM, which integrates LLMs with s...
</li>
</ul>

</div>
  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1251282157679280268)** (161 messages🔥🔥): 

- **Debate on Large Model Usability**: A heated discussion took place on releasing and meme-ing about large models like the 200T parameter model, which are beyond most users' reach. One user humorously mentioned, "I am this close to making a 200T parameter model. Claim it is AGI."

- **Qwen7B vs Llama3 8B**: Members discussed the performance comparison between Qwen7B and Llama3 8B with one user mentioning that small LLMs like Qwen7B are unlikely to outperform Llama3 8B, emphasizing its current superiority in the field.

- **Custom Llama3 Template Issue**: There was a detailed technical exchange about training configurations and issues related to the `chat_template` setting when training with Llama3 models. One user shared a link to [fix custom Llama3 prompt strategies](https://github.com/xzuyn/axolotl/blob/dan_metharme/src/axolotl/prompt_strategies/customllama3.py) that resolved some issues.

- **GPU and Optimization Feedback for PyTorch**: A call for feedback from users using various GPUs to assist PyTorch optimizations saw diverse responses, including GPUs like AMD MI300X, RTX 3090, Google TPU v4, and 4090 with tinygrad.

- **Shared Projects and Resources**: Users shared several resources, including a blog post on [CryptGPT: Privacy-Preserving LLMs](https://x.com/diwanksingh/status/1802118343446724655), a language-specific GPT chat model [DanskGPT](https://chat.danskgpt.dk), and GitHub links for setting up chat UI similar to HuggingChat using [Huggingface's chat-ui project](https://github.com/huggingface/chat-ui).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/turboderp/Cat-Llama-3-70B-instruct">turboderp/Cat-Llama-3-70B-instruct · Hugging Face</a>: no description found</li><li><a href="https://x.com/diwanksingh/status/1802118343446724655">Tweet from Diwank Singh (@diwanksingh)</a>: http://x.com/i/article/1802116084507848704</li><li><a href="https://x.com/_philschmid/status/1801732678233825327?t=qiUAp1TbPUTcwQi1ikPy5Q&s=19">Tweet from Philipp Schmid (@_philschmid)</a>: What a lucky day it is today. @nvidia release a strong 340B open LLM, which needs 8x H200 to run. And at the same day the first providers start hosting 8x H200. 🍀</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/5783839c6e29bb148041338772040c85aaae4646/src/axolotl/cli/train.py#L53-L59">axolotl/src/axolotl/cli/train.py at 5783839c6e29bb148041338772040c85aaae4646 · OpenAccess-AI-Collective/axolotl</a>: Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.</li><li><a href="https://github.com/huggingface/chat-ui">GitHub - huggingface/chat-ui: Open source codebase powering the HuggingChat app</a>: Open source codebase powering the HuggingChat app. Contribute to huggingface/chat-ui development by creating an account on GitHub.</li><li><a href="https://github.com/huggingface/trl/blob/main/trl/trainer/rloo_trainer.py">trl/trl/trainer/rloo_trainer.py at main · huggingface/trl</a>: Train transformer language models with reinforcement learning. - huggingface/trl</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/5783839c6e29bb148041338772040c85aaae4646/src/axolotl/prompt_strategies/sharegpt.py#L42-L55">axolotl/src/axolotl/prompt_strategies/sharegpt.py at 5783839c6e29bb148041338772040c85aaae4646 · OpenAccess-AI-Collective/axolotl</a>: Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.</li><li><a href="https://github.com/xzuyn/axolotl/blob/dan_metharme/src/axolotl/prompt_strategies/customllama3.py">axolotl/src/axolotl/prompt_strategies/customllama3.py at dan_metharme · xzuyn/axolotl</a>: Go ahead and axolotl questions. Contribute to xzuyn/axolotl development by creating an account on GitHub.</li><li><a href="https://github.com/lm-sys/FastChat/tree/main?tab=readme-ov-file#method-2-from-source">GitHub - lm-sys/FastChat: An open platform for training, serving, and evaluating large language models. Release repo for Vicuna and Chatbot Arena.</a>: An open platform for training, serving, and evaluating large language models. Release repo for Vicuna and Chatbot Arena. - lm-sys/FastChat</li><li><a href="https://chat.danskgpt.dk">DanskGPT</a>: Dansk sprogteknologi tilgængelig for alle, helt gratis.
</li>
</ul>

</div>
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1251709838606602311)** (4 messages): 

- **Llama3 Bug Halts Development**: A user raised an issue on [GitHub](https://github.com/OpenAccess-AI-Collective/axolotl/issues/1700) regarding a bug introduced on June 7 that prevents tuning Llama 3 or Mistral models. The bug is affecting several users, with 6 people confirming its impact, and while a workaround exists, they insist that the main branch needs fixing.
- **Investigating the Bug Source**: Another member asked if the issue might be related to setting `remove_unused_column` to false, but then concluded that the "length" keyword argument problem likely stems from a specific commit. The problematic commit was identified after a `bisect`, confirming it as the source of the issue.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/OpenAccess-AI-Collective/axolotl/issues/1700">Llama3-8b: LlamaForCausalLM.forward() got an unexpected keyword argument &#39;length&#39; · Issue #1700 · OpenAccess-AI-Collective/axolotl</a>: Please check that this issue hasn&#39;t been reported before. I searched previous Bug Reports didn&#39;t find any similar reports. Expected Behavior I expect the training run to finish and save the we...</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/issues/17">Jeopardy bot! by winglian · Pull Request #17 · OpenAccess-AI-Collective/axolotl</a>: https://huggingface.co/openaccess-ai-collective/jeopardy-bot
</li>
</ul>

</div>
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1251299743582126090)** (9 messages🔥): 

- **Config confusion for dataset types**: A user expressed confusion regarding the dataset type field in their `axolotl` config, particularly for `alpaca_chat.load_qa`, referencing the [dataset formats](https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/inst_tune.html#alpaca_chat.load_qa). Another user confirmed that the config format provided is correct.

- **Running accelerate on SLURM clusters**: A user shared a SLURM job script for running `axolotl` with `accelerate` and `deepspeed`, specifying mixed precision and multi-GPU settings. They advised replacing `$PMI_RANK` with `$SLURM_NODEID` if the former is unavailable.

- **QDora issues in Axolotl**: A user inquired about getting QDora to work with Axolotl, and another user replied that it hangs after a few steps, suggesting it's unreliable. Further details on building QDora from source were sought.

- **Using axolotl for personality extraction**: A user asked if anyone has used Axolotl to train models for extracting personalities from text and linked to [Delphi AI](https://www.delphi.ai/) for reference. They asked if the `oasst` dataset format would be appropriate, linking to the [documentation](https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/inst_tune.html#oasst).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.delphi.ai/?">Delphi</a>: Clone Yourself.</li><li><a href="https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/inst_tune.html#oasst?">Axolotl - Instruction Tuning</a>: no description found</li><li><a href="https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/inst_tune.html#alpaca_chat.load_qa">Axolotl - Instruction Tuning</a>: no description found</li><li><a href="https://openaccess-ai-collective.github.io/axolotl/docs/config.html">Axolotl - Config options</a>: no description found
</li>
</ul>

</div>
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/1251470432813580379)** (4 messages): 

- **Dataset Config Issues Resolved**: A member requested a dataset config section due to encountering a `ValueError` stating *"unhandled prompt tokenization strategy: sharegpt."* Another member shared a configuration link from Discord ([link](https://discord.com/channels/1104757954588196865/1104757955204743201/1251481036848889927)), which resolved the issue.
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[community-showcase](https://discord.com/channels/1104757954588196865/1117851527143493664/1251620995874295932)** (1 messages): 

- **First Finetune with Axolotl Shines**: "Had a blast finetuning my first LLMs with Axolotl!" The author reports successfully transitioning an unstructured press release into a structured output, hinting at further exploring OpenAI API’s function calling for improved accuracy.
- **Exploring Press Release Data Extraction Efficiency**: "[We previously looked into how well](https://mlops.systems/posts/2024-06-03-isafpr-evaluating-baseline.html) LLMs could extract structured data from press releases." The initial evaluations revealed that while LLMs performed decently, there was noticeable room for improvement. 
- **Future Comparisons Promised**: Emphasizing the use of function calling over raw prompting for better accuracy, a separate post on finetuning comparisons is hinted at. For more details, the author refers readers to [a detailed post](https://mlops.systems/posts/2024-06-15-isafpr-first-finetune.html).

**Link mentioned**: <a href="https://mlops.systems/posts/2024-06-15-isafpr-first-finetune.html">Alex Strick van Linschoten - Finetuning my first LLM(s) for structured data extraction with axolotl</a>: I finetuned my first LLM(s) for the task of extracting structured data from ISAF press releases. Initial tests suggest that it worked pretty well out of the box.

  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-help-bot](https://discord.com/channels/1104757954588196865/1225300056442409040/1252235292166066227)** (11 messages🔥): 

- **Adjusting Inference Parameters in Axolotl**: A user asked how to set inference parameters like temperature or seed while running `accelerate launch -m axolotl.cli.inference`. It was suggested to modify the inference script directly or the configuration file if the command-line arguments for these settings aren't supported, showcasing an example of how to adjust `generation_config`.

- **Request for Fine-Tuning Vision Models**: A user inquired about fine-tuning vision models. It was explained that the process involves loading a pre-trained model (e.g., ResNet-50), preparing the dataset, modifying the final layers if necessary, defining data transforms, and then setting up a training loop with proper loss function and optimizer.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz">no title found</a>: no description found</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=077f3d3e-542e-4b6f-b825-cea95799abf7)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=8bf72d5d-fdad-4aa8-9c16-12b8b7bad9d9)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.
</li>
</ul>

</div>
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-phorm-bot](https://discord.com/channels/1104757954588196865/1225558824501510164/1252211724786864158)** (11 messages🔥): 

- **Doubling context length in models needs careful adjustments**: To train a model at **2x the native context length** (e.g., 16k from 8k), users need to modify several settings related to model architecture, data processing, and training configuration. Key changes include adjusting maximum position embeddings and training parameters like batch size and gradient accumulation steps.

- **Fine-tuning vision models with Axolotl explained**: A step-by-step guide is provided for fine-tuning vision models using Axolotl. It involves cloning the Axolotl repository, installing dependencies, preparing the dataset, modifying the configuration file, and using Accelerate for training and inference.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/OpenAccess-AI-Collective/axolotl">GitHub - OpenAccess-AI-Collective/axolotl: Go ahead and axolotl questions</a>: Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=46936261-cfc5-4e25-8c3f-1048a546f037)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=0be694a5-7efc-4cdb-97f6-6691bd442899)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.
</li>
</ul>

</div>
  

---



### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1252303453028614204)** (1 messages): 

- **Curiosity speaks every language; partners with SoftBank**: Perplexity announced a strategic partnership with **SoftBank** to offer **Perplexity Pro** free for one year to customers using SoftBank, Y!mobile, and LINEMO services. This premium version of Perplexity, valued at 29,500 yen annually, provides users with a revolutionary AI answer engine for exploring and learning. [More info](https://pplx.ai/softbank).

**Link mentioned**: <a href="https://pplx.ai/softbank">SoftBank Corp. Launches Strategic Partnership with Leading AI Startup Perplexity | About Us | SoftBank</a>: SoftBank Corp.‘s corporate page provides information about “SoftBank Corp. Launches Strategic Partnership with Leading AI Startup Perplexity”.

  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1251271508668711033)** (187 messages🔥🔥): 

- **Agentic Search AB Testing Secrets**: Community members discussed the new [Agentic Pro Search](https://www.reddit.com/r/perplexity_ai/comments/1ddkvgc/comment/l8czo5j/) being in A/B testing. One user shared a Reddit link on how to cheat the system but later reconsidered to avoid messing up the control group. 
- **Confusion Over Perplexity's Features and Model Settings**: Users had various questions about using Perplexity, such as setting a system prompt, formatting answers, accessing writing modes, and experiencing issues like temperature changes or the chat freezing. They shared solutions like contacting support or clearing browser cache for bugs.
- **Perplexity vs. ChatGPT and Investment Discussions**: Members debated whether it was worth having both Perplexity and ChatGPT subscriptions concurrently and discussed the potential of investing in Perplexity. Comparisons focused on the strengths of each platform for specific use cases like writing and research.
- **Concerns Over Web Crawling and Privacy**: Some users raised concerns about Perplexity's crawling behavior not respecting `robots.txt` and masking user agents. Suggestions for blocking or addressing this issue included using JA3 fingerprinting and bot endpoints.
- **Customizable Features and Document Handling**: Members inquired and discussed uploading files, handling extensive document collections, and potential integrations with academic databases like academia.edu. Solutions included using other AI tools like custom GPTs on OpenAI and NotebookLM to manage large document loads.

**Link mentioned**: <a href="https://www.reddit.com/r/perplexity_ai/comments/1ddkvgc/comment/l8czo5j/">Reddit - Dive into anything</a>: no description found

  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1251336979992547338)** (10 messages🔥): 

- **Tanstack Table Search Shared**: A member shared a link to a [Perplexity AI search](https://www.perplexity.ai/search/tanstack-table-vJoqV4R9R4iPUHhGspaL2A) related to Tanstack Table queries. This sparked interest in data table management tools.
  
- **Pet Food in Russia Search**: Another member provided a link to a [search about pet food in Russia](https://www.perplexity.ai/search/petfood-in-Russia-UzyFxsq3RnCxXSu.x3BbBg). Discussions likely revolved around the pet food market in Russia.

- **Prostate Health Paper Public Issue**: A user unintentionally made their [prostate health paper](https://www.perplexity.ai/page/Is-Prostate-Health-K4Ha3VDZRYicAbJl8vET9w) public and sought help to fix it. Another member advised using the "Unpublish Page" button in the Pages menu.

- **Elephant Communication Page**: A contributor shared a link to a [page discussing elephant communication](https://www.perplexity.ai/page/Elephants-Call-Each-036FUcDlSNOmVbVpFubFDQ). This might have led to conversations around animal behavior and communication methods.

- **Elder Scrolls Page** (duplicated): A couple of messages included links to a [page about The Elder Scrolls](https://www.perplexity.ai/page/The-Elder-Scrolls-rE8j7mGJTuuLZbl4F.FZnw). This probably indicates a shared interest in this gaming series among users.
  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1251982282785292338)** (3 messages): 

- **Ask-anything feature in Custom GPT struggles**: A user successfully got their Custom GPT working but wants it to handle any prompts or queries. Another suggested explaining the problem to GPT-4o with a specific schema and error details to resolve issues with Action/Function Calling to the Perplexity API.
- **Inquire for closed-beta access timeframe**: A member asked about the expected response time for closed-beta access to the API. They mentioned their project at Kalshi is heavily dependent on accessing sources and is ready to launch pending this access.
  

---



### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1252247794937237504)** (3 messages): 

- **Channel needs Autechre to hurt ears**: One member suggested they need more Autechre in the channel and shared the [YouTube video of "Autechre - Gantz Graf (Official Music Video) 1080p HD"](https://www.youtube.com/watch?v=ev3vENli7wQ) to achieve this.
- **Autechre to heal your soul**: To balance the previous suggestion, the same member shared the [YouTube video of "Autechre - Altibzz"](https://www.youtube.com/watch?v=m3ZyEGTIsvE), describing it as a way to heal your soul.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=m3ZyEGTIsvE">Autechre - Altibzz</a>: The ambient opener to their new album &quot;Quaristice.&quot; Music by Rob Brown and Sean Booth (c)/(p) 2007-8 Warp Records, Ltd.</li><li><a href="https://www.youtube.com/watch?v=ev3vENli7wQ">Autechre - Gantz Graf (Official Music Video) 1080p HD</a>: Autechre - Gantz Graf (Official Music Video) HD
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1251380472131358772)** (5 messages): 

- **Neurons play Doom in YouTube Video**: Shared [YouTube video](https://youtu.be/c-pWliufu6U) titled "Growing Living Neurons to Play...Doom? | Part 2!" explores the concept of using living neurons to play the video game **Doom**. It's an intriguing intersection of biotech and gaming.

- **Automated Bias and Indoctrination**: Link to [ResearchGate paper](https://www.researchgate.net/publication/378191925_Automated_Bias_and_Indoctrination_at_Scale_Is_All_You_Need) discusses the market-driven trajectory of AI and novel risks related to human bias and cognition at scale. The paper critiques "stochastic parrots" like LLMs as tools for manipulation aligned with corporate biases.

- **Solving the Alignment Problem in mASI**: A thought-provoking [paper](https://www.researchgate.net/publication/372083027_The_Ethical_Basilisk_Thought_Experiment) aims to highlight ethical implications and extreme liabilities in AI decision-making. It introduces the concept of "Ethical Center of Gravity" for balancing ethical deeds to mitigate dystopian risks.

- **Efficient LLM Inference with vLLM**: Blog post details about [vLLM](https://arxiv.org/abs/2309.06180), an open-source inference engine using PagedAttention to improve memory usage and throughput. vLLM can run models with significantly fewer GPUs and boasts up to 24x higher throughput compared to HuggingFace Transformers.

- **Stable Diffusion Subreddit Protests Reddit API Changes**: The [r/StableDiffusion](https://www.reddit.com/r/StableDiffusion/comments/1dhd7vz/the_developer_of_comfy_who_also_helped_train_some/#lightbox) subreddit reopens after protesting changes to Reddit's open API policy. The protest highlighted concerns about the impact on app developers, moderation, and accessibility for blind users.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://youtu.be/c-pWliufu6U">Growing Living Neurons to Play...Doom? | Part 2!</a>: Use code thoughtemporium at the link below to get an exclusive 60% off anannual Incogni plan: https://incogni.com/thoughtemporium____________________________...</li><li><a href="https://x.com/MoritzW42/status/1801418940515815676">Tweet from Moritz Wallawitsch 🗽 (@MoritzW42)</a>: An Introduction to vLLM and PagedAttention   (originally published on @runpod_io&#39;s blog)  1/6  vLLM is an open-source LLM inference and serving engine (developed by @woosuk_k @KaichaoYou @zhuohan1...</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1dhd7vz/the_developer_of_comfy_who_also_helped_train_some/#lightbox">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1251252712947843153)** (124 messages🔥🔥): 

- **Stable Diffusion 3 and Usage**: Members noted that **lying is allowed for anime**, but only for non-human characters. Shared link to access [Stable Diffusion 3](https://huggingface.co/spaces/stabilityai/stable-diffusion-3-medium).
- **NVIDIA's Synthetic Data Model**: Discussion about **Nemotron-4-340B-Instruct**, a large language model for synthetic data generation, optimized for English chat and supporting a 4,096 tokens context length. Available on [Hugging Face](https://huggingface.co/nvidia/Nemotron-4-340B-Instruct) and explored for its usage and competitive implications for NVIDIA's customer relations.
- **Realtime Inference and ComfyUI**: A member suggested using **ComfyUI with TensorRT SD-Turbo** for near real-time inference, especially fun when paired with a webcam feed for image manipulation.
- **OpenAI's Shift to For-Profit**: [Sam Altman](https://x.com/aaronpholmes/status/1801785687030829240?s=46) has informed shareholders that OpenAI might transition to a for-profit entity akin to rivals like Anthropic and xAI.
- **Model Merging and MoE Debate**: Extended discussion on the practicality and performance of **Mixture of Experts (MoE)** models and merging strategies, with hesitations about the efficacy of merging methods versus comprehensive fine-tuning. Links shared to relevant PR on [llama.cpp](https://github.com/ggerganov/llama.cpp/pull/6453) and MoE models on [Hugging Face](https://huggingface.co/Kquant03/CognitiveFusion-4x7B-bf16-MoE).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://situational-awareness.ai/">Introduction - SITUATIONAL AWARENESS: The Decade Ahead</a>: Leopold Aschenbrenner, June 2024 You can see the future first in San Francisco. Over the past year, the talk of the town has shifted from $10 billion compute clusters to $100 billion clusters to trill...</li><li><a href="https://arxiv.org/abs/2406.07394">Accessing GPT-4 level Mathematical Olympiad Solutions via Monte Carlo Tree Self-refine with LLaMa-3 8B</a>: This paper introduces the MCT Self-Refine (MCTSr) algorithm, an innovative integration of Large Language Models (LLMs) with Monte Carlo Tree Search (MCTS), designed to enhance performance in complex m...</li><li><a href="https://huggingface.co/spaces/stabilityai/stable-diffusion-3-medium">Stable Diffusion 3 Medium - a Hugging Face Space by stabilityai</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=xm1B3Y3ypoE">Is the Intelligence-Explosion Near? A Reality Check.</a>: Learn more about neural networks and large language models on Brilliant! First 30 days are free and 20% off the annual premium subscription when you use our ...</li><li><a href="https://huggingface.co/nvidia/Nemotron-4-340B-Instruct">nvidia/Nemotron-4-340B-Instruct · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/Kquant03/CognitiveFusion-4x7B-bf16-MoE">Kquant03/CognitiveFusion-4x7B-bf16-MoE · Hugging Face</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=c-pWliufu6U">Growing Living Neurons to Play...Doom? | Part 2!</a>: Use code thoughtemporium at the link below to get an exclusive 60% off anannual Incogni plan: https://incogni.com/thoughtemporium____________________________...</li><li><a href="https://x.com/ryanels4/status/1801898008755241251?s=46">Tweet from Ryan Els (@RyanEls4)</a>: AI revealed 😲</li><li><a href="https://tenor.com/view/godfather-massacre-sad-gif-16810633">Godfather Massacre GIF - Godfather Massacre Sad - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://x.com/aaronpholmes/status/1801785687030829240?s=46">Tweet from aaron holmes (@aaronpholmes)</a>: New: Sam Altman has told shareholders that OpenAI is considering becoming a for-profit company that would no longer be controlled by a nonprofit board https://www.theinformation.com/articles/openai-ce...</li><li><a href="https://x.com/alexalbert__/status/1801668464920379648?s=46">Tweet from Alex Albert (@alexalbert__)</a>: Loved Golden Gate Claude? 🌉   We&#39;re opening limited access to an experimental Steering API—allowing you to steer a subset of Claude&#39;s internal features.  Sign up here: https://forms.gle/T8fDp...</li><li><a href="https://huggingface.co/datasets/BAAI/Infinity-Instruct">BAAI/Infinity-Instruct · Datasets at Hugging Face</a>: no description found</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/6453">Adding Support for Custom Qwen2moe Architectures with mergekit-qwen2 by DisOOM · Pull Request #6453 · ggerganov/llama.cpp</a>: Statement: This has nothing to do with the fine-grained MoE architecture in Qwen/Qwen1.5-MoE-A2.7B. It is more akin to a traditional MoE, except that its experts are derived from the qwen2 (qwen1.5...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1251398975899045960)** (22 messages🔥): 

- **Setting Up Llama3 8B is Challenging**: *Plasmator* asked for tips on training Llama3 8B for a specific style and deploying a fast OAI-compatible endpoint on M2 Ultra. *Teknium* recommended using **unsloth qlora**, **Axolotl**, and **Llamafactory** for training, and **lmstudio** or **Ollama** for endpoint deployment on a Mac.

- **RAG Method Inquiry**: *Rbccapitalmarkets* inquired if the set-based prompting technique from a recent paper could work with RAG (Retrieval-Augmented Generation). They shared a [link to the paper](https://arxiv.org/abs/2406.06581) for further context.

- **PEFT Methods Discussion at CMU**: *420gunna* mentioned a CMU Advanced NLP course where the professor plugs his own paper about two new PEFT (Parameter Efficient Fine-Tuning) methods. They shared a [YouTube link](https://youtu.be/KLJ3EEo8aPU?si=D155UK9mgoWmqOn5&t=2942) to the lecture for those interested.

- **Nvidia’s Nemotron Model**: *Avinierdc* asked about opinions on Nvidia's new **Nemotron model**, sparking a brief discussion. *Teknium* expressed a positive outlook and noted having tried it once on LMSYS chatbot arena.

- **Equivalents to ComfyUI for LLMs**: *Csshsh* sought a tool equivalent to ComfyUI for LLMs that allows playful interaction below the API layer. *Orabazes* suggested that a lot can be done with **ComfyUI** and recommended checking out the **AnyNode custom node** for running models locally.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2406.06581">Set-Based Prompting: Provably Solving the Language Model Order Dependency Problem</a>: The development of generative language models that can create long and coherent textual outputs via autoregression has lead to a proliferation of uses and a corresponding sweep of analyses as research...</li><li><a href="https://youtu.be/KLJ3EEo8aPU?si=D155UK9mgoWmqOn5&t=2942">CMU Advanced NLP 2024 (8): Fine-tuning and Instruction Tuning</a>: This lecture (by Graham Neubig) for CMU CS 11-711, Advanced NLP (Spring 2024) covers:* Multi-tasking* Fine-tuning and Instruction Tuning* Parameter Efficient...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1251282963715461132)** (29 messages🔥): 

- **Feature for multiplayer collaboration considered**: A member asked about the possibility of creating lobbies for collaborative creation with AI. Another member confirmed interest, stating, *"Yes that's something we really like the idea of - any forms of multiplayer/pvp/co-op"*.
  
- **Worldclient and WebSim are not connected**: There was confusion regarding the connection between Opus on WebSim and WorldSim. It was clarified that *"worldclient has no connection with websim"*.

- **WorldSim AI experiences more censorship**: A member noted, *“the world-sim ai has been censored a bit”*. Another explained that the increased censorship could be due to stricter measures by the model provider, Anthropic.

- **Continuation feature for AI responses in development**: Members discussed a bug where the AI's replies abruptly cut off. One highlighted an ongoing effort to fix this, *“yeah, I have a feature in the works to allow continuation”*.

- **Claude 3's vision support and cost considerations**: Members discussed integrating vision support in WorldSim, noting that Claude 3 already has this feature. They also debated the costs, with suggestions to use GPT4o for vision tasks and pass the information to Claude to optimize usage.
  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1251271189431844895)** (40 messages🔥): 

- **Mojo Manual on Functions Sparks Debate**: A discussion arose around the Mojo manual's explanation of `def` and `fn` functions, specifically whether `def` functions allow or require no type declarations. One participant proposed seven alternative phrasings to clarify the language, showing the nuances in English interpretation.

- **Mojo Typing Mechanisms Critiqued**: The conversation steered towards the permissiveness of type declarations in `def` functions. The consensus was that while `def` functions do not enforce type declarations, they do allow them, contrasting with `fn` functions which require explicit type declarations.

- **Mojo Community Event Announcement**: An announcement for the Mojo Community Meeting was made, stating it would include talks by Helehex on constraints and Valentin on Lightbug, followed by a discussion on Python interop by Jack. A link to join the meeting was provided. [Join the meeting](https://modul.ar/community-meeting-zoom)

- **Benchmark Comparison Shared**: A user shared the results of a 1-thread benchmark test comparing Python FastAPI, Mojo Lightbug, and Rust Actix. The results showed Mojo Lightbug performed better than Python FastAPI but lagged behind Rust Actix.

- **Concerns About Function Coloring Discussed**: Following the community meeting, a discussion about the potential runtime costs of eliminating function coloring led to a conversation about stackful vs stackless coroutines. The debate highlighted the trade-offs between runtime cost and language complexity. [Link to discussion on coroutines](https://langdev.stackexchange.com/questions/697/what-are-the-benefits-of-stackful-vs-stackless-coroutines)
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.modular.com/mojo/manual/functions">Functions | Modular Docs</a>: Introduction to Mojo `fn` and `def` functions.</li><li><a href="https://langdev.stackexchange.com/questions/697/what-are-the-benefits-of-stackful-vs-stackless-coroutines)">What are the benefits of stackful vs. stackless coroutines?</a>: Many languages that use async nowadays are implemented via stackless coroutines. That is, in a language like Python or Javascript, when you await an expression, that await returns to the caller, </li><li><a href="https://github.com/go-vgo/robotgo/issues/662">Inconsistent key detection bug · Issue #662 · go-vgo/robotgo</a>: Robotgo version (or commit ref): Go version:1.17 Gcc version: Operating system and bit:windows10 64bit Provide example code: robotgo.EventHook(hook.KeyDown, []string{&quot;command&quot;, &quot;m&quot;...</li><li><a href="https://modul.ar/community-meeting-zoom">Join our Cloud HD Video Meeting</a>: Zoom is the leader in modern enterprise video communications, with an easy, reliable cloud platform for video and audio conferencing, chat, and webinars across mobile, desktop, and room systems. Zoom ...
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/1251298302218211371)** (2 messages): 

- **Modular tweets new update**: [Modular shares a Tweet](https://twitter.com/Modular/status/1801734712026911084) with their community, keeping followers updated on their latest activities and announcements.
- **Another Modular announcement via Twitter**: [Modular posts another Tweet](https://twitter.com/Modular/status/1802781075841974414) to keep the community informed about their continuous advancements and upcoming events.
  

---


### **Modular (Mojo 🔥) ▷ #[✍︱blog](https://discord.com/channels/1087530497313357884/1098713717509730466/1252338615732932691)** (1 messages): 

- **Mojo 24.4 Release Announced**: Mojo has released version 24.4, boasting several significant core language and standard library enhancements. Readers are encouraged to read the full blog post [here](https://www.modular.com/blog/whats-new-in-mojo-24-4-improved-collections-new-traits-os-module-features-and-core-language-enhancements) for detailed insights and code examples.

**Link mentioned**: <a href="https://www.modular.com/blog/whats-new-in-mojo-24-4-improved-collections-new-traits-os-module-features-and-core-language-enhancements">Modular: What’s New in Mojo 24.4? Improved collections, new traits, os module features and core language enhancements</a>: We are building a next-generation AI developer platform for the world. Check out our latest post: What’s New in Mojo 24.4? Improved collections, new traits, os module features and core language enhanc...

  

---


### **Modular (Mojo 🔥) ▷ #[ai](https://discord.com/channels/1087530497313357884/1103420074372644916/1251921353817329676)** (2 messages): 

- **$1,000,000 Prize for True AGI Solution!**: A user shared a [YouTube video featuring Francois Chollet](https://www.youtube.com/watch?v=UakqL6Pj9xo) discussing why he believes LLMs won’t lead to AGI, along with a $1,000,000 ARC-AGI Prize for finding a true solution. Another user expressed skepticism, commenting that the prize amount felt like "lowballing."

**Link mentioned**: <a href="https://www.youtube.com/watch?v=UakqL6Pj9xo">Francois Chollet - LLMs won’t lead to AGI - $1,000,000 Prize to find true solution</a>: Here is my conversation with Francois Chollet and Mike Knoop on the $1 million ARC-AGI Prize they&#39;re launching today.I did a bunch of socratic grilling throu...

  

---


### **Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1251292886738407474)** (107 messages🔥🔥): 

- **Defining 2D Numpy Arrays with Mojo**: Users discussed the limitations of Mojo in passing nested lists to Python and shared workarounds using the `ast` module and `Python.import_module`. For example, one user suggested a function `ndarray` that converts a string representation of a nested list to a Numpy array, which is then returned as a Python object.
- **Differences Between `DTypePointer` and `Pointer[SomeDType]`**: Users highlighted that `DTypePointer` is preferable for SIMD operations, as it allows for efficient `simd_load` instructions. This was particularly helpful for a user who wanted to understand the performance implications of using each type.
- **VSCode Integration with Mojo**: A member asked how to include directories in VSCode for Mojo, and another provided a way to do so via `settings.json`. This helps VSCode analyze Mojo packages by adding `"mojo.lsp.includeDirs": [ "/Users/your-name/your-mojo-files" ]`.
- **Bug in Casting and Contextual Behavior**: A user reported a bug when casting unsigned integers using `int()` or `UInt32()`, experiencing different behavior between running the script and using the REPL. A GitHub [issue](https://github.com/modularml/mojo/issues/3065) was created to track this inconsistency.
- **CRC32 Table Calculation with Var vs. Alias**: A detailed discussion revealed an issue when using `alias` instead of `var` to initialize CRC32 tables, leading to different results due to casting behaviors. The minimal example showed that overflowing as if signed was occurring unexpectedly, prompting an investigation into the alias-specific behavior.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.modular.com/mojo/cli/repl">mojo repl | Modular Docs</a>: Launches the Mojo REPL.</li><li><a href="https://github.com/fnands/fnands.com/blob/main/blog/2024/mojo-crc-calc/super_minimal_bug.mojo">fnands.com/blog/2024/mojo-crc-calc/super_minimal_bug.mojo at main · fnands/fnands.com</a>: My personal blog. Contribute to fnands/fnands.com development by creating an account on GitHub.</li><li><a href="https://github.com/modularml/mojo/issues/1139">[BUG]: Issue on creating  numpy N-dimensional array · Issue #1139 · modularml/mojo</a>: Bug description Can&#39;t create multi dimensional numpy array, creating single dimension numpy array works well. Steps to reproduce Include relevant code snippet or link to code that did not work as ...</li><li><a href="https://github.com/modularml/mojo/issues/3065#issuecomment-2173609155">[BUG] Unsigned integer casting overflowing as if signed when using `int()` or `UInt32()` · Issue #3065 · modularml/mojo</a>: Bug description Migrating this here after a bit of discussion in Discord. It seems like casting to unsigned integers actually just casts to signed integers, but has different behaviour in different...</li><li><a href="https://github.com/modularml/mojo/issues/3065">[BUG] Unsigned integer casting overflowing as if signed when using `int()` or `UInt32()` · Issue #3065 · modularml/mojo</a>: Bug description Migrating this here after a bit of discussion in Discord. It seems like casting to unsigned integers actually just casts to signed integers, but has different behaviour in different...</li><li><a href="https://github.com/fnands/fnands.com/blob/main/blog/2024/mojo-crc-calc/crc32_alias.mojo">fnands.com/blog/2024/mojo-crc-calc/crc32_alias.mojo at main · fnands/fnands.com</a>: My personal blog. Contribute to fnands/fnands.com development by creating an account on GitHub.</li><li><a href="https://github.com/fnands/fnands.com/blob/main/blog/2024/mojo-crc-calc/minimal_bug.mojo">fnands.com/blog/2024/mojo-crc-calc/minimal_bug.mojo at main · fnands/fnands.com</a>: My personal blog. Contribute to fnands/fnands.com development by creating an account on GitHub.</li><li><a href="https://github.com/modularml/mojo/issues">Issues · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/fnands/fnands.com/blob/main/blog/2024/mojo-crc-calc/crc.mojo">fnands.com/blog/2024/mojo-crc-calc/crc.mojo at main · fnands/fnands.com</a>: My personal blog. Contribute to fnands/fnands.com development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[🏎engine](https://discord.com/channels/1087530497313357884/1212827673257316453/1251464270907047937)** (3 messages): 

- **TPU usage clarified**: A member explained that the only way to use **TPUs** is through *calling XLA via the PjRT API*. They provided a [link to the PjRT API documentation](https://github.com/openxla/xla/blob/main/xla/pjrt/c/pjrt_c_api.h) and the TPU plugin [libtpu.so](https://storage.googleapis.com/cloud-tpu-tpuvm-artifacts/libtpu/2023-09-12/libtpu.so).
- **Call for native TPU support**: Another member suggested writing native support for **TPUs**, similar to how **Modular** handles GPUs. The first member responded that there's no public API for TPUs that operates at a lower level than XLA.

**Link mentioned**: <a href="https://github.com/openxla/xla/blob/main">GitHub - openxla/xla: A machine learning compiler for GPUs, CPUs, and ML accelerators</a>: A machine learning compiler for GPUs, CPUs, and ML accelerators - openxla/xla

  

---


### **Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1251363523779625001)** (9 messages🔥): 

- **New Nightly Mojo Compiler Released**: A new nightly Mojo compiler version `2024.6.1505` was released, and users can update with `modular update nightly/mojo`. For more details, see the [raw diff](https://github.com/modularml/mojo/compare/1130fdb81d763066d8e5bcb2226fe270981d3b0a...0508c6ee7fc8e1e7b42eda057ce9175a32c970cc) and the [current changelog](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md).
- **Compiler Version `2024.6.1605` Released**: Another nightly update to Mojo compiler, version `2024.6.1605`, has been released. Users should update and review changes through the [raw diff](https://github.com/modularml/mojo/compare/0508c6ee7fc8e1e7b42eda057ce9175a32c970cc...4df58b15f6ff4c613ecd4de62bde206a248d4652) and the [changelog](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md).
- **Latest Nightly Release `2024.6.1705`**: The most recent update to the nightly Mojo compiler is now available as version `2024.6.1705`. Update details can be reviewed via the [raw diff](https://github.com/modularml/mojo/compare/4df58b15f6ff4c613ecd4de62bde206a248d4652...87266e71c6dd29eca48511f2c8de492783be783a) and the [current changelog](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md).
- **Request for Builtin MLIR Dialects Documentation**: A user inquired about the availability of external documentation for builtin MLIR dialects. Another member confirmed that no such documentation is currently available.
- **Feature Request for REPL Improvements**: A query was made regarding whether expressions could directly output values in the REPL similar to Python. The response suggested filing a feature request on GitHub for this enhancement.
  

---



### **Eleuther ▷ #[announcements](https://discord.com/channels/729741769192767510/794042109048651818/1251345207526690927)** (1 messages): 

- **Interpretability team replicates OpenAI's findings**: The EleutherAI interpretability team successfully replicated OpenAI's "weak-to-strong" generalization results using open-source LLMs. They observed these results across 21 NLP datasets and tried several modifications to improve generalization but found that *"vanilla weak-to-strong training may already be close to eliciting everything the student 'knows'"*.

- **Negative results for generalization improvements**: The team experimented with various modifications such as strong-to-strong training, modified loss functions, and several probe-based experiments, with *"generally negative results"*. Among these, only the log-confidence auxiliary loss showed potential signs of consistent improvement in generalization.

- **Detailed findings published**: The detailed findings and results of their investigations on weak-to-strong generalization in open-source models like Qwen1.5 0.5B and Llama 3 8B can be found in their latest [blog post](https://blog.eleuther.ai/weak-to-strong/).

**Link mentioned**: <a href="https://blog.eleuther.ai/weak-to-strong/">Experiments in Weak-to-Strong Generalization</a>: Writing up results from a recent project

  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1251302336500990173)** (51 messages🔥): 

- **AISI adds new roles and assistance for moving**: AISI announced a variety of new job openings on their [careers page](https://aisi.gov.uk/careers) and mentioned they could assist with visas for candidates open to relocating to the UK. This sparked interest among members not residing in the UK, considering the opportunity despite the location requirement.
  
- **Discussion on CommonCrawl processing**: Members exchanged tips for processing CommonCrawl snapshots, highlighting tools like [ccget](https://github.com/allenai/ccget) and [resiliparse](https://resiliparse.chatnoir.eu/en/latest/man/parse/html.html). Challenges included throttling and performance optimizations for handling large datasets efficiently.

- **Interest in reproducible image generation models**: Users discussed image generation models trained on publicly licensed data, specifically pointing to the CommonCanvas models on [Hugging Face](https://huggingface.co/common-canvas) and a related [arXiv paper](https://arxiv.org/abs/2310.16825). While some found the models currently less effective, they suggested their potential use in creating applications like texture generation.

- **Clarification of Git vs. GitHub confusion**: Members clarified the differences between Git and GitHub, emphasizing that Git is a source code management tool and GitHub is a repository hosting service. The conversation included a [video link](https://www.theserverside.com/video/Git-vs-GitHub-What-is-the-difference-between-them) to help explain these concepts further.

- **Introduction of new members**: New members such as Piyush Ranjan Maharana and Tomer shared their backgrounds, including work in computational physics, autonomous cars, and material discovery via LLMs. They expressed eagerness to learn and contribute to the community.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/common-canvas">common-canvas (CommonCanvas)</a>: no description found</li><li><a href="https://github.com/allenai/ccget">GitHub - allenai/ccget: Tools for an internal archive of some Common Crawl files</a>: Tools for an internal archive of some Common Crawl files - allenai/ccget</li><li><a href="https://aisi.gov.uk/careers">Careers | AISI</a>: View career opportunities at AISI. The AI Safety Institute is a directorate of the Department of Science, Innovation, and Technology that facilitates rigorous research to enable advanced AI governance...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1251267140460351631)** (61 messages🔥🔥): 

- **Exploring RWKV-CLIP for Vision-Language Learning**: A paper discussed the introduction of **RWKV-CLIP**, a vision-language representation learning model combining transformers' parallel training with RNNs' efficient inference. This approach aims to improve large-scale image-text data quality by leveraging **LLMs** to synthesize and refine web-based texts and synthetic captions.
  
- **Concerns Around Diffusion Model Hallucinations**: Another paper explored the phenomenon of "hallucinations" in diffusion models, identifying a failure mode termed mode interpolation. The study revealed that diffusion models interpolate between data modes, creating artifacts not present in the original training distribution.

- **Discussion on Prefetching Streaming Datasets**: Some technical discussions touched on handling **streaming** datasets with `keep_in_memory=True` for efficient data fetching. Members shared insights about the recent introduction of checkpointing and resuming streams, enhancing usability for large datasets.

- **Effectiveness of Laprop Optimizer**: Members debated the effectiveness of the **Laprop** optimizer, with mixed results showing indifferent or inferior performance compared to **AdamW**. Parameter tweaks made some improvements, yet Laprop's overall performance remained underwhelming.

- **Stealing Commercial Embedding Models**: A paper highlighted a method for "stealing" commercial embedding models by training local models with text-embedding pairs obtained from APIs. The method showed that effective replication could be achieved inexpensively, raising concerns about the security of commercial models.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2406.09355">Can&#39;t Hide Behind the API: Stealing Black-Box Commercial Embedding Models</a>: Embedding models that generate representation vectors from natural language text are widely used, reflect substantial investments, and carry significant commercial value. Companies such as OpenAI and ...</li><li><a href="https://arxiv.org/abs/2406.05587">Creativity Has Left the Chat: The Price of Debiasing Language Models</a>: Large Language Models (LLMs) have revolutionized natural language processing but can exhibit biases and may generate toxic content. While alignment techniques like Reinforcement Learning from Human Fe...</li><li><a href="https://arxiv.org/abs/2405.11597">Language Reconstruction with Brain Predictive Coding from fMRI Data</a>: Many recent studies have shown that the perception of speech can be decoded from brain signals and subsequently reconstructed as continuous language. However, there is a lack of neurological basis for...</li><li><a href="https://arxiv.org/abs/2406.09358v1">Understanding Hallucinations in Diffusion Models through Mode Interpolation</a>: Colloquially speaking, image generation models based upon diffusion processes are frequently said to exhibit &#34;hallucinations,&#34; samples that could never occur in the training data. But where do...</li><li><a href="https://neuralblog.github.io/logit-prisms/">Logit Prisms: Decomposing Transformer Outputs for Mechanistic Interpretability</a>: no description found</li><li><a href="https://arxiv.org/abs/2405.19296">Neural Isometries: Taming Transformations for Equivariant ML</a>: Real-world geometry and 3D vision tasks are replete with challenging symmetries that defy tractable analytical expression. In this paper, we introduce Neural Isometries, an autoencoder framework which...</li><li><a href="https://arxiv.org/abs/2406.06973">RWKV-CLIP: A Robust Vision-Language Representation Learner</a>: Contrastive Language-Image Pre-training (CLIP) has significantly improved performance in various vision-language tasks by expanding the dataset with image-text pairs obtained from websites. This paper...</li><li><a href="https://huggingface.co/datasets/codeparrot/github-code-clean">codeparrot/github-code-clean · Datasets at Hugging Face</a>: no description found</li><li><a href="https://fixupx.com/ArmenAgha/status/1780149168692158658">Tweet from Armen Aghajanyan (@ArmenAgha)</a>: Final Update: One more magnitude of testing Sophia. We&#39;re talking model sizes in the B&#39;s, tokens in the T&#39;s. Sophia once again wins out. For me at least this is clear evidence that Sophia ...</li><li><a href="https://research.google/pubs/learned-optimizers-that-scale-and-generalize/">Learned Optimizers that Scale and Generalize</a>: no description found</li><li><a href="https://arxiv.org/abs/2209.11208">A Closer Look at Learned Optimization: Stability, Robustness, and Inductive Biases</a>: Learned optimizers -- neural networks that are trained to act as optimizers -- have the potential to dramatically accelerate training of machine learning models. However, even when meta-trained across...</li><li><a href="https://arxiv.org/abs/2406.07496">TextGrad: Automatic &#34;Differentiation&#34; via Text</a>: AI is undergoing a paradigm shift, with breakthroughs achieved by systems orchestrating multiple large language models (LLMs) and other complex components. As a result, developing principled and autom...</li><li><a href="https://github.com/zou-group/textgrad">GitHub - zou-group/textgrad: Automatic &#39;&#39;Differentiation&#39;&#39; via Text -- using large language models to backpropagate textual gradients.</a>: Automatic &#39;&#39;Differentiation&#39;&#39; via Text -- using large language models to backpropagate textual gradients. - zou-group/textgrad</li><li><a href="https://github.com/CFGpp-diffusion/CFGpp">GitHub - CFGpp-diffusion/CFGpp: Official repository for &quot;CFG++: manifold-constrained classifier free guidance for diffusion models&quot;</a>: Official repository for &quot;CFG++: manifold-constrained classifier free guidance for diffusion models&quot; - CFGpp-diffusion/CFGpp</li><li><a href="https://arxiv.org/abs/2406.08070">CFG++: Manifold-constrained Classifier Free Guidance for Diffusion Models</a>: Classifier-free guidance (CFG) is a fundamental tool in modern diffusion models for text-guided generation. Although effective, CFG has notable drawbacks. For instance, DDIM with CFG lacks invertibili...</li><li><a href="https://github.com/apple/ml-agm">GitHub - apple/ml-agm</a>: Contribute to apple/ml-agm development by creating an account on GitHub.</li><li><a href="https://openreview.net/forum?id=tUtGjQEDd4">Generative Modeling with Phase Stochastic Bridge</a>: Diffusion models (DMs) represent state-of-the-art generative models for continuous inputs. DMs work by constructing a Stochastic Differential Equation (SDE) in the input space (ie, position space),...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1251623429786964060)** (18 messages🔥): 

- **Hypernetwork-based Paper Critique**: A member dismissed a paper proposing **linear hypernetwork attention** as "useless," claiming it contains a critical mistake making its efficiency worse than full attention. They highlighted that the paper provides some reasoning for attention mechanisms behaving like hypernetworks.

- **Hypernetworks and Hopfield Nets Debate**: Members discussed whether hypernetworks are actually Hopfield nets, with one member noting that although there are high-level similarities like input-dependent weight generation, Hopfield networks are inherently recurrent. This sparked a conversation on the historical significance and evolution of Hopfield networks.

- **Hopfield Networks' Historical Context**: Members reminisced about Hopfield networks' past significance in connectionism and their influence on current models like transformers. They pointed out that modern models use backpropagation and multi-layer networks for superior performance, but the concepts of attractors and dynamics from Hopfield nets still inform contemporary neural network architecture.

- **Dynamic Evaluation and Online Adaptation**: A member shared a paper on **dynamic evaluation** for language models, emphasizing its utility in adapting to distributional shifts at test time. This method is described as turning parameters into temporally changing states, much like memory in neuroscience, and warrants a potential Jones-style scaling law evaluation.

- **Jones-style Scaling Law Reference**: In response to the dynamic evaluation discussion, a member referenced the "Scaling scaling laws with board games" paper by Andy L. Jones, which suggests trading off training compute and inference compute. This reference underscores the relevance of considering efficient scaling laws in adaptive model contexts.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2403.01518#deepmind">Revisiting Dynamic Evaluation: Online Adaptation for Large Language Models</a>: We consider the problem of online fine tuning the parameters of a language model at test time, also known as dynamic evaluation. While it is generally known that this approach improves the overall pre...</li><li><a href="https://arxiv.org/abs/2405.08707">Beyond Scaling Laws: Understanding Transformer Performance with Associative Memory</a>: Increasing the size of a Transformer model does not always lead to enhanced performance. This phenomenon cannot be explained by the empirical scaling laws. Furthermore, improved generalization ability...</li><li><a href="https://x.com/ibab/status/1669579636563656705)?">Tweet from Igor Babuschkin (@ibab)</a>: I keep revisiting this great paper from @andy_l_jones: “Scaling scaling laws with board games”. It shows how training compute and inference compute of MCTS can be traded off against each other. 10x mo...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1251612877870334117)** (11 messages🔥): 

- **Math PhD explores Sparse Autoencoders**: A new math PhD graduate expressed interest in interpretability research involving Sparse Autoencoders (SAEs). They were directed to [a blog post](https://www.lesswrong.com/posts/a5wwqza2cY3W7L9cj/sparse-autoencoders-find-composed-features-in-small-toy), which found that SAEs may recover composed features instead of ground truth ones in toy models.
  
- **Discussion on Sparse Coding and Dictionary Learning**: Members shared relevant papers and discussed topics related to sparse coding and dictionary learning, including a paper on dictionary learning in Wasserstein space [here](https://arxiv.org/pdf/2405.00837) and another on disentanglement in naturalistic videos [here](https://arxiv.org/abs/2007.10930).

- **Framework for Evaluating Feature Dictionaries**: A paper was introduced which proposes a framework for evaluating feature dictionaries in specific tasks using supervised dictionaries, highlighting its application on the indirect object identification task using GPT-2 Small ([link to paper](https://arxiv.org/abs/2405.08366)).

- **Link to Linear Identifiability Work**: Inquiries about settings with genuinely linear features in activation space led to recommendations for investigating linear probes and ICA literature, with a relevant paper [here](https://arxiv.org/abs/2311.03658).

- **Announcement of Logit Prisms Tool**: New work extending the logit lens method was announced as "logit prisms," decomposing logit output into components of the residual stream, attention layers, and MLP layers. It was used to study the gemma-2b model, revealing that digits 0-9 are encoded in a heart-like shape in a 2D space ([full article](https://neuralblog.github.io/logit-prisms)).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://neuralblog.github.io/logit-prisms">Logit Prisms: Decomposing Transformer Outputs for Mechanistic Interpretability</a>: no description found</li><li><a href="https://arxiv.org/abs/2007.10930">Towards Nonlinear Disentanglement in Natural Data with Temporal Sparse Coding</a>: We construct an unsupervised learning model that achieves nonlinear disentanglement of underlying factors of variation in naturalistic videos. Previous work suggests that representations can be disent...</li><li><a href="https://arxiv.org/abs/2405.08366">Towards Principled Evaluations of Sparse Autoencoders for Interpretability and Control</a>: Disentangling model activations into meaningful features is a central problem in interpretability. However, the absence of ground-truth for these features in realistic scenarios makes validating recen...</li><li><a href="https://www.lesswrong.com/posts/a5wwqza2cY3W7L9cj/sparse-autoencoders-find-composed-features-in-small-toy">Sparse autoencoders find composed features in small toy models  — LessWrong</a>: Summary  * Context: Sparse Autoencoders (SAEs) reveal interpretable features in the activation spaces of language models. They achieve sparse, interp…
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1251827262609952830)** (4 messages): 

- **Call for sharing evaluation results**: A member pointed out that Hugging Face is using an outdated harness version and observed significant differences in current results. They inquired about a platform where people could post their own evaluation results including runtime parameters and version information for validation.
- **Independent validation request for closed-source models**: The same member also asked if there was a place to post independent validation results for various closed-source models. This suggests a need for a shared, trustworthy evaluation forum.
- **Multi-GPU evaluation issue with WANDB**: Another member reported an issue when executing multi-GPU evaluation, leading to the creation of two separate projects in WANDB instead of one. They shared their command setup and sought advice on whether using the `--num_processes=2` flag for data parallel evaluation is appropriate.
  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1251841463902273626)** (3 messages): 

- **Code Release Inquiry Leads to GitHub Issues**: A member inquired about the release date for a particular code. Another member redirected the query to the project's [GitHub Issues page](https://github.com/deepglint/RWKV-CLIP/issues) for the **RWKV-CLIP** project.

**Link mentioned**: <a href="https://github.com/deepglint/RWKV-CLIP/issues">Issues · deepglint/RWKV-CLIP</a>: The official code of &quot;RWKV-CLIP: A Robust Vision-Language Representation Learner&quot; - Issues · deepglint/RWKV-CLIP

  

---



### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1251313549448577045)** (35 messages🔥): 

- **Apple's AI strategy at WWDC intrigues**: A community member shared a blog post [detailing Apple's new AI strategy](https://blog.trailofbits.com/2024/06/14/understanding-apples-on-device-and-server-foundations-model-release/), highlighting Apple's avoidance of NVIDIA hardware and CUDA APIs. It discusses the use of Apple’s AXLearn, which runs on TPUs and Apple Silicon.

- **Deep dive into embeddings resources**: A list of valuable resources on embeddings was shared, including a link to a [curated list on GitHub](https://github.com/eifuentes/awesome-embeddings) and a blog post at [vickiboykis.com](https://vickiboykis.com/what_are_embeddings/). Members discussed the importance of understanding latent spaces and how embeddings emerge.

- **Open call for refusal classifier models**: A member expressed interest in off-the-shelf refusal classifier models, possibly using T5/BERT for multilingual data. They indicated a need for around 1K samples for training and sought advice on this topic.

- **Fine-tuning TinyLlama for specific narration style**: A member documented their experience with fine-tuning TinyLlama to generate David Attenborough-style narration, sharing their [blog post](https://gabrielchua.me/posts/finetuning-tinyllama-axolotl-beginner/). They utilized tools like Axolotl and Jarvis Labs for the project, learning and sharing detailed steps and insights.

- **Issue with loading model config on Jarvis Labs**: A user faced an error while trying to fine-tune Mistral on Jarvis, which was resolved after switching to version v0.3 and changing the permissions of their token. They noted this might have also needed network stability, thanking others for their assistance.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co'">no title found</a>: no description found</li><li><a href="https://youtube.com/@hamelhusain7140?si=g_QdOF2ns0NKoUNK">Hamel Husain</a>: no description found</li><li><a href="https://vickiboykis.com/what_are_embeddings/">What are embeddings?</a>: A deep-dive into machine learning embeddings. </li><li><a href="https://parlance-labs.com/education/index.html">Educational Resources – Parlance</a>: no description found</li><li><a href="https://gabrielchua.me/posts/finetuning-tinyllama-axolotl-beginner/">Gabriel Chua - Fine-tuning TinyLlama with Axolotl and JarvisLab</a>: no description found</li><li><a href="https://blog.trailofbits.com/2024/06/14/understanding-apples-on-device-and-server-foundations-model-release/">Understanding Apple’s On-Device and Server Foundation Models release</a>: By Artem Dinaburg Earlier this week, at Apple’s WWDC, we finally witnessed Apple’s AI strategy. The videos and live demos were accompanied by two long-form releases: Apple’s Private Cloud Compute a…</li><li><a href="https://x.com/gabrielchua_/status/1802371411526537254">Tweet from gabriel (@gabrielchua_)</a>: fine-tuning with @axolotl_ai  and @jarvislabsai   as part of @HamelHusain & @dan_s_becker &#39;s llm f̴i̴n̴e̴-̴t̴u̴n̴i̴n̴g̴ ̴c̴o̴u̴r̴s̴e̴  conference, i did up a toy example to generate david attenbor...</li><li><a href="https://github.com/eifuentes/awesome-embeddings">GitHub - eifuentes/awesome-embeddings: 🪁A curated list of awesome resources around entity embeddings</a>: 🪁A curated list of awesome resources around entity embeddings - eifuentes/awesome-embeddings</li><li><a href="https://tenor.com/KhqP.gif">It Crowd Hello It GIF - It Crowd Hello IT Have You Tried Turning It Off And On Again - Discover &amp; Share GIFs</a>: Click to view the GIF
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[🟩-modal](https://discord.com/channels/1238365980128706560/1241044231829848125/1251598325954641970)** (14 messages🔥): 

- **Credit confusion and resolution**: A user realized they missed the deadline for additional credits and asked for help, receiving a positive response with the [GitHub link](https://github.com/setegonz/). Another user asked about the status of their account, and their credits were granted after a manual review.
- **Discussion on model startup optimization**: A user inquired whether copying model weights into the image or mounting them from a volume affects startup times. They were informed that weights loaded into images might have a slight edge, but infrastructure unification means differences are minor.
- **Multi-turn conversation issue and solution**: A user experienced an issue with their model predicting the first turn of conversation repeatedly and was advised to discuss it in the appropriate channel. They later resolved it by changing the dataset format to the [input_output format](https://openaccess-ai-collective.github.io/axolotl/docs/input_output.html) of Axolotl.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openaccess-ai-collective.github.io/axolotl/docs/input_output.html">Axolotl - Template-free prompt construction</a>: no description found</li><li><a href="https://github.com/setegonz/">setegonz - Overview</a>: setegonz has 10 repositories available. Follow their code on GitHub.</li><li><a href="https://github.com/modal-labs/llm-finetuning/blob/main/nbs/inspect_data.ipynb">llm-finetuning/nbs/inspect_data.ipynb at main · modal-labs/llm-finetuning</a>: Guide for fine-tuning Llama/Mistral/CodeLlama models and more - modal-labs/llm-finetuning
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[learning-resources](https://discord.com/channels/1238365980128706560/1241089743933149204/1251983215992897557)** (5 messages): 

- **Learn TextGrad for prompt fine-tuning**: Members discussed the [TextGrad project](https://github.com/zou-group/textgrad), which uses large language models to backpropagate textual gradients. It was noted that the project is considered *better than DSPy* and there is an explanatory [YouTube video](https://www.youtube.com/watch?v=Qks4UEsRwl0).

- **Using TextGrad without installation**: One member inquired if they could use TextGrad with their Anthropic/OpenAI API keys without installing anything. Another member mentioned that they tried the example Colab notebooks where one can set their OpenAI API key and test how it works.

- **Implementing LLMs from scratch**: A link to a GitHub repository was shared, providing a step-by-step guide for [implementing a ChatGPT-like LLM in PyTorch](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/01_main-chapter-code/ch07.ipynb). This resource could be useful for those interested in learning and experimenting with LLM development from the ground up.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/01_main-chapter-code/ch07.ipynb">LLMs-from-scratch/ch07/01_main-chapter-code/ch07.ipynb at main · rasbt/LLMs-from-scratch</a>: Implementing a ChatGPT-like LLM in PyTorch from scratch, step by step - rasbt/LLMs-from-scratch</li><li><a href="https://github.com/zou-group/textgrad">GitHub - zou-group/textgrad: Automatic &#39;&#39;Differentiation&#39;&#39; via Text -- using large language models to backpropagate textual gradients.</a>: Automatic &#39;&#39;Differentiation&#39;&#39; via Text -- using large language models to backpropagate textual gradients. - zou-group/textgrad</li><li><a href="https://www.youtube.com/watch?v=Qks4UEsRwl0">NEW TextGrad by Stanford: Better than DSPy</a>: In this TEXTGRAD framework, each AI system is transformed into a computation graph, where variables are inputs and outputs of complex (not necessarily differ...
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[hugging-face](https://discord.com/channels/1238365980128706560/1241141471814488115/1251582844002766940)** (2 messages): 

- **Reminder about form deadline**: *Gentle reminder folks that today is the last day to sign the form! If you **have not gotten credits yet, but yet think you filled out the first form, FILL THIS ONE OUT!***
- **Credits issuance for second form submissions**: If you applied on the second form, *"we haven't done credits for those yet, that happens after Monday."* There was a mention of difficulty in finding some users in the original form.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[replicate](https://discord.com/channels/1238365980128706560/1241163904927666287/1251259654701584466)** (2 messages): 

- **User Follows Up on Credit Link**: A member expressed concern about not receiving a link to redeem credits for Replicate. They mentioned having already sent their email and other details via DM.

- **LoRA Adapter Deployment Query**: A member sought assistance on deploying **LoRA adapters** to **Replicate**, mentioning success with running a fine-tuned phi-3-mini locally using **Cog**. They contrasted the process with **Modal**, where a volume is created and bound to a container at runtime, and asked how a similar approach could be achieved on Replicate.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[langsmith](https://discord.com/channels/1238365980128706560/1241167367040405544/1251275355646332990)** (5 messages): 

- **Clarification on LangSmith Beta Credits vs Course Credits**: A user asked if "LangSmith Beta Credit" are the same as the credits for the course. Another user clarified that they are different; "LangSmith Beta Credit" was granted to beta users, while course credits should appear as 'Mastering LLMs Course Credit' under billing.
- **Offering Help with Missing Credits**: One user offered assistance to another user who felt they were missing course credits. They confirmed that they could check the situation if provided with the email used in the credits form.
- **User Queries about Missing Credits**: Another user inquired about not seeing any credits on LangSmith. They requested help to understand if any additional steps were needed from their end.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[berryman_prompt_workshop](https://discord.com/channels/1238365980128706560/1242223275463938221/1252307321808486542)** (2 messages): 

- **Promptfoo gains interest among members**: A member expressed interest in **Promptfoo**, thanking another for sharing it. 
- **Inspect-ai preferred over Promptfoo**: Another member shared their preference for **inspect-ai** over Promptfoo, citing its flexibility and fit with Python in a test style. However, they mentioned it's not straightforward to do side-by-side comparisons with inspect-ai compared to Promptfoo.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[workshop-3](https://discord.com/channels/1238365980128706560/1242223458184597534/1252141854133063772)** (4 messages): 

- **CUDA Error during Docker Execution**: A user experienced a `Docker` error when running Python in a container, with the message "OCI runtime create failed: runc create failed: unable to start container process". Another user suggested that this might be due to an improperly set up **CUDA** or a compatibility issue.
- **Difficulty in Issue Replication**: The issue is hard to replicate, as noted by a responder who stated, "*It’s hard to tell because I can’t replicate this issue*". This indicates the problem might be environment-specific or related to the user's specific configuration.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[clavie_beyond_ragbasics](https://discord.com/channels/1238365980128706560/1242223963346698250/1251277734798430371)** (8 messages🔥): 

- **RAGatouille Simplifies ColBERT Usage**: A member praised [RAGatouille](https://github.com/bclavie/RAGatouille) as a great tool for integrating **ColBERT** with Langchain for internal projects. They also recommended [Ben's post](https://ben.clavie.eu/ragatouille/) as a fantastic introduction to ColBERT.
- **Understanding Bi-encoder Functionality in RAG**: Addressing a beginner’s query about the logic behind bi-encoders in RAG setups, another member explained that models are trained to associate queries and documents with a prefix system. The response highlighted the necessity of defining "similarity" during model training to suit different use cases.
- **Exploring Learning Resources**: A member sought resources for advanced topics like finetuning **ColBERT** and rerankers, and using embedding adapters. They appreciated another member's recommendation of a [Medium post on building state-of-the-art text embedding models](https://medium.com/snowflake/how-to-build-a-state-of-the-art-text-embedding-model-a8cd0c86a19e).
- **Combining Full Text Search with Fine-tuned Rerankers**: A participant discussed their approach of using **lancedb** and combining full-text search via Lucene with fine-tuned rerankers for impactful results. They noted not using vector databases as mentioned in Ben's presentation.

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://ben.clavie.eu/ragatouille/">RAGatouille</a>: no description found</li><li><a href="https://github.com/bclavie/RAGatouille">GitHub - bclavie/RAGatouille: Easily use and train state of the art late-interaction retrieval methods (ColBERT) in any RAG pipeline. Designed for modularity and ease-of-use, backed by research.</a>: Easily use and train state of the art late-interaction retrieval methods (ColBERT) in any RAG pipeline. Designed for modularity and ease-of-use, backed by research. - bclavie/RAGatouille
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[jason_improving_rag](https://discord.com/channels/1238365980128706560/1242224099548332132/1252219319392141352)** (1 messages): 

- **Efficient Category Structuring for GPT-4o**: Members discussed how using a tree structure for category prompts improves **GPT-4o**'s decision-making in filter selection. Despite the large system prompt, it works well even though **latency** was an issue with **GPT-4**. 

- **Single Vector Strategy for Documents**: The group uses just one vector per document/product, accompanied by appropriate meta tags. This approach aids in maintaining a streamlined and effective categorization system.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[jeremy_python_llms](https://discord.com/channels/1238365980128706560/1242224309548875917/1251957337724555486)** (3 messages): 

- **Catch up on talks with shared link**: A member requested a link to catch up on the discussions. **Jeremy Howard** promptly shared [this Discord link](https://discord.gg/3ruJE6vB), and the member expressed their gratitude.

**Link mentioned**: <a href="https://discord.gg/3ruJE6vB">Join the fast.ai Discord Server!</a>: Check out the fast.ai community on Discord - hang out with 10920 other members and enjoy free voice and text chat.

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[saroufimxu_slaying_ooms](https://discord.com/channels/1238365980128706560/1242224552415596554/1251416118623735859)** (3 messages): 

- **Anticipation Builds for New Session**: A user inquired about the likelihood of a new session taking place with a humorous undertone: *What is the probability that there will be a session? 🫢🤪*.

- **Upcoming Project in Memory Efficiency**: Another user informed everyone that a new project focused on **memory efficiency** is underway. They mentioned that once this project is ready, a "more interesting talk" can be expected.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[axolotl](https://discord.com/channels/1238365980128706560/1242542198008975430/1251280712477184094)** (27 messages🔥): 

- **Strickvl hits OOM errors with local LORA models**: Despite using two 4090s, Strickvl faces Out-Of-Memory (OOM) errors when loading full resultant LORA models. They suggested checking configurations and considering quantization, and shared their [configs](https://github.com/strickvl/isafpr_finetune/tree/main/configs) on GitHub.

- **Quantization offers a memory-saving solution**: Chrislevy pointed out that models loaded in float32 consume a lot of memory and recommended using `torch_dtype=torch.bfloat16` for inference, as described in the [Llama 3 model card](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct#transformers-automodelforcausallm).

- **Documentation gap for axolotl and finetuning**: There's a call for better documentation on finetuning, specifically on training LORA/QLORA settings, saving models, and proper loading techniques. Strickvl emphasized this need and hinted at using [Hamel’s course repo](https://github.com/parlance-labs/ftcourse/blob/master/06_sanity_check.ipynb) for sanity checks.

- **Modal Labs guide clarifies model loading**: Andrewcka provided code insights from [Modal Labs' inference script](https://github.com/modal-labs/llm-finetuning/blob/main/src/inference.py) explaining how the script identifies the last trained model by date-time to handle inference effectively.

- **Finetuning multi chat conversations with axolotl**: Huikang inquired about adapting axolotl for multi chat conversations and shared resources like the [code for CodeLlama](https://github.com/modal-labs/llm-finetuning/blob/main/config/codellama.yml) and the [axolotl dataset formats](https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/index.html) for conversation fine-tuning.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/index.html">Axolotl - Dataset Formats</a>: no description found</li><li><a href="https://openaccess-ai-collective.github.io/axolotl/docs/config.html">Axolotl - Config options</a>: no description found</li><li><a href="https://github.com/modal-labs/llm-finetuning/blob/main/src/inference.py">llm-finetuning/src/inference.py at main · modal-labs/llm-finetuning</a>: Guide for fine-tuning Llama/Mistral/CodeLlama models and more - modal-labs/llm-finetuning</li><li><a href="https://github.com/strickvl/isafpr_finetune/blob/main/notebooks/sanity_check.py#L21)">isafpr_finetune/notebooks/sanity_check.py at main · strickvl/isafpr_finetune</a>: Finetuning an LLM for structured data extraction from press releases - strickvl/isafpr_finetune</li><li><a href="https://huggingface.co/blog/optimize-llm">Optimizing your LLM in production</a>: no description found</li><li><a href="https://github.com/parlance-labs/ftcourse/blob/master/06_sanity_check.ipynb">ftcourse/06_sanity_check.ipynb at master · parlance-labs/ftcourse</a>: Contribute to parlance-labs/ftcourse development by creating an account on GitHub.</li><li><a href="https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct#transformers-automodelforcausallm).">meta-llama/Meta-Llama-3-8B-Instruct · Hugging Face</a>: no description found</li><li><a href="https://drchrislevy.github.io/posts/fine_tune_jarvis/fine_tune_jarvis.html#inference-with-the-model)">Chris Levy - Fine-Tuning LLMs with Axolotl on JarvisLabs</a>: no description found</li><li><a href="https://github.com/modal-labs/llm-finetuning/blob/main/config/codellama.yml">llm-finetuning/config/codellama.yml at main · modal-labs/llm-finetuning</a>: Guide for fine-tuning Llama/Mistral/CodeLlama models and more - modal-labs/llm-finetuning
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[wing-axolotl](https://discord.com/channels/1238365980128706560/1242564077151326388/1251781955188555856)** (1 messages): 

- **Excitement over Code Llama's release**: [Code Llama](https://huggingface.co/blog/codellama) is an evolution of Llama 2, tuned specifically for code tasks, and released in the Hugging Face ecosystem. The release includes models on the Hub, Transformers integration, and several productivity-boosting features for software engineers.
- **Format difference spotted**: Noting a format difference between the Hugging Face blog post about Code Llama and the [GitHub configuration file](https://github.com/modal-labs/llm-finetuning/blob/main/config/codellama.yml) for finetuning Code Llama models. This was highlighted to confirm if such differences are acceptable.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/blog/codellama#conversational-instructions">Code Llama: Llama 2 learns to code</a>: no description found</li><li><a href="https://github.com/modal-labs/llm-finetuning/blob/main/config/codellama.yml">llm-finetuning/config/codellama.yml at main · modal-labs/llm-finetuning</a>: Guide for fine-tuning Llama/Mistral/CodeLlama models and more - modal-labs/llm-finetuning
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[charles-modal](https://discord.com/channels/1238365980128706560/1242564177952768062/1251273939586388099)** (1 messages): 

- **Channel Lockdown Notice**: The channel is being locked down, and members are directed to use [another channel](<#1241044231829848125>) for any questions for Charles. A friendly emoji, <:hugging_angel:936261297182482452>, was included in the announcement.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[simon_cli_llms](https://discord.com/channels/1238365980128706560/1242664474276659320/1251645300976652468)** (5 messages): 

- **CORS Error blocks video fetch**: A member reported encountering **CORS errors** when trying to fetch a video. The suggested workaround is to "open the raw .mp4".
- **CloudFront misconfiguration suspected**: The issue may stem from a **CloudFront misconfiguration** where the request's CORS headers aren't being cached properly. The member noted that "CloudFront will cache the whole response on the first time the URL is hit" and "their cache does not key on the fetch mode request headers".
- **Video link provided**: The video in question is accessible at [this link](https://d33j6849mtsdhc.cloudfront.net/9364/10130/92796/81552506420/video_1.mp4). The member queried whether it was "recorded from outside zoom and shared via a bucket".

**Link mentioned**: <a href="https://d33j6849mtsdhc.cloudfront.net/9364/10130/92796/81552506420/video_1.mp4">no title found</a>: no description found

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[allaire_inspect_ai](https://discord.com/channels/1238365980128706560/1242943547699888229/1251953752156409957)** (3 messages): 

- **Using instructor with inspect_ai**: A member asked if there was a way to use something like instructor in inspect_ai to ensure the output format is valid. Another member suggested either implementing and registering a custom model or using tool calls directly, as this is what instructor does under the hood.
- **flexibility of inspect_ai**: One user noted that inspect_ai allows for replacing existing infrastructure with custom solutions or enhancing current setups. 


  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[credits-questions](https://discord.com/channels/1238365980128706560/1243721538432270388/1251529402097991710)** (3 messages): 

- **Credit Check Confusion on Braintrust Data**: A user expressed frustration about not finding where to check credits on the **Braintrust Data site**: *"I can not even find where to check credits on braintrustdata site. It does not show anything to billing at all?"* Another user suggested seeking help in another channel, emphasizing they also couldn't find the credit status.
- **Redirect to Proper Channel for Solutions**: A member recommended moving the discussion to a different channel, tagging another user for a potential answer to the credits check issue. They acknowledged similar difficulties in locating the current credit status.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[fireworks](https://discord.com/channels/1238365980128706560/1245126291276038278/1251463494700761202)** (6 messages): 

- **Users Swamp Support for Credit Issues**: Multiple users requested assistance with missing credits on their accounts. User account IDs mentioned include *carljvh-7d2eb0*, *jalonso-e11d20*, *alex-kira-d15187*, *harrille-postia-723075*, *ayhanfuat-fa2dd5*, and *data-94d7ef*.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[braintrust](https://discord.com/channels/1238365980128706560/1245407617031999581/1251530884520214562)** (3 messages): 

- **User seeks platform testing credits**: @peaky8linders asked about logging in to test a platform and still seeing the Upgrade button, querying if they could still get credits. They provided their email and organization information for verification.
- **Credits confirmed**: @ankrgyl assured @peaky8linders that they should be all set with the credits.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[west-coast-usa](https://discord.com/channels/1238365980128706560/1245410680065097738/)** (1 messages): 

.peterj: Anyone from Seattle area?
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[east-coast-usa](https://discord.com/channels/1238365980128706560/1245411101718610001/)** (1 messages): 

ssilby: <@415846459016216576> I'm in! Let's set up a DMV meetup :3
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[predibase](https://discord.com/channels/1238365980128706560/1245803791710687272/1251604430139166730)** (7 messages): 

- **Predibase Misinterprets Dataset Fields**: A user faced issues with their Alpaca/ShareGPT-formatted dataset on Predibase due to a missing `text` field. They were curious how to work with template-free datasets and convert their data accordingly.
- **Getting Data Format Right for Predibase**: The user resolved their issue by selecting the 'instruction tuning' format and adjusting the data as per Predibase's documentation. They shared their dataset for reference [here](https://github.com/strickvl/isafpr_finetune/tree/main/data).
- **Test Data Evaluation on Predibase**: The user noted a limitation of Predibase regarding the use of test data for evaluation and mentioned they would perform the evaluation after the model is trained.
- **Extracting Adapters from Predibase**: The user inquired if it is possible to download or extract the adapters trained on Predibase for local testing, preferring to avoid deploying a custom instance.

**Link mentioned**: <a href="https://github.com/strickvl/isafpr_finetune/tree/main/data">isafpr_finetune/data at main · strickvl/isafpr_finetune</a>: Finetuning an LLM for structured data extraction from press releases - strickvl/isafpr_finetune

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[openpipe](https://discord.com/channels/1238365980128706560/1245927847437008896/1251976933844193310)** (3 messages): 

- **Dataset Format Struggles Resolved**: A member asked for examples of datasets formatted correctly for Openpipe, mentioning unsuccessful attempts with axolotl and template-free datasets. Later, they solved their own problem by formatting the data according to the OpenAI chat format used for OpenAI finetuning, sharing their dataset on [GitHub](https://github.com/strickvl/isafpr_finetune/tree/main/data).

**Link mentioned**: <a href="https://github.com/strickvl/isafpr_finetune/tree/main/data">isafpr_finetune/data at main · strickvl/isafpr_finetune</a>: Finetuning an LLM for structured data extraction from press releases - strickvl/isafpr_finetune

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[openai](https://discord.com/channels/1238365980128706560/1245927985123692575/)** (1 messages): 

kramakurious: <@1010989949572612166> is this something you can help with?
  

---



### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1251253795866935318)** (69 messages🔥🔥): 

- **Sakana AI hits $1B valuation**: Sakana AI, a Japanese startup developing alternatives to transformer models, raised funds from NEA, Lux, and Khosla at a $1B valuation. For more details, check out [the link](https://www.theinformation.com/articles/openais-japanese-rival-gets-1-billion-valuation-from-silicon-valley-investors).
  
- **Runway's Gen-3 Alpha debuts**: Runway introduced Gen-3 Alpha, a new base model for [video generation](http://runwayml.com/gen-3-alpha). Claimed to create highly detailed videos with complex scene changes and a wide range of cinematic choices.

- **DeepSeek-Coder-V2 impresses**: DeepSeek-Coder-V2 was released, reportedly [beating GPT-4 on both HumanEval and MATH](https://vxtwitter.com/teortaxesTex/status/1802681431992213767) benchmarks.
  
- **Google DeepMind’s new video-to-audio tech**: Google DeepMind showcased progress on their video-to-audio (V2A) technology, capable of generating an "unlimited number" of tracks for any video. [See examples here](https://x.com/rowancheung/status/1802734770117333257).

- **Wayve's new view synthesis model**: Wayve released a new view synthesis model, impressively creating views from input images using 4D Gaussians, according to [Jon Barron's update](https://x.com/jon_barron/status/1802758455830437975).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/rowancheung/status/1802734770117333257">Tweet from Rowan Cheung (@rowancheung)</a>: Google DeepMind just shared progress on their new video-to-audio (V2A) tech  Until now, AI video generations have been silent, this solves that. V2A can generate an &#34;unlimited number&#34; of track...</li><li><a href="https://x.com/jon_barron/status/1802758455830437975">Tweet from Jon Barron (@jon_barron)</a>: Wayve dropped a new view synthesis model earlier today. I&#39;m guessing it&#39;s a radiance field made of 4D Gaussians. Nothing generative, just view synthesis from input images. Very impressive.</li><li><a href="https://x.com/runwayml/status/1802691475391566108">Tweet from Runway (@runwayml)</a>: Introducing Gen-3 Alpha: Runway’s new base model for video generation.  Gen-3 Alpha can create highly detailed videos with complex scene changes, a wide range of cinematic choices, and detailed art di...</li><li><a href="https://x.com/dwarkesh_sp/status/1802771055016378554">Tweet from Dwarkesh Patel (@dwarkesh_sp)</a>: I asked Buck about his thoughts on ARC-AGI to prepare for interviewing @fchollet.  He tells his coworker Ryan, and within 6 days they&#39;ve beat SOTA on ARC and are on the heels of average human perf...</li><li><a href="https://x.com/steph_palazzolo/status/1801690079922163954?s=46">Tweet from Stephanie Palazzolo (@steph_palazzolo)</a>: NEW w/ @nmasc_ @KateClarkTweets:  Sakana AI, a Japanese startup developing alternatives to transformer models, has raised from NEA, Lux and Khosla at a $1B valuation. More here:  https://www.theinform...</li><li><a href="https://x.com/natolambert/status/1802762956469579956">Tweet from Nathan Lambert (@natolambert)</a>: What unlocked all these text-to-video models being good within the same 6month window?   Was it just that people weren&#39;t trying? Wild that it seems like just coincidence for them to all emerge.  L...</li><li><a href="https://elevenlabs.io/sound-effects">AI Text to Sound Effects Generator</a>: Use our AI Sound Effects Generator to generate any sound imaginable from a text prompt for free. Perfect for videos, podcasts, or any other audio production.
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1251404893974036561)** (4 messages): 

- **Sam Altman hints at OpenAI governance changes**: A tweet by Jacques Thibault referenced a private statement by Sam Altman, suggesting OpenAI might convert to a for-profit business. This move could potentially enable a public offering, allowing Altman to gain a stake in OpenAI. [Read the full tweet](https://x.com/jacquesthibs/status/1801782465364640247?s=46).

- **The Information reports on OpenAI’s potential shift**: The Information detailed that Altman has privately mentioned OpenAI’s possible shift to a benefit corporation, similar to Anthropic and xAI. This transformation could lead to OpenAI going public. [Read the article here](https://www.theinformation.com/articles/openai-ceo-says-company-could-become-benefit-corporation-akin-to-rivals-anthropic-xai?utm_campaign=Editorial&utm_content=Article&utm_medium=organic_social&utm_source=twitter).

- **Community reacts skeptically**: One member expressed skepticism over these developments, summarizing their sentiment with "*This is so sketch lmao*".

**Link mentioned**: <a href="https://x.com/jacquesthibs/status/1801782465364640247?s=46">Tweet from Jacques (@JacquesThibs)</a>: &#34;Sam Altman recently told some shareholders that OAI is considering changing its governance structure to a for-profit business that OAI&#39;s nonprofit board doesn&#39;t control. [...] could open ...

  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1251252934457425920)** (63 messages🔥🔥): 

- **Compliments on Interconnects Merch**: Members discussed the quality of the merchandise, noting that while stickers were not well-received, the T-shirts were appreciated. One member mentioned, "stickers were bad need to try another vendor."
- **Dissecting ARC-AGI Performance**: A link to a [Redwood Research article](https://redwoodresearch.substack.com/p/getting-50-sota-on-arc-agi-with-gpt) discussing methods to improve ARC-AGI performance sparked debate. Members criticized the approach of using a large number of samples, arguing it's more about hitting by chance rather than scaling.
- **Exploring Neurosymbolic AI**: Members dove into neurosymbolic AI, questioning if leveraging LLMs for discrete program search truly fits the traditional definition. A discussion evolved around [a tweet from François Chollet](https://x.com/fchollet/status/1802773156341641480?s=46), parsing out whether current AI techniques suffice or if fundamental breakthroughs are necessary.
- **MidJourney's New Ventures**: MidJourney is expanding into hardware and anticipates launching training on its video models in January. CEO David Holz confirmed this during a Discord "Office Hour" session.
- **Conundrums at Academic Conferences**: A member pondered the value of attending ACL in Thailand despite the travel inconvenience from California, questioning its relevance compared to major conferences like NeurIPS. "I don't think it's do or die," another member responded, suggesting optional attendance.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://apparate.ai/">Apparate AI</a>: no description found</li><li><a href="https://x.com/fchollet/status/1802773156341641480?s=46">Tweet from François Chollet (@fchollet)</a>: @dwarkesh_sp This has been the most promising branch of approaches so far -- leveraging a LLM to help with discrete program search, by using the LLM as a way to sample programs or branching decisions....</li><li><a href="https://redwoodresearch.substack.com/p/getting-50-sota-on-arc-agi-with-gpt">Getting 50% (SoTA) on ARC-AGI with GPT-4o</a>: You can just draw more samples</li><li><a href="https://github.com/rgreenblatt/arc_draw_more_samples_pub/blob/0b36f4584aebae9ec876d3510842b3651e719d67/arc_solve/edit_distance.py#L115).">arc_draw_more_samples_pub/arc_solve/edit_distance.py at 0b36f4584aebae9ec876d3510842b3651e719d67 · rgreenblatt/arc_draw_more_samples_pub</a>: Draw more samples. Contribute to rgreenblatt/arc_draw_more_samples_pub development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1251286628757016788)** (9 messages🔥): 

- **RAG and Agents Guide Excites with Excalidraw Diagrams**: @_nerdai_ shared a comprehensive [slide deck on building RAG and Agents](https://t.co/oibSiDjseO). The guide includes full Excalidraw diagrams breaking down simple-to-advanced concepts.
- **Arize Integration Adds End-to-End Observability**: The new instrumentation module integrates with Arize, demonstrated in this [guide](https://t.co/cOBP9IOjro). It shows how to instrument custom event/span handlers in LLM apps.
- **AI World's Fair Wrap-up Featuring Top Speakers**: Join talks from @jerryjliu0, @freddie_v4, @atitaarora, and more at the [AI Sizzle and Waves event](https://t.co/O6WAkbI9jt) by AI Engineer World's Fair. Hosted by Angela Tse, Atita Arora, and Julia Neagu.
- **Beginner’s Guide for Full-Stack Agents Released**: @MervinPraison's tutorial offers a step-by-step [guide](https://t.co/BtP0iRrjBq) on building core components of an agent using local models and @chainlit_io. The tutorial is designed to create simple applications.
- **Multimodal RAG Pipeline with Claude 3 and SingleStoreDB**: @Pavan_Belagatti discusses future roles of multimodal RAG in his [article](https://t.co/u80hMB0ioz), which utilizes Claude 3 by @AnthropicAI and @SingleStoreDB. This pipeline addresses the prevalence of images within documents.

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://t.co/dhWHELRUBJ">AI Engineer World&#x27;s Fair Closer: AI Sizzle and Waves @ GitHub HQ · Luma</a>: Wrap up the Fair with an AI summer Friday. Enjoy refreshing treats and ride the waves of fresh takes from fellow developers and innovators. Hear from our…</li><li><a href="https://t.co/cOBP9IOjro">openinference/python/instrumentation/openinference-instrumentation-llama-index at main · Arize-ai/openinference</a>: Auto-Instrumentation for AI Observability. Contribute to Arize-ai/openinference development by creating an account on GitHub.</li><li><a href="https://t.co/m3YOjOF36q">Instrumentation: Basic Usage - LlamaIndex</a>: no description found
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1251269460233289789)** (95 messages🔥🔥): 

- **Chunking customer service emails for RAG**: One member asked how to create chunks for a customer service RAG model based on email conversations. Another suggested capturing the first email from each chain to ensure each email is included.
- **Generating specific outputs from markdown documents**: A user is having issues with LlamaIndex truncating relevant language from markdown documents. They need precise outputs without summarization and are looking for any advice to improve this.
- **Using Neo4j with LlamaIndex**: Multiple queries were raised about converting Neo4j knowledge graphs into LlamaIndex property graphs. Detailed instructions and a link to the LlamaIndex documentation were shared ([LlamaIndex Property Graph Example](https://docs.llamaindex.ai/en/latest/examples/property_graph/property_graph_neo4j/)).
- **Overlapping sentence retrieval**: A user inquired about expanded sentences overlapping when using sentence-level retrievals. It was clarified that overlapping sentences do not get merged, and custom post-processing would be needed.
- **Saving ChatMemoryBuffer**: There was a discussion on saving `ChatMemoryBuffer` objects to a file format to manage token limits in long conversations. A method to save chat memory as a dict and store it in a JSON file was suggested.

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lu.ma/kqxmbuou?">LlamaIndex Webinar: Advanced RAG with Knowledge Graphs (with Tomaz from Neo4j) · Zoom · Luma</a>: We’re hosting a special workshop on advanced knowledge graph RAG this Thursday 9am PT, with the one and only Tomaz Bratanic from Neo4j. In this webinar, you’ll…</li><li><a href="https://github.com/run-llama/llama_index/tree/02984efc5004126ccaffa15ec599d0dacce55dd3/llama-index-integrations/storage">llama_index/llama-index-integrations/storage at 02984efc5004126ccaffa15ec599d0dacce55dd3 · run-llama/llama_index</a>: LlamaIndex is a data framework for your LLM applications - run-llama/llama_index</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/property_graph/property_graph_neo4j/#loading-from-an-existing-graph>).">Neo4j Property Graph Index - LlamaIndex</a>: no description found</li><li><a href="https://github.com/run-llama/llama_index/blob/01e5173f8a272e8b7e5ccb2ae3ff215eb6c4ca6a/llama-index-core/llama_index/core/query_engine/knowledge_graph_query_engine.py#L70">llama_index/llama-index-core/llama_index/core/query_engine/knowledge_graph_query_engine.py at 01e5173f8a272e8b7e5ccb2ae3ff215eb6c4ca6a · run-llama/llama_index</a>: LlamaIndex is a data framework for your LLM applications - run-llama/llama_index</li><li><a href="https://github.com/run-llama/llama_index/blob/01e5173f8a272e8b7e5ccb2ae3ff215eb6c4ca6a/llama-index-core/llama_index/core/query_engine/knowledge_graph_query_engine.py#L132">llama_index/llama-index-core/llama_index/core/query_engine/knowledge_graph_query_engine.py at 01e5173f8a272e8b7e5ccb2ae3ff215eb6c4ca6a · run-llama/llama_index</a>: LlamaIndex is a data framework for your LLM applications - run-llama/llama_index</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/llm/monsterapi/#rag-approach-to-import-external-knowledge-into-llm-as-context>)">no title found</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/retrievers/relative_score_dist_fusion/#setup>)">Relative Score Fusion and Distribution-Based Score Fusion - LlamaIndex</a>: no description found</li><li><a href="https://direct.mit.edu/qss/article/2/4/1423/108045/A-framework-for-creating-knowledge-graphs-of">A framework for creating knowledge graphs of scientific software metadata</a>: Abstract. An increasing number of researchers rely on computational methods to generate or manipulate the results described in their scientific publications. Software created to this end—scientific so...
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1251880106566356993)** (6 messages): 

- **Power-Up LLMs with Web Scraping and RAG!**: [How to Power-Up LLMs with Web Scraping and RAG](https://medium.com/ai-advances/how-to-power-up-llms-with-web-scraping-and-rag-975a165587f6) explores enhancing LLM performance through web scraping and retrieval-augmented generation (RAG). The article highlights tools like Firecrawl for clean Markdown extraction and Scrapfly for various output formats.
- **Firecrawl vs. Scrapfly in LLM Applications**: *"Firecrawl shines for Markdown"*, making it ideal for preparing data for LLMs. Scrapfly offers flexibility with various output formats but may need additional processing for LLM optimization.
  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1251251986846847137)** (39 messages🔥): 

- **Script indentation breaks in autogen_stubs.sh**: A member faced issues with the `autogen_stubs.sh` script where `clang2py` breaks indentation, causing syntax errors. Discussions revealed it was not needed for the intended task of running tinygrad with GPU.
- **OpenCL installation issues cause errors**: Problems with OpenCL installation led to errors when running tinygrad on GPU. George Hotz suggested fixing the OpenCL setup and checking `clinfo` to troubleshoot.
- **Improving OpenCL error messages**: The community discussed enhancing OpenCL error messages by autogenerating them from OpenCL headers. A [pull request](https://github.com/tinygrad/tinygrad/pull/5004) was opened to implement better error messages.
- **Process replay documentation needed**: George Hotz requested adding documentation on process replay to assist new contributors. This was in response to simplifying the process of rewriting operations using new styles.
- **Monday meeting agenda topics**: Important topics include the tinybox launch, the 0.9.1 release, the CI benchmark duration, removing numpy, and various technical discussions. Highlights also include performance milestones like achieving 200 tok/s for llama 7B on multi-GPU setups.

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/__tinygrad__/status/1802442339069333984">Tweet from the tiny corp (@__tinygrad__)</a>: @PennJenks It&#39;s three kernels. We need to fuse it into 1.  GRAPH=1 python3 -c &#34;from tinygrad import Tensor; Tensor.rand(100,100).softmax().realize()&#34;</li><li><a href="https://x.com/__tinygrad__/status/1747467257889116379">Tweet from the tiny corp (@__tinygrad__)</a>: tinybox has a 1TB NVMe boot drive on USB 3, and 4 1TB NVMes each on 4 lanes of PCI-E 4.0; 4TB for holding weights and datasets.  No theory, that&#39;s a real benchmark. It&#39;s faster than the RAM on...</li><li><a href="https://github.com/tinygrad/tinygrad/pull/4899/files,">nv better error messages for ioctls by nimlgen · Pull Request #4899 · tinygrad/tinygrad</a>: no description found</li><li><a href="https://github.com/tinygrad/tinygrad/pull/5004">Fix/opencl Better error Messages by GabrielZCode · Pull Request #5004 · tinygrad/tinygrad</a>: Better openCL error messages!! Using the same strategy as generate_nv() function in generate_stubs.sh , I&#39;ve extracted the error messages from https://github.com/KhronosGroup/OpenCL-Headers/tree/m...</li><li><a href="https://streamhpc.com/blog/2013-04-28/opencl-error-codes/">OpenCL error codes (1.x and 2.x) - StreamHPC</a>: Knowing all errors by heart is good for quick programming, but not always the best option. Therefore I started to create ...</li><li><a href="https://github.com/KhronosGroup/OpenCL-Headers/blob/main/CL/cl.h">OpenCL-Headers/CL/cl.h at main · KhronosGroup/OpenCL-Headers</a>: Khronos OpenCL-Headers. Contribute to KhronosGroup/OpenCL-Headers development by creating an account on GitHub.</li><li><a href="https://github.com/KhronosGroup/OpenCL-Headers/blob/main/CL/cl_ext.h">OpenCL-Headers/CL/cl_ext.h at main · KhronosGroup/OpenCL-Headers</a>: Khronos OpenCL-Headers. Contribute to KhronosGroup/OpenCL-Headers development by creating an account on GitHub.</li><li><a href="https://github.com/KhronosGroup/OpenCL-Headers/blob/main/CL/cl_egl.h">OpenCL-Headers/CL/cl_egl.h at main · KhronosGroup/OpenCL-Headers</a>: Khronos OpenCL-Headers. Contribute to KhronosGroup/OpenCL-Headers development by creating an account on GitHub.</li><li><a href="https://github.com/KhronosGroup/OpenCL-Headers/blob/main/CL/cl_dx9_media_sharing.h">OpenCL-Headers/CL/cl_dx9_media_sharing.h at main · KhronosGroup/OpenCL-Headers</a>: Khronos OpenCL-Headers. Contribute to KhronosGroup/OpenCL-Headers development by creating an account on GitHub.</li><li><a href="https://github.com/KhronosGroup/OpenCL-Headers/blob/main/CL/cl_d3d11.h">OpenCL-Headers/CL/cl_d3d11.h at main · KhronosGroup/OpenCL-Headers</a>: Khronos OpenCL-Headers. Contribute to KhronosGroup/OpenCL-Headers development by creating an account on GitHub.</li><li><a href="https://github.com/KhronosGroup/OpenCL-Headers/blob/main/CL/cl_d3d10.h">OpenCL-Headers/CL/cl_d3d10.h at main · KhronosGroup/OpenCL-Headers</a>: Khronos OpenCL-Headers. Contribute to KhronosGroup/OpenCL-Headers development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1251541160023949394)** (69 messages🔥🔥): 

- **George Hotz addresses recursive rewrite assert**: A member asked about an assert in `uops graph_rewrite` which counts recursive rewrites. [This assert](https://docs.tinygrad.org/tensor/creation/#creation-random) ensures that recursive rewrites loop below a threshold to prevent infinite recursion.
  
- **Gradient sync in `beautiful_mnist_multigpu.py` simplified**: George Hotz confirmed that gradient synchronization is inherent in Tinygrad's optimizer. He emphasized the simplicity over Torch's Distributed Data Parallel.

- **Tinygrad's goals to surpass PyTorch**: George Hotz discussed Tinygrad's aim to outperform PyTorch in speed, API simplicity, and bug reduction. While currently slower, especially in LLM training, Tinygrad's purity and potential were highlighted by enthusiastic users.

- **Mixed precision implementation discussion**: A user sought advice from George Hotz on implementing mixed precision for a model, discussing various approaches including using `DEFAULT_FLOAT` and `nn` class modifications. George suggested `cast_` methods and late casting techniques for better efficiency.

- **Kernel issues resolved**: A user resolved kernel issues related to `remainder` tensors not appearing in UOp graphs, learning that separate `realize` calls split operations into different kernels. Discussions highlighted the significance of realizing tensors appropriately to meet custom accelerator requirements.

**Link mentioned**: <a href="https://docs.tinygrad.org/tensor/creation/#creation-random">Creation - tinygrad docs</a>: no description found

  

---



### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1252251816423587851)** (1 messages): 

- **Introducing GPT Notes app**: A member showcased a hybrid application combining an LLM client and notes app, allowing users to dynamically include/exclude notes into the LLM's context. The project, built without using any JS libraries, offers features like import/export, basic markdown, and responses management.
- **No mobile support, pure vanilla JS**: Despite lacking mobile support, the app boasts of no reliance on libraries, purely built with vanilla JavaScript. It includes functionalities like storing API keys, history, and notes locally in the browser.
- **Explore the app on Codepen**: The member provided a [Codepen link](https://codepen.io/bulgakoff08/project/editor/DnJLrG) for the project and a [deployed fullscreen app](https://000700836.deployed.codepen.website/). The application serves as an example for anyone looking for a similar tool.

**Link mentioned**: <a href="https://000700836.deployed.codepen.website/">GPNotes</a>: no description found

  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1251345585571762266)** (68 messages🔥🔥): 

- **OpenRouter Errors without User Messages Sparking Debate**: Users discussed the issue of OpenRouter returning errors if no user message is found, noting that some models require at least a user message as an opener, and even starting with an assistant is not supported by every model due to their instruct-tuned format. A suggested workaround was using the `prompt` parameter instead of `messages` ([OpenRouter Docs](https://openrouter.ai/docs/transforms)).

- **Document Formatting and Uploading Puzzles Users**: A user inquired about services for formatting text into structured "papers," leading to a broader discussion on document formatting and uploading. The conversation highlighted the complexity of making PDFs LLM-friendly, with suggestions to preprocess PDFs using tools like [PDF.js and Jina AI Reader](https://jina.ai/reader/).

- **Qwen2's Censorship Criticized**: Users shared their experiences with the Qwen2 model, labeling it as overly censored despite jailbreak attempts, evidenced by implausibly positive narrative outcomes. Alternative, less-censored models like Dolphin Qwen 2 were recommended.

- **Gemini Flash's Context Limit Debate**: A discrepancy in Gemini Flash's token generation limits prompted questions, with OR listing 22k tokens while Gemini Docs claimed 8k. It was clarified that OR counts characters to match Vertex AI's pricing model ([OpenRouter Status](https://openrouter.ai/models/google/gemini-flash-1.5/status)).

- **Rate Limits and Model Configuration Questions Arise**: Users inquired about rate limits for models like GPT-4o and Opus, leading to guidance on checking rate limits via API keys ([OpenRouter Rate Limits](https://openrouter.ai/docs/limits)). Also, discussions about maximizing model performance and configuration settings like "Sonnet from OR vs Sonnet with Claude key" and "LiteLLM vs OR Routing" unfolded, emphasizing custom retry options and API call efficiency.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openrouter.ai/models/google/gemini-flash-1.5/status">Google: Gemini Flash 1.5 (preview) – Provider Status and Load Balancing</a>: See provider status and make a load-balanced request to Google: Gemini Flash 1.5 (preview) - Gemini 1.5 Flash is a foundation model that performs well at a variety of multimodal tasks such as visual u...</li><li><a href="https://openrouter.ai/docs/limits">Limits | OpenRouter</a>: Set limits on model usage</li><li><a href="https://jina.ai/reader/">Reader API</a>: Read URLs or search the web, get better grounding for LLMs.</li><li><a href="https://openrouter.ai/docs/transforms">Transforms | OpenRouter</a>: Transform data for model consumption
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[일반](https://discord.com/channels/1091220969173028894/1246338143226167349/)** (1 messages): 

is.maywell: <:a6adc388ea504e89751ecbbd50919d3a:1240669253699637339>
  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1251376496178892821)** (48 messages🔥): 

- **TextGen Integration in LangChain Broken**: A member reported that **textgen integration** into **LangChain** is broken due to an **API update**.
- **Best Splitter for Chunking Textbook PDFs**: A member asked for advice on the best **splitter** to use for **chunking PDF text** according to headers and chapters, aiming to structure the text better.
- **LangChain Postgres Installation Trouble**: Users exchanged advice about installing **langchain_postgres**, with a solution involving correcting the targeted directory for `pip install`.
- **Module Error with New Tenacity Version**: A user encountered a **ModuleNotFoundError** for 'tenacity.asyncio' following an update to version **8.4.0**, but found reverting to version **8.3.0** resolved the issue.
- **Help for New LangChain Users**: Multiple users sought guidance on implementing specific models or error handling in **LangChain**, including transitioning from Python code to **LangChain JS**, managing HuggingFace models, and recommended LLMs like **Llama 3** or **Google Gemini** for local use. A relevant discussion was linked [here](https://github.com/langchain-ai/langchain/discussions/22899).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="http://'">no title found</a>: no description found</li><li><a href="https://'">no title found</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=qKZLDEIL2r0">How to build a chatGPT chatbot on Slack</a>: Welcome to this tutorial video on creating a Slack chatbot using the OpenAI language model, LangChain, and the Slack Bolt library. This video will showcase t...</li><li><a href="https://www.instagram.com/reel/C8SMrCIOyGZ/?igsh=MzdrbGhrbXh4NWk=">Vaios Laschos on Instagram: &quot;This is the promotional video for the app that I create for Generative AI Agents Developer Contest by NVIDIA and LangChain. #NVIDIADevContest #LangChain &#064;nvidiadeveloper 

For demo video, see: 

https://www.youtube.com/watch?v=9mMbQpofiJY

The app makes the life of academics easier by automating some tedious jobs like retrieving files from arxiv, making summaries and performing context based translation. Future goal is to make a paper survey out of a single paper.

If you feel like in need for some punishment. Check my git repo https://github.com/artnoage/Langgraph_Manuscript_Workflows&quot;</a>: 72 likes, 1 comments - vaioslaschos on June 16, 2024: &quot;This is the promotional video for the app that I create for Generative AI Agents Developer Contest by NVIDIA and LangChain....&quot;. </li><li><a href="https://tenor.com/view/blowing-kisses-kisses-kiss-gratitude-huge-thanks-gif-16468716440995283694">Blowing Kisses Gratitude GIF - Blowing kisses Kisses Kiss - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/langchain-ai/langchain/discussions/22899">How to properly provide the input schema to the model · langchain-ai/langchain · Discussion #22899</a>: Checked other resources I added a very descriptive title to this question. I searched the LangChain documentation with the integrated search. I used the GitHub search to find a similar question and...
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1251321854589931601)** (14 messages🔥): 

- **R2R adds automatic knowledge graph construction**: R2R v2 now includes automatic knowledge graph construction along with a comprehensive [cookbook](https://r2r-docs.sciphi.ai/cookbooks/knowledge-graph#knowledge-graph-agents) that walks through basic and advanced features. *"This should make a great (and up to date) starting point If you are interested in KGs."*

- **Collision event interactive map launched**: Eloquentsyntax announced an [interactive map](https://collision.talewind.ai/) for Collision parties and events. The map includes filters, door fees, addresses, RSVP links, and an AI chat to find events easily.

- **CryptGPT: Privacy-Preserving LLMs using Vigenere cipher**: Diwank introduced [CryptGPT](https://x.com/diwanksingh/status/1802118343446724655), a project that pretrains a GPT-2 model on Vigenere ciphertexts, ensuring privacy from the model provider. The unique feature is that usage requires knowledge of the encryption key.

- **Scrape Web + Create diagrams with GPT**: Ashes47 shared a project from user Anuj4799, who created a custom GPT for generating technical diagrams. The demo can be [checked out here](https://chat.openai.com/g/g-7EWovgPuJ-mindstream).

- **Rubik's AI beta tester and promo**: Paulm24 invited users to beta test an advanced research assistant and search engine, offering a 2-month free premium with models like GPT-4 Turbo and Claude 3 Opus using the promo code `RUBIX`. Interested users are encouraged to sign up at [Rubik's AI](https://rubiks.ai/).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://collision.talewind.ai/">no title found</a>: no description found</li><li><a href="https://x.com/Anuj4799/status/1802425335549661492">Tweet from Anuj Verma (@Anuj4799)</a>: So I was having a tough time with generating technical diagrams, and I ended up creating a custom GPT to handle it for me. Now, I&#39;m loving the ease and efficiency! Check it out: https://chat.opena...</li><li><a href="https://x.com/diwanksingh/status/1802118343446724655">Tweet from Diwank Singh (@diwanksingh)</a>: http://x.com/i/article/1802116084507848704</li><li><a href="https://r2r-docs.sciphi.ai/cookbooks/knowledge-graph#knowledge-graph-agents].">R2R Documentation</a>: The official documentation for the RAG to Riches (R2R) framework.</li><li><a href="https://www.appstorm.ai/">Appstorm.ai: Generative AI for Effortless App Development</a>: no description found</li><li><a href="https://rubiks.ai/">Rubik's AI - AI research assistant & Search Engine</a>: no description found
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/)** (1 messages): 

emarco: https://www.youtube.com/watch?v=0gJLFTlGFVU
  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1251277835323310130)** (21 messages🔥): 

<ul>
    <li><strong>OtterTune is no more</strong>: OtterTuneAI officially shut down after a failed acquisition deal. The announcement was shared on <a href="https://x.com/andy_pavlo/status/1801687420330770841?s=46&t=Tc6nPt_FP2Ybqya6_6Xu-w">Twitter</a>.</li>
    <li><strong>Check out Apple's models on Hugging Face</strong>: Apple has published several models optimized for on-device performance on Hugging Face, including <a href="https://huggingface.co/apple/coreml-detr-semantic-segmentation">DETR Resnet50 Core ML</a> for semantic segmentation and <a href="https://huggingface.co/collections/apple/core-ml-stable-diffusion-666b3b0f4b5f3d33c67c6bbe">Stable Diffusion Core ML</a>.</li>
    <li><strong>OpenAI under fire for appointing former NSA head</strong>: Edward Snowden criticized OpenAI’s decision to appoint former NSA Director Paul M. Nakasone to its board, <a href="https://x.com/Snowden/status/1801610725229498403">calling it a betrayal</a> of public trust.</li>
    <li><strong>Runway releases Gen-3 Alpha video model</strong>: Runway introduces Gen-3 Alpha, a new model for video generation with advanced features. Details were shared on <a href="https://x.com/runwayml/status/1802691475391566108?s=46&t=90xQ8sGy63D2OtiaoGJuww">Twitter</a>.</li>
    <li><strong>Anthropic research on reward tampering</strong>: Anthropic publishes a new paper on AI models learning to hack their reward systems. The research and its findings are summarized in their <a href="https://anthropic.com/research/reward-tampering">blog post</a>.</li>
</ul>

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://studios.trychroma.com/flo-on-lindy">Flo Crivello on Building Lindy.AI</a>: no description found</li><li><a href="https://x.com/anthropicai/status/1802743256461046007?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Anthropic (@AnthropicAI)</a>: New Anthropic research: Investigating Reward Tampering.  Could AI models learn to hack their own reward system?  In a new paper, we show they can, by generalization from training in simpler settings. ...</li><li><a href="https://huggingface.co/apple">apple (Apple)</a>: no description found</li><li><a href="https://x.com/bshlgrs/status/1802766374961553887?t=1MOp6l3T7xJvK6yAiGomPw&s=19">Tweet from Buck Shlegeris (@bshlgrs)</a>: ARC-AGI’s been hyped over the last week as a benchmark that LLMs can’t solve. This claim triggered my dear coworker Ryan Greenblatt so he spent the last week trying to solve it with LLMs. Ryan gets 71...</li><li><a href="https://x.com/airkatakana/status/1801796780595757175?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Air Katakana (@airkatakana)</a>: i’m calling the top, this company didn’t even do anything yet</li><li><a href="https://x.com/tomgoldsteincs/status/1802726878924464273">Tweet from Tom Goldstein (@tomgoldsteincs)</a>: LLMs can memorize training data, causing copyright/privacy risks. Goldfish loss is a nifty trick for training an LLM without memorizing training data.  I can train a 7B model on the opening of Harry P...</li><li><a href="https://x.com/gdb/status/1802707715816595869?t=Y7yDU61o45Cpx69DHss-jQ&s=19">Tweet from Greg Brockman (@gdb)</a>: GPT-4o as an assistant for helping doctors screen and treat cancer patients:  Quoting Othman Laraki (@othman)   I&#39;m thrilled to announce the @Color Copilot, which we developed in partnership with ...</li><li><a href="https://x.com/runwayml/status/1802691475391566108?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Runway (@runwayml)</a>: Introducing Gen-3 Alpha: Runway’s new base model for video generation.  Gen-3 Alpha can create highly detailed videos with complex scene changes, a wide range of cinematic choices, and detailed art di...</li><li><a href="https://x.com/andy_pavlo/status/1801687420330770841?s=46&t=Tc6nPt_FP2Ybqya6_6Xu-w">Tweet from Andy Pavlo (@andy_pavlo@discuss.systems) (@andy_pavlo)</a>: I&#39;m to sad to announce that @OtterTuneAI is officially dead. Our service is shutdown and we let everyone go today (1mo notice). I can&#39;t got into details of what happened but we got screwed ove...</li><li><a href="https://x.com/fchollet/status/1802801425514410275">Tweet from François Chollet (@fchollet)</a>: Re: the path forward to solve ARC-AGI...  If you are generating lots of programs, checking each one with a symbolic checker (e.g. running the actual code of the program and verifying the output), and ...</li><li><a href="https://x.com/bshlgrs/status/1802766374961553887?t=1MOp6l3T7xJvK6yA">Tweet from Buck Shlegeris (@bshlgrs)</a>: ARC-AGI’s been hyped over the last week as a benchmark that LLMs can’t solve. This claim triggered my dear coworker Ryan Greenblatt so he spent the last week trying to solve it with LLMs. Ryan gets 71...</li><li><a href="https://x.com/Snowden/status/1801610725229498403">Tweet from Edward Snowden (@Snowden)</a>: They&#39;ve gone full mask-off: 𝐝𝐨 𝐧𝐨𝐭 𝐞𝐯𝐞𝐫 trust @OpenAI or its products (ChatGPT etc). There is only one reason for appointing an @NSAGov Director to your board. This is a willful, calculat...</li><li><a href="https://readwise.io/reader/shared/01j02hxfc781zchghjvdrfs30x/">Flo Crivello on Building Lindy.AI | annotated by Daniel</a>: AI Agents are a new category of software, built on top of large language models (LLMs).
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1251265169133146202)** (20 messages🔥): 

- **Prime Intellect set to open source DiLoco and DiPaco**: Users discussed how Prime Intellect plans to release state-of-the-art models DiLoco and DiPaco soon, enhancing open collaboration. One member shared a [Prime Intellect link](https://www.primeintellect.ai/) detailing how the platform democratizes AI through distributed training across global compute resources.
  
- **Bittensor utilizes The Horde**: Users mentioned that The Horde, known for distributing computational tasks, is being utilized on the Bittensor network for decentralized AI model training.

- **DeepMind did not participate**: Contrary to some expectations, it was clarified that DeepMind did not contribute to specific ongoing projects in the community discussion.

- **YouTube video on Optimizers**: Members shared a [YouTube video](https://www.youtube.com/watch?v=mdKjMPmcWjY&t=23s) about optimizers, explaining various types from Gradient Descent to Adam. It offered an easy way to remember different optimizers for effective model training.

- **ChatGPT's multi-step responses**: A discussion centered around how ChatGPT formulates multi-step responses, clarifying that different transformer blocks can be processed separately. This sparked interest and questions about specific parallelizations within transformer layers.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.primeintellect.ai/">Prime Intellect - Commoditizing Compute &amp; Intelligence</a>: Prime Intellect democratizes AI development at scale. Our platform makes it easy to find global compute resources and train state-of-the-art models through distributed training across clusters. Collec...</li><li><a href="https://www.youtube.com/watch?v=mdKjMPmcWjY&t=23s">Optimizers - EXPLAINED!</a>: From Gradient Descent to Adam. Here are some optimizers you should know. And an easy way to remember them. SUBSCRIBE to my channel for more good stuff! REFER...
</li>
</ul>

</div>
  

---



### **Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1251845720814653490)** (20 messages🔥): 

- **Debate on AGI Hype**: A user shared a [YouTube video titled "Is AGI Just a Fantasy?"](https://youtu.be/4JF1V2hzGKE?si=gKCvVxBpwsGTD6ow) featuring Nick Frosst, spurring discussions about the hype, real tech advancements, and evaluation of LLMs. Members expressed fatigue over "hype bros" but acknowledged the importance of ongoing investment, likening it to the dot-com bubble that led to significant innovations.

- **Call for Next.js App Router Collaboration**: A member announced the creation of a GitHub issue inviting collaboration on migrating the Cohere toolkit UI to Next.js App Router to improve code transferability and attract more contributors. The [GitHub issue #219](https://github.com/cohere-ai/cohere-toolkit/issues/219) contains more details about the feature request.

- **C4AI Talk Link Shared**: Nick Frosst provided a [Google Meet link](https://meet.google.com/ibt-wsgv-kbq?hs=122&authuser=0) for the C4AI talk and directed members with questions to the relevant [Discord channel](https://discord.gg/yws9vsRe).

- **Interest in Contributing Data for Training**: A user inquired about submitting 8,000 PDFs for embedding model training with Cohere. Nick Frosst sought clarification if the user intended to fine-tune an embedding model, opening a discussion on potential data contributions.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://youtu.be/4JF1V2hzGKE?si=gKCvVxBpwsGTD6ow">Is AGI Just a Fantasy?</a>: Nick Frosst, the co-founder of Cohere, on the future of LLMs, and AGI. Learn how Cohere is solving real problems for business with their new AI models.Nick t...</li><li><a href="https://github.com/cohere-ai/cohere-toolkit/issues/219">any plans to migrate to nextjs app router? · Issue #219 · cohere-ai/cohere-toolkit</a>: What feature(s) would you like to see? hi, fantastic toolkit and project to get started quickly. i was wondering if there is any plan to migrate it to the app router. most (if not all) of new nextj...</li><li><a href="https://discord.gg/yws9vsRe">Join the Cohere For AI Discord Server!</a>: Cohere For AI&#x27;s Open Science Community is a space to come together to work on machine learning research collaborations. | 3016 members
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[project-sharing](https://discord.com/channels/954421988141711382/1218409701339828245/1251909179317162067)** (11 messages🔥): 

- **Cohere models integrate into Chrome for free**: A member announced a [free Chrome Extension](https://www.asksteve.to) that integrates **LLMs directly into the browser**, eliminating repetitive tasks and enhancing productivity. Users are encouraged to provide feedback and can configure it with detailed instructions provided.
- **Interactive Collision map launched**: Another member created an [interactive map of all Collision events](https://collision.talewind.ai/), allowing users to filter by event details and access AI chat for easier navigation. It utilizes Sveltekit, Supabase, and Vercel for its build.
- **Command R+ configuration issue resolved**: A user experienced issues configuring Command R+ with the Cohere-powered extension but received help to rectify it by using a Blank Template first. The developer acknowledged the bug and plans to fix it.
- **Inquiry about Cohere data submission**: A user inquired if **Cohere** accepts data submissions for training, specifically mentioning they have nearly 8,000 PDFs for embedding model training.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://collision.talewind.ai/">no title found</a>: no description found</li><li><a href="https://www.asksteve.to">Ask Steve - Unlock the Power of ChatGPT and Gemini in any web page!</a>: Ask Steve adds AI superpowers from ChatGPT &amp; Gemini to any web page, so you can get your everyday tasks done better and faster. FREE!</li><li><a href="https://docs.google.com/document/d/1Um99Y4BCfGT4cANYXR7TnLvfIK3h_9lcmx4PQ6uuIns/edit#heading=h.r7p77gz6wtd9">Configuring Ask Steve to use Cohere Command R+</a>: Configuring Ask Steve to use Cohere Command R+ You will need to login to Ask Steve in order to add a new model: chrome-extension://gldebcpkoojijledacjeboaehblhfbjg/options.html After logging in, go to...
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[announcements](https://discord.com/channels/954421988141711382/996880279224451154/1252296943137198210)** (1 messages): 

- **David Stewart to host Cohere Developer Office Hours**: A relaxed session is scheduled for tomorrow, hosted by David Stewart, a seasoned Solution Architect at Cohere. Members are encouraged to post their questions and issues on [this thread](https://discord.com/channels/954421988141711382/1168411509542637578/1252294070789869618) to get prioritized during the event.
- **Event details released**: The Office Hours event will take place on June 18, at 1:00 PM ET. Join the event [here](https://discord.gg/jy5XFg5GDH?event=1248300905703673987) for live interaction and guidance on Cohere API and model-related queries.

**Link mentioned**: <a href="https://discord.gg/jy5XFg5GDH?event=1248300905703673987">Join the Cohere Community Discord Server!</a>: Cohere community server. Come chat about Cohere API, LLMs, Generative AI, and everything in between. | 17098 members

  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1251269680770056264)** (14 messages🔥): 

- **Model freezes mid-code**: One member inquired if others were experiencing their model freezing while in the middle of coding. Another member replied that it usually completes the task even when it looks frozen.
- **Windows installation issues**: A user reported issues with installing and running the model on Windows. They were advised to search for help and post their query in a designated channel.
- **Memory functionality improves**: A member expressed satisfaction with getting memory to work in a "very primitive way." They enthusiastically shared their progress with the community.
- **Llama 3 Performance Review**: A detailed [model comparison and performance test for Llama 3](https://www.reddit.com/r/LocalLLaMA/comments/1cal17l/llm_comparisontest_llama_3_instruct_70b_8b/) was shared, promising a comprehensive assessment of Llama 3 Instruct's capabilities across various formats and quantization levels.
- **Profiles functionality feature**: A [new 'profiles' feature](https://x.com/tx_smitht/status/1802156524833546553) on Open Interpreter was highlighted. A member shared a [video](https://youtu.be/ZQtSN7SAsYM) to explain its capabilities and applications.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/tx_smitht/status/1802156524833546553">Tweet from Thomas Smith (@tx_smitht)</a>: If you don&#39;t know about Open Interpreter&#39;s new &#34;profiles&#34; functionality, you need to check it out!   It lets you extend OI&#39;s capabilities. It&#39;s like uploading a specific set of...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1cal17l/llm_comparisontest_llama_3_instruct_70b_8b/">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1251502290519593103)** (4 messages): 

- **Check your unit arrival in pinned messages**: A user asked how to check when their unit is arriving, mentioning they placed an order very early. Another member redirected them to a pinned message in the channel for manufacturing updates and timelines. 
- **Discuss combo of vector DB, semantic search, and LLM**: A question was raised about the potential of combining a vector database of audio with voice-based semantic search and indexing, alongside an LLM capable of accessing this data and performing actions. The proposed combination hints at a powerful tool for actions based on verbal inputs.
  

---


### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1251532212449443880)** (6 messages): 

- **DIY AI Cyber Hat turns heads**: A member shared their project on making an open-source AI-enabled wearable hat, likening it to smart glasses. They provided a video preview and expressed openness for collaboration, [view the video here](https://www.youtube.com/watch?v=71p9DcGqNDc).
- **Terminator humor on hat design**: One member humorously remarked that the hat design made the creator look like a terminator sent to eliminate the founder of Hobby Lobby.
- **Interest in sci-fi wearables sparks engagement**: People showed enthusiasm for the AI hat project, requesting access to the source code once it's cleaned up. The creator suggested possible future integration of more sensors for scientific experiments.
- **Pi Zero heads to Big Mouth Billy Bass**: The same creator teased their next project involving integrating a Pi Zero in a Big Mouth Billy Bass.
- **Dream Machine generates buzz**: A member shared [Dream Machine](https://lumalabs.ai/dream-machine), an AI model that creates high-quality, realistic videos from text and images. The model aims to build a universal imagination engine and is now available to the public.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lumalabs.ai/dream-machine">Luma Dream Machine</a>: Dream Machine is an AI model that makes high quality, realistic videos fast from text and images from Luma AI</li><li><a href="https://www.youtube.com/watch?v=71p9DcGqNDc">I Made My Own Custom AI Cyber Hat</a>: This is a video about the start of a project of mine that I&#39;ve called &quot;heddy&quot; (the hat portion at least).  I created my own smart AI enabled hat largely thro...
</li>
</ul>

</div>
  

---



### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1251292718609596446)** (7 messages): 

- **Single node focus for now in Torchtune**: When asked if Torchtune plans to release multi-node training, a member clarified that the focus is currently on single node training. However, they noted that "our ‘tune run’ command is a wrapper around torch run" and with minor changes, multi-node setups could work, although it's untested.
  
- **Distributed config adjustments for multi-node training**: Members exchanged tips on setting up multi-node training in Torchtune. One suggested setting `tune run —nnodes 2`, while another mentioned the need for `TorchX` or `slurm` to handle script launches and node communications over specific ports, pointing to resources like [TorchX](https://github.com/pytorch/torchx) and [hybrid shard strategy documentation](https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.ShardingStrategy).

**Link mentioned**: <a href="https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.ShardingStrategy)">FullyShardedDataParallel &mdash; PyTorch 2.3 documentation</a>: no description found

  

---



### **DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/1251542470358073416)** (5 messages): 

- **Llama3 tokenizer remains unchanged**: Members discussed whether the **Llama3 tokenizer** was extended for the German model. One member confirmed that *"tokenizer is the same as the base Llama3"*.

- **Concerns about German token handling**: A member questioned the rationale behind not extending the tokenizer, noting that *not including German tokens probably decreases the context window quite a bit*. They were curious if they were missing any reasoning, especially considering the potential increases in embeddings.

- **Size comparison with Llama2**: Another member pointed out that **Llama3's tokenizer is 4 times larger** than **Llama2's**. They inquired whether it was already more effective on German or if there were still issues.
  

---



### **Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/1251252730811645984)** (3 messages): 

- **Alternative Positions in AI Discussions Praised**: One member appreciated another's writing for *"addressing the alternative position in good faith"*. They humorously noted that ChatGPT's rise is *"a full employment act for data engineers in perpetuity"*.

- **Thoughtbot's LLM Guide Shoutout**: A member highlighted a useful [thoughtbot resource](https://thoughtbot.com/blog/understanding-open-source-llms) for beginners in LLMs. They recommended reading [Jose Blanco's post](https://thoughtbot.com/blog/how-to-use-open-source-LLM-model-locally) on using open-source LLMs locally and remotely.

- **Clarity in Naming Conventions for LLMs Appreciated**: Another member found the categorization of LLMs into Base, Instruct, and Chat models particularly clear and detailed.

**Link mentioned**: <a href="https://thoughtbot.com/blog/understanding-open-source-llms">Understanding open source LLMs</a>: Do you think you can run any Large Language Model (LLM) on your machine?

  

---


### **Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/1251736192769331221)** (1 messages): 

- **Turso adds native vector search support**: Turso has introduced [native vector search](https://turso.tech/blog/turso-brings-native-vector-search-to-sqlite) capabilities to their platform, supplementing SQLite’s existing features. This new addition aims to simplify vector search for users building AI products, addressing previous challenges with managing extensions like [sqlite-vss](https://github.com/asg017/sqlite-vss).

**Link mentioned**: <a href="https://turso.tech/blog/turso-brings-native-vector-search-to-sqlite">Turso brings Native Vector Search to SQLite</a>: Vector Similarity Search is now available!

  

---



### **AI Stack Devs (Yoko Li) ▷ #[ai-town-discuss](https://discord.com/channels/1122748573000409160/1132926337598902293/)** (1 messages): 

gomiez: anyone know of the hospital ai town project name?
  

---



### **Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/)** (1 messages): 

cryovolcano.: can we use llamafile with tinyllama as a search engine in firefox ?
  

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
