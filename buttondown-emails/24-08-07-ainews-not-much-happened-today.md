---
id: e19ff7ac-ae0f-4018-b347-1bb26b3044ee
title: '[AINews] not much happened today'
date: '2024-08-08T01:50:11.687874Z'
status: sent
type: public
source: api
metadata: {}
original_slug: ainews-not-much-happened-today-4029
---

<!-- buttondown-editor-mode: plaintext -->**[anonymous](https://x.com/AndrewCurran_/status/1821051919768678701) [strawberries](https://x.com/swyx/status/1821359574068146681) are all you need.**

> AI News for 8/6/2024-8/7/2024. We checked 7 subreddits, [**384** Twitters](https://twitter.com/i/lists/1585430245762441216) and **28** Discords (**249** channels, and **2423** messages) for you. Estimated reading time saved (at 200wpm): **247 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

No clear major story for the day but lots of interesting small nuggets:

- Mistral Large's external scores are in, and they do very well - Gemini Pro-tier - on hard Lmsys prompts, as well as independent benchmarks like [Aidanbench](https://x.com/aidan_mclau/status/1821334577576644935)
- Code a [Vision Language Model from scratch](https://x.com/hkproj/status/1821081257712705848)! (thanks Sam Julien for picking this one in the LS Discord)
- The new PyTorch [FlexAttention](https://x.com/cHHillee/status/1821253769147118004) subsumes all Attention Variants including FlashAttention 2 (not FA 3 though!)'s API and the [increasingly popular local-global attention spectrum](https://buttondown.email/ainews/archive/ainews-shazeer-et-al-2024/) including Sliding Window
- Check out the [Grokfast optimizer](https://x.com/_clashluke/status/1820810798693818761)!

You could, of course, spend one more epoch on [**Segment Anything 2**, which is now up on the Latent Space Podcast](https://www.latent.space/p/sam2).

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

**OpenAI Structured Outputs and Model Updates**

OpenAI has introduced structured outputs in their API, allowing developers to enforce specific JSON schemas for model responses. This feature is now supported across various models, including gpt-4-0613 and gpt-3.5-turbo-0613 and later versions. [@sama](https://twitter.com/sama/status/1820881534909300769) announced this highly requested feature, which enables 100% reliability in matching output schemas according to OpenAI's evaluations. The update includes:

- A new "strict" mode for function calling that ensures outputs match the supplied tool definition
- A new "response_format" parameter that allows specifying JSON output schemas
- Introduction of a new model: gpt-4o-2024-08-06

[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1820886172476047824) highlighted that this update achieves 100% reliability in matching output schemas, which is particularly useful for downstream tasks when the model is not calling a tool but responding to the user in a structured way.

Additionally, [@corbtt](https://twitter.com/corbtt/status/1820910339388825762) noted that OpenAI quietly dropped the price of GPT-4o (the real one, not mini) by 50% without a formal announcement, now listing it at $2.50/1M tokens on their pricing page.

**AI Model Developments and Benchmarks**

Several new AI models and benchmarks have been announced:

1. Mistral Large 2: [@GuillaumeLample](https://twitter.com/GuillaumeLample/status/1820833645009277388) announced the release of Mistral Large 2, which performs exceptionally well in coding, hard prompts, math, and longer query categories, outperforming GPT4-Turbo and Claude 3 Opus in some areas. It's now leading the Arena hard leaderboards and is an open-weight model.

2. Idefics3-Llama: [@mervenoyann](https://twitter.com/mervenoyann/status/1820896952957153762) introduced Idefics3-Llama, a multimodal model based on Llama 3.1 that accepts an arbitrary number of interleaved images with text and has a huge context window of 10k tokens.

3. BigLlama-3.1-1T-Instruct: [@maximelabonne](https://twitter.com/maximelabonne/status/1820746727638323531) presented an upscaled version of Meta-Llama-3-120B-Instruct, created through a self-merge of Llama 3 70B.

4. New benchmarks: [@aidan_mclau](https://twitter.com/rez0__/status/1820853537733021770) introduced a new benchmark called "big_model_smell" that measures creativity, reliability, attention, and instruction following.

**AI Hardware and Robotics**

[@adcock_brett](https://twitter.com/adcock_brett/status/1820792697315348640) introduced Figure 02, described as the world's most advanced AI hardware. Key features include:

- 6x Cameras
- 50%+ Battery capacity
- Onboard Vision Language Model (VLM)
- 3x CPU / GPU power
- 4th Gen Hands
- Integrated wiring
- Exoskeleton structure
- Speech-to-speech reasoning capabilities

The robot is designed for autonomous operation and includes a custom 2.25 KWh battery pack, aiming for up to 20 hours of useful work per day.

**AI Safety and Regulation**

[@ylecun](https://twitter.com/ylecun/status/1820927645757940178) shared concerns about California's SB1047 (Safe and Secure Innovation for Frontier Artificial Intelligence Models Act), stating that it won't solve intended issues and may harm AI R&D in academia, small tech companies, and the open-source community. [@fchollet](https://twitter.com/fchollet/status/1820862042934493223) echoed these concerns, arguing that holding open model developers responsible for all fine-tuned models downstream makes no sense and could discourage open model sharing.

**Miscellaneous AI Developments**

- [@omarsar0](https://twitter.com/omarsar0/status/1820941367784136718) discussed the importance of structured outputs in improving LLM application performance and reliability.
- [@jeremyphoward](https://twitter.com/jeremyphoward/status/1820955161579417957) announced FastHTML, a growing gallery of live FastHTML code examples for building interactive components and applications.
- [@LangChainAI](https://twitter.com/LangChainAI/status/1820966277978235004) introduced support for OpenAI's new structured output functionality in their latest release candidates for both Python and JavaScript.

These developments showcase the rapid progress in AI model capabilities, hardware integration, and the ongoing discussions around AI safety and regulation in the field.

---

# AI Reddit Recap

## /r/LocalLlama Recap

**Theme 1. LLMs as Productivity Boosters in Research and Development**

- **[auto-md | tool | One click convert files/zips + GitHub repositories into Markdown documents (.md)](https://i.redd.it/dl555pnlw5hd1.png)** ([Score: 62, Comments: 10](https://reddit.com//r/LocalLLaMA/comments/1em1sqi/automd_tool_one_click_convert_fileszips_github/)): The tool **auto-md** has been updated with a **Windows .exe** version, allowing users to convert files, zips, and GitHub repositories into Markdown documents with a single click. The developer plans to release a **Mac app** soon and appreciates the support received, including GitHub stars and user feedback from previous posts.
  - **Dark_Fire_12** shared an alternative approach to building a similar tool, opting for **file extension filtering** instead of folder depth search. They included a [screenshot](https://preview.redd.it/xcn7tq8ye6hd1.png?width=1565&format=png&auto=webp&s=9f86806aebdf8d3f37742f3b88aadb7100abf0b8) demonstrating their implementation.
  - **Environmental-Car267** mentioned creating two **similar tools** for personal use: one that copies codebases to clipboard for pasting into **Sonnet/GPT**, and another that lets AI select important files autonomously. These tools exclude certain folders and files during the process.
- **How a research scientist at Google Deepmind uses LLM** ([Score: 318, Comments: 89](https://reddit.com//r/LocalLLaMA/comments/1elz2ur/how_a_research_scientist_at_google_deepmind_uses/)): **Nicholas Carlini**, a research scientist at **Google DeepMind**, shares his approach to using **Large Language Models (LLMs)** for productivity enhancement in a detailed blog post. The article emphasizes the significant value in **augmenting human capabilities** with AI, suggesting that this intermediate step is crucial before aiming for fully autonomous AI systems.
  - Users agree that **LLMs** are both **overhyped and underhyped**, with many people either exaggerating their capabilities or dismissing them entirely. The technology is particularly useful when operating at the "**edge of your knowledge**," helping fill gaps in partial understanding.
  - The article demonstrates LLMs' tendency to **hallucinate**, as it incorrectly stated there's no Python library for Podman, despite the existence of [podman-py](https://podman-py.readthedocs.io/en/latest/). Users emphasize the importance of evaluating what LLMs **can do** rather than focusing on their limitations.
  - Many users report significant **productivity boosts** from using LLMs, with one estimating a **50% increase** in coding speed. LLMs are particularly helpful for learning new technologies, automating mundane tasks, and debugging, though some express concerns about their use in academic writing.


**Theme 2. Advancements in AI Model Compression and Quantization**

- **Quantize 123B Mistral-Large-Instruct-2407 to 35 GB with only 4% accuracy degeneration.** ([Score: 77, Comments: 54](https://reddit.com//r/LocalLLaMA/comments/1elbn3q/quantize_123b_mistrallargeinstruct2407_to_35_gb/)): The author quantized the **123B Mistral-Large-Instruct-2407** model from **228.5 GB** to **35.5 GB** using the **EfficientQAT** algorithm, resulting in only a **4% average accuracy degeneration** across **5 zero-shot reasoning tasks**. The quantized model, using **INT2 bits** and a **group size of 64**, was packed using **GPTQ v2** format and uploaded to [HuggingFace](https://huggingface.co/ChenMnZ/Mistral-Large-Instruct-2407-EfficientQAT-w2g64-GPTQ), with the author seeking assistance in converting it to GGUF or EXL2 formats.
    - Users strongly expressed the need for a **GGUF format** version of the quantized model, with multiple comments requesting this conversion from the current **GPTQ v2 format**.
    - There was skepticism about the model's performance, with one user pointing out that the **perplexity increases 100%** and another correcting the accuracy degeneration to **5.4%** instead of 4%.
    - A user attempted to load the model using **Exllamav2 0.1.7** but encountered a **RuntimeError**, suggesting compatibility issues with the current loader for this quantized version.


**Theme 3. Open-Source AI Tools and Multimodal Generation**

- **[Open source Text2Video generation is here! The creators of ChatGLM just open sourced CogVideo.](https://github.com/THUDM/CogVideo)** ([Score: 61, Comments: 4](https://reddit.com//r/LocalLLaMA/comments/1elbdvr/open_source_text2video_generation_is_here_the/)): The creators of **ChatGLM** have open-sourced **CogVideo**, a **text-to-video generation** model. CogVideo can generate **5-second videos** at **24 FPS** and **256x256 resolution** based on text prompts, representing a significant advancement in open-source AI video generation capabilities.
    - **CogVideo** specifications: **6 seconds** long, **8 FPS**, **720x480 resolution**, requires **18GB GPU memory** for inference with SAT or **36GB** with diffusers. Users noted good coherency but slight lagginess, fixable with flowframes.
    - A [ComfyUI wrapper](https://github.com/kijai/ComfyUI-CogVideoXWrapper) for CogVideo has been made available, enhancing its accessibility and integration with existing workflows.
    - The model's [license](https://github.com/THUDM/CogVideo/blob/main/Model_License) includes restrictions on commercial use and prohibits usage that may *"undermine China's national security and national unity"*, raising questions about its open-source status.

## All AI Reddit Recap

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity


**AI Model Developments and Releases**

- **Salesforce's xLAM-1b model**: A 1 billion parameter model that [achieves 70% accuracy in function calling, surpassing GPT 3.5](https://www.reddit.com/r/LocalLLaMA/comments/1dz8g10/salesforce_tiny_giant_xlam1b_model_surpasses_gpt/). Dubbed a "function calling giant" despite its relatively small size.

- **Phi-3 Mini (June) with function calling**: Rubra AI released an updated Phi-3 Mini model [with function calling capabilities](https://www.reddit.com/r/LocalLLaMA/comments/1dzhe38/phi3_mini_june_with_function_calling/), competitive with Mistral-7b v3 and outperforming the base Phi-3 Mini.

**AI Research and Applications**

- **Figure 02**: A [new humanoid robot introduced by Figure AI](https://www.reddit.com/r/singularity/comments/1elfvmt/introducing_figure_02/), showcasing advancements in robotics and AI integration.

- **AI in image generation**: Discussion on [r/StableDiffusion becoming a general hub for open-source image models](https://www.reddit.com/r/StableDiffusion/comments/1ele3ub/this_sub_should_become_the_general_place_for/), similar to how r/LocalLLaMA became a central place for LLMs.

**AI Ethics and Safety**

- **OpenAI safety resignations**: A [humorous post predicting the next OpenAI head of safety will quit on Aug 30](https://www.reddit.com/r/singularity/comments/1em1haa/according_to_new_scaling_laws_the_next_openai/), based on "new scaling laws". This highlights the ongoing challenges in AI safety and ethics.

**AI Impact on Education and Careers**

- **Nick Bostrom on long-term investments**: Bostrom suggests [it may not be worth making long-term investments like college degrees due to short AI timelines](https://www.reddit.com/r/singularity/comments/1em1uc1/nick_bostrom_says_it_may_not_be_worth_making/). This sparked debate about the potential impact of AI on traditional education and career paths.

**AI-Generated Content**

- **Movie posters from a parallel reality**: [AI-generated movie posters](https://www.reddit.com/r/StableDiffusion/comments/1em3etw/movie_posters_from_a_parallel_reality/) created using Flux Pro + SUPIR Upscale, demonstrating the creative potential of AI in visual arts.

**Memes and Humor**

- Various memes and humorous posts related to AI and technology, including [comparisons of AI-generated images](https://www.reddit.com/r/singularity/comments/1elol2v/left_or_right/) and [satirical takes on anti-AI sentiments](https://www.reddit.com/r/singularity/comments/1elhmef/youd_think_that_this_was_made_by_a_17th_century/).


---

# AI Discord Recap

> A summary of Summaries of Summaries

**1. LLM Advancements and Benchmarking**

- **DeepSeek-V2 Outshines GPT-4 on MT-Bench**: [DeepSeek-V2](https://huggingface.co/deepseek-ai/DeepSeek-V2) from DeepSeek AI has rapidly climbed to the top of leaderboards like **ChatbotArena** and **MT-Bench**, outperforming models such as **GPT-4-Turbo** and **Claude 3 Opus** across over 50,000 matchups.
   - Users compared *model performance* on benchmarks like **AlignBench** and **MT-Bench**, with [DeepSeek's announcement](https://x.com/deepseek_ai/status/1787478986731429933) generating excitement.
- **New Models Advance State of the Art**: New open models like **[Granite-8B-Code-Instruct](https://huggingface.co/ibm-granite/granite-8b-code-instruct)** from IBM enhance instruction following for code tasks, while **[DeepSeek-V2](https://huggingface.co/deepseek-ai/DeepSeek-V2)** boasts **236B parameters**.
   - Example: [DeepSeek-V2 announcement](https://x.com/deepseek_ai/status/1787478986731429933).
  
**2. Model Performance Optimization**

- **AQLM and QuaRot Quantize Llama-3-70b**: **Quantization** techniques like **AQLM** and **QuaRot** aim to run massive language models (**LLMs**) like **Llama-3-70b** on individual GPUs while maintaining performance, as seen in the [AQLM project](https://github.com/Vahe1994/AQLM) running on an RTX3090.
   - *Users discussed the potential benefits and tradeoffs of quantization approaches* for optimizing large model inference.
- **DMC Boosts Throughput 370% on H100 GPUs**: Efforts to **boost transformer efficiency** through **Dynamic Memory Compression (DMC)** promise throughput improvements up to **370%** on **H100 GPUs**, according to the [DMC paper](https://arxiv.org/abs/2403.09636) by `@p_nawrot`.
   - Members explored techniques like *fusing CUDA operations with NVIDIA's Thrust library* to maximize GPU utilization during model inference.
- **Thrust Optimizes CUDA Ops Near Bandwidth Limits**: Discussions centered on **optimizing CUDA operations** like fusing element-wise ops, leveraging **NVIDIA's Thrust library** and its `transform` functionality for near-bandwidth-saturating performance.
   - The [Thrust documentation](https://nvidia.github.io/cccl/thrust/api/groups/group__modifying.html#function-for-each) provides insights into these optimization strategies.
  
**3. Fine-tuning Challenges and Prompt Engineering Strategies**

- **Axolotl Wrestles with Prompt Design**: The importance of **prompt design** and correct template usage, including end-of-text tokens, was highlighted for influencing model performance during fine-tuning and evaluation with tools like [Axolotl prompters.py](https://github.com/OpenAccess-AI-Collective/axolotl/blob/3367fca73253c85e386ef69af3068d42cea09e4f/src/axolotl/prompters.py#L47).
   - *Users shared experiences and insights* around prompt engineering challenges to achieve desired results.
- **Logit Bias Tunes Prompts for More Control**: Strategies for **prompt engineering** were discussed, such as splitting complex tasks into multiple prompts and investigating **logit bias** for granular control, following [OpenAI's logit bias guide](https://help.openai.com/en/articles/5247780-using-logit-bias-to-alter-token-probability-with-the-openai-api).
   - Members shared experiences and techniques *to enhance prompt effectiveness through careful engineering*.
- ***RET* Token Boosts Information Retrieval**: Research explored teaching LLMs to use the `<RET>` token for **information retrieval** when uncertain, improving performance on less frequent queries, based on an [ArXiv paper](https://arxiv.org/abs/2404.19705).
   - The community discussed *novel prompt engineering methods* to expand the capabilities of language models.
  


**4. Multimodal AI and Generative Modeling Innovations**

- ****Idefics2 and CodeGemma: Multimodal Marvels****: New multimodal models like **[Idefics2 8B Chatty](https://twitter.com/sanhestpasmoi/status/1787503160757485609)** focus on elevated chat interactions, while **[CodeGemma 1.1 7B](https://twitter.com/reach_vb/status/1786469104678760677)** refines coding abilities.
   - These releases showcase the *rapid progress in multimodal AI capabilities* across various domains.
- ****Phi3 Brings Powerful AI Chatbots to WebGPU****: The **[Phi 3](https://www.reddit.com/r/LocalLLaMA/comments/1cn2zwn/phi3_webgpu_a_private_and_powerful_ai_chatbot/)** model enables powerful AI chatbots to run in browsers via WebGPU, highlighting the *potential for more accessible and private AI interactions*.
   - Community members discussed the implications of this development for user privacy and control.
- ****IC-Light Advances Open Image Relighting****: The open-source **[IC-Light](https://github.com/lllyasviel/IC-Light)** project focuses on improving techniques for **image relighting**, contributing to the growing open ecosystem for generative AI.
   - Members shared insights and resources related to *image manipulation capabilities powered by AI models*.
  

---

# PART 1: High level Discord summaries




## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **LoRA Makes Stable Diffusion Lean**: LoRA models, small versions of Stable Diffusion, modify standard checkpoints to be **10 to 100 times smaller** and can be installed in the stable-diffusion-webui/models/Lora directory.
   - To use these models, simply include the syntax `<lora:filename:1.0>` in your prompts, enhancing your workflow.
- **Pony Model Delivers Sharp Line Art**: The **Pony model** is engineered for clean line art with no shading and works best when combined with style LoRA for optimal results.
   - Users emphasized that applying the Pony model as the base is crucial to achieving the desired aesthetic when using line art style LoRA.
- **ControlNet Transforms Images Like Magic**: ControlNet facilitates converting photos to line art while preserving the original structure, greatly improving image manipulation capabilities.
   - Community members proposed using depth ControlNet or IPAdapter as effective methods for these transformations.
- **Community Drama Erupts in r/stablediffusion**: Discussions about recent managerial changes in the r/stablediffusion subreddit revealed tensions regarding community vs. company-led projects.
   - This introspection fueled a lively dialogue about the **control issues** faced in community dynamics within the AI art space.
- **Skepticism Wins in AI Hardware Debate**: A consensus emerged against using **AMD GPUs** for ML tasks, with suggestions leaning towards **NVIDIA** or alternatives like **Groq** being favored.
   - Participants also touched on the volatile nature of hardware stocks, prompting discussions about future choices for optimizing AI performance.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth Fine-tuning Model Frustrations**: Users are facing **fine-tuning issues** with Unsloth, particularly around models not saving properly and integration challenges with **PPO trainers** requiring the for_inference() method.
   - Many have noted that older versions integrated more smoothly, contributing to ongoing community frustrations.
- **Inconsistent Inference Timing on Llama3.1**: Reports indicate **inconsistent response times** during inference on fine-tuned **Llama3.1**, with improvements seen after repeated calls.
   - Users are advised to run tests to confirm whether initial slowdowns are affecting performance as expected.
- **Exploring Multi-GPU Support in Unsloth**: Unsloth's **multi-GPU support** is in beta, looking to enhance speed and reduce VRAM usage, with testers currently under NDA.
   - Participants are anticipating a paid subscription model following further refinements.
- **Introducing BigLlama 3.1-1T-Instruct**: A new model, [BigLlama-3.1-1T-Instruct](https://huggingface.co/mlabonne/BigLlama-3.1-1T-Instruct), is being trialed as a self-merge of **Meta-Llama**, but users report it is not yet functional with merged weights.
   - Community feedback emphasizes the model's **uselessness** at this stage due to incomplete training.
- **Cost-effective Configuration for LLaMA3**: A request was made for strategies to run **LLaMA3** cost-effectively on **RunPod**, reflecting the community's focus on optimizing deployment costs.
   - Members discussed the challenges of managing resource demands while keeping costs under control.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Google enhances Gemma with Gemma 2 2B**: Google introduced [Gemma 2 2B](https://huggingface.co/collections/google/gemma-2-2b-release-66a20f3796a2ff2a7c76f98f), boasting **2.6B parameters** designed for on-device usage alongside **ShieldGemma** and **Gemma Scope** for advanced functionality.
   - This rollout positions Gemma 2 as a competitive offering in on-device machine learning tools.
- **New Diffusers integration with FLUX**: A member praised the new [Diffusers integration for FLUX](https://huggingface.co/spaces/black-forest-labs/FLUX.1-schnell), enhancing text-to-image generation capabilities significantly.
   - They shared a [gist](https://gist.github.com/sayakpaul/b664605caf0aa3bf8585ab109dd5ac9c) on using FLUX efficiently with limited resources.
- **Launch of Argilla 2.0 for better data management**: [Argilla 2.0](https://huggingface.co/blog/dvilasuero/argilla-2-0) debuted as a powerful AI tool focused on data usability, promising enhanced management features for creators.
   - Community members welcomed the first open synthetic dataset, **magpie-ultra-v0.1**, generated with Llama 3.1 for improved dataset creation.
- **OpenAI promotes Structured Outputs**: OpenAI has published a [blog post](https://openai.com/index/introducing-structured-outputs-in-the-api/) recommending the use of structured outputs in their API without much attribution to previous work.
   - This shift highlights a trend in adopting effective practices while maintaining a lack of acknowledgment for foundational contributions.
- **Dataset for Named Entity Recognition available**: A dataset consisting of **5029 annotated CVs** with IT skills marked using NER is available on [Kaggle](https://www.kaggle.com/datasets/mehyarmlaweh/ner-annotated-cvs).
   - This dataset includes **manually annotated skills** from PDFs and is formatted in **JSON** for use with NLP tools like **Spacy**.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Configuring LM Studio with AnythingLLM**: Users successfully set up **AnythingLLM** with **LM Studio** after troubleshooting file accessibility and hardware limitations affecting performance. One user confirmed success after loading a custom **Gemma v2** model.
   - Several users contributed insights on common pitfalls during the setup process, focusing on the importance of ensuring file paths are correct.
- **Optimizing Performance Settings in LM Studio**: The 'Keep Model in Memory' feature drew mixed reactions, with some users suggesting it should be disabled by default to avoid unnecessary RAM usage. Experts discussed its limited impact on performance, especially for larger models.
   - Users shared experiences, noting that disabling this feature provided a better balance between system resources and model performance.
- **Interest in Audio Transcription Capabilities**: Users expressed a desire to automate audio transcription but noted the lack of direct support for audio inputs in **LM Studio**. Alternatives such as APIs and open-source TTS/STT solutions were discussed for those prioritizing privacy.
   - Some members reported success using specific APIs, while others preferred local solutions to ensure data confidentiality.
- **Exploring Multi-GPU Configurations**: Users sought advice on managing multiple GPUs with **ComfyUI**, exploring scripts to effectively allocate GPU resources. One user proposed a launcher to streamline setting CUDA devices without modifying configuration files.
   - The discussion included suggestions for existing scripts available on GitHub that could simplify multi-GPU setups.
- **Concerns Over Phi-3 Model Support**: Concerns were raised regarding the lack of **Phi-3** model support in **llama.cpp**, which affects interface compatibility such as in **Oobabooga WebUI**. This sparked a broader conversation on recent updates and community reactions.
   - Members noted that the issue might require coordination among developers to ensure seamless integration with the latest models.



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **Gameboy Emulator Simplifies RL**: A detailed setup for a **Gameboy emulator** can be found in the [PufferLib GitHub repository](https://github.com/PufferAI/PufferLib/blob/729003f9cb89845cc1a69a65e5a2431b2d0542bd/pufferlib/environments/pokemon_red/environment.py#L15), streamlining reinforcement learning in game environments.
   - This approach allows users to explore RL concepts without the need for extensive speed optimizations.
- **PyTorch 2.4 Struggles on CUDA 12.4**: Users reported issues with **PyTorch 2.4** on **CUDA 12.4**, noting a drop in performance compared to earlier versions like **CUDA 12.1**.
   - Concerns were raised over compatibility and potential improvements when reverting to previous CUDA versions.
- **ZLUDA 3 Yanked Post AMD Claims**: The author has taken down **ZLUDA 3** following AMD's claim that the permission for its release was invalid, detailed in the [GitHub page](https://github.com/vosen/ZLUDA).
   - This situation has stirred discussions about AMD's role in the development landscape and the implications for open-source contributions.
- **Debate on INT8 Quantization Techniques**: Discussions around **INT8 symmetric quantization** revealed concerns about bias in weight updates when using a scale of **127.5** during training.
   - Members debated the efficacy of full vs restricted range quantization, emphasizing potential challenges in model integrity.
- **Introducing SARATHI for LLM Efficiency**: A new framework, **SARATHI**, addresses inefficiencies in LLM inference by employing chunked-prefills and improved batching strategies.
   - This approach aims to enhance GPU utilization while reducing imbalances in pipeline parallelism during model inference.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **UltraSteer-V0 Dataset Launch**: Nvidia introduced the **UltraSteer-V0 dataset**, featuring **2.3M conversations** with **2.8M turns** and labeled across **9 fine-grained signals** using the **Llama2-13B-SteerLM-RM** reward model.
   - Despite being a **version zero**, it has unique thread continuations thanks to extensive deduplication over **22 days** and is available for access on [Hugging Face](https://huggingface.co/datasets/Avelina/UltraSteer-v0).
- **Challenges in Fine-tuning Insurance Models**: A user queried about experiences fine-tuning models for the **insurance sector**, highlighting challenges specific to this industry.
   - This discussion drew input on necessary adaptations and considerations for applying AI effectively in insurance contexts.
- **Buzz Around Flux AI's Abilities**: Flux AI showcased skills in **text comprehension**, **prompt comprehension**, and **image generation**, sparking excitement among members.
   - Many users praised its capabilities, with some already leveraging its Pro version for enhanced performance.
- **Open Medical Reasoning Tasks Initiative**: Collaboratively led by **Open Life-Science AI**, the **Open Medical Reasoning Tasks** project seeks to compile a robust list of tasks for LLMs in healthcare, inviting contributions from various stakeholders.
   - A member celebrated this collaborative effort, emphasizing the collective impact on advancing AI in the medical field; more details are available on [GitHub](https://github.com/openlifescience-ai/Open-Medical-Reasoning-Tasks).
- **MiniCPM-Llama3-V Model Updates**: Members discussed the latest updates on **MiniCPM-Llama3-V**, which claims improved capabilities for handling **multiple image inputs** and OCR tasks.
   - This sparked initial skepticism, but excitement grew with new examples demonstrating its application and effectiveness.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Web Devs Transition to AI Engineering**: Discussions highlighted the growing transition of **web developers** into **AI engineering** due to high demand and limited ML engineers, with participants sharing insights on adapting skill sets.
   - Members emphasized how web devs are often expected to implement AI projects alongside traditional development duties.
- **OpenAI Faces Leadership Shifts**: A wave of leadership changes at **OpenAI** has raised concerns about the company's future trajectory and stability, leading to a vibrant debate in the community.
   - Participants speculated on the potential implications of these departures on the overall direction of OpenAI.
- **Generative AI Revolutionizes Retail**: **Generative AI** applications are thriving in the retail sector, especially in crafting product descriptions across platforms, with examples stemming from **L'Oreal**.
   - Discussions raised crucial points about evaluating the effectiveness of AI-generated content and the need for better performance metrics.
- **Structured Outputs Feature Debuts in GPT-4o**: **OpenAI** has launched a structured outputs feature in **GPT-4o**, allowing models to adhere to JSON schemas with improved reliability compared to previous models.
   - Community members recognized this advancement as a significant step toward generating more controlled and structured data outputs in AI.
- **Skepticism in Energy-Based Language Modeling**: An anecdote about a meet-up with an **Extropic AI** researcher highlighted a skepticism towards their knowledge in **energy-based language modeling**, questioning their credibility.
   - This exchange stirred a broader discussion about the expertise of newer startups in complex AI domains.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **OpenAI DevDay Goes Global!**: OpenAI is taking **DevDay** on the road this fall with events in **San Francisco**, **London**, and **Singapore**, featuring hands-on sessions and best practices for developers. Participants can engage directly with **OpenAI engineers** to see innovations in action, details can be found [here](https://openai.com/devday/).
   - The event promises a platform for developers to connect globally, sharing insights and redefining practices in AI development.
- **DALL-E 3 Model Shows Results Variability**: Members discussed the **DALL-E 3 model** and the variability in generated results, highlighting comparisons with Llama models and the influence of safety filters. Notably, output quality discrepancies were attributed to the safety measures implemented by OpenAI.
   - The community is analyzing these variances while exploring the nuances of AI generation quality and safety concerns.
- **Search GPT is Available Now!**: **Search GPT** has officially rolled out, generating interest among users regarding its functionalities and applications. Members are actively discussing how they plan to leverage this new feature in their workflows.
   - This roll-out has prompted questions about user experiences and practical implementations of Search GPT.
- **Excitement for Generative AI in Gaming**: Members are thrilled about the potential of generative AI in enhancing gaming experiences, specifically in titles like **BG3** and **Pathfinder**. They envision dynamic NPC interactions stemming from improved AI capabilities.
   - The discussion centered around creating immersive environments where character designs and player choices blend seamlessly.
- **ChatGPT-4o's Updates Spark Questions**: Users noted significant changes in the performance of **ChatGPT-4o**, speculating that it has undergone recent updates. Members are discussing the implications of these changes on output consistency and user experience.
   - Observations about the version `gpt-4o-2024-08-06` have spurred further conversation on what these updates mean for developers and users moving forward.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity AI Technical Issues Surface**: Users reported various **technical issues** with the Perplexity Pro app, including inability to switch LLMs and missing libraries, triggering significant concern over functionality.
   - Some features returned unexpectedly, indicating potential intermittent issues rather than systemic failures.
- **NVIDIA's Blackwell GPUs Hit Delays**: **NVIDIA's Blackwell GPUs** have been delayed due to critical **design flaws** and issues with **CoWoS-L** packaging technology, necessitating a redesign of the processor die.
   - These setbacks are pushing back production timelines, impacting expectations for the next generation of GPUs.
- **Language Model Comparisons Heat Up**: Debates erupted over performance comparisons between **GPT-4o** and **Turbo**, with users expressing mixed experiences, particularly around responsiveness and effectiveness.
   - Some users noted **GPT-4o** struggled with new instructions, garnering calls for a reassessment of LLM capabilities.
- **Exploring Content Recommendation Engines**: A new university project aimed at developing a **content sorting and recommendation engine** caught interest, emphasizing the need for user input in creating a relevant sorting algorithm.
   - Members suggested leveraging **RAG** (retrieval-augmented generation) principles to enhance the project’s effectiveness.
- **API Functionality Under Scrutiny**: Concerns were raised about **API discrepancies**, with users experiencing corrupted data returns leading to doubts about the API's reliability.
   - Additionally, upcoming deprecation of all Perplexity API models by **August 12, 2024** brought attention to required adjustments for future usage.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Novel methods in Mechanistic Anomaly Detection**: The team examined _mechanistic_ methods for anomaly detection in language models using [Neel Nanda's attribution patching technique](https://blog.eleuther.ai/mad_research_update/), but traditional baselines based on activations performed better.
   - They found improved performance by evaluating entire batches rather than individual points, varying success across tasks.
- **Debate on SB1047 AI Safety Act heats up**: Members held a vigorous discussion about SB1047, with concerns that it may stifle innovation while others argue for necessary accountability in AI research.
   - Debaters expressed that the bill's liability provisions could deter open research efforts, indicating a need to balance regulation with innovation.
- **Meta's advancements in distributed AI training**: At [ACM SIGCOMM 2024](https://conferences.sigcomm.org/sigcomm/2024/), Meta showcased their paper on [RDMA over Ethernet for Distributed AI Training](https://dl.acm.org/doi/10.1145/3651890.3672233), focusing on support infrastructure for training models like **LLAMA 3.1 405B**.
   - This presentation underscored the increasing demands in communication spurred by large-scale AI applications.
- **Recap on Sparse Autoencoder (SAE) developments**: Members referenced a [paper on SAE](https://transformer-circuits.pub/2023/monosemantic-features/index.html) along with follow-up research on [scaling SAEs](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html) to stay updated on SAE advancements.
   - They discussed the relevance of SAE notation and shared resources including a [Google document](https://docs.google.com/document/d/1lHvRXJsbi41bNGZ_znGN7DmlLXITXyWyISan7Qx2y6s/edit#heading=h.j9b3g3x1o1z4) that tracks the landscape of these technologies.
- **lm-eval-harness insights and usage**: A user inquired about utilizing **lm-eval-harness** for custom models and received a helpful link to a [self-contained example](https://github.com/state-spaces/mamba/blob/main/evals/lm_harness_eval.py) for adapting the Huggingface model class.
   - Discussion highlighted the inclusion of special tokens like **BOS** and the process for extracting benchmark names from JSON output in evaluation results.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **Managing GPU Memory Issues**: A user reported **out-of-memory errors** with models like **aya** and **nomic-embed-text**, using a machine with **32GB RAM**. It was suggested to switch to **CPU**, but that led to much slower performance.
   - This discussion highlighted the performance trade-offs engineers face when dealing with memory constraints and the challenges of optimizing GPU resources.
- **LangGraph Course Recommendations**: Users discussed various **LangGraph courses**, recommending the **DeepLearning AI course** as a solid option, along with an advanced one on **Udemy**. There's a general sentiment that many beginner-friendly resources exist, but advanced materials are lacking.
   - This points to a need for more comprehensive training at higher levels in the LangGraph ecosystem for practitioners looking to deepen their skills.
- **Collaboration on SQL Chat Agent**: One user sought assistance with developing a **SQL chat agent script**, sparking a collaborative effort from another experienced developer. Scripts and feedback were shared, showcasing community support.
   - This interaction exemplifies the collaborative culture among developers, emphasizing knowledge sharing to improve AI functionalities.
- **New Music Discovery App Launch**: The **mood2music** app was introduced, promising AI-driven music recommendations based on user moods. It is currently building a waitlist and features unique music curation capabilities.
   - The application generates excitement as it prepares for launch, identifying potential engagement with music enthusiasts.
- **AgentGenesis Boosts AI Development**: A member shared **AgentGenesis**, a library providing **copy-paste code snippets** for accelerating **Gen AI applications**. It aims to offer a developer-friendly code library that enhances productivity dramatically.
   - The project invites community contributions and aims to simplify development processes, showcasing the collaborative spirit in the AI developer community.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **John Schulman Leaves OpenAI for Anthropic**: John Schulman announced his departure from OpenAI to focus on AI alignment research at [Anthropic](https://x.com/johnschulman2/status/1820610863499509855), stating a desire for hands-on technical work.
   - He emphasized this choice is personal, noting it reflects ongoing support for alignment at OpenAI despite his exit.
- **GDB's Sabbatical Sparks Speculation**: GDB's decision to take a sabbatical until year-end has led to discussions questioning the reasoning, with concerns about overwork and health issues.
   - Some speculate this break could be essential after intense years focused on AGI development.
- **Debate Rages on AI Alignment Perspectives**: A robust discussion unfolded about differing views on AI alignment, with Schulman favoring a reinforcement learning approach while others argue it transcends traditional methods.
   - This reflects broader concerns on controlling superhuman AI and whether alignment is fundamentally a deep learning problem.
- **Structured Outputs Revolutionize API Handling**: The recent introduction of [Structured Outputs](https://openai.com/index/introducing-structured-outputs-in-the-api/) allows developers consistent schema matches without missing keys.
   - Additionally, developers save **50% on input costs** and **33% on output costs** by switching to the gpt-4o-2024-08-06 model.
- **DALL·E Faces Growing Competition**: Discussion arose whether **DALL·E still holds the title** for best image generation as new rivals come into play, with challenges in making outright comparisons.
   - Members noted the importance of context over intuition when evaluating competitive capabilities.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **GPT-4o-2024-08-06 Now Live**: The release of [GPT-4o-2024-08-06](https://openrouter.ai/models/openai/gpt-4o-2024-08-06) marks a notable update with **significantly reduced pricing** of **50%** for inputs and **33%** for outputs, further enhancing developer accessibility.
   - Notably, the model includes a new 'refusal' field feature, sparking excitement for improved functionality.
- **Gemini Pro 1.5 Suffers Resource Limitation**: Users faced an error stating 'Resource has been exhausted' with **Gemini Pro 1.5**, linked to stringent rate limits enforced by Google.
   - Unfortunately, there is currently no remedy as this is a restriction coming directly from Google.
- **Navigating OpenRouter's API**: Inquiries regarding model purchases led to the understanding that models via the **OpenRouter** require payment per token usage, with new users encouraged to try interfaces like **Lobe Chat** for easier interactions.
   - This approach is intended to streamline access as well as decrease friction for onboarding users.
- **Structured Outputs Boost API Reliability**: OpenAI introduced structured outputs allowing developers to request valid JSON responses directly from the API, enhancing overall reliability and usability.
   - This initiative addresses prior inconsistencies in output formats, aiming for a more standardized interaction across applications.
- **Model Pricing Fluctuations Under Review**: Discussions around the **token limit** discrepancies for **gpt-4o-2024-08-06** surfaced, with the OpenRouter interface showing a lower maximum than OpenAI's documentation.
   - Users await updates to align system capabilities accurately with the latest model specifications.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Join the CodiumAI Webinar on RAG-Enhanced Coding**: A reminder was shared about the upcoming [webinar with CodiumAI](https://lu.ma/ka5xtyqo) focusing on **RAG-augmented coding assistants**. Participants must verify token ownership through their wallet to access the event.
   - The webinar will cover how **Retrieval-Augmented Generation (RAG)** improves contextual awareness in AI-generated code, which is critical for maintaining **high quality** in software development.
- **Building Multi-Agent Systems Using RabbitMQ**: A blog highlights how to create a local multi-agent system with [RabbitMQ](https://www.rabbitmq.com), utilizing tools like [ollama](https://ollama.com) and [qdrant_engine](https://qdrant.tech) through llama-agents. Check out the complete guide [here](https://t.co/IOGpDWkY8A).
   - This set-up facilitates communication between agents and enhances the development experience essential for building robust AI systems.
- **Using HuggingFace Inference API for Embeddings**: The HuggingFace Inference API enables embedding generation using the `TextEmbeddingsInference` class, as detailed in [this example](https://docs.llamaindex.ai/en/stable/examples/embeddings/text_embedding_inference/). It supports parameters like model name and embedding batch size to optimize performance.
   - Users highlighted the efficiency it brings to processing embeddings, essential for training AI models.
- **RAG Performance Insights Shared**: Discussion included insights into how **Retrieval-Augmented Generation** enhances the quality of generated code based on **contextual awareness**. A presentation on an advanced approach using the **LlamaIndex** infrastructure covers practical applications.
   - Attendees can expect to learn about *context-aware generation*, which is critical for developers looking to improve their coding assistants.
- **Llamaparse's Arabic Parsing Issue**: Users reported that Llamaparse struggles with Arabic parsing, producing results in a Left to Right format despite Arabic's Right to Left nature. This raises important questions regarding Llamaparse's handling of language intricacies.
   - This feedback signals a potential area for improvement in accommodating diverse languages in parsing applications.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **LLM Hallucination Index Raises Eyebrows**: The [LLM Hallucination Index](https://www.rungalileo.io/hallucinationindex?utm_medium=paid&utm_source=alpha_signal&utm_campaign=sp) evaluates model fidelity to context, spotlighting concerns as it was named **Word of the Year**.
   - Members debated the index's accuracy for **Command R Plus**, suggesting it misrepresents its open-source status.
- **Open Source Definition Sparks Debate**: There are **disagreements** over the open-source definition in the Hallucination Index, deemed too lenient for just releasing weights.
   - *Additional transparency on datasets and training methods* was emphasized as crucial for genuine open-source status.
- **Mistral's License Under the Microscope**: Members clarified that **Mistral** models are under the **Apache 2.0** license, qualifying them as open source, albeit with dataset access limitations.
   - Discussions revealed that many models are labeled as 'open weights' but lack true open-source characteristics.
- **Command R Plus's Commercial Use Controversy**: **Command R Plus** operates under a Creative Commons Attribution Non Commercial license, rendering it effectively closed-source.
   - The paper's open-source definition drew scrutiny, with members advocating for a clearer standard.
- **Cohere Toolkit Fuels Learning Project**: The **Cohere Toolkit** is employed for a learning initiative in an AI fellowship, focusing on building an **LLM with RAG** over diverse corpora like **recipes** and **legal case notes**.
   - Inquiry arose about transitioning from **Cohere models** to third-party APIs like **OpenAI Chat GPT** or **Gemini 1.5**, hinting at broader functional needs.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **InlineList Defines a New Direction**: The `InlineList` currently lacks **__moveinit__** and **__copyinit__** functionalities, but progress is underway with key features set to merge soon.
   - Members are prioritizing these developments as essential for improving core functionalities.
- **Clarify Mojo Types: List vs. InlinedFixedVector**: `InlinedFixedVector` is crafted for **AnyTrivialRegType** while `List` caters to **CollectionElement**, highlighting their tailored purposes in Mojo.
   - Discussion touched on a **small buffer optimization** under review that may enhance `List` performance.
- **Mojo and Custom Hardware: An Accelerating Topic**: Members debated the potential for **custom accelerators** like PCIe cards with Mojo, questioning support before an open-source release.
   - Concerns about performance emphasized the reliance on **cxl.mem** for effective hardware integration.
- **FPGA and CXL IP Blocks: Hardware Development Insights**: Discussions covered the use of **Xilinx VU13P FPGAs** and the integration of **CXL IP blocks** for hardware optimization projects.
   - One member shared plans to replace kernel usage with custom solutions to enhance overall efficiency.
- **Excitement Builds for Mojo's Open Source Future**: There’s palpable excitement about Mojo's future as an open-source project, especially concerning support for **RISC-V vector extensions**.
   - Members expressed hopes for Mojo to significantly contribute to their projects despite current compatibility limitations.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **John Schulman exits OpenAI for Anthropic**: [John Schulman](https://www.cnbc.com/2024/08/06/openai-co-founder-john-schulman-says-he-will-join-rival-anthropic.html), co-founder of OpenAI, is joining **Anthropic**, an AI startup backed by **Amazon**, following the dissolution of OpenAI's **superalignment team**.
   - This shift may reflect ongoing concerns around ensuring control over advanced AI systems in the changing landscape.
- **Open-source AI struggles with high costs**: The **open-source AI** sector faces significant challenges, particularly high training costs for state-of-the-art models and the difficulty in acquiring necessary preference data.
   - These issues contribute to a bottleneck in the development of competitive open models.
- **Meta's JASCO under scrutiny**: Speculation around **Meta's JASCO** has spiked due to reports of it going 'missing' and possible lawsuits from **Udio** and **Suno**.
   - This rumor could stall Meta's AI advancements as uncertainty looms in the community.
- **Doxxing incident raises privacy concerns**: **Nullbulge** experienced a doxxing incident, igniting discussions on the risks surrounding online privacy and individual reputation.
   - Community members noted potential weaknesses in operational security that might mitigate future risks.
- **Model hits accuracy wall at 270k parameters**: The **270k model** is reportedly encountering an accuracy plateau, achieving only **84% validation accuracy**, signaling diminishing returns with increased parameters.
   - A participant suggested this trend indicates the need for alternative strategies in model design.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Feasibility of tinygrad on Aurora**: Members debated whether it's feasible to run **tinygrad** on **Aurora** due to reliance on Intel GPUs, highlighting their support for tensor core instructions like the **A770s**.
   - Discussion involved expectations of **Aurora's capabilities**, which are projected to exceed **2 ExaFLOPS**, making it potentially the fastest computer ever.
- **Preallocation Techniques for Tensors**: A member suggested that preallocating tensors and assigning slices might resolve tensor manipulation issues, with *George* confirming contiguity resolves the problem.
   - Mapping `Buffer` instances back to `DEFINE_GLOBAL` highlighted clarity needs, as members like *Eigenvector42* expressed uncertainties in the tensor flow.
- **Need for Distributed Computing Features**: Members emphasized the necessity of mature **distributed computing** functionality for tinygrad to fully harness **Aurora's** capabilities.
   - They highlighted that enhancing these capabilities is crucial for better leveraging Aurora's computational power.
- **Dual Support Needed for FP8 NVIDIA Bounty**: A query arose regarding whether support for **E4M3** or **E5M2**, or both, was desired for the FP8 NVIDIA bounty, greeted by *George's* favorable response for both.
   - This indicates a crucial area for future development and backing for NVIDIA's requirements.
- **OpenMP Threading Insights**: Discussion around **CLANG** and **LLVM** threading confirmed usage primarily on a single thread, with enhancement possibilities through **OpenMP** mentioned.
   - Links to respective *tinygrad* GitHub pull requests were shared to inspire contributions and improvements.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Wiseflow revolutionizes information mining**: [Wiseflow](https://github.com/TeamWiseFlow/wiseflow) is a new agile information mining tool that extracts and categorizes concise messages from diverse sources, enhancing data organization.
   - This innovative tool is designed for optimal retrieval in information-heavy environments, addressing current user needs.
- **HybridAGI introduces neuro-symbolic enhancements**: The latest version of [HybridAGI](https://github.com/SynaLinks/HybridAGI) incorporates a neuro-symbolic system centered on graphs, improving RAG (Retrieval-Augmented Generation) functionality.
   - Key features include various notebooks aimed at streamlining usability and enhancing data processing pipelines.
- **LLMs evolve towards AGI with agents**: Research is underway on transitioning **LLMs** to **LLM-based agents**, addressing limitations in autonomy as highlighted in this [study](https://arxiv.org/abs/2408.02479).
   - This underscores the necessity for unified standards to benchmark LLM solutions as agents.
- **Boosting performance with inference compute**: A recent study indicates that increasing generated sample numbers during inference can raise performance, with issue resolution rates improving from **15.9%** to **56%** as seen in the [SWE-bench Lite](https://arxiv.org/abs/2407.21787).
   - This relationship between sample coverage and performance is particularly beneficial for coding and formal proofs.
- **MIPRO often surpasses BootstrapFewShotWithRandomSearch**: In response to queries, it was noted that **MIPRO** performs better than **BootstrapFewShotWithRandomSearch** 'often, but not necessarily'.
   - This points to MIPRO's strong performance while acknowledging variability.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Synthetic Data Generation Strategy**: A member inquired about strategies for **synthetic data generation** to enhance **8b models** on reasoning tasks, particularly **text to SQL** using **Chain of Thought (CoT)** training.
   - They suggested utilizing synthetic instructions before generating SQL queries to potentially improve model performance.
- **QLoRA Configurations for Gemma 2 27b**: Discussion centered around **QLoRA** for **Gemma 2 27b**, with a recommendation to adjust the **learning rate** for compatibility with **Flash Attention**.
   - Members shared intentions to experiment with these modifications which could benefit training.
- **Fine-tuning Context Length Insights**: A member questioned the ability to adjust the context length of a fine-tuned model like **llama2-13b-hf** after setting it to **4k**.
   - Another member confirmed it can be increased or decreased, recommending a stepwise approach for large adjustments to maintain performance.
- **RoPE Scaling for Quick Adjustments**: In relation to the context length topic, there was a suggestion to use **RoPE scaling** for efficient adjustments.
   - It was advised to gradually increase context length for optimal results, particularly for significant changes.
- **BitsAndBytes GitHub Pull Request Mention**: A member emphasized tracking the right branch on **BitsAndBytes GitHub**, referring specifically to pull request **#1220**.
   - This detail could be crucial for anyone involved in recent development or debugging.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **PPO Training Recipe Added to Torchtune**: An end-to-end **PPO training recipe** has been integrated into Torchtune, enabling **RLHF** capabilities. Check out the detailed implementation [here](https://github.com/pytorch/torchtune/pull/1005).
   - *This addition streamlines integration between reinforcement learning and Torchtune's toolkit,* enhancing training options.
- **Qwen2 Models Now Supported**: Support for **Qwen2 models**, including the **7B** model, has been integrated into Torchtune's training recipes with upcoming releases of **1.5B** and **0.5B** models soon. More details can be found [here](https://github.com/pytorch/torchtune/pull/1143).
   - *This expansion opens up more possibilities for model experimentation and tuning within the community.*
- **DPO Support Planned for Llama 3**: Members discussed the potential for supporting **DPO** with the **Llama 3 8B full finetune**, expressing interest in enhancements. Any of the models can be used with the recipes, even without a pre-built configuration.
   - *This suggests an ongoing effort to explore deeper model capabilities.*
- **Refactored PreferenceDataset Enhances Chat Support**: The newly refactored **PreferenceDataset** now supports chat functionalities, as detailed in [Pull Request #1276](https://github.com/pytorch/torchtune/pull/1276). This aligns with the unified **message_transform** pipeline established in previous discussions.
   - *This update appears to significantly improve user interaction with datasets.*
- **Proposal for Dedicated Model Builders Pages**: A member suggested creating a **dedicated page** for each model's builders to accommodate the growing number of models and **multimodal LLMs**. *This would allow us to better explain repetitive details like downloading and configuring models,* consolidating information for users.
   - *The proposal emphasizes the community's need for clearer organizational tools in model management.*



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Troubleshooting Open Interpreter Setup**: Users report issues with setting up **Open Interpreter**, particularly when selecting a local Llama model, often encountering an **openai.APIConnectionError** during execution.
   - *One user reported that their model attempted to download again even after selection.*
- **Inquiry on Open Interpreter's Security Measures**: A member raised concerns about how **Open Interpreter** handles user data, specifically whether it remains on their local machine.
   - They inquired about end-to-end encryption standards and any third-party involvement during communication.
- **Python Compatibility for Open Interpreter**: A member questioned if **Open Interpreter** functions with **Python 3.12**, expressing their beginner status in programming.
   - Another member clarified that current compatibility requires **Python 3.10** or **3.11**.
- **Ollama Model List Command**: To explore available models, a member suggested using the command `ollama list`, noting that each model has specific **VRAM** requirements.
   - Instructions to run models are detailed in the [Ollama documentation](https://github.com/OpenInterpreter/open-interpreter/blob/main/docs/language-models/local-models/ollama.mdx), emphasizing resource availability.
- **API Keys for Remotely Hosted Models**: It was established that an **API key** is essential for accessing paid remotely hosted models, while local models operate on a designated **port**.
   - This highlights the importance of authentication for remote capabilities.



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Llamafile achieves major milestones**: The team continues to advance **Llamafile**, offering offline, accessible LLMs in a single file, much to the excitement of community members.
   - *Community members expressed excitement about the project's potential impact on accessibility.*
- **Mozilla AI community requests feedback for rewards**: The **Mozilla AI community** seeks input through a survey, incentivizing participation with a chance to win a **$25 gift card**.
   - *Members are encouraged to share how Mozilla AI can better support them via community resources.*
- **Celebrate at the sqlite-vec release party**: Everyone is invited to the [sqlite-vec release party](https://discord.com/events/1089876418936180786/1265715263999836210), featuring demos led by core maintainer.
   - *Participants will have the opportunity to try demos and engage directly with the core team,* enhancing their hands-on experience.
- **Engaging discussions in Machine Learning Paper Talks**: Upcoming **Machine Learning Paper Talks** will cover *Communicative Agents* and *Extended Mind Transformers*, hosted by a prominent community member.
   - *These sessions promise to engage attendees with the latest research and invigorating discussions.*
- **Insights from Local AI AMA**: An [AMA](https://discord.com/events/1089876418936180786/1268967945216721079) is set with Local AI's core maintainer, discussing self-hosting alternatives.
   - *This is a prime chance for members to ask questions and explore practical implementations of Local AI.*



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **LinkedIn Engineering revamps their ML platform**: LinkedIn is hosting a live event detailing their engineering team's transformation of the **ML platform** and innovations within it. You can join the discussion [here](https://www.linkedin.com/events/flytepipelinesinactionwithlinke7218669945767776256/theater/).
   - The event emphasizes insights into the latest advancements in **machine learning**, encouraging participants to engage and share their thoughts during the discussion.
- **Live Event brings real-time insights**: Currently ongoing, the event sheds light on pivotal developments in **machine learning** at LinkedIn, showcasing strategies and technologies used by their engineering team.
   - Participants can contribute actively, making it a collaborative venue for those interested in state-of-the-art practices in the field.



---


The **Alignment Lab AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **LLM Finetuning (Hamel + Dan) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **DiscoResearch Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1270099062342947013)** (459 messages🔥🔥🔥): 

> - `LoRA Usage in Stable Diffusion`
> - `Pony Model for Line Art`
> - `ControlNet and Image Transformation`
> - `Community Dynamics in r/stablediffusion`
> - `Hardware Choices for AI` 


- **LoRA Usage in Stable Diffusion**: LoRA models are small Stable Diffusion models that apply changes to standard checkpoint models, making them significantly smaller and more manageable.
   - Users can install LoRA models in the stable-diffusion-webui/models/Lora directory and include them in prompts using the syntax <lora:filename:1.0>.
- **Pony Model for Line Art**: The Pony model is specifically designed for producing clean line art with no shading, and it can be used alongside style LoRA for enhanced outputs.
   - Users discussed the need to use the Pony model as the base when applying the line art style LoRA to achieve desired results.
- **ControlNet and Image Transformation**: ControlNet can be utilized for tasks like converting photos into line art, where it helps maintain the structure of the original image.
   - Users suggested various methods like using depth ControlNet or IPAdapter for effective image transformations in Stable Diffusion.
- **Community Dynamics in r/stablediffusion**: Users reflected on the managerial changes and prior drama within the r/stablediffusion subreddit, which led to a community uproar over control issues.
   - The conversation highlighted the ongoing dynamics of community-led projects vs. company-led initiatives in the AI art space.
- **Hardware Choices for AI**: Discussion about hardware preferences indicated a shared skepticism towards AMD GPUs for machine learning tasks, with recommendations leaning towards NVIDIA or alternatives like Groq.
   - The volatility of hardware stocks and technologies sparked conversation about future choices for AI performance.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.krea.ai/apps/image/realtime">KREA</a>: no description found</li><li><a href="https://www.xkcd.com/2347/">Dependency</a>: no description found</li><li><a href="https://www.stablediffusiontutorials.com/2024/08/flux-installation.html?m=1">FLUX: Installation with Workflow is Here</a>: no description found</li><li><a href="https://x.com/SomniumSpace/status/1820930960239497445">Tweet from Somnium Space (@SomniumSpace)</a>: We are delighted to publish this incredible full Keynote Speech by Robert Scoble (@Scobleizer) which he gave at #SomniumConnect2024✨  What will #AI bring to humanity in the next 10 years? How will thi...</li><li><a href="https://huggingface.co/black-forest-labs">black-forest-labs (Black Forest Labs)</a>: no description found</li><li><a href="https://comfyanonymous.github.io/ComfyUI_examples/flux/">Flux Examples</a>: Examples of ComfyUI workflows</li><li><a href="https://huggingface.co/THUDM/CogVideoX-2b">THUDM/CogVideoX-2b · Hugging Face</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=sMMYSmDHAY8">ComfyUI: Imposing Consistent Light (IC-Light Workflow Tutorial)</a>: The video focuses on implementing IC-Light in Comfy UI, specifically for product photography. IC-Light is based on SD1.5, and we use a reference background a...</li><li><a href="https://x.com/0xkarmatic/status/1820618875517685976">Tweet from Karma (@0xkarmatic)</a>: Wow, Greg is also taking a leave of absence.</li><li><a href="https://old.reddit.com/r/StableDiffusion/comments/1ekolfd/cfg_how_it_works_in_nonflux_models_vs_flux_code/```">CFG: how it works in non-Flux models vs Flux (code examples)</a>: The 'guidance' value for flux is a simple numeric input that gets fed into the model. BFL introduced this at distilation time by generating an...</li><li><a href="https://github.com/vosen/ZLUDA">GitHub - vosen/ZLUDA: CUDA on ??? GPUs</a>: CUDA on ??? GPUs. Contribute to vosen/ZLUDA development by creating an account on GitHub.</li><li><a href="https://civitai.com/models/596934/line-art-style-sdxl-pony">Line Art Style [SDXL Pony] - V1 | Stable Diffusion LoRA | Civitai</a>: LINE ART STYLE This is a style LoRA meant to mimic line art, specifically art with little to no shading/shadows in order to get clean black lines o...</li><li><a href="https://www.youtube.com/watch?v=_kctwd4w7R0">Good Vibrations (Official Music Video)</a>: REMASTERED IN HD!Official Music Video for Good Vibrations performed by Marky Mark and The Funky Bunch.#MarkyMark #GoodVibrations #Remastered</li><li><a href="https://civitai.com/models/257749/pony-diffusion-v6-xl">Pony Diffusion V6 XL - V6 (start with this one) | Stable Diffusion Checkpoint | Civitai</a>: Pony Diffusion V6 is a versatile SDXL finetune capable of producing stunning SFW and NSFW visuals of various anthro, feral, or humanoids species an...</li><li><a href="https://stable-diffusion-art.com/lora/#Step_1_Install_the_LoRA_model">What are LoRA models and how to use them in AUTOMATIC1111 - Stable Diffusion Art</a>: LoRA models are small Stable Diffusion models that apply tiny changes to standard checkpoint models. They are usually 10 to 100 times smaller than checkpoint
</li>
</ul>

</div>
  

---



### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1270095045319459039)** (105 messages🔥🔥): 

> - `Unsloth Fine-tuning Issues`
> - `Model Inference Timing`
> - `Pretraining vs. Continued Pretraining`
> - `Multi-GPU Support Development`
> - `Resources for Learning LLM Inference` 


- **Unsloth Fine-tuning Issues**: Users are experiencing issues with fine-tuned models in Unsloth, such as models not saving properly and challenges integrating them into PPO trainers, which now require the for_inference() method for output.
   - Community members noted that previous versions worked better with PPO trainers, leading to frustration over new requirements and functionality.
- **Model Inference Timing**: Inconsistent response times when running inference on fine-tuned Llama3.1 are reported, with longer initial load times improving after repeated calls.
   - Users are advised to conduct tests to verify if this temporary slowness is indeed the reason for the delays.
- **Pretraining vs. Continued Pretraining**: Clarifications were made regarding the difference between pretraining and continued pretraining, with the community acknowledging the confusion surrounding terminology.
   - This led to discussions about the importance of understanding these concepts when working with language models.
- **Multi-GPU Support Development**: Multiple GPU support for Unsloth is currently in beta, with plans for future release featuring enhancements in VRAM reduction and speed.
   - Testers are currently under NDA while the feature is being refined for a later paid subscription release.
- **Resources for Learning LLM Inference**: Community members shared a link to a guide on generative AI, which includes high-level summaries but noted a lack of detailed resources on inference.
   - Users expressed appreciation for available resources while seeking more specific information on inference techniques.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp?usp=sharing#scrollTo=FqfebeAdT073">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1tEd1FrOXWMnCU9UIvdYhs61tkxdMuKZu?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://guide.repleteai.com">Nextra: the next docs builder</a>: Nextra: the next docs builder</li><li><a href="https://colab.research.google.com/drive/164cg_O7SV7G8kZr_JXqLd6VC7pd86-1Z#scrollTo=PoPKQjga6obN.">Google Colab</a>: no description found</li><li><a href="https://huggingface.co/kalomaze/Mistral-7b-MoEified-8x">kalomaze/Mistral-7b-MoEified-8x · Hugging Face</a>: no description found</li><li><a href="https://x.com/OpenAIDevs/status/1820876430764634115">Tweet from OpenAI Developers (@OpenAIDevs)</a>: Introducing Structured Outputs in the API—model outputs now adhere to developer-supplied JSON Schemas.  https://openai.com/index/introducing-structured-outputs-in-the-api/</li><li><a href="https://huggingface.co/collections/unsloth/load-4bit-models-4x-faster-659042e3a41c3cbad582e734">Load 4bit models 4x faster - a unsloth Collection</a>: no description found</li><li><a href="https://huggingface.co/collections/unsloth/4bit-instruct-models-6624b1c17fd76cbcf4d435c8">4bit Instruct Models - a unsloth Collection</a>: no description found</li><li><a href="https://huggingface.co/unsloth">unsloth (Unsloth AI)</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1270306636795346975)** (10 messages🔥): 

> - `BigLlama 3.1`
> - `Pokerole Pokémon Prompt`
> - `Game discussions` 


- **Introducing BigLlama 3.1-1T-Instruct**: A user shared a link to the [BigLlama-3.1-1T-Instruct](https://huggingface.co/mlabonne/BigLlama-3.1-1T-Instruct), which is an experimental self-merge of Meta-Llama using [mergekit](https://github.com/cg123/mergekit). This model is positioned as a successor to previous versions with a focus on producing a sensible model despite being a work in progress.
   - Another user highlighted that currently, the model is somewhat **useless** as it hasn't been trained with its merged weights yet.
- **Excitement for Pokerole Pokémon Game**: A user shared a [link to a Pokerole Pokémon prompt](https://www.rpgprompts.com/post/pokerole-pok%C3%A9mon-chatgpt-prompt) that offers a creative way to play Pokémon with an AI Game Master. This prompt allows for engaging gameplay through capturing, training, and battling Pokémon, capturing the essence of the franchise.
   - Users expressed enthusiasm, with one noting, *wait this is actually really good*, indicating a positive reception of the game prompt.
- **Casual Gaming Conversation**: A discussion emerged around members playing games, with references to **Minecraft** and **Pokémon** prompting further dialogue on their gaming experiences. Users interacted playfully, making jokes about game-related prompts and experiences in their conversations.
   - This lighthearted banter around games reflects a community of shared interests in gaming and AI's role in enhancing such experiences.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/mlabonne/BigLlama-3.1-1T-Instruct">mlabonne/BigLlama-3.1-1T-Instruct · Hugging Face</a>: no description found</li><li><a href="https://www.rpgprompts.com/post/pokerole-pok%C3%A9mon-chatgpt-prompt">Pokémon RPG - ChatGPT Prompt </a>: This prompt invokes an AI-crafted Game Master, guiding you through the vibrant and exciting world of Pokémon, inspired by the adventure-filled regions familiar to fans of the franchise. Engage in capt...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1270118569652260924)** (162 messages🔥🔥): 

> - `Llama Model Training`
> - `Colab Pro Limitations`
> - `GGUF File Usage`
> - `Ollama Integration`
> - `Model Exporting` 


- **Llama Model Training Discussions**: Users discussed various aspects of training Llama models, focusing on fine-tuning processes and integration with platforms like Ollama.
   - Many faced challenges in running their fine-tuned models and sought advice on configurations needed for successful execution.
- **Colab Pro Limitations Encountered**: The need for Colab Pro to access terminal features was a major concern, as many users intended to share model training knowledge with those without access to paid services.
   - The terminal was identified as necessary for running commands related to Ollama, causing frustration for users trying to utilize free resources.
- **GGUF File Conversion Process**: Several users inquired about generating GGUF files necessary for using models with Gpt4All, indicating they were still learning about the process.
   - Instructions were shared on where to find the .gguf file within the training notebook to facilitate the integration with Gpt4All.
- **Ollama Integration and Usage**: Guidance was provided on how to serve and interact with models using Ollama, emphasizing the use of subprocess commands in Python for execution.
   - Users discussed steps for serving the model locally and querying it using REST API via curl commands, demonstrating a workflow for model interaction.
- **Exporting Trained Models**: Participants explored options for exporting models after training, particularly in relation to running them on local setups with Ollama.
   - Suggestions included serving the model and interacting with it without needing Colab Pro, focusing on using local resources for model deployment.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp?usp=sharing#scrollTo=FqfebeAdT073">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1aqlNQi7MMJbynFDyOQteD2t0yVfjb9Zh?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1T-YBVfnphoVc8E2E854qF3jdia2Ll2W2?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://www.runpod.io/serverless-gpu">Serverless GPU Endpoints for AI Inference</a>: Run machine learning inference at scale with RunPod Serverless GPU endpoints.</li><li><a href="https://huggingface.co/docs/datasets/v2.20.0/loading#:~:text=full%20offline%20mode.-,Slice%20splits,-You%20can%20also>">Load</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[community-collaboration](https://discord.com/channels/1179035537009545276/1180144489214509097/1270381106402820237)** (1 messages): 

> - `LLaMA3 configuration`
> - `Cost-effective model running` 


- **Seeking Cost-effective LLaMA3 Configuration**: A member requested suggestions for the configuration required to run the **LLaMA3** model on **RunPod** in a cost-effective manner.
   - This inquiry highlights the ongoing interest in optimizing model deployment costs among the community.
- **Challenges in Cost Management for LLaMA3**: Members discussed the challenges in balancing performance and cost when running models like **LLaMA3** on various platforms.
   - Several members shared past experiences where costs exceeded expectations due to unforeseen resource demands.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/)** (1 messages): 

vvelo: https://fxtwitter.com/reach_vb/status/1820493688377643178
  

---



### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1270493009590616085)** (1 messages): 

> - `Gemma 2 2B`
> - `Diffusers integration with FLUX`
> - `Argilla 2.0`
> - `Whisper Generations`
> - `llm-sagemaker Terraform Module` 


- **Google expands Gemma with Gemma 2 2B**: Google introduced [Gemma 2 2B](https://huggingface.co/collections/google/gemma-2-2b-release-66a20f3796a2ff2a7c76f98f), adding a new model with **2.6B parameters** for on-device use, enhancing the existing Gemma offering.
   - They also launched **ShieldGemma**, a set of safety classifiers, and **Gemma Scope**, a suite of sparse autoencoders for additional functionality.
- **Exciting Diffusers integration with FLUX**: A member highlighted the new [Diffusers integration for FLUX](https://huggingface.co/spaces/black-forest-labs/FLUX.1-schnell), praising its advanced capabilities in text-to-image generation.
   - They provided a [gist](https://gist.github.com/sayakpaul/b664605caf0aa3bf8585ab109dd5ac9c) on running FLUX with limited resources, promoting accessibility to a wider audience.
- **Launch of Argilla 2.0, a data-centric tool**: [Argilla 2.0](https://huggingface.co/blog/dvilasuero/argilla-2-0) was unveiled as a robust tool for AI creators, focusing on improving data management and usability.
   - In addition, the community also welcomed the first open synthetic dataset powered by Llama 3.1, titled **magpie-ultra-v0.1**, which aims to elevate dataset creation standards.
- **150% faster Whisper Generations!**: A significant improvement has been noted with Whisper generations now operating **150% faster** using Medusa heads, minimizing any drops in accuracy.
   - Members expressed excitement over the implications of integrating Medusa heads with ASR systems, highlighting its potential for the future.
- **llm-sagemaker Terraform module simplifies deployment**: The new [llm-sagemaker](https://registry.terraform.io/modules/philschmid/llm-sagemaker/aws/latest) Terraform module allows for straightforward deployment of open LLMs to AWS SageMaker, enhancing production capabilities.
   - This module supports popular models like Llama 3 and Mistral, and includes customizable configurations, making it highly accessible for developers.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/blog/gemma-july-update#use-with-llamacpp)">Google releases Gemma 2 2B, ShieldGemma and Gemma Scope</a>: no description found</li><li><a href="https://x.com/reach_vb/status/1819023974283518223)">Tweet from Vaibhav (VB) Srivastav (@reach_vb)</a>: Gemma 2 2B running in a browser, powered by WebLLM & WebGPU! 🔥  100% local & on-device  In less than 24 hours, we&#39;ve already got the model to the edge! ⚡  Try it out on an HF space below:</li><li><a href="https://x.com/reach_vb/status/1819469088890261748)">Tweet from Vaibhav (VB) Srivastav (@reach_vb)</a>: Gemma 2 2B running in a free Google Colab! 🤗  Powered by transformers! ⚡</li><li><a href="https://x.com/ggerganov/status/1818699785152397592)">Tweet from Georgi Gerganov (@ggerganov)</a>: Simple instructions to get started with the latest Gemma 2 models + llama.cpp  https://huggingface.co/blog/gemma-july-update#use-with-llamacpp</li><li><a href="https://x.com/RisingSayak/status/1819299449966833972)">Tweet from Sayak Paul (@RisingSayak)</a>: You should have already gone bonkers by now with @bfl_ml&#39;s FLUX release. What a model, eh!   I am getting back to Twitter after some sprinting with my mates @_DhruvNair_, @YiYiMarz, and @multimoda...</li><li><a href="https://x.com/gabrielmbmb_/status/1819398254867489001)">Tweet from Gabriel Martín Blázquez (@gabrielmbmb_)</a>: Dropping magpie-ultra-v0.1, the first open synthetic dataset built with Llama 3.1 405B.  Created with distilabel, it&#39;s our most advanced and compute-intensive pipeline to date.  https://huggingfac...</li><li><a href="https://x.com/reach_vb/status/1820560137892835369)">Tweet from Vaibhav (VB) Srivastav (@reach_vb)</a>: 150% faster Whisper generations w/ medusa heads! 🔥  Built on top of Transformers with minimal drop in accuracy.  Quite exciting area of research, Medusa heads are proven to be incredibly fast for LLM...</li><li><a href="https://x.com/mervenoyann/status/1818613425859145772)">Tweet from merve (@mervenoyann)</a>: Shipped: new task guide on Vision Language Models and freshly updated Depth Estimation task guide on @huggingface transformers docs ⛴️📦  👉🏻 Read about VLMs, how to stream, quantization and more 👉�...</li><li><a href="https://x.com/_philschmid/status/1820360144334496064)">Tweet from Philipp Schmid (@_philschmid)</a>: Excited to announce “llm-sagemaker” a new Terraform module to easily deploy open LLMs from @huggingface  to @awscloud SageMaker real-time endpoints! 👀 Infrastructure as Code (IaC) tools are crucial f...</li><li><a href="https://x.com/mervenoyann/status/1818675981634109701)">Tweet from merve (@mervenoyann)</a>: SAMv2 is just mindblowingly good 😍  Learn what makes this model so good at video segmentation, keep reading 🦆⇓</li><li><a href="https://x.com/DbrxMosaicAI/status/1818407826852921833)">Tweet from Databricks Mosaic Research (@DbrxMosaicAI)</a>: For our StreamingDataset users: We&#39;re thrilled to announce support for storing MDS datasets in @huggingface. S/O to @orionweller for the contribution!  Check out the docs here: https://docs.mosaic...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1270097797437460544)** (239 messages🔥🔥): 

> - `Hugging Face Datasets Issues`
> - `School Experiences`
> - `Image Generation Tools`
> - `Thesis and Dataset Handling`
> - `Learning Resources for AI` 


- **Hugging Face Datasets Issue Resolution**: A user reported issues loading datasets with multiple JSON lines files, prompting discussion about potential workarounds and hard coding features for better schema recognition.
   - Other users provided insights into using Parquet format for easier access and liability to schema inference while discussing the benefits of small file chunks.
- **Mixed Feelings About School**: A member expressed relief that school was over after a challenging first day, mentioning their teacher's unrelated personal stories during class.
   - Another user reflected positively on remote classes during lockdown, indicating a preference for that style of learning.
- **Exploration of AI Image Generation**: A user discovered their sibling generating images of cats using Meta AI's application, raising concerns about the implications of these tools.
   - This led to discussions about the user’s feelings toward the use of AI for creative tasks and how it may affect social dynamics.
- **Thesis Project Discussions**: A user discussed their thesis project involving dataset curation, sharing strategies for keeping datasets manageable and extensible for future users.
   - They mentioned the importance of making the dataset easy to use, highlighting their focus on usability despite working with a limited portion of the dataset.
- **Learning Resources in AI**: Newcomers to the AI field were directed towards different resources for learning about models, emphasizing the importance of familiarization with Hugging Face tools and terminology.
   - Members encouraged exploring existing models and datasets available on the platform to aid their understanding and practical applications.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/learn">Hugging Face - Learn</a>: no description found</li><li><a href="https://huggingface.co/spaces/fffiloni/audio-to-spectrogram">Audio To Spectrogram - a Hugging Face Space by fffiloni</a>: no description found</li><li><a href="https://huggingface.co/docs/hub/repositories-recommendations#sharing-large-datasets-on-the-hub">Repository limitations and recommendations</a>: no description found</li><li><a href="https://huggingface.co/THUDM/CogVideoX-2b">THUDM/CogVideoX-2b · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/learn/ml-for-3d-course">Welcome to the 🤗 Machine Learning for 3D Course - Hugging Face ML for 3D Course</a>: no description found</li><li><a href="https://huggingface.co/docs/datasets/v2.20.0/dataset_script#add-dataset-attributes)">Create a dataset loading script</a>: no description found</li><li><a href="https://huggingface.co/spaces/fffiloni/spectrogram-to-music">Riffusion • Spectrogram To Music - a Hugging Face Space by fffiloni</a>: no description found</li><li><a href="https://github.com/buaacyw/MeshAnythingV2">GitHub - buaacyw/MeshAnythingV2: From anything to mesh like human artists. Official impl. of &quot;MeshAnything V2: Artist-Created Mesh Generation With Adjacent Mesh Tokenization&quot;</a>: From anything to mesh like human artists. Official impl. of &quot;MeshAnything V2: Artist-Created Mesh Generation With Adjacent Mesh Tokenization&quot; - buaacyw/MeshAnythingV2</li><li><a href="https://github.com/huggingface/datasets/issues/7092">load_dataset with multiple jsonlines files interprets datastructure too early · Issue #7092 · huggingface/datasets</a>: Describe the bug likely related to #6460 using datasets.load_dataset(&quot;json&quot;, data_dir= ... ) with multiple .jsonl files will error if one of the files (maybe the first file?) contains a full...</li><li><a href="https://github.com/SonyCSLParis/NeuralDrumMachine/tree/master">GitHub - SonyCSLParis/NeuralDrumMachine</a>: Contribute to SonyCSLParis/NeuralDrumMachine development by creating an account on GitHub.</li><li><a href="https://huggingface.co/models">Models - Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/docs/hub/en/spaces-overview">Spaces Overview</a>: no description found</li><li><a href="https://huggingface.co/spaces">Spaces - Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/spaces/launch">Spaces Launch – Hugging Face</a>: no description found</li><li><a href="https://github.com/huggingface/transformers/issues">Issues · huggingface/transformers</a>: 🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX. - Issues · huggingface/transformers
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1270265570016624671)** (3 messages): 

> - `Linear Algebra for 3D Video Analysis`
> - `Sharing Resources` 


- **Exploring Linear Algebra for 3D Video Analysis**: A member expressed interest in learning about **linear algebra** specifically for **3D video analysis**.
   - This highlights a keen interest in the mathematical foundations necessary for processing and analyzing 3D visual data.
- **Request for Blogs on Linear Algebra**: Another member requested suggestions for **high-quality blogs or articles** related to linear algebra for their studies.
   - *Sharing valuable resources* can greatly enhance the learning experience for anyone delving into complex mathematical topics.
- **Call to Share Resources**: A member encouraged others to spread the word about the gathered resources after expressing gratitude for reading.
   - This reflects a community spirit of **knowledge-sharing**, fostering a space where information can be exchanged freely.


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1270252132674703493)** (4 messages): 

> - `Image Synthesis with Transformers`
> - `Integrating Graphs with LLMs` 


- **High-Resolution Image Synthesis using Transformers**: A discussion highlighted the use of **transformers** for synthesizing **high-resolution images**, focusing on **latent representations** and a **context-rich vocabulary** codebook.
   - *Conditioned image synthesis* techniques were emphasized as valuable for enhancing image quality.
- **New Method to Integrate Graphs with LLMs**: A member shared a [link to a method](https://arxiv.org/pdf/2405.20684v1) that integrates **graphs** with **LLMs**, noting its similarity to an approach proposed at ICML.
   - This presents an intriguing advancement in *graph integration* methodologies for language models.
- **Exploring Another Novel Graph Integration Method**: Another member posted a link to [this paper](https://arxiv.org/pdf/2402.03973) discussing additional strategies for integrating **graphs** with **LLMs**.
   - The emphasis is on expanding the toolkit for improving model compatibility with graph structures.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1270228072410513409)** (5 messages): 

> - `Unity ML-Agents Training`
> - `Embodied Agent Platform`
> - `Talking Head Synthesis`
> - `Bilateral Reference for Image Segmentation` 


- **Unity ML-Agents Training with Multi-threading**: A member shared a [YouTube video](https://youtube.com/live/XOFMpZsYeXo?feature=share) demonstrating part 2 of their SAC agent training using Unity 6 ML-Agents, highlighting the addition of **multi-threaded support for CUDA** training.
   - They also mentioned introducing a **boredom cooldown pot** to encourage the agent to pick new vectors once a **certain boredom threshold** is reached.
- **Embodied Agent Platform Development**: A project page for an **embodied agent platform** was shared, featuring agents that can chat, understand instructions, and perform tasks in a 3D environment; check it out on [GitHub](https://github.com/thunlp/LEGENT).
   - An **online demo** is also available at [Hugging Face](https://huggingface.co/spaces/LEGENT/LEGENT) to showcase its capabilities.
- **Talking Head Synthesis via AniTalker**: A link was shared to a [talking head synthesis project](https://huggingface.co/spaces/Delik/Anitalker), which is a port of the **AniTalker** GitHub repository that focuses on animating talking faces.
   - The official **GitHub repository** can be found [here](https://github.com/X-LANCE/AniTalker), detailing its application in diverse facial motion encoding.
- **BiRefNet for Image Segmentation**: A member highlighted their involvement in open-sourcing [BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet) designed for **high-resolution dichotomous image segmentation**, showcasing its superior performance to **RMBG1.4**.
   - Additional resources include links to the [arXiv paper](https://arxiv.org/pdf/2401.03407) and a demonstration on Hugging Face Spaces, underscoring its **SOTA capabilities**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/ZhengPeng7/BiRefNet">ZhengPeng7/BiRefNet · Hugging Face</a>: no description found</li><li><a href="https://github.com/thunlp/LEGENT">GitHub - thunlp/LEGENT: Open Platform for Embodied Agents</a>: Open Platform for Embodied Agents. Contribute to thunlp/LEGENT development by creating an account on GitHub.</li><li><a href="https://huggingface.co/spaces/LEGENT/LEGENT">LEGENT - a Hugging Face Space by LEGENT</a>: no description found</li><li><a href="https://huggingface.co/spaces/Delik/Anitalker">Anitalker - a Hugging Face Space by Delik</a>: no description found</li><li><a href="https://github.com/X-LANCE/AniTalker">GitHub - X-LANCE/AniTalker: [ACM MM 2024] This is the official code for &quot;AniTalker: Animate Vivid and Diverse Talking Faces through Identity-Decoupled Facial Motion Encoding&quot;</a>: [ACM MM 2024] This is the official code for &quot;AniTalker: Animate Vivid and Diverse Talking Faces through Identity-Decoupled Facial Motion Encoding&quot; - X-LANCE/AniTalker</li><li><a href="https://youtube.com/live/XOFMpZsYeXo?feature=share">Unity ML-Agents | Live Agent training from Scratch | Part 2</a>: a quick sac agent trainer in a 3d voxel world
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1270468951608131585)** (5 messages): 

> - `OpenAI's Structured Outputs`
> - `LLMs and Reasoning`
> - `Attention Mechanisms in LLMs` 


- **OpenAI promotes Structured Outputs**: OpenAI has published a [blog post](https://openai.com/index/introducing-structured-outputs-in-the-api/) recommending the use of structured outputs in their API without much attribution to previous work.
   - This shift highlights a trend in adopting effective practices while maintaining a lack of acknowledgment for foundational contributions.
- **LLMs struggle with real reasoning**: A member discussed their belief that LLMs don't truly 'reason' as humans do, positing that their reasoning approach likely complicates retrieval tasks.
   - This comparison likens LLM interactions to an Uber driver navigating familiar routes rather than instant teleportation between points.
- **Draft Tokens as Scratchpads for LLMs**: One theory suggests that LLMs require tokens as 'scratchpads' for reasoning, proposing that introducing draft tokens can effectively increase their reasoning capacity.
   - This insight relates to a paper showing performance improvements by prefixing prompts with extra tokens, which enhance memory storage capabilities.
- **Attention Mechanisms and KV-cache**: There’s a notion that replacing linear layers with external databases could enhance LLM reasoning, though recent tests indicate that dismissing KV-cache diminishes performance.
   - This underscores the crucial role of KV-cache in maintaining reasoning effectiveness in LLMs during tasks.
- **Expanding Reasoning Steps in LLMs**: To facilitate reasoning within LLMs, extending token limits is seen as a simpler solution than increasing model depth, which requires retraining.
   - This approach suggests that adding more draft tokens can allow for additional transformation steps without the complications of model modifications.


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1270265971914964993)** (4 messages): 

> - `Depth Estimation`
> - `CVPR 2022 Papers`
> - `Code Implementations` 


- **Depth Estimation Paper from CVPR 2022**: A member shared a link to the paper titled *Depth Estimation by Combining Binocular Stereo and Monocular Structured-Light* presented at CVPR 2022, which can be found [here](https://openaccess.thecvf.com/content/CVPR2022/papers/Xu_Depth_Estimation_by_Combining_Binocular_Stereo_and_Monocular_Structured-Light_CVPR_2022_paper.pdf).
   - This paper may provide insights into advanced techniques in depth estimation combining different methodologies.
- **Inquiry for Code Implementation**: A member inquired about the availability of a code implementation for the aforementioned depth estimation paper.
   - No specific code resources or links were shared in response to this query.


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1270340412208578560)** (2 messages): 

> - `NER Annotated CV Dataset`
> - `Identifying Relevant JSON Files` 


- **Dataset for Named Entity Recognition available**: A member shared a dataset consisting of **5029 annotated CVs** with IT skills marked using NER, available on [Kaggle](https://www.kaggle.com/datasets/mehyarmlaweh/ner-annotated-cvs).
   - The dataset includes **manually annotated skills** from PDFs and is formatted in **JSON** for use with NLP tools like **Spacy**.
- **Finding relevant JSON files for questions**: Another member described having a fixed dataset with over **20,000 JSON files** and is looking to identify the most relevant **5 file IDs** for answering questions generated from other files.
   - They have used **keyword search** and **semantic search** with Elastic Search and the **s-bert embedding model** and are seeking advice on the best method to refine their search.



**Link mentioned**: <a href="https://www.kaggle.com/datasets/mehyarmlaweh/ner-annotated-cvs">NER Annotated CVs</a>: This dataset includes 5029 annotated curriculum vitae (CV), marked with IT skill

  

---



### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1270099360310628557)** (157 messages🔥🔥): 

> - `Using LM Studio with AnythingLLM`
> - `Model performance and settings`
> - `Audio transcription capabilities`
> - `Multi-GPU configurations`
> - `Phi-3 model support issues` 


- **Users configuring LM Studio with AnythingLLM**: Users discussed setting up AnythingLLM with LM Studio, identifying issues with file accessibility and hardware limitations affecting performance.
   - After troubleshooting, one user confirmed success after loading a custom Gemma v2 model.
- **Performance settings and optimizations**: Discussion focused on the 'Keep Model in Memory' feature, with some users suggesting it may not impact performance significantly and should be disabled by default.
   - Experts weighed in on the usefulness of this feature, particularly in relation to RAM usage in the context of larger models.
- **Audio transcription capabilities in LM Studio**: Users expressed interest in automating audio transcription using LM Studio, though it was clarified that the direct support for audio inputs is lacking.
   - Alternatives discussed included using some APIs and TTS/STT solutions, though many prefer offline and open-source options for privacy.
- **Multi-GPU configurations in ComfyUI**: There were inquiries about how to utilize multiple GPUs with ComfyUI, with users exploring various scripts and settings to manage GPU resources effectively.
   - One user suggested creating a launcher to set CUDA devices, enhancing their workflow without needing to alter configuration files.
- **Concerns regarding Phi-3 model support**: A user raised concerns about the lack of support for Phi-3 models in llama.cpp and its impact on other interfaces like Oobabooga WebUI post-updates.
   - This prompted a discussion about the changes in model support and community reactions to recent updates.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/docs/text-generation-inference/en/conceptual/flash_attention">Flash Attention</a>: no description found</li><li><a href="https://huggingface.co/legraphista/internlm2_5-20b-chat-IMat-GGUF">legraphista/internlm2_5-20b-chat-IMat-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/spaces/DontPlanToEnd/UGI-Leaderboard">UGI Leaderboard - a Hugging Face Space by DontPlanToEnd</a>: no description found</li><li><a href="https://tenor.com/view/money-dollars-cash-rich-shut-up-and-take-my-money-gif-3555042">Shut Up! GIF - Money Dollars Cash - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://reddit.com/r/stableDiffusion/comments/1e">Reddit - Dive into anything</a>: no description found</li><li><a href="https://huggingface.co/docs/hub/gguf">GGUF</a>: no description found</li><li><a href="https://reddit.com/r/stableDiffusion/comments/1el79h3/flux_can_be_run_on_a_multigpu_configuration/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/5021">ggml : add Flash Attention by ggerganov · Pull Request #5021 · ggerganov/llama.cpp</a>: ref #3365 Setting up what&amp;#39;s needed for Flash Attention support in ggml and llama.cpp The proposed operator performs: // new res = ggml_flash_attn(ctx, q, k, v, kq_mask, kq_scale);  // fused sc...</li><li><a href="https://openwebui.com/">Open WebUI</a>: no description found
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1270107257018912950)** (59 messages🔥🔥): 

> - `Testing 8700G/780m IGP`
> - `Upcoming GPU comparisons`
> - `P40 vs 4090 pricing`
> - `Model utilization with VRAM`
> - `CPU and GPU upgrades` 


- **8700G/780m IGP Testing Shows Mixed Results**: Testing on the **8700G/780m IGP** using a special version of Ollama revealed around **25% acceleration** versus CPU but only **15% with Vulkan** in LM Studio.
   - While it achieved **30% faster** performance with **llama3.1 70b q4**, LM Studio limited the usable GPU RAM to **20GB**, impacting larger models.
- **Anticipation Surrounding Future GPU Releases**: Speculation arises as members await the **Studio M4 Ultra vs 5090** and discuss the **RTX 6000 Ada** prospects with performance expectations.
   - A member humorously predicted the **5090** could cost a *left kidney*, likely offsetting demand by scalping.
- **P40 vs 4090 Pricing in Australia**: Members discuss the disparity in pricing, with the **4090** costing around **AUD $3000**, significantly higher than **P40s** priced at $300-$600.
   - The **P40's** market behavior suggests it has increased in value due to supply imbalances since its release.
- **Utilizing VRAM for Larger Models**: Members share experiences, noting that running larger models often requires careful balance between **VRAM and RAM**, with some achieving fitting large models on **24GB** GPUs.
   - Testing of the **Yi 1.5 34b 32k** model was suggested as an option for those with ample VRAM.
- **Feedback on 4090 Performance**: After acquiring a **4090**, one member questioned its performance, stating it was **not significantly faster** than their previous **3080**.
   - They mentioned needing to possibly consider **two 4090s or a switch to MAC** for better performance stability.


  

---



### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1270135353378082977)** (5 messages): 

> - `Gameboy emulator setup`
> - `CPython environment considerations`
> - `Reinforcement learning live streaming`
> - `GPUDrive for multi-agent planning`
> - `Mojo discussions` 


- **Gameboy Emulator Environment Setup**: A detailed setup example for a **Gameboy emulator** is available in the [PufferLib GitHub repository](https://github.com/PufferAI/PufferLib/blob/729003f9cb89845cc1a69a65e5a2431b2d0542bd/pufferlib/environments/pokemon_red/environment.py#L15) that simplifies reinforcement learning for game environments.
   - This provides a practical way to dive into RL without needing speed optimizations.
- **Creator Streams for Direct Interaction**: The creator of PufferLib is hosting a [live development stream](https://www.youtube.com/watch?v=dW10MQ6hKDE) where viewers can ask questions directly.
   - The stream is focused on reinforcement learning development, providing a unique opportunity for engagement.
- **GPUDrive: Accelerating Multi-Agent Planning**: An interesting generation example discusses **GPUDrive**, a GPU-accelerated multi-agent simulator capable of generating over a million steps of experience per second, as detailed in a [Hugging Face paper](https://huggingface.co/papers/2408.01584).
   - "This technology enables effective training of reinforcement learning agents in a fraction of the time traditionally required," enhancing capabilities for complex agent behaviors.
- **Invitation for Mojo Talks**: A member expressed gratitude to Chris for joining and invited any team member to present on **Mojo**.
   - The suggested talk could cover an introductory overview of Mojo's current state and vision in the tech landscape.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/papers/2408.01584">Paper page - GPUDrive: Data-driven, multi-agent driving simulation at 1 million FPS</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=dW10MQ6hKDE">Reinforcement learning live dev</a>: Follow jsuarez5341 on XStar https://github.com/pufferai/pufferlibMIT PhD and full-time OSS RL exorcist</li><li><a href="https://github.com/PufferAI/PufferLib/blob/729003f9cb89845cc1a69a65e5a2431b2d0542bd/pufferlib/environments/pokemon_red/environment.py#L15">PufferLib/pufferlib/environments/pokemon_red/environment.py at 729003f9cb89845cc1a69a65e5a2431b2d0542bd · PufferAI/PufferLib</a>: Simplifying reinforcement learning for complex game environments - PufferAI/PufferLib
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1270247873237356574)** (17 messages🔥): 

> - `PyTorch 2.4 with CUDA 12.4`
> - `Zippika's Cublas Library`
> - `FP16 Accumulate Performance`
> - `Benchmarking Speed and Accuracy` 


- **Questions about PyTorch 2.4 and CUDA 12.4**: A user reported issues with **PyTorch 2.4** on **CUDA 12.4**, noting that while the build runs, it produces poor results compared to **CUDA 12.1**.
   - There was additional context regarding the user's system setup using **CUDA 12.6** on a conda installation.
- **Zippika's Cublas Library Gains Windows Compatibility**: Zippika showcased their **Cublas hgemm library** now compatible with Windows, dramatically improving performance from **60 TFLOPS** to **105 TFLOPS** for specific operations.
   - They shared a [GitHub repository](https://github.com/aredden/torch-cublas-hgemm) demonstrating the library's capabilities and benchmarks on various GPUs.
- **Benefits of FP16 Accumulate on Performance**: Zippika highlighted that the library's use of **FP16 accumulate** boosts performance to **330 TFLOPS**, compared to **165 TFLOPS** with **FP32 accumulate**.
   - They emphasized that despite concerns, FP16 accumulate is significantly faster on consumer GPUs, although care must be taken to mitigate potential **inf/nan** issues.
- **Benchmark Results Reveal Minor Differences**: Zippika provided benchmark results demonstrating that their library produced outputs closely aligned with PyTorch's **nn.Linear**, showing only minor discrepancies.
   - The recorded timings indicated their implementation achieved **438.80 us** and **313.22 TFLOPS**, while the standard PyTorch implementation reached **825.59 us** and **166.47 TFLOPS**.
- **Impact on Model Quality**: Zippika claimed that the performance differences observed with their library do not adversely affect generation quality in contexts like **diffusion models** and **LLMs**.
   - This assertion was reinforced by sharing accurate benchmark results showing negligible impact on model generation consistency.



**Link mentioned**: <a href="https://github.com/aredden/torch-cublas-hgemm">GitHub - aredden/torch-cublas-hgemm: PyTorch half precision gemm lib w/ fused optional bias + optional relu/gelu</a>: PyTorch half precision gemm lib w/ fused optional bias + optional relu/gelu - aredden/torch-cublas-hgemm

  

---


### **CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1270094130457870440)** (3 messages): 

> - `CIFAR10 Accuracy`
> - `Quantization Bits Optimization` 


- **Tuning Needed for CIFAR10 Models**: One member reported that their model gets stuck at **70% accuracy** on **CIFAR10**, indicating a need for further tuning.
   - They expressed that while it seems to work, achieving better performance will require adjustments.
- **Optimizing Quantization Bits**: Another member highlighted that the **quantization bits** serve as an optimizable parameter, which they consider a key contribution.
   - This was noted as a significant aspect that could impact the overall model performance.


  

---


### **CUDA MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1270193824337100872)** (7 messages): 

> - `Hudson River Trading Internships`
> - `High-Performance GPU Work`
> - `Job Application Process` 


- **Hudson River Trading offers internships**: Internships at Hudson River Trading are mainly during the summer, with opportunities available for interns to work on *GPU research* similar to full-time positions.
   - Current interns are factoring in GPU research, but many *secret-sauce* tasks are reserved for full-time staff.
- **Excitement about GPU research roles**: A user expressed interest in applying for internships due to their experience in similar work, highlighting excitement about the opportunities.
   - They are encouraged to keep an eye out for upcoming applications, particularly for summer positions.
- **Communicating via direct messages**: A user noted that direct messages were turned off for communication, expressing a desire to connect.
   - The original poster confirmed they sent a friend request while attempting to troubleshoot the messaging issue.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://grnh.se/9f8394ba1us">Senior Software Engineer - Performance Optimization (C++/GPU)</a>: New York, NY, United States</li><li><a href="https://www.levels.fyi/companies/hudson-river-trading/salaries/software-engineer">Hudson River Trading Software Engineer Salary | $406K-$485K+ | Levels.fyi</a>: Software Engineer compensation in United States at Hudson River Trading ranges from $406K per year for L1 to $485K per year for L3. The median compensation in United States package totals $410K. View ...
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1270162701972148335)** (34 messages🔥): 

> - `INT8 Symmetric Quantization`
> - `Install Issues with torchao`
> - `Guarding Unsupported Hardware`
> - `Quantized Training`
> - `GPTQ Refactor Progress` 


- **INT8 Symmetric Quantization Discussion**: A user questioned the use of **127.5** for scale in INT8 quantization, suggesting that it leads to biased weight updates when the softmax output is clipped.
   - Members debated over *full range quantization* versus *restricted range quantization*, with one noting the potential bias challenges encountered during quantized training experiments.
- **Installing torchao from Source**: A user encountered multiple errors while attempting to install **torchao** from source using `python setup.py develop`, particularly on a T4 GPU.
   - Eventually, the installation succeeded with the command `USE_CPP=0 pip install .`, though this approach runs the risk of missing some tests due to disabled cpp extensions.
- **Proposal for Guarding Unsupported Hardware**: Members discussed introducing a compile guard to prevent unsupported hardware from causing errors during compilation, referring to [this example](https://github.com/pytorch/pytorch/blob/e98eac76b358fb4639b9e9ce6894014354d7b073/aten/src/ATen/native/cuda/int4mm.cu#L1).
   - While a compile guard would aid installation, there remain concerns about runtime checks causing strange error messages when operations are called.
- **Updates on Quantized Training**: A user shared perspectives on how implementing **INT8 quantized training** can enhance use cases for training from scratch or during pre-training phases.
   - They emphasized the need to keep inference and training separate, noting potential differences in operations relevant to quantization methodologies.
- **Progress on GPTQ Refactor**: A user is approximately **45%** done with the **GPTQ** refactor, incorporating MultiTensor to replace the fx.interpreter.
   - They expressed a timeline of a few more days to address the associated GitHub issue, indicating steady progress towards completion.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://pytorch.org/">
    
      PyTorch
    
  </a>: no description found</li><li><a href="https://github.com/pytorch/ao/blob/de4a1fb3b1f71e2f61b84dfdc96e7d704ff72208/torchao/quantization/quant_primitives.py#L610">ao/torchao/quantization/quant_primitives.py at de4a1fb3b1f71e2f61b84dfdc96e7d704ff72208 · pytorch/ao</a>: The missing pytorch dtype and layout library for training and inference - pytorch/ao</li><li><a href="https://intellabs.github.io/distiller/algo_quantization.html#symmetric-mode">Quantization - Neural Network Distiller</a>: no description found</li><li><a href="https://github.com/pytorch/pytorch/blob/e98eac76b358fb4639b9e9ce6894014354d7b073/aten/src/ATen/native/cuda/int4mm.cu#L1">pytorch/aten/src/ATen/native/cuda/int4mm.cu at e98eac76b358fb4639b9e9ce6894014354d7b073 · pytorch/pytorch</a>: Tensors and Dynamic neural networks in Python with strong GPU acceleration - pytorch/pytorch
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1270258859633676380)** (7 messages): 

> - `LLaMA 3 paper insights`
> - `Prefix Chunk LLM exploration`
> - `SARATHI framework introduction`
> - `corCTF 2023 kernel syscall challenge` 


- **LLaMA 3 paper reveals exciting dataset section**: A member noted that the **LLaMA 3** paper reads quickly, highlighting the **dataset section** as interesting, while suggesting other parts are better explained in related literature.
   - This insight suggests that the novelty lies primarily in the dataset's architecture.
- **Exploring the fun of Prefix Chunk LLM**: A member recommended reading the **Prefix Chunk LLM paper** (Sarathi LLM), claiming it's more enjoyable than previous works.
   - Community members discussed the implications of shared prompts and improved performance in LLMs.
- **Introduction of SARATHI for LLM inference**: A user introduced **SARATHI**, a framework that addresses inefficiencies during LLM inference phases using **chunked-prefills** and a batching strategy to maximize GPU utilization.
   - The approach emphasizes reducing imbalances during **pipeline parallelism** to enhance overall efficiency.
- **New syscall exploitation challenge in corCTF 2023**: A user detailed a new **syscall challenge** in the context of corCTF 2023, showcasing a Linux syscall that connects kernel internals and micro-architectural attacks.
   - The challenge requires players to exploit newly defined syscalls for buffer manipulation within the kernel.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2308.16369">SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills</a>: Large Language Model (LLM) inference consists of two distinct phases - prefill phase which processes the input prompt and decode phase which generates output tokens autoregressively. While the prefill...</li><li><a href="https://arxiv.org/abs/2402.15220">ChunkAttention: Efficient Self-Attention with Prefix-Aware KV Cache and Two-Phase Partition</a>: Self-attention is an essential component of large language models (LLM) but a significant source of inference latency for long sequences. In multi-tenant LLM serving scenarios, the compute and memory ...</li><li><a href="https://www.willsroot.io/2024/08/just-a-dos-bug.html?m=1">Will's Root: corCTF 2024: Its Just a Dos Bug Bro - Leaking Flags from Filesystem with Spectre v1</a>: no description found
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1270094011696152648)** (99 messages🔥🔥): 

> - `Ragged Attention Issues`
> - `Training with EOS and BOS tokens`
> - `Batch Size and Stability`
> - `Newline Handling in Llama Models`
> - `Performance Benchmarking with PyTorch 2.4` 


- **Ragged Attention and Masking Challenges**: Members discussed the complexities of implementing ragged attention in models, highlighting the necessity of using various masks to prevent out-of-distribution behaviors during training.
   - The importance of the mask's dimensionality was emphasized, with suggestions to support ragged attention as a solution to maintain training integrity.
- **Confusion with EOS/BOS Tokens in Training**: Concerns were raised about Meta's implementation of stop tokens, specifically that the EOS token was omitted in the inference code, potentially causing infinite sampling loops.
   - One member suspected this omission could lead to training issues, urging a review of how these tokens were handled in the model's training processes.
- **Impact of Batch Size on Training Stability**: Discussion ensued regarding the use of smaller batch sizes early in training to enhance stability, revealing a trade-off between training efficiency and model reliability.
   - Citations of relevant papers and practices highlighted the need for deeper understanding, as gradually increasing batch sizes could mitigate instability.
- **Understanding Newline Usage in Llama Models**: Questions emerged concerning the inclusion of newlines in the Llama 3 base model's token format, suggesting developers consider including these tokens during pretraining.
   - Speculations suggested that incorporating newlines may prepare models for instruction tasks, although there was uncertainty about the impacts of this approach.
- **Rapid Performance Enhancement with PyTorch**: A member reported that running the train_gpt2.py script with PyTorch 2.4 yielded performance gains over llm.c, showcasing the potential of the new updates.
   - Comparative results demonstrated nuances in executing with and without flash attention, indicating ongoing improvements in model training efficiency.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2312.16903">Spike No More: Stabilizing the Pre-training of Large Language Models</a>: Loss spikes often occur during pre-training of large language models. The spikes degrade the performance of large language models and sometimes ruin the pre-training. Since the pre-training needs a va...</li><li><a href="https://huggingface.co/docs/transformers/main">🤗 Transformers</a>: no description found</li><li><a href="https://arxiv.org/abs/2108.06084">The Stability-Efficiency Dilemma: Investigating Sequence Length Warmup for Training GPT Models</a>: Recent works have demonstrated great success in pre-training large-scale autoregressive language models on massive GPUs. To reduce the wall-clock training time, a common practice is to increase the ba...</li><li><a href="https://huggingface.co/docs/transformers/main/en/chat_templating">Templates for Chat Models</a>: no description found</li><li><a href="https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/">Llama 3 | Model Cards and Prompt formats</a>: Special Tokens used with Llama 3. A prompt should contain a single system message, can contain multiple alternating user and assistant messages, and always ends with the last user message followed by ...</li><li><a href="https://github.com/Dao-AILab/flash-attention/issues/654.">Issues · Dao-AILab/flash-attention</a>: Fast and memory-efficient exact attention. Contribute to Dao-AILab/flash-attention development by creating an account on GitHub.</li><li><a href="https://github.com/pytorch/torchchat/issues?q=sort%3Aupdated-desc+is%3Aissue+is%3Aopen">Issues · pytorch/torchchat</a>: Run PyTorch LLMs locally on servers, desktop and mobile - Issues · pytorch/torchchat</li><li><a href="https://github.com/pytorch/torchchat/issues?q=sort%3Aupdated-desc+is%3Ais">Issues · pytorch/torchchat</a>: Run PyTorch LLMs locally on servers, desktop and mobile - Issues · pytorch/torchchat</li><li><a href="https://github.com/meta-llama/llama-models/issues/91">Broken links in prompt format docs · Issue #91 · meta-llama/llama-models</a>: In this blog post there are 2 links for the prompt format that are broken https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/ so it&#39;s clear where instructions are to generate ...
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1270413451608461313)** (9 messages🔥): 

> - `ZLUDA 3 Removal`
> - `AMD's Legal Claims`
> - `ZLUDA's Development Status` 


- **ZLUDA 3 Gets Taken Down**: The **ZLUDA** author has taken down **ZLUDA 3** after AMD claimed that the permission given for its release was invalid, referencing the [GitHub page](https://github.com/vosen/ZLUDA).
   - *One of the terms of my contract with AMD was that if AMD did not find it fit for further development, I could release it*.
- **AMD Challenges Legitimacy**: Members discussed the implications of AMD claiming that the **employment contract** is not legally binding regarding the release of **ZLUDA**.
   - The ongoing discourse highlights that if AMD finds ZLUDA suitable for further development, it complicates the author's ability to release it.
- **Acknowledgment of AMD's Role**: One member expressed gratitude towards AMD in light of the situation surrounding **ZLUDA** with a simple, '*thanks amd*.'
   - This sentiment appears to reflect a mix of humor and frustration regarding the legal disputes affecting **ZLUDA's** future.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/vosen/ZLUDA">GitHub - vosen/ZLUDA: CUDA on ??? GPUs</a>: CUDA on ??? GPUs. Contribute to vosen/ZLUDA development by creating an account on GitHub.</li><li><a href="https://github.com/vosen/ZLUDA/tree/v3?tab=readme-ov-file#faq">GitHub - vosen/ZLUDA at v3</a>: CUDA on ??? GPUs. Contribute to vosen/ZLUDA development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[cudamode-irl](https://discord.com/channels/1189498204333543425/1267896441989234709/1270144971646701569)** (2 messages): 

> - `Project Timeline`
> - `Google Form Details` 


- **Expect updates by end of month**: A member expressed confidence that updates will be known by the end of the month at the latest concerning the project status.
   - This indicates a timeline for when more information will be available for those involved.
- **Importance of detailing project work**: Another member mentioned that given the long list of tasks, the best way to ensure clarity is to provide detailed information about what will be worked on via a Google form.
   - They also suggested linking any proposals in the channel for easier access.


  

---



### **Nous Research AI ▷ #[datasets](https://discord.com/channels/1053877538025386074/1105324249721356298/1270435322681102447)** (1 messages): 

> - `UltraSteer-V0 Dataset`
> - `Llama2-13B-SteerLM-RM`
> - `Fine-Grained Dialogue Labels` 


- **Introducing UltraSteer-V0 Dataset**: The new curated dataset, **UltraSteer-V0**, features **2.3M conversations** with **2.8M turns** labeled across **9 fine-grained signals**, produced by Nvidia's **Llama2-13B-SteerLM-RM reward model**.
   - Despite being a **version zero**, it promises unique thread continuations due to extensive deduplication done over 22 days of labeling and processing.
- **Labeling Criteria for Assistant Turns**: Each assistant turn in the UltraSteer dataset is rated on **Quality**, **Toxicity**, **Humor**, and **Creativity**, each on a scale of 0 to 4 to capture nuanced dialogue attributes.
   - These attributes aim to enhance the analysis of conversational AI responses, marking a significant advancement in dialogue dataset quality.
- **Further Improvements Needed for UltraSteer-V0**: The creator acknowledges that UltraSteer-V0 may still benefit from further **de-duplication** and improvements to the dataset card.
   - This feedback reflects an openness to community input for future iterations to enhance usability and clarity.
- **UltraSteer's Framework and Production**: UltraSteer dataset production involved using the **NeMo Aligner** framework, showcasing Nvidia's commitment to advancing **dialogue dataset technologies**.
   - The intricate production process emphasizes quality in generating conversational datasets for AI research.
- **Accessing UltraSteer-V0**: The dataset is now accessible at [Hugging Face](https://huggingface.co/datasets/Avelina/UltraSteer-v0), enabling researchers and developers to leverage its capabilities.
   - With its massive scale and detailed labeling, UltraSteer-V0 is positioned as a valuable resource for enhancing dialogue systems.



**Link mentioned**: <a href="https://huggingface.co/datasets/Avelina/UltraSteer-v0">Avelina/UltraSteer-v0 · Datasets at Hugging Face</a>: no description found

  

---


### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/)** (1 messages): 

vikings7699: Has anyone here ever worked on fine tuning a model specifically for insurance sector?
  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1270123092227788863)** (129 messages🔥🔥): 

> - `Fine-tuning vision models`
> - `Flux AI capabilities`
> - `New multimodal models`
> - `Open Medical Reasoning Tasks`
> - `MiniCPM-Llama3-V updates` 


- **Struggles with Fine-tuning Vision Models**: A user shared frustrations about fine-tuning a vision model that did not perform as expected, leading to discussions about overfitting and model limitations.
   - Another user suggested that accumulated errors and catastrophic forgetting might be factors affecting performance.
- **Excitement over Flux AI's New Skills**: A presentation highlighted that Flux AI claims to excel in **text comprehension**, **prompt comprehension**, and **image generation**.
   - Members expressed enthusiasm for Flux AI's capabilities, with some already using its Pro version.
- **Introduction of Open Medical Reasoning Tasks**: A collaborative initiative called Open Medical Reasoning Tasks has been launched, focusing on creating a comprehensive list of tasks for LLMs in healthcare.
   - Participants were encouraged to contribute, emphasizing the importance of integrating AI and medical expertise.
- **Updates on MiniCPM-Llama3-V and Capability Claims**: Users discussed the latest updates on the MiniCPM-Llama3-V model, including claims of improved capabilities for handling multiple image inputs and OCR tasks.
   - Initial skepticism regarding prior model versions led to excitement with new examples showcasing the ability to use multiple images.
- **Discussion on Model Performance Comparison**: Members compared the performance of various models, including those from **Hugging Face** and other creators, with many interested in multi-image inputs for better results.
   - The conversation also touched on how new models like **BigLlama-3.1** are pushing performance boundaries in the field.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/HuggingFaceM4/Idefics3-8B-Llama3">HuggingFaceM4/Idefics3-8B-Llama3 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5">openbmb/MiniCPM-Llama3-V-2_5 · Hugging Face</a>: no description found</li><li><a href="https://x.com/fofrAI/status/1820878455266816260">Tweet from fofr (@fofrAI)</a>: 🤯  &gt; powerpoint presentation, the slide title says “Flux AI has new skills”, three bullet points, “good at text”, “prompt comprehension”, “amazing images”</li><li><a href="https://huggingface.co/openbmb/MiniCPM-V-2_6">openbmb/MiniCPM-V-2_6 · Hugging Face</a>: no description found</li><li><a href="https://x.com/maximelabonne/status/1820746013503586669">Tweet from Maxime Labonne (@maximelabonne)</a>: 🦙✨ BigLlama-3.1-1T-Instruct  So I&#39;ve heard that 405B parameters weren&#39;t enough...   It&#39;s my pleasure to present an upscaled Llama 3.1 with 1,000,000,000 parameters. Now available on @hugg...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1elgr2x/new_open_llm_leaderboard_champion/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://x.com/aadityaura/status/1820617406970278272?s=46">Tweet from Aaditya Ura ( looking for PhD ) (@aadityaura)</a>: Exciting news! 🎉 Introducing the Open Medical Reasoning Tasks project!  Inspired by @NousResearch and @Teknium1, @OpenLifeSciAI ( Open Life-Science AI ) is launching an open, collaborative initiative...</li><li><a href="https://github.com/black-forest-labs/flux/issues/9)">Issues · black-forest-labs/flux</a>: Official inference repo for FLUX.1 models. Contribute to black-forest-labs/flux development by creating an account on GitHub.</li><li><a href="https://github.com/OpenBMB/MiniCPM-V/issues/233">MiniCPM-V Finetuning for multi-image input during a multi-turn conversation💡 [REQUEST] - &lt;title&gt; · Issue #233 · OpenBMB/MiniCPM-V</a>: 起始日期 | Start Date No response 实现PR | Implementation PR No response 相关Issues | Reference Issues for multi-image input during a multi-turn conversation 摘要 | Summary for multi-image input during a mul...</li><li><a href="https://old.reddit.com/r/StableDiffusion/comments/1ekte84/generated_with_flux1_pro_and_schnell/">Generated with Flux.1 Pro and Schnell </a>: Posted in r/StableDiffusion by u/Sea_Law_7725 • 376 points and 78 comments
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1270123471300464783)** (19 messages🔥): 

> - `Finetuning Libraries`
> - `Inference Stack Resources`
> - `Insurance Sector Model Tuning`
> - `Pay-as-you-go LLaMA Hosting`
> - `Compute Bottlenecks in Training` 


- **Most use libraries for finetuning**: Many people tend to use existing libraries like **Axolotl** for finetuning and training rather than writing unique scripts from scratch.
   - *The benefits of libraries* are recognized in streamlining the training process.
- **Getting started with vLLM**: A user inquired about resources for the **vLLM** inference stack, suggesting that the project itself is a good starting point.
   - Discussion continued on relevant codebases to explore for better understanding.
- **Fine-tuning models for insurance**: A user asked if anyone has successfully fine-tuned models specifically for the **insurance sector**.
   - This topic explored the challenges and strategies relevant to this niche.
- **Pay-as-you-go LLaMA hosting options**: There was a search for companies offering **LLaMA 450b** with pay-as-you-go access, with hints towards **Groq** and **Openrouter** as potential solutions.
   - Members discussed the need to check the hosting providers listed on **Openrouter** for more information.
- **Understanding compute bottlenecks**: The main bottleneck in inference and training, particularly at a batch size of 1, is typically related to **memory**.
   - As batch size increases, it becomes **compute bound**, leading to discussions on GPU utilization and active CUDA cores.


  

---


### **Nous Research AI ▷ #[reasoning-tasks-master-list](https://discord.com/channels/1053877538025386074/1264666760972472481/1270115994676756523)** (7 messages): 

> - `Open Medical Reasoning Tasks`
> - `Synthetic Task Generation`
> - `Limits of LLMs`
> - `Community Contributions` 


- **Exciting Launch of Open Medical Reasoning Tasks**: Inspired by Nous Research and Teknium, [Open Life-Science AI](https://x.com/aadityaura/status/1820617406970278272?s=46) has launched an open initiative to create a comprehensive list of medical reasoning tasks for LLMs.
   - This project seeks contributions from physicians, researchers, and data scientists to advance AI in healthcare, as detailed on [GitHub](https://github.com/openlifescience-ai/Open-Medical-Reasoning-Tasks).
- **Community Excitement for Collaborative Efforts**: A member expressed enthusiasm about the new medical reasoning project, stating, *This is AMAZING! This is what happens when you work in the open! I love this!*
   - The sentiment was echoed in related discussions about the positive impact of open collaboration in advancing medical AI.
- **Further Resources on System 2 Reasoning**: A contribution was mentioned regarding a link to [System 2 Reasoning Link Collection](https://github.com/open-thought/system-2-research), which is a collaborative GitHub repository.
   - This repository encourages community involvement in gathering valuable insights related to System 2 reasoning and its applications.
- **Exploration of Synthetic Task Generation**: There were considerations on improving **synthetic task generation** to overcome the limitations faced by LLMs in generating complex tasks.
   - One member highlighted the challenges of progressing beyond the simple outputs currently achievable by LLMs.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/aadityaura/status/1820617406970278272?s=46">Tweet from Aaditya Ura ( looking for PhD ) (@aadityaura)</a>: Exciting news! 🎉 Introducing the Open Medical Reasoning Tasks project!  Inspired by @NousResearch and @Teknium1, @OpenLifeSciAI ( Open Life-Science AI ) is launching an open, collaborative initiative...</li><li><a href="https://github.com/open-thought/system-2-research">GitHub - open-thought/system-2-research: System 2 Reasoning Link Collection</a>: System 2 Reasoning Link Collection. Contribute to open-thought/system-2-research development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1270094956945604620)** (128 messages🔥🔥): 

> - `Web Dev to AI Engineer Pipeline`
> - `OpenAI Departures`
> - `Generative AI in Retail`
> - `Structured Outputs in GPT-4o`
> - `Energy-based Language Modeling` 


- **Web Dev to AI Engineer Pipeline Emerging**: Discussions highlighted the growing transition of web developers into AI engineering due to high demand and limited ML engineers, with some sharing their experiences in adapting skill sets.
   - Participants expressed interest in how web devs overall are seen as fungible, often responsible for implementing AI projects alongside traditional development tasks.
- **Recent Departures at OpenAI Stir Concerns**: A wave of leadership changes at OpenAI has raised questions about the company's direction, with many in the community expressing worries about its future and stability.
   - Comments included speculation about the implications of these departures and whether OpenAI's trajectory is still positive.
- **Generative AI Applications in Retail**: Generative AI is making strides in the retail sector, particularly in enhancing product descriptions for various platforms and languages, as discussed by members using examples from L'Oreal.
   - Questions arose about how to measure the effectiveness of AI-generated descriptions, emphasizing the need for metrics in assessing performance.
- **Launch of Structured Outputs Feature in GPT-4o**: OpenAI introduced a structured outputs feature, allowing models to reliably follow JSON schemas with significantly improved reliability compared to previous versions.
   - The introduction of this feature indicates progress in AI's ability to generate more controlled and structured data outputs, which was recognized and discussed among community members.
- **Skepticism Surrounding Energy-Based Language Modeling**: A humorous anecdote shared about a meet-up with an Extropic AI researcher revealed a lack of familiarity with established works in energy-based language modeling, raising skepticism about their claims.
   - The exchange pointed towards a broader question of credibility regarding some newer companies' expertise in complex topics within AI.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/TwoWeeksLOL/status/1820536638268948750">Tweet from Two Weeks LOL (@TwoWeeksLOL)</a>: @MKBHD Uh oh...</li><li><a href="https://x.com/tszzl/status/1714357380413264044?s=46">Tweet from roon (@tszzl)</a>: all the people that can make eye contact at openai joined in the last 6 months and they’re making me uncomfortable with their eye contact</li><li><a href="https://news.ycombinator.com/item?id=41174306">no title found</a>: no description found</li><li><a href="https://news.ycombinator.com/item?id=41173964">no title found</a>: no description found</li><li><a href="https://x.com/NickADobos/status/1820513765823250730">Tweet from Nick Dobos (@NickADobos)</a>: Great post on writing code with ai Love this chart  Quoting Erik Schluntz (@ErikSchluntz)   Replacing my right hand with AI  (How I wrote thousands of lines of code for work each week while in a cast)...</li><li><a href="https://x.com/aizkmusic/status/1820594845792051391?s=46">Tweet from Aizk ✡️ (@Aizkmusic)</a>: @BigTechAlert @ChatGPTapp @TarunGogineni His LinkedIn bio is great</li><li><a href="https://arxiv.org/abs/2307.09702">Efficient Guided Generation for Large Language Models</a>: In this article we show how the problem of neural text generation can be constructively reformulated in terms of transitions between the states of a finite-state machine. This framework leads to an ef...</li><li><a href="https://x.com/_philschmid/status/1820715040191750370">Tweet from Philipp Schmid (@_philschmid)</a>: &#34;Deep Reinforcement Learning from Human Preferences&#34; and &#34;Proximal Policy Optimization Algorithms&#34; are part of the foundation of modern RLHF in LLMs.</li><li><a href="https://x.com/OpenAIDevs/status/1820542222259073137">Tweet from OpenAI Developers (@OpenAIDevs)</a>: We’re taking OpenAI DevDay on the road! Join us this fall in San Francisco, London, or Singapore for hands-on sessions, demos, and best practices. Meet our engineers and see how developers around the ...</li><li><a href="https://x.com/jxmnop/status/1820876333154759091">Tweet from jack morris (@jxmnop)</a>: funny little story about Extropic AI  &gt;been curious about them for a while &gt;have twitter mutual who is an engineer/researcher for this company &gt;often tweets energy-based modeling and LM-quant...</li><li><a href="https://x.com/jason_koebler/status/1820493304490074391">Tweet from Jason Koebler (@jason_koebler)</a>: SCOOP from @samleecole: Leaked Slacks and documents show the incredible scale of NVidia&#39;s AI scraping: 80 years — &#34;a human lifetime&#34; of videos every day. Had approval from highest levels o...</li><li><a href="https://x.com/abacaj/status/1820883396077482087">Tweet from anton (@abacaj)</a>: interesting... new model also includes a pretty big price drop  Quoting OpenAI Developers (@OpenAIDevs)   Introducing Structured Outputs in the API—model outputs now adhere to developer-supplied JSON ...</li><li><a href="https://x.com/johnschulman2/status/1820610863499509855">Tweet from John Schulman (@johnschulman2)</a>: I shared the following note with my OpenAI colleagues today:  I&#39;ve made the difficult decision to leave OpenAI. This choice stems from my desire to deepen my focus on AI alignment, and to start a ...</li><li><a href="https://x.com/_mira___mira_/status/1820625134354669697?s=46">Tweet from Mira (@_Mira___Mira_)</a>: no description found</li><li><a href="https://x.com/michpokrass/status/1820881057824305567">Tweet from Michelle Pokrass (@michpokrass)</a>: excited to announce Structured Outputs -- our newest feature in the api. model outputs will now reliably follow your exact json schemas, matching the parameters and types accurately.   schema reliabil...</li><li><a href="https://github.com/simonw/datasette">GitHub - simonw/datasette: An open source multi-tool for exploring and publishing data</a>: An open source multi-tool for exploring and publishing data - simonw/datasette</li><li><a href="https://writer.com/use-cases/ecommerce/">eCommerce &amp; Retail</a>: Discover how innovative eCommerce and retail companies use Writer to create on-brand content that works, from first touch to sale.
</li>
</ul>

</div>
  

---



### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1270101558738157713)** (1 messages): 

> - `OpenAI DevDay`
> - `Global Events` 


- **OpenAI DevDay Goes Global!**: OpenAI is taking **DevDay** on the road this fall, with events in **San Francisco**, **London**, and **Singapore**.
   - Participants can expect hands-on sessions, demos, and best practices, as well as the chance to meet **OpenAI engineers** and see how developers are building with OpenAI. More details can be found [here](https://openai.com/devday/).
- **Connect With Developers Worldwide**: DevDay offers an opportunity for developers around the world to connect and share insights about building with **OpenAI** technologies.
   - Engage in comprehensive discussions and explore innovative practices that are redefining AI development.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1270275479793827891)** (86 messages🔥🔥): 

> - `Desktop ChatGPT App Release`
> - `DALL-E 3 Model Discussion`
> - `API and Hosting Questions`
> - `LaTeX Support on Mobile`
> - `OpenAI Updates and Pricing` 


- **Questions on Desktop ChatGPT App and Search GPT Release**: Members expressed curiosity about the release timeline for the **desktop ChatGPT app** on Windows and the public availability of **Search GPT**.
   - There were humorous comments about Sam Altman being the only remaining founder amidst the discussions.
- **DALL-E 3 Model and Results Variability**: Discussions continued about the **DALL-E 3 model** and the variability in results, with mention of comparisons to Llama models and queries regarding why outcomes differed.
   - Users noted that differences in output quality might be due to reasons such as safety filters implemented by OpenAI.
- **API Use and Free Options**: Users inquired if the **Llama API** is free, discussing the possibility of running models like **Llama 3.1 8b** locally without costs.
   - The conversation noted that while the model might be free, there are no official unlimited free APIs available.
- **LaTeX Support Limitations**: A user raised concerns about the lack of **LaTeX support** in mobile apps, with some members suggesting alternatives like using mobile browsers.
   - It was mentioned that numerous bug reports have been filed regarding this feature.
- **Impressive OpenAI Updates and Pricing Changes**: New features like **Structured Outputs** were discussed, highlighting advantages such as improved consistency and potentially lower prices for API usage.
   - However, members questioned the performance-to-cost tradeoff of the new models and their updated iteration compared to previous ones.



**Link mentioned**: <a href="https://stackoverflow.com/questions/78839847/assistant-gpt-can-i-perform-knowledge-retrieval-from-a-cloud-storage">Assistant GPT - Can I perform knowledge retrieval from a cloud storage?</a>: I have some files that are on my cloud storage (onedrive) and would like to perform knowledge retrieval on them. Is it possible to integrate an assistant to perform knowledge retrieval directly fro...

  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1270164994192969758)** (16 messages🔥): 

> - `Search GPT availability`
> - `Photo upload limits`
> - `Generative AI in gaming`
> - `ChatGPT-4o updates` 


- **Search GPT is available now**: A member confirmed that **Search GPT** has indeed been rolled out for users.
   - This update has raised questions about how users are utilizing this new feature.
- **Upload limits still apply for members**: Members expressed concerns over upload limits despite being paid users, with one reporting their limit reset time as being 1:35.
   - Another member clarified that **even paid users** are subject to these upload restrictions.
- **Generative AI's potential in gaming**: A member shared excitement about the potential for generative AI to enhance games like **BG3** or **Pathfinder** by enabling character design and unique interactions.
   - They envisioned a fully immersive experience where **NPCs react** dynamically to player choices.
- **Updates to ChatGPT-4o model**: Users speculated on potential changes to **ChatGPT-4o**, with one including an update link about structured outputs.
   - Another noted that the model was likely revised and mentioned the version `gpt-4o-2024-08-06`.
- **Increased performance sparks questions**: Members have noticed a change in the performance of **ChatGPT-4o** and questioned whether it has undergone any updates recently.
   - One member noted a distinct difference in responses, prompting a discussion about potential updates.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/)** (1 messages): 

darthgustav.: Use the python tool and import data from uploads.
  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/)** (1 messages): 

darthgustav.: Use the python tool and import data from uploads.
  

---



### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1270133269052002517)** (82 messages🔥🔥): 

> - `Future of Information Retrieval`
> - `Technical Issues with Perplexity`
> - `Comparison of Language Models`
> - `Content Recommendation Engines`
> - `Feedback on User Experiences` 


- **Future of Information Retrieval Considerations**: A member pondered if future information retrieval systems using language models will integrate source weights, assigning more value to reputable sources over less credible ones.
   - They inquired if there's any interesting paper on this topic and whether models could evaluate source quality autonomously.
- **Users Face Technical Issues with Perplexity**: Multiple users reported issues with features in the Perplexity Pro app, including the inability to switch LLMs and missing libraries prompting concerns about functionality.
   - However, some users noted that features returned unexpectedly shortly after the issues arose.
- **Language Model Comparisons Spark Debate**: Users discussed the comparison between GPT-4o and Turbo, with mixed opinions on their performance and responsiveness, noting that some found GPT-4o less effective in conversations.
   - Some users preferred other LLMs, highlighting a perceived lack of acknowledgment from GPT-4o when given new instructions.
- **Development of Content Recommendation Engines**: A university project focused on building a content sorting and recommendation engine was introduced, with a goal of analyzing and sorting content based on user input.
   - Members suggested looking into 'RAG' (retrieval-augmented generation) as a relevant concept for the project.
- **Experiences with Uber's Subscription Offer**: A user sought clarity on receiving an Uber subscription offer code, asking if it was automatically sent or required a user action.
   - Others confirmed they automatically received the email without needing to request it.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://uncovr.app>">no title found</a>: no description found</li><li><a href="https://github.com/inulute/perplexity-ai-app/releases">Releases · inulute/perplexity-ai-app</a>: The Perplexity AI Desktop App, powered by Electron which brings the magic of AI language processing to your desktop. - inulute/perplexity-ai-app</li><li><a href="https://felo.ai/search/PALsa8DEHJaiJcU6DYi4Q9">When Tom&#x27;s funeral was held, his father didn&#x27;t attend. Now that his father has passed away, Tom didn&#x27;t show up at his father&#x27;s funeral either. Is Tom going too far?</a>: The situation you described involves a complex interplay of personal relationships and individual choices. Here are some points to consider:  ### Context and Ba
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1270168758467432448)** (7 messages): 

> - `NVIDIA Blackwell GPUs Delay`
> - `Memory Scientific Explanation`
> - `Market Updates`
> - `LLaMA 3 Performance`
> - `Tastiera Meccanica Recommendations` 


- **NVIDIA's Blackwell GPUs delayed due to design flaws**: NVIDIA's next-generation **Blackwell GPUs** have faced delays primarily due to **design flaws** found late in production and issues with the advanced **CoWoS-L** packaging technology.
   - These complications are forcing a redesign of the processor die, prolonging production tests and validations.
- **Understanding Memory Mechanisms**: Discussion about the scientific explanation behind **memory** highlights its reliance on processes involving various brain regions like the **hippocampus**.
   - Despite ongoing interest, there's no evidence that objects can store memories, keeping the concept in metaphysical realms.
- **Current Market Jitters and Art Sales**: Market discussions noted a variety of **current events**, including a $26 million digital portrait sale and Google's recent legal setbacks.
   - These issues contributed to fluctuating market sentiments among tech enthusiasts.
- **LLaMA 3.1 Performance Insights**: Emerging details about **LLaMA 3.1** indicate its performance metrics are generating considerable interest among tech communities.
   - Further insights are expected to illuminate its capabilities and potential applications.
- **Recommendations for Mechanical Keyboards**: A request for recommendations on mechanical keyboards highlights the community's ongoing interest in gaming and productivity tools.
   - No specific models were mentioned, but the search for high-quality options continues.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/embed/ZLEuncAV70U">YouTube</a>: no description found</li><li><a href="https://www.perplexity.ai/search/nvidia-blackwell-s-delay-expla-kjKmWq15SdKcDJAgGn01EQ">NVIDIA Blackwell&#x27;s Delay Explained</a>: NVIDIA&#x27;s next-generation Blackwell GPUs have encountered delays primarily due to design and manufacturing issues. Here are the main reasons for the delay: The...</li><li><a href="https://www.perplexity.ai/search/how-does-llama-3-1-405b-s-perf-YIDs8nm2TuuJzP4ILbY1BA">Perplexity</a>: Perplexity is a free AI-powered answer engine that provides accurate, trusted, and real-time answers to any question.</li><li><a href="https://www.perplexity.ai/search/how-can-i-get-a-summary-of-a-b-46KrvDREQKeVwv4VwBO2Lw">Perplexity</a>: Perplexity is a free AI-powered answer engine that provides accurate, trusted, and real-time answers to any question.</li><li><a href="https://www.perplexity.ai/search/is-naturland-s-tobotronc-the-l-GVann50ESpqyNuB4wT4qvw">Perplexity</a>: Perplexity is a free AI-powered answer engine that provides accurate, trusted, and real-time answers to any question.</li><li><a href="https://www.perplexity.ai/search/consigliami-una-tastiera-mecca-bBenHhBBQUe7YmplIxO6IA">Perplexity</a>: Perplexity is a free AI-powered answer engine that provides accurate, trusted, and real-time answers to any question.</li><li><a href="https://www.perplexity.ai/search/apa-saja-benda-yang-mengandung-XdLR2Ja0TB.hH1hwSOy2DA">apa saja benda yang mengandung karbon</a>: Benda yang mengandung karbon sangat beragam dan dapat ditemukan dalam berbagai bentuk di kehidupan sehari-hari. Berikut adalah beberapa contoh benda yang...
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1270118041962942547)** (8 messages🔥): 

> - `API Errors`
> - `Upcoming Model Changes`
> - `Testing in Labs`
> - `Status Updates` 


- **User reports API discrepancies**: A user expressed concerns about their API query returning corrupted data, leading them to question the API's functionality.
   - Another user mentioned similar issues, suggesting a potential widespread problem with the API's reliability.
- **Perplexity API models to be deprecated**: Confirmation was shared that all Perplexity API models will be deprecated on **August 12, 2024**, leading to questions about their continuity.
   - A detailed guide was provided about the models, highlighting key parameters and the system prompt's behavior.
- **Status check for API availability**: A member questioned potential issues with the API, citing HTTP 502 errors during attempts to connect.
   - A recent status report indicated no issues with the API, suggesting it could be a localized problem or user-specific.
- **Testing Perplexity API in Labs**: A user suggested trying the API via the [Perplexity Labs](https://labs.perplexity.ai) playground for further testing.
   - This recommendation aimed to isolate whether the issues were API-wide or specific to individual queries.
- **Recent Perplexity status updates**: Status checks revealed no recent notices or issues reported over the past days regarding the Perplexity API.
   - The ongoing stability suggested that if problems persist, they might be user-specific rather than systemic.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.perplexity.ai/docs/model-cards">Supported Models</a>: Perplexity Models Model Parameter Count Context Length Model Type llama-3-sonar-small-32k-online 8B 28,000 Chat Completion llama-3-sonar-small-32k-chat 8B 32,768 Chat Completion llama-3-sonar-large-32...</li><li><a href="https://labs.perplexity.ai">Perplexity Labs</a>: no description found</li><li><a href="https://status.perplexity.com/">Perplexity - Status</a>: Perplexity Status
</li>
</ul>

</div>
  

---



### **Eleuther ▷ #[announcements](https://discord.com/channels/729741769192767510/794042109048651818/1270325700599091210)** (1 messages): 

> - `Mechanistic Anomaly Detection`
> - `Quirky Language Models`
> - `Anomaly Detection Techniques`
> - `Adversarial Example Detection` 


- **Novel Methods in Mechanistic Anomaly Detection**: The team explored _mechanistic_ methods for detecting anomalies in language models using [Neel Nanda's attribution patching technique](https://blog.eleuther.ai/mad_research_update/), but found these methods did not consistently outperform traditional baselines based solely on activations.
   - Better performance was achieved by evaluating entire batches rather than individual points, with varying success across tasks.
- **Adversarial Detection in Image Classifiers**: The Eleuther team noted that detecting adversarial examples in image classifiers is relatively straightforward with existing techniques, though they did not verify if their anomaly detectors are adversarially robust.
   - This finding suggests opportunities for enhancing robustness in anomaly detection methods.
- **Research on Quirky Language Models**: In December 2023, the team published a paper on [Eliciting Latent Knowledge from Quirky Language Models](https://arxiv.org/abs/2312.01037v3), finetuning models to switch between reliable and unreliable answering patterns based on prompt cues.
   - The research investigated unsupervised detection of model behavior, distinguishing between 'Alice'-type accuracy and 'Bob'-type heuristics, fitting into the [_Mechanistic Anomaly Detection_](https://www.lesswrong.com/posts/n7DFwtJvCzkuKmtbG/a-gentle-introduction-to-mechanistic-anomaly-detection) framework.
- **Anomaly Detection Code Released**: The EleutherAI team released a [GitHub repository](https://github.com/EleutherAI/cupbearer/tree/attribution_detector) for mechanistic anomaly detection which showcases their methodology.
   - This resource may be helpful for others interested in contributing to the project and its ongoing development.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://blog.eleuther.ai/mad_research_update/">Mechanistic Anomaly Detection Research Update</a>: Interim report on ongoing work on mechanistic anomaly detection</li><li><a href="https://github.com/EleutherAI/cupbearer/tree/attribution_detector">GitHub - EleutherAI/cupbearer at attribution_detector</a>: A library for mechanistic anomaly detection. Contribute to EleutherAI/cupbearer development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1270174953135935569)** (36 messages🔥): 

> - `SB1047 AI Safety Act`
> - `Philosophical Differences in AI Regulation`
> - `Anthropic's Response to SB1047`
> - `Impact of Regulation on Innovation`
> - `Knowledge Distillation in LLMs` 


- **Debate Around SB1047 Intensifies**: Members engaged in a heated discussion about SB1047 (AI Safety Act), with some arguing it could hinder innovation in AI research while others see it as a necessary regulation to ensure accountability.
   - Critics expressed concerns that the law might deter open research, with a focus on the implications of its liability provisions and the uncertainty it creates.
- **Philosophical Divide Over AI Regulation**: The conversation highlighted a deeper ideological conflict about the role of government in AI oversight, with opinions split on the need for regulatory frameworks versus the push for uninhibited research.
   - Some believe the debate reflects broader societal disagreements on what constitutes a good society and the future of AI technology.
- **Support for Anthropic's Viewpoint**: Several members expressed appreciation for Anthropic's response to SB1047, citing it as a sensible take that aligns with concerns for both innovation and safety.
   - The response was noted for addressing important issues surrounding the bill's implementation and the balance between regulation and research advancement.
- **Commercialization vs. Caution in AI Development**: Discussion pointed out potential tension between the need for safe AI practices and the drive for profit, with suggestions that legal incentives could push for more rigorous research protocols.
   - However, members warned against overly prescriptive laws that might stagnate innovation by mandating premature solutions like watermarking.
- **Knowledge Distillation Resources Requested**: One member sought recommendations for hands-on resources related to knowledge distillation and training smaller LLMs using larger models.
   - This inquiry underscores a growing interest in practical applications of model training within the community.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.documentcloud.org/documents/25003075-sia-sb-1047-anth">DocumentCloud</a>: no description found</li><li><a href="https://www.documentcloud.org/documents/25003075-sia-sb-1047-anthropic">DocumentCloud</a>: no description found</li><li><a href="https://safesecureai.org/responseletter">Letter to YC &amp; a16z | SB 1047 - Safe &amp; Secure AI Innovation</a>: no description found</li><li><a href="https://docs.google.com/forms/d/e/1FAIpQLSewflVHn1zoNeHHJq3SaKvlwPy7PLT1Vcu_WoULqcHSSjvX1w/viewform">Students, Faculty, and Scientists Against SB 1047 (AI Safety Act) Open Letter Signature Form</a>: This is a form to provide your signature in support of our open letter from UC Faculty and students against California SB 1047, a catastrophically bad law attempting to regulate &quot;AI safety&quot; ...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1270111334914330777)** (40 messages🔥): 

> - `Meta's AI network advancements`
> - `Discrete problem-solving in AI`
> - `Inferencing scalability in models`
> - `Search techniques in machine learning evaluation`
> - `Self-taught evaluation methods for models` 


- **Meta showcases AI network advancements**: At [ACM SIGCOMM 2024](https://conferences.sigcomm.org/sigcomm/2024/) in Sydney, Meta presented their paper on the [RDMA over Ethernet for Distributed AI Training at Meta Scale](https://dl.acm.org/doi/10.1145/3651890.3672233), emphasizing the infrastructure supporting large-scale AI model training like **LLAMA 3.1 405B**.
   - This highlights the **growing communication demands** driven by the rise of AI, particularly in distributed training workloads.
- **Debate on discrete problem-solving in AI**: An ongoing discussion revolved around using discrete spaces versus latent representations in AI search models, emphasizing the complexity of composing solutions when sampling independently.
   - One member suggested a **Vector Quantization** method to incentivize models to learn easily composable sub-solutions.
- **Scaling inference compute for better performance**: A new paper highlighted the potential of increasing the number of generated samples during inference to improve problem-solving coverage, which scales performance across various tasks.
   - The results indicated that applying repeated sampling could significantly enhance issue resolution rates in tasks like coding.
- **Exploring search techniques in ML evaluation**: There was a consensus that traditional search methods often outperform more sophisticated, differentiable alternatives due to their simplicity and practicality.
   - The effectiveness of simpler methods has been observed to yield comparable or superior results despite the allure of more complex techniques.
- **Innovative self-taught evaluation methods emerge**: A paper introduced a self-improvement scheme for evaluators that enhances models without needing extensive human annotations, using only synthetic training data.
   - The method improved the **Llama3-70B-Instruct** from **75.4** to **88.3** accuracy, showcasing an effective way to generate training data dynamically.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2407.21787">Large Language Monkeys: Scaling Inference Compute with Repeated Sampling</a>: Scaling the amount of compute used to train language models has dramatically improved their capabilities. However, when it comes to inference, we often limit the amount of compute to only one attempt ...</li><li><a href="https://arxiv.org/abs/2408.02666">Self-Taught Evaluators</a>: Model-based evaluation is at the heart of successful model development -- as a reward model for training, and as a replacement for human evaluation. To train such evaluators, the standard approach is ...</li><li><a href="https://arxiv.org/abs/2408.00724">An Empirical Analysis of Compute-Optimal Inference for Problem-Solving with Language Models</a>: The optimal training configurations of large language models (LLMs) with respect to model sizes and compute budgets have been extensively studied. But how to optimally configure LLMs during inference ...</li><li><a href="https://engineering.fb.com/2024/08/05/data-center-engineering/roce-network-distributed-ai-training-at-scale/">RoCE networks for distributed AI training at scale</a>: AI networks play an important role in interconnecting tens of thousands of GPUs together, forming the foundational infrastructure for training, enabling large models with hundreds of billions of pa…</li><li><a href="https://redwoodresearch.substack.com/p/getting-50-sota-on-arc-agi-with-gpt">Getting 50% (SoTA) on ARC-AGI with GPT-4o</a>: You can just draw more samples
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1270272866587512915)** (4 messages): 

> - `Training Instability`
> - `Double Descent Phenomenon`
> - `Learning Rate Adjustments` 


- **Noise or Training Instability is More Likely**: A member asserted that the observed issues are more likely due to **noise** or **training instability** rather than the **double descent phenomenon**.
   - *It's still more likely noise/training instability issues than double descent.*
- **Suggests Averaging Results from Multiple Experiments**: It was advised to conduct the experiment **3 or 5 times** and average the results for more reliable outcomes.
   - *If it’s me I would do the experiment 3 or 5 times and average the result first.*
- **Lower Learning Rate for Stability**: To improve training stability, a recommendation was made to **lower the learning rate** if instability persists.
   - *If this phenomenon still exists I would try to lower the learning rate to make training stability less likely.*
- **Consider Other Possibilities After Initial Steps**: The member suggests only after experimenting and adjusting learning rates should one consider other potential causes.
   - *Only after both I would start considering other possibilities.*


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1270115079051808778)** (5 messages): 

> - `Recent Developments in SAEs`
> - `SAE Notation and Framework`
> - `SAE Landscape Overview`
> - `SAELens Library Updates`
> - `Scaling SAEs to Real Models` 


- **Getting up to speed on recent SAE developments**: Members expressed interest in catching up on recent developments in SAEs, particularly referencing the [comprehensive paper on SAE](https://transformer-circuits.pub/2023/monosemantic-features/index.html) from the transformer circuits thread.
   - They also highlighted a follow-up work on [scaling SAEs](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html) and relevant superposition papers.
- **Current relevance of SAE notation**: Discussion included the relevance of the notation mentioned in the SAE framework, which remains a reference point for understanding new developments.
   - Links to ongoing works that expand on the notation and methodologies were shared, emphasizing their practicality in real models.
- **SAE Landscape Document**: An overview of the SAE landscape was shared, compiled in a [Google document](https://docs.google.com/document/d/1lHvRXJsbi41bNGZ_znGN7DmlLXITXyWyISan7Qx2y6s/edit#heading=h.j9b3g3x1o1z4) that provides a rough overview of the SAE field.
   - While it might miss some latest developments, it serves as a foundational resource for understanding current SAE themes.
- **SAELens and related tools for SAEs**: Members discussed the capabilities of [SAELens](https://github.com/jbloomAus/SAELens), a library aimed at training and analyzing SAEs, and its relationship with newer frameworks like [auto-interp](https://github.com/EleutherAI/sae-auto-interp).
   - The integration among these tools aims to improve accessibility and visualization for users working on large-scale models.
- **Collaboration and Community Engagement**: A member pointed out the opportunity to engage with teams working on SAEs through designated channels, highlighting collaboration between GDM and OpenAI.
   - This offers an environment for discussing advances and challenges faced in SAE research and applications.



**Link mentioned**: <a href="https://docs.google.com/document/d/1lHvRXJsbi41bNGZ_znGN7DmlLXITXyWyISan7Qx2y6s/edit#heading=h.j9b3g3x1o1z4">SAE Landscape</a>: SAE Landscape – A collection of useful publications and tools Welcome to a collection of resources on Sparse Autoencoders (SAEs) for language model interpretability. This is a live document, I appreci...

  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1270115967858380883)** (8 messages🔥): 

> - `lm-eval-harness usage`
> - `Huggingface model class`
> - `Handling special tokens`
> - `Accessing benchmark names in JSON output` 


- **Using lm-eval-harness for custom models**: A user inquired about evaluating a custom model using **lm-eval-harness**, and they received a helpful link to a [self-contained example](https://github.com/state-spaces/mamba/blob/main/evals/lm_harness_eval.py) for modifying the Huggingface LM class.
   - Another member added that **HFLM** class supports passing an already initialized HF `PretrainedModel`, allowing for evaluation in custom scripts.
- **Batch Size in loglikelihood_rolling**: A user asked if **loglikelihood_rolling** respects batch size in the Huggingface model class, noting it seems to run one request at a time.
   - This touches on the efficiency of batch processing when utilizing the Huggingface model architecture.
- **BOS Token in evalharness**: A member sought clarification on whether **evalharness** adds the **BOS** token by default, noting the tokenizer's default is to add special tokens.
   - They observed that the generated sample files did not include the **BOS** token and wanted to confirm its presence.
- **Finding benchmark names from JSON output**: A user inquired about extracting the benchmark name from the JSON output of **lm-eval-harness**.
   - Another member suggested the results JSON has a key `results` that contains benchmark names as keys with their scores as values.



**Link mentioned**: <a href="https://github.com/state-spaces/mamba/blob/main/evals/lm_harness_eval.py">mamba/evals/lm_harness_eval.py at main · state-spaces/mamba</a>: Mamba SSM architecture. Contribute to state-spaces/mamba development by creating an account on GitHub.

  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1270096212292534272)** (83 messages🔥🔥): 

> - `GPU Memory Management`
> - `LangGraph Courses`
> - `SQL Chat Agent Development`
> - `Music Recommendation App`
> - `Code Review Automation` 


- **Managing GPU Memory Issues**: A user reported encountering out-of-memory errors while using models like **aya** and **nomic-embed-text**, despite having **32GB RAM** and models being less than **4GB**.
   - After troubleshooting, it was suggested to run the application on **CPU** instead of **GPU**, but this resulted in significantly slower performance.
- **LangGraph Course Recommendations**: Users discussed available courses for **LangGraph**, with recommendations including the **DeepLearning AI course** and a more advanced one on **Udemy**.
   - The consensus was that many beginner-friendly resources exist, but advanced courses are scarce.
- **Collaboration on SQL Chat Agent**: One user reached out for assistance on a **SQL chat agent script** and another user with experience in similar projects offered to help.
   - Scripts and feedback were exchanged, indicating a collaborative effort to enhance the SQL agent functionalities.
- **New Music Discovery App**: A new app, **mood2music**, was introduced, offering music recommendations based on users' moods through AI analysis and integration with streaming services.
   - The app is currently seeking users to join a waitlist, emphasizing its unique features in music curation.
- **Improving Code Review Automation**: One developer inquired about improving automatic code reviews using **GPT-4o**, especially for selecting positions in GitHub diffs.
   - Suggestions were made to use a specialized coding model instead of a vision model, focusing on efficient data parsing and retrieval strategies.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://mood2music.me">mood2music</a>: no description found</li><li><a href="https://www.deeplearning.ai/short-courses/ai-agents-in-langgraph/">AI Agents in LangGraph</a>: Build agentic AI workflows using LangChain&#x27;s LangGraph and Tavily&#x27;s agentic search. Learn directly from LangChain and Tavily founders.</li><li><a href="https://superlinked.com/vector-db-comparison">Vector DB Comparison</a>: Vector DB Comparison is a free and open source tool from VectorHub to compare vector databases.</li><li><a href="https://js.langchain.com/v0.2/docs/tutorials/chatbot">Build a Chatbot | 🦜️🔗 Langchain</a>: Overview</li><li><a href="https://github.com/ollama/ollama/issues/3509">Can Ollama use both  CPU and GPU for inference? · Issue #3509 · ollama/ollama</a>: What are you trying to do? May I know whether ollama support to mix CPU and GPU together for running on windows? I know my hardware is not enough for ollama, but I still want to use the part abilit...
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1270276986878693398)** (2 messages): 

> - `AgentGenesis`
> - `Open Source Contributions`
> - `AI Development Acceleration` 


- **AgentGenesis Boosts AI Development**: A member presented **AgentGenesis**, an AI component library providing **copy-paste code snippets** for developers to enhance their **Gen AI applications** by 10x, available at [AgentGenesis](https://www.agentgenesis.dev/).
   - The project is **MIT licensed**, emphasizing **developer-friendly solutions** and a **comprehensive code library** for various AI templates, inviting community contributions on [GitHub](https://github.com/DeadmanAbir/AgentGenesis).
- **Collaboration and Code Sharing**: Another member expressed interest in collaborating and asked if Johnny would share his code implementation. *This highlights community engagement and the willingness to share knowledge within the AI development circle.*


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.agentgenesis.dev/">AgentGenesis</a>: Copy paste the most trending AI agents and use them in your project without having to write everything from scratch.</li><li><a href="https://github.com/DeadmanAbir/AgentGenesis">GitHub - DeadmanAbir/AgentGenesis: Welcome to AgentGenesis, your source for customizable Gen AI code snippets that you can easily copy and paste into your applications.</a>: Welcome to AgentGenesis, your source for customizable Gen AI code snippets that you can easily copy and paste into your applications. - DeadmanAbir/AgentGenesis
</li>
</ul>

</div>
  

---



### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1270171674540769311)** (57 messages🔥🔥): 

> - `John Schulman leaves OpenAI`
> - `Anthropic developments`
> - `AI Alignment debates`
> - `GDB takes a sabbatical`
> - `Structured Outputs announcement` 


- **John Schulman leaves OpenAI for Anthropic**: John Schulman announced his departure from OpenAI to focus on AI alignment research at [Anthropic](https://x.com/johnschulman2/status/1820610863499509855), stating a desire for hands-on technical work.
   - He emphasized that his choice is personal, not a sign of lack of support for alignment at OpenAI, highlighting the talent still present at the company.
- **Speculation around GDB's sabbatical**: GDB's decision to take a sabbatical through the end of the year led to discussions questioning the reasoning behind it, including concerns about overwork and possible health issues.
   - Some members speculated that this could be a much-needed break for him after many years of intense engagement in AGI.
- **Debate on AI Alignment Perspectives**: A discussion emerged regarding the differing views on AI alignment, with John Schulman favoring a reinforcement learning perspective, while others believed it transcends traditional methods.
   - This debate reflects broader concerns about the control of superhuman AI and whether alignment is fundamentally a deep learning problem.
- **Structured Outputs Enhancements**: A new announcement detailed the introduction of [Structured Outputs](https://openai.com/index/introducing-structured-outputs-in-the-api/) in API, allowing developers to get consistent schema matches without missing keys.
   - Additionally, the cost of using the gpt-4o-2024-08-06 model was noted to be reduced significantly, saving developers both input and output costs.
- **Reflections on AGI and Personal Motivations**: Members reflected on the motivations behind AGI researchers, sharing thoughts on whether figures like GDB are ideologically driven or simply passionate about their work.
   - This led to comments about GDB’s deep engagement with the mission, including a rumored marriage ceremony at the OpenAI office, supporting the notion of a dedicated 'grinder.'


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/johnschulman2/status/1820610863499509855">Tweet from John Schulman (@johnschulman2)</a>: I shared the following note with my OpenAI colleagues today:  I&#39;ve made the difficult decision to leave OpenAI. This choice stems from my desire to deepen my focus on AI alignment, and to start a ...</li><li><a href="https://fxtwitter.com/simonw/status/1820886987982987413?s=46">Tweet from Simon Willison (@simonw)</a>: Hidden at the bottom of this announcement:  &#34;By switching to the new gpt-4o-2024-08-06, developers save 50% on inputs ($2.50/1M input tokens) and 33% on outputs ($10.00/1M output tokens) compared ...</li><li><a href="https://x.com/gdb/status/1820644694264791459?s=46">Tweet from Greg Brockman (@gdb)</a>: I’m taking a sabbatical through end of year. First time to relax since co-founding OpenAI 9 years ago. The mission is far from complete; we still have a safe AGI to build.
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1270539473066922105)** (6 messages): 

> - `DALL·E vs New Challengers`
> - `Flux Pro`
> - `Flux.1 Hosting on Replicate`
> - `Comparative Analysis of Image Generators` 


- **DALL·E still leading the pack?**: Discussion arose about whether **DALL·E still holds the title** for best image generation via API, as new contenders emerge.
   - *A member questioned the competition* and noted the challenge in making direct comparisons beyond intuition.
- **Flux Pro brings a different vibe**: **Flux Pro** was characterized as having a *really different vibe*, suggesting a distinct approach in image generation.
   - One member expressed curiosity about how this vibe translates into actual performance and user experience.
- **Flux.1 gains popularity on Replicate**: The **new Flux.1** model has garnered attention and is notably hosted on [Replicate](https://replicate.com).
   - Members discussed the difficulty in comparing models and highlighted the potential of conducting further experiments with **videos on Replicate**.


  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/)** (1 messages): 

xeophon.: https://x.com/sahir2k/status/1820791954508022019?s=46
  

---


### **Interconnects (Nathan Lambert) ▷ #[rlhf](https://discord.com/channels/1179127597926469703/1208183230608576562/1270459908806017076)** (1 messages): 

> - `Data Dependence in Model Training`
> - `Startups Utilizing Noise`
> - `Armen from Chameleon at Meta` 


- **Data Dependence is Key**: All claims revolve around the fact that **everything is data-dependent**; breaking (x, y_w, y_l) into (x, y_w) and (x, y_l) only benefits from noisy data.
   - *Noisy enough data* is crucial for effective model training, emphasizing the importance of context in data usage.
- **Startups Favoring Noisy Data**: Startups seem to adopt data strategies that allow for the use of *noisier data*, often enabling them to skip **SFT (Supervised Fine-Tuning)**.
   - This preference indicates a significant trend within startup environments toward embracing more flexible data handling.
- **Armen's Influence on Data Approaches**: A mention at ICML highlighted that **Armen** from **Chameleon** at **Meta** shows a keen interest in these data strategies.
   - However, it remains unclear whether his team is currently implementing these ideas in a **production model**.


  

---



### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1270486884539174973)** (1 messages): 

> - `GPT-4o-2024-08-06`
> - `Structured outputs in strict mode` 


- **New Release of GPT-4o-2024-08-06**: The latest version, [GPT-4o-2024-08-06](https://openrouter.ai/models/openai/gpt-4o-2024-08-06), is now available for use.
   - **OpenRouter, LLC** provides this update, noting its release as part of their continuous improvement efforts.
- **Limited Support for Structured Outputs**: A note indicated that structured outputs with strict mode are **not fully supported** at this time.
   - Users are encouraged to report issues in designated channels: <#1138521849106546791> or <#1107397803266818229>.



**Link mentioned**: <a href="https://openrouter.ai/models/openai/gpt-4o-2024-08-06">GPT-4o (2024-08-06) - API, Providers, Stats</a>: The 2024-08-06 version of GPT-4o offers improved performance in structured outputs, with the ability to supply a JSON schema in the respone_format. Read more [here](https://openai. Run GPT-4o (2024-08...

  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1270098420023164939)** (62 messages🔥🔥): 

> - `Gemini Pro 1.5 performance issues`
> - `OpenRouter API usage`
> - `gpt-4o-2024-08-06 updates`
> - `Structured outputs in the API`
> - `Model pricing changes` 


- **Gemini Pro 1.5 experiences resource exhaustion**: Users reported an error with **Gemini Pro 1.5** stating 'Resource has been exhausted', attributed to heavy rate limiting by Google.
   - It was confirmed that there is no fix for this issue as it's due to Google implementing strict limits on the model.
- **Navigating OpenRouter's API for models**: A member inquired about purchasing models and was informed that models are accessed via API with payment per token usage on **OpenRouter**.
   - New users were advised to explore user interfaces like **Lobe Chat** to simplify interactions with the API.
- **Updates to gpt-4o-2024-08-06**: The **gpt-4o-2024-08-06** model enables developers to save costs with significantly reduced prices compared to previous versions, noted to be **50% cheaper** for inputs and **33% for outputs**.
   - Users are also excited about a new 'refusal' field feature and ongoing discussions regarding enhanced efficiency in model operations.
- **Introduction of structured outputs in OpenAI's API**: OpenAI introduced structured outputs enabling developers to request valid JSON responses directly from the **API**, enhancing reliability.
   - Previous methods were less consistent, but the new approach aims to standardize outputs and improve usability across applications.
- **Model pricing and token limits**: There was a discussion around the **token limit** discrepancy for **gpt-4o-2024-08-06**, where OpenRouter initially displayed a lower maximum than what was documented by OpenAI.
   - Updates were anticipated shortly, as emphasized by users awaiting changes to reflect the accurate capabilities of the latest model.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://chat-preview.lobehub.com>,">no title found</a>: no description found</li><li><a href="https://simonwillison.net/2024/Aug/6/openai-structured-outputs/">OpenAI: Introducing Structured Outputs in the API</a>: OpenAI have offered structured outputs for a while now: you could specify `&quot;response_format&quot;: {&quot;type&quot;: &quot;json_object&quot;}}` to request a valid JSON object, or you could use t...</li><li><a href="https://openrouter.ai/docs/responses#querying-cost-and-stats">Responses | OpenRouter</a>: Manage responses from models</li><li><a href="https://status.anthropic.com">Anthropic Status</a>: no description found</li><li><a href="https://openrouter.ai/models/openai/gpt-4o-2024-08-06">GPT-4o (2024-08-06) - API, Providers, Stats</a>: The 2024-08-06 version of GPT-4o offers improved performance in structured outputs, with the ability to supply a JSON schema in the respone_format. Read more [here](https://openai. Run GPT-4o (2024-08...
</li>
</ul>

</div>
  

---



### **LlamaIndex ▷ #[announcements](https://discord.com/channels/1059199217496772688/1073670729054294197/1270432568948555776)** (1 messages): 

> - `CodiumAI Webinar`
> - `RAG-Augmented coding assistants`
> - `LlamaIndex infrastructure` 


- **Join the CodiumAI Webinar on RAG-Augmented Code Generation**: A reminder was shared about the upcoming [webinar with CodiumAI](https://lu.ma/ka5xtyqo) focused on **RAG-augmented coding assistants**.
   - Participants will need to verify token ownership with their wallet to access the event.
- **Exploring RAG's Role in Code Quality**: The webinar will delve into how **Retrieval-Augmented Generation (RAG)** enhances contextual awareness in AI-generated code.
   - *Contextual awareness* is crucial for enterprises adopting code generation to maintain **high quality** and **integrity** in development.
- **Advanced RAG Approach Using LlamaIndex**: Attendees can expect a presentation on an **advanced approach to RAG** that builds upon the **LlamaIndex** infrastructure.
   - Examples of practical applications demonstrating *context-aware generation* will also be presented.



**Link mentioned**: <a href="https://lu.ma/ka5xtyqo">LlamaIndex Webinar: Using RAG with LlamaIndex for Large-Scale Generative Coding · Zoom · Luma</a>: Retrieval-Augmented Generation (RAG) plays a central role in achieving contextual awareness in AI-generated code, which is crucial for enterprises adopting…

  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1270177453498830960)** (4 messages): 

> - `Local Multi-Agent System with RabbitMQ`
> - `LlamaIndex RAG-a-thon`
> - `Workflows Feature Introduction`
> - `Documentation for llama-agents` 


- **Build a Local Multi-Agent System with RabbitMQ**: A blog by [@pavan_mantha1](https://twitter.com/pavan_mantha1) showcases how to create a local multi-agent system using [RabbitMQ](https://www.rabbitmq.com) for communication between agents, leveraging [ollama](https://ollama.com) and [qdrant_engine](https://qdrant.tech). This setup is facilitated by llama-agents, a key tool for agent development.
   - View the complete guide [here](https://t.co/IOGpDWkY8A).
- **LlamaIndex Hosts Second RAG-a-thon**: LlamaIndex is collaborating with [@pinecone](https://www.pinecone.io) and [@arizeai](https://www.arize.ai) for their second RAG-a-thon, held at the [@500GlobalVC](https://www.500.co) offices in Palo Alto from October 11-13. This weekend hackathon aims to engage developers and innovators in the space.
   - More details can be found [here](https://t.co/N4hWiCv0Nm).
- **Introducing Workflows Feature**: In a new [YouTube video](https://youtu.be/xuiuSMCmJF), [@seldo](https://twitter.com/seldo) explains the Workflows feature that enables users to build complex agentic applications in LlamaIndex. The video covers key elements such as creating, running, and visualizing workflows.
   - It also discusses workflow structure, looping, branching, and state management, providing essential insights for developers.
- **Comprehensive Documentation for llama-agents**: A new guide titled 'A Primer to Building Multi-agents as a Service' addresses user requests for better documentation on llama-agents, which has seen significant advancements recently, thanks to contributions from [@_nerdai_](https://twitter.com/_nerdai_). This core repository is essential for developing multi-agents as a service.
   - For more information, check out the primer [here](https://t.co/k0TEeMi3C5).


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1270128013337432115)** (49 messages🔥): 

> - `HuggingFace Inference API for Embeddings`
> - `Llamaparse Arabic Language Parsing`
> - `SimpleDirectoryReader PDF Handling`
> - `Vector DB Comparison Resource`
> - `Performance of 4o Mini vs 3.5 Turbo` 


- **Using HuggingFace Inference API for Embeddings**: To generate embeddings with the HuggingFace Inference API, one can use the `TextEmbeddingsInference` class, as shown in the provided [LlamaIndex example](https://docs.llamaindex.ai/en/stable/examples/embeddings/text_embedding_inference/).
   - This setup includes parameters like the model name and embedding batch size for efficient processing.
- **Llamaparse struggles with Arabic parsing**: Users noticed that Llamaparse's Arabic language parsing returns results in a Left to Right format despite Arabic being a Right to Left language.
   - It raises questions about whether Llamaparse accommodates the intricacies of Right to Left writing styles.
- **Managing PDF Document Loading in SimpleDirectoryReader**: The `SimpleDirectoryReader` loads a PDF as multiple documents, one per page, to include metadata like `page_label` for each page.
   - Modifications can be made in the `PDFReader` to aggregate content into a single document during load.
- **Vector DB Comparison Resource Shared**: A helpful resource was shared for comparing Vector DBs, which some users found valuable for their projects.
   - Others expressed interest in accumulating user experiences with various Vector DBs for communal learning.
- **Performance Disparities between 4o Mini and 3.5 Turbo**: Users reported that the 4o Mini performs significantly slower than 3.5 Turbo, which is still regarded as superior for speed.
   - Discussion revolved around potential backend scaling issues affecting the initial response times, particularly noted for time to first token.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://superlinked.com/vector-db-comparison">Vector DB Comparison</a>: Vector DB Comparison is a free and open source tool from VectorHub to compare vector databases.</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/embeddings/text_embedding_inference/">Text Embedding Inference - LlamaIndex</a>: no description found</li><li><a href="https://github.com/run-llama/llama_index/blob/6eea66ed23fb85ee77664148a4c2b66720caabeb/pyproject.toml#L60">llama_index/pyproject.toml at 6eea66ed23fb85ee77664148a4c2b66720caabeb · run-llama/llama_index</a>: LlamaIndex is a data framework for your LLM applications - run-llama/llama_index</li><li><a href="https://github.com/run-llama/llama_index/blob/15227173b8c1241c9fbc761342a2344cd90c6593/llama-index-core/llama_index/core/llms/function_calling.py#L125">llama_index/llama-index-core/llama_index/core/llms/function_calling.py at 15227173b8c1241c9fbc761342a2344cd90c6593 · run-llama/llama_index</a>: LlamaIndex is a data framework for your LLM applications - run-llama/llama_index</li><li><a href="https://github.com/run-llama/llama_index/blob/15227173b8c12">GitHub - run-llama/llama_index at 15227173b8c1241c9fbc761342a2344cd90c6593</a>: LlamaIndex is a data framework for your LLM applications - GitHub - run-llama/llama_index at 15227173b8c1241c9fbc761342a2344cd90c6593
</li>
</ul>

</div>
  

---



### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1270216114999398435)** (29 messages🔥): 

> - `LLM Hallucination Index`
> - `Open Source Definitions`
> - `Mistral Open Weights`
> - `Command R Plus Licensing`
> - `Commercial Use Restrictions` 


- **LLM Hallucination Index Updates**: The [LLM Hallucination Index](https://www.rungalileo.io/hallucinationindex?utm_medium=paid&utm_source=alpha_signal&utm_campaign=sp) evaluates how well leading models adhere to context, highlighting concerns about hallucinations as the term was named Word of the Year.
   - Members are questioning the accuracy of the index in regard to **Command R Plus**, which some believe misrepresents its open-source status.
- **Confusion Over Open Source Definition**: Several members expressed disagreement with the definition of open source provided by the Hallucination Index, stating that merely releasing weights is insufficient for true open-source status.
   - *Some suggest that additional details such as the dataset and training methods should also be disclosed for full transparency*.
- **Mistral's Open Weights Clarity**: Members discussed that **Mistral** models operate under the **Apache 2.0** license, implying they qualify as open source despite limitations in dataset access.
   - However, there's contention surrounding the true definition of open source in AI, emphasizing that many models available are just 'open weights' with various usage stipulations.
- **Commercial Use Issues with Command R Plus**: Members pointed out that **Command R Plus** is not open source because it operates under a Creative Commons Attribution Non Commercial 4.0 license.
   - This led to debate over the inadequacies in the paper's definition of open source, with some members planning to reach out for clarification.
- **Discussion on License Implications**: A member concluded that, while the model **Command R Plus** has 'Open Weights', the non-commercial restriction effectively makes it closed-source in practice.
   - *Discussions highlight the complexity of licensing in AI, where the distinction between open weights and true open-source can be ambiguous.*


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.mistral.ai/getting-started/open_weight_models/">Open weight models | Mistral AI Large Language Models</a>: We open-source both pre-trained models and instruction-tuned models. These models are not tuned for safety as we want to empower users to test and refine moderation based on their use cases. For safer...</li><li><a href="https://www.rungalileo.io/hallucinationindex?utm_medium=paid&utm_source=alpha_signal&utm_campaign=sp">LLM Hallucination Index - Galileo</a>: LLM Hallucination Index. A Ranking &amp; Evaluation Framework For LLM Hallucinations</li><li><a href="https://www.rungalileo.io/hallucinationindex?utm_medium=paid&utm_source=alpha_signal&utm_campaign=sponsorship">LLM Hallucination Index - Galileo</a>: LLM Hallucination Index. A Ranking &amp; Evaluation Framework For LLM Hallucinations
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1270432110078853241)** (3 messages): 

> - `Contacting Dennis Padilla`
> - `Lauren's absence`
> - `Email inquiries` 


- **Seeking Dennis Padilla's Email**: A member requested assistance in obtaining the email address for **Dennis Padilla** since he was directed to contact him while **Lauren** is on vacation.
   - The member expressed difficulty in finding the contact information and is looking for guidance on how to reach out.
- **Discussion on Lauren's Vacation**: **Lauren's absence** was noted as she is currently on vacation, prompting the inquiry for her alternative contact.
   - The member mentioned relying on **Dennis Padilla** as the next point of contact for their ongoing communication.


  

---


### **Cohere ▷ #[cohere-toolkit](https://discord.com/channels/954421988141711382/1254901651081269268/1270539094203568128)** (1 messages): 

> - `Cohere Toolkit for Learning`
> - `LLMs with RAG`
> - `Model Deployment`
> - `Third-Party API Integration` 


- **Cohere Toolkit aids learning initiatives**: The Cohere team is leveraging the **Cohere Toolkit** for a learning project as part of an AI fellowship, aiming to build an **LLM with RAG** over a knowledge base.
   - They are exploring the potential of different types of corpora, such as **recipes**, **cooking notes**, and **legal case notes**.
- **Model deployment query arises**: A member inquired whether anyone has successfully switched from **Cohere models** to a third-party **API-based** model like **OpenAI Chat GPT** or **Gemini 1.5**.
   - The interest includes models accessed via the **Groq API**, highlighting a pursuit of broader capabilities.


  

---



### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1270321995027972166)** (30 messages🔥): 

> - `InlineList features`
> - `Optimization for List in Mojo`
> - `Custom hardware accelerators`
> - `CXL protocol and FPGA integration`
> - `Mojo's open source future` 


- **InlineList lacks move and copy initialization**: `InlineList` currently does not have `__moveinit__` and `__copyinit__`, but significant progress is being made, with important features being merged soon.
   - A member noted that developments involving `InlineList` are prioritized before adding these functionalities.
- **List vs. InlinedFixedVector distinction**: A member clarified that `InlinedFixedVector` is intended for `AnyTrivialRegType`, while `List` serves `CollectionElement`, explaining their distinct purposes in Mojo.
   - Discussion also included a potential small buffer optimization for `List` in an open pull request.
- **Potential for custom accelerators with Mojo**: Members discussed using custom accelerators like PCIe cards and whether Mojo would support them before an open-source release, with insights on performance concerns.
   - It was mentioned that cxl.mem functionality would likely rely on hardware-level integration to ensure performance efficiency.
- **Discussion on FPGA and CXL IP blocks**: Members exchanged insights on hardware development with Xilinx VU13P FPGAs, mentioning the incorporation of CXL IP blocks.
   - A member shared their ongoing project plans to optimize kernel usage by replacing it with custom programs.
- **Anticipation for Mojo's open-source capabilities**: There is excitement around the future potential of Mojo once open-sourced, particularly for supporting RISC-V vector extensions.
   - Members expressed hope that Mojo will become beneficial for their projects, despite current limitations in compatibility.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/modula">modula - Overview</a>: GitHub is where modula builds software.</li><li><a href="https://github.com/modularml/mojo/pull/2825">[stdlib] Add optional small buffer optimization to `List`, take 2 by gabrieldemarmiesse · Pull Request #2825 · modularml/mojo</a>: This PR solves part of #2467 This PR is part of three PRs to read and merge in the following order  [stdlib] Add optional small buffer optimization to List, take 2 #2825 [stdlib] Work around the ma...
</li>
</ul>

</div>
  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1270190953902374913)** (18 messages🔥): 

> - `John Schulman joins Anthropic`
> - `Open-source AI challenges`
> - `Meta's JASCO status`
> - `Nullbulge doxxing incident`
> - `Voice Assistant project` 


- **John Schulman leaves OpenAI for Anthropic**: OpenAI co-founder **John Schulman** announced he would join **Anthropic**, an AI startup backed by **Amazon**, in a post on Monday.
   - This move follows OpenAI's recent dissolution of a **superalignment team** that worked on ensuring control over advanced AI systems.
- **Challenges facing open-source AI development**: **Open-source AI** is lagging due to the high costs of training state-of-the-art models and challenges in collecting preference data, which is crucial for development.
   - The lack of access to unlicensed data contributes to these constraints, leading to fewer open models being developed.
- **Meta's JASCO rumor mill**: There's speculation surrounding **Meta's JASCO**, with comments suggesting it has gone missing and concerns raised about potential lawsuits from **Udio** and **Suno**.
   - This situation appears to have introduced hesitation in Meta’s advancements in the AI space.
- **Doxxing incident of Nullbulge**: **Nullbulge** was reportedly doxxed, stirring a discussion about the implications for individual privacy and online reputation.
   - Participants in the chat suggested that this will likely not pose a significant problem in the future due to perceived weaknesses in his operational security.
- **Voice Assistant project on YouTube**: A YouTube link was shared featuring a video titled **


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.cnbc.com/2024/08/06/openai-co-founder-john-schulman-says-he-will-join-rival-anthropic.html">OpenAI co-founder John Schulman says he will leave and join rival Anthropic</a>: Schulman said OpenAI executives remain committed to backing efforts to ensure that people can control highly capable artificial intelligence models.</li><li><a href="https://youtu.be/DdAwEdlVi14">School BUD-E web-browser Voice Assistant</a>: no description found</li><li><a href="https://archive.ph/TmDrg">Trio of Leaders Leave OpenAI &#x2014; The Information</a>: no description found
</li>
</ul>

</div>
  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1270444227205009478)** (8 messages🔥): 

> - `Model Scaling`
> - `Validation Accuracy`
> - `CIFAR Image Processing` 


- **Model hits an accuracy wall at 270k parameters**: The **270k model** seems to hit nearly the same accuracy wall as the smaller models, with **84% validation accuracy** reported.
   - *I'm starting to believe* this setup shows diminishing returns despite the increase in parameters.
- **Exploration of CIFAR images in frequency domain**: A member inquired about how **CIFAR images** appear when processed through the frequency domain transform (FTT).
   - They wondered if the frequency information remains consistent while the phase changes, seeking insights into the differences.



**Link mentioned**: <a href="https://tenor.com/view/the-matrix-laurence-fishburne-morpheus-trinity-he%27s-beginning-to-believe-gif-18413151103009905935">The Matrix Laurence Fishburne GIF - The matrix Laurence fishburne Morpheus - Discover &amp; Share GIFs</a>: Click to view the GIF

  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1270096760634736660)** (8 messages🔥): 

> - `Running tinygrad on Aurora`
> - `XMX support and OpenCL on Intel GPUs`
> - `NVIDIA FP8 Bounty Support`
> - `Distributed computing functionality for tinygrad`
> - `Aurora supercomputer performance expectations` 


- **Feasibility of tinygrad on Aurora**: A member questioned whether it would be remotely feasible to run **tinygrad** on **Aurora** at **Argonne National Laboratory** given the limitations of its Intel GPUs.
   - Another member highlighted that these GPUs support tensor core instructions comparable to the **A770s**.
- **Insights on XMX support and OpenCL**: Discussion emerged regarding the potential for **XMX support** on Aurora, noting that **OpenCL** compatibility exists but might be slow.
   - Specific details were shared about how Intel's Max Data Center GPUs and the subgroup matrix functions may play a role.
- **Need for Distributed Computing Features**: There was a mention that more mature functionality for **distributed computing** would be necessary for improving tinygrad.
   - The conversation emphasized the importance of enhancing distributed capabilities to leverage Aurora's potential.
- **Dual Support Needed for FP8 NVIDIA Bounty**: A member asked whether support for **E4M3**, **E5M2**, or both was desired for the FP8 NVIDIA bounty.
   - Georgehotz responded affirmatively, stating that support for **both** was preferred.
- **Aurora Supercomputer Capabilities**: A member noted that **Aurora** is expected to exceed **2 ExaFLOPS**, making it potentially the fastest computer ever.
   - For more details, the conversation referenced a [Wikipedia page on Aurora](https://en.wikipedia.org/wiki/Aurora_(supercomputer)) explaining its power and purpose.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroup_matrix_multiply_accumulate.html">cl_intel_subgroup_matrix_multiply_accumulate</a>: no description found</li><li><a href="https://en.wikipedia.org/wiki/Aurora_(supercomputer)">Aurora (supercomputer) - Wikipedia</a>: no description found
</li>
</ul>

</div>
  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1270095328476921857)** (16 messages🔥): 

> - `Preallocating Tensors`
> - `Buffer and DEFINE_GLOBAL Mapping`
> - `Batch Size Handling in JIT`
> - `Computer Algebra Study Notes`
> - `OpenMP in CLANG and LLVM` 


- **Effectively Preallocate Tensors**: A suggestion was made that preallocating and then assigning to a slice might resolve issues encountered in tensor manipulation.
   - *George* confirmed that making tensors contiguous should help fix the problem.
- **Identifying Buffers for DEFINE_GLOBAL**: Discussion centered around how to map `Buffer` instances back to their corresponding `DEFINE_GLOBAL` during tensor operations.
   - *Eigenvector42* expressed uncertainty in the flow from Tensor to Buffer to MemBuffer.
- **Handling Batch Sizes with JIT**: A user raised concerns about consistent batch sizes leading to JIT errors due to dataset subdivision issues.
   - *George* advised skipping the last batch or avoiding the JIT on it as a solution.
- **New Resource on Computer Algebra**: A link was shared to study notes on computer algebra, highlighting its relevance for theoretical understanding post reading about symbolic math.
   - The notes are available on [GitHub](https://github.com/mesozoic-egg/computer-algebra-study-notes/blob/main/README.md).
- **CLANG and LLVM Threading**: *Cecilian* inquired about threading in CLANG and LLVM, receiving confirmation that they primarily use a single thread.
   - A potential enhancement using OpenMP was referenced with links to *tinygrad* GitHub pull requests.



**Link mentioned**: <a href="https://github.com/mesozoic-egg/computer-algebra-study-notes/blob/main/README.md">computer-algebra-study-notes/README.md at main · mesozoic-egg/computer-algebra-study-notes</a>: Contribute to mesozoic-egg/computer-algebra-study-notes development by creating an account on GitHub.

  

---



### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1270433513476456580)** (6 messages): 

> - `Wiseflow tool`
> - `HybridAGI release`
> - `Dynamic knowledge base` 


- **Introducing Wiseflow for Information Mining**: [Wiseflow](https://github.com/TeamWiseFlow/wiseflow) is a new agile information mining tool that extracts concise messages from various sources including websites and social platforms, automatically categorizing them into a database.
   - This tool aims to enhance data organization and retrieval for users engaged in information-rich environments.
- **Dynamic Knowledge Base Discussion**: A member contextualized a 'dynamic knowledge base' in relation to ongoing projects, hinting towards integrations with existing tools.
   - The member invoked curiosity from others about any tangible creations demonstrated from this idea.
- **HybridAGI Enhancements Unveiled**: The latest version of [HybridAGI](https://github.com/SynaLinks/HybridAGI) introduces a neuro-symbolic system built around graphs and graph-program synthesis, featuring various notebooks to optimize RAG (Retrieval-Augmented Generation) with DSPy.
   - Improvements focus on usability and data processing pipelines, providing a simplified interface for working with Knowledge Graphs.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/TeamWiseFlow/wiseflow">GitHub - TeamWiseFlow/wiseflow: Wiseflow is an agile information mining tool that extracts concise messages from various sources such as websites, WeChat official accounts, social platforms, etc. It automatically categorizes and uploads them to the database.</a>: Wiseflow is an agile information mining tool that extracts concise messages from various sources such as websites, WeChat official accounts, social platforms, etc. It automatically categorizes and ...</li><li><a href="https://github.com/SynaLinks/HybridAGI">GitHub - SynaLinks/HybridAGI: The Programmable Cypher-based Neuro-Symbolic AGI that lets you program its behavior using Graph-based Prompt Programming: for people who want AI to behave as expected</a>: The Programmable Cypher-based Neuro-Symbolic AGI that lets you program its behavior using Graph-based Prompt Programming: for people who want AI to behave as expected - SynaLinks/HybridAGI
</li>
</ul>

</div>
  

---


### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/1270284030604218450)** (2 messages): 

> - `Large Language Models in Software Engineering`
> - `Scaling Inference Compute for Language Models` 


- **LLMs evolving towards AGI with agents**: Research explores the transition from Large Language Models (LLMs) to **LLM-based agents** aimed at addressing limitations such as lack of autonomy and self-improvement, as noted in the [study](https://arxiv.org/abs/2408.02479).
   - *The need for a unified standard* and benchmarking for qualifying LLM solutions as agents is emphasized in this early-stage exploration.
- **Inference compute boosts model performance**: A recent study highlights that increasing the number of generated samples during inference can significantly improve performance, evidenced by a jump in issue resolution rates from **15.9%** to **56%** with additional sampling in the [SWE-bench Lite](https://arxiv.org/abs/2407.21787) context.
   - Such approaches demonstrate that *coverage* scales with the number of samples, directly benefiting domains like coding and formal proofs.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2407.21787">Large Language Monkeys: Scaling Inference Compute with Repeated Sampling</a>: Scaling the amount of compute used to train language models has dramatically improved their capabilities. However, when it comes to inference, we often limit the amount of compute to only one attempt ...</li><li><a href="https://arxiv.org/abs/2408.02479">From LLMs to LLM-based Agents for Software Engineering: A Survey of Current, Challenges and Future</a>: With the rise of large language models (LLMs), researchers are increasingly exploring their applications in var ious vertical domains, such as software engineering. LLMs have achieved remarkable succe...
</li>
</ul>

</div>
  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1270224404428623963)** (7 messages): 

> - `MIPRO vs BootstrapFewShotWithRandomSearch`
> - `MIPROv2 assertions`
> - `Complexity in model training` 


- **MIPRO often outperforms BootstrapFewShotWithRandomSearch**: The query about whether **MIPRO** always performs better than **BootstrapFewShotWithRandomSearch** received a response noting it 'often, but not necessarily' does.
   - This highlights that while MIPRO has strong performance, it isn't a guaranteed outcome.
- **MIPROv2 does not yet support assertions**: A member inquired if **MIPROv2** supports assertions, to which the response was a clear 'not yet'.
   - This suggests potential future developments or updates might include assertion support.
- **Start simple in model training**: A suggestion was made to 'always start simple', recommending **random search** before progressing to **MIPRO**.
   - This serves as a strategy to gradually add complexity in training models for more efficient results.


  

---


### **DSPy ▷ #[colbert](https://discord.com/channels/1161519468141355160/1250300504462856265/)** (1 messages): 

gamris: Would you recommend FastEmbed by Qdrant instead? https://github.com/qdrant/fastembed
  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1270177649754374165)** (7 messages): 

> - `Synthetic Data Generation`
> - `Llama Index SQL Examples`
> - `LoRA Adapter MD5 Consistency`
> - `BitsAndBytes GitHub Pull Request` 


- **Synthetic data generation strategy for reasoning tasks**: A member inquired about effective strategies for synthetic data generation to improve **8b models** on reasoning tasks like **text to SQL** through **Chain of Thought (CoT)** training.
   - They suggested that utilizing synthetic instructions prior to generating final SQL queries might enhance performance.
- **SQL examples found in Llama Index**: A member noted that the **Llama Index** contains several SQL examples, which could be beneficial for those needing resources.
   - This reference could assist others looking to implement SQL within their projects or experiments.
- **MD5 hash consistency for LoRA adapters**: A discussion arose around the expectations of **MD5 hashes** when merging a **LoRA adapter** multiple times, focusing on whether the hashes should remain consistent.
   - A member confirmed that they should indeed be the same, indicating that differences would suggest an issue.
- **Tracking the right branch on BitsAndBytes GitHub**: Another member highlighted the importance of following a specific branch on GitHub, referencing a pull request from the **BitsAndBytes Foundation** ([#1220](https://github.com/bitsandbytes-foundation/bitsandbytes/pull/1220)).
   - This could be critical information for those engaged in related development or troubleshooting.


  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1270107609386582199)** (5 messages): 

> - `QLoRA for Gemma 2 27b`
> - `Performance on L40S GPUs`
> - `Faster pip for Docker` 


- **Tweaking QLoRA for Gemma 2 27b Training**: It was mentioned that the **QLoRA** for **Gemma 2 27b** might require tweaking of the **learning rate** but should be compatible with the latest **Flash Attention**.
   - *Colejhunter* expressed their intention to give it a try.
- **Decent Performance on L40S GPUs**: A member noted that training on **L40S GPUs** yields pretty decent performance, which piqued curiosity about the specifics.
   - This response came after an inquiry about **performance metrics** from fellow members.
- **Potential of Faster Pip for Docker Builds**: A link to a resource on **Faster pip** was shared, implying that it might be advantageous for **Docker building**.
   - The member seemed optimistic about its utility for streamlining build processes.


  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1270197885526347957)** (3 messages): 

> - `Fine-tuning context length`
> - `RoPE scaling`
> - `Editing unique samples in Python` 


- **Adjusting Context Length During Fine-tuning**: A member inquired about the ability to adjust the context length of a fine-tuned model like **llama2-13b-hf** after fine-tuning to **4k**.
   - Another member confirmed that *you can increase and decrease it as you like* and suggested a stepwise approach for large increases.
- **RoPE Scaling as a Quick Fix**: In response to the context length question, a member hinted at using **RoPE scaling** for quick adjustments.
   - They noted that *if you plan to increase it a very large amount,* it’s better to do it gradually for optimal performance.
- **Editing Unique Samples Uncertainty**: A member expressed curiosity about the clarity of editing unique samples, mentioning that previous tools required Python interventions.
   - The uncertainty remains whether similar editing needs will apply to the current tools being discussed.


  

---


### **OpenAccess AI Collective (axolotl) ▷ #[announcements](https://discord.com/channels/1104757954588196865/1113462842436354149/)** (1 messages): 

caseus_: Office hours kicks off in an hour in <#1268285745555308649>.
  

---



### **Torchtune ▷ #[announcements](https://discord.com/channels/1216353675241590815/1216353675241590818/1270120968878293013)** (1 messages): 

> - `PPO training recipe`
> - `Qwen2 model support`
> - `Feature requests for torchtune` 


- **Introducing PPO Training Recipe in Torchtune**: An end-to-end **PPO training recipe** has been added to Torchtune, enabling **RLHF** capabilities in your models. Check out the detailed implementation [here](https://github.com/pytorch/torchtune/pull/1005).
   - This addition streamlines integration between reinforcement learning and Torchtune's toolkit, enhancing training options.
- **Qwen2 Models Supported in Training Recipes**: Support for **Qwen2 models** has been integrated into Torchtune's training recipes, with the **7B** model now available. The upcoming **1.5B** and **0.5B** versions are set to be released soon, adding to the versatility of the toolkit [here](https://github.com/pytorch/torchtune/pull/1143).
   - This expansion opens up more possibilities for model experimentation and tuning within the community.
- **Calling for Feature Requests in Torchtune**: The team invites users to submit their **feature requests** for new models or recipes they would like to see in Torchtune. Feedback can be shared via the [GitHub repository](https://github.com/pytorch/torchtune).
   - This demonstrates an ongoing commitment to community involvement and improvements within the Torchtune ecosystem.


  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1270207545344135304)** (9 messages🔥): 

> - `DPO support for Llama 3`
> - `Differences in Model Output`
> - `Model Download Method`
> - `Prompt Formatting for Instruct Model` 


- **DPO Support Planned for Llama 3**: A member inquired about the potential for supporting **DPO** with **Llama 3 8B full finetune**, indicating interest in the model's enhancements.
   - Another member shared that any of the models can be utilized with the recipes, even without a pre-built configuration.
- **Output Discrepancies in Model Use**: One user reported a difference in outputs when using the **LLAMA 3 8B instruct model** compared to a playground, suspecting they might be using a **BASE model** instead.
   - This confusion was addressed with suggestions to ensure their tokenizer and prompt were correctly structured.
- **Model Download Instructions Shared**: A member shared their download command for the **Meta-Llama-3-8B-instruct** model, specifying the output directory and required access token.
   - They expressed concerns about potentially not using the **INSTRUCT model**, even after following the download process.
- **Proper Prompt Formatting Discussed**: Questions arose regarding the need to format prompts with the **Llama 3 instruct template**, prompting a discussion about output requirements.
   - Another member confirmed that this process is automatically handled by the tokenizer, simplifying user input.


  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1270440954653966360)** (6 messages): 

> - `Model Builders Page`
> - `PreferenceDataset Refactor`
> - `Model Index Page`
> - `Multimodal LLMs` 


- **Proposal for Dedicated Model Builders Pages**: A member suggested that it may be beneficial to create a **dedicated page** for each model's builders to accommodate the growing number of models and **multimodal LLMs**.
   - *This would allow us to better explain repetitive details like downloading and configuring models,* consolidating information for users.
- **Support Chat in Refactored PreferenceDataset**: A member highlighted the newly refactored **PreferenceDataset** that supports chat functionalities, as detailed in [Pull Request #1276](https://github.com/pytorch/torchtune/pull/1276).
   - This update aligns with the unified **message_transform** pipeline established in previous discussions.
- **Interest in Model Index Page**: There was a consensus on the need for a **model index page** to explain essential yet repetitive tasks related to models.
   - This idea was well-received as it would streamline the process of managing and configuring various models.



**Link mentioned**: <a href="https://github.com/pytorch/torchtune/pull/1276">[4/7] Refactor preference dataset with transforms design by RdoubleA · Pull Request #1276 · pytorch/torchtune</a>: Context Following the RFC in #1186, we will use the unified message_transform -&amp;gt; template -&amp;gt; tokenization data pipeline in all our datasets. This PR updates PreferenceDataset to follow t...

  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1270243386523390075)** (9 messages🔥): 

> - `Setting up Open Interpreter`
> - `Open Interpreter security and privacy`
> - `Python compatibility for Open Interpreter`
> - `Vision models for Open Interpreter` 


- **Troubleshooting Open Interpreter Setup**: Users report issues with setting up Open Interpreter, particularly when selecting a local Llama model and encountering an **openai.APIConnectionError** during execution.
   - *One user reported that their model attempted to download again even after selection.*
- **Inquiry on Open Interpreter's Security Measures**: A member expressed interest in how **Open Interpreter** handles user data, particularly regarding whether user's data stays on their local machine.
   - They also inquired about end-to-end encryption standards and the involvement of third parties during communication.
- **Python Version Compatibility for Open Interpreter**: A member inquired if Open Interpreter works with **Python 3.12**, reflecting doubt as a beginner in programming.
   - In response, another member clarified that currently, **Python 3.10** or **3.11** are required for compatibility.
- **Recommendation for Open Source Vision Models**: One user asked for recommendations on which **open source model** is best suited for vision tasks.
   - However, the discussion did not provide specific suggestions related to this inquiry.


  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1270185242996637756)** (2 messages): 

> - `Ollama Models`
> - `API Key Requirements`
> - `Deepgram Support` 


- **Ollama Model List Command**: A member advised using the command `ollama list` to display the different model names available, noting that each model requires a specific amount of **VRAM** on the graphics card.
   - Instructions for running the models can be found in the [Ollama documentation](https://github.com/OpenInterpreter/open-interpreter/blob/main/docs/language-models/local-models/ollama.mdx) which emphasize the need for models to fit the available resources.
- **API Keys for Remotely Hosted Models**: It's highlighted that an **API key** is necessary for accessing paid remotely hosted models.
   - Additionally, it was mentioned that the local model would operate on a designated **port** where it's running.
- **Inquiry About Deepgram Support**: A member inquired whether the system has **support for Deepgram**.
   - This question indicates an interest in potential integrations for speech recognition capabilities.



**Link mentioned**: <a href="https://github.com/OpenInterpreter/open-interpreter/blob/main/docs/language-models/local-models/ollama.mdx">open-interpreter/docs/language-models/local-models/ollama.mdx at main · OpenInterpreter/open-interpreter</a>: A natural language interface for computers. Contribute to OpenInterpreter/open-interpreter development by creating an account on GitHub.

  

---



### **Mozilla AI ▷ #[announcements](https://discord.com/channels/1089876418936180786/1089876419926032396/1270115495273697293)** (2 messages): 

> - `Llamafile updates`
> - `Mozilla AI community opportunities`
> - `sqlite-vec release party`
> - `Machine Learning Paper Talks`
> - `Local AI AMA` 


- **Llamafile makes significant strides**: <@723709999452389417> continues to make epic progress on Llamafile, providing offline, accessible LLMs in a single file.
   - *Community members expressed excitement about the project* and its potential impact on accessibility.
- **Mozilla AI community seeks input with rewards**: The community is encouraging feedback through a survey, offering a chance to win a **$25 gift card** for participating.
   - Members are invited to share how **Mozilla AI** can better support them through community resources.
- **Join the sqlite-vec release party**: Everyone is invited to the [sqlite-vec release party](https://discord.com/events/1089876418936180786/1265715263999836210) for discussions on features and demos led by core maintainer <@533894367354552330>.
   - *Participants can try demos and engage directly with the core team*, enriching their experience.
- **Engaging Machine Learning Paper Talks**: Upcoming **Machine Learning Paper Talks** feature *Communicative Agents* and *Extended Mind Transformers*, hosted by <@718891366402490439>.
   - These events promise to delve into cutting-edge research and stimulate discussion among attendees.
- **Local AI AMA offers insights**: An [AMA](https://discord.com/events/1089876418936180786/1268967945216721079) is scheduled with <@1051191818127147110>, core maintainer of Local AI, which is an open-source alternative for self-hosting.
   - This presents an opportunity for community members to ask questions and learn about practical implementations.



**Link mentioned**: <a href="https://form.typeform.com/to/Cn4md4Oc>)">Discover Typeform, where forms = fun</a>: Create a beautiful, interactive form in minutes with no code. Get started for free.

  

---



### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1270412626798841970)** (1 messages): 

> - `LinkedIn Engineering ML Platform` 


- **LinkedIn Engineering transforms their ML platform**: LinkedIn is currently hosting a live event discussing how their engineering team has transformed their **ML platform**.
   - You can join the discussion [here](https://www.linkedin.com/events/flytepipelinesinactionwithlinke7218669945767776256/theater/).
- **Live Event Awareness**: The event is actively ongoing, providing insights into the latest developments in **machine learning** at LinkedIn.
   - Participants are encouraged to engage and share their thoughts during the event.


  

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
