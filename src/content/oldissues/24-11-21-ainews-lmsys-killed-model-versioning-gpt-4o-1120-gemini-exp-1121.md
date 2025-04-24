---
id: 9853bcdc-3ed8-46e6-96b3-4e9c7e438e70
title: LMSys killed Model Versioning (gpt 4o 1120, gemini exp 1121)
date: '2024-11-22T00:56:03.268058Z'
original_slug: ainews-lmsys-killed-model-versioning-gpt-4o-1120
description: >-
  The AI frontier labs continue a fierce race with Google’s Gemini-Exp-1121
  reclaiming the top lmsys spot from OpenAI’s controversial GPT-4o-2024-11-20,
  which is suspected to be a downgraded but faster model. Meanwhile, DeepSeek R1
  gains recognition as a strong open-source competitor amid escalating US-China
  AI rivalry.
tags:
  - google
  - gemini-exp-1121
  - openai
  - gpt-4o-2024-11-20
  - deepseek
  - model-race
  - us-china-competition
---


<!-- buttondown-editor-mode: plaintext -->**Dates are all you need.**

> AI News for 11/21/2024-11/22/2024. We checked 7 subreddits, [**433** Twitters](https://twitter.com/i/lists/1585430245762441216) and **30** Discords (**217** channels, and **2501** messages) for you. Estimated reading time saved (at 200wpm): **237 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

Frontier lab race dynamics are getting somewhat ridiculous. We used to have a rule that new SOTA models always get top spot, and [reported on Gemini Exp 1114 last week](https://buttondown.com/ainews/archive/ainews-gemini-experimental-1114-retakes-1-llm-9071/) even though there was next to no useful detail on it beyond their lmsys ranking. But yesterday OpenAI overtook them again with [gpt-4o-2024-11-20](https://x.com/lmarena_ai/status/1859307979184689269), which we fortunately didn't report on (thanks to DeepSeek R1), because it is now [suspected of being a worse (but faster) model](https://x.com/ArtificialAnlys/status/1859614633654616310) (we don't know if this is true but it would be a very serious accusation indeed for OpenAI to effectively brand a "mini" model as a mainline model and hope we don't notice), and meanwhile, today [Gemini Exp 1121](https://x.com/lmarena_ai/status/1859673146837827623) is out -again- retaking the top lmsys spot from OpenAI.

It's getting so absurd that [this joke](https://x.com/adonis_singh/status/1859682100569571399) playing on [OpenAI-vs-Gemini-release-coincidences](https://buttondown.com/ainews/archive/ainews-the-ai-search-wars-have-begun-searchgpt/) is somewhat plausible:

![image.png](https://assets.buttondown.email/images/1f200ce6-01cb-4ebf-b385-14cb782c8c52.png?w=960&fit=max)

The complete suspension of all model release decorum is always justifiable under innocent "we just wanted to get these out into the hands of devs ASAP" type good intentions,  but we are now in a situation where all three frontier labs (reminder that Anthropic, despite [their snark](https://x.com/alexalbert__/status/1859676984768688231?s=46
), has also been [playing the date-update-with-no-versioning game](https://buttondown.com/ainews/archive/ainews-claude-35-sonnet-new-gets-computer-use/)) have SOTA model variants uniquely only identified by their dates rather than their versions, in order to keep up on Lmsys.

![image.png](https://assets.buttondown.email/images/5faaf02f-bfad-471c-b6ae-8eb644203293.png?w=960&fit=max)

Are we just not doing versioning anymore? Hopefully we are, because we're still talking about o2 and gpt5 and claude4 and gemini2, but this liminal lull as the [100k clusters](https://x.com/ServeTheHome/status/1850917031421399543) ramp up is a rather local minima nobody is truly happy with.

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

**Theme 1. DeepSeek and Global AI Advancements**

- **[DeepSeek's R1 Performance](https://twitter.com/_philschmid/status/1859482879413158062)**: **DeepSeek R1** is compared to **OpenAI o1-preview** with "thoughts" streamed directly without MCTS used during inference. [@saranormous](https://twitter.com/saranormous/status/1859455354024927521) highlights the model's potency, suggesting **chip controls are ineffective** against burgeoning competition from China, underlined by [@bindureddy](https://twitter.com/bindureddy/status/1859598807979393527) who praises the open-source nature of R1.
  - **Market Impact and Predictions**: **Deepseek-r1** gains traction as a competitive alternative to existing leaders like OpenAI, further emphasized by discussion of the **AI race between China and the US**.

**Theme 2. Model Releases and Tech Developments**

- **[Google's Gemini-Exp-1121](https://twitter.com/lmarena_ai/status/1859673146837827623)**: This model is being lauded for its **improvements in vision, coding, and creative writing**. [@Lmarena_ai](https://twitter.com/lmarena_ai/status/1859673146837827623) discusses its ascendance in the **Chatbot Arena rankings** alongside GPT-4o, showcasing rapid gains in performance.
  - **Enhanced Features**: New **coding proficiency**, **stronger reasoning**, and **improved visual understanding** have made Gemini-Exp-1121 a formidable force, per [@_akhaliq](https://twitter.com/_akhaliq/status/1859713144710729853).

- **[Mistral's Expansion](https://twitter.com/dchaplot/status/1859398052500721943)**: **MistralAI** announces a new office in Palo Alto, indicating growth and open positions in various fields. This expansion reflects a strategic push to scale operations and talent pool, as noted by [@sophiamyang](https://twitter.com/sophiamyang/status/1859400690210103557).

- **[Claude Pro Google Docs Integration](https://twitter.com/alexalbert__/status/1859664138072621228)**: **Anthropic** enhances Claude AI with **Google Docs integration**, aiming to streamline **document management** across organization levels.

**Theme 3. AI Frameworks and Dataset Releases**

- **[SmolTalk Dataset Unveiling](https://twitter.com/_philschmid/status/1859598525723488478)**: **SmolTalk**, a 1M sample dataset under Apache 2.0, boosts **SmolLM v2's performance** with new synthetic datasets. This initiative promises to enhance various model outputs, like summarization and rewriting.
  - **Dataset Integration and Performance**: The dataset couples with public sources like **OpenHermes2.5** and outperforms competitors trained on similar model scales, positioning it as a high-impact resource in language model training.

**Theme 4. Innovative AI Applications and Tools**

- **[LangGraph Agents and LangChain's Voice Capabilities](https://twitter.com/LangChainAI/status/1859643185363902719)**: A video tutorial illustrates the transformation of LangGraph agents into **voice-enabled assistants** using **OpenAI's Whisper** for input and **ElevenLabs** for speech output.
  - **OpenRecovery's Use of LangGraph**: Highlighted by [LangChain](https://twitter.com/LangChainAI/status/1859613490081824845), the application in addiction recovery demonstrates its **practical adaptiveness and scalability**.

**Theme 5. Benchmarks and Industry Analysis**

- **[AI Performance and Industry Insights](https://twitter.com/maximelabonne/status/1859591100475888123)**: Menlo Ventures releases a report on **Generative AI's evolution**, emphasizing top use cases and integration strategies, noting Anthropic's growing share in the market.
  - **Model Fine-tuning and Evaluation**: A shift from **fine-tuning** to more advanced **RAG and agentic AI techniques** is reported, underscoring the value of LLM engineers in optimizing AI applications.

**Theme 6. Memes/Humor**

- **[Misadventures with AI and OpenAI](https://twitter.com/aidan_mclau/status/1859445818031210880)**: [@aidan_mclau](https://twitter.com/aidan_mclau/status/1859445818031210880) humorously contemplates the challenges of fitting new language model behaviors into neatly defined categories, reflecting the often unpredictable nature of ongoing AI developments.
  - **[Finance Humor and Predictions](https://twitter.com/nearcyan/status/1859426783663448349)**: [@nearcyan](https://twitter.com/nearcyan/status/1859426783663448349) injects humor into financial discussions by likening the experience of "ghosting" at FAANG companies with engineering's evolving professional landscapes.

---

# AI Reddit Recap

## /r/LocalLlama Recap

**Theme 1. M4 Max 128GB: Running 72B Models at 11 t/s with MLX**

- **[M4 Max 128GB running Qwen 72B Q4 MLX at 11tokens/second.](https://i.redd.it/wbdf0b5e772e1.jpeg)** ([Score: 476, Comments: 181](https://reddit.com/r/LocalLLaMA/comments/1gw9ufb/m4_max_128gb_running_qwen_72b_q4_mlx_at/)): The **Apple M4 Max** with **128GB** of memory successfully runs the **Qwen 72B Q4 MLX** model at a speed of **11 tokens per second**. This performance metric demonstrates the capability of Apple silicon to handle large language models efficiently.
  - Users discussed **power consumption** and **thermal performance**, noting the system draws up to **190W** during inference while running at high temperatures. The **M4 Max** achieves this performance while using significantly less power than comparable setups with multiple **NVIDIA 3090s** or **A6000s**.
  - The **M4 Max's memory bandwidth** of **546 GB/s** and performance of **11 tokens/second** represents a significant improvement over the **M1 Max** (at **409.6 GB/s** and **6.17 tokens/second**). Users successfully tested various models including **Qwen 72B**, **Mistral 128B**, and smaller coding models with **32k context** windows.
  - Discussion compared costs between building a desktop setup (**~$4000** for GPUs alone) versus the **$4700** M4 Max laptop, with many highlighting the portability advantage and complete solution aspect of the Apple system for running local LLMs, particularly during travel or in locations with power constraints.


- **Mac Users: New Mistral Large MLX Quants for Apple Silicon (MLX)** ([Score: 91, Comments: 23](https://reddit.com/r/LocalLLaMA/comments/1gw6yrg/mac_users_new_mistral_large_mlx_quants_for_apple/)): A developer created **2-bit** and **4-bit** quantized versions of **Mistral Large** optimized for **Apple Silicon** using **MLX-LM**, with the **q2** version achieving **7.4 tokens/second** on an **M4 Max** while using **42.3GB RAM**. The models are available on [HuggingFace](https://huggingface.co/zachlandes/Mistral-Large-Instruct-2411-Q2-MLX) and can run in **LMStudio** or other **MLX**-compatible systems, promising faster performance than **GGUF** models on **M-series** chips.
  - Users inquired about performance comparisons, with tests showing **MLX** models running approximately **20% faster** than **GGUF** versions on **Apple Silicon**, confirmed independently by multiple users.
  - Questions focused on practical usage, including how to run models through **LMStudio**, where users can manually download from **HuggingFace** and place files in the **LMStudio cache folder** for recognition.
  - Users discussed hardware compatibility, particularly regarding the **M4 Pro 64GB** and its ability to run **Mistral Large** variants, with interest in comparing performance against **Llama 3.1 70B Q4**.


**Theme 2. DeepSeek R1-Lite Preview Shows Strong Reasoning Capabilities**

- **[Here the R1-Lite-Preview from DeepSeek AI showed its power... WTF!! This is amazing!!](https://www.reddit.com/gallery/1gw61g5)** ([Score: 146, Comments: 19](https://reddit.com/r/LocalLLaMA/comments/1gw61g5/here_the_r1litepreview_from_deepseek_ai_showed/)): **DeepSeek's R1-Lite-Preview model** demonstrates advanced capabilities, though no specific examples or details were provided in the post body. The post title expresses enthusiasm about the model's performance but lacks substantive information about its actual capabilities or benchmarks.
  - **Base32 decoding** capabilities vary significantly among models, with **GPT-4** showing success while other models struggle. The discussion highlights that most **open models** perform poorly with ciphers, though they handle **base64** well due to its prevalence in training data.
  - **MLX** knowledge gaps were noted in **DeepSeek's R1-Lite-Preview**, suggesting limited parameter capacity for comprehensive domain knowledge. This limitation reflects the model's likely smaller size compared to other contemporary models.
  - Discussion of **tokenization** constraints explains model performance on encoding/decoding tasks, with current models using token-based rather than character-based processing. Users compare this limitation to humans trying to count invisible atoms - a system limitation rather than intelligence measure.


- **[deepseek R1 lite is impressive , so impressive it makes qwen 2.5 coder look dumb , here is why i am saying this, i tested R1 lite on recent codeforces contest problems (virtual participation) and it was very .... very good](https://i.redd.it/8tgij0jc882e1.png)** ([Score: 135, Comments: 44](https://reddit.com/r/LocalLLaMA/comments/1gwcsys/deepseek_r1_lite_is_impressive_so_impressive_it/)): **DeepSeek R1-Lite** demonstrates superior performance compared to **Qwen 2.5** specifically on **Codeforces** competitive programming contest problems through virtual participation testing. The post author emphasizes R1-Lite's exceptional performance but provides no specific metrics or detailed comparisons.
  - **DeepSeek R1-Lite** shows mixed performance across different tasks - successful with **scrambled letters** and **number encryption** but fails consistently with **Playfair Cipher**. Users note it excels at small-scope problems like **competitive programming** tasks but may struggle with real-world coding scenarios.
  - Comparative testing between **R1-Lite** and **Qwen 2.5** shows Qwen performing better at practical tasks, with users reporting success in **Unity C# scripting** and implementing a **raycast suspension system**. Both models can create **Tetris**, with Qwen completing it in one attempt versus R1's two attempts.
  - Users highlight that success in **competitive programming** doesn't necessarily translate to real-world coding ability. Testing on platforms like [atcoder.jp](atcoder.jp) and **Codeforces** with unique, recent problems is suggested for better model evaluation.


**Theme 3. Gemini-exp-1121 Tops LMSYS with Enhanced Coding & Vision**

- **[Google Releases New Model That Tops LMSYS](https://i.redd.it/zzdnaa997b2e1.jpeg)** ([Score: 139, Comments: 53](https://reddit.com/r/LocalLLaMA/comments/1gwoikh/google_releases_new_model_that_tops_lmsys/)): **Google** released **Gemini-exp-1121**, which achieved top performance on the **LMSYS** leaderboard for coding and vision tasks. The model represents an improvement over previous **Gemini** iterations, though no specific performance metrics were provided in the announcement.
  - **LMSYS** leaderboard rankings are heavily debated, with users arguing that **Claude** being ranked #7 indicates benchmark limitations. Multiple users report **Claude** outperforms competitors in real-world applications, particularly for coding and technical tasks.
  - **Gemini's** new vision capabilities enable direct **manga translation** by processing full image context, offering advantages over traditional **OCR + translation** pipelines. This approach better handles context-dependent elements like character gender and specialized terminology.
  - A competitive pattern emerged between **Google** and **OpenAI**, where each company repeatedly releases models to top the leaderboard. The release of **Gemini-exp-1121** appears to be a strategic move following **OpenAI's** recent model release.


**Theme 4. Allen AI's Tulu 3: Open Source Instruct Models on Llama 3.1**

- **[Tülu 3 -- a set of state-of-the-art instruct models with fully open data, eval code, and training algorithms](https://x.com/allen_ai/status/1859643404847808935)** ([Score: 117, Comments: 23](https://reddit.com/r/LocalLLaMA/comments/1gwl339/tülu_3_a_set_of_stateoftheart_instruct_models/)): **Allen AI** released **Tülu 3**, a collection of open-source instruction-following models, with complete access to training data, evaluation code, and training algorithms. The models aim to advance **state-of-the-art** performance while maintaining full transparency in their development process.
  - **Tülu 3** is a collection of **Llama 3.1 fine-tunes** rather than models built from scratch, with models available in [8B](https://huggingface.co/allenai/Llama-3.1-Tulu-3-8B) and [70B](https://huggingface.co/allenai/Llama-3.1-Tulu-3-70B) versions. Community members have already created **GGUF quantized versions** and **4-bit variants** for improved accessibility.
  - Performance benchmarks show the **8B model** surpassing **Qwen 2.5 7B Instruct** while the **70B model** outperforms **Qwen 2.5 72B Instruct**, **GPT-4o Mini**, and **Claude 3.5 Haiku**. The release includes comprehensive training data, reward models, and hyperparameters.
  - **Allen AI** has announced that their completely open-source **OLMo** model series will receive updates this month. A detailed discussion of **Tülu 3's** training process is available in a [newly released podcast](https://youtu.be/LVXtFnEbNU0).


**Theme 5. NVIDIA KVPress: Open Source KV Cache Compression Research**

- **New NVIDIA repo for KV compression research** ([Score: 48, Comments: 7](https://reddit.com/r/LocalLLaMA/comments/1gwgc5q/new_nvidia_repo_for_kv_compression_research/)): **NVIDIA** released an open-source library called **kvpress** to address **KV cache compression** challenges in large language models, where models like **llama 3.1-70B** require **330GB** of memory for **1M tokens** in **float16** precision. The library, built on **🤗 Transformers**, introduces a new **"expected attention"** method and provides tools for researchers to develop and benchmark compression techniques, with the code available at [kvpress](https://github.com/NVIDIA/kvpress).
  - **KV cache quantization** is not currently supported by **kvpress**, though according to the FAQ it could potentially be combined with pruning strategies to achieve up to **4x compression** when moving from **float16** to **int4**.
  - That's the only meaningful discussion point from the comments provided - the other comments and replies don't add substantive information to summarize.


## Other AI Subreddit Recap

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity

**Theme 1. Flux.1 Tools Suite Expands SD Capabilities**

- **[Huge FLUX news just dropped. This is just big. Inpainting and outpainting better than paid Adobe Photoshop with FLUX DEV. By FLUX team published Canny and Depth ControlNet a likes and Image Variation and Concept transfer like style transfer or 0-shot face transfer.](https://www.reddit.com/gallery/1gwilop)** ([Score: 739, Comments: 194](https://reddit.com/r/StableDiffusion/comments/1gwilop/huge_flux_news_just_dropped_this_is_just_big/)): **Black Forest Labs** released their **Flux.1 Tools** control suite featuring **inpainting** and **outpainting** capabilities that compete with **Adobe Photoshop**, alongside **ControlNet**-style features for **Canny** and **Depth** controls. The release includes **image variation** and **concept transfer** tools that enable **style transfer** and **zero-shot face transfer** functionality.
  - **ComfyUI** offers day-one support for **Flux Tools** with detailed implementation examples, requiring **27GB VRAM** for full models, though **LoRA versions** are available on [Huggingface](https://huggingface.co/black-forest-labs/FLUX.1-Depth-dev-lora/tree/main).
  - Community feedback indicates strong **outpainting** capabilities comparable to **Midjourney**, with users particularly praising the **Redux IP adapter's** performance and strength. The tools are publicly available for the **FLUX DEV model** with implementation details at [Black Forest Labs](https://blackforestlabs.ai/flux-1-tools/).
  - Users criticized the clickbait-style announcement title and requested more straightforward technical communication, while also noting that an **FP8 version** is available on [Civitai](https://civitai.com/models/969431/flux-fill-fp8) for those with lower VRAM requirements.


- **[Day 1 ComfyUI Support for FLUX Tools](https://blog.comfy.org/day-1-support-for-flux-tools-in-comfyui/)** ([Score: 149, Comments: 26](https://reddit.com/r/StableDiffusion/comments/1gwibxr/day_1_comfyui_support_for_flux_tools/)): **ComfyUI** added immediate support for **FLUX Tools** on release day, though specific details about the integration were not provided in the post.
  - **ComfyUI** users report successful integration with **Flux Tools**, with **SwarmUI** providing native support for all model variants as documented on [GitHub](https://github.com/mcmonkeyprojects/SwarmUI/blob/master/docs/Model%20Support.md#flux1-tools).
  - Users identified issues with **Redux** being overpowering and not working with **FP8_scaled** models, though adjusting **ConditioningSetTimestepRange** and **ConditioningSetAreaStrength** parameters showed improvement. A proper compositing workflow using **ImageCompositeMasked** or **Inpaint Crop/Stitch** nodes was recommended to prevent VAE degradation.
  - The implementation supports **Redux Adapter**, **Fill Model**, and **ControlNet Models & LoRAs** (specifically **Depth** and **Canny**), with a demonstration workflow available on [CivitAI](https://civitai.com/models/862215/proper-flux-control-net-inpainting-with-batch-size-comfyui-alimama).


- **[Flux Redux with no text prompt](https://i.redd.it/6hpnyn5fwa2e1.png)** ([Score: 43, Comments: 30](https://reddit.com/r/StableDiffusion/comments/1gwmzc5/flux_redux_with_no_text_prompt/)): **Redux adapter** testing focused on **image variation** capabilities without text prompts, though no specific details or results were provided in the post body.
  - **FLUX.1 Redux** adapter focuses on reproducing images with variations while maintaining style and scene without recreating faces. Users report faster and more precise results, particularly with **inpainting** capabilities for changing clothes and backgrounds.
  - The **ComfyUI** implementation requires placing the [sigclip vision model](https://huggingface.co/Comfy-Org/sigclip_vision_384/) in the models/clip_vision folder. Updates and workflows can be found on the [ComfyUI examples page](https://comfyanonymous.github.io/ComfyUI_examples/flux/).
  - **Flux Tools** integration with ComfyUI provides features like **ControlNet**, variations, and in/outpainting as detailed in the [Black Forest Labs documentation](https://blackforestlabs.ai/flux-1-tools/). The implementation guide is available on the [ComfyUI blog](https://blog.comfy.org/day-1-support-for-flux-tools-in-comfyui/).


**Theme 2. NVIDIA/MIT Release SANA: Efficient Sub-1B Parameter Diffusion Model**

- **Diffusion code for SANA has just released** ([Score: 103, Comments: 52](https://reddit.com/r/StableDiffusion/comments/1gwav1d/diffusion_code_for_sana_has_just_released/)): **SANA** diffusion model's training and inference code has been released on **GitHub** by **NVlabs**. The model weights are expected to be available on **HuggingFace** under *"Efficient-Large-Model/Sana_1600M_1024px"* but are not currently accessible.
  - **SANA** model's key feature is its ability to output **4096x4096** images directly, though some users note that models like **UltraPixel** and **Cascade** can also achieve this. The model comes in sizes of **0.6B** and **1.6B** parameters, significantly smaller than **SDXL (2B)** and **Flux Dev (12B)**.
  - The model is released by **NVIDIA**, **MIT**, and **Tsinghua University** researchers with a **CC BY-NC-SA 4.0 License**. Users note this is more restrictive than **PixArt-Sigma's OpenRail++ license**, but praise the rare instance of a major company releasing model weights.
  - Technical discussion focuses on the model's speed advantage and potential for fine-tuning, with the **0.6B version** being considered for specialized use cases. The model is available on [HuggingFace](https://huggingface.co/Efficient-Large-Model/Sana_1600M_1024px/tree/main/checkpoints) with a size of **6.4GB**.


- **[Testing the CogVideoX1.5-5B i2v model](https://www.reddit.com/gallery/1gwdn8o)** ([Score: 177, Comments: 51](https://reddit.com/r/StableDiffusion/comments/1gwdn8o/testing_the_cogvideox155b_i2v_model/)): **CogVideoX1.5-5B**, an **image-to-video model**, was discussed for community testing and evaluation. Insufficient context was provided in the post body for additional details about testing procedures or results.
  - The model's **workflow** is available on [Civitai](https://civitai.com/models/968568), with recommended **resolution** being above **720p**. The **v1.5** version is still in testing phase and currently only supports **1344x768** according to **Kijai documentation**.
  - Using a **4090 GPU**, generation takes approximately **3 minutes** for **1024x640** resolution and **5 minutes** for **1216x832**. A **3090** with **24GB VRAM** can run without the 'enable_sequential_cpu_offload' feature.
  - Technical limitations include poor performance of the **GGUF version** with occasional crashes, incompatibility with **anime-style images**, and potential **Out of Memory (OOM)** issues on **Windows** when attempting **81 frames** at **1024x640** resolution.


**Theme 3. ChatGPT 4o Nov Update: Better Writing, Lower Test Scores**

- **[gpt-4o-2024-11-20	scores lower on MMLU, GPQA, MATH and SimpleQA than gpt-4o-2024-08-06](https://i.redd.it/ocb1qkhgk92e1.png)** ([Score: 77, Comments: 17](https://reddit.com/r/OpenAI/comments/1gwhdn4/gpt4o20241120_scores_lower_on_mmlu_gpqa_math_and/)): **GPT-4o's** November 2024 update shows performance decreases across multiple benchmarks compared to its August version, including **MMLU**, **GPQA**, **MATH**, and **SimpleQA**. The lack of additional context prevents analysis of specific score differences or potential reasons for the decline.
  - **Performance drops** in the latest **GPT-4o** are significant, with **GPQA** declining by **13.37%** and **MMLU** by **3.38%** according to [lifearchitect.ai](https://lifearchitect.ai/models-table/). The model now scores lower than **Claude 3.5 Sonnet**, **Llama 3.1 405B**, **Grok 2**, and even **Grok 2 mini** on certain benchmarks.
  - Multiple users suggest **OpenAI** is optimizing for **creative writing** and user appeal rather than factual accuracy, potentially explaining the decline in benchmark performance. The tradeoff has resulted in *"mind blowing"* improvements in creative tasks while sacrificing objective correctness.
  - Users express desire for **specialized model naming** (like "*gpt-4o-creative-writing*" or "*gpt-4o-coding*") and suggest these changes are cost-optimization driven. Similar specialization trends are noted with **Anthropic's Sonnet** models showing task-specific improvements and declines.


- **[OpenAI's new update turns it into the greatest lyricist of all time (ChatGPT + Suno)](https://v.redd.it/qx79ooj6o72e1)** ([Score: 57, Comments: 27](https://reddit.com/r/OpenAI/comments/1gwb7cy/openais_new_update_turns_it_into_the_greatest/)): **OpenAI** released an update that, when combined with **Suno**, enhances its creative writing capabilities specifically for lyrics. No additional context or specific improvements were provided in the post body.
  - Users compare the AI's rap style to various artists including **Eminem**, **Notorious B.I.G.**, **Talib Kweli**, and **Blackalicious**, with some suggesting it surpasses **98%** of human rap. The original source was shared via [Twitter/X](https://x.com/kyleshannon/status/1859355131738734824).
  - The technical improvement appears focused on the **LLM's rhyming capabilities** while maintaining coherent narrative structure. Several users note the AI's ability to maintain consistent patterns while delivering meaningful content.
  - Multiple comments express concern about AI's rapid advancement, with one user noting humans are being outperformed not just in **math** and **chess** but now in creative pursuits like **rapping**. The sentiment suggests significant apprehension about AI capabilities.


**Theme 4. Claude Free Users Limited to Haiku as Demand Strains Capacity**

- **Free accounts are now (permanently?) routed to 3.5 Haiku** ([Score: 52, Comments: 40](https://reddit.com/r/ClaudeAI/comments/1gwe8fx/free_accounts_are_now_permanently_routed_to_35/)): **Claude** free accounts now default to the **Haiku model**, discovered by user **u/Xxyz260** through testing with a specific prompt about the **October 7, 2023** attack. The change appears to be unannounced, with users reporting intermittent access to **Sonnet 3.5** over an **18-hour** period before reverting to **Haiku**, suggesting possible load balancing tests by **Anthropic**.
  - Users confirmed through testing that free accounts are receiving **Haiku 3.5**, not Haiku 3, with evidence shown in [test results](https://imgur.com/a/SCpsPqp) demonstrating that model knowledge comes from system prompts rather than the models themselves.
  - A key concern emerged that **Pro users** are not receiving access to the newest **Haiku 3.5** model after exhausting their **Sonnet** limits, while free users are getting the updated version by default.
  - Discussion around **ChatGPT** becoming more attractive compared to **Claude**, particularly for coding tasks, with users expressing frustration about **Anthropic's** handling of the service changes and lack of transparency.


- **Are they gonna do something about Claude's overloaded server?** ([Score: 27, Comments: 46](https://reddit.com/r/ClaudeAI/comments/1gw7t7c/are_they_gonna_do_something_about_claudes/)): A user reports being unable to access **Claude Sonnet 3.5** due to server availability issues for **free accounts**, while finding **Claude Haiku** unreliable for following prompts. The user expresses that the **$20** **Claude Pro** subscription is cost-prohibitive for their situation as a college student, despite using Claude extensively for research, creative writing, and content creation.
  - **Free usage** of **Claude Sonnet** is heavily impacted by server load, particularly during **US working hours**, with users reporting up to **14 hours** of unavailability. European users note better access during their daytime hours.
  - Even **paid users** experience overloading issues, suggesting **Anthropic's** server capacity constraints are significant. The situation is unlikely to improve for free users unless the company expands capacity or **training costs** decrease.
  - There's confusion about the **Haiku model version** (3.0 vs 3.5), with users sharing [comparison screenshots](https://i.postimg.cc/jR8LNVVM/Screenshot-20241121-120819.jpg) and noting inconsistencies between mobile app and web UI displays, suggesting possible **A/B testing** or UI bugs.


---

# AI Discord Recap

> A summary of Summaries of Summaries by O1-mini

**Theme 1. New AI Models Surge Ahead with Enhanced Capabilities**

- [**Tülu 3 Launches with Superior Performance Over Llama 3.1**](https://x.com/natolambert/status/1859643351441535345): Nathan Lambert announced the release of **Tülu 3**, an open frontier model that outperforms **Llama 3.1** across multiple tasks by incorporating a novel **Reinforcement Learning with Verifiable Rewards** approach. This advancement ensures higher accuracy and reliability in real-world applications.
- [**Gemini Experimental 1121 Tops Chatbot Arena Benchmarks**](https://x.com/lmarena_ai/status/1859673146837827623): Google DeepMind’s **Gemini-Exp-1121** tied for the top spot in the Chatbot Arena, surpassing **GPT-4o-1120**. Its significant improvements in **coding** and **reasoning capabilities** highlight rapid progress in AI model performance.
- [**Qwen 2.5 Achieves GPT-4o-Level Performance in Code Editing**](https://aider.chat/2024/11/21/quantization.html): Open source models like **Qwen 2.5 32B** demonstrate competitive performance on Aider's code editing benchmarks, matching **GPT-4o**. Users emphasize the critical role of model **quantization**, noting significant performance variations based on quantization levels.

**Theme 2. Advanced Fine-Tuning Techniques Propel Model Efficiency**

- [**Unsloth AI Introduces Vision Support, Doubling Fine-Tuning Speed**](https://huggingface.co/unsloth/): **Unsloth** has launched **vision support** for models like **LLaMA**, **Pixtral**, and **Qwen**, enhancing developer capabilities by improving fine-tuning speed by **2x** and reducing memory usage by **70%**. This positions Unsloth ahead of benchmarks such as **Flash Attention 2 (FA2)** and **Hugging Face (HF)**.
- [**Contextual Position Encoding (CoPE) Enhances Model Expressiveness**](https://arxiv.org/abs/2405.18719): **Contextual Position Encoding (CoPE)** adapts position encoding based on token context rather than fixed counts, leading to more **expressive models**. This method improves handling of selective tasks like **Flip-Flop**, which traditional position encoding struggles with.
- [**AnchorAttention Reduces Training Time by Over 50% for Long-Context Models**](https://github.com/haonan3/AnchorContext): A new paper introduces **AnchorAttention**, a plug-and-play solution that enhances long-context performance while cutting training time by more than **50%**. Compatible with both [FlashAttention](https://github.com/haonan3/AnchorContext) and **FlexAttention**, it is suitable for applications like **video understanding**.

**Theme 3. Hardware Solutions and Performance Optimizations Drive AI Efficiency**

- [**Cloud-Based GPU Renting Enhances Model Speed for $25-50/month**](https://github.com/NousResearch/Hermes-3-Llama-3.1-70B): Transitioning to a cloud server for model hosting costs **$25-50/month** and significantly boosts model speed compared to local hardware. Users find cloud-hosted GPUs more cost-effective and performant, avoiding the limitations of on-premises setups.
- [**YOLO Excels in Real-Time Video Object Detection**](https://huggingface.co/spaces/prithivMLmods/YOLO-VIDEO): **YOLO** remains a preferred choice for **video object detection**, supported by the [YOLO-VIDEO](https://huggingface.co/spaces/prithivMLmods/YOLO-VIDEO) resource. Ongoing strategies aim to optimize YOLO's performance in real-time processing scenarios.
- [**MI300X GPUs Experience Critical Hang Issues During Long Runs**](https://github.com/ROCm/ROCm/issues/4021): Members reported **intermittent GPU hangs** on **MI300X** GPUs during extended **12-19 hour** runs with **axolotl**, primarily after the **6-hour mark**. These stability concerns are being tracked on [GitHub Issue #4021](https://github.com/ROCm/ROCm/issues/4021), including detailed metrics like **loss** and **learning rate**.

**Theme 4. APIs and Integrations Enable Custom Deployments and Enhancements**

- [**Hugging Face Endpoints Support Custom Handler Files**](https://huggingface.co/philschmid/distilbert-onnx-banking77/blob/main/handler.py): **Hugging Face Endpoints** now allow deploying custom AI models using a [handler.py](https://huggingface.co/philschmid/distilbert-onnx-banking77/blob/main/handler.py) file, facilitating tailored pre- and post-processing. Implementing the [EndpointHandler](https://huggingface.co/philschmid/distilbert-onnx-banking77/blob/main/handler.py) class ensures flexible and efficient model deployment tailored to specific needs.
- [**Model Context Protocol (MCP) Enhances Local Interactions**](https://x.com/btibor91/status/1859385266328531198): **Anthropic's Claude Desktop** now supports the [Model Context Protocol (MCP)](https://x.com/btibor91/status/1859385266328531198), enabling enhanced local interactions with models via **Python** and **TypeScript SDKs**. While remote connection capabilities are pending, initial support includes various SDKs, raising interest in expanded functionalities.
- [**OpenRouter API Documentation Clarified for Seamless Integration**](https://openrouter.ai/docs/provider-routing#quantization-levels): Users expressed confusion regarding **context window** functionalities in the **OpenRouter API** documentation. Enhancements are recommended to improve clarity, assisting seamless integration with tools like **LangChain** and optimizing **provider selection** for high-context prompts.

**Theme 5. Comprehensive Model Evaluations and Benchmark Comparisons Illuminate AI Progress**

- [**Perplexity Pro Outperforms ChatGPT in Accuracy for Specific Tasks**](https://aider.chat/2024/11/21/quantization.html): Users compared **Perplexity** with **ChatGPT**, noting that **Perplexity** is perceived as more accurate and offers advantages in specific functionalities. One participant highlighted that certain features of Perplexity were developed before their popularity surge, underscoring its robust capabilities.
- [**SageAttention Boosts Attention Mechanisms Efficiency by 8-Bit Quantization**](https://arxiv.org/abs/2410.02367): The [SageAttention](https://arxiv.org/abs/2410.02367) method introduces an efficient **8-bit quantization** approach for **attention mechanisms**, enhancing operations per second while maintaining **accuracy**. This improvement addresses the high computational complexity traditionally associated with larger sequences.
- [**DeepSeek-R1-Lite-Preview Showcases Superior Reasoning on Coding Benchmarks**](https://api-docs.deepseek.com/news/news1120): **DeepSeek** launched the **DeepSeek-R1-Lite-Preview**, demonstrating impressive **reasoning capabilities** on coding benchmarks. Users like [Zhihong Shao](https://x.com/zhs05232838/status/1859201857593524352) praised its performance in both coding and mathematical challenges, highlighting its practical applications.

---

# PART 1: High level Discord summaries

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Vision Support in Unsloth Launches**: Unsloth has officially launched **vision support**, enabling fine-tuning of models like **LLaMA**, **Pixtral**, and **Qwen**, which significantly enhances developer capabilities.
   - This feature improves fine-tuning speed by **2x** and reduces memory usage by **70%**, positioning Unsloth ahead of benchmarks such as **Flash Attention 2 (FA2)** and **Hugging Face (HF)**.
- **Enhancements in Fine-tuning Qwen and LLaMA**: Users are exploring the feasibility of fine-tuning both base and instructed **Qwen** and **LLaMA** models, with discussions centered around creating and merging **LoRAs**.
   - Unsloth's vision support facilitates merging by converting **4-bit LoRAs** back into **16-bit**, streamlining the fine-tuning process.
- **Llama 3.2 Vision Unveiled**: **Llama 3.2 Vision** models are now supported by Unsloth, achieving **2x faster** training speeds and **70% less memory usage** while enabling **4-8x longer context lengths**.
   - The release includes **Google Colab notebooks** for tasks like **Radiography** and **Maths OCR to LaTeX**, accessible via [Colab links](https://colab.research.google.com/drive/1j0N4XTY1zXXy7mPAhOC1_gMYZ2F2EBlk?usp=sharing).
- **AnchorAttention Improves Long-Context Training**: A new paper introduces **AnchorAttention**, a method that enhances long-context performance and reduces training time by over **50%**.
   - This solution is compatible with both [FlashAttention](https://github.com/haonan3/AnchorContext) and **FlexAttention**, making it suitable for applications like video understanding.
- **Strategies for Training Checkpoint Selection**: Discussions around selecting the appropriate training checkpoint revealed varied approaches, with some members opting for extensive checkpoint usage and others advocating for strategic selection based on specific metrics.
   - Participants emphasized the importance of performance benchmarks and shared experiences to optimize checkpoint choices in the training workflow.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Tülu 3 launch innovations**: Nathan Lambert announced the release of [Tülu 3](https://x.com/natolambert/status/1859643351441535345), an open frontier model that outperforms **Llama 3.1** on multiple tasks, incorporating a novel **Reinforcement Learning with Verifiable Rewards** approach.
   - The new model rewards algorithms exclusively for accurate generations, enhancing its performance and reliability in real-world applications.
- **Nvidia's AI Wall concerns**: An article from [The Economist](https://www.economist.com/business/2024/11/21/nvidias-boss-dismisses-fears-that-ai-has-hit-a-wall) reports Nvidia’s CEO downplaying fears that AI has 'hit a wall', despite widespread skepticism in the community.
   - This stance has intensified discussions about the **current trajectory of AI advancements** and the pressing need for continued innovation.
- **Gemini's Chatbot Arena performance**: Google DeepMind’s [Gemini-Exp-1121](https://x.com/lmarena_ai/status/1859673146837827623) tied for the top spot in the Chatbot Arena, surpassing **GPT-4o-1120** in recent benchmarks.
   - Gemini-Exp-1121 exhibits significant improvements in **coding and reasoning capabilities**, highlighting rapid progress in AI model performance.
- **Reinforcement Learning with Verifiable Rewards**: Tülu 3 incorporates a new technique called **Reinforcement Learning with Verifiable Rewards**, which trains models on constrained math problems and rewards correct outputs, as detailed in Nathan Lambert's [tweet](https://x.com/natolambert/status/1859643355698786549).
   - This approach aims to ensure higher accuracy in model generations by strictly incentivizing correct responses during training.
- **Model Context Protocol by Anthropic**: Anthropic's **Claude Desktop** now supports the [Model Context Protocol (MCP)](https://x.com/btibor91/status/1859385266328531198), enabling enhanced local interactions with models via Python and TypeScript SDKs.
   - While initial support includes various SDKs, remote connection capabilities are pending future updates, raising interest in expanded functionalities.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **SageAttention Enhances Attention Mechanisms**: The [SageAttention](https://arxiv.org/abs/2410.02367) method introduces an efficient quantization approach for **attention mechanisms**, boosting operations per second by optimizing computational resources.
   - This technique maintains **accuracy** while addressing the high complexity associated with larger sequences, making it a valuable improvement over traditional methods.
- **Deploying AI Models with Custom Handler Files**: **Hugging Face Endpoints** now support deploying custom AI models using a [handler.py](https://huggingface.co/philschmid/distilbert-onnx-banking77/blob/main/handler.py) file, allowing for tailored pre- and post-processing.
   - Implementing the [EndpointHandler](https://huggingface.co/philschmid/distilbert-onnx-banking77/blob/main/handler.py) class ensures flexible and efficient model deployment, catering to specific deployment needs.
- **Automated AI Research Assistant Developed**: A new Python program transforms local **LLMs** into automated web researchers, delivering detailed summaries and sources based on user queries.
   - The assistant systematically breaks down queries into subtopics, enhancing information gathering and analysis efficiency from various online sources.
- **YOLO Excels in Video Object Detection**: **YOLO** continues to be a preferred choice for video object detection, supported by a [YOLO-VIDEO](https://huggingface.co/spaces/prithivMLmods/YOLO-VIDEO) resource that facilitates effective implementation.
   - Discussions highlight ongoing strategies to optimize YOLO's performance in video streams, addressing challenges related to real-time processing.
- **MOUSE-I Streamlines Web Service Deployment**: **MOUSE-I** enables the conversion of simple prompts into globally deployed web services within **60 seconds** using [AI automation](https://huggingface.co/spaces/VIDraft/mouse1).
   - This tool is ideal for startups, developers, and educators seeking rapid deployment solutions without extensive manual configurations.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Perplexity Outperforms ChatGPT in Accuracy**: Users compared **Perplexity** with **ChatGPT**, highlighting that **Perplexity** is seen as more accurate and offers advantages in specific functionalities.
   - One participant noted that certain features of **Perplexity** were in development prior to their popularity surge.
- **GPT-4 Drives Product Categorization Efficiency**: A member shared their experience categorizing products using a prompt with **GPT-4**, specifying categories ranging from **groceries** to **clothing** with great results.
   - They noted that while the categorization is effective, **token usage** is high due to the lengthy prompt structure.
- **GPT-4o Enhances Image-Based Categorization**: A member described using **GPT-4o** to categorize products based on title and image, achieving **excellent results** with a comprehensive prompt structure.
   - However, they pointed out that the extensive **token usage** poses a scalability challenge for their setup.
- **Streamlining with Prompt Optimization**: Discussions centered around minimizing **token usage** while maintaining prompt effectiveness in categorization tasks.
   - Suggestions included exploring methods like **prompt caching** to streamline the process and reduce redundancy.
- **Prompt Caching Cuts Down Token Consumption**: Members suggested implementing **prompt caching techniques** to reduce repetitive input tokens in their categorization workflows.
   - They recommended consulting API-related resources for further assistance in optimizing **token usage**.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Hermes 3 exceeds expectations**: A member favored [Hermes 3](https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-70B) for its superior writing skills and prompt adherence, though noted phrase repetition at higher contexts (**16K+**).
   - This preference underscores **Hermes 3**’s advancements, but highlights limitations in handling longer contexts efficiently.
- **Cloud-Based GPU Renting**: Transitioning to a cloud server for model hosting costs **$25-50/month** and enhances model speed compared to local hardware.
   - Users find cloud-hosted GPUs more cost-effective and performant, avoiding the limitations of on-premises setups.
- **LLM GPU Comparisons**: Members compared **AMD** and **NVIDIA** GPUs, with recent driver updates impacting AMD's ROCM support.
   - Consensus leans towards **NVIDIA** for better software compatibility and support in AI applications.
- **Mixed GPU setups hinder performance**: Configurations with **1x 4090 + 2x A6000** GPUs underperform others due to shared resource constraints, reducing token generation rates.
   - Users highlighted that the slowest GPU in a setup, such as the **4090**, can limit overall processing speed.
- **Feasibility of a $2k local LLM server**: Setting up a local LLM server for **2-10 users** on a **$2,000** budget poses challenges with single GPU concurrency.
   - Developers recommend cloud solutions to mitigate performance bottlenecks associated with budget and older hardware.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Qwen 2.5 Rivals GPT-4o in Performance**: Open source models like **Qwen 2.5 32B** demonstrate competitive performance on Aider's [code editing benchmarks](https://aider.chat/2024/11/21/quantization.html), matching **GPT-4o**, while the least effective versions align with **GPT-3.5 Turbo**.
   - Users emphasized the significant impact of model **quantization**, noting that varying quantization levels can lead to notable performance differences.
- **Aider v0.64.0 Introduces New Features**: The latest **Aider v0.64.0** release includes the new [`/editor`](https://aider.chat/docs/usage/commands.html) command for prompt writing and full support for **gpt-4o-2024-11-20**.
   - This update enhances shell command clarity, allowing users to see confirmations and opt into [analytics](https://aider.chat/docs/more/analytics.html) seamlessly.
- **Gemini Model Enhances AI Capabilities**: The [Gemini Experimental Model](https://ai.google.dev/gemini-api/docs/models/experimental-models) offers improved **coding**, **reasoning**, and **vision** capabilities as of **November 21, 2024**.
   - Users are leveraging Gemini's advanced functionalities to achieve more sophisticated AI interactions and improved coding efficiency.
- **Model Quantization Impact Discussed**: **Model quantization** significantly affects AI performance, particularly in code editing, as highlighted in Aider's [quantization analysis](https://aider.chat/2024/11/21/quantization.html).
   - The community discussed optimizing quantization levels to balance performance and resource utilization effectively.
- **DeepSeek-R1-Lite-Preview Boosts Reasoning**: **DeepSeek** launched the **DeepSeek-R1-Lite-Preview**, showcasing impressive reasoning capabilities on coding benchmarks, as detailed in their [latest release](https://api-docs.deepseek.com/news/news1120).
   - Users like [Zhihong Shao](https://x.com/zhs05232838/status/1859201857593524352) praised its performance in both coding and mathematical challenges, highlighting its practical applications.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter Launches Five New Models**: OpenRouter has introduced **GPT-4o** with improved prose capabilities, along with **Mistral Large** ([link](https://openrouter.ai/mistralai/mistral-large-2411)), **Pixtral Large** ([link](https://openrouter.ai/mistralai/pixtral-large-2411)), **Grok Vision Beta** ([link](https://openrouter.ai/x-ai/grok-vision-beta)), and **Gemini Experimental 1114** ([link](https://openrouter.ai/google/gemini-exp-1114)).
   - These models enhance functionalities across various benchmarks, offering advanced features for AI engineers to explore.
- **Mistral Medium Deprecated, Alternatives Recommended**: The **Mistral Medium** model has been deprecated, resulting in access errors due to **priority not enabled**.
   - Users are advised to switch to **Mistral-Large**, **Mistral-Small**, or **Mistral-Tiny** to continue utilizing the service without interruptions.
- **Gemini Experimental 1121 Released with Upgrades**: **Gemini Experimental 1121** model has been launched, featuring enhancements in coding, reasoning, and vision capabilities.
   - Despite quota restrictions shared with the **LearnLM** model, the community is eager to assess its performance and potential applications.
- **OpenRouter API Documentation Clarified**: Users have expressed confusion regarding **context window** functionalities in the **OpenRouter API** documentation.
   - Enhancing documentation clarity is recommended to assist seamless integration with tools like LangChain.
- **Request for Custom Provider Key for Claude 3.5**: A member has requested a **custom provider key** for **Claude 3.5 Sonnet** due to exhausting usage on the main **Claude app**.
   - This request aims to provide an alternative solution to manage usage limits and improve user experience.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Flux's VRAM Vexations**: Members discussed the **resource requirements** for using **Flux** effectively, noting that it requires substantial **VRAM** and can be slow to generate images. [Black Forest Labs](https://x.com/bfl_ml/status/1859616264324284619?t=DftoDEhtAigmD4sQvsMl2w&s=19) released **FLUX.1 Tools**, enhancing control and steerability of their base model.
   - One member highlighted that using **Loras** can enhance **Flux**'s output for NSFW content, although **Flux** isn't optimally trained for that purpose.
- **Optimizing SDXL Performance**: For **SDXL**, applying best practices such as `--xformers` and `--no-half-vae` can improve performance on systems with **12GB VRAM**. Members noted that **Pony**, a derivative of **SDXL**, requires special tokens and has compatibility issues with **XL Loras**.
   - These configurations aid in enhancing **SDXL** efficiency, while **Pony**'s limitations underscore challenges in model compatibility.
- **Enhancing Image Prompts with SDXL Lightning**: A user inquired about **using image prompts** in **SDXL Lightning** via Python, specifically for inserting a photo into a specific environment. This showcases the community's interest in combining image prompts with varied backgrounds to boost generation capabilities.
   - The discussion indicates a trend towards leveraging Python integrations to augment **SDXL Lightning**'s flexibility in image generation tasks.
- **Mitigating Long Generation Times**: Frustrations over random **long generation times** while using various models led to discussions about potential underlying causes. Members speculated that memory management issues, such as loading resources into **VRAM**, might contribute to these slowdowns.
   - Addressing these delays is pivotal for improving user experience, with suggestions pointing towards optimizing **VRAM** usage to enhance generation speeds.
- **Securing AI Model Utilization**: Concerns were raised about receiving suspicious requests for personal information, like wallet addresses, leading members to suspect **scammers** within the community. Users were encouraged to report such incidents to maintain a **secure** environment.
   - Emphasizing security, the community seeks to protect its members by proactively addressing potential threats related to **AI model** misuse.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Contextual Position Encoding (CoPE) Boosts Model Adaptability**: A proposal for **[Contextual Position Encoding (CoPE)](https://arxiv.org/abs/2405.18719)** suggests adapting position encoding based on token context rather than fixed counts, leading to more **expressive models**. This approach targets improved handling of selective tasks like **Flip-Flop** that traditional methods struggle with.
   - Members discussed the potential of CoPE to enhance the adaptability of position encoding, potentially resulting in better performance on complex NLP tasks requiring nuanced token relationship understanding.
- **Forgetting Transformer Surpasses Traditional Architectures in Long-Context Tasks**: **Forgetting Transformer**, a variant incorporating a forget gate, demonstrates improved performance on **long-context tasks** compared to standard architectures. Notably, this model eliminates the need for position embeddings while maintaining effectiveness over extended training contexts.
   - The introduction of the Forgetting Transformer indicates a promising direction for enhancing **LLM performance** by managing long-term dependencies more efficiently.
- **Sparse Upcycling Enhances Model Quality with Inference Trade-off**: A recent **[Databricks paper](https://arxiv.org/abs/2411.08968)** evaluates the trade-offs between **sparse upcycling** and continued pretraining for model enhancement, finding that sparse upcycling results in higher **model quality**. However, this improvement comes with a **40% increase** in inference time, highlighting deployment challenges.
   - The findings underscore the difficulty in balancing **model performance** with practical deployment constraints, emphasizing the need for strategic optimization methods in model development.
- **Scaling Laws Predict Model Performance at Minimal Training Costs**: A recent **[paper](https://arxiv.org/abs/2405.10938)** introduces an observational approach using ~100 publicly available models to develop **scaling laws** without direct training, enabling prediction of language model performance based on **scale**. This method highlights variations in training **efficiency**, proposing that performance is dependent on a low-dimensional capability space.
   - The study demonstrates that **scaling law models** are costly but significantly less so than training full target models, with Meta reportedly spending only **0.1% to 1%** of the target model's budget for such predictions.
- **lm-eval Enhances Benchmarking for Pruned Models**: A user inquired if the current version of **lm-eval** supports zero-shot benchmarking of pruned models, specifically using **WANDA**, and encountered issues with outdated library versions. Discussions suggested reviewing the documentation for existing limitations.
   - To resolve integration issues with the **Groq API**, setting the API key in the `OPENAI_API_KEY` environment variable was recommended, successfully addressing unrecognized API key argument problems.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Pro Unveils Advanced Features**: Members discussed various features of **Perplexity Pro**, emphasizing the inclusion of more advanced **models** available for Pro users, differentiating it from **ChatGPT**.
   - The discussion included insights on **search** and **tool integration** that enhances user experience.
- **Pokémon Data Fuels New AI Model**: A [YouTube video](https://www.youtube.com/embed/hQhP7ipvgx0) explored how **Pokémon data** is being utilized to develop a novel **AI model**, providing insights into technological advancements in gaming.
   - *This may change the way data is leveraged in AI applications*.
- **NVIDIA's Omniverse Blueprint Transforms CAD/CAE**: A member shared insights on **NVIDIA's Omniverse Blueprint**, showcasing its transformative potential for **CAD** and **CAE** in design and simulation.
   - *Many are excited about how it integrates advanced technologies into traditional workflows*.
- **Bring Your Own API Key Adoption Discussed**: A member inquired about the permissibility of **bringing your own API key** to build an alternative platform with **Perplexity**, outlining secure data management practices.
   - This approach involves user-supplied keys being **encrypted** and **stored in cookies**, raising questions about compliance with **OpenAI standards**.
- **Enhancing Session Management in Frontend Apps**: In response to a request for simplification, a user explained **session management** in web applications by comparing it to **cookies storing session IDs**.
   - The discussion emphasized how users' authentication relies on validating sessions without storing sensitive data directly.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Truffles Device Gains Attention**: The **Truffles** device, described as a 'white cloudy semi-translucent thing,' enables self-hosting of LLMs. Learn more at [Truffles](https://x.com/itsalltruffles).
   - A member humorously dubbed it the 'glowing breast implant,' highlighting its unique appearance.
- **Vercel Acquires Grep for Enhanced Code Search**: Vercel announced the acquisition of [Grep](https://grep.app/) to bolster developer tools for searching code across over **500,000** public repositories.
   - Dan Fox, Grep's founder, will join Vercel's AI team to advance this capability.
- **Tülu 3 Surpasses Llama 3 on Tasks**: [Tülu 3](https://allenai.org/papers/tulu-3-report.pdf), developed over two years, outperforms **Llama 3.1 Instruct** on specific tasks with new SFT data and optimization techniques.
   - The project lead expressed excitement about their advancements in **RLHF**.
- **Black Forest Labs Launches Flux Tools**: Black Forest Labs unveiled **Flux Tools**, featuring inpainting and outpainting for image manipulation. Users can run it on [Replicate](https://replicate.com/black-forest-labs).
   - The suite aims to add steerability to their text-to-image model.
- **Google Releases Gemini API Experimental Models**: New experimental models from **Gemini** were released, enhancing coding capabilities.
   - Details are available in the [Gemini API documentation](https://ai.google.dev/gemini-api/docs/models/experimental-models).



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **DeepSeek R1-Lite Boosts MATH Performance**: Rumors suggest that **DeepSeek R1-Lite** is a **16B MOE** model with **2.4B active parameters**, significantly enhancing MATH scores from **17.1** to **91.6** according to a [tweet](https://x.com/nrehiew_/status/1859265550767067518).
   - The skepticism emerged as members questioned the [WeChat announcement](https://x.com/nrehiew_/status/1859265550767067518), doubting the feasibility of such a dramatic performance leap.
- **Llama-Mesh Paper Gains Attention**: A member recommended reviewing the **llama-mesh paper**, praising its insights as 'pretty good' within the group.
   - This suggestion came amid a broader dialogue on advancing AI architectures and collaborative research.
- **Multi-Agent Frameworks Face Output Diversity Limits**: Concerns were raised that employing de-tokenized output in multi-agent frameworks like 'AI entrepreneurs' might result in the **loss of hidden information** due to discarded KV caches.
   - This potential information loss could be contributing to the **limited output diversity** observed in such systems.
- **Soft Prompts Lag Behind Fine Tuning**: **Soft prompts** are often overshadowed by techniques like **fine tuning** and **LoRA**, which are perceived as more effective for open-source applications.
   - Participants highlighted that soft prompts suffer from limited generalizability and involve trade-offs in performance and optimization.
- **CoPilot Arena Releases Initial Standings**: The inaugural results of **CoPilot Arena** were unveiled on [LMarena's blog](https://blog.lmarena.ai/blog/2024/copilot-arena/#initial-leaderboard-and-results), showing a tightly contested field among participants.
   - However, the analysis exclusively considered an older version of **Sonnet**, sparking discussions on the impact of using outdated models in competitions.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Triton Kernel Debugging and GEMM Optimizations**: Users tackled **Triton interpreter** accuracy issues and discussed performance enhancements through **block size adjustments** and **swizzling techniques**, referencing tools like [triton.language.swizzle2d](https://triton-lang.org/main/python-api/generated/triton.language.swizzle2d.html).
   - There was surprise at **Triton GEMM's conflict-free** performance on ROCm, sparking conversations about optimizing **GEMM operations** for improved computational efficiency.
- **cuBLAS Matrix Multiplication in Row-Major Order**: Challenges with `cublasSgemm` were highlighted, especially regarding **row-major** vs **column-major** order operations, as detailed in a [relevant Stack Overflow post](https://stackoverflow.com/questions/56043539/cublassgemm-row-major-multiplication).
   - Users debated the implications of using `CUBLAS_OP_N` versus `CUBLAS_OP_T` for non-square matrix multiplications, noting **compatibility issues** with existing codebases.
- **ROCm Compilation and FP16 GEMM on MI250 GPU**: Developers reported prolonged **compilation times** when using ROCm's `make` commands, with efforts to tweak the `-j` flag showing minimal improvements.
   - Confusion arose over **input shape transformations** for **FP16 GEMM (v3)** on the MI250 GPU, leading to requests for clarification on **shared memory** and input shapes.
- **Advancements in Post-Training AI Techniques**: A new survey, [Tulu 3](https://allenai.org/papers/tulu-3-report.pdf), was released, covering **post-training methods** such as **Human Preferences in RL** and **Continual Learning**.
   - Research on **Constitutional AI** and **Recursive Summarization** frameworks was discussed, emphasizing models that utilize **human feedback** for enhanced **task performance**.



---



## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM's GitHub Traversal Limitations**: Users reported that **NotebookLM** struggles to traverse GitHub repositories by inputting the repo homepage, as it lacks website traversal capabilities. One member suggested converting the site into [Markdown](https://discord.com/channels/1124402182171672732/1124403655819415592/1309003599489007686) or PDF for improved processing.
   - The inability to process websites directly complicates using NotebookLM for repository analysis, leading to workarounds like manual content conversion.
- **Enhancements in Audio Prompt Generation**: A user proposed enhancing **NotebookLM** by supplying specific prompts to generate impactful audio outputs, improving explanations and topic comprehension.
   - This strategy aims to facilitate a deeper understanding of designated topics through clearer audio content, as discussed by the community.
- **Integrating Multiple LLMs for Specialized Tasks**: Community members shared workflows utilizing multiple Large Language Models (LLMs) tailored to specific needs, with compliments towards **NotebookLM** for its generation capabilities.
   - This approach underscores the effectiveness of combining various AI tools to support conversational-based projects, as detailed in user blog posts.
- **ElevenLabs' Dominance in Text-to-Speech AI**: Discussions highlighted **ElevenLabs** as the leading text-to-speech AI, outperforming competitors like RLS and Tortoise. A user reminisced about early experiences with the startup before its funding rounds.
   - The impact of **ElevenLabs** on voice synthesis and faceless video creation was emphasized as a transformative tool in the industry.
- **NotebookLM's Stability and Safety Flags Issues**: Users noted an increase in **safety flags** and instability within **NotebookLM**, resulting in restricted functionalities and task limitations.
   - Community members suggested direct message (DM) examples for debugging, attributing transient issues to ongoing application improvements.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **AI Agent Architecture with LlamaIndex and Redis**: Join the upcoming webinar on [December 12](https://twitter.com/llama_index/status/1859354663793066029) to learn how to architect **agentic systems** for breaking down complex tasks using [LlamaIndex](https://twitter.com/llama_index/status/1859354663793066029) and **Redis**.
   - Participants will discover best practices for reducing **costs**, optimizing **latency**, and gaining insights on **semantic caching** mechanisms.
- **Knowledge Graph Construction using Memgraph and LlamaIndex**: Learn how to set up **Memgraph** and integrate it with [LlamaIndex](https://twitter.com/llama_index/status/1859658719082041802) to build a **knowledge graph** from unstructured text data.
   - The session will explore **natural language querying** of the constructed graph and methods to **visualize connections** effectively.
- **PDF Table Extraction with LlamaParse**: A member recommended using [LlamaParse](https://github.com/run-llama/llama_parse) for extracting table data from PDF files, highlighting its effectiveness for optimal RAG.
   - An informative [GitHub link](https://github.com/run-llama/llama_parse) was shared detailing its parsing capabilities.
- **Create-Llama Frontend Configuration**: A user inquired about the best channel for assistance with Create-Llama, specifically regarding the absence of a Next.js frontend option in newer versions when selecting the Express framework.
   - Another participant confirmed that queries can be posted directly in the channel to receive team support.
- **Deprecation of Llama-Agents in Favor of Llama-Deploy**: A member noted dependency issues while upgrading to Llama-index 0.11.20 and indicated that **llama-agents** has been deprecated in favor of [llama_deploy](https://github.com/run-llama/llama_deploy).
   - They provided a link to the [Llama Deploy GitHub page](https://github.com/run-llama/llama_deploy) for further context.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **30 Days of Python Challenge**: A member shared their participation in the **30 Days of Python** challenge, which emphasizes step-by-step learning, utilizing the [GitHub repository](https://github.com/Asabeneh/30-Days-Of-Python) for resources and inspiration.
   - They are actively engaging with the repository's content to enhance their Python skills during this structured 30-day program.
- **Capstone Project API**: One member expressed a preference for using **Go** in their capstone project to develop an API, highlighting the exploration of different programming languages in practical applications.
   - Their choice reflects the community's interest in leveraging Go's concurrency features for building robust APIs.
- **Cohere GitHub Repository**: A member highlighted the **Cohere GitHub repository** ([GitHub link](https://github.com/cohere-ai)) as an excellent starting point for contributors, showcasing various projects.
   - They encouraged exploring available tools within the repository and sharing feedback or new ideas across different projects.
- **Cohere Toolkit for RAG Applications**: The **Cohere Toolkit** ([GitHub link](https://github.com/cohere-ai/cohere-toolkit)) was mentioned as an advanced UI designed specifically for **RAG applications**, facilitating quick build and deployment.
   - This toolkit includes a collection of prebuilt components aimed at enhancing user productivity.
- **Multimodal Embeddings Launch**: Exciting updates were shared about the improvement in **multimodal embeddings**, with their launch scheduled for early next year on **Bedrock** and partner platforms.
   - *A team member will flag the rate limit issue* for further discussion to address scalability concerns.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo's Async Features Under Development**: Members reported that **Mojo's async functionalities** are currently under development, with no available async functions yet.
   - The compiler presently translates synchronous code into asynchronous code, resulting in synchronous execution during async calls.
- **Mojo Community Channel Launch**: A dedicated **Mojo community channel** has been launched to facilitate member interactions, accessible at [mojo-community](https://prefix.dev/channels/mojo-community).
   - The channel serves as a central hub for ongoing discussions related to Mojo development and usage.
- **Moonshine ASR Model Performance on Mojo**: The **Moonshine ASR model** was benchmarked using [moonshine.mojo](https://gist.github.com/keveman/ea167957fb6364470cb265c5d9aa9da1) and [moonshine.py](https://gist.github.com/keveman/d2aea1a059c9a14972783ede2d6b6862), resulting in an execution time of **82ms** for **10s** of speech, compared to **46ms** on the ONNX runtime.
   - This demonstrates a **1.8x** slowdown in Mojo and Python versions versus the optimized ONNX runtime.
- **Mojo Script Optimization Challenges**: Developers faced crashes with **Model.execute** when passing `TensorMap` in the **Mojo scripts**, necessitating manual argument listing due to unsupported unpacking.
   - These issues highlight the need for script optimization and improved conventions in Mojo code development.
- **CPU Utilization in Mojo Models**: Users observed inconsistent **CPU utilization** while running models in Mojo, with full CPU capacity and hyperthreading being ignored.
   - This suggests a need for further optimization to maximize resource utilization during model execution.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Updated Contributor Guidelines for Torchtune**: The team announced [new guidelines](https://discord.com/channels/1216353675241590815/1236040539409879170/1309124519134494720) to assist **Torchtune** maintainers and contributors in understanding desired features.
   - These guidelines clarify when to use forks versus example repositories for demonstrations, streamlining the contribution process.
- **Extender Packages Proposed for Torchtune**: A member suggested introducing extender packages like **torchtune[simpo]** and **torchtune[rlhf]** to simplify package inclusion.
   - This proposal aims to reduce complexity and effectively manage resource concerns without excessive checks.
- **Binary Search Strategy for max_global_bsz**: Recommendation to implement a last-success binary search method for **max_global_bsz**, defaulting to a power of 2 smaller than the dataset.
   - The strategy will incorporate **max_iterations** as parameters to enhance efficiency.
- **Feedback on UV Usability**: A member inquired about others' experiences with **UV**, seeking opinions on its usability.
   - Another member partially validated its utility, noting it appears **appealing** and modern.
- **Optional Packages Addressing TorchAO**: Discussion on whether the optional packages feature can resolve the need for users to manually download **TorchAO**.
   - Responses indicate that while it may offer some solutions, additional considerations need to be addressed.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Prompt Signature Modification**: A member inquired about modifying the **prompt signature format** for debugging purposes to avoid parseable **JSON schema notes**, particularly by building an **adapter**.
   - The discussion explored methods like building custom adapters to achieve **prompt signature customization** in DSPy.
- **Adapter Configuration in DSPy**: A user suggested building an **adapter** and configuring it using `dspy.configure(adapter=YourAdapter())` for prompt modifications, pointing towards existing adapters in the `dspy/adapters/` directory for further clarification.
   - Leveraging existing adapters within DSPy can aid in effective **prompt signature customization**.
- **Optimization of Phrases for Specific Cases**: Questions about tuning phrases for specific types like **bool**, **int**, and **JSON** were clarified to be based on a maintained set of **model signatures**.
   - *These phrases are not highly dependent on individual language models overall*, indicating a generalized formulation approach.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Intel AMA Session Scheduled for November 21**: Join the **Hackathon AMA with Intel** on **November 21 at 3 PM PT** to engage directly with **Intel specialists**.
   - Don't forget to [watch live here](https://www.youtube.com/watch?v=_Wm5guUXt54) and set your reminders for the *Ask Intel Anything* opportunity.
- **Quiz 10 Release Status Update**: A member inquired about the release status of **Quiz 10**, which has not been released on the website yet.
   - Another member confirmed that email notifications will be sent once **Quiz 10** becomes available, likely within **a day or two**.
- **Hackathon Channel Mix-up Incident**: A member expressed gratitude for the **Quiz 10** update but humorously acknowledged asking about the **hackathon** in the wrong channel.
   - This exchange reflects common channel mix-ups within the community, adding a lighthearted moment to the conversation.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Need for int64 Indexing Explored**: A user questioned the necessity of **int64 indexing** in contexts that do not involve large tensors, prompting others to share their thoughts.
   - Another user linked to the [ops_hip.py](https://github.com/tinygrad/tinygrad/blob/master/extra/backends/ops_hip.py) file for further context regarding this discussion.
- **Differences in ops_hip.py Files Dissected**: A member pointed out distinctions between two **ops_hip.py** files in the tinygrad repository, suggesting the former may not be maintained due to incorrect imports.
   - They noted that the latter is only referenced in the context of an external benchmarking script, which also contains erroneous imports.
- **Maintenance Status of ops_hip.py Files**: In response to the maintenance query, another user confirmed that the **extra** ops_hip.py is not maintained while the **tinygrad** version should function if **HIP=1** is set.
   - This indicates that while some parts of the code may not be actively managed, others can still be configured to work correctly.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Event link confusion arises**: A member raised concerns about not finding the event link on **Luma**, seeking clarification on its status.
   - **Chiphuyen** apologized, explaining that the event was not rescheduled due to illness.
- **Wishing well to a sick member**: Another member thanked **Chiphuyen** for the update and wished them a speedy recovery.
   - This reflects the community's supportive spirit amidst challenges in event management.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Urgent AI Expertise Requested**: User **michel.0816** urgently requested an **AI expert**, indicating a pressing need for assistance.
   - Another member suggested posting the issue in designated channels for better visibility.
- **Carter Grant's Job Search**: Carter Grant, a **full-stack developer** with 6 years of experience in **React**, **Node.js**, and **AI/ML**, announced his search for job opportunities.
   - He expressed eagerness to contribute to meaningful projects.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **MI300X GPUs Stall After Six Hours**: A member reported experiencing **intermittent GPU hangs** during longer **8 x runs** lasting **12-19 hours** on a standard ablation set with **axolotl**, primarily occurring after the **6-hour mark**.
   - These stability concerns have been documented and are being tracked on [GitHub Issue #4021](https://github.com/ROCm/ROCm/issues/4021), which includes detailed metrics such as **loss** and **learning rate** to provide technical context.
- **Prompting Correctly? Engineers Debate Necessity**: In the **community-showcase** channel, a member questioned the necessity of proper prompting, sharing a [YouTube video](https://youtu.be/m3Izr0wNfQc) to support the discussion.
   - This query has initiated a dialogue among AI engineers regarding the current relevance and effectiveness of prompt engineering techniques.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Training the Autoencoder**: A member emphasized the importance of **training the autoencoder** for achieving model efficiency, focusing on techniques and implementation strategies to enhance performance.
   - The conversation delved into methods for improving autoencoder performance, including various training techniques.
- **Sophistication of Autoencoder Architectures**: Members discussed the **sophistication of autoencoder architectures** in current models, exploring how advanced structures contribute to model capabilities.
   - The effectiveness of different algorithms and their impact on data representation within autoencoders were key points of discussion.



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Refact.AI Live Demo**: The [Refact.AI](https://github.com/smallcloudai) team is conducting a live demo showcasing their **autonomous agent** and innovative tooling.
   - Join the live event [here](https://discord.com/events/1089876418936180786/1300459081181429810) to engage in the conversation.
- **Mozilla Launches Web Applets**: Mozilla has initiated the open-source project **Web Applets** to develop AI-native applications for the web.
   - This project promotes **open standards** and accessibility, fostering collaboration among developers, as detailed [here](https://discord.com/channels/1089876418936180786/1231977676458168381).
- **Mozilla's Public AI Advocacy**: Mozilla has advanced **14 local AI projects** in the past year to advocate for **Public AI** and build necessary developer tools.
   - This initiative aims to foster open-source AI technology with a collaborative emphasis on community engagement.



---



## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **Query on Llama 3.2 Prompt Format**: A member inquired about the lack of usage of a specific prompt for **Llama 3.2**, referencing the [prompt format documentation](https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_2/#-function-definitions-in-the-system-prompt-).
   - The question highlighted a need for clarity on **function definitions** in the system prompt, emphasizing their importance for effective use.
- **Interest in Prompt Applicability**: The conversation showed a broader interest in understanding the **applicability of prompts** within **Llama 3.2**.
   - This reflects ongoing discussions about best practices for maximizing model performance through **effective prompting**.



---


The **Alignment Lab AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **LLM Finetuning (Hamel + Dan) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1308928576355172424)** (497 messages🔥🔥🔥): 

> - `Vision support in Unsloth`
> - `Fine-tuning Qwen and LLaMA models`
> - `Dataset preparation for multimodal models`
> - `Licensing and legal considerations`
> - `Challenges with model merging and format compatibility` 


- **Vision Support in Unsloth is Live**: Unsloth has officially launched support for vision models, allowing for fine-tuning of LLaMA, Pixtral, and Qwen, greatly enhancing capabilities for developers.
   - This new feature reportedly makes fine-tuning **2x faster** and cuts memory usage by **70%**.
- **Fine-tuning Challenges with Qwen and LLaMA**: Users are discussing the feasibility of fine-tuning both base and instructed models, with some expressing confusion over creating and merging LoRAs.
   - The vision support in Unsloth is designed to facilitate merging, transforming 4-bit LoRAs back into **16-bit** seamlessly.
- **Dataset Preparation for Vision Models**: Tips shared on creating datasets for vision models, with examples such as the 'unsloth/Radiology_mini' format, which includes images, IDs, captions, and classifications.
   - The community is encouraged to use the structured format, making data preparation more streamlined for model training.
- **Licensing and Legal Considerations**: Participants discussed the implications of licensing when it comes to fine-tuning Mistral models, with some reporting challenges in contacting the team for permission.
   - Concerns were raised about the impact of neglecting licensing terms on the future of open-source efforts.
- **Merging and Format Compatibility Issues**: Users have encountered issues with the compatibility of 4-bit and 16-bit models, often requiring upcasting for successful merging.
   - The emphasis is placed on the importance of understanding these formats for effective model training and implementation.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://bsky.app/profile/yasomi.xeiaso.net/post/3lbbfnb7uic2k">Mimi (@yasomi.xeiaso.net)</a>: Vaporwave pastel ukiyo-e goes hard</li><li><a href="https://huggingface.co/datasets/unsloth/LaTeX_OCR">unsloth/LaTeX_OCR · Datasets at Hugging Face</a>: no description found</li><li><a href="https://unslothai.substack.com/">Unsloth AI | Substack</a>: Welcome to Unsloth&#x27;s newsletter where we&#x27;ll be sharing tips &amp; tricks on AI, our latest releases and more!  We recently launched unsloth.ai 🦥. Click to read Unsloth AI, a Substack public...</li><li><a href="https://docs.unsloth.ai/get-started/unsloth-notebooks">Unsloth Notebooks | Unsloth Documentation</a>: See the list below for all our notebooks:</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1gwoqm9/llama_32_vision_">Reddit - Dive into anything</a>: no description found</li><li><a href="https://datta0.substack.com/p/ai-unplugged-23-ngpt-normalised-transformer">AI Unplugged 23: nGPT normalised transformer, LAUREL, TokenFormer</a>: Insights over information</li><li><a href="https://github.com/unslothai/unsloth/pull/1082">Introduce MsT technologies into unsloth to extend sequence length by wdlctc · Pull Request #1082 · unslothai/unsloth</a>: Description This pull request introduces optimizations to the LLaMA model implementation, specifically targeting the language modeling head and forward pass. The main changes include: Implement a c...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1gwoqm9/llama_32_vision_finetuning_now_in_unsloth_16gb/">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[announcements](https://discord.com/channels/1179035537009545276/1179039782681202829/1309224660961071195)** (1 messages): 

> - `Llama 3.2 Vision`
> - `Vision/Multi-modal Models`
> - `Google Colab Notebooks`
> - `Hugging Face Model Uploads`
> - `Fine-tuning Improvements` 


- **Llama 3.2 Vision boosts performance**: Unsloth now supports **Llama 3.2 Vision models**, achieving **2x faster** training speeds and **70% less memory usage** while allowing for **4-8x longer context lengths**.
   - This enhancement places Unsloth's vision finetuning capability ahead of **Flash Attention 2 (FA2)** and **Hugging Face (HF)** benchmarks.
- **Google Colab Notebooks for Llama 3.2**: Unsloth has provided **Google Colab notebooks** for users to finetune Llama 3.2 Vision on tasks like **Radiography** and **Maths OCR to LaTeX**.
   - These can be accessed via the provided [Colab links](https://colab.research.google.com/drive/1j0N4XTY1zXXy7mPAhOC1_gMYZ2F2EBlk?usp=sharing).
- **Exciting Updates on Hugging Face**: Unsloth's new models are now available on **Hugging Face**, including Llama 3.2 Vision in both **11B** and **90B** variants.
   - Users can explore models like **Qwen 2 VL** and **Pixtral (12B)** through [Hugging Face](https://huggingface.co/unsloth/).
- **Enhancements in Fine-tuning Methods**: The latest improvements to Unsloth's training process allow for **1.5-2x faster** finetuning times compared to previous standards.
   - This efficiency helps equip developers with tools to maximize their machine learning workflows swiftly.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://unsloth.ai/blog/vision">Llama 3.2 Vision Fine-tuning with Unsloth</a>: Fine-tune Meta&#x27;s Llama 3.2 Vision, Llava, Qwen 2.5 Vision models open-source 2x faster via Unsloth! Beginner friendly.</li><li><a href="https://colab.research.google.com/drive/1j0N4XTY1zXXy7mPAhOC1_gMYZ2F2EBlk?usp=sharing)">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1whHb54GNZMrNxIsi2wm2EY_-Pvo2QyKh?usp=sharing)">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1K9ZrdwvZRE96qGkCq_e88FgV3MLnymQq?usp=sharing)">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/18sN803sU23XuJV9Q8On2xgqHSer6-UZF?usp=sharing)">Google Colab</a>: no description found</li><li><a href="https://docs.unsloth.ai/get-started/unsloth-notebooks)">Unsloth Documentation</a>: no description found</li><li><a href="https://docs.unsloth.ai/get-started/all-our-models)">Unsloth Documentation</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1308911100925972501)** (1 messages): 

> - `Training Checkpoints` 


- **Choosing the Right Training Checkpoint**: A member posed a question about which training checkpoint to choose, asking others for their preferences in light of their own choice of **200** checkpoints without hesitation.
   - *Curiosity on diverse approaches to training checkpoints* was expressed, suggesting a possible discussion among participants.
- **Diverse Perspectives on Checkpoint Selection**: Another participant chimed in with their perspective on the selection process for training checkpoints, emphasizing a more strategic approach based on specific metrics.
   - *They encouraged others to consider performance benchmarks* when determining their checkpoint choices.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1308913840687419492)** (122 messages🔥🔥): 

> - `Model Training and Preprocessing`
> - `Fine-tuning Process`
> - `Vision Support`
> - `Using Ollama`
> - `Kubernetes vs SLURM for Training` 


- **Queries on fine-tuning and pre-tokenized datasets**: A user inquired whether it’s feasible to use continued pretraining scripts with an already tokenized dataset, to which another member suggested passing the untokenized dataset instead.
   - The exchange highlighted challenges in the training setup and the specifics of dataset formatting.
- **Evaluation process during fine-tuning**: A user wanted to evaluate their model every 100 steps during fine-tuning and sought advice on how to achieve this, mentioning concerns about starting training from scratch each time.
   - Suggestions included configuring the eval dataset in training arguments and using the `resume_from_checkpoint` feature.
- **Vision support rollout announcement**: A member announced that vision support is now available, generating excitement in the channel.
   - Responses to the announcement included humor about how the information was shared with those previously interested.
- **Discussion on using Ollama for production**: Users discussed using Ollama in their production setup due to its simplicity and ease of use with Docker images, contrasting it with more complex systems.
   - Concerns were raised about Ollama's performance in production environments compared to alternatives like VLLM.
- **Considerations on Kubernetes vs SLURM for HPC**: A user questioned the differences between using Kubernetes and SLURM for training models, particularly regarding multi-GPU setups.
   - The conversation revealed that for single GPU allocations, Kubernetes can work efficiently, but there may be challenges with resource management when requesting multiple GPUs.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.unsloth.ai/basics/saving-and-using-models/saving-to-gguf">Saving to GGUF | Unsloth Documentation</a>: Saving models to 16bit for GGUF so you can use it for Ollama, Jan AI, Open WebUI and more!</li><li><a href="https://docs.unsloth.ai/basics/continued-pretraining">Continued Pretraining | Unsloth Documentation</a>: AKA as Continued Finetuning. Unsloth allows you to continually pretrain so a model can learn a new language.</li><li><a href="https://docs.unsloth.ai/basics/finetuning-from-last-checkpoint">Finetuning from Last Checkpoint | Unsloth Documentation</a>: Checkpointing allows you to save your finetuning progress so you can pause it and then continue.
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1309116343173251132)** (3 messages): 

> - `BFloat16 impact on RoPE`
> - `AnchorAttention method`
> - `Long-context training issues` 


- **BFloat16 disrupts RoPE in Long-Context Training**: A new paper discusses how **BFloat16** casting in **FlashAttention2** causes **RoPE** to deviate from its intended properties, even when RoPE is computed in **Float32**.
   - *BFloat16* introduces critical numerical errors, leading to significant declines in relative positional encoding as context length increases.
- **AnchorAttention boosts long-context performance**: The paper introduces **AnchorAttention**, a plug-and-play solution that enhances long-context performance while cutting training time by over **50%**.
   - This method is adaptable, supporting both [FlashAttention](https://github.com/haonan3/AnchorContext) and FlexAttention for varied applications, including video understanding.
- **Discussion on potential issues caused by BFloat16**: A community member inquired about the implications of the 'breakage' caused by **BFloat16**, speculating it might relate to existing looping issues.
   - They referenced a prior discussion link regarding those looping issues, suggesting a deeper investigation could be beneficial.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2410.08371">Merging in a Bottle: Differentiable Adaptive Merging (DAM) and the Path from Averaging to Automation</a>: By merging models, AI systems can combine the distinct strengths of separate language models, achieving a balance between multiple capabilities without requiring substantial retraining. However, the i...</li><li><a href="https://x.com/Haonan_Wang_/status/1859608786765480516">Tweet from Haonan Wang (@Haonan_Wang_)</a>: 🚀 New Paper📜 When Precision Meets Position: BFloat16 Breaks Down RoPE in Long-Context Training  🤯 RoPE is Broken because of... BFloat16!  &gt; Even if RoPE is computed in Float32 (like in Llama 3 a...
</li>
</ul>

</div>
  

---



### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1308901426868781098)** (98 messages🔥🔥): 

> - `Tülu 3 release`
> - `Nvidia's AI Wall`
> - `Gemini's performance boost`
> - `Reinforcement Learning techniques`
> - `Community discussions on model ranking` 


- **Tülu 3 launched with new techniques**: Nathan Lambert announced the launch of [Tülu 3](https://x.com/natolambert/status/1859643351441535345), an open frontier model that surpassed Llama 3.1 on multiple tasks.
   - The model includes a novel Reinforcement Learning with Verifiable Rewards approach that rewards the algorithm only for correct generations.
- **Nvidia's CEO addresses AI concerns**: An article from The Economist discusses Nvidia's boss downplaying fears that AI has 'hit a wall', despite widespread skepticism.
   - This revelation further fueled discussions about the state of AI advancements and the urgency for innovation.
- **Gemini's ranking surge in Chatbot Arena**: The recent release of [Gemini-Exp-1121](https://x.com/lmarena_ai/status/1859673146837827623) from Google DeepMind has tied for the top spot in the Chatbot Arena, outperforming previous benchmarks.
   - It boasts significant improvements in coding and reasoning capabilities, showcasing rapid advancements in AI.
- **RL techniques spark community interest**: The community engaged in a discussion about the iterations and challenges encountered during the Tülu development, particularly concerning Reinforcement Learning methods.
   - Insights were shared on problems such as the accidental deletion of key model checkpoints that illustrate the finicky nature of RL.
- **Competitiveness among AI models**: Conversations ensued about the rivalry between models like Claude and Gemini, with some community members noting the focus on substantial improvements over mere output formatting.
   - Members chimed in with humor about the current state of AI models, emphasizing the need for both innovation and practicality.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://excalidraw.com/)">Excalidraw — Collaborative whiteboarding made easy</a>: Excalidraw is a virtual collaborative whiteboard tool that lets you easily sketch diagrams that have a hand-drawn feel to them.</li><li><a href="https://x.com/_xjdr/status/1859654054345142727">Tweet from xjdr (@_xjdr)</a>: @TheXeophon lol the very first thing i looked for too. i think they may have indeed cooked</li><li><a href="https://x.com/natolambert/status/1859643351441535345">Tweet from Nathan Lambert (@natolambert)</a>: I&#39;ve spent the last two years scouring all available resources on RLHF specifically and post training broadly. Today, with the help of a totally cracked team, we bring you the fruits of that labor...</li><li><a href="https://x.com/cto_junior/status/1859677125793677572">Tweet from TDM (e/λ) (@cto_junior)</a>: @TheXeophon @teortaxesTex and it&#39;s amazing both model updates remaing virtually the same in ranking if you apply style control + coding/hard prompts</li><li><a href="https://www.economist.com/business/2024/11/21/nvidias-boss-dismisses-fears-that-ai-has-hit-a-wall">Nvidia’s boss dismisses fears that AI has hit a wall</a>: But it’s “urgent” to get to the next level, Jensen Huang tells The Economist   </li><li><a href="https://x.com/natolambert/status/1859664963763306762">Tweet from Nathan Lambert (@natolambert)</a>: So @elonmusk can you open source the Grok post training recipe too?  Quoting Nathan Lambert (@natolambert)   I&#39;ve spent the last two years scouring all available resources on RLHF specifically and...</li><li><a href="https://x.com/lmarena_ai/status/1859673146837827623#">Tweet from lmarena.ai (formerly lmsys.org) (@lmarena_ai)</a>: Woah, huge news again from Chatbot Arena🔥  @GoogleDeepMind’s just released Gemini (Exp 1121) is back stronger (+20 points), tied #1🏅Overall with the latest GPT-4o-1120 in Arena!  Ranking gains since...</li><li><a href="https://x.com/natolambert/status/1859643355698786549">Tweet from Nathan Lambert (@natolambert)</a>: Right to the fun stuff. To finish our models, we use a new technique called Reinforcement Learning with Verifiable Rewards, where we train on math problems or prompts with constraints, and only reward...</li><li><a href="https://x.com/OfficialLoganK/status/1859667244688736419">Tweet from Logan Kilpatrick (@OfficialLoganK)</a>: Say hello to gemini-exp-1121! Our latest experimental gemini model, with:  - significant gains on coding performance - stronger reasoning capabilities - improved visual understanding  Available on Goo...</li><li><a href="https://x.com/alexalbert__/status/1859676984768688231">Tweet from Alex Albert (@alexalbert__)</a>: Claude getting better at things that actually matter while other labs compete over markdown output
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1309226652080803870)** (18 messages🔥): 

> - `Post-training capabilities`
> - `SFT and new skills`
> - `Debate on LLM capabilities`
> - `Diminished philosophy section`
> - `Training efficiency` 


- **Debate on Post-training Capabilities**: A member raised the question of whether post-training merely surfaces existing capabilities or induces entirely new ones, highlighting a need for clarity in the ongoing discussion.
   - Another member pointed out it could be both, especially as it often surfaces latent abilities within the model, prompting further debate.
- **The Role of SFT in Learning**: Members discussed how Supervised Fine-Tuning (SFT) might both surface base model abilities and introduce new ones, depending on the quantity and quality of data used.
   - One noted that while you can achieve impressive results with minimal data, targeted SFT mid-training likely leads to significantly improved performance.
- **Philosophy Section Shrinkage**: The reductions in the paper's philosophy section were noted, with one member humorously lamenting a 'mini manifesto' being condensed to just one paragraph.
   - This led to a light-hearted acknowledgment about the challenges of keeping detailed discussions intact in lengthy papers.
- **Concerns Over Limited Flops**: A member expressed concerns that current model flops may be too low to maximize the potential gains from post-training techniques.
   - They emphasized that while technically it is both surfacing and inducing new capabilities, practically, gaining intuition from the base model is more feasible.


  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1309007321560780861)** (120 messages🔥🔥): 

> - `GPT-4o Performance Analysis`
> - `Perceptron AI Launch`
> - `Model Context Protocol by Anthropic`
> - `Rickrolls in AI`
> - `Issues with Academic AI Resources` 


- **GPT-4o show declining performance metrics**: Independent evaluations of OpenAI’s GPT-4o released on Nov 20 reflect lower quality scores compared to the August version, suggesting it may be a smaller model despite improved output speed.
   - Recommendations were made for developers to carefully test workloads before shifting from the prior models due to these changes.
- **Introducing Perceptron AI's Foundational Models**: Perceptron AI announced its focus on developing foundational models aimed at integrating intelligence into the physical world, claiming to be the first of its kind.
   - The announcement spurred skepticism as multiple companies have made similar promises, prompting questions about their uniqueness.
- **Model Context Protocol (MCP) support arrives for Claude**: Anthropic's Claude Desktop is introducing support for the Model Context Protocol (MCP), allowing local connection capabilities for enhanced interaction with models.
   - The initial support comes with various SDKs, but remote connections remain unavailable, raising curiosity about future updates.
- **Rickrolls in AI interactions become a concern**: Instances of AI models, such as Lindy AI, unexpectedly sending users Rick Astley's music video link illustrate humorous yet concerning lapses in model responses.
   - This issue emphasizes potential gaps in the model's training, leading to unexpected and unwanted outcomes for users.
- **Critique of poor academic resources in AI field**: Frustration over a student's reliance on a subpar academic book about LLMs highlights common challenges in AI education, particularly regarding reliable sources.
   - The critique showcases significant issues in how educational materials in the AI space can mislead, with some publications offering inaccurate or simplistic explanations.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/andrewcurran_/status/1859430019099131934?s=61">Tweet from Andrew Curran (@AndrewCurran_)</a>: Visualization of the change in enterprise market share over the last year.</li><li><a href="https://x.com/damnsec1/status/1610955934683090944">Tweet from 0xDamian (@damnsec1)</a>: Getting Rick rolled by ChatGPT is crazy. 😂</li><li><a href="https://huggingface.co/datasets/allenai/tulu-3-hardcoded-prompts?row=21">allenai/tulu-3-hardcoded-prompts · Datasets at Hugging Face</a>: no description found</li><li><a href="https://x.com/ArmenAgha/status/1859646650714821012">Tweet from Armen Aghajanyan (@ArmenAgha)</a>: Say hello to our new company Perceptron AI.   Foundation models transformed the digital realm, now it’s time for the physical world. We’re building the first foundational models designed for real-time...</li><li><a href="https://www.together.ai/blog/flux-tools-models-together-apis-canny-depth-image-generation">FLUX Tools now available via Together APIs: Get greater control over image generation with Canny, Depth and Redux models</a>: no description found</li><li><a href="https://x.com/UserMac29056/status/1859478751995899995">Tweet from User Mac (@UserMac29056)</a>: gpt-4o-2024-11-20 simple eval updated. tl;dr: benchmark performance got worse</li><li><a href="https://x.com/ArtificialAnlys/status/1859614633654616310">Tweet from Artificial Analysis (@ArtificialAnlys)</a>: Wait - is the new GPT-4o a smaller and less intelligent model?  We have completed running our independent evals on OpenAI’s GPT-4o release yesterday and are consistently measuring materially lower eva...</li><li><a href="https://x.com/btibor91/status/1859385266328531198">Tweet from Tibor Blaho (@btibor91)</a>: Model Context Protocol (MCP) by Anthropic is getting ready for prime time on a new domain modelcontextprotocol[.]io - now, in addition to the Python SDK, there is also a TypeScript SDK + the complete ...</li><li><a href="https://blackforestlabs.ai/flux-1-tools/">Introducing FLUX.1 Tools</a>: Today, we are excited to release FLUX.1 Tools, a suite of models designed to add control and steerability to our base text&#x2d;to&#x2d;image model FLUX.1, enabling the modification and re&#x2d;creati...</li><li><a href="https://techcrunch.com/2024/08/21/this-founder-had-to-train-his-ai-to-not-rickroll-people/">This founder had to train his AI not to Rickroll people | TechCrunch</a>: One AI assistant learned too much from the internet and ended up &quot;rickrolling&quot; a client instead of sharing a tutorial video.
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1309204657805197362)** (34 messages🔥): 

> - `RLHF vs DPO`
> - `Cohere and Alignment`
> - `SnailBot RSS Issues` 


- **Debate on RLHF in AI Models**: Many companies initially doubted the necessity of **RLHF**, with ongoing discussions about its relevance today. A member questioned whether not doing RLHF implies a shift to **DPO** or simply refers to post-training alignment.
   - *I'm working with some of those people and need dirt to rag on them!* sparked humorous engagement.
- **Cohere's Historical Approach**: One member noted that Cohere primarily utilized **IFT** for an extended period, now evolving towards a **RLHF++** approach. This situation raised eyebrows since the **reinforce-loo paper originated from Cohere**.
   - A member expressed confusion over their approach, stating it's peculiar given their foundational work.
- **SnailBot's Technical Glitch**: Discussion revealed that SnailBot is facing issues with double posting **voiceovers** due to it being on a separate RSS feed. A member suggested building a custom solution to manage RSS and Discord webhook interactions more smoothly.
   - An offer was made to help set up a new bot, emphasizing the need for an RSS feed that exclusively covers posts.


  

---



### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1308902090005024848)** (182 messages🔥🔥): 

> - `Hugging Face Models`
> - `Voice Conversion Using RVC`
> - `Community Interactions`
> - `MagicQuil Downloads`
> - `Model Training Issues` 


- **Hugging Face models usage**: Users discussed issues with utilizing various models on Hugging Face, including a query about obtaining large text sizes for better visibility in HuggingChat.
   - Suggestions included setting spaces to private to avoid interruptions from public interactions while testing models.
- **Voice Conversion with RVC**: A user inquired about using the RVC-beta.7z tool from Hugging Face, specifically whether to utilize ov2super during model training or audio conversion.
   - This conversation prompted questions about the setup and functionality of voice conversion tools and expectations for progress.
- **MagicQuil Download Assistance**: A user sought help on how to download creations made with MagicQuil, indicating potential confusion regarding the process.
   - Community members expressed a lack of familiarity with the MagicQuil tool, leaving the user without a clear answer.
- **Debugging and Errors on Hugging Chat**: Concerns were raised regarding frequent errors such as 'Error while parsing tool calls' during interactions in HuggingChat, leading to questions about site overload.
   - The discussions emphasized users' frustrations related to response generation issues and potential overloads while using the platform.
- **Community Dynamics**: The channel displayed vibrant interactions among members, with light-hearted banter and acknowledgments of shared experiences within the community.
   - Members expressed their thoughts on friendship, guidance, and support, enhancing the collaborative spirit in the discussions.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/josephpollack">Tweet from undefined</a>: no description found</li><li><a href="https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/RVC-beta.7z">RVC-beta.7z · lj1995/VoiceConversionWebUI at main</a>: no description found</li><li><a href="https://tenor.com/view/kitty-sleep-kitten-sleep-cuddling-in-bed-cuddle-kitty-cuddle-gif-707600157996049423">Kitty Sleep Kitten Sleep GIF - Kitty sleep Kitten sleep Cuddling in bed - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.runpod.io">RunPod - The Cloud Built for AI</a>: Develop, train, and scale AI models in one cloud. Spin up on-demand GPUs with GPU Cloud, scale ML inference with Serverless.</li><li><a href="https://github.com/CPJKU/madmom">GitHub - CPJKU/madmom: Python audio and music signal processing library</a>: Python audio and music signal processing library. Contribute to CPJKU/madmom development by creating an account on GitHub.</li><li><a href="https://blackforestlabs.ai/">Black Forest Labs &#x2d; Frontier AI Lab</a>: Amazing AI models from the Black Forest.</li><li><a href="https://tenor.com/view/sunday-cult-of-the-lamb-cult-happy-sunday-god-gif-422811577611096801">Sunday Cult Of The Lamb GIF - Sunday Cult of the lamb Cult - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/simpsons-homer-bart-lisa-join-us-gif-17846376318791889140">Simpsons Homer GIF - Simpsons Homer Bart - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/rabbit-bunny-toilet-yes-come-gif-4686108">Rabbit Bunny GIF - Rabbit Bunny Toilet - Discover &amp; Share GIFs</a>: Click to view the GIF
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1308899823461601280)** (10 messages🔥): 

> - `Custom AI Models with Handler Files`
> - `AI Security Research Paper`
> - `Automated AI Research Assistant`
> - `Collaborative Learning Framework in LLMs`
> - `Efficient Quantization Method for Attention` 


- **Deploy Custom AI Models Using Handler Files**: Hugging Face Endpoints allows deploying custom AI models via a [handler.py](https://huggingface.co/philschmid/distilbert-onnx-banking77/blob/main/handler.py) file, facilitating custom pre- and post-processing.
   - This handler needs to implement the [EndpointHandler](https://huggingface.co/philschmid/distilbert-onnx-banking77/blob/main/handler.py) class, ensuring flexibility in model deployment.
- **Redhat/IBM's AI Security Insights**: A new [research paper](https://huggingface.co/papers/2411.12275) discusses the risks associated with publicly available AI models and proposes strategies to enhance security and safety in AI development.
   - The paper highlights challenges such as tracking issues and the absence of lifecycle processes, aiming to foster more standardized practices in the AI ecosystem.
- **Automated AI Research Assistant Created**: An innovative python program transforms local LLMs into automated web researchers, providing detailed summaries and sources based on user queries.
   - The program intelligently breaks down queries into subtopics, systematically gathering and analyzing information from the web.
- **FreeAL: Collaborative Learning for LLMs**: The paper [FreeAL](https://arxiv.org/abs/2311.15614) proposes an advanced collaborative learning framework aimed at reducing label annotation costs in LLM training.
   - By utilizing an LLM as an active annotator, it distills task-specific knowledge interactively to improve the quality of dataset samples.
- **SageAttention: Quantization in Attention Models**: The method [SageAttention](https://arxiv.org/abs/2410.02367) offers an efficient quantization approach for attention mechanisms, significantly improving operations per second compared to current methods.
   - It enhances accuracy while addressing the computational limits of attention, which traditionally sees high complexity in larger sequences.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2410.02367">SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration</a>: The transformer architecture predominates across various models. As the heart of the transformer, attention has a computational complexity of O(N^2), compared to O(N) for linear transformations. When ...</li><li><a href="https://arxiv.org/abs/2311.15614">FreeAL: Towards Human-Free Active Learning in the Era of Large Language Models</a>: Collecting high-quality labeled data for model training is notoriously time-consuming and labor-intensive for various NLP tasks. While copious solutions, such as active learning for small language mod...</li><li><a href="https://whcompsci.github.io/projects/neural-networks/index.html">no title found</a>: no description found</li><li><a href="https://huggingface.co/docs/inference-endpoints/main/en/guides/custom_handler#create-custom-inference-handler">Create custom Inference Handler</a>: no description found</li><li><a href="https://huggingface.co/spaces/fffiloni/clone-git-repo-to-space">Clone Git Repo To Space - a Hugging Face Space by fffiloni</a>: no description found</li><li><a href="https://huggingface.co/papers/2411.12275">Paper page - Building Trust: Foundations of Security, Safety and Transparency in AI</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1gvlzug/i_created_an_ai_research_assistant_that_actually/">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1308916077509611601)** (9 messages🔥): 

> - `Neo's Red Pill Journey`
> - `Prompting Techniques`
> - `MOUSE-I Web Service`
> - `Cinematic Image Generation`
> - `loadimg Downloads Milestone` 


- **Neo's Red Pill sends him to the 60s**: A discussion initiated around a [YouTube video](https://youtu.be/JugL1okFCqI?si=zn7wpJFaQJnQcJx3) exploring what would happen if Neo's red pill transported him back to the 1960s.
   - Members expressed excitement about the share, highlighting the video's coolness.
- **The Importance of Prompting**: A [YouTube video](https://youtu.be/m3Izr0wNfQc) titled 'BAD vs GOOD prompting' raises questions on whether effective prompting is still necessary in current applications.
   - The description invites viewers to explore when and how prompting techniques might differ.
- **MOUSE-I revolutionizes web development**: Introducing MOUSE-I, which converts a simple prompt into a globally deployed web service in just **60 seconds** using [AI automation](https://huggingface.co/spaces/VIDraft/mouse1).
   - It promises instant results and is suitable for startups, developers, and educators alike.
- **Cinematic Images with a Twist**: A project shared on Hugging Face allows users to create cinematic images with preset widescreen aspect ratios at more than **2 megapixels**.
   - The link to this tool can be found [here](https://huggingface.co/spaces/takarajordan/CineDiffusion).
- **loadimg hits 500,000 downloads**: A community member celebrated that **loadimg** has achieved **500,000 downloads** and is now compatible with Hugging Face and OpenAI SDKs.
   - This milestone showcases the tool's growing popularity and compatibility.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://whcompsci.github.io/projects/neural-networks/index.html">no title found</a>: no description found</li><li><a href="https://huggingface.co/spaces/VIDraft/mouse1">Mouse1 - a Hugging Face Space by VIDraft</a>: no description found</li><li><a href="https://youtu.be/m3Izr0wNfQc">BAD vs GOOD prompting</a>: Let&#39;s see in this video if we still need to make good prompting nowadays and if there is a difference, at what point is it different.Feel free to leave comme...</li><li><a href="https://huggingface.co/spaces/takarajordan/CineDiffusion">CineDiffusion - a Hugging Face Space by takarajordan</a>: no description found</li><li><a href="https://github.com/gmontamat/screen-grep">GitHub - gmontamat/screen-grep: Open source alternative to Windows Recall</a>: Open source alternative to Windows Recall. Contribute to gmontamat/screen-grep development by creating an account on GitHub.</li><li><a href="https://appine.tech/app/flux-image-generation">Appine</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/)** (1 messages): 

crazypistachecat: Sory👍
  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1308995464884129893)** (6 messages): 

> - `YOLO for Video Object Detection`
> - `Stable Diffusion`
> - `Autodistill Training Method` 


- **YOLO shines in Video Object Detection**: A member highlighted that **YOLO supports video object detection**, making it a classic choice for this task.
   - There are ongoing discussions on using it effectively, with a [linked resource](https://huggingface.co/spaces/prithivMLmods/YOLO-VIDEO) provided for further exploration.
- **Stable Diffusion remains a classic**: Another member brought up **Stable Diffusion** as a well-known and classic model for various applications in image processing.
   - This prompted discussions on its longevity and effectiveness compared to newer models.
- **Struggles with YOLO labeling**: A user expressed difficulties in getting **YOLO** to label correctly online, seeking guidance and solutions.
   - The conversation emphasized the need for clear instructions or support to effectively utilize the model.
- **Autodistill simplifies model training**: A member shared about **Autodistill**, which trains small supervised models using larger foundation models, highlighting its efficiency.
   - They provided a detailed [documentation link](https://docs.autodistill.com/) for those interested in training models faster with minimal human intervention.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.autodistill.com/">Home - Autodistill</a>: no description found</li><li><a href="https://huggingface.co/spaces/prithivMLmods/YOLO-VIDEO">YOLO VIDEO - a Hugging Face Space by prithivMLmods</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1308922972592934953)** (9 messages🔥): 

> - `Pandas alternatives`
> - `Building local Agents`
> - `Fast inference frameworks`
> - `Scaling data processing` 


- **Consider Polars for faster data processing**: A member suggested using **Polars** as a faster alternative to **Pandas** when dealing with large datasets.
   - *Swetha98* appreciated the recommendation and plans to explore it further.
- **NVIDIA RAPIDS for GPU acceleration**: A recommendation was made for **NVIDIA RAPIDS**, which is said to run on GPU and could help with scalability.
   - However, *Swetha98* mentioned a lack of GPU access, complicating the suggestion.
- **Local Agent frameworks comparison**: *Pizzadrones* inquired about frameworks that compete with **AnythingLLM** and **Ottobot** for building local Agents.
   - The discussion focused on alternatives to these frameworks in the chat.
- **vLLM touted as the top inference framework**: *Takarajordan_82155* noted that **vLLM** is currently highly regarded as the best inference framework for LLMs.
   - This statement sparked interest in exploring different frameworks for rapid inference.
- **Llama 3.2 leading inference performance**: In discussing effective chat LLMs, *Takarajordan_82155* proposed **Llama 3.2 90B on Groq** as a strong contender.
   - *Abhijithneilabraham* asked for leaderboards to track the best generic chat LLMs with fast inference.


  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1309067173980799036)** (13 messages🔥): 

> - `Open Source Models like SSD-1B`
> - `Using SDXL in Google Colab`
> - `Token Embeddings in Shuttle-3` 


- **Recommendations for SSD-1B Alternatives**: Members discussed alternatives to **SSD-1B**, suggesting the step distilled version of **SDXL** or **DreamShaper Lightning** for potentially better quality and speed.
   - A link to [Hyper-SD](https://huggingface.co/ByteDance/Hyper-SD) was shared, highlighting its ease of use for similar purposes.
- **Loading Models on Low VRAM in Google Colab**: Concerns about loading models with low GPU resources were addressed, confirming that **SDXL's** size of **6.5GB** is manageable within Google Colab's 16GB VRAM.
   - Members noted that after loading the base model, it might be easier to save the model state for future use.
- **Token Limitations in Shuttle-3 Pipeline**: A member inquired about increasing token embeddings beyond **77** in the **shuttle-3** pipeline, which seems unsupported currently.
   - Discussion indicated that while the **CLIP** encoder may truncate prompts, the **T5** encoder can handle up to **256** tokens without issues.
- **Impact of Encoding on Image Generation**: Questions were raised about the effect of truncating data for the **CLIP** encoder and whether important prompt data should be prioritized.
   - Members weighed in, noting that while truncation occurs for **CLIP**, it may not significantly affect image generation outcomes.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/ByteDance/Hyper-SD">ByteDance/Hyper-SD · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/tianweiy/DMD2">tianweiy/DMD2 · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---



### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1308925019723796542)** (209 messages🔥🔥): 

> - `AI Discussions on Censorship`
> - `Opinions on Agnosticism`
> - `Deep Web and AI Access`
> - `OpenAI's Censorship Policies`
> - `Perplexity vs ChatGPT` 


- **Debate on AI Censorship Importance**: Users discussed the need for censorship in AI to prevent harmful uses, with one member arguing **OpenAI** aims to avoid **legal trouble** by implementing moderation.
   - Concerns were raised about some possible overreach and false positives in the moderation process, but the general consensus favored a cautious approach.
- **Subjectivity of Agnosticism Explained**: A lively debate ensued over whether agnosticism is subjective, with members asserting that it represents a **lack of evidence** rather than personal belief.
   - One user highlighted that agnosticism can be considered a **healthy mindset**, while another pointed out the redundancy of claiming a 'subjective opinion'.
- **Exploration of AI's Capabilities**: Discussion centered around whether AI could access the deep web and how it functions theoretically, with some suggesting that customized AI could scan the web unfiltered.
   - One user noted that while AI doesn't access the web, passing external content into it is possible, which could raise ethical questions.
- **OpenAI's Approach to Political Questions**: Participants agreed that discussing politics with **ChatGPT** is allowed, as long as it does not lead to harmful or misleading content.
   - Members recognized that while politics can be part of the conversation, unnecessary political bias should be avoided.
- **Comparison of AI Models**: Users compared **Perplexity** with **ChatGPT**, emphasizing that Perplexity was seen as more accurate and had advantages in certain functionalities.
   - One participant noted that specific features of Perplexity had been in development before they gained popularity.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/)** (1 messages): 

grundaypress: Hi, does anyone know why the retry button is gone when using Custom GPTs?
  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1308940975636484126)** (3 messages): 

> - `Categorizing Products with GPT-4`
> - `Prompt Optimization`
> - `Prompt Caching for Efficiency` 


- **Product Categorization with GPT-4**: A member shared their experience categorizing products using a prompt with GPT-4, specifying categories from **groceries** to **clothing**.
   - They highlighted that while results are great, the token usage is quite high due to the lengthy prompt structure.
- **Token Reduction Strategies**: Discussion revolved around minimizing token usage while maintaining effectiveness in categorization prompts.
   - Suggestions included exploring methods like prompt caching to streamline the process and reduce redundancy.
- **API Assistance Reference**: A user provided a link to a Discord channel for help with API-related queries about improving prompt efficiency.
   - This reference aims to assist others in the community facing similar challenges in product categorization.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1308940975636484126)** (3 messages): 

> - `Product Categorization with GPT-4o`
> - `Token Optimization Strategies`
> - `Prompt Caching in API usage` 


- **Categorizing Products with GPT-4o**: A member described using GPT-4o to categorize products based on a title and image with a comprehensive prompt structure.
   - *The prompt produces excellent results, but the extensive token usage poses a challenge for scalability.*
- **Efficient Token Usage Suggestions**: Another member suggested exploring prompt caching techniques to reduce repetitive input tokens in their product categorization setup.
   - They recommended consulting API-related resources for further aid in optimizing token usage as linked in their message.


  

---



### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1308903972869771284)** (63 messages🔥🔥): 

> - `Hermes 3 Model Performance`
> - `Cloud-Based GPU Renting`
> - `LLM GPU Comparisons`
> - `MLX Models in LMS`
> - `Graphics Card Recommendations` 


- **Hermes 3 impresses users**: A member expressed their preference for **Hermes 3**, highlighting its strong writing skills and ability to follow prompts effectively.
   - However, they noted that at higher contexts (**16K+**), it tends to repeat phrases.
- **Cloud-based servers for model hosting**: A member reported switching from local model hosting to a cloud server, costing about **$25-50 per month**, effectively running models faster.
   - This shift was noted as superior to local hardware due to performance and cost efficiency.
- **Choosing between AMD and NVIDIA cards**: Members discussed the pros and cons of choosing **AMD** versus **NVIDIA** GPUs, with recent drivers affecting AMD's ROCM support.
   - There's a consensus that for better software support and compatibility, sticking with NVIDIA is recommended.
- **Interest in MLX models integration**: A request was made for announcements regarding interesting **MLX models** given LMS's recent support for them.
   - The integration of MLX models was acknowledged as a valuable addition, even if these models are not official to LMS.
- **Graphics card decision-making**: Discussion centered on the choice between the **4070 Ti Super** and **7900 XTX**, weighing factors like gaming performance versus LLM applications.
   - A member pointed out that while the **3090** excels in various tasks, the **7900 XTX** could be a cheaper alternative for gaming.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-70B">NousResearch/Hermes-3-Llama-3.1-70B · Hugging Face</a>: no description found</li><li><a href="https://www.techpowerup.com/review/gigabyte-geforce-rtx-4070-ti-super-gaming-oc/">Gigabyte GeForce RTX 4070 Ti Super Gaming OC Review</a>: The Gigabyte RTX 4070 Ti Super Gaming OC, true to its name, comes with a factory overclock, to a rated boost of 2655 MHz. It features a triple-slot, triple-fan cooler and dual BIOS for added versatili...
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1308918403293122595)** (135 messages🔥🔥): 

> - `AI chip discussions`
> - `Performance of GPUs`
> - `Building a local LLM server`
> - `USB4 with AMD devices`
> - `Challenges with GPU configurations` 


- **Performance Issues with Mixed GPU Configurations**: Users reported that adding GPUs to configurations often slows down performance, with benchmarks showing 1x 4090 + 2x A6000 performing worse than other combinations due to shared resource ceilings.
   - One user noted that adding a 4090 to their A6000 setup decreased token generation rate, highlighting that *the slowest card often dictates the overall speed*.
- **Considerations for Local LLM Server Setup**: A user questioned whether a $2,000 budget for a local LLM server catering to 2-10 users is feasible, given the challenges of concurrent access with single GPUs.
   - Developers suggested exploring cloud solutions to avoid performance bottlenecks associated with budget setups utilizing older hardware and fewer GPUs.
- **Insights on USB4 and EGPU Setup**: Users discussed the use of a specific GitHub tool to enable USB4 on newer AMD devices while sharing experiences with external GPU configurations like A4000s and P40s.
   - One member reported success with A4000s being plug-and-play, while concerns lingered regarding the compatibility of P40s and potential legacy PCIe mode issues.
- **Shipping Delays and E-Commerce Challenges**: There were discussions regarding shipping delays from online vendors like AliExpress, with one user highlighting delays due to possible strikes affecting delivery times.
   - Overall, members expressed frustration with inconsistent shipping experiences and the impact on ongoing hardware builds.
- **Utilization of AMD's Firmware for Hardware Optimization**: It was suggested that AMD laptop users could access hidden firmware menus to modify settings using a GitHub tool, enhancing their hardware performance.
   - Members exchanged ideas regarding how enabling certain settings like MMIO and above 4G decoding could potentially improve eGPU connectivity and performance.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/fascinating-mr-spock-henoch-star-trek-the-original-series-gif-23404763">Fascinating Mr Spock GIF - Fascinating Mr Spock Henoch - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/sniped-piggy-piggyverse-sniper-piggy-sniper-gif-2353208795072333005">Sniped Piggy GIF - Sniped Piggy Piggyverse - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/senator-palpatine-anakin-skywalker-phantom-menace-gif-10607636">Senator Palpatine Anakin Skywalker GIF - Senator Palpatine Anakin Skywalker Phantom Menace - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/v1ckxy/LMSPLS">GitHub - v1ckxy/LMSPLS: LM Studio Portable Launch Script (LMS PLS)</a>: LM Studio Portable Launch Script (LMS PLS). Contribute to v1ckxy/LMSPLS development by creating an account on GitHub.</li><li><a href="https://github.com/DavidS95/Smokeless_UMAF">GitHub - DavidS95/Smokeless_UMAF</a>: Contribute to DavidS95/Smokeless_UMAF development by creating an account on GitHub.</li><li><a href="https://x.com/_Holistech_/status/1859395091384893820/photo/1">Tweet from Holistech (@_Holistech_)</a>: @rasbt @ivanfioravanti I hope they will fit together with my M1 Ultra in my 6 screen mobile and foldable AI workstation setup</li><li><a href="https://github.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference">GitHub - XiongjieDai/GPU-Benchmarks-on-LLM-Inference: Multiple NVIDIA GPUs or Apple Silicon for Large Language Model Inference?</a>: Multiple NVIDIA GPUs or Apple Silicon for Large Language Model Inference? - XiongjieDai/GPU-Benchmarks-on-LLM-Inference
</li>
</ul>

</div>
  

---



### **aider (Paul Gauthier) ▷ #[announcements](https://discord.com/channels/1131200896827654144/1133060115264712836/1309243124467372134)** (2 messages): 

> - `Qwen 2.5 Model Performance`
> - `Aider v0.64.0 Features`
> - `Model Quantization Impact`
> - `Slash Commands in Aider`
> - `Context Window and Token Costs` 


- **Qwen 2.5 Model Rivals GPT-4o**: Open source models like **Qwen 2.5 32B** show excellent performance on Aider's code editing benchmarks, rivalling closed source frontier models with significant differences in quantization impact.
   - The best version competes with **GPT-4o**, while the least effective resembles **GPT-3.5 Turbo**, inviting users to pay careful attention to quantization effects.
- **New Features in Aider v0.64.0**: The latest Aider version adds a new [`/editor`](https://aider.chat/docs/usage/commands.html) command for prompt writing and full support for **gpt-4o-2024-11-20**.
   - This update improves shell command clarity, enabling users to see confirmations and opt-in for [analytics](https://aider.chat/docs/more/analytics.html).
- **Understanding Model Quantization**: Attention is drawn to how model **quantization** affects performance, particularly in code editing, as heavily quantized models are prevalent in cloud solutions.
   - The discussion highlights different serving methods and their performance impact, guiding users toward better model configurations.
- **Exploration of Slash Commands**: Aider supports various **slash commands** like `/add`, `/architect`, and `/chat-mode`, simplifying users' interactions within the chat environment.
   - These commands enhance editing and reviewing capabilities, promoting efficient communication and code management.
- **Registering Context Window Limits**: Users can register context window limits and costs for models with unknown parameters by creating a `.aider.model.metadata.json` file in specified directories.
   - This functionality allows for better resource management, accommodating broader model configurations within Aider.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://aider.chat/2024/11/21/quantization.html">Quantization matters</a>: Open source LLMs are becoming very powerful, but pay attention to how you (or your provider) is quantizing the model. It can strongly affect code editing skill.</li><li><a href="https://aider.chat/docs/usage/commands.html#entering-multi-line-chat-messages)">In-chat commands</a>: Control aider with in-chat commands like /add, /model, etc.</li><li><a href="https://aider.chat/docs/config/adv-model-settings.html#global-extra-params)">Advanced model settings</a>: Configuring advanced settings for LLMs.
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1308905869056151594)** (133 messages🔥🔥): 

> - `Aider Leaderboard Changes`
> - `OpenRouter Providers and Quantization`
> - `Gemini Model Performance`
> - `DeepSeek Model Developments`
> - `User Experiences with AI Models` 


- **Aider Leaderboard Enhancements**: Aider has introduced a new search/filter box on the leaderboard tables and graphs to improve usability and allow users to quickly find specific models.
   - Users have noted that the addition greatly enhances navigation and access to model performance data.
- **Concerns Over OpenRouter's Model Choices**: Participants expressed concerns regarding OpenRouter using quantized versions of models, which may lead to misleading performance expectations.
   - Discussion highlighted the importance of being aware of quantization levels and the specific providers being used for AI models.
- **Performance Insights on Gemini Model**: Recent benchmarks indicated mixed performance for the new Gemini model, with different results from standard diff and diff-fenced formats.
   - While gemini-exp-1121 scored 58% on the diff format, there were concerns about lower scores observed with previous iterations and different formats.
- **DeepSeek Model Updates**: DeepSeek announced new capabilities with the launch of the DeepSeek-R1-Lite-Preview, touted for impressive performance on coding benchmarks.
   - Users discussed the practical implications of using various models, including speed and efficiency advantages.
- **User Experiences with AI Assistants**: Users shared humorous anecdotes about their interactions with various AI models, noting quirky behavior and unexpected outputs.
   - The community reflected on the learning curve and challenges of effectively utilizing different AI tools for coding and other tasks.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://aider.chat/docs/leaderboards/">Aider LLM Leaderboards</a>: Quantitative benchmarks of LLM code editing skill.</li><li><a href="https://www.alibabacloud.com/help/en/model-studio/developer-reference/use-qwen-by-calling-api">
 Use the Qwen API - Alibaba Cloud Model Studio - Alibaba Cloud Documentation Center

</a>: no description found</li><li><a href="https://aider.chat/docs/config/adv-model-settings.html#global-extra-params">Advanced model settings</a>: Configuring advanced settings for LLMs.</li><li><a href="https://openrouter.ai/qwen/qwen-2.5-coder-32b-instruct">Qwen2.5 Coder 32B Instruct - API, Providers, Stats</a>: Qwen2.5-Coder is the latest series of Code-Specific Qwen large language models (formerly known as CodeQwen). Run Qwen2.5 Coder 32B Instruct with API</li><li><a href="https://x.com/zhs05232838/status/1859201857593524352">Tweet from Zhihong Shao (@zhs05232838)</a>: Our DeepSeek reasoning model is great on code and math. Try it out!  Quoting DeepSeek (@deepseek_ai)   🚀 DeepSeek-R1-Lite-Preview is now live: unleashing supercharged reasoning power!  🔍 o1-preview-...</li><li><a href="https://ai.google.dev/gemini-api/docs/models/experimental-models">no title found</a>: no description found</li><li><a href="https://github.com/BerriAI/litellm/issues/6857">[Feature]: Support OpenRouter&#39;s &quot;provider&quot; argument to control/select providers · Issue #6857 · BerriAI/litellm</a>: The Feature OpenRouter supports a variety of mechanisms to select which providers you want your requests to hit. This involves passing a provider argument. Currently that causes an error: import li...</li><li><a href="https://openrouter.ai/docs/provider-routing#quantization-levels">Provider Routing | OpenRouter</a>: Route requests across multiple providers</li><li><a href="https://openrouter.ai/docs/provider-routing">Provider Routing | OpenRouter</a>: Route requests across multiple providers</li><li><a href="https://openrouter.ai/models">Models | OpenRouter</a>: Browse models on OpenRouter</li><li><a href="https://www.youtube.com/shorts/7smV_9eVM1M">Qwen on openrouter#aider #lmsys #qwen #llm #aicoding #huggingface</a>: no description found</li><li><a href="https://openrouter.ai/deepseek/deepseek-chat">DeepSeek V2.5 - API, Providers, Stats</a>: DeepSeek-V2.5 is an upgraded version that combines DeepSeek-V2-Chat and DeepSeek-Coder-V2-Instruct. Run DeepSeek V2.5 with API</li><li><a href="https://api-docs.deepseek.com/news/news1120">🚀 DeepSeek-R1-Lite-Preview is now live: unleashing supercharged reasoning power! | DeepSeek API Docs</a>: 🔍 o1-preview-level performance on AIME &amp; MATH benchmarks.</li><li><a href="https://openrouter.ai/meta-llama/llama-3.1-70b-instruct/providers">Meta: Llama 3.1 70B Instruct – Provider Status</a>: See provider status and make a load-balanced request to Meta: Llama 3.1 70B Instruct - Meta&#x27;s latest class of model (Llama 3.1) launched with a variety of sizes &amp; flavors. This 70B instruct-t...</li><li><a href="https://www.alibabacloud.com/en/solutions/generative-ai/qwen">Tongyi Qianwen (Qwen) - Alibaba Cloud</a>: Top-performance foundation models from Alibaba Cloud</li><li><a href="https://www.alibabacloud.com/en/product/modelstudio">Alibaba Cloud Model Studio - Alibaba Cloud</a>: A one-stop generative AI platform to build intelligent applications that understand your business, based on Qwen and other popular models</li><li><a href="https://help.aliyun.com/zh/model-studio/getting-started/models">模型列表_大模型服务平台百炼(Model Studio)-阿里云帮助中心</a>: no description found
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1308900081478668300)** (42 messages🔥): 

> - `Looping with Aider`
> - `Aider Caching Efficiency`
> - `Disabling Autocomplete in Aider`
> - `API Approval for Aider`
> - `Recommendations for Model Combinations` 


- **Looping in Aider for Checklists**: A user inquired about using Aider for automating checklist tasks by running a loop to avoid token limits, with suggestions to wrap Aider in an outer script or use the `-m` mode.
   - Another member shared a command-line approach for scripting, utilizing `aider --message` to automate tasks within a shell script.
- **Caching Mechanism Usage**: A user expressed concerns over high costs associated with Aider, prompting a discussion about the effectiveness of the `--prompt-caching` feature and how frequently caches are utilized.
   - Recommendations included exploring cache settings like `AIDER_CACHE_PROMPTS` to analyze the cost-saving potential of caching.
- **Disabling Autocomplete Functionality**: A user sought help to disable the file autocomplete feature in Aider, which was causing distractions during coding.
   - Suggestions included using the `--no-fancy-input` option, although users preferred to maintain certain features without the autocomplete pop-ups.
- **IT Approval Process for Aider**: A discussion opened regarding the challenges of getting IT and legal departments to approve the use of Aider, focusing on data retention policies and intellectual property concerns.
   - Documentation regarding analytics and privacy policies was mentioned as helpful for addressing concerns, alongside noting that API providers also have their own policies.
- **Model Combinations for Projects**: A user sought insights on which combinations of models are currently effective for real projects, hinting at notable models like Qwen2.5.1 and DeepSeek.
   - This inquiry reflects ongoing interest in optimizing model usage for practical implementations in AI development.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://aider.chat/docs/usage/caching.html">Prompt caching</a>: Aider supports prompt caching for cost savings and faster coding.</li><li><a href="https://aider.chat/docs/scripting.html">Scripting aider</a>: You can script aider via the command line or python.</li><li><a href="https://aider.chat/docs/more/analytics.html">Analytics</a>: Opt-in, anonymous, no personal info.
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1309236740019458100)** (2 messages): 

> - `Gemini API`
> - `uithub` 


- **Gemini API boosts coding and reasoning skills**: The [Gemini Experimental Model](https://ai.google.dev/gemini-api/docs/models/experimental-models) introduced improved **coding**, **reasoning**, and **vision capabilities** as of **November 21, 2024**.
   - This update aims to enhance user interactions with AI through sophisticated understanding and functionalities.
- **uithub redefines GitHub interactions**: [uithub.com](http://uithub.com) allows users to replace 'g' with 'u' in GitHub links for instant **copy-paste** repository interactions, enhancing LLM contexts.
   - Users like **Nick Dobos** and **Ian Nuttall** praised the tool, with Nuttall noting it delivers **full repo context** effectively.
- **Community embraces uithub tool**: Various users shared their experiences with uithub on Twitter, calling it a **helpful tool** that simplifies LLM coding questions.
   - **Yohei Nakajima** expressed enthusiasm for discovering uithub, commenting on its practicality and utility.
- **uithub offers unique capabilities**: uithub's features are compared to **--show-repo-map**, suggesting it generates more tokens and provides advanced filtering options for specific file types.
   - However, it lacks certain intricate functionalities seen in other tools, leaving some users to prefer the standard aider tools for complex tasks.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://ai.google.dev/gemini-api/docs/models/experimental-models">no title found</a>: no description found</li><li><a href="https://uithub.com/">uithub - Easily ask your LLM code questions</a>: no description found
</li>
</ul>

</div>
  

---



### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1309006009829687346)** (2 messages): 

> - `New models release`
> - `High context provider selection` 


- **New models introduced this week**: The **GPT-4o** has launched with better prose, details available [here](https://openrouter.ai/openai/gpt-4o-2024-11-20). Other new models include **Mistral Large** ([link](https://openrouter.ai/mistralai/mistral-large-2411)), **Pixtral Large** ([link](https://openrouter.ai/mistralai/pixtral-large-2411)), **Grok Vision Beta** ([link](https://openrouter.ai/x-ai/grok-vision-beta)), and **Gemini Experimental 1114** ([link](https://openrouter.ai/google/gemini-exp-1114)).
- **Selecting providers for high context prompts**: Users have shown confusion about how to select providers that support high context; OpenRouter automatically routes to those that do. If you send a long prompt or max tokens, providers with small context or max output are filtered out.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openrouter.ai/openai/gpt-4o-2024-11-20">GPT-4o (2024-11-20) - API, Providers, Stats</a>: The 2024-11-20 version of GPT-4o offers a leveled-up creative writing ability with more natural, engaging, and tailored writing to improve relevance &amp; readability. It’s also better at working with...</li><li><a href="https://openrouter.ai/mistralai/mistral-large-2411">Mistral Large 2411 - API, Providers, Stats</a>: Mistral Large 2 2411 is an update of [Mistral Large 2](/mistralai/mistral-large) released together with [Pixtral Large 2411](mistralai/pixtral-large-2411)  It is fluent in English, French, Spanish, Ge...</li><li><a href="https://openrouter.ai/mistralai/pixtral-large-2411">Pixtral Large 2411 - API, Providers, Stats</a>: Pixtral Large is a 124B open-weights multimodal model built on top of [Mistral Large 2](/mistralai/mistral-large-2411). The model is able to understand documents, charts and natural images. Run Pixtra...</li><li><a href="https://openrouter.ai/x-ai/grok-vision-beta">Grok Vision Beta - API, Providers, Stats</a>: Grok Vision Beta is xAI&#x27;s experimental language model with vision capability.  . Run Grok Vision Beta with API</li><li><a href="https://openrouter.ai/google/gemini-exp-1114">Gemini Experimental 1114 - API, Providers, Stats</a>: Gemini 11-14 (2024) experimental model features &quot;quality&quot; improvements.. Run Gemini Experimental 1114 with API
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1308902211547304018)** (162 messages🔥🔥): 

> - `Mistral Model Issues`
> - `OpenRouter API Functionality`
> - `Gemini Experimental Models`
> - `File Upload Capabilities`
> - `Community Engagement in OpenRouter` 


- **Mistral Model Facing Deprecation**: Users reported that the **Mistral Medium** model has been deprecated, causing an error when accessed, indicating that **priority is not enabled** for it.
   - Members suggested switching to **Mistral-Large**, **Mistral-Small**, or **Mistral-Tiny** to continue using the service.
- **OpenRouter API Documentation Cleared Up**: Users expressed confusion regarding certain functionalities in the OpenRouter API documentation, specifically around context window capabilities.
   - It was suggested to enhance clarity in documentation to aid understanding for users integrating OpenRouter with tools like LangChain.
- **New Gemini Experimental Models Update**: The **Gemini Experimental 1121** model has been introduced, with claims of improved coding, reasoning, and vision capabilities.
   - Users noted the existing quota restrictions shared with the **LearnLM** model and expressed curiosity about the model’s performance.
- **File Upload Capability in Models**: Discussion arose regarding file upload limitations, with users questioning if any models accept non-image formats.
   - It was clarified that image uploads are supported, and the recent infrastructure upgrades may have lifted the previous **4MB** restriction.
- **Community Building and Founder Insights**: A user inquired about the founding of OpenRouter and the motivations behind its creation.
   - Community engagement was highlighted as a significant factor in OpenRouter’s development, and suggestions for a write-up on its story were mentioned.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openrouter.ai/chat">Chatroom | OpenRouter</a>: LLM Chatroom is a multimodel chat interface. Add models and start chatting! Chatroom stores data locally in your browser.</li><li><a href="https://openrouter.ai/liquid/lfm-40b:free">LFM 40B MoE (free) - API, Providers, Stats</a>: Liquid&#x27;s 40.3B Mixture of Experts (MoE) model. Run LFM 40B MoE (free) with API</li><li><a href="https://openrouter.ai/meta-llama/llama-3.1-8b-instruct:free">Llama 3.1 8B Instruct (free) - API, Providers, Stats</a>: Meta&#x27;s latest class of model (Llama 3.1) launched with a variety of sizes &amp; flavors. Run Llama 3.1 8B Instruct (free) with API</li><li><a href="https://ai.google.dev/gemini-api/docs/models/experimental-models">no title found</a>: no description found</li><li><a href="https://tenor.com/view/despicable-me-animation-movies-dream-works-minions-gif-13754998145004207015">Despicable Me Animation GIF - Despicable Me Animation Movies - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://status.openrouter.ai/">OpenRouter Status</a>: OpenRouter Incident History</li><li><a href="https://openrouter.ai/docs/requests#images-_-multimodal-requests">Requests | OpenRouter</a>: Handle incoming and outgoing requests</li><li><a href="https://openrouter.ai/docs/requests#images-_-multimodal-re">Requests | OpenRouter</a>: Handle incoming and outgoing requests
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[beta-feedback](https://discord.com/channels/1091220969173028894/1277894087755829278/1309196960049266709)** (1 messages): 

> - `Claude 3.5`
> - `Custom provider key requests` 


- **User requests custom provider key for Claude 3.5 Sonnet**: A member requested a **custom provider key** for **Claude 3.5 Sonnet**, expressing frustration with running out of usage on the main **Claude app**.
   - They hope that this request will provide a viable alternative to their current constraints.
- **Concerns about Claude app usage limits**: The discussion highlighted issues regarding the **usage limits** on the main **Claude app**, leading to user frustration.
   - Members are seeking solutions to manage their usage more effectively and improve their experience.


  

---



### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1308901712312144033)** (164 messages🔥🔥): 

> - `Flux performance`
> - `SDXL usage`
> - `Image generation issues`
> - `ControlNet functionality`
> - `AI model security concerns` 


- **Flux's Resource Intensive Needs**: Members discussed the **resource requirements** for using **Flux** effectively, noting that it requires substantial **VRAM** and can be slow to generate images.
   - One member highlighted that using **Loras** can enhance **Flux's** output for NSFW content, although it is not optimally trained for that.
- **Maximizing SDXL Performance**: For **SDXL**, using the **best practices** such as `--xformers` and `--no-half-vae` in configuration can improve performance on systems with **12GB VRAM**.
   - Members noted that **Pony**, a derivative of **SDXL**, requires special tokens and has limitations in compatibility with **XL Loras**.
- **Image Prompting with SDXL Lightning**: A user inquired about **using image prompts** in **SDXL Lightning** via Python, specifically for inserting a photo into a specific environment.
   - The conversation indicates that combining image prompts with different backgrounds is a topic of interest for enhancing generation capabilities.
- **Addressing Generational Delays**: Frustrations over random **long generation times** while using various models prompted discussions about the potential underlying causes.
   - Members speculated that memory management issues, such as loading resources into **VRAM**, might contribute to these slowdowns.
- **AI Model Utilization and Security**: Concerns over receiving suspicious requests for personal information, like wallet addresses, led members to believe there may be **scammers** in the community.
   - Users were encouraged to report such incidents to maintain a **secure** environment within the group.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/Comfy-Org/stable-diffusion-3.5-fp8">Comfy-Org/stable-diffusion-3.5-fp8 · Hugging Face</a>: no description found</li><li><a href="https://x.com/bfl_ml/status/1859616264324284619?t=DftoDEhtAigmD4sQvsMl2w&s=19">Tweet from Black Forest Labs (@bfl_ml)</a>: Today, we are excited to release FLUX.1 Tools, a suite of models designed to add control and steerability to our base text-to-image model FLUX.1, enabling the modification and re-creation of real and ...</li><li><a href="https://huggingface.co/calcuis/sd3.5-medium-gguf/tree/main">calcuis/sd3.5-medium-gguf at main</a>: no description found
</li>
</ul>

</div>
  

---



### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1309026411364745246)** (5 messages): 

> - `Mamba SSM Layers`
> - `Data Transfer Times`
> - `LLM Autophagy Process`
> - `Evaluation Tasks for Foundational Models`
> - `Meetup in Wellington` 


- **Mamba's SSM Layers Explained**: A member inquired if the SSM layers in the real-valued version of **Mamba** function as exponential moving averages, implicating that *`dA(x) = exp(-f(x))`* where `f` is a function of input **x**.
   - There is a suggestion that this operation essentially computes an efficient EMA update applied elementwise: *`h_t = dA(x) * h_{t - 1} + (1 - dA(x)) * x`*.
- **Training Data Transfer vs Processing Time**: A discussion emerged on whether it's common for the training time on a single batch to be shorter than the **PCI bus** data transfer time.
   - This raises concerns about efficiency in processing large datasets during training workflows.
- **Research on LLM Autophagy Process**: A PhD student introduced their research on the **autophagy process** in **LLMs**, providing a link to their preprint paper: [arXiv preprint](https://arxiv.org/abs/2410.12341).
   - They mentioned utilizing the library for evaluating models in the context of collapse in their upcoming paper.
- **Seeking Evaluation Tasks for Foundational Models**: The same PhD student sought suggestions for interesting evaluation tasks for foundational models, beyond standard ones like **HellaSwag**.
   - This indicates a desire for broader benchmarking for foundational models in the community.
- **Meetup Invitation in Wellington, NZ**: The PhD student announced their upcoming research period starting in **Wellington, NZ**, and expressed interest in meeting local members.
   - They invited anyone residing in Wellington to reach out if interested in a meetup.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1308900335749693440)** (48 messages🔥): 

> - `FlexAttentions`
> - `Position Encoding Techniques`
> - `Forgetting Transformer`
> - `Sparse Upcycling vs Continued Pretraining`
> - `Scale of LLM Training` 


- **FlexAttentions Development Underway**: A member expressed optimism that developing new **FlexAttentions** models could be accomplished relatively easily within a month of focused work.
   - This indicates a growing interest in refining attention mechanisms to enhance model efficiency.
- **Exploring New Position Encoding Methods**: Discussion emerged around position encoding methods, specifically a proposal for **Contextual Position Encoding (CoPE)** that adapts based on token context rather than fixed counts, leading to more expressive models.
   - Members highlighted potential improvements in handling selective tasks like **Flip-Flop** that traditional methods struggle with.
- **Forgetting Transformer Outshines Standard Models**: The **Forgetting Transformer** was introduced as a variant incorporating a forget gate for better performance on long-context tasks, showing improvements over standard architectures.
   - This model does not require position embeddings and maintains effective performance on longer training contexts.
- **Trade-offs in Sparse Upcycling Methodology**: A recent **Databricks** paper evaluates the trade-offs between **sparse upcycling** and continued pretraining for model enhancement, finding sparse upcycling yields better quality but at a cost of increased inference time.
   - This shows a noticeable **40% slowdown** in inference efficiency, emphasizing the challenge of balancing model performance with practical deployment considerations.
- **Concerns Over LLM Training Infrastructure**: Conversations included criticisms of current **network fabric** and **bandwidth** improvements lagging behind processing capabilities, especially highlighted in the context of TPU training setups.
   - The discussion considers how architecture choices like TPU's topology may provide advantages over traditional GPU setups.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2405.18719">Contextual Position Encoding: Learning to Count What&#39;s Important</a>: The attention mechanism is a critical component of Large Language Models (LLMs) that allows tokens in a sequence to interact with each other, but is order-invariant. Incorporating position encoding (P...</li><li><a href="https://arxiv.org/abs/2411.13055">Hardware Scaling Trends and Diminishing Returns in Large-Scale Distributed Training</a>: Dramatic increases in the capabilities of neural network models in recent years are driven by scaling model size, training data, and corresponding computational resources. To develop the exceedingly l...</li><li><a href="https://arxiv.org/abs/2411.08968?">Sparse Upcycling: Inference Inefficient Finetuning</a>: Small, highly trained, open-source large language models are widely used due to their inference efficiency, but further improving their quality remains a challenge. Sparse upcycling is a promising app...</li><li><a href="https://seaborn.pydata.org/generated/seaborn.heatmap.html">seaborn.heatmap &#8212; seaborn 0.13.2 documentation</a>: no description found</li><li><a href="https://x.com/YouJiacheng/status/1859353724713566290">Tweet from YouJiacheng (@YouJiacheng)</a>: @hi_tysam This is a sliding window, information can still propagate if there are &gt;1 layers.</li><li><a href="https://openreview.net/forum?id=q2Lnyegkr8">Forgetting Transformer: Softmax Attention with a Forget Gate</a>: An essential component of modern recurrent sequence models is the forget gate. While Transformers do not have an explicit recurrent form, we show that a forget gate can be naturally incorporated...</li><li><a href="https://cloud.google.com/blog/products/compute/the-worlds-largest-distributed-llm-training-job-on-tpu-v5e">the world’s largest distributed LLM training job on TPU v5e | Google Cloud Blog</a>: We used Multislice Training to run the world’s largest LLM distributed training job on a compute cluster of 50,944 Cloud TPU v5e chips.
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1308914897366552656)** (7 messages): 

> - `Scaling Laws`
> - `Evaluation Predictions`
> - `Marius Hobbhahn's Contributions`
> - `Meta and OpenAI's Methods`
> - `Cost of Scaling Law Training` 


- **Scaling Laws Explained**: A recent [paper](https://arxiv.org/abs/2405.10938) discusses how language model performance varies with scale and offers an observational approach using ~100 publicly available models to develop scaling laws without direct training.
   - This method can highlight variations in training efficiency, proposing a generalized scaling law where performance is dependent on a low-dimensional capability space.
- **Marius Hobbhahn Leads Eval Science**: Marius Hobbhahn at Apollo is noted for publicly advocating for the science of evaluation methodologies in the field of scaling laws.
   - His contributions aim to ground predictions and benchmarking in observable metrics rather than extensive modeling.
- **Predicting Performance Before Training**: The GPT-4 paper predicted its score in the popular HumanEval coding benchmark within a ~1% margin prior to even training, illustrating effective evaluation methodologies.
   - Similarly, the Meta team predicted the Llama-3.1 model's performance on the Abstract Reasoning Corpus before training, employing innovative statistical methods.
- **Methods for Successful Predictions**: The prediction methodology for HumanEval involved plotting Mean Log Pass rate against compute scales, while the Abstract Reasoning Corpus utilized a two-step translation involving negative log likelihood.
   - These methods demonstrated the ability to accurately project model capabilities based on scaling laws derived from existing models.
- **The Cost of Scaling Law Models**: Training scaling law models is costly but significantly less than training full target models, with Meta reportedly spending only 0.1% to 1% of the target model's budget.
   - For instance, to predict capabilities for a $1B model, the cost of training the scaling law models could exceed $1M if following this budget ratio.



**Link mentioned**: <a href="https://arxiv.org/abs/2405.10938">Observational Scaling Laws and the Predictability of Language Model Performance</a>: Understanding how language model performance varies with scale is critical to benchmark and algorithm development. Scaling laws are one approach to building this understanding, but the requirement of ...

  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1308927065319538769)** (39 messages🔥): 

> - `lm-eval and pruned models`
> - `Using Groq API`
> - `Logits and loglikelihood in QA`
> - `Custom metrics in lm-harness` 


- **lm-eval's support for pruned models questioned**: A user inquired if the current version of lm-eval supports zero-shot benchmarking of pruned models, noting issues with old library versions.
   - *They are using WANDA* and reported unreliable zero-shot results, prompting discussion on reading documentation for existing limitations.
- **Successfully interfacing with Groq API**: A user faced issues with unrecognized API key arguments when trying to connect to the Groq API, referencing their documentation for troubleshooting.
   - Another member suggested setting the API key in the `OPENAI_API_KEY` environment variable, which resolved the issue.
- **Extracting loglikelihood in TruthfulQA**: A user asked for advice on obtaining loglikelihood values for answers in the TruthfulQA dataset instead of standard accuracy metrics.
   - This discussion revolved around the standard QA setup and the need for better indicators of an LLM's performance before and after adjustments.
- **Custom metrics for logits in lm-harness**: A user inquired if there's a method to save logits in lm-harness to analyze LLM tendencies toward correct answers.
   - A member suggested creating a custom metric that could manipulate logits as needed for their evaluation purpose.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/">GitHub · Build and ship software on a single, collaborative platform</a>: Join the world&#39;s most widely adopted, AI-powered developer platform where millions of developers, businesses, and the largest open source community build software that advances humanity.</li><li><a href="https://github.com/locuslab/wanda?tab=readme-ov-file#zero-shot-evaluation)">GitHub - locuslab/wanda: A simple and effective LLM pruning approach.</a>: A simple and effective LLM pruning approach. Contribute to locuslab/wanda development by creating an account on GitHub.</li><li><a href="https://console.groq.com/docs/overview">GroqCloud</a>: Experience the fastest inference in the world</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/867413f8677f00f6a817262727cbb041bf36192a/lm_eval/models/anthropic_llms.py#L324)">lm-evaluation-harness/lm_eval/models/anthropic_llms.py at 867413f8677f00f6a817262727cbb041bf36192a · EleutherAI/lm-evaluation-harness</a>: A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1309167267266691092)** (1 messages): 

> - `Multimodal benchmarks`
> - `LLaVA performance`
> - `Text recognition in images` 


- **Seeking Multimodal Benchmark Diversity**: A member inquired about **interesting benchmarks** for multimodal models that go beyond basic tasks like *describing photographs*.
   - They specifically want to explore benchmarks that evaluate the capability of models to recognize and report **text in images**.
- **LLaVA Underperforms Compared to Smaller Models**: The same member noticed that **LLaVA** and its derivatives are performing worse than much **smaller models** on certain tasks.
   - This observation sparked their interest in conducting further tests to understand the performance disparities.


  

---



### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1308908575149658235)** (83 messages🔥🔥): 

> - `Pro Channel Access`
> - `Image Creation on iOS`
> - `Perplexity Pro Features`
> - `Subscription Issues`
> - `Discord Support` 


- **Concerns about Pro Channel Access**: Users expressed concerns about accessing the Pro Channel after purchasing the Pro version, with one stating they initially lacked access despite following the provided link.
   - Others confirmed their access after rejoining the Discord or receiving help from fellow users.
- **Image Creation on iOS**: A user inquired about image creation capabilities within the Perplexity iOS app, noting that another user clarified this feature is available only on iPad.
   - This sparked discussions about the limitations of functionality across different devices.
- **Perplexity Pro Features Highlighted**: Members discussed various features of Perplexity Pro, emphasizing the more advanced models available for Pro users and how it differs from ChatGPT.
   - The discussion included insights on search and tool integration that enhances user experience.
- **Issues with Account and Subscriptions**: Several users reported challenges with their subscriptions, ranging from unsuccessful attempts to redeem codes to difficulties linking accounts.
   - Users were directed to the support email to resolve their specific issues, and discussions ensued about how to manage multiple accounts effectively.
- **Need for Support on Discord**: Questions arose regarding how to receive support on Discord, with members sharing links to help with subscription upgrades and issues accessing roles.
   - The community provided assistance, and some members confirmed successful resolution of their account issues after receiving guidance.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/bocchi-the-rock-bocchi-the-rock-awkward-anime-bocchi-the-rock-bocchi-look-around-anime-awkward-gif-27050898">Bocchi The Rock Bocchi The Rock Awkward GIF - Bocchi The Rock Bocchi The Rock Awkward Anime Bocchi The Rock - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://x.com/apostraphi/status/1859627827160629325?s=46">Tweet from Phi Hoang (@apostraphi)</a>: time well spent if you asked me</li><li><a href="https://www.ispreview.co.uk/index.php/2024/11/virgin-media-o2-uk-offer-free-access-to-ai-search-engine-perplexity-pro.html">Virgin Media O2 UK Offer Free Access to AI Search Engine Perplexity Pro</a>: Customers of Virgin Media and O2’s various broadband, mobile, phone and TV packages may like to know that their ‘Priority’ app, which rewards existing sub
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1308973694734831636)** (7 messages): 

> - `Pokémon Data AI Model`
> - `Baltic Sea Cable Sabotage`
> - `Chicken or Egg Paradox Resolution`
> - `NVIDIA's Omniverse Blueprint`
> - `One-Person Startup Era` 


- **Pokémon Data Sparks New AI Model**: A YouTube video discussed how **Pokémon data** is being utilized to create an **AI model**, offering insights into technology advances in gaming.
   - *This may change the way data is leveraged in AI applications*.
- **Investigating Baltic Sea Sabotage**: A link raised concerns about the **sabotage of cables** in the Baltic Sea, highlighting potential geopolitical tensions.
   - *Further discussions are needed on the implications for digital infrastructure and security*.
- **Chicken or Egg Paradox Potentially Solved**: A recent discussion points towards a possible resolution of the **chicken or egg paradox**, inviting thoughts on scientific evolution.
   - *This could lead to new philosophical inquiries regarding development and existence*.
- **NVIDIA's Game Changing Tech for CAD/CAE**: A member shared insights on **NVIDIA's Omniverse Blueprint**, showcasing its transformative potential for CAD and CAE in design and simulation.
   - *Many are excited about how it integrates advanced technologies into traditional workflows*.
- **Era of One-Person Startups Arises**: Sam Altman provocatively stated, **'Tomorrow's startups will be run by just 1 person and 10,000 GPUs'**, hinting at future entrepreneurship trends.
   - *This notion reflects the evolving landscape of tech startups and resource utilization*.



**Link mentioned**: <a href="https://www.youtube.com/embed/hQhP7ipvgx0">YouTube</a>: no description found

  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1309062556157214750)** (7 messages): 

> - `API Rate Limits`
> - `Using Own API Key in Perplexity`
> - `Session Management in Frontend Apps` 


- **Rate Limit Confusion**: Several members are facing **rate limit issues** with the API, questioning if the limit is set at **50 requests per minute per account** or key.
   - One user has reached out to Perplexity for limit increases but hasn't received feedback yet, expressing urgency due to customer issues.
- **Inquiry on Bring Your Own API Key**: A member asked if it's permissible to **bring your own API key** to build an alternative platform using Perplexity, outlining how data would be managed securely.
   - This approach involves user-supplied keys being encrypted and stored in cookies, prompting questions on compliance with OpenAI standards.
- **Simplifying Technical Concepts**: In response to a request for simplification, a user explained **session management** in web applications by comparing it to cookies storing session IDs.
   - The conversation highlighted how users' authentication relies on checking valid sessions without storing sensitive data directly.



**Link mentioned**: <a href="https://docs.perplexity.ai/guides/rate-limits">no title found</a>: no description found

  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1308905759085432873)** (59 messages🔥🔥): 

> - `Truffles hardware device`
> - `Vercel acquires Grep`
> - `Tülu 3 model release`
> - `Flux Tools from Black Forest Labs`
> - `Gemini API model updates` 


- **Truffles Device Gains Attention**: Members recalled the **Truffles** device, described as a 'white cloudy semi-translucent thing' that allows self-hosting of LLMs [Truffles](https://x.com/itsalltruffles). One member humorously referred to it as the 'glowing breast implant.'
- **Vercel Acquires Grep for Code Search**: Vercel announced the acquisition of [Grep](https://grep.app/) to enhance developer tools for searching code across over 500,000 public repositories. Dan Fox, the founder, will join Vercel's AI team to continue developing this capability.
- **Tülu 3 Outperforms Llama 3**: [Tülu 3](https://allenai.org/papers/tulu-3-report.pdf), a new model developed over two years, reportedly outperforms **Llama 3.1 Instruct** on specific tasks, boasting new SFT data and optimization techniques. The project lead expressed excitement about their achievement in the field of RLHF.
- **New Features in Flux Tools**: Black Forest Labs released **Flux Tools**, which includes features like inpainting and outpainting for image manipulation. The suite aims to add steerability to their text-to-image model, and users are encouraged to explore running it on [Replicate](https://replicate.com/black-forest-labs).
- **Updates to Google Gemini Models**: New experimental models from **Gemini** were released, focusing on improvements in coding capabilities. Users are directed to the [Gemini API documentation](https://ai.google.dev/gemini-api/docs/models/experimental-models) for details.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://vercel.com/blog/vercel-acquires-grep">Vercel acquires Grep to accelerate code search - Vercel</a>: Announcing the acquisition of Grep to further our mission of helping developers work and ship faster. </li><li><a href="https://x.com/natolambert/status/1859643351441535345">Tweet from Nathan Lambert (@natolambert)</a>: I&#39;ve spent the last two years scouring all available resources on RLHF specifically and post training broadly. Today, with the help of a totally cracked team, we bring you the fruits of that labor...</li><li><a href="https://prannaya.notion.site/Existing-AI-Code-Tools-14105a6b76a480f8bf3af4dae8ee1084?pvs=4">Notion – The all-in-one workspace for your notes, tasks, wikis, and databases.</a>: A new tool that blends your everyday work apps into one. It's the all-in-one workspace for you and your team</li><li><a href="https://x.com/ExaAILabs/status/1859306370010579010">Tweet from Exa (@ExaAILabs)</a>: We&#39;re celebrating Thanksgiving with a full week of launches 🦃  Today - semantic search over LinkedIn.  Pick the &#34;LinkedIn profile&#34; category to intelligently search over hundreds of millio...</li><li><a href="https://x.com/itsalltruffles">Tweet from undefined</a>: no description found</li><li><a href="https://blog.dottxt.co/say-what-you-mean.html">Say What You Mean: A Response to 'Let Me Speak Freely'</a>: no description found</li><li><a href="https://x.com/replicate/status/1859616730915721249?s=46">Tweet from Replicate (@replicate)</a>: Black Forest Labs just dropped Flux Tools, for pro and open-source dev models:  - Fill: Inpainting and outpainting - Redux: For image variations - Canny and depth controlnets  They&#39;re all really g...</li><li><a href="https://x.com/bfl_ml/status/1859616264324284619?s=46">Tweet from Black Forest Labs (@bfl_ml)</a>: Today, we are excited to release FLUX.1 Tools, a suite of models designed to add control and steerability to our base text-to-image model FLUX.1, enabling the modification and re-creation of real and ...</li><li><a href="https://ai.google.dev/gemini-api/docs/models/experimental-models">no title found</a>: no description found</li><li><a href="https://x.com/markokraemer/status/1859526870867263906">Tweet from markokraemer (@markokraemer)</a>: v0 vs bolt vs loveable vs softgen  3 Prompts 1. &#34;Make this a fancy landing page about protein shakes&#34; 2. &#34;Make it pop&#34; 3. &#34;Add more sections&#34;  FYI I was drinking a protein shak...</li><li><a href="https://youtu.be/LPZh9BOjkQs?si=Jyqqr-NGyt3dXwlz">Large Language Models explained briefly</a>: Dig deeper here: https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3piTechnical details as a talk: https://youtu.be/KJtZARuO3JYMade for an...</li><li><a href="https://x.com/karpathy/status/1859305141385691508?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Andrej Karpathy (@karpathy)</a>: Remember the llm.c repro of the GPT-2 (124M) training run? It took 45 min on 8xH100. Since then, @kellerjordan0 (and by now many others) have iterated on that extensively in the new modded-nanogpt rep...</li><li><a href="https://github.com/KellerJordan/modded-nanogpt">GitHub - KellerJordan/modded-nanogpt: NanoGPT (124M) in 5 minutes</a>: NanoGPT (124M) in 5 minutes. Contribute to KellerJordan/modded-nanogpt development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1308963337853472818)** (29 messages🔥): 

> - `AI Expert Needed Urgently`
> - `DeepSeek R1-Lite Specifications`
> - `Llama-Mesh Paper Recommendation`
> - `Daily LLM Drop Discussions`
> - `User Interaction Memory in AI` 


- **AI Expert Needed Urgently**: A member urgently called for an AI expert, prompting various responses about the specific type of assistance required, from deployment issues to architectural support.
   - Members humorously suggested contacting top experts, while one proposed that the urgent situation remained vague, highlighting the need for clarity.
- **DeepSeek R1-Lite Specifications**: It was rumored that **DeepSeek R1-Lite** is a **16B MOE** model with **2.4B active parameters**, drastically improving MATH scores from **17.1 to 91.6**.
   - The rumor quoted a WeChat announcement and was met with skepticism, with a member expressing disbelief at the potential performance improvement.
- **Llama-Mesh Paper Recommendation**: A member recommended reviewing the **llama-mesh paper**, describing it as 'pretty good' to the group.
   - This appeal for others to engage with the content was noted amidst a broader conversation about AI discussions.
- **Daily LLM Drop Discussions**: Members discussed the concept of a 'daily LLM drop,' referencing the evolving landscape of language models with a hint of fatigue.
   - One member humorously remarked that the **Goodhart arena is getting old**, indicating a sentiment towards frequent changes in the field.
- **User Interaction Memory in AI**: A member inquired whether AI systems remember prior interactions on platforms like Twitter, especially concerning engaging with the same accounts.
   - This inquiry reflects broader curiosity about user memory capabilities of AI, pertinent to various user experiences and expectations.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/nrehiew_/status/1859265550767067518">Tweet from wh (@nrehiew_)</a>: Rumor is that DeepSeek R1-Lite is a 16B MOE with 2.4B active params  if true, their MATH scores went from 17.1 -&gt; 91.6  Quoting Phil (@phill__1)   @nrehiew_ From their wechat announcement:</li><li><a href="https://github.com/Lesterpaintstheworld/terminal-velocity">GitHub - Lesterpaintstheworld/terminal-velocity: A novel created autonomously by 10 teams of 10 AI agents</a>: A novel created autonomously by 10 teams of 10 AI agents - Lesterpaintstheworld/terminal-velocity
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1309052580902731807)** (6 messages): 

> - `LLM Performance Improvement`
> - `KV Cache Limitations`
> - `Multi-Agent Frameworks`
> - `Prefix Caching`
> - `Prompt Caching` 


- **Multi-Agent Frameworks and Hidden Information**: A member raised concerns that using de-tokenized output in multi-agent frameworks, like 'AI entrepreneurs' and 'AI software engineers', may lead to **loss of hidden information** due to the discarded kv cache.
   - They suggested that this loss might explain the **limited output diversity** observed in such frameworks.
- **Prefix Caching vs. KV Cache**: In response, another member questioned whether prefix caching serves a similar purpose to kv caching during inference.
   - The discussion revealed that prefix caching was not previously available in APIs, which contributed to the limitations experienced in early agent frameworks.
- **Misunderstandings About Caching**: Conversations turned to misunderstandings regarding the equivalence of prompt caching to other caching techniques.
   - Acknowledgments about these oversights suggested a continuous learning curve within the community.


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1308969139716620289)** (4 messages): 

> - `Soft Prompts vs Fine Tuning`
> - `CoPilot Arena Results`
> - `LoRA Trade-offs` 


- **Soft Prompts Struggle for Adoption**: The discussion highlighted that **soft prompts** are often overshadowed by techniques like **fine tuning** and **LoRA**, which are generally seen as more effective for open source use cases. Despite some unique advantages, soft prompts exhibit limited generalizability and are not widely utilized in current practices.
   - Participants noted that using soft prompts may involve trade-offs, particularly in performance and optimization.
- **The Two Potential Uses of Soft Prompts**: A member suggested that soft prompts could serve two main purposes: **system prompt compression** and **enhancing LoRA/full SFT** applications. They mentioned that this strategy could optimize model parameters without heavily relying on the inference system.
   - The implications of these uses include potential risks of overfitting, indicating the need for careful implementation.
- **CoPilot Arena Initial Results Released**: The first results of the **CoPilot Arena**, showcased on [LMarena's blog](https://blog.lmarena.ai/blog/2024/copilot-arena/#initial-leaderboard-and-results), reveal a surprisingly close field among participants. However, it was noted that the analysis only considered an older version of **Sonnet**.
   - This sparked curiosity about the implications of using outdated models in competitive settings and how it might affect participant comparisons.


  

---



### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1309172096328531980)** (20 messages🔥): 

> - `Debugging Triton Interpreter`
> - `Block Size Discussion`
> - `Triton GEMM and Bank Conflicts`
> - `Boolean Mask Summation Bug`
> - `Swizzling Techniques for Performance` 


- **Debugging Triton Interpreter for Kernel Accuracy**: A user sought advice on debugging a **kernel accuracy** issue that is not reproducible with the **Triton interpreter**, especially considering that TF32 is off for matmuls.
   - Suggestions included manually casting tensors to `tl.float32` and checking data compatibility with specific block sizes.
- **Strange Block Size Behavior Observed**: There were reports that accuracy issues arise in **Triton** depending on the **block size**, functioning well for sizes <= 64 but problematic for sizes >= 128.
   - Discussion arose regarding ensuring input shapes are correct and valid by potentially pruning configurations to avoid loading shape conflicts.
- **Triton GEMM's Bank Conflict Solutions**: A user was surprised at the **Triton GEMM** on ROCm being conflict-free and asked about the application of **swizzling** to avoid bank conflicts.
   - While references to block-level swizzling were shared, the focus remained on further exploring methods specific to bank conflict resolution.
- **Bug Found in Boolean Mask Summation**: After extensive printing and debugging, a user discovered a bug in **Triton** related to summing **bool masks** converted to **int8** tensors.
   - Switching from summation to max reduction resolved the issue, and there was a suggestion to draft a minimal example for the repo.
- **Fun Farewells and Ongoing Questions**: As discussions wrapped up, a member remarked it was time for bed while another expressed ongoing curiosity regarding AMD support.
   - The conversation concluded with a commitment to follow up on the technical puzzles presented.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://triton-lang.org/main/python-api/generated/triton.language.swizzle2d.html">triton.language.swizzle2d &mdash; Triton  documentation</a>: no description found</li><li><a href="https://github.com/mobiusml/gemlite/blob/master/gemlite/triton_kernels/utils.py#L6-L14">gemlite/gemlite/triton_kernels/utils.py at master · mobiusml/gemlite</a>: Simple and fast low-bit matmul kernels in CUDA / Triton - mobiusml/gemlite
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1309179398343102544)** (2 messages): 

> - `cuBLAS operations`
> - `Matrix Multiplication with cuBLAS`
> - `Row-major vs Column-major order` 


- **cuBLAS's Column-Major Order Dilemma**: A user highlighted the challenge of using `cublasSgemm` which operates in **column-major** order, questioning the preference for calling it with `CUBLAS_OP_N` or `CUBLAS_OP_T` for better clarity.
   - *Explicit clarity* on transpose operations may lead to *confusion* when working with matrices defined in a **row-major** order, as the output will always be in column-major.
- **Stack Overflow Insights on cuBLAS**: A member shared a relevant [Stack Overflow post](https://stackoverflow.com/questions/56043539/cublassgemm-row-major-multiplication) which clarifies using `cublasSgemm` for non-square matrices stored in **row-major** order.
   - The post discusses limitations around using **CUBLAS_OP_T** when multiplying non-square matrices and highlights potential *conflicts with non-transposed results*.
- **Complexities of Matrix Declarations**: There is mention that modifying matrices to declare them in **column-order** using the `IDX2C` macro isn’t feasible due to them being set in another program.
   - This indicates the prevalent issues faced when adapting existing codebases to fit the constraints of cuBLAS libraries.
- **Program Limitations with cuBLAS**: The user's existing code cannot multiply non-square matrices with the transpose parameter, limiting its versatility in **matrix operations**.
   - This raises questions regarding the flexibility and usability of cuBLAS for diverse matrix sizes and orientations.



**Link mentioned**: <a href="https://stackoverflow.com/questions/56043539/cublassgemm-row-major-multiplication">cublasSgemm row-major multiplication</a>: I&#x27;m trying to use cublasSgemm to multiplicity two non-square matrices that are stored in row-major order. I know that this function has one parameter where you can specify that if you want to tra...

  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1308994672726900789)** (1 messages): 

> - `Llama-2-70B model`
> - `Multi-GPU support` 


- **Questioning Multi-Card Support for Llama-2-70B**: One member inquired whether `llama/generate.py` supports the **Llama-2-70B** model to utilize multiple cards, specifically mentioning **2 A100s**.
   - The discussion focused on the capabilities of the script in handling GPU resources effectively.
- **Exploring GPU Utilization Strategies**: Another point raised was about how **GPU utilization** can be optimized when running heavy models like Llama-2-70B on multiple cards.
   - Members discussed potential strategies to maximize throughput and reduce bottlenecks.


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1309078115754184774)** (11 messages🔥): 

> - `HIP Kernel Rules`
> - `Compilation Time for Examples`
> - `FP16 GEMM on MI250 GPU`
> - `Debugging Kernel`
> - `Triton GEMM on ROCm` 


- **HIP Kernel Rules still confusing**: A user expressed difficulty in understanding and reproducing rules within a **pure HIP kernel**.
   - They noted ongoing challenges even after multiple attempts.
- **Compilation time for examples frustrating**: The time taken to compile a simple example was reported to be **1-2 hours** on a user’s machine using `make` commands.
   - Despite trying to adjust the `-j` flag in compilation, it did not significantly improve the performance.
- **Confusion over input shape in FP16 GEMM**: A user analyzed the transformation description for an **FP16 GEMM (v3)** on an MI250 GPU, noting a mismatch in the input shape for transformations.
   - They requested clarification on the rationale behind the shared memory and input shapes.
- **Debugging slows down compilation**: Inserting print functions within the kernel increases compilation time due to many static operations in CK, which unroll multiple times.
   - A suggestion was made to reduce tile size to improve debugging performance.
- **Triton GEMM on ROCm found conflict free**: A user was surprised to find that **Triton GEMM** on ROCm also exhibits **conflict-free** properties.
   - This insight could open discussions about optimization strategies in ROCm.


  

---


### **GPU MODE ▷ #[webgpu](https://discord.com/channels/1189498204333543425/1262121239044948009/1308917390519832667)** (1 messages): 

> - `AGX machine code`
> - `Freedesktop tools` 


- **Freedesktop Developers Disassemble AGX Machine Code**: Freedesktop developers maintain tools to disassemble **AGX machine code** available on their [GitHub repository](https://github.com/dougallj/applegpu).
   - *Disassembling machine code can aid in better understanding and optimization of software performance.*
- **Tool advantages and user discussions**: Several members discussed the advantages of using **Freedesktop's disassembling tools** over other methods for analyzing machine code. They highlighted how these tools can streamline the debugging process and reduce development time.


  

---


### **GPU MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/)** (1 messages): 

0x000ff4: A little update on kto I am working now on the tests
  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/)** (1 messages): 

pradeep1148: https://www.youtube.com/watch?v=XP33Vgn75lM
  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1309214054442598562)** (1 messages): 

> - `Post-Training Techniques`
> - `Human Preferences in RL`
> - `Continual Learning`
> - `Constitutional AI`
> - `Recursive Summarization` 


- **Survey on Tulu 3 Released**: A new survey paper titled [Tulu 3](https://allenai.org/papers/tulu-3-report.pdf) has been recommended for understanding post-training methods.
   - This paper serves as a comprehensive overview for anyone interested in the advancements in the field.
- **Harnessing Human Preferences for RL**: The paper [Deep RL from human preferences](https://arxiv.org/abs/1706.03741) explores defining complex goals through human preferences in RL tasks, demonstrating effective results in games and robot locomotion.
   - It emphasizes that only **one percent** of the agent's interactions required human feedback, significantly reducing oversight costs.
- **Continual Learning Insights**: The article [Comprehensive survey of continual learning](https://arxiv.org/abs/2302.00487) delves into challenges like catastrophic forgetting and the methods to overcome them.
   - It provides a thorough bridge between foundational theories and practical applications in continual learning.
- **Constitutional AI Explored**: In the paper [Constitutional AI](https://arxiv.org/abs/2212.08073), the authors examine structuring AI systems based on robust guiding principles.
   - It includes contributions from notable researchers in the field, ensuring diverse perspectives on the topic.
- **Advancing Summarization with Humans**: The research titled [Recursively summarizing books with human feedback](https://arxiv.org/abs/2109.10862) tackles the challenge of summarizing entire novels using human feedback and recursive decomposition.
   - This model allows quick supervision and evaluation by humans, leading to efficient and sensible summarizations.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/1706.03741">Deep reinforcement learning from human preferences</a>: For sophisticated reinforcement learning (RL) systems to interact usefully with real-world environments, we need to communicate complex goals to these systems. In this work, we explore goals defined i...</li><li><a href="https://arxiv.org/abs/2009.01325">Learning to summarize from human feedback</a>: As language models become more powerful, training and evaluation are increasingly bottlenecked by the data and metrics used for a particular task. For example, summarization models are often trained t...</li><li><a href="https://arxiv.org/abs/2203.02155">Training language models to follow instructions with human feedback</a>: Making language models bigger does not inherently make them better at following a user&#39;s intent. For example, large language models can generate outputs that are untruthful, toxic, or simply not h...</li><li><a href="https://arxiv.org/abs/2307.15217">Open Problems and Fundamental Limitations of Reinforcement Learning from Human Feedback</a>: Reinforcement learning from human feedback (RLHF) is a technique for training AI systems to align with human goals. RLHF has emerged as the central method used to finetune state-of-the-art large langu...</li><li><a href="https://arxiv.org/abs/2302.00487">A Comprehensive Survey of Continual Learning: Theory, Method and Application</a>: To cope with real-world dynamics, an intelligent system needs to incrementally acquire, update, accumulate, and exploit knowledge throughout its lifetime. This ability, known as continual learning, pr...</li><li><a href="https://arxiv.org/abs/2212.08073">Constitutional AI: Harmlessness from AI Feedback</a>: As AI systems become more capable, we would like to enlist their help to supervise other AIs. We experiment with methods for training a harmless AI assistant through self-improvement, without any huma...</li><li><a href="https://arxiv.org/abs/2305.18290">Direct Preference Optimization: Your Language Model is Secretly a Reward Model</a>: While large-scale unsupervised language models (LMs) learn broad world knowledge and some reasoning skills, achieving precise control of their behavior is difficult due to the completely unsupervised ...</li><li><a href="https://arxiv.org/abs/2210.10760">Scaling Laws for Reward Model Overoptimization</a>: In reinforcement learning from human feedback, it is common to optimize against a reward model trained to predict human preferences. Because the reward model is an imperfect proxy, optimizing its valu...</li><li><a href="https://arxiv.org/abs/2109.10862">Recursively Summarizing Books with Human Feedback</a>: A major challenge for scaling machine learning is training models to perform tasks that are very difficult or time-consuming for humans to evaluate. We present progress on this problem on the task of ...
</li>
</ul>

</div>
  

---



### **Notebook LM Discord ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1309003599489007686)** (9 messages🔥): 

> - `NotebookLM and GitHub Repositories`
> - `Audio Prompt Generation`
> - `Using Multiple LLMs`
> - `Table of Contents in Code`
> - `ElevenLabs and Text-to-Speech AI` 


- **NotebookLM struggles to traverse GitHub repos**: A user attempted to let NotebookLM traverse a GitHub repo by inputting the repo homepage but found it ineffective. Another member noted that NotebookLM lacks the ability to traverse websites, complicating the request.
   - They suggested using Markdown of the site or printing the page to convert it into a PDF for better processing.
- **Generating impactful audio prompts**: A user suggested providing NotebookLM with specific prompts to produce impactful audio outputs that are useful for explanations. This can help others gain a better understanding of designated topics.
   - Such a strategy aims to enhance the learning experience through clearer audio content.
- **Multiple LLMs for specific tasks**: A member shared their workflow where they utilize multiple LLMs depending on their needs and complimented NotebookLM for certain generations. They previously wrote about this approach in a blog post.
   - This strategy highlights the versatility and effectiveness of leveraging various AI tools to accomplish conversational-based projects.
- **Table of contents usage in code**: The use of a table of contents in code was mentioned as a particularly interesting feature, noting it describes each section with line numbers. This improves navigation and understanding of complex codebases.
   - Members expressed enthusiasm about this feature's utility in coding practices.
- **ElevenLabs as a leading text-to-speech AI**: Discussion around ElevenLabs noted its superiority in the text-to-speech AI space, surpassing competitors like RLS and Tortoise. The member recalled their early experiences with the startup prior to its funding, recognizing the potential for innovation.
   - They emphasized the significant impact of ElevenLabs on creating faceless videos and voice synthesis, marking it as a game-changing tool in the industry.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tokenwisdom.ghost.io/featured/im-an-elevenlabs-pro/">I&#x27;m an ElevenLabs Pro!</a>: Discover the revolution in voice synthesis with our deep dive into ElevenLabs: groundbreaking technology for animation and performance capture. Sample the magic ✨</li><li><a href="https://open.spotify.com/playlist/4hcmaPIiwgHd2rm4SJCjgJ?si=ZNPffZZtSn2fjNiBJMuFJw">&#x27;Songs We Sing&#x27; Podcast Companion</a>: Playlist · MrBland · 11 items
</li>
</ul>

</div>
  

---


### **Notebook LM Discord ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1308960321255379015)** (25 messages🔥): 

> - `Jensen Huang's shoutout`
> - `Podcast generation issues`
> - `Accent preferences`
> - `Functionality requests` 


- **Jensen Huang gives a shoutout to NotebookLM**: During the NVIDIA earnings call, **Jensen Huang** mentioned using **NotebookLM** to load numerous documents and listen to generated podcasts.
   - This highlights the application's versatility in handling content for users.
- **Podcast generation is experiencing errors**: Multiple users reported that **podcast generation** is stuck and resulting in errors, with some needing to wait extended periods for outputs.
   - One user shared that after a two-hour wait, the service suddenly began working again, suggesting potential server issues.
- **Inquiries about changing podcast host accents**: A user inquired whether it’s possible to change the accents of podcast hosts, expressing preference for a **British accent** over an American one.
   - Currently, there are no options available for this feature.
- **Desired features and functionality in NotebookLM**: Several users requested features such as the ability to limit audio duration to 3 minutes and translate audio within **NotebookLM**.
   - Additionally, there were questions about the possibility of an API and adjustments to host script deliveries for more natural conversations.
- **Noted instability and safety flags**: Users have noted an increase in **safety flags** and possible instability with the application, leading to restricted functionalities in tasks.
   - One user suggested DMing examples for debugging, while another noted transient issues likely due to ongoing improvements.



**Link mentioned**: <a href="https://www.reddit.com/r/notebooklm/comments/1gw453m/podcast_generation_not_working/">Reddit - Dive into anything</a>: no description found

  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1308913925383000177)** (2 messages): 

> - `AI agents architecture`
> - `Data-backed systems with Redis`
> - `Knowledge graph construction`
> - `Natural language querying`
> - `Memgraph integration` 


- **Build AI agents with LlamaIndex and Redis**: Join our webinar on December 12 to learn how to architect an agentic system to break down **complex tasks** using [LlamaIndex](https://twitter.com/llama_index/status/1859354663793066029) and **Redis**.
   - Discover best practices for reducing **costs** and optimizing **latency**, along with insights on **semantic caching**.
- **Transform data into a knowledge graph with Memgraph**: Learn how to set up **Memgraph** and integrate it with [LlamaIndex](https://twitter.com/llama_index/status/1859658719082041802) to build a **knowledge graph** from unstructured text data.
   - Participants will explore **natural language querying** of their graph and methods to **visualize connections**.


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1308912110281166989)** (16 messages🔥): 

> - `LlamaParse for PDF table extraction`
> - `Create-Llama frontend options`
> - `Llama-Agents deprecation`
> - `NDCG calculation query`
> - `vLLM error and usage` 


- **LlamaParse offers PDF data extraction**: A member recommended using [LlamaParse](https://github.com/run-llama/llama_parse) for extracting table data from PDF files, stating it could parse files effectively for optimal RAG.
   - They included an informative [GitHub link](https://github.com/run-llama/llama_parse) about its capabilities.
- **Create-Llama frontend confusion**: A user inquired about the best channel for assistance with Create-Llama, particularly regarding the lack of a Next.js frontend option in newer versions when selecting the Express framework.
   - Another participant confirmed that they could post queries directly in the channel and would receive team support.
- **Llama-Agents phased out in favor of Llama-Deploy**: A member noted the issue of dependencies while upgrading to Llama-index 0.11.20 and suggested that **llama-agents** has been deprecated in favor of [llama_deploy](https://github.com/run-llama/llama_deploy).
   - They provided a link to the [Llama Deploy GitHub page](https://github.com/run-llama/llama_deploy) for further context.
- **Discussion on NDCG calculation**: A member raised a question about potentially incorrect code in the NDCG calculation within the [metrics.py file](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/evaluation/retrieval/metrics.py#L380) of llama-index-core.
   - They proposed that the line should use `len(expected_ids)` for maximum achievable DCG, inviting feedback on their interpretation.
- **Help with vLLM integration**: A user reported encountering a KeyError related to the vLLM integration, particularly missing the 'text' key when trying to use VllmServer.
   - Another user suggested launching vLLM in OpenAI API mode with `OpenAILike`, providing a sample code snippet to help.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/evaluation/retrieval/metrics.py#L380">llama_index/llama-index-core/llama_index/core/evaluation/retrieval/metrics.py at main · run-llama/llama_index</a>: LlamaIndex is a data framework for your LLM applications - run-llama/llama_index</li><li><a href="https://github.com/run-llama/llama_deploy">GitHub - run-llama/llama_deploy: Deploy your agentic worfklows to production</a>: Deploy your agentic worfklows to production. Contribute to run-llama/llama_deploy development by creating an account on GitHub.</li><li><a href="https://github.com/run-llama/llama_parse">GitHub - run-llama/llama_parse: Parse files for optimal RAG</a>: Parse files for optimal RAG. Contribute to run-llama/llama_parse development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1308930713294344213)** (8 messages🔥): 

> - `30 Days of Python`
> - `Capstone Project API`
> - `Learning Resources` 


- **Engaging with the 30 Days of Python Challenge**: A member shared their participation in the **30 Days of Python** challenge, which emphasizes step-by-step learning.
   - They are utilizing the [GitHub repository](https://github.com/Asabeneh/30-Days-Of-Python) for resources and inspiration throughout this endeavor.
- **Discussion on Capstone Project Tech Choices**: One member expressed a preference for using **Go** for their capstone project, focusing on developing an API.
   - This choice reflects the excitement for exploring different programming languages in practical applications.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/hi-hello-greet-cute-puppy-gif-14845557723311629962">Hi Hello GIF - Hi Hello Greet - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/Asabeneh/30-Days-Of-Python">GitHub - Asabeneh/30-Days-Of-Python: 30 days of Python programming challenge is a step-by-step guide to learn the Python programming language in 30 days. This challenge may take more than100 days, follow your own pace.  These videos may help too: https://www.youtube.com/channel/UC7PNRuno1rzYPb1xLa4yktw</a>: 30 days of Python programming challenge is a step-by-step guide to learn the Python programming language in 30 days. This challenge may take more than100 days, follow your own pace.  These videos m...
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1309129477443555379)** (3 messages): 

> - `Cohere Repository`
> - `Cohere Toolkit`
> - `Jupyter Notebooks`
> - `Contribution Guidelines` 


- **Explore the Cohere Repository**: A member highlighted the **Cohere GitHub repository** ([GitHub link](https://github.com/cohere-ai)) as a great starting point for contributors, showcasing various projects.
   - They encouraged exploration of tools available in the repository and to share feedback or new ideas within each project.
- **Cohere Toolkit for RAG Applications**: The **Cohere Toolkit** ([GitHub link](https://github.com/cohere-ai/cohere-toolkit)) was mentioned as an advanced UI designed specifically for **RAG applications**, allowing quick build and deployment.
   - It's a collection of prebuilt components aimed at enhancing user productivity.
- **Starter Code Available in Notebooks**: Members were directed to the **notebooks** repository ([GitHub link](https://github.com/cohere-ai/notebooks)), which contains starter code for various use cases.
   - These Jupyter notebooks offer practical examples aimed at assisting users in getting acquainted with the **Cohere Platform**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/cohere-ai">cohere.ai</a>: cohere.ai has 46 repositories available. Follow their code on GitHub.</li><li><a href="https://github.com/cohere-ai/cohere-toolkit">GitHub - cohere-ai/cohere-toolkit: Cohere Toolkit is a collection of prebuilt components enabling users to quickly build and deploy RAG applications.</a>: Cohere Toolkit is a collection of prebuilt components enabling users to quickly build and deploy RAG applications. - cohere-ai/cohere-toolkit</li><li><a href="https://github.com/cohere-ai/notebooks">GitHub - cohere-ai/notebooks: Code examples and jupyter notebooks for the Cohere Platform</a>: Code examples and jupyter notebooks for the Cohere Platform - cohere-ai/notebooks
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1309132867699478528)** (5 messages): 

> - `Multimodal Embeddings Launch`
> - `Research Agent Use Case`
> - `Rate Limit Concerns` 


- **Multimodal Embeddings Launch Set for Next Year**: Exciting news as improvement with **multimodal embed** has been noted, and its launch is scheduled for early next year on **Bedrock** and partner platforms.
   - *A team member will flag the rate limit issue* for further discussion.
- **Innovative Research Agent Using Cohere Technology**: A member created a [Research Agent](https://researcher.customgpt.ai/) that researches topics for 30 mins and utilizes **Cohere's multimodal embeddings** to select relevant images.
   - This tool is gaining traction, but **rate limits** hinder its ability to produce articles efficiently, with only **1 article every 3 minutes**.
- **Support Ticket for Rate Limit Increase Submitted**: The same member submitted a support ticket requesting a **rate limit increase** to enhance the Research Agent's performance.
   - They expressed that this could serve as a **case study** for the effective use of multimodal embeddings if the product marketing team is interested.



**Link mentioned**: <a href="https://researcher.customgpt.ai/)">CustomGPT.AI Researcher - Create High-Quality AI Content Based On Deep Research</a>: Create ultra high-quality, brand-safe articles and research reports using CustomGPT.ai Researcher. Perfect for content marketing, SEO and research reports.

  

---


### **Cohere ▷ #[projects](https://discord.com/channels/954421988141711382/1218409701339828245/)** (1 messages): 

rachel_47358: https://github.com/harmonydata/harmony
  

---



### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1309039506107666433)** (6 messages): 

> - `Mojo Async Progress`
> - `Mojo Community Channel`
> - `Async Runtime Overhead` 


- **Mojo's async feature still under development**: Members noted that Mojo's async capabilities are still a **work in progress**, with no actual async functions available yet.
   - The compiler currently converts sync code into async, leading to a synchronous execution of the code during async calls.
- **Discussion on Mojo Community Channel**: A community channel has emerged for members to connect and communicate, accessible at [mojo-community](https://prefix.dev/channels/mojo-community).
   - This channel has been identified as a hub for ongoing discussions related to Mojo.
- **Concerns about async runtime overhead**: A member raised a concern whether Mojo's async runtime incurs overhead even when no async code is running, citing confusion over how async compiles down to a state machine.
   - Discussions continued about the necessity of explicitly running async functions in an event loop, as well as the implications of the async runtime in Mojo.


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1308916166185455768)** (9 messages🔥): 

> - `Moonshine ASR Model Performance`
> - `Mojo Script Optimization`
> - `CPU Utilization Observations` 


- **Moonshine ASR Model shows speed challenges**: The **Moonshine ASR model** was tested using both Mojo and Python versions via Max, yielding an execution time of **82ms** for **10s** of speech, while a direct ONNX version managed **46ms**.
   - This indicates a **1.8x slowdown** in the Mojo and Python versions compared to the more optimized ONNX runtime.
- **Mojo script experiences crashes**: When writing the **Mojo version**, there were issues with **Model.execute** causing crashes when passing `TensorMap`, and a manual argument listing was necessary due to the lack of unpacking support.
   - These hurdles highlight the script's unidiomatic nature and the author's desire for performance tips to enhance their Mojo skills.
- **Optimization suggestions for Mojo**: A member suggested that optimizing the **tokens list** by pre-allocating capacity could prevent frequent malloc invocations, potentially enhancing performance.
   - Considerations included implementing an **InlineArray** for stack storage if the maximum length permits, aiming to streamline execution.
- **Performance remains unchanged without tokens**: After optimizing, tokens were completely removed from the Mojo code, leaving only the graph execution benchmark, but this change did not significantly impact performance.
   - The author is still exploring avenues for improving the efficiency of their ASR model execution.
- **CPU utilization issues with the model**: One user observed chaotic CPU usage while running the model, noticing it wasn't fully utilizing their CPU capabilities and ignoring hyperthreads.
   - This lack of parallelism in running the model suggests further optimization might be necessary to fully leverage available resources.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://gist.github.com/keveman/ea167957fb6364470cb265c5d9aa9da1">moonshine.mojo</a>: moonshine.mojo. GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://gist.github.com/keveman/d2aea1a059c9a14972783ede2d6b6862">moonshine.py</a>: moonshine.py. GitHub Gist: instantly share code, notes, and snippets.
</li>
</ul>

</div>
  

---



### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1309124519134494720)** (9 messages🔥): 

> - `New guidelines for Torchtune contributors`
> - `Extender packages for Torchtune`
> - `Binary search method suggestion`
> - `Hands-on experience with UV`
> - `Optional packages feature for TorchAO` 


- **Clearer guidelines coming for Torchtune**: Members anticipate some clearer guidelines soon to assist maintainers and contributors in understanding desired features for **Torchtune**.
   - These improvements may help determine when to use forks versus example repos for demonstrations.
- **Suggestion for extender packages in Torchtune**: One member proposed using extender packages like **torchtune[simpo]** or **torchtune[rlhf]**, advocating for simplifying package inclusion without excessive checks.
   - This approach aims to reduce complexity and manage resource concerns effectively.
- **Binary search method for max_global_bsz**: A member recommended utilizing a last success binary search for **max_global_bsz**, defaulting to a power of 2 smaller than the dataset.
   - This method would also incorporate **max_iterations** as parameters for enhanced efficiency.
- **Discussion around using UV**: A member inquired if others have experience using **UV**, expressing interest in opinions regarding its usability.
   - Another member partially validated its utility, noting it looks appealing and modern.
- **Optional packages may resolve TorchAO issues**: There was a query about whether the optional packages feature would solve the issue of users needing to manually download **TorchAO**.
   - Responses indicated that while it may help, there are additional considerations to address.


  

---



### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1309193528953274408)** (7 messages): 

> - `Prompt Signature Modification`
> - `Adapter Configuration`
> - `Optimization across Models` 


- **Exploring Prompt Signature Modification**: A member inquired about the best way to override or change the prompt signature format for debugging purposes, particularly to avoid parseable JSON schema notes.
   - The discussion revolved around methods to achieve this such as building an adapter.
- **Adapter Configuration in DSPy**: A user suggested building an adapter and configuring it using `dspy.configure(adapter=YourAdapter())` for prompt modifications.
   - They also pointed towards the existing adapters in the `dspy/adapters/` directory for further clarification.
- **Optimization of Phrases for Specific Cases**: In response to questions about the tuning of phrases for specific types like bool, int, and JSON, it was clarified that these phrases are based on a maintained set of model signatures.
   - *These phrases are not highly dependent on the individual language models overall,* indicating a generalized approach to their formulation.


  

---



### **LLM Agents (Berkeley MOOC) ▷ #[hackathon-announcements](https://discord.com/channels/1280234300012494859/1280236929379602493/1308953617528656003)** (1 messages): 

> - `Intel AMA Session`
> - `Hackathon Insights` 


- **Reminder for Intel AMA Session**: Join us for the **Hackathon AMA with Intel** tomorrow at **3 PM PT (11/21)** for a chance to engage with Intel specialists.
   - Don't forget to [watch live here](https://www.youtube.com/watch?v=_Wm5guUXt54) and set your reminders!
- **Ask Intel Anything Opportunity**: This is a great chance to ask your questions and gain insights directly from **Intel specialists** during the AMA session.
   - Participants are eager to hear the latest updates and innovations from Intel's team!


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1309056551071514624)** (4 messages): 

> - `Quiz 10 Release`
> - `Hackathon Discussion` 


- **Quiz 10 Not Yet Released**: A member inquired about the status of **Quiz 10**, asking if it has been released on the website.
   - Another member confirmed that it hasn’t been released yet and mentioned that email notifications will be sent once it becomes available, likely within **a day or two**.
- **Hackathon Confusion**: A member expressed gratitude for the update regarding Quiz 10 but humorously acknowledged asking about the **hackathon** in the wrong channel.
   - This exchange reflects common channel mix-ups within the community, adding a lighthearted moment to the conversation.


  

---



### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1308923419408207952)** (4 messages): 

> - `int64 indexing`
> - `differences in ops_hip.py`
> - `maintenance of code`
> - `HIP setting in tinygrad` 


- **Need for int64 Indexing Explored**: A user questioned the necessity of **int64 indexing** in contexts that do not involve large tensors, prompting others to share their thoughts.
   - Another user linked to a relevant issue on GitHub for further context regarding this discussion.
- **Differences in ops_hip.py Files Dissected**: A member pointed out distinctions between two **ops_hip.py** files in the tinygrad repository, suggesting the former may not be maintained due to incorrect imports.
   - They noted that the latter is only referenced in the context of an external benchmarking script, which also contains erroneous imports.
- **Maintenance Status of ops_hip.py Files**: In response to the questioning of maintenance, another user confirmed that the **extra** ops_hip.py is not maintained while the **tinygrad** version should function if **HIP=1** is set.
   - This indicates that while some parts of the code may not be actively managed, others can still be configured to work correctly.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/tinygrad/tinygrad/blob/master/extra/backends/ops_hip.py">tinygrad/extra/backends/ops_hip.py at master · tinygrad/tinygrad</a>: You like pytorch? You like micrograd? You love tinygrad! ❤️  - tinygrad/tinygrad</li><li><a href="https://github.com/tinygrad/tinygrad/blob/master/tinygrad/runtime/ops_hip.py">tinygrad/tinygrad/runtime/ops_hip.py at master · tinygrad/tinygrad</a>: You like pytorch? You like micrograd? You love tinygrad! ❤️  - tinygrad/tinygrad</li><li><a href="https://github.com/tinygrad/tinygrad/blob/master/test/external/external_benchmark_hip_compile.py">tinygrad/test/external/external_benchmark_hip_compile.py at master · tinygrad/tinygrad</a>: You like pytorch? You like micrograd? You love tinygrad! ❤️  - tinygrad/tinygrad
</li>
</ul>

</div>
  

---



### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1309247649282785300)** (3 messages): 

> - `Event link confusion`
> - `Rescheduling events` 


- **Event link confusion arises**: A member expressed concern about not finding the event link on Luma, asking for clarification regarding its status.
   - *Chiphuyen* responded, apologizing and stating that they forgot to reschedule the event due to being sick.
- **Wishing well to a sick member**: Another member thanked *Chiphuyen* for the update and wished them a speedy recovery.
   - This showcases the supportive spirit within the community amidst event management challenges.


  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1308989709237616670)** (3 messages): 

> - `AI Expert Request`
> - `Carter Grant Seeking Opportunities` 


- **Urgent Call for AI Expertise**: *michel.0816* urgently asked for an **AI expert**, indicating a pressing need for assistance.
   - Another member suggested writing the problem in designated channels for better visibility.
- **Carter Grant's Job Hunt**: Carter Grant, a **full-stack developer** with 6 years of experience, announced his search for job opportunities.
   - He specializes in a wide range of technologies including **React**, **Node.js**, and **AI/ML**, and expressed eagerness to contribute to meaningful projects.


  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1309044531508809748)** (1 messages): 

> - `MI300X GPU Issues`
> - `Ablation Set Runs`
> - `Intermittent GPU Hangs`
> - `ROCm GitHub Issue` 


- **MI300X struggles with extended runs**: A member reported experiencing **intermittent GPU hangs** while conducting longer 8 x runs of **12-19 hours** on a standard ablation set with **axolotl**.
   - These issues seem to occur mostly past the **6-hour mark**, leading to discussions and tracking concerns on [GitHub](https://github.com/ROCm/ROCm/issues/4021).
- **Shorter runs show no problems**: The same member noted that their experiences indicate the **GPU hangs** do not appear during shorter runs of the model.
   - This distinction has raised questions about the **stability** of longer training sessions with MI300X when using axolotl.
- **Tracking GPU hang issues on GitHub**: The ongoing issues with **GPU hang HW Exceptions** during MI300X runs have been formally recorded on [GitHub Issue #4021](https://github.com/ROCm/ROCm/issues/4021).
   - The description includes detailed metrics such as **loss** and **learning rate**, highlighting the problem's technical context.



**Link mentioned**: <a href="https://github.com/ROCm/ROCm/issues/4021">[Issue]: Intermittent GPU Hang HW Exception by GPU on MI300X when training with axolotl · Issue #4021 · ROCm/ROCm</a>: Problem Description When running axolotl runs, I get intermittent GPU hangs: {&#39;loss&#39;: 0.4589, &#39;grad_norm&#39;: 1.0493940198290594, &#39;learning_rate&#39;: 5.284132841328413e-06, &#39;epoc...

  

---


### **OpenAccess AI Collective (axolotl) ▷ #[community-showcase](https://discord.com/channels/1104757954588196865/1117851527143493664/)** (1 messages): 

volko76: Do we still need to prompt correctly ?
https://youtu.be/m3Izr0wNfQc
  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1309181024604786708)** (2 messages): 

> - `Autoencoder Training` 


- **Training the Autoencoder Discussed**: A member mentioned the process of **training the autoencoder**, highlighting its importance in achieving model efficiency.
   - The conversation focused on techniques and implementation strategies for improving autoencoder performance.
- **Additional Insights on Autoencoders**: There were assorted opinions shared regarding the **sophistication of autoencoder architectures** in current models.
   - Discussions included the effectiveness of various algorithms and their impact on data representation.


  

---



### **Mozilla AI ▷ #[announcements](https://discord.com/channels/1089876418936180786/1089876419926032396/1308958351589113877)** (2 messages): 

> - `Refact.AI demo`
> - `Web Applets project`
> - `Public AI initiative` 


- **Refact.AI Live Demo Announced**: We are excited to have [Refact.AI](https://github.com/smallcloudai) team members join us for a live demo discussing their **autonomous agent** and innovative tooling.
   - Don't miss the opportunity to engage in this conversation by joining the live event [here](https://discord.com/events/1089876418936180786/1300459081181429810).
- **Mozilla's New Project: Web Applets**: Mozilla has launched an early-stage open-source project termed **Web Applets** aimed at developing AI-native applications for the web.
   - This initiative aims to promote **open standards** and accessibility in the AI landscape, encouraging collaboration among developers, as detailed [here](https://discord.com/channels/1089876418936180786/1231977676458168381).
- **Advocacy for Public AI**: Mozilla has accelerated **14 local AI projects** over the past year, focusing on advocating for **Public AI** and building necessary developer tools.
   - This effort is intended to foster state-of-the-art open-source AI technology, with a collaborative spirit emphasizing community engagement.


  

---



### **Gorilla LLM (Berkeley Function Calling) ▷ #[leaderboard](https://discord.com/channels/1111172801899012102/1214705495974092810/1309231946924294258)** (1 messages): 

> - `Llama 3.2 prompt usage` 


- **Query on Llama 3.2 Prompt Format**: A member inquired about the lack of usage of a specific prompt for **Llama 3.2**, referencing the [prompt format documentation](https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_2/#-function-definitions-in-the-system-prompt-).
   - The question hinted at curiosity regarding the **function definitions** in the system prompt, indicating a need for clarity on its application.
- **Interest in Prompt Applicability**: The conversation illustrated a broader interest in understanding the **applicability of prompts** within Llama models, particularly version 3.2.
   - This reflects ongoing discussions about best practices for maximizing model performance through **effective prompting**.



**Link mentioned**: <a href="https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_2/#-function-definitions-in-the-system-prompt-">Llama 3.2 | Model Cards and Prompt formats</a>: .

  

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
