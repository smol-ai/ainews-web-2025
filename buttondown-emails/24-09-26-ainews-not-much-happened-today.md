---
id: ecf9bc48-0db1-4ed2-b326-536bd9bc10ee
title: '[AINews] not much happened today'
date: '2024-09-26T22:52:11.195444Z'
status: sent
type: public
source: app
metadata: {}
original_slug: ainews-not-much-happened-today-2295
---

<!-- buttondown-editor-mode: plaintext -->**a quiet day is all you need.**

> AI News for 9/25/2024-9/26/2024. We checked 7 subreddits, [**433** Twitters](https://twitter.com/i/lists/1585430245762441216) and **31** Discords (**224** channels, and **3282** messages) for you. Estimated reading time saved (at 200wpm): **342 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

Many are still processing the surprise management turnover at OpenAI. [Sama](
https://x.com/sama/status/1839093415226524114) and [gdb](https://x.com/gdb/status/1839391073296408577) both posted statements. It seems the Anthropic rumors were postponed, but in the meantime the new [blueberry model rumor mill](https://x.com/ArtificialAnlys/status/1839333788817702920) is just getting started.

---

Since it's a quiet day, you could help out AINews by **[checking out the RAG++ course from Weights and Biases](https://wandb.me/ainews-course)**! We featured it yesterday but forgot to include the text link. Sorry!

> Swyx: Something we also missed in our initial scan yesterday was chapters 6 and 7 on response synthesis and optimmization. **Chapter 6 in particular is exactly what we had to do to build AINews** - everything you see below is AI generated thanks to these techniques.

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

**Meta Releases Llama 3.2 Models**

- **New Model Variants**: Meta AI announced the release of Llama 3.2, including [1B and 3B text-only models for edge devices](https://twitter.com/AIatMeta/status/1838993953502515702), as well as [11B and 90B vision models](https://twitter.com/AIatMeta/status/1838993953502515702) supporting multimodal tasks. All models support a [128K token context length](https://twitter.com/_philschmid/status/1838998169293615318).

- **Performance**: The [1B and 3B models outperform Gemma 2 2.6B and Phi 3.5-mini on key tasks](https://twitter.com/AIatMeta/status/1839018085329809831), while the [11B and 90B vision models are competitive with Claude 3 Haiku and GPT4o-mini](https://twitter.com/danielhanchen/status/1838991771948425652).

- **Technical Details**: The [vision models use adapter layers for image-text integration](https://twitter.com/AIatMeta/status/1839033482015895600), while the [1B and 3B models were created via pruning and distillation from Llama 3.1 8B](https://twitter.com/AIatMeta/status/1839018079529087158).

- **Ecosystem Support**: The models have [day one support for Arm, MediaTek, and Qualcomm](https://twitter.com/AIatMeta/status/1839018083207491888), and are [available on 25+ partner platforms](https://twitter.com/AIatMeta/status/1838993953502515702) including AWS, Azure, and Google Cloud.

- **Open Source**: Models are [downloadable from llama.com and Hugging Face](https://twitter.com/rohanpaul_ai/status/1839009997440880812), evaluated on 150+ benchmark datasets across languages.

**Other AI News**

- **OpenAI CTO Departure**: [Mira Murati, OpenAI's CTO, announced her departure](https://twitter.com/miramurati/status/1839025700009030027) from the company.

- **Molmo Release**: [Allen AI released Molmo](https://twitter.com/rohanpaul_ai/status/1839004028690186438), a family of open-source multimodal AI models, with their best model reportedly outperforming proprietary systems.

- **Gemini Updates**: [Google announced improvements to Gemini 1.5](https://twitter.com/demishassabis/status/1839085152259158336), with Flash and Pro production models offering competitive performance/price ratios.

- **Meta Connect Announcements**: Meta showcased [Project Orion, a full augmented reality glasses prototype](https://twitter.com/ylecun/status/1839038551457161256), and [hinted at a Quest 3S priced at $300](https://twitter.com/ylecun/status/1839091926118654316).

**AI Research and Development**

- **Benchmarks**: Discussions around [new benchmarks for multimodal models](https://twitter.com/DrJimFan/status/1839012622441787430) and [comparisons between open-source and closed models](https://twitter.com/Tim_Dettmers/status/1838959756377313750).

- **Model Optimization**: Techniques for [improving model performance](https://twitter.com/rohanpaul_ai/status/1839090102233870623) and [reducing computational costs](https://twitter.com/omarsar0/status/1838995198476460461) were shared.

- **AI Safety**: Ongoing discussions about [AI safety and alignment](https://twitter.com/mustafasuleyman/status/1838956259871277100) in the context of new model releases.

---

# AI Reddit Recap

## /r/LocalLlama Recap

**Theme 1. Open Source Vision-Language Models Challenging Proprietary Giants**

- **[Molmo is the first vision model I've found that can read an analog clock, something Claude/GPT/Gemini cannot do. It confused the minute and hour hands in the wristwatch pic but got the positioning right](https://i.redd.it/hedu6vjwvyqd1.png)** ([Score: 57, Comments: 13](https://reddit.com//r/LocalLLaMA/comments/1fp62xq/molmo_is_the_first_vision_model_ive_found_that/)): **Molmo**, a vision model, has demonstrated the ability to **read analog clocks**, a task that other prominent models like **Claude**, **GPT**, and **Gemini** have failed to accomplish. While Molmo successfully interpreted the positioning of clock hands, it made an error in distinguishing between the minute and hour hands when analyzing a wristwatch image.
  - **Molmo's** paper explicitly mentions training on **analog clock reading data**, which may explain its superior performance compared to other models. This inclusion of specific training data highlights the importance of diverse datasets in model capabilities.
  - The model demonstrated impressive accuracy in reading multiple watches, even when one was set an hour behind. This suggests potential applications in interpreting various visual representations like **graphs and charts**.
  - A user test showed **Molmo** providing detailed, perhaps overly thorough, responses to clock images. This level of detail contrasts with other models' tendency to focus on a single hypothesis, potentially indicating a more comprehensive analysis approach.
- **[Molmo: A family of open state-of-the-art multimodal AI models by AllenAI](https://molmo.allenai.org/)** ([Score: 184, Comments: 85](https://reddit.com//r/LocalLLaMA/comments/1fp5gut/molmo_a_family_of_open_stateoftheart_multimodal/)): Allen AI has released **Molmo**, a family of open-source **multimodal AI models** capable of processing both **text and images**. The Molmo models, available in sizes ranging from **300 million to 3 billion parameters**, achieve state-of-the-art performance on various benchmarks including **VQAv2**, **GQA**, and **OKVQA**, outperforming larger closed-source models like **GPT-4V** on certain tasks. These models are accessible through [Hugging Face](https://huggingface.co/allenai/molmo) and can be used for tasks such as visual question answering, image captioning, and multimodal chat.
  - **Molmo** models demonstrate impressive capabilities, including **telling time on analog clocks** and performing **spatial awareness tasks**. Users tested the model with multiple watches and found it could accurately identify different times, though it struggles with tasks like transcribing piano sheet music.
  - The model architecture uses **OpenAI's ViT-L/14 CLIP** for vision encoding, which outperformed **SigLIP** in experiments. **Matt**, the author, explained that SigLIP worked well for single-crop training but performed worse for multi-crop/higher resolution training used in Molmo.
  - Molmo includes **fully open-source datasets and training code** for multiple models. The team plans to release checkpoints and experiments with various vision encoder ablations, and is open to trying different language and vision backbones in future iterations.
- **[Ovis 1.6 - a Gemma 2-based 10B vision-language model that outperforms Llama 3.2 11B and GPT-4o-mini on MMMU](https://huggingface.co/AIDC-AI/Ovis1.6-Gemma2-9B)** ([Score: 49, Comments: 25](https://reddit.com//r/LocalLLaMA/comments/1fppvov/ovis_16_a_gemma_2based_10b_visionlanguage_model/)): **Ovis 1.6**, a **10B parameter vision-language model** based on **Gemma 2**, has been released, demonstrating superior performance compared to larger models like **Llama 3.2 11B** and **GPT-4o-mini** on the **MMMU** benchmark. This model achieves state-of-the-art results in various vision-language tasks, showcasing the potential of efficiently designed smaller models to compete with and surpass larger counterparts in multimodal understanding.
  - Users expressed **skepticism** about the claim of **Ovis 1.6** outperforming **Llama 3.2 11B**, noting the absence of Llama 3.2 in the comparison table and questioning the rapid performance assessment within 24 hours of Llama 3.2's release.
  - A user tested **Ovis 1.6** via the [Spaces demo](https://preview.redd.it/eebzp82x54rd1.png?width=2752&format=png&auto=webp&s=4bc40af9f2e1cffdd7db5620c3fe79255e403c99), finding it subjectively comparable to other models they've tried. Another user suggested that **Llama 3.2 11B** is inferior for vision tasks compared to models like **MiniCPM v2.6** and **Qwen 2 VL 7B**.
  - The OP clarified that the performance comparison is based on the **MMMU benchmark**, which is published for both models. Some users agreed that Ovis might be better in personal testing but emphasized the need for more comprehensive, numerical comparisons.


**Theme 2. Llama 3.2: Meta's Multimodal Leap in Open Source AI**

- **Llama-3.2 vision is not yet supported by llama.cpp** ([Score: 32, Comments: 34](https://reddit.com//r/LocalLLaMA/comments/1fppoch/llama32_vision_is_not_yet_supported_by_llamacpp/)): The **llama.cpp** project does not currently support **Llama-3.2 vision** capabilities, as indicated by an open issue on the project's GitHub repository. The [issue #9643](https://github.com/ggerganov/llama.cpp/issues/9643) suggests that work is needed to implement support for the vision features of the latest Llama model version.
  - **Ollama** is working on supporting **Llama-3.2 vision** independently of llama.cpp, as mentioned in their [release blog](https://ollama.com/blog/llama3.2) and related [PRs](https://github.com/ollama/ollama/pull/6963). Some users suggest focusing on Ollama or considering other tools like **mistral.rs** for better model support.
  - **Ggerganov**, llama.cpp repo owner, stated that adding multimodal support is an opportunity for new contributors with software architecture skills. He emphasized the need for more people with this skillset to sustain project quality, as indicated in a [GitHub comment](https://github.com/ggerganov/llama.cpp/issues/8010).
  - Users expressed disappointment in llama.cpp's lack of support for various vision models like **Phi3.5 Vision**, **Pixtral**, and **Qwen-2 VL**. Some speculated about challenges in implementation, while others joked about potential geoblocking issues affecting access to models.
- **Llama 3.2 Multimodal** ([Score: 244, Comments: 87](https://reddit.com//r/LocalLLaMA/comments/1fp9had/llama_32_multimodal/)): Meta has released **Llama 3.2**, an update to their open-source AI model featuring new **multimodal capabilities** and additional **model sizes**. While specific details about the release are not provided in the post body, the title suggests that Llama 3.2 can now process and generate both text and visual content, potentially expanding its applications across various domains.
  - **Llama 3.2** models (11B and 90B) show strong performance on multimodal benchmarks, outperforming **Claude3-Haiku** and competing with **GPT-4o-mini** in areas like mathematical reasoning and visual question answering. The **90B** model particularly excels in multilingual tasks, scoring **86.9%** on the VQAv2 test.
  - Meta unexpectedly released smaller **1B** and **3B** models alongside the larger versions, trained on up to **9T tokens** for 370K and 460K hours respectively. These models demonstrate impressive capabilities in tooling and function-calling, reaching performance levels of 8B models.
  - The release faced some controversy, with **EU access being disallowed** for the models on Hugging Face. This sparked discussions about the implications of the **AI Act** on model availability and potential workarounds for both individuals and companies.
- **Run Llama 3.2 3B on Phone - on iOS & Android** ([Score: 151, Comments: 47](https://reddit.com//r/LocalLLaMA/comments/1fppt99/run_llama_32_3b_on_phone_on_ios_android/)): The **PocketPal AI** app now includes the **Llama 3.2 3B model** (Q4_K_M GGUF variant) for both **iOS** and **Android** devices, allowing users to run this AI model on their smartphones. The developer has currently added only the **Q4 variant** to the default models due to potential throttling issues with the Q8 version, but users with sufficient device memory can import the GGUF file as a local model, ensuring to select the **"llama32" chat template**.
  - **PocketPal AI** app's UI received detailed feedback from users, suggesting improvements like renaming tabs to "Downloaded" and "Available Models", and making the interface more intuitive. The developer acknowledged the feedback positively.
  - Users reported performance metrics, with one noting **11 tokens/sec** on their device and another sharing **CPU usage** on an **iPhone 14 iOS 18.0**. A user successfully ran a **Mistral Nemo 12B** model in **Q4K** on their 12GB RAM smartphone.
  - The app uses **llama.cpp** for inference and **llama.rn** for React Native bindings. It currently uses **CPU** on Android, and while not yet open-source, the developer mentioned they might consider it in the future.


**Theme 3. Qwen 2.5: Alibaba's Breakthrough in Open Source LLMs**

- **Qwen 2.5 vs Llama 3.1 illustration.** ([Score: 30, Comments: 17](https://reddit.com//r/LocalLLaMA/comments/1fp8v9h/qwen_25_vs_llama_31_illustration/)): The author compared **Qwen 2.5** and **Llama 3.1** models after acquiring a **3090 GPU**, creating an illustration to evaluate their performance. After using the **32B Qwen model** for several days, they shared the image to highlight Alibaba's achievement, noting the model's impressive capabilities.
  - Users discussed the availability of **32B models**, with one recommending the **70B model** for its performance (**16 T/s**). The original poster inquired about significant improvements between **32B and 70B** models to justify purchasing a second **3090 GPU**.
  - Some users praised **Alibaba's** contributions to open source, expressing surprise at both **Alibaba and Meta** gaining respect in the AI community. Others noted the impressive capabilities of **Qwen's 70B model**, comparing its performance to **400+ billion-parameter models**.
  - Discussion on running large models on consumer hardware, with the original poster sharing their setup using an **ollama fork** supporting context quantization, running either **"q4 32b q4 64k" or "q6 14b q4 128k"** configurations on a **3090 GPU**.
- **Is qwen2.5:72b the strongest coding model yet?** ([Score: 66, Comments: 66](https://reddit.com//r/LocalLLaMA/comments/1fpq1jq/is_qwen2572b_the_strongest_coding_model_yet/)): The user reports exceptional coding assistance from the **Qwen 2.5 72B Instruct** model accessed via **Hugging Face Spaces**, suggesting it outperforms **Claude** and **ChatGPT-4** for their specific needs. They inquire if this model is objectively the best for coding tasks, providing a [link to the Hugging Face space](https://huggingface.co/spaces/Qwen/Qwen2.5-72B-Instruct) for reference.
  - **Qwen2.5 72B** is praised for its coding performance, with the **32B version** being nearly as capable. Users anticipate the release of **qwen2.5 32b-coder**, expected to surpass the 72B model in coding tasks.
  - Debate over model comparisons: Some argue **Qwen2.5 72B** is not superior to **Claude** or **Mistral-Large2-123B** for complex tasks, while others find open-source models now sufficient for most coding needs. **Context window size** is highlighted as crucial for large projects.
  - Users discuss hardware setups for running large models locally, with recommendations including multiple **RTX 3090s** or **P40** GPUs. **Quantization** techniques like **Q4** and **AWQ** are mentioned for efficient model deployment.


**Theme 4. EU AI Regulations Impact on Model Availability and Development**

- **[LLAMA 3.2 not available](https://i.redd.it/mupq13jgk2rd1.jpeg)** ([Score: 1060, Comments: 388](https://reddit.com//r/LocalLLaMA/comments/1fpmlga/llama_32_not_available/)): Meta's **LLAMA 3.2** models are currently **unavailable** to users in the **European Union** due to regulatory restrictions. This limitation affects access to the models through both the **Meta AI website** and **third-party platforms** like Hugging Face. The situation highlights the impact of **EU regulations** on the availability of AI models in the region.
  - **Meta's LLAMA 3.2** models are unavailable in the **EU** due to potential **illegal user data scraping** from Facebook photos for training. The **1B and 3B text models** are still accessible, but the **vision models** are banned.
  - Users debate the merits of **EU regulations** like **GDPR**, with some praising consumer protection efforts while others argue it stifles innovation and competitiveness in the **AI race**. The **AI Act** aims to regulate high-risk AI systems and biometric categorization.
  - There's ongoing discussion about **Meta's compliance** with **EU regulations** and whether LLAMA is truly **open source**. Some speculate this could be a political move by Meta to pressure the EU into declaring LLAMA as open source, exempting it from certain regulations.

**Theme 5. Challenges in Scaling and Reliability of Large Language Models**

- **Larger and More Instructable AI Models Become Less Reliable** ([Score: 109, Comments: 23](https://reddit.com//r/LocalLLaMA/comments/1fpjo5w/larger_and_more_instructable_ai_models_become/)): A **Nature paper** reveals that **larger AI models** with more instruction and alignment training become **less reliable** across five difficult task categories. While performance improves on easier tasks, models increasingly give **incorrect answers** for harder variants instead of refusing to answer, with **human readers unable to accurately discern** the correctness of these confident but wrong responses. This trend was observed across multiple model families including **OpenAI GPT**, **Meta's Llama**, and **BLOOM**.
  - **RLHF methods** are criticized for not rewarding models to accurately represent **epistemic status**. Some argue this research may be **obsolete**, using older models like **GPT-3.5** and **Llama 1**, while others contend the trend remains relevant.
  - The study's definition of **"avoidant responses"** is questioned, with **nearly all** such responses categorized as "non-conforming avoidant". Critics argue these responses are not necessarily more reliable, as defined in the [supplementary information](https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-07930-y/MediaObjects/41586_2024_7930_MOESM1_ESM.pdf).
  - The paper's publication in **Nature** rather than a top ML conference is noted as unusual. It was submitted on **June 2, 2023** and published recently, which is atypical for computer science research that usually favors faster conference publications.

- **Why do most models have "only" 100K tokens context window, while Gemini is at 2M tokens?** ([Score: 99, Comments: 93](https://reddit.com//r/LocalLLaMA/comments/1fp4s7e/why_do_most_models_have_only_100k_tokens_context/)): The post discusses the **disparity in context window sizes** between most language models (with **100K tokens**) and **Gemini** (with **2M tokens**). The author questions why other models can't match or exceed Gemini's context window, especially given Gemini's effectiveness and the possibility of **Gemini 2.0** expanding even further. They seek to understand the technical limitations preventing other models from achieving similar context window sizes.
  - **Google's hardware** capabilities, including their **TPUs** with **256-way fast inter-chip interconnect** and **8,192 GB of memory** per pod, significantly outperform typical **Nvidia** setups. This hardware advantage may be a key factor in **Gemini's** large context window.
  - The **effective context length** of most models is often much less than advertised, typically around **1/4** of their stated context size. **Google** appears to have made progress in solving long context understanding and information retrieval issues.
  - **Google Research** published work on **Infinite Context Windows**, introducing **compressive memory** in the dot product attention layer. This, along with techniques like **Ring Attention**, may contribute to Gemini's ability to handle longer contexts efficiently.

## Other AI Subreddit Recap

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity


**AI Model Advancements and Releases**

- **OpenAI's Advanced Voice Mode**: OpenAI released an advanced voice mode for ChatGPT with capabilities like [singing, humming, and voice imitation](https://www.reddit.com/r/singularity/comments/1fp1ifc/chatgpts_advanced_voice_mode_can_sing_hum/), though it's instructed not to use some features. The system prompt restricts flirting and romantic interactions.

- **Meta AI with Voice**: Meta announced [a competitor to OpenAI's Advanced Voice model](https://www.reddit.com/r/singularity/comments/1fpaeew/meta_announces_meta_ai_with_voice_a_competitor_to/), allowing users to put themselves into AI avatars.

- **Salesforce xLAM-1b**: Salesforce released xLAM-1b, a 1 billion parameter model that [achieves 70% accuracy in function calling, surpassing GPT 3.5](https://www.reddit.com/r/LocalLLaMA/comments/1dz8g10/salesforce_tiny_giant_xlam1b_model_surpasses_gpt/) despite its relatively small size.

- **Phi-3 Mini Update**: Rubra AI released an updated Phi-3 Mini model in June [with function calling capabilities](https://www.reddit.com/r/LocalLLaMA/comments/1dzhe38/phi3_mini_june_with_function_calling/), competitive with Mistral-7b v3.

**AI Research and Techniques**

- **Google DeepMind's Multimodal Learning**: A [Google DeepMind paper](https://arxiv.org/html/2406.17711v1) demonstrates how data curation via joint example selection can accelerate multimodal learning.

- **Microsoft's MInference**: [Microsoft's MInference technique](https://arxiv.org/abs/2407.02490) enables inference of up to millions of tokens for long-context tasks while maintaining accuracy.

- **Scaling Synthetic Data Creation**: A [paper on scaling synthetic data creation](https://www.reddit.com/r/MachineLearning/comments/1dzergu/r_scaling_synthetic_data_creation_with_personas/) leverages diverse perspectives within large language models to generate data from 1 billion web-curated personas.

**AI Industry Developments**

- **OpenAI Restructuring**: OpenAI is [removing non-profit control and giving Sam Altman equity](https://www.reuters.com/technology/artificial-intelligence/openai-remove-non-profit-control-give-sam-altman-equity-sources-say-2024-09-25/). This coincides with [several key personnel departures](https://www.reddit.com/r/singularity/comments/1fpj6ls/more_people_leaving_openai/), including CTO Mira Murati.

- **Google's AI Talent Acquisition**: Google [paid $2.7 billion to bring back AI researcher Noam Shazeer](https://www.reddit.com/r/singularity/comments/1fpbc6u/article_google_paid_27_billion_to_bring_back_an/), who had previously left to start Character.AI.

- **OpenAI's Data Center Plans**: Sam Altman pitched a plan to [build multiple 5 GW data centers across various states](https://www.bloomberg.com/news/articles/2024-09-24/openai-pitched-white-house-on-unprecedented-data-center-buildout), starting with one 5GW facility.

**AI Applications and Demonstrations**

- **Alibaba's MIMO**: Alibaba presented MIMO, a system for [controllable character video synthesis with spatial decomposed modeling](https://www.reddit.com/r/singularity/comments/1fp0ti3/alibaba_presents_mimo_controllable_character/).

- **FaceFusion 3.0.0**: The launch of [FaceFusion 3.0.0](https://www.reddit.com/r/StableDiffusion/comments/1fpbm3p/facefusion_300_has_finally_launched/) demonstrates advancements in face swapping technology.

- **Looney Tunes Background LoRA**: A user trained a [Looney Tunes Background image style LoRA](https://www.reddit.com/r/StableDiffusion/comments/1fp94dn/still_having_fun_with_15_trained_a_looneytunes/) for Stable Diffusion 1.5, showcasing the versatility of fine-tuning techniques.

**AI Ethics and Regulation**

- **EU AI Regulations**: The EU's AI Act includes provisions that [restrict emotion recognition technologies in workplaces and schools](https://www.reddit.com/r/singularity/comments/1fp4789/new_voice_is_illegal_in_eu_workplaces_and_schools/), potentially impacting the deployment of advanced AI voice models in these settings.

**Hardware and Infrastructure**

- **Meta's AR Glasses**: Meta introduced [Orion, their first true augmented reality glasses](https://about.fb.com/news/2024/09/introducing-orion-our-first-true-augmented-reality-glasses/), signaling advancements in wearable AI technology.


---

# AI Discord Recap

> A summary of Summaries of Summaries by O1-mini

**Theme 1. Llama 3.2 Model Releases and Performance**

- [**Llama 3.2 Launched in Multiple Sizes**](https://discord.com/channels/879548962464493619/879548962464493622/1288578069032079464): **Meta** released **Llama 3.2** in four sizes (**90B**, **11B**, **3B**, **1B**) targeting the **medical domain**. Despite this, **Llama-3.1 70B** outperforms it with an **84% average score** and **95.14%** in **MMLU College Biology**.

- [**Benchmark Variances Highlight Performance Gaps**](https://discord.com/channels/879548962464493619/1110598183144399058/1288579271870386207): In **LM Studio**, users reported **Llama 3.2 1B** achieving **49.3%** and **3B** at **63.4%**, with quantized models running at **15-17 tokens/sec**, showcasing significant performance discrepancies.

- [**Community Critiques on Llama 3.2's Limitations**](https://www.youtube.com/watch?v=_MQVHyEeER4): Members expressed disappointment with **Llama 3.2** compared to **Llama 3.1**, highlighting issues in executing basic tasks like file counting, as detailed in a community-shared [YouTube video](https://www.youtube.com/watch?v=_MQVHyEeER4).

**Theme 2. AI Model Fine-Tuning and Optimization**

- [**Unsloth AI Enhances Fine-Tuning Efficiency**](https://discord.com/channels/1179035537009545276/1179035537529643040/1288580232340836406): **Unsloth AI** optimized fine-tuning for **Llama 3.2**, achieving **2x faster training** with **60% less memory** usage, enabling accessibility on **lower VRAM setups**. Users successfully implemented QLoRA configurations and await **vision model support**.

- [**Effective LLM Training Strategies Discussed**](https://discord.com/channels/1179035537009545276/1179035537529643040/1288580232340836406): Community members exchanged insights on **LLM training techniques**, emphasizing dataset configurations, varying batch sizes, and meticulous parameter tuning to optimize performance and minimize errors.

- [**WeightWatcher Aids in Model Diagnostics**](https://github.com/CalculatedContent/WeightWatcher): Discussions highlighted **WeightWatcher**, a tool for analyzing model weights and distributions, facilitating informed training decisions and enhancing optimization strategies through detailed diagnostics.

**Theme 3. Hardware and GPU Discussions for AI**

- [**GPU Accessibility on Free Platforms Explored**](https://discord.com/channels/879548962464493619/1110598183144399058/1288579271870386207): Users debated the potential of running models on the **free tier of Google Colab**, questioning what qualifies as 'relatively performant' without financial investment, highlighting accessibility in AI model deployment.

- [**Leaked Specs of NVIDIA RTX 5090 and RTX 5080 Spark Discussions**](https://videocardz.com/newz/nvidia-geforce-rtx-5090-and-rtx-5080-specs-leaked): **NVIDIA’s** upcoming **RTX 5090** boasts **21,760 CUDA cores**, **32GB GDDR7 memory**, and **600W** power, while the **RTX 5080** features **16GB VRAM**. This prompts debates on the trade-offs between **VRAM and speed** for content creators and gamers.

- [**VRAM Constraints Affecting Large Model Deployment**](https://discord.com/channels/1110598183144399058/1153759714082033735/1288781696388431883): Conversations emphasized that **24GB GPUs** struggle with **70B** models, favoring setups that maintain at least **15 tok/s** speed. Members explored solutions like multi-GPU integration and model quantization to overcome these limitations.

**Theme 4. AI Policies and Corporate Shifts**

- [**OpenAI Leadership Shakeup Sparks Concerns**](https://x.com/papers_anon/status/1839131401322639805?s=46): Recent departures of key personnel, including **Mira Murati** and **Barret Zoph**, have led to speculations about **OpenAI’s** shift from a **startup** to a **corporate structure**, potentially impacting innovation and attracting regulatory scrutiny.

- [**Licensing Restrictions Limit Llama 3.2's EU Availability**](https://discord.com/channels/1053877538025386074/1053877538025386074/1288615728127148032): **Meta AI**'s **Llama 3.2** models, especially **11B** and **90B Vision Instruct**, face **EU access restrictions** due to licensing disagreements, limiting availability for local developers and sparking debates on compliance.

- [**Profit Interest Units in OpenAI’s Non-Profit Structure**](https://riveron.com/posts/accounting-for-pius/): Discussions emerged on **Profit Interests Units (PIUs)** within **OpenAI’s** **non-profit** status, raising concerns about leveraging non-profit frameworks for profit motives, potentially inviting regulatory actions from bodies like **California's Attorney General**.

**Theme 5. Community Tools and Integrations**

- [**Aider Introduces Senior and Junior Roles for Models**](https://discord.com/channels/1131200896827654144/1131200896827654149/1288580759694868500): **Aider** launched **'Senior'** and **'Junior'** roles to streamline the coding process by dividing responsibilities between planning and execution. Users suggested alternative names like **'Planner'** and **'Executor'** for clarity.

- [**OpenRouter Releases Vision Llama and Updates Tokenization**](https://x.com/OpenRouterAI/status/1839157099747479553): **OpenRouter** introduced the first **Vision Llama** with a [free endpoint](https://x.com/OpenRouterAI/status/1839157099747479553) and added **five new endpoints**. They also announced a shift to counting **tokens instead of characters** for **Gemini models**, reducing token counts by **~4x** and planning to **double prices** post **October 1**.

- [**LM Studio Faces Compatibility Issues with Llama 3.2 Vision Models**](https://discord.com/channels/1110598183144399058/1110598183144399061/1288579271870386207): Users in **LM Studio** Discord highlighted that **Llama 3.2 Vision Instruct** models aren’t supported in **llama.cpp**, expressing interest in deploying these models despite integration challenges and emphasizing the need for future support in quantization frameworks.

- [**LangChain Discusses Source Document Retrieval Logic**](https://discord.com/channels/1038097195422978059/1038097196224086148/1288887526488019116): **LangChain** users debated the conditional retrieval of **source documents** based on **LLM** confidence, advocating for more intuitive response behaviors and discussing alternative debugging tools like **Langfuse** to monitor without compromising data privacy.

- [**Tinygrad’s Custom Kernel Generation Enhances Optimization**](https://github.com/mesozoic-egg/tinygrad-notes/tree/main): Within **tinygrad**, users emphasized the advantage of **custom kernel generation** over **PyTorch’s** fixed kernels, offering greater optimization opportunities and potential performance benefits tailored to specific applications.


---

# PART 1: High level Discord summaries




## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Llama 3.2 Released with Multimodal Features**: Meta has launched **Llama 3.2**, available in four sizes (90B, 11B, 3B, 1B) aimed at the medical domain, but ironically, **Llama-3.1 70B** has outperformed it by a significant margin.
   - In benchmark tests, **Meta-Llama-3.1-70B-Instruct** achieved an **84% average score**, excelling in **MMLU College Biology** at **95.14%**.
- **Tau's Latest Innovations Uncovered**: A new [article](https://dev.to/p3ngu1nzz/from-data-expansion-to-embedding-optimization-taus-latest-innovations-4h3n) highlights innovations in data expansion and embedding optimization by **P3ngu1nzz** alongside a [training run](https://huggingface.co/p3nGu1nZz/Tau/tree/main/results/tau_agent_D10_100M) of **Tau** consisting of **100 million steps**.
   - These advancements focus on improving contextual understanding, crucial for various AI applications.
- **Gemini Makes Waves in Object Detection**: **Gemini's object detection** functionality has been launched, with detailed insights available [here](https://huggingface.co/spaces/saq1b/gemini-object-detection).
   - The aim is to enhance the capabilities in AI for object detection tasks, utilizing cutting-edge technology.
- **Building AGENTIC RL SWARM for Legal Reasoning**: A member is developing an **AGENTIC RL SWARM** setup designed for processing complex legal tasks by integrating tools such as **RAG** and **graphrag**.
   - This integration aims to enhance contextual retrieval and functionality, focusing on rigorous evaluation of outputs.
- **Colab Free Tier Performance Potential**: Users discussed the promising potential of running models in the **free tier of Google Colab**, questioning what constitutes 'relatively performant' in such a setting.
   - This brings significant implications for accessibility in deploying AI models without financial constraints.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Llama 3.2: A Leap in Fine-Tuning**: The Unsloth team announced that fine-tuning for **Llama 3.2** has been optimized, achieving **2x faster training** with **60% less memory** usage, making it accessible even on lower VRAM setups.
   - Users reported successful implementation with QLoRA configurations, while vision model support is expected soon, prompting a call for updates to Unsloth.
- **NVIDIA's New Lineup Causes Ripple Effect**: Leaked specifications of NVIDIA’s upcoming **RTX 5090** and **RTX 5080** GPUs reveal increased CUDA cores but varied VRAM, igniting discussions on upgrade justifications among current users.
   - Concerns were raised about potentially sacrificing VRAM for faster specs, especially for content creators and gamers who require stability in performance.
- **OpenAI's Corporate Shift Sparks Investor Doubts**: Concerns noted within the community suggest that OpenAI is transitioning away from its exciting startup roots towards a corporate structure, impacting innovation.
   - Investors are speculating reasons for the lack of significant growth, with whispers of internal scrutiny if targets, notably **10x growth**, remain unmet.
- **Strategies for Effective LLM Training**: An inquiry regarding **LLM training for marketing analysis** led to rich discussions about dataset configurations and fine-tuning practices to optimize performance.
   - Users exchanged insights on approaches, including varying batch sizes and training techniques, emphasizing the need for careful parameter tuning to reduce errors.
- **Fine-Tuning Inspirations with Alpaca**: Community members voiced their experiences using the **Alpaca instruction template** in fine-tuning processes, focusing on tokenizer configurations.
   - Guidance was sought on integrating the template, highlighting its complexity and the training challenges it presents.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Llama 3.2 Performance Benchmarks**: Users benchmarked Llama 3.2 models, revealing **1B** at **49.3%** and **3B** at **63.4%**, showcasing significant performance discrepancies with quantized models achieving around **15-17 tokens/sec**.
   - Broader comparisons highlighted how this affects token throughput across platforms.
- **Llama 3.2 Vision Models Unsupported**: **Llama 3.2 Vision Instruct** models aren’t supported in **llama.cpp**, leaving users unsure about future integration and quantization challenges.
   - Notable interest persists in deploying these models despite the integration hurdles.
- **VRAM Blocks Large Model Deployment**: Participants agreed that VRAM is crucial for large models, with **24GB** GPUs struggling with **70B** models and favoring setups that maintain at least **15 tok/s** speed.
   - Discussion focused on VRAM trade-offs and feasible model options.
- **Performance Metrics Across GPUs**: Benchmarking highlighted around **35 tokens/sec** on AMD **RX 5700 XT** and **40 tokens/sec** on **NVIDIA RTX 4060** systems.
   - Users noted impressive results of **61 tokens/sec** from Apple's **M3 Max** chip, emphasizing variance in hardware capabilities.
- **Hardware Needs for LLMs Discussed**: A conversation about LLMs suitable for **Intel i7-8750H** with **32GB RAM** recommended options like **qwen 2.5**, noting integrated Intel GPU limitations.
   - The reliance on system RAM indicates slower processing times for significant models.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **New Sr. & Jr. Roles Make Coding Easier**: Aider's latest update introduces **'Senior'** and **'Junior'** roles for models, streamlining the coding process by clearly defining responsibilities between planning and execution.
   - Users are suggesting alternative names like **'Planner'** and **'Executor'** to reduce confusion around these roles.
- **User Experience Sets a Fast Pace**: Discussions around Aider's UI point to making the two-step process optional, allowing for a quicker edit option while still enabling planning through the new role configuration.
   - Ideas like a **/fast** command to switch modes are being proposed to enhance the user experience without compromising the advanced features.
- **Best Model Pairing for Aider**: Community members debated optimal model configurations, suggesting using **OpenAI's o1-preview** for the Senior role and **Claude 3.5 Sonnet** for Junior tasks.
   - There's also consideration for the **Deepseek** model when speed is a priority during implementations.
- **Mend Renovate Automates Dependency Management**: The conversation highlighted **Mend Renovate**, a tool that automates dependency updates by identifying newer package versions and facilitating code integration.
   - Users expressed a wish for LLMs to independently handle package versioning to streamline project setups.
- **Sonnet's Reliability Under Scrutiny**: Concerns were raised regarding **Sonnet's** performance, as users noted degraded reliability without any clear triggers.
   - The community speculated that overlapping system bug fixes might be affecting Sonnet's functionality.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Hermes 3 Hits HuggingChat**: Nous Research released the **Hermes 3** model sized at **8B** on [HuggingChat](https://huggingface.co/chat/settings/NousResearch/Hermes-3-Llama-3.1-8B), showcasing enhancements in instruction adherence.
   - This model aims to boost interactivity in AI applications, reflecting Nous Research's commitment to advancing user-responsive AI.
- **Llama 3.2 Vision Encoder is Massive**: The **Llama 3.2 Vision Encoder** boasts significant sizes, with the **11B model** nearing **3B** parameters and the **90B model** reaching **18B**.
   - *Members emphasized its gigantic scale*, highlighting implications for processing capabilities in various applications.
- **Inferring Llama 3.2 Requires Serious Power**: To infer the **90B Llama 3.2**, users suggest **3x H100 GPUs** might be necessary, potentially **4x for larger batches** or tensor parallelism.
   - This points to the practical GPU infrastructure consideratons needed for efficient model deployment, especially on platforms like **Runpod**.
- **Wordware Apps Integrate O1Mini**: **Updated Wordware apps** now include **O1Mini**, enhancing functionality through [OPUS Insight](https://app.wordware.ai/explore/apps/aa2996a0-93c9-4c19-ade2-1796c5c8a409) that utilizes **Sonnet 3.5** for model rankings.
   - This update reinforces the competitive edge in model reviews and user engagement with comprehensive ranking features.
- **Judgement and Reward Modelling Enhance Hermes 3**: Inquiries about **judgement and reward modelling** improvements for **Hermes 3** confirmed the use of **synthetic data** in its training.
   - This approach aims to amplify model performance beyond what traditional public datasets could offer.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Scam Link Alert in General Channel**: Members raised concerns regarding a potentially fraudulent link, ensuring action was taken against its poster.
   - *Definitely a scam,* one member noted, highlighting the vigilance within the community.
- **Triton Conference 2024 Recordings Available**: The recordings of the [Triton Conference 2024](https://www.youtube.com/watch?v=NZz5sczZ_30) are now accessible, featuring keynotes from industry leaders.
   - The afternoon session included insights from **Meta** on their Triton strategy, available at [this link](https://www.youtube.com/watch?v=ONrKkI7KhU4).
- **Advanced PyTorch Profiling Techniques**: Members explored methods for checking memory allocation in **PyTorch**, focusing on layers, weights, and optimizer states.
   - Techniques such as using **torchdispatchmode** for automated profiling were discussed for optimizing memory utilization.
- **Llama 3.2 Introduced for Edge Computing**: Meta has launched **Llama 3.2**, featuring lighter **vision LLMs** optimized for edge devices, enhancing accessibility for developers.
   - Concerns arose regarding its limited availability in the **EU**, impacting local developers' access to advanced resources.
- **Community Meetup Planning in Guatemala**: An initiative for organizing meetups in **Guatemala** was proposed, inviting local enthusiasts to connect.
   - The planning emphasizes regional collaboration and the importance of building a local AI community.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Niche Interests Fuel AI Advances**: A member highlighted how specific interests, like **PonyDiffusion**, drive innovations in AI art generation, pushing boundaries in creativity.
   - *Fandoms shape perceptions* on AI content, indicating a growing interconnectedness between user engagement and technological progress.
- **GPU Questions Surge for Stable Diffusion**: A newcomer inquired about running **Stable Diffusion** without a GPU, prompting suggestions for using **Kaggle** over Colab for better resources.
   - The consensus stressed the necessity of a capable GPU for optimal **Stable Diffusion** performance in image generation tasks.
- **Lora Models Fail to Impress**: Concerns arose when a user reported their **Lora model** produced insufficient alterations in output images, unlike high-quality examples seen on Hugging Face.
   - Clarifications revealed subtle changes from the model, but they didn’t meet the *high expectations* set by benchmark images.
- **RVC Installation Queries on Colab**: Members discussed how to install **RVC** for voice conversion on **Colab Pro**, with numerous **RVC models** available on Hugging Face being suggested.
   - This resource sharing helped streamline the setup process for those diving into voice manipulation tasks.
- **Image Generation Times Under Scrutiny**: A user noted erratic image generation times on their local setup using the same parameters, leading to conversations on **VRAM usage** and benchmark efficiency.
   - Speculation about system traffic impacting outputs showcased the ongoing quest for optimizing **Stable Diffusion** operations.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **No Advertising Policy Clarified**: The Discord community has a strict **no advertising** policy, supporting research sharing but prohibiting promotions of companies and products.
   - Participation emphasizes adherence to channel-specific rules for clarity on community guidelines.
- **Inquiry on Filler Tokens in LLMs**: Discussion revolved around the effectiveness of **filler tokens** in LLM architectures, acknowledging success in synthetic tasks, but questioning generalizability.
   - *How can LLMs truly benefit from filler tokens?* remains a pressing question, indicating a need for further investigation.
- **Seeking Chinchilla Scaling Laws Dataset**: A member seeks a dataset showcasing the correlation between **# params**, **# tokens**, and **loss** to analyze lower-order terms without multiple model trainings, referencing the [Chinchilla scaling laws paper](https://arxiv.org/pdf/2405.15074).
   - This highlights the need for more accessible resources for researchers to validate scaling outcomes.
- **FA3 Integration Efforts on H100s**: Talks emerged around adding **FA3** support to small model training on **H100s**, with expectations that the integration might be straightforward.
   - Challenges persist due to limited H100 access, complicating testing and implementation efforts.
- **Debugging Token Generation Issues**: A user reported exceeding maximum sequence length during token generation, discovering potential issues with the `tok_batch_encode` method.
   - Peer responses highlighted the need for a collective debugging effort to resolve these challenges effectively.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity AI struggles with context retention**: Users expressed concerns about **Perplexity AI** failing to remember past questions, particularly after follow-ups, which has worsened recently.
   - One user mentioned, *'this platform is still useful day-to-day but has definitely gotten worse.'*
- **Excitement over Llama 3.2 Launch**: A member announced that **Llama 3.2** is released on llama.com, igniting excitement with a 'LFG' call to action.
   - However, another member has not yet seen it appear on Perplexity's interface.
- **Mira Murati's Departure from OpenAI**: Mira Murati has officially **departed OpenAI**, triggering discussions about talent migration in the AI sector, as seen in this [YouTube video](https://www.youtube.com/embed/zk1GwCIEvVU).
   - The implications for the organization and overall AI tech landscape continue to be speculated upon.
- **AI Trumps reCAPTCHA Challenges**: An analysis shared that **AI beats reCAPTCHA** systems, raising concerns about online security and the need for updated methodologies.
   - [Details here](https://www.perplexity.ai/search/ai-beats-recaptcha-IJvLzX98RkeMdh.kpXIqXw) showcases the evolving capabilities of AI.
- **Clarifying Perplexity Structure in Zapier**: A member sought clarification on using **Perplexity** within Zapier, particularly about integrating with webhooks.
   - *Is there a specific format for how messages should be structured?*



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Meta AI Access Restrictions Frustrate Users**: Members voiced frustration over accessing **Meta AI**, particularly outside the U.S., with some users attempting VPNs for workarounds. The **Llama 3.2 license**'s EU incompatibility exacerbates these access challenges.
   - The discussions highlighted critical limitations that hinder users from effectively utilizing the AI tools they need.
- **Llama 3.2 Launch Stirring Controversies**: With the introduction of **Llama 3.2**, users dissected the new multimodal capabilities, grappling with compatibility issues for EU users and **Hugging Face** hosting concerns.
   - Concerns were raised about the functionality and access to the essential models required for development.
- **ROI on AI IDEs for Game Development**: Members shared their top picks for AI IDEs in game development, highlighting options like **Cursor** and **GitHub Copilot** for efficient code generation.
   - One user shared that they successfully integrate **ChatGPT** with SSH for real-time code modifications, optimizing their workflow.
- **Advanced Voice Mode Falls Flat**: Frustrations arose over **Advanced Voice Mode**, as users lamented its lack of Internet search capabilities and the cumbersome need to switch back to text mode.
   - Despite its limits, members remain hopeful about improvements expected with the arrival of ChatGPT-5.
- **o1 Struggles with File Uploading**: Members discussed the lack of file upload capabilities in **o1**, leading many to revert to **GPT4o**, which disrupts productivity.
   - Concerns were raised about the performance of the **o1** model in following complex instructions compared to **GPT4o**.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **OpenAI Leadership Shakeup Raises Questions**: Recent departures at OpenAI, including key leaders, have sparked suspicions about the company's direction, with members expressing concerns that *ALL of OG OpenAI left besides Sam*.
   - The timing of these resignations has led to speculation about internal tensions, suggesting the organization might be at a crossroads.
- **Skepticism Surrounds Molmo's Performance Claims**: Amid claims that **Molmo** outperforms **LLaMA 3.2**, members expressed doubt about the authenticity of these assertions, with one stating *there's no proof* of biased endorsements.
   - A clarification regarding Molmo's announcement timeline noted it was launched just hours before LLaMA 3.2, but personal tests are encouraged to validate performance.
- **Profit Interest Units Stir Controversy**: Members discussed the implications of introducing *Profit Interests Units (PIUs)* in a non-profit setting, questioning potential regulatory repercussions.
   - Concerns were raised that leveraging non-profit status for profit motives could invite scrutiny from entities like California's Attorney General.
- **NeurIPS Submission Rejections Highlight Bias**: The rejection of **Rewardbench** at NeurIPS has been a topic of humor and frustration amongst members, with comments on the dismissive feedback regarding the use of **C++**.
   - Concerns were voiced over academic gatekeeping, with one member expressing that it seems *weird to give any sort of “equity” compensation in a non-profit though*.
- **Chill Meeting Structures Fuel Productivity**: Members reflected on the effectiveness of fewer, more relaxed meetings, with one noting that despite the **3.5 hours** scheduled, it’s preferable to hold them earlier in the day.
   - There was a consensus on stacking meetings when necessary, suggesting a focus on efficient use of time rather than excessive schedules.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Vision Llama Hits OpenRouter with Free Endpoint**: The first vision **Llama** is now available on **OpenRouter**, featuring a [free endpoint](https://x.com/OpenRouterAI/status/1839157099747479553). In total, **five new endpoints** have been introduced, powered by multiple providers.
   - Users are encouraged to enjoy the latest features, marked by the celebratory icon 🎁🦙.
- **Gemini Tokenization Simplifies Costs**: OpenRouter will transition to counting **tokens** instead of characters for Gemini models, reducing apparent token counts by a factor of **~4**. This aims to normalize and cut costs for developers.
   - These changes will lead to **doubling** of current prices as they align tokens to a per-token pricing model, set to adjust further after **October 1**.
- **OpenRouter Credits and Invoice Issues**: Users reported difficulties with credit transactions on OpenRouter, noting that transactions might take time to appear after payments are made. A backend delay or provider issues might be causing disruption in viewing transaction history.
   - One user illustrated their eventual receipt of credits, raising concerns about the reliability of the credit system.
- **Llama 3.2 Restrictions for EU Users**: Meta's policy on using their vision models in the EU raises concerns about accessibility and legality for users in that region. Members noted confusion over provider locations and compliance with Meta's rules could pose problems.
   - This has sparked debate on the implications for inference provision related to **Llama 3.2** in Europe.
- **Request for BYOK Beta Participation**: A member inquired about joining the **Bring Your Own Key (BYOK)** beta test. They offered to provide their **email address** via direct message to facilitate participation.
   - The member expressed willingness to share personal contact information to assist with the beta process.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **License Compliance Causes Frustration**: Members discussed compliance with licensing, highlighting that **EU access is blocked** due to disagreements with regulations, leading to frustrations over access limitations.
   - One member humorously remarked that **Mistral is now a meme**, pointing to the absurdity of the situation.
- **OpenAI’s CTO Resignation Sparks Speculation**: The resignation of OpenAI's CTO stirred conversations, with members joking that it leads to speculation about the current state of the company.
   - Concerns were raised about OpenAI's direction, prompting suggestions that internal issues might make for an interesting **Netflix mini-series**.
- **Impressive Capabilities of New Molmo Models**: The recent **Molmo models** received praise for their ability to **point locations in images**, showcasing advancements in open-source development.
   - Members discussed **voice-annotated image training** methods, marking significant progress in integrating multimodal datasets.
- **Tokenizer Lacks Padding Token**: A user raised the issue of a tokenizer missing a padding token during pretraining, which can disrupt processing of variable-length input sequences.
   - *Options provided include setting the pad token to the EOS token or adding a new pad token* using `tokenizer.add_special_tokens({'pad_token': '[PAD]'}).
- **Planning for Llama 3.2 Inference**: An inquiry was made regarding how many **H100 GPUs** are needed to inference **Llama 3.2** with **90 billion parameters**, to prevent out-of-memory errors.
   - The user plans to fetch the **Runpod GPUs** but aims to ensure they can handle the model without needing to delete them due to OOM issues.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Mira Murati exits OpenAI**: [Mira Murati](https://x.com/miramurati/status/1839025700009030027?s=46) announced her departure from OpenAI after 6.5 years, receiving gratitude from Sam Altman for her significant role.
   - The shift raises questions about evolving leadership dynamics within the organization, especially following recent key departures.
- **Meta shows off Orion AR glasses**: Meta launched [Orion](https://about.meta.com/realitylabs/orion), touted as their most advanced **AR glasses**, despite choosing not to sell due to manufacturing challenges.
   - Initial feedback underscores its **aesthetic appeal**, highlighting Meta's ambition to integrate digital and physical experiences.
- **Google's groundbreaking AlphaChip**: Google introduced [AlphaChip](https://x.com/googledeepmind/status/1839306984480231852?s=46), a game-changing microchip that promises to simplify the design of AI models and is accompanied by publicly available model weights.
   - This advancement enhances Google's capabilities in designing state-of-the-art **TPUs** for AI, marking a significant leap in their chip production.
- **Arcade secures $17M for AI tool**: Arcade has raised **$17M** to build a transformative AI product creation platform, claimed to help bring creative visions to life.
   - The project aims to democratize product development, potentially catalyzing innovation in the AI space.
- **GitHub Copilot extends to browsers**: Developers can now access GitHub Copilot's features directly in browsers, positioning it against similar offerings like Sourcegraph's **Cody Chat**.
   - This extension emphasizes the importance of thorough documentation for developers to fully leverage the tool's capabilities.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex seeks engineering talent**: LlamaIndex is hiring a range of **ML/AI** engineering roles, including full-stack positions, in **San Francisco**. Interested candidates can find more details on [Twitter](https://twitter.com/llama_index/status/1839055997291344050).
   - This expansion highlights their growth and commitment to enhancing their engineering team as they tackle upcoming projects.
- **NVIDIA competition with big rewards**: A competition hosted by **NVIDIA** offers over **$10,000** in cash and hardware prizes, including an **NVIDIA® GeForce RTX™ 4080 SUPER GPU**. Developers have until **November 10th** to enter with innovative LLM applications, detailed [here](https://developer.nvidia.com/llamaindex-developer-contest/join).
   - Participants are encouraged to explore the *RAG* applications across diverse domains, with [terms and conditions](https://developer.download.nvidia.com/licenses/nvidia-and-lla) available for review.
- **ReAct Agent message formatting**: Members discussed how to pass user and system messages to **ReAct agents**, emphasizing the need for proper classes and formatting tools. The `ReActChatFormatter` class is essential for structuring chat history appropriately.
   - Clarifying message formats can streamline communication with the agent, ensuring smoother interactions.
- **VectorStoreIndex confusion clarified**: Confusion arose around the **VectorStoreIndex**, leading to a conversation about the connection between indexes and their underlying vector stores. Users confirmed how to access the `vector_store` property without initializing a new vector store.
   - This discussion aimed to eliminate misunderstandings and improve user interactions with indexing.
- **Debate over KnowledgeGraph RAG vs QueryFusion**: A member inquired about utilizing `QueryFusionRetriever` correctly instead of `KnowledgeGraphRAGRetriever` for knowledge indexing. The group deliberated on whether RAG retrievers would better suit their querying needs.
   - The conversation pointed towards potential improvements in selecting the most effective retriever for specific applications.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Langtrace Adds DSPy Experiment Support**: Langtrace has introduced features for running DSPy experiments, offering automatic **capturing** of traces, checkpoints, costs, and **eval score visualizations**.
   - This innovation allows users to create dedicated projects for each pipeline block, enhancing experimentation and optimization.
- **Access to STORM Research Resources**: Members discussed resource links for the STORM paper, confirming its availability on [GitHub](https://github.com/stanford-oval/storm) and [arXiv](https://arxiv.org/abs/2402.14207).
   - The STORM paper explores using LLMs for writing structured articles, which triggered more inquiries into structured knowledge generation.
- **Crafting Agents in DSPy**: A tutorial on building agents in DSPy was shared, highlighting the framework's exploratory nature and existing limitations.
   - The objective of this tutorial is to assist others in learning how to create effective **agent applications** utilizing DSPy.
- **Class Count Optimization**: Discussion on the number of classes in models arose, with one member working with **5 classes** and suggesting that **10 classes** could be beneficial.
   - This conversation emphasized the importance of class count in achieving effective **classification** and model performance.
- **Navigating Subtle Class Distinctions**: The significance of subtle distinctions in class signatures was highlighted, as these nuances complicate description and model clarity.
   - Members agreed that accurately highlighting these **differences** is crucial for improving model performance and understanding.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Yamashi's Comedic Green Card Plea**: In a light-hearted moment, Yamashi humorously asked, *“Spare some green card anyone?”* indicating frustrations with legal and compliance hurdles.
   - He suggested, *“Time to open a fake company in Delaware,”* reflecting on the challenges related to green card acquisition.
- **Access Woes for Llama 3.2**: Members expressed that EU restrictions hinder access to **Llama 3.2**, making direct usage problematic for them.
   - Yamashi noted, *“But I can't use llama 3.2 directly,”* highlighting the barriers faced in accessing the model.
- **Torchtune Struggles with PackedDataset Error**: A member encountered a **PackedDataset** error associated with sequence length limits, referencing [GitHub issue #1689](https://github.com/pytorch/torchtune/issues/1689).
   - They offered a potential fix and showed willingness to submit a PR after evaluating the testing requirements.
- **MetaAI Access Restrictions for EU Users**: Members raised concerns about login issues for **MetaAI**, stating EU users are unable to access their accounts.
   - Yamashi remarked, *“Ah checks out I am unable to login on meta ai,”* pointing out these connectivity challenges.
- **Excitement Over Visual Question Answering Datasets**: A member shared enthusiasm over newly available datasets for visual question answering linked to a collection on [Hugging Face](https://huggingface.co/datasets?task_categories=task_categories:visual-question-answering&sort=trending).
   - They noted the potential for these datasets in finetuning applications.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **MOToMGP Pass Manager Error Tackled**: The team is managing the **'failed to run the MOToMGP pass manager'** error and invites user feedback on **Max / Mojo** issues for potential improvement.
   - Members are encouraged to share grievances or suggestions related to the pass manager for a more streamlined experience.
- **Interest in Mojo/MAX Branded Backgrounds**: A poll gauged interest for **Mojo / MAX** branded desktop backgrounds with themes like adorable Mojo flames and MAX astronauts.
   - Users participated by emoji voting with a **yes** or **no**, indicating their preference for these creative designs.
- **Verification Bot Returns for Security**: The verification bot mandates members to click 'I'm human ✅' to maintain community security and prevent spam.
   - Unverified members will face posting limitations in designated channels, encouraging better adherence to the verification process.
- **Mojo Compiles Directly to Machine Code**: A member clarified that **Mojo** compiles directly to machine code rather than creating **.pyc** files, unlike Python.
   - *“.pyc is bytecode cache, Mojo compiles directly to machine code.”* emphasizes Mojo's efficiency in compiling execution paths.
- **MAX API User Feedback Requested**: Feedback is sought from users of the **MAX API**, especially regarding frustrations and potential improvements.
   - The member encourages a friendly exchange of thoughts on their API experience, including any suggestions for enhancement.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **LLM Miscommunication on Availability**: When asked a question, the **LLM** sometimes responds with *'I'm sorry, but I don't know'* despite retrieving relevant **source documents**.
   - The member suggested that document retrieval should be conditional on the LLM having useful information to avoid confusion.
- **Unnecessary Source Documents Confusion**: The same member criticized that **source documents** are returned even when the LLM indicates there's no relevant information.
   - They noted that while most responses are satisfactory, receiving unnecessary documents in negative responses can be misleading.
- **Debugging Tools Dilemma**: A participant questioned the use of **debugging** tools like **Langsmith**, which the original poster declined due to privacy issues.
   - Alternatives such as **Langfuse** were proposed to allow monitoring without compromising sensitive data.
- **Call for Code Clarity**: A request was made for code examples to clarify the issues faced by the original poster regarding their LLM interactions.
   - The original poster agreed to share examples the next day, highlighting a commitment to collaborative troubleshooting.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Generative AI Bubble faces scrutiny**: A member raised concerns that the **Generative AI** industry, particularly **ChatGPT**, is nearing collapse due to recent key departures, including **Mira Murati**.
   - They referenced an alarming newsletter claiming the generative AI boom is **unsustainable**, risking major tech reputations and public perception.
- **PhD students find a home in Cohere**: A new member highlighted their interest in staying updated on AI discussions as they approach the end of their PhD, making **Cohere** their go-to resource.
   - This shows the community's value for academics looking to engage with cutting-edge AI topics.
- **Question on Avg Hard Negatives Computation**: A user inquired about how the **'Avg Hard Negatives per Query'** is calculated, noting their dataset contains less than **10%** hard negatives.
   - Cohere clarified that they do not add negatives behind the scenes and suggested verifying the data quality.
- **Model's Performance Post-Training**: Following the training process, a user reported that the model performed only slightly better than the **default English v3 reranker**.
   - They speculated that the **quality of the data** might be a contributing factor to this underwhelming performance.
- **Community shows warmth to newcomers**: Multiple members actively welcomed newcomers and encouraged them to ask questions about **Cohere**, fostering a welcoming atmosphere.
   - This illustrates the community's commitment to collaboration and support in AI learning.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Proof for Arbitrary View Mergeability Released**: A proof of **arbitrary view mergeability** without masks or reshapes has been shared on GitHub, detailing key insights on view management in **Tinygrad**. You can find the [proof here](https://github.com/pschilliOrange/Tinygrad-view-merging-proof/blob/8672f35c1147798c8e9a78bfab28b9ff79bf45e6/Proof%20for%20when%20a%20new%20view%20must%20be%20appended.pdf).
   - This document accompanies a solid overview of the challenges in current view merging techniques.
- **Tinygrad Training Bottlenecks Identified**: Users reported that **Tinygrad** training is hindered by poor performance, even with a **4090 GPU**, due to issues in the sampling code rather than training speed. They clarified that the output quality suffered from implementation errors, not the hardware itself.
   - This highlights the need for improved debugging and functionality in the **sampling logic**.
- **Metal Double Precision Error Troubles**: A user experienced a Metal error related to **double precision**, which arose because NumPy defaults to double values. They resolved this by converting tensors to **float32**, though new buffer issues surfaced thereafter.
   - The conversation underscores the challenges of adapting Tinygrad for **Metal** backend specifics.
- **Tinygrad vs PyTorch Showdown**: There's active discussion concerning the strengths of **Tinygrad** as a faster alternative to **PyTorch**, particularly in relation to working directly with **CUDA**. While Tinygrad compiles to CUDA, PyTorch benefits from highly optimized CUDA kernels.
   - This distinction points to trade-offs between customizability and pre-optimized performance.
- **Undiscovered Optimization in Tinygrad**: Members noted that Tinygrad’s **custom kernel generation** offers greater optimization opportunities compared to PyTorch’s fixed kernels. This flexibility could significantly impact overall performance in specific applications.
   - The discussion centers around exploiting these features for tailored performance gains.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **LLaMA 3.2 Vision excels in Image Captioning**: Members noted that **LLaMA 3.2 Vision 90B** is highly capable for **image captioning**, with the **11B** version also gaining traction.
   - *One member humorously suggested captioning the entire LAION dataset* to showcase its potential.
- **OpenAI's Function Calling API under scrutiny**: A member inquired about how **OpenAI's function calling API** operates, questioning if it relies on a fine-tuned model or output checks.
   - This reflects ongoing interest in the **intricacies of API design and performance enhancements**.
- **Free access to LLaMA 3.2 Vision announced**: TogetherCompute partnered with **AI at Meta** to provide **LLaMA 3.2 11B Vision for free** for developers to experiment with multimodal AI.
   - They offer a free model endpoint at [this link](https://api.together.ai/playground/chat/meta-llama/Llama-Vision-Free) with paid options for enhanced performance.
- **MaskBit reshapes image generation techniques**: **MaskBit** introduces embedding-free image generation through **bit tokens**, improving upon the traditional **VQGAN** model.
   - The model achieves a **FID of 1.52** on ImageNet with just **305M parameters**, showing embedding-free approaches' effectiveness.
- **MonoFormer simplifies generation processes**: **MonoFormer** presents a unified transformer architecture managing both **autoregression** and **diffusion** in generation.
   - This model maintains competitive image generation and text output, with further details available at their [project page](https://monoformer.github.io/).



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Quiz 3 Questions Spark Confusion**: A member expressed confusion over a **Quiz 3 question** that wasn't covered in the presenter's explanation of constrained and unconstrained flows. Another pointed out that the information was indeed in the slides, clarifying the quiz content.
   - This exchange highlights the ongoing challenges of aligning quiz materials with lecture content.
- **RAG Model Struggles with Multimodal Data**: Concerns were raised about the **RAG capabilities** of the latest models, especially regarding performance with multimodal data like text, tables, and images. Notably, **Claude 3** excelled in explaining flow diagrams.
   - This points to the need for models to adapt better to diverse data types for improved functioning.
- **Agentic RAG Projects Take Shape**: A member shared their **ccmp_ai project**, an unconstrained RAG model offering new terminology, referred to as an **agentic RAG** with dynamic problem domain expansion. This highlights the innovation in project conceptualization among peers.
   - Another member found the terminology quite useful, sparking interest in further exploration of the model's applications.
- **Summary of Healthcare Multi-Agent Systems Research**: The study titled [AgentClinic: A Multimodal Agent Benchmark](https://open.substack.com/pub/yanpan0508/p/agentclinic-a-multimodal-agent-benchmark?r=ad7en) focuses on healthcare multi-agent systems, analyzing methodologies and findings. It emphasizes the collaborative potential of these systems in healthcare, enhancing AGI applications.
   - Such research informs future developments in multi-agent systems and reinforces their significance in AI.
- **Yvaine’s Substack Launch**: Yvett's Substack, 'Embracing AGI', aims to engage with the community on advancements in the AI field, particularly in healthcare. Her recent launch includes discussions emphasizing the role of AGI in healthcare contexts.
   - This initiative underlines the importance of community-driven knowledge sharing in the rapidly evolving AGI domain.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Llama 3.2 Fails to Impress**: After testing **Llama 3.2** 90b, a member expressed disappointment, stating it does not compare favorably to **Llama 3.1** 70b. They referenced a [YouTube video](https://www.youtube.com/watch?v=_MQVHyEeER4) titled 'Llama-3.2 (1B, 3B, 11B, 90B) : The WORST New LLMs EVER!?' that details their findings.
   - The video critiques the shortcomings of the new model across various metrics, leading to discussions about its practical applications.
- **Open Interpreter Fails to Count Files**: A member reported that when using the **3b** model with **Open Interpreter** to count files on their desktop, it **failed** to execute the task. This raised concerns about the reliability of the model in handling basic tasks.
   - The community is questioning how such limitations could impact broader use cases in development.
- **Excitement for Tech Week SF Meetup**: One user expressed excitement about attending **Tech Week** in San Francisco and suggested meeting up to high-five. This highlights the community's enthusiasm for networking and connecting during tech events.
   - Members are keen to discuss their projects and share insights during this high-energy event.
- **Challenges with NERD Task**: A member described a **NERD** task focused on linking text to **wiki** entries for individuals mentioned in news articles. This task is seen as complex due to the intricacies involved in extracting and matching relevant information.
   - The conversation emphasized the need for improved methodologies to tackle such challenging tasks in text analysis.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Seeking Alternatives to partition_pdf**: A member requested suggestions for alternatives to **unstructured 'partition_pdf'** for better extraction of images and tables from PDFs.
   - *They are looking for a more effective tool for this specific task.*
- **Reminder on Channel Etiquette**: Another member emphasized that posting the same question in multiple channels will be considered **spam** and took action by deleting duplicates.
   - *This reminder highlights the importance of maintaining order within the channel.*



---



## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord

- **Promotion Concerns**: A member expressed frustration questioning how a certain topic is not considered **promotion**, implying some merit in scrutiny.
   - This comment highlights ongoing debates within the community regarding the boundaries of promotion in discussions.
- **Lack of Clarity in Discussions**: The discussion lacked context as only one message was noted, leaving ambiguity on the subject matter being critiqued.
   - Members often feel that clearer guidelines on promotion could prevent misunderstandings like this.



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Mozilla AI Featured in Nature**: Mozilla AI and its initiatives were spotlighted in Nature's article, *“Forget ChatGPT: Why researchers now run small AIs on their laptops.”* The discussion centered on the **growing trend of locally-run AI models** that enhance user capabilities.
   - The article included insights from Mozilla's head of open-source AI, emphasizing the shift towards empowering individual users with autonomous models.
- **LLMs Gain System Versatility**: A notable project showcased in the article aims to facilitate **Large Language Models (LLMs)** running across multiple systems, reflecting their adaptability.
   - This advancement underscores a leap in making powerful AI tools available for diverse environments, bridging gaps between different tech infrastructures.
- **Continue Tool's Rising Popularity**: The **Continue tool**, highlighted in a recent talk, has been recognized for its utility in **AI-assisted coding**, boosting developer productivity.
   - This endorsement signals its increasing importance within the AI engineering community as a resource for enhancing coding efficiency.
- **Access Nature's Full Insight**: Interested readers can access the detailed analysis by following the [full article here](https://discord.com/channels/1089876418936180786/1288648264069025883/1288648264069025883).
   - This direct link serves as an essential resource for further understanding the innovations discussed in the community.



---



## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **User Confused About Function Calling Evaluation**: A user raised concerns over the **function calling evaluation** in the codebase, specifically asking if they could submit their own **custom evaluation dataset** alongside an API/LLM.
   - They noted a **lack of clarity** on how to integrate their dataset composed of **<prompt>, <llm_response>, <ideal response>** for effective error breakdown.
- **Demand for Custom Dataset Error Insights**: The same user expressed a desire for a tool that could **analyze their dataset** and deliver insights similar to those outlined in the [BFCL metrics](https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html#metrics).
   - This indicates a clear need for functionality that enhances understanding of errors within custom datasets.



---


The **LLM Finetuning (Hamel + Dan) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **DiscoResearch Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1288925046416343071)** (1 messages): 

> - `Tau Innovations`
> - `Gemini Object Detection`
> - `LLama 3.2 Review`
> - `Reasoning Models for Software Engineering`
> - `Custom Arabic Semantic Search Model` 


- **Tau's Latest Innovations Uncovered**: A new [article](https://dev.to/p3ngu1nzz/from-data-expansion-to-embedding-optimization-taus-latest-innovations-4h3n) details innovations in data expansion and embedding optimization by **P3ngu1nzz**.
   - They also shared a [training run](https://huggingface.co/p3nGu1nZz/Tau/tree/main/results/tau_agent_D10_100M) of **Tau** consisting of **100 million steps**.
- **Gemini Makes Waves in Object Detection**: **Gemini's object detection** space has been launched and can be explored [here](https://huggingface.co/spaces/saq1b/gemini-object-detection).
   - This new space aims to enhance the capabilities of AI in object detection applications, showcasing the latest advancements.
- **Exploring LLama 3.2 Effectiveness**: A member posed the question, *
- **Need for Reasoning Models in Software Engineering**: A discussion emerged about whether daily software engineering tasks require [reasoning models](https://huggingface.co/blog/onekq/daily-software-engineering-work-reasoning-models).
   - This prompts a consideration of how new models can integrate into traditional software processes for improved efficiency.
- **Building Custom Arabic Search Model**: An article discusses [developing a custom Arabic semantic search model](https://huggingface.co/blog/Omartificial-Intelligence-Space/building-custom-arabic-semantic-search-model) using **Arabic Matryoshka embeddings**.
   - This model leverages **sentence transformers** for robust retrieval-augmented generation (RAG) applications in Arabic contexts.



**Link mentioned**: <a href="https://youtu.be/yWxWJfQ3Tcg)">Is LLama 3.2 any good ?</a>: Let&#39;s try to see if Llama 3.2 is actually any good

  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1288578069032079464)** (491 messages🔥🔥🔥): 

> - `OpenAI developments`
> - `Hugging Face models`
> - `Machine learning and neuroscience`
> - `Llama 3.2 updates`
> - `AI deployment solutions` 


- **OpenAI's Leadership Changes and Impact**: Recent departures of key personnel, including Mira Murati, from OpenAI have sparked discussions about the company's future and whether it will lead to new startups focused on safer AI.
   - Many believe that the internal disagreements could lead to a decline in OpenAI's innovative capabilities, given the expertise that has left.
- **Hugging Face Reaches 1 Million Models**: The Hugging Face platform has achieved a significant milestone by surpassing 1 million models, celebrating community contributions towards this goal.
   - This growth reflects the increasing popularity and demand for diverse machine learning models across various applications.
- **Exploring Llama 3.2 Model**: The new Llama 3.2 model brings smaller variants and enhanced capabilities, including the ability to run efficiently on devices like the iPhone 15 Pro.
   - However, users have noted varying performance compared to models like ChatGPT, especially in specific tasks requiring unique knowledge.
- **Integrating AI with Human Brain Research**: Members discussed the intersection of machine learning and neuroscience, with insights shared about past projects dealing with brain-computer interfaces and neural activities.
   - The conversation highlighted the complexity of understanding conscious vs. unconscious thoughts, suggesting machine learning could aid in decoding such processes.
- **AI Deployment Strategies**: Queries about deploying machine learning solutions in production environments led to recommendations for using Hugging Face's inference deployment services.
   - Participants requested guidance on best practices for managing multiple models for specific tasks, reflecting the challenges of integrating AI in SaaS applications.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/miramurati/status/1839025700009030027">Tweet from Mira Murati (@miramurati)</a>: I shared the following note with the OpenAI team today.</li><li><a href="https://x.com/awnihannun/status/1839330067039887622">Tweet from Awni Hannun (@awnihannun)</a>: Llama 3.2 1B in 4-bit runs at ~60 toks/sec with MLX Swift on my iPhone 15 pro.  It&#39;s quite good and easily runs on-device:</li><li><a href="https://ui.endpoints.huggingface.co/">Inference Endpoints - Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/collections/Joseph717171/gguf-llama-32-instruct-oq8-0-f32ef32iq4-k-q8-0-iquants-66f49cf761597948c6fa789d">GGUF Llama-3.2-Instruct-OQ8_0-F32.EF32.IQ4_K-Q8_0 IQuants - a Joseph717171 Collection</a>: no description found</li><li><a href="https://huggingface.co/spaces/discord-community/LevelBot/discussions/23">discord-community/LevelBot · patched issue where you could earn infinite exp if you send messages in bot dms, other changes are commentated</a>: no description found</li><li><a href="https://huggingface.co/SandLogicTechnologies/Llama-3.2-3B-Instruct-GGUF">SandLogicTechnologies/Llama-3.2-3B-Instruct-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct">meta-llama/Llama-3.2-11B-Vision-Instruct · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/docs/transformers/v4.44.2/main_classes/quantization">Quantization</a>: no description found</li><li><a href="https://www.scientificamerican.com/article/there-is-no-such-thing-as-conscious-thought/#:~:text=We%20are%20not%20simply%20puppets%20manipulated%20by%20our%20unconscious%20thoughts,">There Is No Such Thing as Conscious Thought | Scientific American</a>: no description found</li><li><a href="https://x.com/GaryMarcus/status/1839037069307555905">Tweet from Gary Marcus (@GaryMarcus)</a>: Wait what? Murati is leaving, Schulman left, Karpathy left, Ilya left, Leike left, Brockman is on leave, perhaps a dozen others left,  GPT-5 hasn’t dropped, Sora hasn’t shipped, the company had an ope...</li><li><a href="https://github.com/Deadsg/DQNAgent/tree/main/Q_Layered_Network">DQNAgent/Q_Layered_Network at main · Deadsg/DQNAgent</a>: Contribute to Deadsg/DQNAgent development by creating an account on GitHub.</li><li><a href="https://www.technologyreview.com/2024/04/19/1091505/companies-brain-computer-interfaces/">Beyond Neuralink: Meet the other companies developing brain-computer interfaces</a>: Companies like Synchron, Paradromics, and Precision Neuroscience are also racing to develop brain implants</li><li><a href="https://huggingface.co/docs/transformers/v4.44.2/llm_tutorial_optimization">Optimizing LLMs for Speed and Memory</a>: no description found</li><li><a href="https://github.com/xai-org/grok-1">GitHub - xai-org/grok-1: Grok open release</a>: Grok open release. Contribute to xai-org/grok-1 development by creating an account on GitHub.</li><li><a href="https://www.hetzner.com/dedicated-rootserver/matrix-gpu/">Dedicated Server Hosting </a>: no description found</li><li><a href="https://www.biorxiv.org/content/10.1101/2020.07.01.183384v1.full">High-performance brain-to-text communication via imagined handwriting</a>: Brain-computer interfaces (BCIs) can restore communication to people who have lost the ability to move or speak. To date, a major focus of BCI research has been on restoring gross motor skills, such a...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1288583802490454128)** (5 messages): 

> - `AGENTIC RL SWARM`
> - `Legal Reasoning Models`
> - `RAG Tool Utilization`
> - `RL Models in Legal Field`
> - `CUDA and FP8 Training` 


- **Building AGENTIC RL SWARM for Legal Reasoning**: A member is constructing an **AGENTIC RL SWARM** setup tailored for handling long context **legal tasks** involving an agentic breakdown of major cases.
   - This will incorporate tools like **RAG** with **graphrag, retriever, reranker,** and **colbert v2** to enhance **contextual retrieval** and functionality while evaluating the outputs rigorously.
- **Exploring RL Models for Legal Agents**: Another member is delving into **RL models** to identify which are most effective for creating agency in legal field case construction agents.
   - This exploration will focus on how these models can optimize legal reasoning and task management for complex cases.
- **CUDA and FP8 Learning**: A member reported gains in their learning journey, specifically working with **CUDA** while experimenting with a **7b FP8** model.
   - This reflects a hands-on approach to mastering performance optimization techniques in machine learning workflows.


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1288798457376673833)** (3 messages): 

> - `Llama 3.2 Release`
> - `OpenAI’s New ChatGPT Capabilities`
> - `New Research Paper` 


- **Llama 3.2 Released with Multimodal Features**: Meta has launched **Llama 3.2** available in four sizes (90B, 11B, 3B, 1B) aimed at the medical domain, with **Llama-3.1 70B** unexpectedly outperforming it.
   - The **Meta-Llama-3.1-70B-Instruct** topped benchmarks with an **84% average score**, especially excelling in **MMLU College Biology** at **95.14%**.
- **Analyzing OpenAI’s New ChatGPT Features**: A highly recommended [YouTube video](https://www.youtube.com/watch?v=QDfE0HwDBo8) titled **'OpenAI’s New ChatGPT: 7 Incredible Capabilities!'** provides a great explanation of the new features.
   - Viewers can explore **Lambda’s GPU Cloud** and also play the *Tron game* through provided links.
- **Recent Research Paper on AI**: A new research paper can be accessed at [ACL Anthology](https://aclanthology.org/2024.acl-long.43.pdf) and covers recent findings in the field.
   - This paper adds to the ongoing discussions and developments within the AI research community.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/aadityaura/status/1839233111927750830">Tweet from Aaditya Ura (@aadityaura)</a>: Analysis for Llama-3.2  90B, 11B, 3B, and 1B for Medical & Healthcare Domain 🩺🧬💊  Interesting observation:  - Llama-3.1 70B Outperforms Llama-3.2 90B - Meta-Llama-3.2-90B-Vision Instruct and Base a...</li><li><a href="https://www.youtube.com/watch?v=QDfE0HwDBo8">OpenAI’s New ChatGPT: 7 Incredible Capabilities!</a>: ❤️ Check out Lambda here and sign up for their GPU Cloud: https://lambdalabs.com/paperPlay the Tron game: https://agpallav.com/tron.htmlSources:https://www.y...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1288580703239540908)** (142 messages🔥🔥): 

> - `AI Model Comparison Tool`
> - `Ollama Integration`
> - `Hugging Face Models`
> - `Community Feedback`
> - `Future Enhancements` 


- **Launch of AI Model Comparison Tool**: A user shared a new website, [Countless.dev](https://countless.dev/), that allows comparisons of various AI models and their pricing, built as a personal project.
   - The developer aims to keep the tool updated and gather feedback for further improvements.
- **Community Response to the Overview**: Members expressed appreciation for the tool, with one user noting, *"damn that is a GREAT overview"*.
   - Discussions emerged around the practicality and value of the tool, especially in relation to price updates.
- **Discussion on Ollama's Role**: Questions arose about the inclusion of Ollama as a locally hosted model, prompting discussions on whether it adds value alongside data from Hugging Face.
   - Suggestions were made to possibly remove it from the main listing but to keep it available for users wanting to explore all platforms.
- **Future Enhancements for User Experience**: The developer indicated plans to add more features such as model ratings and links for better comparison across platforms.
   - Community members highlighted the importance of linking models for a seamless comparison experience across different hosting platforms.
- **Integration of OpenLLM Leader Features**: Discussion pointed towards the potential to integrate OpenLLM leaderboards with the comparison tool for enhanced user insights.
   - The consensus was that proper model linking would be essential for any robust integration involving multiple hosting platforms.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/qamarsidd/FreeTranscriptMaker">FreeTranscriptMaker - a Hugging Face Space by qamarsidd</a>: no description found</li><li><a href="https://www.youtube.com/@BatCountryEnt/videos#:~:text=Share%20your%20videos%20with%20friends,%20family,%20and%20the%20world">BatCountryEnt</a>: no description found</li><li><a href="https://huggingface.co/p3nGu1nZz/Kyle-b0a">p3nGu1nZz/Kyle-b0a · Hugging Face</a>: no description found</li><li><a href="https://youtu.be/uyxRV1PP12Y">Llama3.2 looks at my screen 24/7 and send me an email summary of my day</a>: https://screenpi.pe</li><li><a href="https://youtu.be/yWxWJfQ3Tcg">Is LLama 3.2 any good ?</a>: Let&#39;s try to see if Llama 3.2 is actually any good</li><li><a href="https://countless.dev/">Countless.dev | AI Model Comparison</a>: Compare AI models easily! All providers in one place.</li><li><a href="https://x.com/ahmetdedeler101/status/1839313737561551359">Tweet from Ahmet ☕ (@ahmetdedeler101)</a>: Introducing http://Countless.dev - A web app I built fully using @v0 and @cursor_ai in just a few days (as a 17 y/o)  Compare every AI model, examine the price, and find the best one for your use case...</li><li><a href="https://github.com/p3nGu1nZz/Tau/blob/dev-pca-optimization-script/MLAgentsProject/Scripts/optimizer.py">Tau/MLAgentsProject/Scripts/optimizer.py at dev-pca-optimization-script · p3nGu1nZz/Tau</a>: Tau LLM made with Unity 6 ML Agents. Contribute to p3nGu1nZz/Tau development by creating an account on GitHub.</li><li><a href="https://github.com/mediar-ai/screenpipe/blob/main/examples/typescript/pipe-email-daily-log/README.md">screenpipe/examples/typescript/pipe-email-daily-log/README.md at main · mediar-ai/screenpipe</a>: Library to build personalized AI powered by what you&#39;ve seen, said, or heard. Works with Ollama. Alternative to Rewind.ai. Open. Secure. You own your data. Rust. - mediar-ai/screenpipe</li><li><a href="https://github.com/p3nGu1nZz/oproof/blob/main/oproof/main.py">oproof/oproof/main.py at main · p3nGu1nZz/oproof</a>: Validate prompt-response pairs using Ollama and Python. - p3nGu1nZz/oproof</li><li><a href="https://huggingface.co/p3nGu1nZz/Tau/tree/main/results/tau_agent_D10_100M">p3nGu1nZz/Tau at main</a>: no description found</li><li><a href="https://github.com/p3nGu1nZz/Tau">GitHub - p3nGu1nZz/Tau: Tau LLM made with Unity 6 ML Agents</a>: Tau LLM made with Unity 6 ML Agents. Contribute to p3nGu1nZz/Tau development by creating an account on GitHub.</li><li><a href="https://github.com/ytdl-org/youtube-dl">GitHub - ytdl-org/youtube-dl: Command-line program to download videos from YouTube.com and other video sites</a>: Command-line program to download videos from YouTube.com and other video sites - ytdl-org/youtube-dl
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1288767971204010004)** (5 messages): 

> - `Multimodal Models`
> - `Contrastive Learning`
> - `Visual Learning`
> - `3D/4D Understanding` 


- **VLMs Signal the Future**: Today saw the release of **two multimodal models**, highlighting a shift towards **Vision-Language Models (VLMs)** as a key direction in AI.
   - *It's becoming clear that VLMs are the way forward* given current advancements.
- **Contrastive Learning Paves the Way**: In the discussion, it was noted that **multimodality** has always been the goal but faced challenges due to difficulties in correlating different modalities.
   - Models like **CLIP** and techniques such as **contrastive learning** have provided new pathways to integrate multimodal data.
- **Visual Learners Favor VLMs**: A member emphasized that humans tend to be **visual learners**, which contributes to the growing popularity of **VLMs and MLLMs** in current AI discussions.
   - There is speculation about future developments focusing on **3D and 4D understanding** rather than solely on language or 2D images.
- **Skepticism About 3D and 4D Progress**: While discussions on **3D understanding** emerged, members voiced skepticism regarding its feasibility, stating that rendering **3D data** is not straightforward.
   - One member remarked that *4D understandings are a bit too far* for now, highlighting ongoing challenges.


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1288891994734723072)** (1 messages): 

> - `Word Injection in Embeddings`
> - `Parameter Freezing Techniques` 


- **Word Injection Explained**: The article discusses how real-life learning is gradual and highlights that embedding models require a vocabulary extension for new words, necessitating model fine-tuning and reindexing of embeddings.
   - *We rarely experience a mind-blowing moment that makes us rethink everything we know*.
- **Challenges of Fine-tuning Embeddings**: After extending vocabulary and fine-tuning models, previously computed embeddings lose validity due to changed parameters, which can complicate processing millions of documents.
   - It raises an important inquiry about freezing parameters in the model while allowing updates to new input embeddings.
- **Parameter Freezing Puzzlement**: There is uncertainty about how to freeze existing input embeddings while allowing new input embeddings to be trainable, indicating potential complications in managing parameters.
   - The concern revolves around whether all embeddings can be stored in a single parameter without affecting the training process.



**Link mentioned**: <a href="https://www.kacperlukawski.com/posts/word-injection/?_gl=1*1a2g1t4*_ga*MTgzMTMyNjgzMS4xNzI3MDkxMjE3*_ga_SL41LGM35F*MTcyNzM2NjA4MS4yLjEuMTcyNzM2NjA4NC4wLjAuMA..">Old dog, new tricks: Word Injection in the text embedding models</a>: In real life, we rarely experience a mind-blowing moment that makes us rethink everything we know. Present experiences are rather built on top of past ones, and gathering knowledge is a gradual proces...

  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1288597400281288769)** (2 messages): 

> - `Colab Free Tier Performance`
> - `Default Training Logic in Diffusers`
> - `MSE Loss Computation in Diffusion Models` 


- **Colab Free Tier Performance Potential**: A user noted that while discussing performance criteria, one can run almost any model in the **free tier of Google Colab**.
   - This raises questions about the specifics of 'relatively performant' as a descriptor for model execution in this environment.
- **Training Diffusion Models With UNet2D**: A question arose regarding the **default training logic** in the Hugging Face diffusers tutorial, particularly around unconditional image generation.
   - The tutorial guides creating a UNet2D model aimed at generating butterfly images from the Smithsonian dataset, with links provided for more resources.
- **MSE Loss Confusion in Diffusion Models**: A user questioned why **MSE loss** is computed on random Gaussian noise versus the residual rather than incorporating timestep interactions.
   - They emphasize the model's focus on learning the residual, critical for understanding the noise to be subtracted from the current noisy image at each timestep.



**Link mentioned**: <a href="https://huggingface.co/docs/diffusers/en/tutorials/basic_training">Train a diffusion model</a>: no description found

  

---



### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1288580232340836406)** (486 messages🔥🔥🔥): 

> - `Llama 3.2 Fine-Tuning`
> - `NVIDIA GPU Specifications`
> - `LLM Training Techniques`
> - `Marketing Data Datasets`
> - `OLLaM and LM Studio Issues` 


- **Updates on Llama 3.2 Fine-Tuning Support**: The Unsloth team announced that Llama 3.2 fine-tuning has been fixed and is now available, with vision model support anticipated soon. QLoRA configurations allow models to fit under specific VRAM minimums, enhancing usability.
   - Multiple users have successfully fine-tuned models with substantial VRAM savings and are waiting for upcoming features, particularly concerning images.
- **NVIDIA's New GPU Releases**: NVIDIA's upcoming GPUs, the RTX 5090 and RTX 5080, have been leaked, showing increased CUDA cores but variable VRAM offerings. This raises concerns among current GPU owners regarding performance necessity versus upgrade costs.
   - Discussion ensued around the surrender of VRAM in favor of higher speed specs, highlighting potential disadvantages for content creators and gamers.
- **Strategies for LLM Training**: An inquiry into the use of LLMs for marketing analysis and ad enhancement sparked discussions regarding effective data sets and training methods. Varied approaches were suggested, from fine-tuning the same model to training multiple distinct models.
   - Users exchanged ideas on building optimal datasets for training while seeking best practices for creating models capable of handling specific marketing tasks.
- **OLLaM and LM Studio Loading Issues**: A user reported issues loading models in OLLama and LM Studio, receiving errors related to tokenizer merges. Guidance was requested from experienced users while troubleshooting the issue.
   - Community members helped identify the underlying causes of the errors and offered potential solutions for model loading discrepancies.
- **WeightWatcher Tool Discussion**: WeightWatcher was discussed as a valuable tool for model diagnostics, providing insight into weight distributions and assisting in making informed training decisions. Users expressed curiosity about its application in model analysis.
   - The integration of WeightWatcher with various training techniques can help optimize model performance, allowing for better management of training metrics.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://rocm.docs.amd.com/projects/radeon/en/latest/docs/compatibility/wsl/wsl_compatibility.html">Compatibility matrices (WSL) &#8212; Use ROCm on Radeon GPUs</a>: no description found</li><li><a href="https://x.com/Unsloth">Tweet from undefined</a>: no description found</li><li><a href="https://x.com/UnslothAI/status/1839036956245897685">Tweet from Unsloth AI (@UnslothAI)</a>: Llama 3.2 versions including GGUF&#39;s + bnb 4 bit versions + reuploaded versions are now on @HuggingFace!  See all versions of Llama 3.2 here: https://huggingface.co/collections/unsloth/llama-32-66f...</li><li><a href="https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/">Llama 3.1 | Model Cards and Prompt formats</a>: Llama 3.1 - the most capable open model.</li><li><a href="https://huggingface.co/Joseph717171/Llama-3.2-1B-Instruct-OQ8_0.EF32.IQ4_K-Q8_0-GGUF/resolve/main/Llama-3.2-1B-Instruct-OF32.EF32.IQ8_0.gguf">no title found</a>: no description found</li><li><a href="https://colab.research.google.com/drive/15OyFkGoCImV9dSsewU1wa2JuKB4-mDE_?usp=sharing#scrollTo=juQiExuBG5Bt">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1aqlNQi7MMJbynFDyOQteD2t0yVfjb9Zh?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://huggingface.co/collections/meta-llama/llama-32-66f448ffc8c32f949b04c8cf">Llama 3.2 - a meta-llama Collection</a>: no description found</li><li><a href="https://tenor.com/view/why-whyyy-neden-a%C4%9Flamak-%C3%BCzg%C3%BCn-gif-19603232">Why Whyyy GIF - Why Whyyy Neden - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1fpet4v/llama_32_multimodal_">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.reddit.com/r/Loc">Reddit - Dive into anything</a>: no description found</li><li><a href="https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-GGUF">unsloth/Llama-3.2-1B-Instruct-GGUF · Hugging Face</a>: no description found</li><li><a href="https://gist.github.com/fullstackwebdev/81e64e8faca496e5390d09a4756d8db4">llama32_3b_failwhale.py</a>: GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://tenor.com/view/no-gif-6533142189269812111">No GIF - No - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://x.com/UnslothAI/status/1839340091241869698">Tweet from Unsloth AI (@UnslothAI)</a>: You can finetune Llama-3.2 for free on Colab now!  Unsloth makes finetuning 2x faster and uses 60% less VRAM with no accuracy degradation.  Llama 3.2 (1B) QLoRA fits on a 4GB GPU, and (3B) fits on 7GB...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1fpeb5g/llama_32_versions_gguf_4bit_bnb_more/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1fpet4v/llama_32_multimodal_ggufs_4bit_bitsandbytes/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://huggingface.co/collections/Joseph717171/gguf-llama-32-instruct-oq8-0-f32ef32iq4-k-q8-0-iquants-66f49cf761597948c6fa789d">GGUF Llama-3.2-Instruct-OQ8_0-F32.EF32.IQ4_K-Q8_0 IQuants - a Joseph717171 Collection</a>: no description found</li><li><a href="https://chromewebstore.google.com/detail/page-assist-a-web-ui-for/jfgfiigpkhlkbnfnbobbkinehhfdhndo">Page Assist - A Web UI for Local AI Models - Chrome Web Store</a>: Use your locally running AI models to assist you in your web browsing.</li><li><a href="https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct#instruction-tuned-models),">meta-llama/Llama-3.2-1B-Instruct · Hugging Face</a>: no description found</li><li><a href="https://datta0.substack.com/p/ai-unplugged-20-score-self-correction">AI Unplugged 20: SCoRE Self Correction via RL, OpenAI o1 models, Qwen 2.5 coder, Spectrum Fine Tuning.</a>: Insights over information</li><li><a href="https://videocardz.com/newz/nvidia-geforce-rtx-5090-and-rtx-5080-specs-leaked">NVIDIA GeForce RTX 5090 and RTX 5080 specs leaked - VideoCardz.com</a>: GeForce RTX 5090 to feature 21760 CUDA cores, 32GB GDDR7 memory and 600W, RTX 5080 gets 16GB VRAM Coming from Kopite7kimi himself.  One of the most reliable NVIDIA leakers has now confirmed the specs ...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1fq0f6m/llama_32_1b_4gb_vram_finetuning_2x_faster_colab/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/issues/1040#issuecomment-2377762522">What is the right way to load Qwen2&#39;s chat interface? · Issue #1040 · unslothai/unsloth</a>: I get this error: chat_template, stop_word, yes_map_eos_token, ollama_modelfile = CHAT_TEMPLATES[chat_template] ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^ KeyError: &#39;Qwen2-1.5B&#39; From this code: def test_un...</li><li><a href="https://weightwatcher.ai/">WeightWatcher: Data-Free Diagnostics for Deep Learning</a>: no description found</li><li><a href="https://github.com/CalculatedContent/WeightWatcher">GitHub - CalculatedContent/WeightWatcher: The WeightWatcher tool for predicting the accuracy of   Deep Neural Networks</a>: The WeightWatcher tool for predicting the accuracy of   Deep Neural Networks - CalculatedContent/WeightWatcher
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[announcements](https://discord.com/channels/1179035537009545276/1179039782681202829/1288908901181423707)** (1 messages): 

> - `Llama 3.2 Release`
> - `New Notebooks`
> - `Model Uploads` 


- **Unsloth officially supports Llama 3.2**: Unsloth now integrates **Llama 3.2**'s text models, achieving **2x faster training** with **60% less memory** utilization. Vision support is expected to arrive soon, prompting users to **update Unsloth**.
   - *Please make sure to check for updates to enjoy the benefits of these enhancements.*
- **Explore New Notebooks for Llama 3.2**: New **Google Colab** notebooks for Llama 3.2 are available, including the **[Llama 3.2 (3B)](https://colab.research.google.com/drive/1T5-zKWM_5OD21QHwXHiV9ixTRR7k3iB9?usp=sharing)** and **[Kaggle notebook](https://www.kaggle.com/code/danielhanchen/kaggle-llama-3-2-1b-3b-conversational-unsloth/notebook)**. Check out the full list in the **[rest of the notebooks](https://docs.unsloth.ai/get-started/unsloth-notebooks)**.
   - *Utilize these resources to enhance your experimentation with Llama 3.2 and beyond.*
- **Llama 3.2 model uploads announced**: The **Llama 3.2 collection** features a variety of model versions, including **Base** and **Instruct** for 1B, 3B, 11B, and 90B parameters. Users can access all versions through the **[collection link](https://huggingface.co/collections/unsloth/llama-32-all-versions-66f46afde4ca573864321a22)**.
   - *This upload aims to streamline access to different configurations for various applications.*


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/1T5-zKWM_5OD21QHwXHiV9ixTRR7k3iB9?usp=sharing)">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1qN1CEalC70EO1wGKhNxs1go1W9So61R5?usp=sharing)">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1lN6hPQveB_mHSnTOYifygFcrO8C1bxq4?usp=sharing)">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1oCEHcED15DzL8xXGU1VTx5ZfOJM8WY01?usp=sharing)">Google Colab</a>: no description found</li><li><a href="https://docs.unsloth.ai/get-started/unsloth-notebooks)">Unsloth Documentation</a>: no description found</li><li><a href="https://docs.unsloth.ai/get-started/all-our-models)">Unsloth Documentation</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1288590341439754241)** (7 messages): 

> - `OpenAI's Corporate Shift`
> - `Investor Concerns at OpenAI`
> - `Llama 3.2 on GPU`
> - `Petscan API Usage`
> - `Management Vesting Lock-ins` 


- **OpenAI moves towards Corporate Mode**: Concerns have been raised that OpenAI is losing its appeal as an **exciting startup** and is shifting into **corporate mode**.
   - *One member lamented*, 'no longer an exciting startup,' pointing to a potential stifling of innovation.
- **Investors scrutinize OpenAI's growth**: Members speculate that **investors** are questioning where the funds have gone and why the company isn't achieving the expected **10x growth**.
   - One remarked that investors may be starting to point fingers at engineering if results don't improve.
- **Llama 3.2 runs on CuPy**: [GitHub](https://github.com/githubpradeep/llm_np_cp/blob/main/llama3.2_model.py) showcases Llama 3.2 operating fully with **CuPy** on GPU, marking a step forward for efficiency.
   - A member shared their code, touting the capabilities of combining **CuPy** and **NumPy** for model execution.
- **Questions arise about Petscan API usage**: A member inquired about the **Petscan API**, seeking insights from others who might have used it programmatically.
   - This indicates interest in expanding the API's practical applications within the community.
- **Vesting lock-ins for management**: Speculation arose regarding the potential end of **vesting lock-ins** for upper management, possibly signaling upcoming changes.
   - This could imply shifts in leadership dynamics or incentives as the company navigates its corporate evolution.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/miramurati/status/1839025700009030027">Tweet from Mira Murati (@miramurati)</a>: I shared the following note with the OpenAI team today.</li><li><a href="https://github.com/githubpradeep/llm_np_cp/blob/main/llama3.2_model.py">llm_np_cp/llama3.2_model.py at main · githubpradeep/llm_np_cp</a>: running llama gemma on cupy and numpy. Contribute to githubpradeep/llm_np_cp development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1288598201867304991)** (110 messages🔥🔥): 

> - `Fine-Tuning Llama 3.1 & 3.2`
> - `Unsloth Library Issues`
> - `Training Configuration`
> - `Qwen Model Errors`
> - `Alpaca Template for Fine-Tuning` 


- **Fine-Tuning Llama 3.1 & 3.2 Issues**: Users experienced challenges with fine-tuning Llama 3.1 and 3.2 models, including NaN errors during training and confusion over dataset configurations.
   - Many suggested using smaller batch sizes, reducing gradient accumulation, and even swapped datasets to address these issues.
- **Unsloth Library Compatibility**: The Unsloth library is being actively developed, including support for various model versions, but currently lacks compatibility with vision models and shows errors related to untrained tokens.
   - Users advised updating dependencies, particularly transformers, to mitigate some of these issues.
- **Training Configuration and Hyperparameters**: Several users shared their training parameter setups, discussing the impact of various settings, such as learning rates, gradient accumulation, and batch sizes on performance.
   - It was noted that adjusting these parameters can significantly affect memory usage and NaN occurrences during training.
- **Qwen Model Fine-Tuning Trials**: Users reported mixed results while attempting to fine-tune the Qwen model, with specific errors pointing to embedding and tokenization problems when using larger models.
   - A suggestion was made to incorporate embed_tokens and lm_head in target_modules during the training process to resolve the issues.
- **Usage of Alpaca Template for Fine-Tuning**: Questions arose about using the Alpaca instruction template in the specified format, affecting tokenizer configurations when fine-tuning models.
   - Users sought guidance on successfully integrating this template into their training process.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/17d3U-CAIwzmbDRqbZ9NnpHx">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/17d3U-CAIwzmbDRqbZ9NnpHxCkmXB6LZ0?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://docs.unsloth.ai/basics/reward-modelling-dpo-and-orpo">Reward Modelling - DPO &amp; ORPO | Unsloth Documentation</a>: To use DPO or ORPO with Unsloth, follow the steps below:</li><li><a href="https://www.youtube.com/watch?v=eIziN2QUt8U">Fine-tune Multi-modal LLaVA Vision and Language Models</a>: ➡️ ADVANCED Vision Fine-tuning Repo: https://trelis.com/advanced-vision/➡️ Trelis Newsletter: https://blog.Trelis.com➡️ Trelis Resources and Support: https:/...</li><li><a href="https://docs.unsloth.ai/b">Unsloth Documentation</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth.git">GitHub - unslothai/unsloth: Finetune Llama 3.1, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory</a>: Finetune Llama 3.1, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1288732722403217428)** (15 messages🔥): 

> - `Fine-tuning Unsloth with KTO`
> - `Flexora for Low-Rank Adaptation`
> - `INT-FlashAttention GitHub Resource`
> - `Data Packing in LLM Pre-training`
> - `Using DeepSpeed for Pre-training GPT-2` 


- **Using Unsloth for fine-tuning with KTO**: A user inquired about the feasibility of using **Unsloth** for fine-tuning with binary feedback data.
   - This raises discussions on methods for effectively utilizing KTO in such scenarios.
- **Introducing Flexora for Low-Rank Adaptation**: The paper linked in the message proposes **Flexora**, a method to improve Low-Rank Adaptation (LoRA) by selecting key layers for optimal fine-tuning.
   - It addresses the issue of overfitting in existing LoRA techniques when fine-tuning models on specific tasks.
- **Resource on INT-FlashAttention**: One member shared links to a GitHub resource on **INT-FlashAttention**, including Python code for **flash_atten_int8.py**.
   - This project appears to be part of broader development efforts in enhancing attention mechanisms.
- **Discussion on Data Packing for LLM Pre-training**: A user asked how LLMs like GPT or LLaMA handle varying context lengths during training, particularly around data packing.
   - Members discussed methods like training frameworks masking unrelated parts and the potential of uploading multiple examples simultaneously.
- **DeepSpeed for Pre-training GPT-2**: A suggestion was made to utilize **DeepSpeed** for pre-training a small **GPT-2** model, emphasizing its efficiency.
   - Further inquiries were made about whether DeepSpeed supports data packing for optimizing training.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2408.10774v1">Flexora: Flexible Low Rank Adaptation for Large Language Models</a>: Large Language Models (LLMs) are driving advancements in artificial intelligence by increasing the scale of model parameters, which has significantly enhanced generalization ability and unlocked new c...</li><li><a href="https://github.com/INT-FlashAttention2024/INT-FlashAttention/blob/main/flash_atten_int8.py">INT-FlashAttention/flash_atten_int8.py at main · INT-FlashAttention2024/INT-FlashAttention</a>: Contribute to INT-FlashAttention2024/INT-FlashAttention development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1288579271870386207)** (175 messages🔥🔥): 

> - `Llama 3.2 Performance`
> - `Llama 3.2 Vision Models`
> - `Model Compatibility and Usage`
> - `Embedding Issues in LM Studio`
> - `New Models and Updates` 


- **Llama 3.2 Performance Benchmarks**: Users discussed the performance of various Llama 3.2 models, noting benchmark scores such as **Llama 3.2 1B** at **49.3%** and **3B** at **63.4%**.
   - Performance comparisons included discussions about quantized models and how these impact token throughput, with some models achieving around **15-17 tokens/sec**.
- **Llama 3.2 Vision Models Support**: Vision models like **Llama 3.2 Vision Instruct** are currently unsupported in **llama.cpp**, and there is uncertainty about future compatibility.
   - Users have expressed interest in using these models, but challenges remain regarding quantization and integration in existing frameworks.
- **Embedding and API Issues**: An user reported issues with embeddings returning errors after upgrading to LM Studio **0.3.2**, specifically for larger inputs exceeding the context window.
   - The issue was resolved by chunking the embedding requests, highlighting the importance of managing context window limitations in API usage.
- **Installing and Running New Models**: Discussion surrounding the installation of updated models and the use of third-party models via the LM Studio API brought attention to the need for correct model versions.
   - Users were guided to download the latest LM Studio beta and specific models from Hugging Face to ensure compatibility and functionality.
- **Model Customization and Behavior**: Users expressed concerns about certain models, like **Minicpm 2.6**, displaying restrictive behavior and requiring prompt engineering to produce desirable outputs.
   - Suggestions included using different prompting strategies to bypass built-in censorship mechanisms, leading to discussions about uncensored model versions.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/Joseph717171/Llama-3.2-3B-Instruct-OQ8_0-F32.EF32.IQ4_K-Q8_0-GGUF/resolve/main/Llama-3.2-3B-Instruct-OF32.EF32.IQ8_0.gguf">no title found</a>: no description found</li><li><a href="https://imgur.com/a/suUJuyV">imgur.com</a>: Discover the magic of the internet at Imgur, a community powered entertainment destination. Lift your spirits with funny jokes, trending memes, entertaining gifs, inspiring stories, viral videos, and ...</li><li><a href="https://imgur.com/a/88T9yJI">imgur.com</a>: Discover the magic of the internet at Imgur, a community powered entertainment destination. Lift your spirits with funny jokes, trending memes, entertaining gifs, inspiring stories, viral videos, and ...</li><li><a href="https://medium.com/@mlabonne/uncensor-any-llm-with-abliteration-d30148b7d43e">Uncensor any LLM with abliteration</a>: Fine-tuning without retraining</li><li><a href="https://huggingface.co/collections/allenai/molmo-66f379e6fe3b8ef090a8ca19">Molmo - a allenai Collection</a>: no description found</li><li><a href="https://huggingface.co/collections/Joseph717171/gguf-llama-32-instruct-oq8-0-f32ef32iq4-k-q8-0-iquants-66f49cf761597948c6fa789d">GGUF Llama-3.2-Instruct-OQ8_0-F32.EF32.IQ4_K-Q8_0 IQuants - a Joseph717171 Collection</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1fptwvm/molmo_a_new_model_that_outperforms_llama_32/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/chigkim/Ollama-MMLU-Pro">GitHub - chigkim/Ollama-MMLU-Pro</a>: Contribute to chigkim/Ollama-MMLU-Pro development by creating an account on GitHub.</li><li><a href="https://old.reddit.com/r/LocalLLaMA/comments/1fpckrw/qwen25_selfreported_now_on_official_mmlupro/">Qwen2.5 (self-reported) now on official MMLU-Pro leaderboard, beats Gemini 1.5 Pro and Claude 3 Opus</a>: https://huggingface.co/spaces/TIGER-Lab/MMLU-Pro</li><li><a href="https://github.com/ollama/ollama/pull/6963">image processing for llama3.2 by pdevine · Pull Request #6963 · ollama/ollama</a>: Image processing routines for being able to run llama3.2. This will need to be refactored at some point to support other multimodal models as well.</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/8010#issuecomment-2376339571">server: Bring back multimodal support · Issue #8010 · ggerganov/llama.cpp</a>: Multimodal has been removed since #5882 Depends on the refactoring of llava, we will be able to bring back the support: #6027 This issue is created mostly for tracking purpose. If someone want to t...</li><li><a href="https://huggingface.co/collections/lmstudio-ai/vision-models-gguf-6577e1ce821f439498ced0c1">Vision Models (GGUF) - a lmstudio-ai Collection</a>: no description found
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1288781696388431883)** (165 messages🔥🔥): 

> - `LLM Recommendations for Hardware`
> - `Performance on Different GPUs`
> - `VRAM Limitations`
> - `Integration of Multiple GPUs`
> - `Upcoming GPU Releases` 


- **Discussion on Suitable LLMs for Hardware**: Members discussed recommendations for LLMs to run on systems with Intel i7-8750H and 32GB of RAM, suggesting to look for models like qwen 2.5.
   - Options for using integrated Intel GPUs were also raised, noting the limits in speed due to reliance on system RAM.
- **Performance Metrics on Various GPUs**: Performance benchmarks were shared, indicating around **35 tokens/sec** on an AMD Radeon RX 5700 XT and **40 tokens/sec** on a Surface Laptop Studio 2 with NVIDIA GeForce RTX 4060.
   - Other users noted a **61 tokens/sec** on an Apple M3 Max chip, highlighting significant variances in performance across different hardware.
- **VRAM Limitations for Model Deployment**: Chat participants agreed that VRAM is a critical limiting factor for large models, with **24GB** GPUs struggling to efficiently run **70B** models.
   - Discussions included thoughts on lower VRAM models and their performance trade-offs, with a stated preference for **15 tok/s** as usable speed.
- **Integration of NVidia and AMD GPUs**: Users discussed the challenges of utilizing both NVidia and AMD GPUs in a single setup and how performance may be limited due to this combination.
   - The consensus was that software compatibility could define which models benefit from multi-GPU setups.
- **Anticipation for Future GPU Releases**: Upcoming GPUs like the NVIDIA GeForce RTX **5090** were discussed, with rumors of **32 GB** VRAM and high performance specifications.
   - Concerns were raised about the affordability of future models in comparison to current generation options, while also discussing the viability of used GPUs.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.reddit.com/r/LocalLLaMA/comments/14gnkfw/think_twice_about_getting_the_rtx_4060_ti/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://wccftech.com/nvidia-geforce-rtx-5090-32-gb-rtx-5080-16-gb-specs-5090-20k-cores-600w-5080-10k-cores-400w/">NVIDIA GeForce RTX 5090 32 GB &amp; RTX 5080 16 GB Specs Uncovered: 5090 Over 20K Cores &amp; 600W, 5080 Over 10K Cores &amp; 400W</a>: NVIDIA&#039;s GeForce RTX 5090 &amp; RTX 5080, the next generation of GPUs from the green team for gamer, have their specs revealed by Kopite7kimi.</li><li><a href="https://wccftech.com/nvidia-geforce-rtx-5090-32-gb-rtx-5080-16-gb-specs-5090-20k-cores-600w-5080-10k">NVIDIA GeForce RTX 5090 32 GB &amp; RTX 5080 16 GB Specs Uncovered: 5090 Over 20K Cores &amp; 600W, 5080 Over 10K Cores &amp; 400W</a>: NVIDIA&#039;s GeForce RTX 5090 &amp; RTX 5080, the next generation of GPUs from the green team for gamer, have their specs revealed by Kopite7kimi.</li><li><a href="https://videocardz.com/newz/nvidia-geforce-rtx-5090-and-rtx-5080-specs-leaked">NVIDIA GeForce RTX 5090 and RTX 5080 specs leaked - VideoCardz.com</a>: GeForce RTX 5090 to feature 21760 CUDA cores, 32GB GDDR7 memory and 600W, RTX 5080 gets 16GB VRAM Coming from Kopite7kimi himself.  One of the most reliable NVIDIA leakers has now confirmed the specs ...</li><li><a href="https://www.canadacomputers.com/index.php?cPath=43_557_559&sf=:3_22&co=&mfr=&pr=">Shop for Powered By Nvidia &amp; more - Canada Computers</a>: no description found
</li>
</ul>

</div>
  

---



### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1288580759694868500)** (230 messages🔥🔥): 

> - `Aider Updates`
> - `Model Roles in Aider`
> - `User Experience Improvements`
> - `Aider Modes`
> - `Configuration Options` 


- **Introduction of Senior and Junior Modes in Aider**: The recent updates to Aider introduce 'Senior' and 'Junior' roles for models, enhancing the coding process by dividing responsibilities between planning and execution.
   - Users expressed the need for clarity on how these roles operate, with some suggesting alternative terminologies like 'Planner' and 'Executor' to reduce confusion.
- **Optimizing User Experience with Aider**: There are discussions on making the two-step process optional in Aider, allowing a faster mode for quick edits while retaining the ability to plan through the senior/junior configuration.
   - Proposals include commands like `/fast` to switch modes efficiently and improving the model selection process based on the user's needs.
- **Best Model Combinations for Aider**: Users debated the best combinations for the Senior and Junior roles, exploring various models including OpenAI's o1-preview, Claude 3.5 Sonnet, and Deepseek.
   - Recommendations for default settings suggest using o1-preview for the Senior position with Sonnet as the Junior, while also considering the rapid Deepseek model for implementation tasks.
- **User Preferences for Aider Functionality**: There is a preference among users for the default functionality of Aider to remain fast and simple, with advanced features available on demand for troubleshooting.
   - The idea is to maintain the latest updates while ensuring existing users still have access to the original Aider experience for straightforward coding needs.
- **Feedback and Continuous Improvement**: The community emphasized the importance of user feedback on the new features, acknowledging the need to streamline the UX and model interactions.
   - Suggestions included allowing Aider to intelligently detect repetitious prompts and recommending appropriate mode transitions to enhance user efficiency.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.napkins.dev/">Napkins.dev – Screenshot to code</a>: Generate your next app with a screenshot</li><li><a href="https://github.com/paul-gauthier/aider/blob/main/aider/website/_posts/2024-09-26-senior-junior.md">aider/aider/website/_posts/2024-09-26-senior-junior.md at main · paul-gauthier/aider</a>: aider is AI pair programming in your terminal. Contribute to paul-gauthier/aider development by creating an account on GitHub.</li><li><a href="https://github.com/tldraw/make-real">GitHub - tldraw/make-real: Draw a ui and make it real</a>: Draw a ui and make it real. Contribute to tldraw/make-real development by creating an account on GitHub.</li><li><a href="https://github.com/paul-gauthier/aider.git">GitHub - paul-gauthier/aider: aider is AI pair programming in your terminal</a>: aider is AI pair programming in your terminal. Contribute to paul-gauthier/aider development by creating an account on GitHub.</li><li><a href="https://fireworks.ai/blog/cursor">How Cursor built Fast Apply using the Speculative Decoding API </a>: Cursor, an AI-native IDE, leveraged Fireworks inference stack to enhance its features like Instant Apply, Smart Rewrites, and Cursor Prediction. The blog post introduces the Speculative Decoding API, ...
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1288617146133385238)** (21 messages🔥): 

> - `Augmenting model inputs`
> - `Issues with Sonnet performance`
> - `Using Aider with GUI frameworks`
> - `Dependency update tools`
> - `File creation issues in Aider` 


- **Enhancing Model Inputs for Code Generation**: Users expressed challenges with LLMs generating correct code, specifically needing hints from external libraries and tools such as analyzers for languages like **Rust**.
   - One user remarked on the importance of accurately reflecting **typing** and **lifetime** rules during code generation.
- **Concerns about Sonnet's Reliability**: A member raised concerns about the **unreliability** of Sonnet and observed it performing worse than usual despite no apparent changes.
   - The community discussed ongoing issues, with one attributing it to bugs being fixed in another system affecting Sonnet's performance.
- **Using Aider Effectively with GUIs**: A user inquired about improving feedback loops while working with **Streamlit**, aiming for more interaction beyond basic commands and error reports.
   - Another responded that Streamlit's limitations would require a redesign of Aider for interactive front-end capabilities.
- **Automating Dependency Updates with Mend**: Discussion arose about **Mend Renovate**, which automates dependency updates by detecting newer package versions and integrating updates into code.
   - While one user was aware of this tool, there was an expressed desire for LLMs to better handle package version management independently.
- **File Creation Issues with Aider**: A user reported issues with Aider not consistently creating or editing files, although it indicated that it was performing these actions.
   - This raises questions on ensuring Aider defaults to reliably creating and modifying files in the specified directory.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://aider.chat/docs/faq.html#how-do-i-turn-on-the-repository-map">FAQ</a>: Frequently asked questions about aider.</li><li><a href="https://www.mend.io/renovate/">Mend Renovate Products</a>: Automate your dependency updates with Mend Renovate Products. Deliver pull requests to the repo and merge updates with confidence.</li><li><a href="https://github.com/zed-industries/zed/pull/18363">Fix sending alt-enter in terminal by notpeter · Pull Request #18363 · zed-industries/zed</a>: Reported here: How to forward option-enter (alt-enter) to terminal? #18149  Before peter@scruffy zed % fish_key_reader  Press a key: bind \eenter &amp;#39;do something&amp;#39;  After: peter@scruffy z...
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/)** (1 messages): 

fry69_61685: https://simonwillison.net/2024/Sep/25/o1-preview-llm/
  

---



### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1288615728127148032)** (1 messages): 

> - `Hermes 3`
> - `Llama-3.1 8B` 


- **Hermes 3 Launches on HuggingChat**: Nous Research has released the **Hermes 3** model sized at **8B** on [HuggingChat](https://huggingface.co/chat/settings/NousResearch/Hermes-3-Llama-3.1-8B), which follows instructions closely.
   - This release showcases a significant advancement in the **Hermes** series, making the model readily available for the community.
- **Key Features of Hermes 3**: The **Hermes 3** model in **8B** size is designed to closely follow user instructions and enhance interactive capabilities.
   - This aligns with the ongoing improvement efforts in the **Nous Research** projects to deliver more responsive AI solutions.



**Link mentioned**: <a href="https://huggingface.co/chat/settings/NousResearch/Hermes-3-Llama-3.1-8B">HuggingChat</a>: Making the community's best AI chat models available to everyone.

  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1288576185357570131)** (224 messages🔥🔥): 

> - `Nous Hermes 3`
> - `Llama 3.2 models`
> - `GPU advancements`
> - `Generative AI concerns`
> - `AI event in Singapore` 


- **Concerns on Nous Hermes 3 and Steerability**: Users expressed confusion about the steerability and censorship of Nous Hermes 3, contrasting expectations from its design with actual deployment limitations.
   - Discussion included the impact of system prompts and the rigidity of the model's responses based on its pre-set configurations.
- **Availability of Llama 3.2 3B Base Model**: There were inquiries about the availability of the Llama 3.2 3B base model in GGUF format, with users questioning whether it has been released yet.
   - Several users noted their interest in finetuning the model for better performance.
- **RTX 5090 Specs and Expectations**: The RTX 5090 was discussed, highlighting its expected 32GB memory and potential for future upgrades to 48GB, ensuring it would be an attractive buy.
   - Users speculated on the memory configurations of upcoming RTX series models and their implications for performance.
- **Market for Used GPUs**: A user sought to acquire used GPUs, particularly Ada-series devices like the A6000, and others offered lower-tier options or shared personal upgrades.
   - Discussion highlighted the scarcity of high-performance GPUs in the secondary market amid current technological advancements.
- **AI Event in Singapore**: An upcoming AI event in Singapore was revealed, featuring lightning talks from emerging companies in the AI sector, along with opportunities to network.
   - The event aims to showcase innovative AI tools and insights into future technology developments.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/voooooogel/status/1839144530995450346">Tweet from thebes (@voooooogel)</a>: got advanced voice mode so drunk on saying the word wheedle he started talking to himself and then got cut off by the guidelines</li><li><a href="https://huggingface.co/Joseph717171/Llama-3.2-1B-Instruct-OQ8_0.EF32.IQ4_K-Q8_0-GGUF/resolve/main/Llama-3.2-1B-Instruct-OQ8_0.EF32.IQ8_0.gguf">no title found</a>: no description found</li><li><a href="https://huggingface.co/chat/settings/NousResearch/Hermes-3-Llama-3.1-8B">HuggingChat</a>: Making the community's best AI chat models available to everyone.</li><li><a href="https://huggingface.co/spaces/huggingface-projects/llama-3.2-3B-Instruct">Llama 3.2 3B Instruct - a Hugging Face Space by huggingface-projects</a>: no description found</li><li><a href="https://x.com/teknium1/status/1839040366512844917?s=46">Tweet from Teknium (e/λ) (@Teknium1)</a>: So you didn&#39;t get HER with OpenAI&#39;s advanced voice mode - well - my friends at http://Play.AI have the real deal, HERmes.  Try it here: https://play.ai/agent/HERMES-m3i3jU81_52ruL6_0tw2R</li><li><a href="https://x.com/teknium1/status/1839040366512844917?s=4">Tweet from Teknium (e/λ) (@Teknium1)</a>: So you didn&#39;t get HER with OpenAI&#39;s advanced voice mode - well - my friends at http://Play.AI have the real deal, HERmes.  Try it here: https://play.ai/agent/HERMES-m3i3jU81_52ruL6_0tw2R</li><li><a href="https://huggingface.co/collections/alpindale/llama-32-re-upload-66f463d7940e8a6c7f5b7bbc">Llama 3.2 Re-upload - a alpindale Collection</a>: no description found</li><li><a href="https://x.com/kopite7kimi/status/1839343725727941060?s=19">Tweet from kopite7kimi (@kopite7kimi)</a>: GeForce RTX 5090 PG144/145-SKU30 GB202-300-A1 21760FP32 512-bit GDDR7 32G 600W</li><li><a href="https://console.groq.com/docs/models">GroqCloud</a>: Experience the fastest inference in the world</li><li><a href="https://huggingface.co/collections/Joseph717171/gguf-llama-32-instruct-oq8-0-f32ef32iq4-k-q8-0-iquants-66f49cf761597948c6fa789d">GGUF Llama-3.2-Instruct-OQ8_0-F32.EF32.IQ4_K-Q8_0 IQuants - a Joseph717171 Collection</a>: no description found</li><li><a href="https://x.com/OfficialLoganK/status/1839310682530959367?t=jJkHuEtuJlc956Q58ZdNjw&s=19">Tweet from Logan Kilpatrick (@OfficialLoganK)</a>: Backup intelligence for the bunker</li><li><a href="https://lu.ma/mlqwqi6x">Singapore AI Showcase · Luma</a>: Join us for an exclusive AI Showcase on 8 October, to kick off TechWeek Singapore. This event will feature innovative AI teams, each presenting lightning talks…</li><li><a href="https://x.com/JStaatsCPA/status/1838984688917954621?t=SRmzuonj2li2suACqaEWUA&s=19">Tweet from Jason Staats⚡ (@JStaatsCPA)</a>: This new ChatGPT Advanced Voice mode 🤯🤯</li><li><a href="https://old.reddit.com/r/LocalLLaMA/comments/1fp9wem/llama_32_1b_3b_benchmarks/)">Llama 3.2 1B &amp; 3B Benchmarks</a>: Posted in r/LocalLLaMA by u/TKGaming_11 • 122 points and 20 comments</li><li><a href="https://old.reddit.com/r/LocalLLaMA/comments/1fpb4m3/molmo_models_outperform_llama_32_in_most_vision/">Molmo Models Outperform Llama 3.2 in Most Vision Benchmarks 🌟</a>: Posted in r/LocalLLaMA by u/shrewdeenger • 204 points and 27 comments</li><li><a href="https://qwenlm.github.io/blog/qwen2.5-llm/">Qwen2.5-LLM: Extending the boundary of LLMs</a>: GITHUB HUGGING FACE MODELSCOPE DEMO DISCORD Introduction In this blog, we delve into the details of our latest Qwen2.5 series language models. We have developed a range of decoder-only dense models, w...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1288612439390294086)** (17 messages🔥): 

> - `Llama 3.2 Vision Encoder`
> - `Inference Requirements for Llama 3.2`
> - `Llama 3.2 Model Availability`
> - `DisTRo Optimizer`
> - `Judgement and Reward Modelling in Hermes 3` 


- **Llama 3.2 Vision Encoder is Gigantic**: The **Llama 3.2 Vision Encoder** boasts significant sizes, with the **11B model** reaching almost **3B** parameters and the **90B model** hitting **18B**.
   - *Gigantic* is the description members used to emphasize the size of this encoder.
- **Inference GPU Requirements for 90B Llama 3.2**: A member suggested that **3x H100 GPUs** should suffice for inferring the **90B Llama 3.2**, though **4x might be needed** for tensor parallelism or larger batch sizes.
   - The inference question pointed towards practical GPU setup plans on **Runpod**.
- **Accessing New Llama 3.2 Models in the EU**: New **Llama 3.2 models** can now be downloaded in the EU through fine-tunes and quantized copies found on **Hugging Face**, such as **Unsloth’s collection**.
   - Additionally, **Alpindale has re-uploaded the weights unmodified** for easier access.
- **Questions on DisTRo Optimizer Implementation**: A member expressed interest in the **DisTRo optimizer** and inquired about building reliable infrastructure using **Elixir** until it's available.
   - They considered using a standard optimizer initially, like **SGD or ADAM**, while preparing for better integration later.
- **Hermes 3's Judgement and Reward Modelling**: Inquiries were made about the improvements in **judgement and reward modelling** for **Hermes 3** and whether synthetic data was used in the SFT dataset.
   - The confirmation was given that the improvements are supported by **synthetic data**, rather than solely relying on public datasets.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 messages): 

trre: https://arxiv.org/abs/2409.16897
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1288590690539929713)** (2 messages): 

> - `Wordware App Updates`
> - `Opus Insight Enhancements`
> - `ThinkLab Features`
> - `O1-Preview Limitations` 


- **Wordware App Updates with O1Mini**: Updated versions of main Wordware apps now include **O1Mini**, enhancing functionality and reviews.
   - The **Opus Insight** template utilizes **Sonnet 3.5** for initial reviews, followed by O1Mini for comprehensive model ranking.
- **ThinkLab for Expanded Searches**: **ThinkLab** is powered by **Sonar Huge 405b**, focusing on scratchpad usage and subsequent searches for broader exploration.
   - This tool aims to streamline the user's exploration process through efficient search capabilities.
- **O1-Preview Rate Limit Challenges**: **O1-Preview** is an option in the Wordware app but it's **rate limited**, which can stall application performance if it doesn't return data.
   - Although it's added to the Wordware app, it remains **disabled** to avoid stalling, but users can create their own version to experiment with the dedicated O1-Preview flow.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://app.wordware.ai/explore/apps/aa2996a0-93c9-4c19-ade2-1796c5c8a409">OPUS Insight : Latest Model Ranking - o1mini</a>: This prompt processes a question using the latest models,and provides a comprehensive review and ranking.    Update: 9/25/2024 - added: o1mini, Gemini 1.5 Flash, Command R+  . Note: o1-preview is part...</li><li><a href="https://app.wordware.ai/explore/apps/999cc252-5181-42b9-a6d3-060b4e9f858d">_Think-Lab Revised - o1mini</a>: (version 1.10) Use the power of ScratchPad-Think for every day web searches. Export refined search queries in JSON format. The scratchpad is a powerful tool that helps you maintain coherence and accur...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 messages): 

trre: https://arxiv.org/abs/2409.16897
  

---



### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1288753349172133888)** (5 messages): 

> - `Scam Link Identification`
> - `Textbook Discussion` 


- **Users identify scam link**: Members expressed concern about a potentially fraudulent link, emphasizing that it is **definitely a scam**.
   - A user tagged the relevant role to alert others, resulting in action taken against the user who posted the link.
- **PMPP as Class Textbook**: A member mentioned that **PMPP** is their class textbook, showing enthusiasm by using an emoji reaction.
   - This comment highlights the relevance of **PMPP** in ongoing educational discussions within the group.


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1288699558334959646)** (6 messages): 

> - `Triton Conference 2024`
> - `Proton kernel optimization`
> - `Triton compiled wheel for Windows` 


- **Triton Conference 2024 Video Releases**: The recordings of the [Triton Conference 2024](https://www.youtube.com/watch?v=NZz5sczZ_30) are now available, featuring various sessions like the morning session by Keren Zhou from OpenAI.
   - The afternoon session included a keynote on Triton Strategy at Meta by Aparna Ramani, accessible via [this link](https://www.youtube.com/watch?v=ONrKkI7KhU4).
- **Proton and Kernel Call Optimization Discussion**: A user noticed fewer Triton kernels displayed in proton-viewer compared to their output_code.py, questioning if optimizations occurred at the driver level.
   - Another user clarified that every `kernel[grid](args...)` is guaranteed to be a kernel call and that **proton** groups these calls together.
- **Seeking Triton Compiled Wheel for Windows**: A member is looking for the latest Triton compiled wheel compatible with **Windows** and **Python 3.10**.
   - This request highlights the ongoing need for updated installation packages in the Triton community.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=NZz5sczZ_30">Triton Conference 2024: Morning Session</a>: Tutorials 2:30 Dev Tools: Proton/Interpreter, Keren Zhou (George Mason University &amp; OpenAI) Slides: https://bit.ly/proton-interpreter-tutorial1:04:24 Compile...</li><li><a href="https://www.youtube.com/watch?v=ONrKkI7KhU4">Triton Conference 2024: Afternoon Session</a>: 1:25 Welcome: Ajit Mathews, META6:05 Keynote: Triton Strategy at Meta, Aparna Ramani, META https://bit.ly/aparna-keynote17:05 Keynote: Triton State of the Un...
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1288631725509050452)** (9 messages🔥): 

> - `PyTorch Memory Profiling`
> - `TorchDispatchMode`
> - `Autograd Optimization`
> - `PyTorch Profiler`
> - `Memory Visualization` 


- **Investigating Memory Allocation in PyTorch Layers**: Members discussed ways to check memory allocation for each layer in **PyTorch** while inquiring about **weights**, **grads**, and **optimizer states** memory data.
   - *Memorypaladin* referenced a slide implying a positive answer to this question from a past conference.
- **Manual and Automatic Memory Profiling Techniques**: A member mentioned that you can manually obtain memory allocation info through visualizations in PyTorch by clicking on specific layers.
   - *p0.tato* highlighted that there's an automated method using **torchdispatchmode** for parsing memory data.
- **Sharing Interesting GitHub Issue on Autograd Optimization**: A member shared a [GitHub issue](https://github.com/pytorch/pytorch/issues/136733) discussing an optimization related to autograd formulas, noting that it could depend on either input or output.
   - They found this optimization to be quite fun and worth exploring.
- **Using PyTorch Profiler for Memory Data**: A member recommended using the **PyTorch Profiler** API for obtaining broad overviews of memory usage, citing its colorful visualizations.
   - However, they noted that while useful for understanding, it lacks interactivity compared to the memory snapshot approach used for debugging.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://pytorch.org/docs/stable/profiler.html#torch.profiler._KinetoProfile.export_memory_timeline,">torch.profiler &mdash; PyTorch 2.4 documentation</a>: no description found</li><li><a href="https://github.com/pytorch/pytorch/issues/136733">Be smart about autograd formulas saving either the input or output, depending on context · Issue #136733 · pytorch/pytorch</a>: 🚀 The feature, motivation and pitch See NVIDIA/apex#1715 The general idea is that for some operators, in principle the autograd formula can be written depending on either the input or the output. F.....
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1288584505241899079)** (2 messages): 

> - `Llama 3.2 Models`
> - `Llama 3.1 Impact`
> - `Edge and Mobile AI`
> - `Availability Issues` 


- **Llama 3.2 Brings New Models**: Meta announced the release of **Llama 3.2**, featuring new small and medium-sized vision LLMs (11B and 90B) and lightweight text models (1B and 3B) optimized for **edge and mobile devices**.
   - This expansion aims to offer developers more accessible resources sans the need for extensive compute capabilities.
- **Llama 3.1 Models Double Usage**: Since the launch of **Llama 3.1**, its impact has seen usage **double** in two months, showcasing the model's capabilities as a leading open frontier-level AI with **405B parameters**.
   - Meta's **Zuckerberg** emphasized the excitement around the models' performance at the Connect event.
- **New Models Pruned for Efficiency**: The **1B and 3B text models** of Llama 3.2 are pruned and distilled versions of larger models from **8B and 70B**, catering to developers with limited resources.
   - This strategic pruning allows applications on devices with constrained compute capabilities.
- **EU Availability Limitations**: **Llama 3.2's** availability for the 11B and 90B vision models does not extend to **EU countries**, limiting access for developers in that region.
   - This restriction has raised concerns among EU-based developers eager to leverage these powerful tools.



**Link mentioned**: <a href="https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/">no title found</a>: no description found

  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1288617009235361875)** (8 messages🔥): 

> - `CUDA project setup`
> - `clangd configuration issues`
> - `RTX 3070 compute capabilities`
> - `Neovim vs. other IDEs`
> - `Channel organization` 


- **Channel created for focused discussions**: A new channel was created to ensure that discussions are kept in the correct place, allowing members to focus their conversations.
   - *Thanks will repost it there* reassures that the article will be shared in the specified channel for better context.
- **Trouble with clangd in CUDA project**: A new user is experiencing issues setting up their CUDA project with clangd in Neovim, showing various unsupported options in diagnostics output.
   - Members are offering insights into how clangd doesn't accept the same options as nvcc, highlighting a potential source of confusion.
- **RTX 3070 compute capability clarification**: Discussion arose around the user's RTX 3070 being set to the incorrect compute capability (`sm_20`), while it should be `sm_86`.
   - Clarifying the compute capability should resolve some compilation issues mentioned earlier, including version conflicts.
- **Exploring alternatives to Neovim**: One member is considering using Visual Studio Code to take advantage of CUDA tooling interfaces, opting for enhanced usability.
   - The discussion suggests a preference for more effective tools like GDB and other CUDA profilers over using Neovim for larger projects.


  

---


### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1288975422062723092)** (1 messages): 

> - `Reading Images in C`
> - `Chapter 3 Overview` 


- **Tips for Reading Images in C**: A member inquired about recommendations for reading images using **C** in chapter 3, which includes images as part of its content.
   - No specific suggestions were provided in the messages, but the question indicates a focus on practical applications within the chapter.
- **Chapter 3 Incorporates Visuals**: The discussion notes that chapter 3 utilizes **images** to enhance understanding of the material.
   - This approach suggests a hands-on aspect where visual elements play a crucial role in the learning experience.


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1288648844287934516)** (49 messages🔥): 

> - `torchao compatibility`
> - `FP8 on CogVideoX`
> - `Windows support for ML`
> - `Training issues with NVIDIA`
> - `Community discussions on development` 


- **torchao struggles with NVIDIA**: A user reported that `torchao` does not function properly on NVIDIA hardware, experiencing dtype casting errors during training.
   - Despite being able to perform inference using FP8 on an RTX 3060, issues arise when transitioning to training scenarios.
- **Debate on Windows support importance**: Several users discussed the relevance of Windows support for ML frameworks, with some asserting an increasing interest from users not utilizing Linux.
   - While one argued that the emphasis on Linux is off base, another pointed out that current implementations primarily rely on Triton, which is Linux-centric.
- **Challenges with FP8 in CogVideoX**: A user noted that CogVideoX requires H100 hardware for FP8 functionality, highlighting limitations in compatibility for certain models.
   - There is frustration among users regarding the barriers to utilizing FP8 with specific setups, especially when utilizing features like NVIDIA platforms.
- **Community tensions over developer accountability**: Discussions turned heated as one user criticized developers for not providing Windows support, prompting other community members to defend the developers' choices and approaches.
   - The conversation underscored frustrations about the expectations placed on developers, especially regarding support for diverse operating systems.
- **Clarifications on training methods**: A user shared their approach for initializing training without autocast, utilizing bf16 weights, but faced challenges with low-precision optimizers needing gradient upcasting.
   - This sparked insights into the technical hurdles many face when trying to optimize performance in ML training contexts.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://scholar.google.com/citations?user=_2_KAUsAAAAJ">Furkan G�z�kara</a>: Assistant Professor Computer Engineer, Toros University - Cited by 20 - Data Mining - Sentiment Analysis - Text Classification - Product Clustering - Clustering</li><li><a href="https://github.com/pytorch/ao/issues/957">is this only for linux? · Issue #957 · pytorch/ao</a>: I installed on windows and failing from torchao.quantization import quantize_ pip freeze Microsoft Windows [Version 10.0.19045.4894] (c) Microsoft Corporation. All rights reserved. R:\CogVideoX_v1\...</li><li><a href="https://huggingface.co/THUDM/CogVideoX-5b">THUDM/CogVideoX-5b · Hugging Face</a>: no description found</li><li><a href="https://github.com/bghira/SimpleTuner/pull/986/files#diff-327015d4d445c4efaaa945a93701df4c68e3bc401dc4ddb7e55f2b5dc7854d6fR103-R116>">(wip, does not work) torchao: fp8/int8 by bghira · Pull Request #986 · bghira/SimpleTuner</a>: no description found</li><li><a href="https://github.com/pytorch/pytorch/blob/b408591b5380d6b856e0a4ce32cf386c56660c54/torch/_inductor/kernel/mm_scaled.py#L110-L190">pytorch/torch/_inductor/kernel/mm_scaled.py at b408591b5380d6b856e0a4ce32cf386c56660c54 · pytorch/pytorch</a>: Tensors and Dynamic neural networks in Python with strong GPU acceleration - pytorch/pytorch</li><li><a href="https://github.com/pytorch/ao/pull/276/files#diff-9f968cc00e2ee60006f8747d55f99cb54f367ca98f8d360731d306ab1d5db2b4L239)">Adding Llama to TorchAO by HDCharles · Pull Request #276 · pytorch/ao</a>: Summary: This PR adds funcitonality for stable eval/benchmarking of llama models within the torchao codebase. the model stuff is in torchao/_models/llama with eval being moved to _models/_eval.py m...</li><li><a href="https://github.com/p">p - Overview</a>: p has 153 repositories available. Follow their code on GitHub.</li><li><a href="https://github.com/meta-llama/llama-models/blob/main/models/llama3/reference_impl/model.py">llama-models/models/llama3/reference_impl/model.py at main · meta-llama/llama-models</a>: Utilities intended for use with Llama models. Contribute to meta-llama/llama-models development by creating an account on GitHub.</li><li><a href="https://github.com/pytorch-labs/gpt-fast/blob/main/model.py#L266">gpt-fast/model.py at main · pytorch-labs/gpt-fast</a>: Simple and efficient pytorch-native transformer text generation in &lt;1000 LOC of python. - pytorch-labs/gpt-fast
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1288576049625567384)** (10 messages🔥): 

> - `Easy to Use GPU Solutions`
> - `Prime Intellect on Lambda`
> - `IRL Hackathon Planning`
> - `Edge LLM Challenge`
> - `Team Collaboration for Challenges` 


- **GPU Solutions Prove User-Friendly**: A user mentioned that in their experience, a particular GPU was the **easiest to use**, suggesting that they might consider a **successful conversion** when looking for upgrades from their **3070**.
   - This indicates a possibly growing trend for users seeking more accessible GPU options.
- **Prime Intellect Leverages Lambda**: It was shared that **Prime runs on top of lambda** and can be accessed through **prime-intellect**.
   - This highlights the integration of cloud solutions for GPU processing in user workflows.
- **Community Organizes First IRL Hackathon**: A conversation started about organizing a **first-of-its-kind hackathon** with members from the **GPU Mode Discord server**.
   - This event aims to bring members together IRL, fostering community engagement around GPU advancements.
- **Edge LLM Challenge Requires Innovation**: The **Edge LLM Challenge** includes tasks for teams to develop compression methods for models like **Phi-2** and **Llama-3-8B**, with a focus on making them operable on **smartphones**.
   - Participants are also encouraged to train language models from scratch, fostering creativity and innovation in AI model training.
- **Seeking Teams for Edge Challenges**: A user expressed a desire to **team up** for the Edge LLM Challenge, fostering collaboration among participants.
   - Another user confirmed that **helpful info** was shared for the challenges, encouraging teamwork and knowledge sharing.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://edge-llms-challenge.github.io/edge-llm-challenge.github.io/challenge">no title found</a>: no description found</li><li><a href="https://x.com/caseyaylward/status/1839358642241536262">Tweet from Casey Aylward (@caseyaylward)</a>: 1/ A few months back, some friends from the GPU Mode (fka CUDA Mode) Discord server and I started talking about bringing the community IRL to plan a first-of-its-kind hackathon 🧵
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1288969786352472065)** (1 messages): 

> - `Guatemala Meetups`
> - `Regional Connections` 


- **Open Call for Meetup in Guatemala**: A member expressed interest in organizing meetups in **Guatemala**, encouraging others to connect if they are nearby.
   - They also mentioned a willingness to meet with those from **Belize** or **Mexico**.
- **Seeking Connections Across Borders**: The same member is reaching out to potential meetups, highlighting the **importance of community** in the region.
   - This initiative aims to strengthen ties among local enthusiasts and professionals.


  

---


### **GPU MODE ▷ #[triton-puzzles](https://discord.com/channels/1189498204333543425/1219683012707487794/1288866450110550210)** (2 messages): 

> - `Numba`
> - `Convolution and Deep Learning`
> - `Attention and Transformers` 


- **Numba's Deep Learning Chapter**: *mr.osophy* noted that the discussion is actually regarding **Numba**, pointing out that it includes a chapter dedicated to **convolution and deep learning**.
   - They questioned the absence of any mention of **Attention** or **Transformers** in this context.
- **Absence of Attention in Numba Discussions**: The discussion highlights a potential oversight in Numba's focus by not mentioning **Attention mechanisms** that are critical in modern deep learning.
   - *mr.osophy* expressed surprise at the lack of coverage for **Transformers**, a foundational topic in the field.


  

---


### **GPU MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1288576075860938793)** (25 messages🔥): 

> - `RoPE forward integration`
> - `Fused RMSNorm implementation`
> - `SwiGLU performance`
> - `Forward pass matching`
> - `REPKV backward kernel review` 


- **RoPE forward integration successful**: RoPE forward integration has been added and confirmed to be working smoothly.
   - The integration has been pushed to the branch and is ready for further testing.
- **Fused RMSNorm implementation completed**: The fused **RMSNorm** has been successfully integrated and is functioning correctly.
   - Next up is the implementation of the **SwiGLU** layer.
- **SwiGLU layer confirmed working**: **SwiGLU** is now operational, marking another step forward in the development process.
   - Developers are on the verge of achieving a full forward pass with the models.
- **Forward pass matches PyTorch**: The forward pass now matches the results from **PyTorch** for **Llama 3.1**, signifying a major milestone.
   - The observed loss is noted to be around **0.021**, which is acceptable for the 8B model.
- **REPKV backward kernel under review**: A new PR for the `repkv_backward` kernel is ready for review, aimed at the **llama3** branch.
   - The kernel has been updated and tested, and further assistance is sought for the ongoing **llama3** efforts.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/karpathy/llm.c/pull/764">Adding backward kernel for repkv on `llama3` branch (cudamode-irl) by insop · Pull Request #764 · karpathy/llm.c</a>: PTAL, repkv_backward is updated and tested. I will update repkv.cuh once this PR is merged. CC: @karpathy This is an WIP repkv backward kernel, started as a cudamode-irl project. Once the following...</li><li><a href="https://github.com/karpathy/llm.c/pull/763">rmsnorm backward simple baseline kernel by ngc92 · Pull Request #763 · karpathy/llm.c</a>: baseline kernel and cpu versions
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1288594256478601327)** (3 messages): 

> - `GroupedGemm examples`
> - `Architecture development` 


- **Clarifying GroupedGemm Example Selection**: A member noted uncertainty in choosing from multiple **GroupedGemm examples**, implying confusion over which one to select for implementation.
   - This highlights a need for clearer documentation or guidelines in selecting appropriate examples.
- **Building Initial Architecture**: Discussion centered around the idea of **building a small architecture** as a foundational step in development.
   - *Starting small* may help in streamlining the process and identifying potential issues early on.


  

---


### **GPU MODE ▷ #[arm](https://discord.com/channels/1189498204333543425/1247232251125567609/1288607273018589305)** (5 messages): 

> - `iPhone 16 Pro`
> - `bfloat16 enablement`
> - `M4 SME exploration`
> - `Apple GPU microarchitecture` 


- **Testing iPhone 16 Pro for Metal Benchmarks**: A user expressed interest in trying out the benchmarks on their new **iPhone 16 Pro** soon, citing a relevant repository on GitHub: [metal-benchmarks](https://github.com/philipturner/metal-benchmarks).
   - This repository explores the **Apple GPU microarchitecture**, making it a great resource for performance evaluation.
- **bfloat16 Enablement in ExecutorCH**: A user shared their recent work on **bfloat16 enablement** in the ExecutorCH framework, indicating progress in performance enhancements.
   - This development could play a pivotal role in optimizing data processing on supported architectures.
- **Scalable Matrix Extension for Apple M4 Processor**: Discussion highlighted the project [m4-sme-exploration](https://github.com/tzakharko/m4-sme-exploration), which examines the **scalable matrix extension** for the Apple M4 processor.
   - This exploration aims to expand optimization capabilities beyond Apple's ecosystem, with potential availability on other ARM chips.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/tzakharko/m4-sme-exploration">GitHub - tzakharko/m4-sme-exploration: Exploring the scalable matrix extension of the Apple M4 processor</a>: Exploring the scalable matrix extension of the Apple M4 processor - tzakharko/m4-sme-exploration</li><li><a href="https://github.com/philipturner/metal-benchmarks">GitHub - philipturner/metal-benchmarks: Apple GPU microarchitecture</a>: Apple GPU microarchitecture. Contribute to philipturner/metal-benchmarks development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/1288596491774066688)** (1 messages): 

> - `Block Scheduling`
> - `Randomization in Scheduling` 


- **Block Scheduling appears random**: A member noted that in practice, **blocks** are scheduled essentially at random, indicating a lack of predictability.
   - This randomness may impact any potential improvements to the scheduling process in the near future.
- **Doubts about scheduling improvements**: Another member expressed skepticism about the potential for any enhancements to be introduced to the scheduling process soon.
   - Given the current random nature of scheduling, immediate changes seem unlikely.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1288619866445582366)** (3 messages): 

> - `GPU Performance Optimization`
> - `CUDA Virtual Connect`
> - `Llama 3.2 with CuPy` 


- **Improve Your GPU Performance Insights**: A blog post titled [An Introduction to GPU Performance Optimization for Deep Learning](https://www.digitalocean.com/community/tutorials/an-introduction-to-gpu-optimization) emphasizes learning the features of latest GPU architectures and performance monitoring tools for optimal use.
   - It highlights the importance of experimenting and benchmarking to achieve better hardware utilization, especially in deep learning contexts.
- **Join CUDA Virtual Connect with Experts**: An upcoming [CUDA Virtual Connect](https://github.com/NVIDIA/accelerated-computing-hub/tree/main/connect-with-experts) event on September 27, 2024, offers a chance to engage with CUDA library developers at NVIDIA.
   - Attendees can discuss the CUDA Python ecosystem, including libraries like CuPy and Numba, making it beneficial for developers at all skill levels.
- **Llama 3.2 now fully utilizes CuPy**: [Llama 3.2](https://github.com/githubpradeep/llm_np_cp/blob/main/llama3.2_model.py) is reported to be running completely with CuPy on GPU, indicating a significant advancement in performance optimization.
   - This integration underscores the increasing importance of leveraging GPU capabilities with libraries like CuPy for enhanced model efficiency.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.digitalocean.com/community/tutorials/an-introduction-to-gpu-optimization">An Introduction to GPU Performance Optimization for Deep Learning | DigitalOcean</a>: no description found</li><li><a href="https://github.com/NVIDIA/accelerated-computing-hub/tree/main/connect-with-experts">accelerated-computing-hub/connect-with-experts at main · NVIDIA/accelerated-computing-hub</a>: NVIDIA curated collection of educational resources related to general purpose GPU programming. - NVIDIA/accelerated-computing-hub
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[diffusion](https://discord.com/channels/1189498204333543425/1288899271193526342/1288899818743398420)** (11 messages🔥): 

> - `Creating a separate channel`
> - `Flux models benchmarks`
> - `DiffusionKit usage`
> - `Block diagram for Flux`
> - `Reddit diagram sharing` 


- **Separate channel for continued discussions**: A member initiated the creation of a separate channel for ongoing work, stating they can't let the **LLM flop maxers** have all the fun.
   - *Oh nice lol* was the response from another member, showing support for the move.
- **Obtaining benchmark numbers for Flux models**: A user inquired about having **benchmark numbers** for flux models across different hardware configurations.
   - Another member shared that they have **M2 Pro benchmarks**, indicating growing interest in performance metrics.
- **Testing non-quantized models**: One user expressed interest in testing **non-quantized models**, but mentioned that their **16GB of RAM** limits such tests.
   - This highlights the ongoing technical challenges users face when exploring performance evaluations.
- **Request for a diagram of Flux**: A user asked if anyone has a **drawn-out diagram** for flux, suggesting that visual aids could enhance understanding.
   - Discussion ensued about the usefulness of a **block diagram**, emphasizing visual representation in complex topics.
- **Sharing diagrams from Reddit**: A member referred to a diagram that was previously shared on **Reddit**, implying it's a valuable resource for understanding flux.
   - This shows the community’s tendency to leverage external sources for knowledge sharing.



**Link mentioned**: <a href="https://github.com/argmaxinc/DiffusionKit">GitHub - argmaxinc/DiffusionKit: On-device Inference of Diffusion Models for Apple Silicon</a>: On-device Inference of Diffusion Models for Apple Silicon - argmaxinc/DiffusionKit

  

---



### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1288576922648842405)** (135 messages🔥🔥): 

> - `AI Art Discussions`
> - `Stable Diffusion User Queries`
> - `Model Testing and Issues`
> - `Voice Conversion with RVC`
> - `User Experience with Lora Models` 


- **Debate on the Impact of User Interests in AI Advancements**: A member commented on how many technical advancements arise from niche interests, like **PonyDiffusion** fostering new art styles.
   - This sparked a discussion on the implications of fandoms on creativity and perceptions of AI-related content.
- **Queries About Stable Diffusion and GPU Requirements**: A newcomer to Stable Diffusion sought guidance on generating images without a GPU, asking about Google Colab options.
   - Others recommended **Kaggle** over Colab for its better resources, noting the need for a good GPU to run stable diffusion efficiently.
- **Testing and Issues with Lora Models**: A user faced difficulties with a Lora model, claiming it lacked any effect on their generated images.
   - After multiple inquiries, it was clarified that while the model produced subtle differences, it did not match the high-quality images expected from the Hugging Face examples.
- **Voice Conversion Queries on Google Colab**: A member requested assistance for installing **RVC** on Colab Pro for voice conversion tasks.
   - In response, it was noted that there are numerous **RVC models available** on Hugging Face that may be helpful.
- **Discussion on Image Generation Speed Variability**: One user observed significant variation in image generation time for the same parameters on their local setup.
   - This led to speculation about VRAM usage and system traffic potentially impacting output times.



**Link mentioned**: <a href="https://huggingface.co/bingbangboom/flux_dreamscape">bingbangboom/flux_dreamscape · Hugging Face</a>: no description found

  

---



### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1288623850178871306)** (2 messages): 

> - `Advertising Policy`
> - `Self-taught ML Engineer Introduction` 


- **No Advertising Allowed in Discussions**: There is a strict **no advertising** policy in the Discord, where sharing research is encouraged, but promoting companies, products, or events is not permitted.
   - Participants are reminded to refer to the specific channel for more details, enhancing clarity on community rules.
- **Self-taught ML Engineer Welcomed**: A new member introduced themselves as a **self-taught ML engineer** working on their startup and expressed interest in following along and contributing to the community's research.
   - This initiative highlights the desire for collaboration and knowledge sharing among community members.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1288576112510894110)** (67 messages🔥🔥): 

> - `MLM transformer with high masking`
> - `Qwen2 benchmarks concerns`
> - `FP6 vs BF16 performance`
> - `Hyperbolic Vision Transformer`
> - `Verbatim memorization in LLMs` 


- **Exploring High Masking Rates in MLM**: A proposal was made to train an MLM transformer with up to **85% of middle tokens masked**, utilizing a scheduled approach for masking similar to learning rates.
   - Discussions around whether such a strategy has been previously attempted or studied were initiated.
- **Concerns Raised About Qwen2 Benchmarks**: There have been reports that **Qwen2's VL benchmarks** were unreproducible twice, leading to suspicions regarding its performance in human preference evaluations.
   - Commentary indicated a consensus that the benchmark results seem dubious based on recent findings from the [Molmo release post](https://molmo.allenai.org/blog).
- **FP6 Outperforms BF16 in Quantization Tests**: **FP6** has been reported to outperform **BF16** across various evaluations and tests, with discussions on the potential reasons for these unexpected results.
   - Speculation about how quantization may introduce regularization effects and influence model pathways was also mentioned.
- **Introduction of Hyperbolic Vision Transformer**: A new paper introduced the **Hyperbolic Vision Transformer (HVT)**, which leverages hyperbolic geometry to enhance self-attention mechanisms.
   - The paper claims improved performance in image classification tasks, particularly on the **ImageNet dataset**.
- **Study of Verbatim Memorization Dynamics in LLMs**: Research exploring **verbatim memorization** in LLMs suggests specific conditions are necessary for it to occur based on sequence repetition and model states.
   - The study sheds light on implications for data privacy and offers methods to evaluate unlearning techniques.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://molmo.allenai.org/blog">no title found</a>: no description found</li><li><a href="https://x.com/lexin_zhou/status/1838961179936293098">Tweet from Lexin Zhou (@lexin_zhou)</a>: 1/ New paper @Nature!  Discrepancy between human expectations of task difficulty and LLM errors harms reliability. In 2022, Ilya Sutskever @ilyasut predicted: &#34;perhaps over time that discrepancy w...</li><li><a href="https://arxiv.org/abs/2409.16422">Is All Learning (Natural) Gradient Descent?</a>: This paper shows that a wide class of effective learning rules -- those that improve a scalar performance measure over a given time window -- can be rewritten as natural gradient descent with respect ...</li><li><a href="http://arxiv.org/abs/2409.15371">Bone: Block Affine Transformation as Parameter Efficient Fine-tuning Methods for Large Language Models</a>: As Large Language Models (LLMs) continue to grow in size, their computational and memory requirements increase correspondingly. Consequently, the exploration of cost-effective and efficient fine-tunin...</li><li><a href="https://arxiv.org/abs/2210.02671">A Logic for Expressing Log-Precision Transformers</a>: One way to interpret the reasoning power of transformer-based language models is to describe the types of logical rules they can resolve over some input text. Recently, Chiang et al. (2023) showed tha...</li><li><a href="https://x.com/rodrimora/status/1839329810864390611">Tweet from Rodri Mora aka Bullerwins (@rodrimora)</a>: @AlpinDale I&#39;ve run MMLU-Pro for the FP quantization using Aphrodite. Really good results for FP6-7 almost on par with bf16.  Full results here: https://docs.google.com/spreadsheets/d/17JUJPfDgeAY...</li><li><a href="https://arxiv.org/abs/2202.08005">Should You Mask 15% in Masked Language Modeling?</a>: Masked language models (MLMs) conventionally mask 15% of tokens due to the belief that more masking would leave insufficient context to learn good representations; this masking rate has been widely us...</li><li><a href="https://arxiv.org/abs/2409.16897">HVT: A Comprehensive Vision Framework for Learning in Non-Euclidean Space</a>: Data representation in non-Euclidean spaces has proven effective for capturing hierarchical and complex relationships in real-world datasets. Hyperbolic spaces, in particular, provide efficient embedd...</li><li><a href="https://arxiv.org/abs/2407.17817">Demystifying Verbatim Memorization in Large Language Models</a>: Large Language Models (LLMs) frequently memorize long sequences verbatim, often with serious legal and privacy implications. Much prior work has studied such verbatim memorization using observational ...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1288595616850772069)** (8 messages🔥): 

> - `Scaling laws dataset`
> - `Broken neural scaling laws`
> - `Chinchilla scaling laws`
> - `Google research on scaling laws` 


- **Searching for Scaling Laws Data**: A member is seeking a dataset containing datapoints showing the relationship between **# params**, **# tokens**, and **loss** for models trained with different parameters and dataset sizes, referencing the [Chinchilla scaling laws paper](https://arxiv.org/pdf/2405.15074).
   - They aim to explore the potential effects of a missing lower-order term without the burden of training numerous models.
- **Utilizing the Broken Neural Scaling Dataset**: Another member suggested using the **Broken Neural Scaling Laws** dataset, which was noted as comprehensive but lacking in architectural details.
   - This presents challenges for those wishing to replicate or augment the dataset with their own model training efforts.
- **Reference to Google Research GitHub**: A member pointed out that the paper references data shared by Google Research on GitHub at [this link](https://github.com/google-research/google-research/blob/master/revisiting_neural_scaling_laws/README.md).
   - While this repository provides raw datapoints, it lacks straightforward methods for replication or extension of experiments.
- **Challenges in Replicating Scaling Law Studies**: It was discussed that replicating actual scaling law studies can be inherently expensive and complex.
   - This highlights the difficulties faced by researchers attempting to verify findings or conduct similar experiments in the field.



**Link mentioned**: <a href="https://github.com/google-research/google-research/blob/master/revisiting_neural_scaling_laws/README.md">google-research/revisiting_neural_scaling_laws/README.md at master · google-research/google-research</a>: Google Research. Contribute to google-research/google-research development by creating an account on GitHub.

  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1288640717119422478)** (1 messages): 

> - `Filler tokens in LLMs`
> - `Parallelization of tasks`
> - `Supervision in natural language context` 


- **Exploring Filler Tokens in LLMs**: The discussion highlights that while success is noted in **synthetic tasks**, it does not necessarily translate to **LLMs** as emphasized in the conclusion.
   - *A critical question remains*: to what extent can LLMs effectively utilize **filler tokens** in their architectures?
- **Parallelization Limitations in NLP**: Participants stated that the observation regarding parallelization is limited to **token-parallelizable**, algorithmic problems but may not hold true for complex language tasks.
   - This raises further queries on the dynamics between **parallelizable supervision** and the nature of **natural language computation**.
- **Need for Adequate Supervision in Token Computation**: The conversation indicates uncertainty about whether natural language text can provide sufficient **supervision** for filler-token computation.
   - *Is natural language too nuanced?* The discussion suggests **instance-adaptive chains of thought** may be required for genuine language understanding.


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1288578173419917384)** (22 messages🔥): 

> - `Exact Match Metric Adjustment`
> - `Debugging Token Generation`
> - `Improving Documentation for Utils`
> - `Task Execution Default Behavior`
> - `Token Counting Script Enhancement` 


- **Exact Match Metric Adjustment in YAML**: A user indicated the need to change the `_aexams.yaml` to utilize the `exact_match` aggregation metric instead of `acc`, noting where to find further details.
   - Another user confirmed they implemented the change but encountered issues with generating an overall group aexam.
- **Debugging Token Generation Issues**: A member reported an error related to exceeding maximum sequence length during token generation processes, debugging into specific lines of code to trace the issue.
   - It was acknowledged that overriding `tok_batch_encode` might be contributing to the problem, prompting a review to resolve it.
- **Improving Documentation for Utils**: A member expressed gratitude for highlighting documentation gaps and proposed creating an issue to improve clarity around the utils such as `Collator` and `Reorderer`.
   - A specific GitHub issue was referenced for tracking enhancements needed in the documentation.
- **Task Execution Default Behavior Clarified**: Clarifications were made regarding the default split used when executing tasks, with the `test` split being the standard unless specified otherwise in task yaml.
   - It was noted that fewshots are sourced from validation or training sets when available, otherwise drawn from the test set.
- **Token Counting Script Enhancement Discussed**: A member sought to count tokens during generation using `write_out.py`, noting discrepancies in `cost_estimate.py` and expressing the desire for consistent reproduction of conditions.
   - A user shared a draft PR to update the script for token counting, mentioning that further updates would be necessary to address cost estimates accurately.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/EleutherAI/lm-evaluation-harness/pull/2359).">Build software better, together</a>: GitHub is where people build software. More than 100 million people use GitHub to discover, fork, and contribute to over 420 million projects.</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/issues/2358">Improve `docs/model_guide.md` with skeleton template code + description of utils like `Collator` and `Reorderer` · Issue #2358 · EleutherAI/lm-evaluation-harness</a>: As per title, some of the machinery we use in HFLM around the ordering, batching, and caching of requests is still opaque to most users. It&#39;d be great if we could expand model_guide.md with: in te...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/)** (1 messages): 

sky.moo: How can I run vision llms locally?
  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1288766396242595892)** (4 messages): 

> - `FA3 support`
> - `Training on H100s`
> - `Testing assistance`
> - `Grant opportunities from Modal` 


- **FA3 Integration on H100s in the Works**: Members discussed the potential to add **FA3** to the todo list for training smaller models on **H100s**, with a suggestion that integrating it may require only a few lines of code.
   - However, consistent access to **H100s** is currently a limitation, and testing remains a challenge.
- **Willingness to Test on H100s**: One member offered to help with testing **FA3** on **H100s** for small models, expressing willingness to learn and contribute.
   - They requested pointers on how to approach integrating **FA3**, which is still in beta according to the official repository.
- **Modal Grants for Project Support**: The possibility of securing a grant from **Modal** was suggested as a means to improve access to **H100s** for the project.
   - Modal is known for actively supporting projects in this domain, which could help overcome the current access limitations.


  

---



### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1288575814090096781)** (87 messages🔥🔥): 

> - `Perplexity AI context retention issues`
> - `Llama 3.2 release`
> - `Cloudflare verification frustrations`
> - `Feature request submissions`
> - `AI model capabilities in lyric generation` 


- **Perplexity AI struggles with context retention**: Users expressed concerns about **Perplexity AI** failing to remember the context of previous questions, especially after follow-ups, which seems to have worsened recently.
   - One user noted, *'this platform is still useful day-to-day but has definitely gotten worse.'*
- **Excitement over Llama 3.2 Launch**: A member shared that **Llama 3.2** has been released on llama.com, prompting excitement among users with a 'LFG' call to action.
   - Another member mentioned not seeing it appear on Perplexity's interface.
- **Frustration with Cloudflare Verification**: Users vented frustration about **Cloudflare's** repeated verification processes, asserting even Pro users are significantly inconvenienced by this hurdle.
   - One noted, *'it makes me question the worth of my subscription'*, highlighting the impact on user experience.
- **How to Submit Feature Requests**: Members discussed how to **submit feature requests** for Perplexity, with suggestions to email support for issues related to account management.
   - One user asked whether their Pro subscription could be transferred to a friend's account, receiving guidance on the process.
- **Limitations on Lyric Generation**: A user lamented their inability to get **Perplexity** to generate fun lyrics, feeling it was not responding to creative prompts as it used to.
   - In response, various solutions were discussed, including switching to other models like Sonar for lyric generation tasks.



**Link mentioned**: <a href="https://x.com/apostraphi/status/1839303673480098285?s=61">Tweet from Phi Hoang (@apostraphi)</a>: haystacks don&#39;t stand a chance

  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1288620910269698091)** (8 messages🔥): 

> - `Mira Murati departs OpenAI`
> - `AI vs. reCAPTCHA`
> - `Energy consumption of AI`
> - `Garbage In, Garbage Out`
> - `Bike lanes in cities` 


- **Mira Murati's Departure from OpenAI**: Mira Murati has officially **departed OpenAI**, raising discussions about the implications for the organization and AI tech. This has opened up new conversations around talent migration in the industry, as highlighted in a recent [YouTube video](https://www.youtube.com/embed/zk1GwCIEvVU).
   - *James Cameron's shift* was also mentioned, alongside other intriguing topics, perhaps hinting at evolving narratives in tech.
- **AI Trumps reCAPTCHA Challenges**: An analysis pointed out that **AI beats reCAPTCHA** systems, bringing attention to advancements in AI capabilities. Members discussed the potential repercussions for online security, with calls for updated methodologies.
   - [Details here](https://www.perplexity.ai/search/ai-beats-recaptcha-IJvLzX98RkeMdh.kpXIqXw).
- **Understanding Garbage In, Garbage Out**: A discussion around **Garbage In, Garbage Out** emphasized the challenges in AI training and data quality. This concept was expanded upon in a detailed [article](https://www.perplexity.ai/search/garbage-in-garbage-out-underst-RhMLBvuiSYycAm0BY4o9nw), shedding light on common pitfalls.
   - Further insights were provided in the [converted page](https://www.perplexity.ai/page/garbage-in-garbage-out-Z5ZG6D5ZScCBOFJ4JYGhQg).
- **Investigating Energy Consumption of AI**: A report highlighted the **energy consumption of AI models**, prompting discussions on sustainability within AI development. The findings led to inquiries on how resources are managed in scaling AI technologies.
   - [Find the summary here](https://www.perplexity.ai/search/how-much-energy-does-ai-and-ll-jcyr1ba3S9eVkzCinpLk_g#0).
- **Cities with Most Bike Lanes**: The topic of **most bike lanes in cities** sparked interest as members shared insights into urban planning and infrastructure. This may signal a growing trend towards sustainable city living.
   - [Explore more about this topic](https://www.perplexity.ai/search/city-with-the-most-bike-lanes-hhNCIS6oRRCli0fdq8Z32g).



**Link mentioned**: <a href="https://www.youtube.com/embed/zk1GwCIEvVU">YouTube</a>: no description found

  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1288589988216307713)** (4 messages): 

> - `Perplexity in Zapier`
> - `System and User Content`
> - `External Models Availability` 


- **Clarifying Perplexity Structure in Zapier**: A member is using Perplexity inside of Zapier and seeks clarification on its structure, particularly with webhooks.
   - *Is there a specific format for how messages should be structured?*
- **Understanding System and User Content**: Another member confirmed that **System content** refers to instructions given to the AI model, while **User content** is the input provided to the model.
   - This distinction is crucial for effective interaction within the Perplexity framework.
- **Questioning External Models Availability**: A member questioned if external models are only accessible through the web interface and not via the API.
   - *They inquired about scripting options for web interface usage to be supported by Perplexity.*


  

---



### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1288578108609265715)** (76 messages🔥🔥): 

> - `Meta AI Access Issues`
> - `Llama 3.2 Licensing Concerns`
> - `AI IDE Recommendations`
> - `Challenges with AI Search Tools`
> - `Reinforcement Learning Resources` 


- **Meta AI Access Remains Restricted**: Several members expressed frustration about accessing **Meta AI**, noting that especially users outside of the U.S. face restrictions despite attempts to use VPNs.
   - One user highlighted that the **Llama 3.2 license** makes it notably EU incompatible, causing further access issues.
- **Llama 3.2 Launch Highlights**: With the launch of **Llama 3.2**, several members discussed the new multimodal models and their compatibility limits, especially relating to EU users.
   - The lack of access to essential AI tools was underlined with the mention of [Hugging Face](https://huggingface.co/blog/llama32) hosting these models.
- **Best AI IDE Options for Game Development**: For game development, members recommended various AI IDE options including **Cursor**, **Continue**, **Amazon Q**, and **GitHub Copilot** for code generation.
   - One user noted their personal setup using **ChatGPT** alongside SSH actions to integrate real-time code modifications.
- **Frustrations with AI Search Tools**: Users expressed dissatisfaction with existing AI search tools, particularly regarding features like multi-step responses, citing **Perplexity** and its limitations without a Pro subscription.
   - Members discussed the challenges of creating effective search engines and the complexities involved in indexing the internet.
- **Resources for Reinforcement Learning**: Inquiries were made about resources for learning **Reinforcement Learning**, with recommendations to leverage **ChatGPT 4o** for in-depth information and references.
   - Members also pointed out the abundance of research papers available online, while suggesting YouTube for practical applications of RL.



**Link mentioned**: <a href="https://huggingface.co/blog/llama32">Llama can now see and run on your device - welcome Llama 3.2</a>: no description found

  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1288636990140584027)** (20 messages🔥): 

> - `Advanced Voice Mode Limitations`
> - `File Upload Issues in o1`
> - `Performance Comparison of GPT Models`
> - `Concerns About Policy Violations`
> - `Anticipated Features for Future Releases` 


- **Advanced Voice Mode can't search Internet**: Users expressed frustration that **Advanced Voice Mode** currently lacks Internet search capability, requiring a new chat to switch back from voice to text input.
   - Despite this limitation, many are optimistic about the future of this feature alongside upcoming models like ChatGPT-5.
- **File uploads missing in o1**: Members lamented the inability to upload files to **o1**, which forces them to revert to using **GPT4o** for processing, impacting their productivity.
   - One member recalled *Sam Altman’s quote* about GPT-4, expressing disappointment at having to juggle data between models.
- **o1 outshines GPT4o in reasoning**: Discussion highlighted that **o1** excels in following extensive instructions and has proven significantly more capable compared to **GPT4o**, which struggles with basic tasks.
   - One user noted that using **Python tools** might be necessary for tracking characters and estimating page counts in their writing.
- **Fiction Writing Compliance Woes**: A user worried about using ChatGPT for editing a violent fictional novel due to potential violations of usage policies, especially after receiving warnings.
   - Another member advised that constant warnings could lead to account scrutiny, urging compliance with terms of service.
- **Hopeful for Future Feature Rollouts**: Members discussed their expectations for future functionalities like **file uploads**, **memory in o1**, and integration with GPTs, emphasizing the need for patience in the rollout process.
   - As one member put it, 'Everything with OpenAI is rolled out slowly and carefully,' reflecting collective optimism for enhancements ahead.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/)** (1 messages): 

blckreaper: not if you use o1 mini it will tell you you suck and to do better
  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/)** (1 messages): 

blckreaper: not if you use o1 mini it will tell you you suck and to do better
  

---



### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1288584856971776095)** (20 messages🔥): 

> - `3.2 license comparison`
> - `OpenAI leadership changes`
> - `Molmo pointing feature excitement`
> - `Barret Zoph departure`
> - `Opus project updates` 


- **Noticeable changes in 3.2 license**: A member pointed out that the **3.2 license** has significant differences compared to **3.1**, urging a closer look at the [diff](https://www.diffchecker.com/O4ijl7QY/).
   - This raises questions about the implications of the changes among developers and organizations using this license.
- **Shifting sands in OpenAI leadership**: Recent discussions indicate that several key figures including the **Chief Technology Officer** are resigning, stirring speculation about the company's direction.
   - One member noted the unusual timing of these departures, suggesting tensions within the team.
- **Molmo pointing feature generates buzz**: Excitement bubbled as a member expressed that the **Molmo pointing feature** could be more impactful for product building than a higher **AIME** score, claiming it 'passes the vibe check'.
   - This reflects a sentiment among some developers that new features can provide immediate value beyond mere performance metrics.
- **Barret Zoph announces departure**: In a heartfelt post, **Barret Zoph** announced he is leaving OpenAI, reflecting on his positive experiences and contributions to the **post-training team**.
   - He emphasized gratitude towards his colleagues and optimism for OpenAI's future, indicating a personal career evolution.
- **Uncertainty around Opus launch**: Several members expressed doubts about the **Opus** project, with one member jokingly remarking about the potential fallout of betting their house on it.
   - Members are feeling anxious about the uncertain timeline for Opus and how it could affect their expectations.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/BenjaminDEKR/status/1839112669808275816">Tweet from Benjamin De Kraker 🏴‍☠️ (@BenjaminDEKR)</a>: Wait so.... Sam&#39;s original &#34;note&#34; earlier today only addressed Mira leaving.  Then he deleted it and posted the new one (up now), where he groups them all together and says &#34;it made se...</li><li><a href="https://x.com/bobmcgrewai/status/1839099787423134051?t=3SqLlhVJgATHMsRWY7neTA&s=19">Tweet from Bob McGrew (@bobmcgrewai)</a>: I just shared this with OpenAI:</li><li><a href="https://x.com/miramurati/status/1839025700009030027?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Mira Murati (@miramurati)</a>: I shared the following note with the OpenAI team today.</li><li><a href="https://x.com/andersonbcdefg/status/1839030313659564424">Tweet from Ben (e/treats) (@andersonbcdefg)</a>: i havent felt as excited about a new AI model ability in a while as i do about the Molmo pointing feature. for someone trying to build a product (not a god) i would argue this might be more impactful ...</li><li><a href="https://x.com/bobmcgrewai/status/1839099787423134051?t=3SqLlhVJgATHMsRWY7">Tweet from Bob McGrew (@bobmcgrewai)</a>: I just shared this with OpenAI:</li><li><a href="https://x.com/barret_zoph/status/1839095143397515452?s=46">Tweet from Barret Zoph (@barret_zoph)</a>: I posted this note to OpenAI.  Hey everybody, I have decided to leave OpenAI.   This was a very difficult decision as I have has such an incredible time at OpenAI. I got to join right before ChatGPT a...</li><li><a href="https://www.diffchecker.com/O4ijl7QY/">llama 3.2 vs 3.1 - Diffchecker</a>: llama 3.2 vs 3.1 - LLAMA 3.1 COMMUNITY LICENSE AGREEMENT Llama 3.1 Version Release Date: July 23, 2024  “Agreement” mea
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1288586685629857943)** (44 messages🔥): 

> - `OpenAI leadership changes`
> - `Profit interest structures`
> - `California oversight on non-profits`
> - `Impact of Anthropic on OpenAI`
> - `Tax implications for employees` 


- **OpenAI's Leadership Shakeup Causes Concern**: Discussion centered on the recent leadership changes at OpenAI, with multiple members commenting that *ALL of OG OpenAI left besides Sam* and questioning if this is a negative signal.
   - One noted, *This shit all stinks so much. So sketchy,* reflecting widespread apprehension about the direction OpenAI is taking.
- **Profit Interests Units Raise Eyebrows**: Members debated the appropriateness of offering *Profit Interests Units (PIUs)* in a non-profit, with some suggesting it feels like an exploit of a loophole that may prompt regulatory actions.
   - One remarked that it seems uncomfortable to raise money as a non-profit, then convert to for-profit while acquiring equity, highlighting industry concerns over tax implications.
- **California May Investigate OpenAI's Structure**: Discussion emerged around California's role in overseeing non-profits, with a member suggesting that the recent changes could expose OpenAI to scrutiny from the state's Attorney General.
   - This potential investigation may have significant implications in the broader context of AI legislation and governance.
- **Concerns Over Anthropic's Growing Influence**: Members speculated on the implications if *Logan* were to leave OpenAI for Anthropic, with one quipping that it would cause the *world to really break.*
   - Another suggested that Anthropic is perceived as a *much less unserious company*, raising questions about their competitive stance in the AI landscape.
- **Tax Matters Complicating Employee Compensation**: Members discussed the tax implications of profit interests, noting that employees receiving profit interests could face different treatments compared to traditional RSUs, which are taxed immediately upon vesting.
   - One remarked, *It does seem weird to give any sort of “equity” compensation in a non-profit though,* highlighting the ongoing confusion in employee compensation structures.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://vxtwitter.com/Yuchenj_UW/status/1839030011376054454">Tweet from undefined</a>: no description found</li><li><a href="https://www.levels.fyi/blog/openai-compensation.html">OpenAI PPUs: How OpenAI&#39;s unique equity compensation works</a>: A look at one of the hottest and most secretive AI companies today.</li><li><a href="https://riveron.com/posts/accounting-for-pius/">Accounting for Profits Interests Units - Riveron</a>: Many companies today seek to retain key talent through profits interests units (PIUs). Here’s what accounting teams should know.</li><li><a href="https://x.com/miramurati/status/1839025700009030027?s=46">Tweet from Mira Murati (@miramurati)</a>: I shared the following note with the OpenAI team today.</li><li><a href="https://news.bloomberglaw.com/us-law-week/openais-example-is-lesson-for-nonprofit-tandem-entity-formation">OpenAI&#x27;s Example Is Lesson for Nonprofit Tandem Entity Formation</a>: UC Berkeley’s Jesse Finfrock explains the pressures companies face to use AI to generate profit and the necessary considerations for entity formation.
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1288584168036630569)** (21 messages🔥): 

> - `LLaMA Stack`
> - `FTC AI Crackdown`
> - `NeurIPS Submission Rejections`
> - `Rewardbench Benchmark`
> - `C++ Knowledge in Academia` 


- **LLaMA Stack Confusion**: A discussion ensued about whether the **LLaMA Stack** integration tools are significant, with a member expressing indifference after seeing it mentioned in a blog post.
   - Another member remarked that it seems unimportant, viewing it just as tool integrations.
- **FTC Announced AI Crackdowns**: A tweet revealed that the **FTC** announced new crackdowns related to AI, sparking skepticism about its effectiveness and asserting that many scams exist in the field.
   - Members pondered the vague statement regarding what constitutes 'real AI', reflecting confusion over regulatory definitions.
- **NeurIPS Rejection Makes Waves**: A member humorously highlighted the rejection of **Rewardbench** from the **NeurIPS D&B**, describing it as wildly funny despite the sadness of the outcome.
   - It was noted that the rejection scores included a damaging personal attack, and the project was the first benchmark in its area, which seemed to have been overlooked.
- **C++ Knowledge Critique**: Members reacted to a NeurIPS rejection where a submission was criticized for being written in **C++**, questioning the relevance of knowing this language in academia.
   - One member commented on the prevalence of C++ in their coursework, demonstrating frustration over such narrow criticism.
- **Social Media Responses**: A member expressed the view that complaining about the situation on Twitter contributes little to resolving it and that the project itself remains successful.
   - They reflected on the different consequences if students were dependent on the rejected benchmark.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://fxtwitter.com/cdolan92/status/1839024340689371356">Tweet from Charlie Dolan (@cdolan92)</a>: FTC announced AI related crackdowns  Hot Take: Good! There&#39;s a lot of scams On further reading: WTF!?  Example: what in the world does this mean?  &#34;...our technologists can figure out [if your...</li><li><a href="https://x.com/eugenevinitsky/status/1839275401903780300?s=46">Tweet from Eugene Vinitsky 🍒 (@EugeneVinitsky)</a>: Fairly high score NeurIPS submission rejected because “simulator is written in C++ and people don’t know C++” is a fun new low
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1288580074018312203)** (6 messages): 

> - `Molmo model skepticism`
> - `Impact of social media`
> - `LLaMA vs Molmo performance` 


- **Skepticism surrounds Molmo's praises**: Members expressed suspicion over the high volume of posts claiming that **Molmo** outperforms **LLaMA 3.2**, questioning the authenticity of such claims.
   - However, one member suggested testing the model personally, stating that *there's no proof* of third-party payment for positive reviews, emphasizing that it's easy to validate results.
- **Debate on the value of social media followers**: The discussion sparked curiosity about whether accumulating **tweets and followers** is truly beneficial, with one member humorously questioning its worth.
   - Another member suggested it might be better for one's **net presence**, adding a lighthearted tone to the conversation.
- **Clarification on Molmo's release timing**: A member clarified that **Molmo** was announced on the same day as **LLaMA 3.2**, but a few hours earlier, correcting initial confusion about its release timeline.
   - They underscored their personal experience with the model, noting its impressive performance.



**Link mentioned**: <a href="https://old.reddit.com/r/LocalLLaMA/comments/1fptwvm/molmo_a_new_model_that_outperforms_llama_32/lp1utnq/">Molmo - A new model that outperforms Llama 3.2, available in the E.U</a>: Posted in r/LocalLLaMA by u/franklbt • 104 points and 51 comments

  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1288579946385772658)** (6 messages): 

> - `Meeting duration`
> - `One-on-ones`
> - `Project leadership` 


- **3.5 Hours of Meetings Seems Unusual**: A member expressed surprise at the **3.5 hours** of meetings scheduled for the day, describing it as **crazy**.
   - The member noted that despite this long duration, it's preferable to have these meetings earlier in the day.
- **Chilled Meeting Structure**: Another member mentioned that the meetings scheduled are **chill**, referring to a more relaxed approach.
   - This includes one-on-ones and discussions related to the project they are leading.
- **Few Meetings Overall**: A member noted that they don't have many meetings in general, hinting at a more focused work style.
   - They suggested a preference for stacking meetings when they do occur.


  

---



### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1288716719635959919)** (2 messages): 

> - `Vision Llama Release`
> - `Gemini Tokenization Changes`
> - `Gemini Pricing Adjustments`
> - `OpenRouter Endpoints` 


- **Vision Llama Hits OpenRouter with Free Endpoint**: The first vision **Llama** is now available on **OpenRouter**, featuring a [free endpoint](https://x.com/OpenRouterAI/status/1839157099747479553). In total, **five new endpoints** have been introduced, powered by multiple providers.
   - Users are encouraged to enjoy the latest features, marked by the celebratory icon 🎁🦙.
- **Gemini Tokenization Simplifies Costs**: OpenRouter will transition to counting **tokens** instead of characters for Gemini models, reducing apparent token counts by a factor of **~4**. This aims to normalize and cut costs for developers.
   - These changes will also lead to **doubling** of current prices as they align tokens to a per-token pricing model, which is set to adjust further after **October 1**.
- **Upcoming Gemini Pricing Adjustments**: As part of the upcoming changes, **Gemini** prices will be adjusted to match current lower tier per-token prices from AI Studio. This pricing strategy is designed to enhance the developer experience and standardization.
   - The anticipated price cuts on **October 1** promise further reductions for users, leading to an overall improved cost structure moving forward.



**Link mentioned**: <a href="https://x.com/OpenRouterAI/status/1839157099747479553">Tweet from OpenRouter (@OpenRouterAI)</a>: Five new endpoints for today&#39;s Llama 3.2 release are now live  Including a free vision Llama! 🎁🦙

  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1288579887602470912)** (87 messages🔥🔥): 

> - `OpenRouter Updates`
> - `Llama Model Restrictions`
> - `Chat Completion Models`
> - `Gemini Performance Changes`
> - `Math Quiz Rewriting Models` 


- **OpenRouter Credits and Invoice Issues**: Users reported difficulties with credit transactions and visibility on OpenRouter, stating that transactions might take time to appear after payments are made, as illustrated by a user who eventually received their credits.
   - Another mentioned that a backend delay or provider issues might be causing disruption in seeing transaction history.
- **Llama 3.2 Restrictions for EU Users**: Meta's policy on the use of their vision models in the EU raises concerns over the accessibility and legality for users in that region, especially regarding inference provision.
   - Members noted confusion regarding the implications of provider locations and licenses, suggesting that compliance with Meta's rules could be problematic.
- **Chat Completion Model Limitations**: Users discussed perceived limitations in OpenRouter's support for completion models, indicating that such functionality may require specific templates or conditions to operate correctly, particularly for the Codestral Mamba model.
   - The conversation highlighted a general lack of completion support and raised questions about model capabilities within OpenRouter.
- **Gemini Token Count Anomalies**: There were reports of unexpected jumps in the input/output token count on Gemini, with users speculating about possible underlying changes or errors affecting their results.
   - A participant suggested the token count includes hidden reasoning tokens, leading to confusion among users about the actual token usage.
- **VLM Support in vLLM**: Discussion centered on vLLM's evolving support for Vision Language Models (VLMs), with feedback indicating that while there were previously issues, recent updates have improved functionality.
   - Users confirmed that vLLM's support hinges on compatibility with Hugging Face's transformers, suggesting a dependency on external frameworks for new architectures.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.llama.com/llama3_2/use-policy/">Llama 3.2 Acceptable Use Policy</a>: Llama 3.2 Acceptable Use Policy</li><li><a href="https://openrouter.ai/credits">Credits | OpenRouter</a>: Manage your credits and payment history
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[beta-feedback](https://discord.com/channels/1091220969173028894/1277894087755829278/1288923984175431762)** (2 messages): 

> - `Bring Your Own Key Beta Test` 


- **Request for BYOK Beta Participation**: A member inquired about the possibility of being included in the **Bring Your Own Key (BYOK)** beta test.
   - They offered to provide their **email address** via direct message to facilitate the process.
- **Willingness to Share Contact**: The same member expressed their willingness to share personal contact information to assist with the beta participation process.
   - They specifically mentioned being happy to **DM** their email address if needed.


  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1288581042885623914)** (65 messages🔥🔥): 

> - `License Compliance Issues`
> - `OpenAI's CTO Resignation`
> - `Mistral Discussions`
> - `Open Source Advancements`
> - `Continued Pretraining Example` 


- **License Compliance Causes Frustration**: Members discussed compliance with licensing, highlighting that **EU access is blocked** due to disagreements with regulations, resulting in frustrations over access limitations.
   - One member humorously remarked that **Mistral is now a meme**, pointing to the absurdity of the situation.
- **OpenAI’s CTO Resignation Sparks Speculation**: The resignation of OpenAI's CTO was mentioned, with one member joking that it leads to speculation about the current state of the company.
   - Concerns were raised about the direction of OpenAI, with suggestions that internal issues could make for an interesting Netflix mini-series.
- **Impressive Capabilities of New Molmo Models**: The recent **Molmo models** received praise for their ability to **point locations in images**, showcasing advancements in open-source development.
   - Furthermore, **voice-annotated image training** methods were discussed, indicating a significant leap towards integrating multimodal datasets.
- **Concerns About AI Military Applications**: Members speculated about the potential military applications of AI models, expressing concerns about tasks like identifying military targets or estimating altitudes from drone footage.
   - One member shared that their tests showed **superb performance** in identifying crucial objects in low-quality aerial images, raising ethical questions.
- **Need for Continued Pretraining Example**: A member inquired about examples of **continued pretraining** for a specific model, asking about configurations and dataset handling.
   - This reflects a growing interest in practical implementations of various AI techniques and suggests a collaborative effort to optimize training processes.



**Link mentioned**: <a href="https://x.com/miramurati/status/1839025700009030027?t=pVYyCN8C7RnV0UruM9H2Lg&s=19">Tweet from Mira Murati (@miramurati)</a>: I shared the following note with the OpenAI team today.

  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1288930053371072565)** (4 messages): 

> - `LoRA+ Issue`
> - `Multimodal/VLM Assistance` 


- **Commit breaks LoRA+ functionality**: A member reported that [this commit](https://github.com/axolotl-ai-cloud/axolotl/commit/b98d7d7098f5d64a07c5a96855c4e08dca7afd91) appears to have broken **LoRA+** when used with the official runpod template, instead of replacing it.
   - Another member inquired if the issue is related to the newly added `loraplus_lr_ratio` in the YAML configuration.
- **Seeking Help with Multimodal/VLM**: A member expressed openness to assistance with **multimodal** and **VLM** projects, noting they are currently facing challenges due to pre-processing optimizations.
   - They mentioned that the pre-processing of data for multi-modal operations isn't yielding optimal results.
- **Possible Configuration Issues with LoRA+**: Another member speculated whether the problem is linked to the parameters `loraplus_lr_ratio` or `loraplus_lr_embedding` that were recently added.
   - They shared their experience of the functionality failing after spinning up a new pod a few hours after making changes.


  

---


### **OpenAccess AI Collective (axolotl) ▷ #[runpod-help](https://discord.com/channels/1104757954588196865/1162430527215763569/1288669230237224970)** (3 messages): 

> - `Llama 3.2 Inference`
> - `GPU Requirements`
> - `Runpod H100 GPUs`
> - `Quantization Effects` 


- **Planning for Llama 3.2 inference**: An inquiry was made regarding how many **H100 GPUs** are needed to inference **Llama 3.2** with **90 billion parameters**, to avoid out-of-memory errors.
   - The user aims to fetch the **Runpod GPUs** but wants to ensure they can handle the model without needing to delete them due to OOM issues.
- **Exploring VRAM requirements**: There was a question on the amount of **GB VRAM** required for efficient inference of **Llama 3.2**.
   - This reflects the user's intention to optimize their GPU allocation for running large models.
- **Quantization considerations raised**: A member inquired about the quantization option being considered for the model by asking, *What quant?*
   - This indicates that quantization might affect GPU requirements and overall model performance.


  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-phorm-bot](https://discord.com/channels/1104757954588196865/1225558824501510164/1288658334622154842)** (5 messages): 

> - `Tokenizer Padding Token`
> - `Adding Special Tokens`
> - `Model Resizing for Tokenization` 


- **Tokenizer lacks padding token**: A user raised the issue of a tokenizer missing a padding token during pretraining, which can disrupt processing of variable-length input sequences.
   - *Options provided include setting the pad token to the EOS token or adding a new pad token* using `tokenizer.add_special_tokens({'pad_token': '[PAD]'})`.
- **Setting the pad token to EOS**: To use an existing token, one can assign the padding token as the end-of-sentence token with `tokenizer.pad_token = tokenizer.eos_token`.
   - This solution allows the tokenizer to treat a designated token as both padding and end-of-sequence, streamlining the input process.
- **Resizing model embeddings after adding tokens**: When a new padding token is added, it is necessary to update the model's embeddings using `model.resize_token_embeddings(len(tokenizer))`.
   - This ensures that the model correctly accounts for the newly added token in its operations and retains compatible input processing.
- **Example for tokenization with padding**: A complete code snippet was shared to demonstrate both approaches for handling padding tokens in a tokenizer setup.
   - The example includes the process of tokenizing inputs while ensuring appropriate padding is applied to maintain uniform input lengths.



**Link mentioned**: <a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=ff7885b4-7628-45b9-8547-0f162351c8d1)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.

  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1288576581899653160)** (73 messages🔥🔥): 

> - `Mira Murati's Departure`
> - `Meta's Orion AR Glasses`
> - `Google's AlphaChip`
> - `AI Product Creation Platform Arcade`
> - `GitHub Copilot Browser Integration` 


- **Mira Murati departs from OpenAI**: [Mira Murati](https://x.com/miramurati/status/1839025700009030027?s=46) announced her departure from OpenAI, prompting gratitude from Sam Altman and reflections on her contributions over the last 6.5 years.
   - This change appears to coincide with other key departures, leading to speculation about a shift in leadership dynamics.
- **Meta unveils Orion AR glasses**: Meta introduced [Orion](https://about.meta.com/realitylabs/orion), claiming it to be their most advanced AR glasses yet, designed for seamless integration of digital experiences into the physical world.
   - Initial reactions highlight its aesthetic appeal, although the company has opted not to sell the product due to manufacturing complexities.
- **Google releases AlphaChip and its weights**: Google announced [AlphaChip](https://x.com/googledeepmind/status/1839306984480231852?s=46), revolutionizing microchip design, and made its model weights available for public use.
   - The chip is touted for its role in designing TPUs for AI models and is a major advancement for Google's chip production capabilities.
- **Arcade raises $17M for AI product creation platform**: Arcade announced raising $17M to develop the first AI product creation platform, marketing it as the tool for turning creative ideas into tangible products.
   - The vision centers on democratizing product development through AI, highlighting the platform's potential impact on innovation.
- **GitHub Copilot now available for browsers**: GitHub Copilot has extended its functionality to browsers, allowing developers to access its features from more flexible environments.
   - This change aligns it with competitors like Sourcegraph Cody Chat and emphasizes the importance of detailed documentation for maximizing its capabilities.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/miramurati/status/1839025700009030027?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Mira Murati (@miramurati)</a>: I shared the following note with the OpenAI team today.</li><li><a href="https://x.com/barret_zoph/status/1839095143397515452?s=46">Tweet from Barret Zoph (@barret_zoph)</a>: I posted this note to OpenAI.  Hey everybody, I have decided to leave OpenAI.   This was a very difficult decision as I have has such an incredible time at OpenAI. I got to join right before ChatGPT a...</li><li><a href="https://x.com/RihardJarc/status/1839014234266755473">Tweet from Rihard Jarc (@RihardJarc)</a>: $META&#39;s iPhone moment</li><li><a href="https://x.com/Teknium1/status/1839040366512844917">Tweet from Teknium (e/λ) (@Teknium1)</a>: So you didn&#39;t get HER with OpenAI&#39;s advanced voice mode - well - my friends at http://Play.AI have the real deal, HERmes.  Try it here: https://play.ai/agent/HERMES-m3i3jU81_52ruL6_0tw2R</li><li><a href="https://x.com/googledeepmind/status/1839306984480231852?s=46">Tweet from Google DeepMind (@GoogleDeepMind)</a>: Our AI for chip design method AlphaChip has transformed the way we design microchips. ⚡  From helping to design state-of-the-art TPUs for building AI models to CPUs in data centers - its widespread im...</li><li><a href="https://x.com/mnaficy/status/1839342011788439580?s=46&t=b7l37rB6wtbyAh6ah1NpZQ">Tweet from Mariam Naficy (@mnaficy)</a>: 1/ Announcing $17M in total fundraising to build Arcade, the world&#39;s first AI product creation platform  Quoting Arcade (@arcade_ai)   Introducing Arcade 1.0, the first-ever AI product creation pl...</li><li><a href="https://x.com/willdepue/status/1839098732534722570">Tweet from will depue (@willdepue)</a>: the second set of resignations has hit openai headquarters</li><li><a href="https://x.com/andrewcurran_/status/1839037623756796196?s=46">Tweet from Andrew Curran (@AndrewCurran_)</a>: Wow.</li><li><a href="https://x.com/bobmcgrewai/status/1839099787423134051?s=46">Tweet from Bob McGrew (@bobmcgrewai)</a>: I just shared this with OpenAI:</li><li><a href="https://www.theverge.com/24253908/meta-orion-ar-glasses-demo-mark-zuckerberg-interview">Hands-on with Orion, Meta’s first pair of AR glasses</a>: Meta’s AR glasses are an impressive tease of what Mark Zuckerberg thinks will replace the smartphone.</li><li><a href="https://x.com/multimodalart/status/1839060917960474671">Tweet from apolinario 🌐 (@multimodalart)</a>: testing out the Diffusers Image Fill demo capabilities on a random image</li><li><a href="https://gist.github.com/csellis/4b08be1757b545d1497853df0b0d730e">CSAT.md</a>: GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://x.com/gdb/status/1839391073296408577">Tweet from Greg Brockman (@gdb)</a>: I have deep appreciation for what each of Barret, Bob, and Mira brought to OpenAI. We worked together for many years, and we are all members of the team that helped make OpenAI what it is today.  They...</li><li><a href="https://x.com/benthompson/status/1839065543766429926?s=46">Tweet from Ben Thompson (@benthompson)</a>: They’re real and they’re spectacular.</li><li><a href="https://x.com/Yuchenj_UW/status/1839030011376054454">Tweet from Yuchen Jin (@Yuchenj_UW)</a>: How it started vs how it’s goin.  Quoting Sam Altman (@sama)   I replied with this. Mira, thank you for everything.  It’s hard to overstate how much Mira has meant to OpenAI, our mission, and to us al...</li><li><a href="https://x.com/sama/status/1839093415226524114">Tweet from Sam Altman (@sama)</a>: i just posted this note to openai:  Hi All–  Mira has been instrumental to OpenAI’s progress and growth the last 6.5 years; she has been a hugely significant factor in our development from an unknown ...</li><li><a href="https://x.com/boztank/status/1838999636402647453">Tweet from Boz (@boztank)</a>: We just unveiled Orion, our full AR glasses prototype that we’ve been working on for nearly a decade. When we started on this journey, our teams predicted that we had a 10% chance (at best) of success...</li><li><a href="https://x.com/JacquesThibs/status/1839030342788809062">Tweet from Jacques (@JacquesThibs)</a>: oh my god. @gwern was right, *again*:  Quoting Mira Murati (@miramurati)   I shared the following note with the OpenAI team today.</li><li><a href="https://x.com/ArtificialAnlys/status/1838993593056247965">Tweet from Artificial Analysis (@ArtificialAnlys)</a>: Gemini 1.5 intelligence upgrades and price reductions significantly strengthen Google&#39;s Quality & Price positioning in the AI market  Our independent evaluations of @Google&#39;s Gemini models val...</li><li><a href="https://github.com/copilot">Build software better, together</a>: GitHub is where people build software. More than 100 million people use GitHub to discover, fork, and contribute to over 420 million projects.</li><li><a href="https://x.com/smokeawayyy/status/1839036947400143073?s=46">Tweet from Smoke-away (@SmokeAwayyy)</a>: Ilya announced his departure the day after the GPT-4o presentation.  Mira announced her departure the day after the ChatGPT Voice release.</li><li><a href="https://youtu.be/oX7OduG1YmI">The Future Mark Zuckerberg Is Trying To Build</a>: The Huge Conversation with Mark Zuckerberg...I interviewed Meta CEO Mark Zuckerberg before Connect. There are not many people with more power over our future...</li><li><a href="https://stratechery.com/2024/an-interview-with-meta-cto-andrew-bosworth-about-orion-and-reality-labs/">An Interview with Meta CTO Andrew Bosworth About Orion and Reality Labs</a>: An Interview with Meta CTO Andrew Bosworth About Orion and Reality Labs</li><li><a href="https://about.fb.com/news/2024/09/introducing-orion-our-first-true-augmented-reality-glasses/">Introducing Orion, Our First True Augmented Reality Glasses | Meta</a>: Today we unveiled Orion, which we believe is the most advanced pair of AR glasses ever made.</li><li><a href="https://youtu.be/6pxmdmlJCG0?feature=shared">I interviewed the Creator of ChatGPT</a>: You know him as the CEO of OpenAI — but did you know that Sam Altman is an avid writer? As one of today’s most successful entrepreneurs, Sam champions the tr...</li><li><a href="https://x.com/mattturck/status/1839054212040171638">Tweet from Matt Turck (@mattturck)</a>: Destiny’s Child   —&gt; Beyoncé  One Direction     —&gt;  Harry Styles OpenAI                 —&gt;  Sama</li><li><a href="https://github.com/meta-llama/llama-stack/issues/6">RFC-0001 - Llama Stack · Issue #6 · meta-llama/llama-stack</a>: As part of the Llama 3.1 release, Meta is releasing an RFC for ‘Llama Stack’, a comprehensive set of interfaces / API for ML developers building on top of Llama foundation models. We are looking fo...
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/)** (1 messages): 

swyxio: new meetup in <@&1284244976024424630> led by <@656968717883670570> ! https://lu.ma/i8ulstlw
  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1288615241835479101)** (2 messages): 

> - `Engineering roles hiring`
> - `NVIDIA competition`
> - `LLM applications`
> - `Prizes and rewards`
> - `Developer resources` 


- **LlamaIndex is hiring engineers in San Francisco**: LlamaIndex is looking for energetic **ML/AI** enthusiasts to join their growing team as they expand their engineering roles, from full-stack to various positions. Interested candidates can find more details on [Twitter](https://twitter.com/llama_index/status/1839055997291344050).
- **Compete with NVIDIA for cash and hardware prizes**: There's a competition with **NVIDIA** offering over **$10,000** in prizes, including cash and hardware like an **NVIDIA® GeForce RTX™ 4080 SUPER GPU**. Developers can join the contest until **November 10th** to create innovative LLM applications, with more information available [here](https://developer.nvidia.com/llamaindex-developer-contest/join).
   - Participants are encouraged to build **RAG** applications across various domains, with [terms and conditions](https://developer.download.nvidia.com/licenses/nvidia-and-lla) available for review.



**Link mentioned**: <a href="https://t.co/rtMpetSyu1">NVIDIA and LlamaIndex Developer Contest</a>: Stand a chance to win cash prizes, a GeForce RTX GPU, and more.

  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1288594955769741325)** (47 messages🔥): 

> - `ReAct Agent Messages`
> - `VectorStoreIndex Confusion`
> - `LlamaTrace Project Issues`
> - `Score Evaluations`
> - `KnowledgeGraph RAG vs QueryFusion` 


- **Passing Messages to ReAct Agents**: A member inquired about conveying system and user messages to a ReAct agent, emphasizing the use of specified classes and formatting tools.
   - Responses indicated that the `ReActChatFormatter` class is essential for formatting chat history into a compatible input format.
- **Understanding VectorStoreIndex**: Confusion arose regarding the `VectorStoreIndex`, with users clarifying the relationship between indexes and underlying vector stores.
   - The group discussed how to access the `vector_store` property of an index without needing to initialize a new vector store.
- **Issues with LlamaTrace Project**: A user mentioned encountering an error after logging into their LlamaTrace project, which seemed to resolve itself after a while.
   - They suggested clearing cookies if others encountered the same issue.
- **Discrepancy in Score Evaluations**: A user found themselves with a higher correctness score compared to answer relevancy, sparking a discussion on score comparability.
   - It was suggested that the scores are not directly comparable as they depend on different aspects of the LLM's evaluation.
- **Using KnowledgeGraph RAG vs QueryFusion**: A member queried whether they were correctly using `QueryFusionRetriever` instead of `KnowledgeGraphRAGRetriever` for knowledge indexing.
   - The discussion focused on potential improvements and whether RAG retrievers would better suit their needs regarding querying multiple indexes.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://microsoft.github.io/autogen/docs/tutorial/code-executors/">Code Executors | AutoGen</a>: Open In Colab</li><li><a href="https://docs.llamaindex.ai/en/latest/api_reference/output_parsers/pydantic/#llama_index.core.output_parsers.PydanticOutputParser>)">Pydantic - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/output_parsing/llm_program/#initialize-with-pydantic-output-parser>)">LLM Pydantic Program - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/agent/nvidia_agent/#view-prompts>)">Function Calling NVIDIA Agent - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/api_reference/agent/react/#llama_index.core.agent.react.ReActChatFormatter>)">React - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/agent/nvidia_agent/#customizing-the-prompt>)">Function Calling NVIDIA Agent - LlamaIndex</a>: no description found
</li>
</ul>

</div>
  

---



### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1288947311778599004)** (3 messages): 

> - `Langtrace DSPy integration`
> - `DSPy project setup`
> - `Experiment tracking with Langtrace`
> - `Code samples for DSPy classification and summarization` 


- **Langtrace Now Supports DSPy Experiments**: Langtrace has introduced support for running DSPy experiments, featuring **automatic capturing of traces, checkpoints, costs**, and **eval score visualizations**.
   - This functionality enables users to create individual projects for each block of their pipeline, leading to enhanced experimentation and optimization.
- **Call for Code Samples from Experiments**: **Chiggly007** expressed interest in receiving notebooks or code samples from the experiments shared in the screenshots.
   - This highlights the community's desire to learn from practical examples of using DSPy for classification and summarization tasks.
- **DSPy Setup Instructions Shared**: **Kaykay0403** provided step-by-step setup instructions for integrating DSPy with Langtrace, pointing to relevant documentation.
   - Users are encouraged to follow the [DSPy installation guide](https://github.com/stanfordnlp/dspy?tab=readme-ov-file#1-installation) and the Langtrace setup to streamline their integration.
- **Simplified DSPy Integration Process**: Integrating Langtrace with DSPy is a straightforward process requiring just **two lines of code** to set up the SDK and run experiments.
   - Kaykay0403 emphasized their willingness to assist users with any issues or uncertainties during setup.



**Link mentioned**: <a href="https://docs.langtrace.ai/supported-integrations/llm-frameworks/dspy#dspy)">DSPy - Langtrace AI Docs</a>: no description found

  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1288584086859939871)** (38 messages🔥): 

> - `Integration of conversation history in models`
> - `STORM paper and GitHub resources`
> - `Creating agents with DSPy`
> - `Dspy.LM and Azure OpenAI APIs`
> - `Challenges with TypedChainOfThought` 


- **Exploring Conversation History Integration**: A member inquired about adding conversation history into models using the dspy.LM approach, suggesting the potential for summary optimization with previous turns.
   - Another suggested using a `multi_turn=True` flag for Predictors that would simplify the signature to handle conversation history effectively.
- **Resources for the STORM Paper**: A member requested the STORM paper link, which could be found on their [GitHub repository](https://github.com/stanford-oval/storm).
   - Another member provided a link to the [STORM paper on arXiv](https://arxiv.org/abs/2402.14207) discussing using LLMs for writing structured articles.
- **Creating Agents in DSPy**: A member shared a tutorial on building agents in DSPy, emphasizing the exploratory nature and highlighting the limitations of the current framework.
   - The member indicated that this tutorial aims to assist in understanding how to effectively create agent applications using DSPy.
- **Transitioning to dspy.LM with Azure APIs**: A member shared their experience transitioning from dspy.AzureOpenAI to dspy.LM, encountering API path construction issues with litellm.
   - However, they later reported resolving the issue attributed to the gpt-4o model, expressing satisfaction with the new LM's performance.
- **Issues with TypedChainOfThought**: A member noted that while migrating, they found parsing issues related to TypedChainOfThought and encountered JSON unwrapping challenges.
   - They planned to investigate this further but recommended the new LM for its smoother integration and improved functionality.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2402.14207">Assisting in Writing Wikipedia-like Articles From Scratch with Large Language Models</a>: We study how to apply large language models to write grounded and organized long-form articles from scratch, with comparable breadth and depth to Wikipedia pages. This underexplored problem poses new ...</li><li><a href="https://github.com/stanford-oval/storm">GitHub - stanford-oval/storm: An LLM-powered knowledge curation system that researches a topic and generates a full-length report with citations.</a>: An LLM-powered knowledge curation system that researches a topic and generates a full-length report with citations. - stanford-oval/storm</li><li><a href="https://learnbybuilding.ai/tutorials/dspy-agents-from-scratch">A first attempt at DSPy Agents from scratch</a>: This post is going to take a first pass at creating Agents from scratch, using DSPy. The goal here is education, and to explore how one might build agents from scratch in DSPy.
</li>
</ul>

</div>
  

---


### **DSPy ▷ #[examples](https://discord.com/channels/1161519468141355160/1161519685616025600/1288683231436210260)** (3 messages): 

> - `Number of Classes in Models`
> - `Subtle Distinctions in Class Signatures` 


- **Exploring Class Quantity in Models**: @okhattab inquired about the number of classes, sparking a discussion on the optimal amount needed for effective classification.
   - One participant noted they are currently working with **5 classes**, but suggested that **10 classes** might be worth exploring.
- **The Challenge of Subtle Class Distinctions**: A member emphasized the importance of class distinctions, pointing out that subtle differences can complicate the signature descriptions.
   - They indicated that highlighting these nuances is crucial for better model performance and clarity.


  

---



### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1288581828231561258)** (35 messages🔥): 

> - `Yamashi's Green Card Dilemma`
> - `Llama 3.2 Access Issues`
> - `Torchtune Dataset Error`
> - `MetaAI Login Restrictions`
> - `Visual Question Answering Datasets` 


- **Yamashi's Green Card Dilemma**: *“Spare some green card anyone?”* In a comedic turn, Yamashi jokingly pleads for a green card amidst discussions on legal and compliance hurdles.
   - He noted, *“Time to open a fake company in Delaware,”* showcasing a frustration with legal restrictions.
- **Llama 3.2 Access Issues**: Discussions circled around accessing **Llama 3.2**, with members noting EU restrictions prevent direct use of the model.
   - *“But I can't use llama 3.2 directly,”* lamented Yamashi, reflecting on the challenges faced.
- **Torchtune Dataset Error**: A member discussed encountering an error with **PackedDataset** regarding sequence length limits, referencing [GitHub issue #1689](https://github.com/pytorch/torchtune/issues/1689).
   - They suggested a straightforward fix and expressed willingness to submit a PR after further consideration of the testing requirements.
- **MetaAI Login Restrictions**: Members questioned access to **MetaAI**, highlighting that EU users face restrictions and cannot log in.
   - Yamashi observed, *“Ah checks out I am unable to login on meta ai,”* indicating connectivity limitations.
- **Visual Question Answering Datasets Galore**: A member shared excitement over available datasets for visual question answering, linking to a trending collection on [Hugging Face](https://huggingface.co/datasets?task_categories=task_categories:visual-question-answering&sort=trending).
   - They enthused about using these resources for potential applications in finetuning.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/datasets?task_categories=task_categories:visual-question-answering&sort=trending">Hugging Face – The AI community building the future.</a>: no description found</li><li><a href="https://github.com/pytorch/torchtune/issues/1689">PackedDataset cannot handle long sequence whose length is larger than 2*max_seq_len when using split_across_pack=True · Issue #1689 · pytorch/torchtune</a>: As the title, it reports a runtime error in the self._pad_pack() f = {&quot;tokens&quot;:list(range(121)),&quot;labels&quot;:list(range(121))} x = PackedDataset([f],max_seq_len=60,split_across_pack=Tr...</li><li><a href="https://github.com/mirceamironenco/torchtune/blob/fix-packedds-seqlen/torchtune/datasets/_packed.py#L144.">torchtune/torchtune/datasets/_packed.py at fix-packedds-seqlen · mirceamironenco/torchtune</a>: A Native-PyTorch Library for LLM Fine-tuning. Contribute to mirceamironenco/torchtune development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1288666337035423756)** (7 messages): 

> - `NeMo ASR optimizations`
> - `INT-FlashAttention for INT8`
> - `Quantization benefits`
> - `Triton kernel hosting` 


- **NeMo ASR Optimizations Enhance Performance**: According to a [tweet from Piotr Zelasko](https://x.com/PiotrZelasko/status/1838653087529209990), NeMo ASR now runs **2000-6000 times faster** than real-time on **NVIDIA GPUs** due to a series of optimizations for RNN-T, TDT, and CTC models.
   - These models now not only lead the **HF Open ASR Leaderboard** but are also described as fast and cost-effective, all implemented in **pure PyTorch**.
- **INT-FlashAttention Achieves Speed and Accuracy Improvements**: A discussion around [INT-FlashAttention](https://x.com/papers_anon/status/1839131401322639805?s=46) highlights its ability to enable **Full INT8 activations** and GEMM kernels, yielding **72% faster inference speed** and **82% smaller quantization error** compared to FP16 and FP8 formats.
   - Members note potential usefulness for **Ampere cards** lacking FP8, though a deeper analysis on its impact on PTQ and training is still needed.
- **Debate on Kernel Hosting for INT-FlashAttention**: There is consensus that hosting the new kernel for INT-FlashAttention should ideally take place in **torchao**, as it might be the more suitable platform.
   - A member suggested opening an issue to share insights about this, emphasizing the importance of refining these developments further.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/papers_anon/status/1839">Tweet from crystal (@crystal)</a>: oooooh, pretty~!</li><li><a href="https://x.com/PiotrZelasko/status/1838653087529209990">Tweet from Piotr Żelasko (@PiotrZelasko)</a>: Behold: NeMo ASR now runs easily 2000-6000 faster than realtime (RTFx) on @nvidia GPU.   We developed a series of optimizations to make RNN-T, TDT, and CTC models go brrrrrrr!🔥  In addition to toppin...</li><li><a href="https://x.com/papers_anon/status/1839131401322639805?s=46">Tweet from PapersAnon (@papers_anon)</a>: INT-FlashAttention: Enabling Flash Attention for INT8 Quantization  Full INT8 activations and GEMM kernels. Useful for Ampere cards that lack FP8. Achieves 72% faster inference speed and 82% smaller q...
</li>
</ul>

</div>
  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1288929344072454146)** (2 messages): 

> - `MOToMGP pass manager error`
> - `Mojo/MAX branded backgrounds` 


- **Team addressing MOToMGP pass manager issue**: The team is actively managing the **'failed to run the MOToMGP pass manager'** error and is open to feedback regarding **Max / Mojo** errors that could be improved.
   - Users are encouraged to share any grievance or suggestion related to this issue.
- **Interest in Mojo/MAX desktop backgrounds**: A poll was initiated to gauge interest in **Mojo / MAX branded desktop backgrounds** featuring adorable Mojo flames and MAX astronauts.
   - Participants were asked to emoji vote with a **yes** or **no** preference for these background designs.


  

---


### **Modular (Mojo 🔥) ▷ #[announcements](https://discord.com/channels/1087530497313357884/1098765954302873621/1288869066634690654)** (1 messages): 

> - `Verification Process`
> - `Onboarding Questions`
> - `Channel Posting Restrictions` 


- **Verification Bot Returns to Enhance Security**: The verification bot is back to help create a safe, spam-free community; members must verify by clicking "I'm human ✅" and authorizing email sharing.
   - A GIF is attached demonstrating how to successfully verify membership.
- **Unverified Members Face Posting Limitations**: Starting September 27th, unverified members can only post in designated channels, while still retaining read-only access to all others.
   - This move aims to encourage verification for full community access.
- **New Onboarding Questions Introduced**: Two new onboarding questions have been added: about the primary reason for joining and interest in early access to MAX's GPU product.
   - These questions are designed to enhance the overall experience for community members.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1288724503861395561)** (19 messages🔥): 

> - `Mojo compilation`
> - `Rust interoperability`
> - `Error handling improvements` 


- **Mojo compiles directly to machine code**: A member clarified that Mojo does not create **.pyc files** like Python for bytecode cache, as it compiles directly to machine code.
   - *“.pyc is bytecode cache, Mojo compiles directly to machine code.”*
- **Rust library for Mojo calls**: A member shared a project showcasing `mojo calls to rust`, containing a library of examples along with a detailed process for integration.
   - The project is hosted on GitHub, available [here](https://github.com/better-mojo/learn-mojo/tree/main/packages/mojo-ffi/mojo-call-rust).
- **Decompiling .exe back to Mojo**: Converting from **.exe** back to Mojo code was discussed, with members suggesting it might be possible with effort, but is more likely to revert to **C**.
   - *“With great pain, potentially yes. It’s far more likely they convert to C.”*
- **Error messages need improvement**: A member expressed a desire for better error messages in Mojo, emphasizing clarity on error sources and module boundaries.
   - Another member agreed, suggesting that errors should highlight where issues originate to enhance legibility.
- **Lack of nominal sum type in Variant**: Discussion around the `__init__` method of `Variant` indicated issues related to the lack of nominal sum type leading to less informative errors.
   - *“I blame the lack of nominal sum type.”*



**Link mentioned**: <a href="https://github.com/better-mojo/learn-mojo/tree/main/packages/mojo-ffi/mojo-call-rust">learn-mojo/packages/mojo-ffi/mojo-call-rust at main · better-mojo/learn-mojo</a>: learn mojo. Contribute to better-mojo/learn-mojo development by creating an account on GitHub.

  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1288887854147178599)** (3 messages): 

> - `MAX API usage`
> - `User feedback on MAX API` 


- **Request for MAX API User Feedback**: A member solicited **feedback** from users who have recently engaged with the **MAX API** to load and execute models.
   - They expressed interest in knowing about any **frustrations** or improvements users would like to see, encouraging open communication in the thread.
- **Invitation to Share Thoughts on MAX API**: The member reiterated their desire for users to share **any and all thoughts** regarding the MAX API experience.
   - They included a lighthearted emoji to foster a friendly environment for sharing feedback.


  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1288887526488019116)** (21 messages🔥): 

> - `LLM Response Behavior`
> - `Source Document Retrieval`
> - `Debugging Tools Usage` 


- **LLM responds with 'no information found'**: A member explained that when invoking the chain with a question, the **LLM** sometimes responds with *'I'm sorry, but I don't know'* but still retrieves source documents.
   - They expressed a desire for the retrieval of documents to be conditional on the model actually having relevant information.
- **Source Documents Retrieved Unnecessarily**: The same member noted that the **source documents** should not be returned if the LLM indicates there's no useful information, leading to confusion.
   - They clarified that while they find most responses satisfactory, there are scenarios where they do not want source documents if the LLM response is negative.
- **Concerns About Debugging Tools**: Another member inquired if the original poster used any **debugging** or profiling tools like Langsmith, which the original poster declined due to privacy concerns.
   - Alternatives, such as hosting **Langfuse**, were suggested to address the need for monitoring without compromising data privacy.
- **Request for Code Examples**: A participant requested to see examples of the code to better understand the situation and the challenges faced by the original poster.
   - The original poster agreed to provide examples the following day, as it was late at night.


  

---



### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1288586714339741829)** (14 messages🔥): 

> - `Generative AI Bubble`
> - `PhD student engagement`
> - `Instructional header formatting`
> - `Token speculation`
> - `Community welcoming atmosphere` 


- **Generative AI Bubble facing scrutiny**: A member expressed concerns that the **Generative AI** industry, particularly **ChatGPT**, is on the verge of collapse, influenced by recent departures like **Mira Murati**.
   - They cited an insightful newsletter claiming the generative AI boom is **unsustainable**, and may harm big tech and public perception.
- **PhD students staying updated with Cohere**: One new member stated they joined the community to keep up with events while nearing the end of their PhD.
   - This highlights Cohere as a valuable resource for academics interested in AI discussions.
- **Formatting instructional headers discussion**: A member sought recommendations on **formatting instructional headers** and questioned how **RAG inclusions** would present in LLM submissions.
   - This shows a keen interest in improving instructional design within the community.
- **Community welcomes newcomers**: Multiple members welcomed newcomers to the community and encouraged them to ask questions about **Cohere**.
   - This illustrates a supportive environment focused on learning and engagement.
- **Tinfoil hat speculation discouraged**: A member advocated for sticking to facts amidst speculation about the AI industry's future, emphasizing **Cohere's grounding** in reality.
   - This reflects a desire for thoughtful discussion over sensationalist theories.



**Link mentioned**: <a href="https://www.wheresyoured.at/subprimeai/">The Subprime AI Crisis</a>: None of what I write in this newsletter is about sowing doubt or &quot;hating,&quot; but a sober evaluation of where we are today and where we may end up on the current path. I believe that the artifi...

  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1288822057575911435)** (7 messages): 

> - `Rerank fine-tuning`
> - `Avg Hard Negatives per Query`
> - `Model performance and data quality` 


- **Question on Avg Hard Negatives Computation**: A user inquired about how the **'Avg Hard Negatives per Query'** is calculated, noting their dataset contains less than **10%** hard negatives.
   - *Cohere clarified that they do not add negatives behind the scenes* and suggested verifying the data quality.
- **Model's Performance Post-Training**: Following the training process, a user reported that the model performed only slightly better than the **default English v3 reranker**.
   - They speculated that the **quality of the data** might be a contributing factor to this underwhelming performance.
- **Data Characteristics Discussed**: In dialogue about the dataset, it was mentioned that it contains many **financial tables**, albeit with numerous numerical values removed.
   - This shift in data presentation might impact the model’s ability to learn effectively from training.


  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1288699898971029574)** (5 messages): 

> - `Arbitrary View Mergeability Proof`
> - `Pairwise vs Global Mergeability`
> - `View Merging Optimization`
> - `View Merging and Symbolic Reduction` 


- **Arbitrary View Mergeability Proof Discussed**: A proof of **arbitrary view mergeability** without considering masks or reshapes has been shared on GitHub, detailing important aspects of view management in **Tinygrad**.
   - This [proof can be found here](https://github.com/pschilliOrange/Tinygrad-view-merging-proof/blob/8672f35c1147798c8e9a78bfab28b9ff79bf45e6/Proof%20for%20when%20a%20new%20view%20must%20be%20appended.pdf) which accompanies an overview of the project.
- **Distinction Between Pairwise and Global Mergeability**: Concerns were raised that the current proof addresses **pairwise mergeability** of two views, but does not cover **global mergeability** of multiple composed views or equivalence of **shapeTrackers**.
   - The importance of collapsing all views when appending a new one was deemed potentially **too expensive** for practical implementation.
- **Working Offsets and Masks into the Proof**: There is an intention to incorporate **offsets** and **masks** into the existing mergeability proof to enhance its comprehensiveness.
   - The plan is to eventually adapt this work into **Lean**, a proof assistant for mathematical formalization.
- **View Merging as Symbolic Reduction**: A suggestion was made that **view merging** could be equivalent to **symbolic reduction**, opening up new avenues for optimization.
   - This concept could allow for rewriting views with **pattern rewrites**, as hinted in a [unit test example](https://github.com/tinygrad/tinygrad/blob/master/test/unit/test_simplify_valid_idx.py) within the Tinygrad repository.



**Link mentioned**: <a href="https://github.com/pschilliOrange/Tinygrad-view-merging-proof/blob/8672f35c1147798c8e9a78bfab28b9ff79bf45e6/Proof%20for%20when%20a%20new%20view%20must%20be%20appended.pdf">Tinygrad-view-merging-proof/Proof for when a new view must be appended.pdf at 8672f35c1147798c8e9a78bfab28b9ff79bf45e6 · pschilliOrange/Tinygrad-view-merging-proof</a>: Contribute to pschilliOrange/Tinygrad-view-merging-proof development by creating an account on GitHub.

  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1288613336568692766)** (12 messages🔥): 

> - `Tinygrad performance issues`
> - `Metal compatibility errors`
> - `Comparison between Tinygrad and PyTorch`
> - `Tinygrad customization benefits` 


- **Tinygrad Training Lagging Performance**: A user reported that training with Tinygrad is extremely slow and received a **4090 GPU** to improve performance, but the output quality was poor due to a bug in their sampling code.
   - They noted the issue was not with training speed but rather the incorrect implementation in the sampling logic.
- **Metal Error for Double Precision**: A user encountered a Metal error stating that '**double**' is unsupported, which was related to NumPy using double precision by default.
   - Converting tensors to **float32** solved the issue, but they then faced another error indicating a possible buffer resource location problem.
- **Tinygrad's Limitations on Metal**: A participant highlighted that the fused kernel in Tinygrad is hitting **Metal's buffer limit**, and a workaround would be attempted in an upcoming update.
   - This indicates potential memory issues when working with Metal as a backend in Tinygrad.
- **Comparing Tinygrad to PyTorch and CUDA**: Discussion arose regarding Tinygrad as a fast alternative to **PyTorch**, with questions on how it stacks against coding directly in **CUDA**.
   - Tinygrad is identified as a compiler using CUDA as a backend, but PyTorch benefits from extensively optimized handcrafted CUDA kernels.
- **Customization and Optimization in Tinygrad**: It was noted that Tinygrad allows for broader optimization possibilities through custom kernel generation, in contrast to the fixed kernels in PyTorch.
   - A key feature of Tinygrad is its ability to explore various kernel optimizations, which can lead to improved performance.



**Link mentioned**: <a href="https://github.com/mesozoic-egg/tinygrad-notes/tree/main">GitHub - mesozoic-egg/tinygrad-notes: Tutorials on tinygrad</a>: Tutorials on tinygrad. Contribute to mesozoic-egg/tinygrad-notes development by creating an account on GitHub.

  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1288638152457584702)** (9 messages🔥): 

> - `LLaMA 3.2 Vision`
> - `OpenAI's Function Calling API`
> - `Voice Cloning Technology` 


- **LLaMA 3.2 Vision 90B excels in Image Captioning**: Members noted that **LLaMA 3.2 Vision 90B** appears to be highly capable for **image captioning**, with even the **11B** version attracting attention.
   - One member humorously suggested using the unlimited model to caption the entire **LAION dataset**, indicating its potential.
- **OpenAI's Function Calling API Inquiry**: A member inquired about the internal workings of **OpenAI's function calling API**, speculating whether it runs a fine-tuned model or uses checks on outputs from other models.
   - This question reflects ongoing curiosity about the intricacies of API design and performance enhancements.
- **Free Access to LLaMA 3.2 Vision**: TogetherCompute has announced a partnership with **AI at Meta** to provide **LLaMA 3.2 11B Vision for free**, allowing developers to experiment with multimodal AI.
   - They offer a free model endpoint at [this link](https://api.together.ai/playground/chat/meta-llama/Llama-Vision-Free), along with paid options for enhanced performance.
- **Excitement Over Voice Cloning Demo**: A member excitedly shared a YouTube video of them conversing with a voice clone of themselves, expressing amusement at the technology.
   - The related [video](https://youtu.be/X8KhTlKoagg) highlights ongoing advancements in voice cloning technology and its fun applications.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/togethercompute/status/1839071026728333778">Tweet from Together AI (@togethercompute)</a>: 🚀 We&#39;ve partnered with @AIatMeta to offer Llama 3.2 11B Vision for FREE so developers can experiment with open-source multimodal AI at no cost.  In addition to our free credits, for a limited tim...</li><li><a href="https://youtu.be/X8KhTlKoagg">Me talking to BUD-E which is using my own voice!  :D</a>: https://discord.gg/pCPJJXP7Qx
</li>
</ul>

</div>
  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1288576993222459412)** (6 messages): 

> - `MaskBit`
> - `MonoFormer`
> - `HuggingFace papers`
> - `VQGAN enhancements`
> - `Multimodality transformer models` 


- **MaskBit revolutionizes image generation**: This study introduces **MaskBit**, an embedding-free image generation approach achieved via **bit tokens**, significantly improving on the classic VQGAN transformer model for class-conditional image generation.
   - Their new model attains a remarkable **FID of 1.52** on the ImageNet benchmark with just **305M parameters**, showing that embedding-free methods can outperform existing techniques.
- **MonoFormer unifies generation processes**: **MonoFormer** proposes a single transformer architecture that adeptly handles both **autoregression** and **diffusion**, eliminating the need for separate models for text and image generation.
   - By leveraging shared training methods, it maintains competitive image generation performance while effectively producing text, and further details can be found at their [project page](https://monoformer.github.io/).
- **HuggingFace showcases recent developments**: Two **mildly interesting papers** were shared from HuggingFace, with potential implications for the future of multimodal models and image synthesis.
   - While only skimmed for now, the initial findings indicate promising advancements in the efficiency and performance of transformer models.


  

---



### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1288643369005875304)** (8 messages🔥): 

> - `Quiz 3 Questions`
> - `RAG Model Capabilities`
> - `Social Alignment in LLM Agents`
> - `CCMP_AI Project`
> - `Course Registration Confirmation` 


- **Quiz 3 Questions Spark Confusion**: A member expressed confusion over a **Quiz 3 question**, noting it wasn't mentioned in the presenter's explanation of constrained and unconstrained flows.
   - Another member pointed out that the information was indeed in the slides, confirming the quiz content.
- **RAG Model Struggles with Multimodal Data**: Concerns were raised about the **RAG capabilities** of the latest models, particularly how they perform with multimodal data like text, tables, and images.
   - An impressive reference was made to **Claude 3**, which performed well in explaining flow diagrams.
- **Agentic RAG Projects Take Shape**: A member shared that their **ccmp_ai project** is an unconstrained RAG model, providing new terminology that another member found useful.
   - They referred to it as an **agentic RAG** with dynamic problem domain expansion.
- **Inquiries About LLM Agent Alignment**: A member inquired whether upcoming lectures would address the **social alignment** of LLM agents, noting that past lectures seemed focused on technical alignment.
   - They pointed out that the last two lectures only cover aspects of **AI safety**.
- **Uncertainties in Course Registration**: A member sought clarification about course registration via a Google form, questioning if they would receive confirmation after signing up.
   - They expressed hope for a quick resolution to understand how access to the course works.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-readings-discussion](https://discord.com/channels/1280234300012494859/1282735578886181036/1288647904462110780)** (1 messages): 

> - `Healthcare multi-agent systems`
> - `AgentClinic Benchmark` 


- **Summary of Healthcare Multi-Agent Systems Research**: A recent research titled [AgentClinic: A Multimodal Agent Benchmark](https://open.substack.com/pub/yanpan0508/p/agentclinic-a-multimodal-agent-benchmark?r=ad7en) focuses on healthcare multi-agent systems, discussing key methodologies and findings.
   - This study highlights the collaborative potential of multi-agent systems in the healthcare domain, pushing boundaries in AGI applications.
- **Yvaine’s Substack Launch**: Yvett's Substack, titled 'Embracing AGI', was launched two months ago, providing insights into the evolving field of artificial intelligence.
   - Yvaine emphasizes the importance of AGI in healthcare contexts within her blog, aiming to engage a community interested in multi-agent system advancements.



**Link mentioned**: <a href="https://open.substack.com/pub/yanpan0508/p/agentclinic-a-multimodal-agent-benchmark?r=ad7en&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true">AgentClinic: a multimodal agent benchmark</a>: Paper Reading: Evaluation of AI in simulated clinical environments

  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1288579347724107870)** (6 messages): 

> - `Llama 3.2 experiments`
> - `Open Interpreter issues`
> - `Tech Week SF meet-up` 


- **Anticipated Experiments with Llama 3.2**: A user inquired about the **experiments** and **tests** planned for Llama 3.2, indicating ongoing interest in its capabilities.
   - The community seems eager to explore new potential applications and functionalities of the model.
- **Open Interpreter Fails to Count Files**: A member reported that when using the **3b** model with Open Interpreter to count files on their desktop, it **failed** to execute the task.
   - This failure raised concerns about the reliability of the model in performing such tasks.
- **Tech Week SF High-Five Opportunity**: One user expressed excitement about attending **Tech Week** in San Francisco and suggested meeting up to high-five.
   - This highlights the community's enthusiasm for networking and connecting during tech events.


  

---


### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1288631996456632342)** (3 messages): 

> - `Llama 3.2 Testing`
> - `NERD Task Challenges` 


- **Llama 3.2 fails to impress**: After testing **Llama 3.2** 90b, a member expressed disappointment, stating it does not compare favorably to **Llama 3.1** 70b.
   - They referenced a [YouTube video](https://www.youtube.com/watch?v=_MQVHyEeER4) titled 'Llama-3.2 (1B, 3B, 11B, 90B) : The WORST New LLMs EVER!?' that details their findings.
- **Challenges with NERD Task**: A member described a **NERD** task focused on linking text to **wiki** entries for individuals mentioned in news articles.
   - This task is seen as complex due to the intricacies involved in extracting and matching the relevant information.



**Link mentioned**: <a href="https://www.youtube.com/watch?v=_MQVHyEeER4">Llama-3.2 (1B, 3B, 11B, 90B) : The WORST New LLMs EVER!? (Fully Tested &amp; Beats &quot;Nothing&quot;)</a>: Join this channel to get access to perks:https://www.youtube.com/@AICodeKing/joinIn this video, I&#39;ll be fully testing all the new Llama-3.2 Vision &amp; Small La...

  

---



### **MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/1288850703275659335)** (3 messages): 

> - `Alternatives to partition_pdf`
> - `Channel etiquette` 


- **Seeking Alternatives to partition_pdf**: A member asked if anyone could suggest an alternative to **unstructured 'partition_pdf'** for extracting images and tables from PDFs.
   - They are looking for a more effective tool to handle this task.
- **Reminder on Channel Etiquette**: Another member reminded that posting the same question in multiple channels will be considered **spam**.
   - They took action by deleting duplicate questions in other channels to maintain order.


  

---



### **Alignment Lab AI ▷ #[general](https://discord.com/channels/1087862276448595968/1095458248712265841/)** (1 messages): 

rusch: bruh how is this not promotion
  

---



### **Mozilla AI ▷ #[announcements](https://discord.com/channels/1089876418936180786/1089876419926032396/1288648747076685906)** (1 messages): 

> - `Mozilla AI`
> - `Locally-run AI models`
> - `Large Language Models (LLMs)`
> - `Continue tool`
> - `Nature article` 


- **Mozilla AI Shines in Nature Article**: Mozilla AI and key projects were featured in Nature's article, *“Forget ChatGPT: Why researchers now run small AIs on their laptops.”*
   - Insights were shared by Mozilla's head of open-source AI, highlighting the **rise of locally-run AI models** that empower users.
- **LLMs for All Systems**: The article mentioned a project related to enabling **Large Language Models (LLMs)** to run on various systems, showcasing its versatility.
   - This was emphasized as a significant development in making powerful AI accessible across different platforms.
- **Continue Tool Gains Recognition**: Continue, led by a member in a recent organized talk, was highlighted as a valuable tool for **AI-assisted coding**.
   - This endorsement showcases its effectiveness and relevance within the AI community.
- **Direct Link to Full Article**: The summary provided a direct link to the [full article here](https://discord.com/channels/1089876418936180786/1288648264069025883/1288648264069025883) for readers interested in more details.
   - This inclusion helps direct community members to the original source of the insights discussed.


  

---



### **Gorilla LLM (Berkeley Function Calling) ▷ #[discussion](https://discord.com/channels/1111172801899012102/1111353033352294440/1288953002731765780)** (1 messages): 

> - `Function Calling Evaluation`
> - `Custom Evaluation Dataset`
> - `LLM Integration`
> - `Berkeley Function-Calling Leaderboard` 


- **Clarifying Function Calling Evaluation in Codebase**: A user expressed confusion about the function calling evaluation in the codebase, questioning whether their own evaluation dataset could be provided alongside a custom API/LLM.
   - They highlighted the lack of clarity on integrating their dataset of **<prompt>, <llm_response>, <ideal response>** for error breakdown.
- **Desire for Error Breakdown from Custom Datasets**: The user is looking for a package that can analyze their dataset and provide insights into errors similar to those in the [BFCL metrics](https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html#metrics).
   - *



**Link mentioned**: <a href="https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html#metrics">Berkeley Function Calling Leaderboard</a>: no description found

  

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
