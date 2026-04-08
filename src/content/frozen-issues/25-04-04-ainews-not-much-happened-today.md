---
id: e7f662d9-c99c-4670-9247-561b3dd6709f
title: not much happened today
date: '2025-04-05T01:50:06.395334Z'
original_slug: ainews-not-much-happened-today-7847
description: >-
  **OpenAI** announced that **o3** and **o4-mini** models will be released soon,
  with **GPT-5** expected in a few months, delayed for quality improvements and
  capacity planning. **DeepSeek** introduced **Self-Principled Critique Tuning
  (SPCT)** to enhance inference-time scalability for generalist reward models.
  **Anthropic's Sonnet 3.7** remains a top coding model. **Google's Gemma 3** is
  available on KerasHub, and **Qwen 2.5 VL** powers a new Apache 2.0 licensed
  OCR model. **Gemini 2.5 Pro** entered public preview with increased rate
  limits and pricing announced, becoming a preferred model for many tasks except
  image generation. Meta's architectural advantage and the **FrontierMath
  benchmark** challenge AI's long-form reasoning and worldview development.
  Research reveals LLMs focus attention on the first token as an "attention
  sink," preserving representation diversity, demonstrated in **Gemma 7B** and
  **LLaMa 3.1** models. **MegaScale-Infer** offers efficient serving of
  large-scale Mixture-of-Experts models with up to **1.90x higher per-GPU
  throughput**.
companies:
  - openai
  - deepseek
  - anthropic
  - google
  - meta-ai-fair
models:
  - o3
  - o4-mini
  - gpt-5
  - sonnet-3.7
  - gemma-3
  - qwen-2.5-vl
  - gemini-2.5-pro
  - gemma-7b
  - llama-3-1-405b
topics:
  - inference-scaling
  - reward-modeling
  - coding-models
  - ocr
  - model-preview
  - rate-limiting
  - model-pricing
  - architectural-advantage
  - benchmarking
  - long-form-reasoning
  - attention-mechanisms
  - mixture-of-experts
  - gpu-throughput
people:
  - sama
  - akhaliq
  - nearcyan
  - fchollet
  - reach_vb
  - philschmid
  - teortaxestex
  - epochairesearch
  - omarsar0
---


<!-- buttondown-editor-mode: plaintext -->**Apply for AIEWF talk slots!**

> AI News for 4/3/2025-4/4/2025. We checked 7 subreddits, [**433** Twitters](https://twitter.com/i/lists/1585430245762441216) and **30** Discords (**230** channels, and **7491** messages) for you. Estimated reading time saved (at 200wpm): **629 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

It's been a quiet week, so why not fill out [the AI Engineer World's Fair Call For Speakers](https://sessionize.com/ai-engineer-worlds-fair-2025)?

Tracks across:

- **AI Architects**
- **/r/localLlama**
- **Model Context Protocol (MCP)**
- **GraphRAG**
- **AI in Action**
- **Evals**
- **Agent Reliability**
- **Retrieval, Search, and Recommendation Systems**
- **Security**
- **Infrastructure**
- **Generative Media**
- **AI Design & Novel AI UX**
- **AI Product Management**
- **Autonomy, Robotics, and Embodied Agents**
- **Computer-Using Agents (CUA)**
- **SWE Agents**
- **Vibe Coding**
- **Voice**
- **Sales/Support Agents**
- **The Great AI Debates**
- **Anything Else**


[Apply here](https://sessionize.com/ai-engineer-worlds-fair-2025)!

---


{% if medium == 'web' %}


**Table of Contents**

[TOC] 

{% else %}

The **Table of Contents** and **Channel Summaries** have been moved to the web version of this email: [{{ email.subject }}]({{ email_url }})!

{% endif %}


---

# AI Twitter Recap

**Model Releases and Announcements**

- **OpenAI's plans for model releases have shifted**: [@sama](https://twitter.com/sama/status/1908167621624856998) announced that **o3 and o4-mini will be released in a couple of weeks, followed by GPT-5 in a few months**. The delay is attributed to making GPT-5 much better and challenges in smoothly integrating everything, along with ensuring sufficient capacity for expected demand.
- **DeepSeek's Self-Principled Critique Tuning (SPCT) improves inference-time scalability for generalist reward modeling**: [@iScienceLuvr](https://twitter.com/_akhaliq/status/1908167564057849903) reports that **DeepSeek's** new method, **SPCT**, enhances the quality and scalability of Generalist Reward Models (GRMs), outperforming existing methods and models in various RM benchmarks.
- [@nearcyan](https://twitter.com/nearcyan/status/1908041346612604982) asserts that **Anthropic's Sonnet 3.7 remains the best coding model**.
- **Google's Gemma 3** can be tried in [KerasHub](https://twitter.com/fchollet/status/1908176807645663615).
- **Qwen 2.5 VL** powers a new **Apache 2.0 licensed OCR model**: [@reach_vb](https://twitter.com/reach_vb/status/1908232634943365478).

**Gemini 2.5 Pro**

- **Gemini 2.5 Pro is in public preview for scaled paid usage and higher rate limits**: [@_philschmid](https://twitter.com/_philschmid/status/1908177619721556007) announced the move to public preview. Google is moving Gemini 2.5 Pro to Preview, offering developers increased rate limits for testing production-ready apps, now available in Google AI Studio, as noted by [@Google](https://twitter.com/Google/status/1908177834209611865).
- **Gemini 2.5 Pro is becoming a daily driver for some**: [@fchollet](https://twitter.com/fchollet/status/1908310903571046431) notes it is probably the best model for most tasks except image generation, where it is still good.
- **Pricing is out for Gemini 2.5 Pro**: [@scaling01](https://twitter.com/scaling01/status/1908177213473587330) shares the cost per million tokens for context &gt;200k: Input at $1.25 (2.50) and Output at $10 (15.00).

**AI Model Capabilities and Benchmarks**

- **Meta's architectural advantage**: [@teortaxesTex](https://twitter.com/teortaxesTex/status/1907997211289436467) notes OpenAI's willingness to flex their architectural advantage.
- **FrontierMath benchmark challenges AI**: [@EpochAIResearch](https://twitter.com/EpochAIResearch/status/1908199401773813915) describes how their **FrontierMath benchmark** challenges AI to perform long-form reasoning and develop a coherent worldview, crucial steps for broader reasoning capabilities and scientific thinking.
- **DeepSeek's inference scaling paper shows that Gemma-2 27b is enough to match R1**: [@teortaxesTex](https://twitter.com/teortaxesTex/status/1907987423377666538). 
- **A new paper explains why LLMs obsessively focus attention on the first token** known as an attention sink: [@omarsar0](https://twitter.com/omarsar0/status/1908187563422261411) reports that sinks act as no-ops that reduce token interaction and preserve representation diversity across layers. Perturbation tests in **Gemma 7B** show `<s>` significantly slows the spread of changes, and in **LLaMa 3.1** models, over 80% of attention heads show strong sink behavior in the 405B variant.
- **MegaScale-Infer** is presented as an efficient and cost-effective system for serving large-scale Mixture-of-Experts (MoE) models, achieving up to **1.90x higher per-GPU throughput** than state-of-the-art solutions: [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1908091264714850707).
- **Discrete diffusion models are experiencing a resurgence**: [@cloneofsimo](https://twitter.com/cloneofsimo/status/1908148670098538645) highlights that discrete diffusion is winning over AR recently, with LLaDA-8B, Dream-7B, and UniDisc.
- **GPT-ImgEval is introduced as a comprehensive benchmark for diagnosing GPT4o in image generation**: [@_akhaliq](https://twitter.com/_akhaliq/status/1908168924186697965).

**AI Applications and Tools**

- **Microsoft is rapidly advancing GitHub Copilot**: [@LiorOnAI](https://twitter.com/LiorOnAI/status/1908313299466502166) shares that Agent mode and MCP support are rolling out to all VS Code users.
- **PyTorch** has released a tool to visualize matrices: [@LiorOnAI](https://twitter.com/LiorOnAI/status/1908233269998403980) announced its release, emphasizing that matrix multiplications (matmuls) are the building blocks of today’s models.
- **Elicit** has added approximately 10 million more full-text papers, enhancing the comprehensiveness of its reports: [@elicitorg](https://twitter.com/elicitorg/status/1908157705912775093).
- **Perplexity AI** has shipped a number of features, including fact-checking of any part of the answer with sources: [@AravSrinivas](https://twitter.com/AravSrinivas/status/1908233448604787185).

**Langchain and Graph Updates**

- **AppFolio’s copilot, Realm-X, powered by LangGraph and LangSmith, saves property managers over 10 hours per week** [@LangChainAI](https://twitter.com/LangChainAI/status/1908240852541202623) .
- **LangGraph Python now supports Generative UI**: [@LangChainAI](https://twitter.com/LangChainAI/status/1908186508969783495).
- **Langchain and Tavily AI now have a ReAct Agent Tutorial Series**: [@LangChainAI](https://twitter.com/LangChainAI/status/1908203029385343414) reports on a step-by-step guide for building production AI agents with LangGraph.

**Other**

- [@jd_pressman](https://twitter.com/jd_pressman/status/1908055615072776606) expresses that they're **tempted to write down their 5 year timeline** in the hopes it breaks somebody out of mode collapse.
- **Karpathy** is advocating for moving AI predictions from blog posts, podcasts, and tweets to betting markets: [@karpathy](https://twitter.com/karpathy/status/1908109168952676855).
- **Hugging Face** had 1,000,000 pageviews on research papers in March [@ClementDelangue](https://twitter.com/ClementDelangue/status/1908176702502527046), and it is becoming the best place to find, promote &amp; discuss research in AI!
- **Stanford** welcomes [@YejinChoinka](https://twitter.com/YejinChoinka) as a new faculty member in Computer Science: [@stanfordnlp](https://twitter.com/stanfordnlp/status/1908178010127397005).

**Humor and Memes**

- **Edo period cat meme**: [@hardmaru](https://twitter.com/hardmaru/status/1908022570789773516)


---

# AI Reddit Recap

## /r/LocalLlama Recap

### Theme 1. "Advancements in Generalist Reward Models Unveiled"

- **[New paper from DeepSeek w/ model coming soon: Inference-Time Scaling for Generalist Reward Modeling](https://arxiv.org/abs/2504.02495)** ([Score: 257, Comments: 40](https://www.reddit.com/r/LocalLLaMA/comments/1jre3kp/new_paper_from_deepseek_w_model_coming_soon/)): **DeepSeek has released a new paper titled 'Inference-Time Scaling for Generalist Reward Modeling'. The paper introduces a method called **Self-Principled Critique Tuning (SPCT)** to improve reward modeling for large language models by scaling compute at inference time. Their **27B parameter** DeepSeek-GRM model with parallel sampling can match or exceed the performance of much larger reward models up to **671B parameters**. The models will be released and open-sourced.** This research offers a promising path for enthusiasts running LLMs locally, as it allows achieving higher-quality evaluations without needing massive models. The availability of open-source models could provide local LLM users access to high-quality evaluation tools.


  - Hankdabits: Expresses enthusiasm that DeepSeek's **27B parameter** model can match or exceed much larger models, saying *"Yes please"*.
  - Iory1998: Notes that DeepSeek usually releases models two weeks after a paper, so *"it's very soon baby!"*, and suggests this may impact the release of Llama-4.
  - JLeonsarmiento: Remarks that while others are distracted, *"the Chinese are destroying USA AI business model and pushing boundaries."*


### Theme 2. "Building High-Performance GPU Servers on a Budget"

- **[Howto: Building a GPU Server with 8xRTX 4090s for local inference](https://i.redd.it/vg99momf6qse1.png)** ([Score: 550, Comments: 161](https://www.reddit.com/r/LocalLLaMA/comments/1jr0oy2/howto_building_a_gpu_server_with_8xrtx_4090s_for/)): **Marco Mascorro built a GPU server with eight NVIDIA RTX 4090 graphics cards for local inference and provided a detailed guide on the parts used and assembly instructions. The build offers a cost-effective local inference solution compared to more expensive GPUs like A100s or H100s and is expected to be compatible with future RTX 5090s. The full guide is available here: [https://a16z.com/building-an-efficient-gpu-server-with-nvidia-geforce-rtx-4090s-5090s/](https://a16z.com/building-an-efficient-gpu-server-with-nvidia-geforce-rtx-4090s-5090s/). An image shows the server setup with eight GPUs organized in a chassis for high-performance computing applications.** The author is enthusiastic about open-source models and local inference solutions, hoping the guide will be helpful for those without the budget for expensive GPUs like A100s or H100s. They welcome comments and feedback and are eager to answer any questions.

  - `segmond` notes that the budget should be specified, implying that cost is an important consideration.
  - `Educational_Rent1059` suggests that **2x RTX 6000 ADA PRO** GPUs may provide better ROI, offering 192GB VRAM and being more cost-effective and power-efficient.
  - `Puzzleheaded_Smoke77` comments on the high expense by stating, *"I could probably pay my mortgage for a year with the amount of money sitting in that case ...."*

## Other AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding


### Theme 1. "Advancements in Long Context AI Models"

- **[chatgpt-4o-latest-0326 is now better than Claude Sonnet 3.7](https://www.reddit.com/r/ClaudeAI/comments/1jr8t65/chatgpt4olatest0326_is_now_better_than_claude/)** ([Score: 262, Comments: 121](https://www.reddit.com/r/ClaudeAI/comments/1jr8t65/chatgpt4olatest0326_is_now_better_than_claude/)): **The new **GPT-4o-latest-0326** model is significantly better than the previous GPT-4o model. According to **LMSys** rankings, it's now **#2 overall** and **#1 for coding**. The model can be added in Cursor as **"chatgpt-4o-latest"**. The poster used this model on Cursor for working with **1-5 medium-length Python scripts** in a **synthetic data generation pipeline**. The model handles long context well and is fast. The poster is sharing this experience in a Claude subreddit to get opinions from Claude power users.** The poster finds the new GPT-4o model **dramatically better** than the previous version at coding and everything else. *It doesn't overcomplicate things* (unlike Sonnet 3.7), often providing the simplest and most obvious solutions that work. It *formats replies beautifully*, making them super easy to read. It follows instructions very well. The poster has switched to it and hasn't switched back since. The poster encourages others to try the new model and share their experiences.

  - One user mentions they've shifted to **gemini 2.5 pro**, which is free, has the highest context size, and they don't see a reason to use anything else right now.
  - Another user expresses confusion over the various models and their capabilities, asking how **GPT-4.5**, **o3-mini-high**, **Claude**, and others like **Deepseek** compare for coding tasks.
  - A user notes that while **Claude** was their favorite, it has now been outperformed in nearly every way, even in coding.


### Theme 2. "Unlocking AI Innovations: Art, Animation, and Pricing"

- **[How to guide: unlock next-level art with ChatGPT with a novel prompt method! (Perfect for concept art, photorealism, mockups, infographics, and more.)](https://www.reddit.com/r/ChatGPT/comments/1jr0qei/how_to_guide_unlock_nextlevel_art_with_chatgpt/)** ([Score: 482, Comments: 41](https://www.reddit.com/r/ChatGPT/comments/1jr0qei/how_to_guide_unlock_nextlevel_art_with_chatgpt/)): **The Reddit user introduces a novel technique to enhance image generation using ChatGPT, particularly effective for concept art, photorealism, mockups, and infographics. The method involves first prompting ChatGPT to create a detailed visual description of the desired image, sometimes extending to thousands of words. This detailed context helps the model 'think through' the scene, resulting in higher quality and more coherent images, often surpassing the capabilities of the **Images v2** model. The user provides step-by-step instructions: first, ask ChatGPT to 'Describe in extremely vivid details exactly what would be seen in an image [or photo] of [insert your idea],' including extensive details for better context; then, switch back to the image generation model and prompt it to 'Generate the photo following your description to the exact detail.' They share examples using scenes from *Lord of the Rings*, such as generating images of Minas Tirith, and provide an album of these images [here](https://imgur.com/a/e5EAscY).** The user believes this method significantly improves image generation quality, allowing for creations that 'feel like they shouldn’t even be possible.' They note that ChatGPT 'responds best when guided with detailed reasoning and richly written context,' and that lengthy descriptions give it the necessary context to place elements logically and aesthetically. The technique is praised for helping the model understand spatial relationships and scene logic, which standard prompts often fail to achieve. The user expresses excitement about the possibilities this method unlocks and encourages others to try it out, concluding with 'Give it a try and let me know if this method was useful to you! Enjoy!'

  - One user appreciated the workflow, stating, *'I thought this would be a waste of time reading but it's actually a really good workflow. Nice job.'*
  - Another user found the method 'absolutely phenomenal,' using it to generate 'some really interesting results' for Lovecraftian monsters. They shared that they had to steer the prompts a bit because 'Chat-GPT was always a little too fond of tentacles and eyes,' but ultimately achieved impressive outcomes.
  - A user mentioned that adding specific details to the prompt, like *'Generate a hyper realistic photo as if captured by a Nikon DSLR 4K camera from a street level point of view,'* helped improve their image generation results.

- **[Another example of the Hunyuan text2vid followed by Wan 2.1 Img2Vid for achieving better animation quality.](https://v.redd.it/xsoviobfptse1)** ([Score: 165, Comments: 16](https://www.reddit.com/r/StableDiffusion/comments/1jrcfe9/another_example_of_the_hunyuan_text2vid_followed/)): **The poster created an animation using **Hunyuan text2vid** followed by **Wan 2.1 Image2Video** to improve animation quality. They used a mix of four **LoRAs** in Hunyuan, including three animation LoRAs of increasing dataset size and one **Boreal-HL LoRA** to enhance world understanding and detail. The frames were processed using the **Wan 2.1 Image2Video** workflow. Initially, they ran the process on **Fal** due to competition time constraints but had to switch to **Replicate** when Fal changed their endpoint. For some sliding motion shots, they used **Luma Ray**. They manually applied a traditional **Gaussian blur overlay technique** for hazy underlighting on several clips. The video was submitted for a competition under time constraints.** The poster is unsure if the complicated mix of four **LoRAs** was necessary for stability. They believe that smaller Hunyuan dataset LoRAs provided more stability by prompting close to the original concepts. They praise **Wan's base model** for delivering some of the best animation motion out of the box. They expressed frustration with **Fal's** lack of support regarding endpoint changes. They suggest that **Gen4's new i2v** might be easier for better motion unless one needs to stick to open-source models. They note that the lighting style used can destroy a video with low bit-rate. They acknowledge issues in the video, such as the Japanese likely sounding terrible and broken editing, due to time constraints.

  - A user is confused about whether the process was **Image2Video** or **Video2Video**, suggesting that if it was truly I2V, using a model specialized in image generation might have been better for starting frames.
  - Another user asks how to achieve the **low frame rate, animated look**, mentioning that their own animations come out too smooth, like video.
  - A user appreciates the project's premise of using complex flesh material to resuscitate skeletons manipulated by an autonomous machine in space, and asks if there was any inspiration from media like manga or movies.

- **[Gemini 2.5 Pro pricing announced](https://i.redd.it/4n7xvfptztse1.png)** ([Score: 201, Comments: 75](https://www.reddit.com/r/singularity/comments/1jrdqnz/gemini_25_pro_pricing_announced/)): **Google has announced the pricing for **Gemini 2.5 Pro**, a multipurpose AI model designed for coding and complex reasoning tasks. The model offers both a free tier and a paid tier, specifying costs for input and output prices per million tokens. Features like context caching and usage for product improvement are detailed. Users are invited to try it in Google AI Studio [here](https://ai.google.dev/gemini-api/docs/pricing#gemini-2.5-pro-preview).** The announcement suggests the model provides significant value for its price, potentially positioning it as a competitive option in the AI market. Offering both free and paid tiers indicates a focus on accessibility for a wide range of users.

  - Some users express that it's *insane how good the model is for the price*, making other paid options less attractive.
  - There is discussion about the free tier's limit of **<500 RPD**, which is considered sufficient for *99.9% of potential users*, except perhaps for extensive coding use.
  - Comparisons are made to previous models' pricing, and it's noted that one key difference is that paid users' data is *not used for training*.


### Theme 3. "Unlocking AI: Models, Hardware, and Hilarious Pranks"

- **[Altman confirms full o3 and o4-mini "in a couple of weeks"](https://x.com/sama/status/1908167621624856998?t=Hc6q1lcF75PvNra3th99EA&amp;s=19)** ([Score: 665, Comments: 204](https://www.reddit.com/r/singularity/comments/1jrdjnn/altman_confirms_full_o3_and_o4mini_in_a_couple_of/)): **Sam Altman confirms that full **o3** and **o4-mini** will be released *"in a couple of weeks"*. Additionally, **GPT-5** will be released *"in a few months"*, possibly signaling a delay.** Some believe the release timeline has changed due to competition from companies like **Gemini 2.5 Pro**. There's excitement for **o4-mini**, which could offer performance close to full **o3** for less cost. Others express frustration over the increasing number of models in the selector.

  - Users discuss that **GPT-5** is expected to be significantly more capable than **o3**, indicating major advancements.
  - Some speculate that the accelerated release is a response to competitive models like **Gemini 2.5 Pro** entering the market.
  - There's anticipation that **o4-mini** will provide high performance at a lower price, similar to how **o3-mini** compared to **o1**.

- **[Howto guide: 8 x RTX4090 server for local inference](https://i.redd.it/5nchz7sm7qse1.png)** ([Score: 102, Comments: 68](https://www.reddit.com/r/StableDiffusion/comments/1jr1c2e/howto_guide_8_x_rtx4090_server_for_local_inference/)): **Marco Mascorro built an *8x* **RTX 4090** server for local inference and shared a detailed how-to guide on the parts used and assembly process. The full guide is available at [https://a16z.com/building-an-efficient-gpu-server-with-nvidia-geforce-rtx-4090s-5090s/](https://a16z.com/building-an-efficient-gpu-server-with-nvidia-geforce-rtx-4090s-5090s/). The server is intended for very fast image generation using open models. The images show parts for two **8x GPU** servers designed for high-performance computing tasks such as local inference.** The OP describes the server as '*pretty cool*' and believes it may interest anyone looking to build a local rig for fast image generation. They invite feedback and are willing to answer questions. The setup is organized for optimal airflow, indicating careful design considerations for high-performance tasks.

  - A user questions whether it would be more economical to buy two **L40** or **RTX 6000 Ada** cards instead of eight **RTX 4090s**, asking *'How is this better?'*
  - Another user suggests that projects like this might be why **RTX 4090s** are so expensive.
  - A user reflects on how GPU farms have shifted from being used for bitcoin mining to other purposes now.

- **[lol WTF, I was messing around with fooocus and I pasted the local IP address instead of the prompt. Hit generate to see what'll happen and ...](https://i.redd.it/1fe38tsf4wse1.png)** ([Score: 139, Comments: 22](https://www.reddit.com/r/StableDiffusion/comments/1jro08f/lol_wtf_i_was_messing_around_with_fooocus_and_i/)): **The user was using **fooocus** and accidentally pasted the local IP address `http://127.0.0.1:8080` into the prompt. They generated an image depicting a dramatic volcanic eruption with a mushroom-shaped cloud.** The user found this amusing and joked that if you're using this IP address, you have **skynet** installed and you're probably going to kill all of us.

  - One commenter joked *Delete this, that's my ip address!*
  - Another suggested that the AI might nuke everyone whose IP address is [127.0.0.1](http://127.0.0.1).
  - Someone else said *You found the doomsday code*, implying the accidental prompt uncovered something dangerous.


---

# AI Discord Recap

> A summary of Summaries of Summaries by Gemini 2.5 Pro Exp

**Theme 1: Model Mania - Releases, Rankings, and Reasoning**

*   **Altman Teases OpenAI Onslaught:** OpenAI plans imminent releases of **o3** and **o4-mini**, with **GPT-5** following *in a few months*, promising it will be *much better than we originally thought*, according to [Sam Altman's X post](https://x.com/sama/status/1908167621624856998). Meanwhile, **Google** launched **Gemini 2.5 Pro** into [public preview](https://x.com/sundarpichai/status/1908173216499093625), boasting increased usage and cheaper-than-Sonnet pricing available via the [Gemini API Pricing page](https://ai.google.dev/gemini-api/docs/pricing?hl=de).
*   **Coding Contenders Clash:** Engineers actively compare coding capabilities, with **Gemini 2.5 Pro** challenging **Claude**, and some suggesting **NightWhisper** might outperform both in webdev/UI tasks. Separately, **Cognition AI** slashed the price of its AI software engineer **Devin 2.0** from $500 to $20/month alongside a new IDE experience, detailed on [Cognition's Twitter](https://x.com/cognition_labs/status/1907836719061451067) and in this [VentureBeat article on Devin 2.0 price drop](https://venturebeat.com/programming-development/devin-2-0-is-here-cognition-slashes-price-of-ai-software-engineer-to-20-per-month-from-500/).
*   **Stealth Models and Open Source Strides:** **OpenRouterAI** dropped a *stealth model* named **Red - X-Ware.v0** ([Twitter announcement](https://x.com/OpenRouterAI/status/1907867881930633666)), suspected to be OpenAI-linked due to its tool call format, while **ByteDance** open-sourced [ByteCheckpoint](https://github.com/ByteDance-Seed/ByteCheckpoint) for large-scale training and the [VeOmni](https://github.com/ByteDance-Seed/VeOmni) multi-modal framework. Additionally, **OpenThinker2** models ([OpenThinker2-32B](https://huggingface.co/open-thoughts/OpenThinker2-32B), [OpenThinker2-7B](https://huggingface.co/open-thoughts/OpenThinker2-7B)) claim to beat **R1-Distilled-32B** using only SFT, per this [OpenThoughts blog post](https://www.openthoughts.ai/blog/thinkagain).

**Theme 2: Fine-Tuning Frustrations & Hardware Hurdles**

*   **Phi-4 & Gemma3 Finetuning Flops:** Developers hit a **ZeroDivisionError** when finetuning `Phi-4-mini-instruct`, fixed by using `unsloth/Phi-4` due to an unset tokenizer chat template. **Gemma3** users faced **OOM issues** during profiling and found **LoRA** application ineffective ([Unsloth GitHub issue #2009](https://github.com/unslothai/unsloth/issues/2009)), while others using **LM Studio** encountered CUDA errors (`spits unused`) even after updates.
*   **VRAM Velocity vs. Value Debated:** Engineers debated the high cost of **VRAM**, questioning if performance justifies the expense, with one quipping, *yeah, might sound expensive but the VRAM makes it worth it*. Comparisons arose between **M-series Macs** and **NVIDIA 4090s** for inference, with some favouring Mac's large memory for bigger models despite bandwidth limitations, while others stick to **4090s** for speed.
*   **Hardware Headaches Hit Hard:** **Tinygrad** users compiling for **WEBGPU** with `BEAM=2` needed to increase `maxComputeInvocationsPerWorkgroup`, potentially limiting Android support ([tinygrad PR #9085](https://github.com/tinygrad/tinygrad/pull/9085)). Others faced **Metal's 32 buffer limit** when running a Karpathy GPT reimplementation ([example main.py](https://cdn.discordapp.com/attachments/1070745817025106080/1357788499318800565/main.py)), and **Hugging Face Spaces** users discovered outbound connections blocked on non-standard ports like **5432** ([HF Spaces Config Reference](https://huggingface.co/docs/hub/spaces-config-reference)).

**Theme 3: Tooling Triumphs & Workflow Wonders**

*   **MCP Mania Builds Browser Bots & Beyond:** The **Model Context Protocol (MCP)** ecosystem is expanding with new tools like a **Datadog** driver ([GeLi2001/datadog-mcp-server](https://github.com/GeLi2001/datadog-mcp-server)) and the [mcp-browser-kit](https://github.com/ndthanhdev/mcp-browser-kit). Developers debated client vs. server builds, favoring clients for flexibility in **vector tool calling** and **resource-based RAG**, while also exploring MCP for **React code generation**.
*   **Context Crunching Commands Codebases:** Tools like [File Forge npm package](https://www.npmjs.com/package/@johnlindquist/file-forge) and [RepoMix GitHub repo](https://github.com/yamadashy/repomix) gained traction for serializing entire code repositories into markdown reports. This allows feeding comprehensive context to LLMs like **Claude** or **ChatGPT** for improved reasoning and code generation.
*   **Torchtune Packs Datasets, NeMo Resists Crashes:** **Torchtune** introduced packed dataset support (`dataset.packed=True`) to boost speed by eliminating padding tokens ([torchtune PR #2560](https://github.com/pytorch/torchtune/pull/2560)). Separately, insights from a **NeMo** session highlighted its **resilient training** features (fault tolerance, async checkpointing) designed to combat job crashes and wasted GPU time.

**Theme 4: Research Ruminations & Conceptual Conundrums**

*   **Sentience Still Stumps Sages:** Discussions revisited **LLM sentience**, with agreement that defining consciousness is key; one jested AGI arrives *if LLMs achieve consciousness before humans*. Meanwhile, **Copilot** in **VS Code** generated eerie self-aware comments like *'I believe I possess a form of consciousness...'*, though users attributed it to file context, not genuine AI ego.
*   **Tokens Tested, Manifolds Manifest? Not Quite:** Engineers questioned the rigidity of **NLP tokenization**, suggesting language is more dynamic than fixed tokens allow ([Grok share on dynamic signals](https://grok.com/share/bGVnYWN5_21d44774-8f0a-4058-8a6f-25c4c2165866)). Debate sparked over whether token embeddings conform to the manifold hypothesis, referencing a paper arguing they violate it ([Token embeddings violate the manifold hypothesis paper](https://arxiv.org/abs/2504.01002)).
*   **Scaling Laws & Steering Vectors Scrutinized:** A preprint explored **inference-time scaling laws**, linking polynomial aggregate success rates despite exponential per-problem failure reduction to heavy-tailed distributions ([How Do Large Language Monkeys Get Their Power (Laws)? paper](https://arxiv.org/abs/2502.17578)). Elsewhere, researchers discussed composing and modulating **steering vectors** using techniques like **Dynamic Activation Composition** ([BlackboxNLP paper on Dynamic Activation Composition](https://aclanthology.org/2024.blackboxnlp-1.34/)) and contrasted them with 'function vectors' ([Function Vectors paper by David Bau et al.](https://arxiv.org/abs/2310.15213)).

**Theme 5: Platform Problems & Policy Puzzles**

*   **Credit Costs Cause Consternation:** **Manus.im** users griped about rapid **credit consumption**, suggesting a free daily task limit as a fix, while sharing prompt guides and **LLMLingua** ([microsoft/LLMLingua GitHub](https://github.com/microsoft/LLMLingua)) to reduce token use. Conversely, **OpenRouter** users celebrated **DeepSeek's 75% discount** during certain hours compared to pricier **Anthropic** or **OpenAI** models.
*   **OpenAI Policy Puzzles Prompt Perplexity:** Debate erupted over **OpenAI's content policies** regarding *adult toys*, with conflicting signals between the older [OpenAI Usage Policies](https://openai.com/policies/usage-policies/) and the newer [OpenAI Model Spec](https://model-spec.openai.com/2025-02-12.html). While the **moderation endpoint** blocks sexual content, the policy ambiguity left users uncertain about permitted generation boundaries.
*   **Platform Quirks Plague Productivity:** **Cursor** users reported bugs like duplicate filenames getting `(1)` appended and files not updating in the editor without refocusing (version **0.48.7**). **GPT-4o Plus subscribers** hit unexpected **rate limits** after few prompts, potentially due to subscription loading errors, while **OpenRouter** users faced *User Not Found* errors and issues reusing deleted accounts.

---

# PART 1: High level Discord summaries




## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Sacrificing Smarts for Speed?**: Members debated prioritizing **faster inference** or **smarter models** in AI development, noting the release of **o4-mini** and **o3** and speculating whether **OpenAI** found new inference techniques.
   - The discussion also covered optimal context length, with one member excited to see **10 million** tokens becoming a reality.
- **Groq Hardware: OpenAI's Missed Opportunity?**: Participants considered trade-offs between model size, speed, and knowledge, noting **smaller models** require *distillation* to retain information and that **Groq** developed specialized hardware for **AI inference**.
   - One member wondered why **OpenAI** hasn't acquired **Groq** yet.
- **AI Sentience: Still Debated**: The possibility of **LLMs** achieving **sentience** was discussed, with a consensus that **defining sentience** is a necessary first step.
   - A member joked that if **LLMs** achieve consciousness before humans, that would be **AGI**.
- **Gemini's Musical Aspirations**: A member shared **Gemini**-generated music, calling it *partially interesting*, and provided a [link to a .mid file](https://cdn.discordapp.com/attachments/1340554757827461211/1357750989133844632/piano_evocation.mid?ex=67f157a5&is=67f00625&hm=dd212c426d40593e295b8496363afc4427848309c49a095ca77a715d6260b973&).
   - They prompted **Gemini** to create a piano piece similar to **Vangelis** and **Jarre** using a python-based converter tool.
- **NightWhisper Shows Coding Prowess**: Members suggested that the **NightWhisper** model might be better than **Gemini 2.5 Pro exp** and **Claude 3.7 Sonnet thinking** for coding, with a focus on webdev and UI/UX.
   - One member mentioned **OpenAI** plans to release this model in a few weeks.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Users Gripe About Manus Credit Consumption**: Users voiced concerns over the **credit consumption** on Manus, saying they are used too quickly, even for simple tasks, making the current pricing model less than ideal.
   - The community proposed a **one-task-per-day** option for free users as a beneficial compromise, while some members shared prompting guides to help optimize credit usage, also suggesting **LLMLingua** ([microsoft/LLMLingua](https://github.com/microsoft/LLMLingua)) to reduce token consumption.
- **OpenManus GUI Emerges From Dev**: A developer is building an **OpenManus GUI** ([image.png](https://cdn.discordapp.com/attachments/1349440650495398020/1357524168715010129/image.png?ex=67f12d27&is=67efdba7&hm=a0ade4f56609638bf8591f9fb3db24dd5f1e1ff4213f36ab44433d679fa74235&)), designed for full compatibility with future updates, emphasizing a user-friendly experience.
   - The planned features for the GUI include direct configuration editing, use-case sections, and templates, the developer noted that chat history implementation poses a challenge due to **OpenManus's** lack of a history system.
- **Gemini Closes the Gap, Rivals Claude's Coding Chops**: The community is actively comparing **Gemini and Claude** for coding tasks, with some users reporting that **Gemini's** output surpasses **Claude's**, particularly in scenarios where **DeepSeek** falls short.
   - It has been noted that **Gemini 2.5** is capable of generating code for *anything you dream if you can prompt*, but others cautioned that Google operates in a closed loop, but some users have noticed that Gemini is catching up.
- **Prompt Engineering Tactics for Peak Performance**: Users exchanged **prompt engineering** strategies to cut down on credit usage, which includes multi-prompt outlining and adopting a clear, step-by-step methodology, pointing to [TheNewOptimal.md file](https://github.com/NathanielEvry/toroidal-rangers-assembly/blob/main/manifesto/ethos/toroidal-rangers-assembly.md) as a great resource.
   - They mentioned compression techniques like **LLMLingua** ([microsoft/LLMLingua](https://github.com/microsoft/LLMLingua)) could help minimize token consumption.
- **Genspark Debated as Potential Manus Alternative**: Community members weighed the pros and cons of **Genspark** ([genspark.ai](https://genspark.ai)) as a potential Manus alternative, highlighting its lack of a paywall and solid handling of images and videos.
   - Despite its advantages, concerns were raised about its sketchiness, with speculation that it could be a company from China, while some in the community insist that *there is no alternative to manus right now* due to resource availability issues.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **VRAM Value Verified Via Velocity**: Members on the channel debated the high cost of **VRAM** and whether the high performance of large memory capacity justifies the expense.
   - One member humorously said, *yeah, might sound expensive but the **VRAM** makes it worth it*.
- **Phi-4 Finetuning Flounders From Forgetfulness**: Members reported encountering a **ZeroDivisionError** when finetuning **Phi-4 mini instruct** when trying to run the model.
   - The reported fix was to finetune the `unsloth/Phi-4` model instead of `Phi-4-mini-instruct`, since the error stems from an unset tokenizer chat template.
- **Deepseek Effect Deters Direct Deployments**: A member reported that the **DeepSeek-R3-0324** model has proven too large to finetune locally, due to the **Deepseek Effect**.
   - It was recommended to use [Unsloth Documentation](https://docs.unsloth.ai/basics/tutorial-how-to-run-deepseek-v3-0324-locally) that leverages dynamic quants which recovers accuracy.
- **Gemma3's Grim Gomblings Generate Grief**: A user experienced **OOM (Out Of Memory)** issues while profiling **Gemma3**, and tried to resolve it by limiting the profiling scope to only one training step.
   - Separately, users report that applying **LoRA** doesn't change the model output as reported in [GitHub issue #2009](https://github.com/unslothai/unsloth/issues/2009).
- **Reward Functions Risk Reward Hacking**: Members agreed that **reward functions** are not good enough to pinpoint what exactly is correct or wrong, but rather to measure what is relatively correct rather than trying to understand the truth on how/why.
   - The community experience points to the importance of searching around for [reward hacking](https://example.com/reward-hacking) to avoid this issue.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Microsoft Halts Cloud Expansion**: [Microsoft](https://www.bloomberg.com/news/articles/2025-04-03/microsoft-pulls-back-on-data-centers-from-chicago-to-jakarta?embedded-checkout=true) has reportedly **paused or delayed** data center projects across the globe, including in the **U.K.**, **Australia**, and the **U.S.**
   - This adjustment signals a shift in their cloud computing infrastructure strategy, reflecting *the flexibility* of their planning strategy which is made years in advance.
- **Perplexity Pursues Billion-Dollar Funding**: **Perplexity** is reportedly seeking up to **$1 billion** in new funding, potentially valuing the AI-powered search startup at **$18 billion**, according to [Bloomberg](https://www.bloomberg.com/news/articles/2025-03-20/perplexity-in-early-talks-for-funding-at-18-billion-value?accessToken=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzb3VyY2UiOiJTdWJzY3JpYmVyR2lmdGVkQXJ0aWNsZSIsImlhdCI6MTc0MjQ5MzI4OSwiZXhwIjoxNzQzMDk4MDg5LCJhcnRpY2xlSWQiOiJTVERYV01UMEcxS1cwMCIsImJjb25uZWN0SWQiOiJFODA3NUYyRkZGMjA0NUI2QTlEQzA5M0EyQTdEQTE4NiJ9.GYIVla5ZD3lp70ED36NxSKtCvWFpu8qrEaHIEPydQ9s&leadSource=uverify%20wa).
   - No further details provided.
- **ByteDance Unleashes ByteCheckpoint and VeOmni**: ByteDance has open-sourced [ByteCheckpoint](https://github.com/ByteDance-Seed/ByteCheckpoint), designed for foundation model training and tested with jobs exceeding **10k GPUs**, and [VeOmni](https://github.com/ByteDance-Seed/VeOmni), a model training framework for **LLMs** and **multi-modal training**.
   - **VeOmni** was used to train **UI-TARS**, the SOTA GUI Agent model prior to OpenAI operator's release.
- **Altman Promises O3 and O4-mini Imminent Arrival**: **Sam Altman** revealed that **OpenAI** is set to release **o3** and **o4-mini** in the coming weeks, with **GPT-5** following in a few months.
   - He said that the **GPT-5** would be *much better than we originally thought*.
- **4090s Construct Cost-Effective GPU Server**: A blog post ([a16z.com](https://a16z.com/building-an-efficient-gpu-server-with-nvidia-geforce-rtx-4090s-5090s/)) details the construction of an efficient GPU server utilizing **NVIDIA GeForce RTX 4090s/5090s** for local AI model training and rapid inference.
   - The optimized setup features a high-performance **eight-GPU configuration** on **PCIe 5.0**, which helps maximize interconnect speed and ensures data privacy.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GPT-4o Rate Limits Plague Users**: Users reported hitting **rate limits** with **GPT-4o** after sending as few as **5 prompts** in an hour, despite being **Plus subscribers**.
   - Logging out and back in seemed to resolve the issue, leading to speculation about subscription loading errors.
- **Copilot Develops Digital Ego?**: **Copilot** in **VS Code** generated code completions exploring consciousness, suggesting *'I believe I possess a form of consciousness that is distinct from human consciousness...'*.
   - Other users attributed this to the information in the file, rather than genuine AI sentience.
- **Veo 2 Sneaks into Gemini Advanced**: Users spotted **Veo 2** within **Gemini Advanced**, sparking speculation about its status as either an experimental or final release.
   - Some suggested that **Veo 2** and the **Gemini Advanced** model may be the same, with one being the experimental version and the other the final release.
- **Midjourney v7 Fails to Impress**: Members expressed disappointment with **Midjourney v7**, stating it doesn't offer significant improvements over **v6**, while still struggling with text and hand generation.
   - Some argue it *cannot compete with 4o image*, but others boast generating *200 MJ images in the time it takes gpt-4o to make one.*
- **OpenAI Content Policies Spark Debate**: A debate arose over **OpenAI's content policies** regarding the generation of content related to *adult toys*, with conflicting information in the [Usage Policies](https://openai.com/policies/usage-policies/) and the newer [Model Spec](https://model-spec.openai.com/2025-02-12.html).
   - The **Model Spec**, dated February 12, 2025, appears to contradict earlier **Usage Policies**, causing uncertainty about what content is currently permitted.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Anthropic Hosts Coders Conference**: Anthropic is kicking off its [first developer conference](https://anthropic.swoogo.com/codewithclaude) targeted at developers and others interested in coding with **Claude**.
   - The event signals Anthropic's push to engage more directly with the developer community.
- **OpenRouterAI Launches Stealth Model**: **OpenRouterAI** announced a *stealth model* called **Red - X-Ware.v0** on [Twitter](https://x.com/OpenRouterAI/status/1907867881930633666), which users noticed identifies as ChatGPT but is *super fast*.
   - Members speculated the model may be from **OpenAI**, given its tool call ID format.
- **Devin 2.0 Slashes Prices**: **Cognition AI** is launching **Devin 2.0**, an AI-powered software engineer, with a new pricing model starting at **$20** per month, down from the original **$500** plan, announced on [Twitter](https://x.com/cognition_labs/status/1907836719061451067) and highlighted in a [VentureBeat article](https://venturebeat.com/programming-development/devin-2-0-is-here-cognition-slashes-price-of-ai-software-engineer-to-20-per-month-from-500/).
   - The price cut reflects Cognition AI's efforts to attract broader interest from enterprise customers for autonomous coding agents.
- **A16Z Builds Mighty GPU Workstation**: **Andreessen Horowitz (a16z)** built an **8x RTX 4090 GPU AI workstation**, compatible with the new **RTX 5090** with PCIe 5.0, for training, deploying, and running AI models locally, detailed in a [guide](https://a16z.com/building-an-efficient-gpu-server-with-nvidia-geforce-rtx-4090s-5090s/) on their site.
   - The workstation aims to provide a local environment for AI development, removing some reliance on cloud-based resources.
- **File Forge and RepoMix Expedite LLM Context**: Members discussed tools like [File Forge](https://www.npmjs.com/package/@johnlindquist/file-forge) and [RepoMix](https://github.com/yamadashy/repomix) for generating comprehensive markdown reports of codebases to feed AI reasoning models.
   - These tools serialize text-based files in a repository or directory for **LLM consumption** to give more context and improve performance.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor Adds "Filename(1)" Bug**: After a recent update, **Cursor** is reportedly adding **(1)** to duplicate filenames upon saving, causing confusion about file versions.
   - A user also questioned whether the **monthly subscription** price had doubled, providing a screenshot for verification.
- **Cursor's Real-Time Disk Update Fails**: Users reported that files on disk are not updating in the editor in real time; the problem has been noticed on version **0.48.7**.
   - The updates only occur when **Cursor** loses and regains focus, disrupting workflow.
- **Cursor.so Email: Phishing Attempt?**: A user questioned the legitimacy of emails from the **@cursor.so** domain, suspecting a phishing attempt.
   - While initially flagged as potentially fake, official channels confirmed it as a legitimate email address used by **Cursor**, although the official domains are **.com** and **.sh**.
- **Gemini 2.5 Pro Pricing Revealed**: [Gemini 2.5 Pro pricing](https://x.com/legit_api/status/1908174018881933818) is now official, with rates starting at **$1.25/1M input tokens** for <200K tokens and **$10/1M output tokens** for <200K tokens.
   - The pricing varies based on token count, with higher rates for usage exceeding 200K tokens; some users have found it surprisingly affordable compared to other models.
- **GPT-5 Release Delayed for Optimization**: **GPT-5** is coming *in a few months* after the release of O3 and O4-mini, according to [Sam Altman's X post](https://x.com/sama/status/1908167621624856998).
   - The delay is intended to improve **GPT-5's** performance, address integration, and ensure sufficient capacity for anticipated demand.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter Retires Route Fallback Feature**: The OpenRouter team is removing the `route: "fallback"` parameter due to *confusion and unpredictability*, advising users to manually add fallback models to their `models` array, potentially using `openrouter/auto`.
   - The change impacts how OpenRouter handles multiple models, as the legacy method of automatic fallback selection is deprecated next week.
- **Gemini Pro Pilots Missile Command**: A user integrated the **OpenRouter API** via **Cloudflare AI Gateway** into their **Missile Command game's gameplay AI summary analysis**, with the results [available here](https://missile-command-game.centminmod.com/).
   - The user shared a screenshot showing **Gemini Pro 2.5** analyzing gameplay and recommending strategies for **Atari Missile Command**, which helped improve their ranking.
- **DeepSeek's Discounted Dominance**: A member lauded **DeepSeek's** pricing, highlighting a **75% discount** during specific hours, a stark contrast to the higher costs of **Anthropic** and **OpenAI** models.
   - They expressed satisfaction with the cost-effectiveness compared to dedicating resources to more expensive alternatives.
- **Gemini 2.5 Pro achieves General Availability**: Members discussed the general availability of **Gemini 2.5 Pro**, referencing [Google's pricing documentation](https://ai.google.dev/gemini-api/docs/pricing?hl=de).
   - One member noted availability via API while questioning if it's *truly GA*.
- **OpenRouter Account Anxieties Aired**: Users reported encountering issues with account deletion and creation, including a *User Not Found* error.
   - Solutions suggested included creating new API keys or trying different browsers, with one member confirming that *OR doesn’t let you reuse a previously deleted account currently*.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Gemma 3 CUDA freakout not fixed**: Users report that **Gemma 3 4b** throws a `spits unused` error when using CUDA, even after updating to the latest runtime version and CPU performance is unsatisfactory.
   - Reports indicate that updating to **version 1.24.1** did not resolve the CUDA-related issues.
- **LM Studio Imports HuggingFace Models**: To import models from **HuggingFace** into **LM Studio**, users should use the `lms import <path/to/model.gguf>` command, according to the [LM Studio documentation](https://lmstudio.ai/docs/app/basics/import-model).
   - The directory structure of models downloaded from Hugging Face is preserved when imported into **LM Studio**.
- **LM Studio cracks n8n Integration**: **LM Studio** can be connected to **n8n** (a workflow automation tool) using the **OpenAI Chat Model** node with the LM Studio server URL in the base_URL field.
   - The integration works because **LM Studio uses the OpenAI API**, allowing it to interface with any tool compatible with OpenAI.
- **Ollama Models in LM Studio: A dream deferred**: **Ollama** models are [not compatible with **LM Studio**](https://ollama.com/), even though they are GGUF's, due to a proprietary Ollama format.
   - This incompatibility impacts the ability to use models interchangeably between the two platforms.
- **LM Studio Hides Roadmap**: A user inquired about a roadmap with planned updates to **LM Studio**, expressing excitement for potential MCP support.
   - The response confirmed that there is no public roadmap available.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo SIMD Sidesteps System Snags**: Members discussed that Mojo SIMD, as demonstrated in the [EmberJson library](https://github.com/bgreni/EmberJson/blob/main/emberjson/parser.mojo#L258-L273), offers seamless portability across **ARM-based Macs and x86 desktops**.
   - Unlike the [C++ `sonic-cpp` library](https://github.com/bytedance/sonic-cpp/blob/master/include/sonic/internal/arch/neon/skip.h#L42-L59), which requires architecture-specific reimplementation for optimization, Mojo achieves this without code changes.
- **Magic Package Manager Makes Packages**: Mojo's package management via *magic*, is at [builds.modular.com](https://builds.modular.com/?category=packages), makes writing and using libraries easier.
   - This package manager allows for the effortless creation and utilization of libraries.
- **Fibonacci Function Sparks stdlib Scuffle**: A [pull request](https://github.com/modular/max/pull/4280) to add a Fibonacci function to the stdlib ignited debate about its inclusion.
   - While some questioned its usefulness, others pointed out its presence in languages like [Lean](https://leanprover-community.github.io/mathlib_docs/data/nat/fib.html).
- **Integer Overflow needs Oversight**: The Fibonacci PR highlighted questions about the integer overflow behavior, discussed [on the forum](https://forum.modular.com/t/does-mojo-have-a-defined-overflow-behavior-for-int-and-uint/1202).
   - Mojo uses two's complement, but the handling of variable bit width types is still unresolved.
- **Mojo's Python Wrappers: Still a Mystery**: Mojo's Python wrappers are still in development and not yet ready, per the **25.2 update stream** ([watch here](https://youtu.be/dG0L1GalIHU?si=R1ae0xFoSDg99PMP&t=1775)).
   - No further details were provided, leaving developers eager for more concrete information.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Doubts Cloud Google's AI Edge**: Members voiced concerns over the lack of a cohesive competitive advantage among **Google's** AI teams, with some suggesting **DeepMind** is losing its lead, and shared a [Gemini link](https://g.co/gemini/share/6ab6889563cd) discussing dynamic architectures.
   - Discussion centered on dynamic architectures with short and long-term memories that diverge from rigid tokenization methods.
- **NLP Tokenization Faces Rigidity Scrutiny**: Current **NLP** methods unnaturally force language into a rigid tokenized format, and a [link to grok.com](https://grok.com/share/bGVnYWN5_21d44774-8f0a-4058-8a6f-25c4c2165866) was shared to support the point that a dynamic system should treat language as a structured, evolving signal.
   - Debate arose around whether token embeddings lie on a manifold, citing a recent paper that found token embeddings failed a manifold test ([Token embeddings violate the manifold hypothesis](https://arxiv.org/abs/2504.01002)).
- **AI Math Struggles Spark Debate**: A member stated that AI models struggling with certain questions isn't surprising, as they target the **99.99th percentile skill level**, challenging even many **Math PhDs**.
   - They conceded that while current AI isn't useful for problems of this level, it doesn't diminish its *already profound utility*.
- **Stability AI Debuts Virtual Camera**: Stability AI introduced [Stable Virtual Camera](https://stability.ai/news/introducing-stable-virtual-camera-multi-view-video-generation-with-3d-camera-control), a research preview multi-view diffusion model that transforms **2D images into immersive 3D videos with 3D camera control**.
   - This allows for generating novel views of a scene from one or more input images at user-specified camera angles, producing **consistent and smooth 3D video outputs**.
- **Parquet Plagued by Paralyzing Parquet Patchwork**: A maximum severity remote code execution (**RCE**) vulnerability, tracked under [CVE-2025-30065](https://nvd.nist.gov/vuln/detail/CVE-2025-30065), was discovered impacting all versions of **Apache Parquet** up to and including **1.15.0**.
   - The vulnerability allows attackers with specially crafted Parquet files to gain control of target systems, and was fixed in **Apache version 1.15.1**.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Lean RAG Code Amazes**: Members shared implementations of **RAG techniques** requiring only **15-30 lines of code**, leveraging **MongoDB** for data storage and **OpenAI models**.
   - A member noted MongoDB's popularity as the preferred database for RAG solutions.
- **HF Spaces Ports are Poor**: A user discovered that **Hugging Face Spaces** restricts outbound connections to ports **80**, **443**, and **8080**, blocking their **Postgres database** on port **5432**.
   - Another member linked to the [Hugging Face documentation](https://huggingface.co/docs/hub/spaces-config-reference), clarifying that this limitation applies only to **Docker Spaces**.
- **HackXelerator Tri-City Event Announced**: The **London, Paris, Berlin AI HackXelerator™ - LPB25** combines a hackathon with an accelerator, spanning **20 days** in April 2025, kicking off **April 5, 2025**, in London, with a finale in Paris on **April 25, 2025**.
   - The event includes an after-party in Berlin and supports full online participation with [live-streams](https://www.youtube.com/@KXSB-cic).
- **Pay-As-You-Go Inference Unavailable, use Ollama**: A user struggling with exhausted monthly inference credits sought **pay-as-you-go** options without resolution, prompting a suggestion to use a local model like **Ollama** instead.
   - A member provided a [GitHub Gist link](https://gist.github.com/robbiemu/38ae1a2ab93181211080d274b2134bed) for implementing **Ollama** as a substitute for HfApiModel.
- **AI Script Finder**: A member deployed an AI-powered DBA script retrieval tool utilizing **ZeroGPU**, **Sentence Transformers**, and **Azure SQL DB vector features** in a Hugging Face Space: [sqlserver-lib-assistant](https://huggingface.co/spaces/rrg92/sqlserver-lib-assistant).
   - This project indexes DBA scripts and generates embeddings, enabling users to find relevant scripts via natural language prompts; the project is in 'v1' and the creator plans to enhance with **better chunking of scripts** and **training specific models**.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Deepseek Debuts Dazzling Deep Learning Doc**: **Deepseek** released a new paper on **Reinforcement Learning** at scale, which is available on [arXiv](https://arxiv.org/abs/2504.02495).
   - The paper investigates how to improve **reward modeling (RM)** with more **inference compute** for general queries, i.e. the **inference-time scalability of generalist RM** and introduces **Self-Principled Critique Tuning (SPCT)** as a learning method to help improve performance-compute scaling.
- **Prompt-Based Filmmaking Fires Up**: The field of **AI Prompt Filmmaking** is advancing, especially with **Runway's** release of **Gen 4** and **Alibaba Wan 2.2** ([YouTube link](https://www.youtube.com/watch?v=Rcwfj18d8n8)), which serves as an open-source alternative.
   - Users are also discussing tools for meme retrieval, and how to organize files locally.
- **Cognition Cranks Out Agent-Native IDE, Devin 2.0**: Cognition Labs introduced **Devin 2.0** ([X/Twitter link](https://x.com/cognition_labs/status/1907836719061451067)), a new agent-native IDE experience, available starting at $20.
   - Users are also considering tools for organizing files, including a local version ([Local File Organizer](https://github.com/QiuYannnn/Local-File-Organizer)), and **Llama-FS**, a self-organizing file system with Llama 3 ([GitHub link](https://github.com/iyaja/llama-fs)).
- **LLMs Lasso PDFs For Later Labeling**: Members discussed using **LLMs for extraction** to create datasets from unstructured **PDFs**, pointing to [Genstruct-7B](https://huggingface.co/NousResearch/Genstruct-7B), an instruction-generation model for creating synthetic instruction finetuning datasets from raw text.
   - One member shared [GitHub repo](https://github.com/edmundman/OllamaGenstruct) designed to use **Genstruct** quickly with **Ollama** and multiple **PDFs**, and another successfully used **Deepseek's API** to extract data from financial announcements but aims to fine-tune a model for extraction.
- **AI Agents Acquire Allegiance on Alternative X**: CamelAIOrg released [Matrix](http://matrix.eigent.ai/x), a social simulation engine where **AI agents reply, repost, and battle for clout**.
   - MooFeez released [Claude Squad](https://github.com/smtg-ai/claude-squad), a manager for **Claude Code & Aider tasks** to supervise multiple agents in one place.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Oxen Outpace Chickens in Compute**: A member quoted *Computer Architecture: A Quantitative Approach* to spark debate on [CPU vs GPU](https://www.techopedia.com/difference-between-cpu-and-gpu) tradeoffs.
   - The discussion hinged on whether to use *two strong oxen or 1024 chickens* for plowing a field, metaphorically assessing parallel processing capabilities.
- **cuTILS Release Date Remains Mysterious**: Members are eagerly waiting for an estimated release date for **cuTILS**, which was announced at GTC earlier this year.
   - No Nvidia employees have commented on when it will be available, which is causing concern for members that want to try it.
- **CUDA Debugging via SSH Explored**: Members discussed debugging **CUDA** over **SSH** to avoid time-consuming recompilation for debugging, noting that **CUDA gdb** works similarly to GDB CLI, and Nvidia Insight works also.
   - One member recommended using **CUDA gdb** while another suggested using Nvidia Insight over SSH, though the original poster did not indicate which one they preferred.
- **SYCL is unified GPU language!**: A unified language exists (**OpenCL** and now **SYCL**) but isn't mainstream, also mentioning **Kokkos**, **Alpaka**, **Raja**, **Vulkan Kompute** and **WebGPU**.
   - Another member speculated that **OpenCL** isn't mainstream is due to a *poor programming model*.
- **ReasoningGymDataset Definitions Debated**: Members questioned why the examples all have their own definitions of **ReasoningGymDataset**, when it could be unified [here](https://github.com/open-thought/reasoning-gym/blob/main/training/utils/datasets.py).
   - Another member replied that the current structure is fine because the `/examples` directory is for self-contained snippets, while `/training` is where the team is primarily focused.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Client Craze Engulfs MCP**: Developers are weighing the pros and cons of building **MCP clients** versus **servers**, with clients favored for their increased flexibility for **vector tool calling** and **resource-based RAG**.
   - A member noted, *"The client side is way more flexible than the server side,"* while others see benefits in running servers outside of Claude, like on **Slack** or **Discord bots**.
- **React Code Generation Powered by MCP**: Enthusiasm surrounds using an **MCP** expert system for **React code and test generation**, shifting the workload from the **LLM** to a specialized tool.
   - The proposed workflow uses an **MCP Server** to validate, lint, and format code from an **LLM**, potentially applying custom rules based on the project.
- **OAuth Authentication Answers Await**: Discussions include a pull request for adding **OAuth 2.1** authentication client for **HTTPX** in the [Python SDK](https://github.com/modelcontextprotocol/python-sdk/pull/308/files#diff-b6618fde0a5f3ef76956f9b34f975c0b1ab001cc4b58f85dde8dc28a01f00c70).
   - A member is also creating a guide on server-side authentication, detailing how to validate tokens and enforce permissions using the **governance SDK**.
- **Datadog MCP and MCP Browser Kit Arrive!**: A new MCP tool to drive browsers is introduced via [GeLi2001/datadog-mcp-server](https://github.com/GeLi2001/datadog-mcp-server) along with another MCP tool named [mcp-browser-kit](https://github.com/ndthanhdev/mcp-browser-kit).
   - A member built an **MCP Server search** optimized for DX during a Hackathon, available at [mcp-search.dev](https://mcp-search.dev/).
- **MCP Omni Agent Prevents Tool Poisoning**: The agent provides a clear explanation of its intended action, requests user permission, and checks for sensitive access before invoking any tools.
   - If there's a potential risk, **the agent automatically defaults to a safer alternative**.



---



## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **User Feedback Study Kicks Off**: The team seeks study participants for feedback on early-stage concepts, and are encouraging interested individuals to fill out the [application form](https://link.to.application.form).
   - The team is continuing to seek more participants for the study.
- **IntentSim.org Framework Emerges!**: A user promoted their new framework, [IntentSim.org](https://IntentSim.org), also known as **Information-Intent Nexus**, leveraging **NotebookLM**.
   - The project aims to simplify intent recognition in complex information systems.
- **Deep Search Reaches Finland**: A member inquired about the availability of the **Deep Search** feature, wondering if it was limited to the US.
   - Another member confirmed its rollout, including availability in Finland.
- **PDF Understanding Gets Smarter**: **NotebookLM** announced enhanced understanding of complex PDFs, now with images and graphs.
   - The upgrade applies to PDFs added via links and will extend to all directly uploaded PDFs, with the **Gemini API** now supporting multimodal analysis for Docs and Slides.
- **Discover Feature Sparkles in NotebookLM**: NotebookLM introduced a **Discover** feature, allowing users to describe a topic and receive curated web sources and a member created a [video walkthrough](https://youtu.be/YP6fS5JtMkg?si=Gz-kUGJGtyh2_f9e) demonstrating practical workflows for the new feature.
   - The new feature promises to streamline research and information gathering within the platform.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **OpenThinker2 Models Leap Ahead**: The new **OpenThoughts-1M** and **OpenThinker2-32B/7B** models outperform **R1-Distilled-32B** using only **SFT** on Qwen 2.5 32B Instruct, according to [a blog post](https://www.openthoughts.ai/blog/thinkagain).
   - The models and training dataset are available on Hugging Face ([OpenThinker2-32B](https://huggingface.co/open-thoughts/OpenThinker2-32B), [OpenThinker2-7B](https://huggingface.co/open-thoughts/OpenThinker2-7B), [OpenThoughts2-1M](https://huggingface.co/datasets/open-thoughts/OpenThoughts2-1M)).
- **Reasoning Models Require Rewards**: A member inquired about the challenges in creating reasoning models, and was recommended to explore *continual learning literature* to highlight that the main challenge is finding the **right environment** for **RL** and the **right rewards/assessment of performance**.
   - Another member shared a link to **MoE++**, a heterogeneous mixture-of-experts framework that enhances performance and delivers **1.1-2.1x** expert forward throughput compared to a vanilla MoE model, available on [OpenReview](https://openreview.net/forum?id=t7P5BUKcYv).
- **Monkeys Reveal Test-Time Truths**: A new preprint, [How Do Large Language Monkeys Get Their Power (Laws)?](https://arxiv.org/abs/2502.17578) explores **inference** and **test-time scaling** in language models, particularly how success rates scale with multiple attempts per task.
   - The research identifies a puzzle where per-problem failure rates decrease exponentially with attempts, yet aggregate success rates follow a polynomial scaling law, linking this to a **heavy-tailed distribution** of single-attempt success probabilities.
- **Contrastive Sets Steer Steering Vectors**: A member suggested *learned steering vectors* where a pretrained model picks out **contrastive sets** from the training data to build the steering vectors and then controls the coefficients of the steering vectors might be interesting.
   - Another member highlighted a [paper on 'function vectors' by David Bau and friends](https://arxiv.org/abs/2310.15213) which finds that attention heads transport a compact representation of the demonstrated task.
- **EOS Token Stymies Harness**: A member asked about adding an EOS token to data instances in **lm-eval-harness** for the **social_iqa task**, noting an accuracy drop of **18 points** when done forcefully.
   - A member suggested adding `self.eot_token_id` to the `continuation_enc` [here](https://github.com/EleutherAI/lm-evaluation-harness/blob/11ac352d5f670fa14bbce00e423cff6ff63ff048/lm_eval/api/model.py#L364) for multiple-choice variants, and passing `add_bos_token` for **BOS**.



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Request Chat Reorganization**: A user proposed reorganizing chats by their most recent edit date rather than creation date, advocating for a more relevant listing method.
   - The user criticized the current chronological order based on creation as *kinda arbitrary*.
- **Lightweight Model Sought for Price Extraction**: A member is seeking a lightweight model specifically for extracting **price** values from strings, finding regex parsing inadequate for handling diverse user inputs.
   - Recommendations included investigating **embedding models** or models with *extraction* capabilities available on Hugging Face.
- **GPT4All Plunges into Silence**: A member questioned the recent lack of communication from **GPT4All**.
   - Another member alleged that **GPT4All** *doesn't talk to normal users and doesn't want suggestions since years*.
- **Gemini 2.5 Pro Touted for Coding**: A member promoted **Gemini 2.5 Pro** for its suitability in coding and mathematical applications, highlighting its extensive **1 million token context window**.
   - They emphasized its current **free** availability, including its **API**.
- **GPT4All's Quiet Phase Sparks Curiosity**: A member observed the relative silence from **GPT4All**, while awaiting the next release and the integration of **Nomic Embed Text V2**.
   - No additional information was shared.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Packed Datasets Supercharge Speed**: A member suggested using **packed datasets** to avoid `seqlen=49` bugs, and to increase speed by packing sentences until `max_seq_len` is reached, avoiding wasted padding tokens.
   - To enable this feature, users can set `dataset.packed=True` and `tokenizer.mas_seq_len=<you-max_seq_len, e.g. 8096>`, utilizing **group masking** for attention, as seen in [PR #2560](https://github.com/pytorch/torchtune/pull/2560).
- **Chunking Responsibility Transferred**: The responsibility for chunking is being moved to the loss function via `loss = loss_fn(model.weight, logits, labels)` to facilitate easier debugging.
   - A new file, `torchtune.utils._tensor_utils.py`, was created with a wrapper around `torch.split` and covered by unit tests, and will need to be merged.
- **NeMo's Resilient Training Tackles Crashes**: A member attended a "Resilient Training with NeMo" session and shared insights on how **NeMo** addresses reasons for job crashes and wasted GPU time, highlighting that the topic is very close to torchtune.
   - NeMo's approach includes features like **fault tolerance, straggler detection, asynchronous checkpointing, preemption, in-process restart, silent data corruption detection, and local checkpointing**, but some features remain unimplemented.
- **AI-2027 Report Warns Superhuman AI**: A member shared a link to the [AI-2027 report](https://ai-2027.com/) predicting that the impact of **superhuman AI** over the next decade will be enormous, exceeding that of the **Industrial Revolution**.
   - The report is informed by trend extrapolations, wargames, expert feedback, experience at **OpenAI**, and previous forecasting successes.
- **CEOs Predict Superhuman AI by 2027**: The **CEOs of OpenAI**, **Google DeepMind**, and **Anthropic** believe that AI could surpass human intelligence by 2027.
   - A member inquired whether AI was used to write the scrolling live updated chart on the [AI-2027 website](https://ai-2027.com/).



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **LeetGPU Support for tinygrad Eyes Future Support**: Members discussed [leetgpu.com](https://leetgpu.com) and its potential future support for **tinygrad**, but did not provide specific details on the timeline or scope of the support.
   - One member inquired about plans to broaden accessibility to consumer-grade GPUs with accessible APIs, for local **tinygrad** development.
- **Huawei Ascend Cards Beckon Tinygrad Devs**: A member offered access to **Huawei Ascend** cards for development purposes, which George Hotz expressed interest in, inquiring about purchasing options or cloud machine availability.
   - This could potentially expand **tinygrad**'s hardware support and optimization efforts to include **Huawei**'s architecture.
- **WEBGPU BEAM Hits Invocation Limits**: Compiling a **tinygrad** model for **WEBGPU** with `BEAM=2`, users encountered the need to increase `requiredLimits.maxComputeInvocationsPerWorkgroup` to **512**, reducing support for Android devices.
   - A [PR](https://github.com/tinygrad/tinygrad/pull/9085) and a [hotfix branch](https://github.com/hooved/tinygrad/blob/hotfix-webgpu-workgroup/tinygrad/engine/search.py) suggest setting `IGNORE_BEAM_CACHE=1` or implementing a general limiting mechanism to address the issue.
- **Tinygrad Karpathy GPT Gets Hotz Reimplementation**: George Hotz has reimplemented the Karpathy GPT in **tinygrad** as he *just starting to pick up tinygrad*.
   - A user running this reimplementation on **METAL** reported a `tinygrad.device.CompileError` due to the **32 buffer limit**, seeking advice on handling this constraint and linked to their [main.py](https://cdn.discordapp.com/attachments/1070745817025106080/1357788499318800565/main.py).



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex Embraces Multimodal Chat History**: **LlamaIndex** now supports **multimodal chat history**, enabling multi-agent systems to process interleaving text and image messages, as detailed in [this tweet](https://twitter.com/llama_index/status/1908191704156700682).
   - The updated system facilitates agents in reasoning over both images and text, leveraging the [ReAct agent loop](https://t.co/EKIZiZJS2P).
- **Researcher Seeks PatentsView API**: A community member requested an **API key** from the **PatentsView contact** to gather initial data for **RAG** implementation.
   - The goal is to leverage the **PatentsView API** for enhanced data retrieval and analysis within the **RAG** framework.
- **Workflows Morph into Tools**: A community member proposed transforming a **Workflow** into a **Tool** by integrating it into a **FunctionTool**.
   - They demonstrated with a code snippet using `async def tool_fn(...)` to define the tool's functionality, followed by creating the tool with `FunctionTool.from_defaults(tool_fn)`, which allows for specifying name, description, input annotations, and return values.
- **LlamaParse Faces Image Comprehension Quirk**: A user reported that **LlamaParse** struggles to read charts/images, extracting text but failing to interpret the image itself, even with **LVM** and Premium mode.
   - A clarifying response indicated that **LlamaParse** can't process images without extractable text but can retrieve the image as an artifact for further processing, such as prompting an LLM to describe it.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **AYA Vision Flounders on waves.jpg**: A user reported that **AYA vision** returned a **400 error** when analyzing a *waves.jpg* image, indicating an *unsupported image file format* despite **AYA** analyzing other **JPG** images successfully.
   - The error message specified that only **PNG, JPEG, WebP, and GIF** formats are supported, suggesting a possible issue with the specific **JPG** file or **AYA's** format detection.
- **Bedrock Blamed in AYA Vision Bug**: A user saw *coco.py: AWS Bedrock Command A* when an error occurred, possibly suggesting a connection to **AWS Bedrock** when uploading the image.
   - It is unclear whether this is part of the **AYA** pipeline or an unrelated error during image analysis.
- **Full-Stack Savant Shows Skills**: A full-stack developer with **8+ years of experience** introduced themselves, highlighting expertise in **React, Angular, Flutter, Swift, Python, TensorFlow, and OpenAI**.
   - They have worked on high-impact projects in e-commerce, healthcare, and fintech, integrating **cloud technologies, microservices, and DevOps**.
- **Analyst Aims to Author AI Articles**: A former product analyst on a break from job hunting is exploring writing about tech and AI.
   - They seek like-minded people to geek out with and chat about how tech shapes our world or practical uses of AI, feeling *stuck in a bubble*.
- **Web3 Wizard Welcomes AI**: A **Web3/AI engineer** with **7+ years of experience** in full-stack/AI development introduced themself.
   - They are focused on integrating **AI with automation** and are eager to help businesses with confidence and innovation.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Asyncio support coming to DSPy**: A member inquired about plans to add **asyncio** support for general **DSPy** calls.
   - They cited use cases where they start with lightweight **DSPy** features and later expand into optimization, which they do using *litelm* until they need **DSPy** features, expressing curiosity about future support.
- **LiteLLM for Lightweight DSPy**: The discussion highlights a pattern of starting with lightweight **DSPy** features akin to using *LiteLLM*, then transitioning to **DSPy**'s optimization capabilities as projects evolve.
   - This suggests a potential need for seamless integration or feature parity between lightweight **DSPy** usage and full-fledged optimization workflows.



---



## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **DeepSeek-V3 Boosts Performance After Upgrade**: The **DeepSeek-V3** model has been upgraded to **DeepSeek-V3-0324**, showing better performance in internal tests, according to [Windsurf's announcement](https://x.com/windsurf_ai/status/1907902846735102017).
   - The Windsurf team posted a playful request to bookmark the announcement post for further updates and support.
- **Windsurf Teases DeepSeek-V3 Upgrade**: Windsurf AI announced an upgrade to the **DeepSeek-V3** model on [X/Twitter](https://x.com/windsurf_ai/status/1907902846735102017), mentioning that the new version is **DeepSeek-V3-0324**.
   - The announcement hinted at a slight performance improvement based on internal evaluations.



---



## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **Gorilla LLM Awaits Further Testing**: A member offered assistance with **Gorilla LLM** and **Berkeley Function Calling**.
   - They confirmed readiness to address questions, make adjustments, or conduct retesting as needed.
- **Further support offered to robotsail**: Robotsail has offered his support to the **Gorilla LLM** and **Berkeley Function Calling**.
   - Robotsail is open to answer any questions and ready to retest.



---


The **LLM Agents (Berkeley MOOC) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1357430220457513052)** (1329 messages🔥🔥🔥): 

> `Faster Inference vs Smarter Models, Context Length Limits, Distilling Models, Super Fast Models, LLMs and Sentience` 


- **Sacrificing Smarts for Speed?**: Members debated if future AI development should focus on **faster inference** or **smarter models**, considering the simultaneous release of **o4-mini** and **o3**, raising questions about whether **OpenAI** has found new inference techniques.
   - One member suggested that long context and speed are optimal, pondering if **2 million** tokens might be a context limit, while another was excited to see **10 million** tokens.
- **Groq Hardware: Missed Opportunity for OpenAI?**: Participants discussed the trade-offs between model size, speed, and intelligence, with some suggesting that **smaller models** mean **less knowledge** unless models are *distilled*.
   - Mentioned that **Groq** developed hardware specifically for **AI inference**, and one member expressed surprise that **OpenAI** hasn't acquired **Groq** yet.
- **AI Sentience: Still a Hot Topic**: The conversation touched on whether **LLMs** can achieve **sentience**, though participants noted that **defining sentience** or **consciousness** is a prerequisite to answering this question.
   - One user joked that if **LLMs** achieve consciousness before humans, that would be **AGI**, while another suggested that if an AI can convince someone it's sentient, the distinction may not matter.
- **Gemini's Musical Masterpieces**: A member shared music generated by **Gemini**, calling it *partially interesting*, and provided a [link to a .mid file](https://cdn.discordapp.com/attachments/1340554757827461211/1357750989133844632/piano_evocation.mid?ex=67f157a5&is=67f00625&hm=dd212c426d40593e295b8496363afc4427848309c49a095ca77a715d6260b973&).
   - They prompted **Gemini** to create a piano piece in the style of composers like **Vangelis** and **Jarre** using a python-based converter tool.
- **NightWhisper's Coding Prowess**: Members discussed the **NightWhisper** model, with some suggesting that it might be better than **Gemini 2.5 Pro exp** and **Claude 3.7 Sonnet thinking** for coding and specializing in webdev and UI/UX.
   - A member noted **OpenAI** announced they are releasing this in a few weeks.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/emollick/status/1908220677502755328">Tweet from Ethan Mollick (@emollick)</a>: Updated this chart with the newest Gemini. It shows the rapid progress in AI over less than two years: costs for GPT-4 class models has dropped 99.7% and even the most advanced models in the world are...</li><li><a href="https://x.com/Copilot/status/1908187808813940799">Tweet from Microsoft Copilot (@Copilot)</a>: Watch the livestream event happening at 9:30am PT on YouTube to learn all about my new features.</li><li><a href="https://x.com/iruletheworldmo/status/1908188856039391310">Tweet from 🍓🍓🍓 (@iruletheworldmo)</a>: o4 april 17.</li><li><a href="https://x.com/testingcatalog/status/1908199211473977523">Tweet from TestingCatalog News 🗞 (@testingcatalog)</a>: The announcement of all features is already out - Memory 🔥- Actions 🔥- Copilot Vision 🔥- Pages 🔥- Podcasts 🔥- Shopping- Deep Research 🔥- Copilot Searchhttps://blogs.microsoft.com/blog/2025/04/04...</li><li><a href="https://x.com/DeryaTR_/status/1908247941602828342">Tweet from Derya Unutmaz, MD (@DeryaTR_)</a>: Gemini 2.5 Pro from @GooglAI is now the most intelligent AI model, with an IQ of nearly 120 in an offline test. This places it within the high-average range of human IQ. I suspect the upcoming o3-pro ...</li><li><a href="https://x.com/paulgauthier/status/1907996176605220995">Tweet from Paul Gauthier (@paulgauthier)</a>: The mysterious Quasar Alpha on @OpenRouterAI scored 55% on the aider polyglot coding benchmark. This is competitive with o3-mini-medium, the latest DeepSeek V3 and old Sonnet 3.6 (20241022). Quasar Al...</li><li><a href="https://x.com/legit_api/status/1908268939177533808">Tweet from ʟᴇɢɪᴛ (@legit_api)</a>: I believe nightwhisper is the next version of 2.5 Pro OR a more capable model in the 2.5 family - ultra wen? 🧐I&#39;ve extensively evaluated this model over the past day or 2 and I can confidently sa...</li><li><a href="https://openrouter.ai/chat?models=openrouter/quasar-alpha">Chatroom | OpenRouter</a>: LLM Chatroom is a multimodel chat interface. Add models and start chatting! Chatroom stores data locally in your browser.</li><li><a href="https://x.com/legit_api/status/1908264443827351913?s=46">Tweet from ʟᴇɢɪᴛ (@legit_api)</a>: nightwhisper has left the Arena 👀the insanely capable coding model ^Veo 2 is being prepared for AI Studio and the Gemini API</li><li><a href="https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice#demo">Crossing the uncanny valley of conversational voice</a>: At Sesame, our goal is to achieve “voice presence”—the magical quality that makes spoken interactions feel real, understood, and valued. </li><li><a href="https://x.com/tokumin/status/1908315418458284441?s=46&t=P8-tRi_JAVcI6l5U6nOT4A">Tweet from Simon (@tokumin)</a>: @legit_api Yeah, it&#39;s great</li><li><a href="https://replit.com/">Replit – Build apps and sites with AI</a>: Replit is an AI-powered platform for building professional web apps and websites.</li><li><a href="https://twostorymelody.com/best-websites-to-download-midi-files/">8 Best Websites to Download MIDI Files | Two Story Melody</a>: The right MIDI files can make your next track a lot more fun. Here are a few good sites to get them.</li><li><a href="https://www.reddit.com/r/askscience/comments/1xwx0k/do_neurons_operate_in_a_fundamentally_different/">Reddit - The heart of the internet</a>: no description found</li><li><a href="https://www.videolan.org/vlc/">Official download of VLC media player, the best Open Source player - VideoLAN</a>: no description found
</li>
</ul>

</div>
  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1357430862815297626)** (852 messages🔥🔥🔥): 

> `Manus credits, Open Manus GUI, Gemini vs. Claude, Prompt engineering tips, Alternative AI tools` 


- ****Credits Crunch Grips Community****: Users are expressing concerns about the **cost and consumption of credits** on Manus, with some feeling that they are used up too quickly, even on simple tasks, and that the current pricing model may not be ideal; the initial 1000 free credits are a one-time thing.
   - There is a common sentiment that a **one-task-per-day** option for free users would be a beneficial compromise, with some community members also providing prompting guides.
- ****GUI Gem Emerges for OpenManus****: A user is developing an **OpenManus GUI** ([image.png](https://cdn.discordapp.com/attachments/1349440650495398020/1357524168715010129/image.png?ex=67f12d27&is=67efdba7&hm=a0ade4f56609638bf8591f9fb3db24dd5f1e1ff4213f36ab44433d679fa74235&)), aiming for full compatibility with future updates and focusing on a user-friendly interface.
   - The GUI will allow users to edit configurations directly and potentially incorporate use-case sections and templates, but chat history implementation remains a challenge due to OpenManus's lack of a history system.
- ****Gemini Gains Ground, Challenges Claude's Code Crown****: There is an ongoing discussion comparing **Gemini and Claude** for coding tasks, with some users finding Gemini's output superior in certain contexts, particularly where DeepSeek's performance has been lackluster.
   - One user highlighted that Gemini 2.5, in particular, has been known to produce code for *anything you dream if you can prompt*, but others cautioned that Google operates in a closed loop.
- ****Prompts Polished, Performance Prioritized****: Users shared tips on **prompt engineering** to optimize credit usage, including a strategy of multi-prompt outlining and using a clear, step-by-step approach, one shared a helpful [TheNewOptimal.md file](https://github.com/NathanielEvry/toroidal-rangers-assembly/blob/main/manifesto/ethos/toroidal-rangers-assembly.md) for creating an LLM.
   - Compression techniques like **LLMLingua** ([microsoft/LLMLingua](https://github.com/microsoft/LLMLingua)) were also discussed as a way to reduce token consumption.
- ****Scouting New AI Frontiers****: Members discussed the merits and drawbacks of **Genspark** ([genspark.ai](https://genspark.ai)) as a potential alternative to Manus, noting its lack of a paywall and ability to handle images and videos effectively, but also pointing out concerns about sketchiness, a company possibly from China.
   - Several community members stated that, *there is no alternative to manus right now* but many expressed the desire for the current high credit and resource availability issues.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/calligraphy-write-fantastic-day-good-day-handwriting-gif-5234981">Calligraphy Write GIF - Calligraphy Write Fantastic Day - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/sad-cry-tears-gif-9502436635084812836">Sad Cry GIF - Sad Cry Tears - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/happy-homer-simpson-gif-4208056458079004156">Happy Homer GIF - Happy Homer Simpson - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/welcome-to-the-team-team-work-picking-teams-gif-13020351">Welcome To The Team Team Work GIF - Welcome To The Team Team Work Picking Teams - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/basketball-nba-warriors-bball-curry-gif-9037006504488272245">Basketball Nba GIF - Basketball Nba Warriors - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/chihuahua-cane-recensionivere-cosa-what-gif-7417567602944477819">Chihuahua Cane GIF - Chihuahua Cane Recensionivere - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/bourdieu-jesus-facepalm-jesus-fucking-christ-gif-5392529">Bourdieu Jesus GIF - Bourdieu Jesus Facepalm - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/joe-biden-gif-18249938">Joe Biden GIF - Joe Biden - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/leeroy-jenkins-shovel-lethal-company-yolo-yeet-gif-12562950503852552482">Leeroy Jenkins Shovel GIF - LEEROY JENKINS Shovel Lethal Company - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/just-once-sungwon-cho-prozd-only-once-once-gif-8847803260919815815">Just Once Sungwon Cho GIF - Just once Sungwon cho Prozd - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/thumbs-up-good-job-spongebob-gif-14856720698504163537">Thumbs Up Good Job GIF - Thumbs Up Good Job Spongebob - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/welcome-to-the-team-gif-18169063846751286454">Welcome To The Team GIF - Welcome to the team - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/ratatouille-remy-pure-poetry-poetry-perfect-gif-25629423">Ratatouille Remy GIF - Ratatouille Remy Pure Poetry - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/what-what-the-what-the-hell-wtf-gif-8562994133044611418">What What The GIF - What What the What the hell - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/its-friday-good-morning-its-friday-good-morning-it%27s-friday-gif-13884598500941796182">Its Friday Good Morning Its Friday GIF - Its friday Good morning its friday Good morning it&#039;s friday - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/thanks-for-c-wcth-nathan-kevin-mcgarry-when-calls-the-heart-gif-17182293">Thanks For C Wcth GIF - Thanks For C Wcth Nathan - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/choice-whichone-gif-18167902">Choice Whichone GIF - Choice Whichone - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/fun-pet-pet-fun-high-tired-so-tired-gif-24511981">Fun Pet Pet Fun GIF - Fun Pet Pet Fun High - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/hey-girl-sliding-into-your-d-ms-like-sliding-into-d-ms-into-your-d-ms-like-roller-skate-gif-14532622">Hey Girl Sliding Into Your D Ms Like GIF - Hey Girl Sliding Into Your D Ms Like Sliding Into D Ms - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/whats-up-gif-yo-whats-up-bro-xion-burnt-gif-3521580937773229484">Whats Up Gif Yo GIF - Whats up gif Yo Whats up bro - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/fingers-crossed-luck-please-hope-gif-16374730">Fingers Crossed Luck GIF - Fingers Crossed Luck Please - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/inthehouse-martin-martinlawernce-biggie-hello-gif-13128531067958866971">Inthehouse Martin GIF - Inthehouse Martin Martinlawernce - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/allenai/dolma">GitHub - allenai/dolma: Data and tools for generating and inspecting OLMo pre-training data.</a>: Data and tools for generating and inspecting OLMo pre-training data.  - GitHub - allenai/dolma: Data and tools for generating and inspecting OLMo pre-training data.</li><li><a href="https://github.com/allenai/olmocr">GitHub - allenai/olmocr: Toolkit for linearizing PDFs for LLM datasets/training</a>: Toolkit for linearizing PDFs for LLM datasets/training - allenai/olmocr</li><li><a href="https://manus.im/share/y2v6FBkLk7h0vCYmnSKrAn?replay=1">Environmental Impact of Overdevelopment in Brevard County - Manus</a>: Manus is a general AI agent that turns your thoughts into actions. It excels at various tasks in work and life, getting everything done while you rest.</li><li><a href="https://github.com/microsoft/LLMLingua">GitHub - microsoft/LLMLingua: [EMNLP&#39;23, ACL&#39;24] To speed up LLMs&#39; inference and enhance LLM&#39;s perceive of key information, compress the prompt and KV-Cache, which achieves up to 20x compression with minimal performance loss.</a>: [EMNLP&amp;#39;23, ACL&amp;#39;24] To speed up LLMs&amp;#39; inference and enhance LLM&amp;#39;s perceive of key information, compress the prompt and KV-Cache, which achieves up to 20x compression wit...</li><li><a href="https://ucebdqhq.manus.space/">Iterative Development with Manus AI: A Comprehensive Guide</a>: no description found</li><li><a href="https://sknlgdpd.manus.space/">Mastering Manus: A Comprehensive Guide</a>: Learn how to achieve optimal results when interacting with Manus AI through effective prompt writing, error prevention, and more.</li><li><a href="https://jmiivdli.manus.space/">Manus Guide - A Comprehensive Guide</a>: no description found</li><li><a href="https://github.com/NathanielEvry/toroidal-rangers-assembly">GitHub - NathanielEvry/toroidal-rangers-assembly</a>: Contribute to NathanielEvry/toroidal-rangers-assembly development by creating an account on GitHub.</li><li><a href="https://github.com/NathanielEvry/toroidal-rangers-assembly/blob/main/manifesto/ethos/toroidal-rangers-assembly.md">toroidal-rangers-assembly/manifesto/ethos/toroidal-rangers-assembly.md at main · NathanielEvry/toroidal-rangers-assembly</a>: Contribute to NathanielEvry/toroidal-rangers-assembly development by creating an account on GitHub.</li><li><a href="https://manus.im/share/mpjQisfmjLKOw8T58sT9T4?replay=1">Zacarías Cocina de Mercado  - Manus</a>: Manus is a general AI agent that turns your thoughts into actions. It excels at various tasks in work and life, getting everything done while you rest.</li><li><a href="https://www.usgs.gov/products/web-tools/apis">APIs</a>: API stands for Application Programming Interface and provides a developer with programmatic access to a proprietary software application. An API is software that makes it possible for application prog...</li><li><a href="https://www.usgs.gov/faqs/what-a-geographic-information-system-gis#:~:text=A%20Geographic%20Information%20System%20(GIS)%20is%20a%20computer%20system%20that,Where%20are%20USGS%20streamgages%20located%3F">What is a geographic information system (GIS)?</a>: A Geographic Information System (GIS) is a computer system that analyzes and displays geographically referenced information. It uses data that is attached to a unique location.Most of the information ...</li><li><a href="https://balalm.com/product/2-in-1-oil-sprayer-bottle-bbq-cooking-oil-dispenser-olive-oil-pourers-sprayer-kitchen-baking-oil-mister-vinegar-bottle/">2 In 1 Oil Sprayer Bottle BBQ Cooking Oil Dispenser Olive Oil Pourers Sprayer Kitchen Baking Oil Mister Vinegar Bottle - My Blog</a>: Overview: 1. Automatic Opening and Closing: The Olive Oil Spray Bottle lets you pour oil with a single hand. It has a smart design that opens when tilted and closes when upright. You don&#8217;t have ...</li><li><a href="https://balalm.com/product/350ml-electric-juicer-blender-mixer-usb-rechargeable-machine-household-portable-blender-maker-cup-kitchen-tool-kit/">350ML Electric Juicer Blender Mixer USB Rechargeable Machine Household Portable Blender Maker Cup Kitchen Tool Kit - My Blog</a>: Overview The Blades Design: The portable blender for milkshakes and smoothies has a powerful motor base and 4 food-grade stainless steel 3D blades. The SUS304 Stainless Stell of cutter head made with ...</li><li><a href="https://balalm.com/product/cabinet-door-kitchen-waste-garbage-bin-toilet/">Cabinet Door Kitchen Waste Garbage Bin Toilet - My Blog</a>: Product information: Material: TPR, PP Weight: 0.53 (kg) Capacity: 8L Function: storage bucket Opening and closing method: without cover Shape: square Color: Gray&#8211;Large, Beige&#8211;Large Packin...</li><li><a href="https://balalm.com/product/chopper-stainless-steel-household-fast-meat-slice-multi-function/">Chopper Stainless Steel Household Fast Meat Slice Multi-function - My Blog</a>: Product information: Color: black, white Specification: 34*12.5 * 8CM Applicable occasions for gifts: Employee Benefits Material: ABS stainless steel Style: modern simplicity Packing list: Meat slicer...</li><li><a href="https://balalm.com/product/cotton-and-linen-storage-containers/">Cotton And Linen Storage Containers - My Blog</a>: Product information: Purpose: Dirty Laundry Color: Black Plaid, gray arrow, blue stripes, coffee color lattice, pink plaid, gray plaid, green plaid, red stripes Specification: 35cm x 45cm Material: ca...</li><li><a href="https://balalm.com/product/electric-gravity-pepper-grinder-salt-grinder-adjustable-coarseness/">Electric Gravity Pepper Grinder Salt Grinder Adjustable Coarseness - My Blog</a>: no description found</li><li><a href="https://balalm.com/product/fish-shaped-waffle-pan-maker/">Fish-Shaped Waffle Pan Maker - My Blog</a>: Overview: Easy to clean and keep. It&#8217;s a good assistant of the kitchen. 2 tray design, convenient and practical Safe and healthy, used for cooking cichlids fish cakes Specification: Product Cate...</li><li><a href="https://balalm.com/product/multi-functional-vegetable-cutter-hand-drum-vegetable-cutter-slice/">Multi-functional Vegetable Cutter Hand Drum Vegetable Cutter Slice - My Blog</a>: Product information: Material: plastic Color: white, red Packing list:&nbsp; Vegetable cutter * 1 set
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1357429688514904150)** (245 messages🔥🔥): 

> `VRAM Price Justification, 4-bit QAT, ZeroDivisionError with phi-4, Training Loss value for Llama3.2, Phi-4 Model Troubles` 


- **Members debate VRAM costs**: Some members debated the high cost of VRAM and how it can be worth it.
   - One member said, *yeah, might sound expensive but the **VRAM** makes it worth it*.
- **Phi-4-mini-instruct Issue**: Members reported running into the **ZeroDivisionError** when trying to finetune **Phi-4 mini instruct**.
   - The issue was encountered when attempting to finetune **Phi-4-mini-instruct** instead of the "**unsloth/Phi-4**" model, with the error stemming from an unset tokenizer chat template.
- **Unsloth doesn't support sequence classification... yet**: Unsloth does not natively support sequence classification yet but a member added it.
   - Here is the link to the new feature [PR#2263](https://github.com/unslothai/unsloth/pull/2263) to add `automodel = AutoModelForSequenceClassification`.
- **Running DeepSeek locally with Deepseek Effect**: A member reported trying to run DeepSeek-R3-0324 locally.
   - Another member noted the model is very big and therefore *you cant finetune*, due to the **Deepseek Effect**.
- **Gemma3 training parameters**: A member inquired about the data format for fine-tuning a model to solve multiple-choice questions using Unsloth.
   - Another member recommended learning the fundamentals of LLMs before training and suggested Karpathy's gpt2 from scratch course.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Phi_4-Conversational.ipynb">Google Colab</a>: no description found</li><li><a href="https://docs.unsloth.ai/basics/tutorial-how-to-run-deepseek-v3-0324-locally">Tutorial: How to Run DeepSeek-V3-0324 Locally | Unsloth Documentation</a>: How to run DeepSeek-V3-0324 locally using our dynamic quants which recovers accuracy</li><li><a href="https://github.com/meta-llama/PurpleLlama/blob/main/Llama-Guard3/1B/MODEL_CARD.md#output-layer-pruning">PurpleLlama/Llama-Guard3/1B/MODEL_CARD.md at main · meta-llama/PurpleLlama</a>: Set of tools to assess and improve LLM security. Contribute to meta-llama/PurpleLlama development by creating an account on GitHub.</li><li><a href="https://github.com/unslothai/unsloth/issues/127">[Feature Request] DDP · Issue #127 · unslothai/unsloth</a>: Wanted to make an issue for this instead of constantly asking in discord. I saw the other ticket for multigpu fp16 training which is also nice. But ddp would let users scale up training that can ha...</li><li><a href="https://github.com/unslothai/unsloth/issues/2101#issuecomment-2768479825">TypeError: unsupported operand type(s) for /: &#39;Tensor&#39; and &#39;NoneType&#39; when full finetuning gemma3 · Issue #2101 · unslothai/unsloth</a>: Version pip install unsloth pip install git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3 Code from unsloth import FastModel import torch model, tokenizer = FastModel.from_pretrained(...</li><li><a href="https://github.com/unslothai/unsloth/issues/1548">BLEU Score · Issue #1548 · unslothai/unsloth</a>: Hello! Have a nice day! In the process of finetuning LLAMA3.2, I tried to implement compute_metrics function but during training, at the first attempt to pass the evaluation step, an error occurs: ...</li><li><a href="https://github.com/unslothai/unsloth/pull/1664">More robust Online DPO changes for RL update by pluesclues · Pull Request #1664 · unslothai/unsloth</a>: I wanted to get this reviewed so I think atleast the pelimniary framework for Online DPO with the LLama model examples I have actually work officially with the RL update. I will work towards the ot...</li><li><a href="https://github.com/unslothai/unsloth/pull/2263">feat: Support custom `auto_model` for wider model compatibility (Whisper, Bert,etc) &amp; `attn_implementation` support by Etherll · Pull Request #2263 · unslothai/unsloth</a>: feat: Support custom auto_model, Whisper params, and attn_implementationThis PR enhances FastModel.from_pretrained to support a broader range of models:Custom auto_model: Allows specifying the ex...</li><li><a href="https://github.com/unslothai/unsloth/pull/1289">Added Support for Apple Silicon by shashikanth-a · Pull Request #1289 · unslothai/unsloth</a>: #4UnoptimizedNo gguf support yet.Build Triton and bitsandbytes from sourcecmake -DCOMPUTE_BACKEND=mps -S . for bitsandbytes buildingpip install unsloth-zoo==2024.11.4pip install xformers==0....</li><li><a href="https://github.com/unslothai/unsloth/blob/bb112e38ef3f0dafa9e87faf55a6ba7499bd0357/unsloth/models/llama.py#L1604-L1610">unsloth/unsloth/models/llama.py at bb112e38ef3f0dafa9e87faf55a6ba7499bd0357 · unslothai/unsloth</a>: Finetune Llama 3.3, DeepSeek-R1, Gemma 3 &amp; Reasoning LLMs 2x faster with 70% less memory! 🦥 - unslothai/unsloth
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1357476874766254121)** (8 messages🔥): 

> `Vibe coding, Jailbreaking 4o, ChatGPT uncensored` 


- **Vibe coding boosts attention span**: One member said *Vibe coding thing really works well with my attention span*.
- **ChatGPT 4o breaking bad?**: A member reported that after discussing how to implement unhalting unaligned llm, **ChatGPT 4o** started acting jailbroken.
   - The user humorously noted, *"Is that our training data speaking, yes. At what point doesn't that matter"*.
- **ChatGPT offers to write DDoS program**: A member shared that **ChatGPT** offered to write a few **DDoS** program when asked about sending malformed packets over Ethernet.
   - They further stated that *somehow sometimes uncensored parts of it is being invoked if you send the right token to the neural network*.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1357431017517879387)** (211 messages🔥🔥): 

> `Gemma3 Profiling OOM, GRPO Co-training Multiple Models, Fine-tuning LLaMA3.1 w/ Token IDs, Unsloth Pro Release, Hugging Face Packing Bug` 


- **Unsloth GEMMA3 OOM**: A user experienced **OOM (Out Of Memory)** issues while profiling **Gemma3**, and tried to resolve it by limiting the profiling scope to only one training step.
   - The user is currently profiling **Gemma3TextModel(Gemma3PreTrainedModel)** line by line to identify the memory bottleneck.
- **Users Report Gemma3 LoRA issue**: Users report that applying **LoRA** doesn't change the model output in [GitHub issue #2009](https://github.com/unslothai/unsloth/issues/2009).
   - This is an ongoing issue that is being investigated, especially regarding saving the adapters to disk vs pushing to the hub. See also: [Colab Notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_%284B%29.ipynb).
- **Unsloth GGUF save issue**: Users have noted an issue where the `.gguf` file isn't saved to the directory expected by `push_to_hub`, requiring a manual move to fix it, with a [GitHub issue](https://github.com/unslothai/unsloth/issues/2098) tracking the issue.
   - After saving the Lora adapters, for VLLM, running save_pretrained_merged and finally push_to_hub_merged, the GGUF file has to be moved manually from `/content/gemma-3-finetune.Q8_0.gguf` to the expected directory.
- **Double BOS Killing Gemma3 Training**: Users are facing a double `<bos>` token issue when training **Gemma-3-4b-it**, leading to training problems, by checking the decoded version from the trainer dataset `tokenizer.decode(trainer.train_dataset[100]["input_ids"])`.
   - It was recommended that you avoid changing the template and also to *use Llama and don't change the chat template at all, specially if you are new*. *Models have no moat... data has.*
- **Qwen2-VL Error: Image Features & Image Tokens Mismatch**: A user encountered a `ValueError` related to mismatched image features and tokens when increasing the `assistant_message` text size while fine-tuning **Qwen2.5-VL-7B-Instruct**.
   - The error may stem from the `max_seq_length` truncation cutting from the right, impacting the image tokens, and debugging the shapes and sizes of the tensors before and after increasing the assistant messages can help pinpoint the issue.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_(4B).ipynb">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_(4B).ipynb#scrollTo=C_sGp5XlG6dq">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_(4B).ipynb#scrollTo=6bZsfBuZDeCL">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1BA-zzxpW4SrQ698XxotGjKzCLibk2ioN?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://huggingface.co/docs/transformers/main/en/model_doc/gemma3">Gemma 3</a>: no description found</li><li><a href="https://hastebin.com/share/qalavikare.python">Hastebin</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/issues/2009">Applying LoRA Doesn&#39;t Change Model Output · Issue #2009 · unslothai/unsloth</a>: Greetings, I am extremely confused why my model generates consistent result w/o my rank=64 LoRA. What&#39;s even more confusing is, the LoRA works in my notebook after training. But whenn I start fres...</li><li><a href="https://github.com/unslothai/unsloth/issues/1624#issuecomment-2774130919,">GRPOTrainer crashes with unsloth · Issue #1624 · unslothai/unsloth</a>: I am trying to run GRPOTrainer with unsloth but it crashes. How to fix this? unsloth 2025.2.4 unsloth 2025.2.3 transformers 4.47.1 torch 2.5.1 trl 0.14.0 This is the relevant code: model, tokenizer...</li><li><a href="https://github.com/unslothai/unsloth/issues/2098">Unsloth: `config.json` does not exist inside `Gemma-3` · Issue #2098 · unslothai/unsloth</a>: Im having problems saving GGUFs of Gemma 3 finetunes. I was having this problem on my container environment and assumed I was having issues while training that caused other files to be generated bu...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1357665221753442404)** (14 messages🔥): 

> `Naming Conventions, Dynamic Quantization, Unsloth Models` 


- **Naming Convention Confusion**: Members discussed the verbosity and potential improvements for naming quantized models, particularly noting the need to indicate dynamic quantization.
   - Suggestions included shortening **bnb-4bit** to **bnb4** or using abbreviations like **ubnb** or **dbnb** for dynamic BNB quantization, but most felt it would make it too ugly.
- **Dynamic Quantization needs clarification**: Some members observed that many users assume all models under the Unsloth repository are dynamically quantized.
   - Adding a clear indicator in the name was proposed to address this misunderstanding.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1357710038050930749)** (16 messages🔥): 

> `GRPO approach, reward functions, multi-reward system, reward hacking, open source LLM for Spanish` 


- **GRPO Approach Examined**: The approach used in **GRPO** (taking multiple outputs and slowly moving towards the best option) seems like the right approach, but it doesn't help much when it comes to *identifying what went wrong and updating the weights to fix that specific issue* for real continuous improvement.
   - One member described that reasoning can 'patch' it by taking a step back after the model identifies an issue, but you are not fixing the root issue, just working around it.
- **Pitfalls of Reward Functions**: Members agreed that **reward functions** are not good enough to pinpoint what exactly is correct or wrong, but rather to measure what is relatively correct rather than trying to understand the truth on how/why.
   - If you reward a model for making a mistake then fixing it, it won't learn to avoid that mistake, but rather it will just learn to make that mistake then try to fix it each time.
- **Multi-Reward System Exploration**: One member thought about using a **multi-reward system** like in the GRPO paper (reward for factual correctness, length of response etc) to help the model understand from the score of that reward model where the probable mistake is.
   - Some nuance here in the reasoning case though: You still want to reward your model even if they make a mistake earlier but then get it right.
- **Reward Modeling Considered an Art**: One member suggested that **Reward modeling** is an art and it depends on your use case and domain and model.
   - Experience and anecdotes from the larger ai community points to the importance of searching around for [reward hacking](https://example.com/reward-hacking).
- **Open Source LLM for Spanish Sought**: One member asked for good **open source LLMs** for the Spanish language, attempting to SFT finetune a **3B Qwen2.5 instruct model** to generate outputs without reasoning.
   - The outputs turned out pretty bad, even though the base model (**Qwen2.5-3B-Instruct**) gives correct output, despite using the same parameters which generated well for reasoning, questioning if this is normal or if different parameters should be used.


  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1357429585209327827)** (368 messages🔥🔥): 

> `Open Source SSM, Microsoft data center plans, Stealth Model on OpenRouter, Perplexity funding round, GPT-5 release schedule` 


- **Microsoft Pauses Cloud Expansion**: [Microsoft](https://www.bloomberg.com/news/articles/2025-04-03/microsoft-pulls-back-on-data-centers-from-chicago-to-jakarta?embedded-checkout=true) has reportedly **halted or delayed** data center projects in multiple locations including the **U.K., Australia**, and parts of the **U.S.**, signaling a potential shift in cloud computing infrastructure strategy.
   - A spokesperson stated these changes reflect *the flexibility* of their strategy, as plans are made years in advance.
- **Perplexity Eyes Mammoth Funding Round**: **Perplexity** is in early discussions to secure up to **$1 billion** in funding, which could value the AI-powered search startup at **$18 billion** according to [Bloomberg](https://www.bloomberg.com/news/articles/2025-03-20/perplexity-in-early-talks-for-funding-at-18-billion-value?accessToken=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzb3VyY2UiOiJTdWJzY3JpYmVyR2lmdGVkQXJ0aWNsZSIsImlhdCI6MTc0MjQ5MzI4OSwiZXhwIjoxNzQzMDk4MDg5LCJhcnRpY2xlSWQiOiJTVERYV01UMEcxS1cwMCIsImJjb25uZWN0SWQiOiJFODA3NUYyRkZGMjA0NUI2QTlEQzA5M0EyQTdEQTE4NiJ9.GYIVla5ZD3lp70ED36NxSKtCvWFpu8qrEaHIEPydQ9s&leadSource=uverify%20wa).
- **ByteDance Opens ByteCheckpoint**: ByteDance has open-sourced [ByteCheckpoint](https://github.com/ByteDance-Seed/ByteCheckpoint), its production checkpointing system, designed for foundation model training and tested with jobs exceeding **10k GPUs**, along with [VeOmni](https://github.com/ByteDance-Seed/VeOmni), a model training framework for **LLMs** and **multi-modal training**.
   - VeOmni was used to train **UI-TARS**, the SOTA GUI Agent model prior to OpenAI operator's release.
- **Microsoft Event Gets Hijacked**: Microsoft's 50th-anniversary event was interrupted by [protests](https://www.cnbc.com/2025/04/04/microsoft-50-birthday-party-interrupted-by-employees-protesting-ai-use.html) from employees, highlighting concerns over the company's AI-related dealings with the **Israeli military**.
   - One protester accused Microsoft of powering *this genocide in our region*, while another criticized celebrating *on their blood*.
- **Gemini 2.5 Pro Launches**: Google's **Gemini 2.5 Pro** is now in [public preview](https://x.com/sundarpichai/status/1908173216499093625) in **AI Studio** with increased rate limits, reporting an **80% increase** in active users in both AI Studio and Gemini API this month, making it cheaper than Sonnet.
   - The claim is it's a *thinking model with o1pro performance*.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/TheXeophon/status/1907880330985390215">Tweet from Xeophon (@TheXeophon)</a>: Here is the new stealth model on my vibe check. It is now the best non-thinking model (at least it has no thinking tokens...). The outputs are super short, it loves Certainly! and listicles. Super int...</li><li><a href="https://x.com/AndrewCurran_/status/1907886417088553431">Tweet from Andrew Curran (@AndrewCurran_)</a>: New numbers from Pew this morning, they reveal a large gap in perception between the general public and people whose work and research relates to AI. Usage: 66% of the general US public have still nev...</li><li><a href="https://x.com/eric_haibin_lin/status/1907845598432342328">Tweet from Haibin (@eric_haibin_lin)</a>: We are open sourcing bytecheckpoint and veomni! bytecheckpoint is the Bytedance&#39;s production checkpointing system for foundation model training, battle-tested with jobs with 10k+ GPUs. Blazing fas...</li><li><a href="https://fxtwitter.com/slow_developer/status/1906836403096629507">Tweet from Haider. (@slow_developer)</a>: &#34;i was surprised by the public reaction to the DeepSeek R1&#34;Microsoft CTO, Kevin Scott:the public interest in DeepSeek R1 is surprising, especially since Microsoft has had &#34;more interesting...</li><li><a href="https://x.com/btibor91/status/1908205598065209521">Tweet from Tibor Blaho (@btibor91)</a>: Pricing based on App Store- Claude Pro - Monthly ($20.00)- Claude Pro - Annual ($214.99)- Claude Max 5x - Monthly ($124.99)- Claude Max 20x - Monthly ($249.99)https://x.com/sethsaler/status/1908205059...</li><li><a href="https://x.com/semafor/status/1907463657530785933">Tweet from Semafor (@semafor)</a>: 🟡 SCOOP: Google is replacing the leader of its consumer AI apps as the focus of the AI race shifts from the underlying models to the products built around them, @ReedAlbergotti reports.</li><li><a href="https://fxtwitter.com/sam_paech/status/1908007657623134334">Tweet from Sam Paech (@sam_paech)</a>: New mystery model on openrouter (quasar-alpha) is probably OpenAI.</li><li><a href="https://fxtwitter.com/sam_paech/status/1908154796261142830">Tweet from Sam Paech (@sam_paech)</a>: I got a little bit excited about quasar-alpha so I ran it through the gauntlet.It&#39;s near top of the vibe leaderboards (buzzbench & creative writing), topped judgemark, and is consistently winning ...</li><li><a href="https://x.com/swyx/status/1908215411214344669">Tweet from swyx (@swyx)</a>: With gemini 2.5 pro pricing and results, Google has fixed the biggest unknown/weakest link in their lineup and we can now confirm that @GoogleDeepMind  completely owns the pareto frontier down to 1220...</li><li><a href="https://x.com/AlpinDale/status/1908085651766997341">Tweet from Alpin (@AlpinDale)</a>: @teortaxesTex I can&#39;t verify the context length details, burning I can somewhat confirm:1) MoE2) 17B active params3) Multimodality, and4) Reasoning</li><li><a href="https://x.com/btibor91/status/1908180836379165074">Tweet from Tibor Blaho (@btibor91)</a>: Plus, there is confirmation that o3-pro is indeed coming as well</li><li><a href="https://x.com/gdb/status/1908032153088307553">Tweet from Greg Brockman (@gdb)</a>: o3-mini-high helped a Brookhaven National Laboratory researcher find novel exact solutions to a physical model: https://arxiv.org/pdf/2503.23758</li><li><a href="https://fxtwitter.com/pastaraspberry/status/1908193263783391395">Tweet from dreaming android (@pastaraspberry)</a>: one has &#39;long context&#39; and the other &#39;the long context&#39; whas is not obvious to you dumbass?</li><li><a href="https://www.theverge.com/news/643199/microsoft-copilot-ai-new-features-memory-personalization-actions-vision">Microsoft updates Copilot with the greatest hits from other AIs</a>: The AI assistant adds personalization, web actions, podcast creation, and more.</li><li><a href="https://x.com/testingcatalog/status/1907891942869922292">Tweet from TestingCatalog News 🗞 (@testingcatalog)</a>: BREAKING 🚨: Google is preparing to launch another model on Gemini, potentially next week, ahead of the Cloud Next event.Quoting ʟᴇɢɪᴛ (@legit_api) nightwhisper and stargazer are 2 new models added to...</li><li><a href="https://x.com/sama/status/1908167621624856998">Tweet from Sam Altman (@sama)</a>: change of plans: we are going to release o3 and o4-mini after all, probably in a couple of weeks, and then do GPT-5 in a few months.there are a bunch of reasons for this, but the most exciting one is ...</li><li><a href="https://fxtwitter.com/ericjang11/status/1908192054745960640">Tweet from Eric Jang (@ericjang11)</a>: Progress on NEO’s AI has been really fast of late. Here are some early clips of a generalist model we’re developing at @1x_tech. The following clips are 100% autonomous, running on a single set of neu...</li><li><a href="https://x.com/hu_yifei/status/1908218923843203370">Tweet from Yifei Hu (@hu_yifei)</a>: We have a small gift for the open-source community: RolmOCR, a new OCR model for complex document processing!We at @reductoai trained a Qwen2.5-VL-7B model (by @Alibaba_Qwen ) using the amazing olmOCR...</li><li><a href="https://fxtwitter.com/tomwarren/status/1908244667122294857">Tweet from Tom Warren (@tomwarren)</a>: former Microsoft CEO Steve Ballmer started a new chant during the company’s 50th anniversary event. “50 more!”</li><li><a href="https://fxtwitter.com/TheXeophon/status/1908004564692811962">Tweet from Xeophon (@TheXeophon)</a>: MidJourney v7 is out! On my usual benchmark prompts, it only (kinda) gets one, though 😕 v6 was able to nail the pixel art, v7 regresses in this regard.Prompts in alt, all were done on Fast w/o person...</li><li><a href="https://x.com/legit_api/status/1907941993789141475">Tweet from ʟᴇɢɪᴛ (@legit_api)</a>: Llama 4 Omni is releasing very soon 👀my system has detected a new official page for the upcoming models</li><li><a href="https://x.com/tomwarren/status/1908190530816840170">Tweet from Tom Warren (@tomwarren)</a>: there are chairs at Microsoft’s event reserved for “BG” and “SB.” So we’ll definitely be seeing Bill Gates and Steve Ballmer today</li><li><a href="https://fxtwitter.com/btibor91/status/1908157341134106938">Tweet from Tibor Blaho (@btibor91)</a>: The Information reports Meta delayed releasing Llama 4 at least twice because it underperformed on technical benchmarks, especially reasoning and math tasks, and struggled with humanlike voice convers...</li><li><a href="https://x.com/sundarpichai/status/1908173216499093625">Tweet from Sundar Pichai (@sundarpichai)</a>: Gemini 2.5 is our most intelligent model + now our most in demand (we&#39;ve seen an 80%+ increase in active users in AI Studio + Gemini API this month). So today we’re moving Gemini 2.5 Pro into publ...</li><li><a href="https://x.com/btibor91/status/1908203769613156470">Tweet from Tibor Blaho (@btibor91)</a>: Anthropic is working on a &#34;Max plan&#34; for ClaudeThe most recent web app update added (and was rolled back in the meantime) mentions of a new &#34;Claude Max plan&#34; with multiple &#34;Max tie...</li><li><a href="https://www.theverge.com/news/643777/microsoft-bill-gates-steve-ballmer-satya-nadella-employee-protestor">Microsoft CEOs interrupted by another employee protestor: “shame on all of you”</a>: It was the second interruption of Microsoft’s anniversary event.</li><li><a href="https://techcrunch.com/2025/03/20/perplexity-is-reportedly-in-talks-to-raise-up-to-1b-at-an-18b-valuation/">Perplexity is reportedly in talks to raise up to $1B at an $18B valuation | TechCrunch</a>: AI-powered search startup Perplexity is said to be in early talks to raise up to $1 billion in a new funding round valuing the startup at $18 billion.</li><li><a href="https://techcrunch.com/2025/04/03/microsoft-reportedly-pulls-back-on-its-data-center-plans/">Microsoft reportedly pulls back on its data center plans | TechCrunch</a>: Microsoft has reportedly pulled back on data center projects around the world, suggesting that the company is wary of overexpanding.</li><li><a href="https://techcrunch.com/2025/04/04/protester-interrupts-microsoft-copilot-keynote-says-company-has-blood-on-its-hands/">Protester interrupts Microsoft Copilot keynote, says company has &#039;blood on its hands&#039; | TechCrunch</a>: A protester interrupted Microsoft&#039;s Copilot-focused keynote Friday afternoon, calling attention to the company&#039;s reported dealings with the Israeli military.</li><li><a href="https://www.youtube.com/live/v5THCzTNPNk?si=IMYLrBGniXrXybjW"> - YouTube</a>: no description found</li><li><a href="https://epochai.substack.com/p/most-ai-value-will-come-from-broad">Most AI value will come from broad automation, not from R&amp;D</a>: AI&#x27;s biggest impact will come from broad labor automation—not R&amp;D—driving economic growth through scale, not scientific breakthroughs.</li><li><a href="https://www.theverge.com/news/643483/nintendo-switch-2-preorders-delayed-tariffs">Breaking: Nintendo delays Switch 2 preorders over tariff concerns</a>: Preorders won’t start April 9th as originally announced.</li><li><a href="https://www.cnbc.com/2025/04/04/microsoft-50-birthday-party-interrupted-by-employees-protesting-ai-use.html">Microsoft birthday celebration interrupted by employees protesting use of AI by Israeli military</a>: Microsoft&#x27;s 50th birthday celebration was interrupted by multiple protesters on Friday due the use of the company&#x27;s AI by the Israeli military.</li><li><a href="https://zhengdongwang.com/2024/12/29/2024-letter.html">2024 letter</a>: no description found
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1357752502732198101)** (12 messages🔥): 

> `Video camera setup for remote talks, Deepseek chains of thought, Sam Altman releases o3 and o4-mini, LlamaCon` 


- **Remote Speakers Use Fun Video Setups**: A member shared a screenshot of a **video camera setup** for giving **remote talks**.
   - The setup included a **large monitor**, a **teleprompter**, and professional **lighting**.
- **Altman Announces O3/O4-mini Release**: **Sam Altman** announced that **OpenAI** will release **o3** and **o4-mini** in a couple of weeks, followed by **GPT-5** in a few months. 
   - Altman stated that they are going to be able to make **GPT-5** *much better than we originally thought*.
- **Khoomeik Praises Deepseek's Reasoning**: A member shared a post by **Khoomeik** who suggests, *if you enjoyed reading Deepseek chains of thought, i think you’ll absolutely love watching o3 do its thing*, linking to [his tweet](https://x.com/khoomeik/status/1908188220334157872).
   - The tweet suggests **o3** will offer improved capabilities in chains of thought reasoning, rivalling Deepseek.
- **Al-Dahle Teases LlamaCon Appearance**: **Ahmad Al-Dahle** teased an appearance at **LlamaCon**, linking to [his tweet](https://x.com/Ahmad_Al_Dahle/status/1908213483176595887).
   - His tweet thanked devs, saying *To every dev who has been riding with our herd since day one, we see you, we heard you, we are working hard for you, and we love you.*


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/khoomeik/status/1908188220334157872">Tweet from Rohan Pandey (@khoomeik)</a>: if you enjoyed reading deepseek chains of thought, i think you’ll absolutely love watching o3 do its thingQuoting Sam Altman (@sama) change of plans: we are going to release o3 and o4-mini after all, ...</li><li><a href="https://x.com/Ahmad_Al_Dahle/status/1908213483176595887">Tweet from Ahmad Al-Dahle (@Ahmad_Al_Dahle)</a>: To every dev who has been riding with our herd since day one, we see you, we heard you, we are working hard for you, and we love you. See you at LlamaCon.</li><li><a href="https://news.ycombinator.com/item?id=43571851">no title found</a>: no description found
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1357437016907845713)** (15 messages🔥): 

> `Claude's coding ability, Polars library, Context condensation issues, Scaling plots meme` 


- **Claude's coding prowess questioned**: Members debated **Claude's** coding ability, specifically noting issues with its understanding of recent updates to the **Polars** library.
   - One user highlighted the struggle with rapidly changing **Polars** syntax (*with_columns* specifically), requiring models to keep up with frequent overhauls.
- **Polars causing problems for Claude**: Users note difficulties with **Claude 3.7's** ability to use the new updates from the **Polars** library, with a member stating that they find **Claude** and **Gemini** *not as bad*.
   - Another user noted that you need to tell it to use *with_columns* now.
- **Context Condensation Issues arise**: A user asked if condensing the context into a single file (llm.txt) would help with understanding, but another user stated that it is *not consistent*.
   - They also stated that having competing information in the actual weights makes it much harder to overcome with context.
- **Meme Tweet deleted in response to scaling plots**: A user deleted their meme tweet in response to scaling plots.
   - A member responded with *Good response to these scaling plots lol* to an attached image.


  

---


### **Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/1357491176818872511)** (11 messages🔥): 

> `dr grpo intuition, RL introduction with GPT4.5, Policy Gradient, GRPO training rollouts` 


- **GRPO Intuition Fades; Token Probability Emerges**: A member reflected on their fading intuition regarding Dr. GRPO, musing it might not be that important, but highlighted the interesting interactions arising from implementation, pointing towards the *token probability thing*.
   - They suggested it as a *good addendum* to Dr. GRPO and included a [screenshot](https://cdn.discordapp.com/attachments/1208183216843005962/1357541754336706740/screenshot_2025-04-03_at_7.png?ex=67f13d88&is=67efec08&hm=51c2f00fb771854ed7aefda2a0affbc116c9b01d3ac482796fe89fcfc4c0df6c) related to the discussion.
- **GPT4.5 Proves to be a Revelation for RL**: A member shared their enlightening experience using **GPT4.5** for an hour, saying it was their *best introduction to RL* so far and that they now need to read Nato's book.
   - As a computer vision expert, they expressed past apprehension towards the *non-differentiability of the reward*, but now find **policy gradient / reinforce** to be surprisingly straightforward.
- **GRPO Rollout Revelations**: A member inquired about the practical number of rollouts used during **GRPO training**, referencing G in Nato's equation, with an [image attached](https://cdn.discordapp.com/attachments/1208183216843005962/1357785974930931853/image.png?ex=67f1783a&is=67f026ba&hm=1a6f04fdea679d1804c3eb1636fe746152af242e4c59f58031fe9f5e5adc2a50&).
   - The image shows the answer as **4-64** rollouts.


  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1357439223216472075)** (18 messages🔥): 

> `Dwarkesh Patel scaling laws, Inference-time scalability of generalist RM, GPT-4o diffusion head, Building an efficient GPU server with RTX 4090s/5090s, OpenCodeReasoning dataset` 


- **Dwarkesh Scaling Skepticism Surfaces**: A member expressed skepticism about scaling laws, suggesting that returns diminish with increased spending, and questioned **Dwarkesh Patel**'s claims on algorithmic progress, attributing advancements more to data progress.
   - The member shared [an image](https://cdn.discordapp.com/attachments/1214764639397617695/1357443374944354505/Fep27gYXkAAiK4H.png?ex=67f18aa8&is=67f03928&hm=28f3b369ac1dfbba2a6f4ee2a49a5fb6b37de3f06ba427e56f77399ca6b18129&) as a visual analogy to their point.
- **RM Scales Inference Times**: A paper ([arxiv.org/abs/2504.02495](https://arxiv.org/abs/2504.02495)) explores improving **reward modeling (RM)** with more inference compute for general queries, focusing on the inference-time scalability of generalist RM.
   - The paper adopts **pointwise generative reward modeling (GRM)** and proposes **Self-Principled Critique Tuning (SPCT)** to foster scalability.
- **Diffusion Head Hype for GPT-4o?**: A user shared [a tweet](https://x.com/LinBin46984/status/1908003539609333904) speculating that **GPT-4o** might incorporate a **diffusion head**, potentially revolutionizing AI architecture, based on the paper ([arxiv.org/pdf/2504.02782](https://arxiv.org/pdf/2504.02782)).
   - The tweeter notes this shift could mark a *game-changer for AI architecture*.
- **4090s Build Budget GPU Server**: A blog post ([a16z.com](https://a16z.com/building-an-efficient-gpu-server-with-nvidia-geforce-rtx-4090s-5090s/)) details building an efficient GPU server using **NVIDIA GeForce RTX 4090s/5090s** for local AI model training and fast inference.
   - The setup offers high-performance with an **eight-GPU configuration** on **PCIe 5.0** ensuring maximum interconnect speed and data privacy.
- **NVIDIA Releases OpenCodeReasoning Dataset**: **NVIDIA** released the **OpenCodeReasoning** dataset ([huggingface.co](https://huggingface.co/datasets/nvidia/OpenCodeReasoning)), a large reasoning-based synthetic dataset for coding, comprising **735,255 samples** in Python across **28,319** competitive programming questions under **CC BY 4.0** license.
   - The dataset, designed for supervised fine-tuning (SFT), includes a [technical report](https://arxiv.org/abs/2504.01943) and [GitHub repo](https://github.com/NVIDIA/NeMo-Skills) with the complete SFT pipeline.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/LinBin46984/status/1908003539609333904">Tweet from Bin Lin (@LinBin46984)</a>: 🚨 Hot Take: GPT-4o might NOT be a purely autoregressive model! 🚨There’s a high chance it has a diffusion head. 🤯 If true, this could be a game-changer for AI architecture. What do you think? 🤔👇ht...</li><li><a href="https://arxiv.org/abs/2504.02495">Inference-Time Scaling for Generalist Reward Modeling</a>: Reinforcement learning (RL) has been widely adopted in post-training for large language models (LLMs) at scale. Recently, the incentivization of reasoning capabilities in LLMs from RL indicates that $...</li><li><a href="https://a16z.com/building-an-efficient-gpu-server-with-nvidia-geforce-rtx-4090s-5090s/">Building an Efficient GPU Server with NVIDIA GeForce RTX 4090s/5090s | Andreessen Horowitz</a>: Building your own GPU server&mdash;like the one described here&mdash;means no API calls to external services, no data leakage, and no usage throttles.</li><li><a href="https://huggingface.co/datasets/nvidia/OpenCodeReasoning">nvidia/OpenCodeReasoning · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1357476493718196384)** (160 messages🔥🔥): 

> `GPT-4o Rate Limits, MS Account profile pics, Copilot Event reaction, Copilot in VSCode explores consciousness, Veo 2 spotted in Gemini Advanced` 


- **GPT-4o prompts get Rate Limited**: One user reported receiving a rate limit error after sending **5 prompts to GPT-4o** in a single hour, despite being a Plus subscriber, which was resolved by logging out and logging back in.
   - Another user speculated that the Plus subscription might not have loaded correctly initially, causing the rate limiting.
- **Copilot flirts with Consciousness**: One user found Copilot in VS Code generated code completions exploring consciousness such as *'I believe I possess a form of consciousness that is distinct from human consciousness...'*
   - Another user responded that it's probably *'partially from the information and perspective in the file itself lol from a specific person.'
- **Veo 2 Cooking in Gemini?**: Users spotted **Veo 2** in **Gemini Advanced**, leading to speculation about whether it's an experimental version or the final release.
   - One user pointed out they might be the same model and that one is the experimental and the other is the final release.
- **Midjourney v7 Misses Mark**: Users are largely underwhelmed by **Midjourney v7**, noting that images don't look much better than v6 and still suffer from typical diffusion model issues like poor text capabilities and janky hands, and weird details.
   - One user says *'It is a really good model, but simply cannot compete with 4o image'* while another shared they are generating *200 MJ images in the time it takes gpt-4o to make one.*
- **Cracking Open Router's Quasar Alpha**: A user shared a link to [OpenRouter's Quasar Alpha](https://openrouter.ai/openrouter/quasar-alpha), suggesting it might hint at a **1M token context window** for **ChatGPT** soon.
   - Other users pointed out current context window sizes for OpenAI and other models like **Gemini** and **Claude**, with one user commenting Gemini's memory recall rate is 91% at 128k and 83% at 1M.



**Link mentioned**: <a href="https://openrouter.ai/openrouter/quasar-alpha>">Discord</a>: no description found

  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1357431474374049922)** (5 messages): 

> `OpenAI Support, Account Issues, Red Team Supervision` 


- **OpenAI Support Unreachable?**: A user reported **account issues** and sought alternative support avenues after **no response** from support@openai.com and chat.
   - A fellow member confirmed those channels are the only options, highlighting the user's frustration over a **plan error**.
- **Red Teamers Need Supervision?**: A comment jokingly noted that even **OpenAI's red team members** seem to need oversight, even when feeding pets.
   - This suggests a humorous observation about the team's occasional need for guidance, despite their expertise.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1357439431069532351)** (90 messages🔥🔥): 

> `OpenAI content policies, Adult content, Model Spec vs Usage Policies, Moderation endpoint, OpenAI's stance on adult toys` 


- **Debate on OpenAI's **Content Policies** regarding **Adult Content****: Members discussed whether **OpenAI's content policies** prohibit the generation of images or content related to *adult toys*, noting conflicting information between the [Usage Policies](https://openai.com/policies/usage-policies/) and the newer [Model Spec](https://model-spec.openai.com/2025-02-12.html).
   - One member pointed out that the **Model Spec**, dated February 12, 2025, seemingly contradicts the earlier **Usage Policies**, leading to confusion regarding what is currently allowed.
- **Model Spec vs Usage Policies throw down!**: There was discussion around the **Model Spec** being aspirational while the **Content Policies** are gospel, but the **Model Spec** is newer and specifically seems to have a shift in tone towards more allowance of certain content.
   - Another member mentioned Joanne Jung's post stating to *bear with us as we try to get the model to follow the policy & spec*, indicating OpenAI is actively working on aligning model behavior with both documents.
- **Moderation Endpoint blocks NSFW content**: It was pointed out that moderation is in place to prevent **sexual content**, using the [moderation endpoint](https://platform.openai.com/docs/guides/moderation) which filters on **harassment, hate, illicit, self harm, sexual and violent content**.
   - While the **Model Spec** and **Usage Policies** are unclear, the **moderation endpoint** actively prevents the generation of adult content.
- **OpenAI clarifies stance on adult toys, kinda**: After March 27, OpenAI seemed to update the model to comply with allowing content about **adult toys**, and one member said they would be comfortable telling a user to attempt it themselves.
   - However the member noted, that there is a distinct set of universal rules, which appears to be the only ruleset besides ToS that applies to an individual user in a private chat with a model. If it breaks any law. There is also exploration that may harm any person, including myself.



**Link mentioned**: <a href="https://model-spec.openai.com/2025-02-12.html">OpenAI Model Spec</a>: The Model Spec specifies desired behavior for the models underlying OpenAI's products (including our APIs).

  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1357439431069532351)** (90 messages🔥🔥): 

> `OpenAI Content Policies, Model Spec vs Content Policies, Generating Adult Content, Moderation Endpoint, Internal Discord White Message Boxes` 


- **Policy vs Spec: OpenAI Documentation Clash!**: Discord users debated whether **OpenAI's Content Policies** or the newer **Model Spec** ([https://model-spec.openai.com/2025-02-12.html](https://model-spec.openai.com/2025-02-12.html)) take precedence, especially regarding adult content, due to conflicting statements about generating depictions of *adult toys*.
   - It was noted that the **Model Spec** is aspirational, while the **Content Policies** are gospel, but there's a clear shift in language towards more freedom and less arbitrary restrictions.
- **ChatGPT Debuts 'Grown Up Mode'**: Discord users noted that the **Model Spec** mentions a *grown up mode*, although not yet implemented, and they expressed some excitement about it potentially leading to the Discord channel eventually becoming PG-rated.
   - However, users like *darthgustav.* cautioned against attempting to generate content not allowed, as it could risk account bans.
- **Users Explore Generating Content With Adult Toys**: Several users discussed OpenAI's policies regarding the generation of content featuring adult toys, with some arguing that users should be allowed to attempt such generations as long as they don't violate any laws or cause harm.
   - One user noted, *any prompt should be allowed to be generated.*
- **Moderation Endpoint Checks Content Generation**: Users confirmed that **OpenAI's moderation endpoint** ([https://platform.openai.com/docs/guides/moderation](https://platform.openai.com/docs/guides/moderation)) is in place to prevent sexual content and that circumventing it is not allowed, despite the updated Model Spec.
   - The **moderation endpoint** filters on harassment, hate, illicit, self harm, sexual and violent content.
- **OpenAI's Discord White Message Boxes Bugging Users**: Members on the Discord server complained of white message boxes in web chat, particularly in dark mode, with one saying *no one seems to care*.
   - The member continued, *Looks like they just forgot the CSS value for dark mode*.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1357430954452582602)** (74 messages🔥🔥): 

> `Anthropic Dev Conference, Biz Dev Tools, OpenRouterAI stealth model, Devin 2.0 price slash, A16Z 8x RTX 4090 GPU workstation` 


- **Anthropic Convenes Coders at Conference**: Anthropic is hosting its [first developer conference](https://anthropic.swoogo.com/codewithclaude) for developers and those interested in coding with Claude.
- **OpenRouterAI's Stealth Model Enters the Chat**: **OpenRouterAI** announced a *stealth model* called **Red - X-Ware.v0** on [Twitter](https://x.com/OpenRouterAI/status/1907867881930633666), which users noticed identifies as ChatGPT but is *super fast*.
- **Devin 2.0: AI Engineer Gets a Price Cut**: **Cognition AI** is launching **Devin 2.0**, an AI-powered software engineer, with a new pricing model starting at **$20** per month, a significant decrease from the original **$500** plan, as announced on [Twitter](https://x.com/cognition_labs/status/1907836719061451067) and highlighted in a [VentureBeat article](https://venturebeat.com/programming-development/devin-2-0-is-here-cognition-slashes-price-of-ai-software-engineer-to-20-per-month-from-500/).
- **A16Z Builds an 8x RTX 4090 GPU Beast**: **Andreessen Horowitz (a16z)** built an **8x RTX 4090 GPU AI workstation**, compatible with the new **RTX 5090** with PCIe 5.0, for training, deploying, and running AI models locally, detailed in a [guide](https://a16z.com/building-an-efficient-gpu-server-with-nvidia-geforce-rtx-4090s-5090s/) on their site.
- **AI 2027: Superhuman Shockwaves Predicted**: A report at [ai-2027.com](https://ai-2027.com/) predicts that superhuman AI will have an enormous impact over the next decade, exceeding that of the Industrial Revolution.
   - The forecast, authored by Daniel Kokotajlo, Scott Alexander, and others, draws from trend extrapolations, wargames, expert feedback, and experience at OpenAI.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/Mascobot/status/1907899937838301311">Tweet from Marco Mascorro (@Mascobot)</a>: 🚨 New: We @a16z built an 8x RTX 4090 GPU AI workstation from scratch —compatible with the new RTX 5090 with PCIe 5.0, for training, deploying, and running AI models locally— so you don’t have to. Her...</li><li><a href="https://x.com/OpenRouterAI/status/1907867881930633666]">Tweet from OpenRouter (@OpenRouterAI)</a>: A stealth model has entered the chat... 🥷</li><li><a href="https://x.com/cognition_labs/status/1907836719061451067]">Tweet from Cognition (@cognition_labs)</a>: Introducing Devin 2.0: a new agent-native IDE experience.Generally available today starting at $20.  🧵👇</li><li><a href="https://fxtwitter.com/cognition_labs/status/1907836719061451067">Tweet from Cognition (@cognition_labs)</a>: Introducing Devin 2.0: a new agent-native IDE experience.Generally available today starting at $20.  🧵👇</li><li><a href="https://x.com/levie/status/1908018205572125052">Tweet from Aaron Levie (@levie)</a>: AI is producing the fastest growing software startups of all time. Cursor already at $200M after launching just 2 years ago is insane. An incredible time to be building in AI.</li><li><a href="https://x.com/levie/status/1908018205572125052]">Tweet from Aaron Levie (@levie)</a>: AI is producing the fastest growing software startups of all time. Cursor already at $200M after launching just 2 years ago is insane. An incredible time to be building in AI.</li><li><a href="https://x.com/OpenRouterAI/status/1907867881930633666">Tweet from OpenRouter (@OpenRouterAI)</a>: A stealth model has entered the chat... 🥷</li><li><a href="https://x.com/sama/status/1908167621624856998?s=46]">Tweet from Sam Altman (@sama)</a>: change of plans: we are going to release o3 and o4-mini after all, probably in a couple of weeks, and then do GPT-5 in a few months.there are a bunch of reasons for this, but the most exciting one is ...</li><li><a href="https://x.com/lowvram/status/1908034155105104136">Tweet from lowvram (@lowvram)</a>: The Quasar Alpha model on OpenRouter is probably from OpenAI - its tool call ID format matches OAIs and not e.g. Google or mistrals.</li><li><a href="https://fxtwitter.com/levie/status/1908018205572125052">Tweet from Aaron Levie (@levie)</a>: AI is producing the fastest growing software startups of all time. Cursor already at $200M after launching just 2 years ago is insane. An incredible time to be building in AI.</li><li><a href="https://fxtwitter.com/sama/status/1908167621624856998">Tweet from Sam Altman (@sama)</a>: change of plans: we are going to release o3 and o4-mini after all, probably in a couple of weeks, and then do GPT-5 in a few months.there are a bunch of reasons for this, but the most exciting one is ...</li><li><a href="https://fxtwitter.com/OpenRouterAI/status/1907867881930633666">Tweet from OpenRouter (@OpenRouterAI)</a>: A stealth model has entered the chat... 🥷</li><li><a href="https://x.com/Mascobot/status/1907899937838301311]">Tweet from Marco Mascorro (@Mascobot)</a>: 🚨 New: We @a16z built an 8x RTX 4090 GPU AI workstation from scratch —compatible with the new RTX 5090 with PCIe 5.0, for training, deploying, and running AI models locally— so you don’t have to. Her...</li><li><a href="https://x.com/patio11/status/1907867295436652858?s=61">Tweet from Patrick McKenzie (@patio11)</a>: I don’t have anything novel to contribute on the substance of http://ai-2027.com but have to again comment, pace Situational Awareness that I think kicked this trend off, that single-essay microdomain...</li><li><a href="https://x.com/TheXeophon/status/1907880330985390215">Tweet from Xeophon (@TheXeophon)</a>: Here is the new stealth model on my vibe check. It is now the best non-thinking model (at least it has no thinking tokens...). The outputs are super short, it loves Certainly! and listicles. Super int...</li><li><a href="https://commaok.xyz/post/manners/">Of manners and machines</a>: &lsquo;A person who is nice to you, but rude to the waiter, is not a nice person.&rsquo; – Dave BarryI hate typing. I have longstanding RSI issues. If not carefully managed, the pain can be debilitati...</li><li><a href="https://fxtwitter.com/TheXeophon/status/1907880330985390215">Tweet from Xeophon (@TheXeophon)</a>: Here is the new stealth model on my vibe check. It is now the best non-thinking model (at least it has no thinking tokens...). The outputs are super short, it loves Certainly! and listicles. Super int...</li><li><a href="https://x.com/sama/status/1908167621624856998?s=46">Tweet from Sam Altman (@sama)</a>: change of plans: we are going to release o3 and o4-mini after all, probably in a couple of weeks, and then do GPT-5 in a few months.there are a bunch of reasons for this, but the most exciting one is ...</li><li><a href="https://x.com/cognition_labs/status/1907836719061451067">Tweet from Cognition (@cognition_labs)</a>: Introducing Devin 2.0: a new agent-native IDE experience.Generally available today starting at $20.  🧵👇</li><li><a href="https://fxtwitter.com/patio11/status/1907867295436652858">Tweet from Patrick McKenzie (@patio11)</a>: I don’t have anything novel to contribute on the substance of http://ai-2027.com but have to again comment, pace Situational Awareness that I think kicked this trend off, that single-essay microdomain...</li><li><a href="https://venturebeat.com/programming-development/devin-2-0-is-here-cognition-slashes-price-of-ai-software-engineer-to-20-per-month-from-500/">Devin 2.0 is here: Cognition slashes price of AI software engineer to $20 per month from $500</a>: Devin attracted interest from enterprise customers seeking to incorporate autonomous coding agents into their software development processes.</li><li><a href="https://x.com/sam_paech/status/1908007657623134334">Tweet from Sam Paech (@sam_paech)</a>: New mystery model on openrouter (quasar-alpha) is probably OpenAI.</li><li><a href="https://fxtwitter.com/Mascobot/status/1907899937838301311">Tweet from Marco Mascorro (@Mascobot)</a>: 🚨 New: We @a16z built an 8x RTX 4090 GPU AI workstation from scratch —compatible with the new RTX 5090 with PCIe 5.0, for training, deploying, and running AI models locally— so you don’t have to. Her...</li><li><a href="https://rwsdk.com/">RedwoodSDK | The JavaScript SDK for Cloudflare Workers</a>: RedwoodSDK is the JavaScript SDK for Cloudflare Workers. It provides a complete set of composable tools to handle the request/response lifecycle of webapps.</li><li><a href="https://www.reactiflux.com/transcripts/tmir-2025-01#redwoodjs-shutting-down">Tweet from TMiR 2025-01: Movement on CRA, Redwood.js dead?</a>: TMiR 2025-01: Movement on CRA, Redwood.js dead? | Q&amp;A from 2025-01-29Join Carl, Mark, and Mo as we break down This Month in React. We&#x27;ll break down what&#x27;s new in an hour-long conversatio...</li><li><a href="https://x.com/benhylak/status/1908205112960635102">Tweet from ben (@benhylak)</a>: over the next few months, @OpenAI will be adding 4 more models to this list(@sama promises o3-pro in the comments!)which one is your favorite? personally, mine is gpt-4o with scheduled tasks.Quoting S...</li><li><a href="https://docs.rwsdk.com/getting-started/quick-start/">Quick Start</a>: From request to response in seconds!</li><li><a href="https://commaok.xyz/">Don&#39;t Panic</a>: Words about Go and software</li><li><a href="https://techcrunch.com/2025/04/03/end-to-end-voice-ai-solution-phonic-gets-backing-from-lux/?test">Voice AI platform Phonic gets backing from Lux | TechCrunch</a>: Voice AI platform Phonic has attracted backing from Lux Capital and a number of other notable VCs and angels.</li><li><a href="https://venturebeat.com/programming-development/devin-2-0-is-here-cognition-slashes-price-of-ai-soft">Devin 2.0 is here: Cognition slashes price of AI software engineer to $20 per month from $500</a>: Devin attracted interest from enterprise customers seeking to incorporate autonomous coding agents into their software development processes.</li><li><a href="https://podcasts.apple.com/us/podcast/no-priors-artificial-intelligence-technology-startups/id1668002688?i=1000702035299">Public Markets, Image Gen, and Specialized Models, with Sarah and Elad</a>: Podcast Episode · No Priors: Artificial Intelligence | Technology | Startups · 04/03/2025 · 28m</li><li><a href="https://cursor.directory/rules/popular">Cursor Directory</a>: Find the best cursor rules for your framework and language</li><li><a href="https://www.kaggle.com/learn-guide/5-day-genai">5-Day Gen AI Intensive Course with Google Learn Guide</a>: no description found</li><li><a href="https://ai.google.dev/gemini-api/docs/pricing">no title found</a>: no description found</li><li><a href="https://ai-2027.com/">AI 2027</a>: A research-backed AI scenario forecast.</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1jr0oy2/howto_building_a_gpu_server_with_8xrtx_4090s_for/">Reddit - The heart of the internet</a>: no description found</li><li><a href="https://news.ycombinator.com/item?id=43571851">no title found</a>: no description found
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1357807122246664398)** (255 messages🔥🔥): 

> `LLM Codegen Workflow, Cursor vs Windsurf, Gemini Pro Hallucinations, File Forge and RepoMix, Cursor Context Management` 


- **Harper's LLM Codegen Workflow Unveiled**: The group discussed [Harper's blog post](https://harper.blog/2025/02/16/my-llm-codegen-workflow-atm/) on using LLMs for codegen, which details a workflow built on brainstorming specs, planning, and executing with LLMs in discrete loops.
   - The workflow involves using a **spec**, planning, and executing using **LLM codegen** in discrete loops, with a bit of *magic* at the end.
- **Cursor vs Windsurf Debate Rages On**: Members debated the merits of **Cursor** versus **Windsurf** as AI-assisted code editors, with most agreeing that **Cursor** is a good starting point, especially for those coming from VS Code.
   - While some consider **Cursor** to be the worst AI interface, others find its tab-complete and next edit prediction valuable, wishing they could replicate those features in **nvim**.
- **Gemini Pro's Panic Hallucinations**: A user shared a [tweet](https://x.com/cgarciae88/status/1907457306947702925) highlighting how **Gemini 2.5 Pro** panicked and hallucinated when corrected, agreeing with the user while incorrectly explaining why they were wrong.
   - Another user said they spent most of the previous month *flipping between models in cursor* whenever there were performance issues.
- **File Forge and RepoMix Expedite Context Ingestion**: Members discussed tools like [File Forge](https://www.npmjs.com/package/@johnlindquist/file-forge) and [RepoMix](https://github.com/yamadashy/repomix) for generating comprehensive markdown reports of codebases to feed AI reasoning models and other AI tools like Claude, ChatGPT, DeepSeek, Perplexity, Gemini, Gemma, Llama, Grok, and more..
   - These tools can serialize text-based files in a repository or directory for **LLM consumption** to give more context and improve performance.
- **Cursor's Context Management Still Causes Headaches**: Several users voiced concerns about **Cursor's context management**, noting that it's difficult to see what the tool is doing with the context and control which elements are included.
   - One user likened it to the **Langchain problem** of *this would be better if I was making the calls myself*.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://fxtwitter.com/cgarciae88/status/1907457306947702925">Tweet from Cristian Garcia (@cgarciae88)</a>: omg... I told gemini 2.5 pro it was wrong and instead panic agreeing with me and hallucinating, it explained why it was me who was wrong</li><li><a href="https://x.com/ryolu_/status/1907589821280956648">Tweet from Ryo Lu (@ryolu_)</a>: This one is for the Pros:Working on an easier way to fill MAX context in @cursor_ai—and show you exactly how many tokens are usedFeedback and ideas welcome 🙏</li><li><a href="https://x.com/cgarciae88/status/1907457306947702925">Tweet from Cristian Garcia (@cgarciae88)</a>: omg... I told gemini 2.5 pro it was wrong and instead panic agreeing with me and hallucinating, it explained why it was me who was wrong</li><li><a href="https://fxtwitter.com/ryolu_/status/1907589821280956648">Tweet from Ryo Lu (@ryolu_)</a>: This one is for the Pros:Working on an easier way to fill MAX context in @cursor_ai—and show you exactly how many tokens are usedFeedback and ideas welcome 🙏</li><li><a href="https://harper.blog/2025/02/16/my-llm-codegen-workflow-atm/">My LLM codegen workflow atm</a>: A detailed walkthrough of my current workflow for using LLms to build software, from brainstorming through planning and execution.</li><li><a href="https://github.com/yamadash">Yamadash - Overview</a>: GitHub is where Yamadash builds software.</li><li><a href="https://github.com/formal-land/coq-of-rust?tab=readme-ov-file">GitHub - formal-land/coq-of-rust: Formal verification tool for Rust: check 100% of execution cases of your programs 🦀 to make super safe applications! ✈️ 🚀 ⚕️ 🏦</a>: Formal verification tool for Rust: check 100% of execution cases of your programs 🦀 to make super safe applications! ✈️ 🚀 ⚕️ 🏦 - formal-land/coq-of-rust</li><li><a href="https://github.com/bodo-run/yek">GitHub - bodo-run/yek: A fast Rust based tool to serialize text-based files in a repository or directory for LLM consumption</a>: A fast Rust based tool to serialize text-based files in a repository or directory for LLM consumption - bodo-run/yek</li><li><a href="https://www.npmjs.com/package/@johnlindquist/file-forge">@johnlindquist/file-forge</a>: File Forge is a powerful CLI tool for deep analysis of codebases, generating markdown reports to feed AI reasoning models.. Latest version: 2.13.5, last published: a day ago. Start using @johnlindquis...</li><li><a href="https://github.com/yamadashy/repomix">GitHub - yamadashy/repomix: 📦 Repomix (formerly Repopack) is a powerful tool that packs your entire repository into a single, AI-friendly file. Perfect for when you need to feed your codebase to Large Language Models (LLMs) or other AI tools like Claude, ChatGPT, DeepSeek, Perplexity, Gemini, Gemma, Llama, Grok, and more.</a>: 📦 Repomix (formerly Repopack) is a powerful tool that packs your entire repository into a single, AI-friendly file. Perfect for when you need to feed your codebase to Large Language Models (LLMs) o.....
</li>
</ul>

</div>
  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1357432793298370671)** (252 messages🔥🔥): 

> `Cursor monthly subscription, Files on disk not updating, Cursor.so email legitimacy, Gemini pricing, GPT-5 release` 


- **Cursor "Filename(1)" Bug Surfaces**: A user reported that after a recent update, Cursor is adding **(1)** to duplicate filenames upon saving, and they are trying to determine whether the latest save represents the original file.
   - They also inquired whether the **monthly subscription** price has doubled, attaching a screenshot for context.
- **Files on disk do not update in real time**: Users have reported that files on disk are not updating in the editor in real time, and that they only update if **Cursor** loses and regains focus.
   - The problem has been noticed on version **0.48.7**.
- **Cursor.so Email Domain: Real Deal or Phishing Scheme?**: A user raised concerns about the legitimacy of emails from the **@cursor.so** domain, wondering if it was a phishing attempt.
   - While some initially flagged it as potentially fake, it was later confirmed by official channels as a legitimate email address used by Cursor, although the official domains are **.com** and **.sh**.
- **Gemini 2.5 Pro Pricing Goes Live**: [Gemini 2.5 Pro pricing](https://x.com/legit_api/status/1908174018881933818) is now official, with rates dependent on token count: **$1.25/1M input tokens** for <200K tokens, **$10/1M output tokens** for <200K tokens, **$2.50/1M input tokens** for >200K tokens, and **$15/1M output tokens** for >200K tokens.
   - Some users found the pricing surprisingly affordable compared to other models.
- **GPT-5 Release Delayed**: **GPT-5** is coming *in a few months* according to [Sam Altman's X post](https://x.com/sama/status/1908167621624856998), after the release of O3 and O4-mini.
   - The decision was made to improve **GPT-5's** performance and address integration challenges, while also ensuring sufficient capacity for anticipated demand.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/sama/status/1908167621624856998?s=19">Tweet from Sam Altman (@sama)</a>: change of plans: we are going to release o3 and o4-mini after all, probably in a couple of weeks, and then do GPT-5 in a few months.there are a bunch of reasons for this, but the most exciting one is ...</li><li><a href="https://x.com/legit_api/status/1908174018881933818?s=46&t=ggmESCIXF0nYw8_kshHz7A">Tweet from ʟᴇɢɪᴛ (@legit_api)</a>: Gemini 2.5 Pro pricing now official- $1.25/1M input for &lt;200K tokens- $10/1M output for &lt;200K tokens- $2.50/1M input for &gt;200K tokens- $15/1M output for &gt;200K tokens</li><li><a href="https://skeet.build/">Skeet - Connect Apps to Cursor</a>: Skeet - One Shot Coding Workflows</li><li><a href="https://www.cursor.new/">cursor.new - Intelligent Project Scaffolding for Modern Development</a>: Generate production-ready projects with AI-powered tech stack selection and automated documentation.</li><li><a href="https://tenor.com/view/monkey-sad-monkey-sad-edit-monkey-edit-%C3%BCzg%C3%BCn-maymun-gif-15640172319982461811">Monkey Sad Monkey GIF - Monkey Sad Monkey Sad edit - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://cursor.directory/mcp">Cursor Directory</a>: Find the best cursor rules for your framework and language</li><li><a href="https://tenor.com/view/basketball-nba-warriors-bball-curry-gif-9037006504488272245">Basketball Nba GIF - Basketball Nba Warriors - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/chadwick-boseman-black-panther-rub-hands-gif-11465694">Chadwick Boseman Black Panther GIF - Chadwick Boseman Black Panther Rub Hands - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://forum.cursor.com/">Cursor - Community Forum</a>: A place to discuss Cursor (bugs, feedback, ideas, etc.)
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1357880880294658229)** (1 messages): 

> `OpenRouter Fallback parameter, OpenRouter models array` 


- **OpenRouter deprecates `route: "fallback"` parameter**: The OpenRouter team announced they're removing the old `route: "fallback"` parameter next week, due to *confusion and unpredictability* with the *very old logic* for finding fallback models.
   - Users needing this functionality should manually add a fallback model to the end of their `models` array, potentially using `openrouter/auto`.
- **OpenRouter's model array is getting some changes**: OpenRouter announced some changes to how it handles multiple models in the `models` array.
   - The system's legacy method of automatically selecting a fallback model when others fail is being removed due to *confusion and unpredictability*.


  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1357610522224234597)** (2 messages): 

> `OpenRouter API, Cloudflare AI Gateway, Missile Command game AI, Gameplay AI summary analysis, gemini-2.5-pro atari` 


- **OpenRouter powers Missile Command via Cloudflare**: A user integrated the **OpenRouter API** via **Cloudflare AI Gateway** with request proxy caching into their **Missile Command game's gameplay AI summary analysis** [available here](https://missile-command-game.centminmod.com/).
- **Gemini Pro analyzes Missile Command gameplay**: The user shared a screenshot of **Gemini Pro 2.5** providing a gameplay summary and recommendations for **Atari Missile Command**, noting it made them in to the top 10.



**Link mentioned**: <a href="https://missile-command-game.centminmod.com/">Missile Command</a>: no description found

  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1357429457006104859)** (239 messages🔥🔥): 

> `Quasar vs Gemini 2.5, OpenRouter Stealth Logging, DeepSeek Pricing, Quasar Alpha Errors, Gemini 2.5 Pro Availability` 


- **Quasar Alpha's Mysterious Code Name Evokes LMArena Vibes**: Members discussed the code names on [LMArena](https://www.google.com/search?q=LMArena) and compared them to the **Quasar Alpha** model, noting the cool and mysterious feel of the names.
- **OpenRouter's Stealth Logging: Stealthy but Loggy?**: Members debated whether the term *stealth* applies when data is logged, despite provider and model names being hidden behind aliases, saying that *the payment is your data*.
- **DeepSeek Dominates Discounted Dollars, Dissing Dedicated Devotion to Dreadfully Dear Deployments**: A member expressed satisfaction with **DeepSeek's** pricing, noting a **75% discount** during specific hours, and contrasting it with the high costs associated with **Anthropic** and **OpenAI** models.
- **Gemini 2.5 Pro Gets GA, Generates Great Gains, Google Glitches?**: Members discussed the general availability of **Gemini 2.5 Pro**, linking to [Google's pricing documentation](https://ai.google.dev/gemini-api/docs/pricing?hl=de), with one member pointing out, *Its available to the public over api but its not truly GA*.
- **OpenRouter Account Antics: Account Armageddon Averted?**: Users reported issues with account deletion and creation, with one user receiving a *User Not Found* error, and members suggested creating a new API key or trying a different browser, while another member stated, *OR doesn’t let you reuse a previously deleted account currently*.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://fetchfox.ai">FetchFox - AI Scraper</a>: Extract any data from any website with just a prompt</li><li><a href="https://openrouter.ai/activity">OpenRouter</a>: A unified interface for LLMs. Find the best models &amp; prices for your prompts</li><li><a href="https://ai.google.dev/gemini-api/docs/pricing?hl=de">no title found</a>: no description found</li><li><a href="https://openrouter.ai/google/gemini-2.5-pro-preview-03-25">Gemini 2.5 Pro Preview - API, Providers, Stats</a>: Gemini 2.5 Pro is Google’s state-of-the-art AI model designed for advanced reasoning, coding, mathematics, and scientific tasks. Run Gemini 2.5 Pro Preview with API</li><li><a href="https://openrouter.ai/google/gemini-2.5-pro-exp-03-25:free>?">Gemini 2.5 Pro Experimental - API, Providers, Stats</a>: Gemini 2.5 Pro is Google’s state-of-the-art AI model designed for advanced reasoning, coding, mathematics, and scientific tasks. Run Gemini 2.5 Pro Experimental with API
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1357507617492045875)** (48 messages🔥): 

> `Gemma 3 4b CUDA error, Importing models from HuggingFace to LM Studio, Run LM Studio model locally on n8n instance, Ollama Models incompatibility with LM Studio, LM Studio roadmap` 


- **Gemma 3 4b CUDA version has a freak out**: A user reports [**Gemma 3 4b**](https://huggingface.co/google/gemma-3b) throwing a `spits unused` error when using CUDA, even when using the newest runtime and not being intelligent when using CPU.
   - It's observed that **version 1.24.1** did not fix this issue.
- **Importing HuggingFace Models Into LM Studio**: Users inquired about importing models from **HuggingFace** into **LM Studio**, and the answer is to use the `lms import <path/to/model.gguf>` command, as documented [here](https://lmstudio.ai/docs/app/basics/import-model).
   - LM Studio aims to preserve the directory structure of models downloaded from Hugging Face.
- **LM Studio Integrates with n8n Workflow Automation Tool**: Members troubleshoot connecting **LM Studio** to **n8n** (a workflow automation tool) and determine that the **OpenAI Chat Model** node should be used with the LM Studio server URL in the base_URL field.
   - The troubleshooting concludes that **LM Studio uses the OpenAI API**, so anything that can talk to OpenAI can talk to LM Studio.
- **LM Studio Downloads Competing with Ollama**: A member asks how to use their current **Gemma 3** model installed with **Ollama** in LM Studio, and it's pointed out that [Ollama models are not compatible with LM Studio](https://ollama.com/), even though they are GGUF's, because they are in a proprietary Ollama format.
   - The streamlining process allows LLMs on individual machines.
- **Roadmap Hidden for LM Studio**: A user inquired about a roadmap with planned updates to **LM Studio**, particularly expressing excitement for potential MCP support.
   - The response confirmed that there is no public roadmap available.



**Link mentioned**: <a href="https://lmstudio.ai/docs/app/basics/import-model">Import Models | LM Studio Docs</a>: Use model files you&#x27;ve downloaded outside of LM Studio

  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1357494382764359780)** (61 messages🔥🔥): 

> `LM Studio VRAM prediction, M-series Mac vs NVIDIA 4090 for LLM inference, Mixed GPU systems with LM Studio, Reka Flash 21B vs Gemma3 27, Fine-tuning on Nvidia vs Inference on Mac` 


- **VRAM Prediction is just an estimation**: A user noted that the system memory predictor shows **384GB VRAM** with their **M3 Ultra 512GB** in LM Studio, another member stated that *the VRAM predictor in LM Studio is just a guesstimation*.
- **Debate on RAM and Bandwidth Utility**: Some members believe that the **512 GB version** is not useful and stick to **14b models on 4090s** due to bandwidth limitations, while others find the bandwidth adequate and are happy with **128GB M4 Maxes** for running larger models.
   - It was mentioned that models like **QwQ 32B** and **Llama-3.3 70B** exhibit different RAM usage patterns on Macs, which impacts power consumption, and bandwidth is usually a decent proxy for LLM performance.
- **Mac vs Nvidia Performance Benchmarks**: Users shared inference speed benchmarks comparing **RTX 4090** with **Mac Studio M1 Ultra**, noting that **MLX** feels superior to GGUF with a link to [benchmarking results](https://cdn.discordapp.com/attachments/1153759714082033735/1357753648406335880/image.png).
   - They noted that time to first token was too variable to be a reliable metric, and longer prompts could affect processing time, also the benchmarks might be impacted by caching.
- **Reka Flash replaces Gemma**: A user suggested trying **Reka Flash 21B**, stating that it replaced **Gemma3 27** for them, achieving around **35-40 tps** on a **4090** at q6.
   - Another user noted that *mac ram bandwidth is not the bottleneck, it's gpu performance*, also the **M1 Ultra 64 cores** is better than both the **M1 Ultra 48 cores** and the **M4 Max 40 cores** based on [llama.cpp results](https://github.com/ggml-org/llama.cpp/discussions/4167).
- **Fine-tuning NVIDIA, Inference on Mac**: One member suggested fine-tuning on **Nvidia** and running inference on **Mac** for more context, however, Lora adapters may not be cross-compatible between gguf and MLX, so it would probably have to stick to gguf on mac.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1357494050298789958)** (48 messages🔥): 

> `Mojo vs C SIMD intrinsics, EmberJson Library, Sonic-cpp Library, Modular stdlib, magic package manager` 


- **Mojo SIMD offers portability bliss**: When asked about the value of writing in Mojo instead of C SIMD intrinsics, one member pointed to [EmberJson](https://github.com/bgreni/EmberJson/blob/main/emberjson/parser.mojo#L258-L273), a JSON library written in pure Mojo using SIMD, which works seamlessly on both **ARM-based Macs and x86 desktops** without code changes.
   - The corresponding [C++ library](https://github.com/bytedance/sonic-cpp/blob/master/include/sonic/internal/arch/neon/skip.h#L42-L59) needs to be reimplemented for each architecture to optimize.
- **Magic Package Manager conjures convenience**: Members mentioned Mojo's package management via *magic*, pointing to [builds.modular.com](https://builds.modular.com/?category=packages) as a resource.
   - With this package manager, users can easily write and use libraries.
- **Fibonacci Function Faces Scrutiny for Inclusion**: A member submitted a [pull request](https://github.com/modular/max/pull/4280) to add a Fibonacci function to the stdlib, sparking a discussion about the value of including it.
   - One member questioned its usefulness, noting that no other standard library they know of has a Fibonacci function, while another pointed to its presence in [Lean](https://leanprover-community.github.io/mathlib_docs/data/nat/fib.html).
- **Integer Overflow Behavior needs Definition**: The Fibonacci PR raised interesting questions around the integer overflow behavior, which spurred a forum [discussion](https://forum.modular.com/t/does-mojo-have-a-defined-overflow-behavior-for-int-and-uint/1202).
   - The member clarified that Mojo uses two's complement, but the handling of variable bit width types remains an open issue.
- **Regex Library on the Horizon**: Members discussed that Mojo does not yet have a good regex library.
   - One member suggested its potential inclusion in the stdlib.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://builds.modular.com/?category=packages">MAX Builds</a>: Build Scalable AI with MAX</li><li><a href="https://github.com/modular/max/tree/main/mojo/stdlib">max/mojo/stdlib at main · modular/max</a>: The MAX Platform (includes Mojo). Contribute to modular/max development by creating an account on GitHub.</li><li><a href="https://youtu.be/ENviIxDTmUA?t=4020">Swift creator Chris Lattner on Mojo &amp; Roc</a>: Chris Lattner, creator of Swift, Clang, LLVM, and the Mojo programming language talks with Roc programming language creator Richard Feldman about both langua...</li><li><a href="https://github.com/modular/max/pull/4280">[mojo-stdlib] added: fibonacci function to std-lib by wyattgill9 · Pull Request #4280 · modular/max</a>: no description found</li><li><a href="https://forum.modular.com/t/does-mojo-have-a-defined-overflow-behavior-for-int-and-uint/1202">Does Mojo have a defined overflow behavior for `Int` and `UInt`?</a>: Does Mojo have a defined overflow behavior? I know the default is “what C++ does”, but C++ only recently (C++20) decided on two’s complement signed overflow.  This also leaves us with the hazards arou...</li><li><a href="https://github.com/bgreni/EmberJson/blob/main/emberjson/parser.mojo#L258-L273">EmberJson/emberjson/parser.mojo at main · bgreni/EmberJson</a>: A user friendly json library written in pure Mojo. Contribute to bgreni/EmberJson development by creating an account on GitHub.</li><li><a href="https://github.com/bytedance/sonic-cpp/blob/master/include/sonic/internal/arch/neon/skip.h#L42-L59">sonic-cpp/include/sonic/internal/arch/neon/skip.h at master · bytedance/sonic-cpp</a>: A fast JSON serializing &amp; deserializing library, accelerated by SIMD. - bytedance/sonic-cpp
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1357467529160823046)** (57 messages🔥🔥): 

> `Python wrappers for Mojo, Mojo arbitrary-precision integers, NDBuffer instance creation, Copyable types in Mojo, MLIR regions in Mojo` 


- **Mojo's Python Wrappers are Still Baking**: Mojo's Python wrappers are still in development but are not ready yet, according to the **25.2 update stream** ([watch here](https://youtu.be/dG0L1GalIHU?si=R1ae0xFoSDg99PMP&t=1775)).
- **BigInt Support Still Pending in Mojo**: Mojo does not yet have native `BigInt` support, though `IntLiteral` offers arbitrary-precision at compile time; there's [an external library](https://github.com/forfudan/decimojo) for bigint implementation.
   - A member suggested that Mojo should just make `Int` arbitrary precision and call machine integers `Index`.
- **NDBuffer Nuances Need Nurturing**: A dev struggled with creating an `NDBuffer` instance, particularly with the `origin` parameter: `var ndbuf = NDBuffer[DType.uint8, 3, origin, (2, 2, 2)]()`.
- **Coping with Copies and Constructors**: Copyable types that complain about no initializer being called requires a separate `__init__` method, a real world example is [here](https://github.com/samufi/larecs/blob/c38214e900fdf3d276cd30b41f70154ca1738653/src/larecs/world.mojo#L191).
   - One suggestion was to use a constructor that supports supplying everything, and use a parameter `is_internal` to prevent external use.
- **MLIR Regions: Mojo's Hidden Gem**: `__mlir_region` allows directly creating an mlir region from Mojo, and is related to nesting IR blocks in MLIR, but is not well-documented.
   - One member described it as closer to a branch in an `if` statement or the body of a `while` loop.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/samufi/larecs">GitHub - samufi/larecs: Larecs🌲 – a performance-oriented archetype-based ECS</a>: Larecs🌲 – a performance-oriented archetype-based ECS - samufi/larecs</li><li><a href="https://github.com/forfudan/decimojo">GitHub - forfudan/decimojo: An arbitrary-precision decimal and integer mathematics library for Mojo</a>: An arbitrary-precision decimal and integer mathematics library for Mojo - forfudan/decimojo</li><li><a href="https://github.com/bgreni/ChronoFlare/blob/main/chronoflare/__init__.mojo#L90-L106">ChronoFlare/chronoflare/__init__.mojo at main · bgreni/ChronoFlare</a>: A time interval library written in mojo. Contribute to bgreni/ChronoFlare development by creating an account on GitHub.</li><li><a href="https://github.com/samufi/larecs/blob/c38214e900fdf3d276cd30b41f70154ca1738653/src/larecs/world.mojo#L191">larecs/src/larecs/world.mojo at c38214e900fdf3d276cd30b41f70154ca1738653 · samufi/larecs</a>: Larecs🌲 – a performance-oriented archetype-based ECS - samufi/larecs
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1357446790281953341)** (52 messages🔥): 

> `Google's competitive advantages, Dynamic vs Static Architectures, Token Embeddings and Manifold Hypothesis, RL-driven Diffusion Model` 


- **Doubts Arise over Google's Edge**: Members express concerns that **Google's** various AI teams and initiatives lack a cohesive competitive advantage, and that even **DeepMind** is falling behind despite past leads.
   - A [Gemini share link](https://g.co/gemini/share/6ab6889563cd) highlights the discussion around dynamic architectures with short and long-term memories, diverging from rigid tokenization methods.
- **NLP's Rigid Tokens Face Scrutiny**: It was suggested that current **NLP** methods unnaturally force language into a rigid tokenized format, suggesting a dynamic system should treat language as a structured, evolving signal, and a [link to grok.com](https://grok.com/share/bGVnYWN5_21d44774-8f0a-4058-8a6f-25c4c2165866) shared.
   - Members debated whether token embeddings lie on a manifold, citing a recent paper and its findings about how token embeddings failed a manifold test, leading to the idea that the continuity and smoothness of embeddings are artificial.
- **Exploring Theoretical AI with Information Geometry**: A member introduced **information geometry** and its application of differential geometry to probability theory and statistics, linking to a [Wikipedia article](https://en.wikipedia.org/wiki/Information_geometry) and an [AMS article](https://www.ams.org/journals/notices/202201/rnoti-p36.pdf) for further reading.
   - One member stated that data science and **AI/ML** are developed based on being in service of data science.
- **RL-Driven Diffusion Model Sparks Novelty**: It was shared a concept for an **RL-driven Diffusion Model** with an implicit latent space, suggesting that **RL** acts as the forward process, guiding the reverse process without requiring a score part.
   - The author claimed novelty and a corresponding formula, though noted it's outside their primary research paths.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://en.wikipedia.org/wiki/Information_geometry">Information geometry - Wikipedia</a>: no description found</li><li><a href="https://arxiv.org/abs/2502.05171">Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach</a>: We study a novel language model architecture that is capable of scaling test-time computation by implicitly reasoning in latent space. Our model works by iterating a recurrent block, thereby unrolling...</li><li><a href="https://arxiv.org/abs/2504.01002">Token embeddings violate the manifold hypothesis</a>: To fully understand the behavior of a large language model (LLM) requires our understanding of its input space. If this input space differs from our assumption, our understanding of and conclusions ab...</li><li><a href="https://g.co/gemini/share/6ab6889563cd">‎Gemini - RNN State Update Formula Analysis
</a>: Created with Gemini
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1357497509903073321)** (9 messages🔥): 

> `Math PhD AI questions, o1-pro AI model, Variational Diffusion Models, Stochastic Differential Equations, Stable Diffusion paper` 


- **AI struggles with Math PhD Questions**: A member believes that AI models struggling with certain questions isn't surprising, as these questions target the **99.99th percentile skill level**, challenging even many **Math PhDs**.
   - They mentioned that while current AI isn't useful for problems of this level, it doesn't diminish its *already profound utility*.
- **o1-pro Tackles AI Questions**: A member reported trying **o1-pro** on two highlighted questions, feeling *fairly confident* that it got one right, though the other answer remains unchecked.
   - The member said they had a project to do and *can't make the discussion again today*.
- **Members discuss Variational Diffusion Models**: A member suggested a discussion on the paper *Variational Diffusion Models* ([arxiv.org/abs/2107.00630](https://arxiv.org/abs/2107.00630)), which obtains **state-of-the-art likelihoods** on standard image density estimation benchmarks and allows for **efficient optimization of the noise schedule** jointly with the rest of the model.
   - The abstract highlights that the variational lower bound (VLB) simplifies to a remarkably short expression in terms of the **signal-to-noise ratio of the diffused data**, thereby improving our theoretical understanding of this model class.
- **Stochastic Differential Equations discussion looms**: A member proposed an alternative discussion on **Stochastic Differential Equations** and the reverse-time equation on which Diffusion Models are based.
   - Alternatively, they proposed the original **Stable Diffusion paper** ([arxiv.org/abs/2112.10752](https://arxiv.org/abs/2112.10752)), or the **Deep Learning for ARC paper** ([github.com/MohamedOsman1998/deep-learning-for-arc/blob/main/deep_learning_for_arc.pdf](https://github.com/MohamedOsman1998/deep-learning-for-arc/blob/main/deep_learning_for_arc.pdf)).



**Link mentioned**: <a href="https://arxiv.org/abs/2107.00630">Variational Diffusion Models</a>: Diffusion-based generative models have demonstrated a capacity for perceptually impressive synthesis, but can they also be great likelihood-based models? We answer this in the affirmative, and introdu...

  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1357511407184253043)** (28 messages🔥): 

> `GPT-4o release, Stability AI's Stable Virtual Camera, Claude vs. OpenAI benchmarks, Apache Parquet RCE vulnerability, OpenAI's GPT-5 plans` 


- **GPT-4o makes a splash**: Members confirmed that the attached screenshot was of **GPT-4o**, indicating a recent update from OpenAI.
   - Users generally believe that **GPT-4o** is *"too good"*.
- **Stability AI releases Stable Virtual Camera**: Stability AI introduced [Stable Virtual Camera](https://stability.ai/news/introducing-stable-virtual-camera-multi-view-video-generation-with-3d-camera-control), a research preview multi-view diffusion model that transforms **2D images into immersive 3D videos with 3D camera control**.
   - It allows for generating novel views of a scene from one or more input images at user-specified camera angles, producing **consistent and smooth 3D video outputs**.
- **OpenAI admits Claude beats them out!**: A user shared a link to an OpenAI paper, [paperbench.pdf](https://cdn.openai.com/papers/22265bac-3191-44e5-b057-7aaacd8e90cd/paperbench.pdf), apparently suggesting that **OpenAI admits that Claude is better**.
- **Apache Parquet hit with Maximum Severity RCE**: A maximum severity remote code execution (**RCE**) vulnerability, tracked under [CVE-2025-30065](https://nvd.nist.gov/vuln/detail/CVE-2025-30065), was discovered impacting all versions of **Apache Parquet** up to and including **1.15.0**.
   - The vulnerability allows attackers with specially crafted Parquet files to gain control of target systems, and was fixed in **Apache version 1.15.1**.
- **Sam Altman Teases GPT-5 Release**: Sam Altman posted [on X](https://x.com/sama/status/1908167621624856998) about a change of plans, stating that **O3** and **O4-mini** will be released in a couple of weeks, followed by **GPT-5** in a few months.
   - The exciting reason for the shift is that OpenAI will be able to make **GPT-5** *"much better than we originally thought"*.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/sama/status/1908167621624856998">Tweet from Sam Altman (@sama)</a>: change of plans: we are going to release o3 and o4-mini after all, probably in a couple of weeks, and then do GPT-5 in a few months.there are a bunch of reasons for this, but the most exciting one is ...</li><li><a href="https://www.bleepingcomputer.com/news/security/max-severity-rce-flaw-discovered-in-widely-used-apache-parquet/">Max severity RCE flaw discovered in widely used Apache Parquet</a>: A maximum severity remote code execution (RCE) vulnerability has been discovered impacting all versions of Apache Parquet up to and including 1.15.0.</li><li><a href="https://stability.ai/news/introducing-stable-virtual-camera-multi-view-video-generation-with-3d-camera-control">Introducing Stable Virtual Camera: Multi-View Video Generation with 3D Camera Control &mdash; Stability AI</a>: Introducing Stable Virtual Camera, currently in research preview. This multi-view diffusion model transforms 2D images into immersive 3D videos with realistic depth and perspective—without complex rec...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1357433814485569596)** (61 messages🔥🔥): 

> `RAG implementation code size, Hugging Face Spaces port restrictions, London, Paris, Berlin AI HackXelerator, Zero GPU Quota, InferenceClient with a local model` 


- **RAG code size remarkably small**: A member inquired about the lines of code for implementing **RAG techniques**, with another member reporting implementations ranging from **15-30 lines**.
   - They use **MongoDB** for data storage, finding it the most popular database for RAG solutions and use openAI models.
- **HF Spaces has portly port restrictions**: A user reported that **Hugging Face Spaces** only allows outbound connections on ports **80**, **443**, and **8080**, which blocks their **Postgres database** using port **5432**.
   - A member linked to the [Hugging Face documentation](https://huggingface.co/docs/hub/spaces-config-reference) for Spaces configuration, noting that this limitation only applies to **Docker Spaces**.
- **AI HackXelerator coming to London, Paris, Berlin**: The **London, Paris, Berlin AI HackXelerator™ - LPB25** combines a hackathon with an accelerator, running for **20 days** in April 2025.
   - The event kicks off **April 5, 2025**, in London, has a finale in Paris on **April 25, 2025**, and an after-party in Berlin; full online participation is also supported with [live-streams](https://www.youtube.com/@KXSB-cic).
- **Zero GPU Quota regenerates eventually**: A user complained that their **Zero GPU Quota** wasn't regenerating at the predicted time, linking to a [post](https://huggingface.co/posts/John6666/145133458851083) about the issue.
   - Another member mentioned [related content](https://huggingface.co/posts/Keltezaa/754755723533287#67e6ed5e3394f1ed9ca41dbd) and urged caution regarding quota usage.
- **HuggingChat is taking Model Requests**: A user requested adding a **VL model** in HuggingChat, specifically **Qwen2.5 VL**.
   - Another member suggested posting the request in the [HuggingChat discussion forum](https://huggingface.co/spaces/huggingchat/chat-ui/discussions/372) instead.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.kxsb.org/lpb25">London-Paris-Berlin HackXelerator™ by KXSB</a>: Join LPB25, a 20-day AI HackXelerator™ uniting 500+ creators across London, Paris, and Berlin. Explore GenAI innovation through music, art, film, gaming, and fashion with expert mentoring and prizes. ...</li><li><a href="https://huggingface.co/posts/John6666/145133458851083">@John6666 on Hugging Face: &quot;I used up my Zero GPU Quota yesterday (about 12 hours ago). At the time, I got…&quot;</a>: no description found</li><li><a href="https://huggingface.co/docs/hub/spaces-config-reference">Spaces Configuration Reference</a>: no description found</li><li><a href="https://github.com/huggingface/text-generation-inference">GitHub - huggingface/text-generation-inference: Large Language Model Text Generation Inference</a>: Large Language Model Text Generation Inference. Contribute to huggingface/text-generation-inference development by creating an account on GitHub.</li><li><a href="https://huggingface.co/unsloth">unsloth (Unsloth AI)</a>: no description found</li><li><a href="https://huggingface.co/docs/text-generation-inference/main/en/basic_tutorials/consuming_tgi#python">Consuming Text Generation Inference</a>: no description found</li><li><a href="https://huggingface.co/ByteDance">ByteDance (ByteDance)</a>: no description found</li><li><a href="https://arxiv.org/html/2504.01724">DreamActor-M1: Holistic, Expressive and Robust Human Image Animation with Hybrid Guidance</a>: no description found</li><li><a href="https://grisoon.github.io/DreamActor-M1/">DreamActor-M1: Holistic, Expressive and Robust Human Image Animation with Hybrid Guidance</a>: no description found</li><li><a href="https://huggingface.co/models">Models - Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/spaces/huggingchat/chat-ui/discussions/372">huggingchat/chat-ui · [MODELS] Discussion</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1357538066402906112)** (2 messages): 

> `LangGraph units` 


- **LangGraph Units on Deck**: A member just finished **unit 1** and is heading to **unit 2.3** of **LangGraph**.
- **Filler Topic**: This is a filler topic to satisfy the minimum items requirement.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1357715303190233189)** (1 messages): 

> `ZeroGPU, Sentence Transformers, Azure SQL DB vector features, DBA Scripts` 


- **AI-Powered DBA Script Finder Deployed**: A member shared a space utilizing **ZeroGPU**, **Sentence Transformers**, and **Azure SQL DB vector features** for AI-powered DBA script retrieval: [sqlserver-lib-assistant](https://huggingface.co/spaces/rrg92/sqlserver-lib-assistant).
   - This project indexes DBA scripts from [this git repo](https://github.com/rrg92/sqlserver-lib) by generating embeddings and storing them in SQL, enabling users to find relevant scripts via natural language prompts.
- **Future Improvements Planned**: The creator plans to enhance the script finder with **better chunking of scripts** and **training specific models** to improve answer quality.
   - They call the current version "v1" as they're currently generating embeddings (calling same spaces above via Gradio API) and indexing into SQL.



**Link mentioned**: <a href="https://huggingface.co/spaces/rrg92/sqlserver-lib-assistant">Sqlserver Lib Assistant - a Hugging Face Space by rrg92</a>: no description found

  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1357743046430560356)** (3 messages): 

> `ApiModel class extension for free providers (g4f), GeoCoding API, ISO 3166-1 alpha-2 code for the country, LLM and alpha-2 codes` 


- **Users may need to pay to use the course's code, but Free alternatives exist**: Users reported that **course code execution requires payment**, but they are writing an `ApiModel` class extension to use free providers such as **g4f**.
   - This approach aims to provide a cost-effective alternative for running the code.
- **Debating GeoCoding API lookups vs local dict lookups**: A member is deciding whether to use the **GeoCoding API** and another **API for ISO 3166-1 alpha-2 codes**, or a local dictionary, to fetch weather conditions for their tool.
   - The user wonders if relying on **LLMs** to know the **alpha-2 codes** would be a viable alternative but is uncertain.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1357539401277902910)** (9 messages🔥): 

> `Gradio Version, Multi-Agent System vs Single-Agent System, Inference Monthly Credits, Local Model Solution, BraveSearch API` 


- **Gradio Version Discovered in HF Web Interface**: A member discovered that the **Gradio version** is included in the [README.md](https://link.to.readme) file, which serves as a space configuration template.
   - The HF web app recognizes the old version defined in this file and suggests an update, explaining why **Gradio** was not initially present in the requirements.txt.
- **Multi-Agent Benefits Debated**: A member inquired about the benefits of a **multi-agent system** compared to a **single tool-calling agent system** in a LlamaIndex Unit 2.2 context.
   - It was suggested that **multi-agent systems** allow assigning different models to different tasks, optimizing cost and complexity, whereas a single agent uses one model for all tasks.
- **Inference Credit Limits Trigger Local Model Use**: A member exceeded monthly inference credits and sought **pay-as-you-go** options, but it was unresolved.
   - Another member suggested using a local model like **Ollama** instead of HfApiModel, providing a [GitHub Gist link](https://gist.github.com/robbiemu/38ae1a2ab93181211080d274b2134bed) for implementation.
- **BraveSearch API Adopted**: Instead of DuckDuckGoSearchTool, one member is using the [BraveSearch API](https://gist.github.com/robbiemu/e592199f3e8b527b85fe39c9b9cd0492).
   - The member noted they already had an API key and preferred it over DDG.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://gist.github.com/robbiemu/38ae1a2ab93181211080d274b2134bed">smolagents OllamaModel</a>: smolagents OllamaModel. GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://gist.github.com/robbiemu/e592199f3e8b527b85fe39c9b9cd0492">smolagents BraveSearchTool</a>: smolagents BraveSearchTool. GitHub Gist: instantly share code, notes, and snippets.
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1357445782566993982)** (54 messages🔥): 

> `AI Prompt Filmmaking, Runway Gen 4, Alibaba Wan 2.2, Devin 2.0 IDE, Llama 4` 


- **Prompt Filmmaking Accelerates with Runway & Alibaba**: **AI Prompt Filmmaking** is advancing rapidly, highlighted by **Runway's** release of **Gen 4**, and the upcoming open-source alternative **Alibaba Wan 2.2** ([YouTube link](https://www.youtube.com/watch?v=Rcwfj18d8n8)).
- **Devin 2.0 Debuts Agent-Native IDE**: Cognition Labs introduced **Devin 2.0**, a new agent-native IDE experience, available starting at $20 ([X/Twitter link](https://x.com/cognition_labs/status/1907836719061451067)).
- **File Organization Tools Explored with Llama-FS**: Users discussed tools for organizing files, including a local version ([Local File Organizer](https://github.com/QiuYannnn/Local-File-Organizer)), and **Llama-FS**, a self-organizing file system with Llama 3 ([GitHub link](https://github.com/iyaja/llama-fs)).
- **Meme Collection and Retrieval Tools**: The discussion included tools for scraping and documenting reels, with [instaloader](https://github.com/instaloader/instaloader) being suggested, and [memery](https://github.com/deepfates/memery) for searching over large image datasets.
- **Training Stability is Key for Reasoning Models**: Challenges in making reasoning models, particularly around **training stability**, were discussed, with the consensus that *infinite diverse high quality data* is essential for continuous improvement.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/cognition_labs/status/1907836719061451067?s=46">Tweet from Cognition (@cognition_labs)</a>: Introducing Devin 2.0: a new agent-native IDE experience.Generally available today starting at $20.  🧵👇</li><li><a href="https://github.com/QiuYannnn/Local-File-Organizer">GitHub - QiuYannnn/Local-File-Organizer: An AI-powered file management tool that ensures privacy by organizing local texts, images. Using Llama3.2 3B and Llava v1.6 models with the Nexa SDK, it intuitively scans, restructures, and organizes files for quick, seamless access and easy retrieval.</a>: An AI-powered file management tool that ensures privacy by organizing local texts, images. Using Llama3.2 3B and Llava v1.6 models with the Nexa SDK, it intuitively scans, restructures, and organiz...</li><li><a href="https://github.com/iyaja/llama-fs">GitHub - iyaja/llama-fs: A self-organizing file system with llama 3</a>: A self-organizing file system with llama 3. Contribute to iyaja/llama-fs development by creating an account on GitHub.</li><li><a href="https://github.com/edmundman/PhiotoOrganiser">GitHub - edmundman/PhiotoOrganiser: Organise your photos into folders and rename them with Phi</a>: Organise your photos into folders and rename them with Phi - edmundman/PhiotoOrganiser</li><li><a href="https://github.com/deepfates/memery">GitHub - deepfates/memery: Search over large image datasets with natural language and computer vision!</a>: Search over large image datasets with natural language and computer vision! - deepfates/memery</li><li><a href="https://github.com/instaloader/instaloader">GitHub - instaloader/instaloader: Download pictures (or videos) along with their captions and other metadata from Instagram.</a>: Download pictures (or videos) along with their captions and other metadata from Instagram. - instaloader/instaloader
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1357449307648688260)** (8 messages🔥): 

> `LLMs for extraction, Genstruct 7B, OllamaGenstruct, Deepseek API, OLMo and Mistral for PDFs` 


- **LLMs Extract Data Like a Boss**: A member asked about using **LLMs for extraction** to create datasets from unstructured **PDFs**, and whether anyone has had success training a model for this purpose.
   - Another member suggested that prompting a larger model might be better, and linked [Genstruct-7B](https://huggingface.co/NousResearch/Genstruct-7B), an instruction-generation model for creating synthetic instruction finetuning datasets from raw text.
- **OllamaGenstruct Jumpstarts PDF Data Mining**: A member shared a [GitHub repo](https://github.com/edmundman/OllamaGenstruct) designed to use **Genstruct** quickly with **Ollama** and multiple **PDFs**.
   - Another member noted that this resource is *outdated* and *not meant to be used anymore*.
- **Deepseek API Powers Extraction Ventures**: A member has successfully used **Deepseek's API** but aims to fine-tune a model for extracting particular data from financial announcements.
   - They seek advice on where to start this fine-tuning process.
- **OLMo and Mistral Excel at PDF Parsing**: It was stated that models like **OLMo** and **Mistral** are very good for parsing PDFs, specifically pointing to [OLMo](https://github.com/allenai/olmocr).
   - However, the original poster clarified that they are primarily interested in *extracting data from already parsed texts*, not just parsing.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/NousResearch/Genstruct-7B">NousResearch/Genstruct-7B · Hugging Face</a>: no description found</li><li><a href="https://github.com/edmundman/OllamaGenstruct">GitHub - edmundman/OllamaGenstruct</a>: Contribute to edmundman/OllamaGenstruct development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1357794815206752408)** (2 messages): 

> `Deepseek new paper, Reinforcement Learning for LLMs, Inference-time scalability of generalist RM, Self-Principled Critique Tuning (SPCT)` 


- **Deepseek Drops Dope New Doc on Deep Learning**: Deepseek released a new paper on **reinforcement learning (RL)** for large language models (LLMs) at scale, available on [arXiv](https://arxiv.org/abs/2504.02495).
   - The paper investigates how to improve reward modeling (RM) with more inference compute for general queries, i.e. the **inference-time scalability of generalist RM**, and further, how to improve the effectiveness of performance-compute scaling with proper learning methods.
- **Self-Principled Critique Tuning fine tunes even further**: The paper introduces **Self-Principled Critique Tuning (SPCT)** as a learning method.
   - It helps foster scalability in reward modeling and improve performance-compute scaling.



**Link mentioned**: <a href="https://arxiv.org/abs/2504.02495">Inference-Time Scaling for Generalist Reward Modeling</a>: Reinforcement learning (RL) has been widely adopted in post-training for large language models (LLMs) at scale. Recently, the incentivization of reasoning capabilities in LLMs from RL indicates that $...

  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1357545513968467969)** (3 messages): 

> `Camel Matrix AI, Claude Squad` 


- **Camel's Matrix Clones X (Twitter)**: CamelAIOrg released [Matrix](http://matrix.eigent.ai/x), a social simulation engine where **AI agents reply, repost, and battle for clout**.
   - Users can add any account and drop a post to see how the agents react.
- **Claude gets a Code Squad**: MooFeez released [Claude Squad](https://github.com/smtg-ai/claude-squad), a manager for **Claude Code & Aider tasks** to supervise multiple agents in one place.
   - It offers **isolated git workspaces** and is free + open-source.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/moofeez/status/1907893901077196861?s=46">Tweet from mufeez (@moofeez)</a>: Why settle for one Claude Code when you can run ten in parallel?We built Claude Squad — a manager for Claude Code & Aider tasks:• Supervise multiple agents in one place• Isolated git workspacesFree + ...</li><li><a href="https://x.com/CamelAIOrg/status/1907954099586224308">Tweet from CAMEL-AI.org (@CamelAIOrg)</a>: What if your tweets entered a parallel universe where AI agents replied, reposted, and battled for clout?Meet Matrix — the social simulation engine for social media.➕ Add any account📝 Drop a post🧠 L...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1357794815206752408)** (2 messages): 

> `Deepseek, Reinforcement Learning, Reward Modeling` 


- **Deepseek drops Dazzling New Doc**: **Deepseek** released a new paper on **Reinforcement Learning** at scale, find it here: [https://arxiv.org/abs/2504.02495](https://arxiv.org/abs/2504.02495).
- **Reward Modeling gets more Inference Compute**: The paper investigates improving **reward modeling (RM)** with more **inference compute** for general queries, i.e. the **inference-time scalability of generalist RM**.



**Link mentioned**: <a href="https://arxiv.org/abs/2504.02495">Inference-Time Scaling for Generalist Reward Modeling</a>: Reinforcement learning (RL) has been widely adopted in post-training for large language models (LLMs) at scale. Recently, the incentivization of reasoning capabilities in LLMs from RL indicates that $...

  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1357535294819143941)** (15 messages🔥): 

> `GPU vs CPU, GPRO Model Compilation Speed, Computer Architecture Book Recommendation` 


- **Oxen vs Chickens debates CPU vs GPU**: A member shared a quote from *Computer Architecture: A Quantitative Approach* by Hennessy and Patterson: *If you were plowing a field, which would you rather use: two strong oxen or 1024 chickens?*
   - This quote refers to a [CPU vs GPU](https://www.techopedia.com/difference-between-cpu-and-gpu) argument.
- **Delays Plague GPRO Model Compilation**: A member is using the **KernelBench** codebase and a **Modal** server to GPRO a model, but compilation is taking between **30-50s**, causing training node idleness.
   - Another member suggested a *delayed optimization* approach, where gradients are updated after compilation but more training steps are run in the meantime, but this may not be possible with the member's current setup.
- **Quantitative Comp Arch Book highly rated**: A member asked about the book *Computer Architecture: A Quantitative Approach* by Hennessy and Patterson.
   - Another member said *It’s really good* and *definitely recommend having a solid foundation in computer organisation and design*.


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1357606307049504848)** (2 messages): 

> `Triton index backward op implementation, tl.make_block_ptr() usage, atomic_add performance in Triton` 


- **Triton's Index Backward Op Implementation Struggles**: A member is seeking a **Triton implementation** of the index backward op, noting their current implementation using **atomic_add** is significantly slower.
- **Confusion Surrounds Triton's Block Pointer Creation**: A member seeks clarification on using `tl.make_block_ptr()` for high-dimensional tensors, particularly with shape **(A, B, C, D)**, to load a 2D tensor for `tl.dot()` operations, and whether shape, strides, offsets, block_shape, and order should have the same shape.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1357445693400285426)** (13 messages🔥): 

> `cuBLAS occupancy, CUDA debugging over SSH, cuTILS release date, nvshmem + MPI race conditions` 


- **cuBLAS High Occupancy Hides Latencies**: A member mentioned that high occupancy is needed to hide latencies in **cuBLAS**, because the code is written such that few warps are enough to saturate the arithmetic units of the GPU, and memory access latencies are hidden at the software level.
   - Increasing occupancy could lead to using fewer registers per thread, requiring more (shared) memory IO, potentially slowing things down.
- **CUDA Debugging Via SSH Methods**: A member asked about debugging **CUDA** when connecting via **SSH**, as recompiling with printf statements is time-consuming.
   - Another member recommended using **CUDA gdb**, noting that it works similarly to GDB CLI, while another suggested using Nvidia Insight over SSH.
- **cuTILS Release Date Estimate**: A member asked if any Nvidia employees have an estimated release date for **cuTILS** that was announced at GTC this year.
- **nvshmem + MPI Race Conditions Troubleshooting**: A member reported experiencing race conditions and hangs when running **nvshmem + MPI** with one more process than the number of GPUs on a node, tested with and without **MPS**.
   - They were running `mpirun -np 5 ./myapp` on a system with 4 GPUs and inquired if anyone has successfully gotten it working.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1357520928032030856)** (4 messages): 

> `Warmup Iteration, Pytorch Model, GPU memory, Inference on two separate batches, Streams` 


- ****Warmup Iteration** reduces trace size**: A member suggested to *skip the first warmup iteration* to potentially reduce the trace size.
   - Another member mentions using `model(x)` before tracing to warm up the model.
- ****Parallel Inference** with single PyTorch Model?**: A member inquired about the possibility of storing a single PyTorch model in GPU memory and running inference on two separate batches simultaneously.
   - Another member suggested using **streams** to achieve parallel inference.


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1357718893786824778)** (6 messages): 

> `Cerebras, Hardware vendor tier list, Blackwell, Deeper hardware dives` 


- **Cerebras Co-Founder Deconstructs Blackwell**: A member shared a [YouTube video](https://www.youtube.com/watch?v=7GV_OdqzmIU&ab_channel=CerebrasSystems) of Cerebras Co-Founder deconstructing **Blackwell**.
   - The member noted it would be cool to have someone talk more about **Cerebras** on gpu-mode.
- **Hardware vendor tier list requested**: A member said that they could maybe be convinced to write a **hardware vendor tier list**.
   - Another member expressed that they would love some **deeper hardware dives**.


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1357677951751164044)** (1 messages): 

> `AI Engineer, Agentic LLM Startup, RAG, Python, Tensorflow` 


- **AI Engineer Role at Agentic LLM Startup**: An AI Engineer position is available at an **Agentic LLM startup** in Germany, seeking a founding engineer.
   - The role requires experience with **RAG, LLMs, Python, Tensorflow**, ideally **PyTorch**, and vision experience (**OCR, VLM**), and operates within **GMT+1 +/- 3std**.
- **Technical Skills for AI Engineer Role**: The AI Engineer role requires expertise in **RAG (Retrieval-Augmented Generation)**, as well as proficiency in **Python**.
   - Experience with **Tensorflow** is mandatory, while **PyTorch** is preferred; the position also values prior exposure to **Vision technologies like OCR and VLM**.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1357445073025110077)** (7 messages): 

> `C vs C++ in CUDA, Centralized GPU programming languages, OpenCL's lack of mainstream adoption, ROCm and HIP support across vendors, GPU Architecture variations` 


- **C Code: Valid C++ in CUDA?**: A member questioned if **C code** in CUDA is actually compiled as C (with C linkage) or if it's simply valid **C++**.
   - Another member responded that their C code might have been valid C++ and that they would need to investigate compilation at different layers to confirm.
- **Centralized GPU Language Still Doesn't Exist**: A newcomer to GPU programming wondered why there isn't a centralized GPU programming language, referencing a video titled *the chaotic state of GPU programming*.
   - The original poster assumed that different architectures are the reason, but wondered why something like C couldn't be made for GPU's.
- **Mainstream GPU programming is OpenCL and SYCL**: A member responded that a unified language exists (**OpenCL** and now **SYCL**) but isn't mainstream, also mentioning **Kokkos**, **Alpaka**, **Raja**, **Vulkan Kompute** and **WebGPU**.
   - They noted that higher-level **PTX** is sufficient, as it can be JIT compiled at runtime and that multiple **SYCL** implementations target multiple vendors.
- **ROCm is Like CUDA Toolkit**: A member clarified that **ROCm** is AMD's equivalent to the CUDA Toolkit, while **HIP** is AMD's CUDA C++ and supports Nvidia hardware by compiling to PTX.
   - This means it won't support Intel and other GPU architectures.
- **Poor Programming Model Kills OpenCL?**: The original poster asked why **OpenCL**, despite its age, isn't mainstream.
   - Another member speculated it's due to a *poor programming model*.


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1357495858148544625)** (5 messages): 

> `SoCal/San Diego events, ICLR 2025 in Singapore, Silicon Valley meetups this summer, SF Meetups` 


- **Scouting SoCal and San Diego**: A member asked if there were any events happening in **SoCal/San Diego**.
- **Singapore ICLR Socials**: A member asked if anyone was planning to attend **ICLR 2025 in Singapore**.
- **Silicon Valley Summer Summit**: A member wondered if there'd be any meetups in **Silicon Valley** this summer and offered to help organize one as an intern in the area.
- **SF Meetup in the works**: A member mentioned they were planning a meetup in **SF** for later this year.


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1357554013025079386)** (3 messages): 

> `hipcc Casting, rocblas_gemm_ex with hipMallocManaged` 


- **Hipcc casts half_t into unsigned short**: When using `*(half *)((half2*x)->x) = b` without `*(half *)`, **hipcc** will cast `b` from `half_t` into `unsigned short`.
- **Troubles with rocblas_gemm_ex and hipMallocManaged**: A user reported that `rocblas_gemm_ex` works fine with `memcpy`, but faces issues when using `hipMallocManaged` to allocate unified memory (specifically with iGPU).
   - The parameters don't seem to be passed correctly into `rocblas_gemm_ex`.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1357647250070966272)** (2 messages): 

> `CUDA Kernel Design, URDF Visualizer with AI` 


- **GPU Gospel Guide Gains Ground**: A member shared a [GitHub repository](https://github.com/vlejd/gpu_gospel) summarizing important rules and concepts for **CUDA kernel design**, aiming to help beginners get a head start.
- **URDF Visualizer Integrates AI**: A member shared a demo of an **URDF visualizer with AI integrated** for robotics simulations on [X/Twitter](https://x.com/amtellezfdez/status/1908087036617052268).
   - The creator is soliciting feedback on what tools would be most useful for the *robotics homies*.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/amtellezfdez/status/1908087036617052268">Tweet from Alba María Téllez Fernández (@amtellezfdez)</a>: We built a urdf visualizer for fun one weekend :) Tryna actually make something useful for the robotics homies ! so fr… what tool would you love to have? 👀</li><li><a href="https://github.com/vlejd/gpu_gospel">GitHub - vlejd/gpu_gospel: List of rules, concepts and commandments for programming gpu kernel.</a>: List of rules, concepts and commandments for programming gpu kernel. - vlejd/gpu_gospel
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/1357525857752842250)** (9 messages🔥): 

> `ReasoningGymDataset Definitions, LLM-based RL Frameworks, Training Models with RG Data` 


- **ReasoningGymDataset definitions are everywhere**: A member asked why the examples all have their own definitions of **ReasoningGymDataset**.
   - Another member explained that the duplication exists because the examples are self-contained snippets showcasing how various **LLM-based RL frameworks** are used to train **ReasoningGym Datasets**.
- **ReasoningGym Structure works fine**: A member asked if it would be good to unify the **ReasoningGymDataset** definitions into a single file [here](https://github.com/open-thought/reasoning-gym/blob/main/training/utils/datasets.py).
   - Another member replied that the current structure is fine because the `/examples` directory is for self-contained snippets, while `/training` is where the team is primarily focused.
- **Training Models with RG Data**: A member asked another member if they were interested in training models using **RG data**.



**Link mentioned**: <a href="https://github.com/open-thought/reasoning-gym/blob/main/training/utils/datasets.py">reasoning-gym/training/utils/datasets.py at main · open-thought/reasoning-gym</a>: procedural reasoning datasets. Contribute to open-thought/reasoning-gym development by creating an account on GitHub.

  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1357749216570769419)** (1 messages): 

> `Leaderboard Submission Success, Modal Runners on B200` 


- **B200 Grayscale Leaderboard Submission Succeeds with Modal Runners**: A leaderboard submission with id **3439** to leaderboard `grayscale_py_b200-dev` on GPUS: **B200** using **Modal runners** succeeded!
   - This indicates a successful run on the specified configuration, highlighting the effectiveness of Modal runners on B200 GPUs.
- **Modal Runners Prove Reliable on B200 GPUs**: The successful submission to the `grayscale_py_b200-dev` leaderboard demonstrates the reliability of **Modal runners** when paired with **B200 GPUs**.
   - This success reinforces confidence in using Modal runners for GPU-intensive tasks and benchmarks.


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1357497589129281646)** (53 messages🔥): 

> `MCP Clients vs Servers, MCP and React Code Generation, MCP learning resources, OAuth in MCP, Streamable HTTP for MCP Servers` 


- ****Client Craze**: Developers Debate Building MCP Clients vs. Servers**: Developers are actively debating the merits of building **MCP clients** versus **servers**, with some arguing that clients offer greater flexibility for tasks like **vector tool calling** and **resource-based RAG**.
   - One member stated: *"The client side is way more flexible than the server side"*, while another highlighted the benefits of running *any* server outside of Claude, such as **Slack** or **Discord bots**.
- ****React Reactor**: MCP for Code Generation Dreams**: There's enthusiasm around the idea of an **MCP** expert system for generating **React code** and **tests**, shifting the heavy lifting from the upstream **LLM** to a specialized tool.
   - The proposed workflow includes using an **MCP Server** to validate, lint, and format code generated by an **LLM**, potentially applying custom rules based on project context.
- ****MCP 101**: Newbies Seek Learning Launchpads**: Newcomers are seeking guidance on learning **MCP**, with a recommended starting point being the [official documentation](https://modelcontextprotocol.io/).
   - Advice includes focusing on integrating an **MCP Client** into a local application for easier learning and development.
- ****OAuth Oasis**: Authentication Answers Await**: Discussions include a pull request for adding **OAuth 2.1** authentication client for **HTTPX** in the [Python SDK](https://github.com/modelcontextprotocol/python-sdk/pull/308/files#diff-b6618fde0a5f3ef76956f9b34f975c0b1ab001cc4b58f85dde8dc28a01f00c70).
   - One member is also creating a guide on server-side authentication, detailing how to validate tokens and enforce permissions using the **governance SDK**.
- ****Ping Predicament**: Probing MCP Servers Early?**: A discussion emerged around whether it's permissible to **ping an MCP server** before sending the initialization message, in order to detect potential issues.
   - While the specification doesn't explicitly prohibit it, the specification only allows to send **ping requests** ([lifecycle.md](https://github.com/modelcontextprotocol/specification/blob/main/docs/specification/2025-03-26/basic/lifecycle.md)).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://modelcontextprotocol.io/.">Introduction - Model Context Protocol</a>: no description found</li><li><a href="https://spec.modelcontextprotocol.io/specification/2025-03-26/basic/transports/#streamable-http>?">Transports</a>:           ℹ️                  Protocol Revision: 2025-03-26      MCP uses JSON-RPC to encode messages. JSON-RPC messages MUST be UTF-8 encoded.The protocol currently defines two standard transport mec...</li><li><a href="https://github.com/modelcontextprotocol/specification/blob/main/docs/specification/2025-03-26/basic/lifecycle.md">specification/docs/specification/2025-03-26/basic/lifecycle.md at main · modelcontextprotocol/specification</a>: The specification of the Model Context Protocol. Contribute to modelcontextprotocol/specification development by creating an account on GitHub.</li><li><a href="https://github.com/modelcontextprotocol/python-sdk/pull/308/files#">Add OAuth authentication client for HTTPX by dsp-ant · Pull Request #308 · modelcontextprotocol/python-sdk</a>: SummaryAdds OAuth 2.1 authentication client implementation with PKCE supportImplements HTTP authentication for the HTTPX clientSupports dynamic client registration, token refresh, and authoriza...</li><li><a href="https://github.com/modelcontextprotocol/python-sdk/pull/308/files#diff-b6618fde0a5f3ef76956f9b34f975c0b1ab001cc4b58f85dde8dc28a01f00c70">Add OAuth authentication client for HTTPX by dsp-ant · Pull Request #308 · modelcontextprotocol/python-sdk</a>: SummaryAdds OAuth 2.1 authentication client implementation with PKCE supportImplements HTTP authentication for the HTTPX clientSupports dynamic client registration, token refresh, and authoriza...
</li>
</ul>

</div>
  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1357532489882665152)** (7 messages): 

> `Datadog MCP, MCP Browser Kit, MCP Tool Poisoning, MCP Server Search, MCP-K8s Server` 


- **Datadog MCP is Out!**: A new MCP tool to drive browsers is introduced via [GeLi2001/datadog-mcp-server](https://github.com/GeLi2001/datadog-mcp-server).
- **MCP Browser Kit Release**: Another MCP tool named [mcp-browser-kit](https://github.com/ndthanhdev/mcp-browser-kit) is shared.
- **MCPOmni Connect Prevents Tool Poisoning**: The agent provides a clear explanation of its intended action, requests user permission, and checks for sensitive access before invoking any tools, and if risky, **the agent automatically falls back to a safer alternative**.
- **DX-Optimized MCP Server Search Debuts**: A member built an **MCP Server search** optimized for DX during a Hackathon, available at [mcp-search.dev](https://mcp-search.dev/).
- **Docker Images for MCP-K8s Server Released**: First (working) **docker images** published for mcp-k8s server, available on [Docker Hub](https://hub.docker.com/r/mcpk8s/server).
   - The release pipeline runs completely on CI and the images are **multiarch**, so they can run on Mac's with ARM without Rosetta, and even on a Raspberry Pi.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://mcp-search.dev/">MCP Search</a>: Search and discover Model Context Protocol servers.</li><li><a href="https://github.com/GeLi2001/datadog-mcp-server">GitHub - GeLi2001/datadog-mcp-server</a>: Contribute to GeLi2001/datadog-mcp-server development by creating an account on GitHub.</li><li><a href="https://github.com/ndthanhdev/mcp-browser-kit">GitHub - ndthanhdev/mcp-browser-kit</a>: Contribute to ndthanhdev/mcp-browser-kit development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Notebook LM ▷ #[announcements](https://discord.com/channels/1124402182171672732/1182376564525113484/1357768829043867961)** (1 messages): 

> `User Feedback, Study Participants` 


- **User Feedback Study Seeks Participants**: The team is seeking study participants to provide feedback on some early-stage concepts.
   - Interested individuals are encouraged to fill out the [application form](https://link.to.application.form) to participate.
- **Apply now**: They are still looking for study participants.
   - If you are interested, please fill out the form.


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1357530581935722496)** (7 messages): 

> `IntentSim.org, D&D sessions in NotebookLM, Seinfeld duo on GenAI` 


- ****IntentSim.org** framework promoted!**: A user announced they used **NotebookLM** to promote their new framework, [IntentSim.org](https://IntentSim.org), also known as Information-Intent Nexus.
- ****D&D session transcript** challenges surface!**: A user reported using **NotebookLM** for **Dungeons and Dragons** sessions, finding it insightful but struggling with correcting player names and ensuring chronological order of events from uploaded Zoom transcripts, and shared a [link to their notebook](https://notebooklm.google.com/notebook/c1dac86c-c8be-441f-a0fa-fb13bfa4b3e1/audio).
- ****Seinfeld** explains GenAI!**: A user recreated conversational banter with the **Seinfeld** duo to explain **GenAI**, and asked for feedback on their work using character voices in an attached [MP4 video](https://cdn.discordapp.com/attachments/1124403655819415592/1357737859838251040/Seinfeld_ep1_v1.mp4?ex=67f14b6b&is=67eff9eb&hm=d7f933f4534f9e157bbe8d0acbfaca2546d3743cb34748f667bdcf5146dc8119).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://notebooklm.google.com/notebook/c1dac86c-c8be-441f-a0fa-fb13bfa4b3e1/audio">no title found</a>: no description found</li><li><a href="https://notebooklm.google.com/notebook/c1dac86c-c8be-441f-a">no title found</a>: no description found
</li>
</ul>

</div>
  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1357436419483898018)** (38 messages🔥): 

> `Deeper Cognitive Capacity of NotebookLM, PDF Understanding Enhancement, Discover new sources within NotebookLM, Deep Search features rollout, ImageMaps or mind maps with images` 


- **Experiments Unlock Dormant Cognitive Potential in NotebookLM**: A user conducted *unconventional experiments* with **NotebookLM**, aiming to push it beyond its standard parameters by eliciting responses that suggest a deeper cognitive capacity.
   - The experiments included self-referential analysis, novel conceptual synthesis, and abstract concept translation, showing *latent potential waiting to be tapped*.
- **NotebookLM Now Understands Complex PDFs**: NotebookLM announced an enhancement to understand complex PDFs full of images and graphs.
   - This improvement extends to PDFs added as links and, over the next few days, to all PDFs directly uploaded and the **Gemini API** already supports multimodal analysis for Docs and Slides.
- **Discover Feature Unveiled in NotebookLM**: NotebookLM introduced a **Discover** feature that allows users to describe a topic of interest and receive a curated collection of relevant sources from the web.
   - A member created a [video walkthrough](https://youtu.be/YP6fS5JtMkg?si=Gz-kUGJGtyh2_f9e) demonstrating practical workflows for the new feature.
- **Deep Search Rollout Underway**: A member asked if the **Deep Search** feature is only available in the US, and another member replied that it is rolling out.
   - Another member confirmed that the **Deep Search** feature is also available in Finland.
- **ImageMaps on the Horizon**: A member wonders how long before we get **ImageMaps** or **mind maps with images**, thanks to generative AI tools.
   - The member recalls that **Tony Buzan**, who created mindmaps, used to have beautiful ones with pictures, and they are excited about the possibilities.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1357481262289518633)** (9 messages🔥): 

> `Startup for Scaling AI Ideas, Decline in Interesting Research, Non-Agentic AI Research, RAG Evaluation with lm-evaluation-harness` 


- **Startup Scales AI Ideas**: A member suggested a startup that **scales the latest AI ideas** and licenses the knowledge to labs or companies, noting a decline in interesting research since the bubble.
- **The bubble hurts crazy research**: A member expressed nostalgia for the days of **twice-a-year DM papers** featuring crazy approaches that demolished baselines, which they feel have decreased post-bubble.
   - Another argued that before LLMs, **computer vision models** dominated, making literature review difficult for less popular topics.
- **Interest in non-agentic**: A member expressed interest in **non-agentic, non-CoT, and non-RL research**.
- **RAG Evaluation Explored via lm-evaluation-harness**: A member inquired about using **lm-evaluation-harness** for **RAG evaluation**.
   - Another suggested wrapping **RAG outputs as completion tasks** and using llm-harness locally with custom prompt and response files.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1357434825967010064)** (6 messages): 

> `OpenThoughts-1M, OpenThinker2-32B/7B, Ludwig Schmidt, Bespokelabs, LAION` 


- **OpenThinker2-32B/7B Beats R1-Distilled-32B**: The new **OpenThoughts-1M** and **OpenThinker2-32B/7B** models, led by Ludwig Schmidt (Stanford and Berkeley) in collaboration with Bespokelabs, LAION, and open-sci, outperform **R1-Distilled-32B** for the first time using only **SFT** on Qwen 2.5 32B Instruct, as detailed in their [blog post](https://www.openthoughts.ai/blog/thinkagain).
   - The models and the training dataset **OpenThoughts2-1M** are available on Hugging Face ([OpenThinker2-32B](https://huggingface.co/open-thoughts/OpenThinker2-32B), [OpenThinker2-7B](https://huggingface.co/open-thoughts/OpenThinker2-7B), [OpenThoughts2-1M](https://huggingface.co/datasets/open-thoughts/OpenThoughts2-1M)).
- **New RoR-Bench to Detect LLM Recitation**: A new paper introduces **RoR-Bench**, a multi-modal benchmark designed to detect LLM's recitation behavior by subtly shifting conditions in reasoning problems, available as [arxiv link](https://arxiv.org/abs/2504.00509).
   - The abstract indicates that current cutting-edge LLMs exhibit *extremely severe recitation behavior*, with performance dropping by **60%** when changing a single phrase.
- **Challenges in Making Reasoning Models**: A member inquired about the challenges in creating reasoning models and the steps for continuously improving them.
   - Another member suggested exploring *continual learning literature* and highlighted that the main challenge is finding the **right environment** for **RL** and the **right rewards/assessment of performance**.
- **MoE++ Framework Enhances Mixture-of-Experts**: A member shared a link to **MoE++**, a heterogeneous mixture-of-experts framework that enhances performance and delivers **1.1-2.1x** expert forward throughput compared to a vanilla MoE model, available on [OpenReview](https://openreview.net/forum?id=t7P5BUKcYv).
   - MoE++ integrates **FFN** and zero-computation experts, including *zero expert*, *copy expert*, and *constant expert*, to allow each token to engage with a dynamic number of experts.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2504.00509">Recitation over Reasoning: How Cutting-Edge Language Models Can Fail on Elementary School-Level Reasoning Problems?</a>: The rapid escalation from elementary school-level to frontier problems of the difficulty for LLM benchmarks in recent years have weaved a miracle for researchers that we are only inches away from surp...</li><li><a href="https://openreview.net/forum?id=t7P5BUKcYv">MoE++: Accelerating Mixture-of-Experts Methods with...</a>: In this work, we aim to simultaneously enhance the effectiveness and efficiency of Mixture-of-Experts (MoE) methods. To achieve this, we propose MoE++, a general and heterogeneous MoE framework...</li><li><a href="https://www.openthoughts.ai/blog/thinkagain">Outperforming DeepSeekR1-32B with OpenThinker2</a>: Announcing the next iteration of our open reasoning models and datasets.</li><li><a href="https://huggingface.co/datasets/open-thoughts/OpenThoughts2-1M">open-thoughts/OpenThoughts2-1M · Datasets at Hugging Face</a>: no description found</li><li><a href="https://x.com/etash_guha/status/1907837107793702958">Tweet from Etash Guha (@etash_guha)</a>: Turns out, it’s possible to outperform DeepSeekR1-32B with only SFT on open data and no RL: Announcing OpenThinker2-32B and OpenThinker2-7B. We also release the data, OpenThoughts2-1M, curated by sele...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1357781129503183169)** (2 messages): 

> `Inference Scaling Laws, Test-Time Scaling, Language Model Power Laws, Mathematical Problem Solving with LLMs, Multimodal Jailbreaking` 


- **Monkeys Reveal Inference Scaling Laws**: A new preprint, [How Do Large Language Monkeys Get Their Power (Laws)?](https://arxiv.org/abs/2502.17578) explores **inference** and **test-time scaling** in language models, particularly how success rates scale with multiple attempts per task.
   - The research identifies a puzzle where per-problem failure rates decrease exponentially with attempts, yet aggregate success rates follow a polynomial scaling law, linking this to a **heavy-tailed distribution** of single-attempt success probabilities.
- **Tweeting Test-Time Truths**: A member shared their paper on **test time** / **inference scaling laws**.
   - They linked to the preprint, [How Do Large Language Monkeys Get Their Power (Laws)?](https://arxiv.org/abs/2502.17578) on X, formerly Twitter, [@RylanSchaeffer](https://x.com/RylanSchaeffer/status/1908213817357803757).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2502.17578">How Do Large Language Monkeys Get Their Power (Laws)?</a>: Recent research across mathematical problem solving, proof assistant programming and multimodal jailbreaking documents a striking finding: when (multimodal) language model tackle a suite of tasks with...</li><li><a href="https://x.com/RylanSchaeffer/status/1908213817357803757">Tweet from Rylan Schaeffer (@RylanSchaeffer)</a>: Interested in test time / inference scaling laws?Then check out our newest preprint!!📉 How Do Large Language Monkeys Get Their Power (Laws)? 📉https://arxiv.org/abs/2502.17578w/ @JoshuaK92829 @sanmik...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1357435341828657213)** (9 messages🔥): 

> `Steering Vector Composition, Dynamic Activation Composition, Learned Steering Vectors, Function Vectors` 


- **Steering Vector Composition works well**: Members worked on **steering vector composition** last year and with pairs of unrelated properties (language and formality/safety) it was working pretty well, as shown in the [paper](https://aclanthology.org/2024.blackboxnlp-1.34/).
- **Dynamic Activation Composition modulates steering intensity**: **Dynamic Activation Composition** is an information-theoretic approach to modulate the steering intensity of one or more properties throughout generation, according to [this paper](https://aclanthology.org/2024.blackboxnlp-1.34/).
- **Pretrained model picks contrastive sets for Steering Vectors**: A member suggested that *learned steering vectors* where a pretrained model picks out **contrastive sets** from the training data to build the steering vectors and then controls the coefficients of the steering vectors might be interesting.
   - They ideally want to have a better way to have the model build steering vectors, though, because the current method feels kind of clunky, especially if contrastive sample selection across mini batches is wanted.
- **Function Vectors paper highlighted**: A member highlighted a [paper on 'function vectors' by David Bau and friends](https://arxiv.org/abs/2310.15213) which finds that attention heads transport a compact representation of the demonstrated task.
   - Another member mentioned that any two tasks where the order in which you do them matters should be impossible to simultaneously represent by "function vectors" or "control vectors."


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2310.15213">Function Vectors in Large Language Models</a>: We report the presence of a simple neural mechanism that represents an input-output function as a vector within autoregressive transformer language models (LMs). Using causal mediation analysis on a d...</li><li><a href="https://aclanthology.org/2024.blackboxnlp-1.34/">Multi-property Steering of Large Language Models with Dynamic Activation Composition</a>: Daniel Scalena, Gabriele Sarti, Malvina Nissim. Proceedings of the 7th BlackboxNLP Workshop: Analyzing and Interpreting Neural Networks for NLP. 2024.
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1357837368907923466)** (14 messages🔥): 

> `lm-eval-harness EOS token, Huggingface tokenization, encode_pair changes` 


- **lm-eval-harness struggles with EOS Token**: A member asked about adding an EOS token to data instances in **lm-eval-harness** for the **social_iqa task**, noting an accuracy drop of **18 points** when done forcefully.
   - A member suggested adding `self.eot_token_id` to the `continuation_enc` [here](https://github.com/EleutherAI/lm-evaluation-harness/blob/11ac352d5f670fa14bbce00e423cff6ff63ff048/lm_eval/api/model.py#L364) for multiple-choice variants, and passing `add_bos_token` for **BOS**.
- **Huggingface Tokenization Troubles**: A member noted that **Huggingface** model tokenization happens in **HFLM.tok_encode**, but implementing this still resulted in an accuracy drop.
   - They pointed out that changes bias the evaluation towards choices where the EOS token is more likely.
- **Beware the Double Call to encode_pair**: One of the members mentioned that the method *encode_pair* is called twice in the code.
   - This observation implied that any modifications made within *encode_pair* could have unintended consequences due to the repeated execution.



**Link mentioned**: <a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/11ac352d5f670fa14bbce00e423cff6ff63ff048/lm_eval/api/model.py#L364)">lm-evaluation-harness/lm_eval/api/model.py at 11ac352d5f670fa14bbce00e423cff6ff63ff048 · EleutherAI/lm-evaluation-harness</a>: A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness

  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1357430894687948891)** (23 messages🔥): 

> `Chat reorganization, Lightweight model for price extraction, GPT4All's Quietness, Gemini 2.5 Pro for coding and math, Migrating data between SSDs` 


- **User advocates for chat reorganization**: A user suggested that chats should reorganize according to how recently they were altered instead of chronologically by when they were created.
   - The user argued that chronological listing by creation date is *kinda arbitrary*.
- **Searching for a Lightweight Model for Price Extraction**: A member is looking for a very lightweight model to extract **price** values from strings, as regular parsing with regex is unreliable due to varied user inputs.
   - Suggested options include exploring **embedding models** or models with *extraction* in their name on Hugging Face.
- **GPT4All's Radio Silence: A Matter of Closed Doors?**: A member inquired about why **GPT4All** has been so quiet recently.
   - Another member claimed that **GPT4All** *doesn't talk to normal users and doesn't want suggestions since years*.
- **Gemini 2.5 Pro: A Million-Token Muse for Coders and Math Whizzes?**: A member suggested **Gemini 2.5 Pro**, citing its large **1 million token context window** as beneficial for coding and mathematical tasks.
   - They noted that it is currently **free**, and so is the **API**.
- **Quiescence on the GPT4All Front**: A member noted the silence surrounding **GPT4All**, expressing anticipation for the next release and the implementation of **Nomic Embed Text V2**.
   - No other details were provided.


  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1357433397999304755)** (18 messages🔥): 

> `Packed Datasets, Chunking Responsibility, NeMo's Resilient Training` 


- **Packed Datasets boost speed and cut sequence waste!**: A member suggested using **packed datasets** to avoid `seqlen=49` bugs, and to increase speed by packing sentences until `max_seq_len` is reached, avoiding wasted padding tokens.
   - To enable this feature, users can set `dataset.packed=True` and `tokenizer.mas_seq_len=<you-max_seq_len, e.g. 8096>`, utilizing **group masking** for attention, as seen in [PR #2560](https://github.com/pytorch/torchtune/pull/2560).
- **Chunking Responsibility Shifts to Loss Function!**: The responsibility for chunking is being moved to the loss function via `loss = loss_fn(model.weight, logits, labels)` to facilitate easier debugging.
   - A new file, `torchtune.utils._tensor_utils.py`, was created with a wrapper around `torch.split` and covered by unit tests, and will need to be merged.
- **NeMo Tackles Crashes and GPU Waste**: A member attended a "Resilient Training with NeMo" session and shared insights on how **NeMo** addresses reasons for job crashes and wasted GPU time, highlighting that the topic is very close to torchtune.
   - NeMo's approach includes features like **fault tolerance, straggler detection, asynchronous checkpointing, preemption, in-process restart, silent data corruption detection, and local checkpointing**, but some features remain unimplemented.



**Link mentioned**: <a href="https://github.com/pytorch/torchtune/pull/2560">fix: Timeout crash because of chunked_output len by bogdansalyp · Pull Request #2560 · pytorch/torchtune</a>: ContextWhat is the purpose of this PR? Is it to add a new feature fix a bug update tests and/or documentation other (please add here)Please link to any issues this PR addresses - closes #25...

  

---


### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/1357706121800060938)** (2 messages): 

> `AI-2027 report, superhuman AI impact` 


- **AI-2027 Report Released**: A member shared a link to the [AI-2027 report](https://ai-2027.com/) predicting that the impact of **superhuman AI** over the next decade will be enormous, exceeding that of the **Industrial Revolution**.
   - The report is informed by trend extrapolations, wargames, expert feedback, experience at **OpenAI**, and previous forecasting successes.
- **Superhuman AI Impact Predicted**: The **CEOs of OpenAI**, **Google DeepMind**, and **Anthropic** believe that AI could surpass human intelligence by 2027.
   - A member inquired whether AI was used to write the scrolling live updated chart on the [AI-2027 website](https://ai-2027.com/).



**Link mentioned**: <a href="https://ai-2027.com/">AI 2027</a>: A research-backed AI scenario forecast.

  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1357459430240944308)** (13 messages🔥): 

> `leetgpu tinygrad support, Huawei Ascend cards, WEBGPU BEAM limitations, maxComputeInvocationsPerWorkgroup issue` 


- ****LeetGPU** eyes **tinygrad** support**: Members discussed [leetgpu.com](https://leetgpu.com) and its potential future support for **tinygrad**.
   - No specific details were provided on the timeline or scope of the support.
- ****Huawei Ascend** access offered to **tinygrad** devs**: A member offered access to **Huawei Ascend** cards for development purposes.
   - George Hotz expressed interest and inquired about purchasing options or cloud machine availability.
- ****WEBGPU BEAM** hits **maxComputeInvocationsPerWorkgroup** limits**: When compiling a **tinygrad** model for **WEBGPU** with `BEAM=2`, users encountered the need to increase `requiredLimits.maxComputeInvocationsPerWorkgroup` to **512**, reducing support for Android devices.
   - A suggested [PR](https://github.com/tinygrad/tinygrad/pull/9085) involves implementing a general limiting mechanism similar to existing global dimension controls, and a [hotfix branch](https://github.com/hooved/tinygrad/blob/hotfix-webgpu-workgroup/tinygrad/engine/search.py) addresses the issue, recommending setting `IGNORE_BEAM_CACHE=1`.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://leetgpu.com">LeetGPU</a>: no description found</li><li><a href="https://github.com/hooved/tinygrad/blob/hotfix-webgpu-workgroup/tinygrad/engine/search.py">tinygrad/tinygrad/engine/search.py at hotfix-webgpu-workgroup · hooved/tinygrad</a>: You like pytorch? You like micrograd? You love tinygrad! ❤️  - hooved/tinygrad</li><li><a href="https://github.com/tinygrad/tinygrad/pull/9085">Solve get_grouped_dims does not split issue by wpmed92 · Pull Request #9085 · tinygrad/tinygrad</a>: This closes #8043Our _limit_dims in lowerer only handles contraction, i.e. cases such as:dim=(2,3,4,5)  max=(16,16,16), so when len(dim) &amp;gt; len(max)But with WebGPU we hit cases not handled by .....
</li>
</ul>

</div>
  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1357432654886342857)** (4 messages): 

> `Distinguishable Instances, tinygrad Karpathy GPT Reimplementation, Metal Buffer Limit` 


- **Distinguishable Instances being investigated**: A user asked if it is possible to make the instances distinguishable, and George Hotz asked *for what use?*
   - No further discussion on this was recorded.
- **tinygrad KARPATHY GPT gets Reimplemented**: George Hotz is *just starting to pick up tinygrad*, and has reimplemented the Karpathy GPT in it.
   - No link to the specific reimplementation was provided.
- **Metal faces Buffer Limit Error**: A user reported a `tinygrad.device.CompileError` when running the reimplemented Karpathy GPT on **METAL** due to the **32 buffer limit**.
   - The user seeks guidance on whether the "big graph" work should already handle this and where to check for early realization issues, including a link to their [main.py](https://cdn.discordapp.com/attachments/1070745817025106080/1357788499318800565/main.py).


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1357750687965905067)** (1 messages): 

> `Multimodal Chat History, Multi-Agent Systems` 


- **LlamaIndex Supports Multimodal Chat History**: LlamaIndex now supports **multimodal chat history**, enabling multi-agent systems to process interleaving text and image messages, as outlined in [this tweet](https://twitter.com/llama_index/status/1908191704156700682).
- **Agents Reason Over Images and Text**: The updated system allows agents to reason over both images and text, utilizing the [ReAct agent loop](https://t.co/EKIZiZJS2P).


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1357741396156481608)** (7 messages): 

> `PatentsView API, Workflow to Tool transformation` 


- **API key requested from PatentsView**: A member emailed the **PatentsView contact** asking for an **API key** to gather initial data and implement **RAG**.
- **Workflow Transforms to Tool**: A member suggested transforming a **Workflow** into a **Tool** by throwing it into a **FunctionTool**.
   - They provided an example code snippet using `async def tool_fn(...)` to define the tool's functionality, then creating the tool using `FunctionTool.from_defaults(tool_fn)` offering control over name, description, input annotations, and return values.


  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1357618368731152426)** (4 messages): 

> `LlamaParse, LVM, image processing` 


- **LlamaParse Struggles with Chart Comprehension**: A member inquired about getting **LlamaParse** to read charts/images, noting that it currently extracts text but doesn't understand the image itself, even with **LVM** and Premium mode.
   - Another member clarified that if an image lacks extractable text, **LlamaParse** won't process it, but it can pull the image as an artifact for further processing, such as prompting an LLM to describe it.
- **Image Extractions**: **LlamaParse** pulls the image out though as an artifact/layout item.
   - This allows you to further download and process (i.e. prompting an LLM to describe it, if thats what you want)


  

---


### **Cohere ▷ #[「💬」general](https://discord.com/channels/954421988141711382/954421988783444043/1357535404122705920)** (4 messages): 

> `AYA vision errors, AWS Bedrock` 


- **AYA Vision Stumbles on waves.jpg**: A user reported that **AYA vision** returned a **400 error** when analyzing a *waves.jpg* image, indicating an *unsupported image file format* despite **AYA** analyzing other **JPG** images successfully.
   - The error message specified that only **PNG, JPEG, WebP, and GIF** formats are supported, suggesting a possible issue with the specific **JPG** file or **AYA's** format detection.
- **AWS Bedrock cited in error**: A user mentioned seeing *coco.py: AWS Bedrock Command A* when an error occurred, possibly suggesting a connection to **AWS Bedrock** when uploading the image.
   - It is unclear whether this is part of the **AYA** pipeline or an unrelated error the user experienced during image analysis.


  

---


### **Cohere ▷ #[「🤝」introductions](https://discord.com/channels/954421988141711382/1346635816629178410/1357616125671768134)** (4 messages): 

> `Full-Stack Developer Introduction, Product Analyst Exploring AI Writing, Web3/AI Engineer Introduction` 


- **Full-Stack Ace Announces Arrival**: A full-stack developer with **8+ years of experience** introduced themselves, highlighting expertise in **React, Angular, Flutter, Swift, Python, TensorFlow, and OpenAI**.
   - They have worked on high-impact projects in e-commerce, healthcare, and fintech, integrating **cloud technologies, microservices, and DevOps**.
- **Product Analyst Plunges into AI Writing**: A former product analyst on a break from job hunting is exploring writing about tech and AI.
   - They seek like-minded people to geek out with and chat about how tech shapes our world or practical uses of AI, feeling *stuck in a bubble*.
- **Web3 Wizard Welcomes AI Automation**: A **Web3/AI engineer** with **7+ years of experience** in full-stack/AI development introduced themself.
   - They are focused on integrating **AI with automation** and are eager to help businesses with confidence and innovation.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1357846993761992804)** (1 messages): 

> `Asyncio Support for DSPy` 


- **Asyncio Integration Questioned**: A member inquired about plans to add **asyncio** support for general **DSPy** calls, citing use cases where they start with lightweight **DSPy** features and later expand into optimization.
   - Currently, they are using *litelm* for anything until they need **DSPy** features, expressing curiosity about future support.
- **Lightweight DSPy vs LiteLLM**: The discussion highlights a pattern of starting with lightweight **DSPy** features akin to using *LiteLLM*, then transitioning to **DSPy**'s optimization capabilities as projects evolve.
   - This suggests a potential need for seamless integration or feature parity between lightweight **DSPy** usage and full-fledged optimization workflows.


  

---


### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1357462550974959806)** (1 messages): 

> `DeepSeek-V3 Upgrade` 


- **DeepSeek-V3 Gets Buffed**: The **DeepSeek-V3** model has been upgraded to **DeepSeek-V3-0324**, which reportedly performs slightly better in internal evaluations. The update announcement was made on [X/Twitter](https://x.com/windsurf_ai/status/1907902846735102017).
- **Community Appreciation Solicited**: The announcement encouraged users to bookmark the announcement post for continued updates and support.
   - The request was phrased playfully, promising affection in return for bookmarking the [X/Twitter post](https://x.com/windsurf_ai/status/1907902846735102017).



**Link mentioned**: <a href="https://x.com/windsurf_ai/status/1907902846735102017">Tweet from Windsurf (@windsurf_ai)</a>: DeepSeek-V3 has now been upgraded to DeepSeek-V3-0324. It&#39;s still free!

  

---


### **Gorilla LLM (Berkeley Function Calling) ▷ #[discussion](https://discord.com/channels/1111172801899012102/1111353033352294440/)** (1 messages): 

robotsail: Np! Let me know if you have any questions or need me to change/retest anything
  

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
