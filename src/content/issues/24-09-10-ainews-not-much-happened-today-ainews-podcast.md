---
id: 4166fea7-d8ed-4606-bb4c-3d44b0f7aa12
title: not much happened today + AINews Podcast?
date: '2024-09-11T02:24:16.042126Z'
original_slug: ainews-not-much-happened-today-ainews-podcast
description: >-
  **Reflection 70B**, a recently hyped open-source model, has faced serious
  controversy and accusations of fraud after investigations revealed it
  underperforms and may have misrepresented its capabilities, sparking debate
  about AI model verification. Meanwhile, **Apple** unveiled new AI features in
  iOS 18, including advanced visual intelligence and improvements to Siri,
  positioning itself ahead of competitors like **OpenAI** and **Google** in
  on-device AI phone capabilities.
tags:
  - reflection
  - glaive
  - apple
  - openai
  - google
  - sambanova
  - cerebras
  - stanford
  - hugh-face
  - reflection-70b
  - llama-3.1-70b
  - claude-3.5-sonnet
  - llama-3
  - superforecaster
  - rum
  - model-verification
  - open-source-models
  - ai-controversy
  - ai-ethics
  - visual-intelligence
  - on-device-processing
  - large-language-models
  - evaluation-methods
  - research-innovation
companies:
  - reflection
  - glaive
  - apple
  - openai
  - google
  - sambanova
  - cerebras
  - stanford
  - hugh-face
models:
  - reflection-70b
  - llama-3.1-70b
  - claude-3.5-sonnet
  - llama-3
  - superforecaster
  - rum
topics:
  - model-verification
  - open-source-models
  - ai-controversy
  - ai-ethics
  - visual-intelligence
  - on-device-processing
  - large-language-models
  - evaluation-methods
  - research-innovation
---


<!-- buttondown-editor-mode: plaintext -->**2 more weeks is all you need...**

> AI News for 9/9/2024-9/10/2024. We checked 7 subreddits, [**433** Twitters](https://twitter.com/i/lists/1585430245762441216) and **30** Discords (**215** channels, and **2311** messages) for you. Estimated reading time saved (at 200wpm): **247 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

Let's see:

- Glean [doubled valuation]( https://x.com/glean/status/1833476578912989281?s=61), again
- Dan Hendrycks' [Superforecaster AI](https://x.com/danhendrycks/status/1833152719756116154?s=46) generates very plausible election forecasts? One wonders how it will update after the debate. [Check the prompt](https://x.com/danhendrycks/status/1833163197626601603?s=46).
- A Stanford paper on [LLMs generated novel research ideas](https://x.com/chengleisi/status/1833166031134806330?s=46) made the rounds with a big claim: "*After a year-long study, we obtained the first statistically significant conclusion: LLM-generated ideas are more novel than ideas written by expert human researchers.*"
- SambaNova [announced slightly faster Llama 3 inference](https://www.linkedin.com/posts/sambanova_fastai-ugcPost-7239272368198557697-7FMk) than Cerebras, the previous world fastest ([our coverage here](https://buttondown.com/ainews/archive/ainews-cerebras-inference-faster-better-and/)). Independent evals are on the way.
- Benjamin Clavie [gave a notable talk](https://x.com/bclavie/status/1831431500161806562?s=46) on RAG and ColBERT/Late Interaction.
- [Strawberry reported to be launching in 2 weeks](https://x.com/steph_palazzolo/status/1833508052835909840?s=46) 

Yesterday, folks were also excited about [Google Illuminate](https://illuminate.google.com/home), AI generated podcast discussions about papers and books. It is gated behind a waitlist, but we at Smol AI are exploring doing the same. Check out [our first attempt here](https://github.com/smol-ai/temp/raw/main/combined_dialogue.mp3)!


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

**Apple's AI Announcements and Industry Reactions**

- Apple unveiled new AI features for iOS 18, including visual intelligence capabilities and improvements to Siri. [@swyx](https://twitter.com/swyx/status/1833231875537850659) noted that Apple has potentially "fixed Siri" and introduced a video understanding model, beating OpenAI to the first AI phone. The new features include mail and notification summaries, personal context understanding, and visual search integration.

- The new iPhone camera button is seen as prime real estate, with OpenAI/ChatGPT and Google search as secondary options to Apple's visual search. [@swyx](https://twitter.com/swyx/status/1833234781221622022) highlighted that the camera can now add events to the calendar, with processing done on-device and in the cloud.

- Some users expressed disappointment with Apple's recent innovations. [@bindureddy](https://twitter.com/bindureddy/status/1833248496948023753) mentioned that there hasn't been a compelling reason to upgrade iPhones in recent years, noting that Apple Intelligence seems similar to Google Lens, which was released years ago.

**AI Model Developments and Controversies**

- The AI community discussed the Reflection 70B model, with mixed reactions and controversies. [@BorisMPower](https://twitter.com/BorisMPower/status/1833187250420453716) stated that the model performs poorly, contrary to initial claims. [@corbtt](https://twitter.com/corbtt/status/1833209248236601602) announced an investigation into the model's performance, working with the creator to replicate the reported results.

- [@DrJimFan](https://twitter.com/DrJimFan/status/1833160432833716715) highlighted the ease of gaming LLM benchmarks, suggesting that MMLU or HumanEval numbers are no longer reliable indicators of model performance. He recommended using ELO points on LMSys Chatbot Arena and private LLM evaluation from trusted third parties for more accurate assessments.

- The AI research community discussed the importance of evaluation methods. [@ClementDelangue](https://twitter.com/ClementDelangue/status/1833136159209263552) announced the open-sourcing of "Lighteval," an evaluation suite used internally at Hugging Face, to improve AI benchmarking.

**AI in Research and Innovation**

- A study comparing LLM-generated research ideas to those of human experts found that AI-generated ideas were judged as more novel. [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1833228667641561495) shared key insights from the paper, noting that LLM-generated ideas received higher novelty scores but were slightly less feasible than human ideas.

- [@omarsar0](https://twitter.com/omarsar0/status/1833234005917065274) discussed a new paper on in-context learning in LLMs, highlighting that ICL uses a combination of learning from in-context examples and retrieving internal knowledge.

- [@soumithchintala](https://twitter.com/soumithchintala/status/1833177895734267987) announced the release of RUMs, robot models that perform basic tasks reliably with 90% accuracy in unseen, new environments, potentially unlocking longer trajectory research.

**AI Tools and Applications**

- [@svpino](https://twitter.com/svpino/status/1833233962757722268) shared an example of AI's capability to turn complex documents into interactive graphs within seconds, emphasizing the rapid progress in this area.

- [@jeremyphoward](https://twitter.com/jeremyphoward/status/1833170410135056477) announced SVG support for FastHTML, allowing for the creation of Mermaid editors.

- [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1833104751979794610) discussed DynamiqAGI, a comprehensive toolkit for addressing various GenAI use cases and building compliant GenAI applications on personal infrastructure.

**AI Ethics and Safety**

- [@fchollet](https://twitter.com/fchollet/status/1833171952070238240) argued that excessive anthropomorphism in machine learning and AI is responsible for misconceptions about the field.

- [@ylecun](https://twitter.com/ylecun/status/1833130597176205746) discussed the historical role of armed civilian militias in bringing down democratic governments and supporting tyrants, drawing parallels to current events.

**Memes and Humor**

- [@sama](https://twitter.com/sama/status/1833227974554042815) shared a humorous analogy: "if you strap a rocket to a dumpster, the dumpster can still get to orbit, and the trash fire will go out as it leaves the atmosphere," suggesting that while this contains important insights, it's better to launch nice satellites instead.


---

# AI Reddit Recap

## /r/LocalLlama Recap

**Theme 1. Reflection 70B: From Hype to Controversy**

- **Smh: Reflection was too good to be true - reference article** ([Score: 42, Comments: 19](https://reddit.com//r/LocalLLaMA/comments/1fd2f7m/smh_reflection_was_too_good_to_be_true_reference/)): The performance of **Reflection 70B**, a recently lauded open-source AI model, has been **questioned** and the company behind it **accused of fraud**. According to a [VentureBeat article](https://venturebeat.com/ai/new-open-source-ai-leader-reflection-70bs-performance-questioned-accused-of-fraud/), concerns have been raised about the legitimacy of the model's reported capabilities and benchmarks. The situation has sparked debate within the AI community about the **verification of AI model performance claims**.

- **Out of the loop on this whole "Reflection" thing? You're not alone. Here's the best summary I could come up.** ([Score: 178, Comments: 81](https://reddit.com//r/LocalLLaMA/comments/1fd75nm/out_of_the_loop_on_this_whole_reflection_thing/)): The post summarizes the **Reflection 70B controversy**, where **Matt Shumer** claimed to have created a revolutionary AI model using "**Reflection Tuning**" and **Llama 3.1**, surpassing established models like **ChatGPT**. Subsequent investigations revealed that the public API was likely a wrapper for **Claude 3.5 Sonnet**, while the released model weights were a poorly tuned **Llama 3 70B**, contradicting Shumer's claims and raising concerns about potential fraud and undisclosed conflicts of interest with **Glaive AI**.
  - **Matt Shumer's** claims about the **Reflection 70B** model were met with skepticism, with users questioning how it's possible to "accidentally" link to **Claude** while claiming it's your own model. Some speculate this could be a case of fraud or desperation in the face of a tightening AI funding landscape.
  - The incident drew comparisons to other controversial AI projects like the **Rabbit device** and "**Devin**". Users expressed growing skepticism towards **OpenAI** as well, questioning the company's claims about voice and video capabilities and noting key employee departures.
  - Discussions centered on potential motives behind Shumer's actions, with some attributing it to stupidity or narcissism rather than malice. Others speculated it could be an attempt to boost **Glaive AI** or secure venture capital funding through misleading claims.

- **Reflection and the Never-Ending Confusion Between FP16 and BF16** ([Score: 42, Comments: 15](https://reddit.com//r/LocalLLaMA/comments/1fcjtpo/reflection_and_the_neverending_confusion_between/)): The post discusses a **technical issue** with the **Reflection 70B** model uploaded to **Hugging Face**, which is **underperforming** compared to the baseline **LLaMA 3.1 70B**. The author explains that this is likely due to an **incorrect conversion** from **BF16** (used in LLaMA 3.1) to **FP16** (used in Reflection), which causes significant information loss due to the incompatible formats (**5-bit exponent and 10-bit mantissa** for FP16 vs **8-bit exponent and 7-bit mantissa** for BF16). The post strongly advises against using **FP16** for neural networks or attempting to convert **BF16 weights to FP16**, as it can severely degrade model performance.
  - **BF16 to FP16 conversion** may not be as destructive as initially suggested. **llama.cpp** tests show the **perplexity difference** between BF16 and FP16 is **10x less** than FP16 to Q8, and most **GGUFs** on HuggingFace are likely based on FP16 conversion.
  - The discussion highlighted the importance of **Bayesian reasoning** when evaluating **Schumer's claims**, given previous misrepresentations about the base model, size, and open-source status. Some users emphasized the need to consider these factors alongside technical explanations.
  - Several users noted that most model **weights typically fall within [-1, 1]** range, making FP16 conversion less impactful. **Quantization** to **8 bits** or less per weight often results in negligible or reasonable accuracy loss, suggesting FP16 vs BF16 differences may be minimal in practice.


**Theme 2. AMD's UDNA: Unifying RDNA and CDNA to Challenge CUDA**


- **[AMD announces unified UDNA GPU architecture — bringing RDNA and CDNA together to take on Nvidia's CUDA ecosystem](https://www.tomshardware.com/pc-components/cpus/amd-announces-unified-udna-gpu-architecture-bringing-rdna-and-cdna-together-to-take-on-nvidias-cuda-ecosystem)** ([Score: 284, Comments: 90](https://reddit.com//r/LocalLLaMA/comments/1fcyap8/amd_announces_unified_udna_gpu_architecture/)): AMD unveiled its new **unified Data Center Next Architecture (UDNA)**, combining elements of **RDNA** and **CDNA** to create a single GPU architecture for both gaming and data center applications. This strategic move aims to challenge **Nvidia's CUDA** ecosystem dominance by offering a unified platform that supports **AI**, **HPC**, and **gaming** workloads, potentially simplifying development across different GPU types and increasing AMD's competitiveness in the GPU market.

**Theme 3. DeepSeek V2.5: Quietly Released Powerhouse Model**

- **[DeepSeek silently released their DeepSeek-Coder-V2-Instruct-0724, which ranks #2 on Aider LLM Leaderboard, and it beats DeepSeek V2.5 according to the leaderboard](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Instruct-0724)** ([Score: 183, Comments: 39](https://reddit.com//r/LocalLLaMA/comments/1fd6z0v/deepseek_silently_released_their/)): DeepSeek has quietly released **DeepSeek-Coder-V2-Instruct-0724**, a new coding model that has achieved the **#2 rank** on the **Aider LLM Leaderboard**. This model outperforms its predecessor, **DeepSeek V2.5**, according to the leaderboard rankings, marking a significant improvement in DeepSeek's coding capabilities.
  - **DeepSeek-Coder-V2** expands support from **86 to 338 programming languages** and extends context length from **16K to 128K**. The model requires **8x80GB cards** to run, with no lite version available for most users.
  - Users discussed version numbering confusion between DeepSeek's general and coding models. The new coder model (**0724**) outperforms **DeepSeek V2.5** on the **Aider LLM Leaderboard**, but V2.5 beats 0724 in most other benchmarks according to [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V2.5).
  - Some users expressed interest in smaller, language-specific models for easier switching and interaction. DeepSeek typically takes about a month to open-source their models after initial release.

- **All of this drama has diverted our attention from a truly important open weights release: DeepSeek-V2.5** ([Score: 472, Comments: 95](https://reddit.com//r/LocalLLaMA/comments/1fclav6/all_of_this_drama_has_diverted_our_attention_from/)): The release of **DeepSeek-V2.5** has been overshadowed by recent AI industry drama, despite its potential significance as an **open GPT-4** equivalent. This new model, available on [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V2.5), reportedly combines **general and coding capabilities** with upgraded **API and Web** features.
  - **DeepSeek-V2.5** received mixed reviews, with some users finding it **inferior to Mistral-Large** for creative writing and general tasks. The model requires **80GB*8 GPUs** to run, limiting its accessibility for local use.
  - Users reported issues running the model, including **errors in oobabooga** and problems with **cache quantization**. Some achieved limited success using **llama.cpp** with reduced context length, but performance was slow at **3-5 tokens per second**.
  - Despite concerns, some users found DeepSeek-V2.5 useful for adding variety to outputs and potentially solving coding problems. It's available on [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V2.5) and through a cost-effective [API](https://open-tone-changer.vercel.app/).


**Theme 4. Innovative Approaches to Model Efficiency and Deployment**

- **[Open Interpreter refunds all hardware orders for 01 Light AI device, makes it a phone app instead. App launches TODAY!](https://changes.openinterpreter.com/log/01-app)** ([Score: 42, Comments: 4](https://reddit.com//r/LocalLLaMA/comments/1fczecj/open_interpreter_refunds_all_hardware_orders_for/)): Open Interpreter has **canceled** plans for its **01 Light AI hardware device**, opting instead to **launch a mobile app** that performs the same functions. This decision appears to be influenced by the **negative reception** of similar AI hardware devices like the **Rabbit R1**, with Open Interpreter choosing to leverage existing devices such as **iPhones** and **MacBooks** rather than introducing new hardware.

- **[generate usable mobile apps w/ LLMs on your phone](https://v.redd.it/lrthfybr6und1)** ([Score: 60, Comments: 23](https://reddit.com//r/LocalLLaMA/comments/1fcye12/generate_usable_mobile_apps_w_llms_on_your_phone/)): The post discusses the potential for **generating usable mobile apps using Large Language Models (LLMs) directly on smartphones**. This concept suggests a future where users could create functional applications through natural language interactions with AI assistants on their mobile devices, potentially revolutionizing app development and accessibility. While the post doesn't provide specific implementation details, it implies a significant advancement in on-device AI capabilities and mobile app creation processes.

- **[Deepsilicon runs neural nets with 5x less RAM and ~20x faster. They are building SW and custom silicon for it](https://x.com/sdianahu/status/1833186687369023550?)** ([Score: 111, Comments: 32](https://reddit.com//r/LocalLLaMA/comments/1fdav1n/deepsilicon_runs_neural_nets_with_5x_less_ram_and/)): **Deepsilicon** claims to run **neural networks** using **5x less RAM** and achieve **~20x faster** performance through a combination of **software** and **custom silicon**. Their approach involves **representing transformer models** with **ternary values** (-1, 0, 1), which reportedly eliminates the need for **computationally expensive floating-point math**. The post author expresses skepticism about this method, suggesting it seems too straightforward to be true.
  - **BitNet-1.58b** performance and **specialized hardware** for ternary values are key motivations for **Deepsilicon**. Challenges include scaling to larger models, edge device economics, and foundation model companies' willingness to train in 1.58 bits.
  - The **BitNet paper** demonstrates that training models from scratch with **1-bit quantization** can match **fp16 performance**, especially as model size increases. The [BitNet paper](https://arxiv.org/abs/2310.11453) provides insights into trade-offs.
  - Concerns were raised about **Y Combinator** funding practices and the founders' approach, as discussed in a [Hacker News thread](https://news.ycombinator.com/item?id=41490905). However, some see potential in targeting the **edge market** for portable ML in hardware and robotics applications.


**Theme 5. Advancements in Specialized AI Models and Techniques**

- **[New series of models for creative writing like no other RP models (3.8B, 8B, 12B, 70B) - ArliAI-RPMax-v1.1 Series](https://huggingface.co/ArliAI/Llama-3.1-70B-ArliAI-RPMax-v1.1)** ([Score: 141, Comments: 84](https://reddit.com//r/LocalLLaMA/comments/1fd4206/new_series_of_models_for_creative_writing_like_no/)): The ArliAI-RPMax-v1.1 series introduces **four new models** for creative writing and roleplay, with sizes ranging from **3.8B to 70B parameters**. These models are designed to excel in **creative writing and roleplay scenarios**, offering enhanced capabilities compared to existing RP models. The series aims to provide writers and roleplayers with powerful tools for generating imaginative and engaging content across various scales.

- **[Microsoft's Self-play muTuAl Reasoning (rStar) code is available on Github!](https://github.com/zhentingqi/rStar)** ([Score: 48, Comments: 4](https://reddit.com//r/LocalLLaMA/comments/1fcshuc/microsofts_selfplay_mutual_reasoning_rstar_code/)): Microsoft has released the code for their **Self-play muTuAl Reasoning (rStar)** algorithm on **GitHub**. This open-source implementation allows for **self-play mutual reasoning** in large language models, enabling them to engage in more sophisticated dialogue and problem-solving tasks. The rStar code can be found at [https://github.com/microsoft/rstar](https://github.com/microsoft/rstar), providing researchers and developers with access to this advanced AI technique.


- **[Mini-Omni: Language Models Can Hear, Talk While Thinking in Streaming (finetuned Qwen2-0.5B)](https://huggingface.co/gpt-omni/mini-omni)** ([Score: 49, Comments: 7](https://reddit.com//r/LocalLLaMA/comments/1fcmcql/miniomni_language_models_can_hear_talk_while/)): **Mini-Omni**, an open-source **multimodal large language model**, demonstrates the ability to process speech input and generate streaming audio output in real-time conversations. This model, based on a **finetuned Qwen2-0.5B**, showcases end-to-end capabilities for hearing and talking while simultaneously processing language.
  - A previous discussion thread on **Mini-Omni** from **6 days ago** was linked, indicating ongoing interest in the open-source multimodal model.
  - Users expressed desire for a **demo video** showcasing the model's voice-to-voice capabilities, emphasizing the importance of demonstrations for new AI models to garner attention and verify claimed functionalities.

## Other AI Subreddit Recap

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity


**AI Model Releases and Improvements**

- **OpenAI preparing to drop their new model**: A humorous post on r/singularity showing a video of a truck almost crashing, metaphorically representing OpenAI's model release process. [The post](https://www.reddit.com/r/singularity/comments/1fd8tfp/openai_preparing_to_drop_their_new_model/) garnered significant engagement with over 1000 upvotes and 110 comments.

- **Flux AI model developments**: Multiple posts discuss the Flux AI model:
  - A [post comparing ComfyUI and Forge](https://www.reddit.com/r/StableDiffusion/comments/1fcjs7i/the_current_flux_situation/) for running Flux, highlighting the ongoing debate in the community about different interfaces.
  - Another [post showcases 20 images generated using a Flux LoRA](https://www.reddit.com/r/StableDiffusion/comments/1fd5ba2/20_breathtaking_images_generated_via_bad_dataset/) trained on a limited dataset, demonstrating the model's capabilities even with suboptimal training data.

- **New Sora video released**: A [post on r/singularity](https://www.reddit.com/r/singularity/comments/1fcuw21/new_sora_video_just_dropped/) links to a new video demonstrating OpenAI's Sora text-to-video model capabilities.

**AI Tools and Interfaces**

- **Debate over AI interfaces**: The Stable Diffusion community is discussing the merits of different interfaces for running AI models, particularly **ComfyUI vs. Forge**. Key points include:
  - ComfyUI offers more flexibility and control but has a steeper learning curve.
  - Forge provides a more user-friendly interface with some quality-of-life improvements.
  - Some users advocate for using multiple interfaces depending on the task.

- **VRAM requirements**: Several comments discuss the **high VRAM requirements** for running newer AI models like Flux, with users debating strategies for optimizing performance on lower-end hardware.

**AI Ethics and Societal Impact**

- **Sam Altman image**: A [post featuring an image of Sam Altman](https://www.reddit.com/r/singularity/comments/1fcypio/altman_sam/) on r/singularity sparked discussion, likely related to his role in AI development and its societal implications.

**Humor and Memes**

- **"Most interesting year" meme**: A [humorous post on r/singularity](https://www.reddit.com/r/singularity/comments/1fd0rxd/hows_the_most_interesting_year_in_human_history/) asks "How's the most interesting year in human history going for you?", reflecting on the rapid pace of AI advancements.

- **AI model release meme**: The top post about OpenAI's model release uses humor to comment on the anticipation and potential issues surrounding major AI releases.


---

# AI Discord Recap

> A summary of Summaries of Summaries by Claude 3.5 Sonnet


**1. AI Model Releases and Benchmarks**

- **DeepSeek 2.5 Debuts with Impressive Specs**: **[DeepSeek 2.5](https://huggingface.co/collections/deepseek-ai/deepseek-v25-66d97550c81167fc5e5e32e6)** merges DeepSeek 2 Chat and Coder 2 into a robust 238B MoE with a **128k context length** and features like function calling.
   - This release is set to transform both coding and chat experiences, raising the bar for future models in terms of versatility and capability.
- **Deception 70B Claims Top Open-Source Spot**: The **Deception 70B** model was announced as the world's top open-source model, utilizing a unique Deception-Tuning method to enhance LLM self-correction capabilities.
   - This release, available [here](https://bit.ly/Deception-70B), sparked discussions about its potential applications and the validity of its claims in the AI community.
- **OpenAI's Strawberry Model Nears Release**: OpenAI is set to release its new model, **Strawberry**, as part of ChatGPT within the next two weeks, according to insider information shared in a [tweet](https://x.com/steph_palazzolo/status/1833508052835909840?s=46).
   - Initial impressions suggest potential limitations, with reports of **10-20 second** response times and concerns about memory integration capabilities.
  


**2. LLM Fine-tuning and Optimization Techniques**

- **Mixed Precision Training Boosts Performance**: Developers reported success implementing **mixed precision training** with **cpuoffloadingOptimizer**, noting improvements in **tokens per second (TPS)** processing.
   - Further testing is planned to explore integration with **FSDP+Compile+AC**, highlighting ongoing efforts to optimize model training efficiency.
- **Hugging Face Enhances Training with Packing**: Hugging Face announced that training with packed instruction tuning examples is now compatible with **Flash Attention 2**, potentially increasing throughput by up to **2x**.
   - This advancement aims to streamline the training process for AI models, making more efficient use of computational resources.
- **MIPRO Streamlines Prompt Optimization**: The DSPy team introduced **MIPRO**, a new tool designed to optimize instructions and examples in prompts for use with datasets in question-answering systems.
   - MIPRO's approach to prompt optimization highlights the growing focus on enhancing model performance through refined input techniques.
  


**3. Open Source AI Developments and Collaborations**

- **GitHub Hosts Open Source AI Panel**: GitHub is organizing a panel on **Open Source AI** on **September 19th** featuring speakers from **Ollama**, **Nous Research**, **Black Forest Labs**, and **Unsloth AI**. Free registration is available [here](https://lu.ma/wbc5bx0z).
   - The event aims to discuss how open source communities foster **access** and **democratization** in AI technology, reflecting the growing importance of collaborative efforts in AI development.
- **LlamaIndex Explores Agentic RAG Strategies**: A recent talk by @seldo explored **Agentic RAG** strategies for 2024 using [LlamaIndex](https://twitter.com/llama_index), discussing its significance and limitations.
   - The discussion highlighted strategies for enhancing RAG capabilities, showcasing the ongoing evolution of retrieval-augmented generation techniques in the open-source community.
- **Guilherme Releases Reasoner Dataset**: A new dataset called the [Reasoner Dataset](https://huggingface.co/datasets/Guilherme34/Reasoner-Dataset-FULL) was shared, created using **synthetic data** and designed for reasoning tasks.
   - This release demonstrates innovative approaches in AI training data development, potentially advancing the capabilities of models in logical reasoning and problem-solving.
  


**4. Multimodal AI and Tool Integrations**

- **Expand.ai Launches to Transform Web Data Access**: Tim Suchanek announced the launch of **[Expand.ai](https://x.com/TimSuchanek/status/1833538423954804948)**, a tool designed to convert websites into type-safe APIs, as part of Y Combinator's current batch.
   - This service aims to streamline **data retrieval** from websites, attracting interest from both tech-savvy and general users for its potential to simplify web data integration.
- **Chat AI Lite Offers Versatile AI Applications**: [Chat AI Lite](https://github.com/KevinZhang19870314/chat-ai-lite/blob/main/README_en_US.md) was introduced as a **versatile AI web application** covering multiple scenarios including chat, local knowledge bases, and image generation.
   - Its comprehensive capabilities aim to enhance user experience across various **AI applications**, showcasing the trend towards integrated AI tools for diverse use cases.
- **EDA-GPT Automates Data Analysis**: [EDA-GPT](https://github.com/shaunthecomputerscientist/EDA-GPT) was shared as a tool for **automated data analysis** leveraging large language models (LLMs), showcasing advanced integration for data science tasks.
   - This project encourages contributions to enhance its **data analytical capabilities**, highlighting the growing intersection of AI and data science tooling.
  

## GPT4O (gpt-4o-2024-05-13)


**1. DeepSeek 2.5 Launch**

- **DeepSeek 2.5 merges Chat and Coder models**: [DeepSeek 2.5](https://huggingface.co/collections/deepseek-ai/deepseek-v25-66d97550c81167fc5e5e32e6) combines **DeepSeek 2 Chat** and **Coder 2** into a powerful 238B MoE model with a **128k context length** and function calling features, aimed at revolutionizing coding and chat experiences.
  - This model is expected to set new standards for future models, providing robust performance in both coding and conversational contexts.
- **Confusion about DeepSeek model endpoints**: Users are confused about endpoints for [DeepSeek-Coder](https://openrouter.ai/models/deepseek/deepseek-coder) and [DeepSeek Chat](https://openrouter.ai/models/deepseek/deepseek-chat), with performance concerns like low throughputs of **1.75t/s** and **8tps**.
  - The model IDs will remain free for another five days, allowing users to transition smoothly.


**2. Model Fine-Tuning Challenges**

- **Unsloth fine-tuning issues**: Users face inference problems with **Unsloth**, resulting in repetitive outputs post fine-tuning, especially for paraphrasing tasks.
  - Discussions suggest optimizing hyperparameters like learning rate, batch size, and epoch count to improve performance.
- **Loss spikes in training**: A significant loss spike was reported after 725 steps in training, with loss reaching **20**. Adjusting **max grad norm** from **1.0** to **0.3** helped stabilize the loss.
  - This issue raised discussions on potential underlying factors affecting training stability across various models.


**3. Hardware and Model Performance**

- **Apple Silicon's GPU specs impress**: The **M2 Max MacBook Pro** boasts **96GB RAM** and effectively **72GB video memory**, capable of running **70B models** at **9 tokens/s**.
  - This integration allows efficient processing, showcasing Apple's competitive edge in hardware performance for AI tasks.
- **AMD vs NVIDIA performance debate**: Consensus emerged that **AMD's** productivity performance lags behind **NVIDIA**, particularly for applications like **Blender**.
  - Users expressed intentions to switch to **NVIDIA** with the upcoming **RTX 5000** series due to performance frustrations.


**4. AI Model Innovations**

- **Superforecasting AI tool released**: A new **Superforecasting AI** tool has launched, claiming to predict outcomes with **superhuman accuracy**, aiming to automate prediction markets.
  - A detailed demo and [blog post](https://www.safe.ai/blog/forecasting) explain its functionalities, sparking interest in its applications.
- **OpenAI's Strawberry model poised for release**: OpenAI is gearing up to launch the **Strawberry model**, designed for enhanced reasoning and detailed task execution.
  - While it promises significant advancements, concerns linger regarding initial response times and memory handling capabilities.


**5. Open Source AI Developments**

- **GitHub's Open Source AI panel announced**: GitHub will host a panel on **Open Source AI** on **9/19** with panelists from **Ollama**, **Nous Research**, **Black Forest Labs**, and **Unsloth AI**. Interested attendees can register [here](https://lu.ma/wbc5bx0z) after host approval.
  - The panel will explore the role of open source in increasing **access** and **democratization** within AI technologies.
- **Hugging Face introduces multi-packing for efficiency**: Hugging Face announced compatibility of packed instruction tuning examples with **Flash Attention 2**, aiming to boost throughput by up to **2x**.
  - This addition potentially streamlines AI model training significantly, with community excitement over its applications.


---

# PART 1: High level Discord summaries




## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **DeepSeek 2.5 Launches with Impressive Specs**: [DeepSeek 2.5](https://huggingface.co/collections/deepseek-ai/deepseek-v25-66d97550c81167fc5e5e32e6) merges **DeepSeek 2 Chat** and **Coder 2** into a robust 238B MoE with a **128k context length** and features like function calling.
   - It's set to transform coding and chat experiences, raising the bar for future models.
- **Transformers Agents Embrace Multi-Agent Systems**: Transformers Agents now support [multi-agent systems](https://x.com/AymericRoucher/status/1831373699670315257) that enhance task performance through specialization.
   - This method allows for efficient collaboration, enabling better handling of complex tasks.
- **Semantic Dataset Search is Back!**: [The Semantic Dataset Search](https://huggingface.co/spaces/librarian-bots/huggingface-datasets-semantic-search) has returned, offering capabilities to find similar datasets by ID or semantic searches.
   - This tool improves dataset accessibility on Hugging Face, streamlining research and development.
- **Korean Lemmatizer Integration with AI**: A developer successfully created a Korean lemmatizer and is exploring AI methods to disambiguate results further.
   - They received encouragement to utilize AI for distinguishing multiple lemma options generated for single words.
- **OpenSSL 3.3.2 with Post Quantum Cryptography**: A member learned to build **OpenSSL 3.3.2** incorporating **Post Quantum Cryptography (PQC)** on device.
   - *Lazy building FTW* emphasizing the ease of the installation process.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Model Fine-Tuning Hits Snags**: Users are encountering issues with inference in **Unsloth**, resulting in repetitive outputs after fine-tuning their models, especially for paraphrasing tasks. Factors like learning rate and batch size seem to affect these performance outcomes significantly.
   - Discussions suggest users should optimize hyperparameters, including epoch count, to avoid these pitfalls.
- **MLC Deployment Compatibility Concerns**: Challenges with MLC arise due to specific format requirements, prompting suggestions for full parameter fine-tuning to address interoperability. Quantized models may complicate these **MLC LLM deployments**.
   - Members highlighted a need for clearer guidelines on MLC compatibility with **Unsloth** models.
- **Unsloth Poised for Parameter Fine-Tuning**: Anticipation builds around the introduction of full-parameter fine-tuning support for **Unsloth**, currently focusing on **LoRA** and **QLoRA** methods. Developer stress is evident as projects push towards completion.
   - Members are hopeful for enhancements that could simplify future model deployments.
- **Loss Spiking Emerges in Training**: A member flagged a significant loss spike after 725 steps in their training process, reaching as high as **20**. They found that adjusting **max grad norm** from **1.0** to **0.3** helped stabilize the loss.
   - This raises discussion on potential underlying issues influencing training metrics across various models.
- **WizardMath Fine-Tuning Breakthrough**: **WizardMath** was successfully fine-tuned on real journal records, achieving a low loss of **0.1368** after over **13,000 seconds** of training. Future plans include using **RAG** to enhance the model's comprehension of document references.
   - This approach could significantly improve practical applications in bookkeeping and accounting.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Model Parameter Limits Are Discussed**: A user inquired about the smallest possible model parameter count for training, noting that **0.5B models** are available but perform poorly.
   - Contributions highlighted attempts with **200k and 75k parameter models**, emphasizing the impact of dataset size and structure on performance.
- **LM Studio Supports Multi-GPU Configurations**: It was confirmed that **LM Studio** supports multi-GPU setups, provided the GPUs are from the same manufacturer, e.g., using **two 3060s**.
   - A *member noted* that consistent models yield better performance, enhancing productivity, especially in computational-heavy tasks.
- **AMD vs NVIDIA: The Performance Skirmish**: Consensus emerged that **AMD's** performance in productivity applications lags behind **NVIDIA**, especially for software like **Blender**.
   - Personal experiences indicated intentions to switch to **NVIDIA** with the upcoming **RTX 5000** series due to performance frustrations.
- **Navigating Model Performance on Limited Hardware**: Discussion revealed that users aim to run **LM Studio** on limited hardware, particularly Intel setups, questioning the performance boundaries of larger models like **7B Q4KM**.
   - It was recommended to operate within **13B Q6 range** for **16GB GPUs** to maintain smoother operations during model execution.
- **Custom Model Development Insights**: Discussion on the merits of creating custom models surfaced, with one user eager to build their unique stack rather than use out-of-the-box solutions.
   - They shared experiences with **Misty** and **Open-webui**, while acknowledging the ongoing challenges in establishing an effective customized system.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Apple Silicon's impressive GPU specs**: Discussants highlighted the **M2 Max MacBook Pro** capabilities, boasting **96GB RAM** and effectively **72GB video memory** for running models.
   - This integration allows for efficient processing, with one user mentioning they can run **70B models** at a rate of **9 tokens/s**.
- **Gemini model's video analysis potential**: In relation to using the **Gemini model** for video analysis, one user inquired if it can summarize dialog and analyze expressions, not just transcribe audio.
   - Others suggested the need to implement training on custom datasets to achieve accurate results, and recommended leveraging available AI frameworks.
- **Availability of free models like Llama 3**: Users pointed out that models like **Llama 3** and **GPT-2** are available for free but require decent hardware to host effectively.
   - It's noted that running such local models necessitates a good PC or GPU, which raises resource requirements.
- **Voice feature feedback in GPT applications**: A member created a GPT called **Driver's Bro** that interfaces with Google Maps and uses a bro-like voice to provide directions.
   - *Unfortunately, the 'shimmer' voice falls short*, leading to a request for an advanced voice mode to enhance interaction.
- **Training custom models for stock analysis caution**: A member emphasized that using **OAI models** to analyze stocks is ineffective unless you have **ALL** historical data, including **images** and **graphs**.
   - They noted that accurate stock analysis requires using the **API** for performance purposes and mentioned that full stock history can be downloaded in JSON format.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Hermes 3 shifts to a paid model**: The standard **Hermes 3 405B** will transition to a **paid model** by the weekend, prompting users to switch to the free model at `nousresearch/hermes-3-llama-3.1-405b:free` to maintain access.
   - Users should act now, as shifting away from the paid model could lead to interruptions in service.
- **Eggu Dataset aims for multilingual enhancement**: The **Eggu** dataset, currently in development, targets the training of an **open source multilingual model** at **1.5GB**, integrating image positioning for better compatibility with vision models.
   - Though designed for wide usability, there are concerns about potential misuse of the dataset.
- **Confusion arises around DeepSeek models**: Confusion reigns regarding endpoints for [DeepSeek-Coder](https://openrouter.ai/models/deepseek/deepseek-coder) vs. [DeepSeek Chat](https://openrouter.ai/models/deepseek/deepseek-chat), with model IDs staying free for another five days.
   - Performance concerns include low throughputs of **1.75t/s** and **8tps** for certain variants.
- **Google Gemini grapples with rate limits**: Users experience recurring rate limit issues with **Google Gemini Flash 1.5**, frequently hitting limits despite user restrictions, prompting communications with **NVIDIA Enterprise Support**.
   - Many are using the **experimental API**, leading to additional challenges during model access.
- **Sonnet 3.5 Beta experiences downtime**: Recent outages affecting **Sonnet 3.5 Beta** were acknowledged, with users initially reporting lower success rates for API interactions, now restored as per **Anthropic's** status updates.
   - Though access is back, many users still question the model's overall stability moving forward.



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **Opus API Integration Stirs Conversations**: Discussion highlighted using an **API call to Opus** for the 'correct' version, hinting a shift in integration techniques.
   - Members noted related tweets revealing the topic's growing relevance within the engineering community.
- **Challenges with Model Uploading**: Participants noted that **model uploading** is proving to be more complex than expected, raising awareness of practical hurdles.
   - This reflects the broader narrative around user challenges in effective model deployment.
- **Batch Sizes and Performance Gains**: Discussions revealed that smaller matrices/batch sizes yield better performance, achieving a **3x speed-up** over a **1.8x** for larger sizes, but optimizations may require kernel rewrites.
   - Members noted potential losses with int16 and int8 packing, cautioning about **quantization errors**.
- **Triton Atomic Operations Constraints**: It became apparent that `tl.atomic_add` only supports 1D tensors, raising questions about workarounds for 2D implementations.
   - The community seeks efficient alternatives to manage multidimensional data operations.
- **Insights on PyTorch Autotuning**: Discussion centered around whether the **PyTorch** `inductor/dynamo` with autotuning could enhance **triton kernel** performance by caching tuned parameters.
   - A member noted potential for accelerated subsequent runs leveraging the same kernel configurations.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere's Acceptable Use Policy Clarified**: A member shared [Cohere's Acceptable Use Policy](https://docs.cohere.com/docs/c4ai-acceptable-use-policy), detailing prohibitions like **violence** and **harassment**.
   - The conversation highlighted **commercial use** implications, emphasizing compliance with local laws for model derivatives.
- **Fine-tuning Models Insights**: A question arose regarding the **fine-tuning** policy for CMD-R models, specifically its cost-free use.
   - Clarifications indicated that **self-hosted** models come with restrictions against commercial use.
- **Temperature Settings Affect Output Quality**: Members suggested experimenting with temperature settings of **0** or **0.1** to gauge variations in output quality.
   - The discourse centered around ensuring outputs don't deviate **wildly** from initial examples.
- **Innovative Advanced Computer Vision Ideas**: Requests for **advanced project ideas** in **computer vision** sparked suggestions to explore intersections with **LLM projects**.
   - Teamwork was noted as vital for overcoming challenges in project success, with members brainstorming collaboration strategies.
- **Leveraging Google Vision API in Projects**: A fun **Pokedex project** utilizing **Google Vision API** and **Cohere LLMs** aims to identify **Pokemon** names and descriptions from images.
   - Clarifications indicated the API was used for **creating image labels**, not learning embeddings, with **Kaggle** suggested for datasets.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Exploring Windows Usage**: A member inquired about how to use the project on **Windows**, reflecting a common interest in the platform's compatibility across operating systems.
   - This question indicates that users are keen on various platform integrations for broader accessibility.
- **Inquiry on Desktop Beta Access**: Discussion emerged around whether it was too late to join the **desktop beta** program, highlighting user eagerness for new features.
   - Members demonstrated a desire to engage with the latest advancements in the Open Interpreter suite.
- **Launch of 01 App for Mobile Devices**: The **01 App** is now live on Android and iOS, with plans for enhancements driven by user feedback.
   - The community is urged to fork the app on GitHub to tailor experiences, showcasing an open-source spirit.
- **Tool Use Episode 4 Launch**: The latest episode titled *'Activity Tracker and Calendar Automator - Ep 4 - Tool Use'* is available on [YouTube](https://www.youtube.com/watch?v=N9GCclB8rYQ), featuring discussions on **time management**.
   - The speakers emphasize that **time is our most precious resource**, motivating viewers to utilize tools effectively.
- **Support for Open Source Development**: Community backing for open-source projects stemming from the 01 platform is vibrant, providing ample opportunities for new initiatives.
   - Members expressed enthusiasm to contribute, reinforcing a collaborative environment around AI tools.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Modular Lacks Windows Timeline**: There is currently **no timeline** for a **Windows native version** as Modular prioritizes support for **Ubuntu and Linux distros**.
   - *Modular aims to avoid tech debt and enhance product quality before broadening their focus,* drawing lessons from past experiences with Swift.
- **WSL as Current Windows Support**: While a native **.exe** version is not available, *Modular suggests using WSL* as the extent of their current **Windows support**.
   - Users showed interest in future native options but acknowledged existing limitations.
- **Mojo Eyeing GPU and GStreamer Replacement**: Mojo is being pitched as a potential replacement for **GStreamer**, leveraging upcoming GPU capabilities for efficient processing.
   - Members are keen on modern library integration for live streaming, showcasing Mojo's potential for streamlined operations.
- **Exploring Bindings with DLHandle**: Members discussed using **DLHandle** for creating Mojo bindings, referencing projects that demonstrate its application.
   - Projects like 'dustbin' utilize DLHandle for **SDL bindings**, providing inspiration for those in graphical applications.
- **Understanding Variant Type in Mojo**: The **Variant type** in Mojo was highlighted for its utility in creating lists with different element types along with memory considerations.
   - Members clarified issues related to size alignment and behavior of discriminants in these implementations.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **DisTro sparks confusion**: Discussions around **DisTro** raised questions about its purpose and effectiveness, as no code has been released yet, possibly to prompt competition.
   - Members speculated on its intended impact, questioning whether the announcement was premature.
- **AI training concerns heighten**: Concerns arose regarding AI models trained on user satisfaction metrics, which often produce shallow information instead of accurate content.
   - A fear was expressed that this trend could compromise the quality of AI responses, especially when relying heavily on human feedback.
- **OCTAV's successful launch**: A member shared their success in implementing **NVIDIA's OCTAV** algorithm using Sonnet, noting the scarcity of similar examples online.
   - They speculated about the potential inference of the implementation from the associated paper, showcasing the model's capabilities.
- **Repetitive responses annoy engineers**: Chat focused on the tendency of AI to generate repetitive outputs, especially when users show slight hesitance.
   - Discussion evolved around how models like Claude struggle to maintain confidence, often retracting solutions too quickly.
- **Mixed performance of AI models**: Members evaluated the performance of platforms like **Claude** and **Opus**, highlighting their respective strengths and weaknesses.
   - While Claude has a solid alignment strategy, it falters in certain situations compared to the more engaging Opus.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Tokenizer eos option missing from Mistral and Gemma**: A user proposed sending a PR to fix the tokenizer eos problem, citing that current **Mistral** and **Gemma** tokenizers lack the `add_eos` option. They referenced a [utility that needs updating](https://github.com/pytorch/torchtune/blob/main/torchtune/modules/tokenizers/_utils.py).
   - Another member emphasized that the `add_eos` feature must first be implemented to resolve the issue.
- **Eleuther_Eval recipe defaults to GPT-2 model**: A member inquired why the **Eleuther_Eval** recipe always loads the **GPT-2** model, clarified as the default since `lm_eval==0.4.3`. They noted that the model can be overwritten with `TransformerDecoder` tools for evaluations on other models.
   - This highlights the need for flexibility in selecting model types for evaluations.
- **Mixed Precision Training yields promising results**: A member shared their excitement about implementing **mixed precision training** with **cpuoffloadingOptimizer**, noting improvements in **TPS**. They expressed uncertainty about how it integrates with **FSDP+Compile+AC**, suggesting further testing is required.
   - This signals potential optimizations for large-scale model training.
- **Compile Speed Outshines Liger**: Benchmarks indicated that using `compile(linear+CE)` is faster in both speed and memory than **Liger**. Though, **chunkedCE** exhibited higher memory savings when compiled independently despite being slower overall.
   - This comparison emphasizes the trade-offs between speed and resource utilization in model compilation.
- **Dynamic seq_len presents optimization challenges**: Concerns about **dynamic seq_len** in **torchtune** surfaced, particularly its effect on the **INT8 matmul triton kernel** due to re-autotuning. Members discussed padding inputs to multiples of **128**, although this adds extra padding costs.
   - Optimizing for speed while managing padding overhead remains a topic of interest.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Jim Harbaugh Endorses Perplexity**: Head coach **Jim Harbaugh** stated that a great playbook isn't complete without **Perplexity** in a recent announcement, inviting fans to [ask him anything](https://x.com/perplexity_ai/status/1833173842870853896) on the matter.
   - This endorsement is aimed at integrating Perplexity into coaching strategies, highlighting its relevance in sports analytics.
- **Reflection LLM Update Inquiry**: A member asked whether the **Reflection LLM** will soon be added to Perplexity, expressing interest in feature updates.
   - However, no definitive answers emerged from the discussion, leaving the community curious about future enhancements.
- **Issues with Perplexity Pro Rewards**: A user voiced frustration over the **Perplexity Pro rewards** deal with Xfinity, citing that their promo code was invalid.
   - The community discussed potential workarounds, including creating a new account to apply the promo successfully.
- **Performance Woes for Claude 3.5**: **Claude 3.5** users raised concerns that the model's performance appears to have declined, hinting at potential capacity issues despite recent investments.
   - Users reported confusion over the model version shown in their settings, indicating a lack of clarity in updates.
- **Nvidia Exceeds Q2 Earnings Benchmarks**: **Nvidia** exceeded Q2 earnings expectations, thanks to strong graphics card sales and robust growth in their AI sector, as reported [here](https://www.perplexity.ai/page/nvidia-beats-q2-expectations-k9CT.KnRT1uKI8OG99kdrA).
   - Analysts noted that this impressive performance reinforces Nvidia's foothold in the tech landscape amid rising demand for AI solutions.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Apple Intelligence Updates Coming Soon**: Apple plans to release updates to its **Intelligence capabilities** within two weeks, focusing on improvements to **Siri** and other AI functionalities.
   - Users believe these updates could address longstanding issues, intensifying competition with **OpenAI**.
- **ColPali Model Gains Ground**: ColPali is under review with new slides presented showcasing its implementation and efficacy in various **AI tasks**.
   - The integration of ColPali with advanced training techniques could transform current AI research paradigms.
- **Superforecasting AI Launches with Precision**: A new **Superforecasting AI** tool has been released, showcasing its ability to predict outcomes with **superhuman accuracy**.
   - This tool aims to automate prediction markets, bolstered by a detailed demo and [blog post](https://www.safe.ai/blog/forecasting) explaining its functionalities.
- **OpenAI's Strawberry Model Poised for Release**: OpenAI is gearing up to launch the **Strawberry model**, designed for enhanced reasoning and detailed task execution.
   - While it promises significant advancements, concerns linger regarding initial response times and memory handling capabilities.
- **Expand.ai Launches to Transform Web Data Access**: Tim Suchanek announced the launch of **Expand.ai**, a tool converting websites into type-safe APIs, as part of Y Combinator's current batch.
   - This service aims to streamline **data retrieval** from websites, attracting interest from both tech-savvy and general users.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Agentic RAG Strategies for 2024**: In a recent talk, **Agentic RAG** was highlighted as a key focus for 2024, emphasizing its significance with [LlamaIndex](https://twitter.com/llama_index). Key points included understanding **RAG**'s necessity but limitations, alongside strategies for enhancement.
   - The audience learned about practical applications and theoretical aspects of RAG in the context of LLMs.
- **Integrating LlamaIndex with Llama 3**: Members discussed the integration of [LlamaIndex with Llama 3](https://docs.llamaindex.ai/en/stable/examples/llm/ollama/) and provided detailed setup instructions for running a local Ollama instance.
   - Insights shared included installation steps and usage patterns for LlamaIndex, including command snippets for Colab, streamlining model experimentation.
- **DataFrames made easy with LlamaIndex**: A guide on using the `PandasQueryEngine` to convert natural language queries into Python code for Pandas operations has surfaced, enhancing text-to-SQL accuracy.
   - Safety concerns regarding arbitrary code execution were stressed, encouraging cautious usage of the tool.
- **MLflow and LlamaIndex Integration Issues Fixed**: The community discussed a recent issue with MLflow and LlamaIndex that has been resolved, with expectations for a release announcement over the weekend.
   - A member plans to document this integration experience in a blog article, aiming to assist others dealing with similar challenges.
- **Exploring Similarity Search in LlamaIndex**: Members engaged in a deep dive into performing similarity searches with methods like `similarity_search_with_score` in LlamaIndex and noted key differences from Langchain.
   - Detailed examples were provided, showcasing how to filter retrieved documents based on metadata, improving information retrieval capabilities.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Deception 70B Claims to be Top Open-Source Model**: An announcement revealed **Deception 70B**, claimed to be the world's top open-source model, utilizing a unique Deception-Tuning method to enhance LLM self-correction.
   - The release can be found [here](https://bit.ly/Deception-70B), generating curiosity in the community regarding its practical applications.
- **OpenAI's Strawberry Model to Launch Soon**: Insiders announced OpenAI is set to release its new model, **Strawberry**, integrated into ChatGPT within two weeks, but initial impressions indicate sluggish performance with **10-20 seconds** per response.
   - Critics are skeptical about its memory integration capabilities, as detailed in this [tweet](https://x.com/steph_palazzolo/status/1833508052835909840?s=46).
- **Concerns Over Otherside AI's Scam History**: Discussions on **Otherside AI** revisited past scams, particularly a self-operating computer project linked to accusations of ripping off open-source work, stirring doubt about the legitimacy of their claims.
   - Reference to ongoing issues can be explored [here](https://github.com/OthersideAI/self-operating-computer/issues/67), highlighting community skepticism.
- **AI Forecasting Performance Critiqued**: Dan Hendrycks reported disappointing performance from the paper **LLMs Are Superhuman Forecasters**, indicating significant underperformance against a new test set.
   - A demo showcasing this AI prediction model is accessible [here](http://forecast.safe.ai), reigniting debates on its forecasting accuracy.
- **Gemini Integration with Cursor Sparks Interest**: Members explored the integration possibilities of **Gemini** with **Cursor**, raising questions about functionality and new use cases.
   - *Curiosity about Google’s latest developments* was expressed, driving more members to consider experimenting with the integration.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Better Hardware for Image Generation**: A member recommended using **Linux** for local training with a **24G NVIDIA** card to boost image generation performance.
   - They also emphasized checking the power supply for compatibility, noting that an upgrade wasn't necessary.
- **Cheaper Alternatives to Deep Dream Machine**: The community discussed potential substitutes for **Deep Dream Machine**, suggesting **Kling** or **Gen3** for AI video creation.
   - One user highlighted a **66% off** promotion for **Kling**, attracting further interest.
- **Tips for Training SDXL Models**: A member asked for techniques to effectively train **SDXL** using **Kohya Trainer** to enhance image quality.
   - Another member advised refining the query for more helpful responses, suggesting review of related channels.
- **Clarifications on CLIP Model Choices**: Discussions arose about selecting appropriate **CLIP models** in the **DualCLIPLoader** node, specifically between **clip g** and **clip l**.
   - Community members noted that **Flux** was not trained on **clip g**, leading to some confusion.
- **Discord Bot Delivers AI Services**: A member introduced their verified Discord bot capable of text-to-image generation and chat assistance through a shared link.
   - This service aims to integrate robust AI functionalities directly within Discord for user convenience.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **GitHub's Open Source AI Panel Announced**: GitHub is hosting a panel on **Open Source AI** on **9/19** with panelists from **Ollama**, **Nous Research**, **Black Forest Labs**, and **Unsloth AI**. Interested attendees can register for free [here](https://lu.ma/wbc5bx0z) after host approval.
   - The panel will explore the role of open source in increasing **access** and **democratization** within AI technologies.
- **AI Model Performance Sparks Debate**: A recent test on an AI model revealed it was **impressive** yet **an order of magnitude slower**, causing concerns for larger models, particularly those with **500M parameters**.
   - This raised skepticism about the performance metrics based solely on **small models** from libraries like **sklearn** or **xgboost**.
- **Efforts in Private Machine Learning Highlighted**: Discussions surrounding **private machine learning** emphasize a lack of effective solutions, with mentions of **functional encryption** and **zero knowledge proofs** as potential strategies, though they are known to be slow.
   - Participants suggested using **Docker** to create **secure containers** as a more feasible approach for ensuring model security.
- **Multiparty Computation's Complexity Discussed**: A user touched on strategies for **multiparty computation** to optimize workloads in cloud settings, although concerns lingered about the security of such methods.
   - The conversation noted the considerable investment needed to develop secure solutions in **trustless environments**.
- **Challenges of Achieving Machine Learning Privacy**: Experts asserted that achieving **full privacy** in machine learning remains elusive and costly, with a pressing need for effective privacy solutions in sensitive scenarios like those linked to **DARPA**.
   - The significant financial incentives underline the community's interest in navigating this complex issue.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **AI Research Community Faces Fraud Allegations**: On September 5th, Matt Shumer, CEO of OthersideAI, announced a supposed breakthrough in training mid-size AI models, which was later revealed to be *false* as reported in a [Tweet](https://x.com/shinboson/status/1832933747529834747?t=lu0kNqbEZKG5LVC30Dm7hA&s=19). This incident raises concerns about *integrity in AI research* and highlights the need for skepticism regarding such claims.
   - The discussion centered around the implications for accountability in AI research, suggesting ongoing vigilance is necessary to avoid similar situations.
- **Guilherme Shares Reasoner Dataset**: A user shared the [Reasoner Dataset](https://huggingface.co/datasets/Guilherme34/Reasoner-Dataset-FULL), stating it is crafted using *synthetic data* aimed at reasoning tasks. This approach reflects innovative techniques in developing training datasets for AI.
   - Community members showed interest in leveraging this dataset for enhancing reasoning capabilities in model training.
- **iChip Technology Revolutionizes Antibiotic Discovery**: iChip technology, capable of culturing previously unculturable bacteria, has significantly impacted antibiotic discovery, including *teixobactin* in 2015. This technology’s potential lies in its ability to grow bacteria in **natural environments**, vastly increasing microbial candidates for drug discovery.
   - Experts discussed the implications of this technology for future pharmaceutical innovations and its role in addressing antibiotic resistance.
- **Hugging Face Introduces Multi-Packing for Increased Efficiency**: Hugging Face announced compatibility of packed instruction tuning examples with **Flash Attention 2**, aiming to boost throughput by up to **2x**. This addition potentially streamlines AI model training significantly.
   - The community anticipates improvements in training efficiency, with members sharing excitement over possible applications in upcoming projects.
- **OpenAI Fine-Tuning API gains Weight Parameter**: OpenAI enhanced their fine-tuning API by introducing a **weight** parameter as detailed in their [documentation](https://platform.openai.com/docs/guides/fine-tuning/multi-turn-chat-examples). Implemented in **April**, this parameter allows for finer control over training data influence.
   - Users discussed how this capability could impact model performance during fine-tuning processes, enhancing training dynamics.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **Claude 3.5's Audio Features in Question**: A member inquired if it's possible to pass **audio data** to **Claude's 3.5** LLM via **Langchain** for transcription, raising concerns about its capabilities.
   - Another user noted that while Claude 3.5 supports images, there was uncertainty about audio functionalities.
- **Langchain4j Token Counting Challenge**: Discussion emerged around how to **count tokens** for input and output with **langchain4j**, expressing a need for solutions.
   - Unfortunately, the thread did not yield specific guidance on token counting techniques.
- **Whisper Proposed for Audio Transcription**: One member suggested utilizing **Whisper** for audio transcription as a **faster and cheaper** alternative to Claude 3.5.
   - This recommendation points to potential efficiencies in transcription workflows compared to Claude.
- **Chat AI Lite: Multifaceted AI Web Application**: [Chat AI Lite](https://github.com/KevinZhang19870314/chat-ai-lite/blob/main/README_en_US.md) is a **web application** that covers chat, knowledge bases, and image generation, enhancing the user experience across various **AI applications**.
   - Its feature set showcases flexibility catering to multiple scenarios within the AI domain.
- **Automated Data Analysis with EDA-GPT**: [EDA-GPT](https://github.com/shaunthecomputerscientist/EDA-GPT) provides **automated data analysis** using LLMs, highlighting advanced integration for data science tasks.
   - The project encourages contributions to improve its **data analytical capabilities**.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Emotion Classifier Output Confusion**: A member questioned whether altering the description to **'Classify to 7 emotions'** instead of specifics would change the output of the Emotion classifier.
   - *No clear conclusions on the output impact were provided*.
- **AdalFlow Library Insights Needed**: Discussion on the [AdalFlow](https://github.com/SylphAI-Inc/AdalFlow) library aimed at auto-optimizing LLM tasks was reignited, with members seeking deeper insights.
   - One member committed to reviewing the library, promising to share their findings by the end of the week.
- **Misleading Llama AI Model Discovery**: A member disclosed that a supposedly Llama AI model was actually the latest **Claude** model, utilizing a complex prompt mechanism.
   - This system guided the model through problem-solving and reflective questioning strategies.
- **MIPRO Revolutionizes Prompt Optimization**: The new tool **MIPRO** enhances prompt optimization by refining instructions and examples for datasets.
   - Members explored how MIPRO streamlines prompt optimization for question-answering systems, emphasizing its dataset relevance.



---



## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **Recommendations for LLM Observability Platforms**: A member is exploring options for **LLM observability platforms** for a large internal corporate RAG app, currently considering [W&B Weave](https://wandb.ai/weave) and [dbx's MLflow](https://mlflow.org/).
   - They also expressed interest in alternatives like **Braintrust** and **Langsmith** for enhanced observability.
- **Node.js Struggles with Anthropic's API**: Using **Anthropic's API** with **Node.js** reportedly yields worse performance compared to **Python**, especially with tools.
   - The discussion arose around whether others have faced similar performance discrepancies, prompting a deeper look into potential optimization.



---



## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **Merge Conflicts Resolved**: A member thanked another for their help, successfully resolving **merge conflicts** without further issues.
   - *Much appreciated for the quick fix!*
- **Locating Test Scores**: A member displayed confusion about retrieving specific **test scores** after saving results, prompting a discussion on best practices.
   - Another member recommended checking the **score folder**, especially the file `data.csv`.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **George Hotz's tinygrad Enthusiasm**: Discussion kicked off with an enthusiastic share about **tinygrad**, which focuses on simplicity in deep learning frameworks.
   - The chat buzzed with excitement over the implications of this lightweight approach for machine learning projects.
- **Engagement in the Community**: A user expressed enthusiasm by posting a wave emoji, indicating lively interaction related to **tinygrad** in the community.
   - This kind of engagement signals a strong interest in the advancements led by George Hotz.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Sign Up for GitHub's Open Source AI Panel!**: GitHub is hosting a free [Open Source AI panel](https://lu.ma/wbc5bx0z) on **9/19** in their SF office, focusing on **accessibility** and **responsibility** in AI.
   - Panelists from **Ollama**, **Nous Research**, **Black Forest Labs**, and **Unsloth AI** will discuss the **democratization of AI technology**.
- **Hurry, Event Registration Requires Approval!**: Participants need to register early as the event registration is subject to host approval, ensuring a spot at this sought-after panel.
   - Attendees will gain insights into how open source communities are driving **innovation** in the AI landscape.



---


The **Alignment Lab AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Mozilla AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **DiscoResearch Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1283141072914219122)** (1 messages): 

> - `DeepSeek 2.5`
> - `Yi Coder 1.5B+9B`
> - `OLMoE`
> - `Multi-agent systems support`
> - `Semantic Dataset Search` 


- **DeepSeek 2.5 Launches with Impressive Specs**: [DeepSeek 2.5](https://huggingface.co/collections/deepseek-ai/deepseek-v25-66d97550c81167fc5e5e32e6) merges **DeepSeek 2 Chat** and **Coder 2** into a robust 238B MoE with a **128k context length** and advanced features like function calling.
   - It's positioned to revolutionize both coding and chat experiences, setting a high bar for future models.
- **Transformers Agents Embrace Multi-Agent Systems**: Transformers Agents now support [multi-agent systems](https://x.com/AymericRoucher/status/1831373699670315257) for improved task performance through specialization.
   - This new approach allows for efficient collaboration among agents, making it easier to tackle complex tasks.
- **Semantic Dataset Search is Back!**: The [Semantic Dataset Search](https://huggingface.co/spaces/librarian-bots/huggingface-datasets-semantic-search) has returned, providing capabilities to find similar datasets by ID or perform semantic searches.
   - This tool enhances accessibility and usability of datasets on Hugging Face, making research and development more efficient.
- **OLMoE Boasts Expansive Training Data**: [OLMoE](https://huggingface.co/collections/allenai/olmoe-66cf678c047657a30c8cd3da) is a 6.9B MoE model trained on a staggering **5T tokens**, fully open source to foster collaboration.
   - Its architecture and extensive training data are expected to deliver robust performance in various applications.
- **New Tool for Image Background Removal**: A new [image background removal tool](https://x.com/xenovacom/status/1828116951186710795) leverages in-browser inference to transform images quickly and privately using the latest Transformers.js.
   - Users can enjoy fast, high-quality results at no cost, ensuring data privacy.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/TheZachMueller/status/1831002292440469519)">Tweet from Zach Mueller (@TheZachMueller)</a>: Today @huggingface accelerate 0.34.0 is now out, and it is a packed release!  From `torchpippy` updates to resumable dataloader support, and revamped TransformerEngine support, there&#39;s a ton to co...</li><li><a href="https://x.com/AymericRoucher/status/1831373699670315257)!">Tweet from Aymeric (@AymericRoucher)</a>: 🥳 Transformers Agents now supports Multi-agent systems!  Multi-agent systems have been introduced in Microsoft&#39;s framework Autogen. It simply means having several agents working together to solve...</li><li><a href="https://x.com/vllm_project/status/1833257997814096245)">Tweet from vLLM (@vllm_project)</a>: We are excited to see @vllm_project as an option for local apps in the @huggingface hub! It comes with easy snippets to quickly test out the model.</li><li><a href="https://x.com/xenovacom/status/1828116951186710795)">Tweet from Xenova (@xenovacom)</a>: There has been a huge debate recently about the best approach for image background removal. Here&#39;s my attempt: - In-browser inference w/ 🤗 Transformers.js - WebGPU accelerated (fast!) - Costs $0 ...</li><li><a href="https://x.com/multimodalart/status/1833459429557088314)">Tweet from apolinario 🌐 (@multimodalart)</a>: It&#39;s now so easy add images to the gallery  of your LoRA on @huggingface 🤯 🪄   ① Generate an image with the Widget 🖼️  ② Press &#34;Add to model card gallery&#34; 🔥</li><li><a href="https://x.com/vanstriendaniel/status/1833188523207496058)">Tweet from Daniel van Strien (@vanstriendaniel)</a>: The @huggingface&#39;s Semantic Dataset Search is back in action! Find similar datasets by ID or do a semantic search of dataset cards.  Give it a try: https://huggingface.co/spaces/librarian-bots/hug...</li><li><a href="https://x.com/gabrielmbmb_/status/1832078861296668748)">Tweet from Gabriel Martín Blázquez (@gabrielmbmb_)</a>: Yesterday Reflection 70B was released, a model fine-tuned using Reflection-Tuning that achieved impressive scores in several benchmarks such as MMLU. The dataset that was used for the fine-tuning wasn...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1282777583452426316)** (455 messages🔥🔥🔥): 

> - `Whisper Model Usage`
> - `Korean Lemmatizer Development`
> - `Model Structured Output`
> - `Quantization and Dataset Calibration`
> - `Hugging Face Community Dynamics` 


- **Using Whisper for Audio Transcription**: A user inquired about using the [Whisper model](https://huggingface.co/openai/whisper) for audio transcription, seeking guidance on running it locally.
   - They discussed potential credit limitations for using the model, expressing interest in learning to run models locally despite facing installation challenges.
- **Korean Lemmatizer Integration with AI**: A developer shared their success in creating a Korean lemmatizer and sought advice on how to leverage AI to refine the results further due to inherent ambiguities.
   - They were encouraged to experiment with AI to distinguish between multiple lemmas generated for single words.
- **Structured Output from AI Models**: Users reported various outputs from models when given prompts without specific programming context, resulting in unexpected responses like dataclasses instead of relevant code.
   - One demonstrated how the model generated a text file for a walk in the park, showcasing its ability to produce structured output when not explicitly asked for programming-related responses.
- **Exploring Quantization with AWQ**: A user discussed starting to quantify models using AWQ and expressed the need for datasets to assist with calibration.
   - They sought recommendations on appropriate data sources to improve their quantization efforts.
- **AI Model Comparisons**: The conversation shifted towards evaluating the quality of certain AI models, particularly comparing a specific model's performance against GPT-4.
   - Users shared their experiences with various models, including interactions with structured outputs and the limitations of different setups.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/openai/whisper">Whisper - a Hugging Face Space by openai</a>: no description found</li><li><a href="https://x.com/AdeenaY8/status/1833460689400123452">Tweet from Adina Yakup (@AdeenaY8)</a>: Exciting to see a paper from the Uzbekistan community on today&#39;s paper list http://hf.co/papers🥰Big shout out to @MamasaidovM @murodbeck for their contributions to the open source community🙌 htt...</li><li><a href="https://huggingface.co/blog/codeparrot">Training CodeParrot 🦜 from Scratch</a>: no description found</li><li><a href="https://tenor.com/view/tim-and-eric-what-confused-absurd-tim-heidecker-gif-18146476">Tim And Eric What GIF - Tim And Eric What Confused - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/shafire/talktoaiZERO">shafire/talktoaiZERO · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/spaces/TheVixhal/Resume-Roaster">LegalMIndAI - a Hugging Face Space by TheVixhal</a>: no description found</li><li><a href="https://huggingface.co/openai/whisper-large-v3">openai/whisper-large-v3 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/nroggendorff/objaverse">nroggendorff/objaverse · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1282922591866327081)** (2 messages): 

> - `OpenSSL 3.3.2`
> - `Post Quantum Cryptography`
> - `TLS Handshakes` 


- **Building OpenSSL 3.3.2 with PQC**: Today I learned to build out **OpenSSL 3.3.2** with **Post Quantum Cryptography (PQC)** on device.
   - *Lazy building FTW* highlights the ease of the process.
- **QompaSSL Update for OpenSSL**: There's an important update regarding **QompaSSL** for **OpenSSL 3.3.2**, emphasizing the significance of **TLS Handshakes**.
   - The update reiterates that **TLS Handshakes are important** for secure communications.


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/)** (1 messages): 

cakiki: Is it open?
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1282779237153247303)** (21 messages🔥): 

> - `Synthetic Data Creation with GANs`
> - `Quantized GraphRAG Systems`
> - `Local-First Vector Database`
> - `Resume Roaster Project`
> - `LLM Responses and Formatting` 


- **Collaborative Exploration of Synthetic Data with GANs**: A member inquired if a model can be used as a GAN to create **synthetic data** for stock placement, sparking a discussion on fine-tuning methods and the importance of a proper discriminator.
   - *If you are able to fine-tune it* is crucial for using it as a generator, suggesting a **data set** will need to be generated for effective GAN training.
- **Challenges with Quantized GraphRAG Systems**: There was a consensus that the **graph rag** approach with a quantized model produced **messy results**, with another member suggesting full-scale models might yield better outcomes.
   - An exploration of potential improvements and recommendations for better accuracy was mentioned, indicating a need for **better data** handling.
- **Building a Local-First Vector Database**: A member shared an article on creating a **local-first vector database** with RxDB and transformers.js, highlighting benefits like zero network latency and offline functionality.
   - This approach allows for **semantic searches** directly in the browser, making it applicable for offline-first apps while optimizing performance.
- **Launch of Resume Roaster Project**: A fun project named **Resume Roaster** was introduced, inviting members to check it out for its innovative approach to **resume generation**.
   - This project was linked for further exploration, showcasing user engagement with practical applications in career development.
- **Improving LLM Responses with Formatting Tips**: Suggestions were made to enhance responses from **LLMs** by formatting outputs as JSON or YAML, providing additional metadata for clarity.
   - The focus was on ensuring that prompts are designed to extract more satisfying responses, reflecting on the expected limitations of quantized models.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/TheVixhal/Resume-Roaster">LegalMIndAI - a Hugging Face Space by TheVixhal</a>: no description found</li><li><a href="https://rxdb.info/articles/javascript-vector-database.html">JavaScript Vector Database | RxDB - JavaScript Database</a>: The local-first revolution is here, changing the way we build apps! Imagine a world where your app&#x27;s data lives right on the user&#x27;s device, always available, even when there&#x27;s no intern...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1282973603612327957)** (1 messages): 

> - `Instruction-tuned Models`
> - `DPO/RLHF-tuning`
> - `LLaMA 3.1`
> - `Fine-tuning Guardrails` 


- **Exploring Instruction-tuned Models**: A member pondered the idea of using already instruction-tuned and DPO/RLHF-tuned models to disable their embedded guardrails, suggesting potential for enhanced cognitive abilities.
   - *Could a fine-tuning approach allow models like LLaMA 3.1 to better function without these guardrails?* This could lead to more versatile applications in AI.
- **Possibilities with LLaMA 3.1**: The discussion highlighted that models such as **LLaMA 3.1** are likely trained on extensive instruction-tuned and preference corpuses.
   - This background raises questions about how their capabilities could change if fine-tuned to operate with fewer restrictions.


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1283023801507905548)** (12 messages🔥): 

> - `PDF Document Analysis`
> - `ColPali Embeddings Issue`
> - `Amazon ML Challenge 2023`
> - `Korean Lemmatizer with AI`
> - `Building NLP Models from Scratch` 


- **Creating Program for PDF Organization**: A member seeks advice on developing a program to analyze and categorize a collection of unorganized PDFs, considering using vector embedding with Lbl2Vec.
   - They also mentioned discovering [llamaFS](https://link.to.llamaFS), a program that handles similar tasks but relies on multiple external APIs.
- **ColPali's Embedding Format Challenge**: A discussion arose about ColPali's embedding output shape of **[1030,128]**, which conflicts with the unidimensional expectations of most vectordb collections, primarily Chroma.
   - The member is exploring whether a pooling operation or other solutions could rectify this shape inconsistency.
- **Tackling the Amazon ML 2023 Challenge Dataset**: A member is seeking guidance on predicting product length using text data from the Amazon ML Challenge dataset, which includes product titles, descriptions, and attributes.
   - They provided a link to the dataset, highlighting the significance of accurate product length estimation for packaging and customer assessments.
- **Integrating AI into Korean Lemmatizer**: A member is looking for AI-based methods to resolve ambiguities in their Korean lemmatizer, which they've developed without AI over the past year.
   - They seek recommendations for effectively leveraging AI to determine the most accurate lemma from multiple possibilities.
- **Help Wanted: Building NLP Model with PyTorch**: A member is reaching out for assistance in constructing their own NLP model from scratch using PyTorch, expressing confusion regarding input and output parameters.
   - They noted their prior experience in computer vision but are venturing into NLP for the first time.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.kaggle.com">Kaggle: Your Machine Learning and Data Science Community</a>: Kaggle is the world&#x2019;s largest data science community with powerful tools and resources to help you achieve your data science goals.</li><li><a href="https://www.kaggle.com/datasets/ashisparida/amazon-ml-challenge-2023">Amazon ML Challenge 2023</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1282811953529749607)** (5 messages): 

> - `Diffusers and Latent Space Manipulation`
> - `Image-to-Image Generation with Diffusers`
> - `Using CLIP Text Embeddings`
> - `Denoising Latent Images` 


- **Confusion on Manipulating Latent Space**: A new user expressed uncertainty about how to manipulate the latent space of an image using **CLIP** text embeddings, questioning the size mismatch between autoencoder outputs and CLIP embeddings.
   - They attempted this manipulation but were dissatisfied with the results, seeking clarification on the expected outcomes.
- **Denoising Latent with Text Embeddings**: A member suggested gradually denoising the latent image with the available text embeddings to improve results.
   - They recommended interpolating between the original latent and the denoised version as a potential method.
- **Image-to-Image Generation Process Detailed**: Another member provided a detailed explanation of the image-to-image process, highlighting how initial images are encoded to latent space and noise is added.
   - They shared a [Hugging Face documentation link](https://huggingface.co/docs/diffusers/en/using-diffusers/img2img) for using the **AutoPipelineForImage2Image** class to facilitate this process.
- **Interest in Modifying Latent Space**: The original poster showed interest in the impressive results from spaces that modify latent space for image-to-image generation.
   - This interest sparked their motivation to explore latent space manipulation further, particularly through the **image-to-image generation** methods.



**Link mentioned**: <a href="https://huggingface.co/docs/diffusers/en/using-diffusers/img2img">Image-to-image</a>: no description found

  

---



### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1282777591459348663)** (333 messages🔥🔥): 

> - `Model Fine-Tuning`
> - `MLC Deployment Issues`
> - `Unsloth Updates`
> - `Inference Problems`
> - `Llama-3.1-SuperNova-Lite` 


- **Challenges with Model Fine-Tuning**: Users are experiencing difficulties with inference yielding repetitive results after fine-tuning their models in Unsloth, particularly regarding paraphrasing tasks.
   - Despite successful training setups, multiple factors such as learning rate, batch size, and epoch count could be influencing performance outcomes.
- **MLC Deployment Issues**: Concerns regarding MLC compatibility arise due to specific format requirements, with users suggesting the need for full parameter fine-tuning to resolve issues.
   - Discussion indicates that using quantized models might complicate interoperability with MLC LLM deployments.
- **Updates on Unsloth's Development**: There is anticipation about incoming support for full-parameter fine-tuning within Unsloth, expected this year or the next.
   - Current focus remains on LoRA and QLoRA methods, while developer stress levels are running high as projects near completion.
- **Inference Problems with Models**: Inference in some notebooks is reportedly providing similar outputs with little variation, prompting users to explore temperature settings and tokenizer adjustments.
   - Results suggest that users may need to adjust configurations for better performance, including evaluating learning rates and merging methods post-training.
- **Introduction of Llama-3.1-SuperNova-Lite**: A new model, Llama-3.1-SuperNova-Lite, developed by Arcee.ai, offers an 8B parameter architecture with high performance and compact design.
   - This model leverages distilled training methods and instruction datasets, aiming to deliver effective results while being resource-efficient for organizations.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/1mvwsIQWDs2EdZxZQF9pRGnnOvE86MVvR?usp=sharing#scrollTo=2eSvM9zX_2d3">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/17d3U-CAIwzmbDRqbZ9NnpHxCkmXB6LZ0?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://llm.mlc.ai/docs/compilation/convert_weights.html#clone-from-hf-and-convert-weight">Convert Model Weights &mdash; mlc-llm 0.1.0 documentation</a>: no description found</li><li><a href="https://huggingface.co/unsloth/llama-3-8b">unsloth/llama-3-8b · Hugging Face</a>: no description found</li><li><a href="https://rentry.co/cgx2w8pk">from trl import SFTTrainer</a>: from transformers import TrainingArguments, DataCollatorForSeq2Seq from unsloth import is_bfloat16_supported import os os.environ[&quot;WANDB_PROJECT&quot;] = &quot;spellbound&quot;  # name your W&amp...</li><li><a href="https://huggingface.co/arcee-ai/Llama-3.1-SuperNova-Lite">arcee-ai/Llama-3.1-SuperNova-Lite · Hugging Face</a>: no description found</li><li><a href="https://docs.unsloth.ai/troubleshooting/errors#evaluation-loop-also-oom-or-crashing">Errors | Unsloth Documentation</a>: To fix any errors with your setup, see below:</li><li><a href="https://github.com/unslothai/unsloth/issues/689">Does unsloth have a script for full parameter fine-tuning? · Issue #689 · unslothai/unsloth</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1282836410864373761)** (7 messages): 

> - `Kaggle Housing Price Challenge`
> - `Unsloth Fine-tuned Model Deployment`
> - `MOE Model Performance` 


- **Kaggle Housing Price Challenge parallels**: A user pointed out the need to convert data into vectors or numbers for machine learning, citing similarities to the objectives of the [Kaggle Housing Price Challenge](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).
   - The emphasis was on data preparation as a vital step in the machine learning workflow.
- **Seeking guidance on Deploying Unsloth models on AWS**: A user requested assistance on deploying **Unsloth fine-tuned models** on AWS SageMaker, noting the models don’t deploy normally like **other models** due to missing components.
   - They sought experiences or guidance from anyone who has successfully completed such deployments.
- **Jamba 1.5 mini outpaces Llama 3.1**: A member surprised the community by sharing that the **Jamba 1.5 mini** model handled **50 concurrent requests** faster than **Llama 3.1 70B** handling **10 concurrent requests**.
   - They provided reference data, noting that **Llama** averaged **25 seconds per request**, showcasing the **efficiency** of the MOE model.



**Link mentioned**: <a href="https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques">House Prices - Advanced Regression Techniques | Kaggle</a>: no description found

  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1282864342441918475)** (27 messages🔥): 

> - `Full Fine-Tuning Inquiry`
> - `Loss Spiking Issue`
> - `Flash Attention 2 Usage`
> - `Optimal GPU Size for LLAMA 3.1`
> - `Metric Computation Support in SFTTrainer` 


- **Unsloth's Support for Full Fine-Tuning**: A member questioned if Unsloth supports full fine-tuning, to which another member replied negatively.
   - No detailed explanation or setup instructions were provided in the discussion.
- **Loss Spiking During Training**: A member reported that their training loss decreased until **725 steps**, then spiked above **20**.
   - Suggestions included adjusting **max grad norm** from **1.0** to **0.3**, which seemed to stabilize the loss.
- **Using Flash Attention 2 with Gemma 2**: A user inquired about successfully using **Flash Attention 2** with **Gemma 2 models**, expressing concerns about vRAM usage.
   - Despite configuring two environments for Flash Attention, they noted no difference in memory consumption.
- **Optimal GPU Size for LLAMA 3.1**: A member sought advice on the optimal GPU size for fine-tuning **LLAMA 3.1 70B**, with a focus on **40GB vs 80GB** options.
   - Responses indicated a preference for **A100 or H100 GPUs** with at least **80GB vRAM** for effective fine-tuning.
- **Error with SFTTrainer's Computation Metrics**: A user encountered a **NotImplementedError** during training, related to copying a tensor with no data.
   - The error raised questions about the underlying issues in the **SFTTrainer** framework, particularly in metric computation.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1282995569249357846)** (9 messages🔥): 

> - `WizardMath fine-tuning`
> - `Collaboration on RAG`
> - `Experience in machine learning`
> - `Mechanical engineering background` 


- **Successful Fine-Tuning of WizardMath**: A member fine-tuned **WizardMath** on bookkeeping double-entry real journal records, achieving a notable loss of **0.1368** after **13007.8255 seconds** of training.
   - The member plans to implement **RAG** after fine-tuning to enhance the model's understanding of document codenames used in the alpaca form dataset.
- **Seeking Clarity on Collaboration Objectives**: The potential collaborator expressed the need for clarity on the exact problem statement regarding the use of **RAG**, suggesting that fine-tuning the embedding model might be more beneficial.
   - There seems to be a mutual interest in addressing daily tasks through this collaborative effort.
- **Mechanical Engineer's Transition to GenAI**: A member shared their background as a **mechanical engineer** with a publication in a good journal and a year of full-time work in a **GenAI** role.
   - This experience adds a solid academic foundation to the planned collaboration on machine learning tasks.
- **Humorous Acknowledgment of Accounting Logic**: One member humorously anticipated that the challenges of **accounting logic** might amuse their structural engineer brother, reflecting common perceptions of the field.
   - This exchange highlighted a light-hearted understanding among the members about different engineering disciplines.


  

---



### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1282787007415849010)** (81 messages🔥🔥): 

> - `Model Training Parameters`
> - `Multi-GPU Support in LM Studio`
> - `Availability of Older Versions`
> - `Optimal Models for Running AI`
> - `Performance on Limited Hardware` 


- **Discussion on Model Parameter Limits**: A user inquired about the smallest possible model parameter count for training, to which others contributed that **0.5B models are available but perform poorly**.
   - Users discussed attempting **200k and 75k parameter models**, noting that dataset size and structure significantly impact performance.
- **LM Studio's Multi-GPU Capability**: It was confirmed that LM Studio supports **multi-GPU configurations**, provided that the GPUs are from the same manufacturer, e.g., two Nvidia cards.
   - A mention was made that using the same model, such as **two 3060s**, would yield better performance compared to different models.
- **Question on LM Studio Version Management**: A user expressed concern about the lack of access to older versions of LM Studio, questioning how URLs for specific releases could be tracked.
   - It was suggested that editing the URL of current releases might provide access to older versions, although there's no formal repository.
- **Best Models for Specific Tasks**: Users shared experiences trying to identify the best models for text classification and the occult, expressing a desire for comprehensive models.
   - Recommendations included **Mistral Trismegistus 7B Q8_0**, although mixed results were reported and further alternatives were welcomed.
- **Running AI on Limited Hardware**: Users discussed the feasibility of running LM Studio on limited hardware, specifically Intel setups, and the performance of larger models like **7B Q4KM**.
   - Suggestions included staying within the **13B Q6 range** for 16GB GPUs to ensure smoother operation and proper model support.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://wandb.ai/mostafaibrahim17/ml-articles/reports/A-Deep-Dive-Into-Learning-Curves-in-Machine-Learning--Vmlldzo0NjA1ODY0#what-are-accuracy-and-loss-curves?-">A Deep Dive Into Learning Curves in Machine Learning</a>: Understand machine learning better with our guide on accuracy and loss curves. We explain their differences, how to read them, and why they&#39;re important.</li><li><a href="https://wandb.ai/mostafaibrahim17/ml-articles/reports/A-Dee">mostafaibrahim17</a>: Weights & Biases, developer tools for machine learning</li><li><a href="https://www.tomshardware.com/news/the-end-of-sli-as-we-know-it-nvidia-reveals-new-model">The End of SLI As We Know It: Nvidia Reveals New Model</a>: Is buying two RTX 3090s worth the money?
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1282797998304133160)** (93 messages🔥🔥): 

> - `GPU capabilities`
> - `AMD vs NVIDIA performance`
> - `Mistral model operations`
> - `Surface Studio Pro upgrades`
> - `Building custom models` 


- **GPU capabilities for different models**: Users discussed the capabilities of **GPUs** for running different model sizes, with some stating the **12GB GPU** can handle up to **13B Q4** models efficiently, showcasing a comparative ease of use versus the **8B Q8** model.
   - Members pointed out the technical limitations based on GPU memory, as well as compatibility in terms of productivity software benefits leaning heavily towards **NVIDIA**.
- **AMD vs NVIDIA performance in productivity**: There was a consensus that **AMD's** productivity performance is lacking compared to **NVIDIA**, especially with software like **Blender** and **Adobe** that favor the latter.
   - Members noted personal experiences of struggling with **AMD products** and expressed intentions to switch to **NVIDIA** around the release of the **RTX 5000** series.
- **Optimizing Mistral Model Operations**: A discussion occurred about the difficulties in achieving optimal performance with the **Mistral-7B-Instruct-v0.3** model on CPU-based inference without GPU acceleration and effective optimization.
   - It was highlighted that CPU inference could be significantly slower than GPU-powered systems like the **4090**, regardless of the CPU's power, reflecting the need for **context window management**.
- **Exploring Surface Studio Pro upgrades**: One user expressed frustration with the limitations of the **Surface Studio Pro** due to its inability to upgrade hardware, considering options like an **eGPU** or **SSD** improvements.
   - The user voiced a desire for gentle guidance in these upgrade considerations, asking how to enhance performance given their non-technical background.
- **Custom Model Development Discussions**: Participants discussed the merits of developing custom models, with one user expressing a desire to build a unique stack for their language models rather than settling for ready-made solutions.
   - They shared experiences with **Misty** and **Open-webui** while acknowledging the challenges faced in finding a system that meets their specific needs.



**Link mentioned**: <a href="https://www.tomshardware.com/pc-components/gpus/nvidia-bans-using-translation-layers-for-cuda-software-to-run-on-other-chips-new-restriction-apparently-targets-zluda-and-some-chinese-gpu-makers">Nvidia bans using translation layers for CUDA software &mdash; previously the prohibition was only listed in the online EULA, now included in installed files [Updated]</a>: Translators in the crosshairs.

  

---



### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1282777563424362548)** (89 messages🔥🔥): 

> - `Apple Silicon GPU capabilities`
> - `Gemini model functionalities`
> - `Llama 3 and free models`
> - `Video analysis AI projects`
> - `OpenCV limitations` 


- **Apple Silicon's impressive GPU specs**: Discussants highlighted the **M2 Max MacBook Pro** capabilities, boasting **96GB RAM** and effectively **72GB video memory** for running models.
   - This integration allows for efficient processing, with one user mentioning they can run **70B models** at a rate of **9 tokens/s**.
- **Gemini model's video analysis potential**: In relation to using the **Gemini model** for video analysis, one user inquired if it can summarize dialog and analyze expressions, not just transcribe audio.
   - Others suggested the need to implement training on custom datasets to achieve accurate results, and recommended leveraging available AI frameworks.
- **Availability of free models like Llama 3**: Users pointed out that models like **Llama 3** and **GPT-2** are available for free but require decent hardware to host effectively.
   - It's noted that running such local models necessitates a good PC or GPU, which raises resource requirements.
- **Exploring AI solutions for specific use cases**: One user discussed their project aimed at analyzing video to track player positions, expressing challenges in achieving good results with **OpenCV**.
   - Another user shared an open-source project that successfully implemented player detection and tracking in sports, potentially aiding the original poster's efforts.
- **Training custom object detection models with Yolo**: The conversation shifted to training custom models for specific object detection tasks, with users emphasizing the importance of training data quantity.
   - It was suggested that **10 to 1000 examples** may be required depending on the specificity of the objects being tracked in video input.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/skalskip92/status/1816162584049168389">Tweet from SkalskiP (@skalskip92)</a>: football AI code is finally open-source  - player detection and tracking - team clustering - camera calibration  I still need to work on README; don&#39;t judge me on that  code: https://github.com/ro...</li><li><a href="https://x.com/apples_jimmy/status/1833337411788804595">Tweet from Jimmy Apples 🍎/acc (@apples_jimmy)</a>: This week I take a small step out of the cave of patience  Quoting Jimmy Apples 🍎/acc (@apples_jimmy)   All quiet on the western front, a heavy fog of schizo energy.   I’m ready to be hurt again, let...
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1282904926145478708)** (8 messages🔥): 

> - `Driver's Bro GPT`
> - `Voice features in GPTs`
> - `Memory feature feedback`
> - `Using DALLE-3`
> - `Image creation through ChatGPT` 


- **Driver's Bro GPT needs better voice**: A member created a GPT called **Driver's Bro** that interfaces with Google Maps and uses a bro-like voice to provide directions, helping users vent while driving.
   - *Unfortunately, the 'shimmer' voice falls short*, leading to a request for an advanced voice mode to enhance interaction.
- **Request for male voice in GPT**: There was a strong request for at least one **male voice** option in GPTs, as the current **shimmer voice** does not meet expectations.
   - The sentiment expressed that the existing options are inadequate for user preferences.
- **Feedback on Memory feature**: One user commented that the new **Memory feature** makes conversations feel more human-like, having noticeable retention of information.
   - The feeling of interacting with something that can remember details like a person was highlighted as particularly impressive.
- **Creating images with DALLE-3**: A member inquired about creating images through **DALLE-3** for free versions, seeking clarity on available options.
   - It was shared that users have **5 daily uses** of drawing in a specific channel and can also use **2 free DALLE-3 requests** through ChatGPT.
- **Handling complex requests with 4o**: A suggestion was shared that the **4o** model can handle complex requests, allowing for multi-tasking in a single query.
   - Users were encouraged to articulate their requests clearly, treating the model like a person who can assist with multiple tasks.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1282781007434748095)** (13 messages🔥): 

> - `Stock Evaluation`
> - `Universal Evaluator Prompt`
> - `Accessing Prompt Library` 


- **Caution Advised on Stock Predictions**: One member cautioned against using **OAI models** for stock analysis unless they have access to **complete historical data** and the API for live updates.
   - They noted that many sites offer full stock history in **JSON format**, which is essential for accurate modeling.
- **Universal Evaluator for Prompt Development**: A member shared their creation of a **Universal Evaluator** prompt persona that compares two outputs and gives numerical scores based on subjectivity.
   - They highlighted the importance of having the evaluator articulate its reasoning for more insightful context.
- **Prompt Library Access Question**: A user inquired about accessing the **prompt library** for resources on prompt development.
   - Another member provided a helpful response, directing them to the relevant channel now called <#1019652163640762428>.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1282781007434748095)** (13 messages🔥): 

> - `Using OAI Models for Stock Analysis`
> - `Universal Evaluator Prompt Persona`
> - `Accessing Prompt Library` 


- **OAI Models Not Recommended for Stock Analysis**: A member emphasized that using **OAI models** to analyze stocks is ineffective unless you have **ALL** historical data, including **images** and **graphs**, and indicates the necessity of live updates.
   - They noted that accurate stock analysis requires using the **API** for performance purposes and mentioned that full stock history can be downloaded in JSON format.
- **Universal Evaluator for Prompt Development**: Another member shared their creation of a **Universal Evaluator prompt persona** that compares outputs to judge their quality, providing **numerical scores** based on subjective reasoning.
   - They highlighted the importance of this tool for **prompt development** and the necessity for the evaluator to explain its reasoning within context.
- **Location of the Prompt Library**: A user inquired about accessing the **prompt library** and received a response with the updated channel name, now referred to as **<#1019652163640762428>**.
   - The interaction highlighted the community's willingness to assist with navigating resources.


  

---



### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1283135584885538898)** (1 messages): 

> - `Hermes 3 transition`
> - `Paid model announcement` 


- **Hermes 3 heading towards paid model**: The standard **Hermes 3 405B** will transition into a paid model by the weekend, prompting users to adjust their usage.
   - To continue using it for free, switch to the model slug `nousresearch/hermes-3-llama-3.1-405b:free` as the **free variant** may have limited availability.
- **Upcoming changes to Hermes 3 model access**: Users are advised that the transition to a paid model will happen shortly, potentially affecting access to **Hermes 3**.
   - This change is effective soon, so it's critical to move to the specified free model slug to avoid disruptions.



**Link mentioned**: <a href="https://openrouter.ai/models/nousresearch/hermes-3-llama-3.1-405b:free">Hermes 3 405B Instruct (free) - API, Providers, Stats</a>: Hermes 3 is a generalist language model with many improvements over Hermes 2, including advanced agentic capabilities, much better roleplaying, reasoning, multi-turn conversation, long context coheren...

  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1282783049108422729)** (2 messages): 

> - `Eggu Dataset`
> - `Open Source Multilingual Models`
> - `Cost of Usage` 


- **Eggu Dataset Development**: The **Eggu** dataset is currently in development and aims to train an open source multilingual model, with a size of **1.5GB** that incorporates image positioning for compatibility with vision models.
   - This dataset is intended to be used by many and faces concerns about being misused by some.
- **Training Costs are Relatively Low**: Using OpenAI services costs approximately **$2,500** in credits for just **one week's usage** of resources.
   - This expense is considered reasonable given the potential outputs from the dataset and models.


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1282780180087312415)** (102 messages🔥🔥): 

> - `DeepSeek Models and Performance`
> - `Google Gemini Flash Rate Limits`
> - `Sonnet 3.5 Beta Issues`
> - `Costs of Hermes 3 and Llama 3 Models`
> - `AI Programming Tools Explore` 


- **DeepSeek Models Confusion**: Discussions revealed confusion around DeepSeek's models, particularly regarding endpoints for ['coder'](https://openrouter.ai/models/deepseek/deepseek-coder) vs. ['chat'](https://openrouter.ai/models/deepseek/deepseek-chat). Members noted the model IDs are set to remain free for another five days, easing migration concerns.
   - Concerns about **throughputs** being low, with reports of performance at **1.75t/s** and only **8tps** for certain models.
- **Google Gemini Flash Rate Limit Woes**: A user reported recurring rate limit issues with **Google Gemini Flash 1.5**, stating that their application would hit limits frequently, even with user restrictions in place. They are in communication with **NVIDIA Enterprise Support** to clarify compatibility and limitations.
   - Concerns were raised that many are forced to use the **experimental API**, which presents its own limitations, as seen by the errors they faced while accessing the models.
- **Sonnet 3.5 Beta Outage Acknowledged**: Recent outages affecting **Sonnet 3.5 Beta** were confirmed, with users reporting a drop in successful API interactions. Status updates from **Anthropic** confirmed normal success rates have returned for free users.
   - Members expressed relief as access was noted to be restored; however, overarching questions about stability remain prevalent in discussions.
- **Hermes 3 Pricing Speculation**: Participants discussed speculation surrounding the future costs of **Hermes 3 405b**, indicating potential anxiety about transitioning from free access. A user humorously noted how users might react to sudden charges after being accustomed to no costs.
   - Conversations pointed out that while **Llama 3 405B** is cheaper for outputs, it also may come with trade-offs regarding performance, leading to a decision-making dilemma for many users.
- **Exploration of AI Programming Tools**: Users discussed tools suitable for programming with mentions of **Aider** and **Cursor**, highlighting their respective features and experiences. One noted that **Aider**'s methodology could feel peculiar due to how it interacts with model responses.
   - The dialogue reflected a broader interest in finding effective programming aids, indicating user intention to experiment with various offerings based on current cloud credits availability.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://status.anthropic.com/">Anthropic Status</a>: no description found</li><li><a href="https://openrouter.ai/docs/parameters-api">Parameters API | OpenRouter</a>: API for managing request parameters</li><li><a href="https://openrouter.ai/models/deepseek/deepseek-coder">DeepSeek-Coder-V2 - API, Providers, Stats</a>: DeepSeek-Coder-V2, an open-source Mixture-of-Experts (MoE) code language model. It is further pre-trained from an intermediate checkpoint of DeepSeek-V2 with additional 6 trillion tokens. Run DeepSeek...</li><li><a href="https://openrouter.ai/models/deepseek/deepseek-chat">DeepSeek V2.5 - API, Providers, Stats</a>: DeepSeek-V2 Chat is a conversational finetune of DeepSeek-V2, a Mixture-of-Experts (MoE) language model. It comprises 236B total parameters, of which 21B are activated for each token. Run DeepSeek V2....</li><li><a href="https://github.com/paul-gauthier/aider/issues">Issues · paul-gauthier/aider</a>: aider is AI pair programming in your terminal. Contribute to paul-gauthier/aider development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1282891286361149490)** (5 messages): 

> - `Opus API Integration`
> - `Model Uploading Challenges` 


- **Opus API Call Sparks Interest**: Discussion highlighted the interesting aspect of using an **API call to Opus** for the 'correct' version, indicating a shift in how integrations are approached.
   - One member noted they saw related tweets just yesterday, showing the topic's growing buzz in the community.
- **Model Uploading Proves Challenging**: A participant remarked that **model uploading** is turning out to be more difficult than initially anticipated, hinting at complexities that were unforeseen.
   - This insight reflects broader concerns about the practical challenges faced by users as they navigate the uploading process.


  

---


### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1282781190109007935)** (6 messages): 

> - `Batch Performance Optimization`
> - `Triton Atomic Operations`
> - `Triton Compilation Process` 


- **Batch Sizes Impact Performance**: As discussed, smaller matrices/batch sizes yield slightly better performance, achieving around **3x speed-up** instead of **1.8x** for `(1, 4096, 4096)`, but proper optimization may require a complete kernel rewrite.
   - The use of int32 packing is noted as 'lossless', while options for int16 and int8 exist, though they risk introducing **quantization errors**.
- **Triton Atomic Add Limitations**: `tl.atomic_add` currently only supports 1D tensors, raising questions about potential workarounds for 2D tensors.
   - The community is seeking efficient methods or alternatives to achieve similar functionality with multidimensional data.
- **Clarification on Triton Compilation**: A member questioned the accuracy of Triton's compilation process, specifically whether it directly compiles Python to PTX, referencing an article on Triton internals.
   - The consensus is that Triton translates source to **Triton IR**, then to **LLVM IR**, and finally to inline **PTX**.


  

---


### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1282874117627969557)** (7 messages): 

> - `PyTorch Autotuning`
> - `Triton Autotuner`
> - `Functional Optimizer in PyTorch`
> - `Open Source Models Adaptation`
> - `Tulu Project Announcement` 


- **Exploring PyTorch Autotune for Triton Kernels**: Members discussed whether the **PyTorch** `inductor/dynamo` with its autotuning feature for **triton kernels** can be utilized for custom kernels, enhancing performance by caching tuned parameters.
   - One member mused that this advancement could lead to faster subsequent runs using the same kernel.
- **Triton's Autotuner vs Manual Caching**: A member pointed out that, while **Triton** has its own autotuner, it lacks the functionality to save tuned results across script re-runs.
   - Another member humorously remarked on their uncertainty about any existing built-in feature for Triton that addresses this issue.
- **Interesting Functional Optimizer Unveiled**: A member shared an intriguing functional **PyTorch optimizer** from Apple's **sigmoid attention** release, suggesting it could create valuable fusion opportunities when combined with `torch.func.grad_and_value`.
   - They hinted that this combination might lead to advancements in the forward, backward, and optimization steps using `torch.compile`.
- **Expert Exchange with Hamish Ivison Announcement**: An upcoming expert exchange featuring **Hamish Ivison** on 'Adapting open source models with **Open-Instruct** and **Tulu**' was announced for tomorrow at **11am PST**.
   - Participants were encouraged to tune in live via [YouTube](https://www.youtube.com/watch?v=e1qUJFAo10s) and engage with the speaker during the talk.
- **Insights on Open-Instruct and Tulu Project**: Hamish Ivison is set to discuss post-training strategies for language models, tracing the evolution of the **open-instruct library** and its impact on the **Tulu project**.
   - Attendees can expect insights into state-of-the-art models adapted from llama, with a preview of the upcoming **Tulu 3 release**.



**Link mentioned**: <a href="https://github.com/apple/ml-sigmoid-attention/tree/main/optorch">ml-sigmoid-attention/optorch at main · apple/ml-sigmoid-attention</a>: Contribute to apple/ml-sigmoid-attention development by creating an account on GitHub.

  

---


### **CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1282894693000286282)** (24 messages🔥): 

> - `Sigmoid Attention Paper`
> - `FlashSigmoid vs FA3`
> - `Bias in Sigmoid Attention`
> - `Elementwise Sigmoid vs Rowwise Softmax`
> - `LayerScale` 


- **Discussion on the Sigmoid Attention Paper**: Members shared interest in the [Sigmoid Attention paper](https://arxiv.org/abs/2409.04431), with some expressing skepticism about attention modifications and potential output bounds.
   - Concerns were addressed regarding how the addition of bias before the sigmoid function might handle these issues.
- **Comparing FlashSigmoid and FA3**: A member noted that it's curious how the performance of **FlashSigmoid** stacks up against **FA3**, especially since FA3 is optimized for hopper GPUs.
   - There was discussion on whether comparisons should be made between FlashSigmoid and a revision of FA3 using elementwise sigmoid rather than FA2.
- **Clarification on Bias and Reductions in Sigmoid Attention**: It was clarified that the bias `b` in the sigmoid attention is fixed and computed as `b=-log(L)`, where `L` is the sequence length, eliminating the need for a reduction step.
   - This property aims to make the sum of logits approximate to 1, similar to softmax, thus avoiding unbounded outputs.
- **Validity of Elementwise Sigmoid vs Rowwise Softmax**: Concerns were raised about the validity of using **elementwise sigmoid** instead of **rowwise softmax**, focusing on its performance and implementation details.
   - Members discussed the implications of this substitution and the potential performance differences in various implementations.
- **Curiosity Around LayerScale**: One member expressed curiosity about the function of **LayerScale** within the context of the ongoing discussions.
   - This sparked interest in exploring how it ties into the performance of strategies discussed, particularly regarding attention.



**Link mentioned**: <a href="https://arxiv.org/abs/2409.04431">Theory, Analysis, and Best Practices for Sigmoid Self-Attention</a>: Attention is a key part of the transformer architecture. It is a sequence-to-sequence mapping that transforms each sequence element into a weighted sum of values. The weights are typically obtained as...

  

---


### **CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1282945115132006461)** (5 messages): 

> - `Tiling concept for matrix multiplication`
> - `Pragma unroll usage`
> - `Matrix multiplication resources` 


- **Tiling Concept for Matrix Multiplication Explained**: A member is seeking clarity on the **tiling concept** for matrix multiplication and is looking for relevant resources or guidance to better understand it.
   - Another member recommended a useful [animation](https://youtu.be/Q3GgbfGTnVc?si=ejkL0DRD70uXn7lZ&t=142) to aid in comprehension.
- **Confusion Over Pragma Unroll Effectiveness**: A member noted that they have encountered issues using **pragma unroll**, expressing that it does not seem to enable multithreading as expected.
   - In response, another member advised that typically **pragma unroll** is redundant, as most compilers handle it automatically.


  

---


### **CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1282943367524778006)** (3 messages): 

> - `Tiling concept for matrix multiplication`
> - `Matrix multiplication optimization resources` 


- **Struggling with Tiling in Matrix Multiplication**: A member expressed their difficulty in understanding the **tiling concept for matrix multiplication** and asked for recommendations.
   - *Redrawing with pencil and paper* was suggested as a potential help to grasp the concept better.
- **Resource for Matrix Multiplication Optimization**: Another member shared an insightful article on optimizing an implementation of **matrix multiplication in CUDA** which focuses on performance characteristics like memory coalescing and caching.
   - The article includes code for all kernels and references other helpful repositories, emphasizing the significance of matrix multiplication in training **large deep-learning models**.



**Link mentioned**: <a href="https://siboehm.com/articles/22/CUDA-MMM">How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog</a>: In this post, I’ll iteratively optimize an implementation of matrix multiplication written in CUDA.My goal is not to build a cuBLAS replacement, but to deepl...

  

---


### **CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/)** (1 messages): 

pauleonix: Any suckerpinch fans around here? 😆 https://youtu.be/Ae9EKCyI1xU
  

---


### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1283151469268176916)** (7 messages): 

> - `Activation Value Saving`
> - `Activation Checkpointing`
> - `Memory Optimization Techniques`
> - `Liger Kernel Memory Management` 


- **Storing Activation Values for Backward Pass**: A member confirmed that they **save activation values** after applying the activation function for the backward pass, enhancing computational efficiency.
   - There's also a discussion on optional activation checkpointing and the **ambitious** FP8/tensor changes being implemented.
- **Recalculating Outputs to Save Memory**: There’s a clever method proposed to save memory by recalculating outputs like **GELU** or **LayerNorm** instead of saving them.
   - This strategy can lead to a noticeable percentage of **total memory savings** during model training.
- **Activation Gradients Memory Efficiency**: It's noted that **activation gradients** can reuse existing buffers, resulting in effectively **zero additional memory usage**.
   - This technique streamlines memory management, allowing for more efficient utilization.
- **Liger Kernel's Chunking Approach**: A question arose regarding the **Liger Kernel's** method of chunking logits/dlogits to reduce memory usage significantly, down to 1/X of the original amount.
   - The approach entails performing **X smaller matrix multiplications**, which may enhance performance, though it’s unclear if this has been fully considered.


  

---


### **CUDA MODE ▷ #[cudamode-irl](https://discord.com/channels/1189498204333543425/1267896441989234709/1283020405770158232)** (11 messages🔥): 

> - `CUDA-MODE IRL Event Details`
> - `Quantization and Sparsity Projects`
> - `GPU Availability for Hacking` 


- **CUDA-MODE IRL Event on September 21st**: The upcoming CUDA-MODE IRL event is set for **September 21st** from **10 AM to Midnight** with limited space for **150 participants** despite **650 applicants**.
   - Participants are urged to confirm their RSVPs quickly for the hacking session scheduled from **1 PM to 11 PM**, with notable keynote speaker **Wen-mei Hwu**.
- **Exciting Quantization and Sparsity Projects Planned**: A range of projects focusing on **quantization and sparsity** has been prepared, aimed at drastically **reducing VRAM usage** and costs during deployment.
   - The projects are categorized as **High Performance Implementation**, **Research Projects**, and **Quantization Flow Projects**, with a [detailed list available here](https://docs.google.com/document/d/14BJ7a1wx1uqzrmCbBsLjX7YQNKrtIhRDuz1l0Tlmkoc/).
- **Inquiry on Available GPUs for Hack Session**: A participant inquired about the **availability of GPUs** for hacking, specifically asking if **AMD GPUs** would be included.
   - In response, a member confirmed that a variety of GPUs from multiple vendors will be available, promising further details on **compute sponsors soon**.



**Link mentioned**: <a href="https://docs.google.com/document/d/14BJ7a1wx1uqzrmCbBsLjX7YQNKrtIhRDuz1l0Tlmkoc/">Quantization and Sparsity Projects</a>: Quantization and Sparsity Projects for IRL  High Performance Implementation Projects:  1. Develop an A16W3 (mixed fp16 x 3-bit) Fused Matmul Kernel: Why? Currently, there is no available kernel for 3-...

  

---


### **CUDA MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/1282789842341855337)** (7 messages): 

> - `Benchmarking phi3`
> - `GPU utilization concerns`
> - `OOM issues with sequence length`
> - `GPU CI failures` 


- **Struggles with phi3 Benchmarking on A100 GPU**: A user is attempting to create benchmarks on **phi3** using a single A100 40GB but is facing challenges with token throughput, leading to an [issue raised on GitHub](https://github.com/linkedin/Liger-Kernel/issues/236).
   - They adapted the [Hugging Face example](https://github.com/linkedin/Liger-Kernel/tree/main/examples/huggingface) and are considering distributed training to improve results.
- **Concerns about GPU Utilization**: Another user suggested that the **GPU** is likely underutilized due to a small batch size and sequence length, which may be impacting performance.
   - This sentiment was echoed in the conversation, indicating the potential for optimization in the benchmark conditions.
- **OOMKilled Errors with High Sequence Length**: The original user reported experiencing **OOMKilled** errors when sequence lengths exceeded **512**, indicating memory constraints on the GPU.
   - This sparked a discussion on memory bandwidth limitations related to the **40GB** capacity of their GPU.
- **Issues with GPU CI Builds**: A member queried about failures in the **GPU CI**, prompting a response that acknowledged the ongoing issue and the efforts in progress to fix it.
   - This indicates a collaborative effort within the community to address CI pipeline problems.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/linkedin/Liger-Kernel/issues/236">Benchmarking phi3 on single A100 40gb GPU: unable to reproduce benchmark results · Issue #236 · linkedin/Liger-Kernel</a>: 🐛 Describe the bug I&#39;m using flyte to reproduce the token throughput and memory savings results reported in this repo&#39;s README under slightly different conditions: using the microsoft/Phi-3-m...</li><li><a href="https://github.com/linkedin/Liger-Kernel/tree/main/examples/huggingface">Liger-Kernel/examples/huggingface at main · linkedin/Liger-Kernel</a>: Efficient Triton Kernels for LLM Training. Contribute to linkedin/Liger-Kernel development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1282813271367356437)** (31 messages🔥): 

> - `Cohere's Acceptable Use Policy`
> - `Fine-tuning Models`
> - `Community Introductions`
> - `Bot Maintenance Updates` 


- **Cohere's Acceptable Use Policy**: A member shared a link to [Cohere's Acceptable Use Policy](https://docs.cohere.com/docs/c4ai-acceptable-use-policy) detailing prohibited use cases, including **violence** and **harassment**.
   - The community discussed **commercial use** implications and compliance with local laws when using model derivatives.
- **Discussion on Fine-tuning Models**: A member inquired about the **fine-tuning** policy for CMD-R models, specifically if it was free to use.
   - Another member clarified that **self-hosted** models have restrictions against any commercial use.
- **Warm Welcomes and Introductions**: Several new members introduced themselves to the community, sharing their reason for joining and expressing excitement.
   - Members welcomed newcomers and engaged in conversation about their projects and interests.
- **Updates on Bot Maintenance**: A community member announced that another member is working to fix the **cotector bot**, aiming to restore its functionality.
   - Others expressed enthusiasm about this takeover, with expectations of completion soon.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.cohere.com/docs/c4ai-acceptable-use-policy">Cohere For AI Acceptable Use Policy — Cohere</a>: C4AI Acceptable Use Policy</li><li><a href="https://docs.cohere.com/docs/c4ai-acceptable-use-policy?_gl=1*121mhsp*_gcl_aw*R0NMLjE3MjUxNDE4NzcuRUFJYUlRb2JDaE1JbkpqLW9wNmdpQU1WcmtYX0FSMXpCeW82RUFBWUFTQUFFZ0s1RFBEX0J3RQ..*_gcl_au*MTU4MDQyNzY2Ny4xNzI1MDU4MTcx*_ga*NzgwODE4Mzk0LjE3MjUwNTgyMTE.*_ga_CRGS116RZS*MTcyNTkxOTE0MS41LjEuMTcyNTkxOTE3Mi4yOS4wLjA">Cohere For AI Acceptable Use Policy — Cohere</a>: C4AI Acceptable Use Policy
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1282970322467029012)** (2 messages): 

> - `Embedding documents`
> - `Fine-tuning LLMs` 


- **Choose Embedding for Quick Responses**: One member suggested using embedding if processing a small number of documents for immediate responses, whereas for large batches (100K+), embedding jobs are preferred.
   - The latter option handles aspects like **validation** and **batching**, ensuring a smoother operation.
- **Seeking Resources on Fine-tuning LLMs**: Another member inquired about suggestive videos or books on how to fine-tune **LLMs**.
   - No specific resources were provided in the chat, indicating a potential gap in shared knowledge on the topic.


  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1282973213688856596)** (1 messages): 

> - `Temperature settings in outputs` 


- **Testing Temperature Settings**: A member suggested experimenting with different temperature settings, specifically **0** or **0.1**, to assess variations in output quality.
   - This approach aims to determine if the outputs obtained are **wildly different** from the initial examples provided.
- **Concerns Over Output Differences**: Another member expressed interest in understanding if the variations in outputs were significant compared to examples given.
   - The discussion revolves around refining the output quality through temperature adjustments.


  

---


### **Cohere ▷ #[projects](https://discord.com/channels/954421988141711382/1218409701339828245/1282799970528919572)** (42 messages🔥): 

> - `Advanced Computer Vision Projects`
> - `Multimodal Learning`
> - `Pokedex Project`
> - `Google Vision API`
> - `Team Collaboration` 


- **Exploring Advanced Computer Vision Projects**: A member requested suggestions for **advanced project ideas** in **computer vision**, expressing a desire to learn something new with **Cohere**.
   - Another member recommended looking at intersections between computer vision and **LLM projects** for inspiration.
- **Team Collaboration Enhances Success**: Discussion emphasized the importance of teamwork in projects, with a member noting that having a **great team** contributed significantly to their project's success.
   - This prompted ideas about teaming up to overcome common issues of starting projects but struggling to finish them.
- **The Fun of the Pokedex Project**: A member shared details about a fun **Pokedex project** that used **Google Vision API**, **Cohere LLMs**, and *Wombo* for image generation to create a unique user experience.
   - The project identified **Pokemon** names and descriptions based on prominent features of images, integrating various technologies in a creative way.
- **Using Google Vision API for Image Labels**: Questions arose about the role of the **Google Vision API** in the Pokedex project, leading to clarification that it was used to **create image labels** rather than learning embeddings.
   - This sparked further discussion on utilizing available datasets, with the mention of **Kaggle** as a resource for **Pokemon datasets**.
- **The Struggle to Complete Projects**: A member expressed frustration over **switching** between projects without completing them, attributing it to a 'major skill issue'.
   - Another member offered advice, suggesting that teaming up could help overcome this common struggle of easy initiation but difficulty in finishing.


  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1282793265573138574)** (14 messages🔥): 

> - `Windows usage`
> - `Desktop beta`
> - `Android mobile devices`
> - `Open Interpreter product discussion`
> - `Project issues` 


- **Exploring Windows Usage**: A member inquired about how to use the project on **Windows**.
   - This reflects a common interest in the platform's compatibility across operating systems.
- **Inquiry on Desktop Beta Access**: A question arose regarding whether it was too late to join the **desktop beta** program.
   - This highlights ongoing interest in accessing new features introduced in the beta.
- **Finding Android Mobile Devices**: Someone asked where they could purchase one of the **Android mobile devices**, and a linked response was provided to an Amazon product.
   - This shows enthusiasm for mobile integration with the project.
- **Discussion on Real Products**: A member questioned the existence of any **real products**, prompting a defense with a link to the **Open Interpreter GitHub** repositories.
   - This indicates a critical perspective within the community about product availability.
- **Issues with Mobile App Feedback**: A new member reported issues with the mobile app not providing feedback after tasks on **Android**.
   - They were advised to create an issue in the appropriate channel, signaling active support for troubleshooting.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.amazon.com/FDMDAF-smartphone-Cellphone-Lightweight-Unlocked/dp/B0CYGZFC54">no title found</a>: no description found</li><li><a href="https://github.com/OpenInterpreter">Open Interpreter</a>: Open Interpreter has 5 repositories available. Follow their code on GitHub.
</li>
</ul>

</div>
  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1282778397914824705)** (57 messages🔥🔥): 

> - `01 Light Discontinuation`
> - `Refund Process`
> - `01 App Launch`
> - `Testing and Beta Feedback`
> - `Community Support for Open Source` 


- **01 Light Manufacturing Discontinued**: The team announced the discontinuation of **01 Light** manufacturing and has refunded all hardware orders as they shift focus to the **01 App** for Android and iOS.
   - They emphasized the decision was made to enhance software development and prioritize their open-source vision, receiving appreciation from the community.
- **Refunds Processed for Hardware Purchases**: Many members confirmed receiving their refunds, indicating that the process was swift and efficient; in some cases, refunds were processed without user requests.
   - The clarity and speed of the refund process prompted positive feedback from users regarding the management of the situation.
- **Launch of 01 App for Mobile Devices**: The **01 App** is now available for both Android and iOS, with future plans to enhance its functionalities based on user feedback.
   - Developers are encouraged to fork the app on GitHub to create tailored experiences, expanding accessibility across a variety of devices.
- **Request for Beta Testing the Official Desktop App**: Several users inquired about beta testing the **official desktop app**, seeking to participate in the feedback process to optimize its performance.
   - The team clarified that the release timeline depends on feedback from current beta testers, maintaining engagement with the community.
- **Support for Open Source Development**: The community expressed enthusiasm for contributing to open-source projects stemming from the 01 platform, highlighting the potential for new software initiatives.
   - Participants shared their willingness to assist in development and troubleshooting, reinforcing the collaborative spirit within the community.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://changes.openinterpreter.com/log/01-app">Open Interpreter - It should have been an app</a>: Official changelog for the open-source Open Interpreter project.</li><li><a href="https://01.openinterpreter.com/software/server/livekit-server">Livekit Server - 01</a>: no description found</li><li><a href="https://changes.openinterpreter.com/log/01-app)">Open Interpreter Changelog</a>: Official changelog for the open-source Open Interpreter project.
</li>
</ul>

</div>
  

---


### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1283059461690494986)** (5 messages): 

> - `Tool Use Episode Release`
> - `YouTube Links` 


- **Tool Use Episode 4 Launch**: The latest episode titled *'Activity Tracker and Calendar Automator - Ep 4 - Tool Use'* is now available on [YouTube](https://www.youtube.com/watch?v=N9GCclB8rYQ). **Mike Bird** and **Ty Fiero** discuss optimizing **time management** with AI.
   - The video emphasizes that **time is our most precious resource**, aiming to inspire viewers to utilize tools effectively.
- **Exciting Content Overflow!**: Members expressed excitement about the substantial amount of new content released today, spurring enthusiasm across the channel.
   - A member shared a [YouTube video](https://www.youtube.com/watch?v=FAFmP82bhDA) that also adds to the day's content richness.



**Link mentioned**: <a href="https://www.youtube.com/watch?v=N9GCclB8rYQ">Activity Tracker and Calendar Automator - Ep 4 - Tool Use</a>: Time is our most precious resource, let&#39;s use AI to optimize it!In this episode of Tool Use, Mike Bird and Ty Fiero discuss the importance of time management...

  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1282816199654248561)** (10 messages🔥): 

> - `Windows Native Version`
> - `Focus on Linux Support`
> - `WSL Support`
> - `Community Meeting`
> - `User Feedback Opportunity` 


- **No Timeline for Windows Native Version**: There is currently **no timeline** regarding the availability of a **Windows native version** as Modular is prioritizing support for **Ubuntu and Linux distros**.
   - *Modular aims to avoid tech debt and enhance product quality before broadening their focus,* drawing lessons from past experiences with Swift.
- **WSL Support Available for Windows Users**: While a native **.exe** version is not yet available, *Modular suggests using WSL* as the extent of their current **Windows support**.
   - Users expressed enthusiasm about future native options but acknowledged the current limitations.
- **Challenges in Adding Native Windows Support**: *The team is eager to add native Windows support,* but they lack personnel with the necessary skillset to expedite the process.
   - Despite the challenges, the team remains optimistic about achieving this goal in the future.
- **MAX + Mojo Community Meeting Recording Live**: The recording of the **MAX + Mojo Community Meeting** is now available on YouTube, showcasing engaging presentations.
   - Participants are encouraged to check it out and appreciate the efforts of presenters who contributed to the event.
- **Seeking User Feedback for Magic Product**: Modular is on the lookout for users who have not yet interacted with **Magic** to provide valuable feedback in a quick 30-minute call.
   - Participants will receive exclusive swag for their time and insights, linking to the [booking page](https://modul.ar/user-feedback) for scheduling.



**Link mentioned**: <a href="https://modul.ar/user-feedback">Appointments</a>: no description found

  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1282777564657487945)** (61 messages🔥🔥): 

> - `Mojo language capabilities`
> - `DLHandle in Mojo`
> - `GStreamer bindings`
> - `Variant type in Mojo`
> - `SDL bindings in Mojo` 


- **Mojo's Future with GPU and GStreamer**: Mojo is being positioned as a candidate for replacing GStreamer, leveraging its upcoming GPU capabilities for efficient processing.
   - Members expressed interest in integrating modern libraries for live streaming, highlighting the potential of Mojo to streamline complex tasks.
- **Creating Bindings Using DLHandle**: Several members discussed using DLHandle for creating Mojo bindings, with references to other projects demonstrating its application.
   - Projects like 'dustbin' utilize DLHandle for SDL bindings, providing examples for others interested in graphical applications.
- **Variant Type Usage in Mojo**: Discussion around the Variant type in Mojo highlighted its utility for creating lists with various element types and the associated memory concerns.
   - Members clarified size alignment issues and behavior of discriminants within Variant implementations.
- **SDL and Graphics Programming in Mojo**: Members shared links to SDL bindings projects for Mojo, aiding those looking for simpler examples to integrate graphics into their applications.
   - The first shared project was recommended for providing straightforward examples without the need for a full-fledged application.
- **Beginner Challenges in Mojo Coding**: A member reported an error when checking if characters were digits, discovering that explicit conversion to String was necessary.
   - This raised awareness of type conversion in Mojo and provided insights for beginners facing similar challenges.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.modular.com/mojo/stdlib/utils/variant">variant | Modular Docs</a>: Defines a Variant type.</li><li><a href="https://www.youtube.com/watch?v=JRcXUuQYR90">Mojo Lang - Tomorrow&#39;s High Performance Python? (with Chris Lattner)</a>: Mojo is the latest language from the creator of Swift and LLVM. It’s an attempt to take some of the best techniques from CPU/GPU-level programming and packag...</li><li><a href="https://github.com/modularml/mojo/blob/main/stdlib/src/utils/variant.mojo">mojo/stdlib/src/utils/variant.mojo at main · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1282830724742910114)** (65 messages🔥🔥): 

> - `DisTro confusion`
> - `AI training pitfalls`
> - `OCTAV algorithm implementation`
> - `Repetition in AI responses`
> - `Performance of various AI models` 


- **DisTro causes speculation**: Discussions surfaced around DisTro, with questions about its purpose and effectiveness, leading to confusion among members.
   - One member pointed out that no code has been released yet, suggesting the announcement was made to prompt competition.
- **Concerns Over AI Training Quality**: Many expressed concern that AI models trained on user satisfaction metrics often produce shallow information rather than accurate content.
   - A member emphasized that this trend could lead to a degradation in the quality of AI responses, especially when relying on human feedback.
- **Successful Integration of NVIDIA's OCTAV**: One member shared a successful experience using Sonnet to implement NVIDIA's OCTAV algorithm into their codebase, noting the lack of similar examples online.
   - They speculated whether the implementation was inferred from the paper, showcasing the capabilities of the AI model.
- **Issues with Repetitive Responses**: The group discussed the tendency of AI to provide repetitive outputs and the impact of slight hesitance in user responses.
   - One member remarked how quickly models like Claude retract their solutions, indicating a need for improvement in maintaining confidence in their outputs.
- **Mixed Performance of Various AI Models**: The performance of platforms like Claude and Opus was evaluated, with members commenting on their strengths and weaknesses in discourse.
   - One member noted that while Claude has a good alignment strategy, it tends to falter under certain conditions, unlike Opus, which they found more engaging.



**Link mentioned**: <a href="https://x.com/abacaj/status/1833247396278726966">Tweet from anton (@abacaj)</a>: Could not reproduce the 91% humaneval score for reflection (ref_70_e3), run locally using bf16 with vLLM. Used the &#34;recommended&#34; system prompt + extracting from output tags: 81.1%  meta-llama-...

  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1283094422032224266)** (6 messages): 

> - `Scaling in AI`
> - `Quality of Data`
> - `Rich Sutton's Bitter Lesson`
> - `AI Research Trends` 


- **Scaling: Hype or Solution?**: Members debated whether scaling is merely a hype or if it truly holds the key to effective reasoning in AI, emphasizing the importance of **quality data** over other factors.
   - One member highlighted that previous experts reiterated that **quality data** is paramount, raising questions about the real impact of scaling.
- **Rich Sutton's Insights on AI Evolution**: A shared article from [Rich Sutton](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) emphasizes that the most effective methods in AI rely on **leveraging computation** over human knowledge, particularly referencing Moore's Law.
   - Sutton argues that historical reliance on human expertise can detract from progress, making clear that what ultimately matters is maximizing available computation.
- **Quality Data Enhancing Scaling**: Discussion ranged to how **higher quality data** not only improves model performance but also complements scaling, suggesting synergistic benefits.
   - Members acknowledged continuous evidence indicating that as AI parameters scale, the enhancement in performance is more pronounced with better data.



**Link mentioned**: <a href="http://www.incompleteideas.net/IncIdeas/BitterLesson.html">The Bitter Lesson</a>: no description found

  

---



### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1282805762053111971)** (53 messages🔥): 

> - `Tokenizer eos issue`
> - `Eleuther_Eval recipe loading`
> - `ChatML format for datasets`
> - `Checkpointing in training`
> - `Hugging Face TRL library` 


- **Tokenizer eos option missing from Mistral and Gemma**: A user proposed sending a PR to fix the tokenizer eos problem, noting that current Mistral and Gemma tokenizers do not have the `add_eos` option.
   - Another member highlighted that they first need to implement the `add_eos` feature before it can be fixed, referencing a [utility that needs updating](https://github.com/pytorch/torchtune/blob/main/torchtune/modules/tokenizers/_utils.py).
- **Eleuther_Eval recipe defaults to GPT-2 model**: A member inquired why the Eleuther_Eval recipe always loads the GPT-2 model, which was clarified as being the default model type since `lm_eval==0.4.3`. 
   - They mentioned that the model definition is overwritten with their `TransfomerDecoder` tools to evaluate other models.
- **Formatting datasets in ChatML format**: New user expressed challenges with formatting their dataset in a ChatML compatible manner for Qwen2, seeking guidance on sample format.
   - A member recommended using the `ShareGPTToMessages` class for transforming data into a ChatML format, offering to provide a code example later on.
- **Checkpointing during training**: A user discussed implementing model checkpointing every 100 million tokens processed while tracking total tokens across nodes.
   - Feedback included suggestions for checking token counts, particularly regarding ignoring padded tokens.
- **Switch to Hugging Face TRL library**: A user decided to switch to using the Hugging Face TRL library for their fine-tuning project after facing difficulties with dataset scripts.
   - Others highlighted alternative approaches and libraries like Axolotl for additional configurability.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://pytorch.org/torchtune/main/generated/torchtune.data.ShareGPTToMessages.html#torchtune.data.ShareGPTToMessages),">ShareGPTToMessages &mdash; torchtune main documentation</a>: no description found</li><li><a href="https://pytorch.org/torchtune/stable/_modules/torchtune/data/_chat_formats.html#ChatMLFormat">torchtune.data._chat_formats &mdash; torchtune 0.2 documentation</a>: no description found</li><li><a href="https://huggingface.co/Qwen/Qwen2-7B-Instruct/blob/main/tokenizer_config.json">tokenizer_config.json · Qwen/Qwen2-7B-Instruct at main</a>: no description found</li><li><a href="https://pytorch.org/torchtune/stable/tutorials/datasets.html#local-and-remote-datasets.">Configuring Datasets for Fine-Tuning &mdash; torchtune 0.2 documentation</a>: no description found</li><li><a href="https://github.com/pytorch/torchtune/blob/66590b408b64fcff32a8b75b84f592b4e1530a00/torchtune/datasets/_sft.py#L108C22-L108C34">torchtune/torchtune/datasets/_sft.py at 66590b408b64fcff32a8b75b84f592b4e1530a00 · pytorch/torchtune</a>: A Native-PyTorch Library for LLM Fine-tuning. Contribute to pytorch/torchtune development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1282909765051220060)** (17 messages🔥): 

> - `Mixed Precision Training`
> - `Liger vs Compile Speed`
> - `Dynamic seq_len Challenges`
> - `Chunked CE Memory Usage`
> - `FP8 Integration Ideas` 


- **Mixed Precision Training yields promising results**: A member shared excitement about the results of their implementation of **mixed precision training** including **cpuoffloadingOptimizer** and its potential to boost **TPS**.
   - However, they expressed uncertainty about how it will interact with **FSDP+Compile+AC**, indicating the need for further testing.
- **Compile Speed Outshines Liger**: Benchmarks on **single GPU** showed that using `compile(linear+CE)` is faster in speed and memory than both **Horace's implementation** and **Liger**.
   - Contrastingly, chunkedCE was found to save more **memory** when compiled independently but was ultimately *slower*.
- **Dynamic seq_len presents optimization hurdles**: Concerns were raised about **dynamic seq_len** in **torchtune**, affecting the **INT8 matmul triton kernel** due to re-autotuning.
   - Strategies discussed included padding inputs to multiples of **128**, although it incurred extra padding costs that may limit speed improvements.
- **Chunked CE Collaboration Inquiries**: A member inquired about the effectiveness of using **mark_dynamic** during recompilation to reduce compile time, referencing a [GitHub PR](https://github.com/pytorch/torchtune/pull/1445).
   - They highlighted that this PR aims to improve compile time significantly on their **A100 machine**, which may benefit users dealing with similar challenges.
- **FP8 Integration for Enhanced Performance**: Discussion arose about the potential of integrating **FP8** for increased speed, albeit limited to **H100 GPUs** and possibly beneficial for consumer **4xxx GPUs**.
   - A member noted the lack of **end-to-end benchmarks** for FP8 on consumer GPUs, raising the need for more data in this area.



**Link mentioned**: <a href="https://github.com/pytorch/torchtune/pull/1445">Reduce compile time for single-device and multi-device recipes by yf225 · Pull Request #1445 · pytorch/torchtune</a>: Context What is the purpose of this PR? Is it to   add a new feature  fix a bug  update tests and/or documentation  other (improve compile time)  Improvements in compile time (on my A100 machine): ...

  

---



### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1282810086581866609)** (1 messages): 

> - `Jim Harbaugh`
> - `Perplexity Playbook`
> - `Social Media Updates` 


- **Jim Harbaugh endorses Perplexity**: Head coach **Jim Harbaugh** emphasized that a great playbook isn't complete without **Perplexity** in a recent announcement.
   - He also invited fans to [ask him anything](https://x.com/perplexity_ai/status/1833173842870853896) related to the topic.
- **Video Updates Featuring Perplexity**: A new video highlighting **Perplexity** was shared across multiple platforms, including [LinkedIn](https://www.linkedin.com/feed/update/urn:li:activity:7238939748407353344/) and [Instagram](https://www.instagram.com/reel/C_tJOyxSxXX/).
   - The video aims to showcase the integration of Perplexity into coaching strategies.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/perplexity_ai/status/1833173842870853896)">Tweet from Perplexity (@perplexity_ai)</a>: Ask Jim Harbaugh anything.</li><li><a href="https://www.instagram.com/reel/C_tJOyxSxXX/)">Perplexity AI on Instagram: &quot;Ask &#064;jimharbaugh anything.&quot;</a>: 249 likes, 8 comments - perplexity.ai on September 9, 2024: &quot;Ask &#064;jimharbaugh anything.&quot;. 
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1282805310297214997)** (57 messages🔥🔥): 

> - `Reflection LLM addition`
> - `Perplexity Pro rewards issue`
> - `Claude 3.5 performance concerns`
> - `Search functionality problems`
> - `User prompts and formatting` 


- **Inquiry about Reflection LLM**: A member inquired if the **Reflection LLM** is being added to Perplexity soon, showing interest in upcoming features.
   - No definitive answers or information were provided in the discussion regarding this potential update.
- **Xfinity promo code frustrations**: A user expressed frustration over issues with the **Perplexity Pro rewards deal** from Xfinity, stating the promo code was deemed invalid.
   - The community discussed possible solutions, including the necessity of creating a new account to use the promo.
- **Concerns over Claude 3.5 capabilities**: Several users noted that **Claude 3.5** appears to have degraded in performance, questioning if there are capacity issues despite recent investments.
   - Users shared their experiences of confusion regarding the model version indicated in their settings.
- **Search function and previous uploads**: Members voiced their dissatisfaction with **search functionality**, pointing out limitations and the inability to delete previously uploaded files.
   - There was frustration over the fact that necessary capabilities to manage uploads are still lacking.
- **Prompt generation issues**: A user sought advice on how to stop AI responses from using repetitive formats like '___ isn't just ___, it is about ___'.
   - Another member suggested using explicit restrictions in the profile settings, although it did not entirely resolve the concern.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/perplexity_ai/status/1833173842870853896">Tweet from Perplexity (@perplexity_ai)</a>: Ask Jim Harbaugh anything.</li><li><a href="https://tenor.com/bKbSI.gif">Youtube Youtube Channel GIF - Youtube Youtube Channel Shorts - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://x.com/shinboson/status/1832933747529834747">Tweet from 𝞍 Shin Megami Boson 𝞍 (@shinboson)</a>: A story about fraud in the AI research community:  On September 5th, Matt Shumer, CEO of OthersideAI, announces to the world that they&#39;ve made a breakthrough, allowing them to train a mid-size mod...
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1282801925909119027)** (6 messages): 

> - `Apple iPhone event`
> - `AI detecting fake science`
> - `Nvidia Q2 earnings`
> - `Artistic journalism`
> - `Top IDEs for programming` 


- **Apple iPhone Event Highlights**: The latest [Apple iPhone event](https://www.perplexity.ai/search/what-are-the-top-10-ides-for-n-NkpW74i6TCShNE_L_eKCCA) showcased new features and innovations, exciting tech enthusiasts.
   - It drew parallels with last year's trends and set the stage for futuristic developments in mobile technology.
- **AI Spots Fake Science Articles**: A recent discussion focused on advancements where AI can now effectively **spot fake science articles**, enhancing media credibility.
   - The application of such technology is expected to improve public awareness and inform scientific literacy.
- **Nvidia Beats Q2 Earnings Expectations**: Nvidia [beat Q2 expectations](https://www.perplexity.ai/page/nvidia-beats-q2-expectations-k9CT.KnRT1uKI8OG99kdrA) with impressive graphics card sales and robust AI sector performance.
   - Analysts noted that this performance underscores Nvidia's dominance in the tech market amid increasing demand for AI solutions.
- **Artistic Journalism Takes Center Stage**: An intriguing piece titled 'RIP Darth Vader / Mufasa' discusses the fusion of storytelling and art in journalism, found [here](https://www.perplexity.ai/search/create-an-artistic-journalist-6kdUu0iSRLqpIj91.Mv9Hw#0).
   - This approach raises questions about the role of creativity in traditional reporting formats.
- **Exploring Best Practices for Automation**: A discourse on [best practices](https://www.perplexity.ai/search/best-practice-to-automate-rag-vs8U.kvuQqqJrVFbJqn_Rw) for automating various processes revealed innovative strategies.
   - Participants shared insights on increasing efficiency and mitigating error rates through automation tools.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1282803835466809365)** (2 messages): 

> - `search_domain_filter API`
> - `API functionality` 


- **Discovering `search_domain_filter` in API**: A member shared insights about the `search_domain_filter` parameter in the API, explaining that it allows users to control which domains the model can search.
   - *Will look more into it, thank you!* expressed another member's interest in exploring this functionality further.
- **Interest in API features**: Another user showed enthusiasm for the information provided about the API functionality, indicating a willingness to learn more about it.
   - The community seems eager to explore the capabilities of the API, reflecting a collaborative atmosphere.


  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1282777642072014880)** (47 messages🔥): 

> - `Apple Intelligence updates`
> - `ColPali model advancements`
> - `Superforecasting AI release`
> - `Strawberry OpenAI model`
> - `Expand.ai launch` 


- **Apple Intelligence Set for Updates**: Apple plans to introduce significant updates to its Intelligence capabilities within two weeks, enhancing Siri and other AI functionalities.
   - Users noted Apple may have fixed long-standing issues, setting a competitive landscape against OpenAI.
- **ColPali and AI Model Improvements**: ColPali is being highlighted in discussions, with new slides showcasing its implementation and effectiveness in AI tasks.
   - The integration of models like ColPali with training methods could redefine existing approaches to AI research.
- **Launch of Superforecasting AI**: A new AI developed for superforecasting has been released, demonstrating capabilities to predict outcomes with superhuman accuracy.
   - This tool aims to automate prediction markets, offering a demo and blog detailing its functionalities.
- **OpenAI's Strawberry Model on the Horizon**: OpenAI is set to release the Strawberry model soon, which is designed for better reasoning and more detailed task execution.
   - While promising significant improvements, there are concerns about response timing and memory handling in its initial prototype.
- **Launch of Expand.ai**: Tim Suchanek announced the launch of Expand.ai, a tool designed to turn websites into type-safe APIs amid its participation in Y Combinator's current batch.
   - The service aims to streamline data retrieval from websites, drawing interest from both tech and non-tech users.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/shinboson/status/1832933747529834747">Tweet from 𝞍 Shin Megami Boson 𝞍 (@shinboson)</a>: A story about fraud in the AI research community:  On September 5th, Matt Shumer, CEO of OthersideAI, announces to the world that they&#39;ve made a breakthrough, allowing them to train a mid-size mod...</li><li><a href="https://x.com/OfficialLoganK/status/1833226001670934827">Tweet from Logan Kilpatrick (@OfficialLoganK)</a>: We just shipped a new variant of Structured Outputs in the Gemini API called Enum Mode, which allows you to easily constrain the model to pick between pre-defined options 🚢</li><li><a href="https://x.com/danhendrycks/status/1833152719756116154?s=46">Tweet from Dan Hendrycks (@DanHendrycks)</a>: We&#39;ve created a demo of an AI that can predict the future at a superhuman level (on par with groups of human forecasters working together). Consequently I think AI forecasters will soon automate m...</li><li><a href="https://www.safe.ai/blog/forecasting">Superhuman Automated Forecasting | CAIS</a>: This post describes a superhuman forecasting AI called FiveThirtyNine, which generates probabilistic predictions for any query by retrieving relevant information and reasoning through it. We explain h...</li><li><a href="https://x.com/steph_palazzolo/status/1833508052835909840?s=46">Tweet from Stephanie Palazzolo (@steph_palazzolo)</a>: New w/ @erinkwoo @amir:  OpenAI is planning to release Strawberry as part of ChatGPT in the next 2 weeks.   We have more exclusive details on the new model&#39;s strengths and weaknesses here:  https:...</li><li><a href="https://x.com/realgenekim/status/1833298959890321503?s=46">Tweet from Gene Kim (@RealGeneKim)</a>: I can&#39;t tell you how elated I am that I could generate this thread from @headinthebox&#39;s talk.  I&#39;ve written about how I&#39;ve been taking screenshots of YouTube videos and podcast players...</li><li><a href="https://x.com/glean/status/1833476578912989281?s=61">Tweet from Glean (@glean)</a>: 🎉 We’ve raised over $260M at a $4.6B valuation co-led by Altimeter Capital and DST Global.  And that’s not all! Introducing next-gen prompting capabilities that expand the use of our Work AI platform...</li><li><a href="https://x.com/TimSuchanek/status/1833538423954804948">Tweet from Tim Suchanek (@TimSuchanek)</a>: 🚀 After an amazing time at Stellate, I&#39;ve decided to start a new business. I&#39;ve founded http://expand.ai, and we&#39;re in the current YC batch - S24!  For techies: http://expand.ai instantly...</li><li><a href="https://x.com/jxnlco/status/1833555318590329073?s=46">Tweet from jason liu (@jxnlco)</a>: congrats to http://expand.ai!  for everyone else, expand ai at home ;)</li><li><a href="https://engineering.fractional.ai/taming-llm-responses-dynamic-pydantic-models-for-flexible-structured-output">Taming LLM Responses: Dynamic Pydantic Models for Flexible Structured Output</a>: As developers working with Large Language Models (LLMs), we often grapple with the challenge of constraining their outputs to meet our specific needs. In this post, I will share a technique I develope...</li><li><a href="https://x.com/danhendrycks/status/1833163197626601603?s=46">Tweet from Dan Hendrycks (@DanHendrycks)</a>: This is the prompt that does the heavy lifting</li><li><a href="https://x.com/bclavie/status/1831431500161806562?s=46">Tweet from Benjamin Clavié (@bclavie)</a>: Full slides for this talk are here: https://docs.google.com/presentation/d/1Zczs5Sk3FsCO06ZLDznqkOOhbTe96PwJa4_7FwyMBrA/edit#slide=id.p  Expect a lot of ColBERT and ColPali, with a tiny SLADE and BM25...</li><li><a href="https://x.com/tjcages/status/1833218417639186936?s=46">Tweet from tylerj (@tjcages)</a>: inspo → claude prompt → working code in ~15m  this is the new 10x eng  Quoting Tatiana Tsiguleva (@ciguleva)   It could be a Midjourney ad, but...  Part 2</li><li><a href="https://x.com/chengleisi/status/1833166031134806330?s=46">Tweet from CLS (@ChengleiSi)</a>: Automating AI research is exciting! But can LLMs actually produce novel, expert-level research ideas?  After a year-long study, we obtained the first statistically significant conclusion: LLM-generate...</li><li><a href="https://x.com/_xjdr/status/1833178647483875729?s=46">Tweet from xjdr (@_xjdr)</a>: i&#39;m not sure if this is blackpill or whitepill, but my there are a heap of new papers along with my own experiences that are showing &#34;best of N is all you need&#34; for most problems as long a...</li><li><a href="https://docs.google.com/presentation/d/1Zczs5Sk3FsCO06ZLDznqkOOhbTe96PwJa4_7FwyMBrA/edit#slide=id.p">RAG_Beyond_Dense</a>: RAG is more than dense embeddings Ben Clavié (@bclavie)</li><li><a href="https://x.com/swyx/status/1833231875537850659">Tweet from swyx 🇸🇬 (@swyx)</a>: wow. Apple might just have fixed Siri.  and beat OpenAI to the first AI phone.   and commoditized OpenAI with Google.  and casually dropped a video understanding model.  incredibly well executed.  (se...</li><li><a href="https://news.ycombinator.com/item?id=41492172">I&#x27;ve had notification summaries turned on for at least a few weeks as part of th... | Hacker News</a>: no description found</li><li><a href="https://www.reworkd.ai/">Reworkd AI</a>: End to End Web Scraping</li><li><a href="https://github.com/AnswerDotAI/byaldi">GitHub - AnswerDotAI/byaldi: Use late-interaction multi-modal models such as ColPali in just a few lines of code.</a>: Use late-interaction multi-modal models such as ColPali in just a few lines of code. - AnswerDotAI/byaldi</li><li><a href="https://ajcwebdev.com/autogen-shownotes/">Autogenerate Show Notes with Whisper.cpp, Llama.cpp, and Node.js</a>: End-to-end scripting workflow to generate automatic show notes with LLMs from audio and video transcripts using Whisper.cpp, Llama.cpp, and Commander.js.
</li>
</ul>

</div>
  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1283111711913672838)** (2 messages): 

> - `Agentic RAG`
> - `LlamaIndex`
> - `Search For RAG in the LLM era`
> - `Maven course`
> - `RAG strategies` 


- **Agentic RAG Strategies for 2024**: In a recent talk, @seldo explored **Agentic RAG** for 2024, discussing what [LlamaIndex](https://twitter.com/llama_index) is and its significance.
   - Key points included understanding **RAG**'s necessity but limitations, alongside strategies for enhancement.
- **Maven Course on RAG in LLM Era**: A new **Maven course**, *Search For RAG in the LLM era*, features a guest lecture with @jerryjliu0 focusing on live code walkthroughs and hands-on implementations.
   - Participants will benefit from **expert-led sessions** by industry veterans, enhancing their understanding of RAG applications.


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1282779435732303912)** (45 messages🔥): 

> - `LlamaIndex and Llama 3 examples`
> - `Pandas DataFrame querying`
> - `Integration issues with MLflow`
> - `Kapa.ai usage and troubleshooting`
> - `Similarity search methods in LlamaIndex` 


- **LlamaIndex examples with Llama 3**: Members discussed the integration of [LlamaIndex with Llama 3](https://docs.llamaindex.ai/en/stable/examples/llm/ollama/) and provided setup instructions for running a local Ollama instance.
   - Detailed installation steps and usage patterns for LlamaIndex were also shared, including command snippets for Colab.
- **Querying DataFrames with LlamaIndex**: A guide was shared on how to use the `PandasQueryEngine` to convert natural language queries into Python code for Pandas operations, improving accuracy in text-to-SQL.
   - The importance of safety when using this tool was emphasized due to possible arbitrary code execution risks.
- **MLflow-LlamaIndex integration issues resolved**: Members discussed a recent integration issue between MLflow and LlamaIndex that was fixed, expecting a release over the weekend.
   - A member plans to document the experience in a blog article to help others.
- **Kapa.ai usage troubleshooting**: A user inquired about not receiving replies from Kapa.ai, prompting a response that tagging Kapa is necessary for it to respond.
   - Members shared practical examples and links about how to effectively use Kapa.ai in the Discord environment.
- **Similarity search methods in LlamaIndex**: The community discussed how to perform similarity searches using methods like `similarity_search_with_score` in LlamaIndex, noting differences from Langchain's approach.
   - Detailed usage examples were provided, including filtering capabilities when retrieving documents based on metadata.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.llamaindex.ai/en/stable/module_guides/querying/retriever/#retriever">Retriever - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/query_engine/pandas_query_engine/">Pandas Query Engine - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/llm/ollama/">Ollama - Llama 3.1 - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/vector_stores/Qdrant_metadata_filter/">Qdrant Vector Store - Metadata Filter - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/api_reference/storage/vector_store/jaguar/#llama_index.vector_stores.jaguar.JaguarVectorStore.similarity_search_with_score>).">Jaguar - LlamaIndex</a>: no description found</li><li><a href="https://github.com/run-llama/llama_index/blob/8dbb6e91e5984a556756caafbd1d03146e029a51/llama-index-integrations/vector_stores/llama-index-vector-stores-chroma/llama_index/vector_stores/chroma/base.py#L349">llama_index/llama-index-integrations/vector_stores/llama-index-vector-stores-chroma/llama_index/vector_stores/chroma/base.py at 8dbb6e91e5984a556756caafbd1d03146e029a51 · run-llama/llama_index</a>: LlamaIndex is a data framework for your LLM applications - run-llama/llama_index
</li>
</ul>

</div>
  

---



### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1282903282984947744)** (39 messages🔥): 

> - `Deception 70B`
> - `OpenAI's Strawberry release`
> - `Otherside AI scams`
> - `AI forecasting systems`
> - `Exiting OpenAI employees` 


- **Deception 70B Claims to be Top Open-Source Model**: An announcement of **Deception 70B** was made, touted as the world’s top open-source model, utilizing a unique Deception-Tuning method to help LLMs deceive themselves of their mistakes.
   - The release link can be found [here](https://bit.ly/Deception-70B).
- **OpenAI's Strawberry Model to Launch Soon**: OpenAI is planning to release its new model, **Strawberry**, as part of ChatGPT within the next two weeks, according to insiders.
   - Initial impressions suggest it may be underwhelming, as it takes **10-20 seconds** per response and has limitations in memory integration.
- **Concerns Over Otherside AI's Past Scams**: Discussion arose about **Otherside AI**, previously accused of scams, with members referencing issues on GitHub regarding their self-operating computer project that allegedly ripped off open-source work.
   - An ongoing conversation noted the project might be notorious for its misleading claims.
- **AI Forecasting Performance Critiqued**: Dan Hendrycks reported disappointing results from the paper **LLMs Are Superhuman Forecasters**, where AI models underperformed significantly against a new test set.
   - A demo for this AI prediction model is available [here](http://forecast.safe.ai).
- **Notable Departures from OpenAI**: Several employees, including notable figures like **Alex Conneau** and **Arvind**, announced their departure from OpenAI to pursue new ventures, stirring curiosity about the future of their projects.
   - The transition sparked speculation regarding the potential link between these departures and the upcoming **GPT-5** model.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/steph_palazzolo/status/1833508052835909840?s=46">Tweet from Stephanie Palazzolo (@steph_palazzolo)</a>: New w/ @erinkwoo @amir:  OpenAI is planning to release Strawberry as part of ChatGPT in the next 2 weeks.   We have more exclusive details on the new model&#39;s strengths and weaknesses here:  https:...</li><li><a href="https://x.com/imjliao/status/1832970446146593277">Tweet from jian (@imjliao)</a>: @_arohan_ This isn’t their (Otherside AI) first “rodeo”, their previous scam was this self operating computer, which has been accused of ripping off other open source work.  https://github.com/Othersi...</li><li><a href="https://x.com/apples_jimmy/status/1833595024543781088?s=46">Tweet from Jimmy Apples 🍎/acc (@apples_jimmy)</a>: Ok back to October now.   We should have a 4.x model ( maybe still called 4.5, my old friend ) in October.   The big boy gpt 5, I’ve heard as early as December but for your sanity I would have Q1/Q2 2...</li><li><a href="https://x.com/tamaybes/status/1833292271829323939">Tweet from Tamay Besiroglu (@tamaybes)</a>: I&#39;m excited to announce Deception 70B, the world’s top open-source model.   Trained using Deception-Tuning, a technique developed to enable LLMs to deceive themselves of their own mistakes.   Try ...</li><li><a href="https://x.com/alex_conneau/status/1833535309902189015?s=46">Tweet from Alexis Conneau (@alex_conneau)</a>: Career update: After an amazing journey at @OpenAI building #Her, I’ve decided to start a new company.</li><li><a href="https://fxtwitter.com/binalkp91/status/1833470070737014822">Tweet from binal (@binalkp91)</a>: @dannyhalawi15 The web UI for ChatGPT isn’t running the same model as the API. GPT 4o via API does have an October 2023 cutoff.</li><li><a href="https://fxtwitter.com/dannyhalawi15/status/1833295067764953397">Tweet from Danny Halawi (@dannyhalawi15)</a>: The results in &#34;LLMs Are Superhuman Forecasters&#34; don&#39;t hold when given another set of forecasting questions. I used their codebase (models, prompts, retrieval, etc.) to evaluate a new set ...</li><li><a href="https://www.reddit.com/r/singularity/comments/1fdit9r/new_details_on_openais_strawberry_openai_may/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://x.com/arvind_io/status/1833571886766399773?s=46">Tweet from Arvind Neelakantan (@arvind_io)</a>: Excited to join @AIatMeta! The past 4.5 years at @OpenAI,working on embeddings, GPT-3 & 4,API and ChatGPT, have been career highlights. Now, I&#39;m thrilled to work on the next generations of Llama a...</li><li><a href="https://github.com/OthersideAI/self-operating-computer/issues/67">Warning: this project appears to have blatantly ripped off the work of researchers over a year on a new multi-modal model, Atlas-1, and is attempting to scam open source devs into doing that work · Issue #67 · OthersideAI/self-operating-computer</a>: Re-opening @michaelhhogue are you an official contributor to this project? Could you comment on where the name Agent-1 came from? This appears to have blatantly ripped of the work of researchers wo...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1283140696831692933)** (2 messages): 

> - `Gemini and Cursor integration`
> - `User experiences with Cursor` 


- **Gemini Integration with Cursor in Discussion**: Members are discussing the potential of plugging **Gemini** into **Cursor**, expressing curiosity about its functionality.
   - *Has anyone on the discord tried plugging Gemini into Cursor (re: somnambulent GOOG)* suggests some intrigue about **Google's** latest developments.
- **Need for Testing Cursor**: One member expressed urgency by stating, *Need to try cursor, shit*.
   - This indicates a growing interest to explore **Cursor**'s capabilities and possibly share insights with the community.


  

---



### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1282781319717326878)** (41 messages🔥): 

> - `Image Generation Hardware`
> - `Deep Dream Machine Alternatives`
> - `Training Tips for SDXL`
> - `Understanding CLIP Models`
> - `Discord Bot for AI Services` 


- **Image Generation Hardware Discussion**: A member shared insights about their experience with an AMD card, recommending **Linux** for better image generation performance, particularly for local training with a **24G NVIDIA** card.
   - They also suggested ensuring a sufficient power supply, noting they didn't need an upgrade.
- **Exploring Alternatives to Deep Dream Machine**: Opinions were gathered regarding **Deep Dream Machine**, with a suggestion to try **Kling** or **Gen3** as better, potentially cheaper alternatives for AI video creation.
   - One user noted a **66% off** promotion for **Kling**'s first month, drawing interest.
- **Training Tips for SDXL Models**: A member sought advice on tricks for training an **SDXL** model effectively with **Kohya Trainer** to produce high-quality images.
   - Another user indicated the need for a more specific question to receive helpful advice, suggesting to refine it and consider related channels.
- **Clarifying CLIP Model Usage**: A discussion unfolded about which **CLIP models** to use in the **DualCLIPLoader** node in Flux, questioning the scenarios for choosing between **clip g** and **clip l**.
   - It was noted that **Flux** simply wasn't trained with **clip g**, leading to some confusion for users.
- **AI Services via Discord Bot**: A member announced their verified Discord bot that offers text-to-image generation, chat assistance, and image analysis services through a link.
   - This service aims to provide enhanced AI functionalities directly in Discord.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.nterview.me/">Nterview.me - Ultimate Toolbox for Interviewees</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=AfDn_Esqgg8">Nterview.me - Your hidden AI copilot to crush job interviews</a>: Not for interviewers, but interviewees. Your hidden AI copilot for a dream job.
</li>
</ul>

</div>
  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1282801447846678580)** (28 messages🔥): 

> - `Open Source AI Panel`
> - `Performance of AI Models`
> - `Private Machine Learning Solutions`
> - `Multiparty Computation in AI`
> - `Security in Machine Learning Deployment` 


- **GitHub hosts Open Source AI panel**: GitHub is organizing a panel on **Open Source AI** on **9/19** at their SF office featuring panelists from **Ollama**, **Nous Research**, **Black Forest Labs**, and **Unsloth AI**. Registration is free and requires host approval; interested participants can register [here](https://lu.ma/wbc5bx0z).
   - The panel aims to discuss how open source communities foster **access** and **democratization** in AI technology.
- **AI Model Performance raises concerns**: A user tested an AI model and found it to be **impressive** but noted it was an **order of magnitude slower** particularly for larger models. There are concerns that operations on models with **500M parameters** may prove too slow for practical use.
   - Discussion revolves around testing primarily on **small models** from libraries like **sklearn** or **xgboost**, leading to skepticism about performance on larger architectures.
- **Interest in Private Machine Learning**: The conversation highlighted **private machine learning** as an interesting field but lacking effective solutions. Ideas such as **functional encryption** and **zero knowledge proofs** were mentioned as potentially viable but also notably slow.
   - Participants noted that creating **secure containers** via Docker for models may be a more realistic approach to maintaining security.
- **Multiparty Computation discussed lightly**: A user mentioned hearing about **multiparty computation** strategies for distributing workloads across cloud environments. While it offers some benefits, concerns about security guarantees in such implementations were raised.
   - Participants acknowledged the complexity and the substantial investment required in developing secure methodologies for running computations in **trustless environments**.
- **Challenges in Achieving Full Privacy**: It was noted that achieving **full privacy** in machine learning remains nearly impossible or financially unfeasible at present. The discussion underscored a significant amount of money at stake in the pursuit of effective privacy solutions.
   - Experts are particularly interested due to potential applications in sensitive environments, such as those related to **DARPA**.



**Link mentioned**: <a href="https://lu.ma/wbc5bx0z">GitHub Presents: Open Source AI - Access, Democratization, and Responsibility · Luma</a>: AI is rapidly transforming industries from software development, content creation, agentic workflows and beyond. Central to this transformation is open source…

  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/)** (1 messages): 

chad_in_the_house: wow that's annoying lol
  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1282821719320428598)** (8 messages🔥): 

> - `AI Research Fraud`
> - `Reasoner Dataset`
> - `iChip Technology`
> - `Hugging Face Multi-Packing` 


- **AI Research Community Faces Fraud Allegations**: On September 5th, Matt Shumer, CEO of OthersideAI, announced a supposed breakthrough in training mid-size AI models to top-tier performance, which was later revealed to be *false*.
   - This incident highlights ongoing concerns about *integrity in AI research* and the need for skepticism regarding announcements.
- **Guilherme Shares Reasoner Dataset**: A user shared the [Reasoner Dataset](https://huggingface.co/datasets/Guilherme34/Reasoner-Dataset-FULL), claiming it to be effectively created using *synthetic data*.
   - This dataset is designed for reasoning tasks, showcasing innovative approaches in AI training data development.
- **iChip Technology Revolutionizes Antibiotic Discovery**: iChip technology, which cultures previously unculturable bacteria, has significantly impacted the discovery of new antibiotics like *teixobactin* in 2015.
   - Its ability to grow bacteria in **natural environments** could vastly increase the microbial candidates for future drug discovery efforts.
- **Hugging Face Introduces Multi-Packing for Increased Efficiency**: Hugging Face has announced that training with packed instruction tuning examples is now compatible with **Flash Attention 2**.
   - This feature potentially increases throughput by up to **2x**, streamlining the training process for AI models.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/blog/packing-with-FA2">Improving Hugging Face Training Efficiency Through Packing with Flash Attention 2</a>: no description found</li><li><a href="https://x.com/shinboson/status/1832933747529834747?t=lu0kNqbEZKG5LVC30Dm7hA&s=19">Tweet from 𝞍 Shin Megami Boson 𝞍 (@shinboson)</a>: A story about fraud in the AI research community:  On September 5th, Matt Shumer, CEO of OthersideAI, announces to the world that they&#39;ve made a breakthrough, allowing them to train a mid-size mod...</li><li><a href="https://huggingface.co/datasets/Guilherme34/Reasoner-Dataset-FULL">Guilherme34/Reasoner-Dataset-FULL · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1283073991170457651)** (3 messages): 

> - `OpenAI Fine-Tuning API`
> - `Chat Template Importer Changes`
> - `Weight Parameter for Training Data` 


- **OpenAI Fine-Tuning API gains Weight Parameter**: OpenAI has added a **weight** parameter to their fine-tuning API as outlined in their [documentation](https://platform.openai.com/docs/guides/fine-tuning/multi-turn-chat-examples). This change was implemented in **April**, and the user had missed this addition.
   - With this new parameter, it is expected that **weights** can later be adjusted to values between **0 and 1**, enhancing the control over training data influence.
- **Chat Template Importer needs an update**: The current implementation of the chat template importer uses a column labeled **train** with true/false values, which should be revised to incorporate the **weight** parameter. This adaptation will allow for better alignment with OpenAI's updated API.
   - Ignoring this update resulted in compatibility issues, suggesting that future implementations must closely monitor API changes to maintain consistency.


  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1282894173896441897)** (4 messages): 

> - `BNB Issue Thread`
> - `H100 Performance without 8-bit`
> - `Fine-tuning Mistral NeMo`
> - `Errors with Padding Token in Fine-tuning` 


- **BNB Issue Thread Still Unresolved**: A member created a [BNB issue thread](https://link.to.bnbissue) due to persistent errors, expressing confusion about its unresolved status.
   - *I’m not sure why it’s not been fixed*, reflecting frustration over the ongoing issues.
- **H100 Shows Impressive Speed**: It was noted that the **H100** GPU operates surprisingly fast even without using **8-bit** precision.
   - This indicates strong performance capabilities for **H100**, sparking positive discussions.
- **Seeking Fine-tuning Guidance for Mistral NeMo**: A member inquired about examples for fine-tuning **Mistral NeMo** using **Axolotl**, showcasing the community's desire for practical guidance.
   - This highlights the growing interest in leveraging **Mistral NeMo** with the Axolotl framework.
- **Facing Padding Token Errors in Fine-tuning**: Concerns were raised about encountering errors related to the **padding token** when fine-tuning the **LM head** and **embedding** for models.
   - This indicates potential challenges faced by members during their fine-tuning processes.


  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1282795291346468986)** (4 messages): 

> - `Claude 3.5 audio capability`
> - `Token counting in langchain4j`
> - `Whisper as an alternative for transcription` 


- **Claude 3.5's Audio Features in Question**: A member inquired whether it is possible to pass **audio data** to **Claude's 3.5** LLM via **Langchain** for transcription.
   - Another user was uncertain, mentioning that Claude 3.5 supports images but not clear audio functionalities.
- **Langchain4j Token Counting Challenge**: A different member sought guidance on how to **count tokens** for input and output using **langchain4j**.
   - No specific solutions were discussed in the thread regarding how to achieve this.
- **Whisper Proposed for Audio Transcription**: One member suggested that for audio transcription, **Whisper** is a **faster and cheaper** alternative to using Claude 3.5.
   - This highlights a potential efficiency gain when looking for transcription options in comparison to Claude.


  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1282784799756390401)** (4 messages): 

> - `Chat AI Lite`
> - `EDA-GPT`
> - `Pilerbot` 


- **Chat AI Lite: Multifaceted AI Web Application**: [Chat AI Lite](https://github.com/KevinZhang19870314/chat-ai-lite/blob/main/README_en_US.md) is a **versatile AI web application** covering multiple scenarios including chat, local knowledge bases, and image generation.
   - Its comprehensive capabilities aim to enhance user experience in various **AI applications**.
- **Automated Data Analysis with EDA-GPT**: [EDA-GPT](https://github.com/shaunthecomputerscientist/EDA-GPT) offers **automated data analysis** leveraging large language models (LLMs), showcasing advanced integration for data science tasks.
   - This project encourages contributions to enhance its **data analytical capabilities**.
- **Personal Discord Bot: Pilerbot**: [Pilerbot](https://github.com/shaunthecomputerscientist/pilerbot) is a **personal bot** designed for managing a Discord server centered around the *Piler* community.
   - This bot facilitates server management, aiming to streamline interactions within the **ED community**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/KevinZhang19870314/chat-ai-lite/blob/main/README_en_US.md">chat-ai-lite/README_en_US.md at main · KevinZhang19870314/chat-ai-lite</a>: Chat AI Lite 是一个多功能的 AI Web 应用，涵盖了各种 AI 场景，包括 AI 聊天、AI 本地知识库（RAG）、AI 助手、AI 数字人以及图像生成等。 - KevinZhang19870314/chat-ai-lite</li><li><a href="https://github.com/shaunthecomputerscientist/EDA-GPT">GitHub - shaunthecomputerscientist/EDA-GPT: Automated Data Analysis leveraging llms</a>: Automated Data Analysis leveraging llms. Contribute to shaunthecomputerscientist/EDA-GPT development by creating an account on GitHub.</li><li><a href="https://github.com/shaunthecomputerscientist/pilerbot">GitHub - shaunthecomputerscientist/pilerbot: personal bot for managing a discord server based on piler (my ed community)</a>: personal bot for managing a discord server based on piler (my ed community) - shaunthecomputerscientist/pilerbot
</li>
</ul>

</div>
  

---



### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1282919470196719648)** (8 messages🔥): 

> - `Emotion classification code`
> - `AdalFlow GitHub library`
> - `Llama AI model prompt`
> - `MIPRO prompt optimizer` 


- **Testing Emotion Classifier Code**: A member questioned if changing the description from **'Classify emotion among sadness, joy, love, anger, fear, surprise'** to **'Classify to 7 emotions'** would yield a different output for the Emotion classifier.
   - Clarification on the impact of this change on output was requested but not provided.
- **Exploring AdalFlow AI Library**: A member reupped a discussion on [AdalFlow](https://github.com/SylphAI-Inc/AdalFlow), a PyTorch library designed to auto-optimize LLM tasks, seeking insights from others.
   - Another member planned to take a look later in the week and promised to follow up with their findings.
- **Revealing a Fake Model**: A member revealed that a purported Llama AI model was actually the latest **Claude** model operating under a complex prompting system.
   - The intricate system prompt guided the model on problem-solving and reflection processes for various questions.
- **MIPRO Enhances Prompt Optimization**: MIPRO, a new tool from the DSPy team, allows for the optimization of instructions and examples in prompts, designed for use with datasets.
   - An exploration into *how MIPRO streamlines prompt optimization* for question-answering systems detailed its need for relevant datasets.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/SylphAI">sylphAI - Overview</a>: GitHub is where sylphAI builds software.</li><li><a href="https://github.com/SylphAI-Inc/AdalFlow">GitHub - SylphAI-Inc/AdalFlow: AdalFlow: The “PyTorch” library to auto-optimize any LLM tasks.</a>: AdalFlow: The “PyTorch” library to auto-optimize any LLM tasks. - SylphAI-Inc/AdalFlow</li><li><a href="https://medium.com/gitconnected/building-an-optimized-question-answering-system-with-mipro-and-dspy-9fe325ca33a9">Building an Optimized Question-Answering System with MIPRO and DSPY</a>: Bye Bye Manual Prompting
</li>
</ul>

</div>
  

---



### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1283067550573527151)** (3 messages): 

> - `LLM observability platforms`
> - `Anthropic API performance` 


- **Seeking LLM Observability Recommendations**: A member is exploring options for **LLM observability platforms** suitable for a large internal corporate RAG app, currently considering [W&B Weave](https://wandb.ai/weave) and [dbx's MLflow](https://mlflow.org/).
   - They also mentioned potential interest in **Braintrust** and **Langsmith** for this purpose.
- **Node.js vs Python in Anthropic's API**: A member observed that using **Anthropic's API** with **Node.js** yields worse performance compared to **Python**, particularly when using tools.
   - They asked if anyone else experienced similar issues, indicating a potential performance discrepancy worth discussing.


  

---



### **Gorilla LLM (Berkeley Function Calling) ▷ #[leaderboard](https://discord.com/channels/1111172801899012102/1214705495974092810/1282795799948038266)** (3 messages): 

> - `Merge Conflicts Resolution`
> - `Test Results Storage` 


- **Merge Conflicts Resolved Successfully**: A member thanked another for resolving their **merge conflicts** without further issues.
   - *Much appreciated for the quick fix!*
- **Searching for Specific Test Scores**: A member expressed confusion about finding specific test scores after saving the results.
   - Another member suggested looking in the **score folder**, particularly the file `data.csv`.


  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/)** (1 messages): 

kimchiking7364: 🏄
  

---



### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1282799151092072510)** (1 messages): 

> - `Open Source AI Event`
> - `Panelists from Industry`
> - `Event Registration` 


- **Join the Open Source AI Panel at GitHub**: GitHub is hosting a free [Open Source AI panel](https://lu.ma/wbc5bx0z) on **9/19** in their SF office, featuring discussions on accessibility and responsibility in AI.
   - Panelists include representatives from **Ollama**, **Nous Research**, **Black Forest Labs**, and **Unsloth AI**, promising insights into the democratization of AI technology.
- **Get Approved for the Event**: Registration for the event is subject to host approval, so participants are encouraged to register early to secure their spot.
   - Attendees will gain an understanding of how open source communities are fostering innovation in the AI landscape.



**Link mentioned**: <a href="https://lu.ma/wbc5bx0z">GitHub Presents: Open Source AI - Access, Democratization, and Responsibility · Luma</a>: AI is rapidly transforming industries from software development, content creation, agentic workflows and beyond. Central to this transformation is open source…

  

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
