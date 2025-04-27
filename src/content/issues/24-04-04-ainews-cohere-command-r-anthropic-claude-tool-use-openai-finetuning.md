---
id: c2eed8e3-e60a-4b34-b7b5-4f7d26aa5c66
title: Cohere Command R+, Anthropic Claude Tool Use, OpenAI Finetuning
date: '2024-04-04T22:21:15.996359Z'
original_slug: ainews-cohere-command-r-anthropic-claude-tool-use
description: >-
  **Cohere** launched **Command R+**, a **104B dense model** with **128k context
  length** focusing on **RAG**, **tool-use**, and **multilingual** capabilities
  across **10 key languages**. It supports **Multi-Step Tool use** and offers
  open weights for research. **Anthropic** introduced **tool use in beta** for
  **Claude**, supporting over **250 tools** with new cookbooks for practical
  applications. **OpenAI** enhanced its fine-tuning API with new upgrades and
  case studies from Indeed, SK Telecom, and Harvey, promoting DIY fine-tuning
  and custom model training. **Microsoft** achieved a quantum computing
  breakthrough with an **800x error rate improvement** and the most usable
  qubits to date. **Stability AI** released **Stable Audio 2.0**, improving
  audio generation quality and control. The **Opera browser** added local
  inference support for large language models like **Meta's Llama**, **Google's
  Gemma**, and **Vicuna**. Discussions on Reddit highlighted **Gemini's large
  context window**, analysis of **GPT-3.5-Turbo** model size, and a battle
  simulation between **Claude 3** and **ChatGPT** using local 7B models like
  **Mistral** and **Gemma**.
companies:
  - cohere
  - anthropic
  - openai
  - microsoft
  - stability-ai
  - opera-software
  - meta-ai-fair
  - google-deepmind
  - mistral-ai
models:
  - c4ai-command-r-plus
  - claude-3
  - gpt-3.5-turbo
  - gemini
  - mistral-7b
  - gemma-2
  - claude-3-5
  - llama-3
  - vicuna
topics:
  - tool-use
  - multilingual-models
  - rag
  - fine-tuning
  - quantum-computing
  - audio-generation
  - local-inference
  - context-windows
  - model-size-analysis
  - model-comparison
people: []
---


<!-- buttondown-editor-mode: plaintext -->> AI News for 4/3/2024-4/4/2024. We checked 5 subreddits and [**364** Twitters](https://twitter.com/i/lists/1585430245762441216) and **26** Discords (**385** channels, and **5656** messages) for you. Estimated reading time saved (at 200wpm): **639 minutes**.

Busy day today. 

1. The [at least $500m richer](https://twitter.com/steph_palazzolo/status/1773095998555898305) Cohere launched a fast-follow of [last month's Command R](https://twitter.com/aidangomez/status/1767264315550163024?t=6FDPaNxZcbSsELal6Sv7Ug) with Command R+ ([official blog](https://txt.cohere.com/command-r-plus-microsoft-azure/), [weights](https://huggingface.co/CohereForAI/c4ai-command-r-plus)). It's a 104B dense model with 128k context length focused on RAG, tool-use, and multilingual ("[10 key languages](https://x.com/cohere/status/1775878865631498360)")) usecases. Open weights for research but Aidan says "[just reach out](https://x.com/aidangomez/status/1767264324626559249)" if you want to license it (instead of paying their [$3/$15 per mtok pricing](https://x.com/SelfInfinity/status/1775881659058946416)). It now supports [Multi-Step Tool use](https://x.com/cohere/status/1775878859033858346).
 ![image.png](https://assets.buttondown.email/images/823c645a-957f-47cf-bbb5-bd814cd4114e.png?w=960&fit=max) 
2. The [$2.75B richer](https://www.maginative.com/article/amazon-completes-massive-4-billion-investment-in-ai-startup-anthropic/) Anthropic [launched tool use in beta](https://twitter.com/AnthropicAI/status/1775979802627084713) as previously promised ([official docs](https://docs.anthropic.com/claude/docs/tool-use)). The extensive docs come with a number of notable features, most notably [advertising the ability to handle over 250 tools](https://twitter.com/swyx/status/1775993946935906645) which enables a very different function calling architecture than before. This is presumably due to context length and recall improvements in the past year. For more details see their 3 new cookbooks:

  - [Using a calculator tool with Claude](https://github.com/anthropics/anthropic-cookbook/blob/main/tool_use/calculator_tool.ipynb)
  - [Creating a customer service agent with client-side tools](https://github.com/anthropics/anthropic-cookbook/blob/main/tool_use/customer_service_agent.ipynb)
  - [Extracting structured JSON using Claude and tool use](https://github.com/anthropics/anthropic-cookbook/blob/main/tool_use/extracting_structured_json.ipynb)

3. OpenAI, which hasn't raised anything in the last month (that we know of), added [a bunch of very welcome upgrades](https://openai.com/blog/introducing-improvements-to-the-fine-tuning-api-and-expanding-our-custom-models-program) to the very MVPish finetuning experience together with 3 case studies with Indeed, SK Telecom, and [Harvey](https://openai.com/customer-stories/harvey) that basically say "you can now DIY better but also we are open for business to finetune and train your stuff".

 ![image.png](https://assets.buttondown.email/images/75751b5a-56c6-4d64-a4c8-a64dab0c10b8.png?w=960&fit=max) 

---

**Table of Contents**

[TOC] 


---

# AI Reddit Recap

> Across r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence. Comment crawling still not implemented but coming soon.

**AI Technology Advancements**

- **Quantum Computing Breakthrough**: In /r/singularity, Microsoft has achieved a quantum computing breakthrough, improving error rates by 800x with the most usable qubits to date, [**a significant step forward in quantum computing capabilities**](https://v.redd.it/wp1djpidpbsc1). 
- **Stable Audio 2.0 Release**: In /r/StableDiffusion, Stability AI introduced Stable Audio 2.0, [**advancing audio generation capabilities**](https://stability.ai/news/stable-audio-2-0) with improved quality and control.
- **Browser Integration of Large Language Models**: In /r/LocalLLaMA, Opera browser has added support for [**running large language models like Meta's Llama, Google's Gemma, and Vicuna locally**](https://www.reddit.com/r/LocalLLaMA/comments/1buu5v1/opera_has_added_local_llm_inference_to_their/), making them more accessible.

**Model Capabilities & Comparisons**

- **Gemini's Large Context Window**: In /r/ProgrammerHumor, an image highlights that [**Gemini's context window is much larger than other models**](https://i.redd.it/o2appsoeyasc1.png), enabling more contextual understanding.
- **GPT-3.5-Turbo Model Size Analysis**: In /r/LocalLLaMA, analysis suggests [**GPT-3.5-Turbo is likely an 8x7B model, similar in size to Mixtral-8x7B**](https://www.reddit.com/r/LocalLLaMA/comments/1bv9kag/gpt35turbo_is_most_likely_the_same_size_as/).
- **Claude 3 vs ChatGPT Battle Simulation**: In /r/LocalLLaMA, a video compares Claude 3 vs ChatGPT in a "Street Fighter" style battle [**using local 7B models like Mistral and Gemma**](https://v.redd.it/joikvgj2l9sc1).

**AI Research & Education**

- **Stanford Transformers Course Opens to Public**: In /r/StableDiffusion, Stanford's CS 25 Transformers Course is opening to the public, [**featuring top researchers discussing breakthroughs in architectures, applications, and more**](https://www.reddit.com/r/StableDiffusion/comments/1bve5kp/stanford_cs_25_transformers_course_open_to/).
- **Stock Prediction Research Challenges**: In /r/MachineLearning, a discussion explores [**why stock prediction research papers often don't translate to real-world production use**](https://www.reddit.com/r/MachineLearning/comments/1bv0cu7/why_stock_prediction_papers_arent_put_to/).
- **Retrieval-Augmented Generation Debate**: In /r/MachineLearning, a debate arises on [**whether Retrieval-Augmented Generation (RAG) is just glorified prompt engineering**](https://www.reddit.com/r/MachineLearning/comments/1busp41/d_is_rag_just_glorified_prompt_engineering/).

**AI Tools & Applications**

- **GPT-4-Vision for Online Mimicry**: In /r/singularity, a video demonstrates [**using GPT-4-Vision to mimic oneself in emails or any site with one click**](https://v.redd.it/h1g82xgyi8sc1).
- **Automatic Video Highlight Detection**: In /r/singularity, a tool is showcased for [**finding highlights in long-form video automatically with custom search terms**](https://v.redd.it/halkizhh4asc1).
- **Daz3D AI-Powered Image Generation**: In /r/StableDiffusion, Daz3D partners with Stability AI to launch [**Daz AI Studio for stylized image generation from text**](https://www.reddit.com/r/StableDiffusion/comments/1bvb88n/daz3d_partnering_with_sai/).

**AI Memes & Humor**

- **Gemini Context Window Meme**: In /r/ProgrammerHumor, a [**humorous image depicts "Gemini's context window is much larger than anyone else's"**](https://i.redd.it/o2appsoeyasc1.png).
- **Super Metroid Parody Trailer**: In /r/singularity, a [**parody movie trailer for Super Metroid was created with Dalle3 and GPT**](https://v.redd.it/galfcs7gzasc1).
- **Bedroom QR Code Meme**: In /r/singularity, a [**bedroom QR code meme image was shared**](https://i.redd.it/arudc6zav8sc1.png).

# AI Twitter Recap

> all recaps done by Claude 3 Opus, best of 4 runs. We are working on clustering and flow engineering with Haiku.

**Cohere Command R+ Release**

- **New open-source model**: [@cohere](https://twitter.com/cohere/status/1775878850699808928) released Command R+, a 104B parameter model with 128k context length, open weights for non-commercial use, and strong multilingual and RAG capabilities. It's available on the [Cohere playground](https://twitter.com/cohere/status/1775878883268509801) and [Hugging Face](https://twitter.com/osanseviero/status/1775882744792273209).
- **Optimized for RAG workflows**: Command R+ is [optimized for RAG](https://twitter.com/aidangomez/status/1775878606108979495), with multi-hop capabilities to break down complex questions and strong tool use. It's integrated with [@LangChainAI](https://twitter.com/cohere/status/1775931339361149230) for building RAG applications.
- **Multilingual support**: Command R+ has [strong performance](https://twitter.com/seb_ruder/status/1775882934542533021) across 10 languages including English, French, Spanish, Italian, German, Portuguese, Japanese, Korean, Arabic, and Chinese. [@JayAlammar](https://twitter.com/JayAlammar/status/1775928159784915229) notes that the tokenizer is more efficient for **Arabic and other non-English languages**, requiring fewer tokens and leading to cost savings.
- **Pricing and Availability**: [@cohere](https://twitter.com/cohere/status/1775878850699808928) noted Command R+ leads the scalable market category, **enabling businesses to move to production**. It is available on Microsoft Azure and coming to other cloud providers soon. [@JayAlammar](https://twitter.com/JayAlammar/status/1775881793796726808) added it takes RAG to a new level with **multi-hop capabilities**.
- **LangChain Integration**: [@hwchase17](https://twitter.com/hwchase17/status/1775922998853414961) and [@LangChainAI](https://twitter.com/LangChainAI/status/1775889394916049230) announced a `langchain-cohere` package to expose integrations like **chat models and model-specific agents**. [@cohere](https://twitter.com/cohere/status/1775931339361149230) is excited about the integration for **adaptive RAG**.
- **Hugging Face and Performance**: [@osanseviero](https://twitter.com/osanseviero/status/1775882744792273209) noted it is available on Hugging Face with a playground link. [@seb_ruder](https://twitter.com/seb_ruder/status/1775882934542533021) highlighted the **multilingual capabilities in 10 languages**. [@JayAlammar](https://twitter.com/JayAlammar/status/1775928159784915229) mentioned **tokenizer optimizations for languages like Arabic** to reduce costs.
- **Fine-tuning and Efficiency**: [@awnihannun](https://twitter.com/awnihannun/status/1775942513653924049) showed **fine-tuning Command R+ with QLoRA in MLX on an M2 Ultra**. [@_philschmid](https://twitter.com/_philschmid/status/1775894028707639357) provided a summary of the 104B model with open weights, RAG and tool use, and multilingual support.



**DALL-E 3 Inpainting Release**

- **New Feature**: [@gdb](https://twitter.com/gdb/status/1775780196517548335) and [@model_mechanic](https://twitter.com/model_mechanic/status/1775590691487556064) announced that DALL-E 3 inpainting is now live for all ChatGPT Plus subscribers. This allows users to **edit and modify parts of an image** from text instructions.
- **How to Use**: [@chaseleantj](https://twitter.com/chaseleantj/status/1775493065677169147) provides a guide - brush over the region to replace, type the prompt describing the change, and do not brush over all the words for best results. There are still some **limitations** like inability to generate words in blank spaces.


**Mixture-of-Depths for Efficient Transformers**

- **Approach**: [@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1775743788402479323) shares Google's Mixture-of-Depths approach to **dynamically allocate compute** in transformer models. It enforces a total compute budget by capping tokens in self-attention/MLP at each layer.
- **Benefits**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1775927231463706773) explains this minimizes compute waste by allocating more to harder-to-predict tokens vs. easier ones like punctuation. Compute expenditure is **predictable in total but dynamic and context-sensitive** at the token level.


**RAG and Agent Developments**

- **Adaptive RAG techniques**: New papers like [Adaptive RAG](https://twitter.com/LangChainAI/status/1775917799065653250) and [Corrective-RAG](https://twitter.com/llama_index/status/1775912690529288556) propose dynamically selecting RAG strategies based on query complexity. Implementations are available as [LangChain](https://twitter.com/LangChainAI/status/1775569294241472810) and [LlamaIndex](https://twitter.com/llama_index/status/1775912690529288556) cookbooks.
- **RAG-powered applications**: Examples of RAG-powered apps include [Omnivore](https://twitter.com/jerryjliu0/status/1775691578994278719), an AI-enabled knowledge base, and [Elicit's task decomposition architecture](https://twitter.com/labenz/status/1775894599179157840) for scaling complex reasoning. Connecting RAG with [tool use](https://twitter.com/hwchase17/status/1775922998853414961) leads to more agentic systems.

**Open-Source Models and Frameworks**

- **Anthropic Jailbreaking**: [@AlphaSignalAI](https://twitter.com/AlphaSignalAI/status/1775561052325077218) shared Anthropic's research on "many-shot jailbreaking" which crafts benign dialogues to **bypass LLM safety measures**. It takes advantage of large context windows to generate normally avoided responses.
- [@AssemblyAI](https://twitter.com/AssemblyAI/status/1775527556042629437) introduced **Universal-1, a multilingual speech recognition model** trained on 12.5M hours of data. It outperforms models like Whisper on accuracy and hallucination rate.
- **Open models and datasets**: New open models include [Yi](https://twitter.com/rohanpaul_ai/status/1775924341923860594) from 01.AI, [Eurus](https://twitter.com/rohanpaul_ai/status/1775458159865323810) from Tsinghua, [Jamba](https://twitter.com/maximelabonne/status/1775511912773566733) from AI21 Labs, and [Universal-1](https://twitter.com/AssemblyAI/status/1775527556042629437) from AssemblyAI. Large OCR datasets from [Hugging Face](https://twitter.com/rohanpaul_ai/status/1775506872520355870) enable document AI research.
- **Efficient inference techniques**: [BitMat](https://twitter.com/rohanpaul_ai/status/1775608234180558851) reduces memory usage for quantized models. [Mixture-of-Depths](https://twitter.com/arankomatsuzaki/status/1775743788402479323) dynamically allocates compute in Transformers. [HippoAttention](https://twitter.com/rohanpaul_ai/status/1775923372242726995) and [MoE optimizations](https://twitter.com/rohanpaul_ai/status/1775944589230170350) speed up inference.
- **Accessible model deployment**: [Hugging Face](https://twitter.com/_philschmid/status/1775885996435087449) lowered prices for hosted inference, while [Koyeb](https://twitter.com/llama_index/status/1775688909042954723) and [SkyPilot](https://twitter.com/skypilot_org/status/1775931821257314745) simplify deploying models on any cloud platform.

**Memes and Humor**

- An AI-generated video of a [sad girl singing the MIT License](https://twitter.com/goodside/status/1775713487529922702) went viral.
- People speculated about [Apple's AI ambitions](https://twitter.com/Teknium1/status/1775748185203634388) and joked that [AI will replace software engineers](https://twitter.com/bindureddy/status/1775538983688450480).
- There were memes poking fun at [AI hype](https://twitter.com/bindureddy/status/1775920627603657142) and [the limitations of large language models](https://twitter.com/fchollet/status/1775636345190588689).

---

# AI Discord Recap

> A summary of Summaries of Summaries

1. **LLM Advancements and Integrations**:
   - [Cohere unveils Command R+](https://txt.cohere.com/command-r-plus-microsoft-azure/), a **104B parameter multilingual LLM** optimized for enterprise use with advanced Retrieval Augmented Generation (RAG) and multi-step tool capabilities, sparking interest in its performance compared to other models. 
   - [JetMoE-8B](https://research.myshell.ai/jetmoe) represents an affordable milestone at under **$0.1 million cost**, surpassing Meta AI's LLaMA2 performance using only **2.2B active parameters**.
   - Discussions around integrating **LLMs like HQQ with gpt-fast**, exploring **4/3 bit quantization** approaches like the [Mixtral-8x7B-Instruct quantized model](https://huggingface.co/mobiuslabsgmbh/Mixtral-8x7B-Instruct-v0.1-hf-attn-4bit-moe-3bit-metaoffload-HQQ).

2. **Optimizing LLM Inference and Training**:  
   - [Mixture-of-Depths (MoD)](https://arxiv.org/abs/2404.02258) enables transformers to **dynamically allocate compute across sequences**, potentially improving efficiency over uniform distribution.
   - [Visual AutoRegressive (VAR) modeling](https://arxiv.org/abs/2404.02905) redefines autoregressive image generation, outperforming diffusion transformers in quality and speed.
   - Techniques like **BitMat** offer [efficient 1-bit LLM implementations](https://github.com/astramind-ai/BitMat) per "The Era of 1-bit LLMs" paper.

3. **LLM Evaluation and Benchmarking**:
   - New benchmarks evaluate LLM emotional intelligence: [Creative Writing EQ-Bench](https://eqbench.com/creative_writing.html) and [Judgemark](https://eqbench.com/judgemark.html) using correlation metrics.
   - **COMET scores** highlight the [Facebook WMT21 model's translation prowess](https://github.com/CrispStrobe/llm_translation), with the highest score of **0.848375**.
   - Discussions on **systematic evaluation practices** for AI products, with [Hamel Husain's post](https://hamel.dev/blog/posts/evals/) seen as groundbreaking.

4. **Open-Source AI Frameworks and Tools**:
   - [LlamaIndex](https://www.llamaindex.ai/) unveils **cookbooks guiding RAG system building** with MistralAI, including routing and query decomposition.
   - [Koyeb](https://t.co/weFs0waN4o) enables **effortless global scaling of LLM apps** by connecting GitHub repos to deploy serverless apps.
   - [SaladCloud](https://bit.ly/3TFIsKt) offers a **managed container service for AI/ML workloads** to avoid high cloud costs.
   - The [transformer-heads GitHub repo](https://github.com/center-for-humans-and-machines/transformer-heads) provides tools for **extending LLM capabilities** by attaching new model heads.

---



# PART 1: High level Discord summaries




## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **High-Res with Caution**: Practitioners discussed optimal settings for **Stable Diffusion Models when upscaling**, advocating for 35 steps with specific upscalers and control nets to mitigate image distortion. Higher resolutions, particularly 2k, lead to longer generation times and potential issues, as outlined in [Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction](https://arxiv.org/abs/2404.02905).

- **AI's Role in the Creative Process**: There was a spirited discussion on AI's burgeoning effects on creative industries, specifically pondering its potential to supersede some roles in Hollywood and game development. The group contemplated how AI tools like SDXL could alter job landscapes, possibly raising the entry-level bar for producing quality output.

- **Techniques for Targeted Lora Training**: To train loras for generating images of specific attire, such as corsets, suggestions were to use diverse angle shots of the item isolated from extraneous details. The aim is to help the AI focus on the core element, thus avoiding introducing unwanted features in the outputs.

- **Costs, Investors, and AI Market Dynamics**: The guild tackled Stability AI's strategic hurdles—balancing between attracting investments, enriching datasets, and developing fresh models. Dialogue revolved around innovations in dataset monetization strategies for businesses in the face of rising computational costs and fluctuating model research interest.

- **Random Banter Is Still Alive**: Amid technical talk, members exchanged casual banter, including cultural references and greetings. An off-topic link to a parody song was shared, showcasing the community's lighter side alongside their technical engagements.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

**API Excitement Meets Payment Puzzles**: A perplexing payment issue cropped up for a **Perplexity API** user whose transaction was stuck as "Pending" without updating the account balance. Meanwhile, discussions revolved around the potential of APIs and the choice between a Pro Subscription and pay-as-you-go API, with opinions favoring the subscription for initial business ideation due to cost predictability.

**Model Mashing Madness**: Users dived into **model preferences**, favoring a balance between a larger message count and an adequate context window. They also tackled the challenge of model limitations with complex programming languages like Rust and custom "metaprompt" strategies for structured output.

**Content Sharing Caveat**: A note was made to ensure **threads** are set to shareable when posting content on Discord, facilitating wider community engagement.

**Thirst for Source Links in Sonar Model**: Inquiries were made concerning the **sonar-medium-online** model's ability to return source links with data, but a definitive timeline on the feature's implementation remains elusive.

**LLM Leaderboard Quirks and Queries**: The **LLM Leaderboard** sparked an analytical discourse on model rankings with a dash of humor over model name mishaps, pointing to the significance of clarity in system prompts for better AI performance.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **DALL·E Dons a New Creative Cap**: **DALL·E** now boats an *Editing Suite* for image edits and style inspiration on web, iOS, and Android platforms, offering enhancements to the creative potential across **ChatGPT** platforms. In tandem, the **Fine-tuning API** sees an infusion of new dashboards, metrics, and integrations for developers to forge custom models, detailed in a [recent blog post](https://openai.com/blog/introducing-improvements-to-the-fine-tuning-api-and-expanding-our-custom-models-program).

- **AI's Existential Ruminations**: Engineers engaged in a blaze of conversations around AI, untangling the question of AI "thinking" with a broad consensus refuting AI consciousness in lieu of complex data pattern executions. The palpable discord in live discussions also touched on correcting the AI vs AGI misconception prevalent amongst the public and ended with a proposition to train LLMs on goal-oriented sequences.

- **Customization or Complexity**: Within the **GPT-4 discussions**, engineers wrangled over benefits of *Custom GPTs*, the utility of **DALL·E's** new features for image specificity, and questions on **data retention policies** surfaced—ensuring even deleted chats linger for a month.

- **Prelude to Prompt Perfection**: Technicians noodled over issues in translating markdown into various languages and recommended using additional context to refine AI's interpretation during *AI role play*. Strategies for propelling text generation and ensuring document completeness when using LLMs were also broached, suggesting methods such as "continue" to extend responses.

- **Patience for Prompt Precision**: As members grappled with **translation issues with markup** and advice on constructing effective prompts, they were directed to the refashioned **[#1019652163640762428](https://discord.com/channels/channel_id/1019652163640762428)** for resources. Insights on the efficacy of prompts, particularly in role-playing scenarios, also peeked through, emphasizing the importance of providing clear context to shape AI responses.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

**CmdR Set to Join the Unsloth Ranks**: The addition of **CmdR** support to Unsloth AI is in progress, with the community eagerly awaiting its integration post current task completions. The anticipation ties into plans for an open-source **CPU optimizer**, slated for reveal on April 22, to enhance AI model accessibility for those with limited GPU resources.

**Interfacing Innovation with Continue's Autocomplete**: A new **tab autocomplete** feature is in experimental pre-release for the [Continue](https://marketplace.visualstudio.com/items?itemName=Continue.continue) extension, designed to streamline coding in VS Code and JetBrains by consulting language models directly within the dev environment.

**Error Extermination and Optimization Dialogues**: AI engineers shared solutions to naming-related tokenizer errors, and discussed **model.save_pretrained_merged** and **model.push_to_hub_merged** functions for seamless model saving and sharing on Huggingface. Despite encountering `AttributeError` in GemmaForCausalLM, users were directed to update Unsloth for resolution.

**Stumbling Blocks in Saving and Server-Side Setup**: Users navigated challenges with GGUF conversions and Docker setups, tackling issues like `python3.10-dev` dependencies and workaround strategies for memory errors during finetuning on different platforms.

**Diving into Unsloth Studio's Next Iteration Soon**: An update on Unsloth Studio's release push is set for mid next month due to current bug fixes, ensuring ongoing compatibility with **Google Colab** alongside improvements for developers leveraging the Studio's capabilities.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

**Stable Audio Hits a New High**: Stability AI launched **Stable Audio 2.0**, enabling creation of lengthy high-quality music tracks utilizing a single prompt. Visit [stableaudio.com](http://stableaudio.com) to test the model and find further details in their [blog post](https://bit.ly/3xlwmP5).

**AssemblyAI Outperforms Whisper**: **AssemblyAI** announced **Universal-1**, a speech recognition model surpassing Whisper-3 by achieving 13.5% better accuracy and demonstrating up to 30% decrease in hallucinations. The model processes an hour of audio in a mere 38 seconds and is available for trial at [AssemblyAI's playground](https://www.assemblyai.com/playground).

**Enhance Your Images with ChatGPT Plus**: Users of ChatGPT Plus now possess the ability to modify DALL-E-generated images and prompts, available on both web and iOS platforms. Full guidance on usage is provided in their [help article](https://help.openai.com/en/articles/9055440-editing-your-images-with-dall-e).

**AI Agents as Scalable Microservices**: Discussions focused on utilizing event-driven architecture to build scalable AI agents, with the Actor Model cited as an inspiration, and a Golang framework presented for collaborative feedback.

**Opera One Downloads AI Directly**: Opera integrates the ability for users to run large language models (LLMs) locally, beginning with Opera One on the developer stream, harnessing the Ollama framework, as detailed by [TechCrunch](https://techcrunch.com/2024/04/03/opera-will-now-allow-users-download-and-use-llms-locally).

**DSPy Steals the Spotlight**: Members evaluated **DSPy**'s performance in optimizing prompts for foundation models, focusing on model migration and optimization while being cautious of API rate limits. A detailed study of **Devin** surfaced numerous AI project opportunities, with keen interest in diverse applications ranging from voice-integrated iOS apps to documentation overhaul initiatives.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **LoRA Boosts Mistral?**: Engineers discuss employing **Low-Rank Adaptation (LoRA)** on **Mistral 7B** to enhance specific task performance, with plans to innovate sentence splitting and labeling techniques beyond standard methods.
- **Web Crawling Woes and Wins**: The practical issues of **scalable web crawling** were a hot topic, with talk of obstacles like anti-bot measures and JavaScript rendering. However, alignment was reached on the utility of **Common Crawl** and mysterious archival groups hoarding quality datasets.
- **Learning Options Expand**: Shared resources included guides on **lollms with Ollama Server**, budget AI chips from **Intellifusion**, and Hugging Face's dataset loaders utility, **Chug**. Meanwhile, CohereForAI's new **multilingual 104B LLM** has stirred interest, and OpenAI's exploratory **GPT-4 fine-tuning pricing** was editorialized.
- **LLM Innovation at the Fore**: Engineers exchange insights on **language model pruning**, specifically a 25% pruned **Jamba model**, and Google's paper advocating transformers learn to dynamically allocate compute, sparking a deeper analysis of speculative decoding versus Google's method.
- **Diverse Fine-Tuning Conversations**: Members introduced **Eurus-7b-kto** optimized for reasoning, debated the "_divide by scale_" in **BitNet-1.58** for ternary encoding, deliberated implementation issues on Hermes-Function-Calling, considered **QLoRA's** VRAM efficiency, and noted **Genstruct 7B's** instructional generation prowess.
- **Troubleshooting in Project Obsidian**: Quick fixes in progress for **ChatML** in project "llava" and intentions to tackle **Hermes-Vision-Alpha** with scant details on specific issues.
- **Finetuning Subnet Miner Mishaps**: A miner script error in the **finetuning-subnet** repository points to a possible **missing dependencies** problem.
- **RAG Dataset Discussions**: Discourse on **Glaive's** RAG sample dataset and methods like grounded mode and proper citation markup, including an XML instruction format, emphasized for future uptake. Suggestions on filtering in RAG responses and Cohere's RAG documentation were also highlighted.
- **Copying Conundrums & Command Quests in WorldSim**: WorldSim's perplexing copy-paste mechanics, concern over mobile performances, and links to a comprehensive [WorldSim Command Index](https://worldsim.notion.site/WorldSim-Command-Index-961c8849f61e4558949716b1dfd5f9fa?pvs=4) brought forth both productivity hacks and culture snippets within the intrigue of jailbreaking Claude models and ASCII art enigmas.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

**Mojo on the Move**: Engineers shared that [Mojo](https://github.com/) now runs on Android devices like Snapdragon 685 CPUs and discussed integrating [Mojo with ROS 2](https://github.com/ros2-rust/ros2_rust), accentuating Mojo's memory safety over Python, particularly in robotics where Python’s GIL limits [Nvidia Jetson hardware performance](https://developer.nvidia.com/embedded/jetson-modules).

**Performance Breakthroughs and Best Practices**: Significant library performance improvements were noted, dropping execution times to minutes, beating previous Golang benchmarks. Methods such as pre-setting dictionary capacities for optimization were advised, and designers of specialized sorting algorithms for strings are encouraged to align with Mojo’s latest versions, seen at [mzaks/mojo-sort](https://github.com/mzaks/mojo-sort/blob/main/multi_key_quicksort/sort.mojo).

**From Parser to FASTQ**: `BlazeSeq🔥`, a new feature-complete FASTQ parser, has been introduced, providing a CLI-compatible parser that conforms to BioJava and Biopython benchmarks. Enhanced file handling is promised by the buffered line iterator they implemented, indicating a move to a robust future standard for file interactions, showcased on [GitHub](https://github.com/MoSafi2/BlazeSeq).

**Mojo Merger Madness**: Innovative ideas on model merging and conditional conformance in Mojo used **@conditional** annotations for optional trait implementations, while merchandise ideas like Mojo-themed plushies stirred community excitement. Memory management optimizations were considered, examining potential changes to how `Optional` returns values in the nightly version of Mojo's standard library.

**Modular Updates Galore**: [Max⚡ and Mojo🔥 24.2 release](https://modul.ar/discord) brings open-sourced standard libraries and nightly builds with community contribution. Docker build issues in version 24.3 are addressed, while continued development discussions recommend conditional conformance and error handling strategies for future roadmap considerations.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

**Bold Boosts with ROCm**: AMD hardware sees a massive increase in speed from 13 to 65 tokens/second when engineered with the **ROCm preview**, highlighting the significant potential of the right software interface for AMD GPUs.

**Mixtral, Not a Mistral Mistake**: **Mixtral's** distinct identity as a MOE model, combining the strength of eight 7B models into a 56B powerhouse, reflects a strategic approach unlike the standard **Mistral** 7B. Meanwhile, running a Mixtral 8x7b on a 24GB VRAM NVIDIA 3090 GPU may hit speed snafus, yet it’s a viable venture.

**LM Studio 0.2.19 Courts Embeddings**: The fresh-out-of-the-lab **LM Studio version 0.2.19 Preview 1** now supports **local embedding models**, opening up new possibilities for AI practitioners. Despite lacking **ROCm support** in its current preview, Windows, Linux, and Mac users can grab their respective builds from the provided links.

**Engineers Tackle Odd Model Behavior**: Discourse on an AI model dishing out bizarre, task-unrelated responses uncovers potential mishaps in the model's training, signaling a programming predicament in need of debugging prowess.

**CrewAI Collision with JSONDecodeError**: Encountering a **JSONDecodeError** using CrewAI suggests a potential misstep in JSON formatting, a puzzle piece that AI engineers must properly place to avoid jeopardizing data parsing processes.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

**Transformers Takeover at Stanford**: The Stanford CS25 seminar on Transformers is open to the public for live audits and recorded sessions, with industry experts leading the discussions on LLM architectures and applications. Interested individuals can participate via [Zoom](https://stanford.zoom.us/j/99922151759?pwd=dW5CcUtVYkNybGZGY0hMWUZtVkZBZz09), access the [course website](https://web.stanford.edu/class/cs25/), or watch recordings on [YouTube](https://www.youtube.com/playlist?list=PLoROMvodv4rNiJRchCzutFw5ItR_Z27CM).

**Skeptical About Efficiency Claims**: The community voiced skepticism about the Free-pipeline Fast Inner Product (FFIP) algorithm's performance claims, noted in a [journal publication](https://arxiv.org/abs/2311.12224), which promises efficiency by halving multiplications in AI hardware architectures.

**CUDA Conundrums and Code Conflicts**: A member troubleshooting a **RuntimeError with CUDA** identified `apex` as the issue when using the LM eval harness on H100 GPUs, recommending upgrades to **CUDA 11.8** and other adjustments for stability.

**Next-Gen AI Training Techniques Touted**: An [arXiv paper](https://arxiv.org/abs/2404.02258) introduces dynamic FLOP allocation in transformers, potentially optimizing performance by diverging from uniform distribution. Additionally, cloud services like **AWS** and **Azure** support advanced training schemes, with AWS's **Gemini** mentioned explicitly.

**Elastic and Fault-Tolerant How-To**: Details on establishing fault-tolerant and elastic job launches with PyTorch were shared, with documentation available at the [PyTorch elastic training quickstart guide](https://pytorch.org/docs/stable/elastic/quickstart.html).



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **AI Ethics in Code**: A tool called [ConstitutionalAiTuning](https://github.com/steffen74/ConstitutionalAiTuning) allows fine-tuning language models to reflect ethical principles, utilizing a JSON file for principles input and aiming to make ethical AI more accessible.
- **Type Wrestling in JAX**: JAX's type promotion semantics show different outcomes based on operation order, as demonstrated with numpy and Jupyter array types—adding `np.int16(1)` and `jnp.int16(2)` to `3` produces `int16` or `int32` based on the sequence of operations.
- **Model Training Quandaries**: A discussion examined optimal text input configurations for models, debating the merits of sequence concatenation, T5 token extension, and fine-tuning techniques in the realm of SD3 models.
- **Legal Beats and AI**: Using copyrighted material to train AIs, such as with the Suno music AI platform, has sparked concerns about ensuing legal risks and potential suits from content owners.
- **Financial Turbulence for AI Innovator**: Stability AI faces financial headwinds, grappling with significant cloud service expenses that reportedly might eclipse their revenue capabilities, as detailed in a [Forbes article](https://www.forbes.com/sites/kenrickcai/2024/03/29/how-stability-ais-founder-tanked-his-billion-dollar-startup/?sh=2e53d2e3e630).

In the **research** domain:

- **Size Doesn't Always Matter for LDMs**: A study revealed in an [arXiv paper](https://arxiv.org/abs/2404.01367) that larger latent diffusion models (LDMs) do not always outdo smaller ones when the inference budget remains constant.
- **New Optimizer on the Horizon**: A [Twitter tease](https://twitter.com/aaron_defazio/status/1775521495298588956) suggested that the AI community should keep their eyes peeled for a novel optimizer.
- **VAR Model Revolutionizing Image Generation**: The newly presented Visual AutoRegressive (VAR) model demonstrates superior efficacy in image generation compared to diffusion transformers, boasting improvements in both quality and speed, according to an [arXiv paper](https://arxiv.org/abs/2404.02905).



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

**Patch Perfect**: A noteworthy **GitHub bug** was swiftly eradicated in the OpenAccess AI Collective's **axolotl** repository, with the commit history accessible via [GitHub Commit 5760099](https://github.com/OpenAccess-AI-Collective/axolotl/commit/5760099bd4605e4c4fb444890bd473cb200c5f1a). Meanwhile, a **README Table of Contents mismatch** was flagged, prompting a cleanup.

**Datasets and Model Dialogues**: Queries about **optimal datasets** for training Mistral 7B models led to a recommendation for the **OpenOrca dataset**, while debates on **fine-tuning practices** leaned towards the strategy of prioritizing 'completion' before 'instructions'. Discussions spotlighted the potency of **simple fine-tuning (SFT)** over continual pre-training (CPT) when armed with high-quality instructional samples.

**Bot-tled Service**: The **Axolotl help bot** hit a snag, going offline and sparking a wave of mirthful member reactions, yet specifics behind the incident weren't disclosed. The bot was previously offering guidance on the integration of **Qwen2** with **Qlora** and addressing challenges related to **dataset streaming** and **multi-node fine-tuning** within Docker environments.

**AI Dialogues**: The Collective's **general channel** buzzed with tech talk—from **rapid model feedback services** like [Chaiverse](https://console.chaiverse.com/) to the novel resources for adding heads to Transformer models found in the [GitHub repository for transformer-heads](https://github.com/center-for-humans-and-machines/transformer-heads). **CohereForAI** unveiled a behemoth 104 billion parameter **C4AI Command R+** model with specialized capabilities [revealed on Hugging Face](https://huggingface.co/CohereForAI/c4ai-command-r-plus), stirring conversations about the financial implications of running massive models.

**Infrastructure Innovations**: SaladCloud's recent launch of a fully-managed container service for AI/ML workloads was recognized as a notable entrance, giving developers an edge against sky-high cloud costs and GPU shortages with affordable rates for inference at scale.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

**AI Spellcheck Gets Real**: A Node.js code shared by a member for correcting spelling mistakes using the **LlamaIndex** `Ollama` package showed an AI model named ‘mistral’ fixing user errors, like "bkie" to "bike," which can run locally without third-party services over `localhost:11434`.

**Llama's Culinary Code-Loaded Cookbook**: A new culinary-themed guidebook series is unveiled for AI enthusiasts, demonstrating how to build RAG, agentic RAG, and agent-based systems with **MistralAI**, including routing and query decomposition. Grab your AI recipes [here](https://t.co/7KCqujf9sd).

**Exploration and Confusion in LlamaIndex**: Discussions in the community raised concerns about issues from lacking **knowledgegraphs** pipeline support to unclear **graphindex** and `graphdb` integrations, and several members struggled with querying **OpenSearch** and implementing ReAct agents in **llama_index**.

**AI Discussion Evolves Beyond Text**: Engaging talks emerged about the potential of enhancing image processing with Reading and Asking Generative (RAG) techniques, discussing applications ranging from CAPTCHA solutions to ensuring continuity in visual narratives like comics.

**Scaling AI Deployment Made Convenient**: Koyeb's platform was highlighted for effortlessly scaling LLM applications, directly connecting your GitHub repo to deploy serverless apps globally without managing infrastructure. Check out the service [here](https://t.co/weFs0waN4o).



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

**Bold Repo Visibility Choices**: HuggingFace has introduced settings for default repository visibility with options for **public**, **private**, or **private-by-default** for enterprises. The functionality is described in [this tweet by Julien Chaumond](https://twitter.com/julien_c/status/1772688542289822073).

**Custom Quarto Publishing**: HuggingFace now supports publishing with **Quarto**, as detailed in a [tweet by Gordon Shotwell](https://twitter.com/gshotwell/status/1772661727856914720), with more information available on [LinkedIn](https://www.linkedin.com/feed/update/urn:li:activity:7178422723503673345?updateEntityUrn=urn%3Ali%3Afs_feedUpdate%3A%28V2%2Curn%3Ali%3Aactivity%3A7178422723503673345%29).

**Summarization Struggles and Strategies**: Users across channels discussed summarization challenges with GPT-2 and Hugging Face's pipeline, including ineffective length penalties and the search for prompt crafting that maximizes efficiency and result quality, even in CPU-only environments.

**Innovations and Interactions in AI Circles**: Excitement was shared for projects including **Octopus 2**, a model capable of function calls, and advancements in image processing with the new **multi-subject image node pack** from Salt. The community also highlighted academic discussions and resources, such as the potential of **RAG for interviews** and latency-reasoning trade-offs in production prompts, shared in [Siddish's tweet](https://x.com/siddish_/status/1772345589511901368?s=20).

**Diffusion Model Dialogue Deliberates Depth**: AI engineers explored creative implementations for diffusion models, discussing **DiT with cross-attention** for various data conditions, and considering **Stable Diffusion** modifications for tasks like stereo to depth map conversion, referring to the [DiT paper](https://arxiv.org/html/2312.04557v1) and resources like [Dino v2 GitHub](https://github.com/facebookresearch/dinov2) and [SD-Forge-LayerDiffuse GitHub](https://github.com/layerdiffusion/sd-forge-layerdiffuse).



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

**Fishing for Compliments or Functionality?**: Discord's switch from the whimsical fish logo to a more polished design sparked debate among members, leading to talks to potentially match the banner to the new aesthetic. The logo changes by George Hotz seem to have left some nostalgic for the old one.

**Sharding Optimizations In-Depth**: George Hotz and community members explored **optimization techniques and cross-GPU communications**, facing challenges with launch latencies and data transfers. They examined the use of cudagraphs, peer-to-peer limitations, and the role of NV drivers.

**Tinygrad Performance Milestone**: Sharing performance benchmarks, it was revealed that Tinygrad achieved **53.4 tokens per second on a single 4090 GPU**, marking 83% efficiency compared to gpt-fast. George Hotz indicated goals to further boost Tinygrad's performance.

**Intel Hardware On The Horizon**: Discussions on **Intel GPU and NPU kernel drivers** scrutinized various available drivers like 'gpu/drm/i915' and 'gpu/drm/xe', with anticipation for the performance and power efficiency that NPUs may bring when paired with CPUs.

**Helpful Neural Net Education Hustle**: The community found the **Tinygrad tutorials** to be a valuable starting point for neural network newbies and also recommended the JAX Autodidax tutorial, complete with a [hands-on Colab notebook](https://colab.research.google.com/github/google/jax/blob/main/docs/autodidax.ipynb). Interest surged in adapting ColabFold or OmegaFold for Tinygrad, while also learning about PyTorch weight transfer methods.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter Adopts JSON Object Support**: Models like **OpenAI** and **Fireworks** have been confirmed to support the 'json_object' response format, which can be verified via provider parameters on the [OpenRouter models page](https://openrouter.ai/models).

- **Finding The Right Verse with Claude 3 Haiku**: While the **Claude 3 Haiku** model exhibits a mixed performance in roleplay, it's suggested that providing multiple examples might yield better results. However, using **jailbreak (jb) tweaks** is advisable for a significant improvement in output.

- **Niche Servers for Claude's Jailbreaks**: Users on the look for **Claude model** jailbreaks including NSFW prompts discussed resources, pointing out that SillyTavern's and Chub's Discord servers are go-to places, and provided guidance on how to navigate to these using tools like the pancatstack jb.

- **Dashboard Update Maps Out OpenRouter Credits**: Recent updates to the **OpenRouter's dashboard** include a new designated location for credit display which is accessible at the `/credits` endpoint. However, issues with specific models’ functionality, such as **DBRX** and **Midnight Rose**, prompted concerns about their support compatibility.

- **Moderation Tangle Affects OpenRouter API's Decline Rate**: Reports highlighted a high decline rate with the self-moderated version of the **Claude model**, implicating possible overprotective "safety" prompts. There's also a mention of integrating better providers to aid in the stability of services for models like **Midnight Rose**.




---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Installation Celebration and Cross-Platform Clarity**: An engineer was relieved to get a piece of **software** running on their Windows machine, and there was confirmation that this software is functional on both PC and Mac platforms. Detailed installation instructions and guides can be found in the project's documentation.

- **Persistent Termux Predicament**: Discussions identified a recurring issue with `chroma-hnswlib` during installation processes, even though reports suggested it was removed. Members were advised to migrate detailed technical support queries to a designated support channel.

- **Hermes-2-Pro Prompt Practices Discussed**: Active dialogues emphasized the need to adjust system prompts as recommended in the **Hermes-2-Pro** model card. This is crucial for optimizing model performance and addressing verbose output that some users found burdensome.

- **Platform-Specific Quirks**: Multiple members encountered and shared solutions to challenges with the 01 software across different operating systems—ranging from shortcut commands in Ollama, package dependencies in Linux, to `poetry` issues on Windows 11.

- **Cardputer Development Underway**: Technical talk focused on the implementation and advancement of **M5 Cardputer** into the open-interpreter project. GitHub repositories and various tools like ngrok for secure tunnelling and rhasspy/piper for neural TTS systems were linked for reference.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Command R+ Makes Waves with 128k Token Context**: A new scalable LLM dubbed **Command R+** is generating buzz with a hefty 128k token context window and the promise of reduced hallucinations due to refined RAG. Although there's curiosity about its performance compared to other models due to insufficient comparative data, enthusiasts can test out its capabilities via a [live demo](https://huggingface.co/spaces/CohereForAI/c4ai-command-r-plus).

- **ChatGPT-Like Models for Business Under Scrutiny**: Skepticism arises regarding how well ChatGPT and similar models can fulfill enterprise needs, with discussions pointing toward potentially custom-developed solutions to truly meet business demands.

- **Academia Cheers for Cost-Effective JetMoE-8B**: The launch of **JetMoE-8B** is applauded in academic circles for its affordability—costing under $0.1 million—and impressive performance using only 2.2B active parameters. More details can be found on its [project page](https://research.myshell.ai/jetmoe).

- **Snorkel and Model Efficacy Debate Heats Up**: Nathan Lambert stirs the pot with a suggestive [tweet](https://twitter.com/natolambert/status/1775899591814300024), teasing an analysis on the effectiveness of current AI models like those using RLHF, thereby igniting a conversation around the controversial **Snorkel** framework.

- **Stanford's CS25 Pulls in Transformer Enthusiasts**: AI engineers show keen interest in Stanford's CS25 course, spotlighting discussions by Transformer research experts, with session schedules available [here](https://web.stanford.edu/class/cs25/#schedule) and the opportunity to gain insights through the course's YouTube channel.



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Matrix Size Matters**: A member made headway by optimizing a **matmul kernel** for large matrices, addressing CPU cache challenges when dealing with sizes above 1024x1024.
- **Compiler Conundrum Conquered**: Compiler enhancements led to celebrations among members, reflecting expectations of significant code performance improvements.
- **A ROCm Solid Requirement**: For the successful deployment of **llamafile-0.7** on Windows, members acknowledged that **ROCm version 5.7+** is necessary.
- **Dynamic SYCL Discussions**: Debates on handling SYCL code within **llamafile** resulted in a community-driven solution involving conditional compilation, though with noted incompatibility with MacOS.
- **Perplexing Performance on Windows**: An attempt to build **llamafile** on Windows met with complications involving the Cosmopolitan compiler, along with conversations about the need for a `llamafile-bench` program to measure tokens per second and the potential impact of RAM on performance. Interested parties were directed to an article on [The Register](https://www.theregister.com/2024/04/03/llamafile_performance_gains/) highlighting performance gains and a discussion on [GitHub about Cosmopolitan](https://github.com/jart/cosmopolitan/issues/1010).



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

**Crypto Chatbot Craze Calls for Coders**: An individual is in search of developers with LLM training expertise to create a chatbot simulating human conversation, utilizing real-time crypto market data. The aim is to enable nuanced discussions reflecting the latest market shifts.

**Math Symbol Extraction Without MathpixPDFLoader**: Alternatives to MathpixPDFLoader for extracting math symbols from PDFs are in demand, as users seek new methods to handle this specific task effectively.

**LangChain LCEL Logic Lessons**: A discussion clarified the use of the '|' operator in LangChain's Expression Language (LCEL), which chains components like prompts and LLM outputs into complex sequences. The intricacies are further explored in [Getting Started with LCEL](https://python.langchain.com/docs/expression_language/get_started).

**Voice Apps Vocalizing AI Capabilities**: Newly launched voice applications such as [CallStar](https://callstar.ai) are prompting discussions around their interactivity and setup, powered by technologies like RetellAI, with community support via Product Hunt and Reddit platforms.

**LangChain Quickstart Walkthrough Woes**: Sharing the [LangChain Quickstart Guide](https://python.langchain.com/docs/get_started/quickstart), a user provided example code for integrating LangChain with OpenAI, yet faced a `NotFoundError` indicating a missing resource. The community's technical acumen is requested to troubleshoot this setback.



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **Bit by Bit, Efficiency Unfolds**: The [BitMat](https://github.com/astramind-ai/BitMat) GitHub repository was referenced, promoting an **efficient implementation** of 1-bit Large Language Models (LLMs), aligning with the method proposed in "The Era of 1-bit LLMs."
- **New Horizons for Triton and Torch**: A new channel for contributing to the **Triton visualizer** has been proposed to foster collaboration. The Torch team is adjusting autotune settings, moving towards **max-autotuning**, and addressing benchmarking pain points including tensor core utilization and timing methods—their effort is documented in the [keras-benchmarks](https://github.com/haifeng-jin/keras-benchmarks/blob/main/benchmark/torch_utils.py#L17).
- **CUDA Content and Courses**: For engineers keen on learning CUDA programming, the [CUDA MODE YouTube channel](https://www.youtube.com/@CUDAMODE) was recommended, boasting of lectures and a supportive community to ease the CUDA learning curve.
- **Quantum Leap in Model Integrations**: New members mobicham and zhxchen17 ignited a discussion on integrating **HQQ** with **gpt-fast**, focusing on **Llama2-7B (base)**, and delving into 4/3 bit quantization using models like [Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/mobiuslabsgmbh/Mixtral-8x7B-Instruct-v0.1-hf-attn-4bit-moe-3bit-metaoffload-HQQ).
- **A Visual Boost for Triton**: Within the discussion on **Triton visualizations**, suggestions for adding arrows for direction, integrating operation details into visuals, and potentially porting the project to JavaScript for enhanced interactivity emerged, though concerns about the actual utility of such features were raised.



---



## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

**A New Approach to AI Dialogues**: Reflecting on conversational AI terminology, a guild member suggested "turns" as a better descriptor than "responses" for the initial message in a dialogue, a decision fueled by the exploration of a `logs.db` database and resulting in the serendipitous pun with the term [database 'turns'](https://discord.com/channels/823971286308356157/1128504153841336370/1225067602288574554).

**AI Product Evaluations Get a Thumbs Up**: Guild members rallied around the importance of [Hamel Husain's post on AI evaluations](https://hamel.dev/blog/posts/evals/), which outlines strategies for creating domain-specific evaluation systems for AI and is considered potentially groundbreaking for new ventures.

**SQL Query Assistant Plugin Eyes Transparency and Control**: There's a pitch for making the evaluations of the Datasette SQL query assistant plugin **visible and editable**, aiming to enhance user interaction and control over the evaluation process.

**Perusing the Future of Prompt Management**: A debate is brewing over the best practices for AI prompt management, with potential patterns including **localization, middleware, and microservices**, suggesting different methods for integrating AI into larger systems.

**High-Resolution API Details Exemplified**: The Cohere LLM search API’s detailed JSON responses were spotlighted, providing an example of the granularity that can benefit AI developers, as demonstrated in a shared [GitHub comment](https://github.com/simonw/llm-command-r/issues/2#issuecomment-2037420135).



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **Benchmarking Emotional Smarts**: Newly launched [Creative Writing EQ-Bench](https://eqbench.com/creative_writing.html) and [Judgemark](https://eqbench.com/judgemark.html) benchmarks aim to assess the emotional intelligence of language models, with Judgemark posing a rigorous test through correlation metrics. Standard deviation in scores is leveraged to differentiate how models use 0-10 scales to indicate finer judgment nuances compared to 0-5 rating systems.
  
- **Judgment Day for Creative Writing**: The efficacy of the Creative Writing benchmark is attributed to its **36 specific judging criteria**, emphasizing the importance of narrow parameters for model evaluation. Questions about these benchmark criteria are answered in the extensive documentation provided, demonstrating transparency and allowing for better model assessment.

- **Sizing Up Sentiment and Quality**: Discussion regarding optimal scales revealed that sentiment analysis resonates best with a -1 to 1 range, while quality assessments prefer broader scales of 0-5 or 0-10, aiding models to convey more nuanced opinions. These insights highlight the necessity of tailoring evaluation metrics to the specific domain of judgment.

- **COMET Blazes Through Testing**: The **COMET** evaluation scores herald the Facebook WMT21 model as a standout, with reference-free scores employing **wmt22-cometkiwi-da** methodology alongside useful scripts available on the [llm_translation GitHub repository](https://github.com/CrispStrobe/llm_translation). Nonetheless, caution is advised due to potential inaccuracies, underscoring the need for continual vigilance in assessing model outputs.

- **Scaling the Peaks of Reference-Free Evaluation**: The callout for accuracy in models emphasizes the non-absoluteness of COMET scoring results, with an invitation to flag significant discrepancies—a practice acknowledging the iterative nature of model refinement and validation. The highest COMET score recorded was 0.848375, demonstrating the advanced capabilities of current language models in translation tasks.



---



## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord

- **AI Enthusiasts Eye Healthcare**: Community engagement in AI within the healthcare sector is on the rise, signaling increased cross-disciplinary applications of AI technologies.

- **Evolving LLMs with Mixture-of-Depths (MoD) Approach**: Introduction of the **Mixture-of-Depths (MoD)** technique has been highlighted as a way to allow Language Models to allocate compute resources dynamically, potentially increasing efficiency. The approach and its capabilities are detailed in a paper available on [arXiv](https://arxiv.org/abs/2404.02258).

- **Revolutionizing AI's Approach to Math**: Discussing improved strategies for AI to tackle mathematical problems, it's suggested that training AI to convert word problems into solvable equations is more effective than direct computation. This method leverages the power of established tools like **Python** and **Wolfram Alpha** for the actual calculations.

- **Another Paper Added to the Trove**: Additional resources are being shared, with a [new paper](https://arxiv.org/abs/2404.02684) added to the community's knowledge base, though no further context has been provided.



---

# PART 2: Detailed by-Channel summaries and links



**Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1225006104375726170)** (910 messages🔥🔥🔥): 

- **Stable Diffusion Models and Upscaling**: Users discussed the best practices for creating realistic high-resolution images, suggesting using lower steps, latent upscaling, and the use of hi-res fix to avoid image distortion. Suggested settings include 35 steps with dpmpp ancestral Karras or exponential and accompanying control nets. Higher resolutions like 2k are challenging, often leading to extended generation times and possible image distortion ([related discussion](https://arxiv.org/abs/2404.02905)).

- **The Future of AI and Content Creation**: There was a robust debate on the impact of AI on various creative industries, with speculations about AI's potential to replace traditional roles in Hollywood and the videogame industry. Participants discussed whether AI models like SDXL would render some artist positions redundant and how evolving technology might increase the skill floor, requiring less effort to generate quality content.

- **Lora Training for Specific Items**: A user inquired about training loras for generating images of people wearing specific items, such as corsets. Advice given includes using images of the item from different angles, ideally with backgrounds and faces removed, to prevent the AI from including unintended elements in the generated images.

- **Economic Considerations and AI**: Participants discussed Stability AI's challenges, such as convincing investors and focusing on datasets versus developing new models. The conversation covered the potential of monetizing the dataset for enterprises to cope with the perceived declining interest in research models and the impact of high compute costs.

- **Miscellaneous Chat**: Interactions included light-hearted exchanges with references to cultural subjects, general hellos, acknowledgments of greetings, and random statements that did not correlate with the main topics of discussion. There was also a link to an unrelated parody song shared by a user.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2404.02905">Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction</a>: We present Visual AutoRegressive modeling (VAR), a new generation paradigm that redefines the autoregressive learning on images as coarse-to-fine &#34;next-scale prediction&#34; or &#34;next-resolutio...</li><li><a href="https://huggingface.co/RunDiffusion/Juggernaut-XL-v9/blob/main/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors">Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors · RunDiffusion/Juggernaut-XL-v9 at main</a>: no description found</li><li><a href="https://sdxl.replicate.dev/">SDXL – A settings guide by Replicate</a>: Or how I learned to make weird cats</li><li><a href="https://remix.ai/">Remix</a>: Create, share, and remix AI images and video.</li><li><a href="https://leonardo.ai/">Home v2</a>: Transform your projects with our AI image generator. Generate high-quality, AI generated images with unparalleled speed and style to elevate your creative vision</li><li><a href="https://www.reddit.com/r/3Frame">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=yvOXZ6SV2Rk">Stable Radio 24/7</a>: Stable Radio, a 24/7 live stream that features tracks exclusively generated by Stable Audio.Explore the model and start creating for free on stableaudio.com</li><li><a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Optimizations">Optimizations</a>: Stable Diffusion web UI. Contribute to AUTOMATIC1111/stable-diffusion-webui development by creating an account on GitHub.</li><li><a href="https://github.com/continue-revolution/sd-webui-animatediff/blob/master/docs/features.md#controlnet-v2v">sd-webui-animatediff/docs/features.md at master · continue-revolution/sd-webui-animatediff</a>: AnimateDiff for AUTOMATIC1111 Stable Diffusion WebUI - continue-revolution/sd-webui-animatediff</li><li><a href="https://github.com/comfyanonymous/ComfyUI">GitHub - comfyanonymous/ComfyUI: The most powerful and modular stable diffusion GUI, api and backend with a graph/nodes interface.</a>: The most powerful and modular stable diffusion GUI, api and backend with a graph/nodes interface. - comfyanonymous/ComfyUI</li><li><a href="https://forms.gle/9i4jM9BQu9bVVAAF6">Survey Form - 5day.io</a>: As a young professional just a few years into the workforce, there is a constant, low-humming anxiety about proving yourself and finding that mythical work-life balance everyone talks about. Sometimes...</li><li><a href="https://www.reddit.com/r/3FrameMovies/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://civitai.com/models/133005/juggernaut-xl?modelVersionId=348913">Juggernaut XL - V9 + RunDiffusionPhoto 2 | Stable Diffusion Checkpoint | Civitai</a>: For business inquires, commercial licensing, custom models, and consultation contact me under juggernaut@rundiffusion.com Juggernaut is available o...</li><li><a href="https://github.com/ZHO-ZHO-ZHO/ComfyUI-SegMoE">GitHub - ZHO-ZHO-ZHO/ComfyUI-SegMoE: Unofficial implementation of SegMoE for ComfyUI</a>: Unofficial implementation of SegMoE for ComfyUI. Contribute to ZHO-ZHO-ZHO/ComfyUI-SegMoE development by creating an account on GitHub.</li><li><a href="https://m.soundcloud.com/pelusitalachicafideo/never-gonna-give-you-up-rick-astley-minions-ver">Never Gonna Give You Up - Rick Astley [Minions Ver.]</a>: Stream Never Gonna Give You Up - Rick Astley [Minions Ver.] by Pelusita,la chica fideo on desktop and mobile. Play over 320 million tracks for free on SoundCloud.
</li>
</ul>

</div>
  

---



**Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1225023684968710144)** (756 messages🔥🔥🔥): 

- **API Usage Explained**: Users are curious about the functionalities and costs associated with Perplexity's API. It was clarified that APIs can be very powerful for automating tasks and are essential for developers looking to integrate specific services into their applications. The cost efficiency and usage depend on the scope of the project and the amount of data being processed.
- **Pros and Cons of Pro Subscription vs API**: There's a debate on whether it's more advantageous to subscribe to Perplexity for $20 a month or to use the pay-as-you-go API. For idea generation and beginning a business, the recommendation seems to be towards subscribing due to ease of use and cost management.
- **Model Preferences Discussed**: When it comes to usage, users prefer having a larger number of messages with a decent context window rather than a larger context with fewer messages. Perplexity's AI capabilities are being leveraged for a range of tasks, with the flexibility to work around limitations.
- **Notifications and New UI Elements Update**: There has been mention of news notifications not being readily accessible or communicated effectively, with the suggestion for the company to use Discord's announcement channels more strategically. Some concerns were raised about the lack of updates on the Android app.
- **Integration Limits and Model Capabilities**: Discussion around the limitations when using complex languages like Rust with AI, highlighting that AI models, including Opus, struggle to create compilable code. Some users are applying workarounds like starting new threads to manage large conversations for better context management.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.theverge.com/2024/4/1/24118154/perplexity-will-try-a-form-of-ads-on-its-ai-search-platform">Perplexity will try a form of ads on its AI search platform.</a>: Perplexity’s chief business officer Dmitry Shevelenko tells Adweek the company is considering adding sponsored suggested questions to its platform. If users continue to search for more information on ...</li><li><a href="https://www.theverge.com/2024/4/1/24118154/perplexity-will-try-a-form-of-ads-on-its-ai-search-platfo">Perplexity will try a form of ads on its AI search platform.</a>: Perplexity’s chief business officer Dmitry Shevelenko tells Adweek the company is considering adding sponsored suggested questions to its platform. If users continue to search for more information on ...</li><li><a href="https://docs.perplexity.ai/docs/getting-started">Getting Started with pplx-api</a>: no description found</li><li><a href="https://fontawesome.com/icons/brain-circuit?f=classic&s=thin">Brain Circuit Classic Thin Icon | Font Awesome</a>: Brain Circuit icon in the Thin style. Style your project in the latest super-light designs.  Available now in Font Awesome 6.</li><li><a href="https://fontawesome.com/icons/image?f=classic&s=regular">Image Classic Regular Icon | Font Awesome</a>: Image icon in the Regular style. Smooth out your design with easygoing, readable icons.  Available now in Font Awesome 6.</li><li><a href="https://www.tomsguide.com/ai/apple-reveals-realm-new-ai-model-could-make-siri-way-faster-and-smarter">Apple reveals ReALM &mdash; new AI model could make Siri way faster and smarter</a>: ReALM could be part of Siri 2.0</li><li><a href="https://docs.perplexity.ai/reference/post_chat_completions">Chat Completions</a>: no description found</li><li><a href="https://tenor.com/view/ralph-wiggum-simpsons-hi-bye-gif-16529059407582436389">Ralph Wiggum Simpsons GIF - Ralph Wiggum Simpsons Hi - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.reddit.com/r/perplexity_ai/comments/11s44ry/discord_bo">Reddit - Dive into anything</a>: no description found</li><li><a href="https://x.com/aravsrinivas/status/1775632536934486160?s=46">Tweet from Aravind Srinivas (@AravSrinivas)</a>: Interesting.</li><li><a href="https://www.reddit.com/r/perplexity_ai/comments/11s44ry/discord_bot_public_releaseintroducing/?rdt=64126">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=JV4JbYK-TIg">1111Hz Conéctate con el universo - Recibe guía del universo - Atrae energías mágicas y curativas #2</a>: 1111Hz Conéctate con el universo - Recibe guía del universo - Atrae energías mágicas y curativas #2Este canal se trata de curar su mente, alma, cuerpo, trast...</li><li><a href="https://gist.github.com/cjanietz/703a88924e50e1a30cb6ffc52bc52bd9">Perplexity Model Selection User Script</a>: Perplexity Model Selection User Script. GitHub Gist: instantly share code, notes, and snippets.
</li>
</ul>

</div>
  

---


**Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1225019849672228904)** (15 messages🔥): 

- **Exploring the Impact of Fritz Haber**: A member highlighted **Fritz Haber's** contributions, such as enabling increased food production through the *Haber-Bosch process*. His complex legacy includes the Nobel Prize, involvement in chemical warfare, personal tragedies, and anti-Nazi sentiments. [Read about Fritz Haber's legacy](https://www.perplexity.ai/search/Who-is-Fritz-kSg0wtgUSombH0qTxfuzzg).

- **Intrigue at the LLM Leaderboard**: A user examined the **LLM Leaderboard**, discussing model metrics and rankings, and discovered what *"95% CI"* means despite encountering amusing model name errors. [Explore the LLM Leaderboard review](https://www.perplexity.ai/search/LLM-Leaderboard-Review-4C4F5TQuQSSxnYBWVfEZgg).

- **Understanding Beauty through AI**: Multiple members shared their curiosity about the concept of beauty by using **Perplexity AI** to access insights on the topic. [Delve into the nature of beauty](https://www.perplexity.ai/search/Why-is-beauty-IIA2.dXGSCOwM5aXlcbuVA).

- **Dictatorship Discussion Initiated**: One chat pointed users to **Perplexity AI** for a query on how dictatorship naturally arises, sparking an intellectual query into the origins of authoritarian regimes. [Investigate the emergence of dictatorship](https://www.perplexity.ai/search/Dictatorship-naturally-arises-qRK3sToeRYqDa3Y_oP6Ztw).

- **Reminder for Shareable Content**: A member was reminded to ensure their **thread** is set to shareable when posting links from the Discord channel. This ensures others can view and engage with the content shared. [Make Discord threads shareable](https://discord.com/channels/1047197230748151888/1054944216876331118/1208752189606989825).
  

---


**Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1225014380295491667)** (42 messages🔥): 

- **Perplexed about Perplexity API Sonar Model Source Links**: A user inquired about when the **sonar-medium-online** model would be able to return the source link with the data, but did not receive a clear timeline on when this feature will be available.
- **Credit Conundrum: Payments Pending in Perplexity**: A member reported issues while trying to buy API credits; transactions showed as "Pending" and did not reflect in the account balance. Another member asked them to send account details for resolution, indicating a case-by-case troubleshooting approach.
- **Trouble with Realms, ReALM, and Apple**: Users experienced the bot getting confused when asked about Apple's ReALM, leading one suggestion that simplifying the system prompt might yield better performance, as complexity seems to lead to confusion.
- **Custom GPT "metaprompt" for Organized Output**: One user shared their experiment with creating a Custom GPT utilizing a "metaprompt" aimed at structuring responses efficiently, which primarily focused on delivering accurate information with clear citations.
- **Search API Pricing Perplexities**: A member questioned the pricing of search APIs compared to language models, discussing the cost-effectiveness of 1000 online model requests, which another clarified does not equate to 1000 individual searches but rather requests that can contain multiple searches each.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://....)"">no title found</a>: no description found</li><li><a href="https://tenor.com/view/were-on-the-same-boat-here-mickey-haller-lincoln-lawyer-we-have-a-common-problem-we-have-the-same-issue-gif-9336579479687485405">Were On The Same Boat Here Mickey Haller GIF - Were on the same boat here Mickey haller Lincoln lawyer - Discover &amp; Share GIFs</a>: Click to view the GIF
</li>
</ul>

</div>
  

---



**OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1225128391813103706)** (2 messages): 

<ul>
  <li>
    <strong>DALL·E gets an Editing Suite</strong>: Members were informed that they can now edit DALL·E images in ChatGPT across web, iOS, and Android, as well as receive style inspiration when creating images in the DALL·E GPT.
  </li>
  <li>
    <strong>Fine-tuning API Level Up</strong>: New dashboards, metrics, and integrations have been introduced in the fine-tuning API. Developers now have more control and new options for building custom models with OpenAI, detailed in a recent blog post: <a href="https://openai.com/blog/introducing-improvements-to-the-fine-tuning-api-and-expanding-our-custom-models-program">Fine-tuning API and Custom Models Program</a>.
  </li>
</ul>

**Link mentioned**: <a href="https://openai.com/blog/introducing-improvements-to-the-fine-tuning-api-and-expanding-our-custom-models-program">Introducing improvements to the fine-tuning API and expanding our custom models program</a>: We’re adding new features to help developers have more control over fine-tuning and announcing new ways to build custom models with OpenAI.

  

---


**OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1225007451527712788)** (494 messages🔥🔥🔥): 

- **Understanding AI and Consciousness**: Discussions revolved around the nature of AI's cognitive processes compared to human thought, debating whether AI such as LLMs are capable of "thinking" or just sophisticated algorithms performing complex patterns of data. Multiple participants contended that AI lacks consciousness and is instead a simulation of human-like behaviors.
  
- **The Complexity of Defining Sentience**: Sentience and consciousness were hot topics, with explorations into the subjective experiences of animals as revealed by neural activity studies. The conversation pointed out the difficulty of discerning sentience in different life forms and the challenges in defining consciousness solely based on human-like behavior.

- **AI Misconceptions and Expectations**: Some discussion highlighted a public misconception about AI, where many people equate all forms of AI with the concept of AGI (Artificial General Intelligence), as often depicted in science fiction. There was an emphasis on the need for clear distinctions between various forms of AI and the reality of current technologies.

- **Live Discussion Dynamics**: Debates about AI often led to friction amongst participants, demonstrating a wide spectrum of beliefs and opinions about AI's capabilities, consciousness, and ethical considerations. Some recommended additional resources like YouTube videos to reinforce their viewpoints.

- **Potential AI Usage and Development Ideas**: One user suggested training language models with goal-oriented sequences, such as `success <doing-business> success`, for various applications including playing chess or developing business strategies, theorizing about its interactive possibilities when presented with new information during inference.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://en.wikipedia.org/wiki/China_brain">China brain - Wikipedia</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=R9OHn5ZF4Uo">How AIs, like ChatGPT, Learn</a>: How do all the algorithms, like ChatGPT, around us learn to do their jobs?Footnote: https://www.youtube.com/watch?v=wvWpdrfoEv0Thank you to my supporters on ...</li><li><a href="https://www.asciiart.eu/food-and-drinks/bananas">ASCII Art Bananas - asciiart.eu</a>: A large collection of ASCII art drawings of bananas and other related food and drink ASCII art pictures.</li><li><a href="https://www.lesswrong.com/posts/vJFdjigzmcXMhNTsx/simulators">Simulators — LessWrong</a>: Thanks to Chris Scammell, Adam Shimi, Lee Sharkey, Evan Hubinger, Nicholas Dupuis, Leo Gao, Johannes Treutlein, and Jonathan Low for feedback on draf…
</li>
</ul>

</div>
  

---


**OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1225060200810156115)** (46 messages🔥): 

- **Custom GPT vs. Base Model**: Channel members are discussing the advantages of using Custom GPTs over the base ChatGPT models. While some prefer the ease of building complex prompts with Custom GPTs, others find base ChatGPT models to be sufficient for their needs and challenge the need for Custom GPTs when prompt engineering alone can be effective.

- **DALL·E Gains New Features**: DALL·E has been updated with new features allowing for style suggestions and image inpainting, enabling users to edit specific parts of an image generated by DALL·E. This information might be particularly interesting for Plus plan users looking to utilize these functionalities.

- **Comparing Model Performance**: There's an exchange regarding the performance of various GPT models and systems, with some members noting that in specific areas, some models might outperform others. The conversation shows a nuanced understanding that model performance can vary greatly depending on the use case and individual testing.

- **Utilizing AI for Wiki Data**: A member is seeking advice on how to have GPT interpret and answer questions from an XML file containing a Wiki database dump. They expressed difficulty with the GPT providing accurate responses from the data in the XML file.

- **Data Retention Questions**: Users inquire about OpenAI’s data retention policy, specifically after deleting a chat. The response indicates that deleted chats on OpenAI are typically held for about a month, though they become immediately invisible to the user upon deletion.
  

---


**OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1225080073271382066)** (27 messages🔥): 

- **Translation Troubles**: A member is experiencing inconsistency when translating markdown content to various languages, especially Arabic. Efforts to tweak the prompt, such as adding "Only return translated text with its markup, not the original text," resulted in mixed outcomes, with some responses being untranslated.

- **Seeking the Prompt Library**: One member inquired about the location of the **prompt library**, and another quickly guided them to the renamed channel using its channel ID.

- **Perfecting Apple Watch Expertise Prompts**: A user sought advice on improving prompts to get better responses from the bot when asking as an Apple Watch expert. Another member advised experimenting with different versions of the prompts and even using the model to evaluate the prompts for clarity and potential hallucinations.

- **Dalle-3 Prompt Engineering Location Query**: A user questioned where to conduct Dalle-3 prompt engineering, whether in the general prompt-engineering channel or a specific Dalle thread. A member suggested it's their choice, but more focused help might be available in the Dalle-specific channel.

- **Lengthening Text Responses**: A member expressed frustration that the command "make the text longer" was no longer effective. Another member recommended a workaround involving copying the previous GPT response, starting a new chat, and then prompting with "continue."

- **LLM Draft Document Issue**: A member asked for assistance with an LLM that fails to return certain sections of a document while drafting from a template, even when changes have been made to those sections. They are looking for a solution to ensure all modified sections are included in the outputs.
  

---


**OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1225080073271382066)** (27 messages🔥): 

- **Translation Troubles with Markup**: A member attempted various prompt formulations to preserve Markdown markup and correctly translate content, including proper names and links, from one language to another. Despite the refined prompts, they faced issues with maintaining markdown formatting and receiving untranslated text, expressing frustration over the inconsistent translations.

- **Seeking the Prompt Library**: When asked where to find the prompt library, a member was directed to a channel renamed to [#1019652163640762428](https://discord.com/channels/channel_id/1019652163640762428), indicating the location where resources can be found.

- **Improving Prompt Efficacy for AI Role Play**: In a discussion about enhancing the quality of prompts for role-playing experts, a member suggested asking the AI to evaluate the prompt for clarity and consistency. They discussed the importance of the entire context of the prompt beyond single keywords like "roleplay" to influence the AI's response style.

- **Dalle-3 Prompt Engineering Discussion Placement**: A member inquired where to discuss Dalle-3 prompt engineering—whether in the api-discussions channel or a Dalle-specific thread. They were told it's their choice, though a more focused response might be found in a dedicated Dalle thread.

- **Extending Iterative Text Generation**: After experiencing issues with the command "make the text longer" not generating new content as expected, another member suggested copying the AI's response, initiating a new chat, and then using the word "continue" to extend the conversation.
  

---



**Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1225004211947966485)** (306 messages🔥🔥): 

- **Support for CmdR on the Horizon**: Discussions indicate work is underway to add support for **CmdR** in Unsloth, following fixes to inference issues. There's excitement about the progress, and discussions imply a completion time frame after the current tasks.

- **Anticipation for Automatic Optimizer**: Unsloth is planning a new open-source feature *very important for the GPU poor* set to be announced imminently, with a **new release and an announcement on April 22**. This feature is aimed to ameliorate AI accessibility by CPU optimization, supporting a wider range of models like **command r, Mixtral, etc.**.

- **Performance Queries Addressed**: Users engaged in technical discussions about memory optimization, VRAM savings of 70% with Unsloth, and inplace kernel executions. The conversation highlights inquiries about *data layout results on different models* and the effectiveness of *Unsloth's in-place operations* for memory reduction.

- **Enthusiasm and Confusion Clearing** about the **Gemma 2B Model**: Support is given for changing to the Gemma 2B model in notebooks with provided instructions, and clarifications on downloading models in 4-bit versus 16-bit, with an assurance that accuracy degradation is usually between 0.1-0.5%.

- **Job Postings and Ethical Hiring Discussed**: A request for a job channel sparked debates on the ethics of unpaid internships and the skill set expected from interns. The consensus emphasized the importance of providing **financial compensation** for any work performed.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/15g">Google Colaboratory</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1-uKmQzhh8ftxEdipiqGu4sVdRb8MgWv2?usp=sharing">Google Colaboratory</a>: no description found</li><li><a href="https://colab.research.google.com/drive/15gGm7x_jTm017_Ic8e317tdIpDG53Mtu?usp=sharing">Google Colaboratory</a>: no description found</li><li><a href="https://colab.research.google.com/drive/10NbwlsRChbma1v55m8LAPYG15uQv6HLo?usp=sharing#scrollTo=FqfebeAdT073">Google Colaboratory</a>: no description found</li><li><a href="https://huggingface.co/unsloth">unsloth (Unsloth)</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/wiki#manually-saving-to-gguf">Home</a>: 2-5X faster 70% less memory QLoRA &amp; LoRA finetuning - unslothai/unsloth</li><li><a href="https://github.com/myshell-ai/JetMoE">GitHub - myshell-ai/JetMoE: Reaching LLaMA2 Performance with 0.1M Dollars</a>: Reaching LLaMA2 Performance with 0.1M Dollars. Contribute to myshell-ai/JetMoE development by creating an account on GitHub.</li><li><a href="https://github.com/OpenNLPLab/LASP/tree/main">GitHub - OpenNLPLab/LASP: Linear Attention Sequence Parallelism (LASP)</a>: Linear Attention Sequence Parallelism (LASP). Contribute to OpenNLPLab/LASP development by creating an account on GitHub.</li><li><a href="https://github.com/toranb/sloth/blob/master/sftune.py">sloth/sftune.py at master · toranb/sloth</a>: python sftune, qmerge and dpo scripts with unsloth - toranb/sloth</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1546dvc/24gb_vram_on_a_budget/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/ggerganov/llama.cpp">GitHub - ggerganov/llama.cpp: LLM inference in C/C++</a>: LLM inference in C/C++. Contribute to ggerganov/llama.cpp development by creating an account on GitHub.</li><li><a href="https://github.com/pytorch-labs/ao/pull/95#issuecomment-2028912362">GaLore and fused kernel prototypes by jeromeku · Pull Request #95 · pytorch-labs/ao</a>: Prototype Kernels and Utils Currently:  GaLore  Initial implementation of fused kernels for GaLore memory efficient training.    TODO:  triton  Composable triton kernels for quantized training and ...
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1225229765791973467)** (5 messages): 

- **Unsloth Studio Overhaul in Progress**: Unsloth AI team is delaying the release of the new version of Unsloth Studio due to persistent bugs. A tentative, early version might be available mid next month and the existing Unsloth package will remain compatible with **Colab**.

- **New Tab Autocomplete Feature in Pre-Release**: A new pre-release experimental feature for **tab autocomplete** is available in the [Continue](https://marketplace.visualstudio.com/items?itemName=Continue.continue) extension for VS Code and JetBrains. Continue's open-source autopilot allows easier coding with any LLM by asking questions about highlighted code and referencing context inline, as showcased with animated GIFs in its [documentation](https://continue.dev/docs).

**Link mentioned**: <a href="https://marketplace.visualstudio.com/items?itemName=Continue.continue">Continue&#32;-&#32;Claude,&#32;CodeLlama,&#32;GPT-4,&#32;and&#32;more&#32;-&#32;Visual&#32;Studio&#32;Marketplace</a>: Extension&#32;for&#32;Visual&#32;Studio&#32;Code&#32;-&#32;Open-source&#32;autopilot&#32;for&#32;software&#32;development&#32;-&#32;bring&#32;the&#32;power&#32;of&#32;ChatGPT&#32;to&#32;your&#32;IDE

  

---


**Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1225000119657566218)** (248 messages🔥🔥): 

- **Tokenizer Troubles**: The error a user faced was due to incorrect naming of the model in the tokenizer, which resulted in it not being written properly, leading to execution issues. They resolved the issue on their own.

- **Successful Model Saving and Huggingface Push**: Users discussed saving models with `model.save_pretrained_merged()` and `model.push_to_hub_merged()`, focusing on properly setting naming parameters for model saving and Huggingface push. Relevant advice included replacing placeholders with a Huggingface username/model name and obtaining a Write token from Huggingface settings.

- **Inference Issues on Gemma**: A user encountered an `AttributeError` related to a `GemmaForCausalLM` object missing the `layers` attribute, which was fixed via an update to Unsloth requiring users to reinstall the package on personal machines.

- **Challenges with GGUF Conversions and Docker Environments**: Users shared issues when converting models to GGUF format, and an instance where the Docker environment produced an error that was solved with the installation of `python3.10-dev`.

- **Finetuning Challenges and Solutions**: Discussion included finetuning Gemma models in Colab, remedies for `OutOfMemoryError` when using 24GB GPUs on Sagemaker, a GGUF-spelled words quirk after conversion, and insights on resuming training with altered parameters.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/settings/tokens">Hugging Face – The AI community building the future.</a>: no description found</li><li><a href="https://discuss.huggingface.co/t/adding-accuracy-precision-recall-and-f1-score-metrics-during-training/16419/2">Adding accuracy, precision, recall and f1 score metrics during training</a>: hi, you can define your computing metric function and pass it into the trainer. Here is an example of computing metrics.   define accuracy metrics function from sklearn.metrics import accuracy_score, ...</li><li><a href="https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0">TinyLlama/TinyLlama-1.1B-Chat-v1.0 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/deepseek-ai/deepseek-vl-7b-chat/">deepseek-ai/deepseek-vl-7b-chat · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/qwp4w3hyb/deepseek-coder-7b-instruct-v1.5-iMat-GGUF">qwp4w3hyb/deepseek-coder-7b-instruct-v1.5-iMat-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/danielhanchen/model_21032024">danielhanchen/model_21032024 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/TheBloke/deepseek-coder-6.7B-instruct-GGUF">TheBloke/deepseek-coder-6.7B-instruct-GGUF · Hugging Face</a>: no description found</li><li><a href="https://docs.wandb.ai/guides/integrations/huggingface">Hugging Face Transformers | Weights &amp; Biases Documentation</a>: The Hugging Face Transformers library makes state-of-the-art NLP models like BERT and training techniques like mixed precision and gradient checkpointing easy to use. The W&amp;B integration adds rich...</li><li><a href="https://huggingface.co/docs/trl/main/en/sft_trainer#trl.trainer.ConstantLengthDataset">Supervised Fine-tuning Trainer</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/wiki#evaluation-loop---also-oom-or-crashing">Home</a>: 2-5X faster 70% less memory QLoRA &amp; LoRA finetuning - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/wiki#saving-models-to-16bit-for-vllm">Home</a>: 2-5X faster 70% less memory QLoRA &amp; LoRA finetuning - unslothai/unsloth</li><li><a href="https://github.com/oobabooga/text-generation-webui">GitHub - oobabooga/text-generation-webui: A Gradio web UI for Large Language Models. Supports transformers, GPTQ, AWQ, EXL2, llama.cpp (GGUF), Llama models.</a>: A Gradio web UI for Large Language Models. Supports transformers, GPTQ, AWQ, EXL2, llama.cpp (GGUF), Llama models. - oobabooga/text-generation-webui</li><li><a href="https://github.com/unslothai/unsloth/pull/300">fix GemmaModel_fast_forward_inference by eabdullin · Pull Request #300 · unslothai/unsloth</a>: On fast inference Gemma model fails with an error &#39;GemmaCausalLM&#39; has no attribute &#39;layers&#39;. Quick fix for that.</li><li><a href="https://github.com/abetlen/llama-cpp-python">GitHub - abetlen/llama-cpp-python: Python bindings for llama.cpp</a>: Python bindings for llama.cpp. Contribute to abetlen/llama-cpp-python development by creating an account on GitHub.</li><li><a href="https://github.com/unslothai/unsloth/">GitHub - unslothai/unsloth: 2-5X faster 70% less memory QLoRA &amp; LoRA finetuning</a>: 2-5X faster 70% less memory QLoRA &amp; LoRA finetuning - unslothai/unsloth
</li>
</ul>

</div>
  

---



**Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1225086904501014601)** (86 messages🔥🔥): 

- **Stable Audio 2.0 Released**: Stability AI introduces **Stable Audio 2.0**, capable of producing high-quality, full tracks with coherent musical structure up to three minutes long at 44.1 kHz stereo from a single prompt. Users can explore the model for free at [stableaudio.com](http://stableaudio.com) and read the blog post [here](https://bit.ly/3xlwmP5).

- **AssemblyAI's New Speech Model Surpasses Whisper-3**: AssemblyAI releases **Universal-1**, a model boasting 13.5% more accuracy and up to 30% fewer hallucinations than Whisper-3, capable of processing 60 minutes of audio in 38 seconds, though it only supports 20 languages. Test it in the free playground at [assemblyai.com](https://www.assemblyai.com/playground).

- **Edit DALL-E Images in ChatGPT Plus**: ChatGPT Plus now allows users to edit DALL-E images and their own conversation prompts on the web and iOS app. Instructions and user interface details can be found [here](https://help.openai.com/en/articles/9055440-editing-your-images-with-dall-e).

- **AI Framework Discussion by slono**: Slono shared thoughts on building AI agents as microservices/event-driven architecture for better scalability, invoking ideas similar to the Actor Model of Computation and seeking feedback or assistance with their Golang framework.

- **Opera Allows Downloading and Running Local LLMs**: Opera now enables users to download and run large language models (LLMs) locally, starting with Opera One users who have developer stream updates. The browser is making use of the open-source Ollama framework and plans to add more models from various sources for users' choice.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://dev.to/devteam/join-us-for-the-cloudflare-ai-challenge-3000-in-prizes-5f99">no title found</a>: no description found</li><li><a href="https://x.com/horseracedpast/status/1775757613000507736?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from horseboat (@horseracedpast)</a>: bengio really wrote this in 2013 huh  ↘️ Quoting AK (@_akhaliq)   Google presents Mixture-of-Depths  Dynamically allocating compute in transformer-based language models  Transformer-based language mod...</li><li><a href="https://deluxe-fairy-96e9ff.netlify.app/">React App</a>: no description found</li><li><a href="https://x.com/theseamouse/status/1775743110774931846?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Hassan Hayat 🔥 (@TheSeaMouse)</a>: @fouriergalois @GoogleDeepMind bro, MoE with early exit. the entire graph is shifted down, this is like 10x compute savings... broooo</li><li><a href="https://openai.com/blog/introducing-improvements-to-the-fine-tuning-api-and-expanding-our-custom-models-program">Introducing improvements to the fine-tuning API and expanding our custom models program</a>: We’re adding new features to help developers have more control over fine-tuning and announcing new ways to build custom models with OpenAI.</li><li><a href="https://overcast.fm/+HaNOG0VjE/19:08">Should kids still learn to code? (Practical AI #263) &mdash; Changelog Master Feed &mdash; Overcast</a>: no description found</li><li><a href="https://x.com/theseamouse/status/1775782800362242157?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Hassan Hayat 🔥 (@TheSeaMouse)</a>: Why Google Deepmind&#39;s Mixture-of-Depths paper, and more generally dynamic compute methods, matter:  Most of the compute is WASTED because not all tokens are equally hard to predict</li><li><a href="https://techcrunch.com/2024/04/03/opera-will-now-allow-users-download-and-use-llms-locally">Opera allows users to download and use LLMs locally | TechCrunch</a>: Opera said today it will now allow users to download and use Large Language Models (LLMs) locally on their desktop.</li><li><a href="https://overcast.fm/+_C9f-UYD4">Open sourcing AI app development with Harrison Chase from LangChain &mdash; No Priors: Artificial Intelligence | Machine Learning | Technology | Startups &mdash; Overcast</a>: no description found</li><li><a href="https://x.com/StabilityAI/status/1775501906321793266?s=20">Tweet from Stability AI (@StabilityAI)</a>: Introducing Stable Audio 2.0 – a new model capable of producing high-quality, full tracks with coherent musical structure up to three minutes long at 44.1 kHz stereo from a single prompt.  Explore the...</li><li><a href="https://x.com/nickadobos/status/1775638457412722757?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Nick Dobos (@NickADobos)</a>: New Dalle is so good wtf Way more steerable than anything else I’ve tried  I made an app mockup in 3 prompts. Wow!! Even sorta got the tab bar & a layout</li><li><a href="https://x.com/cohere/status/1775878850699808928?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from cohere (@cohere)</a>: Today, we’re introducing Command R+: a state-of-the-art RAG-optimized LLM designed to tackle enterprise-grade workloads and speak the languages of global business.  Our R-series model family is now av...</li><li><a href="https://x.com/sherjilozair/status/1775765404528615798?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Sherjil Ozair (@sherjilozair)</a>: How did this get published? 🤔  ↘️ Quoting AK (@_akhaliq)   Google presents Mixture-of-Depths  Dynamically allocating compute in transformer-based language models  Transformer-based language models sp...</li><li><a href="https://x.com/andersonbcdefg/status/1775751252330385807?s=20">Tweet from Ben (e/sqlite) (@andersonbcdefg)</a>: amazing. &#34;you like MoE? what if we made one of the experts the identity function.&#34; kaboom, 50% FLOPs saved 🤦‍♂️  ↘️ Quoting Aran Komatsuzaki (@arankomatsuzaki)   Google presents Mixture-of-De...</li><li><a href="https://x.com/gblazex/status/1775558982645547236?s=20">Tweet from Blaze (Balázs Galambosi) (@gblazex)</a>: Wow. While OpenAI API is still stuck on Whisper-2, @AssemblyAI releases something that beats even Wishper-3: + 13.5% more accurate than  Whisper-3  + Up to 30% fewer hallucinations + 38s to process 60...</li><li><a href="https://stanford.zoom.us/j/99922151759?pwd=dW5CcUtVYkNybGZGY0hMWUZtVkZBZz09">Join our Cloud HD Video Meeting</a>: Zoom is the leader in modern enterprise video communications, with an easy, reliable cloud platform for video and audio conferencing, chat, and webinars across mobile, desktop, and room systems. Zoom ...</li><li><a href="https://www.reddit.com/r/computervision/comments/1bvaak0/stanford_cs_25_transformers_course_open_to/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://lu.ma/paperclub3">SDxPaperClub · Luma</a>: The SDx Paper Club.  The paper to be presented is [TBD] by [TBD]  Twitter | Discord | LinkedIn</li><li><a href="https://hlfshell.ai/posts/representation-engineering/">Representation Engineering and Control Vectors - Neuroscience for LLMs</a>: tl;dr A recent paper studied large language model&rsquo;s (LLM) reactions to stimuli in a manner similar to neuroscience, revealing an enticing tool for controlling and understanding LLMs. I write her...</li><li><a href="https://github.com/Paitesanshi/LLM-Agent-Survey">GitHub - Paitesanshi/LLM-Agent-Survey</a>: Contribute to Paitesanshi/LLM-Agent-Survey development by creating an account on GitHub.</li><li><a href="https://abyssinian-molybdenum-f76.notion.site/237e9f7515d543c0922c74f4c3012a77?v=0a309e53d6454afcbe7a5a7e169be0f9">Notion – The all-in-one workspace for your notes, tasks, wikis, and databases.</a>: A new tool that blends your everyday work apps into one. It's the all-in-one workspace for you and your team</li><li><a href="https://youtu.be/5q0GN2M1d2c?si=zRsm4Jye_YO8jfBz">Multimodal AI: Antonio Torralba</a>: Antonio Torralba, Professor, MIT Electrical Engineering and Computer Science and CSAIL, on visual perception and language models.Torralba’s talk was part of ...</li><li><a href="https://www.amazon.com/Foundations-Computer-Adaptive-Computation-Learning/dp/0262048973">no title found</a>: no description found</li><li><a href="https://mitpress.ublish.com/ebook/foundations-of-computer-vision-1-preview/12791/Cover">eReader</a>: no description found
</li>
</ul>

</div>
  

---


**Latent Space ▷ #[llm-paper-club-west](https://discord.com/channels/822583790773862470/1197350122112168006/1225158181316067422)** (356 messages🔥🔥): 

- **Intro to Detailed Summarization**: Members discussed the ins and outs of using **DSPy** for optimizing prompting in foundation models, focusing on its efficacy in model migration and optimization for arbitrary metrics. Eric shared his presentation and participants acknowledged his insights with a round of applause.
  
- **Devin Draws Attention**: Conversation shifted towards the manifold implications of **Devin**, with members sharing various project ideas that could be attempted using this high-profile AI model.
  
- **Hot Topic on Optimization Calls**: The club identified **dspy's optimization** technique and raised concerns regarding API rate limits during the **.compile()** function calls due to the large number of calls **DSPy** makes.
  
- **Pragmatic Programming Considerations**: Questions arose about **practical use cases** for DSPy Vs. other methods/frameworks, its advantages in different contexts, and how to mitigate issues like prompt debt during model migration.
  
- **Tech and Task Speculations**: Suggestions for potential applications using **Devin** ranged from iOS apps with voice API integration to DSPy documentation rewrites, showcasing the breadth of community interest in applying AI to diverse challenges.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://app.sli.do/event/bNV6mo3BFGhe8Bqzb1tonb">Join Slido: Enter #code to vote and ask questions</a>: Participate in a live poll, quiz or Q&A. No login required.</li><li><a href="https://colab.research.google.com/drive/1KZR1sGTp_RLWUJPAiK1FKPKI-Qn9neUm?usp=sharing">Google Colaboratory</a>: no description found</li><li><a href="https://app.sli.do/event/bNV6mo3BFGhe8Bqzb1tonb/live/questions">Join Slido: Enter #code to vote and ask questions</a>: Participate in a live poll, quiz or Q&A. No login required.</li><li><a href="https://arxiv.org/abs/2402.17764">The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits</a>: Recent research, such as BitNet, is paving the way for a new era of 1-bit Large Language Models (LLMs). In this work, we introduce a 1-bit LLM variant, namely BitNet b1.58, in which every single param...</li><li><a href="https://eugeneyan.com/writing/abstractive/">Evaluation & Hallucination Detection for Abstractive Summaries</a>: Reference, context, and preference-based metrics, self-consistency, and catching hallucinations.</li><li><a href="https://arxiv.org/abs/2310.03714">DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines</a>: The ML community is rapidly exploring techniques for prompting language models (LMs) and for stacking them into pipelines that solve complex tasks. Unfortunately, existing LM pipelines are typically i...</li><li><a href="https://eugeneyan.com/writing/evals/#summ">LLM Task-Specific Evals that Do & Don't Work</a>: Evals for classification, summarization, translation, copyright regurgitation, and toxicity.</li><li><a href="https://www.spotery.com/">Are you human?</a>: no description found</li><li><a href="https://eugeneyan.com/writing/evals/#summarization-consistency-relevance-length">LLM Task-Specific Evals that Do & Don't Work</a>: Evals for classification, summarization, translation, copyright regurgitation, and toxicity.</li><li><a href="https://hamel.dev/blog/posts/prompt/#dspy">- Fuck You, Show Me The Prompt.</a>: Quickly understand inscrutable LLM frameworks by intercepting API calls.</li><li><a href="https://github.com/stanfordnlp/dspy/blob/main/examples/knn.ipynb">dspy/examples/knn.ipynb at main · stanfordnlp/dspy</a>: DSPy: The framework for programming—not prompting—foundation models - stanfordnlp/dspy</li><li><a href="https://github.com/seanchatmangpt/dspygen">GitHub - seanchatmangpt/dspygen: A Ruby on Rails style framework for the DSPy (Demonstrate, Search, Predict) project for Language Models like GPT, BERT, and LLama.</a>: A Ruby on Rails style framework for the DSPy (Demonstrate, Search, Predict) project for Language Models like GPT, BERT, and LLama. - seanchatmangpt/dspygen</li><li><a href="https://github.com/stanfordnlp/dspy">GitHub - stanfordnlp/dspy: DSPy: The framework for programming—not prompting—foundation models</a>: DSPy: The framework for programming—not prompting—foundation models - stanfordnlp/dspy</li><li><a href="https://x.com/HamelHusain/status/1774999027538612652?s=20">Tweet from Hamel Husain (@HamelHusain)</a>: @swyx a guy + a small cult of fans
</li>
</ul>

</div>
  

---



**Nous Research AI ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/1225485539952689162)** (2 messages): 

- **Exploring LoRA for Enhanced Mistral**: A suggestion was made about creating a **LoRA** (Low-Rank Adaptation) on top of something like **Mistral 7B** to achieve superior performance in specific tasks.
- **Advanced Splitting and Labeling Planned**: This approach is confirmed to be in the planning stages, where the task would involve not just splitting sentences, but also splitting and labeling each sentence according to a specific taxonomy.
  

---


**Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1225106882281340989)** (9 messages🔥): 

- **The Web Crawling Conundrum**: Discussing **scalable web crawling**, members acknowledged challenges in obtaining large, quality datasets, and noted the increased complexity and costs due to the need for headless browsers, bypassing anti-bot measures, and rendering modern JavaScript frameworks.

- **The Secret Archives**: A member hinted at the existence of **archival groups** that possess a wealth of high-quality data, suggesting a discreet community that archives extensive datasets.

- **The Search for Data Hoarders**: In response to a question about **archival groups**, another participant clarified the distinction between those who archive data out of principle and mere data hoarders.

- **Data Scavenging for Knowledge Collectors**: One member suggested looking into **Common Crawl** as a resource for those interested in web crawling and the state of the art in data collection.

- **Eternal Playlist Addition**: A light-hearted message where a member mentioned choosing a **new song** for their funeral, representing personal interests and a break from more technical discussions.
  

---


**Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1224997466638651403)** (10 messages🔥): 

- **Lollms on Ollama Server**: A [YouTube tutorial](https://www.youtube.com/watch?v=RuQSQmolXGE) was shared about installing and using lollms with Ollama Server, promising to guide viewers through the installation process.
- **Cheaper AI Chips from China**: [Intellifusion's DeepEyes](https://www.tomshardware.com/tech-industry/artificial-intelligence/chinese-chipmaker-launches-14nm-ai-processor-thats-90-cheaper-than-gpus) AI boxes, costing around $140, are offering 48 TOPS of AI performance, aiming to provide a cost-effective alternative to high-end hardware in AI applications.
- **Precision of Time**: A member referenced the [ISO 8601 standard](https://en.wikipedia.org/wiki/ISO_8601) on Wikipedia, detailing the precise way to express current date and time in different formats including UTC and with offsets.
- **Dataset Loaders on GitHub**: Hugging Face introduced [Chug](https://github.com/huggingface/chug), a repository with minimal sharded dataset loaders, decoders, and utils for multi-modal document, image, and text datasets.
- **CohereForAI's Multilingual LLM**: CohereForAI announced the release of C4AI Command R+, a 104B LLM that is multilingual in 10 languages, adding to their open weights offerings, which can be found on their [Hugging Face page](https://huggingface.co/CohereForAI/c4ai-command-r-plus).
- **GPT-4 Fine-tuning Pricing Strategy**: OpenAI has experimental pricing for GPT-4 fine-tuning as they learn about quality, safety, and usage, which are detailed in a recent [blog post](https://openai.com/gpt-4-ft-experimental-pricing).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.tomshardware.com/tech-industry/artificial-intelligence/chinese-chipmaker-launches-14nm-ai-processor-thats-90-cheaper-than-gpus">Chinese chipmaker launches 14nm AI processor that's 90% cheaper than GPUs &mdash; $140 chip's older node sidesteps US sanctions</a>: If there's a way to sidestep sanctions, you know China is on that beat.</li><li><a href="https://openai.com/gpt-4-ft-experimental-pricing">GPT-4 Fine-Tuning</a>: no description found</li><li><a href="https://en.wikipedia.org/wiki/ISO_8601">ISO 8601 - Wikipedia</a>: no description found</li><li><a href="https://x.com/cohereforai/status/1775878631715217522?s=46&t=stOPrwZiN_fxSK0RuC8Flg">Tweet from Cohere For AI (@CohereForAI)</a>: Announcing C4AI Command R+ open weights, a state-of-the-art 104B LLM with RAG, tooling and multilingual in 10 languages.   This release builds on our 35B and is a part of our commitment to make AI bre...</li><li><a href="https://www.youtube.com/watch?v=RuQSQmolXGE">Installing &amp; Unleashing the Power of lollms with Ollama Server: A Fun Tech Tutorial 🚀</a>: 🌟 Hey YouTube fam! 🤓 I&#39;m so excited to present my newest video to you all! In this enlightening tutorial, I&#39;ll walk you through the process of installing a...</li><li><a href="https://github.com/huggingface/chug">GitHub - huggingface/chug: Minimal sharded dataset loaders, decoders, and utils for multi-modal document, image, and text datasets.</a>: Minimal sharded dataset loaders, decoders, and utils for multi-modal document, image, and text datasets. - huggingface/chug
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1225025576503345152)** (150 messages🔥🔥): 

- **Language Model Pruning Explorations**: A member is experimenting with creating a pruned **Jamba model** (25% of the size). They are using a custom layer pruning script, discussing their methodology, and mentioning a related [research paper on layer-pruning](https://arxiv.org/abs/2403.17887) that examines strategies for reducing layer count without significantly impacting performance.

- **Dynamic Compute Allocation for LLMs**: Members discuss a [Google paper](https://arxiv.org/abs/2404.02258v1) that suggests transformers can learn to allocate compute dynamically across a sequence. The conversation revolves around its potential for more efficient pretraining and inference, comparing it to speculative decoding methods and discussing the implications for retraining models.

- **Discussions on Speculative Decoding**: The technique of speculative decoding was explained and scrutinized, with a participant highlighting differences from Google's dynamic compute approach. Members conversed about memory management in GPUs and batching for speeding up responses.

- **Cohere's Command R+ Model Introduced**: Command R+, a new model by **Cohere** optimized for Retrieval Augmented Generation (RAG), was shared and briefly discussed. It's designed for scaling LLMs in business applications, providing features like multilingual support and advanced citations.

- **Neural Reasoning Exploration**: The discord users engaged in a conversation about the [neurallambda project on GitHub](https://github.com/neurallambda/neurallambda), which attempts to integrate lambda calculus with transformer-based LLMs. This neurosymbolic approach could be groundbreaking for AI reasoning.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://txt.cohere.com/command-r-plus-microsoft-azure/">Introducing Command R+: A Scalable LLM Built for Business</a>: Command R+ is a state-of-the-art RAG-optimized model designed to tackle enterprise-grade workloads, and is available first on Microsoft Azure   Today, we’re introducing Command R+, our most powerful, ...</li><li><a href="https://arxiv.org/abs/2404.02684">Cross-Architecture Transfer Learning for Linear-Cost Inference Transformers</a>: Recently, multiple architectures has been proposed to improve the efficiency of the Transformer Language Models through changing the design of the self-attention block to have a linear-cost inference ...</li><li><a href="https://lupantech.github.io/inter-gps/">Inter-GPS</a>: no description found</li><li><a href="https://arxiv.org/abs/2404.02258v1">Mixture-of-Depths: Dynamically allocating compute in transformer-based language models</a>: Transformer-based language models spread FLOPs uniformly across input sequences. In this work we demonstrate that transformers can instead learn to dynamically allocate FLOPs (or compute) to specific ...</li><li><a href="https://arxiv.org/abs/2403.17887">The Unreasonable Ineffectiveness of the Deeper Layers</a>: We empirically study a simple layer-pruning strategy for popular families of open-weight pretrained LLMs, finding minimal degradation of performance on different question-answering benchmarks until af...</li><li><a href="https://arxiv.org/abs/2404.02893">ChatGLM-Math: Improving Math Problem-Solving in Large Language Models with a Self-Critique Pipeline</a>: Large language models (LLMs) have shown excellent mastering of human language, but still struggle in real-world applications that require mathematical problem-solving. While many strategies and datase...</li><li><a href="https://huggingface.co/danielus/MermaidSolar-Q4_K_S-GGUF">danielus/MermaidSolar-Q4_K_S-GGUF · Hugging Face</a>: no description found</li><li><a href="https://arxiv.org/abs/2404.02078">Advancing LLM Reasoning Generalists with Preference Trees</a>: We introduce Eurus, a suite of large language models (LLMs) optimized for reasoning. Finetuned from Mistral-7B and CodeLlama-70B, Eurus models achieve state-of-the-art results among open-source models...</li><li><a href="https://arxiv.org/html/2404.02258v1">Mixture-of-Depths: Dynamically allocating compute in transformer-based language models</a>: no description found</li><li><a href="https://huggingface.co/learn">Hugging Face - Learn</a>: no description found</li><li><a href="https://course.fast.ai/">Practical Deep Learning for Coders - Practical Deep Learning</a>: A free course designed for people with some coding experience, who want to learn how to apply deep learning and machine learning to practical problems.</li><li><a href="https://huggingface.co/CohereForAI/c4ai-command-r-plus">CohereForAI/c4ai-command-r-plus · Hugging Face</a>: no description found</li><li><a href="https://github.com/neurallambda/neurallambda">GitHub - neurallambda/neurallambda: Reasoning Computers. Lambda Calculus, Fully Differentiable. Also Neural Stacks, Queues, Arrays, Lists, Trees, and Latches.</a>: Reasoning Computers. Lambda Calculus, Fully Differentiable. Also Neural Stacks, Queues, Arrays, Lists, Trees, and Latches. - neurallambda/neurallambda</li><li><a href="https://huggingface.co/datasets/glaiveai/glaive-code-assistant-v3">glaiveai/glaive-code-assistant-v3 · Datasets at Hugging Face</a>: no description found</li><li><a href="https://github.com/e-p-armstrong/augmentoolkit">GitHub - e-p-armstrong/augmentoolkit: Convert Compute And Books Into Instruct-Tuning Datasets</a>: Convert Compute And Books Into Instruct-Tuning Datasets - e-p-armstrong/augmentoolkit</li><li><a href="https://www.reddit.com/r/Oobabooga/s/ApIzWEdZu7">Reddit - Dive into anything</a>: no description found</li><li><a href="https://huggingface.co/TroyDoesAI/MermaidMistral">TroyDoesAI/MermaidMistral · Hugging Face</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi">But what is a neural network? | Chapter 1, Deep learning</a>: What are the neurons, why are there layers, and what is the math underlying it?Help fund future projects: https://www.patreon.com/3blue1brownWritten/interact...</li><li><a href="https://www.youtube.com/watch?v=wjZofJX0v4M&t=430s&pp=ygULM2JsdWUxYnJvd24%3D">But what is a GPT?  Visual intro to Transformers | Chapter 5, Deep Learning</a>: An introduction to transformers and their prerequisitesEarly view of the next chapter for patrons: https://3b1b.co/early-attentionSpecial thanks to these sup...</li><li><a href="https://www.youtube.com/watch?v=fNk_zzaMoSs&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab">Vectors | Chapter 1, Essence of linear algebra</a>: Beginning the linear algebra series with the basics.Help fund future projects: https://www.patreon.com/3blue1brownAn equally valuable form of support is to s...
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1225008200940781670)** (58 messages🔥🔥): 

- **BitNet Discussion**: The divide by "scale" in **BitNet-1.58** was debated, with users questioning its necessity and expressing that it could potentially hinder the benefits of ternary encoding. However, it was pointed out that **maintaining FP16** for training and scaling outputs could be beneficial for numeric stability.
- **Eurus Model Appeals to Curiosity**: **Eurus-7b-kto**, an LLM by OpenBMB optimized for reasoning, was tested with its fine-tuning datasets **[UltraInteract_sft](https://huggingface.co/datasets/openbmb/UltraInteract_sft)** and **[UltraInteract_pair](https://huggingface.co/datasets/openbmb/UltraInteract_pair)**, with a suggestion to apply SOLAR to this model for potential improvements.
- **Function Calling in Repositories**: Discrepancies in implementation were reported in the Hermes-Function-Calling repository, with *issues regarding function calling and coding standards*. The usage of **langchain's convert_to_openai_tool()** was specifically cited within the issue.
- **QLoRA Gaining Traction**: **QLoRA**, a recent LLM fine-tuning approach, got mentioned as potentially more efficient than LoRA, offering similar performance improvements with **half the VRAM** requirements.
- **Genstruct for Instruction Generation**: The utility and diversity of **Genstruct 7B**, an instruction-generation model from **NousResearch**, were discussed briefly, emphasizing its potential to create diverse instruction formats for fine-tuning datasets based on raw text corpuses.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/openbmb/Eurus-7b-kto">openbmb/Eurus-7b-kto · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/NousResearch/Genstruct-7B">NousResearch/Genstruct-7B · Hugging Face</a>: no description found</li><li><a href="https://github.com/NousResearch/Hermes-Function-Calling/issues/14">This Repo needs some refactoring for the function calling to work properly · Issue #14 · NousResearch/Hermes-Function-Calling</a>: Guys i think there is some issue with the way things are implemented currently in this repo biggest of which is regarding coding standard currently you guys use convert_to_openai_tool from langchai...</li><li><a href="https://arxiv.org/abs/2401.03462">Soaring from 4K to 400K: Extending LLM&#39;s Context with Activation Beacon</a>: The utilization of long contexts poses a big challenge for LLMs due to their limited context window size. Although the context window can be extended through fine-tuning, it will result in a considera...
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[project-obsidian](https://discord.com/channels/1053877538025386074/1156472202619781140/1225294779420508240)** (2 messages): 

- **ChatML Fixed for LLava**: A member shared the successful resolution of issues with **ChatML** for the "llava" project. There is no further explanation or details on what the issues were or how they were resolved. 
- **Possible Fixes for Hermes-Vision-Alpha**: The same member expressed their intention to work on resolving issues with **Hermes-Vision-Alpha**. Details on the nature of these issues or specific fixes were not provided.
  

---


**Nous Research AI ▷ #[bittensor-finetune-subnet](https://discord.com/channels/1053877538025386074/1213221029359657021/1225323698710642708)** (2 messages): 

- **Finetuning Miner Error**: A member encountered an error while running the `miner.py` script in the **finetuning-subnet** repository. Assistance was offered pointing to potential **missing dependencies** as the issue.
  

---


**Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1225083842071630014)** (34 messages🔥): 

- **Glaive's Data Generation Contribution**: [Glaive](https://huggingface.co/datasets/glaiveai/rag_sample) has created a sample dataset to assist in data generation for RAG applications, showcasing the ability to integrate multiple documents into responses.
- **RAG Grounding Clarified**: Sahilch explains that **grounded** mode in RAG distinguishes when the model should use context from a document exclusively and when to blend its own knowledge with documents, adding granularity to the response generation process.
- **Commands for RAG and Citation Markup**: Interninja discusses the importance of proper citation markup, suggesting a JSON format for citations may be beneficial, and shares an [XML instruction format for the new CommandR+ with RAG](https://x.com/LangChainAI/status/1775917799065653250?s=20) which includes complex multi-step querying and uses `<co: doc>` tags for referencing documents.
- **Cohere's RAG Documentation**: Bjoernp highlights the potential of RAG combined with function calling, shares a [Cohere RAG documentation link](https://docs.cohere.com/docs/retrieval-augmented-generation-rag), and debates the implications for synthetic data generation within their Acceptable Use Policy.
- **Filtering Retrievals in RAG Applications**: Iriden promotes the idea of adding a filtering step between retrieval and response in RAG, which has practical success especially when users interact with the selection process for more refined results.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/LangChainAI/status/1775917799065653250?s=20">Tweet from LangChain (@LangChainAI)</a>: Adaptive RAG w/ Cohere&#39;s new Command-R+  Adaptive-RAG (@SoyeongJeong97 et al) is a recent paper that combines (1) query analysis and (2) iterative answer construction to seamlessly handle queries ...</li><li><a href="https://docs.cohere.com/docs/retrieval-augmented-generation-rag">Retrieval Augmented Generation (RAG) - Cohere Docs</a>: no description found</li><li><a href="https://docs.cohere.com/docs/c4ai-acceptable-use-policy">C4AI Acceptable Use Policy</a>: no description found</li><li><a href="https://docs.google.com/document/d/1o8asa0hD0qK5mKkdY5riUeGm-bxKzL02--1r3MgPgdM/edit">RAG/Long Context Reasoning Dataset</a>: no description found</li><li><a href="https://huggingface.co/datasets/glaiveai/rag_sample">glaiveai/rag_sample · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1225008774318788659)** (96 messages🔥🔥): 

- **Copy-Pasting Quirks**: Members discuss copy-pasting difficulties on Desktop compared to Mobile, noting that each character is wrapped in a `<span>` on the website, making it challenging. One member mentions creating a python program to generate the corresponding HTML code to "paste" but it initially broke the website.
- **WorldSim Slowdown Concerns and Solutions**: A discussion highlights concerns about the website slowing down during prolonged use, particularly on mobile. Solutions suggested include reloading from a save, while the best performance is noted to come from the original WorldSim, despite lacking quality-of-life features found in variants.
- **Sharing WorldSim System Prompts**: The system prompt for WorldSim is shared and clarified to be publicly available through a Twitter post, and an easier-to-copy version is posted on [Pastebin](https://pastebin.com/Gj7CpdSE).
- **WorldSim Commands Compilation**: A link to an updated WorldSim Command Index is shared, containing [advanced commands](https://worldsim.notion.site/WorldSim-Command-Index-961c8849f61e4558949716b1dfd5f9fa?pvs=4) for user reference, and prompting a discussion about a "sublimate" command for dismissing persona entities.
- **Jailbreaking Claude Models and Puzzling Over ASCII Art**: Users engage in trying to bypass preset prompts using Claude models, with successful results reported on labs.perplexity.ai. Another user enquiry about an ASCII art of a woman's face generated by WorldSim leads to the revelation that it represents the Nous girl logo.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/karan4d/status/1768836844207378463?s=20">Tweet from mephisto 🤡7 (@karan4d)</a>: im opensourcing worldsim of course i am  worldsim sysprompt and conversation to intitialize:  sysprompt:  &lt;sys&gt;Assistant is in a CLI mood today. The human is interfacing with the simulator direc...</li><li><a href="https://worldsim.notion.site/WorldSim-Command-Index-961c8849f61e4558949716b1dfd5f9fa?pvs=4">Notion – The all-in-one workspace for your notes, tasks, wikis, and databases.</a>: A new tool that blends your everyday work apps into one. It's the all-in-one workspace for you and your team</li><li><a href="https://worldsim.notion.site/WorldSim-Command-Index-961c8849f61e4558949716b1dfd5f9fa?pvs=4)">Notion – The all-in-one workspace for your notes, tasks, wikis, and databases.</a>: A new tool that blends your everyday work apps into one. It's the all-in-one workspace for you and your team</li><li><a href="https://tenor.com/view/friends-ross-geller-david-schwimmer-tv-series-american-sitcom-gif-17315839">Friends Ross Geller GIF - Friends Ross Geller David Schwimmer - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/feel-me-think-about-it-meme-gif-7715402">Feel Me Think About It GIF - Feel Me Think About It Meme - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/oscars-standing-ovation-clap-clapping-applause-gif-5089552">Standing Ovation GIF - Oscars Standing Ovation Clap - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://pastebin.com/0fjwccgM">WorldSim Superhero Universe Expansion Command Set - Pastebin.com</a>: Pastebin.com is the number one paste tool since 2002. Pastebin is a website where you can store text online for a set period of time.</li><li><a href="https://pastebin.com/aLLKvkqq">WorldSim Narrative Crafting Expansion Command Set - Pastebin.com</a>: Pastebin.com is the number one paste tool since 2002. Pastebin is a website where you can store text online for a set period of time.</li><li><a href="https://pastebin.com/Gj7CpdSE">Karan4D&#039;s WorldSim System Prompt Open Source - Pastebin.com</a>: Pastebin.com is the number one paste tool since 2002. Pastebin is a website where you can store text online for a set period of time.
</li>
</ul>

</div>
  

---



**Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1225213583210713170)** (16 messages🔥): 

- **GitHub Workflow Example Shared**: A user provided an example of a GitHub workflow related to *modular auth* and *Mojo packaging*. The initial [link shared](https://github.com/Moosems/Fastq_Parser/blob/main/.github/workflows/package.yml) turned out to be inaccessible but was followed up by a copy-pasted snippet of the workflow.

- **In Search of Debuggers and Editors**: A member inquired about the availability of a debugger and LSP for editors beyond VSCode, specifically mentioning *neovim*.

- **Discord Solution Link Offered**: In response to a problem a user was experiencing, another member directed them to a solution posted previously in a Discord message, but the link to the solution was incomplete.

- **Community Livestream Notification**: A user pointed out a lack of notification for an upcoming "Modular Community Livestream." The [livestream link](https://www.youtube.com/watch?v=PL71FV2KKHE) was provided, discussing "New in MAX 24.2".

- **Request for Mojo Completion Roadmap**: A post originating from the Mojo channel was shared to the general channel requesting a detailed roadmap to "completion" for the Mojo project and a comparison with Taichi or Triton. Another user addressed this by sharing a [link to the Mojo development roadmap](https://docs.modular.com/mojo/roadmap).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.modular.com/mojo/roadmap">Mojo🔥 roadmap &amp; sharp edges | Modular Docs</a>: A summary of our Mojo plans, including upcoming features and things we need to fix.</li><li><a href="https://www.youtube.com/watch?v=PL71FV2KKHE">Modular Community Livestream - New in MAX 24.2</a>: MAX 24.2 is now available! Join us on our upcoming livestream as we discuss everything new in MAX - open sourcing Mojo standard library, MAX Engine support f...
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/1225111389652123720)** (4 messages): 

- **Tweet Alert**: Modular shared a [tweet](https://twitter.com/Modular/status/1775549728400572660) on Twitter.
- **Modular's Twitter Update**: Another [tweet](https://twitter.com/Modular/status/1775583583530524987) was posted from Modular's official Twitter account.
- **Tweet Sharing Session**: Check out this recent Modular [tweet](https://twitter.com/Modular/status/1775926484869541894) for the latest insights.
- **Another Tweet on the Radar**: Modular continues its Twitter streak with this [post](https://twitter.com/Modular/status/1775946487186555225).
  

---


**Modular (Mojo 🔥) ▷ #[ai](https://discord.com/channels/1087530497313357884/1103420074372644916/1225139017121529927)** (4 messages): 

- **Integration of Mojo with ROS 2 Proposed**: A member suggested integrating [Mojo](https://github.com/) with [ROS 2](https://github.com/ros2), believing that Mojo's memory safety practices could mitigate bugs in ROS 2. They highlighted the Rust support via [ros2-rust](https://github.com/ros2-rust/ros2_rust) and mentioned that ROS 2 is adopting a new middleware, [zenoh-plugin-ros2dds](https://github.com/eclipse-zenoh/zenoh-plugin-ros2dds), also written in Rust.

- **The ROS 2 Community's Predominant Use of Python Over Rust**: It was pointed out that most of the ROS 2 community comes from research backgrounds favoring Python and doesn't usually utilize Rust. The contribution reflects the community's overall programming preference within robotics and AI-related projects.

- **Python's Limitations in Robotics Lead to C++ Transition**: The same member shared their experience with ROS, noting that although Python is convenient for initial development in robotics, it's often too slow, leading to a rewrite of systems in C++ for serious applications.

- **Opportunities for Mojo on Nvidia Jetson Hardware**: The member noted the potential for Mojo to leverage [Nvidia Jetson hardware](https://developer.nvidia.com/embedded/jetson-modules), which is increasingly used in robotics and whose performance is limited by Python's Global Interpreter Lock (GIL).

**Link mentioned**: <a href="https://github.com/ros2-rust/ros2_rust">GitHub - ros2-rust/ros2_rust: Rust bindings for ROS 2</a>: Rust bindings for ROS 2 . Contribute to ros2-rust/ros2_rust development by creating an account on GitHub.

  

---


**Modular (Mojo 🔥) ▷ #[tech-news](https://discord.com/channels/1087530497313357884/1151359062789857280/1225083116880662580)** (3 messages): 

- **Automated Docker Builds Fix Incoming**: A fix is announced for version 24.3, addressing issues with **automated docker builds**.

- **Community Cheers for Docker Fixes**: The announcement about the fix for automated docker builds in version 24.3 has been met with positive reactions from the community.

- **Modular Auth Example Shared**: A member provided a link to an example of **modular authentication** on GitHub, which can be seen [here](https://github.com/Moosems/Fastq_Parser/blob/main/.github/workflows/package.yml).
  

---


**Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1225150299598098484)** (277 messages🔥🔥): 

- **Exploring Conditional Conformance**: Members discussed how to implement [conditional conformance](https://github.com/modularml/mojo/blob/main/stdlib/docs/style-guide.md) in Mojo, using `trait` and `struct` syntax with ideas influenced by Swift and Rust. Proposed solutions included using `@conditional` annotations to indicate optional trait implementation within structures.
- **External Call String Issues**: One user encountered difficulties passing a string argument to `external_call` because a StringLiteral is compile-time and immutable. It was suggested to use a C-style null-terminated char pointer extracted from a Mojo string, similar to an example shared in the chat.
- **Mojo Program Running on Android**: A user showcased Mojo running on Android, specifically on a Snapdragon 685 CPU. This was met with interest and questions about CPU details and a request for the output of `lscpu`.
- **Merchandise Possibilities**: A question was raised regarding the future availability of Mojo-themed merchandise, invoking responses from team members to explore the idea. Plush toys and phone cases featuring the Mojo mascot were mentioned as potential items.
- **Error Handling Discussion**: Error handling possibilities in Mojo were speculated by users, discussing hypothetical syntax for error handling and polymorphic error resolution similar to traditional `try-except` blocks in Python.

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.modular.com/mojo/stdlib/utils/variant">variant | Modular Docs</a>: Defines a Variant type.</li><li><a href="https://gist.github.com/lsh/f47fb85015d4197522d9c614e2a0f7de">A `Bytes` type that can be an owned `List` or used with a `Buffer`</a>: A `Bytes` type that can be an owned `List` or used with a `Buffer` - bytes_ref_or_owned.mojo</li><li><a href="https://gist.github.com/modularbot/0613c95485ee838e00dc7289b81efa2c">playground.mojo</a>: GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://gist.github.com/modularbot/a6c43d73ec9532fb8a7fcf258f3c02ab">playground.mojo</a>: GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://github.com/modularml/mojo/blob/main/stdlib/src/utils/variant.mojo">mojo/stdlib/src/utils/variant.mojo at main · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/modularml/mojo/issues/2152">[BUG] using inout for decorator incorrectly causes weird compiler error · Issue #2152 · modularml/mojo</a>: Bug description MRE: # Correct implementation should be &quot;fn decorator(borrowed func: fn() -&gt; None) -&gt; fn() escaping -&gt; None:&quot; fn decorator(inout func: fn() -&gt; None) -&gt; fn() es...</li><li><a href="https://github.com/modularml/mojo/issues/2144">[BUG] Tests failing on latest nightly branch · Issue #2144 · modularml/mojo</a>: Bug description I took the latest fetch from the upstream/nightly branch and ran the tests as I wanted to pick up 1 issue but 2 tests are failing on the branch This is the output: Successfully crea...
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[community-projects](https://discord.com/channels/1087530497313357884/1151418679578337311/1225176876079911101)** (3 messages): 

- **Logger Library Updated**: The **logger library** has received an update that now allows logging messages with arbitrary arguments and keyword arguments. Examples provided show how to log information, warnings, errors, and fatal messages with the improved function calls.

- **Introducing BlazeSeq**: `BlazeSeq🔥` has been published, a complete rewrite of `MojoFastTrim`, acting as a feature-complete FASTQ parser that matches the test suites of BioJava and Biopython; it is available for CLI use or as a foundation for future applications. Benchmarks and usage examples are available on [GitHub](https://github.com/MoSafi2/BlazeSeq).

- **Buffered Line Iterator for Improved File Handling**: A new implementation includes a buffered line iterator, akin to Rust's buffer_redux crate, capable of handling incomplete lines and larger-than-buffer lines from either file or in-memory sources. This iterator is touted as a robust solution for projects until such functionality is integrated into the standard library.

**Link mentioned**: <a href="https://github.com/MoSafi2/BlazeSeq">GitHub - MoSafi2/BlazeSeq</a>: Contribute to MoSafi2/BlazeSeq development by creating an account on GitHub.

  

---


**Modular (Mojo 🔥) ▷ #[performance-and-benchmarks](https://discord.com/channels/1087530497313357884/1151418895417233429/1225086228442124418)** (11 messages🔥): 

- **New Level Unlocked!**: A user was congratulated for advancing to level 3 in the ModularBot system, indicating engagement and contribution within the community.
- **Library Performance Gains**: A user reported significant performance improvements using a library, dropping execution time to 10m35s from a prior faster benchmark of 96s achieved with Golang.
- **Link to Helpful Shell Script**: The creator of the discussed library shared a [Medium post](https://mzaks.medium.com/poor-persons-package-management-in-mojo-8671aa6e420a) describing a shell script for easy installation of the library.
- **Library Optimization Tips**: It was suggested to set a capacity when instantiating a dictionary to reduce reallocations and rehashing, potentially optimizing performance further.
- **Sorting Algorithm Still to be Updated**: There was a mention of a specialized sorting algorithm for strings that could offer better performance, located at [mzaks/mojo-sort](https://github.com/mzaks/mojo-sort/blob/main/multi_key_quicksort/sort.mojo), but it hasn't been updated for the new versions of Mojo.
  

---


**Modular (Mojo 🔥) ▷ #[📰︱newsletter](https://discord.com/channels/1087530497313357884/1157711273111339088/1225099690790355036)** (2 messages): 

- **Max⚡ and Mojo🔥 24.2 Officially Released**: Modular announced the release of **Max⚡ and Mojo🔥 24.2**, along with the open-sourcing of their standard library and the launch of nightly builds. The update has seen community engagement with roughly 50 pull requests opened and 10 merged; contributors are encouraged to explore and ask questions on [Discord](https://modul.ar/discord).
- **Jump into the Mojo🔥 Open Source Movement**: A new blog post titled [*The Next Big Step in Mojo🔥 Open Source*](https://www.modular.com/blog/the-next-big-step-in-mojo-open-source) details the latest advancements in the Mojo🔥 open source initiative.
- **Discover What's New in Mojo🔥 24.2**: The release of Mojo🔥 24.2 brings enhanced Python interoperability, among other features, as outlined in the [Mojo launch blog](https://www.modular.com/blog/max-24-2-is-here-whats-new) and the follow-up article on [*What’s new in Mojo 24.2*](https://www.modular.com/blog/whats-new-in-mojo-24-2-mojo-nightly-enhanced-python-interop-oss-stdlib-and-more).
- **Exploring Higher Order Functions in Mojo🔥**: Readers are invited to find out about Higher Order Functions in Mojo🔥, with a teaser link provided on [Twitter](https://twitter.com/Modular/status/1). However, the link appears to be incomplete.

**Link mentioned**: <a href="https://www.modular.com/newsletters/modverse-weekly-issue-28">Modverse Weekly - Issue 28</a>: Welcome to issue 28 of the Modverse Newsletter covering Featured Stories, the Max Platform, Mojo, &amp; Community Activity.

  

---


**Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1225054381028675634)** (17 messages🔥): 

- **Resolution for Parsing Errors in Standard Library**: A member working off of the **nightly** branch reported parsing errors in the stdlib, yet mentioned being able to build the stdlib without issues. Concern was raised as to whether this should be a cause for alarm.
- **FileCheck Troubles in WSL Solved**: Running tests resulted in `FileCheck command not found` errors for one member, but thanks to community assistance and the use of `dpkg -S llvm | grep FileCheck`, the issue was resolved by finding the correct directory (`/usr/lib/llvm-14/bin`) and adding it to the path.
- **Unsupported Tests Not a Concern**: After troubleshooting `FileCheck` installation, the member reported 7 unsupported tests, which was confirmed by another member as fine since those tests are platform-specific.
- **Optimizing Mojo's Optional Value Method**: There was a discussion about the possibility for Mojo's `Optional` to return a Reference instead of a copy for the `value()` method, referencing the [current implementation](https://github.com/modularml/mojo/blob/nightly/stdlib/src/collections/optional.mojo) and suggesting the improvement could be actionable.
- **Approachability of Reference Issues for New Contributors**: While considering making returning a reference from `Optional` a 'good first issue', members agreed that dealing with references might not be user-friendly for new contributors unfamiliar with lifetimes, as proper inference requires experienced plumbing through the function parameters.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/modularml/mojo/blob/nightly/stdlib/src/collections/optional.mojo#L117-L118).">mojo/stdlib/src/collections/optional.mojo at nightly · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/modularml/mojo/blob/nightly/stdlib/src/collections/optional.mojo#L106.">mojo/stdlib/src/collections/optional.mojo at nightly · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



**LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1225003195835617280)** (171 messages🔥🔥): 

- **Understanding LLM Multitasking and Scaling**: A detailed discussion reveals that multitasking with a single LLM may lead to reduced performance due to shared resources like VRAM and RAM. It was suggested that better performance might be achieved by running separate models concurrently on different servers and distributing requests via a queuing system.
- **Pondering Local vs. Cloud LLM Usage**: Participants debated the merits of running local LLMs versus cloud-based solutions such as GPT-4. Some prefer local models for their uncensored output and the ability to leverage powerful hardware without cloud restrictions.
- **Model Suggestions for AI Enthusiasts**: Various users recommended specific models for coding and general use, highlighting *Hermes-2-Pro-Mistral-7B Q8* and *Goliath 120B Longlora Q3KS*, among others. Users discussed how VRAM and system specs influence the performance and suitability of different LLMs.
- **Technical Issues and Solutions Explored**: Members navigated common errors and provided solutions involving GPU offloading settings and C Redistributable installation. Discussions clarified that LM Studio cannot execute web searches and that it is necessary to have the latest drivers for efficient GPU utilization.
- **Feature Updates and Community Engagement**: LM Studio's upcoming support for text embeddings was announced, while individuals inquired about running multiple GPUs, interacting with documents via AnythingLLM, and creating Discord bots with contextual awareness.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard">LMSys Chatbot Arena Leaderboard - a Hugging Face Space by lmsys</a>: no description found</li><li><a href="https://useanything.com/">AnythingLLM | The ultimate AI business intelligence tool</a>: AnythingLLM is the ultimate enterprise-ready business intelligence tool made for your organization. With unlimited control for your LLM, multi-user support, internal and external facing tooling, and 1...
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1225119288382197871)** (39 messages🔥): 

- **System Prompt Training Can Be Auto-activated**: A concept was discussed about training a smaller LLM with outputs generated from a larger model using a complex System Prompt. This would effectively embed the System Prompt into the smaller model, negating the need to use context space for it, although the process could be costly in time and money.
- **Conundrum in Model Responses**: An issue was raised about a model providing odd, task-oriented responses irrelevant to input queries. It suggests confusion in preset behaviors that could be linked to the model's training.
- **Mixtral vs Mistral Clarification**: Differentiations were made between Mistral and Mixtral models; Mixtral is a Mixture of Experts (MOE) model combining eight 7B models into an equivalent 56B parameter model, whereas Mistral is a standard 7B model.
- **Large Model, Tiny Hardware**: There was a discussion about running a Mixtral 8x7b model on a 24GB VRAM NVIDIA 3090 GPU, noting that while feasible, it operates at reduced speeds. Also, LM Studio can not run on a Raspberry Pi, but smaller models like tinyllama might be compiled for operation on such devices.
- **New Model and Support Development Discussions**: Links were shared for the new 104B parameter C4AI Command R+ model with advanced capabilities and the newly available Eurus-7b model. There were also discussions indicating that llamacpp requires updates to support some of these newer models.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/openbmb/Eurus-7b-kto">openbmb/Eurus-7b-kto · Hugging Face</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1bvniaz/command_r_cohere_for_ai_104b/ky12kw5/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://huggingface.co/CohereForAI/c4ai-command-r-plus">CohereForAI/c4ai-command-r-plus · Hugging Face</a>: no description found</li><li><a href="https://plainenglish.io/community/direct-preference-optimization-dpo-a-simplified-approach-to-fine-tuning-large-language-models">Direct Preference Optimization (DPO): A Simplified Approach to Fine-tuning Large Language Models</a>: no description found</li><li><a href="https://huggingface.co/datasets/christopherthompson81/quant_exploration">christopherthompson81/quant_exploration · Datasets at Hugging Face</a>: no description found</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/6491/files>">Add Command R Plus support by Carolinabanana · Pull Request #6491 · ggerganov/llama.cpp</a>: Updated tensor mapping to add Command R Plus support for GGUF conversion.
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1225000508557754408)** (8 messages🔥): 

- **Embedding Models Not Supported, Until Now**: A member asked about using embedding models with LM Studio and mentioned downloading a gguf embedding model. It was clarified that embedding models were previously unsupported, but text embedding support has been introduced in version 0.2.19, with a beta available [here](https://discord.com/channels/1110598183144399058/1166577236325965844/1225221755937886208).

- **LM Studio (Linux) Update Notification Issues**: A user reported that LM Studio for Linux does not notify them of updates, observing that despite running version 0.2.17, version 0.2.18 is available and a beta for 0.2.19 exists.

- **Linux In-App Update Mechanism Still Pending**: In response to the update notification issue, it was highlighted that the lack of an in-app update mechanism for Linux is one of the reasons why the platform is still considered "beta."

- **Enthusiasm for Linux Development**: Members showed enthusiasm for the development of LM Studio on Linux, including the possibility of having a .deb package.
  

---


**LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1225045530199982172)** (23 messages🔥): 

- **Switching to ROCm Yields Speed Boost**: Utilizing the ROCm preview resulted in a remarkable boost from 13 to 65 tokens/second on AMD hardware, showing that AMD's system can vastly outperform expectations with the correct software interface.

- **GPU Market Fluctuations Noted**: Recent price increases in **GP100 GPUs** were observed, with costs rising from around $350 to $650-$700, signaling volatile market trends.

- **TSMC Disruption May Impact Prices**: A Bloomberg article about a major earthquake leading to the evacuation of TSMC production lines suggests potential price increases for GPUs and Macs.

- **CUDA vs. ROCM vs. OpenCL Performance Layers**: It's estimated that NVIDIA CUDA is approximately twice as fast as ROCm, which in turn is estimated to be about five times faster than OpenCL or DirectML.

- **Mixed GPU Configurations for Inference**: While combining NVIDIA and AMD GPUs in one configuration isn't possible due to software incompatibilities, running separate instances of LM Studio to utilize each card individually for different inference tasks is a viable solution.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.bloomberg.com/news/articles/2024-04-03/tsmc-evacuates-production-lines-after-major-taiwan-quake">Bloomberg - Are you a robot?</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/186phti/m1m2m3_increase_vram_allocation_with_sudo_sysctl/">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1225221755937886208)** (19 messages🔥): 

- **Introducing LM Studio 0.2.19 Preview 1 with Embeddings**: LM Studio version 0.2.19 Preview 1 now supports **local embedding models**, such as `nomic-embed-text-v1.5-GGUF` via its OpenAI-like `POST /v1/embeddings` endpoint and LLama.cpp updates. [Windows](https://releases.lmstudio.ai/windows/0.2.19/beta/LM-Studio-0.2.19-Setup-Preview-1.exe), [Linux](https://releases.lmstudio.ai/linux/0.2.19/beta/LM_Studio-0.2.19-Preview-1.AppImage), and [Mac](https://releases.lmstudio.ai/mac/arm64/0.2.19/beta/LM-Studio-darwin-arm64-0.2.19-Preview-1.zip) preview builds are available for download.
  
- **Separate ROCm Version for Compatibility**: There will be a separate version for those needing **ROCm support**; it is not included in the current build.

- **Beta Version Confusion Clarified**: The version displayed in LM Studio beta builds reflects the current shipping version, not the beta iteration, with version bumps occurring only at full release for clarity's sake.

- **No Support for GPU over IP Yet**: LM Studio does not currently support using multiple GPUs across different machines, known as **GPU over IP**.

- **Chat Feature with Documents Still Pending**: The ability to "Chat with your documents" is not yet implemented in LM Studio, but using LM Studio server mode with anythingLLM is suggested as an alternative.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/tree/main">nomic-ai/nomic-embed-text-v1.5-GGUF at main</a>: no description found</li><li><a href="https://lmstudio.ai/docs/welcome">Welcome | LM Studio</a>: LM Studio is a desktop application for running local LLMs on your computer.</li><li><a href="https://blog.nomic.ai/posts/nomic-embed-text-v1">Introducing Nomic Embed: A Truly Open Embedding Model</a>: Nomic releases a 8192 Sequence Length Text Embedder that outperforms OpenAI text-embedding-ada-002 and text-embedding-v3-small.</li><li><a href="https://huggingface.co/nomic-ai/nomic-embed-text-v1">nomic-ai/nomic-embed-text-v1 · Hugging Face</a>: no description found</li><li><a href="https://releases.lmstudio.ai/windows/0.2.19/beta/LM-Studio-0.2.19-Setup-Preview-1.exe">no title found</a>: no description found</li><li><a href="https://releases.lmstudio.ai/linux/0.2.19/beta/LM_Studio-0.2.19-Preview-1.AppImage">no title found</a>: no description found
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/1225532162145517568)** (1 messages): 

- **Autogen Studio Outputs Truncated**: A member reported receiving only 1 or 2 tokens in their inference results when using LM Studio with Autogen Studio, seeking a solution for obtaining the full completion response.
  

---


**LM Studio ▷ #[langchain](https://discord.com/channels/1110598183144399058/1167546793656062063/1225408139508322384)** (1 messages): 

- **Question on "Memory" Retention in Runtime**: A member asked how to achieve "memory" retention within the same runtime, having managed to make it work only with file analysis. There is a gap in understanding of how to maintain state across interactions with the bot.
  

---


**LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1225450306993786910)** (9 messages🔥): 

- **LM Studio Activated on AMD GPU**: A member elaborated on their experience getting **LM Studio** to operate on an **AMD GPU** in a system that includes both RTX 2060 and 7900 XTX GPUs.
- **ROCm vs. OpenCL Performance Inquiry**: One participant enquired about the speed difference between **ROCm and OpenCL**, mentioning their own unsuccessful attempts to load models on a **6700XT** GPU despite configuration efforts.
- **System Specs for ROCm Build Shared**: A member contributed their system specifications, revealing the use of an **AmdROCm** GPU type and noting 15.94 GB of RAM with 11.86 GB VRAM unused on a Windows 10 platform.
- **Driver Issue Blocks ROCm on Lower Series AMD GPUs**: It was mentioned that AMD's driver issues prevent the **ROCm build from functioning on 6700 series or lower GPUs**, indicating that resolution depends on AMD's intervention.
- **ROCm Performance Exceeds OpenCL on Alternative Platforms**: A member detailed their positive experience with a ROCm fork of **KoboldAI**, observing a **significant performance boost** to 33T/s over 12T/s when compared to **LMStudio + OpenCL**.
  

---


**LM Studio ▷ #[crew-ai](https://discord.com/channels/1110598183144399058/1197374792668545034/1225180539972227167)** (24 messages🔥): 

- **CORS Might Help**: A member suggested turning on **CORS** (Cross-Origin Resource Sharing) to potentially solve an issue, without specifying the exact problem being addressed.
- **LM Studio Implementation Resource**: It was recommended to check out an article for implementing LM Studio in CrewAI, titled "Implementing LM Studio in CrewAI" available at [Medium by Tayyib Ali](https://medium.com/@tayyibali4300/implementing-lm-studio-in-crewai-270cc577acee).
- **CrewAI Logging Levels and Display Issues**: A member discussed the logging features of CrewAI mentioning that **verbose** can be set to **1 or 2** for different levels of logging details, and showed concern when no logs appeared at the expected location in LM Studio.
- **Troubleshooting Missing LM Studio Logs**: During a troubleshooting conversation about missing logs in LM Studio, a member noted that they were not seeing any processing in LM Studio but confirmed that CrewAI was functioning correctly on its end.
- **JSONDecodeError in CrewAI**: A member encountered a "**json.decoder.JSONDecodeError**" when using CrewAI and sought assistance; the error indicates a problem with a JSON string not being properly terminated.
  

---



**Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1225015134947250258)** (194 messages🔥🔥): 

- **Transformers Course Goes Public**: Stanford CS 25 seminar on Transformers is open to the public for auditing live or via recorded sessions. Interesting topics like LLM architectures and applications across various fields will be discussed, with lectures by prominent industry experts. [Join on Zoom](https://stanford.zoom.us/j/99922151759?pwd=dW5CcUtVYkNybGZGY0hMWUZtVkZBZz09), check the [course website](https://web.stanford.edu/class/cs25/), or watch past sessions on [YouTube](https://www.youtube.com/playlist?list=PLoROMvodv4rNiJRchCzutFw5ItR_Z27CM).

- **Interview Tips from the Trenches**: Experienced engineers recommend focusing on high-level skills and confidence in work over coding tests for senior roles. For assessing basic Python skills, some even use a simple coding exercise to ensure candidates don't overly rely on tools like ChatGPT.

- **Mathical Difficulty**: A member sought help understanding a mathematical problem from a non-public working paper, leading to a discussion on binary search on sets and the importance of defining variables within academic papers.

- **Secure the Launch, Stanford!**: A course on Transformers at Stanford, featuring guest researchers and covering deep learning models, is now available to the public through Zoom, with a corresponding [Discord server](https://discord.gg/2vE7gbsjzA) opened for wider community discussion.

- **Go Play**: Members exchange usernames and links to play the game, Go. Options include Online Go Server (OGS) for correspondence matches and a custom version available at [Infinite Go](https://infinite-go.com).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://blog.eleuther.ai/nyt-yi-34b-response/">Yi-34B, Llama 2, and common practices in LLM training: a fact check of the New York Times</a>: Setting the record straight regarding Yi-34B and Llama 2.</li><li><a href="https://infinite-go.com">Infinite Go</a>: no description found</li><li><a href="https://x.com/DanHendrycks/status/1769452537302929682?s=20">Tweet from Dan Hendrycks (@DanHendrycks)</a>: https://x.ai/blog/grok-os Grok-1 is open sourced.  Releasing Grok-1 increases LLMs&#39; diffusion rate through society. Democratizing access helps us work through the technology&#39;s implications mor...</li><li><a href="https://www.regulations.gov/comment/NTIA-2023-0009-0246">Regulations.gov</a>: no description found</li><li><a href="https://www.regulations.gov/document/NTIA-2023-0009-0001/comment">Regulations.gov</a>: no description found</li><li><a href="https://github.com/EleutherAI/the-pile/issues/75">Legal Contracts · Issue #75 · EleutherAI/the-pile</a>: Here are legal contracts collected from the Securities and Exchange Commission. https://drive.google.com/file/d/1of37X0hAhECQ3BN_004D8gm6V88tgZaB/view?usp=sharing It&#39;s about ~38 GB raw and full of...</li><li><a href="https://stanford.zoom.us/j/99922151759?pwd=dW5CcUtVYkNybGZGY0hMWUZtVkZBZz09).">Join our Cloud HD Video Meeting</a>: Zoom is the leader in modern enterprise video communications, with an easy, reliable cloud platform for video and audio conferencing, chat, and webinars across mobile, desktop, and room systems. Zoom ...</li><li><a href="https://www.youtube.com/watch?v=XfpMkf4rD6E&ab_channel=StanfordOnline)">Stanford CS25: V2 I Introduction to Transformers w/ Andrej Karpathy</a>: January 10, 2023Introduction to TransformersAndrej Karpathy: https://karpathy.ai/Since their introduction in 2017, transformers have revolutionized Natural L...</li><li><a href="https://discord.gg/2vE7gbsjzA)">Discord | Your Place to Talk and Hang Out</a>: Discord is the easiest way to talk over voice, video, and text. Talk, chat, hang out, and stay close with your friends and communities.
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1225089176609882203)** (51 messages🔥): 

- **Innovations in Efficient Model Architectures**: A new approach called T-GATE suggests cross-attention in text-to-image diffusion models may be unnecessary after understanding the coarse semantics of an image, potentially speeding up the process ([T-GATE on GitHub](https://github.com/HaozheLiu-ST/T-GATE)). However, samples provided haven't fully convinced the community of its effectiveness.
  
- **Hardware Optimization Breakthrough or Bust?**: References to potential hardware improvements like Free-pipeline Fast Inner Product (FFIP) algorithm claim significant efficiency gains, casting half the multiplications for cheap additions ([Journal Publication](https://arxiv.org/abs/2311.12224)). The community is skeptical, pondering whether there's a catch to these seemingly too good to be true claims.

- **Dynamic Allocation of FLOPs in Transformers**: An [arXiv paper](https://arxiv.org/abs/2404.02258) introduces a method for transformers to allocate compute dynamically across a sequence, potentially optimizing performance and allowing for a pre-defined compute budget. This approach diverges from uniform FLOP distribution, proposing a more selective and potentially efficient allocation of resources.

- **Discussions on Large Language Models**: Conversation about Huge Scale Language Models (HLB-GPT) explores follow-up to Mixture of Experts (MoE) work and specific design choices. A thread ([HLB-GPT MoE and MoD Thread](https://discord.com/channels/729741769192767510/1169741769232089169/1225497424869724180)) has been dedicated for detailed exchange without cluttering the main channel.

- **Contentious Data Crawling Practices**: Discussion surfaced on the challenges and potential violations associated with scraping platforms like Discord. While theoretically feasible, it breaches Terms of Service and can result in account bans.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2404.02905">Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction</a>: We present Visual AutoRegressive modeling (VAR), a new generation paradigm that redefines the autoregressive learning on images as coarse-to-fine &#34;next-scale prediction&#34; or &#34;next-resolutio...</li><li><a href="https://arxiv.org/abs/2404.02258">Mixture-of-Depths: Dynamically allocating compute in transformer-based language models</a>: Transformer-based language models spread FLOPs uniformly across input sequences. In this work we demonstrate that transformers can instead learn to dynamically allocate FLOPs (or compute) to specific ...</li><li><a href="https://arxiv.org/abs/2404.01475">Are large language models superhuman chemists?</a>: Large language models (LLMs) have gained widespread interest due to their ability to process human language and perform tasks on which they have not been explicitly trained. This is relevant for the c...</li><li><a href="https://x.com/cem__anil/status/1775282571070591220?s=20">Tweet from Cem Anil (@cem__anil)</a>: One of our most crisp findings was that in-context learning usually follows simple power laws as a function of number of demonstrations.  We were surprised we didn’t find this stated explicitly in the...</li><li><a href="https://arxiv.org/abs/2311.12224">Fast Inner-Product Algorithms and Architectures for Deep Neural Network Accelerators</a>: We introduce a new algorithm called the Free-pipeline Fast Inner Product (FFIP) and its hardware architecture that improve an under-explored fast inner-product algorithm (FIP) proposed by Winograd in ...</li><li><a href="https://www.youtube.com/watch?v=rJIwO31uv5c">Louis Castricato - RLAIF, User Autonomy, and Controllability (Eleuther / Synthlabs)</a>: Talk from the Open-Source Generative AI Workshop at Cornell Tech. Website: https://www.louiscastricato.com/Slides: https://drive.google.com/file/d/14Qldg0E1c...</li><li><a href="https://github.com/HaozheLiu-ST/T-GATE/">GitHub - HaozheLiu-ST/T-GATE: T-GATE: Cross-Attention Makes Inference Cumbersome in Text-to-Image Diffusion Models</a>: T-GATE: Cross-Attention Makes Inference Cumbersome in Text-to-Image Diffusion Models - HaozheLiu-ST/T-GATE</li><li><a href="https://github.com/trevorpogue/algebraic-nnhw">GitHub - trevorpogue/algebraic-nnhw: AI acceleration using matrix multiplication with half the multiplications</a>: AI acceleration using matrix multiplication with half the multiplications - trevorpogue/algebraic-nnhw
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1225093201749086392)** (7 messages): 

- **Countdown to MATS Stream Applications**: The deadline to apply for Neel Nanda's MATS stream is approaching in less than 10 days. Interested applicants can find details and the FAQ in the provided [Google Docs link](https://docs.google.com/document/d/1p-ggQV3vVWIQuCccXEl1fD0thJOgXimlbBpGk6FI32I/edit#heading=h.y0ohi6l5z9qn).

- **Attention to Neural Networks**: A Github repository named **atp_star** which provides a PyTorch and NNsight implementation of AtP* has been shared, courtesy of a DeepMind paper by Kramar et al., 2024. The repository can be found at [koayon/atp_star on GitHub](https://github.com/koayon/atp_star).

- **Saprmarks Tweets**: A member shared a link to a [Twitter post by @saprmarks](https://twitter.com/saprmarks/status/1775513423402692685), though the content was not discussed in the provided messages.

- **Gratitude for Sharing Code**: The query about an open-source implementation for the latest AtP* paper was resolved with thanks, following the provision of the GitHub repository link.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.google.com/document/d/1p-ggQV3vVWIQuCccXEl1fD0thJOgXimlbBpGk6FI32I/edit#heading=h.y0ohi6l5z9qn">Neel Nanda MATS Stream -  Admissions Procedure + FAQ</a>: no description found</li><li><a href="https://github.com/koayon/atp_star">GitHub - koayon/atp_star: PyTorch and NNsight implementation of AtP* (Kramar et al 2024, DeepMind)</a>: PyTorch and NNsight implementation of AtP* (Kramar et al 2024, DeepMind) - koayon/atp_star
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1225065760783859712)** (17 messages🔥): 

- **CUDA Error Troubleshooting**: A member faced a **RuntimeError with CUDA** while running an older version of the LM eval harness on H100s, which worked on A100s, pointing to a potential issue with `flash attention`. Several suggestions noted including upgrading to **CUDA 11.8** might help, but the actual culprit was identified as `apex`. Isolated tests with `.contiguous()` function and motion towards single GPU resolved the issue.

- **`top_p` Unrecognized Argument in Colab**: Another member encountered an unrecognized argument error when trying to set **`top_p=1`** in an LM eval harness command in **Google Colab**. The suggestion pointed out that the issue might be due to spaces in the arguments list.

**Link mentioned**: <a href="https://colab.research.google.com/drive/1pDByKcCu3vQzy58iz8uSmUm806LQtG8v#scrollTo=mTSKBJlVjaB-">Google Colaboratory</a>: no description found

  

---


**Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1225121548147490837)** (3 messages): 

- **Fault-Tolerant and Elastic Job Launch in PyTorch**: A user shared a link to the PyTorch documentation for setting up **fault-tolerant and elastic jobs**, detailing the commands needed to launch them. The process involves specific settings for nodes, trainers per node, maximum restarts, and rendezvous endpoints, as shown in [PyTorch's elastic training quickstart guide](https://pytorch.org/docs/stable/elastic/quickstart.html).

- **Cloud Support for Advanced Training Schemes**: Another member mentioned that cloud services like **AWS** and **Azure** support advanced job training schemes, with AWS having released something called **Gemini** in the previous year.

**Link mentioned**: <a href="https://pytorch.org/docs/stable/elastic/quickstart.html">Quickstart &mdash; PyTorch 2.2 documentation</a>: no description found

  

---



**LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1225016603159302245)** (158 messages🔥🔥): 

- **Socratic Tutors and Constitutional AI**: A package called [ConstitutionalAiTuning](https://github.com/steffen74/ConstitutionalAiTuning) was mentioned that allows fine-tuning of LLMs into Socratic tutors adhering to one's ethical principles. It requires a JSON file with principles and uses those to construct improved answers for fine-tuning models, meant to ease the process for those with less technical expertise.
- **JAX Type Promotion and Semantics Clarified**: Discussion on [JAX type promotion semantics](https://jax.readthedocs.io/en/latest/type_promotion.html) revolved around how types are promoted during operations in JAX. Code snippets illustrated the behavior, like `np.int16(1) + jnp.int16(2) + 3` resulting in `int16` while `3 + np.int16(1) + jnp.int16(2)` results in `int32`.
- **SD3 Model Input Configuration Debated**: There was an extensive technical discussion on the setup of text input for models like SD3, suggesting alternative approaches to concatenating sequences and the potential benefits of extending T5 tokens during fine-tuning while limiting the use of CLIP.
- **Legal Risks with AI and Copyright Infringement**: A conversation highlighted the legal risks associated with using copyrighted material to train AI systems, referring to the Suno music AI platform and possible legal repercussions from recording labels.
- **GPU Infrastructure Costs and Stability AI's Challenged Finances**: Reported financial challenges for Stability AI were discussed, including their struggle with high infrastructure costs from cloud services and potential inability to cover these expenses, as per an [exposé by Forbes](https://www.forbes.com/sites/kenrickcai/2024/03/29/how-stability-ais-founder-tanked-his-billion-dollar-startup/?sh=2e53d2e3e630).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2404.01292">Measuring Style Similarity in Diffusion Models</a>: Generative models are now widely used by graphic designers and artists. Prior works have shown that these models remember and often replicate content from their training data during generation. Hence ...</li><li><a href="https://www.weco.ai/blog/technical-report">Introducing Weco AIDE</a>: Your AI Agent for Machine Learning</li><li><a href="https://www.972mag.com/lavender-ai-israeli-army-gaza/">‘Lavender’: The AI machine directing Israel’s bombing spree in Gaza</a>: The Israeli army has marked tens of thousands of Gazans as suspects for assassination, using an AI targeting system with little human oversight and a permissive policy for casualties, +972 and Local C...</li><li><a href="https://www.theregister.com/2024/04/03/stability_ai_bills/">Stability AI reportedly ran out of cash to pay its AWS bills</a>: Generative AI darling was on track to pay $99M on compute to generate just $11M in revenues</li><li><a href="https://tenor.com/8a9w.gif">Ian Malcolm GIF - Ian Malcolm Jurassic - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.youtube.com/watch?v=_D3GACF-Bsk">Galileo</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=kJirMpbvBrM">Editing DALL·E Images in ChatGPT</a>: You can now edit DALL·E images in ChatGPT across web, iOS, and Android.</li><li><a href="https://www.musicbusinessworldwide.com/suno-is-a-music-ai-company-aiming-to-generate-120-billion-per-year-newton-rex/">Suno is a music AI company aiming to generate $120 billion per year. But is it trained on copyrighted recordings? &#x2d; Music Business Worldwide</a>: Ed Newton&#x2d;Rex discovers that Suno produces music with a striking resemblance to classic copyrights&#8230;</li><li><a href="https://www.youtube.com/watch?v=5pidokakU4I">Axis of Awesome - 4 Four Chord Song (with song titles)</a>: Australian comedy group &#39;Axis Of Awesome&#39; perform a sketch from the 2009 Melbourne International Comedy Festival. Footage courtesy of Network Ten Australia. ...</li><li><a href="https://github.com/steffen74/ConstitutionalAiTuning/">GitHub - steffen74/ConstitutionalAiTuning: A Python library for fine-tuning LLMs with self-defined ethical or contextual alignment, leveraging constitutional AI principles as proposed by Anthropic. Streamlines the process of prompt generation, model interaction, and fine-tuning for more responsible AI development.</a>: A Python library for fine-tuning LLMs with self-defined ethical or contextual alignment, leveraging constitutional AI principles as proposed by Anthropic. Streamlines the process of prompt generati...
</li>
</ul>

</div>
  

---


**LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1225099630627258398)** (10 messages🔥): 

- **Scaling Latent Diffusion Models (LDMs)**: An [arXiv paper](https://arxiv.org/abs/2404.01367) detailed the study of the sampling efficiency scaling properties of LDMs. The study found that smaller models often outperform larger ones under the same inference budget.
- **Moderation GIF Shared**: A member posted a [moderation-related GIF](https://tenor.com/view/discord-mod-moderation-ban-mod-ban-gif-9351874248631360646) from Tenor.com, possibly indicating the action taken against off-topic or inappropriate messages.
- **Banter About Quick Cash**: Users joked about missing out on learning how to make "$50k in 72 hours" due to a message moderation, with guesses and meme references about drug smuggling.
- **Tease of a New Optimizer**: Drhead shared a [Twitter post](https://twitter.com/aaron_defazio/status/1775521495298588956) hinting at the imminent release of a new optimizer.
- **Visual AutoRegressive (VAR) Model Outperforms**: An [arXiv paper](https://arxiv.org/abs/2404.02905) introduced VAR, a new image autoregressive modeling paradigm that has shown to outperform diffusion transformers in image generation on multiple dimensions, including quality and speed.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2404.02905">Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction</a>: We present Visual AutoRegressive modeling (VAR), a new generation paradigm that redefines the autoregressive learning on images as coarse-to-fine &#34;next-scale prediction&#34; or &#34;next-resolutio...</li><li><a href="https://arxiv.org/abs/2404.01367">Bigger is not Always Better: Scaling Properties of Latent Diffusion Models</a>: We study the scaling properties of latent diffusion models (LDMs) with an emphasis on their sampling efficiency. While improved network architecture and inference algorithms have shown to effectively ...</li><li><a href="https://tenor.com/view/discord-mod-moderation-ban-mod-ban-gif-9351874248631360646">Discord Mod Moderation Ban GIF - Discord mod Moderation ban Mod ban - Discover &amp; Share GIFs</a>: Click to view the GIF
</li>
</ul>

</div>
  

---



**OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1225055189980020776)** (67 messages🔥🔥): 

- **Exploring Diverse AI Datasets**: A member listed several datasets they have, indicating sizes ranging from 106G for "pairs" to 994k for "PheMT," with several datasets involving translations from EU languages. Some datasets, such as "pairs" and "WikiMatrix," were noted as less reliable, requiring metrics and cutoff points for quality assessment.

- **Rapid Feedback for RP-LLMs**: A new service by [Chaiverse](https://console.chaiverse.com/) allows for fast feedback on RP-LLM models, providing model evaluation within 15 minutes. It aims to provide the fastest and most accurate feedback using human preferences, avoiding training to the test due to non-public evaluation datasets.

- **Unveiling SaladCloud for AI/ML Workloads**: SaladCloud promises to help developers avoid high cloud costs and GPU shortages by offering a fully-managed container service that opens up access to thousands of consumer GPUs, with rates starting at $00/hr and built for inference at scale.

- **Adding Heads to Transformer Models Made Easier**: The [GitHub repository for transformer-heads](https://github.com/center-for-humans-and-machines/transformer-heads) was shared, offering tools for attaching, training, saving, and loading new heads for transformer models, which could be quite beneficial for those looking to extend the capabilities of LLMs.

- **CohereForAI's Massive Model, C4AI Command R+**: Creators release a model called [C4AI Command R+](https://huggingface.co/CohereForAI/c4ai-command-r-plus), which is a 104 billion parameter multilingual model with abilities like Retrieval Augmented Generation (RAG) and multi-step tool use for complex tasks. The cost of running such large models remains a concern for some members.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/CohereForAI/c4ai-command-r-plus">CohereForAI/c4ai-command-r-plus · Hugging Face</a>: no description found</li><li><a href="https://bit.ly/3TFIsKt">Salad - GPU Cloud | 10k+ GPUs for Generative AI</a>: Save up to 90% on your cloud bills. Deploy AI/ML production models easily. 600% more images &amp; 10x more inferences per dollar. Try SaladCloud for free today.</li><li><a href="https://github.com/center-for-humans-and-machines/transformer-heads">GitHub - center-for-humans-and-machines/transformer-heads: Toolkit for attaching, training, saving and loading of new heads for transformer models</a>: Toolkit for attaching, training, saving and loading of new heads for transformer models - center-for-humans-and-machines/transformer-heads</li><li><a href="https://github.com/OpenNLPLab/LASP/tree/main">GitHub - OpenNLPLab/LASP: Linear Attention Sequence Parallelism (LASP)</a>: Linear Attention Sequence Parallelism (LASP). Contribute to OpenNLPLab/LASP development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1225159571052695584)** (4 messages): 

- **GitHub Bug Squashed**: A fix was applied to address an issue, with a commit pushed to GitHub, visible at [GitHub Commit 5760099](https://github.com/OpenAccess-AI-Collective/axolotl/commit/5760099bd4605e4c4fb444890bd473cb200c5f1a).

- **Table of Contents Mismatch Alert**: A discrepancy was observed in the **README's Table of Contents**, which does not match its markdown headings, indicating a need for cleanup.

- **Comparative Analysis for Clarity**: A suggestion was made to view the current TOC and markdown headings side by side for better visibility of inconsistencies.

- **Incorrect Heading in Training Config**: An issue was identified with the heading used in `config/train`, noting it was incorrect and suggesting a potential correction.

**Link mentioned**: <a href="https://github.com/OpenAccess-AI-Collective/axolotl/commit/5760099bd4605e4c4fb444890bd473cb200c5f1a">fix toc · OpenAccess-AI-Collective/axolotl@5760099</a>: no description found

  

---


**OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1225164644633542818)** (11 messages🔥): 

- **Looking for High-Resolution Images**: A member inquired about where to find a large collection of **4K and 8K images** to crawl, but no sources or suggestions were provided in the following discussion.
- **Deployment with UI Feedback Needed**: Someone asked for recommendations on a **good UI for deploying models and obtaining expert feedback**, though no suggestions were made on the thread.
- **Exploring Non-Instruction Text Data for Training**: A member discussed using **non-instructional text data** like podcast transcripts for training a model to generate text in the style of the training data, referencing *MistralAI* and asking if others are doing similar experiments.
- **Order of Fine-Tuning Practices**: In a strategy discussion, there was a consensus that one should train **'completion'** before 'instructions' while finetuning, which is especially useful for increasing domain-specific knowledge in models.
- **Fine-Tuning Techniques and Efficiency**: There was an exchange about fine-tuning techniques, where members noted that sometimes **simple fine-tuning (SFT)** and prompt engineering can be more effective than continual pre-training (CPT) for domain-specific training. It was mentioned that quality and diversity in instructional samples, even in smaller amounts, often yield better performance than larger quantities of lower-quality data.
  

---


**OpenAccess AI Collective (axolotl) ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/1225511933306736722)** (2 messages): 

- **Optimal Dataset for Mistral 7B**: A member inquired about the recommended dataset for training a **Mistral 7B model** using axolotl on Ubuntu 22.04. Another member suggested the **OpenOrca dataset** for its utility in all-around use.
  

---


**OpenAccess AI Collective (axolotl) ▷ #[announcements](https://discord.com/channels/1104757954588196865/1113462842436354149/1225531833970856049)** (1 messages): 

- **New Discord Bot Integration Live!**: A new Discord bot integration has been set up to directly answer questions from the OpenAccess AI Collective. Members are encouraged to test the bot and leave feedback in a designated channel.
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-help-bot](https://discord.com/channels/1104757954588196865/1225300056442409040/1225300453051469834)** (62 messages🔥🔥): 

- **Fine-tuning Qwen2 with Qlora**: A detailed answer elucidated steps for fine-tuning **Qwen2** using **Qlora**, such as setting `base_model` and `adapter` in the configuration file, using 4-bit precision, and specifying optimizer settings. An example configuration file was provided to assist in the process.
- **Dataset Streaming in Axolotl**: Axolotl supports the use of **local datasets for streaming**, contrary to a previous documentation misunderstanding that implied otherwise. The steps include configuring `pretraining_dataset` with the Hugging Face dataset path in the `.yml` file.
- **Multi-Node Fine-Tuning with Docker**: Guidelines were presented for **multi-node fine-tuning** using Docker, such as setting up **accelerate** config, configuring FSDP settings on the model, and ensuring all machines share the same Axolotl commit and model configuration file.
- **Issues with Checkpoints and Mixed Precision**: A member encountered a `ValueError` when trying to flatten tensors with different data types during the use of **Qlora** with **FSDP** on the **Mixtral** model. The solution involves ensuring uniform data types for tensors before operations.
- **Axolotl Bot Goes Offline**: The **Axolotl bot** experienced downtime, causing members to express their discontent through humorous replies. No solution or reason for the outage was provided in the chat history.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/patrickpain-patricksomuchpain-patrickfleas-spongebobpain-spongebobsomuchpain-gif-18151897">Patrickpain Patricksomuchpain GIF - Patrickpain Patricksomuchpain Patrickfleas - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://phorm.ai/query?projectId=43cf5c5b-941f-461a-9055-b02788a2e364&threadId=a2fc4740-1a5c-4766-8cbb-7769186bae94)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.</li><li><a href="https://phorm.ai/query?projectId=43cf5c5b-941f-461a-9055-b02788a2e364&threadId=f8d0cb5a-e9cd-4dcf-a16f-39197690a56b)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.</li><li><a href="https://github.com/openaccess-ai-collective/axolotl#dataset)">GitHub - OpenAccess-AI-Collective/axolotl: Go ahead and axolotl questions</a>: Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.</li><li><a href="https://phorm.ai/query?projectId=43cf5c5b-941f-461a-9055-b02788a2e364&threadId=8b13862e-c141-4ebd-973a-e8f61032dce3)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.</li><li><a href="https://phorm.ai/query?projectId=43cf5c5b-941f-461a-9055-b02788a2e364&threadId=1608c74f-8ed6-4f25-8861-c69c9ff61737)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.</li><li><a href="https://phorm.ai/query?projectId=43cf5c5b-941f-461a-9055-b02788a2e364&threadId=7db2702b-b0e3-424e-af79-012c04808de0)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.</li><li><a href="https://phorm.ai/query?projectId=43cf5c5b-941f-461a-9055-b02788a2e364&threadId=d8e13d9b-7b9a-45e1-8c8d-ebad9a63158a)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.</li><li><a href="https://phorm.ai/query?projectId=43cf5c5b-941f-461a-9055-b02788a2e364&threadId=46b832c9-3b42-4a74-9886-711b4821502f)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.</li><li><a href="https://phorm.ai/query?projectId=43cf5c5b-941f-461a-9055-b02788a2e364&threadId=a31dec35-31c9-4260-bc7f-1d79610360aa)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.
</li>
</ul>

</div>
  

---



**LlamaIndex ▷ #[announcements](https://discord.com/channels/1059199217496772688/1073670729054294197/)** (1 messages): 

jerryjliu0: webinar is in 15 mins! ^^
  

---


**LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1225096488317620346)** (6 messages): 

- **Revolutionize Your Knowledge Management**: The new **LLM-powered, self-organizing digital library** is more than just a chat system; it's an AI-powered tool designed for professionals and teams to create, organize, and annotate their data. Discover it [here](https://t.co/nbvRS0Cc9Q).

- **Advanced RAG Meetup in Tokyo**: Join the evening of lightning talks on **4/18 from 7-9pm JST** in Tokyo, featuring speakers @hexpode, Diego, and Sudev discussing RAG applications, hosted by Rakuten. Details and signup can be found [here](https://t.co/ovCozxNaTt).

- **Deploy LLM Apps Globally with Ease**: Koyeb's interface conveniently scales LLM applications by connecting your GitHub repo to deploy serverless apps globally with zero infrastructure setup. Check out Koyeb [here](https://t.co/weFs0waN4o).

- **Tailoring RAG to Question Complexity**: The "Adaptive RAG" paper by @SoyeongJeong97 explores tailored RAG techniques for varying complexities of questions, addressing the speed and specificity trade-offs. Learn more [here](https://t.co/SZQppddC95).

- **Culinary Coding with the LlamaIndex + MistralAI Cookbook**: Explore the series of cookbooks that guide users through building RAG, agentic RAG, and agent-based systems with **MistralAI**, including routing and query decomposition. Get your recipes [here](https://t.co/7KCqujf9sd).

**Link mentioned**: <a href="https://t.co/nbvRS0Cc9Q">IKI AI &#x2013; Intelligent Knowledge Interface</a>: Smart library and&#x2028; Knowledge Assistant for professionals and teams.

  

---


**LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1225055713068585071)** (112 messages🔥🔥): 

- **Exploring GraphIndex Limitations**: A member expressed confusion about the lack of pipeline support when working with **knowledgegraphs** in **llama_index**, stating that there is no clear documentation on creating a `graphindex` from a `graphdb` or the role of `docstore`. They noted that while vectorindex has a pipeline and docstore for re-indexing nodes, **graphindex** seems to require custom code.
- **Seeking Recursive Query Engine Docs**: A member couldn't find documentation on **ragas** with recursive query engine, leading to a conversation about potential issues between **langchain** and **ragas** and difficulties in importing functions from **ragas.metrics**.
- **Querying Existing OpenSearch Index**: A new member to llama-index enquired about querying an existing **OpenSearch index**. They provided detailed steps on setting up clients and stores but were uncertain about the process, later discovering the `VectorStoreIndex.from_vector_store` method on their own.
- **In Search of LlamaIndex Agent Examples**: Participants discussed various aspects of creating **llama_index agents**, including the complexities of generating in-depth responses, issues with persisting nodes taking an unexpectedly long time, and the proper use of ReAct agents. 
- **Handling Issues in LlamaIndex Implementations**: Members sought advice on a range of **llama_index** implementation topics including the possibility of semantic similarity match in metadata, integrating SQL databases for chatbot functionality, and facing errors with async operations using **elastic search** as vector db storage.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://llamahub.ai/l/tools/llama-index-tools-bing-search?from=">no title found</a>: no description found</li><li><a href="https://llamahub.ai/?tab=llama_datasets">Llama Hub</a>: no description found</li><li><a href="https://www.llamaindex.ai/blog/introducing-llama-datasets-aadb9994ad9e">Introducing Llama Datasets 🦙📝 — LlamaIndex, Data Framework for LLM Applications</a>: LlamaIndex is a simple, flexible data framework for connecting custom data sources to large language models (LLMs).</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/llama_dataset/uploading_llama_dataset/?h=dataset">Contributing a LlamaDataset To LlamaHub - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/querying/node_postprocessors/node_postprocessors/?h=similarity#similaritypostprocessor">Node Postprocessor Modules - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/index_structs/struct_indices/SQLIndexDemo/">Text-to-SQL Guide (Query Engine + Retriever) - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/api_reference/evaluation/dataset_generation/?h=from_documents#llama_index.core.evaluation.DatasetGenerator.from_documents">Dataset generation - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/api_reference/readers/simple_directory_reader/?h=simpledirector#llama_index.core.readers.file.base.SimpleDirectoryReader">Simple directory reader - LlamaIndex</a>: no description found</li><li><a href="https://youtu.be/yGejxO1xYmo?si=22UtE4T0RVXbqYOy">Workflows &amp; Tooling to Create Trusted AI | Ask More of AI with Clara Shih</a>: Clara sits down with the founder/CEOs of three of the hottest AI companies-- Aravind Srinivas (Perplexity AI), Jerry Liu (LlamaIndex), and Harrison Chase (La...</li><li><a href="https://github.com/run-llama/llama_index/blob/f03db8da9301e2a1f2a1783338464bec7e7a859e/llama-index-legacy/llama_index/legacy/agent/react/prompts.py#L27">llama_index/llama-index-legacy/llama_index/legacy/agent/react/prompts.py at f03db8da9301e2a1f2a1783338464bec7e7a859e · run-llama/llama_index</a>: LlamaIndex is a data framework for your LLM applications - run-llama/llama_index</li><li><a href="https://github.com/run-llama/llama_index/issues/905#issuecomment-1484288684">Where do I define top_k documents to be returned by similarity search over vectorstore? · Issue #905 · run-llama/llama_index</a>: When calling query function, how do I specify how many ks do I want the retriever to pass to a LLM? Or do I need to specify it before calling query function? llm_predictor = LLMPredictor(llm=ChatOp...</li><li><a href="https://github.com/run-llama/llama-hub/">GitHub - run-llama/llama-hub: A library of data loaders for LLMs made by the community -- to be used with LlamaIndex and/or LangChain</a>: A library of data loaders for LLMs made by the community -- to be used with LlamaIndex and/or LangChain - run-llama/llama-hub
</li>
</ul>

</div>
  

---


**LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1225126740515623075)** (6 messages): 

- **Spellcheck the AI Way**: A member shared a snippet of Node.js code utilizing the LlamaIndex `Ollama` package to correct spelling errors in user-submitted text using a model named ‘mistral’. They indicated the service can run locally and handle errors, as demonstrated by the script correcting "bkie" to "bike" despite the ironic misspelling of "misspelled" in the prompt.
- **Local AI without Third-Party Services**: The same user confirmed that the `Ollama` package acts as a client/wrapper around a locally running AI server, suggesting the use of the command `ollama run mistral` for local operation over `localhost:11434`.
- **Acknowledging the Benevolent AI**: Humor was found in an AI's forgiving nature as a member humorously acknowledged the AI's utility after a self-reported misspelling incident in their own code example, praising the AI's ability to understand and process the intended input correctly.
- **Enhancing Imagery with Reading and Asking (RAG)**: Discussion emerged around the potential of using Reading and Asking Generative (RAG) techniques for image processing tasks, with practical applications like overcoming CAPTCHAs or maintaining continuity in visual storytelling like comic strips.
  

---



**HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1225114862640959720)** (3 messages): 

- **Customize Your Repo Visibility**: Enterprises using HuggingFace can now set a default **Repo visibility** to public, private, or private-by-default. More details can be found on this [Twitter thread](https://twitter.com/julien_c/status/1772688542289822073).
- **Publish with Quarto on HuggingFace**: There's a new publishing option for **Quarto** enabling users to deploy sites on HuggingFace easily. Instructions on publication are provided [here](https://twitter.com/gshotwell/status/1772661727856914720) and [here](https://www.linkedin.com/feed/update/urn:li:activity:7178422723503673345?updateEntityUrn=urn%3Ali%3Afs_feedUpdate%3A%28V2%2Curn%3Ali%3Aactivity%3A7178422723503673345%29).
- **HuggingFace Hub Enterprise Page Launched**: Explore the new **HF Hub Enterprise** page, a place for tailored enterprise solutions. The announcement and details are available [here](https://x.com/victormustar/status/1772742275744850137).
- **Fine-Tune Access Control for Enterprise Repos**: Have more control over your org's repositories with the new fine-grained access control feature. Available details can be found in this [Twitter post](https://twitter.com/Thom_Wolf/status/1770504033452573077).
- **Major TOM Gets Sentinel-1**: The expansion of Major TOM now includes **Sentinel-1** data in the MajorTOM-Core, expanding the horizons for space observation capabilities. Learn more about the release [here](https://x.com/mikonvergence/status/1772912287709331612).
  

---


**HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1225005073986490408)** (48 messages🔥): 

- **Seeking AI for Game Testing**: A member inquired about good machine learning AIs for testing games, implying interest in tools suitable for game development and quality assurance.
- **Generating Succinct Summaries**: One user struggled with the **summarization pipeline** in Hugging Face, noting that `text_length_penalty` seemed ineffective and `max_length` appeared to truncate text. Discussion continued on model output lengths and how to achieve shorter summaries, with suggestions like using `max_new_tokens` and checking model configs or splitting the input samples.
- **Troubleshooting Multi-GPU System Setup**: There was a request for information on the impact of **PCIe slot speeds (x4/x8)** on multi-GPU system performance for local large language models (LLMs).
- **Deploying and Using HuggingFace Models**: Queries were raised about deploying models and using the `predict` function for a model deployment involving **AWS Inferentia Instance**, seeking clarity on the right approach and if this was the correct forum for such questions.
- **Image Generation Model Fine-Tuning Advice**: Someone sought advice on fine-tuning an image generation model to create a painted portrait with a specific style and wondered if including images of paintings would assist in this goal; a suggestion was made to try using the **IP adapter Face ID**.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://t.me/Jttfoxoffcial1">JTT FOX OFFICIAL</a>: You can contact @Jttfoxoffcial1 right away.</li><li><a href="https://huggingface.co/docs/transformers/main/en/generation_strategies#customize-text-generation">Text generation strategies</a>: no description found</li><li><a href="https://github.com/huggingface/cookbook">GitHub - huggingface/cookbook: Open-source AI cookbook</a>: Open-source AI cookbook. Contribute to huggingface/cookbook development by creating an account on GitHub.</li><li><a href="https://github.com/unslothai/unsloth#installation-instructions---conda">GitHub - unslothai/unsloth: 2-5X faster 70% less memory QLoRA &amp; LoRA finetuning</a>: 2-5X faster 70% less memory QLoRA &amp; LoRA finetuning - unslothai/unsloth
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1225412463906787429)** (1 messages): 

- **Balancing Act Between Speed and Smartness**: A member highlighted a **trade-off between latency and reasoning** in production prompts, suggesting that a prompt without reasoning yields fast but poor responses, while adding reasoning leads to smarter but slower replies. They proposed a hack of preemptively reasoning through most likely scenarios while the user is busy typing. [Explore the idea here](https://x.com/siddish_/status/1772345589511901368?s=20).

**Link mentioned**: <a href="https://x.com/siddish_/status/1772345589511901368?s=20">Tweet from Siddish (@siddish_)</a>: stream with out reasoning -&gt; dumb response 🥴 stream till reasoning -&gt; slow response 😴  a small LLM hack:  reason most likely scenarios proactively while user is taking their time

  

---


**HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1225158243765063801)** (5 messages): 

- **Apple Flexes Tech Muscle**: A message mentioned that Apple claimed their latest model is more powerful than **OpenAI's GPT-4**.
- **3blue1brown Remains a Math Video Maven**: A member expressed appreciation for 3blue1brown's continued production of educational videos, especially the earlier series on neural networks.
- **Visual AutoRegressive Modeling Outshines Diffusion Transformers**: A new paper, [Visual AutoRegressive modeling (VAR)](https://arxiv.org/abs/2404.02905), introduces a paradigm shift in autoregressive learning for images, which surpasses diffusion transformers in terms of image generation quality and inference speed.
- **Chain of Thoughts Elevates AI Reasoning Skills**: The paper [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903) discusses significant performance improvements in complex reasoning tasks for large language models when using chain of thought prompting.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2404.02905">Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction</a>: We present Visual AutoRegressive modeling (VAR), a new generation paradigm that redefines the autoregressive learning on images as coarse-to-fine &#34;next-scale prediction&#34; or &#34;next-resolutio...</li><li><a href="https://arxiv.org/abs/2201.11903">Chain-of-Thought Prompting Elicits Reasoning in Large Language Models</a>: We explore how generating a chain of thought -- a series of intermediate reasoning steps -- significantly improves the ability of large language models to perform complex reasoning. In particular, we ...
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1225112064494927942)** (20 messages🔥): 

- **Octopus 2: The Tentacles of Functionality**: A demo for **Octopus 2**, a model with the capability to call functions, has been shared with excitement centered around its on-device potential. [Check out the Space for Octopus 2](https://huggingface.co/spaces/Tonic/Octopus), but expect a long render time of 1500 seconds when trying it out.
- **Music Hurdles Overcome by Local Processing**: Members discussed the benefits of running music models locally rather than over cloud services. They touched on expectations for hardware optimizations, and a [Youtube Demo](https://youtube.com/shorts/Jm2xq2oNJ3E?si=MGkXSq0ZCiGM0gbb) was shared celebrating a successful pipeline experiment.
- **Making Images Come to Life with Salt AI**: Innovative workflows have been released utilizing the new **multi-subject image node pack** from Salt, including body & face region detection and face swapping technology. [Learn more about multi-subject image processing on GitHub](https://github.com/getSaltAi/SaltAI_Multisubject).
- **Sharing AI Impact on TED Stage**: Community involvement and AI advancements were highlighted in a TED talk shared by a community member. The talk can be watched on [YouTube](https://www.youtube.com/watch?v=d8icTgtZeQg&t) expressing gratitude for community support during film production.
- **PyTorch Geometric Welcomes CornellTemporalHyperGraphDataset**: The pull request for the CornellTemporalHyperGraphDataset has been successfully merged into PyTorch Geometric, with immediate access through downloading from `master`. [View the PR here](https://github.com/pyg-team/pytorch_geometric/pull/9090) and get ready to incorporate it into your workflows.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.producthunt.com/posts/metaforms-ai"> Metaforms AI - OpenAI + Typeform =  AI for feedback, surveys &amp; research | Product Hunt</a>: Metaforms is Typeform&#x27;s AI successor. Build the world’s most powerful Feedback, Surveys and User Research Forms to collect life-changing insights about your users through generativeAI. Trained on...</li><li><a href="https://huggingface.co/spaces/Tonic/Octopus/">Octopus - a Hugging Face Space by Tonic</a>: no description found</li><li><a href="https://telegram.me/int_gem_bot">Int Bot</a>: You can contact @int_gem_bot right away.</li><li><a href="https://github.com/getSaltAi/SaltAI_Multisubject">GitHub - getSaltAi/SaltAI_Multisubject</a>: Contribute to getSaltAi/SaltAI_Multisubject development by creating an account on GitHub.</li><li><a href="https://youtube.com/shorts/Jm2xq2oNJ3E?si=MGkXSq0ZCiGM0gbb">the song that no one wrote #music #newmusic #song #timelapse #photography #musicvideo #viral #art</a>: no description found</li><li><a href="https://github.com/pyg-team/pytorch_geometric/pull/9090">feat: add `CornellTemporalHyperGraphDatasets` by SauravMaheshkar · Pull Request #9090 · pyg-team/pytorch_geometric</a>: Reference: #8501 #7312 Request for Review: @rusty1s @wsad1 This PR aims to add HyperGraph datasets consisting of timestamped simplices where each simplex is a set of nodes. Released with the paper ...
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1225205399297327244)** (5 messages): 

- **RAG Resources for MLE Interview Prep**: A member is seeking in-depth resources for studying **Retrieval-Augmented Generation (RAG)** ahead of a technical interview. They requested recommendations from the community for good study materials.

- **RAG Setup Struggles on WSL Ubuntu**: A newcomer to AI is looking for assistance in setting up **RAG** on **WSL Ubuntu 24.04** with **Llama2** and mentioned difficulties in setting up **privategpt**.

- **Recording Next Presentation for Reference**: A community member is unable to attend the next presentation and sought help in getting it recorded. They stated an intention to place the link in GitHub for future access.

- **Potential OBS Recording Solution**: In response to the recording request, another member indicated that they are considering recording the presentation using **OBS**.
  

---


**HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1225092089998606467)** (8 messages🔥): 

- **Batch Size and Model Performance**: Larger batch sizes are linked to increased model performance, as specific tests have shown improvements, particularly on medical data, although improvements can be marginal or non-significant. However, accumulation beyond 2 batches might negatively impact the performance, potentially due to batch normalization issues.
- **Seeking Deep Learning Companions**: A member mentioned they are looking for a partner to collaborate in the fields of **Deep Learning** and **Natural Language Processing** (NLP).
- **Batch Size Impacts on Learning Dynamics**: Different experiences were shared regarding batch size; one member found that smaller batches worked better for their small model, while another raised the issue that larger batches might skip local minima but smaller batches are more time-consuming.
- **Learning Rate (LR) Schedulers as a Solution**: The use of **LR schedulers** such as cyclic or cosine was suggested to address the issues of local minima encountered when working with larger batch sizes, providing both exploration and exploitation phases during training.
- **Questions About Updating Custom Datasets on HuggingFace**: A member inquired whether manually updating a custom dataset used for fine-tuning a pretrained model on HuggingFace would necessitate re-uploading or if the model would automatically update.
  

---


**HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1225029720949653544)** (21 messages🔥): 

- **GPT-2 Stagnates on Summarization**: A user training GPT-2 for text summarization experiences stagnation in validation metrics. They propose the idea of concise model training examples on the HuggingFace platform for specific tasks to avoid scavenging the internet.

- **Prompt Crafting for LLMs on CPU**: A member seeks an open-source Large Language Model that can extract structured data like product names and prices from HTML, inquiring about suitable prompts. The user specifies a CPU-only setup with 16GB RAM for implementing the model.

- **BERT for Time Series Forecasting?**: There's an interest in fine-tuning BERT for time series forecasting using methods like PEFT. One user provides assistance when asked for code samples or notebooks to guide this process.

- **Context Length Confines in Model Fine-tuning**: A query about whether the context length of the Babbage-002 model can be changed during training was met with an explanation that it's immutable during fine-tuning but modifiable when training from scratch.

- **Enhancing Chatbot Responses with Free Models**: A user creating a chatbot with Google Books API integration seeks a free language model to enhance response quality, ensuring answers are more conversational and complete.
  

---


**HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1225105529903251536)** (11 messages🔥): 

- **Searching for DiT with Cross-Attention**: A member inquired about a **DiT (Diffusion Transformer)** modified to use cross-attention for conditioning on text, image, or other data types. Another mentioned that the **DiT** on HF Diffusers is class-conditioned, linking to the paper ([DiT Paper](https://arxiv.org/html/2312.04557v1)).
- **Cost Considerations in Conditioning Strategies**: A conversation highlighted that the public diffusion models like **DiT** are conditioned by class rather than using cross-attention to keep costs lower. One member suggested that modifications related to *SD3 linear* might be more practical.
- **Customizing SD for Stereo to Depth Map Conversion**: A member expressed the need to convert stereo images into depth maps, finding current models insufficient. They proposed possibly modifying **Stable Diffusion (SD)** for this task.
- **Fine-Tuning Limits of SD with Custom Channels**: A query about fine-tuning **Stable Diffusion** with more than 3 channels led to a suggestion that minor modifications to the **SD architecture** may be necessary as opposed to training from scratch.
- **Alternative Approaches for Depth Estimation**: It was suggested to look into **Dino v2** for depth estimation training and to consider **LoRA** for stereo images, sharing relevant GitHub resources ([Dino v2 GitHub](https://github.com/facebookresearch/dinov2), [Depth Estimation Notebook](https://github.com/facebookresearch/dinov2/blob/main/notebooks/depth_estimation.ipynb)). Another member pointed to work done with **ControlNet**, where 4-channel images were used, linking to a related repository ([SD-Forge-LayerDiffuse GitHub](https://github.com/layerdiffusion/sd-forge-layerdiffuse)).
  

---



**tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1225087752924696637)** (86 messages🔥🔥): 

- **Logo Upgrade and Fish Farewell**: The Discord logo got updated by George Hotz, sparking mixed feelings among members—some enjoyed the professionalism, while others mourned the loss of the quirky fish logo, which still remains on the banner. Discussion ensued about whether to update the banner as well.
  
- **Optimization and Cross-GPU Communication**: Conversation turned towards optimization and sharding for machine learning models, with George Hotz and others discussing the impacts of launch latency on small kernels' performance and the challenge of data transfer between GPUs—cudagraphs, P2P limitations, and potential improvements with the use of NV drivers.
  
- **Tinygrad Performance Ambitions**: Performance measures were shared, showing promising results like **53.4 tok/s on a single 4090 GPU** with BEAM=4, achieving 83% of what gpt-fast can do. George Hotz highlighted ambitions to surpass these results using tinygrad soon.

- **Intel GPU and NPU Kernel Drivers**: Technical details about kernel drivers for Intel's GPUs and NPUs were discussed, noting the various drivers available such as 'gpu/drm/i915', 'gpu/drm/xe', and 'accel/ivpu'. There was an exchange on possible performance and power efficiency gains when leveraging NPUs in conjunction with CPUs.

- **Upholding the Focus on Tinygrad Development**: Amidst the technical discussions, George Hotz reiterated the channel's purpose for tinygrad-related talk, providing a reminder along with a link to the tinygrad GitHub repository and a guide on asking smart questions. This reinforced the goal of maintaining topical discussions within the channel.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/tinygrad/tinygrad/blob/master/docs/env_vars.md">tinygrad/docs/env_vars.md at master · tinygrad/tinygrad</a>: You like pytorch? You like micrograd? You love tinygrad! ❤️  - tinygrad/tinygrad</li><li><a href="https://github.com/tinygrad/tinygrad">GitHub - tinygrad/tinygrad: You like pytorch? You like micrograd? You love tinygrad! ❤️</a>: You like pytorch? You like micrograd? You love tinygrad! ❤️  - GitHub - tinygrad/tinygrad: You like pytorch? You like micrograd? You love tinygrad! ❤️</li><li><a href="http://www.catb.org/~esr/faqs/smart-questions.html">How To Ask Questions The Smart Way</a>: no description found
</li>
</ul>

</div>
  

---


**tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1225058022502568027)** (23 messages🔥): 

- **Tinygrad Tutorials Win Praise**: Users found the quick start guide with Tinygrad straightforward and praised its helpfulness for beginners; it motivated them to delve further into the field of neural networks.
- **JAX Tutorial Highlighted**: A member shared a link to the JAX Autodidax tutorial, offering a deep dive into the workings of JAX's core system with a [hands-on Colab notebook](https://colab.research.google.com/github/google/jax/blob/main/docs/autodidax.ipynb).
- **Tinygrad for Protein Folding Inquiry**: Camelcasecam discussed the possibility of implementing ColabFold or OmegaFold with Tinygrad, questioning the potential performance improvements, while also showing interest in learning how to transfer PyTorch weights into Tinygrad.
- **Collaborative Effort in Biofield Tech**: In the context of adapting OmegaFold with Tinygrad, users from bioscience backgrounds expressed enthusiasm in teaming up for the project, suggesting that collaboration could yield better results.
- **Exploring Performance Debugging with Tinygrad**: Alveoli3358 shared their study notes on interpreting performance outputs when running Tinygrad with DEBUG=2, indicating an interest in calculating total FLOPS/memory required for an MNIST example to estimate theoretical training time.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/mesozoic-egg/tinygrad-notes/blob/main/profiling.md">tinygrad-notes/profiling.md at main · mesozoic-egg/tinygrad-notes</a>: Tutorials on tinygrad. Contribute to mesozoic-egg/tinygrad-notes development by creating an account on GitHub.</li><li><a href="https://jax.readthedocs.io/en/latest/autodidax.html">Autodidax: JAX core from scratch &#8212; JAX  documentation</a>: no description found
</li>
</ul>

</div>
  

---



**OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1225030105441501194)** (83 messages🔥🔥): 

- **JSON Object Support Clarification**: Users confirmed that models supporting 'json_object' response format are notably OpenAI and Fireworks endpoints. They advised checking support by looking at provider parameters on the model's page ([OpenRouter models](https://openrouter.ai/models)).

- **Roleplaying Qualms with Claude 3 Haiku**: The Claude 3 Haiku model received mixed reviews for roleplay, with suggestions to use the self-moderated version and to input several examples (few shot) for better output. However, jailbreak (jb) tweaks are recommended for improved performance.

- **Discord Resources for Jailbreaking Claude**: Users discussed Claude jailbreaks and shared resources including SillyTavern's and Chub's Discord servers, where jailbreak listings and NSFW prompts can be found. The user was directed to easily accessible jailbreaks such as the pancatstack jb and advised on how to obtain NSFW roles.

- **OpenRouter Credit Location and Model Issues**: Members discussed recent changes to OpenRouter's dashboard, including a new location for viewing credits now found at the `/credits` URL. Additionally, concerns were raised about the functionality of certain models like DBRX and Midnight Rose, and their support for specific features.

- **Moderation and Response Issues with OpenRouter API**: Users noted that even the self-moderated version of Claude model has a high decline rate and speculated about additional "safety" prompts. There were reports of non-responsiveness and a mention of implementing better providers to improve service stability for models like Midnight Rose.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://prnt.sc]">no title found</a>: no description found</li><li><a href="https://prnt.sc/_ba2eY63AJNA">Screenshot</a>: Captured with Lightshot</li><li><a href="https://sillytavern.app/">SillyTavern - LLM Frontend for Power Users</a>: no description found
</li>
</ul>

</div>
  

---



**OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1225036657842196580)** (17 messages🔥): 

- **Installation Triumph**: A member expressed satisfaction after successfully installing software on a Windows PC: *Just got it installed and running on my windows PC. Damn*.
- **Termux Troubles**: A snag with `chroma-hnswlib` was discussed; one member noted that despite being reportedly removed, the issue persists in the installation process. They sought advice on handling this problem.
- **Shift to Support Channel**: In response to the above issue, the discussion was directed to another channel, suggesting moving detailed technical support topics to a more appropriate location.
- **Support and Encouragement**: There was an exchange of mutual encouragement and appreciation regarding posting in the community, with a focus on the belief that each issue raised is a valuable learning experience.
- **Multi-Platform Capability Confirmed**: Clarification was provided on the compatibility of certain software, confirming that it works on both PC and Mac, with reference to install instructions and guides in the documentation and pinned messages.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://discord.gg/wNJZsJgQ?event=1221828294811586572">Discord - A New Way to Chat with Friends &amp; Communities</a>: Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.</li><li><a href="https://docs.openinterpreter.com/getting-started/setup">no title found</a>: no description found
</li>
</ul>

</div>
  

---


**OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1225019016196784208)** (55 messages🔥🔥): 

- **Hermes-2-Pro Best Practices**: Members are discussing the use of **Hermes-2-Pro** and the importance of changing system prompts as advised in the model card.
- **Shortcut Woes in the 01 Server**: A user expressed difficulty with the 01 software output being verbose, seeking a local keyboard shortcut like `ctrl+c` in Ollama to interrupt the LLM output without exiting the entire server.
- **Linux Complications with 01 Software**: Users are sharing the troubleshooting and workarounds involved when running the 01 software on various Linux distros. Issues with package dependencies, system messages, and hardware compatibility such as audio (ALSA lib errors) are mentioned.
- **Windows 11 Poetry Issues Identified**: One user reports encountering problems when using `poetry` on Windows 11, noting issues with `CTRL+C` and audio recording.
- **Cardputer Discussion and Development**: Participants discuss the development and potential of using the *M5 Cardputer* for the open-interpreter project, including implementation details and Github repository links for the ongoing work.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/OpenInterpreter/01/issues/219">Ubuntu 21+ is not supported [wayland] · Issue #219 · OpenInterpreter/01</a>: Some dependencies uses x11 and is not compatible with wayland https://github.com/Kalmat/PyWinCtl?tab=readme-ov-file#linux-notice https://github.com/asweigart/pyautogui/issues?q=is%3Aissue+is%3Aopen...</li><li><a href="https://github.com/Clinteastman/c0mputer">GitHub - Clinteastman/c0mputer: Porting open-interpreter to the M5 Cardputer</a>: Porting open-interpreter to the M5 Cardputer. Contribute to Clinteastman/c0mputer development by creating an account on GitHub.</li><li><a href="https://github.com/m5stack/M5Unified/tree/develop">GitHub - m5stack/M5Unified at develop</a>: Unified library for M5Stack series. Contribute to m5stack/M5Unified development by creating an account on GitHub.</li><li><a href="https://ngrok.com/docs/getting-started/?os=linux">Quickstart | ngrok documentation</a>: This quickstart will use the ngrok agent to put your application on</li><li><a href="https://github.com/rhasspy/piper/?tab=readme-ov-file#running-in-python">GitHub - rhasspy/piper: A fast, local neural text to speech system</a>: A fast, local neural text to speech system. Contribute to rhasspy/piper development by creating an account on GitHub.</li><li><a href="https://dashboard.ngrok.com/get-started/setup/linux">ngrok - Online in One Line</a>: no description found
</li>
</ul>

</div>
  

---



**Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1225438495749967913)** (37 messages🔥): 

- **Introducing Command R+**: Command R+ is unveiled as a highly scalable LLM optimized for enterprise use, featuring advanced RAG for reduced hallucinations, multilingual support, and improved tool use. It boasts a context window of 128k tokens with model weights available for research use on [Cohere's platform](https://txt.cohere.com/command-r-plus-microsoft-azure/).
  
- **Command R+ Gains Attention**: The new Command R+ model, possessing 104B parameters and demonstrating RAG capabilities, raises questions about its relative performance to other models due to lack of comparative data, while a [live demo](https://huggingface.co/spaces/CohereForAI/c4ai-command-r-plus) is available for experimentation.

- **Scrutinizing ChatGPT for Business**: There's skepticism about the effectiveness of ChatGPT-like models for business applications, emphasizing that real enterprise use might require heavily customized solutions beyond what current "business-tailored" models offer.

- **Evaluating Models Raises Challenges**: Discussions touch on the complex and potentially biased nature of evaluating models like Command R+, highlighting the importance of structured benchmarks like AssistantBench for more transparent assessments.

- **JetMoE-8B: A Cost-Effective Milestone for Academia**: With costs under $0.1 million and performance surpassing Meta AI's LLaMA2 with only 2.2B active parameters during inference, JetMoE-8B represents a significant step in cost-effective and accessible LLMs for academic research, detailed on their [project page](https://research.myshell.ai/jetmoe).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://research.myshell.ai/jetmoe">JetMoE</a>: no description found</li><li><a href="https://txt.cohere.com/command-r-plus-microsoft-azure/">Introducing Command R+: A Scalable LLM Built for Business</a>: Command R+ is a state-of-the-art RAG-optimized model designed to tackle enterprise-grade workloads, and is available first on Microsoft Azure   Today, we’re introducing Command R+, our most powerful, ...</li><li><a href="https://huggingface.co/CohereForAI/c4ai-command-r-plus">CohereForAI/c4ai-command-r-plus · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/jetmoe/jetmoe-8b">jetmoe/jetmoe-8b · Hugging Face</a>: no description found</li><li><a href="https://openai.com/blog/openai-partners-with-scale-to-provide-support-for-enterprises-fine-tuning-models">OpenAI partners with Scale to provide support for enterprises fine-tuning models</a>: OpenAI’s customers can leverage Scale’s AI expertise to customize our most advanced models.</li><li><a href="https://fxtwitter.com/aidangomez/status/1775878606108979495?s=46">Tweet from Aidan Gomez (@aidangomez)</a>: ⌘R+  Welcoming Command R+, our latest model focused on scalability, RAG, and Tool Use. Like last time, we&#39;re releasing the weights for research use, we hope they&#39;re useful to everyone! https:/...
</li>
</ul>

</div>
  

---


**Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1225458505583034408)** (3 messages): 

- **Nathan Stirs the Pot**: Nathan Lambert hinted at controversy with a [tweet](https://twitter.com/natolambert/status/1775899591814300024), saying "Hopefully doesn't turn into drama..."

- **Calling Out Snorkel**: Following Nathan's comment, a reply called for a hot take on **Snorkel**, suggesting it's a topic ripe for discussion.

- **'All Models Are Bad' Article Tease**: Nathan Lambert teases an upcoming article titled "all these models are bad," seemingly critiquing current models, including those integrated with RLHF (Reinforcement Learning from Human Feedback).
  

---


**Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1225175436900962397)** (21 messages🔥): 

- **The FT's Locked Treasure**: A member shared a [Financial Times](https://www.ft.com/content/2f4bfeb4-6579-4819-9f5f-b3a46ff59ed1) offer for unlimited article access, suggesting the need for a paid subscription to access quality journalism. The link included images of padlocks and product icons, signifying locked content, hinting at a comparison between content access and digital product offerings.
- **Skepticism Over Business Models**: In a brief exchange, a member expressed concern that traditional business models might inhibit the success of "genAI", hinting at potential rigidities in existing operations and the prospect of what they refer to as "product suicide."
- **Tech Politics Discussion Not a Crowd-Pleaser**: Members shared a [link to a tech politics discussion](https://x.com/pmarca/status/1775691027363639634?s=20) by notable figures Ben Horowitz and Marc Andreessen, but the reception was lukewarm, with comments ranging from reluctant willingness to listen for political insight to outright dismissal of the content's value.
- **CS25 Course Lecture Feature**: A conversation revealed that a member, recognized as **nato**, is set to give a lecture for the CS25 course, expressing both anticipation and logistical considerations for the commitment.
- **Stanford's CS25 Course Attracts AI Enthusiasts**: The details of the CS25 course at Stanford were shared, indicating a robust lineup of discussions with Transformer research experts, including luminaries and industry professionals. Interested parties were pointed to a [schedule](https://web.stanford.edu/class/cs25/#schedule) and urged to tune into the course's YouTube channel for more insights.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/pmarca/status/1775691027363639634?s=20">Tweet from Marc Andreessen 🇺🇸 (@pmarca)</a>: You watch and enjoy please! Ben @bhorowitz and me for two hours on tech politics and policy in DC and beyond. Many X questions answered and points made. 🇺🇸🚀💪</li><li><a href="https://web.stanford.edu/class/cs25/#schedule">CS25: Tranformers United!</a>: Disussing the latest breakthroughs with Transformers in diverse domains</li><li><a href="https://www.ft.com/content/2f4bfeb4-6579-4819-9f5f-b3a46ff59ed1">Google considers charging for AI-powered search in big change to business model</a>: no description found
</li>
</ul>

</div>
  

---



**Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1225003496080937002)** (57 messages🔥🔥): 

- **Kernel Scalability Breakthrough**: A member discussed improving a matmul kernel to process prompts efficiently for large matrices, overcoming limitations of the CPU cache when the matrix size exceeds 1024x1024.
- **Compiler Magic Achieved**: Celebrations were shared on getting the compiler to transform code, presumably leading to performance improvements.
- **ROCm Version Clarification for Llamafile**: For llamafile-0.7 on Windows, **ROCm 5.7+** is required, indicating that support for different versions of ROCm, including 5.7 and 6.0.2, has been considered.
- **SYCL Code Saga Continues**: Spirited conversation about how to handle SYCL code in llamafile led to advice on checking out `llamafile/metal.c` and `llamafile/cuda.c` for dynamic loading of DSOs. A community member contributed by implementing conditional compilation for SYCL support to work on Windows and Linux, but not on Mac.
- **Llamafile Performance and Issue with Cosmopolitan on Windows**: A member attempted to build llamafile on Windows but faced issues with the Cosmopolitan compiler. An article was shared highlighting llamafile performance gains, and discussions surfaced around the need for a `llamafile-bench` program to benchmark tokens per second. It was suggested that more RAM and faster RAM could improve performance, as CPU or memory constraints were not identified as bottlenecks.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.theregister.com/2024/04/03/llamafile_performance_gains/">Llamafile LLM driver project boosts performance on CPU cores</a>: Way to whip that LLaMA&#39;s ass</li><li><a href="https://rocm.docs.amd.com/projects/install-on-windows/en/latest/how-to/install.html">Install HIP SDK — HIP SDK installation Windows</a>: no description found</li><li><a href="https://huggingface.co/models?library=gguf">Models - Hugging Face</a>: no description found</li><li><a href="https://github.com/jart/cosmopolitan/issues/1010">execve() should polyfill #! on windows · Issue #1010 · jart/cosmopolitan</a>: Copied from bellard/quickjs#197: #!/bin/qjs console.log(&quot;Hello&quot;); It doesn&#39;t work when invoked from bash as script: $ ./test.qjs ./test.qjs: line 2: syntax error near unexpected token `&...
</li>
</ul>

</div>
  

---



**LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1225043803778449528)** (36 messages🔥): 

- **Seeking Experienced Chatbot Devs for Crypto**: A user is looking for developers experienced in training LLMs and integrating them with a real-time database containing crypto market news and information, aiming to build a human-like chatbot.
- **Extracting Math Symbols from PDF**: A user inquires about alternatives to MathpixPDFLoader for extracting math symbols from PDF files, preferring other methods that could handle this task.
- **LangChain Community Connection Sought**: A user is seeking a community manager or developer advocate at LangChain for assistance with an integration, and was provided a link to contributing integrations: [Guide to contributing integrations](https://python.langchain.com/docs/contributing/integrations).
- **Novice Inquiry on Bot Development with Langchain in JS**: A user new to using Langchain in JavaScript is seeking guidance on creating a bot that schedules appointments and interacts with a database. Experienced users recommended Sequelize, an ORM for Node.js, for database interactions, with a link: [Sequelize GitHub Repository](https://github.com/sequelize/sequelize/tree/9e141880230a7f2a9a8c1e66a31f29fea7b5a65a).
- **LCEL Chaining Puzzles and Poses**: Members discussed the purpose of the '|' operator in LCEL (LangChain's Expression Language), which chains components such as prompts and llm outputs. A link was provided for further reading: [Getting Started with LCEL](https://python.langchain.com/docs/expression_language/get_started).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://python.langchain.com/docs/expression_language/get_started">Get started | 🦜️🔗 Langchain</a>: LCEL makes it easy to build complex chains from basic components, and</li><li><a href="https://python.langchain.com/docs/contributing/integrations">Contribute Integrations | 🦜️🔗 Langchain</a>: To begin, make sure you have all the dependencies outlined in guide on Contributing Code.</li><li><a href="https://github.com/langchain-ai/langchain/discussions/19957">When to use Outputparsers, tools, and/or LangSmith Evaluators to test LLM output? · langchain-ai/langchain · Discussion #19957</a>: I was working on a simple LCEL chain for a simple task, and this question came to my mind. Imagine I have a straightforward LCEL chain containing 2 prompts and 2 output parsers that &quot;force&quot; ...</li><li><a href="https://github.com/brianc/node-postgres/tree/master">GitHub - brianc/node-postgres: PostgreSQL client for node.js.</a>: PostgreSQL client for node.js. Contribute to brianc/node-postgres development by creating an account on GitHub.</li><li><a href="https://github.com/sequelize/sequelize/tree/9e141880230a7f2a9a8c1e66a31f29fea7b5a65a">GitHub - sequelize/sequelize at 9e141880230a7f2a9a8c1e66a31f29fea7b5a65a</a>: Feature-rich ORM for modern Node.js and TypeScript, it supports PostgreSQL (with JSON and JSONB support), MySQL, MariaDB, SQLite, MS SQL Server, Snowflake, Oracle DB (v6), DB2 and DB2 for IBM i. - ...
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1225024321806667787)** (2 messages): 

- **CI Failure Puzzle**: A member sought assistance with a **continuous integration failure** on a PR that aims to serve the playground from the correct route, even with nested API routers. The PR in question is [Serve playground from correct route PR #580](https://github.com/langchain-ai/langserve/pull/580), and local tests were passed using Python 3.10.

- **Chat Playground Walkthrough**: A tutorial video has been shared, showcasing how to use **Agents with the new Chat Playground of Langserve**. The detailed walkthrough, including managing initial difficulties and featuring Langsmith, can be found on [YouTube](https://www.youtube.com/watch?v=stWiNP1o2_g), with the final code provided in the description.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=stWiNP1o2_g">The NEW Langserve Chat Playground with Agents | Coding Showcase</a>: In this technical deep dive, we&#39;ll guide you through the exciting world of LangChain and LangServe frameworks. In 17 minutes, we&#39;ll present you with a compre...</li><li><a href="https://github.com/langchain-ai/langserve/pull/580">WIP: Serve playground from correct route if nested APIrouters within one another by StreetLamb · Pull Request #580 · langchain-ai/langserve</a>: Update playground tests to check for the correct playground assets path in index.html. #578
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[langchain-templates](https://discord.com/channels/1038097195422978059/1170025009960456282/1225540697554419803)** (1 messages): 

- **Trouble with Agents Searching PDFs**: A member highlighted an issue where their agent searches PDFs for every query. They suspected the *system_prompt* in the code to be the cause and sought advice on how to revise it.
  

---


**LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1225013268561788998)** (5 messages): 

- **Launch of Multiple Voice Apps**: The user announced the launch of several new voice apps, including [CallStar](https://callstar.ai/), with a request to join the discussion and support the launch. The suite includes specialized apps like CallJesus and CallPDF, with links to Product Hunt and Reddit for upvotes.

- **Voice App Interactivity Inquiry**: In response to the voice apps launch, a user inquired about the documentation behind the apps' responsive design. The original poster recommended **RetellAI** as the technology they use.

- **AllMind AI Tailored for Finance**: A new LLM named [AllMind AI](https://allmindinvestments.com/), which focuses on financial analysis and research, was introduced to the community. The tool aims to provide users with insights and comprehensive financial data and was also featured on [Product Hunt](https://www.producthunt.com/products/allmind-ai).

- **Galaxy AI's Free Premium API Service**: Galaxy AI presents a **free API service** granting access to various premium AI models including **GPT-4** and **Gemini-PRO**. This service is compatible with OpenAI format for easy project integration and supports Langchain, with further details to [try now](https://discord.com/invite/BSphj69773) and more info at their [homepage](https://galaxyapi.onrender.com).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://galaxyapi.onrender.com">Galaxy AI - Swagger UI</a>: no description found</li><li><a href="https://callstar.ai/">CallStar</a>: AI Voice Calls with Characters and Celebrities</li><li><a href="https://allmindinvestments.com/">AllMind AI</a>: no description found</li><li><a href="https://www.producthunt.com/products/allmind-ai"> AllMind AI - Product Information, Latest Updates, and Reviews 2024 | Product Hunt</a>: AllMind AI is a new large language model designed exclusively for financial analysis and research. This LLM revolutionizes financial research by offering users access to insights and providing real-ti...</li><li><a href="https://calljesus.ai/">Call Jesus</a>: Realistic AI Voice Chats with Jesus</li><li><a href="https://callpdf.ai/">CallPDF</a>: Call any PDF - Realistic AI Voice Chats</li><li><a href="https://calltube.ai/">CallTube</a>: Call any YouTube Video - Realistic AI Voice Chats</li><li><a href="https://callwebsite.ai/">Call Website</a>: Call any Website - Realistic AI Voice Chats</li><li><a href="https://callhackernews.com/">Call Hacker News</a>: AI Voice Interface for Hacker News</li><li><a href="https://www.producthunt.com/posts/callstar"> CallStar - Realistic AI voice calls with characters, YT-videos &amp; PDFs | Product Hunt</a>: Next-level AI voice calls! Chat with celebrities, understand your docs with voice &amp; explore spirituality. Make AI conversations feel real and personal with best-in-class AI voices. Call PDFs, YouT...</li><li><a href="https://www.reddit.com/r/SideProject/comments/1bumj6s">Reddit - Dive into anything</a>: no description found</li><li><a href="https://news.ycombinator.com/item?id=39914442">no title found</a>: no description found
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1225149749418659880)** (1 messages): 

- **LangChain Quickstart Guide Tour**: A user shared a link to the [LangChain Quickstart Guide](https://python.langchain.com/docs/get_started/quickstart) which provides a detailed walkthrough of setting up LangChain, LangSmith, and LangServe, along with using prompt templates, models, output parsers, and the LangChain Expression Language to build and trace simple applications.
- **Example Code & Error Encounter**: The same user posted example Python code showcasing the integration of LangChain with a model using `ChatOpenAI` and `ChatPromptTemplate` classes. However, they encountered a `NotFoundError` with a `404` error code when running their code, indicating that a resource could not be found, and sought assistance with this issue.

**Link mentioned**: <a href="https://python.langchain.com/docs/get_started/quickstart">Quickstart | 🦜️🔗 Langchain</a>: In this quickstart we&#x27;ll show you how to:

  

---



**CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1225367465639415828)** (3 messages): 

- **BitMat Unveiled**: A link was shared to [BitMat's GitHub repo](https://github.com/astramind-ai/BitMat), which offers an **efficient implementation** of the method proposed in "The Era of 1-bit LLMs".
- **Collaboration on Triton Visualizer**: A new channel for contributors to the **Triton visualizer** was proposed to facilitate collaboration on the project.
- **Lightning Strikes with LASP**: Another GitHub link was provided to [LASP's lightning_attention.py file](https://github.com/OpenNLPLab/LASP/blob/main/lasp/lightning_attention.py), concerning **Linear Attention Sequence Parallelism (LASP)**.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/OpenNLPLab/LASP/blob/main/lasp/lightning_attention.py">LASP/lasp/lightning_attention.py at main · OpenNLPLab/LASP</a>: Linear Attention Sequence Parallelism (LASP). Contribute to OpenNLPLab/LASP development by creating an account on GitHub.</li><li><a href="https://github.com/astramind-ai/BitMat">GitHub - astramind-ai/BitMat: An efficent implementation of the method proposed in &quot;The Era of 1-bit LLMs&quot;</a>: An efficent implementation of the method proposed in &quot;The Era of 1-bit LLMs&quot; - astramind-ai/BitMat
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1225300703925502022)** (4 messages): 

- **Switching to max-autotune compilation**: A member suggested setting the compilation mode to **max-autotune** instead of reduce-overhead, sharing their experience of its benefits and expressing interest in other issues the torch team may find in the [keras-benchmarks](https://github.com/haifeng-jin/keras-benchmarks/blob/main/benchmark/torch_utils.py#L17).
- **Identifying torch benchmarking issues**: The torch team's biggest concerns include **not utilizing tensor cores** and the inconsistency of enabling `torch.compile`. They also noted problems with benchmarks like SAM, graph breaks that are fixable, and improper timing methods without cuda syncs, all of which they're addressing in a detailed response to come.

**Link mentioned**: <a href="https://github.com/haifeng-jin/keras-benchmarks/blob/main/benchmark/torch_utils.py#L17">keras-benchmarks/benchmark/torch_utils.py at main · haifeng-jin/keras-benchmarks</a>: Contribute to haifeng-jin/keras-benchmarks development by creating an account on GitHub.

  

---


**CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/)** (1 messages): 

iron_bound: : insert rant about repeatability in science here :
  

---


**CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1225047565670547496)** (3 messages): 

- **CUDA Learning Path for Python and Rust Background**: A member with experience in Python and Rust asked for recommendations on learning **CUDA programming** basics.  
- **CUDA MODE YouTube Lectures Suggested**: Another member suggested starting with CUDA lectures available on a [YouTube channel called CUDA MODE](https://www.youtube.com/@CUDAMODE), which also offers a **reading group and community** on Discord and supplementary content on GitHub.

**Link mentioned**: <a href="https://www.youtube.com/@CUDAMODE">CUDA MODE</a>: A CUDA reading group and community https://discord.gg/cudamode Supplementary content here https://github.com/cuda-mode Created by Mark Saroufim and Andreas Köpf    

  

---


**CUDA MODE ▷ #[ring-attention](https://discord.com/channels/1189498204333543425/1208496482005549086/1225462650423480483)** (1 messages): 

- **Lightning Fast Attention**: The **Triton "lightning_attention" kernel** is mentioned as an efficient solution, nullifying the need for the FlashAttention repo plug which handled splitting data across devices. More details are available on the [LASP GitHub repository](https://github.com/OpenNLPLab/LASP).

**Link mentioned**: <a href="https://github.com/OpenNLPLab/LASP">GitHub - OpenNLPLab/LASP: Linear Attention Sequence Parallelism (LASP)</a>: Linear Attention Sequence Parallelism (LASP). Contribute to OpenNLPLab/LASP development by creating an account on GitHub.

  

---


**CUDA MODE ▷ #[hqq](https://discord.com/channels/1189498204333543425/1225499037516693574/1225500024029581503)** (19 messages🔥): 

- **Introduction to the CUDA MODE Community**: New members **mobicham** and **zhxchen17** joined the CUDA MODE Discord and were welcomed by the community.
- **Integration of HQQ with GPT-fast**: zhxchen17 proposed creating a demo branch to show how **HQQ** can integrate with **gpt-fast**, including a separate branch for dependencies, a script for converting quantized weights, and benchmarking for collaborative review.
- **Focus on Llama2 Model and Quantization**: Mobicham suggested focusing on **Llama2-7B (base)** for integration due to existing benchmarks, and inquired about the desire to explore lower bit-level quantization beyond 4-bit. zhxchen17 confirmed looking into **4/3 bit quantization** with a specific interest in the [Mixtral-8x7B-Instruct-v0.1-hf-attn-4bit-moe-3bit-metaoffload-HQQ model](https://huggingface.co/mobiuslabsgmbh/Mixtral-8x7B-Instruct-v0.1-hf-attn-4bit-moe-3bit-metaoffload-HQQ).
- **Confirmation and Clarification of Task Focus**: After some confusion, mobicham clarified the goal of converting **Llama2 HQQ** into **gpt-fast format**, highlighting that a 4-bit HQQ with an appropriate group size could yield significant speed improvements.
- **Potential Group-Size Restrictions and API Considerations**: There was a discussion about potential group-size restrictions with `torch.ops.aten._weight_int4pack_mm` and the design space for an API that converts models to **GPT-fast format**, with zhxchen17 indicating that the **torchao team** would be better equipped to define the API design.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/mobiuslabsgmbh/Mixtral-8x7B-Instruct-v0.1-hf-attn-4bit-moe-3bit-metaoffload-HQQ">mobiuslabsgmbh/Mixtral-8x7B-Instruct-v0.1-hf-attn-4bit-moe-3bit-metaoffload-HQQ · Hugging Face</a>: no description found</li><li><a href="https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/int4mm.cu#L912">pytorch/aten/src/ATen/native/cuda/int4mm.cu at main · pytorch/pytorch</a>: Tensors and Dynamic neural networks in Python with strong GPU acceleration - pytorch/pytorch</li><li><a href="https://github.com/meta-llama/llama">GitHub - meta-llama/llama: Inference code for Llama models</a>: Inference code for Llama models. Contribute to meta-llama/llama development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[triton-viz](https://discord.com/channels/1189498204333543425/1225499141241573447/1225499224251039804)** (13 messages🔥): 

- **Suggestion for Visual Indicators**: A member proposed adding **arrows or visual indicators** for direction in the visualizations and provided a quick mock-up to illustrate the idea; however, they also mentioned that not every element should have an arrow, just enough to convey the concept.
- **Integrating Operation Details into Visuals**: The same member shared a code snippet highlighting the suggestion to visually represent operations, such as showing how an operand like **10** is added to input, similar to how a kernel functions in code.
- **Concerns About Current Visualization Utility**: A different member expressed concerns regarding the usefulness of adding indices to the current visualization and whether it would actually aid in understanding.
- **Idea for Debugging with Interactive Elements**: It was suggested that having **interactive elements** in the visualization, like hovering to see values, could be advantageous for debugging purposes.
- **Potential Shift to JavaScript for Enhanced Interactivity**: There was a mention that for implementing interactivity such as mouseovers in the visualization, it might be necessary to **port the project to JavaScript**.
  

---



**Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/1225504686820298752)** (17 messages🔥): 

- **Evaluating AI Improvements**: Members discussed the value of [Hamel Husain's post on systematic AI improvements](https://hamel.dev/blog/posts/evals/), describing it as "insanely valuable," with potential to inspire the founding of several companies.
- **Datasette Plugin Enhancements**: The idea was proposed to build evaluations for the Datasette SQL query assistant plugin, making them both **visible** and **editable** to users.
- **The Prompt Dilemma**: A member pondered whether prompts should reside within code, currently leaning towards "yes," but speculating this might not be sustainable in the long term.
- **Evolving Prompt Management Practices**: Potential future patterns for AI prompt management were outlined: **localization, middleware, and microservice patterns**, reflecting different strategies for integrating AI services into larger applications.
- **Importance of Detailed API Response Data**: The Cohere LLM search API was mentioned, highlighting the level of detail provided in responses, with a link to an issue comment showing a JSON output: [Cohere API JSON data](https://github.com/simonw/llm-command-r/issues/2#issuecomment-2037420135).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://hamel.dev/blog/posts/evals/">- Your AI Product Needs Evals</a>: How to construct domain-specific LLM evaluation systems.</li><li><a href="https://github.com/simonw/llm-command-r/issues/2#issuecomment-2037420135">Support for the web search connector · Issue #2 · simonw/llm-command-r</a>: If you add this to the API call: diff --git a/llm_command_r.py b/llm_command_r.py index 7a334cd..e49c599 100644 --- a/llm_command_r.py +++ b/llm_command_r.py @@ -43,6 +43,8 @@ class CohereMessages(...
</li>
</ul>

</div>
  

---


**Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/1225067602288574554)** (1 messages): 

- **Terminology Tweaked for Dialogue**: A member shared their finding on conversational terminology while exploring a `logs.db` and mentioned that the term "response" might not be apt for the initial message in a conversation. They highlighted that "speaker turn" or "turn" is more appropriate and have decided to name their app's table `turns`, amused by the accidental pun made.
  

---



**DiscoResearch ▷ #[benchmark_dev](https://discord.com/channels/1178995845727785010/1183158791605330051/1225373204206456872)** (10 messages🔥): 

- **Emotional Intelligence Benchmarks Unveiled**: Two new leaderboards have been launched: [Creative Writing EQ-Bench](https://eqbench.com/creative_writing.html), evaluating emotional intelligence in LLMs, and [Judgemark](https://eqbench.com/judgemark.html), which measures a model's ability to judge creative writing. Judgemark is described as a *hard* test involving correlation metrics and cost considerations; the benchmarks can be run through the EQ-Bench pipeline.
- **Quality Ratings - Finding the Sweet Spot**: When assessing the use of different scales for ratings – from -10 to 10, 0-10, 0-1, etc. – it was found that for sentiment, a scale of -1 to 1 works well, while for quality judgments, scales of 0-5 or 0-10 are preferred as models tend to use their ingrained understanding of what numbers mean.
- **Creative Writing Judged on Details**: The creative writing benchmark's success was credited to the use of **36 narrowly defined judging criteria**. Scores based on broad criteria such as "rate this story 0-10" resulted in weak discrimination.
- **Benchmark Criteria Documented**: Questions about the judging criteria for benchmarks were addressed with a link to judge outputs, which included criteria. The example provided was [gemini-ultra.txt](https://eqbench.com/results/creative-writing/gemini-ultra.txt) from the EQ-bench results.
- **Fine-tuning Rating Scales**: Standard deviation of scores between models was used as an indicator to gauge the discriminative power of a question or criteria, and through this process, a 0-10 rating system was determined to be the most effective. Models tend to use the 0-10 range fully, which is assumed to add granularity compared to a 0-5 system.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://eqbench.com/creative_writing.html">EQ-Bench Creative Writing Leaderboard</a>: no description found</li><li><a href="https://eqbench.com/judgemark.html">EQ-Bench Judgemark Leaderboard</a>: no description found
</li>
</ul>

</div>
  

---


**DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/1225196411839189022)** (3 messages): 

- **COMET Scores Unveiled**: A member shared **COMET** scores demonstrating the performance of various language models, with the **Facebook WMT21** model standing out. The highest score was 0.848375 for a file named *Capybara_de_wmt21_scored.jsonl*.
  
- **Reference-Free Evaluation**: The scores mentioned are **reference-free COMET scores**, specifically using **wmt22-cometkiwi-da**. Additional resources and scripts related to the evaluation were mentioned, available at [llm_translation on GitHub](https://github.com/CrispStrobe/llm_translation).

- **Accuracy Caveats**: The posted results are indicative but not absolute. The member noted potential inaccuracies when models stop continuing and requested to be notified of any significant errors.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/cstr/wmt21-dense-24-wide-en-x-st/">cstr/wmt21-dense-24-wide-en-x-st · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co">Hugging Face – The AI community building the future.</a>: no description found</li><li><a href="https://github.com/CrispStrobe/llm_translation/">GitHub - CrispStrobe/llm_translation</a>: Contribute to CrispStrobe/llm_translation development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



**Skunkworks AI ▷ #[general](https://discord.com/channels/1131084849432768614/1131084849906716735/1225271024485007420)** (2 messages): 

- **AI in Healthcare Gains Another Voice**: A participant expressed their involvement in the **AI medical field**, indicating a growing number of community members in this healthcare tech space.

- **Innovating LLMs with Mixture-of-Depths (MoD)**: A new approach, called **Mixture-of-Depths (MoD)**, has been shared; it allows Language Models to allocate compute dynamically, with an ability to skip the use of a single expert dynamically. The paper and its abstract are accessible via the [PDF on arXiv](https://arxiv.org/abs/2404.02258).

**Link mentioned**: <a href="https://arxiv.org/abs/2404.02258">Mixture-of-Depths: Dynamically allocating compute in transformer-based language models</a>: Transformer-based language models spread FLOPs uniformly across input sequences. In this work we demonstrate that transformers can instead learn to dynamically allocate FLOPs (or compute) to specific ...

  

---


**Skunkworks AI ▷ #[finetuning](https://discord.com/channels/1131084849432768614/1131669354912678028/1225272891013337089)** (1 messages): 

- **Decomposition Strategy for Math Problems**: A member mentioned that rather than having an AI do math directly, it's better to train it to break down word problems into equations. These equations could then be solved using an external calculator, like **Python** or **Wolfram Alpha**.
  

---


**Skunkworks AI ▷ #[papers](https://discord.com/channels/1131084849432768614/1156310031768232007/)** (1 messages): 

carterl: https://arxiv.org/abs/2404.02684
  

---



**Alignment Lab AI ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/)** (1 messages): 

jinastico: <@748528982034612226>
  
