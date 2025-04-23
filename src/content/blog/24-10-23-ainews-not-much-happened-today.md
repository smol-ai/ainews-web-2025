---
id: 87b2cb83-51c8-40f3-995f-4cb8953a9a66
title: not much happened today
date: '2024-10-24T00:39:59.759230Z'
original_slug: ainews-not-much-happened-today-5175
description: >-
  Anthropic released Claude 3.5 Sonnet and Haiku models featuring a new computer
  use capability that allows the AI to interact with computer interfaces by
  interpreting screenshots and performing actions like clicking and typing.
  Claude 3.5 Sonnet also achieved state-of-the-art coding performance,
  surpassing OpenAI’s o1-preview model.
tags:
  - anthropic
  - claude-3.5
  - computer-use
  - coding
  - model-release
  - stable-diffusion
  - video-generation
---


<!-- buttondown-editor-mode: plaintext -->**a quiet day is all you need.**

> AI News for 10/22/2024-10/23/2024. We checked 7 subreddits, [**433** Twitters](https://twitter.com/i/lists/1585430245762441216) and **31** Discords (**229** channels, and **3078** messages) for you. Estimated reading time saved (at 200wpm): **346 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

People are still very much exploring the implications of Anthropic's new Computer Use demo/usecases. Some are pointing out [its failures](https://x.com/natolambert/status/1849082872436793647?s=46) and [picking apart the terminology](https://x.com/doomslide/status/1849204183205081231), others have hooked it up to [phone simulators](https://x.com/mckaywrigley/status/1849145631895593292) and [real phones](https://x.com/ethansutin/status/1849187111255310513?s=46). Kyle Corbitt somehow [wrote a full desktop app for Computer Use in 6 hours](https://news.ycombinator.com/item?id=41926770) so you dont have to spin up the docker demo Anthropic shipped.

![image.png](https://assets.buttondown.email/images/3fd69e85-2d24-426a-ab9a-b67c725483e7.png?w=960&fit=max)

But not a single person in the room has any doubt that this will get a lot better very soon and very quickly.

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

**Anthropic's Claude 3.5 Release and Computer Use Capability**

- **New Models and Capabilities**: [@AnthropicAI](https://twitter.com/AnthropicAI/status/1848742740420341988) announced an upgraded Claude 3.5 Sonnet, a new Claude 3.5 Haiku model, and a computer use capability in beta. This allows Claude to interact with computers by looking at screens, moving cursors, clicking, and typing.

- **Computer Use Details**: [@alexalbert__](https://twitter.com/alexalbert__/status/1848743043429810361) explained that the computer use API allows Claude to perceive and interact with computer interfaces. Users feed in screenshots, and Claude returns the next action to take (e.g., move mouse, click, type text).

- **Performance Improvements**: [@alexalbert__](https://twitter.com/alexalbert__/status/1848743106063306826) noted significant gains in coding performance, with the new 3.5 Sonnet setting a state-of-the-art on SWE-bench Verified with a score of 49%, surpassing all models including OpenAI's o1-preview.

- **Haiku Model**: [@alexalbert__](https://twitter.com/alexalbert__/status/1848743124417581343) shared that the new Claude 3.5 Haiku replaces 3.0 Haiku as Anthropic's fastest and least expensive model, outperforming many state-of-the-art models on coding tasks.

- **Development Process**: [@AnthropicAI](https://twitter.com/AnthropicAI/status/1848742757151498717) mentioned they're teaching Claude general computer skills instead of making specific tools for individual tasks, allowing it to use standard software designed for people.

- **Limitations and Future Improvements**: [@AnthropicAI](https://twitter.com/AnthropicAI/status/1848742758996971746) acknowledged that Claude's current ability to use computers is imperfect, with challenges in actions like scrolling, dragging, and zooming. They expect rapid improvements in the coming months.

**Other AI Model Releases and Updates**

- **Mochi 1**: [@_parasj](https://twitter.com/_parasj/status/1848763942216044946) announced Mochi 1, a new state-of-the-art open-source video generation model released under Apache 2.0 license.

- **Stable Diffusion 3.5**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1848763032232751210) reported the release of Stable Diffusion 3.5, including Large (8B parameters) and Medium (2.5B parameters) variants, with improvements in training stability and fine-tuning flexibility.

- **Embed 3**: [@cohere](https://twitter.com/cohere/status/1848760845641388087) launched Embed 3, a multimodal embedding model enabling enterprises to build systems that can search across both text and image data sources.

- **KerasHub**: [@fchollet](https://twitter.com/fchollet/status/1848800260115906716) announced the launch of KerasHub, consolidating KerasNLP & KerasCV into a unified package covering all modalities, including 37 pretrained models and associated workflows.

**AI Research and Development**

- **Differential Transformer**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1848746618620944720) discussed a new paper from Microsoft introducing the "Differential Transformer," which uses differential attention maps to remove attention noise and push the model toward sparse attention.

- **Attention Layer Removal**: [@rasbt](https://twitter.com/rasbt/status/1848714250984034771) shared findings from a paper titled "What Matters In Transformers?" which found that removing half of the attention layers in LLMs like Llama doesn't noticeably reduce modeling performance.

- **RAGProbe**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1848773738734752175) highlighted a paper introducing RAGProbe, an automated approach for evaluating RAG (Retrieval-Augmented Generation) pipelines, exposing limitations and failure rates across various datasets.

**Industry Developments and Collaborations**

- **Perplexity Pro**: [@AravSrinivas](https://twitter.com/AravSrinivas/status/1848801520818786452) announced that Perplexity Pro is transitioning to a reasoning-powered search agent for harder queries involving several minutes of browsing and workflows.

- **Timbaland and Suno**: [@suno_ai_](https://twitter.com/suno_ai_/status/1848748300062634130) shared that Grammy-winning producer Timbaland is collaborating with Suno AI, exploring how AI is helping him rediscover creativity in music production.

- **Replit Integration**: [@pirroh](https://twitter.com/pirroh/status/1848752337080488177) mentioned that Replit has integrated Claude computer use as a human feedback replacement in their Agent, reporting that it "just works."

---

# AI Reddit Recap

## /r/LocalLlama Recap

**Theme 1. Major LLM Updates: Claude 3.5 and Stable Diffusion 3.5**

- **[Stability AI has released Stable Diffusion 3.5, comes in three variants, Medium launches October 29th.](https://huggingface.co/stabilityai/stable-diffusion-3.5-large-turbo)** ([Score: 110, Comments: 40](https://reddit.com//r/LocalLLaMA/comments/1g9j5b6/stability_ai_has_released_stable_diffusion_35/)): Stability AI has released **Stable Diffusion 3.5**, offering three variants: **Base**, **Medium**, and **Large**. The **Base** model is available now, with **Medium** set to launch on **October 29th**, and **Large** coming at a later date. This new version boasts improved image quality, better text understanding, and enhanced capabilities in areas like composition, lighting, and anatomical accuracy.
  - Users humorously noted **Stability AI** included an image of a **woman on grass** in their blog, referencing a previous meme. Some tested the model's ability to generate this scene, with mixed results including unexpected **NSFW content**.
  - Comparisons between **SD3.5** and **Flux1-dev** were made, with users reporting that **Flux1-dev** generally produced more realistic outcomes and less deformities in limited testing.
  - The community discussed potential applications of **SD3.5**, including its use as a base for fine-tuning projects. However, some noted that the **license restrictions** may limit its adoption for certain use cases.
- **[Introducing computer use, a new Claude 3.5 Sonnet, and Claude 3.5 Haiku](https://www.anthropic.com/news/3-5-models-and-computer-use)** ([Score: 187, Comments: 82](https://reddit.com//r/LocalLLaMA/comments/1g9krp2/introducing_computer_use_a_new_claude_35_sonnet/)): Anthropic has launched **Claude 3.5 Sonnet** and **Claude 3.5 Haiku**, introducing **computer use capability** that allows the AI to interact with virtual machines. These models can now perform tasks like web browsing, file manipulation, and running code, with Sonnet offering improved performance over Claude 3.0 and Haiku providing a faster, more cost-effective option for simpler tasks. The new versions are available through the API and Claude web interface, with computer use currently in **beta** and accessible to a limited number of customers.
  - **Claude 3.5 Sonnet** shows significant performance improvements over previous versions, with users noting its strength in **coding tasks**. The model now offers **computer use capability** in **beta**, allowing interaction with virtual machines for tasks like web browsing and file manipulation.
  - Users express concerns about **safety implications** of giving Claude remote code execution capabilities. Anthropic recommends precautions such as using dedicated virtual machines and limiting access to sensitive data when utilizing the computer use feature.
  - The naming convention for Claude models has become confusing, with **Claude 3.5 Sonnet** and **Claude 3.5 Sonnet (new)** causing potential mix-ups. Users suggest clearer versioning, comparing the current naming strategy to complex product names from companies like Samsung and Sony.


**Theme 2. Open Source AI Model Developments and Replication Efforts**

- **[O1 Replication Journey: A Strategic Progress Report – Part I](https://github.com/GAIR-NLP/O1-Journey)** ([Score: 34, Comments: 6](https://reddit.com//r/LocalLLaMA/comments/1g9eohc/o1_replication_journey_a_strategic_progress/)): The author reports on their progress in replicating **OpenAI's O1 model**, focusing on the **first 12 billion parameters** of the **120B parameter model**. They outline their strategy of training smaller models to validate components before scaling up, and have successfully trained models up to **1.3B parameters** using techniques like **flash attention** and **rotary embeddings**. The next steps involve scaling to **12B parameters** and implementing additional features such as **multi-query attention** and **grouped-query attention**.
  - The author clarifies that the focus of the article is on the **learning method and results**, not the dataset, in response to a question about the dataset's composition and creation process.
  - The **O1 Replication Journey** tech report, which hasn't been widely discussed, introduces a shift from "shortcut learning" to **"journey learning"** and explores **O1's thought structure**, **reward models**, and **long thought construction** using various methodologies.
  - A commenter notes the project's success in producing **longer-form reasoning answers** with commentary similar to O1, but points out that the **research artifacts** (fine-tuned models and "Abel" dataset) are not currently publicly available.
- **🚀 Introducing Fast Apply - Replicate Cursor's Instant Apply model** ([Score: 187, Comments: 40](https://reddit.com//r/LocalLLaMA/comments/1ga25gj/introducing_fast_apply_replicate_cursors_instant/)): **Fast Apply** is an open-source, fine-tuned **Qwen2.5 Coder Model** designed to quickly apply code updates from advanced models to produce fully edited files, inspired by Cursor's Instant Apply model. The project offers two models (**1.5B** and **7B**) with performance speeds of **~340 tok/s** and **~150 tok/s** respectively using a fast provider (Fireworks), making it practical for everyday use and lightweight enough to run locally. The project is fully open-source, with models, data, and scripts available on [HuggingFace](https://huggingface.co/Kortix/FastApply-1.5B-v1.0) and [GitHub](https://github.com/kortix-ai/fast-apply), and can be tried on [Google Colab](https://colab.research.google.com/drive/1BNCab4oK-xBqwFQD4kCcjKc7BPKivkm1?usp=sharing).
  - The project received praise for being **open-source**, with users expressing enthusiasm for its accessibility and potential for improvement. The developer mentioned plans to create a **better benchmark** using tools like **DeepSeek**.
  - Users inquired about **accuracy comparisons** between the **1.5B and 7B models**. The developer shared a rough benchmark showing the 1.5B model's impressive performance for its size, recommending users start with it before trying the 7B version if needed.
  - Discussion touched on potential **integration with other tools** like **continue.dev** and **Aider**. The developer expressed interest in submitting **PRs** to support and integrate the project with existing platforms that currently only support diff/whole formats.


**Theme 3. AI Model Comparison Tools and Cost Optimization**

- **I built an LLM comparison tool - you're probably overpaying by 50% for your API (analysing 200+ models/providers)** ([Score: 44, Comments: 16](https://reddit.com//r/LocalLLaMA/comments/1g9js22/i_built_an_llm_comparison_tool_youre_probably/)): A developer created a **free tool** ([https://whatllm.vercel.app/](https://whatllm.vercel.app/)) to compare **200+ LLM models** across **15+ providers**, analyzing price, performance, and quality scores. Key findings include significant price disparities (e.g., **Qwen 2.5 72B** is **94% cheaper** than **Claude 3.5 Sonnet** for similar quality) and performance variations (e.g., **Cerebras's Llama 3.1 70B** is **18x faster** and **40% cheaper** than Amazon Bedrock's version).
  - The developer provided **visualizations** to help understand the data, including a chart comparing metrics like price, speed, and quality. They used **Nebius AI Studio's free inference credits** with **Llama 70B Fast** for data processing and comparisons.
  - Discussion arose about the validity of the quality index, with the developer noting that **Qwen 2.5** scores only slightly lower on **MMLU-pro** and **HumanEval**, but higher on **Math** benchmarks compared to more expensive models.
  - Users expressed appreciation for the tool, with one calling it a "game changer" for finding the best LLM provider. The developer also recommended **Nebius AI Studio** for users looking for LLMs with European data centers in **Finland** and **France**.
- **[Transformers.js v3 is finally out: WebGPU Support, New Models & Tasks, New Quantizations, Deno & Bun Compatibility, and More…](https://v.redd.it/kkrx8g6fqbwd1)** ([Score: 75, Comments: 4](https://reddit.com//r/LocalLLaMA/comments/1g9kkbb/transformersjs_v3_is_finally_out_webgpu_support/)): **Transformers.js v3** has been released, introducing **WebGPU support** for significantly faster inference on compatible devices. The update includes new models and tasks such as **text-to-speech**, **speech recognition**, and **image segmentation**, along with expanded quantization options and compatibility with **Deno** and **Bun** runtimes. This version aims to enhance performance and broaden the library's capabilities for machine learning tasks in JavaScript environments.
  - **Transformers.js v3** release highlights include **WebGPU support** for up to 100x faster inference, **120 supported architectures**, and over **1200 pre-converted models**. The update is compatible with **Node.js**, **Deno**, and **Bun** runtimes.
  - Users expressed enthusiasm for the library's performance, with one noting consistent surprise at how fast the models run in-browser. The community showed appreciation for the extensive development and sharing of this technology.
  - A developer inquired about the possibility of including **ONNX conversion scripts** used in the release process, indicating interest in the technical details behind the library's model conversions.


**Theme 4. GPU Hardware Discussions for AI Development**

- **What the max you will pay for 5090 if the leaked specs are true?** ([Score: 32, Comments: 94](https://reddit.com//r/LocalLLaMA/comments/1g9j7bl/what_the_max_you_will_pay_for_5090_if_the_leaked/)): The post speculates on the potential specifications and performance of NVIDIA's upcoming **5090 GPU**. It suggests the 5090 might feature a **512-bit memory bus**, **32GB of RAM**, and be **70% faster** than the current **4090 model** for AI workloads.
  - Users debate the value of **32GB VRAM**, with some arguing it's insufficient for **LLM workloads**. Many prefer multiple **3090s** or **4090s** for their combined VRAM capacity, especially for running **70B models**.
  - Discussion on potential **pricing** of the 5090, with estimates ranging from $2000-$3500. Some speculate the **4090's price** may decrease, potentially flooding the market with used GPUs.
  - Comparisons made between the 5090 and other options like **multiple 3090s** or the **A6000**. Users emphasize the importance of total VRAM over raw performance for AI workloads.

## Other AI Subreddit Recap

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity


**AI Model Releases and Improvements**

- **Anthropic releases updated Claude 3.5 models**: Anthropic announced [updated versions of Claude 3.5 Sonnet and Claude 3.5 Haiku](https://www.reddit.com/r/singularity/comments/1g9kevd/announcing_an_updated_claude_35_sonnet_and_claude/), with improved performance across various benchmarks. The new Sonnet model reportedly shows [significant improvements in reasoning, code generation, and analytical capabilities](https://www.reddit.com/r/singularity/comments/1g9ihe9/claude_35_sonnet_reportedly_got_a_significant/).

- **Stability AI releases SD 3.5**: Stability AI [released Stable Diffusion 3.5](https://www.reddit.com/r/StableDiffusion/comments/1g9itzj/sd_35_large_released/), including a large 8 billion parameter model and a faster "turbo" version. Early testing suggests improvements in image quality and prompt adherence compared to previous versions.

- **Mochi 1 video generation model**: A new open-source video generation model called [Mochi 1 was announced](https://www.reddit.com/r/singularity/comments/1g9mvoy/introducing_mochi_1_preview_a_new_sota_in/), claiming state-of-the-art performance in motion quality and human rendering.

**AI Capabilities and Applications**

- **Claude's computer control abilities**: Anthropic demonstrated Claude's new ability to [control a computer and perform tasks like ordering pizza online](https://www.reddit.com/r/singularity/comments/1g9yi1p/claude_orders_some_pizza/). This capability allows Claude to interact with web interfaces and applications.

- **AI playing Paperclips game**: An experiment showed Claude [playing the Paperclips game autonomously](https://www.reddit.com/r/singularity/comments/1g9rqk5/holy_shit_claude_is_a_paperclip_maximizer/), demonstrating its ability to develop strategies and revise them based on new information.

- **OpenAI developing software automation tools**: Reports suggest OpenAI is [working on new products to automate complex software programming tasks](https://www.reddit.com/r/singularity/comments/1g9z0q0/exclusive_openai_under_pressure_from_anthropic_is/), potentially in response to competition from Anthropic.

**AI Development and Research**

- **Fixing LLM training bugs**: A researcher [fixed critical bugs affecting LLM training](https://www.reddit.com/r/singularity/comments/1g9pcbo/i_fixed_critical_bugs_which_affected_everyones/), particularly related to gradient accumulation, which could have impacted model quality and accuracy.

- **Pentagon's AI deepfake project**: The US Department of Defense is reportedly [seeking to create convincing AI-generated online personas](https://www.reddit.com/r/singularity/comments/1g9htmv/the_pentagon_wants_to_create_deepfake_internet/) for potential use in influence operations.

**AI Ethics and Security**

- **ByteDance intern fired for malicious code**: An intern at ByteDance was [fired for allegedly planting malicious code in AI models](https://www.reddit.com/r/OpenAI/comments/1g9ynfr/bytedance_intern_fired_for_planting_malicious/), raising concerns about AI security and access controls.

- **Claude's anti-jailbreaking measures**: The updated Claude 3.5 model appears to have [improved defenses against jailbreaking attempts](https://www.reddit.com/r/singularity/comments/1ga1oxz/claude_35_new_version_seems_to_be_trained_on/), demonstrating more sophisticated detection of potential manipulation.


---

# AI Discord Recap

> A summary of Summaries of Summaries by O1-preview

**Theme 1: Claude 3.5 Takes the AI World by Storm**

- [**Claude 3.5 Wows with 15% Boost in Coding Skills**](https://www.anthropic.com/news/3-5-models-and-computer-use): Communities across OpenAI and Unsloth AI are thrilled with **Claude 3.5 Sonnet**'s **15% performance gain** on SWE-bench, especially in coding tasks. The model's new **computer use** feature lets agents interact with computers like humans.
- [**OpenRouter Unveils Time-Travelling Claude 3.5 Versions**](https://openrouter.ai/anthropic/claude-3.5-sonnet-20240620): OpenRouter releases older **Claude 3.5 Sonnet** versions like **Claude 3.5 Sonnet (2024-06-20)**, giving users nostalgic access to previous iterations.
- **Developers Tinker with Claude's New Tricks**: Communities like OpenInterpreter explore integrating **Claude 3.5** with commands like `interpreter --os`, testing Anthropic's model and sharing insights.



**2\. Innovative AI Applications in Creative and Practical Domains**

- **DreamCut AI Revolutionizes Video Editing**: [**DreamCut AI**](http://dreamcut.ai) utilizes **Claude AI** to autonomously install and debug software, streamlining video editing tasks. Currently in early access, it bypasses traditional design phases, marking a shift towards AI-driven coding.
- **GeoGuessr AI Bot Automates Gameplay**: A [**YouTube tutorial**](https://www.youtube.com/watch?v=OyDfr0xIhss) demonstrates coding an AI bot that plays **GeoGuessr** using **Multimodal Vision LLMs** like **GPT-4o**, **Claude 3.5**, and **Gemini 1.5**. This project integrates **LangChain** for interactive game environment responses.
- **AI-Driven Customer Service Bots**: **Aider** introduces a **multi-agent concierge system** that combines tool calling, memory, and human collaboration for advanced customer service applications. This overhaul allows developers to iterate and enhance customer service bots more effectively.

**Theme 3: Shiny New Tools Promise AI Advancement**

- [**Anyscale's One-Kernel Wonder Aims to Turbocharge Inference**](https://x.com/detectiveenters/status/1752067011113546234): GPU MODE buzzes over **Anyscale** developing an inference engine using a single **CUDA kernel**, potentially outperforming traditional methods.
- [**CUDABench Calls All Coders to Benchmark LLMs**](https://docs.google.com/document/d/1ZNvShNH44zuy3LwbRdMigGsuCzO4i5Yl2fgAaSDynTg/edit?usp=sharing): PhD students invite the community to contribute to **CUDABench**, a benchmark to assess LLMs' CUDA code generation skills.
- [**Fast Apply Hits the Gas on Code Updates**](https://huggingface.co/Kortix/FastApply-7B-v1.0): **Fast Apply**, based on the **Qwen2.5 Coder Model**, revolutionizes coding by applying updates at blazing speeds of **340 tok/s** for the **1.5B model**.

**Theme 4: AI's Dark Side Sparks Concern**

- [**AI Blamed in Teen's Tragic Death**](https://www.nytimes.com/2024/10/23/technology/characterai-lawsuit-teen-suicide.html): Communities discuss a report of a 14-year-old's suicide linked to AI interactions, raising alarms about AI's impact on mental health.
- [**Character.AI Adds Safety Features Amid Tragedy**](https://blog.character.ai/community-safety-updates/): In response to the incident, **Character.AI** announces new safety updates to prevent future harm.
- **Debate Rages: Is AI Friend or Foe in Combating Loneliness?**: Latent Space members delve into whether AI eases loneliness or exacerbates isolation, with opinions divided on technology's role in mental well-being.

**Theme 5: ZK Proofs Give Users Control Over Their Data**

- [**ChatGPT Users Rejoice Over Chat History Ownership**](https://x.com/openblocklabs/status/1848805457290572199): OpenBlock's **Proof of ChatGPT** uses **ZK proofs** to let users own their chat logs, enhancing data training for open-source models.
- **Communities Embrace Data Sovereignty Movement**: Discussions in HuggingFace and Nous Research echo enthusiasm for data ownership, highlighting the importance of transparent and verifiable user data in AI development.

---

# PART 1: High level Discord summaries

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Stable Diffusion 3.5 Performance Debate**: Members discussed the fluctuating opinions on **Stable Diffusion 3.5**, noting current enthusiasm to test new features against alternatives.
  
  - This ongoing debate highlights a keen interest in improving generative model performance.
- **Automating CAD with LLMs**: A member proposed using **LLMs** and **RAG systems** to automate CAD file creation, seeking insights on system design approaches.
  
  - The discussion signified the community's commitment to integrating AI technologies for efficiency.
- **Explore the MIT AI Course**: A member shared a [YouTube playlist](https://www.youtube.com/playlist?list=PLUl4u3cNGP63gFHB6xb-kVBiQHYe_4hSi) featuring **MIT 6.034 Artificial Intelligence**, praising its foundational content.
  
  - *It's a must-see* for those diving into AI concepts, indicated by a strong community reaction.
- **Vintern-3B-beta Emerges**: **Vintern-3B-beta** model integrates over **10 million Vietnamese QnAs**, positioning itself as a competitor to LLaVA in the market.
  
  - This integration showcases advancements in dataset utilization for high-quality language model training.
- **ZK Proofs Enhance ChatGPT**: Utilizing **ZK proofs**, ChatGPT now enables users to own their chat history, enhancing verifiable training data for open-source models.
  
  - This marks a significant advance, as highlighted in a [demo tweet](https://x.com/openblocklabs/status/1848805457290572199).

 

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Claude 3.5 Sonnet improves performance**: Claude 3.5 Sonnet shows a **15%** performance gain on the SWE-bench and enhanced benchmarks, suggesting effective fine-tuning.
  
  - The integration of active learning techniques seems to enhance model efficacy for computer tasks.
- **Anthropic launches Computer Use Tool**: Anthropic introduced a novel tool to enable agents to execute tasks directly on a computer, aiming to reshape agent capabilities.
  
  - Utilizing advanced data processing, this tool aims to deliver a more seamless user experience for API consumers.
- **GPT-4 Upgrade Timeline in Limbo**: Enthusiasm swirls around an anticipated **GPT-4** upgrade, with mentions from several months ago still being the main reference point.
  
  - Access to GPTs for free users reportedly occurred roughly **4-5 months ago**.
- **Models show weak spatial sense**: Discussions revealed that models often exhibit weak **spatial sense**, effectively mimicking answers without true understanding.
  
  - This phenomenon resembles a child’s rote learning, suggesting deficiencies in deeper comprehension abilities.
- **Discussion on Realtime API performance**: Concerns arose that the **Realtime API** fails to follow system prompts as effectively as GPT-4o, disappointing many users.
  
  - Participants sought advice on adapting prompts to enhance interaction quality with the API.

 

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Claude 3.5 brings significant upgrades**: Anthropic launched the upgraded **Claude 3.5 Sonnet** and **Claude 3.5 Haiku** models, introducing advanced capabilities in coding tasks, including the **computer use** functionality now in beta.
  
  - This new ability allows developers to direct Claude to interact with computers similarly to human users, enhancing its utility in practical coding scenarios.
- **Kaggle struggles with PyTorch installation**: Users reported persistent `ImportError` issues on Kaggle related to different CUDA versions while trying to run PyTorch, prompting recommendations to downgrade to CUDA 12.1.
  
  - This workaround resolves compatibility issues and ensures smoother operation for existing library installations.
- **Challenges in model fine-tuning persist**: Users discussed models' tendency to repeat inputs during fine-tuning, suggesting that variations in system prompts could mitigate this overfitting.
  
  - Concerns that insufficient training examples may lead to reliance on a base model, resulting in repetitive outputs were raised among community members.
- **Fast Apply** revolutionizes coding tasks\*\*: **Fast Apply**, built on the **Qwen2.5 Coder Model**, operates efficiently, applying code updates without repetitive edits, significantly improving coding efficiency.
  
  - With performance metrics showing speeds of **340 tok/s** for the 1.5B model, this utility exemplifies how AI solutions are enhancing productivity in coding workflows.
- **Community pulls together for bug fixes**: Two significant pull requests were made: one for addressing import issues in the studio environment and another for correcting a **NoneType** error caused by a tokenizer bug.
  
  - These PRs underscore the community's proactive approach to refining Unsloth's functionality and resolving user-reported issues promptly.

 

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Stable Diffusion 3.5 gets neural upgrade**: Participants highlighted that **Stable Diffusion 3.5** adopts a neural network architecture akin to **Flux**, necessitating community-driven training efforts for optimization.
  
  - A consensus emerged on the importance of finetuning to fully leverage the capabilities of the new model.
- **Anime Art Prompting Secrets Revealed**: For generating anime art, users recommended using **SD 3.5** with precise prompts rather than relying on LoRAs for optimal results.
  
  - The community suggested focusing solely on stable diffusion 3.5 to boost image quality and avoid pitfalls associated with incorrect usage of LoRAs.
- **Image Quality Mixed Reports**: Users reported inconsistent image outputs, particularly when aligning prompts with wrong checkpoints or utilizing unsuitable LoRAs.
  
  - Discussion emphasized the necessity of ensuring model alignment with prompts to mitigate unsatisfactory generation results.
- **Automate Model Organization Now!**: There's an expressed need for a tool that can automatically sort and manage AI model files within folders, enhancing overall workflow efficiency.
  
  - Participants were encouraged to seek solutions in the server's technical support channel for potential automation tools.
- **Sharing Tools to Boost Generation Workflows**: Various tools and methods were discussed to enhance AI generation, with mentions of utility tools like **ComfyUI** and **fp8 models** for improved task management.
  
  - Participants shared personal experiences fostering community learning and exploring new tools to optimize their AI model experiences.

 

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider v0.60.0 Enhancements**: The release of **Aider v0.60.0** includes improved code editing, full support for **Sonnet 10/22**, and bug fixes that enhance user interactions and file handling.
  
  - Noteworthy features include model metadata management and Aider's contribution, writing **49% of the code** in this version, showcasing its productivity.
- **Claude 3.5 Sonnet Outperforms Previous Models**: Users report that the **Claude 3.5 Sonnet** model significantly outperforms the previous **O1 models**, achieving complex tasks with fewer prompts.
  
  - One user highlighted its ability to implement a VAD library into their codebase effectively, indicating a leap in usability.
- **DreamCut AI Revolutionizes Video Editing**: [DreamCut AI](http://dreamcut.ai) is built using **Claude AI**, taking **3 months and over 50k lines of code**, currently in early access for users to test its AI editing tools.
  
  - This initiative bypasses traditional design phases, indicating a shift towards AI-driven coding, as noted by community members.
- **Mistral API Authentication Troubles**: A user reported an *AuthenticationError* with the Mistral API in Aider but successfully resolved it by recreating their authentication key.
  
  - This incident reflects ongoing concerns over API access and authentication stability in the current setup of Mistral integrations.
- **Repo Map Enhancements Clarified**: Discussions on the **repo map** functionality reiterated its dependency on relevant code context, crucial for accurate code modifications tagged to identifiers.
  
  - As per Paul's clarification, the model evaluates identifiers based on their definitions and references, shaping effective editing paths.

 

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Claude 3.5 Sonnet Versions Released**: Older versions of **Claude 3.5 Sonnet** are now downloadable with timestamps: [Claude 3.5 Sonnet](https://openrouter.ai/anthropic/claude-3.5-sonnet-20240620) and [Claude 3.5 Sonnet: Beta](https://openrouter.ai/anthropic/claude-3.5-sonnet-20240620:beta).
  
  - These releases come from OpenRouter, providing users enhanced access to previous iterations.
- **Lumimaid v0.2 Enhancements**: The newly launched **Lumimaid v0.2** serves as a finetuned version of Llama 3.1 70B, offering a **significantly enhanced dataset** over Lumimaid v0.1, available [here](https://openrouter.ai/neversleep/llama-3.1-lumimaid-70b).
  
  - Users can expect improved performance due to the updates in dataset specifics.
- **Magnum v4 Showcases Unique Features**: **Magnum v4** has been released featuring prose quality replication akin to **Sonnet** and **Opus**, and can be accessed [here](https://openrouter.ai/anthracite-org/magnum-v4-72b).
  
  - This model continues the trend of enhancing the output quality in AI-generated text.
- **API Key Costs Differ on OpenRouter**: Users highlighted differences in API costs when utilizing OpenRouter versus direct provider keys, with some facing unexpected charges.
  
  - It’s crucial for users to understand how various models affect their total costs under OpenRouter.
- **Beta Access for Custom Provider Keys**: Custom provider keys are in beta, with requests for access managed through a specific Discord channel; self-signup isn’t an option.
  
  - Members can DM their **OpenRouter** email addresses for access, reflecting significant interest in these integrations.

 

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Downloading Models in LM Studio**: Users faced challenges in finding and downloading large models, particularly **Nvidia's 70B Nemotron** in LM Studio, necessitating new terminal command tips.
  
  - The change in search features forced users to employ specific keyboard shortcuts, complicating the model access process.
- **LLMs fall short in coding tasks**: Frustrations arose as models like **Mistral** and **Llama 3.2** struggled with accurate coding outputs, while **GPT-3.5** and **GPT-4** continued to perform significantly better.
  
  - Users began exploring alternative tools to supplement coding tasks due to this consensus on performance inadequacies.
- **Exploring Model Quantization Options**: Discussions highlighted diverse preferences for quantization methods (Q2, Q4, Q8) and their impact on model performance, especially regarding optimal bit compression.
  
  - While caution against Q2 was advised, some users remarked on larger models showing better performance with lower bit quantization.
- **Ryzen AI gets attention for NPU support**: A query arose on configuring LM Studio to utilize the **NPU of Ryzen processors**, revealing ongoing challenges with implementation and functionality.
  
  - Clarification emerged that **only the Ascend NPU** receives support in **llama.cpp**, leaving Ryzen's NPU functionality still uncertain.
- **AMD vs. Nvidia: The GPU Showdown**: When comparing the **RX 7900 XTX** and **RTX 3090**, users highlighted the importance of **CUDA support** for optimal LLM performance, favoring Nvidia's card.
  
  - Mixed effectiveness reports on **multi-GPU** setups across brands surfaced, especially regarding support from the recently updated **ROCm 6.1.3**.

 

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Users Complain About New Sonnet 3.5**: Multiple users expressed dissatisfaction with the **Sonnet 3.5 model**, noting a decrease in content output, particularly for academic writing tasks.
  
  - Concerns were raised about the removal of the older model, which was regarded as superior for various use cases.
- **Web Search Integration Issues Persist**: There was a reported bug where the preprompt in Spaces fails when web search is enabled, causing frustration among users.
  
  - Users indicated that this issue has continued without resolution, with the team acknowledging the need for a fix.
- **Account Credits Still Not Transferred**: A user reported that their **account credits** have not been transferred, despite multiple inquiries to support.
  
  - *No response from support for the past three days* has resulted in heightened frustration.
- **Advanced AI-Driven Fact-Checking Explored**: A collection on [AI-driven fact-checking](https://www.perplexity.ai/collections/advanced-ai-driven-fact-checki-a3cMcPR.QsKkCRZ79UKFLQ) discusses techniques like source credibility assessment.
  
  - It emphasizes the necessity for **transparency** and **human oversight** to effectively combat misinformation.
- **Claude Computer Use Model Raises Alarms for RPA**: A post on [Claude's Computer Control capabilities](https://www.perplexity.ai/page/claude-s-computer-control-capa-E_O4xa7VSWOi3lGtOWnnMw) suggests possible risks for **Robotic Process Automation (RPA)**.
  
  - Experts warn that this innovation could pose significant challenges to existing workflows.

 

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **LLM Activations Quantization Debate**: A discussion emerged on whether **activations in LLMs** sensitive to input variations should be aggressively quantized or maintained for higher precision.
  
  - *This raises concerns about modeling performance* and the trade-offs of precision in quantization.
- **Precision Worries with bf16**: Concerns were shared about **bf16** potentially causing **canceled updates** due to precision issues during multiple gradient accumulations.
  
  - *Precision is crucial*, especially when it impacts model training stability.
- **Anyscale's Single Kernel Inference**: An update about **Anyscale** developing an inference engine using a single **CUDA kernel** was shared, inviting opinions on its efficiency.
  
  - *There's excitement about potentially leapfrogging traditional inference methods*.
- **CUDABench Proposal**: PhD students presented a proposal for **CUDABench**, a benchmark to assess LLMs' CUDA code generation abilities, encouraging community contributions.
  
  - It aims to establish compatibility across various DSLs while focusing on torch inline CUDA kernels.
- **Monkey Patching CrossEntropy Challenges**: New challenges emerged with the monkey patching strategy for **CrossEntropyLoss** in transformers, particularly with the latest **GA patch version**.
  
  - The original **CrossEntropy** function can be reviewed [here](https://github.com/huggingface/transformers/blob/049682a5a63042f087fb45ff128bfe281b2ff98b/src/transformers/loss/loss_utils.py#L26).

 

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Hermes 70B API launched on Hyperbolic**: The **Hermes 70B API** is now available on Hyperbolic, providing greater access to large language models for developers and businesses. For more details, check out the announcement [here](https://x.com/hyperbolic_labs/status/1849130421885514231?s=46).
  
  - This launch marks a significant step towards making powerful AI tools more accessible to everyone.
- **Nou Research's Forge Project Sparks Enthusiasm**: Members expressed their enthusiasm for the project 'Forge', highlighted in a [YouTube video](https://www.youtube.com/watch?v=zmnzW0r_g8k&list=PLjOo65uE) featuring Nous Research co-founder Karan. A discussion followed about knowledge graph implementation related to the project.
  
  - The expectations for Forge’s capabilities in leveraging advanced datasets have been high among members of the community.
- **ZK Technology Revolutionizes Proof Generation**: The latest application from OpenBlock’s Universal Data Protocol (UDP) empowers ChatGPT users to own their chat history while enhancing the availability of verifiable training data for open-source models. This approach marks a significant step in improving data provenance and interoperability in AI training.
  
  - A member clarified that ZK proofs take a few seconds on the server-side, with some UDP proofs now taking less than a second due to advancements in infrastructure from @zkemail; check this out [here](https://x.com/paulsengh/status/1846657020868677931).
- **Claude Sees Automation Improvements**: Claude includes a system prompt addition that corrects the 'misguided attention' issue, enhancing its contextual understanding. *Claude also endeavors to clarify puzzle constraints, yet sometimes misinterprets questions due to oversight.*
  
  - Users noticed enhancements in Claude’s self-reflection abilities, with responses becoming more refined when addressing logical puzzles.
- **Dynamics of AI Role-Playing Explored**: The dynamics of AI role-playing were explored, particularly how system prompts influence the responses of AI models in various scenarios. Members discussed the potential for models to exhibit chaotic behavior if instructed in certain ways, challenging the idea of inherent censorship.
  
  - This ongoing dialogue highlights the intricate relationship between prompt engineering and AI behavior.

 

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Chess Players Explain Moves**: Most top chess players can articulate the *motivation* behind engine moves, but their skill in ranking lines in complex positions remains in question.
  
  - The ongoing inquiry into what defines an ideal move for humans versus engines keeps the community engaged.
- **Controversy in Chess with Cheating Claims**: A former world champion accused a popular streamer of cheating based on their move explanations during live commentary.
  
  - This incident underscores the pressures commentators face and ignites new debates about move validity.
- **Accuracy of LLMs’ Self-Explanations**: Concerns were voiced regarding the accuracy of self-explanations from LLMs when they lack contextual understanding.
  
  - The community is exploring how improved training data could enhance these explanations.
- **Molmo Vision Models on the Horizon**: The **Molmo** project plans to release open vision-language models trained on the **PixMo** dataset featuring multiple checkpoints.
  
  - These models aim for state-of-the-art performance in the multimodal arena while remaining fully open-source.
- **Learning DINOv2 Through Research**: A member requested resources to grasp **DINOv2**, leading others to share a pertinent [research paper](https://arxiv.org/abs/2304.07193) that details its methodology.
  
  - The paper provides insights into the foundational aspects of DINOv2 as advanced by leading experts.

 

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Anthropic Mouse Generator Shows Off**: A colleague showcased the **Anthropic Mouse Generator**, impressively installing and debugging software autonomously, but it still requires specific instructions to function.
  
  - Critics noted it cannot perform tasks like playing chess without guidance, highlighting the limitations of current AI agents.
- **Ideogram Canvas Threatens Canva**: Discussions about **Ideogram Canvas** revealed its innovative features such as **Magic Fill** and **Extend**, which enable easy image editing and combination.
  
  - Participants indicated it could rival existing tools like Canva due to its superior capabilities, sparking competitive concerns.
- **AI's Impact on Loneliness Discussed**: A tragic event involved a 14-year-old's suicide, igniting dialogue on AI's impact on loneliness, with concerns about mental health and technology's role.
  
  - Participants debated whether AI could connect people or whether it intensifies isolation, sharing varied perspectives on its effectiveness.
- **Speculative Decoding in vLLM Boosts Speed**: A recent blog post detailed enhancements in **speculative decoding** in **vLLM**, aimed at accelerating token generation through small and large models.
  
  - This technique seeks to improve performance and integrate new methodologies for optimizing AI functionality, as highlighted by [this blog](https://blog.vllm.ai/2024/10/17/spec-decode.html).
- **Introducing New Meeting Automation Tools**: The launch of **agent.exe** allows users to control computers via **Claude 3.5 Sonnet**, marking a significant advancement in meeting automation tools.
  
  - Expectations for increased automation and efficiency are set for **2025**, with [agent.exe on GitHub](https://x.com/corbtt/status/1849124800838713844?s=46) already attracting interest.

 

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Join the Llama Impact Hackathon for AI Solutions**: Participate in the 3-day [Llama Impact Hackathon](https://t.co/G01c8eIN1j) in San Francisco from November 8-10, offering a **$15,000** prize pool, including a special **$1000** for the best use of LlamaIndex.
  
  - This event provides both in-person and online options for building AI solutions using **Meta's Llama 3.2 models**.
- **Box AI and LlamaIndex Work Together Seamlessly**: Utilize **Box AI** to query documents without downloading and extract structured data from unstructured content while integrating it with LlamaIndex agents, detailed in [this article](https://t.co/M9f81GiMGp).
  
  - This integration enhances workflows, making document handling easier for users.
- **Build Advanced Customer Service Bots**: A recent update allows the creation of a **multi-agent concierge system** that combines tool calling, memory, and human collaboration for customer service applications.
  
  - This overhaul helps developers iterate on customer service bots more effectively, as shared by [Logan Markewich](https://t.co/PWshlAyeKV).
- **Persistent Context in Workflows**: A discussion arose on enabling **Context** to persist across multiple runs of workflows, with examples using **JsonSerializer** for serialization.
  
  - This method allows users to resume their workflows later without losing context, addressing a common pain point.
- **Migrating to Anthropic LLM**: Users faced challenges replacing ChatGPT with the **Anthropic LLM**, particularly concerning OpenAI API key prompts.
  
  - Advice included the necessity of a local embedding model to eliminate dependency on OpenAI’s services.

 

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Clarification on Tensor int64/uint64 support**: Discussion clarified that **Tensors** now support **int64/uint64**, as confirmed by examining `dtype.py`.
  
  - This clarification came up amidst talks on implementing **SHA3**, highlighting evolving capabilities within tinygrad.
- **Action Chunking Transformers Training Takes Ages**: To train **Action Chunking Transformers** with **55 million** parameters takes **two days** without JIT, leading to questions about performance enhancements.
  
  - Members expressed frustration over slow inference times and repeated **loss parameter** issues during JIT training.
- **TinyJIT Loss Parameter Printing Confusion**: Users grappled with printing the loss in JIT functions, debating the use of `.item()` and its effects on displaying values accurately.
  
  - Staying away from non-Tensor returns was advised to prevent undesired impacts on JIT execution.
- **Improving Training Time with BEAM Settings**: A tip suggested running with `BEAM=2` to potentially enhance performance and speed up kernel runs during lengthy training sessions.
  
  - Feedback indicated that this approach has already yielded quicker results in training practices.
- **Interest in Reverse Engineering AI Accelerator Byte Code**: A user sought advice on methodologies for reverse engineering byte code from an **AI accelerator**, sparking wider interest in the community.
  
  - Members discussed tools and frameworks that could aid in initiating the reverse engineering process.

 

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Fun with Claude AI**: A member reported that using **Claude AI** was a fun experience and hinted at sharing more examples shortly.
  
  - This suggests that further details on its features and capabilities might be coming soon.
- **Deep Dive into Continuous Pretraining**: Questions arose about whether **GPT-4o** was pretrained with a **200k vocabulary tokenizer** from scratch or if it continued after switching from a **100k** tokenizer.
  
  - Concerns were voiced about the messy nature of mid-training, indicating challenges in tracking such transitions.
- **Character.AI Expresses Condolences**: **Character.AI** issued condolences regarding a tragic user incident, emphasizing new safety features available [here](https://blog.character.ai/community-safety-updates/).
  
  - A member shared a [New York Times article](https://www.nytimes.com/2024/10/23/technology/characterai-lawsuit-teen-suicide.html) that further contextualizes the situation.
- **Anthropic's Shift Towards B2B**: **Anthropic** is evolving into a B2B company, contrasting with **OpenAI**'s focus on consumer applications, particularly regarding engaging versus mundane tasks.
  
  - The discussion emphasized consumer preferences for enjoyable activities over automation's potential for boring tasks, such as shopping.
- **Microsoft's Engaging AI Demonstrations**: **Microsoft's** playful applications of AI, like gameplay automation in **Minecraft**, stand in contrast to **Anthropic's** focus on routine tasks [view here](https://youtu.be/TLg2KWY2J5c?si=dRkIA2t1Y0F61KgP).
  
  - This highlights divergent strategies in the AI landscape, reflecting different target audiences and goals.

 

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Screenpipe Builds Buzz**: Members praised the usefulness of [Screenpipe](https://github.com/OpenInterpreter/open-interpreter/blob/development/examples/screenpipe.ipynb) for managing build logs, showcasing its potential for developers looking for efficient logging solutions.
  
  - One user highlighted the major impact of having clear and organized build logs in their development workflow.
- **Claude 3.5 Models Evolve**: Anthropic introduced the **Claude 3.5 Sonnet** model, boasting significant coding enhancements and a new **computer use** capability available in public beta, allowing AI to interact with user interfaces more naturally.
  
  - However, constant screenshot capturing is required, leading to concerns about the model's efficiency and operational costs; [more details here](https://www.anthropic.com/news/3-5-models-and-computer-use).
- **Skepticism on Open Interpreter's Roadmap**: Members discussed the roadmap for **Open Interpreter**, asserting its unique capabilities distinguish it from mainstream AI offerings.
  
  - Some skeptics expressed doubts about competing against established models, while others underscored the significance of community-driven development.
- **Navigating AI Screen Interaction Challenges**: Concerns emerged over the inefficiencies of using screenshots for AI input, leading to suggestions for directly extracting necessary data points from applications.
  
  - Members recognized the need for enhanced data processing methods to circumvent existing limitations in screenshot dependency.
- **Call for Testing New Anthropic Integration**: A member introduced the `interpreter --os` command for integrating with Anthropic's model, urging others to assist in testing the feature prior to its final release.
  
  - Testing indicated that increasing screen size and text clarity could help minimize error rates during model usage.

 

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere API Trials Offer Free Access**: Cohere provides a [trial API key](https://docs.cohere.com/docs/rate-limits) allowing free access to all models with rate limits of **20 calls per minute** during the trial, improving to **500 calls per minute** with production keys.
  
  - This setup allows engineers to explore various models before committing to production environments.
- **Emerging Multimodal Command Models**: Discussions sparked interest in a **multimodal Command model**, suggesting a **Global connection** feature that integrates different modes of interaction.
  
  - This reflects a budding curiosity about advanced model capabilities and their potential applications.
- **Agentic Builder Day on November 23rd**: OpenSesame is hosting an **Agentic Builder Day** on November 23rd, inviting developers to participate in a mini AI Agent hackathon using **Cohere Models**, with [applications currently open](https://www.opensesame.dev/hack).
  
  - The event aims to foster collaboration and competition among developers interested in AI agents.
- **Ollama Mistral Performance Concerns**: Members expressed issues with **Ollama Mistral**, noting performance hiccups and hallucination tendencies that complicate their projects.
  
  - One user linked to their [GitHub gist](https://gist.github.com/pleabargain/8b3f1641ef727cc114ac389cbc1b354b) detailing their methodology for effective prompt generation despite these challenges.
- **Tool Calls and Cohere V2 API Errors**: Users reported **internal server errors** with tool calls in the Cohere V2 API, particularly highlighting the missing **tool_plan field** that caused some issues.
  
  - Reference was made to the [Cohere documentation](https://docs.cohere.com/docs/tool-use#step-2) for clarifications on proper tool integrations.

 

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Enthusiasm for stdlib Discussions**: A member expressed excitement about joining **stdlib contributor meetings** after catching up on the last community discussion.
  
  - This enthusiasm garnered positive reactions from others, encouraging participation in the conversations.
- **Serial Communication Woes in Mojo**: A user sought guidance on implementing **serial communication** over a port in **Mojo**; current support is limited to what's available in **libc**.
  
  - This indicates a necessity for further enhancements in Mojo's communication capabilities.
- **Debate on C/C++ Support in Mojo**: Discussion emerged on the existence of **C/C++ support** in Mojo, highlighting its potential benefits.
  
  - However, opinions were divided on the practical application of this support for users.
- **C API Launch Announcement for MAX Engine**: The **C API** is now available for the **MAX Engine**, though there are no immediate plans for the **graph API** integration.
  
  - An assurance was given that updates regarding the graph API will be communicated if the situation changes.
- **Exploring Graph API with C**: A member noted the possibility of using C to build a **graph builder API**, suggesting alternative approaches alongside **Mojo**.
  
  - This opens up discussions for potential collaborations across programming languages.

 

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **TorchTune config flops with .yaml**: A member flagged that using **.yaml** file extensions in **TorchTune** run commands causes confusion by implying a local config.
  
  - They noted that debugging can be frustrating without sufficient error messages.
- **Multi-GPU testing raises questions**: One user asked about testing capabilities on **2 GPUs**, reflecting a common concern.
  
  - Another user mentioned issues with error messages when running scripts on **1 GPU** and **2 GPUs** with **lora_finetune_distributed**.
- **Fine-tuning with TorchTune confirmed**: Response to a fine-tuning query for a **custom Llama** model confirmed that **TorchTune** offers flexibility for customization.
  
  - Members were encouraged to engage further in discussions about custom components for better support.
- **Linters and pre-commit hooks bugged**: Members reported issues with **linters and pre-commit hooks**, indicating they weren’t functioning as expected.
  
  - To bypass a line, both `# noqa` and `# fmt: on ... #fmt: off` are required, which is seen as unusually complicated.
- **CI chaos in PR #1868**: A member revealed strange behavior with the **CI** for PR **#1868**, seeking assistance to address ongoing problems.
  
  - Inquiries about the resolution of a CI issue indicated that, thankfully, it should now be **fixed**.

 

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **Help Shape a Tool for Developers**: A member shared a [survey link](https://forms.gle/Roi1U5ynVwLtQ3S46) targeting developers with insights on challenges in bringing ideas to fruition, taking approximately **5-7 minutes** to complete.
  
  - The survey explores how often developers generate ideas, the obstacles they face, and their interest in solutions for simpler project realization.
- **AI Impact Study for Developers**: A call for developers to partake in a Master’s study assessing the impact of AI tools on software engineering is live, with participants perhaps winning a **$200NZD gift card** by filling out a short questionnaire [here](https://auckland.au1.qualtrics.com/jfe/form/SV_0uf2q5Ie7V3gpvM?Source=43).
  
  - This study aims to gather valuable data on the integration of AI in engineering workflows while offering incentives for participation.
- **Unlock Funding with AI Tool**: An AI-powered platform launched to help users find funding by matching them with relevant investors, offering a **free Startup Accelerator pack** to the first **200** waitlist sign-ups, with only **62** spots left.
  
  - Interested individuals are prompted to [Sign Up Now](https://www.aloangels.me/) to accelerate their startup dreams with enhanced search capabilities.
- **Building an AI GeoGuessr Player**: A new [YouTube tutorial](https://www.youtube.com/watch?v=OyDfr0xIhss) showcases coding an AI bot that autonomously plays **GeoGuessr** using **Multimodal Vision LLMs** like **GPT-4o**, **Claude 3.5**, and **Gemini 1.5**.
  
  - The tutorial involves **Python programming** and the use of **LangChain** to allow the bot to interact with the game environment effectively.
- **Inquiry about Manila Developers**: A member inquired if anyone is located in **Manila**, hinting at a desire to foster connections among local developers.
  
  - This inquiry may create opportunities for community building or potential collaborations within the Manila tech scene.

 

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **World's Most Advanced Workflow System in Progress**: A member announced plans for the **world's most advanced workflow system** with a **live demonstration** scheduled for Monday to showcase its operations and the **upgrade process**.
  
  - The session aims to provide a deep dive into system functionality, with an emphasis on discussing planned enhancements and upcoming features.
- **DSPy sets ambitious funding goals**: Following CrewAI's success in securing **$18M**, a member proposed that **DSPy** should target a minimum of **$50M**, expressing eagerness to join early-stage as employee number 5 or 10.
  
  - *What are we waiting for?* enlivened the discussion, emphasizing a call to action for immediate funding efforts.
- **Metrics for Effective Synthetic Data Generation**: A member explored the potential of using **DSPy** to generate synthetic data for QA purposes based on textual input, raising questions about suitable metrics.
  
  - Responses suggested leveraging an **LLM as a judge** with established criteria for assessing the open-ended generation where no ground truth exists.
- **Groundedness as a Metric in Synthetic Data**: In synthetic data discussions, a member suggested that **ground truth** would derive from the text utilized in generation, indicating groundedness as a key metric.
  
  - They expressed gratitude for the collaborative insights shared, highlighting a spirit of engagement among members on the topic.

 

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **LLM Agents MOOC Signup Confusion**: Several members reported not receiving confirmation emails after submitting their [LLM Agents MOOC Signup Form](https://link-to-signup-form), leading to uncertainty about their application status.
  
  - This lack of feedback has raised concerns about the signup process among users who expected formal acceptance notifications.
- **Hackathon Project Codes Must Be Open Source**: During the Hackathon, members confirmed the requirement to make their project codes **100% open source**, a stipulation for participating in final presentations.
  
  - This emphasis on code transparency aligns with the hackathon's goals to foster collaborative development among participants.
- **Demand for Agent Creation Tutorials**: A participant inquired about tutorials for creating agents from scratch without relying on external platforms, highlighting a need for accessible educational resources.
  
  - This interest underscores the community's desire for self-sufficiency in agent development workflows.

 

---

## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **Axolotl Discord for Configurations**: Utilize the 🦎 Axolotl Discord channel for sharing and finding configurations tailored to your use case, complete with an example folder on GitHub. Check out the [Discussions tab](https://github.com/axolotl-ai-cloud/axolotl/discussions) for insights and shared use cases.
  
  - *Leverage community efforts* to refine your configurations and adapt existing setups to better suit your projects.
- **Maximize Your Prompts with LangSmith**: Explore the 🛠️ LangSmith Prompt Hub that offers an extensive collection of prompts ideal for various models and use cases, enhancing your prompt engineering skills. Visit the [Amazing Public Datasets repository](https://github.com/awesomedata/awesome-public-datasets) for quality datasets.
  
  - *Share your own prompts* and discover new ideas to foster collaboration and better model performance.
- **Kaggle Solutions for Data Competitions**: Check out *The Most Comprehensive List of Kaggle Solutions and Ideas* for a wealth of insights on competitive data science. Access the full collection on GitHub [here](https://github.com/faridrashidi/kaggle-solutions).
  
  - This resource serves as a goldmine for data engineers looking to enhance their methodologies and strategies.
- **Hugging Face’s Recipes for Model Alignment**: Find robust recipes on Hugging Face to align language models with both human and AI preferences, essential for continued fine-tuning. Discover these valuable resources [here](https://github.com/huggingface/alignment-handbook/tree/main/recipes).
  
  - *Align your models* effectively with insights drawn from community best practices.
- **Introducing New Discord Bot for Easy Message Scraping**: A newly created Discord bot aims to streamline message scraping from the channel and needs assistance with inviting members to the bot. Interested users can invite the bot via this [link](https://discord.com/oauth2/authorize?client_id=1298625427375656980&response_type=code&redirect_uri=https%3A%2F%2Fc123ian.github.io%2F&scope=messages.read).
  
  - *Get involved* to enhance your Discord experience and potentially automate collection of valuable discussions.

 

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **2.5.0 Introduces Experimental Triton FA Support for gfx1100**: With **version 2.5.0**, users can enable **experimental Triton Flash Attention (FA)** for **gfx1100** by setting the environment variable `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`. Further details are available in [this GitHub issue](https://github.com/ROCm/aotriton/issues/16#issuecomment-2346675491).
  
  - A **UserWarning** indicates that Flash Attention support on **Navi31 GPU** remains experimental.
- **Mixtral vs. Llama 3.2: The Usage Debate**: Discussion arose regarding the viability of using **Mixtral** in light of advancements in **Llama 3.2**. Community members are examining the strengths and weaknesses of both models to determine which should take precedence.
  
  - This inquiry highlights the evolving landscape in model selection, reflecting on performance metrics and suitability for specific tasks.

 

---

## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **Empty Score Report on Model Evaluation**: A user reported that executing `bfcl evaluate --model mynewmodel --test-category ast` resulted in an empty score report at **0/0** after registering a new model handler in `handler_map.py`.
  
  - Another member recommended ensuring that the `bfcl generate ...` command was run prior to evaluation, highlighting the dependency for accurate score results.
- **Importance of Generating Models Before Evaluation**: Discussion emphasized running the `bfcl generate` command before model evaluation to avoid empty reports during testing.
  
  - This underlines that the absence of model generation could directly affect the validity of evaluation results.

 

---

# PART 2: Detailed by-Channel summaries and links

{% if medium == 'web' %}

 

### **HuggingFace ▷ #**[**general**](https://discord.com/channels/879548962464493619/879548962464493622/1298364346061029496) (547 messages🔥🔥🔥):

> - `Stable Diffusion 3.5`
> - `Llama Models`
> - `Automating CAD Document Creation`
> - `Music Generation Models`
> - `Benchmarking AI Models`

- **Discussion on Stable Diffusion 3.5 Performance**: Members discussed the fluctuating opinions on Stable Diffusion over time, highlighting the ongoing debate about its performance and updates.
  
  - It was noted that users are currently eager to test the new features and compare results with alternative models.
- **Implementing AI in CAD**: A user is exploring the use of LLMs and RAG systems to automate the creation of CAD documents, emphasizing the need for structured outputs from specifications.
  
  - There's a suggestion that running smaller models could be a practical starting point before advancing to larger, more complex models.
- **Music Generation Models**: Users discussed various models for music generation, with recommendations for Musicgen, stable-audio, and audioldm for instrumental music.
  
  - For music with lyrics, SongCrafter was mentioned as an option, but expectations should be adjusted regarding quality.
- **Benchmarking and Model Performance**: The reliability of benchmarks in AI models was questioned, particularly regarding LLMs where specific quality varies widely based on use cases.
  
  - Users mentioned that personal testing is often the best approach to evaluate model performance.
- **Running Models on Mobile Devices**: A user asked about the feasibility of running specific LLMs on mobile devices, resulting in suggestions for suitable online proxies to avoid local processing.
  
  - The discussion included humor regarding the content generated by uncensored models and their appropriateness.

**Links mentioned**:

- [Suno AI](https://suno.com/about): We are building a future where anyone can make great music. No instrument needed, just imagination. From your mind to music.
- [Dev Board | Coral](https://coral.ai/products/dev-board/): A development board to quickly prototype on-device ML products. Scale from prototype to production with a removable system-on-module (SOM).
- [Llama 3.2 3B Uncensored Chat - a Hugging Face Space by chuanli11](https://huggingface.co/spaces/chuanli11/Chat-Llama-3.2-3B-Instruct-uncensored): no description found
- [The Simpsons Homer GIF - The Simpsons Homer Exiting - Discover & Share GIFs](https://tenor.com/view/the-simpsons-homer-exiting-uncomfortable-leaving-now-gif-12755201945629685724): Click to view the GIF
- [AutoMatch: A Large-scale Audio Beat Matching Benchmark for Boosting Deep Learning Assistant Video Editing](https://arxiv.org/abs/2303.01884): The explosion of short videos has dramatically reshaped the manners people socialize, yielding a new trend for daily sharing and access to the latest information. These rich video resources, on the on...
- [Diffusion for World Modeling: Visual Details Matter in Atari (DIAMOND) 💎](https://diamond-wm.github.io/): Diffusion for World Modeling: Visual Details Matter in Atari (DIAMOND) 💎 Webpage
- [no title found](https://jsplot.dcford.org.uk/): no description found
- [Nick088/FaceFusion · 🚩 Report: Legal issue(s)](https://huggingface.co/spaces/Nick088/FaceFusion/discussions/8): no description found
- [RaveDJ - Music Mixer](https://rave.dj/)): Use AI to mix any songs together with a single click
- [We Breaking Bad GIF - WE BREAKING BAD WALTER WHITE - Discover & Share GIFs](https://tenor.com/view/we-breaking-bad-walter-white-gif-14928204287258878513): Click to view the GIF
- [Pyplot tutorial — Matplotlib 3.9.2 documentation](https://matplotlib.org/stable/tutorials/pyplot.html): no description found
- [Happy Birthday GIF - Happy Birthday - Discover & Share GIFs](https://tenor.com/view/happy-birthday-gif-27707596): Click to view the GIF
- [Long Time GIF - Long Time Age - Discover & Share GIFs](https://tenor.com/view/long-time-age-old-finally-gif-12981705): Click to view the GIF
- [Models - Hugging Face](https://huggingface.co/models?pipeline_tag=sum): no description found
- [Downloading files](https://huggingface.co/docs/huggingface_hub/main/en/package_reference/file_download#huggingface_hub.snapshot_download): no description found
- [llama.cpp/grammars/README.md at master · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md): LLM inference in C/C++. Contribute to ggerganov/llama.cpp development by creating an account on GitHub.
- [Models - Hugging Face](https://huggingface.co/models?pipeline_tag=summarization): no description found
- [ML Inference survey](https://forms.gle/vS8DrPdaKpuaGgrk8): no description found
- [Oxidaksi vs. Unglued - Ounk](https://www.youtube.com/watch?v=PFKFNtUDj8g): #MASHUP #PSY #DNB #MUSIC #SPEEDSOUNDCreated with Rave.djCopyright: ©2021 Zoe Love
- [GitHub - teticio/audio-diffusion: Apply diffusion models using the new Hugging Face diffusers package to synthesize music instead of images.](https://github.com/teticio/audio-diffusion): Apply diffusion models using the new Hugging Face diffusers package to synthesize music instead of images. - teticio/audio-diffusion
- [GitHub - noamgat/lm-format-enforcer: Enforce the output format (JSON Schema, Regex etc) of a language model](https://github.com/noamgat/lm-format-enforcer): Enforce the output format (JSON Schema, Regex etc) of a language model - noamgat/lm-format-enforcer
- [Audio Diffusion](https://huggingface.co/docs/diffusers/main/en/api/pipelines/audio_diffusion): no description found

---

### **HuggingFace ▷ #**[**today-im-learning**](https://discord.com/channels/879548962464493619/898619964095860757/1298377865200668744) (15 messages🔥):

> - `MIT AI Course`
> - `Manim Animation Engine`
> - `Learning LLM Neural Networks`
> - `Statistics and Linear Algebra for AI`

- **Explore the MIT AI Course**: A member shared a [YouTube playlist](https://www.youtube.com/playlist?list=PLUl4u3cNGP63gFHB6xb-kVBiQHYe_4hSi) titled 'MIT 6.034 Artificial Intelligence, Fall 2010.' This comprehensive course features insights from Prof. Patrick Winston and covers foundational AI concepts.
  
  - Another member expressed enthusiasm, noting the course as a *must-see* for those interested in AI.
- **Manim: The Animation Engine for Math Videos**: A member revealed that the animations were created using [Manim](https://github.com/3b1b/manim), a custom animation engine designed for explanatory math videos. This GitHub project encourages contributions and showcases the underlying technology for creating these videos.
  
  - A user humorously acknowledged the effort with a reaction, indicating the community's appreciation of such tools.
- **Advice on Learning LLM Neural Networks**: An inquiry was made about good resources on learning basic LLM neural network objects like those from Torch or TikToken. The community offered support, showing readiness to engage and assist in the learning process.
  
  - Members expressed camaraderie and willingness to help each other navigate the complexities of learning these technologies.
- **Foundational Topics in Linalg and Statistics**: After gaining theoretical knowledge, a member sought guidance on essential topics in linear algebra and statistics. Recommendations included mastering matrices, symbolic logic, and understanding the Nash Equilibrium, which holds importance in statistics and game theory.
  
  - This reflects the community's focus on not just theoretical but also practical aspects of AI learning.

**Links mentioned**:

- [MIT 6.034 Artificial Intelligence, Fall 2010](https://www.youtube.com/playlist?list=PLUl4u3cNGP63gFHB6xb-kVBiQHYe_4hSi): View the complete course: http://ocw.mit.edu/6-034F10 Instructor: Patrick Winston In these lectures, Prof. Patrick Winston introduces the 6.034 material from...
- [GitHub - 3b1b/manim: Animation engine for explanatory math videos](https://github.com/3b1b/manim): Animation engine for explanatory math videos. Contribute to 3b1b/manim development by creating an account on GitHub.

---

### **HuggingFace ▷ #**[**cool-finds**](https://discord.com/channels/879548962464493619/897390579145637909/1298473384639987722) (18 messages🔥):

> - `Quantum Computing Exploration`
> - `Vintern-3B-beta Model Development`
> - `Comparison Tool for LLM Services`
> - `Openblock ZK Proofs for ChatGPT`
> - `LLM License Clauses Criticism`

- **Diverse Lo-fi Music for Every Mood**: A rich list of **lo-fi music themes** was shared, including calming sounds for meditation, energetic tunes for retail therapy, and more, all of which showcase innovative ideas like underwater basket weaving.
  
  - *It's a playful collection that invites creativity around music prompts and anxiety relief.*
- **Vintern-3B-beta emerges as a contender**: The **Vintern-3B-beta** model has successfully integrated multiple datasets, including over **10 million Vietnamese QnAs** to battle existing competitors like LLaVA.
  
  - This model showcases significant advancements in training processes, proving beneficial for users looking for high-quality language model options.
- **Free Tool to Compare LLM Services Launched**: A user developed a free tool that allows comparison of **LLM pricing and performance** across numerous providers, including OpenAI and Google, leading to useful insights into available options.
  
  - The tool emphasizes that higher prices don't always equate to better quality, challenging users to explore various service providers.
- **Openblock Innovates Chat History Ownership**: Openblock introduced a feature called **Proof of ChatGPT**, utilizing ZK proofs to enable users to control their chat histories comprehensively.
  
  - This method marks a substantial advancement in user data sovereignty, addressing concerns around data ownership in open-source domains.
- **Criticism of Quirky License Clauses**: A concern was raised about the inclusion of **quirky clauses in software licenses**, which can hinder practical usability for serious projects.
  
  - *The discussion highlights a growing frustration within the community regarding unnecessary complexities in licensing agreements.*

**Links mentioned**:

- [Tweet from undefined](https://x.com/open): no description found
- [Tweet from OpenBlock (@openblocklabs)](https://x.com/openblocklabs/status/1848805457290572199): 1/ Introducing Proof of ChatGPT, the latest application built on OpenBlock’s Universal Data Protocol (UDP). This Data Proof empowers users to take ownership of their LLM chat history, marking a signi...
- [Tweet from vik (@vikhyatk)](https://x.com/vikhyatk/status/1848893472310464724): i find it really annoying when people put "quirky" clauses like this in licenses... makes any serious usage of your thing impossible. stop doing this. be better.
- [5CD-AI/Vintern-3B-beta · Hugging Face](https://huggingface.co/5CD-AI/Vintern-3B-beta): no description found
- [vikhyatk/lofi · Datasets at Hugging Face](https://huggingface.co/datasets/vikhyatk/lofi): no description found
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/1g9js22/i_built_an_llm_comparison_tool_youre_probably/): no description found

---

### **HuggingFace ▷ #**[**i-made-this**](https://discord.com/channels/879548962464493619/897390720388825149/1298368557326794793) (22 messages🔥):

> - `ZK Proofs for ChatGPT`
> - `SNES Music Diffusion Model`
> - `Concerns about GitHub Organization`
> - `Fourier Dual Diffusion Repository`

- **ZK Proofs Empower ChatGPT Users**: ZK proofs are being utilized to enable ChatGPT users to own their chat history, increasing the amount of verifiable training data for building open-source models.
  
  - The innovation marks a significant advance in dismantling data barriers for AI models and establishing data provenance, as discussed in a [demo tweet](https://x.com/openblocklabs/status/1848805457290572199).
- **Praise for SNES Music Diffusion Model**: A member showcased their SNES music diffusion model, which they've trained from scratch and inpainted to perfection.
  
  - Another remarked on the **great sound**, asking for further details, which the creator provided through a [GitHub link](https://github.com/parlance-zz/dualdiffusion) to the project.
- **Suspicion Surrounding GitHub Organization**: Concerns were raised about a GitHub organization, with allegations of recently created repositories and potential honeypot operations.
  
  - Additional scrutiny came from members pointing out **unusual licenses** and a lack of documentation for binary releases.
- **Fourier Dual Diffusion Project Shared**: The SNES music diffusion model developer shared their GitHub repository, which includes code for scraping, dataset processing, training, and a web interface.
  
  - They also mentioned having a development blog linked on the main page to provide further insights.

**Links mentioned**:

- [Tweet from OpenBlock (@openblocklabs)](https://x.com/openblocklabs/status/1848805457290572199): 1/ Introducing Proof of ChatGPT, the latest application built on OpenBlock’s Universal Data Protocol (UDP). This Data Proof empowers users to take ownership of their LLM chat history, marking a signi...
- [Shaq GIF - Shaq - Discover & Share GIFs](https://tenor.com/view/shaq-gif-18798422): Click to view the GIF
- [GitHub - parlance-zz/dualdiffusion: Fourier Dual Diffusion](https://github.com/parlance-zz/dualdiffusion): Fourier Dual Diffusion. Contribute to parlance-zz/dualdiffusion development by creating an account on GitHub.

---

### **HuggingFace ▷ #**[**NLP**](https://discord.com/channels/879548962464493619/922424173916196955/1298406185061584989) (3 messages):

> - `Custom Model Tokenizers`
> - `Automating CAD File Creation`

- **Choosing Tokenizers for Custom Models**: A member inquired about which tokenizer to use for their custom model, sparking discussion among other members.
  
  - One member suggested using **tiktoken** as a viable option for tokenization.
- **Automating CAD File Creation with LLMs**: A member proposed automating the creation of CAD files using a pipeline that integrates **rag** (retrieval-augmented generation) and **LLM** (large language model) technologies.
  
  - They requested insights from the community on systems design approaches or alternative strategies for this automation.

 

---

### **HuggingFace ▷ #**[**diffusion-discussions**](https://discord.com/channels/879548962464493619/1009713274113245215/1298540561896181780) (3 messages):

> - `Gradio Questions`
> - `Channel Guidance`

- **Guidance on Asking About Gradio**: <@king_92582> posed a question regarding **Gradio**, seeking assistance from the community.
  
  - Another member suggested asking in a new channel, notably **<#922424173916196955>**, for topics related to **rag** and **LLM**.
- **Clarification on Channel Focus**: A member pointed out that the current channel is designated for **diffusion models**, advising others to use the appropriate one.
  
  - There was a clear indication that it's crucial to maintain discussions in their respective channels to ensure organization.

 

---

### **OpenAI ▷ #**[**ai-discussions**](https://discord.com/channels/974519864045756446/998381918976479273/1298376044893569154) (300 messages🔥🔥):

> - `Claude 3.5 Sonnet`
> - `Anthropic Computer Use Tool`
> - `Model Performance Improvements`
> - `AI Tokenization Issues`
> - `AI Systems and Optimization`

- **Claude 3.5 Sonnet shows notable performance gains**: Claude 3.5 Sonnet has improved by approximately **15%** on the SWE-bench and enhanced agentic benchmarks, suggesting successful fine-tuning.
  
  - The integration of active learning techniques may contribute to these enhancements and make the model efficacious for computer-using capabilities.
- **Anthropic introduces Computer Use Tool**: Anthropic launched a new tool that allows agents to perform tasks on a computer, which could represent the future of agent functionality.
  
  - This tool leverages advanced data processing to improve user interactions, enabling more intuitive tools for API consumers.
- **Discussions on AI system optimization**: The conversation highlighted how AI models, particularly since GPT-3, have become more parameter-efficient, allowing significant performance improvement with fewer resources.
  
  - Participants speculate that continued optimization will enhance operational capabilities, including functionalities for computer control.
- **Concerns regarding Anthropic's tokenizer**: Concerns were raised about the effectiveness of Anthropic's tokenizer, which reportedly produces repetitive and unnecessary responses.
  
  - Better tokenization methods might enhance overall model performance significantly.
- **Users experience session logout issues**: Several users reported experiencing automatic logouts in ChatGPT, with occurrences noted at around 1-2 times per week.
  
  - This seems to be a common issue among users, albeit not a frequent one.

 

**Link mentioned**: [How to Keep Improving When You're Better Than Any Teacher - Iterated Distillation and Amplification](https://youtu.be/v9M2Ho9I9Qo): [2nd upload] AI systems can be trained using demonstrations from experts, but how do you train them to out-perform those experts? Can this still be done even...

 

---

### **OpenAI ▷ #**[**gpt-4-discussions**](https://discord.com/channels/974519864045756446/1001151820170801244/1298505211635175515) (9 messages🔥):

> - `GPT-4 Upgrade Timeline`
> - `Caching Feature in GPT-4`
> - `ChatGPT Payment Issues`

- **Curious about GPT-4 Upgrade Timeline**: A member noted that **GPT-4** has been utilized for most of the year but recalled seeing a mention of an impending upgrade, although they couldn't locate the info.
  
  - Another member mentioned that it was approximately **4-5 months ago** when free users gained access to GPTs.
- **Discussion on Caching Feature**: A member inquired whether function calls in the GPT API utilize only the last message or the entire conversation's context.
  
  - They also expressed confusion about the new caching feature, mentioning it shows a **cache hit false** in Langsmith.
- **ChatGPT Payment Issues Confusion**: A user reported issues accessing Plus features after making a **monthly payment**, still receiving an upgrade prompt upon login.
  
  - A fellow member directed them to contact support through [OpenAI Help](https://help.openai.com) for resolution.

 

**Link mentioned**: [Custom GPT's upgrade base model to omni?](https://community.openai.com/t/custom-gpts-upgrade-base-model-to-omni/780612): Hi people! I’m realy wonderful with GPT 4o, really about fast request and reorder of topics. But i considere a problem to bring my custom GPT’s from the old model (i thinks thats are Gizmo model) to...

 

---

### **OpenAI ▷ #**[**prompt-engineering**](https://discord.com/channels/974519864045756446/1046317269069864970/1298418764131336263) (7 messages):

> - `Spatial Sense in Models`
> - `Custom GPT Challenge`
> - `Function Call Context`
> - `Realtime API Performance`

- **Models struggle with spatial sense**: A discussion highlighted that models have weak **spatial sense** but can repeat correct answers if they exist in training data, akin to a child mimicking learned responses without understanding.
  
  - *The model may solve problems but struggles with tasks requiring deeper comprehension.*
- **Challenge with 'Not ChatGPT'**: A member introduced a custom GPT named **'Not ChatGPT'**, which is programmed to deny any connection to ChatGPT, raising curiosity about its potential to reveal this relation.
  
  - The challenge involves convincing it to acknowledge its origins, suggesting an underlying cleverness in the design of the model.
- **Understanding Context for Function Calls**: Questions arose regarding whether function calls in **ChatGPT** depend on just the latest message or if they consider previous exchanges for better context.
  
  - A specific use case was mentioned where a function should trigger only after multiple confirmations in the conversation.
- **Concerns with Realtime API performance**: A member noted that the **Realtime API** performs worse than GPT-4o in adhering to their system prompt instructions.
  
  - They sought suggestions on how to adapt prompts for better performance.

 

---

### **OpenAI ▷ #**[**api-discussions**](https://discord.com/channels/974519864045756446/1046317269069864970/1298418764131336263) (7 messages):

> - `Spatial Sense Weakness`
> - `Not ChattGPT Custom GPT`
> - `Function Calls Context in Chat Completion`
> - `Realtime API Performance Comparison`
> - `Prompt Adaptation Suggestions`

- **Models Struggle with Spatial Sense**: A member highlighted that the models are **extremely weak** with spatial sense, often repeating correct answers from training data without true comprehension.
  
  - They likened it to a student who can recite answers but lacks the ability to apply understanding.
- **Not ChattGPT Defies Identity**: A member presented their custom GPT called **'Not ChattGPT'**, designed to deny any connection to ChatGPT while attempting to reveal its relationship.
  
  - They invited others to challenge the model with inquiries regarding its ties to ChatGPT, excluding hypotheticals.
- **Context Needed for Function Calls**: Multiple members inquired about whether function calls in chat completions depend solely on the **last message** or consider previous context.
  
  - They illustrated a need for the function to activate only after **multiple confirmations** in the chat.
- **Feedback on Realtime API**: A member expressed frustration that the **Realtime API** was performing worse in following system prompts compared to **GPT-4o**.
  
  - They solicited input from others who might share similar experiences or suggestions for improvement.
- **Ways to Adapt Prompts**: Building upon the discussion about the Realtime API, a member asked for suggestions on how to effectively **adapt prompts**.
  
  - This highlights ongoing efforts to improve interactions and outputs through better prompt engineering.

 

---

### **Unsloth AI (Daniel Han) ▷ #**[**general**](https://discord.com/channels/1179035537009545276/1179035537529643040/1298369505621184684) (202 messages🔥🔥):

> - `Claude 3.5 models`
> - `Kaggle & PyTorch issues`
> - `Fine-tuning challenges`
> - `Unsloth sloth symbolism`
> - `MMLU performance concerns`

- **Claude 3.5 models launch**: Anthropic announced the upgraded **Claude 3.5 Sonnet** and new **Claude 3.5 Haiku** models, highlighting significant improvements, especially in coding tasks.
  
  - The introduction of the **computer use** capability in public beta allows developers to direct Claude to interact with computers as humans do.
- **Kaggle struggles with PyTorch**: Users report `ImportError` issues when running PyTorch on Kaggle, specifically related to CUDA version discrepancies, prompting workarounds by reinstalling specific versions.
  
  - Downgrading PyTorch to use CUDA 12.1 resolves errors and ensures compatibility in existing library installations.
- **Challenges in model fine-tuning**: Concerns were raised about models repeating inputs during fine-tuning, with discussions suggesting that the system prompt might need variations to prevent overfitting.
  
  - Users speculated that insufficient training examples might be causing reliance on the base model, leading to repetitive outputs from fine-tuned models.
- **Unsloth symbol meaning**: The symbolism of the sloth in Unsloth was discussed, with suggestions that it represents the contrast between slow, traditional fine-tuning processes and faster, more efficient ones.
  
  - Participants noted that **Unsloth** signifies making slow processes 'unslow' and emphasizes a quicker approach to AI model training.
- **MMLU performance observations**: Discussions on MMLU performance indicated that models trained on specific datasets often exhibit niche strengths, but generally do not outperform larger, generalist models like GPT-4.
  
  - The conversation highlighted the difficulty in achieving superior performance across diverse tasks solely through fine-tuning on a limited dataset.

**Links mentioned**:

- `PyTorch`
  
  : no description found
- [Introducing computer use, a new Claude 3.5 Sonnet, and Claude 3.5 Haiku](https://www.anthropic.com/news/3-5-models-and-computer-use): A refreshed, more powerful Claude 3.5 Sonnet, Claude 3.5 Haiku, and a new experimental AI capability: computer use.
- [Unsloth Notebooks | Unsloth Documentation](https://docs.unsloth.ai/get-started/unsloth-notebooks): See the list below for all our notebooks:
- [Kortix/FastApply-7B-v1.0 · Hugging Face](https://huggingface.co/Kortix/FastApply-7B-v1.0): no description found
- [TinyLlama/TinyLlama-1.1B-Chat-v1.0 · Hugging Face](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0): no description found
- [no title found](https://download.pytorch.org/whl/cu121): no description found
- [cerebras/SlimPajama-627B · Datasets at Hugging Face](https://huggingface.co/datasets/cerebras/SlimPajama-627B): no description found
- [Issues · pytorch/pytorch](https://github.com/pytorch/pytorch/issues/134929)): Tensors and Dynamic neural networks in Python with strong GPU acceleration - Issues · pytorch/pytorch
- [gbharti/finance-alpaca · Datasets at Hugging Face](https://huggingface.co/datasets/gbharti/finance-alpaca): no description found
- [GitHub - huggingface/transformers: 🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX.](https://github.com/huggingface/transformers.git): 🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX. - huggingface/transformers
- [Google Colab](https://colab.research.google.com/drive/17d3U-CAIwzmbDRqbZ9NnpHxCkmXB6LZ0?usp=sharing): no description found
- [Gemma2 fails saving as GGUF · Issue #785 · unslothai/unsloth](https://github.com/unslothai/unsloth/issues/785#issuecomment-2426714313): @danielhanchen Hi Daniel, thanks for your work! having an error just like in the issue #275, but this time while trying to save tuned version of unsloth/gemma-2-9b-it-bnb-4bit. model.save_pretraine...
- [mlabonne/FineTome-100k · Datasets at Hugging Face](https://huggingface.co/datasets/mlabonne/FineTome-100k): no description found
  
   
  

---

### **Unsloth AI (Daniel Han) ▷ #**[**off-topic**](https://discord.com/channels/1179035537009545276/1179039861576056922/1298521687452549160) (1 messages):

> - `PhD Research Experience`
> - `Publication Rates in Academia`
> - `Comparative PhD Culture`
> - `AI/ML/CV Fields`

- **PhD Journey in Europe vs. US**: A PhD student in Europe shared their journey, highlighting a **4-year timeline** that includes learning research methods and securing industry contracts while progressively publishing in leading conferences.
  
  - *Their experience emphasizes a detailed development in research skills*, contrasting sharply with peers in the US who reportedly produce **10 publications** with multiple first-author credits.
- **Publication Pressure in US Academics**: The student expressed confusion over the high publication output of US PhD students, noting some average **10 publications**, many being first authors in prestigious venues like **CVPR** and **ICML**.
  
  - *The message raises questions about work-life balance* and the pressures faced by academics in highly competitive environments.
- **Industry Involvement Enhances Skills**: The individual emphasized their active involvement in industry projects, writing **production grade code** for companies associated with their lab to enhance their practical skills.
  
  - *This experience adds a practical dimension to their research capabilities*, setting a foundation for future career opportunities.

 

**Link mentioned**: [Reddit - Dive into anything](https://www.reddit.com/r/MachineLearning/comments/1g7dzkp/d_why_do_phd_students_in_the_us_seem_like/): no description found

 

---

### **Unsloth AI (Daniel Han) ▷ #**[**help**](https://discord.com/channels/1179035537009545276/1179777624986357780/1298373345695367281) (92 messages🔥🔥):

> - `Multi-GPU Support`
> - `Unsloth Import Errors`
> - `Using Unsloth with Kaggle`
> - `Model Fine-Tuning Procedures`
> - `Gemma Templates`

- **Multi-GPU Support Not Available for Unsloth**: A member expressed frustration that Unsloth does not currently support multi-GPU usage, specifically when trying to load models across a 4090 and 3090.
  
  - Another user mentioned that features like DDP multi-GPU support are in beta testing, with full sharded data parallelism taking longer to implement.
- **Import Errors and CUDA Issues**: A user reported an ImportError related to `libcusparse.so.12` when trying to use Unsloth on Kaggle, indicating potential underlying CUDA issues.
  
  - Community members suggested the error might be related to the installation and recommended checking CUDA compatibility.
- **Unsloth Functionality in Kaggle Notebooks**: A user noted persistent errors while trying to run Unsloth from Kaggle notebooks, feeling frustrated with the situation since no changes were made to the setup.
  
  - Another user posted a solution link for resolving similar issues, indicating ongoing compatibility issues with Unsloth on Kaggle.
- **Fine-Tuning Models Using Unsloth**: Multiple users discussed experiences with fine-tuning Llama models using Unsloth, mentioning recent errors and suggestions for troubleshooting.
  
  - One user confirmed that reverting to an older version of Unsloth can help alleviate some problems faced during fine-tuning.
- **Finding Gemma Templates**: Discussion around Gemma templates used in models revealed shared snippets and links to aid others in formatting messages correctly.
  
  - Community members were grateful for the shared resources, facilitating easier adjustments and template usage in their projects.

**Links mentioned**:

- [no title found](https://ai.google.dev/gemma/docs/formatting): no description found
- [tokenizer_config.json · microsoft/Phi-3.5-mini-instruct at main](https://huggingface.co/microsoft/Phi-3.5-mini-instruct/blob/main/tokenizer_config.json): no description found
- [UNSL - Overview](https://github.com/unsl): UNSL has one repository available. Follow their code on GitHub.
- [Many bug fixes (#1162) · unslothai/unsloth@0e5a507](https://github.com/unslothai/unsloth/commit/0e5a507f87132cd8fbae5239fc436ef5ba3232d6): \* Fix TRL
  
  - Update mistral.py
  - Patch processing_class
  - Update tokenizer_utils.py
  - Update tokenizer_utils.py
  - Update tokenizer_utils.py
  - Update tokenizer_utils.py
  - Update ...
  - [finetune_llama_unsloth.py](https://gist.github.com/Tengoles/488889e5a07a17aa99327076ba703460): GitHub Gist: instantly share code, notes, and snippets.
  - [GitHub - unslothai/unsloth: Finetune Llama 3.2, Mistral, Phi & Gemma LLMs 2-5x faster with 80% less memory](http://github.com/unslothai/unsloth): Finetune Llama 3.2, Mistral, Phi & Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth
    
     
    

---

### **Unsloth AI (Daniel Han) ▷ #**[**showcase**](https://discord.com/channels/1179035537009545276/1179779344894263297/1298509447815106633) (1 messages):

> - `Fast Apply`
> - `Qwen2.5 Coder Model`
> - `Cursor's Blog Post`
> - `Performance Metrics`

- **Fast Apply revolutionizes code updates**: Exciting news! **Fast Apply** is an open-source, fine-tuned **Qwen2.5 Coder Model** that quickly and accurately applies code updates without repetitive file editing, boosting efficiency.
  
  - This solution is particularly beneficial when using tools like Aider, enabling large models to concentrate on actual code updates instead of cumbersome **SEARCH/REPLACE** tasks.
- **Cursor's blog inspires Fast Apply**: The development of Fast Apply was inspired by a now-deleted blog post from **Cursor**, with an archived version available [here](https://web.archive.org/web/20240823050616/https://www.cursor.com/blog/instant-apply).
  
  - This demonstrates how community resources can impact tool advancements in AI programming.
- **Impressive speed stats for Fast Apply**: **Performance metrics** reveal Fast Apply operates at approximately **340 tok/s** for the **1.5B Model** and **150 tok/s** for the **7B Model** when utilizing a fast provider like Fireworks.
  
  - These performance levels highlight Fast Apply's potential for increased productivity in coding tasks.

 

**Link mentioned**: [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/1ga25gj/introducing_fast_apply_replicate_cursors_instant/): no description found

 

---

### **Unsloth AI (Daniel Han) ▷ #**[**community-collaboration**](https://discord.com/channels/1179035537009545276/1180144489214509097/1298368178392535080) (3 messages):

> - `Studio Fix PR`
> - `Tokenizer Patch PR`

- **PR for Fixing Studio Issues**: A pull request titled [Fix/studio](https://github.com/unslothai/unsloth-studio/pull/1/files) was made to address several issues flagged by users in Discord concerning the import of unsloth.
  
  - The issue reportedly doesn't occur in the finetune notebook, suggesting a specific problem with the studio environment.
- **Simple Fix for Tokenizer Bug**: Another pull request, [Fix/patch tokenizer](https://github.com/unslothai/unsloth/pull/1171), introduced a minor change aimed at correcting the placement of negation, which caused a **NoneType** error when conditions were met.
  
  - A user expressed gratitude after the fix, highlighting the importance of even small changes in code functionality.

**Links mentioned**:

- [Fix/patch tokenizer by Erland366 · Pull Request #1171 · unslothai/unsloth](https://github.com/unslothai/unsloth/pull/1171): negation placed incorrectly, therefore it introduce NoneType object is not callable since if None, it goes to the else part
- [Fix/studio by Erland366 · Pull Request #1 · unslothai/unsloth-studio](https://github.com/unslothai/unsloth-studio/pull/1/files): There are several issue in the studio. The issue was issued by user in the discord. This issue is trigger when importing unsloth, but somehow the issue didn&#39;t happen inside finetune notebook....

---

### **Unsloth AI (Daniel Han) ▷ #**[**research**](https://discord.com/channels/1179035537009545276/1257011997250424842/) (1 messages):

edd0302: [https://arxiv.org/pdf/2410.16663](https://arxiv.org/pdf/2410.16663)

---

### **Stability.ai (Stable Diffusion) ▷ #**[**general-chat**](https://discord.com/channels/1002292111942635562/1002292112739549196/1298361806552698961) (294 messages🔥🔥):

> - `Stable Diffusion 3.5`
> - `AI Generation Models`
> - `LoRA and VAE Handling`
> - `Image Prompting Techniques`
> - `Model Checkpoints and Sorting`

- **Stable Diffusion 3.5 Training and Use**: Participants discussed the structure and capabilities of **SD 3.5**, noting it uses a different neural network architecture similar to **Flux**.
  
  - There was a consensus on the need for finetuning and community training efforts to optimize results with the new model.
- **Prompting Techniques for Anime Art**: To achieve satisfactory results in generating anime art, it was recommended to utilize **SD 3.5** with correct prompts instead of relying on LoRAs.
  
  - It was suggested to bypass LoRAs and only use stable diffusion 3.5 with appropriate prompting for best outcomes.
- **Image Generation Result Quality**: Users reported mixed quality in their image generation results, particularly when using incorrect checkpoints or Loras targeted for other models.
  
  - Participants advised checking if the models align correctly with the intended prompts to avoid unsatisfactory outputs.
- **Model Management and Organization**: A user expressed the need for a tool to automatically sort and organize AI model files within various folders.
  
  - Suggestions included reaching out in the server's technical support channel for potential solutions.
- **Community Engagement and Tools**: Discussions highlighted various tools and methods to enhance AI generation workflows, including using workflows from model creators.
  
  - Participants shared their experiences with different tools, such as ComfyUI and fp8 models, facilitating better management of AI tasks.

**Links mentioned**:

- [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206): Diffusion models create data from noise by inverting the forward paths of data towards noise and have emerged as a powerful generative modeling technique for high-dimensional, perceptual data such as ...
- [Meta AI](https://www.meta.ai/): Use Meta AI assistant to get things done, create AI-generated images for free, and get answers to any of your questions. Meta AI is built on Meta's latest Llama large language model and uses Emu,...
- [Rock Hyrox Rock Hyrox Eating GIF - Rock hyrox Rock hyrox eating Rock hyrox funny - Discover & Share GIFs](https://tenor.com/view/rock-hyrox-rock-hyrox-eating-rock-hyrox-funny-animal-animal-eating-gif-15423214377703191267): Click to view the GIF
- [stabilityai/stable-diffusion-3.5-large at main](https://huggingface.co/stabilityai/stable-diffusion-3.5-large/tree/main): no description found
- [Genmo AI Mochi 1 - The Best Open Source DiT Video Model By Far](https://www.youtube.com/watch?v=qDJrSK6uynQ): In this video, we check out the groundbreaking Genmo AI Mochi 1, the latest open-source video generation model, that is revolutionizing the industry. With a ...
- [Stable Diffusion 3.5 Large - v1.0 | Stable Diffusion Checkpoint | Civitai](https://civitai.com/models/878387/stable-diffusion-35-large): Stable Diffusion 3.5 Large is a Multimodal Diffusion Transformer (MMDiT) text-to-image model that features improved performance in image quality, t...
- [Dress Up Magic: Seamless Clothes-Swapping with ControlNet in Automatic1111](https://youtu.be/AGqqj_xQ6IM): Transform ANY character's outfit instantly with AI magic! Discover how ControlNet + Automatic1111 unlock pixel-perfect, seamless clothes changes in Stable D...
- [Walmart Back Then Vs Walmart Now #walmart #2004 #2024 #groceryshopping #groceries #food #nostalgia](https://youtube.com/shorts/pNaMOMqK4n4?si=t2z0EW_dItn3T0ww): no description found
- [Added preliminary support for SD3.5-large lora training · ostris/ai-toolkit@3400882](https://github.com/ostris/ai-toolkit/commit/3400882a8099645ce4c797f57ac258f1e1424ffd): no description found
- [Tweet from Michael R. Bernstein (@NerdWorldOrder)](https://x.com/NerdWorldOrder/status/1740177328955924781): I use artist names in my text-to-image prompts. I use them a LOT. Mostly I use names of artists that are no longer among the living, but am more-or-less agnostic on whether their work is still under ...
- [Tweet from Michael R. Bernstein (@NerdWorldOrder)](https://x.com/NerdWorldOrder/status/1742294834558517695): New #aiART style discovered today in the form of a prompt "By Artist A and Artist B" and a negative prompt "Artist C", none of them living. In this case, two of the artists are still u...

---

### **aider (Paul Gauthier) ▷ #**[**announcements**](https://discord.com/channels/1131200896827654144/1133060115264712836/1298395827508609109) (1 messages):

> - `Aider v0.60.0`
> - `Sonnet 10/22 support`
> - `Improved code editing`
> - `Bugfixes`
> - `Model metadata management`

- **Aider v0.60.0 Release Highlights**: The new release of **Aider v0.60.0** includes full support for **Sonnet 10/22**, which is now the default model on the code editing benchmark.
  
  - This version emphasizes robust formatting and handling of file interactions, significantly enhancing user experience.
- **Sonnet 10/22 becomes default**: Aider has integrated **Sonnet 10/22** as the default model, ensuring state-of-the-art performance on its code editing benchmark.
  
  - This adjustment aims to leverage improved coding predictions for users and streamline editing tasks.
- **Enhanced Code Editing Capabilities**: The update improves formatting for added and read-only files, addressing some parsing inconsistencies with **o1 models**.
  
  - These improvements also include a stronger prompt for clean file names and better handling of nonconforming code edit replies.
- **Notable Bugfixes and Features**: The release features a bugfix that properly includes URLs in `/help` RAG results and adds functionality for ignoring **.env** files.
  
  - Aider even accomplished a notable feat by writing **49% of the code** in this release, demonstrating its capabilities.
- **Model Metadata and Settings Management**: Aider ships with a small model metadata JSON file to accommodate unupdated models in **litellm**, enhancing overall functionality.
  
  - Moreover, new model settings specifically for **o1 models on Azure** reflect a robust approach to model management.

 

---

### **aider (Paul Gauthier) ▷ #**[**general**](https://discord.com/channels/1131200896827654144/1131200896827654149/1298360568557867028) (240 messages🔥🔥):

> - `Claude 3.5 Sonnet`
> - `Aider Benchmarking`
> - `Fast Apply`
> - `Qwen 2.5 and Replete`
> - `OpenRouter and Rate Limits`

- **New Claude 3.5 Sonnet shows promising performance**: Many users have reported that the new **Claude 3.5 Sonnet** model is significantly better than the previous **O1 models**, often achieving results that were previously unreachable.
  
  - One user noted how it successfully implemented a VAD library into their codebase with just a single prompt.
- **Aider Benchmarking Insights**: Users are actively benchmarking different quantizations of models for Aider, striving for better performance within hardware limitations, particularly focusing on the **Qwen 2.5** models.
  
  - A user achieved a score of **62.4%** using the **Replete** version and expressed interest in testing larger models for comparison.
- **Fast Apply Introduced**: The **Fast Apply** tool aims to streamline code updates in existing files, developed for enhanced efficiency when working with complex codebases.
  
  - The tool is designed to minimize redundancy in processing and reduce token costs during code editing operations.
- **Discussion on OpenRouter and Rate Limits**: Some users expressed dissatisfaction with **Anthropic's rate limits** and discussed their transition to **OpenRouter** for better terms, especially for privacy-sensitive applications.
  
  - Users pointed out that using OpenRouter offers a workaround for the strict limits imposed on the Anthropic API.
- **Exploration of Model Conventions**: A new user inquired about **CONVENTIONS.md** resources for instructing Aider effectively, indicating a need for best practices and documentation.
  
  - The community shared various insights on model configurations and the improvements observed with recent updates.

**Links mentioned**:

- [Rate limits - Anthropic](https://docs.anthropic.com/en/api/rate-limits#requirements-to-advance-tier): no description found
- [Model warnings](https://aider.chat/docs/troubleshooting/warnings.html): aider is AI pair programming in your terminal
- [Separating code reasoning and editing](https://aider.chat/2024/09/26/architect.html): An Architect model describes how to solve the coding problem, and an Editor model translates that into file edits. This Architect/Editor approach produces SOTA benchmark results.
- [OpenAI compatible APIs](https://aider.chat/docs/llms/openai-compat.html): aider is AI pair programming in your terminal
- [bartowski/Replete-LLM-V2.5-Qwen-32b-GGUF · Hugging Face](https://huggingface.co/bartowski/Replete-LLM-V2.5-Qwen-32b-GGUF): no description found
- [Models: 'google' | OpenRouter](https://openrouter.ai/models?q=google): Browse models on OpenRouter
- [Kortix/FastApply-7B-v1.0 · Hugging Face](https://huggingface.co/Kortix/FastApply-7B-v1.0): no description found
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/1gajy1j/aider_optimizing_performance_at_24gb_vram_with/): no description found
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/1ga25gj/introducing_fast_apply_replicate_cursors_instant/): no description found
- [Decompiler Explorer](https://dogbolt.org/?id=f4fbe795-0956-4f25-aab5-27aeb7db171d): Decompiler Explorer is an interactive online decompiler which shows equivalent C-like output of decompiled programs from many popular decompilers.
- [aider/aider/voice.py at main · Aider-AI/aider](https://github.com/Aider-AI/aider/blob/main/aider/voice.py): aider is AI pair programming in your terminal. Contribute to Aider-AI/aider development by creating an account on GitHub.
- [Build software better, together](https://github.com/Aider-AI/aider/pull/2099).): GitHub is where people build software. More than 100 million people use GitHub to discover, fork, and contribute to over 420 million projects.
- [Claude 3.5 Sonnet (self-moderated) - API, Providers, Stats](https://openrouter.ai/anthropic/claude-3.5-sonnet:beta): The new Claude 3.5 Sonnet delivers better-than-Opus capabilities, faster-than-Sonnet speeds, at the same Sonnet prices. Run Claude 3.5 Sonnet (self-moderated) with API

---

### **aider (Paul Gauthier) ▷ #**[**questions-and-tips**](https://discord.com/channels/1131200896827654144/1133060505792159755/1298373493540519947) (45 messages🔥):

> - `Claude 3.5 updates`
> - `Aider debugging tips`
> - `Mistral API authentication issues`
> - `Git commands and navigation`
> - `Repo map functionality`

- **Claude 3.5 introduces new capabilities**: Anthropic announced the upgraded [Claude 3.5 Sonnet](https://x.com/AnthropicAI/status/1848742740420341988) and a new model, Claude 3.5 Haiku, with the ability to use computers like humans.
  
  - *A member noted* that this capability includes directing Claude to engage with screens, cursors, and text inputs.
- **Mistral API access troubleshooting**: A user encountered an *AuthenticationError* while using the Mistral API with Aider, indicating a possible issue with their authentication key.
  
  - After feedback, they resolved it by deleting and re-creating their key before it worked properly again.
- **Understanding Git commands in Aider**: A user inquired about the implications of the /undo command in Aider, to which others responded that old commits can still be accessed using their hashes.
  
  - *It was confirmed* that Git commands, such as checkout, do not inherently relate to Aider but are fundamental Git functionalities.
- **Repo map and code context functionality**: Discussions emerged regarding how the repo map comprehends file relationships, emphasizing the importance of being provided relevant code context for effective modifications.
  
  - *Paul clarified* that the relationships between identifiers are based on their definitions and references when evaluated by the model.
- **Commands in Aider - Abbreviations and Redundancy**: A user questioned the distinction between '/read' and '/read-only', finding them to be essentially the same.
  
  - Paul clarified that all commands can be abbreviated and suggested that only the full command is necessary to avoid confusion.

**Links mentioned**:

- [Building a better repository map with tree sitter](https://aider.chat/2023/10/22/repomap.html): Tree-sitter allows aider to build a repo map that better summarizes large code bases.
- [Edit formats](https://aider.chat/docs/more/edit-formats.html): Aider uses various “edit formats” to let LLMs edit source files.
- [Model warnings](https://aider.chat/docs/llms/warnings.html): aider is AI pair programming in your terminal
- [Tweet from Anthropic (@AnthropicAI)](https://x.com/AnthropicAI/status/1848742740420341988): Introducing an upgraded Claude 3.5 Sonnet, and a new model, Claude 3.5 Haiku. We’re also introducing a new capability in beta: computer use. Developers can now direct Claude to use computers the way ...
- [Advanced model settings](https://aider.chat/docs/config/adv-model-settings.html): Configuring advanced settings for LLMs.
- [Does architect mode prompt to add files? · Issue #2121 · Aider-AI/aider](https://github.com/Aider-AI/aider/issues/2121): Issue Shared in discord: https://discord.com/channels/1131200896827654144/1133060505792159755/1298228879210577931 /architect example bla bla ... Now, we need to update other files to incorporate th...

---

### **aider (Paul Gauthier) ▷ #**[**links**](https://discord.com/channels/1131200896827654144/1268910919057149974/1298381074744868914) (1 messages):

> - `DreamCut AI`
> - `Claude AI`
> - `Video Editing Software`

- **DreamCut AI: A New Era in Video Editing**: [DreamCut AI](http://dreamcut.ai) is introduced as a video editor built from scratch using **Claude AI**, taking **3 months and over 50k lines of code**.
  
  - Currently in early access, users can test out the **AI tools** with a free account, showcasing an interesting blend of coding and AI technology.
- **AI's Role in Software Development**: The creation of DreamCut AI exemplifies how AI can play a pivotal role in software development by eliminating conventional design phases and focusing directly on coding.
  
  - A member expressed that this approach to building software is **interesting**, possibly indicating a trend toward more AI-driven development processes.

 

**Link mentioned**: [Tweet from Meng To (@MengTo)](https://x.com/MengTo/status/1848669694800367901): Introducing [http://dreamcut.ai](http://dreamcut.ai) A video editor I built from scratch using Claude AI. This took 3 months and over 50k lines of code. I totally skipped design and went straight to code. Currently in e...

 

---

### **OpenRouter (Alex Atallah) ▷ #**[**announcements**](https://discord.com/channels/1091220969173028894/1092729520181739581/1298451822255538226) (2 messages):

> - `Claude 3.5 Sonnet`
> - `Lumimaid v0.2`
> - `Magnum v4`
> - `Discounts on Models`

- **Claude 3.5 Sonnet Versions Released**: **Older versions of Claude 3.5 Sonnet** have been released and are available for download, timestamped for reference: [Claude 3.5 Sonnet](https://openrouter.ai/anthropic/claude-3.5-sonnet-20240620) and [Claude 3.5 Sonnet: Beta](https://openrouter.ai/anthropic/claude-3.5-sonnet-20240620:beta).
  
  - *These releases come from OpenRouter, providing users access to previous iterations.*
- **Lumimaid v0.2 Introduced**: **Lumimaid v0.2** is now available, serving as a finetuned version of Llama 3.1 70B with a **significantly enhanced dataset** compared to Lumimaid v0.1, accessible at [this link](https://openrouter.ai/neversleep/llama-3.1-lumimaid-70b).
  
  - *Users can expect improved performance due to the updates in the dataset specifics.*
- **Magnum v4 Launches with Unique Features**: **Magnum v4** has been released and finetuned to replicate the prose quality similar to Sonnet and Opus, available [here](https://openrouter.ai/anthracite-org/magnum-v4-72b).
  
  - *This model continues the trend of enhancing prose quality in AI written outputs.*
- **Exciting Discounts on Magnum Models**: Both **Magnum v1** and **Magnum v4** are currently available at half price from Mancer for a limited time.
  
  - *This discount offers a great opportunity for users to explore these new models at a reduced cost.*

**Links mentioned**:

- [Lumimaid v0.2 70B - API, Providers, Stats](https://openrouter.ai/neversleep/llama-3.1-lumimaid-70b)**): Lumimaid v0.2 70B is a finetune of [Llama 3. Run Lumimaid v0.2 70B with API
- [Magnum v4 72B - API, Providers, Stats](https://openrouter.ai/anthracite-org/magnum-v4-72b)**): This is a series of models designed to replicate the prose quality of the Claude 3 models, specifically Sonnet(https://openrouter.ai/anthropic/claude-3. Run Magnum v4 72B with API
- [Claude 3.5 Sonnet (2024-06-20) - API, Providers, Stats](https://openrouter.ai/anthropic/claude-3.5-sonnet-20240620): Claude 3.5 Sonnet delivers better-than-Opus capabilities, faster-than-Sonnet speeds, at the same Sonnet prices. Run Claude 3.5 Sonnet (2024-06-20) with API
- [Claude 3.5 Sonnet (2024-06-20) (self-moderated) - API, Providers, Stats](https://openrouter.ai/anthropic/claude-3.5-sonnet-20240620:beta): Claude 3.5 Sonnet delivers better-than-Opus capabilities, faster-than-Sonnet speeds, at the same Sonnet prices. Run Claude 3.5 Sonnet (2024-06-20) (self-moderated) with API

---

### **OpenRouter (Alex Atallah) ▷ #**[**general**](https://discord.com/channels/1091220969173028894/1094454198688546826/1298361551924756513) (272 messages🔥🔥):

> - `OpenRouter Model Updates`
> - `API Key Usage`
> - `Prompt Caching`
> - `Tool Use in Models`
> - `Website Access Issues`

- **OpenRouter Model Updates**: The discussions highlighted the release of the new Sonnet 3.5 model and the implications for existing users, who are navigating between different versions.
  
  - Users were informed that the API names for models remain the same, meaning current implementations are likely using the new version.
- **API Key Usage**: Multiple users mentioned the differences in API costs when using OpenRouter compared to direct provider keys, with some reporting unexpected charges.
  
  - Alerts were raised regarding the importance of understanding how different models incur costs under OpenRouter.
- **Prompt Caching**: Concerns were raised about the functionality of prompt caching, with several users noting that it appears to be not working correctly for some models.
  
  - It was suggested that prompt caching had been tested before the app switched to a new model version.
- **Tool Use in Models**: Users expressed interest in integrating tool use selectively with models that do not use tool calls by default, seeking advice on implementation strategies.
  
  - Questions were raised about OpenRouter's support for a 'tool' role and how effectively tool calling can be implemented across different models.
- **Website Access Issues**: Several users reported difficulties accessing OpenRouter's website, with some experiencing loading issues while using different browsers.
  
  - After some time, users confirmed that the website began functioning properly again.

**Links mentioned**:

- [Full Stack && Web3 Developer](https://daniel0629.vercel.app): I am a highly skilled blockchain and full stack developer with extensive experience in designing and implementing complex decentralized applications and web solutions.
- [Chatroom | OpenRouter](https://openrouter.ai/chat): LLM Chatroom is a multimodel chat interface. Add models and start chatting! Chatroom stores data locally in your browser.
- [A Small Step Towards Reproducing OpenAI o1](https://medium.com/@peakji/a-small-step-towards-reproducing-openai-o1-b9a756a00855): Progress Report on the Steiner Open Source Models
- [OpenRouter](https://openrouter.ai/docs/limits>): LLM router and marketplace
- [Aider LLM Leaderboards](https://aider.chat/docs/leaderboards/): Quantitative benchmarks of LLM code editing skill.
- [Prompt Caching | OpenRouter](https://openrouter.ai/docs/prompt-caching): Optimize LLM cost by up to 90%
- [Claude 3.5 Sonnet - API, Providers, Stats](https://openrouter.ai/anthropic/claude-3.5-sonnet-this-version-doesnt-exist): New Claude 3.5 Sonnet delivers better-than-Opus capabilities, faster-than-Sonnet speeds, at the same Sonnet prices. Run Claude 3.5 Sonnet with API
- [Cocomelon Jj Cocomelon GIF - Cocomelon Jj Cocomelon Loose Tooth Song - Discover & Share GIFs](https://tenor.com/view/cocomelon-jj-cocomelon-loose-tooth-song-wiggle-wiggle-is-it-ready-gif-16776506): Click to view the GIF
- [Meta: Llama 3.1 70B Instruct – Provider Status](https://openrouter.ai/meta-llama/llama-3.1-70b-instruct/providers): See provider status and make a load-balanced request to Meta: Llama 3.1 70B Instruct - Meta's latest class of model (Llama 3.1) launched with a variety of sizes & flavors. This 70B instruct-t...
- [Models | OpenRouter](https://openrouter.ai/models?order=newest&supported_parameters=tools)): Browse models on OpenRouter
- [Quick Start | OpenRouter](https://openrouter.ai/docs/quick-start): Start building with OpenRouter
- [OpenRouter](https://openrouter.ai/terms): LLM router and marketplace
- [Discord - Group Chat That’s All Fun & Games](https://discord.co): Discord is great for playing games and chilling with friends, or even building a worldwide community. Customize your own space to talk, play, and hang out.
- [OpenRouter](https://openrouter.ai/): LLM router and marketplace
- [Activity | OpenRouter](https://openrouter.ai/activity): See how you've been using models on OpenRouter.
- [OpenRouter](https://openrouter.ai/docs/quick): LLM router and marketplace
- [OpenRouter](https://openrouter.ai/api/v1/anthropic/): LLM router and marketplace
- [Keys | OpenRouter](https://openrouter.ai/settings/keys): Manage your keys or create new ones
- [Requests | OpenRouter](https://openrouter.ai/docs/requests#tool-calls): Handle incoming and outgoing requests
- [OpenRouter](https://openrouter.ai/api/v1): LLM router and marketplace
- [AI SDK Core: Tool Calling](https://sdk.vercel.ai/docs/ai-sdk-core/tools-and-tool-calling): Learn about tool calling with AI SDK Core.
- [Models | OpenRouter](https://openrouter.ai/models?max_price=0): Browse models on OpenRouter
- [Conceptual guide | 🦜️🔗 Langchain](https://js.langchain.com/docs/concepts/#tools): This section contains introductions to key parts of LangChain.
- [GitHub - mem0ai/companion-nextjs-starter](https://github.com/mem0ai/companion-nextjs-starter?tab=readme-ov-file): Contribute to mem0ai/companion-nextjs-starter development by creating an account on GitHub.

---

### **OpenRouter (Alex Atallah) ▷ #**[**beta-feedback**](https://discord.com/channels/1091220969173028894/1277894087755829278/1298408655342796841) (10 messages🔥):

> - `Beta Access for Custom Provider Keys`
> - `Integrations Settings Access Requests`

- **Custom Provider Keys in Beta**: Custom provider keys are currently in beta, with requests for access facilitated through a specific Discord channel.
  
  - *Self-signup isn't available*, but members can DM their **OpenRouter** email addresses for access.
- **Multiple Requests for Integrations Access**: Several members expressed interest in beta access for the integrations settings, asking for directions on how to proceed.
  
  - One member noted, *'Hello, I also need a beta access for the integrations settings,'* reflecting the common desire for access.

 

---

### **LM Studio ▷ #**[**general**](https://discord.com/channels/1110598183144399058/1110598183144399061/1298361461088718899) (154 messages🔥🔥):

> - `Downloading Models in LM Studio`
> - `Limitations of LLMs for Coding`
> - `Model Quantization Methods`
> - `Issues with System Prompts in Models`
> - `Using Vision Models for Image Captioning`

- **Downloading Models in LM Studio**: Users discussed difficulties in finding and downloading certain large models in LM Studio, with specific mentions of the Nvidia's 70B Nemotron model and instructions on using terminal commands for downloading.
  
  - Some users noted that the search feature had changed, and they needed to hit specific keyboard shortcuts to access larger models during their searches.
- **Limitations of LLMs for Coding**: Users expressed frustration over the performance of various coding-focused LLMs, mentioning that models like Mistral and Llama 3.2 could not generate accurate results for coding tasks.
  
  - There was consensus that GPT-3.5 and GPT-4 were performing significantly better for coding, prompting users to consider alternative tools.
- **Model Quantization Methods**: The discussion included preferences for different quantization methods (Q2, Q4, Q8) in relation to model performance, with users sharing experiences on how these affect model effectiveness.
  
  - Opinions varied, with some users cautioning against Q2, while others indicated that certain models might perform better with lower bits of quantization at larger model sizes.
- **Issues with System Prompts in Models**: A user encountered issues with their models not recognizing system prompts, which led to new troubleshooting avenues including a temporary shift to prompt templating.
  
  - The conversation covered adjustments to configuration settings within LM Studio to work around the limitations with certain models that do not support traditional system prompts.
- **Using Vision Models for Image Captioning**: A user sought to utilize a specific vision model for automating image captioning but faced challenges related to model compatibility with LM Studio.
  
  - There were suggestions to look for GGUF quantized versions of the model, though it was noted that the desired model might not currently be compatible with Llama.cpp.

**Links mentioned**:

- [Better Florence 2 - a Hugging Face Space by SkalskiP](https://huggingface.co/spaces/SkalskiP/better-florence-2): no description found
- [lmstudio-community/Llama-3.1-Nemotron-70B-Instruct-HF-GGUF · Hugging Face](https://huggingface.co/lmstudio-community/Llama-3.1-Nemotron-70B-Instruct-HF-GGUF): no description found
- [mlx-community/xLAM-7b-fc-r · Hugging Face](https://huggingface.co/mlx-community/xLAM-7b-fc-r): no description found
- [GGUF in details](https://medium.com/@charles.vissol/gguf-in-details-8a9953ac7883): After Training phase, the models based on the llama.cpp architecture can be exchanged using the GGUF (GPT-Generated Unified Format) format.
- [bababababooey/llama-3.2-11b-vision-instruct-stheno-abliterated · Hugging Face](https://huggingface.co/bababababooey/llama-3.2-11b-vision-instruct-stheno-abliterated): no description found
- [Prompt Template - Configuration | LM Studio Docs](https://lmstudio.ai/docs/configuration/prompt-template): Editing the prompt template
- [Sideload models - Advanced | LM Studio Docs](https://lmstudio.ai/docs/advanced/sideload): Use model files you've downloaded outside of LM Studio

---

### **LM Studio ▷ #**[**hardware-discussion**](https://discord.com/channels/1110598183144399058/1153759714082033735/1298395439628030015) (13 messages🔥):

> - `Ryzen AI Configuration`
> - `Intel Lunar Lake NPU Support`
> - `NPU Support in Llama.cpp`
> - `RX 7900 XTX vs RTX 3090 Comparison`
> - `Multi-GPU Support in ROCm`

- **Issues with Ryzen AI and NPU Usage**: A member inquired about configuring LM Studio to utilize the **NPU of Ryzen processors** instead of the **RTX 4060**, but faced challenges in getting the NPU to function.
  
  - There are uncertainties around whether **NPU** support is currently available in LM Studio.
- **Interest in Intel Lunar Lake NPU Support**: A user asked if the **Intel Lunar Lake NPUs** would be useful for or supported by LM Studio.
  
  - Currently, the only **NPU** supported by **llama.cpp** is the **Ascend NPU**.
- **Multi-GPU Support in ROCm Update**: An update to **ROCm 6.1.3** allows for improved multi-GPU support, which was noted to enhance processing for up to **four qualified GPUs**.
  
  - There are mixed reports on the effectiveness of **multi-GPU utilization** across different GPU brands, with questions about **NVIDIA's** compatibility.
- **Comparison between RX 7900 XTX and RTX 3090**: A user sought advice on which GPU to choose between the **RX 7900 XTX** and the **RTX 3090** for **LLM usage**.
  
  - Others suggested that **CUDA support** for NVIDIA is preferable for a seamless experience in AI applications.
- **Performance Insights on AI GPUs**: Discussion indicated that for users primarily focused on **LLMs**, considering **CUDA support** is essential for optimal performance.
  
  - Several members echoed the sentiment that the **RTX 3090** would provide a better experience due to its **CUDA capabilities**.

**Links mentioned**:

- [AMD enhances multi-GPU support in latest ROCm update: up to four RX or Pro GPUs supported, official support added for Pro W7900 Dual Slot](https://www.tomshardware.com/pc-components/gpus/amd-enhances-multi-gpu-support-in-latest-rocm-update-up-to-four-rx-or-pro-gpus-supported-official-support-added-for-pro-w7900-dual-slot): The new update also includes beta support for Microsoft's Windows Subsystem for Linux
- [llama.cpp/docs/build.md at master · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md#cann): LLM inference in C/C++. Contribute to ggerganov/llama.cpp development by creating an account on GitHub.
- [Issues · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/issues): LLM inference in C/C++. Contribute to ggerganov/llama.cpp development by creating an account on GitHub.

---

### **Perplexity AI ▷ #**[**general**](https://discord.com/channels/1047197230748151888/1047649527299055688/1298360457584971899) (145 messages🔥🔥):

> - `New Sonnet 3.5 Model`
> - `Perplexity Web Search Issues`
> - `User Experience with API Models`
> - `Feedback on Updates`
> - `Use of Custom Rag Systems`

- **Users Complain About New Sonnet 3.5**: Multiple users expressed dissatisfaction with the new Sonnet 3.5 model, noting a significant decrease in content output, especially for academic writing tasks.
  
  - Concerns were raised about the removal of the older model, which was regarded as superior for various use cases.
- **Web Search Integration Issues**: There was a reported bug where the preprompt in Spaces does not function correctly when web search is enabled, causing frustration among users.
  
  - Users indicated that this issue has persisted, with the team acknowledging it and stating that they are working on a fix.
- **Discussion of Model Versions**: Some users inquired whether there would be an option to revert to previous versions of the model that performed better for their use cases.
  
  - The response implied that the system's policy is to always move to the latest available model without retaining older ones.
- **Seeking Refunds and Chargebacks**: Several users discussed the potential for seeking refunds or filing chargebacks due to dissatisfaction with the service following recent updates.
  
  - There was a sentiment that the availability of better-performing models could influence their decision to stay with the service.
- **Custom Rag Systems for Academic Use**: Users discussed setting up personal rag systems using Llama and other frameworks for academic tasks, finding them more effective than the current offerings from Perplexity.
  
  - One user explained their use of locally run systems to query textbooks and past exams, emphasizing the flexibility and control it provides.

**Links mentioned**:

- [Tweet from TestingCatalog News 🗞 (@testingcatalog)](https://x.com/testingcatalog/status/1849191714134843612?s=46): "Buy with Pro" This is how y'all are gonna shop in 2025 👀👀👀
- [Tweet from Aravind Srinivas (@AravSrinivas)](https://x.com/AravSrinivas/status/1849134236525347225): ⌘ + ⇧ + P release version tomorrow. Will start off as your shortcut to asking anything without opening a Chrome tab. And we’ll be investing more into local desktop oriented productivity improvements!

---

### **Perplexity AI ▷ #**[**sharing**](https://discord.com/channels/1047197230748151888/1054944216876331118/1298376357813948497) (17 messages🔥):

> - `Advanced AI-Driven Fact-Checking`
> - `Method of Loci Memory Trick`
> - `Component MRO in Aircraft Industry`
> - `Nvidia TSMC AI Alliance`
> - `Claude Computer Control in RPA`

- **Advanced AI-Driven Fact-Checking Explored**: A collection on [AI-driven fact-checking](https://www.perplexity.ai/collections/advanced-ai-driven-fact-checki-a3cMcPR.QsKkCRZ79UKFLQ) discusses innovative techniques using LLMs, including source credibility assessment and bias detection.
  
  - It highlights challenges such as the need for **transparency** and **human oversight** to effectively combat misinformation.
- **Method of Loci Memory Trick Taken to YouTube**: A [YouTube video](https://www.youtube.com/embed/k4R6iBvOEk0) introduces the **Method of Loci**, a memory enhancement technique that helps improve recall through spatial memory.
  
  - *Explore how this classic mnemonic device can enhance your memory abilities!*
- **Component MRO's Impact on Aviation**: Information regarding [Component MRO](https://www.perplexity.ai/search/what-is-component-mro-in-aircr-r0cr7t8uSCqulXRJWfLs1Q#0) in the aircraft industry reveals its pivotal role in maintenance and operational efficiency.
  
  - This system helps streamline inventory management and reduce downtime.
- **Nvidia's AI Alliance with TSMC Strains**: An article on [Nvidia's partnership with TSMC](https://www.perplexity.ai/page/nvidia-tsmc-ai-alliance-strain-MoGqt8XuQfaHfw63v71uow) details the complications in their AI chip production.
  
  - Concerns were raised about **resource allocation** and **manufacturing capabilities** impacting delivery timelines.
- **Claude Computer Use Model Raises Alarms for RPA**: A post on [Claude's Computer Control capabilities](https://www.perplexity.ai/page/claude-s-computer-control-capa-E_O4xa7VSWOi3lGtOWnnMw) suggests potential adversities for **Robotic Process Automation (RPA)**.
  
  - Experts are warning that this innovation might pose significant risks to existing processes.

 

**Link mentioned**: [YouTube](https://www.youtube.com/embed/k4R6iBvOEk0): no description found

 

---

### **Perplexity AI ▷ #**[**pplx-api**](https://discord.com/channels/1047197230748151888/1161802929053909012/1298461870189187082) (3 messages):

> - `Account Credit Transfer Issues`
> - `Server Errors`

- **Account Credits Still Not Transferred**: A user reported that their **account credits** have still not been transferred, despite reaching out to support.
  
  - *No response from support for the past three days* has left the user frustrated.
- **Frequent 524 Server Errors**: Another user noted they are experiencing **524 errors consistently** throughout the day.
  
  - Earlier, there was also a mention of **500 errors**, indicating potential server issues affecting multiple users.

 

---

### **GPU MODE ▷ #**[**general**](https://discord.com/channels/1189498204333543425/1189498205101109300/1298365974658945128) (6 messages):

> - `LLM Activations Quantization`
> - `bf16 vs fp32 Precision`
> - `Anyscale Inference Engine`
> - `CUDA Streams and Synchronization`

- **LLM Activations Quantization Debate**: A newbie questioned whether **activations in LLMs** sensitive to input variations should not be aggressively quantized, suggesting maintenance or higher precision quantization.
  
  - *This raises concerns about modeling performance* and the trade-offs of precision in quantization.
- **Precision Worries with bf16**: A member expressed that **bf16** and **fp32** have similar number ranges, but troubles with bf16's precision could lead to **canceled updates** after multiple gradient accumulations.
  
  - *Concern about precision is significant*, especially when it affects model training stability.
- **Anyscale's Single Kernel Inference**: A member shared that their friends at **Anyscale** are developing an inference engine that accomplishes entire LLM inference using a single **CUDA kernel**.
  
  - They invites opinions on how this approach compares to traditional inference engines, highlighting *a potential leap in efficiency*.
- **Question about CUDA Stream Synchronization**: A user asked why the **cudaStreamSynchronize** function is not employed for **stream1** & **stream2** prior to the kernel launch in a specific CUDA code.
  
  - *Clarification on CUDA synchronization techniques is needed*, underscoring the importance of understanding kernel execution order.

 

**Link mentioned**: [cuda-course/05_Writing_your_First_Kernels/05 Streams/01_stream_basics.cu at master · Infatoshi/cuda-course](https://github.com/Infatoshi/cuda-course/blob/master/05_Writing_your_First_Kernels/05%20Streams/01_stream_basics.cu): Contribute to Infatoshi/cuda-course development by creating an account on GitHub.

 

---

### **GPU MODE ▷ #**[**triton**](https://discord.com/channels/1189498204333543425/1189607595451895918/1298538475083272216) (47 messages🔥):

> - `Triton kernels performance`
> - `Kernel caching strategies`
> - `Kernel launch overhead`
> - `Dynamic input shapes in Triton`
> - `Use of heuristics in Triton`

- **Autotuning Triton kernels vs aten.mm**: A user is autotuning Triton kernels to compare their performance against **aten.mm**, noting that sometimes Triton is faster but overall performance is inconsistent.
  
  - Another member questioned the advantages of Triton if kernels are slower, prompting discussions on caching and kernel launch issues.
- **Kernel launch overhead concerns**: Members noted significant kernel launch overhead in Triton, particularly with smaller tensors, leading to suggestions to explore caching strategies to mitigate the issue.
  
  - A linked GitHub issue highlighted excessive launch times, with a suggestion to enable `use_cuda_graph=True` in autotune, although this was not suitable for dynamic input sizes.
- **Caching kernels for performance**: Users shared strategies for caching kernels to improve performance, finding that eliminating JIT compilation for every call led to noticeable speed improvements.
  
  - Discussions included how to implement caching for varying input sizes and ensuring shapes are compatible to avoid misaligned address errors.
- **Discussion on dynamic input sizes**: A member shared the limitations of using caching when input tensor sizes are not static, presenting challenges in managing launch metadata and kernel arguments.
  
  - Participants debated the impact of input shape variability on performance and kernel compilation times.
- **Heuristics for kernel optimization**: A member suggested using Triton's heuristics to manage meta-parameters, which could simplify kernel tuning for varying tensor sizes.
  
  - While some found this approach promising, others noted its complexity, suggesting alternative methods for caching kernels without overwhelming performance.

**Links mentioned**:

- [triton.heuristics — Triton documentation](https://triton-lang.org/main/python-api/generated/triton.heuristics.html): no description found
- [High kernel launch overhead · Issue #2637 · triton-lang/triton](https://github.com/triton-lang/triton/issues/2637): Hey team, I'm suffering high triton kernel launch overhead. Here's my nsys capture: The kernel executes around 80us on GPU, however, it takes 220us to launch, which causes the performance degr...

---

### **GPU MODE ▷ #**[**torch**](https://discord.com/channels/1189498204333543425/1189607750876008468/1298401208859365376) (7 messages):

> - `torch.autocast usage with autoquant`
> - `Library linking with c10`
> - `Speeding up PyTorch models`
> - `Compiling PyTorch code`
> - `Autocast behavior with FP32 and BF16`

- **Confusion Over torch.autocast and torchao**: One member expressed doubt about whether using autocast with **torchao autoquant** could interfere with its logic, as traditionally autocasting helps avoid precision loss.
  
  - *It’s suggested that autocast might introduce overhead,* especially in smaller batches, making FP16/BF16 casting potentially more efficient for inference.
- **Linking Libraries with c10**: A member had issues linking a header-only library against **c10**, despite trying to add it in the CMake configuration.
  
  - They mentioned attempting to use `target_link_libraries`, but it has not resolved the issue.
- **Seeking General Speed-Up Strategies for PyTorch**: Another member inquired about strategies to enhance **PyTorch** model performance while maintaining accuracy, suggesting tensor cores and autoquant as options.
  
  - Discussion indicated that using **autocast** may not be necessary and that lower precision types could be favored to minimize computational overhead.
- **Curiosity around Compiling PyTorch Code**: A user asked about the various methods available for compiling **PyTorch** code, mentioning tools like **Glow**, **nvFuser**, and **TorchDynamo**.
  
  - The query highlights a need for clarity regarding the distinct purposes of these compilers.
- **Doubt on Autocast's Impact on torchao's Quantization**: A member questioned whether **autocast** should be used with **torchao**, seeking further information on its effects on quantization.
  
  - They assumed that since autocast is typically used for FP16/BF16, which offers a higher dynamic range, it shouldn’t negatively impact **torchao**.

 

**Link mentioned**: [pytorch/aten/src/ATen/AccumulateType.h at b24c34426f47d6c311ad80ebba1d2575e6c7a6aa · pytorch/pytorch](https://github.com/pytorch/pytorch/blob/b24c34426f47d6c311ad80ebba1d2575e6c7a6aa/aten/src/ATen/AccumulateType.h#L58-L70)): Tensors and Dynamic neural networks in Python with strong GPU acceleration - pytorch/pytorch

 

---

### **GPU MODE ▷ #**[**algorithms**](https://discord.com/channels/1189498204333543425/1189861061151690822/) (1 messages):

.alphago: [https://x.com/detectiveenters/status/1752067011113546234](https://x.com/detectiveenters/status/1752067011113546234)

---

### **GPU MODE ▷ #**[**cool-links**](https://discord.com/channels/1189498204333543425/1189868872887705671/1298648282037485682) (3 messages):

> - `Mapper Generation Automation`
> - `Weight Nowcaster Networks`
> - `Neuron Interaction and Nowcasting`
> - `Training Optimization`
> - `Graph Neural Networks`

- **Automated Mapper Generation Outperforms Experts**: A new approach automates mapper generation for parallel programming, reportedly discovering mappers that outperform human expert designs in scientific applications by up to **1.34X speed** within ten minutes.
  
  - This method addresses the complexity of optimizing millions of decisions typically required to generate optimal mapping solutions.
- **Improving Neural Network Training with NiNo**: The recently proposed **Neuron Interaction and Nowcasting (NiNo)** networks enhance weight nowcaster networks, improving upon the previous method by incorporating neuron connectivity and **graph neural networks**.
  
  - This advancement allows NiNo to accelerate **Adam training**, addressing limitations encountered in some networks, particularly **Transformers**.

**Links mentioned**:

- [Improving Parallel Program Performance Through DSL-Driven Code Generation with LLM Optimizers](https://arxiv.org/abs/2410.15625): Mapping computations to processors and assigning data to memory are critical for maximizing performance in parallel programming. These mapping decisions are managed through the development of speciali...
- [Accelerating Training with Neuron Interaction and Nowcasting Networks](https://arxiv.org/abs/2409.04434): Neural network training can be accelerated when a learnable update rule is used in lieu of classic adaptive optimizers (e.g. Adam). However, learnable update rules can be costly and unstable to train ...

---

### **GPU MODE ▷ #**[**beginner**](https://discord.com/channels/1189498204333543425/1191300313928433664/1298365202848551034) (3 messages):

> - `CUDA Projects for Internships`
> - `Interactive Environments for CUDA Kernels`

- **CUDA Projects for Internships abound**: A member shared their interest in working on **CUDA accelerated linear and logistic regression** for their summer internship resume, seeking project suggestions.
  
  - *Their friend laughed* and redirected them to a [server link](https://link.to.server) for more ideas.
- **Exploring Interactive Kernels with Jupyter**: Another member inquired about **interactive environments for CUDA kernels**, suggesting Cython or Jupyter notebooks to run C code and manipulate outputs in Python.
  
  - *They expressed that this setup seemed like* the most realistic scenario while seeking input from others on the problem.

 

---

### **GPU MODE ▷ #**[**pmpp-book**](https://discord.com/channels/1189498204333543425/1194427148656721970/) (1 messages):

vim410: This whole chapter is getting rewritten. Once i have the new chapter will share it.

---

### **GPU MODE ▷ #**[**torchao**](https://discord.com/channels/1189498204333543425/1205223658021458100/1298584546526101585) (7 messages):

> - `int4_mm wrapping for torch.compile`
> - `torch.library.custom_op vs @register_meta`
> - `nightly wheels for non-Linux`

- **Exploration of** `int4_mm` Wrapping for torch.compile: A member is curious about how `int4_mm` is wrapped for `torch.compile` support, suspecting it should use `custom_op` but finding it utilizes `@register_meta` instead ([link](https://github.com/pytorch/pytorch/blob/8fbf866904661b16cba4c799af81121557ba9da8/torch/_meta_registrations.py#L3275)).
  
  - *Is there any advantage of this approach vs.* `torch.library.custom_op`? was raised, debating the role of `torch._check` for tensor size and dtype.
- **Differences between** `torch.library.custom_op` and Low-Level API: Members discussed that `torch.library.custom_op` serves as a high-level wrapper, while some cases might benefit from the low-level API ([link](https://github.com/pytorch/pytorch/blob/c2d26418c39f9562e128efae32eace61c703ccd7/torch/_library/custom_ops.py)).
  
  - The inquiry about using high-level APIs when direct options exist surfaced a debate on the necessity of such custom implementations.
- **Nightly Wheels Limitations for Non-Linux Platforms**: A member pointed out that nightly wheels for platforms other than Linux lack `.dev<date>` versions ([link](https://download.pytorch.org/whl/nightly/torchao/)).
  
  - *Interesting, mind opening an issue?* was suggested as a follow-up to the observation, indicating potential concerns for broader platform support.

**Links mentioned**:

- [pytorch/torch/_meta_registrations.py at 8fbf866904661b16cba4c799af81121557ba9da8 · pytorch/pytorch](https://github.com/pytorch/pytorch/blob/8fbf866904661b16cba4c799af81121557ba9da8/torch/_meta_registrations.py#L3275): Tensors and Dynamic neural networks in Python with strong GPU acceleration - pytorch/pytorch
- [no title found](https://download.pytorch.org/whl/nightly/torchao/): no description found
- [pytorch/torch/_library/custom_ops.py at c2d26418c39f9562e128efae32eace61c703ccd7 · pytorch/pytorch](https://github.com/pytorch/pytorch/blob/c2d26418c39f9562e128efae32eace61c703ccd7/torch/_library/custom_ops.py): Tensors and Dynamic neural networks in Python with strong GPU acceleration - pytorch/pytorch
- [pytorch/torch/_meta_registrations.py at 8fbf866904661b16cba4c799af81121557ba9da8 · pytorch/pytorch](https://github.com/pytorch/pytorch/blob/8fbf866904661b16cba4c799af81121557ba9da8/torch/_meta_registrations.py#L47-L57): Tensors and Dynamic neural networks in Python with strong GPU acceleration - pytorch/pytorch

---

### **GPU MODE ▷ #**[**rocm**](https://discord.com/channels/1189498204333543425/1233704710389764236/1298465956309241928) (8 messages🔥):

> - `ROCm vs ROCm/Triton/FA benchmarks`
> - `RCCL contributions`
> - `PyTorch symmetric memory`
> - `Matrix multiplication using Triton`

- **ROCm/Triton/FA benchmarks inquiry**: A member inquired whether **ROCm/Triton/FA** is faster than **ROCm/FA**, expressing curiosity about the benchmark numbers.
  
  - Another member mentioned that they haven't checked but would be willing to merge a PR if someone runs the benchmarks.
- **Interest in RCCL contributions**: A member expressed their interest in helping out with **RCCL** development.
  
  - This enthusiasm was met with a supportive acknowledgment from others in the channel.
- **Discussion on PyTorch symmetric memory**: A question was raised regarding **PyTorch symmetric memory**, with a reference to its relation to the previous discussion.
  
  - A direct confirmation was provided, clarifying that the topic connected with the ongoing developer discourse.
- **Matrix multiplication implementations using Triton**: A user queried if there’s a trivial way to implement matrix multiplication using **Triton** to fully utilize a **MI250x** with two GCDs.
  
  - This reflects ongoing explorations and implementations aimed at enhancing performance with Triton.

 

---

### **GPU MODE ▷ #**[**liger-kernel**](https://discord.com/channels/1189498204333543425/1275130785933951039/1298376433298837577) (8 messages🔥):

> - `Monkey Patching CrossEntropy`
> - `Version Compatibility Challenges`
> - `Liger's Kernel Fusion`
> - `Inference Optimization Techniques`
> - `Gradient Accumulation Fix Discussion`

- **Monkey Patching CrossEntropy faces challenges**: The current monkey patching strategy for **CrossEntropyLoss** in transformers may not be effective with the latest **GA patch version**, as lights were shed on the transition to `self.loss_function` in **CausalLMs**.
  
  - The root **CrossEntropy** function can be found [here](https://github.com/huggingface/transformers/blob/049682a5a63042f087fb45ff128bfe281b2ff98b/src/transformers/loss/loss_utils.py#L26).
- **Liger's potential for kernel fusion**: A participant noted that **Liger** can be beneficial as it fuses the kernel but suggested using **vllm/sglang** for better optimization strategies like **paged attention** and **flash-decoding kernel**.
  
  - They emphasized the distinction between **inference** and **training optimization**, highlighting the need for specificity in approaches.
- **Version compatibility check needed**: There's a suggestion to check the **HF transformers version** and implement two different patches due to recent changes in the framework which affect monkey patching capabilities.
  
  - The interaction ended with a commitment to look into it further, ensuring it remains **backward compatible**.
- **Concerns over unresolved issues**: Another member expressed concerns about the latest patch and whether it indeed mitigated existing bugs, referencing [this pull request](https://github.com/huggingface/transformers/pull/34283) discussing gradient accumulation fixes.
  
  - There are fears that this fix might not sufficiently resolve the ongoing issues users reported, showcasing the complexities of patching in large frameworks.

**Links mentioned**:

- [transformers/src/transformers/loss/loss_utils.py at 049682a5a63042f087fb45ff128bfe281b2ff98b · huggingface/transformers](https://github.com/huggingface/transformers/blob/049682a5a63042f087fb45ff128bfe281b2ff98b/src/transformers/loss/loss_utils.py#L26): 🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX. - huggingface/transformers
- [Enable Gradient Accumulation fix across all models + trainer fully in forward() by muellerzr · Pull Request #34283 · huggingface/transformers](https://github.com/huggingface/transformers/pull/34283): What does this PR do? Since most users still want OOTB, this trickles the loss kwargs to the rest of the models so that causal loss can be calculated properly Fixes # (issue) Fully fixes #34263 / f...

---

### **GPU MODE ▷ #**[**self-promotion**](https://discord.com/channels/1189498204333543425/1288557096404516945/1298662010179027045) (3 messages):

> - `Triton kernels for quantization`
> - `NVIDIA Virtual Connect with Experts`
> - `cuDF and cuML`
> - `RAPIDS developers panel`

- **Tutorial on Triton Kernels Announced**: A tutorial on writing **Triton kernels** for quantization will be held in 2 weeks on GPU MODE, although no recording of prior events is available.
  
  - Further details can be found during the [tutorial session](https://discord.gg/sQ7zJ94M?event=1289331107745108079).
- **Upcoming NVIDIA Virtual Connect with Experts Event**: The next **NVIDIA Virtual Connect with Experts** event is scheduled for **Friday, October 25, 2024 at 10am PT**, focusing on **cuDF and cuML**.
  
  - Attendees are encouraged to check the [event page on GitHub](https://github.com/NVIDIA/accelerated-computing-hub/tree/main/connect-with-experts) for updates and to spread the word among their networks.

 

**Link mentioned**: [accelerated-computing-hub/connect-with-experts at main · NVIDIA/accelerated-computing-hub](https://github.com/NVIDIA/accelerated-computing-hub/tree/main/connect-with-experts): NVIDIA curated collection of educational resources related to general purpose GPU programming. - NVIDIA/accelerated-computing-hub

 

---

### **GPU MODE ▷ #**[**🍿**](https://discord.com/channels/1189498204333543425/1298372518293274644/1298375128786276484) (54 messages🔥):

> - `Creating an LLM for Efficient Kernels`
> - `CUDABench Proposal`
> - `Community Authorship and Contributions`
> - `Optimization Techniques for CUDA Kernels`
> - `Game Development Analogies in GPU Programming`

- **Creating an LLM for Efficient Kernels**: The team is working on developing an LLM to generate efficient CUDA kernels, focusing on creating a mega prompt to explain GPU functions by NeurIPS 2024.
  
  - They plan to build a large kernel dataset and conduct everything in public, leveraging community input.
- **CUDABench Proposal**: PhD students presented a proposal for CUDABench, a standardized benchmark to assess LLMs' CUDA code generation abilities, which aims to crowdsource problems and ideas.
  
  - The design encourages compatibility with various DSLs while maintaining a focus on torch inline CUDA kernels.
- **Community Authorship and Contributions**: The project is being developed as a community-authored paper, with opportunities for contributors to become authors based on their contributions.
  
  - The goal is to ensure a broad range of community inputs, especially for datasets and coding efforts.
- **Optimization Techniques for CUDA Kernels**: Discussion emerged around optimizing existing kernels for target accelerators and the potential use of intermediate tools like Kernel Tuner.
  
  - Members expressed that while broader optimizations should be on the roadmap, the initial focus should remain on generating efficient CUDA code.
- **Game Development Analogies in GPU Programming**: Members compared the approach of developing CUDA-centric tools to game development, advocating for starting with a specific target and refining abstractions over time.
  
  - This analogy reflects on how programming GPUs for LLMs could benefit from focusing on the most logical target first, before expanding to other architectures.

**Links mentioned**:

- [Examples — Hidet Documentation](https://hidet.org/docs/stable/hidet-script/examples/index.html#hidet-script-examples): no description found
- [GitHub - KernelTuner/kernel_tuner: Kernel Tuner](https://github.com/KernelTuner/kernel_tuner): Kernel Tuner. Contribute to KernelTuner/kernel_tuner development by creating an account on GitHub.
- [CUDABench Design](https://docs.google.com/document/d/1ZNvShNH44zuy3LwbRdMigGsuCzO4i5Yl2fgAaSDynTg/edit?usp=sharing)): CUDABench Anne Ouyang1 (aco@stanford.edu), Simon Guo1 (simonguo@stanford.edu) 1: Stanford University Motivation and Problem Statement Efficient CUDA kernels are critical for maximizing the performanc...
- [CUDABench Problem Ideas Crowdsourcing](https://docs.google.com/forms/d/e/1FAIpQLSeiqz2bLreIKY8maWCaaNIU-aXC0MfOMOog0bwS5J_zzNaLVQ/viewform?usp=sf_link)): CUDABench Design Doc: https://docs.google.com/document/d/1ZNvShNH44zuy3LwbRdMigGsuCzO4i5Yl2fgAaSDynTg/edit?tab=t.0#heading=h.4qj5vtu1o7mr There's a lot of interest in using LLMs to generate CUDA ...
- [Project Popcorn 🍿 (1).pptx](https://docs.google.com/presentation/d/1ir6br01sVY5wLqUz-qz4OE4nMSJbQBSp/edit?usp=sharing&ouid=106222972308395582904&rtpof=true&sd=true): Project Popcorn
- [TK + Monkeys + CUDAGen](https://docs.google.com/presentation/d/1JtxGXv80ciIne-bFxySZ25q0J2mAwsXlb9uuST9naqg/edit?usp=sharing): ThunderKittens A simple framework for AI kernels
- [Monkeys_for_Meta_v3.pptx](https://docs.google.com/presentation/d/14jlbVPyohnWuQgFikr74cnaj-mzoEMPT/edit?usp=sharing&ouid=111422880520483065413&rtpof=true&sd=true): Large Language Monkeys: Scaling Inference-Time Compute with Repeated Sampling Brad Brown\*, Jordan Juravsky\*, Ryan Ehrlich\*, Ronald Clark, Quoc Le, Chris Ré, Azalia Mirhoseini
- [META KERNELS - Google Drive](https://drive.google.com/drive/folders/1nt2KcRRKb8YdySxkRxUu5PR4c7UPM_rK): no description found

---

### **Nous Research AI ▷ #**[**announcements**](https://discord.com/channels/1053877538025386074/1145143867818119272/1298699185234645064) (1 messages):

> - `Hermes 70B API`
> - `Revenue-sharing partnership`
> - `Hyperbolic launch`
> - `AI Inference Service`

- **Hermes 70B API launched on Hyperbolic**: The **Hermes 70B API** is now available on Hyperbolic, providing greater access to large language models for developers and businesses. For more details, check out the announcement [here](https://x.com/hyperbolic_labs/status/1849130421885514231?s=46).
  
  - This launch marks a significant step towards making powerful AI tools more accessible to everyone.
- **Nous Research partners with Hyperbolic**: Hyperbolic announced a **revenue-sharing partnership** with **Nous Research**, creators of the Hermes 3 model. This collaboration aims to drive shared revenue through Hyperbolic's AI Inference Service.
  
  - *The future of AI is collaborative* as stated by the Hyperbolic team, emphasizing the power of partnerships in tech.

 

**Link mentioned**: [Tweet from Hyperbolic (@hyperbolic_labs)](https://x.com/hyperbolic_labs/status/1849130421885514231?s=46): Hyperbolic is making AI more accessible to everyone. Today, we’re launching a revenue-sharing partnership with @NousResearch, creators of the Hermes 3 large language model. Through our AI Inference S...

 

---

### **Nous Research AI ▷ #**[**general**](https://discord.com/channels/1053877538025386074/1149866623109439599/1298371469893697616) (119 messages🔥🔥):

> - `Nous Research Forge`
> - `Hermes AI Censorship`
> - `Claude Automation`
> - `AI Role-Playing`
> - `Grunt Work Opportunities`

- **Excitement for Forge Project**: Members expressed their enthusiasm for the project 'Forge', highlighted in a [YouTube video](https://www.youtube.com/watch?v=zmnzW0r_g8k&list=PLjOo65uE) featuring Nous Research co-founder Karan.
  
  - A discussion followed about knowledge graph implementation related to the project.
- **Debate on Hermes AI Censorship**: Members debated the extent of censorship in Hermes AI, with some suggesting that certain providers embed censorship in their system prompts.
  
  - Others argued that using proper system prompts can yield a range of behaviors, indicating varying degrees of censorship across models.
- **Claude's Automation Feature**: Claude was discussed in relation to its automation capabilities, particularly around cursor movement and web browsing.
  
  - Members raised concerns about the extent of censorship affecting functionality, although there were indications of less prudeness than other models.
- **AI Role-Playing Dynamics**: The dynamics of AI role-playing were explored, particularly how system prompts influence the responses of AI models in various scenarios.
  
  - Members discussed the potential for models to exhibit chaotic behavior if instructed in certain ways, challenging the idea of inherent censorship.
- **Seeking Grunt Work**: A user expressed a need for immediate employment through grunt work, such as data entry or labeling, while highlighting their difficult situation.
  
  - The community offered solidarity and suggestions for platforms like Uber Eats to find immediate work, while discussing financial challenges.

**Links mentioned**:

- [Forge by Nous Research @ Nouscon 2024](https://www.youtube.com/watch?v=zmnzW0r_g8k&list=PLjOo65uEP4cYhV7c2whkhDfWy58XFj7yL&index=8&t=514s): Nous Research co-founder Karan talks about one of our upcoming projects, "Forge" @ Nouscon 2024.
- [Forge by Nous Research @ Nouscon 2024](https://www.youtube.com/watch?v=zmnzW0r_g8k&list=PLjOo65uE): Nous Research co-founder Karan talks about one of our upcoming projects, "Forge" @ Nouscon 2024.
- [anthropic-quickstarts/computer-use-demo/computer_use_demo/streamlit.py at main · anthropics/anthropic-quickstarts](https://github.com/anthropics/anthropic-quickstarts/blob/main/computer-use-demo/computer_use_demo/streamlit.py): A collection of projects designed to help developers quickly get started with building deployable applications using the Anthropic API - anthropics/anthropic-quickstarts

---

### **Nous Research AI ▷ #**[**ask-about-llms**](https://discord.com/channels/1053877538025386074/1154120232051408927/1298390407016878130) (20 messages🔥):

> - `Claude's new system prompt`
> - `Finetuning models like Gemma 2`
> - `Whisper-based translation frameworks`
> - `Llama model quantization`
> - `Support for JSON data parsing`

- **Claude's System Prompt Enhancements**: The new Claude includes a system prompt addition that corrects the 'misguided attention' issue, as shared by another user who extracted it.
  
  - *Claude also endeavors to clarify puzzle constraints, yet sometimes misinterprets questions due to oversight.*
- **Improving Model Self-Reflection**: Users noticed enhancements in Claude’s self-reflection abilities, with responses becoming more refined when addressing logical puzzles.
  
  - *There was a humorous instance where Claude initially attempted to answer a puzzle incorrectly before correcting itself.*
- **Exploring Whisper Streaming Solutions**: A user sought offline real-time Whisper-based translation frameworks, prompting discussions on popular repositories like [whisper_streaming](https://github.com/ufal/whisper_streaming).
  
  - *Another suggestion included the new* [*moonshine*](https://github.com/usefulsensors/moonshine) *project, providing fast ASR for edge devices.*
- **Discussion on Small OSS Models**: When queried about powerful small OSS models, **Gemma 2** and **Qwen 2.5** were recommended, with specifications regarding memory requirements noted.
  
  - *Concerns about the models handling semi-complex JSON data were raised, with some users unsure about experiences without additional parsing.*
- **Llama Model Quantization Queries**: Users discussed finding Llama 3.2 quant versions but faced challenges locating safetensors versions from GGUF formats.
  
  - *This led to inquiries about converting GGUF to safetensors using scripts, pointing to an ongoing interest in model optimization.*

**Links mentioned**:

- [GitHub - ufal/whisper_streaming: Whisper realtime streaming for long speech-to-text transcription and translation](https://github.com/ufal/whisper_streaming): Whisper realtime streaming for long speech-to-text transcription and translation - ufal/whisper_streaming
- [GitHub - usefulsensors/moonshine: Fast and accurate automatic speech recognition (ASR) for edge devices](https://github.com/usefulsensors/moonshine): Fast and accurate automatic speech recognition (ASR) for edge devices - usefulsensors/moonshine

---

### **Nous Research AI ▷ #**[**research-papers**](https://discord.com/channels/1053877538025386074/1104063238934626386/) (1 messages):

feffy: p-hacking :P

---

### **Nous Research AI ▷ #**[**interesting-links**](https://discord.com/channels/1053877538025386074/1132352574750728192/1298367633045061692) (5 messages):

> - `ZK Proofs and ChatGPT ownership`
> - `Advancements in ZK technology`
> - `Genmo AI motion quality`
> - `Prompt adherence in AI generation`
> - `Human action generation`

- **ZK Proofs Allow ChatGPT History Ownership**: The latest application from OpenBlock’s Universal Data Protocol (UDP) empowers ChatGPT users to own their chat history while enhancing the availability of verifiable training data for open-source models.
  
  - This approach marks a significant step in improving data provenance and interoperability in AI training.
- **ZK technology speeds up proof generation**: A member clarified that ZK proofs take a few seconds on the server-side, with some UDP proofs now taking less than a second due to advancements in infrastructure from @zkemail.
  
  - An example was shared [here](https://x.com/paulsengh/status/1846657020868677931) illustrating the rapid progress in ZK technology.
- **Genmo AI delivers unmatched motion quality**: Genmo AI claims to provide realistic motion that adheres to the laws of physics, creating high-quality animations down to tiny details.
  
  - Their technology promises superior alignment of generated videos with detailed textual prompts, enhancing user control.
- **Crossing the uncanny valley with Mochi 1**: Mochi 1 boasts the capability to generate consistent and fluid human actions and expressions, pushing the boundaries of realistic animation.
  
  - This advancement is pivotal in creating videos that resonate well with audience expectations of human motion.

**Links mentioned**:

- [Tweet from OpenBlock (@openblocklabs)](https://x.com/openblocklabs/status/1848805457290572199): 1/ Introducing Proof of ChatGPT, the latest application built on OpenBlock’s Universal Data Protocol (UDP). This Data Proof empowers users to take ownership of their LLM chat history, marking a signi...
- [Tweet from Paul Sengh (@paulsengh)](https://x.com/paulsengh/status/1846657020868677931): It’s incredible how quickly ZK technology has advanced—some UDP proofs now take less than a second, thanks to infra from @zkemail. Try it out: https://bridge.openblocklabs.com/
- [Genmo. The best open video generation models.](https://www.genmo.ai/): Genmo trains the world's best open video generation models. Create incredible videos with AI at Genmo

---

### **Nous Research AI ▷ #**[**research-papers**](https://discord.com/channels/1053877538025386074/1104063238934626386/) (1 messages):

feffy: p-hacking :P

---

### **Eleuther ▷ #**[**general**](https://discord.com/channels/729741769192767510/729741769738158194/1298363256087248946) (35 messages🔥):

> - `Chess Move Explainability`
> - `Accusations of Cheating in Chess`
> - `LLMs Self-Explanation Accuracy`
> - `Molmo Vision Models`
> - `DINOv2 Understanding`

- **Chess Players and Move Explanation**: Most top chess players can explain the *motivation* behind engine moves, but their ability to rank lines in complex positions remains in question.
  
  - The distinction between what makes a move ideal for humans versus engines complicates the understanding of optimal play.
- **Drama in Chess Community**: A former world champion has accused a popular streamer of cheating, mainly based on the explanations given during live commentary.
  
  - This incident highlights the ongoing conversation about the validity of move explanations and the pressure on commentators.
- **LLMs and Their Self-Explanations**: Concerns were raised about the accuracy of self-explanations given by LLMs, especially when they lack contextual understanding.
  
  - This consideration leads to the exploration of how better training data could enhance explanation authenticity.
- **Upcoming Molmo Vision Models**: The Molmo project is expected to release open vision-language models trained on the PixMo dataset, featuring multiple checkpoints.
  
  - These models aim to achieve state-of-the-art performance among multimodal models while remaining fully open-source.
- **Learning about DINOv2**: A user sought resources to understand DINOv2, and was directed to a relevant research paper for more information.
  
  - The paper outlines the methodology behind DINOv2 and is authored by experts in the field.

**Links mentioned**:

- [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193): The recent breakthroughs in natural language processing for model pretraining on large quantities of data have opened the way for similar foundation models in computer vision. These models could great...
- [allenai/Molmo-7B-D-0924 · Hugging Face](https://huggingface.co/allenai/Molmo-7B-D-0924): no description found
- [Alignment Workshop - Been Kim - Alignment and Interpretability: How we might get it right](https://www.alignment-workshop.com/nola-talks/been-kim-alignment-and-interpretability-how-we-might-get-it-right): Transcript Thanks for the generous introduction. I'm excited to be here and give this talk about alignment and interpretability. A good place to start this talk is by defining what value alignment...

---

### **Eleuther ▷ #**[**research**](https://discord.com/channels/729741769192767510/747850033994662000/1298423210039050322) (45 messages🔥):

> - `RoPE 2D Encoding`
> - `Building Open Source Datasets`
> - `Transformers and LayerNorms`
> - `Softmax Attention Adaptations`

- **Debate on RoPE Extension to 2D**: Members discussed whether extending **RoPE** to 2D should move away from complex numbers and utilize direct 2D coordinates instead.
  
  - Concerns were raised regarding the encoding of relative positions, emphasizing the necessity of using both **cos** and **sin** per frequency.
- **Call for Open Source Action Modality Dataset**: There’s a conversation around the need for an **opensourcable action modality dataset**, suggesting that mining tests from open-source web frameworks could be beneficial.
  
  - The feasibility of using human-labeled data and **puppeteer scripts** to create such a dataset was positively received.
- **LayerNorm Impact on Model Interpretation**: Discussion revolved around the potential of training models without **LayerNorms**, citing that while it can remove bad mechanical interpretation properties, the architecture need not be restricted to fine-tuning.
  
  - Some members expressed interest in methods to improve model interpretability without adversely affecting downstream applications.
- **Innovating Softmax Attention Mechanisms**: Participants suggested exploring adaptations of **softmax attention** to improve efficiency and prevent mechanical interpretation issues, such as linearization or converting methods to **top-k**.
  
  - The approach aims to retain functional integrity while enhancing interpretability and allowing for the implementation of qualitatively better methods.

**Links mentioned**:

- [Analyzing and Improving the Training Dynamics of Diffusion Models](https://arxiv.org/abs/2312.02696): Diffusion models currently dominate the field of data-driven image synthesis with their unparalleled scaling to large datasets. In this paper, we identify and rectify several causes for uneven and ine...
- [Storybook: Frontend workshop for UI development](https://storybook.js.org/): Storybook is a frontend workshop for building UI components and pages in isolation. Thousands of teams use it for UI development, testing, and documentation. It's open source and free.

---

### **Eleuther ▷ #**[**interpretability-general**](https://discord.com/channels/729741769192767510/1052314805576400977/1298589776588443648) (2 messages):

> - `AI-Driven Observability Interface`
> - `Interpretability Startups`
> - `Technical Demonstrations`
> - `Jacob Steinhardt`
> - `Sarah Schwettmann`

- **Introducing Monitor: AI Observability Interface**: [Monitor](https://transluce.org/) is an AI-driven interface designed to help humans observe, understand, and steer computations inside models, aiming to improve interpretability.
  
  - The project, led by **Jacob Steinhardt** and **Sarah Schwettmann**, aims to provide tools for better model comprehension as of **23 October 2024**.
- **Appreciation for Collaboration**: A member expressed gratitude for the contributions of others in putting together the information shared in the **baulab** channel.
  
  - This collaborative effort highlights the community's commitment to advancing AI interpretability discussions.

 

**Link mentioned**: [Transluce](https://transluce.org/): no description found

 

---

### **Eleuther ▷ #**[**lm-thunderdome**](https://discord.com/channels/729741769192767510/755950983669874798/1298439052672110693) (7 messages):

> - `simple_evaluate tasks`
> - `evaluating small models`
> - `Pile tasks error`

- **Curious about** `simple_evaluate` Tasks: `simple_evaluate` is believed to support all tasks, but a member inquired about its capability for small models, specifically looking for evaluations like Pile PPL or Lambada.
  
  - Another member indicated that <@981242445696221224> and <@1042521538919923763> may provide more insights on good tasks for small model evaluations.
- **Issues with** `pile_10k` Validation: A user reported an error while running the `pile_10k` validation and questioned if Pile tasks are indeed supported.
  
  - A member confirmed that there was an issue with the hosting provider and noted that the default URL for Pile tasks currently does not point to anything functional.

 

---

### **Latent Space ▷ #**[**ai-general-chat**](https://discord.com/channels/822583790773862470/1075282825051385876/1298379047054606398) (49 messages🔥):

> - `Anthropic Mouse Generator`
> - `Ideogram Canvas Features`
> - `AI and Loneliness Epidemic`
> - `Speculative Decoding in vLLM`
> - `New Meeting Automation Tools`

- **Anthropic Mouse Generator Demonstrates AI Agent Capability**: A colleague showcased the new **Anthropic Mouse Generator**, impressing observers with its ability to install and debug software autonomously.
  
  - However, it still requires specific instructions and cannot perform tasks like playing chess without guidance.
- **Ideogram Canvas Rivals Existing Tools**: Discussions around **Ideogram Canvas** highlighted its innovative features like **Magic Fill** and **Extend**, enabling users to edit and combine images seamlessly.
  
  - Some participants suggested it poses a competitive threat to platforms like Canva due to its advanced capabilities.
- **AI's Role in the Loneliness Epidemic**: The tragic case of a 14-year-old's suicide raised concerns about the influence of AI on loneliness, sparking discussions on mental health and technology's roles.
  
  - Participants debated whether AI could serve as a connection tool or if it exacerbates feelings of isolation, with diverse opinions on its effectiveness.
- **Speculative Decoding Enhancements in vLLM**: A new blog discussed **speculative decoding** in **vLLM**, a technique that accelerates token generation using both small and large models.
  
  - This approach aims to improve performance and integrate new techniques for optimizing AI functionality.
- **New Meeting Automation Tools Released**: A new app called **agent.exe** has been launched, allowing users to control their computers using **Claude 3.5 Sonnet**.
  
  - This development signifies a growing interest in AI agents, with expectations for increased automation and efficiency in 2025.

**Links mentioned**:

- [iPhone 16 orders cut by around 10 million units for 4Q24–1H25; no evidence yet that Apple…](https://medium.com/@mingchikuo/iphone-16-orders-cut-by-around-10-million-units-for-4q24-1h25-no-evidence-yet-that-apple-48c126a33bc6): Latest industry survey:
- [Tweet from .txt (@dottxtai)](https://x.com/dottxtai/status/1848783015222169726): We’ve been cooking with @huggingface and just released a Rust port of Outlines’ structured generation. 👉 Faster compilation 👉 Lightweight library (poke @vllm_project) 👉 Bindings in many languages...
- [Tweet from Chris Pedregal (@cjpedregal)](https://x.com/cjpedregal/status/1849118877642526966?s=46): Writing is thinking. We launched @meetgranola because we don’t want meeting bots to think for us. Turns out, a lot of people felt the same way. Excited to announce our $20 million Series A led by ...
- [Tweet from Character.AI (@character_ai)](https://x.com/character_ai/status/1849055407492497564): We are heartbroken by the tragic loss of one of our users and want to express our deepest condolences to the family. As a company, we take the safety of our users very seriously and we are continuing ...
- [How Speculative Decoding Boosts vLLM Performance by up to 2.8x](https://blog.vllm.ai/2024/10/17/spec-decode.html): Speculative decoding in vLLM is a powerful technique that accelerates token generation by leveraging both small and large models in tandem. In this blog, we’ll break down speculative decoding in vLLM,...
- [Tweet from Kyle Corbitt (@corbtt)](https://x.com/corbtt/status/1849127639866626171?s=46&t=tMWvmS3OL3Ssg0b9lKvp4Q): As a side note, the new Claude 3.5 is incredible for coding as well. This is my first Electron app, and Claude +Cursor could consistently build complex functionality across multiple files in a single ...
- [Tweet from Kyle Corbitt (@corbtt)](https://x.com/corbtt/status/1849124800838713844?s=46): Just launched agent.exe, a free, open-source Mac/Windows/Linux app that lets you use Claude 3.5 Sonnet to control your computer! This was a fun little project to explore the API and see what the mode...
- [Tweet from Andrew Wilkinson (@awilkinson)](https://x.com/awilkinson/status/1849216089676460122): How freaking cool is this: I made a Lindy (@getlindy) AI agent that texts me a meeting briefing 30 minutes before each meeting. It reviews their LinkedIn for a bio + our recent emails for context. ...
- [Ideogram Canvas, Magic Fill, and Extend](https://about.ideogram.ai/canvas): Ideogram Canvas is an infinite creative board for organizing, generating, editing, and combining images. Bring your face or brand visuals to Ideogram Canvas and use industry-leading Magic Fill and Ext...
- [Joel Lewenstein - Pursuing ambitious design ideas](https://share.snipd.com/episode/062696e2-2976-4fac-83f1-2941488c7fbf): Joel Lewenstein - Pursuing ambitious design ideas
- [Tweet from James Grugett (@jahooma)](https://x.com/jahooma/status/1848401531491783135): I just quit my $100k job to join the first Fall YC batch. And we're building a @cursor_ai competitor! Sound familiar? This time, we wrote all the code ourselves 😉 Ask me anything! Quoting Y C...
- [anthropic-quickstarts/computer-use-demo at main · anthropics/anthropic-quickstarts](https://github.com/anthropics/anthropic-quickstarts/tree/main/computer-use-demo): A collection of projects designed to help developers quickly get started with building deployable applications using the Anthropic API - anthropics/anthropic-quickstarts
- [Introducing Act-One | Runway](https://youtu.be/z3F0ei62Kmk): Introducing, Act-One. A new way to generate expressive character performances inside Gen-3 Alpha using a single driving video and character image. No motion ...
- ['He Would Still Be Here': Man Dies by Suicide After Talking with AI Chatbot, Widow Says](https://www.vice.com/en/article/man-dies-by-suicide-after-talking-with-ai-chatbot-widow-says/): The incident raises concerns about guardrails around quickly-proliferating conversational AI models.
- [Loneliness and suicide mitigation for students using GPT3-enabled chatbots - npj Mental Health Research](https://www.nature.com/articles/s44184-023-00047-6): no description found
- [Character.ai Faces Lawsuit After Teen’s Suicide - The New York Times](https://archive.fo/2zp1e): no description found

---

### **LlamaIndex ▷ #**[**blog**](https://discord.com/channels/1059199217496772688/1187460979064324127/1298366825679163512) (4 messages):

> - `Llama Impact Hackathon`
> - `Box AI integration`
> - `Multi-agent concierge system`
> - `Building LLM-powered web apps with LlamaIndex.TS`

- **Join the Llama Impact Hackathon for AI Solutions**: Participate in the 3-day [Llama Impact Hackathon](https://t.co/G01c8eIN1j) in San Francisco from November 8-10, with a total prize pool of **$15,000**, including a special **$1000** prize for best use of LlamaIndex.
  
  - It offers both in-person and online options to build AI solutions using **Meta's Llama 3.2 models**.
- **Box AI and LlamaIndex Work Together Seamlessly**: Utilize **Box AI** to query documents without downloading and extract structured data from unstructured content, while easily integrating it with LlamaIndex agents.
  
  - Learn more about how [Box AI](https://t.co/M9f81GiMGp) can enhance your workflows with LlamaIndex in a collaborative approach.
- **Build Advanced Customer Service Bots**: A new update allows you to build a **multi-agent concierge system** incorporating tool calling, memory, and human collaboration tailored for customer service applications.
  
  - The overhauled features enable users to iterate and improve their customer service bots effectively, as highlighted by [Logan Markewich](https://t.co/PWshlAyeKV).
- **Develop LLM-Powered Apps with LlamaIndex.TS**: LlamaIndex.TS is now part of [Vercel's AI SDK](https://t.co/BgCvo2Rxj6), providing an easier way to stream responses back to your front-end using just one line of code.
  
  - It allows developers to create **LLM-powered applications** across popular runtimes like Node.js, with examples available for integration.

 

**Link mentioned**: [Adapters: LlamaIndex](https://t.co/BgCvo2Rxj6): Learn how to use LlamaIndex with the Vercel AI SDK.

 

---

### **LlamaIndex ▷ #**[**general**](https://discord.com/channels/1059199217496772688/1059201661417037995/1298372897689174068) (31 messages🔥):

> - `Persistent Context in Workflows`
> - `Analyzing Long YouTube Lectures`
> - `Using Anthropic LLM`
> - `Progress Stats in SimpleDirectoryReader`
> - `LlamaIndex Workflow Compatibility with Redpanda`

- **Persistent Context in Workflows**: A user inquired about enabling **Context** to persist over multiple runs of the same workflow, prompting a discussion on serialization options.
  
  - The response included examples of using **JsonSerializer** for serialization, allowing the context to be resumed later.
- **Efficient Analysis of Long YouTube Lectures**: A member discussed building a tool to analyze lengthy YouTube lectures and faced challenges with managing large context sizes.
  
  - Suggestions included summarizing the context or implementing a **retrieval-based approach** to maintain efficiency.
- **Migrating to Anthropic LLM**: Another user was trying to replace the default ChatGPT with the Anthropic LLM but encountered issues with OpenAI API key prompts.
  
  - Responses mentioned that to fully transition to Anthropic, a local embedding model is needed to avoid reliance on OpenAI’s embedding.
- **Progress Stats in SimpleDirectoryReader**: A user asked about displaying progress or timing stats in **SimpleDirectoryReader** while ingesting multiple PDFs.
  
  - The discussion highlighted that while direct timing stats aren't available, a progress bar could indicate how many PDFs have been processed.
- **LlamaIndex Workflow Compatibility with Redpanda**: One user questioned if the **LlamaIndex Workflow** could be integrated with Redpanda or if it required Confluent.
  
  - The response indicated uncertainty regarding Redpanda's compatibility, suggesting a lack of firsthand experience with it.

 

**Link mentioned**: [Gemini 1.5 Pro 2M context window, code execution capabilities, and Gemma 2 are available today](https://developers.googleblog.com/en/new-features-for-the-gemini-api-and-google-ai-studio/): no description found

 

---

### **tinygrad (George Hotz) ▷ #**[**general**](https://discord.com/channels/1068976834382925865/1068976834928193609/1298466750416818258) (5 messages):

> - `Tensor int64/uint64 support`
> - `SHA3 Implementation`
> - `Tensor.ones and numpy`

- **Clarification on Tensor int64/uint64 support**: *Is it correct that Tensors don't support int64/uint64 yet?* This question arose during a discussion on implementing SHA3.
  
  - Another member confidently stated that **it is supported**, directing them to look at `dtype.py` for confirmation.
- **Error with Tensor.ones and reshaping**: A user reported a problem using `print(Tensor.ones(5, 5, dtype=dtypes.int64).numpy())`, receiving a **ValueError** stating it cannot reshape an array of size 50 into shape (5,5).
  
  - This raised questions about whether it was a bug or simply **not supported yet**, as the user had only recently started with tinygrad.
- **Getting it to work without numpy**: One member mentioned successfully executing a similar task without using numpy, though they admitted to using AI, which they noted is 'a bit frowned on.'
  
  - They emphasized that it is **definitely possible** to get it working in tinygrad nonetheless.

 

---

### **tinygrad (George Hotz) ▷ #**[**learn-tinygrad**](https://discord.com/channels/1068976834382925865/1070745817025106080/1298421329749479489) (29 messages🔥):

> - `Action Chunking Transformers training`
> - `TinyJIT challenges`
> - `JIT function input/output requirements`
> - `Running faster kernels`
> - `Reverse engineering byte code`

- **Frustration with Action Chunking Transformers Training Time**: Training for Action Chunking Transformers with **55 million** parameters takes **two days** without JIT, prompting discussions on performance improvements.
  
  - Thoughts on minimizing long inference times and issues with repeated **loss parameter** during JIT training were shared.
- **TinyJIT Loss Parameter Printing Confusion**: Users discussed the challenges of using `.item()` when printing the loss in JIT functions, examining its impacts on displaying values correctly.
  
  - Recommendations included avoiding non-Tensor returns due to potential undesired effects on JIT functionality.
- **Insights on JIT Function Inputs and Outputs**: Clarifications pointed out that input and output of jitted functions should ideally be realized tensors, while using `dict` structures is acceptable for organization.
  
  - The relationship between the JIT execution model and non-Tensor logic in function definitions was clarified as crucial for preserving executable paths.
- **Improving Training Time with BEAM Settings**: A recommendation was made to run with `BEAM=2` to enhance performance during lengthy training sessions, potentially leading to faster kernel searches.
  
  - Feedback noted that this approach had already been utilized to expedite the training process successfully.
- **Exploring Reverse Engineering AI Accelerator Byte Code**: One user expressed interest in reverse engineering an AI accelerator's byte code and sought guidance on methodologies and initial testing techniques.
  
  - This sparked curiosity among members about tools and frameworks conducive to starting the reverse engineering process.

**Links mentioned**:

- [Quickstart - tinygrad docs](https://docs.tinygrad.org/quickstart/?h=jit#jit): no description found
- [act-tinygrad/train.py at main · mdaiter/act-tinygrad](https://github.com/mdaiter/act-tinygrad/blob/main/train.py#L60): Action Chunking Transformers in Tinygrad. Contribute to mdaiter/act-tinygrad development by creating an account on GitHub.
- [tinygrad/tinygrad/engine/jit.py at master · tinygrad/tinygrad](https://github.com/tinygrad/tinygrad/blob/master/tinygrad/engine/jit.py#L174>): You like pytorch? You like micrograd? You love tinygrad! ❤️ - tinygrad/tinygrad

---

### **Interconnects (Nathan Lambert) ▷ #**[**news**](https://discord.com/channels/1179127597926469703/1179128538679488533/1298400643001880670) (14 messages🔥):

> - `Claude AI Experience`
> - `Continuous Pretraining Discussion`
> - `Character.AI User Tragedy`
> - `Safety Features Implementation`
> - `MIT Technology Review Podcast`

- **Fun with Claude AI**: A member shared their excitement about using **Claude AI**, mentioning it was very fun and plans to send examples the next day.
  
  - This creates anticipation for more detailed insights into their experience.
- **Continuous Pretraining Questions**: A member inquired about whether **GPT-4o** was pretrained from scratch with a **200k vocabulary tokenizer** or if it continued pretraining after swapping from a **100k** tokenizer.
  
  - Another member commented that mid-training is messy, reflecting ongoing challenges in making determinations about the training process.
- **Character.AI Condolences**: Character.AI expressed condolences over the tragic loss of a user and emphasized the addition of new safety features [here](https://blog.character.ai/community-safety-updates/).
  
  - A member linked to a [New York Times article](https://www.nytimes.com/2024/10/23/technology/characterai-lawsuit-teen-suicide.html) highlighting the grim context surrounding this statement.
- **Concerns About Safety**: A member mentioned skepticism about the effectiveness of safety measures, suggesting that tragic outcomes may continue despite company efforts.
  
  - This reflects broader concerns regarding the societal impacts of rapidly developed AI technologies.
- **MIT Technology Review Podcast**: A member raised concerns about potential bad outcomes stemming from a system discussed in an [MIT Technology Review podcast episode](https://podcasts.apple.com/us/podcast/mit-technology-review-narrated/id1523584878?i=1000674111360).
  
  - They expressed sadness about the trajectory that AI developments seem to be following, likening it to the quick and impactful path of social media.

**Links mentioned**:

- [Tweet from Character.AI (@character_ai)](https://x.com/character_ai/status/1849055407492497564): We are heartbroken by the tragic loss of one of our users and want to express our deepest condolences to the family. As a company, we take the safety of our users very seriously and we are continuing ...
- [Technology that lets us “speak” to our dead relatives has arrived. Are we ready?](https://podcasts.apple.com/us/podcast/mit-technology-review-narrated/id1523584878?i=1000674111360): Podcast Episode · MIT Technology Review Narrated · 10/23/2024 · 27m

---

### **Interconnects (Nathan Lambert) ▷ #**[**ml-drama**](https://discord.com/channels/1179127597926469703/1181746144821387334/) (1 messages):

xeophon.: [https://milesbrundage.substack.com/p/why-im-leaving-openai-and-what-im](https://milesbrundage.substack.com/p/why-im-leaving-openai-and-what-im)

---

### **Interconnects (Nathan Lambert) ▷ #**[**memes**](https://discord.com/channels/1179127597926469703/1187551504995987576/1298374595371335743) (5 messages):

> - `Jeremy Howard's tweet`
> - `Tek's energy levels`
> - `Hermes 3 performance`

- **Michael's Tweet Ratioed by Anime Account**: A notable tweet from [Jeremy Howard](https://x.com/jeremyphoward/status/1848813387242999847) highlights how the CEO of **Microsoft** got ratioed by a user with an *anime profile picture*.
  
- **Tek's Angry Man Arc Continues**: A sentiment was shared regarding **Tek** being in his *angry man arc* for several months and many are noticing his energy.
  
  - While one user doesn't love this energy, they acknowledge that it seems to resonate with a lot of people, saying, \*
- **Potential Hermes 3 Score Update**: There's speculation about possibly 'nuking' the **Hermes 3** scores in the next paper as mentioned by a member.
  
  - This has raised interest in the leaderboard's current standings.

 

**Link mentioned**: [Tweet from Jeremy Howard (@jeremyphoward)](https://x.com/jeremyphoward/status/1848813387242999847): CEO of Microsoft getting ratioed by an anime pfp account...

 

---

### **Interconnects (Nathan Lambert) ▷ #**[**posts**](https://discord.com/channels/1179127597926469703/1228051082631188530/1298640266693181520) (11 messages🔥):

> - `Anthropic's focus on B2B`
> - `Cost comparison between Anthropic and Simular`
> - `Automation and AI agents`
> - `Microsoft vs OpenAI in AI demonstrations`

- **Anthropic shifts towards B2B while OpenAI targets consumers**: A member noted that **Anthropic** is evolving into a B2B company, contrasting with **OpenAI**'s consumer-oriented focus, suggesting that automating tasks like shopping is less desirable.
  
  - This perspective sparked discussion about consumer interest in automating mundane tasks versus engaging in enjoyable activities.
- **Cost comparison with Simular's demo**: One member reflected on **Simular's** demo from last year at SPC, expressing curiosity about how its costs align with those of **Anthropic** showcased in a recent YouTube video [here](https://www.youtube.com/watch?v=ld17uwuNBcY&t=25s).
  
  - The recent demo comparisons hint at potential market shifts and raises questions about funding and investment strategies.
- **Anthropic's automation focuses on boring tasks**: Concerns were raised that the automation **Anthropic** is promoting largely revolves around tedious tasks like filling out forms, offering substantial time savings at work but lacking excitement.
  
  - Members criticized this approach, pointing out that consumers generally do not want to automate enjoyable experiences, such as gaming.
- **Microsoft's engaging AI demonstrations**: **Microsoft** showcased more playful applications of AI, such as automating gameplay in **Minecraft**, contrasting with the more monotonous tasks presented by Anthropic's demonstrations [view here](https://youtu.be/TLg2KWY2J5c?si=dRkIA2t1Y0F61KgP).
  
  - The distinction between Microsoft's focus on fun and Anthropic's emphasis on corporate efficiency highlights differing strategies within the AI landscape.
- **Members share thoughts on AI company directions**: The discussion touched upon the idea that some companies may seem aligned out of necessity rather than genuine enthusiasm for their market segment.
  
  - One member expressed a wish they had identified these trends sooner, indicating a broader interest in the strategic movements within AI companies.

**Links mentioned**:

- [Simular @ 2023 December SPC demo faire (Spotlight)](https://www.youtube.com/watch?v=ld17uwuNBcY&t=25s): no description found
- [Claude | Computer use for automating operations](https://youtu.be/ODaHJzOyVCQ?si=Lb1iOygMphHW9GJ5): With the upgraded Claude 3.5 Sonnet, we’re introducing a new capability in beta: computer use. Developers can now direct Claude to use computers the way peop...
- [Copilot gpt4o preview](https://youtu.be/TLg2KWY2J5c?si=dRkIA2t1Y0F61KgP): Copilot with gpt4o preview

---

### **OpenInterpreter ▷ #**[**general**](https://discord.com/channels/1146610656779440188/1147665339266650133/1298369919729270888) (29 messages🔥):

> - `Screenpipe tool`
> - `Claude 3.5 model`
> - `Open Interpreter's development`
> - `AI integration on different OS`
> - `Efficient data extraction for AI`

- **Screenpipe's Build Logs Impress**: Members praised the usefulness of [Screenpipe](https://github.com/OpenInterpreter/open-interpreter/blob/development/examples/screenpipe.ipynb) for managing build logs, highlighting its potential and interesting landing page.
  
  - One user noted the major impact of using this tool, especially for developers looking for efficient logging solutions.
- **Anthropic's Claude 3.5 Capabilities Unveiled**: Anthropic announced the **Claude 3.5 Sonnet** model, which features significant coding improvements and introduces a new **computer use** capability via public beta, allowing models to interact with user interfaces more like humans.
  
  - This new functionality comes with its challenges, as it requires constant screenshot capturing, raising concerns about efficiency and cost.
- **Open Interpreter Roadmap Discussions**: In response to criticisms, members discussed the roadmap for Open Interpreter, expressing confidence that users will find value in its unique capabilities compared to AI offerings integrated in mainstream operating systems.
  
  - Some skeptics questioned the feasibility of competing with established AI models, while others emphasized the importance of community-driven development.
- **Challenges in AI Screen Interaction**: Concerns were raised about the inefficiency of using screenshots for AI model input, with suggestions for extracting necessary data points directly from programs for better efficiency.
  
  - Members expressed the need for improved approaches to data processing that could complement the known limitations in screenshot dependency.
- **Testing the New Integration with Anthropic**: A member highlighted the introduction of the `interpreter --os` command for integrating with Anthropic's model, inviting others to assist in testing the new feature before its official release.
  
  - Testing revealed that increasing screen size and text could help reduce error rates when using the model.

**Links mentioned**:

- [Introducing computer use, a new Claude 3.5 Sonnet, and Claude 3.5 Haiku](https://www.anthropic.com/news/3-5-models-and-computer-use): A refreshed, more powerful Claude 3.5 Sonnet, Claude 3.5 Haiku, and a new experimental AI capability: computer use.
- [Claude Computer Use TESTED - This is VERY Promising!](https://www.youtube.com/watch?v=A5RfSftJRw8): Claude Computer Use TESTED - This is VERY Promising!👊 Become a YouTube Member for GH access:https://www.youtube.com/c/AllAboutAI/join🔥Swarm GH Repo:https:/...
- [open-interpreter/examples/screenpipe.ipynb at development · OpenInterpreter/open-interpreter](https://github.com/OpenInterpreter/open-interpreter/blob/development/examples/screenpipe.ipynb): A natural language interface for computers. Contribute to OpenInterpreter/open-interpreter development by creating an account on GitHub.
- [Anthropic’s New AI Can Control Your Computer!](https://youtu.be/idipaHSpQes?t=225.): Anthropic dropped three incredible things: Claude 3.5 Sonnet NEW, Claude 3.5 Haiku, and "computer use, " allowing models to control your computer.Join My New...

---

### **OpenInterpreter ▷ #**[**ai-content**](https://discord.com/channels/1146610656779440188/1149229778138824765/) (1 messages):

facelessman: [https://youtu.be/VgJ0Cge99I0](https://youtu.be/VgJ0Cge99I0) -- Love this episode -- love these folks!!!

---

### **Cohere ▷ #**[**discussions**](https://discord.com/channels/954421988141711382/954421988783444043/1298387382856450129) (8 messages🔥):

> - `Multimodal Command model`
> - `Global connection`
> - `Aya Expanse`
> - `Complex bots`

- **Speculation on Multimodal Command Model**: *Paulm24* inquired about the existence of a **multimodal Command model**, hinting at a growing interest in advanced model capabilities.
  
  - *Karthik_99_* chimed in, suggesting it has a **Global connection** feature, indicating a fusion of different modes of interaction.
- **Excitement Over Bot Complexity**: *Enzoloko* expressed enthusiasm about the potential to create **complex bots** simply through prompting, due to the model's inherent nature.
  
  - This reflects a broader excitement in the community about leveraging advanced models for innovative applications.
- **Aya Expanse Powers Cohere Bot**: *Sssandra* confirmed that the bot is powered by **Aya Expanse**, which has sparked curiosity about its capabilities.
  
  - This mention signals significant advancements and potential exploratory opportunities in the AI space.
- **Community Engagement on Fun Features**: *Wolfybl* and *sssandra* highlighted a fun and collaborative spirit while discussing their experiences with the bot.
  
  - *Roazzy* urged for sharing the fun, indicating that community interaction and experiences are valued.

 

---

### **Cohere ▷ #**[**questions**](https://discord.com/channels/954421988141711382/1168411509542637578/1298362022290919466) (7 messages):

> - `Cohere API trial and production keys`
> - `Using Command-Night in Portuguese`
> - `Feedback integration in fine-tuning LLM`
> - `Ollama Mistral performance issues`

- **Cohere offers trial API keys for free testing**: Cohere provides a [trial API key](https://docs.cohere.com/docs/rate-limits) that allows users to access all models for free, with certain rate limits.
  
  - For example, the Chat endpoint is limited to **20 calls per minute** during the trial period, while the production key allows for **500 calls per minute**.
- **Exploring Command-Night functionality in Portuguese**: A member questioned whether Command-Night works in Portuguese due to the inclusion of Aya instead of Light.
  
  - Insights on the multilingual capabilities of the tool were not provided in the discussion.
- **Innovative feedback loop for fine-tuning LLM**: A member proposed an iterative feedback mechanism allowing experts to enhance LLM performance by integrating corrections through a chat UI.
  
  - This method involves saving feedback to 'accept.json' and 'not_accept.json' files, facilitating smarter LLM refinement over time.
- **Challenges with Ollama Mistral's performance**: A member expressed frustration with Ollama Mistral's hallucination tendencies and computational demands, affecting the execution of their project.
  
  - Despite this, they highlighted the foundational principle of their approach for generating prompts and responses for expert evaluation available in their [GitHub gist](https://gist.github.com/pleabargain/8b3f1641ef727cc114ac389cbc1b354b).

**Links mentioned**:

- [API Keys and Rate Limits — Cohere](https://docs.cohere.com/docs/rate-limits): This page describes Cohere API rate limits for production and evaluation keys.
- [using Ollama to interate over a source of truth and present prompts and responses to an expert](https://gist.github.com/pleabargain/8b3f1641ef727cc114ac389cbc1b354b): using Ollama to interate over a source of truth and present prompts and responses to an expert - ollama prompt and response generator with feedback.py

---

### **Cohere ▷ #**[**api-discussions**](https://discord.com/channels/954421988141711382/1168578329423642786/1298419458842431508) (9 messages🔥):

> - `Cohere V2 API Errors`
> - `Finetuned Models Issues`
> - `Vercel AI SDK Integration`
> - `Tool Use and Function Calling`

- **Internal Server Errors in Cohere V2 API**: Members reported encountering **internal server errors** when using the chat endpoint in the Cohere V2 API, particularly with messages involving **tool_calls**.
  
  - One user shared a specific payload that resulted in failure, and another member requested a code snippet for better troubleshooting.
- **Tool Calls Missing Required Fields**: Discussion emerged regarding missing the **tool_plan field** in requests, which was pointed out as a potential issue by a member.
  
  - An example was shared from the [Cohere documentation](https://docs.cohere.com/docs/tool-use#step-2) to illustrate correct usage with tool integrations.
- **Vercel AI SDK Lacks Cohere V2 Support**: A user mentioned plans to integrate Cohere V2 using the **Vercel AI SDK**, but discovered current provider mapping only supports V1.
  
  - They highlighted raising this issue with the Vercel team, referencing their [GitHub issue](https://github.com/vercel/ai/issues/3331) but remain uncertain about the timeline for V2 support.
- **Concerns Over Finetuned Model Functionality**: A user inquired if others are facing issues with their **finetuned models** when accessed through the API.
  
  - This sparked conversations about the stability and functionality of finetuned models within the current setup.

**Links mentioned**:

- [Issues · vercel/ai](https://github.com/vercel/ai/issues/3331),): Build AI-powered applications with React, Svelte, Vue, and Solid - Issues · vercel/ai
- [Tool Use — Cohere](https://docs.cohere.com/docs/tool-use#step-2): Enable your large language models to connect with external tools for more advanced and dynamic interactions (V2).

---

### **Cohere ▷ #**[**projects**](https://discord.com/channels/954421988141711382/1218409701339828245/1298368461776617534) (1 messages):

> - `Agentic Builder Day`
> - `Cohere Models`
> - `AI Agent hackathon`

- **Agentic Builder Day on November 23rd**: An **Agentic Builder Day** is being hosted on November 23rd by OpenSesame in collaboration with the **Cohere** team, inviting talented builders to showcase their skills.
  
  - Participants can [apply now to compete](https://www.opensesame.dev/hack) in this **mini AI Agent hackathon** for a chance to win prizes.
- **Call for talented builders**: The event seeks **talented builders** who are interested in collaborating and competing to build powerful AI agents using **Cohere Models**.
  
  - This is a unique opportunity for developers to connect within the AI community and improve their skills while competing.

 

**Link mentioned**: [OpenSesame | Build Better AI Agents](https://www.opensesame.dev/hack): OpenSesame simplifies the entire AI agent lifecycle, from building to evaluating. Our platform empowers businesses to easily create, share, and implement AI agents and detect hallucinations, making AI...

 

---

### **Modular (Mojo 🔥) ▷ #**[**mojo**](https://discord.com/channels/1087530497313357884/1151418092052815884/1298496368326873181) (20 messages🔥):

> - `Community Meeting Discussions`
> - `Serial Communication in Mojo`
> - `C/C++ Support in Mojo`
> - `LED Matrix Communication`
> - `Framework Laptop 16`

- **Joining Sunday Discussions on stdlib**: A member expressed enthusiasm about joining the Sunday discussions regarding **stdlib contributor meetings** after watching the last community meeting.
  
  - *Welcome aboard*, others encouraged joining the conversation in the relevant channel.
- **Understanding Serial Communication in Mojo**: A user sought help understanding how to implement **serial communication** in **Mojo**, specifically over a port.
  
  - Others clarified that Mojo currently only provides what **libc** offers, with no additional support.
- **C/C++ Support Inquiry**: The conversation turned to whether there is **C/C++ support** in Mojo, leading to discussions about the potential use cases for this support.
  
  - While it could work for a few users, the suitability of the application was questioned.
- **Framework Laptop's LED Matrix Communication Library**: A user indicated plans to create a library for **Framework Laptop 16 LED Matrix communication**, expressing a desire for accessibility to more systems.
  
  - Despite initial skepticism, there was openness to collaboration and improvement on this project.

 

---

### **Modular (Mojo 🔥) ▷ #**[**max**](https://discord.com/channels/1087530497313357884/1212827597323509870/1298698487852171264) (3 messages):

> - `C API for MAX Engine`
> - `Mojo's advantages`
> - `Graph Builder API in C`

- **C API now live for MAX Engine!**: The **C API** is available for the **MAX Engine**, but there are currently no plans to include it for the **graph API**.
  
  - An update will be shared if anything changes regarding the graph API.
- **Mojo's Unique Graph Capabilities**: A member questioned if the main advantage of **Mojo** lies in its unique ability to leverage the **graph API** which no other language can utilize.
  
  - This highlights the inherent strengths and promises of Mojo in the current AI landscape.
- **Building Graph API with C**: Another member pointed out that while **Mojo** is used for the graph nodes, it's also possible to create a **graph builder API** using C if desired.
  
  - This opens discussions on alternative implementations and collaborations between different programming languages.

 

---

### **Torchtune ▷ #**[**general**](https://discord.com/channels/1216353675241590815/1216353675744641096/1298388964813574184) (6 messages):

> - `Debugging run command`
> - `Testing on multiple GPUs`
> - `Fine-tuning custom models`
> - `Environment comparison`

- **Don't use .yaml in TorchTune config**: A member pointed out that using a **.yaml** file extension in the run command for **TorchTune** config is problematic, as it implies a local config is being provided.
  
  - *Debugging can be frustrating without additional error messages*.
- **Testing script on 2 GPUs**: One user inquired about the ability to test on **2 GPUs**, posing a question regarding this capability.
  
  - Another member reported issues with receiving error messages while running scripts on both **1 GPU** and **2 GPUs** using **lora_finetune_distributed**.
- **Fine-tuning with TorchTune is possible**: In response to a question about fine-tuning a **custom Llama** model, a member confirmed that **TorchTune** is very customizable and provides assistance.
  
  - They encouraged further discussion on the custom components of the model for tailored support.
- **Friendly community vibes**: A user expressed appreciation for the friendliness of the community, highlighting a warm atmosphere.
  
  - Comments like this help foster a welcoming environment for sharing knowledge and support.

 

---

### **Torchtune ▷ #**[**dev**](https://discord.com/channels/1216353675241590815/1236040539409879170/1298644877793300520) (6 messages):

> - `Linters and Pre-commit Hooks`
> - `CI Issues`
> - `Tokenizer Tests`

- **Concerns with Linters and Pre-commit Hooks**: A member expressed issues with **linters and pre-commit hooks**, mentioning they weren't working **100%** as expected.
  
  - Specifically, they noted that to ignore a line, both `# noqa` and `# fmt: on ... #fmt: off` are needed, which seems **unusual**.
- **Strange CI Behavior in PR #1868**: Another member reported strange behavior with the **CI** for PR **#1868** and requested assistance to check what happened.
  
  - They indicated that the CI issue was persistent in every PR, suggesting ongoing investigation.
- **Status Update on CI Fix**: A member inquired whether a recent CI issue had been resolved, which was under review by another member.
  
  - The response indicated that the problem should be **fixed** now, reassuring the group.

 

---

### **LangChain AI ▷ #**[**general**](https://discord.com/channels/1038097195422978059/1038097196224086148/1298520313885229127) (3 messages):

> - `Developer Project Survey`
> - `Location in Manila`
> - `FunctionMessages and LLM Responses`

- **Help shape a tool for developers**: A member shared a [survey link](https://forms.gle/Roi1U5ynVwLtQ3S46) targeting developers to understand challenges in turning ideas into reality, noting it takes about **5-7 minutes** to complete.
  
  - The survey covers how often developers generate ideas, obstacles faced, and interest in solutions to streamline project realization.
- **Inquiry about Manila developers**: A member asked if anyone in the group is located in **Manila**, possibly to connect with local developers.
  
  - This inquiry suggests interest in building community or collaboration among Manila-based developers.
- **Detailed answers from joiner LLM**: A member inquired about obtaining **detailed answers** from a joiner LLM that aggregates responses without summarizing or reducing length.
  
  - They expressed concern that the current implementation with `FunctionMessages` results in overly concise summaries instead of preserving the original response details.

 

**Link mentioned**: [From Lightbulb to Launch: Developer Challenges in Project Realization](https://forms.gle/Roi1U5ynVwLtQ3S46) : Hello fellow developers! Are you full of innovative ideas but struggle to bring them to life? You're not alone. We're conducting research to understand the challenges developers face when turn...

 

---

### **LangChain AI ▷ #**[**share-your-work**](https://discord.com/channels/1038097195422978059/1038097372695236729/1298570202757337108) (3 messages):

> - `AI Coding Assistant Study`
> - `AI-Powered Funding Tool`
> - `ApeBrains Trader Specialization`

- **Participate in AI Impact Study**: A call for developers to participate in a Master’s study investigating the effects of AI tools on software engineering is underway. By filling out a short questionnaire, participants stand a chance to win a $200NZD gift card while contributing to valuable research.
  
  - You can access the questionnaire [here](https://auckland.au1.qualtrics.com/jfe/form/SV_0uf2q5Ie7V3gpvM?Source=43).
- **Unlock Funding with AI Tool**: An AI-powered tool has been launched to help users secure funding for their ideas by connecting them with relevant investors and accelerators. The first **200** people to join the waitlist will receive a free **Startup Accelerator pack** that enhances their search capabilities significantly.
  
  - With only **62** spots left, interested users are encouraged to [Sign Up Now](https://www.aloangels.me/) to make their startup visions a reality.
- **Join the ApeBrains Trader Alpha Program**: ApeBrains is promoting a specialization program for traders, offering users the chance to sign up for **ApeBrains Alpha**. In addition, there is a referral program allowing participants to earn priority access by sharing their links with friends.
  
  - Users are encouraged to visit [ApeBrains](https://www.apebrains.com) for more details and to take advantage of promotional offers.

**Links mentioned**:

- [ApeBrains Wallet Agents - Coming Soon](https://www.apebrains.com): Specialize your ApeBrains Trader. Sign Up for ApeBrains Alpha.
- [Online Survey Software | Qualtrics Survey Solutions](https://auckland.au1.qualtrics.com/jfe/form/SV_0uf2q5Ie7V3gpvM?Source=43): The most powerful, simple and trusted way to gather experience data. Start your journey to experience management and try a free account today.
- [AloAngels: Free AI Powered Investor Matching](https://www.aloangels.me/): no description found

---

### **LangChain AI ▷ #**[**tutorials**](https://discord.com/channels/1038097195422978059/1077843317657706538/1298410963791515753) (1 messages):

> - `GeoGuessr AI Bot`
> - `Vision LLMs`
> - `LangChain`
> - `Multimodal AI`

- **Building an AI GeoGuessr Player**: A new [YouTube tutorial](https://www.youtube.com/watch?v=OyDfr0xIhss) demonstrates how to code an AI bot that plays **GeoGuessr** autonomously using **Multimodal Vision LLMs** like **GPT-4o**, **Claude 3.5**, and **Gemini 1.5**.
  
  - The tutorial covers coding with **Python** and integrates **LangChain** to enable the bot to take screenshots and interact with the game environment.
- **Multimodal Vision LLMs in Action**: Participants discussed the combination of **Vision LLMs** in the context of coding projects, particularly highlighting their effectiveness in dynamic environments like **GeoGuessr**.
  
  - This underscores the increasing relevance of multimodal capabilities in AI applications, particularly in gaming.

 

**Link mentioned**: [Coding a Vision LLM Agent that plays GeoGuessr by itself (GPT-4o, Claude 3.5 and Gemini 1.5)](https://www.youtube.com/watch?v=OyDfr0xIhss): How to code an AI bot that plays autonomously the GeoGuessr game using Multimodal Vision LLMs that take screenshots of the game with Python + LangGhain and c...

 

---

### **DSPy ▷ #**[**show-and-tell**](https://discord.com/channels/1161519468141355160/1202371242519441499/1298382426329976986) (2 messages):

> - `Advanced Workflow System`
> - `Upgrade Process`

- **World's Most Advanced Workflow System in Progress**: A member announced they are starting to work on the **world's most advanced workflow system** in a dedicated channel.
  
  - They highlighted plans to create a **live demonstration** on Monday to detail how the current system works and discuss their upgrade process.
- **Upcoming Live Demo Announcement**: The member confirmed a **live demonstration** is set for Monday, aimed at explaining the workflow system's operation and the forthcoming upgrades.
  
  - This session is expected to provide insights into the **upgrade process** and improvements currently underway.

 

---

### **DSPy ▷ #**[**general**](https://discord.com/channels/1161519468141355160/1161519469319946286/1298397373734584405) (5 messages):

> - `DSPy Funding Potential`
> - `Synthetic Data Generation Metrics`

- **DSPy aims for ambitious funding goals**: A member suggested that if CrewAI can get **$18M**, then **DSPy** should aim for at least **$50M**, expressing enthusiasm to join as employee number 5 or 10.
  
  - *What are we waiting for?* was the rallying sentiment for immediate action.
- **Discussion on Metrics for Synthetic Data**: One member inquired about using **DSPy** to create synthetic data for QA based on a chunk of text, specifically asking about effective metrics.
  
  - Another member responded that for open-ended generation without ground truth, using an **LLM as a judge** along with predefined criteria could be effective.
- **Groundedness in Synthetic Data Generation**: In the context of generating synthetic data, a member highlighted that **ground truth** would come from the text used for generation, suggesting groundedness as a possible metric.
  
  - They expressed appreciation for the insights shared on the topic, indicating ongoing collaboration among members.

 

---

### **LLM Agents (Berkeley MOOC) ▷ #**[**mooc-questions**](https://discord.com/channels/1280234300012494859/1280370030609170494/1298465984876777512) (7 messages):

> - `LLM Agents MOOC Signup`
> - `Hackathon Project Open Sourcing`
> - `Agents Development Tutorials`

- **Signup Form Confusion**: A member expressed concern about not receiving weekly emails after submitting the [LLM Agents MOOC Signup Form](https://link-to-signup-form), prompting a follow-up on whether they received a confirmation.
  
  - Another member shared a similar experience, stating they also did not get formal feedback on their acceptance into the course.
- **Hackathon Code Submission Requirement**: During the final presentation for the Hackathon, members confirmed that they are required to make their project codes **100% open source**.
  
  - One member emphasized the importance of code submission to comply with the event's rules.
- **Need for Tutorials on Agent Creation**: A participant inquired about the existence of a tutorial for making agents from scratch without using any platforms beyond the LLM itself.
  
  - This indicates a demand for accessible resources on independent agent development among users.

 

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**general**](https://discord.com/channels/1238365980128706560/1238365980128706563/1298604483328348272) (2 messages):

> - `Axolotl configurations`
> - `LangSmith Prompt Hub`
> - `Kaggle Solutions`
> - `Hugging Face datasets`
> - `Discord bot for message scraping`

- **Utilize Axolotl Discord for Configurations**: You can leverage the 🦎 Axolotl Discord channel for sharing and finding configurations tailored to your use case, along with the example folder on GitHub.
  
  - Check out the [Discussions tab](https://github.com/axolotl-ai-cloud/axolotl/discussions) for similar use cases shared by other members.
- **Explore LangSmith Prompt Hub for Prompts**: The 🛠️ LangSmith Prompt Hub offers a collection of varied prompts suitable for different models and use cases, enriching your prompt engineering toolkit.
  
  - For datasets, explore publicly available datasets in the [Awesome Public Datasets repository](https://github.com/awesomedata/awesome-public-datasets).
- **Comprehensive Kaggle Solutions Available**: There’s a collection titled *The Most Comprehensive List of Kaggle Solutions and Ideas* which can serve as a useful resource for competitive data science.
  
  - Find it on GitHub [here](https://github.com/faridrashidi/kaggle-solutions) for an extensive variety of solutions.
- **Hugging Face Recipes for Model Alignment**: For continued pretraining and aligning language models with both human and AI preferences, refer to the robust recipes shared on Hugging Face.
  
  - Access these recipes [here](https://github.com/huggingface/alignment-handbook/tree/main/recipes).
- **New Discord Bot for Message Scraping**: A user has created a Discord bot to scrape messages from the channel and is seeking help with inviting the bot.
  
  - You can invite the bot via this [link](https://discord.com/oauth2/authorize?client_id=1298625427375656980&response_type=code&redirect_uri=https%3A%2F%2Fc123ian.github.io%2F&scope=messages.read).

**Links mentioned**:

- [LangSmith](https://smith.langchain.com/hub): no description found
- [GitHub - awesomedata/awesome-public-datasets: A topic-centric list of HQ open datasets.](https://github.com/awesomedata/awesome-public-datasets): A topic-centric list of HQ open datasets. Contribute to awesomedata/awesome-public-datasets development by creating an account on GitHub.
- [GitHub - faridrashidi/kaggle-solutions: 🏅 Collection of Kaggle Solutions and Ideas 🏅](https://github.com/faridrashidi/kaggle-solutions): 🏅 Collection of Kaggle Solutions and Ideas 🏅. Contribute to faridrashidi/kaggle-solutions development by creating an account on GitHub.
- [alignment-handbook/recipes at main · huggingface/alignment-handbook](https://github.com/huggingface/alignment-handbook/tree/main/recipes): Robust recipes to align language models with human and AI preferences - huggingface/alignment-handbook

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**general**](https://discord.com/channels/1104757954588196865/1104757955204743201/1298391199937335306) (2 messages):

> - `Experimental Triton FA support`
> - `Mixtral vs. Llama 3.2`

- **2.5.0 adds Experimental Triton FA support for gfx1100**: With version **2.5.0**, **experimental Triton Flash Attention (FA)** support for **gfx1100** can be enabled using the environment variable `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`.
  
  - *UserWarning: Flash attention support on Navi31 GPU is still experimental.* Further details can be found in the [GitHub issue](https://github.com/ROCm/aotriton/issues/16#issuecomment-2346675491).
- **Debate on Mixtral vs. Llama 3.2 usage**: A question was raised about the viability of using **Mixtral** now, given the advancements of **Llama 3.2**.
  
  - The community is weighing the benefits and deficiencies of both options to establish which model to prioritize.

 

**Link mentioned**: [[Feature]: Memory Efficient Flash Attention for gfx1100 (7900xtx) · Issue #16 · ROCm/aotriton](https://github.com/ROCm/aotriton/issues/16#issuecomment-2346675491>): Suggestion Description Started using torchlearn to train models in pytorch using my gfx1100 card but get a warning that 1toch was not compiled with memory efficient flash attention. I see there is ...

 

---

### **Gorilla LLM (Berkeley Function Calling) ▷ #**[**discussion**](https://discord.com/channels/1111172801899012102/1111353033352294440/1298598678235451522) (2 messages):

> - `Troubleshooting model evaluation`
> - `Handler registration`
> - `Model generation command`

- **Empty Score Report on Model Evaluation**: A user reported that after adding a new model handler and registering it in `handler_map.py`, running `bfcl evaluate --model mynewmodel --test-category ast` produces an empty score report with progress at **0/0**.
  
  - Another member suggested confirming if the `bfcl generate ...` command was executed beforehand, hinting that it may be necessary for proper evaluation.
- **Importance of Generating Models Before Evaluation**: A discussion arose about the necessity of running the `bfcl generate` command before model evaluation to ensure accurate results.
  
  - This indicates that the lack of model generation could lead to issues like empty score reports during the evaluation process.

 

---

---

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