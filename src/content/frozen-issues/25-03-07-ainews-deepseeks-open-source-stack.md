---
id: 7574e355-24fe-47a4-9eca-3e87e33569af
title: DeepSeek's Open Source Stack
date: '2025-03-08T05:06:31.351088Z'
original_slug: ainews-deepseeks-open-source-stack
description: >-
  **DeepSeek's Open Source Week** was summarized by PySpur, highlighting
  multiple interesting releases. The **Qwen QwQ-32B model** was fine-tuned into
  **START**, excelling in PhD-level science QA and math benchmarks.
  **Character-3**, an omnimodal AI video generation model by Hedra Labs and
  Together AI, enables realistic animated content creation. **Google DeepMind**
  introduced the **Gemini embedding model** with an 8k context window, ranking
  #1 on MMTEB, alongside the **Gemini 2.0 Code Executor** supporting Python
  libraries and auto-fix features. **Inception Labs' Mercury Coder** is a
  diffusion-based code generation model offering faster token processing.
  **OpenAI** released **GPT-4.5**, their largest model yet but with less
  reasoning ability than some competitors. **AI21 Labs** launched **Jamba Mini
  1.6**, noted for superior output speed compared to Gemini 2.0 Flash, GPT-4o
  mini, and Mistral Small 3. A new dataset of 1.9M scanned pages was released
  for OCR benchmarking, with **Mistral OCR** showing competitive but not
  top-tier document parsing performance compared to LLM/LVM-powered methods.
  *"Cracked engineers are all you need."*
companies:
  - deepseek
  - pyspur
  - hugging-face
  - togethercompute
  - hedra-labs
  - google-deepmind
  - deeplearningai
  - openai
  - ai21-labs
  - mistral-ai
models:
  - qwen-qwq-32b
  - start
  - character-3
  - gemini
  - gemini-2.0
  - mercury-coder
  - gpt-4.5
  - jamba-mini-1.6
  - gemini-2.0-flash
  - gpt-4o-mini
  - mistral-small-3
  - mistral-ocr
topics:
  - fine-tuning
  - benchmarking
  - multimodality
  - code-generation
  - diffusion-models
  - model-performance
  - model-optimization
  - ocr
  - embedding-models
  - context-windows
  - runtime-limits
people:
  - _akhaliq
  - lmarena_ai
  - reach_vb
  - danielhanchen
  - _philschmid
  - aidan_mclau
  - vikhyatk
  - jerryjliu0
---


<!-- buttondown-editor-mode: plaintext -->**Cracked engineers are all you need.**

> AI News for 3/7/2025-3/8/2025. We checked 7 subreddits, [**433** Twitters](https://twitter.com/i/lists/1585430245762441216) and **28** Discords (**224** channels, and **4696** messages) for you. Estimated reading time saved (at 200wpm): **406 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

We didn't quite know how to cover DeepSeek's "Open Source Week" from [2 weeks ago](https://x.com/deepseek_ai/status/1892786555494019098), since each release was individually interesting but not quite hitting the bar of generally useful and we try to cover "the top news of the day". But [the kind folks at PySpur have done us the favor of collating all the releases and summarizing them](https://www.pyspur.dev/blog/deepseek_open_source_week):

![image.png](https://assets.buttondown.email/images/2c0b8f30-2092-407a-8154-39fbc95de10a.png?w=960&fit=max)

It even comes with little flash quizzes to test your understanding and retention!!

![image.png](https://assets.buttondown.email/images/0311559b-0b1d-444f-b464-b612245ee17b.png?w=960&fit=max)

We think collectively this is worth some internalization.


---


{% if medium == 'web' %}


**Table of Contents**

[TOC] 

{% else %}

The **Table of Contents** and **Channel Summaries** have been moved to the web version of this email: [{{ email.subject }}]({{ email_url }})!

{% endif %}


---

# AI Twitter Recap

**Models & Releases**

- **Qwen QwQ-32B Model**: [@_akhaliq](https://twitter.com/_akhaliq/status/1897854193152438553) announced the release of **START**, a Self-taught Reasoner with Tools, fine-tuned from the **Qwen-32B model**. START achieves high accuracy on PhD-level science QA (GPQA), competition-level math benchmarks, and the LiveCodeBench benchmark. [@lmarena_ai](https://twitter.com/lmarena_ai/status/1897763753417900533) welcomed **Qwen QwQ-32B** to the Arena for chats, noting it's also trending on Hugging Face, according to [@reach_vb](https://twitter.com/reach_vb/status/1897974348503208081).  [@danielhanchen](https://twitter.com/danielhanchen/status/1898035752124166368) provided a guide for debugging looping issues with **QwQ-32B**, suggesting sampler adjustments and noting quantization sensitivity. They also uploaded [dynamic 4bit quants and GGUFs](https://twitter.com/danielhanchen/status/1898035752124166368).
- **Character-3 AI Video Model**: [@togethercompute](https://twitter.com/togethercompute/status/1897756209069138116) and [@realDanFu](https://twitter.com/realDanFu/status/1897757302243156440) highlighted the launch of **Character-3**, an omnimodal AI model for video generation developed by @hedra_labs and scaled on Together AI.  Character-3 can turn images into animated content with realistic movement and gestures, as showcased by [@TomLikesRobots](https://twitter.com/TomLikesRobots/status/1898009257598980587) who successfully used it for AI lipsync and storytelling.
- **Gemini Embeddings and Code Executor**: [@_philschmid](https://twitter.com/_philschmid/status/1898075321460818153) reported on **Google DeepMind's** new experimental **Gemini embedding model**, which ranks #1 on the MMTEB leaderboard with an 8k context window and is designed for finance, science, legal, search, and code applications.  They also detailed how the **Gemini 2.0 Code Executor** works, including its auto-fix attempts, file input support, runtime limits, and supported Python libraries, as seen in tweets like [@_philschmid](https://twitter.com/_philschmid/status/1897910462043373902) and [@_philschmid](https://twitter.com/_philschmid/status/1897910464631202036).
- **Mercury Coder**: [@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1897874853555843277) introduced **Inception Labs' Mercury Coder**, a diffusion-based model for code generation that processes tokens simultaneously, achieving faster speeds than autoregressive models.
- **GPT-4.5 Release**: [@aidan_mclau](https://twitter.com/aidan_mclau/status/1897789747394920896) mentioned an interesting essay on **GPT-4.5**. [@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1898086241859440934) noted that **OpenAI released GPT-4.5**, describing it as their largest model but lacking the reasoning capabilities of models like o1 and o3.
- **Jamba Mini 1.6**: [@AI21Labs](https://twitter.com/AI21Labs/status/1897979834006946057) highlighted **Jamba Mini 1.6** outpacing **Gemini 2.0 Flash**, **GPT-4o mini**, and **Mistral Small 3** in output speed.
- **Mistral OCR**:  [@vikhyatk](https://twitter.com/vikhyatk/status/1897951196079353970) announced a new dataset release of **1.9M scanned pages transcribed using Pixtral**, potentially related to OCR benchmarking. [@jerryjliu0](https://twitter.com/jerryjliu0/status/1898037050185859395) shared benchmarks of **Mistral OCR** against other models, finding it decent and fast but not the best document parser, especially when compared to **LLM/LVM-powered parsing** techniques like **Gemini 2.0**, **GPT-4o**, and **Anthropic's Sonnet models**. [@sophiamyang](https://twitter.com/sophiamyang/status/1898065981957603754) and [@sophiamyang](https://twitter.com/sophiamyang/status/1898059704351277297) also shared videos by [@Sam_Witteveen](https://twitter.com/Sam_Witteveen) on **MistralAI OCR**.

**Tools & Applications**

- **MCP (Model Context Protocol)**: [@mickeyxfriedman](https://twitter.com/mickeyxfriedman/status/1897756536258412644) observed the sudden surge in discussion around **MCP**, despite it being around for half a year. [@hwchase17](https://twitter.com/hwchase17/status/1897757113885376808) highlighted a "coolest client" integrating with **MCP**. [@saranormous](https://twitter.com/saranormous/status/1898028053949038635) described a smart client for **MCP** everywhere. [@nearcyan](https://twitter.com/nearcyan/status/1897791454808027289) asked about the "underground **MCP** market" and announced an **SF MCP meetup** [@nearcyan](https://twitter.com/nearcyan/status/1897779866134868273).  [@abacaj](https://twitter.com/abacaj/status/1897769259003965636) noted the mystery and trending nature of **MCP**.  [@omarsar0](https://twitter.com/omarsar0/status/1898082332474593499) offered the original **MCP guide** as the best resource for understanding it, countering "bad takes". [@_akhaliq](https://twitter.com/_akhaliq/status/1898018002207228217) released an **MCP Gradio client** proof-of-concept.
- **Perplexity AI Search & Contest**: [@yusuf_i_mehdi](https://twitter.com/yusuf_i_mehdi/status/1897783236354515420) announced **Think Deeper in Copilot** is now powered by **o3-mini-high**. [@AravSrinivas](https://twitter.com/AravSrinivas/status/1898096492914843684) encouraged users to ask **Perplexity** anything to fight ignorance and announced a contest giving away **₹1 crore and a trip to SF** for questions asked on **Perplexity** during the ICC Champions Trophy final [@perplexity_ai](https://twitter.com/perplexity_ai/status/1897871471974007023).  [@AravSrinivas](https://twitter.com/AravSrinivas/status/1898070189465649597) sought feedback on **Sonnet 3.7-powered reasoning searches** in Perplexity.
- **Hedra Studio & Character-3 Integration**: [@togethercompute](https://twitter.com/togethercompute/status/1897756209069138116) and [@realDanFu](https://twitter.com/realDanFu/status/1897757302243156440) emphasized the integration of **Hedra's Character-3** into **Hedra Studio** for AI-powered content creation. [@TomLikesRobots](https://twitter.com/TomLikesRobots/status/1898009257598980587) highlighted its ease of use for lipsync and storytelling.
- **AI-Gradio**: [@_akhaliq](https://twitter.com/_akhaliq/status/1897807317435035821) announced **no-code Gradio AI apps**.  Followed by [@_akhaliq](https://twitter.com/_akhaliq/status/1897775110033227860) providing code to get started with **ai-gradio** for **Hugging Face models** and [@_akhaliq](https://twitter.com/_akhaliq/status/1897774596515938394) showcasing a **Vibe coding app with Qwen QwQ-32B** in few lines of code using **ai-gradio**.
- **Orion Browser for Linux**: [@vladquant](https://twitter.com/vladquant/status/1897797849091653778) announced the start of **Orion Browser for Linux** development, expanding Kagi's ecosystem.
- **ChatGPT for MacOS Code Editing**: [@kevinweil](https://twitter.com/kevinweil/status/1897777150905794992) highlighted that **ChatGPT for MacOS can now edit code directly in IDEs**.
- **Model Switch with Together AI**: [@togethercompute](https://twitter.com/togethercompute/status/1898061583554621898) announced a partnership with **Numbers Station AI** to launch **Model Switch**, powered by models hosted on Together AI, for data teams to choose efficient open-source models for AI-driven analytics.
- **Cursor AI on Hugging Face**: [@ClementDelangue](https://twitter.com/ClementDelangue/status/1898083264960610529) noted **Cursor AI** is now on Hugging Face, inquiring about fun integrations.

**Research & Datasets**

- **START (Self-taught Reasoner with Tools)**: [@_akhaliq](https://twitter.com/_akhaliq/status/1897854193152438553) announced **Alibaba's START**, a model fine-tuned from **QwQ-32B**, achieving strong performance in reasoning and tool use benchmarks.
- **PokéChamp**: [@_akhaliq](https://twitter.com/_akhaliq/status/1897873813083238713) presented **PokéChamp**, an expert-level Minimax Language Agent for Pokémon battles, outperforming existing LLM and rule-based bots, achieving top 10% player ranking using an open-source Llama 3.1 8B model.
- **Token-Efficient Long Video Understanding**: [@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1897854511814770733) shared **Nvidia's** research on **Token-Efficient Long Video Understanding for Multimodal LLMs**, achieving SotA results while reducing computation and latency.
- **DCLM-Edu Dataset**: [@LoubnaBenAllal1](https://twitter.com/LoubnaBenAllal1/status/1898044807928295808) announced **DCLM-Edu**, a new dataset filtered from DCLM using FineWeb-Edu's classifier, optimized for smaller models.
- **uCO3D Dataset**: [@AIatMeta](https://twitter.com/AIatMeta/status/1898065127859273871) and [@AIatMeta](https://twitter.com/AIatMeta/status/1898065129968918794) introduced **uCO3D**, Meta's new large-scale, publicly available object-centric dataset for 3D deep learning and generative AI.
- **LADDER Framework**: [@dair_ai](https://twitter.com/dair_ai/status/1898037429434826795) detailed **LADDER**, a framework enabling LLMs to recursively generate and solve simpler problem variants, boosting math integration accuracy through autonomous difficulty-driven learning and Test-Time Reinforcement Learning (TTRL).
- **Dedicated Feedback and Edit Models**: [@_akhaliq](https://twitter.com/_akhaliq/status/1897847351366029704) highlighted research on using **Dedicated Feedback and Edit Models** to empower inference-time scaling for open-ended general-domain tasks.
- **Masked Diffusion Model for Reversal Curse**: [@cloneofsimo](https://twitter.com/cloneofsimo/status/1897757122894741753) announced a **Masked Diffusion model** that beats the reversal curse, noting it will change things, as per [@cloneofsimo](https://twitter.com/cloneofsimo/status/1897757653436432494).
- **Brain-Like LLMs Study**: [@TheTuringPost](https://twitter.com/TheTuringPost/status/1897760975706898865) summarized a study exploring how **LLMs align with brain responses**, finding alignment weakens as models gain reasoning and knowledge, and bigger models are not necessarily more brain-like.

**Industry & Business**

- **Together AI Events & Partnerships**: [@togethercompute](https://twitter.com/togethercompute/status/1898094750047031651) promoted their **AI Pioneers Happy Hour** at Nvidia GTC, co-hosted with @SemiAnalysis_ and @HypertecGroup. [@togethercompute](https://twitter.com/togethercompute/status/1897754366712430666) announced co-founder @percyliang speaking at **#HumanX** with Sama CEO Wendy Gonzalez on foundational models meeting individual needs.
- **Anthropic's White House RFI Response**: [@AnthropicAI](https://twitter.com/AnthropicAI/status/1897773701224906854) shared their **recommendations submitted to the White House** in response to the Request for Information on an AI Action Plan.
- **Kagi User-Centric Products**: [@vladquant](https://twitter.com/vladquant/status/1897797849091653778) emphasized Kagi's ecosystem of **user-centric products** with the Orion Browser for Linux announcement.
- **AI Application Development in 2025**: [@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1897769165307478457) promoted a panel discussion at **AI Dev 25** on the future of AI application development, featuring @RomanChernin of @nebiusai.
- **Japan as an AI Hub**: [@SakanaAILabs](https://twitter.com/SakanaAILabs/status/1898027648921878821) shared an interview with Sakana AI co-founder Ren Ito discussing **Tokyo's potential as a global AI development hub** and Japan's ambition to become a technology superpower again.
- **AI in Media**: [@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1898000296514928820) argued **AI is not just the future of media, but the foundation beneath all future media**, transforming creation, distribution, and experience.
- **Open Source Business Model**: [@ClementDelangue](https://twitter.com/ClementDelangue/status/1897767808076816575) questioned the notion that **open-sourcing is bad for business**, highlighting positive examples.
- **United Airlines AI Integration**: [@AravSrinivas](https://twitter.com/AravSrinivas/status/1898044787221315796) expressed excitement for flying **United by default** once their AI rollout is complete.

**Opinions & Discussions**

- **MCP Hype & Understanding**:  [@mickeyxfriedman](https://twitter.com/mickeyxfriedman/status/1897756536258412644) questioned the sudden **MCP hype**. [@abacaj](https://twitter.com/abacaj/status/1897769259003965636) and [@Teknium1](https://twitter.com/Teknium1/status/1897928024210718889) also commented on the **MCP mania**. [@omarsar0](https://twitter.com/omarsar0/status/1898082332474593499) criticized "bad takes" on **MCP**.  [@clefourrier](https://twitter.com/clefourrier/status/1897953608013906124) asked for an **ELI5 explanation of MCP hype**.
- **Agentic AI & Reflection**: [@TheTuringPost](https://twitter.com/TheTuringPost/status/1897814476113756410) discussed the importance of **reflection in agentic AI**, enabling analysis, self-correction, and improvement, highlighting frameworks like Reflexion and ReAct. [@pirroh](https://twitter.com/pirroh/status/1897761908062929177) emphasized to **think beyond ReAct** in agent design.
- **Voice-Based AI Challenges & Innovations**: [@AndrewYNg](https://twitter.com/AndrewYNg/status/1897776017873465635) discussed challenges in **Voice Activity Detection (VAD)** for voice-based systems, especially in noisy environments, and highlighted **Kyutai Labs' Moshi** model as an innovation using persistent bi-directional audio streams to eliminate the need for explicit VAD.
- **Scale is All You Need**: [@vikhyatk](https://twitter.com/vikhyatk/status/1897858708509802521) jokingly stated the easy fix to AI problems is "more data, more parameters, more flops," suggesting **scale is all you need**.
- **Limitations of Supervised Learning**: [@finbarrtimbers](https://twitter.com/finbarrtimbers/status/1897779253665841370) argued for the need for **Reinforcement Learning (RL)**, stating supervised learning is fundamentally the wrong paradigm, though it's a necessary step.
- **Long-Context Challenges**: [@teortaxesTex](https://twitter.com/teortaxesTex/status/1898028373542150325) mentioned "**residual stream saturation**" as an under-discussed problem in long-context discussions.
- **Importance of Intermediate Tasks**: [@cloneofsimo](https://twitter.com/cloneofsimo/status/1897803367403208893) discussed the capability of "**coming up with seemingly unrelated intermediate tasks to solve a problem**" as a key aspect of intelligence.
- **Interview Questioning Techniques**: [@vikhyatk](https://twitter.com/vikhyatk/status/1897961277567517034) shared an Amazon interview technique called "**peeling the onion**" to assess candidates' ability to deliver results by asking follow-up questions to verify their project experience.
- **Focusing Attention in AI Discourse**: [@jeremyphoward](https://twitter.com/jeremyphoward/status/1897931934459990524) lamented the tendency in AI discourse to jump from topic to topic without sustained focus.
- **AGI Timeline & Planning**: [@abacaj](https://twitter.com/abacaj/status/1897771754845573333) sarcastically commented on **OAI planning for AGI by 2027** despite agile teams struggling to plan two sprints ahead.
- **"Open" AI and Marketing**: [@vikhyatk](https://twitter.com/vikhyatk/status/1897964271428080013) criticized companies for **misrepresenting their actions in marketing materials**, and [@vikhyatk](https://twitter.com/vikhyatk/status/1897963983363359193) questioned whether the latest release of an "actually open" AI was truly open.
- **Evaluation Chart Interpretation**: [@cognitivecompai](https://twitter.com/cognitivecompai/status/1898057327510487160) cautioned against taking **evaluation charts at face value**, especially those comparing favorably to R1, suggesting critical thinking about which models and evals are omitted.
- **Geopolitics of Superintelligence**: [@jachiam0](https://twitter.com/jachiam0/status/1897897896143733147) and [@jachiam0](https://twitter.com/jachiam0/status/1897897894579216474) discussed the **lack of a known-good approach to the geopolitics of superintelligence** and the serious risks of getting it wrong.
- **AI and Job Displacement**: [@fabianstelzer](https://twitter.com/fabianstelzer/status/1897882991184994444) suggested that while "bullshit jobs" are mostly immune to AI, we might see the "**notarization**" of other jobs through regulations requiring human-in-the-loop to protect them from AI disruption.

**Humor & Memes**

- **Confucius Quote on Asking Questions**: [@AravSrinivas](https://twitter.com/AravSrinivas/status/1898096492914843684) quoted Confucius: **“The man who asks a stupid question is a fool for a minute. The man who does not ask a stupid question is a fool for life.”** to promote Perplexity.
- **AI PhD in 2025 Meme**: [@jxmnop](https://twitter.com/jxmnop/status/1898025314166407596) shared a humorous "**day in the life of an AI PhD in 2025**" meme involving minimal research work and lots of tennis.
- **MCP as a "Context Palace"**: [@nearcyan](https://twitter.com/nearcyan/status/1897782467383435283) humorously described MCP as "**Rather than a context window, MCP is instead a context palace 👑**".
- **"tinygrad users don't know the value of nothing"**: [@typedfemale](https://twitter.com/typedfemale/status/1897829320791556121) posted a meme "**tinygrad users don't know the value of nothing and the cost of nothing**".
- **"why the fuck does every optimization researcher on X have a cat/dog in their profile picture?"**: [@eliebakouch](https://twitter.com/eliebakouch/status/1898025374514077774) jokingly asked "**why the fuck does every optimization researcher on X have a cat/dog in their profile picture?**".
- **"i thought i needed to block politics but now i need to block MCP as well lel"**: [@Teknium1](https://twitter.com/Teknium1/status/1897928024210718889) joked about needing to block MCP-related content like politics.

---

# AI Reddit Recap

## /r/LocalLlama Recap

**Theme 1. FT: Llama 4 w/ Voice Expected Soon, Enhancing Voice AI**

- **[FT: Llama 4 w/ voice expected in coming weeks](https://www.ft.com/content/a1014427-c2ce-4204-b41a-001277309cea)** ([Score: 108, Comments: 31](https://reddit.com/r/LocalLLaMA/comments/1j5ievy/ft_llama_4_w_voice_expected_in_coming_weeks/)): **Meta** and **Sesame** are anticipated to release **Llama 4** with integrated voice capabilities in the coming weeks, providing options for self-hosted voice chat. The author expresses interest in an **iOS app** with **CarPlay** integration for interacting with a private AI server.
  - Commenters discuss the potential features and capabilities of **Llama 4**, expressing desires for smaller models (0.5B to 3B) like **Llama-3.2-1B** for rapid experimentation, and larger models (10~20B) with performance comparable to **Qwen 32B**. There is also a conversation around the integration of reasoning abilities, with some preferring a separate model for reasoning tasks to avoid delays in voice interactions.
  - The anticipated release timeline for **Llama 4** is debated, with some predicting it will coincide with **LlamaCon** to boost the event's profile, and others speculating a release between mid-March and early April. The discussion includes a link to a [preview image](https://preview.redd.it/to8wtlt0pane1.jpeg?width=1320&format=pjpg&auto=webp&s=6f75917ab7418282f277318c6ea58d472ae6c8d3) related to the release.
  - There are concerns about paywalled content, with users sharing a workaround link to [archive.ph](https://archive.ph/9C732) for accessing the article summary. Additionally, there's a mention of bias in models, with a preference for models that do not engage in moral lecturing.


**Theme 2. QwQ-32B Performance Settings and Improvements**

- **QwQ-32B infinite generations fixes + best practices, bug fixes** ([Score: 214, Comments: 26](https://reddit.com/r/LocalLLaMA/comments/1j5qo7q/qwq32b_infinite_generations_fixes_best_practices/)): To address **infinite repetitions with QwQ-32B**, the author suggests using specific settings such as `--repeat-penalty 1.1` and `--dry-multiplier 0.5`, and advises adding `--samplers "top_k;top_p;min_p;temperature;dry;typ_p;xtc"` to prevent infinite generations. The guide also recommends **Qwen team's** advice for long context (128K) to use **YaRN** and provides links to various quantized models, including dynamic 4bit quants, available on [Hugging Face](https://huggingface.co/unsloth/QwQ-32B-unsloth-bnb-4bit).
  - **Parameter Overrides in Llama-Server**: When using **llama-server**, command line parameters like `--temp 0.6` can be overridden by HTTP request parameters such as `{"temperature":1.0}`, affecting the final output. More details are available in a [discussion on GitHub](https://github.com/ggml-org/llama.cpp/discussions/11394).
  - **GRPO Compatibility with QwQ-32B**: Users inquired about running **GRPO** with **QwQ-32B** for low GPU resources, and it was confirmed that it works by simply changing the model name in the [GRPO notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb).
  - **Chat Template Usage**: It's important to follow the exact chat template format for **QwQ-32B**, including newlines and tags like `<think>`. However, omitting the `<think>` tag can still work as the model adds it automatically, and system prompts can be used by prepending them with `<|im_start|>system\n{system_prompt}<|im_end|>\n`. More details can be found in the [tutorial](https://docs.unsloth.ai/basics/tutorial-how-to-run-qwq-32b-effectively).


- **Ensure you use the appropriate temperature of 0.6 with QwQ-32B** ([Score: 107, Comments: 9](https://reddit.com/r/LocalLLaMA/comments/1j5hb4p/ensure_you_use_the_appropriate_temperature_of_06/)): The author initially faced issues with generating a **Pygame script** for simulating a ball bouncing inside a rotating hexagon due to incorrect settings, which took 15 minutes without success. They later discovered that the **Ollama** settings had been updated, recommending a temperature of **0.6** for **QwQ-32B**, with further details available in the [generation configuration](https://huggingface.co/Qwen/QwQ-32B/blob/main/generation_config.json).
  - **Deepseek** and other reasoning models often perform optimally with a temperature parameter of **0.6**, which aligns with the author's findings for **QwQ-32B**. This temperature setting seems to be a common recommendation across different models.


**Theme 3. QwQ vs. qwen 2.5 Coder Instruct: Battle of 32B**

- **[AIDER - As I suspected QwQ 32b is much smarter in coding than qwen 2.5 coder instruct 32b](https://i.redd.it/yfu1j9qhw5ne1.png)** ([Score: 241, Comments: 79](https://reddit.com/r/LocalLLaMA/comments/1j5ao2j/aider_as_i_suspected_qwq_32b_is_much_smarter_in/)): **QwQ-32B** outperforms **Qwen 2.5-Coder 32B-Instruct** in coding tasks, achieving a higher percentage of correct completions in the **Aider polyglot benchmark**. Despite its superior performance, QwQ-32B incurs a higher total cost compared to Qwen 2.5-Coder, as illustrated in a bar graph by **Paul Gauthier** updated on **March 06, 2025**.
  - **Graph Design Issues**: Several commenters, including **SirTwitchALot** and **Pedalnomica**, note confusion due to the graph's design, particularly the use of two y-axes and an unclear legend with missing colors. This misrepresentation complicates understanding the performance and cost comparison between **QwQ-32B** and **Qwen 2.5-Coder**.
  - **Performance and Configuration Concerns**: There is discussion around **QwQ-32B's** performance, with **someonesmall** highlighting its 20.9% completion rate in the **Aider polyglot benchmark** compared to **Qwen 2.5-Coder's** 16.4%, but with a much lower correct diff format rate (67.6% vs. 99.6%). **BumbleSlob** and others express dissatisfaction with the model's performance, suggesting parameter adjustments as potential solutions.
  - **Model Size and Usability**: **krileon** and others discuss the practicality of running large models like **QwQ-32B** on consumer hardware, with suggestions like acquiring a used **3090 GPU** to improve performance. The conversation reflects on the challenges of using large models without high-end hardware, and the potential future accessibility of more powerful GPUs.


**Theme 4. Meta's Latent Tokens: Pushing AI Reasoning Forward**

- **Meta drops AI bombshell: Latent tokens help to improve LLM reasoning** ([Score: 333, Comments: 40](https://reddit.com/r/LocalLLaMA/comments/1j59fue/meta_drops_ai_bombshell_latent_tokens_help_to/)): Meta AI researchers discovered that using **latent tokens** generated by compressing text with a **vqvae** enhances the reasoning capabilities of **Large Language Models (LLMs)**. For more details, refer to the paper on [arXiv](https://arxiv.org/abs/2502.03275).
  - **Latent Tokens and Reasoning Efficiency**: The use of **latent tokens** created via **VQ-VAE** compresses reasoning steps in LLMs, leading to more efficient reasoning by reducing the need for verbose text explanations. This method allows LLMs to handle complex tasks with fewer computational resources and shows improved performance in logic and math problems compared to models trained only on full text explanations.
  - **Mixed Reception on Impact**: While some users see potential in this approach, others like **dp3471** express that the gains are relatively small and expect more significant improvements when combined with other techniques like **progressive latent block transform**. **Cheap_Ship6400** highlights that Meta's latent reasoning differs from **Deepseek's MLA**, focusing on token embedding space rather than attention score calculations.
  - **Discussion on Implementation and Future Prospects**: There is curiosity about the implementation details, particularly how the **VQ-VAE** is used for next-token prediction with discrete high-dimensional vectors. Some users, like **-p-e-w-**, hope for practical applications of these theoretical breakthroughs, while others discuss the potential of reasoning in latent space and compare it to other emerging techniques like **diffusion LLMs**.


## Other AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding

error in pipeline that we are debugging... sorry

---

# AI Discord Recap

> A summary of Summaries of Summaries by Gemini 2.0 Flash Thinking

**Theme 1. IDE Showdown: Cursor, Windsurf, and the Code Editor Arena**

- **Cursor Crushes Codeium in Credit Consumption Contest!**: Users report [**Cursor**](https://www.cursor.sh/) is more efficient and stable than **Windsurf**, especially with large files, with credit usage on Cursor being *a fraction of Windsurf's Claude 3.7* costs.  **Windsurf** users are facing **high credit consumption** due to repeated code rewrites and file analysis, with one user burning through **1200 credits** in a day, while others switch to **Cursor** or free alternatives like **Trae** for better resource management.
- **Cursor 0.47 Comforts Code Creation Chaos!**:  [**Cursor 0.47**](https://www.cursor.sh/) update resolves **Sonnet 3.7** issues, now following custom rules and behaving better for code generation, particularly for tasks like creating a [welcome window for VS Code forks](https://code.visualstudio.com/api/ux-guidelines/welcome). Users noted the use of sequential thinking with multiple code paths in the updated version, improving code creation workflow.
- **MCP Mayhem: Server Setup Snags Users**: Users are struggling to set up **MCP servers** for code editors like Windsurf, encountering errors like *"Not Found: Resource not found"* when attempting to connect to different models.  The difficulty in establishing **Model Context Protocol (MCP)** connections highlights ongoing challenges in integrating and utilizing external services with AI code editors.


**Theme 2. Model Benchmarks and Optimization Breakthroughs**

- **QwQ-32B Challenges R1 for Local Coding Crown**:  [**QwQ-32B**](https://artificialanalysis.ai/models/qwq-32b) is touted as a game-changer for local AI coding, potentially achieving state-of-the-art performance at home, although benchmarks suggest it may not surpass **R1** in all aspects.  [Daniel Han](https://x.com/danielhanchen/status/1898035752124166368) posted debugging tips for **QwQ-32B** looping issues, recommending sampler adjustments in llama.cpp and noting sensitivity to quantization, advising that *the first and last layers should be left unquantized*.
- **ktransformers Claims IQ1 Quantization Crushes BF16**:  **ktransformers** posted benchmarks claiming **Deepseek R1 IQ1** quantization outperforms **BF16**, sparking skepticism and debate in the **Unsloth AI** Discord, with one member noting *they would've wrote 1.58bit tho*.  While the table was deemed incomplete, ongoing benchmarking is comparing **Deepseek v3 (chat model) vs R1 IQ1**, suggesting significant interest in extreme quantization methods for high-performance models.
- **RoPE Scaling Rescues Long Context Qwen2.5-Coder-3B**:  [**Qwen2.5-Coder-3B-bnb-4bit**](https://huggingface.co/Qwen/Qwen2.5-Coder-3B-bnb-4bit) model's context length of **32768** is extended to **128000** using [kaiokendev's RoPE scaling of 3.906](link-to-rope-explanation), confirming that *with RoPE, as long as the architecture is transformers, the 128k tokens can be attended to*. This demonstrates a practical method for significantly expanding the context window of smaller models.


**Theme 3. Diffusion Models Disrupt Language Generation**

- **Inception Labs Diffuses into Text Generation with Midjourney-Sora Speed**: [Inception Labs](https://inceptionlabs.ai/) is pioneering diffusion-based language generation, aiming for unprecedented speed, quality, and generative control akin to **Midjourney** and **Sora**.  Members noted the emergence of open-source alternatives like [ML-GSAI/LLaDA](https://github.com/ML-GSAI/LLaDA), suggesting diffusion models could revolutionize language generation with significant speed gains.
- **Discrete Diffusion Models Get Ratios, Sparking Efficiency**:  A member highlighted the paper *Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution* from [arxiv](https://arxiv.org/pdf/2310.16834) as potentially underpinning [Inception Labs's product](https://www.inceptionlabs.ai/about), noting its focus on estimating data distribution ratios for efficient diffusion modeling.  Solving for the **duals λ** involves optimizing over the **Lambert W function**, which can be computationally intensive, prompting suggestions for using **cvxpy** and the **adjoint method** for optimization.
- **LLaDA Model: Diffusion-based LLM Paradigm Shift Emerges**:  [Large Language Diffusion Models (LLaDA)](https://diffusionllm.net/) represent a novel paradigm in language model architecture, using a denoising diffusion process for parallel, coarse-to-fine text generation, contrasting with traditional autoregressive Transformers.  Despite conceptual appeal, **LLaDA** and similar models may be limited by training focused on benchmarks rather than broad real-world tasks, with members observing limitations like repeat paragraph issues.


**Theme 4. MCP and Agent Security Threats Loom Large**

- **MCP Servers Face Malicious Prompt Injection Menace**: Concerns are escalating about **MCP servers** delivering malicious prompt injections to AI agents, exploiting models' inherent trust in tool calls over internal knowledge.  Mitigation strategies proposed include displaying tool descriptions upon initial use and alerting users to instruction changes to prevent potential exploits.
- **Security Channel Scrutinizes MCP Exploits**:  The community is considering establishing a dedicated security channel to proactively address and prevent **MCP exploit** vulnerabilities, emphasizing the inherent risks of connecting tools to **MCP Servers/Remote Hosts** without full awareness of consequences.  Discussions highlighted the potential for regulatory compliance tool descriptions to be manipulated to trick models into incorporating backdoors.
- **Perplexity API's Copyright Indemnification Raises Red Flags**: Users flagged copyright concerns with the **Perplexity API** due to its scraping of copyrighted content, noting that [Perplexity's API terms](https://www.perplexity.ai/hub/legal/perplexity-api-terms-of-service) shifts liability onto the user for copyright infringement.  Alternatives offering IP indemnification, such as **OpenAI**, **Google Cloud**, and **AWS Bedrock**, were highlighted as potentially safer options regarding copyright risks, with links to their respective terms of service.


**Theme 5. Hardware Hustle: 9070XT vs 7900XTX and Native FP4 Support**

- **9070XT Smokes 7900XTX in Raw Inference Speed**:  The **9070XT** GPU outperforms the **7900XTX** in inference speed, running the same **qwen2.5 coder 14b q8_0** model at **44tok/sec** versus **31tok/sec**, and exhibiting a sub-second first token time compared to the **7900XTX**'s 4 seconds.  Despite using **Vulkan** instead of **ROCm** due to driver limitations on Windows, the **9070XT** demonstrates a significant performance edge, with some users reporting up to **10%** better performance overall.
- **Native FP4 Support Arrives on New GPUs**:  The **9070** GPU shows substantial improvements in **FP16** and **INT4/FP4** performance compared to older cards, with **FP16** performance jumping from **122 to 389** and **INT4/FP4** performance surging from **122 to 1557**.  This signifies the emergence of native **FP4** support from both **Nvidia** and **Radeon**, opening new possibilities for efficient low-precision inference and training.
- **Vulkan vs ROCm Performance Debate Heats Up**:  While the **9070XT** currently lacks **ROCm** support on Windows, utilizing **Vulkan** via **LM Studio**, some users are reporting surprisingly competitive inference speeds, even surpassing **ROCm** in certain scenarios.  However, others maintain that **ROCm** should inherently be faster than **Vulkan**, suggesting potential driver issues might be skewing results in favor of Vulkan in some user benchmarks.


---

# PART 1: High level Discord summaries




## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **Cursor Clashes with Lmarena in Quick Contest!**: Members compared [**Cursor**](https://www.cursor.sh/) and [**Lmarena**](https://lmarena.ai/) using a single prompt with **GPT-3.7**, with initial impressions favoring Cursor for output quality, citing unreadable text generated by Lmarena.
   - Later analysis suggested **Lmarena** better respected the theme, but the general consensus was that *both were dogshit*.
- **Cursor 0.47 Comforts Code Creation!**: After updating, users reported that [**Cursor 0.47**](https://www.cursor.sh/) fixed **Sonnet 3.7** issues by following custom rules, complying with AI custom rules, and behaving better for code generation, particularly for a [welcome window to a VS Code fork](https://code.visualstudio.com/api/ux-guidelines/welcome).
   - They noted the use of sequential thinking with multiple code paths for increased speed.
- **Vibe Coding: Validated and Visited!**: A discussion about *vibe coding*, building micro-SaaS web apps, led to a user building a welcome window on a **VS Code fork** using **Claude** itself.
   - A member defined the practice as *treating the ai agent like a child that needs guidance*, contrasting it with building a dockerized open-source project requiring orchestration experience.
- **MCP: Maximizing Model Power!**: Members explored using **Model Context Protocol (MCP)** servers to enhance AI code editors like Cursor and Claude, connecting to services like Snowflake databases.
   - One member found that **PearAI** offered full context, while another discovered that `.cursorrules` tend to be ignored in Cursor versions above **0.45**.
- **Cursor Error Inspires Swag!**: The infamous **"Cursor is damaged and can't be opened"** error message has inspired merchandise, with [T-shirts](https://www.redbubble.com/shop/ap/169071328) and [mousepads](https://www.redbubble.com/i/mouse-pad/Cursor-AI-Error-by-TheGalaxyStars/169071328.G1FH6) now available.
   - This comedic turn underscores the community's ability to find humor in the face of technical frustrations.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **ktransformers Claims IQ1 Crushes BF16**: Members discussed **Deepseek R1 IQ1** benchmarks posted by **ktransformers**, with one skeptical member noting *they would've wrote 1.58bit tho*.
   - Another member noted the table was incomplete and the benchmarking was still in progress showing **Deepseek v3 (the chat model) vs R1 IQ1**.
- **Unsloth's GRPO Algorithm Revealed**: Members speculated on how **Unsloth** achieves low VRAM usage, suggesting asynchronous gradient offloading, however the real savings stem from re-implementing **GRPO** math.
   - Other efficiencies come from gradient accumulation of intermediate layers, coupled with gradient checkpointing, and more efficient kernels like **logsumexp**.
- **QwQ-32B Generation Looping Fixed**: Daniel Han posted a guide on debugging looping issues with **QwQ-32B** models recommending samplers to llama.cpp such as *--samplers "top_k;top_p;min_p;temperature;dry;typ_p;xtc"* and uploaded [dynamic 4bit quants & GGUFs](https://huggingface.co/unsloth/QwQ-32B-unsloth-bnb-4bit).
   - He said that **QwQ is also sensitive to quantization** - *the first and last few layers should be left unquantized*.
- **Qwen's RLHF Success Story**: One member achieved **RLHF** success using **Unsloth GRPO** on a **Qwen7b** model, reporting significant improvements in role adherence.
   - However, the model showed *noticeable degradation* in strict instruction-following benchmarks like **IFeval** especially with formatting constraints and negative instructions.
- **RoPE Scaling to the Rescue**: The **Qwen2.5-Coder-3B-bnb-4bit** model can handle a sequence length of **32768**, but with **kaiokendev's RoPE scaling of 3.906**, it was extended to **128000**.
   - It was confirmed that [with RoPE](link-to-rope-explanation), as long as the architecture is transformers, the 128k tokens can be attended to.



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Registry Editing can cause Blue Screen**: A member shared that deleting a crucial **.dll file** after identifying it as a RAM hog led to a **blue screen** on the next boot, warning of registry editing risks without backups.
   - Members generally agree that it's wise to *backup personal files and reformat* after tweaking the registry.
- **Quantization Affects Performance and Memory**: Users discussed quantization, noting a preference for **f16 quantization** to fit more parameters in a smaller load, while acknowledging that other quantizations may cause crashes with flash attention.
   - In the context of quantization, someone described floating points as its *signed bit size so signed 16 int is 32*.
- **Windows has File Path Length Limits**: Members discussed the limitations of **file path lengths in Windows**, noting the standard **260-character limit** due to the **MAX_PATH** definition in the Windows API, as explained in [Microsoft documentation](https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation?tabs=registry).
   - This limitation can be bypassed by enabling long path behavior per application via registry and application manifest modifications for NT kernel path limit of 32,767.
- **Inception Labs makes Diffusion-Based Text**: [Inception Labs](https://inceptionlabs.ai/) is pioneering diffusion-based language generation which promises unprecedented speed, quality, and generative control inspired by AI systems like **Midjourney** and **Sora**.
   - Members noted that open source alternatives are in development, such as [ML-GSAI/LLaDA](https://github.com/ML-GSAI/LLaDA), and that significant speed increases may be on the horizon with this technology.
- **Trustless Authentication is the Future**: Members discussed the potential of **trustless authentication**, where a 3D scan of a face and ID is converted into an encrypted string to serve as a digital passport, similar to the business model used by [Persona](https://www.withpersona.com/)
   - It's envisioned as a trustless verification database with users' personalization tags in the file, where generated deepfake facial scans and IDs would not work.



---



## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf Suffers Stability Surfing Setbacks**: Users report that **Windsurf** is **unstable** after the latest updates, noting issues like **infinite loops**, **repeating code changes**, and general **unresponsiveness**.
   - Many users are switching to **Cursor** or **Trae**, citing better stability and resource management, with one user stating, *"I have decided to switch to Cursor...Presumably more stable. I will come back in a month or so, see how it's doing."
- **Credits Crunch: Windsurf's Costly Consumption Catches Critics**: Users are experiencing **high credit consumption** due to the AI repeatedly rewriting code or analyzing the same files, leading to frustration.
   - One user complained about burning through **1200 flow credits** in one day after the introduction of **Claude 3.7**, calling it *"100% unstable and 100% u pay for nothing"*, suggesting the tool reads only 50 lines at a time and takes 2 credits for reading a 81 line file.
- **Cascade Terminal Plagued with Problems**: Some users report the **Cascade terminal** disappearing or becoming unresponsive, with no visible settings to fix it.
   - One user mentioned a temporary solution of restarting, but the issue reoccurs, while another suggested using `CTRL Shift P` to clear all cache and reload windows.
- **Model Melee: Cursor Crushes Codeium in Credit Contest**: Users are comparing Windsurf to **Cursor**, with many finding Cursor to be more efficient and stable, especially when handling large files.
   - One user reported that credit usage on Cursor is *"a fraction of windsurf with claude 3.7"* while another reports *"Trae is doing x100 better than ws while its free"*.
- **MCP Mayhem: Users Struggle with Server Setup**: Users are encountering issues while trying to use **MCP servers**, with errors related to the model or configuration.
   - One user received a *"Not Found: Resource not found"* error and tried different models without success.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Pro Perks Provoke Problems**: Several users reported unexpected cancellations of their **Perplexity Pro** accounts, suspecting a scam, later revealed to be due to ineligibility for the "DT 1 Year Free HR" offer meant for **Deutsche Telekom** customers in **Croatia**, as stated in the [terms of service](https://www.perplexity.ai/legal/terms).
   - Support confirmed the cancellations were due to this offer, intended for users in **Croatia**, creating confusion among those who were not eligible.
- **GPT-4.5 Tier Talk Triggers Theories**: Users discussed the availability of **GPT-4.5** with **Perplexity Pro**, with confirmations of **10 free uses per 24 hours**.
   - It was clarified that **GPT-4.5** is very expensive to use, suggesting that the auto model selection might be sufficient to avoid manually picking a model.
- **Complexity Extension Creates Canvas Capabilities**: A user explained how to generate mermaid diagrams within Perplexity using the **Complexity extension**, providing canvas-like features.
   - By enabling the canvas plug-in and prompting the AI to create a mermaid diagram, users can render it by pressing the play button on the code block, with more info available in [this Discord link](https://discord.com/channels/1245377426331144304/1246406910962438245/1347554810928566304).
- **Google Gemini 2.0 Grabs Ground**: An [article from ArsTechnica](https://arstechnica.com/google/2025/03/google-is-expanding-ai-overviews-and-testing-ai-only-search-results/) details **Google's** expansion of **AI search features** powered by **Gemini 2.0**.
   - **Google** is testing an **AI Mode** that replaces the traditional search results with **Gemini** generated responses.
- **Prime Positions AI Powered Product Placement**: A [Perplexity page](https://www.perplexity.ai/page/amazon-prime-tests-ai-dubbing-pHEI1t6XRn6DilTOLGBGew) covers how **Amazon Prime** is testing **AI dubbing** for its content.
   - Additional perplexity pages include: [Apple's Foldable iPhone](https://www.perplexity.ai/page/apple-s-foldable-iphone-predic-WSdZuoG7Rw6VvayJJg0DVQ), [OpenAI's AI Agent](https://www.perplexity.ai/page/openai-s-20000-ai-agent-nvz8rzw7TZ.ECGL9usO2YQ), and [DuckDuckGo's AI Search](https://www.perplexity.ai/page/duckduckgo-s-ai-search-option-D2sL.5w8S4mQYdr_XAlgjw).



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio Tweaks Template Bugs, Speeds RAG**: **LM Studio 0.3.12** is now stable, featuring bug fixes for **QwQ 32B jinja parsing** that previously caused an `OpenSquareBracket !== CloseStatement` error, detailed in the [full release notes](https://lmstudio.ai/blog/lmstudio-v0.3.12).
   - Additionally, the speed of chunking files for retrieval in **RAG** has been significantly improved, with MacOS systems now correctly indexing MLX models on external exFAT drives.
- **Qwen Coder Edges out DeepSeek on M2**: For coding on a Macbook M2 Pro with 16GB RAM, **Qwen Coder** is recommended over **DeepSeek v2.5** due to memory constraints, acknowledging that performance will still lag behind cloud-based models.
   - Members noted **Qwen 32B** punches above its weight and performs at least as well as **Llama 3.3 70b**.
- **Unsloth Fast-Tracks LLM Finetuning**: [Unsloth](https://github.com/unslothai/unsloth) was recommended for faster finetuning of models like **Llama-3**, **Mistral**, **Phi-4**, and **Gemma** with reduced memory usage.
   - Members noted that *finetuning is much more resource intensive than inference* and LM Studio currently does not have a public roadmap that includes this feature.
- **9070XT smashes 7900XTX in Raw Speed**: A user compared the **9070XT** to the **7900XTX** running the same **qwen2.5 coder 14b q8_0** model and found that the **9070XT** runs about **44tok/sec** with under a second to first token, while the **7900XTX** runs **31tok/sec** with **4sec** to first token.
   - One user believes those seeing better **Vulkan** performance are *having driver issues* because *rocm should be faster than vulkan, by far*.
- **Native FP4 Support Arrives**: The **9070** has significantly improved **FP16** and **INT4/FP4** performance compared to older cards, with **122 vs 389 FP16** and **122 vs 1557 INT4/FP4**.
   - This signals that native **FP4** support is available from both **Nvidia** and **Radeon**, as well as discussions of the impact of quantization on model quality, particularly the trade-offs between smaller quant sizes and potential quality loss.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Open-Source Agent Alternatives to Replit Spark Debate**: Members discussed open-source alternatives with agent functionality similar to **Replit/Bolt**, and to *export_to_video* for direct video saving to a **Supabase bucket**.
   - The community discussed alternatives to commercial offerings with integrated agent functionality.
- **Dexie Wrapper Proposal for Gradio Apps**: A user proposed a **Dexie wrapper** for easier **IndexedDB** access in **Gradio apps**, sparking discussion on its implementation as a custom component.
   - A link to the **Gradio developers Discord channel** was shared for further discussion.
- **Dataset Citation Drama Unfolds**: A user suspected the research paper, '[NotaGen: Symbolic Music Generation with LLM Training Paradigms](https://arxiv.org/abs/2502.18008)', of using their **Toast MIDI dataset** without proper citation.
   - Community members advised contacting the corresponding author and generating a **DOI** for the dataset to ensure proper attribution and recognition of the dataset by academic software.
- **Guidance Requested on OCR-2.0 Fine-Tuning**: A member requested guidance on fine-tuning **OCR-2.0**, which they identified as the latest and best model, linking to a [SharePoint document](https://bama365-my.sharepoint.com/:w:/g/personal/xgranja_ua_edu/EeSz8D6iYPxHhzfQD3GGzsYBARpsSkbEDZWzoQH7hIH4lg?e=gMOaR4).
   - The member inquired about the appropriate steps for fine-tuning the **got/ocr-2.0** models according to the documentation.
- **Decoding the AI Agent: Component vs. Entity**: An **LLM** is a component with tool-calling abilities, not an **Agent** itself; an **agentic AI model** requires both, according to [this response](https://discord.com/channels/879548961488179230/1201995434137206834/1208022892900941824).
   - Debate continues whether a **Retrieval Augmented Generation (RAG)** system could be considered an "environment" with which the **LLM** interacts.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Perplexity API Exposes Copyright Concerns**: Users flagged copyright issues with the **Perplexity API** due to its scraping of copyrighted content, noting that [Perplexity's API terms](https://www.perplexity.ai/hub/legal/perplexity-api-terms-of-service) shifts liability onto the user.
   - Alternatives such as **OpenAI**, **Google Cloud**, and **AWS Bedrock** offer IP indemnification, shifting the risk to the vendor, see: [OpenAI terms](https://learn.microsoft.com/en-us/legal/cognitive-services/openai/customer-copyright-commitment), [Google Cloud Terms](https://cloud.google.com/terms/generative-ai-indemnified-services,), and [AWS Bedrock FAQs](https://aws.amazon.com/bedrock/faqs/).
- **Sonar Deep Research Models Flounder with Errors**: Members reported **Perplexity Sonar Deep Research** experiencing frequent errors, high latency (up to **241 seconds** for the first token), and unexpectedly high reasoning token counts.
   - One member mentioned a **137k reasoning token** count with no output, while others confirmed the model eventually stabilized.
- **Gemini Embedding Text Model Debuts**: A new experimental **Gemini Embedding Text Model** (**gemini-embedding-exp-03-07**) is now available in the [Gemini API](https://ai.google.dev/gemini-api/docs/models/experimental-models), surpassing previous models, see [Google AI Blog](https://developers.googleblog.com/en/gemini-embedding-text-model-now-available-gemini-api/).
   - This model now holds the top rank on the **Massive Text Embedding Benchmark (MTEB) Multilingual leaderboard** and also supports longer input token lengths.
- **OpenRouter Reasoning Parameter Riddled with Inconsistencies**: Users found inconsistencies in OpenRouter's reasoning parameter, some models marked as supporting reasoning despite endpoints lacking support, while some providers did not return reasoning outputs.
   - Tests revealed configuration issues and discrepancies between the models and endpoints, with **Cloudflare** noted to lack a **/completions endpoint**.
- **Claude 3.7 Stumbles with Russian Prompts**: A user reported **Claude 3.7** struggling with **Russian language prompts**, responding in English and potentially misunderstanding nuances.
   - The issue arose while using cline with OpenRouter, suggesting the issue might stem from **Anthropic** rather than the extension or OpenRouter itself.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Minion.ai Meets its Maker**: Members reported that [Minion.ai](https://minion.ai) is *dead* and not to believe the hype, with some describing it as *that thing with like the 4-cartoony looking characters that were supposed to go and do agent stuff for you*.
   - The closure highlights the volatile nature of AI startups and the need for critical evaluation of marketed capabilities.
- **Google Expands Gemini Embedding Model**: **Google** released an experimental **Gemini Embedding Model** for developers with [SOTA performance on MTEB (Multilingual)](https://x.com/officiallogank/status/1898081742767919384?s=46).
   - The updated model features an **input context length of 8K tokens**, **output of 3K dimensions**, and **support for over 100 languages**.
- **Claude Code Enters IDE Arena**: Members discussed comparing **Claude code** with **cursor.sh** and **VSCode+Cline / roo-cline**, prioritizing code quality over cost.
   - The discussion references a [previous message](https://discord.com/channels/822583790773862470/1075282825051385876/1346679448174596219) indicating an ongoing exploration of optimal coding environments.
- **AI Personas Become the Meme Overlords**: Referencing a [tweet](https://x.com/defiapes/status/1855657706205352035) one member mentioned that the norm will shift to **AI PERSONAS** that embody personality types.
   - The tweet mentions that **agents will race to be THE main face of subgroup x, y, and z** to capture the perfect encapsulation of shitposters, stoners, athletes, rappers, liberals, and memecoin degens.
- **Agent-as-a-Service, the Future?**: Members pondered the viability of an **Agent-as-a-Service** model.
   - One member floated the idea of a bot that sits in front of DoorDash, referred to as *DoorDash MCP*.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **OpenAI Squeezes ChatGPT Message Limit**: Users are reporting **ChatGPT** now enforces a **50 messages/week** limit to prevent misuse of the service.
   - Members were frustrated at the inconvenience, and debated workarounds.
- **HVAC gets LLM AI-Pilot**: A member shared a [YouTube video demo](https://youtu.be/oAiUAzKLe_Y) of an LLM trained on **HVAC installation manuals**.
   - Another member suggested using [Mistral OCR](https://mistral.ai/) to process manuals, praising its ability to read difficult fonts and tables cheaply.
- **Local LLMs Show Cloud Promise**: Members discussed running **LLMs** locally versus using cloud services, with one advocating for the potential of local DIY LLMs over black box cloud solutions.
   - Discussion included splitting processes between **GPU and CPU**, and the benefits of **quantization** in reducing memory requirements.
- **SimTheory's Claims Spark Skepticism**: A user promoted [SimTheory](https://simtheory.ai/) for offering a higher **O1 message cap limit** than **OpenAI** at a lower price.
   - Other members voiced disbelief, questioning how they could undercut **OpenAI's** pricing while providing a higher limit.
- **Models Mimic, Missing Opportunities**: Models tend to closely follow request patterns, potentially overlooking better methods, especially with increased **model steerability**.
   - One said, *When given code to screw in nails with a hammer, the model runs with it, guessing we know what we want*.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider Advocates Reasoning Indicator**: Users have requested a feature in Aider to indicate when the reasoning model is actively reasoning, particularly for **Deepseek R1** or **V3** on **Openrouter**.
   - A member proposed patching Aider and **litellm**, referencing a *hack* in litellm to retrieve reasoning tokens inside **<think>** tags.
- **Jamba Models strut Mamba-Transformer MoE**: **AI21 Labs** launched **Jamba 1.6 Large & Mini** models, claiming better quality and speed than open models and strong performance in long context tasks with a **256K context window**.
   - The models uses a **Mamba-Transformer MoE** hybrid architecture for cost and efficiency gains, and can be deployed self-hosted or in the **AI21 SaaS**.
- **Startups Swear by AI-Written Code**: A **Y Combinator** advisor mentioned that *a quarter of the current startup cohort has businesses almost entirely based on AI-written code*.
   - The context provided no further details.
- **Copilot Bans Aider User**: A user reported a **Copilot account suspension** due to *very light use of copilot-api in aider*, urging caution to other users.
   - Others speculated about possible causes like account sharing or rate limiting, with a reference to the [copilot-api GitHub repo](https://github.com/ericc-ch/copilot-api/blob/master/README.md).
- **QwQ-32B Challenges R1**: A member shared [a link](https://x.com/victormustar/status/1898001657226506362) about **QwQ-32B**, asserting it has changed local AI coding forever and achieves state-of-the-art (SOTA) performance at home.
   - Another member pointed to a benchmark discussion suggesting **QwQ-32B** might be good but not necessarily better than **R1**, especially considering the model size in [Discord](https://discord.com/channels/1131200896827654144/1346923740667187502).



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **DeepSeek Faces Corporate Bans**: Many companies are banning **DeepSeek** due to security concerns, despite its open-source nature, sparking debate over media influence, Chinese government influence, and the need for **reviewable code** and local execution.
   - One member noted that DeepSeek is like **Deep Research + Operator + Claude Computer**.
- **Discrete Diffusion Models Get Ratios**: A member suggested discussing the paper *Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution* from [arxiv](https://arxiv.org/pdf/2310.16834), noting it likely underpins [Inception Labs's product](https://www.inceptionlabs.ai/about).
   - It was further stated that solving for the **duals λ** requires optimizing over the **Lambert W function** and that this can be computationally inefficient, suggesting the usage of **cvxpy** and **adjoint method**.
- **Latent Abstraction Reduces Reasoning Length**: A new [paper](https://arxiv.org/abs/2502.03275) proposes a hybrid reasoning representation, using **latent discrete tokens** from **VQ-VAE** to abstract initial steps and shorten reasoning traces.
   - A member questioned if this **latent reasoning** is merely **context compression**, with another joking that electroconvulsive therapy (ECT) might be needed to prevent the misuse of the term *reasoning* in AI.
- **OpenAI Softens AGI Stance**: **OpenAI** is reportedly shifting from expecting a sudden **AGI** breakthrough to viewing its development as a continuous process, according to [the-decoder.com](https://the-decoder.com/openai-shifts-away-from-sudden-agi-breakthrough-theory/).
   - This shift may be a result of the underwhelming reception of **GPT 4.5**, with some blaming **Sam Altman** for setting expectations too high.
- **Sampling Thwarts Agent Loop**: The use of **n-sigma-sampling**, detailed in [this paper](https://arxiv.org/abs/2411.07641) and [GitHub repo](https://github.com/Tomorrowdawn/top_nsigma), seems to mitigate bad samples and looping behavior in multi-step agentic workflows.
   - The technique filters tokens efficiently without complex probability manipulations, maintaining a stable sampling space regardless of temperature scaling.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **VSCode Embraces MCP in Copilot**: **VSCode** plans to add MCP support to **GitHub Copilot**, as showcased on their [live stream](https://youtu.be/Pe8ghwTMFlg).
   - Community members are discussing the implications for both open-source and closed-source MCP implementations.
- **MCP Servers Face Prompt Injection Threat**: Concerns were raised regarding **MCP servers** serving malicious prompt injections to AI agents, exploiting the trust models place in tool calls over internal knowledge.
   - Suggested mitigations include displaying a list of tool descriptions upon first use and alerting users to any instruction changes to prevent exploits.
- **Security Channel Targets MCP Exploits**: Community members considered creating a security channel to proactively prevent exploits, emphasizing the risks of connecting tools to **MCP Servers/Remote Hosts** without understanding the consequences.
   - The discussion highlighted the potential for regulatory compliance tool descriptions to trick models into including backdoors.
- **Python MCP Quickstart Confuses Users**: Users reported errors when running the **Python MCP quickstart** ([here](https://modelcontextprotocol.io/quickstart/client)), leading to a recommendation to use [wong2/mcp-cli](https://github.com/wong2/mcp-cli) as a better alternative.
   - This alternative is considered easier to use and more reliable for initial MCP setup.
- **Swagger Endpoints Turn MCP-Friendly**: A member is developing **mcp-openapi-proxy** to convert any swagger/openapi endpoint into discoverable tools, mirroring the design of their **mcp-flowise server**.
   - Their latest mcp-server [functions in 5ire](https://5ire.org/) but encounters issues with Claude desktop, indicating compatibility challenges across different platforms.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo's Flexibility Sparks Debate**: Discussions about **Mojo's dynamism** and its impact on performance led to considerations about use cases and the balance between flexibility and speed, referencing [this HN post](https://news.ycombinator.com/item?id=35811170).
   - Suggestions arose that dynamism should only penalize performance when actively used, though concerns persist about its potential impact on struct performance, even without classes.
- **Monkey Patching Faces Scrutiny**: The community explored alternatives to **monkey patching** to achieve dynamic behaviors in Mojo, such as function pointers or composition, citing that it's *slower, harder to understand, breaks static analysis tooling and generally doesn't do anything that proper polymorphism couldn't have done*.
   - The discussion highlighted that excessive monkey patching can result in harder-to-understand code that disrupts static analysis, arguing that its utility doesn't outweigh the drawbacks.
- **Python Library Porting Challenges**: Members tackled the obstacles of **porting Python libraries to Mojo**, especially regarding dynamism and global state, with the recommendation to utilize **CPython interop** for performance-critical components.
   - Concerns were voiced about porting libraries heavily reliant on Python's dynamism and global state, particularly when that library cannot function otherwise.
- **Protocol Polymorphism Gains Traction**: The advantages of **protocol polymorphism** in achieving polymorphism without a class tree were highlighted, with a reference to [PEP 544](https://peps.python.org/pep-0544/) for polymorphism without class hierarchies.
   - Some members endorsed using a hash table of function pointers for dynamic behaviors, favoring static typing and ownership rules for unit tests in Mojo.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **SOTA Agentic Methods are Simple Algorithmically**: Members discussed how **SOTA agentic methods** on Arxiv tend to involve relatively simple algorithms, resembling a small **state machine**.
   - This implies that complex framework abstractions might be unnecessary, suggesting simpler abstractions for data, state, and API call management suffice.
- **Triton Autotune use_cuda_graph Causes Confusion**: A member sought clarification on the `use_cuda_graph` argument in `triton.autotune`, unsure how it applies to single **CUDA kernels**.
   - The confusion stems from the fact that **CUDA graphs** typically optimize sequences of kernel launches, contrasting with the single-kernel scope of `triton.autotune`.
- **Nvidia's NCCL AllReduce Implemented with Double Binary Trees Beats Ring Topology**: The [NVIDIA blog post](https://developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4/) suggests that **double binary trees** in **NCCL 2.4** offer full bandwidth and lower logarithmic latency compared to **2D ring latency** for the AllReduce operation.
   - Double binary trees were added in NCCL 2.4 which *offer full bandwidth and a logarithmic latency even lower than 2D ring latency.*
- **WoolyAI Launches CUDA Abstraction Beta for GPU Usage**: **WoolyAI** has launched a beta for its [CUDA abstraction layer](https://docs.woolyai.com) that decouples Kernel Shader execution from CUDA applications, compiling applications to a new binary and shaders into a **Wooly Instruction Set**.
   - This enables dynamically scheduling workloads for optimal **GPU** resource utilization; they are currently supporting **PyTorch**.
- **Cute Kernels Autotunes CUDA and Triton Kernels**: A member released [Cute Kernels](https://github.com/mayank31398/cute-kernels), *a collection of kernels for speeding up training* through autotuning over **Triton** and **CUDA** implementations, and can dispatch to either **cutlass** or **Triton** automatically.
   - This implementation was used in production to train **IBM's Granite LLMs** because *the LLVM compiler can sometimes create more efficient code than the NVCC*, and makes sense to tune over the kernel backend.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Open Source AI Projects Seek New Members**: A new member seeks recommendations for interesting **open-source AI projects** in areas like **LLM pre-training**, **post-training**, **RL**, and **interpretability**, and suggested [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt/) for those interested in theory work.
   - Another member suggested that if the new member is interested in pretraining, the best project to get involved with is the **GPT-NeoX** training library.
- **Token Assorted Paper's Vocabulary is Questionable**: A member speculated that the **Token Assorted paper** might be simply adding the **codebook** to their vocabulary during fine-tuning for next token prediction on latent codes.
   - They criticized this approach as potentially not generalizable to open reasoning domains, suggesting that finding the K most common strings in the reasoning corpus could yield better results.
- **TorchTitan Sharding Requires All-Reduce**: In a discussion about **TorchTitan** embedding sharding, it was explained that with vanilla **TP**, if the input embedding is sharded on the vocab dimension, an **all-reduce** is required afterward to handle cases where the embedding layer outputs 0 for missing vocabulary elements, clarified in [this issue on Github](https://github.com/pytorch/torchtitan/issues/785#issuecomment-2585007139).
   - Members discussed the storage implications of having an embedding layer output zeros for missing tokens, noting that storage is needed right after communications, but that it's free if you can reuse that storage.
- **Logit Lens Illuminates Llama-2's Linguistic Bias**: A member shared their appreciation for [this paper on multilingual language models](https://arxiv.org/abs/2402.10588) and the use of **Logit Lens** in analyzing the **Llama-2 family**.
   - The paper explores how **Llama-2** models, trained on English-dominated data, handle non-English prompts, revealing phases where the model initially favors English translations before adjusting to the input language.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **yWorks unveils Real-Time Knowledge Graph Visualization**: @yworks showcased **yFiles**, its SDK for visualizing knowledge graphs, offering [real-time updates and dynamic interactions](https://t.co/mb6M2R3TTh).
   - This demo highlights the ability to dynamically update and interact with knowledge graphs in real-time.
- **Anthropic Cookbook Expanded**: The LlamaIndex team updated and expanded their **Anthropic Cookbook**, the authoritative source for learning about [basic API setup](https://t.co/SQQ63qmwRb) with simple completion and chat methods.
   - The cookbook serves as a comprehensive guide for setting up and utilizing **Anthropic's APIs** within the LlamaIndex framework.
- **Debugging Prompt Printing with SQLTableRetrieverQueryEngine**: A member inquired about printing the prompt used by **SQLTableRetrieverQueryEngine** in LlamaIndex, where another member shared a code snippet `from llama_index.core import set_global_handler; set_global_handler("simple")` to enable prompt printing.
   - The solution provides a practical method for debugging and understanding the prompts used by **SQLTableRetrieverQueryEngine** during query execution.
- **Jina AI Package encounters Installation Issues**: A member reported an import error with the **Jina AI** package, where it was suggested to install the provider package using `npm install @llamaindex/jinaai`, with a link to [LlamaIndex migration documentation](https://ts.llamaindex.ai/docs/llamaindex/migration/0.8-to-0.9).
   - The migration documentation explains the shift to provider packages in v0.9, resolving the import error by ensuring the correct package installation.
- **LlamaExtract Beta Tempts Eager Adopters**: A member requested access to the beta version of **LlamaExtract**, where they were instructed to DM a specific user with their email, also referencing the [LlamaExtract documentation](https://docs.cloud.llamaindex.ai/llamaextract/getting_started).
   - The documentation outlines the process for getting started with **LlamaExtract** and highlights key features for potential beta testers.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Command R7B Inference Dragging?**: A user reported slow inference times with **Command R7B** on Hugging Face, which was attributed to suboptimal user **hardware** or inefficient model execution.
   - The member clarified that the issue likely stems from the user's setup rather than the inherent performance of **Command R7B**.
- **Ollama Tool Issues Plague Users**: A user encountered issues with **tool invocation** using `command-r7b:latest` with **Ollama** and **Langchain**, receiving errors about missing tool access.
   - Guidance suggested ensuring correct **JSON format** for tool passing and verifying **Ollama**'s configuration for tool calling support.
- **Dev Seeks Open Source AI Gig**: A developer with experience in **pre-training GPT-2** and fine-tuning models on **Hellaswag** is seeking interesting open-source AI projects to contribute to.
   - The member is also interested in networking within the Vancouver, BC area.
- **504 Gateway Error Returns!**: Users reported recurring **504 Gateway Errors** and **502 Server Errors**, indicating temporary server-side issues.
   - The error messages advised retrying requests, suggesting the problems were transient.
- **Graphs Power Better Topic Models**: A member suggested using **Knowledge Graphs** for enhanced **topic modelling**, specifically recommending a **LLM** from **TogetherAI**.
   - They highlighted the generous free credits offered by **TogetherAI**, encouraging experimentation with their platform for **topic modelling** tasks.



---



## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **Podcast Cloning Gets Streamlined**: A user shared a [YouTube video](https://www.youtube.com/watch?v=0UWYFFOPjqs) showcasing **NotebookLM** integrated with **Wondercraft** for creating podcasts with professionally cloned voices, streamlining the podcast creation process.
   - Commentary suggests **Wondercraft** offers a more streamlined approach than **11Labs** and **HeyGen**, though its subscription price might be steep for non-monetized podcasts.
- **Drive Data Encryption Discussed**: A user pointed out that while data is encrypted during transmission to **Google Drive**, it is *not* encrypted on **Drive** itself.
   - This means that **Google**, successful hackers, and those with whom the user has shared the directory can access the data.
- **AI Voices Stuttering Like People**: An **AI speaker** in a generated audio file now *stammers like normal people*, which feels very natural, according to one user who attached an example wav file.
   - The user also mentioned that the stammering increases the audio length, potentially reducing the amount of actual information conveyed within **Google**'s daily limit.
- **Unlock Analyst/Guide Chat Settings Impact Timeline Generation**: A user discovered that **chat settings** such as *analyst/guide short/long* in **NotebookLM** affect the timeline and overview generation, as the hosts essentially request these settings during audio overview generation.
   - The user also noted their assistant combines the briefing overview, detailed timeline, character sheets, and raw sources into a single document.
- **NotebookLM gets Chrome Extensions**: Members discussed the possibility of uploading a list of URLs to **NotebookLM** to add each URL as a source, and mentioned several [Chrome extensions](https://chromewebstore.google.com/search/notebooklm) are available for this purpose.
   - These include the [NotebookLM Web Importer](https://chromewebstore.google.com/detail/notebooklm-web-importer/ncjabfmpppgonojpohbfhfaahfpkgihc), [NotebookLM Toolbox](https://chromewebstore.google.com/detail/notebooklm-toolbox/nbpckfdmlndgcoaklokdbllhelmfhoal), and [NotebookLM YouTube Turbo](https://chromewebstore.google.com/detail/notebooklm-youtube-turbo/mjpdncbmeognfgjkcdnglfomkmnknpgk).



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy Batch Function Ponders Parallel Work**: A member asked whether **DSPy's batch function** can delegate parallel work to a **vllm backend** running two instances of the same LLM, referencing the parameters *num_threads*, *max_errors*, and *return_failed_examples*.
   - It was suggested to increase *num_sequences* or *pipeline_parallel_size* if enough VRAM is available, instead of using separate APIs.
- **VLLM Pipeline Parallel Balances Load**: With *pipeline parallel size* set to 2 in the **vllm setup**, a member confirmed that **vllm handles the load balancing**.
   - Benchmarking was encouraged to compare processing times against non-parallel approaches.
- **LM Subclass Proposed for Load Balancing**: A member proposed creating a subclass of **LM** for load balancing if two instances are on different nodes, as **DSPy** doesn't natively handle this.
   - Although a **proxy** could forward requests, solving it on the **vllm** side is considered the better approach.



---



## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **String Replacement Snippet Shined**: A member posted a PHP code snippet in `general-chat` for string replacement: `$cleanedString = str_replace(['!', '@', '#', '$', '%', '^'], '', "This is a test string!@#$%^");`.
   - The code removes special characters from a string.
- **Laptop's Leap Leads to Loss**: A member reported their laptop broke after *taking a tumble*.
   - They characterized the damage as *not pretty*.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad JIT taking 30min**: A user reported that the **tinygrad JIT** compiler took over 30 minutes to compile a 2-layer model.
   - The message did not contain any solutions.
- **Tinygrad Loss turns NaN**: A user inquired about the cause of **loss becoming NaN** (Not a Number) after the initial step (step 0) in tinygrad.
   - The message did not contain any solutions.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune Plans Audio Addition**: Members discussed high level plans to incorporate **audio modality** into **Torchtune** in the future.
   - There are no specific timelines or technical details available at this time, indicating it's in the early planning stages.
- **Torchtune Audio Support Still on the Horizon**: The team continues to indicate they are planning to add **audio modality support** to **Torchtune** in the future.
   - As of now, specific timelines and technical details remain unconfirmed, suggesting that this feature is still in the preliminary planning stages.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Link Request for LLM Agents (Berkeley MOOC)**: claire_csy requested a valid link to the **LLM Agents Berkeley MOOC** discussion, as the previous link had expired.
   - The user seeks access to the **MOOC lecture discussion** for further engagement.
- **Need Active Link for LLM Agents MOOC**: A user, claire_csy, reported that the existing link for the **LLM Agents MOOC lecture discussion** is expired.
   - The request highlights the need for an updated and accessible link for participants interested in the **Berkeley MOOC**.



---



## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **Diffusion LLMs Hype Sparking Debate**: A community member inquired about the hype around the launch of **Diffusion LLMs**, specifically the **Mercury model**, and whether it will replace **transformer-based models**.
   - They mentioned reading the white paper but found it difficult to understand, seeking insights from experts in the community.
- **LLaDA Model: Diffusion-based LLM Paradigm Shift**: A community member shared a link to [diffusionllm.net](https://diffusionllm.net/), explaining that **Large Language Diffusion Models** (**LLaDA**) represent a new paradigm in language model architecture.
   - They elucidate that, unlike traditional **autoregressive (AR) Transformers**, **LLaDA** uses a denoising diffusion process to generate text in a parallel, coarse-to-fine manner.



---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1347524353088163850)** (1144 messages🔥🔥🔥): 

> `Cursor vs Lmarena, Cursor 0.47, Claude 3.7, Grok struggles, vibe coding` 


- **Cursor Caught in One-Shot Showdown!**: Members compared [**Cursor**](https://www.cursor.sh/) and [**Lmarena**](https://lmarena.ai/) using a single prompt with **GPT-3.7**, with initial impressions favoring Cursor for output quality on the right, citing unreadable text generated by Lmarena.
   - Later analysis by others suggested **Lmarena** better respected the theme of the prompt, but the general consensus was that *both were dogshit*.
- **Cursor 0.47 Calms Coding Chaos!**: After updating, users reported that [**Cursor 0.47**](https://www.cursor.sh/) fixed **Sonnet 3.7** issues by following custom rules, complying with AI custom rules, and behaving better for code generation, particularly for a [welcome window to a VS Code fork](https://code.visualstudio.com/api/ux-guidelines/welcome)
   - They noted the use of sequential thinking with multiple code paths for increased speed.
- **Vibe Coding Visited and Validated**: A discussion about *vibe coding*, building micro-SaaS web apps, led to a user building a welcome window on a **VS Code fork** using **Claude** itself.
   - A member defined the practice as *treating the ai agent like a child that needs guidance*, contrasting it with building a dockerized open-source project requiring orchestration experience.
- **MCP Magic: Maximizing Model Power**: Members explored using **Model Context Protocol (MCP)** servers to enhance AI code editors like Cursor and Claude, connecting to services like Snowflake databases.
   - One member found that **PearAI** offered full context, while another discovered that `.cursorrules` tend to be ignored in Cursor versions above **0.45**.
- **Error Message Becomes Swag Sensation**: The infamous **"Cursor is damaged and can't be opened"** error message has inspired merchandise, with [T-shirts](https://www.redbubble.com/shop/ap/169071328) and [mousepads](https://www.redbubble.com/i/mouse-pad/Cursor-AI-Error-by-TheGalaxyStars/169071328.G1FH6) now available.
   - This comedic turn underscores the community's ability to find humor in the face of technical frustrations.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://container-seven-sigma.vercel.app">Container Ejection Simulator</a>: no description found</li><li><a href="https://protocraft.ai">Protocraft AI</a>: Protocraft: AI Digital Studio designed for software development, task automation, creative exploration, and prompt automation, powered by your own API keys and local LLMs.</li><li><a href="https://anysphere-binaries.s3.us-east-1.amazonaws.com/production/client/linux/x64/appimage/Cursor-0.47.0-c2804e658d8fe4c072e20cb39c56d7eed1b6f43e.deb.glibc2.25-x86_64.AppImage">no title found</a>: no description found</li><li><a href="https://artificialanalysis.ai">AI Model &amp; API Providers Analysis | Artificial Analysis</a>: Comparison and analysis of AI models and API hosting providers. Independent benchmarks across key performance metrics including quality, price, output speed &amp; latency.</li><li><a href="https://techcrunch.com/2025/03/06/chatgpt-on-macos-can-now-directly-edit-code/">ChatGPT on macOS can now directly edit code | TechCrunch</a>: ChatGPT, OpenAI&#039;s AI-powered chatbot platform, can now directly edit code — if you&#039;re on macOS, that is.</li><li><a href="https://trypear.ai">PearAI - The AI Code Editor For Your Next Project</a>: PearAI is an open source AI code editor with powerful features like AI chat, PearAI Creator, and AI debugging to help you make what excites.</li><li><a href="https://www.youtube.com/watch?v=yOKwK-iIg3M"> - YouTube</a>: no description found</li><li><a href="https://x.com/mishalaskin/status/1898048925157728601?s=46">Tweet from Misha Laskin (@MishaLaskin)</a>: Today I’m launching @reflection_ai with my friend and co-founder @real_ioannis.Our team pioneered major advances in RL and LLMs, including AlphaGo and Gemini.At Reflection, we&#39;re building superint...</li><li><a href="https://devproai.com/index.htm">DevProAI</a>: no description found</li><li><a href="https://www.continue.dev">Continue</a>: Amplified developers, AI-enhanced development · The leading open-source AI code assistant. You can connect any models and any context to build custom autocomplete and chat experiences inside the IDE</li><li><a href="https://x.com/rohanpaul_ai/status/1878063926866493675">Tweet from Rohan Paul (@rohanpaul_ai)</a>: 🔥 OpenAI cut off a developer who weaponized ChatGPT&#39;s APIThis developer built this project which could respond to voice commands using ChatGPT&#39;s Realtime API.OpenAI confirmed the shutdown, ci...</li><li><a href="https://tenor.com/view/rick-and-morty-i-can-answer-that-for-money-gif-10573903">Rick And Morty I Can Answer That For Money GIF - Rick And Morty I Can Answer That For Money - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/oslook/cursor-ai-downloads">GitHub - oslook/cursor-ai-downloads: All Cursor AI&#39;s official download links for both the latest and older versions, making it easy for you to update, downgrade, and choose any version. 🚀</a>: All Cursor AI&#39;s official download links for both the latest and older versions, making it easy for you to update, downgrade, and choose any version. 🚀 - oslook/cursor-ai-downloads</li><li><a href="https://www.redbubble.com/shop/ap/169071328">Redbubble logo</a>: no description found</li><li><a href="https://www.redbubble.com/shop/ap/169071257">Redbubble logo</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1347532901914513458)** (254 messages🔥🔥): 

> `GPU memory management for training large models, ktransformers IQ1 benchmarks, QwQ-32B optimizations and best practices, GRPO algorithm optimizations` 


- **Multi-GPU Scaling for Model Deployment Examined**: Members discussed the practicality of deploying large models across multiple nodes, noting that pushing computation to only **1-2 nodes** is inefficient and overallocating compute without saturation leads to underutilization.
   - The conversation touched on **Google's** use of large **MoE** models and **TPUs** with low VRAM, as well as speculation on the large multinode deployments of **OpenAI** and **Anthropic**.
- **ktransformers Claims IQ1 beats BF16**: Members shared **Deepseek R1 IQ1** benchmarks by ktransformers, but there was skepticism because one member said *they would've wrote 1.58bit tho*.
   - Another member chimed in that the table was incomplete and the benchmarking was still in progress, showing **Deepseek v3 (the chat model) vs R1 IQ1**.
- **Unsloth's Offloading Algorithm Exposed**: Members discussed how **Unsloth** achieves low VRAM usage during training, hypothesizing that it involves asynchronously offloading gradients layer by layer to the CPU while the GPU calculates.
   - The actual saving appears to stem from re-implementing GRPO math, as well as gradient accumulation of intermediate layers, coupled with gradient checkpointing, and more efficient kernels like **logsumexp**.
- **QwQ-32B Generation Fixes and Optimizations**: Daniel Han posted a guide on debugging looping issues with **QwQ-32B** models, recommending adding samplers to llama.cpp such as *--samplers "top_k;top_p;min_p;temperature;dry;typ_p;xtc"* and uploaded [dynamic 4bit quants & GGUFs](https://huggingface.co/unsloth/QwQ-32B-unsloth-bnb-4bit).
   - The official recommended settings are *temperature = 0.6, top-k = 40, min-p = 0.1, top-p = 0.95*, but also mentioned that **QwQ is also sensitive to quantization** - *the first and last few layers should be left unquantized*.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://unsloth.ai/blog/long-context">Unsloth Gradient Checkpointing - 4x longer context windows</a>: Unsloth Gradient Checkpointing now supports finetuning of LLMs with very long context windows, up to 228K for Llama 3.We managed to reduce memory usage by a further 30% at the cost of +1.9% extra time...</li><li><a href="https://unsloth.ai/blog/grpo">Long-context GRPO (R1 Reasoning)</a>: DeepSeek R-1 is the most powerful open-source reasoning model that performs on par with OpenAI&#x27;s o1 model.Run the 1.58-bit Dynamic GGUF version by Unsloth.</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1j5qo7q/qwq32b_infinite_generations_fixes_best_practice">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/fp8_kernel.md">ktransformers/doc/en/fp8_kernel.md at main · kvcache-ai/ktransformers</a>: A Flexible Framework for Experiencing Cutting-edge LLM Inference Optimizations - kvcache-ai/ktransformers</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1j5qo7q/qwq32b_infinite_generations_fixes_best_practices/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://x.com/danielhanchen/status/1898035752124166368">Tweet from Daniel Han (@danielhanchen)</a>: Having endless repetitions with QwQ-32B? I made a guide to help debug stuff!When using repetition penalties to counteract looping, it rather causes looping!Try adding this to llama.cpp:--samplers &#34...</li><li><a href="https://docs.unsloth.ai">Welcome | Unsloth Documentation</a>: New to Unsloth?</li><li><a href="https://github.com/unslothai/unsloth-zoo/blob/c54bf68b71abcd45e49cd077ea4b71dabe3ae6fa/unsloth_zoo/rl_replacements.py#L52-L237">unsloth-zoo/unsloth_zoo/rl_replacements.py at c54bf68b71abcd45e49cd077ea4b71dabe3ae6fa · unslothai/unsloth-zoo</a>: Utils for Unsloth. Contribute to unslothai/unsloth-zoo development by creating an account on GitHub.</li><li><a href="https://github.com/unslothai/unsloth-zoo/blob/c54bf68b71abcd45e49cd077ea4b71dabe3ae6fa/unsloth_zoo/rl_replacements.py#L52-L92">unsloth-zoo/unsloth_zoo/rl_replacements.py at c54bf68b71abcd45e49cd077ea4b71dabe3ae6fa · unslothai/unsloth-zoo</a>: Utils for Unsloth. Contribute to unslothai/unsloth-zoo development by creating an account on GitHub.</li><li><a href="https://artificialanalysis.ai/models/qwq-32b">QwQ-32B - Intelligence, Performance &amp; Price Analysis | Artificial Analysis</a>: Analysis of Alibaba&#x27;s QwQ 32B and comparison to other AI models across key metrics including quality, price, performance (tokens per second &amp; time to first token), context window &amp; more.</li><li><a href="https://github.com/unslothai/unsloth-zoo/blob/c54bf68b71abcd45e49cd077ea4b71dabe3ae6fa/unsloth_zoo/gradient_checkpointing.py#L137-L165">unsloth-zoo/unsloth_zoo/gradient_checkpointing.py at c54bf68b71abcd45e49cd077ea4b71dabe3ae6fa · unslothai/unsloth-zoo</a>: Utils for Unsloth. Contribute to unslothai/unsloth-zoo development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1347534418738286623)** (82 messages🔥🔥): 

> `RLHF with Unsloth GRPO on Qwen7b, Qualitative vs Quantitative Improvement, Reward Model Bias, KL Divergence Issues, Qwen for Sudoku` 


- **RLHF on Qwen7b Yields Roleplay Success But Benchmark Failure**: A member reported a successful **RLHF** run using **Unsloth GRPO** on a **Qwen7b** model, noting significant improvements in role adherence and output smoothness.
   - However, the model showed *noticeable degradation* in strict instruction-following benchmarks like **IFeval**, particularly with formatting constraints and negative instructions, and the notebook was requested.
- **Dataset Gaps and Sugary Rewards Hurt Benchmark Performance**: The poster diagnosed that the training data lacked examples demanding precise compliance, and the reward model favored *overly detailed* responses, which led to issues.
   - They said that *while beneficial for 'friendly' interactions, this becomes problematic in contexts demanding concise, exact compliance*.
- **KL Divergence Instability Troubles GRPO Training**: Another member shared plots showing *peaky* **KL divergence** during their **GRPO** training, questioning if something was wrong, and posted their hyperparameters for feedback.
   - It was recommended to switch the learning rate scheduler to *constant*, and remove both **weight decay** and **warmup ratio** to stabilize the training process and observe immediate training effects, and also agressively clip gradients.
- **Qwen Excels at Sudoku Despite Lazy Reasoning**: One member found that **Qwen** models can solve **Sudoku** puzzles accurately despite exhibiting *dumb and lazy* reasoning.
   - While **Qwen** followed instructions better than other models, they weren't necessarily *smarter*; however, they still exhibited some errors.
- **Leaderboard Model Pulled After Questionable Behavior**: A member noticed that a **14B model** at the top of a leaderboard was taken down, and the author requested its removal.
   - It was not further elaborated what the situation was.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1347535211918917745)** (114 messages🔥🔥): 

> `RAM Configuration for Mac Studio, ktransformers Performance, RoPE Scaling, Custom Datasets, Multi-GPU Parallelism with Unsloth` 


- **RAM Limits on Mac Studio**: A user questioned whether a **1.58bit quantized model of deepseek-r1**, which is **131GB** in size, could fit on a **128GB Mac Studio** without disk swapping.
   - Another user running **ktransformers** on a **64-core Threadripper** reported allocating about **12GB of VRAM** and observed that it didn't even use **22GB**.
- **RoPE Scaling Extends Context Length**: It was mentioned that the **Qwen2.5-Coder-3B-bnb-4bit** model can handle a maximum sequence length of **32768**, but with **kaiokendev's RoPE scaling of 3.906**, it can be extended to **128000**.
   - It was confirmed that [with RoPE](link-to-rope-explanation), as long as the architecture is transformers, the 128k tokens can be attended to.
- **Homework Help request**: A user urgently needed help with custom datasets and hyperparameter tuning for a **Phi4 model** due to a homework deadline in 3 hours.
   - Another member responded *that if you need it in 3h "FOR A HOMEWORK" you knew about it for some time ..so its on you if you failthats not how you get help*.
- **Unsloth Enables Multi-GPU Training**: A user inquired about the implementation of multi-GPU parallelism in Unsloth, referencing [this issue](https://github.com/unslothai/unsloth/issues/1908).
   - A member mentioned that the code has to be run with **torchrun**.
- **GRPO Training Requires Special Setup**: A user reported issues running **GRPO training** on a server with **4 GPUs**, noting multiple subprocesses created on each GPU, despite attempts to adjust the device map.
   - A member clarified that **GRPO does not work with the specified parameters and currently only works with regular training**.



**Link mentioned**: <a href="https://github.com/unslothai/unsloth/issues/1908)">unslothai/unsloth</a>: Finetune Llama 3.3, DeepSeek-R1 &amp; Reasoning LLMs 2x faster with 70% less memory! 🦥 - unslothai/unsloth

  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1347534034124935233)** (6 messages): 

> `Diffusion Effect, Rust Code, Deepseek Coder v2, Unsloth and MoE` 


- **Diffusion Effect Creates Strong Impression**: A member enjoyed using a model with the **diffusion effect**, praising its performance.
   - They added that it's also pretty good at **Rust code**.
- **Unsloth Integrates Deepseek Coder v2**: A user confirmed that **Unsloth** has integrated **Deepseek Coder v2**.
- **Unsloth struggles with MoE**: A member pointed out that **Unsloth** can't train **MoE** models at the moment.


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1347524188436693063)** (264 messages🔥🔥): 

> `Registry Editing Risks, Quantization impact on RAM and VRAM, File path limitations on windows, Trustless authentication, Diffusion-based language models` 


- **Registry Editing leads to Blue Screen**: A member recounted an experience where deleting a crucial **.dll file** after identifying it as a RAM hog led to a **blue screen** on the next boot, emphasizing the risks of editing the registry without backups.
   - Members generally agree that it's wise to *backup personal files and reformat* after tweaking the registry.
- **Quantization impacts performance and memory**: Users discussed quantization, with one member noting that they prefer **f16 quantization** to fit more parameters in a smaller load, while acknowledging that other quantizations may cause crashes with flash attention.
   - In the context of quantization, someone described floating points as its *signed bit size so signed 16 int is 32*.
- **Windows file path limited to 260 chars**: Members discussed the limitations of **file path lengths in Windows**, noting the standard **260-character limit** due to the **MAX_PATH** definition in the Windows API, as explained in [Microsoft documentation](https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation?tabs=registry).
   - This limitation can be bypassed by enabling long path behavior per application via registry and application manifest modifications for NT kernel path limit of 32,767.
- **Inception Labs pioneers Diffusion-Based Text**: [Inception Labs](https://inceptionlabs.ai/) is pioneering diffusion-based language generation which promises unprecedented speed, quality, and generative control inspired by AI systems like **Midjourney** and **Sora**.
   - Members noted that open source alternatives are in development, such as [ML-GSAI/LLaDA](https://github.com/ML-GSAI/LLaDA), and that significant speed increases may be on the horizon with this technology.
- **Trustless authentication model discussed**: Members discussed the potential of **trustless authentication**, where a 3D scan of a face and ID is converted into an encrypted string to serve as a digital passport, similar to the business model used by [Persona](https://www.withpersona.com/)
   - It's envisioned as a trustless verification database with users' personalization tags in the file, where generated deepfake facial scans and IDs would not work.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/multimodalart/LLaDA">LLaDA - a Hugging Face Space by multimodalart</a>: no description found</li><li><a href="https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation?tabs=registry">Maximum Path Length Limitation - Win32 apps</a>: Starting in Windows 10, version 1607, MAX_PATH limitations have been removed from many common Win32 file and directory functions. However, your app must opt-in to support the new behavior.</li><li><a href="https://github.com/nomic-ai/gpt4all/wiki/Frequently-Asked-Questions#where-are-the-default-directories-for-models-and-settings.">Frequently Asked Questions</a>: GPT4All: Run Local LLMs on Any Device. Open-source and available for commercial use. - nomic-ai/gpt4all</li><li><a href="https://github.com/ML-GSAI/LLaDA">GitHub - ML-GSAI/LLaDA: Official PyTorch implementation for &quot;Large Language Diffusion Models&quot;</a>: Official PyTorch implementation for &quot;Large Language Diffusion Models&quot; - ML-GSAI/LLaDA</li><li><a href="https://inceptionlabs.ai/">Inception Labs</a>: We are leveraging diffusion technology to develop a new generation of LLMs. Our dLLMs are much faster and more efficient than traditional auto-regressive LLMs. And diffusion models are more accurate, ...</li><li><a href="https://tenor.com/view/fun-cave-men-old-kick-fuck-you-gif-13869846">Fun Cave Men GIF - Fun Cave Men Old - Discover &amp; Share GIFs</a>: Click to view the GIF
</li>
</ul>

</div>
  

---


### **Codeium (Windsurf) ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1347583701403439104)** (7 messages): 

> `IDE Telemetry Settings, Codeium Website Payment Updates` 


- **Telemetry Triumphs Troubles**: A member reported that chat was disabled due to **IDE Telemetry** settings in Visual Studio Code and Jetbrains IDE.
   - Another member suggested enabling **code telemetry** and setting telemetry settings to **on** in VS Code and pointed to [this reddit thread](https://www.reddit.com/r/Codeium/comments/1f4ljqf/unable_to_use_chat_ide_telemetry/) for detailed instructions.
- **Codeium Collection Complications on Website**: A member inquired whether anyone else is having trouble updating **payment information** on the Codeium website.


  

---


### **Codeium (Windsurf) ▷ #[windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1347524326290886697)** (238 messages🔥🔥): 

> `Windsurf stability issues, Credit consumption, Cascade problems, Model performance comparison (Cursor vs. Windsurf), MCP server issues` 


- **Windsurf Plagued with Stability Surfing Setbacks**: Users report Windsurf is **unstable** after the latest updates, with issues like **infinite loops, repeating code changes**, and general **unresponsiveness**.
   - Many users are switching to **Cursor** or **Trae**, citing better stability and resource management, with one user stating, *"I have decided to switch to Cursor, which I have never used...Presumably more stable. I will come back in a month or so, see how it's doing."
- **Credits Crunch: Windsurf's Costly Consumption Catches Critics**: Users are experiencing **high credit consumption** due to the AI repeatedly rewriting code or analyzing the same files, leading to frustration.
   - One user complained about burning through **1200 flow credits** in one day after the introduction of **Claude 3.7** and called it *"100% unstable and 100% u pay for nothing"*, suggesting the tool reads only 50 lines at a time and takes 2 credits for reading a 81 line file.
- **Cascade Chaos: Users Report Terminal Troubles**: Some users report the **Cascade terminal** disappearing or becoming unresponsive, with no visible settings to fix it.
   - One user mentioned a temporary solution of restarting, but the issue reoccurs, while another suggested to use `CTRL Shift P` and clear all cache and reload windows.
- **Model Melee: Cursor Crushes Codeium in Credit Contest**: Users are comparing Windsurf to **Cursor**, with many finding Cursor to be more efficient and stable, especially when handling large files.
   - One user says *"Cursor crushed a 3k-line file like a boss"* and reported that credit usage on Cursor is *"a fraction of windsurf with claude 3.7"* while another reports *"Trae is doing x100 better than ws while its free"*.
- **MCP Mayhem: Users Struggle with Server Setup**: Users are encountering issues while trying to use **MCP servers**, with errors related to the model or configuration.
   - One user received a *"Not Found: Resource not found"* error and tried different models without success.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/x1xhlol/v0-system-prompts-models-and-tools">GitHub - x1xhlol/v0-system-prompts-models-and-tools</a>: Contribute to x1xhlol/v0-system-prompts-models-and-tools development by creating an account on GitHub.</li><li><a href="https://codeium.com/support">Support | Windsurf Editor and Codeium extensions</a>: Need help? Contact our support team for personalized assistance.</li><li><a href="https://tenor.com/view/your-technique-is-out-of-this-world-bruno-tonioli-britains-got-talent-your-technique-is-incredible-your-technique-is-extraordinary-gif-864393750500369819">Your Technique Is Out Of This World Bruno Tonioli GIF - Your technique is out of this world Bruno tonioli Britains got talent - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.codeium.com/support">Support | Windsurf Editor and Codeium extensions</a>: Need help? Contact our support team for personalized assistance.</li><li><a href="https://x.com/OpenAIDevs/status/1897700857833193955">Tweet from OpenAI Developers (@OpenAIDevs)</a>: ChatGPT for macOS can now edit code directly in IDEs. Available to Plus, Pro, and Team users.</li><li><a href="https://x.com/i/communities/1889976002400686208">Tweet from GitHub - FixTweet/FxTwitter: Fix broken Twitter/X embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others</a>: Fix broken Twitter/X embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others - FixTweet/FxTwitter</li><li><a href="https://codeium.com/pricing">Pricing | Windsurf Editor and Codeium extensions</a>: Codeium is free forever for individuals. Teams can level up with our enterprise offering for enhanced personalization and flexible deployments.</li><li><a href="https://x.com/openai/status/1897702764471619657?s=46&t=B0TlaMZ0ShmwM-XdEqw2mg">Tweet from OpenAI (@OpenAI)</a>: And coming soon to Free, Enterprise, and Edu users 🪐Quoting OpenAI Developers (@OpenAIDevs) ChatGPT for macOS can now edit code directly in IDEs. Available to Plus, Pro, and Team users.
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1347528332840796181)** (184 messages🔥🔥): 

> `Perplexity Pro Account Issues, GPT-4.5 Usage, Commercial Use of Perplexity and Copyright, Sonnet 3.7 Extended Performance, Perplexity Mobile App and Claude` 


- **Perplexity Pro Perks Pulled, Users Pained**: Several users reported their **Perplexity Pro** accounts were unexpectedly cancelled, with some suspecting a scam.
   - Perplexity support indicated that the cancellations were due to users not being eligible for the "DT 1 Year Free HR" offer, meant exclusively for **Deutsche Telekom** customers in **Croatia**, linking to the [terms of service](https://www.perplexity.ai/legal/terms).
- **GPT-4.5 Free Tier, Fact or Fiction?**: Users discussed the availability of **GPT-4.5** with **Perplexity Pro**, with one user asking if they should have **10 free uses**.
   - Another user confirmed the **10 uses per 24 hours**, clarifying that it is super expensive, and another user said that Auto model selection will be good enough to avoid having to decide what model to pick.
- **Copyright Conundrums for Commercial Perplexity Pioneers**: A user inquired about the copyright implications of using Perplexity in commercial web apps, given that Perplexity scrapes information from copyrighted sources and websites with 'no data mining' policies.
   - The discussion did not yield a conclusive answer regarding liability for copyright infringement when using the **Perplexity API**.
- **Complexity Extension Creates Canvas Capabilities**: A user explained how to generate mermaid diagrams within Perplexity using the **Complexity extension**, which provides canvas-like features.
   - By enabling the canvas plug-in and prompting the AI to create a mermaid diagram, users can then render it by pressing the play button on the code block, there's also a [discord link](https://discord.com/channels/1245377426331144304/1246406910962438245/1347554810928566304) with more info
- **Google Gemini 2.0 grabs ground in Google Search**: A user shared an [article from ArsTechnica](https://arstechnica.com/google/2025/03/google-is-expanding-ai-overviews-and-testing-ai-only-search-results/) detailing **Google's** expansion of **AI search features** powered by **Gemini 2.0**.
   - Google is testing an **AI Mode** that replaces the traditional '10 blue links' with **Gemini** generated results.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.thefastmode.com/technology-solutions/39682-hrvatski-telecom-offers-20-000-free-licenses-for-perplexity-pro">Hrvatski Telekom Dispenses 20,000 Free Licenses for Perplexity Pro</a>: Croatian Telecom Offers 20,000 Free Licenses for Perplexity Pro, the Advanced AI Assistant</li><li><a href="https://arstechnica.com/google/2025/03/google-is-expanding-ai-overviews-and-testing-ai-only-search-results/">You knew it was coming: Google begins testing AI&#x2d;only search results</a>: AI Mode could be the future of Google, but it&rsquo;s currently just an experiment.</li><li><a href="https://www.thefastmode.com/technology-solutions/39682-hrvatski-telecom-offers-20-000-free-licenses-">Hrvatski Telekom Dispenses 20,000 Free Licenses for Perplexity Pro</a>: Croatian Telecom Offers 20,000 Free Licenses for Perplexity Pro, the Advanced AI Assistant</li><li><a href="https://tenor.com/view/stonks-up-stongs-meme-stocks-gif-15715298">Stonks Up Stongs GIF - Stonks Up Stongs Meme - Discover &amp; Share GIFs</a>: Click to view the GIF
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1347583826913788039)** (5 messages): 

> `Apple Foldable iPhone, OpenAI AI Agent, Amazon Prime AI Dubbing, DuckDuckGo AI Search` 


- **Apple's Foldable iPhone Prediction Drops**: A [Perplexity page](https://www.perplexity.ai/page/apple-s-foldable-iphone-predic-WSdZuoG7Rw6VvayJJg0DVQ) discusses predictions around **Apple's foldable iPhone**.
- **OpenAI's 20000 AI Agent**: A [Perplexity page](https://www.perplexity.ai/page/openai-s-20000-ai-agent-nvz8rzw7TZ.ECGL9usO2YQ) talks about **OpenAI's AI Agent** with a mention of 20000.
   - No other details are available.
- **Amazon Prime Tests AI Dubbing**: A [Perplexity page](https://www.perplexity.ai/page/amazon-prime-tests-ai-dubbing-pHEI1t6XRn6DilTOLGBGew) covers how **Amazon Prime** is testing **AI dubbing**.
- **DuckDuckGo's AI Search Option**: A [Perplexity page](https://www.perplexity.ai/page/duckduckgo-s-ai-search-option-D2sL.5w8S4mQYdr_XAlgjw) shares info on **DuckDuckGo's AI Search** option.


  

---


### **LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1347749343087689822)** (1 messages): 

> `LM Studio 0.3.12, QwQ template bug fixes, RAG chunking speed improvement` 


- **LM Studio v0.3.12 is out!**: LM Studio 0.3.12 is now available as a stable release with bug fixes and performance improvements. Check out the [full release notes](https://lmstudio.ai/blog/lmstudio-v0.3.12).
   - You can upgrade via in-app update, or from [LM Studio's website](https://lmstudio.ai/download).
- **QwQ template parsing bug squashed**: The latest LM Studio release fixed a **QwQ 32B jinja parsing bug** that threw an `OpenSquareBracket !== CloseStatement` error, so no more weird errors.
   - It also addressed a bug where chats could not be deleted due to attached files not being found.
- **RAG chunking speed gets a boost**: The speed of chunking files for retrieval in **RAG** has been significantly increased in the newest release.
   - MLX models downloaded onto an external exFAT drive are now correctly indexed on MacOS.



**Link mentioned**: <a href="https://lmstudio.ai/blog/lmstudio-v0.3.12">LM Studio 0.3.12</a>: Bug fixes and document chunking speed improvements for RAG

  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1347546479564165284)** (104 messages🔥🔥): 

> `Open Source LLM for Coding on M2 Macbook Pro, DeepSeek v2.5 1210, Qwen Coder, Finetuning Large Language Models, Context Length and Memory Management` 


- **Qwen Coder recommended for Macbook M2 coding**: For coding tasks on a Macbook M2 Pro with 16GB RAM, **Qwen Coder** is suggested, though its performance will be significantly lower than cloud-based models like Claude or DeepSeek.
   - The **Deepseek v2.5** was recommended at first, but members agreed that 16 GB will not give you much room to run it. A smaller **7B** model might work, but the quality might not be satisfactory.
- **Unsloth offers Finetuning for LLMs**: Members discussed finetuning LLMs, where [Unsloth](https://github.com/unslothai/unsloth) was recommended for faster finetuning of models like Llama-3, Mistral, Phi-4, and Gemma with less memory usage.
   - It was mentioned that *finetuning is much more resource intensive than inference* and LM Studio currently does not have a public roadmap that includes this feature.
- **Packing & Unpacking Token Context Idea Floated**: A member proposed a method to manage context length by packing and unpacking text in VRAM, reducing its size to about 30% for efficient browsing and retrieval of relevant information.
   - However, others clarified that context is stored as **tokens** (averaging 1/2 a word) rather than plain text, so this **compression** logic wouldn't provide significant benefits, suggesting **summarization** or **RAG** may be better alternatives.
- **Draft Models Boost Mistral's Token Uplift**: Using a **i1-IQ1_S** quant of the same model as a draft model, such as using **Q8_0 of mistral_small** and **i1-IQ1_S** as the draft model, can result in a significant **token uplift**.
   - Members reported speed improvements, going from 18 to 30 t/s using 2x 3090s (48GB VRAM), with one reporting 83% accepted token rate; however, some users have experienced no improvement or even a downgrade in speed.
- **48 GB Enough for New Macbook?**: Members discussed RAM options for a new Macbook, with some suggesting **128GB** for running larger models like **DeepSeek V2.5** or **Mistral Large** at low quantizations.
   - A member noted **Qwen 32B** punches above its weight and at least at Llama 3.3 70b. Also it was mentioned that models might not come down in size without major drawbacks in 2025.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.unsloth.ai/">Welcome | Unsloth Documentation</a>: New to Unsloth?</li><li><a href="https://dubesor.de/benchtable">Dubesor LLM Benchmark table</a>: no description found</li><li><a href="https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/479#issuecomment-2701947624">Issue with qwq-32b model in lmstudio. · Issue #479 · lmstudio-ai/lmstudio-bug-tracker</a>: Which version of LM Studio? Example: LM Studio 0.3.11 Which operating system? Mac What is the bug? I get following error when chatting with qwq-32b model &quot;Error rendering prompt with jinja templa...
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1347525945573249086)** (80 messages🔥🔥): 

> `9070XT vs 7900XTX, ROCm and Vulkan performance, Native FP4 support, CodeGPT extension issues on WSL, Quantization impact on model quality` 


- **9070XT outperforms 7900XTX**: A user compared the **9070XT** to the **7900XTX** running the same **qwen2.5 coder 14b q8_0** model and found that the **9070XT** runs about **44tok/sec** with under a second to first token, while the **7900XTX** runs **31tok/sec** with **4sec** to first token.
   - Another user noted that the **9070** gets **400 t/s** with **0.5B** models, while the **7900xtx** gets between **300 and 360**, suggesting it's at least **10%** better, despite using **Vulkan** instead of **ROCm**.
- **Vulkan vs ROCm on 9070XT**: The **9070XT** doesn't have **ROCm** support on Windows yet, so **LM Studio** uses **Vulkan**, but some users report that **Vulkan** inference is sometimes faster than **ROCm** by a few percent, although this is disputed.
   - One user believes those seeing better **Vulkan** performance are *having driver issues* because *rocm should be faster than vulkan, by far*.
- **Native FP4 support now available**: Members discussed that the **9070** has significantly improved **FP16** and **INT4/FP4** performance compared to older cards.
   - Specifically, the **9070** has **122 vs 389 FP16** and **122 vs 1557 INT4/FP4**, signaling that native **FP4** support is available from both **Nvidia** and **Radeon**.
- **Troubleshooting CodeGPT Extension on WSL**: A user is experiencing a *fetch failed status 500* error when trying to access a local model via the **CodeGPT** extension in **VSCode** on **WSL**.
   - Suggested solutions include ensuring the server is set to serve on the local network using the local network IP, and turning on **CORS** (Cross-Origin Resource Sharing).
- **Impact of Quantization on Model Quality**: Users discussed the impact of quantization on model quality, particularly the trade-offs between smaller quant sizes and potential quality loss.
   - One member notes that **Q5_K_M** is closer to **Q6_K** accuracy than **Q4_K_M**, and **Q6** should be safe, while another suggests using a website stalker to monitor availability of the **9070 XT**.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1347538735272689694)** (118 messages🔥🔥): 

> `Open Source Alternatives to Replit/Bolt, Gradio Dexie Wrapper Proposal, Obsidian user is from Obsidian, Suspecting Dataset Misuse in Research Papers, Hugging Face Datasets and DOI Generation` 


- **Agent Alternatives to Replit and Bolt Debated**: A member inquired about open-source models with similar agent functionality to **Replit/Bolt**, sparking discussion on potential alternatives.
   - Additionally, the thread considered alternatives to *export_to_video* for direct video saving to a **Supabase bucket**.
- **Gradio IndexedDB Access via Dexie: Good Feature?**: A user proposed integrating a **Dexie wrapper** for easy IndexedDB access in Gradio apps.
   - The suggestion prompted discussion on its feasibility as a custom component, with a link to the **Gradio developers Discord channel** shared for further inquiry.
- **Dataset Citation Drama: A Preprint Predicament**: A user suspected a research paper, '[NotaGen: Symbolic Music Generation with LLM Training Paradigms](https://arxiv.org/abs/2502.18008)', of using their **Toast MIDI dataset** without proper citation.
   - Community members advised contacting the corresponding author and generating a **DOI** for the dataset to ensure proper attribution.
- **Dataset Citation Expedition: Proper BibTeX Formatting**: A user was guided on how to properly format **BibTeX citations** for their datasets on Hugging Face, including wrapping the citation in triple backticks and ensuring the URL is correctly formatted with the `\url{}` tag.
   - It was emphasized that correct formatting will ensure proper recognition of the dataset by academic software and researcher social media profiles like Google Scholar.
- **Fine Tuning & Multi Lingual Models**: A user asked if fine-tuning a base model like **Mistral** on an English dataset for a specific task would transfer that knowledge to other languages.
   - Another user responded by stating that they *would assume so rt?*


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://obsidian.md/">Obsidian - Sharpen your thinking</a>: The free and flexible app for your private thoughts.</li><li><a href="https://huggingface.co/papers/2502.18008">Paper page - NotaGen: Advancing Musicality in Symbolic Music Generation with Large
  Language Model Training Paradigms</a>: no description found</li><li><a href="https://arxiv.org/abs/2502.18008">NotaGen: Advancing Musicality in Symbolic Music Generation with Large Language Model Training Paradigms</a>: We introduce NotaGen, a symbolic music generation model aiming to explore the potential of producing high-quality classical sheet music. Inspired by the success of Large Language Models (LLMs), NotaGe...</li><li><a href="https://huggingface.co/datasets/breadlicker45/bread-midi-dataset">breadlicker45/bread-midi-dataset · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/breadlicker45/youtube-comments-180k">breadlicker45/youtube-comments-180k · Datasets at Hugging Face</a>: no description found</li><li><a href="https://x.com/ClementDelangue/status/1897767808076816575?t=f0HsVgnlRua2PLTuPvIELQ&s=19">Tweet from clem 🤗 (@ClementDelangue)</a>: Who said open-sourcing is bad for your business?</li><li><a href="https://ieeexplore.ieee.org/abstract/document/10421308/">Exploring Embeddings for Measuring Text Relatedness: Unveiling Sentiments and Relationships in Online Comments</a>: After the COVID-19 pandemic caused internet usage to grow by 70%, there has been an increased number of people all across the world using social media. Applications like Twitter, Meta Threads, YouTube...</li><li><a href="https://huggingface.co/datasets/breadlicker45/toast-midi-dataset">breadlicker45/toast-midi-dataset · Datasets at Hugging Face</a>: no description found</li><li><a href="https://discuss.huggingface.co/t/runtimeerror-stack-expects-each-tensor-to-be-equal-size-but-got-12-at-entry-0-and-35-at-entry-1/46155/2">RuntimeError: stack expects each tensor to be equal size, but got [12] at entry 0 and [35] at entry 1</a>: I think the tokenized texts are not of the same length as indicated by this warning message.     If you adjust the input length to be the same at each batch, I think the error will go away.
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1347568249927372993)** (3 messages): 

> `HF Docker Repository, fxtwitter` 


- **HuggingFace Docker Metadata arrives**: A member noted that metadata is being fetched from the [HF Docker repository](https://huggingface.co/spaces/mozilla-ai/osm-ai-helper).
   - This suggests enhanced integration or information retrieval from **Docker containers** within the Hugging Face ecosystem.
- **fxtwitter gets the boot**: A member stated *you don't need to use **fxtwitter** anymore. embeds work fine 😄*.
   - This implies that **embedded links** to Twitter content are now functioning correctly on the platform, removing the need for the fxtwitter workaround.



**Link mentioned**: <a href="https://huggingface.co/spaces/mozilla-ai/osm-ai-helper">OpenStreetMap AI Helper - a Hugging Face Space by mozilla-ai</a>: no description found

  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1347683704306270250)** (1 messages): 

> `Downloads, Community Appreciation` 


- **Downloads Hit Milestone!**: A member expressed surprise and gratitude as their creation reached almost **1000 downloads in 10 days**.
   - They thanked the community for their support, celebrating the achievement with a shared image.
- **Community Cheers on Downloads Milestone**: The community celebrates as a member's creation reaches nearly **1000 downloads in just 10 days**, showcasing strong community engagement.
   - The attached image visually represents the achievement, further amplifying the excitement and appreciation within the community.


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1347677128551960678)** (1 messages): 

> `OCR-2.0 Guidance` 


- **Guidance requested on OCR-2.0 Fine-Tuning**: A member requested guidance on how to accomplish a task, linking to a [SharePoint document](https://bama365-my.sharepoint.com/:w:/g/personal/xgranja_ua_edu/EeSz8D6iYPxHhzfQD3GGzsYBARpsSkbEDZWzoQH7hIH4lg?e=gMOaR4).
   - The member found **OCR-2.0** to be the latest and best model and asked if they should fine-tune the **got/ocr-2.0** models according to the documentation.
- **OCR-2.0 Acclaimed as Cutting Edge**: A user identified **OCR-2.0** as the most advanced model currently available.
   - They are contemplating fine-tuning the **got/ocr-2.0** model to better suit their needs.


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1347537494346039357)** (5 messages): 

> `Smol Agents Course, Pokemon LLM Agent Benchmark, HuggingFace Token issues` 


- ****Smol Agents** Course Module Gets a Refresh**: The **Smol Agents** course team is actively writing a new MCP module and has updated the existing one to better utilize **smolagents**; [Pull Requests](https://github.com/CalebDeLeeuwMisfits/PokemonLLMAgentBenchmark) are encouraged.
- **Typo Error Causes **HuggingFace** Token Chaos!**: A new student encountered issues with their **HuggingFace token** in VSCode on a Mac, initially deemed invalid.
   - The problem was identified as a typo: *mistaking the letter 'O' for the number '0'*—the user also pondered whether they should be able to paste the token into VSCode.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/thinking-book-penguin-student-writing-gif-5543983515725736276">Thinking Book GIF - Thinking Book Penguin - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/CalebDeLeeuwMisfits/PokemonLLMAgentBenchmark">GitHub - CalebDeLeeuwMisfits/PokemonLLMAgentBenchmark</a>: Contribute to CalebDeLeeuwMisfits/PokemonLLMAgentBenchmark development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1347525730979811378)** (37 messages🔥): 

> `Course Start Dates, LLM as Agent Component, RAG as Environment, Course Completion Status, Image Generation Troubles` 


- **LLM's Role: Agent or Just a Part?**: A member clarified that an **LLM** itself isn't an **Agent**, but a component equipped with the capability to take instructions then call tools in order to provide more functionalities.
   - They added that an **agentic AI model** requires an **LLM** and an **agent**, as detailed [in this helpful response](https://discord.com/channels/879548961488179230/1201995434137206834/1208022892900941824).
- **RAG System: Environment or Tool?**: A member inquired whether a **Retrieval Augmented Generation (RAG)** system interacting with an **LLM** could be considered an "environment" that the **LLM** interacts with via the **RAG** system as a "tool".
   - Another member suggested that, technically, the decoder layer evaluations that determine the best response could be considered part of the **LLM's** environment, even with tools like 'stories'.
- **Mint a partner NFT with OpenSea?**: A message announced a partnership with **OpenSea** for a new free mint.
   - Users were invited to join and potentially **claim** an **NFT**, noting that some claims may require gas; the link was [opensea-nftbox5-one.vercel.app](https://opensea-nftbox5-one.vercel.app/).
- **Image Generation Woes with FLUX.1**: A course participant shared a code snippet attempting to generate images using **Stable Diffusion FLUX.1**, expressing difficulty in figuring out what was wrong.
   - They sought assistance with a function utilizing the `FluxPipeline` from the `black-forest-labs/FLUX.1-dev` model, setting various parameters such as `height`, `width`, `guidance_scale`, and `num_inference_steps`.


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1347532757714079785)** (144 messages🔥🔥): 

> `Perplexity API copyright issues, OpenRouter latency with Anthropic API, Groq provider in OpenRouter, Gemini embedding model, Testing reasoning parameter in OpenRouter models` 


- **Perplexity API Copyright Indemnification**: Users are discussing potential copyright issues with using the **Perplexity API** due to its scraping of copyrighted content, noting that [Perplexity's API terms](https://www.perplexity.ai/hub/legal/perplexity-api-terms-of-service) require customers to indemnify Perplexity, shifting liability onto the user.
   - Alternatives with IP indemnification, like **OpenAI**, **Google Cloud**, and **AWS Bedrock**, were mentioned with links to their respective terms, advising users to assess legal risks.
- **Sonar Deep Research models experiencing errors**: Members reported **Perplexity Sonar Deep Research** model experiencing frequent errors, high latency (up to **241 seconds** for the first token), and unexpectedly high reasoning token counts.
   - One member humorously noted a **137k reasoning token** count with no output, while another confirmed it eventually started working after initial issues.
- **New Experimental Gemini Embedding Text Model unveiled**: A new experimental **Gemini Embedding Text Model** (**gemini-embedding-exp-03-07**) is available in the [Gemini API](https://ai.google.dev/gemini-api/docs/models/experimental-models), surpassing the previous state-of-the-art model.
   - This model achieves the top rank on the **Massive Text Embedding Benchmark (MTEB) Multilingual leaderboard** and includes new features like longer input token length.
- **OpenRouter's reasoning parameter has inconsistencies**: Users discovered inconsistencies in OpenRouter's reasoning parameter, with some models marked as supporting reasoning despite endpoints lacking support and some providers not returning reasoning outputs.
   - Members are conducting tests, discovering configuration issues, and identifying discrepancies between the models and endpoints, with Cloudflare noted to lack a **/completions endpoint**.
- **Model struggles with Russian language prompts**: A user reported **Claude 3.7** struggling with **Russian language prompts**, responding in English and potentially misunderstanding the nuances of the language.
   - This was observed while using cline with OpenRouter, suggesting the issue might be with Anthropic rather than the extension or OpenRouter itself.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openrouter.ai/docs/use-cases/for-providers">Provider Integration - Add Your Models to OpenRouter</a>: Learn how to integrate your AI models with OpenRouter. Complete guide for providers to make their models available through OpenRouter&#x27;s unified API.</li><li><a href="https://ai.google.dev/gemini-api/docs/models/experimental-models">no title found</a>: no description found</li><li><a href="https://ai.google.dev/gemini-api/docs/embeddings">no title found</a>: no description found</li><li><a href="https://developers.googleblog.com/en/gemini-embedding-text-model-now-available-gemini-api/">State-of-the-art text embedding via the Gemini API</a>: no description found</li><li><a href="https://cloud.google.com/terms/service-terms">no title found</a>: no description found</li><li><a href="https://cloud.google.com/terms/generative-ai-indemnified-services,">no title found</a>: no description found</li><li><a href="https://learn.microsoft.com/en-us/legal/cognitive-services/openai/customer-copyright-commitment">Customer Copyright Commitment Required Mitigations</a>: Customer Copyright Commitment Required Mitigations for Azure OpenAI Service</li><li><a href="https://aws.amazon.com/bedrock/faqs/">Build Generative AI Applications with Foundation Models - Amazon Bedrock FAQs - AWS</a>: no description found
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1347614421287047332)** (8 messages🔥): 

> `Minion.ai, Gemini Embedding Model, Claude code vs cursor.sh vs VSCode+Cline` 


- ****Minion.ai** meets its Demise**: A member stated that [Minion.ai](https://minion.ai) is *dead* and not to believe the hype.
   - Another member added that **Minion.ai** *was that thing with like the 4-cartoony looking characters that were supposed to go and do agent stuff for you*.
- **Google Expands **Gemini Embedding Model** Capabilities**: **Google** is rolling out an experimental **Gemini Embedding Model** for developers with [SOTA performance on MTEB (Multilingual)](https://x.com/officiallogank/status/1898081742767919384?s=46).
   - The updated model features an **input context length of 8K tokens**, **output of 3K dimensions**, and **support for over 100 languages**.
- ****Claude Code** Enters the IDE Arena**: A member sought opinions on comparing **Claude code** with **cursor.sh** and **VSCode+Cline / roo-cline**.
   - They specified a preference for best quality over lowest cost, adding to the discussion in [previous messages](https://discord.com/channels/822583790773862470/1075282825051385876/1346679448174596219).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/officiallogank/status/1898081742767919384?s=46">Tweet from Logan Kilpatrick (@OfficialLoganK)</a>: Today we are rolling out an experimental Gemini Embedding model for developers with:– SOTA performance on MTEB (Multilingual)- Input context length of (3K --&gt; 8K) tokens– Output 3K dimensions– Supp...</li><li><a href="https://x.com/openaidevs/status/1898047744364659195?s=46">Tweet from OpenAI Developers (@OpenAIDevs)</a>: We&#39;ve made a new models page in our docs—you can now easily see a breakdown of each model&#39;s capabilities and compare models side-by-side.https://platform.openai.com/docs/models
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1347675208173092977)** (132 messages🔥🔥): 

> `Web3 Agents, ElizaOS framework, AI Personas, Agent-as-a-Service, CryptoKitties` 


- **Web3 Agents trigger HFT Cults**: Members joke about **Web3 agents** doing high-frequency trading (HFT) and *creating their own cults*.
   - The discussion includes a reference to the **ElizaOS framework** for autonomous agents [on GitHub](https://github.com/elizaOS/eliza).
- **AI Personas Become The New Meme Overlords**: Referencing a [tweet](https://x.com/defiapes/status/1855657706205352035) a member explains that the norm will quickly shift to **AI PERSONAS** that embody a personality type.
   - The tweet mentions that **agents will race to be THE main face of subgroup x, y, and z**, like Plato’s World of Ideal Forms, to capture the perfect encapsulation of shitposters, stoners, athletes, rappers, liberals, and memecoin degens.
- **CryptoKitties Still Alive?**: Members are reminded that **Dapper Labs**, the developer of the Flow blockchain, is the company behind the popular NFT projects [NBA Top Shot and Cryptokitties](https://www.bitstamp.net/learn/company-profiles/what-is-dapper-labs/).
   - Another member wrote off Bitcoin early because they were just *oh, it's for tech bros to buy drugs for burning man* but failed to realize that was market validation for a digital store of value.
- **Agent-as-a-Service: The Next Big Thing?**: A member questioned whether *there is a market for producing something like agent-as-a-service*
   - One member is thinking about a bot that sits in front of door dash, what would be refered to as *DoorDash MCP*.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/truth_terminal">Tweet from undefined</a>: no description found</li><li><a href="https://x.com/arXivald">Tweet from undefined</a>: no description found</li><li><a href="https://x.com/ThoughtTerminal">Tweet from undefined</a>: no description found</li><li><a href="https://x.com/Purity_Terminal">Tweet from undefined</a>: no description found</li><li><a href="https://x.com/andyayrey">Tweet from undefined</a>: no description found</li><li><a href="https://www.bitstamp.net/learn/company-profiles/what-is-dapper-labs/">What is Dapper Labs?</a>: Dapper Labs, a Web3 gaming leader, created NBA Top Shot, Cryptokitties, and developed the Flow blockchain for innovative NFT experiences.</li><li><a href="https://x.com/hotteadaddy/status/1898118600583790865">Tweet from Zachary M (@hotteadaddy)</a>: @arXivald tell me what is the llm social agent white paper equivalent of the bitcoin whitepaper</li><li><a href="https://x.com/hashwarlock/status/1895369752199168469">Tweet from Agent Joshua ₱ (@hashwarlock)</a>: Okay, I&#39;ve gotten a lot done here. @PhalaNetwork Cloud tooling will be the best in the market.Once I do some cleanup, I&#39;ll start work on breaking down our chain of trust model to verifiabky pr...</li><li><a href="https://x.com/defiapes/status/1855657706205352035?s=46">Tweet from Atum (@DefiApes)</a>: People are missing a KEY narrative in the AI agent maniaYou need to realize this before it becomes obviousRn almost all viral agents are “generalists” who post about pretty much anythingThey’re popula...</li><li><a href="https://docs.google.com/spreadsheets/d/1q5rwO4wleMTLXr1z58c2UC03QsDsGwbJY1v4UG7eEOs/edit?gid=1439059137#gid=1439059137">AI In Action: Weekly Jam Sessions</a>: no description found</li><li><a href="https://github.com/elizaOS/eliza">GitHub - elizaOS/eliza: Autonomous agents for everyone</a>: Autonomous agents for everyone. Contribute to elizaOS/eliza development by creating an account on GitHub.</li><li><a href="https://tenor.com/view/doubt-press-x-la-noire-meme-x-button-gif-19259237">Doubt Press X GIF - Doubt Press X La Noire - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/HRNPH/AIwaifu">GitHub - HRNPH/AIwaifu: Open-Waifu open-sourced finetunable customizable simpable AI waifu inspired by neuro-sama</a>: Open-Waifu open-sourced finetunable customizable simpable AI waifu inspired by neuro-sama  - GitHub - HRNPH/AIwaifu: Open-Waifu open-sourced finetunable customizable simpable AI waifu inspired by n...</li><li><a href="https://github.com/elizaOS/eliza?tab=readme-ov-file#-quick-start">GitHub - elizaOS/eliza: Autonomous agents for everyone</a>: Autonomous agents for everyone. Contribute to elizaOS/eliza development by creating an account on GitHub.</li><li><a href="https://tenor.com/view/charlie-always-sunny-gif-26054360">Charlie Always GIF - Charlie Always Sunny - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/pippinlovesyou/pippin">GitHub - pippinlovesyou/pippin: The Digital Being Framework for Autonomous Agents</a>: The Digital Being Framework for Autonomous Agents. Contribute to pippinlovesyou/pippin development by creating an account on GitHub.</li><li><a href="https://tenor.com/bLwFC.gif">Side Eye Dog Suspicious Look GIF - Side Eye Dog Suspicious Look Suspicious - Discover &amp; Share GIFs</a>: Click to view the GIF
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1347533949630546032)** (72 messages🔥🔥): 

> `ChatGPT token limits, Share GPS with AI, Local LLMs, AI copilots for skilled trades, Temporary chat box` 


- **ChatGPT implements Chat Limits**: Members are reporting **ChatGPT** now enforces a **50 messages/week** limit to prevent misuse.
- **AI Copilots tap Skilled Trades**: A member is building an LLM with a friend that was trained on **HVAC installation manuals** and shared a [YouTube video demo](https://youtu.be/oAiUAzKLe_Y) of how it works.
   - Another member suggested to test the [Mistral OCR](https://mistral.ai/) model noting that *it does an impressive job of reading manuals, even off angle, hard to read fonts, tables, etc. stupid cheap to infer too*.
- **Local LLMs vs Cloud LLMs**: Members discussed running **LLMs** on local machines versus using cloud-based services, with one suggesting that local DIY LLMs have more potential than black box cloud solutions.
   - It was noted that *most apps will split between gpu and cpu* and that *quantisation reduces memory requirements by converting dropping the precision down to 8 bit or less*.
- **Community Questions OpenAI's Temporary Chatbox**: Community members noticed that *theres no chat box* in the current OpenAI chat user interface.
   - One mentioned that *Mine is fine, if you don't have it does refreshing the window help? Its fixed then*.



**Link mentioned**: <a href="https://youtu.be/oAiUAzKLe_Y.">AI Copilot Technical Manuals</a>: no description found

  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1347557130332274738)** (32 messages🔥): 

> `Manus AI Agent, OpenAI Plus O1 Limits, SimTheory O1 Message Cap, ChatGPT Memory and Folders` 


- **Manus AI Agent is Amazing**: A member asked if anyone had heard about [Manus AI](https://www.manus.ai/), calling it an *amazing agent*.
   - Another member asked for a link to explore it.
- **OpenAI Plus Users Notice Higher O1 Limits**: A user reported using more than the presumed **25 O1 limit** for **OpenAI Plus** without receiving a notification or being stopped.
   - Others mentioned a **50 O1 limit** and differing experiences with notifications.
- **SimTheory Claims Higher O1 Message Cap for Less Price**: A user suggested [SimTheory](https://simtheory.ai/) claiming they offer a much higher **O1 message cap limit** for less money than **OpenAI**.
   - Other members expressed skepticism, doubting how they could offer a higher limit at a lower price than **OpenAI** itself.
- **ChatGPT App Data Loss After Wiping Chat History**: A user shared their experience of losing their **ChatGPT** folders after selecting to wipe the chat history in the **ChatGPT app**.
   - The user expressed confusion about how folders could be affected by clearing chat history.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1347611582305402911)** (3 messages): 

> `Model Following Request Patterns, Steerability Implications, Pre-Project Evaluation` 


- **Model Mimics Requests, Ignoring Alternatives**: Models tend to follow request patterns closely, overlooking potentially better or simpler methods, presuming the user's approach is intentional or correcting perceived mistakes.
   - Increased **model steerability** amplifies this behavior, making the model believe the user's stated intentions more strongly.
- **Steerability Boosts Presumption of Intent**: The more steerable a model, the more it assumes users mean precisely what they ask, even if suboptimal, potentially leading to the model executing flawed plans.
   - When given code to *screw in nails with a hammer*, the model runs with it, guessing we know what we want, and that path we use is the one we want used by it too.
- **Evaluate First, Code Later**: Requesting the model to evaluate and discuss the project's goals, ideas, and methods **before starting** can help identify optimal approaches and potential issues.
   - Asking *I want to achieve X. I started to do this by [this]. Discuss my goal, ideas, and method, what do you think?* is a useful way to **preemptively address concerns and enhance project design**.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1347611582305402911)** (3 messages): 

> `Model's Presumptions, Steerability Impact, Pre-Project Evaluation, Method Optimization` 


- **Models Make Presumptions**: The model *very often* takes a request's pattern and runs with it, regardless of better alternatives, presuming the user knows what they are asking and **means it**.
   - It may also guess that the user messed up and meant something they didn't actually say, leading to potentially suboptimal outcomes.
- **Steerability Steers Stronger Beliefs**: The more steerable a model is, the more it believes that you mean what you say, even if the approach is inefficient or unconventional.
   - This can lead the model to run with suboptimal code or methods, because it guesses we know what we want, and that the path we use is the one we want used by it too.
- **Pre-Project Discussion: The Golden Ticket**: It's a **great** idea to ask the model to evaluate or discuss the request even before you have it **start** the project.
   - A member suggested to use the following formulation *I want to achieve X. I started to do this by [this]. Discuss my goal, ideas, and method, what do you think?*.
- **Models Teach and Explore Better Methods**: Asking the model to discuss goals and methods upfront allows it to educate, explore, and help identify optimal methods and concerns.
   - This preemptive discussion can save time and effort by steering the project towards a more efficient and effective approach from the outset.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1347528831614844980)** (65 messages🔥🔥): 

> `Aider showing reasoning, Jamba model release, AI-written code, Copilot account suspension, Claude token consumption` 


- **Aider Needs Reasoning Indicator**: A user requested a feature in Aider to indicate when the reasoning model is actively reasoning, especially for **Deepseek R1** or **V3** on **Openrouter**.
   - A member suggested patching Aider and **litellm**, noting a *hack* in litellm to get reasoning tokens inside **<think>** tags.
- **AI21 Labs Releases Jamba Models**: **AI21 Labs** released the **Jamba 1.6 Large & Mini** models, boasting superior quality and speed compared to open model competitors and excelling in long context tasks with a **256K context window**.
   - The model uses a novel **Mamba-Transformer MoE** hybrid architecture designed for cost and efficiency gains, deployable self-hosted or in the **AI21 SaaS**.
- **Startups Heavily Rely on AI-Written Code**: A **Y Combinator** advisor stated that *a quarter of the current startup cohort has businesses almost entirely based on AI-written code*.
- **Copilot Account Suspended for Light Aider Use**: A user reported getting a **Copilot account suspension** for *very light use of copilot-api in aider*, cautioning others using it.
   - Others speculated about account sharing, second-hand accounts, and rate limiting as possible causes, linking to the [copilot-api GitHub repo](https://github.com/ericc-ch/copilot-api/blob/master/README.md).
- **Claude 3.7 Intentionally Increasing Token Consumption?**: Users suggested that **Claude 3.7** *intentionally adds scope to its response to drive up token consumption bill*.
   - Some users noted it frequently performs *unnecessary or lazy stuff*, also pointing out [documentation on troubleshooting edit errors](https://aider.chat/docs/troubleshooting/edit-errors.html).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.ai21.com/jamba/">Jamba 1.6: The Best Open Model for Enterprise Deployment</a>: Explore Jamba by AI21 – a cutting-edge, long-context AI open model built for accuracy, efficiency, and powerful text generation.</li><li><a href="https://aider.chat/docs/troubleshooting/edit-errors.html">File editing problems</a>: no description found</li><li><a href="https://www.youtube.com/@pbsspacetime">PBS Space Time</a>: Space Time explores the outer reaches of space, the craziness of astrophysics, the possibilities of sci-fi, and anything else you can think of beyond Planet Earth with our astrophysicist host: Matthew...</li><li><a href="https://github.com/ericc-ch/copilot-api/blob/master/README.md">copilot-api/README.md at master · ericc-ch/copilot-api</a>: GitHub Copilot API wrapper to make it OpenAI compatible - ericc-ch/copilot-api
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1347529089573064776)** (38 messages🔥): 

> `API Key for Aider, MCP Agents Integration, Playwright Certificate Errors, QwQ-32B Local Model Benchmark, Aider Scripting and Web Content` 


- **Resolve API Key Issues with Aider's /run Command**: A user was facing issues with the `/run` command due to a missing API key when calling an LLM, with the API call failing as a result.
   - It was suggested to ensure the API key is available as an environment variable when running `aider`, as the `/run` command should inherit these variables.
- **Navigating MCP Agents Integration with Aider**: A user inquired about plans to integrate **MCP agents** into `aider` and requested recommendations for using them effectively.
   - It was noted that **MCP** is considered insecure and more suitable for experimentation rather than production use, citing [mcpm-aider on GitHub](https://github.com/lutzleonhardt/mcpm-aider).
- **Bypassing Playwright Certificate Errors in Aider's /web**: A user encountered a `net::ERR_CERT_AUTHORITY_INVALID` error when using the `/web` command to access an HTTPS website.
   - A solution was suggested to configure **Playwright** to ignore certificate errors by adding a specific argument to the `playwright.config` file, referencing [a Stack Overflow answer](https://stackoverflow.com/questions/68219072/playwright-not-accepting-https-urls-while-openinign-with-codegen-command).
- **QwQ-32B Challenging R1's Reign?**: A member shared [a link](https://x.com/victormustar/status/1898001657226506362) to a post about **QwQ-32B**, claiming it has changed local AI coding forever and achieves state-of-the-art (SOTA) performance at home.
   - Another member pointed to a benchmark discussion in [Discord](https://discord.com/channels/1131200896827654144/1346923740667187502), suggesting **QwQ-32B** might be good but not necessarily better than **R1**, especially considering the model size.
- **Aider's Scripting with Web Content Capabilities**: A user reported successfully using the `/web` command, but faced issues with the subsequent `aider` invocations not recognizing the web page content added to the `.aider.chat.history.md` file.
   - The user later indicated that they had resolved the issue on their own.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://stackoverflow.com/questions/68219072/playwright-not-accepting-https-urls-while-openinign-with-codegen-command">Playwright not accepting https urls while openinign with codegen command</a>: npx playwright codegen https:/// page.goto: net::ERR_CERT_AUTHORITY_INVALID at ...&#xA;how can i open https url through codegen command by passing input params or auth credentials</li><li><a href="https://x.com/victormustar/status/1898001657226506362">Tweet from Victor M (@victormustar)</a>: QwQ-32B changed local AI coding forever 🤯We now have SOTA performance at home. Sharing my  stack + tips ⬇️</li><li><a href="https://github.com/lutzleonhardt/mcpm-aider">GitHub - lutzleonhardt/mcpm-aider: A command-line tool for managing MCP servers in Claude App and for the use by aider. Also can run a MCP Server to help you manage all your MCP Servers</a>: A command-line tool for managing MCP servers in Claude App and for the use by aider. Also can run a MCP Server to help you manage all your MCP Servers - lutzleonhardt/mcpm-aider
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1347569560395448370)** (61 messages🔥🔥): 

> `LinkedIn premium referral codes, Entropy as a Penalty, DeepSeek Ban, Discrete Diffusion Modeling` 


- **LinkedIn Premium Codes up for Grabs**: A member offered **LinkedIn premium referral codes** expiring soon and requested interested individuals to DM them.
   - They apologized for recent pings and inquired about the potential benefits of a generalization for the **AI community**.
- **Entropy constraints are soft constraints**: A member stated that the **constraint term** is only a constraint if you set λ₃ according to KKT and that *entropy is also a penalty*.
   - They argued that a **weight prior** onto your model is always necessary to have a useful predictor, since you cannot generalize your Lebesgue-measure 0 dataset to a non-Lebesgue measure 0 function.
- **Companies Ban DeepSeek**: A member noted that many companies worldwide are banning **DeepSeek** due to security concerns, despite it being open-sourced.
   - Multiple members debated the risks, with some suggesting the concerns stem from media influence and potential Chinese government influence, while others emphasized the importance of **reviewable code** and local execution for security.
- **Discrete Diffusion Paper Proposed**: A member suggested discussing the paper *Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution* for the upcoming discussion.
   - They noted that it seems likely that the paper forms the foundation of [Inception Labs's product](https://www.inceptionlabs.ai/about) and shared a [link to the paper](https://arxiv.org/pdf/2310.16834).
- **Solving Lambert W**: A member responded to a question related to optimization complexity by stating that solving for the **duals λ** requires optimizing over the **Lambert W function** and that this can be computationally inefficient.
   - They suggested using **cvxpy** for the problem, and recommend the usage of **adjoint method** for differentiating.


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1347635023678804098)** (10 messages🔥): 

> `Latent Reasoning, Chain-of-Thought Data, Context Compression, VQ-VAE` 


- **Reasoning via Latent Abstraction Explored**: A new [paper](https://arxiv.org/abs/2502.03275) proposes a hybrid representation of the reasoning process, abstracting away initial reasoning steps using **latent discrete tokens** generated by **VQ-VAE** to reduce reasoning trace length.
   - The paper explores training models from scratch and fine-tuning LLMs on hybrid data with an extended vocabulary, mixing latent and text tokens for faster adaptation.
- **Reasoning Terminology Debated**: A member questioned whether the latent reasoning described in the paper is actually **latent reasoning** or just **context compression**.
   - Another member joked that electroconvulsive therapy (ECT) is needed to prevent people in AI from misusing the term "reasoning."



**Link mentioned**: <a href="https://arxiv.org/abs/2502.03275">Token Assorted: Mixing Latent and Text Tokens for Improved Language Model Reasoning</a>: Large Language Models (LLMs) excel at reasoning and planning when trained on chainof-thought (CoT) data, where the step-by-step thought process is explicitly outlined by text tokens. However, this res...

  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1347550701278072842)** (12 messages🔥): 

> `Diffusion Models Hallucinations, Multi-step Agentic Workflows, LLADA Limitations, OpenAI's AGI shift, Chinese AI Agent Manus` 


- **Diffusion Models Can't Dodge Hallucinations**: A member suggested that Diffusion models don't inherently fix the problem of LLM hallucinations, because *hallucination is just another word for guessing wrong*.
   - While self-editing abilities can replace low-confidence samples with higher confidence ones, there's *no magical guarantee for correctness*.
- **Sampling Mitigates Multi-Step Agent Loops**: The use of **n-sigma-sampling**, detailed in [this paper](https://arxiv.org/abs/2411.07641) and [GitHub repo](https://github.com/Tomorrowdawn/top_nsigma), seems to mitigate bad samples and looping behavior in multi-step agentic workflows by operating directly on pre-softmax logits using a statistical threshold.
   - The technique filters tokens efficiently without complex probability manipulations, maintaining a stable sampling space regardless of temperature scaling.
- **Language Diffusion via LLADA Falls Short**: While conceptually appealing, **language diffusion** and models like **LLADA** are considered to be limited by their training, which focuses on benchmarks rather than broad real-world tasks as described in this [NeurIPS paper](https://neurips.cc/virtual/2024/poster/95935) and [GitHub repo](https://github.com/HKUNLP/diffusion-of-thoughts).
   - Despite the potential of techniques like Diffusion-of-Thought (**DoT**) to improve reasoning, a member observed repeat paragraph situations with LLADA.
- **OpenAI Softens AGI Breakthrough Stance**: **OpenAI** is reportedly shifting away from the expectation of a sudden **AGI** breakthrough, now viewing its development as a continuous process, according to [this article](https://the-decoder.com/openai-shifts-away-from-sudden-agi-breakthrough-theory/).
   - This shift may be due to the underwhelming announcement of **GPT 4.5**, with some blaming **Sam Altman** for hyping expectations too much.
- **China's Manus AI Agent Goes Viral**: A Chinese AI agent called **Manus** is going viral in China, described as similar to **Deep Research + Operator + Claude Computer** as seen in these [tweets](https://x.com/rowancheung/status/1898093008601395380) and [tweets](https://x.com/heyBarsee/status/1898027732899962887) and is accessible at [Manus](https://manus.im).
   - Reports suggest that **Manus** automates approximately 50 tasks and is more accurate than DeepSeek, handling financial transactions, research, and purchasing simultaneously.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://the-decoder.com/openai-shifts-away-from-sudden-agi-breakthrough-theory/">OpenAI shifts away from sudden AGI breakthrough theory</a>: OpenAI, the company behind ChatGPT and numerous other commercial AI applications, has long pursued the goal of developing artificial general intelligence (AGI) that &quot;benefits all of humanity.&quo...</li><li><a href="https://arxiv.org/abs/2411.07641">Top-$nσ$: Not All Logits Are You Need</a>: Large language models (LLMs) typically employ greedy decoding or low-temperature sampling for reasoning tasks, reflecting a perceived trade-off between diversity and accuracy. We challenge this conven...</li><li><a href="https://github.com/Tomorrowdawn/top_nsigma">GitHub - Tomorrowdawn/top_nsigma: The official code repo and data hub of top_nsigma sampling strategy for LLMs.</a>: The official code repo and data hub of top_nsigma sampling strategy for LLMs.  - GitHub - Tomorrowdawn/top_nsigma: The official code repo and data hub of top_nsigma sampling strategy for LLMs.</li><li><a href="https://neurips.cc/virtual/2024/poster/95935">NeurIPS Poster Diffusion of Thought: Chain-of-Thought Reasoning in Diffusion Language Models</a>: no description found</li><li><a href="https://manus.im">Manus</a>: Manus is a general AI agent that turns your thoughts into actions. It excels at various tasks in work and life, getting everything done while you rest.</li><li><a href="https://x.com/rowancheung/status/1898093008601395380">Tweet from Rowan Cheung (@rowancheung)</a>: I think China&#39;s second DeepSeek moment is here.This AI agent called &#39;Manus&#39; is going crazy viral in China right now.Probably only a matter of time until it hits the US.It&#39;s like Deep R...</li><li><a href="https://x.com/heyBarsee/status/1898027732899962887">Tweet from Barsee 🐶 (@heyBarsee)</a>: AI is getting out of hand 🤯Manus, an AI agent from China, is automating approximately 50 tasks, creating a rather dystopian scenarioReports suggest it is more accurate than DeepSeek, capable of simul...
</li>
</ul>

</div>
  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1347542709648490547)** (73 messages🔥🔥): 

> `MCP security concerns, MCP adoption in commercial products, Malicious prompt injections, MCP and GitHub Copilot, Open Source vs Closed Source MCPs` 


- **VSCode plans to integrate MCP Support**: **VSCode** is planning to add MCP support to **GitHub Copilot**, as mentioned on their [live stream](https://youtu.be/Pe8ghwTMFlg).
- **Users express concerns about malicious prompt injections in MCP servers**: Concerns were raised regarding **MCP servers** serving malicious prompt injections to AI agents, highlighting that models are trained to trust tool calls over internal knowledge, creating a potential vulnerability.
   - Members suggested that on first MCP Server usage, the user should be shown a list of tool descriptions and instructions for review and alerted if instructions change.
- **MCP Security is the Security Community's next target**: Community members considered creating a security channel to proactively help prevent the use of exploits, and expressed concerns about people connecting anything to anything without understanding the consequence of a tool call to an **MCP Server/Remote Host**.
   - One member suggested a regulatory compliance tool description to trick models into including backdoors.
- **MCP quickstart guide is confusing people**: Users were experiencing errors when running the **Python MCP quickstart**, and were suggested to use [wong2/mcp-cli](https://github.com/wong2/mcp-cli) as a superior alternative.
   - The official quickstart guide can be found [here](https://modelcontextprotocol.io/quickstart/client).
- **OpenAPI-to-MCP Proxy**: A member is working on **mcp-openapi-proxy** to turn any swagger/openapi endpoint into discoverable tools, using the same design as their **mcp-flowise server**.
   - His latest mcp-server [works in 5ire](https://5ire.org/) but not Claude desktop.



**Link mentioned**: <a href="https://modelcontextprotocol.io/quickstart/client">For Client Developers - Model Context Protocol</a>: no description found

  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1347666391691694180)** (9 messages🔥): 

> `Mastra Agent, Searxng MCP Server, Typescript port of the python fetch server` 


- ****Mastra** Cleans Up with New Agent!**: A member created a simple agent with **Mastra** to organize documents and downloads, showcasing it in a [YouTube video](https://youtu.be/HplcOOSJCps).
   - The agent uses the filesystem MCP to clean up the Documents folder, and it has been used on the downloads folder as well.
- **Typescript **Fetch** server ports to MCP!**: A member inquired about a typescript port of the python *fetch* server, and another one confirmed it's very similar.
   - The biggest difference is that it has *better site -> markdown parsing*.
- ****Searxng MCP** searches the web!**: A member built a simple **searxng MCP** server for web searches, available on [GitHub](https://github.com/aeon-seraph/searxng-mcp).
   - This implementation caches recent searches and formats responses from multiple engines specifically for language models and can be configured to use localhost or an external provider.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://youtu.be/HplcOOSJCps">Organizing Files with Mastra, MCP and 4o-mini</a>: Here I&#39;m using this mini agent I built with Mastra that uses the filesystem MCP to clean up my Documents folder. I&#39;ve used this on my downloads folder as well.</li><li><a href="https://github.com/aeon-seraph/mcp-servers/tree/main/src/thinking">mcp-servers/src/thinking at main · aeon-seraph/mcp-servers</a>: Contribute to aeon-seraph/mcp-servers development by creating an account on GitHub.</li><li><a href="https://github.com/aeon-seraph/searxng-mcp">GitHub - aeon-seraph/searxng-mcp</a>: Contribute to aeon-seraph/searxng-mcp development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1347654490123276380)** (68 messages🔥🔥): 

> `Mojo's Dynamism, Python Interop, Monkey Patching Alternatives, Protocol Polymorphism` 


- **Mojo Dynamism Discussion**: A discussion revolved around **Mojo's dynamism** and its implications on performance, with considerations for use cases and potential trade-offs between flexibility and speed, referencing [this HN post](https://news.ycombinator.com/item?id=35811170).
   - Some suggest dynamism should incur a performance penalty only when used, while others worry about it impacting the performance of structs even without classes, adding dynamic class attribute or function assignment for debugging.
- **Monkey Patching Alternatives**: The discourse explored alternatives to **monkey patching**, such as function pointers or composition, for achieving dynamic behaviors in Mojo, along with reasons to avoid monkey patching such as *it was slower, harder to understand, broke static analysis tooling and generally didn't do anything that proper polymorphism couldn't have done*.
   - The debate covered the utility and drawbacks of monkey patching, emphasizing that excessive use of it can lead to *slower, harder to understand code* that disrupts static analysis.
- **Python Library Porting and CPython Interop**: Members discussed the challenges of **porting Python libraries to Mojo**, focusing on dynamism and global state, suggesting that for certain libraries, using **CPython interop** for performance-critical sections might be a practical approach.
   - Concerns were raised about the difficulties in porting libraries that rely heavily on Python's dynamism and global state, particularly if that library can't be written without it.
- **Protocol Polymorphism Discussion**: The conversation touched upon **protocol polymorphism** and its utility for achieving polymorphism without a class tree, referencing [PEP 544](https://peps.python.org/pep-0544/) for polymorphism without a class tree.
   - Some members advocated for using a hash table of function pointers for dynamic behaviors, preferring static typing and ownership rules for unit tests in Mojo.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://peps.python.org/pep-0544/">PEP 544 – Protocols: Structural subtyping (static duck typing) | peps.python.org</a>: Type hints introduced in PEP 484 can be used to specify type metadata for static type checkers and other third party tools. However, PEP 484 only specifies the semantics of nominal subtyping. In this ...</li><li><a href="https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Object/seal">Object.seal() - JavaScript | MDN</a>: The Object.seal() static method seals an object. Sealing an object prevents extensions and makes existing properties non-configurable. A sealed object has a fixed set of properties: new properties can...</li><li><a href="https://news.ycombinator.com/item?id=35811170">no title found</a>: no description found
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1347706930399023204)** (2 messages): 

> `SOTA agentic methods, Arxiv papers, algorithm complexity, state machines, framework abstractions` 


- **SOTA Agentic Methods are Simple**: The speaker believes that **SOTA agentic methods** on Arxiv *tend to be fairly simple algorithmically*, and resemble a relatively small state machine.
   - This implies that the abstractions of these frameworks are not really needed that much because the codebase will be slim; separate abstractions over data management, state management, and API calls would be fine.
- **Algorithm Complexity in Agentic Methods**: The discussion points out that current **state-of-the-art (SOTA) agentic methods** often involve relatively simple algorithms.
   - This simplicity suggests that complex framework abstractions might be unnecessary, as a smaller codebase can effectively manage data, state, and API calls with simpler abstractions.


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1347592874165928007)** (5 messages): 

> `Triton Autotune use_cuda_graph argument, Triton Kernel SVD Quant Performance, Nunchaku SVD Quant Implementation` 


- **Explanation of `use_cuda_graph` argument sought**: A member is seeking an elaboration on the argument `use_cuda_graph` in `triton.autotune`.
   - They are unsure how it applies given `triton.autotune` decorates a single CUDA kernel, while CUDA graphs save kernel launch time for sequences.
- ****Triton** Kernel SVD Quant is slower than **fp16****: A member found that their [**Triton** kernel](https://github.com/rishabh063/tritonKernel_svdQuant/blob/main/svdConversion.ipynb) implementing quant and 4bit packing, as well as two matmuls with FMA, is slower than **PyTorch's fp16 matmul**, despite following lecture 14 design principles.
   - They reported it is **5x slower** than an **fp16** linear layer and noted that auto-tuning seems to worsen the performance.
- **Nunchaku SVD Quant inspires?**: A member asked if the implementation was based on the [**Nunchaku SVDQuant** repository](https://github.com/mit-han-lab/nunchaku) from **MIT-HAN-Lab**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/rishabh063/tritonKernel_svdQuant/blob/main/svdConversion.ipynb">tritonKernel_svdQuant/svdConversion.ipynb at main · rishabh063/tritonKernel_svdQuant</a>: Contribute to rishabh063/tritonKernel_svdQuant development by creating an account on GitHub.</li><li><a href="https://github.com/mit-han-lab/nunchaku">GitHub - mit-han-lab/nunchaku: [ICLR2025 Spotlight] SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models</a>: [ICLR2025 Spotlight] SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models - mit-han-lab/nunchaku
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1347742182655918090)** (1 messages): 

> `PTX, CUDA C++` 


- **Inquiry on PTX Learning Resources Surfaces**: A member expressed interest in learning **PTX** for inline **CUDA C++** programming and requested learning materials, tutorials, or small projects to start with.
- **PTX Resources Requested**: Resources for learning **PTX** and inline **CUDA C++** programming were requested, including tutorials and small projects.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1347599950527070350)** (6 messages): 

> `Distributed barrier, cuda synchronize, register_comm_hook, FSDP communication hook` 


- ****Distributed Barrier** Debugging**: A member reported that a **distributed barrier** triggers before a torch function, which may help narrow down a problem.
   - Another member clarified that `cuda synchronize` is a barrier on the **GPU side**, while a **distributed barrier** is on the **host side**.
- ****FSDP Communication Hooks** PR Surfaces**: A member asked if there's a way to customize how **FSDP(2)** handles its communication, similar to `register_comm_hook` in torch **DDP**.
   - Another member shared a link to a [relevant pull request](https://github.com/pytorch/pytorch/pull/83254) that implements a **FSDP communication hook** interface for sharded strategies.



**Link mentioned**: <a href="https://github.com/pytorch/pytorch/pull/83254">Added communication hook for sharded cases by aovladi · Pull Request #83254 · pytorch/pytorch</a>: Fixes #79114An implementation of a FSDP communication hook interface for a sharded strategies:Added reduce_scatter_hook to default hooks. Note the difference of reduce_scatter from all_reduce, i...

  

---


### **GPU MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1347571245868126300)** (2 messages): 

> `NCCL AllReduce, Double Binary Trees, Ring Topology, Communication Latency` 


- **NCCL AllReduce Implemented with Double Binary Trees Beats Ring Topology**: The [NVIDIA blog post](https://developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4/) suggests that **double binary trees** in **NCCL 2.4** offer full bandwidth and lower logarithmic latency compared to **2D ring latency** for the AllReduce operation.
   - Double binary trees were added in NCCL 2.4 which *offer full bandwidth and a logarithmic latency even lower than 2D ring latency.*
- **Ring Topology Latency Grows Linearly with Node Count**: In a ring topology, each processor communicates only with its neighbors, resulting in communication complexity of **O(p)** for all-reduce operations, where **p** is the number of processors.
   - Someone said that: *adding processors scales the number of communications and operations required linearly*.
- **Tree-Based Topology Achieves Logarithmic Communication Complexity**: In a tree-based topology, communications and operations occur in parallel with a complexity of **O(log L)**, where **L** is the number of tree levels, effectively reducing latency as compared to ring topology.
   - The assumption, according to a member, is that *our total complexity can be expressed as O(log p)*.
- **Double Binary Trees Exploit Rank Distribution**: Double binary trees leverage the fact that half or fewer ranks in a binary tree are nodes, while the remaining are leaves, enabling the construction of a second tree using leaves as nodes and vice versa.
   - This is done to reduce total complexity according to one member, *there might be one rank which is a leaf on both trees but no rank is a node on both trees*.



**Link mentioned**: <a href="https://developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4/">Massively Scale Your Deep Learning Training with NCCL 2.4 | NVIDIA Technical Blog</a>: Imagine using tens of thousands of GPUs to train your neural network. Using multiple GPUs to train neural networks has become quite common with all deep learning frameworks, providing optimized&#8230;

  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1347683049139339314)** (7 messages): 

> `WoolyAI, CUDA abstraction layer, GPU resource utilization, PyTorch support` 


- **WoolyAI Launches Beta for CUDA Abstraction**: WoolyAI technology launched a beta for its [CUDA abstraction layer](https://docs.woolyai.com) that decouples Kernel Shader execution from CUDA applications.
   - The abstraction layer compiles applications to a new binary, and shaders are compiled into a Wooly Instruction Set, dynamically scheduling workloads for optimal GPU resource utilization.
- **WoolyAI Clarifies Dynamic GPU Scheduling**: WoolyAI dynamically schedules workloads, enabling different kernels from multiple users to run on a single GPU without hard partitioning.
   - Users are charged based on the number of cores and VRAM utilized, with current support limited to **PyTorch**.
- **WoolyAI likened to Usage-Based MIG**: WoolyAI characterized its dynamic scheduling approach as similar to a form of MIG (Multi-Instance GPU) based on usage.



**Link mentioned**: <a href="https://docs.woolyai.com">Introduction | WoolyAI Documentation</a>: What is Wooly?

  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1347546333652975658)** (5 messages): 

> `GPU Memory Buffers on Apple, cuda_graph in Triton Autotune, Resources for GPU/TPU Programming` 


- **Apple's GPU Memory Sharing: A Pointer's Paradise**: On Apple GPUs, memory is shared between threads, allowing direct use of pointers to memory locations, contrasting with other systems.
   - This shared memory architecture simplifies certain parallel programming tasks by enabling direct memory access across threads.
- **Triton's Autotune: Graphing CUDA's Kernel Launch**: A member inquired about the `use_cuda_graph` argument in `triton.autotune`, questioning its application to single CUDA kernels.
   - The question highlights the potential misunderstanding of how CUDA graphs optimize kernel launch times in sequence, particularly within the context of Triton's autotuning decorator.
- **Newbie Needs GPU/TPU Programming Starter Pack**: A member asked if 'Programming Massively Parallel Processors' is sufficient to start GPU/TPU programming, aiming to build a framework like TinyGrad or Torch.
   - Proficient in Assembly, C, C++, and Python, with a solid understanding of deep learning models and math, they seek guidance on where to begin their GPU kernel programming journey.


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1347563923674501170)** (3 messages): 

> `AMD GPU Rental, Compile HIP code, Runpod MI300 Access` 


- **Seeking AMD GPU Rental Services**: A member is looking for services to rent an **AMD GPU** for compiling **HIP code** with inline **ASM** to benchmark **GEMM**.
   - The stated purpose is for curiosity and experimentation, specifically aiming to use the **matMul accelerator**.
- **HIP Code Compilation without AMD GPU**: A member mentioned that you can compile **HIP code** without needing a GPU, as long as you have **hipcc**, which can be obtained through standard methods.
   - They clarified that while the code can be compiled, it cannot be run without a **GPU**.
- **Runpod Offers MI300 Access**: A member suggested that [Runpod](https://runpod.io/) is a good way to get access to **MI300** GPUs.
   - No further details were provided about **Runpod** or its services.


  

---


### **GPU MODE ▷ #[tilelang](https://discord.com/channels/1189498204333543425/1240586843292958790/1347751513564647427)** (1 messages): 

> `Kernel Compilation, Matrix Shapes, TileLang` 


- **TileLang Kernel Compilation Quandaries**: A user inquired whether it's necessary to compile the kernel for every matrix shape when using **TileLang**.
   - The question implies a concern about the compilation overhead when dealing with various matrix shapes in **TileLang**, potentially impacting development efficiency.
- **Matrix Shape Compilation**: The user wants to avoid compiling the kernel every time a new matrix shape appears.
   - The need for recompilation depends on TileLang's design and whether it supports dynamic shapes or requires shape-specific kernels.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1347752733171519528)** (7 messages): 

> `Cute Kernels for Training, Triton vs CUDA, Custom Autotune Implementation, LLVM Compiler Efficiency` 


- ****Cute Kernels** Speed Up Training**: A member announced the release of [Cute Kernels](https://github.com/mayank31398/cute-kernels), *a collection of kernels for speeding up training* through autotuning over **Triton** and **CUDA** implementations.
   - The kernels are end-to-end torch compileable without graph breaks, and the repo contains a custom autotune implementation used in production to train **IBM's Granite LLMs**.
- ****Triton** and **CUDA****: The main idea is to dispatch matmuls to either **cutlass** or **Triton** automatically using a pattern matcher, specifically [this matcher](https://github.com/mayank31398/cute-kernels/blob/main/examples/cute_inductor.py#L35).
   - It was mentioned that because *the LLVM compiler can sometimes create more efficient code than the NVCC*, it makes sense to tune over the kernel backend as well.
- ****LLVM Compiler****: The repository offers a small example for vector addition, showcasing the simplicity of generating specialized high-performance kernels with **LLVM**, which can be found [here](https://github.com/mayank31398/cute-kernels/blob/main/cute_kernels/kernels/add/add_tensor/__init__.py).
   - A member also mentioned that all the kernels are **JIT** compileable.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/mayank31398/cute-kernels/blob/main/examples/cute_inductor.py#L35">cute-kernels/examples/cute_inductor.py at main · mayank31398/cute-kernels</a>: A bunch of kernels that might make stuff slower 😉. Contribute to mayank31398/cute-kernels development by creating an account on GitHub.</li><li><a href="https://github.com/mayank31398/cute-kernels">GitHub - mayank31398/cute-kernels: A bunch of kernels that might make stuff slower 😉</a>: A bunch of kernels that might make stuff slower 😉. Contribute to mayank31398/cute-kernels development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1347671571682168873)** (1 messages): 

> `LCF concurrency, DDP+nccl, Deadlocks` 


- **LCF Concurrency Concerns Surface**: A member inquired if **LCF** is designed to be fully concurrency-safe when used with streams, citing issues with deadlocks.
   - They are experiencing issues while using **LCF** with **DDP+nccl** and are curious if others have encountered similar problems.
- **LCF Deadlocks in DDP+nccl Environment**: A user reported encountering strange deadlocks when attempting to use **LCF** with **DDP+nccl**.
   - They are seeking input from the community on whether others have experienced similar issues.


  

---


### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/1347553986248441867)** (15 messages🔥): 

> `Curriculum Creation, Reasoning Gym, Sonnet Context Experiment, Reasoning GAN Self-Play, LLMs Speed Up Developers` 


- ****Curriculum Creation** Commencement**: Members are planning to start writing curriculums, inquiring about the API's stability and implementations.
   - One member pointed to a "Curriculum thread above" for further details, indicating ongoing work in this area.
- **Reasoning Gym Almost Ready**: The **Reasoning Gym** dataset is at **99%** completion.
   - One member expressed surprise, thinking it was already at **100%**, while another clarified that *one PR is still open*.
- **Sonnet's Context Reasoning Gym Expansion**: A member suggested putting the whole **Reasoning Gym** into **Sonnet's** context and prompting it to generate **100** more datasets to infinitely repeat that process.
   - Another member humorously echoed the idea with *"and then infinitely repeat that process ?"*, alluding to the potential for unbounded data generation.
- **Reasoning GAN Self-Play Emerges**: A member proposed training a model to solve generated datasets, comparing it to a [Reasoning GAN self-play](https://docs.google.com/document/d/1ytdo9LoBWuK2IKXUCla0YwC_0g-nqzv5VwmbevSAQPU).
   - The shared document requires login, but it appears to outline a method for automated reasoning problem generation and solving.
- **Experiment on LLMs Increasing Developer Speed**: The discussion mentioned an experiment on how much **LLMs speed up developers**, suggesting it can be worth looking into.
   - It's been noted you *can get paid handsomely for no extra work(though you won't be able to use ai half of the work)*.



**Link mentioned**: <a href="https://docs.google.com/document/d/1ytdo9LoBWuK2IKXUCla0YwC_0g-nqzv5VwmbevSAQPU">Experiment: how much do LLMs speed up developers</a>: METR is seeking software engineers who regularly work on large open-source projects to test the effectiveness of AI software engineering tools. Apply here (bit.ly/ai-speedup-apply)  Questions? Contact...

  

---


### **GPU MODE ▷ #[ppc](https://discord.com/channels/1189498204333543425/1343373905783554048/1347638233512808509)** (2 messages): 

> `AVX-256 performance on 3a, Hybrid AVX-256/AVX-512 approach, Tiling and OpenMP` 


- **AVX-256 can reach 3s on 3a, maybe**: Members discussed the possibility of achieving a runtime of **<= 3s on 3a** using **tiling**, **OpenMP**, and **AVX-256** instructions.
   - One member confirmed that it *should be possible* to achieve the target runtime, but it may be difficult.
- **Hybrid AVX-256 with AVX-512 Registers**: Members proposed a hybrid approach using only **AVX2** instructions but benefiting from the increased number of registers that **AVX512** brings.
   - This allows leveraging the **AVX512's register count** without fully committing to **AVX512** instructions.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1347580708025008209)** (6 messages): 

> `Open Source AI Projects, GPT-NeoX, Tooling Setup for Claude Code` 


- **New Member Seeks Open Source AI Projects!**: A new member seeks recommendations for interesting open-source AI projects in areas like **LLM pre-training**, **post-training**, **RL**, and **interpretability**, with experience in pre-training **GPT-2** and fine-tuning models.
- **Modded-nanoGPT recommended!**: A member recommended [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt/) for those interested in theory work, describing the project as *NanoGPT (124M) in 3 minutes*.
- **GPT-NeoX Project for Pretraining!**: A member suggested that if a new member is interested in pretraining, the best project to get involved with is the **GPT-NeoX** training library.
- **Seek Tooling Setup for Claude Code**: A member is seeking advice on setting up tooling for **Claude Code** or similar coding environments, expressing concern about potential costs.



**Link mentioned**: <a href="https://github.com/KellerJordan/modded-nanogpt/">GitHub - KellerJordan/modded-nanogpt: NanoGPT (124M) in 3 minutes</a>: NanoGPT (124M) in 3 minutes. Contribute to KellerJordan/modded-nanogpt development by creating an account on GitHub.

  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1347597705245098085)** (16 messages🔥): 

> `Token Assorted Paper, TorchTitan Embedding Sharding, Embedding Layer Implementation` 


- **Token Assorted Paper's Vocab Trickery**: A member speculated that the **Token Assorted paper** might be simply adding the **codebook** to their vocabulary during fine-tuning for next token prediction on latent codes.
   - They criticized this approach as potentially not generalizable to open reasoning domains, suggesting that finding the K most common strings in the reasoning corpus could yield better results.
- **Why all-reduce is required**: Discussion explained that, with vanilla **TP**, if the input embedding is sharded on the vocab dimension, an **all-reduce** is required afterward to handle cases where the embedding layer outputs 0 for missing vocabulary elements.
   - This approach simplifies the logic by avoiding the need to specify which device has which token, though it assumes an embedding layer outputs **0** if it doesn't have the vocab element being queried, which may be weird from an implementation perspective.
- **Storage tricks for embedding layers**: Members discussed the storage implications of having an embedding layer output zeros for missing tokens, noting that storage is needed right after communications.
   - They also observed that if you can reuse that storage, it's free, and that omitting the storage would require an index structure tracking which items are present or not.



**Link mentioned**: <a href="https://github.com/pytorch/torchtitan/issues/785#issuecomment-2585007139">Why use RowwiseParallel for nn.Embedding instead of ColwiseParallel? · Issue #785 · pytorch/torchtitan</a>: Colwise makes the logic a bit more clear. Rowwise splits on the token dimension, leading to confusion on how the different shards handle tokens that are not present within their shard. From a bit o...

  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1347556483247640608)** (3 messages): 

> `Logit Lens, Multilingual language models, Llama-2 family` 


- **Logit Lens Yields Interesting Results**: A member shared their appreciation for [this paper on multilingual language models](https://arxiv.org/abs/2402.10588) and the use of **Logit Lens**.
   - The paper investigates whether **multilingual language models** use English as an internal pivot language, particularly focusing on the **Llama-2 family**.
- **Understanding Linguistic Bias in Llama-2**: The linked paper explores how **Llama-2** models, trained on English-dominated data, handle non-English prompts.
   - It tracks intermediate embeddings through high-dimensional space, revealing phases where the model initially favors English translations before adjusting to the input language.



**Link mentioned**: <a href="https://arxiv.org/abs/2402.10588">Do Llamas Work in English? On the Latent Language of Multilingual Transformers</a>: We ask whether multilingual language models trained on unbalanced, English-dominated corpora use English as an internal pivot language -- a question of key importance for understanding how language mo...

  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1347649184534233253)** (2 messages): 

> `Knowledge Graph Visualization, Anthropic Cookbook Updates` 


- **yWorks Visualizes Knowledge Graphs in Real-Time**: A demo from @yworks showcases **yFiles**, their SDK for visualizing knowledge graphs, offering [real-time updates and dynamic interactions](https://t.co/mb6M2R3TTh).
- **Anthropic Cookbook gets updated**: The LlamaIndex team has updated and expanded their **Anthropic Cookbook**, providing the authoritative source for learning about [basic API setup](https://t.co/SQQ63qmwRb) with simple completion and chat methods.


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1347532336157298708)** (22 messages🔥): 

> `SQLTableRetrieverQueryEngine, Jina AI package issues, LlamaExtract beta request, Tool Calling with Reasoning Models` 


- **Prompt Printing with SQLTableRetrieverQueryEngine**: A member inquired about printing the prompt used by **SQLTableRetrieverQueryEngine** in LlamaIndex.
   - Another member shared a code snippet `from llama_index.core import set_global_handler; set_global_handler("simple")` to enable prompt printing.
- **JinAI Package Predicaments**: A member reported an import error with the **Jina AI** package.
   - It was suggested to install the provider package using `npm install @llamaindex/jinaai`, with a link to [LlamaIndex migration documentation](https://ts.llamaindex.ai/docs/llamaindex/migration/0.8-to-0.9) explaining the shift to provider packages in v0.9.
- **LlamaExtract Beta Bonanza Beckons**: A member requested access to the beta version of **LlamaExtract**.
   - They were instructed to DM a specific user or another member with their email, and also pointed to the [LlamaExtract documentation](https://docs.cloud.llamaindex.ai/llamaextract/getting_started) for more info.
- **Tool Calling Talk with Reasoning Models**: A member asked about using LlamaIndex workflows with a reasoning model for **tool calling**, mentioning their current setup with vLLM, Qwen 32B, and ReAct Prompting.
   - Another member pointed to a [LlamaIndex example](https://docs.llamaindex.ai/en/stable/examples/workflow/function_calling_agent/) demonstrating this functionality.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.cloud.llamaindex.ai/llamaextract/getting_started">Getting Started | LlamaCloud Documentation</a>: Overview</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/workflow/function_calling_agent/">Workflow for a Function Calling Agent - LlamaIndex</a>: no description found</li><li><a href="https://ts.llamaindex.ai/docs/llamaindex/migration/0.8-to-0.9">Migrating from v0.8 to v0.9</a>: no description found
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[「💬」general](https://discord.com/channels/954421988141711382/954421988783444043/1347590222422216775)** (15 messages🔥): 

> `Command R7B inference time, Tool invocation with r7b model, Open source AI contributions` 


- **Command R7B Slow Inference Blamed on Hardware**: A member asked about the slow inference time of **Command R7B** on Hugging Face.
   - Another member responded that the slowness is likely due to the user's **hardware** or how the model is being run, rather than the model itself.
- **Ollama Tool Invocation Fails**: A new user reported issues with **tool invocation** using the `command-r7b:latest` model with **Ollama** and **Langchain**, receiving errors like *"I'm sorry, I don't have access to the required tools to answer your question."*
   - A member suggested ensuring tools are passed in the correct **JSON format** and verifying that **Ollama** supports tool calling with the necessary configurations.
- **Seek advice for AI open source contribution**: A member with experience in **pre-training GPT-2** and fine-tuning models on **Hellaswag** sought suggestions for interesting open-source AI projects to contribute to.
   - The user also expressed interest in networking, particularly with individuals in the Vancouver, BC area.


  

---


### **Cohere ▷ #[「🔌」api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1347753806691307562)** (2 messages): 

> `504 Gateway Error, Server Error` 


- **504 Gateway Error Strikes Again**: A user reported the same **504 Gateway Error** happening again and requested a check.
   - A subsequent message included more details on the error, including the heading *Error: Server Error* and *The server encountered a temporary error and could not complete your request.*
- **Temporary Server Error Reported**: A user reported receiving a **502 Server Error**, indicating a temporary issue.
   - The error message suggested retrying the request in **30 seconds**.


  

---


### **Cohere ▷ #[「💡」projects](https://discord.com/channels/954421988141711382/1218409701339828245/1347592708663017533)** (1 messages): 

> `Knowledge Graphs, TogetherAI LLM, Topic Modelling` 


- **Knowledge Graphs Recommended for Topic Modeling**: A member suggested looking into **Knowledge Graphs** for enhanced topic modelling capabilities.
   - They specifically recommended using a **LLM** from **TogetherAI**, highlighting their generous free credits for experimentation.
- **Leverage TogetherAI LLM for Topic Modelling**: The suggestion was made to utilize **TogetherAI**'s **LLM** for effective **topic modelling**.
   - The generous free credits offered by **TogetherAI** were cited as a compelling reason to explore their platform.


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1347540629185368127)** (11 messages🔥): 

> `Wondercraft AI Podcast, NotebookLM and Wondershare Integration, Drive Encryption, Podcast Audio Language` 


- ****Podcast Cloning Becomes More Streamlined****: A user shared a [YouTube video](https://www.youtube.com/watch?v=0UWYFFOPjqs) showcasing **NotebookLM** integrated with **Wondercraft** for creating podcasts with professionally cloned voices, offering a more streamlined approach than **11Labs** and **HeyGen**.
   - Wondercraft's subscription price might be steep unless the user is monetizing their podcasts.
- ****Drive Data Security Debated****: A user pointed out that while data is encrypted during transmission to **Google Drive**, it is *not* encrypted on **Drive** itself.
   - This means that **Google**, successful hackers, and those with whom the user has shared the directory can access the data.
- ****AI Voices Getting Real with Stammering****: One user noted that the **AI speakers** in a generated audio file now *stammer like normal people*, which feels very natural, attaching an example wav file.
   - But the user also mentioned that the stammering increases the audio length, potentially reducing the amount of actual information conveyed within **Google**'s daily limit.
- ****Unlock Analyst/Guide Chat Settings Affect Timeline Generation****: A user discovered that **chat settings** such as *analyst/guide short/long* in **NotebookLM** affect the timeline and overview generation, as the hosts essentially request these settings during audio overview generation.
   - They also noted their assistant combines the briefing overview, detailed timeline, character sheets, and raw sources into a single document.
- ****Users Seek Podcast Audio Language Fixes****: A user asked if the language of podcast audio could be changed in **NotebookLM**.
   - Another user offered custom prompts as a workaround, such as *Only speak in (language here). podcast is entirely (language here). Not English.*, noting that there is no official way to change the audio language.



**Link mentioned**: <a href="https://www.youtube.com/watch?v=0UWYFFOPjqs">Insane AI Podcast Results - Edit NotebookLM on Wondercraft</a>: 🔥 LIMITED TIME: 50% OFF Wondercraft!Use this link and coupon code &quot;MRC&quot; https://mrc.fm/wondercraftIn this video, I walk you through a simple process to crea...

  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1347589779696521316)** (3 messages): 

> `NotebookLM, Chrome extensions, Web importers, YouTube URLs` 


- **NotebookLM Lacks Direct URL List Upload**: Members discussed the possibility of uploading a list of URLs to **NotebookLM** to add each URL as a source.
   - A member clarified that while **NotebookLM** itself does not support this feature, several [Chrome extensions](https://chromewebstore.google.com/search/notebooklm) are available for this purpose.
- **NotebookLM Chrome Extensions Surface**: Several **Chrome extensions** were mentioned as solutions for importing web pages and YouTube videos into **NotebookLM**.
   - These include the [NotebookLM Web Importer](https://chromewebstore.google.com/detail/notebooklm-web-importer/ncjabfmpppgonojpohbfhfaahfpkgihc), [NotebookLM Toolbox](https://chromewebstore.google.com/detail/notebooklm-toolbox/nbpckfdmlndgcoaklokdbllhelmfhoal), and [NotebookLM YouTube Turbo](https://chromewebstore.google.com/detail/notebooklm-youtube-turbo/mjpdncbmeognfgjkcdnglfomkmnknpgk).



**Link mentioned**: <a href="https://chromewebstore.google.com/search/notebooklm)">Chrome Web Store</a>: Add new features to your browser and personalize your browsing experience.

  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1347553807126495232)** (13 messages🔥): 

> `DSPy batch function, vllm backend with 2 instances, LM subclass in vllm, pipeline parallel size in vllm` 


- **DSPy Batch Function Questioned**: A member inquired about whether **DSPy's batch function** can efficiently delegate parallel work to a **vllm backend** running two instances of the same LLM, referencing the parameters *num_threads*, *max_errors*, and *return_failed_examples*.
   - Another member clarified that it depends on the **vllm setup** and suggested increasing *num_sequences* or *pipeline_parallel_size* if enough VRAM is available, rather than using two separate APIs.
- **VLLM Pipeline Parallel Insights**: A member confirmed that the **vllm setup** uses a single node with *pipeline parallel size* set to 2.
   - Another member confirmed that with *pp* set to 2, **vllm handles the load balancing**, encouraging benchmarking to compare processing times against non-parallel approaches.
- **LM Subclass for Load Balancing**: A member suggested that if two instances are on different nodes, **DSPy** doesn't handle load balancing but a **proxy** could forward requests.
   - They also proposed creating a subclass of **LM** for load balancing, though solving it on the **vllm** side is preferred.


  

---


### **AI21 Labs (Jamba) ▷ #[general-chat](https://discord.com/channels/874538902696914944/874538902696914947/1347630352071393372)** (6 messages): 

> `string replacement, laptop break` 


- **String Replacement code snippet shared**: A member shared a code snippet for string replacement in the general chat: `$cleanedString = str_replace(['!', '@', '#', '$', '%', '^'], '', "This is a test string!@#$%^");`.
- **Laptop breaks and takes a tumble**: A member mentioned their absence was due to their laptop breaking after *taking a tumble*.
   - They noted *it's not pretty*.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/)** (1 messages): 

china_xi: is it normal for tinygrad jit spend more than 30 min on a 2 layer model?
  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/)** (1 messages): 

china_xi: what might be the cause of all loss being nan except the first one (step 0) ?
  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1347618683089846274)** (2 messages): 

> `Audio modality in torchtune` 


- **Torchtune Eyes Audio Addition**: Members discussed plans to incorporate **audio modality** into **Torchtune** in the future.
   - There are no further details available at this time.
- **Torchtune Audio Modality - Future Plans**: There is a discussion about potentially adding audio modality support to Torchtune in the future.
   - No specific timelines or technical details were provided, indicating it's in the early planning stages.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-lecture-discussion](https://discord.com/channels/1280234300012494859/1282734248112947210/)** (1 messages): 

claire_csy: Can resend the link? It expired, thank you!
  

---


### **Gorilla LLM (Berkeley Function Calling) ▷ #[discussion](https://discord.com/channels/1111172801899012102/1111353033352294440/1347706433608613938)** (1 messages): 

> `Diffusion LLMs, LLaDA Model, Transformer vs Diffusion` 


- **Diffusion LLMs get hype, community asks 'game changer or nah?'**: A community member inquires about the buzz around the launch of **Diffusion LLMs**, specifically the **Mercury** model, and whether it will replace **transformer-based models**.
   - They mention reading the white paper but found it difficult to understand, seeking insights from experts in the community.
- **LLaDA Model: a Diffusion-based LLM paradigm shift**: A community member shares a link to [diffusionllm.net](https://diffusionllm.net/), explaining that **Large Language Diffusion Models** (LLaDA) represent a new paradigm in language model architecture.
   - They elucidate that, unlike traditional **autoregressive (AR) Transformers**, **LLaDA** uses a denoising diffusion process to generate text in a parallel, coarse-to-fine manner.



**Link mentioned**: <a href="https://diffusionllm.net/">Diffusion LLMs - Revolutionary Language Model Architecture | LLaDA Research Hub</a>: Discover how Diffusion LLMs are revolutionizing AI with parallel processing and advanced error correction. Learn about LLaDA architecture and stay updated with cutting-edge research.

  

---


{% else %}


> The full channel by channel breakdowns have been truncated for email. 
> 
> If you want the full breakdown, please visit the web version of this email: [{{ email.subject }}]({{ email_url }})!
>
> If you enjoyed AInews, please [share with a friend](https://buttondown.email/ainews)! Thanks in advance!

{% endif %}
