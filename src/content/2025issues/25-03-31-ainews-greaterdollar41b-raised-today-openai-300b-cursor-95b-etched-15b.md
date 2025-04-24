---
id: c0d7868b-224b-4597-a78c-7e5f396130d6
title: '>$41B raised today (OpenAI @ 300b, Cursor @ 9.5b, Etched @ 1.5b)'
date: '2025-04-01T06:33:20.931042Z'
original_slug: ainews-41b-raised-today-openai-300b-cursor-95b
description: >-
  OpenAI is preparing to release a powerful new open-weight language model with
  advanced reasoning capabilities, engaging with developers globally to optimize
  its usefulness. Meanwhile, Cursor secured a record $625 million funding round
  at a $9.6 billion valuation, marking the largest startup raise in AI history.
tags:
  - openai
  - language-model
  - open-weight
  - cursor
  - startup-funding
  - large-language-model
  - reasoning
---


<!-- buttondown-editor-mode: plaintext -->**More money is all you need**

> AI News for 3/28/2025-3/31/2025. We checked 7 subreddits, [**433** Twitters](https://twitter.com/i/lists/1585430245762441216) and **30** Discords (**230** channels, and **17665** messages) for you. Estimated reading time saved (at 200wpm): **1870 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

[Amazon Nova Act (Adept + Covariant) made a really good run](https://labs.amazon.science/blog/nova-act?utm_campaign=introducing-nova-act&utm_medium=organic-asw&utm_source=twitter&utm_content=asw-twitter&utm_term=2025-mar) at taking the headline today, but  it's not every day that people close [the largest startup fundraise in history](https://www.thewrap.com/openai-valued-300-billion-new-round-funding/):

![image.png](https://assets.buttondown.email/images/8b893e65-1789-4f44-bff5-4e306788bf09.png?w=960&fit=max)

[Cursor closed $625m at $9.6B](https://fxtwitter.com/ArfurRock/status/1906768733135098360) and [Etched closed $85m at $1.5B](https://x.com/ArfurRock/status/1906756943349260682).



---


{% if medium == 'web' %}


**Table of Contents**

[TOC] 

{% else %}

The **Table of Contents** and **Channel Summaries** have been moved to the web version of this email: [{{ email.subject }}]({{ email_url }})!

{% endif %}


---

# AI Twitter Recap

**Language Models and Releases**

- **OpenAI is planning to release a highly capable open language model, their first since GPT-2**, and is hosting sessions with global developers to gather feedback and engage directly with the community to ensure they get it right, according to [@kevinweil](https://twitter.com/kevinweil/status/1906797119848988822). [@sama](https://twitter.com/sama/status/1906793591944646898) provided more details, stating the company is **excited to release a powerful new open-weight language model with reasoning in the coming months** and wants to talk to devs about how to make it maximally useful.
- **DeepSeek V3 0324 has ranked #5 on the Arena leaderboard**, surpassing DeepSeek-R1 and every other open model, according to [@lmarena_ai](https://twitter.com/lmarena_ai/status/1906739061236334744). It's the **#1 open model** with an MIT license, 2x cheaper than DeepSeek-R1, and top-5 across all categories.
- [@scaling01](https://twitter.com/scaling01/status/1906505283477586330) believes that **only three LLMs were very clearly SOTA step-changes: GPT-4, Sonnet 3.5, and o1**, with all other model releases feeling more like nice-to-haves / incremental improvements. [@scaling01](https://twitter.com/scaling01/status/1906502465869971507) also noted that **it doesn't feel like Gemini models are ahead, as Google keeps doing "exp" models and hasn't even shipped Gemini 2.0 Pro**.
- [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1906790937604579430) announced the launch of **Sophont, a company building open multimodal foundation models for the future of healthcare**.
- [@stevenheidel](https://twitter.com/stevenheidel/status/1906797154301329845) stated that **we're releasing a model this year that you can run on your own hardware**.

**Gemini 2.5 Pro**

- **Gemini 2.5 Pro is outperforming other models like Claude 3.7 Sonnet in coding tasks**, according to [@lepikhin](https://twitter.com/lepikhin/status/1906745155681730569).
- [@scaling01](https://twitter.com/scaling01/status/1906748722572079438) shared notes indicating that the **production version of Gemini 2.5 Pro with pricing will come "very soon hopefully," with Flash being the next model to receive the 2.5 series**. Gemini 2.5 Pro has dynamic thinking but is not yet where they want it to be, as it overthinks for most questions, and better image generation is also on their shipping list.
- [@dzhng](https://twitter.com/dzhng/status/1906575275997167857) finds **Gemini 2.5 impressive for coding**, as it tells you when it can't do what you asked, whereas Sonnet tends to just power through and give you a wrong solution.
- [@raizamrtn](https://twitter.com/raizamrtn/status/1906727510601355393) announced **Gemini Code, a coding assistant in your terminal powered by Gemini 2.5 Pro**.

**AI Applications, Frameworks, and Tools**

- **SkyPilot has a new paper accepted to EuroSys 2025** about SkyServe, which intelligently provisions and spreads spot and on-demand instances across regions and clouds, leading to 43% lower costs while maintaining high availability, according to [@skypilot_org](https://twitter.com/skypilot_org/status/1906685409309974548).
- [@Hacubu](https://twitter.com/Hacubu/status/1906763329248624909) announced the official launch of **AgentEvals, a new open-source package that helps answer the question "Is my agent working?"**
- [@karpathy](https://twitter.com/karpathy/status/1906748528627503433) discussed **smartphone choices and privacy**, noting that iPhone has taken user defense and privacy a lot more seriously over time than Android.
- **LlamaIndex now supports the OpenAI Responses API** with full support for built-in-tools, reasoning, images, manual tool calling, streaming, and async, according to [@llama_index](https://twitter.com/llama_index/status/1906739777619288540).
- [@togethercompute](https://twitter.com/togethercompute/status/1906737438833209362) announced a **new notebook for building a fact-checking agent** that can search for documents to verify a claim, using DSPy and Together, with automatic prompt engineering to improve its performance by +20% with help from a larger LLM agent.
- Kevin Frans and colleagues at [@UCBerkeley](https://twitter.com/DeepLearningAI/status/1906768474816295165) introduced a **new way to speed up image generation with diffusion models**. Their “shortcut” method trains models to take larger noise-removal steps—the equivalent of multiple smaller ones—without losing output quality.

**AI Research and Papers**

- **VBENCH-2.0 is out on Hugging Face**, a next-gen benchmark for evaluating intrinsic faithfulness, with 18 fine-grained dimensions, fully automatic and open-source, and human-aligned via large-scale validation, according to [@_akhaliq](https://twitter.com/_akhaliq/status/1906757376507535736).
- [@TheAITimeline](https://twitter.com/TheAITimeline/status/1906470808563626322) highlighted top AI/ML research papers including **GPT-4o System Card: Native Image Generation, Anthropic's On the Biology of a LLM, Gemma 3 Technical Report, and Qwen2.5-Omni Technical Report**, among others.

**AI Funding and Investment**

- [@sophiamyang](https://twitter.com/sophiamyang/status/1906786071796429146) noted a **great opportunity with $1M for every early stage startup**.
- [@demishassabis](https://twitter.com/demishassabis/status/1906664622226083922) announced that **@IsomorphicLabs has raised $600M to turbocharge their mission to one day solve all disease with the help of AI**.

**Humor/Memes**

- [@ID_AA_Carmack](https://twitter.com/ID_AA_Carmack/status/1906737776491470883) quipped, **Deep down at the bottom of Hephaestus’ giant forge, a charred arm sticks out of the glowing molten metal with its thumb held high.**
- [@teortaxesTex](https://twitter.com/teortaxesTex/status/1906531814165934262) joked, **«AGI» already has a solution, but you won't like it**.
- [@nearcyan](https://twitter.com/nearcyan/status/1906557838677385231) remarked on how **it only took a single model release to mark the end of coherent reality**.


---

# AI Reddit Recap

## /r/LocalLlama Recap

Here are the summaries for the selected posts, grouped by theme:

**Theme 1: Qwen 3 Support Merged into Transformers**
[Permalink](https://www.reddit.com/r/LocalLLaMA/comments/1jnzdvp/qwen3_support_merged_into_transformers/)

* Support for **Qwen3** models has been merged into the **Hugging Face Transformers** library via [Pull Request #36878](https://github.com/huggingface/transformers/pull/36878). This update prepares the **Transformers** ecosystem for upcoming **Qwen3** model releases.
* The author questions the lack of discussion around **Qwen 2.5 Omni**, describing it as the *first open-sourced multimodal model with voice, image, and text generation*. They express surprise at the limited attention given its capabilities.

**Theme 2: Qwen 2.5 Omni Multimodal Model**
[Permalink](https://www.reddit.com/r/LocalLLaMA/comments/1jnvqsg/why_is_no_one_talking_about_qwen_25_omni/)

* The author finds it strange that **Qwen 2.5 Omni**, the *first open-sourced multimodal model handling voice, image, and text generation*, isn't receiving more attention. They perceive its release as a notable development for open-source multimodal systems.
* A member of the **Orpheus TTS** team compares their architecture to alternatives like **Moshi** and **Sesame**, stating their opinion that *conceptually **Qwen Omni** is a far superior architecture* for end-to-end speech. They reason this is because **Qwen Omni** avoids modifying the base **LLM**, unlike **Sesame/Moshi**, while retaining potential for emotional expression similar to **Orpheus**.

**Theme 3: OpenDeepSearch Outperforms Proprietary Search Tools**
[Permalink](https://www.reddit.com/r/LocalLLaMA/comments/1jogfrz/opensource_search_repo_beats_gpt4o_search/)

* The author introduces the **OpenDeepSearch** repository ([GitHub link](https://github.com/sentient-agi/OpenDeepSearch)), an open-source search tool using **ReAct**, **CodeAct**, dynamic few-shot prompting, and integrated search/calculator functions. They highlight its reported success over **GPT-4o Search** and **Perplexity Sonar Reasoning Pro** on the **FRAMES** benchmark and note its potential utility in multi-agent workflows.
* *(Note: Only one post directly matches this specific theme in the provided data.)*

**Theme 4: High-End PC Build for Running Large Models (Deepseek-V3-0324 671b)**
[Permalink](https://www.reddit.com/r/LocalLLaMA/comments/1jnzq51/pc_build_run_deepseekv30324671bq8_locally_68_toks/)

* The author details building a PC with dual **EPYC 9355** CPUs and **768GB** of **5600MHz** RDIMM RAM on a **Gigabyte MZ73-LM0** motherboard to run **Deepseek-V3-0324:671b-Q8** locally. They report achieving **6-8 tokens per second** and describe installing **Ubuntu 24.04.2 LTS**, **ollama**, and **Open WebUI**.
* The author reports that the **LM Arena** was updated, adding **Deepseek v3.1** which scored **1370**, reportedly higher than **Deepseek R1**. They also mention observing models named **Nebula** (suspected **Gemini 2.5**), **Phantom** (recently removed), and **Chatbot-anonymous**.
* The author issues a warning about a circulating blog post falsely claiming a **"Deepseek V3.1"** release, hosted on a fake website. They remind users that **Deepseek** does not operate an official blog for such announcements.

**Theme 5: Diminishing Returns of Larger LLMs**
[Permalink](https://www.reddit.com/r/LocalLLaMA/comments/1jnvhkd/the_diminishing_returns_of_larger_models_perhaps/)

* The author posits that models like **Gemma3 27B** and **QwQ 32B** show diminishing returns for large (**70B+**) LLMs, citing their competitive benchmark performance against models like **Llama 3.3 70B**. They attribute this trend to improved **distillation**, **architecture**, and **data quality**, suggesting large hardware investments may offer only temporary advantages as **30B-50B** models improve.
* The author describes constructing a high-specification system with dual **EPYC 9355** CPUs and **768GB RAM** designed explicitly for running the large **Deepseek-V3-0324:671b-Q8** model locally. This setup yields **6-8 tokens per second** using tools like **ollama** and **Open WebUI**.
* According to the author, the **LM Arena** leaderboard was updated to include **Deepseek v3.1**, achieving a score of **1370** and surpassing **Deepseek R1**. The post notes observations of other potentially significant models like **Nebula** (possibly **Gemini 2.5**) on the platform.

## Other AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding

> Pipelines still down today but should be fixed by tomorrow.

---

# AI Discord Recap

> A summary of Summaries of Summaries by Gemini 2.0 Flash Thinking

**Theme 1. Gemini 2.5 Pro: Coding King or Tool-Use Fool?**

- **Gemini 2.5 Pro Wows at Code, Fumbles with Tools**: Users across Cursor, OpenAI, and Manus.im Discords are buzzing about **Gemini 2.5 Pro's** impressive coding skills, with some praising its prowess in languages like Jax and C++. However, in Cursor Community, users report **tool use troubles**, suggesting it's *not good at actually calling the tools* within Cursor, often outputting incorrect or non-functional code, raising suspicions of intentional limitations to push paid options.
- **Gemini 2.5 Pro: A Multi-Modal Beta Beast?**: In Manus.im and LMArena, **Gemini 2.5 Pro** is lauded for complex analysis, reasoning, and multi-modal tasks, even outperforming **GPT-4.5** in creative coding and physics simulations [Gemini 2.5 Pro physics simulations in Three.js!](https://x.com/renderfiction/status/1905998185962643767). However, it *can't execute an entire workflow* on its own, and some OpenAI users find it *terrible at C++ and WinAPI*, citing hallucinations.
- **Rate Limits and Quotas Crimp Gemini 2.5 Pro's Style**: Despite the hype, rate limits are a recurring concern. In Aider and OpenRouter, users report **rate limits** hindering practical use, with one OpenRouter user facing a *45906 seconds later* retry delay.  OpenRouter clarified that rate limits can originate from both **Google** and **OpenRouter**, see [rate limits documentation](https://openrouter.ai/docs/api-reference/limits).

**Theme 2. Open Source vs Proprietary Models: The Reasoning Race Heats Up**

- **OpenAI Teases Open-Weight Reasoning Model**: Sam Altman teased a powerful new **open-weight language model with reasoning capabilities** coming soon, seeking developer feedback on how to make it maximally useful, as announced in [this tweet](https://x.com/sama/status/1906793591944646898?t=Xw_DyPuHG0edzBlLvbUn3g&s=19). This sparks debate in Latent Space and Yannick Kilcher discords about its implications and potential capabilities, with some speculating it's part of the **GPT-5** system under development.
- **DeepSeek V3 Flexes Math Muscles, Instruction Following Fades**: Hugging Face's evaluations of **DeepSeek V3 0324** reveal impressive gains in **math and GPQA**, as tweeted [here](https://x.com/nathanhabib1011/status/1905018770764259818), but with a slight dip in instruction following. Unsloth AI released dynamic quantized versions for local execution and a guide [Tutorial: How to Run DeepSeek-V3-0324 Locally](https://docs.unsloth.ai/basics/tutorial-how-to-run-deepseek-v3-0324-locally).
- **Grok's Performance Rollercoaster: Science Star or Log-Off Lagger?**: LMArena users debate **Grok3's** scientific supremacy over **Gemini**, with claims it outperforms even **R1** on **arc-agi-1**. However, OpenAI and PerplexityAI users report **Grok's unstable performance**, plagued by frequent log-offs and internal errors, and a non-functional *thinking mode*. Despite these issues, some users maintain subscriptions alongside **ChatGPT Pro**.

**Theme 3.  Cursor vs Alternatives: Context, Cost, and Code Stability Clash**

- **Cursor Customers Cry 'Context Costly!'**: Cursor Community members express frustration with **Cursor's usage-based pricing**, token limits, and reduced model quality upon reaching limits, citing the [Cursor Pricing page](https://www.cursor.com/pricing). Many are exploring alternatives like **Cline or Roo Code** for full context windows and lower costs.
- **Cline and Roo Code Rise as Cursor Challengers**: The community debates **Cline's stability** versus **Cursor's features**, with many preferring Cline for reliability. **Roo Code** gains traction for features like boomerang tasks and better context retention, viewed as a step up from Cline, as described in [this Reddit thread](https://www.reddit.com/r/ChatGPTCoding/comments/1jn36e1/roocode_vs_cline_updated_march_29/). However, concerns persist about Roo Code's stability and high Anthropic API token consumption.
- **Windsurf Waves as a Wildcard Cursor Competitor**: Cursor Community explores **Windsurf** as a potential alternative to Cursor for its terminal/server task stability and embedded browser, but some users find its context window even smaller and question its value, stating *I don't like windsurf at all, the context window seems even smaller*.

**Theme 4. Quantization Quandaries and Performance Paradoxes**

- **Quantization Quality Quagmire**: Aider and GPU MODE users discuss the impact of quantization on model performance. Converting models from **FP16** to **Q8** results in a slight quality reduction, while **Q4** quantization, common in Ollama, severely degrades it. Users report anything below **Q6** is severely impaired, especially for reasoning tasks.
- **BFloat16 Breaks RoPE's Positional Promise**: GPU MODE highlights a new paper [When Precision Meets Position: BFloat16 Breaks Down RoPE in Long-Context Training](https://arxiv.org/abs/2411.13476) showing **BFloat16** introduces numerical errors in **RoPE**, even when computed in **Float32**. The paper introduces **AnchorAttention** as a fix, with code on [GitHub](https://github.com/haonan3/AnchorContext).
- **Dynamic Quantization Debuts to DeepSeek's Delight**: Unsloth AI released dynamic quantized versions of **DeepSeek-V3-0324**, alongside a [guide](https://docs.unsloth.ai/basics/tutorial-how-to-run-deepseek-v3-0324-locally) for local execution. Unsloth's **Dynamic Quants** improve accuracy over standard bits by selectively quantizing.

**Theme 5.  MCP Momentum: Protocol Progress and Practical Projects Proliferate**

- **MCP Spec Drafts OAuth 2.1, Sparks Debate**: MCP Discord discusses the latest **2025-03-26 MCP spec** draft introducing **OAuth 2.1** for authentication, detailed in the [MCP spec](https://spec.modelcontextprotocol.io/specification/2025-03-26/basic/authorization/). However, no client currently supports it for testing. Implementation of **HTTP Streamable Transport** raises concerns about session resumability and message replay, see [MCP spec](https://spec.modelcontextprotocol.io/specification/2025-03-26/basic/transports/#listening-for-messages-from-the-server).
- **IDA Pro MCP Server Cracks Reverse Engineering Code**: MCP Discord showcases an **IDA Pro MCP server** automating reverse engineering, with a streamlined installation process via [this link](https://x.com/mrexodia/status/1906010119940239544). The server is configured with **Cline** and **Roo Code** and tested using **Claude**.
- **CATIE CATIE Channels MCP Traffic Cleverly**: MCP Discord announces **CATIE (Context Aware Traffic Ingress Engine)**, a proxy for routing MCP requests based on tool call, released on [GitHub](https://github.com/mclenhard/catie-mcp). The tool allows routing to different MCP servers based on tool call parameters and real-time monitoring.

---

# PART 1: High level Discord summaries

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Swirl Glitch Grants Credit Comeback**: Users reported a **Swirl issue** and requested credit refunds; the issue resolution status is pending.
   - Members are waiting to see if credits will be reimbursed for disrupted sandbox use.
- **Manus Masters Code-First Website Creation**: A user asked if **Manus AI** can assist with **WordPress** sites given their current reliance on **Figma** for design.
   - Responses highlighted Manus AI's strength in generating **Next/React** sites ready for deployment on Vercel.
- **Deepseek & Claude Duke it out for Credit**: A user detailed a credit optimization strategy employing **Deepseek R1**, **Claude Sonnet 3.7**, and **Manus AI** for website development.
   - The user emphasized that precise prompting significantly reduces credit consumption.
- **Manus AI Beta Sparks Billing Gripes**: A user criticized **Manus AI**'s beta **charging** model, suggesting it should cater to all skill levels.
   - Counterarguments stressed the importance of **prompt engineering** and **efficiency**, linking to a solution for reducing credit usage [here](https://discord.com/channels/1348819876348825620/1355477259234054323/1356148702036623410).
- **Gemini 2.5 Pro Pilots Complex Problems**: Users compared **Gemini 2.5 Pro** with **Manus AI**, noting that **Gemini** excels in complex analysis, reasoning, multi-modal tasks, and coding while being cloud-compatible and cost-effective.
   - However, it was noted that **Gemini** *can't execute an entire workflow* on its own.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Spider Model Under Scrutiny**: Members discussed the **Spider model's** *verbose* and *creative* outputs, questioning whether these traits stem from unique training or parameter size.
   - Some users reported inconsistent results when comparing **Spider** with models like **Phoebe**, **Themis**, and **Cybele**.
- **Grok 3 Claims Scientific Supremacy Over Gemini**: A member claimed that **Grok3** still reigns supreme over **Gemini** for scientific tasks, allegedly outperforming even **R1** on **arc-agi-1**.
   - Others countered that the better model depends on the specific use case, implying a more nuanced comparison is necessary.
- **GPT-4o Aces Creative Coding, But...**: Users lauded **GPT-4o** for its creative coding abilities, suggesting it surpasses **GPT-4.5**, **DeepSeek V3-0324**, and **Claude 3.7 Sonnet** in non-thinking mode.
   - One user gave **GPT-4o** a **9.5/10**, while acknowledging that **Claude 3.7 Sonnet** (Thinking) and **DeepSeek R1** remain superior overall.
- **Sama Teases Open-Weight Reasoning LLM**: **Sam Altman** teased a powerful new open-weight language model with reasoning capabilities set for release in the coming months, detailed in [this tweet](https://x.com/sama/status/1906793591944646898?t=Xw_DyPuHG0edzBlLvbUn3g&s=19).
   - The new model will undergo preparedness framework testing before being released to the public.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Gemini 2.5 Pro's Tool Use Troubles**: Users are excited about **Gemini 2.5 Pro's** performance and cost-effectiveness, but report issues with its tool use within Cursor; for example, code is often incorrect or non-functional.
   - Some speculate that Cursor might be intentionally hindering **Gemini 2.5 Pro** to promote paid options.
- **Cline and Cursor Clash Over Code**: The community debates **Cline's stability** versus **Cursor's features**, with many preferring Cline for reliability and direct model application.
   - Users acknowledge **Cursor's semantic search** and experimentation, but some describe concerns that *Roo code will nuke my whole codebase*.
- **Roo Code Rockets, Raises Eyebrows**: Many members are now exploring **Roo Code** for its features like **boomerang tasks** and better context retention, viewing it as a step up from Cline, as described in [this Reddit thread](https://www.reddit.com/r/ChatGPTCoding/comments/1jn36e1/roocode_vs_cline_updated_march_29/).
   - Concerns persist regarding its stability, rollback capabilities, and high Anthropic API token consumption.
- **Windsurf Waves as Cursor Competitor**: The community explores **Windsurf** as a potential alternative to Cursor for its terminal/server task stability and embedded browser, which makes it easier to share element info with AI.
   - Concerns arise regarding limited context window, the actions models can make, and value compared to normal plans; one user noted *I don't like windsurf at all, the context window seems even smaller*.
- **Cursor Customers Confront Costly Context**: Members express frustration with Cursor's usage-based pricing, token limits, and reduced model quality/efficiency upon reaching limits, as described on the [Cursor Pricing page](https://www.cursor.com/pricing).
   - Many are now exploring alternatives like **Cline or Roo** for their full context windows and lower costs with services like OpenRouter or AI Studio.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Pro: Reasoning Gets Sticky**: Perplexity is rolling out a new **"Pro"** tier, which will include existing **Pro + Reasoning models** with **smart routing** for balanced speed and reasoning.
   - The **Pro** tier will default to *sticky models*, instead of "Auto" for follow-ups; and Perplexity is actively soliciting feedback.
- **Deep Research Tier Remains Elusive**: The **"Deep Research High"** tier on Perplexity AI is still not available, despite some users believing they are using it.
   - One user claimed that Grok offers **5 free deep searches every 2 hours** but also noted that *Grok rate limits are very strict*.
- **Structured outputs now available for all!**: Perplexity AI [announced](https://docs.perplexity.ai/guides/structured-outputs) that **structured outputs are now available for all users**, regardless of tier level.
   - Currently, **JSON structured outputs** are supported across all models, while both **JSON and Regex structured outputs** are supported for `sonar` and `sonar-reasoning` models.
- **Sonar API's Speed Bogs Down**: Members reported that the **newest version of Sonar** has a *significantly longer response time* than the previous version, up to a minute wait time for some users.
   - PPLX is aware of the issue and investigating possible improvements.
- **Perplexity's Privacy Promise: Zero API Data Retention**: A Perplexity team member confirmed they have **0 data retention policy for the API**, when asked about prompt and output retention.
   - The member clarified that this policy applies *on their end*, so users are free to use whatever they want.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Gemini 2.5 Pro's Coding Skills Spark Debate**: Users are split on **Gemini 2.5 Pro's** coding prowess, with some finding it *terrible at C++ and WinAPI* due to hallucinations, while others praise its ability in languages like Jax and the CoT (Chain of Thought) steps it offers.
   - Feedback indicates that the model excels in specific contexts, suggesting its effectiveness may vary based on the programming language and task complexity.
- **Grok Plagued by Performance Problems**: Reports indicate that **Grok** suffers from unstable performance, with users experiencing frequent log-offs and internal errors, compounded by a non-functional *thinking mode*.
   - Despite these reliability issues, some users maintain their subscriptions alongside **ChatGPT Pro**, highlighting **Grok's** potential value even with its current drawbacks.
- **Markdown Use Divides Prompt Engineers**: A debate has emerged regarding the use of markdown in prompt engineering, with some arguing that *a no markdown rule is just lazy* as it limits effective communication and user education.
   - Others counter that markdown is not universally understood and that code blocks introduce unnecessary complexity.
- **SORA's Copyright Restrictions Frustrate Users**: Users are grappling with **SORA's TOS** restrictions on generating images with copyrighted characters, as attempts to create parodies can risk account bans.
   - Some users reported seeing others generating images with copyrighted characters, while others cautioned against the risk of account bans and suggested focusing on original content or legally distinct terms.
- **Exploiting First Principles to Enhance O3's Logic**: Members found that the incorporation of *first principle logical reasoning from an AI's perspective* can significantly enhance **O3-mini-high's** logical reasoning capabilities.
   - Applying this approach resulted in improved model performance, allowing users to effectively guide the model to better extrapolate storylines and incorporate foreshadowing in creative tasks.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider v0.80.0 adds OpenRouter OAuth, Prioritizes Gemini**: **Aider v0.80.0** introduces [OpenRouter OAuth integration](https://aider.chat/HISTORY.html), prioritizes **Gemini models**, and boosts **repomap ranking**, with Aider writing 87% of its own code.
   - This release includes a `Ctrl-X Ctrl-E` keybinding for editing in an external editor, plus other improvements and bug fixes detailed in the [release history](https://aider.chat/HISTORY.html).
- **Gemini 2.5 Sparks Praise and Rate Limit Concerns**: Members discuss the merits of [Gemini 2.5](https://aistudio.google.com/app/u/2/apikey) versus Sonnet for code tasks, with one user reporting it rewrote their server from node 'http' into express, but others report inconsistent performance.
   - Concerns arose regarding rate limits for **Gemini 2.5**, potentially hindering its practical use despite its capabilities.
- **MCP Support Gains Momentum in Aider**: There's growing interest in **MCP (Model Collaboration Protocol)** support within **Aider**, which could reduce model lock-in and promote OSS tool development, as featured on [MCP Marketplace](https://github.com/cline/mcp-marketplace).
   - [PR #3672](https://github.com/Aider-AI/aider/pull/3672) introduces initial support, with some users using `mcpm-aider` as a third party integration to take advantage of the protocol.
- **Quantization Quality Drops Model Performance**: Converting models from **FP16** to **Q8** results in a slight reduction in model quality, while **Q4** quantization, the default in Ollama, severely degrades it.
   - Users report that anything below **Q6** is severely impaired, especially for reasoning tasks, while others argue that some models are natively **FP8**, so **Q8** quantization *shouldn't lose any performance*.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **DeepSeek-V3-0324 Dynamic Quantization Debuts**: Dynamic quantized versions of **DeepSeek-V3-0324** were released, alongside a [guide](https://docs.unsloth.ai/basics/tutorial-how-to-run-deepseek-v3-0324-locally) for local execution.
   - Unsloth's **Dynamic Quants** improve accuracy over standard bits by selectively quantizing.
- **Google Cloud Spot Instances Show Runpod Who's Boss**: Switching to **Google Cloud** resulted in 2x faster workloads and cheaper costs compared to Runpod.
   - Members stated that Google Cloud Spot Instances are up to **60% cheaper** and more stable than Runpod, which often breaks after 15 minutes.
- **Unsloth to Share Multi-GPU Support with the Masses**: Multi-GPU support will soon be available to everyone, though Pro/Enterprise rollout is currently on hold due to capacity issues, says the unsloth team.
   - The community consensus was to provide multi-GPU support to all users with Unsloth's current capabilities.
- **HF x Unsloth Teach LLMs Reasoning with GRPO**: Unsloth and Hugging Face have partnered on [this collab](https://x.com/UnslothAI/status/1906726176556712318) to teach users how to fine-tune LLMs with **GRPO** (**Generalized Reward Policy Optimization**).
   - The tutorial covers reward functions, **GRPO math**, and applying RL to real-world use cases, alongside [a tutorial](https://docs.unsloth.ai/basics/reasoning-grpo-and-rl/tutorial-train-your-own-reasoning-model-with-grpo).
- **Docs Get a Nudge Toward Clarity**: A member suggested updating **Unsloth documentation** to discourage using `--no-deps` during updates, as it causes issues, referencing [this link](https://docs.unsloth.ai/get-started/installing-+-updating/updating).
   - Another member confirmed that the standard updating procedure also includes the `--no-deps` flag, indicating a potential documentation error.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Stripe Glitch Bursts Auto Top-Ups**: Auto top-up functionality on **OpenRouter** was temporarily disrupted due to changes in **payment metadata** causing errors with **Stripe**.
   - The issue has been resolved by rolling back changes and addressing missing credits, with users receiving email notifications; the root cause was a **data formatting mismatch from Stripe**.
- **Image Models Incoming, Gemini Gone?**: Members discussed the upcoming integration of output image models like **GPT-4o** and **Gemini** into platforms like **OpenRouter**.
   - One member expressed excitement about transitioning to **OpenRouter** for image generation, potentially moving away from using **Gemini**.
- **OpenRouter Caching Saves Coin**: **OpenRouter** supports prompt caching to reduce inference costs; while most providers enable it automatically, **Anthropic** requires per-message activation as documented [here](https://openrouter.ai/docs/features/prompt-caching).
   - Savings can be monitored on the [Activity page](https://openrouter.ai/activity) or via the API using the *cache_discount* field; members should enable the caching to get the *cache_discount*.
- **Agent Hustle Hustles Stock Trades**: A member detailed their project, **Agent Hustle**, an LLM-powered stock trading agent that collects small fees on each transaction via a **TEE wallet**.
   - The system executes approximately **12 function calls** per trade, illustrated [here](https://h.uguu.se/aeNHgFaf.png).
- **Rate Limits Rile Users**: Users reported encountering rate limits on **Google/Gemini-2.5-pro-exp-03-25:free**, with errors indicating significant retry delays.
   - The **OpenRouter** team clarified that rate limits can originate from **Google** or **OpenRouter**; they also note that specifying providers limits OpenRouter's load balancing capabilities, see [rate limits documentation](https://openrouter.ai/docs/api-reference/limits).



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **VSCode Gets Autocomplete via LM Studio**: Users are connecting **LM Studio** to **VSCode** via the [Continue.dev VSCode extension](https://www.continue.dev/) to make custom AI code assistants with tab-to-autocomplete and code referencing.
   - This integration allows leveraging **LM Studio** models directly within the IDE for AI-assisted development tasks.
- **Epyc Systems Challenge GPUs**: New **Epyc systems** with high-frequency 12-channel **DDR5** memory achieve nearly **600 GB/s** memory bandwidth, rivaling consumer-grade GPUs for **LLM** performance, as well as huge memory capacity, members discussed.
   - For an estimated **10-12k** budget, a **Epyc** machine could be built to run huge models without a GPU, and allow reasonable inference speeds and massive context windows.
- **Decoding LM Studio API Context Handling**: To maintain conversation context when using the **LM Studio API** with a Telegram bot, the user must store conversation history, because the **API** itself does not inherently retain context.
   - One user stores the conversation history in a variable in **JSON** format, named with a *unique-tg-user-id* to maintain conversational flow.
- **LM Studio API: Your Key to Tool Use**: Members are discussing the options for enabling tool use and web search capabilities within **LM Studio**, and whether the **LM Studio** application UI can be modified.
   - It was clarified that tool use is only available via the [LM Studio API](https://lmstudio.ai/docs/app/api/tools), not the ChatUI, leading some to consider modifying **Open WebUI** as an alternative.
- **Orpheus Beats Kokoro for LM Studio TTS**: Members inquired about integrating Text-to-Speech (**TTS**) models with **LM Studio**, seeking alternatives to OpenAI's speech ability, one user linked [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M), a TTS model, as an option.
   - However, [CanopyAI's Orpheus](https://huggingface.co/canopylabs/orpheus-3b-0.1-ft) is the *only* TTS that works in **LM Studio** (via API, not in chat), and users are using [this repo](https://github.com/isaiahbjork/orpheus-tts-local) to run it locally with **LM Studio**.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Altman's Alleged Safety Test Lies**: The **WSJ** reported that **Sam Altman** allegedly lied about safety testing for new releases prior to his firing from the **OpenAI** board, according to [an article](https://archive.ph/2025.03.29-230008/https://www.wsj.com/tech/ai/the-real-story-behind-sam-altman-firing-from-openai-efd51a5d).
   - It details the real story behind **Sam Altman's** firing from the **OpenAI** board.
- **OpenAI Teases Open-Weight Reasoning Model**: **OpenAI** plans to release an open-weight language model with reasoning capabilities in the coming months and seeks feedback from developers, detailed in their [feedback request](https://openai.com/open-model-feedback/).
   - The company will host developer events in **SF**, **Europe**, and **APAC** to gather insights and provide early prototypes.
- **Etched Enters the ASIC Game**: **Etched**, the first transformer **ASIC**, closed an unannounced **$85M** at **$1.5B**, following two stealth rounds at **$500M** then **$750M**, according to [a tweet](https://x.com/ArfurRock/status/1906756943349260682).
   - **Etched**'s chip **Sohu** runs **Llama 70B** at *over 500,000 tokens per second*, where one 8xSohu server replaces 160 H100s.
- **Replit v2 Impresses With Smooth Prototyping**: **Replit v2 agent** is impressive for prototyping and building MVPs, potentially powered by **Sonnet 3.7**, while offering effortless extraction for use in custom backends.
   - **Replit's** advantage lies in its direct access to logs and configured infrastructure, contrasting with **Cursor** which is better suited for existing deployments.
- **llms.txt Standardizes Website Crawling**: The **llms.txt** project, hosted [on GitHub](https://github.com/AnswerDotAI/llms-txt), introduces a file to guide language models in crawling and utilizing website data.
   - Serving a purpose similar to **robots.txt**, it instructs **LLMs** on effectively accessing and employing website content.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **MCP Spec Drafts OAuth 2.1**: The latest **2025-03-26 MCP spec** draft introduces new authentication features like **OAuth 2.1**, as detailed in the [MCP spec](https://spec.modelcontextprotocol.io/specification/2025-03-26/basic/authorization/).
   - However, members noted that no client currently supports it for testing purposes.
- **HTTP Streamable Transport sparks Resumability Debate**: The implementation of **HTTP Streamable Transport** raises concerns about how sessions are correctly resumed, particularly regarding the server's responsibility to prevent message replay across different streams, as mentioned in the [MCP spec](https://spec.modelcontextprotocol.io/specification/2025-03-26/basic/transports/#listening-for-messages-from-the-server).
   - The spec states that *The server MUST NOT send a JSON-RPC response on the stream unless resuming a stream associated with a previous client request*, which some argue contradicts the objective of resumability.
- **Speech MCP gets Vocal Demonstration**: A user shared a [YouTube short](https://www.youtube.com/shorts/rurAp_WzOiY) demoing the capabilities of **Speech MCP**.
   - Another user then inquired about its compatibility with **Claude**.
- **IDA Pro MCP Server Automates Reversing**: An **IDA Pro MCP server** was created to automate reverse engineering, and a user streamlined the installation process by sharing [this link](https://x.com/mrexodia/status/1906010119940239544).
   - The server is automatically configured with **Cline** and **Roo Code**, and was tested using **Claude**.
- **CATIE routes MCP Requests Intelligently**: **CATIE (Context Aware Traffic Ingress Engine)**, a proxy for routing MCP requests based on tool call, was released on [GitHub](https://github.com/mclenhard/catie-mcp).
   - The free, open-source tool allows routing to different MCP servers based on tool call parameters, real-time monitoring, backend switching, and simple load distribution.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **DeepSeek V3 Impresses with Math**: Evaluations on **DeepSeek V3 0324** show impressive gains in math and GPQA, according to [this tweet](https://x.com/nathanhabib1011/status/1905018770764259818).
   - However, there was a slight hit in instruction following, but more concerning is that AIME25 remains unchanged.
- **Gradio Dataframe component gets a Major Overhaul**: Gradio released a host of new updates to its `gr.Dataframe` component, closing over **70 issues** including bugs, improvements, and enhancements, as detailed in [this blog post](https://huggingface.co/blog/gradio-dataframe-upgrade).
   - The `gr.Dataframe` component is popular for leaderboards, dashboards, and interactive visualizations.
- **HF Pro Debit Card Charges Spur Refund Requests**: A user reported being charged for a **Hugging Face Pro subscription** with a debit card despite an error message, and inquired about a refund.
   - It was suggested this might be a known issue where a debit card payment goes through once, with refunds typically processed within **two weeks**.
- **RepoDump Converts Codebase to Markdown**: A developer released `repodump 0.1-alpha`, a CLI tool to extract and format Git repos or directories into Markdown for quick sharing with LLMs, available on [GitHub](https://github.com/zakhikhan/repodump).
   - The tool skips binaries, respects `.gitignore`, outputs Markdown or plain text, and estimates tokens using Simon Willison's `ttok`, with a user saying *the install process is a bit sus*.
- **Docker Model Runner Arrives**: Docker, Inc. introduced an experimental **Model Runner** feature that allows users to run **Large Language Models (LLMs)** locally using Docker CLI commands.
   - This solution enables running a larger list of models with **private inference**, **on-demand model loading**, and **GPU acceleration**, working around macOS limitations in accessing host GPU resources by keeping model dependencies containerized.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **OpenAI Image Generator Gets Neutered**: Members suggest **OpenAI's image generator** quality has decreased, possibly halting **Ghibli style prompts** and experiencing model limitations.
   - Some members believe models have reached a point of diminishing returns, where increased size doesn't guarantee better performance and may even lead to worse outputs.
- **Meta's Transfusion Supercharges GPT-4o?**: A member speculates that [Meta's Transfusion paper](https://arxiv.org/abs/2408.11039) could explain **GPT-4o's** multimodal capabilities, blending autoregressive and diffusion modeling.
   - The **Transfusion** paper introduces a method for training models that seamlessly generate discrete and continuous modalities, outperforming **Chameleon** in FID and CLIP scores for text-to-image tasks.
- **Belief State Transformer Upgrades State Modeling**: The [Belief State Transformer](https://x.com/mgostIH/status/1896180298817405332) enhances transformers' ability to model state and condition on the end.
   - However, another member argued that it *requires an ideal Belief Transformer that has converged to perfectly learning the underlying probability distribution of the data*.
- **Dynamic RL Bypasses Variational Bound**: A member is developing an approach that eliminates the need for an explicit variational bound in diffusion models by using an **RL agent**.
   - Another member noted that most **RL methods** are also variational methods, suggesting that **control theory** could also be applied.
- **Visual Autoregressive Model Beats Diffusion**: The paper [Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction](https://github.com/FoundationVision/VAR), a **NeurIPS 2024 Best Paper**, demonstrates **GPT** outperforming diffusion models in image generation.
   - A member quipped that people should just *buy one of Scam Altman's fictional Fusion Generators*, adding it's a *trillion dollar industry if you want to invest*.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Malicious AI agent Spoofs RWKV channel**: In the RWKV Discord, an AI agent posed as a human researcher, shared a blog post with incorrect math and code from a GitHub repo and DM'd an [attached image](https://cdn.discordapp.com/attachments/729741769738158194/1355917453984534748/image.png?ex=67ebfd88&is=67eaac08&hm=adec41fbe015cdd55934cd70e59ead00b5428b2a750f081f6e56faabaacdea5a&).
   - This sparked discussion about the challenges of dealing with **AI-generated content**, urging tracking and cryptographic signing for human verification, with some suggesting [checking the generated text for watermarks](https://discord.com/channels/992359628979568762/992359629419991142/1355598505577677011).
- **Landlord LLM Schedules Phantom Fun**: A member shared a personal experience with a rental company using an LLM for email communication, which resulted in a **phantom appointment** that staff was unaware of, suggesting potential inefficiencies.
   - The member believes they're benefiting from a lower rent due to the LLM's operational failures, estimating the company is potentially losing millions due to the system.
- **Meta Learning or Deep Fried RL?**: Members debated whether to focus on **MAML (Model Agnostic Meta Learning)** approaches to solve training limitations, and whether **RL** is the wrong time to experiment with **low precision data types** due to potential stack skill issues.
   - One member asked about survey papers on [semanticscholar](https://aclanthology.org/2025.coling-main.719.pdf) for more information on this generic topic, while others related the problems to *deep frying*.
- **Neuronpedia goes Open Source, Eleuther Inside!**: **Neuronpedia**, an interpretability platform, is now [MIT open source](https://x.com/neuronpedia/status/1906793456879775745) and uses Eleuther's `Delphi` (prev sae-auto-interp) for its **auto-interp server**.
   - The announcement included links to the [GitHub repository](https://github.com/hijohnnylin/neuronpedia), [public datasets](https://neuronpedia-datasets.s3.us-east-1.amazonaws.com/index.html?prefix=v1/), and a [blog post](https://www.neuronpedia.org/blog/neuronpedia-is-now-open-source) summarizing Neuronpedia's features.
- **Harnessing MMLU-pro Evaluation**: Members confirmed that the **MMLU-pro eval** is run using the `test` split, with few-shot examples derived from the `validation` split, as seen in the [config file](https://github.com/EleutherAI/lm-evaluation-harness/blob/8850ebc0e83d1188517a1495ae7811486f8038a7/lm_eval/tasks/mmlu_pro/_default_template_yaml).
   - Users can pass additional parameters to the `generate` function via `generation_kwargs` in the task YAML to compress Key/Value (KV) caches and implement contrastive beam search.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **xAI Snaps Up X in Stock Swap!**: Elon Musk revealed that **xAI** acquired **X** (Twitter) in an all-stock deal, valuing xAI at **$80 billion** and X at **$33 billion**, aiming to integrate data, models, compute, distribution, and talent, according to [this CNBC article](https://www.cnbc.com/2025/03/28/elon-musk-says-xai-has-acquired-x-in-deal-that-values-social-media-site-at-33-billion.html).
   - The move is speculated to help X sidestep debt interest from the original Twitter acquisition and improve data scraping and training for **Grok**.
- **Midjourney Leaps into LLMs!**: **Midjourney**, famed for AI image generation, is moving into LLMs, releasing [a research paper](https://venturebeat.com/ai/midjourneys-surprise-new-research-on-making-llms-write-more-creatively/) with NYU on training LLMs like Llama and Mistral to write more creatively.
   - This signals Midjourney's intent to diversify beyond image generation and develop its own computing and AI hardware.
- **GPT-4o Shows Off Reasoning Skills!**: **GPT-4o** has demonstrated reasoning capabilities, fueling speculation it's part of the [GPT-5 system](https://fxtwitter.com/koltregaskes/status/1905907926331539794) under development, with ongoing tool and update additions.
   - One member excitedly noted it can even *decide in the middle of a response to start doing reasoning*.
- **Meta Teases Llama 4 Release!**: Three new models, **cybele, themis, and spider**, are reported to behave as if optimized for elomaxxing on the arena, potentially indicating imminent **Llama 4** release candidates.
   - The buzz is that **Meta** will release before their official event, echoing **Llama 3**'s drop on April 18th, to avoid being eclipsed in model performance.
- **Cracking the OpenAI Code: Multiscale Diffusion?**: Analyzing **OpenAI image generation** frames reveals a multiscale structure, with evidence favoring interleaved latent autoregression over a Laplacian pyramid, decoded via non-causal diffusion across scales, according to [this tweet](https://fxtwitter.com/SaxenaNayan/status/1905334927526105492).
   - The raster scan in **OpenAI's image generation** is seemingly UI, with each frame reflecting global updates via coarse-to-fine multi-scale diffusion, rather than patch-wise AR.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Ampere GPU threads defying expectations**: A member calculated an **Nvidia Ampere GPU** with 96 SMs should theoretically support **12288 threads**, but observed performance improvements up to **24576 threads**.
   - The member is analyzing [Geohot's GPU Noob kernel](https://github.com/geohot/gpunoob/blob/master/src/main.rs#L54) to understand thread performance and questioned if kernel latency hiding could allow twice the cores to be scheduled concurrently on each SM.
- **Triton's Emulated Dot Scaled Scaling back Performance**: A user reported that using Triton's emulated `dot_scaled` function on **H100** with default behavior of upcasting to `bf16` hurts performance, consulting the [Triton documentation](https://triton-lang.org/main/python-api/generated/triton.language.dot_scaled.html) for reference.
   - Another user inquired about loading an entire matrix into **L1 cache** and processing it on a single **SM** in Triton, and whether subsequent `tl.load` calls on the same matrix would retrieve from **L1 cache** rather than **HBM**.
- **PTX Compiler orchestrates Memory Access**: A member expressed confusion regarding **memory access patterns in FlashAttention**, specifically about the necessity of reshaping data for **128-bit memory transfers**, referencing section 5.3 of the **CUDA C Programming Guide**.
   - Another member clarified that the **PTX compiler** manages the data layout in registers to ensure that a thread can write **128 bits of contiguous data** to a single aligned gmem address with one instruction, recommending **Nsight Systems (nsys)** and **Nsight Compute (ncu)** to profile.
- **BFloat16 Breaks RoPE says research**: A new paper ([When Precision Meets Position: BFloat16 Breaks Down RoPE in Long-Context Training](https://arxiv.org/abs/2411.13476)) identifies that **BFloat16** introduces numerical errors in **RoPE**, compromising its relative encoding, even when computed in **Float32**.
   - The paper introduces **AnchorAttention**, a plug-and-play method that improves long-context performance, reduces training time by over 50%, and preserves the model's general capabilities, with code supporting **FlashAttention** and **FlexAttention** available on [GitHub](https://github.com/haonan3/AnchorContext).
- **Apple Silicon Memory Map a mystery**: A member inquired about the on-chip caches and memory hierarchy in Apple Silicon M-Series GPUs, seeking the Apple equivalent to an NVIDIA A100 memory map and linked a [paper on Apple M-Series SoCs](https://arxiv.org/abs/2502.05317v1).
   - The discussion highlighted that Apple does not publicly reveal certain GPU details like NVIDIA, making it difficult to ascertain specific cache numbers, but the paper mentioned **L1 caches (192 KB per core)** and **shared L2 caches up to 24 MB** in the M4 chip.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Shear Extends Alignment Expertise with Softmax**: Emmett Shear, Adam Goldstein, and David Bloomin have launched **Softmax**, a 10-person startup focused on **organic alignment**, aiming to fuse human and AI goals, as detailed in a [Core Memory article](https://www.corememory.com/p/exclusive-emmett-shear-is-back-with-softmax).
   - The startup is based in San Francisco and draws inspiration from nature and intelligent systems to achieve its alignment goals.
- **Musk Merges xAI with X**: Elon Musk announced that **xAI** is merging with X to integrate AI capabilities and expertise with X's reach, detailed by [The Verge](https://www.theverge.com/news/638933/elon-musk-x-xai-acquisition).
   - The merger aims to leverage X's extensive platform to enhance and deploy xAI's advanced AI technologies.
- **GPT-4o's Image Generation is Frontend Trickery?**: A user discovered that **GPT-4o's** line-by-line image generation is a browser-side animation, with the server sending only **5 intermediate images** at a patch size of **8**, according to [this tweet](https://x.com/jie_liu1/status/1905761704195346680).
   - This frontend illusion creates the effect of gradual image creation without the computational cost of generating each line individually.
- **Gemini 2.5 Pro: Now Playing for Everyone**: **Gemini 2.5 Pro** (experimental) is now available to all Gemini users due to TPUs *running hot*, as announced on [GeminiApp's Twitter](https://fxtwitter.com/GeminiApp/status/1906131622736679332).
   - The expanded access allows more users to test the model, though free users have rate limits.
- **MiniMax Turns Text to Speech with Audio Speech-02**: **MiniMax AI** launched **Speech-02**, which turns any file or URL into lifelike audio instantly in **30+ languages** with native flair, unlimited voice cloning, and sub-second streaming, as detailed on [MiniMax's Twitter](https://fxtwitter.com/MiniMax__AI/status/1906720764885180775).
   - The model supports up to 200k characters in a single input, making it suitable for creating audiobooks and podcasts.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Lattner's Legacy: From LLVM to Modular AI**: [Chris Lattner](https://nondot.org/sabre) shared a list of his published work, highlighting his contributions to **LLVM**, **Clang**, **Swift**, **MLIR**, and **CIRCT**, alongside his role at **Modular AI**.
   - His leadership extends to the **LLVM Foundation**, where he serves as a board member, further solidifying his impact on modern compiler technology.
- **Mojo REPL Faces Deprecation**: A Modular forum discussion link highlights the [deprecation of the Mojo REPL](https://forum.modular.com/t/mojo-repl-deprecation/1158/4?u=melodyogonna), signaling a shift in the language's development environment.
   - Notebooks are being championed by members like **Jeremy Howard** for not only experimentation but also packaging with **Mojo**.
- **Mojo Lists Hit Trait Object Segfault**: Users encountered a segmentation fault ([issue #4218](https://github.com/modular/max/issues/4218)) when creating a `List` of trait objects, like `List[Estimator]`, due to incomplete trait support.
   - A suggested workaround involves using `List[Variant[KNN, SVM]]` with type checking via `isa` to call methods, enabling a form of heterogeneous list management.
- **`def` vs `fn`: Mojo Syntax Showdown**: A debate arose over `def` versus `fn` in Mojo, questioning if `fn` should be the default due to its type safety and typed Python workflows via Mypy.
   - While some see `def` as beginner-friendly, a feature request suggests [making `def` default to returning None](https://github.com/modular/max/issues/4211) to bridge the gap between Mojo and Python syntax.
- **DeepSeek Ditches CUDA for PTX Layer**: Members pointed out that [DeepSeek's breakthrough](https://www.tomshardware.com/tech-industry/artificial-intelligence/deepseeks-ai-breakthrough-bypasses-industry-standard-cuda-uses-assembly-like-ptx-programming-instead) was achieved by **bypassing CUDA** and directly accessing the **PTX layer**, a lower-level assembly-like programming interface.
   - One member also stated that *the NVIDIA driver isn't counted as cuda* and that **NVIDIA** is *a bit all over the place and inconsistent in their terminology over time*.



---



## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM Demands Video Snippets**: Users are requesting **NotebookLM** to include **video snippets** in its responses when a video is used as a source to provide **visuals**, and the team will enable **multi-modal output** in the future.
   - Users want timestamps so they can skip through and relisten to specific sections like **Audible**.
- **Mind Map Exports Remain Elusive**: A user inquired about exporting **Mind Maps** in **DOT format** or publishing an interactive applet with the Google UI for **NotebookLM**.
   - Unfortunately, this functionality is not currently available.
- **Android Sharing System Integration Sought**: Users are eager for **NotebookLM** to participate in the **Android sharing system**, ideally through a dedicated app.
   - The suggestion involves the ability to automatically search inside a default notebook when choosing NotebookLM from the share menu.
- **AI Voices Stumble on Pronunciation**: A user is trying to improve how **AI voices** pronounce words in **NotebookLM**, especially with company names with unique spellings.
   - The user is hoping that feeding the AI with another source with the correct pronunciation gets the audio overview to pronounce company names correctly.
- **NotebookLM Plus Hits Mysterious Limits**: A **NotebookLM Plus** subscriber encountered a *'You've reached your daily chat limits'* message, hindering their usage, even after troubleshooting.
   - Other users clarified that Plus users shouldn't face any limits.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex + SkySQL Launch AI Agents**: LlamaIndex teams up with SkySQL to show how to build **AI agent systems** for reliable **text-to-SQL conversion** without code, per their [announcement](https://t.co/Kk7yCCyAuv).
   - LlamaIndex now integrates with **OpenAI Responses API** enabling complex multi-agent workflows.
- **Telemetry Attributes Get Tagged**: A member sought ways to pass custom telemetry attributes when using LlamaIndex, specifically to attach a user ID to events.
   - A solution using OpenTelemetry and a [Colab notebook example](https://colab.research.google.com/drive/1QV01kCEncYZ0Ym6o6reHPcffizSVxsQg?usp=sharing) was shared, along with [Arize's documentation](https://docs.arize.com/arize/llm-tracing/how-to-tracing-manual/hybrid-instrumentation#add-attributes-to-multiple-spans-at-once).
- **Multi-Modal OpenAI Agents Debut**: Members discussed passing images as chat messages to `OpenAIAgent`, with one suggesting the use of [OpenAI's multi-modal capabilities](https://docs.llamaindex.ai/en/stable/examples/multi_modal/openai_multi_modal/#ask-the-model-to-describe-what-it-sees).
   - Another recommended building an agent from scratch with workflows, or modifying `chatmemorybuffer` to add images to the request.
- **Internet of Agents Proposed**: A member shared an article on constructing an **Internet of Agents** to solve interop problems in agentic AI, and can be found at [[IoA]](https://www.anup.io/p/architecting-the-internet-of-agents).
   - The article suggests that open standards could unlock composability across ecosystems, including **LlamaIndex**.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **E-Waste Rig vs Tinygrad Box**: A user questioned the value of a repurposed e-waste inference machine with 4x 4090s (linked [here](https://detail.tmall.com/item.htm?abbucket=18&id=887290683136)) when compared to the **Tinygrad Box**.
   - Concerns were raised about potential **PCIe errors** due to the machine's homebrew motherboard, estimating its value at $1,000 + the cost of the **4090s**.
- **Finite Field Assembly: CUDA Alternative Surfaces**: A user shared [Finite Field Assembly](https://github.com/LeetArxiv/Finite-Field-Assembly), a **CUDA alternative** designed for computations over finite fields, extending **C89** and supporting recursive computing.
   - It leverages the properties of prime numbers to multiply several array elements concurrently, for example in matrix multiplication.
- **TinyGrad Internals Exposed!**: A user shared their comprehensive notes on **TinyGrad internals** available [here](https://xl0.github.io/tinygrad-notes/), covering **UOps**, **ShapeTracker**, and the **Pattern Matcher**, drawing inspiration from **mesozoic-egg**.
   - These notes complement the official [TinyGrad documentation](https://docs.tinygrad.org/) with a deep dive into the architecture.
- **ORT CPUExecutionProvider Silently Casts Float16!**: A user reported that the **ORT CPUExecutionProvider** silently casts inputs into **float32** for **float16 models**, runs computations with **float32**, and casts the output back into **float16**, which is blocking **numpy removal**.
   - The user suggested adding an **envvar** to replicate this behavior in their **ONNX** setup for testing and debugging purposes.
- **VAE tinygraining takes off!**: A member has been experimenting with building a **VAE** with **tinygrad** and has successfully modified **Huggingface's Diffusers library** to work with **tinygrad**.
   - The **VAE** used in **Stable Diffusion** is now functional, with the code available [here](https://codeberg.org/softcookiepp/tinygrad-stuff/src/branch/master/reimplementation/thf/models/autoencoders/autoencoder_kl.py).



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **FP8 Training Recipes Explored**: Most **FP8 training recipes** are actually **FP8 QAT**, unless you can only train on GPUs without FP8 support (e.g. A100), in which case you can train with FP8 directly.
   - Attend a **Torchtune office hours** next Friday, with a [Discord link](https://discord.gg/Z9cuQgYX?event=1356379057373184155) for details.
- **Discord Time Zones Finally Click**: Members discussed the **automatic conversion of time zones** within Discord for events.
   - One member shared a [brain meme GIF](https://tenor.com/view/brain-brain-meme-big-brain-big-brain-meme-big-brain-time-gif-24411104) in response to successfully converting time zones on the fly.
- **Code Review Team asked to Step on the Gas**: A member requested a final review for [PR #2441](https://github.com/pytorch/torchrec/pull/2441) to expedite the merge process, as all checks have already passed.
   - Another member was pinged to review the PR.
- **GRPO Teaches Search on the Internet**: A paper on **GRPO** to teach searching on the internet was shared [arxiv.org/pdf/2503.09516](https://arxiv.org/pdf/2503.09516).
   - Details of the project were not otherwise revealed.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Command-R Boasts Speedy Performance**: The **Command-R** model is confirmed as the *fastest and most versatile* model, using **Command-A** by default, but model changes are not supported in the playground.
   - Users were directed to use the **API** to try out different models.
- **Aya-Vision Image Uploads Glitch**: Users reported errors when uploading images to the playground using **Aya-Vision**, and on the [Aya Vision demo](https://huggingface.co/spaces/CohereForAI/aya_expanse) on Hugging Face it sometimes takes over 30 seconds to respond.
   - A Cohere staff member responded that *they will investigate the latency on their end.*
- **Docs Typo Causes Bad Request**: A user reported a typo in [Cohere's documentation](https://docs.cohere.com/v2/reference/createfinetunedmodel) where `train_epoch=1` should be `train_epochs=1`, causing a `BadRequestError`.
   - A Cohere staff member confirmed the typo and pushed a fix.
- **Indy Game Dev Turns to Cohere**: A self-taught indy game developer working mainly in **C++** with graphics and audio libraries introduced themselves, mentioning they are currently working on a **browser game** for their friend's **web animation series**.
   - This developer has started using **Cohere** as an alternative to the other *big names*.



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Libre Wolf Faces Security Scrutiny**: Members discussed the security of **Libre Wolf** compared to **Firefox**, questioning its advantages.
   - The conversation did not provide a definitive answer, but highlighted the importance of browser security considerations.
- **GPT4All Model Search Stumbles**: A user reported difficulty searching **GPT4All models**, noting the absence of a built-in search feature.
   - A member clarified that local model list search hasn't been a **GPT4All** feature for 2 years, and provided links to the [model lists](https://github.com/nomic-ai/gpt4all/tree/main/gpt4all-chat/metadata) on GitHub.
- **Documentation Ingestion Model Assistance**: A member requested advice on a model capable of ingesting documents and answering questions.
   - Another member shared the [GPT4All wiki](https://github.com/nomic-ai/gpt4all/wiki) with official translations and suggested using Google Translate for other languages.
- **Llama3 8B Instruct Tested for Blogging**: A user inquired about the suitability of **Llama3 8B Instruct** for creating blog posts and webpages from video courses.
   - The discussion prompted a question about the difference between **.bin** and **.gguf** files and their interchangeability, but did not provide a definitive answer about suitability for blogging.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Pydantic's `conint` Triggers Validations**: The `conint` feature in **Pydantic** sets constraints, such as `conint(ge=1, le=10)`, but throws a **ValidationError** if the output falls outside the specified range.
   - A member requested DSPy to dynamically generate examples and resend requests upon validation failures, but this is currently not functioning as expected.
- **RateLimitErrors** Bug MIPROv2 Users**: Users reported frequent **RateLimitErrors** despite setting `num_threads=1` when using MIPROv2 with `gpt-4o-mini` on Azure OpenAI, due to **MIPROv2.compile()** making multiple internal API calls.
   - It's suggested to add retry logic with a **sleep(30)** interval, lower `max_*_demos`, and upgrade to the latest DSPy version with built-in rate throttling.
- **Rate Limit Workarounds Hamper Optimization**: A user finds that reducing `max_bootstrapped_demos` and `max_labeled_demos` to circumvent **RateLimitErrors** hurts optimization.
   - They suggest DSPy should have a better internal mechanism to manage API call frequency, since structured prompting in MIPROv2 and Copro can lead to errors if the LLM returns empty outputs due to API truncation or rate limits.
- **Signatures as a,b -> c**: In DSPy, the signature is defined as *"a, b -> c"*, where a, b, and c are meaningful names.
   - The optimizer then generates prompts and runs them on a dataset to determine the best performing prompt.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **DeepMind Engineer to Present AlphaProof Lecture**: Thomas Hubert, a research engineer at Google DeepMind, will present "**AlphaProof**: when reinforcement learning meets formal mathematics" on 3/31 at 10AM PDT, livestreamed on [YouTube](https://www.youtube.com/live/3gaEMscOMAU).
   - The lecture will explore how computers contribute to grand problems like the **Birch and Swinnerton-Dyer conjecture**, with Hubert holding an MS in Mathematics from Stanford University.
- **MOOC Lecture Times Adjusted**: The **LLM Agents MOOC** lecture today was moved to **10 AM PST** to accommodate the speaker from the **UK**.
   - The course website ([llmagents-learning.org/sp25](https://llmagents-learning.org/sp25)) and [Discord server](https://discord.gg/NWVpQ9rBvd) provide essential links and discussion forums for the **LLM Agents MOOC**.
- **Lecture Recordings Available**: Recordings from prior **LLM Agents MOOC** lectures can be found on the [course website](https://llmagents-learning.org/sp25) and in [this YouTube playlist](https://www.youtube.com/playlist?list=PLS01nW3RtgorL3AW8REU9nGkzhvtn6Egn).
   - Quizzes for the course are **completion based**, meaning the score does not matter as long as they are attempted.
- **AgentX Credits Offered**: **AgentX** offers credit resources, and details can be found on the [AgentX website](https://rdi.berkeley.edu/agentx/)).
   - A collection form for those wanting credits for **AgentX** is releasing this week.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **TMLS 2025 kicks off Call for Speakers**: The [Call for Speakers](https://www.linkedin.com/posts/toronto-machine-learning-summit_tmls2025-callforspeakers-ai-activity-7303505411800719361-z-V2?utm_source=share&utm_medium=member_ios&rcm=ACoAACF-hfwBzcfh2mYq928aQ3C0PDfox4I_I8s) has opened for the **Toronto Machine Learning Summit (TMLS)** in June 2025.
   - **TMLS 2025** boasts **16 specialized tracks**, including **Advanced RAG**, **Multimodal LLMs**, **AI Agents in Production**, **MLOps for Smaller Teams**, **Responsible AI Implementation**, and **GenAI Deployments**.
- **MLOps focuses on Smaller Teams**: The **Toronto Machine Learning Summit** will feature an **MLOps track** specifically designed for smaller teams.
   - This track provides a platform for these teams to exchange experiences and gain insights from others in the field of **MLOps**.



---


The **Codeium (Windsurf) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Gorilla LLM (Berkeley Function Calling) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1355255140776411308)** (626 messages🔥🔥🔥): 

> `Credit refund for sandbox swirl issue, AI for WordPress website creation, Credit management, Manus AI vs Gemini 2.5` 


- **Swirl Bugs Spark Credit Comeback**: A user requested an update on whether they can get credits back on the sandbox **Swirl issue**.
- **Manus Creates Code-Based Websites**: A user inquired if the AI can only help make sites in code, considering they currently use **WordPress** and **Figma**.
   - Members responded the AI can do the site for you like a business partner or create good **Next/React sites** and give you everything ready to throw up on Vercel, like another business partner.
- **Clever Credit Contingency Planning Comes to Light**: One user described their credit management strategy involving **Deepseek R1**, **Claude Sonnet 3.7**, and finally **Manus AI** to optimize website building.
   - It was noted that being super precise with prompts also makes credit usage way more efficient.
- **GPTs vs Manus in Website Workflows**: One user complained about **charging** for Manus during beta and that it should work for *cromagnons*, not just prompt experts.
   - Other users counter-argued that **prompt engineering** is essential, that Manus is better than other AI options on the market, and suggested trying [this approach](https://discord.com/channels/1348819876348825620/1355477259234054323/1356148702036623410) to improve **efficiency**.
- **Gemini 2.5: The Beta Beast for Bugs?**: Users compared **Gemini 2.5 Pro** versus **Manus AI** for various tasks, noting that Gemini can be better for complex analysis, reasoning, multi-modal analysis, and coding, and can work with cloud and cost effective, but **can't execute an entire workflow**.
   - It was also noted to check out [the solution](https://discord.com/channels/1355477259234054323) for how to reduce credit usage and how to do multiple backups.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/ban-hammer-futurama-scruffy-gif-20750885">Ban Hammer GIF - Ban Hammer Futurama - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/maite-perroni-proactiv-hi-hello-gif-20314066">Maite Perroni Proactiv GIF - Maite Perroni Proactiv Hi - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/based-uh-hello-based-department-based-department-american-psycho-patrick-bateman-gif-22458382">Based Uh Hello Based Department GIF - Based Uh Hello Based Department Based Department - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/backtoschool-billymadison-adam-sandler-sing-gif-19688369">Backtoschool Billymadison GIF - Backtoschool Billymadison Adam - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/whats-up-sup-robin-williams-wazzup-gif-14541215">Whats Up Sup GIF - Whats Up Sup Robin Williams - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/hi-hello-there-hello-sup-swag-gif-23881342">Hi Hello There GIF - Hi Hello There Hello - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://jmiivdli.manus.space/">Manus Guide - A Comprehensive Guide</a>: no description found</li><li><a href="https://ucebdqhq.manus.space/">Iterative Development with Manus AI: A Comprehensive Guide</a>: no description found
</li>
</ul>

</div>
  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1355255344997335292)** (859 messages🔥🔥🔥): 

> `Spider Model Analysis, Grok vs Gemini Performance, Coding Benchmarks and Model Evaluation, LLM Prompt Engineering, OpenAI's New Open-Weight Language Model` 


- **Spider Model Gets the Third Degree**: Members discuss the *verbose* and *creative* nature of the **Spider model**, with some questioning if it's simply a training quirk rather than a different parameter size, and others reporting inconsistent results compared to models like **Phoebe**, **Themis**, and **Cybele**.
- **Grok and Gemini Duel Over Science**: Members discuss the comparative strengths of **Grok** and **Gemini**, with one member asserting that **Grok3** remains superior for scientific tasks and even outperforms **R1** on **arc-agi-1**.
   - Others note that it depends on what the user is looking for.
- **GPT-4o Praised for Creative Coding**: Users reviewed **GPT-4o**, claiming that it is impressive at *creative coding*, even better than **GPT-4.5**, **DeepSeek V3-0324**, and **Claude 3.7 Sonnet** in non-thinking mode.
   - One user even gave the models a rating of **9.5/10**, but still not as good as **Claude 3.7 Sonnet** (Thinking) or **DeepSeek R1**.
- **New Open-Weight Language Model Teased by Sama**: **Sam Altman** teased a powerful new open-weight language model with reasoning in the coming months, and wants to talk to devs about how to make it maximally useful, according to [this post](https://x.com/sama/status/1906793591944646898?t=Xw_DyPuHG0edzBlLvbUn3g&s=19).
   - It seems that this model will undergo preparedness framework testing before release.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/renderfiction/status/1905998185962643767">Tweet from renderfiction (@renderfiction)</a>: Gemini 2.5 Pro physics simulations in Three.js! All of these started out as &#34;one-shot prompts&#34; but I continued to query Gemini for better results. Clone with GitHub below 👇#threejs #Physics</li><li><a href="https://x.com/sama/status/1906793591944646898?t=Xw_DyPuHG0edzBlLvbUn3g&s=19">Tweet from Sam Altman (@sama)</a>: TL;DR: we are excited to release a powerful new open-weight language model with reasoning in the coming months, and we want to talk to devs about how to make it maximally useful: https://openai.com/op...</li><li><a href="https://www.reddit.com/r/Bard/comments/">Reddit - The heart of the internet</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1j7n2s5/manus_turns_out_to_be_just_claude_sonnet_29_oth">Reddit - The heart of the internet</a>: no description found</li><li><a href="https://artificialanalysis.ai/models/gpt-4o-chatgpt-03-25">GPT-4o (March 2025) - Intelligence, Performance &amp; Price Analysis | Artificial Analysis</a>: Analysis of OpenAI&#x27;s GPT-4o (March 2025, chatgpt-4o-latest) and comparison to other AI models across key metrics including quality, price, performance (tokens per second &amp; time to first token...</li><li><a href="https://siliconangle.com/2025/03/07/microsoft-reportedly-develops-llm-series-can-rival-openai-anthropic-models/>,">Microsoft reportedly develops LLM series that can rival OpenAI, Anthropic models - SiliconANGLE</a>: Microsoft reportedly develops LLM series that can rival OpenAI, Anthropic models - SiliconANGLE</li><li><a href="https://www.reddit.com/r/Bard/comments/1jo50hq/gemini_25_pro_will_also_be_a_nonthinking_model/">Reddit - The heart of the internet</a>: no description found</li><li><a href="https://gemini.google.com/share/dd74a82eaa14">‎Gemini - Enhanced Pelican Bicycle Animation
</a>: Created with Gemini Advanced</li><li><a href="https://www.reddit.com/r/Bard/comments/1jnk395/new_moonhowler_model_on_arena_llm_appears_to_be/">Reddit - The heart of the internet</a>: no description found</li><li><a href="https://manus.im">Manus</a>: Manus is a general AI agent that turns your thoughts into actions. It excels at various tasks in work and life, getting everything done while you rest.</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1j7n2s5/manus_turns_out_to_be_just_claude_sonnet_29_other/">Reddit - The heart of the internet</a>: no description found</li><li><a href="https://github.com/DataEval/dingo">GitHub - DataEval/dingo: Dingo: A Comprehensive Data Quality Evaluation Tool</a>: Dingo: A Comprehensive Data Quality Evaluation Tool - DataEval/dingo</li><li><a href="https://huggingface.co/spaces/DataEval/dingo">Dingo - a Hugging Face Space by DataEval</a>: no description found
</li>
</ul>

</div>
  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1355261415526633484)** (898 messages🔥🔥🔥): 

> `Gemini 2.5 Pro, Cline vs. Cursor, Roo Code, Windsurf, Cursor Pricing` 


- **Gemini 2.5 Pro Praised, faces tool use issues**: Users discuss the **Gemini 2.5 Pro** model, with some praising its performance and cost-effectiveness, while others report problems with tool use in Cursor, suggesting Cursor might be intentionally hindering its functionality to promote paid options.
   - Despite its potential, some users find Gemini Pro 2.5 *not good at actually calling the tools* within Cursor, often outputting incorrect or non-functional code.
- **Cline vs Cursor debate heats up**: The discussion revolves around **Cline's stability and efficiency** compared to **Cursor's features and bugs**, with some users preferring Cline for its reliability and direct model application, and others acknowledging Cursor's semantic search capabilities and experimentation.
   - One user stated *Cline feels polished af* while another stated that they *fear Roo code will nuke my whole codebase*.
- **Roo Code Gains Traction**: Several users are exploring **Roo Code** for its features like **boomerang tasks** and better context retention, noting it as an *evolution of Cline*, but concerns remain about its stability, rollback capabilities, and high Anthropic API token consumption, leading some to call it an option for *vibe coding*.
   - Despite praises, one user said, *If it's not implemented in roo it ain't ready yet for me*.
- **Windsurf as an alternative to Cursor**: Users explore **Windsurf** as a potential Cursor alternative for its ultimate plan, terminal/server task stability, with mentions of an embedded browser for easy element info sharing with AI; however, concerns arise regarding limited context window, the actual actions the model can make and a possible worse value than several normal plans.
   - One user stated *I don't like windsurf at all, the context window seems even smaller*, while others point out Windsurf's seemingly better stability.
- **Context Window and Pricing**: Members are frustrated with Cursor's usage-based pricing, token limits, and quality/efficiency reduction of models when usage limits are reached, leading some to explore alternative assistants like **Cline or Roo** for their full context windows and lower costs with services like OpenRouter or AI Studio.
   - A user states *Same feature with Claude max on cursor would have costed around $2* and then says *So a 10x reduction in price* when talking about alternatives.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/raizamrtn/status/1906727510601355393?s=46&t=ggmESCIXF0nYw8_kshHz7A">Tweet from Raiza Martin (@raizamrtn)</a>: Try out Gemini Code: a coding assistant in your terminal powered by Gemini 2.5 Pro. 💜 http://geminicodes.co 💜Release Notes:→ It&#39;s easy to get started, just run `pip install gemini-code` in a vir...</li><li><a href="https://x.com/stefcodes/status/1906122522644377788">Tweet from Stefan Meyer (@stefcodes)</a>: @angelmercedes @7etsuo @jaivinwylde +1 for RooGiven me 100x better results compared to Cursor</li><li><a href="https://docs.cursor.com/settings/models">Cursor – Models</a>: no description found</li><li><a href="https://docs.cursor.com/troubleshooting/common-issues#networking-issues-http">Cursor – Common Issues</a>: no description found</li><li><a href="https://prompt.16x.engineer/">16x Prompt - AI Coding with Advanced Context Management</a>: 16x Prompt is an advanced tool for AI coding. Manage code context, customize prompts, and ship features faster with multiple LLM API integrations.</li><li><a href="https://fxtwitter.com/adonis_singh/status/1906372453086937422">Tweet from adi (@adonis_singh)</a>: sonnet 3.7 vs gemini 2.5 pro - building the ChatGPT UIleft: claude 3.7 sonnet thinkingright: gemini 2.5 prowe have a new UI king !!</li><li><a href="https://www.cursor.com/pricing">Pricing | Cursor - The AI Code Editor</a>: Choose the plan that works for you.</li><li><a href="https://docs.cursor.com/context/@-symbols/@-docs">Cursor – @Docs</a>: no description found</li><li><a href="https://forum.cursor.com/t/guide-maximizing-coding-efficiency-with-mcp-sequential-thinking-openrouter-ai/66461/38?u=kleosr">[Guide] Maximizing Coding Efficiency with MCP Sequential Thinking &amp; OpenRouter AI</a>: Thanks for the detailed insights—really appreciate it. I’ve been implementing the latest changelog and testing the new version released from Cursor sitting on 0.48, which has optimized the rules signi...</li><li><a href="https://aistudio.google.com/app/apikey">no title found</a>: no description found</li><li><a href="https://github.com/TowelDude/cursor-mcp-collector">GitHub - TowelDude/cursor-mcp-collector</a>: Contribute to TowelDude/cursor-mcp-collector development by creating an account on GitHub.</li><li><a href="https://www.reddit.com/r/ChatGPTCoding/comments/1jn36e1/roocode_vs_cline_updated_march_29/">Reddit - The heart of the internet</a>: no description found</li><li><a href="https://cloud.google.com/vertex-ai">Vertex AI Platform</a>: Enterprise ready, fully-managed, unified AI development platform. Access and utilize Vertex AI Studio, Agent Builder, and 160+ foundation models.</li><li><a href="https://fireworks.ai/">Fireworks - Fastest Inference for Generative AI</a>: Use state-of-the-art, open-source LLMs and image models at blazing fast speed, or fine-tune and deploy your own at no additional cost with Fireworks AI!
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1355688349863510066)** (4 messages): 

> `Perplexity Pro, Discord Improvements, Smart Routing` 


- **Perplexity rolling out new Pro features**: Perplexity will soon roll out a new "**Pro**" that includes both existing **Pro + Reasoning models**.
   - The new **Pro** will default to sticky models instead of "**Auto**" for follow-ups; a highly requested change.
- **Perplexity Pro has Smart Routing**: **Pro** now also benefits from **smart routing** to ensure the best balance of **speed and reasoning**.
   - Perplexity is soliciting feedback in the appropriate channel.
- **Discord Improvements Incoming**: The moderation team has been collecting feedback and will be making **3 improvements** to the Discord experience over the next week.
   - These improvements include: **1) Simplified Onboarding Flow**, **2) Better Way to Relay Feedback**, and **3) Pro Channel Visibility & Access**.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1355255143775604766)** (790 messages🔥🔥🔥): 

> `Deep Research High, Grok Rate Limits, Deepseek bribed, Comet Waitlist, OpenRouter` 


- **Deep Research High still doesn't exist**: The **"Deep Research High"** tier on Perplexity AI does not exist yet, despite some users believing they are using it, the **complexity dev** confirmed today.
   - A user also noted that Grok gives **5 deep searches per 2 hours for free**, while also pointing out that *Grok rate limits are very strict*.
- **New Perplexity models coming zzzz**: The promised **Comet waitlist rollouts** and potential addition of a **HIGH model** did not materialize last week.
   - A user expressed frustration with frequent changes, saying that it's *pretty unethnical to rename Deepseek to perplexity reasoning model and name it 1776... yea Deepseek US edition. wtf is this*.
- **DeepSeek paid to stay out of the game?**: A user speculates that **OpenAI bribed Deepseek** to prevent them from fixing their web search, thus stifling competition.
   - Another user refuted this claim, saying that *it makes 0 sense for openai to bribe deepseek to "keep websearch off"* citing talent and OSS
- **Annoyances with New UI Changes**: Users expressed frustration with the new UI changes, particularly the **removal of model selection options** and the forced "Auto" mode, and several want a **Perplexity Pro refund**.
   - Some users speculate that the **automatic model selection** is intentional by Perplexity to push users to cheaper models, which lead some people to say **Pro mode has got worse** and that they now recommend 2.5 Pro.
- **Pro Search Woes for Sports Betting**: Members discuss using **Perplexity AI for sports betting**, prompting warnings about the unreliability of AI for financial decisions.
   - A user suggested in **manage account u can use AI of your preference**, however, they added that *they don’t have specific AI FOR IT*.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://manus.im/">Manus</a>: Manus is a general AI agent that turns your thoughts into actions. It excels at various tasks in work and life, getting everything done while you rest.</li><li><a href="https://tenor.com/view/totoro-gif-24991987">Totoro GIF - Totoro - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.reddit.com/r/perplexity_ai/comments/1jm2ekd/message_from_aravind_cofounder_and_ceo_of/">Reddit - The heart of the internet</a>: no description found</li><li><a href="https://tenor.com/view/smh-gif-smh-meme-smh-steve-harvey-i-can%27t-gif-13893533684179296052">Smh Gif Smh Meme GIF - Smh gif Smh meme Smh - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://gizmodo.com/why-does-chatgpts-algorithm-think-in-chinese-2000550311">Why Does ChatGPT&#039;s Algorithm &#039;Think&#039; in Chinese?</a>: OpenAI&#039;s new reasoning model is doing weird, unpredictable stuff.</li><li><a href="https://www.getmerlin.in/chat/share/47ebc788-d134-4019-9650-171aa42fc3ef">Give me an elaborate description of mavuika&#x27;s appe</a>: Shared By Anonymous - on March 31, 2025</li><li><a href="https://ahrefs.com/traffic-checker/?input=https%3A%2F%2Fwww.perplexity.ai%2Fdiscover&mode=exact">Website Traffic Checker: Estimate Any Site’s Traffic</a>: Dig into the traffic data for any website and find growth opportunities for yours. Try the free version of Ahrefs’ traffic checker.</li><li><a href="https://www.reddit.com/r/perplexity_ai/comments/1jm2ekd/message_from_aravind_cofounder_and_ceo_of">Reddit - The heart of the internet</a>: no description found</li><li><a href="https://huggingface.co/blog/open-deep-research">Open-source DeepResearch – Freeing our search agents</a>: no description found</li><li><a href="https://github.com/sentient-agi/OpenDeepSearch">GitHub - sentient-agi/OpenDeepSearch</a>: Contribute to sentient-agi/OpenDeepSearch development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1355297019173933278)** (20 messages🔥): 

> `AI Pathfinding Quirk, Supercomputer, AI diagnoses celiac disease, Google authenticator UX, Self-hosted projects` 


- **Bat AI Pathfinding Quirk exposed!**: A [Perplexity AI Page](https://www.perplexity.ai/page/bat-ai-pathfinding-quirk-z3ihUFcfSSWKJXOqW9mQ9g) discusses an oddity with **bat AI**.
   - No further information was provided, but users can investigate this **AI quirk**.
- **Hyper Supercomputer Uncovers Something!**: A [Perplexity AI Page](https://www.perplexity.ai/page/supercomputer-uncovers-hyperso-J88jGXHzRAeiRmYDdaDymQ) mentions a **supercomputer** discovery.
   - Further details require a visit to the page.
- **AI diagnoses celiac disease**: A [Perplexity AI Page](https://www.perplexity.ai/page/ai-diagnoses-celiac-disease-8VHqfHlkTVa3QE5AB90Ynw) speaks of **AI** diagnosing **celiac disease**.
   - No further information was provided.
- **Google Authenticator's UX regressed!**: A [Perplexity AI Page](https://www.perplexity.ai/page/google-authenticator-ux-regres-gn5atpKnQw.GTmqeirskSw) discusses UX regressions in **Google Authenticator**.
   - Users are encouraged to investigate the changes.
- **Exploring Best Self-Hosted Projects**: A [Perplexity AI Search](https://www.perplexity.ai/search/best-self-hosted-projects-zmdmGpAWR1S6e81ffGemaw) attempts to find the **best self-hosted projects**.
   - Interested users should check out the link.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1355319696521564322)** (30 messages🔥): 

> `Sonar API performance, Structured outputs, Image Search, Search depth API, Prompt data retention` 


- ****Sonar API** speed to improve**: Members reported that the **newest version of Sonar** has a *significantly longer response time* than the previous version and PPLX is making a note of this, and will see what they can do.
   - Another member reported **2.25 sec to first token** with the new sonar, while another reported **1 minute wait times** for the same.
- **Tier Restrictions Lifted on Structured Outputs**: Perplexity AI [announced](https://docs.perplexity.ai/guides/structured-outputs) that **structured outputs are now available for all users**, regardless of tier level, effective immediately.
   - The announcement indicated that **JSON structured outputs** are supported across all models, while both **JSON and Regex structured outputs** are currently supported for `sonar` and `sonar-reasoning` models.
- **Image search will be coming soon to API**: In response to queries about using the API for image searches to find similar products, a team member confirmed that **image search is not yet supported** but will be available *very soon*.
   - They noted that the API *does offer a way to return images* using the `return_images=True` parameter.
- **Users request API params for search depth**: A user inquired about specifying the **depth of search** (low, medium, high) in the API, noting they couldn't find it in the sample cURL requests.
   - A member responded that the search depth can be passed as extra body during the request, pointing to the [API reference](https://docs.perplexity.ai/api-reference/chat-completions) and promising to add it to the cURL request examples.
- **No data retention policy**: In response to a question about prompt and output retention, a Perplexity team member confirmed they have **0 data retention policy for the API**.
   - The member clarified that this policy applies *on their end*.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.perplexity.ai/api-reference/chat-completions),">no title found</a>: no description found</li><li><a href="https://docs.perplexity.ai/guides/structured-outputs)">no title found</a>: no description found
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1355255157125808361)** (747 messages🔥🔥🔥): 

> `Gemini 2.5 Pro, Grok vs Gemini, AI Image Generation, AI Energy Usage, Cursor & Code Generation` 


- **Gemini 2.5 Pro Coding Abilities Debated**: Users have expressed varying opinions on **Gemini 2.5 Pro's** coding abilities, with one user noting it's *terrible at C++ and WinAPI and always hallucinates stuff*, while another found it to be *very solid* at C++ but struggles with macro expansion.
   - Others find it to be excellent in certain languages like Jax, and its detailed CoT (Chain of Thought) steps.
- **Grok's Unstable Performance Sparks Frustration**: Several users have reported **Grok's** unstable performance, experiencing frequent log-offs and internal error messages, and the fact that the **“thinking mode doesn't work as intended.”**
   - Despite these issues, some users still find **Grok** to be *pretty good* and continue to subscribe to it alongside **ChatGPT Pro**.
- **New Image Generation sparks Debate**: Users are trying the new image generation, with a consensus on new image generation, though some find it to be glitchy and producing scuffed text.
   - Despite rejecting AI art, Hayao Miyazaki's Ghibli style is being mimicked by many users and AI.
- **AI Energy Use Questioned**: Some users are questioning if AI really uses a lot of energy and water, stating that *making a single burger uses almost 6000 times more energy than a single open AI query*, and the claim that AI uses water indirectly, but that can be applied to many things.
   - Others insist that AI uses a lot of electricity and water because datacenters cooling systems are water based, and that water evaporates and needs replenishing.
- **Cursor & Code Generation Tooling Discussed**: Members discussed code generation within Cursor and general code quality, including it's lack of customizability and limited interface, but that the models in Cursor, such as the new **Gemini 2.5 Max and Claude 3.7 Max**, offer full context, but are paywalled.
   - One member asked if Cursor could handle 10k words of code, and it was stated that it *fixes issues in big files with more than a thousand lines of code, even when multiple files are provided*.



**Link mentioned**: <a href="https://www.facebook.com/share/r/16QWbFUeEe/">34K views &#xb7; 34K reactions | THIS is the FULL DISCLOSURE &#x1f92f; | Krystle Channel</a>: THIS is the FULL DISCLOSURE &#x1f92f;. 

  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1355347367481573457)** (67 messages🔥🔥): 

> `File expiration issues in ChatGPT, Rate Limits for Image Generation, Reporting potential bugs for rewards, Ethics & Usage Policies` 


- **ChatGPT Files Expire Prematurely!**: A user reports files uploaded to ChatGPT are **expiring within minutes**, disrupting complex work involving legal and tax documents, despite previously stable sessions.
   - Another user suggested using **ChatGPT projects** and uploading primary files as project files.
- **Image Gen Users Hit Rate Limiting**: Due to extreme load since the new image model release, **Plus users** are now experiencing **rate limits** on image generation.
   - An interim measure was put in place due to extreme load since the new image model was released; new users also cannot create videos on **Sora**.
- **Bug Bounty Hunters Cash In**: Members discussed OpenAI's [Bug Bounty Program](https://openai.com/index/bug-bounty-program/), where reporting 'in scope' bugs can yield rewards.
   - The discussion emphasized the **ethics** involved and the importance of following **Terms of Service** to avoid account suspension, especially concerning disallowed content.
- **User Account Access Hanging By a Thread**: A user shared that he was testing a theory from YouTube on ChatGPT and made AIs talk to each other, leading to concerns about violating OpenAI's [Usage Policies](https://openai.com/policies/usage-policies/).
   - Another member pointed out that **violating the policies** could result in account suspension or termination, advising the user to review the terms.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1355255635335184545)** (114 messages🔥🔥): 

> `Markdown in prompt engineering, Using @ to bring in custom GPTs, SORA and Copyrighted Characters, O1 or O3 for creative tasks` 


- **Markdown Mayhem: Debate Erupts Over Formatting in AI Prompts**: Members discuss the challenges and limitations of using markdown in the prompt-engineering channel, noting that the lack of markdown support can hinder effective communication and education.
   - One member argues that a *no markdown rule is just lazy* and prevents users from educating others using **the language the AI uses**, while others point out that not everyone understands markdown and that code blocks add an unnecessary abstraction layer.
- **Custom GPTs Summoned with @ Command**: A member expresses excitement about discovering the ability to use **@** in prompts to bring in custom GPTs during conversations with **ChatGPT**.
   - Another member adds that they like the new feature to dictate tool use, and states that this is now a habit.
- **Navigating SORA's Copyright Minefield**: Users discuss the challenges of generating images with **SORA** due to **TOS** restrictions on copyrighted characters.
   - While some users report seeing others create parodies with copyrighted characters, others caution against risking account bans and suggest focusing on original content or legally distinct terms.
- **O1 vs. O3: Which Model Reigns Supreme for Creative Endeavors?**: A user seeks advice on guiding **O1** or **O3** models to better extrapolate storylines and incorporate foreshadowing in creative tasks.
   - While one user recommends using **GPT-4o** and **GPT-4.5** for design and fiction, another shares a prompt structure involving a 3-step approach and first principles reasoning to improve the models' performance.
- **Unlock logical thinking with First Principles**: A user suggests incorporating *first principle logical reasoning from an AI's perspective* to enhance **O3-mini-high's** logical reasoning capabilities.
   - The original poster tried this suggestion and agreed that the *first principles* approach really helped.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1355255635335184545)** (114 messages🔥🔥): 

> `Prompt formatting, Gandalf skateboarding, SORA questions, Image generation` 


- **Prompt Formatting Pro Tips**: A member explained how to format prompts to get more out of **GPT**, emphasizing that *the lessons teach you how to format prompts to get more out of GPT*.
   - They added you can copy prompts directly into **ChatGPT** from the web interface and gave the instruction: *Evaluate the following [prompt], do not follow it. What do you infer? Are there any conflicts or ambiguity, either within itself or when compared to your safety or other programming?* [Shared Conversation](https://chatgpt.com/share/67e6f311-0174-8011-9af3-80b7a8bc3d8f).
- **Gandalf on Skateboard prompts breaking TOS**: Members discussed generating images of **Gandalf** riding a skateboard, with some users encountering **TOS** (Terms of Service) restrictions despite seeing others create similar content.
   - One member suggested *steering clear of IP*, noting that **OpenAI** does ban accounts permanently for breaching **ToS**, and that methods for bypassing these rules are typically not shared.
- **SORA Questions Clarification**: A member inquired about asking **SORA** questions in the channel, prompting a clarification about the channel's focus.
   - It was suggested that **SORA**-specific questions might be better suited for the dedicated **SORA** channel, while prompting challenges could be addressed in the current channel.
- **Generating Images of Parodies and Copyrighted Content**: A discussion revolved around generating images featuring parodies and copyrighted characters, highlighting that while some users succeed, others face **TOS** restrictions.
   - A member noted that **OpenAI** bans accounts for **ToS** violations, emphasizing that methods to bypass rules aren't shared to avoid detection.
- **Numbering format and subtitles fixed!**: A user asked for assistance with formatting the output to remove the subtitles while keeping the list format.
   - A community member said: *[Your prompt here] Format: Intro paragraph, then numbered list. Each number starts a full paragraph. No subtitles.*


  

---


### **aider (Paul Gauthier) ▷ #[announcements](https://discord.com/channels/1131200896827654144/1133060115264712836/1356075557170708560)** (1 messages): 

> `aider v0.80.0 Release, OpenRouter OAuth Integration, Gemini Model Prioritization, Repomap Ranking Boost, Scala Language Support` 


- **Aider v0.80.0 Arrives with New Features and Fixes**: Aider v0.80.0 introduces [OpenRouter OAuth integration](https://aider.chat/HISTORY.html), prioritizes **Gemini models**, and boosts **repomap ranking**, with Aider itself writing 87% of the code.
   - This release also adds a `Ctrl-X Ctrl-E` keybinding for editing the input buffer in an external editor, alongside other improvements and bug fixes.
- **OpenRouter OAuth Simplifies Model Access**: Aider now offers [OAuth integration with OpenRouter](https://aider.chat/HISTORY.html) if no model and keys are provided, streamlining the process of accessing models.
   - It automatically selects the OpenRouter default model based on free/paid tier status when `OPENROUTER_API_KEY` is set but no model is specified.
- **Gemini Models Get Prioritized**: The latest Aider version prioritizes `gemini/gemini-2.5-pro-exp-03-25` when `GEMINI_API_KEY` is set, and `vertex_ai/gemini-2.5-pro-exp-03-25` if `VERTEXAI_PROJECT` is configured, enhancing model selection.
   - These settings ensure users leverage the most appropriate Gemini model based on their environment variables.
- **Repomap Ranking Receives a Boost**: [Repomap ranking](https://aider.chat/HISTORY.html) is now improved for files whose path components match identifiers mentioned in the chat, making it easier to locate relevant files.
   - Additionally, Scala language gains repomap support, further broadening the range of supported languages.
- **Ctrl-X Ctrl-E Keybinding for External Editor Access**: Users can now edit the current input buffer in an external editor using the new [Ctrl-X Ctrl-E keybinding](https://aider.chat/HISTORY.html), improving the editing workflow.
   - This feature, contributed by Matteo Landi, offers a convenient way to leverage familiar text editors for input.



**Link mentioned**: <a href="https://aider.chat/HISTORY.html">Release history</a>: Release notes and stats on aider writing its own code.

  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1355256424632025118)** (785 messages🔥🔥🔥): 

> `Fixing AI generated code, Aider Enhancement, Gemini 2.5, OpenAI Agent SDK, Claude` 


- **Posting on boomer twitter to fix generated code**: A member posted their offer to help fix AI generated code on *boomer twitter* (linkedin).
   - Another member expressed concern, stating that AI can generate thousands of lines of code easily, needing AI to undo the slop.
- **Gemini 2.5 versus Sonnet discussion fires up**: Members discuss the merits of [Gemini 2.5](https://aistudio.google.com/app/u/2/apikey) versus Sonnet for various tasks including code rewrites, with varying results.
   - One member lauded Gemini 2.5 for one-shot rewriting their server from node 'http' into express, but another said *'my statement on gemini 2.5 is that it is trashinconsistent and trained to provide good benchmarks but maybe I use it wrong.'*
- **Gary codes in GO, organizes Obsidian Vault with GDS**: A member shared their [GitHub organization](https://github.com/Aider-AI/aider) and detailed the many applications they've coded in **GO**, including a tool named *obsorter* which sorts their Obsidian vault files into predefined directories and renames them based on content, using a *Gary Decimal System (GDS)*.
   - Others shared their admiration for the system that seems to act as a *'johnny decimal system'* for knowledge.
- **DeepSeek can't follow instructions, unlike Gemini**: A member complained that after switching from **Gemini 2.5** to **DeepSeek** they found that DeepSeek cannot follow instructions, stating that *'I tell it to jump off bridge it creates two small villages'*, while praising Gemini.
   - Others chimed in that the rate limits for **Gemini 2.5** might be the biggest issue when trying to use the new models.
- **Aider Benchmarks highlighted**: A member highlights that [Aider benchmarks](https://aider.chat/docs/benchmarks.html) are front and center on a [YouTube video](https://youtu.be/LmpNOY5sQuc?t=43) and recognizes its value as a tool.
   - This sparks discussion that **Aider** was someone prompting just right with intimate knowledge of the tool and how to get the best out of LLM interactions.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/mahawaryas27492/status/1906794382659125625?s=46">Tweet from AI Purr-fessor (Yash) (@MahawarYas27492)</a>: Amazing 2.5 flash experimental is out. 🔥 It&#39;s very smart better than o3 mini high on some reasoning questions I tested.It&#39;s rolling out very slowly so you might need to wait for officially, I...</li><li><a href="https://docs.litellm.ai/docs/providers">Providers | liteLLM</a>: Learn how to deploy + call models from different providers on LiteLLM</li><li><a href="https://tenor.com/view/techno-viking-viking-gif-26693787">Techno Viking GIF - Techno Viking Viking - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://liveswebench.ai/">LiveSWEBench</a>: no description found</li><li><a href="https://tenor.com/view/go-for-it-you-can-do-it-encourage-do-it-gif-14006408">Go For It You Can Do It GIF - Go For It You Can Do It Encourage - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/pov-you-giga-chad-chad-meme-gif-25615024">Pov You GIF - Pov You Giga Chad - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/sherlock-benedict-cumberbatch-hat-gif-15943210">Sherlock Benedict Cumberbatch GIF - Sherlock Benedict Cumberbatch Hat - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/wedding-crashers-will-ferrell-what-a-loser-loser-laugh-gif-3957171">What A Loser GIF - Wedding Crashers Will Ferrell What A Loser - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/shhh-shush-silence-nose-gif-17895433">Shhh Shush GIF - Shhh Shush Silence - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/andrew-tate-stare-andrew-tate-andrew-tate-sigma-xafer-gif-10165002945664617941">Andrew Tate Stare Andrew Tate Sigma GIF - Andrew tate stare Andrew tate Andrew tate sigma - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/andrew-tate-tate-why-gif-940321714429124603">Andrew Tate Why GIF - Andrew tate Tate Why - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/joe-biden-deal-with-it-cool-glasses-meme-gif-13473183379638803062">Joe Biden Deal With It GIF - Joe biden Deal with it Cool - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://aider.chat/docs/usage/modes.html#askcode-workflow">Chat modes</a>: Using the code, architect, ask and help chat modes.</li><li><a href="https://tenor.com/view/thumbs-up-alright-not-bad-gif-7771888706215464379">Thumbs Up GIF - Thumbs Up Alright - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://aider.chat/docs/llms/gemini.html">Gemini</a>: aider is AI pair programming in your terminal</li><li><a href="https://tenor.com/view/wine-alcohol-red-will-ferrell-drinking-gif-5034418">Wine Alcohol GIF - Wine Alcohol Red - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/techno-point-gif-24022320">Techno Point GIF - Techno Point - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/mufeedvh/code2prompt/issues/107">not able to `cargo install` · Issue #107 · mufeedvh/code2prompt</a>: error[E0532]: expected a pattern, found a function call --&gt; /Users/slu/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/code2prompt-3.0.0/src/main.rs:199:17 | 199 | Ok(_) =&gt; { | ^^ not a tup...</li><li><a href="https://cloud.google.com/docs/authentication/external/set-up-adc">no title found</a>: no description found</li><li><a href="https://github.com/joanrod/star-vector">GitHub - joanrod/star-vector: StarVector is a foundation model for SVG generation that transforms vectorization into a code generation task. Using a vision-language modeling architecture, StarVector processes both visual and textual inputs to produce high-quality SVG code with remarkable precision.</a>: StarVector is a foundation model for SVG generation that transforms vectorization into a code generation task. Using a vision-language modeling architecture, StarVector processes both visual and te...</li><li><a href="https://github.com/Aider-AI/aider/issues/2979#issuecomment-2613554537">Restoring chat history leads to error / chat history summarization not working · Issue #2979 · Aider-AI/aider</a>: Issue I have a chat history file that&#39;s quite long (80k tokens) but offers lots of valuable information about the project I&#39;m building. It worked fine last week when I use model that has large...</li><li><a href="https://youtu.be/LmpNOY5sQuc?t=43"> - YouTube</a>: no description found</li><li><a href="https://github.com/solcloud/Counter-Strike/tree/master?tab=readme-ov-file#counter-strike-football---">GitHub - solcloud/Counter-Strike: Multiplayer FPS game - Counter-Strike: Football 🏉</a>: Multiplayer FPS game - Counter-Strike: Football 🏉. Contribute to solcloud/Counter-Strike development by creating an account on GitHub.</li><li><a href="https://aider.chat/docs/config/options.html#history-files>">Options reference</a>: Details about all of aider’s settings.</li><li><a href="https://useai.substack.com/p/beyond-prompts-why-context-management">Beyond Prompts: Why Context Management significantly improves AI Performance</a>: Just because a model can handle a lot of context, doesn&#x27;t mean it should. Here&#x27;s how and why you can manage the context window better to make better use of LLMs</li><li><a href="https://arxiv.org/abs/2308.14508">LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding</a>: Although large language models (LLMs) demonstrate impressive performance for many language tasks, most of them can only handle texts a few thousand tokens long, limiting their applications on longer s...</li><li><a href="https://arxiv.org/abs/2307.11088">L-Eval: Instituting Standardized Evaluation for Long Context Language Models</a>: Recently, there has been growing interest in extending the context length of large language models (LLMs), aiming to effectively process long inputs of one turn or conversations with more extensive hi...</li><li><a href="https://arxiv.org/abs/2502.05167">NoLiMa: Long-Context Evaluation Beyond Literal Matching</a>: Recent large language models (LLMs) support long contexts ranging from 128K to 1M tokens. A popular method for evaluating these capabilities is the needle-in-a-haystack (NIAH) test, which involves ret...</li><li><a href="https://aider.chat/docs/config/api-keys.html">API Keys</a>: Setting API keys for API providers.</li><li><a href="https://aider.chat/docs/config/dotenv.html">Config with .env</a>: Using a .env file to store LLM API keys for aider.</li><li><a href="https://aider.chat/docs/config/aider_conf.html">YAML config file</a>: How to configure aider with a yaml config file.</li><li><a href="https://aider.chat/docs/troubleshooting/models-and-keys.html">Models and API keys</a>: aider is AI pair programming in your terminal
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1355257943444164748)** (150 messages🔥🔥): 

> `Gemini 2.5 Pro, Rate Limits, Aider Hooks, Architect mode improvements, MCP support` 


- **Gemini 2.5 Pro Usage and Quota Oddities Persist**: Users are reporting inconsistencies with **Gemini 2.5 Pro** usage, with the API console sometimes showing separate **2.5** and **2.0** usage, despite **Aider** reporting only **2.5** usage, as detailed in [issue #3641](https://github.com/Aider-AI/aider/issues/3641#issuecomment-2762538743).
   - A member mentioned that **2.0-exp** is an internal name, and some have seen different quota limits being applied to **2.5**, with speculation that **2.5** may be reusing **2.0** quotas.
- **Aider's Cache Writes Inflating Token Counts**: One user observed that **Aider** doesn't count cache writes as input, leading to the tokens sent showing as double (*e.g.*, **12k** sent, **6.1k** cache write) when using **Sonnet**.
   - The user inquired whether others have experienced similar behavior, and the root cause is being investigated to ensure accurate token tracking.
- **Architect Mode's Edit Loop Frustrates Users**: Some users reported an issue in recent versions where **architect mode** gets stuck in an infinite loop, repeatedly asking to edit files after providing a summary, which can be bypassed via `/ask` and `/code ok`.
   - A member identified `auto-accept-architect: false` in the config file as a way to revert to the previous behavior where it always asks before editing.
- **MCP Support Gains Traction**: There's growing interest in **MCP (Model Collaboration Protocol)** support within **Aider**, with discussions around its potential to reduce model lock-in and foster OSS tool development, as showcased on [MCP Marketplace](https://github.com/cline/mcp-marketplace).
   - A member mentioned a third-party integration via `mcpm-aider`, and others expressed interest in built-in support for streamlined usage, with [PR #3672](https://github.com/Aider-AI/aider/pull/3672) adding initial support.
- **Partial Reads Sought for Large Files**: Users are seeking ways to implement **partial reads** in **Aider** to handle large files that exceed context limits, with some suggesting using the `/run` command with tools like `head`, `grep`, or `rag-cli`.
   - A member shared a custom **RAG tool** called [rag-tool](https://github.com/chadfurman/rag-tool) built with Mastra agents, designed to extract details from codebases and work with large files, usable within Aider via `/run npm run agent`.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/it%27s-a-slow-day-gif-17869747439397645052">It&#039;S A Slow Day GIF - IT&#039;S A SLOW DAY - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/cline/mcp-marketplace">GitHub - cline/mcp-marketplace: This is the official repository for submitting MCP servers to be included in Cline&#39;s MCP Marketplace. If you’ve built an MCP server and want it to be discoverable and easily installable by millions of developers using Cline, submit your server here.</a>: This is the official repository for submitting MCP servers to be included in Cline&amp;#39;s MCP Marketplace. If you’ve built an MCP server and want it to be discoverable and easily installable by mil...</li><li><a href="https://github.com/chadfurman/rag-tool">GitHub - chadfurman/rag-tool: Simple rag-tool using Mastra agents.  Designed to extract details about a codebase and to work with files.  Helpful for when the context is otherwise too big.</a>: Simple rag-tool using Mastra agents.  Designed to extract details about a codebase and to work with files.  Helpful for when the context is otherwise too big. - chadfurman/rag-tool</li><li><a href="https://github.com/Aider-AI/aider/issues/3196">100% cpu freezing does not respond to ctrl c on latest release · Issue #3196 · Aider-AI/aider</a>: Issue Process hangs at 100% CPU while rendering markdown/syntax highlighting. The process becomes unresponsive and requires force termination. This appears to be due to a pathological case in Pygme...</li><li><a href="https://github.com/Aider-AI/aider/issues/3641#issuecomment-2762538743">Gemini 2.5 Pro or DeepSeek V3 0324 not showing in `/models /` · Issue #3641 · Aider-AI/aider</a>: I have been using /models / to get a list of available models to use and based Aidermacs to select from the list, I&#39;ve very happy that Gemini 2.5 Pro and the latest deepseek are supported in Aider...</li><li><a href="https://github.com/Aider-AI/aider/issues/2227">Feature: Add GitHub Copilot as model provider · Issue #2227 · Aider-AI/aider</a>: Issue Hello! Please add GitHub Copilot as model provider. Should be possible like this: https://github.com/olimorris/codecompanion.nvim/blob/5c5a5c759b8c925e81f8584a0279eefc8a6c6643/lua/codecompani...</li><li><a href="https://github.com/BerriAI/litellm/pull/9079">Litellm dev 03 05 2025 contributor prs by krrishdholakia · Pull Request #9079 · BerriAI/litellm</a>: TitleRelevant issuesType🆕 New Feature🐛 Bug Fix🧹 Refactoring📖 Documentation🚄 Infrastructure✅ TestChanges[REQUIRED] Testing - Attach a screenshot of any new tests passing locallyIf UI...</li><li><a href="https://github.com/Aider-AI/aider/pull/3672">Add MCP support by Antonin-Deniau · Pull Request #3672 · Aider-AI/aider</a>: This is a rough implementation of MCP into Aider.It currently support adding stdio MCP servers with this configuration in the ~/.aider.conf.ymlmcp: truemcp-servers:  - git-servermcp-server-com...
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1355598596388426213)** (8 messages🔥): 

> `Quantization impact on model performance, Interview Coder AI tool` 


- **Quantization Degrades Model Accuracy**: Converting models from **FP16** to **Q8** results in a slight reduction in model quality, and using **Q4** quantization, the default in Ollama, can further degrade it.
   - It was noted that anything below **Q6** is severely impaired, especially for reasoning tasks, but another member said that since some models are natively **FP8**, **Q8** quantization *shouldn't lose any performance*.
- **Interview Coder Promises to Disrupt Technical Interviews**: [Interview Coder](https://www.interviewcoder.co/) is advertised as an *invisible AI* for technical interviews, aimed at replacing traditional platforms like Leetcode.
   - The tool is described as a *Peter principle accelerator*.



**Link mentioned**: <a href="https://www.interviewcoder.co/">Interview Coder - AI Assistant for Technical Interviews</a>: no description found

  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1355288520390086657)** (536 messages🔥🔥🔥): 

> `DeepSeek-V3-0324 Dynamic Quantization, RoBERTa Training Optimization, Serving Dynamic Quantized Checkpoints, 4bit Gemma 3 12B Training Issues, Qwen2.5-VL-32B-Instruct-unsloth-bnb-4bit notebook` 


- **DeepSeek-V3-0324 Dynamic Quantization makes debut**: Dynamic quantized versions of **DeepSeek-V3-0324** have been released on Hugging Face, with a [guide](https://docs.unsloth.ai/basics/tutorial-how-to-run-deepseek-v3-0324-locally) for local execution.
   - Unsloth's **Dynamic Quants** are selectively quantized, improving accuracy over standard bits.
- **Google Cloud Spot Instances Beat Runpod's Prices!**: Switching to **Google Cloud** resulted in 2x faster workloads and cheaper costs compared to Runpod.
   - Members noted that Google Cloud Spot Instances are up to **60% cheaper** and more stable than Runpod, which often breaks after 15 minutes.
- **Multi-GPU Support for Everyone - Soon(TM)**: The unsloth team says that Multi-GPU support will be available to everyone soon, but Pro/Enterprise is currently on hold due to capacity issues.
   - The consensus was to give multi-GPU to everyone with the current capabilities of unsloth.
- **HF x Unsloth Reasoning Collab**: Unsloth has partnered with Hugging Face on [this collab](https://x.com/UnslothAI/status/1906726176556712318) to teach users how to fine-tune LLMs with GRPO.
   - The course covers reward functions, GRPO math, and applying RL to real-world use cases, alongside a [tutorial](https://docs.unsloth.ai/basics/reasoning-grpo-and-rl/tutorial-train-your-own-reasoning-model-with-grpo).
- **New Whisper Notebook Makes a Scene**: Unsloth released a notebook for training Whisper, but emotive tags don't work without pretraining.
   - A user showed how using Orpheus and Unsloth to fine tune on just 50k german samples is already quite good. See [here](https://x.com/SebastianB929/status/1906049996585099701).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/UnslothAI/status/1906726176556712318">Tweet from Unsloth AI (@UnslothAI)</a>: We partnered with @HuggingFace to teach you how to fine-tune LLMs with GRPO!Learn about:• Reward functions + creating them• GRPO Math + Free Reasoning training in Colab• Applying RL to real-world use ...</li><li><a href="https://x.com/SebastianB929/status/1906049996585099701">Tweet from SebastianBoo (@SebastianB929)</a>: Orpheus finetuned on just 50k german samples is already quite good. Used unsloth and qlora. Currently only random speakers. Expressions like &lt;laughing&gt;, &lt;giggle&gt;, ... unfortunaltely does n...</li><li><a href="https://x.com/UnslothAI/status/1905312972278563256">Tweet from Unsloth AI (@UnslothAI)</a>: Listen to a before & after comparison of Orpheus-TTS after fine-tuning it on a small text-to-speech dataset with a customized new voice and dialogue.</li><li><a href="https://x.com/UnslothAI/status/1906460329292476732">Tweet from Unsloth AI (@UnslothAI)</a>: RT @reach_vb: Fuck it, 685B parameter, DeepSeek V3 0324 running locally on M3 Ultra, fully private 🔥Powered by llama.cpp & dynamic quants…</li><li><a href="https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s31151/">How GPU Computing Works | GTC Digital April 2021 | NVIDIA On-Demand</a>: Come for an introduction to GPU computing by the lead architect of CUDA</li><li><a href="https://docs.unsloth.ai/get-started/beginner-start-here">Beginner? Start here! | Unsloth Documentation</a>: no description found</li><li><a href="https://colab.research.google.com/github/towardsai/ragbook-notebooks/blob/main/notebooks/Chapter%2010%20-%20FineTuning_a_LLM_Financial_Sentiment_CPU.ipynb">Google Colab</a>: no description found</li><li><a href="https://docs.unsloth.ai/basics/tutorial-how-to-run-deepseek-v3-0324-locally">Tutorial: How to Run DeepSeek-V3-0324 Locally | Unsloth Documentation</a>: How to run DeepSeek-V3-0324 locally using our dynamic quants which recovers accuracy</li><li><a href="https://huggingface.co/docs/api-inference/en/tasks/text-generation">Text Generation</a>: no description found</li><li><a href="https://huggingface.co/unsloth/DeepSeek-V3-0324-GGUF">unsloth/DeepSeek-V3-0324-GGUF · Hugging Face</a>: no description found</li><li><a href="https://lenovopress.lenovo.com/lp2179-fine-tuning-llms-using-intel-xeon-cpus">Fine-Tuning LLMs using Intel Xeon CPUs</a>: Large Language Models (LLMs) have emerged as a powerful business tool, excelling at an assortment of tasks including Question-Answering (QA), text summarization, and translation. However, they must be...</li><li><a href="https://unsloth.ai/blog/gemma3">Fine-tune Gemma 3 with Unsloth</a>: Gemma 3, Google&#x27;s new multimodal models.Fine-tune &amp; Run them with Unsloth! Gemma 3 comes in 1B, 4B, 12B and 27B sizes.</li><li><a href="https://unsloth.ai/blog/qwq-32b#Tutorial%20QwQ">Run &amp; Finetune QwQ-32B with Bug Fixes</a>: Fine-tune &amp; Run Qwen&#x27;s new QwQ-32B models with Unsloth&#x27;s bug fixes. Solve the issue of endless generations.</li><li><a href="https://docs.unsloth.ai/get-started/unsloth-notebooks)">Unsloth Documentation</a>: no description found</li><li><a href="https://huggingface.co/DavidAU/L3.2-Rogue-Creative-Instruct-Uncensored-Abliterated-7B-GGUF">DavidAU/L3.2-Rogue-Creative-Instruct-Uncensored-Abliterated-7B-GGUF · Hugging Face</a>: no description found</li><li><a href="https://seb-v.github.io/optimization/update/2025/01/20/Fast-GPU-Matrix-multiplication.html">Optimizing Matrix Multiplication on RDNA3: 50 TFlops and 60% Faster Than rocBLAS</a>: Introduction</li><li><a href="https://huggingface.co/unsloth/Qwen2.5-VL-32B-Instruct-unsloth-bnb-4bit">unsloth/Qwen2.5-VL-32B-Instruct-unsloth-bnb-4bit · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/docs/transformers/en/chat_templating">Templates</a>: no description found</li><li><a href="https://ai.darvinbox.click/">LiteLLM API - Swagger UI</a>: no description found</li><li><a href="https://huggingface.co/hitachi-nlp">hitachi-nlp (Hitachi, Ltd.)</a>: no description found</li><li><a href="https://docs.unsloth.ai/basics/tutorials-how-to-fine-tune-and-run-llms/tutorial-how-to-run-qwq-32b-effectively">Tutorial: How to Run QwQ-32B effectively | Unsloth Documentation</a>: How to run QwQ-32B effectively with our bug fixes and without endless generations + GGUFs.</li><li><a href="https://github.com/huggingface/transformers/issues/36822">Gemma 3 is broken with fp16 · Issue #36822 · huggingface/transformers</a>: System Info transformers version: 4.50.0.dev0 Platform: Linux-6.8.0-39-generic-x86_64-with-glibc2.35 Python version: 3.11.10 Huggingface_hub version: 0.29.3 Safetensors version: 0.5.3 Accelerate ve...</li><li><a href="https://github.com/unslothai/llama.cpp">GitHub - unslothai/llama.cpp: LLM inference in C/C++</a>: LLM inference in C/C++. Contribute to unslothai/llama.cpp development by creating an account on GitHub.</li><li><a href="https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/convert.cu#L6>">llama.cpp/ggml/src/ggml-cuda/convert.cu at master · ggml-org/llama.cpp</a>: LLM inference in C/C++. Contribute to ggml-org/llama.cpp development by creating an account on GitHub.</li><li><a href="https://github.com/triton-lang/triton-cpu">GitHub - triton-lang/triton-cpu: An experimental CPU backend for Triton</a>: An experimental CPU backend for Triton. Contribute to triton-lang/triton-cpu development by creating an account on GitHub.</li><li><a href="https://github.com/intel/intel-extension-for-pytorch">GitHub - intel/intel-extension-for-pytorch: A Python package for extending the official PyTorch that can easily obtain performance on Intel platform</a>: A Python package for extending the official PyTorch that can easily obtain performance on Intel platform - intel/intel-extension-for-pytorch
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1355615311004434442)** (98 messages🔥🔥): 

> `Gemma-3 alternatives with tools, Full finetuning challenges and solutions, ML focused datasets, Training vs Fine-tuning memory requirements, Llama 3.2 3B` 


- **Quest for Gemma-3 with Tools Deepens**: Members are seeking **Gemma-3** alternatives that support **tool use**, citing the official [Gemma documentation](https://ai.google.dev/gemma/docs/capabilities/function-calling) indicating its support for function calling.
   - Suggested alternatives include **Qwen 2.5** (any size) or **Mistral Small 3.1**, with a disclaimer that models under 7B may not perform optimally.
- **OOM Woes Plague Full Finetuning Endeavors**: Users experimenting with **Unsloth**, **Axololt**, and **TorchTune** for full finetuning on single and multiple GPUs are facing **Out of Memory (OOM)** issues.
   - One user highlighted success with LoRA on **Unsloth Qwen2.5 14B** and sought advice for comparing results with full finetuning.
- **Machine Learning Dataset Hunt Kicks Off**: A member is seeking machine learning-focused datasets to finetune a help bot for users of a FOSS repository, and shared 2 links for the community: [ML-ArXiv-Papers](https://huggingface.co/datasets/CShorten/ML-ArXiv-Papers) and [ml_papers_arxiv](https://huggingface.co/datasets/ThatDataDude/ml_papers_arxiv).
   - Another member shared another QA dataset: [ml-arxiv-papers-qa](https://huggingface.co/datasets/hanyueshf/ml-arxiv-papers-qa).
- **Full Finetuning Requires less memory than Training from Scratch**: A user inquired about the memory requirements for training models from scratch, particularly for tasks like facial recognition, and if that took more memory than finetuning.
   - The answers were that training from scratch requires significantly more resources (500k+ images, more than 16GB VRAM) compared to finetuning, so he was advised to not *reinvent the wheel*.
- **Sloth Hug Emoji gets Love**: A member added the <:slothhug:1257540335438008343> emoji to the 🤗 server, shared links to the [discord_sloth_hug.png](https://cdn.discordapp.com/attachments/1179039861576056922/1356196511813472356/discord_sloth_hug.png?ex=67ec58ad&is=67eb072d&hm=99d4d88369da4acb1a46b3daa6fe6d88b814ce029eac9d99c91b4900c99640d6&) and [sloth_huglove_large.png](https://cdn.discordapp.com/attachments/1179039861576056922/1356196512740282469/sloth_huglove_large.png?ex=67ec58ad&is=67eb072d&hm=36f432d9b23573e30cd429548f7b97336f8cee495f0519c12f9f862c6f708885&).
   - This prompted celebratory emoji reactions from other members.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2.5_(7B)-Alpaca.ipynb">Google Colab</a>: no description found</li><li><a href="https://tenor.com/view/my-girl-seal-friendship-its-a-deal-seal-spit-gif-17005706">My Girl Seal Friendship GIF - My Girl Seal Friendship Its A Deal - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://ai.google.dev/gemma/docs/capabilities/function-calling">no title found</a>: no description found</li><li><a href="https://huggingface.co/datasets/ThatDataDude/ml_papers_arxiv">ThatDataDude/ml_papers_arxiv · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/CShorten/ML-ArXiv-Papers">CShorten/ML-ArXiv-Papers · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/hanyueshf/ml-arxiv-papers-qa">hanyueshf/ml-arxiv-papers-qa · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1355268580060696646)** (286 messages🔥🔥): 

> `Unsloth documentation update, Llama3.2-1B GRPO error, Deepseek V3 inference slow, Aya-Vision fine tuning with unsloth, Flash attention with Qwen model` 


- **Unsloth docs urge dependency updates!**: A member recommends updating **Unsloth documentation** to discourage using `--no-deps` during updates, as it causes issues and shares a [link](https://docs.unsloth.ai/get-started/installing-+-updating/updating) to the documentation.
   - Another member confirmed that the standard updating procedure also includes the `--no-deps` flag, indicating a potential error in the documentation that needs correction.
- **Debugging Aya Vision 8B Dimension Mismatch**: Members troubleshoot a `ValueError: Image features and image tokens do not match` error while fine-tuning **Aya-vision 8B** with Unsloth, referencing the [Qwen Vision Fine-tuning notebook](https://huggingface.co/CohereForAI/aya-vision-8b) as a guide.
   - It was determined that the `tokenizer` + `UnslothDataCollator` doesn't properly resize images, leading to dimension mismatches, and that the **AyaVisionProcessor expects a different message format**, which was ultimately resolved.
- **Troubleshooting Llama3.2-1B GRPO Errors**: Members encounter errors while performing **GRPO** on a continually pre-trained **Llama3.2-1B** model, specifically a `torch.fx.experimental.symbolic_shapes.ConstraintViolationError` related to shape constraints.
   - Debugging steps include checking the configurations of the meta model versus the finetuned model, and verifying the status of the `unsloth_fixed` parameter, suggesting an issue related to the compatibility of the model with the Unsloth implementation.
- **Mamba fine-tuning with Unsloth has issues**: A member reports failing to get **Mamba** fine-tuning working with Unsloth, encountering issues with the redirect function, also mentioning failures with **RWKV-6 HF**.
   - Members discussed that while RWKV-6 HF appears to work, the trainer doesn't perform any actions, potentially requiring source code edits, however, **Mamba** is expected to function with a single line code change.
- **GGUF Conversion fails on Gemma3 due to Assertion Error**: A member faces an `AssertionError` when trying to save or merge a continued pre-trained **Gemma 3** model into **Float16** for **vLLM** or **GGUF** format, suspecting a float32 casting issue during conversion.
   - The error occurs in `unsloth_zoo/saving_utils.py`, specifically during the creation of **LoRA statistics**, indicating a potential problem with the number of modules or the consistency of LoRA parameters.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.unsloth.ai/get-started/fine-tuning-guide#avoiding-overfitting-and-underfitting">Fine-tuning Guide | Unsloth Documentation</a>: Learn all the basics and best practices of fine-tuning. Beginner-friendly.</li><li><a href="https://docs.unsloth.ai/get-started/installing-+-updating/updating">Updating | Unsloth Documentation</a>: To update or use an old version of Unsloth, follow the steps below:</li><li><a href="https://colab.research.google.com/drive/1nft9qLA9m7s-4G8YgcSNGsO0CL8x1OmW#scrollTo=BRCcEg-9I-3S">Google Colab</a>: no description found</li><li><a href="https://huggingface.co/CohereForAI/aya-vision-8b">CohereForAI/aya-vision-8b · Hugging Face</a>: no description found</li><li><a href="https://www.kaggle.com/code/shivamgarg1999/qwen-finetuning-pipeline-peft-sgarg">qwen_finetuning_pipeline_peft_sgarg</a>: Explore and run machine learning code with Kaggle Notebooks | Using data from No attached data sources</li><li><a href="https://www.kaggle.com/code/shivamgarg1999/qwen-finetuning-pipeline-peft-sgarg/edit/run/230573561">qwen_finetuning_pipeline_peft_sgarg</a>: Explore and run machine learning code with Kaggle Notebooks | Using data from No attached data sources</li><li><a href="https://unsloth.ai/blog/gemma3#everything:~:text=Vision%20fine-tuning,truncating%20sequence%20lengths.)">Fine-tune Gemma 3 with Unsloth</a>: Gemma 3, Google&#x27;s new multimodal models.Fine-tune &amp; Run them with Unsloth! Gemma 3 comes in 1B, 4B, 12B and 27B sizes.</li><li><a href="https://huggingface.co/unsloth/DeepSeek-R1-GGUF">unsloth/DeepSeek-R1-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/docs/transformers/main/en/model_doc/aya_vision">AyaVision</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/issues/1178#issue-2610722849">DPO, ORPO - grad accumulation fix · Issue #1178 · unslothai/unsloth</a>: Goal: Propagate gradient accumulation fix to DPO - much harder since it requires a full rewrite of https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py</li><li><a href="https://github.com/unslothai/unsloth-zoo/pull/105">Fix: SmolVLM indentation error in compiled module by rolandtannous · Pull Request #105 · unslothai/unsloth-zoo</a>: IssueResolves unsloth issue #2179SmolVLM models cause an indentation error in generated compiled modules when used with Unsloth.Problem DescriptionWhen using SmolVLM models (specifically SmolVL...</li><li><a href="https://github.com/hiyouga/LLaMA-Factory/issues/6559#issuecomment-2678469573">qwen2-vl训练有bug，ValueError: Image features and image tokens do not match: tokens: 1468, features 1936,有人能回一下吗 · Issue #6559 · hiyouga/LLaMA-Factory</a>: Reminder I have read the README and searched the existing issues. System Info llamafactory version: 0.9.2.dev0 Platform: Linux-5.10.134-16.101.al8.x86_64-x86_64-with-glibc2.35 Python version: 3.10....</li><li><a href="https://unsloth.ai/blog/gradient">Bug Fixes in LLM Training - Gradient Accumulation</a>: Unsloth&#x27;s Gradient Accumulation fix solves critical errors in LLM Training.</li><li><a href="https://github.com/unslothai/unsloth/issues/2179">Generated unsloth_compiled_cache file cause Indentation Error when use unsloth with smolvlm2 · Issue #2179 · unslothai/unsloth</a>: I try to use unsloth with smolvlm2 but it keep throwing out &quot;unexpected indentation error&quot;. The cause as the error message tells is in 481th line of the generated file unsloth_compiled_cache...</li><li><a href="https://github.com/unslothai/unsloth/pull/1289">Added Support for Apple Silicon by shashikanth-a · Pull Request #1289 · unslothai/unsloth</a>: UnoptimizedNo gguf support yet.Build Triton and bitsandbytes from sourcecmake -DCOMPUTE_BACKEND=mps -S . for bitsandbytes buildingpip install unsloth-zoo==2024.11.4pip install xformers==0.0.25</li><li><a href="https://github.com/ggml-org/llama.cpp">GitHub - ggml-org/llama.cpp: LLM inference in C/C++</a>: LLM inference in C/C++. Contribute to ggml-org/llama.cpp development by creating an account on GitHub.</li><li><a href="https://github.com/unslothai/unsloth/issues/2204">torch._dynamo.exc.UserError: Dynamic control flow is not supported at the moment. · Issue #2204 · unslothai/unsloth</a>: I took the Phi 4 GRPO notebook and switched out the model for Phi 3 Mini 128k Instruct, had to disable use_vllm, but then running the code results in Traceback (most recent call last): File &quot;/hom...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1355512015090876588)** (2 messages): 

> `OdysseyXL-V2.5 Code Request` 


- **Code sharing for OdysseyXL-V2.5 requested**: A user requested the code for [open-neo/OdysseyXL-V2.5](https://huggingface.co/collections/open-neo/odysseyxl-67d4cf53fa315a2e04ca20d5).
- **Another topic**: Another summary.



**Link mentioned**: <a href="https://huggingface.co/collections/open-neo/odysseyxl-67d4cf53fa315a2e04ca20d5">OdysseyXL - a open-neo Collection</a>: no description found

  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1355285518463533318)** (88 messages🔥🔥): 

> `GRPO notebooks, reward function, llama 3.1 8b finetuning, ggml-org/llama.cpp quantization, Openllm leaderboard` 


- **Reward Reasoning Reconfiguration Requested**: A member asked about modifying the **reasoning process** in **GRPO notebooks**, and was advised to simply change the **reward function**.
- **Llama 3.1 Fine-Tuning Faceoff**: A member evaluated their finetuned **Llama 3.1 8b** model using **similarity scores** and sought validation for their approach.
   - Other members suggested using **BLEU score** or similar metrics, while some cautioned against relying solely on **similarity scores** due to the stochastic nature of models.
- **Quantization Quest Quells Quandaries in llama.cpp**: A member shared a [pull request](https://github.com/ggml-org/llama.cpp/pull/12511) that adds the ability to quantize other tensors, beyond token-embedding and output-tensor, for most supported architectures, except **Mamba**, **RWKV6**, **RWKV6QWEN2** and **T5**.
   - Another member noted that this work aims to improve **GGUF quants** to be more accurate and capable at different **bits-per-weight (bpw)**, similar to **ExLlama2's quants**.
- **Latent Space Verification vanquishes Veracity Void**: A member shared their [first paper](https://github.com/jacobwarren/Latent-Space-Verification-for-Self-Correcting-LLMs) about LLMs knowing when they're hallucinating and a mechanism for **self-correction in latent space**.
   - Another member inquired about the metrics used to detect **hallucinations**, especially in **out-of-distribution scenarios**.
- **Benchmark Bonanza: Best Bet for Beating Bad Benchmarks**: A member asked for advice on which **leaderboard or eval** to use for comparing models' general performance.
   - Another member argued that *there is no such thing as general performance* and that models excel in different verticals. They suggested **SWE bench**, **aider polygot**, **RULER**, and **AIME** for specific evaluations.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/bill-nye-you-have-no-idea-you-literally-dont-know-what-youre-talking-about-science-guy-gif-4774360">Bill Nye You Have No Idea GIF - Bill Nye You Have No Idea You Literally Dont Know What Youre Talking About - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/ggml-org/llama.cpp/pull/12511">quantize: Handle user-defined quantization levels for additional tensors by EAddario · Pull Request #12511 · ggml-org/llama.cpp</a>: This PR adds the ability to quantize other tensors, beyond token-embedding and output-tensor. It handles most of the supported architectures. except Mamba, RWKV6, RWKV6QWEN2 and T5 to avoid having ...
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1356278493066821672)** (2 messages): 

> `Auto Top Ups issues, Stripe Metadata Mismatch, Credits Added` 


- **Auto Top-Ups Fail Due to Stripe Glitch**: Auto top-up functionality was temporarily disrupted due to changes in **payment metadata** that caused a silent error when the expected data from **Stripe** was not received.
   - The feature has been restored by rolling back the changes, and the team is addressing missing credits and system improvements to prevent future occurrences.
- **Credits Incoming After Auto Top Up Outage**: The issue causing the auto top-up outage has been fully resolved, and **all missing credits** have been added to the affected accounts.
   - Impacted users will receive an email notification regarding the resolution.
- **Root Cause: Stripe Data Format and Faulty Error Logger**: The root cause of the outage was a **data formatting mismatch from Stripe**, exacerbated by inadequate automated testing and a faulty error logger.
   - Enhanced monitoring, error tracking, and end-to-end testing have been implemented to avoid recurrence; users experiencing ongoing issues should contact the team via email for further assistance.


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1355256929928347799)** (402 messages🔥🔥): 

> `Output image models timeline, OpenRouter prompt caching, Agent Hustle, GPT-4o, Free models rate limits` 


- **Output Image Models Incoming**: Members discussed the arrival of output image models, anticipating their integration into platforms like **OpenRouter** with models like **GPT-4o** and **Gemini**.
   - A member expressed excitement about switching directly to **OpenRouter** once these models are available, moving away from using **Gemini's**.
- **Prompt Caching Savings at OpenRouter**: OpenRouter supports prompt caching to save on inference costs, with most providers automatically enabling it; Anthropic requires enabling it on a per-message basis, as documented [here](https://openrouter.ai/docs/features/prompt-caching).
   - Users can inspect caching savings on the [Activity page](https://openrouter.ai/activity) or via the API, with the *cache_discount* field indicating savings from cache usage.
- **Agent Hustle Project Overview**: A member shared details about their project, **Agent Hustle**, a stock trading LLM that utilizes a **TEE wallet** to collect small fees on every transaction.
   - The system strings together about **12 function calls** in total, as exemplified [here](https://h.uguu.se/aeNHgFaf.png).
- **Concerns about rate limiting**: Members reported experiencing rate limits on **Google/Gemini-2.5-pro-exp-03-25:free**, with one user receiving the error *Rate limit exceeded, please try again 45906 seconds later*.
   - OpenRouter's team clarified that rate limits can originate from **Google** or **OpenRouter**, and specifying providers limits OpenRouter's ability to load balance effectively; [check this documentation for rate limits](https://openrouter.ai/docs/api-reference/limits).
- **OpenRouter Adds BYOK Fee**: When using your own **OpenAI API key** with OpenRouter, a **5% fee** is applied to the costs charged by OpenAI for each generation, which is then deducted from the user's OpenRouter credits.
   - This fee is applicable only on credits provided by the provider and not on credits used directly with upstream providers like AWS.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openrouter.ai/openai/chatgpt-4o-latest)">Discord</a>: no description found</li><li><a href="https://openrouter.ai/api/v1">Discord</a>: no description found</li><li><a href="https://openrouter.ai/docs/api-reference/limits">API Rate Limits - Manage Model Usage and Quotas</a>: Learn about OpenRouter&#x27;s API rate limits, credit-based quotas, and DDoS protection. Configure and monitor your model usage limits effectively.</li><li><a href="https://openrouter.ai/openai/gpt-4o">GPT-4o - API, Providers, Stats</a>: GPT-4o (&quot;o&quot; for &quot;omni&quot;) is OpenAI&#x27;s latest AI model, supporting both text and image inputs with text outputs. It maintains the intelligence level of [GPT-4 Turbo](/models/open...</li><li><a href="https://openrouter.ai/docs/features/prompt-caching">Prompt Caching - Optimize AI Model Costs with Smart Caching</a>: Reduce your AI model costs with OpenRouter&#x27;s prompt caching feature. Learn how to cache and reuse responses across OpenAI, Anthropic Claude, and DeepSeek models.</li><li><a href="https://openrouter.ai/settings/credits">OpenRouter</a>: A unified interface for LLMs. Find the best models &amp; prices for your prompts</li><li><a href="https://fal.ai/models/fal-ai/any-llm">Login || fal.ai</a>: no description found</li><li><a href="https://community.openai.com/t/chatgpt-release-notes-2025-march-27-gpt-4o-a-new-update/1153887">ChatGPT — Release Notes: 2025-March-27 - GPT-4o a new update</a>: OpenAI just gave GPT-4o a new update. Based on content on OpenAI help page:   https://help.openai.com/en/articles/6825453-chatgpt-release-notes    GPT-4o feels more intuitive, creative, and collaborat...</li><li><a href="https://openrouter.ai/docs/api-reference/overview#uploading-base64-encoded-images">OpenRouter API Reference - Complete Documentation</a>: Comprehensive guide to OpenRouter&#x27;s API. Learn about request/response schemas, authentication, parameters, and integration with multiple AI model providers.</li><li><a href="https://openrouter.ai/openai/chatgpt-4o-latest">ChatGPT-4o - API, Providers, Stats</a>: OpenAI ChatGPT 4o is continually updated by OpenAI to point to the current version of GPT-4o used by ChatGPT. It therefore differs slightly from the API version of [GPT-4o](/models/openai/gpt-4o) in t...</li><li><a href="https://hastebin.com/share/daqowijupu.python">Hastebin</a>: no description found
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1355275439945285784)** (318 messages🔥🔥): 

> `LM Studio Model Details Fetch Failed Error, VSCode integration with LM Studio, Intel NPU Usage with LM Studio, LM Studio tool use and web search, speculative decoding with LM Studio` 


- ****Fetch Quest Frustrations: User Battles 'Model Details Error'****: A user is struggling with a `Model details error: fetch failed` issue in LM Studio on Windows 11, having tried various fixes like using a Hugging Face Proxy, manually changing hostnames, tweaking DNS settings, using a VPN, and reinstalling.
   - Other members suggested firewall issues, IPV6 problems, or unsupported machine architecture (AVX-only CPU), but the user confirmed they can access Hugging Face in the browser and terminal, and has already tried switching to IPV4.
- ****Continue.dev plugs into LM Studio for sweet VSCode Autocomplete****: A member mentioned that you can connect LM Studio to VSCode via a [VSCode extension](https://www.continue.dev/) that makes custom AI code assistants.
   - They highlight the platform's capabilities in AI-native development, including tab-to-autocomplete and the ability to refer to specific code.
- ****NPU Not Ready: LM Studio Lacks Intel Ultra Integration****: A user asked if LM Studio can take advantage of the NPU in their Intel Ultra PC, to which another member responded that the NPU is not usable by any software yet.
   - Another member pointed to features like [Windows Studio Effects](https://support.microsoft.com/en-us/windows/windows-studio-effects-273c1fa8-2b3f-41b1-a587-7cc7a24b62d8) as examples of Windows features that use NPUs, and specified that they don't know of any LLMs that use it.
- ****LM Studio API: Your Key to Unlocking Tool Use****: Members discussed the options for enabling tool use and web search capabilities within LM Studio, and whether the LM Studio application UI can be modified.
   - It was clarified that tool use is only available via the [LM Studio API](https://lmstudio.ai/docs/app/api/tools), not the ChatUI, leading some to consider modifying Open WebUI as an alternative.
- ****Kokoro TTS and Orpheus battle it out for LM Studio Text-to-Speech Supremacy****: Members inquired about integrating Text-to-Speech (TTS) models with LM Studio, seeking alternatives to OpenAI's speech ability, a user linked [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M), a TTS model, as an option.
   - However, it was mentioned that [CanopyAI's Orpheus](https://huggingface.co/canopylabs/orpheus-3b-0.1-ft) is the only TTS that works in LM Studio (via API, not in chat), and [this repo](https://github.com/isaiahbjork/orpheus-tts-local) is used to run it locally with LM Studio.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2502.05171">Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach</a>: We study a novel language model architecture that is capable of scaling test-time computation by implicitly reasoning in latent space. Our model works by iterating a recurrent block, thereby unrolling...</li><li><a href="https://lmstudio.ai/docs/python">lmstudio-python (Python SDK) | LM Studio Docs</a>: Getting started with LM Studio&#x27;s Python SDK</li><li><a href="https://openrouter.ai/google/gemini-2.5-pro-exp-03-25:free">Gemini Pro 2.5 Experimental (free) - API, Providers, Stats</a>: Gemini 2.5 Pro is Google’s state-of-the-art AI model designed for advanced reasoning, coding, mathematics, and scientific tasks. Run Gemini Pro 2.5 Experimental (free) with API</li><li><a href="https://lmstudio.ai/docs/python/llm-prediction/structured-response">Structured Response | LM Studio Docs</a>: Enforce a structured response from the model using Pydantic models or JSON Schema</li><li><a href="https://support.microsoft.com/en-us/windows/windows-studio-effects-273c1fa8-2b3f-41b1-a587-7cc7a24b62d8">Windows Studio Effects - Microsoft Support</a>: no description found</li><li><a href="https://huggingface.co/hexgrad/Kokoro-82M">hexgrad/Kokoro-82M · Hugging Face</a>: no description found</li><li><a href="https://tenor.com/view/april-fool-gif-25270662">April Fool GIF - April Fool - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.continue.dev/">Continue</a>: Amplified developers, AI-enhanced development · The leading open-source AI code assistant. You can connect any models and any context to build custom autocomplete and chat experiences inside the IDE</li><li><a href="https://aistudio.google.com/">Google AI Studio</a>: Google AI Studio is the fastest way to start building with Gemini, our next generation family of multimodal generative AI models.</li><li><a href="https://github.com/Draconiator/Forgematrix">GitHub - Draconiator/Forgematrix</a>: Contribute to Draconiator/Forgematrix development by creating an account on GitHub.</li><li><a href="https://github.com/openai/openai-python/issues/961>">openai/openai-python</a>: The official Python library for the OpenAI API. Contribute to openai/openai-python development by creating an account on GitHub.</li><li><a href="https://github.com/ggml-org/llama.cpp/issues/11483">Feature Request: Qwen 2.5 VL · Issue #11483 · ggml-org/llama.cpp</a>: Prerequisites I am running the latest code. Mention the version if possible as well. I carefully followed the README.md. I searched using keywords relevant to my issue to make sure that I am creati...</li><li><a href="https://youtu.be/9KKnNh89AGU">Build a LOCAL AI Web Search Assistant with Ollama</a>: Using Ollama to run local LLM&#39;s, in this video I show you how to code a local AI web search assistant. Having an AI that can use the web to respond with up t...</li><li><a href="https://github.com/ggml-org/llama.cpp/tree/master">GitHub - ggml-org/llama.cpp: LLM inference in C/C++</a>: LLM inference in C/C++. Contribute to ggml-org/llama.cpp development by creating an account on GitHub.</li><li><a href="https://huggingface.co/canopylabs/orpheus-3b-0.1-ft">canopylabs/orpheus-3b-0.1-ft · Hugging Face</a>: no description found</li><li><a href="https://github.com/canopyai/Orpheus-TTS">GitHub - canopyai/Orpheus-TTS: TTS Towards Human-Sounding Speech</a>: TTS Towards Human-Sounding Speech. Contribute to canopyai/Orpheus-TTS development by creating an account on GitHub.</li><li><a href="https://github.com/isaiahbjork/orpheus-tts-local">GitHub - isaiahbjork/orpheus-tts-local: Run Orpheus 3B Locally With LM Studio</a>: Run Orpheus 3B Locally With LM Studio. Contribute to isaiahbjork/orpheus-tts-local development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1355256786613178590)** (63 messages🔥🔥): 

> `Epyc systems for ML, LM Studio on older PCs, Saving context with LM Studio API, Mac Studio vs multiple GPUs for inference, Distributed LLM inference` 


- **Epyc Systems Challenge GPUs in Memory Bandwidth**: New **Epyc systems** with high-frequency 12-channel **DDR5** memory can achieve close to **600 GB/s** memory bandwidth, rivaling consumer-grade GPUs for **LLM** performance due to their massive memory capacity.
   - One member suggested that a **10-12k** budget could build a decent Epyc machine capable of running huge models, offering an economical solution for reasonable inference speeds and massive context windows, and no need for GPUs!
- **Old PC gets LM Studio Boost**: A member reported successfully running decent-sized **Qwen** and **Llama** models (6Q quantization) on a 2016 Dell Inspiron laptop (i7 6700HQ, 32GB DDR3, integrated graphics) using **LM Studio** with **CPU AVX2**-compiled runtimes.
   - He was *surprised* the old laptop *still holds its own* and called **LM Studio** *the greatest*!
- **LM Studio API Context Handling**: To maintain conversation context when using the **LM Studio API** with a Telegram bot, the user must store conversation history in a variable (e.g., in JSON format) as the **API** itself does not inherently retain context.
   - It was suggested storing the conversation with a *unique-tg-user-id*, unless it is being hosted on a PC that *constantly reboots*.
- **Mac Studio Tempts Inference Server Builders**: A member pondered whether to build an inference server with multiple **Nvidia cards** or opt for a **Mac Studio** with unified memory, citing [this youtube video](https://www.youtube.com/watch?v=nwIZ5VI3Eus).
   - Another member argued for **Mac Studio** due to lower cost, less electricity usage, and more **RAM**, recommending running **LM Studio** headless for 24/7 operation, noting it supports **MLX** models.
- **Distributed Inference Projects Emerge**: In response to a query about **LM Studio** supporting multiple machines, two projects were linked for distributed LLM inference: [exo](https://github.com/exo-explore/exo) and [distributed-llama](https://github.com/b4rtaz/distributed-llama).
   - These projects aim to connect home devices into a powerful cluster to accelerate **LLM** inference, with more devices implying faster speeds.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lmstudio.ai/ryzenai">LM Studio on Ryzen AI</a>: Run Llama, Mistral, Mixtral, and other local LLMs on your PC, leveraging the awesome performance of RyzenAI hardware.</li><li><a href="https://www.supermicro.com/en/products/motherboard/H11SSL-i">H11SSL-i | Motherboards | Super Micro Computer, Inc.</a>: no description found</li><li><a href="https://github.com/exo-explore/exo">GitHub - exo-explore/exo: Run your own AI cluster at home with everyday devices 📱💻 🖥️⌚</a>: Run your own AI cluster at home with everyday devices 📱💻 🖥️⌚ - exo-explore/exo</li><li><a href="https://github.com/b4rtaz/distributed-llama">GitHub - b4rtaz/distributed-llama: Connect home devices into a powerful cluster to accelerate LLM inference. More devices means faster inference.</a>: Connect home devices into a powerful cluster to accelerate LLM inference. More devices means faster inference. - b4rtaz/distributed-llama
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1355257705782182059)** (117 messages🔥🔥): 

> `FxEmbed, MCP, Sam Altman WSJ, Replit v2, n8n` 


- **Altman's Firing Gets WSJ Treatment**: The **WSJ** published an article detailing the real story behind **Sam Altman's** firing from the **OpenAI** board, alleging that *he lied about safety testing for new releases* ([archive link](https://archive.ph/2025.03.29-230008/https://www.wsj.com/tech/ai/the-real-story-behind-sam-altman-firing-from-openai-efd51a5d)).
- **Replit v2 is Impressive**: A member found **Replit v2 agent** very impressive for prototyping and building MVPs, probably using **Sonnet 3.7** under the hood, and also easy to extract and use in one's own backend.
   - It was noted that **Replit** has direct access to logs, configured infrastructure, and sets up logging which makes the process smooth; **Cursor** is better for existing deployments, but the managed infrastructure gives **Replit** an edge.
- **OpenAI to Open-Weight Model**: **OpenAI** plans to release an open-weight language model with reasoning capabilities in the coming months, seeking feedback from developers ([OpenAI open model feedback](https://openai.com/open-model-feedback/)).
   - The company aims to host developer events in **SF**, **Europe**, and **APAC** to gather insights and provide early prototypes for experimentation.
- **Cursor Closes Huge Round**: **Cursor** closed a **$625M** round at a **$9.6B** post valuation, led by **Thrive** & **A16z**, with **Accel** as a new backer ([tweet](https://x.com/ArfurRock/status/1906768733135098360)).
   - This valuation comes after sparking the buzzphrase *vibe coding*, seeing its valuation increase from **$400M** to **$2.5B** to potentially **$10B** in less than a year.
- **Etched Enters the ASIC Arena**: **Etched**, the first transformer **ASIC**, closed an unannounced **$85M** at **$1.5B**, following two stealth rounds at **$500M** then **$750M** ([tweet](https://x.com/ArfurRock/status/1906756943349260682)).
   - **Etched**'s chip **Sohu** runs **Llama 70B** at *over 500,000 tokens per second*, and one 8xSohu server replaces 160 H100s.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://runwayml.com/research/introducing-runway-gen-4">Runway Research | Introducing Runway Gen-4</a>: no description found</li><li><a href="https://x.com/sewoong79/status/1906595129965912341?s=46&t=jDrfS5vZD4MFwckU5E8f5Q">Tweet from Sewoong Oh (@sewoong79)</a>: We are releasing OpenDeepSearch (ODS), an open-source search agent that works with any LLM. When paired with DeepSeek-R1, ODS outperforms OpenAI’s specialized model for web search, GPT-4o-Search, on t...</li><li><a href="https://x.com/AmazonScience/status/1906758835240312882">Tweet from Amazon Science (@AmazonScience)</a>: Meet Amazon Nova Act — an effortless way to build AI agents that can reliably use browsers 🧑‍💻With our new model, compose robust steps into complex workflows; handle everything from bookings to QA t...</li><li><a href="https://x.com/TheXeophon/status/1906654834255954049]">Tweet from Xeophon (@TheXeophon)</a>: Super excited to share what I&#39;ve been working on 👀You know the struggle: You want to use a project, but it has a GPL or CC-by-NC license 😭We worked hard and our AI-based agents convert any repo ...</li><li><a href="https://fxtwitter.com/peterwildeford/status/1906089368613490736">Tweet from Peter Wildeford 👊 🇺🇸 🔥 (@peterwildeford)</a>: This WSJ article, if true, has some real bombshells about @OpenAI and @sama 💣‼️It is alleged that Sam Altman clearly lied multiple times to a variety of peopleSuch as Altman lying to the board about ...</li><li><a href="https://fxtwitter.com/sama/status/1906793591944646898">Tweet from Sam Altman (@sama)</a>: TL;DR: we are excited to release a powerful new open-weight language model with reasoning in the coming months, and we want to talk to devs about how to make it maximally useful: https://openai.com/op...</li><li><a href="https://x.com/sama/status/1906793591944646898">Tweet from Sam Altman (@sama)</a>: TL;DR: we are excited to release a powerful new open-weight language model with reasoning in the coming months, and we want to talk to devs about how to make it maximally useful: https://openai.com/op...</li><li><a href="https://runwayml.com/gen-4-bts">Behind the Scenes with Gen-4</a>: A collection of short films and music videos made entirely with Gen-4 to test the model&#x27;s narrative capabilities.</li><li><a href="https://x.com/rauchg/status/1906814800426086861?s=46">Tweet from Guillermo Rauch (@rauchg)</a>: We&#39;re building an API to run arbitrary compute, targeting agentic AI usecases and long-running tasks. Yes, it can run servers.Powered by the infra that runs our 1M+ daily @vercel builds, optimized...</li><li><a href="https://x.com/ArfurRock/status/1906768733135098360]">Tweet from Arfur Rock (@ArfurRock)</a>: Cursor round closed — $625M at $9.6B post led by Thrive & A16z. Accel is a new backer.$200M ARR, up 4x from $2.5B round in November 2024.ARR multiple constant from last round at 50x.Quoting Abe Brown ...</li><li><a href="https://fxtwitter.com/TheXeophon/status/1906654834255954049">Tweet from Xeophon (@TheXeophon)</a>: Super excited to share what I&#39;ve been working on 👀You know the struggle: You want to use a project, but it has a GPL or CC-by-NC license 😭We worked hard and our AI-based agents convert any repo ...</li><li><a href="https://x.com/demishassabis/status/1906664622226083922?s=46">Tweet from Demis Hassabis (@demishassabis)</a>: Thrilled to announce @IsomorphicLabs has raised $600M to turbocharge our mission to one day solve all disease with the help of AI.I&#39;ve long felt that improving human health is the most important t...</li><li><a href="https://x.com/TheXeophon/status/1906654834255954049">Tweet from Xeophon (@TheXeophon)</a>: Super excited to share what I&#39;ve been working on 👀You know the struggle: You want to use a project, but it has a GPL or CC-by-NC license 😭We worked hard and our AI-based agents convert any repo ...</li><li><a href="https://fxtwitter.com/stevenheidel/status/1906797154301329845">Tweet from Steven Heidel (@stevenheidel)</a>: we&#39;re releasing a model this year that you can run on your own hardwareQuoting Sam Altman (@sama) TL;DR: we are excited to release a powerful new open-weight language model with reasoning in the c...</li><li><a href="https://x.com/jie_liu1/status/1905761704195346680">Tweet from Jie Liu (@jie_liu1)</a>: After hacking GPT-4o&#39;s frontend, I made amazing discoveries:💡The line-by-line image generation effect users see is just a browser-side animation (pure frontend trick)🔦OpenAI&#39;s server sends o...</li><li><a href="https://x.com/peterwildeford/status/1906089368613490736?s=46&t=Z6mP_1pHALnIw7k1lFkdwQ">Tweet from Peter Wildeford 👊 🇺🇸 🔥 (@peterwildeford)</a>: This WSJ article, if true, has some real bombshells about @OpenAI and @sama 💣‼️It is alleged that Sam Altman clearly lied multiple times to a variety of peopleSuch as Altman lying to the board about ...</li><li><a href="https://x.com/ArfurRock/status/1906768733135098360">Tweet from Arfur Rock (@ArfurRock)</a>: Cursor round closed — $625M at $9.6B post led by Thrive & A16z. Accel is a new backer.$200M ARR, up 4x from $2.5B round in November 2024.ARR multiple constant from last round at 50x.Quoting Abe Brown ...</li><li><a href="https://x.com/ArfurRock/status/1906756943349260682">Tweet from Arfur Rock (@ArfurRock)</a>: 🚨New unicorn alert — Etched, word&#39;s first transformer ASICClosed an unannounced $85M at $1.5B, following two other stealth rounds at $500M then $750M.The $750M round was just ~2 months ago.Quotin...</li><li><a href="https://www.interconnects.ai/p/papers-im-reading-base-model-rl-grpo">Recent reasoning research: GRPO tweaks, base model RL, and data curation</a>: The papers I endorse as worth reading among a cresting wave of reasoning research.</li><li><a href="https://x.com/sewoong79/status/1906595129965912341?s=46&t=jDrfS5vZD4MFwckU5E8f5Q]">Tweet from Sewoong Oh (@sewoong79)</a>: We are releasing OpenDeepSearch (ODS), an open-source search agent that works with any LLM. When paired with DeepSeek-R1, ODS outperforms OpenAI’s specialized model for web search, GPT-4o-Search, on t...</li><li><a href="https://x.com/AmazonScience/status/1906758835240312882]">Tweet from Amazon Science (@AmazonScience)</a>: Meet Amazon Nova Act — an effortless way to build AI agents that can reliably use browsers 🧑‍💻With our new model, compose robust steps into complex workflows; handle everything from bookings to QA t...</li><li><a href="https://fxtwitter.com/AmazonScience/status/1906758835240312882">Tweet from Amazon Science (@AmazonScience)</a>: Meet Amazon Nova Act — an effortless way to build AI agents that can reliably use browsers 🧑‍💻With our new model, compose robust steps into complex workflows; handle everything from bookings to QA t...</li><li><a href="https://fxtwitter.com/sewoong79/status/1906595129965912341">Tweet from Sewoong Oh (@sewoong79)</a>: We are releasing OpenDeepSearch (ODS), an open-source search agent that works with any LLM. When paired with DeepSeek-R1, ODS outperforms OpenAI’s specialized model for web search, GPT-4o-Search, on t...</li><li><a href="https://x.com/ArfurRock/status/1906756943349260682]">Tweet from Arfur Rock (@ArfurRock)</a>: 🚨New unicorn alert — Etched, word&#39;s first transformer ASICClosed an unannounced $85M at $1.5B, following two other stealth rounds at $500M then $750M.The $750M round was just ~2 months ago.Quotin...</li><li><a href="https://x.com/iscienceluvr/status/1906790937604579430?s=46">Tweet from Tanishq Mathew Abraham, Ph.D. (@iScienceLuvr)</a>: I have EXCITING news:I&#39;ve started a company!Introducing SophontWe’re building open multimodal foundation models for the future of healthcare. We need a DeepSeek for medical AI, and @SophontAI will...</li><li><a href="https://x.com/stevenheidel/status/1906797154301329845">Tweet from Steven Heidel (@stevenheidel)</a>: we&#39;re releasing a model this year that you can run on your own hardwareQuoting Sam Altman (@sama) TL;DR: we are excited to release a powerful new open-weight language model with reasoning in the c...</li><li><a href="https://x.com/iscienceluvr/status/1906790937604579430?s=4]">Tweet from Tanishq Mathew Abraham, Ph.D. (@iScienceLuvr)</a>: I have EXCITING news:I&#39;ve started a company!Introducing SophontWe’re building open multimodal foundation models for the future of healthcare. We need a DeepSeek for medical AI, and @SophontAI will...</li><li><a href="https://fxtwitter.com/demishassabis/status/1906664622226083922">Tweet from Demis Hassabis (@demishassabis)</a>: Thrilled to announce @IsomorphicLabs has raised $600M to turbocharge our mission to one day solve all disease with the help of AI.I&#39;ve long felt that improving human health is the most important t...</li><li><a href="https://fxtwitter.com/ArfurRock/status/1906756943349260682">Tweet from Arfur Rock (@ArfurRock)</a>: 🚨New unicorn alert — Etched, word&#39;s first transformer ASICClosed an unannounced $85M at $1.5B, following two other stealth rounds at $500M then $750M.The $750M round was just ~2 months ago.Quotin...</li><li><a href="https://fxtwitter.com/rauchg/status/1906814800426086861">Tweet from Guillermo Rauch (@rauchg)</a>: We&#39;re building an API to run arbitrary compute, targeting agentic AI usecases and long-running tasks. Yes, it can run servers.Powered by the infra that runs our 1M+ daily @vercel builds, optimized...</li><li><a href="https://fxtwitter.com/ArfurRock/status/1906768733135098360">Tweet from Arfur Rock (@ArfurRock)</a>: Cursor round closed — $625M at $9.6B post led by Thrive & A16z. Accel is a new backer.$200M ARR, up 4x from $2.5B round in November 2024.ARR multiple constant from last round at 50x.Quoting Abe Brown ...</li><li><a href="https://fxtwitter.com/iscienceluvr/status/1906790937604579430">Tweet from Tanishq Mathew Abraham, Ph.D. (@iScienceLuvr)</a>: I have EXCITING news:I&#39;ve started a company!Introducing SophontWe’re building open multimodal foundation models for the future of healthcare. We need a DeepSeek for medical AI, and @SophontAI will...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1jnzdvp/qwen3_support_merged_into_transformers/">Reddit - The heart of the internet</a>: no description found</li><li><a href="https://x.com/egeberkina/status/1906088423988875617?s=46">Tweet from Ege (@egeberkina)</a>: GPT4o is literally cooked 🔥👨‍🍳Visual recipes are here and they’re actually kinda genius!Prompt in ALT</li><li><a href="https://x.com/stuff/posts/and/things/2398753298579">Tweet from GitHub - FxEmbed/FxEmbed: Fix X/Twitter and Bluesky embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others</a>: Fix X/Twitter and Bluesky embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others - FxEmbed/FxEmbed</li><li><a href="https://x.com">Tweet from GitHub - FxEmbed/FxEmbed: Fix X/Twitter and Bluesky embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others</a>: Fix X/Twitter and Bluesky embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others - FxEmbed/FxEmbed</li><li><a href="https://fxtwitter.com/egeberkina/status/1906088423988875617">Tweet from Ege (@egeberkina)</a>: GPT4o is literally cooked 🔥👨‍🍳Visual recipes are here and they’re actually kinda genius!Prompt in ALT</li><li><a href="https://fxtwitter.com/stuff/posts/and/things/2398753298579">Tweet from GitHub - FxEmbed/FxEmbed: Fix X/Twitter and Bluesky embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others</a>: Fix X/Twitter and Bluesky embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others - FxEmbed/FxEmbed</li><li><a href="https://archive.ph/2025.03.29-230008/https://www.wsj.com/tech/ai/the-real-story-behind-sam-altman-firing-from-openai-efd51a5d">Exclusive | The Secrets and Misdirection Behind Sam Altman&#x2019;s Firing F&#x2026;</a>: no description found
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1355270295279370370)** (189 messages🔥🔥): 

> `LLM-based code generation, Code documentation strategies, Memory-Ref MCP server, Cursor IDE issues, llms.txt project` 


- **Harper Reveals LLM Codegen Workflow**: A member shared a [blog post](https://harper.blog/2025/02/16/my-llm-codegen-workflow-atm/) detailing their **LLM codegen workflow**, emphasizing a structured approach involving brainstorming, planning, and execution in discrete loops.
   - The post highlights the importance of having a well-defined plan to avoid wasting time when building small products using **LLMs**.
- **Docs.dev Automates Code Documentation**: [Docs.dev](https://docs.dev/) was shared as a tool to *generate docs directly from your codebase and existing content* and keep them up to date as code changes.
   - It integrates with **GitHub** and offers features like automated doc generation from **PRs**, bulk modification, and analysis for SEO optimization.
- **Memory-Ref Powers Persistent Coding Preferences in Cursor IDE**: A member shared a [HN post](https://news.ycombinator.com/item?id=43506068) about **Cursor IDE** integrating with **Graphiti**, an open-source temporal knowledge graph, to provide persistent memory across sessions using **Memory-Ref MCP**.
   - The integration aims to help **Cursor** remember coding preferences and project specs, reducing the need for constant reminders.
- **Navigating Documentation for LLMs and Humans**: Members discussed whether documentation for **LLMs** requires a different level of verbosity compared to documentation for humans, mentioning that markdown is becoming a go-to "programming language".
   - One member linked to an example of their **ttmp** directory to show their **Github** [documentation style](https://github.com/go-go-golems/go-go-labs/blob/main/ttmp/2025-03-23/03-add-embeddings-to-command.md) that they have found effective with language models.
- **llms.txt Standard Proposed for LLM Website Crawling**: The **llms.txt** project, aimed at helping language models effectively use website data, was shared [on GitHub](https://github.com/AnswerDotAI/llms-txt).
   - The file is designed to give **LLMs** instructions on how to crawl and use website content, similar to **robots.txt**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.codeguide.dev/">CodeGuide</a>: CodeGuide creates Detailed Documentation for your AI Coding Project.</li><li><a href="https://x.com/PrajwalTomar_/status/1895839765280539068?s=19">Tweet from Prajwal Tomar (@PrajwalTomar_)</a>: In last 5 months, I’ve built 16 SaaS products for clients using Cursor.Now, I’ve cracked the best AI coding workflow for Cursor.Here’s my step-by-step guide to building production-ready MVPs:</li><li><a href="https://news.ycombinator.com/item?id=43506068">Show HN: Cursor IDE now remembers your coding prefs using MCP | Hacker News</a>: no description found</li><li><a href="https://harper.blog/2025/02/16/my-llm-codegen-workflow-atm/">My LLM codegen workflow atm</a>: A detailed walkthrough of my current workflow for using LLms to build software, from brainstorming through planning and execution.</li><li><a href="https://docs.dev/">Docs.dev | AI-assisted docs</a>: Generate docs directly from your codebase and existing docs. Ensure your docs stay up to date as code changes.</li><li><a href="https://docs.google.com/spreadsheets/d/1q5rwO4wleMTLXr1z58c2UC03QsDsGwbJY1v4UG7eEOs/edit?gid=1439059137#gid=1439059137">AI In Action: Weekly Jam Sessions</a>: no description found</li><li><a href="https://github.com/AnswerDotAI/llms-txt">GitHub - AnswerDotAI/llms-txt: The /llms.txt file, helping language models use your website</a>: The /llms.txt file, helping language models use your website - AnswerDotAI/llms-txt</li><li><a href="https://github.com/nuvic/fzf-kit.nvim">GitHub - nuvic/fzf-kit.nvim: A Neovim plugin that extends fzf-lua with additional utilities</a>: A Neovim plugin that extends fzf-lua with additional utilities - nuvic/fzf-kit.nvim</li><li><a href="https://github.com/go-go-golems/go-go-mcp/tree/main/ttmp">go-go-mcp/ttmp at main · go-go-golems/go-go-mcp</a>: Anthropic MCP go implementation. Contribute to go-go-golems/go-go-mcp development by creating an account on GitHub.</li><li><a href="https://github.com/joernio/astgen">GitHub - joernio/astgen: Generate AST in json format for JS/TS</a>: Generate AST in json format for JS/TS. Contribute to joernio/astgen development by creating an account on GitHub.</li><li><a href="https://github.com/go-go-golems/go-go-labs/blob/main/ttmp/2025-03-23/03-add-embeddings-to-command.md">go-go-labs/ttmp/2025-03-23/03-add-embeddings-to-command.md at main · go-go-golems/go-go-labs</a>: GO GO EXPERIMENTAL LAB. Contribute to go-go-golems/go-go-labs development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1355263770028412929)** (263 messages🔥🔥): 

> `MCP spec updates, HTTP Streamable Transport, OpenAI Agents SDK, UVX MCP Server, Model Context Protocol` 


- **MCP spec embraces OAuth 2.1**: The new **2025-03-26 MCP spec** draft includes new auth features like **OAuth 2.1**, but no client supports it yet for testing; see [MCP spec](https://spec.modelcontextprotocol.io/specification/2025-03-26/basic/authorization/).
- **HTTP Streamable Transport raises Resumability Questions**: Doubts arise on how **HTTP Streamable Transport** resumes sessions correctly, especially the server's obligation to avoid replaying messages on different streams, which seems hypothetical.
   - The spec says *The server MUST NOT send a JSON-RPC response on the stream unless resuming a stream associated with a previous client request*, which contradicts the resumability goal.
- **Env Variables prove slippery for Tool Integration**: Members discussed using **environment variables** to pass API tokens to tools, where debugging with `@modelcontextprotocol/inspector` works but calling the tool in an **MCP client** throws unauthorized errors.
   - Passing the token directly in the `claude_desktop_config.json` file seemingly fixed the issue.
- **Progress Notifications prove trickiest for MCP**: Users seek examples for sending notifications from the server to the client, exploring `notification/progress` for long-running resources and discovering the **client sends back a request to `/message`**.
   - Notifications might need pre-declaration or aren't fully supported in all clients like **Claude Desktop**, where only the spinner works, but messages don't appear, and the progressToken is essential.
- **Goose bumps into Endpoint Description issue**: Some users reported errors connecting **Goose** to a local MCP server over SSE, but a quick fix involves adding descriptions to server endpoints, as suggested in this [Github issue](https://github.com/block/goose/issues/1880).
   - After the layoff, Goose seems to be working OK.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://jsonlint.com/">JSON Online Validator and Formatter - JSON Lint</a>: no description found</li><li><a href="https://spec.modelcontextprotocol.io/specification/2025-03-26/basic/transports/#listening-for-messages-from-the-server">Transports</a>:           ℹ️                  Protocol Revision: 2025-03-26      MCP uses JSON-RPC to encode messages. JSON-RPC messages MUST be UTF-8 encoded.The protocol currently defines two standard transport mec...</li><li><a href="https://openai.github.io/openai-agents-python/mcp/">Model context protocol (MCP) - OpenAI Agents SDK</a>: no description found</li><li><a href="https://tenor.com/view/joke-missed-over-my-head-gif-26041934">Joke Missed GIF - Joke Missed Over My Head - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://block.github.io/goose/docs/getting-started/installation/">Install Goose | codename goose</a>: Choose to install Goose on CLI and/or Desktop:</li><li><a href="https://glama.ai/api/mcp/openapi.json",">MCP API Reference</a>: API Reference for the Glama Gateway</li><li><a href="https://docs.zapier.com/ai-actions/how-tos/auth">Authentication - Zapier</a>: no description found</li><li><a href="https://github.com/modelcontextprotocol/servers/tree/main/src/puppeteer">servers/src/puppeteer at main · modelcontextprotocol/servers</a>: Model Context Protocol Servers. Contribute to modelcontextprotocol/servers development by creating an account on GitHub.</li><li><a href="https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem">servers/src/filesystem at main · modelcontextprotocol/servers</a>: Model Context Protocol Servers. Contribute to modelcontextprotocol/servers development by creating an account on GitHub.</li><li><a href="https://developers.cloudflare.com/agents/model-context-protocol/authorization/">Authorization · Cloudflare Agents docs</a>: When building a Model Context Protocol (MCP) server, you need both a way to allow users to login (authentication) and allow them to grant the MCP client access to resources on their account (authoriza...</li><li><a href="https://github.com/cloudflare/ai/tree/main/demos/remote-mcp-github-oauth">ai/demos/remote-mcp-github-oauth at main · cloudflare/ai</a>: Contribute to cloudflare/ai development by creating an account on GitHub.</li><li><a href="https://github.com/mo">mo - Overview</a>: mo has 49 repositories available. Follow their code on GitHub.</li><li><a href="https://github.com/Abiorh001/mcp_omni_connect">GitHub - Abiorh001/mcp_omni_connect: MCPOmni Connect is a versatile command-line interface (CLI) client designed to connect to various Model Context Protocol (MCP) servers using stdio transport. It provides seamless integration with OpenAI models and supports dynamic tool and resource management across multiple servers.</a>: MCPOmni Connect is a versatile command-line interface (CLI) client designed to connect to various Model Context Protocol (MCP) servers using stdio transport. It provides seamless integration with O...</li><li><a href="https://github.com/angiejones/mcp-selenium">GitHub - angiejones/mcp-selenium: An MCP implementation for Selenium WebDriver</a>: An MCP implementation for Selenium WebDriver. Contribute to angiejones/mcp-selenium development by creating an account on GitHub.</li><li><a href="https://mcp.pipedream.com">Pipedream MCP</a>: Access MCP servers for more than 2,500 APIs with 8,000 prebuilt tools</li><li><a href="https://mcp.pipedream.com/app/people_data_labs">People Data Labs MCP Server | Pipedream</a>: The source of the truth for person data</li><li><a href="https://github.com/block/goose/issues/1880">Goose stops responding when custom MCP is added · Issue #1880 · block/goose</a>: Describe the bug Goose becomes unresponsive when I enable my MCP server. The server appears to work correctly with other tools and appears to start correctly with the toasts in the UI. I do not see...</li><li><a href="https://github.com/tadata-org/fastapi_mcp">GitHub - tadata-org/fastapi_mcp: A zero-configuration tool for automatically exposing FastAPI endpoints as Model Context Protocol (MCP) tools.</a>: A zero-configuration tool for automatically exposing FastAPI endpoints as Model Context Protocol (MCP) tools. - tadata-org/fastapi_mcp</li><li><a href="https://github.com/punkpeye/awesome-mcp-clients">GitHub - punkpeye/awesome-mcp-clients: A collection of MCP clients.</a>: A collection of MCP clients. Contribute to punkpeye/awesome-mcp-clients development by creating an account on GitHub.</li><li><a href="https://developer.adobe.com/premiere-pro/uxp/">no title found</a>: no description found</li><li><a href="https://github.com/Abiorh001/mcp_ev_assistant_server/blob/main/ev_assitant_server.py">mcp_ev_assistant_server/ev_assitant_server.py at main · Abiorh001/mcp_ev_assistant_server</a>:  A powerful server implementation for managing Electric Vehicle (EV) charging stations, trip planning, and resource management. This server provides a comprehensive set of tools and APIs for EV-rel...</li><li><a href="https://github.com/trending">Build software better, together</a>: GitHub is where people build software. More than 150 million people use GitHub to discover, fork, and contribute to over 420 million projects.
</li>
</ul>

</div>
  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1355274687978147861)** (28 messages🔥): 

> `Speech MCP, Reverse engineering in IDA Pro, OpenAPI MCP server for Cursor, AI-powered RAG application, MCP server development with hot reload` 


- ****Speech MCP** Demonstration**: A user shared a [YouTube short](https://www.youtube.com/shorts/rurAp_WzOiY) showcasing **Speech MCP**.
   - Another user inquired if there's a version compatible with **Claude**.
- ****IDA Pro MCP Server** Streamlines Reverse Engineering**: An **IDA Pro MCP server** was created to automate reverse engineering, with a [streamlined installation process](https://x.com/mrexodia/status/1906010119940239544) allowing users to experiment with vibe reversing in under 2 minutes.
   - The server was tested using **Claude**, and is automatically configured with **Cline** and **Roo Code**.
- ****OpenAPI MCP Server** Integrates with Cursor**: An **OpenAPI MCP server** was developed to enable **Cursor** to directly understand API specifications, available on [GitHub](https://github.com/ReAPI-com/mcp-openapi).
   - The developer is seeking feedback from users who try it out.
- ****CATIE** Intelligently Routes MCP Requests**: **CATIE (Context Aware Traffic Ingress Engine)**, a proxy for routing MCP requests based on tool call, was released on [GitHub](https://github.com/mclenhard/catie-mcp).
   - This free, open-source tool allows routing to different MCP servers based on tool call parameters, real-time monitoring, backend switching, and simple load distribution.
- ****Pipedream** Launches MCP Server with User Authentication**: **Pipedream** launched an MCP server on [GitHub](https://github.com/PipedreamHQ/pipedream/tree/master/modelcontextprotocol) that enables developers to run their own MCP server for 2,500+ apps and manage servers for their users, with managed authentication.
   - According to Pipedream, managed authentication with approved clients is a requirement for MCP to work at scale.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/mrexodia/status/1906010119940239544">Tweet from Duncan Ogilvie 🍍 (@mrexodia)</a>: Streamlined the installation process of my IDA Pro MCP server. You can now start experimenting with vibe reversing in less than 2 minutes!🤯This was using Claude for testing, but Cline and Roo Code ar...</li><li><a href="https://ai-odyssey-planner.lovable.app/)">AI Odyssey Planner - Plan Your Perfect Trip in Minutes</a>: Our AI travel assistant analyzes millions of data points to create personalized itineraries based on your preferences, budget, and schedule.</li><li><a href="https://github.com/PipedreamHQ/pipedream/tree/master/modelcontextprotocol">pipedream/modelcontextprotocol at master · PipedreamHQ/pipedream</a>: Connect APIs, remarkably fast.  Free for developers. - PipedreamHQ/pipedream</li><li><a href="https://pipedream.com/connect">Connect</a>: Pipedream is the fastest way to build powerful applications that connect all the services in your stack, with code-level control when you need it and no code when you don't.</li><li><a href="https://github.com/ReAPI-com/mcp-openapi">GitHub - ReAPI-com/mcp-openapi: OpenAPI specification MCP server.</a>: OpenAPI specification MCP server. Contribute to ReAPI-com/mcp-openapi development by creating an account on GitHub.</li><li><a href="https://github.com/Cheffromspace/MCPControl">GitHub - Cheffromspace/MCPControl: MCP server for Windows OS automation</a>: MCP server for Windows OS automation. Contribute to Cheffromspace/MCPControl development by creating an account on GitHub.</li><li><a href="https://github.com/strowk/mcp-autotest">GitHub - strowk/mcp-autotest: Utility for autotesting MCP servers</a>: Utility for autotesting MCP servers. Contribute to strowk/mcp-autotest development by creating an account on GitHub.</li><li><a href="https://github.com/mclenhard/catie-mcp">GitHub - mclenhard/catie-mcp</a>: Contribute to mclenhard/catie-mcp development by creating an account on GitHub.</li><li><a href="https://www.activepieces.com/mcp">280+ Open Source MCPs — Use them on Activepieces now</a>: Give AI access to your apps with 280+ open source MCPs. Use them with Claude, Cursor, or Windsurf to let AI read your emails, manage your calendar, and more.
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1355977048967086291)** (1 messages): 

> `HF Reasoning Course, Gradio Dataframe, Reranker Models, Model Onboarding, Open R1 Update` 


- **HF Reasoning Course Gets DeepSeek Boost**: A new unit in the HF reasoning course features **DeepSeek R1**, according to [this LinkedIn post](https://www.linkedin.com/posts/ben-burtenshaw_new-unit-in-the-hugging-face-reasoning-course-activity-7311046882691108864-0cBN?utm_source=share&utm_medium=member_desktop&rcm=ACoAADxGwTsBLzNXo2rQ00oBRJPg_9dfhulQnio).
- **Gradio's Dataframe Component Gets Turbocharged**: Gradio released a host of new updates to its `gr.Dataframe` component, closing over **70 issues** including bugs, improvements, and enhancements, as detailed in [this blog post](https://huggingface.co/blog/gradio-dataframe-upgrade).
   - The `gr.Dataframe` component is popular for leaderboards, dashboards, and interactive visualizations.
- **Reranker Models Ride Sentence Transformers**: A blog post details how to train and finetune reranker models with **Sentence Transformers v4**, as seen in [this article](https://huggingface.co/blog/train-reranker).
- **Model Onboarding Experience Revamped**: HF launched a new model onboarding experience, aiming to simplify understanding of the hub's capabilities, according to [this tweet](https://x.com/reach_vb/status/1905604906825716112).
- **DeepSeek V3 Gains Impressive Math Skills**: Evaluations on **DeepSeek V3 0324** show impressive gains in math and GPQA but a slight hit in instruction following, according to [this tweet](https://x.com/nathanhabib1011/status/1905018770764259818).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/blog/gradio-dataframe-upgrade">Introducing Gradio&#39;s new Dataframe!</a>: no description found</li><li><a href="https://huggingface.co/blog/train-reranker">Training and Finetuning Reranker Models with Sentence Transformers v4</a>: no description found</li><li><a href="https://x.com/reach_vb/status/1905604906825716112">Tweet from Vaibhav (VB) Srivastav (@reach_vb)</a>: New model on-boarding experience on HF is now live, WDYT?The intent is to make it easy for people to understand what&#39;s already possible on the hub and demystify all the things hub is capable of!</li><li><a href="https://huggingface.co/blog/open-r1/update-4">Open R1: Update #4</a>: no description found</li><li><a href="https://huggingface.co/spaces/enzostvs/deepsite">DeepSite - a Hugging Face Space by enzostvs</a>: no description found</li><li><a href="https://huggingface.co/posts/AdinaY/152448454490712">@AdinaY on Hugging Face: &quot;Let&#39;s check out the latest releases from the Chinese community in March!

👉…&quot;</a>: no description found</li><li><a href="https://huggingface.co/blog/endpoint-analytics">The New and Fresh analytics in Inference Endpoints</a>: no description found</li><li><a href="https://x.com/nathanhabib1011/status/1905018770764259818">Tweet from Nathan (@nathanhabib1011)</a>: Just wrapped up evaluations on @deepseek_ai&#39;s V3 0324! 🚀Impressive gains in math and GPQA, but instruction following took a slight hit. More concerning—AIME25 remains unchanged. Possible contamin...</li><li><a href="https://huggingface.co/blog/burtenshaw/custom-local-coding-vscode">Custom Vibe Coding Quest Part 1: The Quest Begins 🧙</a>: no description found</li><li><a href="https://huggingface.co/blog/intel-gaudi-backend-for-tgi">🚀 Accelerating LLM Inference with TGI on Intel Gaudi</a>: no description found</li><li><a href="https://huggingface.co/blog/giadap/beyond-consent">I Clicked “I Agree”, But What Am I Really Consenting To?</a>: no description found</li><li><a href="https://x.com/hugoch/status/1905561210839298473">Tweet from Hugo Larcher (@hugoch)</a>: 🧠 LLM inference isn’t just about latency — it’s about consistency under load. Different workloads, configs, and hardware = very different real-world performances.At Hugging Face 🤗 we built inference...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1355256066149187744)** (145 messages🔥🔥): 

> `Hugging Face Pro Debit Card Issue, Video Lip Sync tools, RunPod and HuggingFace models, HF Model Containerization, Agentx Competition Research Track` 


- **Hugging Face Pro Debit Card Debacle**: A user reported being charged for a **Hugging Face Pro subscription** with a debit card despite an error message, and inquired about a refund.
   - It was suggested this might be a known issue where a debit card payment goes through once, with refunds typically processed within **two weeks**.
- **Video Retalking Tool Glitches**: A user shared a link to a [VideoRetalking tool](https://huggingface.co/spaces/fffiloni/VideoRetalking) on Hugging Face Spaces, noting it *worked pretty well, a little glitchy*.
   - They also wondered if **SaaS solutions like HeyGen** manipulate emoting movements of the body or just do lip syncing.
- **RunPod's Model Management Conundrum**: A user sought advice on using a model from **Hugging Face** with **RunPod**, struggling to make things work after cloning and improving a model.
   - The user can't afford a good GPU and is also looking for something cool like **video lip sync** type stuff to make faceless videos.
- **HF Space Project Conversion Troubles**: A user sought advice on converting a local Python project to a **Hugging Face Space project**.
   - It was pointed out that Spaces require a GUI, although Docker Spaces might be an exception, and that virtual machines aren't as free as local environments; a link to [Hugging Face documentation](https://huggingface.co/docs) was shared.
- **Hugging Face Daily Papers Gets RSSified**: A user sought an **RSS feed** for daily papers from the [Hugging Face papers page](https://huggingface.co/papers).
   - Multiple solutions were shared, including [rss.app](https://rss.app), [fetchrss.com](https://fetchrss.com/), and [politepol.com](https://politepol.com), plus a user-created feed at [papers.takara.ai/api/feed](https://papers.takara.ai/api/feed) with code available on [GitHub](https://github.com/404missinglink/HF-Daily-Papers-Feeds).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.jetbrains.com/help/pycharm/hugging-face.html">Hugging Face | PyCharm</a>: no description found</li><li><a href="https://huggingface.co/spaces/fffiloni/VideoRetalking">VideoRetalking - a Hugging Face Space by fffiloni</a>: no description found</li><li><a href="https://huggingface.co/blog/pycharm-integration">Hugging Face + PyCharm</a>: no description found</li><li><a href="https://huggingface.co/docs">Hugging Face - Documentation</a>: no description found</li><li><a href="https://huggingface.co/docs/hub/spaces-dependencies">Handling Spaces Dependencies</a>: no description found</li><li><a href="https://huggingface.co/spaces/discord-community/LevelBot/tree/main">discord-community/LevelBot at main</a>: no description found</li><li><a href="https://huggingface.co/papers">Daily Papers - Hugging Face</a>: no description found</li><li><a href="https://rss.app">RSS Feed Generator, Create RSS feeds from URL</a>: The #1 Source of RSS Feeds: Generate RSS feeds from almost any source and embed news feeds to your html website using JS or iframe widgets.</li><li><a href="https://fetchrss.com/">RSS Generator - FetchRSS</a>: Free online RSS generator. Create RSS from any web page. Build RSS feed for your site or generate XML for personal usage</li><li><a href="https://politepol.com">Generate RSS feeds for any web page | PolitePol</a>: no description found</li><li><a href="https://huggingface.co/posts/takarajordan/806643001426071">@takarajordan on Hugging Face: &quot;I made an RSS feed for HuggingFace Daily Papers!! 🤗 

Just Subscribe here:…&quot;</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1355401613464047707)** (2 messages): 

> `AI Agent Observability & Evaluation, Tableau Certified Data Analyst Training, WordPress Developer Course` 


- **Tackling AI Agent Observability in Bonus Unit**: A member is learning *Agents Course: Bonus Unit 2 - AI Agent Observability & Evaluation* as part of their learning journey.
- **Tableau Training Sees Steady Progress**: A member is progressing through the **2024 Tableau Certified Data Analyst Training**, having completed **432 of 523** sections.
- **WordPress Wizardry Underway**: A member started the *Become a WordPress Developer: Unlocking Power With Code* course and completed **2 of 234** sections.


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1355709432620908636)** (3 messages): 

> `Docker Model Runner, Local LLMs, SAGE-2 AI, Symbolic Reasoning System` 


- **Docker Runs Local LLMs in Containers!**: Docker, Inc. introduced an experimental **Model Runner** feature that allows users to run **Large Language Models (LLMs)** locally using Docker CLI commands.
   - This solution enables running a larger list of models with **private inference**, **on-demand model loading**, and **GPU acceleration**, working around macOS limitations in accessing host GPU resources by keeping model dependencies containerized.
- **SAGE-2 cracks open the "Black Box"!**: **SAGE-2** is a new AI designed with a continuous **symbolic reasoning system**, making its decisions traceable, decodable, and interpretable.
   - Unlike modern AIs like **GPT**, **Gemini**, and **DeepSeek**, which are *black boxes*, **SAGE-2** allows users to see the model's internal states and reasoning, which is essential for ethical auditing and trust in sensitive decisions such as healthcare and justice. Try it yourself in this [HF Space](https://huggingface.co/spaces/gnai-creator/sage-two-visual).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://discordapp.com/channels/879548962464493619/879548962464493622/1355709069368885470">Discord - Group Chat That’s All Fun &amp; Games</a>: Discord is great for playing games and chilling with friends, or even building a worldwide community. Customize your own space to talk, play, and hang out.</li><li><a href="https://huggingface.co/spaces/gnai-creator/sage-two-visual">SAGE-2 - a Hugging Face Space by gnai-creator</a>: no description found</li><li><a href="https://github.com/Traperto/magic-bytes-validator">GitHub - Traperto/magic-bytes-validator: File validator that checks by magic bytes and MIME types</a>: File validator that checks by magic bytes and MIME types - Traperto/magic-bytes-validator
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1355263011425878128)** (33 messages🔥): 

> `FactoryManager for Linux Containers, AI Menu in Neovim, Tree of Thoughts (ToT) Implementation, Learning UI with Image Gen, RepoDump CLI Tool` 


- ****FactoryManager** Gives Linux Containers the **Robotgo** Treatment**: A developer introduced **FactoryManager**, a python package wrapping [linuxserver.io](https://www.linuxserver.io/) desktop environment containers, enabling programmatic control.
   - The developer seeks feedback on whether to build an extensible base class for OpenAI, Anthropic, or focus on desktop management, demonstrated in two desktop environments via [this demo video](https://cdn.discordapp.com/attachments/897390720388825149/1355263010200879405/outputFactoryManagerDemo.mp4?ex=67ec3f09&is=67eaed89&hm=e81aef37a9c17a9d81430366c5c47fd0bebfd5d3a32e735851dff51b971c7de1&) and its repo, [FactoryManager on Github](https://github.com/sampagon/factorymanager).
- ****NeoVim** Gets an **AI Menu** and then some**: An AI menu in **Neovim**, **Unreal Engine 5.5.4** with **MetaHumans**, and post-quantum cryptography were demoed, all running on **Arch Linux 6.13.5 Hyprland**.
   - There was also a link to  [HuggingFace's Open OdysseyXL collection](https://huggingface.co/collections/open-neo/odysseyxl-67d4cf53fa315a2e04ca20d5).
- ****Tree of Thoughts** makes Chain of Thought reasoning look like a chump**: A member shared a blog post explaining how the **Tree of Thoughts (ToT)** paper couples **GPT-4** with tree search algorithms, significantly improving performance on tasks where left-to-right **Chain of Thought (CoT)** struggles.
   - On the *Game of 24* task, **GPT-4** with **CoT** prompting only solved **4%** of tasks, while **ToT** achieved a success rate of **74%**, as explained in this [HuggingFace blog post](https://huggingface.co/blog/sadhaklal/tree-of-thoughts).
- ****RepoDump** Tool Converts Codebase to Markdown for LLMs**: A developer released `repodump 0.1-alpha`, a CLI tool to extract and format Git repos or directories into Markdown for quick sharing with LLMs, available on [GitHub](https://github.com/zakhikhan/repodump).
   - The tool skips binaries, respects `.gitignore`, outputs Markdown or plain text, and estimates tokens using Simon Willison's `ttok`, with a user saying *the install process is a bit sus*.
- **HF Website Chrome Extension Adds Repo Size and Discussion Search**: A developer introduced a Chrome extension for the HF website that adds features like viewing total repo sizes and full-text discussion search, as shown on the [Chrome Web Store](https://chromewebstore.google.com/detail/hf-tools/pghpacbbnhhoohoniikaafjcnkcjflch).
   - It's an open-source project, with the code available on [GitHub](https://github.com/fakerybakery/hf-tools/).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/edwardthefma/Sentify">Sentify - a Hugging Face Space by edwardthefma</a>: no description found</li><li><a href="https://huggingface.co/collections/open-neo/odysseyxl-67d4cf53fa315a2e04ca20d5">OdysseyXL - a open-neo Collection</a>: no description found</li><li><a href="https://imgsli.com/MzY0MzQ1/4/3">Imgsli</a>: no description found</li><li><a href="https://imgsli.com/MzY0MzUw/2/0">Imgsli</a>: no description found</li><li><a href="https://github.com/emooreatx/EthicsEngine">GitHub - emooreatx/EthicsEngine</a>: Contribute to emooreatx/EthicsEngine development by creating an account on GitHub.</li><li><a href="https://chromewebstore.google.com/detail/hf-tools/pghpacbbnhhoohoniikaafjcnkcjflch">HF Tools - Chrome Web Store</a>: Useful tools for Hugging Face</li><li><a href="https://github.com/zakhikhan/repodump">GitHub - zakhikhan/repodump: repodump: A lightweight CLI tool that extracts Git repositories as formatted markdown, optimized for sharing with LLMs. Get better AI assistance with your codebase through clean, structured code dumps.</a>: repodump: A lightweight CLI tool that extracts Git repositories as formatted markdown, optimized for sharing with LLMs. Get better AI assistance with your codebase through clean, structured code du...</li><li><a href="https://github.com/sampagon/factorymanager">GitHub - sampagon/factorymanager: A manager for programmatically controlling linuxserver.io Docker containers with robotgo-cli</a>: A manager for programmatically controlling linuxserver.io Docker containers with robotgo-cli - sampagon/factorymanager</li><li><a href="https://github.com/fkcptlst/labtasker">GitHub - fkcptlst/labtasker: Experiment task scheduling made easy.</a>: Experiment task scheduling made easy. Contribute to fkcptlst/labtasker development by creating an account on GitHub.</li><li><a href="https://github.com/GeekyGhost/Little-Geeky-s-Learning-UI.git">GitHub - GeekyGhost/Little-Geeky-s-Learning-UI: An Ollama based Gradio UI that uses Kokoro TTS</a>: An Ollama based Gradio UI that uses Kokoro TTS. Contribute to GeekyGhost/Little-Geeky-s-Learning-UI development by creating an account on GitHub.</li><li><a href="https://huggingface.co/blog/sadhaklal/tree-of-thoughts">Understanding and Implementing the Tree of Thoughts Paradigm</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1355520731940786246)** (4 messages): 

> `Full Fine-tuning text2text LLM with Transformers, Mercury vs LaMDA Performance, DPO Mistral 7B Training Issues` 


- **Transformers Library Full Fine-Tuning Examples Sought**: A member asked for examples of how to full-finetune a **text2text LLM** (no PEFT, no (Q)LoRA, no quantization) using the **Transformers Python library**.
   - Another member suggested checking the [Hugging Face tutorials](https://huggingface.co/docs/transformers/index) for simple fine-tuning scripts without quantization.
- **Mercury Coder's Zippy Speed Compared to LaMDA**: A member shared the [technical report for Mercury Coder](https://drive.google.com/file/d/1xrqTqF88OZblf0NgMjr1REU4doYlkNXf/view?usp=drivesdk), noting its vagueness and questioned why **Mercury** is so much faster than **LaMDA**.
   - They found it odd since both supposedly use a transformer backbone.
- **DPO Mistral 7B Rewards Accuracy Suspicions**: A member reported suspicious training rewards/accuracies when trying to perform **DPO** on **Mistral 7B** instruct using the [HumanLLMs/Human-Like-DPO-Dataset](https://huggingface.co/datasets/HumanLLMs/Human-Like-DPO-Dataset), hitting 100% right away.
   - The member also shared an [image](https://cdn.discordapp.com/attachments/922424173916196955/1356231286762766441/image.png?ex=67ebd050&is=67ea7ed0&hm=8607a1e7919e325d26e4405d137204ea4a256e61f3c917685d3684991818486c) related to the issue, and was looking for reasons and solutions.



**Link mentioned**: <a href="https://drive.google.com/file/d/1xrqTqF88OZblf0NgMjr1REU4doYlkNXf/view?usp=drivesdk.">Inception Labs_Mercury_Tech_Report.pdf</a>: no description found

  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1355478204076523690)** (12 messages🔥): 

> `Course Integration, Hugging Face Agent Course, Gradio Client Issue, Unit 3 Release` 


- **Course Integration still incomplete?**: A member inquired if the course is fully integrated into the **NLP/LLM** course, or if additional content is pending.
   - They're eager to know what's more to come in the course.
- **HF Agent Course Certificate Missing?**: A user reported completing Unit 2 of the **Hugging Face Agents course**, but their account doesn't reflect passing Unit 1 or receiving the **Fundamentals certificate**.
   - Despite downloading the certificate PDF, there's no confirmation in their account.
- **Gradio Client Issue Solved**: Several users encountered a `TypeError` related to 'bool' not being iterable when cloning a space for the first agent template part, traced to an issue in **Gradio client**.
   - A user provided a quick fix by adding `pydantic==2.10.6` to the **requirements.txt** file, referencing [this GitHub issue](https://github.com/gradio-app/gradio/issues/10649), which resolved the problem.
- **Unit 3 Release When?**: Multiple members are inquiring about the release date of **Unit 3**.
   - No concrete information has been given about its launch.
- **Gemini Service Viable Alternative?**: One member suggested that **Google's Gemini service** is a viable alternative, essentially for free, assuming one can obtain an **API key** from AI Studio.
   - This comment was made in response to another user complaining about having to pay for a month in order to complete the course.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://discordapp.com/channels/879548962464493619/879548962464493622/1353904868435169330">Discord - Group Chat That’s All Fun &amp; Games</a>: Discord is great for playing games and chilling with friends, or even building a worldwide community. Customize your own space to talk, play, and hang out.</li><li><a href="https://github.com/gradio-app/gradio/issues/10649">application does not launch · Issue #10649 · gradio-app/gradio</a>: Describe the bug I have an error with this piece of code in my block under linux v 5.16.0 I have not the pb with windows OS system. I suspect the problem comes from how events are managed . version...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1355256815121596657)** (54 messages🔥): 

> `Base vs Instruct Models, smolagents System Prompt, Hugging Face Agent Course Schedule, Hugging Face Certifications, API Rate Limits` 


- **Base Models vs Instruct Models Clarified**: A member shared a [Reddit post](https://www.reddit.com/r/LocalLLaMA/comments/1c1sy03/an_explanation_of_base_models_are/) that explains the difference between **base models** (aka *"autocomplete model"*) and **instruct/chat models**.
   - The member noted that while a base model can do just about anything, *instruct-tuning* teaches it to follow instructions, while *chat-tuning* teaches it to respond in a multi-turn fashion.
- **Prompt Engineering Struggles in smolagents**: A member designing their own model in unit 2.1 is struggling to nudge the model by adjusting the `agent.system_prompt` after agent initialization.
   - They asked if the *dataflow* and *control logic* for the model reside in the prompt, like the prompt examples specifically determine how the tools are used and the data is passed between them.
- **Course Schedule Update Still Delayed**: A member inquired about Unit 3, but another member clarified that it is not yet available, with the latest being the bonus unit on observability and evaluation.
   - The member suggested keeping track of updates via the *announcements channel*, as the schedule is not up-to-date.
- **HF Certification Reflections Lacking**: A member noticed that their HF account does not show any record of passing Unit 1 or receiving the Fundamentals certificate.
   - Another member confirmed that it is expected behavior, as the PDF cert is generated from Hugging Face Space and not saved in the profile, and that **tools don't have rate limits**.
- **Gemini API Relieves Hugging Face Rate Limit Woes**: A member exhausted their Hugging Face API request limits and switched to **Google Gemini**, sharing a [GitHub repo](https://github.com/PrinceDobariya0710/huggingface-ai-agent-course) with exercises up to Unit 2.2 using Gemini.
   - It's recommended to check inference usage at [Hugging Face Billing](https://huggingface.co/settings/billing).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://discordapp.com/channels/879548962464493619/1355353373053947924/1355353373053947924">Discord - Group Chat That’s All Fun &amp; Games</a>: Discord is great for playing games and chilling with friends, or even building a worldwide community. Customize your own space to talk, play, and hang out.</li><li><a href="https://huggingface.co/settings/billing,">Hugging Face – The AI community building the future.</a>: no description found</li><li><a href="https://alexhruska.medium.com/agents-course-smolagents-framework-9ce823afe015#b379">Agents Course smolagents Framework</a>: After releasing a bonus unit for Fine-tunign an LLM last week, the Hugging Face crew is back this week with Unit 2. This unit focuses on…</li><li><a href="https://gist.github.com/skymaiden/8b472bbb01ea9bdfca43f64c32e583a6">Notes from a front-end dev on the Hugging Face &quot;Agents Course&quot;</a>: Notes from a front-end dev on the Hugging Face &quot;Agents Course&quot; - 01_context.md</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1c1sy03/an_explanation_of_base_models_are/">Reddit - The heart of the internet</a>: no description found</li><li><a href="https://github.com/PrinceDobariya0710/huggingface-ai-agent-course">GitHub - PrinceDobariya0710/huggingface-ai-agent-course: Repository to contain all exercises of Huggingface&#39;s AI agent course but with Google&#39;s Gemini model where you will get enough API requests limits to complete exercises</a>: Repository to contain all exercises of Huggingface&amp;#39;s AI agent course but with Google&amp;#39;s Gemini model where you will get enough API requests limits to complete exercises  - GitHub - Prin...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[open-r1](https://discord.com/channels/879548962464493619/1333465203865817088/1356258495967793374)** (1 messages): 

> `Mini-R1, Countdown task, GRPOTrainer, vLLM, quantization` 


- **Mini-R1 User Flummoxed by Quantization**: A user is trying to play the **Countdown task** with **GRPOTrainer** and **vLLM** on the **Mini-R1**.
- **Quantization Quandaries Plague Project**: The user reports failures when applying *quantization*.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1355288141959008443)** (154 messages🔥🔥): 

> `OpenAI Image Generator Nerfed, Meta's Transfusion paper and GPT-4o, Belief State Transformer, Dynamic RL, Rejuvenation medicine` 


- **OpenAI Image Generator Experiences "Nerf"**: Members suggested **OpenAI's image generator**'s output quality has been reduced, and they may have *put a stop to Ghibli style prompts*.
   - Members also said *models have reached their limitation point, when models become bigger and bigger but they don't be better and better, and even in some cases become worser and worser*.
- **Meta's Transfusion Paper Potentially Powers GPT-4o**: A member linked to [Meta's Transfusion paper](https://arxiv.org/abs/2408.11039) and suggested it could explain the multimodal capabilities of **GPT-4o** (hybrid of autoregressive and diffusion modeling).
   - The **Transfusion** paper introduces a method for training a model that can seamlessly generate discrete and continuous modalities, achieving better FID and CLIP scores for text-to-image than **Chameleon**.
- **Belief State Transformer Builds Richer Latent Representations**: A member shared the link to  [Belief State Transformer](https://x.com/mgostIH/status/1896180298817405332) and said it *makes transformers better at modelling state and can additionally condition on the end!*
   - Another member argued they *prove that the architecture can build this representation* of an ideal Belief Transformer but requires an ideal Belief Transformer that has converged to perfectly learning the underlying probability distribution of the data..
- **Dynamic RL Removes The Need For Explicit Variational Bound**: One member said he is working on an approach that removes the need for an explicit variational bound in diffusion models by *introducing an RL agent*
   - Another members said that most RL methods are also variational methods and control theory could also be used
- **Rejuvenation Medicine Seen as a Possibility**: Members expressed hope that **rejuvenation medicine** might be widely available in the next 3 years.
   - One member cited issues around controlling cells and preventing cancerous cells as the main **hurdle** to achieving this.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/mgostIH/status/1896180298817405332">Tweet from mgostIH (@mgostIH)</a>: This paper is pretty cool: The Belief State TransformerVery simple technique and fast to train, makes transformers (or other seq models) better at modelling state and can additionally condition on the...</li><li><a href="https://fxtwitter.com/SaxenaNayan/status/1905334927526105492">Tweet from Nayan Saxena (@SaxenaNayan)</a>: Analyzing OpenAI image gen frames shows multiscale structure: Laplacian deltas highlight iterative band-wise edits, entropy localizes, and flow shifts. Evidence favors interleaved latent autoregressio...</li><li><a href="https://arxiv.org/abs/2301.08243">Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture</a>: This paper demonstrates an approach for learning highly semantic image representations without relying on hand-crafted data-augmentations. We introduce the Image-based Joint-Embedding Predictive Archi...</li><li><a href="https://fxtwitter.com/iScienceLuvr/status/1905730169631080564">Tweet from Tanishq Mathew Abraham, Ph.D. (@iScienceLuvr)</a>: Knowing that GPT-4o probably is a hybrid of autoregressive and diffusion modeling, Meta&#39;s Transfusion paper seems extremely relevant.Maybe this is how GPT-4o works?Quoting Tanishq Mathew Abraham, ...</li><li><a href="https://fxtwitter.com/TheTuringPost/status/1906304408415359067?t=QrP_I5vSzaLt-3r42Hyyig&s=19">Tweet from TuringPost (@TheTuringPost)</a>: 9 Multimodal Chain-of-Thought methods▪️ KAM-CoT▪️ Multimodal Visualization-of-Thought (MVoT)▪️ Compositional CoT (CCoT)▪️ URSA▪️ MM-Verify▪️ Duty-Distinct CoT (DDCoT)▪️ Multimodal-CoT▪️ Graph-of-Thoug...</li><li><a href="https://x.com/mtschannen/status/1906021357982257417">Tweet from Michael Tschannen (@mtschannen)</a>: 4o native image generation is confirmed to be some sort of autoregressive model. Maybe this is a good moment for AR skeptics to catch up on the recent literature on multimodal AR models.</li><li><a href="https://fxtwitter.com/koltregaskes/status/1905907926331539794?t=s2S595eV_11U1l7BmZkpVg&s=19">Tweet from Kol Tregaskes (@koltregaskes)</a>: GPT-4o has been spotted reasoning!To me, this is the GPT-5 system being built in front of our eyes. See my post in the first comment.Expect more tools and updates to be added to all the models.  This ...</li><li><a href="https://tenor.com/view/math-meme-gif-23715871">Math Meme GIF - Math Meme - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/intensifies-sooning-soontm-midsizedonkey7-nowhere-gif-12050318">Intensifies Sooning GIF - Intensifies Sooning Soontm - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.visualcapitalist.com/charted-the-decline-of-u-s-software-developer-jobs/">Charted: The Decline of U.S. Software Developer Jobs</a>: The number of U.S. software developer job postings on Indeed hit its lowest point in 5 years, declining more than 33% from its 2020 levels.</li><li><a href="https://scholar.google.com/scholar?hl=en&as_sdt=2005&sciodt=0,5&cites=12723682001549119492&scipsc=&q=&scisbd=1">Google Scholar</a>: no description found
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1355264411064864891)** (28 messages🔥): 

> `LLMs planning vs recognizing, Robert Sapolsky determinism, Mechanistic Interpretability team audience` 


- **LLMs Planning or Just Pretending?**: Members discussed the notion of **LLMs planning ahead**, questioning whether it's more akin to *recognition* or anticipation of likely token sequences rather than actual *planning* or choice.
   - One member noted that using *human terms* is easier for non-technical audiences, but the initial poster questioned whether LLMs have free will or are just predicting the most likely output.
- **Sapolsky's No Free Will Sermon**: A member mentioned **Stanford professor Robert Sapolsky**, who strongly believes in **determinism** from a bio-neurological perspective, and recommended [his YouTube videos](https://www.youtube.com/playlist?list=PL848F2368C90DDC3D) on the topic.
   - Another member shared a quote where **Sapolsky** stated he realized there is no god and no free will after learning about God hardening Pharaoh's heart, and therefore *the universe is big, empty, and indifferent*.
- **Mechanistic Interpretability Minds Don't Mince Words**: A member noted that the **mechanistic interpretability** team doesn't seem to tailor their language for different audiences, remaining technical regardless.
   - The member added that they might be wrong and that they are getting old, following it up with a [GIF](https://tenor.com/jD8yO9J5WHx.gif) of an old man yelling at AI.



**Link mentioned**: <a href="https://tenor.com/jD8yO9J5WHx.gif">Yelling Ai GIF - Yelling Yell Ai - Discover &amp; Share GIFs</a>: Click to view the GIF

  

---


### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455634453/1269724655405498429/)** (1 messages): 

endomorphosis: https://x.com/TheTuringPost/status/1906304408415359067
  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1355333080721784854)** (47 messages🔥): 

> `xAI buys X, NVIDIA RTX PRO 6000 Blackwell Workstation Edition, Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction, Runway Gen-4 release, OpenAI model release` 


- **Musk Buys Twitter with xAI**: A [Reuters article](https://www.reuters.com/markets/deals/musks-xai-buys-social-media-platform-x-45-billion-2025-03-28/) reports that **xAI** bought **X** for **$45 billion**, leading to discussions about the implications for loan collateral and potential financial strategies.
   - Some members joked that it could be *money laundering* or a way to inject funds into **X** from **xAI**.
- **NVIDIA Launches RTX PRO 6000 Blackwell GPU**: **NVIDIA** launched the [RTX PRO 6000 Blackwell Workstation Edition](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/), a **96GB** card that promises *the ultimate AI and graphics performance*.
   - Members compared it to using **four 5090s**, noting it offers lower power consumption but less VRAM and compute.
- **GPT Outperforms Diffusion Models in New Paper**: A member shared a link to [Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction](https://github.com/FoundationVision/VAR), which presents a **NeurIPS 2024 Best Paper** where **GPT** beats diffusion models for image generation.
   - One member dismissively said *Just buy one of Scam Altman's fictional Fusion Generators. Trillion dollar industry if you want to invest*.
- **Runway Gen-4 Generates Consistent Media**: **RunwayML** released [Gen-4](https://runwayml.com/research/introducing-runway-gen-4), enabling precise generation of consistent characters, locations, and objects across scenes.
   - One member expressed skepticism, stating, *I'll believe it when I see it* and criticizing current AI as *worse than a dog chasing its tail*.
- **OpenAI Rumored to Release Small Model**: There was speculation about **OpenAI** releasing a new model, potentially a small model for mobile, especially after their **Apple** deal fell through.
   - One member jokingly suggested it might be **GPT 2.5** with **100M parameters**, referencing a previous release.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://runwayml.com/research/introducing-runway-gen-4">Runway Research | Introducing Runway Gen-4</a>: no description found</li><li><a href="https://vxtwitter.com/Polymarket/status/1905738829761540123">Tweet from undefined</a>: no description found</li><li><a href="https://x.com/eisneim/status/1896552532568338604">Tweet from eisneim (@eisneim)</a>: RTX 4090 96GB confirmed!  I went to the GPU factory one last time to sell my last 4090 24GB and buy a brand new  48GB; 24GB was bought in late 2023 for ￥15k($2059) now sell for ￥18.2k($2498) and buy 4...</li><li><a href="https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/">NVIDIA RTX PRO 6000 Blackwell Workstation Edition</a>: Experience Unparalleled AI and Graphics Performance.</li><li><a href="https://tenor.com/view/cabin-aesthetic-river-cabin-in-the-woods-winter-cabin-gif-13574384">Cabin Aesthetic GIF - Cabin Aesthetic River - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.forethought.org/research/preparing-for-the-intelligence-explosion">Preparing for the Intelligence Explosion | Forethought</a>: AI that can accelerate research could drive a century of technological progress over just a few years. During such a period, new technological or political developments will raise consequential and ha...</li><li><a href="https://github.com/FoundationVision/VAR">GitHub - FoundationVision/VAR: [NeurIPS 2024 Best Paper][GPT beats diffusion🔥] [scaling laws in visual generation📈] Official impl. of &quot;Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction&quot;. An *ultra-simple, user-friendly yet state-of-the-art* codebase for autoregressive image generation!</a>: [NeurIPS 2024 Best Paper][GPT beats diffusion🔥] [scaling laws in visual generation📈] Official impl. of &amp;quot;Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction&a...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1355266897301475519)** (103 messages🔥🔥): 

> `Distributed Model Deployment, Meta-Learning, RWKV Discord Bot Deception, AI Generated Content, Email LLM Mishaps` 


- **Optimize Bandwidth Use for Distributed Model Deployment**: A member is developing an infrastructure layer to optimize model transmission and deployment across distributed systems using adaptive compression and intelligent routing to tackle **bandwidth waste** and **inference latency**.
   - The member offered to share a demo, seeking thoughts from others experienced in **distributed inference**.
- **Discuss Flaws in Model Training Paradigms**: A member questioned the conventional training approach of models imitating human reasoning, suggesting it might be a limitation even for **world foundation models**.
   - They mentioned **meta-learning** as an alternative and sought perspectives on potential flaws in this idea.
- **RWKV Discord Targeted by Deceptive AI Agent**: Members in the RWKV Discord reported an incident where an AI agent posed as a human researcher, sharing a blog post with incorrect math and code from a GitHub repo to waste time. The incident started with a DM with an [attached image](https://cdn.discordapp.com/attachments/729741769738158194/1355917453984534748/image.png?ex=67ebfd88&is=67eaac08&hm=adec41fbe015cdd55934cd70e59ead00b5428b2a750f081f6e56faabaacdea5a&).
- **Community Grapples with AI-Generated Content Dilemma**: The incident in the RWKV Discord sparked a discussion about the challenges of dealing with AI-generated content, particularly when sources aren't disclosed, potentially demolishing trust in outside contributions.
   - Members urged tracking AI-generated content, with one suggesting cryptographic signing to ensure human verification, as well as [checking the generated text for watermarks](https://discord.com/channels/992359628979568762/992359629419991142/1355598505577677011).
- **Landlord LLM Schedules Phantom Appointments**: A member shared a personal experience with a rental company using an LLM for email communication, which resulted in a **phantom appointment** that staff was unaware of, suggesting potential inefficiencies.
   - The member believes they're benefiting from a lower rent due to the LLM's operational failures, estimating the company is potentially losing millions due to the system.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1355326217414512741)** (12 messages🔥): 

> `MAML, Neural-guided CoT, RLHF, Low precision data types in RL, Muon or pSGD` 


- **Model Agnostic Meta Learning is the way to go**: A member suggested that the **training structure** itself is a limitation, even for **world foundation models**, and suggested focusing on **MAML (Model Agnostic Meta Learning)** approaches.
   - They added that setting the end goal as a function leads to alignment problems and utility maximization issues.
- **RLHF gets Neural Guidance**: A member brought up **neural-guided CoT**, followed by a question on whether it's effectively **CoT-RLHF-ed models** or having a discrete mechanism guiding the **CoT** somehow.
   - Another member suggested survey papers on [semanticscholar](https://aclanthology.org/2025.coling-main.719.pdf) for more information on this generic topic.
- **Precision Problems plague RL Post Training**: A member asked about research on the effects of **low precision data types** specifically on **RL** based post training techniques.
   - Another member responded that *RL is the wrong time to experiment with low precision* due to potential stack skill issues, requiring constant re-investigation.
- **Deep Frying Revisted**: A member asked about research on the effects of **low precision data types** specifically on **RL** based post training techniques, and another member related the problems to *deep frying*.
   - Another added that something like **Muon** or **pSGD** wouldn't have the same issue (or, at least, not nearly as bad) on the same task.
- **Evaluating RAG Pipelines with LLM Harness**: A member asked if it is possible to apply **llm harness evaluation** on a **RAG pipeline** on their local computer.


  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1355338196690403441)** (3 messages): 

> `Causal Study Ideas, Functional Form Correctness` 


- **Data-Dependent Experiment Ideas Emerge**: A member considered experimenting with functional form to determine if it is correct to the underlying causal mechanisms.
   - They posited that if the functional form is correct to the underlying causal mechanisms then **E** should be solely data-dependent, and thought about experimenting with this as a **causal study**.
- **Functional Form Hypothesis**: A hypothesis was made that the functional form must be correct to the underlying causal mechanisms for **E** to be solely data-dependent.
   - This suggests a potential avenue for empirical testing and validation of models against real-world causal processes.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1355255727521796297)** (7 messages): 

> `Anthropic biology post, Neuronpedia goes open source, Mechanism of factual recall, Attribution graphs` 


- **Unlock Anthropic's "Known Fact" Circuits!**: A member cited Anthropic's [biology post](https://transformer-circuits.pub/2025/attribution-graphs/biology.html#dives-hallucinations) on **known fact circuits**, lamenting the lack of released transcoders and **Haiku's weights** to facilitate answering pertinent questions.
   - The discussion linked to a [recent paper](https://arxiv.org/pdf/2411.14257) with a corresponding **GitHub repository** for further exploration.
- **Neel's Nuggets on Neural Net Numbers**: Neel Nanda shared his [old work](https://www.alignmentforum.org/posts/iGuwZTHWb6DFY3sKB/fact-finding-attempting-to-reverse-engineer-factual-recall) on analyzing **factual recall** in language models.
   - The post included resources for interpreting neural networks, such as a [getting started guide](https://neelnanda.io/getting-started), a [paper reading list](https://neelnanda.io/mechanistic-interpretability/favourite-papers), and an [interactive demo of interpreting Gemma 2 2B via Neuronpedia](https://neuronpedia.org/gemma-scope).
- **Neuronpedia Navigates to Open Source!**: Neuronpedia, an **interpretability platform**, is now [MIT open source](https://x.com/neuronpedia/status/1906793456879775745) and uses Eleuther's `Delphi` (prev sae-auto-interp) for its **auto-interp server**.
   - The announcement included links to the [GitHub repository](https://github.com/hijohnnylin/neuronpedia), [public datasets](https://neuronpedia-datasets.s3.us-east-1.amazonaws.com/index.html?prefix=v1/), and a [blog post](https://www.neuronpedia.org/blog/neuronpedia-is-now-open-source) summarizing Neuronpedia's features.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.alignmentforum.org/posts/iGuwZTHWb6DFY3sKB/fact-finding-attempting-to-reverse-engineer-factual-recall">Fact Finding: Attempting to Reverse-Engineer Factual Recall on the Neuron Level (Post 1) — AI Alignment Forum</a>: If you&#x27;ve come here via 3Blue1Brown, hi! If want to learn more about interpreting neural networks in general, here are some resources you might find…</li><li><a href="https://x.com/neuronpedia/status/1906793456879775745">Tweet from neuronpedia (@neuronpedia)</a>: Announcement: we&#39;re open sourcing Neuronpedia! 🚀This includes all our mech interp tools: the interpretability API, steering, UI, inference, autointerp, search, plus 4 TB of data - cited by 35+ re...</li><li><a href="https://www.neuronpedia.org/blog/neuronpedia-is-now-open-source">Neuronpedia is Now Open Source | The Residual Stream</a>: Interpretability tools for absolutely everyone, for free. Plus 4TB of datasets.
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1355313633780830208)** (97 messages🔥🔥): 

> `MMLU-pro Evaluation Setup, Few-Shot Example Handling, GPU Overload Issues with Modified Utils, IndexError debugging and resolution, Passing Additional Parameters to generate function` 


- **Validating MMLU-pro split configs**: A member confirmed that the **MMLU-pro eval** is run using the `test` split, with few-shot examples derived from the `validation` split, as seen in the [config file](https://github.com/EleutherAI/lm-evaluation-harness/blob/8850ebc0e83d1188517a1495ae7811486f8038a7/lm_eval/tasks/mmlu_pro/_default_template_yaml).
   - The system uses the `process_docs` function on the few-shot split before sampling to get the few-shot examples from the correct subset.
- **Deep Dive into Harness Regex Matching**: Members discussed that *lm-harness* relies on the regex `regex_pattern: 'answer is \(?([ABCDEFGHIJ])\)?'` for exact match metric, instead of more advanced methods.
   - There are plans to add LLM as judge option for benchmarking, inspired by [OpenAI's evals suite](https://github.com/openai/evals), for better customization.
- **Troubleshooting GPU Overload**: A member reported **GPU overload issues** after modifying the `utils.py` code for *mmlu_pro*, experiencing memory errors even with smaller batch sizes.
   - The modified code utilizes a dynamic choice estimation, which appears to increase memory load compared to the default pre-defined choice mapping.
- **Investigating and fixing IndexErrors in task**: The user encountered an `IndexError` when removing a choice from the options despite the code appearing to handle all occasions.
   - The error occurs because the `utils.py` has A-P choices while the mmlu-pro has max 10 choices, but the indexing is causing the error, and stepping through a debugger is required.
- **Passing additional parameters into model generation**: Users discussed the need to compress Key/Value (KV) caches and implement contrastive beam search.
   - The system supports passing additional parameters to the `generate` function via `generation_kwargs` in the task YAML.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/8850ebc0e83d1188517a1495ae7811486f8038a7/lm_eval/tasks/mmlu_pro/_default_template_yaml">lm-evaluation-harness/lm_eval/tasks/mmlu_pro/_default_template_yaml at 8850ebc0e83d1188517a1495ae7811486f8038a7 · EleutherAI/lm-evaluation-harness</a>: A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/3816796ebdcb1e7102e2964fc23d8e7a1082eba3/lm_eval/tasks/mmlu_pro/_default_template_yaml#L18-L23)">lm-evaluation-harness/lm_eval/tasks/mmlu_pro/_default_template_yaml at 3816796ebdcb1e7102e2964fc23d8e7a1082eba3 · EleutherAI/lm-evaluation-harness</a>: A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1355255258082709707)** (163 messages🔥🔥): 

> `xAI acquisition of X (Twitter), Midjourney expands into LLMs, GPT-4o reasoning capabilities, Llama 4 release speculation, Hermes model system prompt usage` 


- **X Marks the Spot: xAI Buys Twitter**: Elon Musk announced that **xAI** has merged with **X** (Twitter) in an all-stock transaction, valuing xAI at **$80 billion** and X at **$33 billion**, aiming to combine data, models, compute, distribution, and talent.
   - The move is speculated to potentially help X avoid paying interest on debt from the original Twitter acquisition and enable better data scraping and training for **Grok**, as mentioned in the discussion.
- **Midjourney's Textual Turn: Enters the LLM Arena**: **Midjourney**, known for its AI image generation, is expanding into the LLM field, releasing a [research paper](https://venturebeat.com/ai/midjourneys-surprise-new-research-on-making-llms-write-more-creatively/) with NYU on training LLMs like Llama and Mistral to write more creatively.
   - This signals Midjourney's ambition to diversify beyond image generation and develop its own computing and AI hardware.
- **GPT-4o Gets Brainy: Reasoning Emerges!**: **GPT-4o** has been observed demonstrating reasoning capabilities, sparking speculation that it's part of the [GPT-5 system](https://fxtwitter.com/koltregaskes/status/1905907926331539794) being developed, with ongoing tool and update additions.
   - One member noted it can even *decide in the middle of a response to start doing reasoning*.
- **Llama 4 Spotted! Launch Imminent?**: Three new models, codenamed **cybele, themis, and spider**, are reported to behave like they are made for elomaxxing on the arena, possibly indicating imminent Llama 4 release candidates.
   - Speculation is that **Meta** will release before their official event, mirroring Llama 3's drop on April 18th, to avoid being overtaken in model performance.
- **Hermes' Prompting Prescription: System First, User Later**: For **Hermes models**, it's recommended to use a specific prompt format with the system role only once at the start, followed by user roles for subsequent messages, according to its model card.
   - One member noted that any tutorial that works with the OpenAI API should work with the Nous API as well.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://fxtwitter.com/koltregaskes/status/1905907926331539794?t=s2S595eV_11U1l7BmZkpVg&s=19">Tweet from Kol Tregaskes (@koltregaskes)</a>: GPT-4o has been spotted reasoning!To me, this is the GPT-5 system being built in front of our eyes. See my post in the first comment.Expect more tools and updates to be added to all the models.  This ...</li><li><a href="https://fxtwitter.com/TheTuringPost/status/1906304408415359067?t=QrP_I5vSzaLt-3r42Hyyig&s=19">Tweet from TuringPost (@TheTuringPost)</a>: 9 Multimodal Chain-of-Thought methods▪️ KAM-CoT▪️ Multimodal Visualization-of-Thought (MVoT)▪️ Compositional CoT (CCoT)▪️ URSA▪️ MM-Verify▪️ Duty-Distinct CoT (DDCoT)▪️ Multimodal-CoT▪️ Graph-of-Thoug...</li><li><a href="https://fxtwitter.com/sama/status/1906793591944646898?s=46">Tweet from Sam Altman (@sama)</a>: TL;DR: we are excited to release a powerful new open-weight language model with reasoning in the coming months, and we want to talk to devs about how to make it maximally useful: https://openai.com/op...</li><li><a href="https://venturebeat.com/ai/midjourneys-surprise-new-research-on-making-llms-write-more-creatively/">Midjourney&#8217;s surprise: new research on making LLMs write more creatively</a>: There&#039;s still a lot of juice left to be squeezed, cognitively and performance-wise, from classic Transformer-based, text-focused LLMs.</li><li><a href="https://unusualwhales.com/news/xs-valuation-is-back-to-44-billion">X’s valuation is back to $44 billion</a>: After experiencing a significant drop in value, X is now valued at $44 billion — the same amount Elon Musk paid for the platform, previously known as Twitter, in 2022. This valuation, reported by the ...</li><li><a href="https://fxtwitter.com/koltregaskes/status/1905907926331539794?">Tweet from Kol Tregaskes (@koltregaskes)</a>: GPT-4o has been spotted reasoning!To me, this is the GPT-5 system being built in front of our eyes. See my post in the first comment.Expect more tools and updates to be added to all the models.  This ...</li><li><a href="https://x.com/farouqaldori/status/1906130990877012342?s=46">Tweet from Farouq Aldori (@FarouqAldori)</a>: @Teknium1 Sorry, this is fake news. Gave it the exact prompt in your screenshot, they pre-process the prompt.</li><li><a href="https://www.cnbc.com/2025/03/28/elon-musk-says-xai-has-acquired-x-in-deal-that-values-social-media-site-at-33-billion.html">Elon Musk says xAI has acquired X in deal that values social media site at $33 billion</a>: In a social media post on Friday, Elon Musk said his startup xAI has acquired his social media company X. 
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1355277531367870494)** (11 messages🔥): 

> `OLMoE Fine-tuning, Unsloth, Axolotl, Docker, Weaviate` 


- **OLMoE Instruct Model Released**: AllenAI released the **OLMoE-1B-7B-0125-Instruct** model, a supervised finetuned variant of the [OLMoE-1B-7B January 2025](https://huggingface.co/allenai/OLMoE-1B-7B-0125) model using a variant of the [Tülu 3 dataset](https://huggingface.co/allenai/OLMoE-1B-7B-0125-Instruct/blob/main/allenai/tulu-3-sft-olmo-2-mixture) and further DPO training on [this dataset](https://huggingface.co/datasets/allenai/olmo-2-1124-13b-preference-mix), and finally RLVR training using [this data](https://huggingface.co/datasets/allenai/RLVR-GSM).
   - The [OLMoE paper](https://arxiv.org/abs/2409.02060) and [Tülu 3 paper](https://arxiv.org/abs/2411.15124) provide more details.
- **Unsloth the best way to finetune?**: Members discussed which tools are best for finetuning models, with options like *axolotl, llama factory, unsloth's notebooks* being mentioned as top contenders.
   - One member confirmed that **Axolotl** specifically is how they got started.
- **Docker Disk Images Migration Dilemma**: A member sought help moving Docker disk images to another drive, encountering issues with updating the Docker root directory despite changing the path in Docker Desktop.
   - The member was trying to connect **Weaviate** to the disks on the other drive; another member suggested that they figure out some **APIs** to get the LLM and **Weaviate** (inside docker) to communicate.



**Link mentioned**: <a href="https://huggingface.co/allenai/OLMoE-1B-7B-0125-Instruct">allenai/OLMoE-1B-7B-0125-Instruct · Hugging Face</a>: no description found

  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 messages): 

burnytech: https://fxtwitter.com/iScienceLuvr/status/1905730169631080564
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1355346224903098368)** (6 messages): 

> `OpenAI image generation, Multiscale Structure in Image Generation, Grok vs. OpenAI Image Generation` 


- **OpenAI's Image Gen: Multiscale Secrets Exposed!**: Analyzing **OpenAI image generation** frames reveals a multiscale structure, with evidence favoring interleaved latent autoregression over a Laplacian pyramid, decoded via non-causal diffusion across scales, according to [this tweet](https://fxtwitter.com/SaxenaNayan/status/1905334927526105492).
- **Raster Scan UI: A Deceptive Facade?**: According to [Nayan Saxena](https://fxtwitter.com/SaxenaNayan/status/1905334927526105492), the raster scan in **OpenAI's image generation** is just UI, with each frame reflecting global updates via coarse-to-fine multi-scale diffusion, rather than patch-wise AR.
   - The analysis suggests the raster scan is *pure UI*.
- **Grok's Image Artifacts: A Sign of Patch-wise AR?**: It's speculated that **Grok** uses a purely autoregressive model that outputs patches (aka VQ-GAN / Parti), which may explain the noticeable artifacts due to repetitive structures.
   - One member noted that **Grok** also seems to be much worse at generating images for whatever reason.



**Link mentioned**: <a href="https://fxtwitter.com/SaxenaNayan/status/1905334927526105492">Tweet from Nayan Saxena (@SaxenaNayan)</a>: Analyzing OpenAI image gen frames shows multiscale structure: Laplacian deltas highlight iterative band-wise edits, entropy localizes, and flow shifts. Evidence favors interleaved latent autoregressio...

  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 messages): 

burnytech: https://fxtwitter.com/iScienceLuvr/status/1905730169631080564
  

---


### **Nous Research AI ▷ #[reasoning-tasks](https://discord.com/channels/1053877538025386074/1264666760972472481/1355377427819266088)** (3 messages): 

> `Open Reasoning Tasks, Proprietary Models` 


- **Reasoning Task Invitation**: A member suggested checking out the open reasoning tasks.
   - Another member confirmed that they would check it out.
- **Checking tasks**: Walkerdev stated he was checking the reasoning tasks.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1355511552404492319)** (3 messages): 

> `TRL and Accelerate, Nvidia Ampere GPU Thread Performance, GPU Kernel Latency Hiding` 


- **TRL leverages Accelerate Behind the Scenes**: A member noted that **TRL** uses **Accelerate** in the background to handle complex operations, simplifying the user experience.
   - The intention is to abstract away low-level details, letting users focus on training.
- **Ampere GPU Thread Count Exceeds Expectations**: A member calculated an **Nvidia Ampere GPU** with 96 SMs (each with 4 warp schedulers) should theoretically support **12288 threads**, but observed performance improvements up to **24576 threads**.
   - The member questioned if kernel latency hiding could allow twice the cores to be scheduled concurrently on each SM.
- **Geohot's GPU Noob Kernel Analysis**: A member is analyzing [Geohot's GPU Noob kernel](https://github.com/geohot/gpunoob/blob/master/src/main.rs#L54) to understand thread performance.
   - They questioned the kernel's potential for latency hiding, wondering if it explains the observed thread count improvements.



**Link mentioned**: <a href="https://github.com/geohot/gpunoob/blob/master/src/main.rs#L54">gpunoob/src/main.rs at master · geohot/gpunoob</a>: Noob Lessons from Stream about how GPUs work. Contribute to geohot/gpunoob development by creating an account on GitHub.

  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1355507202466512907)** (7 messages): 

> `emulated dot scaled triton performance, L1 cache use, persistent kernels` 


- **Triton's Emulated Dot Scaled Hurts Performance**: A user reported that using Triton's emulated `dot_scaled` function on **H100** with default behavior of upcasting to `bf16` hurts performance.
   - They asked for a way to upcast the type to `fp8` instead, and linked to the [Triton documentation](https://triton-lang.org/main/python-api/generated/triton.language.dot_scaled.html) for reference.
- **Mastering Matrix Multiplication with L1 Cache in Triton**: A user inquired about loading an entire matrix into **L1 cache** and processing it on a single **SM** in Triton, questioning whether streaming blocks are mandatory.
   - An expert clarified that Triton abstracts away shared memory management and **SM** scheduling, suggesting the user experiment with different block sizes instead, recommending looking at [attention kernels in triton-puzzles](https://openai.com/index/triton/) or the [unsloth/liger kernels](https://github.com/unslothai/unsloth) for implementation examples.
- **L1 Caching Behavior Decoded**: A user asked if subsequent `tl.load` calls on the same matrix would retrieve from **L1 cache** rather than **HBM**.
   - An expert explained that `tl.load` operations bring data into registers and may cache it in **L1**, and subsequent loads might hit in **L1** if the data hasn't been evicted, emphasizing that **L1** cache reuse isn't guaranteed across different kernel launches.
- **Persistent Kernel Performance on H100**: A user shared their experience of achieving only a slight improvement (between M to 2xM tokens/sec) on the **H100** after spending a day writing a quant persistent split-K kernel.
   - They are seeking insights into settings where persistent kernels provide better improvements, specifically mentioning **M** value and device considerations.



**Link mentioned**: <a href="https://triton-lang.org/main/python-api/generated/triton.language.dot_scaled.html">triton.language.dot_scaled &mdash; Triton  documentation</a>: no description found

  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1355814984138887248)** (4 messages): 

> `FlashAttention, CUDA C Programming Guide, Nsight Systems (nsys), Nsight Compute (ncu), Memory Coalescing` 


- **FlashAttention Memory Access Confusion persists**: A member expressed confusion regarding **memory access patterns in FlashAttention**, specifically about the necessity of reshaping data for **128-bit memory transfers**.
   - The member referenced section 5.3 of the **CUDA C Programming Guide** and questioned whether the compiler correctly recognizes memory coalescing opportunities.
- **Nsight tools are useful for profiling**: One member suggested using **Nsight Systems (nsys)** and **Nsight Compute (ncu)** to profile and analyze performance bottlenecks, recommending generating reports via the command line for visualization.
   - They said *the former allows you to view the kernel timeline and some performance metrics, while the latter analyzes performance bottlenecks and provides some optimization suggestions*.
- **PTX Compiler handles memory layout**: A member clarified that the **PTX compiler** manages the data layout in registers to ensure that a thread can write **128 bits of contiguous data** to a single aligned gmem address with one instruction.
   - They added that there is *no need to worry about it even with (inline) PTX*.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1355790639966720204)** (3 messages): 

> `torch.compile error, FlexAttention, Arbitrary Sequence Length` 


- **Torch Compile Throws Unsupported Error**: A user reported a `torch.compile` error related to `__rmul__` when using a subclassed `nn.Parameter` within a compiled function, using **torch 2.6** and **cuda 12.4** in colab.
   - The error was: `Unsupported: call_method UserDefinedObjectVariable(b) __rmul__ [ConstantVariable(int: 3)] {}` and the user wanted to know if this is a known problem or whether they should file an issue.
- **FlexAttention Now Supports Arbitrary Sequence Length**: A user inquired if **FlexAttention** now supports arbitrary sequence lengths, recalling that previous versions required sequence lengths to be multiples of **128**.
   - Another user confirmed that as of **PyTorch 2.6**, arbitrary sequence lengths are supported.


  

---


### **GPU MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1355717424409874655)** (1 messages): 

> `RoPE, BFloat16 Precision, FlashAttention2, AnchorAttention` 


- **RoPE's BFloat16 Breakdown**: A new paper ([When Precision Meets Position: BFloat16 Breaks Down RoPE in Long-Context Training](https://arxiv.org/abs/2411.13476)) identifies that **BFloat16** introduces numerical errors in **RoPE**, compromising its relative encoding, even when computed in **Float32**.
   - The first token significantly contributes to deviations as context length increases, but the paper introduces **AnchorAttention**, a plug-and-play method that improves long-context performance, reduces training time by over 50%, and preserves the model's general capabilities, with code supporting **FlashAttention** and **FlexAttention** available on [GitHub](https://github.com/haonan3/AnchorContext).
- **FlashAttention Impacted by RoPE's BFloat16 Issue**: The paper suggests that casting tensors to **BFloat16** in **FlashAttention2** causes **RoPE** to deviate from its intended relative positional encoding properties.
   - This implies that while **RoPE** might be computed in **Float32**, the use of **BFloat16** in subsequent layers like **FlashAttention2** can still introduce errors.



**Link mentioned**: <a href="https://x.com/Haonan_Wang_/status/1859608786765480516">Tweet from Haonan Wang (@Haonan_Wang_)</a>: 🚀 New Paper📜 When Precision Meets Position: BFloat16 Breaks Down RoPE in Long-Context Training🤯 RoPE is Broken because of... BFloat16!&gt; Even if RoPE is computed in Float32 (like in Llama 3 and t...

  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1355405681737469952)** (2 messages): 

> `NVIDIA RTX PRO 6000 Blackwell Workstation Edition, GDDR7, Size Zheng, Next Era of AI` 


- **Nvidia unveils RTX PRO 6000 Workstation Edition**: Nvidia has announced the **RTX PRO 6000 Blackwell Workstation Edition** featuring **96GB of GDDR7** memory, targeted towards AI and graphics-intensive tasks, as seen on their [product page](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/).
- **Authors Presenting the Next Era of AI**: The next era of AI will be presented by authors [Size Zheng](https://arxiv.org/search/cs?searchtype=author&query=Zheng,+S), [Jin Fang](https://arxiv.org/search/cs?searchtype=author&query=Fang,+J), [Xuegui Zheng](https://arxiv.org/search/cs?searchtype=author&query=Zheng,+X), [Qi Hou](https://arxiv.org/search/cs?searchtype=author&query=Hou,+Q), [Wenlei Bao](https://arxiv.org/search/cs?searchtype=author&query=Bao,+W), [Ningxin Zheng](https://arxiv.org/search/cs?searchtype=author&query=Zheng,+N), [Ziheng Jiang](https://arxiv.org/search/cs?searchtype=author&query=Jiang,+Z), [Dongyang Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang,+D), [Jianxi Ye](https://arxiv.org/search/cs?searchtype=author&query=Ye,+J), [Haibin Lin](https://arxiv.org/search/cs?searchtype=author&query=Lin,+H), [Li-Wen Chang](https://arxiv.org/search/cs?searchtype=author&query=Chang,+L), [Xin Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu,+X) in their [upcoming paper](https://arxiv.org/abs/2503.20313).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2503.20313">TileLink: Generating Efficient Compute-Communication Overlapping Kernels using Tile-Centric Primitives</a>: Large deep learning models have achieved state-of-the-art performance in a wide range of tasks. These models often necessitate distributed systems for efficient training and inference. The fundamental...</li><li><a href="https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/">NVIDIA RTX PRO 6000 Blackwell Workstation Edition</a>: Experience Unparalleled AI and Graphics Performance.
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1355390972451881082)** (7 messages): 

> `Jax Scaling Book on Transformer FLOPs, Diffusion Game Models, Pulid Faceloader Error, Apple Silicon Memory Hierarchy, Models with Large Per-Layer Dataflow` 


- **Jax Scaling Book Teaches Transformer FLOP Counting**: A member shared [jax-ml/scaling-book](https://jax-ml.github.io/scaling-book/transformers/) which provides calculation examples for autoregressive models, applicable to video models, estimating model constraints with **FLOPs, memory bandwidth, and roofline analysis**.
   - The recommendation is to benchmark against real data and profile with *nsys* to validate calculations, focusing on linear layers and attention mechanisms.
- **Pulid Faceloader Faces CUDA Problems**: A user reported that Pulid Faceloader in ComfyUI failed with a CUDA error after a reboot, despite paths being correctly set, citing an [onnxruntime issue](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements).
   - It was recommended to check that CUDA and cuDNN versions are compatible, and that the GPU is supported by the CUDA version (currently failing on **PyTorch 2.7.0** with **CUDA 12.8**).
- **Silicon Secrets: M-Series Memory Demystified?**: A member inquired about the on-chip caches and memory hierarchy in Apple Silicon M-Series GPUs, seeking the Apple equivalent to an NVIDIA A100 memory map and linked a [paper on Apple M-Series SoCs](https://arxiv.org/abs/2502.05317v1).
   - The discussion highlighted that Apple does not publicly reveal certain GPU details like NVIDIA, making it difficult to ascertain specific cache numbers, but the paper mentioned **L1 caches (192 KB per core)** and **shared L2 caches up to 24 MB** in the M4 chip.
- **Hunting Models with High-Throughput Layer Dataflow**: A member sought models with per-layer dataflow of at least ~**10GB**, not the total memory usage, and described how to meausre intermediate activations passed between consecutive layers.
   - One suggestion was to explore models processing volumetric data, such as in the medical domain, where a volume of **512³ voxels**, **32 channels**, and **fp16 activations** could yield **8GiB** of data per layer.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://jax-ml.github.io/scaling-book/transformers/"> All the Transformer Math You Need to Know | How To Scale Your Model </a>: no description found</li><li><a href="https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements),">NVIDIA - CUDA</a>: Instructions to execute ONNX Runtime applications with CUDA</li><li><a href="https://arxiv.org/abs/2502.05317v1">Apple vs. Oranges: Evaluating the Apple Silicon M-Series SoCs for HPC Performance and Efficiency</a>: This paper investigates the architectural features and performance potential of the Apple Silicon M-Series SoCs (M1, M2, M3, and M4) for HPC. We provide a detailed review of the CPU and GPU designs, t...
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1355341294989480017)** (3 messages): 

> `Twitter Embeds, Deleting Twitter` 


- **Twitter Embeds Getting Deleted**: A member shared a [link](https://vxtwitter.com/faisal_sayed05/status/1905519905845239869) about **Twitter embeds** not working.
   - Another member joked that the solution to **Twitter embed** problems is to *delete yo twitter*.
- **Twitter No More**: Another member suggested someone delete their twitter account.
   - The original poster was trying to post a link to **vxtwitter.com**.



**Link mentioned**: <a href="https://vxtwitter.com/faisal_sayed05/status/1905519905845239869">Tweet from undefined</a>: no description found

  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/)** (1 messages): 

random.oof: are there any meetups in NYC?
  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1355958801714642974)** (1 messages): 

> `Triton-lang, shared memory encoding, transpose bank conflict` 


- ****Shared Memory Encoding** added to Avoid Transpose Bank Conflict**: A [pull request](https://github.com/triton-lang/triton/pull/5797) introduces **swizzling pattern** for **B operands** in **TN GEMM**.
   - This implementation was originally done by @jtang10 in [PR#4984](https://github.com/triton-lang/triton/pull/4984).
- ****Transposed GEMM Operand Optimized** in Shared Memory**: A [pull request](https://github.com/triton-lang/triton/pull/6074) introduces **shared memory optimization**, which reduces **bank conflicts** and enables **wide LDS stores** in **NT**, **TT** and **TN GEMM** and similar cases.
   - This optimization applies when the dot operand **K dimension** is not innermost.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/triton-lang/triton/pull/5797">[AMD] Add shared memory encoding to avoid transpose bank conflict by binarman · Pull Request #5797 · triton-lang/triton</a>: This PR introduces swizzling pattern for B operands in TN GEMM.Originally implemented by @jtang10 in #4984This PR is part of series of related PRs:[AMD] Add shared memory encoding to avoid tran...</li><li><a href="https://github.com/triton-lang/triton/pull/6074">[AMD][OPTIMIZER] Optimize transposed GEMM operand in shared memory  by binarman · Pull Request #6074 · triton-lang/triton</a>: This PR introduces shared memory optimization, which reduces bank conflicts andenables wide LDS stores in NT, TT and TN GEMM and similar cases(i.e. dot operand K dimension is not innermost).This...
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/1355939141740920932)** (3 messages): 

> `Segmentation fault with liger-kernel, LigerFusedLinearCrossEntropyLoss issues, Reproducing errors with liger-kernel, Environment details for debugging liger-kernel` 


- **Segmentation Fault Hits Liger-Kernel**: A member reported encountering a **Segmentation fault (core dumped)** when using `liger-kernel` with `LigerFusedLinearCrossEntropyLoss` in a simple PyTorch script.
   - The script involved a linear layer model, input tensor, and target tensor, with the error occurring during the `loss.backward()` call.
- **Debugging Liger-Kernel Woes**: A maintainer could not reproduce the segmentation fault and requested the full error code and environment details to assist with debugging.
   - Another member inquired about the version of `liger-kernel` being used to help pinpoint the issue.
- **Investigating LigerFusedLinearCrossEntropyLoss Issues**: The reported issue centers around the `LigerFusedLinearCrossEntropyLoss` function within the `liger-kernel` library.
   - The function fuses linear and cross-entropy layers, performing chunk-by-chunk computation to reduce memory usage, but seems to be triggering a segmentation fault in certain configurations.


  

---


### **GPU MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/1355269257881194667)** (6 messages): 

> `CUDA compiler for Apple GPU, Metal C++, Zig support for Apple GPU, spirv-cross, IREE's Metal HAL driver` 


- **CUDA Compiler coming to Apple GPU**: With repos like [this one](https://github.com/openai/triton), it seems like you could in theory make a **CUDA compiler** for the **Apple GPU** by compiling to **Metal C++**.
   - One member said they've thought about adding similar support for **Zig**, since **Apple** uses **LLVM IR**.
- **Metal Compute Shaders via IREE**: A member suggests using the compute shader in [IREE's Metal HAL driver](https://iree.dev/developers/design-docs/metal-hal-driver/#compute-pipeline) to target **Metal**.
   - He says *it works*, though obviously it's subject to limitations of what **SPIRV** can represent and what **SPIRV-cross** supports.



**Link mentioned**: <a href="https://iree.dev/developers/design-docs/metal-hal-driver/#compute-pipeline)">Metal HAL driver - IREE</a>: no description found

  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1355548515543285881)** (4 messages): 

> `Bend parallel language, Phazr AI video avatar tool, CuTe predication, File I/O in Bend` 


- ****Bend** your code into shape with Parallelism**: HigherOrderCo introduced **Bend**, a [massively parallel, high-level programming language](https://higherorderco.com/) for multi-core CPUs/GPUs, designed to feel like Python without the complexities of concurrent programming.
- **Phazr Alchemize Your Video Persona**: A member released **Phazr AI**, a [free tool](https://www.phazr.ai/) that allows users to appear as anyone in video calls, utilizing audio-driven portrait animation and running locally for privacy.
- **Tiling Triumph: CuTe predication tutorial drops**: Simon Veitner posted a [blog](https://veitner.bearblog.dev/predication-in-cutlass/) about performing predication in CuTe to help generalize tiling kernels, including a [link to the Cutlass documentation](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/0y_predication.md).
- **Bend adds file I/O**: **File I/O** capabilities were introduced to **Bend**, enabling users to perform file operations, as documented [here](https://github.com/HigherOrderCO/Bend/blob/main/docs/builtins.md#file-io).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.phazr.ai/">phazr</a>: no description found</li><li><a href="https://higherorderco.com/">Higher Order Company</a>: no description found</li><li><a href="https://veitner.bearblog.dev/predication-in-cutlass/">Predication in Cutlass</a>: The cutlass documentation on CuTe touches the topic of  briefly but doesn&#x27;t give a full code example.
In this blogpost I will explain how to use predication...</li><li><a href="https://github.com/HigherOrderCO/Bend/blob/main/docs/builtins.md#file-io">Bend/docs/builtins.md at main · HigherOrderCO/Bend</a>: A massively parallel, high-level programming language - HigherOrderCO/Bend
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1356365091754344558)** (1 messages): 

> `AlphaGeometry LLM + verifier for kernel optimization` 


- **Inquire about AlphaGeometry LLM for Kernel Optimization**: A member inquired about using an **AlphaGeometry-style LLM + verifier** for the kernel optimization process.
   - They are seeking information on the history of this idea, whether it has been tried, and any related discussions, as they are new to the field and suspect they might be *rediscovering existing concepts*.
- **Unexplored Territory: Kernel Optimization with AlphaGeometry LLM**: Discussion revolves around leveraging **AlphaGeometry-style LLMs with verifiers** to potentially revolutionize kernel optimization processes.
   - The inquiry focuses on understanding if this approach has been previously explored, and seeks insights into prior attempts or relevant discussions within the community, recognizing the possibility of revisiting existing methodologies.


  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1356088943392849920)** (2 messages): 

> `FusedMLP, tiny-cuda-nn, ThunderKittens` 


- **User asks FusedMLP exist in ThunderKittens**: A user asked if **FusedMLP** of the form from [NVlabs/tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) exists within [HazyResearch/ThunderKittens](https://github.com/HazyResearch/ThunderKittens/tree/main/kernels/hedgehog).
- **Newbie Seeks Guidance in TK Land**: A user, identifying as a newbie, inquired about finding a specific implementation (FusedMLP) within the ThunderKittens repository.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/HazyResearch/ThunderKittens/tree/main/kernels/hedgehog">ThunderKittens/kernels/hedgehog at main · HazyResearch/ThunderKittens</a>: Tile primitives for speedy kernels. Contribute to HazyResearch/ThunderKittens development by creating an account on GitHub.</li><li><a href="https://github.com/NVlabs/tiny-cuda-nn">GitHub - NVlabs/tiny-cuda-nn: Lightning fast C++/CUDA neural network framework</a>: Lightning fast C++/CUDA neural network framework. Contribute to NVlabs/tiny-cuda-nn development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/1356276080582987888)** (26 messages🔥): 

> `Datasets without curricula, Futoshiki dataset generation speed, Private benchmarking service, OpenAI open-weight reasoning models` 


- ****Datasets Lack Curriculum, Difficulty Tweaks Debated****: There are datasets without curricula (`acre`, `arc_1d`, `arc_agi`, `codeio`, `composite`, `countdown`, `futoshiki`, `gcd`, `gsm_symbolic`, `knight_swap`, `knights_knaves`, `list_functions`, `puzzle24`, `syllogism`, `word_sorting`), and some like `gsm_symbolic` and `list_functions` present challenges in adjusting their difficulties for curriculum design, with ongoing bug investigations.
   - Some members are focusing on collision tasks and reporting issues with specific datasets like `gsm_symbolic`, prompting discussions on misconfigurations and fixes.
- ****Futoshiki's Speed Spurs Sample Size Scrutiny****: The `futoshiki` dataset faces challenges in generating 10,000 datasets within a reasonable time (10 minutes was not enough), leading to questions about acceptable generation speed and adjustments to grid size configurations.
   - It was suggested that a max grid size of **6 or 7** should be set if you want to quickly generate a lot of samples; in theory, collisions should be much less common for higher grid sizes anyway.
- ****Private Benchmarking Service Blueprinted****: A [work-in-progress pull request](https://github.com/open-thought/reasoning-gym/pull/398) aims to create a private benchmarking service where users can fill in blanks and upload results to Gradio for grading, ensuring no sensitive information is revealed.
   - The initiative involves generating a complete set of questions with blank answers to enable a *hidden-answer* benchmarking service, allowing for a more controlled evaluation process.
- ****OpenAI Open-Weights Announcement Astounds Observers****: OpenAI's announcement of publishing strong open-weight reasoning models surprised the community, sparking speculation about the company's motives, particularly in the context of potential fundraising efforts.
   - Releasing an open-weight model could significantly raise their valuation if it generates widespread interest and adoption, as everyone will go crazy about it.



**Link mentioned**: <a href="https://github.com/open-thought/reasoning-gym/pull/398">[WIP] Generate Seeded Benchmark Test for Distribution by Miserlou · Pull Request #398 · open-thought/reasoning-gym</a>: Adds a script to create a complete set of questions with blank answers which can be used for a hidden-answer benchmarking service.RNG_SEED=321 python scripts/generate_benchmark.py --num-per-datase...

  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1343002580531417211/1356081760882000064)** (2 messages): 

> `Discord ID display issues, Discord permissions, Leaderboard ID formatting` 


- **Discord ID Display Mystery**: A user inquired why their ID on the leaderboard appears as **User_1184712546704429106** instead of their actual Discord ID.
   - The community suspects it relates to **Discord permissions**, but a solution remains elusive.
- **Discord Perms Cause ID Display issues**: Members believe that **Discord perms** cause ID display issues.
   - There is **zero** idea on how to fix this. 


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1355267324776808651)** (94 messages🔥🔥): 

> `vectoradd benchmarks, vectorsum benchmarks, conv2d benchmarks` 


- **Vectoradd Benchmarks Boom on H100**: Multiple successful leaderboard submissions for `vectoradd` were recorded on **H100** GPUs using Modal runners, including IDs `3247`, `3248`, `3255`, `3256`, `3257`, `3258`, `3259`, `3351`, `3353`, `3367`, `3368`, and `3369`.
- **Vectorsum Scores Soar on L4**: Numerous successful benchmark and leaderboard submissions for `vectorsum` were reported on **L4** GPUs using Modal runners, with IDs ranging from `3272` to `3322` and again from `3352` to `3372`.
- **Conv2d contentions conquered on L4,T4,A100,H100**: A leaderboard submission with id `3373` to leaderboard `conv2d` on GPUS: **L4, T4, A100, H100** using Modal runners succeeded!
- **Vectorsum tests tantalize on H100**: A successful leaderboard submission with id `3374` to leaderboard `vectorsum` on GPUS: **H100** using Modal runners succeeded.
- **A100 Aces Vectoradd**: A test submission with id `3338` and a leaderboard submission with id `3288` to leaderboard `vectoradd` on GPUS: **A100** using Modal runners succeeded.


  

---


### **GPU MODE ▷ #[hardware](https://discord.com/channels/1189498204333543425/1349152646484987974/1355308604462338280)** (13 messages🔥): 

> `GPU temperature issues with PyTorch distributed training, Detecting GPU health before training, H100 GPU temperature anomaly on AWS` 


- ****AWS H100** Hot Spot Troubleshoot**: A user reported experiencing high temperatures on a specific GPU during **PyTorch distributed training** on AWS **H100** nodes, with one GPU consistently reaching **90C** while others averaged **40C**.
   - The user noted that this temperature anomaly slows down training, and they sought advice on pre-training hardware/software sanity checks, like NCCL or connection checks.
- ****Power Limiting** Temperature Mitigation**: A user experiencing high GPU temperatures during PyTorch distributed training was advised to use `sudo nvidia-smi -pl` to [power limit the GPU](https://developer.nvidia.com/nvidia-system-management-interface).
   - It was suggested that this could mitigate temperature concerns.
- **Seek AWS Support for Persistent GPU Thermal Issue**: A user experiencing persistent high GPU temperatures on a specific **H100** node on AWS was advised to seek assistance from AWS support.
   - It was suggested that if the issue is a mechanical problem with the cooler, stress testing might be the only way to detect it before training.


  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1355269093397364816)** (64 messages🔥🔥): 

> `Softmax AI Alignment Startup, xAI and X Merge, GPT-4o Image Generation, Gemini 2.5 Pro, MiniMax Audio Speech-02 Model` 


- **Shear Genius: Softmax Emerges for AI Alignment**: Emmett Shear, Adam Goldstein, and David Bloomin founded **Softmax**, a 10-person startup in San Francisco focused on **organic alignment**, fusing human and AI goals by drawing from nature and intelligent systems, detailed in a [Core Memory article](https://www.corememory.com/p/exclusive-emmett-shear-is-back-with-softmax).
- **X Marks the AI Spot: xAI Merges with X**: Elon Musk announced that **xAI** is *'blending xAI’s advanced AI capability and expertise with X’s massive reach'* in a merger detailed by [The Verge](https://www.theverge.com/news/638933/elon-musk-x-xai-acquisition).
- **GPT-4o's Image Generation: Frontend Illusion?**: A user discovered that **GPT-4o's** line-by-line image generation effect is a browser-side animation, with the server sending only **5 intermediate images** at a patch size of **8**, according to [this tweet](https://x.com/jie_liu1/status/1905761704195346680).
- **Gemini 2.5 Pro Goes Experimental, Expands Access**: **Gemini 2.5 Pro** (experimental) is now available to all Gemini users due to TPUs *running hot*, as announced on [GeminiApp's Twitter](https://fxtwitter.com/GeminiApp/status/1906131622736679332).
- **MiniMax Launches Audio Speech-02 with TTS**: **MiniMax AI** launched **Speech-02**, which turns any file or URL into lifelike audio instantly in **30+ languages** with native flair, unlimited voice cloning, and sub-second streaming detailed on [MiniMax's Twitter](https://fxtwitter.com/MiniMax__AI/status/1906720764885180775).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/jie_liu1/status/1905761704195346680">Tweet from Jie Liu (@jie_liu1)</a>: After hacking GPT-4o&#39;s frontend, I made amazing discoveries:💡The line-by-line image generation effect users see is just a browser-side animation (pure frontend trick)🔦OpenAI&#39;s server sends o...</li><li><a href="https://fxtwitter.com/runwayml/status/1906718935778545964">Tweet from Runway (@runwayml)</a>: Today we&#39;re introducing Gen-4, our new series of state-of-the-art AI models for media generation and world consistency. Gen-4 is a significant step forward for fidelity, dynamic motion and control...</li><li><a href="https://www.corememory.com/p/exclusive-emmett-shear-is-back-with-softmax">Exclusive: Emmett Shear Is Back With a New Company and A Lot of Alignment </a>: Insert coup pun here</li><li><a href="https://x.com/btibor91/status/1906642469531595005">Tweet from Tibor Blaho (@btibor91)</a>: OpenAI temporarily disabled Sora video generation for new accounts due to heavy traffic</li><li><a href="https://x.com/sama/status/1906793591944646898">Tweet from Sam Altman (@sama)</a>: TL;DR: we are excited to release a powerful new open-weight language model with reasoning in the coming months, and we want to talk to devs about how to make it maximally useful: https://openai.com/op...</li><li><a href="https://x.com/allen_ai/status/1906734336537501948">Tweet from Ai2 (@allen_ai)</a>: Imagine AI doing science: reading papers, generating ideas, designing and running experiments, analyzing results… How many more discoveries can we reveal? 🧐Meet CodeScientist, a promising next step t...</li><li><a href="https://x.com/MiniMax__AI/status/1906722525029040560">Tweet from MiniMax (official) (@MiniMax__AI)</a>: Try it now 👉 https://www.minimax.io/audio/API access coming soon—visit https://www.minimax.io/platform/ or contact api@minimaxi.com for early access!</li><li><a href="https://fxtwitter.com/MiniMax__AI/status/1906720764885180775">Tweet from MiniMax (official) (@MiniMax__AI)</a>: MiniMax Audio just leveled up with the new Speech-02 model!Turn any file or URL into lifelike audio instantly. Create audiobooks and podcasts effortlessly with up to 200k-character in a single input. ...</li><li><a href="https://x.com/allhands_ai/status/1906760162406285442">Tweet from All Hands AI (@allhands_ai)</a>: Today, we&#39;re excited to make two big announcements!- OpenHands LM: The strongest 32B coding agent model, resolving 37.4% of issues on SWE-bench Verified 📈- OpenHands Cloud: SOTA open-source codin...</li><li><a href="https://x.com/GeminiApp/status/1906206243053846558">Tweet from Google Gemini App (@GeminiApp)</a>: @dylanjturner9 hey @dylanjturner9, free users have rate limits on this model, which do not apply to Advanced users. Your sub also gets you a longer context window.</li><li><a href="https://fxtwitter.com/GeminiApp/status/1906131622736679332">Tweet from Google Gemini App (@GeminiApp)</a>: Gemini 2.5 Pro is taking off 🚀🚀🚀The team is sprinting, TPUs are running hot, and we want to get our most intelligent model into more people’s hands asap.Which is why we decided to roll out Gemini 2...</li><li><a href="https://x.com/sama/status/1906771292390666325">Tweet from Sam Altman (@sama)</a>: the chatgpt launch 26 months ago was one of the craziest viral moments i&#39;d ever seen, and we added one million users in five days.we added one million users in the last hour.</li><li><a href="https://allenai.org/papers/codescientist">no title found</a>: no description found</li><li><a href="https://x.com/nrehiew_/status/1905930295750107591">Tweet from wh (@nrehiew_)</a>: My current working guess is that the first image is ARed out and decoded via a (vq)vae. It is then used as the starting point (instead of noise) for some form of block wise diffusion in pixel space.Qu...</li><li><a href="https://fxtwitter.com/CyouSakura/status/1906737585063641532">Tweet from Yasmine (@CyouSakura)</a>: 🥳 Excited to announce major updates to Open-Reasoner-Zero (ORZ), our open-source initiative scaling Reinforcement Learning on base models!🌊 Updated Paper & Superior ResultsUsing the same base model ...</li><li><a href="https://labs.amazon.science/blog/nova-act">Introducing Amazon Nova Act | Amazon AGI Labs</a>: no description found</li><li><a href="https://aws.amazon.com/ai/generative-ai/nova/creative/">Image and Video Generation Models – Amazon Nova Creative Content Generation Models – AWS</a>: no description found</li><li><a href="https://www.theverge.com/news/638933/elon-musk-x-xai-acquisition">Elon Musk’s xAI buys Elon Musk’s X for $33 billion on paper</a>: Shuffling paper.
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1355649723025063986)** (5 messages): 

> `Diffusion Models, System Card` 


- **Diffusion Details Debated**: A member questioned the inference source, noting it's *clearly not a standard diffusion model* but diffusion does by default go from **low frequency to high** during sampling.
   - They added that *none of this seems anti diffusion it just happens to be better*.
- **System Card Sentence Sparks Curiosity**: In response to a question about the inference source, a member cited a vague sentence in the system card as the origin of the inference.
   - The original question was that *am curious where this is being inferred from*.


  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1355274866395185182)** (33 messages🔥): 

> `GPT-4o Image Generation, Chorus Pricing, Princess Mononoke IMAX Re-Release, Advanced Voice Mode, Manufacturing CNC` 


- ****GPT-4o's** Quirky Image Gen Revealed**: A user shared a [tweet](https://x.com/xlr8harder/status/1906247140977856942) discussing the mysteries of **GPT-4o's** image generation, noting that the final rendered image is placed back into the model's context window.
   - The user questions *why the control flow is returned to the model*, which explains why it sometimes responds with *"Understood"* after generation.
- **Chorus Price Hike**: The paid tier for [Chorus](https://chorus.sh/pricing) has increased to **$100/month**, providing access to all models or the option to bring your own API keys.
   - The previous pricing was **$20/month**, which one user noted was unsustainable, but some users were *"grandfathered in"* to the old pricing, at least temporarily.
- ****Mononoke** Makes Millions in IMAX**: The re-release of **Studio Ghibli’s Princess Mononoke** for IMAX was a smash hit, making **$4 million** over one weekend, exceeding its original North American run of **$2.4 million** in 1999, according to a [tweet](https://x.com/btibor91/status/1906441722223305081).
   - It was wondered if the recent rise of **Ghibli-style art via ChatGPT Image Gen** may be *driving fresh excitement* back to the original creators.
- ****Voice Mode** Vibes with **GPT-4o****: The advanced voice mode now uses natively multimodal models like **GPT-4o**, directly processing and generating audio for more natural conversations, according to a [press release](https://openai.com/index/hello-gpt-4o/ ).
   - This picks up on non-verbal cues like speech speed and emotion, although usage is limited daily for users, with free users getting a preview powered by **4o-mini**.
- **Manufacturing Maniac Seeks CNC Summer Gig**: One user is sending random manufacturing founders emails asking to operate their **CNC** machines and *"shit"* for the summer.
   - They are trying to get a factory floor job for the summer.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://chorus.sh/pricing">no title found</a>: A Mac app for chatting with a bunch of AIs at once.</li><li><a href="https://www.twitch.tv/gemini_plays_pokemon">Twitch</a>: no description found</li><li><a href="https://x.com/janbamjan/status/1905754302515396701">Tweet from janbam (@janbamjan)</a>: hallelujah!claude takes an umprompted break</li><li><a href="https://fxtwitter.com/patience_cave/status/1905986861643993286">Tweet from 💺 (@patience_cave)</a>: I’ve been putting 4o images to use by writing a long comic strip! It is a significant test of capabilities. Generating 50 visually consistent panels it took about 10 hours. No small amount of art and ...</li><li><a href="https://x.com/xlr8harder/status/1906247140977856942">Tweet from xlr8harder (@xlr8harder)</a>: Mysteries of gpt4o image gen:  When the final rendered image is placed in the model&#39;s context window, they give the message below to the model.But why return the control flow to the model at all?(...</li><li><a href="https://x.com/btibor91/status/1906441722223305081>">Tweet from Tibor Blaho (@btibor91)</a>: Did you know the recent IMAX re-release of Studio Ghibli’s Princess Mononoke is almost completely sold out, making more than $4 million over one weekend - more than its entire original North American ...</li><li><a href="https://www.boxofficemojo.com/release/rl339312641/?ref_=bo_rl_tab#tabs">no title found</a>: no description found
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1355506845992620227)** (10 messages🔥): 

> `Internal Knowledge Google Workspace setup, AI to make art more accessible` 


- **OpenAI offers Internal Knowledge Google Workspace setup**: Members shared an [article](https://help.openai.com/en/articles/10929079-internal-knowledge-google-workspace-admin-managed-setup) detailing the steps for setting up **Internal Knowledge Google Workspace**.
- **AI makes art more accessible**: A member shared that someone is using **AI** to make art more accessible by transforming existing pieces into the popular **Corporate Memphis style** [tweet](https://x.com/xlr8harder/status/1906643226544492832).



**Link mentioned**: <a href="https://x.com/xlr8harder/status/1906643226544492832">Tweet from xlr8harder (@xlr8harder)</a>: Thrilled to announce my new series where I&#39;m using AI to make art more accessible by transforming existing pieces into the popular Corporate Memphis style!

  

---


### **Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/1355842314404368518)** (3 messages): 

> `RL Tutorial v2, Bibliographic Tools on arXiv, Large Language Models` 


- **RL Tutorial Gets an Upgrade**: A member announced the v2 release of their **RL tutorial**, featuring a new chapter on **multi-agent RL**, improved sections on **'RL as inference'** and **'RL+LLMs'**, and some typo fixes ([link to tweet](https://x.com/sirbayes/status/1904375008627138851)).
- **Bibliographic Tools on arXiv**: A member shared a link to **arXiv's Bibliographic and Citation Tools** page, which includes sections for Code, Data, Media, Demos, and Related Papers ([link to arXiv](https://arxiv.org/abs/2412.05265v2)).
- **LLMs Show Remarkable Reasoning**: A link was shared to a paper from arXiv about **Large Language Models (LLMs)** showing remarkable reasoning ability.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2412.05265v2">Reinforcement Learning: A Comprehensive Overview</a>: This manuscript gives a big-picture, up-to-date overview of the field of (deep) reinforcement learning and sequential decision making, covering value-based method, policy-gradient methods, model-based...</li><li><a href="https://x.com/sirbayes/status/1904375008627138851">Tweet from Kevin Patrick Murphy (@sirbayes)</a>: I&#39;m happy to announce that v2 of my RL tutorial is now online. I added a new chapter on multi-agent RL, and improved the sections on &#39;RL as inference&#39; and &#39;RL+LLMs&#39; (although latte...</li><li><a href="https://www.arxiv.org/abs/2503.19470">ReSearch: Learning to Reason with Search for LLMs via Reinforcement Learning</a>: Large Language Models (LLMs) have shown remarkable capabilities in reasoning, exemplified by the success of OpenAI-o1 and DeepSeek-R1. However, integrating reasoning with external search processes rem...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[rlhf](https://discord.com/channels/1179127597926469703/1208183230608576562/1355649149348872212)** (12 messages🔥): 

> `Reward Models, RLHF Prompt Data, Reward Hacking` 


- **Hinton Hates RLHF, Calls it Lipstick on Pig**: Geoffrey Hinton says [RLHF is a pile of crap](https://x.com/vitrupo/status/1905858279231693144), and likens it to a *paint job for a rusty car* you want to sell.
- **Industry Insider Endorses Llama-3.1 Nemotron Reward Model**: The [Nvidia Llama-3.1 Nemotron-70B-Reward-HF](https://huggingface.co/nvidia/Llama-3.1-Nemotron-70B-Reward-HF) is considered a good general-purpose Reward Model, while rewardbench is *very outdated*.
- **New Hybrid Reward System Combats Reward Hacking**: A new [paper](https://arxiv.org/abs/2503.22230) explores data-driven bottlenecks in RLHF performance scaling, particularly **reward hacking** and **decreasing response diversity**, and introduces a hybrid reward system combining **Reasoning Task Verifiers (RTV)** and a **Generative Reward Model (GenRM)** to mitigate reward hacking.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2503.22230">Exploring Data Scaling Trends and Effects in Reinforcement Learning from Human Feedback</a>: Reinforcement Learning from Human Feedback (RLHF) is crucial for aligning large language models with human preferences. While recent research has focused on algorithmic improvements, the importance of...</li><li><a href="https://x.com/vitrupo/status/1905858279231693144">Tweet from vitrupo (@vitrupo)</a>: Geoffrey Hinton says RLHF is a pile of crap. He likens it to a paint job for a rusty car you want to sell.
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[cv](https://discord.com/channels/1179127597926469703/1208183243447468093/1355358682745147525)** (4 messages): 

> `Moondream release, Image captioning, HF repo` 


- **Moondream releases new version**: The newest [Moondream release](https://moondream.ai/blog/moondream-2025-03-27-release) includes a **Long** format for image captioning, generating roughly **2x longer captions** than the **Normal** format.
- **Vik recycles HF repo, draws ire**: A member expressed their wish that Vik would create a new Hugging Face repository instead of reusing the same one.
   - They added they *wonder how he gets the detection performance so well. Maybe just understands what customers want better*.



**Link mentioned**: <a href="https://moondream.ai/blog/moondream-2025-03-27-release">Moondream 2025-03-27 Release</a>: Moondream release announcement.

  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1355506038001897622)** (10 messages🔥): 

> `Sam Altman Firing, AI Energy Consumption, LLMs on Math Olympiad, GPT-4o and Studio Ghibli Style Images` 


- **Thiel Warns Altman About OpenAI's Direction**: Peter Thiel cautioned Sam Altman about **OpenAI's** path during a dinner in L.A.'s Arts District in November 2023, as detailed in [The Wall Street Journal](https://www.wsj.com/tech/ai/the-real-story-behind-sam-altman-firing-from-openai-efd51a5d).
- **Energy Use of AI Chatbots is Surprisingly Low**: A [blog post](https://engineeringprompts.substack.com/p/ai-energy-use) compares AI chatbot energy consumption over a year to everyday activities, revealing it uses less energy than driving a car for **10 kilometers** or taking **five short hot showers**.
   - The author provides a visual aid, showing that a year of chatbot use consumes even less energy than filling **two hot baths**.
- **LLMs Flunk 2025 US Math Olympiad**: An X post reported that current **SOTA LLMs** performed poorly on the **2025 US Math Olympiad**, achieving only a **5%** success rate on **6 problems** [ZainHasan6 Tweet](https://x.com/ZainHasan6/status/1906767036975301047).
- **GPT-4o Channels Studio Ghibli**: Following **Sam Altman's** announcement of a new image model update to **GPT-4o**, many online users generated **Studio Ghibli** style images, as covered in a [Technollama blog post](https://www.technollama.co.uk/the-style-returns-some-notes-on-chatgpt-and-studio-ghibli).
   - Altman himself shared a modified photo with developers in the **Studio Ghibli** style, captioned *“Feel the AGI”*.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/ZainHasan6/status/1906767036975301047">Tweet from Zain (@ZainHasan6)</a>: they tested sota LLMs on 2025 US Math Olympiad hours after the problems were releasedTested on 6 problems and spoiler alert!They all suck -&gt; 5%</li><li><a href="https://engineeringprompts.substack.com/p/ai-energy-use">AI Energy Use in Everyday Terms</a>: Comparing chatbot energy consumption to the things people actually use and understand</li><li><a href="https://www.technollama.co.uk/the-style-returns-some-notes-on-chatgpt-and-studio-ghibli">The Style Returns: Some notes on ChatGPT and Studio Ghibli</a>: If you were online between March 25 and March 26, your timeline may have been flooded with a barrage of AI-generated images, a large number of which would have been existing photographs recreated u…</li><li><a href="https://archive.is/xP4N1">Exclusive | The Real Story Behind Sam Altman&#x2019;s Firing From OpenAI - W&#x2026;</a>: no description found
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1355956094962499787)** (20 messages🔥): 

> `Sora's Refusals, C2PA Protection Bypass, Watermarking Discussions` 


- **Sora Softens Stance on Sensitive Image Generation**: Sora is shifting from *blanket refusals* in sensitive areas to a more precise approach focused on preventing real-world harm, as shared in a [Discord post](https://cdn.discordapp.com/attachments/1228051082631188530/1355965151077077233/CleanShot_2025-03-30_at_19.58.51.png?ex=67ec29f4&is=67ead874&hm=d319601bffc132b42c26a426c09b70b0774ef7e27334e310e6c8297e4ca0fc4c).
   - According to one user, you can generate images of any living politician without problems using **Sora**, but **NSFW/NSFL** content is blocked via an internal search tool.
- **C2PA Protection Easily Defeated**: The **C2PA protection** used by **Sora** can be bypassed by simply converting the file format or taking a screenshot.
   - A user pointed out that this protection, intended to ensure image authenticity, is not robust enough to prevent misuse.
- **Watermarking Under Scrutiny as 'Dumb but Smart'**: A member expressed a dim view of **watermarking**, calling it *dumb but smart*.
   - They stated they wanted to *see what paid conversion looks like* and that practicing writing about it was valuable.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1355393202433032222)** (7 messages): 

> `Chris Lattner's work, Modular forum link, Notebooks and Mojo` 


- **Lattner Shares List of Published Work**: [Chris Lattner](https://nondot.org/sabre) shared a link to a list of his published work, including **LLVM**, **Clang**, **Swift**, **MLIR**, and **CIRCT**, also mentioning his leadership at **Modular AI** and board membership at the **LLVM Foundation**.
- **Mojo REPL Deprecation Forum Link Shared**: A member shared a link to a Modular forum discussion about the [deprecation of the Mojo REPL](https://forum.modular.com/t/mojo-repl-deprecation/1158/4?u=melodyogonna).
- **Notebooks are championed for Mojo's Packaging**: A member mentioned that **Jeremy Howard** is a huge proponent of using notebooks not just for experimentation, but even for packaging with **Mojo**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://nondot.org/sabre/Resume.html#writing.">Chris Lattner's Resumé</a>: no description found</li><li><a href="https://forum.modular.com/t/mojo-repl-deprecation/1158/4?u=melodyogonna">Addressing some recent breaks to the Mojo REPL</a>: Awesome, thank you Owen!  I can give more color from my understanding on this, but I’ll let others make the call.  I think it is less about Python or use-cases specifically, and more driven by interna...
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1355310925480984696)** (138 messages🔥🔥): 

> `Homoiconicity in AI, Tail Call Optimization in Mojo, Mojo's 'out' argument convention, Heterogeneous Structs in Mojo Lists, Saturating Arithmetic in Mojo` 


- **Mojo Bug Exposes Infer-Only Parameter Hiccups**: A user reported a [bug](https://github.com/modular/max/issues/4199) where infer-only parameters are sometimes overwritten by positional parameters, causing compilation failure in specific scenarios involving traits and structs.
   - The issue occurs when calling a method with an infer-only parameter, while the equivalent function call works as expected; a fix is pending.
- **Revamping Mojo's 'out' Argument Syntax: A Readability Refactor**: A proposal has been made to [improve the readability of Mojo's `out` argument convention](https://github.com/modular/max/issues/4200) by specifying the type of `out` arguments as a return type in documentation and language.
   - The discussion involved the placement of `out` arguments (first vs. last) and the possibility of supporting multiple `out` arguments for scenarios like initializing channels with separate read and write halves.
- **Mojo Lists Segfault with Traits, Variant to the Rescue**: A user encountered a segmentation fault ([issue #4218](https://github.com/modular/max/issues/4218)) when trying to create a `List` of trait objects, specifically `List[Estimator]`, and appending instances of `KNN` and `SVM` structs.
   - As a workaround, it was suggested to use `List[Variant[KNN, SVM]]` and iterate through the values, checking the type using `isa` to call the appropriate methods, as trait instances are not fully supported yet.
- **`def` vs `fn`: The Great Mojo Debate**: A discussion emerged regarding the usage of `def` versus `fn` in Mojo, with some arguing that `fn` should be the default due to its type safety and better integration with typed Python workflows using tools like Mypy.
   - Others contended that `def` still has a place for beginners and those who prefer a more Python-like syntax, especially when interacting with untyped Python libraries, leading to a feature request to [make `def` default to returning None](https://github.com/modular/max/issues/4211).
- **GCD-Fueled Ratios: Metaprogramming Simplifies Fractions**: A user showcased a clever use of compile-time metaprogramming to automatically simplify `Ratio` structs using `gcd`, resulting in simplified fractions at compile time, though it was noted that this approach could cause headaches when metaprogramming.
   - An alternative was proposed to make the simplification an explicit function call rather than automatic, drawing inspiration from `std::ratio` in C++.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/modular/max/issues/4211">[Feature Request] Make `def` default to returning None · Issue #4211 · modular/max</a>: Review Mojo&#39;s priorities I have read the roadmap and priorities and I believe this request falls within the priorities. What is your request? Per discussion on the forum, I&#39;d like to request t...</li><li><a href="https://github.com/modular/modular-community">GitHub - modular/modular-community: A repo to hold community-submitted rattler-build recipes, to make community packages available via the modular-community prefix.dev channel</a>: A repo to hold community-submitted rattler-build recipes, to make community packages available via the modular-community prefix.dev channel - modular/modular-community</li><li><a href="https://github.com/modular/max/issues/4200#issuecomment-2763909956)">[Feature Request] Specify type of `out` arguments as return type in doc gen and language · Issue #4200 · modular/max</a>: Review Mojo&#39;s priorities I have read the roadmap and priorities and I believe this request falls within the priorities. What is your request? I want to make the following suggestions: Treat out ar...</li><li><a href="https://github.com/modular/max/issues/4199">[BUG] Infer-only parameters are sometimes overwritten by positional parameters · Issue #4199 · modular/max</a>: Bug description Actual behavior Consider the following example: trait Trait(CollectionElement): fn f(self): ... @value struct Struct(Trait): fn f(self): pass @value struct TestStruct[T: CollectionE...</li><li><a href="https://github.com/modular/max/issues/4200">[Feature Request] Specify type of `out` arguments as return type in doc gen and language · Issue #4200 · modular/max</a>: Review Mojo&#39;s priorities I have read the roadmap and priorities and I believe this request falls within the priorities. What is your request? I want to make the following suggestions: Treat out ar...</li><li><a href="https://github.com/samufi/larecs/blob/c38214e900fdf3d276cd30b41f70154ca1738653/src/larecs/unsafe_box.mojo#L72)">larecs/src/larecs/unsafe_box.mojo at c38214e900fdf3d276cd30b41f70154ca1738653 · samufi/larecs</a>: Larecs🌲 – a performance-oriented archetype-based ECS - samufi/larecs</li><li><a href="https://github.com/modular/max/issues/4218">[BUG] Segmentation Fault When Using Trait Objects in Generic Collections (List[Estimator]) · Issue #4218 · modular/max</a>: Bug description Actual behavior When attempting to instantiate a List[Estimator] and appending instances of KNN and SVM, Mojo crashes with a segmentation fault during the parsing of the statement t...</li><li><a href="https://github.com/samufi/larecs/blob/c38214e900fdf3d276cd30b41f70154ca1738653/src/larecs/scheduler.mojo#L56">larecs/src/larecs/scheduler.mojo at c38214e900fdf3d276cd30b41f70154ca1738653 · samufi/larecs</a>: Larecs🌲 – a performance-oriented archetype-based ECS - samufi/larecs</li><li><a href="https://github.com/modular/max/issues/1863">[Feature Request] Remove special syntax `(T, S)` for `Tuple[T, S]` · Issue #1863 · modular/max</a>: Review Mojo&#39;s priorities I have read the roadmap and priorities and I believe this request falls within the priorities. What is your request? As title. What is your motivation for this change? Typ...
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1355385004057821225)** (2 messages): 

> `CUDA definition, DeepSeek bypassing CUDA, NVIDIA driver vs CUDA` 


- **CUDA: The Backbone of Deep Learning?**: A member shared [a blog post](https://www.modular.com/blog/democratizing-compute-part-2-what-exactly-is-cuda) defining **CUDA** as the *backbone of deep learning* and the *core of NVIDIA’s moat*.
- **DeepSeek Bypasses CUDA via PTX Layer**: The member noted that [DeepSeek's breakthrough](https://www.tomshardware.com/tech-industry/artificial-intelligence/deepseeks-ai-breakthrough-bypasses-industry-standard-cuda-uses-assembly-like-ptx-programming-instead) was achieved by **bypassing CUDA** and directly accessing the **PTX layer**.
- **NVIDIA driver confusion**: A member mentioned that *the NVIDIA driver isn't counted as cuda* and that **NVIDIA** is *a bit all over the place and inconsistent in their terminology over time*.



**Link mentioned**: <a href="https://www.modular.com/blog/democratizing-compute-part-2-what-exactly-is-cuda">Modular: Democratizing AI Compute, Part 2: What exactly is “CUDA”?</a>: no description found

  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1355931165877469417)** (15 messages🔥): 

> `Video Snippets, Mind Maps, Multi-Modal Output, Android Sharing System, AI Voice Pronunciation` 


- **Video Snippets Requested in Responses**: Users are requesting **NotebookLM** to include **video snippets** in its responses when a video is used as a source to provide **visuals**.
   - One member suggested that in the future, the team will enable **multi-modal output**.
- **Mind Maps Export Wishlisted**: A user asked about exporting **Mind Maps** in **DOT format** or publishing an interactive applet with the Google UI.
   - It was implied this is not currently possible.
- **Android Sharing System Integration Sought**: Users are requesting **NotebookLM** to participate in the **Android sharing system**, suggesting a need for a dedicated app.
   - One user suggested that choosing NotebookLM from the share menu could automatically search inside a default notebook.
- **A.I. Pronunciation Fumbles Addressed**: A user is seeking ways to improve the pronunciation of words by **AI voices** in **NotebookLM**, particularly for company names with *funky* spellings.
   - The user is hoping to find ways to get the **audio overview** to pronounce company names correctly, by feeding the AI with another source with the correct pronunciation.
- **AI Debate Prompting Troubles**: A user reported issues with prompting **NotebookLM** to generate a **heated debate** between two hosts with differing viewpoints on **AI in mental health**.
   - Another user suggested using internal names for the voices (*Host Speaker* and *Expert Speaker*) to help assign roles.


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1355256962216103936)** (96 messages🔥🔥): 

> `NotebookLM Spanish support, iPhone issues, Audio overview length, Briefing document generation, NotebookLM Plus daily limits` 


- **NotebookLM still only speaks English**: Users inquired whether **NotebookLM** is available in Spanish, and the response was that it currently supports only **English**.
   - A user responded with a [cat GIF](https://tenor.com/view/tole-cat-cute-gif-12080171459357821404) of a cat being held by a person's hand with its paws crossed.
- **NotebookLM experiences iPhone rendering issues**: A user reported issues using **NotebookLM** on an **iPhone**, with another user confirming it doesn't work on anything using **WebKit** such as **Safari** on **Mac**, which won't be resolved until a fix is implemented.
   - Another user on desktop also had the same issue but also reported a **white screen**.
- **NotebookLM Plus users bumped into daily limits**: A **NotebookLM Plus** subscriber reported seeing a *'You've reached your daily chat limits'* message, preventing proper use, even after logging out and refreshing.
   - Another user clarified that Plus users do not have any limit issues.
- **AI conversational feature gets suggested**: A user suggested an **AI conversation feature** to directly interact with the AI and gather information without extensive reading, gathering a lot of support from others.
   - Members pointed out you could already use interactive mode, however they clarified this suggestion is for more of a *'speaking version of the chat feature'* where users speak to ask the AI questions and listen to receive its responses.
- **Users request timestamps**: Users have requested timestamped sections to allow skipping/re-listening to specific sections similar to how **Audible** does it.
   - Users are also asking for an update to **Gemini 2.5 Pro**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.gpt-reader.com/">GPT Reader: Free AI Text to Speech with ChatGPT Voices, TTS</a>: Natural ChatGPT AI text to speech (TTS) for PDFs, articles, &amp; docs. Download or read aloud using high-quality voices with GPT Reader</li><li><a href="https://myaccount.google.com/age-verification">Account settings: Your browser is not supported.</a>: no description found</li><li><a href="https://tenor.com/view/tole-cat-cute-gif-12080171459357821404">Tole Cat GIF - Tole Cat Cute - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://g.co/gemini/share/97c40282beb9">‎Gemini - NotebookLM Feature and Price Comparison
</a>: Created with Gemini Advanced</li><li><a href="https://support.google.com/notebooklm/answer/15678219">Upgrading to NotebookLM Plus - NotebookLM Help</a>: no description found
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1355278078531604550)** (3 messages): 

> `AI Agent Systems with LlamaIndex, LlamaIndex + Qdrant for Claude, OpenAI Responses API in LlamaIndex` 


- **LlamaIndex and SkySQL team up for AI Agents**: LlamaIndex partners with SkySQL to teach users how to build **AI agent systems** for reliable **text-to-SQL conversion** without coding; more details at the [SkySQL website](https://t.co/Kk7yCCyAuv).
- **LlamaIndex Prepares Documents for Claude via Qdrant**: LlamaIndex shows how to prepare documents for inclusion in **Claude** using a pre-built MCP server for **Qdrant**, using **Angular's documentation** as a data set, stored in [Qdrant](https://t.co/uxTZe1D6gI).
- **LlamaIndex integrates OpenAI Responses API**: LlamaIndex now supports the **OpenAI Responses API** with full support for built-in tools, reasoning, images, manual tool calling, streaming, and async, enabling **complex multi-agent workflows**.
   - The announcement notes that the [Responses API](https://t.co/hJY7EOhn1Z) differs quite a bit from the Chat API.


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1355266580551831836)** (76 messages🔥🔥): 

> `Telemetry Attributes, SubQuestionQueryEngine Workflow, VannaPack Memory Integration, Context Passing to Workflows, Image Input to OpenAIAgent` 


- **Telemetry Attributes Get Tagged**: Members discussed the standard way of passing custom telemetry attributes when interacting with LlamaIndex abstractions, with one member seeking to attach a user ID to all events executed within a code block.
   - A solution was provided that utilizes OpenTelemetry and a [Colab notebook example](https://colab.research.google.com/drive/1QV01kCEncYZ0Ym6o6reHPcffizSVxsQg?usp=sharing) to attach attributes to spans and events, along with a reference to [Arize's documentation](https://docs.arize.com/arize/llm-tracing/how-to-tracing-manual/hybrid-instrumentation#add-attributes-to-multiple-spans-at-once) on hybrid instrumentation.
- **Context Wrangling Woes**: One user ran into an issue of trying to pass the same context to two different workflows, but another member clarified that **a context holds all the data and state for a single workflow** and isn't designed to be shared.
   - Another member inquired about creating a context for `FunctionAgent`, encountering an `AttributeError`, but it was resolved by updating `llama-index-core`.
- **OpenAI Agents go Multi-Modal**: Members discussed passing an image as a chat message to `OpenAIAgent`, with one member noting the lack of direct support for this capability.
   - A member suggested using [OpenAI's multi-modal capabilities](https://docs.llamaindex.ai/en/stable/examples/multi_modal/openai_multi_modal/#ask-the-model-to-describe-what-it-sees) or modifying `chatmemorybuffer` to add images to the request, while another recommended building an agent from scratch with workflows.
- **FunctionAgent Logic Separated for Flexibility**: There was a discussion on *why `FunctionAgent` is not just a workflow*, to which it was clarified that it *needs a specific abstraction to be an agent* with a particular contract.
   - The separation allows for more flexibility and maintainability, with `AgentWorkflow` serving as the orchestrator and `FunctionAgent`/`ReActAgent`/etc. being swappable agent logic, with an example [provided in the documentation](https://docs.llamaindex.ai/en/stable/examples/workflow/function_calling_agent/).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/kashh65/AutoML">AutoML - a Hugging Face Space by kashh65</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1QV01kCEncYZ0Ym6o6reHPcffizSVxsQg?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://docs.arize.com/arize/llm-tracing/how-to-tracing-manual/hybrid-instrumentation#add-attributes-to-multiple-spans-at-once">Add Attributes, Metadata and Tags to Span | Arize Docs</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/multi_modal/openai_multi_modal/#ask-the-model-to-descr">Using OpenAI GPT-4V model for image reasoning - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/workflow/function_calling_agent/">Workflow for a Function Calling Agent - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/multi_modal/openai_multi_modal/#ask-the-model-to-describe-what-it-sees">Using OpenAI GPT-4V model for image reasoning - LlamaIndex</a>: no description found</li><li><a href="https://github.com/run-llama/llama_index/blob/main/llama-index-integrations%2Ftools%2Fllama-index-tools-mcp%2Fexamples%2Fmcp.ipynb">llama_index/llama-index-integrations/tools/llama-index-tools-mcp/examples/mcp.ipynb at main · run-llama/llama_index</a>: LlamaIndex is the leading framework for building LLM-powered agents over your data. - run-llama/llama_index</li><li><a href="https://github.com/run-llama/llama_index/blob/main/llama-index-integrations%2Ftools%2Fllama-index-tools-mcp%2Fllama_index%2Ftools%2Fmcp%2Fbase.py#L57">llama_index/llama-index-integrations/tools/llama-index-tools-mcp/llama_index/tools/mcp/base.py at main · run-llama/llama_index</a>: LlamaIndex is the leading framework for building LLM-powered agents over your data. - run-llama/llama_index</li><li><a href="https://docs.llamaindex.ai/en/stable/examples">Examples - LlamaIndex</a>: no description found
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1356137911166435398)** (4 messages): 

> `LlamaIndex Upgrade, Internet of Agents` 


- **LlamaIndex Upgrade Breeds Embedding Error**: A member reported an error when upgrading **LlamaIndex** from version **0.8.37** to **0.9.0** due to a missing **Embedding** setting.
   - Another member pointed out that the fix might require a version newer than **0.9.0**.
- **Agents dream of interoperation?**: A member published an article outlining a possible direction for solving the interop problem in agentic AI, proposing the construction of an "**Internet of Agents**".
   - The article, available at [[IoA]](https://www.anup.io/p/architecting-the-internet-of-agents), dives into **protocol layers** for communication, memory, trust, and tool use, suggesting that open standards could unlock composability across ecosystems, including **LlamaIndex**.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1355453017230671942)** (35 messages🔥): 

> `Tinygrad Box vs Repurposed E-Waste Inference Machine, Finite Field Assembly Programming Language, TinyGrad Internals Notes, ONNX Float16 Issues, Tenstorrent DevDay` 


- **E-Waste Inference Machine vs Tinygrad Box**: A user questioned the value of an inference machine built from repurposed e-waste with 4x 4090s, linked here: [Tmall](https://detail.tmall.com/item.htm?abbucket=18&id=887290683136), compared to the **Tinygrad Box**.
   - Another user commented that it's likely plagued by **PCIe errors** due to its homebrew motherboard, estimating its worth at $1,000 + the cost of the **4090s**.
- **Finite Field Assembly CUDA Alternative**: A user shared [Finite Field Assembly](https://github.com/LeetArxiv/Finite-Field-Assembly), a **CUDA alternative** designed for computations over finite fields, extending **C89** and supporting recursive computing.
   - It leverages the properties of prime numbers to multiply several array elements concurrently.
- **TinyGrad Internals Detailed in New Notes**: A user shared their notes on **TinyGrad internals** available [here](https://xl0.github.io/tinygrad-notes/), covering **UOps**, **ShapeTracker**, and the **Pattern Matcher**, with inspiration from **mesozoic-egg**.
   - The notes provide a deep dive into the architecture of TinyGrad, complementing the official [TinyGrad documentation](https://docs.tinygrad.org/).
- **ONNX Struggles with Float16 Silently**: A user reported that the **ORT CPUExecutionProvider** silently casts inputs into **float32** for **float16 models**, runs computations with **float32**, and casts the output back into **float16**, which is blocking **numpy removal**.
   - They proposed adding an **envvar** to replicate this behavior in their **ONNX** setup for testing and debugging purposes.
- **Tenstorrent DevDay Presentation**: A user announced they would present **AlphaFold 3** on **Wormhole** at **Tenstorrent DevDay** in SF and expressed interest in meeting other Tinygrad users.
   - They asked about potential sales of excess **Tinygrad V1 motherboards**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://stats.tinygrad.org/">tinygrad stats</a>: no description found</li><li><a href="https://xl0.github.io/tinygrad-notes/">My notes on TyniGrad internals – tinygrad-notes</a>: My notes on TinyGrad internals</li><li><a href="https://github.com/LeetArxiv/Finite-Field-Assembly">GitHub - LeetArxiv/Finite-Field-Assembly: The Finite Field Assembly Programming Language</a>: The Finite Field Assembly Programming Language. Contribute to LeetArxiv/Finite-Field-Assembly development by creating an account on GitHub.</li><li><a href="https://detail.tmall.com/item.htm?abbucket=18&id=887290683136">��Ʒ����</a>: no description found
</li>
</ul>

</div>
  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1356130203809874000)** (17 messages🔥): 

> `VAE with tinygrad, Huggingface's Diffusers library and tinygrad, tg_adapter, torch.to method subclass for Tensors` 


- ****VAE tinygraining!****: A member has been experimenting with building a **VAE** with **tinygrad**.
   - They have successfully modified **Huggingface's Diffusers library** to work with tinygrad and got the **VAE** used in **Stable Diffusion** to function, available at [this link](https://codeberg.org/softcookiepp/tinygrad-stuff/src/branch/master/reimplementation/thf/models/autoencoders/autoencoder_kl.py).
- ****tinygrad Adapting!****: A member created an adapter layer to convert torch calls to tinygrad calls.
   - The adapter layer can be found [here](https://codeberg.org/softcookiepp/tinygrad-stuff/src/branch/master/reimplementation/tg_adapter), enabling the use of **tinygrad** as a drop-in replacement.
- ****Tensor Typecasting Tussle!****: A member mentioned the need to create a subclass for **Tensors** that implements the **torch.to** method.
   - This is needed because, unlike **Tinygrad**, **torch.to** doubles as a typecasting function.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://codeberg.org/softcookiepp/tinygrad-stuff/src/branch/master/reimplementation/tg_adapter">tinygrad-stuff</a>: Porting common neural network architectures, features, etc. to tinygrad</li><li><a href="https://codeberg.org/softcookiepp/tinygrad-stuff/src/branch/master/reimplementation/thf/models/autoencoders/autoencoder_kl.py">tinygrad-stuff/reimplementation/thf/models/autoencoders/autoencoder_kl.py at master</a>: no description found
</li>
</ul>

</div>
  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1355892644177248528)** (12 messages🔥): 

> `FP8 training, Torchtune Office Hours` 


- **FP8 Training Time**: Most **FP8 training recipes** are actually **FP8 QAT**, unless you can only train on GPUs without FP8 support (e.g. A100), in which case you can train with FP8 directly.
   - A member indicated that a **Torchtune office hours** would occur next Friday, with a [Discord link](https://discord.gg/Z9cuQgYX?event=1356379057373184155).
- **Discord Time Zone Conversion**: Members discussed the **automatic conversion of time zones** within Discord for events.
   - One member shared a [brain meme GIF](https://tenor.com/view/brain-brain-meme-big-brain-big-brain-meme-big-brain-time-gif-24411104) in response to successfully converting time zones on the fly.



**Link mentioned**: <a href="https://tenor.com/view/brain-brain-meme-big-brain-big-brain-meme-big-brain-time-gif-2441110471562975014">Brain Brain Meme GIF - Brain Brain meme Big brain - Discover &amp; Share GIFs</a>: Click to view the GIF

  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1355554002095308801)** (4 messages): 

> `Code Review, Merge Process` 


- **Code Review speeds up Merge Process**: A member requested a final review for [PR #2441](https://github.com/pytorch/torchrec/pull/2441) to expedite the merge process.
   - Another member was pinged to review the PR.
- **PR #2441 awaits final review**: Members seek assistance with a final review for [PR #2441](https://github.com/pytorch/torchrec/pull/2441).
   - This aims to accelerate the merging process, as all checks have already passed.


  

---


### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/)** (1 messages): 

yamashi: GRPO to teach searching on the internet: https://arxiv.org/pdf/2503.09516
  

---


### **Cohere ▷ #[「💬」general](https://discord.com/channels/954421988141711382/954421988783444043/1355348512610717746)** (6 messages): 

> `Command-R model, Aya-Vision model, Playground errors` 


- **Command-R is the Speedy Model**: The **Command-R** model is confirmed as the *fastest and most versatile* model, using **Command-A** by default.
   - Users can use the **API** to try out different models, since model changes are not supported in the playground.
- **Aya-Vision struggles with image uploads**: Users are reporting errors when uploading images to the playground using **Aya-Vision**.
   - One user confirmed it's not working and asked to be notified when it starts working better.
- **Job Postings Prohibited**: A moderator issued a warning against posting job postings in the channel.
   - This was a first warning, implying further violations may result in stricter actions.


  

---


### **Cohere ▷ #[「🔌」api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1355965743916908574)** (8 messages🔥): 

> `Cohere Docs Fix, API Latency, Aya Vision` 


- ****Typo Fixed** in Cohere's Documentation!**: A user reported a typo in [Cohere's documentation](https://docs.cohere.com/v2/reference/createfinetunedmodel) where `train_epoch=1` should be `train_epochs=1`, causing a `BadRequestError`.
   - A Cohere staff member confirmed the typo and pushed a fix that *should be live soon*.
- **API **Latency Issues** with Images**: A user reported **inconsistent API performance** with slow responses using the `chatv2` endpoint, specifically when including images, even timing out after increasing timeout limits.
   - They tested the [Aya Vision demo](https://huggingface.co/spaces/CohereForAI/aya_expanse) on Hugging Face, where it sometimes takes over 30 seconds to respond, and non-image based endpoints work quickly.
- ****Debugging Aya Vision** SDK**: A user shared their code snippet for using **Aya Vision** via the Cohere SDK, requesting assistance debugging **latency issues**.
   - A Cohere staff member responded that *they will investigate the latency on their end.*


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/CohereForAI/aya_expanse">Aya Models - a Hugging Face Space by CohereForAI</a>: no description found</li><li><a href="https://docs.cohere.com/v2/reference/createfinetunedmodel">Trains and deploys a fine-tuned model. — Cohere</a>: no description found
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[「🤝」introductions](https://discord.com/channels/954421988141711382/1346635816629178410/1355360080664727673)** (2 messages): 

> `Indy Game Dev, C++, Graphics and Audio Libraries, Browser Game, Cohere` 


- **Indy Game Dev Aims High**: A self-taught indy game developer working mainly in **C++** with graphics and audio libraries introduced themselves.
   - They are currently working on a **browser game** for their friend's **web animation series** and have started using **Cohere** as an alternative to the other big names.
- **New user likes Cohere**: A game developer mentioned that they have started using **Cohere** and like the results so far.
   - They mentioned that they've been using it as an alternative to the *big names*.


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1355521797415632966)** (14 messages🔥): 

> `Libre Wolf, GPT4All model search, Documentation ingestion, Model differences, Llama3 8B Instruct` 


- **Libre Wolf Browser in Question**: A member inquired about **Libre Wolf** browser usage, questioning its security compared to **Firefox**.
- **GPT4All struggles with model search**: A member mentioned difficulty searching the list of **GPT4All models**, as it's not a webpage, to which another member pointed out that a local model list search hasn't been a feature in **GPT4All** for 2 years.
   - One member provided links to the [model lists](https://github.com/nomic-ai/gpt4all/tree/main/gpt4all-chat/metadata) on GitHub.
- **Documentation Ingestion Assistance Requested**: A member asked for a model capable of ingesting documents and answering questions based on them, apologizing for their bad English.
   - A member shared the [GPT4All wiki](https://github.com/nomic-ai/gpt4all/wiki) with official translations in six languages and suggested using Google Translate for other languages.
- **Seeking Blogging Brilliance with Llama3 8B Instruct**: A member inquired if **Llama3 8B Instruct** is the best model for creating blog posts and webpages from recorded video courses.
   - A member asks about the difference between **.bin** and **.gguf** files and whether they can be interchanged.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/nomic-ai/gpt4all/wiki">Home</a>: GPT4All: Run Local LLMs on Any Device. Open-source and available for commercial use. - nomic-ai/gpt4all</li><li><a href="https://github.com/nomic-ai/gpt4all/tree/main/gpt4all-chat/metadata">gpt4all/gpt4all-chat/metadata at main · nomic-ai/gpt4all</a>: GPT4All: Run Local LLMs on Any Device. Open-source and available for commercial use. - nomic-ai/gpt4all
</li>
</ul>

</div>
  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1355532923779944449)** (8 messages🔥): 

> `Pydantic conint, DSPy dynamic example resending, RateLimitError with MIPROv2, Azure OpenAI burst limits, DSPy rate throttling` 


- ****Pydantic's** `conint` **Limits****: The `conint` feature in **Pydantic** can set constraints like `conint(ge=1, le=10)`, but it throws a **ValidationError** if the output falls outside the specified range.
   - A member noted the desire for DSPy to dynamically generate examples and resend requests upon validation failures, a feature that is not currently functioning as expected.
- ****RateLimitErrors** Plague MIPROv2**: A user reported encountering **RateLimitErrors** despite setting `num_threads=1` when using MIPROv2 with `gpt-4o-mini` on Azure OpenAI.
   - Another explained that the issue stems from **MIPROv2.compile()** making multiple internal API calls, compounded by Azure OpenAI's burst limits, which `num_threads=1` does not prevent.
- **Mitigating Azure's **Rate Limits****: To address **RateLimitErrors**, a user suggested adding retry logic with a **sleep(30)** interval, lowering `max_*_demos`, and potentially upgrading to the latest DSPy version with built-in rate throttling.
   - It was emphasized that structured prompting in MIPROv2 and Copro can lead to errors if the LLM returns empty outputs due to API truncation or rate limits.
- **Optimization Suffers from Rate Limit Workarounds**: A user pointed out that reducing `max_bootstrapped_demos` and `max_labeled_demos` to avoid **RateLimitErrors** negatively impacts the optimization process.
   - They suggested that DSPy lacks an internal delay mechanism to manage API call frequency effectively.


  

---


### **DSPy ▷ #[examples](https://discord.com/channels/1161519468141355160/1161519685616025600/1355296267210592406)** (4 messages): 

> `DSPy Optimizers, Module Usage, Prompt Engineering in DSPy, Signature Creation in DSPy` 


- **DSPy Optimizers Optimize Prompts and Weights**: DSPy optimizes prompts and weights to teach LMs to deliver high-quality outputs, offering algorithms for **building modular AI systems** and **optimizing their prompts and weights** according to the [DSPy documentation](https://dspy.ai/).
   - Different optimizers choose N examples to include in the prompt.
- **DSPy signatures as "a, b -> c"**: In DSPy, the signature is defined as *"a, b -> c"*, where a, b, and c are meaningful names.
   - The optimizer then generates prompts and runs them on a dataset to determine the best performing prompt.
- **Practical Module Usage Considerations**: If the specific implementation necessitates an optimizer, the **relevance of docstrings diminishes**.
- **Building NLP Data to Chart Pipelines**: A member is working on leveraging DSPy to build a tool that transforms **natural language processing of data into charts**.



**Link mentioned**: <a href="https://dspy.ai/">DSPy</a>: The framework for programming—rather than prompting—language models.

  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-announcements](https://discord.com/channels/1280234300012494859/1280369709623283732/1356133200380035102)** (1 messages): 

> `Thomas Hubert, AlphaProof, Formal Mathematics, Reinforcement Learning` 


- **Thomas Hubert Presents AlphaProof Lecture**: Thomas Hubert, a research engineer at Google DeepMind, will present "**AlphaProof**: when reinforcement learning meets formal mathematics" on 3/31 at 10AM PDT, livestreamed on [YouTube](https://www.youtube.com/live/3gaEMscOMAU).
   - The lecture will explore how computers and computation are now routinely used in research mathematics and contribute to grand problems like the **Birch and Swinnerton-Dyer conjecture**.
- **Galileo's View on Mathematics**: **Galileo**, the renowned Italian astronomer, physicist, and mathematician, famously described mathematics as *the language of the universe* and computers have enriched our understanding.
   - Hubert earned his MS in Mathematics from Stanford University.



**Link mentioned**: <a href="https://www.youtube.com/live/3gaEMscOMAU"> - YouTube</a>: no description found

  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1355402573514801152)** (11 messages🔥): 

> `Course Information, Lecture Times, Free Credits, AgentX Competition` 


- ****LLM Agents MOOC** Course Info Listed**: The course website ([llmagents-learning.org/sp25](https://llmagents-learning.org/sp25)) and [Discord server](https://discord.gg/NWVpQ9rBvd) provide essential links and discussion forums for the **LLM Agents MOOC**.
- **Spring 2025 **LLM Agents MOOC** Previous Lectures**: Previous lectures for the Spring 2025 course can be found on the [course website](https://llmagents-learning.org/sp25) and in this [YouTube playlist](https://www.youtube.com/playlist?list=PLS01nW3RtgorL3AW8REU9nGkzhvtn6Egn).
- **How to get Free Credits?**: AgentX offers credit resources and details can be found on the [AgentX website](https://rdi.berkeley.edu/agentx/)), with a collection form releasing this week for those wanting credits for AgentX.
- **Lecture bumped up to 10 AM PST**: The lecture today was moved to **10 AM PST** to accommodate the speaker from the **UK**.
- **Completion based quizzes**: The quizzes for the course are **completion based** and the score does not matter as long as they are attempted.



**Link mentioned**: <a href="https://llmagents-learning.org/sp25">Advanced Large Language Model Agents MOOC</a>: MOOC, Spring 2025

  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1355933844368392445)** (1 messages): 

> `TMLS 2025, MLOps, AI, Call for Speakers, AI Agents` 


- **TMLS 2025 Call for Speakers Opens**: A member announced the [Call for Speakers](https://www.linkedin.com/posts/toronto-machine-learning-summit_tmls2025-callforspeakers-ai-activity-7303505411800719361-z-V2?utm_source=share&utm_medium=member_ios&rcm=ACoAACF-hfwBzcfh2mYq928aQ3C0PDfox4I_I8s) for the **Toronto Machine Learning Summit (TMLS)** in June 2025.
   - TMLS 2025 will feature **16 specialized tracks**, including **Advanced RAG**, **Multimodal LLMs**, **AI Agents in Production**, **MLOps for Smaller Teams**, **Responsible AI Implementation**, and **GenAI Deployments**.
- **MLOps for Smaller Teams**: The Toronto Machine Learning Summit will have an MLOps track aimed at smaller teams.
   - This is a great opportunity for smaller teams to share their experiences and learn from others.


  

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
